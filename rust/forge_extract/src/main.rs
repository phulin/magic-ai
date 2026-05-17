//! Forge choice-situation extractor (Rust port of the parser portion of
//! `scripts/extract_forge_choice_situations.py`).
//!
//! Reads a Forge games zip (members are `*.jsonl.gz`), extracts choice
//! candidates (priority / attack / block / may / choose), normalizes the
//! pre-choice snapshot, and writes one JSONL record per selected situation.
//! A separate Python pass is responsible for tokenization + torch sharding.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use anyhow::{anyhow, Context, Result};
use arrow_array::builder::{Int32Builder, ListBuilder, UInt8Builder};
use arrow_array::{
    ArrayRef, BooleanArray, Float32Array, Int32Array, Int64Array, LargeStringArray, RecordBatch,
    StringArray, UInt16Array, UInt32Array, UInt8Array,
};
use arrow_ipc::writer::{FileWriter, IpcWriteOptions};
use arrow_ipc::{CompressionType, MetadataVersion};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use clap::{Parser, ValueEnum};
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use once_cell::sync::Lazy;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sonic_rs::{json, JsonContainerTrait, JsonValueMutTrait, JsonValueTrait, Value};
use zip::ZipArchive;

const FORMAT_VERSION: u32 = 2;
const CHOICE_KINDS: &[&str] = &["priority", "attack", "block", "may", "choose"];
const DEFAULT_KIND_PRIORITY: &[&str] = &["may", "block", "attack", "choose", "priority"];

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    JsonlGz,
    Arrow,
}

#[derive(Parser, Debug)]
#[command(about = "Extract Forge choice situations to JSONL.GZ or Arrow IPC")]
struct Args {
    /// Forge game archive (zip of *.jsonl.gz members).
    #[arg(long)]
    zip: Option<PathBuf>,

    /// Output .jsonl.gz path, or output directory when --output-format arrow.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Output serialization format.
    #[arg(long, value_enum, default_value_t = OutputFormat::JsonlGz)]
    output_format: OutputFormat,

    /// Comma-separated kinds (or "all").
    #[arg(long, default_value = "all")]
    kinds: String,

    /// "hash" or "priority".
    #[arg(long, default_value = "hash")]
    selection: String,

    /// Maximum selected situations per game.
    #[arg(long, default_value_t = 2)]
    choices_per_game: usize,

    /// Emit every extractable decision per game in trajectory order. Also
    /// infers priority passes, but only from priority windows where the
    /// player had at least one non-pass playable action.
    #[arg(long, default_value_t = false)]
    trajectory: bool,

    /// Kind ordering when --selection priority.
    #[arg(long, default_value = "may,block,attack,choose,priority")]
    kind_priority: String,

    /// Optional cap on games processed.
    #[arg(long)]
    limit_games: Option<usize>,

    /// Number of parser threads (defaults to rayon's choice).
    #[arg(long)]
    threads: Option<usize>,

    /// Update progress details every N zip members. Set to 0 to disable the progress bar.
    #[arg(long, default_value_t = 1000)]
    progress_every: usize,

    /// gzip compression level for the output (0-9).
    #[arg(long, default_value_t = 6)]
    compresslevel: u32,

    /// Target rows per Arrow shard. Shards flush only after a game batch, so
    /// they may exceed this by one game.
    #[arg(long, default_value_t = 262_144)]
    arrow_shard_rows: usize,

    /// Optional Python-exported token-table artifact. When set with Arrow
    /// output, the extractor emits pre-tokenized state rows.
    #[arg(long)]
    token_tables: Option<PathBuf>,

    /// Debug-only: emit token ids for one already-normalized snapshot JSON.
    #[arg(long, hide = true)]
    emit_state_json: Option<PathBuf>,
}

fn make_progress_bar(total: usize, enabled: bool) -> ProgressBar {
    if !enabled {
        return ProgressBar::hidden();
    }

    let progress_bar = ProgressBar::new(total as u64);
    progress_bar.set_draw_target(ProgressDrawTarget::stderr_with_hz(2));
    progress_bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] \
             {pos}/{len} members ({per_sec}, ETA {eta_precise}) {msg}",
        )
        .expect("valid progress template")
        .progress_chars("#>-"),
    );
    progress_bar
}

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    let enabled_kinds = parse_kinds(&args.kinds)?;
    let kind_priority = parse_kind_priority(&args.kind_priority)?;
    let selection = match args.selection.as_str() {
        "hash" => Selection::Hash,
        "priority" => Selection::Priority,
        other => return Err(anyhow!("unknown --selection {other:?}")),
    };
    if args.token_tables.is_some()
        && !matches!(args.output_format, OutputFormat::Arrow)
        && args.emit_state_json.is_none()
    {
        return Err(anyhow!(
            "--token-tables is only supported with --output-format arrow"
        ));
    }
    let token_tables = args
        .token_tables
        .as_ref()
        .map(|path| load_token_table_bundle(path))
        .transpose()?
        .map(Arc::new);

    if let Some(snapshot_path) = &args.emit_state_json {
        let bundle = token_tables
            .as_deref()
            .ok_or_else(|| anyhow!("--emit-state-json requires --token-tables"))?;
        let f =
            File::open(snapshot_path).with_context(|| format!("opening {:?}", snapshot_path))?;
        let snapshot: Value = sonic_rs::from_reader(BufReader::new(f))
            .with_context(|| format!("parsing {:?}", snapshot_path))?;
        let state = emit_tokenized_state(&snapshot, bundle)?;
        let spec = emit_decision_spec(&snapshot, bundle);
        let payload = json!({
            "token_ids": state.token_ids,
            "seq_length": state.token_ids.len(),
            "card_ref_positions": state.card_ref_positions,
            "card_ref_overflow": state.card_ref_overflow,
            "action_token_ids": spec.as_ref().map(|s| s.token_ids.as_slice()).unwrap_or(&[]),
            "action_seq_length": spec.as_ref().map(|s| s.token_ids.len()).unwrap_or(0),
            "decision_type": spec.as_ref().map(|s| s.decision_type).unwrap_or(DECISION_NONE),
            "pointer_anchor_positions": spec.as_ref().map(|s| s.anchor_positions.as_slice()).unwrap_or(&[]),
            "pointer_anchor_kinds": spec.as_ref().map(|s| s.anchor_kinds.as_slice()).unwrap_or(&[]),
            "pointer_anchor_subjects": spec.as_ref().map(|s| s.anchor_subjects.as_slice()).unwrap_or(&[]),
            "pointer_anchor_handles": spec.as_ref().map(|s| s.anchor_handles.as_slice()).unwrap_or(&[]),
            "legal_edge_bitmap": spec.as_ref().map(|s| s.legal_edge_bitmap.as_slice()).unwrap_or(&[]),
            "legal_edge_n_blockers": spec.as_ref().map(|s| s.legal_edge_n_blockers).unwrap_or(0),
            "legal_edge_n_attackers": spec.as_ref().map(|s| s.legal_edge_n_attackers).unwrap_or(0),
        });
        println!("{}", sonic_rs::to_string(&payload)?);
        return Ok(());
    }

    let zip_path = args
        .zip
        .as_ref()
        .ok_or_else(|| anyhow!("--zip is required unless --emit-state-json is set"))?;
    let out_path = args
        .out
        .as_ref()
        .ok_or_else(|| anyhow!("--out is required unless --emit-state-json is set"))?
        .clone();
    let pretokenized = token_tables.is_some();
    if matches!(args.output_format, OutputFormat::Arrow) {
        std::fs::create_dir_all(&out_path).with_context(|| format!("creating {:?}", out_path))?;
        ensure_arrow_output_dir_is_clean(&out_path)?;
    }

    // Enumerate members up-front (cheap; zip central directory only).
    let members: Vec<String> = {
        let f = File::open(zip_path).with_context(|| format!("opening {:?}", zip_path))?;
        let mut zf = ZipArchive::new(BufReader::new(f))?;
        (0..zf.len())
            .filter_map(|i| zf.by_index(i).ok().map(|e| e.name().to_string()))
            .filter(|n| n.ends_with(".jsonl.gz"))
            .collect()
    };
    let mut members = members;
    members.sort();
    if let Some(limit) = args.limit_games {
        members.truncate(limit);
    }
    eprintln!("members: {}", members.len());

    // Writer thread: serialize output without blocking parser workers on disk I/O.
    let (tx, rx) = mpsc::sync_channel::<WriterBatch>(64);
    let level = args.compresslevel;
    let output_format = args.output_format;
    let arrow_shard_rows = args.arrow_shard_rows;
    let manifest_out_path = out_path.clone();
    let writer_handle = thread::spawn(move || -> Result<WriterStats> {
        match output_format {
            OutputFormat::JsonlGz => write_jsonl_gz(rx, out_path, level),
            OutputFormat::Arrow => write_arrow_dir(rx, out_path, arrow_shard_rows, pretokenized),
        }
    });

    let progress_every = args.progress_every;
    let total_members = members.len();
    let progress_bar = make_progress_bar(total_members, progress_every > 0);
    let members_done = std::sync::atomic::AtomicUsize::new(0);
    let zip_path = zip_path.clone();
    let games_seen = std::sync::atomic::AtomicUsize::new(0);
    let games_written = std::sync::atomic::AtomicUsize::new(0);
    let candidates_seen = std::sync::atomic::AtomicUsize::new(0);
    let kind_counts: [std::sync::atomic::AtomicUsize; 5] = Default::default();

    let parse_result = members.par_iter().try_for_each_init(
        || -> Result<ZipArchive<BufReader<File>>> {
            let f = File::open(&zip_path)?;
            Ok(ZipArchive::new(BufReader::new(f))?)
        },
        |zf_res, member| -> Result<()> {
            let result = (|| -> Result<()> {
                let zf = match zf_res {
                    Ok(z) => z,
                    Err(e) => return Err(anyhow!("zip init: {e}")),
                };
                let entry = zf.by_name(member)?;
                let rows = read_jsonl_member(entry)?;
                let meta = match meta_from_rows(&rows) {
                    Some(m) if !m.game_id.is_empty() => m,
                    _ => return Ok(()),
                };
                games_seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                let candidates = extract_candidates(&rows, &enabled_kinds, args.trajectory);
                candidates_seen.fetch_add(candidates.len(), std::sync::atomic::Ordering::Relaxed);
                if candidates.is_empty() {
                    return Ok(());
                }

                let selected: Vec<Candidate> = if args.trajectory {
                    candidates
                } else {
                    match selection {
                        Selection::Hash => {
                            stable_choices(candidates, args.choices_per_game, meta.game_id)
                        }
                        Selection::Priority => {
                            priority_choices(candidates, &kind_priority, args.choices_per_game)
                        }
                    }
                };
                if selected.is_empty() {
                    return Ok(());
                }

                let mut json_batch: Vec<Vec<u8>> = Vec::new();
                let mut arrow_batch: Vec<ArrowDecisionRow> = Vec::new();
                match output_format {
                    OutputFormat::JsonlGz => json_batch.reserve(selected.len()),
                    OutputFormat::Arrow => arrow_batch.reserve(selected.len()),
                }
                for c in selected {
                    let kind_idx = choice_kind_index(c.kind).unwrap();
                    kind_counts[kind_idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let rec = build_record(&c, &meta, member);
                    match output_format {
                        OutputFormat::JsonlGz => json_batch.push(sonic_rs::to_vec(&rec)?),
                        OutputFormat::Arrow => {
                            let mut row = ArrowDecisionRow::from_record(&rec)?;
                            if let Some(bundle) = token_tables.as_deref() {
                                let state = emit_tokenized_state(rec.state.snapshot, bundle)?;
                                let spec = emit_decision_spec(rec.state.snapshot, bundle);
                                row.attach_tokenized_state(state, spec);
                            }
                            arrow_batch.push(row)
                        }
                    }
                }
                games_written.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let batch = match output_format {
                    OutputFormat::JsonlGz => WriterBatch::JsonLines(json_batch),
                    OutputFormat::Arrow => WriterBatch::ArrowRows(arrow_batch),
                };
                tx.send(batch).map_err(|e| anyhow!("writer dropped: {e}"))?;
                Ok(())
            })();

            if progress_every > 0 {
                let done = members_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                progress_bar.inc(1);
                if done % progress_every == 0 || done == total_members {
                    progress_bar.set_message(format!(
                        "seen={} written={} candidates={}",
                        games_seen.load(std::sync::atomic::Ordering::Relaxed),
                        games_written.load(std::sync::atomic::Ordering::Relaxed),
                        candidates_seen.load(std::sync::atomic::Ordering::Relaxed),
                    ));
                }
            }
            result
        },
    );
    progress_bar.finish_and_clear();
    parse_result?;
    drop(tx);

    let writer_stats = writer_handle
        .join()
        .map_err(|_| anyhow!("writer thread panicked"))??;

    let kind_summary = json!({
        "written_priority": kind_counts[0].load(std::sync::atomic::Ordering::Relaxed),
        "written_attack": kind_counts[1].load(std::sync::atomic::Ordering::Relaxed),
        "written_block": kind_counts[2].load(std::sync::atomic::Ordering::Relaxed),
        "written_may": kind_counts[3].load(std::sync::atomic::Ordering::Relaxed),
        "written_choose": kind_counts[4].load(std::sync::atomic::Ordering::Relaxed),
    });
    let summary = json!({
        "output_format": match args.output_format {
            OutputFormat::JsonlGz => "jsonl.gz",
            OutputFormat::Arrow => "arrow",
        },
        "games_seen": games_seen.load(std::sync::atomic::Ordering::Relaxed),
        "games_written": games_written.load(std::sync::atomic::Ordering::Relaxed),
        "candidates_seen": candidates_seen.load(std::sync::atomic::Ordering::Relaxed),
        "records_written": writer_stats.records,
        "shards_written": writer_stats.shards,
        "kinds": kind_summary,
    });
    if matches!(args.output_format, OutputFormat::Arrow) {
        write_arrow_manifest(
            &manifest_out_path,
            &ArrowManifestStats {
                shards: writer_stats.shards,
                records_written: writer_stats.records,
                shard_target_rows: args.arrow_shard_rows,
                games_seen: games_seen.load(std::sync::atomic::Ordering::Relaxed) as u64,
                games_written: games_written.load(std::sync::atomic::Ordering::Relaxed) as u64,
                candidates_seen: candidates_seen.load(std::sync::atomic::Ordering::Relaxed) as u64,
                kind_counts: [
                    kind_counts[0].load(std::sync::atomic::Ordering::Relaxed) as u64,
                    kind_counts[1].load(std::sync::atomic::Ordering::Relaxed) as u64,
                    kind_counts[2].load(std::sync::atomic::Ordering::Relaxed) as u64,
                    kind_counts[3].load(std::sync::atomic::Ordering::Relaxed) as u64,
                    kind_counts[4].load(std::sync::atomic::Ordering::Relaxed) as u64,
                ],
                pretokenized,
            },
        )?;
    }
    println!("{}", sonic_rs::to_string_pretty(&summary)?);
    Ok(())
}

#[derive(Clone, Copy)]
enum Selection {
    Hash,
    Priority,
}

struct WriterStats {
    records: u64,
    shards: u64,
}

enum WriterBatch {
    JsonLines(Vec<Vec<u8>>),
    ArrowRows(Vec<ArrowDecisionRow>),
}

struct ArrowDecisionRow {
    format_version: u16,
    game_id: String,
    archive_member: String,
    kind_id: u8,
    candidate_index: u32,
    candidate_count: u32,
    source_seq: i64,
    target_seq: i64,
    perspective_id: String,
    perspective_name: String,
    winner_id: Option<String>,
    winner_name: Option<String>,
    terminal_sign: f32,
    snapshot_json: String,
    observed_json: String,
    outcome_players_json: String,
    outcome_extras_json: String,
    token_ids: Option<Vec<i32>>,
    seq_length: Option<u32>,
    card_ref_positions: Option<Vec<i32>>,
    token_overflow: bool,
    card_ref_overflow: bool,
    action_token_ids: Option<Vec<i32>>,
    action_seq_length: Option<u32>,
    decision_type: Option<i32>,
    pointer_anchor_positions: Option<Vec<i32>>,
    pointer_anchor_kinds: Option<Vec<i32>>,
    pointer_anchor_subjects: Option<Vec<i32>>,
    pointer_anchor_handles: Option<Vec<i32>>,
    legal_edge_bitmap: Option<Vec<u8>>,
    legal_edge_n_blockers: Option<u32>,
    legal_edge_n_attackers: Option<u32>,
}

impl ArrowDecisionRow {
    fn from_record(record: &OutputRecord<'_>) -> Result<Self> {
        Ok(Self {
            format_version: u16::try_from(record.format_version)
                .context("format_version does not fit uint16")?,
            game_id: record.game_id.to_string(),
            archive_member: record.archive_member.to_string(),
            kind_id: u8::try_from(choice_kind_index(record.choice.kind).unwrap())
                .context("kind id does not fit uint8")?,
            candidate_index: u32::try_from(record.choice.candidate_index)
                .context("candidate_index does not fit uint32")?,
            candidate_count: u32::try_from(record.choice.candidate_count)
                .context("candidate_count does not fit uint32")?,
            source_seq: record.choice.source_seq,
            target_seq: record.choice.target_seq,
            perspective_id: record.choice.perspective_id.to_string(),
            perspective_name: record.choice.perspective_name.to_string(),
            winner_id: record.outcome.winner_id.map(str::to_string),
            winner_name: record.outcome.winner_name.map(str::to_string),
            terminal_sign: record.outcome.terminal_sign as f32,
            snapshot_json: sonic_rs::to_string(record.state.snapshot)
                .context("serializing snapshot_json")?,
            observed_json: sonic_rs::to_string(record.choice.observed)
                .context("serializing observed_json")?,
            outcome_players_json: sonic_rs::to_string(record.outcome.players)
                .context("serializing outcome_players_json")?,
            outcome_extras_json: sonic_rs::to_string(record.outcome.extras)
                .context("serializing outcome_extras_json")?,
            token_ids: None,
            seq_length: None,
            card_ref_positions: None,
            token_overflow: false,
            card_ref_overflow: false,
            action_token_ids: None,
            action_seq_length: None,
            decision_type: None,
            pointer_anchor_positions: None,
            pointer_anchor_kinds: None,
            pointer_anchor_subjects: None,
            pointer_anchor_handles: None,
            legal_edge_bitmap: None,
            legal_edge_n_blockers: None,
            legal_edge_n_attackers: None,
        })
    }

    fn attach_tokenized_state(&mut self, state: TokenizedState, spec: Option<DecisionSpecState>) {
        self.seq_length = Some(state.token_ids.len() as u32);
        self.token_ids = Some(state.token_ids);
        self.card_ref_positions = Some(state.card_ref_positions);
        self.token_overflow = false;
        self.card_ref_overflow = state.card_ref_overflow;
        if let Some(spec) = spec {
            self.action_seq_length = Some(spec.token_ids.len() as u32);
            self.action_token_ids = Some(spec.token_ids);
            self.decision_type = Some(spec.decision_type);
            self.pointer_anchor_positions = Some(spec.anchor_positions);
            self.pointer_anchor_kinds = Some(spec.anchor_kinds);
            self.pointer_anchor_subjects = Some(spec.anchor_subjects);
            self.pointer_anchor_handles = Some(spec.anchor_handles);
            self.legal_edge_bitmap = Some(spec.legal_edge_bitmap);
            self.legal_edge_n_blockers = Some(spec.legal_edge_n_blockers as u32);
            self.legal_edge_n_attackers = Some(spec.legal_edge_n_attackers as u32);
        } else {
            self.action_seq_length = Some(0);
            self.action_token_ids = Some(Vec::new());
            self.decision_type = Some(DECISION_NONE);
            self.pointer_anchor_positions = Some(Vec::new());
            self.pointer_anchor_kinds = Some(Vec::new());
            self.pointer_anchor_subjects = Some(Vec::new());
            self.pointer_anchor_handles = Some(Vec::new());
            self.legal_edge_bitmap = Some(Vec::new());
            self.legal_edge_n_blockers = Some(0);
            self.legal_edge_n_attackers = Some(0);
        }
    }
}

#[derive(Clone, Deserialize)]
struct SequenceTable {
    offsets: Vec<usize>,
    tokens: Vec<i32>,
}

impl SequenceTable {
    fn get(&self, index: usize) -> &[i32] {
        if index + 1 >= self.offsets.len() {
            return &[];
        }
        let start = self.offsets[index];
        let end = self.offsets[index + 1];
        &self.tokens[start..end]
    }
}

#[derive(Clone, Deserialize)]
struct TokenTableArtifact {
    schema_version: u32,
    tokenizer: TokenizerArtifactMeta,
    tables: RustTokenTables,
    decision_spec: DecisionSpecTables,
}

#[derive(Clone, Deserialize)]
struct TokenizerArtifactMeta {
    unk_token_id: Option<i32>,
}

#[derive(Clone, Deserialize)]
struct RustTokenTables {
    dict_open_id: i32,
    dict_close_id: i32,
    card_open_id: i32,
    stack_open_id: i32,
    stack_close_id: i32,
    turn_min: i32,
    turn_max: i32,
    life_min: i32,
    life_max: i32,
    count_min: i32,
    count_max: i32,
    step_count: usize,
    owner_count: usize,
    structural: SequenceTable,
    zone_open: SequenceTable,
    zone_close: SequenceTable,
    mana_glyph: SequenceTable,
    turn_step: SequenceTable,
    life_owner: SequenceTable,
    count: SequenceTable,
    card_closer: Vec<i32>,
    status_tapped: Vec<i32>,
    status_untapped: Vec<i32>,
    card_ref: Vec<i32>,
    dict_entry: Vec<i32>,
    card_body: SequenceTable,
    row_to_name: Vec<String>,
}

#[derive(Clone, Deserialize)]
struct DecisionSpecTables {
    spec_open_id: i32,
    spec_close_id: i32,
    decision_type_id: i32,
    legal_attacker_id: i32,
    legal_blocker_id: i32,
    legal_target_id: i32,
    legal_action_id: i32,
    max_value_open_id: i32,
    max_value_close_id: i32,
    player_ref_ids: Vec<i32>,
    decision_type_name_ids: Vec<i32>,
    max_value_digits: SequenceTable,
}

struct TokenTableBundle {
    tables: RustTokenTables,
    decision_spec: DecisionSpecTables,
    name_to_row: HashMap<String, usize>,
    unk_token_id: Option<i32>,
}

fn load_token_table_bundle(path: &PathBuf) -> Result<TokenTableBundle> {
    let bytes = std::fs::read(path).with_context(|| format!("reading token tables {:?}", path))?;
    let artifact: TokenTableArtifact =
        sonic_rs::from_slice(&bytes).with_context(|| format!("parsing {:?}", path))?;
    if artifact.schema_version != 1 {
        return Err(anyhow!(
            "unsupported token-table artifact schema_version={}",
            artifact.schema_version
        ));
    }
    let mut name_to_row = HashMap::with_capacity(artifact.tables.row_to_name.len());
    for (row, name) in artifact.tables.row_to_name.iter().enumerate() {
        if row > 0 {
            name_to_row.insert(name.clone(), row);
        }
    }
    Ok(TokenTableBundle {
        unk_token_id: artifact.tokenizer.unk_token_id,
        tables: artifact.tables,
        decision_spec: artifact.decision_spec,
        name_to_row,
    })
}

struct TokenizedState {
    token_ids: Vec<i32>,
    card_ref_positions: Vec<i32>,
    card_ref_overflow: bool,
}

struct DecisionSpecState {
    token_ids: Vec<i32>,
    decision_type: i32,
    anchor_positions: Vec<i32>,
    anchor_kinds: Vec<i32>,
    anchor_subjects: Vec<i32>,
    anchor_handles: Vec<i32>,
    legal_edge_bitmap: Vec<u8>,
    legal_edge_n_blockers: usize,
    legal_edge_n_attackers: usize,
}

const DECISION_PRIORITY: i32 = 0;
const DECISION_DECLARE_ATTACKERS: i32 = 1;
const DECISION_DECLARE_BLOCKERS: i32 = 2;
const DECISION_CHOOSE_TARGETS: i32 = 3;
const DECISION_MAY: i32 = 4;
const DECISION_CHOOSE_MODE: i32 = 5;
const DECISION_CHOOSE_X: i32 = 6;
const DECISION_NONE: i32 = -1;

const ANCHOR_LEGAL_ATTACKER: i32 = 0;
const ANCHOR_LEGAL_BLOCKER: i32 = 1;
const ANCHOR_LEGAL_TARGET: i32 = 2;
const ANCHOR_LEGAL_ACTION: i32 = 3;
const ANCHOR_DEFENDER: i32 = 4;

const MAX_CARD_REFS: usize = 256;
const FRAG_BOS_STATE: usize = 0;
const FRAG_CLOSE_STATE_EOS: usize = 1;
const FRAG_CLOSE_SELF: usize = 2;
const FRAG_CLOSE_OPP: usize = 3;
const FRAG_CLOSE_OPTION: usize = 4;
const FRAG_SELF_MANA: usize = 10;
const FRAG_OPP_MANA: usize = 11;

const ZONE_HAND: usize = 0;
const ZONE_BATTLEFIELD: usize = 1;
const ZONE_GRAVEYARD: usize = 2;
const ZONE_EXILE: usize = 3;
const ZONE_LIBRARY: usize = 4;

const OWNER_SELF: usize = 0;
const OWNER_OPP: usize = 1;

struct PlacedCard {
    row: usize,
    uuid_idx: i32,
    tapped_known: bool,
    tapped: bool,
}

struct PlacementIndex {
    cards_by_zone: HashMap<(usize, usize), Vec<PlacedCard>>,
    row_order: Vec<usize>,
    row_to_dict_slot: HashMap<usize, usize>,
    overflow: bool,
}

struct DirectEmitter<'a> {
    bundle: &'a TokenTableBundle,
    out: Vec<i32>,
    card_ref_positions: Vec<i32>,
    scalar_owner_open: i8,
    option_open: bool,
}

impl<'a> DirectEmitter<'a> {
    fn new(bundle: &'a TokenTableBundle) -> Self {
        Self {
            bundle,
            out: Vec::with_capacity(2048),
            card_ref_positions: vec![-1; MAX_CARD_REFS],
            scalar_owner_open: -1,
            option_open: false,
        }
    }

    fn tables(&self) -> &RustTokenTables {
        &self.bundle.tables
    }

    fn write_one(&mut self, token_id: i32) {
        self.out.push(token_id);
    }

    fn emit_frag(&mut self, frag: usize) {
        let span = self.bundle.tables.structural.get(frag);
        self.out.extend_from_slice(span);
    }

    fn close_scalar_owner(&mut self) {
        if self.scalar_owner_open == OWNER_SELF as i8 {
            self.emit_frag(FRAG_CLOSE_SELF);
        } else if self.scalar_owner_open == OWNER_OPP as i8 {
            self.emit_frag(FRAG_CLOSE_OPP);
        }
        self.scalar_owner_open = -1;
    }

    fn close_option(&mut self) {
        if self.option_open {
            self.emit_frag(FRAG_CLOSE_OPTION);
            self.option_open = false;
        }
    }

    fn emit_card_ref(&mut self, k: i32) {
        if k < 0 {
            return;
        }
        let idx = k as usize;
        if idx >= self.tables().card_ref.len() || idx >= MAX_CARD_REFS {
            return;
        }
        if self.card_ref_positions[idx] < 0 {
            self.card_ref_positions[idx] = self.out.len() as i32;
        }
        self.write_one(self.tables().card_ref[idx]);
    }

    fn emit(
        mut self,
        snapshot: &Value,
        placement: &PlacementIndex,
        self_player: Option<&Value>,
        opp_player: Option<&Value>,
    ) -> TokenizedState {
        self.emit_frag(FRAG_BOS_STATE);

        if !placement.row_order.is_empty() {
            if !placement.row_to_dict_slot.is_empty() {
                self.write_one(self.tables().dict_open_id);
                for row in &placement.row_order {
                    let Some(slot) = placement.row_to_dict_slot.get(row).copied() else {
                        continue;
                    };
                    self.write_one(self.tables().dict_entry[slot]);
                    self.out
                        .extend_from_slice(self.bundle.tables.card_body.get(*row));
                    self.out
                        .extend_from_slice(self.bundle.tables.card_closer.as_slice());
                }
                self.write_one(self.tables().dict_close_id);
            }
        }

        self.emit_turn_step(snapshot_turn(snapshot), step_id(snapshot_step(snapshot)));

        for (owner, player) in [(OWNER_SELF, self_player), (OWNER_OPP, opp_player)] {
            self.emit_life(owner, player_life(player));
            for (color, amount) in player_mana_pool(player).iter().enumerate() {
                if *amount > 0 {
                    self.emit_mana(owner, color, *amount);
                }
            }
        }
        self.close_scalar_owner();

        for zone in [ZONE_BATTLEFIELD, ZONE_HAND, ZONE_GRAVEYARD] {
            for owner in [OWNER_SELF, OWNER_OPP] {
                if zone == ZONE_HAND && owner == OWNER_OPP {
                    continue;
                }
                self.emit_zone(
                    zone,
                    owner,
                    placement
                        .cards_by_zone
                        .get(&(zone, owner))
                        .map(Vec::as_slice)
                        .unwrap_or(&[]),
                    placement,
                );
            }
        }
        for owner in [OWNER_SELF, OWNER_OPP] {
            let cards = placement
                .cards_by_zone
                .get(&(ZONE_EXILE, owner))
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if !cards.is_empty() {
                self.emit_zone(ZONE_EXILE, owner, cards, placement);
            }
        }
        for (owner, player) in [(OWNER_SELF, self_player), (OWNER_OPP, opp_player)] {
            self.emit_open_zone(ZONE_LIBRARY, owner);
            self.emit_count(player_library_count(player));
            self.emit_close_zone(ZONE_LIBRARY, owner);
        }

        self.close_scalar_owner();
        self.write_one(self.tables().stack_open_id);
        self.write_one(self.tables().stack_close_id);

        self.close_scalar_owner();
        self.close_option();
        self.emit_frag(FRAG_CLOSE_STATE_EOS);

        TokenizedState {
            token_ids: self.out,
            card_ref_positions: self.card_ref_positions,
            card_ref_overflow: placement.overflow,
        }
    }

    fn emit_turn_step(&mut self, turn: i32, step_id: usize) {
        let t = turn.clamp(self.tables().turn_min, self.tables().turn_max);
        let idx = ((t - self.tables().turn_min) as usize) * self.tables().step_count + step_id;
        self.out
            .extend_from_slice(self.bundle.tables.turn_step.get(idx));
    }

    fn emit_life(&mut self, owner: usize, life: i32) {
        self.close_scalar_owner();
        let clamped = life.clamp(self.tables().life_min, self.tables().life_max);
        let idx = ((clamped - self.tables().life_min) as usize) * self.tables().owner_count + owner;
        self.out
            .extend_from_slice(self.bundle.tables.life_owner.get(idx));
        self.scalar_owner_open = owner as i8;
    }

    fn emit_mana(&mut self, owner: usize, color: usize, amount: i32) {
        if self.scalar_owner_open >= 0 && self.scalar_owner_open != owner as i8 {
            self.close_scalar_owner();
        }
        if self.scalar_owner_open < 0 {
            self.emit_frag(if owner == OWNER_SELF {
                FRAG_SELF_MANA
            } else {
                FRAG_OPP_MANA
            });
            self.scalar_owner_open = owner as i8;
        }
        for _ in 0..amount {
            self.out
                .extend_from_slice(self.bundle.tables.mana_glyph.get(color));
        }
    }

    fn emit_open_zone(&mut self, zone: usize, owner: usize) {
        self.close_scalar_owner();
        self.close_option();
        let idx = zone * self.tables().owner_count + owner;
        self.out
            .extend_from_slice(self.bundle.tables.zone_open.get(idx));
    }

    fn emit_close_zone(&mut self, zone: usize, owner: usize) {
        self.close_scalar_owner();
        self.close_option();
        let idx = zone * self.tables().owner_count + owner;
        self.out
            .extend_from_slice(self.bundle.tables.zone_close.get(idx));
    }

    fn emit_zone(
        &mut self,
        zone: usize,
        owner: usize,
        cards: &[PlacedCard],
        placement: &PlacementIndex,
    ) {
        self.emit_open_zone(zone, owner);
        for card in cards {
            self.emit_placed_card_ref(card, placement);
        }
        self.emit_close_zone(zone, owner);
    }

    fn emit_placed_card_ref(&mut self, card: &PlacedCard, placement: &PlacementIndex) {
        self.close_scalar_owner();
        self.emit_card_ref(card.uuid_idx);
        self.write_one(self.tables().card_open_id);
        if let Some(slot) = placement.row_to_dict_slot.get(&card.row).copied() {
            self.write_one(self.tables().dict_entry[slot]);
        } else if card.row < self.tables().row_to_name.len() {
            self.out
                .extend_from_slice(self.bundle.tables.card_body.get(card.row));
        }
        if card.tapped_known {
            if card.tapped {
                self.out
                    .extend_from_slice(self.bundle.tables.status_tapped.as_slice());
            } else {
                self.out
                    .extend_from_slice(self.bundle.tables.status_untapped.as_slice());
            }
        } else if card.tapped {
            self.out
                .extend_from_slice(self.bundle.tables.status_tapped.as_slice());
        }
        self.out
            .extend_from_slice(self.bundle.tables.card_closer.as_slice());
    }

    fn emit_count(&mut self, amount: i32) {
        self.close_scalar_owner();
        let clamped = amount.clamp(self.tables().count_min, self.tables().count_max);
        let idx = (clamped - self.tables().count_min) as usize;
        self.out
            .extend_from_slice(self.bundle.tables.count.get(idx));
    }
}

fn emit_tokenized_state(snapshot: &Value, bundle: &TokenTableBundle) -> Result<TokenizedState> {
    let players = snapshot
        .get("players")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("snapshot missing players array"))?;
    let self_player = players.first();
    let opp_player = players.get(1);
    let placement = build_placement_index(bundle, self_player, opp_player);
    Ok(DirectEmitter::new(bundle).emit(snapshot, &placement, self_player, opp_player))
}

fn emit_decision_spec(snapshot: &Value, bundle: &TokenTableBundle) -> Option<DecisionSpecState> {
    let pending = snapshot.get("pending")?;
    let kind = pending.get("kind").and_then(Value::as_str).unwrap_or("");
    let decision_type = decision_type_for_pending_kind(kind)?;
    let options = pending
        .get("options")
        .and_then(Value::as_array)
        .map(|arr| &arr[..])
        .unwrap_or(&[]);
    let spec = &bundle.decision_spec;
    let mut out = DecisionSpecState {
        token_ids: vec![
            spec.spec_open_id,
            spec.decision_type_id,
            *spec
                .decision_type_name_ids
                .get(decision_type as usize)
                .unwrap_or(&0),
        ],
        decision_type,
        anchor_positions: Vec::new(),
        anchor_kinds: Vec::new(),
        anchor_subjects: Vec::new(),
        anchor_handles: Vec::new(),
        legal_edge_bitmap: Vec::new(),
        legal_edge_n_blockers: 0,
        legal_edge_n_attackers: 0,
    };

    match decision_type {
        DECISION_PRIORITY => {
            for (subject_idx, option_idx) in canonical_priority_option_indices(options)
                .into_iter()
                .enumerate()
            {
                out.emit_anchor(
                    ANCHOR_LEGAL_ACTION,
                    subject_idx as i32,
                    option_idx as i32,
                    spec.legal_action_id,
                );
            }
        }
        DECISION_DECLARE_ATTACKERS => {
            for (idx, _) in options.iter().enumerate() {
                out.emit_anchor(
                    ANCHOR_LEGAL_ATTACKER,
                    idx as i32,
                    idx as i32,
                    spec.legal_attacker_id,
                );
            }
            for player_idx in 0..2 {
                let token = spec.player_ref_ids.get(player_idx).copied().unwrap_or(0);
                out.emit_anchor(ANCHOR_DEFENDER, player_idx as i32, player_idx as i32, token);
            }
        }
        DECISION_DECLARE_BLOCKERS => {
            let attacker_order = blocker_attacker_order(options);
            for (idx, _) in options.iter().enumerate() {
                out.emit_anchor(
                    ANCHOR_LEGAL_BLOCKER,
                    idx as i32,
                    idx as i32,
                    spec.legal_blocker_id,
                );
            }
            for (idx, _) in attacker_order.iter().enumerate() {
                out.emit_anchor(
                    ANCHOR_LEGAL_ATTACKER,
                    idx as i32,
                    idx as i32,
                    spec.legal_attacker_id,
                );
            }
            out.legal_edge_n_blockers = options.len();
            out.legal_edge_n_attackers = attacker_order.len();
            out.legal_edge_bitmap = vec![0; options.len() * attacker_order.len()];
            let attacker_index = attacker_order
                .iter()
                .enumerate()
                .map(|(i, id)| (id.as_str(), i))
                .collect::<HashMap<_, _>>();
            for (blocker_idx, option) in options.iter().enumerate() {
                for target in value_array_any(option, &["valid_targets"]) {
                    if let Some(id) = value_str_any(target, &["id"]) {
                        if let Some(attacker_idx) = attacker_index.get(id) {
                            out.legal_edge_bitmap
                                [blocker_idx * attacker_order.len() + *attacker_idx] = 1;
                        }
                    }
                }
            }
        }
        DECISION_CHOOSE_TARGETS => {
            for (idx, _) in options.iter().enumerate() {
                out.emit_anchor(
                    ANCHOR_LEGAL_TARGET,
                    idx as i32,
                    idx as i32,
                    spec.legal_target_id,
                );
            }
        }
        DECISION_MAY => {}
        DECISION_CHOOSE_MODE => {
            out.emit_max_value(options.len() as i32, spec);
        }
        DECISION_CHOOSE_X => {
            let amount = pending
                .get("amount")
                .and_then(Value::as_i64)
                .map(|v| v as i32)
                .unwrap_or_else(|| options.len().saturating_sub(1) as i32);
            out.emit_max_value(amount, spec);
        }
        _ => {}
    }
    out.token_ids.push(spec.spec_close_id);
    Some(out)
}

impl DecisionSpecState {
    fn emit_anchor(&mut self, kind: i32, subject: i32, handle: i32, token_id: i32) {
        self.anchor_positions.push(self.token_ids.len() as i32);
        self.anchor_kinds.push(kind);
        self.anchor_subjects.push(subject);
        self.anchor_handles.push(handle);
        self.token_ids.push(token_id);
    }

    fn emit_max_value(&mut self, value: i32, spec: &DecisionSpecTables) {
        let clamped = value.max(0) as usize;
        self.token_ids.push(spec.max_value_open_id);
        self.token_ids
            .extend_from_slice(spec.max_value_digits.get(clamped));
        self.token_ids.push(spec.max_value_close_id);
    }
}

fn decision_type_for_pending_kind(kind: &str) -> Option<i32> {
    match kind {
        "priority" => Some(DECISION_PRIORITY),
        "attackers" => Some(DECISION_DECLARE_ATTACKERS),
        "blockers" => Some(DECISION_DECLARE_BLOCKERS),
        "permanent" | "cards_from_hand" | "card_from_library" => Some(DECISION_CHOOSE_TARGETS),
        "may" => Some(DECISION_MAY),
        "mode" => Some(DECISION_CHOOSE_MODE),
        "number" => Some(DECISION_CHOOSE_X),
        _ => None,
    }
}

fn blocker_attacker_order(options: &[Value]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for option in options {
        for target in value_array_any(option, &["valid_targets"]) {
            if let Some(id) = value_str_any(target, &["id"]) {
                if seen.insert(id.to_string()) {
                    out.push(id.to_string());
                }
            }
        }
    }
    out
}

fn canonical_priority_option_indices(options: &[Value]) -> Vec<usize> {
    let mut out = Vec::with_capacity(options.len());
    let mut seen = HashSet::with_capacity(options.len());
    for (idx, option) in options.iter().enumerate() {
        if is_mana_ability_option(option) {
            continue;
        }
        let key = priority_option_key(option);
        if !seen.insert(key) {
            continue;
        }
        out.push(idx);
    }
    out
}

fn is_mana_ability_option(option: &Value) -> bool {
    let kind = value_str_any(option, &["kind"])
        .unwrap_or("")
        .to_ascii_lowercase();
    if kind.contains("mana") {
        return true;
    }
    if kind != "activate" {
        return false;
    }
    let label = value_str_any(option, &["label"])
        .unwrap_or("")
        .to_ascii_lowercase();
    label.contains(" add {")
        || label.contains(" adds {")
        || label.contains(": add ")
        || label.contains(" add mana")
        || label.contains(" adds mana")
}

fn priority_option_key(option: &Value) -> String {
    let kind = value_str_any(option, &["kind"]).unwrap_or("");
    let card_name = value_str_any(option, &["card_name", "source_name", "name"]).unwrap_or("");
    if kind.starts_with("play") {
        return format!("{kind}\t{card_name}");
    }
    let label = value_str_any(option, &["label"]).unwrap_or("");
    format!("{kind}\t{card_name}\t{label}")
}

fn value_str_any<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    for key in keys {
        if let Some(s) = value.get(*key).and_then(Value::as_str) {
            return Some(s);
        }
    }
    None
}

fn value_i64_any(value: &Value, keys: &[&str]) -> Option<i64> {
    for key in keys {
        if let Some(n) = value.get(*key).and_then(Value::as_i64) {
            return Some(n);
        }
        if let Some(n) = value.get(*key).and_then(Value::as_u64) {
            return Some(n as i64);
        }
    }
    None
}

fn value_bool_any(value: &Value, keys: &[&str]) -> Option<bool> {
    for key in keys {
        if let Some(b) = value.get(*key).and_then(Value::as_bool) {
            return Some(b);
        }
    }
    None
}

fn value_array_any<'a>(value: &'a Value, keys: &[&str]) -> &'a [Value] {
    for key in keys {
        if let Some(arr) = value.get(*key).and_then(Value::as_array) {
            return arr;
        }
    }
    &[]
}

fn snapshot_turn(snapshot: &Value) -> i32 {
    value_i64_any(snapshot, &["turn"]).unwrap_or(0) as i32
}

fn snapshot_step(snapshot: &Value) -> &str {
    value_str_any(snapshot, &["step"]).unwrap_or("")
}

fn step_id(raw: &str) -> usize {
    match raw.to_ascii_uppercase().as_str() {
        "UNTAP" => 0,
        "UPKEEP" => 1,
        "DRAW" => 2,
        "PRECOMBAT_MAIN" | "PRECOMBAT MAIN" => 3,
        "BEGIN_COMBAT" | "BEGINNING_OF_COMBAT" | "BEGIN COMBAT" => 4,
        "DECLARE_ATTACKERS" | "DECLARE ATTACKERS" => 5,
        "DECLARE_BLOCKERS" | "DECLARE BLOCKERS" => 6,
        "FIRST_STRIKE_DAMAGE" | "COMBAT_DAMAGE" | "COMBAT DAMAGE" => 7,
        "END_COMBAT" | "END COMBAT" => 8,
        "POSTCOMBAT_MAIN" | "POSTCOMBAT MAIN" => 9,
        "END" | "END_TURN" => 10,
        "CLEANUP" => 11,
        _ => 12,
    }
}

fn player_life(player: Option<&Value>) -> i32 {
    player
        .and_then(|p| value_i64_any(p, &["Life", "life"]))
        .unwrap_or(0) as i32
}

fn player_library_count(player: Option<&Value>) -> i32 {
    player
        .and_then(|p| value_i64_any(p, &["LibraryCount", "librarySize"]))
        .unwrap_or(0) as i32
}

fn player_mana_pool(player: Option<&Value>) -> [i32; 6] {
    let mut out = [0; 6];
    let Some(player) = player else {
        return out;
    };
    let Some(pool) = player.get("ManaPool").or_else(|| player.get("manaPool")) else {
        return out;
    };
    if let Some(s) = pool.as_str() {
        for ch in s.chars() {
            match ch.to_ascii_uppercase() {
                'W' => out[0] += 1,
                'U' => out[1] += 1,
                'B' => out[2] += 1,
                'R' => out[3] += 1,
                'G' => out[4] += 1,
                'C' => out[5] += 1,
                _ => {}
            }
        }
        return out;
    }
    let keys = ["White", "Blue", "Black", "Red", "Green", "Colorless"];
    for (idx, key) in keys.iter().enumerate() {
        let lower = key.to_ascii_lowercase();
        out[idx] = pool
            .get(*key)
            .and_then(Value::as_i64)
            .or_else(|| pool.get(lower.as_str()).and_then(Value::as_i64))
            .unwrap_or(0) as i32;
    }
    out
}

fn card_name(card: &Value) -> &str {
    value_str_any(card, &["Name", "name"]).unwrap_or("")
}

fn card_id(card: &Value) -> &str {
    value_str_any(card, &["ID", "id"]).unwrap_or("")
}

fn build_placement_index(
    bundle: &TokenTableBundle,
    self_player: Option<&Value>,
    opp_player: Option<&Value>,
) -> PlacementIndex {
    let mut cards_by_zone: HashMap<(usize, usize), Vec<PlacedCard>> = HashMap::new();
    let mut rows_seen: HashSet<usize> = HashSet::new();
    let mut card_id_to_uuid_idx: HashMap<String, i32> = HashMap::new();
    let mut next_uuid = 0usize;
    let mut overflow = false;

    let walks = [
        (
            ZONE_BATTLEFIELD,
            OWNER_SELF,
            value_array_any(
                self_player.unwrap_or(&NULL_VALUE),
                &["Battlefield", "battlefield"],
            ),
            true,
        ),
        (
            ZONE_BATTLEFIELD,
            OWNER_OPP,
            value_array_any(
                opp_player.unwrap_or(&NULL_VALUE),
                &["Battlefield", "battlefield"],
            ),
            true,
        ),
        (
            ZONE_HAND,
            OWNER_SELF,
            value_array_any(self_player.unwrap_or(&NULL_VALUE), &["Hand", "hand"]),
            false,
        ),
        (
            ZONE_HAND,
            OWNER_OPP,
            value_array_any(opp_player.unwrap_or(&NULL_VALUE), &["Hand", "hand"]),
            false,
        ),
        (
            ZONE_GRAVEYARD,
            OWNER_SELF,
            value_array_any(
                self_player.unwrap_or(&NULL_VALUE),
                &["Graveyard", "graveyard"],
            ),
            false,
        ),
        (
            ZONE_GRAVEYARD,
            OWNER_OPP,
            value_array_any(
                opp_player.unwrap_or(&NULL_VALUE),
                &["Graveyard", "graveyard"],
            ),
            false,
        ),
        (
            ZONE_EXILE,
            OWNER_SELF,
            value_array_any(self_player.unwrap_or(&NULL_VALUE), &["Exile", "exile"]),
            false,
        ),
        (
            ZONE_EXILE,
            OWNER_OPP,
            value_array_any(opp_player.unwrap_or(&NULL_VALUE), &["Exile", "exile"]),
            false,
        ),
    ];

    for (zone, owner, raw_cards, is_battlefield) in walks {
        let mut bucket = Vec::with_capacity(raw_cards.len());
        for raw in raw_cards {
            let name = card_name(raw);
            let cid = card_id(raw);
            let row = bundle.name_to_row.get(name).copied().unwrap_or(0);
            let uuid_idx = if cid.is_empty() {
                -1
            } else if let Some(idx) = card_id_to_uuid_idx.get(cid) {
                *idx
            } else if next_uuid < MAX_CARD_REFS {
                let idx = next_uuid as i32;
                next_uuid += 1;
                card_id_to_uuid_idx.insert(cid.to_string(), idx);
                idx
            } else {
                overflow = true;
                -1
            };
            let tapped_known = is_battlefield;
            let tapped = value_bool_any(raw, &["Tapped", "tapped"]).unwrap_or(false);
            bucket.push(PlacedCard {
                row,
                uuid_idx,
                tapped_known,
                tapped,
            });
            if row > 0 {
                rows_seen.insert(row);
            }
        }
        cards_by_zone.insert((zone, owner), bucket);
    }
    let mut row_order = rows_seen.into_iter().collect::<Vec<_>>();
    row_order.sort_unstable();
    let mut row_to_dict_slot =
        HashMap::with_capacity(row_order.len().min(bundle.tables.dict_entry.len()));
    for row in &row_order {
        let slot = row_to_dict_slot.len();
        if slot >= bundle.tables.dict_entry.len() {
            break;
        }
        if bundle
            .unk_token_id
            .map(|unk| bundle.tables.dict_entry[slot] == unk)
            .unwrap_or(false)
        {
            break;
        }
        row_to_dict_slot.insert(*row, slot);
    }
    PlacementIndex {
        cards_by_zone,
        row_order,
        row_to_dict_slot,
        overflow,
    }
}

fn write_jsonl_gz(
    rx: mpsc::Receiver<WriterBatch>,
    out_path: PathBuf,
    compresslevel: u32,
) -> Result<WriterStats> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let f = File::create(&out_path).with_context(|| format!("creating {:?}", out_path))?;
    let mut gz = GzEncoder::new(BufWriter::new(f), Compression::new(compresslevel));
    let mut records = 0u64;
    for batch in rx {
        let WriterBatch::JsonLines(lines) = batch else {
            return Err(anyhow!("arrow batch sent to json writer"));
        };
        for line in lines {
            gz.write_all(&line)?;
            gz.write_all(b"\n")?;
            records += 1;
        }
    }
    gz.try_finish()?;
    Ok(WriterStats { records, shards: 0 })
}

fn write_arrow_dir(
    rx: mpsc::Receiver<WriterBatch>,
    out_dir: PathBuf,
    shard_target_rows: usize,
    pretokenized: bool,
) -> Result<WriterStats> {
    if shard_target_rows == 0 {
        return Err(anyhow!("--arrow-shard-rows must be greater than zero"));
    }
    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {:?}", out_dir))?;
    ensure_arrow_output_dir_is_clean(&out_dir)?;
    let schema = arrow_decision_schema(pretokenized);
    let mut writer = ArrowShardWriter {
        out_dir,
        schema,
        shard_target_rows,
        shard_index: 0,
        pending: Vec::new(),
        game_index: Vec::new(),
        records: 0,
    };
    for batch in rx {
        let WriterBatch::ArrowRows(rows) = batch else {
            return Err(anyhow!("json batch sent to arrow writer"));
        };
        writer.push_game_batch(rows)?;
    }
    writer.finish()
}

fn ensure_arrow_output_dir_is_clean(out_dir: &PathBuf) -> Result<()> {
    for entry in std::fs::read_dir(out_dir).with_context(|| format!("reading {:?}", out_dir))? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name == "manifest.json"
            || name == "game_index.arrow"
            || (name.starts_with("part-") && name.ends_with(".arrow"))
        {
            return Err(anyhow!(
                "Arrow output directory {:?} already contains {name}; choose a new directory",
                out_dir
            ));
        }
    }
    Ok(())
}

struct ArrowShardWriter {
    out_dir: PathBuf,
    schema: SchemaRef,
    shard_target_rows: usize,
    shard_index: u64,
    pending: Vec<ArrowDecisionRow>,
    game_index: Vec<ArrowGameIndexRow>,
    records: u64,
}

impl ArrowShardWriter {
    fn push_game_batch(&mut self, rows: Vec<ArrowDecisionRow>) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let row_start = u32::try_from(self.pending.len()).context("row_start does not fit u32")?;
        let row_count = u32::try_from(rows.len()).context("row_count does not fit u32")?;
        let shard_idx = u32::try_from(self.shard_index).context("shard_index does not fit u32")?;
        self.game_index.push(ArrowGameIndexRow {
            game_id: rows[0].game_id.clone(),
            shard_idx,
            batch_idx: 0,
            row_start,
            row_count,
        });
        self.records += rows.len() as u64;
        self.pending.extend(rows);
        if self.pending.len() >= self.shard_target_rows {
            self.flush()?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<WriterStats> {
        self.flush()?;
        write_arrow_game_index(&self.out_dir, &self.game_index)?;
        Ok(WriterStats {
            records: self.records,
            shards: self.shard_index,
        })
    }

    fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let rows = std::mem::take(&mut self.pending);
        let batch = arrow_batch_from_rows(self.schema.clone(), &rows)?;
        let path = self
            .out_dir
            .join(format!("part-{:06}.arrow", self.shard_index));
        let tmp = path.with_extension("arrow.tmp");
        let f = File::create(&tmp).with_context(|| format!("creating {:?}", tmp))?;
        let mut buf = BufWriter::new(f);
        let options = IpcWriteOptions::try_new(8, false, MetadataVersion::V5)?
            .try_with_compression(Some(CompressionType::ZSTD))?;
        {
            let mut writer =
                FileWriter::try_new_with_options(&mut buf, self.schema.as_ref(), options)?;
            writer.write(&batch)?;
            writer.finish()?;
        }
        buf.flush()?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("renaming {:?} to {:?}", tmp, path))?;
        self.shard_index += 1;
        Ok(())
    }
}

struct ArrowGameIndexRow {
    game_id: String,
    shard_idx: u32,
    batch_idx: u32,
    row_start: u32,
    row_count: u32,
}

fn arrow_game_index_schema() -> SchemaRef {
    let mut metadata = HashMap::new();
    metadata.insert(
        "format".to_string(),
        "forge_pretrain_game_index_arrow".to_string(),
    );
    metadata.insert("format_version".to_string(), "1".to_string());
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("game_id", DataType::Utf8, false),
            Field::new("shard_idx", DataType::UInt32, false),
            Field::new("batch_idx", DataType::UInt32, false),
            Field::new("row_start", DataType::UInt32, false),
            Field::new("row_count", DataType::UInt32, false),
        ],
        metadata,
    ))
}

fn arrow_game_index_batch(schema: SchemaRef, rows: &[ArrowGameIndexRow]) -> Result<RecordBatch> {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.game_id.as_str()),
        )),
        Arc::new(UInt32Array::from(
            rows.iter().map(|r| r.shard_idx).collect::<Vec<_>>(),
        )),
        Arc::new(UInt32Array::from(
            rows.iter().map(|r| r.batch_idx).collect::<Vec<_>>(),
        )),
        Arc::new(UInt32Array::from(
            rows.iter().map(|r| r.row_start).collect::<Vec<_>>(),
        )),
        Arc::new(UInt32Array::from(
            rows.iter().map(|r| r.row_count).collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema, columns).context("building Arrow game index batch")
}

fn write_arrow_game_index(out_dir: &PathBuf, rows: &[ArrowGameIndexRow]) -> Result<()> {
    let schema = arrow_game_index_schema();
    let batch = arrow_game_index_batch(schema.clone(), rows)?;
    let path = out_dir.join("game_index.arrow");
    let tmp = path.with_extension("arrow.tmp");
    let f = File::create(&tmp).with_context(|| format!("creating {:?}", tmp))?;
    let mut buf = BufWriter::new(f);
    let options = IpcWriteOptions::try_new(8, false, MetadataVersion::V5)?
        .try_with_compression(Some(CompressionType::ZSTD))?;
    {
        let mut writer = FileWriter::try_new_with_options(&mut buf, schema.as_ref(), options)?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    buf.flush()?;
    std::fs::rename(&tmp, &path).with_context(|| format!("renaming {:?} to {:?}", tmp, path))
}

fn arrow_decision_schema(pretokenized: bool) -> SchemaRef {
    let mut metadata = HashMap::new();
    metadata.insert(
        "format".to_string(),
        "forge_pretrain_decision_arrow".to_string(),
    );
    metadata.insert(
        "format_version".to_string(),
        if pretokenized { "2" } else { "1" }.to_string(),
    );
    metadata.insert(
        "stage".to_string(),
        if pretokenized {
            "pretokenized"
        } else {
            "extracted"
        }
        .to_string(),
    );
    let mut fields = vec![
        Field::new("format_version", DataType::UInt16, false),
        Field::new("game_id", DataType::Utf8, false),
        Field::new("archive_member", DataType::Utf8, false),
        Field::new("kind_id", DataType::UInt8, false),
        Field::new("candidate_index", DataType::UInt32, false),
        Field::new("candidate_count", DataType::UInt32, false),
        Field::new("source_seq", DataType::Int64, false),
        Field::new("target_seq", DataType::Int64, false),
        Field::new("perspective_id", DataType::Utf8, false),
        Field::new("perspective_name", DataType::Utf8, false),
        Field::new("winner_id", DataType::Utf8, true),
        Field::new("winner_name", DataType::Utf8, true),
        Field::new("terminal_sign", DataType::Float32, false),
        Field::new("snapshot_json", DataType::LargeUtf8, false),
        Field::new("observed_json", DataType::LargeUtf8, false),
        Field::new("outcome_players_json", DataType::LargeUtf8, false),
        Field::new("outcome_extras_json", DataType::LargeUtf8, false),
    ];
    if pretokenized {
        fields.extend([
            Field::new(
                "token_ids",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                false,
            ),
            Field::new("seq_length", DataType::UInt32, false),
            Field::new(
                "card_ref_positions",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                false,
            ),
            Field::new("token_overflow", DataType::Boolean, false),
            Field::new("card_ref_overflow", DataType::Boolean, false),
            Field::new(
                "action_token_ids",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                false,
            ),
            Field::new("action_seq_length", DataType::UInt32, false),
            Field::new("decision_type", DataType::Int32, false),
            Field::new(
                "pointer_anchor_positions",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                false,
            ),
            Field::new(
                "pointer_anchor_kinds",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                false,
            ),
            Field::new(
                "pointer_anchor_subjects",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                false,
            ),
            Field::new(
                "pointer_anchor_handles",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                false,
            ),
            Field::new(
                "legal_edge_bitmap",
                DataType::List(Arc::new(Field::new("item", DataType::UInt8, true))),
                false,
            ),
            Field::new("legal_edge_n_blockers", DataType::UInt32, false),
            Field::new("legal_edge_n_attackers", DataType::UInt32, false),
        ]);
    }
    Arc::new(Schema::new_with_metadata(fields, metadata))
}

fn arrow_batch_from_rows(schema: SchemaRef, rows: &[ArrowDecisionRow]) -> Result<RecordBatch> {
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(UInt16Array::from(
            rows.iter().map(|r| r.format_version).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.game_id.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.archive_member.as_str()),
        )),
        Arc::new(UInt8Array::from(
            rows.iter().map(|r| r.kind_id).collect::<Vec<_>>(),
        )),
        Arc::new(UInt32Array::from(
            rows.iter().map(|r| r.candidate_index).collect::<Vec<_>>(),
        )),
        Arc::new(UInt32Array::from(
            rows.iter().map(|r| r.candidate_count).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.source_seq).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.target_seq).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.perspective_id.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.perspective_name.as_str()),
        )),
        Arc::new(StringArray::from_iter(
            rows.iter().map(|r| r.winner_id.as_deref()),
        )),
        Arc::new(StringArray::from_iter(
            rows.iter().map(|r| r.winner_name.as_deref()),
        )),
        Arc::new(Float32Array::from(
            rows.iter().map(|r| r.terminal_sign).collect::<Vec<_>>(),
        )),
        Arc::new(LargeStringArray::from_iter_values(
            rows.iter().map(|r| r.snapshot_json.as_str()),
        )),
        Arc::new(LargeStringArray::from_iter_values(
            rows.iter().map(|r| r.observed_json.as_str()),
        )),
        Arc::new(LargeStringArray::from_iter_values(
            rows.iter().map(|r| r.outcome_players_json.as_str()),
        )),
        Arc::new(LargeStringArray::from_iter_values(
            rows.iter().map(|r| r.outcome_extras_json.as_str()),
        )),
    ];
    if schema.field_with_name("token_ids").is_ok() {
        let mut token_builder = ListBuilder::new(Int32Builder::new());
        let mut card_ref_builder = ListBuilder::new(Int32Builder::new());
        let mut action_builder = ListBuilder::new(Int32Builder::new());
        let mut anchor_pos_builder = ListBuilder::new(Int32Builder::new());
        let mut anchor_kind_builder = ListBuilder::new(Int32Builder::new());
        let mut anchor_subject_builder = ListBuilder::new(Int32Builder::new());
        let mut anchor_handle_builder = ListBuilder::new(Int32Builder::new());
        let mut edge_builder = ListBuilder::new(UInt8Builder::new());
        for row in rows {
            let token_ids = row
                .token_ids
                .as_ref()
                .ok_or_else(|| anyhow!("pretokenized schema row missing token_ids"))?;
            token_builder.values().append_slice(token_ids);
            token_builder.append(true);
            let card_ref_positions = row
                .card_ref_positions
                .as_ref()
                .ok_or_else(|| anyhow!("pretokenized schema row missing card_ref_positions"))?;
            card_ref_builder.values().append_slice(card_ref_positions);
            card_ref_builder.append(true);
            let action_token_ids = row
                .action_token_ids
                .as_ref()
                .ok_or_else(|| anyhow!("pretokenized schema row missing action_token_ids"))?;
            action_builder.values().append_slice(action_token_ids);
            action_builder.append(true);
            let anchor_positions = row.pointer_anchor_positions.as_ref().ok_or_else(|| {
                anyhow!("pretokenized schema row missing pointer_anchor_positions")
            })?;
            anchor_pos_builder.values().append_slice(anchor_positions);
            anchor_pos_builder.append(true);
            let anchor_kinds = row
                .pointer_anchor_kinds
                .as_ref()
                .ok_or_else(|| anyhow!("pretokenized schema row missing pointer_anchor_kinds"))?;
            anchor_kind_builder.values().append_slice(anchor_kinds);
            anchor_kind_builder.append(true);
            let anchor_subjects = row.pointer_anchor_subjects.as_ref().ok_or_else(|| {
                anyhow!("pretokenized schema row missing pointer_anchor_subjects")
            })?;
            anchor_subject_builder
                .values()
                .append_slice(anchor_subjects);
            anchor_subject_builder.append(true);
            let anchor_handles = row
                .pointer_anchor_handles
                .as_ref()
                .ok_or_else(|| anyhow!("pretokenized schema row missing pointer_anchor_handles"))?;
            anchor_handle_builder.values().append_slice(anchor_handles);
            anchor_handle_builder.append(true);
            let legal_edge_bitmap = row
                .legal_edge_bitmap
                .as_ref()
                .ok_or_else(|| anyhow!("pretokenized schema row missing legal_edge_bitmap"))?;
            edge_builder.values().append_slice(legal_edge_bitmap);
            edge_builder.append(true);
        }
        columns.extend([
            Arc::new(token_builder.finish()) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter()
                    .map(|r| r.seq_length.unwrap_or(0))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(card_ref_builder.finish()) as ArrayRef,
            Arc::new(BooleanArray::from(
                rows.iter().map(|r| r.token_overflow).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                rows.iter().map(|r| r.card_ref_overflow).collect::<Vec<_>>(),
            )),
            Arc::new(action_builder.finish()) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter()
                    .map(|r| r.action_seq_length.unwrap_or(0))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                rows.iter()
                    .map(|r| r.decision_type.unwrap_or(-1))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(anchor_pos_builder.finish()) as ArrayRef,
            Arc::new(anchor_kind_builder.finish()) as ArrayRef,
            Arc::new(anchor_subject_builder.finish()) as ArrayRef,
            Arc::new(anchor_handle_builder.finish()) as ArrayRef,
            Arc::new(edge_builder.finish()) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter()
                    .map(|r| r.legal_edge_n_blockers.unwrap_or(0))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(
                rows.iter()
                    .map(|r| r.legal_edge_n_attackers.unwrap_or(0))
                    .collect::<Vec<_>>(),
            )),
        ]);
    }
    RecordBatch::try_new(schema, columns).context("building Arrow record batch")
}

struct ArrowManifestStats {
    shards: u64,
    records_written: u64,
    shard_target_rows: usize,
    games_seen: u64,
    games_written: u64,
    candidates_seen: u64,
    kind_counts: [u64; 5],
    pretokenized: bool,
}

fn write_arrow_manifest(out_dir: &PathBuf, stats: &ArrowManifestStats) -> Result<()> {
    let manifest = json!({
        "format": "forge_pretrain_decision_arrow",
        "format_version": if stats.pretokenized { 2 } else { 1 },
        "stage": if stats.pretokenized { "pretokenized" } else { "extracted" },
        "compression": "zstd",
        "game_index": "game_index.arrow",
        "shards": stats.shards,
        "records_written": stats.records_written,
        "shard_target_rows": stats.shard_target_rows,
        "stats": {
            "games_seen": stats.games_seen,
            "games_written": stats.games_written,
            "candidates_seen": stats.candidates_seen,
            "written_priority": stats.kind_counts[0],
            "written_attack": stats.kind_counts[1],
            "written_block": stats.kind_counts[2],
            "written_may": stats.kind_counts[3],
            "written_choose": stats.kind_counts[4],
        },
    });
    let path = out_dir.join("manifest.json");
    std::fs::write(&path, sonic_rs::to_string_pretty(&manifest)?)
        .with_context(|| format!("writing {:?}", path))
}

#[derive(Debug, Deserialize)]
struct Row {
    #[serde(default)]
    record: Option<String>,
    #[serde(rename = "type", default)]
    row_type: Option<String>,
    #[serde(default)]
    seq: Option<i64>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    snapshot: Option<Value>,
    #[serde(rename = "gameId", default)]
    game_id: Option<String>,
    #[serde(rename = "winnerId", default)]
    winner_id: Option<String>,
    #[serde(rename = "winnerName", default)]
    winner_name: Option<String>,
    #[serde(default)]
    players: Option<Vec<Value>>,
    #[serde(default)]
    extras: Option<Value>,
}

impl Row {
    fn record(&self) -> &str {
        self.record.as_deref().unwrap_or("")
    }

    fn row_type(&self) -> &str {
        self.row_type.as_deref().unwrap_or("")
    }

    fn seq(&self) -> i64 {
        self.seq.unwrap_or(0)
    }

    fn description(&self) -> &str {
        self.description.as_deref().unwrap_or("")
    }

    fn snapshot(&self) -> Option<&Value> {
        self.snapshot.as_ref()
    }

    fn snapshot_or_null(&self) -> &Value {
        self.snapshot.as_ref().unwrap_or(&NULL_VALUE)
    }

    fn snapshot_step(&self) -> &str {
        self.snapshot()
            .and_then(|s| s.get("step"))
            .and_then(Value::as_str)
            .unwrap_or("")
    }

    fn has_object_snapshot(&self) -> bool {
        self.snapshot
            .as_ref()
            .map(Value::is_object)
            .unwrap_or(false)
    }
}

type NormalizedSnapshotCache = HashMap<(i64, String), Value>;

fn choice_kind_index(kind: &str) -> Option<usize> {
    match kind {
        "priority" => Some(0),
        "attack" => Some(1),
        "block" => Some(2),
        "may" => Some(3),
        "choose" => Some(4),
        _ => None,
    }
}

fn parse_kinds(raw: &str) -> Result<Vec<String>> {
    if raw == "all" {
        return Ok(CHOICE_KINDS.iter().map(|s| s.to_string()).collect());
    }
    let mut out = Vec::new();
    for item in raw.split(',') {
        let k = item.trim();
        if k.is_empty() {
            continue;
        }
        if !CHOICE_KINDS.iter().any(|c| *c == k) {
            return Err(anyhow!("unknown kind {k:?}"));
        }
        out.push(k.to_string());
    }
    Ok(out)
}

fn parse_kind_priority(raw: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for item in raw.split(',') {
        let k = item.trim();
        if k.is_empty() {
            continue;
        }
        if !CHOICE_KINDS.iter().any(|c| *c == k) {
            return Err(anyhow!("unknown priority kind {k:?}"));
        }
        out.push(k.to_string());
    }
    if out.is_empty() {
        out = DEFAULT_KIND_PRIORITY
            .iter()
            .map(|s| s.to_string())
            .collect();
    }
    Ok(out)
}

// --- IO ---------------------------------------------------------------------

fn read_jsonl_member<R: std::io::Read>(reader: R) -> Result<Vec<Row>> {
    // gzip + jsonl. Filter to META and EVENTs that carry a snapshot (matches python).
    let gz = MultiGzDecoder::new(reader);
    let buf = BufReader::new(gz);
    let mut rows = Vec::new();
    for line in buf.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = match sonic_rs::from_str(&line) {
            Ok(row) => row,
            Err(_) => continue, // skip malformed lines defensively
        };
        let keep = match row.record() {
            "META" => true,
            "EVENT" => row.has_object_snapshot(),
            _ => false,
        };
        if keep {
            rows.push(row);
        }
    }
    Ok(rows)
}

// --- Meta -------------------------------------------------------------------

struct Meta<'a> {
    game_id: &'a str,
    winner_id: Option<&'a str>,
    winner_name: Option<&'a str>,
    players: Option<&'a [Value]>,
    extras: Option<&'a Value>,
}

fn meta_from_rows(rows: &[Row]) -> Option<Meta<'_>> {
    for row in rows {
        if row.record() != "META" {
            continue;
        }
        let game_id = row.game_id.as_deref().unwrap_or("");
        let winner_id = row.winner_id.as_deref();
        let winner_name = row.winner_name.as_deref();
        let players = row.players.as_deref();
        let extras = row.extras.as_ref();
        return Some(Meta {
            game_id,
            winner_id,
            winner_name,
            players,
            extras,
        });
    }
    None
}

// --- Snapshot normalization -------------------------------------------------

fn step_label(raw: &str) -> &'static str {
    match raw.to_ascii_uppercase().as_str() {
        "UNTAP" => "Untap",
        "UPKEEP" => "Upkeep",
        "DRAW" => "Draw",
        "PRECOMBAT_MAIN" => "Precombat Main",
        "BEGIN_COMBAT" | "BEGINNING_OF_COMBAT" => "Begin Combat",
        "DECLARE_ATTACKERS" => "Declare Attackers",
        "DECLARE_BLOCKERS" => "Declare Blockers",
        "FIRST_STRIKE_DAMAGE" | "COMBAT_DAMAGE" => "Combat Damage",
        "END_COMBAT" => "End Combat",
        "POSTCOMBAT_MAIN" => "Postcombat Main",
        "END" | "END_TURN" => "End",
        "CLEANUP" => "Cleanup",
        _ => "Unknown",
    }
}

fn step_label_or_passthrough(raw: &str) -> String {
    let mapped = step_label(raw);
    if mapped == "Unknown" && !raw.is_empty() {
        raw.to_string()
    } else {
        mapped.to_string()
    }
}

fn mana_pool(raw: &str) -> Value {
    let mut counts = [0u32; 6];
    for ch in raw.chars() {
        let idx = match ch.to_ascii_uppercase() {
            'W' => 0,
            'U' => 1,
            'B' => 2,
            'R' => 3,
            'G' => 4,
            'C' => 5,
            _ => continue,
        };
        counts[idx] += 1;
    }
    json!({
        "White": counts[0],
        "Blue": counts[1],
        "Black": counts[2],
        "Red": counts[3],
        "Green": counts[4],
        "Colorless": counts[5],
    })
}

fn zone_card(raw: &Value, owner_id: &str, zone: &str, index: usize) -> Value {
    if let Some(s) = raw.as_str() {
        return json!({
            "ID": format!("{owner_id}:{zone}:{index}:{s}"),
            "Name": s,
        });
    }
    if let Some(obj) = raw.as_object() {
        let name = obj
            .get(&"name")
            .and_then(Value::as_str)
            .or_else(|| obj.get(&"Name").and_then(Value::as_str))
            .unwrap_or("");
        let cid = obj
            .get(&"id")
            .and_then(Value::as_str)
            .or_else(|| obj.get(&"ID").and_then(Value::as_str))
            .map(str::to_string)
            .unwrap_or_else(|| format!("{owner_id}:{zone}:{index}:{name}"));
        if let Some(t) = obj.get(&"tapped").or_else(|| obj.get(&"Tapped")) {
            return json!({
                "ID": cid,
                "Name": name,
                "Tapped": t.as_bool().unwrap_or(false),
            });
        }
        return json!({
            "ID": cid,
            "Name": name,
        });
    }
    json!({
        "ID": format!("{owner_id}:{zone}:{index}:unknown"),
        "Name": "",
    })
}

fn player(raw: &Value) -> Value {
    let Some(obj) = raw.as_object() else {
        return json!({
            "ID": "",
            "Name": "",
            "Life": 0,
            "HandCount": 0,
            "GraveyardCount": 0,
            "LibraryCount": 0,
            "Hand": [],
            "Graveyard": [],
            "Exile": [],
            "ManaPool": mana_pool(""),
        });
    };
    let pid = obj
        .get(&"id")
        .and_then(Value::as_str)
        .or_else(|| obj.get(&"ID").and_then(Value::as_str))
        .unwrap_or("")
        .to_string();
    let name = obj
        .get(&"name")
        .and_then(Value::as_str)
        .or_else(|| obj.get(&"Name").and_then(Value::as_str))
        .unwrap_or("")
        .to_string();
    let life = obj
        .get(&"life")
        .or_else(|| obj.get(&"Life"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let library_count = obj
        .get(&"librarySize")
        .or_else(|| obj.get(&"LibraryCount"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let hand: Vec<Value> = obj
        .get(&"hand")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, c)| zone_card(c, &pid, "hand", i))
                .collect()
        })
        .unwrap_or_default();
    let graveyard: Vec<Value> = obj
        .get(&"graveyard")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, c)| zone_card(c, &pid, "graveyard", i))
                .collect()
        })
        .unwrap_or_default();
    let exile: Vec<Value> = obj
        .get(&"exile")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, c)| zone_card(c, &pid, "exile", i))
                .collect()
        })
        .unwrap_or_default();
    let hand_count = obj
        .get(&"handSize")
        .and_then(Value::as_u64)
        .unwrap_or(hand.len() as u64);
    let mp_raw = obj.get(&"manaPool").and_then(Value::as_str).unwrap_or("");

    json!({
        "ID": pid,
        "Name": name,
        "Life": life,
        "HandCount": hand_count,
        "GraveyardCount": graveyard.len(),
        "LibraryCount": library_count,
        "Hand": hand,
        "Graveyard": graveyard,
        "Exile": exile,
        "ManaPool": mana_pool(mp_raw),
    })
}

fn normalize_snapshot(raw: &Value, perspective_id: &str) -> Value {
    let players_in: Vec<Value> = raw
        .get("players")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().cloned().collect())
        .unwrap_or_default();
    let mut players: Vec<Value> = players_in.iter().map(player).collect();

    // Group battlefield by controller.
    let mut bf_by_player: HashMap<String, Vec<Value>> = HashMap::new();
    for p in &players {
        if let Some(id) = p.get("ID").and_then(Value::as_str) {
            bf_by_player.entry(id.to_string()).or_default();
        }
    }
    if let Some(arr) = raw.get("battlefield").and_then(Value::as_array) {
        for (i, card) in arr.iter().enumerate() {
            let controller = card
                .get("controllerId")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            bf_by_player
                .entry(controller.clone())
                .or_default()
                .push(zone_card(card, &controller, "bf", i));
        }
    }
    for p in &mut players {
        let id = p
            .get("ID")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let bf = bf_by_player.get(&id).cloned().unwrap_or_default();
        if let Some(obj) = p.as_object_mut() {
            obj.insert(&"Battlefield", bf);
        }
    }

    // Reorder so perspective player comes first.
    if !perspective_id.is_empty() {
        if let Some(idx) = players
            .iter()
            .position(|p| p.get("ID").and_then(Value::as_str) == Some(perspective_id))
        {
            if idx != 0 {
                let head = players.remove(idx);
                players.insert(0, head);
            }
        }
    }

    let active_id = raw
        .get("activePlayerId")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let stack: Vec<Value> = raw
        .get("stack")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, obj)| {
                    let name = if let Some(o) = obj.as_object() {
                        o.get(&"name")
                            .and_then(Value::as_str)
                            .or_else(|| o.get(&"description").and_then(Value::as_str))
                            .unwrap_or("")
                            .to_string()
                    } else {
                        obj.as_str().unwrap_or("").to_string()
                    };
                    json!({"id": format!("stack:{i}:{name}"), "name": name})
                })
                .collect()
        })
        .unwrap_or_default();

    let turn = raw.get("turn").and_then(Value::as_u64).unwrap_or(0);
    let step = step_label_or_passthrough(raw.get("step").and_then(Value::as_str).unwrap_or(""));

    json!({
        "turn": turn,
        "active_player": active_id,
        "step": step,
        "players": players,
        "stack": stack,
    })
}

// --- pending builders -------------------------------------------------------

fn forge_kind_to_option_kind(kind: &str) -> Option<&'static str> {
    match kind {
        // Legacy schema.
        "PLAY_LAND" => Some("play"),
        "CAST_SPELL" => Some("cast"),
        "ACTIVATE" | "ACTIVATE_MANA" => Some("activate"),
        // Current schema.
        "LAND" => Some("play"),
        "SPELL" => Some("cast"),
        "ACTIVATED" => Some("activate"),
        _ => None,
    }
}

fn priority_pending(raw_snapshot: &Value, _perspective_id: &str) -> Option<Value> {
    let actions = raw_snapshot
        .get("playableActions")
        .and_then(Value::as_array)?;
    let mut options: Vec<Value> = Vec::with_capacity(actions.len() + 1);
    let mut seen_options: HashSet<String> = HashSet::new();
    for (i, act) in actions.iter().enumerate() {
        let Some(obj) = act.as_object() else { continue };
        let kind = obj
            .get(&"type")
            .and_then(Value::as_str)
            .or_else(|| obj.get(&"kind").and_then(Value::as_str))
            .unwrap_or("");
        if kind == "PASS" {
            // Pass is appended unconditionally below.
            continue;
        }
        let description = obj
            .get(&"description")
            .and_then(Value::as_str)
            .unwrap_or("");
        if is_mana_ability_action(kind, description) {
            continue;
        }
        let opt_kind = match forge_kind_to_option_kind(kind) {
            Some(k) => k,
            None => continue,
        };
        let ability_id = obj.get(&"abilityId").and_then(Value::as_str).unwrap_or("");
        let source_id = obj.get(&"sourceId").and_then(Value::as_str).unwrap_or("");
        let source_name = obj.get(&"sourceName").and_then(Value::as_str).unwrap_or("");
        let description = obj
            .get(&"description")
            .and_then(Value::as_str)
            .unwrap_or("");
        let label = if description.is_empty() {
            source_name
        } else {
            description
        };
        let cost = obj.get(&"cost").and_then(Value::as_str).unwrap_or("");
        let id = if ability_id.is_empty() {
            format!("opt:{i}")
        } else {
            ability_id.to_string()
        };
        let dedupe_key = priority_option_dedupe_key(opt_kind, source_name, label);
        if !seen_options.insert(dedupe_key) {
            continue;
        }
        options.push(json!({
            "id": id,
            "kind": opt_kind,
            "card_id": source_id,
            "card_name": source_name,
            "permanent_id": source_id,
            "label": label,
            "mana_cost": cost,
        }));
    }
    options.push(json!({"id": "pass", "kind": "pass", "label": "Pass priority"}));
    Some(json!({
        "kind": "priority",
        "player_idx": 0,
        "options": options,
    }))
}

fn priority_option_dedupe_key(kind: &str, source_name: &str, label: &str) -> String {
    if kind == "play" {
        return format!("{kind}\t{source_name}");
    }
    format!("{kind}\t{source_name}\t{label}")
}

fn attackers_pending(
    raw_snapshot: &Value,
    perspective_id: &str,
    display_ids: &DisplayIdIndex,
) -> Option<Value> {
    let bf = raw_snapshot.get("battlefield").and_then(Value::as_array)?;
    let mut options: Vec<Value> = Vec::new();
    let mut display_usage = HashMap::new();
    for card in bf {
        let Some(obj) = card.as_object() else {
            continue;
        };
        if obj.get(&"controllerId").and_then(Value::as_str) != Some(perspective_id) {
            continue;
        }
        let power = obj.get(&"power").and_then(Value::as_i64).unwrap_or(0);
        if power <= 0 {
            continue;
        }
        let id = card_handle(card, display_ids, &mut display_usage);
        let name = obj
            .get(&"name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        options.push(json!({
            "id": id,
            "kind": "attacker",
            "card_id": id,
            "card_name": name,
            "permanent_id": id,
            "label": name,
        }));
    }
    if options.is_empty() {
        return None;
    }
    Some(json!({"kind": "attackers", "player_idx": 0, "options": options}))
}

fn blockers_pending(
    raw_snapshot: &Value,
    perspective_id: &str,
    attacker_refs: &[(String, String)],
    display_ids: &DisplayIdIndex,
) -> Option<Value> {
    let bf = raw_snapshot.get("battlefield").and_then(Value::as_array)?;

    let mut attacker_targets: Vec<Value> = Vec::new();
    for (prefix, name) in attacker_refs {
        if prefix.starts_with("forge:") {
            attacker_targets.push(json!({"id": prefix, "label": name}));
            continue;
        }
        for card in bf {
            let Some(obj) = card.as_object() else {
                continue;
            };
            let cid = obj.get(&"id").and_then(Value::as_str).unwrap_or("");
            if cid.starts_with(prefix.as_str()) {
                let name = obj.get(&"name").and_then(Value::as_str).unwrap_or("");
                attacker_targets.push(json!({"id": cid, "label": name}));
                break;
            }
        }
    }
    if attacker_targets.is_empty() {
        return None;
    }
    let mut options: Vec<Value> = Vec::new();
    let mut display_usage = HashMap::new();
    for card in bf {
        let Some(obj) = card.as_object() else {
            continue;
        };
        if obj.get(&"controllerId").and_then(Value::as_str) != Some(perspective_id) {
            continue;
        }
        let power = obj.get(&"power").and_then(Value::as_i64).unwrap_or(0);
        if power <= 0 {
            continue;
        }
        if obj.get(&"tapped").and_then(Value::as_bool).unwrap_or(false) {
            continue;
        }
        let id = card_handle(card, display_ids, &mut display_usage);
        let name = obj
            .get(&"name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        options.push(json!({
            "id": id,
            "kind": "block",
            "card_id": id,
            "card_name": name,
            "permanent_id": id,
            "label": name,
            "valid_targets": attacker_targets,
        }));
    }
    if options.is_empty() {
        return None;
    }
    Some(json!({"kind": "blockers", "player_idx": 0, "options": options}))
}

// --- helpers on raw snapshot ------------------------------------------------

fn raw_player_id_by_name<'a>(snapshot: &'a Value, name: &str) -> &'a str {
    if let Some(arr) = snapshot.get("players").and_then(Value::as_array) {
        for p in arr {
            if p.get("name").and_then(Value::as_str) == Some(name) {
                return p.get("id").and_then(Value::as_str).unwrap_or("");
            }
        }
    }
    ""
}

fn raw_player_name_by_id<'a>(snapshot: &'a Value, id: &str) -> &'a str {
    if let Some(arr) = snapshot.get("players").and_then(Value::as_array) {
        for p in arr {
            if p.get("id").and_then(Value::as_str) == Some(id) {
                return p.get("name").and_then(Value::as_str).unwrap_or("");
            }
        }
    }
    ""
}

fn raw_opponent_id<'a>(snapshot: &'a Value, pid: &str) -> &'a str {
    if let Some(arr) = snapshot.get("players").and_then(Value::as_array) {
        for p in arr {
            let candidate = p.get("id").and_then(Value::as_str).unwrap_or("");
            if !candidate.is_empty() && candidate != pid {
                return candidate;
            }
        }
    }
    ""
}

// --- regexes / label parsing ------------------------------------------------

static ATTACKS_HEADER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?P<attacker_player>.+?) attacks (?P<defender_player>.+?) with (?P<count>\d+) creatures?$").unwrap()
});
static ATTACKER_LINE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"^Attacker:\s+(?P<name>.+?)\s+\[(?P<id>[0-9a-f]+)\]\s+\((?P<pt>[^)]+)\)\s+(?P<rest>.+)$",
    )
    .unwrap()
});
static BLOCKER_TOKEN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?P<name>.+?)\s+\[(?P<id>[0-9a-f]+)\]\s+\((?P<pt>[^)]+)\)").unwrap());
static PLAYER_PREFIX_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(?P<player>Player[A-Z0-9_ -]+)").unwrap());
static STACK_PUSH_NAME_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^STACK_PUSH\s+(?P<name>.+?)\s+by\s+(?P<player>.+?)$").unwrap());
static LOG_PUTS_BF_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?P<player>.+?) puts (?P<name>.+?) \[[^\]]+\] from hand onto the Battlefield")
        .unwrap()
});
static COMBAT_CARD_REF_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?P<name>[^()\n]+?)\s+\((?P<id>[0-9]+)\)").unwrap());
static ASSIGNED_ATTACK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?P<player>.+?) assigned (?P<attackers>.+) to attack (?P<defender>.+?)\.$")
        .unwrap()
});
static ASSIGNED_BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?P<player>.+?) assigned (?P<blockers>.+?) to block (?P<attacker>.+?)\.$")
        .unwrap()
});
static DIDNT_BLOCK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(?P<player>.+?) didn't block (?P<attacker>.+?)\.$").unwrap());

type DisplayIdIndex = HashMap<String, Vec<String>>;

#[derive(Clone)]
struct CombatCardRef {
    name: String,
    id_prefix: String,
}

#[derive(Clone)]
struct AttacksHeader {
    attacker_player: String,
    defender_player: String,
    count: usize,
}

fn parse_attacks_header(desc: &str) -> Option<AttacksHeader> {
    let cap = ATTACKS_HEADER_RE.captures(desc)?;
    Some(AttacksHeader {
        attacker_player: cap["attacker_player"].to_string(),
        defender_player: cap["defender_player"].to_string(),
        count: cap["count"].parse().ok()?,
    })
}

#[derive(Clone)]
struct AttackerLine {
    name: String,
    id_prefix: String,
    blockers: Vec<(String, String)>, // (name, id_prefix)
}

fn parse_attacker_line(desc: &str) -> Option<AttackerLine> {
    let cap = ATTACKER_LINE_RE.captures(desc)?;
    let name = cap["name"].to_string();
    let id_prefix = cap["id"].to_string();
    let rest = cap["rest"].to_string();
    let mut blockers = Vec::new();
    if rest != "unblocked" && rest.starts_with("blocked by ") {
        let body = &rest["blocked by ".len()..];
        for blk in BLOCKER_TOKEN_RE.captures_iter(body) {
            blockers.push((blk["name"].to_string(), blk["id"].to_string()));
        }
    }
    Some(AttackerLine {
        name,
        id_prefix,
        blockers,
    })
}

fn forge_display_handle(display_id: &str) -> String {
    format!("forge:{display_id}")
}

fn clean_combat_name(raw: &str) -> String {
    let mut s = raw.trim().trim_start_matches(',').trim();
    if let Some(rest) = s.strip_prefix("and ") {
        s = rest.trim();
    }
    s.to_string()
}

fn parse_combat_card_refs(raw: &str) -> Vec<CombatCardRef> {
    COMBAT_CARD_REF_RE
        .captures_iter(raw)
        .filter_map(|cap| {
            let name = clean_combat_name(&cap["name"]);
            if name.is_empty() {
                return None;
            }
            Some(CombatCardRef {
                name,
                id_prefix: forge_display_handle(&cap["id"]),
            })
        })
        .collect()
}

fn combat_ref_json(card: &CombatCardRef) -> Value {
    json!({"name": card.name, "id_prefix": card.id_prefix})
}

fn push_unique_combat_ref(out: &mut Vec<CombatCardRef>, card: CombatCardRef) {
    if out.iter().any(|c| c.id_prefix == card.id_prefix) {
        return;
    }
    out.push(card);
}

fn parse_assigned_attack(desc: &str) -> Option<(String, String, Vec<CombatCardRef>)> {
    let cap = ASSIGNED_ATTACK_RE.captures(desc)?;
    let attackers = parse_combat_card_refs(&cap["attackers"]);
    if attackers.is_empty() {
        return None;
    }
    Some((
        cap["player"].to_string(),
        cap["defender"].to_string(),
        attackers,
    ))
}

fn parse_assigned_block(desc: &str) -> Option<(String, Vec<CombatCardRef>, CombatCardRef)> {
    let cap = ASSIGNED_BLOCK_RE.captures(desc)?;
    let blockers = parse_combat_card_refs(&cap["blockers"]);
    let mut attackers = parse_combat_card_refs(&cap["attacker"]);
    if blockers.is_empty() || attackers.is_empty() {
        return None;
    }
    Some((cap["player"].to_string(), blockers, attackers.remove(0)))
}

fn parse_didnt_block(desc: &str) -> Option<(String, CombatCardRef)> {
    let cap = DIDNT_BLOCK_RE.captures(desc)?;
    let mut attackers = parse_combat_card_refs(&cap["attacker"]);
    if attackers.is_empty() {
        return None;
    }
    Some((cap["player"].to_string(), attackers.remove(0)))
}

fn has_assigned_attack_log(row: &Row) -> bool {
    if row.row_type() != "LOG" {
        return false;
    }
    row.description()
        .lines()
        .any(|line| parse_assigned_attack(line).is_some())
}

fn has_new_block_log(row: &Row) -> bool {
    if row.row_type() != "LOG" {
        return false;
    }
    row.description()
        .lines()
        .any(|line| parse_assigned_block(line).is_some() || parse_didnt_block(line).is_some())
}

fn assigned_attack_observed(window: &[&Row]) -> Option<Value> {
    let mut actor_name = String::new();
    let mut defender_name = String::new();
    let mut attackers: Vec<CombatCardRef> = Vec::new();
    let mut raw_lines: Vec<String> = Vec::new();

    for row in window {
        if row.row_type() != "LOG" {
            continue;
        }
        let desc = row.description();
        for line in desc.lines() {
            let Some((actor, defender, cards)) = parse_assigned_attack(line) else {
                continue;
            };
            if actor_name.is_empty() {
                actor_name = actor;
            }
            if defender_name.is_empty() {
                defender_name = defender;
            }
            for card in cards {
                push_unique_combat_ref(&mut attackers, card);
            }
            raw_lines.push(line.to_string());
        }
    }
    if attackers.is_empty() {
        return None;
    }
    Some(json!({
        "raw": raw_lines.join("\n"),
        "actor_name": actor_name,
        "defender_name": defender_name,
        "attackers": attackers.iter().map(combat_ref_json).collect::<Vec<_>>(),
    }))
}

fn assigned_block_observed(window: &[&Row]) -> Option<Value> {
    let mut actor_name = String::new();
    let mut attackers: Vec<CombatCardRef> = Vec::new();
    let mut assignments: Vec<Value> = Vec::new();
    let mut raw_lines: Vec<String> = Vec::new();

    for row in window {
        if row.row_type() != "LOG" {
            continue;
        }
        let desc = row.description();
        for line in desc.lines() {
            if let Some((actor, blockers, attacker)) = parse_assigned_block(line) {
                if actor_name.is_empty() {
                    actor_name = actor;
                }
                push_unique_combat_ref(&mut attackers, attacker.clone());
                for blocker in blockers {
                    assignments.push(json!({
                        "attacker_name": attacker.name,
                        "attacker_id_prefix": attacker.id_prefix,
                        "blocker_name": blocker.name,
                        "blocker_id_prefix": blocker.id_prefix,
                    }));
                }
                raw_lines.push(line.to_string());
                continue;
            }
            if let Some((actor, attacker)) = parse_didnt_block(line) {
                if actor_name.is_empty() {
                    actor_name = actor;
                }
                push_unique_combat_ref(&mut attackers, attacker);
                raw_lines.push(line.to_string());
            }
        }
    }
    if attackers.is_empty() && assignments.is_empty() {
        return None;
    }

    let snap = window
        .first()
        .map(|r| r.snapshot_or_null())
        .unwrap_or(&NULL_VALUE);
    let active = snap
        .get("activePlayerId")
        .and_then(Value::as_str)
        .unwrap_or("");
    Some(json!({
        "raw": raw_lines.join("\n"),
        "actor_name": actor_name,
        "attacker_player": raw_player_name_by_id(snap, active),
        "attackers": attackers.iter().map(combat_ref_json).collect::<Vec<_>>(),
        "assignments": assignments,
        "no_block": assignments.is_empty(),
    }))
}

fn display_key(controller_id: &str, card_name: &str) -> String {
    format!("{controller_id}\t{card_name}")
}

fn add_display_ref(
    display_ids: &mut DisplayIdIndex,
    controller_id: &str,
    card_name: &str,
    id_prefix: &str,
) {
    if controller_id.is_empty() || card_name.is_empty() || !id_prefix.starts_with("forge:") {
        return;
    }
    let ids = display_ids
        .entry(display_key(controller_id, card_name))
        .or_default();
    if !ids.iter().any(|id| id == id_prefix) {
        ids.push(id_prefix.to_string());
    }
}

fn display_ids_from_observed(
    raw_snapshot: &Value,
    perspective_id: &str,
    kind: &str,
    observed: &Value,
) -> DisplayIdIndex {
    let mut out = DisplayIdIndex::new();
    match kind {
        "attack" => {
            for card in observed
                .get("attackers")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                add_display_ref(
                    &mut out,
                    perspective_id,
                    card.get("name").and_then(Value::as_str).unwrap_or(""),
                    card.get("id_prefix").and_then(Value::as_str).unwrap_or(""),
                );
            }
        }
        "block" => {
            let active_id = raw_snapshot
                .get("activePlayerId")
                .and_then(Value::as_str)
                .unwrap_or("");
            for card in observed
                .get("attackers")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                add_display_ref(
                    &mut out,
                    active_id,
                    card.get("name").and_then(Value::as_str).unwrap_or(""),
                    card.get("id_prefix").and_then(Value::as_str).unwrap_or(""),
                );
            }
            for assignment in observed
                .get("assignments")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                add_display_ref(
                    &mut out,
                    perspective_id,
                    assignment
                        .get("blocker_name")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                    assignment
                        .get("blocker_id_prefix")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                );
            }
        }
        _ => {}
    }
    out
}

fn card_handle(
    card: &Value,
    display_ids: &DisplayIdIndex,
    display_usage: &mut HashMap<String, usize>,
) -> String {
    let controller = card
        .get("controllerId")
        .and_then(Value::as_str)
        .unwrap_or("");
    let name = card.get("name").and_then(Value::as_str).unwrap_or("");
    let key = display_key(controller, name);
    if let Some(ids) = display_ids.get(&key) {
        let used = display_usage.entry(key).or_insert(0);
        if *used < ids.len() {
            let id = ids[*used].clone();
            *used += 1;
            return id;
        }
    }
    card.get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("{controller}:bf:{name}"))
}

fn priority_label(desc: &str, event_type: &str) -> Option<Value> {
    if event_type == "STACK_PUSH"
        && (contains_ascii_case_insensitive(desc, " cast ")
            || contains_ascii_case_insensitive(desc, " activated "))
    {
        return Some(json!({"raw": desc, "event_type": event_type}));
    }
    if contains_ascii_case_insensitive(desc, " played ") {
        return Some(json!({"raw": desc, "event_type": event_type}));
    }
    if let Some(cap) = STACK_PUSH_NAME_RE.captures(desc) {
        return Some(json!({
            "raw": desc,
            "event_type": event_type,
            "card_name": cap["name"].to_string(),
            "player_name": cap["player"].to_string(),
        }));
    }
    if let Some(cap) = LOG_PUTS_BF_RE.captures(desc) {
        return Some(json!({
            "raw": desc,
            "event_type": event_type,
            "card_name": cap["name"].to_string(),
            "player_name": cap["player"].to_string(),
            "is_land_play": true,
        }));
    }
    None
}

fn is_pass_action(kind: &str) -> bool {
    kind.eq_ignore_ascii_case("PASS") || kind.eq_ignore_ascii_case("PASS_PRIORITY")
}

fn is_mana_ability_action(kind: &str, description: &str) -> bool {
    let upper_kind = kind.to_ascii_uppercase();
    if upper_kind.contains("MANA") {
        return true;
    }
    let lower = description.to_ascii_lowercase();
    let looks_like_activation = upper_kind == "ACTIVATE"
        || upper_kind == "ACTIVATED"
        || upper_kind == "ACTIVATE_ABILITY"
        || upper_kind.contains("ACTIVAT");
    looks_like_activation
        && (lower.contains("add {")
            || lower.contains(" adds {")
            || lower.contains("add one mana")
            || lower.contains("adds one mana")
            || lower.contains("add mana")
            || lower.contains("adds mana")
            || lower.contains(": add "))
}

fn has_policy_relevant_non_pass_action(snapshot: &Value) -> bool {
    snapshot
        .get("playableActions")
        .and_then(Value::as_array)
        .map(|actions| {
            actions.iter().any(|action| {
                let kind = action
                    .get("type")
                    .and_then(Value::as_str)
                    .or_else(|| action.get("kind").and_then(Value::as_str))
                    .unwrap_or("");
                let description = action
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                !is_pass_action(kind) && !is_mana_ability_action(kind, description)
            })
        })
        .unwrap_or(false)
}

fn contains_ascii_case_insensitive(haystack: &str, needle: &str) -> bool {
    if haystack.contains(needle) {
        return true;
    }
    let needle = needle.as_bytes();
    if needle.is_empty() {
        return true;
    }
    haystack
        .as_bytes()
        .windows(needle.len())
        .any(|window| window.eq_ignore_ascii_case(needle))
}

fn choose_label(desc: &str) -> Option<Value> {
    if !contains_ascii_case_insensitive(desc, "choose") {
        return None;
    }
    let actor = PLAYER_PREFIX_RE
        .captures(desc)
        .map(|c| c["player"].to_string())
        .unwrap_or_default();
    Some(json!({"raw": desc, "actor_name": actor}))
}

fn may_label(rows: &[&Row], idx: usize) -> Option<Value> {
    let row = rows[idx];
    let desc = row.description();
    let row_type = row.row_type();
    if row_type != "STACK_RESOLVE" || !contains_ascii_case_insensitive(desc, "you may") {
        return None;
    }
    let actor = PLAYER_PREFIX_RE
        .captures(desc)
        .map(|c| c["player"].to_string())
        .unwrap_or_default();
    let start = idx.saturating_sub(4);
    let mut effect_logs: Vec<String> = Vec::new();
    for r in &rows[start..=idx] {
        if r.row_type() == "LOG" {
            let d = r.description();
            if !d.is_empty() {
                effect_logs.push(d.to_string());
            }
        }
    }
    let accepted = !effect_logs.is_empty();
    Some(json!({
        "raw": desc,
        "actor_name": actor,
        "accepted": accepted,
        "effect_logs": effect_logs,
    }))
}

fn may_source_row<'a>(rows: &[&'a Row], idx: usize) -> &'a Row {
    let desc = rows[idx].description();
    for j in (0..idx).rev() {
        let r = rows[j];
        let t = r.row_type();
        if t == "PRIORITY" {
            return r;
        }
        if t == "STACK_PUSH" && r.description() == desc {
            return r;
        }
    }
    rows[idx.saturating_sub(1)]
}

// --- candidate model --------------------------------------------------------

#[derive(Clone)]
struct Candidate {
    kind: &'static str,
    source_seq: i64,
    target_seq: i64,
    perspective_id: String,
    perspective_name: String,
    snapshot: Value,
    observed: Value,
    candidate_index: usize,
    candidate_count: usize,
}

fn make_candidate(
    kind: &'static str,
    source: &Row,
    target: &Row,
    perspective_id: &str,
    perspective_name: &str,
    observed: Value,
    snapshot_cache: &mut NormalizedSnapshotCache,
) -> Option<Candidate> {
    let raw_snapshot = source.snapshot()?;
    let source_seq = source.seq();
    let target_seq = target.seq();
    let mut normalized = snapshot_cache
        .entry((source_seq, perspective_id.to_string()))
        .or_insert_with(|| normalize_snapshot(raw_snapshot, perspective_id))
        .clone();
    let display_ids = display_ids_from_observed(raw_snapshot, perspective_id, kind, &observed);

    match kind {
        "priority" => {
            if let Some(pending) = priority_pending(raw_snapshot, perspective_id) {
                normalized
                    .as_object_mut()
                    .unwrap()
                    .insert(&"pending", pending);
            }
        }
        "attack" => {
            let attacker_prefixes: Vec<String> = observed
                .get("attackers")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|a| {
                            a.get("id_prefix").and_then(Value::as_str).map(String::from)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let no_attack = observed
                .get("no_attack")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if attacker_prefixes.is_empty() && !no_attack {
                return None;
            }
            let pending = attackers_pending(raw_snapshot, perspective_id, &display_ids)?;
            let opt_ids: Vec<String> = pending
                .get("options")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .map(|o| {
                            o.get(&"permanent_id")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string()
                        })
                        .collect()
                })
                .unwrap_or_default();
            for prefix in &attacker_prefixes {
                if !opt_ids.iter().any(|oid| oid.starts_with(prefix.as_str())) {
                    return None;
                }
            }
            normalized
                .as_object_mut()
                .unwrap()
                .insert(&"pending", pending);
        }
        "block" => {
            let attacker_refs: Vec<(String, String)> = observed
                .get("attackers")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|a| {
                            let prefix = a.get("id_prefix").and_then(Value::as_str)?;
                            let name = a.get("name").and_then(Value::as_str).unwrap_or("");
                            Some((prefix.to_string(), name.to_string()))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let pending =
                blockers_pending(raw_snapshot, perspective_id, &attacker_refs, &display_ids)?;
            let opt_ids: Vec<String> = pending
                .get("options")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .map(|o| {
                            o.get(&"permanent_id")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string()
                        })
                        .collect()
                })
                .unwrap_or_default();
            if let Some(arr) = observed.get("assignments").and_then(Value::as_array) {
                for assign in arr {
                    let blk_prefix = assign
                        .get("blocker_id_prefix")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if !opt_ids.iter().any(|oid| oid.starts_with(blk_prefix)) {
                        return None;
                    }
                }
            }
            normalized
                .as_object_mut()
                .unwrap()
                .insert(&"pending", pending);
        }
        _ => {}
    }

    Some(Candidate {
        kind,
        source_seq,
        target_seq,
        perspective_id: perspective_id.to_string(),
        perspective_name: perspective_name.to_string(),
        snapshot: normalized,
        observed,
        candidate_index: 0,
        candidate_count: 0,
    })
}

// --- main extraction --------------------------------------------------------

fn event_rows(rows: &[Row]) -> Vec<&Row> {
    rows.iter()
        .filter(|r| r.record() == "EVENT" && r.has_object_snapshot())
        .collect()
}

fn attack_observed(events: &[&Row], header_idx: usize, header: &AttacksHeader) -> Value {
    let mut attackers: Vec<Value> = Vec::new();
    for row in &events[header_idx + 1..] {
        let snap = row.snapshot_or_null();
        let step = snap.get("step").and_then(Value::as_str).unwrap_or("");
        if step != "DECLARE_ATTACKERS" && step != "DECLARE_BLOCKERS" {
            break;
        }
        if row.row_type() != "LOG" {
            continue;
        }
        let desc = row.description();
        if let Some(stripped) = desc.strip_prefix("Attacker:") {
            let _ = stripped;
            if let Some(parsed) = parse_attacker_line(desc) {
                attackers.push(json!({"name": parsed.name, "id_prefix": parsed.id_prefix}));
                if attackers.len() >= header.count {
                    break;
                }
            }
        }
    }
    json!({
        "raw": format!("{} attacks {} with {} creatures", header.attacker_player, header.defender_player, header.count),
        "actor_name": header.attacker_player,
        "defender_name": header.defender_player,
        "attackers": attackers,
    })
}

fn block_observed(events: &[&Row], header_idx: usize, header: &AttacksHeader) -> Option<Value> {
    let mut assignments: Vec<Value> = Vec::new();
    let mut seen_attackers: Vec<Value> = Vec::new();
    for row in &events[header_idx + 1..] {
        let snap = row.snapshot_or_null();
        let step = snap.get("step").and_then(Value::as_str).unwrap_or("");
        if step != "DECLARE_ATTACKERS" && step != "DECLARE_BLOCKERS" {
            break;
        }
        if row.row_type() != "LOG" {
            continue;
        }
        let desc = row.description();
        if !desc.starts_with("Attacker:") {
            continue;
        }
        let Some(parsed) = parse_attacker_line(desc) else {
            continue;
        };
        seen_attackers.push(json!({"name": parsed.name, "id_prefix": parsed.id_prefix}));
        for (bname, bid) in &parsed.blockers {
            assignments.push(json!({
                "attacker_name": parsed.name,
                "attacker_id_prefix": parsed.id_prefix,
                "blocker_name": bname,
                "blocker_id_prefix": bid,
            }));
        }
        if seen_attackers.len() >= header.count {
            break;
        }
    }
    if assignments.is_empty() {
        return None;
    }
    Some(json!({
        "raw": format!("{} attacks {} with {} creatures (blockers declared)", header.attacker_player, header.defender_player, header.count),
        "actor_name": header.defender_player,
        "attacker_player": header.attacker_player,
        "attackers": seen_attackers,
        "assignments": assignments,
    }))
}

fn extract_candidates(rows: &[Row], enabled: &[String], trajectory: bool) -> Vec<Candidate> {
    let events = event_rows(rows);
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut snapshot_cache = NormalizedSnapshotCache::new();
    let mut last_priority_by_player: HashMap<&str, (usize, &Row)> = HashMap::new();
    let mut consumed_priority_sources: HashSet<(i64, &str)> = HashSet::new();
    let want_priority = enabled.iter().any(|k| k == "priority");
    let want_attack = enabled.iter().any(|k| k == "attack");
    let want_block = enabled.iter().any(|k| k == "block");
    let want_may = enabled.iter().any(|k| k == "may");
    let want_choose = enabled.iter().any(|k| k == "choose");

    for (idx, row) in events.iter().enumerate() {
        let snap = row.snapshot_or_null();
        let row_type = row.row_type();
        let desc = row.description();
        let priority_id = snap
            .get("priorityPlayerId")
            .and_then(Value::as_str)
            .unwrap_or("");
        if row_type == "PRIORITY" && !priority_id.is_empty() {
            last_priority_by_player.insert(priority_id, (idx, row));
        }

        if want_priority {
            if let Some(label) = priority_label(desc, row_type) {
                let actor_id = priority_id;
                if let Some((_, source)) = last_priority_by_player.get(&actor_id).copied() {
                    let source_key = (source.seq(), actor_id);
                    if !consumed_priority_sources.contains(&source_key) {
                        if let Some(c) = make_candidate(
                            "priority",
                            source,
                            row,
                            actor_id,
                            raw_player_name_by_id(snap, actor_id),
                            label,
                            &mut snapshot_cache,
                        ) {
                            consumed_priority_sources.insert(source_key);
                            candidates.push(c);
                        }
                    }
                }
            }
        }

        if row_type == "LOG" {
            if let Some(header) = parse_attacks_header(desc) {
                if want_attack {
                    let observed = attack_observed(&events, idx, &header);
                    if !observed
                        .get("attackers")
                        .and_then(Value::as_array)
                        .map(|a| a.is_empty())
                        .unwrap_or(true)
                    {
                        let actor_id = raw_player_id_by_name(snap, &header.attacker_player);
                        let perspective = if !actor_id.is_empty() {
                            actor_id
                        } else {
                            snap.get("activePlayerId")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                        };
                        if let Some(c) = make_candidate(
                            "attack",
                            row,
                            row,
                            perspective,
                            &header.attacker_player,
                            observed,
                            &mut snapshot_cache,
                        ) {
                            candidates.push(c);
                        }
                    }
                }
                if want_block {
                    if let Some(observed) = block_observed(&events, idx, &header) {
                        let defender = header.defender_player.clone();
                        let mut actor_id = raw_player_id_by_name(snap, &defender);
                        if actor_id.is_empty() {
                            let active = snap
                                .get("activePlayerId")
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            actor_id = raw_opponent_id(snap, active);
                        }
                        // Find first Attacker: row to use its snapshot as source.
                        let mut block_source: &Row = row;
                        for jrow in &events[idx + 1..] {
                            if jrow.row_type() != "LOG" {
                                continue;
                            }
                            let jdesc = jrow.description();
                            if jdesc.starts_with("Attacker:") {
                                block_source = jrow;
                                break;
                            }
                        }
                        if let Some(c) = make_candidate(
                            "block",
                            block_source,
                            block_source,
                            actor_id,
                            &defender,
                            observed,
                            &mut snapshot_cache,
                        ) {
                            candidates.push(c);
                        }
                    }
                }
            }
        }

        if want_may {
            if let Some(label) = may_label(&events, idx) {
                let actor_name = label
                    .get("actor_name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let actor_id = raw_player_id_by_name(snap, &actor_name);
                let perspective = if actor_id.is_empty() {
                    priority_id
                } else {
                    actor_id
                };
                let source = may_source_row(&events, idx);
                if let Some(c) = make_candidate(
                    "may",
                    source,
                    row,
                    perspective,
                    &actor_name,
                    label,
                    &mut snapshot_cache,
                ) {
                    candidates.push(c);
                }
            }
        }

        if want_choose && row_type == "LOG" && contains_ascii_case_insensitive(desc, "choose") {
            if let Some(label) = choose_label(desc) {
                let actor_name = label
                    .get("actor_name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let actor_id = raw_player_id_by_name(snap, &actor_name);
                let perspective = if actor_id.is_empty() {
                    priority_id
                } else {
                    actor_id
                };
                if let Some(c) = make_candidate(
                    "choose",
                    row,
                    row,
                    perspective,
                    &actor_name,
                    label,
                    &mut snapshot_cache,
                ) {
                    candidates.push(c);
                }
            }
        }
    }

    if want_attack || want_block {
        synthesize_combat_passes(
            &events,
            want_attack,
            want_block,
            &mut snapshot_cache,
            &mut candidates,
        );
    }
    if trajectory && want_priority {
        infer_priority_passes(
            &events,
            &consumed_priority_sources,
            &mut snapshot_cache,
            &mut candidates,
        );
    }

    candidates.sort_by_key(|c| (c.source_seq, c.target_seq, kind_rank(c.kind)));
    let total = candidates.len();
    for (i, c) in candidates.iter_mut().enumerate() {
        c.candidate_index = i;
        c.candidate_count = total;
    }
    candidates
}

fn kind_rank(kind: &str) -> u8 {
    match kind {
        "priority" => 0,
        "attack" => 1,
        "block" => 2,
        "may" => 3,
        "choose" => 4,
        _ => 5,
    }
}

fn infer_priority_passes(
    events: &[&Row],
    consumed_priority_sources: &HashSet<(i64, &str)>,
    snapshot_cache: &mut NormalizedSnapshotCache,
    out: &mut Vec<Candidate>,
) {
    for (idx, row) in events.iter().enumerate() {
        if row.row_type() != "PRIORITY" {
            continue;
        }
        let Some(snap) = row.snapshot() else {
            continue;
        };
        if !has_policy_relevant_non_pass_action(snap) {
            continue;
        }
        let actor_id = snap
            .get("priorityPlayerId")
            .and_then(Value::as_str)
            .unwrap_or("");
        if actor_id.is_empty() {
            continue;
        }
        let seq = row.seq();
        if consumed_priority_sources.contains(&(seq, actor_id)) {
            continue;
        }
        let target = events.get(idx + 1).copied().unwrap_or(row);
        let observed = json!({
            "raw": "Pass priority",
            "event_type": "PASS",
            "is_inferred": true,
        });
        if let Some(c) = make_candidate(
            "priority",
            row,
            target,
            actor_id,
            &raw_player_name_by_id(snap, actor_id),
            observed,
            snapshot_cache,
        ) {
            out.push(c);
        }
    }
}

/// Walk DECLARE_ATTACKERS / DECLARE_BLOCKERS step windows. Newer Forge logs
/// write combat choices as "assigned X to attack/block" lines rather than the
/// older "Attacker:" summaries, so extract those here. For windows with no
/// positive action, synthesize an empty-attack / empty-block candidate so the
/// policy sees "chose not to act" decisions.
fn synthesize_combat_passes(
    events: &[&Row],
    want_attack: bool,
    want_block: bool,
    snapshot_cache: &mut NormalizedSnapshotCache,
    out: &mut Vec<Candidate>,
) {
    let mut i = 0;
    while i < events.len() {
        let step = events[i].snapshot_step().to_string();
        if step != "DECLARE_ATTACKERS" && step != "DECLARE_BLOCKERS" {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i;
        while end < events.len() {
            let s = events[end].snapshot_step();
            if s != step {
                break;
            }
            end += 1;
        }
        let window = &events[start..end];

        if step == "DECLARE_ATTACKERS" && want_attack {
            let any_header = window.iter().any(|r| {
                r.row_type() == "LOG"
                    && parse_attacks_header(r.description())
                        .map(|h| h.count > 0)
                        .unwrap_or(false)
            });
            if !any_header {
                let source = window[0];
                if let Some(observed) = assigned_attack_observed(window) {
                    if let Some(snap) = source.snapshot() {
                        let actor_name = observed
                            .get("actor_name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let mut actor_id = raw_player_id_by_name(snap, &actor_name);
                        if actor_id.is_empty() {
                            actor_id = snap
                                .get("activePlayerId")
                                .and_then(Value::as_str)
                                .unwrap_or("");
                        }
                        let target = window
                            .iter()
                            .find(|r| has_assigned_attack_log(r))
                            .copied()
                            .unwrap_or(source);
                        if !actor_id.is_empty() {
                            if let Some(c) = make_candidate(
                                "attack",
                                source,
                                target,
                                actor_id,
                                &actor_name,
                                observed,
                                snapshot_cache,
                            ) {
                                out.push(c);
                            }
                        }
                    }
                    i = end;
                    continue;
                }
                if let Some(snap) = source.snapshot() {
                    let active = snap
                        .get("activePlayerId")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if !active.is_empty() {
                        let actor_name = raw_player_name_by_id(snap, active);
                        let observed = json!({
                            "raw": format!("{} declares no attackers", actor_name),
                            "actor_name": actor_name,
                            "attackers": [],
                            "no_attack": true,
                        });
                        if let Some(c) = make_candidate(
                            "attack",
                            source,
                            source,
                            active,
                            actor_name,
                            observed,
                            snapshot_cache,
                        ) {
                            out.push(c);
                        }
                    }
                }
            }
        } else if step == "DECLARE_BLOCKERS" && want_block {
            let any_attacker_line = window
                .iter()
                .any(|r| r.row_type() == "LOG" && r.description().starts_with("Attacker:"));
            if !any_attacker_line {
                if let Some(observed) = assigned_block_observed(window) {
                    let source = window[0];
                    if let Some(snap) = source.snapshot() {
                        let actor_name = observed
                            .get("actor_name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let mut actor_id = raw_player_id_by_name(snap, &actor_name);
                        if actor_id.is_empty() {
                            let active = snap
                                .get("activePlayerId")
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            actor_id = raw_opponent_id(snap, active);
                        }
                        let target = window
                            .iter()
                            .find(|r| has_new_block_log(r))
                            .copied()
                            .unwrap_or(source);
                        if !actor_id.is_empty() {
                            if let Some(c) = make_candidate(
                                "block",
                                source,
                                target,
                                actor_id,
                                &actor_name,
                                observed,
                                snapshot_cache,
                            ) {
                                out.push(c);
                            }
                        }
                    }
                    i = end;
                    continue;
                }
            }
            let any_block = window.iter().any(|r| {
                if r.row_type() != "LOG" {
                    return false;
                }
                let d = r.description();
                d.starts_with("Attacker:")
                    && parse_attacker_line(d)
                        .map(|p| !p.blockers.is_empty())
                        .unwrap_or(false)
            });
            if !any_block {
                let mut attacker_records: Vec<Value> = Vec::new();
                for r in window {
                    let d = r.description();
                    if !d.starts_with("Attacker:") {
                        continue;
                    }
                    if let Some(p) = parse_attacker_line(d) {
                        attacker_records.push(json!({"name": p.name, "id_prefix": p.id_prefix}));
                    }
                }
                if !attacker_records.is_empty() {
                    let source = window
                        .iter()
                        .find(|r| r.description().starts_with("Attacker:"))
                        .copied()
                        .unwrap_or(window[0]);
                    if let Some(snap) = source.snapshot() {
                        let active = snap
                            .get("activePlayerId")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let defender = raw_opponent_id(snap, active);
                        if !defender.is_empty() {
                            let defender_name = raw_player_name_by_id(snap, defender);
                            let observed = json!({
                                "raw": "no blocks declared",
                                "actor_name": defender_name,
                                "attackers": attacker_records,
                                "assignments": [],
                                "no_block": true,
                            });
                            if let Some(c) = make_candidate(
                                "block",
                                source,
                                source,
                                defender,
                                defender_name,
                                observed,
                                snapshot_cache,
                            ) {
                                out.push(c);
                            }
                        }
                    }
                }
            }
        }
        i = end;
    }
}

// --- selection --------------------------------------------------------------

fn blake2_8(bytes: &[u8]) -> u64 {
    let mut hasher = Blake2bVar::new(8).unwrap();
    hasher.update(bytes);
    let mut out = [0u8; 8];
    hasher.finalize_variable(&mut out).unwrap();
    u64::from_be_bytes(out)
}

fn stable_choices(mut candidates: Vec<Candidate>, count: usize, game_id: &str) -> Vec<Candidate> {
    if count == 0 {
        return Vec::new();
    }
    if count >= candidates.len() {
        return candidates;
    }
    let mut keyed: Vec<(u64, Candidate)> = candidates
        .drain(..)
        .map(|c| {
            let key = format!("{}:{}", game_id, c.candidate_index);
            (blake2_8(key.as_bytes()), c)
        })
        .collect();
    keyed.sort_by_key(|(k, _)| *k);
    keyed.into_iter().take(count).map(|(_, c)| c).collect()
}

fn priority_choices(
    candidates: Vec<Candidate>,
    kind_priority: &[String],
    count: usize,
) -> Vec<Candidate> {
    if count == 0 {
        return Vec::new();
    }
    let mut by_kind: Vec<Vec<Candidate>> = (0..CHOICE_KINDS.len()).map(|_| Vec::new()).collect();
    for c in candidates.into_iter() {
        if let Some(idx) = choice_kind_index(c.kind) {
            by_kind[idx].push(c);
        }
    }
    let mut selected: Vec<Candidate> = Vec::new();
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for kind in kind_priority {
        if let Some(kind_idx) = choice_kind_index(kind) {
            let bucket = &mut by_kind[kind_idx];
            for c in bucket.drain(..) {
                if seen.insert(c.candidate_index) {
                    selected.push(c);
                    if selected.len() >= count {
                        return selected;
                    }
                }
            }
        }
    }
    for bucket in by_kind {
        for c in bucket {
            if seen.insert(c.candidate_index) {
                selected.push(c);
                if selected.len() >= count {
                    return selected;
                }
            }
        }
    }
    selected
}

// --- record building --------------------------------------------------------

const EMPTY_PLAYERS: &[Value] = &[];

static EMPTY_EXTRAS: Lazy<Value> = Lazy::new(|| json!({}));
static NULL_VALUE: Lazy<Value> = Lazy::new(|| json!(null));

fn terminal_sign(winner_id: Option<&str>, perspective_id: &str) -> f64 {
    match winner_id {
        Some(w) if !w.is_empty() && !perspective_id.is_empty() => {
            if w == perspective_id {
                1.0
            } else {
                -1.0
            }
        }
        _ => 0.0,
    }
}

#[derive(Serialize)]
struct OutputRecord<'a> {
    format: &'static str,
    format_version: u32,
    game_id: &'a str,
    archive_member: &'a str,
    choice: OutputChoice<'a>,
    state: OutputState<'a>,
    outcome: OutputOutcome<'a>,
}

#[derive(Serialize)]
struct OutputChoice<'a> {
    kind: &'static str,
    candidate_index: usize,
    candidate_count: usize,
    source_seq: i64,
    target_seq: i64,
    perspective_id: &'a str,
    perspective_name: &'a str,
    observed: &'a Value,
}

#[derive(Serialize)]
struct OutputState<'a> {
    snapshot: &'a Value,
}

#[derive(Serialize)]
struct OutputOutcome<'a> {
    winner_id: Option<&'a str>,
    winner_name: Option<&'a str>,
    terminal_sign: f64,
    players: &'a [Value],
    extras: &'a Value,
}

fn build_record<'a>(
    c: &'a Candidate,
    meta: &'a Meta<'a>,
    archive_member: &'a str,
) -> OutputRecord<'a> {
    OutputRecord {
        format: "forge_choice_situation",
        format_version: FORMAT_VERSION,
        game_id: meta.game_id,
        archive_member,
        choice: OutputChoice {
            kind: c.kind,
            candidate_index: c.candidate_index,
            candidate_count: c.candidate_count,
            source_seq: c.source_seq,
            target_seq: c.target_seq,
            perspective_id: &c.perspective_id,
            perspective_name: &c.perspective_name,
            observed: &c.observed,
        },
        state: OutputState {
            snapshot: &c.snapshot,
        },
        outcome: OutputOutcome {
            winner_id: meta.winner_id,
            winner_name: meta.winner_name,
            terminal_sign: terminal_sign(meta.winner_id, &c.perspective_id),
            players: meta.players.unwrap_or(EMPTY_PLAYERS),
            extras: meta.extras.unwrap_or(&EMPTY_EXTRAS),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event_row(row_type: &str, seq: i64, description: Option<&str>, snapshot: &Value) -> Row {
        Row {
            record: Some("EVENT".to_string()),
            row_type: Some(row_type.to_string()),
            seq: Some(seq),
            description: description.map(str::to_string),
            snapshot: Some(snapshot.clone()),
            game_id: None,
            winner_id: None,
            winner_name: None,
            players: None,
            extras: None,
        }
    }

    fn seq_table(rows: Vec<Vec<i32>>) -> SequenceTable {
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let mut tokens = Vec::new();
        offsets.push(0);
        for row in rows {
            tokens.extend(row);
            offsets.push(tokens.len());
        }
        SequenceTable { offsets, tokens }
    }

    fn fake_token_bundle() -> TokenTableBundle {
        let mut structural = vec![vec![]; 12];
        structural[FRAG_BOS_STATE] = vec![1, 2];
        structural[FRAG_CLOSE_STATE_EOS] = vec![3, 4];
        structural[FRAG_CLOSE_SELF] = vec![5];
        structural[FRAG_CLOSE_OPP] = vec![6];
        structural[5] = vec![70];
        structural[6] = vec![71];
        structural[FRAG_SELF_MANA] = vec![72];
        structural[FRAG_OPP_MANA] = vec![73];

        let zone_rows = vec![vec![80]; 10];
        let close_rows = vec![vec![81]; 10];
        let count_rows = (0..=200).map(|i| vec![2000 + i]).collect::<Vec<_>>();
        let mut name_to_row = HashMap::new();
        name_to_row.insert("Overflow Card".to_string(), 2);

        TokenTableBundle {
            unk_token_id: Some(999),
            name_to_row,
            decision_spec: DecisionSpecTables {
                spec_open_id: 31,
                spec_close_id: 32,
                decision_type_id: 33,
                legal_attacker_id: 34,
                legal_blocker_id: 35,
                legal_target_id: 36,
                legal_action_id: 37,
                max_value_open_id: 38,
                max_value_close_id: 39,
                player_ref_ids: vec![40, 41],
                decision_type_name_ids: vec![42, 43, 44, 45, 46, 47, 48],
                max_value_digits: seq_table((0..=10).map(|i| vec![60 + i]).collect()),
            },
            tables: RustTokenTables {
                dict_open_id: 7,
                dict_close_id: 8,
                card_open_id: 9,
                stack_open_id: 10,
                stack_close_id: 11,
                turn_min: 0,
                turn_max: 200,
                life_min: -30,
                life_max: 300,
                count_min: 0,
                count_max: 200,
                step_count: 13,
                owner_count: 2,
                structural: seq_table(structural),
                zone_open: seq_table(zone_rows),
                zone_close: seq_table(close_rows),
                mana_glyph: seq_table(vec![vec![100]; 6]),
                turn_step: seq_table(vec![vec![300]; 201 * 13]),
                life_owner: seq_table(vec![vec![400]; 331 * 2]),
                count: seq_table(count_rows),
                card_closer: vec![12],
                status_tapped: vec![13],
                status_untapped: vec![14],
                card_ref: (0..MAX_CARD_REFS).map(|i| 5000 + i as i32).collect(),
                dict_entry: vec![900, 901, 999],
                card_body: seq_table(vec![vec![], vec![1001], vec![1002]]),
                row_to_name: vec![
                    "<unknown>".to_string(),
                    "Valid Card".to_string(),
                    "Overflow Card".to_string(),
                ],
            },
        }
    }

    #[test]
    fn direct_emitter_omits_actions_and_uses_sequence_local_dict_slots() {
        let bundle = fake_token_bundle();
        let snapshot = json!({
            "turn": 1,
            "step": "Precombat Main",
            "players": [
                {
                    "ID": "p1",
                    "Life": 20,
                    "LibraryCount": 40,
                    "ManaPool": {},
                    "Hand": [{"ID": "h1", "Name": "Overflow Card"}],
                    "Battlefield": [],
                    "Graveyard": [],
                    "Exile": []
                },
                {
                    "ID": "p2",
                    "Life": 20,
                    "LibraryCount": 40,
                    "ManaPool": {},
                    "Hand": [],
                    "Battlefield": [],
                    "Graveyard": [],
                    "Exile": []
                }
            ],
            "pending": {
                "kind": "priority",
                "options": [{"kind": "play", "card_name": "Overflow Card"}]
            },
            "stack": []
        });

        let out = emit_tokenized_state(&snapshot, &bundle).unwrap();

        assert!(!out.token_ids.contains(&999));
        assert!(out.token_ids.contains(&900));
        assert!(out.token_ids.contains(&1002));
        assert!(!out.token_ids.contains(&70));
        assert!(!out.token_ids.contains(&71));
    }

    #[test]
    fn decision_spec_dedupes_priority_actions_and_skips_mana_abilities() {
        let bundle = fake_token_bundle();
        let snapshot = json!({
            "pending": {
                "kind": "priority",
                "options": [
                    {"kind": "play", "card_name": "Forest", "label": "Play Forest"},
                    {"kind": "play", "card_name": "Forest", "label": "Play Forest"},
                    {"kind": "activate", "card_name": "Forest", "label": "Forest - PlayerA adds {G}."},
                    {"kind": "activate", "card_name": "Jayemdae Tome", "label": "Jayemdae Tome - Draw a card."},
                    {"kind": "pass", "label": "Pass priority"}
                ]
            }
        });

        let spec = emit_decision_spec(&snapshot, &bundle).unwrap();

        assert_eq!(spec.decision_type, DECISION_PRIORITY);
        assert_eq!(spec.token_ids, vec![31, 33, 42, 37, 37, 37, 32]);
        assert_eq!(spec.anchor_kinds, vec![ANCHOR_LEGAL_ACTION; 3]);
        assert_eq!(spec.anchor_subjects, vec![0, 1, 2]);
        assert_eq!(spec.anchor_handles, vec![0, 3, 4]);
    }

    #[test]
    fn parses_assigned_attack_list() {
        let (_, defender, attackers) = parse_assigned_attack(
            "PlayerB assigned Dancing Scimitar (93), Zephyr Falcon (73) and Zephyr Falcon (71) to attack PlayerA.",
        )
        .unwrap();

        assert_eq!(defender, "PlayerA");
        assert_eq!(
            attackers
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Dancing Scimitar", "Zephyr Falcon", "Zephyr Falcon"]
        );
        assert_eq!(
            attackers
                .iter()
                .map(|c| c.id_prefix.as_str())
                .collect::<Vec<_>>(),
            vec!["forge:93", "forge:73", "forge:71"]
        );
    }

    #[test]
    fn parses_assigned_block_and_no_block_lines() {
        let (_, blockers, attacker) = parse_assigned_block(
            "PlayerB assigned Carrion Ants (121) to block Ironroot Treefolk (53).",
        )
        .unwrap();
        assert_eq!(blockers[0].name, "Carrion Ants");
        assert_eq!(blockers[0].id_prefix, "forge:121");
        assert_eq!(attacker.name, "Ironroot Treefolk");
        assert_eq!(attacker.id_prefix, "forge:53");

        let (_, unblocked) = parse_didnt_block("PlayerB didn't block Giant Spider (56).").unwrap();
        assert_eq!(unblocked.name, "Giant Spider");
        assert_eq!(unblocked.id_prefix, "forge:56");
    }

    #[test]
    fn zero_attack_header_emits_no_attack_candidate() {
        let snapshot = json!({
            "activePlayerId": "p2",
            "priorityPlayerId": "p2",
            "players": [
                {"id": "p1", "name": "PlayerA", "life": 20, "librarySize": 40},
                {"id": "p2", "name": "PlayerB", "life": 20, "librarySize": 40}
            ],
            "battlefield": [
                {"id": "p2:bf:0:Grizzly Bears", "controllerId": "p2", "name": "Grizzly Bears", "power": 2}
            ],
            "step": "DECLARE_ATTACKERS"
        });
        let rows = vec![event_row(
            "LOG",
            21,
            Some("PlayerB attacks PlayerA with 0 creatures"),
            &snapshot,
        )];

        let candidates = extract_candidates(&rows, &["attack".to_string()], true);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].kind, "attack");
        assert_eq!(
            candidates[0]
                .observed
                .get("no_attack")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            candidates[0]
                .observed
                .get("attackers")
                .and_then(Value::as_array)
                .map(|arr| arr.len()),
            Some(0)
        );
    }

    #[test]
    fn priority_pass_inference_requires_policy_relevant_action() {
        let pass_only = json!({
            "playableActions": [
                {"type": "PASS", "description": "Pass priority"}
            ]
        });
        let mana_only = json!({
            "playableActions": [
                {
                    "type": "ACTIVATE_ABILITY",
                    "sourceName": "Forest",
                    "description": "{T}: Add {G}."
                },
                {"type": "PASS", "description": "Pass priority"}
            ]
        });
        let actionable = json!({
            "playableActions": [
                {
                    "type": "ACTIVATED",
                    "sourceName": "Prodigal Sorcerer",
                    "description": "{T}: Deal 1 damage to any target."
                },
                {"type": "PASS", "description": "Pass priority"}
            ]
        });

        assert!(!has_policy_relevant_non_pass_action(&pass_only));
        assert!(!has_policy_relevant_non_pass_action(&mana_only));
        assert!(has_policy_relevant_non_pass_action(&actionable));
    }

    #[test]
    fn trajectory_uses_priority_source_once() {
        let snapshot = json!({
            "activePlayerId": "p2",
            "priorityPlayerId": "p2",
            "players": [
                {"id": "p1", "name": "PlayerA", "life": 20, "librarySize": 40},
                {"id": "p2", "name": "PlayerB", "life": 20, "librarySize": 40}
            ],
            "playableActions": [
                {"type": "LAND", "sourceName": "Island", "description": "Play Island"},
                {"type": "SPELL", "sourceName": "Ornithopter", "description": "Cast Ornithopter"},
                {"type": "PASS", "description": "Pass priority"}
            ]
        });
        let rows = vec![
            event_row("PRIORITY", 11, None, &snapshot),
            event_row("LOG", 12, Some("PlayerB played Island (97)"), &snapshot),
            event_row(
                "STACK_PUSH",
                13,
                Some("STACK_PUSH Ornithopter by PlayerB"),
                &snapshot,
            ),
        ];

        let candidates = extract_candidates(&rows, &["priority".to_string()], true);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].source_seq, 11);
        assert_eq!(candidates[0].target_seq, 12);
        assert_eq!(
            candidates[0].observed.get("raw").and_then(Value::as_str),
            Some("PlayerB played Island (97)")
        );
    }

    #[test]
    fn arrow_writer_round_trips_one_shard() {
        use arrow_array::Array;

        let out_dir = std::env::temp_dir().join(format!(
            "forge_extract_arrow_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let row = ArrowDecisionRow {
            format_version: FORMAT_VERSION as u16,
            game_id: "game-1".to_string(),
            archive_member: "game-1.jsonl.gz".to_string(),
            kind_id: 0,
            candidate_index: 1,
            candidate_count: 2,
            source_seq: 10,
            target_seq: 11,
            perspective_id: "p1".to_string(),
            perspective_name: "PlayerA".to_string(),
            winner_id: Some("p1".to_string()),
            winner_name: Some("PlayerA".to_string()),
            terminal_sign: 1.0,
            snapshot_json: "{\"pending\":{\"kind\":\"priority\"}}".to_string(),
            observed_json: "{\"raw\":\"Pass priority\"}".to_string(),
            outcome_players_json: "[]".to_string(),
            outcome_extras_json: "{}".to_string(),
            token_ids: None,
            seq_length: None,
            card_ref_positions: None,
            token_overflow: false,
            card_ref_overflow: false,
            action_token_ids: None,
            action_seq_length: None,
            decision_type: None,
            pointer_anchor_positions: None,
            pointer_anchor_kinds: None,
            pointer_anchor_subjects: None,
            pointer_anchor_handles: None,
            legal_edge_bitmap: None,
            legal_edge_n_blockers: None,
            legal_edge_n_attackers: None,
        };

        let (tx, rx) = mpsc::sync_channel(1);
        tx.send(WriterBatch::ArrowRows(vec![row])).unwrap();
        drop(tx);

        let stats = write_arrow_dir(rx, out_dir.clone(), 1, false).unwrap();
        assert_eq!(stats.records, 1);
        assert_eq!(stats.shards, 1);

        let file = File::open(out_dir.join("part-000000.arrow")).unwrap();
        let mut reader = arrow_ipc::reader::FileReader::try_new(file, None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
        let game_id = batch
            .column_by_name("game_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let kind_id = batch
            .column_by_name("kind_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();
        assert_eq!(game_id.value(0), "game-1");
        assert_eq!(kind_id.value(0), 0);

        let file = File::open(out_dir.join("game_index.arrow")).unwrap();
        let mut reader = arrow_ipc::reader::FileReader::try_new(file, None).unwrap();
        let index_batch = reader.next().unwrap().unwrap();
        assert_eq!(index_batch.num_rows(), 1);
        let index_game_id = index_batch
            .column_by_name("game_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let row_count = index_batch
            .column_by_name("row_count")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(index_game_id.value(0), "game-1");
        assert_eq!(row_count.value(0), 1);

        std::fs::remove_dir_all(out_dir).ok();
    }

    #[test]
    fn pretokenized_arrow_writer_round_trips_token_columns() {
        use arrow_array::Array;
        use arrow_array::{Int32Array, ListArray};

        let out_dir = std::env::temp_dir().join(format!(
            "forge_extract_pretok_arrow_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut row = ArrowDecisionRow {
            format_version: FORMAT_VERSION as u16,
            game_id: "game-1".to_string(),
            archive_member: "game-1.jsonl.gz".to_string(),
            kind_id: 0,
            candidate_index: 0,
            candidate_count: 1,
            source_seq: 10,
            target_seq: 11,
            perspective_id: "p1".to_string(),
            perspective_name: "PlayerA".to_string(),
            winner_id: Some("p1".to_string()),
            winner_name: Some("PlayerA".to_string()),
            terminal_sign: 1.0,
            snapshot_json: "{\"pending\":{\"kind\":\"priority\"}}".to_string(),
            observed_json: "{\"raw\":\"Pass priority\"}".to_string(),
            outcome_players_json: "[]".to_string(),
            outcome_extras_json: "{}".to_string(),
            token_ids: None,
            seq_length: None,
            card_ref_positions: None,
            token_overflow: false,
            card_ref_overflow: false,
            action_token_ids: None,
            action_seq_length: None,
            decision_type: None,
            pointer_anchor_positions: None,
            pointer_anchor_kinds: None,
            pointer_anchor_subjects: None,
            pointer_anchor_handles: None,
            legal_edge_bitmap: None,
            legal_edge_n_blockers: None,
            legal_edge_n_attackers: None,
        };
        row.attach_tokenized_state(
            TokenizedState {
                token_ids: vec![11, 12, 13],
                card_ref_positions: {
                    let mut positions = vec![-1; MAX_CARD_REFS];
                    positions[4] = 2;
                    positions
                },
                card_ref_overflow: false,
            },
            Some(DecisionSpecState {
                token_ids: vec![31, 33, 42, 37, 32],
                decision_type: DECISION_PRIORITY,
                anchor_positions: vec![3],
                anchor_kinds: vec![ANCHOR_LEGAL_ACTION],
                anchor_subjects: vec![0],
                anchor_handles: vec![0],
                legal_edge_bitmap: Vec::new(),
                legal_edge_n_blockers: 0,
                legal_edge_n_attackers: 0,
            }),
        );

        let (tx, rx) = mpsc::sync_channel(1);
        tx.send(WriterBatch::ArrowRows(vec![row])).unwrap();
        drop(tx);

        let stats = write_arrow_dir(rx, out_dir.clone(), 1, true).unwrap();
        assert_eq!(stats.records, 1);
        let file = File::open(out_dir.join("part-000000.arrow")).unwrap();
        let mut reader = arrow_ipc::reader::FileReader::try_new(file, None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(
            batch.schema().metadata().get("stage").map(String::as_str),
            Some("pretokenized")
        );
        let seq_length = batch
            .column_by_name("seq_length")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(seq_length.value(0), 3);
        let token_ids = batch
            .column_by_name("token_ids")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let token_values = token_ids.value(0);
        let token_values = token_values.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(token_values.values(), &[11, 12, 13]);

        std::fs::remove_dir_all(out_dir).ok();
    }

    #[test]
    fn jsonl_gz_fixture_extracts_to_arrow_and_reloads() {
        let root = std::env::temp_dir().join(format!(
            "forge_extract_jsonl_to_arrow_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture_path = root.join("fixture.jsonl.gz");
        let arrow_dir = root.join("arrow");

        let snapshot = json!({
            "turn": 3,
            "step": "PRECOMBAT_MAIN",
            "activePlayerId": "p1",
            "priorityPlayerId": "p1",
            "players": [
                {
                    "id": "p1",
                    "name": "PlayerA",
                    "life": 20,
                    "handSize": 2,
                    "librarySize": 41,
                    "manaPool": "G"
                },
                {
                    "id": "p2",
                    "name": "PlayerB",
                    "life": 17,
                    "handSize": 3,
                    "librarySize": 39,
                    "manaPool": ""
                }
            ],
            "battlefield": [],
            "playableActions": [
                {
                    "type": "LAND",
                    "sourceId": "hand:forest",
                    "sourceName": "Forest",
                    "description": "Play Forest"
                },
                {"type": "PASS", "description": "Pass priority"}
            ]
        });
        let rows = [
            json!({
                "record": "META",
                "gameId": "game-fixture-1",
                "winnerId": "p1",
                "winnerName": "PlayerA",
                "players": [
                    {"id": "p1", "name": "PlayerA", "hasWon": true},
                    {"id": "p2", "name": "PlayerB", "hasLost": true}
                ],
                "extras": {"turns": 3}
            }),
            json!({
                "record": "EVENT",
                "type": "PRIORITY",
                "seq": 10,
                "snapshot": snapshot
            }),
            json!({
                "record": "EVENT",
                "type": "LOG",
                "seq": 11,
                "description": "PlayerA played Forest (42)",
                "snapshot": snapshot
            }),
        ];
        {
            let file = File::create(&fixture_path).unwrap();
            let mut gz = GzEncoder::new(file, Compression::new(1));
            for row in rows {
                gz.write_all(sonic_rs::to_string(&row).unwrap().as_bytes())
                    .unwrap();
                gz.write_all(b"\n").unwrap();
            }
            gz.finish().unwrap();
        }

        let parsed_rows = read_jsonl_member(File::open(&fixture_path).unwrap()).unwrap();
        assert_eq!(parsed_rows.len(), 3);
        let meta = meta_from_rows(&parsed_rows).unwrap();
        let candidates = extract_candidates(&parsed_rows, &["priority".to_string()], false);
        let selected = stable_choices(candidates, 1, meta.game_id);
        assert_eq!(selected.len(), 1);

        let arrow_rows = selected
            .iter()
            .map(|candidate| {
                let record = build_record(candidate, &meta, "fixture.jsonl.gz");
                ArrowDecisionRow::from_record(&record).unwrap()
            })
            .collect::<Vec<_>>();
        let (tx, rx) = mpsc::sync_channel(1);
        tx.send(WriterBatch::ArrowRows(arrow_rows)).unwrap();
        drop(tx);

        let stats = write_arrow_dir(rx, arrow_dir.clone(), 16, false).unwrap();
        write_arrow_manifest(
            &arrow_dir,
            &ArrowManifestStats {
                shards: stats.shards,
                records_written: stats.records,
                shard_target_rows: 16,
                games_seen: 1,
                games_written: 1,
                candidates_seen: 1,
                kind_counts: [1, 0, 0, 0, 0],
                pretokenized: false,
            },
        )
        .unwrap();
        assert_eq!(stats.records, 1);
        assert_eq!(stats.shards, 1);
        assert!(arrow_dir.join("manifest.json").exists());
        assert!(arrow_dir.join("game_index.arrow").exists());

        let file = File::open(arrow_dir.join("part-000000.arrow")).unwrap();
        let mut reader = arrow_ipc::reader::FileReader::try_new(file, None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);

        let game_id = batch
            .column_by_name("game_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let archive_member = batch
            .column_by_name("archive_member")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let kind_id = batch
            .column_by_name("kind_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();
        let candidate_index = batch
            .column_by_name("candidate_index")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let source_seq = batch
            .column_by_name("source_seq")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let target_seq = batch
            .column_by_name("target_seq")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let terminal_sign = batch
            .column_by_name("terminal_sign")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let observed_json = batch
            .column_by_name("observed_json")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();
        let snapshot_json = batch
            .column_by_name("snapshot_json")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();

        assert_eq!(game_id.value(0), "game-fixture-1");
        assert_eq!(archive_member.value(0), "fixture.jsonl.gz");
        assert_eq!(kind_id.value(0), 0);
        assert_eq!(candidate_index.value(0), 0);
        assert_eq!(source_seq.value(0), 10);
        assert_eq!(target_seq.value(0), 11);
        assert_eq!(terminal_sign.value(0), 1.0);

        let observed: Value = sonic_rs::from_str(observed_json.value(0)).unwrap();
        assert_eq!(
            observed.get("raw").and_then(Value::as_str),
            Some("PlayerA played Forest (42)")
        );
        let snapshot: Value = sonic_rs::from_str(snapshot_json.value(0)).unwrap();
        assert_eq!(
            snapshot
                .get("pending")
                .and_then(|p| p.get("kind"))
                .and_then(Value::as_str),
            Some("priority")
        );
        assert_eq!(
            snapshot
                .get("players")
                .and_then(Value::as_array)
                .and_then(|players| players.first())
                .and_then(|player| player.get("ID"))
                .and_then(Value::as_str),
            Some("p1")
        );

        std::fs::remove_dir_all(root).ok();
    }
}

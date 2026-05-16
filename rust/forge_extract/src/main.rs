//! Forge choice-situation extractor (Rust port of the parser portion of
//! `scripts/extract_forge_choice_situations.py`).
//!
//! Reads a Forge games zip (members are `*.jsonl.gz`), extracts choice
//! candidates (priority / attack / block / may / choose), normalizes the
//! pre-choice snapshot, and writes one JSONL record per selected situation.
//! A separate Python pass is responsible for tokenization + torch sharding.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use anyhow::{Context, Result, anyhow};
use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use clap::Parser;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use regex::Regex;
use serde_json::{json, Map, Value};
use zip::ZipArchive;

const FORMAT_VERSION: u32 = 2;
const MANA_COLORS: &[&str] = &["White", "Blue", "Black", "Red", "Green", "Colorless"];
const CHOICE_KINDS: &[&str] = &["priority", "attack", "block", "may", "choose"];
const DEFAULT_KIND_PRIORITY: &[&str] = &["may", "block", "attack", "choose", "priority"];

#[derive(Parser, Debug)]
#[command(about = "Extract Forge choice situations to JSONL (parser-only Rust port)")]
struct Args {
    /// Forge game archive (zip of *.jsonl.gz members).
    #[arg(long)]
    zip: PathBuf,

    /// Output .jsonl.gz path.
    #[arg(long)]
    out: PathBuf,

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

    /// Print progress every N games.
    #[arg(long, default_value_t = 1000)]
    progress_every: usize,

    /// gzip compression level for the output (0-9).
    #[arg(long, default_value_t = 6)]
    compresslevel: u32,
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

    // Enumerate members up-front (cheap; zip central directory only).
    let members: Vec<String> = {
        let f = File::open(&args.zip).with_context(|| format!("opening {:?}", args.zip))?;
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

    // Writer thread: serialize JSONL output and gzip-compress.
    let (tx, rx) = mpsc::sync_channel::<Vec<Vec<u8>>>(64);
    let out_path = args.out.clone();
    let level = args.compresslevel;
    let writer_handle = thread::spawn(move || -> Result<WriterStats> {
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let f = File::create(&out_path).with_context(|| format!("creating {:?}", out_path))?;
        let mut gz = GzEncoder::new(BufWriter::new(f), Compression::new(level));
        let mut records = 0u64;
        for batch in rx {
            for line in batch {
                gz.write_all(&line)?;
                gz.write_all(b"\n")?;
                records += 1;
            }
        }
        gz.try_finish()?;
        Ok(WriterStats { records })
    });

    let progress_every = args.progress_every;
    let zip_path = args.zip.clone();
    let games_seen = std::sync::atomic::AtomicUsize::new(0);
    let games_written = std::sync::atomic::AtomicUsize::new(0);
    let candidates_seen = std::sync::atomic::AtomicUsize::new(0);
    let kind_counts: [std::sync::atomic::AtomicUsize; 5] =
        Default::default();

    members.par_iter().try_for_each_init(
        || -> Result<ZipArchive<BufReader<File>>> {
            let f = File::open(&zip_path)?;
            Ok(ZipArchive::new(BufReader::new(f))?)
        },
        |zf_res, member| -> Result<()> {
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
        let n = games_seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

        let candidates = extract_candidates(member, &meta, &rows, &enabled_kinds, args.trajectory);
        candidates_seen.fetch_add(candidates.len(), std::sync::atomic::Ordering::Relaxed);
        if candidates.is_empty() {
            return Ok(());
        }

        let selected: Vec<Candidate> = if args.trajectory {
            candidates
        } else {
            match selection {
                Selection::Hash => stable_choices(candidates, args.choices_per_game),
                Selection::Priority => priority_choices(
                    candidates,
                    &kind_priority,
                    args.choices_per_game,
                ),
            }
        };
        if selected.is_empty() {
            return Ok(());
        }

        let mut batch: Vec<Vec<u8>> = Vec::with_capacity(selected.len());
        let candidate_count = selected[0].candidate_count;
        for c in selected {
            let kind_idx = CHOICE_KINDS.iter().position(|k| *k == c.kind).unwrap();
            kind_counts[kind_idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let rec = build_record(&c, &meta, candidate_count);
            batch.push(serde_json::to_vec(&rec)?);
        }
        games_written.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        tx.send(batch).map_err(|e| anyhow!("writer dropped: {e}"))?;

        if progress_every > 0 && n % progress_every == 0 {
            eprintln!(
                "{{\"games_seen\":{}, \"games_written\":{}, \"candidates_seen\":{}}}",
                n,
                games_written.load(std::sync::atomic::Ordering::Relaxed),
                candidates_seen.load(std::sync::atomic::Ordering::Relaxed),
            );
        }
        Ok(())
        },
    )?;
    drop(tx);

    let writer_stats = writer_handle
        .join()
        .map_err(|_| anyhow!("writer thread panicked"))??;

    let kind_summary: Map<String, Value> = CHOICE_KINDS
        .iter()
        .enumerate()
        .map(|(i, k)| {
            (
                format!("written_{}", k),
                Value::from(kind_counts[i].load(std::sync::atomic::Ordering::Relaxed)),
            )
        })
        .collect();
    let summary = json!({
        "games_seen": games_seen.load(std::sync::atomic::Ordering::Relaxed),
        "games_written": games_written.load(std::sync::atomic::Ordering::Relaxed),
        "candidates_seen": candidates_seen.load(std::sync::atomic::Ordering::Relaxed),
        "records_written": writer_stats.records,
        "kinds": kind_summary,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[derive(Clone, Copy)]
enum Selection {
    Hash,
    Priority,
}

struct WriterStats {
    records: u64,
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
        out = DEFAULT_KIND_PRIORITY.iter().map(|s| s.to_string()).collect();
    }
    Ok(out)
}

// --- IO ---------------------------------------------------------------------

fn read_jsonl_member<R: std::io::Read>(reader: R) -> Result<Vec<Value>> {
    // gzip + jsonl. Filter to META and EVENTs that carry a snapshot (matches python).
    let gz = MultiGzDecoder::new(reader);
    let buf = BufReader::new(gz);
    let mut rows = Vec::new();
    for line in buf.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue, // skip malformed lines defensively
        };
        let record = v.get("record").and_then(Value::as_str).unwrap_or("");
        let keep = match record {
            "META" => true,
            "EVENT" => v.get("snapshot").map(|s| s.is_object()).unwrap_or(false),
            _ => false,
        };
        if keep {
            rows.push(v);
        }
    }
    Ok(rows)
}

// --- Meta -------------------------------------------------------------------

#[derive(Clone)]
struct Meta {
    game_id: String,
    winner_id: Option<String>,
    winner_name: Option<String>,
    players: Vec<Value>,
    extras: Value,
}

fn meta_from_rows(rows: &[Value]) -> Option<Meta> {
    for row in rows {
        if row.get("record").and_then(Value::as_str) != Some("META") {
            continue;
        }
        let game_id = row
            .get("gameId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let winner_id = row
            .get("winnerId")
            .and_then(Value::as_str)
            .map(str::to_string);
        let winner_name = row
            .get("winnerName")
            .and_then(Value::as_str)
            .map(str::to_string);
        let players = row
            .get("players")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let extras = row.get("extras").cloned().unwrap_or(json!({}));
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

fn mana_pool(raw: &str) -> Map<String, Value> {
    let mut m: Map<String, Value> = MANA_COLORS
        .iter()
        .map(|c| (c.to_string(), Value::from(0u32)))
        .collect();
    for ch in raw.chars() {
        let key = match ch.to_ascii_uppercase() {
            'W' => "White",
            'U' => "Blue",
            'B' => "Black",
            'R' => "Red",
            'G' => "Green",
            'C' => "Colorless",
            _ => continue,
        };
        if let Some(Value::Number(n)) = m.get_mut(key) {
            let v = n.as_u64().unwrap_or(0) + 1;
            *n = serde_json::Number::from(v);
        }
    }
    m
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
            .get("name")
            .and_then(Value::as_str)
            .or_else(|| obj.get("Name").and_then(Value::as_str))
            .unwrap_or("");
        let cid = obj
            .get("id")
            .and_then(Value::as_str)
            .or_else(|| obj.get("ID").and_then(Value::as_str))
            .map(str::to_string)
            .unwrap_or_else(|| format!("{owner_id}:{zone}:{index}:{name}"));
        let mut out = Map::new();
        out.insert("ID".into(), Value::from(cid));
        out.insert("Name".into(), Value::from(name.to_string()));
        if let Some(t) = obj.get("tapped") {
            out.insert("Tapped".into(), Value::from(t.as_bool().unwrap_or(false)));
        } else if let Some(t) = obj.get("Tapped") {
            out.insert("Tapped".into(), Value::from(t.as_bool().unwrap_or(false)));
        }
        return Value::Object(out);
    }
    json!({
        "ID": format!("{owner_id}:{zone}:{index}:unknown"),
        "Name": "",
    })
}

fn player(raw: &Value) -> Value {
    let obj = raw.as_object().cloned().unwrap_or_default();
    let pid = obj
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| obj.get("ID").and_then(Value::as_str))
        .unwrap_or("")
        .to_string();
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| obj.get("Name").and_then(Value::as_str))
        .unwrap_or("")
        .to_string();
    let life = obj
        .get("life")
        .or_else(|| obj.get("Life"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let library_count = obj
        .get("librarySize")
        .or_else(|| obj.get("LibraryCount"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let hand: Vec<Value> = obj
        .get("hand")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, c)| zone_card(c, &pid, "hand", i))
                .collect()
        })
        .unwrap_or_default();
    let graveyard: Vec<Value> = obj
        .get("graveyard")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, c)| zone_card(c, &pid, "graveyard", i))
                .collect()
        })
        .unwrap_or_default();
    let exile: Vec<Value> = obj
        .get("exile")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, c)| zone_card(c, &pid, "exile", i))
                .collect()
        })
        .unwrap_or_default();
    let hand_count = obj
        .get("handSize")
        .and_then(Value::as_u64)
        .unwrap_or(hand.len() as u64);
    let mp_raw = obj.get("manaPool").and_then(Value::as_str).unwrap_or("");

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
        .cloned()
        .unwrap_or_default();
    let mut players: Vec<Value> = players_in.iter().map(player).collect();

    // Group battlefield by controller.
    let mut bf_by_player: indexmap::IndexMap<String, Vec<Value>> = indexmap::IndexMap::new();
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
            obj.insert("Battlefield".into(), Value::from(bf));
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
                        o.get("name")
                            .and_then(Value::as_str)
                            .or_else(|| o.get("description").and_then(Value::as_str))
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
    let actions = raw_snapshot.get("playableActions").and_then(Value::as_array)?;
    let mut options: Vec<Value> = Vec::with_capacity(actions.len() + 1);
    for (i, act) in actions.iter().enumerate() {
        let Some(obj) = act.as_object() else { continue };
        let kind = obj
            .get("type")
            .and_then(Value::as_str)
            .or_else(|| obj.get("kind").and_then(Value::as_str))
            .unwrap_or("");
        if kind == "PASS" {
            // Pass is appended unconditionally below.
            continue;
        }
        let opt_kind = match forge_kind_to_option_kind(kind) {
            Some(k) => k,
            None => continue,
        };
        let ability_id = obj.get("abilityId").and_then(Value::as_str).unwrap_or("");
        let source_id = obj.get("sourceId").and_then(Value::as_str).unwrap_or("");
        let source_name = obj.get("sourceName").and_then(Value::as_str).unwrap_or("");
        let description = obj.get("description").and_then(Value::as_str).unwrap_or("");
        let label = if description.is_empty() { source_name } else { description };
        let cost = obj.get("cost").and_then(Value::as_str).unwrap_or("");
        let id = if ability_id.is_empty() {
            format!("opt:{i}")
        } else {
            ability_id.to_string()
        };
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

fn attackers_pending(
    raw_snapshot: &Value,
    perspective_id: &str,
    display_ids: &DisplayIdIndex,
) -> Option<Value> {
    let bf = raw_snapshot.get("battlefield").and_then(Value::as_array)?;
    let mut options: Vec<Value> = Vec::new();
    let mut display_usage = indexmap::IndexMap::new();
    for card in bf {
        let Some(obj) = card.as_object() else { continue };
        if obj.get("controllerId").and_then(Value::as_str) != Some(perspective_id) {
            continue;
        }
        let power = obj.get("power").and_then(Value::as_i64).unwrap_or(0);
        if power <= 0 {
            continue;
        }
        let id = card_handle(card, display_ids, &mut display_usage);
        let name = obj.get("name").and_then(Value::as_str).unwrap_or("").to_string();
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
            let Some(obj) = card.as_object() else { continue };
            let cid = obj.get("id").and_then(Value::as_str).unwrap_or("");
            if cid.starts_with(prefix.as_str()) {
                let name = obj.get("name").and_then(Value::as_str).unwrap_or("");
                attacker_targets.push(json!({"id": cid, "label": name}));
                break;
            }
        }
    }
    if attacker_targets.is_empty() {
        return None;
    }
    let mut options: Vec<Value> = Vec::new();
    let mut display_usage = indexmap::IndexMap::new();
    for card in bf {
        let Some(obj) = card.as_object() else { continue };
        if obj.get("controllerId").and_then(Value::as_str) != Some(perspective_id) {
            continue;
        }
        let power = obj.get("power").and_then(Value::as_i64).unwrap_or(0);
        if power <= 0 {
            continue;
        }
        if obj.get("tapped").and_then(Value::as_bool).unwrap_or(false) {
            continue;
        }
        let id = card_handle(card, display_ids, &mut display_usage);
        let name = obj.get("name").and_then(Value::as_str).unwrap_or("").to_string();
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

fn raw_player_id_by_name(snapshot: &Value, name: &str) -> String {
    if let Some(arr) = snapshot.get("players").and_then(Value::as_array) {
        for p in arr {
            if p.get("name").and_then(Value::as_str) == Some(name) {
                return p
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
            }
        }
    }
    String::new()
}

fn raw_player_name_by_id(snapshot: &Value, id: &str) -> String {
    if let Some(arr) = snapshot.get("players").and_then(Value::as_array) {
        for p in arr {
            if p.get("id").and_then(Value::as_str) == Some(id) {
                return p
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
            }
        }
    }
    String::new()
}

fn raw_opponent_id(snapshot: &Value, pid: &str) -> String {
    if let Some(arr) = snapshot.get("players").and_then(Value::as_array) {
        for p in arr {
            let candidate = p.get("id").and_then(Value::as_str).unwrap_or("");
            if !candidate.is_empty() && candidate != pid {
                return candidate.to_string();
            }
        }
    }
    String::new()
}

// --- regexes / label parsing ------------------------------------------------

static ATTACKS_HEADER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?P<attacker_player>.+?) attacks (?P<defender_player>.+?) with (?P<count>\d+) creatures?$").unwrap()
});
static ATTACKER_LINE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^Attacker:\s+(?P<name>.+?)\s+\[(?P<id>[0-9a-f]+)\]\s+\((?P<pt>[^)]+)\)\s+(?P<rest>.+)$").unwrap()
});
static BLOCKER_TOKEN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?P<name>.+?)\s+\[(?P<id>[0-9a-f]+)\]\s+\((?P<pt>[^)]+)\)").unwrap()
});
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

type DisplayIdIndex = indexmap::IndexMap<String, Vec<String>>;

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

fn has_assigned_attack_log(row: &Value) -> bool {
    if row.get("type").and_then(Value::as_str) != Some("LOG") {
        return false;
    }
    row.get("description")
        .and_then(Value::as_str)
        .map(|desc| desc.lines().any(|line| parse_assigned_attack(line).is_some()))
        .unwrap_or(false)
}

fn has_new_block_log(row: &Value) -> bool {
    if row.get("type").and_then(Value::as_str) != Some("LOG") {
        return false;
    }
    row.get("description")
        .and_then(Value::as_str)
        .map(|desc| {
            desc.lines().any(|line| {
                parse_assigned_block(line).is_some() || parse_didnt_block(line).is_some()
            })
        })
        .unwrap_or(false)
}

fn assigned_attack_observed(window: &[&Value]) -> Option<Value> {
    let mut actor_name = String::new();
    let mut defender_name = String::new();
    let mut attackers: Vec<CombatCardRef> = Vec::new();
    let mut raw_lines: Vec<String> = Vec::new();

    for row in window {
        if row.get("type").and_then(Value::as_str) != Some("LOG") {
            continue;
        }
        let desc = row.get("description").and_then(Value::as_str).unwrap_or("");
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

fn assigned_block_observed(window: &[&Value]) -> Option<Value> {
    let mut actor_name = String::new();
    let mut attackers: Vec<CombatCardRef> = Vec::new();
    let mut assignments: Vec<Value> = Vec::new();
    let mut raw_lines: Vec<String> = Vec::new();

    for row in window {
        if row.get("type").and_then(Value::as_str) != Some("LOG") {
            continue;
        }
        let desc = row.get("description").and_then(Value::as_str).unwrap_or("");
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
        .and_then(|r| r.get("snapshot"))
        .cloned()
        .unwrap_or(json!({}));
    let active = snap
        .get("activePlayerId")
        .and_then(Value::as_str)
        .unwrap_or("");
    Some(json!({
        "raw": raw_lines.join("\n"),
        "actor_name": actor_name,
        "attacker_player": raw_player_name_by_id(&snap, active),
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
    display_usage: &mut indexmap::IndexMap<String, usize>,
) -> String {
    let controller = card.get("controllerId").and_then(Value::as_str).unwrap_or("");
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
    let lower = desc.to_ascii_lowercase();
    if event_type == "STACK_PUSH" && (lower.contains(" cast ") || lower.contains(" activated ")) {
        return Some(json!({"raw": desc, "event_type": event_type}));
    }
    if lower.contains(" played ") {
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

fn choose_label(desc: &str) -> Option<Value> {
    if !desc.to_ascii_lowercase().contains("choose") {
        return None;
    }
    let actor = PLAYER_PREFIX_RE
        .captures(desc)
        .map(|c| c["player"].to_string())
        .unwrap_or_default();
    Some(json!({"raw": desc, "actor_name": actor}))
}

fn may_label(rows: &[&Value], idx: usize) -> Option<Value> {
    let row = rows[idx];
    let desc = row.get("description").and_then(Value::as_str).unwrap_or("");
    let row_type = row.get("type").and_then(Value::as_str).unwrap_or("");
    if row_type != "STACK_RESOLVE" || !desc.to_ascii_lowercase().contains("you may") {
        return None;
    }
    let actor = PLAYER_PREFIX_RE
        .captures(desc)
        .map(|c| c["player"].to_string())
        .unwrap_or_default();
    let start = idx.saturating_sub(4);
    let mut effect_logs: Vec<String> = Vec::new();
    for r in &rows[start..=idx] {
        if r.get("type").and_then(Value::as_str) == Some("LOG") {
            let d = r.get("description").and_then(Value::as_str).unwrap_or("");
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

fn may_source_row<'a>(rows: &[&'a Value], idx: usize) -> &'a Value {
    let desc = rows[idx]
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("");
    for j in (0..idx).rev() {
        let r = rows[j];
        let t = r.get("type").and_then(Value::as_str).unwrap_or("");
        if t == "PRIORITY" {
            return r;
        }
        if t == "STACK_PUSH"
            && r.get("description").and_then(Value::as_str).unwrap_or("") == desc
        {
            return r;
        }
    }
    rows[idx.saturating_sub(1)]
}

// --- candidate model --------------------------------------------------------

#[derive(Clone)]
struct Candidate {
    kind: &'static str,
    game_id: String,
    archive_member: String,
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
    archive_member: &str,
    game_id: &str,
    source: &Value,
    target: &Value,
    perspective_id: &str,
    perspective_name: &str,
    observed: Value,
) -> Option<Candidate> {
    let raw_snapshot = source.get("snapshot")?;
    let mut normalized = normalize_snapshot(raw_snapshot, perspective_id);
    let display_ids = display_ids_from_observed(raw_snapshot, perspective_id, kind, &observed);

    match kind {
        "priority" => {
            if let Some(pending) = priority_pending(raw_snapshot, perspective_id) {
                normalized
                    .as_object_mut()
                    .unwrap()
                    .insert("pending".into(), pending);
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
                            o.get("permanent_id")
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
                .insert("pending".into(), pending);
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
            let pending = blockers_pending(raw_snapshot, perspective_id, &attacker_refs, &display_ids)?;
            let opt_ids: Vec<String> = pending
                .get("options")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .map(|o| {
                            o.get("permanent_id")
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
                .insert("pending".into(), pending);
        }
        _ => {}
    }

    Some(Candidate {
        kind,
        game_id: game_id.to_string(),
        archive_member: archive_member.to_string(),
        source_seq: source.get("seq").and_then(Value::as_i64).unwrap_or(0),
        target_seq: target.get("seq").and_then(Value::as_i64).unwrap_or(0),
        perspective_id: perspective_id.to_string(),
        perspective_name: perspective_name.to_string(),
        snapshot: normalized,
        observed,
        candidate_index: 0,
        candidate_count: 0,
    })
}

// --- main extraction --------------------------------------------------------

fn event_rows(rows: &[Value]) -> Vec<&Value> {
    rows.iter()
        .filter(|r| {
            r.get("record").and_then(Value::as_str) == Some("EVENT")
                && r.get("snapshot").map(|s| s.is_object()).unwrap_or(false)
        })
        .collect()
}

fn attack_observed(events: &[&Value], header_idx: usize, header: &AttacksHeader) -> Value {
    let mut attackers: Vec<Value> = Vec::new();
    for row in &events[header_idx + 1..] {
        let snap = row.get("snapshot").cloned().unwrap_or(json!({}));
        let step = snap.get("step").and_then(Value::as_str).unwrap_or("");
        if step != "DECLARE_ATTACKERS" && step != "DECLARE_BLOCKERS" {
            break;
        }
        if row.get("type").and_then(Value::as_str) != Some("LOG") {
            continue;
        }
        let desc = row.get("description").and_then(Value::as_str).unwrap_or("");
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

fn block_observed(events: &[&Value], header_idx: usize, header: &AttacksHeader) -> Option<Value> {
    let mut assignments: Vec<Value> = Vec::new();
    let mut seen_attackers: Vec<Value> = Vec::new();
    for row in &events[header_idx + 1..] {
        let snap = row.get("snapshot").cloned().unwrap_or(json!({}));
        let step = snap.get("step").and_then(Value::as_str).unwrap_or("");
        if step != "DECLARE_ATTACKERS" && step != "DECLARE_BLOCKERS" {
            break;
        }
        if row.get("type").and_then(Value::as_str) != Some("LOG") {
            continue;
        }
        let desc = row.get("description").and_then(Value::as_str).unwrap_or("");
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

fn extract_candidates(
    archive_member: &str,
    meta: &Meta,
    rows: &[Value],
    enabled: &[String],
    trajectory: bool,
) -> Vec<Candidate> {
    let events = event_rows(rows);
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut last_priority_by_player: indexmap::IndexMap<String, (usize, &Value)> =
        indexmap::IndexMap::new();
    let mut consumed_priority_sources: std::collections::HashSet<(i64, String)> =
        std::collections::HashSet::new();
    let want_priority = enabled.iter().any(|k| k == "priority");
    let want_attack = enabled.iter().any(|k| k == "attack");
    let want_block = enabled.iter().any(|k| k == "block");
    let want_may = enabled.iter().any(|k| k == "may");
    let want_choose = enabled.iter().any(|k| k == "choose");

    for (idx, row) in events.iter().enumerate() {
        let snap = row.get("snapshot").cloned().unwrap_or(json!({}));
        let row_type = row.get("type").and_then(Value::as_str).unwrap_or("");
        let desc = row.get("description").and_then(Value::as_str).unwrap_or("");
        let priority_id = snap
            .get("priorityPlayerId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if row_type == "PRIORITY" && !priority_id.is_empty() {
            last_priority_by_player.insert(priority_id.clone(), (idx, row));
        }

        if want_priority {
            if let Some(label) = priority_label(desc, row_type) {
                let actor_id = priority_id.clone();
                if let Some((_, source)) = last_priority_by_player.get(&actor_id).copied() {
                    let source_key = (
                        source.get("seq").and_then(Value::as_i64).unwrap_or(0),
                        actor_id.clone(),
                    );
                    if consumed_priority_sources.contains(&source_key) {
                        continue;
                    }
                    if let Some(c) = make_candidate(
                        "priority",
                        archive_member,
                        &meta.game_id,
                        source,
                        row,
                        &actor_id,
                        &raw_player_name_by_id(&snap, &actor_id),
                        label,
                    ) {
                        consumed_priority_sources.insert(source_key);
                        candidates.push(c);
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
                        let actor_id = raw_player_id_by_name(&snap, &header.attacker_player);
                        let perspective = if !actor_id.is_empty() {
                            actor_id.clone()
                        } else {
                            snap.get("activePlayerId")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string()
                        };
                        if let Some(c) = make_candidate(
                            "attack",
                            archive_member,
                            &meta.game_id,
                            row,
                            row,
                            &perspective,
                            &header.attacker_player,
                            observed,
                        ) {
                            candidates.push(c);
                        }
                    }
                }
                if want_block {
                    if let Some(observed) = block_observed(&events, idx, &header) {
                        let defender = header.defender_player.clone();
                        let mut actor_id = raw_player_id_by_name(&snap, &defender);
                        if actor_id.is_empty() {
                            let active = snap
                                .get("activePlayerId")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            actor_id = raw_opponent_id(&snap, &active);
                        }
                        // Find first Attacker: row to use its snapshot as source.
                        let mut block_source: &Value = row;
                        for jrow in &events[idx + 1..] {
                            if jrow.get("type").and_then(Value::as_str) != Some("LOG") {
                                continue;
                            }
                            let jdesc =
                                jrow.get("description").and_then(Value::as_str).unwrap_or("");
                            if jdesc.starts_with("Attacker:") {
                                block_source = jrow;
                                break;
                            }
                        }
                        if let Some(c) = make_candidate(
                            "block",
                            archive_member,
                            &meta.game_id,
                            block_source,
                            block_source,
                            &actor_id,
                            &defender,
                            observed,
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
                let actor_id = raw_player_id_by_name(&snap, &actor_name);
                let perspective = if actor_id.is_empty() {
                    priority_id.clone()
                } else {
                    actor_id
                };
                let source = may_source_row(&events, idx);
                if let Some(c) = make_candidate(
                    "may",
                    archive_member,
                    &meta.game_id,
                    source,
                    row,
                    &perspective,
                    &actor_name,
                    label,
                ) {
                    candidates.push(c);
                }
            }
        }

        if want_choose
            && row_type == "LOG"
            && desc.to_ascii_lowercase().contains("choose")
        {
            if let Some(label) = choose_label(desc) {
                let actor_name = label
                    .get("actor_name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let actor_id = raw_player_id_by_name(&snap, &actor_name);
                let perspective = if actor_id.is_empty() {
                    priority_id.clone()
                } else {
                    actor_id
                };
                if let Some(c) = make_candidate(
                    "choose",
                    archive_member,
                    &meta.game_id,
                    row,
                    row,
                    &perspective,
                    &actor_name,
                    label,
                ) {
                    candidates.push(c);
                }
            }
        }
    }

    if want_attack || want_block {
        synthesize_combat_passes(
            &events,
            archive_member,
            meta,
            want_attack,
            want_block,
            &mut candidates,
        );
    }
    if trajectory && want_priority {
        infer_priority_passes(
            &events,
            archive_member,
            meta,
            &consumed_priority_sources,
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
    events: &[&Value],
    archive_member: &str,
    meta: &Meta,
    consumed_priority_sources: &std::collections::HashSet<(i64, String)>,
    out: &mut Vec<Candidate>,
) {
    for (idx, row) in events.iter().enumerate() {
        if row.get("type").and_then(Value::as_str) != Some("PRIORITY") {
            continue;
        }
        let Some(snap) = row.get("snapshot") else {
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
        let seq = row.get("seq").and_then(Value::as_i64).unwrap_or(0);
        if consumed_priority_sources.contains(&(seq, actor_id.to_string())) {
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
            archive_member,
            &meta.game_id,
            row,
            target,
            actor_id,
            &raw_player_name_by_id(snap, actor_id),
            observed,
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
    events: &[&Value],
    archive_member: &str,
    meta: &Meta,
    want_attack: bool,
    want_block: bool,
    out: &mut Vec<Candidate>,
) {
    let mut i = 0;
    while i < events.len() {
        let step = events[i]
            .get("snapshot")
            .and_then(|s| s.get("step"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if step != "DECLARE_ATTACKERS" && step != "DECLARE_BLOCKERS" {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i;
        while end < events.len() {
            let s = events[end]
                .get("snapshot")
                .and_then(|s| s.get("step"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if s != step {
                break;
            }
            end += 1;
        }
        let window = &events[start..end];

        if step == "DECLARE_ATTACKERS" && want_attack {
            let any_header = window.iter().any(|r| {
                r.get("type").and_then(Value::as_str) == Some("LOG")
                    && parse_attacks_header(
                        r.get("description").and_then(Value::as_str).unwrap_or(""),
                    )
                    .is_some()
            });
            if !any_header {
                let source = window[0];
                if let Some(observed) = assigned_attack_observed(window) {
                    if let Some(snap) = source.get("snapshot") {
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
                                .unwrap_or("")
                                .to_string();
                        }
                        let target = window
                            .iter()
                            .find(|r| has_assigned_attack_log(r))
                            .copied()
                            .unwrap_or(source);
                        if !actor_id.is_empty() {
                            if let Some(c) = make_candidate(
                                "attack",
                                archive_member,
                                &meta.game_id,
                                source,
                                target,
                                &actor_id,
                                &actor_name,
                                observed,
                            ) {
                                out.push(c);
                            }
                        }
                    }
                    i = end;
                    continue;
                }
                if let Some(snap) = source.get("snapshot") {
                    let active = snap
                        .get("activePlayerId")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    if !active.is_empty() {
                        let actor_name = raw_player_name_by_id(snap, &active);
                        let observed = json!({
                            "raw": format!("{} declares no attackers", actor_name),
                            "actor_name": actor_name,
                            "attackers": [],
                            "no_attack": true,
                        });
                        if let Some(c) = make_candidate(
                            "attack",
                            archive_member,
                            &meta.game_id,
                            source,
                            source,
                            &active,
                            &actor_name,
                            observed,
                        ) {
                            out.push(c);
                        }
                    }
                }
            }
        } else if step == "DECLARE_BLOCKERS" && want_block {
            let any_attacker_line = window.iter().any(|r| {
                r.get("type").and_then(Value::as_str) == Some("LOG")
                    && r.get("description")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .starts_with("Attacker:")
            });
            if !any_attacker_line {
                if let Some(observed) = assigned_block_observed(window) {
                    let source = window[0];
                    if let Some(snap) = source.get("snapshot") {
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
                                .unwrap_or("")
                                .to_string();
                            actor_id = raw_opponent_id(snap, &active);
                        }
                        let target = window
                            .iter()
                            .find(|r| has_new_block_log(r))
                            .copied()
                            .unwrap_or(source);
                        if !actor_id.is_empty() {
                            if let Some(c) = make_candidate(
                                "block",
                                archive_member,
                                &meta.game_id,
                                source,
                                target,
                                &actor_id,
                                &actor_name,
                                observed,
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
                if r.get("type").and_then(Value::as_str) != Some("LOG") {
                    return false;
                }
                let d = r.get("description").and_then(Value::as_str).unwrap_or("");
                d.starts_with("Attacker:")
                    && parse_attacker_line(d)
                        .map(|p| !p.blockers.is_empty())
                        .unwrap_or(false)
            });
            if !any_block {
                let mut attacker_records: Vec<Value> = Vec::new();
                for r in window {
                    let d = r.get("description").and_then(Value::as_str).unwrap_or("");
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
                        .find(|r| {
                            r.get("description")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .starts_with("Attacker:")
                        })
                        .copied()
                        .unwrap_or(window[0]);
                    if let Some(snap) = source.get("snapshot") {
                        let active = snap
                            .get("activePlayerId")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let defender = raw_opponent_id(snap, &active);
                        if !defender.is_empty() {
                            let defender_name = raw_player_name_by_id(snap, &defender);
                            let observed = json!({
                                "raw": "no blocks declared",
                                "actor_name": defender_name,
                                "attackers": attacker_records,
                                "assignments": [],
                                "no_block": true,
                            });
                            if let Some(c) = make_candidate(
                                "block",
                                archive_member,
                                &meta.game_id,
                                source,
                                source,
                                &defender,
                                &defender_name,
                                observed,
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

fn stable_choices(mut candidates: Vec<Candidate>, count: usize) -> Vec<Candidate> {
    if count == 0 {
        return Vec::new();
    }
    if count >= candidates.len() {
        return candidates;
    }
    let mut keyed: Vec<(u64, Candidate)> = candidates
        .drain(..)
        .map(|c| {
            let key = format!("{}:{}", c.game_id, c.candidate_index);
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
    let mut by_kind: indexmap::IndexMap<&str, Vec<Candidate>> = indexmap::IndexMap::new();
    for k in CHOICE_KINDS {
        by_kind.insert(*k, Vec::new());
    }
    for c in candidates.into_iter() {
        by_kind.entry(c.kind).or_default().push(c);
    }
    let mut selected: Vec<Candidate> = Vec::new();
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for kind in kind_priority {
        if let Some(bucket) = by_kind.get_mut(kind.as_str()) {
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
    for (_, bucket) in by_kind {
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

fn build_record(c: &Candidate, meta: &Meta, candidate_count: usize) -> Value {
    let _ = candidate_count;
    json!({
        "format": "forge_choice_situation",
        "format_version": FORMAT_VERSION,
        "game_id": c.game_id,
        "archive_member": c.archive_member,
        "choice": {
            "kind": c.kind,
            "candidate_index": c.candidate_index,
            "candidate_count": c.candidate_count,
            "source_seq": c.source_seq,
            "target_seq": c.target_seq,
            "perspective_id": c.perspective_id,
            "perspective_name": c.perspective_name,
            "observed": c.observed,
        },
        "state": {
            "snapshot": c.snapshot,
        },
        "outcome": {
            "winner_id": meta.winner_id,
            "winner_name": meta.winner_name,
            "terminal_sign": terminal_sign(meta.winner_id.as_deref(), &c.perspective_id),
            "players": meta.players,
            "extras": meta.extras,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_assigned_attack_list() {
        let (_, defender, attackers) = parse_assigned_attack(
            "PlayerB assigned Dancing Scimitar (93), Zephyr Falcon (73) and Zephyr Falcon (71) to attack PlayerA.",
        )
        .unwrap();

        assert_eq!(defender, "PlayerA");
        assert_eq!(
            attackers.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
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
        let meta = Meta {
            game_id: "game".to_string(),
            winner_id: None,
            winner_name: None,
            players: vec![],
            extras: json!({}),
        };
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
            json!({
                "record": "EVENT",
                "type": "PRIORITY",
                "seq": 11,
                "snapshot": snapshot,
            }),
            json!({
                "record": "EVENT",
                "type": "LOG",
                "seq": 12,
                "description": "PlayerB played Island (97)",
                "snapshot": snapshot,
            }),
            json!({
                "record": "EVENT",
                "type": "STACK_PUSH",
                "seq": 13,
                "description": "STACK_PUSH Ornithopter by PlayerB",
                "snapshot": snapshot,
            }),
        ];

        let candidates = extract_candidates(
            "game.jsonl.gz",
            &meta,
            &rows,
            &["priority".to_string()],
            true,
        );

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].source_seq, 11);
        assert_eq!(candidates[0].target_seq, 12);
        assert_eq!(
            candidates[0].observed.get("raw").and_then(Value::as_str),
            Some("PlayerB played Island (97)")
        );
    }
}

use std::fmt::Display;
use std::hint::black_box;
use std::str::FromStr;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Crate {
    Pickled,
    SerdePickle,
    Both,
}

impl FromStr for Crate {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "pickled" => Ok(Self::Pickled),
            "serde-pickle" => Ok(Self::SerdePickle),
            "both" => Ok(Self::Both),
            other => bail!("unknown crate {other:?}; expected pickled, serde-pickle, or both"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    ToVec,
    ReusedWriter,
    Both,
}

impl FromStr for Mode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "to-vec" => Ok(Self::ToVec),
            "reused-writer" => Ok(Self::ReusedWriter),
            "both" => Ok(Self::Both),
            other => bail!("unknown mode {other:?}; expected to-vec, reused-writer, or both"),
        }
    }
}

#[derive(Debug)]
struct Args {
    selected_crate: Crate,
    mode: Mode,
    tensors: usize,
    total_mib: usize,
    warmup: usize,
    iterations: usize,
    check_rust_roundtrip: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            selected_crate: Crate::Both,
            mode: Mode::Both,
            tensors: 4,
            total_mib: 256,
            warmup: 3,
            iterations: 10,
            check_rust_roundtrip: false,
        };

        let mut items = std::env::args().skip(1);
        while let Some(arg) = items.next() {
            match arg.as_str() {
                "--crate" => {
                    args.selected_crate = parse_next(&mut items, "--crate")?;
                }
                "--mode" => {
                    args.mode = parse_next(&mut items, "--mode")?;
                }
                "--tensors" => {
                    args.tensors = parse_next(&mut items, "--tensors")?;
                }
                "--total-mib" => {
                    args.total_mib = parse_next(&mut items, "--total-mib")?;
                }
                "--warmup" => {
                    args.warmup = parse_next(&mut items, "--warmup")?;
                }
                "--iterations" => {
                    args.iterations = parse_next(&mut items, "--iterations")?;
                }
                "--check-rust-roundtrip" => {
                    args.check_rust_roundtrip = true;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other:?}; pass --help for usage"),
            }
        }

        if args.tensors == 0 {
            bail!("--tensors must be greater than zero");
        }
        if args.total_mib == 0 {
            bail!("--total-mib must be greater than zero");
        }
        if args.iterations == 0 {
            bail!("--iterations must be greater than zero");
        }

        Ok(args)
    }
}

fn parse_next<T>(items: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T: FromStr,
    T::Err: Display,
{
    let raw = items
        .next()
        .ok_or_else(|| anyhow!("{name} requires a value"))?;
    raw.parse()
        .map_err(|err| anyhow!("invalid {name} value {raw:?}: {err}"))
}

fn print_help() {
    println!(
        "\
Benchmark pickled vs serde-pickle on tensor-shaped pickle payloads.

This serializes Python built-ins: tensor metadata plus contiguous tensor bytes.
It is not a torch.save-compatible object benchmark.

Usage:
  cargo run --release -- [options]

Options:
  --crate <pickled|serde-pickle|both>      default: both
  --mode <to-vec|reused-writer|both>       default: both
  --tensors <n>                            default: 4
  --total-mib <n>                          default: 256
  --warmup <n>                             default: 3
  --iterations <n>                         default: 10
  --check-rust-roundtrip                   deserialize each result with its crate
"
    );
}

#[derive(Serialize)]
struct TensorBundle<'a> {
    format: &'static str,
    tensors: Vec<TensorPayload<'a>>,
}

#[derive(Serialize)]
struct TensorPayload<'a> {
    name: String,
    dtype: &'static str,
    shape: [usize; 2],
    strides: [usize; 2],
    device: &'static str,
    requires_grad: bool,
    #[serde(with = "serde_bytes")]
    data: &'a [u8],
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let total_bytes = args
        .total_mib
        .checked_mul(1024 * 1024)
        .context("--total-mib overflowed usize")?;
    let total_bytes = total_bytes - (total_bytes % std::mem::size_of::<f32>());
    let payload = make_payload(total_bytes);
    let bundle = make_bundle(&payload, args.tensors)?;
    let payload_mib = total_bytes as f64 / 1024.0 / 1024.0;

    println!(
        "payload: {:.1} MiB across {} f32 CPU tensors",
        payload_mib, args.tensors
    );
    println!("iterations: {} warmup: {}", args.iterations, args.warmup);
    println!("note: this is metadata + Python bytes, not a torch.save tensor object");

    let crates = match args.selected_crate {
        Crate::Pickled => vec![Crate::Pickled],
        Crate::SerdePickle => vec![Crate::SerdePickle],
        Crate::Both => vec![Crate::Pickled, Crate::SerdePickle],
    };
    let modes = match args.mode {
        Mode::ToVec => vec![Mode::ToVec],
        Mode::ReusedWriter => vec![Mode::ReusedWriter],
        Mode::Both => vec![Mode::ToVec, Mode::ReusedWriter],
    };

    for mode in modes {
        let mut baseline: Option<BenchResult> = None;
        for selected_crate in &crates {
            warmup(&bundle, *selected_crate, mode, args.warmup)?;
            let result = bench(
                &bundle,
                *selected_crate,
                mode,
                args.iterations,
                args.check_rust_roundtrip,
            )?;
            print_result(&result, payload_mib);
            if let Some(base) = baseline {
                println!(
                    "  relative to {} {}: {:.2}x",
                    base.crate_name,
                    base.mode_name,
                    base.seconds_per_iter.as_secs_f64() / result.seconds_per_iter.as_secs_f64()
                );
            } else {
                baseline = Some(result);
            }
        }
    }

    Ok(())
}

fn make_payload(bytes: usize) -> Vec<u8> {
    let mut payload = vec![0_u8; bytes];
    for (idx, byte) in payload.iter_mut().enumerate() {
        *byte = ((idx.wrapping_mul(31).wrapping_add(17)) & 0xff) as u8;
    }
    payload
}

fn make_bundle(payload: &[u8], tensors: usize) -> Result<TensorBundle<'_>> {
    let bytes_per_tensor = payload.len() / tensors;
    let bytes_per_tensor = bytes_per_tensor - (bytes_per_tensor % std::mem::size_of::<f32>());
    if bytes_per_tensor == 0 {
        bail!("payload is too small for {tensors} tensors");
    }
    let rows = 1024_usize.min(bytes_per_tensor / std::mem::size_of::<f32>());
    let cols = bytes_per_tensor / std::mem::size_of::<f32>() / rows;
    if cols == 0 {
        bail!("payload is too small to form a 2D f32 tensor");
    }

    let mut tensor_payloads = Vec::with_capacity(tensors);
    for tensor_idx in 0..tensors {
        let start = tensor_idx * bytes_per_tensor;
        let end = start + bytes_per_tensor;
        tensor_payloads.push(TensorPayload {
            name: format!("tensor_{tensor_idx}"),
            dtype: "float32",
            shape: [rows, cols],
            strides: [cols, 1],
            device: "cpu",
            requires_grad: false,
            data: &payload[start..end],
        });
    }

    Ok(TensorBundle {
        format: "magic-ai.tensor-bytes.v1",
        tensors: tensor_payloads,
    })
}

#[derive(Clone, Copy)]
struct BenchResult {
    crate_name: &'static str,
    mode_name: &'static str,
    bytes_written: usize,
    seconds_per_iter: Duration,
}

fn warmup(
    bundle: &TensorBundle<'_>,
    selected_crate: Crate,
    mode: Mode,
    iterations: usize,
) -> Result<()> {
    if iterations == 0 {
        return Ok(());
    }
    let _ = bench(bundle, selected_crate, mode, iterations, false)?;
    Ok(())
}

fn bench(
    bundle: &TensorBundle<'_>,
    selected_crate: Crate,
    mode: Mode,
    iterations: usize,
    check_rust_roundtrip: bool,
) -> Result<BenchResult> {
    let crate_name = crate_name(selected_crate);
    let mode_name = mode_name(mode);

    let mut bytes_written = 0_usize;
    let start = Instant::now();
    match mode {
        Mode::ToVec => {
            for _ in 0..iterations {
                let output = serialize_to_vec(bundle, selected_crate)?;
                bytes_written = output.len();
                black_box(output);
            }
        }
        Mode::ReusedWriter => {
            let mut output = Vec::new();
            for _ in 0..iterations {
                output.clear();
                serialize_to_writer(bundle, selected_crate, &mut output)?;
                bytes_written = output.len();
                black_box(&output);
            }
        }
        Mode::Both => unreachable!("Mode::Both is expanded before benchmarking"),
    }
    let elapsed = start.elapsed();

    if check_rust_roundtrip {
        let output = serialize_to_vec(bundle, selected_crate)?;
        check_round_trip(selected_crate, &output)?;
    }

    Ok(BenchResult {
        crate_name,
        mode_name,
        bytes_written,
        seconds_per_iter: elapsed / iterations as u32,
    })
}

fn serialize_to_vec(bundle: &TensorBundle<'_>, selected_crate: Crate) -> Result<Vec<u8>> {
    match selected_crate {
        Crate::Pickled => pickled::to_vec(bundle, Default::default()).context("pickled to_vec"),
        Crate::SerdePickle => {
            serde_pickle::to_vec(bundle, Default::default()).context("serde-pickle to_vec")
        }
        Crate::Both => unreachable!("Crate::Both is expanded before benchmarking"),
    }
}

fn serialize_to_writer(
    bundle: &TensorBundle<'_>,
    selected_crate: Crate,
    output: &mut Vec<u8>,
) -> Result<()> {
    match selected_crate {
        Crate::Pickled => {
            pickled::to_writer(output, bundle, Default::default()).context("pickled to_writer")
        }
        Crate::SerdePickle => serde_pickle::to_writer(output, bundle, Default::default())
            .context("serde-pickle to_writer"),
        Crate::Both => unreachable!("Crate::Both is expanded before benchmarking"),
    }
}

fn check_round_trip(selected_crate: Crate, output: &[u8]) -> Result<()> {
    match selected_crate {
        Crate::Pickled => {
            let _value: pickled::Value =
                pickled::value_from_slice(output, Default::default()).context("pickled read")?;
        }
        Crate::SerdePickle => {
            let _value: serde_pickle::Value =
                serde_pickle::value_from_slice(output, Default::default())
                    .context("serde-pickle read")?;
        }
        Crate::Both => unreachable!("Crate::Both is expanded before benchmarking"),
    }
    Ok(())
}

fn crate_name(selected_crate: Crate) -> &'static str {
    match selected_crate {
        Crate::Pickled => "pickled",
        Crate::SerdePickle => "serde-pickle",
        Crate::Both => "both",
    }
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::ToVec => "to-vec",
        Mode::ReusedWriter => "reused-writer",
        Mode::Both => "both",
    }
}

fn print_result(result: &BenchResult, payload_mib: f64) {
    let seconds = result.seconds_per_iter.as_secs_f64();
    let output_mib = result.bytes_written as f64 / 1024.0 / 1024.0;
    println!(
        "{} {}: {:.3} ms/iter, {:.1} payload MiB/s, output {:.1} MiB",
        result.crate_name,
        result.mode_name,
        seconds * 1_000.0,
        payload_mib / seconds,
        output_mib,
    );
}

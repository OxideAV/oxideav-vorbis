//! VQ codebook trainer CLI (round-1 scaffold for task #93).
//!
//! Reads raw 16-bit signed little-endian PCM from `--input` (or stdin
//! when `--input -`), runs the in-tree LBG trainer in
//! `oxideav_vorbis::trainer`, and emits a Rust source file containing
//! one or more trained `VqCodebook` constants. The output file is
//! intended to be `include!`'d by `encoder.rs` in round 2.
//!
//! ```text
//! Usage: vq-train [OPTIONS]
//!
//! Options:
//!   --input <path>       Input PCM file, or "-" for stdin (default: stdin)
//!   --channels <N>       Channels in input PCM (default: 1)
//!   --sample-rate <N>    Sample rate, used only for documentation in the
//!                        generated header comment (default: 44100)
//!   --blocksize <N>      MDCT block size, must be a power of two
//!                        (default: 2048 — encoder long block)
//!   --books <N>          Number of codebooks to train (default: 4)
//!   --dim <N>            VQ-codebook dimension (default: 16)
//!   --codewords <N>      Codewords per book (default: 256, must be 2^k)
//!   --output <path>      Output Rust source file
//!                        (default: src/trained_books.rs)
//!   -h, --help           Print this help and exit.
//! ```
//!
//! Round-1 deliverable: the trainer scaffold lands but does **not**
//! wire trained books into the encoder's residue stage. That's round 2.

use std::fs::File;
use std::io::{self, BufWriter};
use std::path::PathBuf;
use std::process::ExitCode;

use oxideav_vorbis::trainer::{
    emit_books_rs, extract_partition_vectors, read_pcm_s16le_to_mono, train_books, TrainerConfig,
    TrainerError,
};

#[derive(Debug)]
struct Args {
    input: Option<PathBuf>, // None = stdin
    channels: usize,
    sample_rate: u32,
    blocksize: usize,
    books: usize,
    dim: usize,
    codewords: usize,
    output: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            input: None,
            channels: 1,
            sample_rate: 44_100,
            blocksize: 2048,
            books: 4,
            dim: 16,
            codewords: 256,
            output: PathBuf::from("src/trained_books.rs"),
        }
    }
}

fn print_help() {
    eprintln!(
        "vq-train — train Vorbis residue VQ codebooks via LBG (oxideav-vorbis task #93)\n\
         \n\
         Usage: vq-train [OPTIONS]\n\
         \n\
         Options:\n\
         \x20 --input <path>       Input PCM file, or \"-\" for stdin (default: stdin)\n\
         \x20 --channels <N>       Channels in input PCM (default: 1)\n\
         \x20 --sample-rate <N>    Sample rate (header comment only, default: 44100)\n\
         \x20 --blocksize <N>      MDCT block size, power of two (default: 2048)\n\
         \x20 --books <N>          Number of codebooks to train (default: 4)\n\
         \x20 --dim <N>            VQ-codebook dimension (default: 16)\n\
         \x20 --codewords <N>      Codewords per book, must be 2^k (default: 256)\n\
         \x20 --output <path>      Output Rust file (default: src/trained_books.rs)\n\
         \x20 -h, --help           Print this help and exit\n"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        let mut val = || iter.next().ok_or_else(|| format!("{arg}: missing value"));
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--input" => {
                let v = val()?;
                args.input = if v == "-" {
                    None
                } else {
                    Some(PathBuf::from(v))
                };
            }
            "--channels" => {
                args.channels = val()?.parse().map_err(|e| format!("--channels: {e}"))?;
            }
            "--sample-rate" => {
                args.sample_rate = val()?.parse().map_err(|e| format!("--sample-rate: {e}"))?;
            }
            "--blocksize" => {
                args.blocksize = val()?.parse().map_err(|e| format!("--blocksize: {e}"))?;
            }
            "--books" => {
                args.books = val()?.parse().map_err(|e| format!("--books: {e}"))?;
            }
            "--dim" => {
                args.dim = val()?.parse().map_err(|e| format!("--dim: {e}"))?;
            }
            "--codewords" => {
                args.codewords = val()?.parse().map_err(|e| format!("--codewords: {e}"))?;
            }
            "--output" => {
                args.output = PathBuf::from(val()?);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(args)
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if !args.blocksize.is_power_of_two() || !(64..=8192).contains(&args.blocksize) {
        return Err(format!(
            "--blocksize must be a power of two in 64..=8192 (got {})",
            args.blocksize
        )
        .into());
    }
    if !args.codewords.is_power_of_two() {
        return Err(format!(
            "--codewords must be a power of two for clean LBG growth (got {})",
            args.codewords
        )
        .into());
    }
    if (args.blocksize / 2) % args.dim != 0 {
        return Err(format!(
            "(blocksize/2) must be divisible by --dim ({}/2 not divisible by {})",
            args.blocksize, args.dim
        )
        .into());
    }

    eprintln!(
        "[vq-train] reading PCM (channels={}, sample_rate={}) from {}",
        args.channels,
        args.sample_rate,
        args.input
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<stdin>".into())
    );
    let pcm = match args.input.as_ref() {
        Some(path) => {
            let f = File::open(path)?;
            read_pcm_s16le_to_mono(f, args.channels)?
        }
        None => read_pcm_s16le_to_mono(io::stdin().lock(), args.channels)?,
    };
    eprintln!("[vq-train] {} mono samples loaded", pcm.len());

    eprintln!(
        "[vq-train] extracting partition vectors (blocksize={}, dim={})",
        args.blocksize, args.dim
    );
    let vectors = extract_partition_vectors(&pcm, args.blocksize, args.dim);
    eprintln!("[vq-train] extracted {} training vectors", vectors.len());

    let cfg = TrainerConfig {
        blocksize: args.blocksize,
        dim: args.dim,
        codewords: args.codewords,
        ..TrainerConfig::default()
    };

    eprintln!(
        "[vq-train] training {} books, {} codewords each via LBG",
        args.books, args.codewords
    );
    let books = match train_books(&vectors, args.books, &cfg) {
        Ok(b) => b,
        Err(TrainerError::NotEnoughVectors {
            extracted,
            required,
        }) => {
            return Err(format!(
                "corpus too small: extracted {extracted} vectors per shard, need at least {required}. \
                 Increase corpus size or reduce --codewords / --books."
            )
            .into());
        }
        Err(e) => return Err(Box::new(e)),
    };

    eprintln!("[vq-train] writing {}", args.output.display());
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let f = File::create(&args.output)?;
    let mut w = BufWriter::new(f);
    emit_books_rs(&books, &mut w)?;
    eprintln!("[vq-train] done");
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_help();
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("vq-train: {e}");
            ExitCode::FAILURE
        }
    }
}

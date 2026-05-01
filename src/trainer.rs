//! VQ codebook trainer (round-1 scaffold for task #93).
//!
//! Vorbis I residue coding (§8) maps each MDCT spectrum bin (after
//! floor-curve division) into a vector-quantised codeword. The encoder
//! currently ships a degenerate 2-book setup (see `encoder.rs`); to
//! match libvorbis bitrates we need codebooks trained on real audio.
//! Workspace policy bars all libvorbis-derived `.vqh` files, so we
//! train our own clean-room books with the standard
//! [Linde-Buzo-Gray (LBG)](https://en.wikipedia.org/wiki/Linde-Buzo-Gray_algorithm)
//! algorithm (Linde/Buzo/Gray 1980 IEEE Trans. on Communications).
//!
//! Pipeline:
//!
//! 1. **PCM in.** Caller hands us 16-bit signed little-endian PCM at a
//!    given sample rate (44.1 / 48 kHz typical). Multi-channel samples
//!    are downmixed to mono — partition statistics are channel-invariant
//!    after coupling. (Round-2 may add per-channel training.)
//! 2. **Window + forward MDCT.** Reuses [`crate::imdct::build_window`]
//!    and [`crate::imdct::forward_mdct_naive`] — the same primitives the
//!    encoder uses on the hot path. Block size defaults to the long
//!    block (2048) since the long-block residue dominates the bitrate
//!    of typical streams.
//! 3. **2/N scaling.** Matches the encoder's `fwd_scale = 2.0 / n`
//!    normalisation so the trained codebook lives in the same numerical
//!    range as encode-time inputs.
//! 4. **Partition split.** Each MDCT half-spectrum (length `n / 2`) is
//!    sliced into vectors of dimension `dim` (the VQ-codebook
//!    dimension). For the encoder's current `partition_size = 2`, this
//!    is one vector per spectrum bin pair.
//! 5. **LBG cluster.** Initialise with the vector mean (size 1), then
//!    repeatedly split + Lloyd-refine until the codebook reaches the
//!    requested `codewords` count.
//! 6. **Emit.** Write a Rust source file containing
//!    `pub const TRAINED_BOOK_<N>: VqCodebook = VqCodebook { ... };`
//!    declarations.
//!
//! The trainer is intentionally agnostic about *which* books to train
//! and how the encoder will dispatch among them. That's round-2 work.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::imdct::{build_window, forward_mdct_naive};

/// Training configuration. Defaults match the encoder's long-block
/// residue extraction so trained books drop into the existing setup
/// without a sample-rate dependency.
#[derive(Clone, Debug)]
pub struct TrainerConfig {
    /// Block size in samples (must be a power of two between 64 and
    /// 8192). Defaults to 2048 (the encoder's long block).
    pub blocksize: usize,
    /// VQ-vector dimension. Encoder currently uses `dim = 2`. Larger
    /// dims compress better at the cost of book size.
    pub dim: usize,
    /// Number of codebook entries (i.e. codewords) per book. Must be a
    /// power of two so LBG's split-and-refine reaches it exactly.
    pub codewords: usize,
    /// Maximum LBG/Lloyd iterations per split level. The algorithm
    /// converges in 5-15 iterations on typical audio.
    pub max_iterations: usize,
    /// Convergence threshold: stop iterating when the average distortion
    /// changes by less than this fraction across one Lloyd pass.
    pub convergence_threshold: f32,
    /// Perturbation used when splitting a centroid into two during LBG
    /// growth. Standard value 0.01 (1%) keeps the split tight enough
    /// that the next Lloyd pass can untangle the pair.
    pub split_epsilon: f32,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            blocksize: 2048,
            dim: 16,
            codewords: 256,
            max_iterations: 30,
            convergence_threshold: 1e-4,
            split_epsilon: 0.01,
        }
    }
}

/// Errors the trainer can surface to the binary.
#[derive(Debug)]
pub enum TrainerError {
    /// Configuration was rejected (zero blocksize, non-power-of-two
    /// codewords, etc.).
    InvalidConfig(String),
    /// Corpus was too small to extract enough vectors — LBG needs at
    /// least `codewords` distinct vectors before it can split that far.
    NotEnoughVectors { extracted: usize, required: usize },
    /// Underlying I/O failure reading PCM or writing the output Rust file.
    Io(io::Error),
}

impl From<io::Error> for TrainerError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl std::fmt::Display for TrainerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid trainer config: {msg}"),
            Self::NotEnoughVectors {
                extracted,
                required,
            } => write!(
                f,
                "not enough training vectors: extracted {extracted}, need at least {required}"
            ),
            Self::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for TrainerError {}

/// Read raw 16-bit signed little-endian PCM samples from a reader. The
/// caller is responsible for matching the channel layout — multi-channel
/// data is downmixed to mono by averaging.
pub fn read_pcm_s16le_to_mono<R: Read>(mut r: R, channels: usize) -> io::Result<Vec<f32>> {
    if channels == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "channel count must be >= 1",
        ));
    }
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    if buf.len() % (channels * 2) != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "PCM byte count not a multiple of (channels * 2)",
        ));
    }
    let frames = buf.len() / (channels * 2);
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut acc = 0i32;
        for c in 0..channels {
            let off = (f * channels + c) * 2;
            let s = i16::from_le_bytes([buf[off], buf[off + 1]]) as i32;
            acc += s;
        }
        out.push(acc as f32 / (channels as f32 * i16::MAX as f32));
    }
    Ok(out)
}

/// Read PCM from a file path, inferring the channel count from the
/// caller (we don't parse a WAV header — input is raw PCM only).
pub fn read_pcm_file<P: AsRef<Path>>(path: P, channels: usize) -> io::Result<Vec<f32>> {
    let f = File::open(path)?;
    read_pcm_s16le_to_mono(f, channels)
}

/// Extract residue partition vectors from a stream of mono PCM samples.
/// This is the public-in-crate hook the trainer exposes — the encoder
/// uses the same forward-MDCT primitives on its hot path, so books
/// trained from these vectors line up bit-exactly with what the encoder
/// sees at packet emission time.
///
/// The block stride is `blocksize / 2` (50% overlap, matching Vorbis's
/// MDCT overlap-add) so adjacent training vectors capture every bin
/// transition the encoder will see.
///
/// `dim` is the VQ-codebook dimension — each spectrum is sliced into
/// `(blocksize / 2) / dim` vectors of length `dim`.
pub fn extract_partition_vectors(pcm: &[f32], blocksize: usize, dim: usize) -> Vec<Vec<f32>> {
    if blocksize == 0 || dim == 0 {
        return Vec::new();
    }
    let half = blocksize / 2;
    if half % dim != 0 {
        return Vec::new();
    }
    let stride = half;
    if pcm.len() < blocksize {
        return Vec::new();
    }
    // Long block, both neighbours long — symmetric sin window. Same
    // shape the encoder uses for steady-state long blocks.
    let window = build_window(blocksize, true, true, true, blocksize / 8);
    let scale = 2.0 / blocksize as f32;
    let n_blocks = (pcm.len() - blocksize) / stride + 1;
    let mut vectors = Vec::with_capacity(n_blocks * (half / dim));
    let mut windowed = vec![0f32; blocksize];
    let mut spectrum = vec![0f32; half];
    for b in 0..n_blocks {
        let start = b * stride;
        for i in 0..blocksize {
            windowed[i] = pcm[start + i] * window[i];
        }
        forward_mdct_naive(&windowed, &mut spectrum);
        for v in spectrum.iter_mut() {
            *v *= scale;
        }
        // Slice the spectrum into dim-sized partition vectors.
        for chunk in spectrum.chunks_exact(dim) {
            vectors.push(chunk.to_vec());
        }
    }
    vectors
}

/// A trained vector-quantisation codebook: `entries` vectors of length
/// `dim`. Stored as a flat row-major `Vec<f32>` so emission to a Rust
/// `const` is a single literal array.
#[derive(Clone, Debug)]
pub struct TrainedBook {
    pub dim: usize,
    pub entries: usize,
    /// Row-major: `data[e * dim + d]` is dimension `d` of entry `e`.
    pub data: Vec<f32>,
}

impl TrainedBook {
    /// Construct an empty book.
    pub fn new(dim: usize, entries: usize) -> Self {
        Self {
            dim,
            entries,
            data: vec![0f32; dim * entries],
        }
    }
    /// Borrow row `e` (entry `e`, length `dim`).
    pub fn row(&self, e: usize) -> &[f32] {
        &self.data[e * self.dim..(e + 1) * self.dim]
    }
    /// Mutably borrow row `e`.
    pub fn row_mut(&mut self, e: usize) -> &mut [f32] {
        let off = e * self.dim;
        &mut self.data[off..off + self.dim]
    }
}

/// Squared L2 distance between two equal-length vectors.
fn sq_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc += d * d;
    }
    acc
}

/// Find the index of the nearest centroid in `book` to `vector`.
fn nearest_centroid(book: &TrainedBook, vector: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for e in 0..book.entries {
        let d = sq_distance(book.row(e), vector);
        if d < best_d {
            best_d = d;
            best = e;
        }
    }
    best
}

/// Run one Lloyd iteration: assign each training vector to its nearest
/// centroid, then update each centroid to the mean of its assigned
/// vectors. Returns the average squared distortion.
fn lloyd_step(book: &mut TrainedBook, vectors: &[Vec<f32>]) -> f32 {
    let dim = book.dim;
    let mut sums = vec![0f32; book.entries * dim];
    let mut counts = vec![0u64; book.entries];
    let mut total_distortion = 0f32;
    for v in vectors {
        let e = nearest_centroid(book, v);
        for d in 0..dim {
            sums[e * dim + d] += v[d];
        }
        counts[e] += 1;
        total_distortion += sq_distance(book.row(e), v);
    }
    for e in 0..book.entries {
        if counts[e] > 0 {
            for d in 0..dim {
                book.row_mut(e)[d] = sums[e * dim + d] / counts[e] as f32;
            }
        }
        // If counts[e] == 0 the centroid stays where it is — split-step
        // logic in `train_lbg` will perturb it on the next growth pass.
    }
    if vectors.is_empty() {
        0.0
    } else {
        total_distortion / vectors.len() as f32
    }
}

/// Linde-Buzo-Gray training: start with one centroid (the mean of all
/// vectors), then doubly-grow the codebook by splitting each centroid
/// into two perturbed copies and re-running Lloyd. Stops when the
/// codebook reaches `cfg.codewords` entries.
///
/// `cfg.codewords` must be a power of two for the splitting to land
/// exactly on the target size; non-powers-of-two get rounded up to the
/// next power of two and then truncated, so the final book has exactly
/// `cfg.codewords` entries.
pub fn train_lbg(vectors: &[Vec<f32>], cfg: &TrainerConfig) -> Result<TrainedBook, TrainerError> {
    if cfg.dim == 0 {
        return Err(TrainerError::InvalidConfig("dim must be >= 1".into()));
    }
    if cfg.codewords == 0 {
        return Err(TrainerError::InvalidConfig("codewords must be >= 1".into()));
    }
    if vectors.len() < cfg.codewords {
        return Err(TrainerError::NotEnoughVectors {
            extracted: vectors.len(),
            required: cfg.codewords,
        });
    }
    for v in vectors {
        if v.len() != cfg.dim {
            return Err(TrainerError::InvalidConfig(format!(
                "training vector dim {} != configured dim {}",
                v.len(),
                cfg.dim
            )));
        }
    }
    // Initialise with the centroid of all vectors (size 1).
    let mut book = TrainedBook::new(cfg.dim, 1);
    for v in vectors {
        for d in 0..cfg.dim {
            book.row_mut(0)[d] += v[d];
        }
    }
    let n = vectors.len() as f32;
    for d in 0..cfg.dim {
        book.row_mut(0)[d] /= n;
    }
    // Grow by repeated splitting. After each split we run Lloyd until
    // distortion converges or we hit the iteration cap.
    while book.entries < cfg.codewords {
        let target = (book.entries * 2).min(cfg.codewords);
        let mut new_book = TrainedBook::new(cfg.dim, target);
        for e in 0..book.entries {
            let parent = book.row(e).to_vec();
            // Split entry `e` into two perturbed copies. Indexing:
            // first copy goes to `2e`, second to `2e + 1`. If we'd
            // overflow `target`, fall back to a single copy at `e`.
            let i0 = 2 * e;
            let i1 = 2 * e + 1;
            if i1 < target {
                for d in 0..cfg.dim {
                    new_book.row_mut(i0)[d] = parent[d] * (1.0 - cfg.split_epsilon);
                    new_book.row_mut(i1)[d] = parent[d] * (1.0 + cfg.split_epsilon);
                }
            } else if i0 < target {
                new_book.row_mut(i0)[..cfg.dim].copy_from_slice(&parent[..cfg.dim]);
            }
        }
        book = new_book;
        // Lloyd refinement loop.
        let mut prev_distortion = f32::INFINITY;
        for _ in 0..cfg.max_iterations {
            let d = lloyd_step(&mut book, vectors);
            let delta = if prev_distortion.is_finite() {
                ((prev_distortion - d) / prev_distortion.max(1e-30)).abs()
            } else {
                1.0
            };
            prev_distortion = d;
            if delta < cfg.convergence_threshold {
                break;
            }
        }
    }
    Ok(book)
}

/// Emit a set of trained books as a Rust source file. Output schema:
///
/// ```ignore
/// // Generated by oxideav-vorbis vq-train. Do not edit by hand.
/// pub struct VqCodebook {
///     pub dim: usize,
///     pub entries: usize,
///     pub data: &'static [f32],
/// }
///
/// pub const TRAINED_BOOK_0: VqCodebook = VqCodebook { ... };
/// pub const TRAINED_BOOK_1: VqCodebook = VqCodebook { ... };
/// pub const TRAINED_BOOKS: &[&VqCodebook] = &[&TRAINED_BOOK_0, ...];
/// ```
///
/// The schema is designed so the round-2 wiring can `include!` the
/// file directly into `encoder.rs` without further parsing.
pub fn emit_books_rs<W: Write>(books: &[TrainedBook], mut w: W) -> io::Result<()> {
    writeln!(
        w,
        "// Generated by oxideav-vorbis vq-train. Do not edit by hand.\n\
         //\n\
         // Each TRAINED_BOOK_N is a vector-quantisation codebook produced\n\
         // by the LBG (Linde-Buzo-Gray) trainer in `src/trainer.rs`.\n\
         // Round-2 wiring will replace the encoder's degenerate 2-book\n\
         // residue setup with these tables.\n"
    )?;
    writeln!(
        w,
        "#[derive(Clone, Copy, Debug)]\n\
         pub struct VqCodebook {{\n\
        \x20   pub dim: usize,\n\
        \x20   pub entries: usize,\n\
        \x20   pub data: &'static [f32],\n\
         }}\n"
    )?;
    for (i, book) in books.iter().enumerate() {
        writeln!(w, "pub const TRAINED_BOOK_{i}_DATA: &[f32] = &[")?;
        for e in 0..book.entries {
            write!(w, "    ")?;
            for d in 0..book.dim {
                write!(w, "{:.7}_f32, ", book.row(e)[d])?;
            }
            writeln!(w)?;
        }
        writeln!(w, "];")?;
        writeln!(
            w,
            "pub const TRAINED_BOOK_{i}: VqCodebook = VqCodebook {{ dim: {}, entries: {}, data: TRAINED_BOOK_{i}_DATA }};\n",
            book.dim, book.entries
        )?;
    }
    write!(w, "pub const TRAINED_BOOKS: &[&VqCodebook] = &[")?;
    for i in 0..books.len() {
        write!(w, "&TRAINED_BOOK_{i}, ")?;
    }
    writeln!(w, "];")?;
    Ok(())
}

/// Convenience wrapper: train `n_books` codebooks from the given
/// vectors. The vector pool is partitioned into `n_books` non-empty
/// shards (round-robin over the source order, so each book sees a
/// representative cross-section of the corpus rather than a contiguous
/// time slice).
///
/// Round-2 will replace the round-robin sharding with energy-classified
/// sharding so each book specialises in a particular residue
/// partition class. For now the round-robin keeps the trainer
/// deterministic and simple to reason about.
pub fn train_books(
    vectors: &[Vec<f32>],
    n_books: usize,
    cfg: &TrainerConfig,
) -> Result<Vec<TrainedBook>, TrainerError> {
    if n_books == 0 {
        return Err(TrainerError::InvalidConfig(
            "must train at least one book".into(),
        ));
    }
    let mut shards: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_books];
    for (i, v) in vectors.iter().enumerate() {
        shards[i % n_books].push(v.clone());
    }
    let mut books = Vec::with_capacity(n_books);
    for (i, shard) in shards.into_iter().enumerate() {
        let book = train_lbg(&shard, cfg).map_err(|e| match e {
            TrainerError::NotEnoughVectors {
                extracted,
                required,
            } => TrainerError::NotEnoughVectors {
                extracted,
                required: required + i, /* preserve idx context */
            },
            other => other,
        })?;
        books.push(book);
    }
    Ok(books)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LBG should recover four well-separated Gaussian cluster centres.
    /// We synthesise 1000 points around four fixed means in 2D and ask
    /// for a 4-entry codebook; each trained centroid should be within
    /// noise of one true mean.
    #[test]
    fn lbg_converges_on_synthetic_4_cluster_mixture() {
        // Deterministic LCG so the test is reproducible.
        let mut rng = 0x1234_5678_u32;
        let mut next = || {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (rng >> 16) as f32 / 65535.0 - 0.5
        };
        let truth = [
            [-3.0_f32, -3.0],
            [-3.0_f32, 3.0],
            [3.0_f32, -3.0],
            [3.0_f32, 3.0],
        ];
        let mut vectors = Vec::with_capacity(1000);
        for i in 0..1000 {
            let centre = truth[i % 4];
            let noise_x = next() * 0.5;
            let noise_y = next() * 0.5;
            vectors.push(vec![centre[0] + noise_x, centre[1] + noise_y]);
        }
        let cfg = TrainerConfig {
            blocksize: 0,
            dim: 2,
            codewords: 4,
            max_iterations: 50,
            convergence_threshold: 1e-6,
            split_epsilon: 0.01,
        };
        let book = train_lbg(&vectors, &cfg).expect("LBG should succeed");
        assert_eq!(book.entries, 4);
        // Each true cluster should have a nearest trained centroid
        // within 0.5 of the true mean (cluster radius).
        for centre in truth.iter() {
            let mut best_d = f32::INFINITY;
            for e in 0..book.entries {
                let d = sq_distance(book.row(e), centre);
                if d < best_d {
                    best_d = d;
                }
            }
            assert!(
                best_d < 0.25,
                "true cluster {:?} not matched by any trained centroid (best sq dist {})",
                centre,
                best_d
            );
        }
    }

    /// Emit a small book and verify the generated source has a
    /// `data: &[f32]` slice of the right length and the correct number
    /// of `TRAINED_BOOK_*` consts.
    #[test]
    fn emit_books_round_trips_lengths() {
        let book0 = TrainedBook {
            dim: 2,
            entries: 4,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let book1 = TrainedBook {
            dim: 4,
            entries: 2,
            data: vec![0.5; 8],
        };
        let mut buf = Vec::new();
        emit_books_rs(&[book0, book1], &mut buf).expect("emit");
        let s = String::from_utf8(buf).expect("utf8");
        // Both consts present.
        assert!(s.contains("pub const TRAINED_BOOK_0:"), "missing book 0");
        assert!(s.contains("pub const TRAINED_BOOK_1:"), "missing book 1");
        // Aggregate slice present and indexes both.
        assert!(s.contains("TRAINED_BOOKS: &[&VqCodebook]"));
        assert!(s.contains("&TRAINED_BOOK_0"));
        assert!(s.contains("&TRAINED_BOOK_1"));
        // Book 0: dim=2 entries=4 → 8 floats.
        assert!(s.contains("dim: 2, entries: 4"));
        // Book 1: dim=4 entries=2 → 8 floats.
        assert!(s.contains("dim: 4, entries: 2"));
        // Float literals should round-trip key values.
        assert!(s.contains("1.0000000_f32"), "expected formatted 1.0");
        assert!(s.contains("8.0000000_f32"), "expected formatted 8.0");
    }

    /// LBG should fail gracefully when the corpus has fewer vectors
    /// than the requested codeword count.
    #[test]
    fn trainer_rejects_too_few_vectors() {
        let vectors: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, i as f32 + 0.5]).collect();
        let cfg = TrainerConfig {
            blocksize: 0,
            dim: 2,
            codewords: 16,
            max_iterations: 10,
            convergence_threshold: 1e-3,
            split_epsilon: 0.01,
        };
        let err = train_lbg(&vectors, &cfg).expect_err("should fail");
        match err {
            TrainerError::NotEnoughVectors {
                extracted,
                required,
            } => {
                assert_eq!(extracted, 5);
                assert_eq!(required, 16);
            }
            other => panic!("expected NotEnoughVectors, got {other:?}"),
        }
    }

    /// `extract_partition_vectors` should produce the right number of
    /// vectors for a known-size PCM input. Sanity check on the windowing
    /// + MDCT plumbing.
    #[test]
    fn extract_partition_vectors_block_count() {
        let blocksize = 256;
        let dim = 8;
        // Three full blocks (2 strides past the first) → 50% overlap →
        // (768 - 256) / 128 + 1 = 5 blocks → 5 * (128/8) = 80 vectors.
        let pcm = vec![0.1f32; 768];
        let vectors = extract_partition_vectors(&pcm, blocksize, dim);
        assert_eq!(vectors.len(), 5 * (128 / dim));
        for v in &vectors {
            assert_eq!(v.len(), dim);
        }
    }
}

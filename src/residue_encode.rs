//! Residue VQ-encode cascade planner (Vorbis I §8.6.2 / §8.6.3 / §8.6.4
//! / §8.6.5, encode direction).
//!
//! The crate already carries the residue **WRITE** path: the
//! [`crate::encoder::write_residue_body`] primitive serialises a §8.6.2
//! residue body from a set of [`crate::encoder::ResidueVectorPlan`]s,
//! each of which carries the per-partition classifications and the
//! per-`(partition, pass)` value-codebook **entry-index** lists the
//! decoder reads back. The leaf [`crate::vq::quantize_vector`] turns one
//! target vector into the nearest codebook entry. What sat between them
//! — the glue that slices a real residue partition into the per-VQ-read
//! sub-vectors, walks the cascade stages feeding each stage's residual
//! to the next, and emits the entry-index lists — was the residue-side
//! encode followup named in [`crate::vq`] and the crate README. This
//! module is that glue.
//!
//! ## What the decoder does, and the inverse
//!
//! A residue decode (§8.6.2; see [`crate::residue`]) reconstructs one
//! decode vector's spectral residual by *accumulating*: for each
//! cascade pass `0..=7`, each partition whose classification selects a
//! value book at that stage reads a run of VQ codewords and **adds**
//! each unpacked vector into the residual (`v[idx] += val`). The
//! addressing of where each VQ vector's elements land differs by format:
//!
//! * **Format 0 (§8.6.3).** `step = n / dimensions`; VQ read `i`
//!   scatters its element `j` to partition-relative position
//!   `i + j*step`. So read `i` covers the strided positions
//!   `{i, i+step, i+2*step, …, i+(dims-1)*step}`.
//! * **Formats 1 and 2 (§8.6.4 / §8.6.5).** VQ read `k` appends its
//!   `dimensions` elements contiguously at partition-relative positions
//!   `[k*dims .. k*dims + dims)`, truncated at `n`. (Format 2 is
//!   "reducible to format 1" per §8.6.5; this module plans the single
//!   interleaved decode vector the same way.)
//!
//! Encoding inverts this. Given a partition's target residual scalars,
//! [`plan_partition_cascade`] walks the same `(stage, read)` schedule in
//! the **write** direction:
//!
//! 1. The partition's running residual starts as the target scalars.
//! 2. For each populated cascade stage (in pass order `0..=7`), the
//!    residual is **gathered** into the same per-VQ-read sub-vectors the
//!    decoder would scatter into (format-0 strided gather, format-1/2
//!    contiguous gather).
//! 3. Each sub-vector is quantised with [`crate::vq::quantize_vector`],
//!    yielding the entry index the decoder will read and the entry's
//!    decoded reconstruction.
//! 4. The reconstruction is **subtracted** from the running residual at
//!    exactly the positions the decoder would add it, so the next stage
//!    quantises the leftover error — the cascade refinement §8.6.2 step
//!    19's `+=` accumulation expresses.
//!
//! Because the gather positions in step 2 are the precise inverse of the
//! decoder's scatter, and the quantiser's reported `vector` is
//! bit-identical to `unpack_vector(book, entry)` (the decoder's
//! reconstruction), feeding the resulting entry lists back through
//! [`crate::encoder::write_residue_partition`] and the residue decoder
//! reproduces the chosen approximation exactly. The round-trip is
//! bit-exact on the *bitstream* (entry indices ↔ codewords); the
//! *PCM* it reconstructs is the nearest-entry approximation of the
//! target, which is the lossy quantisation Vorbis residue coding is.
//!
//! ## Scope
//!
//! Two layers live here. The lower one plans the value-codeword entry
//! lists *given the classifications* — the `partition_entries` field of a
//! [`crate::encoder::ResidueVectorPlan`]: [`plan_partition_cascade`] for
//! one partition, [`plan_vector_partition_entries`] for a whole vector's
//! pre-chosen classification row. The upper one **chooses** the
//! per-partition classification (which cascade column a partition takes,
//! hence its bit cost) directly from the spectrum:
//! [`plan_vector_classifications`] scores every candidate classification's
//! reconstruction distortion (via [`plan_partition_cascade_scored`]) and
//! keeps the closest, and [`plan_vector_residue`] is the top-of-stack
//! splitter producing the index-aligned `classifications` +
//! `partition_entries` a [`crate::encoder::ResidueVectorPlan`] holds with
//! no hand-supplied classifications. Threading the resulting plans into a
//! full packet is the existing [`crate::encoder::write_residue_body`] /
//! [`crate::encoder::write_audio_packet`] path.

use crate::codebook::{VorbisCodebook, VqLookup};
use crate::vq::{quantize_vector, QuantizeError};

/// Errors that can arise while planning a residue partition's cascade
/// (Vorbis I §8.6.2 / §8.6.3 / §8.6.4).
#[derive(Debug, Clone, PartialEq)]
pub enum ResidueEncodeError {
    /// `residue_type` was a value other than 0, 1, or 2. §8.6 defines
    /// only the three formats.
    UnsupportedResidueType(u16),
    /// `partition_size` was zero. §8.6.1 stores `residue_partition_size`
    /// as a `read 24 bits + 1` field, so the legal range starts at 1.
    ZeroPartitionSize,
    /// A cascade-stage value book had `dimensions == 0`. §8.6.3 divides
    /// the partition size by `[codebook_dimensions]` and §8.6.4 advances
    /// by one dimension per scalar; a zero-dimension book can never
    /// cover a partition.
    ZeroDimensions {
        /// The cascade stage (pass) index `0..=7` whose book is bad.
        pass: usize,
    },
    /// A cascade-stage value book had `codebook_lookup_type == 0`
    /// (entropy-only). §8.6.1: a book used in VQ context must carry a
    /// vector lookup, otherwise it yields no `[entry_temp]` vector.
    ScalarValueBook {
        /// The cascade stage (pass) index `0..=7` whose book is bad.
        pass: usize,
    },
    /// Format 0 requires the value book's `dimensions` to evenly divide
    /// the partition size (§8.6.3 step 1: `step = n / dimensions`, and
    /// steps 2..5 cover exactly `step × dimensions = n` scalars).
    Format0NotDivisible {
        /// The cascade stage (pass) index `0..=7` whose book is bad.
        pass: usize,
        /// `residue_partition_size` (the §8.6.3 `[n]`).
        partition_size: u32,
        /// The value book's `dimensions`.
        dimensions: u16,
    },
    /// The supplied residual slice's length did not equal
    /// `partition_size`. The planner quantises exactly the partition's
    /// scalars; a length mismatch is a caller bug.
    ResidualLengthMismatch {
        /// `partition_size`.
        expected: usize,
        /// The supplied slice length.
        actual: usize,
    },
    /// [`plan_vector_classifications`] was given an empty `value_books`
    /// table, so there was no classification to choose from. §8.6.1
    /// stores `residue_classifications` as `read 6 bits + 1`, so a real
    /// residue always has at least one classification.
    NoClassifications,
    /// [`plan_vector_classifications_rd`] was given a `lambda` that was
    /// NaN or negative. The Lagrange multiplier is a bits→distortion
    /// exchange rate, which must be a non-negative finite number; a
    /// negative value would reward spending bits, and NaN poisons every
    /// comparison.
    NonFiniteLambda(f64),
    /// A [`ResidueConfigCandidate`] declared `partitions_per_classword
    /// == 0`. §8.6.2 packs one classword per `classbook`-dimension
    /// partitions, a count that is always `>= 1`; a zero cannot describe a
    /// real residue header and would divide by zero in the classword
    /// accounting.
    ZeroPartitionsPerClassword {
        /// The candidate index whose `partitions_per_classword` was zero.
        config: usize,
    },
    /// A per-read [`quantize_vector`] call failed — almost always a
    /// fully-unused value book ([`QuantizeError::NoUsableEntries`]) or a
    /// hand-constructed book whose multiplicand table desyncs its
    /// declared shape. The cascade stage and the inner error are carried
    /// for context.
    Quantize {
        /// The cascade stage (pass) index `0..=7` whose quantise failed.
        pass: usize,
        /// The VQ-read ordinal within the partition that failed.
        read: usize,
        /// The underlying quantiser error.
        source: QuantizeError,
    },
    /// [`plan_vector_classifications_rd_weighted`] was given a weight
    /// slice whose length did not match the partition count implied by
    /// `scalars.len() / partition_size`.
    WeightLengthMismatch {
        /// The partition count (expected weight count).
        expected: usize,
        /// The supplied weight count.
        actual: usize,
    },
    /// A per-partition perceptual weight was NaN, infinite, or
    /// negative. A weight scales squared distortion, so it must be a
    /// finite non-negative factor.
    BadWeight {
        /// The partition whose weight was rejected.
        partition: usize,
        /// The offending weight.
        value: f64,
    },
}

impl core::fmt::Display for ResidueEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ResidueEncodeError::UnsupportedResidueType(t) => write!(
                f,
                "vorbis residue encode: unsupported residue_type {t} (§8.6 defines 0, 1, 2)"
            ),
            ResidueEncodeError::ZeroPartitionSize => write!(
                f,
                "vorbis residue encode: partition_size=0 (§8.6.1 stores it as read-24-bits + 1, so >= 1)"
            ),
            ResidueEncodeError::ZeroDimensions { pass } => write!(
                f,
                "vorbis residue encode: stage-{pass} value book dimensions=0 (§8.6.3/§8.6.4)"
            ),
            ResidueEncodeError::ScalarValueBook { pass } => write!(
                f,
                "vorbis residue encode: stage-{pass} value book has lookup_type 0 (§8.6.1 requires a value mapping in VQ context)"
            ),
            ResidueEncodeError::Format0NotDivisible {
                pass,
                partition_size,
                dimensions,
            } => write!(
                f,
                "vorbis residue encode: stage-{pass} format-0 partition_size {partition_size} not divisible by codebook dimensions {dimensions} (§8.6.3 step 1)"
            ),
            ResidueEncodeError::ResidualLengthMismatch { expected, actual } => write!(
                f,
                "vorbis residue encode: residual length {actual} != partition_size {expected} (§8.6.2)"
            ),
            ResidueEncodeError::NoClassifications => write!(
                f,
                "vorbis residue encode: empty value_books — no classification to choose from (§8.6.1: residue_classifications >= 1)"
            ),
            ResidueEncodeError::NonFiniteLambda(lambda) => write!(
                f,
                "vorbis residue encode: rate-distortion lambda {lambda} is not a finite non-negative exchange rate"
            ),
            ResidueEncodeError::ZeroPartitionsPerClassword { config } => write!(
                f,
                "vorbis residue encode: candidate {config} has partitions_per_classword=0 (§8.6.2 classbook dimension is >= 1)"
            ),
            ResidueEncodeError::Quantize { pass, read, source } => write!(
                f,
                "vorbis residue encode: stage-{pass} read-{read} quantise failed: {source}"
            ),
            ResidueEncodeError::WeightLengthMismatch { expected, actual } => write!(
                f,
                "vorbis residue encode: {actual} partition weights supplied for {expected} partitions"
            ),
            ResidueEncodeError::BadWeight { partition, value } => write!(
                f,
                "vorbis residue encode: partition-{partition} weight {value} is not a finite non-negative factor"
            ),
        }
    }
}

impl std::error::Error for ResidueEncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ResidueEncodeError::Quantize { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// The number of VQ codewords one residue partition body holds, given
/// the residue format, the partition size, and a value book's
/// dimensions — the encode-side mirror of
/// [`crate::encoder::residue_partition_codeword_count`] kept local so
/// the planner can size its gather without crossing module boundaries.
///
/// * **Format 0 (§8.6.3).** `step = n / dimensions` reads; dimensions
///   must divide `n`.
/// * **Formats 1 and 2 (§8.6.4 / §8.6.5).** `ceil(n / dimensions)`
///   reads; the surplus elements of the final read (when `dims` does
///   not divide `n`) are read-and-discarded by the decoder.
fn partition_reads(
    residue_type: u16,
    partition_size: u32,
    dimensions: u16,
    pass: usize,
) -> Result<usize, ResidueEncodeError> {
    let n = partition_size as usize;
    let dims = dimensions as usize;
    if residue_type == 0 {
        if n % dims != 0 {
            return Err(ResidueEncodeError::Format0NotDivisible {
                pass,
                partition_size,
                dimensions,
            });
        }
        Ok(n / dims)
    } else {
        Ok(n.div_ceil(dims))
    }
}

/// Gather the residual element index for VQ read `read`, element `j`,
/// in the format's addressing scheme — the inverse of the decoder's
/// per-partition scatter.
///
/// * **Format 0 (§8.6.3 steps 4..5).** Element `j` of read `i` sits at
///   partition-relative `i + j*step`.
/// * **Formats 1 and 2 (§8.6.4 steps 3..5).** Element `j` of read `k`
///   sits at partition-relative `k*dims + j`.
///
/// Returns `None` when the position falls outside the partition (only
/// possible for the final format-1/2 read when `dims` does not divide
/// `n` — those surplus elements the decoder discards, so the planner
/// targets them at `0.0`, the value an unused tail quantises toward
/// without disturbing the in-range error).
fn gather_index(
    residue_type: u16,
    read: usize,
    j: usize,
    step: usize,
    dims: usize,
    n: usize,
) -> Option<usize> {
    let idx = if residue_type == 0 {
        // §8.6.3: scatter element j to i + j*step.
        read + j * step
    } else {
        // §8.6.4: append contiguously, k*dims + j.
        read * dims + j
    };
    if idx < n {
        Some(idx)
    } else {
        None
    }
}

/// Plan one residue partition's cascade (Vorbis I §8.6.2 step 19 in the
/// write direction): quantise `residual` through every populated stage
/// of `stage_books`, returning the per-stage value-codebook entry-index
/// lists ready for [`crate::encoder::ResidueVectorPlan::partition_entries`].
///
/// `residual` is the partition's target spectral residual — exactly
/// `partition_size` scalars. `stage_books[pass]` is the value codebook
/// the decoder reads at cascade stage `pass` for this partition's
/// classification (`None` for an 'unused' stage, which §8.6.2 step 18
/// skips and which this planner emits `None` for, untouched).
/// `residue_type` selects the §8.6.3 strided or §8.6.4 contiguous
/// addressing.
///
/// The returned array is the `[Option<Vec<u32>>; 8]` row a
/// [`crate::encoder::ResidueVectorPlan`] holds per partition: `Some(list)`
/// at every stage `stage_books` has a book (the list length pinned by the
/// format's codeword count), `None` at every 'unused' stage. The running
/// residual is refined stage by stage — each stage quantises the leftover
/// `target − Σ(earlier reconstructions)`, so a multi-stage cascade
/// successively narrows the approximation exactly as the decoder's
/// additive accumulation widens it back.
///
/// # Errors
///
/// Returns a [`ResidueEncodeError`] for an unsupported `residue_type`, a
/// zero `partition_size`, a `residual` whose length is not
/// `partition_size`, a stage book with zero dimensions / no vector
/// lookup / a format-0 divisibility failure, or a per-read quantiser
/// failure (e.g. a fully-unused value book). Validation of each stage
/// precedes its emission; the running residual is consumed left to
/// right, so on error the stages already planned are discarded.
pub fn plan_partition_cascade(
    residual: &[f32],
    stage_books: &[Option<&VorbisCodebook>; 8],
    residue_type: u16,
    partition_size: u32,
) -> Result<[Option<Vec<u32>>; 8], ResidueEncodeError> {
    plan_partition_cascade_scored(residual, stage_books, residue_type, partition_size)
        .map(|scored| scored.entries)
}

/// A planned partition cascade together with the residual error it leaves.
///
/// [`plan_partition_cascade`] discards the leftover residual once it has
/// the entry lists. The classification chooser
/// ([`plan_vector_classifications`]) needs that leftover to compare
/// candidate classifications, so [`plan_partition_cascade_scored`]
/// returns it: `error_sq` is the squared-Euclidean norm of the residual
/// remaining after every cascade stage has subtracted its
/// reconstruction — exactly the distortion the decoder will reconstruct
/// (`target − Σ reconstructions`). A smaller `error_sq` means a closer
/// approximation of the partition's target scalars.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredPartitionCascade {
    /// The per-stage value-codebook entry-index lists, identical to what
    /// [`plan_partition_cascade`] returns.
    pub entries: [Option<Vec<u32>>; 8],
    /// The squared-Euclidean norm of the residual left after the cascade
    /// — the partition's reconstruction distortion. `0.0` for an
    /// all-'unused' cascade only when the target itself is all-zero.
    pub error_sq: f64,
    /// The number of populated cascade stages — a proxy for the bit cost
    /// (every populated stage emits at least one value codeword). Used as
    /// the chooser's tie-break: at equal distortion, prefer the cheaper
    /// classification.
    pub populated_stages: usize,
    /// The exact value-codeword bit cost the cascade emits for this
    /// partition: the sum of `book.codeword_lengths[entry]` over every
    /// entry the cascade chose, across all populated stages. This is the
    /// *value-codeword* contribution only — the §8.6.2 classword (the
    /// per-partition classification index packed through the residue's
    /// classbook) is amortised at the vector level (one classword can
    /// cover several partitions in formats 1 / 2), so it is scored
    /// separately by the rate-aware vector chooser rather than charged
    /// here. `0` for an all-'unused' cascade (no value codewords emitted).
    ///
    /// Because [`crate::vq::quantize_vector`] never selects a sparse
    /// (`UNUSED_ENTRY`) entry, every charged length is in `1..=32`, so the
    /// sum is the precise number of bits the §8.6.2 write path packs for
    /// this partition's value codewords — the rate term a rate-distortion
    /// classification chooser trades against `error_sq`.
    pub bit_cost: u64,
}

/// [`plan_partition_cascade`] but additionally reporting the residual
/// distortion the cascade leaves (see [`ScoredPartitionCascade`]). The
/// entry lists are bit-identical to the unscored routine; this variant
/// just keeps the leftover residual instead of discarding it.
///
/// # Errors
///
/// Identical to [`plan_partition_cascade`].
pub fn plan_partition_cascade_scored(
    residual: &[f32],
    stage_books: &[Option<&VorbisCodebook>; 8],
    residue_type: u16,
    partition_size: u32,
) -> Result<ScoredPartitionCascade, ResidueEncodeError> {
    if residue_type > 2 {
        return Err(ResidueEncodeError::UnsupportedResidueType(residue_type));
    }
    if partition_size == 0 {
        return Err(ResidueEncodeError::ZeroPartitionSize);
    }
    if residual.len() != partition_size as usize {
        return Err(ResidueEncodeError::ResidualLengthMismatch {
            expected: partition_size as usize,
            actual: residual.len(),
        });
    }

    let n = partition_size as usize;
    // The running residual the cascade refines — starts as the target.
    let mut work = residual.to_vec();
    let mut out: [Option<Vec<u32>>; 8] = Default::default();
    let mut populated_stages = 0usize;
    let mut bit_cost = 0u64;

    for (pass, slot) in stage_books.iter().enumerate() {
        // §8.6.2 step 18: 'unused' stages read and write nothing.
        let Some(book) = slot else { continue };

        let dims = book.dimensions as usize;
        if dims == 0 {
            return Err(ResidueEncodeError::ZeroDimensions { pass });
        }
        // §8.6.1: a VQ-context book must carry a vector lookup.
        if matches!(book.lookup, VqLookup::None) {
            return Err(ResidueEncodeError::ScalarValueBook { pass });
        }
        let reads = partition_reads(residue_type, partition_size, book.dimensions, pass)?;
        // Format 0's step stride; unused (0) for the contiguous formats.
        let step = if residue_type == 0 { n / dims } else { 0 };

        let mut entries = Vec::with_capacity(reads);
        for read in 0..reads {
            // Gather this read's target sub-vector from the running
            // residual at the exact positions the decoder will scatter
            // back into (surplus tail elements → 0.0).
            let mut target = Vec::with_capacity(dims);
            for j in 0..dims {
                let v = match gather_index(residue_type, read, j, step, dims, n) {
                    Some(idx) => work[idx],
                    None => 0.0,
                };
                target.push(v);
            }

            // Quantise: pick the nearest codebook entry to the target.
            let q = quantize_vector(book, &target)
                .map_err(|source| ResidueEncodeError::Quantize { pass, read, source })?;

            // Subtract the chosen entry's reconstruction from the running
            // residual at the same positions, so the next stage refines
            // the leftover error (§8.6.2's additive cascade, inverted).
            for (j, &recon) in q.vector.iter().enumerate() {
                if let Some(idx) = gather_index(residue_type, read, j, step, dims, n) {
                    work[idx] -= recon;
                }
            }
            // Charge the chosen entry's value-codeword length. The
            // quantiser only returns 'used' entries, so the length is in
            // `1..=32`; a defensive `.get` keeps a malformed hand-built
            // book (a too-short `codeword_lengths`) from panicking — such an
            // entry contributes `0` bits, which can only *under*-charge,
            // never over-charge, the rate estimate.
            let len = book
                .codeword_lengths
                .get(q.entry as usize)
                .copied()
                .unwrap_or(0);
            bit_cost += u64::from(len);
            entries.push(q.entry);
        }
        out[pass] = Some(entries);
        populated_stages += 1;
    }

    // The residual remaining after every stage has subtracted its
    // reconstruction is the decoder's reconstruction error (the format-1/2
    // surplus tail elements are decoder-discarded, so they never appear in
    // `work` past `n` — `work` is length `n` throughout).
    let mut error_sq = 0.0f64;
    for &w in &work {
        error_sq += f64::from(w) * f64::from(w);
    }

    Ok(ScoredPartitionCascade {
        entries: out,
        error_sq,
        populated_stages,
        bit_cost,
    })
}

/// Plan the `partition_entries` field of a full decode vector's
/// [`crate::encoder::ResidueVectorPlan`] (Vorbis I §8.6.2 in the write
/// direction): for each partition, gather its `partition_size` scalars
/// from `scalars`, look up the classification's cascade row in
/// `value_books`, and quantise the cascade with [`plan_partition_cascade`].
///
/// `scalars` is the decode vector's spectral residual over the
/// `[residue_begin, residue_end)` window the residue covers — exactly
/// `classifications.len() × partition_size` values, laid out
/// partition-major (partition `p` occupies `scalars[p*ps .. (p+1)*ps]`).
/// `classifications[p]` is the classification partition `p` takes (the
/// caller's psychoacoustic choice, the same value the matching
/// [`crate::encoder::ResidueVectorPlan::classifications`] entry will
/// carry). `value_books[class][pass]` is the resolved value codebook for
/// classification `class` at cascade stage `pass` (`None` for an unused
/// stage) — the cascade the residue header's `residue_books[class]`
/// describes, already mapped to codebooks.
///
/// The returned `Vec` is one `[Option<Vec<u32>>; 8]` row per partition,
/// in partition order — exactly
/// [`crate::encoder::ResidueVectorPlan::partition_entries`]. Each
/// partition is quantised independently (the decoder accumulates each
/// partition into a disjoint slice of the vector), but within a
/// partition the cascade refines stage to stage.
///
/// # Errors
///
/// Returns a [`ResidueEncodeError`] for an unsupported `residue_type`, a
/// zero `partition_size`, a `scalars` length that is not
/// `classifications.len() × partition_size`, or any per-partition cascade
/// failure (carried verbatim from [`plan_partition_cascade`]). A
/// classification that indexes outside `value_books` is treated as an
/// all-'unused' cascade (mirroring the decoder's padding of short
/// `residue_books` rows), not an error.
pub fn plan_vector_partition_entries(
    scalars: &[f32],
    classifications: &[u32],
    value_books: &[[Option<&VorbisCodebook>; 8]],
    residue_type: u16,
    partition_size: u32,
) -> Result<Vec<[Option<Vec<u32>>; 8]>, ResidueEncodeError> {
    if residue_type > 2 {
        return Err(ResidueEncodeError::UnsupportedResidueType(residue_type));
    }
    if partition_size == 0 {
        return Err(ResidueEncodeError::ZeroPartitionSize);
    }
    let ps = partition_size as usize;
    let expected_len = classifications.len().saturating_mul(ps);
    if scalars.len() != expected_len {
        return Err(ResidueEncodeError::ResidualLengthMismatch {
            expected: expected_len,
            actual: scalars.len(),
        });
    }

    let empty_row: [Option<&VorbisCodebook>; 8] = [None; 8];
    let mut rows = Vec::with_capacity(classifications.len());
    for (p, &class) in classifications.iter().enumerate() {
        // A classification with no configured row (e.g. an unused
        // classification slot) decodes nothing — its cascade is
        // all-'unused', so the planner emits an all-`None` row without
        // touching the scalars. Mirrors the decoder's padding of short
        // `residue_books` rows.
        let stage_books = value_books.get(class as usize).unwrap_or(&empty_row);
        let partition_scalars = &scalars[p * ps..(p + 1) * ps];
        let row =
            plan_partition_cascade(partition_scalars, stage_books, residue_type, partition_size)?;
        rows.push(row);
    }
    Ok(rows)
}

/// One partition's chosen classification together with the cascade plan
/// and the distortion that choice achieves — the output unit of
/// [`plan_vector_classifications`].
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionClassChoice {
    /// The classification index chosen for this partition — the value
    /// that goes into the matching
    /// [`crate::encoder::ResidueVectorPlan::classifications`] slot.
    pub classification: u32,
    /// The cascade entry-index lists for the chosen classification — the
    /// matching [`crate::encoder::ResidueVectorPlan::partition_entries`]
    /// row.
    pub entries: [Option<Vec<u32>>; 8],
    /// The squared reconstruction distortion the chosen classification
    /// leaves on this partition.
    pub error_sq: f64,
    /// The value-codeword bit cost of the chosen classification's cascade
    /// for this partition — [`ScoredPartitionCascade::bit_cost`] of the
    /// winning candidate. The pure-distortion chooser
    /// ([`plan_vector_classifications`]) records it for downstream
    /// rate accounting; the rate-distortion chooser
    /// ([`plan_vector_classifications_rd`]) actively trades it against
    /// `error_sq`.
    pub bit_cost: u64,
}

/// Choose the per-partition classification *and* plan its cascade for one
/// residue decode vector (Vorbis I §8.6.2, encode direction) — the
/// classification-selection layer that sat open above
/// [`plan_vector_partition_entries`].
///
/// [`plan_vector_partition_entries`] takes the per-partition
/// classifications as a *given* (the caller's psychoacoustic choice) and
/// fills in the entry lists. This routine closes that gap from the
/// distortion side: for every partition it tries **each candidate
/// classification** in `value_books`, plans its cascade with
/// [`plan_partition_cascade_scored`], and keeps the classification whose
/// cascade reconstructs the partition's target scalars most closely
/// (minimum squared error). Ties in distortion are broken toward the
/// classification with **fewer populated cascade stages** (the cheaper
/// encoding — every populated stage emits value codewords), and then
/// toward the **lower classification index** (deterministic).
///
/// `scalars` is the decode vector's spectral residual over the residue
/// window, partition-major (partition `p` occupies
/// `scalars[p*ps .. (p+1)*ps]`); its length must be a multiple of
/// `partition_size`, and the number of partitions it implies is the
/// returned `Vec`'s length. `value_books[class][pass]` is the resolved
/// value codebook for classification `class` at cascade stage `pass`
/// (`None` for an unused stage) — the same `[[Option<&VorbisCodebook>; 8]]`
/// table [`plan_vector_partition_entries`] consumes, one row per
/// classification the residue header configures. `residue_type` selects
/// the §8.6.3 strided or §8.6.4 contiguous addressing.
///
/// The returned `Vec` is one [`PartitionClassChoice`] per partition, in
/// partition order. Splitting it into the parallel `classifications` /
/// `entries` arrays yields a ready-to-serialise
/// [`crate::encoder::ResidueVectorPlan`]; the convenience splitter is
/// [`plan_vector_residue`].
///
/// Every classification a partition could take is feasible by
/// construction: a classification whose cascade is all-'unused' (or whose
/// row is shorter than the configured count) reconstructs the partition
/// as all-zero, leaving `error_sq == ‖target‖²`. So the chooser always
/// finds at least one usable classification when `value_books` is
/// non-empty.
///
/// # Errors
///
/// Returns a [`ResidueEncodeError`] for an unsupported `residue_type`, a
/// zero `partition_size`, a `scalars` length that is not a multiple of
/// `partition_size`, an empty `value_books` (no classification to choose
/// from), or any per-classification cascade failure carried verbatim from
/// [`plan_partition_cascade_scored`] (e.g. a stage book with zero
/// dimensions or no vector lookup — a malformed header the chooser cannot
/// silently skip past).
pub fn plan_vector_classifications(
    scalars: &[f32],
    value_books: &[[Option<&VorbisCodebook>; 8]],
    residue_type: u16,
    partition_size: u32,
) -> Result<Vec<PartitionClassChoice>, ResidueEncodeError> {
    if residue_type > 2 {
        return Err(ResidueEncodeError::UnsupportedResidueType(residue_type));
    }
    if partition_size == 0 {
        return Err(ResidueEncodeError::ZeroPartitionSize);
    }
    if value_books.is_empty() {
        return Err(ResidueEncodeError::NoClassifications);
    }
    let ps = partition_size as usize;
    if scalars.len() % ps != 0 {
        return Err(ResidueEncodeError::ResidualLengthMismatch {
            // The expected length is the nearest multiple at or below the
            // supplied length; report the supplied length's remainder via
            // the canonical "not a partition multiple" framing.
            expected: (scalars.len() / ps) * ps,
            actual: scalars.len(),
        });
    }
    let num_partitions = scalars.len() / ps;

    let mut choices = Vec::with_capacity(num_partitions);
    for p in 0..num_partitions {
        let partition_scalars = &scalars[p * ps..(p + 1) * ps];

        let mut best: Option<PartitionClassChoice> = None;
        let mut best_stages = usize::MAX;
        for (class, stage_books) in value_books.iter().enumerate() {
            let scored = plan_partition_cascade_scored(
                partition_scalars,
                stage_books,
                residue_type,
                partition_size,
            )?;

            // Lexicographic preference: (distortion ↑, populated stages ↑,
            // classification index ↑). `<` on f64 keeps the first-seen
            // (lower-index) candidate on an exact distortion-and-stage tie.
            let replace = match &best {
                None => true,
                Some(cur) => {
                    scored.error_sq < cur.error_sq
                        || (scored.error_sq == cur.error_sq
                            && scored.populated_stages < best_stages)
                }
            };
            if replace {
                best_stages = scored.populated_stages;
                best = Some(PartitionClassChoice {
                    classification: class as u32,
                    entries: scored.entries,
                    error_sq: scored.error_sq,
                    bit_cost: scored.bit_cost,
                });
            }
        }

        // `value_books` is non-empty, so at least one classification was
        // scored; `best` is `Some`.
        choices.push(best.expect("value_books non-empty ⇒ a classification was chosen"));
    }

    Ok(choices)
}

/// Choose the per-partition classification by a **rate-distortion**
/// criterion (Vorbis I §8.6.2, encode direction): for every partition,
/// try each candidate classification, plan its cascade with
/// [`plan_partition_cascade_scored`], and keep the one minimising the
/// Lagrangian cost `error_sq + lambda · bit_cost`.
///
/// This is the rate-aware sibling of [`plan_vector_classifications`].
/// That routine minimises reconstruction distortion alone (with a
/// stage-count tie-break), which always prefers the densest cascade that
/// happens to reconstruct best — it never trades a little distortion for
/// a cheaper encoding. A real encoder operating to a bit budget must make
/// that trade: a partition that is *almost* as well reconstructed by a
/// short, cheap cascade should take the cheap one. The classic lever is
/// the Lagrange multiplier `lambda` (bits-to-distortion exchange rate):
///
/// * `lambda == 0.0` reduces *exactly* to [`plan_vector_classifications`]
///   — pure distortion, with the same `(distortion ↑, populated stages ↑,
///   classification index ↑)` tie-break. (At `lambda == 0` the cost is
///   `error_sq`; ties on cost fall through to the same secondary keys.)
/// * Larger `lambda` weights rate more heavily, pulling the choice toward
///   cheaper (fewer-bit) classifications even at some distortion cost — a
///   higher-`lambda` pass is the encoder's response to a tighter bit
///   budget.
///
/// The `bit_cost` charged is the per-partition value-codeword cost
/// ([`ScoredPartitionCascade::bit_cost`]); the §8.6.2 classword is
/// amortised across partitions at the vector level and so is not part of
/// the per-partition trade here.
///
/// Tie-break, applied to the Lagrangian cost: lower cost wins; on an
/// exact cost tie, fewer populated stages (cheaper); then lower
/// classification index (deterministic). The returned
/// [`PartitionClassChoice`]s carry the chosen `error_sq` and `bit_cost`
/// unchanged, so a caller can recompute or audit the trade.
///
/// `scalars`, `value_books`, `residue_type`, and `partition_size` have
/// the identical meaning and validation as [`plan_vector_classifications`].
///
/// # Errors
///
/// Identical to [`plan_vector_classifications`], plus
/// [`ResidueEncodeError::NonFiniteLambda`] if `lambda` is NaN or negative
/// (a negative exchange rate would reward *spending* bits, which is never
/// the intended trade).
pub fn plan_vector_classifications_rd(
    scalars: &[f32],
    value_books: &[[Option<&VorbisCodebook>; 8]],
    residue_type: u16,
    partition_size: u32,
    lambda: f64,
) -> Result<Vec<PartitionClassChoice>, ResidueEncodeError> {
    plan_vector_classifications_rd_impl(
        scalars,
        value_books,
        residue_type,
        partition_size,
        lambda,
        None,
    )
}

/// Choose the per-partition classification by a **perceptually
/// weighted** rate-distortion criterion (Vorbis I §8.6.2, encode
/// direction): for every partition `p`, minimise the Lagrangian
/// `weights[p] · error_sq + lambda · bit_cost`.
///
/// This is the noise-to-mask-aware sibling of
/// [`plan_vector_classifications_rd`]. That routine charges every
/// partition's squared residue-domain error equally — but the decoder
/// multiplies the residue by the rendered floor (§4.3.6), so equal
/// residue error is *not* equally audible: noise in a partition whose
/// floor rides far above the masking threshold surfaces loudly, while
/// noise in a fully-masked partition is inaudible at any residue error.
/// Scaling each partition's distortion by a caller-supplied weight —
/// canonically [`crate::psy::residue_partition_weights`]'s
/// mean-normalised `(floor/threshold)²` factors — turns the Lagrangian
/// into an approximate NMR-vs-bits trade: heavily weighted (audible)
/// partitions attract denser cascades, lightly weighted (masked)
/// partitions give their bits up first as `lambda` rises.
///
/// `weights` must hold one finite non-negative factor per partition
/// (`scalars.len() / partition_size` entries). All-`1.0` weights make
/// the choice **identical** to [`plan_vector_classifications_rd`]
/// (bit-for-bit — the cost arithmetic degenerates to the unweighted
/// form). A `0.0` weight makes that partition's choice rate-only: the
/// cheapest cascade wins regardless of distortion (with the same
/// stage-count / index tie-breaks).
///
/// The returned [`PartitionClassChoice`]s carry the **unweighted**
/// `error_sq` (the physical squared distortion the decoder will
/// reconstruct) and the exact `bit_cost`, so a caller can re-derive
/// the weighted cost as `weights[p] · error_sq + lambda · bit_cost`.
///
/// # Errors
///
/// Identical to [`plan_vector_classifications_rd`], plus
/// [`ResidueEncodeError::WeightLengthMismatch`] for a weight slice
/// that does not match the partition count and
/// [`ResidueEncodeError::BadWeight`] for a NaN/±∞/negative weight.
pub fn plan_vector_classifications_rd_weighted(
    scalars: &[f32],
    value_books: &[[Option<&VorbisCodebook>; 8]],
    residue_type: u16,
    partition_size: u32,
    lambda: f64,
    weights: &[f64],
) -> Result<Vec<PartitionClassChoice>, ResidueEncodeError> {
    plan_vector_classifications_rd_impl(
        scalars,
        value_books,
        residue_type,
        partition_size,
        lambda,
        Some(weights),
    )
}

/// Shared core of the unweighted and weighted rate-distortion
/// classification choosers. `weights == None` charges every partition's
/// distortion at factor `1.0` (the unweighted Lagrangian).
fn plan_vector_classifications_rd_impl(
    scalars: &[f32],
    value_books: &[[Option<&VorbisCodebook>; 8]],
    residue_type: u16,
    partition_size: u32,
    lambda: f64,
    weights: Option<&[f64]>,
) -> Result<Vec<PartitionClassChoice>, ResidueEncodeError> {
    if residue_type > 2 {
        return Err(ResidueEncodeError::UnsupportedResidueType(residue_type));
    }
    if partition_size == 0 {
        return Err(ResidueEncodeError::ZeroPartitionSize);
    }
    if value_books.is_empty() {
        return Err(ResidueEncodeError::NoClassifications);
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(ResidueEncodeError::NonFiniteLambda(lambda));
    }
    let ps = partition_size as usize;
    if scalars.len() % ps != 0 {
        return Err(ResidueEncodeError::ResidualLengthMismatch {
            expected: (scalars.len() / ps) * ps,
            actual: scalars.len(),
        });
    }
    let num_partitions = scalars.len() / ps;

    if let Some(w) = weights {
        if w.len() != num_partitions {
            return Err(ResidueEncodeError::WeightLengthMismatch {
                expected: num_partitions,
                actual: w.len(),
            });
        }
        if let Some(p) = w.iter().position(|v| !v.is_finite() || *v < 0.0) {
            return Err(ResidueEncodeError::BadWeight {
                partition: p,
                value: w[p],
            });
        }
    }

    let mut choices = Vec::with_capacity(num_partitions);
    for p in 0..num_partitions {
        let partition_scalars = &scalars[p * ps..(p + 1) * ps];
        let weight = weights.map_or(1.0, |w| w[p]);

        let mut best: Option<PartitionClassChoice> = None;
        let mut best_cost = f64::INFINITY;
        let mut best_stages = usize::MAX;
        for (class, stage_books) in value_books.iter().enumerate() {
            let scored = plan_partition_cascade_scored(
                partition_scalars,
                stage_books,
                residue_type,
                partition_size,
            )?;

            // Lagrangian rate-distortion cost. `bit_cost` is exact bits;
            // `error_sq` is squared sample distortion (scaled by the
            // partition's perceptual weight, 1.0 when unweighted);
            // `lambda` is the bits→distortion exchange rate. The
            // accumulation is in f64 to keep the comparison stable
            // across a wide dynamic range.
            let cost = weight * scored.error_sq + lambda * scored.bit_cost as f64;

            // Lexicographic preference: (cost ↑, populated stages ↑,
            // classification index ↑). `<` keeps the first-seen
            // (lower-index) candidate on an exact cost-and-stage tie.
            let replace = match &best {
                None => true,
                Some(_) => {
                    cost < best_cost || (cost == best_cost && scored.populated_stages < best_stages)
                }
            };
            if replace {
                best_cost = cost;
                best_stages = scored.populated_stages;
                best = Some(PartitionClassChoice {
                    classification: class as u32,
                    entries: scored.entries,
                    error_sq: scored.error_sq,
                    bit_cost: scored.bit_cost,
                });
            }
        }

        choices.push(best.expect("value_books non-empty ⇒ a classification was chosen"));
    }

    Ok(choices)
}

/// Plan a complete residue decode vector from raw spectral residual
/// (Vorbis I §8.6.2, encode direction): choose each partition's
/// classification via [`plan_vector_classifications`] and assemble the
/// result into the [`crate::encoder::ResidueVectorPlan`]-shaped parallel
/// arrays the residue WRITE path consumes.
///
/// This is the top of the residue-encode stack: it turns a vector's
/// target spectral residual directly into the `classifications` +
/// `partition_entries` a [`crate::encoder::ResidueVectorPlan`] holds, with
/// no hand-supplied classifications. The two returned `Vec`s are
/// index-aligned (partition `p`'s classification is `classifications[p]`
/// and its cascade is `partition_entries[p]`), so a caller builds the
/// plan as `ResidueVectorPlan { classifications, partition_entries }`.
///
/// # Errors
///
/// Identical to [`plan_vector_classifications`].
#[allow(clippy::type_complexity)]
pub fn plan_vector_residue(
    scalars: &[f32],
    value_books: &[[Option<&VorbisCodebook>; 8]],
    residue_type: u16,
    partition_size: u32,
) -> Result<(Vec<u32>, Vec<[Option<Vec<u32>>; 8]>), ResidueEncodeError> {
    let choices = plan_vector_classifications(scalars, value_books, residue_type, partition_size)?;
    let mut classifications = Vec::with_capacity(choices.len());
    let mut partition_entries = Vec::with_capacity(choices.len());
    for choice in choices {
        classifications.push(choice.classification);
        partition_entries.push(choice.entries);
    }
    Ok((classifications, partition_entries))
}

/// A residue decode-vector plan together with the aggregate
/// rate-distortion figures it achieves — the scored output of
/// [`plan_vector_residue_rd`] and the unit
/// [`select_residue_config`] compares candidates by.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredVectorResidue {
    /// The per-partition classification indices, partition order — the
    /// `classifications` field of a [`crate::encoder::ResidueVectorPlan`].
    pub classifications: Vec<u32>,
    /// The per-partition cascade entry-index lists, partition order — the
    /// `partition_entries` field of a [`crate::encoder::ResidueVectorPlan`].
    pub partition_entries: Vec<[Option<Vec<u32>>; 8]>,
    /// Total reconstruction distortion summed over every partition: the
    /// `Σ error_sq` the chosen plan leaves. This is the §8.6.6 decode
    /// vector's squared reconstruction error against the target residual.
    pub total_error_sq: f64,
    /// Total value-codeword bit cost summed over every partition: the
    /// `Σ bit_cost`. Excludes the classword bits (one classword per
    /// `partitions_per_classword` partitions, charged separately by
    /// [`select_residue_config`] because it depends on the residue
    /// header's classbook, not the value cascade).
    pub total_value_bits: u64,
}

/// Plan a complete residue decode vector by the rate-distortion
/// criterion and report its aggregate figures (Vorbis I §8.6.2, encode
/// direction) — the scored, rate-aware top of the residue-encode stack.
///
/// This is to [`plan_vector_residue`] what
/// [`plan_vector_classifications_rd`] is to
/// [`plan_vector_classifications`]: it chooses each partition's
/// classification by minimising `error_sq + lambda · bit_cost`, then
/// assembles the index-aligned `classifications` + `partition_entries`
/// the WRITE path consumes — and additionally returns the summed
/// distortion and value-bit cost so a caller can compare whole-vector
/// candidates.
///
/// `lambda == 0` recovers the pure-distortion plan
/// ([`plan_vector_residue`] augmented with the aggregate figures).
///
/// # Errors
///
/// Identical to [`plan_vector_classifications_rd`].
pub fn plan_vector_residue_rd(
    scalars: &[f32],
    value_books: &[[Option<&VorbisCodebook>; 8]],
    residue_type: u16,
    partition_size: u32,
    lambda: f64,
) -> Result<ScoredVectorResidue, ResidueEncodeError> {
    let choices =
        plan_vector_classifications_rd(scalars, value_books, residue_type, partition_size, lambda)?;
    let mut classifications = Vec::with_capacity(choices.len());
    let mut partition_entries = Vec::with_capacity(choices.len());
    let mut total_error_sq = 0.0f64;
    let mut total_value_bits = 0u64;
    for choice in choices {
        total_error_sq += choice.error_sq;
        total_value_bits += choice.bit_cost;
        classifications.push(choice.classification);
        partition_entries.push(choice.entries);
    }
    Ok(ScoredVectorResidue {
        classifications,
        partition_entries,
        total_error_sq,
        total_value_bits,
    })
}

/// Plan a complete residue decode vector by the **perceptually
/// weighted** rate-distortion criterion and report its aggregate
/// figures (Vorbis I §8.6.2, encode direction).
///
/// This is to [`plan_vector_residue_rd`] what
/// [`plan_vector_classifications_rd_weighted`] is to
/// [`plan_vector_classifications_rd`]: each partition's classification
/// minimises `weights[p] · error_sq + lambda · bit_cost`, and the
/// index-aligned `classifications` + `partition_entries` are assembled
/// for the WRITE path. The returned
/// [`ScoredVectorResidue::total_error_sq`] is the **unweighted**
/// physical distortion sum (the same quantity the unweighted planner
/// reports, so rate/SNR measurements stay comparable);
/// `total_value_bits` is exact. All-`1.0` weights reproduce
/// [`plan_vector_residue_rd`] bit-for-bit.
///
/// # Errors
///
/// Identical to [`plan_vector_classifications_rd_weighted`].
pub fn plan_vector_residue_rd_weighted(
    scalars: &[f32],
    value_books: &[[Option<&VorbisCodebook>; 8]],
    residue_type: u16,
    partition_size: u32,
    lambda: f64,
    weights: &[f64],
) -> Result<ScoredVectorResidue, ResidueEncodeError> {
    let choices = plan_vector_classifications_rd_weighted(
        scalars,
        value_books,
        residue_type,
        partition_size,
        lambda,
        weights,
    )?;
    let mut classifications = Vec::with_capacity(choices.len());
    let mut partition_entries = Vec::with_capacity(choices.len());
    let mut total_error_sq = 0.0f64;
    let mut total_value_bits = 0u64;
    for choice in choices {
        total_error_sq += choice.error_sq;
        total_value_bits += choice.bit_cost;
        classifications.push(choice.classification);
        partition_entries.push(choice.entries);
    }
    Ok(ScoredVectorResidue {
        classifications,
        partition_entries,
        total_error_sq,
        total_value_bits,
    })
}

/// One candidate residue configuration for [`select_residue_config`].
///
/// A Vorbis stream's setup can offer several residue configurations the
/// encoder may route a submap's spectrum through — differing in
/// `residue_type` (the §8.6.3/§8.6.4/§8.6.5 addressing), in
/// `partition_size` (the §8.6.1 `residue_partition_size`), and in the
/// value-codebook table (`value_books`, the cascade columns each
/// classification offers). Coarser partitions and cheaper books spend
/// fewer bits at higher distortion; this struct bundles one such
/// candidate so the selector can score them on equal footing.
#[derive(Debug)]
pub struct ResidueConfigCandidate<'a> {
    /// The residue format (§8.6) this candidate uses.
    pub residue_type: u16,
    /// The §8.6.1 `residue_partition_size` this candidate uses. The
    /// target `scalars` length must be a multiple of it.
    pub partition_size: u32,
    /// The candidate's classification → cascade-column value-book table,
    /// one row per classification (`value_books[class][pass]`).
    pub value_books: &'a [[Option<&'a VorbisCodebook>; 8]],
    /// The classbook codeword length, in bits, charged once per
    /// classword. §8.6.2 packs one classword per `classwords_per_codeword`
    /// partitions (formats 1 / 2) or one per partition (format 0);
    /// `select_residue_config` multiplies this by the classword count to
    /// charge the classification side of the rate, which differs by
    /// candidate (a candidate with more classifications needs a wider
    /// classbook). Pass `0` to ignore the classword cost (value-bits-only
    /// comparison).
    pub classword_bits: u8,
    /// How many partitions one classword covers — the residue header's
    /// `classbook` dimension (§8.6.2). `1` for the per-partition format-0
    /// classword; the configured group size for formats 1 / 2. Must be
    /// `>= 1`.
    pub partitions_per_classword: u32,
}

/// The result of [`select_residue_config`]: the index of the winning
/// candidate plus its scored plan and the total Lagrangian cost it
/// achieved (value bits + classword bits folded into the rate term).
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedResidueConfig {
    /// The index into the candidate slice that won.
    pub config_index: usize,
    /// The winning candidate's scored plan.
    pub plan: ScoredVectorResidue,
    /// The total classword bits charged for the winning candidate
    /// (`classword_bits × ⌈partitions / partitions_per_classword⌉`).
    pub classword_bits_total: u64,
    /// The winning Lagrangian cost
    /// `total_error_sq + lambda · (total_value_bits + classword_bits_total)`.
    pub cost: f64,
}

/// Choose the best whole-vector residue *configuration* by rate-distortion
/// (Vorbis I §8.6, encode direction): score every candidate's
/// rate-distortion plan and keep the one minimising
/// `total_error_sq + lambda · total_bits`, where `total_bits` is the
/// candidate's value-codeword bits **plus** its classword bits.
///
/// [`plan_vector_residue_rd`] picks the best classification *within* a
/// fixed configuration (one residue type, one partition size, one book
/// table). This routine sits one level up: it picks the best
/// *configuration* among several the setup offers. The classword cost is
/// folded in here — not inside [`plan_vector_residue_rd`] — because it is
/// a property of the residue header (its classbook), constant across the
/// partitions of a given candidate but different between candidates
/// (a candidate with a denser partitioning emits more classwords; one
/// with more classifications needs a wider classbook).
///
/// Tie-break on an exact Lagrangian cost: lower total bits (cheaper),
/// then lower candidate index (deterministic).
///
/// `scalars` is the target residual for the whole decode vector; every
/// candidate plans the *same* target through its own configuration, so
/// the distortions are directly comparable. `lambda` is the shared
/// bits→distortion exchange rate (`>= 0`, finite).
///
/// # Errors
///
/// * [`ResidueEncodeError::NoClassifications`] if `candidates` is empty.
/// * [`ResidueEncodeError::NonFiniteLambda`] if `lambda` is NaN or
///   negative.
/// * Any error [`plan_vector_residue_rd`] would surface for a candidate
///   (an unsupported residue type, a `scalars` length not a multiple of
///   that candidate's `partition_size`, an empty `value_books`, a bad
///   cascade book, or a candidate with `partitions_per_classword == 0`,
///   reported as [`ResidueEncodeError::ZeroPartitionSize`]'s sibling
///   [`ResidueEncodeError::ZeroPartitionsPerClassword`]).
pub fn select_residue_config(
    scalars: &[f32],
    candidates: &[ResidueConfigCandidate<'_>],
    lambda: f64,
) -> Result<SelectedResidueConfig, ResidueEncodeError> {
    if candidates.is_empty() {
        return Err(ResidueEncodeError::NoClassifications);
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(ResidueEncodeError::NonFiniteLambda(lambda));
    }

    let mut best: Option<SelectedResidueConfig> = None;
    let mut best_cost = f64::INFINITY;
    let mut best_bits = u64::MAX;
    for (i, cand) in candidates.iter().enumerate() {
        if cand.partitions_per_classword == 0 {
            return Err(ResidueEncodeError::ZeroPartitionsPerClassword { config: i });
        }
        let plan = plan_vector_residue_rd(
            scalars,
            cand.value_books,
            cand.residue_type,
            cand.partition_size,
            lambda,
        )?;
        let num_partitions = plan.classifications.len();
        // §8.6.2 packs one classword per `partitions_per_classword`
        // partitions; the final group is padded, so round up.
        let classword_count =
            (num_partitions as u64).div_ceil(u64::from(cand.partitions_per_classword));
        let classword_bits_total = classword_count * u64::from(cand.classword_bits);
        let total_bits = plan.total_value_bits + classword_bits_total;
        let cost = plan.total_error_sq + lambda * total_bits as f64;

        let replace = match &best {
            None => true,
            Some(_) => cost < best_cost || (cost == best_cost && total_bits < best_bits),
        };
        if replace {
            best_cost = cost;
            best_bits = total_bits;
            best = Some(SelectedResidueConfig {
                config_index: i,
                plan,
                classword_bits_total,
                cost,
            });
        }
    }

    Ok(best.expect("candidates non-empty ⇒ a config was chosen"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::VqLookup;
    use crate::vq::unpack_vector;

    /// Build a tessellation (lookup-2) VQ codebook with explicit
    /// per-entry codeword lengths (all `1` = all used) for planner tests.
    fn tess_book(dimensions: u16, entries: u32, multiplicands: Vec<u32>) -> VorbisCodebook {
        VorbisCodebook {
            dimensions,
            entries,
            codeword_lengths: vec![1; entries as usize],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands,
            },
        }
    }

    /// Reconstruct a decode vector the way the residue decoder would:
    /// for each partition, for each populated stage, scatter the entry's
    /// unpacked vector additively per the format's addressing rule. This
    /// is the independent oracle the planner's round-trip is checked
    /// against (it does not call the planner).
    fn decode_reconstruct(
        rows: &[[Option<Vec<u32>>; 8]],
        classifications: &[u32],
        value_books: &[[Option<&VorbisCodebook>; 8]],
        residue_type: u16,
        partition_size: u32,
    ) -> Vec<f32> {
        let ps = partition_size as usize;
        let mut out = vec![0.0f32; rows.len() * ps];
        for (p, (row, &class)) in rows.iter().zip(classifications.iter()).enumerate() {
            let base = p * ps;
            for (pass, slot) in row.iter().enumerate() {
                let Some(entries) = slot else { continue };
                let book = value_books[class as usize][pass].unwrap();
                let dims = book.dimensions as usize;
                if residue_type == 0 {
                    let step = ps / dims;
                    for (read, &entry) in entries.iter().enumerate() {
                        let v = unpack_vector(book, entry).unwrap();
                        for (j, &val) in v.iter().enumerate() {
                            let idx = base + read + j * step;
                            if read + j * step < ps {
                                out[idx] += val;
                            }
                        }
                    }
                } else {
                    for (read, &entry) in entries.iter().enumerate() {
                        let v = unpack_vector(book, entry).unwrap();
                        for (j, &val) in v.iter().enumerate() {
                            let rel = read * dims + j;
                            if rel < ps {
                                out[base + rel] += val;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    // ---------- error paths ----------

    #[test]
    fn unsupported_residue_type_is_rejected() {
        let book = tess_book(2, 2, vec![0, 0, 1, 1]);
        let stages: [Option<&VorbisCodebook>; 8] = {
            let mut s: [Option<&VorbisCodebook>; 8] = [None; 8];
            s[0] = Some(&book);
            s
        };
        assert_eq!(
            plan_partition_cascade(&[0.0, 0.0], &stages, 3, 2),
            Err(ResidueEncodeError::UnsupportedResidueType(3))
        );
    }

    #[test]
    fn zero_partition_size_is_rejected() {
        let stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        assert_eq!(
            plan_partition_cascade(&[], &stages, 1, 0),
            Err(ResidueEncodeError::ZeroPartitionSize)
        );
    }

    #[test]
    fn residual_length_mismatch_is_rejected() {
        let stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        assert_eq!(
            plan_partition_cascade(&[0.0, 0.0, 0.0], &stages, 1, 4),
            Err(ResidueEncodeError::ResidualLengthMismatch {
                expected: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn scalar_value_book_is_rejected() {
        let book = VorbisCodebook {
            dimensions: 2,
            entries: 2,
            codeword_lengths: vec![1, 1],
            lookup: VqLookup::None,
        };
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&book);
        assert_eq!(
            plan_partition_cascade(&[0.0, 0.0], &stages, 1, 2),
            Err(ResidueEncodeError::ScalarValueBook { pass: 0 })
        );
    }

    #[test]
    fn format0_not_divisible_is_rejected() {
        // dims=2 does not divide partition_size=3 (format 0).
        let book = tess_book(2, 2, vec![0, 0, 1, 1]);
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&book);
        assert_eq!(
            plan_partition_cascade(&[0.0, 0.0, 0.0], &stages, 0, 3),
            Err(ResidueEncodeError::Format0NotDivisible {
                pass: 0,
                partition_size: 3,
                dimensions: 2,
            })
        );
    }

    #[test]
    fn fully_unused_book_surfaces_quantize_error() {
        use crate::codebook::UNUSED_ENTRY;
        let book = VorbisCodebook {
            dimensions: 2,
            entries: 2,
            codeword_lengths: vec![UNUSED_ENTRY, UNUSED_ENTRY],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1],
            },
        };
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&book);
        let err = plan_partition_cascade(&[1.0, 1.0], &stages, 1, 2).unwrap_err();
        assert!(matches!(
            err,
            ResidueEncodeError::Quantize {
                pass: 0,
                read: 0,
                source: QuantizeError::NoUsableEntries,
            }
        ));
    }

    // ---------- single-stage planning ----------

    #[test]
    fn all_unused_cascade_emits_all_none() {
        let stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        let row = plan_partition_cascade(&[1.0, 2.0, 3.0, 4.0], &stages, 1, 4).unwrap();
        assert!(row.iter().all(|s| s.is_none()));
    }

    #[test]
    fn single_stage_format1_picks_nearest_entries() {
        // 2-dim book, delta=1/min=0 → entries at [0,0],[1,1],[2,2],[3,3].
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&book);
        // partition_size 4, format 1 → 2 reads of 2 dims each.
        // read 0 target [3.4,3.4] → e3 [3,3]; read 1 target [1.9,1.9] → e2 [2,2].
        let row = plan_partition_cascade(&[3.4, 3.4, 1.9, 1.9], &stages, 1, 4).unwrap();
        assert_eq!(row[0].as_ref().unwrap(), &vec![3u32, 2]);
        for (pass, slot) in row.iter().enumerate() {
            if pass != 0 {
                assert!(slot.is_none());
            }
        }
    }

    #[test]
    fn single_stage_format0_strided_addressing() {
        // dims=2, partition_size=4 → step=2, 2 reads.
        // read 0 covers positions {0, 2}; read 1 covers {1, 3}.
        // delta=1/min=0 → entries at [0,0],[1,1],[2,2],[3,3].
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&book);
        // scalars laid out as positions [0,1,2,3] = [3.0, 2.0, 3.0, 2.0]:
        //   read 0 gathers {pos0, pos2} = [3,3] → e3.
        //   read 1 gathers {pos1, pos3} = [2,2] → e2.
        let row = plan_partition_cascade(&[3.0, 2.0, 3.0, 2.0], &stages, 0, 4).unwrap();
        assert_eq!(row[0].as_ref().unwrap(), &vec![3u32, 2]);
    }

    // ---------- cascade refinement ----------

    #[test]
    fn two_stage_cascade_refines_residual() {
        // Coarse book: entries at [0,0],[4,4],[8,8],[12,12] (delta 4).
        let coarse = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let coarse = VorbisCodebook {
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 4.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1, 2, 2, 3, 3],
            },
            ..coarse
        };
        // Fine book: entries at [-1,-1],[0,0],[1,1],[2,2] (delta 1, min -1).
        let fine = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![1; 4],
            lookup: VqLookup::Tessellation {
                minimum_value: -1.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1, 2, 2, 3, 3],
            },
        };
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&coarse);
        stages[1] = Some(&fine);
        // target [5.0, 5.0]:
        //   stage 0 picks [4,4] (nearest coarse), residual [1,1].
        //   stage 1 picks [1,1] (exact fine hit), residual [0,0].
        // sum [4,4]+[1,1] = [5,5] = exact target.
        let row = plan_partition_cascade(&[5.0, 5.0], &stages, 1, 2).unwrap();
        let stages_slice: Vec<[Option<&VorbisCodebook>; 8]> = vec![stages];
        let recon = decode_reconstruct(&[row], &[0], &stages_slice, 1, 2);
        assert_eq!(recon, vec![5.0, 5.0]);
    }

    // ---------- full-vector round-trip ----------

    #[test]
    fn vector_round_trip_format1_multi_partition() {
        let book = tess_book(2, 5, vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&book);
        let value_books = vec![row0];
        // 3 partitions of size 4, all classification 0.
        let classifications = vec![0u32, 0, 0];
        let scalars = vec![
            0.1, 0.9, 2.2, 1.6, // partition 0
            3.4, 4.1, 0.0, 0.4, // partition 1
            2.0, 2.0, 1.0, 3.0, // partition 2
        ];
        let rows =
            plan_vector_partition_entries(&scalars, &classifications, &value_books, 1, 4).unwrap();
        assert_eq!(rows.len(), 3);
        // Each partition reconstruction must equal the per-read nearest
        // entries the decoder reads back.
        let recon = decode_reconstruct(&rows, &classifications, &value_books, 1, 4);
        // The reconstruction is the nearest-entry snap of each 2-vector.
        // Build the expected snap independently via quantize_vector.
        let mut expected = Vec::new();
        for chunk in scalars.chunks(2) {
            let q = quantize_vector(&book, chunk).unwrap();
            expected.extend_from_slice(&q.vector);
        }
        assert_eq!(recon, expected);
    }

    #[test]
    fn vector_round_trip_format0_strided() {
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&book);
        let value_books = vec![row0];
        let classifications = vec![0u32, 0];
        // 2 partitions of size 4. Format-0 strided gather/scatter.
        // Book entries (delta=1/min=0): [0,0],[1,1],[2,2],[3,3].
        let scalars = vec![
            3.0, 2.0, 3.0, 2.0, // partition 0
            0.0, 3.0, 0.0, 3.0, // partition 1
        ];
        let rows =
            plan_vector_partition_entries(&scalars, &classifications, &value_books, 0, 4).unwrap();
        let recon = decode_reconstruct(&rows, &classifications, &value_books, 0, 4);
        // Independent expectation: gather strided sub-vectors, snap each.
        // partition 0: reads {pos0,pos2}=[3,3]→[3,3], {pos1,pos3}=[2,2]→[2,2].
        // partition 1: reads {pos0,pos2}=[0,0]→[0,0], {pos1,pos3}=[3,3]→[3,3].
        assert_eq!(recon, vec![3.0, 2.0, 3.0, 2.0, 0.0, 3.0, 0.0, 3.0]);
    }

    #[test]
    fn out_of_range_classification_uses_empty_cascade() {
        let book = tess_book(2, 2, vec![0, 0, 5, 5]);
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&book);
        let value_books = vec![row0];
        // classification 1 has no row → all-unused cascade → all-None,
        // and the scalars for that partition are untouched.
        let classifications = vec![0u32, 1];
        let scalars = vec![5.0, 5.0, 9.9, 9.9];
        let rows =
            plan_vector_partition_entries(&scalars, &classifications, &value_books, 1, 2).unwrap();
        assert_eq!(rows.len(), 2);
        // partition 0 (class 0) planned to entry 1 ([5,5] exact).
        assert_eq!(rows[0][0].as_ref().unwrap(), &vec![1u32]);
        // partition 1 (class 1, no row) → all None.
        assert!(rows[1].iter().all(|s| s.is_none()));
    }

    #[test]
    fn format2_planned_as_format1() {
        // Format 2 reduces to format 1 (§8.6.5): same contiguous gather.
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&book);
        let row1 = plan_partition_cascade(&[3.4, 3.4, 1.9, 1.9], &stages, 1, 4).unwrap();
        let row2 = plan_partition_cascade(&[3.4, 3.4, 1.9, 1.9], &stages, 2, 4).unwrap();
        assert_eq!(row1, row2);
    }

    // ---------- scored cascade ----------

    #[test]
    fn scored_cascade_reports_exact_hit_zero_error() {
        // Book entries (delta=1/min=0): [0,0],[1,1],[2,2],[3,3].
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&book);
        // target [3,3] is an exact entry hit → zero residual.
        let scored = plan_partition_cascade_scored(&[3.0, 3.0], &stages, 1, 2).unwrap();
        assert_eq!(scored.error_sq, 0.0);
        assert_eq!(scored.populated_stages, 1);
        assert_eq!(scored.entries[0].as_ref().unwrap(), &vec![3u32]);
        // One value codeword emitted; `tess_book` gives every entry a
        // codeword length of 1, so the charged rate is exactly 1 bit.
        assert_eq!(scored.bit_cost, 1);
    }

    #[test]
    fn scored_cascade_reports_quantisation_residual() {
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&book);
        // target [3.5, 3.5] → nearest [3,3], residual [0.5, 0.5], ‖·‖²=0.5.
        let scored = plan_partition_cascade_scored(&[3.5, 3.5], &stages, 1, 2).unwrap();
        assert!((scored.error_sq - 0.5).abs() < 1e-9);
        assert_eq!(scored.populated_stages, 1);
    }

    #[test]
    fn scored_unused_cascade_error_is_target_norm() {
        // No stages populated → reconstruction is all-zero, so the error
        // is the squared norm of the target itself.
        let stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        let scored = plan_partition_cascade_scored(&[1.0, 2.0, 2.0], &stages, 1, 3).unwrap();
        assert_eq!(scored.error_sq, 9.0); // 1 + 4 + 4
        assert_eq!(scored.populated_stages, 0);
        // No value codewords emitted by an all-'unused' cascade.
        assert_eq!(scored.bit_cost, 0);
    }

    #[test]
    fn scored_cascade_bit_cost_sums_value_codeword_lengths() {
        // A book whose per-entry codeword lengths vary by entry, so the
        // charged rate is genuinely entry-dependent (not just `reads × 1`).
        // Entries [0,0],[1,1],[2,2],[3,3] with lengths {2, 5, 3, 4}.
        let book = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![2, 5, 3, 4],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1, 2, 2, 3, 3],
            },
        };
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&book);
        // partition_size 4, format 1 → 2 reads of 2 dims.
        // read 0 target [3,3] → entry 3 (len 4); read 1 target [1,1] → entry 1 (len 5).
        let scored = plan_partition_cascade_scored(&[3.0, 3.0, 1.0, 1.0], &stages, 1, 4).unwrap();
        assert_eq!(scored.entries[0].as_ref().unwrap(), &vec![3u32, 1]);
        assert_eq!(scored.bit_cost, 4 + 5); // sum of the two chosen lengths
        assert_eq!(scored.populated_stages, 1);
        assert_eq!(scored.error_sq, 0.0);
    }

    #[test]
    fn scored_cascade_bit_cost_accumulates_across_stages() {
        // Two stages on the same book → both stages emit a codeword for the
        // single read; the rate is the sum across stages, even when stage 2
        // adds nothing to the reconstruction (it still costs its codeword).
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]); // lengths all 1
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&book);
        stages[2] = Some(&book); // a non-adjacent stage to exercise pass order
        let scored = plan_partition_cascade_scored(&[3.0, 3.0], &stages, 1, 2).unwrap();
        // Two populated stages, one read each, every length 1 → 2 bits.
        assert_eq!(scored.populated_stages, 2);
        assert_eq!(scored.bit_cost, 2);
    }

    #[test]
    fn scored_entries_match_unscored() {
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&book);
        let unscored = plan_partition_cascade(&[3.4, 3.4, 1.9, 1.9], &stages, 1, 4).unwrap();
        let scored = plan_partition_cascade_scored(&[3.4, 3.4, 1.9, 1.9], &stages, 1, 4).unwrap();
        assert_eq!(unscored, scored.entries);
    }

    // ---------- classification selection ----------

    #[test]
    fn classify_picks_lower_distortion_class() {
        // Class 0: coarse book entries [0,0],[4,4],[8,8],[12,12] (delta 4).
        let coarse = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![1; 4],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 4.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1, 2, 2, 3, 3],
            },
        };
        // Class 1: fine book entries [0,0],[1,1],[2,2],[3,3] (delta 1).
        let fine = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&coarse);
        let mut row1: [Option<&VorbisCodebook>; 8] = [None; 8];
        row1[0] = Some(&fine);
        let value_books = vec![row0, row1];
        // target [3,3]: coarse nearest [4,4] err 2; fine nearest [3,3] err 0.
        // → class 1 wins on distortion.
        let choices = plan_vector_classifications(&[3.0, 3.0], &value_books, 1, 2).unwrap();
        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0].classification, 1);
        assert_eq!(choices[0].error_sq, 0.0);
        assert_eq!(choices[0].entries[0].as_ref().unwrap(), &vec![3u32]);
    }

    #[test]
    fn classify_ties_break_toward_fewer_stages() {
        // Two classifications both reconstruct the target exactly, but one
        // uses one stage and the other uses two. Equal distortion (0) →
        // the cheaper (single-stage) class must win.
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        // class 0: single stage on `book`.
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&book);
        // class 1: two stages, both on `book` — also reaches [3,3] exactly
        // (stage 0 picks [3,3], stage 1 picks [0,0]) but costs an extra
        // codeword.
        let mut row1: [Option<&VorbisCodebook>; 8] = [None; 8];
        row1[0] = Some(&book);
        row1[1] = Some(&book);
        let value_books = vec![row0, row1];
        let choices = plan_vector_classifications(&[3.0, 3.0], &value_books, 1, 2).unwrap();
        assert_eq!(choices[0].classification, 0);
        assert_eq!(choices[0].error_sq, 0.0);
    }

    // ---------- rate-distortion classification selection ----------

    /// `lambda == 0` must reduce the RD chooser to the pure-distortion
    /// chooser bit-for-bit: same classifications, same entries, same
    /// per-partition distortion. Uses the per-partition-independent
    /// fixture so both partitions exercise a real choice.
    #[test]
    fn rd_lambda_zero_matches_distortion_chooser() {
        let coarse = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![1; 4],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 4.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1, 2, 2, 3, 3],
            },
        };
        let fine = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&coarse);
        let mut row1: [Option<&VorbisCodebook>; 8] = [None; 8];
        row1[0] = Some(&fine);
        let value_books = vec![row0, row1];
        let scalars = vec![8.0, 8.0, 2.0, 2.0];

        let dist = plan_vector_classifications(&scalars, &value_books, 1, 2).unwrap();
        let rd = plan_vector_classifications_rd(&scalars, &value_books, 1, 2, 0.0).unwrap();
        assert_eq!(dist, rd);
    }

    /// A large `lambda` must flip the choice from a dense, lower-distortion
    /// cascade to a cheaper one when the extra bits aren't worth their
    /// small distortion gain. Class 1 reconstructs the target *slightly*
    /// better but costs an extra codeword; with `lambda` large the cheap
    /// single-stage class 0 wins.
    #[test]
    fn rd_large_lambda_prefers_cheaper_class() {
        // Single book, lengths all 1. Target [3.0, 3.0].
        // class 0: one stage  → picks [3,3], error 0, cost 1 bit.
        // class 1: two stages → picks [3,3] then [0,0], error 0, cost 2 bits.
        // Pure distortion ties (both error 0) → fewer-stages tie-break
        // already prefers class 0; to make the *rate* term load-bearing we
        // give class 1 a strictly lower distortion that a big lambda
        // overrides. Use a finer second stage so class 1 wins on distortion
        // at lambda 0 but loses at large lambda.
        let coarse = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![1; 4],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 2.0, // entries [0,0],[2,2],[4,4],[6,6]
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1, 2, 2, 3, 3],
            },
        };
        let fine = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]); // delta 1
                                                                  // class 0: coarse alone → nearest to [3,3] is [2,2] or [4,4], err 2.
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&coarse);
        // class 1: coarse then fine → coarse [2,2] (or [4,4]) leaves ±1, fine
        // refines to error 0, but costs 2 codewords.
        let mut row1: [Option<&VorbisCodebook>; 8] = [None; 8];
        row1[0] = Some(&coarse);
        row1[1] = Some(&fine);
        let value_books = vec![row0, row1];
        let scalars = vec![3.0, 3.0];

        // lambda 0: class 1 wins (lower distortion).
        let rd0 = plan_vector_classifications_rd(&scalars, &value_books, 1, 2, 0.0).unwrap();
        assert_eq!(rd0[0].classification, 1);
        assert_eq!(rd0[0].error_sq, 0.0);
        assert_eq!(rd0[0].bit_cost, 2);

        // Large lambda: the extra bit of class 1 outweighs its distortion
        // gain (class 0 leaves error 2 for 1 bit; class 1 error 0 for 2
        // bits; cost_0 = 2 + λ·1, cost_1 = 0 + λ·2; class 0 wins once
        // 2 + λ < 2λ ⇒ λ > 2). Pick λ = 10.
        let rd_hi = plan_vector_classifications_rd(&scalars, &value_books, 1, 2, 10.0).unwrap();
        assert_eq!(rd_hi[0].classification, 0);
        assert_eq!(rd_hi[0].bit_cost, 1);
        assert!(rd_hi[0].error_sq > 0.0);
    }

    #[test]
    fn rd_rejects_negative_and_nan_lambda() {
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut row: [Option<&VorbisCodebook>; 8] = [None; 8];
        row[0] = Some(&book);
        let value_books = vec![row];
        assert_eq!(
            plan_vector_classifications_rd(&[3.0, 3.0], &value_books, 1, 2, -1.0),
            Err(ResidueEncodeError::NonFiniteLambda(-1.0))
        );
        let nan = plan_vector_classifications_rd(&[3.0, 3.0], &value_books, 1, 2, f64::NAN);
        assert!(matches!(
            nan,
            Err(ResidueEncodeError::NonFiniteLambda(l)) if l.is_nan()
        ));
    }

    // ---------- perceptually weighted rate-distortion selection ----------

    /// The coarse/fine two-class fixture from
    /// `rd_large_lambda_prefers_cheaper_class`: class 0 = coarse alone
    /// (err 2 on [3,3], 1 bit), class 1 = coarse+fine (err 0, 2 bits).
    fn coarse_fine_books() -> (VorbisCodebook, VorbisCodebook) {
        let coarse = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![1; 4],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 2.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1, 2, 2, 3, 3],
            },
        };
        let fine = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        (coarse, fine)
    }

    /// All-`1.0` weights must reproduce the unweighted RD chooser
    /// bit-for-bit at several lambdas.
    #[test]
    fn weighted_all_ones_matches_unweighted() {
        let (coarse, fine) = coarse_fine_books();
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&coarse);
        let mut row1: [Option<&VorbisCodebook>; 8] = [None; 8];
        row1[0] = Some(&coarse);
        row1[1] = Some(&fine);
        let value_books = vec![row0, row1];
        let scalars = vec![3.0, 3.0, 8.0, 8.0, 1.0, 1.0];
        let ones = vec![1.0f64; 3];

        for lambda in [0.0, 0.5, 2.0, 10.0] {
            let plain =
                plan_vector_classifications_rd(&scalars, &value_books, 1, 2, lambda).unwrap();
            let weighted = plan_vector_classifications_rd_weighted(
                &scalars,
                &value_books,
                1,
                2,
                lambda,
                &ones,
            )
            .unwrap();
            assert_eq!(plain, weighted, "lambda {lambda}");
            let v_plain = plan_vector_residue_rd(&scalars, &value_books, 1, 2, lambda).unwrap();
            let v_weighted =
                plan_vector_residue_rd_weighted(&scalars, &value_books, 1, 2, lambda, &ones)
                    .unwrap();
            assert_eq!(v_plain, v_weighted, "lambda {lambda}");
        }
    }

    /// A high perceptual weight must buy back the denser cascade the
    /// unweighted chooser gives up at the same lambda: at λ = 10 the
    /// cheap class wins unweighted (cost 2 + 10 = 12 vs 0 + 20 = 20),
    /// but at weight 100 the audible distortion dominates
    /// (200 + 10 = 210 vs 0 + 20 = 20) and the fine class wins.
    #[test]
    fn high_weight_buys_the_denser_cascade() {
        let (coarse, fine) = coarse_fine_books();
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&coarse);
        let mut row1: [Option<&VorbisCodebook>; 8] = [None; 8];
        row1[0] = Some(&coarse);
        row1[1] = Some(&fine);
        let value_books = vec![row0, row1];
        let scalars = vec![3.0, 3.0];

        let unweighted =
            plan_vector_classifications_rd_weighted(&scalars, &value_books, 1, 2, 10.0, &[1.0])
                .unwrap();
        assert_eq!(unweighted[0].classification, 0);

        let weighted =
            plan_vector_classifications_rd_weighted(&scalars, &value_books, 1, 2, 10.0, &[100.0])
                .unwrap();
        assert_eq!(weighted[0].classification, 1);
        assert_eq!(weighted[0].error_sq, 0.0);
        assert_eq!(weighted[0].bit_cost, 2);
    }

    /// A zero weight makes a partition's choice rate-only: the cheaper
    /// cascade wins even though the denser one has strictly lower
    /// distortion.
    #[test]
    fn zero_weight_partition_takes_the_cheapest_cascade() {
        let (coarse, fine) = coarse_fine_books();
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&coarse);
        let mut row1: [Option<&VorbisCodebook>; 8] = [None; 8];
        row1[0] = Some(&coarse);
        row1[1] = Some(&fine);
        let value_books = vec![row0, row1];
        let scalars = vec![3.0, 3.0];

        // At λ = 0.1 the fine class wins under weight 1 (0.2 vs 2.1)…
        let w1 = plan_vector_classifications_rd_weighted(&scalars, &value_books, 1, 2, 0.1, &[1.0])
            .unwrap();
        assert_eq!(w1[0].classification, 1);
        // …but under weight 0 only the bits matter (0.1 vs 0.2).
        let w0 = plan_vector_classifications_rd_weighted(&scalars, &value_books, 1, 2, 0.1, &[0.0])
            .unwrap();
        assert_eq!(w0[0].classification, 0);
        assert_eq!(w0[0].bit_cost, 1);
    }

    /// Weights apply per partition: identical targets with different
    /// weights choose differently within one vector.
    #[test]
    fn weights_are_per_partition() {
        let (coarse, fine) = coarse_fine_books();
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&coarse);
        let mut row1: [Option<&VorbisCodebook>; 8] = [None; 8];
        row1[0] = Some(&coarse);
        row1[1] = Some(&fine);
        let value_books = vec![row0, row1];
        let scalars = vec![3.0, 3.0, 3.0, 3.0];

        let choices = plan_vector_classifications_rd_weighted(
            &scalars,
            &value_books,
            1,
            2,
            10.0,
            &[1.0, 100.0],
        )
        .unwrap();
        assert_eq!(choices[0].classification, 0, "masked partition → cheap");
        assert_eq!(choices[1].classification, 1, "audible partition → dense");
    }

    #[test]
    fn weighted_rejects_bad_weight_vectors() {
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut row: [Option<&VorbisCodebook>; 8] = [None; 8];
        row[0] = Some(&book);
        let value_books = vec![row];
        let scalars = vec![3.0, 3.0, 1.0, 1.0];

        assert_eq!(
            plan_vector_classifications_rd_weighted(&scalars, &value_books, 1, 2, 1.0, &[1.0]),
            Err(ResidueEncodeError::WeightLengthMismatch {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            plan_vector_classifications_rd_weighted(
                &scalars,
                &value_books,
                1,
                2,
                1.0,
                &[1.0, -0.5]
            ),
            Err(ResidueEncodeError::BadWeight {
                partition: 1,
                value: -0.5
            })
        );
        let nan = plan_vector_classifications_rd_weighted(
            &scalars,
            &value_books,
            1,
            2,
            1.0,
            &[f64::NAN, 1.0],
        );
        assert!(matches!(
            nan,
            Err(ResidueEncodeError::BadWeight { partition: 0, value }) if value.is_nan()
        ));
    }

    #[test]
    fn classify_ties_break_toward_lower_index() {
        // Two identical classifications → equal distortion + equal stage
        // count → the lower index wins (deterministic).
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut row: [Option<&VorbisCodebook>; 8] = [None; 8];
        row[0] = Some(&book);
        let value_books = vec![row, row];
        let choices = plan_vector_classifications(&[3.5, 3.5], &value_books, 1, 2).unwrap();
        assert_eq!(choices[0].classification, 0);
    }

    #[test]
    fn classify_per_partition_independent() {
        // Coarse vs fine class; two partitions whose ideal class differs.
        let coarse = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![1; 4],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 4.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1, 2, 2, 3, 3],
            },
        };
        let fine = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&coarse);
        let mut row1: [Option<&VorbisCodebook>; 8] = [None; 8];
        row1[0] = Some(&fine);
        let value_books = vec![row0, row1];
        // partition 0 target [8,8] → exact coarse hit (class 0).
        // partition 1 target [2,2] → exact fine hit (class 1).
        let scalars = vec![8.0, 8.0, 2.0, 2.0];
        let choices = plan_vector_classifications(&scalars, &value_books, 1, 2).unwrap();
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0].classification, 0);
        assert_eq!(choices[1].classification, 1);
        assert_eq!(choices[0].error_sq, 0.0);
        assert_eq!(choices[1].error_sq, 0.0);
    }

    #[test]
    fn classify_empty_value_books_is_rejected() {
        let value_books: Vec<[Option<&VorbisCodebook>; 8]> = vec![];
        assert_eq!(
            plan_vector_classifications(&[1.0, 2.0], &value_books, 1, 2),
            Err(ResidueEncodeError::NoClassifications)
        );
    }

    #[test]
    fn classify_non_multiple_length_is_rejected() {
        let book = tess_book(2, 2, vec![0, 0, 1, 1]);
        let mut row: [Option<&VorbisCodebook>; 8] = [None; 8];
        row[0] = Some(&book);
        let value_books = vec![row];
        // 5 scalars, partition_size 2 → not a multiple.
        assert_eq!(
            plan_vector_classifications(&[1.0, 2.0, 3.0, 4.0, 5.0], &value_books, 1, 2),
            Err(ResidueEncodeError::ResidualLengthMismatch {
                expected: 4,
                actual: 5,
            })
        );
    }

    #[test]
    fn classify_propagates_cascade_error() {
        // A stage book with no vector lookup must surface, not be skipped.
        let bad = VorbisCodebook {
            dimensions: 2,
            entries: 2,
            codeword_lengths: vec![1, 1],
            lookup: VqLookup::None,
        };
        let mut row: [Option<&VorbisCodebook>; 8] = [None; 8];
        row[0] = Some(&bad);
        let value_books = vec![row];
        assert_eq!(
            plan_vector_classifications(&[0.0, 0.0], &value_books, 1, 2),
            Err(ResidueEncodeError::ScalarValueBook { pass: 0 })
        );
    }

    #[test]
    fn plan_vector_residue_round_trips_against_decode() {
        // Full top-of-stack: raw scalars → chosen classifications + entries
        // → decode_reconstruct must equal the per-partition nearest snap of
        // whichever class the chooser selected.
        let coarse = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![1; 4],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 4.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1, 2, 2, 3, 3],
            },
        };
        let fine = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&coarse);
        let mut row1: [Option<&VorbisCodebook>; 8] = [None; 8];
        row1[0] = Some(&fine);
        let value_books = vec![row0, row1];
        let scalars = vec![8.0, 8.0, 2.0, 2.0, 0.3, 0.3];
        let (classifications, entries) = plan_vector_residue(&scalars, &value_books, 1, 2).unwrap();
        assert_eq!(classifications.len(), 3);
        assert_eq!(entries.len(), 3);
        // Reconstruct via the independent decode oracle and confirm each
        // partition equals the chosen class's nearest-entry snap.
        let recon = decode_reconstruct(&entries, &classifications, &value_books, 1, 2);
        let mut expected = Vec::new();
        for (p, chunk) in scalars.chunks(2).enumerate() {
            let class = classifications[p] as usize;
            let book = value_books[class][0].unwrap();
            let q = quantize_vector(book, chunk).unwrap();
            expected.extend_from_slice(&q.vector);
        }
        assert_eq!(recon, expected);
    }

    #[test]
    fn classify_chooses_class_minimising_distortion_over_grid() {
        // Sweep targets; the chosen class must always have distortion <=
        // every other class's distortion for that partition.
        let coarse = VorbisCodebook {
            dimensions: 1,
            entries: 4,
            codeword_lengths: vec![1; 4],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 4.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 1, 2, 3],
            },
        };
        let fine = VorbisCodebook {
            dimensions: 1,
            entries: 4,
            codeword_lengths: vec![1; 4],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 1, 2, 3],
            },
        };
        let mut row0: [Option<&VorbisCodebook>; 8] = [None; 8];
        row0[0] = Some(&coarse);
        let mut row1: [Option<&VorbisCodebook>; 8] = [None; 8];
        row1[0] = Some(&fine);
        let value_books = vec![row0, row1];
        for t in 0..=120u32 {
            let target = t as f32 * 0.1;
            let scalars = vec![target];
            let choices = plan_vector_classifications(&scalars, &value_books, 1, 1).unwrap();
            let chosen = &choices[0];
            for stage_books in &value_books {
                let s = plan_partition_cascade_scored(&scalars, stage_books, 1, 1).unwrap();
                assert!(
                    chosen.error_sq <= s.error_sq + 1e-9,
                    "target {target}: chosen class error {} > {}",
                    chosen.error_sq,
                    s.error_sq
                );
            }
        }
    }

    #[test]
    fn format1_non_divisible_tail_is_discarded() {
        // partition_size=3, dims=2 → ceil(3/2)=2 reads. Read 1 covers
        // positions {2, (3=out of range)} — the surplus element targets
        // 0.0 and the decoder discards it.
        // Book entries (delta=1/min=0): [0,0],[1,1],[2,2],[3,3].
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut stages: [Option<&VorbisCodebook>; 8] = [None; 8];
        stages[0] = Some(&book);
        let row = plan_partition_cascade(&[4.0, 4.0, 2.0], &stages, 1, 3).unwrap();
        let entries = row[0].as_ref().unwrap();
        assert_eq!(entries.len(), 2);
        // read 0 [4,4] → nearest [3,3] = e3.
        assert_eq!(entries[0], 3);
        // read 1 [2, 0] (tail padded to 0.0) → distances:
        //   e0 [0,0]: 4+0=4; e1 [1,1]: 1+1=2; e2 [2,2]: 0+4=4 → e1.
        assert_eq!(entries[1], 1);
    }

    // ---------- scored whole-vector plan + config selection ----------

    #[test]
    fn vector_rd_aggregates_distortion_and_bits() {
        // Single fine book (lengths all 1), 2 partitions of size 2.
        // partition 0 target [3,3] exact hit (err 0, 1 bit);
        // partition 1 target [1.5,1.5] → nearest [2,2] or [1,1] (err 0.5, 1 bit).
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut row: [Option<&VorbisCodebook>; 8] = [None; 8];
        row[0] = Some(&book);
        let value_books = vec![row];
        let scalars = vec![3.0, 3.0, 1.5, 1.5];
        let scored = plan_vector_residue_rd(&scalars, &value_books, 1, 2, 0.0).unwrap();
        assert_eq!(scored.classifications, vec![0, 0]);
        assert_eq!(scored.partition_entries.len(), 2);
        // Σ distortion = 0 + 0.5 = 0.5; Σ value bits = 1 + 1 = 2.
        assert!((scored.total_error_sq - 0.5).abs() < 1e-9);
        assert_eq!(scored.total_value_bits, 2);

        // lambda 0 must agree with the unscored splitter on the plan.
        let (cls, ent) = plan_vector_residue(&scalars, &value_books, 1, 2).unwrap();
        assert_eq!(scored.classifications, cls);
        assert_eq!(scored.partition_entries, ent);
    }

    #[test]
    fn select_config_prefers_cheaper_config_at_high_lambda() {
        // Two candidate configs over the SAME target, both format 1, ps 2.
        // Config A: a single fine book → reconstructs exactly (err 0) but
        //   spends 1 value bit/partition.
        // Config B: a single coarse book → leaves distortion but the same
        //   bit count here; to make rate load-bearing we instead vary the
        //   classword charge: A has a wide classbook, B a narrow one.
        let fine = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]); // delta 1, len 1
        let coarse = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![1; 4],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 3.0, // entries [0,0],[3,3],[6,6],[9,9]
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1, 2, 2, 3, 3],
            },
        };
        let mut fine_row: [Option<&VorbisCodebook>; 8] = [None; 8];
        fine_row[0] = Some(&fine);
        let fine_books = vec![fine_row];
        let mut coarse_row: [Option<&VorbisCodebook>; 8] = [None; 8];
        coarse_row[0] = Some(&coarse);
        let coarse_books = vec![coarse_row];

        // Target where fine reconstructs exactly and coarse leaves a little
        // error: [3,3] is an exact entry for BOTH (coarse e1, fine e3), so
        // pick [2,2] instead — fine exact (err 0), coarse nearest [3,3]
        // (err 2).
        let scalars = vec![2.0, 2.0];

        let candidates = vec![
            ResidueConfigCandidate {
                residue_type: 1,
                partition_size: 2,
                value_books: &fine_books,
                classword_bits: 0,
                partitions_per_classword: 1,
            },
            ResidueConfigCandidate {
                residue_type: 1,
                partition_size: 2,
                value_books: &coarse_books,
                classword_bits: 0,
                partitions_per_classword: 1,
            },
        ];

        // At any lambda the fine config has lower distortion AND equal bits
        // (1 value codeword each), so it always wins. Confirm config 0.
        let sel = select_residue_config(&scalars, &candidates, 1.0).unwrap();
        assert_eq!(sel.config_index, 0);
        assert_eq!(sel.plan.total_error_sq, 0.0);
    }

    #[test]
    fn select_config_classword_cost_flips_choice() {
        // Same target reconstructed exactly by both candidates (err 0,
        // equal value bits), but candidate 0 carries an expensive classword
        // and candidate 1 a free one. With lambda > 0 the classword charge
        // must flip the winner to candidate 1.
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut row: [Option<&VorbisCodebook>; 8] = [None; 8];
        row[0] = Some(&book);
        let books = vec![row];
        let scalars = vec![3.0, 3.0]; // exact hit, err 0, 1 value bit.

        let candidates = vec![
            ResidueConfigCandidate {
                residue_type: 1,
                partition_size: 2,
                value_books: &books,
                classword_bits: 8, // expensive classword
                partitions_per_classword: 1,
            },
            ResidueConfigCandidate {
                residue_type: 1,
                partition_size: 2,
                value_books: &books,
                classword_bits: 0, // free classword
                partitions_per_classword: 1,
            },
        ];
        let sel = select_residue_config(&scalars, &candidates, 1.0).unwrap();
        assert_eq!(sel.config_index, 1);
        assert_eq!(sel.classword_bits_total, 0);
        // total cost = error(0) + lambda(1) * (value 1 + classword 0) = 1.
        assert!((sel.cost - 1.0).abs() < 1e-9);
    }

    #[test]
    fn select_config_charges_one_classword_per_group() {
        // 4 partitions, partitions_per_classword = 2 → 2 classwords.
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut row: [Option<&VorbisCodebook>; 8] = [None; 8];
        row[0] = Some(&book);
        let books = vec![row];
        // 4 partitions of size 2 = 8 scalars, all exact hits.
        let scalars = vec![3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0];
        let candidates = vec![ResidueConfigCandidate {
            residue_type: 1,
            partition_size: 2,
            value_books: &books,
            classword_bits: 5,
            partitions_per_classword: 2,
        }];
        let sel = select_residue_config(&scalars, &candidates, 0.5).unwrap();
        // 4 partitions / 2 per classword = 2 classwords × 5 bits = 10 bits.
        assert_eq!(sel.classword_bits_total, 10);
        // value bits: 4 partitions × 1 bit = 4.
        assert_eq!(sel.plan.total_value_bits, 4);
    }

    #[test]
    fn select_config_rejects_empty_and_bad_lambda_and_zero_group() {
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut row: [Option<&VorbisCodebook>; 8] = [None; 8];
        row[0] = Some(&book);
        let books = vec![row];
        let empty: Vec<ResidueConfigCandidate<'_>> = vec![];
        assert_eq!(
            select_residue_config(&[3.0, 3.0], &empty, 1.0),
            Err(ResidueEncodeError::NoClassifications)
        );
        let cands = vec![ResidueConfigCandidate {
            residue_type: 1,
            partition_size: 2,
            value_books: &books,
            classword_bits: 0,
            partitions_per_classword: 0,
        }];
        assert_eq!(
            select_residue_config(&[3.0, 3.0], &cands, 1.0),
            Err(ResidueEncodeError::ZeroPartitionsPerClassword { config: 0 })
        );
        let ok_cands = vec![ResidueConfigCandidate {
            residue_type: 1,
            partition_size: 2,
            value_books: &books,
            classword_bits: 0,
            partitions_per_classword: 1,
        }];
        assert_eq!(
            select_residue_config(&[3.0, 3.0], &ok_cands, -1.0),
            Err(ResidueEncodeError::NonFiniteLambda(-1.0))
        );
    }
}

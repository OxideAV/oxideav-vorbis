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
//! This module plans the value-codeword entry lists only — the
//! `partition_entries` field of a [`crate::encoder::ResidueVectorPlan`].
//! Choosing the per-partition *classification* (which cascade column a
//! partition takes, hence its bit cost) is a separate psychoacoustic
//! decision the caller still owns; [`plan_vector_partition_entries`]
//! takes the classifications as given and fills in the entry lists for
//! them. Threading the resulting plans into a full packet is the
//! existing [`crate::encoder::write_residue_body`] /
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
            ResidueEncodeError::Quantize { pass, read, source } => write!(
                f,
                "vorbis residue encode: stage-{pass} read-{read} quantise failed: {source}"
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
            entries.push(q.entry);
        }
        out[pass] = Some(entries);
    }

    Ok(out)
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
}

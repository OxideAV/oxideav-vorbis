//! Floor 0 VQ-encode glue (Vorbis I §6.2.2, encode direction).
//!
//! The crate already carries the floor-0 **WRITE** path: the
//! [`crate::encoder::write_floor0_packet`] primitive serialises a §6.2.2
//! floor-0 audio-packet body — the `[amplitude]` field, the
//! `[booknumber]` value-book selector, and a run of VQ codewords — from a
//! [`crate::encoder::Floor0Packet::Curve`] whose `entries` member is the
//! explicit list of value-book **entry indices** the §6.2.2 step-7 loop
//! decodes. The leaf [`crate::vq::quantize_vector`] turns one target
//! vector into the nearest codebook entry. What sat between them — the
//! glue that slices a real floor-0 LSP coefficient list into the
//! per-VQ-read sub-vectors, undoes the §6.2.2 cross-vector `[last]`
//! accumulation, quantises each, and emits the entry-index run — was the
//! floor-0 encode followup named in the crate README. This module is that
//! glue.
//!
//! ## What the decoder does, and the inverse
//!
//! A floor-0 packet decode (§6.2.2; see [`crate::floor0`]) rebuilds the
//! `[coefficients]` LSP filter vector by reading value-book VQ entries
//! one at a time, each unpacking to a `dimensions`-length `[temp_vector]`,
//! then **accumulating** a running scalar `[last]` across vectors:
//!
//! * **step 6.** `[last] = 0`.
//! * **steps 7..11**, repeated until `len(coefficients) >= floor0_order`:
//!   1. decode a VQ entry and `unpack_vector` it into `[temp_vector]`
//!      (length `dimensions`);
//!   2. **add** `[last]` to every scalar of `[temp_vector]`;
//!   3. set `[last]` to the *last* scalar of the (now-offset)
//!      `[temp_vector]`;
//!   4. concatenate `[temp_vector]` onto `[coefficients]`.
//!
//! So the coefficient list is a piecewise-cumulative reconstruction: the
//! `k`-th vector is offset by the running sum of every *previous* vector's
//! final scalar.
//!
//! Encoding inverts this. Given a target coefficient list,
//! [`plan_floor0_coefficients`] walks the same vector schedule in the
//! **write** direction:
//!
//! 1. A running `last` starts at `0.0` (the decoder's step-6 seed).
//! 2. For each of the `ceil(order / dimensions)` vectors the decoder
//!    reads, the target sub-vector is **un-offset** by subtracting the
//!    current `last` from each of its `dimensions` target scalars — the
//!    inverse of the decoder's step-8 add — giving the raw VQ-decode
//!    target the codebook quantises against.
//! 3. That raw target is quantised with [`crate::vq::quantize_vector`],
//!    yielding the entry index the decoder will read and the entry's
//!    decoded reconstruction (`unpack_vector(book, entry)`).
//! 4. `last` is advanced to the **reconstructed** vector's final scalar
//!    *plus the offset that was in force* — i.e. the decoder's step-9
//!    `[last]` after step-8 re-adds the offset to the reconstruction. Using
//!    the reconstruction (not the target) keeps the planner's `last` in
//!    lockstep with the decoder's, so every subsequent vector un-offsets
//!    against the value the decoder will actually carry, and the chosen
//!    entry run reconstructs the planner's own approximation bit-for-bit.
//!
//! Because step 2's un-offset is the precise inverse of the decoder's
//! step-8 add, and the quantiser's reported `vector` is bit-identical to
//! `unpack_vector(book, entry)` (the decoder's reconstruction), feeding
//! the resulting entry run back through
//! [`crate::encoder::write_floor0_packet`] and the floor-0 decoder
//! reproduces the planner's approximation exactly. The round-trip is
//! bit-exact on the *bitstream* (entry indices ↔ codewords); the
//! reconstructed *coefficients* are the nearest-entry approximation of the
//! target, which is the lossy quantisation floor-0 LSP coding is.
//!
//! ## Scope
//!
//! This module plans the value-codeword entry run only — the `entries`
//! field of a [`crate::encoder::Floor0Packet::Curve`]. Choosing the
//! per-packet `amplitude` and `booknumber`, and deriving the target LSP
//! `[coefficients]` from a desired floor curve (§6.2.3 inverted), are
//! separate encode decisions the caller still owns;
//! [`plan_floor0_coefficients`] takes the value book and the target
//! coefficients as given and fills in the entry run. Threading the result
//! into a full packet is the existing
//! [`crate::encoder::write_floor0_packet`] /
//! [`crate::encoder::write_audio_packet`] path.

use crate::codebook::{VorbisCodebook, VqLookup};
use crate::vq::{quantize_vector, QuantizeError};

/// Errors that can arise while planning a floor-0 packet's VQ entry run
/// (Vorbis I §6.2.2, encode direction).
#[derive(Debug, Clone, PartialEq)]
pub enum Floor0EncodeError {
    /// `order` was zero. §6.2.1 stores `floor0_order` as a `read 8 bits`
    /// field, but the decoder constructor rejects a zero order (the
    /// §6.2.2 step-7 loop would read no vectors), so no curve could
    /// round-trip. The planner mirrors that gate.
    ZeroOrder,
    /// The value book's `dimensions` was zero. §6.2.2 step 7 adds
    /// `dimensions` coefficients per vector; a zero-dimension book would
    /// loop forever, so the decoder treats it as undecodable and the
    /// planner refuses it.
    ZeroDimensions,
    /// The value book carried no VQ vector lookup (`lookup_type == 0`).
    /// §6.2.2 step 7 unpacks each entry to a vector; a scalar-only book
    /// has none, so the floor-0 decoder rejects it at construction time.
    /// Mirrors [`QuantizeError::NoVectorForType0`].
    ScalarValueBook,
    /// `coefficients.len()` did not match the count the §6.2.2 step-7..11
    /// loop fills. The decoder reads `ceil(order / dimensions)` vectors,
    /// concatenating `dimensions` scalars each, so the target must carry
    /// exactly that many coefficients (a partial final vector is read in
    /// full, so the target length is `ceil(order / dimensions) *
    /// dimensions`, not `order`).
    CoefficientLengthMismatch {
        /// `ceil(order / dimensions) * dimensions` — the count the
        /// decoder fills (including the surplus of a partial final
        /// vector).
        expected: usize,
        /// The supplied `coefficients.len()`.
        actual: usize,
    },
    /// A per-vector [`quantize_vector`] call failed (e.g. a fully-unused
    /// value book, or a hand-built codebook with a desynced multiplicand
    /// table). The `vector` index and inner error are carried for context.
    Quantize {
        /// Index of the VQ vector (`0..ceil(order / dimensions)`) whose
        /// quantisation failed.
        vector: usize,
        /// The inner quantiser error.
        source: QuantizeError,
    },
}

impl core::fmt::Display for Floor0EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Floor0EncodeError::ZeroOrder => {
                write!(f, "vorbis floor0 encode: order is zero (§6.2.1)")
            }
            Floor0EncodeError::ZeroDimensions => write!(
                f,
                "vorbis floor0 encode: value book dimensions=0 (§6.2.2 step 7 would loop)"
            ),
            Floor0EncodeError::ScalarValueBook => write!(
                f,
                "vorbis floor0 encode: value book has lookup_type 0 (§6.2.2 step 7 needs a VQ vector)"
            ),
            Floor0EncodeError::CoefficientLengthMismatch { expected, actual } => write!(
                f,
                "vorbis floor0 encode: coefficients.len()={actual} != ceil(order/dimensions)*dimensions={expected} (§6.2.2 step 7..11)"
            ),
            Floor0EncodeError::Quantize { vector, source } => write!(
                f,
                "vorbis floor0 encode: vector-{vector} quantise failed: {source}"
            ),
        }
    }
}

impl std::error::Error for Floor0EncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Floor0EncodeError::Quantize { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// The number of VQ vectors a floor-0 packet body reads for a given
/// `order` and value-book `dimensions` — `ceil(order / dimensions)`, the
/// count the §6.2.2 step-7..11 loop iterates before `len(coefficients) >=
/// order` falls out. A partial final vector still counts as one full read
/// (its surplus scalars beyond `order` the §6.2.3 curve step discards).
#[must_use]
pub fn floor0_vector_count(order: usize, dimensions: usize) -> usize {
    debug_assert!(dimensions > 0, "caller must reject zero-dimension books");
    order.div_ceil(dimensions)
}

/// Plan one floor-0 packet's VQ entry run (Vorbis I §6.2.2 steps 7..11 in
/// the write direction): quantise a target LSP `coefficients` list into
/// the per-vector value-book entry indices the decoder reads back.
///
/// `coefficients` is the target LSP filter coefficient list — exactly
/// `ceil(order / dimensions) * dimensions` scalars (the count the decoder
/// fills, *including* the surplus of a partial final vector; see
/// [`floor0_vector_count`]). `book` is the value codebook the packet's
/// `[booknumber]` selects. `order` is `floor0_order`.
///
/// The returned `Vec<u32>` is the `entries` field of a
/// [`crate::encoder::Floor0Packet::Curve`]: one entry index per VQ vector,
/// in stream order, length `ceil(order / dimensions)`. The running
/// `[last]` accumulator is threaded vector to vector exactly as §6.2.2
/// steps 6/8/9 thread it on decode, so the chosen entries reconstruct the
/// nearest-entry approximation of `coefficients` bit-for-bit.
///
/// # Errors
///
/// Returns a [`Floor0EncodeError`] for a zero `order`, a value book with
/// zero dimensions or no VQ vector lookup, a `coefficients` length that
/// does not match the decoder's fill count, or a per-vector quantiser
/// failure (e.g. a fully-unused value book). Validation precedes any
/// quantisation; on error no partial entry run is returned.
pub fn plan_floor0_coefficients(
    coefficients: &[f32],
    book: &VorbisCodebook,
    order: usize,
) -> Result<Vec<u32>, Floor0EncodeError> {
    if order == 0 {
        return Err(Floor0EncodeError::ZeroOrder);
    }
    let dims = book.dimensions as usize;
    if dims == 0 {
        return Err(Floor0EncodeError::ZeroDimensions);
    }
    // §6.2.2 step 7 unpacks each entry to a vector; a scalar-only book
    // (lookup_type 0) has none. quantize_vector would reject it too, but
    // fail closed up front so the error names the floor-0 condition.
    if matches!(book.lookup, VqLookup::None) {
        return Err(Floor0EncodeError::ScalarValueBook);
    }

    let vectors = floor0_vector_count(order, dims);
    // The decoder concatenates `dimensions` scalars per vector and reads
    // `vectors` of them — so the target must carry the full padded length,
    // not just `order` (the final vector's surplus is read-and-discarded).
    let expected = vectors * dims;
    if coefficients.len() != expected {
        return Err(Floor0EncodeError::CoefficientLengthMismatch {
            expected,
            actual: coefficients.len(),
        });
    }

    // §6.2.2 step 6: [last] = 0. Threaded across vectors in lockstep with
    // the decoder — but advanced from each chosen entry's *reconstruction*
    // (not the target), so the planner's offset matches the value the
    // decoder will carry.
    let mut last = 0.0f32;
    let mut entries = Vec::with_capacity(vectors);

    for v in 0..vectors {
        let base = v * dims;
        // Un-offset the target sub-vector by the running [last] — the
        // inverse of the decoder's step-8 add — to get the raw VQ-decode
        // target the codebook quantises against.
        let mut target = Vec::with_capacity(dims);
        for j in 0..dims {
            target.push(coefficients[base + j] - last);
        }

        let q = quantize_vector(book, &target)
            .map_err(|source| Floor0EncodeError::Quantize { vector: v, source })?;

        // §6.2.2 step 8 re-adds the offset to the reconstruction, then
        // step 9 sets [last] to that offset reconstruction's final scalar.
        // The quantiser's `vector` is the raw reconstruction, so the
        // decoder's post-step-8 last scalar is `vector.last() + last`.
        last += *q
            .vector
            .last()
            .expect("a nonzero-dimension book unpacks to a nonempty vector");
        entries.push(q.entry);
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::VqLookup;
    use crate::vq::unpack_vector;

    /// Build a tessellation (lookup-2) VQ value codebook with explicit
    /// multiplicands (min 0, delta 1, sequence_p off), so entry `e`
    /// unpacks to `multiplicands[e*dims .. e*dims+dims]` as f32. Every
    /// entry is marked used (`codeword_lengths` all `1`); the planner's
    /// quantiser scans entries directly (no tree build), so a degenerate
    /// length set suffices to keep every entry reachable.
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

    /// Independent oracle: reconstruct the `[coefficients]` the §6.2.2
    /// step-7..11 loop would build from an entry run, by re-running the
    /// decode-side `[last]` accumulation directly. Does not call the
    /// planner — this is what the planner's round-trip is checked against.
    fn decode_reconstruct(entries: &[u32], book: &VorbisCodebook, order: usize) -> Vec<f32> {
        let mut coeffs = Vec::new();
        let mut last = 0.0f32;
        for &entry in entries {
            let mut temp = unpack_vector(book, entry).unwrap();
            for x in &mut temp {
                *x += last;
            }
            last = *temp.last().unwrap();
            coeffs.extend_from_slice(&temp);
            if coeffs.len() >= order {
                break;
            }
        }
        coeffs
    }

    // ---------- error paths ----------

    #[test]
    fn zero_order_is_rejected() {
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        assert_eq!(
            plan_floor0_coefficients(&[0.0, 0.0], &book, 0),
            Err(Floor0EncodeError::ZeroOrder)
        );
    }

    #[test]
    fn zero_dimension_book_is_rejected() {
        let book = tess_book(0, 1, vec![]);
        assert_eq!(
            plan_floor0_coefficients(&[], &book, 2),
            Err(Floor0EncodeError::ZeroDimensions)
        );
    }

    #[test]
    fn scalar_value_book_is_rejected() {
        let book = VorbisCodebook {
            dimensions: 2,
            entries: 2,
            codeword_lengths: vec![1; 2],
            lookup: VqLookup::None,
        };
        assert_eq!(
            plan_floor0_coefficients(&[0.0, 0.0], &book, 2),
            Err(Floor0EncodeError::ScalarValueBook)
        );
    }

    #[test]
    fn coefficient_length_mismatch_is_rejected() {
        // dims=2, order=3 → ceil(3/2)=2 vectors, expected length 4.
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        assert_eq!(
            plan_floor0_coefficients(&[1.0, 2.0, 3.0], &book, 3),
            Err(Floor0EncodeError::CoefficientLengthMismatch {
                expected: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn fully_unused_book_surfaces_quantize_error() {
        let book = VorbisCodebook {
            dimensions: 2,
            entries: 2,
            codeword_lengths: vec![crate::codebook::UNUSED_ENTRY; 2],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1],
            },
        };
        let err = plan_floor0_coefficients(&[0.0, 0.0], &book, 2).unwrap_err();
        assert!(matches!(
            err,
            Floor0EncodeError::Quantize {
                vector: 0,
                source: QuantizeError::NoUsableEntries,
            }
        ));
    }

    // ---------- vector-count helper ----------

    #[test]
    fn vector_count_rounds_up() {
        assert_eq!(floor0_vector_count(4, 2), 2); // exact
        assert_eq!(floor0_vector_count(3, 2), 2); // partial final
        assert_eq!(floor0_vector_count(1, 2), 1);
        assert_eq!(floor0_vector_count(5, 1), 5); // dims 1
    }

    // ---------- planning ----------

    #[test]
    fn single_vector_no_last_carry() {
        // dims == order: exactly one VQ vector, [last] never carries
        // across vectors. Entries unpack to [0,0],[1,1],[2,2],[3,3].
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let target = vec![2.0, 2.0];
        let entries = plan_floor0_coefficients(&target, &book, 2).unwrap();
        assert_eq!(entries, vec![2]);
        assert_eq!(decode_reconstruct(&entries, &book, 2), vec![2.0, 2.0]);
    }

    #[test]
    fn last_accumulator_threads_across_vectors() {
        // dims=1, order=3: three single-element vectors. The decoder adds
        // the running [last] to each, so coefficients are cumulative. The
        // planner un-offsets each vector by the prior reconstruction.
        // Entries unpack to {0,1,2,3,4}.
        let book = tess_book(1, 5, vec![0, 1, 2, 3, 4]);
        // Reconstructed coefficients [1,3,6]:
        //   vec0 raw 1 (last 0→1), vec1 raw 2 (last 1→3), vec2 raw 3 (last 3→6).
        let target = vec![1.0, 3.0, 6.0];
        let entries = plan_floor0_coefficients(&target, &book, 3).unwrap();
        assert_eq!(entries, vec![1, 2, 3]);
        assert_eq!(decode_reconstruct(&entries, &book, 3), target);
    }

    #[test]
    fn partial_final_vector_pads_target_length() {
        // dims=2, order=3: ceil(3/2)=2 vectors, target length 2*2=4 (the
        // final vector's surplus scalar is read then discarded by §6.2.3).
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        // vec0 → [2,2] (entry 2), last=2; vec1 raw target [3-2,3-2]=[1,1] → entry 1.
        let target = vec![2.0, 2.0, 3.0, 3.0];
        let entries = plan_floor0_coefficients(&target, &book, 3).unwrap();
        assert_eq!(entries, vec![2, 1]);
        let recon = decode_reconstruct(&entries, &book, 3);
        assert_eq!(&recon[..3], &target[..3]);
    }

    #[test]
    fn nearest_entry_when_target_inexact() {
        // A target no entry reconstructs exactly: the planner picks the
        // nearest. dims=1 entries {0,1,2,3,4}; target 2.4 → entry 2.
        let book = tess_book(1, 5, vec![0, 1, 2, 3, 4]);
        let target = vec![2.4];
        let entries = plan_floor0_coefficients(&target, &book, 1).unwrap();
        assert_eq!(entries, vec![2]);
        assert_eq!(decode_reconstruct(&entries, &book, 1), vec![2.0]);
    }

    #[test]
    fn lossy_last_tracks_reconstruction_not_target() {
        // The crux of the encode inverse: when a vector quantises
        // inexactly, the planner must advance [last] from the *chosen
        // entry's reconstruction*, not the target, so the next vector
        // un-offsets against the value the decoder will carry. dims=2,
        // order=4 → 2 vectors. Entries unpack to [0,0],[1,1],[2,2],[3,3].
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        // target [3,3, 1,1]:
        //   vec0 raw [3,3] → entry 3 (exact), last→3.
        //   vec1 raw [1-3,1-3]=[-2,-2] → nearest entry 0 ([0,0]), recon
        //   offsets back to [3,3]. So coefficients reconstruct to [3,3,3,3].
        let target = vec![3.0, 3.0, 1.0, 1.0];
        let entries = plan_floor0_coefficients(&target, &book, 4).unwrap();
        assert_eq!(entries, vec![3, 0]);
        assert_eq!(
            decode_reconstruct(&entries, &book, 4),
            vec![3.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn round_trip_multi_vector_exact() {
        // A reconstructable cumulative curve: every vector hits an entry
        // exactly, so the planner's run decode-reconstructs the target.
        // dims=2 entries [0,0]..[3,3]; order=4 → 2 vectors.
        let book = tess_book(2, 4, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        // Want coefficients [2,2, 5,5]:
        //   vec0 [2,2] entry 2, last=2; vec1 raw [5-2,5-2]=[3,3] entry 3.
        let target = vec![2.0, 2.0, 5.0, 5.0];
        let entries = plan_floor0_coefficients(&target, &book, 4).unwrap();
        assert_eq!(entries, vec![2, 3]);
        assert_eq!(decode_reconstruct(&entries, &book, 4), target);
    }
}

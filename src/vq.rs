//! Vorbis I VQ vector unpack (Vorbis I §3.2.1 "VQ lookup table vector
//! representation" + §3.3 "Use of the codebook abstraction").
//!
//! After a codebook is parsed (§3.2.1; see [`crate::codebook`]) and an
//! entry index has been recovered from the packet bitstream by walking
//! the canonical Huffman tree (§3.3; see [`crate::huffman`]), a VQ
//! context (anything other than a pure scalar-entropy context) needs to
//! transform that entry index into a fixed-dimension vector of floats.
//!
//! The transformation rule is dictated by the codebook's
//! `codebook_lookup_type`:
//!
//! * `0` — entropy-only codebook. No vector exists. The spec is
//!   emphatic: "requesting decode using a codebook of lookup type 0 in
//!   any context expecting a vector return value (even in a case where
//!   a vector of dimension one) is forbidden. If decoder setup or
//!   decode requests such an action, that is an error condition
//!   rendering the packet undecodable." This module surfaces that
//!   condition as [`UnpackError::NoVectorForType0`].
//!
//! * `1` — lattice VQ. The `codebook_multiplicands` table is small
//!   (its length is `lookup1_values(entries, dimensions)` per §9.2.3)
//!   and is mixed-base-permuted at decode time to yield the
//!   per-entry vector. Per §3.2.1 "Vector value decode: Lookup type 1":
//!
//!   ```text
//!   1) [last] = 0
//!   2) [index_divisor] = 1
//!   3) iterate [i] over the range 0 ... [codebook_dimensions]-1:
//!       4) [multiplicand_offset] = ([lookup_offset] /
//!          [index_divisor]) integer-modulo [codebook_lookup_values]
//!       5) value_vector[i] = (multiplicands[multiplicand_offset]) *
//!          [delta_value] + [minimum_value] + [last]
//!       6) if [sequence_p] is set, set [last] = value_vector[i]
//!       7) [index_divisor] = [index_divisor] *
//!          [codebook_lookup_values]
//!   8) vector calculation completed.
//!   ```
//!
//! * `2` — tessellation VQ. The `codebook_multiplicands` table is the
//!   full one-to-one map (length `entries × dimensions`). Per §3.2.1
//!   "Vector value decode: Lookup type 2":
//!
//!   ```text
//!   1) [last] = 0
//!   2) [multiplicand_offset] = [lookup_offset] * [codebook_dimensions]
//!   3) iterate [i] over the range 0 ... [codebook_dimensions]-1:
//!       4) value_vector[i] = (multiplicands[multiplicand_offset]) *
//!          [delta_value] + [minimum_value] + [last]
//!       5) if [sequence_p] is set, set [last] = value_vector[i]
//!       6) increment [multiplicand_offset]
//!   7) vector calculation completed.
//!   ```
//!
//! In both cases `[last]` is **cumulative**: when `sequence_p` is set,
//! the value just emitted is fed into the next iteration's addend, so a
//! `sequence_p = 1` vector is a running prefix-sum of the per-element
//! `multiplicand × delta + minimum` terms. (`sequence_p = 0` simply
//! holds `last = 0` for every iteration, making each element
//! independent.)
//!
//! The lookup index `[lookup_offset]` is the entry number returned by
//! [`crate::huffman::HuffmanTree::decode_entry`]. Vorbis residue decode
//! (§8.6) calls into this routine once per partition once per stage of
//! the cascade; this module provides the leaf transformation only —
//! the residue-side glue belongs in a future round.

use crate::codebook::{VorbisCodebook, VqLookup};

/// Errors that can arise while unpacking a VQ vector from a codebook
/// entry (Vorbis I §3.2.1 / §3.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnpackError {
    /// `lookup_offset` was outside the codebook's entry range, i.e.
    /// `lookup_offset >= codebook.entries`. §3.2.1 / §3.3 do not
    /// directly name this case because they assume the index came
    /// from a successful Huffman walk against the codebook's own tree
    /// (which by construction never returns an out-of-range entry);
    /// this is a safety net for callers that synthesise a lookup
    /// offset by other means.
    EntryOutOfRange {
        /// The bogus `lookup_offset`.
        lookup_offset: u32,
        /// The codebook's `entries` field, for context.
        entries: u32,
    },
    /// The codebook's `codebook_lookup_type` is `0` (entropy-only) and
    /// no VQ vector exists. Per §3.3: "requesting decode using a
    /// codebook of lookup type 0 in any context expecting a vector
    /// return value (even in a case where a vector of dimension one)
    /// is forbidden." Surface this as a structured error rather than
    /// returning an empty vector.
    NoVectorForType0,
    /// The codebook's `codebook_dimensions` is zero, leaving no
    /// per-entry vector to recover. §3.2.1 does not directly forbid
    /// a zero-dimensional codebook (the field is a 16-bit unsigned
    /// with no lower bound declared), but the §3.2.1 lookup-decode
    /// pseudocode iterates `[i]` over `0..dimensions-1` which is
    /// empty for `dimensions = 0` — the result would be a
    /// zero-length vector and no consumer of the codebook
    /// abstraction needs that. Treat it as a malformed stream.
    ZeroDimensions,
    /// The multiplicand table length doesn't match the codebook's
    /// declared shape. For lookup type 2 the table must have
    /// `entries × dimensions` elements; for lookup type 1 it must
    /// have `lookup1_values(entries, dimensions)` elements. A
    /// mismatch here would normally be a [`crate::codebook::parse_codebook`]
    /// bug; the check is conservative so a hand-constructed
    /// [`VorbisCodebook`] passed in from elsewhere can't desync the
    /// arithmetic.
    MultiplicandShapeMismatch {
        /// Recorded multiplicand count.
        got: usize,
        /// Expected multiplicand count.
        expected: usize,
    },
}

impl core::fmt::Display for UnpackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            UnpackError::EntryOutOfRange {
                lookup_offset,
                entries,
            } => write!(
                f,
                "vorbis VQ unpack: lookup_offset={lookup_offset} >= entries={entries} (§3.2.1)"
            ),
            UnpackError::NoVectorForType0 => write!(
                f,
                "vorbis VQ unpack: codebook_lookup_type = 0 cannot return a vector (§3.3)"
            ),
            UnpackError::ZeroDimensions => {
                write!(f, "vorbis VQ unpack: codebook_dimensions = 0 (§3.2.1)")
            }
            UnpackError::MultiplicandShapeMismatch { got, expected } => write!(
                f,
                "vorbis VQ unpack: multiplicand table length {got} != expected {expected} (§3.2.1)"
            ),
        }
    }
}

impl std::error::Error for UnpackError {}

/// Unpacks the VQ vector for codebook entry `lookup_offset` (Vorbis I
/// §3.2.1 "VQ lookup table vector representation").
///
/// On success, returns a freshly-allocated `Vec<f32>` of exactly
/// `codebook.dimensions as usize` floats. On any §3.2.1 / §3.3 error
/// condition, returns a structured [`UnpackError`].
///
/// The codebook's `codebook_lookup_type` selects the algorithm:
///
/// * [`VqLookup::None`] → [`UnpackError::NoVectorForType0`].
/// * [`VqLookup::Lattice`] → §3.2.1 "Vector value decode: Lookup type 1"
///   (mixed-base permutation of the small multiplicand table).
/// * [`VqLookup::Tessellation`] → §3.2.1 "Vector value decode: Lookup
///   type 2" (direct slice of the full multiplicand table).
///
/// `sequence_p` is honoured in both lookup-1 and lookup-2 paths:
/// when set, the running `[last]` accumulator carries the prior
/// `value_vector` element forward, making the output a prefix sum;
/// when clear, `[last]` stays at `0.0` and each element is independent.
pub fn unpack_vector(
    codebook: &VorbisCodebook,
    lookup_offset: u32,
) -> Result<Vec<f32>, UnpackError> {
    if codebook.dimensions == 0 {
        return Err(UnpackError::ZeroDimensions);
    }
    if lookup_offset >= codebook.entries {
        return Err(UnpackError::EntryOutOfRange {
            lookup_offset,
            entries: codebook.entries,
        });
    }
    let dims = codebook.dimensions as usize;

    match &codebook.lookup {
        VqLookup::None => Err(UnpackError::NoVectorForType0),
        VqLookup::Lattice {
            minimum_value,
            delta_value,
            sequence_p,
            multiplicands,
            ..
        } => unpack_lattice(
            *minimum_value,
            *delta_value,
            *sequence_p,
            multiplicands,
            codebook.entries,
            dims,
            lookup_offset,
        ),
        VqLookup::Tessellation {
            minimum_value,
            delta_value,
            sequence_p,
            multiplicands,
            ..
        } => unpack_tessellation(
            *minimum_value,
            *delta_value,
            *sequence_p,
            multiplicands,
            codebook.entries,
            dims,
            lookup_offset,
        ),
    }
}

/// §3.2.1 "Vector value decode: Lookup type 1" — the lattice path.
fn unpack_lattice(
    minimum_value: f32,
    delta_value: f32,
    sequence_p: bool,
    multiplicands: &[u32],
    entries: u32,
    dims: usize,
    lookup_offset: u32,
) -> Result<Vec<f32>, UnpackError> {
    // The recorded multiplicand count for a type-1 codebook is
    // `lookup1_values(entries, dimensions)`; we cross-check against
    // the supplied table to catch hand-constructed `VorbisCodebook`s
    // that desync the arithmetic.
    let lookup_values = crate::codebook::lookup1_values(entries, dims as u16) as usize;
    if multiplicands.len() != lookup_values {
        return Err(UnpackError::MultiplicandShapeMismatch {
            got: multiplicands.len(),
            expected: lookup_values,
        });
    }
    if lookup_values == 0 {
        return Err(UnpackError::MultiplicandShapeMismatch {
            got: 0,
            expected: 0,
        });
    }

    let mut value_vector = Vec::with_capacity(dims);
    let mut last: f32 = 0.0;
    // §3.2.1 step 2: index_divisor = 1
    let mut index_divisor: u32 = 1;
    // The spec's `[lookup_offset]` is the codebook entry index. Keep
    // it as `u32` throughout the mixed-base extraction (each per-dim
    // digit is in `0..lookup_values`, which for any realistic codebook
    // fits in `u32` by a wide margin).
    let lookup_values_u32 = lookup_values as u32;
    for _i in 0..dims {
        // §3.2.1 step 4: multiplicand_offset =
        //   (lookup_offset / index_divisor) integer-modulo lookup_values
        let mult_off = (lookup_offset / index_divisor) % lookup_values_u32;
        // §3.2.1 step 5: value_vector[i] =
        //   multiplicands[mult_off] * delta_value + minimum_value + last
        let v = (multiplicands[mult_off as usize] as f32) * delta_value + minimum_value + last;
        value_vector.push(v);
        // §3.2.1 step 6: if sequence_p, last = value_vector[i]
        if sequence_p {
            last = v;
        }
        // §3.2.1 step 7: index_divisor *= lookup_values
        // (Saturate so a pathological combination doesn't wrap and
        // corrupt the next iteration's modulo. For real Vorbis
        // codebooks `lookup_values^dimensions <= entries < 2^24`, so
        // saturation is decoder-fence, not a hot path.)
        index_divisor = index_divisor.saturating_mul(lookup_values_u32);
    }
    Ok(value_vector)
}

/// §3.2.1 "Vector value decode: Lookup type 2" — the tessellation path.
fn unpack_tessellation(
    minimum_value: f32,
    delta_value: f32,
    sequence_p: bool,
    multiplicands: &[u32],
    entries: u32,
    dims: usize,
    lookup_offset: u32,
) -> Result<Vec<f32>, UnpackError> {
    // The recorded multiplicand count for a type-2 codebook is
    // `entries × dimensions`.
    let expected =
        (entries as usize)
            .checked_mul(dims)
            .ok_or(UnpackError::MultiplicandShapeMismatch {
                got: multiplicands.len(),
                expected: usize::MAX,
            })?;
    if multiplicands.len() != expected {
        return Err(UnpackError::MultiplicandShapeMismatch {
            got: multiplicands.len(),
            expected,
        });
    }

    let mut value_vector = Vec::with_capacity(dims);
    let mut last: f32 = 0.0;
    // §3.2.1 step 2: multiplicand_offset = lookup_offset * dimensions
    let base = (lookup_offset as usize) * dims;
    for i in 0..dims {
        // §3.2.1 step 4: value_vector[i] =
        //   multiplicands[multiplicand_offset] * delta + min + last
        let v = (multiplicands[base + i] as f32) * delta_value + minimum_value + last;
        value_vector.push(v);
        // §3.2.1 step 5: if sequence_p, last = value_vector[i]
        if sequence_p {
            last = v;
        }
        // §3.2.1 step 6: increment multiplicand_offset (the `+ i` in
        // the indexing expression already encodes this; nothing else
        // to do per iteration).
    }
    Ok(value_vector)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::VqLookup;

    /// Builds a minimal [`VorbisCodebook`] for vector-unpack testing.
    /// The Huffman codeword lengths are filled in as `1` for every
    /// entry to keep the struct internally consistent (the unpack
    /// routine never reads them, but a downstream test that round-trips
    /// through `parse_codebook` would need a well-formed length list).
    fn make_codebook(dimensions: u16, entries: u32, lookup: VqLookup) -> VorbisCodebook {
        VorbisCodebook {
            dimensions,
            entries,
            codeword_lengths: vec![1; entries as usize],
            lookup,
        }
    }

    // ---------- error paths ----------

    /// §3.3: "requesting decode using a codebook of lookup type 0 in
    /// any context expecting a vector return value ... is forbidden."
    #[test]
    fn type0_returns_no_vector_error() {
        let cb = make_codebook(2, 4, VqLookup::None);
        assert_eq!(unpack_vector(&cb, 0), Err(UnpackError::NoVectorForType0));
    }

    /// Out-of-range `lookup_offset` is rejected before any arithmetic.
    #[test]
    fn out_of_range_lookup_offset_is_rejected() {
        let cb = make_codebook(
            1,
            2,
            VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0],
            },
        );
        assert_eq!(
            unpack_vector(&cb, 2),
            Err(UnpackError::EntryOutOfRange {
                lookup_offset: 2,
                entries: 2,
            })
        );
    }

    /// `dimensions = 0` produces a structured error rather than an
    /// empty vector.
    #[test]
    fn zero_dimensions_is_rejected() {
        let cb = make_codebook(
            0,
            1,
            VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![],
            },
        );
        assert_eq!(unpack_vector(&cb, 0), Err(UnpackError::ZeroDimensions));
    }

    /// A type-2 codebook with the wrong-shape multiplicand table is
    /// surfaced as a structured error.
    #[test]
    fn tessellation_shape_mismatch_is_caught() {
        let cb = make_codebook(
            2,
            3,
            VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                // 3 entries × 2 dims = 6 expected, give 5.
                multiplicands: vec![0, 1, 2, 3, 4],
            },
        );
        assert_eq!(
            unpack_vector(&cb, 0),
            Err(UnpackError::MultiplicandShapeMismatch {
                got: 5,
                expected: 6,
            })
        );
    }

    /// A type-1 codebook with the wrong-shape multiplicand table is
    /// surfaced as a structured error. `lookup1_values(4, 2) = 2`
    /// (`2^2 = 4 <= 4`, `3^2 = 9 > 4`), so the expected length is 2.
    #[test]
    fn lattice_shape_mismatch_is_caught() {
        let cb = make_codebook(
            2,
            4,
            VqLookup::Lattice {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 1, 2],
            },
        );
        assert_eq!(
            unpack_vector(&cb, 0),
            Err(UnpackError::MultiplicandShapeMismatch {
                got: 3,
                expected: 2,
            })
        );
    }

    // ---------- §3.2.1 lookup type 2 (tessellation) ----------

    /// 2-D tessellation, `sequence_p = 0`, identity delta/minimum.
    /// Entry 0 → multiplicands[0..2], entry 1 → multiplicands[2..4],
    /// entry 2 → multiplicands[4..6].
    #[test]
    fn tessellation_independent_2d_identity() {
        let cb = make_codebook(
            2,
            3,
            VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![1, 2, 3, 5, 8, 13],
            },
        );
        assert_eq!(unpack_vector(&cb, 0).unwrap(), vec![1.0, 2.0]);
        assert_eq!(unpack_vector(&cb, 1).unwrap(), vec![3.0, 5.0]);
        assert_eq!(unpack_vector(&cb, 2).unwrap(), vec![8.0, 13.0]);
    }

    /// 2-D tessellation, `sequence_p = 0`, non-trivial delta + min:
    /// `v[i] = mult[i] * 2.0 + 0.5`.
    #[test]
    fn tessellation_independent_with_min_and_delta() {
        let cb = make_codebook(
            2,
            2,
            VqLookup::Tessellation {
                minimum_value: 0.5,
                delta_value: 2.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![1, 3, 5, 7],
            },
        );
        // entry 0: [1*2+0.5, 3*2+0.5] = [2.5, 6.5]
        assert_eq!(unpack_vector(&cb, 0).unwrap(), vec![2.5, 6.5]);
        // entry 1: [5*2+0.5, 7*2+0.5] = [10.5, 14.5]
        assert_eq!(unpack_vector(&cb, 1).unwrap(), vec![10.5, 14.5]);
    }

    /// 3-D tessellation, `sequence_p = 1` (cumulative). For entry 0
    /// with multiplicands `[1, 2, 3]`, delta `1.0`, min `0.0`:
    ///
    ///   last=0; v[0]=1*1+0+0=1, last=1;
    ///           v[1]=2*1+0+1=3, last=3;
    ///           v[2]=3*1+0+3=6, last=6
    ///
    /// → `[1, 3, 6]` (the prefix sum of `[1, 2, 3]`).
    #[test]
    fn tessellation_cumulative_is_prefix_sum() {
        let cb = make_codebook(
            3,
            1,
            VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: true,
                multiplicands: vec![1, 2, 3],
            },
        );
        assert_eq!(unpack_vector(&cb, 0).unwrap(), vec![1.0, 3.0, 6.0]);
    }

    /// `sequence_p = 1` with nonzero min: each step adds
    /// `mult[i] * delta + min` to the running sum.
    ///
    /// entries = 1, dims = 4, mult = [0,1,2,3], delta = 1, min = 0.5:
    ///   last=0; v[0]=0+0.5+0=0.5, last=0.5
    ///           v[1]=1+0.5+0.5=2.0, last=2.0
    ///           v[2]=2+0.5+2.0=4.5, last=4.5
    ///           v[3]=3+0.5+4.5=8.0
    /// → [0.5, 2.0, 4.5, 8.0]
    #[test]
    fn tessellation_cumulative_with_min_offset() {
        let cb = make_codebook(
            4,
            1,
            VqLookup::Tessellation {
                minimum_value: 0.5,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: true,
                multiplicands: vec![0, 1, 2, 3],
            },
        );
        assert_eq!(unpack_vector(&cb, 0).unwrap(), vec![0.5, 2.0, 4.5, 8.0]);
    }

    // ---------- §3.2.1 lookup type 1 (lattice) ----------

    /// 2-D lattice with `entries = 4`, `lookup_values = 2`:
    ///   entry 0 → multiplicand_offset for i=0: (0/1) mod 2 = 0
    ///                                 for i=1: (0/2) mod 2 = 0
    ///   entry 1 → i=0: (1/1) mod 2 = 1
    ///             i=1: (1/2) mod 2 = 0
    ///   entry 2 → i=0: (2/1) mod 2 = 0
    ///             i=1: (2/2) mod 2 = 1
    ///   entry 3 → i=0: (3/1) mod 2 = 1
    ///             i=1: (3/2) mod 2 = 1
    ///
    /// With multiplicands `[10, 20]`, delta `1.0`, min `0.0`:
    ///   entry 0 → [10, 10]
    ///   entry 1 → [20, 10]
    ///   entry 2 → [10, 20]
    ///   entry 3 → [20, 20]
    #[test]
    fn lattice_independent_2d_mixed_base() {
        let cb = make_codebook(
            2,
            4,
            VqLookup::Lattice {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![10, 20],
            },
        );
        assert_eq!(unpack_vector(&cb, 0).unwrap(), vec![10.0, 10.0]);
        assert_eq!(unpack_vector(&cb, 1).unwrap(), vec![20.0, 10.0]);
        assert_eq!(unpack_vector(&cb, 2).unwrap(), vec![10.0, 20.0]);
        assert_eq!(unpack_vector(&cb, 3).unwrap(), vec![20.0, 20.0]);
    }

    /// 3-D lattice with `entries = 8`, `lookup_values = 2` (since
    /// `2^3 = 8 <= 8` and `3^3 = 27 > 8`). For entry index 5 = 0b101:
    ///   i=0: (5/1) mod 2 = 1
    ///   i=1: (5/2) mod 2 = 0
    ///   i=2: (5/4) mod 2 = 1
    /// → multiplicands picked: [mult[1], mult[0], mult[1]].
    /// With mult = [3, 7], delta = 1.0, min = 0.5: [7.5, 3.5, 7.5].
    #[test]
    fn lattice_independent_3d_binary_lattice() {
        let cb = make_codebook(
            3,
            8,
            VqLookup::Lattice {
                minimum_value: 0.5,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![3, 7],
            },
        );
        assert_eq!(unpack_vector(&cb, 5).unwrap(), vec![7.5, 3.5, 7.5]);
    }

    /// Lattice with `sequence_p = 1` is the prefix-sum of the
    /// independent path. 2-D, entries=4, lookup_values=2, mult=[10,20]
    /// delta=1, min=0:
    ///   entry 0 independent → [10, 10] → cumulative [10, 20]
    ///   entry 1 independent → [20, 10] → cumulative [20, 30]
    ///   entry 2 independent → [10, 20] → cumulative [10, 30]
    ///   entry 3 independent → [20, 20] → cumulative [20, 40]
    #[test]
    fn lattice_cumulative_is_prefix_sum() {
        let cb = make_codebook(
            2,
            4,
            VqLookup::Lattice {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: true,
                multiplicands: vec![10, 20],
            },
        );
        assert_eq!(unpack_vector(&cb, 0).unwrap(), vec![10.0, 20.0]);
        assert_eq!(unpack_vector(&cb, 1).unwrap(), vec![20.0, 30.0]);
        assert_eq!(unpack_vector(&cb, 2).unwrap(), vec![10.0, 30.0]);
        assert_eq!(unpack_vector(&cb, 3).unwrap(), vec![20.0, 40.0]);
    }

    // ---------- per-codebook §3.3 round-trip ----------

    /// Round-trip the trace-doc §3 worked example: an 8-entry codebook
    /// configured for tessellation VQ, 8 dimensions. We hand-author a
    /// codebook (no Huffman tree needed for the unpack itself, since
    /// the unpack only consumes `entries`, `dimensions`, and `lookup`),
    /// then check every entry against a manually-computed reference.
    #[test]
    fn round_trip_8x8_tessellation() {
        let entries: u32 = 8;
        let dims: u16 = 8;
        // 64-element table: m[entry*8 + i] = entry * 8 + i (mod 256).
        let multiplicands: Vec<u32> = (0..64).collect();
        let cb = make_codebook(
            dims,
            entries,
            VqLookup::Tessellation {
                minimum_value: -1.0,
                delta_value: 0.5,
                value_bits: 8,
                sequence_p: false,
                multiplicands: multiplicands.clone(),
            },
        );
        for entry in 0..entries {
            let got = unpack_vector(&cb, entry).unwrap();
            assert_eq!(got.len(), dims as usize);
            for i in 0..(dims as usize) {
                let m = multiplicands[entry as usize * dims as usize + i];
                let expected = (m as f32) * 0.5 + (-1.0);
                assert_eq!(got[i], expected, "entry={entry} i={i}");
            }
        }
    }

    /// Round-trip every entry of a small lattice codebook. With
    /// `entries=9`, `dimensions=2`, `lookup_values = 3` (since
    /// `3^2 = 9 <= 9`, `4^2 = 16 > 9`), each entry index `e`
    /// decomposes as `(e mod 3, (e/3) mod 3)`. With multiplicands
    /// `[7, 11, 13]`, delta `1.0`, min `0.0`, we expect:
    ///   e=0 → [7, 7]    (digits 0, 0)
    ///   e=1 → [11, 7]   (digits 1, 0)
    ///   e=2 → [13, 7]   (digits 2, 0)
    ///   e=3 → [7, 11]   (digits 0, 1)
    ///   ...
    ///   e=8 → [13, 13]  (digits 2, 2)
    #[test]
    fn round_trip_3x3_lattice() {
        let multiplicands = vec![7u32, 11, 13];
        let cb = make_codebook(
            2,
            9,
            VqLookup::Lattice {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: multiplicands.clone(),
            },
        );
        for e in 0u32..9 {
            let got = unpack_vector(&cb, e).unwrap();
            let d0 = (e % 3) as usize;
            let d1 = ((e / 3) % 3) as usize;
            assert_eq!(
                got,
                vec![multiplicands[d0] as f32, multiplicands[d1] as f32],
                "entry={e}"
            );
        }
    }

    /// Sanity-check that the `sequence_p` semantics match the spec's
    /// literal text: when `sequence_p = 1`, `[last]` is updated to
    /// `value_vector[i]` (the post-min, post-delta, post-last value),
    /// not to `multiplicands[...] * delta` alone. The distinction
    /// matters whenever `minimum_value != 0`. Specifically, the
    /// `min_offset` test above already exercises this: each step adds
    /// `min` *plus* the previous full value, not just `mult * delta`.
    #[test]
    fn sequence_p_carries_full_value_not_just_mult_delta() {
        let cb = make_codebook(
            3,
            1,
            VqLookup::Tessellation {
                minimum_value: 1.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: true,
                multiplicands: vec![0, 0, 0],
            },
        );
        // Every multiplicand is 0, so each step adds (0*1 + 1 + last):
        //   v[0] = 0 + 1 + 0 = 1, last = 1
        //   v[1] = 0 + 1 + 1 = 2, last = 2
        //   v[2] = 0 + 1 + 2 = 3
        // If `last` were instead set to `mult * delta` (the wrong
        // reading), every step would be 0+1+0=1, giving [1,1,1].
        assert_eq!(unpack_vector(&cb, 0).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    /// Sanity-check that `sequence_p = 0` keeps `[last]` pinned at 0
    /// and never carries the previous value forward.
    #[test]
    fn sequence_p_zero_keeps_each_element_independent() {
        let cb = make_codebook(
            3,
            1,
            VqLookup::Tessellation {
                minimum_value: 1.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 0],
            },
        );
        // sequence_p=0 ⇒ last stays 0 ⇒ each step = 0*1 + 1 + 0 = 1.
        assert_eq!(unpack_vector(&cb, 0).unwrap(), vec![1.0, 1.0, 1.0]);
    }
}

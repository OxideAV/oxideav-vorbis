//! Lookup-type optimiser for Vorbis VQ codebooks.
//!
//! Vorbis I §3.2.1 defines three "lookup" formats for codebook value
//! tables:
//!
//! * `lookup_type 0` — no value table at all. The codeword indexes a
//!   scalar entry whose meaning is supplied by the consumer (floor1 Y
//!   amplitudes, residue partition class numbers, ...). Bitstream cost
//!   beyond the Huffman length list and the 4-bit lookup-type field is
//!   zero.
//!
//! * `lookup_type 1` — implicit Cartesian-grid lookup. The bitstream
//!   stores `lookup_values` multiplicands of `value_bits` bits each,
//!   where `lookup_values^dim ≥ entries`. Entry `e`'s `d`-th coordinate
//!   decodes to
//!   `multiplicands[(e / lookup_values^d) mod lookup_values] * delta + min`.
//!   Cheap when `entries ≈ lookup_values^dim` and all per-dim values
//!   share the same grid.
//!
//! * `lookup_type 2` — flat per-entry table. The bitstream stores
//!   `entries × dim` multiplicands, one per (entry, coord) pair. Most
//!   general but most expensive.
//!
//! libvorbis-style trained books are produced as flat per-entry data
//! (lookup_type 2). When the per-coordinate value distribution actually
//! lies on a small Cartesian grid — the typical case for hand-tuned
//! `{−5..+5}^d` style residue books and for grid-quantised LBG output —
//! the same content can be re-emitted as lookup_type 1 at a fraction of
//! the multiplicand-bits cost.
//!
//! ## What this module does
//!
//! [`detect_lookup1`] inspects a `lookup_type 2` codebook and either
//! returns a `lookup_type 1` rewrite that decodes bit-identically, or
//! reports why the rewrite is impossible (off-grid values, entries that
//! don't permute the Cartesian product, ...). [`optimise`] is the
//! convenience wrapper that promotes a codebook in place when a smaller
//! form exists and leaves it untouched otherwise.
//!
//! ## What this module does NOT do
//!
//! It does not modify the encoder's choice of `value_bits`, `min`, or
//! `delta` — the value-quantisation grid is taken as already chosen.
//! It does not retrain VQ centroids onto a grid; off-grid trained books
//! (e.g. the LBG centroids in `trained_books.rs`) decline to optimise
//! and remain at lookup_type 2.

use crate::codebook::{Codebook, VqLookup};

/// Maximum per-dimension distinct value count we'll consider for
/// lookup_type 1 promotion. The Vorbis decoder accepts up to
/// `value_bits = 16` (32-bit address space for multiplicands), so the
/// hard ceiling is 65535. In practice the heuristic ceiling is much
/// lower: trained books that genuinely cluster on a grid use at most
/// ~32 distinct values per dim before the grid is so dense that
/// lookup_type 2 storage `(entries × dim)` beats lookup_type 1 storage
/// `(lookup_values × value_bits)` again. 32 is a soft cap here — the
/// detector still tries any value_bits the input codebook claims, but
/// books with > `MAX_GRID_PER_DIM` distinct per-dim values are
/// rejected without even attempting the grid fit.
pub const MAX_GRID_PER_DIM: usize = 32;

/// Result of inspecting a codebook for lookup-type-1 promotion.
#[derive(Debug, Clone)]
pub enum LookupOptimisation {
    /// The codebook is already at minimum size for its content. No
    /// rewrite produced.
    AlreadyMinimal,
    /// The codebook can be promoted to `lookup_type 1` with the
    /// supplied `VqLookup`. Substituting `cb.vq = Some(new_vq)` on the
    /// original codebook yields a smaller setup-header footprint while
    /// decoding bit-identically through `Codebook::vq_lookup`.
    PromoteToLookup1 {
        /// The promoted VQ table.
        new_vq: VqLookup,
        /// Multiplicand-byte savings (positive number).
        bits_saved: usize,
    },
    /// The codebook cannot be losslessly compressed further. The
    /// `reason` string is for diagnostic logging only — callers should
    /// branch on the variant, not the message.
    NotPossible { reason: &'static str },
}

/// Inspect `cb` and report whether it can be losslessly promoted from
/// `lookup_type 2` to `lookup_type 1`.
///
/// Returns [`LookupOptimisation::AlreadyMinimal`] if the codebook is
/// already at lookup_type 0 or 1 (no further reduction possible by
/// this pass). Returns [`LookupOptimisation::NotPossible`] when the
/// codebook is lookup_type 2 but its per-entry data does not lie on a
/// Cartesian grid (the trained-book common case). Returns
/// [`LookupOptimisation::PromoteToLookup1`] with the new VQ table when
/// the rewrite is exact.
///
/// The detector treats multiplicand integers (not floats), so all
/// "equality" checks are exact `u32` comparisons — no floating-point
/// epsilon. This is why the detector only needs to see the codebook's
/// already-quantised `multiplicands` table; `min` / `delta` /
/// `value_bits` are passed through unchanged.
pub fn detect_lookup1(cb: &Codebook) -> LookupOptimisation {
    let vq = match &cb.vq {
        Some(vq) => vq,
        None => return LookupOptimisation::AlreadyMinimal,
    };
    if vq.lookup_type != 2 {
        return LookupOptimisation::AlreadyMinimal;
    }
    let dim = cb.dimensions as usize;
    let entries = cb.entries as usize;
    if dim == 0 || entries == 0 {
        return LookupOptimisation::AlreadyMinimal;
    }
    if vq.multiplicands.len() != entries.saturating_mul(dim) {
        return LookupOptimisation::NotPossible {
            reason: "lookup_type 2 multiplicand count mismatch",
        };
    }

    // Step 1: collect the union of distinct multiplicand values across
    // ALL entries and dimensions. lookup_type 1 uses one shared grid,
    // so per-dim grids must coincide.
    let mut grid_set: Vec<u32> = Vec::with_capacity(MAX_GRID_PER_DIM + 1);
    for &m in &vq.multiplicands {
        if !grid_set.contains(&m) {
            if grid_set.len() >= MAX_GRID_PER_DIM {
                return LookupOptimisation::NotPossible {
                    reason: "values exceed MAX_GRID_PER_DIM distinct per dim",
                };
            }
            grid_set.push(m);
        }
    }
    grid_set.sort_unstable();
    let lookup_values = grid_set.len();

    // Step 2: lookup_values^dim must be ≥ entries (otherwise the
    // Cartesian product can't index every entry).
    let product = match pow_u64(lookup_values as u64, dim as u32) {
        Some(p) if p >= entries as u64 => p,
        _ => {
            return LookupOptimisation::NotPossible {
                reason: "grid^dim < entries (cannot index all entries)",
            };
        }
    };
    let _ = product; // silence unused-but-checked warning
    if lookup_values == 0 {
        return LookupOptimisation::NotPossible {
            reason: "empty grid",
        };
    }

    // Step 3: each entry's coordinates must equal the lookup_type-1
    // decode formula `grid[(e / lookup_values^d) mod lookup_values]`.
    // We must also pick the SMALLEST `lookup_values` such that the
    // decode formula matches; the `grid_set.len()` we computed is the
    // count of distinct values, which is the minimal grid size.
    for e in 0..entries {
        let mut idx_div: u64 = 1;
        for d in 0..dim {
            let mult_index = ((e as u64 / idx_div) as usize) % lookup_values;
            let want = grid_set[mult_index];
            let got = vq.multiplicands[e * dim + d];
            if got != want {
                return LookupOptimisation::NotPossible {
                    reason: "entry coords do not match Cartesian-grid index decomposition",
                };
            }
            idx_div = idx_div.saturating_mul(lookup_values as u64).max(1);
        }
    }

    // Step 4: report savings. lookup_type 2 stores `entries × dim`
    // values; lookup_type 1 stores `lookup_values`. Both use the same
    // `value_bits`.
    let old_bits = entries * dim * vq.value_bits as usize;
    let new_bits = lookup_values * vq.value_bits as usize;
    if new_bits >= old_bits {
        // No win — refuse the promotion. Can happen if entries ×
        // dim ≤ lookup_values, e.g. tiny codebooks where the grid
        // happens to hold every entry distinctly.
        return LookupOptimisation::NotPossible {
            reason: "lookup_type 1 form is not smaller",
        };
    }
    let bits_saved = old_bits - new_bits;

    let new_vq = VqLookup {
        lookup_type: 1,
        min: vq.min,
        delta: vq.delta,
        value_bits: vq.value_bits,
        sequence_p: vq.sequence_p,
        multiplicands: grid_set,
    };
    LookupOptimisation::PromoteToLookup1 { new_vq, bits_saved }
}

/// Apply [`detect_lookup1`] to `cb` and replace `cb.vq` in place when a
/// promotion is available. Returns the bits saved (0 if no rewrite was
/// applied). The codebook's Huffman state (`codeword_lengths`,
/// `codewords`) is untouched.
pub fn optimise(cb: &mut Codebook) -> usize {
    match detect_lookup1(cb) {
        LookupOptimisation::PromoteToLookup1 { new_vq, bits_saved } => {
            cb.vq = Some(new_vq);
            bits_saved
        }
        _ => 0,
    }
}

/// Apply [`optimise`] to every codebook in `setup` and return the total
/// bits saved across all books.
pub fn optimise_setup(setup: &mut crate::setup::Setup) -> usize {
    let mut total = 0usize;
    for cb in &mut setup.codebooks {
        total += optimise(cb);
    }
    total
}

/// `base^exp` checked against `u64` overflow. Returns `None` on
/// overflow, `Some(value)` otherwise.
fn pow_u64(base: u64, exp: u32) -> Option<u64> {
    let mut acc: u64 = 1;
    for _ in 0..exp {
        acc = acc.checked_mul(base)?;
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::{Codebook, VqLookup};

    /// Build a synthetic dim-2 lookup_type-2 codebook whose 9 entries
    /// span a 3×3 grid {0, 5, 10}. This is the canonical case where
    /// promotion to lookup_type 1 with `lookup_values = 3` shrinks the
    /// multiplicand table from 9*2 = 18 values to 3.
    fn grid_3x3() -> Codebook {
        let grid = [0u32, 5, 10];
        let dim = 2;
        let entries = 9u32;
        // entry e -> coords (grid[e%3], grid[e/3])
        let mut multiplicands = Vec::with_capacity(18);
        for e in 0..entries {
            multiplicands.push(grid[(e as usize) % 3]);
            multiplicands.push(grid[(e as usize) / 3]);
        }
        Codebook {
            dimensions: dim,
            entries,
            codeword_lengths: vec![4u8; entries as usize],
            vq: Some(VqLookup {
                lookup_type: 2,
                min: -10.0,
                delta: 1.0,
                value_bits: 4,
                sequence_p: false,
                multiplicands,
            }),
            codewords: Vec::new(),
        }
    }

    #[test]
    fn detect_promotes_3x3_grid_book() {
        let cb = grid_3x3();
        match detect_lookup1(&cb) {
            LookupOptimisation::PromoteToLookup1 { new_vq, bits_saved } => {
                assert_eq!(new_vq.lookup_type, 1);
                assert_eq!(new_vq.multiplicands, vec![0, 5, 10]);
                // 18*4 - 3*4 = 60 bits saved
                assert_eq!(bits_saved, 60);
            }
            other => panic!("expected promotion, got {other:?}"),
        }
    }

    #[test]
    fn promoted_book_decodes_identically() {
        let original = grid_3x3();
        let mut promoted = original.clone();
        optimise(&mut promoted);
        assert_eq!(promoted.vq.as_ref().unwrap().lookup_type, 1);
        for e in 0..original.entries {
            let a = original.vq_lookup(e).unwrap();
            let b = promoted.vq_lookup(e).unwrap();
            assert_eq!(a.len(), b.len(), "dim mismatch for entry {e}");
            for (ax, bx) in a.iter().zip(b.iter()) {
                assert!((ax - bx).abs() < 1e-6, "entry {e} mismatch: {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn detect_refuses_off_grid_book() {
        // dim 2, 4 entries: (0,0), (1,1), (2,3), (4,5) — the last entry's
        // coords (4, 5) are not on any 4-element shared grid.
        let multiplicands = vec![0, 0, 1, 1, 2, 3, 4, 5];
        let cb = Codebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![2u8; 4],
            vq: Some(VqLookup {
                lookup_type: 2,
                min: 0.0,
                delta: 1.0,
                value_bits: 4,
                sequence_p: false,
                multiplicands,
            }),
            codewords: Vec::new(),
        };
        match detect_lookup1(&cb) {
            LookupOptimisation::NotPossible { .. } => {}
            other => panic!("expected NotPossible, got {other:?}"),
        }
    }

    #[test]
    fn detect_skips_lookup_type_0() {
        let cb = Codebook {
            dimensions: 1,
            entries: 4,
            codeword_lengths: vec![2u8; 4],
            vq: None,
            codewords: Vec::new(),
        };
        match detect_lookup1(&cb) {
            LookupOptimisation::AlreadyMinimal => {}
            other => panic!("expected AlreadyMinimal, got {other:?}"),
        }
    }

    #[test]
    fn detect_skips_lookup_type_1() {
        let cb = Codebook {
            dimensions: 2,
            entries: 9,
            codeword_lengths: vec![4u8; 9],
            vq: Some(VqLookup {
                lookup_type: 1,
                min: 0.0,
                delta: 1.0,
                value_bits: 4,
                sequence_p: false,
                multiplicands: vec![0, 5, 10],
            }),
            codewords: Vec::new(),
        };
        match detect_lookup1(&cb) {
            LookupOptimisation::AlreadyMinimal => {}
            other => panic!("expected AlreadyMinimal, got {other:?}"),
        }
    }

    #[test]
    fn detect_refuses_too_many_distinct_values() {
        // Build a 1-D, 64-entry book where every entry has a distinct
        // multiplicand value. With dim=1, lookup_type 1 stores `entries`
        // values (no compression vs lookup_type 2), so the heuristic
        // shouldn't promote even if the grid is technically Cartesian.
        let mut multiplicands = Vec::with_capacity(64);
        for v in 0..64u32 {
            multiplicands.push(v);
        }
        let cb = Codebook {
            dimensions: 1,
            entries: 64,
            codeword_lengths: vec![6u8; 64],
            vq: Some(VqLookup {
                lookup_type: 2,
                min: 0.0,
                delta: 1.0,
                value_bits: 6,
                sequence_p: false,
                multiplicands,
            }),
            codewords: Vec::new(),
        };
        // 64 distinct values > MAX_GRID_PER_DIM = 32 → reject.
        match detect_lookup1(&cb) {
            LookupOptimisation::NotPossible { .. } => {}
            other => panic!("expected NotPossible (cap), got {other:?}"),
        }
    }

    #[test]
    fn detect_refuses_when_promotion_does_not_shrink() {
        // dim 1, 4 entries, 4 distinct values. lookup_type 2 stores
        // 4*4 = 16 bits; lookup_type 1 with lookup_values=4 stores
        // 4*4 = 16 bits — no win. Detector should reject.
        let multiplicands = vec![0u32, 1, 2, 3];
        let cb = Codebook {
            dimensions: 1,
            entries: 4,
            codeword_lengths: vec![2u8; 4],
            vq: Some(VqLookup {
                lookup_type: 2,
                min: 0.0,
                delta: 1.0,
                value_bits: 4,
                sequence_p: false,
                multiplicands,
            }),
            codewords: Vec::new(),
        };
        match detect_lookup1(&cb) {
            LookupOptimisation::NotPossible { .. } => {}
            other => panic!("expected NotPossible (no win), got {other:?}"),
        }
    }

    #[test]
    fn pow_u64_overflow_returns_none() {
        assert_eq!(pow_u64(2, 63), Some(1u64 << 63));
        assert_eq!(pow_u64(2, 64), None);
        assert_eq!(pow_u64(0, 0), Some(1));
        assert_eq!(pow_u64(1, 100), Some(1));
    }
}

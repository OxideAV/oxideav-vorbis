//! Floor 1 envelope-fit glue (Vorbis I §7.2.4 step 2 / §10.1 dB-table
//! inverse, encode direction).
//!
//! The crate already carries the floor-1 amplitude-unwrap glue
//! ([`crate::floor1_encode::plan_floor1_y`]): given a target *reconstructed*
//! post list `[floor1_final_Y]` — the **integer** amplitudes the §7.2.4
//! step-2 curve synthesis draws — it produces the packet-domain
//! `[floor1_Y]` vector the floor-1 WRITE primitive serialises. What sat in
//! front of *that* — turning a desired **linear-domain** floor envelope
//! (one magnitude per spectral bin, the kind the forward MDCT produces)
//! into those integer posts — was the remaining floor-1 encode followup
//! the crate README named. This module is that glue: the §7.2.4 step-2 /
//! §10.1 dB-table **inverse**.
//!
//! ## What the decoder does, and the inverse
//!
//! The §7.2.4 step-2 curve synthesis (see
//! [`crate::floor1::Floor1Decoder`]) renders the floor in two passes:
//!
//! 1. It draws straight integer line segments through the
//!    `(x_list[i], final_Y[i] * multiplier)` posts (sorted by ascending
//!    `x`), producing one integer value `v ∈ [0, 256)` per spectral bin.
//! 2. It substitutes each bin's integer value through the §10.1
//!    `floor1_inverse_dB_table` ([`crate::floor1::INVERSE_DB_TABLE`]):
//!    `linear[bin] = INVERSE_DB_TABLE[v]`.
//!
//! So the **post** at `x_list[i]` lands the decoder's pre-substitution
//! integer line at value `final_Y[i] * multiplier`; the bin at exactly
//! that `x` then decodes to `INVERSE_DB_TABLE[final_Y[i] * multiplier]`.
//! Inverting one post is therefore a two-stage map:
//!
//! * **dB-table inverse.** The 256-entry table is *strictly increasing*
//!   (a dB ladder from `1.06e-7` to `1.0`), so a target linear amplitude
//!   maps to the unique table index whose value is **nearest** —
//!   [`invert_inverse_db`] finds it by monotone search.
//! * **multiplier inverse.** The post's integer amplitude is
//!   `final_Y = round(idx / multiplier)`, clamped to `[0, range)` (the
//!   §7.2.4 step-1 amplitude range), so that `final_Y * multiplier`
//!   re-lands as close to `idx` as the multiplier grid allows.
//!
//! [`plan_floor1_envelope`] applies this per post: it samples the target
//! envelope at each post's `x` coordinate, inverts the dB table, divides
//! by the multiplier, and clamps — yielding the `target_final` post vector
//! [`crate::floor1_encode::plan_floor1_y`] consumes. The chain
//! `plan_floor1_envelope → plan_floor1_y → write_floor1_packet → decode`
//! reconstructs a floor curve that, **at every post `x`**, matches the
//! target envelope to within the multiplier-grid + dB-ladder quantisation;
//! between posts the curve is the floor-1 integer line interpolation
//! (intrinsic to floor 1, not a fidelity loss this module introduces).
//!
//! ## Why post-`x` fidelity, not per-bin
//!
//! Floor 1 is a piecewise-linear envelope coder: it can only place posts
//! at the header's `x_list` coordinates and draws straight lines between
//! them. A target envelope that is not piecewise-linear in the dB-index
//! domain between adjacent posts is approximated, never reproduced
//! bit-exactly — that is the floor's design, not a planner deficiency. The
//! exactness guarantee this module offers is: **at each post `x`, the
//! reconstructed linear value is `INVERSE_DB_TABLE[round-nearest grid]`**,
//! the closest value floor 1 can represent there.
//!
//! ## Scope
//!
//! This module plans the integer `target_final` post vector only. Choosing
//! the per-partition master-selector `cval`s, the `[nonzero]` flag, and the
//! Huffman packing remain the existing
//! [`crate::floor1_encode::plan_floor1_y`] /
//! [`crate::encoder::write_floor1_packet`] responsibilities.

use crate::floor1::INVERSE_DB_TABLE;
use crate::floor1_encode::full_x_list;
use crate::setup::Floor1Header;

/// `[range]` lookup keyed by `floor1_multiplier - 1` (§7.2.4 step-1
/// step 1): the exclusive upper bound on every reconstructed post value.
const RANGE_TABLE: [u32; 4] = [256, 128, 86, 64];

/// Errors that can arise while fitting a floor-1 post vector to a target
/// linear-domain envelope (Vorbis I §7.2.4 step 2, encode direction).
#[derive(Debug, Clone, PartialEq)]
pub enum Floor1EnvelopeError {
    /// `header.multiplier` was outside `1..=4`. §7.2.3 step 1 derives
    /// `[range]` from a 2-bit field `+ 1`; the decoder rejects anything
    /// else, so the fitter mirrors that gate. Carries the bad value.
    IllegalMultiplier(u8),
    /// The supplied envelope was shorter than the largest post `x`
    /// coordinate the header places. Every post samples the envelope at
    /// its `x` index, so the envelope must cover `0..=max_x`.
    EnvelopeTooShort {
        /// The largest post `x` coordinate (`2^rangebits`, the upper
        /// endpoint, is the maximum the decoder ever injects).
        max_x: usize,
        /// The supplied envelope length.
        actual: usize,
    },
    /// The envelope carried a non-finite or negative sample. The §10.1
    /// table is a non-negative linear-amplitude ladder; a NaN/-∞/negative
    /// target has no nearest table index. Carries the offending bin.
    NonFiniteEnvelope {
        /// The bin index whose sample was rejected.
        bin: usize,
        /// The offending sample value.
        value: f32,
    },
}

impl core::fmt::Display for Floor1EnvelopeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Floor1EnvelopeError::IllegalMultiplier(m) => write!(
                f,
                "vorbis floor1 envelope: multiplier {m} out of 1..=4 (§7.2.3 step 1)"
            ),
            Floor1EnvelopeError::EnvelopeTooShort { max_x, actual } => write!(
                f,
                "vorbis floor1 envelope: envelope length {actual} < max post x {max_x} (§7.2.4 step 2)"
            ),
            Floor1EnvelopeError::NonFiniteEnvelope { bin, value } => write!(
                f,
                "vorbis floor1 envelope: bin {bin} sample {value} is not a finite non-negative amplitude (§10.1)"
            ),
        }
    }
}

impl std::error::Error for Floor1EnvelopeError {}

/// Invert the §10.1 `floor1_inverse_dB_table`: return the table **index**
/// `0..=255` whose [`INVERSE_DB_TABLE`] value is nearest the target linear
/// amplitude.
///
/// The table is strictly increasing, so the search is a monotone
/// lower-bound followed by a single neighbour comparison:
///
/// * a target at or below `INVERSE_DB_TABLE[0]` (`≈1.06e-7`) maps to `0`;
/// * a target at or above `INVERSE_DB_TABLE[255]` (`1.0`) maps to `255`;
/// * otherwise the partition point `p` (first index with
///   `INVERSE_DB_TABLE[p] >= target`) and its predecessor `p-1` straddle
///   the target; the nearer of the two (linear distance) wins, ties going
///   to the **lower** index (the smaller amplitude — a conservative floor
///   choice).
///
/// `target` must be finite and non-negative; the caller
/// ([`plan_floor1_envelope`]) gates that before calling.
#[must_use]
pub fn invert_inverse_db(target: f32) -> u8 {
    let table = &INVERSE_DB_TABLE;
    // Saturate at the ladder ends.
    if target <= table[0] {
        return 0;
    }
    if target >= table[table.len() - 1] {
        return (table.len() - 1) as u8;
    }
    // First index whose value is >= target (the table is strictly
    // increasing, so `partition_point` gives a clean lower bound).
    let p = table.partition_point(|&v| v < target);
    // `p` is in `1..table.len()` here: target > table[0] rules out 0, and
    // target < table[last] guarantees some entry is >= target before the
    // end. Its predecessor is the largest entry strictly below target.
    let hi = table[p];
    let lo = table[p - 1];
    // Nearest neighbour, ties to the lower index.
    if (hi - target) < (target - lo) {
        p as u8
    } else {
        (p - 1) as u8
    }
}

/// Fit a floor-1 reconstructed-post vector `[floor1_final_Y]` to a desired
/// linear-domain floor `envelope`.
///
/// `envelope[x]` is the target linear amplitude the reconstructed floor
/// should take at spectral bin `x` (the same domain the §7.2.4 step-2
/// dB-table substitution produces and the forward MDCT magnitude lives
/// in). For each of the `[floor1_values]` posts — the two implicit
/// endpoints (`x = 0` and `x = 2^rangebits`) first, then the header's
/// explicit `x_list` coordinates — the fitter:
///
/// 1. samples `envelope` at the post's `x`;
/// 2. inverts the §10.1 dB table ([`invert_inverse_db`]) to the nearest
///    256-ladder index;
/// 3. divides by `multiplier` with round-to-nearest and clamps to
///    `[0, range)`, giving the integer post value `final_Y`.
///
/// The returned `Vec<i32>` is exactly the `target_final` slice
/// [`crate::floor1_encode::plan_floor1_y`] consumes (length
/// `[floor1_values]`, endpoints first). Feeding it through that planner,
/// [`crate::encoder::write_floor1_packet`], and the floor-1 decoder
/// reconstructs a floor whose value **at each post `x`** is the nearest
/// representable approximation of `envelope[x]`.
///
/// # Errors
///
/// Returns a [`Floor1EnvelopeError`] for a multiplier outside `1..=4`, an
/// `envelope` shorter than the largest post `x`, or a non-finite/negative
/// envelope sample. Validation precedes the per-post fit; on error no
/// partial vector is returned.
pub fn plan_floor1_envelope(
    envelope: &[f32],
    header: &Floor1Header,
) -> Result<Vec<i32>, Floor1EnvelopeError> {
    if !(1..=4).contains(&header.multiplier) {
        return Err(Floor1EnvelopeError::IllegalMultiplier(header.multiplier));
    }
    let range = RANGE_TABLE[(header.multiplier - 1) as usize] as i32;
    let multiplier = header.multiplier as i32;

    let x_list = full_x_list(header);
    let max_x = x_list.iter().copied().max().unwrap_or(0) as usize;
    if envelope.len() <= max_x {
        return Err(Floor1EnvelopeError::EnvelopeTooShort {
            max_x,
            actual: envelope.len(),
        });
    }

    // Fail closed on any non-finite / negative sample the fit reads.
    for &x in &x_list {
        let bin = x as usize;
        let v = envelope[bin];
        if !v.is_finite() || v < 0.0 {
            return Err(Floor1EnvelopeError::NonFiniteEnvelope { bin, value: v });
        }
    }

    let mut posts: Vec<i32> = Vec::with_capacity(x_list.len());
    for &x in &x_list {
        let target = envelope[x as usize];
        // §10.1 dB-table inverse: nearest 256-ladder index.
        let idx = invert_inverse_db(target) as i32;
        // §7.2.4 step-2 multiplier inverse: round-to-nearest division so
        // `final_Y * multiplier` re-lands as close to `idx` as the grid
        // allows, then clamp into the legal post range.
        let final_y = ((idx + multiplier / 2) / multiplier).clamp(0, range - 1);
        posts.push(final_y);
    }

    Ok(posts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::Floor1Class;

    /// Build a minimal one-partition floor-1 header with the given explicit
    /// `x_list` and multiplier. `rangebits` is chosen large enough to hold
    /// every supplied coordinate and the implicit upper endpoint. Only the
    /// fields the fitter reads (`multiplier`, `rangebits`, `x_list`) are
    /// load-bearing; the class metadata is a well-formed placeholder.
    fn header(x_list: Vec<u32>, multiplier: u8, rangebits: u8) -> Floor1Header {
        Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: x_list.len() as u8,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![None],
            }],
            multiplier,
            rangebits,
            x_list,
        }
    }

    // ---------- dB-table inverse ----------

    #[test]
    fn db_table_is_strictly_increasing() {
        for w in INVERSE_DB_TABLE.windows(2) {
            assert!(w[0] < w[1], "table must be strictly increasing: {w:?}");
        }
    }

    #[test]
    fn invert_db_saturates_at_ladder_ends() {
        // Below the floor → index 0; above the top → index 255.
        assert_eq!(invert_inverse_db(0.0), 0);
        assert_eq!(invert_inverse_db(-1.0), 0);
        assert_eq!(invert_inverse_db(1.0e-12), 0);
        assert_eq!(invert_inverse_db(1.0), 255);
        assert_eq!(invert_inverse_db(100.0), 255);
    }

    #[test]
    fn invert_db_recovers_each_exact_table_value() {
        // Every table value inverts back to its own index (it is the
        // nearest entry to itself).
        for (i, &v) in INVERSE_DB_TABLE.iter().enumerate() {
            assert_eq!(invert_inverse_db(v), i as u8, "index {i} value {v}");
        }
    }

    #[test]
    fn invert_db_picks_nearest_neighbour() {
        // A target just above table[10] but closer to table[10] than
        // table[11] returns 10; one closer to table[11] returns 11.
        let lo = INVERSE_DB_TABLE[10];
        let hi = INVERSE_DB_TABLE[11];
        let near_lo = lo + (hi - lo) * 0.25;
        let near_hi = lo + (hi - lo) * 0.75;
        assert_eq!(invert_inverse_db(near_lo), 10);
        assert_eq!(invert_inverse_db(near_hi), 11);
    }

    #[test]
    fn invert_db_breaks_ties_to_lower_index() {
        // The exact midpoint between two entries goes to the lower index
        // (the conservative, smaller-amplitude floor choice).
        let lo = INVERSE_DB_TABLE[100];
        let hi = INVERSE_DB_TABLE[101];
        let mid = (lo + hi) * 0.5;
        assert_eq!(invert_inverse_db(mid), 100);
    }

    // ---------- envelope fit ----------

    #[test]
    fn rejects_illegal_multiplier() {
        let h = header(vec![2, 4], 5, 3);
        assert_eq!(
            plan_floor1_envelope(&[0.5; 16], &h),
            Err(Floor1EnvelopeError::IllegalMultiplier(5))
        );
    }

    #[test]
    fn rejects_short_envelope() {
        // rangebits 4 → upper endpoint x = 16, so a length-16 envelope is
        // one short (needs to cover index 16).
        let h = header(vec![4, 8], 1, 4);
        match plan_floor1_envelope(&[0.5; 16], &h) {
            Err(Floor1EnvelopeError::EnvelopeTooShort { max_x, actual }) => {
                assert_eq!(max_x, 16);
                assert_eq!(actual, 16);
            }
            other => panic!("expected EnvelopeTooShort, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_finite_sample() {
        let h = header(vec![4, 8], 1, 5);
        let mut env = vec![0.5f32; 33];
        env[8] = f32::NAN;
        match plan_floor1_envelope(&env, &h) {
            Err(Floor1EnvelopeError::NonFiniteEnvelope { bin, .. }) => assert_eq!(bin, 8),
            other => panic!("expected NonFiniteEnvelope, got {other:?}"),
        }
    }

    #[test]
    fn rejects_negative_sample() {
        let h = header(vec![4, 8], 1, 5);
        let mut env = vec![0.5f32; 33];
        env[4] = -0.1;
        match plan_floor1_envelope(&env, &h) {
            Err(Floor1EnvelopeError::NonFiniteEnvelope { bin, value }) => {
                assert_eq!(bin, 4);
                assert!(value < 0.0);
            }
            other => panic!("expected NonFiniteEnvelope, got {other:?}"),
        }
    }

    #[test]
    fn post_count_matches_floor1_values() {
        // 3 explicit x + 2 implicit endpoints = 5 posts.
        let h = header(vec![4, 8, 12], 1, 5);
        let posts = plan_floor1_envelope(&[0.5; 33], &h).unwrap();
        assert_eq!(posts.len(), 5);
    }

    #[test]
    fn flat_envelope_yields_flat_posts() {
        // A constant envelope inverts to one ladder index everywhere; with
        // multiplier 1 every post is that index. All posts equal.
        let h = header(vec![4, 8, 12], 1, 5);
        let target = INVERSE_DB_TABLE[120];
        let env = vec![target; 33];
        let posts = plan_floor1_envelope(&env, &h).unwrap();
        for &p in &posts {
            assert_eq!(p, 120);
        }
    }

    #[test]
    fn posts_clamp_into_range() {
        // multiplier 2 → range 128. A top-of-ladder envelope inverts to
        // index 255; /2 round-nearest = 128, clamped to 127.
        let h = header(vec![4, 8], 2, 5);
        let env = vec![1.0f32; 33];
        let posts = plan_floor1_envelope(&env, &h).unwrap();
        for &p in &posts {
            assert!((0..128).contains(&p), "post {p} out of [0,128)");
            assert_eq!(p, 127);
        }
    }

    #[test]
    fn endpoints_come_first_in_x_order() {
        // Endpoint x=0 and x=2^rangebits=32 sample the envelope ends; an
        // ascending envelope makes the lower endpoint the smallest post and
        // the upper endpoint the largest.
        let h = header(vec![4, 8, 12], 1, 5);
        // Strictly increasing envelope over [0, 32].
        let env: Vec<f32> = (0..33)
            .map(|i| INVERSE_DB_TABLE[(i * 7).min(255)])
            .collect();
        let posts = plan_floor1_envelope(&env, &h).unwrap();
        // post[0] is x=0 (smallest, env[0] = table[0] → index 0), post[1]
        // is x=32 (largest of all; env[32] = table[32*7=224] → index 224).
        assert_eq!(posts[0], 0);
        assert_eq!(posts[1], 224);
    }
}

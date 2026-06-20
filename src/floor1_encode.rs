//! Floor 1 amplitude-unwrap glue (Vorbis I §7.2.4 step 1, encode
//! direction).
//!
//! The crate already carries the floor-1 **WRITE** path: the
//! [`crate::encoder::write_floor1_packet`] primitive serialises a §7.2.3
//! floor-1 audio-packet body — the `[nonzero]` flag, the two endpoint
//! amplitudes, and the per-partition class/sub-book Huffman codewords —
//! from a [`crate::encoder::Floor1Packet`] whose `floor1_y` member is the
//! explicit `[floor1_Y]` vector (one **packet-domain** value per
//! x-coordinate, including the two endpoints). What sat in front of it —
//! the glue that turns a *target reconstructed* post list (`final_Y`, the
//! integer amplitudes the §7.2.4 step-2 curve synthesis actually draws)
//! into that packet-domain `[floor1_Y]` vector — was the floor-1 analogue
//! of the floor-0 [`crate::floor0_encode`] glue. This module is that glue.
//!
//! ## What the decoder does, and the inverse
//!
//! A floor-1 packet decode (§7.2.4 step 1; see
//! [`crate::floor1::Floor1Decoder`]) **unwraps** the always-non-negative
//! packet values `[floor1_Y]` into the signed line-relative `[floor1_final_Y]`
//! posts it then renders. For every post `i` (the two endpoints `0`/`1`
//! pass straight through) it:
//!
//! * predicts the line value `[predicted]` at `x_list[i]` from the two
//!   already-reconstructed neighbour posts `low_neighbor(i)` /
//!   `high_neighbor(i)` via [`crate::floor1::render_point`];
//! * computes `[highroom] = range - predicted`, `[lowroom] = predicted`,
//!   `[room] = 2 * min(highroom, lowroom)`;
//! * maps the packet value `[val] = floor1_Y[i]` to the reconstructed
//!   `final_Y[i]`:
//!   * `val == 0` → `final_Y[i] = predicted` (the post is *unflagged*: the
//!     curve skips it);
//!   * `0 < val < room` (the **zig-zag** region, where both directions
//!     have room): even `val` → `predicted + val/2` (deviation *up*); odd
//!     `val` → `predicted - (val+1)/2` (deviation *down*);
//!   * `val >= room` (the **linear** region, only one direction has room):
//!     if `highroom > lowroom` → `predicted + (val - lowroom)` (up); else
//!     → `predicted - val + highroom - 1` (down).
//!
//! Because `low_neighbor` / `high_neighbor` only ever look at posts
//! *before* `i`, the prediction for `i` depends solely on already-decoded
//! posts; the decode is a strict left-to-right reconstruction.
//!
//! Encoding inverts each post's map exactly. [`plan_floor1_y`] walks the
//! same left-to-right schedule, and for each post computes the identical
//! `predicted` / `highroom` / `lowroom` / `room` from the posts it has
//! already reconstructed, then chooses the unique packet value `val` whose
//! decode reproduces the target post:
//!
//! 1. `d = target_final[i] - predicted` (the signed line-relative
//!    deviation the decoder will recover).
//! 2. `d == 0` → `val = 0` (the unflagged post).
//! 3. `d > 0`: the zig-zag even candidate is `val = 2 * d`; it is the
//!    decode-correct choice while `val < room`. Otherwise the value lives
//!    in the upper linear extension, valid only when `highroom > lowroom`,
//!    and `val = d + lowroom`.
//! 4. `d < 0`: the zig-zag odd candidate is `val = -2 * d - 1`; correct
//!    while `val < room`. Otherwise the value lives in the lower linear
//!    extension, valid only when `highroom <= lowroom`, and
//!    `val = highroom - 1 - d`.
//!
//! Every chosen `val` must be a legal packet field: `0 <= val < range`
//! (it is Huffman-coded, but the endpoints are written in `ilog(range-1)`
//! bits and no decode path produces `final_Y` outside `[0, range)`). A
//! target post the §7.2.4 map cannot reach in `[0, range)` — e.g. a
//! positive deviation when only the downward direction has linear room —
//! is rejected; it is not a representable floor-1 reconstruction.
//!
//! Because each post's chosen `val` reproduces `target_final[i]` exactly
//! (the §7.2.4 step-1 map is a bijection between `[0, range)` packet
//! values and the reconstructable post values), the planner threads the
//! **reconstructed** post forward as the decoder will, and feeding the
//! resulting `floor1_Y` back through [`crate::encoder::write_floor1_packet`]
//! and the floor-1 decoder reproduces `target_final` bit-for-bit — an
//! exact, *lossless* round-trip (unlike the floor-0 / residue VQ glue,
//! whose quantisation is the lossy step; floor-1 post coding is exact, the
//! lossy choice being which `final_Y` posts to target in the first place).
//!
//! ## Scope
//!
//! This module plans the packet-domain `[floor1_Y]` vector only — the
//! `floor1_y` field of a [`crate::encoder::Floor1Packet`]. Choosing the
//! per-partition master-selector `cval` values (`partition_cvals`) and the
//! `[nonzero]` flag, packing the Y values through the class/sub-book
//! Huffman codewords, and choosing the target `final_Y` posts from a
//! desired floor envelope are separate encode decisions the caller still
//! owns. Threading the result into a full packet is the existing
//! [`crate::encoder::write_floor1_packet`] /
//! [`crate::encoder::write_audio_packet`] path.

use crate::floor1::{high_neighbor, low_neighbor, render_point};
use crate::setup::Floor1Header;

/// `[range]` lookup keyed by `floor1_multiplier - 1` (§7.2.4 step-1
/// step 1 / §7.2.3 step 1): the unwrap modulus and the exclusive upper
/// bound on every packet-domain `[floor1_Y]` value.
const RANGE_TABLE: [u32; 4] = [256, 128, 86, 64];

/// Errors that can arise while planning a floor-1 packet's `[floor1_Y]`
/// vector (Vorbis I §7.2.4 step 1, encode direction).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Floor1EncodeError {
    /// `header.multiplier` was outside `1..=4`. §7.2.3 step 1 derives
    /// `[range]` from a 2-bit field `+ 1`; the decoder rejects anything
    /// else, so the planner mirrors that gate. Carries the bad value.
    IllegalMultiplier(u8),
    /// The supplied target post count did not match `[floor1_values]` —
    /// the full x-coordinate count including the two implicit endpoints
    /// (`header.x_list.len() + 2`). The decoder produces exactly that many
    /// `final_Y` posts, so the target must carry one per post.
    PostLengthMismatch {
        /// `[floor1_values]` = `header.x_list.len() + 2` — the count the
        /// decoder reconstructs.
        expected: usize,
        /// The supplied `target_final.len()`.
        actual: usize,
    },
    /// An endpoint post (index 0 or 1) was outside `[0, range)`. The two
    /// endpoints pass straight into the packet `[floor1_Y]` field (§7.2.4
    /// step-1 steps 4/5) and are written in `ilog(range-1)` bits, so each
    /// must already be a legal packet value.
    EndpointOutOfRange {
        /// `0` or `1` — which endpoint.
        index: usize,
        /// The supplied endpoint post value.
        value: i32,
        /// The exclusive upper bound `[range]`.
        range: u32,
    },
    /// A non-endpoint target post cannot be reached by the §7.2.4 step-1
    /// map with any packet value in `[0, range)`. This happens when the
    /// signed deviation `target_final[i] - predicted` points in the
    /// direction that has no linear room (e.g. a positive deviation past
    /// the zig-zag region when `highroom <= lowroom`), or simply exceeds
    /// the representable extent. The post is not a reconstructable
    /// floor-1 amplitude for this geometry.
    UnreachablePost {
        /// The post index (`2..floor1_values`).
        index: usize,
        /// The supplied target post value.
        target: i32,
        /// The line prediction at this post.
        predicted: i32,
        /// `[range]` for context.
        range: u32,
    },
}

impl core::fmt::Display for Floor1EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Floor1EncodeError::IllegalMultiplier(m) => write!(
                f,
                "vorbis floor1 encode: multiplier {m} out of 1..=4 (§7.2.3 step 1)"
            ),
            Floor1EncodeError::PostLengthMismatch { expected, actual } => write!(
                f,
                "vorbis floor1 encode: target_final.len()={actual} != floor1_values={expected} (§7.2.4 step 1)"
            ),
            Floor1EncodeError::EndpointOutOfRange {
                index,
                value,
                range,
            } => write!(
                f,
                "vorbis floor1 encode: endpoint post {index} = {value} out of [0, {range}) (§7.2.4 step-1 step {})",
                index + 4
            ),
            Floor1EncodeError::UnreachablePost {
                index,
                target,
                predicted,
                range,
            } => write!(
                f,
                "vorbis floor1 encode: post {index} target {target} (predicted {predicted}) not reachable in [0, {range}) (§7.2.4 step 1)"
            ),
        }
    }
}

impl std::error::Error for Floor1EncodeError {}

/// Reconstruct the full `[floor1_X_list]` the §7.2.4 curve decoder uses —
/// the two implicit endpoints (`0` and `2^rangebits`) at positions 0/1,
/// then the header's explicit per-partition x-coordinates — exactly as
/// [`crate::floor1::Floor1Decoder::new`] does. Shared with
/// [`crate::floor1_envelope::plan_floor1_envelope`], which fits posts at
/// these same coordinates.
pub fn full_x_list(header: &Floor1Header) -> Vec<u32> {
    let mut x_list = Vec::with_capacity(header.x_list.len() + 2);
    x_list.push(0);
    x_list.push(1u32 << header.rangebits);
    x_list.extend_from_slice(&header.x_list);
    x_list
}

/// Plan one floor-1 packet's packet-domain `[floor1_Y]` vector (Vorbis I
/// §7.2.4 step 1 in the write direction): turn a target *reconstructed*
/// post list into the always-non-negative packet values the decoder
/// unwraps back into those posts.
///
/// `target_final` is the desired `[floor1_final_Y]` — exactly
/// `[floor1_values]` integer posts (`header.x_list.len() + 2`, the two
/// endpoints first, then one per explicit x-coordinate in header order).
/// These are the integer amplitudes the §7.2.4 step-2 curve synthesis
/// draws (before the `* multiplier` scale and the dB-table lookup), *not*
/// linear-domain spectral magnitudes.
///
/// The returned `Vec<u32>` is the `floor1_y` field of a
/// [`crate::encoder::Floor1Packet`]: one packet value per post, in the
/// same order, length `[floor1_values]`. Feeding it (with `nonzero = true`)
/// through [`crate::encoder::write_floor1_packet`] and the floor-1 decoder
/// reproduces `target_final` exactly.
///
/// # Errors
///
/// Returns a [`Floor1EncodeError`] for a multiplier outside `1..=4`, a
/// `target_final` length that does not match `[floor1_values]`, an
/// endpoint post outside `[0, range)`, or a non-endpoint post the §7.2.4
/// map cannot reach in `[0, range)`. Validation of the multiplier and
/// length precedes any per-post work; on error no partial vector is
/// returned.
pub fn plan_floor1_y(
    target_final: &[i32],
    header: &Floor1Header,
) -> Result<Vec<u32>, Floor1EncodeError> {
    if !(1..=4).contains(&header.multiplier) {
        return Err(Floor1EncodeError::IllegalMultiplier(header.multiplier));
    }
    let range = RANGE_TABLE[(header.multiplier - 1) as usize];
    let range_i = range as i32;

    let x_list = full_x_list(header);
    let values = x_list.len();
    if target_final.len() != values {
        return Err(Floor1EncodeError::PostLengthMismatch {
            expected: values,
            actual: target_final.len(),
        });
    }

    // §7.2.4 step-1 steps 4/5: the two endpoints pass straight through to
    // the packet field. They are written in `ilog(range-1)` bits, so each
    // must already be a legal packet value in `[0, range)`.
    for (index, &v) in target_final.iter().take(2).enumerate() {
        if v < 0 || v >= range_i {
            return Err(Floor1EncodeError::EndpointOutOfRange {
                index,
                value: v,
                range,
            });
        }
    }

    let mut floor1_y: Vec<u32> = Vec::with_capacity(values);
    // The reconstructed posts, threaded forward exactly as the decoder
    // threads `[floor1_final_Y]`. The endpoints are their own targets;
    // every later post reconstructs to its target (the map is exact), so
    // these equal `target_final` post-by-post — but we build them from the
    // chosen `val` to stay in literal lockstep with the decode loop the
    // prediction reads from.
    let mut final_y: Vec<i32> = Vec::with_capacity(values);

    for &v in target_final.iter().take(2) {
        floor1_y.push(v as u32);
        final_y.push(v);
    }

    // §7.2.4 step-1 step 6: iterate over the remaining posts.
    for i in 2..values {
        let low = low_neighbor(&x_list, i);
        let high = high_neighbor(&x_list, i);

        // §7.2.4 step-1 step 9: the line prediction at x_list[i] from the
        // two already-reconstructed neighbour posts.
        let predicted = render_point(
            x_list[low] as i32,
            final_y[low],
            x_list[high] as i32,
            final_y[high],
            x_list[i] as i32,
        );

        // §7.2.4 step-1 steps 11..15.
        let highroom = range_i - predicted;
        let lowroom = predicted;
        let room = if highroom < lowroom {
            highroom * 2
        } else {
            lowroom * 2
        };

        let target = target_final[i];
        let d = target - predicted;

        // Choose the unique packet value whose §7.2.4 step-1 decode
        // reproduces `target`. `d == 0` is the unflagged post (val 0);
        // otherwise the zig-zag candidate is taken while it stays below
        // `room`, falling back to the single linear extension that has
        // room when it does not.
        let val: i32 = if d == 0 {
            0
        } else if d > 0 {
            // Upward deviation. Zig-zag even candidate `2*d` is correct
            // while `< room`; beyond that only the upper linear extension
            // (present iff highroom > lowroom) can reach it.
            let zig = 2 * d;
            if zig < room {
                zig
            } else if highroom > lowroom {
                // §7.2.4 step-1 step 22 inverse: final = val - lowroom +
                // predicted ⇒ val = d + lowroom.
                d + lowroom
            } else {
                return Err(Floor1EncodeError::UnreachablePost {
                    index: i,
                    target,
                    predicted,
                    range,
                });
            }
        } else {
            // Downward deviation (d < 0). Zig-zag odd candidate
            // `-2*d - 1` is correct while `< room`; beyond that only the
            // lower linear extension (present iff highroom <= lowroom) can
            // reach it.
            let zig = -2 * d - 1;
            if zig < room {
                zig
            } else if highroom <= lowroom {
                // §7.2.4 step-1 step 23 inverse: final = predicted - val +
                // highroom - 1 ⇒ val = highroom - 1 - d.
                highroom - 1 - d
            } else {
                return Err(Floor1EncodeError::UnreachablePost {
                    index: i,
                    target,
                    predicted,
                    range,
                });
            }
        };

        // The chosen value must be a legal packet field in `[0, range)`.
        if val < 0 || val >= range_i {
            return Err(Floor1EncodeError::UnreachablePost {
                index: i,
                target,
                predicted,
                range,
            });
        }

        floor1_y.push(val as u32);
        // Thread the reconstruction the decoder will carry. For a value we
        // accepted this equals `target`, but recompute via the decode map
        // so any disagreement surfaces as a test failure rather than a
        // silently-wrong prediction for later posts.
        final_y.push(decode_post(val, predicted, highroom, lowroom, room));
    }

    Ok(floor1_y)
}

/// The §7.2.4 step-1 forward map from a packet value `val` to the
/// reconstructed `final_Y` post, given the already-derived prediction and
/// room values. Mirrors the decode side in
/// [`crate::floor1::Floor1Decoder`]; used by the planner only to thread
/// the reconstructed post forward (the value the later-post prediction
/// reads), and by tests as a self-check.
fn decode_post(val: i32, predicted: i32, highroom: i32, lowroom: i32, room: i32) -> i32 {
    if val == 0 {
        return predicted;
    }
    if val >= room {
        if highroom > lowroom {
            val - lowroom + predicted
        } else {
            predicted - val + highroom - 1
        }
    } else if val % 2 == 1 {
        predicted - (val + 1) / 2
    } else {
        predicted + val / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::{Floor1Class, Floor1Header};

    /// Build a minimal but non-degenerate floor-1 header: `rangebits = 7`
    /// (endpoint at x = 128), two partitions of one class, explicit
    /// x-coordinates spread across the range. The codebook fields are
    /// irrelevant to step-1 unwrap (the planner never touches them), so
    /// the class carries `subclasses = 0` and no books.
    fn header_with_x(multiplier: u8, rangebits: u8, x: Vec<u32>) -> Floor1Header {
        let dims = x.len() as u8;
        Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: dims,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![None],
            }],
            multiplier,
            rangebits,
            x_list: x,
        }
    }

    /// Independent oracle: re-run the §7.2.4 step-1 amplitude synthesis on
    /// a packet `[floor1_Y]` vector to recover `[floor1_final_Y]`. Does not
    /// call the planner — this is the decode side the round-trip checks
    /// against, written from the spec text directly.
    fn decode_final_y(floor1_y: &[u32], header: &Floor1Header) -> Vec<i32> {
        let range = RANGE_TABLE[(header.multiplier - 1) as usize] as i32;
        let x_list = full_x_list(header);
        let values = x_list.len();
        let mut final_y = vec![0i32; values];
        final_y[0] = floor1_y[0] as i32;
        final_y[1] = floor1_y[1] as i32;
        for i in 2..values {
            let low = low_neighbor(&x_list, i);
            let high = high_neighbor(&x_list, i);
            let predicted = render_point(
                x_list[low] as i32,
                final_y[low],
                x_list[high] as i32,
                final_y[high],
                x_list[i] as i32,
            );
            let val = floor1_y[i] as i32;
            let highroom = range - predicted;
            let lowroom = predicted;
            let room = if highroom < lowroom {
                highroom * 2
            } else {
                lowroom * 2
            };
            final_y[i] = if val == 0 {
                predicted
            } else if val >= room {
                if highroom > lowroom {
                    val - lowroom + predicted
                } else {
                    predicted - val + highroom - 1
                }
            } else if val % 2 == 1 {
                predicted - (val + 1) / 2
            } else {
                predicted + val / 2
            };
        }
        final_y
    }

    /// Assert the planner's `floor1_Y` decodes back to `target` exactly,
    /// and that every emitted value is a legal packet field.
    fn assert_roundtrip(target: &[i32], header: &Floor1Header) {
        let range = RANGE_TABLE[(header.multiplier - 1) as usize] as i32;
        let plan = plan_floor1_y(target, header).expect("plan should succeed");
        assert_eq!(plan.len(), target.len(), "one packet value per post");
        for &v in &plan {
            assert!(
                (v as i32) < range,
                "packet value {v} must be < range {range}"
            );
        }
        let recon = decode_final_y(&plan, header);
        assert_eq!(
            &recon, target,
            "decode of planned floor1_Y must equal target"
        );
    }

    // ---------- error paths ----------

    #[test]
    fn illegal_multiplier_rejected() {
        let mut h = header_with_x(1, 7, vec![64, 32, 96]);
        h.multiplier = 5;
        assert_eq!(
            plan_floor1_y(&[0, 0, 0, 0, 0], &h),
            Err(Floor1EncodeError::IllegalMultiplier(5))
        );
        h.multiplier = 0;
        assert_eq!(
            plan_floor1_y(&[0, 0, 0, 0, 0], &h),
            Err(Floor1EncodeError::IllegalMultiplier(0))
        );
    }

    #[test]
    fn post_length_mismatch_rejected() {
        // x_list has 3 explicit + 2 endpoints = 5 posts.
        let h = header_with_x(2, 7, vec![64, 32, 96]);
        let err = plan_floor1_y(&[0, 0, 0, 0], &h).unwrap_err();
        assert_eq!(
            err,
            Floor1EncodeError::PostLengthMismatch {
                expected: 5,
                actual: 4,
            }
        );
    }

    #[test]
    fn endpoint_out_of_range_rejected() {
        // multiplier 4 → range 64; an endpoint of 64 is out of [0, 64).
        let h = header_with_x(4, 7, vec![64]);
        let err = plan_floor1_y(&[64, 0, 0], &h).unwrap_err();
        assert_eq!(
            err,
            Floor1EncodeError::EndpointOutOfRange {
                index: 0,
                value: 64,
                range: 64,
            }
        );
        // Negative endpoint at index 1.
        let err = plan_floor1_y(&[0, -1, 0], &h).unwrap_err();
        assert_eq!(
            err,
            Floor1EncodeError::EndpointOutOfRange {
                index: 1,
                value: -1,
                range: 64,
            }
        );
    }

    #[test]
    fn unreachable_upward_post_rejected() {
        // Endpoints both 0 → the prediction for the interior post is 0,
        // so highroom = range, lowroom = 0, room = 0. Any nonzero target
        // is a deviation; an upward deviation can use the upper linear
        // extension (highroom > lowroom), but a target at/above `range`
        // is unreachable. Pick range 64 (multiplier 4) and target 64.
        let h = header_with_x(4, 7, vec![64]);
        let err = plan_floor1_y(&[0, 0, 64], &h).unwrap_err();
        assert!(matches!(
            err,
            Floor1EncodeError::UnreachablePost { index: 2, .. }
        ));
    }

    #[test]
    fn unreachable_downward_post_rejected() {
        // Endpoints both at range-1 → prediction is range-1, lowroom
        // large, highroom = 1, room = 2. A downward target reachable via
        // the lower linear extension exists, but an upward target (> range-1)
        // is impossible. Force a target above range-1 at the interior post.
        // multiplier 4 → range 64, endpoints 63, target 70 (> 63) upward
        // with highroom(=1) <= lowroom(=63): no upper linear room.
        let h = header_with_x(4, 7, vec![64]);
        let err = plan_floor1_y(&[63, 63, 70], &h).unwrap_err();
        assert!(matches!(
            err,
            Floor1EncodeError::UnreachablePost { index: 2, .. }
        ));
    }

    // ---------- round-trips ----------

    #[test]
    fn flat_zero_roundtrips() {
        let h = header_with_x(2, 7, vec![64, 32, 96]);
        assert_roundtrip(&[0, 0, 0, 0, 0], &h);
    }

    #[test]
    fn endpoints_only_two_posts() {
        // Degenerate floor with no explicit x-coordinates: only the two
        // endpoints. (x_list empty → 2 posts.)
        let h = header_with_x(1, 8, vec![]);
        assert_roundtrip(&[10, 200], &h);
    }

    #[test]
    fn small_upward_deviation_zigzag() {
        // Endpoints 50/50 → prediction 50 for interior posts; small
        // upward deviations land in the zig-zag (even) region.
        let h = header_with_x(1, 7, vec![64, 32, 96]);
        assert_roundtrip(&[50, 50, 53, 47, 60], &h);
    }

    #[test]
    fn mixed_deviations_roundtrip() {
        // A spread of up/down deviations across several posts, exercising
        // both zig-zag parities and the running prediction.
        let h = header_with_x(1, 8, vec![128, 64, 192, 32, 96, 160, 224]);
        assert_roundtrip(&[100, 140, 90, 200, 60, 150, 120, 180, 80], &h);
    }

    #[test]
    fn upper_linear_extension_roundtrip() {
        // Endpoints near 0 (prediction ~0, lowroom small, highroom large)
        // with a large upward target forces the upper linear extension
        // (val >= room, highroom > lowroom).
        let h = header_with_x(2, 7, vec![64]);
        // range 128; endpoints 2/2 → predicted 2, lowroom 2, highroom 126,
        // room 4. Target 120 is a +118 deviation, well past room → upper
        // linear extension.
        assert_roundtrip(&[2, 2, 120], &h);
    }

    #[test]
    fn lower_linear_extension_roundtrip() {
        // Endpoints near range-1 (prediction high, highroom small) with a
        // large downward target forces the lower linear extension
        // (val >= room, highroom <= lowroom).
        let h = header_with_x(2, 7, vec![64]);
        // range 128; endpoints 125/125 → predicted 125, lowroom 125,
        // highroom 3, room 6. Target 5 is a -120 deviation, past room →
        // lower linear extension.
        assert_roundtrip(&[125, 125, 5], &h);
    }

    #[test]
    fn full_range_sweep_roundtrips() {
        // Every reachable target at a single interior post, with the
        // endpoints fixed, must round-trip. Endpoints 0/range-1 give a
        // sloped prediction; sweep the interior target over [0, range).
        for &mult in &[1u8, 2, 3, 4] {
            let range = RANGE_TABLE[(mult - 1) as usize] as i32;
            // Interior x between the two endpoints (x = 64, endpoints at
            // 0 and 128).
            let h = header_with_x(mult, 7, vec![64]);
            for target in 0..range {
                // Endpoints both 0 → prediction 0; every value in
                // [0, range) is reachable (upper linear extension covers
                // the top, zig-zag the bottom).
                let posts = [0, 0, target];
                let plan = plan_floor1_y(&posts, &h)
                    .unwrap_or_else(|e| panic!("mult {mult} target {target}: {e}"));
                let recon = decode_final_y(&plan, &h);
                assert_eq!(recon[2], target, "mult {mult} target {target} round-trip");
                assert!(
                    (plan[2] as i32) < range,
                    "mult {mult} target {target} value < range"
                );
            }
        }
    }

    #[test]
    fn decode_post_matches_inline_oracle() {
        // The private `decode_post` helper must agree with the spec map
        // for representative (val, predicted, room) combinations.
        let cases = [
            (0, 50, 100),   // unflagged
            (2, 50, 100),   // zig-zag even
            (3, 50, 100),   // zig-zag odd
            (100, 50, 100), // linear boundary (val == room)
        ];
        for (val, predicted, _) in cases {
            let range = 256i32;
            let highroom = range - predicted;
            let lowroom = predicted;
            let room = if highroom < lowroom {
                highroom * 2
            } else {
                lowroom * 2
            };
            let got = decode_post(val, predicted, highroom, lowroom, room);
            let want = if val == 0 {
                predicted
            } else if val >= room {
                if highroom > lowroom {
                    val - lowroom + predicted
                } else {
                    predicted - val + highroom - 1
                }
            } else if val % 2 == 1 {
                predicted - (val + 1) / 2
            } else {
                predicted + val / 2
            };
            assert_eq!(got, want, "decode_post({val}, {predicted})");
        }
    }
}

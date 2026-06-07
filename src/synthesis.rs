//! Vorbis I audio-packet synthesis primitives: the Vorbis window
//! (§1.3.2 / §4.3.1 "packet type, mode and window decode") and inverse
//! channel coupling (§4.3.5 "inverse coupling").
//!
//! These two operations sit in the §4.3 audio-packet decode pipeline
//! *after* floor + residue decode (handled by [`crate::floor0`],
//! [`crate::floor1`] and [`crate::residue`]) and *before* the inverse
//! MDCT. They are pure, stateless transforms with no bitstream reads, so
//! they are factored into this standalone module ahead of the full
//! packet-level driver that will tie them together in a later round.
//!
//! # The Vorbis window (§1.3.2 / §4.3.1)
//!
//! Every Vorbis window is built from the slope function
//!
//! ```text
//! y = sin(π/2 · sin²((x + 0.5)/n · π))
//! ```
//!
//! where `n` is the window size and `x` ranges `0 ..= n-1` (§1.3.2,
//! "Vorbis windows all use the slope function"). A window's overall shape
//! depends on whether the current frame is a long block and whether its
//! neighbours are long or short blocks, because the 50%-overlapping MDCT
//! requires the rising edge of one window to mirror the falling edge of
//! its neighbour ("dissimilar lapping requirements can affect overall
//! shape"). [`vorbis_window`] implements the eight-step generation
//! procedure of §4.3.1 verbatim:
//!
//! 1. `window_center = n / 2`.
//! 2. If this is a long block (`blockflag` set) whose previous neighbour
//!    is short (`previous_window_flag` clear), the left edge is the
//!    `blocksize_0`-wide hybrid ramp; otherwise it is the full-half-block
//!    ramp.
//! 3. The symmetric rule for the right edge (`next_window_flag`).
//! 4. The lead-in `0 ..= left_window_start-1` is zero.
//! 5. The rising edge `left_window_start ..= left_window_end-1` is the
//!    slope function with `[i] - left_window_start` as `x` and `left_n`
//!    as `n`.
//! 6. The flat plateau `left_window_end ..= right_window_start-1` is one.
//! 7. The falling edge `right_window_start ..= right_window_end-1` is the
//!    slope function with a `+π/2` phase shift (the spec writes the inner
//!    argument as `… · π/2 + π/2`).
//! 8. The tail `right_window_start ..= n-1` is zero.
//!
//! For a short block (`blockflag` clear) the window is always the plain
//! symmetric shape; the previous/next flags are not present in the packet
//! and are ignored.
//!
//! # Inverse channel coupling (§4.3.5)
//!
//! Vorbis stereo (and generic multichannel) coupling stores, for each
//! coupling step, one residue vector as *magnitude* and one as *angle* in
//! a square-polar representation. [`inverse_couple`] converts a
//! magnitude/angle pair back to Cartesian (left/right) in place, applying
//! the four-quadrant rule of §4.3.5 step 3 element by element. The
//! §4.3.5 loop runs the coupling steps **in descending order**;
//! [`inverse_couple_all`] drives that loop over a slice of per-channel
//! residue vectors and a mapping's coupling step list.

use crate::setup::MappingCouplingStep;

/// `π/2`, the constant lead factor of the Vorbis slope function. Computed
/// from [`core::f64::consts::PI`] so the value carries full `f64`
/// precision before the final `sin`.
const HALF_PI: f64 = core::f64::consts::PI / 2.0;

/// Errors that can arise while constructing a Vorbis window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowError {
    /// `n` was not a positive power of two. Vorbis blocksizes are one of
    /// `{64, 128, … , 8192}` (§4.2.2); window generation indexes
    /// `n/2`, `n/4`, `n*3/4` and therefore requires `n` divisible by 4
    /// with the `blocksize_0`-relative hybrid ranges staying in bounds.
    NotPowerOfTwo {
        /// The offending blocksize.
        n: usize,
    },
    /// A long-block hybrid ramp needs `blocksize_0 <= n` so the
    /// `n/4 ± blocksize_0/4` window edges stay within `0 ..= n`. §4.2.2
    /// guarantees `blocksize_0 <= blocksize_1`, so a violation means the
    /// caller passed mismatched blocksizes.
    ShortBlockTooLarge {
        /// The current (long) blocksize `n`.
        n: usize,
        /// The short blocksize `blocksize_0`.
        blocksize_0: usize,
    },
}

impl core::fmt::Display for WindowError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WindowError::NotPowerOfTwo { n } => {
                write!(
                    f,
                    "vorbis window: blocksize {n} is not a positive power of two"
                )
            }
            WindowError::ShortBlockTooLarge { n, blocksize_0 } => write!(
                f,
                "vorbis window: short blocksize {blocksize_0} exceeds long blocksize {n}"
            ),
        }
    }
}

impl std::error::Error for WindowError {}

/// The Vorbis slope function `y = sin(π/2 · sin²((x + 0.5)/n · π))`
/// (§1.3.2). `x` is the integer sample offset within a ramp of length
/// `n`. Computed in `f64` and returned as `f32` to match the spectral
/// pipeline's working precision.
///
/// The intermediate `sin((x + 0.5)/n · π)` is squared before the outer
/// `sin(π/2 · …)`, exactly as written; this is *not* the same as the
/// per-edge §4.3.1 step-5/step-7 forms, which scale the inner argument by
/// `π/2` (a half-period ramp) rather than `π`. This bare form is exposed
/// for callers and tests that want the canonical slope; the window
/// builder uses the §4.3.1 per-edge arguments directly.
#[must_use]
pub fn slope(x: f64, n: f64) -> f32 {
    let inner = ((x + 0.5) / n * core::f64::consts::PI).sin();
    (HALF_PI * inner * inner).sin() as f32
}

/// Build the length-`n` Vorbis window for one audio frame per the §4.3.1
/// generation procedure.
///
/// * `n` — the decode blocksize of this frame (`blocksize_0` if the
///   mode's `blockflag` is clear, else `blocksize_1`; §4.3.1 step 3).
/// * `blocksize_0` — the stream's short blocksize, used to size the
///   hybrid ramps when a long block laps a short neighbour. Ignored for
///   short blocks but still validated.
/// * `blockflag` — the mode's `[vorbis_mode_blockflag]`: `true` for a
///   long block, `false` for a short block.
/// * `previous_window_flag` / `next_window_flag` — the two flag bits read
///   from the packet for a long block (§4.3.1 step 4a.i/ii). They are
///   *only* meaningful when `blockflag` is set; for a short block they
///   are ignored (the window is always the plain symmetric short shape,
///   §4.3.1 step 4b).
///
/// Returns the window as a `Vec<f32>` of length `n`: a zero lead-in, a
/// rising edge, a plateau of ones, a falling edge, and a zero tail.
///
/// # Errors
///
/// [`WindowError::NotPowerOfTwo`] if `n` is not a positive power of two,
/// or [`WindowError::ShortBlockTooLarge`] if a hybrid ramp would need
/// `blocksize_0 > n`.
pub fn vorbis_window(
    n: usize,
    blocksize_0: usize,
    blockflag: bool,
    previous_window_flag: bool,
    next_window_flag: bool,
) -> Result<Vec<f32>, WindowError> {
    if n == 0 || !n.is_power_of_two() {
        return Err(WindowError::NotPowerOfTwo { n });
    }
    // The hybrid-ramp arithmetic uses `blocksize_0/4`; for it to land on
    // valid window indices, `blocksize_0` must itself be a power of two
    // not exceeding `n`. We only consult it on a long block, but validate
    // unconditionally so a malformed pairing is caught early.
    if blockflag && (blocksize_0 == 0 || !blocksize_0.is_power_of_two() || blocksize_0 > n) {
        return Err(WindowError::ShortBlockTooLarge { n, blocksize_0 });
    }

    // §4.3.1 step 1.
    let window_center = n / 2;

    // §4.3.1 step 2 — left edge.
    let (left_window_start, left_window_end, left_n);
    if blockflag && !previous_window_flag {
        left_window_start = n / 4 - blocksize_0 / 4;
        left_window_end = n / 4 + blocksize_0 / 4;
        left_n = blocksize_0 / 2;
    } else {
        left_window_start = 0;
        left_window_end = window_center;
        left_n = n / 2;
    }

    // §4.3.1 step 3 — right edge.
    let (right_window_start, right_window_end, right_n);
    if blockflag && !next_window_flag {
        right_window_start = n * 3 / 4 - blocksize_0 / 4;
        right_window_end = n * 3 / 4 + blocksize_0 / 4;
        right_n = blocksize_0 / 2;
    } else {
        right_window_start = window_center;
        right_window_end = n;
        right_n = n / 2;
    }

    let mut window = vec![0.0f32; n];

    // §4.3.1 step 4 — lead-in `0 ..= left_window_start-1` is zero (already
    // zeroed by the `vec!` initialiser).

    // §4.3.1 step 5 — rising edge. The inner argument scales by `π/2`
    // (a quarter-period ramp), unlike the bare [`slope`] form.
    let left_n_f = left_n as f64;
    for (offset, i) in (left_window_start..left_window_end).enumerate() {
        let inner = ((offset as f64 + 0.5) / left_n_f * HALF_PI).sin();
        window[i] = (HALF_PI * inner * inner).sin() as f32;
    }

    // §4.3.1 step 6 — plateau `left_window_end ..= right_window_start-1`
    // is one.
    for w in &mut window[left_window_end..right_window_start] {
        *w = 1.0;
    }

    // §4.3.1 step 7 — falling edge. Same form with a `+π/2` phase shift.
    let right_n_f = right_n as f64;
    for (offset, i) in (right_window_start..right_window_end).enumerate() {
        let inner = ((offset as f64 + 0.5) / right_n_f * HALF_PI + HALF_PI).sin();
        window[i] = (HALF_PI * inner * inner).sin() as f32;
    }

    // §4.3.1 step 8 — tail `right_window_end ..= n-1` is zero (already
    // zeroed).

    Ok(window)
}

/// Errors that can arise while applying the §4.3.6 / §4.3.7 window
/// pre-multiplication primitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowPremultiplyError {
    /// The IMDCT-output frame and the §4.3.1-built window disagreed
    /// on their length. The Vorbis §4.3.7 pipeline mandates one
    /// window sample per time-domain sample (both sourced from the
    /// same `n` per §4.3.1 step 3), so a length mismatch indicates a
    /// caller bug — e.g. building the window from one mode's
    /// `(blockflag, n)` and the frame from another's. Fail-closed:
    /// the slice is left untouched.
    LengthMismatch {
        /// Length of the IMDCT-output `time_frame`.
        frame_len: usize,
        /// Length of the §4.3.1 window slice.
        window_len: usize,
    },
}

impl core::fmt::Display for WindowPremultiplyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WindowPremultiplyError::LengthMismatch {
                frame_len,
                window_len,
            } => write!(
                f,
                "vorbis §4.3.6 window pre-multiplication: frame length \
                 {frame_len} disagrees with window length {window_len}"
            ),
        }
    }
}

impl std::error::Error for WindowPremultiplyError {}

/// Apply the §4.3.6 / §4.3.7 window pre-multiplication: every IMDCT
/// time-domain sample is multiplied in place by its matching §4.3.1
/// window sample.
///
/// The Vorbis I spec describes this step under §4.3.7 "inverse MDCT",
/// closing paragraph: the IMDCT output is windowed according to the
/// §4.3.1 window. The window is built once per packet via
/// [`vorbis_window`]; this primitive consumes a built window slice and
/// a length-`n` IMDCT frame and rescales the frame, yielding the
/// windowed time-domain samples that the §4.3.8 overlap-add primitive
/// ([`crate::overlap::OverlapAdd::push_frame`]) consumes.
///
/// The window's lead-in and tail bins are zero by §4.3.1 construction
/// (step 4 / step 8), so a side effect of this multiplication is that
/// the IMDCT samples falling outside the active overlap region are
/// zeroed — the §4.3 decode pipeline carries no separate zeroing step.
///
/// The transform is *in place*: the same caller-owned `time_frame`
/// slice carries the IMDCT output on entry and the windowed samples
/// on return. This matches the [`inverse_couple`] §4.3.5 pattern and
/// avoids one length-`n` allocation per channel per packet on the hot
/// path.
///
/// # Errors
///
/// [`WindowPremultiplyError::LengthMismatch`] when `time_frame.len()`
/// disagrees with `window.len()`. The slice is left unmodified on
/// error: the length check runs before any multiplication.
///
/// # Spec sources
///
/// * `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.3.7 (the IMDCT step's
///   closing window-application clause).
/// * `docs/audio/vorbis/Vorbis_I_spec.pdf` §1.3.2 (the slope-function
///   definition the §4.3.1 window builds from).
/// * `docs/audio/vorbis/imdct-cross-reference.md` §"Window-function
///   equivalence" (paraphrases the §1.3.2 + §4.3.6 window as the
///   product the spec applies after the IMDCT).
pub fn window_premultiply(
    time_frame: &mut [f32],
    window: &[f32],
) -> Result<(), WindowPremultiplyError> {
    if time_frame.len() != window.len() {
        return Err(WindowPremultiplyError::LengthMismatch {
            frame_len: time_frame.len(),
            window_len: window.len(),
        });
    }
    for (sample, &w) in time_frame.iter_mut().zip(window.iter()) {
        *sample *= w;
    }
    Ok(())
}

/// Apply the §4.3.5 inverse-coupling rule to a single
/// `(magnitude, angle)` scalar pair, returning `(new_magnitude,
/// new_angle)` in Cartesian (left/right) form.
///
/// The four cases of §4.3.5 step 3 are reproduced exactly:
///
/// ```text
/// if M > 0:
///   if A > 0: new_M = M;       new_A = M - A
///   else:     new_A = M;       new_M = M + A
/// else:
///   if A > 0: new_M = M;       new_A = M + A
///   else:     new_A = M;       new_M = M - A
/// ```
#[must_use]
pub fn couple_scalar(m: f32, a: f32) -> (f32, f32) {
    if m > 0.0 {
        if a > 0.0 {
            (m, m - a)
        } else {
            (m + a, m)
        }
    } else if a > 0.0 {
        (m, m + a)
    } else {
        (m - a, m)
    }
}

/// Inverse-couple one magnitude/angle vector pair in place (§4.3.5 steps
/// 1..3). Both vectors must already hold the decoded residue for their
/// respective channels and must be the same length (the per-vector decode
/// length is `n/2`, §4.3.4 step 5). Each scalar position is decoupled
/// independently via [`couple_scalar`].
///
/// # Panics
///
/// Panics if `magnitude` and `angle` have different lengths; the §4.3.5
/// loop pairs residue vectors that are always `n/2` long, so a mismatch
/// indicates a caller bug rather than stream data.
pub fn inverse_couple(magnitude: &mut [f32], angle: &mut [f32]) {
    assert_eq!(
        magnitude.len(),
        angle.len(),
        "inverse_couple: magnitude/angle length mismatch"
    );
    for (m, a) in magnitude.iter_mut().zip(angle.iter_mut()) {
        let (new_m, new_a) = couple_scalar(*m, *a);
        *m = new_m;
        *a = new_a;
    }
}

/// Errors that can arise while driving the full §4.3.5 inverse-coupling
/// loop over a residue-vector bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CouplingError {
    /// A coupling step's magnitude- or angle-channel index pointed past
    /// the supplied residue-vector slice. The setup parser already
    /// range-checks these against `audio_channels` (§4.2.4 "Mappings"
    /// step 2c.ii); a violation here means the residue slice does not
    /// have one vector per channel.
    ChannelOutOfRange {
        /// The coupling step index (counting from the descending loop's
        /// nominal order).
        step: usize,
        /// The offending channel index.
        channel: usize,
        /// The number of residue vectors available.
        channels: usize,
    },
    /// A coupling step named the same channel for magnitude and angle.
    /// §4.2.4 "Mappings" forbids this, so it should never survive setup
    /// parse; checked defensively because in-place decoupling of a vector
    /// with itself would be incorrect.
    SameChannel {
        /// The coupling step index.
        step: usize,
        /// The channel index used for both magnitude and angle.
        channel: usize,
    },
}

impl core::fmt::Display for CouplingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CouplingError::ChannelOutOfRange {
                step,
                channel,
                channels,
            } => write!(
                f,
                "inverse coupling: step {step} channel {channel} out of range \
                 (have {channels} residue vectors)"
            ),
            CouplingError::SameChannel { step, channel } => write!(
                f,
                "inverse coupling: step {step} uses channel {channel} for both \
                 magnitude and angle"
            ),
        }
    }
}

impl std::error::Error for CouplingError {}

/// Run the full §4.3.5 inverse-coupling pass over a slice of per-channel
/// residue vectors, applying every coupling step **in descending order**
/// (`coupling_steps-1 … 0`) as the spec mandates.
///
/// `residues[channel]` is the decoded residue vector for each channel;
/// each coupling step decouples the `(magnitude_channel, angle_channel)`
/// pair named in the mapping configuration. Decoupling is done in place,
/// so after the call each vector holds the channel's fine spectral detail
/// (§4.3.5 closing note).
///
/// # Errors
///
/// [`CouplingError::ChannelOutOfRange`] if a step names a channel index
/// `>= residues.len()`, or [`CouplingError::SameChannel`] if a step names
/// the same channel for both magnitude and angle.
pub fn inverse_couple_all(
    residues: &mut [Vec<f32>],
    coupling: &[MappingCouplingStep],
) -> Result<(), CouplingError> {
    let channels = residues.len();
    // §4.3.5: "for each [i] from [vorbis_mapping_coupling_steps]-1
    // descending to 0".
    for (step, cs) in coupling.iter().enumerate().rev() {
        let mag = cs.magnitude_channel as usize;
        let ang = cs.angle_channel as usize;
        if mag >= channels {
            return Err(CouplingError::ChannelOutOfRange {
                step,
                channel: mag,
                channels,
            });
        }
        if ang >= channels {
            return Err(CouplingError::ChannelOutOfRange {
                step,
                channel: ang,
                channels,
            });
        }
        if mag == ang {
            return Err(CouplingError::SameChannel { step, channel: mag });
        }
        // Split the slice so we can borrow the two vectors mutably at
        // once. `mag != ang` is guaranteed above.
        let (lo, hi) = (mag.min(ang), mag.max(ang));
        let (head, tail) = residues.split_at_mut(hi);
        let (a, b) = (&mut head[lo], &mut tail[0]);
        if mag < ang {
            inverse_couple(a, b);
        } else {
            inverse_couple(b, a);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::MappingCouplingStep;

    // ---- slope function (§1.3.2) ----

    #[test]
    fn slope_endpoints_and_midpoint() {
        // At x just below 0 the inner sin → 0; at x near n the inner sin
        // → 0 again (sin(π) = 0), so the bare slope is small at both ends
        // and peaks in the middle of the [0, n] range.
        let n = 256.0;
        let first = slope(0.0, n);
        let mid = slope(127.5, n); // (x+0.5)/n = 0.5 → inner sin(π/2)=1 → sin(π/2)=1
        let last = slope(255.0, n);
        assert!(first < 0.05, "slope near 0 should be tiny, got {first}");
        assert!((mid - 1.0).abs() < 1e-6, "slope mid should be 1, got {mid}");
        assert!(last < 0.05, "slope near n should be tiny, got {last}");
    }

    #[test]
    fn slope_is_symmetric_about_center() {
        // sin² is symmetric about (x+0.5)/n = 0.5, so slope(x) ==
        // slope(n-1-x).
        let n = 128.0;
        for x in 0..64 {
            let a = slope(x as f64, n);
            let b = slope((127 - x) as f64, n);
            assert!((a - b).abs() < 1e-6, "x={x}: {a} != {b}");
        }
    }

    // ---- window generation (§4.3.1) ----

    #[test]
    fn window_rejects_non_power_of_two() {
        assert_eq!(
            vorbis_window(100, 64, false, true, true),
            Err(WindowError::NotPowerOfTwo { n: 100 })
        );
        assert_eq!(
            vorbis_window(0, 64, false, true, true),
            Err(WindowError::NotPowerOfTwo { n: 0 })
        );
    }

    #[test]
    fn window_rejects_short_block_larger_than_long() {
        // Long block n=256 with a claimed short block of 512.
        assert_eq!(
            vorbis_window(256, 512, true, false, false),
            Err(WindowError::ShortBlockTooLarge {
                n: 256,
                blocksize_0: 512
            })
        );
    }

    #[test]
    fn short_window_is_full_symmetric_shape() {
        // blockflag clear → left edge 0..n/2, plateau empty, right edge
        // n/2..n. The result is the classic symmetric MDCT window with no
        // zero lead-in or tail (left_window_start == 0,
        // right_window_end == n).
        let n = 64;
        let w = vorbis_window(n, 64, false, true, true).unwrap();
        assert_eq!(w.len(), n);
        // First and last are the slope-function endpoints, both small but
        // strictly positive (the (x+0.5) offset keeps them off zero).
        assert!(w[0] > 0.0 && w[0] < 0.1);
        assert!(w[n - 1] > 0.0 && w[n - 1] < 0.1);
        // Squared-sum reconstruction property: for a symmetric window the
        // overlap of the rising edge with its mirror sums to one at every
        // bin (w[i]² + w[i + n/2]² == 1 for the n/2-overlap region).
        for i in 0..n / 2 {
            let s = w[i] * w[i] + w[i + n / 2] * w[i + n / 2];
            assert!((s - 1.0).abs() < 1e-5, "i={i}: power sum {s} != 1");
        }
    }

    #[test]
    fn long_window_with_long_neighbors_is_symmetric() {
        // blockflag set but both neighbour flags set → full-half-block
        // edges, identical shape to a short window of the same n.
        let n = 256;
        let long = vorbis_window(n, 64, true, true, true).unwrap();
        let plain = vorbis_window(n, 64, false, true, true).unwrap();
        assert_eq!(long.len(), n);
        for i in 0..n {
            assert!((long[i] - plain[i]).abs() < 1e-6, "i={i}");
        }
    }

    #[test]
    fn long_window_short_previous_has_zero_leadin_and_hybrid_ramp() {
        // n=256, blocksize_0=64. previous_window_flag clear → left edge is
        // the 64-wide hybrid ramp at n/4 ± blocksize_0/4 = 64 ± 16.
        let n = 256;
        let bs0 = 64;
        let w = vorbis_window(n, bs0, true, false, true).unwrap();
        let left_window_start = n / 4 - bs0 / 4; // 48
        let left_window_end = n / 4 + bs0 / 4; // 80
                                               // Lead-in 0..48 is exactly zero.
        for (i, &v) in w.iter().enumerate().take(left_window_start) {
            assert_eq!(v, 0.0, "lead-in bin {i} not zero");
        }
        // Rising edge 48..80 is strictly increasing from ~0 to ~1.
        for i in left_window_start..left_window_end - 1 {
            assert!(w[i] <= w[i + 1] + 1e-6, "rising edge not monotone at {i}");
        }
        // Plateau 80..center(128) is exactly one (right edge starts at
        // window_center=128 because next_window_flag is set).
        for (i, &v) in w.iter().enumerate().take(n / 2).skip(left_window_end) {
            assert!((v - 1.0).abs() < 1e-6, "plateau bin {i} not one");
        }
    }

    #[test]
    fn long_window_short_next_has_zero_tail_and_hybrid_ramp() {
        // Symmetric to the previous test: next_window_flag clear → right
        // edge is the 64-wide hybrid ramp at 3n/4 ± blocksize_0/4.
        let n = 256;
        let bs0 = 64;
        let w = vorbis_window(n, bs0, true, true, false).unwrap();
        let right_window_start = n * 3 / 4 - bs0 / 4; // 176
        let right_window_end = n * 3 / 4 + bs0 / 4; // 208
                                                    // Falling edge 176..208 is strictly decreasing from ~1 to ~0.
        for i in right_window_start..right_window_end - 1 {
            assert!(w[i] + 1e-6 >= w[i + 1], "falling edge not monotone at {i}");
        }
        // Tail 208..256 is exactly zero.
        for (i, &v) in w.iter().enumerate().skip(right_window_end) {
            assert_eq!(v, 0.0, "tail bin {i} not zero");
        }
    }

    #[test]
    fn adjacent_long_short_windows_lap_to_unity_power() {
        // Trans-document property of §1.3.2: when a long block's right
        // edge is a hybrid ramp sized for a short neighbour, that ramp
        // mirrors the short block's left edge so their squared overlap
        // sums to one. Check that the long block's hybrid falling edge
        // squared, summed with the short block's rising edge squared
        // (reversed), equals one across the 32-sample overlap.
        let bs1 = 256;
        let bs0 = 64;
        // Long block whose NEXT neighbour is short.
        let long = vorbis_window(bs1, bs0, true, true, false).unwrap();
        // Short block whose PREVIOUS neighbour is long (short windows
        // ignore the flags; shape is fixed).
        let short = vorbis_window(bs0, bs0, false, true, true).unwrap();
        let rstart = bs1 * 3 / 4 - bs0 / 4; // 176
                                            // The long falling edge spans rstart..rstart+bs0 (32 samples
                                            // each side -> bs0/2*2 = bs0); the short rising edge spans
                                            // 0..bs0/2. They overlap sample-for-sample after the 3/4-vs-1/4
                                            // alignment.
        for j in 0..bs0 / 2 {
            let l = long[rstart + j];
            let s = short[j];
            let sum = l * l + s * s;
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "overlap bin {j}: power {sum} != 1"
            );
        }
    }

    // ---- window pre-multiplication (§4.3.6 / §4.3.7 closing step) ----

    #[test]
    fn window_premultiply_pointwise_product() {
        // Canonical use: a length-`n` time frame is multiplied element
        // by element by the length-`n` window. Use a non-trivial window
        // and frame so we can spot any sign / stride bug.
        let mut frame = vec![1.0_f32, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0];
        let window = vec![0.25_f32, 0.5, 1.0, 0.75, 0.0, 0.125, 1.0, -0.5];
        let expected: Vec<f32> = frame
            .iter()
            .zip(window.iter())
            .map(|(a, b)| a * b)
            .collect();
        window_premultiply(&mut frame, &window).unwrap();
        for (i, (got, want)) in frame.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-7, "bin {i}: {got} != {want}");
        }
    }

    #[test]
    fn window_premultiply_with_built_vorbis_window_zeroes_leadin_and_tail() {
        // Couples the new primitive with the §4.3.1 [`vorbis_window`]
        // builder: feed a long-block hybrid window whose lead-in /
        // tail bins are exactly zero, and confirm the multiplication
        // pins those frame bins to zero regardless of the IMDCT-side
        // contents.
        let n = 256;
        let bs0 = 64;
        // previous_window_flag clear → 48-bin lead-in is zero; tail
        // (right_window_end..n) starts at 3n/4 + bs0/4 = 208 → 48-bin
        // tail is zero.
        let window = vorbis_window(n, bs0, true, false, false).unwrap();
        // Non-zero everywhere so any zero in the result must come from
        // the window.
        let mut frame = vec![123.0_f32; n];
        window_premultiply(&mut frame, &window).unwrap();
        // Lead-in `0..48` is zero.
        for (i, &v) in frame.iter().enumerate().take(48) {
            assert_eq!(v, 0.0, "lead-in bin {i} not zero");
        }
        // Tail `208..256` is zero.
        for (i, v) in frame.iter().enumerate().skip(208) {
            assert_eq!(*v, 0.0, "tail bin {i} not zero");
        }
        // Plateau `80..176` is unchanged (window == 1.0 there).
        for (i, v) in frame.iter().enumerate().take(176).skip(80) {
            assert!((v - 123.0).abs() < 1e-4, "plateau bin {i}: {v} != 123");
        }
    }

    #[test]
    fn window_premultiply_zero_window_zeroes_frame() {
        // §4.3.2 "zeroed" packet path: when the spec mandates a zero
        // output, the windowing of the IMDCT-of-zero is still zero by
        // linearity. Pin the simpler fact: an all-zero window forces
        // the frame to zero regardless of input.
        let mut frame = vec![42.0_f32; 16];
        let window = vec![0.0_f32; 16];
        window_premultiply(&mut frame, &window).unwrap();
        for v in &frame {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn window_premultiply_unity_window_is_identity() {
        // The §4.3.1 plateau is exactly one; pinned here as a property
        // of the primitive at slice-level — an all-ones window leaves
        // the frame untouched.
        let mut frame = vec![1.0_f32, -2.0, 3.5, -4.25];
        let before = frame.clone();
        let window = vec![1.0_f32; frame.len()];
        window_premultiply(&mut frame, &window).unwrap();
        assert_eq!(frame, before);
    }

    #[test]
    fn window_premultiply_empty_slices_are_noop() {
        // Degenerate but valid: zero-length slices have matching length
        // and the multiplication loop runs zero times.
        let mut frame: Vec<f32> = vec![];
        let window: Vec<f32> = vec![];
        window_premultiply(&mut frame, &window).unwrap();
        assert!(frame.is_empty());
    }

    #[test]
    fn window_premultiply_rejects_length_mismatch_frame_longer() {
        let mut frame = vec![1.0_f32, 2.0, 3.0, 4.0];
        let before = frame.clone();
        let window = vec![1.0_f32, 1.0, 1.0];
        assert_eq!(
            window_premultiply(&mut frame, &window),
            Err(WindowPremultiplyError::LengthMismatch {
                frame_len: 4,
                window_len: 3,
            })
        );
        // Fail-closed: no samples mutated.
        assert_eq!(frame, before);
    }

    #[test]
    fn window_premultiply_rejects_length_mismatch_window_longer() {
        let mut frame = vec![1.0_f32, 2.0];
        let before = frame.clone();
        let window = vec![1.0_f32, 1.0, 1.0, 1.0];
        assert_eq!(
            window_premultiply(&mut frame, &window),
            Err(WindowPremultiplyError::LengthMismatch {
                frame_len: 2,
                window_len: 4,
            })
        );
        assert_eq!(frame, before);
    }

    #[test]
    fn window_premultiply_preserves_sign() {
        // Negative IMDCT samples times positive window samples stay
        // negative; pin this so a stray `.abs()` or `+=` would trip.
        let mut frame = vec![-1.0_f32, -2.0, -3.0, -4.0];
        let window = vec![0.5_f32; 4];
        window_premultiply(&mut frame, &window).unwrap();
        assert_eq!(frame, vec![-0.5, -1.0, -1.5, -2.0]);
    }

    #[test]
    fn window_premultiply_error_display_is_descriptive() {
        // The error message is consumer-facing (surfaces via
        // `AudioPacketError`); pin a substring of its render so a
        // copy-edit regression is caught.
        let err = WindowPremultiplyError::LengthMismatch {
            frame_len: 1024,
            window_len: 512,
        };
        let s = format!("{err}");
        assert!(s.contains("1024"));
        assert!(s.contains("512"));
        assert!(s.contains("window"));
    }

    // ---- inverse coupling scalar rule (§4.3.5 step 3) ----

    #[test]
    fn couple_scalar_all_four_quadrants() {
        // M>0, A>0: new_M=M, new_A=M-A
        assert_eq!(couple_scalar(5.0, 2.0), (5.0, 3.0));
        // M>0, A<=0: new_A=M, new_M=M+A
        assert_eq!(couple_scalar(5.0, -2.0), (3.0, 5.0));
        // M<=0, A>0: new_M=M, new_A=M+A
        assert_eq!(couple_scalar(-5.0, 2.0), (-5.0, -3.0));
        // M<=0, A<=0: new_A=M, new_M=M-A
        assert_eq!(couple_scalar(-5.0, -2.0), (-3.0, -5.0));
    }

    #[test]
    fn couple_scalar_zero_magnitude_uses_else_branch() {
        // M == 0 is not "> 0", so the else branch runs.
        // A>0: new_M=0, new_A=0+A=A
        assert_eq!(couple_scalar(0.0, 3.0), (0.0, 3.0));
        // A<=0: new_A=0, new_M=0-A
        assert_eq!(couple_scalar(0.0, -3.0), (3.0, 0.0));
    }

    #[test]
    fn inverse_couple_in_place_pairwise() {
        let mut m = vec![5.0, -5.0, 0.0, 5.0];
        let mut a = vec![2.0, 2.0, -3.0, -2.0];
        inverse_couple(&mut m, &mut a);
        assert_eq!(m, vec![5.0, -5.0, 3.0, 3.0]);
        assert_eq!(a, vec![3.0, -3.0, 0.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn inverse_couple_length_mismatch_panics() {
        let mut m = vec![1.0, 2.0];
        let mut a = vec![1.0];
        inverse_couple(&mut m, &mut a);
    }

    // ---- inverse coupling driver (§4.3.5 loop) ----

    #[test]
    fn inverse_couple_all_single_step_stereo() {
        // One coupling step (magnitude=0, angle=1), the canonical stereo
        // case. residues[0] is L-magnitude, residues[1] is R-angle.
        let mut residues = vec![vec![5.0, -5.0], vec![2.0, 2.0]];
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 0,
            angle_channel: 1,
        }];
        inverse_couple_all(&mut residues, &coupling).unwrap();
        assert_eq!(residues[0], vec![5.0, -5.0]);
        assert_eq!(residues[1], vec![3.0, -3.0]);
    }

    #[test]
    fn inverse_couple_all_runs_steps_in_descending_order() {
        // Two coupling steps that share channel 1: step 0 couples (0,1),
        // step 1 couples (1,2). The spec runs step 1 FIRST (descending),
        // then step 0. Build a case where order matters.
        let mut residues = vec![vec![4.0], vec![2.0], vec![1.0]];
        let coupling = vec![
            MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            },
            MappingCouplingStep {
                magnitude_channel: 1,
                angle_channel: 2,
            },
        ];
        inverse_couple_all(&mut residues, &coupling).unwrap();
        // Step 1 first: M=res[1]=2, A=res[2]=1. M>0,A>0 →
        // res[1]=2, res[2]=2-1=1.
        // Step 0 next: M=res[0]=4, A=res[1]=2. M>0,A>0 →
        // res[0]=4, res[1]=4-2=2.
        assert_eq!(residues[0], vec![4.0]);
        assert_eq!(residues[1], vec![2.0]);
        assert_eq!(residues[2], vec![1.0]);
    }

    #[test]
    fn inverse_couple_all_descending_order_is_observable() {
        // Reverse the dependency so the order is observable: step 0
        // couples (1,2), step 1 couples (0,1). Descending runs step 1
        // (0,1) first then step 0 (1,2).
        let mut residues = vec![vec![10.0], vec![6.0], vec![2.0]];
        let coupling = vec![
            MappingCouplingStep {
                magnitude_channel: 1,
                angle_channel: 2,
            },
            MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            },
        ];
        inverse_couple_all(&mut residues, &coupling).unwrap();
        // Step 1 first: M=res[0]=10, A=res[1]=6 → res[0]=10, res[1]=4.
        // Step 0 next: M=res[1]=4, A=res[2]=2 → res[1]=4, res[2]=2.
        assert_eq!(residues[0], vec![10.0]);
        assert_eq!(residues[1], vec![4.0]);
        assert_eq!(residues[2], vec![2.0]);
    }

    #[test]
    fn inverse_couple_all_handles_angle_below_magnitude_index() {
        // magnitude_channel > angle_channel exercises the split_at_mut
        // ordering branch.
        let mut residues = vec![vec![2.0], vec![5.0]];
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 1,
            angle_channel: 0,
        }];
        inverse_couple_all(&mut residues, &coupling).unwrap();
        // M=res[1]=5, A=res[0]=2. M>0,A>0 → res[1]=5, res[0]=5-2=3.
        assert_eq!(residues[0], vec![3.0]);
        assert_eq!(residues[1], vec![5.0]);
    }

    #[test]
    fn inverse_couple_all_rejects_out_of_range_channel() {
        let mut residues = vec![vec![1.0]];
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 0,
            angle_channel: 1,
        }];
        assert_eq!(
            inverse_couple_all(&mut residues, &coupling),
            Err(CouplingError::ChannelOutOfRange {
                step: 0,
                channel: 1,
                channels: 1,
            })
        );
    }

    #[test]
    fn inverse_couple_all_rejects_same_channel() {
        let mut residues = vec![vec![1.0], vec![2.0]];
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 1,
            angle_channel: 1,
        }];
        assert_eq!(
            inverse_couple_all(&mut residues, &coupling),
            Err(CouplingError::SameChannel {
                step: 0,
                channel: 1,
            })
        );
    }

    #[test]
    fn inverse_couple_all_empty_coupling_is_noop() {
        let mut residues = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let before = residues.clone();
        inverse_couple_all(&mut residues, &[]).unwrap();
        assert_eq!(residues, before);
    }
}

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
//!
//! # Forward channel coupling — the encoder counterpart of §4.3.5
//!
//! [`forward_couple_scalar`] / [`forward_couple`] / [`forward_couple_all`]
//! are the encoder-side primitives the encoder applies *before* the
//! per-channel floor/residue path, mirroring the round-29 `mdct_naive` /
//! round-32 `FrameSplitter` "encoder counterpart of a decoder primitive"
//! pattern. The §4.3.5 inverse rule is a deterministic four-quadrant
//! map `(M, A) -> (new_M, new_A)`; inverting it gives a deterministic
//! map `(L, R) -> (M, A)` such that
//!
//! ```text
//! forward_couple_scalar(L, R) = (M, A)
//! inverse_couple_scalar(M, A) = (L, R)
//! ```
//!
//! is a per-sample identity for every real `(L, R)` pair. The four
//! forward cases come straight out of the §4.3.5 step-3 rule by
//! algebraic inversion:
//!
//! | §4.3.5 case (sign of `M`, `A`) | `new_M` | `new_A` | Forward recovery |
//! | --- | --- | --- | --- |
//! | `M > 0`, `A > 0` | `M`     | `M - A` | `M = L`, `A = L - R`; fires when `L > 0 AND L > R` |
//! | `M > 0`, `A ≤ 0` | `M + A` | `M`     | `M = R`, `A = L - R`; fires when `R > 0 AND L ≤ R` |
//! | `M ≤ 0`, `A > 0` | `M`     | `M + A` | `M = L`, `A = R - L`; fires when `L ≤ 0 AND R > L` |
//! | `M ≤ 0`, `A ≤ 0` | `M - A` | `M`     | `M = R`, `A = R - L`; fires when `R ≤ 0 AND R ≤ L` |
//!
//! Each `(L, R)` lands in exactly one of the four cases — the
//! conditions are mutually exclusive and exhaustive, and the boundary
//! values (zero, ties) are absorbed by the existing `> 0` / `≤ 0`
//! splits the inverse uses on `M` and `A`. The
//! `forward_then_inverse_couple_is_identity_*` tests pin the
//! round-trip property exhaustively on a grid of `(L, R)` values
//! covering every quadrant and every boundary tie.
//!
//! # Spectrum factoring — the encoder counterpart of §4.3.6
//!
//! The §4.3.6 "dot product" step multiplies, element by element, each
//! channel's floor curve by its residue vector to produce that channel's
//! length-`n/2` audio spectrum (`spectrum[i] = floor[i] · residue[i]`,
//! the element-wise product modelled by [`crate::packet::dot_product`] /
//! [`crate::packet::dot_product_all`] on the decode side).
//!
//! [`factor_spectrum_scalar`] / [`factor_spectrum`] / [`factor_spectrum_all`]
//! are the encoder-side inverse: given a target audio spectrum and the
//! floor curve the encoder has already chosen for the channel, they
//! recover the residue vector the encoder must quantise and emit so that
//! the decoder's §4.3.6 product reproduces the target spectrum. The
//! recovery is the algebraic inverse of the element-wise product,
//!
//! ```text
//! residue[i] = spectrum[i] / floor[i]
//! ```
//!
//! and the round-trip property
//! `dot_product(floor, factor_spectrum(spectrum, floor)) == spectrum`
//! holds bit-exactly wherever `floor[i]` is finite and nonzero.
//!
//! The one structural subtlety is a zero floor bin. The §4.3.2 floor
//! synthesis can return a zero element (e.g. a floor-1 line at the curve
//! minimum, or the all-zero curve of an `'unused'` channel, §4.3.3).
//! Where `floor[i] == 0` the decode product `floor[i] · residue[i]` is
//! zero for *any* residue value — the residue bin is unconstrained by the
//! spectrum, so a target spectrum that is consistent with the floor must
//! itself have `spectrum[i] == 0` there. The factoring emits the
//! canonical residue `0.0` for such bins (the smallest-magnitude
//! representative, the natural choice for a quantiser) and rejects a
//! target whose `spectrum[i] != 0` over a zero floor bin, since no finite
//! residue could reproduce it. A non-finite floor (`NaN`/`±∞`, never
//! produced by the §4.3.2 synthesis but guarded against caller bugs) is
//! likewise rejected rather than yielding a non-finite or NaN residue.

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

/// Apply the §4.3.5 forward-coupling rule to a single
/// `(left, right)` Cartesian scalar pair, returning
/// `(magnitude, angle)` in square-polar form.
///
/// This function is the per-sample algebraic inverse of
/// [`couple_scalar`]: for every real input `(l, r)` the round-trip
/// identity
///
/// ```text
/// let (m, a) = forward_couple_scalar(l, r);
/// couple_scalar(m, a) == (l, r)
/// ```
///
/// holds exactly (modulo `f32` rounding; for representable inputs the
/// `+` / `-` operations are exact, so the identity is bit-exact for
/// every legal pair).
///
/// The four §4.3.5 step-3 cases (sign of `M`, sign of `A`) partition
/// the `(L, R)` plane via the conditions tabulated in the module
/// header; each case carries a closed-form recovery of `(M, A)`:
///
/// ```text
/// L > 0  AND L > R  : M = L, A = L - R   (mirrors `M > 0, A > 0`)
/// R > 0  AND L ≤ R  : M = R, A = L - R   (mirrors `M > 0, A ≤ 0`)
/// L ≤ 0  AND R > L  : M = L, A = R - L   (mirrors `M ≤ 0, A > 0`)
/// R ≤ 0  AND R ≤ L  : M = R, A = R - L   (mirrors `M ≤ 0, A ≤ 0`)
/// ```
#[must_use]
pub fn forward_couple_scalar(l: f32, r: f32) -> (f32, f32) {
    if l > 0.0 {
        if l > r {
            // §4.3.5 case (M > 0, A > 0). M = L, A = L - R.
            (l, l - r)
        } else {
            // §4.3.5 case (M > 0, A ≤ 0). M = R, A = L - R.
            (r, l - r)
        }
    } else if r > l {
        // §4.3.5 case (M ≤ 0, A > 0). M = L, A = R - L.
        (l, r - l)
    } else {
        // §4.3.5 case (M ≤ 0, A ≤ 0). M = R, A = R - L.
        (r, r - l)
    }
}

/// Forward-couple one Cartesian (left/right) vector pair in place,
/// producing a square-polar (magnitude/angle) vector pair (§4.3.5,
/// applied in encoder order). Both vectors must already hold the
/// post-MDCT Cartesian spectra for their respective channels and must
/// be the same length (the encoder per-vector length is `n/2`,
/// mirroring the decoder §4.3.4 step 5). Each scalar position is
/// coupled independently via [`forward_couple_scalar`].
///
/// After the call, `left` holds the magnitude vector `M` and `right`
/// holds the angle vector `A` of §4.3.5; these are the values the
/// residue encoder will quantise and emit.
///
/// # Panics
///
/// Panics if `left` and `right` have different lengths; the §4.3.5
/// loop pairs Cartesian vectors that are always `n/2` long, so a
/// mismatch indicates a caller bug rather than stream data.
pub fn forward_couple(left: &mut [f32], right: &mut [f32]) {
    assert_eq!(
        left.len(),
        right.len(),
        "forward_couple: left/right length mismatch"
    );
    for (l, r) in left.iter_mut().zip(right.iter_mut()) {
        let (new_m, new_a) = forward_couple_scalar(*l, *r);
        *l = new_m;
        *r = new_a;
    }
}

/// Run the full §4.3.5 forward-coupling pass over a slice of
/// per-channel Cartesian spectra, applying every coupling step **in
/// ascending order** (`0 … coupling_steps-1`) — the reverse of the
/// decoder-side [`inverse_couple_all`] descent. Ascending order is the
/// correct encoder direction because the decoder undoes the encoder
/// effects in reverse: if the encoder applied step 0 → step 1 → … in
/// order, the decoder must undo step N-1 → step N-2 → … → step 0 to
/// recover the original Cartesian spectra.
///
/// `channels[c]` is the Cartesian spectrum for channel `c`; each
/// coupling step couples the named `(magnitude_channel, angle_channel)`
/// pair in place. After the call:
///
/// * For every coupled `magnitude_channel`, the vector holds the
///   square-polar magnitude `M` ready to feed into the residue encoder.
/// * For every coupled `angle_channel`, the vector holds the square-
///   polar angle `A` ready to feed into the residue encoder.
/// * Uncoupled channels remain unchanged.
///
/// The function is the byte-exact inverse of [`inverse_couple_all`]
/// when the coupling step list is identical and the per-channel
/// vectors have the same length. The round-trip property
/// `inverse_couple_all(forward_couple_all(x)) == x` holds for every
/// legal input.
///
/// # Errors
///
/// [`CouplingError::ChannelOutOfRange`] if a step names a channel
/// index `>= channels.len()`, or [`CouplingError::SameChannel`] if a
/// step names the same channel for both magnitude and angle. The
/// error variants and meanings match [`inverse_couple_all`] exactly so
/// a caller driving both directions through a shared error shape can
/// surface them uniformly.
pub fn forward_couple_all(
    channels: &mut [Vec<f32>],
    coupling: &[MappingCouplingStep],
) -> Result<(), CouplingError> {
    let n_channels = channels.len();
    // The forward-direction loop is the §4.3.5 inverse loop run
    // backwards: ascending order, so each subsequent step sees the
    // square-polar output of every prior step.
    for (step, cs) in coupling.iter().enumerate() {
        let mag = cs.magnitude_channel as usize;
        let ang = cs.angle_channel as usize;
        if mag >= n_channels {
            return Err(CouplingError::ChannelOutOfRange {
                step,
                channel: mag,
                channels: n_channels,
            });
        }
        if ang >= n_channels {
            return Err(CouplingError::ChannelOutOfRange {
                step,
                channel: ang,
                channels: n_channels,
            });
        }
        if mag == ang {
            return Err(CouplingError::SameChannel { step, channel: mag });
        }
        // Split the slice so we can borrow the two vectors mutably at
        // once. `mag != ang` is guaranteed above. The split-vs-name
        // layout mirrors `inverse_couple_all` exactly: `lo`/`hi` pick
        // the slice partition, and the `mag < ang` branch decides
        // which side carries left vs right.
        let (lo, hi) = (mag.min(ang), mag.max(ang));
        let (head, tail) = channels.split_at_mut(hi);
        let (a, b) = (&mut head[lo], &mut tail[0]);
        if mag < ang {
            // `a` is the lower-index slot. mag < ang means mag == lo,
            // so the magnitude channel is `a` (= left), the angle
            // channel is `b` (= right).
            forward_couple(a, b);
        } else {
            // mag > ang. mag == hi, so the magnitude channel is `b`,
            // the angle channel is `a`.
            forward_couple(b, a);
        }
    }
    Ok(())
}

/// The square-polar coupling energy split for one Cartesian channel pair
/// (Vorbis I §4.3.5, encode direction) — the figures a coupling-decision
/// heuristic reads. Computed by *measuring* the forward-coupling output
/// without committing it: the inputs are left untouched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CouplingEnergy {
    /// `Σ M²` — the energy of the magnitude vector the coupling produces
    /// (the per-position larger-channel value, §4.3.5).
    pub magnitude_energy: f64,
    /// `Σ A²` — the energy of the angle vector the coupling produces (the
    /// signed `L − R` / `R − L` difference, §4.3.5). When the two channels
    /// are highly correlated this is small relative to `magnitude_energy`,
    /// which is exactly when coupling concentrates the stereo image into
    /// the magnitude channel and the angle quantises cheaply.
    pub angle_energy: f64,
    /// `Σ L² + Σ R²` — the energy that would be coded if the two channels
    /// were left **uncoupled** (each residue-coded independently). The
    /// square-polar transform is energy-preserving per position
    /// (`M² + A² = max² + (L−R)²`, which is *not* `L² + R²` in general),
    /// so comparing `magnitude_energy + angle_energy` against this is the
    /// honest before/after the decision weighs.
    pub uncoupled_energy: f64,
}

impl CouplingEnergy {
    /// `angle_energy / magnitude_energy` — the coupling "angle ratio".
    /// A small ratio means the angle vector is low-energy relative to the
    /// magnitude vector, i.e. the channels are strongly correlated and
    /// coupling pays off (the angle residue codes cheaply). Returns
    /// `0.0` when both energies are zero (a silent pair: coupling is
    /// trivially neutral), and `f64::INFINITY` when the magnitude energy
    /// is zero but the angle energy is not (which the §4.3.5 `M = max`
    /// construction cannot actually produce, but is defined defensively).
    pub fn angle_ratio(&self) -> f64 {
        if self.magnitude_energy == 0.0 {
            if self.angle_energy == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            self.angle_energy / self.magnitude_energy
        }
    }
}

/// Measure the §4.3.5 square-polar coupling energy split for a Cartesian
/// `(left, right)` channel pair **without** mutating either channel — the
/// non-committing analogue of [`forward_couple`] a coupling-decision
/// heuristic uses to weigh whether to actually couple the pair.
///
/// Each position is forward-coupled with the same [`forward_couple_scalar`]
/// rule the real transform applies, and the resulting magnitude/angle
/// energies (plus the uncoupled `L² + R²` baseline) are accumulated in
/// `f64`. No quantisation is modelled — this is the pre-quantisation
/// energy picture, the cheapest signal a per-region coupling gate can key
/// off.
///
/// # Panics
///
/// Panics if `left` and `right` differ in length, matching
/// [`forward_couple`]'s contract (the §4.3.5 loop pairs `n/2`-long
/// Cartesian vectors).
pub fn coupling_energy(left: &[f32], right: &[f32]) -> CouplingEnergy {
    assert_eq!(
        left.len(),
        right.len(),
        "coupling_energy: left/right length mismatch"
    );
    let mut magnitude_energy = 0.0f64;
    let mut angle_energy = 0.0f64;
    let mut uncoupled_energy = 0.0f64;
    for (&l, &r) in left.iter().zip(right.iter()) {
        let (m, a) = forward_couple_scalar(l, r);
        magnitude_energy += f64::from(m) * f64::from(m);
        angle_energy += f64::from(a) * f64::from(a);
        uncoupled_energy += f64::from(l) * f64::from(l) + f64::from(r) * f64::from(r);
    }
    CouplingEnergy {
        magnitude_energy,
        angle_energy,
        uncoupled_energy,
    }
}

/// Decide whether a Cartesian channel pair is worth coupling (Vorbis I
/// §4.3.5, encode direction), by the [`CouplingEnergy::angle_ratio`]
/// heuristic: couple when the angle vector's energy is at most
/// `max_angle_ratio` times the magnitude vector's energy.
///
/// A correlated stereo pair forward-couples to a small angle vector (the
/// `L − R` difference is near zero where the channels agree), so its angle
/// ratio is low — coupling concentrates the energy into the magnitude
/// channel and the angle residue quantises toward zero. An anti-correlated
/// or independent pair forward-couples to a large angle vector, so coupling
/// would buy nothing (it would just move energy from `R` into `A`). The
/// `max_angle_ratio` threshold is the exchange the caller picks: a larger
/// threshold couples more aggressively.
///
/// `max_angle_ratio` must be finite and non-negative; the function returns
/// `false` for a non-finite or negative threshold (a malformed gate never
/// couples). A silent pair (both energies zero) has angle ratio `0.0`, so
/// it couples for any non-negative threshold — harmless, since coupling a
/// zero pair is the identity.
///
/// # Panics
///
/// Panics on a `left`/`right` length mismatch (see [`coupling_energy`]).
pub fn should_couple(left: &[f32], right: &[f32], max_angle_ratio: f64) -> bool {
    if !max_angle_ratio.is_finite() || max_angle_ratio < 0.0 {
        return false;
    }
    coupling_energy(left, right).angle_ratio() <= max_angle_ratio
}

/// Prune a candidate §4.3.5 coupling-step list down to the steps worth
/// applying, by the [`should_couple`] angle-ratio gate — the encoder's
/// "which channel pairs to actually couple" decision.
///
/// A mapping may *offer* a coupling step list (e.g. couple every adjacent
/// channel pair); this routine decides which of those steps pay off on the
/// **actual** spectra and returns the kept subset (a sub-list of the input,
/// in the same ascending order). The decision is sequential and
/// order-faithful to [`forward_couple_all`]: the steps are visited in
/// ascending order on a *working copy* of the spectra, and a **kept** step
/// forward-couples its pair in the working copy so later steps that
/// reference the same channel see its square-polar magnitude/angle (exactly
/// what the real transform would feed them). A **dropped** step leaves its
/// channels Cartesian in the working copy, so a later step referencing the
/// dropped angle channel still sees the original spectrum. The input
/// `channels` slice is **not** mutated — only an internal copy is.
///
/// Each kept step is the original [`MappingCouplingStep`]; threading the
/// returned list back through [`forward_couple_all`] (and recording it as
/// the mapping's coupling for the matching decode-side
/// [`inverse_couple_all`]) yields a self-consistent encode.
///
/// `max_angle_ratio` is the shared [`should_couple`] threshold; a
/// non-finite or negative value drops every step (the gate never couples).
///
/// # Errors
///
/// [`CouplingError::ChannelOutOfRange`] if a step names a channel index
/// `>= channels.len()`, or [`CouplingError::SameChannel`] if a step names
/// the same channel for both magnitude and angle — the identical
/// validation [`forward_couple_all`] performs, applied to every candidate
/// step (including dropped ones, so a malformed list is rejected rather
/// than silently pruned).
pub fn prune_coupling_steps(
    channels: &[Vec<f32>],
    coupling: &[MappingCouplingStep],
    max_angle_ratio: f64,
) -> Result<Vec<MappingCouplingStep>, CouplingError> {
    let n_channels = channels.len();
    // Validate every candidate step up front so a malformed list is
    // rejected regardless of which steps would be kept.
    for (step, cs) in coupling.iter().enumerate() {
        let mag = cs.magnitude_channel as usize;
        let ang = cs.angle_channel as usize;
        if mag >= n_channels {
            return Err(CouplingError::ChannelOutOfRange {
                step,
                channel: mag,
                channels: n_channels,
            });
        }
        if ang >= n_channels {
            return Err(CouplingError::ChannelOutOfRange {
                step,
                channel: ang,
                channels: n_channels,
            });
        }
        if mag == ang {
            return Err(CouplingError::SameChannel { step, channel: mag });
        }
    }

    // Work on a copy so the decision can model the cumulative effect of the
    // steps it keeps without disturbing the caller's spectra.
    let mut work: Vec<Vec<f32>> = channels.to_vec();
    let mut kept = Vec::new();
    for cs in coupling {
        let mag = cs.magnitude_channel as usize;
        let ang = cs.angle_channel as usize;
        // Decide on the working copy's *current* spectra (post any prior
        // kept couplings), then — if kept — commit the coupling so later
        // steps referencing these channels see the square-polar result.
        let couple = should_couple(&work[mag], &work[ang], max_angle_ratio);
        if couple {
            let (lo, hi) = (mag.min(ang), mag.max(ang));
            let (head, tail) = work.split_at_mut(hi);
            let (a, b) = (&mut head[lo], &mut tail[0]);
            if mag < ang {
                forward_couple(a, b);
            } else {
                forward_couple(b, a);
            }
            kept.push(*cs);
        }
    }
    Ok(kept)
}

/// Errors that can arise while factoring a target audio spectrum into a
/// residue vector (the encoder-side inverse of the §4.3.6 dot product).
#[derive(Debug, Clone, PartialEq)]
pub enum FactorSpectrumError {
    /// `spectrum` and `floor` slices disagree on length. Both are the
    /// per-channel §4.3.6 length `n/2`, so a mismatch indicates a caller
    /// bug rather than stream data.
    LengthMismatch {
        /// Length of the target spectrum slice.
        spectrum_len: usize,
        /// Length of the floor curve slice.
        floor_len: usize,
    },
    /// `floors` and `spectra` have a different number of channels. The
    /// channel-driver [`factor_spectrum_all`] requires one floor curve
    /// per spectrum.
    ChannelCountMismatch {
        /// Number of target spectrum vectors supplied.
        spectra: usize,
        /// Number of floor curves supplied.
        floors: usize,
    },
    /// A floor element is not finite (`NaN` or `±∞`). The §4.3.2 floor
    /// synthesis never produces such a value; a non-finite floor would
    /// make the recovered residue non-finite, so the factoring refuses.
    NonFiniteFloor {
        /// The channel whose floor carried the non-finite element
        /// (`0` for the single-vector [`factor_spectrum`]).
        channel: usize,
        /// The bin index of the offending floor element.
        index: usize,
    },
    /// The target spectrum is nonzero at a bin where the floor is zero.
    /// The §4.3.6 product `floor[i] · residue[i]` is zero for any
    /// residue when `floor[i] == 0`, so no finite residue could
    /// reproduce a nonzero spectrum there — the target is inconsistent
    /// with the chosen floor.
    NonzeroSpectrumOverZeroFloor {
        /// The channel whose target was inconsistent (`0` for the
        /// single-vector [`factor_spectrum`]).
        channel: usize,
        /// The bin index where the floor is zero but the spectrum is not.
        index: usize,
        /// The offending nonzero spectrum value.
        spectrum: f32,
    },
}

impl core::fmt::Display for FactorSpectrumError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FactorSpectrumError::LengthMismatch {
                spectrum_len,
                floor_len,
            } => write!(
                f,
                "spectrum factoring (§4.3.6 inverse): spectrum length {spectrum_len} \
                 does not match floor length {floor_len}"
            ),
            FactorSpectrumError::ChannelCountMismatch { spectra, floors } => write!(
                f,
                "spectrum factoring (§4.3.6 inverse): {spectra} spectrum channels \
                 but {floors} floor curves"
            ),
            FactorSpectrumError::NonFiniteFloor { channel, index } => write!(
                f,
                "spectrum factoring (§4.3.6 inverse): channel {channel} floor \
                 element at index {index} is not finite"
            ),
            FactorSpectrumError::NonzeroSpectrumOverZeroFloor {
                channel,
                index,
                spectrum,
            } => write!(
                f,
                "spectrum factoring (§4.3.6 inverse): channel {channel} target \
                 spectrum {spectrum} at index {index} is nonzero where the floor \
                 is zero (no finite residue reproduces it)"
            ),
        }
    }
}

impl std::error::Error for FactorSpectrumError {}

/// Recover one residue scalar from a target audio-spectrum scalar and
/// the floor scalar the encoder chose for that bin — the scalar
/// algebraic inverse of the §4.3.6 element-wise product
/// `spectrum = floor · residue`.
///
/// Returns `residue = spectrum / floor` for a finite nonzero `floor`.
/// When `floor == 0` the product is zero for any residue, so the bin is
/// unconstrained: the function returns the canonical `0.0` if the target
/// `spectrum` is also zero, and rejects a nonzero target (no finite
/// residue could reproduce it). A non-finite `floor` is rejected.
///
/// The `channel`/`index` coordinates are carried into the error variants
/// only; the math itself is per-scalar.
fn factor_spectrum_scalar(
    spectrum: f32,
    floor: f32,
    channel: usize,
    index: usize,
) -> Result<f32, FactorSpectrumError> {
    if !floor.is_finite() {
        return Err(FactorSpectrumError::NonFiniteFloor { channel, index });
    }
    if floor == 0.0 {
        if spectrum != 0.0 {
            return Err(FactorSpectrumError::NonzeroSpectrumOverZeroFloor {
                channel,
                index,
                spectrum,
            });
        }
        return Ok(0.0);
    }
    Ok(spectrum / floor)
}

/// Factor one channel's target audio spectrum into its residue vector —
/// the encoder-side inverse of [`crate::packet::dot_product`].
///
/// Given the length-`n/2` `spectrum` the encoder wants the decoder to
/// reconstruct and the length-`n/2` `floor` curve the encoder has
/// already chosen for the channel, this writes the residue vector
/// `residue[i] = spectrum[i] / floor[i]` such that the decoder's §4.3.6
/// product `floor ⊙ residue` reproduces `spectrum` bit-exactly wherever
/// the floor is finite and nonzero. Where `floor[i] == 0` the residue is
/// unconstrained and the canonical `0.0` is emitted (the target must
/// itself be zero there — see the module header).
///
/// `residue` is a caller-allocated output buffer; it is overwritten in
/// full on success and left in an unspecified partially-written state on
/// error (the caller discards it).
///
/// # Errors
///
/// [`FactorSpectrumError::LengthMismatch`] if `spectrum`, `floor` and
/// `residue` are not all the same length;
/// [`FactorSpectrumError::NonFiniteFloor`] for a `NaN`/`±∞` floor bin;
/// [`FactorSpectrumError::NonzeroSpectrumOverZeroFloor`] if the target
/// is nonzero at a zero-floor bin. The `channel` coordinate in the
/// latter two is `0` (use [`factor_spectrum_all`] for per-channel
/// coordinates).
pub fn factor_spectrum(
    spectrum: &[f32],
    floor: &[f32],
    residue: &mut [f32],
) -> Result<(), FactorSpectrumError> {
    if spectrum.len() != floor.len() {
        return Err(FactorSpectrumError::LengthMismatch {
            spectrum_len: spectrum.len(),
            floor_len: floor.len(),
        });
    }
    if residue.len() != spectrum.len() {
        return Err(FactorSpectrumError::LengthMismatch {
            spectrum_len: spectrum.len(),
            floor_len: residue.len(),
        });
    }
    for (index, ((out, &s), &fl)) in residue
        .iter_mut()
        .zip(spectrum.iter())
        .zip(floor.iter())
        .enumerate()
    {
        *out = factor_spectrum_scalar(s, fl, 0, index)?;
    }
    Ok(())
}

/// Factor every channel's target audio spectrum into its residue vector
/// — the encoder-side inverse of [`crate::packet::dot_product_all`].
///
/// * `spectra[channel]` is the channel's length-`half_n` target audio
///   spectrum (the §4.3.6 output the encoder wants the decoder to
///   reconstruct).
/// * `floors[channel]` is the channel's chosen floor curve, or [`None`]
///   for a channel the encoder is coding as `'unused'` (§4.3.2 step 6 /
///   §4.3.3 — the decode-side counterpart [`crate::packet::dot_product_all`]
///   models such a channel as a `None` floor and emits the all-zero
///   spectrum). A `None` channel's target must therefore be all zero;
///   the recovered residue is the empty vector (the channel carries no
///   residue in the stream).
///
/// Returns one residue vector per channel, in channel order: the
/// length-`half_n` factored residue for a `Some` floor, or an empty
/// vector for a `None` (unused) floor.
///
/// # Errors
///
/// [`FactorSpectrumError::ChannelCountMismatch`] if `spectra` and
/// `floors` differ in channel count; [`FactorSpectrumError::LengthMismatch`]
/// if a `Some` floor curve and its paired spectrum disagree on length;
/// [`FactorSpectrumError::NonFiniteFloor`] /
/// [`FactorSpectrumError::NonzeroSpectrumOverZeroFloor`] (carrying the
/// channel index) from the per-bin factoring, including a `None`
/// channel whose target spectrum is not all zero (reported as a zero
/// floor over the first nonzero bin).
pub fn factor_spectrum_all(
    spectra: &[Vec<f32>],
    floors: &[Option<Vec<f32>>],
) -> Result<Vec<Vec<f32>>, FactorSpectrumError> {
    if spectra.len() != floors.len() {
        return Err(FactorSpectrumError::ChannelCountMismatch {
            spectra: spectra.len(),
            floors: floors.len(),
        });
    }
    let mut out = Vec::with_capacity(spectra.len());
    for (channel, (spectrum, floor)) in spectra.iter().zip(floors.iter()).enumerate() {
        match floor {
            // An 'unused' channel carries no residue; its target spectrum
            // must be all zero (the decoder emits zero for it regardless).
            None => {
                for (index, &s) in spectrum.iter().enumerate() {
                    if s != 0.0 {
                        return Err(FactorSpectrumError::NonzeroSpectrumOverZeroFloor {
                            channel,
                            index,
                            spectrum: s,
                        });
                    }
                }
                out.push(Vec::new());
            }
            Some(curve) => {
                if curve.len() != spectrum.len() {
                    return Err(FactorSpectrumError::LengthMismatch {
                        spectrum_len: spectrum.len(),
                        floor_len: curve.len(),
                    });
                }
                let mut residue = vec![0.0f32; spectrum.len()];
                for (index, ((out_bin, &s), &fl)) in residue
                    .iter_mut()
                    .zip(spectrum.iter())
                    .zip(curve.iter())
                    .enumerate()
                {
                    *out_bin = factor_spectrum_scalar(s, fl, channel, index)?;
                }
                out.push(residue);
            }
        }
    }
    Ok(out)
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

    // ---- forward channel coupling (§4.3.5, encoder direction) ----

    #[test]
    fn forward_couple_scalar_all_four_quadrants() {
        // Mirror of `couple_scalar_all_four_quadrants` from the inverse
        // side. The §4.3.5 step-3 table maps:
        //   M= 5, A= 2 → (new_M= 5, new_A= 3); so (L,R)=(5,3) → (5, 2).
        //   M= 5, A=-2 → (new_M= 3, new_A= 5); so (L,R)=(3,5) → (5,-2).
        //   M=-5, A= 2 → (new_M=-5, new_A=-3); so (L,R)=(-5,-3) → (-5,2).
        //   M=-5, A=-2 → (new_M=-3, new_A=-5); so (L,R)=(-3,-5) → (-5,-2).
        assert_eq!(forward_couple_scalar(5.0, 3.0), (5.0, 2.0));
        assert_eq!(forward_couple_scalar(3.0, 5.0), (5.0, -2.0));
        assert_eq!(forward_couple_scalar(-5.0, -3.0), (-5.0, 2.0));
        assert_eq!(forward_couple_scalar(-3.0, -5.0), (-5.0, -2.0));
    }

    #[test]
    fn forward_couple_scalar_handles_boundary_ties() {
        // L == R, all signs: every tie should round-trip cleanly back
        // through couple_scalar.
        for &(l, r) in &[(0.0_f32, 0.0), (5.0, 5.0), (-5.0, -5.0)] {
            let (m, a) = forward_couple_scalar(l, r);
            let (rl, rr) = couple_scalar(m, a);
            assert_eq!(
                (rl, rr),
                (l, r),
                "tie (L,R)=({l},{r}) failed round-trip: forward → (M,A)=({m},{a}) → inverse → ({rl},{rr})"
            );
        }
    }

    #[test]
    fn forward_couple_scalar_handles_zeros() {
        // Each axis zero with opposite-sign other coord.
        for &(l, r) in &[(0.0_f32, 5.0), (0.0, -5.0), (5.0, 0.0), (-5.0, 0.0)] {
            let (m, a) = forward_couple_scalar(l, r);
            let (rl, rr) = couple_scalar(m, a);
            assert_eq!((rl, rr), (l, r), "(L,R)=({l},{r}) failed round-trip");
        }
    }

    #[test]
    fn forward_then_inverse_couple_scalar_is_identity_on_grid() {
        // Exhaustive (L, R) round-trip over an integer grid covering
        // every quadrant and every L/R sign comparison.
        for li in -10..=10 {
            for ri in -10..=10 {
                let l = li as f32;
                let r = ri as f32;
                let (m, a) = forward_couple_scalar(l, r);
                let (rl, rr) = couple_scalar(m, a);
                assert_eq!(
                    (rl, rr),
                    (l, r),
                    "round-trip failed at (L,R)=({l},{r}); intermediate (M,A)=({m},{a})"
                );
            }
        }
    }

    #[test]
    fn forward_then_inverse_couple_scalar_is_identity_on_floats() {
        // A handful of non-integer / non-tie probes that exercise the
        // four §4.3.5 cases on values the integer grid never visits.
        let probes: &[(f32, f32)] = &[
            (1.5, 0.25),
            (0.25, 1.5),
            (-1.5, -0.25),
            (-0.25, -1.5),
            (1.5, -0.25),
            (-1.5, 0.25),
            (1e-6, 2e-6),
            (-1e-6, -2e-6),
            (12345.678, -98765.43),
            (-0.001, 0.001),
        ];
        for &(l, r) in probes {
            let (m, a) = forward_couple_scalar(l, r);
            let (rl, rr) = couple_scalar(m, a);
            assert_eq!(
                (rl, rr),
                (l, r),
                "round-trip failed at (L,R)=({l},{r}); intermediate (M,A)=({m},{a})"
            );
        }
    }

    #[test]
    fn forward_couple_pointwise_matches_scalar_function() {
        // Confirm the vector wrapper applies forward_couple_scalar
        // element-by-element with no cross-talk.
        let mut left = vec![5.0_f32, 3.0, -5.0, -3.0];
        let mut right = vec![3.0_f32, 5.0, -3.0, -5.0];
        let expected_m: Vec<f32> = left
            .iter()
            .zip(right.iter())
            .map(|(&l, &r)| forward_couple_scalar(l, r).0)
            .collect();
        let expected_a: Vec<f32> = left
            .iter()
            .zip(right.iter())
            .map(|(&l, &r)| forward_couple_scalar(l, r).1)
            .collect();
        forward_couple(&mut left, &mut right);
        assert_eq!(left, expected_m);
        assert_eq!(right, expected_a);
    }

    #[test]
    fn forward_then_inverse_couple_is_identity_on_vectors() {
        // End-to-end vector round-trip. After forward_couple the
        // vectors hold (M, A); after inverse_couple they return to the
        // original (L, R) bit-exactly for representable inputs.
        let original_l = vec![5.0_f32, 3.0, -5.0, -3.0, 0.0, 1.5, -1.5, 0.25];
        let original_r = vec![3.0_f32, 5.0, -3.0, -5.0, 0.0, 0.25, -0.25, 1.5];
        let mut left = original_l.clone();
        let mut right = original_r.clone();
        forward_couple(&mut left, &mut right);
        inverse_couple(&mut left, &mut right);
        assert_eq!(left, original_l);
        assert_eq!(right, original_r);
    }

    #[test]
    #[should_panic(expected = "forward_couple: left/right length mismatch")]
    fn forward_couple_panics_on_length_mismatch() {
        let mut left = vec![1.0_f32, 2.0];
        let mut right = vec![3.0_f32];
        forward_couple(&mut left, &mut right);
    }

    #[test]
    fn forward_couple_all_single_step_low_high() {
        // Mirror of inverse_couple_all_single_step. Coupling step
        // names channel 0 as magnitude and channel 1 as angle.
        // Forward direction: starting Cartesian (L, R) = (5, 3) on
        // channels (0, 1) should yield (M=5, A=2).
        let mut spectra = vec![vec![5.0_f32], vec![3.0_f32]];
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 0,
            angle_channel: 1,
        }];
        forward_couple_all(&mut spectra, &coupling).unwrap();
        assert_eq!(spectra[0], vec![5.0]);
        assert_eq!(spectra[1], vec![2.0]);
    }

    #[test]
    fn forward_couple_all_single_step_high_low() {
        // Mirror of inverse_couple_all_single_step_high_low. Channel
        // 1 is magnitude, channel 0 is angle. Starting Cartesian
        // (M-channel=1, A-channel=0) holds (L=5, R=2). Wait — the
        // Cartesian inputs are by channel index, not by role. With
        // mag=1, ang=0 the (L, R) pair is (channel-with-lower-index,
        // channel-with-higher-index) = (residues[0], residues[1]).
        //
        // But by the mag/ang naming the forward direction packs
        // channel 1's pre-coupling value as L and channel 0's as R?
        // No — the §4.3.5 inverse pulls magnitude_vector from
        // channel mag and angle_vector from channel ang. So the
        // forward direction places M into channel mag and A into
        // channel ang. The L/R pair is implicit: forward_couple_all
        // reads channel mag's pre-coupling value and channel ang's
        // pre-coupling value, with mag's value used as L and ang's
        // value used as R (because mag was filled by L, ang by R in
        // a hypothetical reversal of the inverse-side identity).
        //
        // Pinning concrete numbers: the inverse-side test puts
        // residues = [2, 5], coupling {mag:1, ang:0}, and gets back
        // (3, 5) — meaning post-inverse channel 0 = 3, channel 1 = 5.
        // The forward direction therefore takes pre-coupling
        // (channel 0, channel 1) = (3, 5) (= (R, L) because
        // ang=channel 0 and mag=channel 1), and should produce
        // (channel 0, channel 1) = (2, 5) (= (A, M)).
        let mut spectra = vec![vec![3.0_f32], vec![5.0_f32]];
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 1,
            angle_channel: 0,
        }];
        forward_couple_all(&mut spectra, &coupling).unwrap();
        assert_eq!(spectra[0], vec![2.0]); // angle channel
        assert_eq!(spectra[1], vec![5.0]); // magnitude channel
    }

    #[test]
    fn forward_couple_all_rejects_out_of_range_channel() {
        let mut spectra = vec![vec![1.0_f32]];
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 0,
            angle_channel: 1,
        }];
        assert_eq!(
            forward_couple_all(&mut spectra, &coupling),
            Err(CouplingError::ChannelOutOfRange {
                step: 0,
                channel: 1,
                channels: 1,
            })
        );
    }

    #[test]
    fn forward_couple_all_rejects_out_of_range_magnitude_channel() {
        let mut spectra = vec![vec![1.0_f32], vec![2.0_f32]];
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 5,
            angle_channel: 0,
        }];
        assert_eq!(
            forward_couple_all(&mut spectra, &coupling),
            Err(CouplingError::ChannelOutOfRange {
                step: 0,
                channel: 5,
                channels: 2,
            })
        );
    }

    #[test]
    fn forward_couple_all_rejects_same_channel() {
        let mut spectra = vec![vec![1.0_f32], vec![2.0_f32]];
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 1,
            angle_channel: 1,
        }];
        assert_eq!(
            forward_couple_all(&mut spectra, &coupling),
            Err(CouplingError::SameChannel {
                step: 0,
                channel: 1,
            })
        );
    }

    #[test]
    fn forward_couple_all_empty_coupling_is_noop() {
        let mut spectra = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        let before = spectra.clone();
        forward_couple_all(&mut spectra, &[]).unwrap();
        assert_eq!(spectra, before);
    }

    #[test]
    fn forward_then_inverse_couple_all_is_identity_single_step() {
        // The encoder-then-decoder round-trip on a typical stereo
        // pair: a sine-ish waveform on the left, a slightly delayed
        // version on the right.
        let original = vec![
            vec![1.0_f32, 2.0, 3.0, -4.0, -2.0, 0.5, -0.5, 0.0],
            vec![0.5_f32, 1.5, 2.5, -3.5, -1.5, 0.0, -1.0, 0.5],
        ];
        let mut spectra = original.clone();
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 0,
            angle_channel: 1,
        }];
        forward_couple_all(&mut spectra, &coupling).expect("forward couple succeeds");
        inverse_couple_all(&mut spectra, &coupling).expect("inverse couple succeeds");
        assert_eq!(spectra, original);
    }

    #[test]
    fn forward_then_inverse_couple_all_is_identity_multi_step() {
        // Multi-step coupling: a four-channel stream with two
        // independent coupling steps. The encoder runs them in
        // ascending order; the decoder undoes them in descending
        // order. The round-trip identity must hold.
        let original = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![0.5_f32, 1.5, 2.5],
            vec![-1.0_f32, -2.0, -3.0],
            vec![-0.5_f32, -1.5, -2.5],
        ];
        let mut spectra = original.clone();
        let coupling = vec![
            // Step 0: couple channels 0 (mag) and 1 (ang).
            MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            },
            // Step 1: couple channels 2 (mag) and 3 (ang).
            MappingCouplingStep {
                magnitude_channel: 2,
                angle_channel: 3,
            },
        ];
        forward_couple_all(&mut spectra, &coupling).expect("forward couple succeeds");
        inverse_couple_all(&mut spectra, &coupling).expect("inverse couple succeeds");
        assert_eq!(spectra, original);
    }

    #[test]
    fn forward_then_inverse_couple_all_is_identity_with_reversed_channel_order() {
        // The mag > ang branch of forward_couple_all has its own slice
        // ordering. Drive the encoder through that branch and confirm
        // the inverse direction still recovers the original.
        let original = vec![
            vec![5.0_f32, -3.0, 0.0, 1.5],
            vec![2.0_f32, 4.0, -1.0, -0.5],
        ];
        let mut spectra = original.clone();
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 1, // mag > ang exercises the swap branch
            angle_channel: 0,
        }];
        forward_couple_all(&mut spectra, &coupling).expect("forward couple succeeds");
        inverse_couple_all(&mut spectra, &coupling).expect("inverse couple succeeds");
        assert_eq!(spectra, original);
    }

    #[test]
    fn forward_couple_all_leaves_uncoupled_channels_alone() {
        // A 5.1 layout where one mapping submap touches channels {0,
        // 1} and leaves channels {2, 3, 4, 5} alone. The encoder
        // forward-couples (0, 1) and must not perturb the other four.
        let mut spectra = vec![
            vec![5.0_f32],  // ch 0 — coupled (will become M)
            vec![3.0_f32],  // ch 1 — coupled (will become A)
            vec![7.0_f32],  // ch 2 — uncoupled
            vec![9.0_f32],  // ch 3 — uncoupled
            vec![11.0_f32], // ch 4 — uncoupled
            vec![13.0_f32], // ch 5 — uncoupled
        ];
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 0,
            angle_channel: 1,
        }];
        forward_couple_all(&mut spectra, &coupling).unwrap();
        assert_eq!(spectra[0], vec![5.0]);
        assert_eq!(spectra[1], vec![2.0]);
        assert_eq!(spectra[2], vec![7.0]);
        assert_eq!(spectra[3], vec![9.0]);
        assert_eq!(spectra[4], vec![11.0]);
        assert_eq!(spectra[5], vec![13.0]);
    }

    #[test]
    fn forward_couple_all_step_order_matters_for_chained_coupling() {
        // §4.3.5 runs decoder steps in descending order, so the
        // encoder must apply them in ascending order to undo each
        // other end-to-end. Driving two coupling steps that share a
        // channel verifies the order direction.
        //
        // Three channels, two steps:
        //   step 0: mag=0, ang=1
        //   step 1: mag=1, ang=2
        //
        // Encoder runs step 0 first, then step 1 (which now sees the
        // square-polar A from step 0 on channel 1). Decoder runs
        // step 1 first, then step 0 — undoing the encoder's ordering
        // exactly. Identity must still hold.
        let original = vec![vec![5.0_f32, 1.0], vec![3.0_f32, 0.5], vec![-2.0_f32, 4.0]];
        let mut spectra = original.clone();
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
        forward_couple_all(&mut spectra, &coupling).expect("forward couple succeeds");
        inverse_couple_all(&mut spectra, &coupling).expect("inverse couple succeeds");
        assert_eq!(spectra, original);
    }

    // ---- coupling-decision energy heuristic (§4.3.5 encode) ----

    #[test]
    fn coupling_energy_does_not_mutate_inputs() {
        // The measuring routine must leave both channels untouched (it is
        // the non-committing analogue of forward_couple).
        let left = vec![3.0_f32, -1.0, 2.5, 0.0];
        let right = vec![2.0_f32, -1.5, 2.0, 4.0];
        let l0 = left.clone();
        let r0 = right.clone();
        let _ = coupling_energy(&left, &right);
        assert_eq!(left, l0);
        assert_eq!(right, r0);
    }

    #[test]
    fn coupling_energy_matches_committed_forward_couple() {
        // The measured magnitude/angle energies must equal the energies of
        // the vectors a real forward_couple would leave behind.
        let mut left = vec![5.0_f32, -2.0, 3.0, 1.0, -4.0];
        let mut right = vec![4.0_f32, -2.5, 1.0, 1.0, -1.0];
        let pre_uncoupled: f64 = left
            .iter()
            .zip(right.iter())
            .map(|(&l, &r)| f64::from(l) * f64::from(l) + f64::from(r) * f64::from(r))
            .sum();
        let measured = coupling_energy(&left, &right);

        forward_couple(&mut left, &mut right); // left ← M, right ← A
        let m_energy: f64 = left.iter().map(|&m| f64::from(m) * f64::from(m)).sum();
        let a_energy: f64 = right.iter().map(|&a| f64::from(a) * f64::from(a)).sum();

        assert!((measured.magnitude_energy - m_energy).abs() < 1e-9);
        assert!((measured.angle_energy - a_energy).abs() < 1e-9);
        assert!((measured.uncoupled_energy - pre_uncoupled).abs() < 1e-9);
    }

    #[test]
    fn correlated_pair_has_low_angle_ratio_anti_correlated_high() {
        // L == R (perfectly correlated): A = L − R = 0 everywhere → angle
        // ratio 0. L == −R (anti-correlated): the angle carries the full
        // difference → a large ratio.
        let l = vec![3.0_f32, 1.0, -2.0, 4.0];
        let same = l.clone();
        let opposite: Vec<f32> = l.iter().map(|&v| -v).collect();

        let corr = coupling_energy(&l, &same);
        assert_eq!(corr.angle_energy, 0.0);
        assert_eq!(corr.angle_ratio(), 0.0);

        let anti = coupling_energy(&l, &opposite);
        assert!(anti.angle_ratio() > corr.angle_ratio());
        // For L == −R the magnitude is |L| and the angle is 2|L| in
        // magnitude, so the energy ratio is exactly 4.
        assert!((anti.angle_ratio() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn should_couple_gates_on_threshold() {
        let l = vec![3.0_f32, 1.0, -2.0, 4.0];
        let same = l.clone();
        let opposite: Vec<f32> = l.iter().map(|&v| -v).collect();

        // Correlated pair (ratio 0) couples for any non-negative threshold.
        assert!(should_couple(&l, &same, 0.0));
        assert!(should_couple(&l, &same, 1.0));
        // Anti-correlated pair (ratio 4) couples only above the threshold.
        assert!(!should_couple(&l, &opposite, 1.0));
        assert!(!should_couple(&l, &opposite, 3.9));
        assert!(should_couple(&l, &opposite, 4.0));
        assert!(should_couple(&l, &opposite, 10.0));
    }

    #[test]
    fn should_couple_rejects_bad_threshold_and_handles_silence() {
        let l = vec![1.0_f32, 2.0];
        let r = vec![1.0_f32, 2.0];
        // A non-finite or negative gate never couples.
        assert!(!should_couple(&l, &r, -1.0));
        assert!(!should_couple(&l, &r, f64::NAN));
        // INFINITY is non-finite → the gate refuses it.
        assert!(!should_couple(&l, &r, f64::INFINITY));
        // A silent pair has angle ratio 0 and couples (the identity).
        let z = vec![0.0_f32; 4];
        let e = coupling_energy(&z, &z);
        assert_eq!(e.magnitude_energy, 0.0);
        assert_eq!(e.angle_energy, 0.0);
        assert_eq!(e.angle_ratio(), 0.0);
        assert!(should_couple(&z, &z, 0.0));
    }

    #[test]
    fn prune_keeps_correlated_drops_anti_correlated() {
        // 4 channels: (0,1) correlated, (2,3) anti-correlated. Candidate
        // steps couple (0←1) and (2←3); only the first should survive a
        // threshold of 1.0 (corr ratio 0 <= 1; anti ratio 4 > 1).
        let base = vec![3.0_f32, 1.0, -2.0, 4.0];
        let chan0 = base.clone();
        let chan1 = base.clone(); // == chan0 → correlated
        let chan2 = base.clone();
        let chan3: Vec<f32> = base.iter().map(|&v| -v).collect(); // anti
        let channels = vec![chan0, chan1, chan2, chan3];
        let coupling = vec![
            MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            },
            MappingCouplingStep {
                magnitude_channel: 2,
                angle_channel: 3,
            },
        ];
        let kept = prune_coupling_steps(&channels, &coupling, 1.0).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].magnitude_channel, 0);
        assert_eq!(kept[0].angle_channel, 1);
    }

    #[test]
    fn prune_does_not_mutate_input_channels() {
        let channels = vec![vec![3.0_f32, 1.0], vec![3.0_f32, 1.0]];
        let snapshot = channels.clone();
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 0,
            angle_channel: 1,
        }];
        let kept = prune_coupling_steps(&channels, &coupling, 1.0).unwrap();
        assert_eq!(kept.len(), 1); // correlated → kept
        assert_eq!(channels, snapshot); // but input untouched
    }

    #[test]
    fn prune_high_threshold_keeps_all_low_threshold_drops_all() {
        let channels = vec![
            vec![3.0_f32, 1.0, -2.0],
            vec![-3.0_f32, -1.0, 2.0], // anti-correlated, ratio 4
        ];
        let coupling = vec![MappingCouplingStep {
            magnitude_channel: 0,
            angle_channel: 1,
        }];
        // Threshold above 4 keeps the anti-correlated pair.
        assert_eq!(
            prune_coupling_steps(&channels, &coupling, 10.0)
                .unwrap()
                .len(),
            1
        );
        // Threshold below 4 drops it.
        assert_eq!(
            prune_coupling_steps(&channels, &coupling, 1.0)
                .unwrap()
                .len(),
            0
        );
        // A non-finite/negative gate drops everything.
        assert_eq!(
            prune_coupling_steps(&channels, &coupling, -1.0)
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn prune_kept_steps_round_trip_through_forward_then_inverse() {
        // The kept-step list, threaded through forward_couple_all then
        // inverse_couple_all, must reconstruct the original spectra — i.e.
        // pruning yields a self-consistent encode/decode coupling set.
        let original = vec![
            vec![5.0_f32, 1.0, -2.0, 3.0],
            vec![5.0_f32, 1.0, -2.0, 3.0], // == ch0 (correlated)
            vec![2.0_f32, -4.0, 1.0, 0.5],
            vec![-2.0_f32, 4.0, -1.0, -0.5], // anti of ch2
        ];
        let coupling = vec![
            MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            },
            MappingCouplingStep {
                magnitude_channel: 2,
                angle_channel: 3,
            },
        ];
        let kept = prune_coupling_steps(&original, &coupling, 1.0).unwrap();
        let mut spectra = original.clone();
        forward_couple_all(&mut spectra, &kept).expect("forward couple kept steps");
        inverse_couple_all(&mut spectra, &kept).expect("inverse couple kept steps");
        assert_eq!(spectra, original);
    }

    #[test]
    fn prune_rejects_out_of_range_and_same_channel() {
        let channels = vec![vec![1.0_f32, 2.0], vec![1.0_f32, 2.0]];
        let oob = vec![MappingCouplingStep {
            magnitude_channel: 0,
            angle_channel: 5,
        }];
        assert_eq!(
            prune_coupling_steps(&channels, &oob, 1.0),
            Err(CouplingError::ChannelOutOfRange {
                step: 0,
                channel: 5,
                channels: 2,
            })
        );
        let same = vec![MappingCouplingStep {
            magnitude_channel: 1,
            angle_channel: 1,
        }];
        assert_eq!(
            prune_coupling_steps(&channels, &same, 1.0),
            Err(CouplingError::SameChannel {
                step: 0,
                channel: 1,
            })
        );
    }

    // ---- spectrum factoring (§4.3.6 inverse) ----

    use crate::packet::{dot_product, dot_product_all};

    #[test]
    fn factor_spectrum_scalar_divides_for_nonzero_floor() {
        // residue = spectrum / floor for a finite nonzero floor.
        assert_eq!(factor_spectrum_scalar(6.0, 2.0, 0, 0).unwrap(), 3.0);
        assert_eq!(factor_spectrum_scalar(-6.0, 2.0, 0, 0).unwrap(), -3.0);
        assert_eq!(factor_spectrum_scalar(6.0, -2.0, 0, 0).unwrap(), -3.0);
        assert_eq!(factor_spectrum_scalar(0.0, 4.0, 0, 0).unwrap(), 0.0);
    }

    #[test]
    fn factor_spectrum_scalar_zero_floor_zero_spectrum_yields_zero() {
        // floor == 0 with a zero target: residue is unconstrained, emit 0.
        assert_eq!(factor_spectrum_scalar(0.0, 0.0, 0, 0).unwrap(), 0.0);
    }

    #[test]
    fn factor_spectrum_scalar_zero_floor_nonzero_spectrum_rejected() {
        // floor == 0 cannot reproduce a nonzero spectrum.
        assert_eq!(
            factor_spectrum_scalar(1.5, 0.0, 2, 7),
            Err(FactorSpectrumError::NonzeroSpectrumOverZeroFloor {
                channel: 2,
                index: 7,
                spectrum: 1.5,
            })
        );
    }

    #[test]
    fn factor_spectrum_scalar_non_finite_floor_rejected() {
        assert_eq!(
            factor_spectrum_scalar(1.0, f32::NAN, 1, 3),
            Err(FactorSpectrumError::NonFiniteFloor {
                channel: 1,
                index: 3,
            })
        );
        assert_eq!(
            factor_spectrum_scalar(1.0, f32::INFINITY, 0, 0),
            Err(FactorSpectrumError::NonFiniteFloor {
                channel: 0,
                index: 0,
            })
        );
    }

    #[test]
    fn factor_spectrum_round_trips_through_dot_product() {
        // The core property: factor a spectrum given a floor, then run
        // the decode-side §4.3.6 dot product back over (floor, residue)
        // and recover the original spectrum bit-exactly.
        let floor = vec![2.0_f32, 0.5, -4.0, 8.0];
        let spectrum = vec![6.0_f32, -1.0, 12.0, 2.0];
        let mut residue = vec![0.0_f32; spectrum.len()];
        factor_spectrum(&spectrum, &floor, &mut residue).unwrap();
        // residue == spectrum / floor element-wise.
        assert_eq!(residue, vec![3.0_f32, -2.0, -3.0, 0.25]);
        let mut reconstructed = vec![0.0_f32; spectrum.len()];
        dot_product(&floor, &residue, &mut reconstructed);
        assert_eq!(reconstructed, spectrum);
    }

    #[test]
    fn factor_spectrum_round_trips_with_zero_floor_bins() {
        // Zero floor bins must carry a zero target; the recovered
        // residue is 0 there and the dot product still reproduces 0.
        let floor = vec![3.0_f32, 0.0, 5.0, 0.0];
        let spectrum = vec![9.0_f32, 0.0, -10.0, 0.0];
        let mut residue = vec![1.0_f32; spectrum.len()]; // garbage pre-fill
        factor_spectrum(&spectrum, &floor, &mut residue).unwrap();
        assert_eq!(residue, vec![3.0_f32, 0.0, -2.0, 0.0]);
        let mut reconstructed = vec![0.0_f32; spectrum.len()];
        dot_product(&floor, &residue, &mut reconstructed);
        assert_eq!(reconstructed, spectrum);
    }

    #[test]
    fn factor_spectrum_rejects_spectrum_floor_length_mismatch() {
        let spectrum = vec![1.0_f32, 2.0];
        let floor = vec![1.0_f32];
        let mut residue = vec![0.0_f32; 2];
        assert_eq!(
            factor_spectrum(&spectrum, &floor, &mut residue),
            Err(FactorSpectrumError::LengthMismatch {
                spectrum_len: 2,
                floor_len: 1,
            })
        );
    }

    #[test]
    fn factor_spectrum_rejects_residue_length_mismatch() {
        let spectrum = vec![1.0_f32, 2.0];
        let floor = vec![1.0_f32, 1.0];
        let mut residue = vec![0.0_f32; 3];
        assert_eq!(
            factor_spectrum(&spectrum, &floor, &mut residue),
            Err(FactorSpectrumError::LengthMismatch {
                spectrum_len: 2,
                floor_len: 3,
            })
        );
    }

    #[test]
    fn factor_spectrum_all_round_trips_through_dot_product_all() {
        // Multi-channel: factor each channel, then dot_product_all back.
        let floors = vec![Some(vec![2.0_f32, 4.0]), Some(vec![1.0_f32, -2.0])];
        let spectra = vec![vec![6.0_f32, 8.0], vec![5.0_f32, 4.0]];
        let residues = factor_spectrum_all(&spectra, &floors).unwrap();
        assert_eq!(residues, vec![vec![3.0_f32, 2.0], vec![5.0_f32, -2.0]]);
        let reconstructed = dot_product_all(&floors, &residues, 2).unwrap();
        assert_eq!(reconstructed, spectra);
    }

    #[test]
    fn factor_spectrum_all_unused_channel_yields_empty_residue() {
        // A None floor (unused channel) with an all-zero target produces
        // an empty residue and dot_product_all emits the zero spectrum.
        let floors = vec![Some(vec![2.0_f32, 4.0]), None];
        let spectra = vec![vec![6.0_f32, 8.0], vec![0.0_f32, 0.0]];
        let residues = factor_spectrum_all(&spectra, &floors).unwrap();
        assert_eq!(residues, vec![vec![3.0_f32, 2.0], Vec::<f32>::new()]);
        // dot_product_all reads only the used channel's residue; the
        // unused channel emits the all-zero spectrum.
        let used_floors = vec![Some(vec![2.0_f32, 4.0]), None];
        let used_residues = vec![vec![3.0_f32, 2.0], vec![0.0_f32, 0.0]];
        let reconstructed = dot_product_all(&used_floors, &used_residues, 2).unwrap();
        assert_eq!(reconstructed, spectra);
    }

    #[test]
    fn factor_spectrum_all_unused_channel_nonzero_target_rejected() {
        // An unused channel whose target spectrum is not all-zero is
        // inconsistent (the decoder emits zero for it regardless).
        let floors = vec![None];
        let spectra = vec![vec![0.0_f32, 0.0, 2.5]];
        assert_eq!(
            factor_spectrum_all(&spectra, &floors),
            Err(FactorSpectrumError::NonzeroSpectrumOverZeroFloor {
                channel: 0,
                index: 2,
                spectrum: 2.5,
            })
        );
    }

    #[test]
    fn factor_spectrum_all_rejects_channel_count_mismatch() {
        let spectra = vec![vec![1.0_f32], vec![2.0_f32]];
        let floors = vec![Some(vec![1.0_f32])];
        assert_eq!(
            factor_spectrum_all(&spectra, &floors),
            Err(FactorSpectrumError::ChannelCountMismatch {
                spectra: 2,
                floors: 1,
            })
        );
    }

    #[test]
    fn factor_spectrum_all_rejects_floor_spectrum_length_mismatch() {
        let spectra = vec![vec![1.0_f32, 2.0]];
        let floors = vec![Some(vec![1.0_f32])];
        assert_eq!(
            factor_spectrum_all(&spectra, &floors),
            Err(FactorSpectrumError::LengthMismatch {
                spectrum_len: 2,
                floor_len: 1,
            })
        );
    }

    #[test]
    fn factor_spectrum_all_propagates_channel_coordinate_on_error() {
        // The per-channel index must be carried into the error from the
        // second channel, not reported as 0.
        let floors = vec![Some(vec![2.0_f32]), Some(vec![0.0_f32])];
        let spectra = vec![vec![4.0_f32], vec![1.0_f32]];
        assert_eq!(
            factor_spectrum_all(&spectra, &floors),
            Err(FactorSpectrumError::NonzeroSpectrumOverZeroFloor {
                channel: 1,
                index: 0,
                spectrum: 1.0,
            })
        );
    }

    #[test]
    fn factor_spectrum_error_display_is_grep_friendly() {
        // Every variant's Display mentions the §4.3.6 inverse and the
        // offending values, so a log grep can find them.
        let variants = [
            FactorSpectrumError::LengthMismatch {
                spectrum_len: 4,
                floor_len: 5,
            },
            FactorSpectrumError::ChannelCountMismatch {
                spectra: 2,
                floors: 1,
            },
            FactorSpectrumError::NonFiniteFloor {
                channel: 1,
                index: 3,
            },
            FactorSpectrumError::NonzeroSpectrumOverZeroFloor {
                channel: 2,
                index: 7,
                spectrum: 1.5,
            },
        ];
        for v in &variants {
            let s = v.to_string();
            assert!(
                s.contains("§4.3.6 inverse"),
                "Display should cite §4.3.6 inverse, got: {s}"
            );
            assert!(!s.is_empty());
        }
        // source() is None for all variants (no wrapped error).
        use std::error::Error as _;
        for v in &variants {
            assert!(v.source().is_none());
        }
    }
}

//! Encoder quality targeting: the quality knob → tuning-parameter map
//! and the lambda-for-bit-budget solver.
//!
//! The Vorbis I specification defines only the decode side; how an
//! encoder exposes its quality/rate trade is unspecified territory.
//! The crate's encode stack has grown a set of independent levers —
//! the residue rate-distortion `lambda`
//! ([`crate::residue_encode::plan_vector_residue_rd_weighted`]), the
//! psychoacoustic margin
//! ([`crate::psy::PsyConfig::threshold_offset_db`]), and the floor-1
//! post budget ([`crate::floor1_layout::design_floor1_header`]) — that
//! all move the same rate/fidelity trade from different directions.
//! This module ties them to **one scalar**:
//!
//! * [`EncoderTuning::from_quality`] maps a quality setting
//!   `q ∈ [0, 1]` to a coherent lever set: `lambda` falls
//!   log-linearly with `q` (each step of `q` buys a constant-ratio
//!   drop in the bits→audibility exchange rate), the masking margin
//!   rises linearly (−7.2 dB at the low-bitrate knee `q = 0.2`,
//!   capped at +6 dB), and the floor post budget grows with the
//!   fidelity the residue will carry. Below the **low-bitrate knee**
//!   ([`LOW_BITRATE_KNEE`]) the levers steepen into a genuine
//!   low-rate mode: `lambda` climbs a further
//!   [`LOW_BITRATE_LAMBDA_DECADES`] decades to `q = 0`, the noise-like
//!   maskers' thresholds rise by [`LOW_BITRATE_NOISE_MARGIN_DB`] while
//!   the tonal maskers' fall by [`LOW_BITRATE_TONAL_MARGIN_DB`] (bits
//!   move from hiss to partials), and the coded band is limited
//!   ([`EncoderTuning::coded_bandwidth_hz`]) toward
//!   [`LOW_BITRATE_MIN_BANDWIDTH_HZ`]. Monotone by construction: a
//!   higher `q` never spends fewer bits or raises the modelled
//!   audible noise.
//! * [`solve_lambda_for_bits`] inverts the rate side: given a bit
//!   budget and any caller-supplied `rate(lambda)` measurement (the
//!   rate of a residue plan, a whole packet, or a whole stream), it
//!   bisects the monotone non-increasing rate–lambda curve to the
//!   cheapest `lambda` that fits the budget. This is the ABR/CBR-side
//!   entry: quality targeting picks `lambda` from `q`; bit targeting
//!   picks `lambda` from the budget.

/// The quality setting below which the knob is a **low-bitrate
/// mode**: above it the lever laws are the r453 calibration (each
/// point of the knob on its own step of the rate/fidelity frontier);
/// below it they steepen so `q = 0` reaches the reference encoder's
/// lowest operating region instead of stopping at ~130 kbps on a
/// tones + hiss corpus (measured: at the old `q = 0` the weighted
/// chooser still coded every partition the model called audible —
/// `lambda = 10⁻¹·⁴` prices a partition's worth of noise-to-mask
/// error at hundreds of bits, so the hiss under a tone bed at −45 dBFS
/// cost 3–4 bits per bin).
pub const LOW_BITRATE_KNEE: f32 = 0.2;

/// How many further decades `lambda` climbs from the knee to `q = 0`
/// (`lambda(0) = lambda(knee) · 10^decades`): at 2 decades the
/// exchange rate at `q = 0` is ~1.2 audibility units per bit, where a
/// near-threshold partition's silence error no longer buys its
/// codewords (measured on the tones + hiss battery: `q = 0` lands at
/// ~24 kbps stereo — the reference encoder's lowest operating
/// region — where 2.5 decades dropped the tonal partitions too and
/// left 13 kbps at single-digit SNR).
pub const LOW_BITRATE_LAMBDA_DECADES: f64 = 2.0;

/// The extra threshold raise applied to **noise-like maskers only**
/// at `q = 0` ([`crate::psy::PsyConfig::noise_margin_db`], linear from
/// `0` at the knee): the low-rate mode drops noise-masked texture
/// (hiss under a tone bed) before it loosens the protection around
/// tonal peaks. The uniform margin (`threshold_offset_db`) stays at
/// its knee value through the low-rate mode: measured, deepening it
/// uniformly (−12 dB more at `q = 0`) let the reconstruction noise
/// around strong partials climb to within a couple of dB of the
/// partial — the tonal offset is only `14.5 + z` dB — and the tones
/// were dropped before the hiss was. 24 dB is where the hiss under a
/// tone bed (its peak-held floor sits ~5 dB over its own level)
/// drops out of the rate-distortion trade at the low-rate `lambda`;
/// 12 dB still spent 40 kbps on it.
pub const LOW_BITRATE_NOISE_MARGIN_DB: f32 = 24.0;

/// The extra threshold **lowering** applied to tonal maskers at
/// `q = 0` ([`crate::psy::PsyConfig::tonal_margin_db`], linear from
/// `0` at the knee): the complement of the noise margin — the bits
/// the low-rate mode can afford go to the partials. Measured on the
/// tones + hiss battery at `q = 0.1`: 17.6 → 24.2 dB at 150 kbps for
/// 12 dB (6 dB gave 23.3 dB); on the staged fixtures +3 dB at equal
/// rate.
pub const LOW_BITRATE_TONAL_MARGIN_DB: f32 = 12.0;

/// The coded bandwidth the low-rate mode limits to at `q = 0`
/// (log-linear from the full band at the knee): §8.6.1 `residue_end`
/// is capped there and the floor is designed / fitted over the coded
/// band only, so the bits saved above the cutoff go to the audible
/// band instead of the model's rate-distortion trade spending a
/// classword per silent partition up there.
pub const LOW_BITRATE_MIN_BANDWIDTH_HZ: f32 = 8_000.0;

/// The full-band limit the knee (and everything above it) codes to.
const FULL_BANDWIDTH_HZ: f32 = 20_000.0;

/// The coherent lever set one quality setting expands to.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderTuning {
    /// The rate-distortion Lagrange multiplier for the residue
    /// choosers. In the perceptually weighted chooser the distortion
    /// term is on the noise-to-mask scale, so `lambda` prices bits in
    /// audibility units: `10⁻¹·⁹²` at the low-bitrate knee `q = 0.2`
    /// down to `10⁻⁴` at `q = 1`, log-linear in between, and a
    /// further [`LOW_BITRATE_LAMBDA_DECADES`] decades up at `q = 0`
    /// ([`EncoderTuning::lambda_for_quality`]). (The law was recalibrated for
    /// the four-class residue ladder: under the old `10⁰ → 10⁻⁴` law
    /// the intermediate classes made the whole low half of the knob
    /// collapse onto near-identical cheap plans and the `q ≈ 0.75`
    /// step a cliff; 2.6 decades spread over the knob place each
    /// measured rate point on its own step of the frontier.)
    pub lambda: f64,
    /// The masking-margin lever for
    /// [`crate::psy::PsyConfig::threshold_offset_db`]: −7.2 dB at and
    /// below the low-bitrate knee `q = 0.2`, rising linearly and
    /// **capped at +6 dB** (reached at `q = 0.75`). The cap is measured, not
    /// aesthetic: the psy floor envelope rides
    /// `max(peak-held |X|, threshold)`, so pushing the threshold ever
    /// lower drags the floor onto `|X|` in every quiet bin — the
    /// residue targets `X/floor` then approach full scale across the
    /// noise floor, which the ladders sized by the loud partitions
    /// quantise poorly. Beyond +6 dB the measured whole-stream SNR
    /// stops rising (and at +12 dB falls) while bytes keep climbing;
    /// the old uncapped law was the second cause of the non-monotone
    /// SNR above `q ≈ 0.7`.
    pub threshold_offset_db: f32,
    /// The floor-1 explicit-post budget for
    /// [`crate::floor1_layout::design_floor1_header`]: 8 posts at
    /// `q = 0` rising to 32 at `q = 1` (a finer envelope is only
    /// worth carrying when the residue will preserve the detail).
    pub floor_post_budget: usize,
    /// The peak-hold smoothing radius for
    /// [`crate::psy::plan_psy_floor_envelope`] (constant `2`: the
    /// guard against inter-post floor dips is quality-independent).
    pub floor_smooth_radius: usize,
    /// The **content-adaptive margin headroom** (dB): the extra
    /// masking margin the whole-stream encoder may grant *per channel*
    /// on top of [`Self::threshold_offset_db`] in its measured balance
    /// pass (`encode_pcm_to_packets`): the first pass is own-decoded,
    /// and a channel whose measured SNR trails the best channel by
    /// more than 3 dB earns a deficit-scaled share of this headroom
    /// for one retry — waveform coding exactly where the stream shows
    /// it pays, without dragging every channel's rate along the way
    /// the old uncapped global margin did. (The grant is measured, not
    /// inferred: no cheap psy statistic identifies the trailing
    /// channel — on the decorrelated stereo fixture the channels carry
    /// identical RMS and near-identical over-masked-energy and
    /// tonality figures while coding 12 dB apart.) `0` through
    /// `q ≤ 0.75` — the encode below the cap knee stays single-pass
    /// and byte-identical — rising linearly to `+6 dB` at `q = 1`
    /// (the old uncapped law's top, now spent selectively). Monotone
    /// non-decreasing in `q`.
    pub adaptive_margin_headroom_db: f32,
    /// The **fine value-ladder divisor**: the integrated encoder's
    /// second-stage residue book quantises with step
    /// `max_abs / fine_step_divisor`, so this divisor sets the
    /// encoder's reconstruction **noise floor** — the SNR ceiling no
    /// amount of extra rate can pass. It is `192` through the low and
    /// middle of the quality range and rises fourfold toward `768` at
    /// `q = 1` (log-linear above `q = 0.7`), giving the top of the
    /// knob genuine SNR headroom: with a fixed divisor the whole-
    /// stream SNR *saturates* near `q ≈ 0.7` (the residue error is
    /// pinned at the ladder step) and the further rate the falling
    /// `lambda` buys only densifies class choices — measured SNR then
    /// wobbles non-monotonically around the fixed ceiling while bytes
    /// triple. Monotone non-decreasing in `q`.
    pub fine_step_divisor: f32,
    /// The **coded bandwidth** (Hz): `None` codes the whole spectrum
    /// (the knee and above); the low-rate mode limits §8.6.1
    /// `residue_end` (and the floor design/fit band) to this
    /// frequency, falling log-linearly from 20 kHz at the knee to
    /// [`LOW_BITRATE_MIN_BANDWIDTH_HZ`] at `q = 0`. Monotone
    /// non-decreasing in `q`.
    pub coded_bandwidth_hz: Option<f32>,
    /// The **noise-masker margin** for
    /// [`crate::psy::PsyConfig::noise_margin_db`]: `0` at the knee and
    /// above, rising linearly to [`LOW_BITRATE_NOISE_MARGIN_DB`] at
    /// `q = 0`. Monotone non-increasing in `q`.
    pub noise_margin_db: f32,
    /// The **tonal-masker margin** for
    /// [`crate::psy::PsyConfig::tonal_margin_db`]: `0` at the knee and
    /// above, rising linearly to [`LOW_BITRATE_TONAL_MARGIN_DB`] at
    /// `q = 0`. Monotone non-increasing in `q`.
    pub tonal_margin_db: f32,
}

/// Errors from the quality → tuning map.
#[derive(Debug, Clone, PartialEq)]
pub enum QualityError {
    /// The quality setting was NaN or outside `[0, 1]`.
    QualityOutOfRange(f32),
}

impl core::fmt::Display for QualityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QualityError::QualityOutOfRange(q) => {
                write!(f, "vorbis quality: setting {q} outside [0, 1]")
            }
        }
    }
}

impl std::error::Error for QualityError {}

impl EncoderTuning {
    /// The [`Self::fine_step_divisor`] law's base value — the divisor
    /// through the low and middle of the quality range (`q <= 0.7`).
    /// [`Self::fine_resolution_scale`] is the divisor normalised to
    /// this base.
    pub const FINE_STEP_DIVISOR_BASE: f32 = 192.0;

    /// The quality knob's fine-ladder **resolution scale**: `1.0`
    /// through `q <= 0.7`, rising log-linearly to `4.0` at `q = 1`
    /// (the [`Self::fine_step_divisor`] law normalised to its base).
    /// A residue fine ladder of *any* geometry divides its step by
    /// this scale so the top of the knob lowers the encoder's
    /// reconstruction noise floor instead of buying saturated-SNR
    /// density — the r413 monotone-knob fix, applicable to the
    /// designed multi-dimensional lattice ladders as well as the
    /// scalar seed ladder.
    #[must_use]
    pub fn fine_resolution_scale(&self) -> f32 {
        self.fine_step_divisor / Self::FINE_STEP_DIVISOR_BASE
    }

    /// Expand a quality setting `q ∈ [0, 1]` into the lever set (see
    /// the struct fields for each lever's law). Monotone: `lambda` is
    /// strictly decreasing in `q`; `threshold_offset_db` and
    /// `floor_post_budget` are non-decreasing.
    ///
    /// # Errors
    ///
    /// [`QualityError::QualityOutOfRange`] for NaN or `q ∉ [0, 1]`.
    pub fn from_quality(q: f32) -> Result<Self, QualityError> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(QualityError::QualityOutOfRange(q));
        }
        let qf = f64::from(q);
        // Depth into the low-bitrate mode: 0 at and above the knee,
        // 1 at q = 0.
        let low = ((LOW_BITRATE_KNEE - q) / LOW_BITRATE_KNEE).clamp(0.0, 1.0);
        Ok(EncoderTuning {
            lambda: Self::lambda_for_quality(q),
            threshold_offset_db: (-12.0 + 24.0 * q.max(LOW_BITRATE_KNEE)).min(6.0),
            adaptive_margin_headroom_db: ((q - 0.75) * 24.0).clamp(0.0, 6.0),
            floor_post_budget: 8 + (24.0 * qf).round() as usize,
            floor_smooth_radius: 2,
            fine_step_divisor: 192.0 * 4f32.powf(((q - 0.7) / 0.3).max(0.0)),
            coded_bandwidth_hz: (low > 0.0).then(|| {
                FULL_BANDWIDTH_HZ * (LOW_BITRATE_MIN_BANDWIDTH_HZ / FULL_BANDWIDTH_HZ).powf(low)
            }),
            noise_margin_db: LOW_BITRATE_NOISE_MARGIN_DB * low,
            tonal_margin_db: LOW_BITRATE_TONAL_MARGIN_DB * low,
        })
    }

    /// The residue `lambda` law: `10^(−1.4 − 2.6·q)` from the knee up
    /// (the r453 calibration), climbing a further
    /// [`LOW_BITRATE_LAMBDA_DECADES`] decades linearly in `q` below
    /// it. Strictly decreasing in `q`; [`Self::quality_for_lambda`]
    /// is its inverse.
    #[must_use]
    pub fn lambda_for_quality(q: f32) -> f64 {
        let q = f64::from(q.clamp(0.0, 1.0));
        let knee = f64::from(LOW_BITRATE_KNEE);
        let base = 10f64.powf(-1.4 - 2.6 * q.max(knee));
        if q < knee {
            base * 10f64.powf(LOW_BITRATE_LAMBDA_DECADES * (knee - q) / knee)
        } else {
            base
        }
    }

    /// The inverse of [`Self::lambda_for_quality`], clamped to
    /// `[0, 1]`.
    #[must_use]
    pub fn quality_for_lambda(lambda: f64) -> f32 {
        let knee = f64::from(LOW_BITRATE_KNEE);
        let at_knee = 10f64.powf(-1.4 - 2.6 * knee);
        let q = if lambda <= at_knee {
            (-1.4 - lambda.log10()) / 2.6
        } else {
            knee - knee * (lambda / at_knee).log10() / LOW_BITRATE_LAMBDA_DECADES
        };
        q.clamp(0.0, 1.0) as f32
    }
}

/// The result of [`solve_lambda_for_bits`].
#[derive(Debug, Clone, PartialEq)]
pub struct LambdaSolution {
    /// The chosen Lagrange multiplier.
    pub lambda: f64,
    /// The measured rate at [`Self::lambda`].
    pub bits: u64,
    /// `true` when `bits <= target_bits`. `false` only when even the
    /// cheapest end of the search range (`lambda_hi`) exceeds the
    /// budget — the returned point is then that cheapest end, the
    /// best the range offers.
    pub within_budget: bool,
}

/// Errors from [`solve_lambda_for_bits`]. `E` is the caller's rate
/// -measurement error type, carried verbatim.
#[derive(Debug, Clone, PartialEq)]
pub enum LambdaSolveError<E> {
    /// The search range was empty, non-finite, or negative
    /// (`0 <= lambda_lo < lambda_hi` is required).
    BadRange {
        /// The supplied low (expensive, high-rate) end.
        lo: f64,
        /// The supplied high (cheap, low-rate) end.
        hi: f64,
    },
    /// `max_iterations` was zero.
    ZeroIterations,
    /// The caller's rate measurement failed at some probe.
    Rate(E),
}

impl<E: core::fmt::Display> core::fmt::Display for LambdaSolveError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LambdaSolveError::BadRange { lo, hi } => write!(
                f,
                "vorbis quality: lambda range [{lo}, {hi}] is not a valid search bracket"
            ),
            LambdaSolveError::ZeroIterations => {
                write!(f, "vorbis quality: max_iterations is zero")
            }
            LambdaSolveError::Rate(e) => write!(f, "vorbis quality: rate measurement failed: {e}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for LambdaSolveError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LambdaSolveError::Rate(e) => Some(e),
            _ => None,
        }
    }
}

/// Find the cheapest `lambda` in `[lambda_lo, lambda_hi]` whose
/// measured rate fits `target_bits`, by bisection over the monotone
/// non-increasing rate–lambda curve.
///
/// `rate(lambda)` is any caller-supplied measurement — the value bits
/// of one [`crate::residue_encode::plan_vector_residue_rd`] plan, a
/// whole packet's serialised size in bits, or a whole stream's. The
/// Lagrangian planners spend monotonically fewer bits as `lambda`
/// rises, so the curve is a non-increasing step function of `lambda`
/// and bisection is exact up to step resolution:
///
/// * if `rate(lambda_lo) <= target_bits`, the budget is loose — the
///   highest-fidelity end is returned immediately;
/// * if `rate(lambda_hi) > target_bits`, the budget is unreachable in
///   the bracket — the cheapest end is returned with
///   [`LambdaSolution::within_budget`] `= false`;
/// * otherwise bisect: the returned point is the **lowest-lambda probe
///   observed within budget** (ties inherent in the step curve resolve
///   toward fidelity), after `max_iterations` halvings or an exact
///   `bits == target_bits` hit, whichever first.
///
/// The chosen `lambda` is always one the curve was actually measured
/// at, and the reported [`LambdaSolution::bits`] is its measurement —
/// no interpolation is invented.
///
/// # Errors
///
/// [`LambdaSolveError::BadRange`] / [`LambdaSolveError::ZeroIterations`]
/// for a malformed search, [`LambdaSolveError::Rate`] carrying the
/// caller's error if a probe fails.
pub fn solve_lambda_for_bits<F, E>(
    target_bits: u64,
    lambda_lo: f64,
    lambda_hi: f64,
    max_iterations: usize,
    mut rate: F,
) -> Result<LambdaSolution, LambdaSolveError<E>>
where
    F: FnMut(f64) -> Result<u64, E>,
{
    if !lambda_lo.is_finite() || !lambda_hi.is_finite() || lambda_lo < 0.0 || lambda_lo >= lambda_hi
    {
        return Err(LambdaSolveError::BadRange {
            lo: lambda_lo,
            hi: lambda_hi,
        });
    }
    if max_iterations == 0 {
        return Err(LambdaSolveError::ZeroIterations);
    }

    let bits_lo = rate(lambda_lo).map_err(LambdaSolveError::Rate)?;
    if bits_lo <= target_bits {
        return Ok(LambdaSolution {
            lambda: lambda_lo,
            bits: bits_lo,
            within_budget: true,
        });
    }
    let bits_hi = rate(lambda_hi).map_err(LambdaSolveError::Rate)?;
    if bits_hi > target_bits {
        return Ok(LambdaSolution {
            lambda: lambda_hi,
            bits: bits_hi,
            within_budget: false,
        });
    }

    // Invariant: rate(lo) > target >= rate(hi). The answer is the
    // smallest lambda whose rate fits; `best` tracks the fitting probe
    // with the lowest lambda seen so far (initially the hi end).
    let mut lo = lambda_lo;
    let mut hi = lambda_hi;
    let mut best = LambdaSolution {
        lambda: lambda_hi,
        bits: bits_hi,
        within_budget: true,
    };
    for _ in 0..max_iterations {
        let mid = 0.5 * (lo + hi);
        let bits = rate(mid).map_err(LambdaSolveError::Rate)?;
        if bits <= target_bits {
            best = LambdaSolution {
                lambda: mid,
                bits,
                within_budget: true,
            };
            hi = mid;
            if bits == target_bits {
                break;
            }
        } else {
            lo = mid;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- quality → tuning ----------

    #[test]
    fn tuning_rejects_out_of_range_quality() {
        for q in [-0.01f32, 1.01, f32::NAN, f32::INFINITY] {
            match EncoderTuning::from_quality(q) {
                Err(QualityError::QualityOutOfRange(_)) => {}
                other => panic!("q = {q}: expected QualityOutOfRange, got {other:?}"),
            }
        }
    }

    #[test]
    fn tuning_endpoints_are_pinned() {
        let lo = EncoderTuning::from_quality(0.0).unwrap();
        let knee = EncoderTuning::from_quality(LOW_BITRATE_KNEE).unwrap();
        let hi = EncoderTuning::from_quality(1.0).unwrap();
        // The knee carries the r453 calibration; q = 0 sits
        // LOW_BITRATE_LAMBDA_DECADES above it.
        let expect_knee = 10f64.powf(-1.4 - 2.6 * f64::from(LOW_BITRATE_KNEE));
        assert!((knee.lambda / expect_knee - 1.0).abs() < 1e-9);
        assert!(
            (lo.lambda / knee.lambda / 10f64.powf(LOW_BITRATE_LAMBDA_DECADES) - 1.0).abs() < 1e-9
        );
        assert!((hi.lambda - 1e-4).abs() < 1e-12);
        assert!((knee.threshold_offset_db - (-7.2)).abs() < 1e-5);
        assert!((lo.threshold_offset_db - (-7.2)).abs() < 1e-5);
        assert_eq!(knee.noise_margin_db, 0.0);
        assert_eq!(knee.tonal_margin_db, 0.0);
        assert_eq!(lo.noise_margin_db, LOW_BITRATE_NOISE_MARGIN_DB);
        assert_eq!(lo.tonal_margin_db, LOW_BITRATE_TONAL_MARGIN_DB);
        assert_eq!(hi.threshold_offset_db, 6.0);
        assert_eq!(knee.coded_bandwidth_hz, None);
        assert_eq!(hi.coded_bandwidth_hz, None);
        assert!((lo.coded_bandwidth_hz.unwrap() - LOW_BITRATE_MIN_BANDWIDTH_HZ).abs() < 1.0);
        let mid_low = EncoderTuning::from_quality(0.1).unwrap();
        let bw = mid_low.coded_bandwidth_hz.unwrap();
        assert!(
            bw > LOW_BITRATE_MIN_BANDWIDTH_HZ && bw < 20_000.0,
            "bw {bw}"
        );
        assert_eq!(lo.floor_post_budget, 8);
        assert_eq!(hi.floor_post_budget, 32);
        assert_eq!(lo.adaptive_margin_headroom_db, 0.0);
        assert_eq!(hi.adaptive_margin_headroom_db, 6.0);
        assert_eq!(lo.floor_smooth_radius, 2);
        assert_eq!(lo.fine_step_divisor, 192.0);
        let mid = EncoderTuning::from_quality(0.7).unwrap();
        assert_eq!(mid.fine_step_divisor, 192.0);
        // The adaptive headroom stays zero through the cap knee, so
        // the whole-stream encode at and below the default quality is
        // untouched by the per-channel lever.
        assert_eq!(mid.adaptive_margin_headroom_db, 0.0);
        assert_eq!(
            EncoderTuning::from_quality(0.75)
                .unwrap()
                .adaptive_margin_headroom_db,
            0.0
        );
        assert!((hi.fine_step_divisor - 768.0).abs() < 1e-3);
    }

    #[test]
    fn tuning_is_monotone_in_quality() {
        let mut prev: Option<EncoderTuning> = None;
        for i in 0..=20 {
            let t = EncoderTuning::from_quality(i as f32 / 20.0).unwrap();
            if let Some(p) = &prev {
                assert!(t.lambda < p.lambda, "lambda strictly falls with q");
                assert!(
                    t.threshold_offset_db >= p.threshold_offset_db,
                    "margin never falls"
                );
                assert!(
                    t.floor_post_budget >= p.floor_post_budget,
                    "post budget never falls"
                );
                assert!(
                    t.fine_step_divisor >= p.fine_step_divisor,
                    "fine ladder resolution never falls"
                );
                assert!(
                    t.adaptive_margin_headroom_db >= p.adaptive_margin_headroom_db,
                    "adaptive margin headroom never falls"
                );
            }
            if let Some(p) = prev.as_ref() {
                assert!(
                    t.coded_bandwidth_hz.unwrap_or(f32::INFINITY)
                        >= p.coded_bandwidth_hz.unwrap_or(f32::INFINITY),
                    "coded bandwidth never falls"
                );
                assert!(
                    t.noise_margin_db <= p.noise_margin_db,
                    "noise-masker margin never rises with q"
                );
                assert!(
                    t.tonal_margin_db <= p.tonal_margin_db,
                    "tonal-masker margin never rises with q"
                );
            }
            prev = Some(t);
        }
    }

    #[test]
    fn lambda_law_inverts_across_the_knee() {
        for i in 0..=40 {
            let q = i as f32 / 40.0;
            let lambda = EncoderTuning::lambda_for_quality(q);
            let back = EncoderTuning::quality_for_lambda(lambda);
            assert!(
                (back - q).abs() < 1e-4,
                "q {q} -> lambda {lambda} -> {back}"
            );
            assert_eq!(EncoderTuning::from_quality(q).unwrap().lambda, lambda);
        }
        assert_eq!(EncoderTuning::quality_for_lambda(1e9), 0.0);
        assert_eq!(EncoderTuning::quality_for_lambda(1e-9), 1.0);
    }

    // ---------- lambda-for-bits bisection ----------

    /// A synthetic monotone non-increasing step curve.
    fn synth_rate(lambda: f64) -> Result<u64, core::convert::Infallible> {
        Ok((1000.0 / (1.0 + 20.0 * lambda)) as u64)
    }

    #[test]
    fn solver_rejects_bad_brackets_and_zero_iterations() {
        assert_eq!(
            solve_lambda_for_bits(100, 1.0, 0.5, 10, synth_rate),
            Err(LambdaSolveError::BadRange { lo: 1.0, hi: 0.5 })
        );
        assert_eq!(
            solve_lambda_for_bits(100, -0.5, 1.0, 10, synth_rate),
            Err(LambdaSolveError::BadRange { lo: -0.5, hi: 1.0 })
        );
        assert_eq!(
            solve_lambda_for_bits(100, 0.0, 1.0, 0, synth_rate),
            Err(LambdaSolveError::ZeroIterations)
        );
    }

    #[test]
    fn loose_budget_returns_the_fidelity_end() {
        // rate(0) = 1000; a 2000-bit budget is loose.
        let s = solve_lambda_for_bits(2000, 0.0, 10.0, 20, synth_rate).unwrap();
        assert_eq!(s.lambda, 0.0);
        assert_eq!(s.bits, 1000);
        assert!(s.within_budget);
    }

    #[test]
    fn unreachable_budget_returns_the_cheap_end_flagged() {
        // rate(10) = 1000/201 = 4; a 2-bit budget is unreachable.
        let s = solve_lambda_for_bits(2, 0.0, 10.0, 20, synth_rate).unwrap();
        assert_eq!(s.lambda, 10.0);
        assert!(!s.within_budget);
        assert!(s.bits > 2);
    }

    #[test]
    fn bisection_lands_within_budget_near_the_target() {
        let target = 500u64;
        let s = solve_lambda_for_bits(target, 0.0, 10.0, 40, synth_rate).unwrap();
        assert!(s.within_budget);
        assert!(s.bits <= target, "fits the budget: {} <= {target}", s.bits);
        // The curve step near 500 bits is fine-grained; 40 halvings of
        // [0, 10] pin the answer to well within 2% of the budget.
        assert!(
            s.bits >= 490,
            "lands close under the budget: {} vs {target}",
            s.bits
        );
        // The reported bits are the actual measurement at the lambda.
        assert_eq!(s.bits, synth_rate(s.lambda).unwrap());
    }

    #[test]
    fn solver_result_is_monotone_in_the_budget() {
        // A bigger budget never gets a bigger lambda (never less
        // fidelity).
        let mut prev_lambda = f64::INFINITY;
        for target in [100u64, 250, 500, 750, 990] {
            let s = solve_lambda_for_bits(target, 0.0, 10.0, 40, synth_rate).unwrap();
            assert!(s.within_budget);
            assert!(
                s.lambda <= prev_lambda,
                "budget {target}: lambda {} must not exceed {prev_lambda}",
                s.lambda
            );
            prev_lambda = s.lambda;
        }
    }

    #[test]
    fn solver_propagates_rate_errors() {
        #[derive(Debug, Clone, PartialEq)]
        struct Boom;
        let r = solve_lambda_for_bits(10, 0.0, 1.0, 5, |_| Err::<u64, Boom>(Boom));
        assert_eq!(r, Err(LambdaSolveError::Rate(Boom)));
    }
}

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
//!   rises linearly (−12 dB at `q = 0`, +12 dB at `q = 1`), and the
//!   floor post budget grows with the fidelity the residue will
//!   carry. Monotone by construction: a higher `q` never spends fewer
//!   bits or raises the modelled audible noise.
//! * [`solve_lambda_for_bits`] inverts the rate side: given a bit
//!   budget and any caller-supplied `rate(lambda)` measurement (the
//!   rate of a residue plan, a whole packet, or a whole stream), it
//!   bisects the monotone non-increasing rate–lambda curve to the
//!   cheapest `lambda` that fits the budget. This is the ABR/CBR-side
//!   entry: quality targeting picks `lambda` from `q`; bit targeting
//!   picks `lambda` from the budget.

/// The coherent lever set one quality setting expands to.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderTuning {
    /// The rate-distortion Lagrange multiplier for the residue
    /// choosers. In the perceptually weighted chooser the distortion
    /// term is on the noise-to-mask scale, so `lambda` prices bits in
    /// audibility units: `10⁻¹·⁴` at `q = 0` down to `10⁻⁴` at
    /// `q = 1`, log-linear in between. (The law was recalibrated for
    /// the four-class residue ladder: under the old `10⁰ → 10⁻⁴` law
    /// the intermediate classes made the whole low half of the knob
    /// collapse onto near-identical cheap plans and the `q ≈ 0.75`
    /// step a cliff; 2.6 decades spread over the knob place each
    /// measured rate point on its own step of the frontier.)
    pub lambda: f64,
    /// The masking-margin lever for
    /// [`crate::psy::PsyConfig::threshold_offset_db`]: −12 dB (coarse,
    /// aggressive masking) at `q = 0`, +12 dB (strict) at `q = 1`.
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
        Ok(EncoderTuning {
            // 10^-1.4 at q=0 → 10^-4 at q=1.
            lambda: 10f64.powf(-1.4 - 2.6 * qf),
            threshold_offset_db: -12.0 + 24.0 * q,
            floor_post_budget: 8 + (24.0 * qf).round() as usize,
            floor_smooth_radius: 2,
            // 192 up to q = 0.7, then log-linear to 4 × 192 = 768 at
            // q = 1 — the top of the knob lowers the ladder noise
            // floor instead of buying more saturated-SNR density.
            fine_step_divisor: 192.0 * 4f32.powf(((q - 0.7) / 0.3).max(0.0)),
        })
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
///-measurement error type, carried verbatim.
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
        let hi = EncoderTuning::from_quality(1.0).unwrap();
        assert!((lo.lambda - 10f64.powf(-1.4)).abs() < 1e-12);
        assert!((hi.lambda - 1e-4).abs() < 1e-12);
        assert_eq!(lo.threshold_offset_db, -12.0);
        assert_eq!(hi.threshold_offset_db, 12.0);
        assert_eq!(lo.floor_post_budget, 8);
        assert_eq!(hi.floor_post_budget, 32);
        assert_eq!(lo.floor_smooth_radius, 2);
        assert_eq!(lo.fine_step_divisor, 192.0);
        let mid = EncoderTuning::from_quality(0.7).unwrap();
        assert_eq!(mid.fine_step_divisor, 192.0);
        assert!((hi.fine_step_divisor - 768.0).abs() < 1e-3);
    }

    #[test]
    fn tuning_is_monotone_in_quality() {
        let mut prev: Option<EncoderTuning> = None;
        for i in 0..=20 {
            let t = EncoderTuning::from_quality(i as f32 / 20.0).unwrap();
            if let Some(p) = prev {
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
            }
            prev = Some(t);
        }
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

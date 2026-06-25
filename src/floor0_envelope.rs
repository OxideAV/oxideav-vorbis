//! Floor 0 envelope-fit glue (Vorbis I §6.2.3, encode direction).
//!
//! The crate already carries the floor-0 VQ-encode glue
//! ([`crate::floor0_encode::plan_floor0_coefficients`]): given a target LSP
//! `[coefficients]` list and the value book a packet's `[booknumber]`
//! selects, it produces the per-vector entry run the §6.2.2 step-7..11 loop
//! reads back. What sat in *front* of that — turning a desired
//! **linear-domain** floor envelope (one magnitude per spectral bin, the
//! domain the forward MDCT produces) into the per-packet `[amplitude]`
//! scalar and the LSP `[coefficients]` themselves — was the floor-0 encode
//! followup the crate README named, the floor-0 analogue of
//! [`crate::floor1_envelope::plan_floor1_envelope`]. This module is the
//! first half of that glue: the §6.2.3 curve-shape **amplitude inverse**.
//!
//! ## What the decoder does, and the inverse
//!
//! The §6.2.3 curve computation (see [`crate::floor0::Floor0Decoder`])
//! renders a length-`n` linear-domain envelope from a per-packet
//! `[amplitude]` integer and the LSP `[coefficients]`:
//!
//! 1. It builds the integer **Bark map** `map[i]` (§6.2.3), folding the `n`
//!    spectral bins onto `bark_map_size` Bark-scale buckets.
//! 2. For each bin it evaluates the LSP polynomial products `p` and `q`
//!    (order-parity dependent, over `cos(coefficients[j])` poles) and forms
//!    `linear[i] = exp(0.11512925 · (amplitude · offset /
//!    ((2^bits − 1) · sqrt(p + q)) − offset))`.
//!
//! Crucially the `sqrt(p + q)` term — the **shape** of the curve in the
//! log domain — depends *only* on the LSP coefficients, never on
//! `[amplitude]`. The amplitude is a single scalar that slides the whole
//! log-curve up or down. Writing `g(ω) = 1 / sqrt(p(ω) + q(ω))`,
//!
//! ```text
//! ln(linear[i]) = K · amplitude · g(ω_i) − C
//! ```
//!
//! with constants `K = 0.11512925 · offset / (2^bits − 1)` and
//! `C = 0.11512925 · offset` that depend only on the header. So the LSP
//! coefficients fix the curve **shape** `g(·)` and the amplitude fixes the
//! overall **gain**.
//!
//! [`fit_floor0_amplitude`] inverts the gain. Given the LSP coefficients
//! (their shape `g`) and a desired linear envelope, it solves for the
//! integer `[amplitude]` whose rendered curve best matches the target in
//! the **log domain** (the domain the floor is summed in, §4.3.6): for each
//! Bark bucket the target log-value implies a desired `amplitude · g`, and
//! the least-squares amplitude over all buckets is `Σ(g · t) / Σ(g²)` with
//! `t = (ln(target) + C) / K`. The result is rounded to the nearest integer
//! and clamped into `[1, 2^bits − 1]` (a zero amplitude is the §6.2.2
//! `'unused'` short-circuit, not a curve).
//!
//! ## Scope
//!
//! This module fits the per-packet `[amplitude]` to a target envelope
//! **given** the LSP coefficients. Deriving the LSP coefficients themselves
//! from a target envelope (the autocorrelation → Levinson-Durbin → LPC →
//! LSP chain) is the next floor-0 encode layer; choosing the entry run for
//! the chosen coefficients is the existing
//! [`crate::floor0_encode::plan_floor0_coefficients`].

use crate::floor0::bark;

/// The §6.2.3 log-domain conversion constant `0.11512925` (`= ln(10)/20`,
/// the dB-to-natural-log scale the curve exponent carries).
const LOG10_OVER_20: f32 = 0.115_129_25;

/// Errors that can arise while fitting a floor-0 `[amplitude]` to a target
/// linear-domain envelope (Vorbis I §6.2.3, encode direction).
#[derive(Debug, Clone, PartialEq)]
pub enum Floor0EnvelopeError {
    /// `order` was zero. §6.2.1 stores `floor0_order` as a `read 8 bits`
    /// field but the decoder rejects a zero order (the §6.2.2 step-7 loop
    /// reads no vectors), so no curve could render. The fitter mirrors that
    /// gate.
    ZeroOrder,
    /// `coefficients.len()` was shorter than `order`. §6.2.3 reads exactly
    /// `order` LSP poles; fewer cannot evaluate the curve. (The decoder
    /// permits a *longer* coefficient list — the surplus is ignored — so
    /// only the short case is an error.)
    TooFewCoefficients {
        /// `order` — the count §6.2.3 requires.
        order: usize,
        /// The supplied `coefficients.len()`.
        actual: usize,
    },
    /// `amplitude_bits` was zero. §6.2.1 stores it as a `read 6 bits`
    /// field; a zero width makes the `[amplitude]` read always zero, so no
    /// nonzero-amplitude curve could ever round-trip. The decoder rejects
    /// such a header, and the fitter mirrors it.
    ZeroAmplitudeBits,
    /// `amplitude_offset` was zero. The §6.2.3 exponent carries
    /// `amplitude · offset / (… )`; with a zero offset the amplitude term
    /// vanishes entirely (every amplitude renders the identical curve), so
    /// the gain is uninvertible. A real floor-0 header always carries a
    /// nonzero offset.
    ZeroAmplitudeOffset,
    /// The supplied envelope was empty. The fitter samples one value per
    /// spectral bin; with no bins there is nothing to fit.
    EmptyEnvelope,
    /// The envelope carried a non-finite or non-positive sample. The
    /// log-domain fit takes `ln(target)`; a `0`, negative, NaN or `∞`
    /// sample has no finite log. Carries the offending bin index.
    NonPositiveSample(usize),
    /// Every LSP shape weight `g(ω)` was zero (a degenerate coefficient set
    /// whose `p + q` diverged at every bucket). With no shape to scale, the
    /// amplitude is indeterminate. Practically unreachable for a real LSP
    /// set; reported rather than dividing by zero.
    DegenerateShape,
}

impl core::fmt::Display for Floor0EnvelopeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Floor0EnvelopeError::ZeroOrder => {
                write!(f, "vorbis floor0 envelope: order is zero (§6.2.1)")
            }
            Floor0EnvelopeError::TooFewCoefficients { order, actual } => write!(
                f,
                "vorbis floor0 envelope: coefficients.len()={actual} < order={order} (§6.2.3)"
            ),
            Floor0EnvelopeError::ZeroAmplitudeBits => write!(
                f,
                "vorbis floor0 envelope: amplitude_bits is zero (§6.2.1; no curve representable)"
            ),
            Floor0EnvelopeError::ZeroAmplitudeOffset => write!(
                f,
                "vorbis floor0 envelope: amplitude_offset is zero (§6.2.3 gain term vanishes)"
            ),
            Floor0EnvelopeError::EmptyEnvelope => {
                write!(f, "vorbis floor0 envelope: target envelope is empty")
            }
            Floor0EnvelopeError::NonPositiveSample(i) => write!(
                f,
                "vorbis floor0 envelope: envelope[{i}] is non-finite or non-positive (no log)"
            ),
            Floor0EnvelopeError::DegenerateShape => write!(
                f,
                "vorbis floor0 envelope: every LSP shape weight g(ω) was zero"
            ),
        }
    }
}

impl std::error::Error for Floor0EnvelopeError {}

/// Header inputs the §6.2.3 curve-shape evaluation needs. A thin view over
/// the [`crate::setup::Floor0Header`] fields the math reads, so the fitter
/// (and its tests) can drive the shape without a full decoder.
#[derive(Debug, Clone, Copy)]
pub struct Floor0ShapeParams {
    /// `floor0_order` — the number of LSP poles read from `coefficients`.
    pub order: usize,
    /// `floor0_rate` — sample-rate hint (Hz) feeding the Bark map.
    pub rate: u32,
    /// `floor0_bark_map_size` — the integer Bark-bucket count.
    pub bark_map_size: u32,
    /// `floor0_amplitude_bits` — width of the `[amplitude]` field.
    pub amplitude_bits: u8,
    /// `floor0_amplitude_offset`.
    pub amplitude_offset: u8,
}

/// Build the §6.2.3 integer Bark map for `n` spectral bins.
///
/// `map[i]` (`i ∈ 0..n`) folds bin `i` onto a `bark_map_size` Bark bucket;
/// the curve render holds one `linear_floor_value` per *distinct* map
/// value. Mirrors the decoder's map computation exactly (the `map[n] = -1`
/// sentinel the decoder appends is not needed here — the fitter walks
/// `0..n`).
fn bark_map(rate: u32, bark_map_size: u32, n: usize) -> Vec<i32> {
    let bark_denominator = bark((0.5 * rate as f64) as f32);
    let mut map = Vec::with_capacity(n);
    for i in 0..n {
        let f = (rate as f64 * i as f64) / (2.0 * n as f64);
        let foobar = (bark(f as f32) * bark_map_size as f32 / bark_denominator).floor() as i32;
        map.push(foobar.min(bark_map_size as i32 - 1));
    }
    map
}

/// Evaluate the §6.2.3 LSP shape weight `g(ω) = 1 / sqrt(p + q)` at the
/// angular frequency `ω = π · map_value / bark_map_size`, for the order-
/// parity-correct `p`/`q` products over `cos(coefficients[j])`.
///
/// This is the amplitude-independent half of the §6.2.3 curve: the decoder
/// forms `exp(K·amplitude·g − C)`, and this returns `g`. The math mirrors
/// [`crate::floor0::Floor0Decoder`]'s private `curve_computation` exactly so
/// a fit against this shape reproduces the decoder's rendered curve.
fn lsp_shape_weight(coeffs: &[f32], order: usize, bark_map_size: u32, map_value: i32) -> f32 {
    let omega = std::f32::consts::PI * map_value as f32 / bark_map_size as f32;
    let cos_omega = omega.cos();

    let (p, q) = if order % 2 == 1 {
        let mut p_prod = 1.0f32 - cos_omega * cos_omega;
        let p_iters = (order - 1) / 2;
        for j in 0..p_iters {
            let c = coeffs[2 * j + 1].cos();
            let term = c - cos_omega;
            p_prod *= 4.0 * term * term;
        }
        let mut q_prod = 0.25f32;
        let q_iters = order / 2 + 1;
        for j in 0..q_iters {
            let c = coeffs[2 * j].cos();
            let term = c - cos_omega;
            q_prod *= 4.0 * term * term;
        }
        (p_prod, q_prod)
    } else {
        let mut p_prod = (1.0f32 - cos_omega) / 2.0;
        let mut q_prod = (1.0f32 + cos_omega) / 2.0;
        let iters = order / 2;
        for j in 0..iters {
            let c_odd = coeffs[2 * j + 1].cos();
            let term_odd = c_odd - cos_omega;
            p_prod *= 4.0 * term_odd * term_odd;
            let c_even = coeffs[2 * j].cos();
            let term_even = c_even - cos_omega;
            q_prod *= 4.0 * term_even * term_even;
        }
        (p_prod, q_prod)
    };

    let pq = p + q;
    let sqrt_pq = pq.max(0.0).sqrt().max(f32::MIN_POSITIVE);
    1.0 / sqrt_pq
}

/// Fit the per-packet floor-0 `[amplitude]` to a target linear-domain
/// envelope, **given** the LSP `[coefficients]` (Vorbis I §6.2.3, encode
/// direction).
///
/// `coefficients` is the LSP pole list (length ≥ `params.order`; surplus
/// ignored, matching the decoder). `envelope` is the desired
/// linear-amplitude floor, one value per spectral bin — exactly the
/// `n`-length curve [`crate::floor0::Floor0Decoder::render_curve`] would
/// produce for the returned amplitude.
///
/// The fit minimises squared error in the **log domain** (the domain §4.3.6
/// sums the floor in): writing the §6.2.3 curve as
/// `ln(linear) = K · amplitude · g(ω) − C`, the per-bin target for
/// `amplitude · g` is `t = (ln(envelope) + C) / K`, and the least-squares
/// amplitude over the (Bark-folded) buckets is `Σ(g · t) / Σ(g²)`. The raw
/// real amplitude is rounded to the nearest integer and clamped into
/// `[1, 2^amplitude_bits − 1]`.
///
/// # Errors
///
/// Returns a [`Floor0EnvelopeError`] for a zero order, a coefficient list
/// shorter than the order, a zero `amplitude_bits` or `amplitude_offset`
/// (the gain is then uninvertible), an empty envelope, a non-finite or
/// non-positive envelope sample, or a fully-degenerate LSP shape.
pub fn fit_floor0_amplitude(
    coefficients: &[f32],
    envelope: &[f32],
    params: &Floor0ShapeParams,
) -> Result<u32, Floor0EnvelopeError> {
    if params.order == 0 {
        return Err(Floor0EnvelopeError::ZeroOrder);
    }
    if coefficients.len() < params.order {
        return Err(Floor0EnvelopeError::TooFewCoefficients {
            order: params.order,
            actual: coefficients.len(),
        });
    }
    if params.amplitude_bits == 0 {
        return Err(Floor0EnvelopeError::ZeroAmplitudeBits);
    }
    if params.amplitude_offset == 0 {
        return Err(Floor0EnvelopeError::ZeroAmplitudeOffset);
    }
    if envelope.is_empty() {
        return Err(Floor0EnvelopeError::EmptyEnvelope);
    }
    for (i, &v) in envelope.iter().enumerate() {
        if !v.is_finite() || v <= 0.0 {
            return Err(Floor0EnvelopeError::NonPositiveSample(i));
        }
    }

    let n = envelope.len();
    let map = bark_map(params.rate, params.bark_map_size, n);

    // K = 0.11512925 · offset / (2^bits − 1);  C = 0.11512925 · offset.
    let bits = params.amplitude_bits as u32;
    let denom_int: u32 = if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let offset = params.amplitude_offset as f32;
    let k = LOG10_OVER_20 * offset / denom_int as f32;
    let c = LOG10_OVER_20 * offset;

    // Least-squares amplitude over the bins. The shape weight g(ω) depends
    // only on the Bark bucket, so distinct bins sharing a bucket carry the
    // same g — the per-bin sum naturally weights a bucket by how many bins
    // fold onto it, matching how the rendered curve replicates the value.
    let mut num = 0.0f64; // Σ(g · t)
    let mut den = 0.0f64; // Σ(g²)
    for (i, &v) in envelope.iter().enumerate() {
        let g = lsp_shape_weight(coefficients, params.order, params.bark_map_size, map[i]);
        // t = (ln(envelope) + C) / K  — the desired amplitude·g.
        let t = (v.ln() + c) / k;
        num += (g as f64) * (t as f64);
        den += (g as f64) * (g as f64);
    }
    if den <= 0.0 {
        return Err(Floor0EnvelopeError::DegenerateShape);
    }

    let amplitude_real = num / den;
    // Round to nearest, clamp into [1, 2^bits − 1]: amplitude 0 is the
    // §6.2.2 'unused' short-circuit, not a curve.
    let rounded = amplitude_real.round();
    let clamped = rounded.clamp(1.0, denom_int as f64);
    Ok(clamped as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::{VorbisCodebook, VqLookup};
    use crate::floor0::Floor0Decoder;
    use crate::setup::Floor0Header;

    /// A minimal valid lookup-2 value codebook so the §6.2.1 constructor
    /// accepts the header. `render_curve` never reads the book (it takes the
    /// amplitude + coefficients directly), so any decodable book suffices.
    fn dummy_value_book() -> VorbisCodebook {
        VorbisCodebook {
            dimensions: 2,
            entries: 2,
            codeword_lengths: vec![1, 1],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: vec![0, 0, 1, 1],
            },
        }
    }

    fn params() -> Floor0ShapeParams {
        Floor0ShapeParams {
            order: 4,
            rate: 44_100,
            bark_map_size: 256,
            amplitude_bits: 10,
            amplitude_offset: 32,
        }
    }

    /// Build a real `Floor0Decoder` matching a `Floor0ShapeParams` so the
    /// fitter's amplitude can be checked against the decoder's *actual*
    /// rendered curve (the ground-truth §6.2.3 render).
    fn decoder_for(p: &Floor0ShapeParams) -> Floor0Decoder {
        let header = Floor0Header {
            order: p.order as u8,
            rate: p.rate as u16,
            bark_map_size: p.bark_map_size as u16,
            amplitude_bits: p.amplitude_bits,
            amplitude_offset: p.amplitude_offset,
            book_list: vec![0],
        };
        // render_curve takes amplitude + coefficients directly and never
        // touches the book list, but the §6.2.1 constructor still validates
        // a non-empty book list resolving to a decodable value book.
        let books = vec![dummy_value_book()];
        Floor0Decoder::new(&header, &books).expect("decoder builds from a valid floor-0 header")
    }

    // ---------- error paths ----------

    #[test]
    fn zero_order_is_rejected() {
        let mut p = params();
        p.order = 0;
        assert_eq!(
            fit_floor0_amplitude(&[0.1; 4], &[1.0; 8], &p),
            Err(Floor0EnvelopeError::ZeroOrder)
        );
    }

    #[test]
    fn too_few_coefficients_is_rejected() {
        let p = params();
        assert_eq!(
            fit_floor0_amplitude(&[0.1; 3], &[1.0; 8], &p),
            Err(Floor0EnvelopeError::TooFewCoefficients {
                order: 4,
                actual: 3
            })
        );
    }

    #[test]
    fn zero_amplitude_bits_is_rejected() {
        let mut p = params();
        p.amplitude_bits = 0;
        assert_eq!(
            fit_floor0_amplitude(&[0.1; 4], &[1.0; 8], &p),
            Err(Floor0EnvelopeError::ZeroAmplitudeBits)
        );
    }

    #[test]
    fn zero_amplitude_offset_is_rejected() {
        let mut p = params();
        p.amplitude_offset = 0;
        assert_eq!(
            fit_floor0_amplitude(&[0.1; 4], &[1.0; 8], &p),
            Err(Floor0EnvelopeError::ZeroAmplitudeOffset)
        );
    }

    #[test]
    fn empty_envelope_is_rejected() {
        let p = params();
        assert_eq!(
            fit_floor0_amplitude(&[0.1; 4], &[], &p),
            Err(Floor0EnvelopeError::EmptyEnvelope)
        );
    }

    #[test]
    fn non_positive_sample_is_rejected() {
        let p = params();
        let mut env = vec![1.0f32; 8];
        env[3] = 0.0;
        assert_eq!(
            fit_floor0_amplitude(&[0.1; 4], &env, &p),
            Err(Floor0EnvelopeError::NonPositiveSample(3))
        );
        env[3] = -2.0;
        assert_eq!(
            fit_floor0_amplitude(&[0.1; 4], &env, &p),
            Err(Floor0EnvelopeError::NonPositiveSample(3))
        );
        env[3] = f32::NAN;
        assert_eq!(
            fit_floor0_amplitude(&[0.1; 4], &env, &p),
            Err(Floor0EnvelopeError::NonPositiveSample(3))
        );
    }

    // ---------- recovery / round-trip ----------

    /// Picking some LSP coefficients + a known amplitude, rendering the
    /// curve with the real decoder, then *fitting* the amplitude back from
    /// that exact curve must recover the original amplitude (the curve is
    /// generated by exactly the model the fit assumes, so recovery is
    /// near-exact up to integer rounding).
    #[test]
    fn fit_recovers_the_amplitude_that_generated_the_curve() {
        let p = params();
        let dec = decoder_for(&p);
        let coeffs = vec![0.3f32, 0.9, 1.6, 2.4];
        for &amp in &[1u32, 7, 64, 200, 511, 1023] {
            let curve = dec.render_curve(amp, &coeffs, 64);
            let fitted = fit_floor0_amplitude(&coeffs, &curve, &p).expect("fit succeeds");
            assert!(
                fitted.abs_diff(amp) <= 1,
                "fitted {fitted} should recover amplitude {amp} within ±1"
            );
        }
    }

    /// The fitted amplitude's rendered curve must track the target curve in
    /// the log domain: rendering with the fitted amplitude reproduces the
    /// shape, and the residual log-error is small.
    #[test]
    fn fitted_amplitude_renders_a_curve_close_to_the_target() {
        let p = params();
        let dec = decoder_for(&p);
        let coeffs = vec![0.2f32, 0.7, 1.4, 2.1];
        let target = dec.render_curve(150, &coeffs, 128);
        let fitted = fit_floor0_amplitude(&coeffs, &target, &p).expect("fit succeeds");
        let rendered = dec.render_curve(fitted, &coeffs, 128);
        // Mean-squared log error must be tiny — the curve is exactly the
        // model the fit assumes, so only integer-amplitude rounding remains.
        let mut sse = 0.0f64;
        for (a, b) in rendered.iter().zip(target.iter()) {
            let d = (a.ln() - b.ln()) as f64;
            sse += d * d;
        }
        let mse = sse / target.len() as f64;
        assert!(mse < 1e-2, "log-domain MSE {mse} should be small");
    }

    /// A target curve whose *shape* matches the coefficients but whose gain
    /// is higher must fit a higher amplitude than a quieter target — the
    /// fit is monotone in target gain.
    #[test]
    fn fit_is_monotone_in_target_gain() {
        let p = params();
        let dec = decoder_for(&p);
        let coeffs = vec![0.4f32, 1.0, 1.8, 2.5];
        let quiet = dec.render_curve(40, &coeffs, 96);
        let loud = dec.render_curve(400, &coeffs, 96);
        let fit_quiet = fit_floor0_amplitude(&coeffs, &quiet, &p).unwrap();
        let fit_loud = fit_floor0_amplitude(&coeffs, &loud, &p).unwrap();
        assert!(
            fit_loud > fit_quiet,
            "louder target ({fit_loud}) must fit a higher amplitude than quieter ({fit_quiet})"
        );
    }

    /// The fitted amplitude is always a legal field value: ≥ 1 (never the
    /// 'unused' zero) and ≤ 2^amplitude_bits − 1.
    #[test]
    fn fitted_amplitude_is_always_in_field_range() {
        let p = params();
        let dec = decoder_for(&p);
        let coeffs = vec![0.5f32, 1.1, 1.9, 2.6];
        // A tiny target (well below amplitude 1's curve) clamps up to 1.
        let tiny = vec![1e-30f32; 64];
        let fitted = fit_floor0_amplitude(&coeffs, &tiny, &p).unwrap();
        assert!((1..=1023).contains(&fitted));
        // A huge target clamps down to the field max.
        let huge = dec.render_curve(1023, &coeffs, 64);
        let scaled: Vec<f32> = huge.iter().map(|x| x * 1e6).collect();
        let fitted = fit_floor0_amplitude(&coeffs, &scaled, &p).unwrap();
        assert_eq!(fitted, 1023);
    }
}

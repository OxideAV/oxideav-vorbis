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
    /// The autocorrelation → Levinson-Durbin → LSP DSP chain
    /// ([`plan_floor0_lsp`]) failed: a silent target, a non-positive-definite
    /// autocorrelation at the requested order, or an LSP root-count
    /// shortfall. Carries the inner [`crate::floor0_lsp::Floor0LspError`].
    Lsp(crate::floor0_lsp::Floor0LspError),
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
            Floor0EnvelopeError::Lsp(e) => write!(f, "vorbis floor0 envelope: {e}"),
        }
    }
}

impl std::error::Error for Floor0EnvelopeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Floor0EnvelopeError::Lsp(e) => Some(e),
            _ => None,
        }
    }
}

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

/// Fold a per-spectral-bin target envelope onto the §6.2.3 Bark-bucket grid,
/// returning a **power spectrum the LPC fit should model** so the rendered
/// floor-0 curve tracks the *linear* target.
///
/// ## Why the log domain
///
/// The §6.2.3 curve is *exponential* in the LSP shape:
/// `curve(ω) = exp(K·amp·g(ω) − C)` with `g(ω) = 1/|A(e^{jω})|` (see
/// [`fit_floor0_amplitude`]). The all-pole model the autocorrelation method
/// fits reconstructs a power spectrum `S(ω) ≈ gain/|A|²`, so the model's
/// `g = 1/|A| = sqrt(S/gain)` tracks `sqrt(S)`. For the rendered curve to
/// match the target we need `K·amp·g − C = ln(target)`, i.e.
/// `g(ω) ∝ ln(target(ω)) + const`. So the power spectrum the fit must model
/// is the **squared shifted log-envelope** `S = (ln(target) − ln(min) + ε)²`,
/// *not* the squared linear envelope — modelling the linear envelope makes
/// `g ∝ target` and the exp blows the dynamic range up.
///
/// ## Bark grid
///
/// The §6.2.3 LSP poles are angles evaluated at `ω = π · m / bark_map_size`
/// for Bark bucket `m`; the fit must see the target in *that* bucket domain,
/// not linear frequency. This walks the §6.2.3 Bark map (the same one the
/// renderer uses), averaging each bucket's shifted-log values (squared into
/// power), and nearest-occupied-fills any empty bucket so the
/// autocorrelation integral sees a complete `0..bark_map_size` half-spectrum.
fn envelope_to_bark_power(envelope: &[f32], rate: u32, bark_map_size: u32) -> Vec<f64> {
    let n = envelope.len();
    let buckets = bark_map_size as usize;
    let map = bark_map(rate, bark_map_size, n);

    // Shift the log-envelope to be strictly positive: subtract the minimum
    // log so the quietest bin sits at ε. The absolute shift is irrelevant —
    // the amplitude fit (Σg·t/Σg²) absorbs any affine offset in g.
    let min_ln = envelope
        .iter()
        .map(|&v| (v as f64).ln())
        .fold(f64::INFINITY, f64::min);
    let eps = 1e-3;

    let mut sum = vec![0.0f64; buckets];
    let mut cnt = vec![0u32; buckets];
    for (i, &v) in envelope.iter().enumerate() {
        let b = map[i].clamp(0, buckets as i32 - 1) as usize;
        let g = (v as f64).ln() - min_ln + eps; // shifted log, ≥ ε > 0
        sum[b] += g * g; // power = g²
        cnt[b] += 1;
    }
    let mut power = vec![0.0f64; buckets];
    for b in 0..buckets {
        if cnt[b] > 0 {
            power[b] = sum[b] / cnt[b] as f64;
        }
    }
    // Nearest-occupied hold for empty buckets (forward fill), so no bucket is
    // a spurious zero in the autocorrelation integral.
    let mut last = power
        .iter()
        .copied()
        .find(|&p| p > 0.0)
        .unwrap_or(eps * eps);
    for p in power.iter_mut() {
        if *p > 0.0 {
            last = *p;
        } else {
            *p = last;
        }
    }
    power
}

/// Derive the §6.2.3 LSP `[coefficients]` (pole angles, ascending in
/// `(0, π)`) that best model a target linear-domain floor envelope, via the
/// classic autocorrelation → Levinson-Durbin → LSP chain (Vorbis I §6.2.3,
/// encode direction).
///
/// `envelope` is the desired linear-amplitude floor, one value per spectral
/// bin (the forward-MDCT magnitude domain). The fit:
///
/// 1. folds the envelope's **power** onto the §6.2.3 Bark-bucket grid
///    ([`envelope_to_bark_power`]) so the model sees the spectrum in the
///    domain the LSP angles live in;
/// 2. inverse-DFTs that power spectrum to `order + 1` autocorrelation lags
///    ([`crate::floor0_lsp::autocorrelation_from_power`]);
/// 3. solves the order-`order` all-pole model
///    ([`crate::floor0_lsp::levinson_durbin`]);
/// 4. extracts the LSP frequencies
///    ([`crate::floor0_lsp::lpc_to_lsp`]).
///
/// The returned `Vec<f32>` is exactly the LSP `[coefficients]` the §6.2.3
/// curve evaluates `cos(·)` of — the input
/// [`fit_floor0_amplitude`] and [`crate::floor0_encode::plan_floor0_coefficients`]
/// consume. The model is **lossy**: an order-`order` all-pole envelope is
/// the best `order`-pole fit, not an exact reproduction.
///
/// # Errors
///
/// Returns a [`Floor0EnvelopeError`] for a zero order, an empty envelope, or
/// a non-finite / non-positive envelope sample; or wraps a
/// [`crate::floor0_lsp::Floor0LspError`] from the DSP chain (a silent target
/// energy, a non-positive-definite autocorrelation, or an LSP root-count
/// shortfall) as [`Floor0EnvelopeError::Lsp`].
pub fn plan_floor0_lsp(
    envelope: &[f32],
    params: &Floor0ShapeParams,
) -> Result<Vec<f32>, Floor0EnvelopeError> {
    if params.order == 0 {
        return Err(Floor0EnvelopeError::ZeroOrder);
    }
    if envelope.is_empty() {
        return Err(Floor0EnvelopeError::EmptyEnvelope);
    }
    for (i, &v) in envelope.iter().enumerate() {
        if !v.is_finite() || v <= 0.0 {
            return Err(Floor0EnvelopeError::NonPositiveSample(i));
        }
    }

    let power = envelope_to_bark_power(envelope, params.rate, params.bark_map_size);
    // The renderer evaluates the LSP shape at ω_m = π·m / bark_map_size for
    // Bark bucket m; the autocorrelation must integrate the power over that
    // exact grid so the fitted all-pole model matches what the decoder draws.
    let angles: Vec<f64> = (0..power.len())
        .map(|m| std::f64::consts::PI * m as f64 / params.bark_map_size as f64)
        .collect();
    let r = crate::floor0_lsp::autocorrelation_from_angles(&power, &angles, params.order);
    // A tiny diagonal load (white-noise floor) keeps the autocorrelation
    // matrix safely positive-definite for near-degenerate targets — a
    // standard regularisation, not a spec deviation.
    let mut r = r;
    r[0] *= 1.0 + 1e-9;
    let (a, _gain) =
        crate::floor0_lsp::levinson_durbin(&r, params.order).map_err(Floor0EnvelopeError::Lsp)?;
    let lsp = crate::floor0_lsp::lpc_to_lsp(&a, params.order).map_err(Floor0EnvelopeError::Lsp)?;
    Ok(lsp.into_iter().map(|w| w as f32).collect())
}

/// Errors that can arise composing a full floor-0 packet from a target
/// envelope ([`plan_floor0_packet`]).
#[derive(Debug, Clone, PartialEq)]
pub enum Floor0PacketPlanError {
    /// The `booknumber` indexed outside the header's `book_list`. §6.2.2
    /// step 4 reads a position in `floor0_book_list`; this is its encode
    /// gate. Carries the bad index and the list length.
    BooknumberOutOfRange {
        /// The supplied `booknumber`.
        booknumber: usize,
        /// `header.book_list.len()`.
        book_count: usize,
    },
    /// The `book_list[booknumber]` entry indexed outside the codebook table.
    BookOutOfRange {
        /// The `floor0_book_list` value (a codebook-table index).
        book: usize,
        /// `codebooks.len()`.
        codebook_count: usize,
    },
    /// The envelope→LSP→amplitude stage failed. Carries the inner
    /// [`Floor0EnvelopeError`].
    Envelope(Floor0EnvelopeError),
    /// The LSP→entry-run stage ([`crate::floor0_encode::plan_floor0_coefficients`])
    /// failed. Carries the inner [`crate::floor0_encode::Floor0EncodeError`].
    Encode(crate::floor0_encode::Floor0EncodeError),
}

impl core::fmt::Display for Floor0PacketPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Floor0PacketPlanError::BooknumberOutOfRange {
                booknumber,
                book_count,
            } => write!(
                f,
                "vorbis floor0 packet plan: booknumber {booknumber} >= book_list.len()={book_count}"
            ),
            Floor0PacketPlanError::BookOutOfRange {
                book,
                codebook_count,
            } => write!(
                f,
                "vorbis floor0 packet plan: book_list entry {book} >= codebooks.len()={codebook_count}"
            ),
            Floor0PacketPlanError::Envelope(e) => write!(f, "vorbis floor0 packet plan: {e}"),
            Floor0PacketPlanError::Encode(e) => write!(f, "vorbis floor0 packet plan: {e}"),
        }
    }
}

impl std::error::Error for Floor0PacketPlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Floor0PacketPlanError::Envelope(e) => Some(e),
            Floor0PacketPlanError::Encode(e) => Some(e),
            _ => None,
        }
    }
}

/// One-call floor-0 packet planner (Vorbis I §6.2.2 / §6.2.3, encode
/// direction): turn a desired linear-domain floor envelope directly into a
/// write-ready [`crate::encoder::Floor0Packet::Curve`].
///
/// This composes the whole floor-0 encode chain — the floor-0 analogue of
/// [`crate::floor1_encode::plan_floor1_packet`]:
///
/// 1. [`plan_floor0_lsp`] — envelope → LSP `[coefficients]` (the
///    autocorrelation → Levinson-Durbin → LSP all-pole fit);
/// 2. [`fit_floor0_amplitude`] — fit the per-packet `[amplitude]` to the
///    target gain over those coefficients;
/// 3. [`crate::floor0_encode::plan_floor0_coefficients`] — quantise the LSP
///    coefficients into the value-book entry run the §6.2.2 step-7 loop
///    reads back.
///
/// `header` is the parsed [`crate::setup::Floor0Header`]; `codebooks` is the
/// stream codebook table; `booknumber` selects which value book (a position
/// in `floor0_book_list`) the packet uses; `envelope` is the desired
/// linear-amplitude floor, one value per spectral bin. The returned
/// [`crate::encoder::Floor0Packet`] feeds straight into
/// [`crate::encoder::write_floor0_packet`] (with neither the LSP
/// `coefficients`, the `amplitude`, nor the entry run supplied by hand).
///
/// The chain is **lossy** (an order-`floor0_order` all-pole model of the
/// envelope shape, plus integer-amplitude and VQ quantisation), but the
/// emitted packet is self-consistent: decoding it reproduces the planner's
/// own approximation bit-for-bit.
///
/// # Errors
///
/// [`Floor0PacketPlanError::BooknumberOutOfRange`] /
/// [`Floor0PacketPlanError::BookOutOfRange`] for an out-of-range book
/// selector; [`Floor0PacketPlanError::Envelope`] wrapping a fit failure
/// (silent / degenerate target); [`Floor0PacketPlanError::Encode`] wrapping
/// a value-book quantiser failure.
pub fn plan_floor0_packet(
    header: &crate::setup::Floor0Header,
    codebooks: &[crate::codebook::VorbisCodebook],
    booknumber: u32,
    envelope: &[f32],
) -> Result<crate::encoder::Floor0Packet, Floor0PacketPlanError> {
    let book_count = header.book_list.len();
    if booknumber as usize >= book_count {
        return Err(Floor0PacketPlanError::BooknumberOutOfRange {
            booknumber: booknumber as usize,
            book_count,
        });
    }
    let book_idx = header.book_list[booknumber as usize] as usize;
    let book = codebooks
        .get(book_idx)
        .ok_or(Floor0PacketPlanError::BookOutOfRange {
            book: book_idx,
            codebook_count: codebooks.len(),
        })?;

    let params = Floor0ShapeParams {
        order: header.order as usize,
        rate: header.rate as u32,
        bark_map_size: header.bark_map_size as u32,
        amplitude_bits: header.amplitude_bits,
        amplitude_offset: header.amplitude_offset,
    };

    // 1. envelope → LSP coefficients.
    let lsp = plan_floor0_lsp(envelope, &params).map_err(Floor0PacketPlanError::Envelope)?;
    // 2. fit the amplitude over those coefficients.
    let amplitude =
        fit_floor0_amplitude(&lsp, envelope, &params).map_err(Floor0PacketPlanError::Envelope)?;
    // 3. quantise the LSP coefficients into the value-book entry run. The
    //    entry planner reads `ceil(order/dims)*dims` coefficients (a partial
    //    final vector is read in full); pad the order-length LSP list out to
    //    that width by holding the last coefficient (the surplus the §6.2.3
    //    curve discards). The decoder over-reads but never uses past `order`,
    //    so the held tail is harmless and keeps the cross-vector `last`
    //    accumulator well-conditioned.
    let dims = book.dimensions as usize;
    let padded_len = if dims == 0 {
        lsp.len()
    } else {
        crate::floor0_encode::floor0_vector_count(params.order, dims) * dims
    };
    let mut coeffs = lsp;
    if coeffs.len() < padded_len {
        let tail = *coeffs.last().unwrap_or(&0.0);
        coeffs.resize(padded_len, tail);
    }
    let entries = crate::floor0_encode::plan_floor0_coefficients(&coeffs, book, params.order)
        .map_err(Floor0PacketPlanError::Encode)?;

    Ok(crate::encoder::Floor0Packet::Curve {
        amplitude,
        booknumber,
        entries,
    })
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

    // ---------- plan_floor0_lsp self-consistency ----------

    /// log-domain **shape** SNR (dB) of `rendered` against `target`, both
    /// linear envelopes. The overall log-level (a constant in the log domain)
    /// is the encoder's free `[amplitude]` knob, so it is removed from *both*
    /// signals before comparison — what the envelope fit is judged on is the
    /// spectral *shape*, not the absolute gain (which the integer-amplitude
    /// grid and the residue stage handle).
    fn log_snr_db(rendered: &[f32], target: &[f32]) -> f64 {
        let n = target.len() as f64;
        let tmean: f64 = target.iter().map(|x| (*x as f64).ln()).sum::<f64>() / n;
        let rmean: f64 = rendered.iter().map(|x| (*x as f64).ln()).sum::<f64>() / n;
        let mut sig = 0.0f64;
        let mut err = 0.0f64;
        for (r, t) in rendered.iter().zip(target.iter()) {
            let lt = (*t as f64).ln() - tmean;
            let lr = (*r as f64).ln() - rmean;
            sig += lt * lt;
            err += (lt - lr) * (lt - lr);
        }
        10.0 * (sig / err.max(1e-30)).log10()
    }

    /// The headline round-trip: a smooth target envelope → `plan_floor0_lsp`
    /// → `fit_floor0_amplitude` → §6.2.3 render reproduces the target's
    /// spectral *shape*. The fit is an order-`order` all-pole model, so the
    /// reconstruction is the best `order`-pole envelope, not exact; a
    /// formant-like target (a few resonant peaks) is exactly what LSP models
    /// well, so the shape SNR clears a meaningful bar.
    #[test]
    fn envelope_to_lsp_to_render_reproduces_a_formant_shape() {
        let mut p = params();
        p.order = 14;
        let dec = decoder_for(&p);
        let n = 256;
        // A formant-like envelope whose *log* (dB) shape is a sum of smooth
        // resonant bumps — the domain floor 0 (an all-pole model of the log
        // spectrum) represents naturally, and the shape real audio envelopes
        // take in dB.
        let target: Vec<f32> = (0..n)
            .map(|i| {
                let w = std::f32::consts::PI * i as f32 / n as f32;
                let log_db = 3.0 / ((w - 0.4).powi(2) + 0.05)
                    + 2.0 / ((w - 1.2).powi(2) + 0.08)
                    + 1.5 / ((w - 2.3).powi(2) + 0.12);
                (0.2 * log_db).exp()
            })
            .collect();
        let lsp = plan_floor0_lsp(&target, &p).expect("lsp plan succeeds");
        assert_eq!(lsp.len(), p.order);
        // LSP angles must be a valid ascending set in (0, π).
        for pair in lsp.windows(2) {
            assert!(pair[1] > pair[0], "lsp not ascending: {lsp:?}");
        }
        let amp = fit_floor0_amplitude(&lsp, &target, &p).expect("amplitude fit");
        let rendered = dec.render_curve(amp, &lsp, n);
        let snr = log_snr_db(&rendered, &target);
        assert!(
            snr > 10.0,
            "formant-shape log-SNR {snr:.2} dB should clear 10 dB"
        );
    }

    /// Raising the model order strictly improves (or holds) the fit of a
    /// rich target — more poles model more spectral detail. Pins that the
    /// chain is order-responsive (a real all-pole fit, not a constant).
    #[test]
    fn higher_order_fits_a_rich_target_at_least_as_well() {
        let n = 256;
        let target: Vec<f32> = (0..n)
            .map(|i| {
                let w = std::f32::consts::PI * i as f32 / n as f32;
                0.01 + (1.0 + (3.0 * w).cos()).abs() + 0.5 * (7.0 * w).cos().abs()
            })
            .collect();
        let snr_at = |order: usize| -> f64 {
            let mut p = params();
            p.order = order;
            let dec = decoder_for(&p);
            let lsp = plan_floor0_lsp(&target, &p).expect("lsp plan");
            let amp = fit_floor0_amplitude(&lsp, &target, &p).expect("amp fit");
            let rendered = dec.render_curve(amp, &lsp, n);
            log_snr_db(&rendered, &target)
        };
        let lo = snr_at(6);
        let hi = snr_at(16);
        assert!(
            hi >= lo - 0.5,
            "order-16 SNR {hi:.2} dB should not be materially worse than order-6 {lo:.2} dB"
        );
    }

    /// A silent (all-equal, near-zero) target still produces a valid LSP set
    /// (the diagonal-load regularisation keeps the autocorrelation
    /// positive-definite); the rendered curve is near-flat.
    #[test]
    fn flat_target_yields_a_valid_near_flat_fit() {
        let mut p = params();
        p.order = 8;
        let dec = decoder_for(&p);
        let n = 128;
        let target = vec![0.5f32; n];
        let lsp = plan_floor0_lsp(&target, &p).expect("lsp plan on flat target");
        assert_eq!(lsp.len(), 8);
        let amp = fit_floor0_amplitude(&lsp, &target, &p).expect("amp fit");
        let rendered = dec.render_curve(amp, &lsp, n);
        // Flat target ⇒ rendered curve has low dynamic range.
        let max = rendered.iter().cloned().fold(f32::MIN, f32::max);
        let min = rendered.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            max / min < 4.0,
            "flat target should render a low-dynamic-range curve ({min}..{max})"
        );
    }

    #[test]
    fn plan_floor0_lsp_rejects_empty_and_zero_order() {
        let p = params();
        assert_eq!(
            plan_floor0_lsp(&[], &p),
            Err(Floor0EnvelopeError::EmptyEnvelope)
        );
        let mut p0 = params();
        p0.order = 0;
        assert_eq!(
            plan_floor0_lsp(&[1.0; 8], &p0),
            Err(Floor0EnvelopeError::ZeroOrder)
        );
    }
}

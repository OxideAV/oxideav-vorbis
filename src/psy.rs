//! Psychoacoustic masking model (encoder side).
//!
//! The Vorbis I specification defines only the *decode* side of the
//! codec; how an encoder decides what spectral detail is audible — and
//! therefore worth bits — is deliberately unspecified ("Vorbis
//! encoders are free to use any means to choose what to code"). This
//! module is a clean-room encoder-side masking model built from
//! standard, textbook psychoacoustics on top of the one perceptual
//! primitive the spec itself carries: the §6.2.3 Bark scale
//! ([`crate::floor0::bark`]).
//!
//! ## Model
//!
//! Given one analysis frame's MDCT coefficient magnitudes (`n/2` bins,
//! the domain the §4.3.7 forward MDCT produces), the model estimates,
//! per bin, the largest quantisation-noise amplitude that stays
//! inaudible — the **masking threshold**:
//!
//! 1. **Critical-band analysis.** Bins are grouped into 1-Bark bands
//!    via the §6.2.3 Bark map (bin `k`'s centre frequency is
//!    `(k + ½) · rate / n`); each band's energy is summed.
//! 2. **Tonality estimate.** Each band's spectral flatness (geometric
//!    over arithmetic mean of the bin energies) separates tone-like
//!    bands (flatness → 0, energy concentrated in few bins) from
//!    noise-like bands (flatness → 1). The tonality coefficient
//!    `α ∈ [0, 1]` interpolates between the two classic masking
//!    offsets: a tonal masker masks *less* relative to its own level
//!    (offset `14.5 + z` dB at critical band `z`) than a noise masker
//!    (offset `5.5` dB) — the tone-masking-noise / noise-masking-tone
//!    asymmetry used throughout the perceptual-coding literature.
//! 3. **Spreading.** A masker's influence decays across the Bark axis
//!    asymmetrically: steeply toward lower frequencies
//!    (−27 dB/Bark) and shallowly toward higher ones (−10 dB/Bark) —
//!    masking spreads upward. Each band's threshold takes the
//!    **maximum** over all maskers' spread contributions (a
//!    conservative, deterministic combination).
//! 4. **Threshold in quiet.** The spread threshold is floored,
//!    per bin, by the absolute threshold of hearing (the standard
//!    three-term analytic ATH approximation in dB SPL over kHz),
//!    calibrated by [`PsyConfig::full_scale_db`] — the SPL assigned to
//!    a full-scale (`|X| = 1.0`) spectral line.
//!
//! The result ([`MaskingAnalysis::threshold`]) is a per-bin **linear
//! amplitude**: reconstruction error at or below it is predicted
//! inaudible. Downstream consumers:
//!
//! * the **floor-1 envelope target** ([`plan_psy_floor_envelope`]) —
//!   a floor that rides the signal envelope where the signal is
//!   audible and the masking threshold where it is not, so
//!   residue-domain quantisation noise (which the decoder multiplies
//!   by the floor, §4.3.6) is shaped to sit under the threshold;
//! * the **residue partition weights**
//!   ([`residue_partition_weights`]) — per-partition factors
//!   `(floor/threshold)²` that turn the residue rate-distortion
//!   chooser's squared error into an approximate noise-to-mask ratio,
//!   steering bits toward partitions where noise would be audible.

use crate::floor0::bark;

/// Configuration for the masking model.
#[derive(Debug, Clone, PartialEq)]
pub struct PsyConfig {
    /// The stream sample rate in Hz (§4.2.2 `audio_sample_rate`).
    /// Must be non-zero; it fixes each bin's centre frequency.
    pub sample_rate: u32,
    /// The sound-pressure level, in dB SPL, assigned to a full-scale
    /// spectral line (`|X| = 1.0`). This calibrates the analytic
    /// absolute-threshold-of-hearing curve against the digital
    /// amplitude domain; `96.0` (the peak SPL a 16-bit signal is
    /// conventionally mapped to) is the default.
    pub full_scale_db: f32,
    /// Extra decibels **subtracted** from the masking threshold — a
    /// safety margin / quality lever. Positive values lower the
    /// threshold (more detail deemed audible → more bits), negative
    /// values raise it (more aggressive masking → fewer bits). `0.0`
    /// is the model's nominal operating point.
    pub threshold_offset_db: f32,
}

impl PsyConfig {
    /// A nominal configuration for the given sample rate:
    /// `full_scale_db = 96.0`, `threshold_offset_db = 0.0`.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        PsyConfig {
            sample_rate,
            full_scale_db: 96.0,
            threshold_offset_db: 0.0,
        }
    }
}

/// The per-frame output of [`compute_masking`].
#[derive(Debug, Clone, PartialEq)]
pub struct MaskingAnalysis {
    /// Per-bin masking threshold as a **linear amplitude** in the same
    /// domain as the input spectrum: reconstruction error at or below
    /// `threshold[k]` at bin `k` is predicted inaudible. Always finite
    /// and strictly positive (the threshold-in-quiet floor guarantees
    /// a positive lower bound even for a silent frame).
    pub threshold: Vec<f32>,
    /// Per-Bark-band tonality coefficient `α ∈ [0, 1]`: `1.0` for a
    /// fully tone-like band (energy concentrated in one bin), `0.0`
    /// for a fully noise-like (flat) or empty band. Band `b` covers
    /// the bins whose §6.2.3 Bark value floors to `b`.
    pub band_tonality: Vec<f32>,
    /// Per-bin Bark band index (`threshold.len()` entries, each
    /// `< band_tonality.len()`), so a consumer can map bins back to
    /// the band-level figures.
    pub bin_band: Vec<usize>,
}

/// Errors the masking model can raise.
#[derive(Debug, Clone, PartialEq)]
pub enum PsyError {
    /// The input spectrum was empty — there is nothing to analyse.
    EmptySpectrum,
    /// A spectrum coefficient was NaN or infinite. Carries the bin.
    NonFiniteSpectrum {
        /// The offending bin index.
        bin: usize,
        /// The offending value.
        value: f32,
    },
    /// `sample_rate` was zero — bin frequencies would be undefined.
    ZeroSampleRate,
    /// A configuration decibel field (`full_scale_db` /
    /// `threshold_offset_db`) was NaN or infinite.
    NonFiniteConfig,
    /// The floor curve and masking threshold passed to a downstream
    /// helper had mismatched lengths.
    LengthMismatch {
        /// The floor / spectrum length supplied.
        floor: usize,
        /// The threshold length supplied.
        threshold: usize,
    },
    /// A residue window `[begin, end)` fell outside the supplied
    /// curves, was empty, or was not a multiple of `partition_size`.
    BadResidueWindow {
        /// `residue_begin`.
        begin: usize,
        /// `residue_end`.
        end: usize,
        /// The curve length the window must fit inside.
        len: usize,
    },
    /// `partition_size` was zero.
    ZeroPartitionSize,
    /// A floor-curve value passed to [`residue_partition_weights`] was
    /// NaN, infinite, or negative. Carries the bin.
    BadFloorValue {
        /// The offending bin index.
        bin: usize,
        /// The offending value.
        value: f32,
    },
}

impl core::fmt::Display for PsyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PsyError::EmptySpectrum => write!(f, "vorbis psy: empty spectrum"),
            PsyError::NonFiniteSpectrum { bin, value } => {
                write!(
                    f,
                    "vorbis psy: non-finite spectrum value {value} at bin {bin}"
                )
            }
            PsyError::ZeroSampleRate => write!(f, "vorbis psy: sample rate is zero"),
            PsyError::NonFiniteConfig => {
                write!(f, "vorbis psy: non-finite dB configuration value")
            }
            PsyError::LengthMismatch { floor, threshold } => write!(
                f,
                "vorbis psy: floor length {floor} != threshold length {threshold}"
            ),
            PsyError::BadResidueWindow { begin, end, len } => write!(
                f,
                "vorbis psy: residue window [{begin}, {end}) invalid for curve length {len}"
            ),
            PsyError::ZeroPartitionSize => write!(f, "vorbis psy: partition size is zero"),
            PsyError::BadFloorValue { bin, value } => {
                write!(f, "vorbis psy: bad floor value {value} at bin {bin}")
            }
        }
    }
}

impl std::error::Error for PsyError {}

/// The absolute threshold of hearing at frequency `f_hz`, in dB SPL —
/// the standard three-term analytic approximation over kHz used
/// throughout the psychoacoustics literature:
///
/// ```text
/// ath(f) = 3.64 f^-0.8 − 6.5 exp(−0.6 (f − 3.3)²) + 10⁻³ f⁴   [f in kHz]
/// ```
///
/// The curve is high at very low frequencies, dips to its minimum in
/// the 3–4 kHz region (where hearing is most sensitive), and rises
/// steeply above ~10 kHz. The evaluation clamps `f` to at least 20 Hz
/// (the nominal lower edge of hearing; below it the power term
/// diverges) and the result into `[-30, 120]` dB so extreme bins stay
/// finite and orderable.
#[must_use]
pub fn ath_db(f_hz: f32) -> f32 {
    let f = (f_hz.max(20.0)) / 1000.0; // kHz, clamped away from 0
    let v = 3.64 * f.powf(-0.8) - 6.5 * (-0.6 * (f - 3.3) * (f - 3.3)).exp() + 1.0e-3 * f.powi(4);
    v.clamp(-30.0, 120.0)
}

/// Masking spread slopes across the Bark axis, dB per Bark: the decay
/// of a masker's influence toward **lower** frequencies (steep) and
/// toward **higher** frequencies (shallow — masking spreads upward).
const SPREAD_DOWN_DB_PER_BARK: f32 = 27.0;
const SPREAD_UP_DB_PER_BARK: f32 = 10.0;

/// The tonal-masker offset at critical band `z`: a tone at band `z`
/// masks noise `14.5 + z` dB below its own level.
fn tonal_offset_db(z: f32) -> f32 {
    14.5 + z
}

/// The noise-masker offset: a noise band masks a tone 5.5 dB below its
/// own level.
const NOISE_OFFSET_DB: f32 = 5.5;

/// Compute the per-bin masking threshold for one analysis frame.
///
/// `spectrum` holds the frame's MDCT coefficients (length `n/2`); the
/// model uses their magnitudes, so signed coefficients are accepted
/// as-is. See the module docs for the model; see [`PsyConfig`] for the
/// calibration knobs.
///
/// # Errors
///
/// [`PsyError::EmptySpectrum`] for an empty input,
/// [`PsyError::NonFiniteSpectrum`] for a NaN/±∞ coefficient,
/// [`PsyError::ZeroSampleRate`] / [`PsyError::NonFiniteConfig`] for a
/// bad configuration.
pub fn compute_masking(spectrum: &[f32], config: &PsyConfig) -> Result<MaskingAnalysis, PsyError> {
    if spectrum.is_empty() {
        return Err(PsyError::EmptySpectrum);
    }
    if let Some(bin) = spectrum.iter().position(|v| !v.is_finite()) {
        return Err(PsyError::NonFiniteSpectrum {
            bin,
            value: spectrum[bin],
        });
    }
    if config.sample_rate == 0 {
        return Err(PsyError::ZeroSampleRate);
    }
    if !config.full_scale_db.is_finite() || !config.threshold_offset_db.is_finite() {
        return Err(PsyError::NonFiniteConfig);
    }

    let n_half = spectrum.len();
    let rate = config.sample_rate as f32;
    // Bin k's centre frequency: the MDCT of a length-n block yields
    // n/2 bins spanning [0, rate/2); bin k sits at (k + ½) · rate / n.
    let bin_freq = |k: usize| (k as f32 + 0.5) * rate / (2.0 * n_half as f32);

    // ---- 1. Critical-band grouping (1-Bark bands). ----
    let mut bin_band = Vec::with_capacity(n_half);
    for k in 0..n_half {
        let z = bark(bin_freq(k)).max(0.0);
        bin_band.push(z as usize);
    }
    let bands = bin_band.last().copied().unwrap_or(0) + 1;

    // Band energy + flatness accumulators over bin energies.
    let mut energy = vec![0.0f64; bands];
    let mut log_sum = vec![0.0f64; bands];
    let mut count = vec![0usize; bands];
    // A tiny energy floor keeps the geometric mean defined for zero
    // bins; it is far below any audible level at every calibration.
    const ENERGY_FLOOR: f64 = 1.0e-30;
    for (k, &x) in spectrum.iter().enumerate() {
        let e = (f64::from(x) * f64::from(x)).max(ENERGY_FLOOR);
        let b = bin_band[k];
        energy[b] += e;
        log_sum[b] += e.ln();
        count[b] += 1;
    }

    // ---- 2. Per-band tonality from spectral flatness. ----
    let mut band_tonality = Vec::with_capacity(bands);
    for b in 0..bands {
        if count[b] == 0 || energy[b] <= ENERGY_FLOOR * count[b] as f64 {
            band_tonality.push(0.0);
            continue;
        }
        let arith = energy[b] / count[b] as f64;
        let geo = (log_sum[b] / count[b] as f64).exp();
        // Flatness ∈ (0, 1]; in dB it is ≤ 0. −60 dB of flatness (or
        // below) is treated as fully tonal.
        let sfm_db = 10.0 * (geo / arith).log10();
        let alpha = (sfm_db / -60.0).clamp(0.0, 1.0);
        band_tonality.push(alpha as f32);
    }

    // Band centre Bark values + band levels in dB (amplitude-calibrated:
    // full-scale line = full_scale_db).
    let mut band_center = vec![0.0f32; bands];
    let mut band_bins = vec![(usize::MAX, 0usize); bands]; // (first, last)
    for (k, &b) in bin_band.iter().enumerate() {
        let (first, _) = band_bins[b];
        if first == usize::MAX {
            band_bins[b].0 = k;
        }
        band_bins[b].1 = k;
    }
    for b in 0..bands {
        let (first, last) = band_bins[b];
        band_center[b] = if first == usize::MAX {
            b as f32 + 0.5
        } else {
            (bark(bin_freq(first)) + bark(bin_freq(last))) * 0.5
        };
    }
    let band_level_db: Vec<f32> = energy
        .iter()
        .map(|&e| config.full_scale_db + 10.0 * (e.max(ENERGY_FLOOR)).log10() as f32)
        .collect();

    // ---- 3. Spread each masker across the Bark axis (max-combine). ----
    let mut band_threshold_db = vec![f32::NEG_INFINITY; bands];
    for i in 0..bands {
        if count[i] == 0 {
            continue;
        }
        let alpha = band_tonality[i];
        let offset = alpha * tonal_offset_db(band_center[i]) + (1.0 - alpha) * NOISE_OFFSET_DB;
        let masker = band_level_db[i] - offset;
        for j in 0..bands {
            let dz = band_center[j] - band_center[i];
            let spread = if dz >= 0.0 {
                -SPREAD_UP_DB_PER_BARK * dz
            } else {
                SPREAD_DOWN_DB_PER_BARK * dz
            };
            let contrib = masker + spread;
            if contrib > band_threshold_db[j] {
                band_threshold_db[j] = contrib;
            }
        }
    }

    // ---- 4. Per-bin threshold: spread masking ∨ threshold in quiet. ----
    let mut threshold = Vec::with_capacity(n_half);
    for k in 0..n_half {
        let quiet = ath_db(bin_freq(k));
        let masked = band_threshold_db[bin_band[k]];
        let t_db = masked.max(quiet) - config.threshold_offset_db;
        // dB SPL → linear amplitude under the full-scale calibration.
        let amp = 10.0f32.powf((t_db - config.full_scale_db) / 20.0);
        // Keep the threshold strictly positive and finite.
        threshold.push(amp.max(f32::MIN_POSITIVE));
    }

    Ok(MaskingAnalysis {
        threshold,
        band_tonality,
        bin_band,
    })
}

/// Build a floor-1 envelope target from a spectrum and its masking
/// analysis: the perceptual replacement for a plain smoothed-magnitude
/// envelope.
///
/// The decoder multiplies the residue vector by the rendered floor
/// (§4.3.6), so residue-domain quantisation noise of roughly unit
/// scale surfaces as spectral noise of the floor's amplitude. Shaping
/// the floor to the masking threshold therefore shapes the noise to
/// sit at the threshold — inaudible by construction. Where the signal
/// rises **above** the threshold it must be carried (it is audible),
/// so there the envelope tracks the signal magnitude; where it falls
/// below, the envelope rides the threshold and the residue
/// `X / floor < 1` quantises toward zero cheaply, discarding only
/// masked detail:
///
/// ```text
/// envelope[k] = max(peak-smoothed |spectrum|[k], threshold[k])
/// ```
///
/// The magnitude track is a symmetric moving **maximum** over
/// `2·smooth_radius + 1` bins (peak-hold smoothing): floor 1 draws
/// straight dB-domain segments between posts, and a peak-hold envelope
/// guarantees the fitted floor never dips below a strong spectral line
/// between posts (which would blow up the residue there).
/// `smooth_radius = 0` disables smoothing.
///
/// The result is clamped into the §10.1 dB-ladder range
/// (`INVERSE_DB_TABLE[0] ..= 1.0`), the exact domain
/// [`crate::floor1_envelope::plan_floor1_envelope`] and
/// [`crate::floor1_layout::design_floor1_header`] consume.
///
/// # Errors
///
/// [`PsyError::LengthMismatch`] if `spectrum` and
/// `masking.threshold` differ in length; [`PsyError::EmptySpectrum`] /
/// [`PsyError::NonFiniteSpectrum`] as in [`compute_masking`].
pub fn plan_psy_floor_envelope(
    spectrum: &[f32],
    masking: &MaskingAnalysis,
    smooth_radius: usize,
) -> Result<Vec<f32>, PsyError> {
    if spectrum.is_empty() {
        return Err(PsyError::EmptySpectrum);
    }
    if let Some(bin) = spectrum.iter().position(|v| !v.is_finite()) {
        return Err(PsyError::NonFiniteSpectrum {
            bin,
            value: spectrum[bin],
        });
    }
    if spectrum.len() != masking.threshold.len() {
        return Err(PsyError::LengthMismatch {
            floor: spectrum.len(),
            threshold: masking.threshold.len(),
        });
    }

    let n = spectrum.len();
    let lo = crate::floor1::INVERSE_DB_TABLE[0];
    let mut envelope = Vec::with_capacity(n);
    for k in 0..n {
        let from = k.saturating_sub(smooth_radius);
        let to = (k + smooth_radius).min(n - 1);
        let mut peak = 0.0f32;
        for &x in &spectrum[from..=to] {
            peak = peak.max(x.abs());
        }
        let v = peak.max(masking.threshold[k]).clamp(lo, 1.0);
        envelope.push(v);
    }
    Ok(envelope)
}

/// Derive per-partition perceptual weights for the residue
/// rate-distortion chooser from a rendered floor curve and a masking
/// analysis.
///
/// The residue planner measures squared error in the **residue
/// domain**; the decoder multiplies that error by the floor (§4.3.6),
/// so the spectral noise at bin `k` is `err[k] · floor[k]` and its
/// audibility is `(err[k] · floor[k] / threshold[k])²` — the
/// noise-to-mask ratio. Charging partition `p`'s squared residue error
/// with the weight
///
/// ```text
/// w[p] = mean over p's bins of (floor[k] / threshold[k])²
/// ```
///
/// turns the chooser's Lagrangian into an approximate NMR-vs-bits
/// trade: partitions whose noise would surface **above** the masking
/// threshold (floor ≫ threshold) weigh heavily and attract bits;
/// partitions fully under the threshold (floor ≤ threshold) weigh
/// lightly and give bits up. The weights are normalised to mean `1.0`
/// across partitions so the caller's `lambda` keeps the same scale as
/// the unweighted chooser.
///
/// `floor` is the **rendered** floor curve (`render_curve`, the exact
/// per-bin values the decoder multiplies back in) over the whole
/// spectrum; `begin..end` is the §8.6.1 residue window (absolute bin
/// coordinates), which must be non-empty, lie inside the curves, and
/// be a multiple of `partition_size`. A floor bin of `0.0` (an
/// 'unused' floor) contributes zero weight — noise there is multiplied
/// by zero and can never be audible.
///
/// # Errors
///
/// [`PsyError::LengthMismatch`], [`PsyError::BadResidueWindow`],
/// [`PsyError::ZeroPartitionSize`], or [`PsyError::BadFloorValue`] for
/// a NaN/±∞/negative floor bin.
pub fn residue_partition_weights(
    floor: &[f32],
    masking: &MaskingAnalysis,
    begin: usize,
    end: usize,
    partition_size: u32,
) -> Result<Vec<f64>, PsyError> {
    if floor.len() != masking.threshold.len() {
        return Err(PsyError::LengthMismatch {
            floor: floor.len(),
            threshold: masking.threshold.len(),
        });
    }
    if partition_size == 0 {
        return Err(PsyError::ZeroPartitionSize);
    }
    let ps = partition_size as usize;
    if begin >= end || end > floor.len() || (end - begin) % ps != 0 {
        return Err(PsyError::BadResidueWindow {
            begin,
            end,
            len: floor.len(),
        });
    }
    if let Some(off) = floor[begin..end]
        .iter()
        .position(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(PsyError::BadFloorValue {
            bin: begin + off,
            value: floor[begin + off],
        });
    }

    let partitions = (end - begin) / ps;
    let mut weights = Vec::with_capacity(partitions);
    let mut sum = 0.0f64;
    for p in 0..partitions {
        let mut acc = 0.0f64;
        let lo = begin + p * ps;
        for (&fl, &th) in floor[lo..lo + ps]
            .iter()
            .zip(&masking.threshold[lo..lo + ps])
        {
            // threshold is strictly positive by construction.
            let r = f64::from(fl) / f64::from(th);
            acc += r * r;
        }
        let w = acc / ps as f64;
        sum += w;
        weights.push(w);
    }
    // Normalise to mean 1 so `lambda` keeps its unweighted scale. An
    // all-zero weight vector (floor identically zero over the window)
    // is left as-is: nothing in the window can ever be audible.
    if sum > 0.0 {
        let mean = sum / partitions as f64;
        for w in &mut weights {
            *w /= mean;
        }
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 44_100;

    fn cfg() -> PsyConfig {
        PsyConfig::new(RATE)
    }

    /// Frequency of bin `k` for a `n_half`-bin spectrum at `RATE`.
    fn freq(k: usize, n_half: usize) -> f32 {
        (k as f32 + 0.5) * RATE as f32 / (2.0 * n_half as f32)
    }

    /// The bin whose centre frequency is nearest `f`.
    fn bin_at(f: f32, n_half: usize) -> usize {
        (0..n_half)
            .min_by(|&a, &b| {
                (freq(a, n_half) - f)
                    .abs()
                    .partial_cmp(&(freq(b, n_half) - f).abs())
                    .unwrap()
            })
            .unwrap()
    }

    // ---------- guards ----------

    #[test]
    fn rejects_empty_spectrum() {
        assert_eq!(compute_masking(&[], &cfg()), Err(PsyError::EmptySpectrum));
    }

    #[test]
    fn rejects_non_finite_spectrum() {
        let mut s = vec![0.1f32; 64];
        s[7] = f32::NAN;
        match compute_masking(&s, &cfg()) {
            Err(PsyError::NonFiniteSpectrum { bin, .. }) => assert_eq!(bin, 7),
            other => panic!("expected NonFiniteSpectrum, got {other:?}"),
        }
    }

    #[test]
    fn rejects_zero_sample_rate() {
        let mut c = cfg();
        c.sample_rate = 0;
        assert_eq!(
            compute_masking(&[0.1; 8], &c),
            Err(PsyError::ZeroSampleRate)
        );
    }

    #[test]
    fn rejects_non_finite_config() {
        let mut c = cfg();
        c.threshold_offset_db = f32::INFINITY;
        assert_eq!(
            compute_masking(&[0.1; 8], &c),
            Err(PsyError::NonFiniteConfig)
        );
    }

    // ---------- ATH ----------

    #[test]
    fn ath_dips_in_the_3_to_4_khz_region() {
        // Hearing is most sensitive around 3–4 kHz: the ATH there must
        // be far below the ATH at 100 Hz and at 16 kHz.
        let dip = ath_db(3_400.0);
        assert!(dip < ath_db(100.0) - 10.0, "dip {dip} vs 100 Hz");
        assert!(dip < ath_db(16_000.0) - 10.0, "dip {dip} vs 16 kHz");
    }

    #[test]
    fn ath_is_finite_and_clamped_across_the_audio_band() {
        let mut f = 1.0f32;
        while f < 30_000.0 {
            let v = ath_db(f);
            assert!(v.is_finite());
            assert!((-30.0..=120.0).contains(&v), "ath({f}) = {v}");
            f *= 1.3;
        }
    }

    // ---------- threshold structure ----------

    #[test]
    fn silence_reduces_to_threshold_in_quiet() {
        // A silent frame's threshold must be exactly the per-bin ATH
        // (no masker anywhere) — strictly positive everywhere.
        let n = 256;
        let s = vec![0.0f32; n];
        let m = compute_masking(&s, &cfg()).unwrap();
        assert_eq!(m.threshold.len(), n);
        for (k, &t) in m.threshold.iter().enumerate() {
            assert!(t > 0.0, "bin {k}");
            let quiet = 10.0f32.powf((ath_db(freq(k, n)) - 96.0) / 20.0);
            // The silent band levels sit ~200+ dB below the ATH, so
            // the max() must have picked the quiet curve exactly.
            assert!(
                (t - quiet).abs() <= quiet * 1e-4,
                "bin {k}: {t} vs quiet {quiet}"
            );
        }
    }

    #[test]
    fn tone_raises_threshold_near_itself() {
        // A single loud 1 kHz line must raise the threshold in its own
        // neighbourhood far above the silent-frame threshold.
        let n = 512;
        let mut s = vec![0.0f32; n];
        let k0 = bin_at(1_000.0, n);
        s[k0] = 0.5;
        let masked = compute_masking(&s, &cfg()).unwrap();
        let quiet = compute_masking(&vec![0.0f32; n], &cfg()).unwrap();
        assert!(
            masked.threshold[k0] > quiet.threshold[k0] * 100.0,
            "at the masker: {} vs quiet {}",
            masked.threshold[k0],
            quiet.threshold[k0]
        );
        // And never below the threshold in quiet anywhere.
        for k in 0..n {
            assert!(masked.threshold[k] >= quiet.threshold[k] * 0.999, "bin {k}");
        }
    }

    #[test]
    fn masking_spreads_farther_upward_than_downward() {
        // The classic spreading asymmetry: at equal Bark distance from
        // the masker, the threshold above the masker exceeds the
        // threshold below it (once both are masking-dominated).
        let n = 1024;
        let mut s = vec![0.0f32; n];
        let f0 = 2_000.0;
        let k0 = bin_at(f0, n);
        s[k0] = 1.0;
        let m = compute_masking(&s, &cfg()).unwrap();

        // Find bins ~1.5 Bark above and below the masker.
        let z0 = bark(f0);
        let up = (0..n)
            .find(|&k| bark(freq(k, n)) >= z0 + 1.5)
            .expect("above-masker bin");
        let down = (0..n)
            .rev()
            .find(|&k| bark(freq(k, n)) <= z0 - 1.5)
            .expect("below-masker bin");
        assert!(
            m.threshold[up] > m.threshold[down] * 2.0,
            "up {} vs down {}",
            m.threshold[up],
            m.threshold[down]
        );
    }

    #[test]
    fn threshold_scales_with_the_masker() {
        // Doubling every spectral line must not lower the threshold
        // anywhere, and must raise it near the masker.
        let n = 512;
        let mut s = vec![0.0f32; n];
        let k0 = bin_at(4_000.0, n);
        s[k0] = 0.05;
        let m1 = compute_masking(&s, &cfg()).unwrap();
        s[k0] = 0.1;
        let m2 = compute_masking(&s, &cfg()).unwrap();
        for k in 0..n {
            assert!(m2.threshold[k] >= m1.threshold[k] * 0.999, "bin {k}");
        }
        assert!(m2.threshold[k0] > m1.threshold[k0] * 1.5);
    }

    #[test]
    fn tonality_separates_tone_from_noise() {
        // A band holding a single strong line is tone-like (α high); a
        // band of equal-amplitude lines is noise-like (α low).
        let n = 1024;
        let mut s = vec![0.0f32; n];
        let k_tone = bin_at(1_000.0, n);
        s[k_tone] = 0.5;
        // Fill the *whole* Bark band around 8 kHz with a flat comb (a
        // partially-filled band would read as tonal: the silent bins
        // crush the geometric mean).
        let probe = compute_masking(&s, &cfg()).unwrap();
        let noise_band = probe.bin_band[bin_at(8_000.0, n)];
        for (x, &b) in s.iter_mut().zip(&probe.bin_band) {
            if b == noise_band {
                *x = 0.05;
            }
        }
        let m = compute_masking(&s, &cfg()).unwrap();
        let tone_band = m.bin_band[k_tone];
        assert!(
            m.band_tonality[tone_band] > 0.5,
            "tone band α = {}",
            m.band_tonality[tone_band]
        );
        assert!(
            m.band_tonality[noise_band] < 0.5,
            "noise band α = {}",
            m.band_tonality[noise_band]
        );
        assert!(m.band_tonality[noise_band] < m.band_tonality[tone_band]);
    }

    #[test]
    fn tonal_masker_masks_less_than_noise_masker() {
        // Same energy, same band: a pure tone's threshold at the
        // masker bin sits lower (relative to the masker level) than a
        // flat noise band's — the tone-masking-noise offset exceeds
        // the noise-masking-tone offset.
        let n = 1024;
        let f0 = 2_000.0;
        let k0 = bin_at(f0, n);

        let mut tone = vec![0.0f32; n];
        tone[k0] = 0.5;
        let mt = compute_masking(&tone, &cfg()).unwrap();

        // Spread the same total energy across the masker's band.
        let band = mt.bin_band[k0];
        let band_bins: Vec<usize> = (0..n).filter(|&k| mt.bin_band[k] == band).collect();
        let per_bin = (0.25f32 / band_bins.len() as f32).sqrt();
        let mut noise = vec![0.0f32; n];
        for &k in &band_bins {
            noise[k] = per_bin;
        }
        let mn = compute_masking(&noise, &cfg()).unwrap();

        assert!(
            mt.threshold[k0] < mn.threshold[k0],
            "tonal threshold {} must be below noise threshold {}",
            mt.threshold[k0],
            mn.threshold[k0]
        );
    }

    #[test]
    fn threshold_offset_lowers_the_threshold() {
        let n = 256;
        let mut s = vec![0.0f32; n];
        s[bin_at(3_000.0, n)] = 0.3;
        let m0 = compute_masking(&s, &cfg()).unwrap();
        let mut c = cfg();
        c.threshold_offset_db = 12.0;
        let m1 = compute_masking(&s, &c).unwrap();
        let expect = 10.0f32.powf(-12.0 / 20.0);
        for k in 0..n {
            let ratio = m1.threshold[k] / m0.threshold[k];
            assert!(
                (ratio - expect).abs() < 1e-3,
                "bin {k}: ratio {ratio} vs {expect}"
            );
        }
    }

    // ---------- floor envelope glue ----------

    #[test]
    fn psy_envelope_tracks_signal_above_threshold_and_threshold_below() {
        let n = 512;
        let mut s = vec![0.0f32; n];
        let k0 = bin_at(1_000.0, n);
        s[k0] = 0.5;
        let m = compute_masking(&s, &cfg()).unwrap();
        let env = plan_psy_floor_envelope(&s, &m, 0).unwrap();
        assert_eq!(env.len(), n);
        // At the strong line the envelope is the signal magnitude.
        assert!((env[k0] - 0.5).abs() < 1e-6);
        // Far away (silent bins) it is the clamped threshold.
        let far = bin_at(15_000.0, n);
        let lo = crate::floor1::INVERSE_DB_TABLE[0];
        let expect = m.threshold[far].clamp(lo, 1.0);
        assert!((env[far] - expect).abs() <= expect * 1e-5);
        // Everywhere: envelope ≥ threshold (clamped) and ≥ |X| (clamped).
        for k in 0..n {
            assert!(env[k] >= m.threshold[k].clamp(lo, 1.0) * 0.999, "bin {k}");
            assert!(env[k] >= s[k].abs().clamp(lo, 1.0) * 0.999, "bin {k}");
        }
    }

    #[test]
    fn psy_envelope_peak_hold_covers_neighbours() {
        let n = 128;
        let mut s = vec![0.0f32; n];
        s[64] = 0.8;
        let m = compute_masking(&s, &cfg()).unwrap();
        let env = plan_psy_floor_envelope(&s, &m, 2).unwrap();
        // The peak-hold with radius 2 lifts bins 62..=66 to the line.
        for (k, &v) in env.iter().enumerate().take(67).skip(62) {
            assert!((v - 0.8).abs() < 1e-6, "bin {k}: {v}");
        }
    }

    #[test]
    fn psy_envelope_rejects_length_mismatch() {
        let s = vec![0.1f32; 64];
        let m = compute_masking(&s, &cfg()).unwrap();
        assert_eq!(
            plan_psy_floor_envelope(&[0.1; 32], &m, 0),
            Err(PsyError::LengthMismatch {
                floor: 32,
                threshold: 64
            })
        );
    }

    // ---------- residue weight glue ----------

    /// A synthetic masking analysis with the given threshold.
    fn masking_of(threshold: Vec<f32>) -> MaskingAnalysis {
        let n = threshold.len();
        MaskingAnalysis {
            threshold,
            band_tonality: vec![0.0],
            bin_band: vec![0; n],
        }
    }

    #[test]
    fn weights_favor_audible_partitions() {
        // Two partitions: floor ≫ threshold in the first (audible
        // noise), floor ≪ threshold in the second (masked). The first
        // weight must dominate, and the mean must be 1.
        let floor = vec![1.0f32, 1.0, 1.0, 1.0, 0.001, 0.001, 0.001, 0.001];
        let m = masking_of(vec![0.01f32; 8]);
        let w = residue_partition_weights(&floor, &m, 0, 8, 4).unwrap();
        assert_eq!(w.len(), 2);
        assert!(w[0] > w[1] * 100.0, "w = {w:?}");
        let mean = (w[0] + w[1]) / 2.0;
        assert!((mean - 1.0).abs() < 1e-9);
    }

    #[test]
    fn uniform_ratio_yields_unit_weights() {
        // floor/threshold constant → every weight exactly 1.
        let floor = vec![0.2f32; 12];
        let m = masking_of(vec![0.05f32; 12]);
        let w = residue_partition_weights(&floor, &m, 0, 12, 4).unwrap();
        for &wi in &w {
            assert!((wi - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn zero_floor_window_keeps_zero_weights() {
        let floor = vec![0.0f32; 8];
        let m = masking_of(vec![0.01f32; 8]);
        let w = residue_partition_weights(&floor, &m, 0, 8, 4).unwrap();
        assert_eq!(w, vec![0.0, 0.0]);
    }

    #[test]
    fn weights_respect_the_residue_window() {
        // Only bins inside [begin, end) contribute.
        let mut floor = vec![100.0f32; 12];
        floor[4..8].fill(1.0);
        let m = masking_of(vec![1.0f32; 12]);
        let w = residue_partition_weights(&floor, &m, 4, 8, 4).unwrap();
        assert_eq!(w.len(), 1);
        assert!((w[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn weight_guards_fire() {
        let m = masking_of(vec![0.01f32; 8]);
        assert_eq!(
            residue_partition_weights(&[0.1; 4], &m, 0, 4, 2),
            Err(PsyError::LengthMismatch {
                floor: 4,
                threshold: 8
            })
        );
        assert_eq!(
            residue_partition_weights(&[0.1; 8], &m, 0, 8, 0),
            Err(PsyError::ZeroPartitionSize)
        );
        assert_eq!(
            residue_partition_weights(&[0.1; 8], &m, 0, 6, 4),
            Err(PsyError::BadResidueWindow {
                begin: 0,
                end: 6,
                len: 8
            })
        );
        assert_eq!(
            residue_partition_weights(&[0.1; 8], &m, 4, 4, 4),
            Err(PsyError::BadResidueWindow {
                begin: 4,
                end: 4,
                len: 8
            })
        );
        let mut bad = vec![0.1f32; 8];
        bad[3] = -1.0;
        match residue_partition_weights(&bad, &m, 0, 8, 4) {
            Err(PsyError::BadFloorValue { bin, .. }) => assert_eq!(bin, 3),
            other => panic!("expected BadFloorValue, got {other:?}"),
        }
    }
}

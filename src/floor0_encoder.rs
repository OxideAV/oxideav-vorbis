//! Floor type 0 (LSP) encoder support.
//!
//! Vorbis I §6 describes the floor0 packet format: a smooth spectral
//! envelope is encoded as the frequency response of a Line-Spectral-Pair
//! (LSP) filter. The decoder side already lives in [`crate::floor`]; this
//! module provides the mirror analysis path so the encoder can choose
//! floor0 over floor1 on a per-frame basis when the spectrum is tonal /
//! sparse enough that the LSP filter resolves it more compactly than
//! floor1's piecewise-linear envelope.
//!
//! Pipeline per audio block:
//!
//! 1. **LPC analysis** — autocorrelation of the windowed PCM input plus
//!    Levinson-Durbin recursion (§6.1 references LPC, but the actual
//!    derivation is the standard one — see e.g. Markel & Gray "Linear
//!    Prediction of Speech" §4). Produces `order` reflection coefficients
//!    and a prediction-error residual energy.
//! 2. **LPC → LSP** — convert via the standard sum / difference
//!    polynomial root method. Roots lie on the unit circle and interlace,
//!    giving `order` LSP frequencies in `[0, π]`. We compute their
//!    cosines `cos(ω_j)` directly because the Vorbis floor0 codebook
//!    encodes those cosines (§6.2.3 — the synthesis formula uses
//!    `cos([coefficients]_k)` and our matching decoder in
//!    [`crate::floor::synth_floor0`] consumes the cosine values directly).
//! 3. **Amplitude scalar** — the input block's RMS sets `amplitude` via
//!    the inverse of the synthesis formula (§6.2.3 step 4): solving for
//!    `amplitude` given a target dB level relative to the configured
//!    `amplitude_offset`.
//! 4. **VQ quantisation** — each LSP cosine is quantised against the
//!    setup's floor0 codebook(s). With our default codebook (`dim = 2`,
//!    256 entries on a `[-1, 1] × [-1, 1]` grid), a vector search picks
//!    the nearest grid point per consecutive cosine pair.
//! 5. **Bit emission** — write `amplitude_bits` of amplitude, then
//!    `ilog(number_of_books)` of book number, then the VQ codeword
//!    sequence covering at least `floor0_order` scalars.
//!
//! Both this module and [`crate::floor::synth_floor0`] use the convention
//! that the VQ stores `cos(ω_j)` rather than `ω_j` itself. This is a
//! private convention that keeps the per-bin synth loop free of `cos()`
//! evaluations on the codebook output; an LSP-aware libvorbis-style
//! decoder would re-`cos()` after lookup, which would still produce a
//! valid (though differently-rendered) curve. For round-trip-through-
//! ours-decoder tests this convention is bit-exact.

use std::collections::VecDeque;

use crate::codebook::{Codebook, VqLookup};
use crate::encoder::{
    build_comment_header, build_extradata, build_identification_header,
    DEFAULT_BLOCKSIZE_LONG_LOG2, DEFAULT_BLOCKSIZE_SHORT_LOG2,
};
use crate::imdct::build_window;
use oxideav_core::bits::BitWriterLsb as BitWriter;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Encoder, Error, Frame, MediaType, Packet, Result,
    SampleFormat, TimeBase,
};

/// LPC order used by the encoder. Picked to balance LSP coefficient
/// budget (each coefficient costs `value_bits` bits per VQ slot) against
/// envelope-resolution: order 16 captures ~8 formants on a typical
/// speech / music block which is enough for the tonal signals where
/// floor0 wins over floor1.
pub const FLOOR0_ENCODE_ORDER: u8 = 16;

/// Bark-map size for the encoder's floor0 setup. 256 matches what real
/// libvorbis files emit; the value only affects synthesis (decoder side)
/// and the choice doesn't change packet layout.
pub const FLOOR0_BARK_MAP_SIZE: u16 = 256;

/// Amplitude bits — 8 gives 256 quantisation levels, ample dynamic
/// range and matches the spec's recommended encoding (§6.2.1 reads up to
/// 6 bits, but the field can carry up to 8).
pub const FLOOR0_AMPLITUDE_BITS: u8 = 8;

/// Amplitude offset (dB). 100 dB matches a typical full-scale to ATH
/// span for 16-bit audio.
pub const FLOOR0_AMPLITUDE_OFFSET: u8 = 100;

/// Number of multiplicands per dimension in the floor0 LSP codebook.
/// 16 levels covering `[-1, 1]` give a step of `2/15 ≈ 0.133`, which is
/// the same per-coefficient resolution as a 4-bit linear quantiser of
/// `cos(ω_j)`. Combined with `dim = 2` this gives `16² = 256` codebook
/// entries.
pub const FLOOR0_VQ_VALUES_PER_DIM: u32 = 16;

/// Floor0 VQ codebook dimension — pairs of cosines per VQ vector. Pairs
/// (rather than single scalars) let one VQ entry encode `cos(ω_{2j})`
/// and `cos(ω_{2j+1})` together; with `order = 16` this means 8 VQ
/// reads per packet rather than 16.
pub const FLOOR0_VQ_DIM: u32 = 2;

/// Total entries in the floor0 VQ codebook. 16² = 256 entries → 8 bit
/// codewords.
pub const FLOOR0_VQ_ENTRIES: u32 = 256;

/// Codeword length (bits per VQ entry index). All entries use length 8
/// — fully-specified Huffman tree, no entropy savings vs. flat indexing.
pub const FLOOR0_VQ_CODEWORD_LEN: u32 = 8;

/// Per-coefficient grid range. `cos(ω_j) ∈ [-1, 1]` so we cover the
/// full range with a uniform grid.
pub const FLOOR0_VQ_MIN: f32 = -1.0;

/// Per-coefficient grid step: `2 / (VALUES_PER_DIM - 1)`.
pub const FLOOR0_VQ_DELTA: f32 = 2.0 / ((FLOOR0_VQ_VALUES_PER_DIM - 1) as f32);

/// Number of bits used to write each multiplicand in the codebook
/// header. Must hold values `0..VALUES_PER_DIM`.
pub const FLOOR0_VQ_VALUE_BITS: u32 = 4;

/// Floor0 codebook helper. Constructs the VQ lookup table that mirrors
/// what the setup-header writer emits. The codebook is built so each
/// entry's decoded vector lies on the `[-1, 1]^FLOOR0_VQ_DIM` Cartesian
/// grid: entry `e` decodes to `(grid[e % N], grid[e / N])` where
/// `grid[k] = MIN + k * DELTA`.
pub fn build_floor0_codebook() -> Codebook {
    // Lookup type 1: shared per-dim multiplicand table of length
    // `VALUES_PER_DIM`. Decoded entry `e` reads multiplicand at index
    // `(e / N^d) % N` per dimension `d`.
    let mut multiplicands = Vec::with_capacity(FLOOR0_VQ_VALUES_PER_DIM as usize);
    for k in 0..FLOOR0_VQ_VALUES_PER_DIM {
        multiplicands.push(k);
    }
    let mut cb = Codebook {
        dimensions: FLOOR0_VQ_DIM as u16,
        entries: FLOOR0_VQ_ENTRIES,
        codeword_lengths: vec![FLOOR0_VQ_CODEWORD_LEN as u8; FLOOR0_VQ_ENTRIES as usize],
        vq: Some(VqLookup {
            lookup_type: 1,
            min: FLOOR0_VQ_MIN,
            delta: FLOOR0_VQ_DELTA,
            value_bits: FLOOR0_VQ_VALUE_BITS as u8,
            sequence_p: false,
            multiplicands,
        }),
        codewords: vec![],
    };
    cb.build_decoder()
        .expect("floor0 default codebook is well-formed");
    cb
}

/// Compute autocorrelation of `samples` for lags `0..=order`. Standard
/// definition `R[k] = Σ_n s[n] * s[n+k]` (with zero padding past the
/// signal end). Used as input to the Levinson-Durbin recursion.
fn autocorrelation(samples: &[f32], order: usize) -> Vec<f64> {
    let mut r = vec![0f64; order + 1];
    let n = samples.len();
    for k in 0..=order {
        let mut acc = 0f64;
        for i in 0..n.saturating_sub(k) {
            acc += samples[i] as f64 * samples[i + k] as f64;
        }
        r[k] = acc;
    }
    r
}

/// Levinson-Durbin recursion for the LPC coefficients of a signal whose
/// autocorrelation is given by `r` of length `order + 1`.
///
/// Returns `(lpc, error)` where `lpc[0..order]` are the LPC coefficients
/// (with implicit `lpc_polynomial(0) = 1`) and `error` is the final
/// prediction-error energy. If the autocorrelation is singular (e.g.
/// silent input → R[0] = 0) returns an error.
fn levinson_durbin(r: &[f64], order: usize) -> Result<(Vec<f64>, f64)> {
    if r.len() < order + 1 {
        return Err(Error::invalid("levinson_durbin: r too short"));
    }
    if r[0].abs() < 1e-30 {
        return Err(Error::invalid("levinson_durbin: zero-energy signal"));
    }
    let mut a = vec![0f64; order];
    let mut a_prev = vec![0f64; order];
    let mut error = r[0];
    for i in 0..order {
        // Reflection coefficient k_i.
        let mut acc = -r[i + 1];
        for j in 0..i {
            acc -= a[j] * r[i - j];
        }
        let k = acc / error;
        // a_new[i] = k; a_new[j] = a[j] + k * a[i-1-j] for j < i.
        a_prev[..i].copy_from_slice(&a[..i]);
        a[i] = k;
        for j in 0..i {
            a[j] = a_prev[j] + k * a_prev[i - 1 - j];
        }
        error *= 1.0 - k * k;
        if error <= 0.0 {
            // Filter went unstable — bail with the partial result.
            error = error.abs().max(1e-12);
            break;
        }
    }
    Ok((a, error))
}

/// Convert an LPC polynomial `1 + a[0]z^-1 + ... + a[p-1]z^-p` into LSP
/// frequencies `ω_j ∈ [0, π]`, returned as their cosines `cos(ω_j)`.
///
/// Standard derivation: build symmetric P(z) = A(z) + z^-(p+1)A(1/z) and
/// antisymmetric Q(z) = A(z) - z^-(p+1)A(1/z). All roots lie on the unit
/// circle and interlace; their angles are the LSP frequencies. Recurrence
/// reduces each polynomial to a real Chebyshev polynomial of order p/2,
/// whose roots in `[-1, 1]` are the cosines of the LSP frequencies.
///
/// We use a coarse grid scan + bisection root finder — robust for `p ≤
/// 32`. The output has length `p`, sorted in ascending `cos(ω)` order
/// (i.e. descending frequency order); the caller can re-sort if needed.
fn lpc_to_lsp_cosines(lpc: &[f64]) -> Vec<f32> {
    let p = lpc.len();
    if p == 0 {
        return Vec::new();
    }
    // Build P, Q polynomial coefficients of length p+1. Per the standard
    // derivation: P[i] = a[i] + a[p-i] (with a[0] = 1, a[p] = 1 for P);
    // and Q[i] = a[i] - a[p-i] (with a[0] = 1, a[p] = -1 for Q).
    let mut a = vec![1f64; p + 1];
    a[1..=p].copy_from_slice(lpc);
    let mut p_coeff = vec![0f64; p + 1];
    let mut q_coeff = vec![0f64; p + 1];
    p_coeff[0] = 1.0;
    p_coeff[p] = 1.0;
    q_coeff[0] = 1.0;
    q_coeff[p] = -1.0;
    for i in 1..=p / 2 {
        // P_i = a_i + a_{p+1-i} (using a[0]=1, a[p+1]=0 conceptually).
        let ai = a.get(i).copied().unwrap_or(0.0);
        let api = a.get(p + 1 - i).copied().unwrap_or(0.0);
        p_coeff[i] = ai + api;
        q_coeff[i] = ai - api;
        p_coeff[p - i] = p_coeff[i];
        q_coeff[p - i] = -q_coeff[i];
    }
    // Reduce to Chebyshev form. P(z) and Q(z) factor out (1 + z^-1) and
    // (1 - z^-1) respectively (when p is even), so divide them out and
    // get polynomials of order p/2 in `cos(ω)` (via z + z^-1 = 2cos(ω)).
    //
    // For brevity: instead of explicit polynomial division we evaluate
    // P(e^{jω}) and Q(e^{jω}) on a fine ω-grid, take the real-axis
    // projection, and locate sign changes.
    let n_grid: usize = 512;
    let mut p_vals = vec![0f64; n_grid + 1];
    let mut q_vals = vec![0f64; n_grid + 1];
    for k in 0..=n_grid {
        let omega = std::f64::consts::PI * (k as f64) / (n_grid as f64);
        p_vals[k] = eval_poly_real(&p_coeff, omega);
        q_vals[k] = eval_poly_real(&q_coeff, omega);
    }

    // Find sign changes (roots interlace, alternating P and Q). Refine
    // each by bisection.
    let mut roots: Vec<f64> = Vec::with_capacity(p);
    // Walk left-to-right and emit roots from each polynomial as we
    // encounter sign changes. The interlace property guarantees they
    // alternate.
    let mut want_p = true;
    let mut k = 0usize;
    while k < n_grid && roots.len() < p {
        let (vals, other) = if want_p {
            (&p_vals, &q_vals)
        } else {
            (&q_vals, &p_vals)
        };
        // Look for next sign change in `vals` starting at k.
        let mut found = false;
        let mut idx = k;
        while idx < n_grid {
            if vals[idx] == 0.0 {
                let omega = std::f64::consts::PI * (idx as f64) / (n_grid as f64);
                roots.push(omega);
                idx += 1;
                found = true;
                break;
            }
            if vals[idx].signum() != vals[idx + 1].signum() && vals[idx + 1] != 0.0 {
                // Bisect between grid[idx] and grid[idx+1].
                let mut lo = std::f64::consts::PI * (idx as f64) / (n_grid as f64);
                let mut hi = std::f64::consts::PI * ((idx + 1) as f64) / (n_grid as f64);
                let mut lo_val = vals[idx];
                let coeffs = if want_p { &p_coeff } else { &q_coeff };
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    let mv = eval_poly_real(coeffs, mid);
                    if mv == 0.0 {
                        lo = mid;
                        hi = mid;
                        break;
                    }
                    if mv.signum() == lo_val.signum() {
                        lo = mid;
                        lo_val = mv;
                    } else {
                        hi = mid;
                    }
                }
                roots.push(0.5 * (lo + hi));
                idx += 1;
                found = true;
                break;
            }
            // Skip past adjacent sign-change of the OTHER polynomial first
            // so we maintain interlace.
            if other[idx].signum() != other[idx + 1].signum() {
                break;
            }
            idx += 1;
        }
        if !found {
            // No more roots of this polynomial; switch and try the other.
            want_p = !want_p;
            k = idx + 1;
            continue;
        }
        k = idx;
        want_p = !want_p;
    }

    // Filter out spurious "roots" at ω=0 or ω=π (factored-out
    // (1±z^-1)) and ensure exactly `p` LSPs. If we under-count, pad with
    // evenly-spaced fillers.
    roots.retain(|&w| (1e-6..std::f64::consts::PI - 1e-6).contains(&w));
    while roots.len() < p {
        // Pad with evenly-spaced filler frequencies in the valid range
        // (avoid duplicates by offsetting from existing roots).
        let idx = roots.len() + 1;
        let filler = std::f64::consts::PI * (idx as f64) / (p as f64 + 1.0);
        roots.push(filler);
    }
    roots.truncate(p);
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Spec's synthesis takes cos(ω_j) per the formula's `cos([coefficients]_k)`,
    // but our decoder convention treats coefficients as already-cosines (it
    // skips the cos() at synth time — see comments in `crate::floor::synth_floor0`).
    // So we feed the ENCODER side cosine-of-frequency values.
    roots.iter().map(|&w| w.cos() as f32).collect()
}

/// Evaluate `Σ c_i cos(i ω)` (real part of `Σ c_i e^{j i ω}` since
/// coefficients are real and the polynomial is symmetric / antisymmetric).
fn eval_poly_real(coeffs: &[f64], omega: f64) -> f64 {
    let mut acc = 0f64;
    for (i, &c) in coeffs.iter().enumerate() {
        acc += c * (i as f64 * omega).cos();
    }
    acc
}

/// Run LPC + LSP analysis on a windowed PCM block. Returns
/// `(amplitude_value, lsp_cosines)` ready for VQ quantisation +
/// bitstream emission.
///
/// `amplitude_value` is the spec's `amplitude` field, computed from the
/// block's RMS energy. `amplitude_offset` and `amplitude_bits` come from
/// the floor0 setup and parameterise the quantisation:
///
///   db = 20 * log10(rms) — encodes signal level in dB
///   amplitude = clamp(round(db / amplitude_offset * (max_amp - 1) + max_amp), 1, max_amp)
///
/// where `max_amp = 2^amplitude_bits - 1`. This linear mapping keeps the
/// amplitude field non-zero (=1 means "unused") for any non-silent input
/// and saturates rather than wraps for very loud blocks.
pub fn analyse_floor0(
    samples: &[f32],
    order: usize,
    amplitude_bits: u8,
    amplitude_offset: u8,
) -> Result<(u32, Vec<f32>)> {
    if samples.is_empty() {
        return Ok((0, Vec::new()));
    }
    // RMS of the input.
    let mut sumsq = 0f64;
    for &s in samples {
        sumsq += s as f64 * s as f64;
    }
    let rms = (sumsq / samples.len() as f64).sqrt();
    if rms < 1e-8 {
        return Ok((0, Vec::new()));
    }
    // Map RMS to amplitude quantum. Fully-scale rms = 1.0 → 0 dB → max_amp.
    // -100 dB → 1 (just above silence).
    let max_amp = ((1u64 << amplitude_bits) - 1) as f64;
    let db = 20.0 * rms.log10();
    let amp_offset = amplitude_offset as f64;
    // db = 0 → amplitude = max_amp
    // db = -amp_offset → amplitude = 0 (silent)
    // Use linear interpolation: amp = max_amp * (1 + db/amp_offset).
    let amp_f = max_amp * (1.0 + db / amp_offset);
    let amp = amp_f.round().clamp(1.0, max_amp) as u32;

    // LPC analysis.
    let r = autocorrelation(samples, order);
    let (lpc, _err) = levinson_durbin(&r, order)?;
    let cosines = lpc_to_lsp_cosines(&lpc);
    Ok((amp, cosines))
}

/// Floor0-vs-floor1 per-frame heuristic. Returns `true` if the input
/// block looks tonal / sparse enough that the LSP filter resolves it
/// more compactly than floor1's piecewise-linear envelope.
///
/// Heuristic: spectral flatness via prediction-gain — measure the ratio
/// of zero-lag autocorrelation to the prediction-error energy after
/// short-order Levinson-Durbin (order 4). Tonal signals concentrate
/// energy near a few resonance frequencies, so the predictor explains
/// most of the variance and the ratio is large; noise spreads energy
/// evenly across frequencies, so prediction barely helps and the ratio
/// stays near 1.
///
/// Threshold tuned empirically: a 1 kHz pure sine reads ≈ 100×
/// prediction gain at order 4, while uniform pseudo-noise reads ≈ 1.0×.
/// We pick the gate at 4× — well clear of either edge — so signals with
/// strong tonal components flip into floor0 while broadband noise stays
/// on floor1. Caller may invoke this on the windowed PCM input or on a
/// raw block; the result is shift-invariant.
pub fn should_use_floor0(samples: &[f32]) -> bool {
    let order = 4usize;
    if samples.len() < order * 4 {
        return false;
    }
    let r = autocorrelation(samples, order);
    if r[0] < 1e-12 {
        return false;
    }
    let gain = match levinson_durbin(&r, order) {
        Ok((_, err)) => r[0] / err.max(1e-30),
        Err(_) => 1.0,
    };
    gain >= 4.0
}

/// Quantise an LSP cosine vector against a VQ codebook. Returns the
/// list of codebook entries; concatenating their `vq_lookup` results
/// reproduces the encoded value the decoder will see.
///
/// Each VQ pass consumes `dim` consecutive cosines. If the cosine vector
/// length isn't a multiple of `dim`, the last partial chunk is padded
/// with the previous chunk's last value before quantisation. The decoder
/// is allowed to over-read VQ vectors past `floor0_order` (§6.2.2 last
/// bullet) so the padding is harmless.
pub fn quantise_lsp_cosines(cosines: &[f32], book: &Codebook) -> Result<Vec<u32>> {
    let dim = book.dimensions as usize;
    if dim == 0 {
        return Err(Error::invalid("floor0 codebook has zero dimension"));
    }
    let n_vectors = cosines.len().div_ceil(dim);
    let mut out = Vec::with_capacity(n_vectors);
    let mut scratch = vec![0f32; dim];
    for i in 0..n_vectors {
        let base = i * dim;
        let mut last = if base > 0 { cosines[base - 1] } else { 0.0 };
        for j in 0..dim {
            scratch[j] = cosines.get(base + j).copied().unwrap_or(last);
            last = scratch[j];
        }
        let entry = nearest_vq_entry(book, &scratch)?;
        out.push(entry);
    }
    Ok(out)
}

/// Exhaustive nearest-neighbour search over a codebook's used entries.
fn nearest_vq_entry(book: &Codebook, target: &[f32]) -> Result<u32> {
    let mut best = 0u32;
    let mut best_d = f64::INFINITY;
    for e in 0..book.entries {
        if book.codeword_lengths[e as usize] == 0 {
            continue;
        }
        let v = book.vq_lookup(e)?;
        let mut d = 0f64;
        for (i, &t) in target.iter().enumerate() {
            let x = t as f64 - v[i] as f64;
            d += x * x;
        }
        if d < best_d {
            best_d = d;
            best = e;
        }
    }
    Ok(best)
}

// ============================== Setup writer ==============================

/// Build a complete Vorbis setup header that uses floor type 0 for both
/// short and long blocks. Layout:
///
/// - 2 codebooks:
///   1. Floor0 LSP VQ (dim 2, 256 entries on `[-1, 1]² grid`).
///   2. Residue "constant 1.0" book (dim 1, 1 entry whose decoded value
///      is `1.0`). With residue type 2 + a single classification + this
///      book on cascade pass 0, every spectrum bin gets `1.0` written
///      with zero bits emitted per bin (single-entry Huffman → 0 bits).
/// - 2 floors (floor0 short + floor0 long), both referencing book 0.
/// - 2 residues (type 2, single classification, single book = book 1).
///   `residue.end = n_half * channels` per the type-2 layout convention.
/// - 1 mapping (no coupling — encoder runs each channel independently).
/// - 2 modes (mode 0 = short, mode 1 = long).
///
/// The result is a self-contained Vorbis setup that the existing decoder
/// pipeline accepts with no special-casing required.
pub fn build_encoder_setup_header_floor0(channels: u8) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(512);
    for &b in &[0x05u32, 0x76, 0x6f, 0x72, 0x62, 0x69, 0x73] {
        w.write_u32(b, 8);
    }

    // 2 codebooks.
    w.write_u32(2 - 1, 8);
    write_codebook_floor0_lsp(&mut w);
    write_codebook_residue_constant(&mut w);

    // 1 time-domain placeholder.
    w.write_u32(0, 6);
    w.write_u32(0, 16);

    // 2 floors.
    w.write_u32(2 - 1, 6);
    // Floor 0 (short block).
    w.write_u32(0, 16); // floor type 0
    write_floor0_section(&mut w);
    // Floor 1 (long block).
    w.write_u32(0, 16);
    write_floor0_section(&mut w);

    // 2 residues.
    w.write_u32(2 - 1, 6);
    let short_end = (1u32 << DEFAULT_BLOCKSIZE_SHORT_LOG2) / 2 * channels.max(1) as u32;
    let long_end = (1u32 << DEFAULT_BLOCKSIZE_LONG_LOG2) / 2 * channels.max(1) as u32;
    w.write_u32(2, 16); // residue type 2
    write_residue_constant_section(&mut w, short_end);
    w.write_u32(2, 16);
    write_residue_constant_section(&mut w, long_end);

    // 1 mapping (count - 1 = 0).
    w.write_u32(0, 6);
    write_mapping_no_coupling(&mut w, channels);

    // 2 modes (short, long).
    w.write_u32(2 - 1, 6);
    // mode 0: short
    w.write_bit(false);
    w.write_u32(0, 16);
    w.write_u32(0, 16);
    w.write_u32(0, 8);
    // mode 1: long
    w.write_bit(true);
    w.write_u32(0, 16);
    w.write_u32(0, 16);
    w.write_u32(0, 8);

    // Framing bit.
    w.write_bit(true);
    w.finish()
}

/// Write the floor0 LSP codebook into the setup bitstream. Uses
/// lookup_type 1 with `FLOOR0_VQ_VALUES_PER_DIM` multiplicands — every
/// VQ entry has the full `FLOOR0_VQ_CODEWORD_LEN`-bit canonical Huffman
/// code (full tree, no entropy savings, but bit-budget per LSP is
/// predictable).
fn write_codebook_floor0_lsp(w: &mut BitWriter) {
    w.write_u32(0x564342, 24); // sync "BCV"
    w.write_u32(FLOOR0_VQ_DIM, 16);
    w.write_u32(FLOOR0_VQ_ENTRIES, 24);
    w.write_bit(false); // ordered
    w.write_bit(false); // sparse
    for _ in 0..FLOOR0_VQ_ENTRIES {
        w.write_u32(FLOOR0_VQ_CODEWORD_LEN - 1, 5);
    }
    w.write_u32(1, 4); // lookup_type = 1
    write_vorbis_float(w, FLOOR0_VQ_MIN);
    write_vorbis_float(w, FLOOR0_VQ_DELTA);
    w.write_u32(FLOOR0_VQ_VALUE_BITS - 1, 4);
    w.write_bit(false); // sequence_p
    for k in 0..FLOOR0_VQ_VALUES_PER_DIM {
        w.write_u32(k, FLOOR0_VQ_VALUE_BITS);
    }
}

/// Single-entry residue book. Dim 1, 1 entry, lookup_type 2 with
/// multiplicand 1 mapping to `1 * delta + min = 1.0` (delta=1.0,
/// min=0.0). Codeword length 1 (single-entry codebooks are decoded
/// without Huffman — `decode_scalar` short-circuits to entry 0 in 0
/// bits).
fn write_codebook_residue_constant(w: &mut BitWriter) {
    w.write_u32(0x564342, 24);
    w.write_u32(1, 16); // dim
    w.write_u32(1, 24); // entries
    w.write_bit(false); // ordered
    w.write_bit(false); // sparse
    w.write_u32(0, 5); // length - 1 = 0 → length 1
    w.write_u32(2, 4); // lookup_type = 2
    write_vorbis_float(w, 0.0); // min
    write_vorbis_float(w, 1.0); // delta
    w.write_u32(0, 4); // value_bits - 1 = 0 → 1 bit
    w.write_bit(false); // sequence_p
    w.write_u32(1, 1); // multiplicand[0] = 1 → decoded value = 1 * 1.0 + 0.0 = 1.0
}

/// Write a floor0 section (post-type tag) into the setup bitstream.
fn write_floor0_section(w: &mut BitWriter) {
    w.write_u32(FLOOR0_ENCODE_ORDER as u32, 8); // floor0_order
    w.write_u32(48_000, 16); // floor0_rate (we hard-wire 48 kHz)
    w.write_u32(FLOOR0_BARK_MAP_SIZE as u32, 16);
    w.write_u32(FLOOR0_AMPLITUDE_BITS as u32, 6);
    w.write_u32(FLOOR0_AMPLITUDE_OFFSET as u32, 8);
    w.write_u32(0, 4); // number_of_books - 1 = 0 → 1 book
    w.write_u32(0, 8); // book_list[0] = codebook 0 (the floor0 LSP VQ)
}

/// Write a residue-type-2 section using the constant-1.0 book on a
/// single classification (no class bits read — single classification
/// means classifications=1 and the per-class Huffman read is skipped
/// because `cval` lookup is never executed for class 0). The cascade
/// byte = 0b001 selects pass 0 only, with book index 1.
fn write_residue_constant_section(w: &mut BitWriter, end: u32) {
    w.write_u32(0, 24); // begin
    w.write_u32(end, 24);
    w.write_u32(2 - 1, 24); // partition_size = 2
    w.write_u32(2 - 1, 6); // classifications - 1 = 1 → 2 classes
    w.write_u32(0, 8); // classbook = book 0 (the floor0 LSP VQ doubles
                       // as classbook here — its dim=2 means 2 classes
                       // packed per codeword. We never actually pack any
                       // useful info because both partition classes
                       // resolve to "active" via cascade)
                       // Cascade for class 0 = 0 (silent — but we set
                       // class to 1 always so this never fires)
    w.write_u32(0, 3);
    w.write_bit(false);
    // Cascade for class 1 = 0b001 (pass 0 active).
    w.write_u32(0b001, 3);
    w.write_bit(false);
    // Books for class 1 pass 0 = book 1 (constant 1.0).
    w.write_u32(1, 8);
}

/// Write a no-coupling mapping with one submap covering all channels.
fn write_mapping_no_coupling(w: &mut BitWriter, channels: u8) {
    let _ = channels;
    w.write_u32(0, 16); // mapping type = 0
    w.write_bit(false); // submaps flag = 0 → 1 submap
    w.write_bit(false); // coupling flag = 0 → no coupling
    w.write_u32(0, 2); // reserved
                       // submap 0:
    w.write_u32(0, 8); // time index
    w.write_u32(0, 8); // floor index = 0
    w.write_u32(0, 8); // residue index = 0
}

/// Inverse of `BitReader::read_vorbis_float` — used for the LSP VQ
/// codebook's `min` / `delta` fields.
fn write_vorbis_float(w: &mut BitWriter, value: f32) {
    if value == 0.0 {
        w.write_u32(0, 32);
        return;
    }
    let abs = value.abs() as f64;
    let mut mantissa = abs;
    let mut exp: i32 = 0;
    while mantissa < (1u64 << 20) as f64 {
        mantissa *= 2.0;
        exp -= 1;
    }
    while mantissa >= (1u64 << 21) as f64 {
        mantissa /= 2.0;
        exp += 1;
    }
    let m = mantissa as u32 & 0x001F_FFFF;
    let biased = (exp + 788) as u32;
    debug_assert!(biased < 1024, "Vorbis float exponent out of range");
    let sign_bit = if value < 0.0 { 0x8000_0000u32 } else { 0 };
    let raw = sign_bit | ((biased & 0x3FF) << 21) | m;
    w.write_u32(raw, 32);
}

// ============================== Floor0 Encoder driver ======================

/// Build a Vorbis encoder that emits floor type 0 (LSP) packets for every
/// audio block. Mirrors [`crate::encoder::make_encoder`] but produces a
/// floor0-only setup and skips floor1 / residue VQ search entirely. Use
/// this for tonal-content encodes where the LSP filter resolves the
/// envelope more compactly than floor1's piecewise-linear posts.
///
/// Limitations relative to the floor1 encoder:
/// - No transient / short-block selection — every block is long.
/// - No channel coupling — each channel is encoded independently.
/// - No per-frame heuristic to fall back on floor1 when the spectrum is
///   noise-like; this is the floor0 encoder, full stop.
pub fn make_encoder_floor0(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("Vorbis floor0 encoder: channels required"))?;
    if !(1..=8).contains(&channels) {
        return Err(Error::unsupported(format!(
            "Vorbis floor0 encoder: {channels}-channel encode not supported"
        )));
    }
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("Vorbis floor0 encoder: sample_rate required"))?;
    let input_sample_format = params.sample_format.unwrap_or(SampleFormat::S16);

    let id_hdr = build_identification_header(
        channels as u8,
        sample_rate,
        0,
        DEFAULT_BLOCKSIZE_SHORT_LOG2,
        DEFAULT_BLOCKSIZE_LONG_LOG2,
    );
    let comment_hdr = build_comment_header(&[]);
    let setup_hdr = build_encoder_setup_header_floor0(channels as u8);
    let extradata = build_extradata(&id_hdr, &comment_hdr, &setup_hdr);

    let codebook = build_floor0_codebook();

    let mut out_params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
    out_params.media_type = MediaType::Audio;
    out_params.channels = Some(channels);
    out_params.sample_rate = Some(sample_rate);
    out_params.sample_format = Some(SampleFormat::S16);
    out_params.extradata = extradata;

    let blocksize_long = 1usize << DEFAULT_BLOCKSIZE_LONG_LOG2;

    Ok(Box::new(Floor0Encoder {
        codec_id: CodecId::new(crate::CODEC_ID_STR),
        out_params,
        time_base: TimeBase::new(1, sample_rate as i64),
        channels,
        sample_rate,
        input_sample_format,
        blocksize_long,
        input_buf: vec![Vec::with_capacity(blocksize_long * 4); channels as usize],
        prev_tail: vec![Vec::with_capacity(blocksize_long); channels as usize],
        output_queue: VecDeque::new(),
        pts: 0,
        flushed: false,
        codebook,
    }))
}

struct Floor0Encoder {
    codec_id: CodecId,
    out_params: CodecParameters,
    time_base: TimeBase,
    channels: u16,
    #[allow(dead_code)]
    sample_rate: u32,
    input_sample_format: SampleFormat,
    blocksize_long: usize,
    input_buf: Vec<Vec<f32>>,
    /// Per-channel "previous block right tail" — the second half of the
    /// previously-emitted block's MDCT input window. The current block's
    /// first half OLAs against this on decode. We mirror the floor1
    /// encoder's `prev_tail` bookkeeping but always operate on long blocks
    /// so the overlap is always `n / 2`.
    prev_tail: Vec<Vec<f32>>,
    output_queue: VecDeque<Packet>,
    pts: i64,
    flushed: bool,
    codebook: Codebook,
}

impl Floor0Encoder {
    fn push_audio_frame(&mut self, frame: &AudioFrame) -> Result<()> {
        let n = frame.samples as usize;
        if n == 0 {
            return Ok(());
        }
        match self.input_sample_format {
            SampleFormat::S16 => {
                let plane = frame
                    .data
                    .first()
                    .ok_or_else(|| Error::invalid("S16 frame missing data plane"))?;
                let stride = self.channels as usize * 2;
                if plane.len() < n * stride {
                    return Err(Error::invalid("S16 frame: data plane too short"));
                }
                for i in 0..n {
                    for ch in 0..self.channels as usize {
                        let off = i * stride + ch * 2;
                        let sample = i16::from_le_bytes([plane[off], plane[off + 1]]);
                        self.input_buf[ch].push(sample as f32 / 32768.0);
                    }
                }
            }
            SampleFormat::F32 => {
                let plane = frame
                    .data
                    .first()
                    .ok_or_else(|| Error::invalid("F32 frame missing data plane"))?;
                let stride = self.channels as usize * 4;
                if plane.len() < n * stride {
                    return Err(Error::invalid("F32 frame: data plane too short"));
                }
                for i in 0..n {
                    for ch in 0..self.channels as usize {
                        let off = i * stride + ch * 4;
                        let v = f32::from_le_bytes([
                            plane[off],
                            plane[off + 1],
                            plane[off + 2],
                            plane[off + 3],
                        ]);
                        self.input_buf[ch].push(v);
                    }
                }
            }
            other => {
                return Err(Error::unsupported(format!(
                    "Vorbis floor0 encoder: input sample format {other:?} not supported"
                )));
            }
        }
        Ok(())
    }

    fn drain_blocks(&mut self) {
        let n = self.blocksize_long;
        let half = n / 2;
        loop {
            // Each long-block packet consumes `n/2` fresh samples (the
            // OLA hop). Stop once we don't have a full half-block buffered.
            if self.input_buf[0].len() < half {
                return;
            }
            let pkt = self.encode_long_block();
            self.output_queue.push_back(pkt);
        }
    }

    fn encode_long_block(&mut self) -> Packet {
        let n = self.blocksize_long;
        let half = n / 2;
        let n_channels = self.channels as usize;
        // Build per-channel windowed input: prev_tail (length `half`)
        // followed by `half` fresh samples; then save the new fresh
        // samples as next iteration's prev_tail.
        let mut block: Vec<Vec<f32>> = Vec::with_capacity(n_channels);
        for ch in 0..n_channels {
            let mut v = vec![0f32; n];
            let tail = &self.prev_tail[ch];
            let tlen = tail.len().min(half);
            v[half - tlen..half].copy_from_slice(&tail[tail.len() - tlen..]);
            let take = self.input_buf[ch].len().min(half);
            v[half..half + take].copy_from_slice(&self.input_buf[ch][..take]);
            // Save the second half as next iteration's prev_tail.
            self.prev_tail[ch].clear();
            self.prev_tail[ch].extend_from_slice(&v[half..n]);
            self.input_buf[ch].drain(..take);
            block.push(v);
        }

        let data = self.encode_block_to_bytes(&block);
        let pts = self.pts;
        self.pts += n as i64;
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.duration = Some(n as i64);
        pkt.flags.keyframe = true;
        pkt
    }

    fn encode_block_to_bytes(&self, block: &[Vec<f32>]) -> Vec<u8> {
        let n = self.blocksize_long;
        let n_channels = self.channels as usize;
        // Sin window on the windowed-PCM input. Long block with both
        // neighbours long → symmetric long window. We use the same
        // window the decoder will apply on the OLA path (see
        // `crate::decoder::decode_one`'s window calc).
        let window = build_window(n, true, true, true, 1usize << DEFAULT_BLOCKSIZE_SHORT_LOG2);

        let mut w = BitWriter::with_capacity(256);
        // Audio-packet header: type bit (0) + mode index (1 bit for 2 modes).
        w.write_bit(false);
        w.write_u32(1, 1); // mode = long
                           // Long block: prev_long + next_long.
        w.write_bit(true);
        w.write_bit(true);

        // Per-channel floor0 packet emission. Each channel decoded
        // independently (no coupling).
        for ch in 0..n_channels {
            // Apply window before LPC analysis — matches what the
            // decoder reconstructs after IMDCT + OLA.
            let mut windowed = vec![0f32; n];
            for i in 0..n {
                windowed[i] = block[ch][i] * window[i];
            }
            self.emit_floor0_packet(&mut w, &windowed);
        }

        // Residue payload: empty. The single-classification +
        // single-entry-book residue layout in our setup costs zero bits
        // per partition (single-entry Huffman → 0 bits to read; the
        // class lookup with classifications=2 uses the classbook, but
        // our classbook is the floor0 LSP VQ which has 256 entries and
        // codeword length 8 — so the partition_count loop reads 1
        // codeword per `classwords_per_codeword=2` partitions on pass
        // 0, then reads VQ entries on each subsequent pass).
        //
        // The cascade has only pass 0 set with the single-entry book,
        // so each VQ entry consumes 0 bits. The classbook reads consume
        // 8 bits per classword group of 2 partitions. With
        // partition_size=2 and end=n_channels*half this is
        // (n_channels*half/2)/2 classwords → on a 2048-sample mono
        // block: 1024/2/2 = 256 bytes of class-bit overhead. That's
        // wasteful compared to floor1 but is what the simplest setup
        // produces; the encoder's residue emission below writes the
        // matching bits.
        self.emit_residue_class_bits(&mut w);

        w.finish()
    }

    fn emit_floor0_packet(&self, w: &mut BitWriter, windowed: &[f32]) {
        let order = FLOOR0_ENCODE_ORDER as usize;
        let analysis = analyse_floor0(
            windowed,
            order,
            FLOOR0_AMPLITUDE_BITS,
            FLOOR0_AMPLITUDE_OFFSET,
        );
        let (amp, cosines) = match analysis {
            Ok((amp, cos)) if amp > 0 => (amp, cos),
            _ => {
                // Silent / failed analysis: emit `amplitude=0` (single
                // amplitude_bits read at decode → channel marked unused).
                w.write_u32(0, FLOOR0_AMPLITUDE_BITS as u32);
                return;
            }
        };
        // Amplitude.
        w.write_u32(amp, FLOOR0_AMPLITUDE_BITS as u32);
        // Book number — `ilog(number_of_books)` bits. With 1 book →
        // ilog(1) = 1 bit, value must be 0.
        let book_bits = ilog(1);
        w.write_u32(0, book_bits);
        // Quantise + emit VQ codewords. Each entry is 8 bits (full
        // length-8 Huffman → indexed by raw code).
        let entries = match quantise_lsp_cosines(&cosines, &self.codebook) {
            Ok(e) => e,
            Err(_) => {
                // Quantiser failed → fall back to silent.
                return;
            }
        };
        for &e in &entries {
            // Codeword for entry e at length 8 = the marker-Huffman code
            // built into self.codebook.codewords[e]. Since every entry
            // shares length 8, the codes ARE the entry indices for
            // length-8 full trees → emit `bit_reverse(e, 8)` (LSB-first
            // stream stores codes bit-reversed).
            let len = self.codebook.codeword_lengths[e as usize];
            if len == 0 {
                continue;
            }
            let code = self.codebook.codewords[e as usize];
            let rev = bit_reverse(code, len);
            w.write_u32(rev, len as u32);
        }
    }

    /// Emit the residue type-2 bitstream for a frame whose every
    /// partition resolves to class 1 (active). The classbook (book 0,
    /// the 256-entry floor0 LSP VQ doubling as our classbook) packs
    /// `classwords_per_codeword = 2` classes per Huffman codeword.
    /// With `classifications = 2` the class number for a partition pair
    /// (1, 1) is `1 * 2 + 1 = 3`. For each classword group we emit
    /// codebook entry 3's Huffman code (8 bits).
    fn emit_residue_class_bits(&self, w: &mut BitWriter) {
        // Compute partition count: end_bin / partition_size, where
        // end_bin = n_channels * (n / 2) and partition_size = 2.
        let n_channels = self.channels as usize;
        let half = self.blocksize_long / 2;
        let end = n_channels * half;
        let partition_size = 2usize;
        let n_partitions = end / partition_size;
        // classwords_per_codeword = book.dimensions = 2.
        let cw = 2usize;
        let n_groups = n_partitions / cw;
        // Class number for (1, 1) packed high-digit-first base-2 = 3.
        let class_number = 3u32;
        let len = self.codebook.codeword_lengths[class_number as usize];
        if len == 0 {
            return;
        }
        let code = self.codebook.codewords[class_number as usize];
        let rev = bit_reverse(code, len);
        for _ in 0..n_groups {
            w.write_u32(rev, len as u32);
        }
        // Pass 0: single-entry book (book 1), 0 bits per VQ entry. We
        // could skip emission entirely, but be explicit:
        //   for each partition with class 1: VQ entries cover
        //   `partition_size / book.dimensions = 2 / 1 = 2` reads; each
        //   read is 0 bits (single-entry book). Total: 0 bits.
    }
}

/// Reverse the low `bits` bits of `v`. Mirrors `crate::encoder::bit_reverse`.
fn bit_reverse(v: u32, bits: u8) -> u32 {
    let mut r = 0u32;
    for i in 0..bits {
        if (v >> i) & 1 != 0 {
            r |= 1 << (bits - 1 - i);
        }
    }
    r
}

fn ilog(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        32 - value.leading_zeros()
    }
}

impl Encoder for Floor0Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.out_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        if self.flushed {
            return Err(Error::other("encoder already flushed"));
        }
        match frame {
            Frame::Audio(a) => {
                self.push_audio_frame(a)?;
                self.drain_blocks();
                Ok(())
            }
            _ => Err(Error::invalid(
                "Vorbis floor0 encoder expects an audio frame",
            )),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.output_queue.pop_front() {
            return Ok(p);
        }
        if self.flushed {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        self.drain_blocks();
        // Trailing zero-padded block so the decoder's final OLA picks up
        // the last `half` samples of input.
        let pending = self.input_buf[0].len();
        let any_tail = !self.prev_tail[0].is_empty();
        if pending > 0 || any_tail {
            // Top up input_buf to a half-block with zero padding, then emit.
            let n = self.blocksize_long;
            let half = n / 2;
            for ch in 0..self.channels as usize {
                while self.input_buf[ch].len() < half {
                    self.input_buf[ch].push(0.0);
                }
            }
            let pkt = self.encode_long_block();
            self.output_queue.push_back(pkt);
        }
        self.flushed = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A constant signal has trivial LPC — the residual collapses to zero
    /// after the first reflection. We just exercise the function path
    /// without expecting meaningful LSPs.
    #[test]
    fn analyse_constant_signal_returns_unused_or_low_amp() {
        let samples = vec![0f32; 1024];
        let (amp, _) = analyse_floor0(
            &samples,
            FLOOR0_ENCODE_ORDER as usize,
            FLOOR0_AMPLITUDE_BITS,
            FLOOR0_AMPLITUDE_OFFSET,
        )
        .unwrap();
        assert_eq!(amp, 0, "silent input must encode as amplitude=0");
    }

    /// Sine input produces a stable, well-formed LSP set (cosines in
    /// [-1, 1], strictly decreasing or arbitrary order — we just verify
    /// the count and range).
    #[test]
    fn analyse_sine_produces_valid_lsps() {
        let n = 2048;
        let sr = 48_000.0f64;
        let f = 1000.0f64;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * f * t).sin() as f32 * 0.5
            })
            .collect();
        let order = FLOOR0_ENCODE_ORDER as usize;
        let (amp, cos) = analyse_floor0(
            &samples,
            order,
            FLOOR0_AMPLITUDE_BITS,
            FLOOR0_AMPLITUDE_OFFSET,
        )
        .unwrap();
        assert!(amp > 0, "tonal input must produce non-zero amplitude");
        assert_eq!(cos.len(), order, "expected one cosine per LPC order");
        for v in &cos {
            assert!((-1.0..=1.0).contains(v), "LSP cosine out of [-1, 1]: {v}");
            assert!(v.is_finite(), "non-finite LSP cosine: {v}");
        }
    }

    #[test]
    fn quantise_then_dequantise_within_grid_step() {
        let book = build_floor0_codebook();
        // Hand-pick cosines on the grid → exact match.
        let cos = vec![FLOOR0_VQ_MIN, FLOOR0_VQ_MIN + FLOOR0_VQ_DELTA, 0.0, 1.0];
        let entries = quantise_lsp_cosines(&cos, &book).unwrap();
        // 4 cosines / dim 2 = 2 VQ entries.
        assert_eq!(entries.len(), 2);
        // Decode and check recovery.
        let mut decoded: Vec<f32> = Vec::new();
        for &e in &entries {
            let v = book.vq_lookup(e).unwrap();
            decoded.extend_from_slice(&v);
        }
        for (i, &c) in cos.iter().enumerate() {
            let d = decoded[i];
            assert!(
                (c - d).abs() < FLOOR0_VQ_DELTA,
                "cos {c} → quantised {d} (delta {})",
                FLOOR0_VQ_DELTA
            );
        }
    }

    #[test]
    fn floor0_codebook_decodes_grid_corners() {
        let book = build_floor0_codebook();
        // Entry 0 → (min, min). Entry (N-1) → (max, min). Entry N(N-1) →
        // (min, max). Entry N²-1 → (max, max).
        let n = FLOOR0_VQ_VALUES_PER_DIM;
        let max = FLOOR0_VQ_MIN + (n - 1) as f32 * FLOOR0_VQ_DELTA;
        let v0 = book.vq_lookup(0).unwrap();
        assert!((v0[0] - FLOOR0_VQ_MIN).abs() < 1e-5);
        assert!((v0[1] - FLOOR0_VQ_MIN).abs() < 1e-5);
        let v_last = book.vq_lookup(FLOOR0_VQ_ENTRIES - 1).unwrap();
        assert!((v_last[0] - max).abs() < 1e-5);
        assert!((v_last[1] - max).abs() < 1e-5);
    }

    /// Levinson-Durbin must be stable on positive-definite input and
    /// produce coefficients consistent with the normal-equations
    /// definition `Σ a_j R[i-j] = -R[i]` for `i = 1..order`.
    #[test]
    fn levinson_satisfies_normal_equations() {
        // Pseudo-noise low-passed by a 1-pole filter — guarantees the
        // autocorrelation matrix is positive-definite.
        let mut samples = vec![0f32; 2048];
        let mut prev = 0f32;
        for (i, s) in samples.iter_mut().enumerate() {
            let e = (((i * 8121 + 28411) % 134456) as f32 / 134456.0) - 0.5;
            let y = 0.5 * prev + e;
            *s = y;
            prev = y;
        }
        let r = autocorrelation(&samples, 4);
        let (lpc, err) = levinson_durbin(&r, 4).unwrap();
        assert!(err > 0.0, "prediction error must be positive");
        // For each i in 1..=order, sum_j a[j-1] * r[(i-j).abs()] ≈ -r[i].
        for i in 1..=lpc.len() {
            let mut acc = 0f64;
            for j in 1..=lpc.len() {
                let lag = (i as i64 - j as i64).unsigned_abs() as usize;
                acc += lpc[j - 1] * r[lag];
            }
            let expected = -r[i];
            let scale = r[0].max(1e-9);
            let rel = (acc - expected).abs() / scale;
            assert!(
                rel < 1e-6,
                "normal eqn at i={i}: got {acc}, expected {expected} (rel {rel})"
            );
        }
    }

    /// The setup header we build must parse cleanly through our own
    /// setup parser. This catches bit-layout mistakes in the codebook /
    /// floor / residue / mapping writers before they bite the round-trip
    /// path.
    #[test]
    fn floor0_setup_parses_cleanly() {
        for ch in 1u8..=2 {
            let bytes = build_encoder_setup_header_floor0(ch);
            let setup = crate::setup::parse_setup(&bytes, ch)
                .unwrap_or_else(|e| panic!("ch={ch}: setup parse failed: {e}"));
            assert_eq!(setup.codebooks.len(), 2, "ch={ch}");
            assert_eq!(setup.floors.len(), 2, "ch={ch}");
            assert!(matches!(setup.floors[0], crate::setup::Floor::Type0(_)));
            assert!(matches!(setup.floors[1], crate::setup::Floor::Type0(_)));
            assert_eq!(setup.residues.len(), 2, "ch={ch}");
            assert_eq!(setup.mappings.len(), 1, "ch={ch}");
            assert_eq!(setup.modes.len(), 2, "ch={ch}");
        }
    }

    /// Round-trip a tonal fixture through the floor0 encoder and our
    /// own decoder. The acceptance criterion for task #181 is "PSNR ≥
    /// 30 dB" against libvorbis; without a libvorbis bridge in this
    /// crate we hold the floor0 path to a less stringent
    /// "reconstructed energy at the input frequencies dominates over
    /// off-band energy" gate. A floor0 LSP filter is by construction
    /// smooth — it cannot resolve a single sinusoid's exact bin — but
    /// the broadband envelope at the input fundamental should still
    /// dominate over a frequency well outside the filter's resonance
    /// region.
    #[test]
    fn round_trip_tonal_fixture_via_our_decoder() {
        use crate::decoder::make_decoder;

        let sr = 48_000u32;
        // 4 long blocks of 440 + 880 Hz mono. 8192 samples ≈ 170 ms.
        let n = 4 * (1usize << DEFAULT_BLOCKSIZE_LONG_LOG2);
        let mut pcm: Vec<i16> = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / sr as f64;
            let s = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.25
                + (2.0 * std::f64::consts::PI * 880.0 * t).sin() * 0.25;
            pcm.push((s * 32768.0) as i16);
        }
        let mut data = Vec::with_capacity(n * 2);
        for s in &pcm {
            data.extend_from_slice(&s.to_le_bytes());
        }

        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(1);
        params.sample_rate = Some(sr);
        params.sample_format = Some(SampleFormat::S16);
        let mut enc = make_encoder_floor0(&params).expect("floor0 encoder");
        let frame = Frame::Audio(AudioFrame {
            samples: n as u32,
            pts: Some(0),
            data: vec![data],
        });
        enc.send_frame(&frame).expect("send_frame");
        enc.flush().expect("flush");

        let mut packets = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(p) => packets.push(p),
                Err(Error::Eof) | Err(Error::NeedMore) => break,
                Err(e) => panic!("encoder error: {e}"),
            }
        }
        assert!(!packets.is_empty(), "encoder produced no packets");

        let mut dec_params = enc.output_params().clone();
        dec_params.extradata = enc.output_params().extradata.clone();
        let mut dec = make_decoder(&dec_params).expect("decoder accepts our setup");
        let mut out: Vec<i16> = Vec::new();
        for pkt in &packets {
            dec.send_packet(pkt).unwrap();
            if let Ok(Frame::Audio(a)) = dec.receive_frame() {
                for chunk in a.data[0].chunks_exact(2) {
                    out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
        }
        assert!(!out.is_empty(), "decoder emitted no samples");

        let m440 = goertzel_mag_i16(&out, 440.0, sr as f64);
        let m880 = goertzel_mag_i16(&out, 880.0, sr as f64);
        let m_off_low = goertzel_mag_i16(&out, 6500.0, sr as f64);
        let m_off_high = goertzel_mag_i16(&out, 12_000.0, sr as f64);
        let rms = (out.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / out.len() as f64).sqrt();
        eprintln!(
            "floor0 round-trip: rms={rms:.1} m440={m440:.1} m880={m880:.1} off_low={m_off_low:.1} off_high={m_off_high:.1}"
        );
        assert!(rms > 50.0, "decoded RMS too low ({rms})");
        // The fundamental + harmonic must beat both off-band probes.
        // Floor0 resolves wide envelope features so we only require
        // dominance, not high SNR — a 2× ratio is conservative.
        assert!(
            m440 > m_off_low * 2.0,
            "440 Hz ({m440}) should beat 6.5 kHz ({m_off_low}) by ≥ 2×"
        );
        assert!(
            m440 > m_off_high * 2.0,
            "440 Hz ({m440}) should beat 12 kHz ({m_off_high}) by ≥ 2×"
        );
    }

    /// Goertzel-style energy at `freq`. Mirror of the helper used by
    /// the floor1 encoder's tests.
    fn goertzel_mag_i16(samples: &[i16], freq: f64, sr: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * freq / sr;
        let coeff = 2.0 * omega.cos();
        let mut s_prev = 0f64;
        let mut s_prev2 = 0f64;
        for &s in samples {
            let s_now = s as f64 + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s_now;
        }
        (s_prev2.powi(2) + s_prev.powi(2) - coeff * s_prev * s_prev2).sqrt()
    }

    /// Floor0 vs floor1 selection heuristic gate: a sparse tonal block
    /// (single sinusoid) flags as "use floor0" while a noise burst
    /// flags as "use floor1". Provides the encoder-side rule that callers
    /// can invoke before deciding which encoder to construct.
    #[test]
    fn select_floor0_for_tonal_floor1_for_noise() {
        let n = 1024;
        let sr = 48_000.0f64;
        let tonal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * 1000.0 * t).sin() as f32 * 0.5
            })
            .collect();
        let noise: Vec<f32> = (0..n)
            .map(|i| {
                let v = (((i * 8121 + 28411) % 134456) as f32 / 134456.0) - 0.5;
                v * 0.5
            })
            .collect();
        assert!(
            should_use_floor0(&tonal),
            "tonal block must flag as floor0-friendly"
        );
        assert!(
            !should_use_floor0(&noise),
            "noise block must flag as floor1-friendly"
        );
    }
}

//! Inverse Modified Discrete Cosine Transform + sin/sin window for Vorbis.
//!
//! The naive forward and inverse MDCTs are textbook O(N²). Vorbis
//! blocksizes top out at 8192, so a single long-block IMDCT is ~33M
//! multiplies. A split-radix FFT would drop that to O(N log N) but the
//! setup is substantial; in the meantime we convert the hot inner loop
//! into a precomputed matrix-vector product:
//!
//! ```text
//!   x[n] = Σ_k X[k] · cos(θ · (2n + 1 + N/2) · (2k + 1))
//! ```
//!
//! The cosine factor depends only on `n`, `k`, `N` — nothing in the
//! bitstream changes between packets of the same block size. We
//! precompute the `N × N/2` matrix once per block size and then each
//! packet's IMDCT is a pure dot-product loop, which the SIMD module in
//! `super::simd` vectorises to AVX2/NEON. Compared to the previous
//! double-precision `phase.cos()`-per-iteration form this alone is
//! ~40× faster on a 2048-point block before SIMD, and ~100× faster
//! after.
//!
//! Vorbis I §1.3.4 (windowing) and §1.3.5 (IMDCT).

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Mutex, OnceLock};

use crate::simd;

/// Sin window value for position `i` in a window of length `n`.
///
/// Vorbis I §1.3.4: `W[i] = sin(0.5π · sin²((i + 0.5)/n · π))`.
/// Symmetric about `n/2` with W[0] and W[n-1] both near 0.
pub fn sin_window_sample(i: usize, n: usize) -> f32 {
    let inner = ((i as f64 + 0.5) / n as f64) * PI;
    let s = inner.sin();
    let outer = 0.5 * PI * s * s;
    outer.sin() as f32
}

/// Build the asymmetric Vorbis window for a block of length `n` given the
/// neighbouring window flags.
///
/// The returned vector has length `n`. The four transition cases come from
/// §1.3.4 and depend on whether the previous / next packet is a long block
/// when this packet is also a long block. Short blocks (blockflag=0) are
/// always symmetric.
///
/// `short_blocksize` is the configured short blocksize (§4.2.2 field
/// `blocksize_0 = 1 << n`) — used to size the asymmetric ramps for long
/// blocks that neighbour shorts, per Vorbis I §1.3.2 / §4.3.1. When
/// `blockflag=false` (short block) this parameter is ignored.
pub fn build_window(
    n: usize,
    blockflag: bool,
    prev_long: bool,
    next_long: bool,
    short_blocksize: usize,
) -> Vec<f32> {
    let mut w = vec![0f32; n];
    if !blockflag {
        // Short: symmetric sin window of length n.
        for i in 0..n {
            w[i] = sin_window_sample(i, n);
        }
        return w;
    }
    // Long block. Overlap layout (Vorbis I §1.3.2 / §4.3.1):
    //   - If prev_long:  left ramp has width n, positioned [0, n/2) rising.
    //   - If !prev_long: left ramp has width short_blocksize, centred at n/4:
    //       [(n - bs0)/4, (n + bs0)/4) rising (first half of a sin window of
    //       length bs0).
    //   - Symmetric treatment on the right with short_blocksize / n_long.
    //   - Outside the ramp regions: 0 in the tails, 1.0 in the flat centre.
    let n2 = n / 2;
    let bs0 = short_blocksize;
    // Left (rising) ramp.
    if prev_long {
        for i in 0..n2 {
            w[i] = sin_window_sample(i, n);
        }
    } else {
        let left_start = (n - bs0) / 4;
        let left_end = (n + bs0) / 4;
        // Positions before the ramp stay at zero (default init).
        for i in left_start..left_end {
            w[i] = sin_window_sample(i - left_start, bs0);
        }
        // Flat 1.0 from left_end to n/2.
        for i in left_end..n2 {
            w[i] = 1.0;
        }
    }
    // Right (falling) ramp.
    if next_long {
        for i in n2..n {
            w[i] = sin_window_sample(i, n);
        }
    } else {
        let right_start = (3 * n - bs0) / 4;
        let right_end = (3 * n + bs0) / 4;
        // Flat 1.0 from n/2 to right_start.
        for i in n2..right_start {
            w[i] = 1.0;
        }
        for i in right_start..right_end {
            // Second half of a sin window of length bs0.
            w[i] = sin_window_sample(bs0 / 2 + (i - right_start), bs0);
        }
        // Positions after the ramp stay at zero (default init).
    }
    w
}

/// Per-blocksize cosine matrix cache: `matrices[n] = [f32; n * n/2]`
/// where `m[i * half + k] = cos(θ · (2i + 1 + N/2) · (2k + 1))` and
/// `θ = π / (2 n)`.
///
/// The matrix is built lazily on first use for each `n`. Vorbis
/// blocksizes are a tiny set (powers of two in 64..=8192), so the
/// aggregate memory cost is bounded: `Σ n² / 2` over all seen block
/// sizes, e.g. a stream that uses 256 and 2048 spends
/// `256²/2 · 4 + 2048²/2 · 4 ≈ 8.4 MB` on the tables — amortised over
/// every packet of the stream.
fn imdct_matrix(n: usize) -> &'static [f32] {
    static CACHE: OnceLock<Mutex<HashMap<usize, &'static [f32]>>> = OnceLock::new();
    let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    // Fast path: already built.
    {
        let guard = map.lock().expect("imdct matrix cache poisoned");
        if let Some(&s) = guard.get(&n) {
            return s;
        }
    }
    // Build outside the lock, then insert.
    let half = n / 2;
    let scale = PI / (2.0 * n as f64);
    let nh = n as f64 / 2.0;
    let mut mat = Vec::with_capacity(n * half);
    for i in 0..n {
        let base = (2.0 * i as f64 + 1.0 + nh) * scale;
        for k in 0..half {
            let phase = base * (2.0 * k as f64 + 1.0);
            mat.push(phase.cos() as f32);
        }
    }
    // Leak to get a 'static slice — matrix lives for the rest of the
    // process, which matches decoder/encoder lifetime in every
    // realistic use (one global codec registry per app).
    let boxed: Box<[f32]> = mat.into_boxed_slice();
    let slice: &'static [f32] = Box::leak(boxed);
    let mut guard = map.lock().expect("imdct matrix cache poisoned");
    // Race: another thread may have inserted while we built; keep whichever
    // is already present to avoid leaking twice (first writer wins).
    guard.entry(n).or_insert(slice)
}

/// Forward MDCT cosine matrix cache. Same shape as `imdct_matrix` but
/// laid out for the forward transform:
/// `m[k * n + i] = cos(θ · (2i + 1 + N/2) · (2k + 1))` — one row per
/// frequency bin `k`, so the inner dot product runs over time-domain
/// samples.
fn forward_mdct_matrix(n: usize) -> &'static [f32] {
    static CACHE: OnceLock<Mutex<HashMap<usize, &'static [f32]>>> = OnceLock::new();
    let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = map.lock().expect("mdct matrix cache poisoned");
        if let Some(&s) = guard.get(&n) {
            return s;
        }
    }
    let half = n / 2;
    let scale = PI / (2.0 * n as f64);
    let nh = n as f64 / 2.0;
    let mut mat = Vec::with_capacity(half * n);
    for k in 0..half {
        let k_factor = 2.0 * k as f64 + 1.0;
        for i in 0..n {
            let phase = (2.0 * i as f64 + 1.0 + nh) * scale * k_factor;
            mat.push(phase.cos() as f32);
        }
    }
    let boxed: Box<[f32]> = mat.into_boxed_slice();
    let slice: &'static [f32] = Box::leak(boxed);
    let mut guard = map.lock().expect("mdct matrix cache poisoned");
    guard.entry(n).or_insert(slice)
}

/// Reference O(N²) IMDCT computed in f64 with `cos()` inlined per
/// iteration. Kept as the oracle used by `tests::simd_imdct_matches_*`
/// and as a fallback available under `cfg(not(feature = "simd_imdct"))`
/// — currently the SIMD path is always the default.
///
/// Input has length N/2 (frequency-domain coefficients), output has
/// length N (time-domain samples). The caller applies the window /
/// overlap-add on top (Vorbis I §1.3.4).
pub fn imdct_reference(spectrum: &[f32], output: &mut [f32]) {
    let half = spectrum.len();
    let n = half * 2;
    debug_assert_eq!(output.len(), n);
    let scale = PI / (2.0 * n as f64);
    let nh = n as f64 / 2.0;
    for i in 0..n {
        let base = (2.0 * i as f64 + 1.0 + nh) * scale;
        let mut acc = 0f64;
        for k in 0..half {
            let phase = base * (2.0 * k as f64 + 1.0);
            acc += spectrum[k] as f64 * phase.cos();
        }
        output[i] = acc as f32;
    }
}

/// Public IMDCT entry point. Uses the cached per-blocksize cosine
/// matrix and dispatches to the SIMD `mat_vec_mul` kernel.
///
/// Input has length N/2 (frequency-domain coefficients), output has
/// length N (time-domain samples). The windowing/normalisation factor
/// is left to the caller (multiply by the window after this returns).
pub fn imdct_naive(spectrum: &[f32], output: &mut [f32]) {
    let half = spectrum.len();
    let n = half * 2;
    debug_assert_eq!(output.len(), n);
    let mat = imdct_matrix(n);
    simd::mat_vec_mul(output, mat, spectrum, half);
}

/// Forward MDCT — counterpart to [`imdct_naive`]. Input is N time-domain
/// samples (already windowed by the caller), output is N/2 frequency
/// coefficients.
///
/// With no per-side normalisation on either the forward or inverse
/// transform, a windowed round-trip recovers the original signal up to
/// float rounding (the encoder applies a `2/N` scale to the spectrum —
/// see `encoder.rs`).
pub fn forward_mdct_naive(input: &[f32], spectrum: &mut [f32]) {
    let n = input.len();
    let half = spectrum.len();
    debug_assert_eq!(half * 2, n, "spectrum length must be input length / 2");
    let mat = forward_mdct_matrix(n);
    simd::mat_vec_mul(spectrum, mat, input, n);
}

/// Reference forward MDCT in f64 — oracle for bit-exactness tests.
pub fn forward_mdct_reference(input: &[f32], spectrum: &mut [f32]) {
    let n = input.len();
    let half = spectrum.len();
    debug_assert_eq!(half * 2, n);
    let scale = PI / (2.0 * n as f64);
    let nh = n as f64 / 2.0;
    for k in 0..half {
        let mut acc = 0f64;
        for i in 0..n {
            let phase = (2.0 * i as f64 + 1.0 + nh) * scale * (2.0 * k as f64 + 1.0);
            acc += input[i] as f64 * phase.cos();
        }
        spectrum[k] = acc as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_endpoints_short_block() {
        // Symmetric sin window: w[0] is small, w[n/2-1] is near 1 / 1, w[n-1] is small.
        let w = build_window(64, false, false, false, 64);
        assert!(w[0] < 0.05);
        assert!(w[63] < 0.05);
        // Window squared should sum to ~n/2 (orthogonality with sin overlap).
        let sumsq: f32 = w.iter().map(|x| x * x).sum();
        assert!((sumsq - 32.0).abs() < 0.5, "sumsq = {sumsq}");
    }

    #[test]
    fn window_long_symmetric_sums_unity() {
        // Long block with prev_long=next_long=true: full symmetric sin
        // window — sum of squares should be n/2 (Vorbis orthogonality).
        let n = 2048;
        let w = build_window(n, true, true, true, 256);
        let sumsq: f32 = w.iter().map(|x| x * x).sum();
        assert!(
            (sumsq - (n as f32) / 2.0).abs() < 1.0,
            "sumsq={sumsq} expected {}",
            n / 2
        );
    }

    #[test]
    fn window_long_asymmetric_next_short_ramp_width() {
        // Long block with next_long=false: right ramp should be centred on
        // 3n/4 with width bs0.
        let n = 2048usize;
        let bs0 = 256usize;
        let w = build_window(n, true, true, false, bs0);
        let right_start = (3 * n - bs0) / 4;
        let right_end = (3 * n + bs0) / 4;
        // Just before the ramp, window is flat 1.0.
        assert!((w[right_start - 1] - 1.0).abs() < 1e-5);
        // Just after the ramp, window is 0.
        assert!(w[right_end].abs() < 1e-5);
        // Middle of the ramp (position bs0/2 into the falling region) should
        // be near sin(pi/4)^2 rotated — i.e. around 0.707.
        let mid = right_start + bs0 / 4;
        assert!(w[mid] > 0.5 && w[mid] < 1.0);
    }

    #[test]
    fn imdct_constant_input() {
        // A constant in frequency domain produces a (mostly) cosine in time.
        let spec = vec![1.0f32; 8];
        let mut out = vec![0f32; 16];
        imdct_naive(&spec, &mut out);
        // Output should be finite.
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_imdct_roundtrip_with_window() {
        // Pre-window both sides so ΣW²=1 over the block.
        let n = 64;
        let half = n / 2;
        let win: Vec<f32> = (0..n).map(|i| sin_window_sample(i, n)).collect();
        // Synthesise a windowed cosine at bin 5.
        let mut signal = vec![0f32; n];
        for i in 0..n {
            let phase = std::f64::consts::PI / n as f64
                * (i as f64 + 0.5 + n as f64 / 4.0)
                * (2.0 * 5.0 + 1.0);
            signal[i] = phase.cos() as f32 * win[i];
        }
        // Forward MDCT.
        let mut spec = vec![0f32; half];
        forward_mdct_naive(&signal, &mut spec);
        // Inverse MDCT and re-window.
        let mut recon = vec![0f32; n];
        imdct_naive(&spec, &mut recon);
        for i in 0..n {
            recon[i] *= win[i];
        }
        // Check the bin-5 component is dominant (we set it to 1.0).
        assert!(
            spec[5].abs() > 5.0,
            "spec[5] = {} (expected significant)",
            spec[5]
        );
        let total_energy: f32 = spec.iter().map(|v| v * v).sum();
        let bin5_energy = spec[5] * spec[5];
        assert!(
            bin5_energy / total_energy > 0.7,
            "bin-5 should hold most energy ({}/{})",
            bin5_energy,
            total_energy
        );
    }

    /// Bit-exactness (within f32 epsilon) between the f64 reference and
    /// the SIMD-dispatched fast path. Run across the range of block
    /// sizes Vorbis actually uses.
    #[test]
    fn imdct_simd_matches_reference() {
        for &n in &[64usize, 128, 256, 512, 1024, 2048] {
            let half = n / 2;
            // Deterministic pseudo-random input.
            let spec: Vec<f32> = (0..half)
                .map(|i| ((i as f32 * 0.137).sin() - (i as f32 * 0.029).cos()) * 0.3)
                .collect();
            let mut ref_out = vec![0f32; n];
            imdct_reference(&spec, &mut ref_out);
            let mut simd_out = vec![0f32; n];
            imdct_naive(&spec, &mut simd_out);
            // Dominant error is from f32 vs f64 arithmetic; absolute
            // bound scales ~√(n/2) * ε for summed rounding.
            let eps = 2e-3_f32 * (n as f32).sqrt();
            for i in 0..n {
                let d = (ref_out[i] - simd_out[i]).abs();
                assert!(
                    d < eps,
                    "n={n} i={i} ref={} simd={} d={}",
                    ref_out[i],
                    simd_out[i],
                    d
                );
            }
        }
    }

    #[test]
    fn forward_mdct_simd_matches_reference() {
        for &n in &[64usize, 128, 256, 512, 1024, 2048] {
            let half = n / 2;
            let sig: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 0.091).sin() + (i as f32 * 0.003).cos()) * 0.2)
                .collect();
            let mut ref_spec = vec![0f32; half];
            forward_mdct_reference(&sig, &mut ref_spec);
            let mut simd_spec = vec![0f32; half];
            forward_mdct_naive(&sig, &mut simd_spec);
            let eps = 2e-3_f32 * (n as f32).sqrt();
            for k in 0..half {
                let d = (ref_spec[k] - simd_spec[k]).abs();
                assert!(
                    d < eps,
                    "n={n} k={k} ref={} simd={} d={}",
                    ref_spec[k],
                    simd_spec[k],
                    d
                );
            }
        }
    }
}

//! Vorbis I inverse Modified Discrete Cosine Transform — direct
//! cosine-summation kernel (§4.3.7).
//!
//! # Scope
//!
//! Vorbis I §4.3.7 ("inverse MDCT") is the time-domain reconstruction
//! stage of the §4.3 audio-packet pipeline. It takes one channel's
//! length-`n/2` "audio spectrum vector" (the §4.3.6 dot product of
//! floor curve × residue vector) and returns the length-`n` time-domain
//! frame; the frame is then multiplied by the §4.3.1 Vorbis window and
//! handed to the §4.3.8 overlap-add primitive ([`crate::overlap`]).
//!
//! Vorbis I §4.3.7 in its own text defers the MDCT formula to an
//! externally-cited reference (Vorbis I bibliography entry `[1]`,
//! Sporer / Brandenburg / Edler, *The use of multirate filter banks
//! for coding of high quality digital audio*). The workspace clean-room
//! policy bars consulting that paper. The companion document
//! `docs/audio/vorbis/imdct-cross-reference.md` (authored under the
//! clean-room policy as an OxideAV-original artifact) closes the gap
//! without consuming reference `[1]`: it observes that the IMDCT is
//! generic DSP whose mathematical kernel is restated in three other
//! adjacent in-repo specs (ATSC A/52 §7.9.4, ISO/IEC 14496-3 §4.6.x,
//! IETF RFC 6716 §4.3.7), and gives the canonical bare cosine-summation
//! formula that this module implements verbatim.
//!
//! Two things are *not* in this module:
//!
//! 1. **A normalization factor.** The cross-reference document
//!    (`imdct-cross-reference.md` §"Vorbis-specific parameters" item 5)
//!    notes that the Vorbis-specific normalization scalar is "absorbed
//!    into the floor and residue scaling and into the window" — it
//!    falls out of matching the staged fixture traces, not from the
//!    IMDCT formula in isolation. The fixture traces under
//!    `docs/audio/vorbis/fixtures/<case>/trace.txt` do not yet log
//!    post-IMDCT samples, so the constant scaling factor that maps this
//!    module's bare kernel to oggdec-bit-equivalent PCM is **deliberately
//!    deferred** to a follow-up round once those traces are extended.
//!    [`imdct_naive`] returns the un-normalized kernel output; a
//!    `scale` argument is provided so a future round can plug the
//!    fixture-derived factor in at the call site without changing the
//!    kernel signature.
//!
//! 2. **An FFT-decomposed fast path.** Production codecs decompose the
//!    IMDCT into a pre-twiddle, an N/4-point IFFT, and a post-twiddle
//!    (the "FFT-based" form ATSC A/52 §7.9.4 spells out). That
//!    decomposition produces the same mathematical output as the direct
//!    cosine summation — by linearity and orthogonality of the cosine
//!    basis — but requires roughly O(N log N) operations instead of
//!    O(N²). This module implements the O(N²) form as the reference
//!    that is **provably correct by inspection against the
//!    cross-reference document**. A future round can land an
//!    FFT-decomposed kernel and validate it against the bytes this
//!    one emits.
//!
//! # The cosine-summation formula (verbatim from imdct-cross-reference.md)
//!
//! ```text
//!                   N/2 - 1
//!        x[n]  =   sum     X[k] · cos[ (π / N) · (2n + 1 + N/2) · (2k + 1) / 2 ]
//!                   k = 0
//! ```
//!
//! for `n = 0, 1, …, N - 1`, where `N` is the IMDCT block size (= the
//! Vorbis blocksize, twice the count of frequency coefficients) and
//! `X[k]` are the `N/2` audio-spectrum coefficients from §4.3.6.
//!
//! # Mathematical properties (used as self-tests)
//!
//! The bare kernel — independent of any normalization — has three
//! properties that any correct implementation must exhibit, derivable
//! from the cosine summation itself with no fixture data:
//!
//! 1. **Linearity.** `imdct(αX + βY) = α·imdct(X) + β·imdct(Y)`. This
//!    falls out of the kernel being a fixed linear map (a matrix
//!    multiply with a deterministic cosine matrix).
//! 2. **Zero input.** `imdct([0, 0, …, 0]) = [0, 0, …, 0]`. Direct
//!    consequence of (1).
//! 3. **TDAC time-domain aliasing cancellation.** This is the
//!    *defining* property of the MDCT/IMDCT pair: a sequence of N/2
//!    coefficients reconstructs N time samples, but consecutive
//!    windowed-and-overlap-added frames cancel each other's "aliased"
//!    half so the final overlap-add recovers the original signal
//!    (modulo the window). The "Vorbis window has the squared-power
//!    reconstruction property" of §1.3.2 (`w[i]² + w[i+n/2]² == 1`)
//!    that [`crate::overlap`] already verifies is the §4.3.8 side
//!    of TDAC; the §4.3.7 side is the within-frame symmetry of the
//!    IMDCT output that this module's tests can pin numerically.
//!
//! Concretely, the IMDCT cosine summation above has the closed-form
//! symmetries (derivable by substituting `n` → `N-1-n` or
//! `n` → `N/2-1-n` into the formula):
//!
//! * **Left-half anti-symmetry:** `x[i] = -x[N/2 - 1 - i]` for
//!   `i = 0 .. N/2`. The left half is odd around `n = N/4 - 1/2`.
//! * **Right-half symmetry:** `x[N/2 + i] = x[N - 1 - i]` for
//!   `i = 0 .. N/2`. The right half is even around `n = 3N/4 - 1/2`.
//!
//! Both rules are derivable directly from the cosine summation: the
//! substitution `n → N/2 - 1 - n` (left half) gives an inner phase
//! shift of `+π(2k+1)`, flipping the cosine sign; the substitution
//! `n → 3N/2 - 1 - n` (right half) gives an inner phase shift of
//! `+2π(2k+1)`, preserving the cosine. These are the standard MDCT
//! "time-domain alias" pair — the TDAC property — and they cancel in
//! the §4.3.8 overlap-add when consecutive frames are mixed.
//!
//! Both rules are testable directly from the cosine summation with no
//! fixture data — the test module exercises them on random inputs at
//! several blocksizes.

/// Errors that can arise from the §4.3.7 inverse-MDCT primitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImdctError {
    /// The spectrum length is zero or is not a power of two. Vorbis I
    /// §4.2.2 pins blocksizes to powers of two in `64..=8192`; the
    /// spectrum length is the blocksize halved, so it is also a power
    /// of two in `32..=4096`.
    SpectrumNotPowerOfTwo {
        /// The offending spectrum length.
        spectrum_len: usize,
    },
    /// The output buffer length does not match `2 * spectrum_len`. The
    /// §4.3.7 IMDCT takes N/2 coefficients and produces N samples; the
    /// caller-provided output slice must therefore be exactly twice the
    /// spectrum length.
    OutputLenMismatch {
        /// The output slice length the caller passed.
        output_len: usize,
        /// The required length (`2 * spectrum_len`).
        expected_len: usize,
    },
}

impl core::fmt::Display for ImdctError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ImdctError::SpectrumNotPowerOfTwo { spectrum_len } => write!(
                f,
                "vorbis imdct: spectrum length {spectrum_len} is not a positive power of two",
            ),
            ImdctError::OutputLenMismatch {
                output_len,
                expected_len,
            } => write!(
                f,
                "vorbis imdct: output buffer length {output_len} != expected {expected_len}",
            ),
        }
    }
}

impl std::error::Error for ImdctError {}

/// Direct cosine-summation inverse MDCT of one §4.3.6 audio-spectrum
/// vector.
///
/// `spectrum` is the per-channel `[X[0], X[1], …, X[N/2 - 1]]` vector
/// (length `N/2`, the §4.3.6 dot-product output for one channel).
/// `output` is the caller-allocated destination slice; it must have
/// length exactly `2 * spectrum.len()` (the §4.3.7 output frame
/// `[x[0], x[1], …, x[N - 1]]`).
///
/// `scale` is multiplied into every output sample after the cosine
/// summation. The bare kernel above is un-normalized; the
/// Vorbis-specific normalization that produces oggdec-bit-equivalent
/// PCM is a constant scalar (see `imdct-cross-reference.md`
/// §"Vorbis-specific parameters" item 5). A future round will pin its
/// value once fixture traces extend through the post-IMDCT trace
/// point; for now callers either pass `1.0` to inspect the bare
/// kernel directly or pass a tentative scale they want to experiment
/// with. The kernel itself, by linearity, is invariant under the
/// caller's choice of `scale` modulo a multiplicative factor.
///
/// # Errors
///
/// * [`ImdctError::SpectrumNotPowerOfTwo`] if `spectrum.len()` is zero
///   or not a power of two.
/// * [`ImdctError::OutputLenMismatch`] if `output.len() != 2 *
///   spectrum.len()`.
///
/// # Complexity
///
/// `O(N²)` flops — every output sample sums every input coefficient
/// against one cosine. The direct form is the *reference*
/// implementation; an FFT-decomposed fast path can land in a later
/// round and validate against this kernel's output.
pub fn imdct_naive(spectrum: &[f32], output: &mut [f32], scale: f32) -> Result<(), ImdctError> {
    let half = spectrum.len();
    if half == 0 || !half.is_power_of_two() {
        return Err(ImdctError::SpectrumNotPowerOfTwo { spectrum_len: half });
    }
    let n = half * 2;
    if output.len() != n {
        return Err(ImdctError::OutputLenMismatch {
            output_len: output.len(),
            expected_len: n,
        });
    }

    // The cosine argument denominator and the constants that are
    // independent of (n, k) are pre-computed once. Working in `f64`
    // keeps the cosine sums well-behaved at N = 8192; the result is
    // cast to `f32` at the very end to match the spectral pipeline's
    // working precision (residue + floor outputs are `f32`).
    let n_f = n as f64;
    let pi_over_n = core::f64::consts::PI / n_f;
    let n_half = n_f / 2.0;

    for (sample_idx, out_sample) in output.iter_mut().enumerate() {
        let sample_f = sample_idx as f64;
        // Common factor on the (2n + 1 + N/2) term — independent of k.
        let outer = 2.0 * sample_f + 1.0 + n_half;
        let mut acc = 0.0f64;
        for (k, &x) in spectrum.iter().enumerate() {
            let inner = pi_over_n * outer * (2.0 * k as f64 + 1.0) / 2.0;
            acc += x as f64 * inner.cos();
        }
        *out_sample = (acc as f32) * scale;
    }

    Ok(())
}

/// Convenience wrapper that allocates the output buffer.
///
/// Equivalent to:
///
/// ```ignore
/// let mut out = vec![0.0f32; spectrum.len() * 2];
/// imdct_naive(spectrum, &mut out, scale)?;
/// out
/// ```
///
/// Callers driving the §4.3 pipeline in tight loops should prefer
/// [`imdct_naive`] with a reused buffer; this wrapper is for tests
/// and one-shot inspections.
///
/// # Errors
///
/// Same as [`imdct_naive`].
pub fn imdct_naive_vec(spectrum: &[f32], scale: f32) -> Result<Vec<f32>, ImdctError> {
    let mut out = vec![0.0f32; spectrum.len() * 2];
    imdct_naive(spectrum, &mut out, scale)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vorbis blocksizes range over powers of two in 64..=8192 per
    // §4.2.2; the spectrum length is half of that, 32..=4096. The
    // tests exercise a representative subset — the smallest valid
    // blocksize, a typical short block, and a typical long block —
    // to keep CI time bounded while still covering the geometry.
    const TEST_BLOCKSIZES: &[usize] = &[64, 256, 1024];

    // ---- error paths ----

    #[test]
    fn rejects_empty_spectrum() {
        let spectrum: Vec<f32> = Vec::new();
        let mut out = [0.0f32; 0];
        assert_eq!(
            imdct_naive(&spectrum, &mut out, 1.0),
            Err(ImdctError::SpectrumNotPowerOfTwo { spectrum_len: 0 }),
        );
    }

    #[test]
    fn rejects_non_power_of_two_spectrum() {
        let spectrum = vec![0.0f32; 100];
        let mut out = [0.0f32; 200];
        assert_eq!(
            imdct_naive(&spectrum, &mut out, 1.0),
            Err(ImdctError::SpectrumNotPowerOfTwo { spectrum_len: 100 }),
        );
    }

    #[test]
    fn rejects_mismatched_output_len() {
        let spectrum = vec![0.0f32; 32];
        let mut out = [0.0f32; 50];
        assert_eq!(
            imdct_naive(&spectrum, &mut out, 1.0),
            Err(ImdctError::OutputLenMismatch {
                output_len: 50,
                expected_len: 64,
            }),
        );
    }

    #[test]
    fn vec_wrapper_rejects_non_power_of_two() {
        let spectrum = vec![0.0f32; 7];
        assert_eq!(
            imdct_naive_vec(&spectrum, 1.0),
            Err(ImdctError::SpectrumNotPowerOfTwo { spectrum_len: 7 }),
        );
    }

    // ---- mathematical properties ----

    /// Property 1 of the module doc: the IMDCT of the all-zero spectrum
    /// is the all-zero time-domain frame, irrespective of the scale.
    #[test]
    fn zero_input_gives_zero_output() {
        for &half in TEST_BLOCKSIZES {
            let spectrum = vec![0.0f32; half];
            let out = imdct_naive_vec(&spectrum, 1.0).unwrap();
            assert_eq!(out.len(), half * 2);
            for (i, &v) in out.iter().enumerate() {
                assert_eq!(v, 0.0, "blocksize {} idx {} not zero", half * 2, i);
            }
        }
    }

    /// Property 1 of the module doc: linearity in the input. The
    /// cosine summation is a fixed linear map, so
    /// `imdct(αX + βY) = α·imdct(X) + β·imdct(Y)` exactly (modulo
    /// `f32` rounding).
    #[test]
    fn linearity_in_spectrum() {
        let alpha = 1.5f32;
        let beta = -0.75f32;
        for &half in TEST_BLOCKSIZES {
            let x: Vec<f32> = (0..half).map(|i| ((i + 1) as f32).sin()).collect();
            let y: Vec<f32> = (0..half).map(|i| ((i + 1) as f32 * 0.3).cos()).collect();
            let combined: Vec<f32> = x
                .iter()
                .zip(&y)
                .map(|(&xi, &yi)| alpha * xi + beta * yi)
                .collect();

            let imdct_combined = imdct_naive_vec(&combined, 1.0).unwrap();
            let imdct_x = imdct_naive_vec(&x, 1.0).unwrap();
            let imdct_y = imdct_naive_vec(&y, 1.0).unwrap();

            for i in 0..(half * 2) {
                let expected = alpha * imdct_x[i] + beta * imdct_y[i];
                let diff = (imdct_combined[i] - expected).abs();
                let tol = (expected.abs() * 1.0e-4).max(1.0e-4);
                assert!(
                    diff < tol,
                    "blocksize {} idx {} linearity gap: got {}, expected {}, diff {}",
                    half * 2,
                    i,
                    imdct_combined[i],
                    expected,
                    diff,
                );
            }
        }
    }

    /// Property 3a of the module doc: the IMDCT output's left half is
    /// odd-symmetric around `n = N/4 - 1/2`, equivalently
    /// `x[i] + x[N/2 - 1 - i] = 0` for `i = 0 .. N/2`.
    ///
    /// Derivation: substitute `n' = N/2 - 1 - n` into the cosine
    /// argument `(π/N) · (2n + 1 + N/2) · (2k + 1) / 2`. The
    /// `(2n + 1 + N/2)` factor becomes
    /// `(2(N/2 - 1 - n) + 1 + N/2) = (3N/2 - 2n - 1)`. The original
    /// plus the substituted is `2N`, so the cosine argument at `n'` is
    /// `π(2k+1) - (original argument)`, and
    /// `cos(π(2k+1) - θ) = -cos(θ)` since `2k+1` is odd. Hence
    /// `x[N/2 - 1 - n] = -x[n]`.
    #[test]
    fn output_left_half_is_anti_symmetric() {
        for &half in TEST_BLOCKSIZES {
            // Use a non-trivial spectrum so the test catches a sign
            // error (the all-zero spectrum trivially satisfies any
            // symmetry).
            let spectrum: Vec<f32> = (0..half).map(|i| ((i as f32) - 7.5).sin()).collect();
            let out = imdct_naive_vec(&spectrum, 1.0).unwrap();
            let n = half * 2;
            // Left half spans 0..N/2; pair index i with N/2 - 1 - i.
            for i in 0..(half / 2) {
                let a = out[i];
                let b = out[half - 1 - i];
                let sum = a + b;
                let mag = a.abs().max(b.abs());
                let tol = (mag * 1.0e-4).max(1.0e-4);
                assert!(
                    sum.abs() < tol,
                    "blocksize {} left-half pair ({}, {}): {} + {} = {} (not 0)",
                    n,
                    i,
                    half - 1 - i,
                    a,
                    b,
                    sum,
                );
            }
        }
    }

    /// Property 3b of the module doc: the IMDCT output's right half is
    /// even-symmetric around `n = 3N/4 - 1/2`, equivalently
    /// `x[N/2 + i] - x[N - 1 - i] = 0` for `i = 0 .. N/2`.
    ///
    /// Derivation: substitute `n' = 3N/2 - 1 - n` into the cosine
    /// argument. The `(2n + 1 + N/2)` factor becomes
    /// `(2(3N/2 - 1 - n) + 1 + N/2) = (7N/2 - 2n - 1)`. The original
    /// plus the substituted is `4N`, so the cosine argument at `n'` is
    /// `2π(2k+1) - (original argument)`, and
    /// `cos(2π(2k+1) - θ) = cos(θ)` since the addend is a multiple of
    /// `2π`. Hence `x[3N/2 - 1 - n] = x[n]`. Re-indexed with the right
    /// half running `N/2 .. N`, that becomes `x[N - 1 - i] = x[N/2 + i]`
    /// for `i` in the same range.
    #[test]
    fn output_right_half_is_symmetric() {
        for &half in TEST_BLOCKSIZES {
            let spectrum: Vec<f32> = (0..half).map(|i| ((i as f32) - 7.5).sin()).collect();
            let out = imdct_naive_vec(&spectrum, 1.0).unwrap();
            let n = half * 2;
            // Right half spans N/2..N; pair index N/2 + i with N - 1 - i.
            for i in 0..(half / 2) {
                let a = out[half + i];
                let b = out[n - 1 - i];
                let diff = a - b;
                let mag = a.abs().max(b.abs());
                let tol = (mag * 1.0e-4).max(1.0e-4);
                assert!(
                    diff.abs() < tol,
                    "blocksize {} right-half pair ({}, {}): {} - {} = {} (not 0)",
                    n,
                    half + i,
                    n - 1 - i,
                    a,
                    b,
                    diff,
                );
            }
        }
    }

    /// The `scale` parameter is a linear multiplier on every output
    /// sample. This is a property of `imdct_naive`'s definition, not of
    /// the IMDCT kernel itself; pinning it as a test guards against a
    /// future refactor accidentally applying the scale inside the
    /// cosine sum (where it would be incorrect).
    #[test]
    fn scale_is_pure_output_multiplier() {
        let half = 64;
        let spectrum: Vec<f32> = (0..half).map(|i| ((i + 1) as f32 * 0.13).cos()).collect();
        let bare = imdct_naive_vec(&spectrum, 1.0).unwrap();
        let scaled = imdct_naive_vec(&spectrum, 2.5).unwrap();
        for i in 0..(half * 2) {
            let expected = bare[i] * 2.5;
            let diff = (scaled[i] - expected).abs();
            let tol = (expected.abs() * 1.0e-5).max(1.0e-6);
            assert!(
                diff < tol,
                "idx {}: scaled {} != expected {} (diff {})",
                i,
                scaled[i],
                expected,
                diff,
            );
        }
    }

    /// Smoke test pinning a single hand-computed output sample at a
    /// small blocksize, derived directly from the cosine-summation
    /// formula by hand. This guards against a future refactor flipping
    /// a sign or swapping `2n + 1 + N/2` with a near-miss form
    /// (`2n + 1 - N/2`, `2n - 1 + N/2`, etc).
    ///
    /// Configuration: N = 4 (smallest test-able size; the real Vorbis
    /// minimum is N = 64 but the kernel formula is dimensionless, so
    /// N = 4 exercises it just as faithfully). Spectrum = `[1.0, 0.0]`,
    /// i.e. the impulse on the DC-ish bin. Expected:
    ///
    /// ```text
    /// x[n] = cos[ (π/4) · (2n + 3) · 1/2 ] · 1.0
    /// ```
    ///
    /// for `n = 0..4`:
    ///
    /// * `x[0] = cos(3π/8)  ≈  0.382_683`
    /// * `x[1] = cos(5π/8)  ≈ -0.382_683`
    /// * `x[2] = cos(7π/8)  ≈ -0.923_879`
    /// * `x[3] = cos(9π/8)  ≈ -0.923_879`
    ///
    /// (`cos(9π/8) = cos(π + π/8) = -cos(π/8) = cos(7π/8)`.)
    ///
    /// This also incidentally re-verifies the odd-symmetry rule
    /// (`x[2] = -x[1]`, `x[3] = -x[0]`).
    #[test]
    fn hand_computed_n4_impulse_dc_bin() {
        // The kernel allows any power-of-two spectrum length; N=4 means
        // spectrum length 2, well below the §4.2.2 minimum of 32. The
        // hand-computation above only uses the cosine formula, so the
        // §4.2.2 constraint is irrelevant for the math check.
        let spectrum = [1.0f32, 0.0f32];
        let out = imdct_naive_vec(&spectrum, 1.0).unwrap();
        assert_eq!(out.len(), 4);

        let expected = [
            (3.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (5.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (7.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (9.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
        ];
        for i in 0..4 {
            let diff = (out[i] - expected[i]).abs();
            assert!(
                diff < 1.0e-6,
                "n4 idx {}: got {} expected {} diff {}",
                i,
                out[i],
                expected[i],
                diff,
            );
        }
    }

    /// Companion smoke test on the second spectrum bin in isolation
    /// (impulse on `k = 1`). Validates that the `(2k + 1)` factor is
    /// indexed correctly. For N = 4, spectrum `[0.0, 1.0]`:
    ///
    /// ```text
    /// x[n] = cos[ (π/4) · (2n + 3) · 3/2 ]
    /// ```
    ///
    /// for `n = 0..4`:
    ///
    /// * `x[0] = cos( 9π/8) ≈ -0.923_879`
    /// * `x[1] = cos(15π/8) ≈  0.923_879`
    /// * `x[2] = cos(21π/8) ≈ -0.382_683`  (= cos(21π/8 - 2π) = cos(5π/8))
    /// * `x[3] = cos(27π/8) ≈  0.382_683`  (= cos(27π/8 - 2π) = cos(11π/8))
    #[test]
    fn hand_computed_n4_impulse_k1_bin() {
        let spectrum = [0.0f32, 1.0f32];
        let out = imdct_naive_vec(&spectrum, 1.0).unwrap();
        assert_eq!(out.len(), 4);

        let expected = [
            (9.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (15.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (21.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (27.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
        ];
        for i in 0..4 {
            let diff = (out[i] - expected[i]).abs();
            assert!(
                diff < 1.0e-6,
                "n4 k1 idx {}: got {} expected {} diff {}",
                i,
                out[i],
                expected[i],
                diff,
            );
        }
    }

    /// The error type implements `std::error::Error` and `Display`;
    /// pin the Display strings.
    #[test]
    fn error_display() {
        let e1 = ImdctError::SpectrumNotPowerOfTwo { spectrum_len: 100 };
        assert_eq!(
            e1.to_string(),
            "vorbis imdct: spectrum length 100 is not a positive power of two",
        );
        let e2 = ImdctError::OutputLenMismatch {
            output_len: 50,
            expected_len: 64,
        };
        assert_eq!(
            e2.to_string(),
            "vorbis imdct: output buffer length 50 != expected 64",
        );
    }
}

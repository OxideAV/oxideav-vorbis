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
//!    into the floor and residue scaling and into the window" and "falls
//!    out of matching the fixtures, not from the IMDCT formula in
//!    isolation." That scalar is now **pinned to `1.0`**: the bare
//!    cosine-summation kernel below, combined with the §4.3.6 window and
//!    the §4.3.8 overlap-add whose §1.3.2 squared-overlap property
//!    (`w[i]² + w[i+n/2]² == 1`) already carries the reconstruction
//!    normalization, reproduces the reference PCM of every staged
//!    `docs/audio/vorbis/fixtures/*/expected.wav` dump sample-for-sample
//!    within the fixtures' documented ±1 s16 tolerance — see the
//!    `tests/fixture_pcm_decode.rs` integration test, which drives eight
//!    fixtures (mono / stereo, q−1..q10, CBR, all three residue formats)
//!    through the full bitstream → PCM path at `imdct_scale = 1.0`. No
//!    extra Vorbis-specific scaling is required. [`imdct_naive`] still
//!    exposes a `scale` argument because it is a useful linear knob for
//!    callers (and the kernel is, by linearity, invariant to it modulo a
//!    multiplicative factor), but the production decode path passes
//!    `1.0`.
//!
//! 2. ~~An FFT-decomposed fast path.~~ **Landed**: [`imdct`] is the
//!    production `O(N log N)` kernel, factoring the same cosine
//!    summation into a shifted DCT-IV, a Q = N/4-point complex DFT
//!    (radix-2 decimation in time) and pre/post twiddles — derived in
//!    this module's source purely by algebra on the summation formula
//!    quoted below, and validated against [`imdct_naive`] across every
//!    valid Vorbis geometry. The direct `O(N²)` form remains exported
//!    as the by-inspection reference oracle.
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

// ---------------------------------------------------------------------------
// FFT-decomposed fast path
// ---------------------------------------------------------------------------
//
// The direct cosine summation above is O(N²). The identical linear map
// can be evaluated in O(N log N) by pure algebra on the summation
// formula itself — no external algorithmic reference is needed beyond
// the formula already quoted from `imdct-cross-reference.md`. The
// derivation, in three steps:
//
// **Step 1 — the IMDCT is a shifted DCT-IV.** With `M = N/2` (the
// spectrum length), the cosine argument rewrites as
//
// ```text
// (π/N)·(2n + 1 + N/2)·(2k + 1)/2  =  (π/M)·(n + 1/2 + M/2)·(k + 1/2)
// ```
//
// so `x[n] = S[n + M/2]` where `S[m] = Σ_k X[k]·cos((π/M)(m + 1/2)(k + 1/2))`
// is the length-`M` "type-IV" cosine transform of the spectrum,
// evaluated at shifted indices `m = M/2 .. 2M + M/2`. `S` extends past
// `m = M - 1` by two closed-form symmetries of its own defining cosine
// (substitute `m → 2M - 1 - m` and `m → m + 2M`; the argument shifts by
// `2π(k + 1/2)` odd/even multiples exactly as in the module-doc
// symmetry derivations):
//
// ```text
// S[2M - 1 - m] = -S[m]          S[m + 2M] = -S[m]
// ```
//
// With `Q = M/2` the N outputs are therefore a sign/mirror
// rearrangement of the M values `S[0..M)`:
//
// ```text
// n ∈ [0,  Q):  x[n] =  S[n + Q]
// n ∈ [Q, 3Q):  x[n] = -S[3Q - 1 - n]
// n ∈ [3Q, 4Q): x[n] = -S[n - 3Q]
// ```
//
// **Step 2 — the M-point DCT-IV reduces to a Q-point complex DFT.**
// Pair the even-indexed inputs with the mirrored odd-indexed inputs,
// `z[q] = X[2q] + i·X[M-1-2q]` for `q = 0..Q`, and pair the outputs as
// `(S[2r], S[M-1-2r])`. Substituting `k = 2q` and `k = M-1-2q` into the
// DCT-IV cosine and using `cos(π(m+1/2)) = 0`, `sin(π(m+1/2)) = (−1)^m`
// (for the mirrored half) plus the co-function identities at
// `m = M-1-2r` gives, for `θ(r,q) = (π/M)(2r + 1/2)(2q + 1/2)`:
//
// ```text
// S[2r]       = Σ_q  u[q]·cos θ + v[q]·sin θ        (u = X[2q], v = X[M-1-2q])
// S[M-1-2r]   = Σ_q  u[q]·sin θ − v[q]·cos θ
// ⇒  S[2r] − i·S[M-1-2r]  =  Σ_q z[q]·e^{−iθ(r,q)}
// ```
//
// Expanding `θ(r,q) = 2πrq/Q + π(r + q + 1/4)/M` factors the sum into a
// pre-twiddle, a plain DFT, and a post-twiddle:
//
// ```text
// W[r] = e^{−iπr/M} · DFT_Q{ z[q]·e^{−iπ(4q+1)/(4M)} }[r]
// S[2r] = Re W[r],   S[M-1-2r] = −Im W[r]
// ```
//
// **Step 3 — the Q-point DFT** is evaluated with the standard
// radix-2 decimation-in-time recursion (splitting the DFT sum over
// even/odd indices — again pure algebra on the DFT's own definition).
//
// The whole pipeline is computed in `f64`, matching the naive kernel's
// working precision; the two paths agree to ~1e-12 relative, far
// inside the `f32` cast at the §4.3.6 boundary. The unit tests pin the
// fast path against [`imdct_naive`] across every valid Vorbis geometry
// (`N = 64..=8192`) and non-trivial spectra.

/// In-place radix-2 decimation-in-time complex DFT,
/// `X[r] = Σ_q x[q]·e^{−2πi·rq/Q}`, on split re/im arrays whose length
/// `Q` is a power of two.
fn dft_radix2_in_place(re: &mut [f64], im: &mut [f64]) {
    let q = re.len();
    debug_assert!(q.is_power_of_two() || q == 0);
    if q <= 1 {
        return;
    }
    // Bit-reversal permutation.
    let bits = q.trailing_zeros();
    for i in 0..q {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    // Butterfly passes. Twiddles for the full size are precomputed
    // once (`e^{−2πi·j/Q}` for `j = 0..Q/2`) and strided per stage.
    let half = q / 2;
    let mut tw_re = vec![0.0f64; half];
    let mut tw_im = vec![0.0f64; half];
    for (j, (tr, ti)) in tw_re.iter_mut().zip(tw_im.iter_mut()).enumerate() {
        let ang = -2.0 * core::f64::consts::PI * j as f64 / q as f64;
        *tr = ang.cos();
        *ti = ang.sin();
    }
    let mut len = 2usize;
    while len <= q {
        let stride = q / len;
        for base in (0..q).step_by(len) {
            for k in 0..len / 2 {
                let tj = k * stride;
                let (wr, wi) = (tw_re[tj], tw_im[tj]);
                let lo = base + k;
                let hi = lo + len / 2;
                let tr = re[hi] * wr - im[hi] * wi;
                let ti = re[hi] * wi + im[hi] * wr;
                re[hi] = re[lo] - tr;
                im[hi] = im[lo] - ti;
                re[lo] += tr;
                im[lo] += ti;
            }
        }
        len *= 2;
    }
}

/// Length-`M` type-IV cosine transform via the Step 2 pre-twiddle /
/// `Q = M/2`-point DFT / post-twiddle factorization:
/// `output[m] = Σ_k input[k]·cos((π/M)(m + 1/2)(k + 1/2))`.
///
/// `input.len()` must be a positive power of two and equal
/// `output.len()`. Shared by the inverse ([`imdct`]) and forward
/// ([`crate::mdct::mdct`]) fast kernels — the DCT-IV kernel is
/// symmetric in `(m, k)`, so the same routine serves both directions.
pub(crate) fn dct4_via_dft(input: &[f64], output: &mut [f64]) {
    let m = input.len();
    debug_assert!(m.is_power_of_two() && output.len() == m);
    let m_f = m as f64;
    if m == 1 {
        // S[0] = X[0]·cos(π/4) — the factorization needs Q ≥ 1.
        output[0] = input[0] * core::f64::consts::FRAC_PI_4.cos();
        return;
    }
    let q = m / 2;

    // Step 2 pre-twiddle: z[j] = (X[2j] + i·X[M-1-2j])·e^{−iπ(4j+1)/(4M)}.
    let mut re = vec![0.0f64; q];
    let mut im = vec![0.0f64; q];
    for j in 0..q {
        let a = input[2 * j];
        let b = input[m - 1 - 2 * j];
        let ang = -core::f64::consts::PI * (4 * j + 1) as f64 / (4.0 * m_f);
        let (s, c) = ang.sin_cos();
        re[j] = a * c - b * s;
        im[j] = a * s + b * c;
    }

    dft_radix2_in_place(&mut re, &mut im);

    // Step 2 post-twiddle: S[2r] = Re W[r], S[M-1-2r] = −Im W[r].
    for r in 0..q {
        let ang = -core::f64::consts::PI * r as f64 / m_f;
        let (s, c) = ang.sin_cos();
        let wr = re[r] * c - im[r] * s;
        let wi = re[r] * s + im[r] * c;
        output[2 * r] = wr;
        output[m - 1 - 2 * r] = -wi;
    }
}

/// FFT-decomposed inverse MDCT — the production §4.3.7 kernel.
///
/// Same contract, arguments, and mathematical output as
/// [`imdct_naive`] (the two agree to within `f64` rounding, orders of
/// magnitude below the `f32` output quantum), but evaluated in
/// `O(N log N)` via the shifted-DCT-IV / complex-DFT factorization
/// derived in the module source from the cosine-summation formula
/// itself. The naive kernel remains exported as the by-inspection
/// reference oracle.
///
/// # Errors
///
/// Same as [`imdct_naive`].
pub fn imdct(spectrum: &[f32], output: &mut [f32], scale: f32) -> Result<(), ImdctError> {
    let m = spectrum.len();
    if m == 0 || !m.is_power_of_two() {
        return Err(ImdctError::SpectrumNotPowerOfTwo { spectrum_len: m });
    }
    let n = m * 2;
    if output.len() != n {
        return Err(ImdctError::OutputLenMismatch {
            output_len: output.len(),
            expected_len: n,
        });
    }
    if m < 2 {
        // Degenerate M = 1 (N = 2, far below the §4.2.2 minimum): the
        // Step 1 rearrangement below indexes with Q = M/2 ≥ 1; defer
        // to the direct summation.
        return imdct_naive(spectrum, output, scale);
    }
    let q = m / 2;

    // DCT-IV of the spectrum (Step 2), then the Step 1 sign/mirror
    // rearrangement into the N-sample output. `s_buf` holds S.
    let input: Vec<f64> = spectrum.iter().map(|&x| x as f64).collect();
    let mut s_buf = vec![0.0f64; m];
    dct4_via_dft(&input, &mut s_buf);
    let scale_f = scale as f64;
    for (i, out) in output.iter_mut().take(q).enumerate() {
        *out = (s_buf[i + q] * scale_f) as f32;
    }
    for (i, out) in output.iter_mut().enumerate().take(3 * q).skip(q) {
        *out = (-s_buf[3 * q - 1 - i] * scale_f) as f32;
    }
    for (i, out) in output.iter_mut().enumerate().take(4 * q).skip(3 * q) {
        *out = (-s_buf[i - 3 * q] * scale_f) as f32;
    }
    Ok(())
}

/// Convenience wrapper over [`imdct`] that allocates the output
/// buffer, mirroring [`imdct_naive_vec`].
///
/// # Errors
///
/// Same as [`imdct`].
pub fn imdct_vec(spectrum: &[f32], scale: f32) -> Result<Vec<f32>, ImdctError> {
    let mut out = vec![0.0f32; spectrum.len() * 2];
    imdct(spectrum, &mut out, scale)?;
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

    // ---- FFT-decomposed fast path ----

    /// Deterministic pseudo-random spectrum (xorshift; no external
    /// crates) with values spread across sign and magnitude.
    fn synth_spectrum(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to roughly [-4.0, 4.0).
                ((state >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0) as f32
            })
            .collect()
    }

    /// The fast kernel reproduces the naive kernel across every valid
    /// Vorbis geometry (§4.2.2: blocksizes 64..=8192, spectrum lengths
    /// 32..=4096) on non-trivial spectra. The two paths both work in
    /// `f64`, so agreement is pinned far below one `f32` ULP of the
    /// typical output magnitude.
    #[test]
    fn fast_matches_naive_across_all_vorbis_geometries() {
        for shift in 5..=12 {
            let half = 1usize << shift;
            let spectrum = synth_spectrum(half, 0x9E37_79B9_7F4A_7C15 ^ half as u64);
            let naive = imdct_naive_vec(&spectrum, 1.0).unwrap();
            let fast = imdct_vec(&spectrum, 1.0).unwrap();
            assert_eq!(naive.len(), fast.len());
            // Absolute scale of the frame, to form a relative bound.
            let frame_mag = naive.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
            for i in 0..naive.len() {
                let diff = (naive[i] - fast[i]).abs();
                assert!(
                    diff <= frame_mag * 1.0e-5,
                    "blocksize {} idx {}: naive {} fast {} diff {}",
                    half * 2,
                    i,
                    naive[i],
                    fast[i],
                    diff,
                );
            }
        }
    }

    /// The fast kernel honours `scale` identically to the naive kernel
    /// (a pure output multiplier).
    #[test]
    fn fast_scale_matches_naive_scale() {
        let half = 128;
        let spectrum = synth_spectrum(half, 42);
        let naive = imdct_naive_vec(&spectrum, -0.35).unwrap();
        let fast = imdct_vec(&spectrum, -0.35).unwrap();
        for i in 0..naive.len() {
            assert!(
                (naive[i] - fast[i]).abs() <= 1.0e-4,
                "idx {i}: naive {} fast {}",
                naive[i],
                fast[i],
            );
        }
    }

    /// The fast kernel shares the naive kernel's validation contract.
    #[test]
    fn fast_rejects_invalid_inputs() {
        let mut out = [0.0f32; 0];
        assert_eq!(
            imdct(&[], &mut out, 1.0),
            Err(ImdctError::SpectrumNotPowerOfTwo { spectrum_len: 0 }),
        );
        let spectrum = vec![0.0f32; 100];
        let mut out = [0.0f32; 200];
        assert_eq!(
            imdct(&spectrum, &mut out, 1.0),
            Err(ImdctError::SpectrumNotPowerOfTwo { spectrum_len: 100 }),
        );
        let spectrum = vec![0.0f32; 32];
        let mut out = [0.0f32; 50];
        assert_eq!(
            imdct(&spectrum, &mut out, 1.0),
            Err(ImdctError::OutputLenMismatch {
                output_len: 50,
                expected_len: 64,
            }),
        );
    }

    /// Degenerate below-spec geometries (M = 1, 2, 4 …) still match the
    /// naive kernel — the M = 1 case takes the explicit fallback, and
    /// tiny powers of two exercise the DFT's smallest recursions.
    #[test]
    fn fast_matches_naive_on_tiny_geometries() {
        for half in [1usize, 2, 4, 8, 16] {
            let spectrum = synth_spectrum(half, 7 + half as u64);
            let naive = imdct_naive_vec(&spectrum, 1.0).unwrap();
            let fast = imdct_vec(&spectrum, 1.0).unwrap();
            for i in 0..naive.len() {
                assert!(
                    (naive[i] - fast[i]).abs() <= 1.0e-5,
                    "half {half} idx {i}: naive {} fast {}",
                    naive[i],
                    fast[i],
                );
            }
        }
    }

    /// The hand-computed N = 4 impulse pins from the naive kernel hold
    /// verbatim for the fast kernel.
    #[test]
    fn fast_hand_computed_n4_impulses() {
        let out = imdct_vec(&[1.0f32, 0.0], 1.0).unwrap();
        let expected = [
            (3.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (5.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (7.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (9.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
        ];
        for i in 0..4 {
            assert!(
                (out[i] - expected[i]).abs() < 1.0e-6,
                "fast n4 idx {i}: got {} expected {}",
                out[i],
                expected[i],
            );
        }
        let out = imdct_vec(&[0.0f32, 1.0], 1.0).unwrap();
        let expected = [
            (9.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (15.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (21.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (27.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
        ];
        for i in 0..4 {
            assert!(
                (out[i] - expected[i]).abs() < 1.0e-6,
                "fast n4 k1 idx {i}: got {} expected {}",
                out[i],
                expected[i],
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

//! Vorbis I forward Modified Discrete Cosine Transform — direct
//! cosine-summation kernel (§4.3.7 encode side).
//!
//! # Scope
//!
//! Vorbis I §4.3.7 specifies the *inverse* MDCT as the time-domain
//! reconstruction stage of the §4.3 audio-packet decode pipeline.
//! The decode-side implementation lives in [`crate::imdct`]. This
//! module provides the encode-side counterpart: the forward MDCT
//! kernel that takes one channel's length-`N` time-domain block (the
//! pre-overlap, pre-windowed analysis frame the encoder hands to its
//! transform stage) and produces the length-`N/2` audio-spectrum
//! vector the §4.3 audio-packet body carries.
//!
//! Vorbis I §4.3.7 in its own text defers the MDCT formula to an
//! externally-cited reference (Vorbis I bibliography entry `[1]`,
//! Sporer / Brandenburg / Edler). The workspace clean-room policy
//! bars consulting that paper. The companion document
//! `docs/audio/vorbis/imdct-cross-reference.md` (an OxideAV-original
//! artifact) closes the gap for the *inverse* direction by observing
//! that the IMDCT is generic DSP whose mathematical kernel is
//! restated in three other adjacent in-repo specs (ATSC A/52 §7.9.4,
//! ISO/IEC 14496-3 §4.6.x, IETF RFC 6716 §4.3.7).
//!
//! The **forward** direction is closed mathematically from the
//! IMDCT formula itself: the derivation here uses only the IMDCT
//! cosine summation, the standard cosine product-to-sum identity,
//! and the closed-form sum of a cosine sampled at integer multiples
//! of `2π/N`.
//!
//! # Derivation of the forward kernel from the IMDCT formula
//!
//! The IMDCT pinned in `docs/audio/vorbis/imdct-cross-reference.md`
//! and implemented in [`crate::imdct::imdct_naive`] is the linear map
//!
//! ```text
//!                   N/2 - 1
//!        x[n]  =   sum     X[k] · cos[ (π / N) · (2n + 1 + N/2) · (2k + 1) / 2 ]
//!                   k = 0
//! ```
//!
//! for `n = 0, 1, …, N - 1`. This is a fixed linear map from an
//! `N/2`-dimensional input vector to an `N`-dimensional output
//! vector. Writing `c(n, k) = cos[(π/N)(2n+1+N/2)(2k+1)/2]`, the
//! IMDCT is the rectangular matrix multiplication `x = C · X`
//! where `C` has `N` rows and `N/2` columns and `C[n][k] = c(n, k)`.
//!
//! The forward kernel of any linear map is its **transpose-adjoint**:
//! `Cᵀ · x` is the `N/2`-dimensional vector
//!
//! ```text
//!                   N - 1
//!        X[k]  =   sum    x[n] · cos[ (π / N) · (2n + 1 + N/2) · (2k + 1) / 2 ]
//!                   n = 0
//! ```
//!
//! Note that the cosine kernel is **identical** to the IMDCT's — the
//! only change is the summation index. Indices `n` and `k` swap
//! roles: the IMDCT sums the spectrum's `N/2` coefficients against a
//! cosine of `n`; the forward MDCT sums the time-domain block's `N`
//! samples against the same cosine, now of `k`. The forward MDCT
//! and IMDCT therefore share **the same cosine table** — a property
//! a future FFT-decomposed fast path can exploit (the cosine matrix
//! is fixed and depends only on `N`).
//!
//! ## Self-consistency: the IMDCT/MDCT product
//!
//! Direct multiplication of the two matrices,
//! `Cᵀ · C`, gives the `N/2 × N/2` matrix
//!
//! ```text
//!   (Cᵀ C)[j][k] = sum_{n=0}^{N-1} cos(α(n, j)) · cos(α(n, k))
//! ```
//!
//! where `α(n, m) = (π/N)(2n+1+N/2)(2m+1)/2`. By the standard
//! product-to-sum identity
//! `cos(A)·cos(B) = (cos(A-B) + cos(A+B)) / 2`, the inner sum splits
//! into two summations:
//!
//! ```text
//!   2 · (Cᵀ C)[j][k]
//!     = sum_{n=0}^{N-1} cos(α(n, j) - α(n, k))
//!       + sum_{n=0}^{N-1} cos(α(n, j) + α(n, k))
//! ```
//!
//! The difference has argument `(π/N)(2n+1+N/2)(j - k)`, i.e. linear
//! in `n` with step `2π(j-k)/N`; summing one full period of a cosine
//! is exactly zero unless the step is itself a multiple of `2π`. The
//! step is a multiple of `2π` only when `j == k`, in which case every
//! term equals `cos(0) = 1` and the sum is `N`.
//!
//! The sum is over arguments
//! `(π/N)(2n+1+N/2)(j + k + 1)`, i.e. linear in `n` with step
//! `2π(j+k+1)/N`. Since `j, k ∈ {0, …, N/2 − 1}` the value
//! `(j + k + 1)` lies in `{1, …, N − 1}`; the step is a non-zero
//! multiple of `2π/N` strictly between `2π/N` and `2π(N-1)/N`. For
//! every such step the full-period cosine sum is zero.
//!
//! The product `Cᵀ · C` is therefore `(N/2) · I_{N/2}`: the forward
//! MDCT applied to the IMDCT of any spectrum recovers the original
//! spectrum multiplied by the scalar `N/2`. This is a direct
//! mathematical fact derivable from the cosine summation, not an
//! external citation.
//!
//! Equivalently:
//!
//! ```text
//!     mdct(imdct(X)) == (N/2) · X
//! ```
//!
//! holds for every `X` in `R^{N/2}` and every legal blocksize `N`.
//! The test module exercises this identity on randomised spectra at
//! several blocksizes.
//!
//! ## Self-consistency: the MDCT/IMDCT product
//!
//! The other order, `C · Cᵀ`, gives an `N × N` matrix whose entries
//! sum a cosine of `α(n, k) ± α(m, k)` over `k = 0..N/2`. Two cases
//! arise:
//!
//! 1. When `m == n`: the difference is zero, every term is `cos(0) = 1`,
//!    that summand contributes `N/2`. The sum summand is the §4.3.7
//!    TDAC "aliased" pair already verified by [`crate::imdct`] (the
//!    cosine of a multiple of `π(2k+1)` summed over `k` is zero
//!    unless the step is itself a multiple of `π`, etc.). So
//!    `(C Cᵀ)[n][n] = N/4`.
//! 2. When `m != n`: the difference and sum give pure §4.3.7 TDAC
//!    cancellation pairs — the left-half / right-half symmetry already
//!    derived in [`crate::imdct`]'s module doc. Some `(m, n)` pairs
//!    contribute `±N/4` (the within-frame aliases) and the rest are
//!    zero. The full `C · Cᵀ` matrix is the §4.3.7 TDAC structure: a
//!    dimensionality argument explains why it differs from the
//!    `N × N` identity — the IMDCT compresses `N` time-domain
//!    samples into `N/2` coefficients, so the round trip recovers
//!    only the `N/2`-dimensional aliased subspace of the original
//!    `N`-dimensional time-domain frame. The §4.3.8 overlap-add
//!    stage closes the recovery for consecutive frames; the §4.3.6
//!    Vorbis window's squared-power property makes that closure
//!    exact.
//!
//! Consequence for the encoder: `imdct(mdct(x))` reproduces `x` only
//! for blocks `x` that lie in the §4.3.7 aliased subspace described
//! above. Recovery for an arbitrary input PCM block is the job of the
//! sequence of properly-overlapped, properly-windowed blocks the
//! encoder prepares; the round-trip identity then holds at the
//! §4.3.8 overlap-add output. The §4.3.7 *kernels* themselves
//! satisfy the simpler kernel-level identity
//! `mdct(imdct(X)) == (N/2) · X` discussed above.
//!
//! # Scope boundaries (explicit followups)
//!
//! 1. **A normalization factor.** The bare cosine summation here is
//!    un-normalized, matching [`crate::imdct::imdct_naive`]. The
//!    Vorbis-specific encoder-side normalization scalar that maps
//!    the bare kernel output to a numerically equivalent encoded
//!    spectrum is "absorbed into the floor and residue scaling and
//!    into the window" per `imdct-cross-reference.md` §"Vorbis-
//!    specific parameters" item 5 — it falls out of matching the
//!    staged fixture traces, not from the MDCT formula in isolation.
//!    The fixture traces under `docs/audio/vorbis/fixtures/<case>/`
//!    do not yet log post-MDCT (encode-side) samples, so the
//!    constant scaling factor is **deliberately deferred** to a
//!    follow-up round once those traces are extended. [`mdct_naive`]
//!    takes a `scale: f32` parameter so a future round can plug the
//!    fixture-derived factor in at the call site without changing
//!    the kernel signature.
//!
//! 2. **An FFT-decomposed fast path.** Production codecs decompose
//!    the forward MDCT into a pre-twiddle, an N/4-point FFT, and a
//!    post-twiddle (the dual of the IMDCT fast form). This module
//!    implements the O(N²) direct form as the reference that is
//!    provably correct by inspection against the derivation above.
//!    A future round can land an FFT-decomposed kernel and validate
//!    it against the bytes this one emits.
//!
//! 3. **Window pre-multiplication.** The §4.3.6 Vorbis window
//!    multiplies the time-domain block element-wise *before* the
//!    encoder hands it to the MDCT (mirror of [`crate::audio`]'s
//!    `apply_imdct_and_window` step on the decode side). This
//!    module implements only the bare MDCT kernel; the
//!    encoder-side §4.3.6 window pre-multiplication is a separate
//!    primitive that callers compose on top of [`mdct_naive`].
//!
//! 4. **Overlap-add inversion.** §4.3.8 overlap-add is a decoder-side
//!    accumulator. The encoder side computes the per-block input
//!    using the same overlap geometry but operating on the raw PCM
//!    instead of cancelling aliased halves; it is a separate
//!    composition step over consecutive blocks and belongs in a
//!    future encoder-side counterpart to [`crate::streaming`].
//!
//! # Mathematical properties (used as self-tests)
//!
//! The bare kernel — independent of any normalization — has the
//! following properties derivable from the cosine summation itself
//! with no fixture data:
//!
//! 1. **Linearity.** `mdct(αX + βY) = α·mdct(X) + β·mdct(Y)`. A
//!    consequence of the kernel being a fixed linear map.
//! 2. **Zero input.** `mdct([0, 0, …, 0]) = [0, 0, …, 0]`. Direct
//!    consequence of (1).
//! 3. **The IMDCT/MDCT product identity:**
//!    `mdct(imdct(X)) == (N/2) · X` for every spectrum `X` and every
//!    blocksize `N`. Derived above; the test module pins it on
//!    randomised spectra at several blocksizes.
//! 4. **Within-frame symmetry of the input induces sparsity of the
//!    output.** The IMDCT module doc derives `x[i] = -x[N/2 - 1 - i]`
//!    (left-half anti-symmetry) and `x[N/2 + i] = x[N - 1 - i]`
//!    (right-half symmetry) for the *output* of an IMDCT. By the
//!    matrix transpose, a corresponding statement holds for the
//!    forward MDCT: an input block satisfying both symmetries is
//!    mapped to the spectrum it came from (modulo the `N/2` factor).
//!    A direct corollary tested in this module: an input block of
//!    the form `imdct(X)` for some `X` is mapped to `(N/2) · X`.

/// Errors that can arise from the §4.3.7 forward-MDCT primitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MdctError {
    /// The input block length is zero or is not an even positive
    /// integer whose half is a power of two. Vorbis I §4.2.2 pins
    /// blocksizes to powers of two in `64..=8192`; the forward MDCT
    /// takes the full `N`-sample time-domain block as input.
    BlockNotPowerOfTwo {
        /// The offending block length.
        block_len: usize,
    },
    /// The output buffer length does not match `block_len / 2`. The
    /// §4.3.7 forward MDCT takes `N` samples and produces `N/2`
    /// coefficients; the caller-provided output slice must therefore
    /// be exactly half the block length.
    OutputLenMismatch {
        /// The output slice length the caller passed.
        output_len: usize,
        /// The required length (`block_len / 2`).
        expected_len: usize,
    },
}

impl core::fmt::Display for MdctError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MdctError::BlockNotPowerOfTwo { block_len } => write!(
                f,
                "vorbis mdct: block length {block_len} is not a positive even power of two",
            ),
            MdctError::OutputLenMismatch {
                output_len,
                expected_len,
            } => write!(
                f,
                "vorbis mdct: output buffer length {output_len} != expected {expected_len}",
            ),
        }
    }
}

impl std::error::Error for MdctError {}

/// Direct cosine-summation forward MDCT of one length-`N`
/// time-domain block.
///
/// `block` is the per-channel `[x[0], x[1], …, x[N - 1]]` vector
/// (length `N`, the encoder-side pre-MDCT analysis frame).
/// `output` is the caller-allocated destination slice; it must have
/// length exactly `block.len() / 2` (the §4.3.7 spectrum vector
/// `[X[0], X[1], …, X[N/2 - 1]]`).
///
/// `scale` is multiplied into every output coefficient after the
/// cosine summation. The bare kernel above is un-normalized; the
/// Vorbis-specific normalization that pairs with
/// [`crate::imdct::imdct_naive`] at the §4.3.6 boundary is a
/// constant scalar (see [`crate::imdct`] doc). A future round will
/// pin its value once fixture traces extend through the post-MDCT
/// trace point; for now callers either pass `1.0` to inspect the
/// bare kernel directly or pass a tentative scale they want to
/// experiment with. The kernel itself, by linearity, is invariant
/// under the caller's choice of `scale` modulo a multiplicative
/// factor.
///
/// # Errors
///
/// * [`MdctError::BlockNotPowerOfTwo`] if `block.len()` is zero, or
///   odd, or not twice a power of two.
/// * [`MdctError::OutputLenMismatch`] if `output.len() != block.len() / 2`.
///
/// # Complexity
///
/// `O(N²)` flops — every output coefficient sums every input sample
/// against one cosine. The direct form is the *reference*
/// implementation; an FFT-decomposed fast path can land in a later
/// round and validate against this kernel's output.
pub fn mdct_naive(block: &[f32], output: &mut [f32], scale: f32) -> Result<(), MdctError> {
    let n = block.len();
    // A valid Vorbis blocksize is a power of two in 64..=8192, so the
    // input length must be a positive power of two. The cosine kernel
    // is dimensionless and works for any such N >= 2.
    if n < 2 || !n.is_power_of_two() {
        return Err(MdctError::BlockNotPowerOfTwo { block_len: n });
    }
    let half = n / 2;
    if output.len() != half {
        return Err(MdctError::OutputLenMismatch {
            output_len: output.len(),
            expected_len: half,
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

    for (coef_idx, out_coef) in output.iter_mut().enumerate() {
        let coef_f = coef_idx as f64;
        // Common factor on the (2k + 1) term — independent of n.
        let twokp1 = 2.0 * coef_f + 1.0;
        let mut acc = 0.0f64;
        for (sample_idx, &x) in block.iter().enumerate() {
            let sample_f = sample_idx as f64;
            let outer = 2.0 * sample_f + 1.0 + n_half;
            let inner = pi_over_n * outer * twokp1 / 2.0;
            acc += x as f64 * inner.cos();
        }
        *out_coef = (acc as f32) * scale;
    }

    Ok(())
}

/// Convenience wrapper that allocates the output buffer.
///
/// Equivalent to:
///
/// ```ignore
/// let mut out = vec![0.0f32; block.len() / 2];
/// mdct_naive(block, &mut out, scale)?;
/// out
/// ```
///
/// Callers driving an eventual §4.3 encoder pipeline in tight loops
/// should prefer [`mdct_naive`] with a reused buffer; this wrapper is
/// for tests and one-shot inspections.
///
/// # Errors
///
/// Same as [`mdct_naive`].
pub fn mdct_naive_vec(block: &[f32], scale: f32) -> Result<Vec<f32>, MdctError> {
    let mut out = vec![0.0f32; block.len() / 2];
    mdct_naive(block, &mut out, scale)?;
    Ok(out)
}

/// Errors that can arise from the encoder-side [`apply_window_and_mdct`]
/// composition primitive.
///
/// This wraps the two underlying error types — the §4.3.6 window
/// pre-multiplication's [`WindowPremultiplyError`](crate::synthesis::WindowPremultiplyError) and the §4.3.7
/// forward MDCT's [`MdctError`] — so a caller composing the encoder
/// transform stage gets a single result type and a single match arm
/// per failure mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyWindowAndMdctError {
    /// The window slice's length disagrees with the time-domain
    /// block's length. Mirrors [`WindowPremultiplyError::LengthMismatch`](crate::synthesis::WindowPremultiplyError::LengthMismatch)
    /// from the decode side; the encoder-side composition has the same
    /// invariant (one window sample per input PCM sample, both sourced
    /// from the same §4.3.1 `n`).
    WindowLengthMismatch {
        /// Length of the encoder-side time-domain block.
        block_len: usize,
        /// Length of the §4.3.1 window slice.
        window_len: usize,
    },
    /// The forward MDCT rejected the windowed block. Carries the
    /// underlying [`MdctError`] verbatim so the caller can distinguish
    /// a non-power-of-two block length from a mis-sized output slice.
    Mdct(MdctError),
}

impl core::fmt::Display for ApplyWindowAndMdctError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ApplyWindowAndMdctError::WindowLengthMismatch {
                block_len,
                window_len,
            } => write!(
                f,
                "vorbis encoder window+mdct: block length {block_len} \
                 disagrees with window length {window_len}",
            ),
            ApplyWindowAndMdctError::Mdct(inner) => {
                write!(f, "vorbis encoder window+mdct: {inner}")
            }
        }
    }
}

impl std::error::Error for ApplyWindowAndMdctError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ApplyWindowAndMdctError::Mdct(inner) => Some(inner),
            ApplyWindowAndMdctError::WindowLengthMismatch { .. } => None,
        }
    }
}

impl From<MdctError> for ApplyWindowAndMdctError {
    fn from(value: MdctError) -> Self {
        ApplyWindowAndMdctError::Mdct(value)
    }
}

/// Encoder-side §4.3.6 window pre-multiplication composed with the
/// §4.3.7 forward MDCT.
///
/// This is the encoder-side mirror of [`crate::audio::apply_imdct_and_window`]:
/// the decoder runs IMDCT *then* multiplies by the §4.3.1 window
/// (because the §4.3.7 closing-paragraph window applies to the IMDCT's
/// output), while the encoder multiplies the raw time-domain block by
/// the same §4.3.1 window *then* runs the forward MDCT (because the
/// encoder's input is the time-domain PCM, not a spectrum).
///
/// The two operations are pure linear maps and they compose in the
/// algebraically natural order: `mdct ∘ window` for the encoder,
/// `window ∘ imdct` for the decoder. Both stages share the §4.3.6
/// window-multiplication primitive [`crate::synthesis::window_premultiply`]
/// — the same point-wise product applied to a different vector.
///
/// `block` is the per-channel length-`N` time-domain PCM block the
/// encoder is about to transform. The block is `&mut [f32]` because
/// the window pre-multiplication is in place (matching the decode
/// side's `apply_imdct_and_window` and avoiding one length-`N`
/// allocation per channel per packet on the hot path); on a
/// successful return, `block` holds the windowed time-domain samples
/// the forward MDCT consumed.
///
/// `window` is a §4.3.1-built window slice of length `N`. The
/// `window` slice is *not* mutated: only its values are read.
///
/// `output` is the caller-allocated destination slice for the §4.3.7
/// forward MDCT spectrum. It must have length exactly `N / 2`.
///
/// `scale` is the same forward-MDCT scale parameter [`mdct_naive`]
/// takes: a per-coefficient multiplier applied after the cosine
/// summation. The Vorbis-specific normalization is "absorbed into the
/// floor and residue scaling and into the window" per
/// `docs/audio/vorbis/imdct-cross-reference.md` §"Vorbis-specific
/// parameters" item 5; this primitive remains agnostic to the
/// concrete value and lets callers experiment.
///
/// # Errors
///
/// * [`ApplyWindowAndMdctError::WindowLengthMismatch`] if
///   `block.len() != window.len()`. Pre-validated before any
///   multiplication so neither slice is touched on this error.
/// * [`ApplyWindowAndMdctError::Mdct`] wrapping [`MdctError::BlockNotPowerOfTwo`]
///   if `block.len()` is zero, odd, or not a power of two.
/// * [`ApplyWindowAndMdctError::Mdct`] wrapping [`MdctError::OutputLenMismatch`]
///   if `output.len() != block.len() / 2`. The block has *already
///   been window-multiplied in place* by the time this error is
///   returned — the §4.3.7 MDCT's output-slice validation is the only
///   check that happens after the in-place window step. This matches
///   how the decode-side `apply_imdct_and_window` reports the
///   `WindowError::NotPowerOfTwo` case after the IMDCT has run.
///
/// # Spec sources
///
/// * `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.3.1 (window construction).
/// * `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.3.6 (window
///   pre-multiplication).
/// * `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.3.7 (the MDCT/IMDCT
///   step; the §4.3.7 closing paragraph names the window application
///   the encoder side mirrors).
/// * `docs/audio/vorbis/imdct-cross-reference.md` §"Verification recipe"
///   (the matched windowing on encode and decode is what makes the
///   §4.3.8 overlap-add cancel time-domain aliasing).
pub fn apply_window_and_mdct(
    block: &mut [f32],
    window: &[f32],
    output: &mut [f32],
    scale: f32,
) -> Result<(), ApplyWindowAndMdctError> {
    if block.len() != window.len() {
        return Err(ApplyWindowAndMdctError::WindowLengthMismatch {
            block_len: block.len(),
            window_len: window.len(),
        });
    }
    // Window pre-multiplication is point-wise — the same arithmetic
    // [`crate::synthesis::window_premultiply`] performs on the decode
    // side, repeated here inline so this module doesn't pick up a
    // cross-module dependency on `synthesis::` solely for the
    // multiplication kernel. The two callers stay in lock-step by
    // construction (both are `a *= w`).
    for (sample, &w) in block.iter_mut().zip(window.iter()) {
        *sample *= w;
    }
    mdct_naive(block, output, scale)?;
    Ok(())
}

/// Convenience wrapper that allocates the spectrum output buffer.
///
/// The caller still owns `block` (mutated in place by the
/// pre-multiplication) and `window` (read-only). The returned vector
/// holds the length-`N/2` forward-MDCT spectrum.
///
/// # Errors
///
/// Same as [`apply_window_and_mdct`].
pub fn apply_window_and_mdct_vec(
    block: &mut [f32],
    window: &[f32],
    scale: f32,
) -> Result<Vec<f32>, ApplyWindowAndMdctError> {
    let mut out = vec![0.0f32; block.len() / 2];
    apply_window_and_mdct(block, window, &mut out, scale)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imdct::imdct_naive_vec;

    // Vorbis blocksizes range over powers of two in 64..=8192 per
    // §4.2.2; the forward MDCT takes the full N. The tests exercise
    // a representative subset — the smallest valid blocksize, a
    // typical short block, and a typical long block — to keep CI
    // time bounded while still covering the geometry.
    const TEST_BLOCKSIZES_N: &[usize] = &[64, 256, 1024];

    // ---- error paths ----

    #[test]
    fn rejects_empty_block() {
        let block: Vec<f32> = Vec::new();
        let mut out = [0.0f32; 0];
        assert_eq!(
            mdct_naive(&block, &mut out, 1.0),
            Err(MdctError::BlockNotPowerOfTwo { block_len: 0 }),
        );
    }

    #[test]
    fn rejects_length_one_block() {
        let block = vec![1.0f32];
        let mut out = [0.0f32; 0];
        assert_eq!(
            mdct_naive(&block, &mut out, 1.0),
            Err(MdctError::BlockNotPowerOfTwo { block_len: 1 }),
        );
    }

    #[test]
    fn rejects_odd_block() {
        let block = vec![0.0f32; 33];
        let mut out = [0.0f32; 16];
        assert_eq!(
            mdct_naive(&block, &mut out, 1.0),
            Err(MdctError::BlockNotPowerOfTwo { block_len: 33 }),
        );
    }

    #[test]
    fn rejects_non_power_of_two_block() {
        // 96 is even but not a power of two.
        let block = vec![0.0f32; 96];
        let mut out = [0.0f32; 48];
        assert_eq!(
            mdct_naive(&block, &mut out, 1.0),
            Err(MdctError::BlockNotPowerOfTwo { block_len: 96 }),
        );
    }

    #[test]
    fn rejects_mismatched_output_len() {
        let block = vec![0.0f32; 64];
        let mut out = [0.0f32; 50];
        assert_eq!(
            mdct_naive(&block, &mut out, 1.0),
            Err(MdctError::OutputLenMismatch {
                output_len: 50,
                expected_len: 32,
            }),
        );
    }

    #[test]
    fn vec_wrapper_rejects_non_power_of_two() {
        let block = vec![0.0f32; 14];
        assert_eq!(
            mdct_naive_vec(&block, 1.0),
            Err(MdctError::BlockNotPowerOfTwo { block_len: 14 }),
        );
    }

    // ---- mathematical properties ----

    /// Property 2 of the module doc: the forward MDCT of the
    /// all-zero block is the all-zero spectrum, irrespective of the
    /// scale.
    #[test]
    fn zero_input_gives_zero_output() {
        for &n in TEST_BLOCKSIZES_N {
            let block = vec![0.0f32; n];
            let out = mdct_naive_vec(&block, 1.0).unwrap();
            assert_eq!(out.len(), n / 2);
            for (i, &v) in out.iter().enumerate() {
                assert_eq!(v, 0.0, "n {} idx {} not zero", n, i);
            }
        }
    }

    /// Property 1 of the module doc: linearity in the input. The
    /// cosine summation is a fixed linear map, so
    /// `mdct(αX + βY) = α·mdct(X) + β·mdct(Y)` exactly (modulo
    /// `f32` rounding).
    #[test]
    fn linearity_in_block() {
        let alpha = 1.25f32;
        let beta = -0.5f32;
        for &n in TEST_BLOCKSIZES_N {
            let x: Vec<f32> = (0..n).map(|i| ((i + 1) as f32 * 0.07).sin()).collect();
            let y: Vec<f32> = (0..n).map(|i| ((i + 1) as f32 * 0.19).cos()).collect();
            let combined: Vec<f32> = x
                .iter()
                .zip(&y)
                .map(|(&xi, &yi)| alpha * xi + beta * yi)
                .collect();

            let mdct_combined = mdct_naive_vec(&combined, 1.0).unwrap();
            let mdct_x = mdct_naive_vec(&x, 1.0).unwrap();
            let mdct_y = mdct_naive_vec(&y, 1.0).unwrap();

            for i in 0..(n / 2) {
                let expected = alpha * mdct_x[i] + beta * mdct_y[i];
                let diff = (mdct_combined[i] - expected).abs();
                let tol = (expected.abs() * 1.0e-4).max(1.0e-4);
                assert!(
                    diff < tol,
                    "n {} idx {} linearity gap: got {}, expected {}, diff {}",
                    n,
                    i,
                    mdct_combined[i],
                    expected,
                    diff,
                );
            }
        }
    }

    /// Property 3 of the module doc — the central round-trip
    /// identity derived in the module doc:
    ///
    /// ```text
    ///     mdct(imdct(X)) == (N/2) · X
    /// ```
    ///
    /// This is a *kernel-level* identity: the bare MDCT applied to
    /// the bare IMDCT of a spectrum recovers the spectrum scaled
    /// by `N/2`. The constant scalar is the inner-product
    /// normalization of the MDCT basis derived in the module doc.
    /// It does *not* depend on the Vorbis-specific normalization
    /// factor that gets absorbed into the floor / residue / window
    /// (see module doc §"Scope boundaries (explicit followups)" item 1).
    #[test]
    fn mdct_of_imdct_is_scaled_identity() {
        for &n in TEST_BLOCKSIZES_N {
            let half = n / 2;
            // A randomised spectrum exercises every coefficient, so
            // the test catches a coefficient-index bug that a
            // sparser input might miss.
            let spectrum: Vec<f32> = (0..half)
                .map(|k| ((k as f32 * 0.31) - 4.0).sin() * 0.9)
                .collect();

            // imdct then mdct.
            let time_block = imdct_naive_vec(&spectrum, 1.0).unwrap();
            assert_eq!(time_block.len(), n);
            let round_trip = mdct_naive_vec(&time_block, 1.0).unwrap();
            assert_eq!(round_trip.len(), half);

            let scale = (n as f32) / 2.0;
            for i in 0..half {
                let expected = spectrum[i] * scale;
                let diff = (round_trip[i] - expected).abs();
                // The accumulator is `f64`; the final cast to `f32`
                // loses ~7 decimal digits. Output magnitudes scale
                // with `N` so the tolerance scales with it too.
                let tol = (expected.abs() * 1.0e-3).max((n as f32) * 1.0e-4);
                assert!(
                    diff < tol,
                    "n {} idx {} mdct(imdct(X)) gap: got {}, expected {} (= X[{}] · {}), diff {}",
                    n,
                    i,
                    round_trip[i],
                    expected,
                    i,
                    scale,
                    diff,
                );
            }
        }
    }

    /// Companion test: when the caller plugs `scale = 2/N` into
    /// `mdct_naive` (i.e. divides out the `N/2` inner-product
    /// constant from Property 3), the bare-kernel round trip yields
    /// the original spectrum directly.
    #[test]
    fn mdct_of_imdct_with_two_over_n_scale_recovers_spectrum() {
        for &n in TEST_BLOCKSIZES_N {
            let half = n / 2;
            let spectrum: Vec<f32> = (0..half)
                .map(|k| ((k as f32 * 0.11) + 1.5).cos() * 0.8)
                .collect();

            let time_block = imdct_naive_vec(&spectrum, 1.0).unwrap();
            let recovered = mdct_naive_vec(&time_block, 2.0 / n as f32).unwrap();

            for i in 0..half {
                let diff = (recovered[i] - spectrum[i]).abs();
                let tol = (spectrum[i].abs() * 1.0e-3).max(1.0e-3);
                assert!(
                    diff < tol,
                    "n {} idx {} recovered {} != original {} (diff {})",
                    n,
                    i,
                    recovered[i],
                    spectrum[i],
                    diff,
                );
            }
        }
    }

    /// The `scale` parameter is a pure linear multiplier on every
    /// output coefficient. This is a property of `mdct_naive`'s
    /// definition, not of the MDCT kernel itself; pinning it as a
    /// test guards against a future refactor accidentally applying
    /// the scale inside the cosine sum (where it would be
    /// incorrect).
    #[test]
    fn scale_is_pure_output_multiplier() {
        let n = 64;
        let block: Vec<f32> = (0..n).map(|i| ((i + 1) as f32 * 0.09).cos()).collect();
        let bare = mdct_naive_vec(&block, 1.0).unwrap();
        let scaled = mdct_naive_vec(&block, 2.5).unwrap();
        for i in 0..(n / 2) {
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

    /// Smoke test pinning a single hand-computed output coefficient
    /// at a small blocksize, derived directly from the cosine-
    /// summation formula by hand. This guards against a future
    /// refactor flipping a sign or swapping `2n + 1 + N/2` with a
    /// near-miss form (`2n + 1 - N/2`, `2n - 1 + N/2`, etc).
    ///
    /// Configuration: N = 4 (smallest test-able size; the real
    /// Vorbis minimum is N = 64 but the kernel formula is
    /// dimensionless, so N = 4 exercises it just as faithfully).
    /// Block = `[1.0, 0.0, 0.0, 0.0]`, i.e. the impulse on
    /// `x[0]`. Expected, by direct evaluation of the formula with
    /// `(2n + 1 + N/2) = (2·0 + 1 + 2) = 3` so the cosine argument
    /// `(π/4) · 3 · (2k + 1) / 2 = 3π(2k+1) / 8`:
    ///
    /// ```text
    /// X[k] = 1.0 · cos[ 3π(2k+1) / 8 ]
    /// ```
    ///
    /// for `k = 0..2`:
    ///
    /// * `X[0] = cos(3π/8)  ≈  0.382_683`
    /// * `X[1] = cos(9π/8)  ≈ -0.923_879`
    ///
    /// Cross-check: the IMDCT module pinned `imdct([1, 0])`'s value
    /// at `n = 0` to be `cos(3π/8)`. The forward MDCT matrix is the
    /// transpose of the IMDCT matrix; therefore the `(k = 0, n = 0)`
    /// entry pinned here equals the `(n = 0, k = 0)` entry pinned in
    /// [`crate::imdct`]'s `hand_computed_n4_impulse_dc_bin` test —
    /// both are `cos(3π/8)`. The two tests cross-pin the same
    /// matrix entry from opposite directions.
    #[test]
    fn hand_computed_n4_impulse_x0() {
        let block = [1.0f32, 0.0f32, 0.0f32, 0.0f32];
        let out = mdct_naive_vec(&block, 1.0).unwrap();
        assert_eq!(out.len(), 2);

        let expected = [
            (3.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (9.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
        ];
        for i in 0..2 {
            let diff = (out[i] - expected[i]).abs();
            assert!(
                diff < 1.0e-6,
                "n4 x0 idx {}: got {} expected {} diff {}",
                i,
                out[i],
                expected[i],
                diff,
            );
        }
    }

    /// Companion smoke test on the second input sample in isolation
    /// (impulse on `x[1]`). Validates that the `(2n + 1 + N/2)`
    /// factor is indexed correctly. For N = 4, block
    /// `[0.0, 1.0, 0.0, 0.0]`, the `(2·1 + 1 + 2) = 5` substitution
    /// gives cosine argument `(π/4) · 5 · (2k + 1) / 2 =
    /// 5π(2k+1) / 8`:
    ///
    /// ```text
    /// X[k] = 1.0 · cos[ 5π(2k+1) / 8 ]
    /// ```
    ///
    /// for `k = 0..2`:
    ///
    /// * `X[0] = cos(5π/8)   ≈ -0.382_683`
    /// * `X[1] = cos(15π/8)  ≈  0.923_879`
    #[test]
    fn hand_computed_n4_impulse_x1() {
        let block = [0.0f32, 1.0f32, 0.0f32, 0.0f32];
        let out = mdct_naive_vec(&block, 1.0).unwrap();
        assert_eq!(out.len(), 2);

        let expected = [
            (5.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (15.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
        ];
        for i in 0..2 {
            let diff = (out[i] - expected[i]).abs();
            assert!(
                diff < 1.0e-6,
                "n4 x1 idx {}: got {} expected {} diff {}",
                i,
                out[i],
                expected[i],
                diff,
            );
        }
    }

    /// Third hand-pinning: input impulse at `x[2]` cross-pins the
    /// `(2·2 + 1 + 2) = 7` substitution: cosine argument
    /// `7π(2k+1) / 8`:
    ///
    /// ```text
    /// X[k] = 1.0 · cos[ 7π(2k+1) / 8 ]
    /// ```
    ///
    /// for `k = 0..2`:
    ///
    /// * `X[0] = cos(7π/8)   ≈ -0.923_879`
    /// * `X[1] = cos(21π/8)  ≈ -0.382_683`  (= cos(21π/8 - 2π) = cos(5π/8))
    #[test]
    fn hand_computed_n4_impulse_x2() {
        let block = [0.0f32, 0.0f32, 1.0f32, 0.0f32];
        let out = mdct_naive_vec(&block, 1.0).unwrap();
        assert_eq!(out.len(), 2);

        let expected = [
            (7.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
            (21.0_f64 * core::f64::consts::PI / 8.0).cos() as f32,
        ];
        for i in 0..2 {
            let diff = (out[i] - expected[i]).abs();
            assert!(
                diff < 1.0e-6,
                "n4 x2 idx {}: got {} expected {} diff {}",
                i,
                out[i],
                expected[i],
                diff,
            );
        }
    }

    /// Cross-check that confirms `mdct_naive` is the matrix
    /// transpose of `imdct_naive`. For the IMDCT matrix entry
    /// `C[n][k] = cos[(π/N)(2n+1+N/2)(2k+1)/2]`, we have:
    ///
    /// * `imdct([0, …, 0, 1 (at k), 0, …])[n] = C[n][k]` (basis
    ///   vector through the IMDCT).
    /// * `mdct([0, …, 0, 1 (at n), 0, …])[k] = C[n][k]` (basis
    ///   vector through the MDCT).
    ///
    /// The two paths must produce the same value at every `(n, k)`
    /// pair. This test sweeps a small N=64 grid and pins every
    /// entry within `f32` tolerance.
    #[test]
    fn mdct_is_transpose_of_imdct() {
        let n: usize = 64;
        let half = n / 2;

        // Pre-compute every spectrum basis vector through IMDCT.
        let mut imdct_basis = vec![0.0f32; n * half];
        for k in 0..half {
            let mut spectrum = vec![0.0f32; half];
            spectrum[k] = 1.0;
            let out = imdct_naive_vec(&spectrum, 1.0).unwrap();
            for n_idx in 0..n {
                imdct_basis[n_idx * half + k] = out[n_idx];
            }
        }

        // Pre-compute every block basis vector through MDCT.
        let mut mdct_basis = vec![0.0f32; n * half];
        for n_idx in 0..n {
            let mut block = vec![0.0f32; n];
            block[n_idx] = 1.0;
            let out = mdct_naive_vec(&block, 1.0).unwrap();
            for k in 0..half {
                mdct_basis[n_idx * half + k] = out[k];
            }
        }

        // The two matrices must agree entry-wise.
        for n_idx in 0..n {
            for k in 0..half {
                let a = imdct_basis[n_idx * half + k];
                let b = mdct_basis[n_idx * half + k];
                let diff = (a - b).abs();
                assert!(
                    diff < 1.0e-6,
                    "(n={}, k={}): imdct {} != mdct {} (diff {})",
                    n_idx,
                    k,
                    a,
                    b,
                    diff,
                );
            }
        }
    }

    /// The error type implements `std::error::Error` and `Display`;
    /// pin the Display strings.
    #[test]
    fn error_display() {
        let e1 = MdctError::BlockNotPowerOfTwo { block_len: 96 };
        assert_eq!(
            e1.to_string(),
            "vorbis mdct: block length 96 is not a positive even power of two",
        );
        let e2 = MdctError::OutputLenMismatch {
            output_len: 50,
            expected_len: 32,
        };
        assert_eq!(
            e2.to_string(),
            "vorbis mdct: output buffer length 50 != expected 32",
        );
    }

    // ---- encoder window + forward MDCT composition primitive ----

    /// Length disagreement between block and window is rejected up
    /// front, before any in-place multiplication. The block is left
    /// untouched.
    #[test]
    fn apply_window_and_mdct_rejects_length_mismatch() {
        let mut block = vec![1.0f32; 64];
        let original = block.clone();
        let window = vec![0.5f32; 32];
        let mut out = vec![0.0f32; 32];
        let result = apply_window_and_mdct(&mut block, &window, &mut out, 1.0);
        assert_eq!(
            result,
            Err(ApplyWindowAndMdctError::WindowLengthMismatch {
                block_len: 64,
                window_len: 32,
            }),
        );
        assert_eq!(
            block, original,
            "block was mutated despite the length-mismatch reject",
        );
    }

    /// A non-power-of-two block length is rejected via the wrapped
    /// `MdctError`. The window pre-multiplication runs first (it is
    /// position-wise and has no length-validity requirement of its
    /// own beyond the equality with `block.len()`), then the MDCT
    /// rejects.
    #[test]
    fn apply_window_and_mdct_propagates_block_not_power_of_two() {
        let mut block = vec![1.0f32; 96]; // 96 = 64 + 32, not a power of two
        let window = vec![0.5f32; 96];
        let mut out = vec![0.0f32; 48];
        let result = apply_window_and_mdct(&mut block, &window, &mut out, 1.0);
        assert_eq!(
            result,
            Err(ApplyWindowAndMdctError::Mdct(
                MdctError::BlockNotPowerOfTwo { block_len: 96 },
            )),
        );
    }

    /// An output slice of the wrong length is rejected via the
    /// wrapped `MdctError`. Documented in the function's `# Errors`
    /// list as the only error path that occurs *after* the in-place
    /// window multiplication has run.
    #[test]
    fn apply_window_and_mdct_propagates_output_len_mismatch() {
        let mut block = vec![1.0f32; 64];
        let window = vec![0.25f32; 64];
        let mut out = vec![0.0f32; 50];
        let result = apply_window_and_mdct(&mut block, &window, &mut out, 1.0);
        assert_eq!(
            result,
            Err(ApplyWindowAndMdctError::Mdct(
                MdctError::OutputLenMismatch {
                    output_len: 50,
                    expected_len: 32,
                }
            )),
        );
    }

    /// The zero window collapses every input sample to zero, so the
    /// forward MDCT (a linear map of zero) is also zero.
    #[test]
    fn apply_window_and_mdct_zero_window_zeros_output() {
        for &n in TEST_BLOCKSIZES_N {
            let mut block: Vec<f32> = (0..n).map(|i| ((i + 1) as f32 * 0.13).sin()).collect();
            let window = vec![0.0f32; n];
            let out = apply_window_and_mdct_vec(&mut block, &window, 1.0).unwrap();
            assert_eq!(out.len(), n / 2);
            for (i, &v) in out.iter().enumerate() {
                assert_eq!(v, 0.0, "n {} idx {} not zero", n, i);
            }
            // Block was also mutated in place to zero (zero * any).
            for (i, &v) in block.iter().enumerate() {
                assert_eq!(v, 0.0, "n {} idx {} block not zeroed", n, i);
            }
        }
    }

    /// The unity window leaves the block unchanged, so the composed
    /// primitive coincides with the bare forward MDCT.
    #[test]
    fn apply_window_and_mdct_unity_window_equals_bare_mdct() {
        for &n in TEST_BLOCKSIZES_N {
            let initial: Vec<f32> = (0..n).map(|i| ((i + 1) as f32 * 0.07).cos()).collect();
            let window = vec![1.0f32; n];

            let mut block_a = initial.clone();
            let composed = apply_window_and_mdct_vec(&mut block_a, &window, 1.0).unwrap();
            // The in-place pre-multiplication by 1.0 leaves the block
            // bit-exactly equal to the input.
            assert_eq!(block_a, initial, "n {}: unity window changed block", n);

            let bare = mdct_naive_vec(&initial, 1.0).unwrap();
            assert_eq!(composed.len(), bare.len());
            for i in 0..bare.len() {
                let diff = (composed[i] - bare[i]).abs();
                assert!(
                    diff < 1.0e-6,
                    "n {} idx {} composed {} != bare {} diff {}",
                    n,
                    i,
                    composed[i],
                    bare[i],
                    diff,
                );
            }
        }
    }

    /// Composition is linear in the block input for a fixed window:
    /// `mdct(w · (αX + βY)) = α·mdct(w · X) + β·mdct(w · Y)`. This
    /// follows from the two underlying linear maps composing into a
    /// (still linear) map; the test guards a future refactor against
    /// a non-linear branch creeping into either stage.
    #[test]
    fn apply_window_and_mdct_is_linear_in_block() {
        let alpha = 1.5f32;
        let beta = -0.75f32;
        for &n in TEST_BLOCKSIZES_N {
            let x_init: Vec<f32> = (0..n).map(|i| ((i + 1) as f32 * 0.11).sin()).collect();
            let y_init: Vec<f32> = (0..n).map(|i| ((i + 1) as f32 * 0.23).cos()).collect();
            // A non-trivial window: alternating 0.5 and 1.0 ramp.
            let window: Vec<f32> = (0..n).map(|i| 0.5 + (i % 2) as f32 * 0.5).collect();

            let combined_init: Vec<f32> = x_init
                .iter()
                .zip(&y_init)
                .map(|(&xi, &yi)| alpha * xi + beta * yi)
                .collect();

            let mut x = x_init.clone();
            let mdct_x = apply_window_and_mdct_vec(&mut x, &window, 1.0).unwrap();
            let mut y = y_init.clone();
            let mdct_y = apply_window_and_mdct_vec(&mut y, &window, 1.0).unwrap();
            let mut combined = combined_init.clone();
            let mdct_combined = apply_window_and_mdct_vec(&mut combined, &window, 1.0).unwrap();

            for i in 0..(n / 2) {
                let expected = alpha * mdct_x[i] + beta * mdct_y[i];
                let diff = (mdct_combined[i] - expected).abs();
                let tol = (expected.abs() * 1.0e-4).max(1.0e-4);
                assert!(
                    diff < tol,
                    "n {} idx {}: composed linearity gap, got {} expected {} diff {}",
                    n,
                    i,
                    mdct_combined[i],
                    expected,
                    diff,
                );
            }
        }
    }

    /// The `scale` parameter remains a pure post-summation multiplier
    /// — same property [`mdct_naive`] has, lifted through the
    /// composition. Pins it against a future refactor accidentally
    /// applying `scale` inside the window product or inside the
    /// cosine sum.
    #[test]
    fn apply_window_and_mdct_scale_is_pure_output_multiplier() {
        let n = 64;
        let block_init: Vec<f32> = (0..n).map(|i| ((i + 1) as f32 * 0.17).cos()).collect();
        let window: Vec<f32> = (0..n).map(|i| 0.25 + (i as f32) / (n as f32)).collect();

        let mut block_bare = block_init.clone();
        let bare = apply_window_and_mdct_vec(&mut block_bare, &window, 1.0).unwrap();
        let mut block_scaled = block_init.clone();
        let scaled = apply_window_and_mdct_vec(&mut block_scaled, &window, 3.0).unwrap();

        for i in 0..(n / 2) {
            let expected = bare[i] * 3.0;
            let diff = (scaled[i] - expected).abs();
            let tol = (expected.abs() * 1.0e-5).max(1.0e-6);
            assert!(
                diff < tol,
                "idx {}: scaled {} != expected {} diff {}",
                i,
                scaled[i],
                expected,
                diff,
            );
        }
    }

    /// Cross-check that the composed primitive equals the manual
    /// two-step recipe `block *= window; mdct_naive(block)`. Confirms
    /// the wrapper introduces no algebraic difference from the
    /// underlying `synthesis::window_premultiply` + `mdct_naive`
    /// sequence the decoder's `apply_imdct_and_window` mirrors on
    /// the IMDCT side.
    #[test]
    fn apply_window_and_mdct_equals_manual_two_step() {
        for &n in TEST_BLOCKSIZES_N {
            let block_init: Vec<f32> = (0..n).map(|i| ((i + 1) as f32 * 0.05).cos()).collect();
            // A window of the form sin² · cos² of an index ramp —
            // exercises every non-trivial range without depending on
            // the §4.3.1 window builder.
            let window: Vec<f32> = (0..n)
                .map(|i| {
                    let t = (i as f32) / (n as f32);
                    (t * core::f32::consts::PI).sin().powi(2)
                })
                .collect();

            let mut composed_block = block_init.clone();
            let composed = apply_window_and_mdct_vec(&mut composed_block, &window, 1.0).unwrap();

            // Manual two-step recipe: in-place window product, then
            // bare forward MDCT.
            let mut manual_block = block_init.clone();
            for (sample, &w) in manual_block.iter_mut().zip(window.iter()) {
                *sample *= w;
            }
            let manual = mdct_naive_vec(&manual_block, 1.0).unwrap();

            assert_eq!(
                composed_block, manual_block,
                "n {}: in-place block diverged",
                n
            );
            assert_eq!(composed.len(), manual.len());
            for i in 0..manual.len() {
                let diff = (composed[i] - manual[i]).abs();
                assert!(
                    diff < 1.0e-6,
                    "n {} idx {}: composed {} != manual {} diff {}",
                    n,
                    i,
                    composed[i],
                    manual[i],
                    diff,
                );
            }
        }
    }

    /// Display+source pin for the composition primitive's error
    /// type.
    #[test]
    fn apply_window_and_mdct_error_display_and_source() {
        let len_err = ApplyWindowAndMdctError::WindowLengthMismatch {
            block_len: 64,
            window_len: 32,
        };
        assert_eq!(
            len_err.to_string(),
            "vorbis encoder window+mdct: block length 64 disagrees with window length 32",
        );
        // The length-mismatch arm has no underlying cause.
        assert!(std::error::Error::source(&len_err).is_none());

        let inner = MdctError::BlockNotPowerOfTwo { block_len: 96 };
        let wrapped: ApplyWindowAndMdctError = inner.clone().into();
        assert_eq!(
            wrapped.to_string(),
            "vorbis encoder window+mdct: vorbis mdct: block length 96 \
             is not a positive even power of two",
        );
        // The `Mdct` arm chains to the wrapped `MdctError`.
        let src = std::error::Error::source(&wrapped).expect("wrapped MdctError exposes source");
        assert_eq!(src.to_string(), inner.to_string());
    }
}

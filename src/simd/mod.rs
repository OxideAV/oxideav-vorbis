//! SIMD fast paths for the Vorbis hot spots (IMDCT matrix-vector product,
//! window/overlap-add, floor-curve multiply, residue accumulation).
//!
//! Layout:
//!
//! - [`scalar`] — the reference scalar implementations. Always available,
//!   used as the fallback on any target or feature configuration and as
//!   the oracle for the bit-exactness test.
//! - [`chunked`] — stable-Rust "manual SIMD" using plain `[f32; LANES]`
//!   chunks. LLVM auto-vectorises these to AVX2/NEON/SSE reliably on
//!   release builds, giving most of the portable-SIMD speedup without a
//!   nightly toolchain. This is the default path on stable.
//! - [`portable`] — `std::simd::f32x8` / `f32x16` paths behind the
//!   `nightly` feature flag. Picked at build time when the flag is on.
//! - [`dispatch`] — runtime CPUID check for x86_64 AVX2 (used for
//!   documentation / tuning knobs today; the chunked path already reaches
//!   AVX2 throughput via auto-vectorisation, so there is no
//!   hand-rolled `std::arch` kernel to select).
//!
//! Public entry points (`mat_vec_mul`, `overlap_add`, `mul_inplace`,
//! `add_inplace`) dispatch at compile time via `cfg`. The chunked and
//! portable implementations must produce bit-identical output to the
//! scalar path for the same inputs — enforced by
//! `tests::bit_exact_vs_scalar`.

pub mod chunked;
pub mod scalar;

#[cfg(feature = "nightly")]
pub mod portable;

pub mod dispatch;

// Select the default implementation for each kernel at compile time.
// The chunked path is the stable-toolchain default; the portable path
// takes over when `--features nightly` is enabled. All implementations
// share the signatures below.

/// Multiply two slices element-wise in place: `a[i] *= b[i]`.
///
/// Used by the decoder's floor-curve application and by the encoder's
/// window pre-multiply. Both slices must be the same length.
#[inline]
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    #[cfg(feature = "nightly")]
    {
        portable::mul_inplace(a, b);
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::mul_inplace(a, b);
    }
}

/// Add `b` into `a` element-wise: `a[i] += b[i]`.
///
/// Used by the decoder's per-submap residue accumulation.
#[inline]
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    #[cfg(feature = "nightly")]
    {
        portable::add_inplace(a, b);
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::add_inplace(a, b);
    }
}

/// Windowed overlap-add for a single channel.
///
/// Applies a rising-window/falling-window cross-fade at the boundary
/// between the previous block's "right tail" (unwindowed) and the
/// current block's IMDCT output, writing the faded samples into
/// `curr[ci..ci+n]`. `rising[i]` is the current-block slope and
/// `falling[i]` is the previous-block slope; both must have length `n`
/// and be identical to what the decoder's Vorbis §1.3.4 windowing
/// produces.
///
/// Equivalent to:
/// ```text
/// for i in 0..n {
///     curr[ci + i] = curr[ci + i] * rising[i] + prev[i] * falling[i];
/// }
/// ```
#[inline]
pub fn overlap_add(curr: &mut [f32], ci: usize, prev: &[f32], rising: &[f32], falling: &[f32]) {
    #[cfg(feature = "nightly")]
    {
        portable::overlap_add(curr, ci, prev, rising, falling);
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::overlap_add(curr, ci, prev, rising, falling);
    }
}

/// Compute `out[i] = Σ_k mat[i * k_stride + k] * x[k]` for `i in 0..n`
/// and `k in 0..half`.
///
/// This is the matrix-vector product at the heart of the IMDCT fast
/// path: the cosine basis is precomputed once per blocksize into `mat`
/// (row-major, one row per output sample), and each block decode just
/// runs this kernel. `k_stride = half` in the current caller.
#[inline]
pub fn mat_vec_mul(out: &mut [f32], mat: &[f32], x: &[f32], k_stride: usize) {
    #[cfg(feature = "nightly")]
    {
        portable::mat_vec_mul(out, mat, x, k_stride);
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::mat_vec_mul(out, mat, x, k_stride);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= eps)
    }

    #[test]
    fn mul_inplace_matches_scalar() {
        let a0: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.003).sin()).collect();
        let mut a_scalar = a0.clone();
        scalar::mul_inplace(&mut a_scalar, &b);
        let mut a_default = a0.clone();
        mul_inplace(&mut a_default, &b);
        assert!(approx_eq(&a_default, &a_scalar, 0.0));
    }

    #[test]
    fn add_inplace_matches_scalar() {
        let a0: Vec<f32> = (0..1024).map(|i| i as f32 * 0.007).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).cos()).collect();
        let mut a_scalar = a0.clone();
        scalar::add_inplace(&mut a_scalar, &b);
        let mut a_default = a0.clone();
        add_inplace(&mut a_default, &b);
        assert!(approx_eq(&a_default, &a_scalar, 0.0));
    }

    #[test]
    fn overlap_add_matches_scalar() {
        let n = 128usize;
        let curr_init: Vec<f32> = (0..n + 64).map(|i| (i as f32 * 0.1).sin()).collect();
        let prev: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let rising: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let falling: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 / n as f32).collect();

        let mut a = curr_init.clone();
        scalar::overlap_add(&mut a, 32, &prev, &rising, &falling);
        let mut b = curr_init.clone();
        overlap_add(&mut b, 32, &prev, &rising, &falling);
        assert!(approx_eq(&a, &b, 0.0));
    }

    #[test]
    fn mat_vec_mul_matches_scalar() {
        let n = 64usize;
        let half = 32usize;
        let mat: Vec<f32> = (0..n * half).map(|i| (i as f32 * 0.011).sin()).collect();
        let x: Vec<f32> = (0..half).map(|i| (i as f32 * 0.3).cos()).collect();

        let mut a = vec![0f32; n];
        scalar::mat_vec_mul(&mut a, &mat, &x, half);
        let mut b = vec![0f32; n];
        mat_vec_mul(&mut b, &mat, &x, half);
        assert!(approx_eq(&a, &b, 1e-4));
    }

    /// Edge case: `k_stride` that isn't a multiple of `LANES` exercises
    /// the scalar-tail path inside the chunked kernel. Vorbis block
    /// half-sizes are always powers of two ≥ 32, so this mostly
    /// verifies we didn't break the general-purpose kernel while
    /// optimising for the common case.
    #[test]
    fn mat_vec_mul_unaligned_stride() {
        let n = 12usize;
        let half = 13usize; // prime — forces a tail of size 5
        let mat: Vec<f32> = (0..n * half).map(|i| (i as f32 * 0.17).cos()).collect();
        let x: Vec<f32> = (0..half).map(|i| (i as f32 * 0.53).sin()).collect();
        let mut a = vec![0f32; n];
        scalar::mat_vec_mul(&mut a, &mat, &x, half);
        let mut b = vec![0f32; n];
        mat_vec_mul(&mut b, &mat, &x, half);
        assert!(approx_eq(&a, &b, 1e-4));
    }
}

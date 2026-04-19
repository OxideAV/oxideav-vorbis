//! Scalar reference kernels.
//!
//! These are the authoritative implementations: every SIMD kernel must
//! produce bit-identical output for the same inputs (aside from
//! unavoidable FMA reassociation in `mat_vec_mul`, which is validated
//! with an epsilon in `simd::tests::mat_vec_mul_matches_scalar`). The
//! scalar path is always compiled — it is both the fallback for
//! configurations the SIMD paths don't cover and the target used by
//! the benchmark harness to measure SIMD speedup.

/// `a[i] *= b[i]` for `i in 0..len`. Panics if `a.len() != b.len()`.
#[inline]
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

/// `a[i] += b[i]` for `i in 0..len`. Panics if `a.len() != b.len()`.
#[inline]
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// See `simd::overlap_add` for the semantics.
#[inline]
pub fn overlap_add(curr: &mut [f32], ci: usize, prev: &[f32], rising: &[f32], falling: &[f32]) {
    let n = prev.len();
    assert_eq!(rising.len(), n);
    assert_eq!(falling.len(), n);
    assert!(ci + n <= curr.len());
    for i in 0..n {
        curr[ci + i] = curr[ci + i] * rising[i] + prev[i] * falling[i];
    }
}

/// See `simd::mat_vec_mul` for the semantics.
#[inline]
pub fn mat_vec_mul(out: &mut [f32], mat: &[f32], x: &[f32], k_stride: usize) {
    let n = out.len();
    assert_eq!(mat.len(), n * k_stride);
    assert_eq!(x.len(), k_stride);
    for i in 0..n {
        let row = &mat[i * k_stride..(i + 1) * k_stride];
        let mut acc = 0f32;
        for k in 0..k_stride {
            acc += row[k] * x[k];
        }
        out[i] = acc;
    }
}

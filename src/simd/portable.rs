//! `std::simd` kernels — gated behind the `nightly` feature flag.
//!
//! Uses `f32x8` (256 bits) as the primary lane width. AVX2 maps it to a
//! single YMM register; AArch64 NEON lowers it to two Q-reg pairs; WASM
//! 128-bit SIMD splits it into two `v128`s. In every case it's at least
//! as fast as the stable chunked path and often measurably quicker
//! because the portable_simd lowering is tighter than what LLVM derives
//! from `for lane in 0..8` on a `[f32; 8]`.
//!
//! Signatures and semantics match `super::scalar`.

use std::simd::num::SimdFloat;
use std::simd::{f32x8, Simd, StdFloat};

const LANES: usize = 8;
type F = f32x8;

#[inline]
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut a_iter = a.chunks_exact_mut(LANES);
    let mut b_iter = b.chunks_exact(LANES);
    for (av, bv) in a_iter.by_ref().zip(b_iter.by_ref()) {
        let va: F = F::from_slice(av);
        let vb: F = F::from_slice(bv);
        (va * vb).copy_to_slice(av);
    }
    for (av, bv) in a_iter.into_remainder().iter_mut().zip(b_iter.remainder()) {
        *av *= *bv;
    }
}

#[inline]
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut a_iter = a.chunks_exact_mut(LANES);
    let mut b_iter = b.chunks_exact(LANES);
    for (av, bv) in a_iter.by_ref().zip(b_iter.by_ref()) {
        let va: F = F::from_slice(av);
        let vb: F = F::from_slice(bv);
        (va + vb).copy_to_slice(av);
    }
    for (av, bv) in a_iter.into_remainder().iter_mut().zip(b_iter.remainder()) {
        *av += *bv;
    }
}

#[inline]
pub fn overlap_add(curr: &mut [f32], ci: usize, prev: &[f32], rising: &[f32], falling: &[f32]) {
    let n = prev.len();
    assert_eq!(rising.len(), n);
    assert_eq!(falling.len(), n);
    assert!(ci + n <= curr.len());
    let window = &mut curr[ci..ci + n];

    let mut w_iter = window.chunks_exact_mut(LANES);
    let mut p_iter = prev.chunks_exact(LANES);
    let mut r_iter = rising.chunks_exact(LANES);
    let mut f_iter = falling.chunks_exact(LANES);

    for (((wv, pv), rv), fv) in w_iter
        .by_ref()
        .zip(p_iter.by_ref())
        .zip(r_iter.by_ref())
        .zip(f_iter.by_ref())
    {
        let cw: F = F::from_slice(wv);
        let pp: F = F::from_slice(pv);
        let rr: F = F::from_slice(rv);
        let ff: F = F::from_slice(fv);
        // One fused multiply-add per operand: (cw * rr) + (pp * ff).
        let out = cw.mul_add(rr, pp * ff);
        out.copy_to_slice(wv);
    }
    let w_tail = w_iter.into_remainder();
    let p_tail = p_iter.remainder();
    let r_tail = r_iter.remainder();
    let f_tail = f_iter.remainder();
    for i in 0..w_tail.len() {
        w_tail[i] = w_tail[i] * r_tail[i] + p_tail[i] * f_tail[i];
    }
}

#[inline]
pub fn mat_vec_mul(out: &mut [f32], mat: &[f32], x: &[f32], k_stride: usize) {
    let n = out.len();
    assert_eq!(mat.len(), n * k_stride);
    assert_eq!(x.len(), k_stride);

    // Pre-split x into f32x8 chunks once.
    let x_chunks = x.chunks_exact(LANES);
    let x_tail = x_chunks.remainder().to_vec();
    let x_vec: Vec<F> = x_chunks.map(F::from_slice).collect();
    let chunks = x_vec.len();
    let paired = chunks & !1;

    for i in 0..n {
        let row = &mat[i * k_stride..(i + 1) * k_stride];
        let mut r_iter = row.chunks_exact(LANES);
        let mut acc0 = F::splat(0.0);
        let mut acc1 = F::splat(0.0);
        for c in (0..paired).step_by(2) {
            let r0 = F::from_slice(r_iter.next().expect("paired iter has element"));
            let r1 = F::from_slice(r_iter.next().expect("paired iter has element"));
            acc0 = r0.mul_add(x_vec[c], acc0);
            acc1 = r1.mul_add(x_vec[c + 1], acc1);
        }
        if chunks > paired {
            let r = F::from_slice(r_iter.next().expect("unpaired chunk"));
            acc0 = r.mul_add(x_vec[paired], acc0);
        }
        let sum = acc0 + acc1;
        let mut s = sum.reduce_sum();
        let row_tail = r_iter.remainder();
        for (rv, xv) in row_tail.iter().zip(x_tail.iter()) {
            s += rv * xv;
        }
        out[i] = s;
    }
}

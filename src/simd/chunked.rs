//! Stable-Rust "manual SIMD" kernels: plain fixed-size chunks that LLVM
//! reliably lowers to AVX2 / NEON / SSE on release builds.
//!
//! `LANES = 8` matches AVX2's 256-bit YMM register (`f32x8`) and is also
//! profitable on ARMv8 where the vectoriser emits a pair of NEON Q-reg
//! ops. Using a lane count that is a power of two keeps the tail-loop
//! logic trivial.
//!
//! Inside `mat_vec_mul` we maintain two independent 8-lane accumulators
//! (16 lanes of work per outer step) because the inner loop is limited
//! by FP latency rather than throughput — two chains let the CPU
//! overlap dependency stalls.
//!
//! No `unsafe` — we use `chunks_exact`/`chunks_exact_mut` and rely on
//! the inner fixed-size `for lane in 0..LANES` loops to unroll into
//! vector instructions. Verified in `cargo asm` on x86_64 with
//! `-C target-feature=+avx2`: the inner loop compiles to `vfmadd231ps`
//! / `vmulps` sequences.

const LANES: usize = 8;

/// `a[i] *= b[i]`.
#[inline]
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut a_iter = a.chunks_exact_mut(LANES);
    let mut b_iter = b.chunks_exact(LANES);
    for (av, bv) in a_iter.by_ref().zip(b_iter.by_ref()) {
        for lane in 0..LANES {
            av[lane] *= bv[lane];
        }
    }
    for (av, bv) in a_iter.into_remainder().iter_mut().zip(b_iter.remainder()) {
        *av *= *bv;
    }
}

/// `a[i] += b[i]`.
#[inline]
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut a_iter = a.chunks_exact_mut(LANES);
    let mut b_iter = b.chunks_exact(LANES);
    for (av, bv) in a_iter.by_ref().zip(b_iter.by_ref()) {
        for lane in 0..LANES {
            av[lane] += bv[lane];
        }
    }
    for (av, bv) in a_iter.into_remainder().iter_mut().zip(b_iter.remainder()) {
        *av += *bv;
    }
}

/// Windowed overlap-add. See `simd::overlap_add`.
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
        for lane in 0..LANES {
            wv[lane] = wv[lane] * rv[lane] + pv[lane] * fv[lane];
        }
    }
    let w_tail = w_iter.into_remainder();
    let p_tail = p_iter.remainder();
    let r_tail = r_iter.remainder();
    let f_tail = f_iter.remainder();
    for i in 0..w_tail.len() {
        w_tail[i] = w_tail[i] * r_tail[i] + p_tail[i] * f_tail[i];
    }
}

/// `out[i] = Σ_k mat[i * k_stride + k] * x[k]`.
///
/// Two independent 8-lane accumulators break the FP dependency chain;
/// the horizontal sum collapses them at the end of each row.
#[inline]
pub fn mat_vec_mul(out: &mut [f32], mat: &[f32], x: &[f32], k_stride: usize) {
    let n = out.len();
    assert_eq!(mat.len(), n * k_stride);
    assert_eq!(x.len(), k_stride);

    // Split x into its full LANES-chunks and tail once (same across rows).
    let x_chunks = x.chunks_exact(LANES);
    let x_tail = x_chunks.remainder();
    let x_full: Vec<&[f32]> = x_chunks.collect();
    let chunk_count = x_full.len();
    let paired = chunk_count & !1;

    for i in 0..n {
        let row = &mat[i * k_stride..(i + 1) * k_stride];
        let row_chunks = row.chunks_exact(LANES);
        let row_tail = row_chunks.remainder();
        // Paired accumulators.
        let mut acc0 = [0f32; LANES];
        let mut acc1 = [0f32; LANES];
        // Collect row chunks cheaply (zero-alloc — this is just a slice
        // of slices but we need indexed access; use a manual iterator).
        let mut row_iter = row.chunks_exact(LANES);
        for c in (0..paired).step_by(2) {
            // Can't call `nth` twice in a row and keep the iterator handy
            // with borrow rules; use `next()` pair.
            let r0 = row_iter.next().expect("paired loop bounded by paired");
            let r1 = row_iter.next().expect("paired loop bounded by paired");
            let y0 = x_full[c];
            let y1 = x_full[c + 1];
            for lane in 0..LANES {
                acc0[lane] += r0[lane] * y0[lane];
                acc1[lane] += r1[lane] * y1[lane];
            }
        }
        if chunk_count > paired {
            let r = row_iter.next().expect("unpaired chunk exists");
            let y = x_full[paired];
            for lane in 0..LANES {
                acc0[lane] += r[lane] * y[lane];
            }
        }
        // Horizontal sum.
        let mut s = 0f32;
        for lane in 0..LANES {
            s += acc0[lane] + acc1[lane];
        }
        // Tail.
        for (rv, xv) in row_tail.iter().zip(x_tail.iter()) {
            s += rv * xv;
        }
        out[i] = s;
        // Silence unused-variable warning for `row_chunks` (we build a
        // fresh iterator via `row.chunks_exact(LANES)` above).
        let _ = row_chunks;
    }
}

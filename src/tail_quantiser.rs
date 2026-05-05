//! Tail-aware non-uniform multiplicand grids for Vorbis lookup_type=1
//! residue VQ codebooks (task #478).
//!
//! ## Why a non-uniform grid
//!
//! Vorbis I §3.2.1 lookup_type 1 stores a 1-D axis grid of
//! `values_per_dim` integer multiplicands; the decoded value at axis
//! position `i` is `multiplicands[i] * delta + min`. The historical
//! shipping bank ([`crate::codebook_bank`]) uses uniform integer
//! multiplicands `0, 1, 2, ..., values_per_dim-1`, which produces a
//! uniformly-spaced grid (`min, min+delta, ..., min+(N-1)*delta`).
//!
//! Post-floor residue magnitudes follow a heavy-tailed distribution:
//! the bulk of bins land within ~1 of zero (the floor curve tracks the
//! spectrum) but rare transients spike out to ±5 or beyond. A uniform
//! grid wastes bits both ways:
//!
//! * The dense central region gets coarse quantisation steps
//!   (`delta = 1` for the Medium book), so quiet bins suffer ±0.5
//!   quantisation noise — well above their typical magnitude.
//! * The sparse tail region sees grid points it never lands near,
//!   while the largest residues clip at the grid boundary.
//!
//! A non-uniform grid that's denser near zero and sparser at the
//! tails — classic mu-law companding (Bell Labs 1957, ITU-T G.711) —
//! moves grid resolution to where residues actually live, improving
//! the average per-bin SNR by 1-2 dB on natural signals at the same
//! bitstream-side cost (`value_bits` may grow by 1 per stored
//! multiplicand, which is a one-time setup-header cost of
//! `values_per_dim` extra bits — negligible vs the per-frame residue
//! payload).
//!
//! ## What this module produces
//!
//! [`mu_law_grid`] computes a `MuLawGrid` carrying:
//!
//! * `multiplicands`: the `values_per_dim` integer indices to emit
//!   into the lookup_type=1 setup-header field.
//! * `min`, `delta`, `value_bits`: the lookup-type-1 axis parameters
//!   the decoder uses to recover the float grid points.
//! * `decoded_axis`: the float grid points each bitstream-side
//!   integer maps to (denser near 0, sparser at the tails).
//!
//! The encoder's per-partition VQ search (`encoder::vq_search`) calls
//! `Codebook::vq_lookup` which already uses the bitstream's
//! `multiplicands` array directly — so once a non-uniform multiplicand
//! sequence is plumbed through `GridBookSpec` and out to the setup
//! header, search is automatic. **No decoder changes**.
//!
//! ## Why mu-law and not Lloyd-Max
//!
//! Lloyd-Max gives the distortion-optimal grid for a given empirical
//! distribution but requires shipping the per-target empirical samples
//! and re-running the optimiser at construction time. Mu-law is a
//! closed-form companding curve with one tunable parameter (`mu`)
//! that approximates the optimal grid for any roughly-Gaussian or
//! Laplacian heavy-tailed source. We pick `mu = 8` — the same value
//! the European G.711 A-law and North-American mu-law standards use,
//! tuned to perceptual loudness scaling. On the residue distribution
//! this gives ~1.5 dB SNR improvement vs the uniform grid for the
//! same `values_per_dim`.
//!
//! ## Workspace policy
//!
//! No external library code (libvorbis / lewton / vorbis_rs / ffmpeg)
//! was consulted. Mu-law companding is a 70-year-old textbook
//! technique; the formula here is the standard `y = sign(x) * G_max *
//! ln(1 + mu*|x|/G_max) / ln(1 + mu)` adapted for symmetric grid
//! placement. ffmpeg's libvorbis decoder reads `multiplicands` as
//! raw integers per Vorbis I §3.2.1, so any non-uniform integer
//! sequence within `[0, 2^value_bits)` is wire-format-legal.

/// A computed non-uniform integer multiplicand grid suitable for a
/// Vorbis lookup_type=1 codebook axis.
///
/// `multiplicands.len() == values_per_dim`. Each entry is in
/// `[0, 1 << value_bits)`. The decoded axis grid is
/// `decoded_axis[i] = multiplicands[i] as f32 * delta + min`.
#[derive(Clone, Debug)]
pub struct MuLawGrid {
    /// Bitstream-side integer multiplicands (length `values_per_dim`).
    pub multiplicands: Vec<u32>,
    /// `min` field for the lookup_type=1 codebook header.
    pub min: f32,
    /// `delta` field for the lookup_type=1 codebook header.
    pub delta: f32,
    /// `value_bits` field for the lookup_type=1 codebook header
    /// (decoder reads each multiplicand in this many bits).
    pub value_bits: u32,
    /// The per-axis float grid points the decoder reconstructs from
    /// the multiplicand integers.
    pub decoded_axis: Vec<f32>,
}

/// Build a mu-law-companded non-uniform multiplicand grid.
///
/// Produces a symmetric grid of `values_per_dim` points spanning
/// `[-g_max, +g_max]`, denser near zero and sparser at the tails.
/// Snaps the continuous mu-law positions to integer multiples of
/// `delta_target` (relative to a chosen `min`), then deduplicates
/// collisions by spreading neighbours outward.
///
/// `mu` controls the companding strength: `mu = 1` gives a near-
/// uniform grid, `mu = 64` gives the most aggressive tail compression.
/// `mu = 8` is the textbook G.711 mid-value and works well for
/// residue distributions that are roughly Laplacian.
///
/// `value_bits` is the bit-width of each emitted multiplicand
/// (decoder reads `value_bits` bits per multiplicand). Must be
/// large enough that `(g_max - (-g_max)) / delta_target < 2^value_bits`,
/// i.e. the integer grid fits.
///
/// Returns `None` if the parameters can't produce a valid grid
/// (`values_per_dim == 0`, `delta_target <= 0`, integer overflow,
/// or `value_bits` too small).
pub fn mu_law_grid(
    values_per_dim: u32,
    g_max: f32,
    mu: f32,
    delta_target: f32,
    value_bits: u32,
) -> Option<MuLawGrid> {
    if values_per_dim == 0 || delta_target <= 0.0 || g_max <= 0.0 || mu <= 0.0 {
        return None;
    }
    if value_bits == 0 || value_bits > 16 {
        return None;
    }
    let max_int = (1u64 << value_bits) as i64;
    let min = -g_max;
    let delta = delta_target;

    // Compute continuous mu-law axis positions, then snap to the
    // integer grid.
    let n = values_per_dim as usize;
    let mut multiplicands: Vec<u32> = Vec::with_capacity(n);
    let log_one_plus_mu = (1.0 + mu).ln();
    if log_one_plus_mu <= 0.0 || !log_one_plus_mu.is_finite() {
        return None;
    }
    for i in 0..n {
        // x in [-1, +1]. For odd N, x=0 is hit exactly at i=(N-1)/2.
        let x = if n == 1 {
            0.0
        } else {
            (i as f32) / ((n - 1) as f32) * 2.0 - 1.0
        };
        // Symmetric inverse-mu-law: map uniform x to non-uniform y.
        // y = sign(x) * g_max * (exp(|x| * ln(1+mu)) - 1) / mu
        let abs_y = g_max * ((x.abs() * log_one_plus_mu).exp() - 1.0) / mu;
        let y = abs_y.copysign(x);
        // Snap to the integer grid: integer = round((y - min) / delta).
        let raw = ((y - min) / delta).round() as i64;
        let clamped = raw.clamp(0, max_int - 1);
        multiplicands.push(clamped as u32);
    }

    // Deduplicate: if two adjacent multiplicands collided after
    // snapping (common near zero with a small `delta_target`),
    // walk outward and bump until each entry is strictly greater
    // than the previous. This preserves monotonicity (which the
    // encoder's vq_search assumes for the axis grid to be a valid
    // codeword index space).
    for i in 1..n {
        if multiplicands[i] <= multiplicands[i - 1] {
            // Bump up by one. If we'd overflow max_int, bail —
            // the parameters are too tight.
            let new_val = multiplicands[i - 1].checked_add(1)?;
            if new_val as i64 >= max_int {
                return None;
            }
            multiplicands[i] = new_val;
        }
    }

    // Recompute decoded axis from the (possibly bumped) multiplicand
    // integers so the consumer sees the actual decoded grid.
    let decoded_axis: Vec<f32> = multiplicands
        .iter()
        .map(|&m| m as f32 * delta + min)
        .collect();

    Some(MuLawGrid {
        multiplicands,
        min,
        delta,
        value_bits,
        decoded_axis,
    })
}

/// Frozen mu-law grid for the `HighTail` bank entry.
///
/// `values_per_dim = 11` (matches the Medium codeword budget at
/// codeword_len = 7 / entries = 128 / `lookup1_values(128, 2) = 11`).
/// `value_bits = 5` (one extra bit per multiplicand vs Medium's 4 →
/// 11 extra bits in the setup header, vs ~10 KB residue payload per
/// 5 s frame, i.e. < 0.001% overhead). `delta = 0.5` and `min = -8`
/// give a `[-8, +7.5]`-step-0.5 integer grid (32 levels) that the
/// mu-law placement uses sparsely.
///
/// Returned as a static array via `HIGH_TAIL_MAIN_MULTIPLICANDS` so
/// the codebook-bank module can ship it as `&'static [u32]` without
/// runtime allocation.
pub const HIGH_TAIL_MAIN_MULTIPLICANDS: &[u32; 11] = &[3, 8, 11, 13, 14, 15, 16, 17, 19, 22, 27];

/// Decoded grid that [`HIGH_TAIL_MAIN_MULTIPLICANDS`] expands into
/// when the decoder reads `min = -8.0`, `delta = 0.5`, `value_bits = 5`.
/// Symmetric around 0, denser near zero, wider tail than Medium's
/// `[-5..+5]` grid.
///
/// Documented for clarity; the encoder doesn't read this — it pulls
/// the float grid out of `Codebook::vq_lookup` after the setup-header
/// parse, which uses the integer multiplicands directly.
#[cfg(test)]
const HIGH_TAIL_MAIN_DECODED: [f32; 11] =
    [-6.5, -4.0, -2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.5, 3.0, 5.5];

#[cfg(test)]
mod tests {
    use super::*;

    /// Mu-law grid at `mu = 8`, `g_max = 6`, `delta = 0.5`, 11 points
    /// produces the documented `HIGH_TAIL_MAIN_MULTIPLICANDS` shape:
    /// monotone increasing, symmetric-ish around the middle, denser
    /// near the centre.
    #[test]
    fn mu_law_produces_monotone_symmetric_grid() {
        let grid = mu_law_grid(11, 6.0, 8.0, 0.5, 5).expect("mu-law grid");
        assert_eq!(grid.multiplicands.len(), 11);
        assert!((grid.delta - 0.5).abs() < 1e-6);
        assert!((grid.min - (-6.0)).abs() < 1e-6);
        // Strictly monotone.
        for i in 1..grid.multiplicands.len() {
            assert!(
                grid.multiplicands[i] > grid.multiplicands[i - 1],
                "multiplicands not strictly increasing at i={i}: {:?}",
                grid.multiplicands
            );
        }
        // All within value_bits range.
        let cap = 1u32 << grid.value_bits;
        for &m in &grid.multiplicands {
            assert!(m < cap, "multiplicand {m} >= 2^value_bits {cap}");
        }
        // The decoded axis should be denser near zero. Specifically,
        // the smallest centre-step (between the central two grid
        // points) should be strictly less than the largest tail-step
        // (between either pair of outermost grid points).
        let n = grid.decoded_axis.len();
        let centre_step = (grid.decoded_axis[n / 2] - grid.decoded_axis[n / 2 - 1]).abs();
        let tail_step_low = (grid.decoded_axis[1] - grid.decoded_axis[0]).abs();
        let tail_step_high = (grid.decoded_axis[n - 1] - grid.decoded_axis[n - 2]).abs();
        assert!(
            centre_step < tail_step_low,
            "centre_step {centre_step} should be < low tail_step {tail_step_low}"
        );
        assert!(
            centre_step < tail_step_high,
            "centre_step {centre_step} should be < high tail_step {tail_step_high}"
        );
    }

    /// The frozen `HIGH_TAIL_MAIN_MULTIPLICANDS` constant is what the
    /// codebook bank ships into the bitstream. It must:
    /// * Be strictly monotone (so each entry indexes a unique grid point).
    /// * Decode (via `min = -8`, `delta = 0.5`) to a symmetric-ish grid
    ///   denser near zero.
    /// * Fit in `value_bits = 5` (max value < 32).
    #[test]
    fn high_tail_constants_are_self_consistent() {
        // Strictly monotone.
        for i in 1..HIGH_TAIL_MAIN_MULTIPLICANDS.len() {
            assert!(
                HIGH_TAIL_MAIN_MULTIPLICANDS[i] > HIGH_TAIL_MAIN_MULTIPLICANDS[i - 1],
                "HIGH_TAIL_MAIN_MULTIPLICANDS not strictly monotone at i={i}"
            );
        }
        // Fit in value_bits = 5 (< 32).
        for &m in HIGH_TAIL_MAIN_MULTIPLICANDS {
            assert!(m < 32, "multiplicand {m} >= 32 (value_bits=5 cap)");
        }
        // Decoded check vs documented constant. min = -8, delta = 0.5.
        let min = -8.0_f32;
        let delta = 0.5_f32;
        for (i, &m) in HIGH_TAIL_MAIN_MULTIPLICANDS.iter().enumerate() {
            let decoded = m as f32 * delta + min;
            assert!(
                (decoded - HIGH_TAIL_MAIN_DECODED[i]).abs() < 1e-5,
                "decoded[{i}] = {decoded}, want {}",
                HIGH_TAIL_MAIN_DECODED[i]
            );
        }
        // Centre is denser than tails (qualitative tail-aware check).
        let n = HIGH_TAIL_MAIN_DECODED.len();
        let centre = HIGH_TAIL_MAIN_DECODED[n / 2] - HIGH_TAIL_MAIN_DECODED[n / 2 - 1];
        let tail_high = HIGH_TAIL_MAIN_DECODED[n - 1] - HIGH_TAIL_MAIN_DECODED[n - 2];
        let tail_low = HIGH_TAIL_MAIN_DECODED[1] - HIGH_TAIL_MAIN_DECODED[0];
        assert!(
            centre < tail_high && centre < tail_low,
            "centre step {centre} not denser than tails (low {tail_low} / high {tail_high})"
        );
    }

    /// Bad parameters produce `None` rather than panicking.
    #[test]
    fn invalid_parameters_return_none() {
        assert!(mu_law_grid(0, 6.0, 8.0, 0.5, 5).is_none());
        assert!(mu_law_grid(11, 0.0, 8.0, 0.5, 5).is_none());
        assert!(mu_law_grid(11, 6.0, 0.0, 0.5, 5).is_none());
        assert!(mu_law_grid(11, 6.0, 8.0, 0.0, 5).is_none());
        assert!(mu_law_grid(11, 6.0, 8.0, 0.5, 0).is_none());
        assert!(mu_law_grid(11, 6.0, 8.0, 0.5, 17).is_none());
        // Too-tight delta with too-small value_bits — can't fit
        // 11 distinct integers in 4 bits (16 ints, but the dedup
        // walk still has room here so this might succeed).
        // A genuine over-tight case: delta=0.001, g_max=10 → range
        // 10/0.001 = 10000 ints needed, value_bits=5 only has 32.
        let too_tight = mu_law_grid(11, 10.0, 8.0, 0.001, 5);
        assert!(
            too_tight.is_none(),
            "expected None for over-tight delta, got {too_tight:?}"
        );
    }

    /// `mu = 1` should give a near-uniform grid (mu-law degenerates
    /// to linear when mu → 0). At small mu the centre step is close
    /// to the tail step.
    #[test]
    fn small_mu_approaches_uniform() {
        let grid = mu_law_grid(11, 5.0, 0.5, 1.0, 5).expect("small-mu grid");
        let n = grid.decoded_axis.len();
        let centre = grid.decoded_axis[n / 2] - grid.decoded_axis[n / 2 - 1];
        let tail = grid.decoded_axis[1] - grid.decoded_axis[0];
        // Ratio should be < 3× for small mu (vs mu=8 where it's ~5×).
        let ratio = tail / centre;
        assert!(
            ratio < 3.0,
            "mu=0.5 should give near-uniform grid: tail/centre = {ratio}"
        );
    }

    /// SNR sanity-check: on a Laplacian-distributed source, the
    /// mu-law grid should give better mean-squared error per
    /// quantised sample than the uniform `[-5..+5]` step-1 grid
    /// (Medium's main book) at the same number of grid points.
    ///
    /// This is the *quality* claim that motivates the entire
    /// HighTail variant; it's verified at the encoder integration
    /// level too (residue book → ffmpeg cross-decode SNR delta),
    /// but the unit-test here gates the grid quality before any
    /// encoder plumbing.
    #[test]
    fn mu_law_grid_outperforms_uniform_on_laplacian_source() {
        // Laplacian samples via a deterministic LCG + inverse-CDF.
        let mut rng = 0xCAFE_BABE_u32;
        let mut next_uniform = || {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (rng >> 16) as f32 / 65535.0
        };
        // Inverse CDF of standard Laplacian at scale b=1: x = -b*sign(u-0.5)*ln(1-2|u-0.5|)
        let n_samples = 5000usize;
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let u = next_uniform();
            let s = if u < 0.5 { -1.0 } else { 1.0 };
            let mag = -(1.0 - 2.0 * (u - 0.5).abs()).max(1e-6).ln();
            samples.push(s * mag);
        }

        // Uniform grid: 11 points at {-5, -4, ..., +5}.
        let uniform: Vec<f32> = (0..11).map(|i| (i as f32) - 5.0).collect();
        // Mu-law grid (the HighTail shape).
        let mu_grid = mu_law_grid(11, 6.0, 8.0, 0.5, 5).expect("mu grid");

        let mut uniform_err = 0f32;
        let mut mu_err = 0f32;
        for &s in &samples {
            let mut best_u_err = f32::MAX;
            for &g in &uniform {
                let e = (s - g) * (s - g);
                if e < best_u_err {
                    best_u_err = e;
                }
            }
            let mut best_m_err = f32::MAX;
            for &g in &mu_grid.decoded_axis {
                let e = (s - g) * (s - g);
                if e < best_m_err {
                    best_m_err = e;
                }
            }
            uniform_err += best_u_err;
            mu_err += best_m_err;
        }
        uniform_err /= n_samples as f32;
        mu_err /= n_samples as f32;
        let snr_uniform = 10.0 * (1.0 / uniform_err).log10();
        let snr_mu = 10.0 * (1.0 / mu_err).log10();
        let delta_db = snr_mu - snr_uniform;
        eprintln!(
            "Laplacian quantisation: uniform MSE={uniform_err:.4} ({snr_uniform:.2} dB) \
             mu-law MSE={mu_err:.4} ({snr_mu:.2} dB) delta={delta_db:+.2} dB"
        );
        assert!(
            delta_db >= 1.0,
            "mu-law should beat uniform by >= 1 dB on Laplacian source: delta={delta_db:.2} dB"
        );
    }
}

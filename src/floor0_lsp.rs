//! Floor 0 LSP-coefficient derivation (Vorbis I §6.2.3, encode direction).
//!
//! The §6.2.3 floor-0 curve shape `g(ω) = 1 / sqrt(p(ω) + q(ω))` (the
//! amplitude-independent half of the curve — see
//! [`crate::floor0_envelope`]) is, up to the order-parity normalisation the
//! decoder folds into `p`/`q`, exactly `1 / |A(e^{jω})|` for an all-pole
//! LPC model `A(z) = 1 + a₁z⁻¹ + … + a_pz⁻ᵖ` whose Line-Spectral-Pair
//! frequencies are the §6.2.3 `coefficients`. So fitting the floor-0 curve
//! *shape* to a desired spectral envelope is the classic speech-coding
//! chain:
//!
//! 1. **power spectrum → autocorrelation.** A real even power spectrum's
//!    inverse DFT is the autocorrelation sequence `r[0..=order]`
//!    ([`autocorrelation_from_power`]).
//! 2. **autocorrelation → LPC.** Levinson-Durbin recursion solves the
//!    Yule-Walker normal equations for the all-pole coefficients and the
//!    residual gain ([`levinson_durbin`]).
//! 3. **LPC → LSP.** The symmetric / antisymmetric polynomials
//!    `P(z) = A(z) + z⁻⁽ᵖ⁺¹⁾A(z⁻¹)` and `Q(z) = A(z) − z⁻⁽ᵖ⁺¹⁾A(z⁻¹)`
//!    have all their roots on the unit circle, interlaced; the angles of
//!    those roots are the LSP frequencies ([`lpc_to_lsp`]). Those angles
//!    are exactly the §6.2.3 `coefficients` the curve evaluates `cos(·)`
//!    of.
//!
//! Every step is generic DSP — the autocorrelation method, the
//! Levinson-Durbin recursion and the LSP root extraction are mathematical
//! facts, not Vorbis inventions (the same chain underpins every LPC speech
//! coder). None of this is read from any reference codec; it is derived
//! from the §6.2.3 curve definition in `Vorbis_I_spec.pdf` plus standard
//! signal-processing identities.
//!
//! The whole chain is **lossy**: the order-`p` all-pole model is the
//! best-fitting `p`-pole envelope of the target, not an exact reproduction.
//! Self-consistency (envelope → LSP → §6.2.3 render reproduces the envelope
//! shape) is the round-trip the tests pin; with no floor-0 fixture in the
//! corpus (no reference encoder emits floor 0), self-consistency is the
//! available ground truth.

use std::f64::consts::PI;

/// Errors that can arise while deriving floor-0 LSP coefficients from a
/// target envelope.
#[derive(Debug, Clone, PartialEq)]
pub enum Floor0LspError {
    /// `order` was zero — no LSP poles to derive.
    ZeroOrder,
    /// The supplied power spectrum was empty (or shorter than 2 samples),
    /// so no autocorrelation lag could be formed.
    EmptySpectrum,
    /// A power-spectrum sample was negative or non-finite. A power spectrum
    /// is non-negative by definition; carries the offending bin index.
    NegativePower(usize),
    /// `r[0]` (the signal energy) was zero or non-finite — a silent target.
    /// An all-pole model of silence is undefined; the caller should emit an
    /// 'unused' (`amplitude = 0`) floor instead.
    ZeroEnergy,
    /// Levinson-Durbin hit a non-positive prediction-error variance (the
    /// autocorrelation matrix was not positive-definite — typically a
    /// degenerate / over-flat target at the requested order). Carries the
    /// recursion stage that failed.
    NotPositiveDefinite(usize),
    /// The LSP root search did not find the expected `order` interlaced
    /// roots on the unit circle (a numerically pathological LPC set).
    /// Carries the number of roots actually found.
    LspRootCountMismatch(usize),
}

impl core::fmt::Display for Floor0LspError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Floor0LspError::ZeroOrder => write!(f, "vorbis floor0 lsp: order is zero"),
            Floor0LspError::EmptySpectrum => {
                write!(
                    f,
                    "vorbis floor0 lsp: power spectrum too short (<2 samples)"
                )
            }
            Floor0LspError::NegativePower(i) => {
                write!(f, "vorbis floor0 lsp: power[{i}] is negative or non-finite")
            }
            Floor0LspError::ZeroEnergy => {
                write!(f, "vorbis floor0 lsp: r[0] energy is zero (silent target)")
            }
            Floor0LspError::NotPositiveDefinite(k) => write!(
                f,
                "vorbis floor0 lsp: Levinson-Durbin error variance non-positive at stage {k}"
            ),
            Floor0LspError::LspRootCountMismatch(found) => write!(
                f,
                "vorbis floor0 lsp: LSP root search found {found} roots (expected order)"
            ),
        }
    }
}

impl std::error::Error for Floor0LspError {}

/// Inverse-DFT a real, even power spectrum sampled at explicit angular
/// frequencies into its autocorrelation sequence `r[0..=max_lag]`.
///
/// `power[m]` is the (non-negative) power at angular frequency `angles[m] ∈
/// [0, π]`. The autocorrelation of a real signal whose power spectrum is
/// `P(ω)` is `r[k] = (1/π) ∫₀^π P(ω) cos(kω) dω`; this evaluates that
/// integral by the midpoint rule over the supplied sample grid (a
/// deterministic numerical quadrature). Each sample covers a `dω` slice; the
/// midpoint weights are computed from the angle spacing so a non-uniform
/// grid (the Bark warp) integrates correctly. Returns `max_lag + 1` lags.
#[must_use]
pub fn autocorrelation_from_angles(power: &[f64], angles: &[f64], max_lag: usize) -> Vec<f64> {
    let len = power.len();
    debug_assert!(len >= 2, "caller validates spectrum length");
    debug_assert_eq!(power.len(), angles.len());
    // Midpoint quadrature widths: each sample owns the interval midway to its
    // neighbours, clamped to [0, π]. Handles the non-uniform Bark grid.
    let mut width = vec![0.0f64; len];
    for m in 0..len {
        let lo = if m == 0 {
            0.0
        } else {
            0.5 * (angles[m - 1] + angles[m])
        };
        let hi = if m == len - 1 {
            PI
        } else {
            0.5 * (angles[m] + angles[m + 1])
        };
        width[m] = (hi - lo).max(0.0);
    }
    let mut r = vec![0.0f64; max_lag + 1];
    for (lag, slot) in r.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for m in 0..len {
            acc += power[m] * (lag as f64 * angles[m]).cos() * width[m];
        }
        // r[k] = (1/π) Σ P(ω_m) cos(kω_m) dω_m.
        *slot = acc / PI;
    }
    r
}

/// Convenience wrapper: autocorrelation from a power spectrum sampled
/// **uniformly** on `[0, π]` at `ω_m = π·m / (len−1)`. Kept for the
/// white/low-frequency unit tests; the Bark-warped encode path uses
/// [`autocorrelation_from_angles`] with the renderer's exact grid.
#[must_use]
pub fn autocorrelation_from_power(power: &[f64], max_lag: usize) -> Vec<f64> {
    let len = power.len();
    let denom = (len - 1).max(1) as f64;
    let angles: Vec<f64> = (0..len).map(|m| PI * m as f64 / denom).collect();
    autocorrelation_from_angles(power, &angles, max_lag)
}

/// Levinson-Durbin recursion: solve the order-`p` Yule-Walker equations for
/// the all-pole LPC coefficients `a[1..=p]` (with implicit `a[0] = 1`) and
/// the final prediction-error variance (the model gain).
///
/// `r` is the autocorrelation `r[0..=p]` (at least `p + 1` lags). Returns
/// `(a, err)` where `a` has length `p` (the coefficients `a₁..a_p`, sign
/// convention `A(z) = 1 + Σ aₖ z⁻ᵏ`) and `err` is the residual variance.
///
/// # Errors
///
/// [`Floor0LspError::ZeroEnergy`] if `r[0] ≤ 0`;
/// [`Floor0LspError::NotPositiveDefinite`] if a recursion stage drives the
/// error variance non-positive (the autocorrelation matrix is singular at
/// the requested order).
pub fn levinson_durbin(r: &[f64], order: usize) -> Result<(Vec<f64>, f64), Floor0LspError> {
    if order == 0 {
        return Err(Floor0LspError::ZeroOrder);
    }
    if r[0] <= 0.0 || !r[0].is_finite() {
        return Err(Floor0LspError::ZeroEnergy);
    }
    let mut a = vec![0.0f64; order + 1];
    a[0] = 1.0;
    let mut err = r[0];
    for i in 1..=order {
        // Reflection coefficient kᵢ = −(r[i] + Σ_{j=1..i-1} a[j]·r[i−j]) / err.
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / err;
        // Update a[1..i-1] symmetrically from a *snapshot* of the previous
        // iteration's coefficients (a[j] += k·a_prev[i−j]); a[i] = k. Using a
        // snapshot avoids the in-place aliasing hazard at the centre when i
        // is even.
        let prev: Vec<f64> = a[..i].to_vec();
        for j in 1..i {
            a[j] = prev[j] + k * prev[i - j];
        }
        a[i] = k;
        err *= 1.0 - k * k;
        if err <= 0.0 || !err.is_finite() {
            return Err(Floor0LspError::NotPositiveDefinite(i));
        }
    }
    // Drop the implicit a[0] = 1; return a₁..a_p.
    Ok((a[1..=order].to_vec(), err))
}

/// Convert an order-`p` LPC coefficient vector `a[1..=p]` (sign convention
/// `A(z) = 1 + Σ aₖ z⁻ᵏ`) into its `p` Line-Spectral-Pair frequencies (in
/// radians, ascending in `(0, π)`).
///
/// Forms the symmetric `P(z) = A(z) + z⁻⁽ᵖ⁺¹⁾A(z⁻¹)` and antisymmetric
/// `Q(z) = A(z) − z⁻⁽ᵖ⁺¹⁾A(z⁻¹)` polynomials, whose unit-circle roots
/// interlace and are the LSP frequencies. The roots are found by evaluating
/// the real Chebyshev-domain reductions of `P` and `Q` on a dense `x =
/// cos ω` grid and bracketing sign changes, then bisecting — a deterministic
/// root isolation (no external solver). `P` and `Q` always carry the
/// trivial roots at `ω = 0` and `ω = π` (and a fixed factor of `1 ± z⁻¹`);
/// those are divided out before the search so only the `p` non-trivial LSP
/// angles remain.
///
/// # Errors
///
/// [`Floor0LspError::LspRootCountMismatch`] if fewer than `order` roots are
/// isolated (a numerically pathological LPC set).
pub fn lpc_to_lsp(a: &[f64], order: usize) -> Result<Vec<f64>, Floor0LspError> {
    if order == 0 {
        return Err(Floor0LspError::ZeroOrder);
    }
    // Build P(z) = A(z) + z^-(p+1) A(z^-1), Q(z) = A(z) - z^-(p+1) A(z^-1).
    // a_full has length p+1 with a_full[0] = 1.
    let mut a_full = vec![1.0f64; order + 1];
    a_full[1..=order].copy_from_slice(&a[..order]);
    let p1 = order + 1;
    let mut pp = vec![0.0f64; p1 + 1]; // P, degree p+1
    let mut qq = vec![0.0f64; p1 + 1]; // Q, degree p+1
    for i in 0..=p1 {
        let ai = if i <= order { a_full[i] } else { 0.0 };
        // z^-(p+1)·A(z^-1): coefficient of z^-i is a_full[(p+1)-i], valid
        // only when 1 ≤ (p+1)-i ≤ order, i.e. 1 ≤ i ≤ p1 and i ≥ 1.
        let rev_idx = p1.checked_sub(i);
        let arev = match rev_idx {
            Some(j) if j <= order => a_full[j],
            _ => 0.0,
        };
        pp[i] = ai + arev;
        qq[i] = ai - arev;
    }
    // Remove the known trivial roots: P always has a root at ω=π (z=−1) for
    // the (1 + z^-1) factor when p is even / ω=0 etc. The standard reduction
    // divides P by (1 + z^-1) and Q by (1 - z^-1) (each removes one trivial
    // root), yielding two even-symmetric polynomials of degree p whose
    // unit-circle roots are the LSP pairs. Synthetic-divide:
    let pr = deflate_symmetric(&pp, true); // P / (1 + z^-1)
    let qr = deflate_symmetric(&qq, false); // Q / (1 - z^-1)

    // Each reduced polynomial is palindromic of degree p; its unit-circle
    // roots come in conjugate pairs, so it reduces to a degree-⌈p/2⌉
    // Chebyshev polynomial in x = cos ω. Evaluate both on a dense grid in
    // x ∈ [-1, 1] (ω ∈ [0, π]) and isolate sign changes.
    let mut roots = Vec::with_capacity(order);
    collect_palindromic_roots(&pr, &mut roots);
    collect_palindromic_roots(&qr, &mut roots);
    roots.sort_by(|x, y| x.partial_cmp(y).unwrap());
    if roots.len() < order {
        return Err(Floor0LspError::LspRootCountMismatch(roots.len()));
    }
    roots.truncate(order);
    Ok(roots)
}

/// Synthetic-divide a palindromic/anti-palindromic polynomial (coefficients
/// low-to-high) by `(1 + z⁻¹)` (`plus = true`) or `(1 − z⁻¹)`
/// (`plus = false`), returning the degree-`(n−1)` quotient. The divisor is
/// always an exact factor of `P`/`Q` (the trivial LSP root), so the
/// remainder is discarded.
fn deflate_symmetric(c: &[f64], plus: bool) -> Vec<f64> {
    // Divide polynomial c (ascending) by (1 ± z^-1) i.e. by (z ± 1) in the
    // reversed sense. Using the recurrence for division by (x + s) with
    // s = ±1 on the ascending coefficients: q[i] = c[i] - s·q[i-1].
    let s = if plus { 1.0 } else { -1.0 };
    let n = c.len();
    let mut q = vec![0.0f64; n - 1];
    let mut carry = 0.0f64;
    for i in 0..n - 1 {
        let v = c[i] - s * carry;
        q[i] = v;
        carry = v;
    }
    q
}

/// Isolate the `(0, π)` roots of a palindromic polynomial (ascending
/// coefficients, the reduced `P`/`Q`) by evaluating it as a function of
/// `ω` on a dense grid and bisecting sign changes. Appends the root angles
/// to `out`.
fn collect_palindromic_roots(coeffs: &[f64], out: &mut Vec<f64>) {
    // Evaluate g(ω) = Σ coeffs[i]·cos? — simplest robust route: evaluate the
    // polynomial at z = e^{jω}; for a real palindromic polynomial the value
    // on the unit circle is real. We sample the real part directly.
    let eval = |omega: f64| -> f64 {
        // Σ coeffs[i] · cos(i·ω − (deg/2)·ω) — equivalently the real part of
        // e^{-j(deg/2)ω} Σ coeffs[i] e^{jiω}. Centre to keep it real & even.
        let deg = coeffs.len() - 1;
        let centre = deg as f64 / 2.0;
        let mut acc = 0.0f64;
        for (i, &c) in coeffs.iter().enumerate() {
            acc += c * ((i as f64 - centre) * omega).cos();
        }
        acc
    };
    // Dense scan ω ∈ (0, π), excluding the trivial endpoints.
    const STEPS: usize = 4096;
    let mut prev_w = 1e-6;
    let mut prev_v = eval(prev_w);
    for s in 1..=STEPS {
        let w = PI * s as f64 / (STEPS as f64 + 1.0);
        let v = eval(w);
        if prev_v == 0.0 {
            out.push(prev_w);
        } else if prev_v.signum() != v.signum() && v != 0.0 {
            // Bisect [prev_w, w] for the sign change.
            let (mut lo, mut hi) = (prev_w, w);
            let (mut flo, _fhi) = (prev_v, v);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let fmid = eval(mid);
                if fmid == 0.0 {
                    lo = mid;
                    hi = mid;
                    break;
                }
                if flo.signum() != fmid.signum() {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fmid;
                }
            }
            out.push(0.5 * (lo + hi));
        }
        prev_w = w;
        prev_v = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The core identity the floor-0 encode chain relies on: the §6.2.3
    /// curve shape `1/sqrt(p+q)` (evaluated over LSP angles from
    /// [`lpc_to_lsp`]) equals `1/|A(e^{jω})|` for the originating LPC — so
    /// LPC → LSP → §6.2.3 render reproduces the all-pole envelope exactly.
    #[test]
    fn lsp_shape_equals_inverse_lpc_magnitude() {
        let order = 4;
        let r: Vec<f64> = (0..=order)
            .map(|k| (-(k as f64) * 0.1).exp() * (0.5 * k as f64).cos())
            .collect();
        let (a, _gain) = levinson_durbin(&r, order).unwrap();
        let lsp = lpc_to_lsp(&a, order).unwrap();
        // |A(e^jω)|
        let mag_a = |omega: f64| -> f64 {
            let mut re = 1.0;
            let mut im = 0.0;
            for (k, &ak) in a.iter().enumerate() {
                let kk = (k + 1) as f64;
                re += ak * (kk * omega).cos();
                im -= ak * (kk * omega).sin();
            }
            (re * re + im * im).sqrt()
        };
        // Vorbis p+q for order 4 (even).
        let pq = |omega: f64| -> f64 {
            let cos_omega = omega.cos();
            let mut p_prod = (1.0 - cos_omega) / 2.0;
            let mut q_prod = (1.0 + cos_omega) / 2.0;
            for j in 0..order / 2 {
                let c_odd = lsp[2 * j + 1].cos();
                let term_odd = c_odd - cos_omega;
                p_prod *= 4.0 * term_odd * term_odd;
                let c_even = lsp[2 * j].cos();
                let term_even = c_even - cos_omega;
                q_prod *= 4.0 * term_even * term_even;
            }
            p_prod + q_prod
        };
        for s in 1..8 {
            let w = PI * s as f64 / 8.0;
            let ga = 1.0 / mag_a(w);
            let gv = 1.0 / pq(w).sqrt();
            assert!(
                (ga / gv - 1.0).abs() < 1e-6,
                "ω={w:.3}: 1/|A|={ga} vs 1/sqrt(p+q)={gv} should match"
            );
        }
    }

    /// Levinson on a known AR(1) autocorrelation `r[k] = ρ^k` recovers the
    /// single LPC coefficient `a₁ = −ρ` and gain `(1−ρ²)`.
    #[test]
    fn levinson_recovers_ar1() {
        let rho = 0.6f64;
        let r: Vec<f64> = (0..=1i32).map(|k| rho.powi(k)).collect();
        let (a, err) = levinson_durbin(&r, 1).unwrap();
        assert!(
            (a[0] - (-rho)).abs() < 1e-9,
            "a1={} expected {}",
            a[0],
            -rho
        );
        assert!((err - (1.0 - rho * rho)).abs() < 1e-9);
    }

    /// Levinson on a positive-definite higher-order autocorrelation
    /// reproduces the autocorrelation when run forward (the normal equations
    /// hold): Σ a[j] r[|i-j|] = 0 for i = 1..p.
    #[test]
    fn levinson_satisfies_normal_equations() {
        // A smooth, positive-definite autocorrelation (decaying cosine).
        let order = 6;
        let r: Vec<f64> = (0..=order)
            .map(|k| (-(k as f64) * 0.2).exp() * (0.3 * k as f64).cos())
            .collect();
        let (a, _err) = levinson_durbin(&r, order).unwrap();
        let mut full = vec![1.0f64];
        full.extend_from_slice(&a);
        for i in 1..=order {
            let mut acc = 0.0f64;
            for (j, &aj) in full.iter().enumerate() {
                let lag = (i as isize - j as isize).unsigned_abs();
                acc += aj * r[lag];
            }
            assert!(acc.abs() < 1e-6, "normal eq i={i} residual {acc}");
        }
    }

    #[test]
    fn levinson_rejects_zero_energy() {
        let r = vec![0.0, 0.0];
        assert_eq!(levinson_durbin(&r, 1), Err(Floor0LspError::ZeroEnergy));
    }

    /// LSP frequencies of a stable LPC model are strictly ascending in
    /// (0, π) and interlaced (a basic stability property of valid LSPs).
    #[test]
    fn lsp_frequencies_are_ascending_in_band() {
        let order = 8;
        let r: Vec<f64> = (0..=order)
            .map(|k| (-(k as f64) * 0.15).exp() * (0.4 * k as f64).cos())
            .collect();
        let (a, _) = levinson_durbin(&r, order).unwrap();
        let lsp = lpc_to_lsp(&a, order).unwrap();
        assert_eq!(lsp.len(), order);
        for w in &lsp {
            assert!(*w > 0.0 && *w < PI, "lsp {w} out of (0,π)");
        }
        for pair in lsp.windows(2) {
            assert!(pair[1] > pair[0], "lsp not ascending: {:?}", lsp);
        }
    }

    /// autocorrelation_from_power on a flat (white) spectrum yields r[0]>0
    /// and near-zero higher lags (white noise is uncorrelated).
    #[test]
    fn flat_spectrum_autocorrelation_is_impulse() {
        let power = vec![1.0f64; 256];
        let r = autocorrelation_from_power(&power, 8);
        assert!(r[0] > 0.0);
        for &rk in &r[1..] {
            assert!(rk.abs() < 1e-2, "flat-spectrum lag should be ~0, got {rk}");
        }
    }

    /// A spectrum with a single low-frequency emphasis yields a positive
    /// first-lag correlation (low-frequency energy ⇒ positive r[1]).
    #[test]
    fn low_freq_emphasis_gives_positive_first_lag() {
        let n = 256;
        let power: Vec<f64> = (0..n)
            .map(|m| {
                let w = PI * m as f64 / (n as f64 - 1.0);
                // Strong DC, weak Nyquist.
                1.0 + 4.0 * (w * 0.5).cos().powi(2)
            })
            .collect();
        let r = autocorrelation_from_power(&power, 4);
        assert!(r[0] > 0.0);
        assert!(r[1] > 0.0, "low-freq emphasis ⇒ r[1] > 0, got {}", r[1]);
    }
}

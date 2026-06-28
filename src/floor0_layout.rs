//! Floor 0 setup-header **order** design (Vorbis I §6.2.1 / §6.2.3, encode
//! direction).
//!
//! The crate's floor-0 *per-packet* encode chain is closed *given a header*:
//! [`crate::floor0_envelope::plan_floor0_lsp`] fits the LSP shape,
//! [`crate::floor0_envelope::fit_floor0_amplitude`] solves the per-packet
//! gain, and [`crate::floor0_envelope::plan_floor0_packet`] composes them
//! against a [`crate::setup::Floor0Header`] the caller supplies. What that
//! chain does **not** decide is the header's `floor0_order` — the number of
//! LSP poles the §6.2.3 curve carries. That is a pure encoder analysis
//! choice (like the floor-1 post count, [`crate::floor1_layout`]): a higher
//! order tracks finer spectral detail at the cost of more coefficient bits;
//! a lower order is cheaper but smooths the envelope.
//!
//! This module is the floor-0 analogue of the floor-1 post-budget choice:
//! the **order selector**. It sweeps a caller-bounded order range, fits each
//! candidate order's LSP shape + amplitude, renders the §6.2.3 curve the
//! decoder would draw ([`crate::floor0::Floor0Decoder::render_curve`]), and
//! scores its **log-domain** fidelity against the desired envelope (the
//! §6.2.3 curve is exponential, so the natural error metric is in `ln`
//! space). Two policies are offered:
//!
//! * [`select_floor0_order`] — the **smallest** order whose log-domain SNR
//!   meets a target (the cheapest model that is "good enough");
//! * [`select_floor0_order_rd`] — the order minimising a rate-distortion
//!   objective `distortion + lambda · order` (the order is a direct,
//!   monotone proxy for the coefficient bit cost, since each pole costs one
//!   value-book codeword).
//!
//! Because no reference encoder emits floor 0 (real-world Vorbis uses floor
//! 1 exclusively), the fidelity is measured against the crate's **own**
//! decoder render — the same self-consistency ground truth the rest of the
//! floor-0 encode path uses.
//!
//! ## Scope
//!
//! This module decides `floor0_order` only. The `bark_map_size`, `rate`,
//! `amplitude_bits`, and `amplitude_offset` are taken from the caller's
//! [`crate::floor0_envelope::Floor0ShapeParams`] template (the order field
//! is overwritten per candidate); a [`suggest_floor0_params`] helper offers
//! spec-grounded defaults for them. The value-codebook *contents* (the
//! rate-distortion bit-allocation design) remain the open followup, exactly
//! as for floor 1.

use crate::codebook::{VorbisCodebook, VqLookup};
use crate::floor0::Floor0Decoder;
use crate::floor0_envelope::{fit_floor0_amplitude, plan_floor0_lsp, Floor0ShapeParams};
use crate::setup::Floor0Header;

/// Errors that can arise while selecting a floor-0 order (Vorbis I §6.2.1,
/// encode direction).
#[derive(Debug, Clone, PartialEq)]
pub enum Floor0LayoutError {
    /// The supplied envelope was empty. The fidelity score samples one
    /// curve value per bin; with no bins there is nothing to fit.
    EmptyEnvelope,
    /// The order search range was empty or inverted (`min_order > max_order`,
    /// or `max_order == 0`). Carries the bounds.
    EmptyOrderRange {
        /// The requested minimum order.
        min_order: usize,
        /// The requested maximum order.
        max_order: usize,
    },
    /// `max_order` exceeded the §6.2.1 8-bit `floor0_order` ceiling (255).
    /// Carries the offending value.
    OrderTooLarge(usize),
    /// Every candidate order's fit failed (e.g. degenerate envelope the
    /// Levinson-Durbin recursion could not model). No order could be
    /// scored, so none can be selected.
    NoViableOrder,
}

impl core::fmt::Display for Floor0LayoutError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Floor0LayoutError::EmptyEnvelope => {
                write!(f, "vorbis floor0 layout: envelope is empty (§6.2.3)")
            }
            Floor0LayoutError::EmptyOrderRange {
                min_order,
                max_order,
            } => write!(
                f,
                "vorbis floor0 layout: order range [{min_order}, {max_order}] is empty (§6.2.1)"
            ),
            Floor0LayoutError::OrderTooLarge(o) => write!(
                f,
                "vorbis floor0 layout: order {o} exceeds the §6.2.1 8-bit ceiling (255)"
            ),
            Floor0LayoutError::NoViableOrder => write!(
                f,
                "vorbis floor0 layout: no candidate order could be fitted (§6.2.3)"
            ),
        }
    }
}

impl std::error::Error for Floor0LayoutError {}

/// Spec-grounded default `floor0_rate`, `floor0_bark_map_size`,
/// `floor0_amplitude_bits`, and `floor0_amplitude_offset` for an
/// `n`-bin floor at sample rate `rate`, returned as a
/// [`Floor0ShapeParams`] with `order` left at `0` (the caller / selector
/// fills it).
///
/// The defaults track §6.2.1 / §6.2.3 conventions: the Bark map is sized to
/// the floor bin count (one Bark bucket per bin is the finest useful
/// resolution; the §6.2.3 render clamps the map into `bark_map_size`), the
/// amplitude field is the full 6-bit width §6.2.1 stores, and the amplitude
/// offset is a mid-scale non-zero gain (the §6.2.3 exponent vanishes at a
/// zero offset, so it must be `>= 1`).
#[must_use]
pub fn suggest_floor0_params(n: usize, rate: u32) -> Floor0ShapeParams {
    Floor0ShapeParams {
        order: 0,
        rate,
        // One Bark bucket per spectral bin (clamped to at least 1).
        bark_map_size: (n as u32).max(1),
        // §6.2.1 stores amplitude_bits as a 6-bit field; the full width
        // gives the finest per-packet gain quantisation.
        amplitude_bits: 6,
        // A non-zero mid-scale gain offset (the §6.2.3 exponent needs it
        // `>= 1`); 32 sits mid-range for a 6-bit amplitude field.
        amplitude_offset: 32,
    }
}

/// 10·log10 of the energy ratio between a target curve and the per-bin
/// **log-domain** error, the natural fidelity metric for the §6.2.3
/// exponential curve. Returns `+inf` for an exact log-domain match.
fn log_snr_db(envelope: &[f32], rendered: &[f32]) -> f64 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&e, &r) in envelope.iter().zip(rendered.iter()) {
        // Guard the log against zero/negative samples by flooring to a tiny
        // positive epsilon (the §10.1-style amplitude floor); the curve and
        // envelope are non-negative amplitudes.
        let le = (e.max(1e-30) as f64).ln();
        let lr = (r.max(1e-30) as f64).ln();
        sig += le * le;
        let d = le - lr;
        err += d * d;
    }
    if err == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

/// Fit a single candidate `order` against the envelope and return the
/// rendered §6.2.3 curve, or `None` if the fit failed (a degenerate
/// envelope, an order the recursion cannot model).
fn fit_and_render(envelope: &[f32], params: &Floor0ShapeParams, order: usize) -> Option<Vec<f32>> {
    if order == 0 {
        return None;
    }
    let mut p = *params;
    p.order = order;
    let lsp = plan_floor0_lsp(envelope, &p).ok()?;
    let amplitude = fit_floor0_amplitude(&lsp, envelope, &p).ok()?;
    // Build a header + a placeholder value book so the public decoder
    // render can be used. `render_curve` reads coefficients directly and
    // never consults the value books — but `Floor0Decoder::new` validates
    // the header and requires the book list to resolve to a VQ-lookup
    // codebook, so supply a minimal single-entry tessellation book at index
    // 0. It is structurally valid and never touched by the render.
    let placeholder_book = VorbisCodebook {
        dimensions: 1,
        entries: 1,
        codeword_lengths: vec![1],
        lookup: VqLookup::Tessellation {
            minimum_value: 0.0,
            delta_value: 1.0,
            value_bits: 1,
            sequence_p: false,
            multiplicands: vec![0],
        },
    };
    let header = Floor0Header {
        order: order as u8,
        rate: p.rate as u16,
        bark_map_size: p.bark_map_size as u16,
        amplitude_bits: p.amplitude_bits,
        amplitude_offset: p.amplitude_offset,
        book_list: vec![0],
    };
    let decoder = Floor0Decoder::new(&header, std::slice::from_ref(&placeholder_book)).ok()?;
    let rendered = decoder.render_curve(amplitude, &lsp, envelope.len());
    Some(rendered)
}

/// The scored result of a floor-0 order fit: the candidate order, its
/// log-domain SNR (dB) against the envelope, and the §6.2.1 header field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Floor0OrderFit {
    /// The candidate `floor0_order`.
    pub order: usize,
    /// The log-domain SNR (dB) of the rendered curve against the envelope.
    pub log_snr_db: f64,
}

/// Score every order in `[min_order, max_order]` against the envelope.
///
/// Returns the per-order [`Floor0OrderFit`] list (orders that failed to fit
/// are omitted). The shared shape parameters (`rate`, `bark_map_size`,
/// `amplitude_bits`, `amplitude_offset`) come from `params`; its `order`
/// field is ignored (overwritten per candidate).
///
/// # Errors
///
/// [`Floor0LayoutError::EmptyEnvelope`], [`Floor0LayoutError::EmptyOrderRange`],
/// or [`Floor0LayoutError::OrderTooLarge`].
pub fn score_floor0_orders(
    envelope: &[f32],
    params: &Floor0ShapeParams,
    min_order: usize,
    max_order: usize,
) -> Result<Vec<Floor0OrderFit>, Floor0LayoutError> {
    if envelope.is_empty() {
        return Err(Floor0LayoutError::EmptyEnvelope);
    }
    if max_order == 0 || min_order > max_order {
        return Err(Floor0LayoutError::EmptyOrderRange {
            min_order,
            max_order,
        });
    }
    if max_order > 255 {
        return Err(Floor0LayoutError::OrderTooLarge(max_order));
    }
    let lo = min_order.max(1);
    let mut fits = Vec::new();
    for order in lo..=max_order {
        if let Some(rendered) = fit_and_render(envelope, params, order) {
            fits.push(Floor0OrderFit {
                order,
                log_snr_db: log_snr_db(envelope, &rendered),
            });
        }
    }
    Ok(fits)
}

/// Select the **smallest** floor-0 order whose log-domain SNR meets
/// `target_snr_db` (Vorbis I §6.2.1, encode direction).
///
/// Sweeps `[min_order, max_order]`, fits each candidate, and returns the
/// first (cheapest) order clearing the target. If none clears it, returns
/// the best-scoring order in range (the closest achievable fit) rather than
/// failing — the caller asked for a model, and the densest available one is
/// the most faithful answer.
///
/// # Errors
///
/// As [`score_floor0_orders`], plus [`Floor0LayoutError::NoViableOrder`] if
/// every candidate fit failed.
pub fn select_floor0_order(
    envelope: &[f32],
    params: &Floor0ShapeParams,
    min_order: usize,
    max_order: usize,
    target_snr_db: f64,
) -> Result<Floor0OrderFit, Floor0LayoutError> {
    let fits = score_floor0_orders(envelope, params, min_order, max_order)?;
    if fits.is_empty() {
        return Err(Floor0LayoutError::NoViableOrder);
    }
    // The first order clearing the target (orders are ascending).
    if let Some(&hit) = fits.iter().find(|f| f.log_snr_db >= target_snr_db) {
        return Ok(hit);
    }
    // None cleared it: return the best-scoring (most faithful) order.
    let best = fits
        .iter()
        .copied()
        .max_by(|a, b| {
            a.log_snr_db
                .partial_cmp(&b.log_snr_db)
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .expect("non-empty fits");
    Ok(best)
}

/// Select the floor-0 order minimising a rate-distortion objective
/// `distortion + lambda · order` (Vorbis I §6.2.1, encode direction).
///
/// `distortion` is the log-domain *error* energy (the reciprocal of the SNR
/// numerator/denominator split): lower is better. The order is a direct,
/// monotone proxy for the coefficient bit cost (one value-book codeword per
/// pole), so `lambda · order` is the rate term. `lambda == 0` reduces to
/// "pick the lowest-distortion order"; larger `lambda` trades fidelity for
/// fewer poles. Ties break toward the **lower** order (cheaper).
///
/// # Errors
///
/// As [`score_floor0_orders`], plus [`Floor0LayoutError::NoViableOrder`].
pub fn select_floor0_order_rd(
    envelope: &[f32],
    params: &Floor0ShapeParams,
    min_order: usize,
    max_order: usize,
    lambda: f64,
) -> Result<Floor0OrderFit, Floor0LayoutError> {
    if envelope.is_empty() {
        return Err(Floor0LayoutError::EmptyEnvelope);
    }
    if max_order == 0 || min_order > max_order {
        return Err(Floor0LayoutError::EmptyOrderRange {
            min_order,
            max_order,
        });
    }
    if max_order > 255 {
        return Err(Floor0LayoutError::OrderTooLarge(max_order));
    }
    let lo = min_order.max(1);
    let mut best: Option<(f64, Floor0OrderFit)> = None;
    for order in lo..=max_order {
        let Some(rendered) = fit_and_render(envelope, params, order) else {
            continue;
        };
        // Distortion = log-domain error energy. Recompute it directly so the
        // RD objective uses an energy, not a dB ratio.
        let mut err = 0.0f64;
        for (&e, &r) in envelope.iter().zip(rendered.iter()) {
            let d = (e.max(1e-30) as f64).ln() - (r.max(1e-30) as f64).ln();
            err += d * d;
        }
        let cost = err + lambda * order as f64;
        let fit = Floor0OrderFit {
            order,
            log_snr_db: log_snr_db(envelope, &rendered),
        };
        match &best {
            // Strictly-less keeps the earlier (lower) order on a tie.
            Some((bc, _)) if cost >= *bc => {}
            _ => best = Some((cost, fit)),
        }
    }
    best.map(|(_, f)| f).ok_or(Floor0LayoutError::NoViableOrder)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic linear-domain envelope with a few spectral resonances:
    /// the §6.2.3 LSP curve is an all-pole model, so a spectrum with
    /// distinct peaks rewards a higher order (more poles to place on the
    /// peaks). `n` bins over a smooth base with `peaks` cosine bumps.
    fn resonant_envelope(n: usize, peaks: usize) -> Vec<f32> {
        (0..n)
            .map(|k| {
                let f = k as f32 / n as f32;
                let mut v = 0.05f32;
                for p in 0..peaks {
                    let centre = (p as f32 + 0.5) / peaks as f32;
                    let d = (f - centre) / 0.03;
                    v += 0.6 * (-d * d).exp();
                }
                v
            })
            .collect()
    }

    #[test]
    fn suggest_params_are_spec_valid() {
        let p = suggest_floor0_params(128, 44_100);
        assert_eq!(p.bark_map_size, 128);
        assert_eq!(p.amplitude_bits, 6);
        assert!(p.amplitude_offset >= 1, "offset must be non-zero (§6.2.3)");
        assert_eq!(p.rate, 44_100);
        // bark_map_size floors at 1 for a degenerate n.
        assert_eq!(suggest_floor0_params(0, 8_000).bark_map_size, 1);
    }

    #[test]
    fn empty_envelope_rejected() {
        let p = suggest_floor0_params(0, 44_100);
        assert_eq!(
            score_floor0_orders(&[], &p, 1, 8),
            Err(Floor0LayoutError::EmptyEnvelope)
        );
    }

    #[test]
    fn empty_order_range_rejected() {
        let env = resonant_envelope(64, 2);
        let p = suggest_floor0_params(64, 44_100);
        assert_eq!(
            score_floor0_orders(&env, &p, 5, 3),
            Err(Floor0LayoutError::EmptyOrderRange {
                min_order: 5,
                max_order: 3
            })
        );
        assert_eq!(
            score_floor0_orders(&env, &p, 0, 0),
            Err(Floor0LayoutError::EmptyOrderRange {
                min_order: 0,
                max_order: 0
            })
        );
    }

    #[test]
    fn order_too_large_rejected() {
        let env = resonant_envelope(64, 2);
        let p = suggest_floor0_params(64, 44_100);
        assert_eq!(
            score_floor0_orders(&env, &p, 1, 256),
            Err(Floor0LayoutError::OrderTooLarge(256))
        );
    }

    #[test]
    fn higher_order_fits_a_resonant_envelope_no_worse() {
        // An all-pole model improves (or holds) as poles are added on a
        // multi-resonance spectrum. The best SNR over a wider order range
        // must be at least the best over a narrower low-order range.
        let env = resonant_envelope(128, 4);
        let p = suggest_floor0_params(128, 44_100);
        let low = score_floor0_orders(&env, &p, 1, 4).unwrap();
        let high = score_floor0_orders(&env, &p, 1, 16).unwrap();
        assert!(!low.is_empty() && !high.is_empty());
        let best_low = low
            .iter()
            .map(|f| f.log_snr_db)
            .fold(f64::NEG_INFINITY, f64::max);
        let best_high = high
            .iter()
            .map(|f| f.log_snr_db)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            best_high >= best_low - 1.0,
            "wider order range ({best_high} dB) must not fit worse than narrower ({best_low} dB)"
        );
    }

    #[test]
    fn select_smallest_order_meeting_target_is_cheapest() {
        let env = resonant_envelope(128, 3);
        let p = suggest_floor0_params(128, 44_100);
        // A very low target every viable order clears → the smallest order
        // wins (cheapest).
        let fit = select_floor0_order(&env, &p, 2, 20, f64::NEG_INFINITY).unwrap();
        assert_eq!(fit.order, 2, "lowest order clears a trivial target");

        // A target nothing clears → the best-scoring (most faithful) order
        // is returned rather than an error.
        let fit_hi = select_floor0_order(&env, &p, 2, 20, f64::INFINITY).unwrap();
        let all = score_floor0_orders(&env, &p, 2, 20).unwrap();
        let best = all
            .iter()
            .copied()
            .max_by(|a, b| a.log_snr_db.partial_cmp(&b.log_snr_db).unwrap())
            .unwrap();
        assert_eq!(fit_hi.order, best.order);
    }

    #[test]
    fn rd_lambda_zero_picks_lowest_distortion() {
        let env = resonant_envelope(128, 4);
        let p = suggest_floor0_params(128, 44_100);
        // lambda = 0 → pure distortion minimiser: the chosen order's SNR is
        // the maximum over the range (lowest log-error energy).
        let fit = select_floor0_order_rd(&env, &p, 1, 16, 0.0).unwrap();
        let all = score_floor0_orders(&env, &p, 1, 16).unwrap();
        let best_snr = all
            .iter()
            .map(|f| f.log_snr_db)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((fit.log_snr_db - best_snr).abs() < 1e-6);
    }

    #[test]
    fn rd_large_lambda_prefers_lower_order() {
        let env = resonant_envelope(128, 4);
        let p = suggest_floor0_params(128, 44_100);
        let cheap = select_floor0_order_rd(&env, &p, 1, 16, 0.0).unwrap();
        // A large lambda penalises poles heavily → the chosen order must be
        // no larger than the distortion-only choice (usually strictly less).
        let dear = select_floor0_order_rd(&env, &p, 1, 16, 1e6).unwrap();
        assert!(
            dear.order <= cheap.order,
            "large lambda ({}) must not pick a higher order than lambda 0 ({})",
            dear.order,
            cheap.order
        );
        // At a punishing lambda the rate term dominates → the minimum order.
        assert_eq!(dear.order, 1);
    }
}

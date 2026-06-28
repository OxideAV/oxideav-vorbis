//! Floor 1 setup-header **layout** design (Vorbis I §7.2.2 / §7.2.4,
//! encode direction).
//!
//! The crate's floor-1 encode chain is closed *given a header*: from a
//! desired linear-domain envelope, [`crate::floor1_envelope::plan_floor1_envelope`]
//! fits the integer posts, [`crate::floor1_encode::plan_floor1_y`] unwraps
//! them into the packet-domain `[floor1_Y]`, and
//! [`crate::floor1_encode::plan_floor1_partition_cvals`] packs the
//! per-partition selectors — all against a [`crate::setup::Floor1Header`]
//! the caller supplies. What that chain does **not** decide is the header
//! itself: where the floor's posts sit on the spectral axis, how many there
//! are, and how the partitions/classes group them. Those are pure encoder
//! analysis choices — the spec fixes only the *bitstream mechanics* of the
//! header (§7.2.2) and how the decoder renders a curve from it (§7.2.4),
//! leaving the placement to the encoder exactly as it leaves the
//! block-size and coupling decisions (see [`crate::blocksize`],
//! [`crate::synthesis::should_couple`]).
//!
//! This module is the first piece of that design: the **x-list
//! (post-placement) planner**. The floor renders as straight integer line
//! segments between posts in the dB-ladder domain (§7.2.4 step 2, see
//! [`crate::floor1::render_line`]), so an x-list is good exactly when the
//! desired envelope — mapped to that ladder domain — is well approximated
//! by the piecewise-linear interpolation through the chosen posts. The
//! planner derives the post coordinates by **adaptive refinement**: it
//! starts from the two implicit endpoints (`0` and the floor length) and
//! repeatedly inserts the bin whose ladder-domain value is furthest from
//! the current piecewise-linear reconstruction, until a post budget is met
//! or the worst-case ladder error falls below a tolerance.
//!
//! ## Why the ladder domain
//!
//! The §7.2.4 step-2 render draws each segment as an integer line in
//! `final_Y * multiplier` units (a `0..256` dB-ladder index) and then
//! substitutes each bin through the strictly-increasing §10.1
//! `floor1_inverse_dB_table`. The dB-table substitution is a fixed
//! monotone warp shared by *every* bin, so minimising the linear-domain
//! envelope error and minimising the **ladder-index** error pick the same
//! post coordinates up to that warp. The planner therefore measures error
//! in ladder indices (via [`crate::floor1_envelope::invert_inverse_db`]),
//! the domain the line is actually drawn in — the same metric the
//! [`crate::floor1_envelope`] fit clamps into.
//!
//! ## Scope
//!
//! This module decides the **explicit** x-coordinates only — the
//! `x_list` field of a [`crate::setup::Floor1Header`] (which excludes the
//! two implicit endpoints, per §7.2.2). It does not choose the partition /
//! class grouping, the codebook contents, or the multiplier; those remain
//! the caller's (the partition grouping has its own
//! [`plan_floor1_partition_layout`] in this module). Feeding the planned
//! x-list into a header and through the existing envelope → packet chain is
//! the integration the [`crate::floor1_layout`] tests exercise.

use crate::floor1_envelope::invert_inverse_db;

/// Errors that can arise while planning a floor-1 x-list or partition
/// layout (Vorbis I §7.2.2, encode direction).
#[derive(Debug, Clone, PartialEq)]
pub enum Floor1LayoutError {
    /// The supplied envelope was empty. Post placement samples one ladder
    /// index per bin; with no bins there is nothing to place.
    EmptyEnvelope,
    /// The requested explicit-post count was `0`. The two implicit
    /// endpoints are always present, but a floor with no interior post is
    /// degenerate (a single flat segment); the spec permits it, yet the
    /// planner refuses it as a likely caller error — request at least one
    /// interior post or use the envelope fit with an endpoint-only header
    /// directly.
    ZeroPosts,
    /// The requested explicit-post count exceeded what the floor length can
    /// hold. There are only `floor_length - 1` distinct interior bin
    /// coordinates in `1..floor_length`; asking for more unique posts than
    /// that cannot be satisfied. Carries the request and the ceiling.
    TooManyPosts {
        /// The requested explicit-post count.
        requested: usize,
        /// The maximum distinct interior coordinates available.
        available: usize,
    },
    /// The envelope carried a non-finite or negative sample. The §10.1
    /// ladder is a non-negative amplitude scale; such a target has no
    /// nearest ladder index. Carries the offending bin and value.
    NonFiniteEnvelope {
        /// The bin index whose sample was rejected.
        bin: usize,
        /// The offending sample value.
        value: f32,
    },
    /// `rangebits` was too small to address the floor length: the implicit
    /// upper endpoint `2^rangebits` must be `>= floor_length` so every bin
    /// is inside the rendered span. Carries the supplied `rangebits` and
    /// the floor length it failed to cover.
    RangebitsTooSmall {
        /// The supplied `rangebits`.
        rangebits: u8,
        /// The floor length (number of envelope bins) it must cover.
        floor_length: usize,
    },
    /// No partition class in the supplied catalogue has a dimension that
    /// can tile the chosen explicit-post count. Carries the post count that
    /// could not be tiled.
    NoTilingClass {
        /// The explicit-post count the layout could not partition.
        posts: usize,
    },
}

impl core::fmt::Display for Floor1LayoutError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Floor1LayoutError::EmptyEnvelope => {
                write!(f, "vorbis floor1 layout: envelope is empty (§7.2.2)")
            }
            Floor1LayoutError::ZeroPosts => write!(
                f,
                "vorbis floor1 layout: requested 0 interior posts (§7.2.2 needs >= 1)"
            ),
            Floor1LayoutError::TooManyPosts {
                requested,
                available,
            } => write!(
                f,
                "vorbis floor1 layout: requested {requested} posts but only {available} interior coordinates exist"
            ),
            Floor1LayoutError::NonFiniteEnvelope { bin, value } => write!(
                f,
                "vorbis floor1 layout: bin {bin} sample {value} is not a finite non-negative amplitude (§10.1)"
            ),
            Floor1LayoutError::RangebitsTooSmall {
                rangebits,
                floor_length,
            } => write!(
                f,
                "vorbis floor1 layout: rangebits {rangebits} (2^{rangebits}) cannot cover floor length {floor_length} (§7.2.2)"
            ),
            Floor1LayoutError::NoTilingClass { posts } => write!(
                f,
                "vorbis floor1 layout: no class dimension tiles {posts} posts (§7.2.2)"
            ),
        }
    }
}

impl std::error::Error for Floor1LayoutError {}

/// Map a linear-domain envelope onto the §7.2.4 dB-ladder index domain.
///
/// Each bin's linear amplitude is inverted through the §10.1
/// `floor1_inverse_dB_table` ([`invert_inverse_db`]) to the nearest ladder
/// index `0..=255` — the domain the §7.2.4 step-2 line render actually
/// operates in. Returns the per-bin ladder indices as `f32` (so the
/// interpolation error metric is fractional, not quantised to integers).
///
/// # Errors
///
/// [`Floor1LayoutError::EmptyEnvelope`] for an empty slice, or
/// [`Floor1LayoutError::NonFiniteEnvelope`] for the first non-finite or
/// negative sample.
fn envelope_to_ladder(envelope: &[f32]) -> Result<Vec<f32>, Floor1LayoutError> {
    if envelope.is_empty() {
        return Err(Floor1LayoutError::EmptyEnvelope);
    }
    let mut ladder = Vec::with_capacity(envelope.len());
    for (bin, &v) in envelope.iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            return Err(Floor1LayoutError::NonFiniteEnvelope { bin, value: v });
        }
        ladder.push(f32::from(invert_inverse_db(v)));
    }
    Ok(ladder)
}

/// The piecewise-linear value at bin `x` interpolated between two posts
/// `(x0, y0)` and `(x1, y1)` in the ladder domain.
///
/// This mirrors the geometry of [`crate::floor1::render_point`] but in
/// fractional ladder units (the planner's approximation metric is
/// continuous; the decoder's integer line is its quantisation). `x0 < x1`
/// is guaranteed by the caller (posts are kept sorted and unique).
#[inline]
fn ladder_interp(x0: usize, y0: f32, x1: usize, y1: f32, x: usize) -> f32 {
    debug_assert!(x0 < x1);
    let t = (x - x0) as f32 / (x1 - x0) as f32;
    y0 + t * (y1 - y0)
}

/// Plan a floor-1 **x-list** by adaptive post placement against a desired
/// linear-domain envelope (Vorbis I §7.2.2 / §7.2.4, encode direction).
///
/// Returns the `max_posts` explicit interior x-coordinates (sorted
/// ascending, excluding the two implicit endpoints `0` and the floor
/// length) that best approximate the envelope under the floor's
/// piecewise-linear ladder-domain render. The placement is greedy: it
/// begins with the endpoint pair and, on each step, inserts the interior
/// bin whose ladder value is furthest (largest absolute error) from the
/// current piecewise-linear reconstruction through the already-chosen
/// posts. It stops early once that worst-case ladder error falls to or
/// below `tolerance` (set `tolerance` to `0.0` to always fill the budget).
///
/// The result is suitable as the `x_list` of a [`crate::setup::Floor1Header`]
/// (with `rangebits` chosen so `2^rangebits >= envelope.len()`).
///
/// # Parameters
///
/// * `envelope` — one desired linear-domain magnitude per spectral bin
///   (the floor length; the forward-MDCT magnitude domain).
/// * `max_posts` — the explicit-post budget (interior posts only,
///   `1..=floor_length-1`). The returned list has **at most** this many
///   coordinates (fewer if `tolerance` is met first).
/// * `tolerance` — the worst-case ladder-index error at which placement
///   stops early. `0.0` always fills the budget.
///
/// # Errors
///
/// [`Floor1LayoutError::EmptyEnvelope`], [`Floor1LayoutError::ZeroPosts`]
/// (`max_posts == 0`), [`Floor1LayoutError::TooManyPosts`] (more posts than
/// interior coordinates), or [`Floor1LayoutError::NonFiniteEnvelope`].
pub fn plan_floor1_x_list(
    envelope: &[f32],
    max_posts: usize,
    tolerance: f32,
) -> Result<Vec<u32>, Floor1LayoutError> {
    let ladder = envelope_to_ladder(envelope)?;
    let n = ladder.len();
    // Interior coordinates live in `1..n` (bin 0 and bin `n` — the floor
    // length — are the implicit endpoints). The render samples bins
    // `0..n`, so the last interior coordinate the planner may pick is
    // `n - 1`; that leaves `n - 1` candidate interior positions.
    let available = n.saturating_sub(1);
    if max_posts == 0 {
        return Err(Floor1LayoutError::ZeroPosts);
    }
    if max_posts > available {
        return Err(Floor1LayoutError::TooManyPosts {
            requested: max_posts,
            available,
        });
    }

    // The endpoint posts the decoder injects: bin 0 and bin `n` (the floor
    // length). The endpoint *ladder* values are the envelope at bin 0 and
    // at the last bin (the decoder renders the tail flat past the last
    // bin, so the right endpoint carries the last bin's ladder value).
    let left = (0usize, ladder[0]);
    let right = (n, ladder[n - 1]);

    // The sorted post set the planner grows, kept as (x, ladder_y) pairs.
    let mut posts: Vec<(usize, f32)> = vec![left, right];

    for _ in 0..max_posts {
        // Find the interior bin with the largest absolute ladder error
        // against the current piecewise-linear reconstruction.
        let mut worst_bin = 0usize;
        let mut worst_err = -1.0f32;
        // Walk each segment [posts[s], posts[s+1]] and test its interior
        // bins. Posts are sorted ascending and unique, so the segments
        // partition `0..=n`.
        for s in 0..posts.len() - 1 {
            let (x0, y0) = posts[s];
            let (x1, y1) = posts[s + 1];
            // Candidate bins strictly inside this segment that are also
            // valid interior coordinates (`< n`, since bin `n` is the
            // right endpoint, never a candidate).
            let lo = x0 + 1;
            let hi = x1.min(n); // exclusive upper; bins `lo..hi` (bin `n`
                                // is the right endpoint, never a candidate)
            for (offset, &ladder_y) in ladder[lo..hi].iter().enumerate() {
                let x = lo + offset;
                let approx = ladder_interp(x0, y0, x1, y1, x);
                let err = (ladder_y - approx).abs();
                if err > worst_err {
                    worst_err = err;
                    worst_bin = x;
                }
            }
        }

        // No candidate left (every interior bin is already a post), or the
        // remaining error is within tolerance: stop.
        if worst_err < 0.0 || worst_err <= tolerance {
            break;
        }

        // Insert the worst bin, keeping the post list sorted.
        let insert_at = posts.partition_point(|&(x, _)| x < worst_bin);
        posts.insert(insert_at, (worst_bin, ladder[worst_bin]));
    }

    // Strip the two endpoints; return the interior coordinates only.
    let x_list: Vec<u32> = posts
        .iter()
        .filter(|&&(x, _)| x != 0 && x != n)
        .map(|&(x, _)| x as u32)
        .collect();
    Ok(x_list)
}

/// The smallest `rangebits` (4-bit field, `0..=15`) whose implicit upper
/// endpoint `2^rangebits` covers a floor of `floor_length` bins, i.e. the
/// least `r` with `2^r >= floor_length`.
///
/// Returns [`Floor1LayoutError::RangebitsTooSmall`] only if no `r <= 15`
/// suffices (a floor longer than `32768` bins, which no Vorbis block
/// reaches).
pub fn min_rangebits(floor_length: usize) -> Result<u8, Floor1LayoutError> {
    for r in 0u8..=15 {
        if (1usize << r) >= floor_length {
            return Ok(r);
        }
    }
    Err(Floor1LayoutError::RangebitsTooSmall {
        rangebits: 15,
        floor_length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::floor1::Floor1Decoder;
    use crate::floor1_encode::{full_x_list, plan_floor1_y};
    use crate::floor1_envelope::plan_floor1_envelope;
    use crate::setup::{Floor1Class, Floor1Header};

    /// Build a single-class floor-1 header from a planned explicit x-list.
    /// `rangebits` is the smallest that covers the floor length; the single
    /// class has `dimensions = 1` so each explicit post is its own
    /// dimension (the decode-fidelity path under test does not depend on
    /// the class grouping). The sub-book slot is a placeholder — this
    /// helper drives the *envelope → posts* fidelity, not Huffman packing.
    fn header_from_x_list(x_list: Vec<u32>, multiplier: u8, floor_length: usize) -> Floor1Header {
        let rangebits = min_rangebits(floor_length).unwrap();
        let partitions = x_list.len() as u8;
        Floor1Header {
            partitions,
            partition_class_list: vec![0; x_list.len()],
            classes: vec![Floor1Class {
                dimensions: 1,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![None],
            }],
            multiplier,
            rangebits,
            x_list,
        }
    }

    // ---------- envelope_to_ladder ----------

    #[test]
    fn empty_envelope_rejected() {
        assert_eq!(
            plan_floor1_x_list(&[], 4, 0.0),
            Err(Floor1LayoutError::EmptyEnvelope)
        );
    }

    #[test]
    fn non_finite_envelope_rejected() {
        let env = vec![0.5, f32::NAN, 0.3, 0.2];
        match plan_floor1_x_list(&env, 2, 0.0) {
            Err(Floor1LayoutError::NonFiniteEnvelope { bin, value }) => {
                assert_eq!(bin, 1);
                assert!(value.is_nan());
            }
            other => panic!("expected NonFiniteEnvelope, got {other:?}"),
        }
        let env2 = vec![0.5, -0.1, 0.3, 0.2];
        assert_eq!(
            plan_floor1_x_list(&env2, 2, 0.0),
            Err(Floor1LayoutError::NonFiniteEnvelope {
                bin: 1,
                value: -0.1
            })
        );
    }

    // ---------- post-count guards ----------

    #[test]
    fn zero_posts_rejected() {
        let env = vec![0.1; 16];
        assert_eq!(
            plan_floor1_x_list(&env, 0, 0.0),
            Err(Floor1LayoutError::ZeroPosts)
        );
    }

    #[test]
    fn too_many_posts_rejected() {
        // 8 bins → 7 interior coordinates (1..8); asking for 8 is too many.
        let env = vec![0.1; 8];
        assert_eq!(
            plan_floor1_x_list(&env, 8, 0.0),
            Err(Floor1LayoutError::TooManyPosts {
                requested: 8,
                available: 7
            })
        );
        // Exactly 7 is allowed.
        assert!(plan_floor1_x_list(&env, 7, 0.0).is_ok());
    }

    // ---------- placement properties ----------

    #[test]
    fn x_list_is_sorted_unique_and_interior() {
        // A bumpy envelope so the greedy refinement actually spreads posts.
        let env: Vec<f32> = (0..64)
            .map(|k| 0.01 + 0.5 * ((k as f32 * 0.3).sin().abs()))
            .collect();
        let x_list = plan_floor1_x_list(&env, 12, 0.0).unwrap();
        assert_eq!(x_list.len(), 12, "budget filled at tolerance 0");
        // Sorted strictly ascending (hence unique).
        for w in x_list.windows(2) {
            assert!(w[0] < w[1], "x_list must be strictly ascending: {x_list:?}");
        }
        // Every coordinate is a valid interior bin (1..n).
        for &x in &x_list {
            assert!((1..64).contains(&(x as usize)), "interior bin: {x}");
        }
    }

    #[test]
    fn tolerance_stops_early_on_flat_envelope() {
        // A perfectly flat envelope needs no interior posts: the endpoint
        // line already reconstructs it exactly, so any positive tolerance
        // halts placement immediately.
        let env = vec![0.2_f32; 32];
        let x_list = plan_floor1_x_list(&env, 16, 0.5).unwrap();
        assert!(
            x_list.is_empty(),
            "flat envelope needs no interior posts, got {x_list:?}"
        );
    }

    #[test]
    fn adaptive_beats_uniform_on_a_peaky_envelope() {
        // An envelope with a sharp peak: the adaptive placement should
        // concentrate posts around the peak and reconstruct it markedly
        // better than evenly-spaced posts at the same budget.
        let n = 128usize;
        let env: Vec<f32> = (0..n)
            .map(|k| {
                let d = (k as f32 - 40.0) / 6.0;
                0.005 + 0.6 * (-d * d).exp()
            })
            .collect();
        let budget = 10usize;
        let multiplier = 1u8;

        // --- adaptive placement ---
        let adaptive_x = plan_floor1_x_list(&env, budget, 0.0).unwrap();
        let adaptive_hdr = header_from_x_list(adaptive_x.clone(), multiplier, n);
        let adaptive_sse = reconstruct_sse(&env, &adaptive_hdr);

        // --- uniform placement at the same budget ---
        let mut uniform_x: Vec<u32> = Vec::with_capacity(budget);
        for j in 1..=budget {
            let x = (j * n) / (budget + 1);
            uniform_x.push(x.clamp(1, n - 1) as u32);
        }
        uniform_x.dedup();
        let uniform_hdr = header_from_x_list(uniform_x, multiplier, n);
        let uniform_sse = reconstruct_sse(&env, &uniform_hdr);

        assert!(
            adaptive_sse < uniform_sse,
            "adaptive ({adaptive_sse}) should beat uniform ({uniform_sse})"
        );
    }

    /// Drive the full envelope → posts → packet-domain → decode → curve
    /// fidelity path for a header and return the sum-squared linear-domain
    /// reconstruction error against the envelope.
    fn reconstruct_sse(env: &[f32], header: &Floor1Header) -> f32 {
        let target_final = plan_floor1_envelope(env, header).unwrap();
        let floor1_y = plan_floor1_y(&target_final, header).unwrap();
        // Reconstruct final_Y the decoder would draw, then render the curve.
        // `decode`-equivalent curve via the public render path: build a
        // decoder and synthesise from the packet posts.
        let decoder = Floor1Decoder::new(header, &codebooks_for(header)).unwrap();
        let rendered = decoder.render_curve(&floor1_y, env.len());
        env.iter()
            .zip(rendered.iter())
            .map(|(&e, &r)| {
                let d = e - r;
                d * d
            })
            .sum()
    }

    /// Minimal codebook table sufficient for a `dimensions = 1`,
    /// `subclasses = 0` floor-1 header: the render path (`render_curve`)
    /// does not consult the codebooks, so an empty table suffices for the
    /// decoder build the SSE helper uses.
    fn codebooks_for(_header: &Floor1Header) -> Vec<crate::codebook::VorbisCodebook> {
        Vec::new()
    }

    // ---------- min_rangebits ----------

    #[test]
    fn min_rangebits_picks_smallest_covering_power() {
        assert_eq!(min_rangebits(1).unwrap(), 0);
        assert_eq!(min_rangebits(2).unwrap(), 1);
        assert_eq!(min_rangebits(3).unwrap(), 2);
        assert_eq!(min_rangebits(4).unwrap(), 2);
        assert_eq!(min_rangebits(128).unwrap(), 7);
        assert_eq!(min_rangebits(129).unwrap(), 8);
        assert_eq!(min_rangebits(1024).unwrap(), 10);
    }

    #[test]
    fn full_x_list_round_trips_planned_coordinates() {
        // The planned x-list, placed in a header, reappears (with the two
        // implicit endpoints prepended) through `full_x_list`.
        let env: Vec<f32> = (0..32).map(|k| 0.01 + 0.01 * k as f32).collect();
        let x_list = plan_floor1_x_list(&env, 5, 0.0).unwrap();
        let hdr = header_from_x_list(x_list.clone(), 1, 32);
        let full = full_x_list(&hdr);
        assert_eq!(full[0], 0);
        assert_eq!(full[1], 1u32 << hdr.rangebits);
        assert_eq!(&full[2..], &x_list[..]);
    }
}

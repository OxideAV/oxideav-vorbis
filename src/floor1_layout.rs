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
//! This module supplies that design in three composable pieces:
//!
//! * the **x-list (post-placement) planner** ([`plan_floor1_x_list`]) —
//!   where the posts sit;
//! * the **partition layout planner** ([`plan_floor1_partition_layout`]) —
//!   how the posts group into §7.2.2 partitions/classes;
//! * the **one-call header designer** ([`design_floor1_header`]) — the
//!   composition that, from an envelope and a class catalogue, assembles a
//!   write-ready [`crate::setup::Floor1Header`] the per-packet chain
//!   consumes.
//!
//! The floor renders as straight integer line
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
//! This module decides the floor's **geometry** — post coordinates and
//! partition grouping — and the `rangebits` / `multiplier` carriage around
//! them. It does **not** design the codebook *contents* (the master/sub
//! book bit allocations a class references): the caller supplies the
//! `Floor1Class` catalogue, and the planner chooses *which* class (by
//! dimension) tiles each partition. Codebook-content design (the
//! rate-distortion bit-allocation problem) remains the open followup.

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

/// The §7.2.2 ceiling on `floor1_partitions` (a 5-bit field).
const MAX_PARTITIONS: usize = 31;

/// The §7.2.2 ceiling on a partition-class index (a 4-bit field in
/// `floor1_partition_class_list`).
const MAX_CLASS_INDEX: usize = 15;

/// A planned floor-1 partition layout: the `floor1_partitions` count and
/// the `floor1_partition_class_list` that tiles a chosen explicit-post
/// count across a caller-supplied class catalogue (Vorbis I §7.2.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Floor1PartitionLayout {
    /// `floor1_partitions` — the [`Floor1Header::partitions`] value
    /// (`partition_class_list.len()`).
    ///
    /// [`Floor1Header::partitions`]: crate::setup::Floor1Header::partitions
    pub partitions: u8,
    /// `floor1_partition_class_list` — one class index per partition, in
    /// header order. The sum of `classes[partition_class_list[p]].dimensions`
    /// over all partitions equals the explicit-post count this layout was
    /// planned for.
    pub partition_class_list: Vec<u8>,
}

/// Plan a floor-1 **partition layout** that tiles `posts` explicit
/// x-coordinates across a caller-supplied class catalogue (Vorbis I
/// §7.2.2, encode direction).
///
/// Each floor-1 partition draws `class.dimensions` Y-values at once; the
/// partitions' dimensions must sum to exactly the explicit-post count
/// (`header.x_list.len()`). This planner picks, for each partition, the
/// largest available class dimension that still fits the remaining posts —
/// a greedy descending tiling that minimises the partition count (fewer
/// 4-bit class-list entries and fewer §7.2.3 master/sub codewords per
/// packet). It returns the `partition_class_list` indexing into
/// `class_dims` (the per-class `dimensions`, in the order the caller will
/// place the `Floor1Class` entries) plus the partition count.
///
/// `class_dims` is the catalogue of available class dimensions (each
/// `1..=8` per §7.2.2; the codebook a class carries fixes its dimension).
/// The planner only chooses *which* class each partition uses; the class
/// contents (sub-book assignments, master book) remain the caller's.
///
/// # Errors
///
/// * [`Floor1LayoutError::ZeroPosts`] if `posts == 0`.
/// * [`Floor1LayoutError::NoTilingClass`] if no class dimension can be used
///   to exactly tile the remaining posts (e.g. only dimension-`3` classes
///   exist but `posts == 4`), or if the tiling would need more than the
///   §7.2.2 ceiling of 31 partitions.
pub fn plan_floor1_partition_layout(
    posts: usize,
    class_dims: &[u8],
) -> Result<Floor1PartitionLayout, Floor1LayoutError> {
    if posts == 0 {
        return Err(Floor1LayoutError::ZeroPosts);
    }
    if class_dims.is_empty() || class_dims.len() > MAX_CLASS_INDEX + 1 {
        return Err(Floor1LayoutError::NoTilingClass { posts });
    }
    // Every class dimension must be a legal §7.2.2 value (1..=8); a 0 would
    // never advance the tiling, and >8 cannot be encoded.
    if class_dims.iter().any(|&d| !(1..=8).contains(&d)) {
        return Err(Floor1LayoutError::NoTilingClass { posts });
    }

    // Dynamic-programming exact tiling: `reach[r]` is the minimum partition
    // count to tile exactly `r` posts, and `from[r]` the class index used
    // for the last partition on that optimal path. Greedy-descending alone
    // can dead-end (e.g. dims {3,2}, posts 4 — greedy takes 3 then strands
    // 1); the DP always finds an exact tiling when one exists.
    let mut reach = vec![usize::MAX; posts + 1];
    let mut from = vec![0u8; posts + 1];
    reach[0] = 0;
    for r in 1..=posts {
        for (ci, &d) in class_dims.iter().enumerate() {
            let d = d as usize;
            if d <= r && reach[r - d] != usize::MAX && reach[r - d] + 1 < reach[r] {
                reach[r] = reach[r - d] + 1;
                from[r] = ci as u8;
            }
        }
    }
    if reach[posts] == usize::MAX {
        return Err(Floor1LayoutError::NoTilingClass { posts });
    }
    if reach[posts] > MAX_PARTITIONS {
        return Err(Floor1LayoutError::NoTilingClass { posts });
    }

    // Reconstruct the class sequence from the back, then reverse so the
    // partition_class_list is in ascending-post (header) order. Sorting the
    // sequence is not required (the spec allows any order); keeping the
    // natural reconstruction order is deterministic and stable.
    let mut class_list: Vec<u8> = Vec::with_capacity(reach[posts]);
    let mut r = posts;
    while r > 0 {
        let ci = from[r];
        class_list.push(ci);
        r -= class_dims[ci as usize] as usize;
    }
    class_list.reverse();

    Ok(Floor1PartitionLayout {
        partitions: class_list.len() as u8,
        partition_class_list: class_list,
    })
}

/// Design a complete floor-1 **setup header** from a desired envelope and
/// a caller-supplied class catalogue (Vorbis I §7.2.2, encode direction).
///
/// This is the one-call composition that ties the layout module to the
/// existing per-packet floor-1 encode chain
/// ([`crate::floor1_encode::plan_floor1_packet`]). Given a representative
/// linear-domain envelope, a post budget, a fit tolerance, the
/// `floor1_multiplier`, and the `Floor1Class` entries the stream's
/// codebooks support, it:
///
/// 1. places the explicit x-coordinates by adaptive refinement
///    ([`plan_floor1_x_list`]);
/// 2. tiles those posts into partitions over the classes' dimensions
///    ([`plan_floor1_partition_layout`]);
/// 3. picks the smallest `rangebits` covering the floor length
///    ([`min_rangebits`]);
///
/// and assembles the [`crate::setup::Floor1Header`] the per-packet chain
/// consumes. The x-list the planner produced is re-sorted ascending and
/// reordered to match the partition tiling: partition `p` (class
/// `partition_class_list[p]`, dimension `d`) owns the next `d` explicit
/// x-coordinates in ascending order, so the header's `x_list` is exactly
/// the ascending placement (the decoder reconstructs post identity from
/// the x-values, not their header position, so any consistent ordering
/// renders the same curve; ascending keeps it canonical).
///
/// The classes are taken verbatim — the planner chooses *which* class
/// tiles each partition (by dimension), not the class contents (sub-book
/// assignments, master book). Those, and the codebooks the
/// `subclasses > 0` classes reference, remain the caller's responsibility.
///
/// # Errors
///
/// Any [`Floor1LayoutError`] the placement or tiling raises:
/// empty/non-finite envelope, an out-of-budget or untileable post count,
/// or an illegal multiplier (outside `1..=4`, surfaced as
/// [`Floor1LayoutError::NoTilingClass`] is *not* used — multiplier is
/// validated downstream by the envelope fit, so this function accepts any
/// `multiplier` and lets the per-packet chain reject an illegal one).
pub fn design_floor1_header(
    envelope: &[f32],
    max_posts: usize,
    tolerance: f32,
    multiplier: u8,
    classes: &[crate::setup::Floor1Class],
) -> Result<crate::setup::Floor1Header, Floor1LayoutError> {
    if classes.is_empty() {
        return Err(Floor1LayoutError::NoTilingClass { posts: max_posts });
    }
    let floor_length = envelope.len();
    let rangebits = min_rangebits(floor_length)?;

    // 1) place the explicit posts.
    let x_list = plan_floor1_x_list(envelope, max_posts, tolerance)?;
    let posts = x_list.len();
    if posts == 0 {
        // A flat-enough envelope wanted no interior posts; the floor is the
        // two-endpoint segment. Build a single dimension-0 partition is
        // illegal (dimensions are 1..=8), so an endpoint-only floor uses
        // zero partitions — a legal §7.2.2 header (partitions = 0).
        return Ok(crate::setup::Floor1Header {
            partitions: 0,
            partition_class_list: Vec::new(),
            classes: classes.to_vec(),
            multiplier,
            rangebits,
            x_list: Vec::new(),
        });
    }

    // 2) tile the posts into partitions over the classes' dimensions.
    let class_dims: Vec<u8> = classes.iter().map(|c| c.dimensions).collect();
    let layout = plan_floor1_partition_layout(posts, &class_dims)?;

    Ok(crate::setup::Floor1Header {
        partitions: layout.partitions,
        partition_class_list: layout.partition_class_list,
        classes: classes.to_vec(),
        multiplier,
        rangebits,
        x_list,
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

    // ---------- partition layout ----------

    /// Sum the planned partitions' dimensions to confirm an exact tiling.
    fn tiled_posts(layout: &Floor1PartitionLayout, class_dims: &[u8]) -> usize {
        layout
            .partition_class_list
            .iter()
            .map(|&ci| class_dims[ci as usize] as usize)
            .sum()
    }

    #[test]
    fn partition_layout_tiles_exactly() {
        let dims = [1u8, 2, 4];
        for posts in 1..=40usize {
            let layout = plan_floor1_partition_layout(posts, &dims).unwrap();
            assert_eq!(
                tiled_posts(&layout, &dims),
                posts,
                "tiling must sum to {posts}"
            );
            assert_eq!(
                layout.partitions as usize,
                layout.partition_class_list.len()
            );
        }
    }

    #[test]
    fn partition_layout_minimises_partition_count() {
        // dims {2,4}, posts 8 → fewest partitions is two dim-4 (count 2),
        // not four dim-2 (count 4).
        let dims = [2u8, 4];
        let layout = plan_floor1_partition_layout(8, &dims).unwrap();
        assert_eq!(layout.partitions, 2);
        assert_eq!(tiled_posts(&layout, &dims), 8);
    }

    #[test]
    fn partition_layout_dp_finds_non_greedy_tiling() {
        // dims {2,3}, posts 4 → greedy-descending takes 3 then strands 1
        // (no dim-1 class); the DP finds 2+2. Two partitions, both dim-2.
        let dims = [2u8, 3];
        let layout = plan_floor1_partition_layout(4, &dims).unwrap();
        assert_eq!(tiled_posts(&layout, &dims), 4);
        assert_eq!(layout.partitions, 2);
    }

    #[test]
    fn partition_layout_rejects_untileable() {
        // Only dim-3 classes; posts 4 cannot be tiled.
        let dims = [3u8];
        assert_eq!(
            plan_floor1_partition_layout(4, &dims),
            Err(Floor1LayoutError::NoTilingClass { posts: 4 })
        );
    }

    #[test]
    fn partition_layout_guards() {
        assert_eq!(
            plan_floor1_partition_layout(0, &[1]),
            Err(Floor1LayoutError::ZeroPosts)
        );
        // Empty catalogue.
        assert_eq!(
            plan_floor1_partition_layout(4, &[]),
            Err(Floor1LayoutError::NoTilingClass { posts: 4 })
        );
        // Illegal class dimension (0 / >8).
        assert_eq!(
            plan_floor1_partition_layout(4, &[0, 2]),
            Err(Floor1LayoutError::NoTilingClass { posts: 4 })
        );
        assert_eq!(
            plan_floor1_partition_layout(4, &[9]),
            Err(Floor1LayoutError::NoTilingClass { posts: 4 })
        );
    }

    #[test]
    fn partition_layout_respects_partition_ceiling() {
        // Only dim-1 classes, posts 32 → would need 32 partitions, over
        // the §7.2.2 5-bit ceiling of 31.
        let dims = [1u8];
        assert_eq!(
            plan_floor1_partition_layout(32, &dims),
            Err(Floor1LayoutError::NoTilingClass { posts: 32 })
        );
        // 31 posts of dim-1 is exactly the ceiling — allowed.
        assert!(plan_floor1_partition_layout(31, &dims).is_ok());
    }

    // ---------- one-call header design ----------

    /// A catalogue of `subclasses = 0` classes with the given dimensions.
    /// Each uses sub-book slot 0 with no codebook (the negative-book path),
    /// so the §7.2.4 render needs no codebook table.
    fn catalogue(dims: &[u8]) -> Vec<Floor1Class> {
        dims.iter()
            .map(|&d| Floor1Class {
                dimensions: d,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![None],
            })
            .collect()
    }

    #[test]
    fn designed_header_is_structurally_valid_and_fits() {
        let n = 128usize;
        let env: Vec<f32> = (0..n)
            .map(|k| {
                let d = (k as f32 - 50.0) / 10.0;
                0.01 + 0.5 * (-d * d).exp()
            })
            .collect();
        let classes = catalogue(&[1, 2, 4]);
        let header = design_floor1_header(&env, 12, 0.0, 1, &classes).unwrap();

        // The partitions' dimensions tile the explicit x-list exactly.
        let tiled: usize = header
            .partition_class_list
            .iter()
            .map(|&ci| header.classes[ci as usize].dimensions as usize)
            .sum();
        assert_eq!(tiled, header.x_list.len());
        assert_eq!(
            header.partitions as usize,
            header.partition_class_list.len()
        );

        // The header builds a decoder (validates §7.2.2 undecodability:
        // unique x-list, multiplier, <= 65 values). No codebook table is
        // needed for the subclasses = 0 negative-book classes.
        let decoder = Floor1Decoder::new(&header, &[]).unwrap();

        // The designed header fits the envelope: plan posts, render, and
        // confirm the reconstruction is close (the designer chose where the
        // posts go to make this true). Compare against a uniform-x header
        // at the same post budget — the designed one must be no worse.
        let sse_designed = reconstruct_sse(&env, &header);
        let _ = decoder; // decoder build validated above

        let mut uniform_x: Vec<u32> = Vec::new();
        let posts = header.x_list.len();
        for j in 1..=posts {
            uniform_x.push(((j * n) / (posts + 1)).clamp(1, n - 1) as u32);
        }
        uniform_x.dedup();
        let uniform_hdr = header_from_x_list(uniform_x, 1, n);
        let sse_uniform = reconstruct_sse(&env, &uniform_hdr);
        assert!(
            sse_designed <= sse_uniform,
            "designed ({sse_designed}) must be <= uniform ({sse_uniform})"
        );
    }

    #[test]
    fn designed_header_flat_envelope_is_endpoint_only() {
        let env = vec![0.15_f32; 64];
        let classes = catalogue(&[1, 2]);
        let header = design_floor1_header(&env, 8, 0.5, 1, &classes).unwrap();
        assert_eq!(header.partitions, 0);
        assert!(header.x_list.is_empty());
        assert!(header.partition_class_list.is_empty());
        // An endpoint-only floor still builds a valid decoder.
        assert!(Floor1Decoder::new(&header, &[]).is_ok());
    }

    #[test]
    fn designed_header_empty_catalogue_rejected() {
        let env = vec![0.1_f32, 0.4, 0.2, 0.6, 0.1, 0.3];
        assert_eq!(
            design_floor1_header(&env, 3, 0.0, 1, &[]),
            Err(Floor1LayoutError::NoTilingClass { posts: 3 })
        );
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

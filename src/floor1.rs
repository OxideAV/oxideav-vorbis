//! Vorbis I floor type 1 per-packet decode + curve computation (Vorbis
//! I §7.2.3 "packet decode" + §7.2.4 "curve computation").
//!
//! A *floor* encodes the coarse spectral envelope of one channel in an
//! audio frame. The floor 1 header (§7.2.2) is parsed at setup time by
//! [`crate::setup`] into a [`Floor1Header`]; this module performs the
//! runtime per-packet decode that turns the floor payload of an audio
//! packet into a linear-domain spectral envelope of length `n`
//! (= `blocksize/2`).
//!
//! # Two stages
//!
//! Floor 1 decode is two logical stages, both implemented here:
//!
//! 1. **Packet decode (§7.2.3).** Read the `[nonzero]` flag; if unset
//!    the channel has no energy this frame and decode returns
//!    [`FloorCurve::Unused`]. Otherwise read the two endpoint amplitudes
//!    (`ilog([range]-1)` bits each) and, per partition, read each
//!    class's master-book selector (scalar context) then the per-element
//!    sub-book amplitudes (scalar context). This yields the `[floor1_Y]`
//!    vector. An end-of-packet condition anywhere here is a *nominal*
//!    occurrence: decode returns [`FloorCurve::Unused`] exactly as if
//!    `[nonzero]` had been clear (§7.2.3 closing note).
//!
//! 2. **Curve computation (§7.2.4).** Step 1 unwraps the always-positive
//!    `[floor1_Y]` differences into signed corrections applied through
//!    iterative line prediction, producing `[floor1_final_Y]` and the
//!    `[floor1_step2_flag]` vector. Step 2 sorts the `(X, final_Y,
//!    step2_flag)` triples by ascending X, renders the contiguous
//!    integer line segments with [`render_line`], and finally substitutes
//!    each integer floor sample through [`INVERSE_DB_TABLE`] to obtain a
//!    linear-domain envelope.
//!
//! # The implicit endpoints
//!
//! The header `[floor1_X_list]` stored in [`Floor1Header::x_list`]
//! *excludes* the two implicit endpoints `0` and `2^rangebits`; this
//! module prepends them (matching the §7.2.2 setup loop that injects
//! element 0 = 0 and element 1 = `2^rangebits` before reading the
//! per-partition x-values). `[floor1_values]` is therefore
//! `x_list.len() + 2`.
//!
//! # Helper functions
//!
//! [`low_neighbor`] / [`high_neighbor`] (§9.2.4 / §9.2.5),
//! [`render_point`] (§9.2.6) and [`render_line`] (§9.2.7) are the integer
//! geometry primitives the spec defines for floor 1; they are public so
//! the §4.3 audio-packet driver (a later round) and tests can reuse
//! them.

use crate::codebook::{ilog, VorbisCodebook};
use crate::huffman::{BuildError, HuffmanTree};
use crate::setup::{Floor1Class, Floor1Header};
use oxideav_core::bits::BitReaderLsb;

/// The `{256, 128, 86, 64}` range table indexed by `[floor1_multiplier]
/// - 1` (§7.2.3 step 1 / §7.2.4 step-1 step 1).
const RANGE_TABLE: [u32; 4] = [256, 128, 86, 64];

/// Result of a floor 1 packet decode.
#[derive(Debug, Clone, PartialEq)]
pub enum FloorCurve {
    /// The channel carried no audio energy this frame (the `[nonzero]`
    /// flag was clear, or an end-of-packet condition occurred during
    /// decode). §4.3.2 step 6 sets `[no_residue]` true for this channel.
    Unused,
    /// The decoded linear-domain spectral envelope of length `n`. Each
    /// element is the `[floor1_inverse_dB_static_table]` lookup of the
    /// integer curve value at that frequency bin (§7.2.4 step-2 step 15).
    Curve(Vec<f32>),
}

/// Errors that can arise while preparing or running a floor 1 decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Floor1Error {
    /// A `[floor1_class_masterbooks]` or `[floor1_subclass_books]`
    /// codebook index points outside the supplied codebook table.
    /// §7.2.2: "a `[floor1_class_masterbooks]` or `[floor1_subclass_
    /// books]` scalar element greater than the highest numbered codebook
    /// configured in this stream … renders the stream undecodable."
    BookOutOfRange {
        /// Partition-class index.
        class: usize,
        /// The offending codebook index.
        book: u8,
        /// The number of codebooks available.
        codebook_count: usize,
    },
    /// `[floor1_multiplier]` was outside the valid range `1..=4`. The
    /// setup parser already adds one to the 2-bit field (yielding
    /// `1..=4`), but [`Floor1Decoder::new`] re-checks because it may be
    /// constructed from a hand-built [`Floor1Header`].
    BadMultiplier(u8),
    /// `[floor1_values]` (= `x_list.len() + 2`) exceeds the §7.2.2
    /// hard limit of 65 elements.
    TooManyValues(usize),
    /// A `[floor1_X_list]` value (including the implicit endpoints)
    /// was not unique. §7.2.2: "All vector `[floor1 x list]` element
    /// values must be unique within the vector; a non-unique value
    /// renders the stream undecodable."
    NonUniqueXList,
    /// Building a class master- or sub-book's Huffman tree failed.
    Huffman(BuildError),
}

impl core::fmt::Display for Floor1Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Floor1Error::BookOutOfRange {
                class,
                book,
                codebook_count,
            } => write!(
                f,
                "floor1 class {class} references codebook {book} but only \
                 {codebook_count} codebooks are configured"
            ),
            Floor1Error::BadMultiplier(m) => {
                write!(f, "floor1_multiplier {m} is outside the valid range 1..=4")
            }
            Floor1Error::TooManyValues(n) => write!(
                f,
                "floor1_values {n} exceeds the §7.2.2 limit of 65 elements"
            ),
            Floor1Error::NonUniqueXList => {
                write!(f, "floor1_X_list contains a non-unique value (§7.2.2)")
            }
            Floor1Error::Huffman(e) => write!(f, "floor1 codebook tree build failed: {e}"),
        }
    }
}

impl std::error::Error for Floor1Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Floor1Error::Huffman(e) => Some(e),
            _ => None,
        }
    }
}

impl From<BuildError> for Floor1Error {
    fn from(value: BuildError) -> Self {
        Floor1Error::Huffman(value)
    }
}

/// A pre-validated floor 1 decoder built once from a [`Floor1Header`]
/// and the stream's codebook table. Building it up front validates the
/// §7.2.2 undecodability clauses (book indices in range, multiplier in
/// `1..=4`, `[floor1_values]` ≤ 65, x-list uniqueness) and pre-builds
/// the master-book + sub-book Huffman trees so the per-packet decode is
/// allocation-light.
#[derive(Debug, Clone)]
pub struct Floor1Decoder {
    /// `[floor1_multiplier]`, in `1..=4`.
    multiplier: u8,
    /// The full sorted-by-list-order `[floor1_X_list]` *including* the
    /// two implicit endpoints `0` and `2^rangebits` at positions 0 and
    /// 1 (so this matches the spec's element indexing exactly).
    x_list: Vec<u32>,
    /// `[floor1_partition_class_list]` — the partition-class index for
    /// each partition.
    partition_class_list: Vec<u8>,
    /// Per-partition-class configuration, paired with its pre-built
    /// codebook trees.
    classes: Vec<ClassDecode>,
}

/// One partition class with its codebook trees resolved.
#[derive(Debug, Clone)]
struct ClassDecode {
    /// `[floor1_class_dimensions]` — number of Y values this class
    /// encodes at once.
    dimensions: u8,
    /// `[floor1_class_subclasses]` — `cbits`; `2^cbits` alternate books.
    subclasses: u8,
    /// The master-book Huffman tree, present only when `subclasses > 0`
    /// (§7.2.3 step 11 only reads it when `[cbits] > 0`).
    masterbook_tree: Option<HuffmanTree>,
    /// Per-subclass codebook trees. `None` entries mean the subclass
    /// book index was `-1` (encoded `0`) — "no codebook for this
    /// subclass" (§7.2.3 step 16: a negative book yields a 0 Y value).
    subclass_trees: Vec<Option<HuffmanTree>>,
}

impl Floor1Decoder {
    /// Build a [`Floor1Decoder`] from a parsed [`Floor1Header`] and the
    /// stream's codebook table.
    ///
    /// Validates the §7.2.2 undecodability clauses and pre-builds every
    /// referenced codebook's Huffman decision tree.
    pub fn new(header: &Floor1Header, codebooks: &[VorbisCodebook]) -> Result<Self, Floor1Error> {
        // §7.2.2: floor1_multiplier is 1..=4 (the 2-bit field + 1).
        if !(1..=4).contains(&header.multiplier) {
            return Err(Floor1Error::BadMultiplier(header.multiplier));
        }

        // Reconstruct the full x-list with the two implicit endpoints
        // 0 and 2^rangebits prepended (§7.2.2 steps 15..17). The header
        // stores only the per-partition values read after them.
        let mut x_list: Vec<u32> = Vec::with_capacity(header.x_list.len() + 2);
        x_list.push(0);
        x_list.push(1u32 << header.rangebits);
        x_list.extend_from_slice(&header.x_list);

        // §7.2.2: floor1 x list is limited to 65 elements.
        if x_list.len() > 65 {
            return Err(Floor1Error::TooManyValues(x_list.len()));
        }

        // §7.2.2: all x-list values must be unique.
        for i in 0..x_list.len() {
            for j in (i + 1)..x_list.len() {
                if x_list[i] == x_list[j] {
                    return Err(Floor1Error::NonUniqueXList);
                }
            }
        }

        let mut classes = Vec::with_capacity(header.classes.len());
        for (class_idx, class) in header.classes.iter().enumerate() {
            classes.push(Self::build_class(class_idx, class, codebooks)?);
        }

        Ok(Floor1Decoder {
            multiplier: header.multiplier,
            x_list,
            partition_class_list: header.partition_class_list.clone(),
            classes,
        })
    }

    /// Resolve one partition class's codebooks into Huffman trees.
    fn build_class(
        class_idx: usize,
        class: &Floor1Class,
        codebooks: &[VorbisCodebook],
    ) -> Result<ClassDecode, Floor1Error> {
        let resolve = |book: u8| -> Result<HuffmanTree, Floor1Error> {
            let cb = codebooks
                .get(book as usize)
                .ok_or(Floor1Error::BookOutOfRange {
                    class: class_idx,
                    book,
                    codebook_count: codebooks.len(),
                })?;
            Ok(HuffmanTree::from_codebook(cb)?)
        };

        // §7.2.3 step 11/12: the master book is only consulted when
        // [cbits] > 0. Build it only in that case.
        let masterbook_tree = if class.subclasses > 0 {
            match class.masterbook {
                Some(book) => Some(resolve(book)?),
                None => None,
            }
        } else {
            None
        };

        let mut subclass_trees = Vec::with_capacity(class.subclass_books.len());
        for sub in &class.subclass_books {
            match sub {
                // §7.2.3 step 16: a negative (None) book means "no
                // codebook for this subclass"; nothing to build.
                None => subclass_trees.push(None),
                Some(book) => subclass_trees.push(Some(resolve(*book)?)),
            }
        }

        Ok(ClassDecode {
            dimensions: class.dimensions,
            subclasses: class.subclasses,
            masterbook_tree,
            subclass_trees,
        })
    }

    /// `[floor1_values]` — the number of x-coordinates including the two
    /// implicit endpoints.
    pub fn floor1_values(&self) -> usize {
        self.x_list.len()
    }

    /// Run the §7.2.3 packet decode + §7.2.4 curve computation, producing
    /// a linear-domain spectral envelope of length `n` (or
    /// [`FloorCurve::Unused`]).
    ///
    /// `n` is the number of frequency bins to render (`blocksize/2`),
    /// supplied by the §4.3 audio-packet driver.
    pub fn decode(&self, reader: &mut BitReaderLsb<'_>, n: usize) -> FloorCurve {
        match self.packet_decode(reader) {
            // §7.2.3 closing note + nonzero-unset path: 'unused'.
            None => FloorCurve::Unused,
            Some(floor1_y) => FloorCurve::Curve(self.curve_computation(&floor1_y, n)),
        }
    }

    /// §7.2.3 packet decode. Returns `Some([floor1_Y])` on success or
    /// `None` for the 'unused' status (either `[nonzero]` was clear or
    /// an end-of-packet condition occurred mid-decode).
    fn packet_decode(&self, reader: &mut BitReaderLsb<'_>) -> Option<Vec<u32>> {
        // §7.2.3 step 1: read the [nonzero] flag.
        let nonzero = reader.read_bit().ok()?;
        if !nonzero {
            return None;
        }

        // §7.2.3 step 1 (assuming nonzero set): [range] from the table.
        let range = RANGE_TABLE[(self.multiplier - 1) as usize];
        let amp_bits = ilog(range - 1);

        let mut floor1_y: Vec<u32> = Vec::with_capacity(self.floor1_values());

        // §7.2.3 steps 2..3: the two endpoint amplitudes.
        floor1_y.push(read_u32_nominal(reader, amp_bits)?);
        floor1_y.push(read_u32_nominal(reader, amp_bits)?);

        // §7.2.3 step 4: [offset] = 2.
        // §7.2.3 step 5: iterate over partitions.
        for &class_no in &self.partition_class_list {
            let class = &self.classes[class_no as usize];
            // §7.2.3 steps 7..10.
            let cdim = class.dimensions as usize;
            let cbits = class.subclasses;
            let csub: u32 = (1u32 << cbits) - 1;
            let mut cval: u32 = 0;

            // §7.2.3 steps 11..12: read the master-book selector when
            // [cbits] > 0.
            if cbits > 0 {
                let tree = class.masterbook_tree.as_ref()?;
                cval = decode_entry_nominal(tree, reader)?;
            }

            // §7.2.3 step 13: iterate over the class's dimension.
            for _ in 0..cdim {
                // §7.2.3 step 14: select the sub-book.
                let sub_idx = (cval & csub) as usize;
                // §7.2.3 step 15: cval >>= cbits.
                cval >>= cbits;
                // §7.2.3 steps 16..18.
                match class.subclass_trees.get(sub_idx).and_then(|t| t.as_ref()) {
                    Some(tree) => {
                        let v = decode_entry_nominal(tree, reader)?;
                        floor1_y.push(v);
                    }
                    // book < 0 → Y element is 0.
                    None => floor1_y.push(0),
                }
            }
            // §7.2.3 step 19: [offset] += [cdim] is implicit in the push.
        }

        // §7.2.3 step 20: done.
        Some(floor1_y)
    }

    /// §7.2.4 curve computation (step 1 amplitude synthesis + step 2
    /// curve synthesis), returning the linear-domain envelope of length
    /// `n`.
    fn curve_computation(&self, floor1_y: &[u32], n: usize) -> Vec<f32> {
        let range = RANGE_TABLE[(self.multiplier - 1) as usize] as i32;
        let values = self.floor1_values();

        // -------- step 1: amplitude value synthesis --------
        let mut step2_flag = vec![false; values];
        let mut final_y = vec![0i32; values];

        // §7.2.4 step-1 steps 2..5: seed the two endpoints.
        step2_flag[0] = true;
        step2_flag[1] = true;
        final_y[0] = floor1_y[0] as i32;
        final_y[1] = floor1_y[1] as i32;

        // §7.2.4 step-1 step 6: iterate over the remaining points.
        for i in 2..values {
            let low = low_neighbor(&self.x_list, i);
            let high = high_neighbor(&self.x_list, i);

            // §7.2.4 step-1 step 9: predict the line value at x[i].
            let predicted = render_point(
                self.x_list[low] as i32,
                final_y[low],
                self.x_list[high] as i32,
                final_y[high],
                self.x_list[i] as i32,
            );

            // §7.2.4 step-1 steps 10..15.
            let val = floor1_y[i] as i32;
            let highroom = range - predicted;
            let lowroom = predicted;
            let room = if highroom < lowroom {
                highroom * 2
            } else {
                lowroom * 2
            };

            // §7.2.4 step-1 step 16.
            if val != 0 {
                step2_flag[low] = true;
                step2_flag[high] = true;
                step2_flag[i] = true;
                // §7.2.4 step-1 step 20.
                if val >= room {
                    // §7.2.4 step-1 steps 21..23.
                    if highroom > lowroom {
                        final_y[i] = val - lowroom + predicted;
                    } else {
                        final_y[i] = predicted - val + highroom - 1;
                    }
                } else {
                    // §7.2.4 step-1 steps 24..26.
                    if val % 2 == 1 {
                        final_y[i] = predicted - (val + 1) / 2;
                    } else {
                        final_y[i] = predicted + val / 2;
                    }
                }
            } else {
                // §7.2.4 step-1 steps 27..28.
                step2_flag[i] = false;
                final_y[i] = predicted;
            }
        }

        // §7.2.4 amplitude-clamp suggestion: guard final_Y to [0, range).
        // "valid floor1 setups cannot produce out of range values" but
        // abuse of the codebook machinery can; clamp defensively.
        for y in final_y.iter_mut() {
            *y = (*y).clamp(0, range - 1);
        }

        // -------- step 2: curve synthesis --------
        // Sort (X, final_Y, step2_flag) triples by ascending X.
        let mut order: Vec<usize> = (0..values).collect();
        order.sort_by_key(|&idx| self.x_list[idx]);
        let sorted_x: Vec<i32> = order.iter().map(|&idx| self.x_list[idx] as i32).collect();
        let sorted_y: Vec<i32> = order.iter().map(|&idx| final_y[idx]).collect();
        let sorted_flag: Vec<bool> = order.iter().map(|&idx| step2_flag[idx]).collect();

        let mut floor = vec![0i32; n];

        // §7.2.4 step-2 steps 1..3.
        let mut hx: i32 = 0;
        let mut hy: i32 = 0;
        let mut lx: i32 = 0;
        let mut ly: i32 = sorted_y[0] * self.multiplier as i32;

        // §7.2.4 step-2 step 4: walk the sorted points.
        for i in 1..values {
            // §7.2.4 step-2 step 5: only plot points whose flag is set.
            if sorted_flag[i] {
                hy = sorted_y[i] * self.multiplier as i32;
                hx = sorted_x[i];
                render_line(lx, ly, hx, hy, &mut floor);
                lx = hx;
                ly = hy;
            }
        }

        // §7.2.4 step-2 steps 11..14: extend the curve to [n].
        if (hx as usize) < n {
            render_line(hx, hy, n as i32, hy, &mut floor);
        }
        // (hx > n is impossible here because render_line only ever writes
        // within 0..n and the largest sorted X is 2^rangebits; the floor
        // buffer is fixed at length n so no truncation step is needed.)

        // §7.2.4 step-2 step 15: lookup-substitute through the table.
        floor
            .iter()
            .map(|&v| {
                let idx = (v as usize).min(INVERSE_DB_TABLE.len() - 1);
                INVERSE_DB_TABLE[idx]
            })
            .collect()
    }
}

/// §9.2.4 low_neighbor: position `n` in `v` of the greatest-valued
/// element for which `n < x` and `v[n] < v[x]`.
///
/// Returns `0` if no such element exists (a base case the floor 1
/// prediction loop never actually hits because positions 0 and 1 are
/// the global min/max endpoints, but defined for totality).
pub fn low_neighbor(v: &[u32], x: usize) -> usize {
    let mut best: Option<usize> = None;
    for (n, &vn) in v.iter().enumerate().take(x) {
        if vn < v[x] {
            match best {
                Some(b) if v[b] >= vn => {}
                _ => best = Some(n),
            }
        }
    }
    best.unwrap_or(0)
}

/// §9.2.5 high_neighbor: position `n` in `v` of the lowest-valued
/// element for which `n < x` and `v[n] > v[x]`.
///
/// Returns `0` if no such element exists.
pub fn high_neighbor(v: &[u32], x: usize) -> usize {
    let mut best: Option<usize> = None;
    for (n, &vn) in v.iter().enumerate().take(x) {
        if vn > v[x] {
            match best {
                Some(b) if v[b] <= vn => {}
                _ => best = Some(n),
            }
        }
    }
    best.unwrap_or(0)
}

/// §9.2.6 render_point: the integer Y value at `X` on the line through
/// `(x0, y0)`–`(x1, y1)`.
pub fn render_point(x0: i32, y0: i32, x1: i32, y1: i32, x: i32) -> i32 {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let ady = dy.abs();
    let err = ady * (x - x0);
    let off = err / adx;
    if dy < 0 {
        y0 - off
    } else {
        y0 + off
    }
}

/// §9.2.7 render_line: draw the integer line segment `(x0, y0)`–`(x1,
/// y1)` into `v` (Bresenham-style). The endpoint `x1` is *not* written
/// (it becomes the next segment's start), matching the §7.2.4 step-2
/// chaining where `[lx]` is set to `[hx]` after each call.
///
/// Integer division here rounds toward zero for both positive and
/// negative operands (Rust's `/` on `i32` already does this), as the
/// spec mandates for this function specifically.
pub fn render_line(x0: i32, y0: i32, x1: i32, y1: i32, v: &mut [i32]) {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let mut ady = dy.abs();
    let base = dy / adx;
    let mut x = x0;
    let mut y = y0;
    let mut err = 0i32;
    let sy = if dy < 0 { base - 1 } else { base + 1 };
    ady -= base.abs() * adx;

    if (x as usize) < v.len() {
        v[x as usize] = y;
    }

    // §9.2.7 step 13: iterate x over x0+1 ..= x1-1.
    x += 1;
    while x < x1 {
        err += ady;
        if err >= adx {
            err -= adx;
            y += sy;
        } else {
            y += base;
        }
        if (x as usize) < v.len() {
            v[x as usize] = y;
        }
        x += 1;
    }
}

/// Read `n` bits as an unsigned integer, mapping an end-of-packet to
/// the §7.2.3 'unused' (`None`) nominal occurrence.
fn read_u32_nominal(reader: &mut BitReaderLsb<'_>, n: u32) -> Option<u32> {
    reader.read_u32(n).ok()
}

/// Decode a codebook entry in scalar context, mapping an end-of-packet
/// to the §7.2.3 'unused' (`None`) nominal occurrence.
fn decode_entry_nominal(tree: &HuffmanTree, reader: &mut BitReaderLsb<'_>) -> Option<u32> {
    // The only error variant is `UnexpectedEndOfPacket`, which §7.2.3
    // treats as the nominal 'unused' occurrence — so `.ok()` collapses
    // it to `None`.
    tree.decode_entry(reader).ok()
}

/// §10.1 floor1_inverse_dB_table — the 256-element static lookup table
/// mapping an integer floor curve value to a linear-domain amplitude.
/// Transcribed verbatim from the Vorbis I specification §10.1 (read
/// left-to-right then top-to-bottom).
///
/// The spec prints several values with more significant digits than an
/// `f32` can represent; we keep the literals exactly as the spec writes
/// them (the compiler rounds each to the nearest representable `f32`,
/// which is the canonical decoder value) and silence clippy's
/// `excessive_precision` lint rather than truncate the spec text.
#[allow(clippy::excessive_precision)]
pub static INVERSE_DB_TABLE: [f32; 256] = [
    1.0649863e-07,
    1.1341951e-07,
    1.2079015e-07,
    1.2863978e-07,
    1.3699951e-07,
    1.4590251e-07,
    1.5538408e-07,
    1.6548181e-07,
    1.7623575e-07,
    1.8768855e-07,
    1.9988561e-07,
    2.1287530e-07,
    2.2670913e-07,
    2.4144197e-07,
    2.5713223e-07,
    2.7384213e-07,
    2.9163793e-07,
    3.1059021e-07,
    3.3077411e-07,
    3.5226968e-07,
    3.7516214e-07,
    3.9954229e-07,
    4.2550680e-07,
    4.5315863e-07,
    4.8260743e-07,
    5.1396998e-07,
    5.4737065e-07,
    5.8294187e-07,
    6.2082472e-07,
    6.6116941e-07,
    7.0413592e-07,
    7.4989464e-07,
    7.9862701e-07,
    8.5052630e-07,
    9.0579828e-07,
    9.6466216e-07,
    1.0273513e-06,
    1.0941144e-06,
    1.1652161e-06,
    1.2409384e-06,
    1.3215816e-06,
    1.4074654e-06,
    1.4989305e-06,
    1.5963394e-06,
    1.7000785e-06,
    1.8105592e-06,
    1.9282195e-06,
    2.0535261e-06,
    2.1869758e-06,
    2.3290978e-06,
    2.4804557e-06,
    2.6416497e-06,
    2.8133190e-06,
    2.9961443e-06,
    3.1908506e-06,
    3.3982101e-06,
    3.6190449e-06,
    3.8542308e-06,
    4.1047004e-06,
    4.3714470e-06,
    4.6555282e-06,
    4.9580707e-06,
    5.2802740e-06,
    5.6234160e-06,
    5.9888572e-06,
    6.3780469e-06,
    6.7925283e-06,
    7.2339451e-06,
    7.7040476e-06,
    8.2047000e-06,
    8.7378876e-06,
    9.3057248e-06,
    9.9104632e-06,
    1.0554501e-05,
    1.1240392e-05,
    1.1970856e-05,
    1.2748789e-05,
    1.3577278e-05,
    1.4459606e-05,
    1.5399272e-05,
    1.6400004e-05,
    1.7465768e-05,
    1.8600792e-05,
    1.9809576e-05,
    2.1096914e-05,
    2.2467911e-05,
    2.3928002e-05,
    2.5482978e-05,
    2.7139006e-05,
    2.8902651e-05,
    3.0780908e-05,
    3.2781225e-05,
    3.4911534e-05,
    3.7180282e-05,
    3.9596466e-05,
    4.2169667e-05,
    4.4910090e-05,
    4.7828601e-05,
    5.0936773e-05,
    5.4246931e-05,
    5.7772202e-05,
    6.1526565e-05,
    6.5524908e-05,
    6.9783085e-05,
    7.4317983e-05,
    7.9147585e-05,
    8.4291040e-05,
    8.9768747e-05,
    9.5602426e-05,
    0.00010181521,
    0.00010843174,
    0.00011547824,
    0.00012298267,
    0.00013097477,
    0.00013948625,
    0.00014855085,
    0.00015820453,
    0.00016848555,
    0.00017943469,
    0.00019109536,
    0.00020351382,
    0.00021673929,
    0.00023082423,
    0.00024582449,
    0.00026179955,
    0.00027881276,
    0.00029693158,
    0.00031622787,
    0.00033677814,
    0.00035866388,
    0.00038197188,
    0.00040679456,
    0.00043323036,
    0.00046138411,
    0.00049136745,
    0.00052329927,
    0.00055730621,
    0.00059352311,
    0.00063209358,
    0.00067317058,
    0.00071691700,
    0.00076350630,
    0.00081312324,
    0.00086596457,
    0.00092223983,
    0.00098217216,
    0.0010459992,
    0.0011139742,
    0.0011863665,
    0.0012634633,
    0.0013455702,
    0.0014330129,
    0.0015261382,
    0.0016253153,
    0.0017309374,
    0.0018434235,
    0.0019632195,
    0.0020908006,
    0.0022266726,
    0.0023713743,
    0.0025254795,
    0.0026895994,
    0.0028643847,
    0.0030505286,
    0.0032487691,
    0.0034598925,
    0.0036847358,
    0.0039241906,
    0.0041792066,
    0.0044507950,
    0.0047400328,
    0.0050480668,
    0.0053761186,
    0.0057254891,
    0.0060975636,
    0.0064938176,
    0.0069158225,
    0.0073652516,
    0.0078438871,
    0.0083536271,
    0.0088964928,
    0.009474637,
    0.010090352,
    0.010746080,
    0.011444421,
    0.012188144,
    0.012980198,
    0.013823725,
    0.014722068,
    0.015678791,
    0.016697687,
    0.017782797,
    0.018938423,
    0.020169149,
    0.021479854,
    0.022875735,
    0.024362330,
    0.025945531,
    0.027631618,
    0.029427276,
    0.031339626,
    0.033376252,
    0.035545228,
    0.037855157,
    0.040315199,
    0.042935108,
    0.045725273,
    0.048696758,
    0.051861348,
    0.055231591,
    0.058820850,
    0.062643361,
    0.066714279,
    0.071049749,
    0.075666962,
    0.080584227,
    0.085821044,
    0.091398179,
    0.097337747,
    0.10366330,
    0.11039993,
    0.11757434,
    0.12521498,
    0.13335215,
    0.14201813,
    0.15124727,
    0.16107617,
    0.17154380,
    0.18269168,
    0.19456402,
    0.20720788,
    0.22067342,
    0.23501402,
    0.25028656,
    0.26655159,
    0.28387361,
    0.30232132,
    0.32196786,
    0.34289114,
    0.36517414,
    0.38890521,
    0.41417847,
    0.44109412,
    0.46975890,
    0.50028648,
    0.53279791,
    0.56742212,
    0.60429640,
    0.64356699,
    0.68538959,
    0.72993007,
    0.77736504,
    0.82788260,
    0.88168307,
    0.9389798,
    1.0,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::{VorbisCodebook, VqLookup};
    use oxideav_core::bits::{BitReaderLsb, BitWriterLsb};

    // ---- helper-function tests (§9.2.4 .. §9.2.7) ----

    #[test]
    fn render_point_matches_spec_geometry() {
        // Line (0,0)->(128,16): at X=64 the integer value is 8.
        assert_eq!(render_point(0, 0, 128, 16, 64), 8);
        // Downward line (0,100)->(16,0): at X=8 the value is 50.
        assert_eq!(render_point(0, 100, 16, 0, 8), 50);
        // (0,0)->(10,5): at X=5 the truncating integer value is 2.
        assert_eq!(render_point(0, 0, 10, 5, 5), 2);
    }

    #[test]
    fn render_line_draws_upward_segment() {
        // §9.2.7: x1 is *not* written (it becomes the next segment start).
        let mut v = [0i32; 11];
        render_line(0, 0, 10, 5, &mut v);
        assert_eq!(v, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0]);
    }

    #[test]
    fn render_line_draws_downward_segment() {
        let mut v = [0i32; 11];
        render_line(0, 5, 10, 0, &mut v);
        assert_eq!(v, [5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0]);
    }

    #[test]
    fn render_line_draws_flat_segment() {
        let mut v = [0i32; 9];
        render_line(0, 2, 8, 2, &mut v);
        assert_eq!(v[..8], [2, 2, 2, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn low_and_high_neighbor_match_spec_definitions() {
        // x_list element 2 is value 4; only positions 0 (val 0) and
        // 1 (val 16) precede it.
        let v = [0u32, 16, 4, 8];
        // low_neighbor(v, 2): greatest value < 4 among indices < 2 → index 0.
        assert_eq!(low_neighbor(&v, 2), 0);
        // high_neighbor(v, 2): lowest value > 4 among indices < 2 → index 1.
        assert_eq!(high_neighbor(&v, 2), 1);
        // For index 3 (value 8): preceding values are 0,16,4.
        // low: greatest < 8 → 4 at index 2.
        assert_eq!(low_neighbor(&v, 3), 2);
        // high: lowest > 8 → 16 at index 1.
        assert_eq!(high_neighbor(&v, 3), 1);
    }

    // ---- decoder construction validation (§7.2.2 clauses) ----

    /// Minimal scalar codebook with `entries` length-1 entries (so a
    /// 2-entry book assigns entry 0 → '0', entry 1 → '1'). Scalar
    /// context only consults the Huffman tree, so lookup type is `None`.
    fn scalar_book(entries: u32) -> VorbisCodebook {
        VorbisCodebook {
            dimensions: 1,
            entries,
            codeword_lengths: vec![1u8; entries as usize],
            lookup: VqLookup::None,
        }
    }

    fn header_one_partition() -> Floor1Header {
        // 1 partition of class 0; class 0 has dimensions 2, subclasses 0,
        // one subclass book (index 0). rangebits 4 → endpoint1 = 16.
        // interior x_list = [4, 8].
        Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 2,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![Some(0)],
            }],
            multiplier: 2,
            rangebits: 4,
            x_list: vec![4, 8],
        }
    }

    #[test]
    fn new_rejects_bad_multiplier() {
        let mut h = header_one_partition();
        h.multiplier = 0;
        assert_eq!(
            Floor1Decoder::new(&h, &[scalar_book(2)]).unwrap_err(),
            Floor1Error::BadMultiplier(0)
        );
        h.multiplier = 5;
        assert_eq!(
            Floor1Decoder::new(&h, &[scalar_book(2)]).unwrap_err(),
            Floor1Error::BadMultiplier(5)
        );
    }

    #[test]
    fn new_rejects_book_out_of_range() {
        let mut h = header_one_partition();
        h.classes[0].subclass_books = vec![Some(7)];
        let err = Floor1Decoder::new(&h, &[scalar_book(2)]).unwrap_err();
        assert_eq!(
            err,
            Floor1Error::BookOutOfRange {
                class: 0,
                book: 7,
                codebook_count: 1,
            }
        );
    }

    #[test]
    fn new_rejects_too_many_values() {
        // 64 interior x-values + 2 implicit = 66 > 65.
        let mut h = header_one_partition();
        h.partition_class_list = vec![0];
        h.classes[0].dimensions = 8;
        // Just fabricate 64 unique interior values via x_list; the
        // partition/class config is not consulted by the length check.
        h.x_list = (1u32..=64).collect();
        let err = Floor1Decoder::new(&h, &[scalar_book(2)]).unwrap_err();
        assert_eq!(err, Floor1Error::TooManyValues(66));
    }

    #[test]
    fn new_rejects_non_unique_x_list() {
        let mut h = header_one_partition();
        // interior value 16 collides with the implicit endpoint 2^4 = 16.
        h.x_list = vec![4, 16];
        let err = Floor1Decoder::new(&h, &[scalar_book(2)]).unwrap_err();
        assert_eq!(err, Floor1Error::NonUniqueXList);
    }

    #[test]
    fn floor1_values_includes_implicit_endpoints() {
        let dec = Floor1Decoder::new(&header_one_partition(), &[scalar_book(2)]).unwrap();
        // 2 interior + 2 implicit = 4.
        assert_eq!(dec.floor1_values(), 4);
    }

    // ---- curve computation (§7.2.4) ----

    /// Hand-traced full curve_computation against the Python reference:
    /// x_list = [0,16,4,8], floor1_Y = [40,20,1,0], multiplier 2, n 16.
    /// final_Y = [40,20,34,30], step2 = [T,T,T,F].
    #[test]
    fn curve_computation_matches_hand_trace() {
        let dec = Floor1Decoder::new(&header_one_partition(), &[scalar_book(2)]).unwrap();
        let curve = dec.curve_computation(&[40, 20, 1, 0], 16);
        // Expected integer floor before inverse-dB substitution.
        let expected_int: [usize; 16] = [
            80, 77, 74, 71, 68, 66, 64, 61, 59, 57, 54, 52, 50, 47, 45, 43,
        ];
        let expected: Vec<f32> = expected_int.iter().map(|&i| INVERSE_DB_TABLE[i]).collect();
        assert_eq!(curve, expected);
    }

    // ---- packet decode (§7.2.3) ----

    /// Pack a full floor 1 audio-packet payload for the one-partition
    /// config: nonzero flag, two 7-bit endpoint amplitudes, then two
    /// length-1 scalar codewords. The scalar book is 2-entry length-1:
    /// entry 0 → bit '0', entry 1 → bit '1'.
    fn pack_packet(nonzero: bool, ep0: u32, ep1: u32, code_bits: &[bool]) -> Vec<u8> {
        let mut w = BitWriterLsb::with_capacity(8);
        w.write_bit(nonzero);
        if nonzero {
            // ilog(range-1) = ilog(127) = 7 for multiplier 2.
            w.write_u32(ep0, 7);
            w.write_u32(ep1, 7);
            for &b in code_bits {
                w.write_bit(b);
            }
        }
        w.finish()
    }

    #[test]
    fn packet_decode_unused_when_nonzero_clear() {
        let dec = Floor1Decoder::new(&header_one_partition(), &[scalar_book(2)]).unwrap();
        let packet = pack_packet(false, 0, 0, &[]);
        let mut r = BitReaderLsb::new(&packet);
        assert_eq!(dec.decode(&mut r, 16), FloorCurve::Unused);
    }

    #[test]
    fn packet_decode_full_curve_round_trip() {
        let dec = Floor1Decoder::new(&header_one_partition(), &[scalar_book(2)]).unwrap();
        // endpoints 40, 20; interior Y from a 2-entry book: entry 1 then
        // entry 0 → wrapped Y values [1, 0]. With the hand-trace above
        // this yields final_Y [40,20,34,30].
        let packet = pack_packet(true, 40, 20, &[true, false]);
        let mut r = BitReaderLsb::new(&packet);
        let expected_int: [usize; 16] = [
            80, 77, 74, 71, 68, 66, 64, 61, 59, 57, 54, 52, 50, 47, 45, 43,
        ];
        let expected: Vec<f32> = expected_int.iter().map(|&i| INVERSE_DB_TABLE[i]).collect();
        match dec.decode(&mut r, 16) {
            FloorCurve::Curve(c) => assert_eq!(c, expected),
            FloorCurve::Unused => panic!("expected a curve, got Unused"),
        }
    }

    #[test]
    fn packet_decode_eof_mid_amplitude_is_nominal_unused() {
        let dec = Floor1Decoder::new(&header_one_partition(), &[scalar_book(2)]).unwrap();
        // Only the nonzero bit + a partial endpoint: reader runs dry.
        let mut w = BitWriterLsb::with_capacity(1);
        w.write_bit(true);
        w.write_u32(0b101, 3); // fewer than the 7 endpoint bits.
        let packet = w.finish();
        let mut r = BitReaderLsb::new(&packet);
        // §7.2.3: EOF during decode → 'unused' as if nonzero were clear.
        assert_eq!(dec.decode(&mut r, 16), FloorCurve::Unused);
    }

    #[test]
    fn packet_decode_eof_mid_codeword_is_nominal_unused() {
        let dec = Floor1Decoder::new(&header_one_partition(), &[scalar_book(2)]).unwrap();
        // nonzero + both endpoints but NO codeword bits: the first
        // scalar read hits EOF → 'unused'.
        let packet = pack_packet(true, 40, 20, &[]);
        let mut r = BitReaderLsb::new(&packet);
        assert_eq!(dec.decode(&mut r, 16), FloorCurve::Unused);
    }

    // ---- master/subclass cascade (§7.2.3 steps 11..18) ----

    /// A class with subclasses > 0 reads a master-book selector first,
    /// then per-dimension sub-books chosen by the selector's low bits.
    #[test]
    fn packet_decode_master_subclass_cascade() {
        // class 0: dimensions 2, subclasses 1 → 2 subclass books,
        // masterbook present. csub = (1<<1)-1 = 1, cbits = 1.
        // codebooks: 0 = master (4-entry len-2), 1 = subbook A (2-entry),
        // 2 = subbook B (2-entry).
        let header = Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 2,
                subclasses: 1,
                masterbook: Some(0),
                subclass_books: vec![Some(1), Some(2)],
            }],
            multiplier: 2,
            rangebits: 4,
            x_list: vec![4, 8],
        };
        // 4-entry master book with lengths [2,2,2,2]: canonical codewords
        // entry0=00 entry1=01 entry2=10 entry3=11.
        let master = VorbisCodebook {
            dimensions: 1,
            entries: 4,
            codeword_lengths: vec![2, 2, 2, 2],
            lookup: VqLookup::None,
        };
        let books = vec![master, scalar_book(2), scalar_book(2)];
        let dec = Floor1Decoder::new(&header, &books).unwrap();

        // Choose master selector cval = 2 (binary 10).
        //   dim 0: sub_idx = cval & 1 = 0 → subbook A (index 1); cval >>= 1 → 1.
        //   dim 1: sub_idx = cval & 1 = 1 → subbook B (index 2); cval >>= 1 → 0.
        // So Y[2] read from subbook A, Y[3] from subbook B.
        let mut w = BitWriterLsb::with_capacity(8);
        w.write_bit(true); // nonzero
        w.write_u32(40, 7); // endpoint 0
        w.write_u32(20, 7); // endpoint 1
                            // master codeword for entry 2 = '10' (MSb first).
        w.write_bit(true);
        w.write_bit(false);
        // subbook A: entry 1 → bit '1' (Y[2] wrapped = 1).
        w.write_bit(true);
        // subbook B: entry 0 → bit '0' (Y[3] wrapped = 0).
        w.write_bit(false);
        let packet = w.finish();
        let mut r = BitReaderLsb::new(&packet);

        // Same wrapped Y as the round-trip test → same expected curve.
        let expected_int: [usize; 16] = [
            80, 77, 74, 71, 68, 66, 64, 61, 59, 57, 54, 52, 50, 47, 45, 43,
        ];
        let expected: Vec<f32> = expected_int.iter().map(|&i| INVERSE_DB_TABLE[i]).collect();
        match dec.decode(&mut r, 16) {
            FloorCurve::Curve(c) => assert_eq!(c, expected),
            FloorCurve::Unused => panic!("expected a curve, got Unused"),
        }
    }

    // ---- table sanity ----

    #[test]
    fn inverse_db_table_endpoints() {
        assert_eq!(INVERSE_DB_TABLE.len(), 256);
        assert_eq!(INVERSE_DB_TABLE[0], 1.0649863e-07);
        assert_eq!(INVERSE_DB_TABLE[255], 1.0);
    }

    #[test]
    fn negative_book_subclass_yields_zero_y() {
        // A subclass book set to None (encoded -1) yields Y = 0 for that
        // dimension without consuming any bits (§7.2.3 step 16/18).
        let header = Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 2,
                subclasses: 0,
                masterbook: None,
                // single subclass book is "no codebook".
                subclass_books: vec![None],
            }],
            multiplier: 2,
            rangebits: 4,
            x_list: vec![4, 8],
        };
        let dec = Floor1Decoder::new(&header, &[scalar_book(2)]).unwrap();
        // Only endpoints in the packet; both interior Y are forced to 0,
        // no codeword bits are read.
        let packet = pack_packet(true, 40, 20, &[]);
        let mut r = BitReaderLsb::new(&packet);
        // floor1_Y = [40, 20, 0, 0]; both interior are predicted-only.
        match dec.decode(&mut r, 16) {
            FloorCurve::Curve(c) => assert_eq!(c.len(), 16),
            FloorCurve::Unused => panic!("expected a curve"),
        }
    }
}

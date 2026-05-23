//! Vorbis I floor type 0 per-packet decode + curve computation (Vorbis
//! I §6.2.2 "packet decode" + §6.2.3 "curve computation").
//!
//! Floor type 0 encodes a smooth spectral envelope as the frequency
//! response of a Line-Spectral-Pair (LSP) filter (§6.1). The setup
//! header (parsed by [`crate::setup`] into a [`Floor0Header`]) carries
//! the LSP filter order, a Bark-map size, the per-packet amplitude bit
//! width, an amplitude offset, and the list of value codebooks the
//! per-packet decoder may pick from. This module performs the runtime
//! per-packet decode that turns the floor payload of an audio packet
//! into a linear-domain spectral envelope of length `n` (= `blocksize/2`).
//!
//! # Status
//!
//! libvorbis has never emitted floor 0 (every reference encoder since
//! the original beta uses floor 1). A conformant Vorbis I decoder must
//! still implement the codepath because the spec defines it, and a
//! third-party encoder could theoretically produce a floor-0 stream.
//! No fixture in `docs/audio/vorbis/fixtures/` exercises this codepath;
//! the unit tests in this module are hand-traced against the §6.2.2 /
//! §6.2.3 spec pseudocode.
//!
//! # Two stages
//!
//! 1. **Packet decode (§6.2.2).** Read the [amplitude] field
//!    (`floor0_amplitude_bits` bits). If zero the channel has no energy
//!    this frame and decode returns [`Floor0Curve::Unused`]. Otherwise
//!    read the [booknumber] (`ilog([floor0_number_of_books])` bits),
//!    validate it against `floor0_book_list.len()` (§6.2.2 step 5
//!    undecodability), then loop reading VQ vectors from
//!    `floor0_book_list[booknumber]` and concatenating them into
//!    [coefficients] until the vector reaches at least `floor0_order`
//!    elements. Each VQ vector has [last] added cumulatively (§6.2.2
//!    steps 6..9), then concatenated (step 10). The spec explicitly
//!    permits the loop to over-read: "the number of scalars read into
//!    the vector [coefficients] may be greater than [floor0_order],
//!    the number actually required for curve computation … extra
//!    values are not used and may be ignored or discarded."
//!
//! 2. **Curve computation (§6.2.3).** Build a Bark-scale frequency
//!    map `map[i]` for each output bin `i`, then synthesise the LSP
//!    log-amplitude curve through the order-dependent `[p]`/`[q]`
//!    product formula at every angle `[ω] = π × map[i] /
//!    floor0_bark_map_size`, scaling by the per-packet amplitude
//!    through the `exp(.11512925 × …)` log→linear transform and
//!    replicating the value across every output bin whose `map[i]`
//!    matches the current synthesis bin (the `[iteration_condition]`
//!    chaining).
//!
//! # End-of-packet
//!
//! Per §6.2.2: "an end-of-packet condition during decode should be
//! considered a nominal occurrence; if end-of-packet is reached during
//! any read operation above, floor decode is to return 'unused' status
//! as if the [amplitude] value had read zero at the beginning of
//! decode." [`Floor0Decoder::decode`] therefore folds every
//! end-of-packet into [`Floor0Curve::Unused`].

use crate::codebook::{ilog, VorbisCodebook, VqLookup};
use crate::huffman::{BuildError, HuffmanTree};
use crate::setup::Floor0Header;
use crate::vq::unpack_vector;
use oxideav_core::bits::BitReaderLsb;

/// Result of a floor 0 packet decode.
#[derive(Debug, Clone, PartialEq)]
pub enum Floor0Curve {
    /// The channel carried no audio energy this frame (the [amplitude]
    /// field was zero, or an end-of-packet condition occurred during
    /// decode). §4.3.2 step 6 sets `[no_residue]` true for this channel.
    Unused,
    /// The decoded linear-domain spectral envelope of length `n`,
    /// produced by §6.2.3 curve computation from the per-packet
    /// `[amplitude]` and `[coefficients]` LSP filter coefficients.
    Curve(Vec<f32>),
}

/// Errors that can arise while preparing or running a floor 0 decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Floor0Error {
    /// A `floor0_book_list` entry indexes outside the codebook table
    /// supplied at construction.
    /// §6.2.1: "any element of the array `[floor0_book_list]` that is
    /// greater than the maximum codebook number for this bitstream is
    /// an error condition that also renders the stream undecodable."
    BookOutOfRange {
        /// Position in `floor0_book_list`.
        position: usize,
        /// The offending codebook index.
        book: u8,
        /// The number of codebooks available.
        codebook_count: usize,
    },
    /// `floor0_book_list` was empty. The setup parser builds
    /// `book_list` from `[floor0_number_of_books] = read 4 bits + 1`
    /// so it cannot be 0 from a real stream, but a hand-built
    /// [`Floor0Header`] could still trip this.
    EmptyBookList,
    /// `floor0_order` was zero. The LSP synthesis loop would otherwise
    /// produce a meaningless empty `[p]`/`[q]` product.
    ZeroOrder,
    /// `floor0_bark_map_size` was zero. The §6.2.3 `π × map[i] /
    /// bark_map_size` angle computation would divide by zero.
    ZeroBarkMapSize,
    /// `floor0_amplitude_bits` was zero — the [amplitude] field would
    /// always read 0 and decode would always return [`Floor0Curve::Unused`].
    /// The spec does not explicitly forbid this but it makes the floor
    /// permanently unused, which is almost certainly a malformed setup;
    /// we surface it as a structured error so callers can decide.
    ZeroAmplitudeBits,
    /// A value codebook referenced from `floor0_book_list` did not have
    /// a VQ lookup table (lookup type 0). §3.3: "requesting decode using
    /// a codebook of lookup type 0 in any context expecting a vector
    /// return value … is an error condition rendering the packet
    /// undecodable."
    ValueBookHasNoLookup {
        /// Position in `floor0_book_list`.
        position: usize,
        /// The offending codebook index.
        book: u8,
    },
    /// Building a value codebook's Huffman tree failed.
    Huffman(BuildError),
}

impl core::fmt::Display for Floor0Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Floor0Error::BookOutOfRange {
                position,
                book,
                codebook_count,
            } => write!(
                f,
                "floor0_book_list[{position}] references codebook {book} but only \
                 {codebook_count} codebooks are configured (§6.2.1)"
            ),
            Floor0Error::EmptyBookList => {
                write!(f, "floor0_book_list is empty (§6.2.1 requires ≥ 1 book)")
            }
            Floor0Error::ZeroOrder => write!(f, "floor0_order must be > 0 (§6.2.3)"),
            Floor0Error::ZeroBarkMapSize => {
                write!(f, "floor0_bark_map_size must be > 0 (§6.2.3)")
            }
            Floor0Error::ZeroAmplitudeBits => {
                write!(
                    f,
                    "floor0_amplitude_bits is zero — floor will always read amplitude = 0 (§6.2.2)"
                )
            }
            Floor0Error::ValueBookHasNoLookup { position, book } => write!(
                f,
                "floor0_book_list[{position}] = {book} has no VQ lookup table (§3.3)"
            ),
            Floor0Error::Huffman(e) => write!(f, "floor0 codebook tree build failed: {e}"),
        }
    }
}

impl std::error::Error for Floor0Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Floor0Error::Huffman(e) => Some(e),
            _ => None,
        }
    }
}

impl From<BuildError> for Floor0Error {
    fn from(value: BuildError) -> Self {
        Floor0Error::Huffman(value)
    }
}

/// A value codebook prepared for runtime VQ decode: its Huffman decision
/// tree plus a clone of the codebook itself (so [`unpack_vector`] can
/// read the multiplicand table during decode).
#[derive(Debug, Clone)]
struct ValueBook {
    tree: HuffmanTree,
    codebook: VorbisCodebook,
}

/// A pre-validated floor 0 decoder built once from a [`Floor0Header`]
/// and the stream's codebook table. Building it up front validates the
/// §6.2.1 undecodability clauses (book indices in range, books carry
/// a VQ lookup, nonzero order / map size / amplitude bits) and
/// pre-builds the value-book Huffman trees so the per-packet decode is
/// allocation-light.
#[derive(Debug, Clone)]
pub struct Floor0Decoder {
    /// `floor0_order` (1..=255).
    order: usize,
    /// `floor0_rate` (Hz).
    rate: u32,
    /// `floor0_bark_map_size` (1..=65535).
    bark_map_size: u32,
    /// `floor0_amplitude_bits` (1..=63).
    amplitude_bits: u8,
    /// `floor0_amplitude_offset`.
    amplitude_offset: u8,
    /// `floor0_book_list` with each entry's value-book resolved into a
    /// (Huffman tree, codebook) pair ready for VQ decode.
    value_books: Vec<ValueBook>,
}

impl Floor0Decoder {
    /// Build a [`Floor0Decoder`] from a parsed [`Floor0Header`] and the
    /// stream's codebook table.
    ///
    /// Validates the §6.2.1 / §6.2.3 undecodability clauses and
    /// pre-builds every referenced value codebook's Huffman decision
    /// tree.
    pub fn new(header: &Floor0Header, codebooks: &[VorbisCodebook]) -> Result<Self, Floor0Error> {
        if header.order == 0 {
            return Err(Floor0Error::ZeroOrder);
        }
        if header.bark_map_size == 0 {
            return Err(Floor0Error::ZeroBarkMapSize);
        }
        if header.amplitude_bits == 0 {
            return Err(Floor0Error::ZeroAmplitudeBits);
        }
        if header.book_list.is_empty() {
            return Err(Floor0Error::EmptyBookList);
        }

        let mut value_books = Vec::with_capacity(header.book_list.len());
        for (position, &book) in header.book_list.iter().enumerate() {
            let cb = codebooks
                .get(book as usize)
                .ok_or(Floor0Error::BookOutOfRange {
                    position,
                    book,
                    codebook_count: codebooks.len(),
                })?;
            if matches!(cb.lookup, VqLookup::None) {
                // §3.3: a VQ-context decode against a lookup_type=0
                // codebook is undecodable.
                return Err(Floor0Error::ValueBookHasNoLookup { position, book });
            }
            let tree = HuffmanTree::from_codebook(cb)?;
            value_books.push(ValueBook {
                tree,
                codebook: cb.clone(),
            });
        }

        Ok(Floor0Decoder {
            order: header.order as usize,
            rate: header.rate as u32,
            bark_map_size: header.bark_map_size as u32,
            amplitude_bits: header.amplitude_bits,
            amplitude_offset: header.amplitude_offset,
            value_books,
        })
    }

    /// `floor0_order`.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Run the §6.2.2 packet decode + §6.2.3 curve computation,
    /// producing a linear-domain spectral envelope of length `n` (or
    /// [`Floor0Curve::Unused`]).
    ///
    /// `n` is the number of frequency bins to render (`blocksize/2`),
    /// supplied by the §4.3 audio-packet driver.
    pub fn decode(&self, reader: &mut BitReaderLsb<'_>, n: usize) -> Floor0Curve {
        match self.packet_decode(reader) {
            None => Floor0Curve::Unused,
            Some((amplitude, coefficients)) => {
                Floor0Curve::Curve(self.curve_computation(amplitude, &coefficients, n))
            }
        }
    }

    /// §6.2.2 packet decode. Returns `Some((amplitude, coefficients))`
    /// on a nonzero-amplitude success, or `None` if the amplitude read
    /// zero or any end-of-packet occurred (both map to 'unused' status
    /// per §6.2.2's nominal-occurrence clause).
    ///
    /// The returned `coefficients` vector may be longer than
    /// `floor0_order`; the spec explicitly permits this and instructs
    /// the curve-computation step to ignore trailing scalars.
    fn packet_decode(&self, reader: &mut BitReaderLsb<'_>) -> Option<(u32, Vec<f32>)> {
        // §6.2.2 step 1: [amplitude] = read [floor0_amplitude_bits] bits.
        let amplitude = reader.read_u32(self.amplitude_bits as u32).ok()?;
        // §6.2.2 step 2: if [amplitude] > 0 (else 'unused', return None).
        if amplitude == 0 {
            return None;
        }

        // §6.2.2 step 3: [coefficients] = empty.
        let mut coefficients: Vec<f32> = Vec::with_capacity(self.order);

        // §6.2.2 step 4: [booknumber] = read ilog([floor0_number_of_books]) bits.
        // Per the note: "the book number used for decode can, in fact,
        // be stored in the bitstream in ilog([floor0_number_of_books] - 1)
        // bits. Nevertheless, the above specification is correct and
        // values greater than the maximum possible book value are reserved."
        // We implement the spec literally: `ilog(number_of_books)`.
        let book_index_bits = ilog(self.value_books.len() as u32);
        let booknumber = reader.read_u32(book_index_bits).ok()?;
        // §6.2.2 step 5: booknumber > highest book → undecodable.
        // (As a packet-decode condition the §6.2.2 closing note treats
        // every read failure as the nominal 'unused'; an out-of-range
        // booknumber lands here because the reserved values do not
        // correspond to any value book. Map it to 'unused' rather than
        // erroring, consistent with the spec's nominal-occurrence rule.)
        if (booknumber as usize) >= self.value_books.len() {
            return None;
        }

        let book = &self.value_books[booknumber as usize];
        let dimensions = book.codebook.dimensions as usize;
        if dimensions == 0 {
            // Defensive: a 0-dim codebook would loop forever.
            return None;
        }

        // §6.2.2 step 6: [last] = 0.
        let mut last: f32 = 0.0;
        // §6.2.2 steps 7..11: read vectors until len(coefficients) >= order.
        while coefficients.len() < self.order {
            // §6.2.2 step 7: decode a VQ entry index from the bitstream
            // using the chosen value book, then unpack it into a vector.
            let entry = book.tree.decode_entry(reader).ok()?;
            let mut temp_vector = unpack_vector(&book.codebook, entry).ok()?;
            // §6.2.2 step 8: add [last] to each scalar in [temp_vector].
            // (sequence_p is handled inside unpack_vector for the
            // running prefix within the vector; the §6.2.2 [last]
            // accumulator is *across* vectors and is separate.)
            for v in &mut temp_vector {
                *v += last;
            }
            // §6.2.2 step 9: [last] = value of the last scalar in [temp_vector].
            // (temp_vector has length = dimensions, guaranteed > 0 above.)
            last = *temp_vector.last().unwrap();
            // §6.2.2 step 10: concatenate [temp_vector] onto [coefficients].
            coefficients.extend_from_slice(&temp_vector);
            // §6.2.2 step 11: if (len < order) continue; else fall out.
            // The note: "extra values are not used and may be ignored or
            // discarded" — we keep them in the returned vector but
            // curve_computation only reads the first `order` of them.
            let _ = dimensions; // silence unused-variable warning in some cfgs
        }

        // §6.2.2 step 12: done.
        Some((amplitude, coefficients))
    }

    /// §6.2.3 curve computation. Given the per-packet `amplitude` and
    /// `coefficients` (LSP filter coefficients, of length ≥
    /// `floor0_order`), produce a length-`n` linear-domain envelope.
    fn curve_computation(&self, amplitude: u32, coefficients: &[f32], n: usize) -> Vec<f32> {
        // The amplitude = 0 case is already handled by the packet-decode
        // step returning None, but §6.2.3 also restates it for direct
        // curve callers: amplitude = 0 ⇒ length-n all-zero vector.
        if amplitude == 0 {
            return vec![0.0; n];
        }

        // §6.2.3 Bark map computation:
        //   foobar(i) = floor(bark(rate*i/(2n)) * bark_map_size / bark(.5*rate))
        //   map[i]    = min(bark_map_size - 1, foobar(i))  for i in [0, n-1]
        //   map[n]    = -1
        let bark_denominator = bark((0.5 * self.rate as f64) as f32);
        let mut map: Vec<i32> = Vec::with_capacity(n + 1);
        for i in 0..n {
            // (rate * i) / (2 * n) using f64 to avoid early overflow at
            // typical rates.
            let f = (self.rate as f64 * i as f64) / (2.0 * n as f64);
            let foobar =
                (bark(f as f32) * self.bark_map_size as f32 / bark_denominator).floor() as i32;
            let capped = foobar.min(self.bark_map_size as i32 - 1);
            map.push(capped);
        }
        map.push(-1);

        // The LSP product loop reads at most `order` coefficients; the
        // spec permits the packet decode to over-read, so slice here.
        let coeffs = &coefficients[..self.order];

        // §6.2.3 main synthesis loop.
        let order = self.order;
        let mut output = vec![0.0f32; n];

        // step 1: [i] = 0
        let mut i: usize = 0;
        while i < n {
            // step 2: [ω] = π * map[i] / bark_map_size
            let omega = std::f32::consts::PI * map[i] as f32 / self.bark_map_size as f32;
            let cos_omega = omega.cos();

            // steps 3..4: compute [p] and [q] (order-parity dependent).
            let (p, q) = if order % 2 == 1 {
                // p = (1 - cos^2 ω) * Π_{j=0..(order-3)/2} 4(cos(c[2j+1]) - cos ω)^2
                // q = 0.25         * Π_{j=0..(order-1)/2} 4(cos(c[2j])   - cos ω)^2
                let mut p_prod = 1.0f32 - cos_omega * cos_omega;
                let p_iters = (order - 1) / 2; // j = 0..=(order-3)/2 inclusive
                for j in 0..p_iters {
                    let c = coeffs[2 * j + 1].cos();
                    let term = c - cos_omega;
                    p_prod *= 4.0 * term * term;
                }
                let mut q_prod = 0.25f32;
                let q_iters = order / 2 + 1; // j = 0..=(order-1)/2 inclusive
                for j in 0..q_iters {
                    let c = coeffs[2 * j].cos();
                    let term = c - cos_omega;
                    q_prod *= 4.0 * term * term;
                }
                (p_prod, q_prod)
            } else {
                // p = ((1 - cos ω)/2) * Π_{j=0..(order-2)/2} 4(cos(c[2j+1]) - cos ω)^2
                // q = ((1 + cos ω)/2) * Π_{j=0..(order-2)/2} 4(cos(c[2j])   - cos ω)^2
                let mut p_prod = (1.0f32 - cos_omega) / 2.0;
                let mut q_prod = (1.0f32 + cos_omega) / 2.0;
                let iters = order / 2; // j = 0..=(order-2)/2 inclusive
                for j in 0..iters {
                    let c_odd = coeffs[2 * j + 1].cos();
                    let term_odd = c_odd - cos_omega;
                    p_prod *= 4.0 * term_odd * term_odd;
                    let c_even = coeffs[2 * j].cos();
                    let term_even = c_even - cos_omega;
                    q_prod *= 4.0 * term_even * term_even;
                }
                (p_prod, q_prod)
            };

            // step 4: [linear_floor_value] = exp(.11512925 * (a*offset /
            //         ((2^bits - 1) * sqrt(p+q)) - offset))
            let pq = p + q;
            // sqrt(p+q) cannot be negative; if pq is 0 (an LSP pole on
            // the unit circle exactly at ω) the log diverges. Clamp at
            // a tiny positive value to stay finite — this is a runtime
            // safety net for pathological setups, not a spec deviation.
            let sqrt_pq = pq.max(0.0).sqrt().max(f32::MIN_POSITIVE);
            let bits = self.amplitude_bits as u32;
            let denom_int: u32 = if bits >= 32 {
                u32::MAX
            } else {
                (1u32 << bits) - 1
            };
            let denom = denom_int as f32 * sqrt_pq;
            let offset = self.amplitude_offset as f32;
            let exponent = 0.115_129_25_f32 * (amplitude as f32 * offset / denom - offset);
            let linear_floor_value = exponent.exp();

            // step 5: [iteration_condition] = map[i]
            let iteration_condition = map[i];
            // step 6: output[i] = linear_floor_value
            output[i] = linear_floor_value;
            // step 7: increment [i]
            i += 1;
            // step 8: while map[i] == iteration_condition, continue at step 5.
            // (Implemented as an inner replication loop; map has n+1
            // entries with map[n] = -1 as the sentinel.)
            while i < n && map[i] == iteration_condition {
                output[i] = linear_floor_value;
                i += 1;
            }
            // step 9: if (i < n) loop back to step 2 (outer while handles this).
        }

        output
    }
}

/// §6.2.3 Bark scale formula (post-errata 20150227):
///   bark(x) = 13.1 * atan(.00074 * x) + 2.24 * atan(.0000000185 * x^2) + .0001 * x
///
/// The errata corrects a typesetting bug in earlier prints of the spec
/// where the trailing `.0001x` term was inside the `atan(…)` argument.
pub fn bark(x: f32) -> f32 {
    13.1f32 * (0.000_74_f32 * x).atan()
        + 2.24f32 * (0.000_000_018_5_f32 * x * x).atan()
        + 0.000_1_f32 * x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::{VorbisCodebook, VqLookup};
    use oxideav_core::bits::{BitReaderLsb, BitWriterLsb};

    // ---- helper: Bark formula sanity ----

    #[test]
    fn bark_matches_spec_at_zero() {
        // At x = 0 every term vanishes.
        assert_eq!(bark(0.0), 0.0);
    }

    #[test]
    fn bark_is_monotonic_increasing_on_audio_range() {
        // The Bark scale must be monotonic on the audible range
        // (otherwise the floor0 map computation would multi-value).
        let mut prev = bark(0.0);
        let mut x = 1.0f32;
        while x <= 24_000.0 {
            let cur = bark(x);
            assert!(
                cur > prev,
                "bark({x}) = {cur} should be > bark(prev) = {prev}"
            );
            prev = cur;
            x *= 1.5;
        }
    }

    // ---- value-codebook fixtures ----

    /// One-entry codebook with VQ lookup type 2 returning a single
    /// scalar vector `[value]` of dimension 1. Useful to drive a
    /// deterministic LSP coefficient stream.
    fn one_entry_value_book(value: f32) -> VorbisCodebook {
        VorbisCodebook {
            dimensions: 1,
            entries: 1,
            // Length 1: the canonical Huffman tree builder treats this
            // as the spec errata 20150226 single-entry case (decode
            // sinks one bit and returns entry 0 for either '0' or '1').
            codeword_lengths: vec![1],
            lookup: VqLookup::Tessellation {
                minimum_value: value,
                delta_value: 0.0,
                value_bits: 1,
                sequence_p: false,
                multiplicands: vec![1], // multiplicand * delta + min = value
            },
        }
    }

    /// Two-entry, dimension-2 codebook returning either [v0, v1] (entry 0)
    /// or [v1, v0] (entry 1).
    fn two_entry_dim2_book(v0: f32, v1: f32) -> VorbisCodebook {
        // Tessellation: entry e covers multiplicands[e*2 .. e*2+2].
        // Each scalar = multiplicand * delta + minimum.
        // We pick minimum=0, delta=1 so the multiplicand IS the value.
        // Rust f32 -> u32 conversion needs the value to be a whole
        // small integer; use small distinct integer values in tests.
        let m0 = v0 as u32;
        let m1 = v1 as u32;
        VorbisCodebook {
            dimensions: 2,
            entries: 2,
            codeword_lengths: vec![1, 1],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 4,
                sequence_p: false,
                multiplicands: vec![m0, m1, m1, m0],
            },
        }
    }

    fn minimal_header(book_list: Vec<u8>) -> Floor0Header {
        Floor0Header {
            order: 4,
            rate: 44_100,
            bark_map_size: 256,
            amplitude_bits: 6,
            amplitude_offset: 100,
            book_list,
        }
    }

    // ---- construction validation ----

    #[test]
    fn new_rejects_zero_order() {
        let mut h = minimal_header(vec![0]);
        h.order = 0;
        let err = Floor0Decoder::new(&h, &[one_entry_value_book(1.0)]).unwrap_err();
        assert_eq!(err, Floor0Error::ZeroOrder);
    }

    #[test]
    fn new_rejects_zero_bark_map_size() {
        let mut h = minimal_header(vec![0]);
        h.bark_map_size = 0;
        let err = Floor0Decoder::new(&h, &[one_entry_value_book(1.0)]).unwrap_err();
        assert_eq!(err, Floor0Error::ZeroBarkMapSize);
    }

    #[test]
    fn new_rejects_zero_amplitude_bits() {
        let mut h = minimal_header(vec![0]);
        h.amplitude_bits = 0;
        let err = Floor0Decoder::new(&h, &[one_entry_value_book(1.0)]).unwrap_err();
        assert_eq!(err, Floor0Error::ZeroAmplitudeBits);
    }

    #[test]
    fn new_rejects_empty_book_list() {
        let h = minimal_header(vec![]);
        let err = Floor0Decoder::new(&h, &[one_entry_value_book(1.0)]).unwrap_err();
        assert_eq!(err, Floor0Error::EmptyBookList);
    }

    #[test]
    fn new_rejects_out_of_range_book() {
        let h = minimal_header(vec![5]); // only 1 codebook supplied
        let err = Floor0Decoder::new(&h, &[one_entry_value_book(1.0)]).unwrap_err();
        assert_eq!(
            err,
            Floor0Error::BookOutOfRange {
                position: 0,
                book: 5,
                codebook_count: 1,
            }
        );
    }

    #[test]
    fn new_rejects_lookup_type_0_value_book() {
        let h = minimal_header(vec![0]);
        let scalar = VorbisCodebook {
            dimensions: 1,
            entries: 2,
            codeword_lengths: vec![1, 1],
            lookup: VqLookup::None,
        };
        let err = Floor0Decoder::new(&h, &[scalar]).unwrap_err();
        assert_eq!(
            err,
            Floor0Error::ValueBookHasNoLookup {
                position: 0,
                book: 0,
            }
        );
    }

    // ---- packet decode (§6.2.2) ----

    #[test]
    fn decode_returns_unused_for_zero_amplitude() {
        let header = minimal_header(vec![0]);
        let dec = Floor0Decoder::new(&header, &[one_entry_value_book(0.5)]).unwrap();
        // 6 amplitude bits all zero → unused; no further bits read.
        let packet = vec![0u8; 4];
        let mut r = BitReaderLsb::new(&packet);
        assert_eq!(dec.decode(&mut r, 32), Floor0Curve::Unused);
    }

    #[test]
    fn decode_returns_unused_on_eof_during_amplitude() {
        let header = minimal_header(vec![0]);
        let dec = Floor0Decoder::new(&header, &[one_entry_value_book(0.5)]).unwrap();
        // Empty packet — the very first read_u32(6) fails.
        let packet: Vec<u8> = vec![];
        let mut r = BitReaderLsb::new(&packet);
        assert_eq!(dec.decode(&mut r, 32), Floor0Curve::Unused);
    }

    #[test]
    fn decode_returns_unused_on_eof_during_coefficients() {
        // header has order 4, book 0 is dim-1 → 4 codewords expected.
        let header = minimal_header(vec![0]);
        let dec = Floor0Decoder::new(&header, &[one_entry_value_book(0.5)]).unwrap();
        // Only 1 byte: 6 amplitude bits = 0b000001 (=1), then 0 bits
        // for booknumber (ilog(1) = 1 bit → reads 0), then ran out.
        let packet = vec![0b0000_0001u8];
        let mut r = BitReaderLsb::new(&packet);
        // Single-entry length-1 codebook: decode_entry sinks one bit
        // and returns entry 0. We have 8 bits in the packet, of which
        // 6 are consumed by amplitude and 1 by booknumber → 1 bit left.
        // Reading the first VQ codeword sinks that last bit and
        // returns entry 0, decoding to [0.5]. Now coefficients.len()=1
        // < order=4 so loop tries to decode again → EOF → Unused.
        assert_eq!(dec.decode(&mut r, 8), Floor0Curve::Unused);
    }

    #[test]
    fn packet_decode_produces_expected_amplitude_and_coefficients() {
        // amplitude_bits=6, book_list=[0], order=4, book 0 is dim-2
        // entries=2 length-1 codes. Entry 0 → [3, 7]; entry 1 → [7, 3].
        // Sequence: read amplitude=42, booknumber=0 (ilog(1)=1 bit, =0),
        // then 2 vectors (since 2 vectors of dim 2 = 4 elements ≥ order 4).
        // [last] starts at 0. After v0=[3,7]: temp=[3,7], last=7, coeffs=[3,7].
        // Next codeword=1 → [7,3], plus [last]=7 → [14,10], last=10,
        // coeffs=[3,7,14,10].
        let header = Floor0Header {
            order: 4,
            rate: 44_100,
            bark_map_size: 256,
            amplitude_bits: 6,
            amplitude_offset: 100,
            book_list: vec![0],
        };
        let dec = Floor0Decoder::new(&header, &[two_entry_dim2_book(3.0, 7.0)]).unwrap();

        let mut w = BitWriterLsb::with_capacity(8);
        w.write_u32(42, 6); // amplitude
        w.write_u32(0, 1); // booknumber (ilog(1) = 1)
        w.write_bit(false); // entry 0 (length-1 code)
        w.write_bit(true); // entry 1 (length-1 code)
        let packet = w.finish();

        let mut r = BitReaderLsb::new(&packet);
        let (amplitude, coeffs) = dec.packet_decode(&mut r).expect("decode should succeed");
        assert_eq!(amplitude, 42);
        assert_eq!(coeffs, vec![3.0, 7.0, 14.0, 10.0]);
    }

    #[test]
    fn packet_decode_loops_until_order_reached_with_dim_1_book() {
        // order=4, dim-1 book — needs 4 codewords.
        let header = minimal_header(vec![0]);
        let dec = Floor0Decoder::new(&header, &[one_entry_value_book(2.5)]).unwrap();

        let mut w = BitWriterLsb::with_capacity(8);
        w.write_u32(15, 6); // nonzero amplitude
        w.write_u32(0, 1); // booknumber 0
        for _ in 0..4 {
            w.write_bit(false); // 4 entry-0 codewords
        }
        let packet = w.finish();

        let mut r = BitReaderLsb::new(&packet);
        let (amplitude, coeffs) = dec.packet_decode(&mut r).unwrap();
        assert_eq!(amplitude, 15);
        // Each successive vector adds [last] = previous tail. The book
        // always returns [2.5]; last accumulates: 2.5, 5.0, 7.5, 10.0.
        assert_eq!(coeffs, vec![2.5, 5.0, 7.5, 10.0]);
    }

    // ---- curve computation (§6.2.3) ----

    #[test]
    fn curve_computation_amplitude_zero_returns_all_zero() {
        let header = minimal_header(vec![0]);
        let dec = Floor0Decoder::new(&header, &[one_entry_value_book(0.5)]).unwrap();
        // Even though packet_decode returns None for amplitude=0, the
        // standalone curve_computation honours the same rule (§6.2.3
        // first paragraph: "If the value [amplitude] is zero, the return
        // value is a length [n] vector with all-zero scalars").
        let curve = dec.curve_computation(0, &[0.0; 4], 32);
        assert_eq!(curve, vec![0.0; 32]);
    }

    #[test]
    fn curve_computation_produces_length_n_finite_vector() {
        // Drive the LSP curve with hand-picked coefficient values and
        // verify (a) the length is exactly n, (b) every element is
        // finite and non-negative.
        let header = Floor0Header {
            order: 4,
            rate: 44_100,
            bark_map_size: 256,
            amplitude_bits: 6,
            amplitude_offset: 100,
            book_list: vec![0],
        };
        let dec = Floor0Decoder::new(&header, &[one_entry_value_book(1.0)]).unwrap();
        // LSP coefficients spread across the unit circle.
        let coeffs = [0.5, 1.0, 1.5, 2.0];
        let n = 64;
        let curve = dec.curve_computation(30, &coeffs, n);
        assert_eq!(curve.len(), n);
        for (i, &v) in curve.iter().enumerate() {
            assert!(v.is_finite(), "curve[{i}] = {v} should be finite");
            assert!(v >= 0.0, "curve[{i}] = {v} should be non-negative");
        }
    }

    #[test]
    fn curve_computation_replicates_value_within_iteration_condition() {
        // The §6.2.3 step 8 chaining replicates linear_floor_value for
        // every successive output bin whose map[i] matches the current
        // synthesis bin. With a very small bark_map_size (relative to
        // the rate × n window) many consecutive bins should share a
        // map value → many consecutive output bins should be exactly
        // equal.
        let header = Floor0Header {
            order: 4,
            rate: 44_100,
            bark_map_size: 8, // tiny — most bins map to the same Bark bin
            amplitude_bits: 6,
            amplitude_offset: 100,
            book_list: vec![0],
        };
        let dec = Floor0Decoder::new(&header, &[one_entry_value_book(1.0)]).unwrap();
        let coeffs = [0.3, 0.6, 0.9, 1.2];
        let n = 64;
        let curve = dec.curve_computation(20, &coeffs, n);
        // Find any two adjacent equal samples — must exist given the
        // mismatched n=64 vs bark_map_size=8 setup.
        let mut found_replication = false;
        for w in curve.windows(2) {
            if w[0] == w[1] {
                found_replication = true;
                break;
            }
        }
        assert!(
            found_replication,
            "expected at least one pair of replicated samples under iteration_condition chaining"
        );
    }

    // ---- end-to-end ----

    #[test]
    fn end_to_end_packet_to_curve_round_trip() {
        // Tie packet_decode + curve_computation together via the public
        // `decode` entry point. We feed a non-trivial packet and verify
        // we get a finite, length-n Curve back (not Unused).
        let header = Floor0Header {
            order: 4,
            rate: 44_100,
            bark_map_size: 256,
            amplitude_bits: 6,
            amplitude_offset: 100,
            book_list: vec![0],
        };
        let dec = Floor0Decoder::new(&header, &[two_entry_dim2_book(2.0, 5.0)]).unwrap();

        let mut w = BitWriterLsb::with_capacity(8);
        w.write_u32(25, 6); // amplitude
        w.write_u32(0, 1); // booknumber (ilog(1) = 1)
        w.write_bit(false); // entry 0 → [2, 5]
        w.write_bit(false); // entry 0 → [2, 5], + last=5 → [7, 10]
        let packet = w.finish();

        let mut r = BitReaderLsb::new(&packet);
        let n = 32;
        match dec.decode(&mut r, n) {
            Floor0Curve::Unused => panic!("expected a Curve, got Unused"),
            Floor0Curve::Curve(curve) => {
                assert_eq!(curve.len(), n);
                for (i, &v) in curve.iter().enumerate() {
                    assert!(v.is_finite(), "curve[{i}] = {v}");
                    assert!(v >= 0.0, "curve[{i}] = {v}");
                }
            }
        }
    }

    #[test]
    fn booknumber_out_of_range_maps_to_unused() {
        // book_list has 3 entries → booknumber field is ilog(3) = 2 bits,
        // allowing 0..=3. Booknumber 3 is a reserved value (only 3 books
        // present at indices 0..=2). Per the §6.2.2 nominal-occurrence
        // closing note, we surface it as 'unused'.
        let header = Floor0Header {
            order: 4,
            rate: 44_100,
            bark_map_size: 256,
            amplitude_bits: 6,
            amplitude_offset: 100,
            book_list: vec![0, 0, 0],
        };
        let dec = Floor0Decoder::new(
            &header,
            &[
                one_entry_value_book(1.0),
                one_entry_value_book(1.0),
                one_entry_value_book(1.0),
            ],
        )
        .unwrap();

        let mut w = BitWriterLsb::with_capacity(2);
        w.write_u32(7, 6); // amplitude > 0
        w.write_u32(3, 2); // booknumber = 3 (reserved)
        let packet = w.finish();
        let mut r = BitReaderLsb::new(&packet);
        assert_eq!(dec.decode(&mut r, 16), Floor0Curve::Unused);
    }
}

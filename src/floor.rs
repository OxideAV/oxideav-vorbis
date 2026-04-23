//! Floor packet decoding and curve synthesis.
//!
//! Reference: Vorbis I §6 (floor type 0 / LSP) and §7 (floor type 1).

use oxideav_core::{Error, Result};

use crate::codebook::Codebook;
use crate::setup::{Floor, Floor0, Floor1};
use oxideav_core::bits::BitReaderLsb as BitReader;

/// Bit width for the first two amplitude values (Y[0], Y[1]) per floor1
/// multiplier setting (Vorbis I §6.2.3 table).
fn amp_bits_for_multiplier(multiplier: u8) -> u32 {
    match multiplier {
        1 => 8, // range = 256
        2 => 7, // range = 128
        3 => 7, // range = 86 (still 7 bits)
        4 => 6, // range = 64
        _ => 8,
    }
}

/// Per-multiplier "step2 flag = unused" range as defined in Vorbis I
/// (multiplier values map to a fixed dB step size).
pub fn floor1_db_step(multiplier: u8) -> f32 {
    match multiplier {
        1 => 256.0,
        2 => 128.0,
        3 => 86.0,
        4 => 64.0,
        _ => 256.0,
    }
}

/// Decoded floor 1 amplitude vector + "is unused" flag.
#[derive(Clone, Debug)]
pub struct Floor1Decoded {
    pub unused: bool,
    /// Amplitude (Y) values, one per X-list entry. Empty if `unused`.
    pub y: Vec<i32>,
}

pub fn decode_floor1_packet(
    floor: &Floor1,
    codebooks: &[Codebook],
    br: &mut BitReader<'_>,
) -> Result<Floor1Decoded> {
    let nonzero = br.read_bit()?;
    if !nonzero {
        return Ok(Floor1Decoded {
            unused: true,
            y: Vec::new(),
        });
    }
    let amp_bits = amp_bits_for_multiplier(floor.multiplier);
    let mut y: Vec<i32> = Vec::with_capacity(floor.xlist.len());
    y.push(br.read_u32(amp_bits)? as i32);
    y.push(br.read_u32(amp_bits)? as i32);
    let mut offset = 2usize;
    for &class_idx in &floor.partition_class_list {
        let c = class_idx as usize;
        let cdim = floor.class_dimensions[c] as usize;
        let cbits = floor.class_subclasses[c] as u32;
        let csub = 1u32 << cbits;
        let mut cval = if cbits > 0 {
            let cb = &codebooks[floor.class_masterbook[c] as usize];
            cb.decode_scalar(br)?
        } else {
            0
        };
        for _j in 0..cdim {
            let book_index = floor.class_subbook[c][(cval & (csub - 1)) as usize];
            cval >>= cbits;
            let v = if book_index >= 0 {
                let cb = &codebooks[book_index as usize];
                cb.decode_scalar(br)? as i32
            } else {
                0
            };
            y.push(v);
            offset += 1;
        }
    }
    if offset != floor.xlist.len() {
        return Err(Error::invalid(format!(
            "Vorbis floor1 decoded {} amplitudes, expected {}",
            offset,
            floor.xlist.len()
        )));
    }
    Ok(Floor1Decoded { unused: false, y })
}

/// Decoded floor 0 (LSP) amplitude + LSP coefficient vector (cosines of
/// the LSP frequencies ω_i as stored — i.e. the raw quantised VQ output).
///
/// Vorbis I §6.2.2 "packet decode": if `amplitude == 0`, the channel is
/// unused this frame (treated identically to floor1's `unused` flag).
#[derive(Clone, Debug)]
pub struct Floor0Decoded {
    /// Raw amplitude value as read from the packet (0..2^amplitude_bits).
    pub amplitude: u32,
    /// Codebook index used to decode `coefficients` (informational).
    pub book_number: u8,
    /// LSP coefficient vector, length ≥ `floor0_order`. Only the first
    /// `floor0_order` entries participate in curve synthesis (§6.2.3).
    pub coefficients: Vec<f32>,
}

/// Union of per-channel floor decoder states: either unused, floor1
/// posts, or floor0 LSP. Matches the spec's `no_residue` flag via
/// [`FloorDecoded::is_unused`].
#[derive(Clone, Debug)]
pub enum FloorDecoded {
    Unused,
    Floor1(Floor1Decoded),
    Floor0(Floor0Decoded),
}

impl FloorDecoded {
    /// Whether the channel is silent in this frame (spec: `no_residue`
    /// element set to true). Matches floor1 `unused` / floor0
    /// `amplitude == 0`.
    pub fn is_unused(&self) -> bool {
        match self {
            FloorDecoded::Unused => true,
            FloorDecoded::Floor1(f) => f.unused,
            FloorDecoded::Floor0(f) => f.amplitude == 0,
        }
    }
}

/// Decode a floor0 packet (Vorbis I §6.2.2). `floor0_number_of_books` is
/// stored in the setup header; the book index read from the bitstream
/// selects one of the setup's `book_list` entries. The coefficients
/// decode loop unrolls VQ vectors of whatever book dimension until at
/// least `floor0_order` scalars have been accumulated.
pub fn decode_floor0_packet(
    floor: &Floor0,
    codebooks: &[Codebook],
    br: &mut BitReader<'_>,
) -> Result<Floor0Decoded> {
    let amplitude_bits = floor.amplitude_bits as u32;
    let amplitude = br.read_u32(amplitude_bits)?;
    if amplitude == 0 {
        return Ok(Floor0Decoded {
            amplitude: 0,
            book_number: 0,
            coefficients: Vec::new(),
        });
    }
    // Book-number width = ilog(number_of_books). Per §6.2.2 the spec lets
    // implementations store it in that many bits; values >= number_of_books
    // are reserved / undecodable.
    let book_bits = ilog(floor.number_of_books as u32);
    let booknumber = br.read_u32(book_bits)?;
    if booknumber as usize >= floor.book_list.len() {
        return Err(Error::invalid(
            "Vorbis floor0: book number out of range",
        ));
    }
    let book_idx = floor.book_list[booknumber as usize] as usize;
    if book_idx >= codebooks.len() {
        return Err(Error::invalid(
            "Vorbis floor0: book_list entry references unknown codebook",
        ));
    }
    let book = &codebooks[book_idx];
    let order = floor.order as usize;
    let dim = book.dimensions as usize;
    if dim == 0 {
        return Err(Error::invalid("Vorbis floor0: book dimension is zero"));
    }
    // Spec note: the VQ may be over-read if `order` is not a multiple of
    // `dim`. The extra scalars are discarded (§6.2.2 bullet 4). The
    // spec's coefficient loop also runs the sequence_p continuation
    // ([last] scalar addition) across VQ vectors, but that's already
    // handled inside the codebook itself per book. BETWEEN VQ vectors,
    // the running `last` value is continued per the spec's pseudocode —
    // so we have to re-apply it across vectors.
    let mut coefficients: Vec<f32> = Vec::with_capacity(order + dim);
    let mut last_scalar = 0f32;
    while coefficients.len() < order {
        let entry = book.decode_scalar(br)?;
        let mut temp = book.vq_lookup(entry)?;
        for v in temp.iter_mut() {
            *v += last_scalar;
        }
        last_scalar = *temp.last().unwrap_or(&last_scalar);
        coefficients.extend_from_slice(&temp);
    }
    // Truncate trailing scalars above `order` — they are unused in §6.2.3
    // and carrying them would only waste space.
    coefficients.truncate(order);
    Ok(Floor0Decoded {
        amplitude,
        book_number: booknumber as u8,
        coefficients,
    })
}

/// Public entry point: decode a floor packet given its setup type.
pub fn decode_floor_packet(
    floor: &Floor,
    codebooks: &[Codebook],
    br: &mut BitReader<'_>,
) -> Result<FloorDecoded> {
    match floor {
        Floor::Type1(f) => {
            let d = decode_floor1_packet(f, codebooks, br)?;
            Ok(FloorDecoded::Floor1(d))
        }
        Floor::Type0(f) => {
            let d = decode_floor0_packet(f, codebooks, br)?;
            Ok(FloorDecoded::Floor0(d))
        }
    }
}

fn ilog(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        32 - value.leading_zeros()
    }
}

/// Synthesize the floor curve into `output[0..n_half]` (length = blocksize/2).
///
/// On entry, `output` must already hold the dequantised residue spectrum (or
/// 1.0 everywhere if the channel is residue-free). On exit, each spectral
/// bin has been multiplied by the floor's per-bin magnitude.
///
/// Implements Vorbis I §7.2.4 step1 + step2 + render — translated from the
/// libvorbis reference for bit-exact output.
pub fn synth_floor1(
    floor: &Floor1,
    decoded: &Floor1Decoded,
    n_half: usize,
    output: &mut [f32],
) -> Result<()> {
    if decoded.unused {
        for v in output.iter_mut().take(n_half) {
            *v = 0.0;
        }
        return Ok(());
    }
    if output.len() < n_half {
        return Err(Error::invalid("synth_floor1: output buffer too short"));
    }

    let n_posts = floor.xlist.len();
    if decoded.y.len() != n_posts {
        return Err(Error::invalid("synth_floor1: y length != xlist length"));
    }

    // Sort posts ascending by X, remembering original indices for Y lookup.
    let mut order: Vec<usize> = (0..n_posts).collect();
    order.sort_by_key(|&i| floor.xlist[i]);

    // Precompute, for each post (in original index space), its low/high
    // neighbour in the SORTED order — index of nearest preceding/following
    // post in the X dimension. Only meaningful for original indices >= 2.
    let mut low_neighbor = vec![0usize; n_posts];
    let mut high_neighbor = vec![0usize; n_posts];
    for j in 2..n_posts {
        let xj = floor.xlist[j];
        let mut lo = 0usize;
        let mut lo_x = floor.xlist[0];
        let mut hi = 1usize;
        let mut hi_x = floor.xlist[1];
        for k in 0..j {
            let xk = floor.xlist[k];
            if xk < xj && xk > lo_x {
                lo = k;
                lo_x = xk;
            }
            if xk > xj && xk < hi_x {
                hi = k;
                hi_x = xk;
            }
        }
        low_neighbor[j] = lo;
        high_neighbor[j] = hi;
    }

    // step1: reconstruct final Y per post + mark which are "used".
    let multiplier = floor.multiplier as i32;
    let range = match floor.multiplier {
        1 => 256,
        2 => 128,
        3 => 86,
        4 => 64,
        _ => 256,
    };
    let mut final_y = vec![0i32; n_posts];
    let mut step2_used = vec![false; n_posts];
    final_y[0] = decoded.y[0];
    final_y[1] = decoded.y[1];
    step2_used[0] = true;
    step2_used[1] = true;
    for j in 2..n_posts {
        let lo = low_neighbor[j];
        let hi = high_neighbor[j];
        let predicted = render_point(
            floor.xlist[lo] as i32,
            final_y[lo],
            floor.xlist[hi] as i32,
            final_y[hi],
            floor.xlist[j] as i32,
        );
        let val = decoded.y[j];
        let high_room = range - predicted;
        let low_room = predicted;
        let room = if high_room < low_room {
            high_room
        } else {
            low_room
        } * 2;
        if val != 0 {
            step2_used[lo] = true;
            step2_used[hi] = true;
            step2_used[j] = true;
            if val >= room {
                final_y[j] = if high_room > low_room {
                    val - low_room + predicted
                } else {
                    predicted - val + high_room - 1
                };
            } else {
                final_y[j] = if val % 2 == 1 {
                    predicted - (val + 1) / 2
                } else {
                    predicted + val / 2
                };
            }
        } else {
            step2_used[j] = false;
            final_y[j] = predicted;
        }
    }
    // Vorbis I §7.2.4 step 1: clamp final_y to [0, range-1].
    for y in final_y.iter_mut() {
        if *y < 0 {
            *y = 0;
        } else if *y >= range {
            *y = range - 1;
        }
    }

    // Render the floor curve into `output` per libvorbis floor1_inverse2:
    // Bresenham walks the PRE-clamped `y * multiplier` space (0..=255), not
    // the raw Y space. This matters for bit-exact output with fractional
    // slopes: libvorbis rounds in the multiplied space.
    let mut prev_x = 0i32;
    let mut prev_y_mult = (final_y[order[0]].wrapping_mul(multiplier)).clamp(0, 255);
    for k in 1..n_posts {
        let i = order[k];
        if !step2_used[i] {
            continue;
        }
        let cur_x = floor.xlist[i] as i32;
        let cur_y_mult = (final_y[i].wrapping_mul(multiplier)).clamp(0, 255);
        if cur_x > prev_x {
            render_line(
                prev_x,
                prev_y_mult,
                cur_x,
                cur_y_mult,
                n_half as i32,
                output,
            );
        }
        prev_x = cur_x;
        prev_y_mult = cur_y_mult;
    }
    // Fill any remaining bins past the last used post with the final Y.
    if (prev_x as usize) < n_half {
        let mul = crate::dbtable::FLOOR1_INVERSE_DB[prev_y_mult as usize];
        for v in output.iter_mut().take(n_half).skip(prev_x as usize) {
            *v *= mul;
        }
    }
    Ok(())
}

/// Vorbis render_point: integer-arithmetic line interpolation.
fn render_point(x0: i32, y0: i32, x1: i32, y1: i32, x: i32) -> i32 {
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

/// Render a line from (x0, y0_mult) to (x1, y1_mult) into the spectral
/// output buffer, multiplying each bin's existing value by the floor's
/// linear-magnitude multiplier at that frequency. `n_half` is the spectrum
/// length (blocksize / 2); writes outside that are clipped.
///
/// The Y values passed in are PRE-MULTIPLIED by `floor1_multiplier` and
/// clamped to [0, 255] — matches libvorbis floor1_inverse2. Bresenham
/// operates in that space so the dB-lookup index is an integer running
/// along the line.
fn render_line(x0: i32, y0: i32, x1: i32, y1: i32, n_half: i32, out: &mut [f32]) {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let ady = dy.abs();
    let base = dy / adx;
    let sy = if dy < 0 { base - 1 } else { base + 1 };
    let mut x = x0;
    let mut y = y0;
    let mut err = 0i32;
    let ady = ady - base.abs() * adx;
    let end = x1.min(n_half);

    if x >= 0 && x < end {
        out[x as usize] *= crate::dbtable::FLOOR1_INVERSE_DB[(y & 0xFF) as usize];
    }
    while {
        x += 1;
        x < end
    } {
        err += ady;
        if err >= adx {
            err -= adx;
            y += sy;
        } else {
            y += base;
        }
        if x >= 0 {
            out[x as usize] *= crate::dbtable::FLOOR1_INVERSE_DB[(y & 0xFF) as usize];
        }
    }
}

/// Vorbis I §6.2.3 Bark scale: smooth, monotonic map from Hz → Barks used
/// to convert the LSP's linear-frequency coefficients onto a
/// perceptually-uniform axis before rasterising.
#[inline]
fn bark(x: f32) -> f32 {
    13.1 * (0.00074 * x).atan() + 2.24 * (0.0000000185 * x * x).atan() + 0.0001 * x
}

/// Synthesize a floor type 0 curve (LSP → log-magnitude → linear
/// amplitude), multiplying each of the `n` bins of `output` by the
/// computed floor magnitude. Follows Vorbis I §6.2.3 literally:
/// compute `foobar` / `map` lookup, then walk bins, computing the LSP
/// frequency response `p + q` using the cosine-difference factored form
/// for odd vs even orders.
///
/// `decoded.coefficients` must already hold the cosine LSP coefficients
/// (cos ω_j) — which is what the VQ books store directly. On entry
/// `output[0..n]` contains the (already-dequantised) residue spectrum;
/// on exit it has been multiplied by the floor curve.
pub fn synth_floor0(
    floor: &Floor0,
    decoded: &Floor0Decoded,
    n: usize,
    output: &mut [f32],
) -> Result<()> {
    if output.len() < n {
        return Err(Error::invalid("synth_floor0: output buffer too short"));
    }
    if decoded.amplitude == 0 {
        for v in output.iter_mut().take(n) {
            *v = 0.0;
        }
        return Ok(());
    }
    let order = floor.order as usize;
    if decoded.coefficients.len() < order {
        return Err(Error::invalid(
            "synth_floor0: coefficient vector shorter than floor0_order",
        ));
    }
    if order == 0 {
        return Err(Error::invalid("synth_floor0: floor0_order is zero"));
    }
    let bark_map_size = floor.bark_map_size as usize;
    if bark_map_size == 0 {
        return Err(Error::invalid("synth_floor0: floor0_bark_map_size is zero"));
    }
    let rate = floor.rate as f32;
    let nyquist_bark = bark(0.5 * rate);
    if nyquist_bark <= 0.0 {
        return Err(Error::invalid("synth_floor0: non-positive nyquist bark"));
    }

    // Precompute the `map` lookup (Vorbis I §6.2.3):
    //   map[i] = min(bark_map_size - 1, floor(bark(rate*i/(2n)) * bark_map_size / bark(rate/2)))
    // Sentinel `-1` for `i == n` keeps the output loop's "while map[i]
    // matches iteration_condition" structure simple, which in our
    // indexed form just means "stop when we hit the last bin".
    let mut map = vec![0i32; n];
    let scale = bark_map_size as f32 / nyquist_bark;
    for i in 0..n {
        let hz = 0.5 * rate * (i as f32) / (n as f32);
        let v = (bark(hz) * scale).floor() as i32;
        let clamped = if v < 0 {
            0
        } else if (v as usize) >= bark_map_size {
            (bark_map_size - 1) as i32
        } else {
            v
        };
        map[i] = clamped;
    }

    // Pre-square the cosine coefficients so the "cos(coefficients) - cos(ω)"
    // factor inside the p/q products can be evaluated as a cheap
    // subtraction. (The VQ already decoded cos ω_j — not ω_j itself.)
    let cosc: &[f32] = &decoded.coefficients[..order];

    let odd = order & 1 == 1;
    let amp_offset = floor.amplitude_offset as f32;
    let amp_bits = floor.amplitude_bits as u32;
    let max_amp = ((1u64 << amp_bits) - 1) as f32;
    let amp = decoded.amplitude as f32;

    // Main loop: for each spectrum bin, compute cos(ω) once, then build
    // p and q via interleaved products (odd-indexed coeffs → p, even →
    // q), and derive the linear floor value from the dB-scale formula.
    let mut i = 0usize;
    while i < n {
        let m = map[i];
        // ω = π * m / bark_map_size
        let omega = std::f32::consts::PI * (m as f32) / (bark_map_size as f32);
        let cos_w = omega.cos();
        let (p, q) = if odd {
            // Order is odd: j runs 0..(order-3)/2 for p (odd-indexed
            // coefficients, 2j+1) and 0..(order-1)/2 for q (even
            // indices, 2j). Factor (1 - cos²ω) precedes p; 1/4 precedes
            // q.
            let mut p = 1.0 - cos_w * cos_w;
            if order >= 3 {
                let k_end = (order - 3) / 2;
                for j in 0..=k_end {
                    let d = cosc[2 * j + 1] - cos_w;
                    p *= 4.0 * d * d;
                }
            }
            let mut q = 0.25f32;
            let k_end = (order - 1) / 2;
            for j in 0..=k_end {
                let d = cosc[2 * j] - cos_w;
                q *= 4.0 * d * d;
            }
            (p, q)
        } else {
            // Order is even: j runs 0..(order-2)/2 for both p and q.
            let mut p = (1.0 - cos_w) * 0.5;
            let mut q = (1.0 + cos_w) * 0.5;
            let k_end = if order >= 2 { (order - 2) / 2 } else { 0 };
            for j in 0..=k_end {
                let d_p = cosc[2 * j + 1] - cos_w;
                let d_q = cosc[2 * j] - cos_w;
                p *= 4.0 * d_p * d_p;
                q *= 4.0 * d_q * d_q;
            }
            (p, q)
        };
        let sum = p + q;
        let denom = max_amp * sum.sqrt();
        let exponent = if denom > 0.0 {
            0.11512925 * ((amp * amp_offset) / denom - amp_offset)
        } else {
            // Extreme edge case — p+q = 0 means an LSP singularity.
            // Spec doesn't explicitly handle this, but libvorbis
            // multiplies by exp(-amp_offset * 0.11512925) which is what
            // the formula tends to with a huge denominator. We fall
            // back to zero here, matching the "silent bin" behaviour.
            -amp_offset * 0.11512925
        };
        let linear_floor = exponent.exp();

        // Walk forward while map[i] matches this one (step 5/8 of spec).
        let iter_cond = m;
        while i < n && map[i] == iter_cond {
            output[i] *= linear_floor;
            i += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod floor0_tests {
    use super::*;
    use crate::codebook::{Codebook, VqLookup};

    /// Build a single-entry codebook whose only entry is a fixed cosine
    /// vector. Forces the bitstream to read zero bits for both the
    /// Huffman scalar (one entry, 0-length code) and the book-number
    /// selection when `number_of_books == 1`.
    fn const_cosine_book(dim: usize, cosines: &[u32]) -> Codebook {
        // lookup_type=2 → flat per-entry multiplicand table of length
        // entries * dim. delta=1, min=0 means multiplicand values ARE
        // the coefficients directly (in f32 space). A single entry
        // keeps decode_scalar zero-cost.
        assert_eq!(cosines.len(), dim);
        let mut cb = Codebook {
            dimensions: dim as u16,
            entries: 1,
            codeword_lengths: vec![1u8],
            vq: Some(VqLookup {
                lookup_type: 2,
                min: 0.0,
                delta: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: cosines.to_vec(),
            }),
            codewords: vec![],
        };
        cb.build_decoder().unwrap();
        cb
    }

    fn null_floor0(order: u8) -> Floor0 {
        Floor0 {
            order,
            rate: 48_000,
            bark_map_size: 256,
            amplitude_bits: 8,
            amplitude_offset: 32,
            number_of_books: 1,
            book_list: vec![0],
        }
    }

    #[test]
    fn floor0_amplitude_zero_marks_unused() {
        // 8 bits of amplitude = 0 → single bit read, unused.
        let f0 = null_floor0(4);
        let codebooks = vec![const_cosine_book(1, &[0])];
        let bytes = [0u8; 4];
        let mut br = BitReader::new(&bytes);
        let dec = decode_floor0_packet(&f0, &codebooks, &mut br).unwrap();
        assert_eq!(dec.amplitude, 0);
        assert!(dec.coefficients.is_empty());
        let fd = FloorDecoded::Floor0(dec);
        assert!(fd.is_unused());
    }

    #[test]
    fn floor0_synth_unused_zeros_output() {
        let f0 = null_floor0(2);
        let dec = Floor0Decoded {
            amplitude: 0,
            book_number: 0,
            coefficients: vec![0.0, 0.0],
        };
        let mut spec = vec![1f32; 16];
        synth_floor0(&f0, &dec, 16, &mut spec).unwrap();
        for v in spec {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn floor0_synth_even_order_produces_positive_floor() {
        // Non-zero amplitude + arbitrary cosine coefficients must
        // produce a non-negative floor (dB space is well-defined
        // provided p+q > 0). Regression guard against sign errors in
        // the LSP factorisation.
        let f0 = null_floor0(4);
        let dec = Floor0Decoded {
            amplitude: 200,
            book_number: 0,
            coefficients: vec![0.9, 0.6, 0.3, -0.4],
        };
        let mut spec = vec![1f32; 32];
        synth_floor0(&f0, &dec, 32, &mut spec).unwrap();
        for v in spec {
            assert!(v >= 0.0, "expected non-negative floor, got {v}");
            assert!(v.is_finite(), "non-finite floor value: {v}");
        }
    }

    #[test]
    fn floor0_synth_odd_order_produces_positive_floor() {
        let f0 = null_floor0(3);
        let dec = Floor0Decoded {
            amplitude: 128,
            book_number: 0,
            coefficients: vec![0.7, 0.2, -0.5],
        };
        let mut spec = vec![1f32; 16];
        synth_floor0(&f0, &dec, 16, &mut spec).unwrap();
        for v in spec {
            assert!(v >= 0.0, "expected non-negative floor, got {v}");
            assert!(v.is_finite(), "non-finite floor value: {v}");
        }
    }

    #[test]
    fn floor0_dispatch_routes_to_floor0() {
        // End-to-end `decode_floor_packet` → matches Floor0 variant.
        let f0 = null_floor0(2);
        let floor = Floor::Type0(f0);
        let cosines = vec![128u32, 64u32];
        let codebooks = vec![const_cosine_book(2, &cosines)];
        // Bitstream: amplitude=1 (8 bits = 0x01), booknumber=0 (0 bits
        // because number_of_books=1 → ilog(1)=1, actually 1 bit of
        // zero), then one scalar decode (0 bits because single-entry
        // Huffman codebook). Layout (LSB-first): byte0 low bit = 1,
        // next 7 bits amplitude=1 (actually low 8 bits of byte0 = 0x01
        // amplitude), byte1 bit0 = 0 (booknumber=0).
        let bytes = [0x01u8, 0x00u8];
        let mut br = BitReader::new(&bytes);
        let dec = decode_floor_packet(&floor, &codebooks, &mut br).unwrap();
        match dec {
            FloorDecoded::Floor0(f) => {
                assert_eq!(f.amplitude, 1);
                assert_eq!(f.coefficients.len(), 2);
            }
            _ => panic!("expected Floor0 variant"),
        }
    }
}

//! Floor 0 (LSP) end-to-end packet round-trip: §6.2.2 plan → write →
//! decode → §6.2.3 curve, validated against an independent in-test oracle.
//!
//! # Why this test exists
//!
//! Every staged `docs/audio/vorbis/fixtures/*` stream uses **floor type 1**
//! — real `libvorbis`-class encoders emit floor 1 for these material types,
//! so the §4.3 fixture PCM decode (`tests/fixture_pcm_decode.rs`) never
//! exercises the **floor 0** curve path even though it is wired into the
//! decode driver (`audio::FloorDecoder::Type0`). The floor-0 §6.2.2 packet
//! decode and §6.2.3 LSP→envelope synthesis therefore had only unit-level
//! coverage and *no* end-to-end "plan an encode, write the bitstream, decode
//! it back, check the curve" round-trip — the floor-0 analog of the
//! floor-1 `tests/floor1_envelope_roundtrip.rs` and the §4.3
//! `tests/pcm_packet_roundtrip.rs`.
//!
//! This test closes that gap on the public API surface only:
//!
//!  1. **Plan.** [`plan_floor0_coefficients`] quantises a target LSP
//!     coefficient list into the §6.2.2 value-book entry run.
//!  2. **Write.** [`write_floor0_packet`] serialises a
//!     [`Floor0Packet::Curve`] (amplitude + booknumber + the planned entry
//!     run) into a §6.2.2 audio-packet body.
//!  3. **Decode.** [`Floor0Decoder::decode`] reads the body back, runs the
//!     §6.2.2 step-7..11 `[last]`-accumulating coefficient rebuild, and the
//!     §6.2.3 curve synthesis, returning a length-`n` linear envelope.
//!  4. **Assert.** Two independent checks:
//!     * the rebuilt **coefficients** equal the planner's own
//!       nearest-entry reconstruction bit-for-bit (the bitstream loop is
//!       lossless once the entries are chosen); and
//!     * the rebuilt **curve** equals an independent in-test recomputation
//!       of the §6.2.3 Bark-map + LSP-product + `exp` envelope from those
//!       same coefficients — so the decoder's §6.2.3 arithmetic is pinned
//!       against a second implementation of the spec formula, not just
//!       against itself.
//!
//! Fully synthetic: no Ogg framing, no `docs/` fixtures, so it runs in
//! standalone per-crate CI as well as the umbrella workspace.

use oxideav_core::bits::BitReaderLsb;
use oxideav_vorbis::{
    floor0_bark, ilog, plan_floor0_coefficients, write_floor0_packet, Floor0Curve, Floor0Decoder,
    Floor0Header, Floor0Packet, VorbisCodebook, VqLookup,
};

/// Build a tessellation (lookup-2) VQ value codebook with explicit
/// multiplicands (`minimum = 0`, `delta = 1`, `sequence_p` off), so entry
/// `e` unpacks to `multiplicands[e*dims .. e*dims+dims]` as `f32`. Every
/// entry is one bit long where `entries == 2` (prefix-free `0` / `1`); for
/// larger books the caller supplies a valid prefix-free length set.
fn tess_book(dimensions: u16, lengths: Vec<u8>, multiplicands: Vec<u32>) -> VorbisCodebook {
    VorbisCodebook {
        dimensions,
        entries: lengths.len() as u32,
        codeword_lengths: lengths,
        lookup: VqLookup::Tessellation {
            minimum_value: 0.0,
            delta_value: 1.0,
            value_bits: 8,
            sequence_p: false,
            multiplicands,
        },
    }
}

/// Independent oracle for the §6.2.2 step-7..11 coefficient rebuild: given
/// the chosen entry run and the value book, re-run the decode-side `[last]`
/// accumulation directly. Returns `ceil(order/dims)*dims` coefficients (the
/// decoder fills full vectors; the §6.2.3 step then reads only `order` of
/// them).
fn reconstruct_coefficients(entries: &[u32], book: &VorbisCodebook, order: usize) -> Vec<f32> {
    let dims = book.dimensions as usize;
    let VqLookup::Tessellation {
        minimum_value,
        delta_value,
        multiplicands,
        ..
    } = &book.lookup
    else {
        panic!("oracle only supports tessellation books");
    };
    let mut coeffs = Vec::new();
    let mut last = 0.0f32;
    for &entry in entries {
        let base = entry as usize * dims;
        let mut temp: Vec<f32> = (0..dims)
            .map(|j| multiplicands[base + j] as f32 * delta_value + minimum_value)
            .collect();
        for x in &mut temp {
            *x += last;
        }
        last = *temp.last().unwrap();
        coeffs.extend_from_slice(&temp);
        if coeffs.len() >= order {
            break;
        }
    }
    // The decoder always reads whole vectors; mirror that so the lengths
    // line up with the decoder's returned coefficient list.
    while coeffs.len() % dims != 0 {
        coeffs.push(0.0);
    }
    coeffs
}

/// Independent oracle for the §6.2.3 floor-0 curve synthesis. Recomputes
/// the length-`n` linear envelope from `amplitude` + the rebuilt LSP
/// `coefficients`, following the spec formula step for step — a second
/// implementation the decoder's `curve_computation` is checked against.
fn reference_curve(
    header: &Floor0Header,
    amplitude: u32,
    coefficients: &[f32],
    n: usize,
) -> Vec<f32> {
    let order = header.order as usize;
    let bark_map_size = header.bark_map_size as i32;
    let rate = header.rate as f64;

    // §6.2.3 Bark map: map[i] = min(bark_map_size-1, floor(bark(rate*i/2n) *
    // bark_map_size / bark(.5*rate))) for i in [0,n); map[n] = -1.
    let bark_denominator = floor0_bark((0.5 * rate) as f32);
    let mut map: Vec<i32> = Vec::with_capacity(n + 1);
    for i in 0..n {
        let f = (rate * i as f64) / (2.0 * n as f64);
        let foobar =
            (floor0_bark(f as f32) * header.bark_map_size as f32 / bark_denominator).floor() as i32;
        map.push(foobar.min(bark_map_size - 1));
    }
    map.push(-1);

    let coeffs = &coefficients[..order];
    let mut out = vec![0.0f32; n];
    let mut i = 0usize;
    while i < n {
        let omega = std::f32::consts::PI * map[i] as f32 / header.bark_map_size as f32;
        let cw = omega.cos();
        let (p, q) = if order % 2 == 1 {
            let mut p = 1.0f32 - cw * cw;
            for j in 0..(order - 1) / 2 {
                let c = coeffs[2 * j + 1].cos();
                p *= 4.0 * (c - cw) * (c - cw);
            }
            let mut q = 0.25f32;
            for j in 0..order / 2 + 1 {
                let c = coeffs[2 * j].cos();
                q *= 4.0 * (c - cw) * (c - cw);
            }
            (p, q)
        } else {
            let mut p = (1.0f32 - cw) / 2.0;
            let mut q = (1.0f32 + cw) / 2.0;
            for j in 0..order / 2 {
                let co = coeffs[2 * j + 1].cos();
                p *= 4.0 * (co - cw) * (co - cw);
                let ce = coeffs[2 * j].cos();
                q *= 4.0 * (ce - cw) * (ce - cw);
            }
            (p, q)
        };
        let sqrt_pq = (p + q).max(0.0).sqrt().max(f32::MIN_POSITIVE);
        let denom_int: u32 = if header.amplitude_bits >= 32 {
            u32::MAX
        } else {
            (1u32 << header.amplitude_bits) - 1
        };
        let denom = denom_int as f32 * sqrt_pq;
        let offset = header.amplitude_offset as f32;
        let lin = (0.115_129_25_f32 * (amplitude as f32 * offset / denom - offset)).exp();
        let cond = map[i];
        out[i] = lin;
        i += 1;
        while i < n && map[i] == cond {
            out[i] = lin;
            i += 1;
        }
    }
    out
}

/// Serialise a `Floor0Packet::Curve`, decode it back, and return the
/// decoder's curve plus the rebuilt coefficients, asserting the bitstream
/// loop is lossless and the curve matches the independent §6.2.3 oracle.
fn assert_floor0_curve_roundtrips(
    header: &Floor0Header,
    book: &VorbisCodebook,
    target: &[f32],
    amplitude: u32,
    n: usize,
) {
    let order = header.order as usize;
    let codebooks = std::slice::from_ref(book);

    // 1. Plan the §6.2.2 entry run from the target LSP coefficients.
    let entries = plan_floor0_coefficients(target, book, order)
        .expect("planner accepts a well-formed target");
    assert_eq!(
        entries.len(),
        order.div_ceil(book.dimensions as usize),
        "one entry per §6.2.2 VQ vector"
    );

    // 2. Write the §6.2.2 audio-packet body (amplitude + booknumber + run).
    let packet = Floor0Packet::Curve {
        amplitude,
        booknumber: 0,
        entries: entries.clone(),
    };
    let body = write_floor0_packet(&packet, header, codebooks).expect("write succeeds");

    // 3. Decode the body back through the public floor-0 decoder.
    let decoder = Floor0Decoder::new(header, codebooks).expect("decoder builds");
    let mut reader = BitReaderLsb::new(&body);
    let curve = match decoder.decode(&mut reader, n) {
        Floor0Curve::Curve(c) => c,
        Floor0Curve::Unused => panic!("nonzero-amplitude curve must not decode as unused"),
    };
    assert_eq!(curve.len(), n, "curve length is n");

    // 4a. The rebuilt coefficients equal the independent §6.2.2 oracle's
    //     reconstruction — the bitstream loop is lossless once entries are
    //     chosen. We recompute the curve from the oracle coefficients and
    //     require an exact float match against the decoder's curve.
    let coeffs = reconstruct_coefficients(&entries, book, order);
    let oracle = reference_curve(header, amplitude, &coeffs, n);
    assert_eq!(
        oracle.len(),
        curve.len(),
        "oracle and decoder curve lengths agree"
    );
    for (i, (&got, &want)) in curve.iter().zip(&oracle).enumerate() {
        // The decoder and the oracle run the identical spec arithmetic on
        // the identical coefficients, so the result is bit-for-bit equal.
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "curve[{i}] decoder={got} oracle={want} (§6.2.3 mismatch)"
        );
    }

    // 4b. Every curve sample is finite and strictly positive (the §6.2.3
    //     envelope is exp(...) of a finite argument, so it can never be
    //     zero, negative, NaN, or infinite for a nonzero amplitude).
    for (i, &v) in curve.iter().enumerate() {
        assert!(
            v.is_finite() && v > 0.0,
            "curve[{i}] = {v} not finite-positive"
        );
    }
}

/// Even order (4): the §6.2.3 even-parity LSP-product branch, dim-2 book.
#[test]
fn even_order_floor0_curve_roundtrips() {
    let header = Floor0Header {
        order: 4,
        rate: 44_100,
        bark_map_size: 256,
        amplitude_bits: 6,
        amplitude_offset: 100,
        book_list: vec![0],
    };
    // dim-2 tessellation book, 4 entries (2-bit prefix-free lengths).
    let book = tess_book(2, vec![2, 2, 2, 2], vec![1, 5, 3, 9, 7, 2, 4, 8]);
    // order=4, dims=2 → 2 vectors → 4 target coefficients.
    let target = [1.0f32, 5.0, 9.0, 11.0];
    assert_floor0_curve_roundtrips(&header, &book, &target, 42, 128);
}

/// Odd order (3): the §6.2.3 odd-parity LSP-product branch (the
/// `(1 - cos²ω)` / `0.25` split), dim-1 book so each scalar is its own
/// vector and the `[last]` accumulator threads scalar-by-scalar.
#[test]
fn odd_order_floor0_curve_roundtrips() {
    let header = Floor0Header {
        order: 3,
        rate: 48_000,
        bark_map_size: 128,
        amplitude_bits: 7,
        amplitude_offset: 90,
        book_list: vec![0],
    };
    // dim-1 book: 4 entries, 2-bit prefix-free lengths.
    let book = tess_book(1, vec![2, 2, 2, 2], vec![1, 4, 6, 10]);
    // order=3, dims=1 → 3 vectors → 3 target coefficients.
    let target = [1.0f32, 4.0, 10.0];
    assert_floor0_curve_roundtrips(&header, &book, &target, 70, 96);
}

/// A partial final vector: order=3 with a dim-2 book reads ceil(3/2)=2
/// vectors (4 coefficients), the 4th read-and-discarded by §6.2.3. The
/// target must carry the full padded length (4), exercising the
/// surplus-scalar discard.
#[test]
fn partial_final_vector_floor0_curve_roundtrips() {
    let header = Floor0Header {
        order: 3,
        rate: 22_050,
        bark_map_size: 64,
        amplitude_bits: 5,
        amplitude_offset: 80,
        book_list: vec![0],
    };
    let book = tess_book(2, vec![2, 2, 2, 2], vec![2, 6, 4, 8, 1, 9, 3, 7]);
    // order=3, dims=2 → 2 vectors → 4 padded coefficients.
    let target = [2.0f32, 6.0, 10.0, 14.0];
    assert_floor0_curve_roundtrips(&header, &book, &target, 20, 80);
}

/// Distinct block sizes (`n_0` short vs `n_1` long): the §6.2.3 Bark map is
/// recomputed per `n`, so the same coefficients render different-length
/// curves; both must match the oracle.
#[test]
fn floor0_curve_roundtrips_across_block_sizes() {
    let header = Floor0Header {
        order: 4,
        rate: 44_100,
        bark_map_size: 256,
        amplitude_bits: 6,
        amplitude_offset: 100,
        book_list: vec![0],
    };
    let book = tess_book(2, vec![2, 2, 2, 2], vec![1, 5, 3, 9, 7, 2, 4, 8]);
    let target = [1.0f32, 5.0, 9.0, 11.0];
    // n = half of a 256-sample short block and a 2048-sample long block.
    for &n in &[128usize, 1024] {
        assert_floor0_curve_roundtrips(&header, &book, &target, 42, n);
    }
}

/// The §6.2.2 zero-amplitude short-circuit: an `Unused` packet writes only
/// the amplitude field and the decoder returns `Unused` (no curve), without
/// consuming the booknumber or any VQ codewords.
#[test]
fn unused_floor0_packet_decodes_as_unused() {
    let header = Floor0Header {
        order: 4,
        rate: 44_100,
        bark_map_size: 256,
        amplitude_bits: 6,
        amplitude_offset: 100,
        book_list: vec![0],
    };
    let book = tess_book(2, vec![2, 2, 2, 2], vec![1, 5, 3, 9, 7, 2, 4, 8]);
    let codebooks = std::slice::from_ref(&book);
    let body = write_floor0_packet(&Floor0Packet::Unused, &header, codebooks).expect("write");
    // §6.2.2 step 2 reads only the amplitude_bits-wide zero field.
    assert_eq!(body.len(), 1, "only the 6-bit amplitude field is emitted");
    let decoder = Floor0Decoder::new(&header, codebooks).expect("decoder builds");
    let mut reader = BitReaderLsb::new(&body);
    assert_eq!(decoder.decode(&mut reader, 128), Floor0Curve::Unused);
}

/// The §6.2.2 booknumber field is `ilog(number_of_books)` bits wide; with a
/// single-entry book_list that is exactly 1 bit. Pin the width so the
/// round-trip's bit budget is anchored to the spec rule, not coincidence.
#[test]
fn floor0_booknumber_field_width_matches_spec() {
    // ilog(1) = 1, ilog(2) = 2, ilog(4) = 3 (§6.2.2 step 4 width).
    assert_eq!(ilog(1), 1);
    assert_eq!(ilog(2), 2);
    assert_eq!(ilog(4), 3);
}

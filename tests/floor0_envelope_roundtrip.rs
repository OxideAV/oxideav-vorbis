//! Floor-0 envelope → packet → decode round-trip (Vorbis I §6.2.2 / §6.2.3,
//! encode direction).
//!
//! The crate's floor-0 encode chain is now closed end to end:
//! `floor0_envelope::plan_floor0_lsp` (envelope → LSP coefficients) →
//! `floor0_envelope::fit_floor0_amplitude` (gain) →
//! `floor0_encode::plan_floor0_coefficients` (entry run), composed by the
//! one-call `floor0_envelope::plan_floor0_packet`. This suite drives a
//! desired linear-domain floor envelope through `plan_floor0_packet` →
//! `encoder::write_floor0_packet` → the public `Floor0Decoder` and pins:
//!
//! 1. the planned packet **round-trips** — decoding the written body
//!    reproduces the planner's own LSP approximation bit-for-bit (the
//!    decoder's curve equals an independent §6.2.3 render over the rebuilt
//!    coefficients);
//! 2. the decoded curve tracks the target envelope's **shape** to a
//!    meaningful log-domain SNR (the all-pole model is lossy but faithful);
//! 3. the `booknumber` selector is honoured and out-of-range inputs are
//!    rejected.
//!
//! No staged fixture exercises floor 0 (no reference encoder emits it), so
//! self-consistency against the crate's own decoder is the ground truth.

use oxideav_core::bits::BitReaderLsb;
use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::encoder::{write_floor0_packet, Floor0Packet};
use oxideav_vorbis::floor0::{Floor0Curve, Floor0Decoder};
use oxideav_vorbis::floor0_envelope::{plan_floor0_packet, Floor0PacketPlanError};
use oxideav_vorbis::setup::Floor0Header;

/// A fine-resolution **scalar** (dimensions = 1) value codebook spanning a
/// range wide enough to carry the LSP **delta** values
/// `plan_floor0_coefficients` quantises (consecutive LSP angles differ by
/// ~π/order; the per-vector targets are those deltas un-offset by the running
/// `[last]`). 256 entries (a full balanced 8-bit prefix code, so the decoder
/// Huffman tree and the writer are both well-defined), `delta_value = 0.01`,
/// `minimum_value = -0.5`, covering `[-0.5, +2.05]` in 0.01 steps — fine
/// enough that coefficient quantisation barely perturbs the curve.
fn lsp_value_book() -> VorbisCodebook {
    let entries: u32 = 256;
    let multiplicands: Vec<u32> = (0..entries).collect();
    VorbisCodebook {
        dimensions: 1,
        entries,
        // 256 entries, each length 8 → a full balanced prefix code.
        codeword_lengths: vec![8; entries as usize],
        lookup: VqLookup::Tessellation {
            minimum_value: -0.5,
            delta_value: 0.01,
            value_bits: 8,
            sequence_p: false,
            multiplicands,
        },
    }
}

fn header(order: u8) -> Floor0Header {
    Floor0Header {
        order,
        rate: 44_100,
        bark_map_size: 256,
        amplitude_bits: 10,
        amplitude_offset: 32,
        book_list: vec![0],
    }
}

/// log-domain shape SNR (dB) — the absolute log-level (the encoder's free
/// amplitude knob) is removed from both signals before comparison.
fn log_shape_snr_db(rendered: &[f32], target: &[f32]) -> f64 {
    let n = target.len() as f64;
    let tmean: f64 = target.iter().map(|x| (*x as f64).ln()).sum::<f64>() / n;
    let rmean: f64 = rendered.iter().map(|x| (*x as f64).ln()).sum::<f64>() / n;
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (r, t) in rendered.iter().zip(target.iter()) {
        let lt = (*t as f64).ln() - tmean;
        let lr = (*r as f64).ln() - rmean;
        sig += lt * lt;
        err += (lt - lr) * (lt - lr);
    }
    10.0 * (sig / err.max(1e-30)).log10()
}

/// A formant-like target whose *log* (dB) shape is a sum of smooth resonant
/// bumps — the shape floor 0 (an all-pole model of the log spectrum)
/// represents naturally.
fn formant_target(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let w = std::f32::consts::PI * i as f32 / n as f32;
            let log_db = 3.0 / ((w - 0.4).powi(2) + 0.05)
                + 2.0 / ((w - 1.2).powi(2) + 0.08)
                + 1.5 / ((w - 2.3).powi(2) + 0.12);
            (0.2 * log_db).exp()
        })
        .collect()
}

#[test]
fn envelope_plans_a_packet_that_round_trips_through_the_decoder() {
    let order = 14u8;
    let hdr = header(order);
    let books = vec![lsp_value_book()];
    let n = 256;
    let target = formant_target(n);

    let packet = plan_floor0_packet(&hdr, &books, 0, &target).expect("packet plan succeeds");
    let (amplitude, entries) = match &packet {
        Floor0Packet::Curve {
            amplitude,
            booknumber,
            entries,
        } => {
            assert_eq!(*booknumber, 0);
            (*amplitude, entries.clone())
        }
        Floor0Packet::Unused => panic!("a nonzero envelope must plan a Curve"),
    };
    assert!(amplitude >= 1, "amplitude must be a legal nonzero field");
    // One entry per §6.2.2 VQ vector: ceil(order / dims) = ceil(14/1) = 14.
    assert_eq!(entries.len(), 14, "one entry per VQ vector");

    // Write the body and decode it back.
    let body = write_floor0_packet(&packet, &hdr, &books).expect("write succeeds");
    let decoder = Floor0Decoder::new(&hdr, &books).expect("decoder builds");
    let mut reader = BitReaderLsb::new(&body);
    let curve = match decoder.decode(&mut reader, n) {
        Floor0Curve::Curve(c) => c,
        Floor0Curve::Unused => panic!("nonzero-amplitude curve must not decode as unused"),
    };
    assert_eq!(curve.len(), n);

    // Round-trip exactness: the decoded curve equals an independent render of
    // the planner's own rebuilt coefficients (the bitstream loop is lossless
    // once entries are chosen). Reconstruct coefficients the decode way.
    let coeffs = reconstruct_coefficients(&entries, &books[0], order as usize);
    let oracle = decoder.render_curve(amplitude, &coeffs, n);
    for (i, (&got, &want)) in curve.iter().zip(&oracle).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "curve[{i}] decoder={got} oracle={want} (§6.2.3 mismatch)"
        );
    }

    // Shape fidelity: the decoded curve tracks the target envelope's shape.
    let snr = log_shape_snr_db(&curve, &target);
    assert!(
        snr > 8.0,
        "floor-0 envelope round-trip shape SNR {snr:.2} dB should clear 8 dB"
    );

    // Every curve sample is finite-positive.
    for (i, &v) in curve.iter().enumerate() {
        assert!(
            v.is_finite() && v > 0.0,
            "curve[{i}]={v} not finite-positive"
        );
    }
}

#[test]
fn odd_order_packet_round_trips() {
    let order = 13u8; // odd-parity §6.2.3 LSP-product branch
    let hdr = header(order);
    let books = vec![lsp_value_book()];
    let n = 192;
    let target = formant_target(n);

    let packet = plan_floor0_packet(&hdr, &books, 0, &target).expect("plan succeeds");
    let body = write_floor0_packet(&packet, &hdr, &books).expect("write succeeds");
    let decoder = Floor0Decoder::new(&hdr, &books).expect("decoder builds");
    let mut reader = BitReaderLsb::new(&body);
    let curve = match decoder.decode(&mut reader, n) {
        Floor0Curve::Curve(c) => c,
        Floor0Curve::Unused => panic!("must decode as a curve"),
    };
    let snr = log_shape_snr_db(&curve, &target);
    assert!(
        snr > 8.0,
        "odd-order shape SNR {snr:.2} dB should clear 8 dB"
    );
}

#[test]
fn out_of_range_booknumber_is_rejected() {
    let hdr = header(8);
    let books = vec![lsp_value_book()];
    let target = formant_target(64);
    assert_eq!(
        plan_floor0_packet(&hdr, &books, 1, &target),
        Err(Floor0PacketPlanError::BooknumberOutOfRange {
            booknumber: 1,
            book_count: 1,
        })
    );
}

#[test]
fn book_list_out_of_range_is_rejected() {
    let mut hdr = header(8);
    hdr.book_list = vec![5]; // indexes past a single-codebook table
    let books = vec![lsp_value_book()];
    let target = formant_target(64);
    assert_eq!(
        plan_floor0_packet(&hdr, &books, 0, &target),
        Err(Floor0PacketPlanError::BookOutOfRange {
            book: 5,
            codebook_count: 1,
        })
    );
}

/// Reconstruct the §6.2.2 step-7..11 coefficients from an entry run + a
/// tessellation value book (the decode-side `[last]` accumulation), so the
/// round-trip can be checked against an independent render.
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
    while coeffs.len() % dims != 0 {
        coeffs.push(0.0);
    }
    coeffs
}

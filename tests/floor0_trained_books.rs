//! Floor-0 value-codebook-*content* training round-trip (Vorbis I
//! §3.2.1 / §6.2.2) — the "floor-0 value-codebook *contents*" followup
//! the README names, closed end to end.
//!
//! Retraining a codebook rewrites only its codeword lengths and
//! preserves its VQ lookup, so a planned `Floor0Packet`'s entry run
//! rebuilds the identical LSP coefficient list — and therefore the
//! bit-identical §6.2.3 curve — under the trained book. Training the
//! value book on an envelope corpus must therefore satisfy the sharp
//! contract:
//!
//! * every retrained packet decodes to the **bit-identical** floor
//!   curve, and
//! * the corpus serialises into **strictly fewer bytes** than under
//!   the untrained flat book.
//!
//! The chain, per corpus member: formant-style envelope →
//! `plan_floor0_packet` (LSP fit + amplitude + VQ entry run) →
//! `tally_floor0_packet` (the §6.2.2 emission tally) → after the whole
//! corpus, `BookTallies::retrain` → re-write every packet with the
//! trained book → decode with a `Floor0Decoder` built over the trained
//! book → compare curves and sizes. Because LSP coefficients of
//! similar spectra cluster tightly on the value ladder, the corpus
//! distribution is strongly non-uniform — exactly the case content
//! design exists for. No reference encoder emits floor 0, so
//! self-consistency against the crate's own decoder is the ground
//! truth (as for every other floor-0 suite).
//!
//! Fully synthetic: no Ogg framing, no `docs/` fixtures, so it runs in
//! standalone CI.

use oxideav_core::bits::BitReaderLsb;
use oxideav_vorbis::codebook::{parse_codebook, VorbisCodebook, VqLookup};
use oxideav_vorbis::floor0::{Floor0Curve, Floor0Decoder};
use oxideav_vorbis::floor0_envelope::plan_floor0_packet;
use oxideav_vorbis::setup::Floor0Header;
use oxideav_vorbis::{
    design_entropy_codebook, tally_floor0_packet, write_codebook, write_floor0_packet, BookTallies,
    Floor0Packet,
};

/// Spectral bins the envelopes are defined over (also the render `n/2`).
const BINS: usize = 64;

/// The 256-entry 1-D tessellation LSP value book, flat 8-bit codeword
/// lengths — the untrained starting point. The dyadic ladder
/// `−0.5 + e/64` spans the `0..π` LSP angle range comfortably *and*
/// is exactly representable in the §9.2.2 packed-float container, so
/// the book (and its retrained descendants) are carriage-legal.
fn lsp_value_book() -> VorbisCodebook {
    let entries: u32 = 256;
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![8; entries as usize],
        lookup: VqLookup::Tessellation {
            minimum_value: -0.5,
            delta_value: 0.015625,
            value_bits: 8,
            sequence_p: false,
            multiplicands: (0..entries).collect(),
        },
    }
}

fn header(order: u8) -> Floor0Header {
    Floor0Header {
        order,
        rate: 44_100,
        bark_map_size: 128,
        amplitude_bits: 10,
        amplitude_offset: 32,
        book_list: vec![0],
    }
}

/// The envelope corpus: three formant resonances whose centres drift
/// frame to frame — spectra that stay *similar* across the corpus, so
/// the fitted LSP coefficients (and their VQ entries) cluster.
fn corpus() -> Vec<Vec<f32>> {
    (0..32usize)
        .map(|frame| {
            let drift = 0.02 * (frame % 8) as f32;
            (0..BINS)
                .map(|i| {
                    let w = std::f32::consts::PI * i as f32 / BINS as f32;
                    let log_db = 3.0 / ((w - 0.4 - drift).powi(2) + 0.05)
                        + 2.0 / ((w - 1.2 + drift).powi(2) + 0.08)
                        + 1.5 / ((w - 2.3 - drift).powi(2) + 0.12);
                    (0.2 * log_db).exp()
                })
                .collect()
        })
        .collect()
}

fn decode_curve(hdr: &Floor0Header, books: &[VorbisCodebook], body: &[u8]) -> Vec<f32> {
    let decoder = Floor0Decoder::new(hdr, books).expect("decoder builds");
    let mut reader = BitReaderLsb::new(body);
    match decoder.decode(&mut reader, 2 * BINS) {
        Floor0Curve::Curve(c) => c,
        Floor0Curve::Unused => panic!("planned packet must decode to a curve"),
    }
}

/// The full training loop: plan → tally → retrain → re-write. The
/// trained value book must decode every packet to the bit-identical
/// §6.2.3 curve while the corpus shrinks on the wire, and must stay
/// carriage-legal through the §3.2.1 codebook writer/parser.
#[test]
fn trained_floor0_value_book_shrinks_corpus_and_decodes_identically() {
    let hdr = header(12);
    let books = vec![lsp_value_book()];
    let envelopes = corpus();

    // Plan every packet against the flat book and tally its entry run.
    let mut tallies = BookTallies::new(&books);
    let mut packets: Vec<Floor0Packet> = Vec::with_capacity(envelopes.len());
    for envelope in &envelopes {
        let packet = plan_floor0_packet(&hdr, &books, 0, envelope).expect("plans");
        tally_floor0_packet(&mut tallies, &packet, &hdr).expect("tallies");
        packets.push(packet);
    }
    assert!(tallies.total(0) > 0, "the value book must be exercised");
    // The clustering premise: the corpus uses a strict minority of the
    // 256 entries, so there is real statistical structure to train on.
    let used_entries = tallies
        .counts(0)
        .unwrap()
        .iter()
        .filter(|&&f| f > 0)
        .count();
    assert!(
        used_entries < 128,
        "LSP entries must cluster for the training premise: {used_entries} used"
    );

    // Baseline: serialise + decode under the flat book.
    let mut flat_bytes = 0usize;
    let mut flat_curves: Vec<Vec<f32>> = Vec::with_capacity(packets.len());
    for packet in &packets {
        let body = write_floor0_packet(packet, &hdr, &books).expect("writes");
        flat_curves.push(decode_curve(&hdr, &books, &body));
        flat_bytes += body.len();
    }

    // Retrain (dense) and re-write the same packets.
    let trained = tallies.retrain(&books, 32, true).expect("retrains");
    assert_eq!(trained[0].entries, books[0].entries);
    assert_eq!(trained[0].lookup, books[0].lookup, "VQ ladder preserved");
    assert_ne!(
        trained[0].codeword_lengths, books[0].codeword_lengths,
        "the book must actually be re-optimised"
    );
    let mut trained_bytes = 0usize;
    for (packet, flat_curve) in packets.iter().zip(flat_curves.iter()) {
        let body = write_floor0_packet(packet, &hdr, &trained).expect("writes");
        trained_bytes += body.len();
        let curve = decode_curve(&hdr, &trained, &body);
        assert_eq!(
            &curve, flat_curve,
            "trained book must decode the bit-identical §6.2.3 curve"
        );
    }
    assert!(
        trained_bytes < flat_bytes,
        "training must shrink the corpus: {trained_bytes} vs {flat_bytes} bytes"
    );

    // Carriage legality: the trained book round-trips through the
    // §3.2.1 codebook writer and parser field-for-field.
    let bytes = write_codebook(&trained[0]).expect("writes");
    let mut r = BitReaderLsb::new(&bytes);
    assert_eq!(&parse_codebook(&mut r).expect("parses"), &trained[0]);
}

/// Sparse retraining prunes the (many) never-used ladder entries;
/// re-*planning* the corpus against the sparse book still succeeds
/// (the §3.2.1 quantiser only ever selects used entries), decodes to
/// a curve at least as close to the flat-book plan's, and costs no
/// more than the dense retrain on the training corpus.
#[test]
fn sparse_floor0_retrain_replans_and_costs_no_more_than_dense() {
    let hdr = header(12);
    let books = vec![lsp_value_book()];
    let envelopes = corpus();

    let mut tallies = BookTallies::new(&books);
    let mut packets: Vec<Floor0Packet> = Vec::with_capacity(envelopes.len());
    for envelope in &envelopes {
        let packet = plan_floor0_packet(&hdr, &books, 0, envelope).expect("plans");
        tally_floor0_packet(&mut tallies, &packet, &hdr).expect("tallies");
        packets.push(packet);
    }
    let dense = tallies.retrain(&books, 32, true).expect("dense retrain");
    let sparse = tallies.retrain(&books, 32, false).expect("sparse retrain");
    let pruned = sparse[0]
        .codeword_lengths
        .iter()
        .filter(|&&l| l == 0)
        .count();
    assert!(
        pruned > 0,
        "sparse retrain must prune unseen ladder entries"
    );

    let mut dense_bytes = 0usize;
    let mut sparse_bytes = 0usize;
    for (envelope, packet) in envelopes.iter().zip(packets.iter()) {
        let dense_body = write_floor0_packet(packet, &hdr, &dense).expect("dense writes");
        dense_bytes += dense_body.len();

        // Re-plan against the sparse book. The quantiser skips pruned
        // entries, so the plan may differ — but every corpus entry
        // remains used, and re-planning the *same* envelope picks the
        // same nearest entries.
        let replanned = plan_floor0_packet(&hdr, &sparse, 0, envelope).expect("re-plans");
        let sparse_body = write_floor0_packet(&replanned, &hdr, &sparse).expect("sparse writes");
        sparse_bytes += sparse_body.len();
        assert_eq!(
            decode_curve(&hdr, &sparse, &sparse_body),
            decode_curve(&hdr, &dense, &dense_body),
            "the sparse re-plan must reconstruct the same §6.2.3 curve"
        );
    }
    assert!(
        sparse_bytes <= dense_bytes,
        "sparse book cannot cost more on its own training corpus: {sparse_bytes} vs {dense_bytes}"
    );
}

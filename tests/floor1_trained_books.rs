//! Floor-1 codebook-*content* training round-trip (Vorbis I §3.2.1 /
//! §7.2.3) — the "master/sub codebook contents" followup the README
//! names, closed end to end.
//!
//! Floor-1 post coding is lossless given the fitted `[floor1_Y]`
//! targets, and retraining a codebook rewrites only its codeword
//! *lengths* (the §3.2.1 canonical codewords are implied). So training
//! the floor books on a corpus must satisfy a sharp contract:
//!
//! * every retrained packet decodes to the **bit-identical** floor
//!   curve (same `[floor1_Y]`, same §7.2.4 render), and
//! * the corpus serialises into **strictly fewer bytes** than under
//!   the untrained flat books.
//!
//! The chain, per corpus member: synthetic envelope →
//! `plan_floor1_packet` (fit + unwrap + cval selection) →
//! `tally_floor1_packet` (the §7.2.3 emission tally) → after the whole
//! corpus, `BookTallies::retrain` → re-write every packet with the
//! trained books → decode with a `Floor1Decoder` built over the
//! trained books → compare curves and sizes. Both class shapes are
//! exercised (a `subclasses = 0` class and a `subclasses = 1` class
//! with a master book selecting between a coarse and a fine sub-book),
//! so the master-selector tally path is load-bearing. The trained
//! books are additionally pushed through the §3.2.1 codebook writer
//! and parser to prove they remain carriage-legal in a setup header.
//!
//! Fully synthetic: no Ogg framing, no `docs/` fixtures, so it runs in
//! standalone CI.

use oxideav_core::bits::BitReaderLsb;
use oxideav_vorbis::codebook::{parse_codebook, VorbisCodebook};
use oxideav_vorbis::floor1::Floor1Decoder;
use oxideav_vorbis::floor1_encode::plan_floor1_packet;
use oxideav_vorbis::setup::{Floor1Class, Floor1Header};
use oxideav_vorbis::{
    design_entropy_codebook, tally_floor1_packet, write_codebook, write_floor1_packet, BookTallies,
    Floor1Packet,
};

/// Floor length (spectral bins) the envelopes are defined over.
const FLOOR_LEN: usize = 256;

/// The training header: three partitions mixing both class shapes.
/// Class 0: `subclasses = 0`, two dimensions, slot 0 → book 2 (the
/// full-range 128-entry Y book, since a subclass-less class must carry
/// every Y it meets through its only slot). Class 1: `subclasses = 1`,
/// two dimensions, masterbook 1 (4 entries), slot 0 → book 0 (a
/// 96-entry book carrying only `Y < 96`), slot 1 → book 2. The corpus
/// straddles the 96 boundary in class-1 partitions, so the §7.2.3
/// master-selector path is genuinely load-bearing: the planner must
/// route large-Y dimensions through slot 1. Multiplier 2 → §7.2.3
/// `range = 128`.
fn training_header() -> Floor1Header {
    Floor1Header {
        partitions: 3,
        partition_class_list: vec![0, 1, 0],
        classes: vec![
            Floor1Class {
                dimensions: 2,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![Some(2)],
            },
            Floor1Class {
                dimensions: 2,
                subclasses: 1,
                masterbook: Some(1),
                subclass_books: vec![Some(0), Some(2)],
            },
        ],
        multiplier: 2,
        rangebits: 8,
        x_list: vec![16, 32, 64, 96, 128, 192],
    }
}

/// The untrained starting books: flat (balanced) codeword lengths,
/// built through the designer itself with uniform frequencies so each
/// book is Kraft-complete by construction. Book 0 is the coarse-range
/// sub-book (`Y < 96` only), book 1 the 4-entry master book, book 2
/// the full-range Y book.
fn flat_books() -> Vec<VorbisCodebook> {
    vec![
        design_entropy_codebook(96, 1, &[1u64; 96], 32, false).unwrap(),
        design_entropy_codebook(4, 1, &[1u64; 4], 32, false).unwrap(),
        design_entropy_codebook(128, 1, &[1u64; 128], 32, false).unwrap(),
    ]
}

/// The synthetic envelope corpus: a steep spectral tilt whose overall
/// gain sweeps a decade across the corpus, so the fitted Y values
/// cluster (a trainable, non-uniform symbol distribution — like real
/// audio, where neighbouring frames share their spectral shape) while
/// the class-1 partition's Y values straddle the coarse sub-book's 96
/// boundary from frame to frame.
fn corpus() -> Vec<Vec<f32>> {
    (0..48usize)
        .map(|frame| {
            let gain = 0.05 * 10f32.powf(frame as f32 / 47.0); // 0.05 .. 0.5
            (0..FLOOR_LEN)
                .map(|bin| {
                    let x = bin as f32 / FLOOR_LEN as f32;
                    1e-6 + gain * 10f32.powf(-4.0 * x)
                })
                .collect()
        })
        .collect()
}

fn decode_curve(header: &Floor1Header, books: &[VorbisCodebook], packet_bytes: &[u8]) -> Vec<f32> {
    let decoder = Floor1Decoder::new(header, books).expect("decoder builds");
    let mut reader = BitReaderLsb::new(packet_bytes);
    match decoder.decode(&mut reader, 2 * FLOOR_LEN) {
        oxideav_vorbis::FloorCurve::Curve(c) => c,
        oxideav_vorbis::FloorCurve::Unused => panic!("planned packet must decode to a curve"),
    }
}

/// The full training loop: plan → tally → retrain → re-write. Trained
/// books must decode every packet to the bit-identical curve while the
/// corpus shrinks on the wire, and must stay carriage-legal through
/// the §3.2.1 codebook writer/parser.
#[test]
fn trained_floor1_books_shrink_corpus_and_decode_identically() {
    let header = training_header();
    let books = flat_books();
    let envelopes = corpus();

    // Plan every packet against the flat books and tally its emissions.
    let mut tallies = BookTallies::new(&books);
    let mut packets: Vec<Floor1Packet> = Vec::with_capacity(envelopes.len());
    for envelope in &envelopes {
        let packet = plan_floor1_packet(envelope, &header, &books).expect("plans");
        tally_floor1_packet(&mut tallies, &packet, &header).expect("tallies");
        packets.push(packet);
    }
    // Both Y books and the master book must have been exercised, or
    // the retraining claim below is vacuous.
    assert!(tallies.total(0) > 0, "coarse Y book must be exercised");
    assert!(tallies.total(1) > 0, "master book must be exercised");
    assert!(tallies.total(2) > 0, "fine Y book must be exercised");

    // Baseline: serialise + decode under the flat books.
    let mut flat_bytes = 0usize;
    let mut flat_curves: Vec<Vec<f32>> = Vec::with_capacity(packets.len());
    for packet in &packets {
        let bytes = write_floor1_packet(packet, &header, &books).expect("writes");
        flat_curves.push(decode_curve(&header, &books, &bytes));
        flat_bytes += bytes.len();
    }

    // Retrain (dense: every entry stays encodable for unseen frames).
    let trained = tallies.retrain(&books, 32, true).expect("retrains");
    assert_eq!(trained.len(), books.len());
    for (i, (t, b)) in trained.iter().zip(books.iter()).enumerate() {
        assert_eq!(t.entries, b.entries, "book {i} keeps its shape");
        assert_ne!(
            t.codeword_lengths, b.codeword_lengths,
            "book {i} must actually be re-optimised"
        );
    }

    // Re-write the same packets with the trained books.
    let mut trained_bytes = 0usize;
    for (packet, flat_curve) in packets.iter().zip(flat_curves.iter()) {
        let bytes = write_floor1_packet(packet, &header, &trained).expect("writes");
        trained_bytes += bytes.len();
        let curve = decode_curve(&header, &trained, &bytes);
        assert_eq!(
            &curve, flat_curve,
            "trained books must decode the bit-identical §7.2.4 curve"
        );
    }
    assert!(
        trained_bytes < flat_bytes,
        "training must shrink the corpus: {trained_bytes} vs {flat_bytes} bytes"
    );

    // Carriage legality: every trained book round-trips through the
    // §3.2.1 codebook writer and parser field-for-field.
    for book in &trained {
        let bytes = write_codebook(book).expect("writes");
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(&parse_codebook(&mut r).expect("parses"), book);
    }
}

/// Sparse retraining prunes never-emitted entries. The planner adapts
/// (it only ever selects encodable cvals / Y values against the books
/// it is handed), so re-*planning* the training corpus against the
/// sparse books still succeeds, decodes to the identical curve
/// (`plan_floor1_y` depends only on the header), and costs no more
/// than the dense retrain on the training corpus.
#[test]
fn sparse_retrain_replans_and_costs_no_more_than_dense() {
    let header = training_header();
    let books = flat_books();
    let envelopes = corpus();

    let mut tallies = BookTallies::new(&books);
    let mut packets: Vec<Floor1Packet> = Vec::with_capacity(envelopes.len());
    for envelope in &envelopes {
        let packet = plan_floor1_packet(envelope, &header, &books).expect("plans");
        tally_floor1_packet(&mut tallies, &packet, &header).expect("tallies");
        packets.push(packet);
    }
    let dense = tallies.retrain(&books, 32, true).expect("dense retrain");
    let sparse = tallies.retrain(&books, 32, false).expect("sparse retrain");
    // The corpus does not exercise every Y value, so the sparse books
    // must genuinely prune.
    let pruned: usize = sparse
        .iter()
        .map(|b| b.codeword_lengths.iter().filter(|&&l| l == 0).count())
        .sum();
    assert!(pruned > 0, "sparse retrain must prune unseen entries");

    let mut dense_bytes = 0usize;
    let mut sparse_bytes = 0usize;
    for (envelope, packet) in envelopes.iter().zip(packets.iter()) {
        let dense_out = write_floor1_packet(packet, &header, &dense).expect("dense writes");
        let dense_curve = decode_curve(&header, &dense, &dense_out);
        dense_bytes += dense_out.len();

        // Re-plan against the sparse books: cval selection may differ
        // (a pruned master entry is no longer reachable), but the
        // fitted floor1_y — and therefore the curve — cannot.
        let replanned = plan_floor1_packet(envelope, &header, &sparse).expect("re-plans");
        assert_eq!(
            replanned.floor1_y, packet.floor1_y,
            "Y fit is book-independent"
        );
        let sparse_out = write_floor1_packet(&replanned, &header, &sparse).expect("sparse writes");
        let sparse_curve = decode_curve(&header, &sparse, &sparse_out);
        sparse_bytes += sparse_out.len();
        assert_eq!(sparse_curve, dense_curve, "same fitted curve either way");
    }
    assert!(
        sparse_bytes <= dense_bytes,
        "sparse books cannot cost more on their own training corpus: {sparse_bytes} vs {dense_bytes}"
    );
}

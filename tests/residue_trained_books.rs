//! Residue codebook-*content* training round-trip (Vorbis I §3.2.1 /
//! §8.6.2) — the residue analogue of `tests/floor1_trained_books.rs`.
//!
//! Retraining a codebook rewrites only its codeword lengths and
//! preserves its VQ lookup, so a `ResidueVectorPlan`'s entry indices
//! decode to the identical §8.6.2 residue vector under the trained
//! books. Training the classbook + value books on a corpus must
//! therefore satisfy the sharp contract:
//!
//! * every retrained body decodes to the **bit-identical** residue
//!   vector (same entries, same §3.2.1 VQ reconstructions), and
//! * the corpus serialises into **strictly fewer bytes** than under
//!   the untrained flat books.
//!
//! The chain, per corpus member: synthetic residual →
//! `plan_vector_residue` (from-spectrum classification + cascade) →
//! `tally_residue_plans` (the §8.6.2 emission tally: classwords
//! through the classbook, per-stage entries through the cascade's
//! value books) → after the whole corpus, `BookTallies::retrain` →
//! re-write every body with the trained books → decode with a
//! `ResidueDecoder` built over the trained books → compare vectors
//! and sizes. The three-class header (unused / coarse / coarse+fine)
//! makes the classword distribution genuinely non-uniform, so the
//! classbook itself trains, not just the value books.
//!
//! Fully synthetic: no Ogg framing, no `docs/` fixtures, so it runs in
//! standalone CI.

use oxideav_core::bits::BitReaderLsb;
use oxideav_vorbis::codebook::{parse_codebook, VorbisCodebook, VqLookup};
use oxideav_vorbis::residue::ResidueDecoder;
use oxideav_vorbis::setup::ResidueHeader;
use oxideav_vorbis::{
    design_entropy_codebook, plan_vector_residue, tally_residue_plans, write_codebook,
    write_residue_body, BookTallies, ResidueVectorPlan,
};

/// Blocksize the residue vectors live under (`n/2 = 64` scalars).
const BLOCKSIZE: usize = 128;
const HALF_N: usize = BLOCKSIZE / 2;
const PARTITION_SIZE: u32 = 8;

/// A Kraft-complete 1-D tessellation VQ value book: `2^length` entries
/// all at codeword length `length`, ladder `(e − half)·step` centred
/// on zero — the flat starting point the trainer must beat.
fn signed_value_book(length: u8, step: f32) -> VorbisCodebook {
    let entries: u32 = 1u32 << length;
    let half = entries / 2;
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::Tessellation {
            minimum_value: -(half as f32) * step,
            delta_value: step,
            value_bits: 8,
            sequence_p: false,
            multiplicands: (0..entries).collect(),
        },
    }
}

/// The stream codebook table: classbook (dimensions 2, 9 entries — one
/// per two-digit base-3 classword), a coarse value book, and a fine
/// value book. The classbook starts flat through the designer so it is
/// Kraft-complete.
fn flat_books() -> Vec<VorbisCodebook> {
    let mut classbook = design_entropy_codebook(9, 1, &[1u64; 9], 32, false).unwrap();
    classbook.dimensions = 2;
    vec![
        classbook,
        signed_value_book(4, 0.5),
        signed_value_book(4, 0.125),
    ]
}

/// Three-class format-1 residue header: class 0 'unused', class 1
/// coarse only (book 1 at pass 0), class 2 coarse + fine (books 1, 2).
fn residue_header() -> ResidueHeader {
    let unused: [Option<u8>; 8] = Default::default();
    let mut coarse: [Option<u8>; 8] = Default::default();
    coarse[0] = Some(1);
    let mut fine: [Option<u8>; 8] = Default::default();
    fine[0] = Some(1);
    fine[1] = Some(2);
    ResidueHeader {
        residue_type: 1,
        residue_begin: 0,
        residue_end: HALF_N as u32,
        partition_size: PARTITION_SIZE,
        classifications: 3,
        classbook: 0,
        cascade: vec![0, 0b01, 0b11],
        books: vec![unused, coarse, fine],
    }
}

/// The per-class value-book rows `plan_vector_residue` consumes,
/// resolved against a codebook table.
fn value_book_rows(books: &[VorbisCodebook]) -> Vec<[Option<&VorbisCodebook>; 8]> {
    let unused: [Option<&VorbisCodebook>; 8] = Default::default();
    let mut coarse: [Option<&VorbisCodebook>; 8] = Default::default();
    coarse[0] = Some(&books[1]);
    let mut fine: [Option<&VorbisCodebook>; 8] = Default::default();
    fine[0] = Some(&books[1]);
    fine[1] = Some(&books[2]);
    vec![unused, coarse, fine]
}

/// The synthetic residual corpus, sectioned so the from-spectrum
/// chooser exercises all three classes: a silent head (class 0 wins on
/// the distortion tie — cheapest), a section of values sitting exactly
/// on the coarse book's 0.5 ladder (class 1 and class 2 reconstruct it
/// equally, and the tie breaks toward the fewer-stage class 1), and an
/// off-grid oscillating tail only the coarse+fine cascade tracks
/// (class 2). Frame-to-frame variation keeps the symbol distributions
/// clustered but non-degenerate (trainable).
fn corpus() -> Vec<Vec<f32>> {
    (0..40usize)
        .map(|frame| {
            (0..HALF_N)
                .map(|i| {
                    if i < HALF_N / 4 {
                        0.0
                    } else if i < HALF_N / 2 {
                        // Exactly on the coarse ladder (multiples of 0.5).
                        0.5 * (((i + frame) % 4) as f32) - 1.0
                    } else {
                        let gain = 0.4 + 0.05 * (frame % 5) as f32;
                        let phase = (i as f32) * 0.61 + frame as f32 * 0.37;
                        gain * 2.4 * phase.sin()
                    }
                })
                .collect()
        })
        .collect()
}

fn decode_vector(header: &ResidueHeader, books: &[VorbisCodebook], body: &[u8]) -> Vec<f32> {
    let decoder = ResidueDecoder::new(header, books).expect("decoder builds");
    let mut reader = BitReaderLsb::new(body);
    let mut out = decoder
        .decode(&mut reader, BLOCKSIZE, &[false])
        .expect("decodes");
    assert_eq!(out.len(), 1);
    out.pop().unwrap()
}

/// The full training loop: plan → tally → retrain → re-write. Trained
/// books must decode every body to the bit-identical residue vector
/// while the corpus shrinks on the wire, and must stay carriage-legal
/// through the §3.2.1 codebook writer/parser.
#[test]
fn trained_residue_books_shrink_corpus_and_decode_identically() {
    let header = residue_header();
    let books = flat_books();
    let rows = value_book_rows(&books);
    let residuals = corpus();

    // Plan every vector from spectrum and tally its emissions.
    let mut tallies = BookTallies::new(&books);
    let mut plans: Vec<ResidueVectorPlan> = Vec::with_capacity(residuals.len());
    for residual in &residuals {
        let (classifications, partition_entries) =
            plan_vector_residue(residual, &rows, header.residue_type, header.partition_size)
                .expect("plans");
        let plan = ResidueVectorPlan {
            classifications,
            partition_entries,
        };
        tally_residue_plans(&mut tallies, std::slice::from_ref(&plan), &header, &books)
            .expect("tallies");
        plans.push(plan);
    }
    // The premise: all three books exercised, all three classes chosen.
    assert!(tallies.total(0) > 0, "classbook must be exercised");
    assert!(tallies.total(1) > 0, "coarse book must be exercised");
    assert!(tallies.total(2) > 0, "fine book must be exercised");
    let mut classes_seen = [false; 3];
    for plan in &plans {
        for &c in &plan.classifications {
            classes_seen[c as usize] = true;
        }
    }
    assert_eq!(classes_seen, [true; 3], "corpus must exercise every class");

    // Baseline: serialise + decode under the flat books.
    let mut flat_bytes = 0usize;
    let mut flat_vectors: Vec<Vec<f32>> = Vec::with_capacity(plans.len());
    for plan in &plans {
        let body = write_residue_body(
            std::slice::from_ref(plan),
            &header,
            &books,
            BLOCKSIZE,
            &[false],
        )
        .expect("writes");
        flat_vectors.push(decode_vector(&header, &books, &body));
        flat_bytes += body.len();
    }

    // Retrain (dense) and re-write the same plans.
    let trained = tallies.retrain(&books, 32, true).expect("retrains");
    for (i, (t, b)) in trained.iter().zip(books.iter()).enumerate() {
        assert_eq!(t.entries, b.entries, "book {i} keeps its shape");
        assert_eq!(t.lookup, b.lookup, "book {i} keeps its VQ lookup");
    }
    let mut trained_bytes = 0usize;
    for (plan, flat_vector) in plans.iter().zip(flat_vectors.iter()) {
        let body = write_residue_body(
            std::slice::from_ref(plan),
            &header,
            &trained,
            BLOCKSIZE,
            &[false],
        )
        .expect("writes");
        trained_bytes += body.len();
        let vector = decode_vector(&header, &trained, &body);
        assert_eq!(
            &vector, flat_vector,
            "trained books must decode the bit-identical §8.6.2 vector"
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

/// A 'do not decode' channel next to a decoded one: the tally skips
/// the empty plan, the trained books still round-trip the two-channel
/// body bit-identically, and the corpus still shrinks.
#[test]
fn trained_residue_books_with_do_not_decode_channel() {
    let header = residue_header();
    let books = flat_books();
    let rows = value_book_rows(&books);
    let residuals = corpus();
    let do_not_decode = [false, true];

    let mut tallies = BookTallies::new(&books);
    let mut plan_pairs: Vec<[ResidueVectorPlan; 2]> = Vec::with_capacity(residuals.len());
    for residual in &residuals {
        let (classifications, partition_entries) =
            plan_vector_residue(residual, &rows, header.residue_type, header.partition_size)
                .expect("plans");
        let decoded = ResidueVectorPlan {
            classifications,
            partition_entries,
        };
        let dnd = ResidueVectorPlan {
            classifications: vec![],
            partition_entries: vec![],
        };
        tally_residue_plans(
            &mut tallies,
            &[decoded.clone(), dnd.clone()],
            &header,
            &books,
        )
        .expect("tallies");
        plan_pairs.push([decoded, dnd]);
    }

    let trained = tallies.retrain(&books, 32, true).expect("retrains");
    let mut flat_bytes = 0usize;
    let mut trained_bytes = 0usize;
    for pair in &plan_pairs {
        let flat_body =
            write_residue_body(pair, &header, &books, BLOCKSIZE, &do_not_decode).expect("writes");
        let trained_body =
            write_residue_body(pair, &header, &trained, BLOCKSIZE, &do_not_decode).expect("writes");
        flat_bytes += flat_body.len();
        trained_bytes += trained_body.len();

        let flat_dec = ResidueDecoder::new(&header, &books).expect("builds");
        let trained_dec = ResidueDecoder::new(&header, &trained).expect("builds");
        let mut r1 = BitReaderLsb::new(&flat_body);
        let mut r2 = BitReaderLsb::new(&trained_body);
        let v1 = flat_dec.decode(&mut r1, BLOCKSIZE, &do_not_decode).unwrap();
        let v2 = trained_dec
            .decode(&mut r2, BLOCKSIZE, &do_not_decode)
            .unwrap();
        assert_eq!(v1, v2, "identical two-channel §8.6.2 reconstruction");
    }
    assert!(
        trained_bytes < flat_bytes,
        "training must shrink the corpus: {trained_bytes} vs {flat_bytes} bytes"
    );
}

/// Closed-loop rate-aware training (`train_residue_books_rd`):
/// alternating the §8.6.2 rate-distortion planner with the codebook
/// retrainer descends the corpus Lagrangian monotonically, reaches a
/// fixed point, spends strictly fewer total codeword bits than the
/// first (flat-book) pass, and its final plans serialise + decode
/// through the real §8.6.2 writer/decoder under the trained books.
#[test]
fn rd_training_loop_descends_converges_and_round_trips() {
    let header = residue_header();
    let books = flat_books();
    let residuals = corpus();

    let outcome = oxideav_vorbis::train_residue_books_rd(&residuals, &header, &books, 0.75, 12)
        .expect("trains");

    // Monotone descent of the shared objective.
    assert!(
        outcome.lagrangian_per_iteration.len() >= 2,
        "the loop must run at least two passes on a fresh corpus"
    );
    for w in outcome.lagrangian_per_iteration.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-9,
            "Lagrangian must never rise: {:?}",
            outcome.lagrangian_per_iteration
        );
    }
    assert!(outcome.converged, "the loop must reach a fixed point");

    // The rate side: strictly fewer total codeword bits than the
    // flat-book first pass.
    let first = outcome.total_bits_per_iteration[0];
    let last = *outcome.total_bits_per_iteration.last().unwrap();
    assert!(
        last < first,
        "training must reduce total bits: {last} vs {first}"
    );

    // The final plans round-trip through the real writer + decoder
    // under the trained (sparse) books.
    for plan in &outcome.plans {
        let body = write_residue_body(
            std::slice::from_ref(plan),
            &header,
            &outcome.codebooks,
            BLOCKSIZE,
            &[false],
        )
        .expect("writes under trained books");
        let vector = decode_vector(&header, &outcome.codebooks, &body);
        assert!(
            vector.iter().all(|v| v.is_finite()),
            "trained-book decode must be finite"
        );
    }

    // And the trained books stay carriage-legal.
    for book in &outcome.codebooks {
        let bytes = write_codebook(book).expect("writes");
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(&parse_codebook(&mut r).expect("parses"), book);
    }
}

/// `lambda = 0` reduces the plan step to the distortion-only chooser:
/// the loop still converges, and because the distortion chooser's
/// decisions are price-independent, the plans are already stable after
/// the first retrain (two measured iterations).
#[test]
fn rd_training_loop_lambda_zero_is_distortion_stable() {
    let header = residue_header();
    let books = flat_books();
    let residuals = corpus();

    let outcome = oxideav_vorbis::train_residue_books_rd(&residuals, &header, &books, 0.0, 6)
        .expect("trains");
    assert!(outcome.converged);
    assert_eq!(
        outcome.lagrangian_per_iteration.len(),
        2,
        "distortion-only choices cannot change under re-pricing"
    );
    // At lambda = 0 the Lagrangian is pure distortion — identical
    // across the two passes.
    let l = &outcome.lagrangian_per_iteration;
    assert!((l[0] - l[1]).abs() <= 1e-9 * l[0].abs().max(1.0));
    // The retrain still pays off on the wire.
    assert!(outcome.total_bits_per_iteration[1] < outcome.total_bits_per_iteration[0]);
}

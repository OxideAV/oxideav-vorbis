//! Whole-stream trained-books round-trip (Vorbis I §3.2.1 / §4.3) —
//! the capstone over the codebook-content design stack.
//!
//! The per-subsystem trainer suites (`floor1_trained_books`,
//! `residue_trained_books`, `floor0_trained_books`) each pin one tally
//! walk against its own writer. This suite trains a **complete
//! stream**: a 20-frame mono PCM corpus is encoded through the full
//! §4.3 audio-packet writer against a real `VorbisSetupHeader`
//! (floor-1 plus a three-class residue plus four codebooks), every
//! packet's floor and residue emissions are tallied together into one
//! `BookTallies`, the whole codebook table is retrained, and the same
//! packets are re-serialised under a setup header carrying the
//! trained books. The contract:
//!
//! * every retrained §4.3 packet decodes to the **bit-identical**
//!   windowed PCM frame (floor-1 post coding is lossless given the
//!   fitted targets, and retraining preserves every VQ lookup — the
//!   whole §4.3.2–§4.3.7 numeric pipeline sees identical inputs);
//! * the audio corpus serialises into **strictly fewer bytes**;
//! * the trained setup header itself round-trips through
//!   `write_setup_header` → `parse_setup_header` field-for-field, so
//!   the trained stream is carriage-complete (headers + audio).
//!
//! Fully synthetic: no Ogg framing, no `docs/` fixtures, so it runs in
//! standalone CI.

use oxideav_core::bits::BitReaderLsb;
use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::encoder::write_setup_header;
use oxideav_vorbis::floor1::Floor1Decoder;
use oxideav_vorbis::setup::{
    parse_setup_header, Floor1Class, Floor1Header, FloorHeader, FloorKind, MappingHeader,
    MappingSubmap, ModeHeader, ResidueHeader, VorbisSetupHeader,
};
use oxideav_vorbis::{
    apply_window_and_mdct_vec, decode_audio_packet_windowed, plan_floor1_envelope, plan_floor1_y,
    plan_vector_residue, tally_floor1_packet, tally_residue_plans, write_audio_packet,
    AudioChannelFloor, AudioDecoderState, AudioPacketHeader, BookTallies, Floor1Packet,
    ResidueVectorPlan, WindowedPacketOutcome,
};

const N: usize = 256;
const HALF_N: usize = N / 2;
const PARTITIONS: usize = 16;
const PARTITION_SIZE: u32 = (HALF_N / PARTITIONS) as u32;
const FRAMES: usize = 20;

/// §4.3.1 Vorbis window of length `n`.
fn vorbis_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let inner = (std::f32::consts::PI / n as f32 * (i as f32 + 0.5)).sin();
            (std::f32::consts::FRAC_PI_2 * inner * inner).sin()
        })
        .collect()
}

/// A Kraft-complete 1-D tessellation VQ value book on a dyadic ladder
/// (exactly §9.2.2-packable, so the setup header can carry it).
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

/// A balanced 1-D scalar book (no VQ lookup).
fn scalar_book(entries: u32, length: u8, dimensions: u16) -> VorbisCodebook {
    VorbisCodebook {
        dimensions,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
}

/// The shared floor-1 header: seven interior posts, one
/// `subclasses = 0` class reading its Y values from codebook 0.
fn floor_header() -> Floor1Header {
    let interior: Vec<u32> = [
        HALF_N / 8,
        HALF_N / 4,
        (3 * HALF_N) / 8,
        HALF_N / 2,
        (5 * HALF_N) / 8,
        (3 * HALF_N) / 4,
        (7 * HALF_N) / 8,
    ]
    .iter()
    .map(|&p| p as u32)
    .collect();
    let mut rangebits = 1u8;
    while (1u32 << rangebits) < HALF_N as u32 {
        rangebits += 1;
    }
    Floor1Header {
        partitions: 1,
        partition_class_list: vec![0],
        classes: vec![Floor1Class {
            dimensions: interior.len() as u8,
            subclasses: 0,
            masterbook: None,
            subclass_books: vec![Some(0)],
        }],
        multiplier: 1,
        rangebits,
        x_list: interior,
    }
}

/// The mono setup: codebooks `[floor Y, classbook, coarse, fine]`, one
/// floor, one three-class residue, one mapping, one mode.
fn mono_setup(codebooks: Vec<VorbisCodebook>) -> VorbisSetupHeader {
    let unused: [Option<u8>; 8] = Default::default();
    let mut coarse_stages: [Option<u8>; 8] = Default::default();
    coarse_stages[0] = Some(2);
    let mut fine_stages: [Option<u8>; 8] = Default::default();
    fine_stages[0] = Some(2);
    fine_stages[1] = Some(3);

    VorbisSetupHeader {
        codebooks,
        // §4.2.4 carries at least one zero time-transform placeholder.
        time_placeholders: vec![0],
        floors: vec![FloorHeader {
            floor_type: 1,
            kind: FloorKind::Type1(floor_header()),
        }],
        residues: vec![ResidueHeader {
            residue_type: 1,
            residue_begin: 0,
            residue_end: HALF_N as u32,
            partition_size: PARTITION_SIZE,
            classifications: 3,
            classbook: 1,
            cascade: vec![0, 1 << 0, (1 << 0) | (1 << 1)],
            books: vec![unused, coarse_stages, fine_stages],
        }],
        mappings: vec![MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: Vec::new(),
            mux: Vec::new(),
            submap_configs: vec![MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        }],
        modes: vec![ModeHeader {
            blockflag: false,
            windowtype: 0,
            transformtype: 0,
            mapping: 0,
        }],
        framing_flag: true,
    }
}

/// Frame `k` of the PCM corpus: a harmonic mix whose component
/// amplitudes drift frame to frame — related spectra, so the tallied
/// symbol distributions cluster.
fn pcm_frame(k: usize) -> Vec<f32> {
    let a = 0.7 + 0.02 * (k % 6) as f32;
    let b = 0.45 - 0.015 * (k % 5) as f32;
    let c = 0.20 + 0.01 * (k % 4) as f32;
    (0..N)
        .map(|i| {
            let t = i as f32 / N as f32;
            let tau = 2.0 * std::f32::consts::PI;
            a * (tau * 2.0 * t).sin()
                + b * (tau * 5.0 * t).sin()
                + c * (tau * 11.0 * t).cos()
                + 0.05 * (tau * 23.0 * t + k as f32 * 0.3).sin()
        })
        .collect()
}

/// A descending exponential magnitude envelope bounding `|X|` from
/// above.
fn magnitude_envelope(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let peak = x.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let n_f = (n.max(1) - 1).max(1) as f32;
    (0..n)
        .map(|k| {
            let frac = k as f32 / n_f;
            (peak * (256.0f32).powf(-frac))
                .max(x[k].abs())
                .max(peak / 256.0)
        })
        .collect()
}

/// One planned frame: the fitted floor packet, the residue plan, and
/// the analysis spectrum (for the decode reference).
struct PlannedFrame {
    floor: Floor1Packet,
    residue: ResidueVectorPlan,
}

fn write_frame(setup: &VorbisSetupHeader, frame: &PlannedFrame) -> Vec<u8> {
    let header = AudioPacketHeader {
        mode_number: 0,
        blockflag: false,
        n: N,
        previous_window_flag: false,
        next_window_flag: false,
    };
    let floors = vec![AudioChannelFloor::Type1(frame.floor.clone())];
    let residue_plans = vec![vec![frame.residue.clone()]];
    write_audio_packet(&header, setup, N, N, 1, &floors, &residue_plans)
        .expect("audio packet serialises")
}

fn decode_frame(setup: &VorbisSetupHeader, bytes: &[u8]) -> Vec<f32> {
    let state = AudioDecoderState::new(setup).expect("decoder state");
    let mut reader = BitReaderLsb::new(bytes);
    let outcome = decode_audio_packet_windowed(&mut reader, setup, &state, 1, N, N, 1.0)
        .expect("audio packet decodes");
    match outcome {
        WindowedPacketOutcome::Windowed { mut frames, .. } => frames.remove(0),
        WindowedPacketOutcome::ZeroedWindowed { .. } => {
            panic!("nonzero floor + residue must not zero the packet")
        }
    }
}

#[test]
fn trained_stream_shrinks_and_decodes_bit_identically() {
    let fh = floor_header();
    let window = vorbis_window(N);

    // ---- pass 1: analyse every frame to fix the value-book scale. ----
    let mut spectra: Vec<Vec<f32>> = Vec::with_capacity(FRAMES);
    for k in 0..FRAMES {
        let mut pcm = pcm_frame(k);
        let x = apply_window_and_mdct_vec(&mut pcm, &window, 1.0).expect("forward MDCT");
        spectra.push(x);
    }

    // Fit the floors first (they need only the header), then derive a
    // global residue-target bound so one dyadic book scale serves the
    // whole corpus.
    let floor_book = scalar_book(256, 8, 1);
    let floor_decoder =
        Floor1Decoder::new(&fh, std::slice::from_ref(&floor_book)).expect("floor decoder builds");
    let mut floors: Vec<Floor1Packet> = Vec::with_capacity(FRAMES);
    let mut residue_targets: Vec<Vec<f32>> = Vec::with_capacity(FRAMES);
    let mut max_abs = 0.0f32;
    for x in &spectra {
        let envelope = magnitude_envelope(x);
        let posts = plan_floor1_envelope(&envelope, &fh).expect("envelope fit");
        let floor1_y = plan_floor1_y(&posts, &fh).expect("amplitude wrap");
        let rendered = floor_decoder.render_curve(&floor1_y, HALF_N);
        let target: Vec<f32> = x
            .iter()
            .zip(rendered.iter())
            .map(|(&xv, &fv)| xv / fv)
            .collect();
        max_abs = target.iter().fold(max_abs, |m, &v| m.max(v.abs()));
        residue_targets.push(target);
        floors.push(Floor1Packet {
            nonzero: true,
            floor1_y,
            partition_cvals: vec![0],
        });
    }
    assert!(max_abs > 0.0);
    // Dyadic steps (exactly §9.2.2-packable) covering ±max_abs.
    let mut coarse_step = 0.015625f32;
    while coarse_step * 24.0 < max_abs {
        coarse_step *= 2.0;
    }
    let fine_step = coarse_step / 8.0;

    let flat_books = vec![
        floor_book,
        scalar_book(9, 4, 2), // classbook: 3 classes ^ 2 classwords
        signed_value_book(6, coarse_step),
        signed_value_book(6, fine_step),
    ];
    // Legalise the 9-entry classbook (9 codewords cannot all be one
    // length): rebuild its lengths through the designer, flat freqs.
    let flat_books = {
        let mut books = flat_books;
        books[1] = oxideav_vorbis::redesign_codebook(&books[1], &[1u64; 9], 32, false)
            .expect("classbook legalises");
        books
    };
    let flat_setup = mono_setup(flat_books.clone());

    // ---- pass 2: plan residues, write + decode the flat stream. ----
    let empty: [Option<&VorbisCodebook>; 8] = Default::default();
    let mut row_coarse: [Option<&VorbisCodebook>; 8] = Default::default();
    row_coarse[0] = Some(&flat_books[2]);
    let mut row_fine: [Option<&VorbisCodebook>; 8] = Default::default();
    row_fine[0] = Some(&flat_books[2]);
    row_fine[1] = Some(&flat_books[3]);
    let value_rows = vec![empty, row_coarse, row_fine];

    let mut tallies = BookTallies::new(&flat_books);
    let mut frames: Vec<PlannedFrame> = Vec::with_capacity(FRAMES);
    for (floor, target) in floors.iter().zip(residue_targets.iter()) {
        let (classifications, partition_entries) =
            plan_vector_residue(target, &value_rows, 1, PARTITION_SIZE).expect("residue plan");
        let residue = ResidueVectorPlan {
            classifications,
            partition_entries,
        };
        // Tally both subsystems into ONE tally table.
        tally_floor1_packet(&mut tallies, floor, &fh).expect("floor tally");
        tally_residue_plans(
            &mut tallies,
            std::slice::from_ref(&residue),
            &flat_setup.residues[0],
            &flat_books,
        )
        .expect("residue tally");
        frames.push(PlannedFrame {
            floor: floor.clone(),
            residue,
        });
    }
    // Every book in the stream must have been exercised.
    for b in 0..4 {
        assert!(tallies.total(b) > 0, "book {b} must be exercised");
    }

    let mut flat_bytes = 0usize;
    let mut flat_frames: Vec<Vec<f32>> = Vec::with_capacity(FRAMES);
    for frame in &frames {
        let bytes = write_frame(&flat_setup, frame);
        flat_frames.push(decode_frame(&flat_setup, &bytes));
        flat_bytes += bytes.len();
    }

    // ---- train: one retrain over the whole stream's tallies. ----
    let trained_books = tallies.retrain(&flat_books, 32, true).expect("retrains");
    for (i, (t, b)) in trained_books.iter().zip(flat_books.iter()).enumerate() {
        assert_eq!(t.entries, b.entries, "book {i} keeps its shape");
        assert_eq!(t.lookup, b.lookup, "book {i} keeps its VQ lookup");
        assert_ne!(
            t.codeword_lengths, b.codeword_lengths,
            "book {i} must actually be re-optimised"
        );
    }
    let trained_setup = mono_setup(trained_books);

    // ---- the contract: identical PCM, fewer bytes. ----
    let mut trained_bytes = 0usize;
    for (frame, flat_frame) in frames.iter().zip(flat_frames.iter()) {
        let bytes = write_frame(&trained_setup, frame);
        trained_bytes += bytes.len();
        let decoded = decode_frame(&trained_setup, &bytes);
        assert_eq!(
            &decoded, flat_frame,
            "trained stream must decode to the bit-identical windowed frame"
        );
    }
    assert!(
        trained_bytes < flat_bytes,
        "training must shrink the audio corpus: {trained_bytes} vs {flat_bytes} bytes"
    );

    // ---- carriage: the trained setup header round-trips whole. ----
    let header_bytes = write_setup_header(&trained_setup, 1).expect("setup header writes");
    let reparsed = parse_setup_header(&header_bytes, 1).expect("setup header parses");
    assert_eq!(
        reparsed, trained_setup,
        "the trained stream's setup header must round-trip field-for-field"
    );

    // The flat setup header must round-trip too (control: the trained
    // books are not what makes the header legal).
    let flat_header_bytes = write_setup_header(&flat_setup, 1).expect("flat setup writes");
    assert_eq!(
        parse_setup_header(&flat_header_bytes, 1).expect("flat setup parses"),
        flat_setup
    );
}

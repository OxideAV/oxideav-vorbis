//! Quality-knob → rate/fidelity curve measurement, and bit-budget
//! targeting, over the full psychoacoustic encode stack.
//!
//! `EncoderTuning::from_quality` claims one scalar drives every
//! quality lever coherently; this suite proves it on a real
//! multi-frame encode → decode loop and measures the resulting curve:
//!
//! * **Rate curve** — total §4.3 packet bytes across an 8-frame
//!   corpus are non-decreasing in `q`;
//! * **Fidelity curve** — the spectral SNR (decoded spectrum against
//!   the analysis spectrum) is non-decreasing in `q`, and the mean
//!   noise-to-mask ratio against the *nominal* masking analysis is
//!   non-increasing;
//! * **Bit-budget targeting** — `solve_lambda_for_bits` over the real
//!   stream-rate-vs-lambda curve lands the corpus within a byte
//!   budget, monotonically in the budget.
//!
//! Each quality point designs its floor-1 header from the corpus'
//! first frame at that tuning (post budget + margin are levers), then
//! fits per-frame floor posts and plans the NMR-weighted residue per
//! frame — the same chain `tests/psy_encode_roundtrip.rs` pins
//! qualitatively, here swept and measured. Fully synthetic.

use oxideav_core::bits::BitReaderLsb;
use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::floor1::Floor1Decoder;
use oxideav_vorbis::setup::{
    Floor1Class, Floor1Header, FloorHeader, FloorKind, MappingHeader, MappingSubmap, ModeHeader,
    ResidueHeader, VorbisSetupHeader,
};
use oxideav_vorbis::{
    compute_masking, decode_audio_packet_pre_imdct, design_floor1_header, plan_floor1_envelope,
    plan_floor1_y, plan_psy_floor_envelope, plan_vector_residue_rd_weighted,
    residue_partition_weights, solve_lambda_for_bits, write_audio_packet, AudioChannelFloor,
    AudioDecoderState, AudioPacketHeader, AudioPacketOutcome, EncoderTuning, Floor1Packet,
    PsyConfig, ResidueEncodeError, ResidueVectorPlan,
};

const N: usize = 512;
const HALF_N: usize = N / 2;
const PARTITION_SIZE: u32 = 16;
const SAMPLE_RATE: u32 = 44_100;
const FRAMES: usize = 8;

/// Frame `f` of the corpus: the three tones breathe in level, the
/// mid-band pedestal and high hash re-roll per frame.
fn corpus_frame(f: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; HALF_N];
    let g = 0.8 + 0.05 * (f % 5) as f32;
    x[11] = 0.60 * g;
    x[23] = -0.40 * g;
    x[34] = 0.35 * g;
    for (k, v) in x.iter_mut().enumerate().take(120).skip(40) {
        let h = ((k + 31 * f).wrapping_mul(2_654_435_761) >> 9) & 0xff;
        *v = (h as f32 / 128.0 - 1.0) * 0.002;
    }
    for (k, v) in x.iter_mut().enumerate().skip(HALF_N / 2) {
        let h = ((k + 17 * f).wrapping_mul(2_654_435_761) >> 7) & 0xff;
        *v = (h as f32 / 128.0 - 1.0) * 0.002;
    }
    x
}

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

fn scalar_book(entries: u32, length: u8) -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
}

fn class_catalogue() -> Vec<Floor1Class> {
    [1u8, 2, 4]
        .iter()
        .map(|&d| Floor1Class {
            dimensions: d,
            subclasses: 0,
            masterbook: None,
            subclass_books: vec![Some(0)],
        })
        .collect()
}

fn mono_setup(
    floor_header: Floor1Header,
    coarse: VorbisCodebook,
    fine: VorbisCodebook,
) -> VorbisSetupHeader {
    let floor = FloorHeader {
        floor_type: 1,
        kind: FloorKind::Type1(floor_header),
    };
    let mut stages: [Option<u8>; 8] = Default::default();
    stages[0] = Some(2);
    stages[1] = Some(3);
    let residue = ResidueHeader {
        residue_type: 1,
        residue_begin: 0,
        residue_end: HALF_N as u32,
        partition_size: PARTITION_SIZE,
        classifications: 2,
        classbook: 1,
        cascade: vec![0, 0b11],
        books: vec![Default::default(), stages],
    };
    VorbisSetupHeader {
        codebooks: vec![scalar_book(256, 8), scalar_book(2, 1), coarse, fine],
        time_placeholders: vec![0],
        floors: vec![floor],
        residues: vec![residue],
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

/// One frame's prepared analysis under a tuning: the floor posts, the
/// rendered floor, the residue target, and the perceptual weights.
struct PreparedFrame {
    x: Vec<f32>,
    floor1_y: Vec<u32>,
    target: Vec<f32>,
    weights: Vec<f64>,
}

/// The whole corpus prepared under one tuning: a shared designed
/// header + value books, per-frame fits.
struct PreparedCorpus {
    setup: VorbisSetupHeader,
    header: Floor1Header,
    frames: Vec<PreparedFrame>,
}

fn prepare_corpus(tuning: &EncoderTuning) -> PreparedCorpus {
    let config = PsyConfig {
        threshold_offset_db: tuning.threshold_offset_db,
        ..PsyConfig::new(SAMPLE_RATE)
    };

    // Design the stream's floor header from frame 0 at this tuning.
    let x0 = corpus_frame(0);
    let masking0 = compute_masking(&x0, &config).expect("masking");
    let env0 =
        plan_psy_floor_envelope(&x0, &masking0, tuning.floor_smooth_radius).expect("envelope");
    let classes = class_catalogue();
    let header = design_floor1_header(&env0, tuning.floor_post_budget, 0.0, 1, &classes)
        .expect("floor-1 header designs");
    let floor_book = scalar_book(256, 8);
    let decoder = Floor1Decoder::new(&header, std::slice::from_ref(&floor_book))
        .expect("designed floor decoder builds");

    // Fit every frame against the shared header.
    let mut frames = Vec::with_capacity(FRAMES);
    let mut max_abs = 0.0f32;
    for f in 0..FRAMES {
        let x = corpus_frame(f);
        let masking = compute_masking(&x, &config).expect("masking");
        let env =
            plan_psy_floor_envelope(&x, &masking, tuning.floor_smooth_radius).expect("envelope");
        let posts = plan_floor1_envelope(&env, &header).expect("envelope fit");
        let floor1_y = plan_floor1_y(&posts, &header).expect("amplitude wrap");
        let rendered = decoder.render_curve(&floor1_y, HALF_N);
        let target: Vec<f32> = x
            .iter()
            .zip(rendered.iter())
            .map(|(&xv, &fv)| xv / fv)
            .collect();
        for &t in &target {
            max_abs = max_abs.max(t.abs());
        }
        let weights = residue_partition_weights(&rendered, &masking, 0, HALF_N, PARTITION_SIZE)
            .expect("weights");
        frames.push(PreparedFrame {
            x,
            floor1_y,
            target,
            weights,
        });
    }

    let coarse = signed_value_book(6, max_abs / 24.0);
    let fine = signed_value_book(6, max_abs / 192.0);
    let setup = mono_setup(header.clone(), coarse, fine);
    PreparedCorpus {
        setup,
        header,
        frames,
    }
}

/// Encode + decode the prepared corpus at one lambda: total packet
/// bytes plus the concatenated (reference, decoded) spectra.
struct CorpusPoint {
    bytes: usize,
    reference: Vec<f32>,
    decoded: Vec<f32>,
}

fn encode_corpus(corpus: &PreparedCorpus, lambda: f64) -> Result<CorpusPoint, ResidueEncodeError> {
    let coarse = &corpus.setup.codebooks[2];
    let fine = &corpus.setup.codebooks[3];
    let empty_row: [Option<&VorbisCodebook>; 8] = [None; 8];
    let mut cascade_row: [Option<&VorbisCodebook>; 8] = [None; 8];
    cascade_row[0] = Some(coarse);
    cascade_row[1] = Some(fine);
    let value_books = [empty_row, cascade_row];

    let state = AudioDecoderState::new(&corpus.setup).expect("decoder state builds");
    let mut bytes_total = 0usize;
    let mut reference = Vec::with_capacity(FRAMES * HALF_N);
    let mut decoded_all = Vec::with_capacity(FRAMES * HALF_N);

    for frame in &corpus.frames {
        let scored = plan_vector_residue_rd_weighted(
            &frame.target,
            &value_books,
            1,
            PARTITION_SIZE,
            lambda,
            &frame.weights,
        )?;
        let floors = vec![AudioChannelFloor::Type1(Floor1Packet {
            nonzero: true,
            floor1_y: frame.floor1_y.clone(),
            partition_cvals: vec![0u32; corpus.header.partition_class_list.len()],
        })];
        let residue_plans = vec![vec![ResidueVectorPlan {
            classifications: scored.classifications,
            partition_entries: scored.partition_entries,
        }]];
        let header = AudioPacketHeader {
            mode_number: 0,
            blockflag: false,
            n: N,
            previous_window_flag: false,
            next_window_flag: false,
        };
        let bytes = write_audio_packet(&header, &corpus.setup, N, 2048, 1, &floors, &residue_plans)
            .expect("audio packet serialises");
        bytes_total += bytes.len();

        let mut reader = BitReaderLsb::new(&bytes);
        let outcome = decode_audio_packet_pre_imdct(&mut reader, &corpus.setup, &state, 1, N, 2048)
            .expect("audio packet decodes");
        match outcome {
            AudioPacketOutcome::PreImdct { spectra, .. } => {
                decoded_all.extend_from_slice(&spectra[0]);
            }
            other => panic!("expected PreImdct outcome, got {other:?}"),
        }
        reference.extend_from_slice(&frame.x);
    }

    Ok(CorpusPoint {
        bytes: bytes_total,
        reference,
        decoded: decoded_all,
    })
}

fn snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&r, &d) in reference.iter().zip(decoded) {
        sig += f64::from(r) * f64::from(r);
        let e = f64::from(r) - f64::from(d);
        err += e * e;
    }
    if err == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

/// Mean NMR of the concatenated corpus against the per-frame nominal
/// (offset-0) masking analyses.
fn corpus_nmr(reference: &[f32], decoded: &[f32]) -> f64 {
    let nominal = PsyConfig::new(SAMPLE_RATE);
    let mut acc = 0.0f64;
    for f in 0..FRAMES {
        let x = &reference[f * HALF_N..(f + 1) * HALF_N];
        let d = &decoded[f * HALF_N..(f + 1) * HALF_N];
        let masking = compute_masking(x, &nominal).expect("nominal masking");
        for k in 0..HALF_N {
            let noise = f64::from(x[k]) - f64::from(d[k]);
            let t = f64::from(masking.threshold[k]);
            acc += (noise / t) * (noise / t);
        }
    }
    acc / (FRAMES * HALF_N) as f64
}

#[test]
fn quality_knob_traces_a_monotone_rate_and_fidelity_curve() {
    let qualities = [0.0f32, 0.25, 0.5, 0.75, 1.0];
    let mut bytes_curve = Vec::new();
    let mut snr_curve = Vec::new();
    let mut nmr_curve = Vec::new();

    for &q in &qualities {
        let tuning = EncoderTuning::from_quality(q).expect("tuning");
        let corpus = prepare_corpus(&tuning);
        let point = encode_corpus(&corpus, tuning.lambda).expect("encodes");
        let snr = snr_db(&point.reference, &point.decoded);
        let nmr = corpus_nmr(&point.reference, &point.decoded);
        eprintln!(
            "q {q:.2}: {} B total ({} frames), SNR {snr:.2} dB, NMR {nmr:.5}",
            point.bytes, FRAMES
        );
        bytes_curve.push(point.bytes);
        snr_curve.push(snr);
        nmr_curve.push(nmr);
    }

    // Rate: non-decreasing in q (small floor-packet jitter allowed —
    // the envelope shifts with the margin lever).
    for w in bytes_curve.windows(2) {
        assert!(
            w[0] <= w[1] + 16,
            "rate must not fall as quality rises: {bytes_curve:?}"
        );
    }
    // Fidelity: SNR non-decreasing, NMR non-increasing (small slack).
    for w in snr_curve.windows(2) {
        assert!(
            w[1] >= w[0] - 0.5,
            "SNR must not fall as quality rises: {snr_curve:?}"
        );
    }
    for w in nmr_curve.windows(2) {
        assert!(
            w[1] <= w[0] * 1.05 + 1e-9,
            "NMR must not rise as quality rises: {nmr_curve:?}"
        );
    }
    // The sweep is real: the extremes differ decisively in rate and
    // in both fidelity metrics.
    assert!(
        bytes_curve[4] as f64 >= bytes_curve[0] as f64 * 1.5,
        "q=1 must spend well over q=0: {bytes_curve:?}"
    );
    assert!(
        snr_curve[4] >= snr_curve[0] + 6.0,
        "q=1 must clear q=0 by >= 6 dB SNR: {snr_curve:?}"
    );
    assert!(
        nmr_curve[4] < nmr_curve[0] * 0.25,
        "q=1 must at least quarter the NMR of q=0: {nmr_curve:?}"
    );
    // The top of the knob is audibly transparent under the model.
    assert!(
        nmr_curve[4] < 1.0,
        "q=1 mean NMR {} must sit below 0 dB",
        nmr_curve[4]
    );
}

/// Encode + decode the corpus from ready-made plans (the trainer's
/// output) under an arbitrary setup: total bytes + decoded spectra.
fn encode_with_plans(
    corpus: &PreparedCorpus,
    setup: &VorbisSetupHeader,
    plans: &[ResidueVectorPlan],
) -> CorpusPoint {
    let state = AudioDecoderState::new(setup).expect("decoder state builds");
    let mut bytes_total = 0usize;
    let mut reference = Vec::with_capacity(FRAMES * HALF_N);
    let mut decoded_all = Vec::with_capacity(FRAMES * HALF_N);
    for (frame, plan) in corpus.frames.iter().zip(plans) {
        let floors = vec![AudioChannelFloor::Type1(Floor1Packet {
            nonzero: true,
            floor1_y: frame.floor1_y.clone(),
            partition_cvals: vec![0u32; corpus.header.partition_class_list.len()],
        })];
        let residue_plans = vec![vec![plan.clone()]];
        let header = AudioPacketHeader {
            mode_number: 0,
            blockflag: false,
            n: N,
            previous_window_flag: false,
            next_window_flag: false,
        };
        let bytes = write_audio_packet(&header, setup, N, 2048, 1, &floors, &residue_plans)
            .expect("audio packet serialises");
        bytes_total += bytes.len();
        let mut reader = BitReaderLsb::new(&bytes);
        let outcome = decode_audio_packet_pre_imdct(&mut reader, setup, &state, 1, N, 2048)
            .expect("audio packet decodes");
        match outcome {
            AudioPacketOutcome::PreImdct { spectra, .. } => {
                decoded_all.extend_from_slice(&spectra[0]);
            }
            other => panic!("expected PreImdct outcome, got {other:?}"),
        }
        reference.extend_from_slice(&frame.x);
    }
    CorpusPoint {
        bytes: bytes_total,
        reference,
        decoded: decoded_all,
    }
}

#[test]
fn ladder_trained_books_cut_the_rate_of_the_psy_stream() {
    // The round's two stacks compose: the psy floor + quality tuning
    // fix the residue targets; train_residue_books_rd_ladder then
    // retrains the seed value books (lengths AND reconstruction
    // values) on those targets. The trained stream must serialise into
    // fewer bytes at no fidelity loss beyond the Lagrangian trade,
    // stay §4.2.4 carriage-legal, and keep the model-transparent NMR.
    use oxideav_vorbis::encoder::write_setup_header;
    use oxideav_vorbis::setup::parse_setup_header;
    use oxideav_vorbis::train_residue_books_rd_ladder;

    let tuning = EncoderTuning::from_quality(0.5).expect("tuning");
    let corpus = prepare_corpus(&tuning);
    let residuals: Vec<Vec<f32>> = corpus.frames.iter().map(|f| f.target.clone()).collect();
    let residue_header = corpus.setup.residues[0].clone();

    // Baseline: unweighted RD plans under the seed books (the trainer
    // plans unweighted, so both sides use the same chooser).
    let seed_outcome = train_residue_books_rd_ladder(
        &residuals,
        &residue_header,
        &corpus.setup.codebooks,
        tuning.lambda,
        1, // one iteration = plan-only measurement pass + one train step
    )
    .expect("seed pass");
    let baseline_plans = seed_outcome.plans.clone();
    let baseline = encode_with_plans(&corpus, &corpus.setup, &baseline_plans);

    // Trained: the full closed loop.
    let trained = train_residue_books_rd_ladder(
        &residuals,
        &residue_header,
        &corpus.setup.codebooks,
        tuning.lambda,
        10,
    )
    .expect("trains");
    for w in trained.lagrangian_per_iteration.windows(2) {
        assert!(w[1] <= w[0] + 1e-9, "monotone descent");
    }
    let mut trained_setup = corpus.setup.clone();
    trained_setup.codebooks = trained.codebooks.clone();
    let point = encode_with_plans(&corpus, &trained_setup, &trained.plans);

    let base_nmr = corpus_nmr(&baseline.reference, &baseline.decoded);
    let trained_nmr = corpus_nmr(&point.reference, &point.decoded);
    eprintln!(
        "trained books: {} B → {} B, NMR {base_nmr:.5} → {trained_nmr:.5} (accepted {} ladder updates)",
        baseline.bytes, point.bytes, trained.ladder_updates_accepted
    );

    // Training must genuinely shrink the stream…
    assert!(
        point.bytes < baseline.bytes,
        "trained stream {} B must undercut the seed stream {} B",
        point.bytes,
        baseline.bytes
    );
    // …while the reconstruction stays transparent under the model
    // (the Lagrangian trade may move distortion a little, never
    // catastrophically).
    assert!(
        trained_nmr < 1.0,
        "trained NMR {trained_nmr} must stay below 0 dB"
    );
    assert!(
        trained_nmr <= base_nmr * 2.0 + 1e-6,
        "trained NMR {trained_nmr} must not blow up over baseline {base_nmr}"
    );

    // The trained table is §4.2.4 carriage-legal: the whole setup
    // header round-trips through the writer and parser.
    let header_bytes = write_setup_header(&trained_setup, 1).expect("trained setup writes");
    let parsed = parse_setup_header(&header_bytes, 1).expect("trained setup parses back");
    assert_eq!(parsed, trained_setup, "setup header round-trips");
}

#[test]
fn lambda_bisection_targets_a_byte_budget_on_the_real_stream() {
    // Fix the floor/psy levers at q = 0.7 and drive only the residue
    // lambda: the real rate(lambda) curve of the 8-frame stream is
    // monotone non-increasing, so solve_lambda_for_bits can land the
    // stream inside a byte budget.
    let tuning = EncoderTuning::from_quality(0.7).expect("tuning");
    let corpus = prepare_corpus(&tuning);
    let rate = |lambda: f64| encode_corpus(&corpus, lambda).map(|p| p.bytes as u64 * 8);

    let hi_rate = rate(1e-4).expect("high-fidelity rate");
    let lo_rate = rate(10.0).expect("rate-starved rate");
    assert!(
        hi_rate > lo_rate,
        "the lambda lever must span a real rate range: {hi_rate} vs {lo_rate}"
    );

    // Aim halfway between the extremes.
    let target = (hi_rate + lo_rate) / 2;
    let solution = solve_lambda_for_bits(target, 1e-4, 10.0, 40, rate).expect("solves");
    eprintln!(
        "budget {} bits: lambda {:.6} → {} bits (within {})",
        target, solution.lambda, solution.bits, solution.within_budget
    );
    assert!(solution.within_budget);
    assert!(solution.bits <= target);
    // The solver does not leave gross headroom on the table: the
    // rate one bisection-neighbourhood up crosses the budget, so the
    // landing sits in the top half of the achievable-under-budget
    // range (step curves can be coarse; this is a sanity floor).
    assert!(
        solution.bits * 2 >= lo_rate,
        "landing {} should not collapse to the starved end {lo_rate}",
        solution.bits
    );

    // Budget monotonicity on the real curve: a tighter budget never
    // gets a smaller lambda.
    let tight = solve_lambda_for_bits(lo_rate.max(1), 1e-4, 10.0, 40, rate).expect("solves");
    assert!(tight.within_budget);
    assert!(
        tight.lambda >= solution.lambda,
        "tighter budget → cheaper lambda: {} vs {}",
        tight.lambda,
        solution.lambda
    );
}

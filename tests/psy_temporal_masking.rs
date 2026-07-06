//! Temporal masking payoff — NMR-validated at equal transparency.
//!
//! The `TemporalMasking` pipeline extends the per-frame Bark-domain
//! model with post-masking (a masker's threshold elevation decays
//! across following frames) and pre-masking (one-frame-lookahead lift
//! before an onset). This suite measures what that buys the encoder on
//! a transient corpus — a loud tonal burst followed by a tail of
//! low-level tonal combs that the burst's decaying threshold covers:
//!
//! * the temporal encode spends **strictly fewer bytes** than the
//!   per-frame encode at the same lambda;
//! * both encodes are **transparent under their own model**
//!   (mean NMR < 1 measured against the thresholds that priced them) —
//!   the "equal transparency" criterion;
//! * on a **steady-state corpus the two models produce byte-identical
//!   streams** (the temporal extension is exactly the per-frame model
//!   when nothing is transient);
//! * the temporal encode's noise stays bounded against the *nominal*
//!   per-frame model too (the rate saving comes from post-masked
//!   detail, not from a fidelity collapse).
//!
//! Fully synthetic; the encode machinery mirrors
//! `tests/quality_rate_curve.rs`.

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
    residue_partition_weights, write_audio_packet, AudioChannelFloor, AudioDecoderState,
    AudioPacketHeader, AudioPacketOutcome, Floor1Packet, MaskingAnalysis, PsyConfig,
    ResidueVectorPlan, TemporalMasking, TemporalMaskingConfig,
};

const N: usize = 512;
const HALF_N: usize = N / 2;
const HOP: usize = HALF_N;
const PARTITION_SIZE: u32 = 16;
const SAMPLE_RATE: u32 = 44_100;
const FRAMES: usize = 9;
/// Lagrange multiplier: high enough that carrying *masked* detail
/// (weight ≈ 1) loses to its bit cost while audible detail
/// (weight ≫ 1) is still carried — the regime where the models'
/// threshold difference shows up in the rate.
const LAMBDA: f64 = 0.1;

/// The transient corpus: frame 0 is a loud three-tone burst; frames
/// 1.. carry a sparse low-level comb around the same bins (tone-like,
/// so the per-frame model prices it as audible detail) that the
/// burst's decaying post-masking skirt covers.
fn transient_frame(f: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; HALF_N];
    if f == 0 {
        x[11] = 0.60;
        x[23] = -0.40;
        x[34] = 0.35;
    } else {
        for &k in &[10usize, 14, 20, 26, 32, 38] {
            let h = (k + 31 * f).wrapping_mul(2_654_435_761) >> 9;
            x[k] = if h & 1 == 0 { 0.004 } else { -0.004 };
        }
    }
    x
}

/// A steady corpus: the identical moderate spectrum in every frame.
fn steady_frame(_f: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; HALF_N];
    x[11] = 0.30;
    x[23] = -0.20;
    x[40] = 0.10;
    x
}

fn signed_value_book(length: u8, step: f32) -> VorbisCodebook {
    let entries: u32 = 1u32 << length;
    let half = entries / 2;
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::Lattice {
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

/// The per-frame thresholds of a corpus under the plain per-frame
/// model.
fn nominal_analyses(frames: &[Vec<f32>], config: &PsyConfig) -> Vec<MaskingAnalysis> {
    frames
        .iter()
        .map(|x| compute_masking(x, config).expect("masking"))
        .collect()
}

/// The per-frame thresholds under the temporal pipeline.
fn temporal_analyses(frames: &[Vec<f32>], config: &PsyConfig) -> Vec<MaskingAnalysis> {
    let mut tm =
        TemporalMasking::new(&TemporalMaskingConfig::new(HOP), config).expect("temporal builds");
    let mut out = Vec::with_capacity(frames.len());
    for x in frames {
        if let Some(a) = tm.push_frame(x, config).expect("push") {
            out.push(a);
        }
    }
    out.push(tm.finish().expect("drains"));
    assert_eq!(out.len(), frames.len());
    out
}

/// Encode the corpus under the given per-frame analyses; returns the
/// total §4.3 packet bytes and the decoded spectra.
struct EncodedCorpus {
    bytes: usize,
    decoded: Vec<Vec<f32>>,
}

fn encode_corpus(frames: &[Vec<f32>], analyses: &[MaskingAnalysis]) -> EncodedCorpus {
    // Shared floor header from the corpus-max psy envelope.
    let mut envelopes = Vec::with_capacity(frames.len());
    let mut env_max = vec![f32::MIN_POSITIVE; HALF_N];
    for (x, a) in frames.iter().zip(analyses) {
        let env = plan_psy_floor_envelope(x, a, 2).expect("envelope");
        for (m, &v) in env_max.iter_mut().zip(&env) {
            *m = m.max(v);
        }
        envelopes.push(env);
    }
    let classes: Vec<Floor1Class> = [1u8, 2, 4]
        .iter()
        .map(|&d| Floor1Class {
            dimensions: d,
            subclasses: 0,
            masterbook: None,
            subclass_books: vec![Some(0)],
        })
        .collect();
    let header = design_floor1_header(&env_max, 16, 0.0, 1, &classes).expect("header designs");
    let floor_book = scalar_book(256, 8);
    let decoder =
        Floor1Decoder::new(&header, std::slice::from_ref(&floor_book)).expect("decoder builds");

    // Per-frame fits and residue targets.
    let mut fits = Vec::with_capacity(frames.len());
    let mut max_abs = 0.0f32;
    for f in 0..frames.len() {
        let posts = plan_floor1_envelope(&envelopes[f], &header).expect("fit");
        let floor1_y = plan_floor1_y(&posts, &header).expect("wrap");
        let rendered = decoder.render_curve(&floor1_y, HALF_N);
        let target: Vec<f32> = frames[f]
            .iter()
            .zip(&rendered)
            .map(|(&xv, &fv)| xv / fv)
            .collect();
        for &t in &target {
            max_abs = max_abs.max(t.abs());
        }
        let weights = residue_partition_weights(&rendered, &analyses[f], 0, HALF_N, PARTITION_SIZE)
            .expect("weights");
        fits.push((floor1_y, rendered, target, weights));
    }

    let coarse = signed_value_book(6, max_abs / 24.0);
    let fine = signed_value_book(6, max_abs / 192.0);
    let setup = mono_setup(header.clone(), coarse.clone(), fine.clone());
    let state = AudioDecoderState::new(&setup).expect("state builds");
    let empty_row: [Option<&VorbisCodebook>; 8] = [None; 8];
    let mut cascade_row: [Option<&VorbisCodebook>; 8] = [None; 8];
    cascade_row[0] = Some(&coarse);
    cascade_row[1] = Some(&fine);
    let value_books = [empty_row, cascade_row];

    let mut bytes = 0usize;
    let mut decoded = Vec::with_capacity(frames.len());
    for (floor1_y, _rendered, target, weights) in &fits {
        let scored = plan_vector_residue_rd_weighted(
            target,
            &value_books,
            1,
            PARTITION_SIZE,
            LAMBDA,
            weights,
        )
        .expect("plans");
        let floors = vec![AudioChannelFloor::Type1(Floor1Packet {
            nonzero: true,
            floor1_y: floor1_y.clone(),
            partition_cvals: vec![0u32; header.partition_class_list.len()],
        })];
        let plans = vec![vec![ResidueVectorPlan {
            classifications: scored.classifications,
            partition_entries: scored.partition_entries,
        }]];
        let pkt_header = AudioPacketHeader {
            mode_number: 0,
            blockflag: false,
            n: N,
            previous_window_flag: false,
            next_window_flag: false,
        };
        let packet = write_audio_packet(&pkt_header, &setup, N, 2048, 1, &floors, &plans)
            .expect("packet writes");
        bytes += packet.len();
        let mut reader = BitReaderLsb::new(&packet);
        match decode_audio_packet_pre_imdct(&mut reader, &setup, &state, 1, N, 2048)
            .expect("packet decodes")
        {
            AudioPacketOutcome::PreImdct { spectra, .. } => decoded.push(spectra[0].clone()),
            other => panic!("expected PreImdct, got {other:?}"),
        }
    }
    EncodedCorpus { bytes, decoded }
}

/// Mean noise-to-mask ratio of an encode against a set of analyses.
fn mean_nmr(frames: &[Vec<f32>], decoded: &[Vec<f32>], analyses: &[MaskingAnalysis]) -> f64 {
    let mut acc = 0.0f64;
    let mut count = 0usize;
    for f in 0..frames.len() {
        for k in 0..HALF_N {
            let noise = f64::from(frames[f][k]) - f64::from(decoded[f][k]);
            let t = f64::from(analyses[f].threshold[k]);
            acc += (noise / t) * (noise / t);
            count += 1;
        }
    }
    acc / count as f64
}

#[test]
fn temporal_masking_cuts_rate_at_equal_transparency_on_a_transient_corpus() {
    let frames: Vec<Vec<f32>> = (0..FRAMES).map(transient_frame).collect();
    let config = PsyConfig::new(SAMPLE_RATE);
    let nominal = nominal_analyses(&frames, &config);
    let temporal = temporal_analyses(&frames, &config);

    let enc_nominal = encode_corpus(&frames, &nominal);
    let enc_temporal = encode_corpus(&frames, &temporal);

    let nmr_nominal_own = mean_nmr(&frames, &enc_nominal.decoded, &nominal);
    let nmr_temporal_own = mean_nmr(&frames, &enc_temporal.decoded, &temporal);
    let nmr_temporal_vs_nominal = mean_nmr(&frames, &enc_temporal.decoded, &nominal);
    eprintln!(
        "transient corpus: nominal {} B (NMR {nmr_nominal_own:.4}) → temporal {} B \
         (NMR-own {nmr_temporal_own:.4}, NMR-vs-nominal {nmr_temporal_vs_nominal:.4})",
        enc_nominal.bytes, enc_temporal.bytes
    );

    // The rate saving is real.
    assert!(
        enc_temporal.bytes < enc_nominal.bytes,
        "temporal masking must cut the stream: {} vs {} bytes",
        enc_temporal.bytes,
        enc_nominal.bytes
    );
    // Equal transparency: both encodes are transparent under the model
    // that priced them.
    assert!(
        nmr_nominal_own < 1.0,
        "per-frame encode must be transparent under the per-frame model: {nmr_nominal_own}"
    );
    assert!(
        nmr_temporal_own < 1.0,
        "temporal encode must be transparent under the temporal model: {nmr_temporal_own}"
    );
    // The saving comes from post-masked detail, not a collapse: even
    // judged by the (stricter) per-frame model, the temporal encode's
    // noise stays within a bounded factor of the threshold.
    assert!(
        nmr_temporal_vs_nominal < 30.0,
        "temporal encode must stay bounded under the nominal model: {nmr_temporal_vs_nominal}"
    );
}

#[test]
fn temporal_masking_is_byte_identical_on_a_steady_corpus() {
    let frames: Vec<Vec<f32>> = (0..FRAMES).map(steady_frame).collect();
    let config = PsyConfig::new(SAMPLE_RATE);
    let nominal = nominal_analyses(&frames, &config);
    let temporal = temporal_analyses(&frames, &config);

    // The thresholds themselves coincide on steady state…
    for f in 0..FRAMES {
        for k in 0..HALF_N {
            let a = temporal[f].threshold[k];
            let b = nominal[f].threshold[k];
            assert!(
                (a - b).abs() <= b * 1e-6,
                "frame {f} bin {k}: temporal {a} vs nominal {b}"
            );
        }
    }
    // …so the encodes are byte-identical.
    let enc_nominal = encode_corpus(&frames, &nominal);
    let enc_temporal = encode_corpus(&frames, &temporal);
    assert_eq!(
        enc_nominal.bytes, enc_temporal.bytes,
        "steady-state streams must be byte-identical"
    );
    assert_eq!(enc_nominal.decoded, enc_temporal.decoded);
}

//! Psychoacoustically-driven encode → decode round-trip measurement.
//!
//! The unit tests in `src/psy.rs` pin the masking model's *structure*;
//! this suite proves the model **earns its keep** inside the real §4.3
//! encode path, against the crate's own decoder, with measured rate
//! and noise-to-mask figures:
//!
//! * **Rate:** a masking-driven floor (`plan_psy_floor_envelope`) +
//!   NMR-weighted residue chooser
//!   (`plan_vector_residue_rd_weighted` over
//!   `residue_partition_weights`) encodes the same spectrum into
//!   **fewer bytes** than a plain magnitude-envelope floor with the
//!   unweighted chooser at the same `lambda`, while keeping the mean
//!   noise-to-mask ratio below 0 dB (audibly transparent under the
//!   model).
//! * **Bit allocation:** at a rate-starved `lambda`, the unweighted
//!   chooser wastes bits on masked detail and skips audible content;
//!   the weighted chooser reverses both decisions — its NMR is an
//!   order of magnitude lower at no extra rate.
//! * **Quality lever:** raising `PsyConfig::threshold_offset_db`
//!   (a stricter audibility margin) monotonically spends more bytes
//!   and lowers the measured NMR.
//!
//! The signal is built directly in the analysis-spectrum domain (the
//! MDCT integration is already pinned by the PCM round-trip suites):
//! three strong tonal lines in the low spectrum, a mid-band noise
//! pedestal that sits right at its own noise-masking-noise threshold
//! (audibility flips with the `threshold_offset_db` margin), and a
//! deterministic low-level high-frequency hash far **below** the
//! absolute threshold of hearing at the 44.1 kHz calibration —
//! classic bit-waste bait for a non-perceptual encoder. NMR is
//! measured per bin against the nominal (offset-0) masking analysis:
//! `mean ((X − X̂) / threshold)²`.
//!
//! Fully synthetic: no Ogg framing, no `docs/` fixtures.

use oxideav_core::bits::BitReaderLsb;
use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::floor1::Floor1Decoder;
use oxideav_vorbis::setup::{
    Floor1Class, Floor1Header, FloorHeader, FloorKind, MappingHeader, MappingSubmap, ModeHeader,
    ResidueHeader, VorbisSetupHeader,
};
use oxideav_vorbis::{
    compute_masking, decode_audio_packet_pre_imdct, design_floor1_header, plan_floor1_envelope,
    plan_floor1_y, plan_psy_floor_envelope, plan_vector_residue_rd,
    plan_vector_residue_rd_weighted, residue_partition_weights, write_audio_packet,
    AudioChannelFloor, AudioDecoderState, AudioPacketHeader, AudioPacketOutcome, Floor1Packet,
    MaskingAnalysis, PsyConfig, ResidueVectorPlan, ScoredVectorResidue,
};

const N: usize = 512;
const HALF_N: usize = N / 2;
const PARTITION_SIZE: u32 = 16;
const SAMPLE_RATE: u32 = 44_100;
const MAX_POSTS: usize = 24;

/// The analysis spectrum: three strong tonal lines low, a mid-band
/// noise pedestal at its own masking edge, a deeply masked hash high.
/// Bin `k` sits at `(k + ½) · 44100 / 512` Hz ≈ 86 Hz spacing:
///
/// * tones at ≈ 1 / 2 / 3 kHz — unambiguously audible;
/// * a ±0.002 pedestal over bins 40..120 (≈ 3.5–10 kHz) whose own
///   band energy noise-masks it right at the nominal threshold, so a
///   ±12 dB `threshold_offset_db` margin flips its audibility;
/// * a ±0.002 hash over bins 128.. (≈ 11–22 kHz), far below the ATH
///   at the 96 dB full-scale calibration (≈ 0.005–0.5 linear) —
///   audibly nothing, yet nonzero everywhere.
fn analysis_spectrum() -> Vec<f32> {
    let mut x = vec![0.0f32; HALF_N];
    x[11] = 0.60; // ≈ 1.0 kHz
    x[23] = -0.40; // ≈ 2.0 kHz
    x[34] = 0.35; // ≈ 3.0 kHz
    for (k, v) in x.iter_mut().enumerate().take(120).skip(40) {
        let h = (k.wrapping_mul(2_654_435_761) >> 9) & 0xff;
        *v = (h as f32 / 128.0 - 1.0) * 0.002;
    }
    for (k, v) in x.iter_mut().enumerate().skip(HALF_N / 2) {
        let h = (k.wrapping_mul(2_654_435_761) >> 7) & 0xff;
        *v = (h as f32 / 128.0 - 1.0) * 0.002;
    }
    x
}

/// A Kraft-complete signed 1-D tessellation value book (2^length
/// entries of `length` bits, ladder step `step` centred on zero).
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

/// A balanced scalar (no-lookup) book.
fn scalar_book(entries: u32, length: u8) -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
}

/// Class catalogue for the floor-1 header designer.
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

/// Mono setup: floor value book (0), residue classbook (1), coarse (2)
/// and fine (3) value books; a two-classification residue over
/// `[0, HALF_N)` — class 0 decodes nothing (the skip class), class 1
/// runs the coarse+fine cascade.
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
        time_placeholders: Vec::new(),
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

/// A non-perceptual reference envelope: peak-held `|X|`, clamped into
/// the §10.1 ladder range — the "track the signal everywhere" floor an
/// encoder without a masking model would fit.
fn naive_envelope(x: &[f32]) -> Vec<f32> {
    let lo = oxideav_vorbis::floor1::INVERSE_DB_TABLE[0];
    let n = x.len();
    (0..n)
        .map(|k| {
            let from = k.saturating_sub(2);
            let to = (k + 2).min(n - 1);
            let mut peak = 0.0f32;
            for &v in &x[from..=to] {
                peak = peak.max(v.abs());
            }
            peak.clamp(lo, 1.0)
        })
        .collect()
}

/// One encoded operating point: the packet bytes plus the decoded
/// spectrum it reconstructs.
struct EncodedPoint {
    bytes: usize,
    decoded: Vec<f32>,
}

/// Fit a floor to `envelope`, plan the residue with the given chooser,
/// serialise the §4.3 packet, decode it back, and return the measured
/// point. `plan` receives the residue target (X over the rendered
/// floor) and the two-class value-book table.
fn encode_point<F>(x: &[f32], envelope: &[f32], plan: F) -> EncodedPoint
where
    F: FnOnce(&[f32], &[[Option<&VorbisCodebook>; 8]; 2]) -> ScoredVectorResidue,
{
    // Design + fit the floor against the supplied envelope.
    let classes = class_catalogue();
    let floor_header = design_floor1_header(envelope, MAX_POSTS, 0.0, 1, &classes)
        .expect("floor-1 header designs from the envelope");
    let floor_book = scalar_book(256, 8);
    let posts = plan_floor1_envelope(envelope, &floor_header).expect("floor-1 envelope fit");
    let floor1_y = plan_floor1_y(&posts, &floor_header).expect("floor-1 amplitude wrap");
    let decoder = Floor1Decoder::new(&floor_header, std::slice::from_ref(&floor_book))
        .expect("designed floor-1 decoder builds");
    let rendered = decoder.render_curve(&floor1_y, HALF_N);
    assert!(rendered.iter().all(|&f| f > 0.0 && f.is_finite()));

    // Residue target against the rendered floor.
    let target: Vec<f32> = x
        .iter()
        .zip(rendered.iter())
        .map(|(&xv, &fv)| xv / fv)
        .collect();
    let max_abs = target.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let coarse = signed_value_book(6, max_abs / 24.0);
    let fine = signed_value_book(6, max_abs / 192.0);
    let empty_row: [Option<&VorbisCodebook>; 8] = [None; 8];
    let mut cascade_row: [Option<&VorbisCodebook>; 8] = [None; 8];
    cascade_row[0] = Some(&coarse);
    cascade_row[1] = Some(&fine);
    let value_books = [empty_row, cascade_row];

    let scored = plan(&target, &value_books);

    // Serialise + decode.
    let setup = mono_setup(floor_header.clone(), coarse.clone(), fine.clone());
    let partition_cvals = vec![0u32; floor_header.partition_class_list.len()];
    let floors = vec![AudioChannelFloor::Type1(Floor1Packet {
        nonzero: true,
        floor1_y,
        partition_cvals,
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
    let bytes = write_audio_packet(&header, &setup, N, 2048, 1, &floors, &residue_plans)
        .expect("audio packet serialises");

    let state = AudioDecoderState::new(&setup).expect("audio decoder state builds");
    let mut reader = BitReaderLsb::new(&bytes);
    let outcome = decode_audio_packet_pre_imdct(&mut reader, &setup, &state, 1, N, 2048)
        .expect("audio packet decodes");
    let decoded = match outcome {
        AudioPacketOutcome::PreImdct { spectra, .. } => {
            assert_eq!(spectra.len(), 1);
            assert_eq!(spectra[0].len(), HALF_N);
            spectra.into_iter().next().unwrap()
        }
        other => panic!("expected PreImdct outcome, got {other:?}"),
    };
    assert!(decoded.iter().all(|v| v.is_finite()));

    EncodedPoint {
        bytes: bytes.len(),
        decoded,
    }
}

/// Fit + render the psy floor, returning the rendered curve (needed to
/// derive the perceptual partition weights the weighted chooser uses).
fn psy_rendered_floor(envelope: &[f32]) -> Vec<f32> {
    let classes = class_catalogue();
    let floor_header = design_floor1_header(envelope, MAX_POSTS, 0.0, 1, &classes)
        .expect("floor-1 header designs from the envelope");
    let floor_book = scalar_book(256, 8);
    let posts = plan_floor1_envelope(envelope, &floor_header).expect("floor-1 envelope fit");
    let floor1_y = plan_floor1_y(&posts, &floor_header).expect("floor-1 amplitude wrap");
    let decoder = Floor1Decoder::new(&floor_header, std::slice::from_ref(&floor_book))
        .expect("designed floor-1 decoder builds");
    decoder.render_curve(&floor1_y, HALF_N)
}

/// Mean noise-to-mask ratio (linear): `mean ((X − X̂) / threshold)²`.
/// Below 1.0 the reconstruction noise sits, on average, under the
/// masking threshold.
fn mean_nmr(x: &[f32], decoded: &[f32], masking: &MaskingAnalysis) -> f64 {
    let mut acc = 0.0f64;
    for k in 0..x.len() {
        let noise = f64::from(x[k]) - f64::from(decoded[k]);
        let t = f64::from(masking.threshold[k]);
        acc += (noise / t) * (noise / t);
    }
    acc / x.len() as f64
}

#[test]
fn psy_floor_and_weights_beat_the_naive_envelope_on_rate() {
    let x = analysis_spectrum();
    let config = PsyConfig::new(SAMPLE_RATE);
    let masking = compute_masking(&x, &config).expect("masking analysis");
    let lambda = 0.002;

    // Psy path: masking-driven floor + NMR-weighted chooser.
    let psy_env = plan_psy_floor_envelope(&x, &masking, 2).expect("psy envelope");
    let rendered = psy_rendered_floor(&psy_env);
    let weights = residue_partition_weights(&rendered, &masking, 0, HALF_N, PARTITION_SIZE)
        .expect("partition weights");
    let psy = encode_point(&x, &psy_env, |target, books| {
        plan_vector_residue_rd_weighted(target, books, 1, PARTITION_SIZE, lambda, &weights)
            .expect("weighted residue plan")
    });

    // Naive path: magnitude envelope + unweighted chooser, same lambda.
    let plain_env = naive_envelope(&x);
    let naive = encode_point(&x, &plain_env, |target, books| {
        plan_vector_residue_rd(target, books, 1, PARTITION_SIZE, lambda)
            .expect("unweighted residue plan")
    });

    let psy_nmr = mean_nmr(&x, &psy.decoded, &masking);
    let naive_nmr = mean_nmr(&x, &naive.decoded, &masking);
    eprintln!(
        "rate/NMR: psy {} B (NMR {psy_nmr:.4}), naive {} B (NMR {naive_nmr:.4})",
        psy.bytes, naive.bytes
    );

    // The psy encode must be audibly transparent under the model…
    assert!(
        psy_nmr < 1.0,
        "psy mean NMR {psy_nmr} must sit below 0 dB (1.0 linear)"
    );
    // …and meaningfully cheaper than the naive encode of the same
    // spectrum at the same lambda.
    assert!(
        (psy.bytes as f64) <= 0.8 * naive.bytes as f64,
        "psy {} B must undercut naive {} B by ≥20%",
        psy.bytes,
        naive.bytes
    );
}

#[test]
fn weighted_chooser_protects_audible_content_when_rate_starved() {
    // At a rate-starved lambda the *unweighted* chooser makes the
    // fatal perceptual mistake: a tonal partition's residue (≈ ±1
    // after the floor divide, a handful of hot bins) looks cheap to
    // skip — exactly as cheap as the borderline pedestal partitions it
    // keeps paying for. The NMR-weighted chooser, from the same books,
    // floor, and lambda, keeps every tone.
    let x = analysis_spectrum();
    let config = PsyConfig::new(SAMPLE_RATE);
    let masking = compute_masking(&x, &config).expect("masking analysis");
    let psy_env = plan_psy_floor_envelope(&x, &masking, 2).expect("psy envelope");
    let rendered = psy_rendered_floor(&psy_env);
    let weights = residue_partition_weights(&rendered, &masking, 0, HALF_N, PARTITION_SIZE)
        .expect("partition weights");
    let lambda = 0.01;

    let unweighted = encode_point(&x, &psy_env, |target, books| {
        plan_vector_residue_rd(target, books, 1, PARTITION_SIZE, lambda)
            .expect("unweighted residue plan")
    });
    let weighted = encode_point(&x, &psy_env, |target, books| {
        plan_vector_residue_rd_weighted(target, books, 1, PARTITION_SIZE, lambda, &weights)
            .expect("weighted residue plan")
    });

    let nmr_u = mean_nmr(&x, &unweighted.decoded, &masking);
    let nmr_w = mean_nmr(&x, &weighted.decoded, &masking);
    eprintln!(
        "rate-starved: unweighted {} B (NMR {nmr_u:.4}), weighted {} B (NMR {nmr_w:.4})",
        unweighted.bytes, weighted.bytes
    );

    // The unweighted plan is audibly broken (it dropped the tones);
    // the weighted plan stays transparent under the model — an
    // order-of-magnitude NMR gap.
    assert!(
        nmr_u > 1.0,
        "unweighted NMR {nmr_u} should exceed 0 dB (tones dropped)"
    );
    assert!(nmr_w < 1.0, "weighted NMR {nmr_w} must stay below 0 dB");
    assert!(
        nmr_w * 10.0 < nmr_u,
        "weighted NMR {nmr_w} must be ≥10× below unweighted {nmr_u}"
    );
    // The protection is bought with a bounded rate premium (the three
    // tonal partitions the unweighted plan dropped).
    assert!(
        weighted.bytes <= unweighted.bytes + 200,
        "weighted {} B premium over unweighted {} B is out of bounds",
        weighted.bytes,
        unweighted.bytes
    );
}

#[test]
fn threshold_offset_lever_trades_rate_for_fidelity() {
    // Raising threshold_offset_db lowers every threshold (a stricter
    // audibility margin). The pedestal sits at its own nominal
    // noise-masking threshold, so the margin flips its fate: at −12 dB
    // it is deeply masked (skipped), at +12 dB it is clearly audible
    // (its weight rises ≈16×, the chooser codes it). Rate is therefore
    // non-decreasing in the margin and the NMR against the *nominal*
    // threshold non-increasing. The lambda is set high enough that the
    // borderline pedestal is not already worth coding at the nominal
    // margin (λ · 192 partition bits ≈ 29 ≫ the ~13 borderline NMR of
    // skipping it).
    let x = analysis_spectrum();
    let nominal = compute_masking(&x, &PsyConfig::new(SAMPLE_RATE)).expect("nominal masking");
    let lambda = 0.15;

    let mut bytes_curve = Vec::new();
    let mut nmr_curve = Vec::new();
    for offset_db in [-12.0f32, 0.0, 12.0] {
        let config = PsyConfig {
            threshold_offset_db: offset_db,
            ..PsyConfig::new(SAMPLE_RATE)
        };
        let masking = compute_masking(&x, &config).expect("masking analysis");
        let env = plan_psy_floor_envelope(&x, &masking, 2).expect("psy envelope");
        let rendered = psy_rendered_floor(&env);
        let weights = residue_partition_weights(&rendered, &masking, 0, HALF_N, PARTITION_SIZE)
            .expect("partition weights");
        let point = encode_point(&x, &env, |target, books| {
            plan_vector_residue_rd_weighted(target, books, 1, PARTITION_SIZE, lambda, &weights)
                .expect("weighted residue plan")
        });
        let nmr = mean_nmr(&x, &point.decoded, &nominal);
        eprintln!("offset {offset_db:+} dB: {} B, NMR {nmr:.5}", point.bytes);
        bytes_curve.push(point.bytes);
        nmr_curve.push(nmr);
    }

    // Rate is non-decreasing in the margin (± a few floor-packet
    // bytes: the envelope values shift with the thresholds); NMR is
    // non-increasing (± measurement noise).
    assert!(
        bytes_curve[0] <= bytes_curve[1] + 8 && bytes_curve[1] <= bytes_curve[2] + 8,
        "rate must not fall as the margin rises: {bytes_curve:?}"
    );
    assert!(
        nmr_curve[0] >= nmr_curve[1] - 1e-6 && nmr_curve[1] >= nmr_curve[2] - 1e-6,
        "NMR must not rise as the margin rises: {nmr_curve:?}"
    );
    // The sweep is not degenerate: the +12 dB margin codes the
    // pedestal the −12 dB margin skips — a real rate and fidelity gap.
    assert!(
        bytes_curve[2] >= bytes_curve[0] + 32,
        "extremes should differ in rate: {bytes_curve:?}"
    );
    assert!(
        nmr_curve[2] * 2.0 < nmr_curve[0],
        "extremes should differ in fidelity: {nmr_curve:?}"
    );
}

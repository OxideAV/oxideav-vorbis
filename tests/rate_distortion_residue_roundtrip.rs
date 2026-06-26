//! Rate-distortion residue selection PCM → encode → decode → PCM round-trip.
//!
//! `tests/pcm_adaptive_residue_roundtrip.rs` proves the **distortion-only**
//! classification chooser (`plan_vector_residue`) round-trips through the
//! full §4.3 audio packet. This suite is its rate-aware counterpart: it
//! drives the same synthetic PCM through the **rate-distortion** residue
//! stack — `plan_vector_residue_rd` (Lagrangian per-partition choice) and
//! `select_residue_config` (whole-vector config selection) — and asserts:
//!
//!   1. **λ = 0 fidelity.** At `lambda == 0` the rate-distortion plan is the
//!      distortion plan; it round-trips to the same pinned PCM-domain SNR.
//!   2. **The rate knob bites monotonically.** Sweeping `lambda` upward
//!      never *increases* the plan's total value-codeword bit cost — a
//!      larger Lagrange multiplier only ever trades distortion for fewer
//!      bits (within the bit-budget direction). A large `lambda` spends
//!      strictly fewer bits than `lambda == 0` on a tilted spectrum.
//!   3. **Every rate point still round-trips.** Each `lambda` produces a
//!      plan the real decoder reconstructs to finite PCM; rate reduction
//!      trades fidelity gracefully, it does not corrupt the bitstream.
//!   4. **Config selection is sound.** `select_residue_config` over a
//!      fine/coarse candidate pair picks the lower-cost config and its
//!      chosen plan round-trips.
//!
//! Fully synthetic: no Ogg framing, no `docs/` fixtures, so it runs in
//! standalone CI.

use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::floor1::Floor1Decoder;
use oxideav_vorbis::setup::{
    Floor1Class, Floor1Header, FloorHeader, FloorKind, MappingHeader, MappingSubmap, ModeHeader,
    ResidueHeader, VorbisSetupHeader,
};
use oxideav_vorbis::{
    apply_window_and_mdct_vec, decode_audio_packet_windowed, imdct_naive_vec, plan_floor1_envelope,
    plan_floor1_y, plan_vector_residue_rd, select_residue_config, write_audio_packet,
    AudioChannelFloor, AudioDecoderState, AudioPacketHeader, Floor1Packet, ResidueConfigCandidate,
    ResidueVectorPlan, WindowedPacketOutcome,
};

use oxideav_core::bits::BitReaderLsb;

/// §4.3.1 Vorbis window of length `n`.
fn vorbis_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let inner = (std::f32::consts::PI / n as f32 * (i as f32 + 0.5)).sin();
            (std::f32::consts::FRAC_PI_2 * inner * inner).sin()
        })
        .collect()
}

/// A Kraft-complete 1-D tessellation VQ value book: `2^length` entries all
/// at codeword length `length`, ladder `(e − half)·step` centred on zero.
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

/// A balanced 1-D scalar classbook (no VQ lookup).
fn classbook(entries: u32, length: u8) -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
}

/// 10·log10 of the energy ratio between a target and the per-bin error.
fn snr_db(target: &[f32], got: &[f32]) -> f32 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&t, &g) in target.iter().zip(got.iter()) {
        sig += (t as f64) * (t as f64);
        let e = (t - g) as f64;
        err += e * e;
    }
    if err == 0.0 {
        return f32::INFINITY;
    }
    (10.0 * (sig / err).log10()) as f32
}

/// A non-flat floor-1 header with `interior_x` interior posts.
fn nonflat_floor_header(interior_x: Vec<u32>, rangebits: u8) -> Floor1Header {
    Floor1Header {
        partitions: 1,
        partition_class_list: vec![0],
        classes: vec![Floor1Class {
            dimensions: interior_x.len() as u8,
            subclasses: 0,
            masterbook: None,
            subclass_books: vec![Some(0)],
        }],
        multiplier: 1,
        rangebits,
        x_list: interior_x,
    }
}

/// Mono setup carrying a non-flat floor-1 and a three-class format-1
/// residue (class 0 unused, class 1 coarse, class 2 coarse+fine), exactly
/// as `tests/pcm_adaptive_residue_roundtrip.rs` builds it.
#[allow(clippy::too_many_arguments)]
fn mono_setup_multiclass(
    floor_header: Floor1Header,
    floor_book: VorbisCodebook,
    classbook: VorbisCodebook,
    coarse: VorbisCodebook,
    fine: VorbisCodebook,
    half_n: u32,
    partition_size: u32,
) -> VorbisSetupHeader {
    let floor = FloorHeader {
        floor_type: 1,
        kind: FloorKind::Type1(floor_header),
    };
    let unused: [Option<u8>; 8] = Default::default();
    let mut coarse_stages: [Option<u8>; 8] = Default::default();
    coarse_stages[0] = Some(2);
    let mut fine_stages: [Option<u8>; 8] = Default::default();
    fine_stages[0] = Some(2);
    fine_stages[1] = Some(3);

    let residue = ResidueHeader {
        residue_type: 1,
        residue_begin: 0,
        residue_end: half_n,
        partition_size,
        classifications: 3,
        classbook: 1,
        cascade: vec![0, 1 << 0, (1 << 0) | (1 << 1)],
        books: vec![unused, coarse_stages, fine_stages],
    };

    VorbisSetupHeader {
        codebooks: vec![floor_book, classbook, coarse, fine],
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

/// Synthetic length-`n` PCM with a steep spectral tilt.
fn synthetic_pcm(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            let nn = n as f32;
            0.85 * (2.0 * std::f32::consts::PI * 2.0 * t / nn).sin()
                + 0.55 * (2.0 * std::f32::consts::PI * 5.0 * t / nn).sin()
                + 0.30 * (2.0 * std::f32::consts::PI * 11.0 * t / nn).cos()
                + 0.12 * (2.0 * std::f32::consts::PI * 23.0 * t / nn).sin()
                + 0.04 * (2.0 * std::f32::consts::PI * 45.0 * t / nn).cos()
        })
        .collect()
}

/// A descending exponential magnitude envelope bounding `|X|` from above.
fn magnitude_envelope(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let peak = x.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let n_f = (n.max(1) - 1).max(1) as f32;
    let mut env = vec![0.0f32; n];
    for (k, e) in env.iter_mut().enumerate() {
        let frac = k as f32 / n_f;
        let smooth = peak * (256.0f32).powf(-frac);
        *e = smooth.max(x[k].abs()).max(peak / 256.0);
    }
    env
}

/// The shared analysis fixture: forward window+MDCT → analysis spectrum X,
/// a fitted non-flat floor-1, the rendered floor, the residue target
/// `X / rendered_floor`, the cascade books, and the setup header.
struct Fixture {
    n: usize,
    half_n: usize,
    partition_size: u32,
    window: Vec<f32>,
    x: Vec<f32>,
    residue_target: Vec<f32>,
    setup: VorbisSetupHeader,
    floors: Vec<AudioChannelFloor>,
    coarse: VorbisCodebook,
    fine: VorbisCodebook,
}

fn build_fixture(n: usize, partitions: usize) -> Fixture {
    let half_n = n / 2;
    assert_eq!(half_n % partitions, 0, "partitions must divide half_n");
    let partition_size = (half_n / partitions) as u32;
    let window = vorbis_window(n);

    let mut pcm = synthetic_pcm(n);
    let x = apply_window_and_mdct_vec(&mut pcm, &window, 1.0).expect("forward MDCT");
    assert_eq!(x.len(), half_n);

    let envelope = magnitude_envelope(&x);
    let interior: Vec<u32> = {
        let pts = [
            half_n / 8,
            half_n / 4,
            (3 * half_n) / 8,
            half_n / 2,
            (5 * half_n) / 8,
            (3 * half_n) / 4,
            (7 * half_n) / 8,
        ];
        let mut v: Vec<u32> = pts.iter().map(|&p| p.max(1) as u32).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let mut rangebits = 1u8;
    while (1u32 << rangebits) < half_n as u32 {
        rangebits += 1;
    }
    let floor_header = nonflat_floor_header(interior, rangebits);
    let posts = plan_floor1_envelope(&envelope, &floor_header).expect("floor-1 envelope fit");
    let floor1_y = plan_floor1_y(&posts, &floor_header).expect("floor-1 amplitude wrap");
    let floor_book = classbook(256, 8);

    let decoder = Floor1Decoder::new(&floor_header, std::slice::from_ref(&floor_book))
        .expect("floor-1 decoder");
    let rendered_floor = decoder.render_curve(&floor1_y, half_n);
    let residue_target: Vec<f32> = x
        .iter()
        .zip(rendered_floor.iter())
        .map(|(&xv, &fv)| xv / fv)
        .collect();
    let max_abs = residue_target.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(max_abs > 0.0);

    let coarse_step = max_abs / 24.0;
    let fine_step = coarse_step / 8.0;
    let cb = classbook(4, 2);
    let coarse = signed_value_book(6, coarse_step);
    let fine = signed_value_book(6, fine_step);

    let setup = mono_setup_multiclass(
        floor_header.clone(),
        floor_book.clone(),
        cb,
        coarse.clone(),
        fine.clone(),
        half_n as u32,
        partition_size,
    );
    let floors = vec![AudioChannelFloor::Type1(Floor1Packet {
        nonzero: true,
        floor1_y,
        partition_cvals: vec![0],
    })];

    Fixture {
        n,
        half_n,
        partition_size,
        window,
        x,
        residue_target,
        setup,
        floors,
        coarse,
        fine,
    }
}

impl Fixture {
    /// The three-row classification value-book table (matching the setup's
    /// three classes): class 0 unused, class 1 coarse, class 2 coarse+fine.
    fn value_books(&self) -> Vec<[Option<&VorbisCodebook>; 8]> {
        let empty: [Option<&VorbisCodebook>; 8] = Default::default();
        let mut row_coarse: [Option<&VorbisCodebook>; 8] = Default::default();
        row_coarse[0] = Some(&self.coarse);
        let mut row_fine: [Option<&VorbisCodebook>; 8] = Default::default();
        row_fine[0] = Some(&self.coarse);
        row_fine[1] = Some(&self.fine);
        vec![empty, row_coarse, row_fine]
    }

    /// Serialise + decode a residue plan, returning the PCM-domain SNR.
    fn round_trip_snr(&self, plan: ResidueVectorPlan) -> f32 {
        let residue_plans = vec![vec![plan]];
        let header = AudioPacketHeader {
            mode_number: 0,
            blockflag: false,
            n: self.n,
            previous_window_flag: false,
            next_window_flag: false,
        };
        let bytes = write_audio_packet(
            &header,
            &self.setup,
            self.n,
            self.n,
            1,
            &self.floors,
            &residue_plans,
        )
        .expect("audio packet serialises");
        assert!(!bytes.is_empty());

        let state = AudioDecoderState::new(&self.setup).expect("decoder state");
        let mut reader = BitReaderLsb::new(&bytes);
        let outcome =
            decode_audio_packet_windowed(&mut reader, &self.setup, &state, 1, self.n, self.n, 1.0)
                .expect("audio packet decodes");
        let decoded = match &outcome {
            WindowedPacketOutcome::Windowed { frames, .. } => frames[0].clone(),
            WindowedPacketOutcome::ZeroedWindowed { .. } => {
                panic!("nonzero floor + residue must not zero the packet")
            }
        };
        assert!(decoded.iter().all(|s| s.is_finite()));

        let ref_time = imdct_naive_vec(&self.x, 1.0).expect("reference IMDCT");
        let ref_frame: Vec<f32> = ref_time
            .iter()
            .zip(self.window.iter())
            .map(|(&t, &w)| t * w)
            .collect();
        snr_db(&ref_frame, &decoded)
    }
}

#[test]
fn rd_lambda_zero_round_trips_like_distortion_plan() {
    let fx = build_fixture(1024, 16);
    let vb = fx.value_books();
    let scored = plan_vector_residue_rd(&fx.residue_target, &vb, 1, fx.partition_size, 0.0)
        .expect("rate-distortion plan at lambda 0");
    assert_eq!(
        scored.classifications.len(),
        fx.half_n / fx.partition_size as usize
    );

    let snr = fx.round_trip_snr(ResidueVectorPlan {
        classifications: scored.classifications.clone(),
        partition_entries: scored.partition_entries,
    });
    assert!(
        snr >= 20.0,
        "lambda-0 RD round-trip SNR {snr} dB below pinned 20 dB"
    );
    // The plan must be content-adaptive on this tilted spectrum.
    let first = scored.classifications[0];
    assert!(
        scored.classifications.iter().any(|&c| c != first),
        "lambda-0 classification {:?} is constant — not adaptive",
        scored.classifications
    );
}

#[test]
fn rd_increasing_lambda_never_increases_bit_cost() {
    let fx = build_fixture(1024, 16);
    let vb = fx.value_books();

    // A monotone-increasing lambda sweep. Total value-codeword bits must be
    // non-increasing: a larger Lagrange multiplier only ever trades
    // distortion for fewer bits, never the reverse.
    let lambdas = [0.0f64, 0.01, 0.05, 0.2, 1.0, 5.0, 50.0];
    let mut prev_bits = u64::MAX;
    let mut bits_seq = Vec::new();
    for &lambda in &lambdas {
        let scored = plan_vector_residue_rd(&fx.residue_target, &vb, 1, fx.partition_size, lambda)
            .expect("rate-distortion plan");
        let bits = scored.total_value_bits;
        assert!(
            bits <= prev_bits,
            "bit cost rose from {prev_bits} to {bits} as lambda increased to {lambda}"
        );
        prev_bits = bits;
        bits_seq.push(bits);

        // Every rate point must still round-trip to finite PCM.
        let snr = fx.round_trip_snr(ResidueVectorPlan {
            classifications: scored.classifications,
            partition_entries: scored.partition_entries,
        });
        assert!(snr.is_finite(), "lambda {lambda} produced non-finite PCM");
    }
    // The knob must genuinely move: the cheapest point spends strictly
    // fewer bits than the most accurate (lambda 0) one.
    assert!(
        *bits_seq.last().unwrap() < bits_seq[0],
        "large-lambda bits {} not below lambda-0 bits {} — rate knob inert",
        bits_seq.last().unwrap(),
        bits_seq[0]
    );
}

#[test]
fn rd_higher_lambda_trades_fidelity_for_rate() {
    let fx = build_fixture(1024, 16);
    let vb = fx.value_books();

    let low = plan_vector_residue_rd(&fx.residue_target, &vb, 1, fx.partition_size, 0.0).unwrap();
    let high = plan_vector_residue_rd(&fx.residue_target, &vb, 1, fx.partition_size, 50.0).unwrap();

    // High lambda spends fewer bits (rate down) and leaves more distortion
    // (the Lagrangian's whole point) — the encoder's bit-budget response.
    assert!(high.total_value_bits < low.total_value_bits);
    assert!(high.total_error_sq >= low.total_error_sq);

    let low_snr = fx.round_trip_snr(ResidueVectorPlan {
        classifications: low.classifications,
        partition_entries: low.partition_entries,
    });
    let high_snr = fx.round_trip_snr(ResidueVectorPlan {
        classifications: high.classifications,
        partition_entries: high.partition_entries,
    });
    // The lower-rate plan still round-trips; it is simply less faithful.
    assert!(high_snr.is_finite() && low_snr.is_finite());
    assert!(
        low_snr >= high_snr - 1e-3,
        "lower-rate plan SNR {high_snr} exceeded higher-rate {low_snr} — RD trade inverted"
    );
}

#[test]
fn select_residue_config_picks_round_trippable_winner() {
    let fx = build_fixture(1024, 16);
    let vb_full = fx.value_books(); // unused / coarse / coarse+fine

    // A second, cheaper candidate: coarse-only (drop the fine refinement
    // class). It spends fewer bits but reconstructs less accurately.
    let empty: [Option<&VorbisCodebook>; 8] = Default::default();
    let mut row_coarse: [Option<&VorbisCodebook>; 8] = Default::default();
    row_coarse[0] = Some(&fx.coarse);
    let vb_coarse = vec![empty, row_coarse];

    // The full config offers a strict superset of classes (it can always
    // fall back to unused/coarse), so on value bits + distortion alone it
    // can never lose to the coarse-only config. The real lever that
    // distinguishes them is the *classword width*: a 3-class config needs a
    // ≥2-bit classbook, while the 2-class coarse-only config fits a 1-bit
    // one. Charge that difference (classword_bits 2 vs 1). At a large lambda
    // the per-partition classword saving outweighs the coarse config's
    // distortion penalty, flipping the winner.
    let candidates = vec![
        ResidueConfigCandidate {
            residue_type: 1,
            partition_size: fx.partition_size,
            value_books: &vb_full,
            classword_bits: 2,
            partitions_per_classword: 1,
        },
        ResidueConfigCandidate {
            residue_type: 1,
            partition_size: fx.partition_size,
            value_books: &vb_coarse,
            classword_bits: 1,
            partitions_per_classword: 1,
        },
    ];

    // At a tiny lambda fidelity dominates → the full (fine-capable) config
    // wins; at a large lambda the narrower-classbook coarse config wins.
    let sel_fidelity = select_residue_config(&fx.residue_target, &candidates, 1e-6).unwrap();
    assert_eq!(sel_fidelity.config_index, 0);

    let sel_rate = select_residue_config(&fx.residue_target, &candidates, 1e6).unwrap();
    assert_eq!(sel_rate.config_index, 1);

    // Both winners must round-trip through the real decoder. The setup's
    // residue header configures all three classes, so a plan that only ever
    // selects classes 0/1 (the coarse candidate) is still valid against it.
    for sel in [&sel_fidelity, &sel_rate] {
        let snr = fx.round_trip_snr(ResidueVectorPlan {
            classifications: sel.plan.classifications.clone(),
            partition_entries: sel.plan.partition_entries.clone(),
        });
        assert!(snr.is_finite(), "selected config did not round-trip");
    }
}

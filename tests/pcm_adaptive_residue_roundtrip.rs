//! Adaptive-classification residue PCM → encode → decode → PCM round-trip.
//!
//! The existing `tests/nonflat_floor_pcm_roundtrip.rs` round-trip codes the
//! whole spectrum as **one** residue partition with a **hand-supplied**
//! classification (`classifications: vec![0]`). This suite closes the
//! representative encode the residue classification chooser unlocks: the
//! residue is split into **many partitions**, and
//! [`oxideav_vorbis::plan_vector_residue`] picks **each partition's
//! classification from the spectrum** — no hand-supplied classifications —
//! before the §4.3 audio packet is serialised and decoded back to the time
//! domain.
//!
//! The chain, end to end:
//!
//!  1. **Analysis.** Synthetic length-`N` PCM → §4.3.1 window → §4.3.7
//!     forward MDCT → length-`N/2` analysis spectrum `X`.
//!  2. **Floor fit.** A smoothed `|X|` envelope is fitted to a multi-post
//!     floor-1 (`plan_floor1_envelope` → `plan_floor1_y`); the rendered
//!     floor is recovered with `Floor1Decoder::render_curve`.
//!  3. **Residue target.** `X[k] / rendered_floor[k]` — the per-bin value
//!     the decoder multiplies the floor back into (§4.3.6).
//!  4. **Adaptive plan (the code under test).** The residue window is split
//!     into `partitions` partitions of `partition_size` bins each;
//!     `plan_vector_residue` scores every classification per partition and
//!     keeps the one minimising reconstruction distortion. The result is
//!     the `classifications` + `partition_entries` a `ResidueVectorPlan`
//!     holds, with no hand-supplied classifications.
//!  5. **Encode.** `write_audio_packet` serialises the §4.3.1 prelude, the
//!     non-flat floor-1 body, and the multi-partition residue body.
//!  6. **Decode.** `decode_audio_packet_windowed` rebuilds `floor · residue`,
//!     runs the §4.3.7 IMDCT and §4.3.6 window.
//!  7. **Assert.** The decoded windowed frame tracks `window ⊙ IMDCT(X)` to
//!     a pinned PCM-domain SNR; the adaptive plan matches an
//!     explicit-classification replan bit-for-bit (the selection ↔ entry
//!     round-trip is exact); and the adaptive plan clears a fixed
//!     single-coarse-class baseline.
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
    plan_floor1_y, plan_vector_partition_entries, plan_vector_residue, write_audio_packet,
    AudioChannelFloor, AudioDecoderState, AudioPacketHeader, Floor1Packet, ResidueVectorPlan,
    WindowedPacketOutcome,
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

/// A balanced 1-D scalar classbook (no VQ lookup): `entries` codewords of
/// length `length`. Used as the §8.6.2 classbook (each partition's
/// classification index is Huffman-coded with it).
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

/// A non-flat floor-1 header with `interior_x` interior posts (single
/// partition class, subclasses = 0 so each post is coded directly through
/// the 256-entry value book).
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

/// Build the mono setup header carrying a non-flat floor-1 and a
/// **multi-classification** format-1 residue over `[0, half_n)` with the
/// given `partition_size`. Codebook 0 is the floor value book; 1 is the
/// residue classbook; 2/3 are the cascade value books. Three
/// classifications are configured:
///   class 0 — 'unused' (empty cascade) — codes a partition as all-zero;
///   class 1 — coarse single stage (value book 2);
///   class 2 — coarse + fine two-stage cascade (value books 2, 3).
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

    // class 0: no stages. class 1: stage 0 → book 2. class 2: stage 0 → 2,
    // stage 1 → 3.
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
        cascade: vec![
            0,                   // class 0 — no stages
            1 << 0,              // class 1 — stage 0
            (1 << 0) | (1 << 1), // class 2 — stages 0 and 1
        ],
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

/// Synthetic length-`n` PCM with a steep spectral tilt (strong low-freq,
/// weak high-freq) so the residue energy genuinely varies across the band
/// — the content that makes per-partition adaptive classification pay off.
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

/// A descending exponential magnitude envelope (~48 dB span) that bounds
/// `|X|` from above — a genuinely non-flat floor target.
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

/// One adaptive-residue round-trip for block size `n` with `partitions`
/// residue partitions. Returns:
///   - the decoded windowed frame's PCM-domain SNR vs `window ⊙ IMDCT(X)`,
///   - the SNR of a fixed single-coarse-class baseline (same partitions,
///     every partition forced to class 1),
///   - the chosen classifications.
fn adaptive_roundtrip(n: usize, partitions: usize) -> (f32, f32, Vec<u32>) {
    let half_n = n / 2;
    assert_eq!(half_n % partitions, 0, "partitions must divide half_n");
    let partition_size = (half_n / partitions) as u32;
    let window = vorbis_window(n);

    // 1. forward window + MDCT → analysis spectrum X.
    let mut pcm = synthetic_pcm(n);
    let x =
        apply_window_and_mdct_vec(&mut pcm, &window, 1.0).expect("forward MDCT of analysis block");
    assert_eq!(x.len(), half_n);

    // 2. fit a non-flat floor-1 to a smoothed |X| envelope.
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

    // 3. render the floor the decoder reconstructs, and form the residue
    //    target X / rendered_floor (the §4.3.6 per-bin multiplier).
    let decoder = Floor1Decoder::new(&floor_header, std::slice::from_ref(&floor_book))
        .expect("floor-1 decoder builds");
    let rendered_floor = decoder.render_curve(&floor1_y, half_n);
    assert!(rendered_floor.iter().all(|&f| f > 0.0 && f.is_finite()));
    let residue_target: Vec<f32> = x
        .iter()
        .zip(rendered_floor.iter())
        .map(|(&xv, &fv)| xv / fv)
        .collect();
    let max_abs = residue_target.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(max_abs > 0.0);

    // cascade value books sized to the residue target's range.
    let coarse_step = max_abs / 24.0;
    let fine_step = coarse_step / 8.0;
    let cb = classbook(4, 2);
    let coarse = signed_value_book(6, coarse_step);
    let fine = signed_value_book(6, fine_step);

    // The value_books table the chooser scores against — one row per
    // classification (matching the residue header's three classes).
    let empty: [Option<&VorbisCodebook>; 8] = Default::default();
    let mut row_coarse: [Option<&VorbisCodebook>; 8] = Default::default();
    row_coarse[0] = Some(&coarse);
    let mut row_fine: [Option<&VorbisCodebook>; 8] = Default::default();
    row_fine[0] = Some(&coarse);
    row_fine[1] = Some(&fine);
    let value_books = vec![empty, row_coarse, row_fine];

    // 4. adaptive plan: choose each partition's classification from the
    //    spectrum (the code under test).
    let (classifications, partition_entries) =
        plan_vector_residue(&residue_target, &value_books, 1, partition_size)
            .expect("from-spectrum residue plans");
    assert_eq!(classifications.len(), partitions);

    // Bit-exactness cross-check: an explicit replan on the SAME chosen
    // classifications must yield identical entry lists.
    let replan = plan_vector_partition_entries(
        &residue_target,
        &classifications,
        &value_books,
        1,
        partition_size,
    )
    .expect("explicit-classification residue replan");
    assert_eq!(
        replan, partition_entries,
        "adaptive entry lists differ from explicit replan on the same classifications"
    );

    let setup = mono_setup_multiclass(
        floor_header.clone(),
        floor_book.clone(),
        cb.clone(),
        coarse.clone(),
        fine.clone(),
        half_n as u32,
        partition_size,
    );
    let floors = vec![AudioChannelFloor::Type1(Floor1Packet {
        nonzero: true,
        floor1_y: floor1_y.clone(),
        partition_cvals: vec![0],
    })];

    let adaptive_snr = encode_decode_snr(
        n,
        &setup,
        &window,
        &x,
        &floors,
        ResidueVectorPlan {
            classifications: classifications.clone(),
            partition_entries,
        },
    );

    // Baseline: force every partition to the coarse single-stage class (1).
    let fixed_classes = vec![1u32; partitions];
    let fixed_entries = plan_vector_partition_entries(
        &residue_target,
        &fixed_classes,
        &value_books,
        1,
        partition_size,
    )
    .expect("fixed-class residue plan");
    let fixed_snr = encode_decode_snr(
        n,
        &setup,
        &window,
        &x,
        &floors,
        ResidueVectorPlan {
            classifications: fixed_classes,
            partition_entries: fixed_entries,
        },
    );

    (adaptive_snr, fixed_snr, classifications)
}

/// Serialise a §4.3 audio packet from the given floor + residue plan,
/// decode it back to a windowed frame, and return the PCM-domain SNR vs
/// `window ⊙ IMDCT(X)`.
fn encode_decode_snr(
    n: usize,
    setup: &VorbisSetupHeader,
    window: &[f32],
    x: &[f32],
    floors: &[AudioChannelFloor],
    residue_plan: ResidueVectorPlan,
) -> f32 {
    let blocksize_1 = n;
    let residue_plans = vec![vec![residue_plan]];
    let header = AudioPacketHeader {
        mode_number: 0,
        blockflag: false,
        n,
        previous_window_flag: false,
        next_window_flag: false,
    };
    let bytes = write_audio_packet(&header, setup, n, blocksize_1, 1, floors, &residue_plans)
        .expect("audio packet serialises");
    assert!(!bytes.is_empty());

    let state = AudioDecoderState::new(setup).expect("audio decoder state builds");
    let mut reader = BitReaderLsb::new(&bytes);
    let outcome = decode_audio_packet_windowed(&mut reader, setup, &state, 1, n, blocksize_1, 1.0)
        .expect("audio packet decodes");
    let decoded_frame = match &outcome {
        WindowedPacketOutcome::Windowed { frames, .. } => {
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].len(), n);
            frames[0].clone()
        }
        WindowedPacketOutcome::ZeroedWindowed { .. } => {
            panic!("a nonzero floor + nonzero residue must not zero the packet")
        }
    };

    let ref_time = imdct_naive_vec(x, 1.0).expect("reference IMDCT of analysis spectrum");
    let ref_frame: Vec<f32> = ref_time
        .iter()
        .zip(window.iter())
        .map(|(&t, &w)| t * w)
        .collect();

    assert!(decoded_frame.iter().all(|s| s.is_finite()));
    snr_db(&ref_frame, &decoded_frame)
}

#[test]
fn adaptive_residue_pcm_round_trips_to_time_domain() {
    // 16 partitions of 32 bins each over a 512-bin spectrum.
    let (adaptive, _fixed, classifications) = adaptive_roundtrip(1024, 16);
    assert!(
        adaptive >= 20.0,
        "adaptive-residue PCM round-trip SNR {adaptive} dB below pinned 20 dB"
    );
    // The chooser must be content-adaptive: not every partition takes the
    // same class on a sharply-tilted spectrum.
    let first = classifications[0];
    assert!(
        classifications.iter().any(|&c| c != first),
        "classification selection is constant {classifications:?} — not adaptive"
    );
}

#[test]
fn adaptive_residue_beats_or_matches_fixed_class() {
    // On a sharply-tilted spectrum, adaptive selection (which can spend the
    // fine refinement stage where residue energy is high and the cheap/
    // unused class where it is low) must not be worse than forcing the
    // single coarse class everywhere — and is measurably better here.
    let (adaptive, fixed, _classes) = adaptive_roundtrip(1024, 16);
    assert!(
        adaptive >= fixed - 1e-3,
        "adaptive SNR {adaptive} dB worse than fixed-coarse {fixed} dB"
    );
    assert!(
        adaptive >= fixed + 2.0,
        "adaptive SNR {adaptive} dB only marginally above fixed-coarse {fixed} dB (expected >=2 dB gain)"
    );
}

#[test]
fn adaptive_residue_round_trip_is_robust_across_partition_counts() {
    // The from-spectrum plan must round-trip cleanly whatever the partition
    // granularity is (every partition count dividing half_n).
    for &partitions in &[4usize, 8, 16, 32] {
        let (adaptive, _fixed, classifications) = adaptive_roundtrip(1024, partitions);
        assert_eq!(classifications.len(), partitions);
        assert!(
            adaptive >= 15.0,
            "{partitions}-partition adaptive SNR {adaptive} dB below pinned 15 dB"
        );
    }
}

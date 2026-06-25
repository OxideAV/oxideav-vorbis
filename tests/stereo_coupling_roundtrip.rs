//! Stereo channel-coupling PCM → encode → decode → PCM round-trip
//! (Vorbis I §4.3.5).
//!
//! Every existing encode round-trip in the crate is **mono** with empty
//! coupling; the §4.3.5 inverse-coupling step is exercised only on the
//! *decode* side (the `mode-stereo` fixture, the `audio.rs` driver unit
//! tests). This suite closes the gap: it drives a **stereo** signal through
//! the complete §4.3 packet path with a coupling step, proving the encoder's
//! forward coupling (`synthesis::forward_couple_all`) and the decoder's
//! §4.3.5 inverse coupling (`inverse_couple_all`) compose to reproduce the
//! original L/R signal.
//!
//! The chain, end to end:
//!
//!  1. **Analysis.** Synthetic stereo PCM (distinct L and R content) → two
//!     §4.3.1 windowed §4.3.7 forward MDCTs → per-channel analysis spectra
//!     `XL`, `XR`.
//!  2. **Forward coupling.** `forward_couple_all` turns `(XL, XR)` into the
//!     §4.3.5 square-polar `(magnitude, angle)` representation — the exact
//!     inverse of what the decoder undoes.
//!  3. **Residue.** With a flat floor (`F = 1`) per channel, each coupled
//!     vector *is* the residue target; `plan_partition_cascade` quantises
//!     the magnitude and angle vectors independently.
//!  4. **Encode.** `write_audio_packet` serialises the §4.3.1 prelude, the
//!     two flat floor-1 bodies, and the two residue bodies (one submap, the
//!     coupled pair).
//!  5. **Decode.** `decode_audio_packet_windowed` decodes residue, runs the
//!     §4.3.5 inverse coupling, the §4.3.6 dot product, the §4.3.7 IMDCT and
//!     the §4.3.6 window — emitting two channels.
//!  6. **Assert.** Each decoded channel tracks `window ⊙ IMDCT(X*)` (its own
//!     un-quantised reference) to a pinned PCM-domain SNR, and a control
//!     proves coupling actually ran (the coupled magnitude/angle vectors are
//!     not equal to the raw L/R spectra).
//!
//! Fully synthetic: no Ogg framing, no `docs/` fixtures, so it runs in
//! standalone CI.

use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::setup::{
    Floor1Header, FloorHeader, FloorKind, MappingCouplingStep, MappingHeader, MappingSubmap,
    ModeHeader, ResidueHeader, VorbisSetupHeader,
};
use oxideav_vorbis::{
    apply_window_and_mdct_vec, decode_audio_packet_windowed, forward_couple_all, imdct_naive_vec,
    plan_partition_cascade, write_audio_packet, AudioChannelFloor, AudioDecoderState,
    AudioPacketHeader, Floor1Packet, ResidueVectorPlan, WindowedPacketOutcome,
};

use oxideav_core::bits::BitReaderLsb;

fn vorbis_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let inner = (std::f32::consts::PI / n as f32) * (i as f32 + 0.5);
            let s = inner.sin();
            (std::f32::consts::FRAC_PI_2 * s * s).sin()
        })
        .collect()
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

fn classbook(entries: u32, length: u8) -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
}

/// Build the stereo setup: one **flat** floor-1 (no interior posts — both
/// endpoints carried in the packet body, rendering `F = 1`), one format-1
/// two-stage residue cascade (classbook 0, coarse value book 1, fine value
/// book 2), one mapping with a single coupling step (magnitude ch 0, angle
/// ch 1) and one submap carrying both channels.
fn stereo_setup(coarse: VorbisCodebook, fine: VorbisCodebook, half_n: u32) -> VorbisSetupHeader {
    let cb = classbook(2, 1);

    // Flat floor 1: multiplier 1 → range 256, no interior posts, so the
    // rendered curve is constant at INVERSE_DB_TABLE[endpoint post]. The
    // endpoint posts live in the audio-packet floor body, not here.
    let floor = FloorHeader {
        floor_type: 1,
        kind: FloorKind::Type1(Floor1Header {
            partitions: 0,
            partition_class_list: Vec::new(),
            classes: Vec::new(),
            multiplier: 1,
            rangebits: 4,
            x_list: Vec::new(),
        }),
    };
    let mut stages: [Option<u8>; 8] = Default::default();
    stages[0] = Some(1);
    stages[1] = Some(2);
    let cascade_bits: u8 = (1 << 0) | (1 << 1);
    let residue = ResidueHeader {
        residue_type: 1,
        residue_begin: 0,
        residue_end: half_n,
        partition_size: half_n,
        classifications: 1,
        classbook: 0,
        cascade: vec![cascade_bits],
        books: vec![stages],
    };

    VorbisSetupHeader {
        codebooks: vec![cb, coarse, fine],
        time_placeholders: Vec::new(),
        floors: vec![floor],
        residues: vec![residue],
        mappings: vec![MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: vec![MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            }],
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

/// Two distinct synthetic stereo channels so coupling has real work to do
/// (the magnitude/angle representation differs materially from raw L/R).
fn synthetic_stereo(n: usize) -> (Vec<f32>, Vec<f32>) {
    let left: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32;
            let nn = n as f32;
            0.70 * (2.0 * std::f32::consts::PI * 3.0 * t / nn).sin()
                + 0.30 * (2.0 * std::f32::consts::PI * 8.0 * t / nn).cos()
        })
        .collect();
    let right: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32;
            let nn = n as f32;
            0.50 * (2.0 * std::f32::consts::PI * 5.0 * t / nn).sin()
                + 0.25 * (2.0 * std::f32::consts::PI * 11.0 * t / nn).cos()
        })
        .collect();
    (left, right)
}

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

/// The flat-floor floor-1 packet: just the two endpoint posts, both at the
/// §10.1 table top (index 255 → `F = 1.0`); multiplier 1 → range 256. No
/// interior posts → no partition cvals.
fn flat_floor_packet() -> Floor1Packet {
    Floor1Packet {
        nonzero: true,
        floor1_y: vec![255, 255],
        partition_cvals: Vec::new(),
    }
}

/// Run the whole stereo-coupling round-trip for block size `n`, returning
/// the per-channel PCM-domain SNR (dB) against each channel's own
/// `window ⊙ IMDCT(X*)` reference.
fn coupling_roundtrip(n: usize, blocksize_1: usize) -> (f32, f32) {
    let half_n = n / 2;
    let window = vorbis_window(n);

    // 1. analysis: forward window + MDCT for each channel.
    let (mut l_pcm, mut r_pcm) = synthetic_stereo(n);
    let xl = apply_window_and_mdct_vec(&mut l_pcm, &window, 1.0).expect("L MDCT");
    let xr = apply_window_and_mdct_vec(&mut r_pcm, &window, 1.0).expect("R MDCT");
    assert_eq!(xl.len(), half_n);

    // 2. forward coupling (mag = ch0, ang = ch1).
    let coupling = vec![MappingCouplingStep {
        magnitude_channel: 0,
        angle_channel: 1,
    }];
    let mut coupled = vec![xl.clone(), xr.clone()];
    forward_couple_all(&mut coupled, &coupling).expect("forward couple");
    let (mag, ang) = (coupled[0].clone(), coupled[1].clone());

    // Control: coupling actually transformed the data (mag/ang differ from
    // the raw L/R spectra), so the round-trip genuinely exercises §4.3.5.
    let changed = mag
        .iter()
        .zip(xl.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-6)
        || ang
            .iter()
            .zip(xr.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-6);
    assert!(changed, "forward coupling must transform the L/R spectra");

    // 3. residue target = coupled vectors / flat floor (F = 1). A two-stage
    //    coarse+fine cascade so each vector quantises tightly.
    let max_abs = mag
        .iter()
        .chain(ang.iter())
        .fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(max_abs > 0.0);
    let coarse_step = max_abs / 24.0;
    let fine_step = coarse_step / 8.0;
    let coarse = signed_value_book(6, coarse_step);
    let fine = signed_value_book(6, fine_step);

    let mut refs: [Option<&VorbisCodebook>; 8] = Default::default();
    refs[0] = Some(&coarse);
    refs[1] = Some(&fine);
    let mag_entries =
        plan_partition_cascade(&mag, &refs, 1, half_n as u32).expect("magnitude residue plan");
    let ang_entries =
        plan_partition_cascade(&ang, &refs, 1, half_n as u32).expect("angle residue plan");

    // 4. setup + plans + serialise.
    let setup = stereo_setup(coarse.clone(), fine.clone(), half_n as u32);

    let floors = vec![
        AudioChannelFloor::Type1(flat_floor_packet()),
        AudioChannelFloor::Type1(flat_floor_packet()),
    ];
    // One submap carrying both channels → two residue vectors (mag, ang).
    let residue_plans = vec![vec![
        ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![mag_entries],
        },
        ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![ang_entries],
        },
    ]];

    let header = AudioPacketHeader {
        mode_number: 0,
        blockflag: false,
        n,
        previous_window_flag: false,
        next_window_flag: false,
    };
    let bytes = write_audio_packet(&header, &setup, n, blocksize_1, 2, &floors, &residue_plans)
        .expect("stereo audio packet serialises");
    assert!(!bytes.is_empty());

    // 5. decode back to two windowed frames (the decoder inverse-couples).
    let state = AudioDecoderState::new(&setup).expect("decoder state");
    let mut reader = BitReaderLsb::new(&bytes);
    let outcome = decode_audio_packet_windowed(&mut reader, &setup, &state, 2, n, blocksize_1, 1.0)
        .expect("stereo packet decodes");
    let frames = match &outcome {
        WindowedPacketOutcome::Windowed { frames, .. } => {
            assert_eq!(frames.len(), 2, "two decoded channels");
            frames.clone()
        }
        WindowedPacketOutcome::ZeroedWindowed { .. } => panic!("must not zero the packet"),
    };

    // 6. references: window ⊙ IMDCT(X*) per channel.
    let ref_frame = |x: &[f32]| -> Vec<f32> {
        let t = imdct_naive_vec(x, 1.0).expect("reference IMDCT");
        t.iter().zip(window.iter()).map(|(&v, &w)| v * w).collect()
    };
    let ref_l = ref_frame(&xl);
    let ref_r = ref_frame(&xr);

    (snr_db(&ref_l, &frames[0]), snr_db(&ref_r, &frames[1]))
}

#[test]
fn stereo_coupling_round_trips_to_time_domain() {
    let (snr_l, snr_r) = coupling_roundtrip(256, 1024);
    assert!(
        snr_l >= 25.0,
        "left-channel coupling round-trip SNR {snr_l} dB below 25 dB"
    );
    assert!(
        snr_r >= 25.0,
        "right-channel coupling round-trip SNR {snr_r} dB below 25 dB"
    );
}

#[test]
fn stereo_coupling_round_trip_is_robust_across_block_sizes() {
    for &n in &[128usize, 256, 512] {
        let blocksize_1 = (n * 4).max(1024);
        let (snr_l, snr_r) = coupling_roundtrip(n, blocksize_1);
        assert!(
            snr_l >= 20.0 && snr_r >= 20.0,
            "coupling round-trip at n={n}: SNR L={snr_l} R={snr_r} dB below 20 dB"
        );
    }
}

//! Residue **format 2** multi-channel PCM → encode → decode → PCM
//! round-trip (Vorbis I §8.6.5).
//!
//! Residue format 2 is the §8.6.5 "interleave all channels into one virtual
//! vector, format-1-decode it, then de-interleave" mode. The decoder side is
//! unit-tested (`residue.rs` interleave/de-interleave), but no *encode→decode*
//! round-trip drove format 2 through the full §4.3 audio packet: every
//! existing packet round-trip uses a per-channel format-1 residue. This suite
//! closes that gap.
//!
//! The chain: two-channel synthetic PCM → two windowed forward MDCTs → flat
//! floor (`F = 1`) per channel → **interleave** the two channels' residue
//! targets into one virtual vector (`interleaved[i·ch + j] = channel[j][i]`,
//! the exact inverse of the §8.6.5 step-3 de-interleave) → `plan_partition_cascade`
//! the single interleaved vector → `write_audio_packet` with a `residue_type:
//! 2` header (one residue plan, the interleaved vector) →
//! `decode_audio_packet_windowed` (which interleaves, format-1-decodes, and
//! de-interleaves) → two channels. Each decoded channel tracks its own
//! `window ⊙ IMDCT(X)` reference to a pinned SNR.
//!
//! Fully synthetic: no Ogg framing, no `docs/` fixtures, so it runs in
//! standalone CI.

use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::setup::{
    Floor1Header, FloorHeader, FloorKind, MappingHeader, MappingSubmap, ModeHeader, ResidueHeader,
    VorbisSetupHeader,
};
use oxideav_vorbis::{
    apply_window_and_mdct_vec, decode_audio_packet_windowed, imdct_naive_vec,
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

/// Build a 2-channel setup with a flat floor-1 (no interior posts) and a
/// **format-2** residue cascade over the interleaved `[0, half_n·2)` vector.
/// No coupling: format 2 interleaves the raw per-channel residues itself.
fn stereo_format2_setup(
    coarse: VorbisCodebook,
    fine: VorbisCodebook,
    interleaved_len: u32,
) -> VorbisSetupHeader {
    let cb = classbook(2, 1);
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
    // Format 2: the residue spans the single interleaved vector of length
    // `half_n · ch`; partition_size divides it.
    let residue = ResidueHeader {
        residue_type: 2,
        residue_begin: 0,
        residue_end: interleaved_len,
        partition_size: interleaved_len,
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

fn flat_floor_packet() -> Floor1Packet {
    Floor1Packet {
        nonzero: true,
        floor1_y: vec![255, 255],
        partition_cvals: Vec::new(),
    }
}

/// Run the format-2 two-channel round-trip for block size `n`, returning the
/// per-channel PCM SNR (dB) against `window ⊙ IMDCT(X)`.
fn format2_roundtrip(n: usize, blocksize_1: usize) -> (f32, f32) {
    let half_n = n / 2;
    let ch = 2usize;
    let interleaved_len = half_n * ch;
    let window = vorbis_window(n);

    // 1. analysis MDCTs.
    let (mut l_pcm, mut r_pcm) = synthetic_stereo(n);
    let xl = apply_window_and_mdct_vec(&mut l_pcm, &window, 1.0).expect("L MDCT");
    let xr = apply_window_and_mdct_vec(&mut r_pcm, &window, 1.0).expect("R MDCT");

    // 2. flat floor (F = 1) → residue target is the spectrum itself.
    //    Interleave the two per-channel targets into the §8.6.5 virtual
    //    vector: interleaved[i·ch + j] = channel[j][i].
    let mut interleaved = vec![0.0f32; interleaved_len];
    for i in 0..half_n {
        interleaved[i * ch] = xl[i];
        interleaved[i * ch + 1] = xr[i];
    }

    let max_abs = interleaved.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(max_abs > 0.0);
    let coarse_step = max_abs / 24.0;
    let fine_step = coarse_step / 8.0;
    let coarse = signed_value_book(6, coarse_step);
    let fine = signed_value_book(6, fine_step);

    let mut refs: [Option<&VorbisCodebook>; 8] = Default::default();
    refs[0] = Some(&coarse);
    refs[1] = Some(&fine);
    let entries = plan_partition_cascade(&interleaved, &refs, 1, interleaved_len as u32)
        .expect("interleaved residue plan");

    // 3. setup + plans + serialise. One residue plan: the interleaved vector.
    let setup = stereo_format2_setup(coarse.clone(), fine.clone(), interleaved_len as u32);
    let floors = vec![
        AudioChannelFloor::Type1(flat_floor_packet()),
        AudioChannelFloor::Type1(flat_floor_packet()),
    ];
    let residue_plans = vec![vec![ResidueVectorPlan {
        classifications: vec![0],
        partition_entries: vec![entries],
    }]];

    let header = AudioPacketHeader {
        mode_number: 0,
        blockflag: false,
        n,
        previous_window_flag: false,
        next_window_flag: false,
    };
    let bytes = write_audio_packet(&header, &setup, n, blocksize_1, 2, &floors, &residue_plans)
        .expect("format-2 audio packet serialises");
    assert!(!bytes.is_empty());

    // 4. decode (the decoder interleaves, format-1-decodes, de-interleaves).
    let state = AudioDecoderState::new(&setup).expect("decoder state");
    let mut reader = BitReaderLsb::new(&bytes);
    let outcome = decode_audio_packet_windowed(&mut reader, &setup, &state, 2, n, blocksize_1, 1.0)
        .expect("format-2 packet decodes");
    let frames = match &outcome {
        WindowedPacketOutcome::Windowed { frames, .. } => {
            assert_eq!(frames.len(), 2, "two decoded channels");
            frames.clone()
        }
        WindowedPacketOutcome::ZeroedWindowed { .. } => panic!("must not zero the packet"),
    };

    // 5. references.
    let ref_frame = |x: &[f32]| -> Vec<f32> {
        let t = imdct_naive_vec(x, 1.0).expect("reference IMDCT");
        t.iter().zip(window.iter()).map(|(&v, &w)| v * w).collect()
    };
    (
        snr_db(&ref_frame(&xl), &frames[0]),
        snr_db(&ref_frame(&xr), &frames[1]),
    )
}

#[test]
fn residue_format2_round_trips_to_time_domain() {
    let (snr_l, snr_r) = format2_roundtrip(256, 1024);
    assert!(
        snr_l >= 30.0 && snr_r >= 30.0,
        "format-2 round-trip SNR L={snr_l} R={snr_r} dB below 30 dB"
    );
}

#[test]
fn residue_format2_round_trip_is_robust_across_block_sizes() {
    for &n in &[128usize, 256, 512] {
        let blocksize_1 = (n * 4).max(1024);
        let (snr_l, snr_r) = format2_roundtrip(n, blocksize_1);
        assert!(
            snr_l >= 25.0 && snr_r >= 25.0,
            "format-2 round-trip at n={n}: SNR L={snr_l} R={snr_r} dB below 25 dB"
        );
    }
}

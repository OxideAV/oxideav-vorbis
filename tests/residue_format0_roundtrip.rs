//! Residue **format 0** (§8.6.3 strided scatter) PCM → encode → decode →
//! PCM full-packet round-trip.
//!
//! Format 0 scatters each decoded VQ vector across the residue vector with a
//! stride: read `i`, element `j` lands at `i + j·step` (§8.6.3), versus the
//! contiguous append of formats 1/2 (§8.6.4). The strided scatter has
//! isolated residue-body coverage (`tests/residue_cascade_roundtrip.rs`
//! drives `write_residue_body` → `ResidueDecoder`), but no *audio-packet-level*
//! round-trip drove format 0 through the full §4.3 path — every packet
//! round-trip used a contiguous format-1/2 residue. This suite closes that
//! gap, the format-0 analogue of `tests/residue_format2_roundtrip.rs`.
//!
//! The chain: mono synthetic PCM → windowed forward MDCT → flat floor
//! (`F = 1`) → `plan_partition_cascade` with `residue_type = 0` (the
//! encode-side strided **gather** that inverts the §8.6.3 scatter) over a
//! 2-D value book (its `dimensions = 2` divides the partition per §8.6.3
//! step 1) → `write_audio_packet` with a `residue_type: 0` header →
//! `decode_audio_packet_windowed` (which strided-scatters the decoded
//! entries) → PCM, clearing a pinned SNR against `window ⊙ IMDCT(X)`.
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

fn classbook(entries: u32, length: u8) -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
}

/// A 2-D tessellation VQ value book over a signed ladder, for the §8.6.3
/// strided scatter (`dimensions = 2` must divide the partition size). Entry
/// `e` reconstructs to `(min + (e % axis)·step, min + (e / axis)·step)`.
fn signed_value_book_2d(length: u8, step: f32) -> VorbisCodebook {
    let entries: u32 = 1u32 << length;
    let axis: u32 = 1u32 << (length / 2);
    let half = axis / 2;
    let mut multiplicands: Vec<u32> = Vec::with_capacity(entries as usize * 2);
    for e in 0..entries {
        multiplicands.push(e % axis);
        multiplicands.push(e / axis);
    }
    VorbisCodebook {
        dimensions: 2,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::Tessellation {
            minimum_value: -(half as f32) * step,
            delta_value: step,
            value_bits: 8,
            sequence_p: false,
            multiplicands,
        },
    }
}

/// Mono setup: flat floor-1, **format-0** residue cascade (classbook 0,
/// coarse 2-D value book 1, fine 2-D value book 2) over `[0, half_n)`.
fn mono_format0_setup(
    coarse: VorbisCodebook,
    fine: VorbisCodebook,
    half_n: u32,
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
    let residue = ResidueHeader {
        residue_type: 0,
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

fn synthetic_pcm(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            let nn = n as f32;
            0.70 * (2.0 * std::f32::consts::PI * 3.0 * t / nn).sin()
                + 0.35 * (2.0 * std::f32::consts::PI * 9.0 * t / nn).cos()
                + 0.15 * (2.0 * std::f32::consts::PI * 17.0 * t / nn).sin()
        })
        .collect()
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

/// Run the format-0 mono round-trip for block size `n`, returning the PCM
/// SNR (dB) against `window ⊙ IMDCT(X)`.
fn format0_roundtrip(n: usize, blocksize_1: usize) -> f32 {
    let half_n = n / 2;
    let window = vorbis_window(n);

    // 1. analysis MDCT.
    let mut pcm = synthetic_pcm(n);
    let x = apply_window_and_mdct_vec(&mut pcm, &window, 1.0).expect("forward MDCT");
    assert_eq!(x.len(), half_n);

    // 2. flat floor (F = 1) → residue target is the spectrum itself. Plan
    //    the format-0 strided gather (residue_type 0) over a 2-D value book.
    let max_abs = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(max_abs > 0.0);
    // 2-D books of length 10 → 1024 entries → 32 levels per axis; coarse
    // covers the full ±max range (16·step ≈ 1.3·max) and fine refines at 1/8
    // the step. Two cascade stages clear the SNR bar without an oversized
    // book (keeping the per-partition entry scan fast).
    let coarse_step = max_abs / 12.0;
    let fine_step = coarse_step / 8.0;
    let coarse = signed_value_book_2d(10, coarse_step);
    let fine = signed_value_book_2d(10, fine_step);

    let mut refs: [Option<&VorbisCodebook>; 8] = Default::default();
    refs[0] = Some(&coarse);
    refs[1] = Some(&fine);
    let entries =
        plan_partition_cascade(&x, &refs, 0, half_n as u32).expect("format-0 strided residue plan");

    // 3. setup + plans + serialise.
    let setup = mono_format0_setup(coarse.clone(), fine.clone(), half_n as u32);
    let floors = vec![AudioChannelFloor::Type1(flat_floor_packet())];
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
    let bytes = write_audio_packet(&header, &setup, n, blocksize_1, 1, &floors, &residue_plans)
        .expect("format-0 audio packet serialises");
    assert!(!bytes.is_empty());

    // 4. decode (the decoder strided-scatters the entries).
    let state = AudioDecoderState::new(&setup).expect("decoder state");
    let mut reader = BitReaderLsb::new(&bytes);
    let outcome = decode_audio_packet_windowed(&mut reader, &setup, &state, 1, n, blocksize_1, 1.0)
        .expect("format-0 packet decodes");
    let frame = match &outcome {
        WindowedPacketOutcome::Windowed { frames, .. } => {
            assert_eq!(frames.len(), 1);
            frames[0].clone()
        }
        WindowedPacketOutcome::ZeroedWindowed { .. } => panic!("must not zero the packet"),
    };

    // 5. reference.
    let t = imdct_naive_vec(&x, 1.0).expect("reference IMDCT");
    let ref_frame: Vec<f32> = t.iter().zip(window.iter()).map(|(&v, &w)| v * w).collect();
    snr_db(&ref_frame, &frame)
}

#[test]
fn residue_format0_round_trips_to_time_domain() {
    let snr = format0_roundtrip(256, 1024);
    assert!(snr >= 25.0, "format-0 round-trip SNR {snr} dB below 25 dB");
}

#[test]
fn residue_format0_round_trip_is_robust_across_block_sizes() {
    for &n in &[128usize, 256, 512] {
        let blocksize_1 = (n * 4).max(1024);
        let snr = format0_roundtrip(n, blocksize_1);
        assert!(
            snr >= 20.0,
            "format-0 round-trip at n={n}: SNR {snr} dB below 20 dB"
        );
    }
}

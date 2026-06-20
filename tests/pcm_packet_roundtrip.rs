//! Full §4.3 PCM → encode → decode → PCM time-domain round-trip.
//!
//! This is the crate's first **end-to-end audio-packet** round-trip that
//! returns to the time domain: it drives a synthetic PCM analysis frame
//! through the entire encode side ([`write_audio_packet`], composing the
//! §4.3.1 prelude + a flat floor-1 body + a real §8.6.2 residue body) and
//! then through the entire decode side
//! ([`decode_audio_packet_windowed`], running §4.3.2..§4.3.6 + the §4.3.7
//! IMDCT + the §4.3.6 window), and asserts a PCM-domain signal-to-noise
//! ratio against the time-domain frame the same IMDCT+window produces from
//! the un-quantised analysis spectrum.
//!
//! The chain, end to end:
//!
//!  1. **Analysis.** A length-`N` PCM block is windowed with the §4.3.1
//!     Vorbis window and forward-MDCT'd
//!     ([`apply_window_and_mdct_vec`]) to a length-`N/2` analysis spectrum
//!     `X` — the "audio spectrum vector" §4.3.7 names.
//!  2. **Floor.** A flat floor-1 (no interior posts, both endpoints at the
//!     same post value) renders a constant floor `F = INVERSE_DB_TABLE[v]`
//!     over the whole band. The §4.3.6 dot product reconstructs
//!     `spectrum[k] = F · residue[k]`, so the residue must carry `X[k]/F`.
//!  3. **Residue.** [`plan_partition_cascade`] quantises the per-bin
//!     residue target `X/F` into a §8.6.2 cascade; the full packet is
//!     serialised by [`write_audio_packet`].
//!  4. **Decode.** [`decode_audio_packet_windowed`] reads the packet back,
//!     reconstructs `spectrum' = F · quantise(X/F) ≈ X`, runs the §4.3.7
//!     IMDCT and §4.3.6 window, and returns one length-`N` windowed frame.
//!  5. **Assert.** The decoded windowed frame matches `window ⊙ IMDCT(X)`
//!     — the exact frame the decoder's own IMDCT+window produce from the
//!     un-quantised analysis spectrum — to a pinned PCM-domain SNR. The
//!     only loss is the residue quantiser; the floor, IMDCT, and window
//!     are bit-identical on both sides.
//!
//! Fully synthetic: no Ogg framing, no `docs/` fixtures, so it runs in
//! standalone CI.

use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::floor1::INVERSE_DB_TABLE;
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

/// §4.3.1 Vorbis window of length `n`: `w[i] = sin(π/2·sin²(π/n·(i+½)))`.
/// A centred long/short block with both halves full (no transition).
fn vorbis_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let inner = (std::f32::consts::PI / n as f32 * (i as f32 + 0.5)).sin();
            (std::f32::consts::FRAC_PI_2 * inner * inner).sin()
        })
        .collect()
}

/// A Kraft-complete 1-D tessellation VQ value book: `2^length` entries
/// all at codeword length `length`, ladder `(e − half)·step`.
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
/// length `length`. Used as the §8.6.2 classbook.
fn classbook(entries: u32, length: u8) -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
}

/// 10·log10 of the energy ratio between a target frame and the per-sample
/// error frame.
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

/// Build the mono setup header: codebook 0 = classbook, codebooks 1..=2 =
/// the residue cascade value books; one flat floor-1, one format-1 residue
/// covering `[0, half_n)`, one mapping, one short mode.
fn mono_setup(
    classbook: VorbisCodebook,
    coarse: VorbisCodebook,
    fine: VorbisCodebook,
    half_n: u32,
) -> VorbisSetupHeader {
    // Flat floor 1: multiplier 1 → range 256, no interior posts, so the
    // rendered curve is constant at INVERSE_DB_TABLE[endpoint post] — the
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

    // One classification, cascade stages 0 (coarse) + 1 (fine).
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
        codebooks: vec![classbook, coarse, fine],
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

/// A synthetic length-`n` PCM analysis block: three tones plus a slow
/// tilt, so the analysis spectrum has a non-flat magnitude shape worth
/// coding a residue against.
fn synthetic_pcm(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            0.55 * (2.0 * std::f32::consts::PI * 5.0 * t / n as f32).sin()
                + 0.30 * (2.0 * std::f32::consts::PI * 13.0 * t / n as f32).sin()
                + 0.12 * (2.0 * std::f32::consts::PI * 37.0 * t / n as f32).cos()
        })
        .collect()
}

/// Run the full PCM → encode → decode → PCM round-trip for one block size
/// `n` and return the achieved PCM-domain SNR (dB) against the reference
/// `window ⊙ IMDCT(X)` frame. `blocksize_1` is the long-block size the
/// setup advertises; the packet always uses the short mode (`blockflag =
/// false`, window length `n`).
fn pcm_roundtrip_snr(n: usize, blocksize_1: usize) -> f32 {
    let half_n = n / 2;
    let window = vorbis_window(n);

    // Step 1: forward window + MDCT → analysis spectrum X. scale 1.0 (the
    // analysis scale this crate's IMDCT inverts to 1.0).
    let mut pcm = synthetic_pcm(n);
    let x =
        apply_window_and_mdct_vec(&mut pcm, &window, 1.0).expect("forward MDCT of analysis block");
    assert_eq!(x.len(), half_n);

    // Step 2: flat floor F. The §10.1 INVERSE_DB_TABLE is exponential
    // (≈1.06e-7 at index 0, exactly 1.0 at index 255). Index 255 → F = 1.0,
    // so the residue target is X itself and the value books only need to
    // span X's range.
    let floor_post = 255i32; // multiplier 1 → table index 255 → F = 1.0.
    let f = INVERSE_DB_TABLE[floor_post as usize];
    assert!(
        (f - 1.0).abs() <= f32::EPSILON,
        "floor constant must be 1.0"
    );
    let residue_target: Vec<f32> = x.iter().map(|&xv| xv / f).collect();
    let max_abs = residue_target.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(max_abs > 0.0, "residue target is all-zero");

    // Step 3: residue cascade value books, sized to the residue target's
    // range. A length-6 book has 64 entries → span ±(32·step); choosing
    // `coarse_step = max_abs / 24` gives ±(1.33·max_abs) of headroom so no
    // bin clamps. The fine stage refines at 1/8 the coarse step, driving
    // the per-bin quantisation error to ≈ coarse_step/16.
    let coarse_step = max_abs / 24.0;
    let fine_step = coarse_step / 8.0;
    let cb = classbook(2, 1);
    let coarse = signed_value_book(6, coarse_step);
    let fine = signed_value_book(6, fine_step);
    let mut refs: [Option<&VorbisCodebook>; 8] = Default::default();
    refs[0] = Some(&coarse);
    refs[1] = Some(&fine);
    let entries = plan_partition_cascade(&residue_target, &refs, 1, half_n as u32)
        .expect("residue cascade plans for the whole spectrum");

    // Build the setup + the audio-packet floor and residue plans.
    let setup = mono_setup(cb, coarse.clone(), fine.clone(), half_n as u32);
    let floors = vec![AudioChannelFloor::Type1(Floor1Packet {
        nonzero: true,
        // No interior posts → floor1_y is just the two endpoints, both at
        // `floor_post`, so the rendered curve is flat at F.
        floor1_y: vec![floor_post as u32, floor_post as u32],
        partition_cvals: Vec::new(),
    })];
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

    // Step 3 (cont.): serialise the full §4.3 audio packet.
    let bytes = write_audio_packet(&header, &setup, n, blocksize_1, 1, &floors, &residue_plans)
        .expect("audio packet serialises");
    assert!(!bytes.is_empty());

    // Step 4: decode the packet back to a windowed time-domain frame.
    let state = AudioDecoderState::new(&setup).expect("audio decoder state builds");
    let mut reader = BitReaderLsb::new(&bytes);
    let outcome = decode_audio_packet_windowed(&mut reader, &setup, &state, 1, n, blocksize_1, 1.0)
        .expect("audio packet decodes to windowed frames");
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

    // Step 5: the honest reference is the IMDCT + window of the *original*
    // analysis spectrum X — the un-quantised target. The decoded frame
    // differs only by the residue quantisation, so a high PCM-domain SNR
    // confirms the whole §4.3 packet path (floor render, dot product,
    // IMDCT, window) is correct.
    let ref_time = imdct_naive_vec(&x, 1.0).expect("reference IMDCT of analysis spectrum");
    let ref_frame: Vec<f32> = ref_time
        .iter()
        .zip(window.iter())
        .map(|(&t, &w)| t * w)
        .collect();

    // The decoded frame must carry real energy (a non-trivial signal, not
    // silence) and be finite everywhere.
    let energy: f64 = decoded_frame.iter().map(|&s| (s as f64) * (s as f64)).sum();
    assert!(energy > 1e-6, "decoded frame is silent (energy {energy})");
    assert!(
        decoded_frame.iter().all(|s| s.is_finite()),
        "decoded frame has non-finite samples"
    );

    snr_db(&ref_frame, &decoded_frame)
}

#[test]
fn pcm_analysis_packet_round_trips_to_time_domain() {
    // The two-stage residue cascade (coarse 1/24-of-range step refined at
    // 1/8 that step) drives the reconstruction well above 40 dB in
    // practice; pin a conservative 30 dB floor that still proves the whole
    // path (floor render, §4.3.6 dot product, §4.3.7 IMDCT, §4.3.6 window)
    // is correct end to end. The setup advertises a 1024 long block; this
    // packet uses the 256 short block.
    let snr = pcm_roundtrip_snr(256, 1024);
    assert!(
        snr >= 30.0,
        "PCM time-domain round-trip SNR {snr} dB below pinned 30 dB"
    );
}

#[test]
fn pcm_round_trip_is_robust_across_block_sizes() {
    // The §4.3 path must not be pinned to one geometry: sweep the smallest
    // legal block (64), a typical short block (256), and a typical long
    // block (1024). Each clears the same conservative SNR floor, proving
    // the window build, IMDCT size dispatch, residue partitioning and dot
    // product all scale with `n`.
    for &n in &[64usize, 256, 1024] {
        let blocksize_1 = (n * 4).max(1024);
        let snr = pcm_roundtrip_snr(n, blocksize_1);
        assert!(
            snr >= 30.0,
            "block size {n}: PCM round-trip SNR {snr} dB below pinned 30 dB"
        );
    }
}

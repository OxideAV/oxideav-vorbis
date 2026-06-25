//! Floor-0 PCM → encode → decode → PCM round-trip fidelity.
//!
//! The existing `tests/nonflat_floor_pcm_roundtrip.rs` proves the full §4.3
//! audio-packet path end to end with a non-flat **floor 1**. No staged
//! fixture exercises **floor 0** (no reference encoder emits it), and the
//! floor-0 encode chain — `plan_floor0_packet` (envelope → LSP coefficients
//! → amplitude → entry run) — had no *audio-packet-level* round-trip: its
//! coverage stopped at the isolated floor body
//! (`tests/floor0_envelope_roundtrip.rs`). This suite closes that gap: it
//! drives PCM through the complete §4.3 packet with a floor-0 floor, proving
//! the floor-0 curve, the §4.3.6 dot product, the §4.3.7 IMDCT and the
//! §4.3.6 window all compose correctly when the floor is LSP-based.
//!
//! The chain, end to end:
//!
//!  1. **Analysis.** Synthetic length-`N` PCM → §4.3.1 window → §4.3.7
//!     forward MDCT → length-`N/2` analysis spectrum `X`.
//!  2. **Floor-0 fit.** A smoothed `|X|` envelope → `plan_floor0_packet` (the
//!     autocorrelation → Levinson-Durbin → LSP all-pole fit + amplitude + VQ
//!     entry run). The floor the decoder will reconstruct is recovered with
//!     `Floor0Decoder::render_curve` over the rebuilt coefficients.
//!  3. **Residue against the rendered floor.** Per §4.3.6 the decoder
//!     rebuilds `spectrum[k] = floor[k] · residue[k]`, so the residue target
//!     is `X[k] / rendered_floor[k]` — the exact per-bin multiplier the
//!     decoder reapplies — quantised by `plan_partition_cascade`.
//!  4. **Encode.** `write_audio_packet` serialises the §4.3.1 prelude, the
//!     floor-0 body (`AudioChannelFloor::Type0`), and the residue body.
//!  5. **Decode.** `decode_audio_packet_windowed` rebuilds `floor · residue`,
//!     runs the §4.3.7 IMDCT and §4.3.6 window.
//!  6. **Assert.** The decoded windowed frame tracks `window ⊙ IMDCT(X)` to
//!     a pinned PCM-domain SNR; the rendered floor-0 curve is genuinely
//!     non-flat.
//!
//! Fully synthetic: no Ogg framing, no `docs/` fixtures, so it runs in
//! standalone CI.

use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::floor0::Floor0Decoder;
use oxideav_vorbis::floor0_envelope::plan_floor0_packet;
use oxideav_vorbis::setup::{
    Floor0Header, FloorHeader, FloorKind, MappingHeader, MappingSubmap, ModeHeader, ResidueHeader,
    VorbisSetupHeader,
};
use oxideav_vorbis::{
    apply_window_and_mdct_vec, decode_audio_packet_windowed, imdct_naive_vec,
    plan_partition_cascade, write_audio_packet, AudioChannelFloor, AudioDecoderState,
    AudioPacketHeader, Floor0Packet, ResidueVectorPlan, WindowedPacketOutcome,
};

use oxideav_core::bits::BitReaderLsb;

/// §4.3.1 Vorbis window of length `n`: `w[i] = sin(π/2·sin²(π/n·(i+½)))`.
fn vorbis_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let inner = (std::f32::consts::PI / n as f32) * (i as f32 + 0.5);
            let s = inner.sin();
            (std::f32::consts::FRAC_PI_2 * s * s).sin()
        })
        .collect()
}

/// A signed lookup-2 residue value book: `2^length` entries centred on zero,
/// step `step`.
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

/// A scalar classbook (lookup_type 0): `entries` codewords of width `length`.
fn classbook(entries: u32, length: u8) -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
}

/// The fine-resolution floor-0 LSP value book (256 entries, scalar, step
/// 0.01 over `[-0.5, +2.05]`) — the same shape `floor0_envelope_roundtrip`
/// uses, fine enough that coefficient quantisation barely perturbs the curve.
fn lsp_value_book() -> VorbisCodebook {
    let entries: u32 = 256;
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![8; entries as usize],
        lookup: VqLookup::Tessellation {
            minimum_value: -0.5,
            delta_value: 0.01,
            value_bits: 8,
            sequence_p: false,
            multiplicands: (0..entries).collect(),
        },
    }
}

fn floor0_header(order: u8) -> Floor0Header {
    Floor0Header {
        order,
        rate: 44_100,
        bark_map_size: 256,
        amplitude_bits: 10,
        amplitude_offset: 32,
        book_list: vec![0], // floor0 value book is codebook 0
    }
}

/// Mono setup carrying a floor-0 floor (value book at codebook 0) plus a
/// format-1 two-stage residue cascade over `[0, half_n)` (classbook 1,
/// cascade value books 2/3).
fn mono_setup(order: u8, half_n: u32) -> (VorbisSetupHeader, VorbisCodebook) {
    let lsp_book = lsp_value_book();
    let classbook0 = classbook(2, 1);
    let coarse = signed_value_book(6, 1.0);
    let fine = signed_value_book(6, 1.0 / 8.0);

    let floor = FloorHeader {
        floor_type: 0,
        kind: FloorKind::Type0(floor0_header(order)),
    };

    let mut stages: [Option<u8>; 8] = Default::default();
    stages[0] = Some(2);
    stages[1] = Some(3);
    let cascade_bits: u8 = (1 << 0) | (1 << 1);
    let residue = ResidueHeader {
        residue_type: 1,
        residue_begin: 0,
        residue_end: half_n,
        partition_size: half_n,
        classifications: 1,
        classbook: 1,
        cascade: vec![cascade_bits],
        books: vec![stages],
    };

    let setup = VorbisSetupHeader {
        codebooks: vec![lsp_book.clone(), classbook0, coarse, fine],
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
    };
    (setup, lsp_book)
}

/// Synthetic length-`n` PCM: a few low-frequency tones at decreasing
/// amplitude so `|X|` has a real spectral tilt worth fitting a floor to.
fn synthetic_pcm(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            let nn = n as f32;
            0.80 * (2.0 * std::f32::consts::PI * 3.0 * t / nn).sin()
                + 0.40 * (2.0 * std::f32::consts::PI * 9.0 * t / nn).sin()
                + 0.16 * (2.0 * std::f32::consts::PI * 19.0 * t / nn).cos()
                + 0.06 * (2.0 * std::f32::consts::PI * 33.0 * t / nn).sin()
        })
        .collect()
}

/// A smooth descending magnitude envelope bounding `|X|` from above with a
/// wide-dynamic-range exponential tilt — so the fitted floor-0 curve spans a
/// genuine range and the residue target `X/floor` stays in book range.
fn magnitude_envelope(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let peak = x.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let n_f = (n.max(1) - 1).max(1) as f32;
    (0..n)
        .map(|k| {
            let frac = k as f32 / n_f;
            let smooth = peak * (64.0f32).powf(-frac);
            smooth.max(x[k].abs()).max(peak / 64.0)
        })
        .collect()
}

/// Reconstruct the §6.2.2 coefficients from a floor-0 entry run + a
/// tessellation value book (the decode-side `[last]` accumulation).
fn reconstruct_coefficients(entries: &[u32], book: &VorbisCodebook, order: usize) -> Vec<f32> {
    let dims = book.dimensions as usize;
    let VqLookup::Tessellation {
        minimum_value,
        delta_value,
        multiplicands,
        ..
    } = &book.lookup
    else {
        panic!("oracle only supports tessellation books");
    };
    let mut coeffs = Vec::new();
    let mut last = 0.0f32;
    for &entry in entries {
        let base = entry as usize * dims;
        let mut temp: Vec<f32> = (0..dims)
            .map(|j| multiplicands[base + j] as f32 * delta_value + minimum_value)
            .collect();
        for v in &mut temp {
            *v += last;
        }
        last = *temp.last().unwrap();
        coeffs.extend_from_slice(&temp);
        if coeffs.len() >= order {
            break;
        }
    }
    while coeffs.len() % dims != 0 {
        coeffs.push(0.0);
    }
    coeffs
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

/// Run the whole floor-0 round-trip for block size `n`. Returns the PCM
/// SNR (dB) against `window ⊙ IMDCT(X)` and the rendered floor-0 curve.
fn floor0_roundtrip(n: usize, blocksize_1: usize, order: u8) -> (f32, Vec<f32>) {
    let half_n = n / 2;
    let window = vorbis_window(n);

    // 1. forward window + MDCT → analysis spectrum X.
    let mut pcm = synthetic_pcm(n);
    let x =
        apply_window_and_mdct_vec(&mut pcm, &window, 1.0).expect("forward MDCT of analysis block");
    assert_eq!(x.len(), half_n);

    // 2. fit a floor-0 to a smoothed |X| envelope → a write-ready packet.
    let envelope = magnitude_envelope(&x);
    let (setup, lsp_book) = mono_setup(order, half_n as u32);
    let hdr0 = floor0_header(order);
    let packet =
        plan_floor0_packet(&hdr0, &setup.codebooks, 0, &envelope).expect("floor-0 packet plan");
    let (amplitude, entries) = match &packet {
        Floor0Packet::Curve {
            amplitude, entries, ..
        } => (*amplitude, entries.clone()),
        Floor0Packet::Unused => panic!("a nonzero envelope must plan a Curve"),
    };

    // 3. render the floor the decoder will actually reconstruct (the §6.2.3
    //    curve over the rebuilt coefficients) and divide the spectrum by it.
    let coeffs = reconstruct_coefficients(&entries, &lsp_book, order as usize);
    let decoder =
        Floor0Decoder::new(&hdr0, std::slice::from_ref(&lsp_book)).expect("floor-0 decoder builds");
    let rendered_floor = decoder.render_curve(amplitude, &coeffs, half_n);
    assert_eq!(rendered_floor.len(), half_n);
    assert!(
        rendered_floor.iter().all(|&f| f > 0.0 && f.is_finite()),
        "rendered floor must be strictly positive and finite"
    );

    let residue_target: Vec<f32> = x
        .iter()
        .zip(rendered_floor.iter())
        .map(|(&xv, &fv)| xv / fv)
        .collect();
    let max_abs = residue_target.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(max_abs > 0.0, "residue target is all-zero");

    // residue cascade value books sized to the target's range.
    let coarse_step = max_abs / 24.0;
    let fine_step = coarse_step / 8.0;
    let coarse = signed_value_book(6, coarse_step);
    let fine = signed_value_book(6, fine_step);
    let mut refs: [Option<&VorbisCodebook>; 8] = Default::default();
    refs[0] = Some(&coarse);
    refs[1] = Some(&fine);
    let res_entries = plan_partition_cascade(&residue_target, &refs, 1, half_n as u32)
        .expect("residue cascade plans for the whole spectrum");

    // Rebuild the setup with the *sized* residue cascade books (codebooks
    // 2/3), keeping floor book (0) and classbook (1).
    let mut setup = setup;
    setup.codebooks[2] = coarse.clone();
    setup.codebooks[3] = fine.clone();

    // 4. build the audio-packet plans + serialise the §4.3 packet.
    let floors = vec![AudioChannelFloor::Type0(packet)];
    let residue_plans = vec![vec![ResidueVectorPlan {
        classifications: vec![0],
        partition_entries: vec![res_entries],
    }]];
    let header = AudioPacketHeader {
        mode_number: 0,
        blockflag: false,
        n,
        previous_window_flag: false,
        next_window_flag: false,
    };
    let bytes = write_audio_packet(&header, &setup, n, blocksize_1, 1, &floors, &residue_plans)
        .expect("audio packet serialises");
    assert!(!bytes.is_empty());

    // 5. decode back to a windowed time-domain frame.
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

    // 6. reference = window ⊙ IMDCT(X) — the un-quantised target frame.
    let ref_time = imdct_naive_vec(&x, 1.0).expect("reference IMDCT of analysis spectrum");
    let ref_frame: Vec<f32> = ref_time
        .iter()
        .zip(window.iter())
        .map(|(&t, &w)| t * w)
        .collect();

    let energy: f64 = decoded_frame.iter().map(|&s| (s as f64) * (s as f64)).sum();
    assert!(energy > 1e-6, "decoded frame is silent (energy {energy})");
    assert!(
        decoded_frame.iter().all(|s| s.is_finite()),
        "decoded frame has non-finite samples"
    );

    (snr_db(&ref_frame, &decoded_frame), rendered_floor)
}

#[test]
fn floor0_pcm_round_trips_to_time_domain() {
    // The floor-0 LSP curve, divided out of the spectrum before residue
    // coding, keeps the whole §4.3 path faithful. The two-stage cascade
    // clears a conservative SNR floor.
    let (snr, _) = floor0_roundtrip(256, 1024, 14);
    assert!(
        snr >= 30.0,
        "floor-0 PCM round-trip SNR {snr} dB below pinned 30 dB"
    );
}

#[test]
fn floor0_rendered_curve_is_genuinely_non_flat() {
    // Guard against a degenerate fit that flattened the floor.
    let (_, floor) = floor0_roundtrip(256, 1024, 14);
    let lo = floor.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = floor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(lo > 0.0 && hi.is_finite());
    assert!(
        hi / lo >= 2.0,
        "rendered floor-0 span {hi}/{lo} too flat to be representative"
    );
}

#[test]
fn floor0_round_trip_is_robust_across_block_sizes() {
    for &n in &[128usize, 256, 512] {
        let blocksize_1 = (n * 4).max(1024);
        let (snr, _) = floor0_roundtrip(n, blocksize_1, 12);
        assert!(
            snr >= 20.0,
            "floor-0 round-trip at n={n} SNR {snr} dB below 20 dB floor"
        );
    }
}

#[test]
fn odd_order_floor0_round_trips() {
    // The odd-parity §6.2.3 LSP-product branch through the full packet path.
    let (snr, _) = floor0_roundtrip(256, 1024, 13);
    assert!(
        snr >= 25.0,
        "odd-order floor-0 round-trip SNR {snr} dB below 25 dB"
    );
}

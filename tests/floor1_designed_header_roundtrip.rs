//! Designed-floor-1-header PCM → encode → decode → PCM round-trip.
//!
//! The existing `tests/nonflat_floor_pcm_roundtrip.rs` drives a non-flat
//! floor-1 round-trip from a **hand-built** header — the interior post
//! coordinates and the partition/class layout are written by the test. This
//! suite closes the last gap in the floor-1 encode path: the header itself
//! is **designed from the spectrum** by `design_floor1_header`. The test
//! supplies only an envelope and a class catalogue; the layout planner
//! chooses where the posts go (`plan_floor1_x_list`), how they tile into
//! §7.2.2 partitions (`plan_floor1_partition_layout`), and the covering
//! `rangebits` — then the existing per-packet chain fits, wraps, and
//! serialises against that designed header.
//!
//! The chain, end to end:
//!
//!  1. **Analysis.** Synthetic length-`N` PCM → §4.3.1 window → §4.3.7
//!     forward MDCT → length-`N/2` analysis spectrum `X`.
//!  2. **Header design.** A smoothed `|X|` envelope and a class catalogue
//!     (dimensions `{1, 2, 4}`, all referencing the 256-entry floor value
//!     book) are handed to `design_floor1_header`, which returns a complete
//!     `Floor1Header` with adaptively-placed posts and a DP-tiled partition
//!     layout.
//!  3. **Floor fit.** `plan_floor1_envelope` → `plan_floor1_y` against the
//!     designed header; the rendered floor is recovered with
//!     `Floor1Decoder::render_curve`.
//!  4. **Residue against the rendered floor.** `X[k] / rendered_floor[k]`
//!     is quantised by `plan_partition_cascade`.
//!  5. **Encode + decode.** `write_audio_packet` → `decode_audio_packet_windowed`.
//!  6. **Assert.** The decoded windowed frame tracks `window ⊙ IMDCT(X)` to
//!     a pinned PCM-domain SNR, the designed header is structurally valid
//!     (its decoder builds), and its post count + partition tiling are
//!     self-consistent.
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
    apply_window_and_mdct_vec, decode_audio_packet_windowed, design_floor1_header, imdct_naive_vec,
    plan_floor1_envelope, plan_floor1_y, plan_partition_cascade, write_audio_packet,
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

/// A Kraft-complete 1-D tessellation VQ value book.
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

/// A balanced 1-D scalar book (no VQ lookup): `entries` codewords of
/// `length` bits. Used as the floor value book and the residue classbook.
fn scalar_book(entries: u32, length: u8) -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
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

/// The class catalogue handed to `design_floor1_header`: three classes of
/// dimension 1, 2, and 4, all `subclasses = 0` and all carrying the
/// 256-entry floor value book at codebook index 0 in sub-book slot 0. The
/// designer picks *which* class (by dimension) tiles each partition; every
/// dimension reads its Y values from the same value book.
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

/// Build the mono setup header carrying the designed floor-1 and a format-1
/// two-stage residue cascade over `[0, half_n)`.
fn mono_setup(
    floor_header: Floor1Header,
    floor_book: VorbisCodebook,
    classbook: VorbisCodebook,
    coarse: VorbisCodebook,
    fine: VorbisCodebook,
    half_n: u32,
) -> VorbisSetupHeader {
    let floor = FloorHeader {
        floor_type: 1,
        kind: FloorKind::Type1(floor_header),
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
            0.80 * (2.0 * std::f32::consts::PI * 3.0 * t / nn).sin()
                + 0.40 * (2.0 * std::f32::consts::PI * 9.0 * t / nn).sin()
                + 0.16 * (2.0 * std::f32::consts::PI * 19.0 * t / nn).cos()
                + 0.06 * (2.0 * std::f32::consts::PI * 33.0 * t / nn).sin()
                + 0.02 * (2.0 * std::f32::consts::PI * 51.0 * t / nn).cos()
        })
        .collect()
}

/// A wide-dynamic-range descending magnitude envelope that bounds `|X|`.
fn magnitude_envelope(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut peak = 0.0f32;
    for &v in x {
        peak = peak.max(v.abs());
    }
    let peak = peak.max(1e-6);
    let n_f = (n.max(1) - 1).max(1) as f32;
    let mut env = vec![0.0f32; n];
    for (k, e) in env.iter_mut().enumerate() {
        let frac = k as f32 / n_f;
        let smooth = peak * (256.0f32).powf(-frac);
        *e = smooth.max(x[k].abs()).max(peak / 256.0);
    }
    env
}

/// Run the whole designed-header round-trip for block size `n` with a post
/// budget `max_posts`. Returns the achieved PCM-domain SNR (dB) and the
/// designed header (so the caller can assert its structure).
fn designed_roundtrip(n: usize, blocksize_1: usize, max_posts: usize) -> (f32, Floor1Header) {
    let half_n = n / 2;
    let window = vorbis_window(n);

    // 1. forward window + MDCT → analysis spectrum X.
    let mut pcm = synthetic_pcm(n);
    let x =
        apply_window_and_mdct_vec(&mut pcm, &window, 1.0).expect("forward MDCT of analysis block");
    assert_eq!(x.len(), half_n);

    // 2. DESIGN the floor-1 header from the envelope + class catalogue.
    let envelope = magnitude_envelope(&x);
    let classes = class_catalogue();
    let floor_header = design_floor1_header(&envelope, max_posts, 0.0, 1, &classes)
        .expect("floor-1 header designs from the envelope");

    // The 256-entry floor value book the catalogue's classes reference.
    let floor_book = scalar_book(256, 8);

    // 3. fit + render against the designed header.
    let posts = plan_floor1_envelope(&envelope, &floor_header).expect("floor-1 envelope fit");
    let floor1_y = plan_floor1_y(&posts, &floor_header).expect("floor-1 amplitude wrap");
    let decoder = Floor1Decoder::new(&floor_header, std::slice::from_ref(&floor_book))
        .expect("designed floor-1 decoder builds");
    let rendered_floor = decoder.render_curve(&floor1_y, half_n);
    assert_eq!(rendered_floor.len(), half_n);
    assert!(
        rendered_floor.iter().all(|&f| f > 0.0 && f.is_finite()),
        "rendered floor must be strictly positive and finite"
    );

    // 3 (cont). residue target against the rendered floor.
    let residue_target: Vec<f32> = x
        .iter()
        .zip(rendered_floor.iter())
        .map(|(&xv, &fv)| xv / fv)
        .collect();
    let max_abs = residue_target.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(max_abs > 0.0, "residue target is all-zero");

    let coarse_step = max_abs / 24.0;
    let fine_step = coarse_step / 8.0;
    let cb = scalar_book(2, 1);
    let coarse = signed_value_book(6, coarse_step);
    let fine = signed_value_book(6, fine_step);
    let mut refs: [Option<&VorbisCodebook>; 8] = Default::default();
    refs[0] = Some(&coarse);
    refs[1] = Some(&fine);
    let entries = plan_partition_cascade(&residue_target, &refs, 1, half_n as u32)
        .expect("residue cascade plans for the whole spectrum");

    // 4. build setup + per-partition cval list + serialise the §4.3 packet.
    let setup = mono_setup(
        floor_header.clone(),
        floor_book.clone(),
        cb,
        coarse.clone(),
        fine.clone(),
        half_n as u32,
    );
    // Every class has subclasses = 0, so each partition's cval is the
    // §7.2.3 step-10 initial 0 (no master-book selector is emitted). One
    // cval per partition in the designed layout.
    let partition_cvals = vec![0u32; floor_header.partition_class_list.len()];
    let floors = vec![AudioChannelFloor::Type1(Floor1Packet {
        nonzero: true,
        floor1_y: floor1_y.clone(),
        partition_cvals,
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

    // 6. reference = window ⊙ IMDCT(X).
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

    (snr_db(&ref_frame, &decoded_frame), floor_header)
}

#[test]
fn designed_header_pcm_round_trips_to_time_domain() {
    // The floor-1 header is designed from the spectrum (not hand-built) and
    // the whole §4.3 path still reconstructs to a faithful time-domain
    // frame. The two-stage residue cascade clears a conservative SNR floor.
    let (snr, header) = designed_roundtrip(256, 1024, 12);
    assert!(
        snr >= 35.0,
        "designed-header PCM round-trip SNR {snr} dB below pinned 35 dB"
    );
    // The designed header is self-consistent: its partition tiling sums to
    // the explicit-post count, and it carries a real (non-endpoint-only)
    // floor for this peaky envelope.
    assert!(
        !header.x_list.is_empty(),
        "a tilted spectrum must design interior posts"
    );
    let tiled: usize = header
        .partition_class_list
        .iter()
        .map(|&ci| header.classes[ci as usize].dimensions as usize)
        .sum();
    assert_eq!(tiled, header.x_list.len(), "partitions must tile the x-list");
}

#[test]
fn designed_header_round_trip_is_robust_across_block_sizes() {
    // The designed-header path must scale with geometry: sweep a short and a
    // long block; each clears the same SNR floor with the same post budget.
    for &n in &[128usize, 256, 1024] {
        let blocksize_1 = (n * 4).max(1024);
        let (snr, _) = designed_roundtrip(n, blocksize_1, 12);
        assert!(
            snr >= 35.0,
            "block size {n}: designed-header round-trip SNR {snr} dB below pinned 35 dB"
        );
    }
}

#[test]
fn larger_post_budget_does_not_regress_fidelity() {
    // Designing with more posts spends more bits but must not reconstruct
    // worse than a smaller budget (the adaptive placement only refines the
    // approximation). Pin that an 18-post design is at least as faithful as
    // a 6-post one on the same spectrum.
    let (snr_few, _) = designed_roundtrip(256, 1024, 6);
    let (snr_many, _) = designed_roundtrip(256, 1024, 18);
    assert!(
        snr_many >= snr_few - 1.0,
        "more posts ({snr_many} dB) must not regress vs. fewer ({snr_few} dB)"
    );
}

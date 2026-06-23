//! Non-flat floor-1 PCM → encode → decode → PCM round-trip fidelity.
//!
//! The existing `tests/pcm_packet_roundtrip.rs` end-to-end round-trip uses
//! a **flat** floor (one constant `F = 1.0` across the band) so the residue
//! target is the analysis spectrum `X` itself. This suite closes the
//! harder, more representative case: a **non-flat** floor-1 fitted to the
//! spectrum's shape, with the residue carried against the floor the decoder
//! actually reconstructs.
//!
//! The fidelity hinge — and the reason `Floor1Decoder::render_curve` exists
//! — is §4.3.6 + §7.2.4. The decoder rebuilds the final spectrum as
//! `spectrum[k] = floor[k] · residue[k]`. §7.2.4 step 2 draws **integer
//! line segments** between the floor posts, so the reconstructed floor does
//! **not** equal the desired envelope sampled at the posts between them — it
//! bows along a Bresenham line in the dB-index domain. An encoder that
//! divided `X` by the *desired* envelope would hand residue the wrong
//! per-bin floor and the `floor · residue` product would miss `X`. Dividing
//! `X` by the **rendered** floor (`render_curve`) hands residue the exact
//! per-bin multiplier the decoder will reapply, so the round-trip stays
//! faithful across the whole band.
//!
//! The chain, end to end:
//!
//!  1. **Analysis.** Synthetic length-`N` PCM → §4.3.1 window → §4.3.7
//!     forward MDCT → length-`N/2` analysis spectrum `X`.
//!  2. **Floor fit.** A smoothed `|X|` envelope is fitted to a multi-post
//!     floor-1 (`plan_floor1_envelope` → `plan_floor1_y`); the rendered
//!     floor is recovered with `Floor1Decoder::render_curve`.
//!  3. **Residue against the rendered floor.** The per-bin target
//!     `X[k] / rendered_floor[k]` is quantised by `plan_partition_cascade`.
//!  4. **Encode.** `write_audio_packet` serialises the §4.3.1 prelude, the
//!     non-flat floor-1 body, and the residue body.
//!  5. **Decode.** `decode_audio_packet_windowed` rebuilds `floor · residue`,
//!     runs the §4.3.7 IMDCT and §4.3.6 window.
//!  6. **Assert.** The decoded windowed frame tracks `window ⊙ IMDCT(X)`
//!     (the un-quantised reference) to a pinned PCM-domain SNR; the floor is
//!     proven genuinely non-flat; and a control divide-by-envelope variant
//!     is shown to be measurably worse, pinning *why* the rendered-floor
//!     divide is the correct one.
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
    plan_floor1_y, plan_partition_cascade, write_audio_packet, AudioChannelFloor,
    AudioDecoderState, AudioPacketHeader, Floor1Packet, ResidueVectorPlan, WindowedPacketOutcome,
};

use oxideav_core::bits::BitReaderLsb;

/// §4.3.1 Vorbis window of length `n`: `w[i] = sin(π/2·sin²(π/n·(i+½)))`.
/// Both block halves full (a centred long block, no short-block taper).
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

/// A non-flat floor-1 header: `interior_x` interior posts plus the two
/// implicit endpoints, one partition / class, multiplier 1 (range 256),
/// `rangebits` large enough to hold every interior coordinate and the
/// upper endpoint. Only the geometry fields the renderer/fitter read are
/// load-bearing; the class metadata is a well-formed placeholder.
fn nonflat_floor_header(interior_x: Vec<u32>, rangebits: u8) -> Floor1Header {
    Floor1Header {
        partitions: 1,
        partition_class_list: vec![0],
        classes: vec![Floor1Class {
            dimensions: interior_x.len() as u8,
            subclasses: 0,
            // Subclass book 0 (the 256-entry floor value book at codebook
            // index 0) carries the wrapped interior Y values verbatim, so
            // the decoded posts equal the fitted ones exactly.
            masterbook: None,
            subclass_books: vec![Some(0)],
        }],
        multiplier: 1,
        rangebits,
        x_list: interior_x,
    }
}

/// Build the mono setup header carrying the supplied non-flat floor-1 and a
/// format-1 two-stage residue cascade over `[0, half_n)`. Codebook 0 is the
/// floor-1 value book (referenced by the floor's `subclass_books`); 1 is the
/// residue classbook; 2/3 are the residue cascade value books.
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

/// Synthetic length-`n` PCM: a few tones at decreasing amplitude across the
/// band so `|X|` has a real, monotone-ish *shape* (a spectral tilt) worth
/// fitting a non-flat floor to — not a flat magnitude.
fn synthetic_pcm(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            let nn = n as f32;
            // Strong low-frequency content tapering to weak high-frequency
            // content: a steep spectral tilt so the fitted floor spans a
            // genuinely wide dynamic range across the band.
            0.80 * (2.0 * std::f32::consts::PI * 3.0 * t / nn).sin()
                + 0.40 * (2.0 * std::f32::consts::PI * 9.0 * t / nn).sin()
                + 0.16 * (2.0 * std::f32::consts::PI * 19.0 * t / nn).cos()
                + 0.06 * (2.0 * std::f32::consts::PI * 33.0 * t / nn).sin()
                + 0.02 * (2.0 * std::f32::consts::PI * 51.0 * t / nn).cos()
        })
        .collect()
}

/// A descending magnitude envelope that bounds `|X|` from above with a
/// smooth exponential tilt across the band. The §10.1 dB table spans ~140
/// dB while raw MDCT magnitudes of a few tones span only a handful of dB,
/// so a tight `max(|X|)` envelope lands every post near the top of the
/// table and the fitted floor comes out nearly flat. Designing the
/// envelope as a wide-dynamic-range exponential decay (here ~48 dB head to
/// tail) makes the fitted floor genuinely non-flat — exercising the
/// integer-line-segment rendering between posts — while still sitting at or
/// above `|X|` so the residue target `X/floor` stays inside the value-book
/// range and never divides by zero.
fn magnitude_envelope(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut peak = 0.0f32;
    for &v in x {
        peak = peak.max(v.abs());
    }
    let peak = peak.max(1e-6);
    // Exponential decay from `peak` (bin 0) down to `peak / 256` (last bin):
    // a ~48 dB span. The decay is monotone, so the fitted posts descend
    // monotonically and the rendered floor spans the same wide range.
    let n_f = (n.max(1) - 1).max(1) as f32;
    let mut env = vec![0.0f32; n];
    for (k, e) in env.iter_mut().enumerate() {
        let frac = k as f32 / n_f;
        let smooth = peak * (256.0f32).powf(-frac);
        // Guarantee the envelope bounds |X| at this bin (a strong tone may
        // poke above the smooth tilt); take the larger of the two.
        *e = smooth.max(x[k].abs()).max(peak / 256.0);
    }
    env
}

/// Run the whole non-flat-floor round-trip for block size `n` with a floor
/// fitted to a `divide_target` of either the **rendered** floor (the
/// correct path) or the **desired envelope** (the control). Returns the
/// achieved PCM-domain SNR (dB) against `window ⊙ IMDCT(X)`, plus the
/// rendered floor curve so the caller can prove it is non-flat.
enum DivideBy {
    RenderedFloor,
    DesiredEnvelope,
}

fn nonflat_roundtrip(n: usize, blocksize_1: usize, divide_by: DivideBy) -> (f32, Vec<f32>) {
    let half_n = n / 2;
    let window = vorbis_window(n);

    // 1. forward window + MDCT → analysis spectrum X.
    let mut pcm = synthetic_pcm(n);
    let x =
        apply_window_and_mdct_vec(&mut pcm, &window, 1.0).expect("forward MDCT of analysis block");
    assert_eq!(x.len(), half_n);

    // 2. fit a non-flat floor-1 to a smoothed |X| envelope. Interior posts
    //    spread across the band so the rendered floor has real internal
    //    structure (a tilt), not a single flat segment.
    let envelope = magnitude_envelope(&x);
    let interior: Vec<u32> = {
        // Place interior posts at a handful of in-band coordinates spread
        // over [1, half_n). rangebits must hold the largest one and the
        // implicit upper endpoint 2^rangebits >= half_n.
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
    // rangebits: smallest b with 2^b >= half_n (upper endpoint covers the
    // whole band) and >= every interior coordinate.
    let mut rangebits = 1u8;
    while (1u32 << rangebits) < half_n as u32 {
        rangebits += 1;
    }
    let floor_header = nonflat_floor_header(interior, rangebits);

    let posts = plan_floor1_envelope(&envelope, &floor_header).expect("floor-1 envelope fit");
    let floor1_y = plan_floor1_y(&posts, &floor_header).expect("floor-1 amplitude wrap");

    // The 256-entry floor value book carries every wrapped interior Y in
    // [0, 256) verbatim (multiplier 1 → range 256).
    let floor_book = classbook(256, 8);

    // 3. render the floor the decoder will actually reconstruct.
    let decoder = Floor1Decoder::new(&floor_header, std::slice::from_ref(&floor_book))
        .expect("floor-1 decoder builds");
    let rendered_floor = decoder.render_curve(&floor1_y, half_n);
    assert_eq!(rendered_floor.len(), half_n);
    assert!(
        rendered_floor.iter().all(|&f| f > 0.0 && f.is_finite()),
        "rendered floor must be strictly positive and finite"
    );

    // 3 (cont). residue target = X / divisor. The correct divisor is the
    //    *rendered* floor (what the decoder multiplies back in). The control
    //    divides by the desired envelope to demonstrate it is worse.
    let divisor: Vec<f32> = match divide_by {
        DivideBy::RenderedFloor => rendered_floor.clone(),
        DivideBy::DesiredEnvelope => envelope.clone(),
    };
    let residue_target: Vec<f32> = x
        .iter()
        .zip(divisor.iter())
        .map(|(&xv, &fv)| xv / fv)
        .collect();
    let max_abs = residue_target.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(max_abs > 0.0, "residue target is all-zero");

    // residue cascade value books sized to the target's range.
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

    // 4. build setup + audio-packet plans + serialise the §4.3 packet.
    let setup = mono_setup(
        floor_header.clone(),
        floor_book.clone(),
        cb,
        coarse.clone(),
        fine.clone(),
        half_n as u32,
    );
    let floors = vec![AudioChannelFloor::Type1(Floor1Packet {
        nonzero: true,
        floor1_y: floor1_y.clone(),
        // One partition, class with subclasses = 0 → cval is the §7.2.3
        // step-10 initial 0 (no master-book selector is emitted).
        partition_cvals: vec![0],
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
fn nonflat_floor_pcm_round_trips_to_time_domain() {
    // Dividing the spectrum by the *rendered* floor before residue coding
    // keeps the whole §4.3 path faithful even with a multi-post, non-flat
    // floor. The two-stage cascade clears a conservative SNR floor.
    let (snr, _) = nonflat_roundtrip(256, 1024, DivideBy::RenderedFloor);
    assert!(
        snr >= 35.0,
        "non-flat-floor PCM round-trip SNR {snr} dB below pinned 35 dB"
    );
}

#[test]
fn rendered_floor_is_genuinely_non_flat() {
    // Guard against a degenerate fit that flattened the floor (which would
    // make this test no harder than the flat-floor round-trip). The fitted
    // floor must span a real dynamic range across the band.
    let (_, floor) = nonflat_roundtrip(256, 1024, DivideBy::RenderedFloor);
    let lo = floor.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = floor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(lo > 0.0 && hi.is_finite());
    assert!(
        hi / lo >= 2.0,
        "rendered floor span {hi}/{lo} too flat to exercise the non-flat path"
    );
}

#[test]
fn dividing_by_rendered_floor_beats_dividing_by_envelope() {
    // The fidelity hinge: §7.2.4 step 2 draws integer line segments between
    // posts, so the rendered floor differs from the desired envelope between
    // posts. An encoder that divides X by the *desired envelope* hands
    // residue the wrong per-bin floor; the decoder multiplies the *rendered*
    // floor back in, so the floor·residue product misses X. Dividing by the
    // rendered floor must therefore reconstruct substantially better.
    let (snr_rendered, _) = nonflat_roundtrip(256, 1024, DivideBy::RenderedFloor);
    let (snr_envelope, _) = nonflat_roundtrip(256, 1024, DivideBy::DesiredEnvelope);
    assert!(
        snr_rendered >= snr_envelope + 10.0,
        "rendered-floor divide ({snr_rendered} dB) must clearly beat envelope \
         divide ({snr_envelope} dB) — the floor bows away from the envelope \
         between posts, so the envelope divide hands residue the wrong floor"
    );
}

#[test]
fn nonflat_round_trip_is_robust_across_block_sizes() {
    // The non-flat path must scale with geometry like the flat one does:
    // sweep a short and a long block; each clears the same SNR floor.
    for &n in &[128usize, 256, 1024] {
        let blocksize_1 = (n * 4).max(1024);
        let (snr, _) = nonflat_roundtrip(n, blocksize_1, DivideBy::RenderedFloor);
        assert!(
            snr >= 35.0,
            "block size {n}: non-flat-floor round-trip SNR {snr} dB below pinned 35 dB"
        );
    }
}

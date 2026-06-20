//! Forward-MDCT → floor-1 envelope-fit → encode → decode round-trip.
//!
//! This is the milestone integration test for the floor-1 **encode**
//! chain: it drives a synthetic PCM analysis frame through the whole
//! encode-side spectral pipeline and confirms the crate's own decoder
//! reconstructs the floor envelope to a pinned signal-to-noise ratio.
//!
//! The chain, end to end:
//!
//! 1. **Analysis.** A synthetic length-`N` PCM block is windowed with the
//!    §4.3.1 Vorbis window and run through the §4.3.7 forward MDCT
//!    ([`oxideav_vorbis::apply_window_and_mdct_vec`]), producing a length
//!    `N/2` spectrum. The per-bin magnitude `|X[k]|` is the target floor
//!    envelope — the linear-domain envelope a real encoder would fit a
//!    floor to before coding the residual.
//! 2. **Floor fit (the new code under test).**
//!    [`oxideav_vorbis::plan_floor1_envelope`] inverts the §10.1 dB table
//!    and the §7.2.4 step-2 multiplier scale to fit the integer
//!    `[floor1_final_Y]` posts; [`oxideav_vorbis::plan_floor1_y`] unwraps
//!    those to the packet-domain `[floor1_Y]` vector.
//! 3. **Encode.** The `[floor1_Y]` vector is serialised into a §7.2.3
//!    floor-1 audio-packet body by `write_floor1_packet`.
//! 4. **Decode.** The crate's `Floor1Decoder` reads the body back and runs
//!    §7.2.4 curve synthesis, producing the reconstructed linear floor.
//! 5. **Assert.** At every post `x` the reconstructed curve equals the
//!    nearest representable approximation of the target envelope (an exact
//!    equality the fitter guarantees), and the whole curve tracks the
//!    target to a pinned SNR.
//!
//! No Ogg framing, no `docs/` fixtures: the test is fully synthetic and
//! self-contained, so it runs in standalone CI.

use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::floor1::{Floor1Decoder, FloorCurve, INVERSE_DB_TABLE};
use oxideav_vorbis::setup::{Floor1Class, Floor1Header};
use oxideav_vorbis::{
    apply_window_and_mdct_vec, invert_inverse_db, plan_floor1_envelope, plan_floor1_y,
    write_floor1_packet, Floor1Packet,
};

use oxideav_core::bits::BitReaderLsb;

/// §4.3.1 Vorbis window of length `n` (the analysis window the forward
/// MDCT consumes), `w[i] = sin(π/2 · sin²(π/n · (i + ½)))`. Both block
/// halves use the same window here (a centred long block with no
/// short-block transition), which is the simplest legal §4.3.1 window.
fn vorbis_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let inner = (std::f32::consts::PI / n as f32 * (i as f32 + 0.5)).sin();
            (std::f32::consts::FRAC_PI_2 * inner * inner).sin()
        })
        .collect()
}

/// A scalar (lookup-type-0) value book with `entries` length-7 codewords.
/// A balanced 128-entry book is Kraft-complete (`128 · 2⁻⁷ = 1`), so every
/// entry is reachable; the floor-1 packet writer codes each interior post
/// `Y` directly as the book entry whose index is `Y`, so `entries` must
/// exceed the largest post value (here `range = 128` for multiplier 2).
fn balanced_scalar_book(entries: u32, length: u8) -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
}

/// A floor-1 header with `interior_x.len()` interior posts in a single
/// partition of one class (subclasses 0, one subclass book). multiplier 2
/// → range 128 → 7-bit endpoint amplitudes and a 128-entry value book.
fn envelope_header(rangebits: u8, interior_x: Vec<u32>) -> Floor1Header {
    Floor1Header {
        partitions: 1,
        partition_class_list: vec![0],
        classes: vec![Floor1Class {
            dimensions: interior_x.len() as u8,
            subclasses: 0,
            masterbook: None,
            subclass_books: vec![Some(0)],
        }],
        multiplier: 2,
        rangebits,
        x_list: interior_x,
    }
}

/// Drive the analysis → fit → encode → decode chain for one target
/// envelope and return `(reconstructed_curve, full_x_list, multiplier)`.
fn roundtrip_floor(envelope: &[f32], header: &Floor1Header, n: usize) -> (Vec<f32>, Vec<u32>, u8) {
    let half_n = n / 2;
    assert_eq!(envelope.len(), half_n);

    // Step 2: fit the integer posts and unwrap to packet-domain Y.
    let posts = plan_floor1_envelope(envelope, header).expect("envelope fits into floor-1 posts");
    let floor1_y = plan_floor1_y(&posts, header).expect("posts unwrap to packet Y");

    // Step 3: encode the floor-1 audio-packet body.
    let book = balanced_scalar_book(128, 7);
    let packet = Floor1Packet {
        nonzero: true,
        floor1_y,
        partition_cvals: vec![0],
    };
    let body = write_floor1_packet(&packet, header, std::slice::from_ref(&book))
        .expect("floor-1 body serialises");

    // Step 4: decode it back to a linear floor curve.
    let dec =
        Floor1Decoder::new(header, std::slice::from_ref(&book)).expect("floor-1 decoder builds");
    let mut reader = BitReaderLsb::new(&body);
    let curve = match dec.decode(&mut reader, half_n) {
        FloorCurve::Curve(c) => c,
        FloorCurve::Unused => panic!("decoded an unused floor for a nonzero packet"),
    };
    assert_eq!(curve.len(), half_n);

    // The full x_list the decoder uses: implicit endpoints, then interior.
    let mut full_x = vec![0u32, 1u32 << header.rangebits];
    full_x.extend_from_slice(&header.x_list);
    (curve, full_x, header.multiplier)
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

/// Floor 1 is a *log-domain* envelope coder: it draws straight lines in the
/// §10.1 dB-index domain (the integer floor value before the table lookup),
/// not in linear amplitude. The honest whole-curve fidelity metric is
/// therefore the SNR in that log domain. Map each linear amplitude back to
/// its dB-index position via [`invert_inverse_db`] and measure the ratio of
/// target-index energy to per-bin index error.
fn log_domain_snr_db(target: &[f32], got: &[f32]) -> f32 {
    let t_idx: Vec<f32> = target
        .iter()
        .map(|&t| invert_inverse_db(t) as f32)
        .collect();
    let g_idx: Vec<f32> = got.iter().map(|&g| invert_inverse_db(g) as f32).collect();
    snr_db(&t_idx, &g_idx)
}

#[test]
fn forward_mdct_floor1_envelope_roundtrips_at_posts_exactly() {
    // A 256-sample analysis block → 128-bin spectrum.
    let n = 256usize;
    let half_n = n / 2;
    let window = vorbis_window(n);

    // Synthetic PCM: a couple of tones plus a slow tilt, so the spectrum
    // has a non-flat magnitude envelope worth fitting a floor to.
    let mut pcm: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32;
            0.6 * (2.0 * std::f32::consts::PI * 3.0 * t / n as f32).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 11.0 * t / n as f32).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 29.0 * t / n as f32).sin()
        })
        .collect();

    // Step 1: forward window + MDCT. scale = 1.0 (the analysis scale this
    // crate's IMDCT inverts to 1.0).
    let spectrum = apply_window_and_mdct_vec(&mut pcm, &window, 1.0)
        .expect("forward MDCT of the analysis block");
    assert_eq!(spectrum.len(), half_n);

    // A floor models the *spectral envelope*, not the fine line structure:
    // a real encoder fits it to a smoothed magnitude, not the raw spiky
    // |X[k]|. Smooth the MDCT magnitude with a windowed peak-hold + average
    // and add a small floor so every bin is a positive amplitude — this is
    // the target envelope the floor is fit to.
    let mag: Vec<f32> = spectrum.iter().map(|&s| s.abs()).collect();
    let radius = 4usize;
    let envelope: Vec<f32> = (0..half_n)
        .map(|k| {
            let lo = k.saturating_sub(radius);
            let hi = (k + radius + 1).min(half_n);
            let window = &mag[lo..hi];
            let peak = window.iter().copied().fold(0.0f32, f32::max);
            let mean = window.iter().sum::<f32>() / window.len() as f32;
            // Blend peak and mean for a smooth-but-tracking envelope, with a
            // small noise floor so quiet bins stay positive.
            0.5 * (peak + mean) + 1.0e-4
        })
        .collect();

    // rangebits 7 → upper endpoint x = 128 = half_n; interior posts spread
    // across the band. (rangebits must place 2^rangebits == half_n so the
    // upper endpoint samples the last bin, and every interior x < half_n.)
    let header = envelope_header(7, vec![8, 16, 32, 48, 64, 96, 120]);
    let (curve, full_x, multiplier) = roundtrip_floor(&envelope, &header, n);

    // Post-exact property: at each *rendered* post x (x < half_n — the
    // lower endpoint and every interior post), the reconstructed curve is
    // the nearest representable approximation of the target envelope:
    // INVERSE_DB_TABLE[ clamp(round(invert(env)/mult)) * mult ]. The upper
    // endpoint x == 2^rangebits == half_n only anchors the §7.2.4 step-2
    // line; the decoder renders the tail [last_flagged_post, n) flat at the
    // *last interior* post's value, so the upper endpoint is not itself a
    // rendered bin and is excluded from this exactness check.
    let range = 128i32; // multiplier 2
    let mut checked = 0;
    for &x in &full_x {
        let bin = x as usize;
        if bin >= half_n {
            continue; // upper endpoint — anchor only, not a rendered bin.
        }
        let target = envelope[bin];
        let idx = invert_inverse_db(target) as i32;
        let final_y = ((idx + multiplier as i32 / 2) / multiplier as i32).clamp(0, range - 1);
        let expected = INVERSE_DB_TABLE[(final_y * multiplier as i32) as usize];
        assert!(
            (curve[bin] - expected).abs() <= f32::EPSILON,
            "post x={x} bin={bin}: curve {} != nearest representable {expected}",
            curve[bin]
        );
        checked += 1;
    }
    // 1 lower endpoint + 7 interior posts = 8 rendered posts checked.
    assert_eq!(
        checked, 8,
        "expected 8 rendered posts to be exactness-checked"
    );

    // Whole-curve fidelity, measured in the floor's native log (dB-index)
    // domain — the domain floor 1 draws straight lines in. The smoothed
    // envelope is well-approximated by the piecewise-linear floor there, so
    // the log-domain SNR clears a healthy pinned floor. (A linear-domain SNR
    // would be dominated by the few highest-magnitude bins and is not the
    // metric a log-envelope coder optimizes.)
    let snr = log_domain_snr_db(&envelope, &curve);
    assert!(
        snr >= 20.0,
        "floor-1 envelope log-domain round-trip SNR {snr} dB below pinned 20 dB floor"
    );
}

#[test]
fn flat_envelope_reconstructs_flat_curve() {
    // A constant target envelope fits to one post value everywhere, so the
    // decoded curve is exactly flat at the nearest dB-ladder value — an
    // exact (infinite-SNR) round-trip.
    let n = 128usize;
    let half_n = n / 2;
    let target = INVERSE_DB_TABLE[100];
    let envelope = vec![target; half_n];

    let header = envelope_header(6, vec![8, 16, 32, 48]);
    let (curve, _full_x, _mult) = roundtrip_floor(&envelope, &header, n);

    // multiplier 2: index 100 → post round(100/2)=50 → 50*2=100. Flat at
    // INVERSE_DB_TABLE[100] across the whole curve.
    let expected = INVERSE_DB_TABLE[100];
    for (bin, &c) in curve.iter().enumerate() {
        assert!(
            (c - expected).abs() <= f32::EPSILON,
            "bin {bin}: {c} != flat {expected}"
        );
    }
    assert!(snr_db(&envelope, &curve).is_infinite());
}

#[test]
fn ascending_envelope_yields_monotone_post_curve() {
    // A strictly increasing envelope (in dB-ladder order) makes the fitted
    // posts non-decreasing, so the reconstructed curve is non-decreasing at
    // the post locations — the floor tracks the rising envelope.
    let n = 128usize;
    let half_n = n / 2;
    // Envelope rising smoothly from low to high ladder values.
    let envelope: Vec<f32> = (0..half_n)
        .map(|i| INVERSE_DB_TABLE[(i * 4).min(255)])
        .collect();

    let header = envelope_header(6, vec![8, 16, 32, 48]);
    let (curve, full_x, _mult) = roundtrip_floor(&envelope, &header, n);

    // Sort posts by x and confirm the curve value at each is non-decreasing
    // with x (the envelope is monotone, the fit preserves it).
    let mut xs: Vec<u32> = full_x.clone();
    xs.sort_unstable();
    let mut prev = -1.0f32;
    for &x in &xs {
        let bin = (x as usize).min(half_n - 1);
        assert!(
            curve[bin] >= prev - f32::EPSILON,
            "curve at x={x} ({}) dropped below previous {prev}",
            curve[bin]
        );
        prev = curve[bin];
    }
}

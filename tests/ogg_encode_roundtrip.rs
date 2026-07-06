//! Whole-stream `encode(pcm) → .ogg → decode → pcm` round-trip.
//!
//! `encode_pcm_to_ogg` composes the full encode stack (framing
//! splitter → forward MDCT → psy model → floor-1 design/fit →
//! NMR-weighted RD residue → §4.3 packet writer → §A.2 Ogg
//! encapsulation) into one call; this suite drives it end to end and
//! decodes the result back with the crate's own §4.3 streaming
//! decoder via `decode_ogg_to_pcm`, asserting:
//!
//! * **structure** — every §A.2 page rule on the produced physical
//!   stream (58-byte BOS first page, header pages at granule 0, audio
//!   fresh page, EOS final page carrying the exact input length);
//! * **fidelity** — the decoded PCM matches the input in the time
//!   domain to a pinned SNR at `q = 0.7`, mono and stereo, including
//!   a non-block-aligned length (the §A.2 end-trim path);
//! * **rate behaviour** — a higher quality spends more bytes and
//!   never lowers the measured SNR (small slack);
//! * **degenerate shapes** — sub-block streams and silence encode and
//!   decode cleanly;
//! * **guards** — the shape/configuration rejections fire.
//!
//! Fully synthetic — no fixtures required.

use oxideav_ogg::page::Page;
use oxideav_vorbis::{
    decode_ogg_to_pcm, encode_pcm_to_ogg, ogg_packets, OggFileError, StreamEncoderConfig,
};

const RATE: u32 = 44_100;

/// Parse every page of a physical stream (CRC-verified by the
/// `oxideav-ogg` page parser), keeping each page's on-wire byte length.
fn parse_pages(data: &[u8]) -> Vec<(Page, usize)> {
    let mut pages = Vec::new();
    let mut off = 0usize;
    while off < data.len() {
        let (page, used) = Page::parse(&data[off..]).expect("page parses (CRC verifies)");
        pages.push((page, used));
        off += used;
    }
    pages
}

/// A deterministic tones + noise-floor test signal.
fn test_signal(samples: usize, seed: u32) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            let mut v = 0.42 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.21 * (2.0 * std::f32::consts::PI * 1370.0 * t).sin()
                + 0.10 * (2.0 * std::f32::consts::PI * 31.0 * t).cos();
            let h = (i as u32 + seed).wrapping_mul(2_654_435_761) >> 8;
            v += ((h & 0xffff) as f32 / 32768.0 - 1.0) * 0.003;
            v
        })
        .collect()
}

fn snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
    assert_eq!(reference.len(), decoded.len());
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&r, &d) in reference.iter().zip(decoded) {
        sig += f64::from(r) * f64::from(r);
        let e = f64::from(r) - f64::from(d);
        err += e * e;
    }
    if err == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

#[test]
fn mono_stream_roundtrips_with_a2_structure_and_pinned_snr() {
    // Deliberately not a multiple of n/2 = 512: exercises the §A.2
    // end-trim.
    let samples = 22_000;
    let pcm = vec![test_signal(samples, 1)];
    let config = StreamEncoderConfig::new(RATE, 1);
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");

    // ---- §A.2 structure ----
    let pages = parse_pages(&ogg);
    assert_eq!(pages[0].1, 58, "id header alone on page 0");
    assert!(pages[0].0.is_first());
    assert_eq!(pages[0].0.granule_position, 0);
    assert_eq!(pages[1].0.granule_position, 0, "header pages at granule 0");
    assert!(pages.last().unwrap().0.is_last());
    assert_eq!(
        pages.last().unwrap().0.granule_position,
        samples as i64,
        "final granule is the exact input length"
    );
    let packets = ogg_packets(&ogg).expect("packets assemble");
    assert!(packets.len() > 3);

    // ---- decode + fidelity ----
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.channels, 1);
    assert_eq!(decoded.sample_rate, RATE);
    assert_eq!(decoded.pcm[0].len(), samples, "end-trim to input length");
    let snr = snr_db(&pcm[0], &decoded.pcm[0]);
    eprintln!("mono q0.7: {} bytes, SNR {snr:.2} dB", ogg.len());
    assert!(snr >= 25.0, "mono SNR {snr:.2} dB below the 25 dB pin");
}

/// A second, spectrally distinct signal for the right channel.
fn test_signal_alt(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            0.38 * (2.0 * std::f32::consts::PI * 620.0 * t).sin()
                + 0.18 * (2.0 * std::f32::consts::PI * 2210.0 * t).cos()
        })
        .collect()
}

#[test]
fn stereo_stream_roundtrips_per_channel() {
    let samples = 15_000;
    let pcm = vec![test_signal(samples, 2), test_signal_alt(samples)];
    let config = StreamEncoderConfig::new(RATE, 2);
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.channels, 2);
    for (c, input) in pcm.iter().enumerate() {
        assert_eq!(decoded.pcm[c].len(), samples);
        let snr = snr_db(input, &decoded.pcm[c]);
        eprintln!("stereo ch{c}: SNR {snr:.2} dB");
        // The encoder optimises noise-to-mask, not waveform SNR; 20 dB
        // is the regression pin, not a transparency claim.
        assert!(snr >= 20.0, "channel {c} SNR {snr:.2} dB below 20 dB");
    }
    // The channels are genuinely different signals.
    assert!(snr_db(&pcm[0], &pcm[1]) < 10.0);
}

#[test]
fn quality_knob_trades_rate_for_fidelity_on_the_ogg_stream() {
    let samples = 8_192;
    let pcm = vec![test_signal(samples, 3)];
    let mut low = StreamEncoderConfig::new(RATE, 1);
    low.quality = 0.2;
    let mut high = StreamEncoderConfig::new(RATE, 1);
    high.quality = 0.9;

    let ogg_low = encode_pcm_to_ogg(&pcm, &low).expect("low-q encodes");
    let ogg_high = encode_pcm_to_ogg(&pcm, &high).expect("high-q encodes");
    let snr_low = snr_db(&pcm[0], &decode_ogg_to_pcm(&ogg_low).unwrap().pcm[0]);
    let snr_high = snr_db(&pcm[0], &decode_ogg_to_pcm(&ogg_high).unwrap().pcm[0]);
    eprintln!(
        "q0.2: {} B / {snr_low:.2} dB;  q0.9: {} B / {snr_high:.2} dB",
        ogg_low.len(),
        ogg_high.len()
    );
    assert!(
        ogg_high.len() > ogg_low.len(),
        "higher quality must spend more bytes: {} vs {}",
        ogg_high.len(),
        ogg_low.len()
    );
    assert!(
        snr_high >= snr_low - 0.5,
        "higher quality must not lose SNR: {snr_high:.2} vs {snr_low:.2}"
    );
    assert!(
        snr_high >= snr_low + 3.0,
        "the knob must buy a real fidelity step: {snr_high:.2} vs {snr_low:.2}"
    );
}

#[test]
fn sub_block_and_silent_streams_encode_and_decode() {
    // Shorter than n/2: one finished half-block covers it.
    let pcm = vec![test_signal(300, 4)];
    let config = StreamEncoderConfig::new(RATE, 1);
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("tiny stream encodes");
    let decoded = decode_ogg_to_pcm(&ogg).expect("tiny stream decodes");
    assert_eq!(decoded.pcm[0].len(), 300);

    // Pure silence: the psy floor rides the threshold-in-quiet and the
    // residue quantises to zero; the decode must be (near-)silence.
    let silent = vec![vec![0.0f32; 5000]];
    let ogg = encode_pcm_to_ogg(&silent, &config).expect("silence encodes");
    let decoded = decode_ogg_to_pcm(&ogg).expect("silence decodes");
    assert_eq!(decoded.pcm[0].len(), 5000);
    let peak = decoded.pcm[0].iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(
        peak < 1.0e-3,
        "silence must decode near-silent, peak {peak}"
    );
}

#[test]
fn a_smaller_blocksize_also_roundtrips() {
    let samples = 4_000;
    let pcm = vec![test_signal(samples, 5)];
    let mut config = StreamEncoderConfig::new(RATE, 1);
    config.blocksize = 256;
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes at n=256");
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.pcm[0].len(), samples);
    let snr = snr_db(&pcm[0], &decoded.pcm[0]);
    eprintln!("n=256: SNR {snr:.2} dB");
    assert!(snr >= 20.0, "n=256 SNR {snr:.2} dB below 20 dB");
}

#[test]
fn codebook_training_cuts_the_stream_at_equal_fidelity() {
    // The closed-loop ladder trainer (now covering the encoder's 1-D
    // lattice value books) retrains the generic seed ladders on the
    // stream's own residue targets; the trained stream must be
    // smaller at no meaningful fidelity cost, and still §4.2.4
    // carriage-legal (decode_ogg_to_pcm re-parses the trained setup
    // header from the produced bytes).
    let samples = 16_384;
    let pcm = vec![test_signal(samples, 11)];
    let mut untrained = StreamEncoderConfig::new(RATE, 1);
    untrained.training_iterations = 0;
    let mut trained = StreamEncoderConfig::new(RATE, 1);
    trained.training_iterations = 6;

    let ogg_untrained = encode_pcm_to_ogg(&pcm, &untrained).expect("untrained encodes");
    let ogg_trained = encode_pcm_to_ogg(&pcm, &trained).expect("trained encodes");
    let snr_untrained = snr_db(&pcm[0], &decode_ogg_to_pcm(&ogg_untrained).unwrap().pcm[0]);
    let snr_trained = snr_db(&pcm[0], &decode_ogg_to_pcm(&ogg_trained).unwrap().pcm[0]);
    eprintln!(
        "training: {} B / {snr_untrained:.2} dB → {} B / {snr_trained:.2} dB",
        ogg_untrained.len(),
        ogg_trained.len()
    );
    assert!(
        ogg_trained.len() < ogg_untrained.len(),
        "trained stream {} B must undercut the seed stream {} B",
        ogg_trained.len(),
        ogg_untrained.len()
    );
    // The Lagrangian may trade a little distortion for the rate win,
    // never catastrophically.
    assert!(
        snr_trained >= snr_untrained - 3.0,
        "trained SNR {snr_trained:.2} dB collapsed vs {snr_untrained:.2} dB"
    );
}

#[test]
fn shape_guards_fire() {
    let config = StreamEncoderConfig::new(RATE, 2);
    // Row count != channels.
    assert_eq!(
        encode_pcm_to_ogg(&[vec![0.0; 100]], &config),
        Err(OggFileError::BadChannelCount {
            channels: 2,
            rows: 1
        })
    );
    // Unequal rows.
    assert_eq!(
        encode_pcm_to_ogg(&[vec![0.0; 100], vec![0.0; 99]], &config),
        Err(OggFileError::BadPcmShape)
    );
    // Empty PCM.
    assert_eq!(
        encode_pcm_to_ogg(&[vec![], vec![]], &config),
        Err(OggFileError::BadPcmShape)
    );
    // Bad blocksize.
    let mut bad = StreamEncoderConfig::new(RATE, 1);
    bad.blocksize = 100;
    assert_eq!(
        encode_pcm_to_ogg(&[vec![0.0; 100]], &bad),
        Err(OggFileError::BadBlocksize(100))
    );
    // Zero rate.
    let mut bad = StreamEncoderConfig::new(0, 1);
    bad.sample_rate = 0;
    assert_eq!(
        encode_pcm_to_ogg(&[vec![0.0; 100]], &bad),
        Err(OggFileError::ZeroSampleRate)
    );
    // Bad quality.
    let mut bad = StreamEncoderConfig::new(RATE, 1);
    bad.quality = 2.0;
    assert!(matches!(
        encode_pcm_to_ogg(&[vec![0.0; 100]], &bad),
        Err(OggFileError::Quality(_))
    ));
}

//! The r456 **low-bitrate mode** of the whole-stream encoder: below
//! the quality knee (`quality::LOW_BITRATE_KNEE`) the lever laws
//! steepen — the residue `lambda` climbs 2.5 further decades, the
//! noise-like maskers' thresholds rise while the tonal maskers' fall,
//! and the coded band is limited toward 5 kHz — so `q = 0` reaches the
//! reference encoder's lowest operating region instead of stopping at
//! ~130 kbps.
//!
//! Pins on a stereo tones + hiss corpus (harmonic beds at −10 dBFS
//! under −45 dBFS hiss — the shape the old knob floor spent 240 kbps
//! on): the `q = 0` audio rate, the §8.6.1 `residue_end` cap the
//! bandwidth limit produces, rate monotone in `q` across the knee,
//! every stream end-trim exact through the crate's own decoder, and
//! the ABR entry meeting a 48 kbps budget from below.
//!
//! Fully synthetic: no `docs/` fixtures.

use oxideav_vorbis::quality::{EncoderTuning, LOW_BITRATE_KNEE};
use oxideav_vorbis::{
    decode_ogg_to_pcm, encode_pcm_to_ogg, ogg_packets, parse_setup_header, StreamEncoderConfig,
};

const RATE: u32 = 44_100;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
}

/// Two seconds of stereo tones + hiss: two harmonic beds (220 Hz ×
/// 10, 330 Hz × 8) mixed differently into the channels under a slow
/// amplitude modulation, plus independent hiss at −45 dBFS.
fn tones_and_hiss() -> Vec<Vec<f32>> {
    let n = 2 * RATE as usize;
    let mut s = 3u64;
    let mut l = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / RATE as f32;
        let am = 0.7 + 0.3 * (2.0 * std::f32::consts::PI * 0.5 * t).sin();
        let a: f32 = (1..=10)
            .map(|k| (2.0 * std::f32::consts::PI * 220.0 * k as f32 * t).sin() / k as f32)
            .sum();
        let b: f32 = (1..=8)
            .map(|k| {
                (2.0 * std::f32::consts::PI * 330.0 * k as f32 * t + 0.3 * k as f32).sin()
                    / (k as f32).powf(1.3)
            })
            .sum();
        l.push(am * (0.2 * a + 0.07 * b) + 0.004 * lcg(&mut s));
        r.push(am * (0.14 * a + 0.12 * b) + 0.004 * lcg(&mut s));
    }
    vec![l, r]
}

fn audio_kbps(ogg: &[u8], seconds: f64) -> f64 {
    let packets = ogg_packets(ogg).expect("de-frames");
    let bytes: usize = packets[3..].iter().map(Vec::len).sum();
    bytes as f64 * 8.0 / seconds / 1000.0
}

fn snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&r, &d) in reference.iter().zip(decoded) {
        sig += f64::from(r) * f64::from(r);
        err += (f64::from(r) - f64::from(d)).powi(2);
    }
    10.0 * (sig / err.max(1e-30)).log10()
}

fn encode_at(pcm: &[Vec<f32>], q: f32) -> (Vec<u8>, f64, f64) {
    let seconds = pcm[0].len() as f64 / f64::from(RATE);
    let mut config = StreamEncoderConfig::new(RATE, 2);
    config.quality = q;
    let ogg = encode_pcm_to_ogg(pcm, &config).expect("encodes");
    let decoded = decode_ogg_to_pcm(&ogg).expect("own decode");
    let mut snr = f64::INFINITY;
    for (c, row) in pcm.iter().enumerate() {
        assert_eq!(decoded.pcm[c].len(), row.len(), "q={q}: end-trim exact");
        snr = snr.min(snr_db(row, &decoded.pcm[c]));
    }
    let kbps = audio_kbps(&ogg, seconds);
    (ogg, kbps, snr)
}

#[test]
fn the_knob_floor_reaches_the_low_rate_region_monotonically() {
    let pcm = tones_and_hiss();
    let qs = [0.0f32, 0.1, LOW_BITRATE_KNEE, 0.3];
    let points: Vec<(f64, f64)> = qs
        .iter()
        .map(|&q| {
            let (_, kbps, snr) = encode_at(&pcm, q);
            eprintln!("q={q:.2}: {kbps:.1} kbps audio, min-channel SNR {snr:.2} dB");
            (kbps, snr)
        })
        .collect();
    // The knob floor: the old q = 0 spent ~240 kbps on this shape.
    assert!(
        points[0].0 <= 60.0,
        "q = 0 must reach the low-rate region: {:.1} kbps",
        points[0].0
    );
    // Still a playable encode of the harmonic beds, not silence.
    assert!(
        points[0].1 >= 4.0,
        "q = 0 must keep the partials: {:.2} dB",
        points[0].1
    );
    for w in points.windows(2) {
        assert!(
            w[0].0 <= w[1].0,
            "audio rate must not fall as quality rises: {points:?}"
        );
        assert!(
            w[1].1 >= w[0].1 - 0.5,
            "SNR must not fall as quality rises: {points:?}"
        );
    }
}

#[test]
fn the_low_rate_mode_limits_the_coded_band() {
    let pcm = tones_and_hiss();
    let (ogg, _, _) = encode_at(&pcm, 0.0);
    let packets = ogg_packets(&ogg).expect("de-frames");
    let setup = parse_setup_header(&packets[2], 2).expect("setup parses");
    let tuning = EncoderTuning::from_quality(0.0).unwrap();
    let bw = tuning.coded_bandwidth_hz.expect("q = 0 limits the band");
    // Long residue (1024 bins over 22.05 kHz): the cutoff bin rounded
    // up to a whole 32-bin partition; short residue (128 bins) to a
    // 16-bin partition.
    let expect = |half: usize, ps: usize| -> u32 {
        let bins = (bw / (RATE as f32 / 2.0) * half as f32).ceil() as usize;
        (bins.div_ceil(ps) * ps).min(half) as u32
    };
    assert_eq!(setup.residues[0].residue_end, expect(128, 16));
    assert_eq!(setup.residues[1].residue_end, expect(1024, 32));
    assert!(setup.residues[1].residue_end < 1024, "the band is limited");
    // At the knee and above the whole spectrum is coded.
    let (ogg, _, _) = encode_at(&pcm, LOW_BITRATE_KNEE);
    let packets = ogg_packets(&ogg).expect("de-frames");
    let setup = parse_setup_header(&packets[2], 2).expect("setup parses");
    assert_eq!(setup.residues[1].residue_end, 1024);
}

#[test]
fn abr_meets_a_48_kbps_budget() {
    let pcm = tones_and_hiss();
    let seconds = pcm[0].len() as f64 / f64::from(RATE);
    let mut config = StreamEncoderConfig::new(RATE, 2);
    config.target_bitrate = Some(48_000);
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
    let got = audio_kbps(&ogg, seconds);
    eprintln!("ABR 48 kbps -> {got:.1} kbps audio");
    assert!(got <= 48.0, "budget missed: {got:.1} kbps");
    assert!(got >= 48.0 * 0.4, "budget underspent: {got:.1} kbps");
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.pcm[0].len(), pcm[0].len());
}

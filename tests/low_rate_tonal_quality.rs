//! The r483 low-bitrate structure on tonal material: decaying
//! inharmonic notes panned across a stereo pair — the shape whose
//! low-rate SNR the old floor fit and coupling election lost.
//!
//! Pins, at `q = 0` (the cheapest operating point):
//!
//! * the doubled 4096 / 512 geometry and its `low_rate_geometry =
//!   false` opt-out;
//! * whole-stream own-decode SNR well clear of the r482 figure (the
//!   covering floor no longer sits 20–30 dB above the loudest
//!   partials);
//! * the **quiet channel** of a hard-panned pair keeps its fidelity
//!   (the coupling election charges it the shared-floor noise);
//! * end-trim exact decode.
//!
//! Fully synthetic: no `docs/` fixtures.

use oxideav_vorbis::{
    StreamEncoderConfig, decode_ogg_to_pcm, encode_pcm_to_ogg, ogg_packets,
    parse_identification_header,
};

const RATE: u32 = 44_100;
const SECONDS: f32 = 2.0;

/// Four decaying 12-partial notes, every 0.5 s, panned 80/20 then
/// 20/80 alternately.
fn panned_notes() -> Vec<Vec<f32>> {
    let n = (SECONDS * RATE as f32) as usize;
    let mut l = vec![0.0f32; n];
    let mut r = vec![0.0f32; n];
    for (k, &f0) in [261.6f32, 329.6, 392.0, 293.7].iter().enumerate() {
        let start = k * n / 4;
        let pan = if k % 2 == 0 { 0.2 } else { 0.8 };
        for i in start..n {
            let t = (i - start) as f32 / RATE as f32;
            let mut v = 0.0;
            for h in 1..=12 {
                let hf = h as f32;
                let f = f0 * hf * (1.0 + 0.0004 * hf * hf).sqrt();
                v += (2.0 * std::f32::consts::PI * f * t).sin() * (-t * (1.5 + 0.6 * hf)).exp()
                    / hf.powf(1.1);
            }
            v *= 0.18 * (t / 0.004).min(1.0);
            l[i] += v * (1.0 - pan);
            r[i] += v * pan;
        }
    }
    vec![l, r]
}

fn snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (&a, &b) in reference.iter().zip(decoded) {
        sig += f64::from(a) * f64::from(a);
        err += (f64::from(a) - f64::from(b)).powi(2);
    }
    10.0 * (sig / err.max(1e-30)).log10()
}

/// SNR of `channel` over the span where it is the quiet side (the
/// first note, panned 80/20 toward channel 0).
fn quiet_span_snr(pcm: &[Vec<f32>], decoded: &[Vec<f32>], channel: usize) -> f64 {
    let quarter = pcm[0].len() / 4;
    snr_db(&pcm[channel][..quarter], &decoded[channel][..quarter])
}

#[test]
fn low_rate_tonal_quality_and_geometry() {
    let pcm = panned_notes();
    let mut config = StreamEncoderConfig::new(RATE, 2);
    config.quality = 0.0;
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
    let packets = ogg_packets(&ogg).expect("de-frames");
    let id = parse_identification_header(&packets[0]).expect("id parses");
    assert_eq!((id.blocksize_0, id.blocksize_1), (512, 4096));
    let audio_bits: usize = packets[3..].iter().map(Vec::len).sum::<usize>() * 8;
    let kbps = audio_bits as f64 / f64::from(SECONDS) / 1000.0;
    let decoded = decode_ogg_to_pcm(&ogg).expect("own decode");
    for (row, out) in pcm.iter().zip(&decoded.pcm) {
        assert_eq!(out.len(), row.len(), "end-trim exact");
    }
    let whole: Vec<f32> = pcm.concat();
    let whole_out: Vec<f32> = decoded.pcm.concat();
    let snr = snr_db(&whole, &whole_out);
    let quiet = quiet_span_snr(&pcm, &decoded.pcm, 1);
    eprintln!("q = 0: {kbps:.1} kbps audio, SNR {snr:.2} dB, quiet channel {quiet:.2} dB");
    // r483 measurement: 23.4 kbps, 14.85 dB whole-stream, 16.85 dB
    // on the quiet channel (r482: 15.8 kbps, 1.45 dB, −5.47 dB — the
    // quiet side of the first note decoded louder in error than in
    // signal).
    assert!(
        kbps <= 32.0,
        "q = 0 stays in the low-rate region: {kbps:.1} kbps"
    );
    assert!(snr >= 12.0, "whole-stream SNR {snr:.2} dB below 12 dB");
    assert!(quiet >= 12.0, "quiet-channel SNR {quiet:.2} dB below 12 dB");

    // The opt-out codes on the configured 2048 / 256 pair.
    config.low_rate_geometry = false;
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
    let packets = ogg_packets(&ogg).expect("de-frames");
    let id = parse_identification_header(&packets[0]).expect("id parses");
    assert_eq!((id.blocksize_0, id.blocksize_1), (256, 2048));
}

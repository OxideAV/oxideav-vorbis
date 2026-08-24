//! The measured per-channel adaptive-margin balance pass: at the top
//! of the quality knob the encoder own-decodes its first pass and
//! grants extra masking margin only to channels whose measured SNR
//! trails the best channel — waveform coding where it measurably pays
//! (`EncoderTuning::adaptive_margin_headroom_db` +
//! `encode_pcm_to_packets`'s retry).
//!
//! The corpus is the staged decorrelated stereo fixture: its channels
//! carry identical RMS (−21 dBFS each) but ch1's content codes ~12 dB
//! worse under the shared per-stream ladders — precisely the imbalance
//! no cheap psy statistic identifies (over-masked-energy fraction and
//! energy-weighted tonality were both measured near-identical across
//! the pair), which is why the pass measures instead of guessing.
//!
//! Requires the umbrella `docs/` submodule (fixture corpus); skips on
//! the standalone per-crate CI clone (data availability, not a
//! disabled test).

use oxideav_vorbis::{decode_ogg_to_pcm, encode_pcm_to_ogg, ogg_packets, StreamEncoderConfig};

fn fixtures_root() -> String {
    format!(
        "{}/../../docs/audio/vorbis/fixtures",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn fixtures_available() -> bool {
    std::path::Path::new(&fixtures_root()).is_dir()
}

/// Read a fixture's `expected.wav`: sample rate + per-channel f32 rows.
fn wav_pcm(path: &str) -> (u32, Vec<Vec<f32>>) {
    let data = std::fs::read(path).expect("expected.wav present");
    let mut pos = 12; // RIFF + size + WAVE
    let mut channels = 0u16;
    let mut rate = 0u32;
    let mut rows: Vec<Vec<f32>> = Vec::new();
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let sz = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
            as usize;
        let start = pos + 8;
        if id == b"fmt " {
            channels = u16::from_le_bytes([data[start + 2], data[start + 3]]);
            rate = u32::from_le_bytes([
                data[start + 4],
                data[start + 5],
                data[start + 6],
                data[start + 7],
            ]);
        } else if id == b"data" {
            let end = (start + sz).min(data.len());
            let ch = channels as usize;
            rows = vec![Vec::new(); ch];
            for (i, s) in data[start..end].chunks_exact(2).enumerate() {
                rows[i % ch].push(f32::from(i16::from_le_bytes([s[0], s[1]])) / 32768.0);
            }
        }
        pos = start + sz + (sz & 1);
    }
    assert!(rate > 0 && !rows.is_empty(), "WAV parse: {path}");
    (rate, rows)
}

fn snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
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

fn measure(pcm: &[Vec<f32>], rate: u32, q: f32) -> (usize, Vec<f64>) {
    let mut config = StreamEncoderConfig::new(rate, pcm.len() as u8);
    config.quality = q;
    let ogg = encode_pcm_to_ogg(pcm, &config).expect("encodes");
    let decoded = decode_ogg_to_pcm(&ogg).expect("own decode");
    let packets = ogg_packets(&ogg).expect("de-frames");
    let audio: usize = packets[3..].iter().map(|p| p.len()).sum();
    let snrs: Vec<f64> = pcm
        .iter()
        .enumerate()
        .map(|(c, input)| {
            assert_eq!(decoded.pcm[c].len(), input.len(), "end-trim exact");
            snr_db(input, &decoded.pcm[c])
        })
        .collect();
    (audio, snrs)
}

#[test]
fn adaptive_margin_balances_the_trailing_channel_at_the_top() {
    if !fixtures_available() {
        eprintln!("fixtures not staged; skipping");
        return;
    }
    let (rate, pcm) = wav_pcm(&format!(
        "{}/stereo-44100-q5-typical/expected.wav",
        fixtures_root()
    ));

    // Below the cap knee the balance pass is dormant: the default-band
    // encode carries no adaptive headroom, so rate and fidelity match
    // the pre-lever encoder (r451 measurement: 10353 audio B,
    // [47.8, 29.8] dB).
    let (mid_audio, mid_snrs) = measure(&pcm, rate, 0.7);
    eprintln!("q=0.7: audio {mid_audio} B, per-ch SNR {mid_snrs:.2?} dB");
    assert!(
        mid_audio <= 11_000,
        "q=0.7 audio bytes {mid_audio} above the 11 kB regression bound"
    );
    assert!(
        mid_snrs[0] >= 45.0 && mid_snrs[1] >= 28.0,
        "q=0.7 per-channel SNR regressed: {mid_snrs:.2?}"
    );

    // At the top of the knob the measured retry must close the
    // channel imbalance. r451 measurement: the ungated q=1 encode read
    // [52.3, 40.2] dB (12.1 dB gap) at 14.4 kB audio; the balance pass
    // lands [55.2, 58.2] dB (3.0 dB gap) at 23.1 kB.
    let (top_audio, top_snrs) = measure(&pcm, rate, 1.0);
    eprintln!("q=1: audio {top_audio} B, per-ch SNR {top_snrs:.2?} dB");
    let top_min = top_snrs.iter().copied().fold(f64::INFINITY, f64::min);
    let top_max = top_snrs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        top_min >= 50.0,
        "q=1 min-channel SNR {top_min:.2} dB below 50 dB (was 40.2 dB before the balance pass)"
    );
    assert!(
        top_max - top_min <= 6.0,
        "q=1 channel imbalance {:.2} dB above 6 dB: {top_snrs:.2?}",
        top_max - top_min
    );
    assert!(
        top_audio <= 26_000,
        "q=1 audio bytes {top_audio} above the 26 kB regression bound"
    );

    // The top of the knob buys decisive min-channel headroom over the
    // mid-knob (measured 25.4 dB).
    let mid_min = mid_snrs.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        top_min >= mid_min + 15.0,
        "q=1 min-channel SNR {top_min:.2} dB must clear q=0.7's {mid_min:.2} dB by >= 15 dB"
    );
}

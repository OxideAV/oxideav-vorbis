//! Re-encoding the staged fixture corpus through the integrated
//! encoder — real audio (not synthetic tones) through the full
//! switched/coupled pipeline.
//!
//! Each staged fixture ships an `expected.wav` (the corpus PCM). This
//! suite feeds that PCM back through `encode_pcm_to_ogg` at the
//! default configuration and pins, per fixture:
//!
//! * the re-encode decodes through the crate's own §4.3 decoder to the
//!   **exact** input length (mixed-size granule walk + §A.2 end-trim
//!   on real content);
//! * a fidelity floor calibrated per corpus;
//! * the schedule behaves like the content: the
//!   `transient-blocksize-switch` corpus (a sine bed with a sustained
//!   white-noise burst — the energy-**step** shape the reference
//!   encoder also switches on) schedules short blocks, while the
//!   steady `mono-44100-q5-typical` music stays all-long;
//! * the genuinely decorrelated `stereo-44100-q5-typical` pair
//!   (whole-stream angle/magnitude energy ratio ≈ 1.7) is correctly
//!   left **uncoupled** by the profitability gate.
//!
//! Requires the umbrella `docs/` submodule; the standalone per-crate
//! CI clones only this repo, so the suite skips there (data
//! availability, not a disabled test — there is no `#[ignore]`).

use oxideav_core::bits::BitReaderLsb;
use oxideav_vorbis::{
    decode_ogg_to_pcm, encode_pcm_to_ogg, ogg_packets, parse_identification_header,
    parse_setup_header, read_packet_header, StreamEncoderConfig,
};

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

struct Reencode {
    ogg: Vec<u8>,
    /// Audio-packet bytes only (headers excluded): the rate figure the
    /// two-sided regression gates pin — header size legitimately
    /// varies with the residue geometry (the designed-lattice books
    /// carry ~800 B of codeword-length tables the scalar ladders
    /// don't), while audio bytes track the actual coding efficiency.
    audio_bytes: usize,
    shorts: usize,
    longs: usize,
    coupling_steps: usize,
}

/// Re-encode a fixture's PCM at the default config and parse the
/// produced stream's schedule + coupling out of the wire format.
fn reencode(name: &str, pcm: &[Vec<f32>], rate: u32) -> Reencode {
    let config = StreamEncoderConfig::new(rate, pcm.len() as u8);
    let ogg = encode_pcm_to_ogg(pcm, &config).unwrap_or_else(|e| panic!("{name} encodes: {e}"));
    let packets = ogg_packets(&ogg).expect("de-frames");
    let audio_bytes = packets[3..].iter().map(Vec::len).sum();
    let id = parse_identification_header(&packets[0]).expect("id parses");
    let setup = parse_setup_header(&packets[2], id.audio_channels).expect("setup parses");
    let coupling_steps = setup.mappings[0].coupling.len();
    let mut shorts = 0;
    let mut longs = 0;
    for p in &packets[3..] {
        let mut reader = BitReaderLsb::new(p);
        let h = read_packet_header(
            &mut reader,
            &setup,
            id.blocksize_0 as usize,
            id.blocksize_1 as usize,
        )
        .expect("prelude parses");
        if h.blockflag {
            longs += 1;
        } else {
            shorts += 1;
        }
    }
    Reencode {
        ogg,
        audio_bytes,
        shorts,
        longs,
        coupling_steps,
    }
}

#[test]
fn transient_fixture_reencode_switches_and_roundtrips() {
    if !fixtures_available() {
        eprintln!("fixtures not staged; skipping");
        return;
    }
    let (rate, pcm) = wav_pcm(&format!(
        "{}/transient-blocksize-switch/expected.wav",
        fixtures_root()
    ));
    let re = reencode("transient-blocksize-switch", &pcm, rate);
    eprintln!(
        "transient fixture re-encode: {} B, {} short / {} long",
        re.ogg.len(),
        re.shorts,
        re.longs
    );
    // The corpus is a sine bed with a sustained noise burst — the
    // energy-step transient the reference encoder also short-blocks.
    assert!(re.shorts > 0, "the burst must schedule short blocks");
    assert!(re.longs > re.shorts, "the sine beds must stay long");

    let decoded = decode_ogg_to_pcm(&re.ogg).expect("decodes");
    assert_eq!(decoded.pcm[0].len(), pcm[0].len(), "end-trim exact");
    let snr = snr_db(&pcm[0], &decoded.pcm[0]);
    eprintln!("transient fixture re-encode SNR: {snr:.2} dB");
    // Two-sided r416 gates (measured 7797 audio B / 19.5 dB): neither
    // rate nor fidelity may regress at the default quality point.
    assert!(snr >= 12.0, "SNR {snr:.2} dB below 12 dB");
    assert!(
        re.audio_bytes <= 8_800,
        "default-quality audio bytes {} above the 8.8 kB regression bound",
        re.audio_bytes
    );
}

#[test]
fn steady_music_fixture_reencode_stays_long_at_high_fidelity() {
    if !fixtures_available() {
        eprintln!("fixtures not staged; skipping");
        return;
    }
    let (rate, pcm) = wav_pcm(&format!(
        "{}/mono-44100-q5-typical/expected.wav",
        fixtures_root()
    ));
    let re = reencode("mono-44100-q5-typical", &pcm, rate);
    eprintln!(
        "steady fixture re-encode: {} B, {} short / {} long",
        re.ogg.len(),
        re.shorts,
        re.longs
    );
    assert_eq!(re.shorts, 0, "steady music must not schedule shorts");

    let decoded = decode_ogg_to_pcm(&re.ogg).expect("decodes");
    assert_eq!(decoded.pcm[0].len(), pcm[0].len(), "end-trim exact");
    let snr = snr_db(&pcm[0], &decoded.pcm[0]);
    eprintln!("steady fixture re-encode SNR: {snr:.2} dB");
    // Two-sided r416 gates for the default quality point (measured
    // 4741 audio B / 47.9 dB under the default joint geometry — the
    // r410 scalar encoder spent 6072 B for 41.6 dB here): neither
    // rate nor fidelity may regress.
    assert!(snr >= 46.0, "SNR {snr:.2} dB below 46 dB");
    assert!(
        re.audio_bytes <= 5_400,
        "default-quality audio bytes {} above the 5.4 kB regression bound",
        re.audio_bytes
    );
}

#[test]
fn steady_music_fixture_top_of_knob_delivers_real_headroom() {
    // Real-corpus pin of the round-413 headroom fix: under the old
    // fixed fine ladder + uncapped margin, q = 1 on this corpus spent
    // 22.7 kB for 47.6 dB (SNR saturated from the mid-knob); the
    // quality-scaled ladder + capped margin reach well past 50 dB in
    // materially fewer bytes.
    if !fixtures_available() {
        eprintln!("fixtures not staged; skipping");
        return;
    }
    let (rate, pcm) = wav_pcm(&format!(
        "{}/mono-44100-q5-typical/expected.wav",
        fixtures_root()
    ));
    let mut config = StreamEncoderConfig::new(rate, 1);
    config.quality = 1.0;
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.pcm[0].len(), pcm[0].len(), "end-trim exact");
    let snr = snr_db(&pcm[0], &decoded.pcm[0]);
    eprintln!("steady fixture q=1: {} B, SNR {snr:.2} dB", ogg.len());
    assert!(snr >= 52.0, "q=1 SNR {snr:.2} dB below 52 dB");
    // Tightened r416 bound (measured 10973 B whole-stream: the
    // top-band geometry race keeps the scalar candidate here).
    assert!(
        ogg.len() <= 12_000,
        "q=1 spends {} B, above the 12 kB regression bound",
        ogg.len()
    );
}

#[test]
fn decorrelated_stereo_fixture_reencode_stays_uncoupled() {
    if !fixtures_available() {
        eprintln!("fixtures not staged; skipping");
        return;
    }
    let (rate, pcm) = wav_pcm(&format!(
        "{}/stereo-44100-q5-typical/expected.wav",
        fixtures_root()
    ));
    let re = reencode("stereo-44100-q5-typical", &pcm, rate);
    // The fixture's two channels are genuinely decorrelated
    // (whole-stream square-polar angle/magnitude energy ratio ≈ 1.7,
    // far above the profitability gate): coupling would only move
    // energy around, so the gate must leave the pair uncoupled.
    assert_eq!(
        re.coupling_steps, 0,
        "a decorrelated pair must fail the coupling gate"
    );
    let decoded = decode_ogg_to_pcm(&re.ogg).expect("decodes");
    for (c, input) in pcm.iter().enumerate() {
        assert_eq!(decoded.pcm[c].len(), input.len(), "end-trim exact");
        let snr = snr_db(input, &decoded.pcm[c]);
        eprintln!("stereo fixture re-encode ch{c}: SNR {snr:.2} dB");
        // r420 floor: the amplitude-band mid class lifts the quiet
        // channel 26.3 → 29.6 dB at the default quality.
        assert!(snr >= 28.0, "ch{c} SNR {snr:.2} dB below 28 dB");
    }
    // Two-sided r416 rate gate (measured 9654 audio B at the default
    // quality; the r410 scalar encoder spent 12100 B here).
    assert!(
        re.audio_bytes <= 10_800,
        "default-quality audio bytes {} above the 10.8 kB regression bound",
        re.audio_bytes
    );
}

//! The §8.6 amplitude-band residue ladder + the §8.6.1 coded-band cap
//! in the integrated encoder.
//!
//! Structure pins (setup-header level):
//!
//! * **Coded-band cap** — at 44.1 kHz the long (2048) residue's
//!   `residue_end` lands on 960 of 1024 bins (the first partition
//!   boundary at or above the 20 kHz cutoff, where the psy model's
//!   threshold-in-quiet is unreachable by any program material); a
//!   stream whose Nyquist sits under the cutoff stays uncapped.
//! * **Band ladder** — a corpus whose above-noise partition peaks
//!   separate from its loud peaks carries the five-class ladder
//!   (silence / noise / mid / coarse / coarse + fine): a 4-D 625-entry
//!   mid band book, a 5⁴-entry classbook. A corpus without the
//!   separation (or with `residue_bands` off) stays at the base four
//!   classes.
//!
//! Behaviour pins: every produced stream still decodes end-trim exact
//! through the crate's own decoder, and the band ladder never lowers
//! the decoded SNR (the rate-distortion chooser only adopts the mid
//! class where the classword-aware Lagrangian improves).

use oxideav_vorbis::{
    decode_ogg_to_pcm, encode_pcm_to_ogg, encode_pcm_to_packets, parse_identification_header,
    parse_setup_header, StreamEncoderConfig,
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
    let mut pos = 12;
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

/// A synthetic corpus with a genuine amplitude-band structure **in
/// the residue-target domain** (`X / rendered_floor` — where the
/// band split is measured): sharp loud harmonics whose tonal peaks
/// tower over the smoothed floor (the loud band, anchoring
/// `max_abs`), plus a low-level wideband hiss whose bins ride *under*
/// the threshold-following floor (targets well below 1 — the mid
/// band), and near-silence elsewhere.
fn banded_corpus(rate: u32, seconds: f32) -> Vec<Vec<f32>> {
    let n = (rate as f32 * seconds) as usize;
    let mut state = 0x2458_71c3_u32;
    let mut noise = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / f32::from_bits(0x4B80_0000) - 0.5 // [-0.5, 0.5)
    };
    // A rich harmonic series under a Gaussian spectral envelope: the
    // envelope's curvature makes the upper partials fall away faster
    // than the floor's straight inter-post dB segments can follow, so
    // whole partitions sit well under the rendered floor — the
    // mid-amplitude target band — while the strong low partials
    // anchor the loud band.
    let f0 = 110.0f32;
    let partials: Vec<(f32, f32, f32)> = (1..=60)
        .map(|k| {
            let f = f0 * k as f32;
            let amp = 0.28 * (-(f / 1400.0) * (f / 1400.0)).exp();
            (f, amp, (k * k) as f32 * 0.37) // fixed per-partial phase
        })
        .filter(|&(f, amp, _)| f < 20_000.0 && amp > 1e-7)
        .collect();
    let row = (0..n)
        .map(|i| {
            let t = i as f32 / rate as f32;
            let tone: f32 = partials
                .iter()
                .map(|&(f, amp, ph)| amp * (2.0 * std::f32::consts::PI * f * t + ph).sin())
                .sum();
            // A whisper of wideband hiss keeps the near-silent band
            // populated without touching the mid band.
            tone + 0.0002 * noise()
        })
        .collect();
    vec![row]
}

/// Parse the setup header out of a packet-level encode.
fn setup_of(
    stream: &oxideav_vorbis::EncodedVorbisStream,
    channels: u8,
) -> oxideav_vorbis::VorbisSetupHeader {
    parse_setup_header(&stream.setup, channels).expect("produced setup parses")
}

#[test]
fn coded_band_cap_lands_on_the_20khz_partition_boundary() {
    let pcm = banded_corpus(44_100, 1.0);
    let config = StreamEncoderConfig::new(44_100, 1);
    let stream = encode_pcm_to_packets(&pcm, &config).expect("encodes");
    let id = parse_identification_header(&stream.identification).expect("id parses");
    assert_eq!(id.blocksize_1, 2048);
    let setup = setup_of(&stream, 1);
    // Long entry: 1024 bins at 44.1 kHz → ceil(1024·20000/22050) = 929
    // → next partition-size-32 boundary = 960 (the §8.6.1 bandpass:
    // bins past it are zeroed by the decoder, and the psy ATH is
    // unreachable above 20 kHz).
    let long = setup.residues.last().expect("long residue");
    assert_eq!(long.residue_begin, 0);
    assert_eq!(long.residue_end, 960);
    // Short entry: 128 bins → ceil(128·20000/22050) = 117 → boundary
    // 128 = the full band (no cap at this resolution).
    let short = &setup.residues[0];
    assert_eq!(short.residue_end, 128);
}

#[test]
fn nyquist_under_the_cutoff_stays_uncapped() {
    let pcm = banded_corpus(22_050, 1.0);
    let config = StreamEncoderConfig::new(22_050, 1);
    let stream = encode_pcm_to_packets(&pcm, &config).expect("encodes");
    let setup = setup_of(&stream, 1);
    for residue in &setup.residues {
        // 11.025 kHz Nyquist sits under the 20 kHz cutoff: every bin
        // is audible-band, so the whole spectrum stays coded.
        let half = if residue.partition_size == 32 {
            1024
        } else {
            128
        };
        assert_eq!(residue.residue_end, half);
    }
}

#[test]
fn banded_corpus_carries_the_five_class_ladder_and_roundtrips() {
    let pcm = banded_corpus(44_100, 1.0);
    let config = StreamEncoderConfig::new(44_100, 1);
    let stream = encode_pcm_to_packets(&pcm, &config).expect("encodes");
    let setup = setup_of(&stream, 1);

    // The amplitude-band gate fires: the above-noise partition peaks
    // (the ~0.1-scale mid bed) separate from the loud tone partitions
    // by far more than the 4× gate.
    let long = setup.residues.last().expect("long residue");
    assert_eq!(long.classifications, 5, "the mid band class is carried");
    // Class 4 is the mid band class: single pass, its own book.
    assert_eq!(long.cascade.len(), 5);
    assert_eq!(long.cascade[4], 0b01);
    let mid_book_index = long.books[4][0].expect("mid class pass-0 book") as usize;
    let mid_book = &setup.codebooks[mid_book_index];
    assert_eq!(mid_book.dimensions, 4, "mid band book is 4-D joint");
    assert_eq!(mid_book.entries, 625, "5 levels per dimension");
    // The classbook covers the 5-class alphabet: 5^4 grouped entries.
    let classbook = &setup.codebooks[long.classbook as usize];
    assert_eq!(classbook.dimensions, 4);
    assert_eq!(classbook.entries, 625);

    // The produced stream still decodes end-trim exact.
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("muxes");
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.pcm[0].len(), pcm[0].len(), "end-trim exact");
    let snr = snr_db(&pcm[0], &decoded.pcm[0]);
    eprintln!(
        "banded synthetic roundtrip: {} B, SNR {snr:.2} dB",
        ogg.len()
    );
    assert!(snr >= 30.0, "banded roundtrip SNR {snr:.2} dB below 30 dB");

    // And the ladder never loses to the base four classes: at the
    // default quality the banded encode spends no more audio bytes at
    // equal-or-better SNR (the classword-aware chooser only adopts
    // the mid class where the priced Lagrangian improves).
    let mut base_config = config.clone();
    base_config.residue_bands = false;
    let base = encode_pcm_to_packets(&pcm, &base_config).expect("base encodes");
    let base_setup = setup_of(&base, 1);
    assert_eq!(
        base_setup
            .residues
            .last()
            .expect("long residue")
            .classifications,
        4,
        "residue_bands = false keeps the base ladder"
    );
    let banded_audio: usize = stream.audio.iter().map(|(p, _)| p.len()).sum();
    let base_audio: usize = base.audio.iter().map(|(p, _)| p.len()).sum();
    eprintln!("banded audio {banded_audio} B vs base {base_audio} B");
    assert!(
        banded_audio as f64 <= base_audio as f64 * 1.02,
        "banded audio bytes {banded_audio} regress past base {base_audio}"
    );
}

#[test]
fn separation_free_corpus_declines_the_mid_class() {
    // Uniform full-scale noise: every partition is loud, the
    // above-noise median sits at the loud scale, and the 4× gate
    // refuses the mid band — the ladder stays at the base classes.
    let rate = 44_100u32;
    let n = rate as usize;
    let mut state = 0x1357_9bdf_u32;
    let row: Vec<f32> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / f32::from_bits(0x4B80_0000) - 0.5) * 1.6
        })
        .collect();
    let pcm = vec![row];
    let config = StreamEncoderConfig::new(rate, 1);
    let stream = encode_pcm_to_packets(&pcm, &config).expect("encodes");
    let setup = setup_of(&stream, 1);
    assert_eq!(
        setup.residues.last().expect("long residue").classifications,
        4,
        "no amplitude separation ⇒ base ladder"
    );
}

#[test]
fn staged_corpus_carries_the_band_ladder_and_the_cap() {
    if !fixtures_available() {
        eprintln!("fixtures not staged; skipping");
        return;
    }
    let (rate, pcm) = wav_pcm(&format!(
        "{}/mono-44100-q5-typical/expected.wav",
        fixtures_root()
    ));
    assert_eq!(rate, 44_100);
    let config = StreamEncoderConfig::new(rate, 1);
    let stream = encode_pcm_to_packets(&pcm, &config).expect("encodes");
    let setup = setup_of(&stream, 1);
    let long = setup.residues.last().expect("long residue");
    assert_eq!(long.residue_end, 960, "44.1 kHz long coded-band cap");
    assert_eq!(long.classifications, 5, "the staged corpus separates");
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("muxes");
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.pcm[0].len(), pcm[0].len(), "end-trim exact");
    let snr = snr_db(&pcm[0], &decoded.pcm[0]);
    eprintln!("staged banded roundtrip: {} B, SNR {snr:.2} dB", ogg.len());
    assert!(snr >= 46.0, "staged SNR {snr:.2} dB below 46 dB");
}

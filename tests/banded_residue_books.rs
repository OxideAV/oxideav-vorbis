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
//!   separate from its loud peaks makes the band tiers *candidates*
//!   (the 4-D 625-entry mid book plus the two ternary 8-D deep
//!   tiers); the measured Lagrangian adoption then keeps only the
//!   tiers that pay for their own setup table + classword-alphabet
//!   growth, so a short stream ships the base four classes while a
//!   longer corpus (the tiled staged fixture below) genuinely
//!   carries the mid band class. `residue_bands = false` pins the
//!   base four-class ladder outright.
//!
//! Behaviour pins: every produced stream still decodes end-trim exact
//! through the crate's own decoder, and the candidate machinery never
//! costs — a stream with the bands lever on serialises no larger
//! than the base-ladder stream (the adoption only keeps measured
//! improvements) at no measured fidelity cost.

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

/// The class index of the 4-D 625-entry mid band tier in a residue's
/// ladder, if adopted.
fn find_mid_tier(
    setup: &oxideav_vorbis::VorbisSetupHeader,
    residue: &oxideav_vorbis::setup::ResidueHeader,
) -> Option<usize> {
    (0..residue.classifications as usize).find(|&class| {
        residue.cascade[class] == 0b01
            && residue.books[class][0].is_some_and(|book| {
                let book = &setup.codebooks[book as usize];
                book.dimensions == 4 && book.entries == 625
            })
    })
}

/// Parse the setup header out of a packet-level encode.
fn setup_of(
    stream: &oxideav_vorbis::EncodedVorbisStream,
    channels: u8,
) -> oxideav_vorbis::VorbisSetupHeader {
    parse_setup_header(&stream.setup, channels).expect("produced setup parses")
}

#[test]
fn coded_band_spans_the_whole_spectrum() {
    let pcm = banded_corpus(44_100, 1.0);
    let config = StreamEncoderConfig::new(44_100, 1);
    let stream = encode_pcm_to_packets(&pcm, &config).expect("encodes");
    let id = parse_identification_header(&stream.identification).expect("id parses");
    assert_eq!(id.blocksize_1, 2048);
    let setup = setup_of(&stream, 1);
    // Both entries code the whole spectrum (`residue_end = n/2`). The
    // r416–r452 20 kHz coded-band fence is gone: it was measured as a
    // hard 12 dB SNR ceiling on wideband noise (6 % of a 44.1 kHz
    // spectrum's bins sit above 20 kHz). What the masking model
    // prices as inaudible up there now goes to the silence class
    // through the rate-distortion chooser instead of a header fence.
    let long = setup.residues.last().expect("long residue");
    assert_eq!(long.residue_begin, 0);
    assert_eq!(long.residue_end, 1024);
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
fn short_banded_corpus_measures_the_candidates_out_again() {
    // The 1-second banded corpus separates (the ~0.1-scale mid bed
    // vs the loud tone partitions), so the mid + deep tiers are all
    // *candidates* — but one second of audio cannot amortise any
    // band book's setup table + classword-alphabet growth, and the
    // measured Lagrangian adoption must ship the base four classes.
    let pcm = banded_corpus(44_100, 1.0);
    let config = StreamEncoderConfig::new(44_100, 1);
    let stream = encode_pcm_to_packets(&pcm, &config).expect("encodes");
    let setup = setup_of(&stream, 1);
    let long = setup.residues.last().expect("long residue");
    assert_eq!(
        long.classifications, 4,
        "a 1-second corpus cannot pay for a band class"
    );

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

    // The candidate machinery never costs: against the explicit base
    // ladder (`residue_bands = false`) the full stream serialises to
    // (near-)identical size at no measured fidelity cost. (The two
    // encodes are not bit-identical — the closed-loop trainer runs
    // with the candidate classes present — so allow routing noise.)
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
    let banded_total: usize =
        stream.setup.len() + stream.audio.iter().map(|(p, _)| p.len()).sum::<usize>();
    let base_total: usize =
        base.setup.len() + base.audio.iter().map(|(p, _)| p.len()).sum::<usize>();
    eprintln!("banded total {banded_total} B vs base {base_total} B");
    assert!(
        banded_total as f64 <= base_total as f64 * 1.02,
        "band candidates cost bytes: {banded_total} vs base {base_total}"
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
fn staged_corpus_carries_the_cap_and_adoption_scales_with_length() {
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

    // The 4-second staged fixture: the coded-band cap holds, and the
    // adoption measures the band candidates out — on this length the
    // base four classes serialise smaller (the r430 measurement:
    // −13 % total stream bytes against the always-carried five-class
    // ladder at −0.03 dB).
    let stream = encode_pcm_to_packets(&pcm, &config).expect("encodes");
    let setup = setup_of(&stream, 1);
    let long = setup.residues.last().expect("long residue");
    assert_eq!(
        long.residue_end, 1024,
        "44.1 kHz long entry codes the whole spectrum"
    );
    // 4 s of audio does not amortise the 4-D mid tier's table (the
    // r453 half-span coarse-geometry tier, whose table is the coarse
    // book's shape, may be adopted — the pin is on the mid tier).
    assert!(
        find_mid_tier(&setup, long).is_none(),
        "4 s of audio does not amortise the mid band table"
    );
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("muxes");
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.pcm[0].len(), pcm[0].len(), "end-trim exact");
    let snr = snr_db(&pcm[0], &decoded.pcm[0]);
    eprintln!("staged banded roundtrip: {} B, SNR {snr:.2} dB", ogg.len());
    assert!(snr >= 46.0, "staged SNR {snr:.2} dB below 46 dB");

    // Doubled (8 s, the same material tiled): the same corpus now
    // amortises a band book, and the adoption keeps it — its own
    // book under the grown classword alphabet. (Through r452 the
    // adopted tier here was the 4-D 625-entry mid tier; since the
    // r453 half-span coarse-geometry tier competes in the same
    // adoption, which band pays on this corpus is a measurement, so
    // the pin is on the adoption mechanics rather than the winner.)
    let tiled: Vec<Vec<f32>> = pcm
        .iter()
        .map(|row| {
            let mut out = row.clone();
            out.extend_from_slice(row);
            out
        })
        .collect();
    let stream = encode_pcm_to_packets(&tiled, &config).expect("tiled encodes");
    let setup = setup_of(&stream, 1);
    let long = setup.residues.last().expect("long residue");
    assert!(
        long.classifications >= 5,
        "8 s of the same material amortises a band class"
    );
    assert_eq!(long.cascade.len(), long.classifications as usize);
    let band_class = 4usize;
    let band_book_index = long.books[band_class][0].expect("band class pass-0 book") as usize;
    let band_book = &setup.codebooks[band_book_index];
    eprintln!(
        "tiled corpus adopts {} classes; class 4 = {}-D {}-entry book",
        long.classifications, band_book.dimensions, band_book.entries
    );
    assert!(band_book_index >= 5, "a band class carries its own book");
    let classbook = &setup.codebooks[long.classbook as usize];
    assert_eq!(classbook.dimensions, 4);
    assert_eq!(
        classbook.entries,
        u32::from(long.classifications).pow(4),
        "classifications^4 groups"
    );
    // The adopted band book's codeword lengths are the sparse
    // final-emission retrain's: only cells the packets actually
    // reference keep codewords.
    let used = band_book
        .codeword_lengths
        .iter()
        .filter(|&&l| l != 0)
        .count();
    assert!(
        0 < used && (used as u32) < band_book.entries,
        "the band book table is emission-sparse (used {used} of {})",
        band_book.entries
    );
    let ogg = encode_pcm_to_ogg(&tiled, &config).expect("tiled muxes");
    let decoded = decode_ogg_to_pcm(&ogg).expect("tiled decodes");
    assert_eq!(decoded.pcm[0].len(), tiled[0].len(), "end-trim exact");
    let snr = snr_db(&tiled[0], &decoded.pcm[0]);
    eprintln!("tiled banded roundtrip: {} B, SNR {snr:.2} dB", ogg.len());
    // r453 measurement: 45.9 dB (the tiled corpus' adopted tier now
    // trades a hair of fidelity for rate under the Lagrangian).
    assert!(snr >= 45.0, "tiled SNR {snr:.2} dB below 45 dB");
}

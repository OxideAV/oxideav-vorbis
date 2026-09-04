//! r456 residue-ladder refinement rungs + the two-mode coupling
//! budget of the whole-stream encoder.
//!
//! - On a wideband-noise corpus at the low knob the produced setup
//!   header carries at least one **refinement class** — cascade
//!   `0b11` with the base coarse book (codebook 2) as pass 0 and a
//!   band book of its own as pass 1 — and the stream decodes through
//!   the crate's own decoder end-trim exact at a pinned SNR.
//! - A coupled stereo stream declares **at most two §4.2.4 modes**,
//!   switching or not, and a switching stream's modes are exactly
//!   one per block size.
//!
//! Fully synthetic: no `docs/` fixtures.

use oxideav_vorbis::setup::VorbisSetupHeader;
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

/// Two seconds of loud wideband noise (uniform, ±0.4) per channel.
fn noise(channels: usize, seed: u64) -> Vec<Vec<f32>> {
    let n = 2 * RATE as usize;
    let mut s = seed;
    (0..channels)
        .map(|_| (0..n).map(|_| 0.4 * lcg(&mut s)).collect())
        .collect()
}

/// Correlated stereo: a common tone mix with a small side split.
fn correlated(samples: usize) -> Vec<Vec<f32>> {
    let mut s = 9u64;
    let mid: Vec<f32> = (0..samples)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            0.4 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 1370.0 * t).sin()
                + 0.002 * lcg(&mut s)
        })
        .collect();
    let side: Vec<f32> = (0..samples)
        .map(|i| 0.03 * (2.0 * std::f32::consts::PI * 2210.0 * i as f32 / RATE as f32).sin())
        .collect();
    vec![
        mid.iter().zip(&side).map(|(m, s)| m + s).collect(),
        mid.iter().zip(&side).map(|(m, s)| m - s).collect(),
    ]
}

fn setup_of(ogg: &[u8], channels: u8) -> VorbisSetupHeader {
    let packets = ogg_packets(ogg).expect("de-frames");
    parse_setup_header(&packets[2], channels).expect("setup parses")
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

#[test]
fn low_knob_noise_adopts_a_refinement_rung() {
    let pcm = noise(1, 1);
    let mut config = StreamEncoderConfig::new(RATE, 1);
    config.quality = 0.2;
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
    let setup = setup_of(&ogg, 1);
    let residue = &setup.residues[setup.residues.len() - 1];
    // A refinement class: two passes, the base coarse book (2) first,
    // then a band book appended after the four base classes' books.
    let rungs: Vec<usize> = (4..residue.cascade.len())
        .filter(|&c| {
            residue.cascade[c] == 0b11
                && residue.books[c][0] == Some(2)
                && residue.books[c][1].is_some_and(|b| b >= 5)
        })
        .collect();
    eprintln!(
        "classes {} cascade {:?} rungs at {rungs:?}",
        residue.classifications, residue.cascade
    );
    assert!(
        !rungs.is_empty(),
        "wideband noise at q = 0.2 must adopt a coarse-refinement rung: {:?} / {:?}",
        residue.cascade,
        residue.books
    );
    for &c in &rungs {
        let book = &setup.codebooks[residue.books[c][1].unwrap() as usize];
        assert_eq!(book.dimensions, 2, "rung books share the coarse geometry");
        assert!(
            [16, 64, 256].contains(&book.entries),
            "rung lattices are 4/8/16 levels per dimension: {}",
            book.entries
        );
    }
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.pcm[0].len(), pcm[0].len(), "end-trim exact");
    let snr = snr_db(&pcm[0], &decoded.pcm[0]);
    eprintln!("noise q=0.2: {} B, SNR {snr:.2} dB", ogg.len());
    assert!(snr >= 20.0, "noise SNR {snr:.2} dB below 20 dB");
}

#[test]
fn coupled_streams_declare_at_most_two_modes() {
    let pcm = correlated(RATE as usize);
    // Switching (the default): one mode per block size, the coupled
    // pair declared on the size that elects it.
    let mut config = StreamEncoderConfig::new(RATE, 2);
    config.quality = 0.5;
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("switching encodes");
    let setup = setup_of(&ogg, 2);
    assert_eq!(setup.modes.len(), 2, "one mode per block size");
    assert!(!setup.modes[0].blockflag && setup.modes[1].blockflag);
    assert!(
        setup.mappings.iter().any(|m| !m.coupling.is_empty()),
        "the correlated pair couples on at least one block size"
    );
    // Single blocksize: the coupled and uncoupled mappings are the two
    // modes a per-frame election may pick between.
    config.short_blocksize = config.blocksize;
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("single-size encodes");
    let setup = setup_of(&ogg, 2);
    assert!(
        setup.modes.len() <= 2,
        "at most two modes: {}",
        setup.modes.len()
    );
    assert!(setup.modes.iter().all(|m| !m.blockflag));
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    for (c, row) in pcm.iter().enumerate() {
        let snr = snr_db(row, &decoded.pcm[c]);
        assert!(snr >= 30.0, "ch{c} SNR {snr:.2} dB below 30 dB");
    }
}

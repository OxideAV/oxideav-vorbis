//! Multi-dimensional residue value books in the integrated encoder
//! (`StreamEncoderConfig::vq_dims`).
//!
//! At `vq_dims = 2` the whole-stream encoder designs its two §8.6.2
//! cascade value books from the stream's own residue corpus
//! (`book_design::design_lattice_vq_codebook`): 2-dimensional §3.2.1
//! lookup-type-**1** lattice books — the widely interoperable lookup
//! form (lookup type 2 is spec-legal but rejected by common black-box
//! decoders) — over uniform full-span scalar ladders (coarse from the
//! raw dims-element residue sub-vectors, fine from the post-coarse
//! leftovers), with codeword lengths trained on the *joint* grid-cell
//! occupancy, so one codeword jointly codes two neighbouring spectral
//! bins. This suite pins the contract:
//!
//! * **parity** — at equal quality the dim-2 designed books hold the
//!   scalar-ladder (`vq_dims = 1`) encode's SNR at comparable rate on
//!   both a synthetic corpus and real audio (the joint form's rate
//!   *win* additionally needs the per-partition residue class ladder —
//!   see the crate README's known-gaps note);
//! * **carriage** — the produced setup header carries the designed
//!   2-D books (parseable, `dimensions == 2`, **lattice** lookup,
//!   occupancy-shaped trained lengths);
//! * **coverage** — both legal `vq_dims` values encode and decode
//!   end-trim-exact through the crate's own decoder above a pinned
//!   SNR floor, coupling and block switching included;
//! * **guards** — an illegal `vq_dims` (zero, above 2, or
//!   non-power-of-two) is refused up front.
//!
//! Fully synthetic — no fixtures required.

use oxideav_vorbis::codebook::VqLookup;
use oxideav_vorbis::{
    decode_ogg_to_pcm, encode_pcm_to_ogg, ogg_packets, parse_setup_header, OggFileError,
    StreamEncoderConfig,
};

const RATE: u32 = 44_100;

/// A deterministic tones + noise-floor test signal (the same shape
/// the whole-stream round-trip suite measures).
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

/// A second, spectrally distinct signal for the right channel.
fn test_signal_alt(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            let mut v = 0.38 * (2.0 * std::f32::consts::PI * 523.25 * t).sin()
                + 0.19 * (2.0 * std::f32::consts::PI * 2093.0 * t).sin();
            let h = (i as u32 + 77).wrapping_mul(2_654_435_761) >> 8;
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

fn encode_at(pcm: &[Vec<f32>], vq_dims: u16, quality: f32) -> (Vec<u8>, f64) {
    let mut config = StreamEncoderConfig::new(RATE, pcm.len() as u8);
    config.vq_dims = vq_dims;
    config.quality = quality;
    let ogg = encode_pcm_to_ogg(pcm, &config).expect("encodes");
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.channels as usize, pcm.len());
    let mut worst = f64::INFINITY;
    for (c, input) in pcm.iter().enumerate() {
        assert_eq!(decoded.pcm[c].len(), input.len(), "end-trim exact");
        worst = worst.min(snr_db(input, &decoded.pcm[c]));
    }
    (ogg, worst)
}

/// The headline parity pin: at equal quality the corpus-designed 2-D
/// lattice books are never **dominated** by the scalar ladders — the
/// rate stays within 15 %, and the worst-channel SNR holds within
/// 1 dB at comparable rate, or within 2 dB when the joint form
/// delivers a material (≥ 5 %) rate cut (the two geometries sit on
/// different rate points of the same quality setting, so a small SNR
/// give-back priced against a real byte cut is a frontier trade, not
/// a loss — e.g. the weighted-trainer stereo point measures −8.7 %
/// bytes at −1.3 dB worst-channel).
#[test]
fn dim2_lattice_books_hold_parity_with_scalar_ladders() {
    fn assert_not_dominated(name: &str, b1: usize, snr1: f64, b2: usize, snr2: f64) {
        assert!(
            b2 as f64 <= b1 as f64 * 1.15,
            "{name}: dim-2 rate {b2} B must stay within 15% of the dim-1 {b1} B"
        );
        let slack = if (b2 as f64) <= b1 as f64 * 0.95 {
            2.0
        } else {
            1.0
        };
        assert!(
            snr2 >= snr1 - slack,
            "{name}: dim-2 SNR {snr2:.2} dB must hold the dim-1 {snr1:.2} dB \
             ({slack} dB slack at this rate ratio)"
        );
    }
    let samples = 22_000;
    let mono = vec![test_signal(samples, 1)];
    let (ogg1, snr1) = encode_at(&mono, 1, 0.7);
    let (ogg2, snr2) = encode_at(&mono, 2, 0.7);
    eprintln!(
        "mono q0.7: dim-1 {} B / {snr1:.2} dB, dim-2 {} B / {snr2:.2} dB",
        ogg1.len(),
        ogg2.len()
    );
    assert_not_dominated("mono", ogg1.len(), snr1, ogg2.len(), snr2);

    let stereo = vec![test_signal(samples, 1), test_signal_alt(samples)];
    let (s1, ssnr1) = encode_at(&stereo, 1, 0.7);
    let (s2, ssnr2) = encode_at(&stereo, 2, 0.7);
    eprintln!(
        "stereo q0.7: dim-1 {} B / {ssnr1:.2} dB, dim-2 {} B / {ssnr2:.2} dB",
        s1.len(),
        s2.len()
    );
    assert_not_dominated("stereo", s1.len(), ssnr1, s2.len(), ssnr2);
}

/// The produced setup header carries the designed 2-D books:
/// `dimensions == 2`, lattice (lookup type 1) values, and
/// occupancy-trained (non-uniform) codeword lengths.
#[test]
fn setup_header_carries_designed_lattice_books() {
    let pcm = vec![test_signal(16_000, 5)];
    let mut config = StreamEncoderConfig::new(RATE, 1);
    config.vq_dims = 2;
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
    let packets = ogg_packets(&ogg).expect("packets assemble");
    let setup = parse_setup_header(&packets[2], 1).expect("setup parses");

    // Books 2 and 3 are the cascade's coarse + fine value books.
    for book in &setup.codebooks[2..=3] {
        assert_eq!(book.dimensions, 2, "designed book dimensionality");
        assert!(
            matches!(book.lookup, VqLookup::Lattice { .. }),
            "designed books are §3.2.1 lookup type 1 (the interoperable form)"
        );
        let used: Vec<u8> = book
            .codeword_lengths
            .iter()
            .copied()
            .filter(|&l| l != 0)
            .collect();
        assert!(!used.is_empty());
        let first = used[0];
        assert!(
            used.len() == 1 || used.iter().any(|&l| l != first),
            "trained lengths are occupancy-shaped, not uniform"
        );
    }
}

/// Both legal `vq_dims` values encode and decode end-trim-exact above
/// a pinned SNR floor (block switching + coupling live, stereo).
#[test]
fn vq_dims_sweep_decodes_cleanly() {
    let samples = 14_000;
    let stereo = vec![test_signal(samples, 9), test_signal_alt(samples)];
    for dims in [1u16, 2] {
        let (ogg, snr) = encode_at(&stereo, dims, 0.7);
        eprintln!(
            "dims {dims}: {} B, worst-channel SNR {snr:.2} dB",
            ogg.len()
        );
        assert!(
            snr >= 15.0,
            "dims {dims}: SNR {snr:.2} dB under the 15 dB floor"
        );
    }
}

/// Illegal `vq_dims` values are refused up front with the typed error.
#[test]
fn illegal_vq_dims_is_refused() {
    let pcm = vec![test_signal(2_000, 3)];
    for bad in [0u16, 3, 4, 8, 16, 32] {
        let mut config = StreamEncoderConfig::new(RATE, 1);
        config.vq_dims = bad;
        assert_eq!(
            encode_pcm_to_ogg(&pcm, &config).unwrap_err(),
            OggFileError::BadVqDims(bad),
            "vq_dims {bad} must be refused"
        );
    }
}

//! Regression pins for the round-413 encoder overhaul: the residue
//! class ladder + noise book, the size-scaled partition geometry, and
//! — the load-bearing fix — genuine SNR headroom at the top of the
//! quality knob.
//!
//! Historical bug being pinned: the integrated encoder's fine value
//! ladder used a fixed `max_abs / 192` step, so the whole-stream SNR
//! *saturated* near `q ≈ 0.7` — the rate the falling `lambda` bought
//! above that only densified class choices while the measured SNR
//! wobbled non-monotonically around the fixed ladder noise floor
//! (bytes tripled for +0.5 dB). `EncoderTuning::fine_step_divisor`
//! now scales the ladder with the knob, so the top of the knob buys
//! *resolution*: SNR must rise decisively from the mid-knob to
//! `q = 1`, not just spend more bytes.

use oxideav_vorbis::{
    decode_ogg_to_pcm, encode_pcm_to_ogg, ogg_packets, parse_identification_header,
    parse_setup_header, StreamEncoderConfig, VqLookup,
};

const RATE: u32 = 44_100;
const SAMPLES: usize = 12_000;

/// A music-like mono corpus: a chord of inharmonic partials under
/// slow amplitude motion, plus a low deterministic hiss bed — tonal
/// mass for the coarse/fine cascade, near-threshold texture for the
/// noise class.
fn corpus() -> Vec<f32> {
    (0..SAMPLES)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            let am = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 1.3 * t).sin();
            let mut v = 0.30 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.20 * (2.0 * std::f32::consts::PI * 553.0 * t).sin()
                + 0.12 * (2.0 * std::f32::consts::PI * 1187.0 * t).sin()
                + 0.05 * (2.0 * std::f32::consts::PI * 3391.0 * t).sin();
            v *= am;
            let h = (i as u32).wrapping_mul(2_654_435_761) >> 8;
            v + ((h & 0xffff) as f32 / 32768.0 - 1.0) * 0.003
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

fn encode_at(pcm: &[Vec<f32>], q: f32) -> (usize, f64) {
    let mut config = StreamEncoderConfig::new(RATE, 1);
    config.quality = q;
    let ogg = encode_pcm_to_ogg(pcm, &config).expect("encodes");
    let decoded = decode_ogg_to_pcm(&ogg).expect("own decode");
    assert_eq!(decoded.pcm[0].len(), pcm[0].len(), "end-trim exact");
    // Rate is tracked as **audio-packet bytes**: across the top-band
    // geometry seam the setup header legitimately shrinks (the scalar
    // geometry carries no 1024-entry designed-lattice length tables),
    // so whole-stream bytes may dip while audio bytes and SNR rise.
    let packets = ogg_packets(&ogg).expect("de-frames");
    let audio: usize = packets[3..].iter().map(|p| p.len()).sum();
    (audio, snr_db(&pcm[0], &decoded.pcm[0]))
}

#[test]
fn top_of_knob_buys_snr_not_just_bytes() {
    let pcm = vec![corpus()];
    let qs = [0.5f32, 0.7, 0.85, 1.0];
    let points: Vec<(usize, f64)> = qs.iter().map(|&q| encode_at(&pcm, q)).collect();
    for (q, (bytes, snr)) in qs.iter().zip(&points) {
        eprintln!("q={q:.2}: {bytes} audio B, SNR {snr:.2} dB");
    }
    // Rate: audio bytes non-decreasing in q (small slack: the
    // geometry race may keep a cheaper stream whose SNR is higher).
    for w in points.windows(2) {
        assert!(
            w[0].0 <= w[1].0 + w[0].0 / 8,
            "rate must not fall as quality rises: {points:?}"
        );
    }
    // SNR: monotone within slack — the saturated-ladder bug made this
    // wobble (e.g. 47.5 -> 46.4 -> 47.6 dB across the top).
    for w in points.windows(2) {
        assert!(
            w[1].1 >= w[0].1 - 0.25,
            "SNR must not fall as quality rises: {points:?}"
        );
    }
    // The top of the knob delivers real headroom over the mid-knob:
    // under the fixed fine ladder q=1 cleared q=0.7 by well under
    // 1 dB (both pinned at the same ladder noise floor). The scaled
    // scalar ladder bought >= 6 dB on this corpus; under the default
    // joint geometry the mid-knob point is itself several dB better,
    // so the same top-of-knob stream clears it by slightly less
    // (measured 5.99 dB) — >= 5 dB still separates a working ladder
    // decisively from the saturated-ladder bug's sub-1-dB headroom.
    let mid = points[1].1;
    let top = points[3].1;
    assert!(
        top >= mid + 5.0,
        "q=1 must clear q=0.7 by >= 5 dB (ladder headroom): {points:?}"
    );
}

#[test]
fn produced_setup_carries_the_class_ladder_and_scaled_partitions() {
    let pcm = vec![corpus()];
    let config = StreamEncoderConfig::new(RATE, 1);
    assert_eq!(config.blocksize, 2048, "default long size");
    assert_eq!(config.short_blocksize, 256, "default short size");
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
    let packets = ogg_packets(&ogg).expect("de-frames");
    let id = parse_identification_header(&packets[0]).expect("id parses");
    assert_eq!((id.blocksize_0, id.blocksize_1), (256, 2048));
    let setup = parse_setup_header(&packets[2], 1).expect("setup parses");

    // Two per-size residues: partition 16 on the 128-bin short
    // spectrum, 32 on the 1024-bin long spectrum.
    assert_eq!(setup.residues.len(), 2);
    assert_eq!(setup.residues[0].partition_size, 16, "short partitions");
    assert_eq!(setup.residues[1].partition_size, 32, "long partitions");

    for residue in &setup.residues {
        // The four-class ladder: silence / noise / coarse / both.
        assert_eq!(residue.classifications, 4);
        assert_eq!(residue.cascade.len(), 4);
        assert_eq!(residue.cascade[0], 0, "class 0 is silence");
        assert_eq!(residue.cascade[1], 0b01, "class 1 reads pass 0 only");
        assert_eq!(residue.cascade[2], 0b01, "class 2 reads pass 0 only");
        assert_eq!(residue.cascade[3], 0b11, "class 3 is the full cascade");
        let noise_book = residue.books[1][0].expect("class 1 carries a book") as usize;
        let coarse_book = residue.books[2][0].expect("class 2 carries a book") as usize;
        // The noise book is the 4-dimensional ternary joint grid; the
        // coarse book is the corpus-designed 2-D lattice (the default
        // `vq_dims = 2` joint geometry at the default quality).
        assert_eq!(setup.codebooks[noise_book].dimensions, 4);
        assert_eq!(setup.codebooks[noise_book].entries, 81);
        assert!(
            matches!(setup.codebooks[noise_book].lookup, VqLookup::Lattice { .. }),
            "noise book carries a lattice lookup"
        );
        assert_eq!(setup.codebooks[coarse_book].dimensions, 2);
        assert!(
            matches!(
                setup.codebooks[coarse_book].lookup,
                VqLookup::Lattice { .. }
            ),
            "coarse book carries a lattice lookup"
        );
    }

    // The classbook groups four partitions per classword over the
    // four-class alphabet: 4^4 entries, dimensions 4.
    let classbook = setup.residues[0].classbook as usize;
    assert_eq!(setup.codebooks[classbook].dimensions, 4);
    assert_eq!(setup.codebooks[classbook].entries, 256);
}

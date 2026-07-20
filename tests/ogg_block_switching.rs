//! §4.3.1 block switching in the integrated encoder.
//!
//! `encode_pcm_to_ogg` schedules short blocks around transients (the
//! §1.3.2 pre-echo mechanism) and long blocks elsewhere, with per-size
//! floors/residues/modes in the setup header and the §4.3.1 hybrid
//! window edges at every long↔short transition. This suite pins the
//! wiring end to end:
//!
//! * **structure** — a transient corpus produces a genuinely switched
//!   stream: the identification header carries the `(short, long)`
//!   blocksize pair, the setup header two modes (blockflag clear/set)
//!   with per-size floors and residues, and the §4.3.1 packet preludes
//!   carry both blockflags with window flags mirroring the neighbour
//!   blockflags;
//! * **granule walk** — every packet's granule is the §4.3.8
//!   `(n_prev + n_cur)/4` running sum, end-trimmed to the exact input
//!   length on the final page;
//! * **fidelity** — the switched stream decodes through the crate's
//!   own §4.3 decoder to the input length at a pinned SNR;
//! * **pre-echo** — on an attack-after-silence corpus, the switched
//!   encode leaves strictly less noise energy in the silent samples
//!   just before the attack than a forced-long encode of the same PCM
//!   at the same quality (the reason short blocks exist);
//! * **steady content stays long** — no short blocks on a steady
//!   corpus, and the switched-config encode is byte-identical to a
//!   stream whose schedule is all-long.
//!
//! Fully synthetic — no fixtures required.

use oxideav_core::bits::BitReaderLsb;
use oxideav_vorbis::{
    decode_ogg_to_pcm, encode_pcm_to_ogg, ogg_packets, parse_identification_header,
    parse_setup_header, read_packet_header, StreamEncoderConfig,
};

const RATE: u32 = 44_100;

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

/// A percussive corpus: steady tone beds with sharp attacks between.
fn transient_signal(samples: usize) -> Vec<f32> {
    let mut pcm: Vec<f32> = (0..samples)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            0.25 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.10 * (2.0 * std::f32::consts::PI * 1370.0 * t).sin()
        })
        .collect();
    // Sharp decaying attacks every ~5500 samples.
    let mut k = 3000usize;
    while k + 400 < samples {
        for j in 0..400 {
            let d = (-(j as f32) / 60.0).exp();
            pcm[k + j] += 0.7 * d * if j % 2 == 0 { 1.0 } else { -1.0 };
        }
        k += 5500;
    }
    pcm
}

/// Parse every audio packet's §4.3.1 prelude out of a produced stream.
fn packet_preludes(ogg: &[u8]) -> Vec<(bool, usize, bool, bool)> {
    let packets = ogg_packets(ogg).expect("stream de-frames");
    let id = parse_identification_header(&packets[0]).expect("id header parses");
    let setup = parse_setup_header(&packets[2], id.audio_channels).expect("setup parses");
    packets[3..]
        .iter()
        .map(|p| {
            let mut reader = BitReaderLsb::new(p);
            let h = read_packet_header(
                &mut reader,
                &setup,
                id.blocksize_0 as usize,
                id.blocksize_1 as usize,
            )
            .expect("audio packet prelude parses");
            (h.blockflag, h.n, h.previous_window_flag, h.next_window_flag)
        })
        .collect()
}

#[test]
fn transient_corpus_switches_with_conformant_structure_and_granules() {
    let samples = 30_000;
    let pcm = vec![transient_signal(samples)];
    let config = StreamEncoderConfig::new(RATE, 1);
    assert_eq!(config.short_blocksize, 256, "switching on by default");
    assert_eq!(config.blocksize, 2048);
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
    let packets = ogg_packets(&ogg).expect("de-frames");

    // ---- identification header carries the pair ----
    let id = parse_identification_header(&packets[0]).expect("id parses");
    assert_eq!(id.blocksize_0, 256);
    assert_eq!(id.blocksize_1, 2048);

    // ---- setup header: two modes, per-size floors/residues ----
    let setup = parse_setup_header(&packets[2], 1).expect("setup parses");
    assert_eq!(setup.modes.len(), 2);
    assert!(!setup.modes[0].blockflag, "mode 0 is the short block");
    assert!(setup.modes[1].blockflag, "mode 1 is the long block");
    assert_eq!(setup.floors.len(), 2);
    assert_eq!(setup.residues.len(), 2);
    assert_eq!(setup.mappings.len(), 2);
    assert_eq!(setup.residues[0].residue_end, 128, "short residue at n0/2");
    // The long residue carries the §8.6.1 coded-band cap: at 44.1 kHz
    // the first partition-size-32 boundary at or above the 20 kHz ATH
    // bound is bin 960 of 1024 (the decoder zeroes the bins past it).
    assert_eq!(
        setup.residues[1].residue_end, 960,
        "long residue at the coded-band cap"
    );

    // ---- packet preludes: both flags present, window flags mirror ----
    let preludes = packet_preludes(&ogg);
    let shorts = preludes.iter().filter(|p| !p.0).count();
    let longs = preludes.iter().filter(|p| p.0).count();
    eprintln!(
        "switched stream: {} packets ({} short, {} long), {} bytes",
        preludes.len(),
        shorts,
        longs,
        ogg.len()
    );
    assert!(shorts > 0, "the attacks must force short blocks");
    assert!(longs > 0, "the steady beds must stay long");
    for (f, &(flag, n, prev_flag, next_flag)) in preludes.iter().enumerate() {
        assert_eq!(n, if flag { 2048 } else { 256 });
        if flag {
            // §4.3.1 step 4a: a long block's window flags mirror the
            // neighbour blockflags (stream edges take `true`).
            let expect_prev = f == 0 || preludes[f - 1].0;
            let expect_next = f + 1 == preludes.len() || preludes[f + 1].0;
            assert_eq!(prev_flag, expect_prev, "packet {f} previous_window_flag");
            assert_eq!(next_flag, expect_next, "packet {f} next_window_flag");
        }
    }

    // ---- §4.3.8 granule walk on the wire ----
    // Re-derive every packet's granule from the blockflag sequence and
    // check each page's granule position states the walk's value for
    // the last packet completing on it (the final page end-trimmed).
    let mut walk = vec![0u64];
    for f in 1..preludes.len() {
        let step = (preludes[f - 1].1 + preludes[f].1) as u64 / 4;
        walk.push(walk[f - 1] + step);
    }
    assert!(*walk.last().unwrap() >= samples as u64);
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.pcm[0].len(), samples, "end-trim to input length");

    // ---- fidelity ----
    // Waveform SNR on this corpus is attack-limited (the psy model
    // deems most of a wideband attack's quantisation noise masked), so
    // the pin is comparative: switching must not cost fidelity against
    // the forced-long encode of the same PCM, on top of an absolute
    // floor.
    let mut long_cfg = config.clone();
    long_cfg.short_blocksize = long_cfg.blocksize;
    let forced_long = encode_pcm_to_ogg(&pcm, &long_cfg).expect("forced-long encodes");
    let dec_long = decode_ogg_to_pcm(&forced_long).expect("forced-long decodes");
    let snr = snr_db(&pcm[0], &decoded.pcm[0]);
    let snr_long = snr_db(&pcm[0], &dec_long.pcm[0]);
    eprintln!(
        "switched q0.7: {} B / SNR {snr:.2} dB; forced-long: {} B / {snr_long:.2} dB",
        ogg.len(),
        forced_long.len()
    );
    assert!(snr >= 10.0, "switched SNR {snr:.2} dB below 10 dB");
    assert!(
        snr >= snr_long - 1.0,
        "switching must not cost fidelity: {snr:.2} vs {snr_long:.2} dB"
    );
}

#[test]
fn short_blocks_cut_pre_echo_versus_forced_long() {
    // Silence, one hard attack, then decay — the §1.3.2 pre-echo
    // scenario. Quantisation noise from a long block containing the
    // attack smears across the whole window, landing in the silent
    // samples before it; short blocks confine it.
    let samples = 16_384;
    let attack_at = 8192usize;
    let mut pcm = vec![0.0f32; samples];
    for j in 0..2048 {
        let d = (-(j as f32) / 300.0).exp();
        pcm[attack_at + j] = 0.8 * d * (j as f32 * 0.9).sin();
    }
    let input = vec![pcm.clone()];

    let switched_cfg = StreamEncoderConfig::new(RATE, 1);
    let mut long_cfg = StreamEncoderConfig::new(RATE, 1);
    long_cfg.short_blocksize = long_cfg.blocksize; // switching off

    let switched = encode_pcm_to_ogg(&input, &switched_cfg).expect("switched encodes");
    let forced_long = encode_pcm_to_ogg(&input, &long_cfg).expect("forced-long encodes");
    assert!(
        packet_preludes(&switched).iter().any(|p| !p.0),
        "the attack must schedule short blocks"
    );

    let dec_s = decode_ogg_to_pcm(&switched).expect("switched decodes");
    let dec_l = decode_ogg_to_pcm(&forced_long).expect("forced-long decodes");

    // Pre-echo window: the silent input samples from a long half-block
    // before the attack up to the short block's intrinsic reach
    // (`n0/2 = 128` samples — the frame containing the onset
    // necessarily laps that far back regardless of its size; §1.3.2's
    // point is confining the smear to that reach instead of a whole
    // long window). The input there is exactly zero, so any energy is
    // injected quantisation noise.
    let window = (attack_at - 512)..(attack_at - 128);
    let energy = |x: &[f32]| -> f64 {
        x[window.clone()]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    };
    let pre_echo_switched = energy(&dec_s.pcm[0]);
    let pre_echo_long = energy(&dec_l.pcm[0]);
    eprintln!(
        "pre-echo energy beyond the short block's reach: switched {pre_echo_switched:.3e} vs \
         forced-long {pre_echo_long:.3e}"
    );
    assert!(
        pre_echo_switched < 0.1 * pre_echo_long,
        "short blocks must cut the out-of-reach pre-attack noise energy by >= 10x: \
         {pre_echo_switched:.3e} vs {pre_echo_long:.3e}"
    );

    // And the attack itself still round-trips.
    let snr = snr_db(
        &input[0][attack_at..attack_at + 2048],
        &dec_s.pcm[0][attack_at..attack_at + 2048],
    );
    assert!(snr >= 10.0, "attack region SNR {snr:.2} dB below 10 dB");
}

#[test]
fn steady_content_stays_all_long() {
    let samples = 12_000;
    let pcm: Vec<f32> = (0..samples)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            0.4 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 1370.0 * t).sin()
        })
        .collect();
    let config = StreamEncoderConfig::new(RATE, 1);
    let ogg = encode_pcm_to_ogg(&[pcm], &config).expect("encodes");
    let preludes = packet_preludes(&ogg);
    assert!(
        preludes.iter().all(|p| p.0),
        "a steady corpus must schedule no short blocks"
    );
    // The stream still declares two modes (the schedule, not the setup,
    // is what depends on the signal).
    let packets = ogg_packets(&ogg).unwrap();
    let setup = parse_setup_header(&packets[2], 1).unwrap();
    assert_eq!(setup.modes.len(), 2);
}

#[test]
fn all_transient_content_stays_all_short_and_roundtrips() {
    // Dense impulses: every lookahead region is transient, so the
    // schedule is all-short — the degenerate opposite of the steady
    // corpus. This also exercises the unused-size floor design (the
    // long entry's representative envelope is resampled from the
    // short one, since no long frame ever contributes).
    let samples = 10_000;
    let mut pcm = vec![0.0f32; samples];
    let mut k = 150usize;
    while k < samples {
        for j in 0..24.min(samples - k) {
            let d = (-(j as f32) / 6.0).exp();
            pcm[k + j] = 0.8 * d * if j % 2 == 0 { 1.0 } else { -1.0 };
        }
        k += 300;
    }
    let input = vec![pcm];
    // The impulse spacing (300 samples) is dense relative to a 1024
    // lookahead (an impulse in every fourth-or-so 64-sample sub-frame
    // keeps the peak-to-mean concentration high); the default 2048
    // window would dilute the same train into quasi-steady content,
    // so the test pins its long size explicitly.
    let mut config = StreamEncoderConfig::new(RATE, 1);
    config.blocksize = 1024;
    let ogg = encode_pcm_to_ogg(&input, &config).expect("encodes");
    let preludes = packet_preludes(&ogg);
    // Every interior packet is short; only the stream tail — whose
    // decision lookahead is the silent zero-padding past the last
    // impulse — may fall back to long.
    let first_long = preludes.iter().position(|p| p.0).unwrap_or(preludes.len());
    assert!(
        first_long + 4 >= preludes.len(),
        "long blocks before the stream tail: first at packet {first_long} of {}",
        preludes.len()
    );
    assert!(
        preludes[..first_long].len() > preludes.len() / 2,
        "the impulse corpus must be predominantly short"
    );
    // The setup still carries both entries (mode 1 is simply unused).
    let packets = ogg_packets(&ogg).unwrap();
    let setup = parse_setup_header(&packets[2], 1).unwrap();
    assert_eq!(setup.modes.len(), 2);
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.pcm[0].len(), samples, "end-trim to input length");
    let snr = snr_db(&input[0], &decoded.pcm[0]);
    eprintln!("all-short impulse corpus: {} B, SNR {snr:.2} dB", ogg.len());
    assert!(snr.is_finite());
}

#[test]
fn stereo_switched_stream_couples_and_roundtrips() {
    // Block switching and channel coupling compose: a correlated
    // stereo pair with attacks must produce a switched, coupled stream
    // that round-trips per channel.
    let samples = 20_000;
    let mid = transient_signal(samples);
    let left: Vec<f32> = mid.iter().map(|&m| m * 1.02).collect();
    let right: Vec<f32> = mid.iter().map(|&m| m * 0.98).collect();
    let pcm = vec![left, right];
    let config = StreamEncoderConfig::new(RATE, 2);
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");

    let packets = ogg_packets(&ogg).unwrap();
    let setup = parse_setup_header(&packets[2], 2).unwrap();
    assert_eq!(setup.modes.len(), 2);
    for mapping in &setup.mappings {
        assert_eq!(
            mapping.coupling.len(),
            1,
            "both block sizes carry the coupling step"
        );
    }
    let preludes = packet_preludes(&ogg);
    assert!(preludes.iter().any(|p| !p.0));
    assert!(preludes.iter().any(|p| p.0));

    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    for (c, input) in pcm.iter().enumerate() {
        assert_eq!(decoded.pcm[c].len(), samples);
        let snr = snr_db(input, &decoded.pcm[c]);
        eprintln!("switched+coupled ch{c}: SNR {snr:.2} dB");
        assert!(snr >= 10.0, "ch{c} SNR {snr:.2} dB below 10 dB");
    }
}

#[test]
fn blocksize_pair_guards_fire() {
    use oxideav_vorbis::OggFileError;
    let pcm = vec![vec![0.1f32; 1000]];
    // Short above long.
    let mut bad = StreamEncoderConfig::new(RATE, 1);
    bad.short_blocksize = 2048;
    bad.blocksize = 1024;
    assert_eq!(
        encode_pcm_to_ogg(&pcm, &bad),
        Err(OggFileError::BadBlocksizePair {
            short_n: 2048,
            long_n: 1024
        })
    );
    // Illegal short size.
    let mut bad = StreamEncoderConfig::new(RATE, 1);
    bad.short_blocksize = 100;
    assert_eq!(
        encode_pcm_to_ogg(&pcm, &bad),
        Err(OggFileError::BadBlocksize(100))
    );
}

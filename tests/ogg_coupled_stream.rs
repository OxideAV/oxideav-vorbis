//! §4.3.5 channel coupling in the integrated encoder.
//!
//! `encode_pcm_to_ogg` offers square-polar coupling on adjacent channel
//! pairs, gated per pair on the whole stream's coupling-energy split.
//! This suite pins the wiring end to end:
//!
//! * a correlated stereo pair is coupled — the produced setup header
//!   carries the `(0, 1)` coupling step, the stream is **smaller** than
//!   the dual-mono encode of the same PCM at the same quality, and both
//!   decodes clear the same fidelity bar (rate win at equal quality);
//! * an anti-correlated pair fails the profitability gate — no coupling
//!   step is emitted even with coupling enabled;
//! * a four-channel stream gates each adjacent pair independently;
//! * `coupling: false` carries every channel uncoupled.
//!
//! Fully synthetic — no fixtures required.

use oxideav_vorbis::{
    decode_ogg_to_pcm, encode_pcm_to_ogg, ogg_packets, parse_setup_header, MappingCouplingStep,
    StreamEncoderConfig,
};

const RATE: u32 = 44_100;

/// A deterministic multi-tone test signal.
fn tone_mix(samples: usize, seed: u32) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            let mut v = 0.40 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.20 * (2.0 * std::f32::consts::PI * 1370.0 * t).sin()
                + 0.10 * (2.0 * std::f32::consts::PI * 97.0 * t).cos();
            let h = (i as u32 + seed).wrapping_mul(2_654_435_761) >> 8;
            v += ((h & 0xffff) as f32 / 32768.0 - 1.0) * 0.002;
            v
        })
        .collect()
}

/// A small decorrelated side signal (low-level, spectrally distinct).
fn side_signal(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            0.03 * (2.0 * std::f32::consts::PI * 2210.0 * t).sin()
        })
        .collect()
}

/// Correlated stereo: a common mid signal plus a small L/R side split.
fn correlated_stereo(samples: usize) -> Vec<Vec<f32>> {
    let mid = tone_mix(samples, 7);
    let side = side_signal(samples);
    let left: Vec<f32> = mid.iter().zip(&side).map(|(&m, &s)| m + s).collect();
    let right: Vec<f32> = mid.iter().zip(&side).map(|(&m, &s)| m - s).collect();
    vec![left, right]
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

/// Parse the coupling steps out of a produced stream's setup header.
/// Every mapping (one per block size) must carry the identical step
/// list — the coupling decision is per stream, not per block size.
fn coupling_steps_of(ogg: &[u8], channels: u8) -> Vec<MappingCouplingStep> {
    let packets = ogg_packets(ogg).expect("stream de-frames");
    let setup = parse_setup_header(&packets[2], channels).expect("setup header parses");
    assert!(!setup.mappings.is_empty());
    for mapping in &setup.mappings[1..] {
        assert_eq!(
            mapping.coupling, setup.mappings[0].coupling,
            "all mappings must agree on the coupling steps"
        );
    }
    setup.mappings[0].coupling.clone()
}

#[test]
fn correlated_stereo_couples_and_beats_dual_mono_rate_at_equal_quality() {
    let samples = 22_050;
    let pcm = correlated_stereo(samples);

    let coupled_cfg = StreamEncoderConfig::new(RATE, 2);
    assert!(coupled_cfg.coupling, "coupling is on by default");
    let mut uncoupled_cfg = StreamEncoderConfig::new(RATE, 2);
    uncoupled_cfg.coupling = false;

    let coupled = encode_pcm_to_ogg(&pcm, &coupled_cfg).expect("coupled encodes");
    let uncoupled = encode_pcm_to_ogg(&pcm, &uncoupled_cfg).expect("uncoupled encodes");

    // The setup header records the §4.3.5 step; dual-mono has none.
    assert_eq!(
        coupling_steps_of(&coupled, 2),
        vec![MappingCouplingStep {
            magnitude_channel: 0,
            angle_channel: 1,
        }],
        "correlated pair must pass the coupling gate"
    );
    assert!(coupling_steps_of(&uncoupled, 2).is_empty());

    // Rate: the coupled stream must genuinely undercut dual-mono.
    eprintln!(
        "correlated stereo: coupled {} B vs dual-mono {} B",
        coupled.len(),
        uncoupled.len()
    );
    assert!(
        (coupled.len() as f64) < 0.9 * uncoupled.len() as f64,
        "coupling must save >= 10%: {} vs {} bytes",
        coupled.len(),
        uncoupled.len()
    );

    // Fidelity: both decodes clear the same bar (equal-quality claim).
    let dec_coupled = decode_ogg_to_pcm(&coupled).expect("coupled decodes");
    let dec_uncoupled = decode_ogg_to_pcm(&uncoupled).expect("uncoupled decodes");
    for (c, input) in pcm.iter().enumerate() {
        assert_eq!(dec_coupled.pcm[c].len(), samples, "end-trim to input");
        let snr_c = snr_db(input, &dec_coupled.pcm[c]);
        let snr_u = snr_db(input, &dec_uncoupled.pcm[c]);
        eprintln!("ch{c}: coupled {snr_c:.2} dB, uncoupled {snr_u:.2} dB");
        assert!(snr_c >= 20.0, "coupled ch{c} SNR {snr_c:.2} dB below 20 dB");
        assert!(
            snr_c >= snr_u - 2.0,
            "coupling must not cost meaningful fidelity: ch{c} {snr_c:.2} vs {snr_u:.2} dB"
        );
    }
}

#[test]
fn anti_correlated_stereo_fails_the_profitability_gate() {
    // R = −L: the square-polar angle carries all the energy, so the
    // angle/magnitude ratio sits far above the gate and the pair must
    // be left uncoupled even though coupling is enabled.
    let samples = 12_000;
    let left = tone_mix(samples, 3);
    let right: Vec<f32> = left.iter().map(|&v| -v).collect();
    let pcm = vec![left, right];

    let config = StreamEncoderConfig::new(RATE, 2);
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
    assert!(
        coupling_steps_of(&ogg, 2).is_empty(),
        "anti-correlated pair must be dropped by the gate"
    );

    // And the stream still round-trips.
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    for (c, input) in pcm.iter().enumerate() {
        let snr = snr_db(input, &decoded.pcm[c]);
        assert!(snr >= 20.0, "ch{c} SNR {snr:.2} dB below 20 dB");
    }
}

#[test]
fn four_channel_stream_gates_each_adjacent_pair_independently() {
    // Pair (0,1) correlated; pair (2,3) anti-correlated. Exactly one
    // coupling step must survive the gate.
    let samples = 8_192;
    let stereo = correlated_stereo(samples);
    let c2 = tone_mix(samples, 5);
    let c3: Vec<f32> = c2.iter().map(|&v| -v).collect();
    let pcm = vec![stereo[0].clone(), stereo[1].clone(), c2, c3];

    let config = StreamEncoderConfig::new(RATE, 4);
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
    assert_eq!(
        coupling_steps_of(&ogg, 4),
        vec![MappingCouplingStep {
            magnitude_channel: 0,
            angle_channel: 1,
        }],
        "only the correlated pair may couple"
    );

    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    for (c, input) in pcm.iter().enumerate() {
        assert_eq!(decoded.pcm[c].len(), samples);
        let snr = snr_db(input, &decoded.pcm[c]);
        eprintln!("4ch ch{c}: SNR {snr:.2} dB");
        assert!(snr >= 18.0, "ch{c} SNR {snr:.2} dB below 18 dB");
    }
}

#[test]
fn mono_and_disabled_coupling_emit_no_steps() {
    let samples = 6_000;
    let mono = vec![tone_mix(samples, 9)];
    let config = StreamEncoderConfig::new(RATE, 1);
    let ogg = encode_pcm_to_ogg(&mono, &config).expect("mono encodes");
    assert!(coupling_steps_of(&ogg, 1).is_empty());

    let stereo = correlated_stereo(samples);
    let mut config = StreamEncoderConfig::new(RATE, 2);
    config.coupling = false;
    let ogg = encode_pcm_to_ogg(&stereo, &config).expect("stereo encodes");
    assert!(coupling_steps_of(&ogg, 2).is_empty());
}

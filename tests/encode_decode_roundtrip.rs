//! End-to-end encode → decode round-trip test hitting only the public
//! API surface (no internal submodule access).
//!
//! Covers silence (smallest reasonable packet payload) and a 1 kHz sine
//! (an audible signal we can verify via Goertzel-style DFT magnitude).

use oxideav_codec::CodecRegistry;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Error, Frame, SampleFormat, TimeBase};

fn make_s16_frame(channels: u16, samples_per_channel: usize, pcm: &[i16]) -> Frame {
    let mut data = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        data.extend_from_slice(&s.to_le_bytes());
    }
    Frame::Audio(AudioFrame {
        format: SampleFormat::S16,
        channels,
        sample_rate: 48_000,
        samples: samples_per_channel as u32,
        pts: Some(0),
        time_base: TimeBase::new(1, 48_000),
        data: vec![data],
    })
}

fn encode_decode(channels: u16, samples_per_channel: usize, interleaved: &[i16]) -> Vec<i16> {
    let mut reg = CodecRegistry::new();
    oxideav_vorbis::register(&mut reg);

    let mut params = CodecParameters::audio(CodecId::new("vorbis"));
    params.channels = Some(channels);
    params.sample_rate = Some(48_000);
    params.sample_format = Some(SampleFormat::S16);

    let mut enc = reg.make_encoder(&params).expect("encoder accepts params");
    enc.send_frame(&make_s16_frame(channels, samples_per_channel, interleaved))
        .expect("send_frame");
    enc.flush().expect("flush");

    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(Error::Eof) => break,
            Err(Error::NeedMore) => break,
            Err(e) => panic!("encoder error: {e}"),
        }
    }
    assert!(!packets.is_empty(), "encoder produced no packets");

    let dec_params = enc.output_params().clone();
    let mut dec = reg
        .make_decoder(&dec_params)
        .expect("decoder accepts our extradata");
    let mut out = Vec::new();
    for pkt in &packets {
        dec.send_packet(pkt).expect("send_packet");
        while let Ok(Frame::Audio(af)) = dec.receive_frame() {
            assert_eq!(af.format, SampleFormat::S16);
            assert_eq!(af.channels, channels);
            let plane = &af.data[0];
            for chunk in plane.chunks_exact(2) {
                out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
    }
    out
}

/// Rough energy-at-frequency estimator (Goertzel-ish; good enough for
/// "this frequency dominates that one" assertions).
fn goertzel_mag(samples: &[i16], freq: f64, sr: f64) -> f64 {
    let omega = 2.0 * std::f64::consts::PI * freq / sr;
    let coeff = 2.0 * omega.cos();
    let mut s_prev = 0f64;
    let mut s_prev2 = 0f64;
    for &s in samples {
        let s_now = s as f64 + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s_now;
    }
    (s_prev2.powi(2) + s_prev.powi(2) - coeff * s_prev * s_prev2).sqrt()
}

#[test]
fn silence_mono_roundtrips_via_public_api() {
    let n = 2048 * 4;
    let silence = vec![0i16; n];
    let out = encode_decode(1, n, &silence);
    assert!(!out.is_empty(), "silent roundtrip produced no output");
    let peak = out.iter().map(|&s| s.unsigned_abs()).max().unwrap_or(0);
    assert!(
        peak < 256,
        "silent input should decode near-silent, got peak {peak}"
    );
}

#[test]
fn sine_mono_roundtrips_via_public_api() {
    let n = 2048 * 4;
    let sr = 48_000.0;
    let pcm: Vec<i16> = (0..n)
        .map(|i| {
            let t = i as f64 / sr;
            let s = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.5;
            (s * 32768.0) as i16
        })
        .collect();
    let out = encode_decode(1, n, &pcm);
    assert!(!out.is_empty());
    let target = goertzel_mag(&out, 1000.0, sr);
    let off = goertzel_mag(&out, 7000.0, sr);
    assert!(
        target > off * 10.0,
        "1 kHz tone should dominate 7 kHz by at least 10x: target={target}, off={off}"
    );
}

#[test]
fn sine_stereo_roundtrips_via_public_api() {
    let n = 2048 * 4;
    let sr = 48_000.0;
    let mut pcm: Vec<i16> = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f64 / sr;
        let s = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.5;
        let q = (s * 32768.0) as i16;
        pcm.push(q);
        pcm.push(q);
    }
    let out = encode_decode(2, n, &pcm);
    assert!(!out.is_empty());
    let mut left = Vec::with_capacity(out.len() / 2);
    let mut right = Vec::with_capacity(out.len() / 2);
    for chunk in out.chunks_exact(2) {
        left.push(chunk[0]);
        right.push(chunk[1]);
    }
    let t_l = goertzel_mag(&left, 1000.0, sr);
    let t_r = goertzel_mag(&right, 1000.0, sr);
    let o_l = goertzel_mag(&left, 5000.0, sr);
    let o_r = goertzel_mag(&right, 5000.0, sr);
    assert!(
        t_l > o_l * 5.0,
        "L: 1 kHz should beat 5 kHz, got {t_l} vs {o_l}"
    );
    assert!(
        t_r > o_r * 5.0,
        "R: 1 kHz should beat 5 kHz, got {t_r} vs {o_r}"
    );
}

/// 5.1 channel round-trip via the public CodecRegistry API. Each channel
/// gets a distinct tone; after decode the per-channel Goertzel magnitude
/// at that tone must dominate over a common off-band frequency.
#[test]
fn sine_5_1_roundtrips_via_public_api() {
    let n = 2048 * 4;
    let sr = 48_000.0;
    let freqs = [440.0f64, 520.0, 660.0, 880.0, 1100.0, 140.0];
    let mut pcm: Vec<i16> = Vec::with_capacity(n * 6);
    for i in 0..n {
        let t = i as f64 / sr;
        for &f in &freqs {
            let s = (2.0 * std::f64::consts::PI * f * t).sin() * 0.4;
            pcm.push((s * 32768.0) as i16);
        }
    }
    let out = encode_decode(6, n, &pcm);
    assert!(!out.is_empty());
    // Deinterleave.
    let mut per_ch: Vec<Vec<i16>> = vec![Vec::new(); 6];
    for chunk in out.chunks_exact(6) {
        for (ch, &s) in chunk.iter().enumerate() {
            per_ch[ch].push(s);
        }
    }
    for (ch, &f) in freqs.iter().enumerate() {
        let t = goertzel_mag(&per_ch[ch], f, sr);
        let o = goertzel_mag(&per_ch[ch], 4000.0, sr);
        assert!(
            t > o,
            "5.1 ch{ch}: target {f} ({t}) must beat 4 kHz off-band ({o})"
        );
    }
}

/// Cascade / multi-class residue bitrate gate via the public API. On a
/// sparse-in-frequency signal (pure sine tone) the cascade+multi-class
/// setup should spend considerably fewer residue bits than the prior
/// single-128-entry-book / single-classification encoder would have. We
/// use a conservative bound (total_bytes < baseline_residue_bytes / 2)
/// to guard against accidental regressions that would give back the
/// 30-60% reduction we've just banked.
#[test]
fn sine_mono_cascade_beats_single_book_via_public_api() {
    let n = 2048 * 4;
    let sr = 48_000.0;
    let pcm: Vec<i16> = (0..n)
        .map(|i| {
            let t = i as f64 / sr;
            let s = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.5;
            (s * 32768.0) as i16
        })
        .collect();
    // Run the encoder once and count total bytes of all emitted packets.
    let mut reg = CodecRegistry::new();
    oxideav_vorbis::register(&mut reg);
    let mut params = CodecParameters::audio(CodecId::new("vorbis"));
    params.channels = Some(1);
    params.sample_rate = Some(48_000);
    params.sample_format = Some(SampleFormat::S16);
    let mut enc = reg.make_encoder(&params).expect("encoder");
    enc.send_frame(&make_s16_frame(1, n, &pcm)).expect("send");
    enc.flush().expect("flush");
    let mut total_bytes = 0usize;
    let mut n_packets = 0usize;
    loop {
        match enc.receive_packet() {
            Ok(p) => {
                total_bytes += p.data.len();
                n_packets += 1;
            }
            Err(Error::Eof) | Err(Error::NeedMore) => break,
            Err(e) => panic!("encoder error: {e}"),
        }
    }
    // Single-book baseline residue cost per long-block mono packet:
    // 512 partitions × 8 bits = 4096 bits = 512 bytes.
    let baseline_residue_only = n_packets * 512;
    eprintln!(
        "mono 1 kHz cascade total={total_bytes} bytes baseline-residue-only={baseline_residue_only} bytes ({n_packets} packets)"
    );
    assert!(
        total_bytes * 2 < baseline_residue_only,
        "cascade not saving enough bits: {total_bytes} vs baseline-residue-only {baseline_residue_only}"
    );
}

/// 7.1 channel round-trip via the public API.
#[test]
fn sine_7_1_roundtrips_via_public_api() {
    let n = 2048 * 4;
    let sr = 48_000.0;
    let freqs = [440.0f64, 520.0, 660.0, 770.0, 990.0, 880.0, 1100.0, 140.0];
    let mut pcm: Vec<i16> = Vec::with_capacity(n * 8);
    for i in 0..n {
        let t = i as f64 / sr;
        for &f in &freqs {
            let s = (2.0 * std::f64::consts::PI * f * t).sin() * 0.35;
            pcm.push((s * 32768.0) as i16);
        }
    }
    let out = encode_decode(8, n, &pcm);
    assert!(!out.is_empty());
    let mut per_ch: Vec<Vec<i16>> = vec![Vec::new(); 8];
    for chunk in out.chunks_exact(8) {
        for (ch, &s) in chunk.iter().enumerate() {
            per_ch[ch].push(s);
        }
    }
    for (ch, &f) in freqs.iter().enumerate() {
        let t = goertzel_mag(&per_ch[ch], f, sr);
        let o = goertzel_mag(&per_ch[ch], 4200.0, sr);
        assert!(
            t > o,
            "7.1 ch{ch}: target {f} ({t}) must beat 4.2 kHz off-band ({o})"
        );
    }
}

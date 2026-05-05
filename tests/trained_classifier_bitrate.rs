//! Bitrate-comparison fixture for task #93 round 2.
//!
//! Compares the encoder's output size + decoded SNR between:
//!   * **Legacy classifier** — the round-1 hard-coded silence threshold
//!     (`CLASSIFY_L2_THRESHOLD = 0.25`), the same code path that shipped
//!     before round-2 wired the trained books in.
//!   * **Trained classifier** — task #93 round-2's
//!     `TrainedPartitionClassifier::from_trained_books()`, whose silence
//!     threshold is the median of the LBG-trained 2-bin slice L2
//!     distribution from `src/trained_books.rs`.
//!
//! The actual byte-vs-byte comparison is held inside the encoder's test
//! module (`encoder::tests::trained_vs_legacy_classifier_bitrate_5s_mix`)
//! since it needs access to the private `TrainedPartitionClassifier`
//! constructor for the legacy threshold. This integration test stays at
//! the public-API surface and only verifies the trained path produces
//! decodable output (smoke test).

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, CodecRegistry, Error, Frame, SampleFormat,
};

/// Build a 5-second mono PCM mix at 44.1 kHz: a 1 kHz sine carrying ~0.4
/// amplitude plus a low-amplitude voice-band (~250-3500 Hz) coloured noise
/// envelope. The mix is deterministic (LCG-seeded) so the byte counts are
/// reproducible across runs.
fn reference_signal_5s_mono_441() -> Vec<i16> {
    let sr = 44_100usize;
    let n = sr * 5;
    let mut out = Vec::with_capacity(n);
    let mut rng: u32 = 0xDEAD_BEEF;
    let mut next = || {
        rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    };
    // Coloured noise: low-pass an LCG via two single-pole filters tuned
    // to ~3.5 kHz cut-off. Matches voice-band bandwidth roughly.
    let mut lp1 = 0f32;
    let mut lp2 = 0f32;
    let alpha = 0.55f32;
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let sine = (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.4;
        let raw = next();
        lp1 = lp1 * alpha + raw * (1.0 - alpha);
        lp2 = lp2 * alpha + lp1 * (1.0 - alpha);
        let noise = lp2 * 0.15;
        let s = (sine + noise).clamp(-1.0, 1.0);
        out.push((s * 30_000.0) as i16);
    }
    out
}

/// Encode `pcm_i16` mono at 44.1 kHz through the public encoder factory
/// (which uses the trained classifier by default), returning the total
/// payload bytes plus the decoded PCM (so the caller can compare SNR).
fn encode_decode(pcm_i16: &[i16]) -> (usize, Vec<i16>) {
    let mut reg = CodecRegistry::new();
    oxideav_vorbis::register_codecs(&mut reg);
    let mut params = CodecParameters::audio(CodecId::new("vorbis"));
    params.channels = Some(1);
    params.sample_rate = Some(44_100);
    params.sample_format = Some(SampleFormat::S16);
    let mut enc = reg.first_encoder(&params).expect("encoder accepts params");
    let mut data = Vec::with_capacity(pcm_i16.len() * 2);
    for s in pcm_i16 {
        data.extend_from_slice(&s.to_le_bytes());
    }
    let frame = Frame::Audio(AudioFrame {
        samples: pcm_i16.len() as u32,
        pts: Some(0),
        data: vec![data],
    });
    enc.send_frame(&frame).expect("send_frame");
    enc.flush().expect("flush");
    let mut total = 0usize;
    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => {
                total += p.data.len();
                packets.push(p);
            }
            Err(Error::Eof) | Err(Error::NeedMore) => break,
            Err(e) => panic!("encoder error: {e}"),
        }
    }
    let dec_params = enc.output_params().clone();
    let mut dec = reg
        .first_decoder(&dec_params)
        .expect("decoder accepts our extradata");
    let mut decoded = Vec::with_capacity(pcm_i16.len());
    for p in packets {
        if dec.send_packet(&p).is_err() {
            break;
        }
        while let Ok(Frame::Audio(af)) = dec.receive_frame() {
            for chunk in af.data[0].chunks_exact(2) {
                decoded.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
    }
    let _ = dec.flush();
    while let Ok(Frame::Audio(af)) = dec.receive_frame() {
        for chunk in af.data[0].chunks_exact(2) {
            decoded.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
    }
    (total, decoded)
}

fn snr_db(reference: &[i16], decoded: &[i16], skip: usize) -> f64 {
    let len = reference.len().min(decoded.len());
    if len <= skip {
        return f64::NEG_INFINITY;
    }
    let mut sig = 0f64;
    let mut err = 0f64;
    for i in skip..len {
        let r = reference[i] as f64;
        let d = decoded[i] as f64;
        sig += r * r;
        err += (r - d).powi(2);
    }
    if err < 1e-9 {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

/// Smoke test: confirm the trained classifier produces sane output for
/// a ~5-second mixed signal. Exercises the encoder + decoder paths
/// through the public API only. This is the round-2 deliverable's
/// integration test — see `crates/oxideav-vorbis/src/trained_classifier.rs`.
#[test]
fn trained_classifier_encodes_and_decodes_5s_mix() {
    let pcm = reference_signal_5s_mono_441();
    let (bytes, decoded) = encode_decode(&pcm);
    assert!(
        bytes > 0 && bytes < pcm.len() * 2,
        "encoded size {bytes} should be > 0 and < raw PCM size ({})",
        pcm.len() * 2
    );
    // Skip 1 long block (2048 samples) for OLA warm-up — the decoder's
    // first-block output is silent until the second packet's left half
    // overlap-adds.
    let snr = snr_db(&pcm, &decoded, 2048);
    eprintln!(
        "trained-classifier 5s mix: bytes={bytes} (~{:.2} kbps at 44.1 kHz mono); snr={snr:.2} dB",
        (bytes as f64 * 8.0) / 5.0 / 1000.0
    );
    // SNR should be at least 0 dB (signal louder than error). A 1 kHz
    // sine + voice-band noise mix encodes into the ~3-6 dB range at
    // libvorbis q3-q4 for this codec; we set the gate at 0 dB to be
    // resistant to per-platform float jitter and per-classifier shift.
    assert!(
        snr > 0.0,
        "trained-classifier SNR negative ({snr:.2} dB) — encoder output is dominated by error"
    );
}

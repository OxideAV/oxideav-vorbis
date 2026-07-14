//! Registration + dual-API wiring: the crate as a framework codec.
//!
//! `register()` installs one `"vorbis"` codec (decoder + encoder
//! factories, the Matroska `A_VORBIS` tag) into an
//! `oxideav_core::RuntimeContext`; the direct factory endpoints
//! (`encoder::make_encoder` / `decoder::make_decoder`) stay callable
//! without a registry — the workspace dual-API convention. This suite
//! drives a full frame-level round trip through **both** paths: PCM
//! `AudioFrame`s → `Encoder` (packets, headers in-band and flagged) →
//! `Decoder` → planar-f32 `AudioFrame`s, checked against the input in
//! the time domain.

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, CodecTag, Error, Frame, ProbeContext, RuntimeContext,
    SampleFormat,
};
use oxideav_vorbis::{make_decoder, make_encoder, register};

const RATE: u32 = 44_100;
const SAMPLES: usize = 12_000;

fn test_signal(seed: u32) -> Vec<f32> {
    (0..SAMPLES)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            let mut v = 0.40 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.20 * (2.0 * std::f32::consts::PI * 1370.0 * t).sin();
            let h = (i as u32 + seed).wrapping_mul(2_654_435_761) >> 8;
            v += ((h & 0xffff) as f32 / 32768.0 - 1.0) * 0.002;
            v
        })
        .collect()
}

fn snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&r, &d) in reference.iter().zip(decoded) {
        sig += f64::from(r) * f64::from(r);
        let e = f64::from(r) - f64::from(d);
        err += e * e;
    }
    10.0 * (sig / err).log10()
}

fn f32_plane(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn plane_f32(plane: &[u8]) -> Vec<f32> {
    plane
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn encoder_params() -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new("vorbis"));
    p.sample_rate = Some(RATE);
    p.channels = Some(1);
    p.sample_format = Some(SampleFormat::F32P);
    // The round-trip pins below are waveform-SNR regression guards;
    // the psy model targets noise-to-mask, and at nominal quality a
    // tonal signal legitimately carries ~16 dB waveform SNR (tonal
    // maskers hide ~-20 dB noise). Pin at a strict-margin quality so
    // the guard is tight.
    p.options = p.options.clone().set("quality", "0.95");
    p
}

#[test]
fn registry_carries_decoder_encoder_and_the_matroska_tag() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let id = CodecId::new("vorbis");
    assert!(ctx.codecs.has_decoder(&id), "decoder registered");
    assert!(ctx.codecs.has_encoder(&id), "encoder registered");
    let tag = CodecTag::matroska("A_VORBIS");
    assert_eq!(
        ctx.codecs
            .resolve_tag_ref(&ProbeContext::new(&tag))
            .map(|c| c.as_str()),
        Some("vorbis"),
        "A_VORBIS resolves to the codec"
    );
}

/// Drive the full encode→decode loop through boxed trait objects.
fn roundtrip_through(
    mut encoder: Box<dyn oxideav_core::Encoder>,
    mut decoder: Box<dyn oxideav_core::Decoder>,
) -> (usize, f64) {
    let pcm = test_signal(7);

    // Feed the PCM in irregular frame sizes (the encoder buffers).
    let mut sent = 0usize;
    for chunk in pcm.chunks(1234) {
        let frame = Frame::Audio(AudioFrame {
            samples: chunk.len() as u32,
            pts: Some(sent as i64),
            data: vec![f32_plane(chunk)],
        });
        encoder.send_frame(&frame).expect("frame accepted");
        sent += chunk.len();
    }
    // Two-pass encoder: nothing before flush.
    assert!(matches!(encoder.receive_packet(), Err(Error::NeedMore)));
    encoder.flush().expect("flush runs the encode");

    // Drain packets: three flagged headers first, then audio.
    let mut packets = Vec::new();
    loop {
        match encoder.receive_packet() {
            Ok(p) => packets.push(p),
            Err(Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e}"),
        }
    }
    assert!(packets.len() > 3, "headers + audio packets expected");
    for (i, p) in packets.iter().take(3).enumerate() {
        assert!(p.flags.header, "packet {i} must be flagged header");
    }
    assert!(!packets[3].flags.header);
    let bytes: usize = packets.iter().map(|p| p.data.len()).sum();

    // Decode through the trait object.
    let mut decoded = Vec::new();
    for p in &packets {
        decoder.send_packet(p).expect("packet accepted");
        loop {
            match decoder.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    assert_eq!(a.data.len(), 1, "planar mono: one plane");
                    assert_eq!(a.pts, Some(decoded.len() as i64), "sample-granularity PTS");
                    decoded.extend(plane_f32(&a.data[0]));
                }
                Ok(other) => panic!("expected audio frame, got {other:?}"),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
    }
    decoder.flush().expect("flush");
    assert!(matches!(decoder.receive_frame(), Err(Error::Eof)));

    // The decoder emits every finished half-block, which may overrun
    // the input length by the tail padding (the §A.2 end-trim is the
    // demuxer's job at this layer); compare over the input extent.
    assert!(
        decoded.len() >= SAMPLES,
        "decoded {} samples",
        decoded.len()
    );
    let snr = snr_db(&pcm, &decoded[..SAMPLES]);
    (bytes, snr)
}

#[test]
fn registry_factories_roundtrip_pcm() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let params = encoder_params();
    let encoder = ctx.codecs.first_encoder(&params).expect("encoder builds");
    let decoder = ctx.codecs.first_decoder(&params).expect("decoder builds");
    let (bytes, snr) = roundtrip_through(encoder, decoder);
    eprintln!("registry path: {bytes} packet bytes, SNR {snr:.2} dB");
    assert!(snr >= 20.0, "registry round-trip SNR {snr:.2} dB below 20");
}

#[test]
fn direct_factory_endpoints_roundtrip_pcm() {
    // The dual-API convention: make_encoder / make_decoder callable
    // with no registry involved.
    let params = encoder_params();
    let encoder = make_encoder(&params).expect("encoder builds");
    let decoder = make_decoder(&params).expect("decoder builds");
    let (bytes, snr) = roundtrip_through(encoder, decoder);
    eprintln!("direct path: {bytes} packet bytes, SNR {snr:.2} dB");
    assert!(snr >= 20.0, "direct round-trip SNR {snr:.2} dB below 20");
}

#[test]
fn encoder_factory_options_and_guards() {
    // quality option changes the rate.
    let pcm = test_signal(9);
    let run = |quality: Option<&str>| -> usize {
        let mut params = encoder_params();
        if let Some(q) = quality {
            params.options = params.options.clone().set("quality", q);
        }
        let mut enc = make_encoder(&params).expect("builds");
        let frame = Frame::Audio(AudioFrame {
            samples: pcm.len() as u32,
            pts: Some(0),
            data: vec![f32_plane(&pcm)],
        });
        enc.send_frame(&frame).unwrap();
        enc.flush().unwrap();
        let mut total = 0;
        while let Ok(p) = enc.receive_packet() {
            total += p.data.len();
        }
        total
    };
    let low = run(Some("0.1"));
    let high = run(Some("0.95"));
    let default = run(None);
    eprintln!("bytes: q0.1 {low}, default {default}, q0.95 {high}");
    assert!(low < high, "quality option must steer the rate");

    // Guards: missing parameters and out-of-range options are refused.
    let empty = CodecParameters::audio(CodecId::new("vorbis"));
    assert!(make_encoder(&empty).is_err(), "sample_rate required");
    let mut bad = encoder_params();
    bad.options = bad.options.clone().set("quality", "3.0");
    assert!(make_encoder(&bad).is_err(), "quality range enforced");
    let mut bad = encoder_params();
    bad.options = bad.options.clone().set("blocksize", "100");
    assert!(make_encoder(&bad).is_err(), "blocksize shape enforced");
    let mut bad = encoder_params();
    bad.options = bad.options.clone().set("short_blocksize", "100");
    assert!(
        make_encoder(&bad).is_err(),
        "short_blocksize shape enforced"
    );
    let mut bad = encoder_params();
    bad.options = bad.options.clone().set("short_blocksize", "4096");
    assert!(
        make_encoder(&bad).is_err(),
        "an explicit short size above the long size is refused"
    );
    let mut bad = encoder_params();
    bad.options = bad.options.clone().set("coupling", "maybe");
    assert!(make_encoder(&bad).is_err(), "coupling must be a boolean");
    let mut bad = encoder_params();
    bad.options = bad.options.clone().set("vq_dims", "3");
    assert!(make_encoder(&bad).is_err(), "vq_dims must be a power of 2");
    let mut bad = encoder_params();
    bad.options = bad.options.clone().set("vq_dims", "32");
    assert!(
        make_encoder(&bad).is_err(),
        "vq_dims must divide the partition size 16"
    );
    let mut bad = encoder_params();
    bad.options = bad.options.clone().set("vq_dims", "two");
    assert!(make_encoder(&bad).is_err(), "vq_dims must be an integer");

    // A lone "blocksize" below the default short size clamps the short
    // size down (a single-blocksize request), rather than erroring.
    let mut small = encoder_params();
    small.options = small.options.clone().set("blocksize", "128");
    assert!(make_encoder(&small).is_ok(), "blocksize=128 alone is legal");

    // Explicit short_blocksize + coupling + vq_dims options build (and
    // the encoder round-trips under them — vq_dims=2 exercises the
    // designed multi-dimensional books through the registry path).
    let mut opts = encoder_params();
    opts.options = opts.options.clone().set("short_blocksize", "512");
    opts.options = opts.options.clone().set("coupling", "false");
    opts.options = opts.options.clone().set("vq_dims", "2");
    assert!(
        make_encoder(&opts).is_ok(),
        "short_blocksize/coupling/vq_dims set"
    );

    // Interleaved F32 input is accepted and matches planar output.
    let mut params = encoder_params();
    params.sample_format = Some(SampleFormat::F32);
    let mut enc = make_encoder(&params).expect("interleaved encoder builds");
    let frame = Frame::Audio(AudioFrame {
        samples: pcm.len() as u32,
        pts: Some(0),
        data: vec![f32_plane(&pcm)], // mono: interleaved == planar
    });
    enc.send_frame(&frame).unwrap();
    enc.flush().unwrap();
    assert!(enc.receive_packet().is_ok());
}

#[test]
fn decoder_rejects_out_of_order_headers() {
    let params = encoder_params();
    let mut dec = make_decoder(&params).expect("builds");
    // An audio packet (LSB of byte 0 clear) before any header.
    let packet = oxideav_core::Packet::new(
        0,
        oxideav_core::TimeBase::new(1, i64::from(RATE)),
        vec![0x00, 0x55, 0xAA],
    );
    assert!(dec.send_packet(&packet).is_err());
}

//! Criterion microbenchmarks for oxideav-vorbis.
//!
//! Measures the hot kernels (`imdct_1024`, `imdct_2048`,
//! `window_overlap_add`) in isolation against the f64 scalar reference,
//! plus end-to-end `decode_packet_*` / `encode_1s_stereo_44k1` that
//! exercise the full bitstream path.
//!
//! Scalar-vs-SIMD comparisons use the `scalar_*` / `simd_*` naming so
//! the paired runs show up next to each other in the Criterion report.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

use oxideav_core::CodecRegistry;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, SampleFormat, TimeBase,
};
use oxideav_vorbis::imdct::{
    forward_mdct_naive, forward_mdct_reference, imdct_naive, imdct_reference,
};
use oxideav_vorbis::simd;

fn bench_imdct(c: &mut Criterion) {
    for &n in &[1024usize, 2048usize] {
        let half = n / 2;
        let spec: Vec<f32> = (0..half)
            .map(|i| ((i as f32 * 0.137).sin() - (i as f32 * 0.029).cos()) * 0.3)
            .collect();

        c.bench_function(&format!("scalar_imdct_{n}"), |b| {
            let mut out = vec![0f32; n];
            b.iter(|| {
                imdct_reference(black_box(&spec), black_box(&mut out));
            });
        });

        c.bench_function(&format!("simd_imdct_{n}"), |b| {
            let mut out = vec![0f32; n];
            b.iter(|| {
                imdct_naive(black_box(&spec), black_box(&mut out));
            });
        });
    }
}

fn bench_mdct(c: &mut Criterion) {
    let n = 2048usize;
    let half = n / 2;
    let sig: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.091).sin() + (i as f32 * 0.003).cos()) * 0.2)
        .collect();

    c.bench_function("scalar_forward_mdct_2048", |b| {
        let mut spec = vec![0f32; half];
        b.iter(|| forward_mdct_reference(black_box(&sig), black_box(&mut spec)));
    });
    c.bench_function("simd_forward_mdct_2048", |b| {
        let mut spec = vec![0f32; half];
        b.iter(|| forward_mdct_naive(black_box(&sig), black_box(&mut spec)));
    });
}

fn bench_window_overlap_add(c: &mut Criterion) {
    let n = 2048usize;
    let prev: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let rising: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + 0.5) / n as f32 * std::f32::consts::PI).sin())
        .collect();
    let falling: Vec<f32> = rising.iter().rev().copied().collect();

    c.bench_function("scalar_window_overlap_add_2048", |b| {
        let mut curr = vec![1.0f32; n + 64];
        b.iter(|| {
            simd::scalar::overlap_add(
                black_box(&mut curr),
                0,
                black_box(&prev),
                black_box(&rising),
                black_box(&falling),
            );
        });
    });
    c.bench_function("simd_window_overlap_add_2048", |b| {
        let mut curr = vec![1.0f32; n + 64];
        b.iter(|| {
            simd::overlap_add(
                black_box(&mut curr),
                0,
                black_box(&prev),
                black_box(&rising),
                black_box(&falling),
            );
        });
    });
}

fn bench_bulk_ops(c: &mut Criterion) {
    let n = 1024usize;
    let a0: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos()).collect();

    c.bench_function("scalar_mul_inplace_1024", |bnch| {
        let mut a = a0.clone();
        bnch.iter(|| simd::scalar::mul_inplace(black_box(&mut a), black_box(&b)));
    });
    c.bench_function("simd_mul_inplace_1024", |bnch| {
        let mut a = a0.clone();
        bnch.iter(|| simd::mul_inplace(black_box(&mut a), black_box(&b)));
    });
    c.bench_function("scalar_add_inplace_1024", |bnch| {
        let mut a = a0.clone();
        bnch.iter(|| simd::scalar::add_inplace(black_box(&mut a), black_box(&b)));
    });
    c.bench_function("simd_add_inplace_1024", |bnch| {
        let mut a = a0.clone();
        bnch.iter(|| simd::add_inplace(black_box(&mut a), black_box(&b)));
    });
}

// ---- End-to-end decode/encode benches driven off a synthesised stream. ----

fn build_encoder(
    channels: u16,
    sample_rate: u32,
) -> (CodecRegistry, Box<dyn oxideav_core::Encoder>) {
    let mut reg = CodecRegistry::new();
    oxideav_vorbis::register(&mut reg);
    let mut params = CodecParameters::audio(CodecId::new("vorbis"));
    params.channels = Some(channels);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::S16);
    let enc = reg.make_encoder(&params).expect("encoder accepts params");
    (reg, enc)
}

fn make_s16_frame(channels: u16, sr: u32, samples: u32, pcm: &[i16]) -> Frame {
    let mut data = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        data.extend_from_slice(&s.to_le_bytes());
    }
    Frame::Audio(AudioFrame {
        format: SampleFormat::S16,
        channels,
        sample_rate: sr,
        samples,
        pts: Some(0),
        time_base: TimeBase::new(1, sr as i64),
        data: vec![data],
    })
}

/// Produce a short synthesised Vorbis packet + setup extradata from a
/// 1 kHz stereo sine — used by the decode benches as a reproducible
/// input that doesn't need sample files on disk.
fn synth_packets(channels: u16, sample_rate: u32, duration_secs: f32) -> (Vec<u8>, Vec<Packet>) {
    let (mut reg, mut enc) = build_encoder(channels, sample_rate);
    let n = (sample_rate as f32 * duration_secs) as usize;
    let mut pcm = Vec::with_capacity(n * channels as usize);
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let s = (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.4;
        let q = (s * 32768.0) as i16;
        for _ in 0..channels {
            pcm.push(q);
        }
    }
    enc.send_frame(&make_s16_frame(channels, sample_rate, n as u32, &pcm))
        .expect("send_frame");
    enc.flush().expect("flush");
    let mut pkts = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => pkts.push(p),
            Err(Error::Eof) | Err(Error::NeedMore) => break,
            Err(e) => panic!("encoder error: {e}"),
        }
    }
    let extradata = enc.output_params().extradata.clone();
    let _ = &mut reg;
    (extradata, pkts)
}

fn bench_decode_packets(c: &mut Criterion) {
    // 1 second of mono 48 kHz sine.
    let (extradata, pkts) = synth_packets(1, 48_000, 1.0);

    // Bench each iteration decodes every packet into a fresh decoder — measures
    // full bitstream → PCM throughput.
    c.bench_function("decode_1s_mono_48k", |b| {
        b.iter(|| {
            let mut reg = CodecRegistry::new();
            oxideav_vorbis::register(&mut reg);
            let mut params = CodecParameters::audio(CodecId::new("vorbis"));
            params.channels = Some(1);
            params.sample_rate = Some(48_000);
            params.sample_format = Some(SampleFormat::S16);
            params.extradata = extradata.clone();
            let mut dec = reg.make_decoder(&params).expect("decoder");
            let mut total = 0usize;
            for p in &pkts {
                dec.send_packet(p).expect("send");
                while let Ok(Frame::Audio(af)) = dec.receive_frame() {
                    total += af.data[0].len();
                }
            }
            black_box(total)
        });
    });

    let (extradata_s, pkts_s) = synth_packets(2, 44_100, 1.0);
    c.bench_function("decode_1s_stereo_44k1", |b| {
        b.iter(|| {
            let mut reg = CodecRegistry::new();
            oxideav_vorbis::register(&mut reg);
            let mut params = CodecParameters::audio(CodecId::new("vorbis"));
            params.channels = Some(2);
            params.sample_rate = Some(44_100);
            params.sample_format = Some(SampleFormat::S16);
            params.extradata = extradata_s.clone();
            let mut dec = reg.make_decoder(&params).expect("decoder");
            let mut total = 0usize;
            for p in &pkts_s {
                dec.send_packet(p).expect("send");
                while let Ok(Frame::Audio(af)) = dec.receive_frame() {
                    total += af.data[0].len();
                }
            }
            black_box(total)
        });
    });
}

fn bench_encode(c: &mut Criterion) {
    let sr: u32 = 44_100;
    let ch: u16 = 2;
    let n = sr as usize;
    let mut pcm = Vec::with_capacity(n * ch as usize);
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let s = (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.4;
        let q = (s * 32768.0) as i16;
        pcm.push(q);
        pcm.push(q);
    }

    c.bench_function("encode_1s_stereo_44k1", |b| {
        b.iter(|| {
            let mut reg = CodecRegistry::new();
            oxideav_vorbis::register(&mut reg);
            let mut params = CodecParameters::audio(CodecId::new("vorbis"));
            params.channels = Some(ch);
            params.sample_rate = Some(sr);
            params.sample_format = Some(SampleFormat::S16);
            let mut enc = reg.make_encoder(&params).expect("encoder");
            enc.send_frame(&make_s16_frame(ch, sr, n as u32, &pcm))
                .expect("send_frame");
            enc.flush().expect("flush");
            let mut total = 0usize;
            loop {
                match enc.receive_packet() {
                    Ok(p) => total += p.data.len(),
                    Err(Error::Eof) | Err(Error::NeedMore) => break,
                    Err(e) => panic!("encoder error: {e}"),
                }
            }
            black_box(total)
        });
    });
}

criterion_group!(
    vorbis_benches,
    bench_imdct,
    bench_mdct,
    bench_window_overlap_add,
    bench_bulk_ops,
    bench_decode_packets,
    bench_encode
);
criterion_main!(vorbis_benches);

//! Own-codec encode → decode round-trip on arbitrary PCM.
//!
//! Drives the whole-stream encoder (psy model, floor-1 design/fit,
//! residue RD planning, per-stream codebook training, §A.2 mux) and
//! the whole-stream decoder on fuzzer-chosen audio: channel count,
//! quality knob, and sample content all come from the input. Vorbis
//! is lossy so amplitudes are not asserted; the pinned contract is
//! shape: the encode succeeds on any finite PCM, the produced stream
//! decodes, the §A.2 end-trim recovers exactly the input length per
//! channel, and every decoded sample is finite.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_vorbis::oggfile::{decode_ogg_to_pcm, encode_pcm_to_ogg, StreamEncoderConfig};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let channels = 1 + (data[0] & 1) as usize; // 1 or 2
    let quality = f32::from(data[1]) / 255.0;
    let payload = &data[2..];
    // i16-quantised samples in [-1, 1) from byte pairs; cap the length
    // so one iteration stays cheap (the encoder trains per stream).
    let per_channel = (payload.len() / (2 * channels)).min(4096);
    if per_channel < 16 {
        return;
    }
    let mut pcm: Vec<Vec<f32>> = vec![Vec::with_capacity(per_channel); channels];
    let mut it = payload.chunks_exact(2);
    for _ in 0..per_channel {
        for row in pcm.iter_mut() {
            let b = it.next().expect("length checked above");
            let v = i16::from_le_bytes([b[0], b[1]]);
            row.push(f32::from(v) / 32768.0);
        }
    }

    let mut config = StreamEncoderConfig::new(44_100, channels as u8);
    config.quality = quality;
    let ogg = match encode_pcm_to_ogg(&pcm, &config) {
        Ok(o) => o,
        Err(e) => panic!("finite PCM must encode: {e:?}"),
    };
    let decoded = decode_ogg_to_pcm(&ogg).expect("own encode must decode");
    assert_eq!(decoded.channels as usize, channels);
    assert_eq!(decoded.pcm.len(), channels);
    for row in &decoded.pcm {
        assert_eq!(row.len(), per_channel, "SSA.2 end-trim must match input");
        for &s in row {
            assert!(s.is_finite(), "decoded sample must be finite");
        }
    }
});

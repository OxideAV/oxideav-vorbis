//! End-to-end **container → registry → PCM** wiring: a real `.ogg`
//! file opened through the `oxideav-ogg` demuxer, the codec resolved
//! by the registry (the §4.2.1 `\x01vorbis` payload magic `register()`
//! declares — Ogg has no codec tag at all), the decoder built through
//! the registered factory from the demuxer's own `StreamInfo`
//! parameters (whose Xiph-laced extradata carries the three §4.2
//! headers the demuxer consumed at open time), and the demuxed packet
//! feed decoded to PCM — checked sample-exact (±1 s16) against the
//! fixture's black-box `expected.wav` reference dump.
//!
//! This is the pipeline shape an application uses: nothing below is a
//! crate-internal path. Every hop is public API of `oxideav-core`
//! (registry + traits), `oxideav-ogg` (demuxer) or this crate
//! (`register`). The per-packet / per-stage decode internals are
//! pinned elsewhere (`fixture_pcm_decode`, the trace-conformance
//! suites); this suite pins the *wiring* — that an Ogg-carried Vorbis
//! stream auto-resolves and decodes with no Vorbis-specific glue on
//! the caller's side.
//!
//! # Standalone-CI skip
//!
//! The fixture-driven tests skip when the umbrella `docs/` submodule
//! is absent (per-crate standalone CI clones only this repo); the
//! encoder-fed round-trip below is self-contained and always runs.

use oxideav_core::{CodecResolver, Error, Frame, ReadSeek, RuntimeContext};
use oxideav_vorbis::register;

/// Root of the staged Vorbis fixtures (umbrella `docs/` submodule).
fn fixtures_root() -> String {
    format!(
        "{}/../../docs/audio/vorbis/fixtures",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn fixtures_available() -> bool {
    std::path::Path::new(&fixtures_root()).is_dir()
}

/// Read the s16le samples (interleaved) from the `data` chunk of a WAV.
fn wav_s16(data: &[u8]) -> Vec<i16> {
    let mut pos = 12; // skip RIFF + size + WAVE
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let sz = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
            as usize;
        let start = pos + 8;
        if id == b"data" {
            let end = (start + sz).min(data.len());
            return data[start..end]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
        }
        pos = start + sz + (sz & 1); // chunks are word-aligned
    }
    Vec::new()
}

/// Open `ogg` through the container demuxer with the registry as the
/// codec resolver, build the decoder through the registered factory
/// from the demuxer's `StreamInfo`, decode every demuxed packet, and
/// return the per-channel PCM rows (bitstream channel order).
///
/// Asserts the wiring invariants along the way: the codec id resolves
/// to `"vorbis"`, the demuxer republished the consumed headers as
/// non-empty extradata, and no header packet reaches the packet feed.
fn decode_via_registry(name: &str, ogg: Vec<u8>) -> Vec<Vec<f32>> {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    // The registry must claim the stream from its payload magic alone
    // — the demuxer hands it the BOS packet's leading bytes.
    let input: Box<dyn ReadSeek> = Box::new(std::io::Cursor::new(ogg));
    let mut dmx = oxideav_ogg::demux::open(input, &ctx.codecs).expect("demuxer opens");
    assert_eq!(dmx.streams().len(), 1, "{name}: one logical stream");
    let stream = dmx.streams()[0].clone();
    assert_eq!(
        stream.params.codec_id.as_str(),
        "vorbis",
        "{name}: codec resolves through the registry payload magic"
    );
    assert!(
        !stream.params.extradata.is_empty(),
        "{name}: demuxer republishes the consumed headers as extradata"
    );
    let ch = stream.params.channels.expect("channel count") as usize;

    // The registered factory builds the decoder from the demuxer's own
    // parameters — the extradata pre-configures the §4.2 headers.
    let mut dec = ctx
        .codecs
        .first_decoder(&stream.params)
        .expect("registry decoder builds");

    let mut per_ch: Vec<Vec<f32>> = vec![Vec::new(); ch];
    loop {
        let pkt = match dmx.next_packet() {
            Ok(p) => p,
            Err(Error::Eof) => break,
            Err(e) => panic!("{name}: next_packet: {e}"),
        };
        assert!(
            !pkt.flags.header && pkt.data.first().is_some_and(|b| b & 1 == 0),
            "{name}: only §4.3 audio packets may reach the packet feed"
        );
        dec.send_packet(&pkt).expect("packet accepted");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    assert_eq!(a.data.len(), ch, "{name}: one plane per channel");
                    for (row, plane) in per_ch.iter_mut().zip(&a.data) {
                        row.extend(
                            plane
                                .chunks_exact(4)
                                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                        );
                    }
                }
                Ok(other) => panic!("{name}: expected audio frame, got {other:?}"),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("{name}: receive_frame: {e}"),
            }
        }
    }
    dec.flush().expect("flush");
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    per_ch
}

/// Decode a fixture's `input.ogg` through the full container → registry
/// → PCM chain and compare against its `expected.wav` within the
/// documented ±1 s16 tolerance. `wav_from_bitstream[w]` is the §4.3.9
/// permutation mapping WAV interleave slot `w` to its bitstream
/// channel (identity for mono / stereo).
fn assert_registry_decode_matches_reference(dir: &str, wav_from_bitstream: &[usize]) {
    let base = format!("{}/{dir}", fixtures_root());
    let ogg = std::fs::read(format!("{base}/input.ogg")).expect("fixture input.ogg present");
    let wav = std::fs::read(format!("{base}/expected.wav")).expect("expected.wav present");
    let expected = wav_s16(&wav);
    assert!(!expected.is_empty(), "{dir}: expected.wav has no samples");

    let per_ch = decode_via_registry(dir, ogg);
    let ch = per_ch.len();
    assert_eq!(ch, wav_from_bitstream.len(), "{dir}: channel count");
    let frames = per_ch[0].len();
    assert!(
        frames * ch >= expected.len(),
        "{dir}: decoded {} samples < expected {}",
        frames * ch,
        expected.len()
    );

    let exp_frames = expected.len() / ch;
    let mut max_diff = 0i32;
    let mut mismatches = 0usize;
    for f in 0..exp_frames {
        for (wav_slot, &bitstream_ch) in wav_from_bitstream.iter().enumerate() {
            let dec = (per_ch[bitstream_ch][f] * 32768.0)
                .round()
                .clamp(-32768.0, 32767.0) as i32;
            let exp = expected[f * ch + wav_slot] as i32;
            let diff = (dec - exp).abs();
            max_diff = max_diff.max(diff);
            if diff > 1 {
                mismatches += 1;
            }
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{dir}: {mismatches}/{} samples exceed ±1 s16 (max diff {max_diff}, ch={ch})",
        expected.len()
    );
}

/// Every staged single-logical-stream fixture, with its §4.3.9 WAV
/// permutation. The chained two-stream fixture is exercised by
/// `chained_stream_decode` at the packet layer; the demuxer models a
/// chained link as a stream restart, which is a container-level story
/// this wiring suite doesn't re-pin.
fn single_stream_fixtures() -> Vec<(&'static str, Vec<usize>)> {
    let id1: Vec<usize> = vec![0];
    let id2: Vec<usize> = vec![0, 1];
    vec![
        ("mono-22050-low-rate", id1.clone()),
        ("mono-44100-q5-typical", id1.clone()),
        ("stereo-44100-q5-typical", id2.clone()),
        ("stereo-44100-q10", id2.clone()),
        ("stereo-44100-q-1", id2.clone()),
        ("stereo-96000-high-rate", id2.clone()),
        ("stereo-cbr-128kbps", id2.clone()),
        ("mode-residue-types-0-1-2", id2.clone()),
        ("noise-stream", id1.clone()),
        ("silence-stream", id1.clone()),
        ("mode-floor1-only", id1.clone()),
        ("transient-blocksize-switch", id1.clone()),
        ("with-vorbis-comment-tags", id1.clone()),
        ("with-attached-picture", id1),
        // 5.1: decoder emits §4.3.2 bitstream order; the WAV
        // interleaves [FL, FR, FC, LFE, BL, BR] (see
        // fixture_pcm_decode for the §4.3.9 derivation).
        ("5.1-channel-48000-q5", vec![0, 2, 1, 5, 3, 4]),
    ]
}

#[test]
fn every_staged_fixture_decodes_through_demux_and_registry() {
    if !fixtures_available() {
        eprintln!("SKIP: docs/ fixtures submodule not checked out (standalone CI)");
        return;
    }
    for (dir, perm) in single_stream_fixtures() {
        assert_registry_decode_matches_reference(dir, &perm);
        eprintln!("{dir}: demux → registry → PCM matches expected.wav");
    }
}

#[test]
fn encoder_output_decodes_through_demux_and_registry() {
    // Self-contained (no fixtures): the crate's own encode(pcm) → .ogg
    // through the same container → registry → PCM chain, closing the
    // loop both ways. Runs on standalone CI too.
    const RATE: u32 = 44_100;
    let pcm: Vec<f32> = (0..3 * 2048)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            0.45 * (2.0 * std::f32::consts::PI * 523.25 * t).sin()
                + 0.15 * (2.0 * std::f32::consts::PI * 2093.0 * t).sin()
        })
        .collect();
    let config = oxideav_vorbis::oggfile::StreamEncoderConfig::new(RATE, 1);
    let ogg = oxideav_vorbis::oggfile::encode_pcm_to_ogg(std::slice::from_ref(&pcm), &config)
        .expect("encode succeeds");

    let per_ch = decode_via_registry("encoder-output", ogg);
    assert_eq!(per_ch.len(), 1);
    assert!(
        per_ch[0].len() >= pcm.len(),
        "decoded {} < input {}",
        per_ch[0].len(),
        pcm.len()
    );
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (i, &x) in pcm.iter().enumerate() {
        sig += f64::from(x) * f64::from(x);
        let e = f64::from(x) - f64::from(per_ch[0][i]);
        err += e * e;
    }
    let snr = 10.0 * (sig / err).log10();
    eprintln!("encoder-output: demux → registry SNR {snr:.2} dB");
    assert!(snr >= 20.0, "round-trip SNR {snr:.2} dB below 20");
}

#[test]
fn registry_resolution_is_load_bearing_for_the_demuxer_fallbackless_path() {
    // Sanity for the resolution hop itself: with an *empty* registry
    // the demuxer still opens (its built-in magic table is the
    // documented fallback), but the registry then has no decoder — so
    // the auto-resolution the wired codec provides is exactly what
    // turns "identified" into "decodable".
    if !fixtures_available() {
        eprintln!("SKIP: docs/ fixtures submodule not checked out (standalone CI)");
        return;
    }
    let base = format!("{}/mono-44100-q5-typical", fixtures_root());
    let ogg = std::fs::read(format!("{base}/input.ogg")).expect("fixture input.ogg present");

    let empty = RuntimeContext::new();
    assert!(
        empty.codecs.resolve_payload_magic(b"\x01vorbis").is_none(),
        "empty registry claims nothing"
    );
    let input: Box<dyn ReadSeek> = Box::new(std::io::Cursor::new(ogg));
    let dmx = oxideav_ogg::demux::open(input, &empty.codecs).expect("demuxer opens");
    let params = dmx.streams()[0].params.clone();
    assert!(
        empty.codecs.first_decoder(&params).is_err(),
        "no registered codec: identified but not decodable"
    );

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    assert!(
        ctx.codecs.first_decoder(&params).is_ok(),
        "register() is what makes the demuxed stream decodable"
    );
}

//! Chained-Ogg Vorbis decode → PCM, end-to-end across a logical-bitstream
//! boundary (RFC 3533 §5 chaining).
//!
//! # What this pins
//!
//! `tests/fixture_pcm_decode.rs` drives the §4.3 decode chain over twelve
//! single-bitstream fixtures. The one staged corpus member it explicitly
//! does **not** consume is `chained-streams`: the byte-level concatenation
//! of two independent Ogg/Vorbis streams, each with its own `BOS`/`EOS`
//! page-flag pair and its own `bitstream_serial` cookie
//! (`docs/audio/vorbis/fixtures/chained-streams/notes.md`).
//!
//! Chained-Ogg handling is a real decode capability: at a stream boundary
//! the §4.3.8 overlap-add tail of the previous logical bitstream is no
//! longer valid (the next stream re-primes from its own first packet), and
//! each stream carries its own three Vorbis headers — a fresh
//! identification + setup the decoder must re-parse before its audio
//! packets can decode. This test exercises both:
//!
//! 1. **First stream → `expected.wav`.** The fixture's `expected.wav` is
//!    the decode of the *first* logical bitstream only (per `notes.md`:
//!    "Decoders that only consume the first logical bitstream will produce
//!    `expected.wav`"). We decode stream 0 through the public
//!    [`StreamingDecoder`] path and match it sample-exact within the
//!    documented ±1 s16 lossy tolerance — confirming the chained de-framing
//!    isolated stream 0's packets correctly (no cross-boundary packet
//!    bleed).
//!
//! 2. **Second stream decodes too.** A chained-aware decoder additionally
//!    decodes the second bitstream. We re-parse stream 1's headers, build a
//!    fresh decoder, and confirm it emits a comparable run of PCM —
//!    validating the per-stream reset + re-parse + decode cycle end to end.
//!
//! # Chained-aware de-framing
//!
//! The single-bitstream walker in `fixture_pcm_decode.rs` coalesces lacing
//! segments without tracking the page serial, so a packet continued across
//! a page (`lacing == 255`) would silently bleed across a stream boundary.
//! This test carries a *serial-aware* RFC-3533 page walker (private test
//! scaffolding, never compiled into `src/` — Ogg framing is container work
//! per `docs/IMPLEMENTOR_ROUND.md`): it groups packets per
//! `bitstream_serial`, and a `BOS` page (`header_type & 0x02`) starts a
//! fresh logical bitstream, discarding any dangling continuation from the
//! previous one (continuations never span streams).
//!
//! # Standalone-CI skip
//!
//! Like the other fixture tests, the corpus lives in the umbrella `docs/`
//! submodule, absent in standalone per-crate CI. When the fixtures
//! directory is missing each test prints a skip notice and returns — this
//! is data availability, not a disabled test (no `#[ignore]`).

use oxideav_vorbis::{
    parse_identification_header, parse_setup_header, AudioDecoderState, StreamingDecoder,
    StreamingFrame,
};

/// The pinned §4.3.7 IMDCT normalization scalar (see
/// `tests/fixture_pcm_decode.rs`): the bare cosine-summation kernel plus
/// the §4.3.6 window and §4.3.8 overlap-add need no extra scaling.
const IMDCT_SCALE: f32 = 1.0;

fn fixtures_root() -> String {
    format!(
        "{}/../../docs/audio/vorbis/fixtures",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn fixtures_available() -> bool {
    std::path::Path::new(&fixtures_root()).is_dir()
}

/// One logical Ogg bitstream: its serial cookie and the coalesced packets
/// belonging to it, in stream order.
struct LogicalStream {
    serial: u32,
    packets: Vec<Vec<u8>>,
}

/// Serial-aware RFC-3533 page de-framer (test scaffolding only).
///
/// Walks pages sequentially. Each page belongs to the logical bitstream
/// named by its `bitstream_serial` (bytes 14..18). A page whose
/// `header_type` (byte 5) has bit `0x02` set is a `BOS` page: it opens a
/// fresh logical bitstream. Lacing segments coalesce into packets exactly
/// as in the single-stream walker, but a packet continuation
/// (`lacing == 255` at a page's final segment) is tracked **per serial**,
/// so a dangling continuation from one stream never coalesces into the
/// next.
fn ogg_logical_streams(data: &[u8]) -> Vec<LogicalStream> {
    let mut streams: Vec<LogicalStream> = Vec::new();
    // Per-stream pending continuation buffer, keyed by index into `streams`.
    let mut pending: std::collections::HashMap<u32, Vec<u8>> = std::collections::HashMap::new();
    let mut pos = 0usize;
    while pos + 27 <= data.len() {
        assert_eq!(
            &data[pos..pos + 4],
            b"OggS",
            "Ogg page sync lost at byte {pos}"
        );
        let header_type = data[pos + 5];
        let serial = u32::from_le_bytes([
            data[pos + 14],
            data[pos + 15],
            data[pos + 16],
            data[pos + 17],
        ]);
        let seg_count = data[pos + 26] as usize;
        let seg_table_start = pos + 27;
        let body_start = seg_table_start + seg_count;
        if body_start > data.len() {
            break;
        }

        // A BOS page opens a fresh logical bitstream. Any dangling
        // continuation for this serial from a prior stream chaining is
        // discarded — continuations never span a stream boundary.
        let is_bos = header_type & 0x02 != 0;
        if is_bos {
            pending.remove(&serial);
            streams.push(LogicalStream {
                serial,
                packets: Vec::new(),
            });
        }

        // Locate the active stream slot for this serial (the most recent
        // one opened with this serial).
        let slot = streams
            .iter()
            .rposition(|s| s.serial == serial)
            .expect("page references a serial with no preceding BOS page");

        let seg_table = &data[seg_table_start..body_start];
        let mut body = body_start;
        let mut acc = pending.remove(&serial).unwrap_or_default();
        for &lace in seg_table {
            let l = lace as usize;
            if body + l > data.len() {
                break;
            }
            acc.extend_from_slice(&data[body..body + l]);
            body += l;
            if l < 255 {
                streams[slot].packets.push(std::mem::take(&mut acc));
            }
        }
        if !acc.is_empty() {
            pending.insert(serial, acc);
        }
        pos = body;
    }
    // Flush any trailing continuation per serial.
    for (serial, acc) in pending {
        if let Some(s) = streams.iter_mut().rev().find(|s| s.serial == serial) {
            if !acc.is_empty() {
                s.packets.push(acc);
            }
        }
    }
    streams
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
        pos = start + sz + (sz & 1);
    }
    Vec::new()
}

/// Decode one logical bitstream's packets (header + audio) to per-channel
/// f32 PCM rows through the full public `StreamingDecoder` path.
fn decode_logical(packets: &[Vec<u8>]) -> Vec<Vec<f32>> {
    assert!(
        packets.len() >= 4,
        "expected id + comment + setup + audio packets, got {}",
        packets.len()
    );
    let id = parse_identification_header(&packets[0]).expect("identification header parses");
    let setup = parse_setup_header(&packets[2], id.audio_channels).expect("setup header parses");
    let state = AudioDecoderState::new(&setup).expect("decoder state builds");
    let ch = id.audio_channels as usize;

    let mut dec = StreamingDecoder::new(
        id.audio_channels,
        id.blocksize_0 as usize,
        id.blocksize_1 as usize,
        IMDCT_SCALE,
    );

    let mut per_ch: Vec<Vec<f32>> = vec![Vec::new(); ch];
    for pkt in &packets[3..] {
        let mut r = oxideav_core::bits::BitReaderLsb::new(pkt);
        match dec.push_packet(&mut r, &setup, &state) {
            Ok(StreamingFrame::Pcm {
                per_channel_pcm, ..
            }) => {
                for (row, samples) in per_ch.iter_mut().zip(&per_channel_pcm) {
                    row.extend_from_slice(samples);
                }
            }
            Ok(StreamingFrame::Primed { .. }) => {}
            Err(e) => panic!("decode failed: {e}"),
        }
    }
    per_ch
}

/// Count how many of a logical stream's first three packets parse as the
/// Vorbis identification / comment / setup headers (a well-formed Vorbis
/// logical bitstream carries exactly these three before any audio packet).
fn vorbis_header_packets_present(packets: &[Vec<u8>]) -> bool {
    if packets.len() < 3 {
        return false;
    }
    // Vorbis common header layout: packet type byte (1/3/5) + "vorbis".
    let is_header = |p: &[u8], ptype: u8| p.len() >= 7 && p[0] == ptype && &p[1..7] == b"vorbis";
    is_header(&packets[0], 1) && is_header(&packets[1], 3) && is_header(&packets[2], 5)
}

#[test]
fn chained_streams_split_into_two_logical_bitstreams() {
    if !fixtures_available() {
        eprintln!("SKIP chained-streams: docs/ fixtures submodule not checked out");
        return;
    }
    let ogg =
        std::fs::read(format!("{}/chained-streams/input.ogg", fixtures_root())).expect("input.ogg");
    let streams = ogg_logical_streams(&ogg);

    // The fixture is two independent Ogg/Vorbis streams concatenated.
    assert_eq!(
        streams.len(),
        2,
        "chained-streams must split into exactly two logical bitstreams"
    );
    // Distinct serial cookies (the chaining marker per RFC 3533 §5).
    assert_ne!(
        streams[0].serial, streams[1].serial,
        "chained logical bitstreams carry distinct bitstream_serial values"
    );
    // Each stream carries its own three Vorbis headers before audio.
    for (i, s) in streams.iter().enumerate() {
        assert!(
            vorbis_header_packets_present(&s.packets),
            "stream {i} (serial {:#010x}) must carry id+comment+setup headers",
            s.serial
        );
    }
}

#[test]
fn chained_first_stream_decodes_sample_exact() {
    if !fixtures_available() {
        eprintln!("SKIP chained-streams: docs/ fixtures submodule not checked out");
        return;
    }
    let base = format!("{}/chained-streams", fixtures_root());
    let ogg = std::fs::read(format!("{base}/input.ogg")).expect("input.ogg");
    let wav = std::fs::read(format!("{base}/expected.wav")).expect("expected.wav");
    let expected = wav_s16(&wav);
    assert!(!expected.is_empty(), "expected.wav has no samples");

    let streams = ogg_logical_streams(&ogg);
    assert!(!streams.is_empty(), "at least one logical bitstream");

    // expected.wav is the first logical bitstream only (mono).
    let per_ch = decode_logical(&streams[0].packets);
    assert_eq!(per_ch.len(), 1, "first stream is mono");
    let decoded = &per_ch[0];

    assert!(
        decoded.len() >= expected.len(),
        "decoded {} samples < expected {}",
        decoded.len(),
        expected.len()
    );

    // Sample-exact within the fixture's ±1 s16 lossy tolerance, comparing
    // the reference's length from frame 0 (alignment offset is 0 for the
    // staged fixtures; trailing samples past the reference are the
    // encoder's granule end-trim).
    let mut max_diff = 0i32;
    let mut mismatches = 0usize;
    for (i, &exp) in expected.iter().enumerate() {
        let dec = (decoded[i] * 32768.0).round().clamp(-32768.0, 32767.0) as i32;
        let diff = (dec - exp as i32).abs();
        max_diff = max_diff.max(diff);
        if diff > 1 {
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{mismatches}/{} samples exceed ±1 s16 (max diff {max_diff})",
        expected.len()
    );
}

#[test]
fn chained_second_stream_decodes_independently() {
    if !fixtures_available() {
        eprintln!("SKIP chained-streams: docs/ fixtures submodule not checked out");
        return;
    }
    let base = format!("{}/chained-streams", fixtures_root());
    let ogg = std::fs::read(format!("{base}/input.ogg")).expect("input.ogg");
    let streams = ogg_logical_streams(&ogg);
    assert_eq!(streams.len(), 2, "two logical bitstreams");

    // The second stream re-parses its own headers and decodes through a
    // fresh decoder — the chained-Ogg reset + re-parse + decode cycle.
    let per_ch = decode_logical(&streams[1].packets);
    assert_eq!(per_ch.len(), 1, "second stream is mono");
    let decoded = &per_ch[0];

    // The fixture's two streams are both 0.4 s @ 44.1 kHz (sine 440 then
    // 880), so the second stream's decoded length is in the same ballpark
    // as the first's expected.wav (17512 samples). We don't have a second
    // expected.wav, so assert a substantial non-empty decode rather than
    // a sample-exact match: this confirms the second logical bitstream's
    // headers re-parsed and its audio packets decoded end to end.
    assert!(
        decoded.len() > 8_000,
        "second stream decoded only {} samples; expected a full ~0.4 s clip",
        decoded.len()
    );
    // The two streams differ (440 Hz vs 880 Hz tone), so the second
    // stream's samples must not be all-zero (a silent decode would signal
    // a header/state mix-up across the boundary).
    let nonzero = decoded.iter().any(|&s| s.abs() > 1.0 / 32768.0);
    assert!(
        nonzero,
        "second stream decoded to all-silence — header re-parse likely failed"
    );
}

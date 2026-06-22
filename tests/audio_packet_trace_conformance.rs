//! Per-packet §4.3.1 header-decision conformance against the staged
//! `docs/audio/vorbis/fixtures/*/trace.txt` reference event streams.
//!
//! # What this pins
//!
//! `tests/fixture_pcm_decode.rs` validates the **end** of the decode
//! chain — the final PCM bytes — against each fixture's `expected.wav`.
//! It does not validate the per-packet *structural decisions* the decoder
//! makes on the way there. Every fixture's `notes.md` is explicit that
//! those decisions are themselves load-bearing reference data:
//!
//! > The `trace.txt` event sequence is the load-bearing reference: every
//! > conformant decoder must see the same Ogg page boundaries, the same
//! > setup-header counts, and the same per-packet `mode_number` /
//! > `prev_window` / `next_window` selections.
//!
//! This suite consumes the `AUDIO_PACKET` lines of each trace as opaque
//! ground-truth records and drives every fixture's audio packets through
//! the public §4.3.1 header parser
//! ([`oxideav_vorbis::read_packet_header`]), asserting **line-for-line**
//! that the parser recovers the trace's `mode_number`, `blockflag`,
//! `prev_window` (`previous_window_flag`), `next_window`
//! (`next_window_flag`) and `block_size` (`n`) for every audio packet the
//! trace logs — 505 audio-packet decisions across the 16 staged fixtures.
//! It is a pure header-decision oracle: no floor / residue / IMDCT runs, so
//! a regression in the §4.3.1 prelude (mode-bit width, the short-vs-long
//! window-flag gating, or the blocksize resolution) is caught in isolation,
//! distinct from the PCM-level test where a header bug would surface only
//! as a garbled-audio mismatch.
//!
//! # Trace `packet_idx` is the bitstream index, not a row counter
//!
//! For the longer fixtures the reference decoder logs the first 32 audio
//! packets (`packet_idx` 0..=31) contiguously and then the **final**
//! packet at its true bitstream index (e.g. 44 / 51 / 55) — the trace is
//! sampled to stay compact while still pinning the end-trim packet. Each
//! record is therefore matched against the de-framed audio packet at
//! `audio[packet_idx]`, not the i-th trace row, so the gap is honoured and
//! the last packet (the one that exercises the granule-position end-trim
//! geometry) is validated too. Short fixtures (≤ 25 packets) log every
//! packet contiguously.
//!
//! The trace is a black-box validator *output* (it logs structural events,
//! not implementation internals): consumed here as ground-truth bytes, no
//! reference *source* is read.
//!
//! # Short-block window flags
//!
//! §4.3.1 step 4b transmits **no** window-flag bits for a short block
//! (`blockflag = 0`); the symmetric short window needs none. The parser
//! reports `previous_window_flag = next_window_flag = false` (placeholders)
//! in that case, and across every staged fixture the trace logs
//! `prev_window=0 next_window=0` for every short packet — so the
//! line-for-line equality holds for short and long packets alike.
//!
//! # Standalone-CI skip
//!
//! The fixtures live in the umbrella `docs/` submodule, absent in the
//! per-crate standalone CI clone. When the directory is missing each test
//! prints a skip notice and returns (data availability, not a disabled
//! test — there is no `#[ignore]`).

use oxideav_vorbis::{parse_identification_header, parse_setup_header, read_packet_header};

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

/// Serial-aware RFC-3533 page de-framer (test scaffolding only — Ogg
/// framing is container work, see `tests/fixture_pcm_decode.rs`). Each
/// page belongs to the logical bitstream named by its `bitstream_serial`
/// (bytes 14..18); a `header_type` (byte 5) with bit `0x02` set opens a
/// fresh logical bitstream. Lacing segments coalesce into packets, with a
/// page-final `lacing == 255` continuation tracked **per serial** so a
/// dangling continuation never coalesces across a stream boundary. This
/// matches the de-framer in `tests/chained_stream_decode.rs`, so the
/// chained fixture's two streams are walked independently.
fn ogg_logical_streams(data: &[u8]) -> Vec<LogicalStream> {
    let mut streams: Vec<LogicalStream> = Vec::new();
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

        let is_bos = header_type & 0x02 != 0;
        if is_bos {
            pending.remove(&serial);
            streams.push(LogicalStream {
                serial,
                packets: Vec::new(),
            });
        }

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
    for (serial, acc) in pending {
        if let Some(s) = streams.iter_mut().rev().find(|s| s.serial == serial) {
            if !acc.is_empty() {
                s.packets.push(acc);
            }
        }
    }
    streams
}

/// One `AUDIO_PACKET` trace record — the §4.3.1 header decisions the
/// reference decoder logged for this packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TraceAudioPacket {
    stream_idx: u32,
    packet_idx: u32,
    mode_number: u32,
    blockflag: bool,
    prev_window: bool,
    next_window: bool,
    block_size: usize,
    packet_bytes: usize,
}

/// Pull the `key=value` token for `key` out of a tab/space-split trace
/// line. Trace values are bare tokens (no embedded whitespace) for every
/// field this suite consumes.
fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.split_whitespace()
        .find_map(|tok| tok.strip_prefix(key)?.strip_prefix('='))
}

fn parse_u32(line: &str, key: &str) -> u32 {
    field(line, key)
        .unwrap_or_else(|| panic!("trace line missing {key}: {line}"))
        .parse()
        .unwrap_or_else(|_| panic!("trace {key} not a u32: {line}"))
}

/// Parse every `AUDIO_PACKET` line of a fixture's `trace.txt`, in file
/// order, into structured records.
fn parse_trace_audio_packets(trace: &str) -> Vec<TraceAudioPacket> {
    trace
        .lines()
        .filter(|l| l.starts_with("AUDIO_PACKET"))
        .map(|l| TraceAudioPacket {
            stream_idx: parse_u32(l, "stream_idx"),
            packet_idx: parse_u32(l, "packet_idx"),
            mode_number: parse_u32(l, "mode_number"),
            blockflag: parse_u32(l, "blockflag") != 0,
            prev_window: parse_u32(l, "prev_window") != 0,
            next_window: parse_u32(l, "next_window") != 0,
            block_size: parse_u32(l, "block_size") as usize,
            packet_bytes: parse_u32(l, "packet_bytes") as usize,
        })
        .collect()
}

/// Drive one fixture's audio packets through the §4.3.1 header parser and
/// assert each parsed header matches the trace's `AUDIO_PACKET` record for
/// that packet, line-for-line. Returns the number of audio packets
/// validated (so the caller can pin a per-fixture lower bound).
fn assert_fixture_trace_conformant(dir: &str) -> usize {
    if !fixtures_available() {
        eprintln!("SKIP {dir}: docs/ fixtures submodule not checked out (standalone CI)");
        return 0;
    }
    let base = format!("{}/{}", fixtures_root(), dir);
    let ogg = std::fs::read(format!("{base}/input.ogg")).expect("fixture input.ogg present");
    let trace = std::fs::read_to_string(format!("{base}/trace.txt")).expect("trace.txt present");

    let streams = ogg_logical_streams(&ogg);
    assert!(
        !streams.is_empty(),
        "{dir}: de-framer found no logical streams"
    );

    // Group the trace's AUDIO_PACKET records by stream_idx, preserving the
    // BOS order of the de-framed streams (trace stream_idx 0 == first BOS).
    let all_records = parse_trace_audio_packets(&trace);
    assert!(
        !all_records.is_empty(),
        "{dir}: trace.txt has no AUDIO_PACKET lines"
    );

    let mut total = 0usize;
    for (stream_idx, stream) in streams.iter().enumerate() {
        let records: Vec<TraceAudioPacket> = all_records
            .iter()
            .copied()
            .filter(|r| r.stream_idx as usize == stream_idx)
            .collect();
        if records.is_empty() {
            continue;
        }

        // Each Vorbis logical stream begins with exactly three header
        // packets (id / comment / setup, §4.2.1); audio packets follow.
        assert!(
            stream.packets.len() >= 4,
            "{dir} stream {stream_idx}: expected 3 headers + audio, got {}",
            stream.packets.len()
        );
        let id = parse_identification_header(&stream.packets[0])
            .unwrap_or_else(|e| panic!("{dir} stream {stream_idx}: id header: {e}"));
        let setup = parse_setup_header(&stream.packets[2], id.audio_channels)
            .unwrap_or_else(|e| panic!("{dir} stream {stream_idx}: setup header: {e}"));
        let bs0 = id.blocksize_0 as usize;
        let bs1 = id.blocksize_1 as usize;

        let audio = &stream.packets[3..];

        for rec in &records {
            // The trace's packet_idx is the true bitstream audio-packet
            // index (the trace samples the middle of long streams; see the
            // module header), so index into the de-framed audio packets by
            // packet_idx, not by trace-row position.
            let idx = rec.packet_idx as usize;
            assert!(
                idx < audio.len(),
                "{dir} stream {stream_idx}: trace packet_idx {idx} >= de-framed audio packets {}",
                audio.len()
            );
            let pkt = &audio[idx];
            // The de-framer reconstructs the exact packet body the encoder
            // emitted; its length must equal the trace's logged
            // packet_bytes (an independent cross-check on the framing).
            assert_eq!(
                pkt.len(),
                rec.packet_bytes,
                "{dir} stream {stream_idx} packet {idx}: de-framed body {} bytes != trace {} bytes",
                pkt.len(),
                rec.packet_bytes
            );

            let mut r = oxideav_core::bits::BitReaderLsb::new(pkt);
            let header = read_packet_header(&mut r, &setup, bs0, bs1).unwrap_or_else(|e| {
                panic!("{dir} stream {stream_idx} packet {idx}: header parse failed: {e}")
            });

            assert_eq!(
                header.mode_number, rec.mode_number,
                "{dir} stream {stream_idx} packet {idx}: mode_number {} != trace {}",
                header.mode_number, rec.mode_number
            );
            assert_eq!(
                header.blockflag, rec.blockflag,
                "{dir} stream {stream_idx} packet {idx}: blockflag {} != trace {}",
                header.blockflag, rec.blockflag
            );
            assert_eq!(
                header.n, rec.block_size,
                "{dir} stream {stream_idx} packet {idx}: block_size {} != trace {}",
                header.n, rec.block_size
            );
            assert_eq!(
                header.previous_window_flag, rec.prev_window,
                "{dir} stream {stream_idx} packet {idx}: prev_window {} != trace {}",
                header.previous_window_flag, rec.prev_window
            );
            assert_eq!(
                header.next_window_flag, rec.next_window,
                "{dir} stream {stream_idx} packet {idx}: next_window {} != trace {}",
                header.next_window_flag, rec.next_window
            );
            // §4.3.1 step 3 self-consistency: blockflag must resolve n to
            // exactly one of the two declared blocksizes.
            let expected_n = if header.blockflag { bs1 } else { bs0 };
            assert_eq!(
                header.n, expected_n,
                "{dir} stream {stream_idx} packet {idx}: n {} not the blockflag-selected blocksize {expected_n}",
                header.n
            );
            total += 1;
        }
    }
    total
}

/// Every staged single-logical-stream fixture, asserted line-for-line
/// against its trace, plus a per-fixture audio-packet-count lower bound so
/// a regression that silently decodes fewer packets is caught.
#[test]
fn single_stream_fixtures_match_trace_audio_packet_decisions() {
    if !fixtures_available() {
        eprintln!("SKIP: docs/ fixtures submodule not checked out (standalone CI)");
        return;
    }
    // (fixture dir, expected audio-packet count from its trace.txt).
    let cases: &[(&str, usize)] = &[
        ("mono-22050-low-rate", 33),
        ("mono-44100-q5-typical", 33),
        ("stereo-44100-q5-typical", 33),
        ("stereo-44100-q10", 33),
        ("stereo-44100-q-1", 33),
        ("stereo-96000-high-rate", 33),
        ("stereo-cbr-128kbps", 33),
        ("noise-stream", 33),
        ("silence-stream", 33),
        ("mode-floor1-only", 33),
        ("mode-residue-types-0-1-2", 33),
        ("transient-blocksize-switch", 33),
        ("with-vorbis-comment-tags", 23),
        ("with-attached-picture", 23),
        ("5.1-channel-48000-q5", 25),
    ];
    let mut grand_total = 0usize;
    for (dir, expected) in cases {
        let n = assert_fixture_trace_conformant(dir);
        assert_eq!(
            n, *expected,
            "{dir}: validated {n} audio packets, trace lists {expected}"
        );
        grand_total += n;
    }
    // Sanity floor on the corpus-wide coverage: the 15 single-stream
    // fixtures carry 467 audio-packet decisions.
    assert_eq!(
        grand_total, 467,
        "single-stream corpus total changed: {grand_total}"
    );
}

/// The chained-streams fixture carries two concatenated logical bitstreams
/// (RFC 3533 §5). The serial-aware de-framer separates them, and each
/// stream's audio packets are validated against the trace records carrying
/// that stream's `stream_idx` — proving the §4.3.1 header decode is
/// per-stream-independent across the chaining boundary.
#[test]
fn chained_stream_fixture_matches_trace_per_logical_stream() {
    if !fixtures_available() {
        eprintln!("SKIP: docs/ fixtures submodule not checked out (standalone CI)");
        return;
    }
    let n = assert_fixture_trace_conformant("chained-streams");
    assert_eq!(
        n, 38,
        "chained-streams: validated {n} audio packets across both logical streams, trace lists 38"
    );
}

/// The whole staged corpus carries 505 `AUDIO_PACKET` decisions; this pins
/// the aggregate so a fixture dropped from coverage is loud.
#[test]
fn corpus_wide_audio_packet_decision_total() {
    if !fixtures_available() {
        eprintln!("SKIP: docs/ fixtures submodule not checked out (standalone CI)");
        return;
    }
    let dirs = [
        "mono-22050-low-rate",
        "mono-44100-q5-typical",
        "stereo-44100-q5-typical",
        "stereo-44100-q10",
        "stereo-44100-q-1",
        "stereo-96000-high-rate",
        "stereo-cbr-128kbps",
        "noise-stream",
        "silence-stream",
        "mode-floor1-only",
        "mode-residue-types-0-1-2",
        "transient-blocksize-switch",
        "with-vorbis-comment-tags",
        "with-attached-picture",
        "5.1-channel-48000-q5",
        "chained-streams",
    ];
    let total: usize = dirs
        .iter()
        .map(|d| assert_fixture_trace_conformant(d))
        .sum();
    assert_eq!(
        total, 505,
        "corpus-wide AUDIO_PACKET total changed: {total}"
    );
}

//! §4.3.8 overlap-add output-geometry conformance over the full staged
//! fixture decode, plus a cross-check of the streaming path's per-packet
//! header reports against `trace.txt`.
//!
//! # What this pins
//!
//! `tests/fixture_pcm_decode.rs` validates the decoded PCM *values*;
//! `tests/audio_packet_trace_conformance.rs` validates the §4.3.1 header
//! decisions through the isolated `read_packet_header` parser. This suite
//! pins the piece between them: the §4.3.8 **windowing / overlap-add into
//! PCM** *geometry* as it actually runs inside
//! [`oxideav_vorbis::StreamingDecoder`].
//!
//! Per §4.3.8 the PCM finished for the previous → current packet transition
//! is `prev_n / 4 + cur_n / 4` samples per channel, where `prev_n` / `cur_n`
//! are the two packets' blocksizes (`n`). The first packet primes (no PCM).
//! Driving every staged fixture's whole audio stream through the public
//! `push_packet` path, this suite asserts, for **every** emitted frame:
//!
//! * the priming step lands on (and only on) the first packet;
//! * each subsequent frame's per-channel PCM length equals
//!   `prev_n / 4 + cur_n / 4`, with `prev_n` taken from the previous
//!   packet's own reported `n` — so the overlap-add length contract is
//!   checked across **all** packets of every fixture, including the ones the
//!   trace does not log (the trace samples the middle of long streams);
//! * every channel of a frame has identical length;
//! * the streaming path reports the same `mode_number` / `blockflag` /
//!   `block_size` the trace logged, for each packet the trace records
//!   (indexed by the trace's true bitstream `packet_idx`).
//!
//! It is a geometry + dispatch oracle: it does not re-check the sample
//! values (the PCM test owns that), so a regression in the §4.3.8 lap
//! length, the priming handoff, or the per-packet mode dispatch is isolated
//! from a numeric IMDCT/floor/residue regression.
//!
//! The trace is a black-box validator *output* (structural events only); no
//! reference *source* is read.
//!
//! # Standalone-CI skip
//!
//! Fixtures live in the umbrella `docs/` submodule, absent in the per-crate
//! standalone clone; each test then skips (data availability, not a disabled
//! test — there is no `#[ignore]`).

use oxideav_vorbis::{
    parse_identification_header, parse_setup_header, AudioDecoderState, StreamingDecoder,
    StreamingFrame,
};

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

/// One logical Ogg bitstream's coalesced packets.
struct LogicalStream {
    packets: Vec<Vec<u8>>,
}

/// Serial-aware RFC-3533 page de-framer (test scaffolding only; Ogg framing
/// is container work). Mirrors the sibling conformance suites.
fn ogg_logical_streams(data: &[u8]) -> Vec<LogicalStream> {
    struct Acc {
        serial: u32,
        packets: Vec<Vec<u8>>,
    }
    let mut streams: Vec<Acc> = Vec::new();
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
        if header_type & 0x02 != 0 {
            pending.remove(&serial);
            streams.push(Acc {
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
        .into_iter()
        .map(|a| LogicalStream { packets: a.packets })
        .collect()
}

/// A `(packet_idx, mode_number, blockflag, block_size)` record parsed from a
/// trace's `AUDIO_PACKET` lines, keyed by stream index.
#[derive(Clone, Copy)]
struct TraceRec {
    stream_idx: usize,
    packet_idx: usize,
    mode_number: u32,
    blockflag: bool,
    block_size: usize,
}

fn u32_tok(line: &str, key: &str) -> u32 {
    line.split_whitespace()
        .find_map(|t| t.strip_prefix(key)?.strip_prefix('='))
        .unwrap_or_else(|| panic!("trace line missing {key}: {line}"))
        .parse()
        .unwrap_or_else(|_| panic!("trace {key} not a u32: {line}"))
}

fn parse_trace(trace: &str) -> Vec<TraceRec> {
    trace
        .lines()
        .filter(|l| l.starts_with("AUDIO_PACKET"))
        .map(|l| TraceRec {
            stream_idx: u32_tok(l, "stream_idx") as usize,
            packet_idx: u32_tok(l, "packet_idx") as usize,
            mode_number: u32_tok(l, "mode_number"),
            blockflag: u32_tok(l, "blockflag") != 0,
            block_size: u32_tok(l, "block_size") as usize,
        })
        .collect()
}

/// Drive one fixture's every logical stream through `StreamingDecoder` and
/// pin the §4.3.8 output geometry + trace dispatch. Returns the total
/// emitted-PCM frame count validated.
fn assert_fixture_geometry(dir: &str) -> usize {
    if !fixtures_available() {
        eprintln!("SKIP {dir}: docs/ fixtures submodule not checked out (standalone CI)");
        return 0;
    }
    let base = format!("{}/{}", fixtures_root(), dir);
    let ogg = std::fs::read(format!("{base}/input.ogg")).expect("fixture input.ogg present");
    let trace = std::fs::read_to_string(format!("{base}/trace.txt")).expect("trace.txt present");
    let recs = parse_trace(&trace);
    let streams = ogg_logical_streams(&ogg);

    let mut total_pcm_frames = 0usize;
    for (stream_idx, stream) in streams.iter().enumerate() {
        assert!(
            stream.packets.len() >= 4,
            "{dir} s{stream_idx}: too few packets"
        );
        let id = parse_identification_header(&stream.packets[0])
            .unwrap_or_else(|e| panic!("{dir} s{stream_idx}: id header: {e}"));
        let setup = parse_setup_header(&stream.packets[2], id.audio_channels)
            .unwrap_or_else(|e| panic!("{dir} s{stream_idx}: setup header: {e}"));
        let state = AudioDecoderState::new(&setup)
            .expect("decoder state builds from a parsed setup header");
        let ch = id.audio_channels as usize;

        // Trace records for this logical stream, by bitstream packet index.
        let stream_recs: std::collections::HashMap<usize, TraceRec> = recs
            .iter()
            .filter(|r| r.stream_idx == stream_idx)
            .map(|r| (r.packet_idx, *r))
            .collect();

        let mut dec = StreamingDecoder::new(
            id.audio_channels,
            id.blocksize_0 as usize,
            id.blocksize_1 as usize,
            IMDCT_SCALE,
        );

        let audio = &stream.packets[3..];
        let mut prev_n: Option<usize> = None;
        for (i, pkt) in audio.iter().enumerate() {
            let mut r = oxideav_core::bits::BitReaderLsb::new(pkt);
            let frame = dec
                .push_packet(&mut r, &setup, &state)
                .unwrap_or_else(|e| panic!("{dir} s{stream_idx} pkt {i}: decode failed: {e}"));
            let cur_n = frame.n();

            // Cross-check the streaming path's header report against the
            // trace, where the trace logs this packet.
            if let Some(rec) = stream_recs.get(&i) {
                let (mode_number, blockflag) = match frame {
                    StreamingFrame::Primed {
                        mode_number,
                        blockflag,
                        ..
                    }
                    | StreamingFrame::Pcm {
                        mode_number,
                        blockflag,
                        ..
                    } => (mode_number, blockflag),
                };
                assert_eq!(
                    mode_number, rec.mode_number,
                    "{dir} s{stream_idx} pkt {i}: streaming mode_number != trace"
                );
                assert_eq!(
                    blockflag, rec.blockflag,
                    "{dir} s{stream_idx} pkt {i}: streaming blockflag != trace"
                );
                assert_eq!(
                    cur_n, rec.block_size,
                    "{dir} s{stream_idx} pkt {i}: streaming block_size != trace"
                );
            }

            match frame {
                StreamingFrame::Primed { .. } => {
                    assert_eq!(
                        i, 0,
                        "{dir} s{stream_idx} pkt {i}: priming step on a non-first packet"
                    );
                    assert!(
                        prev_n.is_none(),
                        "{dir} s{stream_idx} pkt {i}: second priming step"
                    );
                }
                StreamingFrame::Pcm {
                    per_channel_pcm, ..
                } => {
                    assert_ne!(
                        i, 0,
                        "{dir} s{stream_idx}: first packet emitted PCM instead of priming"
                    );
                    let pn = prev_n.expect("a Pcm frame must follow at least one prior packet");
                    let expect_len = pn / 4 + cur_n / 4;
                    assert_eq!(
                        per_channel_pcm.len(),
                        ch,
                        "{dir} s{stream_idx} pkt {i}: channel count {} != {ch}",
                        per_channel_pcm.len()
                    );
                    for (c, row) in per_channel_pcm.iter().enumerate() {
                        assert_eq!(
                            row.len(),
                            expect_len,
                            "{dir} s{stream_idx} pkt {i} ch {c}: §4.3.8 length {} != prev_n/4 + cur_n/4 = {expect_len} (prev_n={pn}, cur_n={cur_n})",
                            row.len()
                        );
                    }
                    total_pcm_frames += 1;
                }
            }
            prev_n = Some(cur_n);
        }
    }
    total_pcm_frames
}

#[test]
fn every_fixture_overlap_add_geometry_follows_spec_4_3_8() {
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
    let total: usize = dirs.iter().map(|d| assert_fixture_geometry(d)).sum();
    // Every stream emits one fewer PCM frame than it has audio packets (the
    // first primes); the 16 staged fixtures (17 logical streams, the chained
    // fixture contributing two) emit 654 PCM frames in total. Pinning the
    // exact count guards against a fixture silently dropping out of the
    // sweep or a stream short-circuiting mid-decode.
    assert_eq!(
        total, 654,
        "corpus-wide §4.3.8 PCM-frame count changed: {total}"
    );
}

//! Setup-/identification-header structural conformance against the staged
//! `docs/audio/vorbis/fixtures/*/trace.txt` reference event streams.
//!
//! # What this pins
//!
//! `tests/fixture_pcm_decode.rs` validates the decoded PCM; the companion
//! `tests/audio_packet_trace_conformance.rs` validates the per-audio-packet
//! §4.3.1 header decisions. Neither validates the **setup-time structural
//! configuration** the headers establish — yet every fixture's `notes.md`
//! lists exactly that among the load-bearing reference data:
//!
//! > every conformant decoder must see the same Ogg page boundaries, the
//! > same setup-header counts, and the same per-packet ... selections.
//!
//! Each trace logs structured `VORBIS_HEADER_ID`, `VORBIS_HEADER_SETUP`,
//! `CODEBOOK`, `FLOOR_CONFIG`, `RESIDUE_CONFIG`, `MAPPING_CONFIG` and
//! `MODE_CONFIG` events. This suite parses each fixture's identification
//! (§4.2.2) and setup (§4.2.4) headers through the public parsers and
//! asserts the resulting structures reproduce, field-for-field, every one
//! of those trace events:
//!
//! * **Identification** — `channels`, `sample_rate`, the three bitrates,
//!   `blocksize_0` / `blocksize_1`.
//! * **Setup counts** — `codebook_count`, `floor_count`, `residue_count`,
//!   `mapping_count`, `mode_count` (the trace's `time_count` is the §4.2.4
//!   placeholder list length).
//! * **Per codebook** — `dimensions`, `entries`, `lookup_type` (0 = None,
//!   1 = Lattice, 2 = Tessellation), and for the VQ lookups the resolved
//!   multiplicand `value_bits` (the §3.2.1 read+1 width the parser stores,
//!   which the trace logs directly) and `sequence_p`.
//! * **Per floor** — type, and for floor 1 `partitions`, `multiplier`
//!   (stored `+ 1`, matching the trace's logged value), `rangebits`, and
//!   `x_list_count` (the trace counts the two implicit `[0]` / `[2^rangebits]`
//!   endpoints, so `parsed_x_list.len() + 2 == trace`).
//! * **Per residue** — type, `begin`, `end`, `partition_size`,
//!   `classifications`, `classbook`.
//! * **Per mapping** — type, `submaps`, `coupling_steps`, and the
//!   `magnitude` / `angle` coupling-channel arrays plus the per-submap
//!   `floor` / `residue` index arrays.
//! * **Per mode** — `blockflag`, `windowtype`, `transformtype`, `mapping`.
//!
//! Together with the audio-packet suite this pins the **entire** structural
//! decode of every staged stream — every setup decision and every
//! per-packet decision — against the reference trace, leaving only the
//! lossy sample values to the ±1-s16 PCM test.
//!
//! The trace is a black-box validator *output* (structural events, not
//! implementation internals); no reference *source* is read.
//!
//! # Standalone-CI skip
//!
//! The fixtures live in the umbrella `docs/` submodule, absent in the
//! per-crate standalone CI clone. When missing each test skips (data
//! availability, not a disabled test — there is no `#[ignore]`).

use oxideav_vorbis::{
    parse_identification_header, parse_setup_header, FloorKind, VorbisIdentificationHeader,
    VorbisSetupHeader, VqLookup,
};

fn fixtures_root() -> String {
    format!(
        "{}/../../docs/audio/vorbis/fixtures",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn fixtures_available() -> bool {
    std::path::Path::new(&fixtures_root()).is_dir()
}

/// One logical Ogg bitstream's coalesced packets, keyed by serial.
struct LogicalStream {
    serial: u32,
    packets: Vec<Vec<u8>>,
}

/// Serial-aware RFC-3533 page de-framer (test scaffolding only; Ogg framing
/// is container work). Mirrors `tests/audio_packet_trace_conformance.rs`.
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
        if header_type & 0x02 != 0 {
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

/// Pull the `key=value` token for `key` out of a whitespace-split trace
/// line, returning the raw value token (may itself contain `[..]` for the
/// array fields, which the array helpers parse further).
fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    // Tokens are tab/space separated; array values like `magnitude=[0]`
    // contain no internal whitespace in these traces, so a plain split
    // recovers each `key=value` token whole.
    line.split_whitespace()
        .find_map(|tok| tok.strip_prefix(key)?.strip_prefix('='))
}

fn u32_field(line: &str, key: &str) -> u32 {
    field(line, key)
        .unwrap_or_else(|| panic!("trace line missing {key}: {line}"))
        .parse()
        .unwrap_or_else(|_| panic!("trace {key} not a u32: {line}"))
}

fn i32_field(line: &str, key: &str) -> i32 {
    field(line, key)
        .unwrap_or_else(|| panic!("trace line missing {key}: {line}"))
        .parse()
        .unwrap_or_else(|_| panic!("trace {key} not an i32: {line}"))
}

/// Parse a `key=[a,b,c]` array field into a `Vec<u32>` (empty for `[]`).
fn u32_array_field(line: &str, key: &str) -> Vec<u32> {
    let raw = field(line, key).unwrap_or_else(|| panic!("trace line missing {key}: {line}"));
    let inner = raw
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or_else(|| panic!("trace {key} not a [..] array: {line}"));
    if inner.is_empty() {
        return Vec::new();
    }
    inner
        .split(',')
        .map(|t| {
            t.trim()
                .parse()
                .unwrap_or_else(|_| panic!("trace {key} element not a u32: {line}"))
        })
        .collect()
}

/// Collect the trace lines whose first token equals `tag`.
fn lines_with<'a>(trace: &'a str, tag: &str) -> Vec<&'a str> {
    trace
        .lines()
        .filter(|l| l.split_whitespace().next() == Some(tag))
        .collect()
}

/// Validate one logical stream's parsed headers against the trace lines
/// carrying its `stream_idx`. Returns the number of structural events
/// checked (counts + per-item events) so callers can pin a lower bound.
fn assert_stream_setup_conformant(
    dir: &str,
    stream_idx: usize,
    id: &VorbisIdentificationHeader,
    setup: &VorbisSetupHeader,
    trace: &str,
) -> usize {
    let mut checked = 0usize;
    let in_stream = |l: &&str| u32_field(l, "stream_idx") as usize == stream_idx;

    // --- Identification ---
    let id_line = lines_with(trace, "VORBIS_HEADER_ID")
        .into_iter()
        .find(in_stream)
        .unwrap_or_else(|| panic!("{dir} s{stream_idx}: no VORBIS_HEADER_ID"));
    assert_eq!(
        id.audio_channels as u32,
        u32_field(id_line, "channels"),
        "{dir} s{stream_idx}: channels"
    );
    assert_eq!(
        id.audio_sample_rate,
        u32_field(id_line, "sample_rate"),
        "{dir} s{stream_idx}: sample_rate"
    );
    assert_eq!(
        id.bitrate_maximum,
        i32_field(id_line, "bitrate_max"),
        "{dir} s{stream_idx}: bitrate_max"
    );
    assert_eq!(
        id.bitrate_nominal,
        i32_field(id_line, "bitrate_nominal"),
        "{dir} s{stream_idx}: bitrate_nominal"
    );
    assert_eq!(
        id.bitrate_minimum,
        i32_field(id_line, "bitrate_min"),
        "{dir} s{stream_idx}: bitrate_min"
    );
    assert_eq!(
        id.blocksize_0 as u32,
        u32_field(id_line, "blocksize_0"),
        "{dir} s{stream_idx}: blocksize_0"
    );
    assert_eq!(
        id.blocksize_1 as u32,
        u32_field(id_line, "blocksize_1"),
        "{dir} s{stream_idx}: blocksize_1"
    );
    checked += 1;

    // --- Setup counts ---
    let setup_line = lines_with(trace, "VORBIS_HEADER_SETUP")
        .into_iter()
        .find(in_stream)
        .unwrap_or_else(|| panic!("{dir} s{stream_idx}: no VORBIS_HEADER_SETUP"));
    assert_eq!(
        setup.codebooks.len() as u32,
        u32_field(setup_line, "codebook_count"),
        "{dir} s{stream_idx}: codebook_count"
    );
    assert_eq!(
        setup.time_placeholders.len() as u32,
        u32_field(setup_line, "time_count"),
        "{dir} s{stream_idx}: time_count"
    );
    assert_eq!(
        setup.floors.len() as u32,
        u32_field(setup_line, "floor_count"),
        "{dir} s{stream_idx}: floor_count"
    );
    assert_eq!(
        setup.residues.len() as u32,
        u32_field(setup_line, "residue_count"),
        "{dir} s{stream_idx}: residue_count"
    );
    assert_eq!(
        setup.mappings.len() as u32,
        u32_field(setup_line, "mapping_count"),
        "{dir} s{stream_idx}: mapping_count"
    );
    assert_eq!(
        setup.modes.len() as u32,
        u32_field(setup_line, "mode_count"),
        "{dir} s{stream_idx}: mode_count"
    );
    checked += 1;

    // --- Per codebook ---
    let cb_lines: Vec<&str> = lines_with(trace, "CODEBOOK")
        .into_iter()
        .filter(in_stream)
        .collect();
    assert_eq!(
        cb_lines.len(),
        setup.codebooks.len(),
        "{dir} s{stream_idx}: CODEBOOK line count != parsed codebooks"
    );
    for (i, (cb, line)) in setup.codebooks.iter().zip(&cb_lines).enumerate() {
        assert_eq!(
            u32_field(line, "book_idx") as usize,
            i,
            "{dir} s{stream_idx} cb{i}: book_idx out of order"
        );
        assert_eq!(
            cb.dimensions as u32,
            u32_field(line, "dimensions"),
            "{dir} s{stream_idx} cb{i}: dimensions"
        );
        assert_eq!(
            cb.entries,
            u32_field(line, "entries"),
            "{dir} s{stream_idx} cb{i}: entries"
        );
        let (lookup_type, vq): (u32, Option<(u8, bool)>) = match &cb.lookup {
            VqLookup::None => (0, None),
            VqLookup::Lattice {
                value_bits,
                sequence_p,
                ..
            } => (1, Some((*value_bits, *sequence_p))),
            VqLookup::Tessellation {
                value_bits,
                sequence_p,
                ..
            } => (2, Some((*value_bits, *sequence_p))),
        };
        assert_eq!(
            lookup_type,
            u32_field(line, "lookup_type"),
            "{dir} s{stream_idx} cb{i}: lookup_type"
        );
        if let Some((value_bits, sequence_p)) = vq {
            // The trace logs the resolved multiplicand bit-width, which is
            // exactly the parser's stored `value_bits` (the §3.2.1 read+1).
            assert_eq!(
                value_bits as u32,
                u32_field(line, "value_bits"),
                "{dir} s{stream_idx} cb{i}: value_bits"
            );
            assert_eq!(
                sequence_p as u32,
                u32_field(line, "sequence_p"),
                "{dir} s{stream_idx} cb{i}: sequence_p"
            );
        }
        checked += 1;
    }

    // --- Per floor ---
    let floor_lines: Vec<&str> = lines_with(trace, "FLOOR_CONFIG")
        .into_iter()
        .filter(in_stream)
        .collect();
    assert_eq!(
        floor_lines.len(),
        setup.floors.len(),
        "{dir} s{stream_idx}: FLOOR_CONFIG line count != parsed floors"
    );
    for (i, (floor, line)) in setup.floors.iter().zip(&floor_lines).enumerate() {
        assert_eq!(
            floor.floor_type as u32,
            u32_field(line, "type"),
            "{dir} s{stream_idx} floor{i}: type"
        );
        if let FloorKind::Type1(f1) = &floor.kind {
            assert_eq!(
                f1.partitions as u32,
                u32_field(line, "partitions"),
                "{dir} s{stream_idx} floor{i}: partitions"
            );
            assert_eq!(
                f1.multiplier as u32,
                u32_field(line, "multiplier"),
                "{dir} s{stream_idx} floor{i}: multiplier"
            );
            assert_eq!(
                f1.rangebits as u32,
                u32_field(line, "rangebits"),
                "{dir} s{stream_idx} floor{i}: rangebits"
            );
            // The trace counts the two implicit [0] / [2^rangebits] posts.
            assert_eq!(
                f1.x_list.len() as u32 + 2,
                u32_field(line, "x_list_count"),
                "{dir} s{stream_idx} floor{i}: x_list_count (+2 implicit posts)"
            );
        }
        checked += 1;
    }

    // --- Per residue ---
    let res_lines: Vec<&str> = lines_with(trace, "RESIDUE_CONFIG")
        .into_iter()
        .filter(in_stream)
        .collect();
    assert_eq!(
        res_lines.len(),
        setup.residues.len(),
        "{dir} s{stream_idx}: RESIDUE_CONFIG line count != parsed residues"
    );
    for (i, (res, line)) in setup.residues.iter().zip(&res_lines).enumerate() {
        assert_eq!(
            res.residue_type as u32,
            u32_field(line, "type"),
            "{dir} s{stream_idx} residue{i}: type"
        );
        assert_eq!(
            res.residue_begin,
            u32_field(line, "begin"),
            "{dir} s{stream_idx} residue{i}: begin"
        );
        assert_eq!(
            res.residue_end,
            u32_field(line, "end"),
            "{dir} s{stream_idx} residue{i}: end"
        );
        assert_eq!(
            res.partition_size,
            u32_field(line, "partition_size"),
            "{dir} s{stream_idx} residue{i}: partition_size"
        );
        assert_eq!(
            res.classifications as u32,
            u32_field(line, "classifications"),
            "{dir} s{stream_idx} residue{i}: classifications"
        );
        assert_eq!(
            res.classbook as u32,
            u32_field(line, "classbook"),
            "{dir} s{stream_idx} residue{i}: classbook"
        );
        checked += 1;
    }

    // --- Per mapping ---
    let map_lines: Vec<&str> = lines_with(trace, "MAPPING_CONFIG")
        .into_iter()
        .filter(in_stream)
        .collect();
    assert_eq!(
        map_lines.len(),
        setup.mappings.len(),
        "{dir} s{stream_idx}: MAPPING_CONFIG line count != parsed mappings"
    );
    for (i, (map, line)) in setup.mappings.iter().zip(&map_lines).enumerate() {
        assert_eq!(
            map.mapping_type as u32,
            u32_field(line, "type"),
            "{dir} s{stream_idx} mapping{i}: type"
        );
        assert_eq!(
            map.submaps as u32,
            u32_field(line, "submaps"),
            "{dir} s{stream_idx} mapping{i}: submaps"
        );
        assert_eq!(
            map.coupling.len() as u32,
            u32_field(line, "coupling_steps"),
            "{dir} s{stream_idx} mapping{i}: coupling_steps"
        );
        let mag: Vec<u32> = map
            .coupling
            .iter()
            .map(|c| c.magnitude_channel as u32)
            .collect();
        let ang: Vec<u32> = map
            .coupling
            .iter()
            .map(|c| c.angle_channel as u32)
            .collect();
        assert_eq!(
            mag,
            u32_array_field(line, "magnitude"),
            "{dir} s{stream_idx} mapping{i}: magnitude channels"
        );
        assert_eq!(
            ang,
            u32_array_field(line, "angle"),
            "{dir} s{stream_idx} mapping{i}: angle channels"
        );
        let floors: Vec<u32> = map.submap_configs.iter().map(|s| s.floor as u32).collect();
        let resids: Vec<u32> = map
            .submap_configs
            .iter()
            .map(|s| s.residue as u32)
            .collect();
        assert_eq!(
            floors,
            u32_array_field(line, "floor"),
            "{dir} s{stream_idx} mapping{i}: submap floor indices"
        );
        assert_eq!(
            resids,
            u32_array_field(line, "residue"),
            "{dir} s{stream_idx} mapping{i}: submap residue indices"
        );
        checked += 1;
    }

    // --- Per mode ---
    let mode_lines: Vec<&str> = lines_with(trace, "MODE_CONFIG")
        .into_iter()
        .filter(in_stream)
        .collect();
    assert_eq!(
        mode_lines.len(),
        setup.modes.len(),
        "{dir} s{stream_idx}: MODE_CONFIG line count != parsed modes"
    );
    for (i, (mode, line)) in setup.modes.iter().zip(&mode_lines).enumerate() {
        assert_eq!(
            mode.blockflag as u32,
            u32_field(line, "blockflag"),
            "{dir} s{stream_idx} mode{i}: blockflag"
        );
        assert_eq!(
            mode.windowtype as u32,
            u32_field(line, "windowtype"),
            "{dir} s{stream_idx} mode{i}: windowtype"
        );
        assert_eq!(
            mode.transformtype as u32,
            u32_field(line, "transformtype"),
            "{dir} s{stream_idx} mode{i}: transformtype"
        );
        assert_eq!(
            mode.mapping as u32,
            u32_field(line, "mapping"),
            "{dir} s{stream_idx} mode{i}: mapping"
        );
        checked += 1;
    }

    checked
}

/// Drive a fixture's header packets through the public parsers and validate
/// every logical stream's structural events against the trace.
fn assert_fixture_setup_conformant(dir: &str) -> usize {
    if !fixtures_available() {
        eprintln!("SKIP {dir}: docs/ fixtures submodule not checked out (standalone CI)");
        return 0;
    }
    let base = format!("{}/{}", fixtures_root(), dir);
    let ogg = std::fs::read(format!("{base}/input.ogg")).expect("fixture input.ogg present");
    let trace = std::fs::read_to_string(format!("{base}/trace.txt")).expect("trace.txt present");
    let streams = ogg_logical_streams(&ogg);

    let mut checked = 0usize;
    for (stream_idx, stream) in streams.iter().enumerate() {
        assert!(
            stream.packets.len() >= 3,
            "{dir} s{stream_idx}: fewer than 3 header packets"
        );
        let id = parse_identification_header(&stream.packets[0])
            .unwrap_or_else(|e| panic!("{dir} s{stream_idx}: id header: {e}"));
        let setup = parse_setup_header(&stream.packets[2], id.audio_channels)
            .unwrap_or_else(|e| panic!("{dir} s{stream_idx}: setup header: {e}"));
        checked += assert_stream_setup_conformant(dir, stream_idx, &id, &setup, &trace);
        let _ = stream.serial;
    }
    checked
}

#[test]
fn every_fixture_setup_header_matches_trace() {
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
        .map(|d| assert_fixture_setup_conformant(d))
        .sum();
    // Every fixture contributes id + setup-count events plus one event per
    // codebook / floor / residue / mapping / mode; the corpus sum is a
    // stable count that a dropped fixture (or a parser that emits fewer
    // structures) would break loudly. The 16 staged fixtures carry 842
    // such structural events in total.
    assert_eq!(total, 842, "corpus-wide setup-event count changed: {total}");
}

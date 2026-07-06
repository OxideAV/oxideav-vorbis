//! Fixture-anchored RFC 3533 Ogg framing conformance.
//!
//! The `ogg` module's page parser and CRC are validated against every
//! staged real-world fixture stream (`docs/audio/vorbis/fixtures/*/
//! input.ogg`): every page must parse with a verifying CRC, and
//! re-serializing the parsed pages must reproduce each `input.ogg`
//! **byte-for-byte** — pinning the CRC polynomial/feed convention, the
//! header field layout, and the lacing model against independently
//! produced streams. The packet assembler is then cross-checked
//! against the Vorbis layer: the first packet of every fixture must
//! parse as a §4.2.2 identification header.
//!
//! The fixtures live in the umbrella workspace's `docs/` submodule;
//! the standalone per-crate CI clones only this repo, so these tests
//! skip (with an eprintln marker) when the corpus is absent. This is
//! data availability, not a disabled test.

use oxideav_vorbis::identification::parse_identification_header;
use oxideav_vorbis::ogg::{parse_pages, PacketAssembler};

fn fixtures_root() -> String {
    format!(
        "{}/../../docs/audio/vorbis/fixtures",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn fixtures_available() -> bool {
    std::path::Path::new(&fixtures_root()).is_dir()
}

/// Every fixture directory that carries an `input.ogg`.
fn fixture_streams() -> Vec<(String, Vec<u8>)> {
    let mut streams = Vec::new();
    let root = fixtures_root();
    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .expect("fixtures root readable")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for dir in dirs {
        let ogg = dir.join("input.ogg");
        if let Ok(bytes) = std::fs::read(&ogg) {
            streams.push((dir.file_name().unwrap().to_string_lossy().into(), bytes));
        }
    }
    assert!(!streams.is_empty(), "no fixture input.ogg found in {root}");
    streams
}

#[test]
fn every_fixture_page_parses_crc_verifies_and_reserializes_byte_exact() {
    if !fixtures_available() {
        eprintln!("SKIP: docs/ fixtures submodule not checked out (standalone CI)");
        return;
    }
    for (name, bytes) in fixture_streams() {
        let pages =
            parse_pages(&bytes).unwrap_or_else(|e| panic!("{name}: page parse / CRC failure: {e}"));
        assert!(!pages.is_empty(), "{name}: no pages");

        // Byte-exact re-serialization of the whole physical stream.
        let mut rebuilt = Vec::with_capacity(bytes.len());
        for page in &pages {
            rebuilt.extend_from_slice(&page.serialize().expect("parsed page re-serializes"));
        }
        assert_eq!(
            rebuilt, bytes,
            "{name}: re-serialized stream is not byte-identical"
        );

        // Structural spot checks: first page is BOS; a fresh-stream
        // page 0 is never continued; sequences start at 0.
        assert!(pages[0].bos, "{name}: page 0 must be BOS");
        assert!(!pages[0].continued, "{name}: page 0 must not be continued");
        assert_eq!(pages[0].sequence, 0, "{name}: page 0 sequence");
        // Exactly the pages flagged EOS close a logical stream; every
        // fixture ends with one.
        assert!(pages.last().unwrap().eos, "{name}: final page must be EOS");
    }
}

#[test]
fn fixture_packets_reassemble_and_lead_with_an_identification_header() {
    if !fixtures_available() {
        eprintln!("SKIP: docs/ fixtures submodule not checked out (standalone CI)");
        return;
    }
    for (name, bytes) in fixture_streams() {
        let pages = parse_pages(&bytes).expect("pages parse");
        // Chained fixtures carry several logical streams back to back —
        // run one assembler per logical stream (reset at each BOS
        // boundary) and check every stream's first packet is a §4.2.2
        // identification header.
        let mut assembler = PacketAssembler::new();
        let mut stream_first_packet = true;
        let mut logical_streams = 0usize;
        for page in &pages {
            if page.bos {
                assembler.reset();
                stream_first_packet = true;
                logical_streams += 1;
            }
            let packets = assembler
                .push_page(page)
                .unwrap_or_else(|e| panic!("{name}: assembler failed: {e}"));
            for p in packets {
                if stream_first_packet {
                    let id = parse_identification_header(&p).unwrap_or_else(|e| {
                        panic!("{name}: first packet is not an id header: {e}")
                    });
                    assert!(id.audio_channels > 0, "{name}");
                    assert!(id.audio_sample_rate > 0, "{name}");
                    stream_first_packet = false;
                }
            }
        }
        assert!(logical_streams >= 1, "{name}: no BOS page found");

        // The un-chained single-stream convenience path agrees.
        let first_len = stream_end_of_first_logical(&pages, &bytes);
        let all = oxideav_vorbis::ogg::pages_to_packets(&bytes[..first_len])
            .expect("first logical stream packets assemble");
        assert!(all.len() >= 4, "{name}: fewer than 4 packets");

        // §A.2: the first page is exactly 58 bytes (27-byte header +
        // one lacing byte + the 30-byte identification packet).
        assert_eq!(pages[0].page_len(), 58, "{name}: first page length");
        assert_eq!(pages[0].granule_position, 0, "{name}: header-page granule");
    }
}

/// Byte length of the first logical stream: up to (and including) its
/// EOS page, or the whole file when un-chained.
fn stream_end_of_first_logical(pages: &[oxideav_vorbis::ogg::OggPage], bytes: &[u8]) -> usize {
    let mut pos = 0usize;
    for page in pages {
        pos += page.page_len();
        if page.eos {
            return pos;
        }
    }
    bytes.len()
}

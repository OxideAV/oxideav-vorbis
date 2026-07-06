//! Fixture-anchored RFC 3533 Ogg framing conformance — through the
//! `oxideav-ogg` container dependency.
//!
//! The in-crate page layer was removed in favour of `oxideav-ogg`;
//! this suite pins that the dependency's page parser / serializer and
//! this crate's lacing-model packet reassembly still hold against
//! every staged real-world fixture stream
//! (`docs/audio/vorbis/fixtures/*/input.ogg`): every page must parse
//! with a verifying CRC, re-serializing the parsed pages must
//! reproduce each `input.ogg` **byte-for-byte**, and the first packet
//! of the leading logical stream must parse as a §4.2.2
//! identification header.
//!
//! The fixtures live in the umbrella workspace's `docs/` submodule;
//! the standalone per-crate CI clones only this repo, so these tests
//! skip (with an eprintln marker) when the corpus is absent. This is
//! data availability, not a disabled test.

use oxideav_ogg::page::Page;
use oxideav_vorbis::identification::parse_identification_header;
use oxideav_vorbis::ogg_packets;

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

/// Parse every page (CRC-verified) with its on-wire byte length.
fn parse_pages(name: &str, data: &[u8]) -> Vec<(Page, usize)> {
    let mut pages = Vec::new();
    let mut off = 0usize;
    while off < data.len() {
        let (page, used) = Page::parse(&data[off..])
            .unwrap_or_else(|e| panic!("{name}: page parse / CRC failure at {off}: {e}"));
        pages.push((page, used));
        off += used;
    }
    pages
}

#[test]
fn every_fixture_page_parses_crc_verifies_and_reserializes_byte_exact() {
    if !fixtures_available() {
        eprintln!("SKIP: docs/ fixtures submodule not checked out (standalone CI)");
        return;
    }
    for (name, bytes) in fixture_streams() {
        let pages = parse_pages(&name, &bytes);
        assert!(!pages.is_empty(), "{name}: no pages");

        // Byte-exact re-serialization of the whole physical stream.
        let mut rebuilt = Vec::with_capacity(bytes.len());
        for (page, _) in &pages {
            rebuilt.extend_from_slice(&page.to_bytes());
        }
        assert_eq!(
            rebuilt, bytes,
            "{name}: re-serialized stream is not byte-identical"
        );

        // Structural spot checks: first page is BOS; a fresh-stream
        // page 0 is never continued; sequences start at 0.
        assert!(pages[0].0.is_first(), "{name}: page 0 must be BOS");
        assert!(
            !pages[0].0.is_continued(),
            "{name}: page 0 must not be continued"
        );
        assert_eq!(pages[0].0.seq_no, 0, "{name}: page 0 sequence");
        // Every fixture's physical stream ends with an EOS page.
        assert!(
            pages.last().unwrap().0.is_last(),
            "{name}: final page must be EOS"
        );
    }
}

#[test]
fn fixture_packets_reassemble_and_lead_with_an_identification_header() {
    if !fixtures_available() {
        eprintln!("SKIP: docs/ fixtures submodule not checked out (standalone CI)");
        return;
    }
    for (name, bytes) in fixture_streams() {
        let pages = parse_pages(&name, &bytes);

        // First logical stream only (chained fixtures carry several
        // back to back; multi-link chain walking is the `oxideav-ogg`
        // demuxer's job, exercised in that crate).
        let first_len = stream_end_of_first_logical(&pages, &bytes);
        let packets =
            ogg_packets(&bytes[..first_len]).expect("first logical stream packets assemble");
        assert!(packets.len() >= 4, "{name}: fewer than 4 packets");

        let id = parse_identification_header(&packets[0])
            .unwrap_or_else(|e| panic!("{name}: first packet is not an id header: {e}"));
        assert!(id.audio_channels > 0, "{name}");
        assert!(id.audio_sample_rate > 0, "{name}");

        // §A.2: the first page is exactly 58 bytes (27-byte header +
        // one lacing byte + the 30-byte identification packet).
        assert_eq!(pages[0].1, 58, "{name}: first page length");
        assert_eq!(
            pages[0].0.granule_position, 0,
            "{name}: header-page granule"
        );
    }
}

/// Byte length of the first logical stream: up to (and including) its
/// EOS page, or the whole file when un-chained.
fn stream_end_of_first_logical(pages: &[(Page, usize)], bytes: &[u8]) -> usize {
    let mut pos = 0usize;
    for (page, used) in pages {
        pos += used;
        if page.is_last() {
            return pos;
        }
    }
    bytes.len()
}

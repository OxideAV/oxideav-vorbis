//! Fixture-anchored Vorbis §A.2 remux conformance.
//!
//! Each staged single-stream fixture is de-framed to its packet
//! sequence, the per-packet granule positions are recomputed from the
//! §4.3.1 blocksize walk (packet `i > 0` finishes
//! `prev_n/4 + cur_n/4` more PCM samples; packet 0 primes and finishes
//! none), and the whole stream is re-encapsulated with
//! `mux_vorbis_stream`. The remuxed physical stream must:
//!
//! * de-frame back to the **identical packet sequence**;
//! * obey every §A.2 page rule (58-byte BOS first page, header pages
//!   at granule 0, audio on a fresh page, EOS last page);
//! * stamp every audio page with the blocksize-walk granule of the
//!   last packet completed on it (`-1` for spanned pages);
//! * carry the original stream's final granule position on its last
//!   page (the §A.2 end-trim is passed through), which the walk must
//!   bound within one long block.
//!
//! Skips when the docs/ fixtures submodule is absent (standalone CI).

use oxideav_core::bits::BitReaderLsb;
use oxideav_ogg::page::Page;
use oxideav_vorbis::identification::parse_identification_header;
use oxideav_vorbis::packet::read_packet_header;
use oxideav_vorbis::setup::parse_setup_header;
use oxideav_vorbis::{mux_vorbis_stream, ogg_packets};

/// Parse every page of a physical stream (CRC-verified by the
/// `oxideav-ogg` page parser), keeping each page's on-wire byte length.
fn parse_pages(data: &[u8]) -> Vec<(Page, usize)> {
    let mut pages = Vec::new();
    let mut off = 0usize;
    while off < data.len() {
        let (page, used) = Page::parse(&data[off..]).expect("page parses (CRC verifies)");
        pages.push((page, used));
        off += used;
    }
    pages
}

fn fixtures_root() -> String {
    format!(
        "{}/../../docs/audio/vorbis/fixtures",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn fixtures_available() -> bool {
    std::path::Path::new(&fixtures_root()).is_dir()
}

/// The single-logical-stream fixtures (the chained fixture is §A.1.1
/// legal but out of scope for a single-stream muxer test).
const FIXTURES: &[&str] = &[
    "5.1-channel-48000-q5",
    "mode-floor1-only",
    "mode-residue-types-0-1-2",
    "mono-22050-low-rate",
    "mono-44100-q5-typical",
    "noise-stream",
    "silence-stream",
    "stereo-44100-q-1",
    "stereo-44100-q10",
    "stereo-44100-q5-typical",
    "stereo-96000-high-rate",
    "stereo-cbr-128kbps",
    "transient-blocksize-switch",
    "with-attached-picture",
    "with-vorbis-comment-tags",
];

#[test]
fn every_fixture_remuxes_to_an_a2_conformant_stream_with_identical_packets() {
    if !fixtures_available() {
        eprintln!("SKIP: docs/ fixtures submodule not checked out (standalone CI)");
        return;
    }
    for dir in FIXTURES {
        let path = format!("{}/{dir}/input.ogg", fixtures_root());
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{dir}: read: {e}"));
        let original_pages = parse_pages(&bytes);
        let packets = ogg_packets(&bytes).expect("original packets assemble");
        assert!(packets.len() >= 4, "{dir}: too few packets");

        let id = parse_identification_header(&packets[0]).expect("id header parses");
        let setup = parse_setup_header(&packets[2], id.audio_channels).expect("setup parses");
        let b0 = id.blocksize_0 as usize;
        let b1 = id.blocksize_1 as usize;

        // §4.3.1 blocksize walk → per-packet absolute granule.
        let audio_packets = &packets[3..];
        let mut granules: Vec<u64> = Vec::with_capacity(audio_packets.len());
        let mut total = 0u64;
        let mut prev_n: Option<usize> = None;
        for pkt in audio_packets {
            let mut r = BitReaderLsb::new(pkt);
            let hdr = read_packet_header(&mut r, &setup, b0, b1).expect("audio header parses");
            if let Some(p) = prev_n {
                total += (p / 4 + hdr.n / 4) as u64;
            }
            prev_n = Some(hdr.n);
            granules.push(total);
        }

        // §A.2 end-trim pass-through: the original last page's granule
        // replaces the walk's final value; it may understate the walk
        // by less than one long block, never overstate it.
        let final_granule = original_pages.last().unwrap().0.granule_position;
        assert!(final_granule >= 0, "{dir}: final granule");
        let final_granule = final_granule as u64;
        let natural = *granules.last().unwrap();
        assert!(
            final_granule <= natural && natural - final_granule < b1 as u64,
            "{dir}: end-trim {final_granule} vs natural {natural}"
        );
        *granules.last_mut().unwrap() = final_granule;

        let audio: Vec<(Vec<u8>, u64)> = audio_packets
            .iter()
            .cloned()
            .zip(granules.iter().copied())
            .collect();
        let remuxed = mux_vorbis_stream(0x5EED, &packets[0], &packets[1], &packets[2], &audio)
            .unwrap_or_else(|e| panic!("{dir}: remux: {e}"));

        // Identical packet sequence back out.
        let packets_again = ogg_packets(&remuxed).expect("remuxed packets assemble");
        assert_eq!(packets_again, packets, "{dir}: packet sequence changed");

        // §A.2 structure.
        let pages = parse_pages(&remuxed);
        assert_eq!(pages[0].1, 58, "{dir}: first page");
        assert!(pages[0].0.is_first(), "{dir}");
        assert_eq!(pages[0].0.granule_position, 0, "{dir}");
        assert_eq!(pages[1].0.granule_position, 0, "{dir}: header page granule");
        assert!(pages.last().unwrap().0.is_last(), "{dir}");
        assert_eq!(
            pages.last().unwrap().0.granule_position,
            final_granule as i64,
            "{dir}: final granule"
        );

        // Audio pages: granule = walk value of the last packet
        // completed on the page; -1 when none completes. Locate the
        // first audio page (the page after the one the setup header
        // completes on).
        let mut audio_page_start = None;
        let mut completed = 0usize;
        for (i, (page, _)) in pages.iter().enumerate() {
            completed += page.lacing.iter().filter(|&&l| l < 255).count();
            if completed >= 3 {
                audio_page_start = Some(i + 1);
                break;
            }
        }
        let start = audio_page_start.expect("headers complete somewhere");
        assert!(
            !pages[start].0.is_continued(),
            "{dir}: audio must begin a fresh page"
        );
        let mut idx = 0usize;
        for (page, _) in &pages[start..] {
            let done = page.lacing.iter().filter(|&&l| l < 255).count();
            if done == 0 {
                assert_eq!(page.granule_position, -1, "{dir}: spanned page granule");
            } else {
                idx += done;
                assert_eq!(
                    page.granule_position,
                    granules[idx - 1] as i64,
                    "{dir}: page granule after packet {idx}"
                );
            }
        }
        assert_eq!(idx, audio.len(), "{dir}: all audio packets placed");
    }
}

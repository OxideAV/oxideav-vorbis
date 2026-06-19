//! VORBIS_COMMENT header parse + full audio decode, end-to-end against the
//! two metadata-carrying staged fixtures.
//!
//! # What this pins
//!
//! `tests/fixture_pcm_decode.rs` decodes the audio-shaped fixtures; the two
//! corpus members it does not consume are the metadata fixtures:
//! `with-vorbis-comment-tags` (a §5.2 comment header populated with the
//! canonical TITLE / ARTIST / ALBUM / DATE / GENRE / TRACKNUMBER fields)
//! and `with-attached-picture` (a comment header carrying a >1 KB
//! `METADATA_BLOCK_PICTURE` base64 blob — the FLAC-borrowed cover-art
//! convention Ogg/Vorbis uses since Vorbis I has no native picture block).
//!
//! Both fixtures exercise two things at once:
//!
//! 1. **Comment-header parse.** [`parse_comment_header`] must recover the
//!    vendor string and every `KEY=value` entry (§5.2.1 / §5.2.2),
//!    case-insensitively keyed per §5.2.2.
//!
//! 2. **Decode is unperturbed by metadata.** Both fixtures share the exact
//!    same audio (`sine=440 0.5 s`) and therefore the **same**
//!    `expected.wav` (identical SHA-256 in both `notes.md`). Decoding each
//!    to sample-exact PCM proves a large comment block — long entries, a
//!    picture blob over 1 KB — does not shift the setup/audio packet
//!    boundaries or perturb the §4.3 decode.
//!
//! The Ogg de-framer is the same single-bitstream test scaffolding used by
//! the sibling fixture tests (container framing is not codec work per
//! `docs/IMPLEMENTOR_ROUND.md`); it never compiles into `src/`.
//!
//! # Standalone-CI skip
//!
//! The corpus lives in the umbrella `docs/` submodule, absent in
//! standalone per-crate CI. Each test prints a skip notice and returns
//! when the fixtures directory is missing (data availability, not a
//! disabled test — no `#[ignore]`).

use oxideav_vorbis::{
    parse_comment_header, parse_identification_header, parse_setup_header, AudioDecoderState,
    StreamingDecoder, StreamingFrame,
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

/// Minimal single-bitstream RFC-3533 page de-framer (test scaffolding).
fn ogg_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut pending: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    while pos + 27 <= data.len() {
        assert_eq!(&data[pos..pos + 4], b"OggS", "Ogg page sync lost at {pos}");
        let seg_count = data[pos + 26] as usize;
        let seg_table_start = pos + 27;
        let body_start = seg_table_start + seg_count;
        if body_start > data.len() {
            break;
        }
        let seg_table = &data[seg_table_start..body_start];
        let mut body = body_start;
        for &lace in seg_table {
            let l = lace as usize;
            if body + l > data.len() {
                break;
            }
            pending.extend_from_slice(&data[body..body + l]);
            body += l;
            if l < 255 {
                packets.push(std::mem::take(&mut pending));
            }
        }
        pos = body;
    }
    if !pending.is_empty() {
        packets.push(pending);
    }
    packets
}

fn wav_s16(data: &[u8]) -> Vec<i16> {
    let mut pos = 12;
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

/// Case-insensitive comment lookup (§5.2.2 field names are case-insensitive
/// ASCII). Returns the value of the first entry whose key folds to `key`.
fn comment_value<'a>(comments: &'a [String], key: &str) -> Option<&'a str> {
    comments.iter().find_map(|entry| {
        let eq = entry.find('=')?;
        if entry[..eq].eq_ignore_ascii_case(key) {
            Some(&entry[eq + 1..])
        } else {
            None
        }
    })
}

/// Decode a single-bitstream fixture's audio to mono f32 PCM and return the
/// parsed comment header alongside it.
fn parse_and_decode(dir: &str) -> (Vec<String>, String, Vec<f32>) {
    let base = format!("{}/{dir}", fixtures_root());
    let ogg = std::fs::read(format!("{base}/input.ogg")).expect("input.ogg");
    let packets = ogg_packets(&ogg);
    assert!(packets.len() >= 4, "{dir}: too few packets");

    let id = parse_identification_header(&packets[0]).expect("id header");
    let comment = parse_comment_header(&packets[1]).expect("comment header");
    let setup = parse_setup_header(&packets[2], id.audio_channels).expect("setup header");
    let state = AudioDecoderState::new(&setup).expect("decoder state");
    assert_eq!(id.audio_channels, 1, "{dir}: fixtures are mono");

    let mut dec = StreamingDecoder::new(
        id.audio_channels,
        id.blocksize_0 as usize,
        id.blocksize_1 as usize,
        IMDCT_SCALE,
    );
    let mut pcm: Vec<f32> = Vec::new();
    for pkt in &packets[3..] {
        let mut r = oxideav_core::bits::BitReaderLsb::new(pkt);
        match dec.push_packet(&mut r, &setup, &state) {
            Ok(StreamingFrame::Pcm {
                per_channel_pcm, ..
            }) => {
                pcm.extend_from_slice(&per_channel_pcm[0]);
            }
            Ok(StreamingFrame::Primed { .. }) => {}
            Err(e) => panic!("{dir}: decode failed: {e}"),
        }
    }
    (comment.comments, comment.vendor, pcm)
}

/// Assert a fixture's audio decodes sample-exact (±1 s16) against its
/// `expected.wav`, given already-decoded mono PCM.
fn assert_audio_sample_exact(dir: &str, pcm: &[f32]) {
    let base = format!("{}/{dir}", fixtures_root());
    let wav = std::fs::read(format!("{base}/expected.wav")).expect("expected.wav");
    let expected = wav_s16(&wav);
    assert!(!expected.is_empty(), "{dir}: expected.wav empty");
    assert!(
        pcm.len() >= expected.len(),
        "{dir}: decoded {} < expected {}",
        pcm.len(),
        expected.len()
    );
    let mut max_diff = 0i32;
    let mut mismatches = 0usize;
    for (i, &exp) in expected.iter().enumerate() {
        let dec = (pcm[i] * 32768.0).round().clamp(-32768.0, 32767.0) as i32;
        let diff = (dec - exp as i32).abs();
        max_diff = max_diff.max(diff);
        if diff > 1 {
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{dir}: {mismatches}/{} samples exceed ±1 s16 (max diff {max_diff})",
        expected.len()
    );
}

#[test]
fn comment_tags_parse_and_audio_decodes_sample_exact() {
    if !fixtures_available() {
        eprintln!("SKIP with-vorbis-comment-tags: docs/ fixtures not checked out");
        return;
    }
    let (comments, vendor, pcm) = parse_and_decode("with-vorbis-comment-tags");

    // Vendor + the canonical tag set the fixture's notes.md documents.
    assert_eq!(vendor, "Lavf61.7.100");
    assert_eq!(comment_value(&comments, "TITLE"), Some("Test Title"));
    assert_eq!(comment_value(&comments, "ARTIST"), Some("Test Artist"));
    assert_eq!(comment_value(&comments, "ALBUM"), Some("Test Album"));
    assert_eq!(comment_value(&comments, "DATE"), Some("2026"));
    assert_eq!(comment_value(&comments, "GENRE"), Some("Synth"));
    assert_eq!(comment_value(&comments, "TRACKNUMBER"), Some("1"));
    // §5.2.2 keys are case-insensitive: lowercase lookup resolves too.
    assert_eq!(comment_value(&comments, "title"), Some("Test Title"));

    assert_audio_sample_exact("with-vorbis-comment-tags", &pcm);
}

#[test]
fn attached_picture_parses_and_audio_decodes_sample_exact() {
    if !fixtures_available() {
        eprintln!("SKIP with-attached-picture: docs/ fixtures not checked out");
        return;
    }
    let (comments, vendor, pcm) = parse_and_decode("with-attached-picture");

    assert_eq!(vendor, "Lavf61.7.100");
    assert_eq!(comment_value(&comments, "TITLE"), Some("With Picture"));

    // The METADATA_BLOCK_PICTURE convention: a base64 FLAC-PICTURE blob
    // carried as a long (>1 KB raw) comment value. We recover the raw
    // base64 string and confirm it is the documented FLAC-PICTURE form
    // (picture type 3 = front cover, MIME "image/png") by base64-decoding
    // its leading fields. No image decoder is needed — the FLAC-PICTURE
    // header is plain big-endian fields.
    let blob = comment_value(&comments, "METADATA_BLOCK_PICTURE")
        .expect("METADATA_BLOCK_PICTURE entry present");
    let raw = base64_decode(blob).expect("METADATA_BLOCK_PICTURE is valid base64");
    // FLAC-PICTURE: u32 picture_type, u32 mime_len, mime bytes, ...
    assert!(raw.len() >= 8, "picture blob too short");
    let picture_type = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    assert_eq!(picture_type, 3, "front-cover picture type per the fixture");
    let mime_len = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
    assert!(8 + mime_len <= raw.len(), "mime length overflows blob");
    let mime = std::str::from_utf8(&raw[8..8 + mime_len]).expect("mime utf8");
    assert_eq!(mime, "image/png", "the embedded picture is a PNG");

    assert_audio_sample_exact("with-attached-picture", &pcm);
}

#[test]
fn both_metadata_fixtures_share_the_same_audio() {
    // The two fixtures carry different metadata but identical audio
    // (`sine=440 0.5 s`); their notes.md report the same expected.wav and
    // raw-PCM SHA-256. Decoding both and comparing confirms the comment
    // block — short tags vs. a >1 KB picture blob — does not perturb the
    // setup/audio packet boundaries or the §4.3 decode.
    if !fixtures_available() {
        eprintln!("SKIP metadata-fixtures-share-audio: docs/ fixtures not checked out");
        return;
    }
    let (_, _, tags_pcm) = parse_and_decode("with-vorbis-comment-tags");
    let (_, _, pic_pcm) = parse_and_decode("with-attached-picture");
    let n = tags_pcm.len().min(pic_pcm.len());
    assert!(n > 8_000, "both decode a full ~0.5 s clip");
    // Identical audio: every sample must match exactly (same encoder, same
    // bitstream — no lossy variance between two decodes of the same bytes).
    let mut max_diff = 0f32;
    for i in 0..n {
        max_diff = max_diff.max((tags_pcm[i] - pic_pcm[i]).abs());
    }
    assert_eq!(
        max_diff, 0.0,
        "the two metadata fixtures must decode to bit-identical audio"
    );
}

/// Minimal standard-alphabet base64 decoder (test scaffolding) for the
/// METADATA_BLOCK_PICTURE blob. Ignores `=` padding; rejects other
/// non-alphabet bytes.
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::new();
    let mut acc = 0u32;
    let mut bits = 0u32;
    for &c in s.as_bytes() {
        if c == b'=' {
            break;
        }
        let v = val(c)?;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

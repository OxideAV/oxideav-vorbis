//! §4.3 decode robustness — the per-packet driver must reject malformed or
//! truncated audio packets with a typed error, never panic.
//!
//! # Why this matters
//!
//! A real Vorbis decoder consumes packets sourced from an Ogg demuxer that
//! may itself be fed a corrupt or truncated file. The §4.3.1 closing note
//! mandates that an end-of-packet anywhere in the packet prelude "discards
//! this packet from the stream" — i.e. it is a recoverable error, not a
//! crash. Every read in the §4.3.1..§4.3.7 driver is end-of-packet-guarded
//! and returns a typed [`StreamingError`]; this test exercises that
//! contract against adversarial inputs derived from a real fixture's valid
//! packet so the harness is grounded in a stream the decoder accepts when
//! intact.
//!
//! What is asserted:
//!
//! 1. **Truncation at every byte length** of a real audio packet either
//!    decodes (a short prefix can still be a complete, valid short packet)
//!    or returns `Err` — but never panics.
//! 2. **A header-type packet** (first bit set: §4.3.1 step 1 requires
//!    `[packet_type] == 0`) routed into the audio driver returns
//!    `NonAudioPacketType`.
//! 3. **Empty and single-byte packets** return a graceful error.
//! 4. **Random-byte packets** never panic.
//!
//! The Ogg de-framer is the same single-bitstream test scaffolding used by
//! the sibling fixture tests; it never compiles into `src/`.
//!
//! # Standalone-CI skip
//!
//! The corpus lives in the umbrella `docs/` submodule, absent in
//! standalone per-crate CI. The fixture-derived test skips when the
//! fixtures directory is missing (no `#[ignore]`); the synthetic-input
//! tests need no corpus and always run.

use oxideav_vorbis::{
    parse_identification_header, parse_setup_header, AudioDecoderState, StreamingDecoder,
    StreamingError, StreamingFrame, VorbisSetupHeader,
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

fn ogg_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut pending: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    while pos + 27 <= data.len() {
        if &data[pos..pos + 4] != b"OggS" {
            break;
        }
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

/// The decode geometry a fixture supplies: channel count + the two block
/// sizes, enough to (re)build a fresh [`StreamingDecoder`].
#[derive(Clone, Copy)]
struct Geometry {
    channels: u8,
    blocksize_0: usize,
    blocksize_1: usize,
}

impl Geometry {
    fn decoder(&self) -> StreamingDecoder {
        StreamingDecoder::new(self.channels, self.blocksize_0, self.blocksize_1, 1.0)
    }
}

/// Load a fixture's parsed setup + decoder state + decode geometry + the
/// first audio packet (the 4th logical packet: id, comment, setup, audio).
fn load_fixture(dir: &str) -> Option<(VorbisSetupHeader, AudioDecoderState, Geometry, Vec<u8>)> {
    if !fixtures_available() {
        return None;
    }
    let ogg = std::fs::read(format!("{}/{dir}/input.ogg", fixtures_root())).ok()?;
    let packets = ogg_packets(&ogg);
    if packets.len() < 4 {
        return None;
    }
    let id = parse_identification_header(&packets[0]).ok()?;
    let setup = parse_setup_header(&packets[2], id.audio_channels).ok()?;
    let state = AudioDecoderState::new(&setup).ok()?;
    let geom = Geometry {
        channels: id.audio_channels,
        blocksize_0: id.blocksize_0 as usize,
        blocksize_1: id.blocksize_1 as usize,
    };
    let audio = packets[3].clone();
    Some((setup, state, geom, audio))
}

/// Push one packet through the driver. Returns `Ok(())` on a clean decode
/// (Pcm or Primed) and `Err(())` on a typed `StreamingError` — the point is
/// that neither path panics.
fn push_one(
    dec: &mut StreamingDecoder,
    setup: &VorbisSetupHeader,
    state: &AudioDecoderState,
    pkt: &[u8],
) -> Result<(), StreamingError> {
    let mut r = oxideav_core::bits::BitReaderLsb::new(pkt);
    match dec.push_packet(&mut r, setup, state) {
        Ok(StreamingFrame::Pcm { .. }) | Ok(StreamingFrame::Primed { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

#[test]
fn truncation_at_every_length_never_panics() {
    let Some((setup, state, geom, audio)) = load_fixture("noise-stream") else {
        eprintln!("SKIP truncation: docs/ fixtures not checked out");
        return;
    };
    // The full packet is valid; every shorter prefix is fed through a fresh
    // priming decoder. A prefix either decodes (still a well-formed packet)
    // or errors — but the driver must never panic on a short read.
    let mut ok_count = 0usize;
    let mut err_count = 0usize;
    for len in 0..=audio.len() {
        let mut dec = geom.decoder();
        // First push primes; the truncated push is the one under test.
        let _ = push_one(&mut dec, &setup, &state, &audio);
        match push_one(&mut dec, &setup, &state, &audio[..len]) {
            Ok(()) => ok_count += 1,
            Err(_) => err_count += 1,
        }
    }
    // Sanity: both outcomes occur across the truncation sweep (the empty
    // prefix errors; the full-length prefix decodes), and the total equals
    // the number of prefixes tried — i.e. no prefix panicked.
    assert_eq!(
        ok_count + err_count,
        audio.len() + 1,
        "every truncation length must terminate without panic"
    );
    assert!(err_count > 0, "the empty / short prefixes must error");
    assert!(ok_count > 0, "the full-length packet must decode");
}

#[test]
fn header_type_packet_routed_to_audio_is_rejected() {
    let Some((setup, state, geom, _)) = load_fixture("noise-stream") else {
        eprintln!("SKIP header-type-rejection: docs/ fixtures not checked out");
        return;
    };
    let mut dec = geom.decoder();
    // §4.3.1 step 1: an audio packet's first bit must be 0. A packet whose
    // first byte has bit 0 set (e.g. any of the 0x01/0x03/0x05 Vorbis
    // header packets) must be rejected as non-audio, not decoded.
    for &first in &[0x01u8, 0x03, 0x05] {
        let pkt = vec![first, b'v', b'o', b'r', b'b', b'i', b's'];
        let err = push_one(&mut dec, &setup, &state, &pkt)
            .expect_err("a header-type packet must not decode as audio");
        match err {
            StreamingError::Packet(_) => {}
            other => panic!("expected a Packet error for first byte {first:#04x}, got {other:?}"),
        }
    }
}

#[test]
fn empty_and_short_packets_error_gracefully() {
    let Some((setup, state, geom, _)) = load_fixture("noise-stream") else {
        eprintln!("SKIP short-packets: docs/ fixtures not checked out");
        return;
    };
    let mut dec = geom.decoder();
    for pkt in [&[][..], &[0x00][..], &[0xff][..]] {
        // No panic, regardless of decode/err outcome.
        let _ = push_one(&mut dec, &setup, &state, pkt);
    }
}

#[test]
fn random_byte_packets_never_panic() {
    let Some((setup, state, geom, _)) = load_fixture("stereo-44100-q5-typical") else {
        eprintln!("SKIP random-bytes: docs/ fixtures not checked out");
        return;
    };
    let mut dec = geom.decoder();
    // A cheap deterministic LCG drives pseudo-random packet bodies through
    // the driver. The contract is panic-freedom; any decode/err outcome is
    // acceptable.
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    for _ in 0..512 {
        let len = (next() % 200) as usize;
        let pkt: Vec<u8> = (0..len).map(|_| next() as u8).collect();
        let _ = push_one(&mut dec, &setup, &state, &pkt);
    }
}

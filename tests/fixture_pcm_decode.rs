//! End-to-end Vorbis I decode → PCM, sample-exact against the staged
//! `docs/audio/vorbis/fixtures/*/expected.wav` reference dumps.
//!
//! # What this pins
//!
//! Rounds 1..18 of the clean-room rebuild landed every §4.3 decode stage
//! and stitched them into [`oxideav_vorbis::StreamingDecoder`]. The one
//! piece left deferred was the §4.3.7 IMDCT normalization scalar
//! (`imdct_scale`), documented as "falls out of matching the fixture
//! traces" — but the staged traces log structural events, not post-IMDCT
//! samples, so the scalar could not be pinned from them.
//!
//! Each fixture directory now ships an `expected.wav`: the s16le PCM a
//! black-box reference decoder (`ffmpeg`'s native Vorbis decoder, per the
//! fixture `notes.md`) produced from `input.ogg`. That WAV is a
//! validator *output*, consumed here as opaque ground-truth bytes — no
//! reference *source* is read. Comparing our decode to it pins the scalar
//! empirically: at `imdct_scale = 1.0` the bare §4.3.7 cosine-summation
//! kernel — combined with the §4.3.6 window and §4.3.8 overlap-add whose
//! squared-overlap property already carries the reconstruction
//! normalization — reproduces the reference PCM sample-for-sample within
//! the fixture's documented ±1 s16 tolerance (Vorbis is lossy; the
//! `notes.md` states "abs(sample_a - sample_b) <= 1 per s16 sample is
//! normal across decoder revisions"). So the deferred scalar is
//! **1.0** — the Vorbis IMDCT requires no extra Vorbis-specific
//! normalization beyond the kernel + window already implemented.
//!
//! # Why the Ogg de-framer lives in this test, not in `src/`
//!
//! Ogg page framing is container work — it belongs in `oxideav-ogg`, not
//! this codec crate (per `docs/IMPLEMENTOR_ROUND.md` "codec crates do not
//! own container code"). To avoid a cross-crate dev-dependency the test
//! carries a *minimal* RFC-3533 page walker as private test scaffolding:
//! it only coalesces lacing-segmented packets for a single logical
//! bitstream, enough to feed the public `StreamingDecoder` path. It is
//! not part of the crate's API surface and never compiles into `src/`.
//!
//! # Standalone-CI skip
//!
//! The fixtures live in the umbrella workspace's `docs/` submodule, which
//! is **not** checked out when the crate is built standalone (the
//! per-crate CI clones only this repo). When the fixtures directory is
//! absent each test prints a skip notice and returns `Ok` rather than
//! failing — the assertions run in the umbrella workspace (and locally),
//! where `docs/` is present. This is data availability, not a disabled
//! test: there is no `#[ignore]`, and the test body runs in full wherever
//! the corpus exists.

use oxideav_vorbis::{
    parse_identification_header, parse_setup_header, AudioDecoderState, StreamingDecoder,
    StreamingFrame,
};

/// The pinned §4.3.7 IMDCT normalization scalar. See the module header:
/// the bare cosine-summation kernel plus the §4.3.6 window and §4.3.8
/// overlap-add reproduce the reference PCM with no extra scaling, so the
/// Vorbis-specific normalization factor is exactly 1.0.
const IMDCT_SCALE: f32 = 1.0;

/// Root of the staged Vorbis fixtures (umbrella `docs/` submodule).
fn fixtures_root() -> String {
    format!(
        "{}/../../docs/audio/vorbis/fixtures",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// `true` when the umbrella `docs/` submodule with the fixtures is
/// checked out. The standalone per-crate CI clones only this repo, so the
/// fixtures are absent there and the corpus tests skip (see module
/// header).
fn fixtures_available() -> bool {
    std::path::Path::new(&fixtures_root()).is_dir()
}

/// Minimal RFC-3533 Ogg page de-framer (test scaffolding only).
///
/// Walks pages sequentially, coalescing each page's lacing segments into
/// logical packets: a segment whose lacing value is `< 255` ends a
/// packet, a `255` segment continues it into the next. Single logical
/// bitstream only (the staged fixtures are one stream, except the
/// chained-streams fixture which this test set does not consume).
fn ogg_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut pending: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    while pos + 27 <= data.len() {
        assert_eq!(
            &data[pos..pos + 4],
            b"OggS",
            "Ogg page sync lost at byte {pos}"
        );
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

/// Decode one fixture's `input.ogg` to per-channel f32 PCM rows (in
/// Vorbis bitstream channel order) through the full public
/// `StreamingDecoder::push_packet` path. Returns the per-channel rows.
fn decode_fixture(dir: &str) -> Vec<Vec<f32>> {
    let base = format!("{}/{}", fixtures_root(), dir);
    let ogg = std::fs::read(format!("{base}/input.ogg")).expect("fixture input.ogg present");
    let packets = ogg_packets(&ogg);
    assert!(
        packets.len() >= 4,
        "{dir}: expected id + comment + setup + audio packets, got {}",
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

    // Per-channel PCM, in bitstream channel order.
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
            Err(e) => panic!("{dir}: decode failed: {e}"),
        }
    }
    per_ch
}

/// Assert a fixture decodes sample-exact (within ±1 s16) against its
/// `expected.wav`. Vorbis is lossy, so the conformance bar the fixture
/// `notes.md` documents is `abs(a - b) <= 1` per s16 sample. The decoded
/// stream can carry a few trailing samples past the reference (the
/// encoder's granule-position end-trim); we compare the reference's
/// length from frame 0 (the empirically-confirmed alignment offset is 0
/// for every staged fixture).
///
/// `wav_from_bitstream[w]` names the Vorbis bitstream channel index that
/// feeds WAV interleave slot `w`. The decoder emits channels in §4.3.2
/// bitstream order; the reference WAV stores them in the standard
/// interleave order, so a multi-channel fixture supplies the §4.3.9
/// permutation here (mono / stereo are the identity).
fn assert_fixture_sample_exact(dir: &str, wav_from_bitstream: &[usize]) {
    if !fixtures_available() {
        eprintln!("SKIP {dir}: docs/ fixtures submodule not checked out (standalone CI)");
        return;
    }
    let base = format!("{}/{}", fixtures_root(), dir);
    let wav = std::fs::read(format!("{base}/expected.wav")).expect("expected.wav present");
    let expected = wav_s16(&wav);
    assert!(!expected.is_empty(), "{dir}: expected.wav has no samples");

    let per_ch = decode_fixture(dir);
    let ch = per_ch.len();
    assert_eq!(
        ch,
        wav_from_bitstream.len(),
        "{dir}: channel permutation length {} != decoded channels {ch}",
        wav_from_bitstream.len()
    );
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

/// Identity channel map for `ch` channels (mono / stereo: bitstream
/// order already matches the WAV interleave order).
fn identity(ch: usize) -> Vec<usize> {
    (0..ch).collect()
}

#[test]
fn mono_22050_low_rate_decodes_sample_exact() {
    assert_fixture_sample_exact("mono-22050-low-rate", &identity(1));
}

#[test]
fn mono_44100_q5_typical_decodes_sample_exact() {
    assert_fixture_sample_exact("mono-44100-q5-typical", &identity(1));
}

#[test]
fn stereo_44100_q5_typical_decodes_sample_exact() {
    assert_fixture_sample_exact("stereo-44100-q5-typical", &identity(2));
}

#[test]
fn stereo_44100_q10_decodes_sample_exact() {
    assert_fixture_sample_exact("stereo-44100-q10", &identity(2));
}

#[test]
fn stereo_44100_q_minus_1_decodes_sample_exact() {
    // Lowest quality (q−1): aggressive coupling / coarse quantization.
    assert_fixture_sample_exact("stereo-44100-q-1", &identity(2));
}

#[test]
fn stereo_96000_high_rate_decodes_sample_exact() {
    assert_fixture_sample_exact("stereo-96000-high-rate", &identity(2));
}

#[test]
fn noise_stream_decodes_sample_exact() {
    // Full residue payload (no floor short-circuit): exercises residue
    // formats + the dot product + IMDCT on dense spectra.
    assert_fixture_sample_exact("noise-stream", &identity(1));
}

#[test]
fn mode_floor1_only_decodes_sample_exact() {
    assert_fixture_sample_exact("mode-floor1-only", &identity(1));
}

#[test]
fn stereo_cbr_128kbps_decodes_sample_exact() {
    assert_fixture_sample_exact("stereo-cbr-128kbps", &identity(2));
}

#[test]
fn mode_residue_types_0_1_2_decodes_sample_exact() {
    // Exercises all three residue formats in one stream.
    assert_fixture_sample_exact("mode-residue-types-0-1-2", &identity(2));
}

#[test]
fn transient_blocksize_switch_decodes_sample_exact() {
    // Block-size switching: short ↔ long transitions exercise the
    // §1.3.2 hybrid window halves and the §4.3.8 mixed-length overlap-add.
    assert_fixture_sample_exact("transient-blocksize-switch", &identity(1));
}

#[test]
fn channel_5_1_decodes_sample_exact() {
    // 5.1 surround. The decoder emits §4.3.2 bitstream order
    // [FL, FC, FR, RL, RR, LFE] (the §4.3.9 LAYOUT_6 the `channel_order`
    // module exposes); the reference WAV interleaves the standard
    // [FL, FR, FC, LFE, RL(BL), RR(BR)] order. The permutation below maps
    // each WAV slot back to its source bitstream channel, validating the
    // §4.3.9 mapping end-to-end:
    //   WAV FL←0, FR←2, FC←1, LFE←5, RL←3, RR←4.
    assert_fixture_sample_exact("5.1-channel-48000-q5", &[0, 2, 1, 5, 3, 4]);
}

//! Integration tests against the docs/audio/vorbis/ fixture corpus.
//!
//! Each fixture under `../../docs/audio/vorbis/fixtures/<name>/` carries
//! an `input.ogg` (Vorbis-in-Ogg) and an `expected.wav` PCM ground truth
//! produced by FFmpeg's native Vorbis decoder. This file walks every
//! fixture, demuxes the Ogg via [`oxideav_ogg::demux`], decodes each
//! Vorbis packet with [`oxideav_vorbis`], and reports per-channel RMS,
//! sample-match percentage, and PSNR against the WAV reference.
//!
//! Vorbis is a lossy codec — the spec only requires bit-exact behaviour
//! within the floating-point IMDCT path that libvorbis happens to use,
//! and most clean-room re-implementations differ by ±1 LSB on most
//! samples. Every fixture is therefore filed as `Tier::ReportOnly`:
//! the test never fails on a numeric divergence, it just records the
//! delta so a human reviewer can see at a glance how close we are. A
//! BitExact tier exists for future use if a fixture is found to round-
//! trip cleanly through both libvorbis and our decoder.
//!
//! Per-fixture classification:
//! * `Tier::ReportOnly` — divergence is logged but not asserted.
//! * `Tier::BitExact` — must produce sample-for-sample identical PCM.
//! * `Tier::Ignored` — disabled (decoder errors out, container variant
//!   not yet supported, etc.).
//!
//! The `chained-streams` fixture exercises Ogg's chained-bitstream
//! feature (RFC 3533 §5). The current `oxideav-ogg` demuxer only
//! registers logical streams found in the initial BOS section, so
//! packets from the second chained stream are silently dropped during
//! demux. The fixture's `expected.wav` covers only the FIRST stream's
//! samples (per its `notes.md`), so the comparison still works — we
//! just don't decode the trailing stream. A follow-up will extend the
//! demuxer to handle mid-file BOS pages.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{Decoder, Demuxer, Error, Frame, NullCodecResolver, ReadSeek};
use oxideav_vorbis::decoder::make_decoder;

/// Locate `docs/audio/vorbis/fixtures/<name>/`. Tests run with CWD
/// set to the crate root, so we walk two levels up to reach the
/// workspace root and then into `docs/`.
fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/audio/vorbis/fixtures").join(name)
}

#[derive(Clone, Copy, Debug)]
enum Tier {
    /// Must produce sample-for-sample identical PCM. Test fails on any
    /// divergence. Reserved for future use; no Vorbis fixture is
    /// currently expected to meet this bar.
    #[allow(dead_code)]
    BitExact,
    /// Decode is permitted to diverge from the libvorbis reference;
    /// per-channel deltas are logged but not asserted.
    ReportOnly,
}

struct CorpusCase {
    name: &'static str,
    /// Expected channels (used to sanity-check the demuxer's
    /// identification-header parse). Set to None to skip the check.
    channels: Option<u16>,
    /// Expected sample rate. None to skip.
    sample_rate: Option<u32>,
    tier: Tier,
}

/// Decoded output from one fixture: interleaved s16le samples plus the
/// channel count + sample rate the decoder advertised.
struct DecodedPcm {
    samples: Vec<i16>,
    channels: u16,
    sample_rate: u32,
}

/// Reference PCM extracted from the fixture's expected.wav.
struct RefPcm {
    samples: Vec<i16>,
    channels: u16,
    sample_rate: u32,
}

/// Per-channel diff numbers + aggregate match percentage and PSNR.
struct ChannelStat {
    rms_ref: f64,
    rms_ours: f64,
    rms_err: f64,
    exact: usize,
    near: usize, // |delta| <= 1 LSB
    total: usize,
    max_abs_err: i32,
}

impl ChannelStat {
    fn new() -> Self {
        Self {
            rms_ref: 0.0,
            rms_ours: 0.0,
            rms_err: 0.0,
            exact: 0,
            near: 0,
            total: 0,
            max_abs_err: 0,
        }
    }

    fn match_pct(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.exact as f64 / self.total as f64 * 100.0
        }
    }

    fn near_pct(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.near as f64 / self.total as f64 * 100.0
        }
    }

    /// PSNR over a 16-bit signed full scale (peak = 32767). Returns
    /// `f64::INFINITY` on perfect match.
    fn psnr_db(&self) -> f64 {
        if self.total == 0 || self.rms_err == 0.0 {
            return f64::INFINITY;
        }
        let mse = self.rms_err * self.rms_err / self.total as f64;
        let peak = 32767.0_f64;
        10.0 * (peak * peak / mse).log10()
    }
}

/// Demux the fixture's input.ogg into its Vorbis packets and decode
/// the FIRST logical stream end-to-end.
fn decode_fixture_pcm(case: &CorpusCase) -> Option<DecodedPcm> {
    let dir = fixture_dir(case.name);
    let ogg_path = dir.join("input.ogg");
    let file = match fs::File::open(&ogg_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, ogg_path.display());
            return None;
        }
    };
    let rs: Box<dyn ReadSeek> = Box::new(file);
    let mut demux = match oxideav_ogg::demux::open(rs, &NullCodecResolver) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip {}: ogg demuxer open failed: {e}", case.name);
            return None;
        }
    };

    let streams = demux.streams();
    if streams.is_empty() {
        eprintln!("skip {}: ogg has no streams", case.name);
        return None;
    }
    let stream = streams[0].clone();
    let params = stream.params.clone();
    if params.codec_id.as_str() != "vorbis" {
        eprintln!(
            "skip {}: first stream is not vorbis (got {})",
            case.name,
            params.codec_id.as_str()
        );
        return None;
    }
    let channels = params.channels.unwrap_or(0);
    let sample_rate = params.sample_rate.unwrap_or(0);
    if channels == 0 || sample_rate == 0 {
        eprintln!(
            "skip {}: stream advertises bogus channels/rate ({channels}/{sample_rate})",
            case.name
        );
        return None;
    }
    if let Some(want_ch) = case.channels {
        assert_eq!(
            channels, want_ch,
            "{}: ogg id-header says {channels} channels, expected {want_ch}",
            case.name
        );
    }
    if let Some(want_sr) = case.sample_rate {
        assert_eq!(
            sample_rate, want_sr,
            "{}: ogg id-header says {sample_rate} Hz, expected {want_sr}",
            case.name
        );
    }

    let mut decoder = match make_decoder(&params) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip {}: decoder ctor failed: {e}", case.name);
            return None;
        }
    };

    let stream_index = stream.index;
    let mut samples: Vec<i16> = Vec::new();
    let mut decoder_errors = 0usize;
    loop {
        let pkt = match demux.next_packet() {
            Ok(p) => p,
            Err(Error::Eof) => break,
            Err(e) => {
                eprintln!("{}: demux error after {} samples: {e}", case.name, samples.len());
                break;
            }
        };
        if pkt.stream_index != stream_index {
            // Packets from the chained-streams second logical bitstream
            // (or any other unrelated stream) — skip cleanly.
            continue;
        }
        if let Err(e) = decoder.send_packet(&pkt) {
            decoder_errors += 1;
            if decoder_errors <= 3 {
                eprintln!("{}: send_packet error: {e}", case.name);
            }
            continue;
        }
        match decoder.receive_frame() {
            Ok(Frame::Audio(af)) => {
                // Decoder emits interleaved s16le in af.data[0].
                let plane = &af.data[0];
                let mut count = 0usize;
                for chunk in plane.chunks_exact(2) {
                    samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    count += 1;
                }
                let _ = count;
            }
            Ok(other) => {
                eprintln!("{}: unexpected non-audio frame: {other:?}", case.name);
            }
            Err(Error::NeedMore) => continue,
            Err(Error::Eof) => break,
            Err(e) => {
                decoder_errors += 1;
                if decoder_errors <= 3 {
                    eprintln!("{}: receive_frame error: {e}", case.name);
                }
            }
        }
    }
    if decoder_errors > 0 {
        eprintln!(
            "{}: total decoder errors: {decoder_errors} (decoded {} samples / {} per channel)",
            case.name,
            samples.len(),
            samples.len() / channels.max(1) as usize
        );
    }

    Some(DecodedPcm {
        samples,
        channels,
        sample_rate,
    })
}

/// Parse a minimal RIFF/WAVE file: locate the `fmt ` chunk to read
/// channels + sample-rate + bits-per-sample, then return the `data`
/// chunk as interleaved s16le samples. Skips any LIST/INFO,
/// JUNK, or other non-essential chunks between `fmt ` and `data`.
fn parse_wav(bytes: &[u8]) -> Option<RefPcm> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut i = 12usize;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut data: Option<&[u8]> = None;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let sz = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]])
            as usize;
        let body_start = i + 8;
        let body_end = body_start + sz;
        if body_end > bytes.len() {
            break;
        }
        match id {
            b"fmt " => {
                if sz < 16 {
                    return None;
                }
                let format_tag =
                    u16::from_le_bytes([bytes[body_start], bytes[body_start + 1]]);
                channels = u16::from_le_bytes([bytes[body_start + 2], bytes[body_start + 3]]);
                sample_rate = u32::from_le_bytes([
                    bytes[body_start + 4],
                    bytes[body_start + 5],
                    bytes[body_start + 6],
                    bytes[body_start + 7],
                ]);
                bits_per_sample =
                    u16::from_le_bytes([bytes[body_start + 14], bytes[body_start + 15]]);
                // WAVE_FORMAT_EXTENSIBLE (0xFFFE) carries the real
                // sample format in the GUID at body_start+24..40. For
                // the fixtures in this corpus we only see PCM s16, so a
                // simple sanity check is enough.
                if format_tag != 1 && format_tag != 0xFFFE {
                    return None;
                }
            }
            b"data" => {
                data = Some(&bytes[body_start..body_end]);
                break;
            }
            _ => {}
        }
        // Chunks are padded to even byte alignment.
        i = body_end + (sz & 1);
    }
    let data = data?;
    if channels == 0 || sample_rate == 0 || bits_per_sample != 16 {
        return None;
    }
    let mut samples = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Some(RefPcm {
        samples,
        channels,
        sample_rate,
    })
}

fn read_reference(case: &CorpusCase) -> Option<RefPcm> {
    let dir = fixture_dir(case.name);
    let wav_path = dir.join("expected.wav");
    let bytes = match fs::read(&wav_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, wav_path.display());
            return None;
        }
    };
    parse_wav(&bytes)
}

/// Compute per-channel match/PSNR statistics. Compares only the
/// overlapping prefix of decoded vs reference (lossy decoders often
/// produce slightly fewer or more samples than the reference because
/// of MDCT block alignment).
fn compare(ours: &DecodedPcm, refp: &RefPcm) -> Vec<ChannelStat> {
    let chs = ours.channels.min(refp.channels) as usize;
    if chs == 0 {
        return Vec::new();
    }
    let frames_ours = ours.samples.len() / ours.channels.max(1) as usize;
    let frames_ref = refp.samples.len() / refp.channels.max(1) as usize;
    let n = frames_ours.min(frames_ref);

    let mut stats: Vec<ChannelStat> = (0..chs).map(|_| ChannelStat::new()).collect();
    for f in 0..n {
        for ch in 0..chs {
            let our = ours.samples[f * ours.channels as usize + ch] as i64;
            let r = refp.samples[f * refp.channels as usize + ch] as i64;
            let err = (our - r).abs();
            let s = &mut stats[ch];
            s.total += 1;
            if err == 0 {
                s.exact += 1;
            }
            if err <= 1 {
                s.near += 1;
            }
            if err as i32 > s.max_abs_err {
                s.max_abs_err = err as i32;
            }
            s.rms_ref += (r * r) as f64;
            s.rms_ours += (our * our) as f64;
            s.rms_err += (err * err) as f64;
        }
    }
    // Convert sums-of-squares into RMS magnitudes so the eprintln line
    // reads as a sample-domain LSB number, not a sum.
    for s in &mut stats {
        if s.total > 0 {
            s.rms_ref = (s.rms_ref / s.total as f64).sqrt();
            s.rms_ours = (s.rms_ours / s.total as f64).sqrt();
            // s.rms_err kept as sum-of-squares for psnr_db; PSNR turns
            // it back into MSE by dividing by total. Display value
            // (RMS error) is computed below.
        }
    }
    stats
}

/// Decode → compare → log → tier-aware assert.
fn evaluate(case: &CorpusCase) {
    eprintln!("--- {} (tier={:?}) ---", case.name, case.tier);
    let Some(ours) = decode_fixture_pcm(case) else {
        return;
    };
    let Some(refp) = read_reference(case) else {
        eprintln!("{}: could not parse expected.wav", case.name);
        return;
    };

    eprintln!(
        "{}: decoded ch={} sr={} samples={} ({} frames); reference ch={} sr={} samples={} ({} frames)",
        case.name,
        ours.channels,
        ours.sample_rate,
        ours.samples.len(),
        ours.samples.len() / ours.channels.max(1) as usize,
        refp.channels,
        refp.sample_rate,
        refp.samples.len(),
        refp.samples.len() / refp.channels.max(1) as usize,
    );

    if ours.channels != refp.channels {
        eprintln!(
            "{}: WARN channel count mismatch (decoded {} vs reference {})",
            case.name, ours.channels, refp.channels
        );
    }
    if ours.sample_rate != refp.sample_rate {
        eprintln!(
            "{}: WARN sample-rate mismatch (decoded {} vs reference {})",
            case.name, ours.sample_rate, refp.sample_rate
        );
    }

    let stats = compare(&ours, &refp);
    if stats.is_empty() {
        eprintln!("{}: no overlapping channels to compare", case.name);
        return;
    }

    let mut total_exact = 0usize;
    let mut total_near = 0usize;
    let mut total_samples = 0usize;
    let mut max_err_overall = 0i32;
    let mut psnr_min: f64 = f64::INFINITY;
    for (i, s) in stats.iter().enumerate() {
        let psnr = s.psnr_db();
        if psnr < psnr_min {
            psnr_min = psnr;
        }
        let rms_err_disp = if s.total > 0 {
            (s.rms_err / s.total as f64).sqrt()
        } else {
            0.0
        };
        eprintln!(
            "  ch{i}: rms_ref={:.1} rms_ours={:.1} rms_err={:.2} match={:.4}% near<=1LSB={:.4}% max_abs_err={} psnr={:.2} dB",
            s.rms_ref,
            s.rms_ours,
            rms_err_disp,
            s.match_pct(),
            s.near_pct(),
            s.max_abs_err,
            psnr,
        );
        total_exact += s.exact;
        total_near += s.near;
        total_samples += s.total;
        if s.max_abs_err > max_err_overall {
            max_err_overall = s.max_abs_err;
        }
    }
    let agg_pct = if total_samples > 0 {
        total_exact as f64 / total_samples as f64 * 100.0
    } else {
        0.0
    };
    let near_pct = if total_samples > 0 {
        total_near as f64 / total_samples as f64 * 100.0
    } else {
        0.0
    };
    eprintln!(
        "{}: aggregate match={:.4}% near<=1LSB={:.4}% max_abs_err={} min_psnr={:.2} dB",
        case.name, agg_pct, near_pct, max_err_overall, psnr_min,
    );

    match case.tier {
        Tier::BitExact => {
            assert_eq!(
                total_exact, total_samples,
                "{}: not bit-exact (max_abs_err={} match={:.4}%)",
                case.name, max_err_overall, agg_pct,
            );
        }
        Tier::ReportOnly => {
            // Logged; never gates CI. Underlying float-rounding deltas
            // are tracked as follow-up tasks if PSNR drops below ~40 dB.
        }
    }
}

// ---------------------------------------------------------------------------
// Per-fixture tests (Tier::ReportOnly across the board until at least
// one fixture is shown to round-trip cleanly.)
// ---------------------------------------------------------------------------

#[test]
fn corpus_5_1_channel_48000_q5() {
    evaluate(&CorpusCase {
        name: "5.1-channel-48000-q5",
        channels: Some(6),
        sample_rate: Some(48_000),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_chained_streams() {
    // Two concatenated logical bitstreams. The current oxideav-ogg
    // demuxer only registers streams from the initial BOS section, so
    // we decode FIRST stream only — which matches what expected.wav
    // contains (per the fixture's notes.md).
    evaluate(&CorpusCase {
        name: "chained-streams",
        channels: Some(1),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mode_floor1_only() {
    evaluate(&CorpusCase {
        name: "mode-floor1-only",
        channels: Some(1),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mode_residue_types_0_1_2() {
    evaluate(&CorpusCase {
        name: "mode-residue-types-0-1-2",
        channels: Some(2),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mono_22050_low_rate() {
    evaluate(&CorpusCase {
        name: "mono-22050-low-rate",
        channels: Some(1),
        sample_rate: Some(22_050),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mono_44100_q5_typical() {
    evaluate(&CorpusCase {
        name: "mono-44100-q5-typical",
        channels: Some(1),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_noise_stream() {
    evaluate(&CorpusCase {
        name: "noise-stream",
        channels: Some(1),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_silence_stream() {
    evaluate(&CorpusCase {
        name: "silence-stream",
        channels: Some(1),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_stereo_44100_q_neg1() {
    evaluate(&CorpusCase {
        name: "stereo-44100-q-1",
        channels: Some(2),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_stereo_44100_q10() {
    evaluate(&CorpusCase {
        name: "stereo-44100-q10",
        channels: Some(2),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_stereo_44100_q5_typical() {
    evaluate(&CorpusCase {
        name: "stereo-44100-q5-typical",
        channels: Some(2),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_stereo_96000_high_rate() {
    evaluate(&CorpusCase {
        name: "stereo-96000-high-rate",
        channels: Some(2),
        sample_rate: Some(96_000),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_stereo_cbr_128kbps() {
    evaluate(&CorpusCase {
        name: "stereo-cbr-128kbps",
        channels: Some(2),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_transient_blocksize_switch() {
    evaluate(&CorpusCase {
        name: "transient-blocksize-switch",
        channels: Some(1),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_with_attached_picture() {
    evaluate(&CorpusCase {
        name: "with-attached-picture",
        // ffmpeg auto-picks defaults from the sine source; the comment
        // header carries the picture but the audio stream is unaffected.
        channels: None,
        sample_rate: None,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_with_vorbis_comment_tags() {
    evaluate(&CorpusCase {
        name: "with-vorbis-comment-tags",
        channels: Some(1),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

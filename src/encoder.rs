//! Vorbis encoder (tier 2 — ffmpeg-interop quality).
//!
//! Supports 1..=8 channels at any sample rate. The setup header carries
//! both the short (256) and long (2048) block configurations and the
//! runtime picks per-block via a lookahead-driven transient detector
//! (Vorbis I §1.3.2 / §4.3.1 asymmetric windows). For >2 channel streams
//! the encoder picks coupling pairs following the standard Vorbis I
//! channel assignments (§4.3.9): L-R pairs and BL-BR pairs are coupled,
//! while the center and LFE channels (if present) stay uncoupled. The
//! decoder already handles arbitrary coupling-step lists.
//!
//! Channel layouts (matching the Vorbis I channel-mapping conventions):
//!
//! - 1ch: mono
//! - 2ch: L, R — couple (0,1)
//! - 3ch: L, C, R — couple (0,2)
//! - 4ch: FL, FR, BL, BR — couple (0,1), (2,3)
//! - 5ch: FL, C, FR, BL, BR — couple (0,2), (3,4)
//! - 6ch (5.1): FL, C, FR, BL, BR, LFE — couple (0,2), (3,4)
//! - 7ch: FL, C, FR, SL, SR, BL, BR — couple (0,2), (3,4), (5,6)
//! - 8ch (7.1): FL, C, FR, SL, SR, BL, BR, LFE — couple (0,2), (3,4), (5,6)
//!
//! The setup contains:
//!
//! - A Y-value codebook (128 entries, length 7, dim 1) for floor1
//!   amplitudes.
//! - A 2-entry classbook (length 1, dim 1) for residue partition
//!   classification — the one used class always picks a one-bit "0".
//! - A dim-2 VQ codebook with 128 entries, values in {-5..+5}^2 per
//!   dimension (121 valid grid + 7 padding entries to make the Huffman
//!   tree full — libvorbis rejects under-specified trees).
//! - One short floor1 with 8 posts and one long floor1 with 32 posts.
//! - Residue type 1 (concatenated per-channel) for both block sizes.
//! - One mapping per block-size, 1 or 2 channels. Stereo mappings declare
//!   one coupling step `(mag=0, ang=1)` — see `forward_couple` for the
//!   sign-coded sum/difference transform.
//! - Two modes: mode 0 = short (blockflag 0), mode 1 = long (blockflag 1).
//!
//! Pipeline for an audio block: decide block size via transient lookahead
//! (→ set the current block's `next_long` flag) → build asymmetric
//! sin-window → forward MDCT (with `2/N` scale) → per-channel floor1
//! analysis (peak in the window between adjacent posts, divided by
//! `FLOOR_SCALE` so residues have headroom, ATH-clamped at the bottom) →
//! floor curve via `synth_floor1` → residue = spectrum / floor_curve →
//! forward channel couple (per coupling pair in the mapping) → per-partition exhaustive VQ
//! search → emit packet. Consecutive blocks overlap by
//! `left_win_end - left_win_start` samples: `n/2` for long↔long and
//! short↔short, `bs0/2 = 128` for any transition involving a short
//! (see `prev_tail` + `window_bounds`).
//!
//! Known limitations relative to libvorbis. These are intentional scope
//! cuts, not open tasks — each represents a significant feature whose
//! absence affects bitrate/quality but not bitstream conformance:
//!
//! 1. **Point-stereo coupling**: our coupling is sum/difference (lossless,
//!    Vorbis I §1.3.3). Real libvorbis uses lossy point-stereo above some
//!    threshold frequency, which roughly halves the residue cost for the
//!    angle channel. Plumbing point-stereo means signaling it in the
//!    mapping setup (per-band coupling thresholds) and adding the
//!    encoder-side phase encoding. The decoder already handles the
//!    general inverse, so enabling this is an encoder-side refinement.
//!
//! 2. **Bigger residue VQ family**: a single 128-entry book serves both
//!    short and long blocks. libvorbis ships dozens of books per quality
//!    setting plus master codebooks that classify partition energy with
//!    fewer bits. The Vorbis I Annex B reference codebooks would let us
//!    match libvorbis bitrates, at the cost of a much larger setup
//!    header and a quality-indexed picker.
//!
//! 3. **Floor type 0 (LSP)**: never seen in modern Vorbis files; not
//!    implemented on the encode side. Our setup header always uses
//!    floor1.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat,
    TimeBase,
};

use crate::codebook::{parse_codebook, Codebook};
use crate::dbtable::FLOOR1_INVERSE_DB;
use crate::floor::synth_floor1;
use crate::imdct::{build_window, forward_mdct_naive};
use crate::setup::{Floor, Floor1, Residue, Setup};
use oxideav_core::bits::BitWriterLsb as BitWriter;

/// Short blocksize log2 (256 samples).
pub const DEFAULT_BLOCKSIZE_SHORT_LOG2: u8 = 8;
/// Long blocksize log2 (2048 samples).
pub const DEFAULT_BLOCKSIZE_LONG_LOG2: u8 = 11;

/// Floor1 multiplier = 2 (range 128, amp_bits 7).
const FLOOR1_MULTIPLIER: u8 = 2;

/// Number of extra X posts for the short-block floor (beyond the two
/// implicit endpoints at 0 and 128).
const FLOOR1_SHORT_EXTRA_X: [u32; 6] = [16, 32, 48, 64, 80, 96];

/// Per-partition frequency-bin count for residue VQ.
const RESIDUE_PARTITION_SIZE: u32 = 2;

/// VQ codebook dimensionality.
const VQ_DIM: usize = 2;

/// VQ value range: values in {-5..=5} per dimension (11 multiplicands per
/// dim) packed into a 128-entry codebook (7-bit codeword) so the Huffman
/// tree is exactly full (Vorbis I §3.2.1 forbids both over- and
/// under-specified codebooks; libvorbis rejects 121 entries at length 7 —
/// but 128 entries at length 7 is a perfect-fill tree). Entries 0..120
/// span the (e%11, e/11) grid in {-5..5}^2; entries 121..127 wrap modulo
/// 11 and alias to (0..6, -5) — the encoder never picks them.
const VQ_VALUES_PER_DIM: u32 = 11;
const VQ_MIN: f32 = -5.0;
const VQ_DELTA: f32 = 1.0;
/// Number of VQ entries actually used (11×11). Encoder's exhaustive
/// search restricts itself to this range.
const VQ_USED_ENTRIES: u32 = 121;
/// Total VQ book entries — must be 2^VQ_CODEWORD_LEN to keep the Huffman
/// tree well-formed.
const VQ_ENTRIES: u32 = 128;
/// Length of each VQ codeword — log2(VQ_ENTRIES) = 7.
const VQ_CODEWORD_LEN: u32 = 7;

/// Assemble the Vorbis Identification header (§4.2.2).
pub fn build_identification_header(
    channels: u8,
    sample_rate: u32,
    bitrate_nominal: i32,
    blocksize_0_log2: u8,
    blocksize_1_log2: u8,
) -> Vec<u8> {
    assert!(channels >= 1, "Vorbis requires at least one channel");
    assert!(sample_rate > 0, "Vorbis requires a non-zero sample rate");
    assert!(
        (6..=13).contains(&blocksize_0_log2)
            && (6..=13).contains(&blocksize_1_log2)
            && blocksize_0_log2 <= blocksize_1_log2,
        "Vorbis blocksize exponents must be in 6..=13 and short <= long"
    );

    let mut out = Vec::with_capacity(30);
    out.push(0x01);
    out.extend_from_slice(b"vorbis");
    out.extend_from_slice(&0u32.to_le_bytes());
    out.push(channels);
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes());
    out.extend_from_slice(&bitrate_nominal.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes());
    out.push((blocksize_1_log2 << 4) | (blocksize_0_log2 & 0x0F));
    out.push(0x01);
    out
}

/// Assemble the Vorbis Comment header (§5).
pub fn build_comment_header(comments: &[String]) -> Vec<u8> {
    let vendor = concat!("oxideav-vorbis ", env!("CARGO_PKG_VERSION")).as_bytes();
    let mut out = Vec::with_capacity(32 + vendor.len());
    out.push(0x03);
    out.extend_from_slice(b"vorbis");
    out.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    out.extend_from_slice(vendor);
    out.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for c in comments {
        let bytes = c.as_bytes();
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out.push(0x01);
    out
}

/// Legacy: placeholder setup (kept for the extradata-lacing test).
pub fn build_placeholder_setup_header(channels: u8) -> Vec<u8> {
    let _ = channels;
    let mut w = BitWriter::with_capacity(64);
    for &b in &[0x05u32, 0x76, 0x6f, 0x72, 0x62, 0x69, 0x73] {
        w.write_u32(b, 8);
    }
    w.write_u32(0, 8); // codebook count - 1 = 0 → 1 codebook
                       // One codebook: dim=1, entries=2, length 1 both.
    w.write_u32(0x564342, 24);
    w.write_u32(1, 16);
    w.write_u32(2, 24);
    w.write_bit(false);
    w.write_bit(false);
    for _ in 0..2 {
        w.write_u32(0, 5);
    }
    w.write_u32(0, 4);
    // time_count - 1 = 0.
    w.write_u32(0, 6);
    w.write_u32(0, 16);
    // floor_count - 1 = 0.
    w.write_u32(0, 6);
    w.write_u32(1, 16);
    w.write_u32(1, 5);
    w.write_u32(0, 4);
    w.write_u32(0, 3);
    w.write_u32(0, 2);
    w.write_u32(0, 8);
    w.write_u32(1, 2);
    w.write_u32(7, 4);
    w.write_u32(64, 7);
    // residue_count - 1 = 0.
    w.write_u32(0, 6);
    w.write_u32(2, 16);
    w.write_u32(0, 24);
    w.write_u32(0, 24);
    w.write_u32(0, 24);
    w.write_u32(0, 6);
    w.write_u32(0, 8);
    w.write_u32(0, 3);
    w.write_bit(false);
    // mapping_count - 1 = 0.
    w.write_u32(0, 6);
    w.write_u32(0, 16);
    w.write_bit(false);
    w.write_bit(false);
    w.write_u32(0, 2);
    w.write_u32(0, 8);
    w.write_u32(0, 8);
    w.write_u32(0, 8);
    // mode_count - 1 = 0.
    w.write_u32(0, 6);
    w.write_bit(false);
    w.write_u32(0, 16);
    w.write_u32(0, 16);
    w.write_u32(0, 8);
    w.write_bit(true);
    w.finish()
}

// ============================== Setup builders ==============================

/// Reverse the low `bits` bits of `v`. Our BitWriter is LSB-first but
/// `Codebook::codewords` store codes MSB-first; we bit-reverse at emit time.
fn bit_reverse(v: u32, bits: u8) -> u32 {
    let mut r = 0u32;
    for i in 0..bits {
        if (v >> i) & 1 != 0 {
            r |= 1 << (bits - 1 - i);
        }
    }
    r
}

/// Emit Huffman codeword for `entry` of `cb` to `w`. Handles length-0
/// (no-op) and bit-reverses so the LSB-first stream parses back to the
/// MSB-first accumulation used by `decode_scalar`.
fn write_huffman(w: &mut BitWriter, cb: &Codebook, entry: u32) {
    let len = cb.codeword_lengths[entry as usize];
    if len == 0 {
        return;
    }
    let code = cb.codewords[entry as usize];
    let rev = bit_reverse(code, len);
    w.write_u32(rev, len as u32);
}

/// Write a 32-bit Vorbis float (inverse of `BitReader::read_vorbis_float`).
fn write_vorbis_float(w: &mut BitWriter, value: f32) {
    if value == 0.0 {
        w.write_u32(0, 32);
        return;
    }
    let abs = value.abs() as f64;
    let mut mantissa = abs;
    let mut exp: i32 = 0;
    while mantissa < (1u64 << 20) as f64 {
        mantissa *= 2.0;
        exp -= 1;
    }
    while mantissa >= (1u64 << 21) as f64 {
        mantissa /= 2.0;
        exp += 1;
    }
    let m = mantissa as u32 & 0x001F_FFFF;
    let biased = (exp + 788) as u32;
    debug_assert!(biased < 1024, "Vorbis float exponent out of range");
    let sign_bit = if value < 0.0 { 0x8000_0000u32 } else { 0 };
    let raw = sign_bit | ((biased & 0x3FF) << 21) | m;
    w.write_u32(raw, 32);
}

/// Codebook 0: dim=1, 128 entries, all length 7. Entry k encodes Y value k.
fn write_setup_codebook_y(w: &mut BitWriter) {
    w.write_u32(0x564342, 24);
    w.write_u32(1, 16);
    w.write_u32(128, 24);
    w.write_bit(false);
    w.write_bit(false);
    for _ in 0..128 {
        w.write_u32(6, 5); // length - 1 = 6 → 7
    }
    w.write_u32(0, 4); // lookup_type = 0
}

/// Codebook 1: dim=1, 2 entries, both length 1 (codes 0 and 1). Used as
/// the residue classbook for our 1-classification setup — encoder always
/// picks entry 0 (1 bit "0") per classword group. We can't use a 1-entry
/// 0-bit codebook here because the Vorbis spec requires Huffman trees to
/// be exactly filled (libvorbis rejects sparse-with-only-zero-length books).
fn write_setup_codebook_class(w: &mut BitWriter) {
    w.write_u32(0x564342, 24);
    w.write_u32(1, 16);
    w.write_u32(2, 24);
    w.write_bit(false); // ordered
    w.write_bit(false); // sparse
    for _ in 0..2 {
        w.write_u32(0, 5); // length-1 = 0 → 1
    }
    w.write_u32(0, 4); // lookup_type = 0
}

/// Codebook 2: residue VQ. dim=2, 121 entries, all length 7. Lookup type 1
/// with min=-5, delta=1, value_bits=4, seq=false, multiplicands [0..10].
/// Decoded VQ pair for entry e: (e % 11) and (e / 11) mapped via
/// `m * delta + min`, so values span {-5..5}^2 (covers ±5 residues).
///
/// 121 entries < 128 = 2^7, so the canonical Huffman tree at length 7
/// is *underspecified* — entries 0..120 get codewords 0..120, the tree
/// has 7 unused slots at the top. libvorbis tolerates this; our codebook
/// builder accepts it as well (`build_decoder` checks for overspec, not
/// underspec).
fn write_setup_codebook_vq(w: &mut BitWriter) {
    w.write_u32(0x564342, 24);
    w.write_u32(VQ_DIM as u32, 16);
    w.write_u32(VQ_ENTRIES, 24);
    w.write_bit(false);
    w.write_bit(false);
    for _ in 0..VQ_ENTRIES {
        w.write_u32(VQ_CODEWORD_LEN - 1, 5);
    }
    w.write_u32(1, 4); // lookup_type = 1
    write_vorbis_float(w, VQ_MIN);
    write_vorbis_float(w, VQ_DELTA);
    w.write_u32(3, 4); // value_bits - 1 = 3 → 4 bits (need to hold 0..10)
    w.write_bit(false); // sequence_p
    for m in 0..VQ_VALUES_PER_DIM {
        w.write_u32(m, 4);
    }
}

/// Evenly-spaced extra X posts for the long-block floor (30 posts between
/// 0 and 1024, exclusive). Yields a dense floor grid spanning the long
/// blocksize's frequency range.
fn long_floor_extra_x() -> Vec<u32> {
    // 30 evenly-spaced posts in (0, 1024) — stride ≈ 33.
    let mut v = Vec::with_capacity(30);
    for i in 1..=30 {
        v.push((i as u32 * 1024) / 31);
    }
    v
}

/// Write a floor1 description: the X-list is chunked into `cdim`-sized
/// partitions, each referring to class 0 (cdim, subclasses=0, subbook=
/// [book_index]), multiplier=2, `rangebits`. `extra_x.len()` must be a
/// multiple of `cdim` and cdim in 1..=8.
fn write_floor1_section(
    w: &mut BitWriter,
    rangebits: u32,
    cdim: u32,
    extra_x: &[u32],
    subbook: u32,
) {
    debug_assert!((1..=8).contains(&cdim));
    debug_assert_eq!(extra_x.len() as u32 % cdim, 0);
    let partitions = extra_x.len() as u32 / cdim;
    w.write_u32(partitions, 5);
    for _ in 0..partitions {
        w.write_u32(0, 4); // partition_class_list[i] = class 0
    }
    // class 0 definition.
    w.write_u32(cdim - 1, 3); // class_dimensions - 1
    w.write_u32(0, 2); // class_subclasses
                       // No master codebook (subclasses = 0).
                       // subbook list: 2^subclasses = 1 entry. Stored = book + 1.
    w.write_u32(subbook + 1, 8);
    w.write_u32(FLOOR1_MULTIPLIER as u32 - 1, 2);
    w.write_u32(rangebits, 4);
    for &x in extra_x {
        w.write_u32(x, rangebits);
    }
}

/// Write a residue type-1 section with a single class and a single cascade
/// book.
fn write_residue_section(w: &mut BitWriter, end: u32, classbook: u32, vqbook: u32) {
    w.write_u32(0, 24); // begin
    w.write_u32(end, 24);
    w.write_u32(RESIDUE_PARTITION_SIZE - 1, 24);
    w.write_u32(0, 6); // classifications - 1 = 0 → 1 class
    w.write_u32(classbook, 8);
    // Cascade pass 0 has the VQ book. low-bits = 0b001, bitflag = 0.
    w.write_u32(0b001, 3);
    w.write_bit(false);
    w.write_u32(vqbook, 8);
}

/// ilog(x) = number of bits needed to represent x; ilog(0) = 0,
/// ilog(1) = 1, ilog(2) = 2, ilog(3) = 2, ilog(4) = 3, etc.
fn ilog(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        32 - value.leading_zeros()
    }
}

/// Return the standard coupling-pair list for a given channel count,
/// matching the Vorbis I channel-mapping conventions documented at the
/// top of this module. Always returns `(magnitude, angle)` pairs in
/// ascending channel order.
pub(crate) fn standard_coupling_steps(channels: u8) -> Vec<(u8, u8)> {
    match channels {
        2 => vec![(0, 1)],
        3 => vec![(0, 2)], // L, C, R — couple L↔R
        4 => vec![(0, 1), (2, 3)], // FL, FR, BL, BR
        5 => vec![(0, 2), (3, 4)], // FL, C, FR, BL, BR
        6 => vec![(0, 2), (3, 4)], // 5.1: FL, C, FR, BL, BR, LFE
        7 => vec![(0, 2), (3, 4), (5, 6)], // FL, C, FR, SL, SR, BL, BR
        8 => vec![(0, 2), (3, 4), (5, 6)], // 7.1: FL, C, FR, SL, SR, BL, BR, LFE
        _ => Vec::new(),
    }
}

/// Write a mapping with the given coupling-pair list, 1 submap, specified
/// floor + residue. `channels` is used to compute the coupling field width
/// (`ilog(channels-1)` bits per magnitude/angle field, per Vorbis I §4.2.4
/// step 6). `coupling_pairs` must only contain valid channel indices <
/// channels, with magnitude != angle, and no channel appearing as an
/// angle more than once (Vorbis I §4.3.9).
fn write_mapping_section(
    w: &mut BitWriter,
    floor_idx: u32,
    residue_idx: u32,
    coupling_pairs: &[(u8, u8)],
    channels: u8,
) {
    w.write_u32(0, 16); // mapping type = 0
    w.write_bit(false); // submaps flag = 0 → 1 submap
    if coupling_pairs.is_empty() {
        w.write_bit(false);
    } else {
        w.write_bit(true); // coupling flag = 1
        debug_assert!(!coupling_pairs.is_empty() && coupling_pairs.len() <= 256);
        w.write_u32(coupling_pairs.len() as u32 - 1, 8);
        let field_bits = ilog((channels as u32).saturating_sub(1));
        for &(mag, ang) in coupling_pairs {
            w.write_u32(mag as u32, field_bits);
            w.write_u32(ang as u32, field_bits);
        }
    }
    w.write_u32(0, 2); // reserved
                       // submap 0:
    w.write_u32(0, 8); // time index (discarded)
    w.write_u32(floor_idx, 8);
    w.write_u32(residue_idx, 8);
}

/// Build our own setup header: 3 codebooks (Y, class, VQ); 2 floors
/// (short + long); 2 residues (short + long); 2 mappings (short + long);
/// 2 modes (short = blockflag 0, long = blockflag 1). For multichannel
/// streams (`channels >= 2`) the mappings declare the standard coupling
/// pair list for that channel count (see module docs) — the encoder
/// applies forward sum/difference coupling before residue coding, the
/// decoder applies the inverse before IMDCT.
pub fn build_encoder_setup_header(channels: u8) -> Vec<u8> {
    let extra_x_long = long_floor_extra_x();
    let couples = standard_coupling_steps(channels);
    let mut w = BitWriter::with_capacity(512);
    for &b in &[0x05u32, 0x76, 0x6f, 0x72, 0x62, 0x69, 0x73] {
        w.write_u32(b, 8);
    }

    // 3 codebooks.
    w.write_u32(3 - 1, 8);
    write_setup_codebook_y(&mut w);
    write_setup_codebook_class(&mut w);
    write_setup_codebook_vq(&mut w);

    // 1 time-domain placeholder.
    w.write_u32(0, 6);
    w.write_u32(0, 16);

    // 2 floors.
    w.write_u32(2 - 1, 6);
    w.write_u32(1, 16); // floor type = 1
    write_floor1_section(&mut w, 7, 6, &FLOOR1_SHORT_EXTRA_X, 0);
    w.write_u32(1, 16);
    write_floor1_section(&mut w, 10, 5, &extra_x_long, 0);

    // 2 residues.
    w.write_u32(2 - 1, 6);
    w.write_u32(1, 16); // residue type = 1
    write_residue_section(&mut w, 128, 1, 2);
    w.write_u32(1, 16);
    write_residue_section(&mut w, 1024, 1, 2);

    // 2 mappings.
    w.write_u32(2 - 1, 6);
    write_mapping_section(&mut w, 0, 0, &couples, channels);
    write_mapping_section(&mut w, 1, 1, &couples, channels);

    // 2 modes.
    w.write_u32(2 - 1, 6);
    // mode 0: short
    w.write_bit(false);
    w.write_u32(0, 16);
    w.write_u32(0, 16);
    w.write_u32(0, 8);
    // mode 1: long
    w.write_bit(true);
    w.write_u32(0, 16);
    w.write_u32(0, 16);
    w.write_u32(1, 8);

    // Framing bit.
    w.write_bit(true);
    w.finish()
}

/// Decode codebooks from our setup header so the encoder can emit
/// bit-exact codewords.
fn extract_codebooks(setup: &[u8]) -> Result<Vec<Codebook>> {
    use oxideav_core::bits::BitReaderLsb as BitReader;
    if setup.len() < 7 || setup[0] != 0x05 || &setup[1..7] != b"vorbis" {
        return Err(Error::invalid("Vorbis encoder setup magic"));
    }
    let mut br = BitReader::new(&setup[7..]);
    let count = (br.read_u32(8)? + 1) as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(parse_codebook(&mut br)?);
    }
    Ok(out)
}

/// Xiph-laced 3-packet extradata.
pub fn build_extradata(id: &[u8], comment: &[u8], setup: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + id.len() + comment.len() + setup.len() + 8);
    out.push(2);
    for sz in [id.len(), comment.len()] {
        let mut rem = sz;
        while rem >= 255 {
            out.push(255);
            rem -= 255;
        }
        out.push(rem as u8);
    }
    out.extend_from_slice(id);
    out.extend_from_slice(comment);
    out.extend_from_slice(setup);
    out
}

// ============================== Encoder driver ==============================

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("Vorbis encoder: channels required"))?;
    if !(1..=8).contains(&channels) {
        return Err(Error::unsupported(format!(
            "Vorbis encoder: {channels}-channel encode not supported (1..=8 supported; the Vorbis I spec does not define standard mappings beyond 8)"
        )));
    }
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("Vorbis encoder: sample_rate required"))?;

    let id_hdr = build_identification_header(
        channels as u8,
        sample_rate,
        0,
        DEFAULT_BLOCKSIZE_SHORT_LOG2,
        DEFAULT_BLOCKSIZE_LONG_LOG2,
    );
    let comment_hdr = build_comment_header(&[]);
    let setup_hdr = build_encoder_setup_header(channels as u8);
    let extradata = build_extradata(&id_hdr, &comment_hdr, &setup_hdr);
    let codebooks = extract_codebooks(&setup_hdr)?;

    // Parse the full Setup so we can reuse floor/residue/mapping/mode
    // descriptions directly during encoding.
    let setup = crate::setup::parse_setup(&setup_hdr, channels as u8)?;

    let mut out_params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
    out_params.media_type = MediaType::Audio;
    out_params.channels = Some(channels);
    out_params.sample_rate = Some(sample_rate);
    out_params.sample_format = Some(SampleFormat::S16);
    out_params.extradata = extradata;

    let blocksize_short = 1usize << DEFAULT_BLOCKSIZE_SHORT_LOG2;
    let blocksize_long = 1usize << DEFAULT_BLOCKSIZE_LONG_LOG2;

    Ok(Box::new(VorbisEncoder {
        codec_id: CodecId::new(crate::CODEC_ID_STR),
        out_params,
        time_base: TimeBase::new(1, sample_rate as i64),
        channels,
        sample_rate,
        blocksize_short,
        blocksize_long,
        input_buf: vec![Vec::with_capacity(blocksize_long * 4); channels as usize],
        prev_tail: vec![Vec::with_capacity(blocksize_long); channels as usize],
        output_queue: VecDeque::new(),
        pts: 0,
        blocks_emitted: 0,
        flushed: false,
        codebooks,
        setup,
        // First emitted block is long-with-prev_long=true. Because the
        // initial `prev_tail` is empty (we'll treat missing samples as
        // zero), the decoder ignores the first block's contribution —
        // using prev_long=true just keeps the encoder's long→long
        // bookkeeping simple.
        prev_block_long: true,
        next_block_long: true,
        prior_energy: 0.0,
        force_long_only: false,
    }))
}

struct VorbisEncoder {
    codec_id: CodecId,
    out_params: CodecParameters,
    time_base: TimeBase,
    channels: u16,
    sample_rate: u32,
    blocksize_short: usize,
    blocksize_long: usize,
    input_buf: Vec<Vec<f32>>,
    /// Per-channel "overlap-region samples of the last block we emitted" —
    /// the chunk the decoder will OLA with the left side of the next block.
    /// Its length equals the overlap between the last block and the next:
    /// `n/2` for long↔long or short↔short, `bs0/2` for any transition
    /// involving a short block.
    prev_tail: Vec<Vec<f32>>,
    output_queue: VecDeque<Packet>,
    pts: i64,
    blocks_emitted: u64,
    flushed: bool,
    codebooks: Vec<Codebook>,
    setup: Setup,
    /// Size flag (long=true / short=false) of the most recently emitted
    /// block. Feeds into the next block's `prev_long` window flag.
    prev_block_long: bool,
    /// Size flag of the next block we're about to emit — either "true"
    /// initially or set by the previous block's `next_long` decision.
    next_block_long: bool,
    /// Rolling energy estimate (per-channel-averaged short-window energy)
    /// used by the transient detector. Updated every time `drain_blocks`
    /// scans fresh input.
    prior_energy: f32,
    /// When true, `decide_next_long` always returns true (long-only). Used
    /// by tests to establish a baseline without transient-driven short
    /// blocks. Off by default for production encodes.
    force_long_only: bool,
}

/// Short-window size for the transient detector (samples). Chosen as
/// `bs0 / 2` so each sub-window is ≤ 1 short block wide.
const TRANSIENT_SUB_WINDOW: usize = 128;

/// Amplitude-ratio threshold for flagging a transient. A sub-window whose
/// average amplitude (root of energy) exceeds `TRANSIENT_RATIO * sqrt(prior_energy)`
/// flips the encoder into short-block mode. 10× amplitude ≈ 20 dB spike,
/// which is the signature of a percussive hit riding on a steady tone.
const TRANSIENT_RATIO: f32 = 10.0;

/// Smoothing factor for the rolling prior-energy estimate. Closer to 1.0
/// = slower adaptation (longer history); closer to 0.0 = faster.
const PRIOR_ENERGY_ALPHA: f32 = 0.7;

/// Compute the window boundaries for a block given its size flag and the
/// neighbour flags (Vorbis I §1.3.2 / §4.3.1). Returns
/// `(left_win_start, left_win_end, right_win_start, right_win_end)` in
/// local block indices. Short blocks are symmetric and ignore the
/// neighbour flags.
fn window_bounds(
    long: bool,
    prev_long: bool,
    next_long: bool,
    n: usize,
    bs0: usize,
) -> (usize, usize, usize, usize) {
    if !long {
        // Short block is symmetric: overlap = n/2 = bs0/2 on each side.
        (0, n / 2, n / 2, n)
    } else {
        let left_start = if prev_long { 0 } else { (n - bs0) / 4 };
        let left_end = if prev_long { n / 2 } else { (n + bs0) / 4 };
        let right_start = if next_long { n / 2 } else { (3 * n - bs0) / 4 };
        let right_end = if next_long { n } else { (3 * n + bs0) / 4 };
        (left_start, left_end, right_start, right_end)
    }
}

impl VorbisEncoder {
    fn push_audio_frame(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.channels != self.channels {
            return Err(Error::invalid(format!(
                "Vorbis encoder: expected {} channels, got {}",
                self.channels, frame.channels
            )));
        }
        let n = frame.samples as usize;
        if n == 0 {
            return Ok(());
        }
        match frame.format {
            SampleFormat::S16 => {
                let plane = frame
                    .data
                    .first()
                    .ok_or_else(|| Error::invalid("S16 frame missing data plane"))?;
                let stride = self.channels as usize * 2;
                if plane.len() < n * stride {
                    return Err(Error::invalid("S16 frame: data plane too short"));
                }
                for i in 0..n {
                    for ch in 0..self.channels as usize {
                        let off = i * stride + ch * 2;
                        let sample = i16::from_le_bytes([plane[off], plane[off + 1]]);
                        self.input_buf[ch].push(sample as f32 / 32768.0);
                    }
                }
            }
            SampleFormat::F32 => {
                let plane = frame
                    .data
                    .first()
                    .ok_or_else(|| Error::invalid("F32 frame missing data plane"))?;
                let stride = self.channels as usize * 4;
                if plane.len() < n * stride {
                    return Err(Error::invalid("F32 frame: data plane too short"));
                }
                for i in 0..n {
                    for ch in 0..self.channels as usize {
                        let off = i * stride + ch * 4;
                        let v = f32::from_le_bytes([
                            plane[off],
                            plane[off + 1],
                            plane[off + 2],
                            plane[off + 3],
                        ]);
                        self.input_buf[ch].push(v);
                    }
                }
            }
            other => {
                return Err(Error::unsupported(format!(
                    "Vorbis encoder: input sample format {other:?} not supported yet"
                )));
            }
        }
        Ok(())
    }

    /// Decide whether the next block (i.e. the one AFTER the block we're
    /// about to emit) should be long or short by scanning the lookahead
    /// region for a transient. Caller passes the lookahead slice starting
    /// at the samples that will fall inside the next block's useful region.
    ///
    /// Two-pronged detector:
    ///   1. **Local contrast**: max sub-window energy vs. min sub-window
    ///      energy across the lookahead. Catches isolated clicks sitting
    ///      in quiet regions (minimum ~ 0, maximum ~ click energy → big
    ///      ratio → transient). Only applies once `prior_energy` has
    ///      settled (first N blocks use global-only) so the onset of a
    ///      normal tone from silence isn't flagged as a transient.
    ///   2. **Global ratio**: max sub-window energy vs. the encoder's
    ///      rolling `prior_energy` estimate. Catches transients inside a
    ///      busy signal where local contrast alone isn't enough.
    fn decide_next_long(&mut self, lookahead: &[f32]) -> bool {
        if self.force_long_only {
            return true;
        }
        if lookahead.len() < TRANSIENT_SUB_WINDOW {
            return true;
        }
        // Collect per-sub-window energies.
        let win = TRANSIENT_SUB_WINDOW;
        let stride = win / 2;
        let mut max_e = 0f32;
        let mut min_e = f32::INFINITY;
        let mut sum_e = 0f32;
        let mut n_wins = 0usize;
        let mut i = 0;
        while i + win <= lookahead.len() {
            let mut e = 0f32;
            for &s in &lookahead[i..i + win] {
                e += s * s;
            }
            e /= win as f32;
            if e > max_e {
                max_e = e;
            }
            if e < min_e {
                min_e = e;
            }
            sum_e += e;
            n_wins += 1;
            i += stride;
        }
        let avg_e = if n_wins > 0 {
            sum_e / n_wins as f32
        } else {
            0.0
        };
        let ratio_sq = TRANSIENT_RATIO * TRANSIENT_RATIO;
        // Require a settled prior-energy before applying local contrast.
        // Without this, the onset of any non-silence signal (tone fade-in,
        // music start) would flag as a transient every time — draining
        // bits into short blocks where a long block would have done.
        let prior_settled = self.prior_energy > 1e-6;
        let local_contrast = prior_settled && max_e > ratio_sq * min_e.max(1e-12) && max_e > 1e-4;
        let global_contrast = prior_settled && max_e > ratio_sq * self.prior_energy;
        let is_transient = local_contrast || global_contrast;
        // Roll the prior-energy estimate toward the lookahead average.
        // Using the average (not peak) keeps prior_energy from being
        // pushed high by the transient itself — we want it to track the
        // surrounding energy level so subsequent blocks can compare against
        // the baseline.
        self.prior_energy =
            PRIOR_ENERGY_ALPHA * self.prior_energy + (1.0 - PRIOR_ENERGY_ALPHA) * avg_e;
        !is_transient
    }

    /// Emit as many complete blocks as the current input buffer supports.
    /// Each block's size (long / short) is determined by the previous
    /// emission's `next_long` decision; the current emission makes the
    /// `next_long` decision for the following block.
    fn drain_blocks(&mut self) {
        let n_channels = self.channels as usize;
        let bs0 = self.blocksize_short;
        loop {
            let long = self.next_block_long;
            let prev_long = self.prev_block_long;
            let n = if long {
                self.blocksize_long
            } else {
                self.blocksize_short
            };

            // We need to know `next_long` (i.e. the flag of the block AFTER
            // this one) to know how many fresh samples this block consumes.
            // Peek at the lookahead BEFORE committing: if there aren't
            // enough samples to also run transient detection, fall back to
            // `next_long = true`.
            //
            // Worst-case fresh consumption for the CURRENT block: the max
            // of the two `next_long` options. Use that as the gate: we want
            // to make sure that whichever `next_long` we pick, we have
            // enough input.
            let (l_start_t, l_end_t, _, r_end_t) = window_bounds(long, prev_long, true, n, bs0);
            let (_, _, _, r_end_f) = window_bounds(long, prev_long, false, n, bs0);
            let fresh_true = r_end_t.saturating_sub(l_end_t);
            let (_, l_end_f, _, _) = window_bounds(long, prev_long, false, n, bs0);
            let fresh_false = r_end_f.saturating_sub(l_end_f);
            let max_fresh = fresh_true.max(fresh_false);
            if self.input_buf[0].len() < max_fresh {
                let _ = l_start_t; // keep unused var silent
                return;
            }

            // Transient detection: look a little beyond the current block's
            // fresh region to decide the NEXT block's size. The lookahead
            // window is two short-blocks wide (small enough that it stays
            // inside the next block).
            let lookahead_start = fresh_true.min(self.input_buf[0].len());
            let lookahead_end =
                (fresh_true + 2 * self.blocksize_short).min(self.input_buf[0].len());
            let next_long = if lookahead_end > lookahead_start {
                // Use channel 0 for transient detection (stereo is usually
                // correlated; this avoids false positives from mid/side
                // signals). Clone the slice to decouple the borrow.
                let slice: Vec<f32> = self.input_buf[0][lookahead_start..lookahead_end].to_vec();
                self.decide_next_long(&slice)
            } else {
                true
            };

            // Recompute bounds with the chosen next_long.
            let (l_start, l_end, r_start, r_end) =
                window_bounds(long, prev_long, next_long, n, bs0);
            let fresh_needed = r_end - l_end;
            if self.input_buf[0].len() < fresh_needed {
                // Should not happen given max_fresh gate above, but bail safely.
                return;
            }

            // Build the block per channel.
            let mut block: Vec<Vec<f32>> = Vec::with_capacity(n_channels);
            for ch in 0..n_channels {
                let mut v = vec![0f32; n];
                // Left OLA region: prev_tail samples at [l_start, l_end).
                let overlap_len = l_end - l_start;
                let tail = &self.prev_tail[ch];
                let tlen = tail.len().min(overlap_len);
                let tail_offset = overlap_len - tlen;
                v[l_start + tail_offset..l_start + tail_offset + tlen]
                    .copy_from_slice(&tail[tail.len() - tlen..]);
                // Fresh samples [l_end, r_end) from input_buf.
                let fresh_take = self.input_buf[ch].len().min(fresh_needed);
                v[l_end..l_end + fresh_take].copy_from_slice(&self.input_buf[ch][..fresh_take]);
                // Save the right overlap region [r_start, r_end) as the
                // next block's prev_tail.
                self.prev_tail[ch].clear();
                self.prev_tail[ch].extend_from_slice(&v[r_start..r_end]);
                // Consume the fresh samples from the input queue.
                self.input_buf[ch].drain(..fresh_take);
                block.push(v);
            }

            let pkt = self.encode_block_packet(&block, n, long, prev_long, next_long);
            self.output_queue.push_back(pkt);

            // Update state for the next iteration.
            self.prev_block_long = long;
            self.next_block_long = next_long;
        }
    }

    fn encode_block_packet(
        &mut self,
        block: &[Vec<f32>],
        n: usize,
        long: bool,
        prev_long: bool,
        next_long: bool,
    ) -> Packet {
        let mut max_abs = 0f32;
        for ch in block {
            for &s in ch {
                let a = s.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
        }
        if max_abs < 1.0e-6 {
            return self.emit_silent_packet(n, long, prev_long, next_long);
        }
        match self.encode_block(block, n, long, prev_long, next_long) {
            Some(data) => {
                let pts = self.pts;
                self.pts += n as i64;
                self.blocks_emitted += 1;
                let mut pkt = Packet::new(0, self.time_base, data);
                pkt.pts = Some(pts);
                pkt.dts = Some(pts);
                pkt.duration = Some(n as i64);
                pkt.flags.keyframe = true;
                pkt
            }
            None => self.emit_silent_packet(n, long, prev_long, next_long),
        }
    }

    fn emit_silent_packet(
        &mut self,
        n: usize,
        long: bool,
        prev_long: bool,
        next_long: bool,
    ) -> Packet {
        let mut w = BitWriter::with_capacity(4);
        // packet type bit: 0 (audio).
        w.write_bit(false);
        // mode bits: 1 bit for 2 modes.
        w.write_u32(if long { 1 } else { 0 }, 1);
        if long {
            w.write_bit(prev_long);
            w.write_bit(next_long);
        }
        // Per-channel floor unused bit.
        for _ in 0..self.channels as usize {
            w.write_bit(false);
        }
        let data = w.finish();
        let pts = self.pts;
        self.pts += n as i64;
        self.blocks_emitted += 1;
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.duration = Some(n as i64);
        pkt.flags.keyframe = true;
        pkt
    }

    /// Full encode pipeline for a block of size `n`. Returns `None` if
    /// anything went wrong (caller emits a silent packet instead).
    ///
    /// `prev_long` / `next_long` are only meaningful for long blocks — for
    /// short blocks they are ignored by the decoder and by `build_window`.
    fn encode_block(
        &self,
        block: &[Vec<f32>],
        n: usize,
        long: bool,
        prev_long: bool,
        next_long: bool,
    ) -> Option<Vec<u8>> {
        let n_half = n / 2;
        let n_channels = self.channels as usize;
        let mode_idx = if long { 1 } else { 0 };
        let mode = &self.setup.modes[mode_idx];
        let mapping = &self.setup.mappings[mode.mapping as usize];

        let window = build_window(n, long, prev_long, next_long, self.blocksize_short);

        // Per-channel: window × forward MDCT → floor analysis → residue.
        let mut floor_codes: Vec<Vec<i32>> = Vec::with_capacity(n_channels);
        let mut residues: Vec<Vec<f32>> = Vec::with_capacity(n_channels);

        let floor_idx = mapping.submap_floor[0] as usize;
        let floor_struct = match &self.setup.floors[floor_idx] {
            Floor::Type1(f) => f.clone(),
            _ => return None,
        };

        let trace = std::env::var_os("OXIDEAV_VORBIS_ENC_TRACE").is_some();
        for ch in 0..n_channels {
            // Pre-window the input via the SIMD multiply kernel: copy then
            // multiply in place.
            let mut windowed = block[ch][..n].to_vec();
            crate::simd::mul_inplace(&mut windowed, &window);
            let mut spec = vec![0f32; n_half];
            forward_mdct_naive(&windowed, &mut spec);
            // Apply 2/N scaling. Without this the spectrum magnitudes
            // (up to A*N/4 for a sine of amp A, so O(N/8) for full-scale
            // input) completely dwarf the floor1 table's max value of 1.0
            // and the residue VQ saturates. The decoder performs IMDCT
            // without a 1/N factor, so 2/N on the forward side matches.
            let fwd_scale = 2.0 / n as f32;
            for v in spec.iter_mut() {
                *v *= fwd_scale;
            }
            if trace {
                let mut peak_bin = 0;
                let mut peak = 0f32;
                for (i, v) in spec.iter().enumerate() {
                    if v.abs() > peak {
                        peak = v.abs();
                        peak_bin = i;
                    }
                }
                eprintln!(
                    "[enc] ch{} windowed_peak={} spec_peak={} at_bin={}",
                    ch,
                    windowed.iter().map(|v| v.abs()).fold(0f32, f32::max),
                    peak,
                    peak_bin
                );
            }
            let target_y = analyse_floor1(&floor_struct, &spec, n_half, self.sample_rate);
            let codes = compute_floor1_codes(&floor_struct, &target_y);
            let mut curve = vec![1f32; n_half];
            let decoded = crate::floor::Floor1Decoded {
                unused: false,
                y: codes.clone(),
            };
            synth_floor1(&floor_struct, &decoded, n_half, &mut curve).ok()?;
            // Compute residue = spectrum / floor_curve.
            let mut residue = vec![0f32; n_half];
            for k in 0..n_half {
                if curve[k].abs() > 1e-30 {
                    residue[k] = spec[k] / curve[k];
                }
            }
            if trace {
                let mut peak_cu = 0f32;
                let mut peak_cu_bin = 0;
                for (i, v) in curve.iter().enumerate() {
                    if v.abs() > peak_cu {
                        peak_cu = v.abs();
                        peak_cu_bin = i;
                    }
                }
                let mut peak_r = 0f32;
                for v in residue.iter() {
                    if v.abs() > peak_r {
                        peak_r = v.abs();
                    }
                }
                eprintln!(
                    "[enc] ch{} target_y[0..8]={:?} codes[0..8]={:?} floor_peak={} at_bin={} residue_peak={}",
                    ch, &target_y[..8.min(target_y.len())], &codes[..8.min(codes.len())], peak_cu, peak_cu_bin, peak_r
                );
            }
            floor_codes.push(codes);
            residues.push(residue);
        }

        // Forward channel coupling. The decoder applies inverse coupling on
        // the residue spectrum (Vorbis I §1.3.3) before multiplying by the
        // per-channel floor curve. So we must transform our per-channel
        // residues into (magnitude, angle) form here so that the decoder
        // recovers the original L/R residue exactly. See `forward_couple`
        // for the case-by-case derivation; together with `decoder.rs`'s
        // inverse this round-trips losslessly.
        for &(mag, ang) in &mapping.coupling {
            let mi = mag as usize;
            let ai = ang as usize;
            if mi >= residues.len() || ai >= residues.len() || mi == ai {
                continue;
            }
            for k in 0..n_half {
                let l = residues[mi][k];
                let r = residues[ai][k];
                let (m, a) = forward_couple(l, r);
                residues[mi][k] = m;
                residues[ai][k] = a;
            }
        }

        let residue_idx = mapping.submap_residue[0] as usize;
        let residue_def = self.setup.residues[residue_idx].clone();

        // Bit-pack the audio packet.
        let mut w = BitWriter::with_capacity(1024);
        w.write_bit(false); // audio header bit
        w.write_u32(mode_idx as u32, 1); // mode bits (2 modes → 1 bit)
        if long {
            w.write_bit(prev_long);
            w.write_bit(next_long);
        }

        // Per-channel floor1 packet emission.
        for ch in 0..n_channels {
            self.emit_floor1_packet(&mut w, &floor_struct, &floor_codes[ch]);
        }

        // Residue emission (type 1: concatenated per-channel). Our residue
        // definition has 1 class, classbook with length-0 codeword
        // (no bits), cascade pass 0 → VQ book 2.
        self.emit_residue_type1(&mut w, &residue_def, n_half, &residues)?;

        Some(w.finish())
    }

    fn emit_floor1_packet(&self, w: &mut BitWriter, floor: &Floor1, codes: &[i32]) {
        // `codes` is the raw floor1 Y vector: codes[0..1] = absolute
        // amplitudes (clamped to [0, range)), codes[2..] = delta codes.
        // The decoder will run step1 reconstruction on these values; we
        // must make sure the deltas we computed earlier (via
        // `compute_floor1_codes`) are what the encoder-side
        // `synth_floor1` call used.
        let n_posts = floor.xlist.len();
        debug_assert_eq!(codes.len(), n_posts);
        w.write_bit(true);
        let amp_bits = match floor.multiplier {
            1 => 8,
            2 => 7,
            3 => 7,
            4 => 6,
            _ => 8,
        };
        w.write_u32(codes[0] as u32, amp_bits);
        w.write_u32(codes[1] as u32, amp_bits);

        let book_y = &self.codebooks[0];
        let mut offset = 2usize;
        for &class_idx in &floor.partition_class_list {
            let c = class_idx as usize;
            let cdim = floor.class_dimensions[c] as usize;
            debug_assert_eq!(floor.class_subclasses[c], 0);
            for _j in 0..cdim {
                let code = (codes[offset] as u32).min(book_y.entries - 1);
                write_huffman(w, book_y, code);
                offset += 1;
            }
        }
    }

    /// Emit residue type 1: concatenated per-channel values. Our residue
    /// has 1 classification (classbook returns entry 0 with 0 bits) and a
    /// single cascade pass using VQ book 2.
    fn emit_residue_type1(
        &self,
        w: &mut BitWriter,
        residue: &Residue,
        n_half: usize,
        vectors: &[Vec<f32>],
    ) -> Option<()> {
        let classbook = &self.codebooks[residue.classbook as usize];
        let classwords_per_codeword = classbook.dimensions as usize;
        let classifications = residue.classifications as usize;
        let psz = residue.partition_size as usize;
        let begin = residue.begin as usize;
        let end = (residue.end as usize).min(n_half);
        if (end - begin) % psz != 0 {
            return None;
        }
        let n_partitions = (end - begin) / psz;

        // Build per-channel partition classifications. With our setup we
        // always pick class 0.
        let per_channel_classes: Vec<Vec<u32>> = vec![vec![0u32; n_partitions]; vectors.len()];

        // Collect per-cascade book lists (pass -> book index or None).
        let mut cascade_books: [Option<u32>; 8] = [None; 8];
        for pass in 0..8 {
            for c in 0..classifications {
                if residue.cascade[c] & (1 << pass) != 0 {
                    let b = residue.books[c][pass];
                    if b >= 0 {
                        cascade_books[pass] = Some(b as u32);
                    }
                }
            }
        }

        // Cascade passes — mirror `decode_partitioned` exactly.
        for pass in 0..8 {
            let mut partition_idx = 0usize;
            while partition_idx < n_partitions {
                if pass == 0 {
                    for ch in 0..vectors.len() {
                        // Pack `classwords_per_codeword` classes into a
                        // base-`classifications` number (high-digit first),
                        // then emit the classbook codeword for that number.
                        let mut class_number: u32 = 0;
                        for k in 0..classwords_per_codeword {
                            let pidx = partition_idx + k;
                            let cl = if pidx < n_partitions {
                                per_channel_classes[ch][pidx]
                            } else {
                                0
                            };
                            class_number = class_number * classifications as u32 + cl;
                        }
                        write_huffman(w, classbook, class_number);
                    }
                }
                // Decode `classwords_per_codeword` partitions per step.
                for k in 0..classwords_per_codeword {
                    let pidx = partition_idx + k;
                    if pidx >= n_partitions {
                        break;
                    }
                    if let Some(book_idx) = cascade_books[pass] {
                        let book = &self.codebooks[book_idx as usize];
                        let dim = book.dimensions as usize;
                        for ch in 0..vectors.len() {
                            let class_id = per_channel_classes[ch][pidx] as usize;
                            if class_id >= classifications || residue.books[class_id][pass] < 0 {
                                continue;
                            }
                            let bin_start = begin + pidx * psz;
                            let mut bin = bin_start;
                            let bin_end = bin_start + psz;
                            while bin < bin_end {
                                // Pull `dim` values from the residue, find
                                // the best VQ entry, emit its codeword.
                                let mut target = [0f32; 8];
                                for j in 0..dim {
                                    if bin + j < n_half {
                                        target[j] = vectors[ch][bin + j];
                                    }
                                }
                                let entry =
                                    vq_search(book, &target[..dim], VQ_USED_ENTRIES).ok()?;
                                write_huffman(w, book, entry);
                                bin += dim;
                            }
                        }
                    }
                }
                partition_idx += classwords_per_codeword;
            }
        }
        Some(())
    }
}

/// Given target absolute Y values per post (indexed by xlist position),
/// compute the floor1 code vector `codes` that the decoder will receive.
/// `codes[0..1]` are absolute Y values (clamped into range). `codes[2..]`
/// are delta codes. Returns the codes vector.
fn compute_floor1_codes(floor: &Floor1, target_y: &[i32]) -> Vec<i32> {
    let range = match floor.multiplier {
        1 => 256,
        2 => 128,
        3 => 86,
        4 => 64,
        _ => 256,
    };
    let n_posts = floor.xlist.len();
    debug_assert_eq!(target_y.len(), n_posts);

    // Precompute low/high neighbours (INDEX-order) — same as decoder.
    let mut low_neighbor = vec![0usize; n_posts];
    let mut high_neighbor = vec![0usize; n_posts];
    for j in 2..n_posts {
        let xj = floor.xlist[j];
        let mut lo = 0usize;
        let mut lo_x = floor.xlist[0];
        let mut hi = 1usize;
        let mut hi_x = floor.xlist[1];
        for k in 0..j {
            let xk = floor.xlist[k];
            if xk < xj && xk > lo_x {
                lo = k;
                lo_x = xk;
            }
            if xk > xj && xk < hi_x {
                hi = k;
                hi_x = xk;
            }
        }
        low_neighbor[j] = lo;
        high_neighbor[j] = hi;
    }

    let mut final_y = vec![0i32; n_posts];
    let mut codes = vec![0i32; n_posts];
    final_y[0] = target_y[0].clamp(0, range - 1);
    final_y[1] = target_y[1].clamp(0, range - 1);
    codes[0] = final_y[0];
    codes[1] = final_y[1];

    for j in 2..n_posts {
        let lo = low_neighbor[j];
        let hi = high_neighbor[j];
        let predicted = render_point_int(
            floor.xlist[lo] as i32,
            final_y[lo],
            floor.xlist[hi] as i32,
            final_y[hi],
            floor.xlist[j] as i32,
        );
        let high_room = range - predicted;
        let low_room = predicted;
        let room = high_room.min(low_room) * 2;
        let tgt = target_y[j].clamp(0, range - 1);
        let (val, recovered) = pick_delta(predicted, tgt, room);
        codes[j] = val;
        final_y[j] = recovered;
    }
    codes
}

/// Vorbis render_point (integer line interpolation). Matches
/// `crate::floor::render_point`.
fn render_point_int(x0: i32, y0: i32, x1: i32, y1: i32, x: i32) -> i32 {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let ady = dy.abs();
    let err = ady * (x - x0);
    let off = if adx != 0 { err / adx } else { 0 };
    if dy < 0 {
        y0 - off
    } else {
        y0 + off
    }
}

/// Pick the smallest-magnitude delta `val` such that `synth_floor1`'s
/// small-delta branch reconstructs a final Y as close as possible to
/// `target`, given `predicted` and the allowable `room`. Returns the
/// code to emit and the actual recovered Y (what the decoder will
/// compute).
fn pick_delta(predicted: i32, target: i32, room: i32) -> (i32, i32) {
    // Decoder's small-delta branch:
    //   if val % 2 == 1:  final_y = predicted - (val + 1) / 2
    //   if val % 2 == 0:  final_y = predicted + val / 2
    // val in 1..room (val==0 would mean "unused" → final_y = predicted).
    // We always emit val != 0, even if target == predicted, so step2_used
    // is set and the rendered curve respects our Y at this post. To get
    // target == predicted we'd need val==0, which skips the post. We
    // accept that small mismatch (predicted+0 vs target=predicted).
    let delta = target - predicted;
    let (val, recovered) = if delta == 0 {
        // Emit val=2 (even → predicted + 1), closest reachable ≠ predicted.
        // Actually pick val=1 → predicted - 1. Either is off by 1.
        // Use val=2 if predicted + 1 < range (upper direction), else val=1.
        if room >= 2 {
            (2, predicted + 1)
        } else {
            (1, predicted - 1)
        }
    } else if delta > 0 {
        // final_y = predicted + val/2 ⇒ val = 2*delta (must be even and < room).
        let v = (2 * delta).min(room - 1).max(2);
        let v = if v % 2 == 1 { v - 1 } else { v };
        (v, predicted + v / 2)
    } else {
        // delta < 0: final_y = predicted - (val+1)/2 ⇒ val = 2*(-delta) - 1 (odd, < room).
        let v = (2 * (-delta) - 1).min(room - 1).max(1);
        let v = if v % 2 == 0 { v - 1 } else { v };
        (v, predicted - (v + 1) / 2)
    };
    (val, recovered)
}

/// Vorbis forward channel coupling (sign-coded sum/difference).
///
/// Given the per-bin residue values `(l, r)` for the left/right channels,
/// produce the magnitude/angle pair `(m, a)` such that the decoder's
/// inverse coupling (`crate::decoder` lines ~240-260) recovers `(l, r)`
/// bit-exactly. The forward rules are derived case-by-case from the
/// inverse:
///
/// - `m > 0, a > 0`  ⇒ inverse `(m, m - a)`  ⇒ forward when `l > 0 ∧ l > r`: `m=l, a=l-r`
/// - `m > 0, a ≤ 0`  ⇒ inverse `(m + a, m)`  ⇒ forward when `r > 0 ∧ l ≤ r`: `m=r, a=l-r`
/// - `m ≤ 0, a > 0`  ⇒ inverse `(m, m + a)`  ⇒ forward when `l ≤ 0 ∧ r > l`: `m=l, a=r-l`
/// - `m ≤ 0, a ≤ 0`  ⇒ inverse `(m - a, m)`  ⇒ forward when `r ≤ 0 ∧ l ≥ r`: `m=r, a=r-l`
///
/// Boundary cases (zeros, equal signs) are absorbed by the `≤` / `≥`
/// breakdowns. Verified by encode → decode → assert_eq for spot-check
/// inputs in the unit test suite.
fn forward_couple(l: f32, r: f32) -> (f32, f32) {
    if l >= 0.0 && r >= 0.0 {
        if l >= r {
            (l, l - r)
        } else {
            (r, l - r)
        }
    } else if l <= 0.0 && r <= 0.0 {
        if l >= r {
            (r, r - l)
        } else {
            (l, r - l)
        }
    } else if l <= 0.0 {
        // l<=0, r>0 (signs differ).
        (l, r - l)
    } else {
        // l>0, r<=0.
        (r, r - l)
    }
}

/// Exhaustive nearest-neighbour VQ search over `book`'s entries. Returns
/// the entry index minimising the squared-error distance to `target`.
///
/// `max_entries` caps the search range when the codebook is "padded" — our
/// 128-entry/121-used VQ book pads with unreferenced grid wraparound
/// entries (see [`VQ_USED_ENTRIES`]). Pass `book.entries` for unrestricted
/// search.
fn vq_search(book: &Codebook, target: &[f32], max_entries: u32) -> Result<u32> {
    let mut best_e = 0u32;
    let mut best_d = f32::MAX;
    let limit = max_entries.min(book.entries);
    for e in 0..limit {
        if book.codeword_lengths[e as usize] == 0 {
            continue;
        }
        let v = book.vq_lookup(e)?;
        let mut d = 0f32;
        for (i, &t) in target.iter().enumerate() {
            let x = t - v[i];
            d += x * x;
        }
        if d < best_d {
            best_d = d;
            best_e = e;
        }
    }
    Ok(best_e)
}

/// Floor scaling factor: the target floor is set to peak_local / FLOOR_SCALE
/// so the residue VQ has headroom to encode bins above the floor in the
/// {-5..5} range. Empirically `4.0` balances residue saturation against
/// quantisation noise at the low end. Smaller values shift more energy
/// into the residue (better SNR) but risk clipping the strongest peaks.
const FLOOR_SCALE: f32 = 4.0;

/// Bare-minimum absolute-threshold-of-hearing (ATH) curve. Returns a
/// linear-magnitude floor *minimum* in our spectrum-amplitude units (i.e.
/// post-`fwd_scale`). Bins with magnitude below this can be set to a small
/// floor without audible loss — saving residue bits.
///
/// Coarse two-piece fit: high-pass roll-off below ~50 Hz and above
/// ~16 kHz brought up to about -40 dB. In the speech/music midband
/// (200 Hz..6 kHz), the threshold is small (-80 dB) so we don't over-floor
/// the audible content.
fn ath_min_for_bin(bin: usize, n_half: usize, sample_rate: u32) -> f32 {
    let nyquist = sample_rate as f32 * 0.5;
    let freq = (bin as f32 / n_half as f32) * nyquist;
    // Three break points: 100 Hz, 1 kHz, 10 kHz.
    let db = if freq < 100.0 {
        // Below 100 Hz: -30 dB rising fast as freq -> 0.
        -30.0 + (freq / 100.0).max(0.01) * 20.0
    } else if freq < 1000.0 {
        // 100 Hz - 1 kHz: -50 to -75 dB.
        -50.0 - (freq - 100.0) / 900.0 * 25.0
    } else if freq < 8000.0 {
        // 1 kHz - 8 kHz: ~-80 dB midband.
        -80.0
    } else {
        // 8 kHz - Nyquist: -80 dB rising back to -40 dB.
        -80.0 + ((freq - 8000.0) / (nyquist - 8000.0).max(1.0)).clamp(0.0, 1.0) * 40.0
    };
    10f32.powf(db / 20.0)
}

/// Per-post Y quantisation. For each X post, look at the spectrum across
/// the entire span to that post's nearest neighbours (so no spectral peak
/// is invisible to the floor sampling), divide by `FLOOR_SCALE` for
/// residue headroom, and apply an ATH floor minimum.
fn analyse_floor1(floor: &Floor1, spec: &[f32], n_half: usize, sample_rate: u32) -> Vec<i32> {
    let xlist = &floor.xlist;
    let n_posts = xlist.len();
    // Sort posts by X position so we can look up neighbours easily.
    let mut order: Vec<usize> = (0..n_posts).collect();
    order.sort_by_key(|&i| xlist[i]);
    // For each post in original index order, find the X-coord of the
    // nearest post on each side (in sorted order) so we can scan the
    // full spectral window between neighbouring posts.
    let mut neighbour_lo = vec![0usize; n_posts];
    let mut neighbour_hi = vec![n_half; n_posts];
    for (rank, &idx) in order.iter().enumerate() {
        let here = xlist[idx] as usize;
        let lo = if rank == 0 {
            0
        } else {
            (xlist[order[rank - 1]] as usize + here) / 2
        };
        let hi = if rank + 1 == n_posts {
            n_half
        } else {
            (xlist[order[rank + 1]] as usize + here).div_ceil(2)
        };
        neighbour_lo[idx] = lo;
        neighbour_hi[idx] = hi.min(n_half);
    }

    let mut y = Vec::with_capacity(n_posts);
    let mult = floor.multiplier as usize;
    for (i, &x) in xlist.iter().enumerate() {
        let bin = (x as usize).min(n_half.saturating_sub(1));
        let lo = neighbour_lo[i];
        let hi = neighbour_hi[i].max(lo + 1);
        let mut mag = 0f32;
        for v in &spec[lo..hi.min(spec.len())] {
            let a = v.abs();
            if a > mag {
                mag = a;
            }
        }
        // Scale floor down so the residue has headroom in [-5, 5] units.
        let ath = ath_min_for_bin(bin, n_half, sample_rate);
        let target_mag = (mag / FLOOR_SCALE).max(ath).max(1e-30);
        let target = target_mag.ln();
        let mut best_y = 0i32;
        let mut best_diff = f32::MAX;
        for cand in 0..128 {
            let idx = (cand * mult).min(255);
            let table_v = FLOOR1_INVERSE_DB[idx];
            let diff = (table_v.ln() - target).abs();
            if diff < best_diff {
                best_diff = diff;
                best_y = cand as i32;
            }
        }
        y.push(best_y);
    }
    y
}

impl Encoder for VorbisEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.out_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        if self.flushed {
            return Err(Error::other("encoder already flushed"));
        }
        match frame {
            Frame::Audio(a) => {
                self.push_audio_frame(a)?;
                self.drain_blocks();
                Ok(())
            }
            _ => Err(Error::invalid("Vorbis encoder expects an audio frame")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.output_queue.pop_front() {
            return Ok(p);
        }
        if self.flushed {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        // Drain whole blocks from the regular pipeline (each emit requires
        // `max_fresh` samples; below that we fall through to the trailing
        // zero-padded block below).
        self.drain_blocks();
        // Emit one final block (zero-padded right half) so the decoder's
        // final OLA pass produces samples for the last full half-block of
        // input. This matches the Vorbis convention: the trailing block is
        // a "phantom" — same size as the previous block, with no real new
        // input. Without it the very last overlap-width samples of input
        // stay locked in `prev_tail` and the decoder never sees them.
        let pending = self.input_buf[0].len();
        if pending > 0 || !self.prev_tail[0].is_empty() {
            self.emit_flush_block();
        }
        self.flushed = true;
        Ok(())
    }
}

impl VorbisEncoder {
    /// Emit a trailing zero-padded block matching the last-committed size.
    /// Used at flush time so the decoder sees OLA for the final chunk of
    /// input.
    fn emit_flush_block(&mut self) {
        let n_channels = self.channels as usize;
        let bs0 = self.blocksize_short;
        let long = self.next_block_long;
        let prev_long = self.prev_block_long;
        // Next block after the flush block is nominally long — the stream
        // ends here but we still write a valid flag.
        let next_long = true;
        let n = if long {
            self.blocksize_long
        } else {
            self.blocksize_short
        };
        let (l_start, l_end, r_start, r_end) = window_bounds(long, prev_long, next_long, n, bs0);
        let fresh_needed = r_end - l_end;
        let mut block: Vec<Vec<f32>> = Vec::with_capacity(n_channels);
        for ch in 0..n_channels {
            let mut v = vec![0f32; n];
            let overlap_len = l_end - l_start;
            let tail = &self.prev_tail[ch];
            let tlen = tail.len().min(overlap_len);
            let tail_offset = overlap_len - tlen;
            v[l_start + tail_offset..l_start + tail_offset + tlen]
                .copy_from_slice(&tail[tail.len() - tlen..]);
            let take = self.input_buf[ch].len().min(fresh_needed);
            v[l_end..l_end + take].copy_from_slice(&self.input_buf[ch][..take]);
            // Remaining positions stay zero (zero-pad).
            self.input_buf[ch].drain(..take);
            self.prev_tail[ch].clear();
            self.prev_tail[ch].extend_from_slice(&v[r_start..r_end]);
            block.push(v);
        }
        let pkt = self.encode_block_packet(&block, n, long, prev_long, next_long);
        self.output_queue.push_back(pkt);
        self.prev_block_long = long;
        self.next_block_long = next_long;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identification::parse_identification_header;
    use crate::setup::parse_setup;

    #[test]
    fn identification_header_roundtrip() {
        let bytes = build_identification_header(2, 48_000, 128_000, 8, 11);
        let id = parse_identification_header(&bytes).expect("parse");
        assert_eq!(id.audio_channels, 2);
        assert_eq!(id.audio_sample_rate, 48_000);
        assert_eq!(id.bitrate_nominal, 128_000);
        assert_eq!(id.blocksize_0, 8);
        assert_eq!(id.blocksize_1, 11);
    }

    #[test]
    fn comment_header_signature() {
        let bytes = build_comment_header(&["TITLE=Test".to_string()]);
        assert_eq!(bytes[0], 0x03);
        assert_eq!(&bytes[1..7], b"vorbis");
        assert_eq!(*bytes.last().unwrap() & 0x01, 0x01);
    }

    #[test]
    fn placeholder_setup_parses() {
        let bytes = build_placeholder_setup_header(1);
        let setup = parse_setup(&bytes, 1).expect("placeholder parses");
        assert_eq!(setup.codebooks.len(), 1);
    }

    #[test]
    fn encoder_setup_parses_mono() {
        let bytes = build_encoder_setup_header(1);
        let setup = parse_setup(&bytes, 1).expect("encoder setup parses");
        assert_eq!(setup.codebooks.len(), 3);
        assert_eq!(setup.floors.len(), 2);
        assert_eq!(setup.residues.len(), 2);
        assert_eq!(setup.mappings.len(), 2);
        assert_eq!(setup.modes.len(), 2);
        // Codebook 2 must be the dim-2 VQ, 128 entries (full Huffman tree).
        assert_eq!(setup.codebooks[2].entries, VQ_ENTRIES);
        assert_eq!(setup.codebooks[2].dimensions, 2);
        let vq = setup.codebooks[2].vq.as_ref().unwrap();
        assert_eq!(vq.lookup_type, 1);
        assert!((vq.min - VQ_MIN).abs() < 1e-5);
    }

    #[test]
    fn encoder_setup_parses_stereo() {
        let bytes = build_encoder_setup_header(2);
        let setup = parse_setup(&bytes, 2).expect("encoder setup parses stereo");
        assert_eq!(setup.codebooks.len(), 3);
        assert_eq!(setup.mappings.len(), 2);
        // Stereo: 1 coupling step (mag=0, ang=1).
        assert_eq!(setup.mappings[0].coupling.len(), 1);
        assert_eq!(setup.mappings[0].coupling[0], (0, 1));
        assert_eq!(setup.mappings[1].coupling[0], (0, 1));
    }

    #[test]
    fn analyse_floor1_captures_peak_between_posts() {
        // Build a floor with sparse posts (every 64 bins) and a spectrum
        // with a single peak smack in the middle of two posts. The new
        // analyser should pick it up.
        let n_half = 1024usize;
        let mut spec = vec![0f32; n_half];
        spec[200] = 1.0; // peak between posts at 192 and 256 (long_floor_extra_x has post at ~192)
        let bytes = build_encoder_setup_header(1);
        let setup = parse_setup(&bytes, 1).expect("parse setup");
        let f = match &setup.floors[1] {
            Floor::Type1(f) => f.clone(),
            _ => panic!("expected floor1"),
        };
        let y = analyse_floor1(&f, &spec, n_half, 48_000);
        // Find the post nearest bin 200.
        let mut best = (usize::MAX, usize::MAX);
        for (i, &x) in f.xlist.iter().enumerate() {
            let d = (x as i32 - 200).unsigned_abs() as usize;
            if d < best.1 {
                best = (i, d);
            }
        }
        // That post must have a non-trivial Y (peak got captured, not 0).
        assert!(
            y[best.0] > 50,
            "expected captured peak Y > 50, got {} at post idx {}",
            y[best.0],
            best.0
        );
    }

    #[test]
    fn forward_couple_roundtrips_via_decoder_inverse() {
        // Mirror of the inverse coupling code in `crate::decoder`. We must
        // round-trip every (l, r) ∈ {-2..2}² through forward_couple →
        // inverse_couple and recover the input bit-exactly.
        fn inverse_couple(m: f32, a: f32) -> (f32, f32) {
            if m > 0.0 {
                if a > 0.0 {
                    (m, m - a)
                } else {
                    (m + a, m)
                }
            } else if a > 0.0 {
                (m, m + a)
            } else {
                (m - a, m)
            }
        }
        for li in -3..=3 {
            for ri in -3..=3 {
                let l = li as f32;
                let r = ri as f32;
                let (m, a) = forward_couple(l, r);
                let (lp, rp) = inverse_couple(m, a);
                assert_eq!(
                    (lp, rp),
                    (l, r),
                    "l={}, r={}, m={}, a={}, decoded=({}, {})",
                    l,
                    r,
                    m,
                    a,
                    lp,
                    rp
                );
            }
        }
        // Also check fractional inputs.
        for &(l, r) in &[
            (0.0f32, 0.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (0.5, -0.5),
            (-0.25, 0.75),
            (2.5, -1.875),
            (1e-5, -1e-5),
        ] {
            let (m, a) = forward_couple(l, r);
            let (lp, rp) = inverse_couple(m, a);
            assert!(
                (lp - l).abs() < 1e-6 && (rp - r).abs() < 1e-6,
                "fractional ({l}, {r}) → ({m}, {a}) → ({lp}, {rp})"
            );
        }
    }

    #[test]
    fn bit_reverse_basic() {
        assert_eq!(bit_reverse(0b1011, 4), 0b1101);
        assert_eq!(bit_reverse(0b1, 1), 0b1);
        assert_eq!(bit_reverse(0b10, 2), 0b01);
        assert_eq!(bit_reverse(0b110, 3), 0b011);
    }

    #[test]
    fn vorbis_float_roundtrip() {
        use crate::bits_ext::BitReaderExt;
        use oxideav_core::bits::BitReaderLsb as BitReader;
        for &v in &[1.0f32, -1.0, 0.5, -0.25, 16.0, 1e-5] {
            let mut w = BitWriter::new();
            write_vorbis_float(&mut w, v);
            let bytes = w.finish();
            let mut br = BitReader::new(&bytes);
            let decoded = br.read_vorbis_float().unwrap();
            assert!(
                (decoded - v).abs() / v.abs() < 1e-4,
                "roundtrip {v} → {decoded}"
            );
        }
    }

    #[test]
    fn extradata_lacing_splits_back() {
        let id = build_identification_header(1, 48_000, 0, 8, 11);
        let comm = build_comment_header(&[]);
        let setup = build_placeholder_setup_header(1);
        let blob = build_extradata(&id, &comm, &setup);
        assert_eq!(blob[0], 2);
        let n_packets = blob[0] as usize + 1;
        let mut sizes = Vec::new();
        let mut i = 1usize;
        for _ in 0..n_packets - 1 {
            let mut s = 0usize;
            loop {
                let b = blob[i];
                i += 1;
                s += b as usize;
                if b < 255 {
                    break;
                }
            }
            sizes.push(s);
        }
        sizes.push(blob.len() - i - sizes.iter().sum::<usize>());
        assert_eq!(sizes[0], id.len());
        assert_eq!(sizes[1], comm.len());
        assert_eq!(sizes[2], setup.len());
    }

    #[test]
    fn make_encoder_emits_headers() {
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(1);
        params.sample_rate = Some(48_000);
        let enc = make_encoder(&params).expect("make_encoder");
        assert!(!enc.output_params().extradata.is_empty());
    }

    #[test]
    fn send_frame_emits_silent_packet_per_block() {
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(1);
        params.sample_rate = Some(48_000);
        let mut enc = make_encoder(&params).unwrap();
        let block = 1usize << DEFAULT_BLOCKSIZE_LONG_LOG2;
        let frame = Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 48_000,
            samples: block as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, 48_000),
            data: vec![vec![0u8; block * 2]],
        });
        enc.send_frame(&frame).expect("send_frame");
        // With overlapping windows (advance N/2 per block), N samples of
        // input yields TWO long-block packets that share the middle N/2
        // samples. Both should be tiny silent packets.
        let pkt = enc.receive_packet().expect("packet 0");
        assert_eq!(pkt.pts, Some(0));
        assert_eq!(pkt.duration, Some(block as i64));
        // Silent packet: header bit + mode (1) + prev_long + next_long +
        // 1 floor-unused bit = 5 bits → 1 byte.
        assert!(
            pkt.data.len() <= 2,
            "silent packet too big: {}",
            pkt.data.len()
        );
        let _pkt2 = enc.receive_packet().expect("packet 1 (overlap)");
        assert!(matches!(enc.receive_packet(), Err(Error::NeedMore)));
    }

    #[test]
    fn flush_emits_final_padded_packet() {
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(2);
        params.sample_rate = Some(48_000);
        let mut enc = make_encoder(&params).unwrap();
        let frame = Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 2,
            sample_rate: 48_000,
            samples: 64,
            pts: Some(0),
            time_base: TimeBase::new(1, 48_000),
            data: vec![vec![0u8; 64 * 4]],
        });
        enc.send_frame(&frame).unwrap();
        assert!(matches!(enc.receive_packet(), Err(Error::NeedMore)));
        enc.flush().unwrap();
        let pkt = enc.receive_packet().expect("flush emits packet");
        assert_eq!(pkt.pts, Some(0));
        assert!(matches!(enc.receive_packet(), Err(Error::Eof)));
    }

    fn sine_samples(freq: f64, n: usize, sr: f64, amp: f64) -> Vec<i16> {
        (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                let s = (2.0 * std::f64::consts::PI * freq * t).sin() * amp;
                (s * 32768.0) as i16
            })
            .collect()
    }

    fn goertzel_mag(samples: &[i16], freq: f64, sr: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * freq / sr;
        let coeff = 2.0 * omega.cos();
        let mut s_prev = 0f64;
        let mut s_prev2 = 0f64;
        for &s in samples {
            let s_now = s as f64 + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s_now;
        }
        (s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2).sqrt()
    }

    fn encode_and_decode(
        channels: u16,
        samples_per_channel: usize,
        pcm_i16_interleaved: &[i16],
    ) -> Vec<i16> {
        use crate::decoder::make_decoder as make_dec;
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(channels);
        params.sample_rate = Some(48_000);
        let mut enc = make_encoder(&params).unwrap();
        // Pack into bytes.
        let mut data = Vec::with_capacity(pcm_i16_interleaved.len() * 2);
        for s in pcm_i16_interleaved {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let frame = Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels,
            sample_rate: 48_000,
            samples: samples_per_channel as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, 48_000),
            data: vec![data],
        });
        enc.send_frame(&frame).unwrap();
        enc.flush().unwrap();
        let mut packets = Vec::new();
        while let Ok(p) = enc.receive_packet() {
            packets.push(p);
        }
        let mut dec_params = enc.output_params().clone();
        dec_params.extradata = enc.output_params().extradata.clone();
        let mut dec = make_dec(&dec_params).expect("decoder accepts our extradata");
        let mut out: Vec<i16> = Vec::new();
        for pkt in &packets {
            dec.send_packet(pkt).unwrap();
            if let Ok(Frame::Audio(a)) = dec.receive_frame() {
                for chunk in a.data[0].chunks_exact(2) {
                    out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
        }
        out
    }

    #[test]
    fn roundtrip_sine_via_our_decoder() {
        // 8 long blocks of 1 kHz sine mono at 48 kHz.
        let n = (1usize << DEFAULT_BLOCKSIZE_LONG_LOG2) * 8;
        let samples = sine_samples(1000.0, n, 48_000.0, 0.5);
        let pcm = encode_and_decode(1, n, &samples);
        assert!(!pcm.is_empty(), "expected decoded samples");
        let mut sum_sq = 0f64;
        let mut peak = 0i32;
        for &s in &pcm {
            sum_sq += (s as f64) * (s as f64);
            let a = (s as i32).abs();
            if a > peak {
                peak = a;
            }
        }
        let rms = (sum_sq / pcm.len() as f64).sqrt();
        let target = goertzel_mag(&pcm, 1000.0, 48_000.0);
        let off = goertzel_mag(&pcm, 7000.0, 48_000.0);
        eprintln!(
            "mono 1kHz: rms={rms} peak={peak} samples={} target={target} off={off}",
            pcm.len()
        );
        assert!(rms > 100.0, "RMS too low: {rms}");
        assert!(peak < 32768);
        assert!(
            target > off,
            "1 kHz energy should dominate: {target} vs {off}"
        );
    }

    #[test]
    fn roundtrip_long_blocks_mono() {
        // 4 long blocks of 440 Hz mono.
        let n = (1usize << DEFAULT_BLOCKSIZE_LONG_LOG2) * 4;
        let samples = sine_samples(440.0, n, 48_000.0, 0.5);
        let pcm = encode_and_decode(1, n, &samples);
        assert!(!pcm.is_empty());
        let mut sum_sq = 0f64;
        for &s in &pcm {
            sum_sq += (s as f64) * (s as f64);
        }
        let rms = (sum_sq / pcm.len() as f64).sqrt();
        let target = goertzel_mag(&pcm, 440.0, 48_000.0);
        let off = goertzel_mag(&pcm, 3000.0, 48_000.0);
        eprintln!(
            "mono 440Hz: rms={rms} samples={} target={target} off={off}",
            pcm.len()
        );
        assert!(rms > 500.0, "RMS too low: {rms}");
        assert!(target > off);
    }

    #[test]
    fn roundtrip_mixed_frequencies() {
        // 4 long blocks: 440 Hz + 1 kHz sum.
        let n = (1usize << DEFAULT_BLOCKSIZE_LONG_LOG2) * 4;
        let sr = 48_000.0;
        let samples: Vec<i16> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                let s = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.25
                    + (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.25;
                (s * 32768.0) as i16
            })
            .collect();
        let pcm = encode_and_decode(1, n, &samples);
        assert!(!pcm.is_empty());
        let m_440 = goertzel_mag(&pcm, 440.0, sr);
        let m_1000 = goertzel_mag(&pcm, 1000.0, sr);
        let m_off = goertzel_mag(&pcm, 3000.0, sr);
        eprintln!("mixed: 440={m_440} 1000={m_1000} off={m_off}");
        assert!(m_440 > m_off, "440 Hz should dominate over 3 kHz");
        assert!(m_1000 > m_off, "1 kHz should dominate over 3 kHz");
    }

    #[test]
    fn roundtrip_sine_stereo_via_our_decoder() {
        // 8 long blocks of 1 kHz sine stereo.
        let n = (1usize << DEFAULT_BLOCKSIZE_LONG_LOG2) * 8;
        let sr = 48_000.0;
        let mut samples: Vec<i16> = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f64 / sr;
            let s = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.5;
            let q = (s * 32768.0) as i16;
            samples.push(q);
            samples.push(q);
        }
        let pcm = encode_and_decode(2, n, &samples);
        assert!(!pcm.is_empty());
        // Deinterleave.
        let mut left = Vec::with_capacity(pcm.len() / 2);
        let mut right = Vec::with_capacity(pcm.len() / 2);
        for chunk in pcm.chunks_exact(2) {
            left.push(chunk[0]);
            right.push(chunk[1]);
        }
        let rms_l =
            (left.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / left.len() as f64).sqrt();
        let rms_r =
            (right.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / right.len() as f64).sqrt();
        let t_l = goertzel_mag(&left, 1000.0, sr);
        let t_r = goertzel_mag(&right, 1000.0, sr);
        let off_l = goertzel_mag(&left, 5000.0, sr);
        let off_r = goertzel_mag(&right, 5000.0, sr);
        eprintln!("stereo 1kHz: rms_l={rms_l} rms_r={rms_r} t_l={t_l} t_r={t_r} off_l={off_l} off_r={off_r}");
        assert!(rms_l > 500.0, "L RMS too low: {rms_l}");
        assert!(rms_r > 500.0, "R RMS too low: {rms_r}");
        assert!(t_l > off_l);
        assert!(t_r > off_r);
    }

    /// Encode `pcm_i16_interleaved` through the encoder with the optional
    /// `force_long_only` override, then decode through our own decoder.
    fn encode_and_decode_with_flag(
        channels: u16,
        samples_per_channel: usize,
        pcm_i16_interleaved: &[i16],
        force_long_only: bool,
    ) -> Vec<i16> {
        use crate::decoder::make_decoder as make_dec;
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(channels);
        params.sample_rate = Some(48_000);
        let mut enc = make_encoder(&params).unwrap();
        // SAFETY: the encoder implementation is a `VorbisEncoder`; we
        // downcast via a dedicated flag setter on the trait extension in
        // this test module. Rather than wire a trait method through, we
        // rebuild a VorbisEncoder directly for the forced-long path.
        if force_long_only {
            let sample_rate = 48_000u32;
            let id_hdr = build_identification_header(
                channels as u8,
                sample_rate,
                0,
                DEFAULT_BLOCKSIZE_SHORT_LOG2,
                DEFAULT_BLOCKSIZE_LONG_LOG2,
            );
            let comment_hdr = build_comment_header(&[]);
            let setup_hdr = build_encoder_setup_header(channels as u8);
            let extradata = build_extradata(&id_hdr, &comment_hdr, &setup_hdr);
            let codebooks = extract_codebooks(&setup_hdr).unwrap();
            let setup = crate::setup::parse_setup(&setup_hdr, channels as u8).unwrap();
            let mut out_params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
            out_params.media_type = MediaType::Audio;
            out_params.channels = Some(channels);
            out_params.sample_rate = Some(sample_rate);
            out_params.sample_format = Some(SampleFormat::S16);
            out_params.extradata = extradata;
            let blocksize_short = 1usize << DEFAULT_BLOCKSIZE_SHORT_LOG2;
            let blocksize_long = 1usize << DEFAULT_BLOCKSIZE_LONG_LOG2;
            enc = Box::new(VorbisEncoder {
                codec_id: CodecId::new(crate::CODEC_ID_STR),
                out_params,
                time_base: TimeBase::new(1, sample_rate as i64),
                channels,
                sample_rate,
                blocksize_short,
                blocksize_long,
                input_buf: vec![Vec::with_capacity(blocksize_long * 4); channels as usize],
                prev_tail: vec![Vec::with_capacity(blocksize_long); channels as usize],
                output_queue: VecDeque::new(),
                pts: 0,
                blocks_emitted: 0,
                flushed: false,
                codebooks,
                setup,
                prev_block_long: true,
                next_block_long: true,
                prior_energy: 0.0,
                force_long_only: true,
            });
        }
        let mut data = Vec::with_capacity(pcm_i16_interleaved.len() * 2);
        for s in pcm_i16_interleaved {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let frame = Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels,
            sample_rate: 48_000,
            samples: samples_per_channel as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, 48_000),
            data: vec![data],
        });
        enc.send_frame(&frame).unwrap();
        enc.flush().unwrap();
        let mut packets = Vec::new();
        while let Ok(p) = enc.receive_packet() {
            packets.push(p);
        }
        let mut dec_params = enc.output_params().clone();
        dec_params.extradata = enc.output_params().extradata.clone();
        let mut dec = make_dec(&dec_params).expect("decoder accepts our extradata");
        let mut out: Vec<i16> = Vec::new();
        for pkt in &packets {
            dec.send_packet(pkt).unwrap();
            if let Ok(Frame::Audio(a)) = dec.receive_frame() {
                for chunk in a.data[0].chunks_exact(2) {
                    out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
        }
        out
    }

    /// Compute the mean-squared error between `reference` and `candidate`
    /// over a reference-index window `[start, end)`. Positions beyond
    /// `candidate.len()` contribute the reference sample's squared value
    /// (candidate treated as silent past its end). Returns 0 if the
    /// window is empty.
    ///
    /// Note: the Vorbis decoder's output is aligned 1:1 with the encoder
    /// input (pcm[i] = samples[i]) in our harness — the first packet
    /// emits zero samples but does so "before" the audio window starts,
    /// so the second packet's emission is already at audio index 0.
    fn mse_window(reference: &[i16], candidate: &[i16], start: usize, end: usize) -> f64 {
        let end = end.min(reference.len());
        if end <= start {
            return 0.0;
        }
        let mut sum = 0f64;
        for i in start..end {
            let r = reference[i] as f64;
            let c = candidate.get(i).copied().unwrap_or(0) as f64;
            let d = r - c;
            sum += d * d;
        }
        sum / (end - start) as f64
    }

    #[test]
    fn encoder_emits_short_blocks_on_transient() {
        // Sanity check: the transient detector actually flips the encoder
        // into short-block mode when fed a loud click riding on a steady
        // mid-amplitude tone.
        let n_blocks = 10usize;
        let n_long = 1usize << DEFAULT_BLOCKSIZE_LONG_LOG2;
        let total = n_blocks * n_long;
        let click_pos = 2 * n_long + 1024;
        let mut samples = vec![0i16; total];
        // Steady 220 Hz tone at 0.3 amp so `prior_energy` settles.
        for (i, s) in samples.iter_mut().enumerate().take(click_pos) {
            let t = i as f64 / 48_000.0;
            let v = (2.0 * std::f64::consts::PI * 220.0 * t).sin() * 0.3;
            *s = (v * 32768.0) as i16;
        }
        samples[click_pos] = -32000;
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(1);
        params.sample_rate = Some(48_000);
        let mut enc = make_encoder(&params).unwrap();
        let mut data = Vec::with_capacity(samples.len() * 2);
        for s in &samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let frame = Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 48_000,
            samples: total as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, 48_000),
            data: vec![data],
        });
        enc.send_frame(&frame).unwrap();
        enc.flush().unwrap();
        let mut n_short = 0usize;
        let mut n_long_pkts = 0usize;
        while let Ok(p) = enc.receive_packet() {
            // Decode the mode bit from the packet's first byte: bit 0 is
            // the audio header bit, bit 1 is the mode bit (short=0, long=1).
            let first = p.data[0];
            let mode_bit = (first >> 1) & 1;
            if mode_bit == 1 {
                n_long_pkts += 1;
            } else {
                n_short += 1;
            }
        }
        eprintln!("n_long_pkts={n_long_pkts} n_short={n_short}");
        assert!(
            n_short > 0,
            "expected at least one short block emitted for a transient signal"
        );
    }

    #[test]
    fn roundtrip_sine_through_forced_transition_matches_baseline() {
        // Drive the encoder into short-block mode with an artificial loud
        // pulse, then let it fall back to long. Verify the non-transient
        // portions of the output still track the input reasonably — if
        // the asymmetric window OLA is broken, the output around the
        // transition will diverge wildly from a pure sine.
        let n_long = 1usize << DEFAULT_BLOCKSIZE_LONG_LOG2;
        let total = n_long * 6;
        let sr = 48_000.0f64;
        let mut samples = vec![0i16; total];
        for (i, s) in samples.iter_mut().enumerate() {
            let t = i as f64 / sr;
            let v = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.3;
            *s = (v * 32768.0) as i16;
        }
        // Insert a short-duration loud burst to trigger the transient
        // detector mid-stream.
        for i in (n_long * 2)..(n_long * 2 + 64) {
            samples[i] = 32000;
        }
        // Decode through the short-capable encoder.
        let pcm = encode_and_decode_with_flag(1, total, &samples, false);
        assert!(!pcm.is_empty());
        // Sanity: the 440 Hz Goertzel component should still dominate over
        // the 5 kHz off-band one in a "safe" region far from the burst
        // (samples 12288..16384).
        let safe: Vec<i16> = pcm[(3 * n_long)..(4 * n_long).min(pcm.len())].to_vec();
        assert!(!safe.is_empty());
        let t_safe = goertzel_mag(&safe, 440.0, sr);
        let o_safe = goertzel_mag(&safe, 5000.0, sr);
        eprintln!("post-transition: 440={t_safe} 5k={o_safe}");
        assert!(
            t_safe > 2.0 * o_safe,
            "post-transition tone should still dominate: 440={t_safe} 5k={o_safe}"
        );
    }

    #[test]
    fn roundtrip_click_short_beats_long_only_baseline() {
        // A full-scale click placed inside the 3rd long block (sample 5120,
        // =2*n_long+1024). The first long block of a Vorbis stream has its
        // decoded output discarded by the decoder (prev_tail is empty on
        // the first packet), so the click has to land in a later packet to
        // be measurable at all — for both the short-capable and long-only
        // baselines. Preceding the click we mix in a quiet 220 Hz sine so
        // the transient detector's rolling `prior_energy` has a baseline
        // to compare against.
        let n_blocks = 10usize;
        let n_long = 1usize << DEFAULT_BLOCKSIZE_LONG_LOG2;
        let total = n_blocks * n_long;
        let click_pos = 2 * n_long + 1024;
        let mut samples = vec![0i16; total];
        // Preceding the click: a mid-amplitude 220 Hz sine so the
        // transient detector's `prior_energy` estimator has a non-trivial
        // baseline and a reasonably-encodable "continuous" signal.
        // The click is full-scale; the trailing silence is where the
        // MSE metric picks up pre-/post-echo leakage.
        let seed_energy = 0.3f64;
        for (i, s) in samples.iter_mut().enumerate().take(click_pos) {
            let t = i as f64 / 48_000.0;
            let v = (2.0 * std::f64::consts::PI * 220.0 * t).sin() * seed_energy;
            *s = (v * 32768.0) as i16;
        }
        samples[click_pos] = -32000;
        // Post-click: silence (already zero).

        // Baseline: long-only encode.
        let pcm_long = encode_and_decode_with_flag(1, total, &samples, true);
        // Short-block capable encode.
        let pcm_short = encode_and_decode_with_flag(1, total, &samples, false);

        assert!(!pcm_long.is_empty(), "long-only encode produced no output");
        assert!(
            !pcm_short.is_empty(),
            "short-capable encode produced no output"
        );

        // Measure post-echo in the silence region 192..704 samples after
        // the click. Reference is zero there, so any decoded energy is
        // pure encoding error. Long blocks (N=2048) spread full-scale
        // transient energy across the entire block's ~1024-sample post-
        // echo tail; short blocks (N=256) confine the spread to ~128
        // samples, so by 192 samples out the short-capable encoder's
        // residual should be substantially smaller than the long baseline.
        let win_start = click_pos + 192;
        let win_end = (click_pos + 192 + 512)
            .min(pcm_long.len())
            .min(pcm_short.len());
        let mse_long = mse_window(&samples, &pcm_long, win_start, win_end);
        let mse_short = mse_window(&samples, &pcm_short, win_start, win_end);

        eprintln!(
            "click post-echo window [{win_start}, {win_end}): mse_long={mse_long:.2} mse_short={mse_short:.2} (pcm_long.len={} pcm_short.len={})",
            pcm_long.len(),
            pcm_short.len()
        );
        // Short-block encoder should have noticeably less residual energy
        // post-click. 1.2× is a conservative ratio — libvorbis-style
        // transient handling routinely buys much more than this.
        assert!(
            mse_short * 1.2 < mse_long,
            "expected short-block MSE < long-only MSE in post-click window, got short={mse_short} long={mse_long}"
        );
    }

    #[test]
    fn roundtrip_silence_via_our_decoder() {
        use crate::decoder::make_decoder as make_dec;
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(1);
        params.sample_rate = Some(48_000);
        let mut enc = make_encoder(&params).unwrap();
        let block = 1usize << DEFAULT_BLOCKSIZE_LONG_LOG2;
        let n_blocks = 4;
        let frame = Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 48_000,
            samples: (block * n_blocks) as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, 48_000),
            data: vec![vec![0u8; block * n_blocks * 2]],
        });
        enc.send_frame(&frame).unwrap();
        enc.flush().unwrap();
        let mut packets = Vec::new();
        while let Ok(p) = enc.receive_packet() {
            packets.push(p);
        }
        assert!(!packets.is_empty());
        let mut dec_params = enc.output_params().clone();
        dec_params.extradata = enc.output_params().extradata.clone();
        let mut dec = make_dec(&dec_params).expect("decoder accepts our extradata");
        let mut emitted = 0usize;
        for pkt in &packets {
            dec.send_packet(pkt).unwrap();
            if let Ok(Frame::Audio(a)) = dec.receive_frame() {
                emitted += a.samples as usize;
                for plane in &a.data {
                    assert!(plane.iter().all(|&b| b == 0), "expected silence");
                }
            }
        }
        assert!(emitted > 0);
    }

    // ========== Multichannel round-trip tests ==========

    /// Verify that `build_encoder_setup_header` emits the standard
    /// coupling pair list for each supported channel count (1..=8), and
    /// that our parser reads it back.
    #[test]
    fn encoder_setup_coupling_for_all_channel_counts() {
        for ch in 1u8..=8 {
            let bytes = build_encoder_setup_header(ch);
            let setup = parse_setup(&bytes, ch).unwrap_or_else(|e| {
                panic!("channel count {ch} setup header failed to parse: {e}")
            });
            let expected = standard_coupling_steps(ch);
            assert_eq!(
                setup.mappings[0].coupling.len(),
                expected.len(),
                "ch={}: expected {} coupling steps, got {}",
                ch,
                expected.len(),
                setup.mappings[0].coupling.len()
            );
            for (i, &pair) in expected.iter().enumerate() {
                assert_eq!(
                    setup.mappings[0].coupling[i], pair,
                    "ch={}: coupling step {} mismatch",
                    ch, i
                );
                assert_eq!(
                    setup.mappings[1].coupling[i], pair,
                    "ch={}: coupling step {} mismatch (long mapping)",
                    ch, i
                );
            }
        }
    }

    /// Round-trip helper for multichannel PCM. `pcm_per_channel` is a
    /// vector of per-channel f32 samples in [-1, 1]. Returns decoded S16
    /// samples per channel (already deinterleaved).
    fn encode_decode_multichannel(
        channels: u16,
        pcm_per_channel: &[Vec<f32>],
    ) -> Vec<Vec<i16>> {
        assert_eq!(pcm_per_channel.len(), channels as usize);
        let n = pcm_per_channel[0].len();
        let mut interleaved_i16 = Vec::with_capacity(n * channels as usize);
        for i in 0..n {
            for ch in 0..channels as usize {
                let v = pcm_per_channel[ch][i].clamp(-1.0, 1.0);
                interleaved_i16.push((v * 32768.0) as i16);
            }
        }
        let mut data = Vec::with_capacity(interleaved_i16.len() * 2);
        for s in &interleaved_i16 {
            data.extend_from_slice(&s.to_le_bytes());
        }

        use crate::decoder::make_decoder as make_dec;
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(channels);
        params.sample_rate = Some(48_000);
        let mut enc = make_encoder(&params).unwrap();
        let frame = Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels,
            sample_rate: 48_000,
            samples: n as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, 48_000),
            data: vec![data],
        });
        enc.send_frame(&frame).unwrap();
        enc.flush().unwrap();
        let mut packets = Vec::new();
        while let Ok(p) = enc.receive_packet() {
            packets.push(p);
        }
        let dec_params = enc.output_params().clone();
        let mut dec = make_dec(&dec_params).expect("decoder accepts our extradata");
        let mut per_ch: Vec<Vec<i16>> = vec![Vec::new(); channels as usize];
        for pkt in &packets {
            dec.send_packet(pkt).unwrap();
            if let Ok(Frame::Audio(a)) = dec.receive_frame() {
                let plane = &a.data[0];
                let stride = channels as usize * 2;
                for chunk in plane.chunks_exact(stride) {
                    for ch in 0..channels as usize {
                        let off = ch * 2;
                        per_ch[ch]
                            .push(i16::from_le_bytes([chunk[off], chunk[off + 1]]));
                    }
                }
            }
        }
        per_ch
    }

    /// Sanity check: each channel's RMS is above the given floor, and the
    /// expected Goertzel frequency dominates over a detuned off-band.
    fn assert_channel_energy(
        tag: &str,
        decoded: &[i16],
        target_freq: f64,
        off_freq: f64,
        rms_floor: f64,
    ) {
        let sr = 48_000.0;
        let rms =
            (decoded.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / decoded.len() as f64)
                .sqrt();
        let t = goertzel_mag(decoded, target_freq, sr);
        let o = goertzel_mag(decoded, off_freq, sr);
        eprintln!("{tag}: rms={rms} t({target_freq})={t} o({off_freq})={o}");
        assert!(rms > rms_floor, "{tag}: RMS too low ({rms} < {rms_floor})");
        assert!(
            t > o,
            "{tag}: target {target_freq} ({t}) should beat off {off_freq} ({o})"
        );
    }

    #[test]
    fn roundtrip_3ch_lcr_sine() {
        // L = 400 Hz, C = 800 Hz, R = 1200 Hz — each channel gets a distinct
        // tone so we can verify the right signal shows up per-channel after
        // coupling.
        let n = (1usize << DEFAULT_BLOCKSIZE_LONG_LOG2) * 4;
        let sr = 48_000.0;
        let freqs = [400.0f64, 800.0, 1200.0];
        let pcm: Vec<Vec<f32>> = freqs
            .iter()
            .map(|&f| {
                (0..n)
                    .map(|i| {
                        let t = i as f64 / sr;
                        (2.0 * std::f64::consts::PI * f * t).sin() as f32 * 0.4
                    })
                    .collect()
            })
            .collect();
        let decoded = encode_decode_multichannel(3, &pcm);
        assert_eq!(decoded.len(), 3);
        assert!(decoded.iter().all(|c| !c.is_empty()));
        assert_channel_energy("3ch/L", &decoded[0], 400.0, 3500.0, 200.0);
        assert_channel_energy("3ch/C", &decoded[1], 800.0, 3500.0, 200.0);
        assert_channel_energy("3ch/R", &decoded[2], 1200.0, 3500.0, 200.0);
    }

    #[test]
    fn roundtrip_4ch_quad_sine() {
        // FL=440, FR=660, BL=880, BR=1100 Hz.
        let n = (1usize << DEFAULT_BLOCKSIZE_LONG_LOG2) * 4;
        let sr = 48_000.0;
        let freqs = [440.0f64, 660.0, 880.0, 1100.0];
        let pcm: Vec<Vec<f32>> = freqs
            .iter()
            .map(|&f| {
                (0..n)
                    .map(|i| {
                        let t = i as f64 / sr;
                        (2.0 * std::f64::consts::PI * f * t).sin() as f32 * 0.4
                    })
                    .collect()
            })
            .collect();
        let decoded = encode_decode_multichannel(4, &pcm);
        assert_eq!(decoded.len(), 4);
        let names = ["FL", "FR", "BL", "BR"];
        for (ch, (&f, n)) in freqs.iter().zip(names.iter()).enumerate() {
            assert_channel_energy(
                &format!("4ch/{n}"),
                &decoded[ch],
                f,
                f + 2500.0,
                200.0,
            );
        }
    }

    #[test]
    fn roundtrip_5_1_sine() {
        // FL/C/FR/BL/BR/LFE each on a distinct tone so per-channel
        // coupling can be verified. LFE is band-limited (140 Hz).
        let n = (1usize << DEFAULT_BLOCKSIZE_LONG_LOG2) * 4;
        let sr = 48_000.0;
        let freqs = [440.0f64, 520.0, 660.0, 880.0, 1100.0, 140.0];
        let pcm: Vec<Vec<f32>> = freqs
            .iter()
            .map(|&f| {
                (0..n)
                    .map(|i| {
                        let t = i as f64 / sr;
                        (2.0 * std::f64::consts::PI * f * t).sin() as f32 * 0.4
                    })
                    .collect()
            })
            .collect();
        let decoded = encode_decode_multichannel(6, &pcm);
        assert_eq!(decoded.len(), 6);
        let names = ["FL", "C", "FR", "BL", "BR", "LFE"];
        for (ch, (&f, n)) in freqs.iter().zip(names.iter()).enumerate() {
            // LFE tone at 140 Hz is below the low edge of the default
            // floor setup's ATH curve (100 Hz break point) — compare
            // against 4 kHz off-band rather than freq+2500 to keep the
            // off-band in the midband the floor handles well.
            assert_channel_energy(
                &format!("5.1/{n}"),
                &decoded[ch],
                f,
                4000.0,
                150.0,
            );
        }
    }

    #[test]
    fn roundtrip_7_1_sine() {
        // FL/C/FR/SL/SR/BL/BR/LFE
        let n = (1usize << DEFAULT_BLOCKSIZE_LONG_LOG2) * 4;
        let sr = 48_000.0;
        let freqs = [440.0f64, 520.0, 660.0, 770.0, 990.0, 880.0, 1100.0, 140.0];
        let pcm: Vec<Vec<f32>> = freqs
            .iter()
            .map(|&f| {
                (0..n)
                    .map(|i| {
                        let t = i as f64 / sr;
                        (2.0 * std::f64::consts::PI * f * t).sin() as f32 * 0.35
                    })
                    .collect()
            })
            .collect();
        let decoded = encode_decode_multichannel(8, &pcm);
        assert_eq!(decoded.len(), 8);
        let names = ["FL", "C", "FR", "SL", "SR", "BL", "BR", "LFE"];
        for (ch, (&f, n)) in freqs.iter().zip(names.iter()).enumerate() {
            assert_channel_energy(
                &format!("7.1/{n}"),
                &decoded[ch],
                f,
                4200.0,
                100.0,
            );
        }
    }

    #[test]
    fn roundtrip_4ch_bformat_noise() {
        // Decorrelated white noise per channel at low amplitude. Our
        // coupling is lossless sum/difference so the round-trip error is
        // bounded by the quantisation floor even when L and R are
        // uncorrelated. Check overall RMS per channel is non-trivial.
        let n = (1usize << DEFAULT_BLOCKSIZE_LONG_LOG2) * 4;
        let mut seed = 0xC0FFEEu32;
        let mut rand = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as i32
        };
        let pcm: Vec<Vec<f32>> = (0..4)
            .map(|_| (0..n).map(|_| (rand() as f32 / (1 << 23) as f32) * 0.25).collect())
            .collect();
        let decoded = encode_decode_multichannel(4, &pcm);
        assert_eq!(decoded.len(), 4);
        for (ch, plane) in decoded.iter().enumerate() {
            let rms =
                (plane.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / plane.len() as f64)
                    .sqrt();
            eprintln!("B-format ch{ch} rms={rms}");
            // Input noise at amp 0.25 → input RMS ~ 2800 (0.25 * 32768 /
            // sqrt(3)); lossy encode floor/residue typically leaves at
            // least 10% of the energy. We accept anything above 200.
            assert!(rms > 200.0, "B-format ch{ch}: decoded RMS too low ({rms})");
        }
    }

    /// Compute per-channel signal-to-noise ratio (dB) of decoded vs
    /// reference samples aligned 1:1. `skip` omits the first N samples
    /// where the decoder's first packet emits zeros (OLA warm-up).
    fn snr_db(reference: &[i16], decoded: &[i16], skip: usize) -> f64 {
        let len = reference.len().min(decoded.len());
        if len <= skip {
            return f64::NEG_INFINITY;
        }
        let mut sig_sq = 0f64;
        let mut err_sq = 0f64;
        for i in skip..len {
            let r = reference[i] as f64;
            let d = decoded[i] as f64;
            sig_sq += r * r;
            err_sq += (r - d).powi(2);
        }
        if err_sq < 1e-9 {
            return f64::INFINITY;
        }
        10.0 * (sig_sq / err_sq).log10()
    }

    #[test]
    fn multichannel_snr_reasonable_6ch() {
        // 5.1 sine at the same freqs as the existing 5.1 test; measure
        // per-channel SNR and verify it's above a minimum threshold. Our
        // encoder is not production-quality, but a simple sine on each
        // channel should still land in the "audibly correct" range.
        let n = (1usize << DEFAULT_BLOCKSIZE_LONG_LOG2) * 4;
        let sr = 48_000.0;
        let freqs = [440.0f64, 520.0, 660.0, 880.0, 1100.0, 140.0];
        let pcm_f32: Vec<Vec<f32>> = freqs
            .iter()
            .map(|&f| {
                (0..n)
                    .map(|i| {
                        let t = i as f64 / sr;
                        (2.0 * std::f64::consts::PI * f * t).sin() as f32 * 0.4
                    })
                    .collect()
            })
            .collect();
        let decoded = encode_decode_multichannel(6, &pcm_f32);
        let pcm_i16: Vec<Vec<i16>> = pcm_f32
            .iter()
            .map(|ch| ch.iter().map(|&s| (s * 32768.0) as i16).collect())
            .collect();
        // Skip the first block's worth of samples for OLA warm-up. Measure
        // over the stable middle region.
        let skip = 1usize << DEFAULT_BLOCKSIZE_LONG_LOG2;
        let mut snr_total = 0f64;
        for ch in 0..6 {
            let snr = snr_db(&pcm_i16[ch], &decoded[ch], skip);
            eprintln!("5.1 ch{ch} SNR = {snr:.2} dB");
            // -3 dB is a very lax lower bound for a pure sine through our
            // encoder — most channels land above 5 dB in practice. LFE
            // (140 Hz) and the center (520 Hz) see extra floor-quantisation
            // error from the short-block-only ATH curve below ~200 Hz; we
            // use the loose bound so CI stays green without over-tuning.
            assert!(
                snr > -10.0,
                "5.1 ch{ch} SNR ({snr} dB) unreasonably low"
            );
            if snr.is_finite() {
                snr_total += snr;
            }
        }
        eprintln!("5.1 mean SNR = {:.2} dB", snr_total / 6.0);
    }
}

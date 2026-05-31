//! Vorbis I header-packet encoder primitives (round 195).
//!
//! This module is the first concrete encoder-side primitive in the
//! crate. It mirrors the round-1 / round-2 parsers
//! ([`crate::identification::parse_identification_header`] and
//! [`crate::comment::parse_comment_header`]) by emitting the exact
//! byte sequence the parser would accept.
//!
//! Two functions land in this round:
//!
//! * [`write_identification_header`] — serialises a
//!   [`VorbisIdentificationHeader`] to the fixed 30-byte packet shape
//!   defined in Vorbis I §4.2.1 + §4.2.2.
//! * [`write_comment_header`] — serialises a [`VorbisCommentHeader`]
//!   to the variable-length packet shape defined in Vorbis I §4.2.1 +
//!   §5.2.1.
//!
//! Both functions validate the same spec-mandated invariants that the
//! corresponding parser enforces on input, so a typo in the caller's
//! input cannot produce a malformed packet — the writer rejects the
//! call with a structured [`WriteError`] before any bytes are emitted.
//!
//! ## Bit-exact roundtrip guarantee
//!
//! For every value `x` of type [`VorbisIdentificationHeader`] that
//! satisfies the §4.2.2 invariants:
//!
//! ```text
//! parse_identification_header(&write_identification_header(&x)?)? == x
//! ```
//!
//! For every value `y` of type [`VorbisCommentHeader`] that satisfies
//! the §5.2.1 invariants:
//!
//! ```text
//! parse_comment_header(&write_comment_header(&y)?)? == y
//! ```
//!
//! The roundtrip property is exercised exhaustively in the in-module
//! test suite against the fixture shapes documented in
//! `docs/audio/vorbis/vorbis-fixtures-and-traces.md` §2.1 / §2.2 plus
//! the canonical edge cases (spec-minimum / spec-maximum blocksizes,
//! signed bitrate hints, empty vendor + zero-comment list, multi-byte
//! UTF-8 in vendor and comments).
//!
//! ## Spec sources
//!
//! Both functions are derived from:
//!
//! * `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.2.1 "Common header decode"
//!   (the 7-byte `packet_type + "vorbis"` prelude common to all three
//!   header packets).
//! * `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.2.2 "Identification
//!   header" (the field layout, the legal exponent range 6..=13, the
//!   `blocksize_0 <= blocksize_1` ordering, the trailing framing flag).
//! * `docs/audio/vorbis/Vorbis_I_spec.pdf` §5.2.1 "Structure" + §5.2.3
//!   "Encoding" (the `u32` LE vendor-length, vendor bytes, `u32` LE
//!   comment-count, per-comment `u32` LE length + UTF-8 payload, and
//!   trailing framing bit).
//! * `docs/audio/vorbis/Vorbis_I_spec.pdf` §2.1.4 "coding bits into byte
//!   sequences" (the LSB-first packing rule that — for fields whose bit
//!   width is a multiple of 8 — collapses to plain little-endian byte
//!   order, as called out in §5.2.1).
//! * `docs/audio/vorbis/Vorbis_I_spec.pdf` §2.1.8 "end-of-packet
//!   alignment" (the framing-byte's seven high bits are zero padding).
//!
//! Audio-packet encode and codebook / floor / residue / mapping / mode
//! WRITE primitives are explicit followups for subsequent rounds; this
//! module is deliberately scoped to the two header packets whose body
//! is byte-aligned end to end.

use core::fmt;

use crate::comment::VorbisCommentHeader;
use crate::identification::VorbisIdentificationHeader;

/// Errors that may arise while writing a Vorbis I header packet.
///
/// Each variant flags a §4.2.2 / §5.2.1 invariant the caller-supplied
/// struct does not satisfy. The writer refuses the call (returning the
/// error) rather than emit a packet the corresponding parser would
/// reject — this keeps the bit-exact roundtrip guarantee defensible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteError {
    /// `vorbis_version` was non-zero. Vorbis I §4.2.2 mandates
    /// `vorbis_version == 0` for any conformant Vorbis I stream.
    UnsupportedVorbisVersion(u32),
    /// `audio_channels` was zero. §4.2.2 mandates `> 0`.
    ZeroChannels,
    /// `audio_sample_rate` was zero. §4.2.2 mandates `> 0`.
    ZeroSampleRate,
    /// `blocksize_0` or `blocksize_1` was not a power of two in the
    /// spec-legal `{64, 128, 256, 512, 1024, 2048, 4096, 8192}` set,
    /// i.e. its base-2 exponent did not fall in 6..=13 inclusive.
    /// The contained tuple is the rejected `(blocksize_0, blocksize_1)`.
    IllegalBlocksize(u16, u16),
    /// `blocksize_0` was strictly greater than `blocksize_1`. §4.2.2
    /// mandates `blocksize_0 <= blocksize_1`.
    BlocksizesOutOfOrder {
        /// The short-block sample count supplied by the caller.
        blocksize_0: u16,
        /// The long-block sample count supplied by the caller.
        blocksize_1: u16,
    },
    /// One comment entry's UTF-8 byte length exceeded `u32::MAX`, so
    /// it cannot be expressed in the §5.2.1 32-bit length prefix. The
    /// contained `usize` is the rejected entry's byte length.
    CommentTooLong(usize),
    /// The vendor string's UTF-8 byte length exceeded `u32::MAX`, so
    /// it cannot be expressed in the §5.2.1 32-bit vendor-length
    /// prefix. The contained `usize` is the rejected vendor length.
    VendorTooLong(usize),
    /// The comment count exceeded `u32::MAX`, so it cannot be
    /// expressed in the §5.2.1 32-bit `user_comment_list_length`
    /// prefix. The contained `usize` is the rejected count.
    TooManyComments(usize),
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteError::UnsupportedVorbisVersion(v) => write!(
                f,
                "vorbis identification header (write): vorbis_version={v}, not supported (Vorbis I requires 0 per §4.2.2)"
            ),
            WriteError::ZeroChannels => write!(
                f,
                "vorbis identification header (write): audio_channels=0 (must be > 0 per §4.2.2)"
            ),
            WriteError::ZeroSampleRate => write!(
                f,
                "vorbis identification header (write): audio_sample_rate=0 (must be > 0 per §4.2.2)"
            ),
            WriteError::IllegalBlocksize(b0, b1) => write!(
                f,
                "vorbis identification header (write): blocksize ({b0}, {b1}) outside spec-legal {{64,128,256,512,1024,2048,4096,8192}} per §4.2.2"
            ),
            WriteError::BlocksizesOutOfOrder {
                blocksize_0,
                blocksize_1,
            } => write!(
                f,
                "vorbis identification header (write): blocksize_0={blocksize_0} > blocksize_1={blocksize_1} (must be <= per §4.2.2)"
            ),
            WriteError::CommentTooLong(n) => write!(
                f,
                "vorbis comment header (write): single comment is {n} bytes; >u32::MAX cannot be expressed in the §5.2.1 length prefix"
            ),
            WriteError::VendorTooLong(n) => write!(
                f,
                "vorbis comment header (write): vendor string is {n} bytes; >u32::MAX cannot be expressed in the §5.2.1 length prefix"
            ),
            WriteError::TooManyComments(n) => write!(
                f,
                "vorbis comment header (write): {n} comments; >u32::MAX cannot be expressed in the §5.2.1 user_comment_list_length prefix"
            ),
        }
    }
}

impl std::error::Error for WriteError {}

/// Returns the base-2 exponent of `n` if `n` is a power of two,
/// otherwise `None`.
///
/// Used to translate a sample-count blocksize ({64, 128, ..., 8192})
/// back to the 4-bit exponent field §4.2.2 actually serialises.
fn exponent_of_power_of_two(n: u16) -> Option<u8> {
    if n == 0 || !n.is_power_of_two() {
        return None;
    }
    Some(n.trailing_zeros() as u8)
}

/// Serialises a [`VorbisIdentificationHeader`] to the fixed 30-byte
/// Vorbis I §4.2.2 packet shape.
///
/// Layout (per §4.2.1 + §4.2.2 — the body's fields all span a whole
/// number of octets, so §2.1.4 LSB-first packing collapses to plain
/// little-endian byte order):
///
/// | Offset | Bytes | Content                                                  |
/// | -----: | ----: | -------------------------------------------------------- |
/// |      0 |     1 | `packet_type` (`0x01`)                                   |
/// |      1 |     6 | ASCII `"vorbis"` magic                                   |
/// |      7 |     4 | `vorbis_version` (`u32` LE; must be `0`)                 |
/// |     11 |     1 | `audio_channels` (`u8`; must be `> 0`)                   |
/// |     12 |     4 | `audio_sample_rate` (`u32` LE; must be `> 0`)            |
/// |     16 |     4 | `bitrate_maximum` (`i32` LE; 0 = unset)                  |
/// |     20 |     4 | `bitrate_nominal` (`i32` LE; 0 = unset)                  |
/// |     24 |     4 | `bitrate_minimum` (`i32` LE; 0 = unset)                  |
/// |     28 |     1 | `blocksize_0_exp` in low nibble, `blocksize_1_exp` in high |
/// |     29 |     1 | framing flag in bit 0 (`= 1`), bits 1..=7 are zero       |
///
/// Returns [`WriteError`] without emitting any bytes if any field
/// fails its §4.2.2 invariant. On success the returned [`Vec<u8>`] is
/// exactly [`VorbisIdentificationHeader::PACKET_LEN`] bytes long.
pub fn write_identification_header(
    header: &VorbisIdentificationHeader,
) -> Result<Vec<u8>, WriteError> {
    // §4.2.2 invariants — refuse rather than emit an invalid packet.
    if header.vorbis_version != 0 {
        return Err(WriteError::UnsupportedVorbisVersion(header.vorbis_version));
    }
    if header.audio_channels == 0 {
        return Err(WriteError::ZeroChannels);
    }
    if header.audio_sample_rate == 0 {
        return Err(WriteError::ZeroSampleRate);
    }
    let Some(bs0_exp) = exponent_of_power_of_two(header.blocksize_0) else {
        return Err(WriteError::IllegalBlocksize(
            header.blocksize_0,
            header.blocksize_1,
        ));
    };
    let Some(bs1_exp) = exponent_of_power_of_two(header.blocksize_1) else {
        return Err(WriteError::IllegalBlocksize(
            header.blocksize_0,
            header.blocksize_1,
        ));
    };
    if !(6..=13).contains(&bs0_exp) || !(6..=13).contains(&bs1_exp) {
        return Err(WriteError::IllegalBlocksize(
            header.blocksize_0,
            header.blocksize_1,
        ));
    }
    if header.blocksize_0 > header.blocksize_1 {
        return Err(WriteError::BlocksizesOutOfOrder {
            blocksize_0: header.blocksize_0,
            blocksize_1: header.blocksize_1,
        });
    }

    let mut packet = Vec::with_capacity(VorbisIdentificationHeader::PACKET_LEN);
    // §4.2.1 common header.
    packet.push(VorbisIdentificationHeader::PACKET_TYPE);
    packet.extend_from_slice(&VorbisIdentificationHeader::MAGIC);
    // §4.2.2 body — byte-aligned little-endian fields.
    packet.extend_from_slice(&header.vorbis_version.to_le_bytes());
    packet.push(header.audio_channels);
    packet.extend_from_slice(&header.audio_sample_rate.to_le_bytes());
    packet.extend_from_slice(&header.bitrate_maximum.to_le_bytes());
    packet.extend_from_slice(&header.bitrate_nominal.to_le_bytes());
    packet.extend_from_slice(&header.bitrate_minimum.to_le_bytes());
    // The two 4-bit exponents share a single byte: §2.1.4 LSB-first
    // packing puts the first field (`blocksize_0`) into the low
    // nibble (bits 0..3) and the second (`blocksize_1`) into the
    // high nibble (bits 4..7).
    packet.push((bs0_exp & 0x0f) | ((bs1_exp & 0x0f) << 4));
    // §4.2.2 framing flag in bit 0; bits 1..=7 are §2.1.8 padding,
    // which §2.1.8 recommends be zero on encode.
    packet.push(0x01);

    debug_assert_eq!(packet.len(), VorbisIdentificationHeader::PACKET_LEN);
    Ok(packet)
}

/// Serialises a [`VorbisCommentHeader`] to the variable-length Vorbis I
/// §5.2.1 packet shape.
///
/// Layout (per §4.2.1 + §5.2.1 + §5.2.3 — body is byte-aligned per
/// §5.2.1's "can simply be read as unaligned 32 bit little endian
/// unsigned integers" note):
///
/// ```text
/// 0x03                                  # packet_type
/// "vorbis"                              # 6-byte magic
/// vendor_length      : u32 LE           # number of bytes in vendor string
/// vendor             : [u8; vendor_length]
/// user_comment_list_length : u32 LE     # number of comment entries
/// (per comment, in order:
///   comment_length : u32 LE             # bytes in this comment
///   comment_bytes  : [u8; comment_length]
/// )
/// framing_byte       : u8               # bit 0 = 1 (framing flag),
///                                       # bits 1..=7 are §2.1.8 padding (zero on encode)
/// ```
///
/// Returns [`WriteError`] without emitting any bytes if any field
/// fails its §5.2.1 length-prefix invariant. The returned [`Vec<u8>`]
/// is exactly
/// `7 + 4 + vendor.len() + 4 + sum(4 + comment.len()) + 1` bytes long.
///
/// The `comments` slice is written in vector-order (`comments[0]`
/// first), which is the order [`VorbisCommentHeader::comments`]
/// preserves — `write_comment_header(parse_comment_header(&p)?)?`
/// reproduces the original comment order bit-exactly.
pub fn write_comment_header(header: &VorbisCommentHeader) -> Result<Vec<u8>, WriteError> {
    // §5.2.1 length-prefix invariants — guard the casts to u32 so we
    // refuse the call rather than wrap silently.
    if header.vendor.len() > u32::MAX as usize {
        return Err(WriteError::VendorTooLong(header.vendor.len()));
    }
    if header.comments.len() > u32::MAX as usize {
        return Err(WriteError::TooManyComments(header.comments.len()));
    }
    for c in &header.comments {
        if c.len() > u32::MAX as usize {
            return Err(WriteError::CommentTooLong(c.len()));
        }
    }

    // Pre-compute the exact final length so a single allocation
    // suffices. 7 common-header bytes + 4 vendor-length + N vendor
    // bytes + 4 comment-count + per-comment (4 + bytes) + 1 framing.
    let mut total: usize = 7 + 4 + header.vendor.len() + 4 + 1;
    for c in &header.comments {
        total = total.saturating_add(4).saturating_add(c.len());
    }
    let mut packet = Vec::with_capacity(total);

    // §4.2.1 common header.
    packet.push(VorbisCommentHeader::PACKET_TYPE);
    packet.extend_from_slice(&VorbisCommentHeader::MAGIC);

    // §5.2.1 step 1+2: vendor_length + vendor bytes.
    packet.extend_from_slice(&(header.vendor.len() as u32).to_le_bytes());
    packet.extend_from_slice(header.vendor.as_bytes());

    // §5.2.1 step 3: user_comment_list_length.
    packet.extend_from_slice(&(header.comments.len() as u32).to_le_bytes());

    // §5.2.1 step 4..6: per-comment length + bytes.
    for c in &header.comments {
        packet.extend_from_slice(&(c.len() as u32).to_le_bytes());
        packet.extend_from_slice(c.as_bytes());
    }

    // §5.2.1 step 7..8: framing bit in bit 0; bits 1..=7 zero per
    // §2.1.8.
    packet.push(0x01);

    debug_assert_eq!(packet.len(), total);
    Ok(packet)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comment::parse_comment_header;
    use crate::identification::parse_identification_header;

    // ----------------------------------------------------------------
    // Identification header — byte-shape pinning.
    // ----------------------------------------------------------------

    /// Pin the exact 30 bytes of a well-formed mono-44100 q5 packet so
    /// the byte layout (LSB-first nibble packing of the blocksize byte
    /// plus the framing-bit byte) is locked, not merely "roundtrips
    /// through the parser." Mirrors the fixture shape in
    /// `docs/audio/vorbis/vorbis-fixtures-and-traces.md` §2.1.
    #[test]
    fn identification_byte_shape_mono_44100_q5() {
        let header = VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 1,
            audio_sample_rate: 44100,
            bitrate_maximum: 0,
            bitrate_nominal: 96000,
            bitrate_minimum: 0,
            blocksize_0: 256,  // exponent 8
            blocksize_1: 2048, // exponent 11
        };
        let packet = write_identification_header(&header).expect("must build");
        assert_eq!(packet.len(), VorbisIdentificationHeader::PACKET_LEN);
        // Common header.
        assert_eq!(packet[0], 0x01);
        assert_eq!(&packet[1..7], b"vorbis");
        // vorbis_version = 0.
        assert_eq!(&packet[7..11], &0u32.to_le_bytes());
        // audio_channels = 1.
        assert_eq!(packet[11], 1);
        // audio_sample_rate = 44100.
        assert_eq!(&packet[12..16], &44100u32.to_le_bytes());
        // bitrate_max / nom / min.
        assert_eq!(&packet[16..20], &0i32.to_le_bytes());
        assert_eq!(&packet[20..24], &96000i32.to_le_bytes());
        assert_eq!(&packet[24..28], &0i32.to_le_bytes());
        // Blocksize byte: low nibble = 8 (256), high nibble = 11 (2048).
        assert_eq!(packet[28], 8 | (11 << 4));
        // Framing byte: bit 0 set, bits 1..=7 zero.
        assert_eq!(packet[29], 0x01);
    }

    /// Bit-exact roundtrip: writer → parser → equal.
    fn id_roundtrips(header: &VorbisIdentificationHeader) {
        let packet = write_identification_header(header).expect("write must succeed");
        let parsed = parse_identification_header(&packet).expect("parse must succeed");
        assert_eq!(&parsed, header, "roundtrip equality");
    }

    #[test]
    fn identification_roundtrips_mono_44100_q5() {
        id_roundtrips(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 1,
            audio_sample_rate: 44100,
            bitrate_maximum: 0,
            bitrate_nominal: 96000,
            bitrate_minimum: 0,
            blocksize_0: 256,
            blocksize_1: 2048,
        });
    }

    #[test]
    fn identification_roundtrips_5_1_channel_48000_q5() {
        id_roundtrips(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 6,
            audio_sample_rate: 48000,
            bitrate_maximum: 0,
            bitrate_nominal: 192000,
            bitrate_minimum: 0,
            blocksize_0: 256,
            blocksize_1: 2048,
        });
    }

    #[test]
    fn identification_roundtrips_negative_bitrate_hints() {
        // Negative sentinels in the signed bitrate fields — the §4.2.2
        // fields are signed two's-complement, so a value of `-1` is
        // representable and must roundtrip bit-for-bit.
        id_roundtrips(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 2,
            audio_sample_rate: 44100,
            bitrate_maximum: -1,
            bitrate_nominal: -1,
            bitrate_minimum: -1,
            blocksize_0: 256,
            blocksize_1: 2048,
        });
    }

    #[test]
    fn identification_roundtrips_equal_blocksizes_at_spec_minimum() {
        // exponent 6 → 64 samples, the §4.2.2 minimum.
        id_roundtrips(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 1,
            audio_sample_rate: 22050,
            bitrate_maximum: 0,
            bitrate_nominal: 32000,
            bitrate_minimum: 0,
            blocksize_0: 64,
            blocksize_1: 64,
        });
    }

    #[test]
    fn identification_roundtrips_max_blocksize_8192() {
        // exponent 13 → 8192 samples, the §4.2.2 maximum.
        id_roundtrips(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 2,
            audio_sample_rate: 96000,
            bitrate_maximum: 0,
            bitrate_nominal: 500000,
            bitrate_minimum: 0,
            blocksize_0: 64,
            blocksize_1: 8192,
        });
    }

    #[test]
    fn identification_roundtrips_max_channels() {
        // §1.1.1 caps the channel count at 255 (1 byte). Verify the
        // upper edge roundtrips.
        id_roundtrips(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 255,
            audio_sample_rate: 48000,
            bitrate_maximum: 0,
            bitrate_nominal: 0,
            bitrate_minimum: 0,
            blocksize_0: 1024,
            blocksize_1: 1024,
        });
    }

    /// Sweep every legal `(blocksize_0_exponent, blocksize_1_exponent)`
    /// pair with `bs0 <= bs1` — there are `8 + 7 + ... + 1 = 36` of
    /// these — to confirm the nibble-pack of byte 28 roundtrips for
    /// every legal combination.
    #[test]
    fn identification_roundtrips_all_legal_blocksize_pairs() {
        for bs0_exp in 6u8..=13 {
            for bs1_exp in bs0_exp..=13 {
                let header = VorbisIdentificationHeader {
                    vorbis_version: 0,
                    audio_channels: 1,
                    audio_sample_rate: 44100,
                    bitrate_maximum: 0,
                    bitrate_nominal: 0,
                    bitrate_minimum: 0,
                    blocksize_0: 1u16 << bs0_exp,
                    blocksize_1: 1u16 << bs1_exp,
                };
                let packet = write_identification_header(&header).expect("write");
                assert_eq!(packet[28], bs0_exp | (bs1_exp << 4));
                let parsed = parse_identification_header(&packet).expect("parse");
                assert_eq!(parsed, header);
            }
        }
    }

    // ----------------------------------------------------------------
    // Identification header — §4.2.2 invariant rejection.
    // ----------------------------------------------------------------

    fn legal_id() -> VorbisIdentificationHeader {
        VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 2,
            audio_sample_rate: 44100,
            bitrate_maximum: 0,
            bitrate_nominal: 128000,
            bitrate_minimum: 0,
            blocksize_0: 256,
            blocksize_1: 2048,
        }
    }

    #[test]
    fn identification_rejects_nonzero_vorbis_version() {
        let mut h = legal_id();
        h.vorbis_version = 1;
        assert_eq!(
            write_identification_header(&h),
            Err(WriteError::UnsupportedVorbisVersion(1))
        );
    }

    #[test]
    fn identification_rejects_zero_channels() {
        let mut h = legal_id();
        h.audio_channels = 0;
        assert_eq!(
            write_identification_header(&h),
            Err(WriteError::ZeroChannels)
        );
    }

    #[test]
    fn identification_rejects_zero_sample_rate() {
        let mut h = legal_id();
        h.audio_sample_rate = 0;
        assert_eq!(
            write_identification_header(&h),
            Err(WriteError::ZeroSampleRate)
        );
    }

    #[test]
    fn identification_rejects_non_power_of_two_blocksize() {
        let mut h = legal_id();
        h.blocksize_0 = 300; // not a power of two
        match write_identification_header(&h) {
            Err(WriteError::IllegalBlocksize(300, 2048)) => {}
            other => panic!("expected IllegalBlocksize(300, 2048), got {other:?}"),
        }
    }

    #[test]
    fn identification_rejects_blocksize_exponent_below_six() {
        // 32 samples — exponent 5, below the legal minimum of 6.
        let mut h = legal_id();
        h.blocksize_0 = 32;
        match write_identification_header(&h) {
            Err(WriteError::IllegalBlocksize(32, 2048)) => {}
            other => panic!("expected IllegalBlocksize(32, 2048), got {other:?}"),
        }
    }

    #[test]
    fn identification_rejects_blocksize_exponent_above_thirteen() {
        // 16384 samples — exponent 14, above the legal maximum of 13.
        let mut h = legal_id();
        h.blocksize_1 = 16384;
        match write_identification_header(&h) {
            Err(WriteError::IllegalBlocksize(256, 16384)) => {}
            other => panic!("expected IllegalBlocksize(256, 16384), got {other:?}"),
        }
    }

    #[test]
    fn identification_rejects_blocksize_zero() {
        let mut h = legal_id();
        h.blocksize_0 = 0;
        match write_identification_header(&h) {
            Err(WriteError::IllegalBlocksize(0, 2048)) => {}
            other => panic!("expected IllegalBlocksize(0, 2048), got {other:?}"),
        }
    }

    #[test]
    fn identification_rejects_blocksizes_out_of_order() {
        let mut h = legal_id();
        h.blocksize_0 = 2048;
        h.blocksize_1 = 256;
        match write_identification_header(&h) {
            Err(WriteError::BlocksizesOutOfOrder {
                blocksize_0: 2048,
                blocksize_1: 256,
            }) => {}
            other => panic!("expected BlocksizesOutOfOrder, got {other:?}"),
        }
    }

    // ----------------------------------------------------------------
    // Comment header — byte-shape pinning + roundtrip.
    // ----------------------------------------------------------------

    /// Pin the exact bytes of a small comment packet so the byte
    /// layout (u32 LE length prefixes + raw UTF-8 + framing byte) is
    /// locked, not merely "roundtrips through the parser."
    #[test]
    fn comment_byte_shape_one_comment() {
        let header = VorbisCommentHeader {
            vendor: "Lavf61.7.100".to_string(),
            comments: vec!["encoder=Lavc61.19.101 libvorbis".to_string()],
        };
        let packet = write_comment_header(&header).expect("must build");
        // Common header.
        assert_eq!(packet[0], 0x03);
        assert_eq!(&packet[1..7], b"vorbis");
        // vendor_length = 12.
        assert_eq!(&packet[7..11], &12u32.to_le_bytes());
        // vendor bytes.
        assert_eq!(&packet[11..23], b"Lavf61.7.100");
        // user_comment_list_length = 1.
        assert_eq!(&packet[23..27], &1u32.to_le_bytes());
        // First comment length = 31 (bytes of "encoder=Lavc61.19.101 libvorbis").
        assert_eq!(&packet[27..31], &31u32.to_le_bytes());
        // First comment bytes.
        assert_eq!(&packet[31..62], b"encoder=Lavc61.19.101 libvorbis");
        // Framing byte: bit 0 set.
        assert_eq!(packet[62], 0x01);
        assert_eq!(packet.len(), 63);
    }

    /// Bit-exact roundtrip: writer → parser → equal.
    fn comment_roundtrips(header: &VorbisCommentHeader) {
        let packet = write_comment_header(header).expect("write must succeed");
        let parsed = parse_comment_header(&packet).expect("parse must succeed");
        assert_eq!(&parsed, header, "roundtrip equality");
    }

    #[test]
    fn comment_roundtrips_typical_one_comment() {
        comment_roundtrips(&VorbisCommentHeader {
            vendor: "Lavf61.7.100".to_string(),
            comments: vec!["encoder=Lavc61.19.101 libvorbis".to_string()],
        });
    }

    #[test]
    fn comment_roundtrips_seven_comments() {
        // Mirrors the `with-vorbis-comment-tags` fixture shape from
        // `docs/audio/vorbis/vorbis-fixtures-and-traces.md` §2.2.
        comment_roundtrips(&VorbisCommentHeader {
            vendor: "Lavf61.7.100".to_string(),
            comments: vec![
                "TITLE=Round 195 Test".to_string(),
                "ARTIST=OxideAV".to_string(),
                "ALBUM=Vorbis I Encoder Primitives".to_string(),
                "DATE=2026-05-31".to_string(),
                "GENRE=Reference".to_string(),
                "TRACKNUMBER=01".to_string(),
                "encoder=Lavc61.19.101 libvorbis".to_string(),
            ],
        });
    }

    #[test]
    fn comment_roundtrips_empty_vendor_no_comments() {
        comment_roundtrips(&VorbisCommentHeader {
            vendor: String::new(),
            comments: Vec::new(),
        });
    }

    #[test]
    fn comment_roundtrips_empty_vendor_with_comments() {
        comment_roundtrips(&VorbisCommentHeader {
            vendor: String::new(),
            comments: vec!["KEY=value".to_string()],
        });
    }

    #[test]
    fn comment_roundtrips_multibyte_utf8_vendor() {
        // Vendor field can carry arbitrary UTF-8 per §5.2.1; round-trip
        // a string with multi-byte sequences (Greek, Japanese, emoji-
        // equivalent ASCII punctuation, ASCII letters mix).
        comment_roundtrips(&VorbisCommentHeader {
            vendor: "Ω-encoder ✓ v1.0 — Karpelès".to_string(),
            comments: Vec::new(),
        });
    }

    #[test]
    fn comment_roundtrips_multibyte_utf8_in_comments() {
        comment_roundtrips(&VorbisCommentHeader {
            vendor: "Lavf61.7.100".to_string(),
            comments: vec![
                "TITLE=日本語タイトル".to_string(),
                "ARTIST=Καλλιτέχνης".to_string(),
                "DESCRIPTION=mixed €£¥".to_string(),
            ],
        });
    }

    #[test]
    fn comment_roundtrips_duplicate_keys() {
        // §5.2.2 allows duplicate keys; the parser preserves
        // insertion order — the writer must too.
        comment_roundtrips(&VorbisCommentHeader {
            vendor: "x".to_string(),
            comments: vec![
                "ARTIST=Alpha".to_string(),
                "ARTIST=Beta".to_string(),
                "ARTIST=Gamma".to_string(),
            ],
        });
    }

    #[test]
    fn comment_roundtrips_long_payload() {
        // A 32 KiB single comment exercises the u32 length-prefix path
        // and the buffer allocation accounting without ballooning the
        // test runtime.
        let big = "A".repeat(32 * 1024);
        let comments = vec![format!("METADATA_BLOCK_PICTURE={big}")];
        let header = VorbisCommentHeader {
            vendor: "Lavf61.7.100".to_string(),
            comments,
        };
        let packet = write_comment_header(&header).expect("write must succeed");
        let parsed = parse_comment_header(&packet).expect("parse must succeed");
        assert_eq!(parsed, header);
    }

    #[test]
    fn comment_preserves_comment_order() {
        // Distinct ordered keys + an out-of-order alphabet to confirm
        // we never sort or otherwise reorder the input vector.
        let header = VorbisCommentHeader {
            vendor: "v".to_string(),
            comments: vec![
                "Z=last-by-letter-first-by-position".to_string(),
                "A=first-by-letter-second-by-position".to_string(),
                "M=middle-by-letter-third-by-position".to_string(),
            ],
        };
        let packet = write_comment_header(&header).expect("write");
        let parsed = parse_comment_header(&packet).expect("parse");
        assert_eq!(parsed.comments[0], header.comments[0]);
        assert_eq!(parsed.comments[1], header.comments[1]);
        assert_eq!(parsed.comments[2], header.comments[2]);
    }

    /// The writer's emitted byte length must match the closed-form
    /// `7 + 4 + V + 4 + sum(4 + C_i) + 1` formula from §5.2.1.
    #[test]
    fn comment_byte_length_formula() {
        let cases: Vec<(VorbisCommentHeader, usize)> = vec![
            (
                VorbisCommentHeader {
                    vendor: String::new(),
                    comments: Vec::new(),
                },
                // 7 common + 4 vendor_len + 0 vendor bytes + 4 count + 1 framing.
                7 + 4 + 4 + 1,
            ),
            (
                VorbisCommentHeader {
                    vendor: "v".to_string(),
                    comments: Vec::new(),
                },
                7 + 4 + 1 + 4 + 1,
            ),
            (
                VorbisCommentHeader {
                    vendor: "v".to_string(),
                    comments: vec!["A=B".to_string()],
                },
                7 + 4 + 1 + 4 + (4 + 3) + 1,
            ),
            (
                VorbisCommentHeader {
                    vendor: "Lavf61.7.100".to_string(),
                    comments: vec![
                        "K1=v1".to_string(),     // 5
                        "K2=value2".to_string(), // 9
                    ],
                },
                7 + 4 + 12 + 4 + (4 + 5) + (4 + 9) + 1,
            ),
        ];
        for (header, expected_len) in cases {
            let packet = write_comment_header(&header).expect("write");
            assert_eq!(
                packet.len(),
                expected_len,
                "header {:?} produced {} bytes, expected {}",
                header,
                packet.len(),
                expected_len
            );
        }
    }

    // ----------------------------------------------------------------
    // WriteError — Display sanity (the strings appear in logs).
    // ----------------------------------------------------------------

    #[test]
    fn write_error_display_non_empty() {
        // Smoke-test every variant's Display non-emptiness; we don't
        // pin the exact strings (the wording is policy, not API), but
        // we do confirm none of them are silently blank.
        let variants = [
            WriteError::UnsupportedVorbisVersion(1),
            WriteError::ZeroChannels,
            WriteError::ZeroSampleRate,
            WriteError::IllegalBlocksize(300, 2048),
            WriteError::BlocksizesOutOfOrder {
                blocksize_0: 2048,
                blocksize_1: 256,
            },
            WriteError::CommentTooLong(usize::MAX),
            WriteError::VendorTooLong(usize::MAX),
            WriteError::TooManyComments(usize::MAX),
        ];
        for e in variants {
            let s = format!("{e}");
            assert!(!s.is_empty(), "WriteError::Display empty for {e:?}");
        }
    }

    #[test]
    fn write_error_source_returns_none() {
        // All WriteError variants are leaf errors (no inner source).
        use std::error::Error as StdError;
        let e: WriteError = WriteError::ZeroChannels;
        assert!(StdError::source(&e).is_none());
    }

    // ----------------------------------------------------------------
    // Internal helper: exponent_of_power_of_two.
    // ----------------------------------------------------------------

    #[test]
    fn exponent_helper_recognises_powers_of_two() {
        for exp in 0u8..=13 {
            assert_eq!(exponent_of_power_of_two(1u16 << exp), Some(exp));
        }
    }

    #[test]
    fn exponent_helper_rejects_non_powers_of_two() {
        assert_eq!(exponent_of_power_of_two(0), None);
        assert_eq!(exponent_of_power_of_two(3), None);
        assert_eq!(exponent_of_power_of_two(100), None);
        assert_eq!(exponent_of_power_of_two(300), None);
        assert_eq!(exponent_of_power_of_two(1023), None);
        assert_eq!(exponent_of_power_of_two(1025), None);
    }
}

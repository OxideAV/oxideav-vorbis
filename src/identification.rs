//! Vorbis I identification header parser (Vorbis I §4.2.2).
//!
//! The identification header is the first of three required Vorbis header
//! packets. It declares the stream as Vorbis and carries the global audio
//! parameters: codec version, channel count, sample rate, bitrate hints,
//! and the two MDCT block sizes.
//!
//! ## Packet layout
//!
//! Per the Vorbis I Specification (Xiph.Org, 2020-07-04 revision):
//!
//! * Common header (Vorbis I §4.2.1) — 7 bytes:
//!   - byte 0: `packet_type` (`0x01` for identification)
//!   - bytes 1..7: ASCII `"vorbis"` (`0x76 0x6f 0x72 0x62 0x69 0x73`)
//! * Identification-header payload (Vorbis I §4.2.2) — bit-packed per
//!   the §2.1.4 LSB-first convention:
//!   - 32 bits `vorbis_version` (unsigned, must be 0 for Vorbis I)
//!   - 8 bits  `audio_channels` (unsigned, must be > 0)
//!   - 32 bits `audio_sample_rate` (unsigned, must be > 0)
//!   - 32 bits `bitrate_maximum` (signed, two's complement; hint, 0 = unset)
//!   - 32 bits `bitrate_nominal` (signed)
//!   - 32 bits `bitrate_minimum` (signed)
//!   - 4 bits  `blocksize_0` (exponent: legal range 6..=13 inclusive, so the
//!     resulting block size lies in {64, 128, 256, 512, 1024, 2048, 4096, 8192})
//!   - 4 bits  `blocksize_1` (exponent; must be ≥ `blocksize_0`)
//!   - 1 bit   `framing_flag` (must be 1)
//!
//! Because every field above the blocksize byte spans a whole number of
//! octets and the bitpacking convention writes the LSB of each field
//! first into the least-significant unused bit of the destination byte
//! (§2.1.4), the payload is exactly:
//!
//! * 4 + 1 + 4 + 4 + 4 + 4 = 21 bytes of byte-aligned little-endian fields,
//! * 1 byte holding the two 4-bit blocksize exponents (`blocksize_0` in
//!   the low nibble, `blocksize_1` in the high nibble),
//! * 1 byte whose bit 0 is the framing flag and whose remaining 7 bits
//!   are padding (zeroed per §2.1.8).
//!
//! Total identification header packet length: `7 + 21 + 1 + 1 = 30` bytes.
//!
//! ## What this module is, and is not
//!
//! This round-1 module ships the identification-header parse only. It
//! does **not** parse the comment header (§5), the setup header (§4.2.4),
//! any codebook (§3), floor (§6, §7), residue (§8), mapping (§4.3),
//! mode (§4.3.1), or audio packet (§4.3.2). All of those are explicit
//! followups for subsequent rounds.

/// Parsed Vorbis I identification header (§4.2.2).
///
/// Values are exposed as the spec encodes them. In particular,
/// [`Self::blocksize_0`] and [`Self::blocksize_1`] hold the
/// resolved sample counts (i.e. `1 << exponent`), not the raw
/// 4-bit exponents read from the packet. The bitrate hints are
/// `i32` as Vorbis encodes them as signed integers and a value of
/// 0 means "encoder did not declare this hint" (§4.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VorbisIdentificationHeader {
    /// Vorbis codec version. Must be 0 for any Vorbis I stream.
    pub vorbis_version: u32,
    /// PCM channel count. Vorbis I supports 1..=255 discrete channels
    /// (Vorbis I §1.1.1). Must be > 0.
    pub audio_channels: u8,
    /// PCM sample rate in Hz. Must be > 0.
    pub audio_sample_rate: u32,
    /// Maximum-bitrate hint (signed; 0 = unset, see §4.2.2).
    pub bitrate_maximum: i32,
    /// Nominal-bitrate hint (signed; 0 = unset).
    pub bitrate_nominal: i32,
    /// Minimum-bitrate hint (signed; 0 = unset).
    pub bitrate_minimum: i32,
    /// Resolved short-block sample count (`1 << exponent`). One of
    /// {64, 128, 256, 512, 1024, 2048, 4096, 8192} per §4.2.2.
    pub blocksize_0: u16,
    /// Resolved long-block sample count. One of the same set as
    /// [`Self::blocksize_0`], and must satisfy
    /// `blocksize_1 >= blocksize_0`.
    pub blocksize_1: u16,
}

impl VorbisIdentificationHeader {
    /// Length, in bytes, of a well-formed Vorbis I identification-header
    /// packet (common header + payload + framing byte). Per §4.2.2 a
    /// conformant packet has exactly this length.
    pub const PACKET_LEN: usize = 30;

    /// Common-header packet-type byte for the identification header
    /// (Vorbis I §4.2.1). The comment header is `3`, the setup
    /// header is `5`; audio packets have bit 0 of the first byte
    /// clear (i.e. packet_type is even).
    pub const PACKET_TYPE: u8 = 0x01;

    /// The six magic bytes that follow the packet-type byte in every
    /// Vorbis header packet (Vorbis I §4.2.1).
    pub const MAGIC: [u8; 6] = *b"vorbis";

    /// Returns the unique long block size in samples (max of
    /// `blocksize_0` and `blocksize_1`; in practice this is
    /// `blocksize_1` since the spec mandates `blocksize_1 >=
    /// blocksize_0`).
    #[must_use]
    pub fn long_block_samples(self) -> u16 {
        self.blocksize_1
    }

    /// Returns the unique short block size in samples (`blocksize_0`).
    #[must_use]
    pub fn short_block_samples(self) -> u16 {
        self.blocksize_0
    }
}

/// Errors that may arise while parsing a Vorbis identification header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// The supplied packet was shorter than the 30-byte identification
    /// header (Vorbis I §4.2.2). The contained value is the actual
    /// length of the supplied buffer.
    PacketTooShort(usize),
    /// The 1-byte common-header packet-type did not equal
    /// [`VorbisIdentificationHeader::PACKET_TYPE`].
    WrongPacketType(u8),
    /// The six bytes following the packet type were not the ASCII
    /// magic string `"vorbis"`.
    BadMagic,
    /// The `vorbis_version` field was non-zero. Vorbis I §4.2.2 mandates
    /// `vorbis_version == 0` for any conformant Vorbis I stream.
    UnsupportedVorbisVersion(u32),
    /// The `audio_channels` field was zero. §4.2.2 mandates `> 0`.
    ZeroChannels,
    /// The `audio_sample_rate` field was zero. §4.2.2 mandates `> 0`.
    ZeroSampleRate,
    /// One or both block-size exponents fell outside the spec-legal
    /// range of 6..=13 inclusive, i.e. the resulting block sample
    /// count was not in `{64, 128, 256, 512, 1024, 2048, 4096, 8192}`.
    /// The contained tuple is `(blocksize_0_exponent, blocksize_1_exponent)`.
    IllegalBlocksizeExponent(u8, u8),
    /// `blocksize_0` was strictly greater than `blocksize_1`, which
    /// §4.2.2 forbids ("must be less than or equal to").
    BlocksizesOutOfOrder {
        /// The short-block sample count (computed from the raw 4-bit exponent).
        blocksize_0: u16,
        /// The long-block sample count (computed from the raw 4-bit exponent).
        blocksize_1: u16,
    },
    /// The trailing framing bit was 0; §4.2.2 mandates "must be nonzero".
    BadFramingFlag,
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ParseError::PacketTooShort(n) => write!(
                f,
                "vorbis identification header: packet too short ({n} bytes; need {} per §4.2.2)",
                VorbisIdentificationHeader::PACKET_LEN
            ),
            ParseError::WrongPacketType(t) => write!(
                f,
                "vorbis identification header: wrong packet_type byte 0x{t:02x} (expected 0x01 per §4.2.1)"
            ),
            ParseError::BadMagic => write!(
                f,
                "vorbis identification header: missing 'vorbis' magic per §4.2.1"
            ),
            ParseError::UnsupportedVorbisVersion(v) => write!(
                f,
                "vorbis identification header: vorbis_version={v}, not supported (Vorbis I requires 0 per §4.2.2)"
            ),
            ParseError::ZeroChannels => write!(
                f,
                "vorbis identification header: audio_channels=0 (must be > 0 per §4.2.2)"
            ),
            ParseError::ZeroSampleRate => write!(
                f,
                "vorbis identification header: audio_sample_rate=0 (must be > 0 per §4.2.2)"
            ),
            ParseError::IllegalBlocksizeExponent(b0, b1) => write!(
                f,
                "vorbis identification header: blocksize exponents ({b0}, {b1}) outside legal 6..=13 range per §4.2.2"
            ),
            ParseError::BlocksizesOutOfOrder {
                blocksize_0,
                blocksize_1,
            } => write!(
                f,
                "vorbis identification header: blocksize_0={blocksize_0} > blocksize_1={blocksize_1} (must be <= per §4.2.2)"
            ),
            ParseError::BadFramingFlag => write!(
                f,
                "vorbis identification header: framing_flag=0 (must be nonzero per §4.2.2)"
            ),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parses a Vorbis I identification-header packet from `packet`.
///
/// The byte buffer must contain the entire packet, starting with the
/// `packet_type` byte (§4.2.1). The function validates the common
/// header, the spec-mandated invariants (vorbis_version, channels,
/// sample_rate, blocksize order, framing flag) and returns the
/// parsed [`VorbisIdentificationHeader`].
///
/// Returns [`ParseError`] on any deviation from §4.2.2.
pub fn parse_identification_header(
    packet: &[u8],
) -> Result<VorbisIdentificationHeader, ParseError> {
    if packet.len() < VorbisIdentificationHeader::PACKET_LEN {
        return Err(ParseError::PacketTooShort(packet.len()));
    }

    // Common header (§4.2.1): 0x01 + "vorbis".
    if packet[0] != VorbisIdentificationHeader::PACKET_TYPE {
        return Err(ParseError::WrongPacketType(packet[0]));
    }
    if packet[1..7] != VorbisIdentificationHeader::MAGIC {
        return Err(ParseError::BadMagic);
    }

    // Byte-aligned little-endian fields (§2.1.4 LSB-first packing of
    // multi-of-8 fields collapses to plain little-endian byte order).
    let vorbis_version = u32::from_le_bytes([packet[7], packet[8], packet[9], packet[10]]);
    if vorbis_version != 0 {
        return Err(ParseError::UnsupportedVorbisVersion(vorbis_version));
    }

    let audio_channels = packet[11];
    if audio_channels == 0 {
        return Err(ParseError::ZeroChannels);
    }

    let audio_sample_rate = u32::from_le_bytes([packet[12], packet[13], packet[14], packet[15]]);
    if audio_sample_rate == 0 {
        return Err(ParseError::ZeroSampleRate);
    }

    let bitrate_maximum = i32::from_le_bytes([packet[16], packet[17], packet[18], packet[19]]);
    let bitrate_nominal = i32::from_le_bytes([packet[20], packet[21], packet[22], packet[23]]);
    let bitrate_minimum = i32::from_le_bytes([packet[24], packet[25], packet[26], packet[27]]);

    // The blocksize byte packs two 4-bit unsigned exponents. Per §2.1.4
    // LSB-first packing: the first field's LSB is written into the
    // least-significant unused bit position of the destination byte, so
    // the first 4-bit field (`blocksize_0`) occupies bits 0..3 (the low
    // nibble) and the second (`blocksize_1`) occupies bits 4..7 (high
    // nibble).
    let blocksize_byte = packet[28];
    let bs0_exp = blocksize_byte & 0x0f;
    let bs1_exp = blocksize_byte >> 4;
    if !(6..=13).contains(&bs0_exp) || !(6..=13).contains(&bs1_exp) {
        return Err(ParseError::IllegalBlocksizeExponent(bs0_exp, bs1_exp));
    }
    let blocksize_0: u16 = 1u16 << bs0_exp;
    let blocksize_1: u16 = 1u16 << bs1_exp;
    if blocksize_0 > blocksize_1 {
        return Err(ParseError::BlocksizesOutOfOrder {
            blocksize_0,
            blocksize_1,
        });
    }

    // The framing flag is the very first (LSB) bit of byte 29 per the
    // LSB-first §2.1.4 packing convention; the remaining 7 bits are
    // zero-padding per §2.1.8.
    let framing_byte = packet[29];
    if framing_byte & 0x01 == 0 {
        return Err(ParseError::BadFramingFlag);
    }

    Ok(VorbisIdentificationHeader {
        vorbis_version,
        audio_channels,
        audio_sample_rate,
        bitrate_maximum,
        bitrate_nominal,
        bitrate_minimum,
        blocksize_0,
        blocksize_1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic identification-header packet per §4.2.2.
    /// `bs0_exp` / `bs1_exp` are raw 4-bit exponents (e.g. 8 for a
    /// 256-sample block); `framing` is the 1-bit framing flag.
    #[allow(clippy::too_many_arguments)]
    fn build_id_packet(
        version: u32,
        channels: u8,
        sample_rate: u32,
        br_max: i32,
        br_nom: i32,
        br_min: i32,
        bs0_exp: u8,
        bs1_exp: u8,
        framing: u8,
    ) -> [u8; 30] {
        let mut p = [0u8; 30];
        p[0] = 0x01;
        p[1..7].copy_from_slice(b"vorbis");
        p[7..11].copy_from_slice(&version.to_le_bytes());
        p[11] = channels;
        p[12..16].copy_from_slice(&sample_rate.to_le_bytes());
        p[16..20].copy_from_slice(&br_max.to_le_bytes());
        p[20..24].copy_from_slice(&br_nom.to_le_bytes());
        p[24..28].copy_from_slice(&br_min.to_le_bytes());
        // blocksize_0 in low nibble, blocksize_1 in high nibble.
        p[28] = (bs0_exp & 0x0f) | ((bs1_exp & 0x0f) << 4);
        // Framing bit in bit 0 of byte 29, remaining bits zero-padded.
        p[29] = framing & 0x01;
        p
    }

    /// Canonical mono-44100 q5 fixture shape per
    /// `docs/audio/vorbis/vorbis-fixtures-and-traces.md` §2.1.
    #[test]
    fn parses_mono_44100_q5_typical() {
        // VORBIS_HEADER_ID  vorbis_version=0 channels=1 sample_rate=44100
        //                   bitrate_max=0 bitrate_nominal=96000 bitrate_min=0
        //                   blocksize_0=256 blocksize_1=2048 framing_flag=1
        let packet = build_id_packet(0, 1, 44100, 0, 96000, 0, 8, 11, 1);
        let id = parse_identification_header(&packet).expect("must parse");
        assert_eq!(id.vorbis_version, 0);
        assert_eq!(id.audio_channels, 1);
        assert_eq!(id.audio_sample_rate, 44100);
        assert_eq!(id.bitrate_maximum, 0);
        assert_eq!(id.bitrate_nominal, 96000);
        assert_eq!(id.bitrate_minimum, 0);
        assert_eq!(id.blocksize_0, 256);
        assert_eq!(id.blocksize_1, 2048);
        assert_eq!(id.short_block_samples(), 256);
        assert_eq!(id.long_block_samples(), 2048);
    }

    /// 5.1-channel 48000 Hz exercises the upper edge of the
    /// channel-count axis and a different sample rate.
    #[test]
    fn parses_5_1_channel_48000_q5() {
        let packet = build_id_packet(0, 6, 48000, 0, 192000, 0, 8, 11, 1);
        let id = parse_identification_header(&packet).expect("must parse");
        assert_eq!(id.audio_channels, 6);
        assert_eq!(id.audio_sample_rate, 48000);
        assert_eq!(id.blocksize_0, 256);
        assert_eq!(id.blocksize_1, 2048);
    }

    /// Negative bitrate hints (signed encoding); some encoders use
    /// negative sentinels to flag "unset".
    #[test]
    fn parses_negative_bitrate_hints_as_signed() {
        let packet = build_id_packet(0, 2, 44100, -1, -1, -1, 8, 11, 1);
        let id = parse_identification_header(&packet).expect("must parse");
        assert_eq!(id.bitrate_maximum, -1);
        assert_eq!(id.bitrate_nominal, -1);
        assert_eq!(id.bitrate_minimum, -1);
    }

    /// Equal-blocksize stream (low-rate encoder where short == long).
    /// 64-sample blocks are the spec minimum (`blocksize_0` exponent 6).
    #[test]
    fn parses_equal_blocksizes_at_spec_minimum() {
        let packet = build_id_packet(0, 1, 22050, 0, 32000, 0, 6, 6, 1);
        let id = parse_identification_header(&packet).expect("must parse");
        assert_eq!(id.blocksize_0, 64);
        assert_eq!(id.blocksize_1, 64);
    }

    /// Maximum spec-legal blocksize exponent is 13 → 8192 samples.
    #[test]
    fn parses_maximum_blocksize_8192() {
        let packet = build_id_packet(0, 2, 96000, 0, 500000, 0, 6, 13, 1);
        let id = parse_identification_header(&packet).expect("must parse");
        assert_eq!(id.blocksize_0, 64);
        assert_eq!(id.blocksize_1, 8192);
    }

    #[test]
    fn rejects_short_packet() {
        let packet = [0u8; 10];
        match parse_identification_header(&packet) {
            Err(ParseError::PacketTooShort(10)) => {}
            other => panic!("expected PacketTooShort(10), got {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_packet_type() {
        let mut packet = build_id_packet(0, 1, 44100, 0, 96000, 0, 8, 11, 1);
        packet[0] = 0x03; // comment header byte
        match parse_identification_header(&packet) {
            Err(ParseError::WrongPacketType(0x03)) => {}
            other => panic!("expected WrongPacketType(0x03), got {other:?}"),
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut packet = build_id_packet(0, 1, 44100, 0, 96000, 0, 8, 11, 1);
        packet[1] = b'V'; // capitalise — spec mandates lowercase "vorbis"
        assert_eq!(
            parse_identification_header(&packet),
            Err(ParseError::BadMagic)
        );
    }

    #[test]
    fn rejects_nonzero_vorbis_version() {
        let packet = build_id_packet(1, 1, 44100, 0, 96000, 0, 8, 11, 1);
        match parse_identification_header(&packet) {
            Err(ParseError::UnsupportedVorbisVersion(1)) => {}
            other => panic!("expected UnsupportedVorbisVersion(1), got {other:?}"),
        }
    }

    #[test]
    fn rejects_zero_channels() {
        let packet = build_id_packet(0, 0, 44100, 0, 96000, 0, 8, 11, 1);
        assert_eq!(
            parse_identification_header(&packet),
            Err(ParseError::ZeroChannels)
        );
    }

    #[test]
    fn rejects_zero_sample_rate() {
        let packet = build_id_packet(0, 1, 0, 0, 96000, 0, 8, 11, 1);
        assert_eq!(
            parse_identification_header(&packet),
            Err(ParseError::ZeroSampleRate)
        );
    }

    #[test]
    fn rejects_illegal_blocksize_exponent_below_range() {
        // exponent 5 → 32-sample block, illegal (legal is 6..=13).
        let packet = build_id_packet(0, 1, 44100, 0, 96000, 0, 5, 11, 1);
        match parse_identification_header(&packet) {
            Err(ParseError::IllegalBlocksizeExponent(5, 11)) => {}
            other => panic!("expected IllegalBlocksizeExponent(5, 11), got {other:?}"),
        }
    }

    #[test]
    fn rejects_illegal_blocksize_exponent_above_range() {
        // exponent 14 → 16384-sample block, illegal.
        let packet = build_id_packet(0, 1, 44100, 0, 96000, 0, 8, 14, 1);
        match parse_identification_header(&packet) {
            Err(ParseError::IllegalBlocksizeExponent(8, 14)) => {}
            other => panic!("expected IllegalBlocksizeExponent(8, 14), got {other:?}"),
        }
    }

    #[test]
    fn rejects_blocksizes_out_of_order() {
        // blocksize_0 = 2048, blocksize_1 = 256 — both legal exponents
        // but ordered the wrong way around per §4.2.2.
        let packet = build_id_packet(0, 1, 44100, 0, 96000, 0, 11, 8, 1);
        match parse_identification_header(&packet) {
            Err(ParseError::BlocksizesOutOfOrder {
                blocksize_0: 2048,
                blocksize_1: 256,
            }) => {}
            other => panic!("expected BlocksizesOutOfOrder, got {other:?}"),
        }
    }

    #[test]
    fn rejects_zero_framing_flag() {
        let packet = build_id_packet(0, 1, 44100, 0, 96000, 0, 8, 11, 0);
        assert_eq!(
            parse_identification_header(&packet),
            Err(ParseError::BadFramingFlag)
        );
    }

    /// The framing byte's upper 7 bits are padding (§2.1.8); only
    /// bit 0 carries the framing flag. A decoder must accept any
    /// padding pattern as long as bit 0 is set.
    #[test]
    fn accepts_framing_byte_with_padding_set() {
        let mut packet = build_id_packet(0, 1, 44100, 0, 96000, 0, 8, 11, 1);
        // Set bit 0 to 1 (framing flag) and bits 1..=7 to nonzero
        // garbage. Per §2.1.8 the unused bits are *expected* to be
        // zero on encode, but a decoder validates only the framing
        // bit itself.
        packet[29] = 0xff;
        let id = parse_identification_header(&packet).expect("must parse despite padding");
        assert_eq!(id.audio_channels, 1);
    }
}

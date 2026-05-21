//! Vorbis I comment header parser (Vorbis I §5).
//!
//! The comment header is the second of three required Vorbis header
//! packets. It carries a *vendor string* (set by the encoder) and a list
//! of user comments in `KEY=value` form, where the key is case-insensitive
//! ASCII (Vorbis I §5.2.2) and the value is 8-bit-clean UTF-8.
//!
//! ## Packet layout
//!
//! Per the Vorbis I Specification (Xiph.Org, 2020-07-04 revision):
//!
//! * Common header (Vorbis I §4.2.1) — 7 bytes:
//!   - byte 0: `packet_type` (`0x03` for the comment header)
//!   - bytes 1..7: ASCII `"vorbis"` (`0x76 0x6f 0x72 0x62 0x69 0x73`)
//! * Comment payload (Vorbis I §5.2.1 / §5.2.3):
//!   - 32 bits `vendor_length` (LE u32)
//!   - `vendor_length` bytes UTF-8 vendor string (not NUL terminated)
//!   - 32 bits `user_comment_list_length` (LE u32)
//!   - then `user_comment_list_length` times:
//!     * 32 bits comment `length` (LE u32)
//!     * `length` bytes UTF-8 comment (not NUL terminated; conventionally
//!       `KEY=value`)
//!   - 1 bit `framing_bit` (must be 1)
//!
//! ### Byte alignment
//!
//! §5.2.1 explicitly notes that although the vector lengths and the
//! number of vectors are stored LSB-first per the §2.1.4 bitpacking
//! convention, "since data in the comment header is octet-aligned, they
//! can simply be read as unaligned 32 bit little endian unsigned
//! integers". This module therefore treats the body as byte-aligned
//! little-endian fields right up to the final framing-bit octet, in
//! which only bit 0 carries the framing flag (the remaining 7 bits are
//! §2.1.8 padding).
//!
//! ## End-of-packet handling
//!
//! Vorbis I §4.2 distinguishes end-of-packet during the comment header
//! ("non-fatal error condition") from end-of-packet during the
//! identification or setup headers ("renders the stream undecodable").
//! This parser returns a structured [`ParseError`] in all cases; the
//! caller may choose to treat truncation as a soft error per §4.2 (e.g.
//! a demuxer that wants to keep going) or as fatal (e.g. a strict
//! validator) by matching on [`ParseError::UnexpectedEndOfPacket`].
//!
//! ## What this module is, and is not
//!
//! This round-2 module ships the comment-header parse only. It does
//! **not** validate the `KEY=VALUE` syntax of individual comment
//! entries (§5.2.2 case-folding, allowed ASCII range, `=` separator) —
//! the comment vector format is byte-clean UTF-8 per spec, and any
//! syntactic validation is left to a higher-level consumer. It also
//! does not parse the setup header (§4.2.4) or decode any audio packet
//! (§4.3); see the crate root for round status.

use core::fmt;

/// Parsed Vorbis I comment header (§5).
///
/// The [`vendor`] field is the encoder's vendor identification string
/// (e.g. `"Xiph.Org libVorbis I 20020717"` or, for FFmpeg-encoded
/// streams, `"Lavf61.7.100"`). The [`comments`] vector contains the
/// raw comment entries as decoded UTF-8 strings; the spec encodes them
/// in `KEY=value` form (§5.2.2) but this parser does not split or
/// validate that shape — see [`split_key_value`] for an opt-in helper.
///
/// [`vendor`]: Self::vendor
/// [`comments`]: Self::comments
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VorbisCommentHeader {
    /// Vendor identification string (§5.2.1). UTF-8, not NUL terminated,
    /// possibly empty.
    pub vendor: String,
    /// User comments, in the order they appeared in the packet. Each
    /// entry is the raw decoded UTF-8 string (typically `KEY=value`).
    pub comments: Vec<String>,
}

impl VorbisCommentHeader {
    /// Common-header packet-type byte for the comment header
    /// (Vorbis I §4.2.1).
    pub const PACKET_TYPE: u8 = 0x03;

    /// The six magic bytes that follow the packet-type byte in every
    /// Vorbis header packet (Vorbis I §4.2.1).
    pub const MAGIC: [u8; 6] = *b"vorbis";

    /// Iterates the comments as `(key, value)` pairs by splitting each
    /// entry on the first `=` octet (Vorbis I §5.2.2). Entries that do
    /// not contain an `=` are skipped — the spec mandates the `=`
    /// separator as part of the comment vector format and any entry
    /// without one is malformed at the §5.2.2 layer.
    ///
    /// Per §5.2.2, field names are case-insensitive ASCII in the range
    /// `0x20..=0x7D` excluding `0x3D` (`=`); this helper does not
    /// case-fold or validate the key, leaving that to the caller.
    pub fn key_value_iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.comments
            .iter()
            .filter_map(|entry| split_key_value(entry))
    }
}

/// Splits one `KEY=value` comment entry into its key and value halves
/// on the first `=` octet (Vorbis I §5.2.2). Returns `None` if there is
/// no `=` — such an entry violates §5.2.2 ("immediately followed by
/// ASCII 0x3D").
#[must_use]
pub fn split_key_value(entry: &str) -> Option<(&str, &str)> {
    let pos = entry.as_bytes().iter().position(|&b| b == b'=')?;
    let key = &entry[..pos];
    let value = &entry[pos + 1..];
    Some((key, value))
}

/// Errors that may arise while parsing a Vorbis comment header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The packet was shorter than the 7-byte common header (a comment
    /// header must at minimum carry `packet_type` + `"vorbis"`).
    PacketTooShort(usize),
    /// The 1-byte common-header packet-type did not equal
    /// [`VorbisCommentHeader::PACKET_TYPE`] (`0x03`).
    WrongPacketType(u8),
    /// The six bytes following the packet type were not the ASCII
    /// magic string `"vorbis"`.
    BadMagic,
    /// The packet ended before all of the fields that §5.2.1 mandates
    /// could be read (vendor-length, vendor-string, comment-count,
    /// per-comment length, per-comment payload, framing bit). Vorbis I
    /// §4.2 marks end-of-packet during comment-header decode as a
    /// *non-fatal* error condition; callers may treat this variant as
    /// a soft error and continue parsing the stream.
    UnexpectedEndOfPacket,
    /// A length prefix in the packet exceeded `usize::MAX` on the
    /// running platform, so the implied range cannot be addressed.
    /// (On 32-bit targets a `u32` length is representable in `usize`
    /// only if it is ≤ `i32::MAX`; we conservatively reject anything
    /// that would not fit.)
    LengthOverflow(u32),
    /// The vendor string was not valid UTF-8 (§5.2.1 mandates UTF-8).
    /// The contained `usize` is the byte offset within the vendor
    /// string at which decoding failed.
    InvalidVendorUtf8(usize),
    /// A user comment was not valid UTF-8 (§5.2.2). The contained
    /// `(index, offset)` is the comment index and the byte offset
    /// within that comment at which decoding failed.
    InvalidCommentUtf8(u32, usize),
    /// The trailing framing bit was 0; §5.2.1 step 8 mandates "if
    /// `[framing_bit]` unset or end-of-packet then ERROR".
    BadFramingFlag,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::PacketTooShort(n) => write!(
                f,
                "vorbis comment header: packet too short ({n} bytes; need at least 7 for common header per §4.2.1)"
            ),
            ParseError::WrongPacketType(t) => write!(
                f,
                "vorbis comment header: wrong packet_type byte 0x{t:02x} (expected 0x03 per §4.2.1)"
            ),
            ParseError::BadMagic => write!(
                f,
                "vorbis comment header: missing 'vorbis' magic per §4.2.1"
            ),
            ParseError::UnexpectedEndOfPacket => write!(
                f,
                "vorbis comment header: end of packet during §5.2.1 decode (non-fatal per §4.2)"
            ),
            ParseError::LengthOverflow(n) => write!(
                f,
                "vorbis comment header: length prefix {n} exceeds platform usize"
            ),
            ParseError::InvalidVendorUtf8(off) => write!(
                f,
                "vorbis comment header: vendor string invalid UTF-8 at byte {off} (§5.2.1)"
            ),
            ParseError::InvalidCommentUtf8(idx, off) => write!(
                f,
                "vorbis comment header: comment #{idx} invalid UTF-8 at byte {off} (§5.2.2)"
            ),
            ParseError::BadFramingFlag => write!(
                f,
                "vorbis comment header: framing_bit=0 (§5.2.1 step 8)"
            ),
        }
    }
}

impl std::error::Error for ParseError {}

/// Byte-aligned little-endian cursor over the comment-header payload.
///
/// §5.2.1 explicitly authorises reading the length fields as "unaligned
/// 32 bit little endian unsigned integers", so a plain byte cursor is
/// the precise mechanical realisation of the spec's reader.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn read_u32_le(&mut self) -> Result<u32, ParseError> {
        if self.remaining() < 4 {
            return Err(ParseError::UnexpectedEndOfPacket);
        }
        let v = u32::from_le_bytes([
            self.bytes[self.pos],
            self.bytes[self.pos + 1],
            self.bytes[self.pos + 2],
            self.bytes[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], ParseError> {
        if self.remaining() < n {
            return Err(ParseError::UnexpectedEndOfPacket);
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, ParseError> {
        if self.remaining() < 1 {
            return Err(ParseError::UnexpectedEndOfPacket);
        }
        let b = self.bytes[self.pos];
        self.pos += 1;
        Ok(b)
    }
}

/// Converts a spec-level `u32` length into a `usize` index suitable for
/// slicing a Rust `&[u8]`, rejecting values that would not fit.
///
/// On 64-bit platforms this is always lossless. On 32-bit platforms a
/// `u32` length greater than `i32::MAX` (≈ 2 GiB) is conservatively
/// rejected as untrustworthy, since allocating a buffer of that size is
/// not generally feasible.
fn length_to_usize(n: u32) -> Result<usize, ParseError> {
    let as_usize = n as usize;
    if as_usize as u64 != u64::from(n) {
        return Err(ParseError::LengthOverflow(n));
    }
    Ok(as_usize)
}

/// Decodes a UTF-8 byte slice, returning the byte offset of the first
/// invalid sequence on failure.
fn decode_utf8(bytes: &[u8]) -> Result<String, usize> {
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|e| e.valid_up_to())
}

/// Parses a Vorbis I comment-header packet from `packet`.
///
/// The byte buffer must contain the entire packet, starting with the
/// `packet_type` byte (§4.2.1). The function validates the common
/// header, decodes the vendor string + comment list per §5.2.1, and
/// verifies the trailing framing bit. Returns the parsed
/// [`VorbisCommentHeader`].
///
/// Returns [`ParseError`] on any deviation from §5.2.1 or §4.2.1.
pub fn parse_comment_header(packet: &[u8]) -> Result<VorbisCommentHeader, ParseError> {
    if packet.len() < 7 {
        return Err(ParseError::PacketTooShort(packet.len()));
    }
    if packet[0] != VorbisCommentHeader::PACKET_TYPE {
        return Err(ParseError::WrongPacketType(packet[0]));
    }
    if packet[1..7] != VorbisCommentHeader::MAGIC {
        return Err(ParseError::BadMagic);
    }

    let mut cur = Cursor::new(&packet[7..]);

    // §5.2.1 step 1: vendor_length.
    let vendor_length = cur.read_u32_le()?;
    let vendor_len = length_to_usize(vendor_length)?;
    // §5.2.1 step 2: vendor_string.
    let vendor_bytes = cur.read_bytes(vendor_len)?;
    let vendor = decode_utf8(vendor_bytes).map_err(ParseError::InvalidVendorUtf8)?;

    // §5.2.1 step 3: user_comment_list_length.
    let comment_count = cur.read_u32_le()?;
    let mut comments: Vec<String> = Vec::with_capacity(comment_count.min(64) as usize);

    // §5.2.1 step 4..6: iterate comment_count times reading (u32 length,
    // UTF-8 payload).
    for idx in 0..comment_count {
        let length = cur.read_u32_le()?;
        let len = length_to_usize(length)?;
        let payload = cur.read_bytes(len)?;
        let entry = decode_utf8(payload).map_err(|off| ParseError::InvalidCommentUtf8(idx, off))?;
        comments.push(entry);
    }

    // §5.2.1 step 7..8: framing_bit (LSB of the next octet). The
    // remaining 7 bits are §2.1.8 padding; only bit 0 is the framing
    // flag and must be 1.
    let framing_byte = cur.read_u8()?;
    if framing_byte & 0x01 == 0 {
        return Err(ParseError::BadFramingFlag);
    }

    Ok(VorbisCommentHeader { vendor, comments })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic comment-header packet per §5.2.1 / §5.2.3.
    /// `framing` is the 1-bit framing flag in bit 0 of the trailing
    /// byte; the remaining 7 bits are written as zero padding per §2.1.8.
    fn build_comment_packet(vendor: &str, comments: &[&str], framing: u8) -> Vec<u8> {
        let mut p = Vec::new();
        p.push(VorbisCommentHeader::PACKET_TYPE);
        p.extend_from_slice(&VorbisCommentHeader::MAGIC);
        p.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        p.extend_from_slice(vendor.as_bytes());
        p.extend_from_slice(&(comments.len() as u32).to_le_bytes());
        for c in comments {
            p.extend_from_slice(&(c.len() as u32).to_le_bytes());
            p.extend_from_slice(c.as_bytes());
        }
        p.push(framing & 0x01);
        p
    }

    /// Canonical `mono-44100-q5-typical` fixture shape per
    /// `docs/audio/vorbis/vorbis-fixtures-and-traces.md` §2.2:
    /// one comment `encoder=Lavc61.19.101 libvorbis`, vendor
    /// `Lavf61.7.100`.
    #[test]
    fn parses_mono_44100_q5_typical_shape() {
        let packet = build_comment_packet("Lavf61.7.100", &["encoder=Lavc61.19.101 libvorbis"], 1);
        let header = parse_comment_header(&packet).expect("must parse");
        assert_eq!(header.vendor, "Lavf61.7.100");
        assert_eq!(header.comments.len(), 1);
        assert_eq!(header.comments[0], "encoder=Lavc61.19.101 libvorbis");
    }

    /// `with-vorbis-comment-tags` fixture shape: seven comments with
    /// TITLE/ARTIST/ALBUM/DATE/GENRE/TRACKNUMBER + encoder.
    #[test]
    fn parses_with_vorbis_comment_tags_shape() {
        let entries = [
            "encoder=Lavc61.19.101 libvorbis",
            "title=Test Title",
            "artist=Test Artist",
            "album=Test Album",
            "date=2026",
            "genre=Synth",
            "tracknumber=1",
        ];
        let packet = build_comment_packet("Lavf61.7.100", &entries, 1);
        let header = parse_comment_header(&packet).expect("must parse");
        assert_eq!(header.vendor, "Lavf61.7.100");
        assert_eq!(header.comments.len(), 7);
        for (got, want) in header.comments.iter().zip(entries.iter()) {
            assert_eq!(got, want);
        }
    }

    /// Spec example: historical libvorbis vendor string.
    #[test]
    fn parses_historical_libvorbis_vendor() {
        let packet = build_comment_packet("Xiph.Org libVorbis I 20020717", &[], 1);
        let header = parse_comment_header(&packet).expect("must parse");
        assert_eq!(header.vendor, "Xiph.Org libVorbis I 20020717");
        assert!(header.comments.is_empty());
    }

    /// Empty vendor + empty comment list (a "practically empty" comment
    /// header per §5.2.3, "must be present in the bitstream even if it
    /// is effectively empty").
    #[test]
    fn parses_empty_vendor_and_empty_comment_list() {
        let packet = build_comment_packet("", &[], 1);
        let header = parse_comment_header(&packet).expect("must parse");
        assert_eq!(header.vendor, "");
        assert!(header.comments.is_empty());
    }

    /// UTF-8 vendor string with multi-byte sequences.
    #[test]
    fn parses_utf8_multibyte_vendor() {
        let packet = build_comment_packet("Ω-encoder ✓ v1.0 — Karpelès", &[], 1);
        let header = parse_comment_header(&packet).expect("must parse");
        assert_eq!(header.vendor, "Ω-encoder ✓ v1.0 — Karpelès");
    }

    /// UTF-8 multi-byte comment value (UTF-8 is the §5.2.2 value
    /// encoding).
    #[test]
    fn parses_utf8_multibyte_comment_value() {
        let packet = build_comment_packet("vendor", &["TITLE=東京 / Tōkyō"], 1);
        let header = parse_comment_header(&packet).expect("must parse");
        assert_eq!(header.comments[0], "TITLE=東京 / Tōkyō");
    }

    /// §5.2.2 explicitly permits duplicate field names.
    #[test]
    fn parses_duplicate_field_names_per_spec() {
        let entries = [
            "ARTIST=Dizzy Gillespie",
            "ARTIST=Sonny Rollins",
            "ARTIST=Sonny Stitt",
        ];
        let packet = build_comment_packet("vendor", &entries, 1);
        let header = parse_comment_header(&packet).expect("must parse");
        assert_eq!(header.comments.len(), 3);
        let artists: Vec<_> = header
            .key_value_iter()
            .filter(|(k, _)| *k == "ARTIST")
            .map(|(_, v)| v)
            .collect();
        assert_eq!(
            artists,
            vec!["Dizzy Gillespie", "Sonny Rollins", "Sonny Stitt"]
        );
    }

    /// METADATA_BLOCK_PICTURE is conveyed as a (very long) plain
    /// comment per the trace-doc §2.3. The parser must accept large
    /// payloads.
    #[test]
    fn parses_large_payload_comment() {
        let large_value = "x".repeat(64 * 1024);
        let entry = format!("METADATA_BLOCK_PICTURE={large_value}");
        let packet = build_comment_packet("vendor", &[entry.as_str()], 1);
        let header = parse_comment_header(&packet).expect("must parse large payload");
        assert_eq!(header.comments[0].len(), entry.len());
        assert!(header.comments[0].starts_with("METADATA_BLOCK_PICTURE="));
    }

    /// Framing byte upper 7 bits are §2.1.8 padding; a decoder must
    /// accept any padding pattern as long as bit 0 is set.
    #[test]
    fn accepts_framing_byte_with_padding_set() {
        let mut packet = build_comment_packet("vendor", &["A=b"], 1);
        let last = packet.len() - 1;
        packet[last] = 0xff;
        let header = parse_comment_header(&packet).expect("must parse despite padding");
        assert_eq!(header.comments[0], "A=b");
    }

    /// `key_value_iter` skips entries with no `=`, per §5.2.2's
    /// requirement that the field name is "immediately followed by
    /// ASCII 0x3D".
    #[test]
    fn key_value_iter_skips_entries_without_equals() {
        let entries = ["VALID=ok", "no-equals-here", "TITLE=t"];
        let packet = build_comment_packet("vendor", &entries, 1);
        let header = parse_comment_header(&packet).expect("must parse");
        let kvs: Vec<_> = header.key_value_iter().collect();
        assert_eq!(kvs, vec![("VALID", "ok"), ("TITLE", "t")]);
    }

    /// `split_key_value` splits on the *first* `=`, since §5.2.2 says
    /// the `=` "terminate[s] the field name" — additional `=` bytes are
    /// part of the value.
    #[test]
    fn split_key_value_splits_on_first_equals_only() {
        assert_eq!(
            split_key_value("URL=https://x.test/?a=1"),
            Some(("URL", "https://x.test/?a=1"))
        );
        assert_eq!(split_key_value("BARE="), Some(("BARE", "")));
        assert_eq!(split_key_value("=NO_KEY"), Some(("", "NO_KEY")));
        assert_eq!(split_key_value("NO_EQUALS"), None);
    }

    #[test]
    fn rejects_short_packet() {
        let packet = [0u8; 5];
        assert_eq!(
            parse_comment_header(&packet),
            Err(ParseError::PacketTooShort(5))
        );
    }

    #[test]
    fn rejects_wrong_packet_type() {
        let mut packet = build_comment_packet("v", &[], 1);
        packet[0] = 0x01; // identification-header packet type
        assert_eq!(
            parse_comment_header(&packet),
            Err(ParseError::WrongPacketType(0x01))
        );
    }

    #[test]
    fn rejects_bad_magic() {
        let mut packet = build_comment_packet("v", &[], 1);
        packet[1] = b'V'; // capitalise the 'v'
        assert_eq!(parse_comment_header(&packet), Err(ParseError::BadMagic));
    }

    #[test]
    fn rejects_truncated_vendor_length() {
        // packet_type + magic + only 2 bytes where vendor_length needs 4
        let mut packet = Vec::new();
        packet.push(VorbisCommentHeader::PACKET_TYPE);
        packet.extend_from_slice(&VorbisCommentHeader::MAGIC);
        packet.extend_from_slice(&[0u8; 2]);
        assert_eq!(
            parse_comment_header(&packet),
            Err(ParseError::UnexpectedEndOfPacket)
        );
    }

    #[test]
    fn rejects_truncated_vendor_string() {
        // vendor_length declares 32 bytes but only 4 are present
        let mut packet = Vec::new();
        packet.push(VorbisCommentHeader::PACKET_TYPE);
        packet.extend_from_slice(&VorbisCommentHeader::MAGIC);
        packet.extend_from_slice(&32u32.to_le_bytes());
        packet.extend_from_slice(b"abcd");
        assert_eq!(
            parse_comment_header(&packet),
            Err(ParseError::UnexpectedEndOfPacket)
        );
    }

    #[test]
    fn rejects_truncated_comment_count() {
        // vendor decoded, but no bytes for comment count
        let mut packet = Vec::new();
        packet.push(VorbisCommentHeader::PACKET_TYPE);
        packet.extend_from_slice(&VorbisCommentHeader::MAGIC);
        packet.extend_from_slice(&0u32.to_le_bytes()); // zero-length vendor
        assert_eq!(
            parse_comment_header(&packet),
            Err(ParseError::UnexpectedEndOfPacket)
        );
    }

    #[test]
    fn rejects_truncated_comment_payload() {
        // declare 1 comment of length 10 but supply only 4 bytes
        let mut packet = Vec::new();
        packet.push(VorbisCommentHeader::PACKET_TYPE);
        packet.extend_from_slice(&VorbisCommentHeader::MAGIC);
        packet.extend_from_slice(&0u32.to_le_bytes()); // vendor_length=0
        packet.extend_from_slice(&1u32.to_le_bytes()); // comment_count=1
        packet.extend_from_slice(&10u32.to_le_bytes()); // length=10
        packet.extend_from_slice(b"ABCD");
        assert_eq!(
            parse_comment_header(&packet),
            Err(ParseError::UnexpectedEndOfPacket)
        );
    }

    #[test]
    fn rejects_missing_framing_byte() {
        // declare zero comments, no framing byte at all
        let mut packet = Vec::new();
        packet.push(VorbisCommentHeader::PACKET_TYPE);
        packet.extend_from_slice(&VorbisCommentHeader::MAGIC);
        packet.extend_from_slice(&0u32.to_le_bytes()); // vendor_length=0
        packet.extend_from_slice(&0u32.to_le_bytes()); // comment_count=0
        assert_eq!(
            parse_comment_header(&packet),
            Err(ParseError::UnexpectedEndOfPacket)
        );
    }

    #[test]
    fn rejects_zero_framing_flag() {
        let packet = build_comment_packet("vendor", &[], 0);
        assert_eq!(
            parse_comment_header(&packet),
            Err(ParseError::BadFramingFlag)
        );
    }

    #[test]
    fn rejects_invalid_vendor_utf8() {
        // 0xff is not a legal UTF-8 lead byte
        let mut packet = Vec::new();
        packet.push(VorbisCommentHeader::PACKET_TYPE);
        packet.extend_from_slice(&VorbisCommentHeader::MAGIC);
        packet.extend_from_slice(&3u32.to_le_bytes()); // vendor_length=3
        packet.extend_from_slice(&[0x66, 0x6f, 0xff]); // "fo\xff"
        packet.extend_from_slice(&0u32.to_le_bytes()); // comment_count=0
        packet.push(1u8); // framing
        match parse_comment_header(&packet) {
            Err(ParseError::InvalidVendorUtf8(2)) => {}
            other => panic!("expected InvalidVendorUtf8(2), got {other:?}"),
        }
    }

    #[test]
    fn rejects_invalid_comment_utf8() {
        let mut packet = Vec::new();
        packet.push(VorbisCommentHeader::PACKET_TYPE);
        packet.extend_from_slice(&VorbisCommentHeader::MAGIC);
        packet.extend_from_slice(&0u32.to_le_bytes()); // vendor_length=0
        packet.extend_from_slice(&1u32.to_le_bytes()); // comment_count=1
        packet.extend_from_slice(&4u32.to_le_bytes()); // length=4
        packet.extend_from_slice(&[b'A', b'=', b'B', 0xff]);
        packet.push(1u8);
        match parse_comment_header(&packet) {
            Err(ParseError::InvalidCommentUtf8(0, 3)) => {}
            other => panic!("expected InvalidCommentUtf8(0, 3), got {other:?}"),
        }
    }
}

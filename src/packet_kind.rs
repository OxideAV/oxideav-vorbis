//! Vorbis I packet-kind classifier (§4.2.1 / §4.3.1) and unified
//! header-packet dispatcher.
//!
//! # Scope
//!
//! Every Vorbis I packet in a stream falls into one of four kinds:
//!
//! | Kind             | Discriminator (byte 0)                | Magic (bytes 1..7) |
//! | ---------------- | ------------------------------------- | ------------------ |
//! | Identification   | `0x01`                                | `"vorbis"`         |
//! | Comment          | `0x03`                                | `"vorbis"`         |
//! | Setup            | `0x05`                                | `"vorbis"`         |
//! | Audio            | bit 0 of byte 0 is `0` (even byte 0)  | (no magic)         |
//!
//! The three header packets share the §4.2.1 "common header" prelude
//! (packet-type byte + six ASCII bytes `"vorbis"`), and the audio
//! packet's §4.3.1 step-1 `[packet_type]` bit must be `0` (the LSB of
//! byte 0 in LSB-first packing — see [`crate::packet`] for the
//! audio-packet header reader). The three header packet-type bytes
//! `0x01` / `0x03` / `0x05` all have bit 0 set (= odd byte 0), which
//! is exactly the §4.3.1 "non-audio" check; the parity test is
//! therefore the canonical first-byte split between header and audio
//! packets.
//!
//! This module exposes two complementary entry points:
//!
//! 1. [`classify_packet`] — a cheap byte-0 / magic inspection that
//!    classifies a packet's kind without parsing its body. Useful for a
//!    demuxer that wants to know how to route a packet to the
//!    appropriate per-stream parser.
//! 2. [`parse_header_packet`] — a unified header-packet dispatcher
//!    that classifies and then delegates to the matching
//!    [`crate::identification::parse_identification_header`] /
//!    [`crate::comment::parse_comment_header`] /
//!    [`crate::setup::parse_setup_header`] parser. Returns the parsed
//!    result in a [`HeaderPacket`] sum.
//!
//! Neither entry point reads beyond the §4.2.1 common header in order
//! to classify; the deeper parsing is delegated to the per-header
//! parsers and inherits their exact error surface.
//!
//! # The audio-packet check
//!
//! `classify_packet` reports [`PacketKind::Audio`] when byte 0's LSB is
//! `0`. This matches §4.3.1's step-1 `[packet_type]` test for any
//! conformant audio packet. A header packet (byte 0 in
//! `{0x01, 0x03, 0x05}`) always has byte 0 odd; the audio kind
//! therefore receives every other first byte, including the
//! reserved-for-future-use even values that are not currently
//! specified. Callers consuming a classified audio packet should
//! still delegate to [`crate::packet::read_packet_header`] to validate
//! the full §4.3.1 prelude.
//!
//! # Empty packets
//!
//! A length-zero packet has no byte-0 to inspect. The classifier
//! treats that as the dedicated [`ClassifyError::EmptyPacket`] case so
//! the caller can distinguish it from a malformed first byte.

use crate::comment::{parse_comment_header, VorbisCommentHeader};
use crate::identification::{parse_identification_header, VorbisIdentificationHeader};
use crate::setup::{parse_setup_header, VorbisSetupHeader, SETUP_PACKET_MAGIC, SETUP_PACKET_TYPE};

/// The four kinds of packet defined for a Vorbis I bitstream.
///
/// Returned by [`classify_packet`] from a cheap byte-0 / magic
/// inspection of the packet payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PacketKind {
    /// §4.2.2 identification header (packet-type `0x01`).
    Identification,
    /// §5.2.1 comment header (packet-type `0x03`).
    Comment,
    /// §4.2.4 setup header (packet-type `0x05`).
    Setup,
    /// §4.3 audio packet (byte 0's LSB is `0`).
    Audio,
}

impl PacketKind {
    /// `true` if this kind is one of the three header kinds (i.e. not
    /// [`PacketKind::Audio`]).
    #[must_use]
    pub fn is_header(self) -> bool {
        matches!(
            self,
            PacketKind::Identification | PacketKind::Comment | PacketKind::Setup
        )
    }

    /// `true` if this kind is [`PacketKind::Audio`].
    #[must_use]
    pub fn is_audio(self) -> bool {
        matches!(self, PacketKind::Audio)
    }

    /// The §4.2.1 packet-type byte associated with this kind, or
    /// `None` for [`PacketKind::Audio`] (which is not pinned to a
    /// single byte — every even byte 0 is an audio packet).
    #[must_use]
    pub fn packet_type_byte(self) -> Option<u8> {
        match self {
            PacketKind::Identification => Some(0x01),
            PacketKind::Comment => Some(0x03),
            PacketKind::Setup => Some(SETUP_PACKET_TYPE),
            PacketKind::Audio => None,
        }
    }
}

impl core::fmt::Display for PacketKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PacketKind::Identification => f.write_str("identification header"),
            PacketKind::Comment => f.write_str("comment header"),
            PacketKind::Setup => f.write_str("setup header"),
            PacketKind::Audio => f.write_str("audio packet"),
        }
    }
}

/// Errors [`classify_packet`] can surface from a byte-0 / magic
/// inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassifyError {
    /// The packet payload is empty so there is no byte 0 to inspect.
    EmptyPacket,
    /// Byte 0 is an odd value that is not one of the three header
    /// packet-type bytes (`0x01` / `0x03` / `0x05`). Vorbis I does not
    /// define a header packet with this byte; the bitstream is
    /// malformed.
    UnknownHeaderPacketType(u8),
    /// Byte 0 matches a header packet-type byte but the packet is too
    /// short to carry the §4.2.1 six-byte magic that follows it.
    HeaderTooShortForMagic {
        /// The header packet-type byte found at byte 0.
        packet_type: u8,
        /// The packet payload length.
        packet_len: usize,
    },
    /// Byte 0 matches a header packet-type byte but the §4.2.1
    /// six-byte magic at bytes 1..7 is not the ASCII string `"vorbis"`.
    BadHeaderMagic {
        /// The header packet-type byte found at byte 0.
        packet_type: u8,
        /// The actual six bytes at positions 1..7.
        actual_magic: [u8; 6],
    },
}

impl core::fmt::Display for ClassifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ClassifyError::EmptyPacket => {
                f.write_str("vorbis packet-kind: empty packet has no byte 0 to classify")
            }
            ClassifyError::UnknownHeaderPacketType(byte) => write!(
                f,
                "vorbis packet-kind: byte 0 = {byte:#04x} is odd but is not a defined \
                 Vorbis I header packet-type ({{0x01, 0x03, 0x05}})",
            ),
            ClassifyError::HeaderTooShortForMagic {
                packet_type,
                packet_len,
            } => write!(
                f,
                "vorbis packet-kind: header packet-type {packet_type:#04x} requires \
                 7 bytes for the §4.2.1 common header, got {packet_len}",
            ),
            ClassifyError::BadHeaderMagic {
                packet_type,
                actual_magic,
            } => write!(
                f,
                "vorbis packet-kind: header packet-type {packet_type:#04x} is followed by \
                 magic bytes {actual_magic:?}, expected the ASCII string \"vorbis\"",
            ),
        }
    }
}

impl std::error::Error for ClassifyError {}

/// Classify a Vorbis I packet by inspecting its §4.2.1 common-header
/// prelude (or, for audio packets, just byte 0's LSB per §4.3.1
/// step 1).
///
/// # Audio packets
///
/// If byte 0's LSB is `0`, the packet is an audio packet; the
/// classifier returns [`PacketKind::Audio`] without looking at the
/// rest of the buffer. (Audio packets carry no §4.2.1 magic — that
/// applies to the three header packets only.) Callers consuming a
/// classified audio packet should still drive
/// [`crate::packet::read_packet_header`] over the full §4.3.1 prelude
/// to validate every field.
///
/// # Header packets
///
/// If byte 0's LSB is `1` (odd byte), the classifier requires:
///
/// * Byte 0 ∈ `{0x01, 0x03, 0x05}` — otherwise
///   [`ClassifyError::UnknownHeaderPacketType`].
/// * The packet payload is at least seven bytes — otherwise
///   [`ClassifyError::HeaderTooShortForMagic`].
/// * Bytes 1..7 equal the ASCII string `"vorbis"` — otherwise
///   [`ClassifyError::BadHeaderMagic`].
///
/// All three checks taken together pin the §4.2.1 common header.
///
/// # Errors
///
/// * [`ClassifyError::EmptyPacket`] if `packet.len() == 0`.
/// * The header-specific errors above when byte 0 is an odd value.
pub fn classify_packet(packet: &[u8]) -> Result<PacketKind, ClassifyError> {
    let first = *packet.first().ok_or(ClassifyError::EmptyPacket)?;

    // §4.3.1 step-1 `[packet_type]` test: bit 0 == 0 is an audio packet.
    if first & 0x01 == 0 {
        return Ok(PacketKind::Audio);
    }

    let kind = match first {
        0x01 => PacketKind::Identification,
        0x03 => PacketKind::Comment,
        SETUP_PACKET_TYPE => PacketKind::Setup,
        other => return Err(ClassifyError::UnknownHeaderPacketType(other)),
    };

    if packet.len() < 7 {
        return Err(ClassifyError::HeaderTooShortForMagic {
            packet_type: first,
            packet_len: packet.len(),
        });
    }
    let magic: [u8; 6] = packet[1..7].try_into().expect("len ≥ 7 checked above");
    if magic != SETUP_PACKET_MAGIC {
        return Err(ClassifyError::BadHeaderMagic {
            packet_type: first,
            actual_magic: magic,
        });
    }

    Ok(kind)
}

/// The parsed result of one of the three Vorbis I header packets,
/// returned by [`parse_header_packet`].
///
/// Audio packets are not represented in this enum; they are driven by
/// the per-packet [`crate::audio::decode_one_packet`] /
/// [`crate::streaming::StreamingDecoder::push_packet`] pipeline and
/// require the per-stream setup context to interpret.
///
/// `Eq` is not derived because [`VorbisSetupHeader`] (carried in the
/// `Setup` variant) holds `f32` fields and therefore implements only
/// `PartialEq` — matching the existing per-header parser conventions.
#[derive(Debug, Clone, PartialEq)]
pub enum HeaderPacket {
    /// A parsed §4.2.2 identification header.
    Identification(VorbisIdentificationHeader),
    /// A parsed §5.2.1 comment header.
    Comment(VorbisCommentHeader),
    /// A parsed §4.2.4 setup header.
    Setup(VorbisSetupHeader),
}

impl HeaderPacket {
    /// Discriminator [`PacketKind`] of this parsed header.
    #[must_use]
    pub fn kind(&self) -> PacketKind {
        match self {
            HeaderPacket::Identification(_) => PacketKind::Identification,
            HeaderPacket::Comment(_) => PacketKind::Comment,
            HeaderPacket::Setup(_) => PacketKind::Setup,
        }
    }

    /// Borrow the parsed identification header, if this is the
    /// [`HeaderPacket::Identification`] variant.
    #[must_use]
    pub fn identification(&self) -> Option<&VorbisIdentificationHeader> {
        match self {
            HeaderPacket::Identification(h) => Some(h),
            _ => None,
        }
    }

    /// Borrow the parsed comment header, if this is the
    /// [`HeaderPacket::Comment`] variant.
    #[must_use]
    pub fn comment(&self) -> Option<&VorbisCommentHeader> {
        match self {
            HeaderPacket::Comment(h) => Some(h),
            _ => None,
        }
    }

    /// Borrow the parsed setup header, if this is the
    /// [`HeaderPacket::Setup`] variant.
    #[must_use]
    pub fn setup(&self) -> Option<&VorbisSetupHeader> {
        match self {
            HeaderPacket::Setup(h) => Some(h),
            _ => None,
        }
    }
}

/// Errors [`parse_header_packet`] can surface.
///
/// The three header parser sub-errors are wrapped verbatim so the
/// caller has access to the full diagnostic surface of the underlying
/// parser. Classification failures are surfaced via
/// [`HeaderDispatchError::Classify`]. Receiving an audio packet as
/// input is reported via [`HeaderDispatchError::ExpectedHeaderGotAudio`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderDispatchError {
    /// The packet failed §4.2.1 / §4.3.1 classification.
    Classify(ClassifyError),
    /// The packet classifies as an audio packet
    /// ([`PacketKind::Audio`]). The header dispatcher is for header
    /// packets only; the caller must route audio packets to the
    /// per-packet driver instead.
    ExpectedHeaderGotAudio,
    /// The packet classified as the identification header but the
    /// [`parse_identification_header`] sub-parser rejected it.
    Identification(crate::identification::ParseError),
    /// The packet classified as the comment header but the
    /// [`parse_comment_header`] sub-parser rejected it.
    Comment(crate::comment::ParseError),
    /// The packet classified as the setup header but the
    /// [`parse_setup_header`] sub-parser rejected it.
    Setup(crate::setup::ParseError),
}

impl core::fmt::Display for HeaderDispatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HeaderDispatchError::Classify(e) => write!(f, "{e}"),
            HeaderDispatchError::ExpectedHeaderGotAudio => f.write_str(
                "vorbis header dispatch: expected one of the three header packets, got an \
                 audio packet (byte 0's LSB is 0)",
            ),
            HeaderDispatchError::Identification(e) => write!(f, "{e}"),
            HeaderDispatchError::Comment(e) => write!(f, "{e}"),
            HeaderDispatchError::Setup(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for HeaderDispatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HeaderDispatchError::Classify(e) => Some(e),
            HeaderDispatchError::ExpectedHeaderGotAudio => None,
            HeaderDispatchError::Identification(e) => Some(e),
            HeaderDispatchError::Comment(e) => Some(e),
            HeaderDispatchError::Setup(e) => Some(e),
        }
    }
}

impl From<ClassifyError> for HeaderDispatchError {
    fn from(value: ClassifyError) -> Self {
        HeaderDispatchError::Classify(value)
    }
}

impl From<crate::identification::ParseError> for HeaderDispatchError {
    fn from(value: crate::identification::ParseError) -> Self {
        HeaderDispatchError::Identification(value)
    }
}

impl From<crate::comment::ParseError> for HeaderDispatchError {
    fn from(value: crate::comment::ParseError) -> Self {
        HeaderDispatchError::Comment(value)
    }
}

impl From<crate::setup::ParseError> for HeaderDispatchError {
    fn from(value: crate::setup::ParseError) -> Self {
        HeaderDispatchError::Setup(value)
    }
}

/// Classify `packet` then delegate to the matching header parser,
/// returning the parsed [`HeaderPacket`].
///
/// `audio_channels` is required because the §4.2.4 setup-header
/// parser depends on the identification-header channel count to size
/// per-channel structures. Callers parsing the three Vorbis header
/// packets in order should pass the value from the
/// already-classified identification header.
///
/// # Errors
///
/// * [`HeaderDispatchError::Classify`] if [`classify_packet`] rejects
///   the byte-0 / magic prelude.
/// * [`HeaderDispatchError::ExpectedHeaderGotAudio`] if `packet`
///   classifies as an audio packet — the dispatcher is for header
///   packets only.
/// * [`HeaderDispatchError::Identification`] /
///   [`HeaderDispatchError::Comment`] / [`HeaderDispatchError::Setup`]
///   for body-parse failures on the corresponding sub-parser.
pub fn parse_header_packet(
    packet: &[u8],
    audio_channels: u8,
) -> Result<HeaderPacket, HeaderDispatchError> {
    match classify_packet(packet)? {
        PacketKind::Audio => Err(HeaderDispatchError::ExpectedHeaderGotAudio),
        PacketKind::Identification => {
            let h = parse_identification_header(packet)?;
            Ok(HeaderPacket::Identification(h))
        }
        PacketKind::Comment => {
            let h = parse_comment_header(packet)?;
            Ok(HeaderPacket::Comment(h))
        }
        PacketKind::Setup => {
            let h = parse_setup_header(packet, audio_channels)?;
            Ok(HeaderPacket::Setup(h))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identification::VorbisIdentificationHeader;

    // ----- classify_packet: audio path -----

    #[test]
    fn empty_packet_rejected() {
        assert_eq!(classify_packet(&[]), Err(ClassifyError::EmptyPacket));
    }

    #[test]
    fn audio_packet_byte_zero_lsb_clear() {
        // Every even first byte is an audio packet — bit 0 is the
        // §4.3.1 step-1 `[packet_type]` and must be 0 for audio.
        for first in [0x00u8, 0x02, 0x04, 0x06, 0xfe] {
            assert_eq!(
                classify_packet(&[first, 0xab, 0xcd, 0xef]),
                Ok(PacketKind::Audio),
                "byte 0 = {first:#04x} should classify as audio",
            );
        }
    }

    #[test]
    fn single_byte_audio_packet_accepted() {
        // The classifier only needs byte 0 for audio; a single-byte
        // (degenerate) packet still resolves.
        assert_eq!(classify_packet(&[0x00]), Ok(PacketKind::Audio));
    }

    // ----- classify_packet: header path -----

    fn make_header(packet_type: u8) -> Vec<u8> {
        let mut p = Vec::with_capacity(7);
        p.push(packet_type);
        p.extend_from_slice(b"vorbis");
        p
    }

    #[test]
    fn identification_header_classified() {
        let p = make_header(0x01);
        assert_eq!(classify_packet(&p), Ok(PacketKind::Identification));
    }

    #[test]
    fn comment_header_classified() {
        let p = make_header(0x03);
        assert_eq!(classify_packet(&p), Ok(PacketKind::Comment));
    }

    #[test]
    fn setup_header_classified() {
        let p = make_header(0x05);
        assert_eq!(classify_packet(&p), Ok(PacketKind::Setup));
    }

    #[test]
    fn unknown_odd_packet_type_rejected() {
        // 0x07, 0x09, 0xff — every odd byte not in {0x01, 0x03, 0x05}.
        for first in [0x07u8, 0x09, 0x0b, 0x7f, 0xff] {
            let p = {
                let mut v = vec![first];
                v.extend_from_slice(b"vorbis");
                v
            };
            assert_eq!(
                classify_packet(&p),
                Err(ClassifyError::UnknownHeaderPacketType(first)),
                "byte 0 = {first:#04x} should fail classification",
            );
        }
    }

    #[test]
    fn header_too_short_for_magic() {
        // 0x01 / 0x03 / 0x05 with only 1..7 bytes should fail on the
        // §4.2.1 magic-length check.
        for packet_type in [0x01u8, 0x03, 0x05] {
            for len in 1..7usize {
                let p: Vec<u8> = core::iter::once(packet_type)
                    .chain(std::iter::repeat(0u8).take(len - 1))
                    .collect();
                assert_eq!(
                    classify_packet(&p),
                    Err(ClassifyError::HeaderTooShortForMagic {
                        packet_type,
                        packet_len: len,
                    }),
                    "byte 0 = {packet_type:#04x}, len = {len} should fail magic-length check",
                );
            }
        }
    }

    #[test]
    fn header_bad_magic_rejected() {
        // Right packet-type byte, wrong six-byte magic.
        let mut p = vec![0x01];
        p.extend_from_slice(b"VORBIS"); // uppercase — invalid
        assert_eq!(
            classify_packet(&p),
            Err(ClassifyError::BadHeaderMagic {
                packet_type: 0x01,
                actual_magic: *b"VORBIS",
            }),
        );
    }

    // ----- PacketKind helpers -----

    #[test]
    fn packet_kind_is_header_classification() {
        assert!(PacketKind::Identification.is_header());
        assert!(PacketKind::Comment.is_header());
        assert!(PacketKind::Setup.is_header());
        assert!(!PacketKind::Audio.is_header());
    }

    #[test]
    fn packet_kind_is_audio_classification() {
        assert!(!PacketKind::Identification.is_audio());
        assert!(!PacketKind::Comment.is_audio());
        assert!(!PacketKind::Setup.is_audio());
        assert!(PacketKind::Audio.is_audio());
    }

    #[test]
    fn packet_kind_byte_lookup() {
        assert_eq!(PacketKind::Identification.packet_type_byte(), Some(0x01));
        assert_eq!(PacketKind::Comment.packet_type_byte(), Some(0x03));
        assert_eq!(PacketKind::Setup.packet_type_byte(), Some(0x05));
        assert_eq!(PacketKind::Audio.packet_type_byte(), None);
    }

    #[test]
    fn packet_kind_display_strings_pinned() {
        assert_eq!(
            format!("{}", PacketKind::Identification),
            "identification header"
        );
        assert_eq!(format!("{}", PacketKind::Comment), "comment header");
        assert_eq!(format!("{}", PacketKind::Setup), "setup header");
        assert_eq!(format!("{}", PacketKind::Audio), "audio packet");
    }

    // ----- parse_header_packet: dispatch -----

    /// Build the canonical 30-byte identification packet covered by
    /// the parser's own happy-path test. Mirrors the layout in
    /// `identification.rs` tests.
    fn build_identification_packet() -> Vec<u8> {
        let mut p = Vec::with_capacity(VorbisIdentificationHeader::PACKET_LEN);
        p.push(VorbisIdentificationHeader::PACKET_TYPE);
        p.extend_from_slice(&VorbisIdentificationHeader::MAGIC);
        p.extend_from_slice(&0u32.to_le_bytes()); // vorbis_version
        p.push(2); // audio_channels
        p.extend_from_slice(&44100u32.to_le_bytes()); // audio_sample_rate
        p.extend_from_slice(&0i32.to_le_bytes()); // bitrate_max
        p.extend_from_slice(&128000i32.to_le_bytes()); // bitrate_nominal
        p.extend_from_slice(&0i32.to_le_bytes()); // bitrate_min

        // blocksize byte: bs0 = 6 → 64, bs1 = 11 → 2048
        p.push((11 << 4) | 6);

        // framing flag
        p.push(0x01);
        debug_assert_eq!(p.len(), VorbisIdentificationHeader::PACKET_LEN);
        p
    }

    fn build_comment_packet() -> Vec<u8> {
        let mut p = vec![0x03];
        p.extend_from_slice(b"vorbis");
        let vendor = b"oxideav".as_slice();
        p.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        p.extend_from_slice(vendor);
        p.extend_from_slice(&0u32.to_le_bytes()); // comment_count = 0
        p.push(0x01); // framing flag
        p
    }

    #[test]
    fn dispatch_identification_happy_path() {
        let p = build_identification_packet();
        let parsed = parse_header_packet(&p, 0).expect("identification parse");
        assert_eq!(parsed.kind(), PacketKind::Identification);
        let id = parsed.identification().expect("identification variant");
        assert_eq!(id.audio_channels, 2);
        assert_eq!(id.audio_sample_rate, 44100);
        assert_eq!(id.blocksize_0, 64);
        assert_eq!(id.blocksize_1, 2048);
        assert!(parsed.comment().is_none());
        assert!(parsed.setup().is_none());
    }

    #[test]
    fn dispatch_comment_happy_path() {
        let p = build_comment_packet();
        let parsed = parse_header_packet(&p, 2).expect("comment parse");
        assert_eq!(parsed.kind(), PacketKind::Comment);
        let c = parsed.comment().expect("comment variant");
        assert_eq!(c.vendor, "oxideav");
        assert!(c.comments.is_empty());
        assert!(parsed.identification().is_none());
        assert!(parsed.setup().is_none());
    }

    #[test]
    fn dispatch_audio_packet_rejected() {
        let p = [0x00u8, 0x00, 0x00, 0x00];
        match parse_header_packet(&p, 2) {
            Err(HeaderDispatchError::ExpectedHeaderGotAudio) => {}
            other => panic!("expected ExpectedHeaderGotAudio, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_empty_packet_classify_error_surfaced() {
        match parse_header_packet(&[], 1) {
            Err(HeaderDispatchError::Classify(ClassifyError::EmptyPacket)) => {}
            other => panic!("expected Classify(EmptyPacket), got {other:?}"),
        }
    }

    #[test]
    fn dispatch_bad_magic_classify_error_surfaced() {
        let mut p = vec![0x01];
        p.extend_from_slice(b"VORBIS");
        match parse_header_packet(&p, 1) {
            Err(HeaderDispatchError::Classify(ClassifyError::BadHeaderMagic {
                packet_type: 0x01,
                actual_magic,
            })) if actual_magic == *b"VORBIS" => {}
            other => panic!("expected Classify(BadHeaderMagic), got {other:?}"),
        }
    }

    #[test]
    fn dispatch_identification_body_error_surfaced() {
        // Valid common header, then a body with `vorbis_version` = 1
        // (the parser only accepts 0).
        let mut p = build_identification_packet();
        // bytes 7..11 hold vorbis_version little-endian; set to 1.
        p[7] = 1;
        match parse_header_packet(&p, 0) {
            Err(HeaderDispatchError::Identification(
                crate::identification::ParseError::UnsupportedVorbisVersion(1),
            )) => {}
            other => panic!("expected Identification(UnsupportedVorbisVersion), got {other:?}"),
        }
    }

    #[test]
    fn dispatch_comment_body_error_surfaced() {
        // Comment packet with a vendor length that overruns the packet.
        let mut p = vec![0x03];
        p.extend_from_slice(b"vorbis");
        p.extend_from_slice(&100u32.to_le_bytes()); // vendor_length = 100
        p.extend_from_slice(b"too short"); // < 100 bytes
        match parse_header_packet(&p, 2) {
            Err(HeaderDispatchError::Comment(_)) => {}
            other => panic!("expected Comment(_), got {other:?}"),
        }
    }

    // ----- From impls -----

    #[test]
    fn from_classify_error_lifts_to_dispatch_error() {
        let e: HeaderDispatchError = ClassifyError::EmptyPacket.into();
        assert_eq!(e, HeaderDispatchError::Classify(ClassifyError::EmptyPacket));
    }

    // ----- Display -----

    #[test]
    fn classify_error_display_strings_pinned() {
        let s_empty = format!("{}", ClassifyError::EmptyPacket);
        assert!(s_empty.contains("empty packet"), "{s_empty}");

        let s_unknown = format!("{}", ClassifyError::UnknownHeaderPacketType(0x07));
        assert!(s_unknown.contains("0x07"), "{s_unknown}");

        let s_short = format!(
            "{}",
            ClassifyError::HeaderTooShortForMagic {
                packet_type: 0x01,
                packet_len: 3,
            }
        );
        assert!(s_short.contains("0x01"), "{s_short}");
        assert!(s_short.contains("got 3"), "{s_short}");

        let s_magic = format!(
            "{}",
            ClassifyError::BadHeaderMagic {
                packet_type: 0x03,
                actual_magic: *b"BANANA",
            }
        );
        assert!(s_magic.contains("\"vorbis\""), "{s_magic}");
    }

    #[test]
    fn dispatch_error_display_for_audio_variant() {
        let s = format!("{}", HeaderDispatchError::ExpectedHeaderGotAudio);
        assert!(s.contains("audio packet"), "{s}");
    }

    // ----- source() chain -----

    #[test]
    fn dispatch_error_source_chains_to_inner() {
        use std::error::Error as _;
        let e = HeaderDispatchError::Classify(ClassifyError::EmptyPacket);
        assert!(e.source().is_some());

        let e2 = HeaderDispatchError::ExpectedHeaderGotAudio;
        assert!(e2.source().is_none());
    }
}

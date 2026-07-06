//! RFC 3533 Ogg page framing — the transport layer under the Vorbis I
//! §A ("Embedding Vorbis into an Ogg stream") encapsulation.
//!
//! This module implements the *generic* Ogg page format of RFC 3533 §6
//! in both directions:
//!
//! * **Read side** — [`OggPage::parse`] / [`parse_pages`] walk a
//!   physical bitstream page by page, verifying the capture pattern,
//!   the stream-structure version, and the page CRC;
//!   [`PacketAssembler`] coalesces one logical bitstream's lacing
//!   segments back into the codec packets the pages carry (a lacing
//!   value `< 255` ends a packet; `255` continues it, possibly across
//!   a page boundary via the `continued` header flag).
//! * **Write side** — [`OggPage::serialize`] emits one page with its
//!   CRC computed per RFC 3533 §6 item 7, and [`PageWriter`] runs the
//!   packet→segment→page encapsulation: packets are chopped into
//!   255-byte lacing segments, pages auto-emit when the 255-segment
//!   table fills mid-packet (setting the next page's `continued`
//!   flag), and the page-level granule position records the position
//!   of the **last packet completed on the page** (`-1` when no packet
//!   completes, per RFC 3533 §6 item 4).
//!
//! The Vorbis-specific mapping rules (§A.2: header-page layout,
//! granule-position semantics in PCM samples) live one layer up in the
//! callers; this module is codec-agnostic transport.
//!
//! # CRC
//!
//! RFC 3533 §6 item 7: a 32-bit CRC over the whole page (header with a
//! zeroed CRC field, then the page body), generator polynomial
//! `0x04c11db7`. The register is fed MSB-first with no bit reflection,
//! zero initial value and no final XOR — pinned against the staged
//! real-world fixture streams in `tests/ogg_framing.rs`, every page of
//! which must re-serialize byte-exactly (CRC included).

/// The RFC 3533 §6 item 1 capture pattern that begins every page.
pub const OGG_CAPTURE_PATTERN: [u8; 4] = *b"OggS";

/// Maximum number of lacing segments per page (RFC 3533 §6 item 8: the
/// segment count is one byte).
pub const MAX_PAGE_SEGMENTS: usize = 255;

/// The fixed page-header length before the segment table
/// (RFC 3533 §6: `header_size = number_page_segments + 27`).
pub const PAGE_HEADER_LEN: usize = 27;

/// Errors raised by the Ogg framing layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OggError {
    /// The capture pattern `OggS` was not found at the page start.
    BadCapturePattern {
        /// Byte offset of the failed match.
        offset: usize,
    },
    /// The stream-structure version byte was not 0 (RFC 3533 §6
    /// item 2 — this module implements version 0 only).
    UnsupportedVersion {
        /// The version byte encountered.
        version: u8,
    },
    /// The buffer ended before the page's declared extent.
    TruncatedPage {
        /// Byte offset of the page start.
        offset: usize,
        /// Bytes needed to complete the page.
        needed: usize,
        /// Bytes actually available from the page start.
        available: usize,
    },
    /// The page CRC did not verify (RFC 3533 §6 item 7).
    CrcMismatch {
        /// Byte offset of the page start.
        offset: usize,
        /// CRC stored in the page header.
        stored: u32,
        /// CRC computed over the page with a zeroed CRC field.
        computed: u32,
    },
    /// A page under construction exceeded the 255-segment table limit.
    /// Defensive — [`PageWriter`] auto-emits before this can happen;
    /// hand-built [`OggPage`] values can trip it in `serialize`.
    TooManySegments {
        /// The offending segment count.
        segments: usize,
    },
    /// An [`OggPage`]'s body length disagreed with the sum of its
    /// lacing values.
    BodyLengthMismatch {
        /// Sum of the lacing values.
        expected: usize,
        /// Actual body length.
        actual: usize,
    },
    /// [`PacketAssembler::push_page`] received a page whose `continued`
    /// flag contradicts the assembler's mid-packet state: a fresh page
    /// arrived while a packet was still open, or a continuation page
    /// arrived with no packet open.
    ContinuityBroken {
        /// The page's sequence number.
        sequence: u32,
        /// `true` if the assembler held an unfinished packet.
        mid_packet: bool,
    },
    /// [`PacketAssembler::push_page`] received a page from a different
    /// logical bitstream than the one it locked onto.
    SerialMismatch {
        /// Serial the assembler locked onto.
        expected: u32,
        /// Serial of the offending page.
        actual: u32,
    },
}

impl core::fmt::Display for OggError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OggError::BadCapturePattern { offset } => {
                write!(f, "ogg: no capture pattern at byte {offset}")
            }
            OggError::UnsupportedVersion { version } => {
                write!(f, "ogg: unsupported stream-structure version {version}")
            }
            OggError::TruncatedPage {
                offset,
                needed,
                available,
            } => write!(
                f,
                "ogg: truncated page at byte {offset}: need {needed} bytes, have {available}"
            ),
            OggError::CrcMismatch {
                offset,
                stored,
                computed,
            } => write!(
                f,
                "ogg: CRC mismatch at byte {offset}: stored {stored:#010x}, computed {computed:#010x}"
            ),
            OggError::TooManySegments { segments } => {
                write!(f, "ogg: {segments} lacing segments exceed the 255 limit")
            }
            OggError::BodyLengthMismatch { expected, actual } => write!(
                f,
                "ogg: body length {actual} != lacing sum {expected}"
            ),
            OggError::ContinuityBroken {
                sequence,
                mid_packet,
            } => write!(
                f,
                "ogg: continuity broken at page sequence {sequence} (mid_packet = {mid_packet})"
            ),
            OggError::SerialMismatch { expected, actual } => write!(
                f,
                "ogg: page serial {actual:#010x} != locked stream serial {expected:#010x}"
            ),
        }
    }
}

impl std::error::Error for OggError {}

/// The RFC 3533 §6 item 7 page CRC table: generator polynomial
/// `0x04c11db7`, MSB-first feed, no reflection.
const fn build_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut r = (i as u32) << 24;
        let mut j = 0;
        while j < 8 {
            r = if r & 0x8000_0000 != 0 {
                (r << 1) ^ 0x04c1_1db7
            } else {
                r << 1
            };
            j += 1;
        }
        table[i] = r;
        i += 1;
    }
    table
}

static CRC_TABLE: [u32; 256] = build_crc_table();

/// The RFC 3533 §6 item 7 page checksum: 32-bit CRC, generator
/// polynomial `0x04c11db7`, MSB-first, zero initial register, no final
/// XOR, computed over the whole page with the CRC field itself zeroed.
#[must_use]
pub fn ogg_crc32(data: &[u8]) -> u32 {
    let mut crc = 0u32;
    for &b in data {
        crc = (crc << 8) ^ CRC_TABLE[(((crc >> 24) as u8) ^ b) as usize];
    }
    crc
}

/// One parsed / to-be-written Ogg page (RFC 3533 §6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OggPage {
    /// Header-type bit `0x01`: the page starts with the continuation
    /// of a packet begun on the previous page.
    pub continued: bool,
    /// Header-type bit `0x02`: first page of the logical bitstream.
    pub bos: bool,
    /// Header-type bit `0x04`: last page of the logical bitstream.
    pub eos: bool,
    /// Codec-defined position stamp of the last packet completed on
    /// this page; `-1` when no packet completes (RFC 3533 §6 item 4).
    pub granule_position: i64,
    /// Logical-bitstream serial number (RFC 3533 §6 item 5).
    pub serial: u32,
    /// Page sequence number within the logical bitstream
    /// (RFC 3533 §6 item 6).
    pub sequence: u32,
    /// The lacing values (RFC 3533 §6 item 9), one byte each.
    pub lacing: Vec<u8>,
    /// The page body: the segments' bytes, concatenated. Its length
    /// must equal the sum of the lacing values.
    pub body: Vec<u8>,
}

impl OggPage {
    /// Total on-wire length of this page:
    /// `27 + lacing.len() + body.len()` (RFC 3533 §6).
    #[must_use]
    pub fn page_len(&self) -> usize {
        PAGE_HEADER_LEN + self.lacing.len() + self.body.len()
    }

    /// Serialize the page, computing the RFC 3533 §6 item 7 CRC.
    ///
    /// # Errors
    ///
    /// [`OggError::TooManySegments`] for a lacing table over 255
    /// entries; [`OggError::BodyLengthMismatch`] when the body length
    /// disagrees with the lacing sum.
    pub fn serialize(&self) -> Result<Vec<u8>, OggError> {
        if self.lacing.len() > MAX_PAGE_SEGMENTS {
            return Err(OggError::TooManySegments {
                segments: self.lacing.len(),
            });
        }
        let expected: usize = self.lacing.iter().map(|&l| l as usize).sum();
        if expected != self.body.len() {
            return Err(OggError::BodyLengthMismatch {
                expected,
                actual: self.body.len(),
            });
        }
        let mut out = Vec::with_capacity(self.page_len());
        out.extend_from_slice(&OGG_CAPTURE_PATTERN);
        out.push(0); // stream_structure_version (RFC 3533 §6 item 2)
        let mut header_type = 0u8;
        if self.continued {
            header_type |= 0x01;
        }
        if self.bos {
            header_type |= 0x02;
        }
        if self.eos {
            header_type |= 0x04;
        }
        out.push(header_type);
        out.extend_from_slice(&self.granule_position.to_le_bytes());
        out.extend_from_slice(&self.serial.to_le_bytes());
        out.extend_from_slice(&self.sequence.to_le_bytes());
        out.extend_from_slice(&[0u8; 4]); // CRC placeholder
        out.push(self.lacing.len() as u8);
        out.extend_from_slice(&self.lacing);
        out.extend_from_slice(&self.body);
        let crc = ogg_crc32(&out);
        out[22..26].copy_from_slice(&crc.to_le_bytes());
        Ok(out)
    }

    /// Parse one page starting at `offset` in `data`, verifying the
    /// capture pattern, the version byte and the CRC. Returns the page
    /// and the offset of the next byte after it.
    ///
    /// # Errors
    ///
    /// [`OggError::BadCapturePattern`], [`OggError::UnsupportedVersion`],
    /// [`OggError::TruncatedPage`], or [`OggError::CrcMismatch`].
    pub fn parse(data: &[u8], offset: usize) -> Result<(OggPage, usize), OggError> {
        let avail = data.len().saturating_sub(offset);
        if avail < PAGE_HEADER_LEN {
            return Err(OggError::TruncatedPage {
                offset,
                needed: PAGE_HEADER_LEN,
                available: avail,
            });
        }
        let p = &data[offset..];
        if p[..4] != OGG_CAPTURE_PATTERN {
            return Err(OggError::BadCapturePattern { offset });
        }
        if p[4] != 0 {
            return Err(OggError::UnsupportedVersion { version: p[4] });
        }
        let header_type = p[5];
        let granule_position = i64::from_le_bytes(p[6..14].try_into().expect("8 bytes"));
        let serial = u32::from_le_bytes(p[14..18].try_into().expect("4 bytes"));
        let sequence = u32::from_le_bytes(p[18..22].try_into().expect("4 bytes"));
        let stored_crc = u32::from_le_bytes(p[22..26].try_into().expect("4 bytes"));
        let seg_count = p[26] as usize;
        let body_len: usize = if avail < PAGE_HEADER_LEN + seg_count {
            return Err(OggError::TruncatedPage {
                offset,
                needed: PAGE_HEADER_LEN + seg_count,
                available: avail,
            });
        } else {
            p[PAGE_HEADER_LEN..PAGE_HEADER_LEN + seg_count]
                .iter()
                .map(|&l| l as usize)
                .sum()
        };
        let total = PAGE_HEADER_LEN + seg_count + body_len;
        if avail < total {
            return Err(OggError::TruncatedPage {
                offset,
                needed: total,
                available: avail,
            });
        }
        // CRC verify: the checksum covers the whole page with the CRC
        // field zeroed (RFC 3533 §6 item 7).
        let mut check = p[..total].to_vec();
        check[22..26].fill(0);
        let computed = ogg_crc32(&check);
        if computed != stored_crc {
            return Err(OggError::CrcMismatch {
                offset,
                stored: stored_crc,
                computed,
            });
        }
        let lacing = p[PAGE_HEADER_LEN..PAGE_HEADER_LEN + seg_count].to_vec();
        let body = p[PAGE_HEADER_LEN + seg_count..total].to_vec();
        Ok((
            OggPage {
                continued: header_type & 0x01 != 0,
                bos: header_type & 0x02 != 0,
                eos: header_type & 0x04 != 0,
                granule_position,
                serial,
                sequence,
                lacing,
                body,
            },
            offset + total,
        ))
    }
}

/// Parse every page of a physical bitstream, in order. Pages from all
/// logical bitstreams (multiplexed or chained) are returned in their
/// physical interleave order; use [`PacketAssembler`] per serial to
/// recover each stream's packets.
///
/// # Errors
///
/// Any [`OggPage::parse`] error, at the page where it occurred.
pub fn parse_pages(data: &[u8]) -> Result<Vec<OggPage>, OggError> {
    let mut pages = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        let (page, next) = OggPage::parse(data, pos)?;
        pages.push(page);
        pos = next;
    }
    Ok(pages)
}

/// Reassemble one logical bitstream's packets from its pages
/// (RFC 3533 §5-§6: a lacing value `< 255` ends a packet, `255`
/// continues it; the `continued` header flag carries a packet across a
/// page boundary).
///
/// The assembler locks onto the serial of the first page it sees and
/// rejects pages from other logical bitstreams — callers demuxing a
/// multiplexed physical stream run one assembler per serial.
#[derive(Debug, Clone, Default)]
pub struct PacketAssembler {
    serial: Option<u32>,
    pending: Vec<u8>,
    mid_packet: bool,
}

impl PacketAssembler {
    /// Fresh assembler with no locked serial and no partial packet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The serial this assembler locked onto, if any page has been
    /// pushed yet.
    #[must_use]
    pub fn serial(&self) -> Option<u32> {
        self.serial
    }

    /// `true` while a packet is split across a page boundary and its
    /// continuation has not yet arrived.
    #[must_use]
    pub fn mid_packet(&self) -> bool {
        self.mid_packet
    }

    /// Discard any partial packet and unlock the serial (stream reset /
    /// chain boundary).
    pub fn reset(&mut self) {
        self.serial = None;
        self.pending.clear();
        self.mid_packet = false;
    }

    /// Feed one page; returns the packets that *complete* on it, in
    /// order.
    ///
    /// # Errors
    ///
    /// [`OggError::SerialMismatch`] for a page from another logical
    /// bitstream; [`OggError::ContinuityBroken`] when the page's
    /// `continued` flag contradicts the assembler's mid-packet state
    /// (packet loss / corrupt framing).
    pub fn push_page(&mut self, page: &OggPage) -> Result<Vec<Vec<u8>>, OggError> {
        match self.serial {
            None => self.serial = Some(page.serial),
            Some(s) if s != page.serial => {
                return Err(OggError::SerialMismatch {
                    expected: s,
                    actual: page.serial,
                });
            }
            Some(_) => {}
        }
        if page.continued != self.mid_packet {
            return Err(OggError::ContinuityBroken {
                sequence: page.sequence,
                mid_packet: self.mid_packet,
            });
        }
        let mut packets = Vec::new();
        let mut pos = 0usize;
        for &lace in &page.lacing {
            let l = lace as usize;
            self.pending.extend_from_slice(&page.body[pos..pos + l]);
            pos += l;
            if l < 255 {
                packets.push(std::mem::take(&mut self.pending));
                self.mid_packet = false;
            } else {
                self.mid_packet = true;
            }
        }
        Ok(packets)
    }
}

/// Convenience: parse a single-logical-bitstream physical stream to its
/// packet sequence (the degenerate, unmultiplexed form the Vorbis I §A.1.1
/// restrictions describe). Fails on multiplexed input (second serial).
///
/// # Errors
///
/// Any [`parse_pages`] or [`PacketAssembler::push_page`] error.
pub fn pages_to_packets(data: &[u8]) -> Result<Vec<Vec<u8>>, OggError> {
    let pages = parse_pages(data)?;
    let mut assembler = PacketAssembler::new();
    let mut packets = Vec::new();
    for page in &pages {
        packets.extend(assembler.push_page(page)?);
    }
    Ok(packets)
}

/// Packet→page encapsulation for one logical bitstream (RFC 3533 §4-§6).
///
/// Push packets in order with their codec-defined position stamps; the
/// writer chops each packet into 255-byte lacing segments, auto-emits a
/// page whenever the 255-entry segment table fills mid-packet (marking
/// the following page `continued`), and stamps each page's granule
/// position with the stamp of the last packet completed on it (`-1`
/// when none completes — the "page entirely spanned by one packet"
/// case). [`Self::flush_page`] forces a page boundary (the Vorbis I
/// §A.2 rule that the third header packet finishes its page and audio
/// begins fresh); [`Self::finish`] closes the stream, marking the final
/// page `eos`.
///
/// The first emitted page is automatically marked `bos`
/// (RFC 3533 §6 item 3).
#[derive(Debug, Clone)]
pub struct PageWriter {
    serial: u32,
    sequence: u32,
    lacing: Vec<u8>,
    body: Vec<u8>,
    /// Granule stamp of the last packet completed on the pending page.
    page_granule: Option<i64>,
    /// Granule stamp of the last completed packet overall — reused for
    /// the empty EOS-patch path in [`Self::finish`].
    last_granule: i64,
    /// The pending page starts with a continued packet.
    pending_continued: bool,
    /// The next emitted page is the stream's first (gets `bos`).
    bos_pending: bool,
    /// Byte range of the most recently emitted page in `out`.
    last_page_range: Option<(usize, usize)>,
    out: Vec<u8>,
}

impl PageWriter {
    /// Fresh writer for a logical bitstream with the given serial.
    #[must_use]
    pub fn new(serial: u32) -> Self {
        Self {
            serial,
            sequence: 0,
            lacing: Vec::new(),
            body: Vec::new(),
            page_granule: None,
            last_granule: 0,
            pending_continued: false,
            bos_pending: true,
            last_page_range: None,
            out: Vec::new(),
        }
    }

    /// The bytes of every page emitted so far (pending partial-page
    /// data is *not* included until a flush or auto-emit).
    #[must_use]
    pub fn written(&self) -> &[u8] {
        &self.out
    }

    /// Number of pages emitted so far.
    #[must_use]
    pub fn pages_emitted(&self) -> u32 {
        self.sequence
    }

    /// Append one packet with its codec-defined position stamp
    /// (`granulepos` of the stream *after* this packet; for Vorbis §A.2
    /// audio packets, the end PCM sample position, and `0` for the
    /// three header packets).
    pub fn push_packet(&mut self, packet: &[u8], granulepos: i64) {
        // The pending page never ends mid-packet at entry (mid-packet
        // fills are emitted inside the loop below), so a full pending
        // table here just needs a plain emit first.
        if self.lacing.len() == MAX_PAGE_SEGMENTS {
            self.emit_page(false, false);
        }
        let mut remaining = packet;
        loop {
            while self.lacing.len() < MAX_PAGE_SEGMENTS {
                let take = remaining.len().min(255);
                self.lacing.push(take as u8);
                self.body.extend_from_slice(&remaining[..take]);
                remaining = &remaining[take..];
                if take < 255 {
                    // Final segment — the packet completes here.
                    self.page_granule = Some(granulepos);
                    self.last_granule = granulepos;
                    return;
                }
                // take == 255: the packet continues. When it has no
                // bytes left, the next iteration pushes the required
                // zero-length terminating segment.
            }
            // Segment table full mid-packet: emit and continue the
            // packet on the next page.
            self.emit_page(true, false);
        }
    }

    /// Force a page boundary: emit the pending partial page, if any.
    /// Used for the Vorbis I §A.2 header-page rules (the identification
    /// header alone on the first page; the setup header finishing its
    /// page so audio begins fresh).
    pub fn flush_page(&mut self) {
        if !self.lacing.is_empty() {
            self.emit_page(false, false);
        }
    }

    /// Close the stream: flush any pending data on a final page marked
    /// `eos` (RFC 3533 §6 item 3). When nothing is pending, the most
    /// recently emitted page is re-stamped as `eos` in place (CRC
    /// recomputed). Returns the complete physical bitstream bytes.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        if !self.lacing.is_empty() {
            self.emit_page(false, true);
        } else if let Some((start, end)) = self.last_page_range {
            self.out[start + 5] |= 0x04;
            self.out[start + 22..start + 26].fill(0);
            let crc = ogg_crc32(&self.out[start..end]);
            self.out[start + 22..start + 26].copy_from_slice(&crc.to_le_bytes());
        }
        self.out
    }

    fn emit_page(&mut self, next_continued: bool, eos: bool) {
        let page = OggPage {
            continued: self.pending_continued,
            bos: self.bos_pending,
            eos,
            granule_position: self.page_granule.unwrap_or(-1),
            serial: self.serial,
            sequence: self.sequence,
            lacing: std::mem::take(&mut self.lacing),
            body: std::mem::take(&mut self.body),
        };
        self.bos_pending = false;
        self.pending_continued = next_continued;
        self.page_granule = None;
        self.sequence += 1;
        let start = self.out.len();
        let bytes = page
            .serialize()
            .expect("writer-built pages satisfy the lacing invariants");
        self.out.extend_from_slice(&bytes);
        self.last_page_range = Some((start, self.out.len()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- CRC ----------

    #[test]
    fn crc_of_empty_input_is_zero() {
        assert_eq!(ogg_crc32(&[]), 0);
    }

    #[test]
    fn crc_is_sensitive_to_every_byte() {
        let base = ogg_crc32(b"OggS\x00\x02 test payload");
        for i in 0..18 {
            let mut mutated = b"OggS\x00\x02 test payload".to_vec();
            mutated[i] ^= 0x01;
            assert_ne!(ogg_crc32(&mutated), base, "byte {i} did not affect CRC");
        }
    }

    #[test]
    fn crc_msb_first_single_byte_property() {
        // Feeding a single byte b puts b << 24 through 8 shift steps of
        // the 0x04c11db7 polynomial — cross-check the table against a
        // bit-serial evaluation.
        for b in [0x00u8, 0x01, 0x80, 0xff, 0x5a] {
            let mut r = (b as u32) << 24;
            for _ in 0..8 {
                r = if r & 0x8000_0000 != 0 {
                    (r << 1) ^ 0x04c1_1db7
                } else {
                    r << 1
                };
            }
            assert_eq!(ogg_crc32(&[b]), r, "byte {b:#04x}");
        }
    }

    // ---------- page serialize / parse ----------

    fn sample_page() -> OggPage {
        OggPage {
            continued: false,
            bos: true,
            eos: false,
            granule_position: 0,
            serial: 0x1234_5678,
            sequence: 0,
            lacing: vec![30],
            body: vec![0xAB; 30],
        }
    }

    #[test]
    fn page_roundtrips_through_serialize_and_parse() {
        let page = sample_page();
        let bytes = page.serialize().unwrap();
        assert_eq!(bytes.len(), page.page_len());
        let (parsed, next) = OggPage::parse(&bytes, 0).unwrap();
        assert_eq!(parsed, page);
        assert_eq!(next, bytes.len());
    }

    #[test]
    fn parse_rejects_corrupt_crc() {
        let mut bytes = sample_page().serialize().unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x40;
        match OggPage::parse(&bytes, 0) {
            Err(OggError::CrcMismatch { offset: 0, .. }) => {}
            other => panic!("expected CrcMismatch, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_bad_capture_and_version_and_truncation() {
        let bytes = sample_page().serialize().unwrap();
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert_eq!(
            OggPage::parse(&bad, 0),
            Err(OggError::BadCapturePattern { offset: 0 })
        );
        let mut vers = bytes.clone();
        vers[4] = 1;
        assert_eq!(
            OggPage::parse(&vers, 0),
            Err(OggError::UnsupportedVersion { version: 1 })
        );
        match OggPage::parse(&bytes[..20], 0) {
            Err(OggError::TruncatedPage { .. }) => {}
            other => panic!("expected TruncatedPage, got {other:?}"),
        }
    }

    #[test]
    fn serialize_rejects_bad_lacing() {
        let mut page = sample_page();
        page.lacing = vec![10];
        assert_eq!(
            page.serialize(),
            Err(OggError::BodyLengthMismatch {
                expected: 10,
                actual: 30
            })
        );
        page.lacing = vec![0; 256];
        page.body.clear();
        assert_eq!(
            page.serialize(),
            Err(OggError::TooManySegments { segments: 256 })
        );
    }

    #[test]
    fn header_flags_roundtrip() {
        for (continued, bos, eos) in [
            (false, false, false),
            (true, false, false),
            (false, true, false),
            (false, false, true),
            (true, false, true),
        ] {
            let page = OggPage {
                continued,
                bos,
                eos,
                granule_position: -1,
                serial: 7,
                sequence: 3,
                lacing: vec![],
                body: vec![],
            };
            let bytes = page.serialize().unwrap();
            let (parsed, _) = OggPage::parse(&bytes, 0).unwrap();
            assert_eq!(parsed, page);
        }
    }

    // ---------- writer packet segmentation ----------

    /// Round-trip a packet sequence through PageWriter and the parse +
    /// assemble stack; returns (pages, packets).
    fn writer_roundtrip(packets: &[(Vec<u8>, i64)]) -> (Vec<OggPage>, Vec<Vec<u8>>) {
        let mut w = PageWriter::new(0xDEAD_BEEF);
        for (p, g) in packets {
            w.push_packet(p, *g);
        }
        let bytes = w.finish();
        let pages = parse_pages(&bytes).unwrap();
        let got = pages_to_packets(&bytes).unwrap();
        (pages, got)
    }

    #[test]
    fn writer_roundtrips_simple_packets() {
        let packets: Vec<(Vec<u8>, i64)> = vec![
            (vec![1u8; 30], 0),
            (vec![2u8; 300], 128),
            (vec![3u8; 1], 256),
        ];
        let (pages, got) = writer_roundtrip(&packets);
        assert_eq!(
            got,
            packets.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>()
        );
        assert!(pages[0].bos);
        assert!(pages.last().unwrap().eos);
        assert_eq!(pages.last().unwrap().granule_position, 256);
    }

    #[test]
    fn exact_255_multiple_packet_gets_zero_lacing_terminator() {
        let (pages, got) = writer_roundtrip(&[(vec![9u8; 510], 42)]);
        assert_eq!(got, vec![vec![9u8; 510]]);
        // 255, 255, 0 — the zero terminator marks completion.
        let all_lacing: Vec<u8> = pages.iter().flat_map(|p| p.lacing.clone()).collect();
        assert_eq!(all_lacing, vec![255, 255, 0]);
    }

    #[test]
    fn zero_length_packet_is_a_single_zero_lacing() {
        let (pages, got) = writer_roundtrip(&[(Vec::new(), 5), (vec![1u8; 4], 6)]);
        assert_eq!(got, vec![Vec::new(), vec![1u8; 4]]);
        assert_eq!(pages[0].lacing[0], 0);
    }

    #[test]
    fn oversize_packet_spans_pages_with_continued_flag_and_minus_one_granule() {
        // 255 segments * 255 bytes = 65025 bytes fill page 0 exactly;
        // a 70000-byte packet must continue onto page 1.
        let (pages, got) = writer_roundtrip(&[(vec![7u8; 70_000], 99)]);
        assert_eq!(got, vec![vec![7u8; 70_000]]);
        assert!(pages.len() >= 2);
        assert!(!pages[0].continued);
        assert!(pages[1].continued, "page 1 must continue the packet");
        assert_eq!(
            pages[0].granule_position, -1,
            "no packet completes on page 0"
        );
        assert_eq!(pages.last().unwrap().granule_position, 99);
        // Sequence numbers count up from 0.
        for (i, p) in pages.iter().enumerate() {
            assert_eq!(p.sequence, i as u32);
        }
    }

    #[test]
    fn flush_page_forces_a_boundary() {
        let mut w = PageWriter::new(1);
        w.push_packet(&[1u8; 10], 0);
        w.flush_page();
        w.push_packet(&[2u8; 10], 1);
        let bytes = w.finish();
        let pages = parse_pages(&bytes).unwrap();
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].lacing, vec![10]);
        assert_eq!(pages[1].lacing, vec![10]);
        assert!(!pages[1].continued);
    }

    #[test]
    fn finish_patches_eos_onto_an_already_flushed_final_page() {
        let mut w = PageWriter::new(1);
        w.push_packet(&[1u8; 10], 7);
        w.flush_page(); // page already emitted; finish() must patch it
        let bytes = w.finish();
        let pages = parse_pages(&bytes).unwrap();
        assert_eq!(pages.len(), 1);
        assert!(pages[0].eos, "EOS must be patched in place");
        assert_eq!(pages[0].granule_position, 7);
        // The patched page still CRC-verifies (parse_pages checked it).
    }

    #[test]
    fn empty_writer_finish_produces_no_pages() {
        let w = PageWriter::new(1);
        assert!(w.finish().is_empty());
    }

    // ---------- assembler guards ----------

    #[test]
    fn assembler_rejects_serial_switch_and_broken_continuity() {
        let page_a = OggPage {
            continued: false,
            bos: true,
            eos: false,
            granule_position: 0,
            serial: 1,
            sequence: 0,
            lacing: vec![255],
            body: vec![0; 255],
        };
        let mut asm = PacketAssembler::new();
        assert!(asm.push_page(&page_a).unwrap().is_empty());
        assert!(asm.mid_packet());

        let mut other = page_a.clone();
        other.serial = 2;
        assert_eq!(
            asm.push_page(&other),
            Err(OggError::SerialMismatch {
                expected: 1,
                actual: 2
            })
        );

        // A fresh (non-continued) page while mid-packet is a
        // continuity break.
        let mut fresh = page_a.clone();
        fresh.sequence = 1;
        fresh.lacing = vec![3];
        fresh.body = vec![0; 3];
        assert_eq!(
            asm.push_page(&fresh),
            Err(OggError::ContinuityBroken {
                sequence: 1,
                mid_packet: true
            })
        );
    }
}

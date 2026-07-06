//! Vorbis I §A ("Embedding Vorbis into an Ogg stream") encapsulation.
//!
//! [`crate::ogg`] provides the codec-agnostic RFC 3533 page transport;
//! this module adds the Vorbis-specific mapping rules of §A.2 on top:
//!
//! * the **identification header is placed alone on the first page**
//!   of the logical stream (yielding the canonical 58-byte first page,
//!   marked `bos`);
//! * the **comment and setup headers** begin on the second page and
//!   may span pages; the setup header **finishes the page it ends on**,
//!   so the first audio packet begins a fresh page;
//! * header pages carry **granule position 0**;
//! * audio pages carry the **end PCM sample position of the last
//!   packet completed on the page** (per channel — a stereo stream's
//!   granule does not tick twice as fast), `-1` when a single packet
//!   entirely spans the page;
//! * the **last page is marked `eos`**; a final granule position that
//!   indicates *less* audio than the final packet would naturally
//!   return instructs the decoder to trim the trailing samples — how a
//!   stream ends on a non-block-aligned sample count.
//!
//! The muxer validates the packet sequence with the §4.2.1 / §4.3.1
//! packet-kind classifier: exactly one identification, comment and
//! setup header, in order, then audio packets with non-decreasing
//! granule positions.

use crate::ogg::PageWriter;
use crate::packet_kind::{classify_packet, ClassifyError, PacketKind};

/// Muxer sequencing errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuxError {
    /// A header packet arrived out of §A.2 order (or an audio packet
    /// arrived where a header was required).
    HeaderOrder {
        /// The header kind the muxer expected next.
        expected: PacketKind,
        /// The kind actually classified.
        got: PacketKind,
    },
    /// [`VorbisOggMuxer::push_audio`] was called before all three
    /// header packets were pushed.
    HeadersIncomplete,
    /// A header packet was pushed after audio started.
    HeaderAfterAudio,
    /// The packet failed §4.2.1 / §4.3.1 classification.
    Classify(ClassifyError),
    /// An audio packet's granule position went backwards.
    NonMonotoneGranule {
        /// The previous packet's granule position.
        prev: u64,
        /// The offending packet's granule position.
        got: u64,
    },
    /// [`VorbisOggMuxer::finish`] was called before all three header
    /// packets were pushed — an Ogg/Vorbis stream without its headers
    /// is not decodable and §A.2 gives it no valid shape.
    FinishBeforeHeaders,
}

impl core::fmt::Display for MuxError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MuxError::HeaderOrder { expected, got } => {
                write!(
                    f,
                    "ogg/vorbis mux: expected {expected:?} header, got {got:?}"
                )
            }
            MuxError::HeadersIncomplete => {
                write!(f, "ogg/vorbis mux: audio pushed before the three headers")
            }
            MuxError::HeaderAfterAudio => {
                write!(f, "ogg/vorbis mux: header packet pushed after audio began")
            }
            MuxError::Classify(e) => write!(f, "ogg/vorbis mux: {e}"),
            MuxError::NonMonotoneGranule { prev, got } => write!(
                f,
                "ogg/vorbis mux: granule position {got} < previous {prev}"
            ),
            MuxError::FinishBeforeHeaders => {
                write!(f, "ogg/vorbis mux: finish() before the three headers")
            }
        }
    }
}

impl std::error::Error for MuxError {}

impl From<ClassifyError> for MuxError {
    fn from(value: ClassifyError) -> Self {
        MuxError::Classify(value)
    }
}

/// Soft page-size target for audio pages, in body bytes. RFC 3533 §6
/// describes pages as "usually 4-8 kB"; the muxer flushes the pending
/// page once its body reaches this size (a packet that overshoots it
/// still lands whole unless the 255-segment table forces a split).
const AUDIO_PAGE_TARGET_BYTES: usize = 4096;

/// Which packet the muxer expects next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MuxState {
    ExpectIdentification,
    ExpectComment,
    ExpectSetup,
    Audio,
}

/// Ogg encapsulation of one Vorbis logical bitstream per §A.2.
///
/// Feed the three header packets with [`push_header`](Self::push_header)
/// (order-checked), then audio packets with their absolute end-PCM
/// granule positions via [`push_audio`](Self::push_audio), and call
/// [`finish`](Self::finish) to close the stream. The §A.2 page-break
/// rules (identification header alone on page 0; setup header finishing
/// its page; audio beginning fresh) are applied automatically.
#[derive(Debug, Clone)]
pub struct VorbisOggMuxer {
    writer: PageWriter,
    state: MuxState,
    last_granule: u64,
    audio_packets: u64,
}

impl VorbisOggMuxer {
    /// New muxer for a logical bitstream with the given serial number.
    #[must_use]
    pub fn new(serial: u32) -> Self {
        Self {
            writer: PageWriter::new(serial),
            state: MuxState::ExpectIdentification,
            last_granule: 0,
            audio_packets: 0,
        }
    }

    /// `true` once all three §4.2 header packets have been accepted.
    #[must_use]
    pub fn headers_done(&self) -> bool {
        self.state == MuxState::Audio
    }

    /// Number of audio packets accepted so far.
    #[must_use]
    pub fn audio_packet_count(&self) -> u64 {
        self.audio_packets
    }

    /// Push the next header packet. Must be called exactly three times,
    /// with the identification, comment and setup headers in §4.2.1
    /// order; the packet kind is verified with the §4.2.1 classifier.
    ///
    /// Header packets are stamped with granule position 0 (§A.2). After
    /// the identification header the page is flushed (it sits alone on
    /// the 58-byte first page); after the setup header the page is
    /// flushed again so the first audio packet begins a fresh page.
    ///
    /// # Errors
    ///
    /// [`MuxError::HeaderOrder`] for the wrong packet kind,
    /// [`MuxError::HeaderAfterAudio`] once audio has begun,
    /// [`MuxError::Classify`] for an unclassifiable packet.
    pub fn push_header(&mut self, packet: &[u8]) -> Result<(), MuxError> {
        let expected = match self.state {
            MuxState::ExpectIdentification => PacketKind::Identification,
            MuxState::ExpectComment => PacketKind::Comment,
            MuxState::ExpectSetup => PacketKind::Setup,
            MuxState::Audio => return Err(MuxError::HeaderAfterAudio),
        };
        let got = classify_packet(packet)?;
        if got != expected {
            return Err(MuxError::HeaderOrder { expected, got });
        }
        self.writer.push_packet(packet, 0);
        self.state = match self.state {
            MuxState::ExpectIdentification => {
                // §A.2: the identification header is alone on page 0.
                self.writer.flush_page();
                MuxState::ExpectComment
            }
            MuxState::ExpectComment => MuxState::ExpectSetup,
            MuxState::ExpectSetup => {
                // §A.2: the setup header finishes its page; audio
                // begins fresh.
                self.writer.flush_page();
                MuxState::Audio
            }
            MuxState::Audio => unreachable!("handled above"),
        };
        Ok(())
    }

    /// Push one audio packet with its absolute granule position — the
    /// end PCM sample position (per channel) of the stream after this
    /// packet's samples are returned by decode (§A.2). The first audio
    /// packet primes the §4.3.8 overlap-add and returns no PCM, so its
    /// granule position is the initial PCM offset (0 for a stream that
    /// starts at time zero).
    ///
    /// The final packet's granule position may deliberately understate
    /// the naturally decodable sample count — §A.2's end-trim rule for
    /// ending a stream on a non-block-aligned length.
    ///
    /// # Errors
    ///
    /// [`MuxError::HeadersIncomplete`] before the three headers,
    /// [`MuxError::NonMonotoneGranule`] if `granulepos` went backwards.
    pub fn push_audio(&mut self, packet: &[u8], granulepos: u64) -> Result<(), MuxError> {
        if self.state != MuxState::Audio {
            return Err(MuxError::HeadersIncomplete);
        }
        if granulepos < self.last_granule {
            return Err(MuxError::NonMonotoneGranule {
                prev: self.last_granule,
                got: granulepos,
            });
        }
        self.last_granule = granulepos;
        self.audio_packets += 1;
        self.writer.push_packet(packet, granulepos as i64);
        // Soft page-size policy: keep audio pages in the RFC 3533
        // "usually 4-8 kB" band.
        if self.writer.pending_body_len() >= AUDIO_PAGE_TARGET_BYTES {
            self.writer.flush_page();
        }
        Ok(())
    }

    /// Close the logical stream: flush the pending page and mark the
    /// final page `eos`. Returns the complete physical-bitstream bytes.
    ///
    /// # Errors
    ///
    /// [`MuxError::FinishBeforeHeaders`] when fewer than three header
    /// packets were pushed (a headerless Ogg/Vorbis stream has no
    /// §A.2-valid shape). A stream with headers but no audio packets is
    /// permitted (an empty-but-well-formed stream).
    pub fn finish(self) -> Result<Vec<u8>, MuxError> {
        if self.state != MuxState::Audio {
            return Err(MuxError::FinishBeforeHeaders);
        }
        Ok(self.writer.finish())
    }
}

/// One-call §A.2 encapsulation of a complete packet stream:
/// `(identification, comment, setup)` headers plus `(packet,
/// granulepos)` audio packets, under the given serial.
///
/// # Errors
///
/// Any [`VorbisOggMuxer`] sequencing error.
pub fn mux_vorbis_stream(
    serial: u32,
    identification: &[u8],
    comment: &[u8],
    setup: &[u8],
    audio: &[(Vec<u8>, u64)],
) -> Result<Vec<u8>, MuxError> {
    let mut muxer = VorbisOggMuxer::new(serial);
    muxer.push_header(identification)?;
    muxer.push_header(comment)?;
    muxer.push_header(setup)?;
    for (packet, granulepos) in audio {
        muxer.push_audio(packet, *granulepos)?;
    }
    muxer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::{write_comment_header, write_identification_header};
    use crate::identification::VorbisIdentificationHeader;
    use crate::ogg::{pages_to_packets, parse_pages};
    use crate::VorbisCommentHeader;

    fn id_packet() -> Vec<u8> {
        write_identification_header(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 1,
            audio_sample_rate: 44_100,
            bitrate_maximum: 0,
            bitrate_nominal: 0,
            bitrate_minimum: 0,
            blocksize_0: 1024,
            blocksize_1: 1024,
        })
        .expect("id header writes")
    }

    fn comment_packet() -> Vec<u8> {
        write_comment_header(&VorbisCommentHeader {
            vendor: "oxideav-vorbis".into(),
            comments: vec!["ENCODER=oxideav-vorbis".into()],
        })
        .expect("comment header writes")
    }

    /// A stand-in setup packet: only the §4.2.1 prelude is inspected by
    /// the muxer's classifier, so a synthetic body suffices for
    /// sequencing tests.
    fn setup_packet(len: usize) -> Vec<u8> {
        let mut p = vec![0x05];
        p.extend_from_slice(b"vorbis");
        p.resize(len.max(7), 0xA5);
        p
    }

    /// An audio packet: byte 0's LSB is 0 (§4.3.1 step 1).
    fn audio_packet(len: usize, fill: u8) -> Vec<u8> {
        vec![fill & 0xFE; len.max(1)]
    }

    #[test]
    fn a2_page_layout_holds() {
        let audio: Vec<(Vec<u8>, u64)> = (0..20u64)
            .map(|i| (audio_packet(300, (i as u8) << 1), i * 512))
            .collect();
        let bytes = mux_vorbis_stream(
            0x0DA7,
            &id_packet(),
            &comment_packet(),
            &setup_packet(200),
            &audio,
        )
        .expect("muxes");
        let pages = parse_pages(&bytes).expect("pages parse");

        // §A.2: id header alone on a 58-byte BOS first page, granule 0.
        assert_eq!(pages[0].page_len(), 58);
        assert!(pages[0].bos);
        assert_eq!(pages[0].granule_position, 0);
        assert_eq!(pages[0].lacing.len(), 1);

        // §A.2: comment + setup share page 1, granule 0; audio begins
        // on page 2.
        assert_eq!(pages[1].lacing.iter().filter(|&&l| l < 255).count(), 2);
        assert_eq!(pages[1].granule_position, 0);
        assert!(!pages[2].continued);

        // Last page is EOS with the final packet's granule.
        assert!(pages.last().unwrap().eos);
        assert_eq!(pages.last().unwrap().granule_position, 19 * 512);

        // The packets round-trip exactly.
        let packets = pages_to_packets(&bytes).expect("packets assemble");
        assert_eq!(packets.len(), 3 + audio.len());
        assert_eq!(packets[0], id_packet());
        assert_eq!(packets[1], comment_packet());
        assert_eq!(packets[2], setup_packet(200));
        for (i, (p, _)) in audio.iter().enumerate() {
            assert_eq!(&packets[3 + i], p, "audio packet {i}");
        }
    }

    #[test]
    fn a2_setup_header_spanning_pages_still_ends_its_page() {
        // A setup header far over one page's capacity (255 × 255 bytes)
        // must span pages and still finish its page before audio.
        let setup = setup_packet(70_000);
        let audio = vec![(audio_packet(100, 2), 512u64)];
        let bytes =
            mux_vorbis_stream(1, &id_packet(), &comment_packet(), &setup, &audio).expect("muxes");
        let pages = parse_pages(&bytes).expect("pages parse");
        // Page 1 holds the completed comment header (granule 0) and
        // fills mid-setup, so page 2 continues the setup packet.
        assert_eq!(pages[1].granule_position, 0);
        assert!(pages[2].continued);
        // The audio packet begins a fresh page after the setup ends.
        let audio_page = pages
            .iter()
            .position(|p| p.granule_position == 512)
            .expect("audio page present");
        assert!(!pages[audio_page].continued);
        let packets = pages_to_packets(&bytes).expect("packets assemble");
        assert_eq!(packets[2], setup);
        assert_eq!(packets[3], audio.first().unwrap().0);
    }

    #[test]
    fn audio_pages_respect_the_size_target() {
        // 100 × 600-byte packets: pages flush at ≥ 4 kB, so every page
        // body (bar the last) sits in the 4-8 kB RFC band.
        let audio: Vec<(Vec<u8>, u64)> = (0..100u64)
            .map(|i| (audio_packet(600, 4), (i + 1) * 512))
            .collect();
        let bytes = mux_vorbis_stream(
            2,
            &id_packet(),
            &comment_packet(),
            &setup_packet(80),
            &audio,
        )
        .expect("muxes");
        let pages = parse_pages(&bytes).expect("pages parse");
        for (i, page) in pages.iter().enumerate().skip(2) {
            if i + 1 < pages.len() {
                assert!(
                    page.body.len() >= 4096 || i < 2,
                    "page {i} body {} below the flush target",
                    page.body.len()
                );
                assert!(
                    page.body.len() <= 8192,
                    "page {i} body {} above the RFC band",
                    page.body.len()
                );
            }
        }
        // Every audio page's granule is the last completed packet's.
        let mut idx = 0usize;
        for page in &pages[2..] {
            let completed = page.lacing.iter().filter(|&&l| l < 255).count();
            if completed > 0 {
                idx += completed;
                assert_eq!(page.granule_position, audio[idx - 1].1 as i64);
            } else {
                assert_eq!(page.granule_position, -1);
            }
        }
        assert_eq!(idx, audio.len());
    }

    #[test]
    fn sequencing_guards_fire() {
        let mut m = VorbisOggMuxer::new(9);
        // Audio before headers.
        assert_eq!(
            m.push_audio(&audio_packet(4, 0), 0),
            Err(MuxError::HeadersIncomplete)
        );
        // Wrong first header.
        assert_eq!(
            m.push_header(&comment_packet()),
            Err(MuxError::HeaderOrder {
                expected: PacketKind::Identification,
                got: PacketKind::Comment
            })
        );
        // finish() before headers has no valid §A.2 shape.
        assert_eq!(
            VorbisOggMuxer::new(9).finish(),
            Err(MuxError::FinishBeforeHeaders)
        );

        m.push_header(&id_packet()).unwrap();
        m.push_header(&comment_packet()).unwrap();
        m.push_header(&setup_packet(50)).unwrap();
        assert!(m.headers_done());
        // A fourth header is refused.
        assert_eq!(m.push_header(&id_packet()), Err(MuxError::HeaderAfterAudio));
        // Granule must not go backwards.
        m.push_audio(&audio_packet(4, 0), 1024).unwrap();
        assert_eq!(
            m.push_audio(&audio_packet(4, 0), 512),
            Err(MuxError::NonMonotoneGranule {
                prev: 1024,
                got: 512
            })
        );
        // Equal granule (a zero-sample packet) is fine.
        m.push_audio(&audio_packet(4, 0), 1024).unwrap();
        assert_eq!(m.audio_packet_count(), 2);
    }

    #[test]
    fn headers_only_stream_is_well_formed() {
        let mut m = VorbisOggMuxer::new(3);
        m.push_header(&id_packet()).unwrap();
        m.push_header(&comment_packet()).unwrap();
        m.push_header(&setup_packet(60)).unwrap();
        let bytes = m.finish().expect("finishes");
        let pages = parse_pages(&bytes).expect("pages parse");
        assert_eq!(pages.len(), 2);
        assert!(pages[0].bos);
        assert!(pages[1].eos, "EOS patched onto the flushed header page");
        let packets = pages_to_packets(&bytes).expect("packets assemble");
        assert_eq!(packets.len(), 3);
    }
}

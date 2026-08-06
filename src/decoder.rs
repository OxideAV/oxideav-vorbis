//! [`oxideav_core::Decoder`] implementation — the framework-facing
//! packet-to-frame decoder, plus the crate's direct-API factory
//! endpoint [`make_decoder`].
//!
//! The decoder consumes **Vorbis codec packets** (Ogg framing already
//! stripped by the demuxer): the three §4.2 header packets first —
//! identification, comment, setup, in order, classified with the
//! §4.2.1 packet-kind classifier — then §4.3 audio packets, which run
//! through the crate's streaming §4.3 pipeline
//! ([`crate::streaming::StreamingDecoder`]) and surface as planar-f32
//! [`AudioFrame`]s in bitstream channel order. Timestamps count PCM
//! samples (`time_base = 1 / sample_rate`).
//!
//! The three headers arrive on either of the two carriage shapes
//! containers use:
//!
//! - **in-band** — the demuxer hands every packet through
//!   [`Decoder::send_packet`], headers included (raw packet feeds,
//!   this crate's own tests);
//! - **as codec-private extradata** — the demuxer consumed the
//!   headers while opening the stream and republished them as the
//!   Xiph-laced `CodecParameters::extradata` blob (the `oxideav-ogg`
//!   demuxer, Matroska `CodecPrivate` for `A_VORBIS`). When
//!   [`make_decoder`] receives a non-empty `extradata` it unlaces and
//!   parses the headers up front, so the first `send_packet` may
//!   already be a §4.3 audio packet.

use std::collections::VecDeque;

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Decoder, Error, Frame, Packet, Result, SampleFormat,
};

use crate::audio::AudioDecoderState;
use crate::identification::{parse_identification_header, VorbisIdentificationHeader};
use crate::packet_kind::{classify_packet, PacketKind};
use crate::setup::{parse_setup_header, VorbisSetupHeader};
use crate::streaming::{StreamingDecoder, StreamingFrame};

/// Which header packet the decoder expects next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeaderStage {
    Identification,
    Comment,
    Setup,
    Audio,
}

/// The framework-facing Vorbis decoder. Build via [`make_decoder`] (or
/// through the registry after [`crate::register`]).
pub struct VorbisDecoder {
    codec_id: CodecId,
    stage: HeaderStage,
    id: Option<VorbisIdentificationHeader>,
    setup: Option<(VorbisSetupHeader, AudioDecoderState)>,
    streaming: Option<StreamingDecoder>,
    /// Decoded frames awaiting `receive_frame`.
    queue: VecDeque<Frame>,
    /// Running PCM sample position (per channel) for frame PTS.
    position: i64,
    /// `flush()` was called: drain then EOF.
    flushed: bool,
}

impl std::fmt::Debug for VorbisDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VorbisDecoder")
            .field("stage", &self.stage)
            .field("queued", &self.queue.len())
            .field("position", &self.position)
            .finish()
    }
}

impl VorbisDecoder {
    /// Fresh decoder expecting the three §4.2 header packets in-band.
    #[must_use]
    pub fn new() -> Self {
        VorbisDecoder {
            codec_id: CodecId::new("vorbis"),
            stage: HeaderStage::Identification,
            id: None,
            setup: None,
            streaming: None,
            queue: VecDeque::new(),
            position: 0,
            flushed: false,
        }
    }

    /// Decoder pre-configured from a Xiph-laced codec-private blob —
    /// the `CodecParameters::extradata` shape container layers carry
    /// for Vorbis (the `oxideav-ogg` demuxer republishes the three
    /// consumed header packets there; Matroska stores the same blob as
    /// the `A_VORBIS` `CodecPrivate`). The blob is unlaced with the
    /// container crate's [`oxideav_ogg::mux::xiph_unlace`] (the exact
    /// inverse of [`crate::oggfile::lace_vorbis_headers`]) and the
    /// three packets run through the same §4.2.1 order-checked header
    /// path in-band packets take, so the returned decoder is already
    /// at the audio stage.
    ///
    /// # Errors
    ///
    /// The blob must unlace into exactly three packets that parse as
    /// the §4.2 identification / comment / setup headers in order;
    /// anything else is [`Error::InvalidData`].
    pub fn with_extradata(extradata: &[u8]) -> Result<Self> {
        let packets = oxideav_ogg::mux::xiph_unlace(extradata).ok_or_else(|| {
            Error::invalid("vorbis: extradata is not a Xiph-laced codec-private blob")
        })?;
        if packets.len() != 3 {
            return Err(Error::invalid(format!(
                "vorbis: extradata laces {} packets, expected the 3 §4.2 headers",
                packets.len()
            )));
        }
        let mut dec = Self::new();
        for packet in &packets {
            dec.consume_packet(packet)?;
        }
        debug_assert_eq!(dec.stage, HeaderStage::Audio);
        Ok(dec)
    }

    /// The stream's sample rate once the identification header has
    /// been parsed (frame PTS values count samples at this rate).
    #[must_use]
    pub fn sample_rate(&self) -> Option<u32> {
        self.id.as_ref().map(|id| id.audio_sample_rate)
    }
}

impl Default for VorbisDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for VorbisDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.consume_packet(&packet.data)
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(frame) = self.queue.pop_front() {
            return Ok(frame);
        }
        if self.flushed {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        // The §4.3.8 overlap tail needs a successor packet to resolve;
        // at end of stream there is none (the §A.2 final-page granule
        // already told the demuxer where the stream ends), so flushing
        // just switches receive_frame's empty answer to Eof.
        self.flushed = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        if let Some(streaming) = self.streaming.as_mut() {
            streaming.reset();
        }
        self.queue.clear();
        self.flushed = false;
        // Sample position restarts; the demuxer re-stamps packet
        // timestamps after a seek.
        self.position = 0;
        Ok(())
    }
}

impl VorbisDecoder {
    /// One codec packet's bytes through the §4.2.1 header state
    /// machine / §4.3 audio pipeline — the shared body behind both
    /// [`Decoder::send_packet`] (in-band packets) and
    /// [`VorbisDecoder::with_extradata`] (codec-private headers).
    fn consume_packet(&mut self, data: &[u8]) -> Result<()> {
        let kind = classify_packet(data)
            .map_err(|e| Error::invalid(format!("vorbis: unclassifiable packet: {e}")))?;
        match (self.stage, kind) {
            (HeaderStage::Identification, PacketKind::Identification) => {
                let id = parse_identification_header(data)
                    .map_err(|e| Error::invalid(format!("vorbis: identification header: {e}")))?;
                self.id = Some(id);
                self.stage = HeaderStage::Comment;
                Ok(())
            }
            (HeaderStage::Comment, PacketKind::Comment) => {
                // The §5 comment body carries stream metadata a
                // demuxer surfaces elsewhere; the decode path only
                // requires its presence in the §4.2.1 order.
                self.stage = HeaderStage::Setup;
                Ok(())
            }
            (HeaderStage::Setup, PacketKind::Setup) => {
                let id = self.id.as_ref().expect("id parsed before setup stage");
                let setup = parse_setup_header(data, id.audio_channels)
                    .map_err(|e| Error::invalid(format!("vorbis: setup header: {e}")))?;
                let state = AudioDecoderState::new(&setup)
                    .map_err(|e| Error::invalid(format!("vorbis: decoder state: {e:?}")))?;
                self.streaming = Some(StreamingDecoder::new(
                    id.audio_channels,
                    id.blocksize_0 as usize,
                    id.blocksize_1 as usize,
                    1.0,
                ));
                self.setup = Some((setup, state));
                self.stage = HeaderStage::Audio;
                Ok(())
            }
            (HeaderStage::Audio, PacketKind::Audio) => {
                let (setup, state) = self.setup.as_ref().expect("setup parsed");
                let streaming = self.streaming.as_mut().expect("streaming built");
                let mut reader = oxideav_core::bits::BitReaderLsb::new(data);
                let outcome = streaming
                    .push_packet(&mut reader, setup, state)
                    .map_err(|e| Error::invalid(format!("vorbis: audio packet: {e}")))?;
                if let StreamingFrame::Pcm {
                    per_channel_pcm, ..
                } = outcome
                {
                    let samples = per_channel_pcm.first().map(Vec::len).unwrap_or(0);
                    let data: Vec<Vec<u8>> = per_channel_pcm
                        .iter()
                        .map(|row| row.iter().flat_map(|s| s.to_le_bytes()).collect())
                        .collect();
                    let frame = AudioFrame {
                        samples: samples as u32,
                        pts: Some(self.position),
                        data,
                    };
                    self.position += samples as i64;
                    self.queue.push_back(Frame::Audio(frame));
                }
                Ok(())
            }
            (HeaderStage::Audio, header_kind) => Err(Error::invalid(format!(
                "vorbis: unexpected {header_kind:?} header after audio began"
            ))),
            (_, got) => Err(Error::invalid(format!(
                "vorbis: expected {:?} header packet, got {got:?}",
                self.stage
            ))),
        }
    }
}

/// Direct-API factory endpoint: build a boxed [`VorbisDecoder`] for the
/// given stream parameters. This is also the factory `register`
/// installs into the [`oxideav_core`] codec registry.
///
/// When `params.extradata` is non-empty it is the Xiph-laced
/// codec-private blob (the `oxideav-ogg` demuxer's republished header
/// packets, Matroska `A_VORBIS` `CodecPrivate`): the three §4.2
/// headers are unlaced and parsed up front
/// ([`VorbisDecoder::with_extradata`]), so the demuxed packet feed —
/// which no longer carries the headers — decodes directly. With empty
/// `extradata` the decoder instead self-configures from in-band §4.2
/// header packets (a raw Vorbis packet feed always leads with them).
/// Sample rate / channel count are validated against the
/// identification header by the caller if desired.
///
/// # Errors
///
/// [`Error::InvalidData`](oxideav_core::Error::InvalidData) when a
/// non-empty `extradata` does not unlace into the three §4.2 headers
/// in order.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.extradata.is_empty() {
        Ok(Box::new(VorbisDecoder::new()))
    } else {
        Ok(Box::new(VorbisDecoder::with_extradata(&params.extradata)?))
    }
}

/// The [`SampleFormat`] the decoder emits: planar 32-bit float, one
/// plane per bitstream channel.
pub const OUTPUT_SAMPLE_FORMAT: SampleFormat = SampleFormat::F32P;

//! Whole-stream entry points: `PCM → .ogg` and `.ogg → PCM`.
//!
//! [`encode_pcm_to_ogg`] is the crate's integrated encoder: it composes
//! the §4.3.8-inverse framing splitter, the §4.3.7 forward MDCT, the
//! psychoacoustic masking model, the §7.2 floor-1 design/fit stack, the
//! §8.6 perceptually weighted rate-distortion residue planner, the
//! §4.2/§4.3 packet writers and the §A.2 Ogg encapsulation into one
//! call producing a complete, playable Ogg/Vorbis physical bitstream.
//!
//! [`decode_ogg_to_pcm`] is the inverse convenience: RFC 3533
//! de-framing, the three §4.2 header parses, the §4.3 streaming decode,
//! and the §A.2 granule-position end-trim.
//!
//! # Encoder geometry
//!
//! The stream uses a single blocksize `n` (`blocksize_0 == blocksize_1`,
//! one `blockflag = false` mode). With the §4.3.8 lapping rule each
//! audio packet after the first finishes `n/2` PCM samples, so packet
//! `f` carries the absolute granule position `f · n/2`; the final
//! packet's granule is lowered to the true sample count — the §A.2
//! end-trim that lets a stream end on a non-block-aligned length. The
//! encoder pre-rolls `n/2` zeros (the priming frame's left half lands
//! on pre-stream silence) and zero-pads the tail so the last emitted
//! packet covers the final input sample.
//!
//! # Rate control
//!
//! One [`crate::quality::EncoderTuning`] scalar drives every lever:
//! the psy threshold margin, the floor-1 post budget, and the residue
//! Lagrangian pricing bits in noise-to-mask units (the floor rides
//! `max(signal, masking threshold)` and the residue chooser charges
//! `weights · error²  + λ · bits` per §8.6 partition).

use crate::audio::AudioDecoderState;
use crate::codebook::{VorbisCodebook, VqLookup};
use crate::encoder::{
    write_audio_packet, write_comment_header, write_identification_header, write_setup_header,
    AudioChannelFloor, Floor1Packet, ResidueVectorPlan, WriteAudioPacketError, WriteError,
};
use crate::floor1::Floor1Decoder;
use crate::floor1_encode::{plan_floor1_y, Floor1EncodeError};
use crate::floor1_envelope::{plan_floor1_envelope, Floor1EnvelopeError};
use crate::floor1_layout::{design_floor1_header, Floor1LayoutError};
use crate::framing::{FrameSplitter, FramingError};
use crate::identification::{parse_identification_header, VorbisIdentificationHeader};
use crate::mdct::{mdct_naive_vec, MdctError};
use crate::packet::AudioPacketHeader;
use crate::packet_kind::{classify_packet, ClassifyError, PacketKind};
use crate::psy::{
    plan_psy_floor_envelope, residue_partition_weights, MaskingAnalysis, PsyConfig, PsyError,
    TemporalMasking, TemporalMaskingConfig,
};
use crate::quality::{EncoderTuning, QualityError};
use crate::residue_encode::{plan_vector_residue_rd_weighted, ResidueEncodeError};
use crate::setup::{
    parse_setup_header, Floor1Class, FloorHeader, FloorKind, MappingHeader, MappingSubmap,
    ModeHeader, ResidueHeader, VorbisSetupHeader,
};
use crate::streaming::{StreamingDecoder, StreamingError, StreamingFrame};
use crate::synthesis::{vorbis_window, WindowError};
use crate::VorbisCommentHeader;
use oxideav_core::{CodecId, CodecParameters, StreamInfo, TimeBase};
use oxideav_ogg::page::Page;

/// §8.6.1 residue partition size the integrated encoder uses.
const PARTITION_SIZE: u32 = 16;

/// Configuration for [`encode_pcm_to_ogg`].
#[derive(Debug, Clone, PartialEq)]
pub struct StreamEncoderConfig {
    /// PCM sample rate in Hz (§4.2.2 `audio_sample_rate`).
    pub sample_rate: u32,
    /// Channel count; every channel is carried uncoupled (its own
    /// floor + residue vector under one submap).
    pub channels: u8,
    /// Quality knob `q ∈ [0, 1]` — expanded through
    /// [`EncoderTuning::from_quality`].
    pub quality: f32,
    /// The single analysis/synthesis blocksize `n` (a power of two in
    /// `64..=8192`, §4.2.2). Both identification-header blocksizes are
    /// set to it.
    pub blocksize: usize,
    /// Ogg logical-bitstream serial number.
    pub serial: u32,
    /// Closed-loop residue-codebook training iterations
    /// ([`crate::book_design::train_residue_books_rd_ladder`]): the
    /// generic seed ladders are retrained on the stream's own residue
    /// targets — codeword lengths from usage and reconstruction
    /// values at the target centroids, re-snapped to the
    /// §9.2.2-packable grid — before the packets are planned. `0`
    /// disables training (the fixed seed ladders are used verbatim).
    pub training_iterations: usize,
}

impl StreamEncoderConfig {
    /// A nominal configuration: `quality = 0.7`, `blocksize = 1024`,
    /// 4 codebook-training iterations, serial `0x6F78_7662` (arbitrary
    /// fixed default).
    #[must_use]
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        StreamEncoderConfig {
            sample_rate,
            channels,
            quality: 0.7,
            blocksize: 1024,
            serial: 0x6F78_7662,
            training_iterations: 4,
        }
    }
}

/// Errors from the whole-stream encoder / decoder entry points.
#[derive(Debug, Clone, PartialEq)]
pub enum OggFileError {
    /// `channels` was zero or disagreed with the PCM row count.
    BadChannelCount {
        /// Configured channel count.
        channels: u8,
        /// PCM rows supplied.
        rows: usize,
    },
    /// The PCM rows were empty or of unequal lengths.
    BadPcmShape,
    /// `sample_rate` was zero.
    ZeroSampleRate,
    /// `blocksize` was not a power of two in `64..=8192` (§4.2.2).
    BadBlocksize(usize),
    /// The quality knob was rejected.
    Quality(QualityError),
    /// The §1.3.2 / §4.3.1 window builder failed.
    Window(WindowError),
    /// The §4.3.8-inverse framing splitter failed.
    Framing(FramingError),
    /// The §4.3.7 forward MDCT failed.
    Mdct(MdctError),
    /// The psychoacoustic model failed.
    Psy(PsyError),
    /// Floor-1 header design failed.
    FloorDesign(Floor1LayoutError),
    /// Floor-1 envelope fitting failed.
    FloorFit(Floor1EnvelopeError),
    /// Floor-1 amplitude wrapping failed.
    FloorWrap(Floor1EncodeError),
    /// The floor decoder used for curve rendering failed to build.
    FloorRender(crate::floor1::Floor1Error),
    /// The residue planner failed.
    Residue(ResidueEncodeError),
    /// Closed-loop codebook training failed.
    Training(crate::book_design::BookDesignError),
    /// A header writer failed.
    Write(WriteError),
    /// The audio-packet writer failed.
    WritePacket(WriteAudioPacketError),
    /// The §A.2 muxer refused the packet sequence.
    Mux(MuxError),
    /// RFC 3533 de-framing failed (decode direction) — the rendered
    /// message from the `oxideav-ogg` page parser.
    Ogg(String),
    /// A decode-direction header packet failed to parse.
    Header(String),
    /// The §4.3 streaming decode failed (decode direction).
    Streaming(StreamingError),
    /// The stream ended before the three §4.2 header packets.
    MissingHeaders {
        /// Packets found.
        packets: usize,
    },
}

impl core::fmt::Display for OggFileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OggFileError::BadChannelCount { channels, rows } => write!(
                f,
                "ogg encode: {channels} channels configured but {rows} PCM rows supplied"
            ),
            OggFileError::BadPcmShape => {
                write!(f, "ogg encode: PCM rows empty or of unequal lengths")
            }
            OggFileError::ZeroSampleRate => write!(f, "ogg encode: sample rate is zero"),
            OggFileError::BadBlocksize(n) => write!(
                f,
                "ogg encode: blocksize {n} is not a power of two in 64..=8192"
            ),
            OggFileError::Quality(e) => write!(f, "ogg encode: {e}"),
            OggFileError::Window(e) => write!(f, "ogg encode: {e}"),
            OggFileError::Framing(e) => write!(f, "ogg encode: {e}"),
            OggFileError::Mdct(e) => write!(f, "ogg encode: {e}"),
            OggFileError::Psy(e) => write!(f, "ogg encode: {e}"),
            OggFileError::FloorDesign(e) => write!(f, "ogg encode: {e}"),
            OggFileError::FloorFit(e) => write!(f, "ogg encode: {e}"),
            OggFileError::FloorWrap(e) => write!(f, "ogg encode: {e}"),
            OggFileError::FloorRender(e) => write!(f, "ogg encode: {e}"),
            OggFileError::Residue(e) => write!(f, "ogg encode: {e}"),
            OggFileError::Training(e) => write!(f, "ogg encode: {e}"),
            OggFileError::Write(e) => write!(f, "ogg encode: {e}"),
            OggFileError::WritePacket(e) => write!(f, "ogg encode: {e}"),
            OggFileError::Mux(e) => write!(f, "ogg encode: {e}"),
            OggFileError::Ogg(e) => write!(f, "ogg decode: {e}"),
            OggFileError::Header(e) => write!(f, "ogg decode: header parse: {e}"),
            OggFileError::Streaming(e) => write!(f, "ogg decode: {e}"),
            OggFileError::MissingHeaders { packets } => write!(
                f,
                "ogg decode: stream holds {packets} packets, need the 3 headers"
            ),
        }
    }
}

impl std::error::Error for OggFileError {}

macro_rules! from_err {
    ($src:ty => $variant:ident) => {
        impl From<$src> for OggFileError {
            fn from(value: $src) -> Self {
                OggFileError::$variant(value)
            }
        }
    };
}
from_err!(QualityError => Quality);
from_err!(WindowError => Window);
from_err!(FramingError => Framing);
from_err!(MdctError => Mdct);
from_err!(PsyError => Psy);
from_err!(Floor1LayoutError => FloorDesign);
from_err!(Floor1EnvelopeError => FloorFit);
from_err!(Floor1EncodeError => FloorWrap);
from_err!(crate::floor1::Floor1Error => FloorRender);
from_err!(ResidueEncodeError => Residue);
from_err!(crate::book_design::BookDesignError => Training);
from_err!(WriteError => Write);
from_err!(WriteAudioPacketError => WritePacket);
from_err!(MuxError => Mux);
from_err!(StreamingError => Streaming);

// ───────────────────── §A.2 Ogg encapsulation ─────────────────────
//
// The codec-agnostic RFC 3533 page transport (framing, lacing, CRC,
// pagination) is `oxideav-ogg`'s job. What stays here is the Vorbis
// mapping of §A ("Embedding Vorbis into an Ogg stream"): the
// three-header ordering rule, the per-packet end-PCM-sample granule
// semantics (including the §A.2 end-trim), and the codec-private
// header packaging the container layer carries.

/// Soft page-size target for audio pages, in body bytes. RFC 3533 §6
/// describes pages as "usually 4-8 kB"; [`mux_vorbis_stream`] signals
/// a page boundary to the container layer once the pending body
/// reaches this size (a packet that overshoots it still lands whole
/// unless the 255-segment table forces a split). Small pages keep the
/// per-page granule positions dense enough for third-party decoders
/// to resolve per-packet timestamps (and the §A.2 end-trim) without
/// walking a whole-stream page.
const AUDIO_PAGE_TARGET_BYTES: usize = 4096;

/// §A.2 packet-sequencing errors from [`mux_vorbis_stream`].
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
    /// The packet failed §4.2.1 / §4.3.1 classification.
    Classify(ClassifyError),
    /// An audio packet's granule position went backwards.
    NonMonotoneGranule {
        /// The previous packet's granule position.
        prev: u64,
        /// The offending packet's granule position.
        got: u64,
    },
    /// The `oxideav-ogg` container layer refused the stream — the
    /// rendered message.
    Container(String),
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
            MuxError::Classify(e) => write!(f, "ogg/vorbis mux: {e}"),
            MuxError::NonMonotoneGranule { prev, got } => write!(
                f,
                "ogg/vorbis mux: granule position {got} < previous {prev}"
            ),
            MuxError::Container(e) => write!(f, "ogg/vorbis mux: container: {e}"),
        }
    }
}

impl std::error::Error for MuxError {}

impl From<ClassifyError> for MuxError {
    fn from(value: ClassifyError) -> Self {
        MuxError::Classify(value)
    }
}

/// Package the three §4.2 header packets as the Xiph-laced
/// codec-private blob container layers carry for Vorbis (Matroska
/// `CodecPrivate`, `oxideav-ogg` `StreamInfo::params.extradata`): one
/// byte `packet_count - 1` (= 2), then the 255-terminated lacing sizes
/// of the first two packets, then the three packets back to back (the
/// last packet's size is implicit in the blob length).
#[must_use]
pub fn lace_vorbis_headers(identification: &[u8], comment: &[u8], setup: &[u8]) -> Vec<u8> {
    fn push_lacing(out: &mut Vec<u8>, mut len: usize) {
        while len >= 255 {
            out.push(255);
            len -= 255;
        }
        out.push(len as u8);
    }
    let mut blob = Vec::with_capacity(
        3 + identification.len() / 255
            + comment.len() / 255
            + identification.len()
            + comment.len()
            + setup.len(),
    );
    blob.push(2);
    push_lacing(&mut blob, identification.len());
    push_lacing(&mut blob, comment.len());
    blob.extend_from_slice(identification);
    blob.extend_from_slice(comment);
    blob.extend_from_slice(setup);
    blob
}

/// A `Write + Seek + Send` sink over a shared byte buffer, so the
/// bytes survive handing ownership of the writer to the container
/// muxer.
#[derive(Clone, Default)]
struct SharedBuf(std::sync::Arc<std::sync::Mutex<std::io::Cursor<Vec<u8>>>>);

impl SharedBuf {
    fn take(&self) -> Vec<u8> {
        std::mem::take(self.0.lock().expect("shared buffer lock").get_mut())
    }
}

impl std::io::Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("shared buffer lock").write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl std::io::Seek for SharedBuf {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.0.lock().expect("shared buffer lock").seek(pos)
    }
}

/// One-call §A.2 encapsulation of a complete packet stream:
/// `(identification, comment, setup)` headers plus `(packet,
/// granulepos)` audio packets, under the given logical-bitstream
/// serial number. The header ordering is verified with the §4.2.1
/// packet-kind classifier and the granule positions must be
/// non-decreasing; pagination, lacing and CRC are delegated to
/// `oxideav-ogg`.
///
/// Each audio packet's granule position is the absolute end PCM
/// sample position (per channel) of the stream after the packet's
/// samples are returned by decode. The final packet's granule may
/// deliberately understate the naturally decodable sample count —
/// §A.2's end-trim rule for ending a stream on a non-block-aligned
/// length.
///
/// # Errors
///
/// [`MuxError`] on a §A.2 sequencing violation or a container-layer
/// failure.
pub fn mux_vorbis_stream(
    serial: u32,
    identification: &[u8],
    comment: &[u8],
    setup: &[u8],
    audio: &[(Vec<u8>, u64)],
) -> Result<Vec<u8>, MuxError> {
    // §A.2 packet-sequence validation — codec semantics, kept here.
    for (packet, expected) in [
        (identification, PacketKind::Identification),
        (comment, PacketKind::Comment),
        (setup, PacketKind::Setup),
    ] {
        let got = classify_packet(packet)?;
        if got != expected {
            return Err(MuxError::HeaderOrder { expected, got });
        }
    }
    let mut last_granule = 0u64;
    for (_, granule) in audio {
        if *granule < last_granule {
            return Err(MuxError::NonMonotoneGranule {
                prev: last_granule,
                got: *granule,
            });
        }
        last_granule = *granule;
    }

    // Stream description for the container layer. The `StreamInfo`
    // index doubles as the muxer's on-wire serial; the sample rate
    // (when the id header parses) gives the granule its 1/rate time
    // base.
    let mut params = CodecParameters::audio(CodecId::new("vorbis"));
    params.extradata = lace_vorbis_headers(identification, comment, setup);
    let time_base = match parse_identification_header(identification) {
        Ok(id) => {
            params.sample_rate = Some(id.audio_sample_rate);
            params.channels = Some(u16::from(id.audio_channels));
            TimeBase::new(1, i64::from(id.audio_sample_rate.max(1)))
        }
        Err(_) => TimeBase::new(1, 1),
    };
    let stream = StreamInfo {
        index: serial,
        time_base,
        duration: audio.last().map(|(_, g)| *g as i64),
        start_time: Some(0),
        params,
    };

    let sink = SharedBuf::default();
    let container = |e: oxideav_core::Error| MuxError::Container(e.to_string());
    let mut muxer =
        oxideav_ogg::mux::open_concrete(Box::new(sink.clone()), std::slice::from_ref(&stream))
            .map_err(container)?;
    use oxideav_core::Muxer as _;
    muxer.write_header().map_err(container)?;
    let mut pending_body = 0usize;
    for (i, (packet, granule)) in audio.iter().enumerate() {
        pending_body += packet.len();
        // Page-boundary policy: flush at the soft size target, and
        // always break before the final packet so the last page
        // carries only the end-trim packet — the penultimate page then
        // ends on an exact blocksize-walk granule anchor, keeping the
        // §A.2 final-granule trim locally resolvable for third-party
        // decoders.
        let boundary = pending_body >= AUDIO_PAGE_TARGET_BYTES || i + 2 == audio.len();
        if boundary {
            pending_body = 0;
        }
        let mut pkt =
            oxideav_core::Packet::new(serial, time_base, packet.clone()).with_pts(*granule as i64);
        pkt.flags.unit_boundary = boundary;
        muxer.write_packet(&pkt).map_err(container)?;
    }
    muxer.write_trailer().map_err(container)?;
    drop(muxer);
    Ok(sink.take())
}

/// Parse a physical Ogg stream into its page sequence (CRC-verified by
/// the `oxideav-ogg` page parser).
fn parse_all_pages(data: &[u8]) -> Result<Vec<Page>, OggFileError> {
    let mut pages = Vec::new();
    let mut off = 0usize;
    while off < data.len() {
        let (page, used) =
            Page::parse(&data[off..]).map_err(|e| OggFileError::Ogg(e.to_string()))?;
        off += used;
        pages.push(page);
    }
    Ok(pages)
}

/// De-frame a complete single-logical-stream Ogg physical bitstream
/// into its packet sequence: CRC-verified page parse plus lacing-model
/// packet reassembly (packets spanning pages are concatenated across
/// the continuation boundary).
///
/// # Errors
///
/// [`OggFileError::Ogg`] when a page fails to parse or CRC-verify.
pub fn ogg_packets(data: &[u8]) -> Result<Vec<Vec<u8>>, OggFileError> {
    Ok(assemble_packets(&parse_all_pages(data)?))
}

/// Lacing-model packet reassembly over parsed pages.
fn assemble_packets(pages: &[Page]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut pending: Vec<u8> = Vec::new();
    for page in pages {
        for seg in page.packet_segments() {
            pending.extend_from_slice(&page.data[seg.data.clone()]);
            if seg.terminated {
                packets.push(std::mem::take(&mut pending));
            }
        }
    }
    packets
}

/// A signed 1-D lattice value book: `2^length` entries on the uniform
/// grid `[-half·step, (half−1)·step]`, all codewords `length` bits.
/// §3.2.1 lookup type 1 (for one dimension, `lookup1_values ==
/// entries`, so the lattice table is indexed directly) — the widely
/// interoperable lookup type real-world streams carry.
fn signed_value_book(length: u8, step: f32) -> VorbisCodebook {
    let entries: u32 = 1u32 << length;
    let half = entries / 2;
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::Lattice {
            minimum_value: -(half as f32) * step,
            delta_value: step,
            value_bits: 8,
            sequence_p: false,
            multiplicands: (0..entries).collect(),
        },
    }
}

/// A scalar (lookup-type-0) book with uniform codeword lengths.
fn scalar_book(entries: u32, length: u8) -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
}

/// The floor-1 class catalogue the header designer may tile with:
/// 1-, 2- and 4-dimension classes, no subclasses, all posts on book 0.
fn floor_class_catalogue() -> Vec<Floor1Class> {
    [1u8, 2, 4]
        .iter()
        .map(|&d| Floor1Class {
            dimensions: d,
            subclasses: 0,
            masterbook: None,
            subclass_books: vec![Some(0)],
        })
        .collect()
}

/// The stream's setup header: 4 codebooks (floor posts, residue
/// classwords, coarse + fine value ladders), one floor, one two-class
/// residue (class 0 = silence, class 1 = two-stage cascade), one
/// mapping with every channel uncoupled under a single submap, one
/// `blockflag = false` mode.
fn build_setup(
    floor_header: crate::setup::Floor1Header,
    coarse: VorbisCodebook,
    fine: VorbisCodebook,
    half_n: u32,
) -> VorbisSetupHeader {
    let floor = FloorHeader {
        floor_type: 1,
        kind: FloorKind::Type1(floor_header),
    };
    let mut stages: [Option<u8>; 8] = Default::default();
    stages[0] = Some(2);
    stages[1] = Some(3);
    let residue = ResidueHeader {
        residue_type: 1,
        residue_begin: 0,
        residue_end: half_n,
        partition_size: PARTITION_SIZE,
        classifications: 2,
        classbook: 1,
        cascade: vec![0, 0b11],
        books: vec![Default::default(), stages],
    };
    VorbisSetupHeader {
        codebooks: vec![scalar_book(256, 8), scalar_book(2, 1), coarse, fine],
        time_placeholders: vec![0],
        floors: vec![floor],
        residues: vec![residue],
        mappings: vec![MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: Vec::new(),
            // §4.2.4: the mux table is only present when submaps > 1;
            // with one submap every channel implicitly maps to it.
            mux: Vec::new(),
            submap_configs: vec![MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        }],
        modes: vec![ModeHeader {
            blockflag: false,
            windowtype: 0,
            transformtype: 0,
            mapping: 0,
        }],
        framing_flag: true,
    }
}

/// The packet-level product of [`encode_pcm_to_packets`]: the three
/// §4.2 header packets plus the §4.3 audio packets with their absolute
/// §A.2 granule positions — everything a container muxer needs.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedVorbisStream {
    /// §4.2.2 identification-header packet.
    pub identification: Vec<u8>,
    /// §5 comment-header packet.
    pub comment: Vec<u8>,
    /// §4.2.4 setup-header packet.
    pub setup: Vec<u8>,
    /// §4.3 audio packets, each with the end-PCM-sample granule
    /// position of the stream after it (packet `f` finishes `f · n/2`
    /// samples; the final packet's granule is the exact input length —
    /// the §A.2 end-trim).
    pub audio: Vec<(Vec<u8>, u64)>,
    /// The single blocksize `n` the stream uses.
    pub blocksize: usize,
}

/// Encode per-channel PCM rows into a complete Ogg/Vorbis physical
/// bitstream (§A.2 encapsulation of the three §4.2 headers plus the
/// §4.3 audio packets). See the module docs for the pipeline.
///
/// `pcm` holds one row per channel, all of equal non-zero length, in
/// the nominal `[-1, 1]` range.
///
/// # Errors
///
/// Shape/configuration violations and any stage failure — see
/// [`OggFileError`].
pub fn encode_pcm_to_ogg(
    pcm: &[Vec<f32>],
    config: &StreamEncoderConfig,
) -> Result<Vec<u8>, OggFileError> {
    let stream = encode_pcm_to_packets(pcm, config)?;
    Ok(mux_vorbis_stream(
        config.serial,
        &stream.identification,
        &stream.comment,
        &stream.setup,
        &stream.audio,
    )?)
}

/// The packet-level encoder under [`encode_pcm_to_ogg`]: the full
/// analysis/psy/floor/residue pipeline, stopping *before* the Ogg
/// layer. Container-agnostic consumers (the [`oxideav_core::Encoder`]
/// implementation, external muxers) use this form.
///
/// # Errors
///
/// As [`encode_pcm_to_ogg`].
pub fn encode_pcm_to_packets(
    pcm: &[Vec<f32>],
    config: &StreamEncoderConfig,
) -> Result<EncodedVorbisStream, OggFileError> {
    // ---- validation ----
    if config.channels == 0 || pcm.len() != config.channels as usize {
        return Err(OggFileError::BadChannelCount {
            channels: config.channels,
            rows: pcm.len(),
        });
    }
    let samples = pcm[0].len();
    if samples == 0 || pcm.iter().any(|row| row.len() != samples) {
        return Err(OggFileError::BadPcmShape);
    }
    if config.sample_rate == 0 {
        return Err(OggFileError::ZeroSampleRate);
    }
    let n = config.blocksize;
    if !n.is_power_of_two() || !(64..=8192).contains(&n) {
        return Err(OggFileError::BadBlocksize(n));
    }
    let half_n = n / 2;
    let ch = config.channels as usize;
    let tuning = EncoderTuning::from_quality(config.quality)?;

    // ---- §4.3.8-inverse framing + §4.3.7 forward MDCT ----
    // P frames cover the stream: the priming frame plus one per n/2
    // finished samples. The forward transform is scaled by 4/n so the
    // decode-side bare-kernel IMDCT (scale 1.0) + §4.3.1 window +
    // §4.3.8 overlap-add reconstruct unity PCM: the bare kernels
    // compose as mdct(imdct(X)) == (n/2)·X, and the windowed TDAC
    // overlap-add contributes the remaining factor of ½ (each output
    // sample is reconstructed from its two half-overlapped frames
    // under the w² + w'² = 1 window identity).
    let frames = samples.div_ceil(half_n) + 1;
    let window = vorbis_window(n, n, false, false, false)?;
    let mdct_scale = 4.0 / n as f32;
    let mut spectra: Vec<Vec<Vec<f32>>> = Vec::with_capacity(frames); // [frame][channel][bin]
    {
        let mut splitters: Vec<FrameSplitter> = (0..ch).map(|_| FrameSplitter::new()).collect();
        let pad = frames * half_n - samples;
        for (c, splitter) in splitters.iter_mut().enumerate() {
            splitter.push_pcm(&vec![0.0f32; half_n]); // pre-stream silence
            splitter.push_pcm(&pcm[c]);
            splitter.push_pcm(&vec![0.0f32; pad]);
        }
        for _ in 0..frames {
            let mut per_ch = Vec::with_capacity(ch);
            for splitter in splitters.iter_mut() {
                // take_frame already applies the §4.3.1 analysis
                // window, so the bare forward kernel follows.
                let block = splitter.take_frame(n, &window)?;
                let x = mdct_naive_vec(&block, mdct_scale)?;
                per_ch.push(x);
            }
            spectra.push(per_ch);
        }
    }

    // ---- psychoacoustics + floor-1 header design ----
    // Per channel, the thresholds run through the temporal pipeline:
    // post-masking decay across frames plus the one-frame-lookahead
    // pre-masking lift (the encoder is whole-stream, so lookahead is
    // free). On steady-state content this is exactly the per-frame
    // model.
    let psy_config = PsyConfig {
        threshold_offset_db: tuning.threshold_offset_db,
        ..PsyConfig::new(config.sample_rate)
    };
    let mut maskings: Vec<Vec<MaskingAnalysis>> = vec![Vec::with_capacity(ch); frames];
    for c in 0..ch {
        let mut temporal = TemporalMasking::new(&TemporalMaskingConfig::new(half_n), &psy_config)?;
        let mut emitted = 0usize;
        for per_ch in &spectra {
            if let Some(analysis) = temporal.push_frame(&per_ch[c], &psy_config)? {
                maskings[emitted].push(analysis);
                emitted += 1;
            }
        }
        if let Some(analysis) = temporal.finish() {
            maskings[emitted].push(analysis);
        }
    }
    let mut envelopes = Vec::with_capacity(frames);
    let mut envelope_max = vec![f32::MIN_POSITIVE; half_n];
    for (per_ch, m_row) in spectra.iter().zip(&maskings) {
        let mut e_row = Vec::with_capacity(ch);
        for (x, masking) in per_ch.iter().zip(m_row) {
            let envelope = plan_psy_floor_envelope(x, masking, tuning.floor_smooth_radius)?;
            for (acc, &v) in envelope_max.iter_mut().zip(&envelope) {
                *acc = acc.max(v);
            }
            e_row.push(envelope);
        }
        envelopes.push(e_row);
    }
    let classes = floor_class_catalogue();
    let floor_header =
        design_floor1_header(&envelope_max, tuning.floor_post_budget, 0.0, 1, &classes)?;
    let floor_book = scalar_book(256, 8);
    let floor_decoder = Floor1Decoder::new(&floor_header, std::slice::from_ref(&floor_book))?;

    // ---- per-frame floor fit + residue targets + NMR weights ----
    let mut floor_ys: Vec<Vec<Vec<u32>>> = Vec::with_capacity(frames);
    let mut targets: Vec<Vec<Vec<f32>>> = Vec::with_capacity(frames);
    let mut weights: Vec<Vec<Vec<f64>>> = Vec::with_capacity(frames);
    let mut max_abs = 0.0f32;
    for f in 0..frames {
        let mut y_row = Vec::with_capacity(ch);
        let mut t_row = Vec::with_capacity(ch);
        let mut w_row = Vec::with_capacity(ch);
        for c in 0..ch {
            let posts = plan_floor1_envelope(&envelopes[f][c], &floor_header)?;
            let floor1_y = plan_floor1_y(&posts, &floor_header)?;
            let rendered = floor_decoder.render_curve(&floor1_y, half_n);
            let target: Vec<f32> = spectra[f][c]
                .iter()
                .zip(&rendered)
                .map(|(&xv, &fv)| xv / fv)
                .collect();
            for &t in &target {
                max_abs = max_abs.max(t.abs());
            }
            let w =
                residue_partition_weights(&rendered, &maskings[f][c], 0, half_n, PARTITION_SIZE)?;
            y_row.push(floor1_y);
            t_row.push(target);
            w_row.push(w);
        }
        floor_ys.push(y_row);
        targets.push(t_row);
        weights.push(w_row);
    }
    if max_abs <= 0.0 {
        max_abs = 1.0; // all-silent input: any positive ladder scale works
    }

    // ---- setup: value ladders sized to the residue range ----
    // The ladder steps must be exactly §9.2.2-packable (the codebook
    // header carries them as 21-bit-mantissa floats); the book minimum
    // is −32·step, which shares the step's mantissa (× 2⁵) and is
    // therefore packable whenever the step is.
    let coarse = signed_value_book(6, crate::book_design::pack_nearest(max_abs / 24.0));
    let fine = signed_value_book(6, crate::book_design::pack_nearest(max_abs / 192.0));
    let mut setup = build_setup(floor_header.clone(), coarse, fine, half_n as u32);

    // ---- optional closed-loop codebook training ----
    // The generic seed ladders are retrained on the stream's own
    // residue targets (codeword lengths from usage, reconstruction
    // values at the observed centroids, re-snapped §9.2.2-packable);
    // the trained table replaces the seeds in the setup header and
    // the weighted per-frame planning below runs under it.
    if config.training_iterations > 0 {
        let residuals: Vec<Vec<f32>> = targets.iter().flat_map(|row| row.iter().cloned()).collect();
        let outcome = crate::book_design::train_residue_books_rd_ladder(
            &residuals,
            &setup.residues[0],
            &setup.codebooks,
            tuning.lambda,
            config.training_iterations,
        )?;
        setup.codebooks = outcome.codebooks;
        // The trainer plans *unweighted*; the final packets below are
        // planned with the NMR weights, which can legitimately select
        // a class the unweighted pass never used. Keep the flat seed
        // classbook (both classwords stay encodable at 1 bit each —
        // there is nothing to win on a 2-entry book) and take only
        // the trained value ladders.
        setup.codebooks[1] = scalar_book(2, 1);
    }
    let coarse = setup.codebooks[2].clone();
    let fine = setup.codebooks[3].clone();

    // ---- the three §4.2 header packets ----
    let id_packet = write_identification_header(&VorbisIdentificationHeader {
        vorbis_version: 0,
        audio_channels: config.channels,
        audio_sample_rate: config.sample_rate,
        bitrate_maximum: 0,
        bitrate_nominal: 0,
        bitrate_minimum: 0,
        blocksize_0: n as u16,
        blocksize_1: n as u16,
    })?;
    let comment_packet = write_comment_header(&VorbisCommentHeader {
        vendor: "oxideav-vorbis clean-room encoder".into(),
        comments: vec!["ENCODER=oxideav-vorbis".into()],
    })?;
    let setup_packet = write_setup_header(&setup, config.channels)?;

    // ---- §4.3 audio packets + §A.2 encapsulation ----
    let empty_row: [Option<&VorbisCodebook>; 8] = [None; 8];
    let mut cascade_row: [Option<&VorbisCodebook>; 8] = [None; 8];
    cascade_row[0] = Some(&coarse);
    cascade_row[1] = Some(&fine);
    let value_books = [empty_row, cascade_row];

    let mut audio_packets: Vec<(Vec<u8>, u64)> = Vec::with_capacity(frames);
    let packet_header = AudioPacketHeader {
        mode_number: 0,
        blockflag: false,
        n,
        previous_window_flag: false,
        next_window_flag: false,
    };
    for f in 0..frames {
        let mut floors = Vec::with_capacity(ch);
        let mut plans = Vec::with_capacity(ch);
        for c in 0..ch {
            let scored = plan_vector_residue_rd_weighted(
                &targets[f][c],
                &value_books,
                1,
                PARTITION_SIZE,
                tuning.lambda,
                &weights[f][c],
            )?;
            floors.push(AudioChannelFloor::Type1(Floor1Packet {
                nonzero: true,
                floor1_y: floor_ys[f][c].clone(),
                partition_cvals: vec![0u32; floor_header.partition_class_list.len()],
            }));
            plans.push(ResidueVectorPlan {
                classifications: scored.classifications,
                partition_entries: scored.partition_entries,
            });
        }
        let submap_plans = [plans];
        let packet = write_audio_packet(
            &packet_header,
            &setup,
            n,
            n,
            config.channels,
            &floors,
            &submap_plans,
        )?;
        // §4.3.8: packet f finishes f · n/2 samples; the final packet's
        // granule is the true sample count (§A.2 end-trim).
        let natural = (f * half_n) as u64;
        let granule = if f + 1 == frames {
            samples as u64
        } else {
            natural
        };
        audio_packets.push((packet, granule));
    }
    Ok(EncodedVorbisStream {
        identification: id_packet,
        comment: comment_packet,
        setup: setup_packet,
        audio: audio_packets,
        blocksize: n,
    })
}

/// A decoded Ogg/Vorbis stream: per-channel PCM rows (bitstream
/// channel order) plus the stream parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedOggStream {
    /// Per-channel PCM rows, §A.2 end-trimmed to the final page's
    /// granule position.
    pub pcm: Vec<Vec<f32>>,
    /// §4.2.2 `audio_sample_rate`.
    pub sample_rate: u32,
    /// §4.2.2 `audio_channels`.
    pub channels: u8,
}

/// Decode a complete single-logical-stream Ogg/Vorbis physical
/// bitstream to per-channel PCM: RFC 3533 de-framing, the three §4.2
/// header parses, the §4.3 streaming decode, and the §A.2 end-trim to
/// the final page's granule position.
///
/// # Errors
///
/// See [`OggFileError`].
pub fn decode_ogg_to_pcm(data: &[u8]) -> Result<DecodedOggStream, OggFileError> {
    let pages = parse_all_pages(data)?;
    let packets = assemble_packets(&pages);
    if packets.len() < 3 {
        return Err(OggFileError::MissingHeaders {
            packets: packets.len(),
        });
    }
    let id = parse_identification_header(&packets[0])
        .map_err(|e| OggFileError::Header(e.to_string()))?;
    let setup = parse_setup_header(&packets[2], id.audio_channels)
        .map_err(|e| OggFileError::Header(e.to_string()))?;
    let state =
        AudioDecoderState::new(&setup).map_err(|e| OggFileError::Header(format!("{e:?}")))?;
    let ch = id.audio_channels as usize;
    let mut decoder = StreamingDecoder::new(
        id.audio_channels,
        id.blocksize_0 as usize,
        id.blocksize_1 as usize,
        1.0,
    );
    let mut pcm: Vec<Vec<f32>> = vec![Vec::new(); ch];
    for packet in &packets[3..] {
        let mut reader = oxideav_core::bits::BitReaderLsb::new(packet);
        match decoder.push_packet(&mut reader, &setup, &state)? {
            StreamingFrame::Pcm {
                per_channel_pcm, ..
            } => {
                for (row, samples) in pcm.iter_mut().zip(&per_channel_pcm) {
                    row.extend_from_slice(samples);
                }
            }
            StreamingFrame::Primed { .. } => {}
        }
    }
    // §A.2 end-trim: the final page's granule position may declare
    // fewer samples than decode naturally returned.
    let final_granule = pages
        .iter()
        .rev()
        .map(|p| p.granule_position)
        .find(|&g| g >= 0);
    if let Some(g) = final_granule {
        let keep = (g as usize).min(pcm[0].len());
        for row in &mut pcm {
            row.truncate(keep);
        }
    }
    Ok(DecodedOggStream {
        pcm,
        sample_rate: id.audio_sample_rate,
        channels: id.audio_channels,
    })
}

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
//! The stream carries the §4.2.2 blocksize pair `(blocksize_0,
//! blocksize_1)`. When they differ, the encoder runs §4.3.1 **block
//! switching**: a clean-room energy-envelope transient detector
//! ([`crate::blocksize::plan_block_sequence`]) schedules the short
//! block over attacks (confining quantisation noise to avoid
//! pre-echo) and the long block elsewhere; each long packet's
//! `previous_window_flag` / `next_window_flag` mirror its neighbours'
//! blockflags so the §4.3.1 hybrid window edges lap every long↔short
//! transition, and the setup header carries a floor / residue /
//! mapping / mode set **per block size**. With the §4.3.8 lapping
//! rule packet `f` finishes `(n_{f-1} + n_f) / 4` PCM samples, whose
//! running sum is the packet's absolute granule position; the final
//! packet's granule is lowered to the true sample count — the §A.2
//! end-trim that lets a stream end on a non-block-aligned length. The
//! encoder pre-rolls half the first frame of zeros (the priming
//! frame's left half lands on pre-stream silence) and zero-pads the
//! tail so the last emitted packet covers the final input sample.
//! `blocksize_0 == blocksize_1` degenerates to the single-blocksize,
//! single-mode stream.
//!
//! Adjacent channel pairs are §4.3.5 square-polar **coupled** when
//! profitable — see [`StreamEncoderConfig::coupling`].
//!
//! # Rate control
//!
//! One [`crate::quality::EncoderTuning`] scalar drives every lever:
//! the psy threshold margin, the floor-1 post budget, and the residue
//! Lagrangian pricing bits in noise-to-mask units (the floor rides
//! `max(signal, masking threshold)` and the residue chooser charges
//! `weights · error²  + λ · bits` per §8.6 partition).

use crate::audio::AudioDecoderState;
use crate::blocksize::{plan_block_sequence, BlocksizeError};
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
    compute_masking, plan_psy_floor_envelope, residue_partition_weights, MaskingAnalysis,
    PsyConfig, PsyError, TemporalMasking, TemporalMaskingConfig,
};
use crate::quality::{EncoderTuning, QualityError};
use crate::residue_encode::{plan_vector_residue_rd_weighted, ResidueEncodeError};
use crate::setup::{
    parse_setup_header, Floor1Class, FloorHeader, FloorKind, MappingCouplingStep, MappingHeader,
    MappingSubmap, ModeHeader, ResidueHeader, VorbisSetupHeader,
};
use crate::streaming::{StreamingDecoder, StreamingError, StreamingFrame};
use crate::synthesis::{coupling_energy, forward_couple_all, WindowError};
use crate::VorbisCommentHeader;
use oxideav_core::{CodecId, CodecParameters, StreamInfo, TimeBase};
use oxideav_ogg::page::Page;

/// §8.6.1 residue partition size the integrated encoder uses.
const PARTITION_SIZE: u32 = 16;

/// Cap on the stride-subsampled training corpus handed to
/// [`crate::book_design::design_vq_codebook`] (in sub-vectors): the
/// designer's refinement passes are O(vectors × entries), so a long
/// stream trains on a bounded, deterministic sample of its residue.
const VQ_DESIGN_MAX_VECTORS: usize = 6144;

/// Designed coarse-book entries per value-book dimension
/// ([`StreamEncoderConfig::vq_dims`]): the coarse stage spans the raw
/// residue range, so its cell budget grows with the joint space.
const VQ_COARSE_ENTRIES_PER_DIM: u32 = 64;

/// Designed fine-book entries per dimension — the fine stage covers
/// the (much smaller) post-coarse leftover, where extra cells buy
/// reconstruction accuracy for the top of the quality range.
const VQ_FINE_ENTRIES_PER_DIM: u32 = 128;

/// Ceiling on either designed book's entry count (quantisation scans
/// every used entry per §8.6.2 read).
const VQ_MAX_ENTRIES: u32 = 256;

/// Codeword-length cap handed to the VQ designer's occupancy-optimal
/// length assignment (well under the §3.2.1 hard 32-bit limit; a
/// longer codeword than this prices an entry out of use anyway).
const VQ_DESIGN_MAX_CODEWORD_LEN: u8 = 24;

/// §8.6.1 residue classification count (class 0 = silence, class 1 =
/// the full two-stage cascade).
const RESIDUE_CLASSES: u32 = 2;

/// Partitions per §8.6.2 classword (the classbook's dimensions):
/// grouping lets the trained classword lengths price a common run —
/// e.g. four consecutive silent partitions — at a couple of bits
/// total instead of one codeword per partition.
const CLASS_GROUP_DIMS: u16 = 4;

/// The §4.3.5 coupling gate: a candidate channel pair is coupled when
/// its whole-stream square-polar angle energy is at most this fraction
/// of its magnitude energy ([`CouplingEnergy::angle_ratio`] semantics,
/// accumulated over every frame's residue targets). A strongly
/// correlated stereo pair sits far below this (the `L − R` angle
/// residue quantises toward zero); an independent or anti-correlated
/// pair sits near or above `1.0`, where coupling would only move
/// energy around. `0.5` keeps the gate on the clearly-profitable side.
///
/// [`CouplingEnergy::angle_ratio`]: crate::synthesis::CouplingEnergy::angle_ratio
const COUPLING_MAX_ANGLE_RATIO: f64 = 0.5;

/// Sub-frame count the §4.3.1 transient detector splits its lookahead
/// region into ([`crate::blocksize::detect_transient`]): sixteen
/// sub-frames over the `long_n` lookahead resolve an attack finely
/// enough that a decaying hit over a tone bed still concentrates in
/// one or two sub-frames instead of diluting into the bed energy.
const TRANSIENT_SUBFRAMES: usize = 16;

/// Peak-to-mean energy-concentration ratio above which the lookahead
/// region is called a transient and the encoder schedules the short
/// block. A flat region scores `≈ 1`; a tonal region whose lookahead
/// covers only a fraction of a low-frequency cycle stays under `≈
/// 2.5`; a genuine attack over a tone bed clears `4`–`8`. `3.0` sits
/// between the two regimes.
const TRANSIENT_PEAK_TO_MEAN: f64 = 3.0;

/// Energy-rise factor of the §4.3.1 schedule's second transient
/// criterion: the lookahead's peak sub-frame energy against the
/// previous decision region's mean sub-frame energy. Catches a
/// sustained loudness step (a noise burst over a tone bed) that is
/// flat *within* the window — invisible to the concentration ratio —
/// but whose onset a long block would smear pre-echo across. `4.0`
/// (+6 dB in energy over one lookahead) is well above ordinary
/// musical envelope motion.
const TRANSIENT_ENERGY_RISE: f64 = 4.0;

/// Configuration for [`encode_pcm_to_ogg`].
#[derive(Debug, Clone, PartialEq)]
pub struct StreamEncoderConfig {
    /// PCM sample rate in Hz (§4.2.2 `audio_sample_rate`).
    pub sample_rate: u32,
    /// Channel count. Each channel carries its own floor + residue
    /// vector under one submap; adjacent channel pairs are §4.3.5
    /// square-polar coupled when [`Self::coupling`] is set and the
    /// energy gate finds the pair profitable.
    pub channels: u8,
    /// Offer §4.3.5 square-polar channel coupling on adjacent channel
    /// pairs `(0, 1)`, `(2, 3)`, …. Each candidate pair is gated on
    /// the whole stream's coupling-energy split
    /// ([`crate::synthesis::coupling_energy`] accumulated over every
    /// frame's residue targets): only pairs whose angle/magnitude
    /// energy ratio stays under the profitability threshold are
    /// actually coupled (recorded as mapping coupling steps and
    /// forward-coupled before residue planning). `false` carries every
    /// channel uncoupled.
    pub coupling: bool,
    /// Quality knob `q ∈ [0, 1]` — expanded through
    /// [`EncoderTuning::from_quality`].
    pub quality: f32,
    /// The **long** blocksize `blocksize_1` (a power of two in
    /// `64..=8192`, §4.2.2) — the analysis/synthesis size steady
    /// content uses.
    pub blocksize: usize,
    /// The **short** blocksize `blocksize_0` (a power of two in
    /// `64..=8192`, `<=` [`Self::blocksize`], §4.2.2). When strictly
    /// smaller than the long size, the encoder runs §4.3.1 block
    /// switching: a clean-room energy-envelope transient detector
    /// ([`crate::blocksize::plan_block_sequence`]) schedules short
    /// blocks around attacks (confining quantisation noise to avoid
    /// pre-echo) and long blocks elsewhere, with per-size floors and
    /// residues and the §4.3.1 hybrid window edges at every
    /// long↔short transition. Setting it equal to
    /// [`Self::blocksize`] disables switching (a single-blocksize,
    /// single-mode stream).
    pub short_blocksize: usize,
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
    /// Residue value-book dimensionality: how many consecutive
    /// spectral residue scalars each §8.6.2 VQ codeword covers. A
    /// power of two dividing the residue partition size (16), i.e.
    /// `1 | 2 | 4 | 8 | 16`. At `1` the two cascade value books are
    /// generic scalar ladders sized to the residue range; above it
    /// they are **designed from the stream's own residue corpus**
    /// ([`crate::book_design::design_vq_codebook`]) as
    /// `vq_dims`-dimensional §3.2.1 lookup-type-2 tessellation books
    /// (the coarse book from the raw targets, the fine book from the
    /// post-coarse leftovers), so one trained codeword jointly codes
    /// `vq_dims` neighbouring bins — the shape/correlation gain a
    /// per-scalar ladder cannot reach.
    pub vq_dims: u16,
}

impl StreamEncoderConfig {
    /// A nominal configuration: `quality = 0.7`, long blocksize
    /// `1024` with short blocksize `256` (block switching enabled),
    /// coupling offered on adjacent pairs, 4 codebook-training
    /// iterations, serial `0x6F78_7662` (arbitrary fixed default).
    #[must_use]
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        StreamEncoderConfig {
            sample_rate,
            channels,
            coupling: true,
            quality: 0.7,
            blocksize: 1024,
            short_blocksize: 256,
            serial: 0x6F78_7662,
            training_iterations: 4,
            vq_dims: 1,
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
    /// `short_blocksize` exceeded `blocksize` (§4.2.2 requires
    /// `blocksize_0 <= blocksize_1`).
    BadBlocksizePair {
        /// The configured short (`blocksize_0`) size.
        short_n: usize,
        /// The configured long (`blocksize_1`) size.
        long_n: usize,
    },
    /// `vq_dims` was not a power of two dividing the residue
    /// partition size (§8.6.3 step 1 / §8.6.4: a stage's value-book
    /// dimensions must tile the partition exactly).
    BadVqDims(u16),
    /// The §4.3.1 block-size schedule planner failed.
    Blocksize(BlocksizeError),
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
            OggFileError::BadBlocksizePair { short_n, long_n } => write!(
                f,
                "ogg encode: short blocksize {short_n} exceeds long blocksize {long_n}"
            ),
            OggFileError::BadVqDims(d) => write!(
                f,
                "ogg encode: vq_dims {d} is not a power of two dividing the residue partition size {PARTITION_SIZE}"
            ),
            OggFileError::Blocksize(e) => write!(f, "ogg encode: {e}"),
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
from_err!(BlocksizeError => Blocksize);
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

/// The residue classbook seed: a `CLASS_GROUP_DIMS`-dimensional
/// entropy-only book whose `classes^dims` entries radix-pack one
/// classification per dimension (§8.6.2 classword decode), with
/// uniform codeword lengths (`dims · log2(classes)` bits — Kraft
/// exactly 1 since `classes` is a power of two). Grouping `dims`
/// partitions per classword is what makes a rich class set affordable:
/// the trained (occupancy-optimal) lengths assigned after planning
/// price a common group — e.g. a run of silent partitions — at a few
/// bits total instead of `dims` separate per-partition codewords.
fn class_group_book(classes: u32, dims: u16) -> VorbisCodebook {
    let entries = classes.pow(u32::from(dims));
    let length = (dims as u32 * classes.ilog2()) as u8;
    VorbisCodebook {
        dimensions: dims,
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

/// Linearly resample a positive spectral envelope onto a new bin
/// count. Used to derive a representative design envelope for a block
/// size the schedule never actually used (its floor/residue set must
/// still exist in the setup header for the mode to be legal).
fn resample_envelope(src: &[f32], dst_len: usize) -> Vec<f32> {
    if src.len() == dst_len {
        return src.to_vec();
    }
    let last = (src.len() - 1) as f64;
    let denom = dst_len.saturating_sub(1).max(1) as f64;
    (0..dst_len)
        .map(|i| {
            let pos = i as f64 * last / denom;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(src.len() - 1);
            let t = (pos - lo as f64) as f32;
            src[lo] * (1.0 - t) + src[hi] * t
        })
        .collect()
}

/// The stream's setup header: 4 shared codebooks (floor posts, residue
/// classwords, coarse + fine value books) plus, **per block size**
/// (one entry when `blocksize_0 == blocksize_1`, two — short then long
/// — when the stream switches), a floor, a two-class residue (class 0
/// = silence, class 1 = two-stage cascade over the shared value books,
/// `residue_end` at that size's `n/2`), a mapping carrying the gated
/// §4.3.5 coupling steps under a single submap, and a mode
/// (`blockflag` clear on the short entry, set on the long one).
///
/// The classbook groups [`CLASS_GROUP_DIMS`] partitions per §8.6.2
/// classword (radix-packed); its seed lengths are uniform and the
/// encode path retrains them occupancy-optimal for the final plans.
fn build_setup(
    floor_headers: Vec<crate::setup::Floor1Header>,
    coarse: VorbisCodebook,
    fine: VorbisCodebook,
    half_ns: &[u32],
    coupling: Vec<MappingCouplingStep>,
    switching: bool,
) -> VorbisSetupHeader {
    let floors = floor_headers
        .into_iter()
        .map(|h| FloorHeader {
            floor_type: 1,
            kind: FloorKind::Type1(h),
        })
        .collect();
    let mut both: [Option<u8>; 8] = Default::default();
    both[0] = Some(2);
    both[1] = Some(3);
    let residues = half_ns
        .iter()
        .map(|&half_n| ResidueHeader {
            residue_type: 1,
            residue_begin: 0,
            residue_end: half_n,
            partition_size: PARTITION_SIZE,
            classifications: RESIDUE_CLASSES as u8,
            classbook: 1,
            cascade: vec![0, 0b11],
            books: vec![Default::default(), both],
        })
        .collect();
    let mappings = (0..half_ns.len())
        .map(|e| MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: coupling.clone(),
            // §4.2.4: the mux table is only present when submaps > 1;
            // with one submap every channel implicitly maps to it.
            mux: Vec::new(),
            submap_configs: vec![MappingSubmap {
                time_placeholder: 0,
                floor: e as u8,
                residue: e as u8,
            }],
        })
        .collect();
    let modes = (0..half_ns.len())
        .map(|e| ModeHeader {
            blockflag: switching && e == 1,
            windowtype: 0,
            transformtype: 0,
            mapping: e as u8,
        })
        .collect();
    VorbisSetupHeader {
        codebooks: vec![
            scalar_book(256, 8),
            class_group_book(RESIDUE_CLASSES, CLASS_GROUP_DIMS),
            coarse,
            fine,
        ],
        time_placeholders: vec![0],
        floors,
        residues,
        mappings,
        modes,
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
    /// position of the stream after it (packet `f` finishes
    /// `(n_{f-1} + n_f) / 4` samples per the §4.3.8 lapping rule; the
    /// final packet's granule is the exact input length — the §A.2
    /// end-trim).
    pub audio: Vec<(Vec<u8>, u64)>,
    /// The long blocksize (`blocksize_1`) the stream uses.
    pub blocksize: usize,
    /// The short blocksize (`blocksize_0`); equal to
    /// [`Self::blocksize`] when the stream does not block-switch.
    pub short_blocksize: usize,
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
    let n1 = config.blocksize;
    if !n1.is_power_of_two() || !(64..=8192).contains(&n1) {
        return Err(OggFileError::BadBlocksize(n1));
    }
    let n0 = config.short_blocksize;
    if !n0.is_power_of_two() || !(64..=8192).contains(&n0) {
        return Err(OggFileError::BadBlocksize(n0));
    }
    if n0 > n1 {
        return Err(OggFileError::BadBlocksizePair {
            short_n: n0,
            long_n: n1,
        });
    }
    if !config.vq_dims.is_power_of_two() || PARTITION_SIZE % u32::from(config.vq_dims) != 0 {
        return Err(OggFileError::BadVqDims(config.vq_dims));
    }
    let switching = n0 < n1;
    let ch = config.channels as usize;
    let tuning = EncoderTuning::from_quality(config.quality)?;

    // ---- §4.3.1 block-size schedule ----
    // On a switching stream the per-packet blockflags come from the
    // energy-envelope transient detector over a channel mixdown, and
    // the granule positions from the §4.3.8 walk `(n_prev + n_cur)/4`
    // per packet. A single-blocksize stream is the uniform walk: the
    // priming packet plus one packet per n/2 finished samples.
    let (flags, granules) = if switching {
        let mut mix = pcm[0].clone();
        for row in &pcm[1..] {
            for (m, &v) in mix.iter_mut().zip(row) {
                *m += v;
            }
        }
        let scale = 1.0 / ch as f32;
        for m in &mut mix {
            *m *= scale;
        }
        let plan = plan_block_sequence(
            &mix,
            n0,
            n1,
            TRANSIENT_SUBFRAMES,
            TRANSIENT_PEAK_TO_MEAN,
            TRANSIENT_ENERGY_RISE,
        )?;
        (plan.blockflags, plan.granules)
    } else {
        let half = n1 / 2;
        let frames = samples.div_ceil(half) + 1;
        (
            vec![false; frames],
            (0..frames as u64).map(|f| f * (half as u64)).collect(),
        )
    };
    let frames = flags.len();
    let sizes: Vec<usize> = flags.iter().map(|&fl| if fl { n1 } else { n0 }).collect();

    // ---- §4.3.1 packet preludes + analysis windows ----
    // A long block's window flags mirror its neighbours' blockflags
    // (§4.3.1 step 4a: a clear flag selects the hybrid short-slope
    // edge that laps the adjacent short block). The stream-edge frames
    // take `true` on their outward side: the priming frame's left half
    // and the final frame's right half never reach the output (§4.3.8
    // priming / §A.2 end-trim), so the full-width slope is free.
    let headers: Vec<AudioPacketHeader> = (0..frames)
        .map(|f| AudioPacketHeader {
            mode_number: u32::from(switching && flags[f]),
            blockflag: flags[f],
            n: sizes[f],
            previous_window_flag: flags[f] && (f == 0 || flags[f - 1]),
            next_window_flag: flags[f] && (f + 1 == frames || flags[f + 1]),
        })
        .collect();
    // The handful of distinct window shapes (the short window; the
    // long window with each §4.3.1 edge combination) are built once.
    let mut window_keys: Vec<(bool, bool, bool)> = Vec::new();
    let mut windows: Vec<Vec<f32>> = Vec::new();
    let mut window_of: Vec<usize> = Vec::with_capacity(frames);
    for h in &headers {
        let key = (h.blockflag, h.previous_window_flag, h.next_window_flag);
        let idx = match window_keys.iter().position(|&k| k == key) {
            Some(i) => i,
            None => {
                window_keys.push(key);
                windows.push(h.build_window(n0)?);
                window_keys.len() - 1
            }
        };
        window_of.push(idx);
    }

    // ---- §4.3.8-inverse framing + §4.3.7 forward MDCT ----
    // The forward transform is scaled by 4/n so the decode-side
    // bare-kernel IMDCT (scale 1.0) + §4.3.1 window + §4.3.8
    // overlap-add reconstruct unity PCM: the bare kernels compose as
    // mdct(imdct(X)) == (n/2)·X, and the windowed TDAC overlap-add
    // contributes the remaining factor of ½ (each output sample is
    // reconstructed from its two half-overlapped frames under the
    // w² + w'² = 1 window identity, which the §4.3.1 hybrid edges
    // preserve across long↔short transitions). The per-frame scale
    // keeps this per-frame-linear property on a switched stream.
    let mut spectra: Vec<Vec<Vec<f32>>> = Vec::with_capacity(frames); // [frame][channel][bin]
    {
        // Zero-pad the tail so every frame's analysis span is covered:
        // frame f is centred on granules[f] and spans ±sizes[f]/2.
        let pad = (0..frames)
            .map(|f| granules[f] as usize + sizes[f] / 2)
            .max()
            .unwrap_or(0)
            .saturating_sub(samples);
        let mut splitters: Vec<FrameSplitter> = (0..ch).map(|_| FrameSplitter::new()).collect();
        for (c, splitter) in splitters.iter_mut().enumerate() {
            splitter.push_pcm(&vec![0.0f32; sizes[0] / 2]); // pre-stream silence
            splitter.push_pcm(&pcm[c]);
            splitter.push_pcm(&vec![0.0f32; pad]);
        }
        for f in 0..frames {
            let mut per_ch = Vec::with_capacity(ch);
            for splitter in splitters.iter_mut() {
                // Apply the pending §4.3.8 stride between differing
                // block sizes, then slice; take_frame applies the
                // §4.3.1 analysis window, so the bare kernel follows.
                splitter.advance_pending_stride(sizes[f]);
                let block = splitter.take_frame(sizes[f], &windows[window_of[f]])?;
                per_ch.push(mdct_naive_vec(&block, 4.0 / sizes[f] as f32)?);
            }
            spectra.push(per_ch);
        }
    }

    // ---- psychoacoustics ----
    // On a stream whose frames all share one size, each channel's
    // thresholds run through the temporal pipeline: post-masking decay
    // across frames plus the one-frame-lookahead pre-masking lift (the
    // encoder is whole-stream, so lookahead is free). A genuinely
    // switched stream has a variable frame hop the temporal model does
    // not define, so it uses the per-frame model — pre-echo control on
    // such a stream rests on the short blocks themselves, which is the
    // §1.3.2 mechanism for it.
    let psy_config = PsyConfig {
        threshold_offset_db: tuning.threshold_offset_db,
        ..PsyConfig::new(config.sample_rate)
    };
    let uniform_sizes = sizes.windows(2).all(|w| w[0] == w[1]);
    let mut maskings: Vec<Vec<MaskingAnalysis>> = vec![Vec::with_capacity(ch); frames];
    if uniform_sizes {
        for c in 0..ch {
            let mut temporal =
                TemporalMasking::new(&TemporalMaskingConfig::new(sizes[0] / 2), &psy_config)?;
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
    } else {
        for (f, per_ch) in spectra.iter().enumerate() {
            for x in per_ch {
                maskings[f].push(compute_masking(x, &psy_config)?);
            }
        }
    }

    // ---- floor-1 header design, per block size ----
    // Setup-header entry e covers one block size (0 = short, 1 = long
    // on a switching stream; the single entry otherwise): its floor is
    // designed from the max psy envelope over that size's frames. A
    // size the schedule never used still needs a legal floor — its
    // design envelope is resampled from the used size.
    let n_entries = if switching { 2 } else { 1 };
    let entry_of = |f: usize| usize::from(switching && flags[f]);
    let entry_half = |e: usize| if e == 1 { n1 / 2 } else { n0 / 2 };
    let mut envelopes = Vec::with_capacity(frames);
    let mut env_max: Vec<Vec<f32>> = (0..n_entries)
        .map(|e| vec![f32::MIN_POSITIVE; entry_half(e)])
        .collect();
    let mut env_seen = vec![false; n_entries];
    for (f, (per_ch, m_row)) in spectra.iter().zip(&maskings).enumerate() {
        let e = entry_of(f);
        env_seen[e] = true;
        let mut e_row = Vec::with_capacity(ch);
        for (x, masking) in per_ch.iter().zip(m_row) {
            let envelope = plan_psy_floor_envelope(x, masking, tuning.floor_smooth_radius)?;
            for (acc, &v) in env_max[e].iter_mut().zip(&envelope) {
                *acc = acc.max(v);
            }
            e_row.push(envelope);
        }
        envelopes.push(e_row);
    }
    for e in 0..n_entries {
        if !env_seen[e] {
            let src = env_max[1 - e].clone();
            env_max[e] = resample_envelope(&src, entry_half(e));
        }
    }
    let classes = floor_class_catalogue();
    let floor_book = scalar_book(256, 8);
    let mut floor_headers = Vec::with_capacity(n_entries);
    let mut floor_decoders = Vec::with_capacity(n_entries);
    for (e, env) in env_max.iter().enumerate() {
        // The short floor gets a reduced post budget: its packets
        // recur up to (n1/n0)× as often per second of PCM and cover
        // proportionally fewer bins.
        let budget = if switching && e == 0 {
            (tuning.floor_post_budget / 2).max(4)
        } else {
            tuning.floor_post_budget
        };
        let header = design_floor1_header(env, budget, 0.0, 1, &classes)?;
        let decoder = Floor1Decoder::new(&header, std::slice::from_ref(&floor_book))?;
        floor_headers.push(header);
        floor_decoders.push(decoder);
    }

    // ---- per-frame floor fit + residue targets + NMR weights ----
    let mut floor_ys: Vec<Vec<Vec<u32>>> = Vec::with_capacity(frames);
    let mut targets: Vec<Vec<Vec<f32>>> = Vec::with_capacity(frames);
    let mut weights: Vec<Vec<Vec<f64>>> = Vec::with_capacity(frames);
    for f in 0..frames {
        let e = entry_of(f);
        let half = sizes[f] / 2;
        let mut y_row = Vec::with_capacity(ch);
        let mut t_row = Vec::with_capacity(ch);
        let mut w_row = Vec::with_capacity(ch);
        for c in 0..ch {
            let posts = plan_floor1_envelope(&envelopes[f][c], &floor_headers[e])?;
            let floor1_y = plan_floor1_y(&posts, &floor_headers[e])?;
            let rendered = floor_decoders[e].render_curve(&floor1_y, half);
            let target: Vec<f32> = spectra[f][c]
                .iter()
                .zip(&rendered)
                .map(|(&xv, &fv)| xv / fv)
                .collect();
            let w = residue_partition_weights(&rendered, &maskings[f][c], 0, half, PARTITION_SIZE)?;
            y_row.push(floor1_y);
            t_row.push(target);
            w_row.push(w);
        }
        floor_ys.push(y_row);
        targets.push(t_row);
        weights.push(w_row);
    }

    // ---- §4.3.5 channel coupling (gated per adjacent pair) ----
    // Candidate steps couple the disjoint adjacent pairs (0,1), (2,3),
    // …. The gate is whole-stream (coupling steps live in the setup
    // header's mapping, so the choice is per stream, not per packet):
    // each pair's square-polar energy split is accumulated over every
    // frame's residue targets and the pair is kept only when its angle
    // energy stays under COUPLING_MAX_ANGLE_RATIO × its magnitude
    // energy. Disjoint pairs share no channel with any other step, so
    // the per-pair gate is exact. Kept steps are forward-coupled here —
    // the residue planner below quantises magnitude/angle vectors — and
    // the decoder's §4.3.5 inverse coupling undoes the transform after
    // residue decode, before the §4.3.6 floor multiply. The coupling is
    // applied to the *residue targets* (`X / rendered_floor`), the
    // exact vectors the decoder inverse-couples.
    let coupling_steps: Vec<MappingCouplingStep> = if config.coupling && ch >= 2 {
        (0..ch / 2)
            .filter_map(|pair| {
                let (mag, ang) = (2 * pair, 2 * pair + 1);
                let mut mag_energy = 0.0f64;
                let mut ang_energy = 0.0f64;
                for t_row in &targets {
                    let e = coupling_energy(&t_row[mag], &t_row[ang]);
                    mag_energy += e.magnitude_energy;
                    ang_energy += e.angle_energy;
                }
                (ang_energy <= COUPLING_MAX_ANGLE_RATIO * mag_energy).then_some(
                    MappingCouplingStep {
                        magnitude_channel: mag as u8,
                        angle_channel: ang as u8,
                    },
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    if !coupling_steps.is_empty() {
        for (t_row, w_row) in targets.iter_mut().zip(weights.iter_mut()) {
            forward_couple_all(t_row, &coupling_steps)
                .expect("coupling steps are constructed in range with distinct channels");
            // Merge each coupled pair's per-partition NMR weights to
            // the element-wise max: quantisation error in either
            // coupled vector spreads into both output channels through
            // the inverse coupling, so the more sensitive channel's
            // audibility bound must govern both.
            for step in &coupling_steps {
                let (mag, ang) = (step.magnitude_channel as usize, step.angle_channel as usize);
                for p in 0..w_row[mag].len() {
                    let w = w_row[mag][p].max(w_row[ang][p]);
                    w_row[mag][p] = w;
                    w_row[ang][p] = w;
                }
            }
        }
    }

    let mut max_abs = 0.0f32;
    for t_row in &targets {
        for target in t_row {
            for &t in target {
                max_abs = max_abs.max(t.abs());
            }
        }
    }
    if max_abs <= 0.0 {
        max_abs = 1.0; // all-silent input: any positive ladder scale works
    }

    // ---- setup: the two cascade value books ----
    // vq_dims == 1: generic scalar ladders sized to the residue range.
    // The ladder steps must be exactly §9.2.2-packable (the codebook
    // header carries them as 21-bit-mantissa floats); the book minimum
    // is −32·step, which shares the step's mantissa (× 2⁵) and is
    // therefore packable whenever the step is.
    // vq_dims > 1: multi-dimensional tessellation books designed from
    // the stream's own residue corpus — the coarse book from the raw
    // dims-element sub-vectors, the fine book from the post-coarse
    // leftovers (exactly the targets the §8.6.2 cascade's second stage
    // will see, since plan_partition_cascade subtracts the chosen
    // entry's decoded reconstruction).
    let (coarse, fine) = if config.vq_dims > 1 {
        let d = config.vq_dims as usize;
        // Bounded, deterministic stride subsample of the corpus: the
        // designer is O(vectors × entries) per refinement pass.
        let total_chunks: usize = targets
            .iter()
            .flat_map(|row| row.iter())
            .map(|t| t.len() / d)
            .sum();
        let stride = total_chunks.div_ceil(VQ_DESIGN_MAX_VECTORS).max(1);
        let mut corpus: Vec<f32> = Vec::with_capacity((total_chunks / stride + 1) * d);
        let mut chunk_idx = 0usize;
        for t_row in &targets {
            for target in t_row {
                for chunk in target.chunks_exact(d) {
                    if chunk_idx % stride == 0 {
                        corpus.extend_from_slice(chunk);
                    }
                    chunk_idx += 1;
                }
            }
        }
        let coarse_entries = (VQ_COARSE_ENTRIES_PER_DIM * d as u32).min(VQ_MAX_ENTRIES);
        let fine_entries = (VQ_FINE_ENTRIES_PER_DIM * d as u32).min(VQ_MAX_ENTRIES);
        let coarse = crate::book_design::design_vq_codebook(
            &corpus,
            config.vq_dims,
            coarse_entries,
            8,
            VQ_DESIGN_MAX_CODEWORD_LEN,
        )?
        .codebook;
        // The fine corpus is the coarse stage's leftover: target minus
        // the chosen entry's decoded reconstruction, per sub-vector.
        let mut leftovers: Vec<f32> = Vec::with_capacity(corpus.len());
        for chunk in corpus.chunks_exact(d) {
            let q = crate::vq::quantize_vector(&coarse, chunk)
                .expect("a freshly designed coarse book has >= 1 used entry and matching dims");
            leftovers.extend(chunk.iter().zip(&q.vector).map(|(&t, &v)| t - v));
        }
        let fine = crate::book_design::design_vq_codebook(
            &leftovers,
            config.vq_dims,
            fine_entries,
            8,
            VQ_DESIGN_MAX_CODEWORD_LEN,
        )?
        .codebook;
        (coarse, fine)
    } else {
        (
            signed_value_book(6, crate::book_design::pack_nearest(max_abs / 24.0)),
            signed_value_book(6, crate::book_design::pack_nearest(max_abs / 192.0)),
        )
    };
    let half_ns: Vec<u32> = (0..n_entries).map(|e| entry_half(e) as u32).collect();
    let mut setup = build_setup(
        floor_headers.clone(),
        coarse,
        fine,
        &half_ns,
        coupling_steps,
        switching,
    );

    // ---- optional closed-loop codebook training ----
    // The generic seed ladders are retrained on the stream's own
    // residue targets (codeword lengths from usage, reconstruction
    // values at the observed centroids, re-snapped §9.2.2-packable);
    // the trained table replaces the seeds in the setup header and
    // the weighted per-frame planning below runs under it. On a
    // switching stream the two residue corpora (short-block and
    // long-block targets) train the shared ladders in sequence — the
    // second pass starts from the first pass's books, the classic
    // chained alternating descent.
    if config.training_iterations > 0 {
        for e in 0..n_entries {
            let residuals: Vec<Vec<f32>> = targets
                .iter()
                .enumerate()
                .filter(|(f, _)| entry_of(*f) == e)
                .flat_map(|(_, row)| row.iter().cloned())
                .collect();
            if residuals.is_empty() {
                continue;
            }
            let outcome = crate::book_design::train_residue_books_rd_ladder(
                &residuals,
                &setup.residues[e],
                &setup.codebooks,
                tuning.lambda,
                config.training_iterations,
            )?;
            setup.codebooks = outcome.codebooks;
            // The trainer plans *unweighted*, so its classword
            // statistics do not match the weighted plans the packets
            // below emit. Reset the flat seed classbook — the final
            // classword lengths are trained below from the actual
            // grouped class choices — and take only the trained value
            // books.
            setup.codebooks[1] = class_group_book(RESIDUE_CLASSES, CLASS_GROUP_DIMS);
        }
    }
    let coarse = setup.codebooks[2].clone();
    let fine = setup.codebooks[3].clone();

    // ---- §8.6.2 residue planning (all frames) ----
    // The class rows mirror build_setup: class 0 silence, class 1 the
    // full two-stage cascade. The rate-distortion chooser picks per
    // partition.
    let empty_row: [Option<&VorbisCodebook>; 8] = [None; 8];
    let mut both_row: [Option<&VorbisCodebook>; 8] = [None; 8];
    both_row[0] = Some(&coarse);
    both_row[1] = Some(&fine);
    let value_books = [empty_row, both_row];

    let mut frame_plans: Vec<Vec<ResidueVectorPlan>> = Vec::with_capacity(frames);
    for f in 0..frames {
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
            plans.push(ResidueVectorPlan {
                classifications: scored.classifications,
                partition_entries: scored.partition_entries,
            });
        }
        frame_plans.push(plans);
    }

    // ---- classword + floor-post lengths from the final emissions ----
    // Tally the exact classword emissions the packets below make (the
    // writer's own §8.6.2 grouping, via tally_residue_plans) and the
    // exact §7.2.3 floor-post emissions (tally_floor1_packet — the
    // fitted `floor1_y` values through the shared post book), then
    // make both books' codeword lengths occupancy-optimal — dense
    // policy, so a symbol the corpus never emitted keeps a (long)
    // codeword and each book stays whole-alphabet-encodable. Codeword
    // lengths carry no values, so the packets decode to bit-identical
    // PCM; they only serialise into fewer bits.
    {
        let mut tallies = crate::book_design::BookTallies::new(&setup.codebooks);
        for (f, plans) in frame_plans.iter().enumerate() {
            let e = entry_of(f);
            crate::book_design::tally_residue_plans(
                &mut tallies,
                plans,
                &setup.residues[e],
                &setup.codebooks,
            )?;
            for y_row in floor_ys[f].iter().take(ch) {
                crate::book_design::tally_floor1_packet(
                    &mut tallies,
                    &Floor1Packet {
                        nonzero: true,
                        floor1_y: y_row.clone(),
                        partition_cvals: vec![0u32; floor_headers[e].partition_class_list.len()],
                    },
                    &floor_headers[e],
                )?;
            }
        }
        for book in [0usize, 1] {
            if let Some(freqs) = tallies.counts(book) {
                setup.codebooks[book] =
                    crate::book_design::redesign_codebook(&setup.codebooks[book], freqs, 16, true)?;
            }
        }
    }

    // ---- the three §4.2 header packets ----
    let id_packet = write_identification_header(&VorbisIdentificationHeader {
        vorbis_version: 0,
        audio_channels: config.channels,
        audio_sample_rate: config.sample_rate,
        bitrate_maximum: 0,
        bitrate_nominal: 0,
        bitrate_minimum: 0,
        blocksize_0: n0 as u16,
        blocksize_1: n1 as u16,
    })?;
    let comment_packet = write_comment_header(&VorbisCommentHeader {
        vendor: "oxideav-vorbis clean-room encoder".into(),
        comments: vec!["ENCODER=oxideav-vorbis".into()],
    })?;
    let setup_packet = write_setup_header(&setup, config.channels)?;

    // ---- §4.3 audio packets + §A.2 encapsulation ----
    let mut audio_packets: Vec<(Vec<u8>, u64)> = Vec::with_capacity(frames);
    for (f, plans) in frame_plans.into_iter().enumerate() {
        let e = entry_of(f);
        let mut floors = Vec::with_capacity(ch);
        for y_row in floor_ys[f].iter().take(ch) {
            floors.push(AudioChannelFloor::Type1(Floor1Packet {
                nonzero: true,
                floor1_y: y_row.clone(),
                partition_cvals: vec![0u32; floor_headers[e].partition_class_list.len()],
            }));
        }
        let submap_plans = [plans];
        let packet = write_audio_packet(
            &headers[f],
            &setup,
            n0,
            n1,
            config.channels,
            &floors,
            &submap_plans,
        )?;
        // §4.3.8: packet f finishes (n_{f-1} + n_f) / 4 samples — the
        // schedule's granule walk; the final packet's granule is the
        // true sample count (§A.2 end-trim).
        let granule = if f + 1 == frames {
            samples as u64
        } else {
            granules[f]
        };
        audio_packets.push((packet, granule));
    }
    Ok(EncodedVorbisStream {
        identification: id_packet,
        comment: comment_packet,
        setup: setup_packet,
        audio: audio_packets,
        blocksize: n1,
        short_blocksize: n0,
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

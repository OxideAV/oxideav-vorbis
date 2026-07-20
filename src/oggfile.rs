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

/// §8.6.1 residue partition size for small spectra (short blocks): the
/// class choice stays fine-grained where one block covers few bins.
const PARTITION_SIZE_SHORT: u32 = 16;

/// §8.6.1 residue partition size for large spectra (long blocks): a
/// long block's spectrum is locally homogeneous at twice the bin
/// density, so double-size partitions halve the per-partition
/// overhead (classwords, class-choice granularity) at no measured
/// fidelity cost.
const PARTITION_SIZE_LONG: u32 = 32;

/// The partition size a residue covering `half_n` spectral bins uses:
/// [`PARTITION_SIZE_LONG`] once the spectrum is at least 512 bins
/// (blocksize ≥ 1024), [`PARTITION_SIZE_SHORT`] below.
fn partition_size_for(half_n: u32) -> u32 {
    if half_n >= 512 {
        PARTITION_SIZE_LONG
    } else {
        PARTITION_SIZE_SHORT
    }
}

/// The §8.6.1 residue bandpass cutoff: spectral bins at and above this
/// frequency are left uncoded (`residue_end` caps the coded band; the
/// §8.6.2 decode zeroes every bin past it). The bound is the crate's
/// own psychoacoustic model: the analytic threshold-in-quiet the `psy`
/// module carries rises as `10⁻³·(f/kHz)⁴` dB at the top of the
/// audio band — ≥ 160 dB at 20 kHz, far above any program material
/// full-scale can represent — so residue spent past 20 kHz can never
/// be audible. Streams whose Nyquist sits at or below the cutoff are
/// uncapped, and the cap is rounded **up** to a partition boundary so
/// no bin under the cutoff is ever cut. (At 44.1 kHz and `half_n =
/// 1024` this lands on `residue_end = 960` — the same coded-band cap
/// the staged reference streams carry.)
const RESIDUE_CUTOFF_HZ: f64 = 20_000.0;

/// §8.6.1 `[residue_end]` for a spectrum of `half_n` bins at
/// `sample_rate`: the first partition boundary at or above
/// [`RESIDUE_CUTOFF_HZ`] (bin `k` covers frequencies near
/// `k · (sample_rate / 2) / half_n`), capped at `half_n`.
fn residue_end_for(half_n: u32, sample_rate: u32, partition_size: u32) -> u32 {
    let nyquist = f64::from(sample_rate) / 2.0;
    if nyquist <= RESIDUE_CUTOFF_HZ {
        return half_n;
    }
    let bins = (f64::from(half_n) * RESIDUE_CUTOFF_HZ / nyquist).ceil() as u32;
    bins.div_ceil(partition_size)
        .saturating_mul(partition_size)
        .min(half_n)
}

/// The amplitude-band ladder gate: the **mid band** book is carried
/// only when the median above-noise partition peak sits at or below
/// `max_abs / QUIET_BAND_MIN_RATIO` — without that separation the
/// "mid" band is simply the loud band and the extra class cannot
/// cover its setup-header cost, so the ladder stays at the four base
/// classes.
const QUIET_BAND_MIN_RATIO: f32 = 4.0;

/// Floor on the mid band's span: a corpus whose median above-noise
/// peak is tiny still needs the mid book to reach ordinary
/// near-threshold texture, so the span never shrinks below
/// `max_abs / 32` (the noise class covers the region below — its
/// ternary reach is `max_abs / 48`).
const QUIET_BAND_MAX_RATIO: f32 = 32.0;

/// Minimum number of above-noise coded partitions needed before the
/// mid-band statistics are trusted (and the extra book's setup bytes
/// can possibly amortise).
const QUIET_BAND_MIN_PARTITIONS: usize = 32;

/// Scalar levels per dimension of the mid band book's uniform ladder
/// (entries = `levels^dims` = 5⁴ = 625, §3.2.1 lookup type 1). Five
/// levels give the mid tier a ±2-step reach spanning its band's
/// median partition peak, one codeword per [`NOISE_BOOK_DIMS`] bins —
/// the same joint-dimensionality rate mechanism as the noise class,
/// one amplitude tier up.
const MID_BOOK_LEVELS: u32 = 5;

/// Classword-aware planning refinements: after the value-bit-only
/// first pass, how many plan ↔ re-price alternations the integrated
/// encoder runs with the per-class marginal classword bias (see the
/// planning loop in [`encode_pcm_to_packets`]'s geometry core). Each
/// refinement is a full re-plan (~a third of the planning time), and
/// the measured second refinement changes almost nothing (the class
/// histogram stabilises after one), so one refinement is the
/// default; the loop stops early at a plan fixed point.
const CLASSWORD_PRICE_PASSES: usize = 1;

/// Cap on the per-partition marginal classword price (bits): keeps a
/// rare-but-needed class expensive rather than unreachable.
const CLASSWORD_PRICE_CAP_BITS: u8 = 24;

/// Cap on the stride-subsampled training corpus handed to
/// [`crate::book_design::design_lattice_vq_codebook`] (in
/// sub-vectors): the designer is O(vectors × levels), so a long
/// stream trains on a bounded, deterministic sample of its residue.
const VQ_DESIGN_MAX_VECTORS: usize = 6144;

/// Ceiling on a designed lattice book's entry count
/// (`lookup1_values^dims` — quantisation scans every used entry per
/// §8.6.2 read, and the codeword-length table is carried per entry).
const VQ_LATTICE_MAX_ENTRIES: u32 = 1024;

/// Codeword-length cap handed to the VQ designer's occupancy-optimal
/// length assignment (well under the §3.2.1 hard 32-bit limit; a
/// longer codeword than this prices an entry out of use anyway).
const VQ_DESIGN_MAX_CODEWORD_LEN: u8 = 24;

// (The base four-class ladder — silence / noise / coarse /
// coarse + fine — is built by `ResidueLadder::base`; the
// amplitude-band designer appends a quiet coarse (+ fine) pair, so
// the stream's `residue_classifications` is 4 or 5 depending on the
// corpus statistics. See `ResidueLadder`.)

/// The designed lattice fine ladder's **coverage cap**: the largest
/// fine-resolution scale (see
/// [`EncoderTuning::fine_resolution_scale`]) the `vq_dims = 2` joint
/// geometry can follow. The lattice's per-dimension level count is
/// pinned by the entry ceiling, so its base span carries exactly 2×
/// headroom over the coarse-leftover bound — shrinking the step
/// further clips the leftover extremes and the fidelity *collapses*
/// (measured 48 → 36 dB at 4×). Past the cap the integrated encoder
/// brings the scalar-ladder geometry into play, whose 64 levels span
/// two full coarse steps at any knob-scaled step: the two geometries'
/// rate/SNR frontiers cross near the cap (measured on the staged mono
/// corpus, 8.2 kB / 52.7 dB scalar vs 8.6 kB / 52.6 dB joint at
/// `q = 0.85`), but *where* is stream-dependent, so the top band
/// encodes both candidates and keeps the better — see the geometry
/// selector in [`encode_pcm_to_packets`]. An in-ladder hybrid (both
/// geometries as competing residue classes, chosen per partition) was
/// measured and rejected: the closed-loop trainer plans *unweighted*,
/// routing the loud partitions to the scalar classes and
/// sparse-pruning the joint books' loud cells, after which the final
/// *weighted* plans route those partitions back onto the pruned joint
/// books — whose (perceptually masked but numerically huge)
/// reconstruction error collapsed the measured stream SNR by 13 dB at
/// `q = 0.9`.
const LATTICE_FINE_COVERAGE_CAP: f32 = 2.0;

/// The quality setting whose fine-resolution scale sits exactly at
/// [`LATTICE_FINE_COVERAGE_CAP`] — the joint geometry's cap point
/// (the `fine_step_divisor` law is `192 · 4^((q − 0.7) / 0.3)`, so
/// scale 2 lands at `q = 0.7 + 0.3 · log₄ 2 = 0.85`). The top-band
/// selector encodes its joint candidate at this setting: past it the
/// joint books' resolution is pinned, and a lower `lambda` only buys
/// saturated-SNR density (measured +59 % audio bytes for +0.13 dB
/// from `q = 0.85` to `q = 1` on the staged mono corpus).
const LATTICE_SEAM_QUALITY: f32 = 0.85;

/// Dimensionality of the class-1 noise book: how many consecutive
/// residue bins one noise codeword covers. Quiet partitions are the
/// bulk of a typical spectrum, and a scalar book charges them one
/// codeword **per bin** (≥ 16 bits per partition just to spell
/// near-silence); a 4-dimensional joint book cuts that to 4 codewords
/// whose trained lengths price common quiet patterns at a few bits.
const NOISE_BOOK_DIMS: u16 = 4;

/// Scalar levels per dimension of the noise book's shared uniform
/// ladder (entries = `levels^dims`, §3.2.1 lookup type 1). Three
/// levels span `{−s, 0, s}` — a ternary texture code. Measured
/// against a five-level (625-entry) variant on the staged corpus,
/// ternary wins across the board: the 81-entry grid's occupancy
/// concentrates (shorter trained codewords, −4…−11 % stream bytes at
/// identical SNR through the low and middle of the knob), its
/// codeword-length table costs ~300 B less setup header, and
/// quantisation scans 8× fewer entries; what a ±2s reach carried
/// better is instead picked up by the coarse classes.
const NOISE_BOOK_LEVELS: u32 = 3;

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
    /// spectral residue scalars each §8.6.2 VQ codeword covers —
    /// `2` (default) or `1`. At `1` the two cascade value books are
    /// generic scalar ladders sized to the residue range; at `2` they
    /// are **designed from the stream's own residue corpus**
    /// ([`crate::book_design::design_lattice_vq_codebook`]) as
    /// 2-dimensional §3.2.1 lookup-type-**1** lattice books — the
    /// widely interoperable lookup form — over uniform full-span
    /// ladders, with codeword lengths trained on the *joint*
    /// grid-cell occupancy, so one trained codeword jointly codes two
    /// neighbouring bins, **through the low and middle of the
    /// quality knob**: past the lattice fine ladder's coverage cap
    /// (`quality > 0.85`) the encoder races the scalar-ladder
    /// geometry against the joint geometry frozen at its cap point
    /// and keeps the higher own-decoded SNR (see
    /// [`encode_pcm_to_packets`]), so the knob stays monotone where
    /// the joint books' pinned resolution saturates. Wider
    /// dimensionalities are refused: under the lattice entry ceiling
    /// their per-scalar resolution collapses (see
    /// [`OggFileError::BadVqDims`]).
    pub vq_dims: u16,
    // internal A/B lever for the amplitude-band ladder — exposed for
    // tests/measurement; not part of the stable API
    #[doc(hidden)]
    pub residue_bands: bool,
}

impl StreamEncoderConfig {
    /// A nominal configuration: `quality = 0.7`, long blocksize
    /// `2048` with short blocksize `256` (block switching enabled),
    /// coupling offered on adjacent pairs, 4 codebook-training
    /// iterations, serial `0x6F78_7662` (arbitrary fixed default),
    /// `vq_dims = 2` (the corpus-designed joint lattice books, with
    /// the per-band geometry selection described on [`Self::vq_dims`]).
    ///
    /// The `2048/256` block pair matches the corpus streams'
    /// geometry: against `1024/256` the doubled long transform halves
    /// the per-second packet overhead (floor fits, classwords,
    /// preludes) and doubles the spectral resolution steady content
    /// is coded at — measured on the staged real-audio corpus this is
    /// a 20–40 % stream-byte cut at equal-or-better SNR. The
    /// `vq_dims = 2` default is likewise measured: on the staged
    /// mono corpus at the default quality the joint books spend
    /// −22 % audio bytes at +6.3 dB SNR against the scalar ladders.
    #[must_use]
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        StreamEncoderConfig {
            sample_rate,
            channels,
            coupling: true,
            quality: 0.7,
            blocksize: 2048,
            short_blocksize: 256,
            serial: 0x6F78_7662,
            training_iterations: 4,
            vq_dims: 2,
            residue_bands: true,
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
    /// `vq_dims` was not `1` or `2`. A stage's value-book dimensions
    /// must tile the partition exactly (§8.6.3 step 1 / §8.6.4), and
    /// above 2 the designed lattice's `lookup1_values^dims` product
    /// grid cannot carry a usable per-scalar resolution under the
    /// entry ceiling (a 4-D grid at ≤1024 entries is 5 levels per
    /// scalar — the joint form needs the per-partition class ladder
    /// before wider dimensionalities pay).
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
            OggFileError::BadVqDims(d) => write!(f, "ogg encode: vq_dims {d} is not 1 or 2"),
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
    // Ceil log2: for a non-power-of-two class count the uniform seed
    // lengths under-fill the Kraft sum — legal as a planning proxy
    // (the seed never reaches the wire; the occupancy-optimal dense
    // retrain below replaces it with exact-Kraft lengths).
    let length = (dims as u32 * classes.next_power_of_two().ilog2()) as u8;
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

/// Bounded, deterministic stride subsample of a flat corpus of
/// `dims`-element sub-vectors: the VQ designers are
/// O(vectors × entries) per refinement pass, so a long stream trains
/// on at most `max_vectors` evenly strided sub-vectors.
fn subsample_corpus(corpus: Vec<f32>, dims: usize, max_vectors: usize) -> Vec<f32> {
    let chunks = corpus.len() / dims;
    if chunks <= max_vectors {
        return corpus;
    }
    let stride = chunks.div_ceil(max_vectors).max(1);
    corpus
        .chunks_exact(dims)
        .step_by(stride)
        .flatten()
        .copied()
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

/// The per-partition residue **class ladder** a stream carries: the
/// §8.6.1 `cascade` bitmap + `books` rows (one per class, book indices
/// into the final codebook table) and the value books they reference
/// (appended to the codebook table after the floor-post book (0) and
/// the classbook (1), i.e. the first ladder book is codebook 2).
///
/// The base ladder is the four-class silence / noise / coarse /
/// coarse + fine set; the amplitude-band designer appends a **mid
/// band** class — a joint [`NOISE_BOOK_DIMS`]-dimensional book whose
/// ladder reaches the corpus' median above-noise partition — giving
/// the rate-distortion chooser a per-band value-book assignment:
/// each partition's classword selects the band book whose span (and
/// joint dimensionality) matches its amplitude, priced against the
/// books' exact codeword costs.
struct ResidueLadder {
    /// Value books, in codebook-table order starting at index 2.
    value_books: Vec<VorbisCodebook>,
    /// §8.6.1 per-class cascade bitmap.
    cascade: Vec<u8>,
    /// §8.6.1 per-class, per-pass book indices (codebook-table space).
    books: Vec<[Option<u8>; 8]>,
}

impl ResidueLadder {
    /// The four-class base ladder: class 0 silence, class 1 the joint
    /// noise book (pass 0), class 2 coarse-only (pass 0), class 3 the
    /// coarse + fine two-stage cascade. Codebook order: coarse (2),
    /// fine (3), noise (4) — the historical table layout.
    fn base(coarse: VorbisCodebook, fine: VorbisCodebook, noise: VorbisCodebook) -> Self {
        let mut noise_only: [Option<u8>; 8] = Default::default();
        noise_only[0] = Some(4);
        let mut coarse_only: [Option<u8>; 8] = Default::default();
        coarse_only[0] = Some(2);
        let mut both: [Option<u8>; 8] = Default::default();
        both[0] = Some(2);
        both[1] = Some(3);
        ResidueLadder {
            value_books: vec![coarse, fine, noise],
            cascade: vec![0, 0b01, 0b01, 0b11],
            books: vec![Default::default(), noise_only, coarse_only, both],
        }
    }

    /// Append one further band class carrying a single-pass joint
    /// band book (the mid-amplitude tier: the noise class's shape at
    /// a wider ladder).
    fn push_band_class(&mut self, book: VorbisCodebook) {
        let index = (2 + self.value_books.len()) as u8;
        self.value_books.push(book);
        let mut row: [Option<u8>; 8] = Default::default();
        row[0] = Some(index);
        self.cascade.push(0b01);
        self.books.push(row);
    }

    /// §8.6.1 `residue_classifications` this ladder declares.
    fn classifications(&self) -> u32 {
        self.cascade.len() as u32
    }
}

/// The stream's setup header: the floor-post book (0), the residue
/// classbook (1) and the ladder's value books (2..) plus, **per block
/// size** (one entry when `blocksize_0 == blocksize_1`, two — short
/// then long — when the stream switches), a floor, a residue carrying
/// the ladder's classes with `residue_end` at that size's coded-band
/// cap ([`residue_end_for`]), a mapping carrying the gated §4.3.5
/// coupling steps under a single submap, and a mode (`blockflag`
/// clear on the short entry, set on the long one).
///
/// The classbook groups [`CLASS_GROUP_DIMS`] partitions per §8.6.2
/// classword (radix-packed); its seed lengths are uniform and the
/// encode path retrains them occupancy-optimal for the final plans.
fn build_setup(
    floor_headers: Vec<crate::setup::Floor1Header>,
    ladder: ResidueLadder,
    half_ns: &[u32],
    residue_ends: &[u32],
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
    let classifications = ladder.classifications();
    let residues = half_ns
        .iter()
        .zip(residue_ends)
        .map(|(&half_n, &residue_end)| ResidueHeader {
            residue_type: 1,
            residue_begin: 0,
            residue_end,
            partition_size: partition_size_for(half_n),
            classifications: classifications as u8,
            classbook: 1,
            cascade: ladder.cascade.clone(),
            books: ladder.books.clone(),
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
    let mut codebooks = vec![
        scalar_book(256, 8),
        class_group_book(classifications, CLASS_GROUP_DIMS),
    ];
    codebooks.extend(ladder.value_books);
    VorbisSetupHeader {
        codebooks,
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
    let tuning = EncoderTuning::from_quality(config.quality)?;
    let past_cap = tuning.fine_resolution_scale() > LATTICE_FINE_COVERAGE_CAP * 1.0001;
    if config.vq_dims > 1 && past_cap {
        // ---- top-of-knob geometry selection (vq_dims = 2 only) ----
        // Past the lattice fine ladder's coverage cap the joint
        // geometry saturates, so the scalar geometry (whose fine step
        // follows the knob everywhere) takes over — but *which* knob
        // setting the two frontiers cross at is stream-dependent (on
        // the staged corpus the mono-44100 seam is clean while the
        // mono-22050 joint encode at the cap still leads the scalar
        // encode by ≈ 5 dB one knob step past it). So the top band
        // encodes both candidates — the scalar geometry at the
        // requested quality, and the joint geometry *frozen at its
        // cap point* (its cheapest saturated setting; running it past
        // the cap only buys saturated-SNR density) — and keeps the
        // one whose own-decoded whole-stream SNR is higher, ties to
        // fewer bytes. Monotone by construction: the joint
        // candidate's SNR is a constant in `q`, the scalar
        // candidate's is non-decreasing, and `max` preserves both.
        let seam_tuning = EncoderTuning::from_quality(LATTICE_SEAM_QUALITY)?;
        debug_assert!(
            (seam_tuning.fine_resolution_scale() - LATTICE_FINE_COVERAGE_CAP).abs() < 1e-3,
            "LATTICE_SEAM_QUALITY must sit exactly at the coverage cap"
        );
        let scalar = encode_pcm_to_packets_geometry(pcm, config, &tuning, false)?;
        let joint = encode_pcm_to_packets_geometry(pcm, config, &seam_tuning, true)?;
        let scalar_snr = decoded_stream_snr(&scalar, pcm)?;
        let joint_snr = decoded_stream_snr(&joint, pcm)?;
        let scalar_bytes: usize = scalar.audio.iter().map(|(p, _)| p.len()).sum();
        let joint_bytes: usize = joint.audio.iter().map(|(p, _)| p.len()).sum();
        let keep_scalar =
            scalar_snr > joint_snr || (scalar_snr == joint_snr && scalar_bytes <= joint_bytes);
        return Ok(if keep_scalar { scalar } else { joint });
    }
    encode_pcm_to_packets_geometry(pcm, config, &tuning, config.vq_dims > 1)
}

/// Own-decode a packet stream and report the whole-stream SNR (dB)
/// against the input PCM (`10·log10(Σ signal² / Σ error²)` across all
/// channels) — the top-band geometry selector's ground truth.
fn decoded_stream_snr(stream: &EncodedVorbisStream, pcm: &[Vec<f32>]) -> Result<f64, OggFileError> {
    let id = parse_identification_header(&stream.identification)
        .map_err(|e| OggFileError::Header(e.to_string()))?;
    let setup = parse_setup_header(&stream.setup, id.audio_channels)
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
    let mut decoded: Vec<Vec<f32>> = vec![Vec::new(); ch];
    for (packet, _) in &stream.audio {
        let mut reader = oxideav_core::bits::BitReaderLsb::new(packet);
        if let StreamingFrame::Pcm {
            per_channel_pcm, ..
        } = decoder.push_packet(&mut reader, &setup, &state)?
        {
            for (row, samples) in decoded.iter_mut().zip(&per_channel_pcm) {
                row.extend_from_slice(samples);
            }
        }
    }
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (reference, out) in pcm.iter().zip(&decoded) {
        let n = reference.len().min(out.len());
        for (&r, &d) in reference[..n].iter().zip(&out[..n]) {
            sig += f64::from(r) * f64::from(r);
            let e = f64::from(r) - f64::from(d);
            err += e * e;
        }
    }
    Ok(if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sig / err).log10()
    })
}

/// The single-geometry encode under [`encode_pcm_to_packets`]:
/// `joint_geometry` selects the corpus-designed 2-D lattice books
/// (`true`) or the scalar ladders (`false`), and `tuning` carries the
/// expanded quality levers (the top-band selector deliberately hands
/// the joint candidate its cap-point tuning rather than the requested
/// quality's).
fn encode_pcm_to_packets_geometry(
    pcm: &[Vec<f32>],
    config: &StreamEncoderConfig,
    tuning: &EncoderTuning,
    joint_geometry: bool,
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
    if !config.vq_dims.is_power_of_two() || config.vq_dims > 2 {
        return Err(OggFileError::BadVqDims(config.vq_dims));
    }
    let switching = n0 < n1;
    let ch = config.channels as usize;

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
    // The §8.6.1 coded-band cap per setup entry (residue_end_for), and
    // the per-frame partition size / coded-band views derived from it.
    let residue_ends: Vec<usize> = (0..n_entries)
        .map(|e| {
            let half = entry_half(e) as u32;
            residue_end_for(half, config.sample_rate, partition_size_for(half)) as usize
        })
        .collect();
    let frame_ps = |f: usize| partition_size_for((sizes[f] / 2) as u32) as usize;
    let frame_res_end = |f: usize| residue_ends[entry_of(f)];
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
            // NMR weights cover the §8.6.1 coded band only: bins past
            // `residue_end` are never coded (the decoder zeroes them),
            // so the planner sees exactly one weight per coded
            // partition.
            let w = residue_partition_weights(
                &rendered,
                &maskings[f][c],
                0,
                frame_res_end(f),
                partition_size_for(half as u32),
            )?;
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

    // ---- per-partition peak statistics over the coded band ----
    // Everything downstream — the ladder spans, the amplitude-band
    // split, the band design corpora — is driven by the peak |target|
    // of each §8.6 partition inside the coded band (bins past
    // `residue_end` are never coded, so a loud ultrasonic bin must not
    // widen any ladder).
    let mut max_abs = 0.0f32;
    let mut partition_peaks: Vec<f32> = Vec::new();
    for (f, t_row) in targets.iter().enumerate() {
        let (end, ps) = (frame_res_end(f), frame_ps(f));
        for target in t_row {
            for part in target[..end].chunks_exact(ps) {
                let peak = part.iter().fold(0.0f32, |m, &t| m.max(t.abs()));
                partition_peaks.push(peak);
                max_abs = max_abs.max(peak);
            }
        }
    }
    if max_abs <= 0.0 {
        max_abs = 1.0; // all-silent input: any positive ladder scale works
    }

    // ---- the amplitude-band split ----
    // The per-`(partition, pass)` value-book assignment ranges over
    // amplitude **bands**: the near-silent band is served by silence
    // and the joint [`NOISE_BOOK_DIMS`]-dimensional ternary noise
    // book, the loud band by the coarse (+ fine) cascade pair — and
    // the population in between, which the coarse pair serves at one
    // codeword per `vq_dims` bins, is where a **mid band book** of
    // the noise book's dimensionality pays: one codeword per
    // [`NOISE_BOOK_DIMS`] bins at a ladder reaching the band's median
    // partition. (A same-dimensionality narrower-span coarse + fine
    // pair was measured and rejected: the occupancy-trained codeword
    // lengths already price the amplitude statistics inside one book,
    // so the extra pair bought no rate — the band win must come from
    // joint dimensionality, exactly like the noise class.) The split
    // point is the median peak of the partitions above the noise
    // band, carried only when the corpus genuinely separates
    // (median ≤ max_abs / QUIET_BAND_MIN_RATIO — otherwise the "mid"
    // band IS the loud band) and there are enough such partitions for
    // the statistics — and the extra setup bytes — to pay. The
    // rate-distortion chooser does the actual per-partition band
    // assignment, priced against each book's exact codeword costs: a
    // loud partition clips the mid book (huge distortion, priced
    // out), a near-silent one is served cheaper by the noise class.
    let mid_span: Option<f32> = if config.residue_bands {
        // The noise class's inclusion bound (its ternary reach is
        // max_abs/48; corpus gathering below admits 1.5× that).
        let noise_bound = 1.5 * max_abs / 48.0;
        let mut above: Vec<f32> = partition_peaks
            .iter()
            .copied()
            .filter(|&p| p > noise_bound)
            .collect();
        if above.len() >= QUIET_BAND_MIN_PARTITIONS {
            above.sort_unstable_by(f32::total_cmp);
            let median = above[above.len() / 2];
            (median <= max_abs / QUIET_BAND_MIN_RATIO)
                .then(|| median.max(max_abs / QUIET_BAND_MAX_RATIO))
        } else {
            None
        }
    } else {
        None
    };

    // ---- setup: the cascade value books, one pair per band ----
    // vq_dims == 1: generic scalar ladders sized to the residue range.
    // The ladder steps must be exactly §9.2.2-packable (the codebook
    // header carries them as 21-bit-mantissa floats); the book minimum
    // is −32·step, which shares the step's mantissa (× 2⁵) and is
    // therefore packable whenever the step is.
    // vq_dims > 1: multi-dimensional §3.2.1 lookup-type-1 lattice
    // books designed from the stream's own residue corpus — the
    // coarse book's shared scalar ladder from the raw dims-element
    // sub-vectors, the fine book's from the post-coarse leftovers
    // (exactly the targets the §8.6.2 cascade's second stage will
    // see, since plan_partition_cascade subtracts the chosen entry's
    // decoded reconstruction) — with sparse codeword lengths trained
    // on the *joint* grid-cell occupancy. Lookup type 1 is the widely
    // interoperable lookup form; a type-2 (per-entry-free) table is
    // spec-legal but rejected by common black-box decoders.
    let (coarse, fine) = if joint_geometry {
        let d = config.vq_dims as usize;
        // Design corpus: every coded partition's chunks (bins past
        // `residue_end` are never coded and must not shape the books).
        let mut raw: Vec<f32> = Vec::new();
        for (f, t_row) in targets.iter().enumerate() {
            let (end, ps) = (frame_res_end(f), frame_ps(f));
            for target in t_row {
                for part in target[..end].chunks_exact(ps) {
                    raw.extend_from_slice(part);
                }
            }
        }
        // The widest shared scalar ladder whose full product grid
        // stays under the entry ceiling: §3.2.1 lookup type 1 derives
        // `lookup1_values` from `entries`, so the designed `entries`
        // is exactly `lookup1_values^dims`.
        let mut lv: u32 = 2;
        while (u64::from(lv) + 1).pow(u32::from(config.vq_dims))
            <= u64::from(VQ_LATTICE_MAX_ENTRIES)
        {
            lv += 1;
        }
        // One band's coarse + fine lattice pair over uniform ladders
        // spanning `span`, mirroring the proven scalar-seed
        // proportions (a corpus-quantile ladder would concentrate its
        // levels in the near-zero mass and abandon the rare-but-loud
        // outliers — exactly the audible material). The joint-coding
        // win comes from the occupancy-trained codeword lengths,
        // dense so a cell the subsampled corpus missed stays
        // reachable (the closed-loop trainer prunes against the full
        // corpus below).
        //
        // The fine corpus is the coarse stage's leftover: target
        // minus the chosen entry's decoded reconstruction, per
        // sub-vector. Its base ladder spans two coarse steps — the
        // leftover is bounded by half a coarse step plus grid-snap
        // slack, so the base span carries 2× coverage headroom. The
        // quality knob's fine-resolution scale divides the step —
        // the top of the knob must lower the reconstruction noise
        // floor (with a fixed step the whole-stream SNR saturates
        // near q ≈ 0.7 while the falling lambda only buys
        // saturated-SNR density) — but only down to the coverage
        // bound ([`LATTICE_FINE_COVERAGE_CAP`]): past 2× the shrunk
        // span clips the leftover extremes and the SNR *collapses*
        // (measured 48 → 36 dB at 4×). The scalar ladder has no such
        // cap because its 64 levels always span two full coarse
        // steps; the lattice's per-dimension level count is pinned by
        // the entry ceiling, which is why the whole geometry hands
        // over to the scalar ladders past the cap (see
        // `joint_geometry` above).
        let design_pair = |corpus: Vec<f32>,
                           span: f32|
         -> Result<(VorbisCodebook, VorbisCodebook), OggFileError> {
            let corpus = subsample_corpus(corpus, d, VQ_DESIGN_MAX_VECTORS);
            let corpus = if corpus.is_empty() {
                vec![0.0; d]
            } else {
                corpus
            };
            let coarse_step = crate::book_design::pack_nearest(8.0 * span / (3.0 * lv as f32));
            let coarse_ladder = crate::book_design::uniform_value_ladder(
                -(lv as f32 / 2.0) * coarse_step,
                coarse_step,
                lv,
                8,
            )?;
            let coarse = crate::book_design::design_lattice_vq_codebook(
                &corpus,
                config.vq_dims,
                &coarse_ladder,
                VQ_DESIGN_MAX_CODEWORD_LEN,
                true,
            )?
            .codebook;
            let mut leftovers: Vec<f32> = Vec::with_capacity(corpus.len());
            for chunk in corpus.chunks_exact(d) {
                let q = crate::vq::quantize_vector(&coarse, chunk)
                    .expect("a freshly designed coarse book has >= 1 used entry and matching dims");
                leftovers.extend(chunk.iter().zip(&q.vector).map(|(&t, &v)| t - v));
            }
            let fine_step = crate::book_design::pack_nearest(
                2.0 * coarse_step
                    / lv as f32
                    / tuning
                        .fine_resolution_scale()
                        .min(LATTICE_FINE_COVERAGE_CAP),
            );
            let fine_ladder = crate::book_design::uniform_value_ladder(
                -(lv as f32 / 2.0) * fine_step,
                fine_step,
                lv,
                8,
            )?;
            let fine = crate::book_design::design_lattice_vq_codebook(
                &leftovers,
                config.vq_dims,
                &fine_ladder,
                VQ_DESIGN_MAX_CODEWORD_LEN,
                true,
            )?
            .codebook;
            Ok((coarse, fine))
        };
        design_pair(raw, max_abs)?
    } else {
        // The coarse span is fixed (it must reach the loudest residue
        // target); the fine step follows the quality knob — the top
        // of the knob lowers the encoder's reconstruction noise floor
        // (see [`EncoderTuning::fine_step_divisor`]). The scalar fine
        // ladder's 64 levels span two full coarse steps at any
        // knob-scaled step, so the whole knob is reachable in this
        // geometry.
        (
            signed_value_book(6, crate::book_design::pack_nearest(max_abs / 24.0)),
            signed_value_book(
                6,
                crate::book_design::pack_nearest(max_abs / tuning.fine_step_divisor),
            ),
        )
    };
    // ---- the joint band books (noise + optional mid tier) ----
    // Quiet partitions dominate a typical spectrum; a joint band book
    // codes them at one codeword per NOISE_BOOK_DIMS bins instead of
    // one per bin. Each band book is designed from the stream's own
    // partitions inside its band (every partition whose peak |target|
    // approximately fits the band ladder's reach), so the trained
    // joint occupancy prices the stream's actual texture in that
    // band; a stream with no such partition trains on the all-zero
    // vector (the class then simply loses to its neighbours in the RD
    // chooser).
    let design_band_book = |levels: u32, step: f32| -> Result<VorbisCodebook, OggFileError> {
        let d = NOISE_BOOK_DIMS as usize;
        let reach = (levels / 2) as f32 * step;
        let mut corpus: Vec<f32> = Vec::new();
        for (f, t_row) in targets.iter().enumerate() {
            let (end, ps) = (frame_res_end(f), frame_ps(f));
            for target in t_row {
                for partition in target[..end].chunks_exact(ps) {
                    // 1.5×: include partitions the ladder can only
                    // reach approximately — the RD chooser will weigh
                    // the clipping error against the cheap rate.
                    if partition.iter().all(|t| t.abs() <= 1.5 * reach) {
                        corpus.extend_from_slice(partition);
                    }
                }
            }
        }
        let mut corpus = subsample_corpus(corpus, d, VQ_DESIGN_MAX_VECTORS);
        if corpus.is_empty() {
            corpus = vec![0.0; d];
        }
        let ladder = crate::book_design::uniform_value_ladder(-(reach), step, levels, 8)?;
        Ok(crate::book_design::design_lattice_vq_codebook(
            &corpus,
            NOISE_BOOK_DIMS,
            &ladder,
            VQ_DESIGN_MAX_CODEWORD_LEN,
            true,
        )?
        .codebook)
    };
    let noise = design_band_book(
        NOISE_BOOK_LEVELS,
        crate::book_design::pack_nearest(max_abs / 48.0),
    )?;
    // The mid tier: same joint dimensionality, wider ladder — reach
    // `mid_span` (2 of its 5 levels), covering the band between the
    // noise book's reach and the median coarse-class partition.
    let mid = mid_span
        .map(|span| {
            design_band_book(
                MID_BOOK_LEVELS,
                crate::book_design::pack_nearest(span / (MID_BOOK_LEVELS / 2) as f32),
            )
        })
        .transpose()?;

    // ---- the class ladder + setup header ----
    let mut ladder = ResidueLadder::base(coarse, fine, noise);
    if let Some(mid_book) = mid {
        ladder.push_band_class(mid_book);
    }
    let classifications = ladder.classifications();
    let half_ns: Vec<u32> = (0..n_entries).map(|e| entry_half(e) as u32).collect();
    let residue_ends_u32: Vec<u32> = residue_ends.iter().map(|&e| e as u32).collect();
    let mut setup = build_setup(
        floor_headers.clone(),
        ladder,
        &half_ns,
        &residue_ends_u32,
        coupling_steps,
        switching,
    );

    // ---- optional closed-loop codebook training ----
    // The seed value books are retrained on the stream's own residue
    // targets (codeword lengths from usage, reconstruction values at
    // the observed centroids, re-snapped §9.2.2-packable); the
    // trained table replaces the seeds in the setup header and the
    // weighted per-frame planning below runs under it. On a switching
    // stream the short- and long-block corpora train the shared books
    // in ONE combined pass: the two setup entries share the class
    // rows / value books (only `residue_end` and the partition size
    // differ; training plans under the **long** entry's header since
    // long frames carry the bulk of the bits), and a sequential
    // per-size pass would let the second corpus sparse-prune codewords
    // the first corpus' partitions still need — catastrophic for a
    // large joint lattice, where the two block sizes populate
    // different grid cells.
    if config.training_iterations > 0 {
        // The trainer plans under the **weighted** objective the
        // final packet planning below uses — one NMR weight row per
        // corpus residual. Rows cover the §8.6.1 coded band only,
        // truncated to a whole number of training partitions. The
        // training header is the last entry's (the long size on a
        // switching stream), whose partition size can be double a
        // short frame's: a short frame's weight row is coarsened by
        // pairwise max (quantisation error anywhere in the merged
        // span is bounded by its most sensitive half — the same
        // conservative merge the coupling path uses).
        // Under the unweighted trainer the two objectives routed
        // partitions differently, so the trained lengths priced the
        // wrong emissions and sparse pruning deleted entries the
        // weighted plans wanted.
        let train_ps = setup.residues[n_entries - 1].partition_size as usize;
        let mut residuals: Vec<Vec<f32>> = Vec::new();
        let mut train_weights: Vec<Vec<f64>> = Vec::new();
        for (f, (t_row, w_row)) in targets.iter().zip(&weights).enumerate() {
            let keep = (frame_res_end(f) / train_ps) * train_ps;
            if keep == 0 {
                continue;
            }
            let ratio = (train_ps / frame_ps(f)).max(1);
            for (target, w) in t_row.iter().zip(w_row) {
                residuals.push(target[..keep].to_vec());
                train_weights.push(
                    w.chunks(ratio)
                        .take(keep / train_ps)
                        .map(|chunk| chunk.iter().copied().fold(0.0f64, f64::max))
                        .collect(),
                );
            }
        }
        if !residuals.is_empty() {
            let outcome = crate::book_design::train_residue_books_rd_ladder_weighted(
                &residuals,
                &train_weights,
                &setup.residues[n_entries - 1],
                &setup.codebooks,
                tuning.lambda,
                config.training_iterations,
            )?;
            setup.codebooks = outcome.codebooks;
            // The trainer's classword statistics come from whole-size
            // corpus rows planned under the long header; the packets
            // below re-plan per frame (per-size partition geometry).
            // Reset the flat seed classbook — the final classword
            // lengths are trained below from the actual grouped class
            // choices — and take only the trained value books.
            setup.codebooks[1] = class_group_book(classifications, CLASS_GROUP_DIMS);
        }
    }

    // ---- §8.6.2 residue planning (all frames) ----
    // The per-class value-book rows are resolved generically from the
    // setup header's own §8.6.1 `books` table (all entries share the
    // class rows — only `residue_end` / partition size differ), so the
    // rate-distortion chooser prices exactly the ladder the header
    // declares: the base silence / noise / coarse / coarse + fine
    // classes plus, when the corpus separates, the quiet band's pair.
    // Each partition's classword is thereby a per-band value-book
    // assignment, priced per partition per pass.
    let planning_books = setup.codebooks.clone();
    let value_rows: Vec<[Option<&VorbisCodebook>; 8]> = setup.residues[0]
        .books
        .iter()
        .map(|row| {
            let mut resolved: [Option<&VorbisCodebook>; 8] = Default::default();
            for (pass, slot) in row.iter().enumerate() {
                if let Some(book) = slot {
                    resolved[pass] = Some(&planning_books[*book as usize]);
                }
            }
            resolved
        })
        .collect();

    let plan_all = |bias: Option<&[f64]>| -> Result<Vec<Vec<ResidueVectorPlan>>, OggFileError> {
        let mut frame_plans: Vec<Vec<ResidueVectorPlan>> = Vec::with_capacity(frames);
        for f in 0..frames {
            let end = frame_res_end(f);
            let mut plans = Vec::with_capacity(ch);
            for c in 0..ch {
                let scored = match bias {
                    Some(bias) => crate::residue_encode::plan_vector_residue_rd_weighted_biased(
                        &targets[f][c][..end],
                        &value_rows,
                        1,
                        frame_ps(f) as u32,
                        tuning.lambda,
                        &weights[f][c],
                        bias,
                    )?,
                    None => plan_vector_residue_rd_weighted(
                        &targets[f][c][..end],
                        &value_rows,
                        1,
                        frame_ps(f) as u32,
                        tuning.lambda,
                        &weights[f][c],
                    )?,
                };
                plans.push(ResidueVectorPlan {
                    classifications: scored.classifications,
                    partition_entries: scored.partition_entries,
                });
            }
            frame_plans.push(plans);
        }
        Ok(frame_plans)
    };
    // Classword-aware planning: pass 1 prices value bits alone; each
    // refinement pass then prices every class's **marginal classword
    // bits** from the previous pass's class histogram (`-log2 p(c)` —
    // the per-partition share of an entropy-optimal classword under
    // an independence model, which the dense occupancy retrain below
    // approaches) and re-plans under the biased chooser. Without
    // this, a class adopted for a marginal value-bit win can inflate
    // the classword entropy by more than it saves — the mispricing
    // that made a naive richer class ladder spend *more* audio bytes
    // at identical fidelity. Alternating plan ↔ re-price converges
    // like entropy-constrained quantiser design; two refinements are
    // enough for the histogram to stabilise in practice (the loop
    // stops early at a fixed point).
    let mut frame_plans = plan_all(None)?;
    for _ in 0..CLASSWORD_PRICE_PASSES {
        let mut hist = vec![0u64; classifications as usize];
        let mut total = 0u64;
        for plans in &frame_plans {
            for plan in plans {
                for &c in &plan.classifications {
                    hist[c as usize] += 1;
                    total += 1;
                }
            }
        }
        if total == 0 {
            break;
        }
        let bias: Vec<f64> = hist
            .iter()
            .map(|&h| {
                // Unseen classes are floored at one count (adopting
                // one costs a fresh, long classword codeword), and
                // the price is capped so a rare class stays
                // *expensive* rather than unreachable.
                let p = h.max(1) as f64 / total as f64;
                (-p.log2()).clamp(0.0, f64::from(CLASSWORD_PRICE_CAP_BITS))
            })
            .collect();
        let replanned = plan_all(Some(&bias))?;
        if replanned == frame_plans {
            break;
        }
        frame_plans = replanned;
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

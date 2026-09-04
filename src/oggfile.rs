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
//! switching**: the clean-room loudness-adaptive attack detector
//! ([`crate::blocksize::plan_block_sequence_perceptual`] — high-passed
//! sub-frame energy against a post-masking-decayed envelope, with an
//! absolute audibility floor) schedules the short block over attacks
//! (confining quantisation noise to avoid pre-echo) and the long
//! block elsewhere; each long packet's
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
use crate::blocksize::{plan_block_sequence_perceptual, BlocksizeError};
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
use crate::mdct::{mdct_vec, MdctError};
use crate::packet::AudioPacketHeader;
use crate::packet_kind::{classify_packet, ClassifyError, PacketKind};
use crate::psy::{
    complex_spectrum, compute_masking_with_predictability, plan_psy_floor_envelope,
    residue_bin_weights, residue_partition_weights, unpredictability, Complex, MaskingAnalysis,
    PsyConfig, PsyError, TemporalMasking, TemporalMaskingConfig,
};
use crate::quality::{EncoderTuning, QualityError};
use crate::residue_encode::{plan_vector_classifications_rd_bin_weighted, ResidueEncodeError};
use crate::setup::{
    parse_setup_header, Floor1Class, FloorHeader, FloorKind, MappingCouplingStep, MappingHeader,
    MappingSubmap, ModeHeader, ResidueHeader, VorbisSetupHeader,
};
use crate::streaming::{StreamingDecoder, StreamingError, StreamingFrame};
use crate::synthesis::{forward_couple_all, WindowError};
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

/// The quantile of the per-partition peak |target| distribution the
/// residue ladders are spanned to (see the span selection in the
/// geometry core).
const LADDER_SPAN_QUANTILE: f64 = 0.999;

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

/// Dimensionality of the deep band tiers: how many contiguous §8.6.4
/// residue bins one deep-tier codeword covers. Eight divides both
/// §8.6.1 partition sizes ([`PARTITION_SIZE_SHORT`] /
/// [`PARTITION_SIZE_LONG`]), so a partition is exactly two or four
/// deep-tier reads — the next joint-dimensionality rung above the
/// 4-D band books. At this dimensionality only a **ternary** ladder
/// fits a full §3.2.1 product lattice (3⁸ = 6561 entries; the mid
/// tier's five levels would need 5⁸ ≈ 391 k), so both deep tiers are
/// ternary: the deep noise tier at the noise class's step, the deep
/// mid tier at the full mid-band span.
const BAND8_BOOK_DIMS: u16 = 8;

/// The **refinement rungs** of the coarse cascade: `(step divisor,
/// levels)` pairs, each a second-stage lattice book over the coarse
/// stage's leftover at `coarse_step / divisor` with `levels` per
/// dimension, offered as a `coarse + rung` two-stage class. The base
/// ladder's two-stage class refines the coarse leftover with the fine
/// book at `coarse_step / 16` — a 24 dB step above the coarse-only
/// class with nothing in between, so the rate-distortion chooser at
/// the low knob could only *mix* the two (time-sharing on a convex
/// distortion-rate curve lands above the curve: measured on white
/// noise a third of the partitions at ~17 dB and the rest at ~43 dB,
/// 22 dB whole-stream where the curve's own point at that rate sits
/// near 29 dB). The rungs at `/4` (+12 dB over coarse) and `/8`
/// (+18 dB) put operating points on the curve at ~6 dB spacing. Each
/// spans its predecessor's leftover (`±coarse_step / 2`, plus
/// grid-snap slack) with 2× headroom (`levels · step = 2 ·
/// coarse_step`), so no rung clips; 8² = 64 and 16² = 256 entries,
/// sparse-pruned to the cells used. Candidates only: the Lagrangian
/// adoption loop keeps a rung only where it measures smaller.
const REFINEMENT_RUNGS: [(u32, u32); 3] = [(2, 4), (4, 8), (8, 16)];

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
// amplitude-band designer appends up to three band-class candidates
// — the 4-D mid tier and the two 8-D deep tiers — and the
// Lagrangian adoption loop keeps only the candidates that measure
// smaller, so the stream's `residue_classifications` is 4..=7
// depending on the corpus statistics. See `ResidueLadder`.)

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

/// The frequency below which the §4.3.5 angle vector carries its full
/// audibility weight: fine interaural phase is resolved up to roughly
/// this frequency, so a difference-vector error below it is as audible
/// as a magnitude error.
const ANGLE_FULL_WEIGHT_HZ: f32 = 1_500.0;

/// The frequency from which the angle vector's weight reaches its
/// discount floor (log-frequency interpolation in between): above a
/// few kHz the auditory system localises on envelopes, not carrier
/// phase, so error in the difference vector — which perturbs the
/// stereo image, not the sum — is far less audible than the same
/// error in the magnitude vector.
const ANGLE_DISCOUNT_HZ: f32 = 6_000.0;

/// The angle vector's audibility weight at and above
/// [`ANGLE_DISCOUNT_HZ`] at the bottom of the quality knob (`0.25` =
/// 6 dB more difference-vector noise allowed). The discount fades to
/// none (`1.0`) at the top of the knob, where waveform fidelity is
/// what the knob promises.
const ANGLE_DISCOUNT_FLOOR: f32 = 0.25;

/// The angle vector's audibility weight scale at frequency `f_hz`
/// under `tuning` (see [`ANGLE_FULL_WEIGHT_HZ`], [`ANGLE_DISCOUNT_HZ`],
/// [`ANGLE_DISCOUNT_FLOOR`]): `1.0` below the full-weight edge, the
/// knob's discount floor from the discount edge up, log-frequency
/// linear between. The floor rises from [`ANGLE_DISCOUNT_FLOOR`] at
/// `q = 0` to `1.0` at the +6 dB masking-margin cap (`q = 0.75`), so
/// the top of the knob codes the stereo image at waveform fidelity.
fn angle_weight_scale(f_hz: f32, tuning: &EncoderTuning) -> f32 {
    let knob = ((tuning.threshold_offset_db + 12.0) / 18.0).clamp(0.0, 1.0);
    let floor = ANGLE_DISCOUNT_FLOOR + (1.0 - ANGLE_DISCOUNT_FLOOR) * knob;
    if f_hz <= ANGLE_FULL_WEIGHT_HZ {
        1.0
    } else if f_hz >= ANGLE_DISCOUNT_HZ {
        floor
    } else {
        let t =
            (f_hz / ANGLE_FULL_WEIGHT_HZ).ln() / (ANGLE_DISCOUNT_HZ / ANGLE_FULL_WEIGHT_HZ).ln();
        1.0 + (floor - 1.0) * t
    }
}

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
    /// [`EncoderTuning::from_quality`]. Ignored when
    /// [`Self::target_bitrate`] is set.
    pub quality: f32,
    /// **ABR bit targeting**: when `Some(bits_per_second)`, the
    /// encoder solves for the quality whose whole-stream audio-packet
    /// rate fits the budget instead of using [`Self::quality`] — the
    /// bit-targeting entry [`crate::quality::solve_lambda_for_bits`]
    /// describes, run over real whole-stream encodes: the residue
    /// Lagrangian `lambda` is bisected over its `q ∈ [0, 1]` law (each
    /// probe is a full encode, every other lever following the same
    /// `q`), and the returned stream is the highest-fidelity probe
    /// measured within budget. When even `q = 0` overshoots, the
    /// `q = 0` stream is returned (the cheapest the encoder offers).
    /// The identification header's `bitrate_nominal` carries the
    /// target. Whole-stream targeting is deliberate: the encoder is a
    /// whole-stream design, so the "reservoir" is the file itself —
    /// bits flow to the frames that need them under one `lambda`.
    pub target_bitrate: Option<u32>,
    /// The **long** blocksize `blocksize_1` (a power of two in
    /// `64..=8192`, §4.2.2) — the analysis/synthesis size steady
    /// content uses.
    pub blocksize: usize,
    /// The **short** blocksize `blocksize_0` (a power of two in
    /// `64..=8192`, `<=` [`Self::blocksize`], §4.2.2). When strictly
    /// smaller than the long size, the encoder runs §4.3.1 block
    /// switching: the clean-room loudness-adaptive attack detector
    /// ([`crate::blocksize::plan_block_sequence_perceptual`]) schedules
    /// short blocks around attacks (confining quantisation noise to
    /// avoid pre-echo) and long blocks elsewhere, with per-size floors and
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
            target_bitrate: None,
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
/// coarse + fine set; the amplitude-band designer appends the band
/// tier **candidates** — the 4-D mid band book whose ladder reaches
/// the corpus' median above-noise partition, and the two ternary
/// [`BAND8_BOOK_DIMS`]-dimensional deep tiers (the noise step and the
/// full mid span) — giving the rate-distortion chooser a per-band,
/// per-dimensionality value-book assignment: each partition's
/// classword selects the band book whose span **and** joint
/// dimensionality match its texture, priced against the books' exact
/// codeword costs. Candidates that cannot pay for themselves are
/// measured out again by the adoption loop before packets are
/// written.
/// The cascade shape of an appended band-class candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandShape {
    /// A single-pass class carrying the candidate book alone.
    Single,
    /// The candidate book as pass 0, the base fine book as pass 1.
    PlusFine,
    /// The base coarse book as pass 0, the candidate book as pass 1
    /// (a refinement rung, [`REFINEMENT_RUNGS`]).
    AfterCoarse,
}

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

    /// Append one further band class whose pass 0 is `book` and whose
    /// pass 1 is the base ladder's **fine** book (codebook 3): the
    /// coarse + fine cascade over a different pass-0 grid.
    fn push_cascaded_band_class(&mut self, book: VorbisCodebook) {
        let index = (2 + self.value_books.len()) as u8;
        self.value_books.push(book);
        let mut row: [Option<u8>; 8] = Default::default();
        row[0] = Some(index);
        row[1] = Some(3);
        self.cascade.push(0b11);
        self.books.push(row);
    }

    /// Append one further class whose pass 0 is the base ladder's
    /// **coarse** book (codebook 2) and whose pass 1 is `book`: a
    /// refinement rung of the coarse cascade (see
    /// [`REFINEMENT_RUNGS`]).
    fn push_refined_class(&mut self, book: VorbisCodebook) {
        let index = (2 + self.value_books.len()) as u8;
        self.value_books.push(book);
        let mut row: [Option<u8>; 8] = Default::default();
        row[0] = Some(2);
        row[1] = Some(index);
        self.cascade.push(0b11);
        self.books.push(row);
    }

    /// Append a band-class candidate in its declared shape.
    fn push_band(&mut self, book: VorbisCodebook, shape: BandShape) {
        match shape {
            BandShape::Single => self.push_band_class(book),
            BandShape::PlusFine => self.push_cascaded_band_class(book),
            BandShape::AfterCoarse => self.push_refined_class(book),
        }
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
/// band over the whole spectrum, a mapping carrying the gated §4.3.5
/// coupling steps under a single submap, and a mode (`blockflag`
/// clear on the short entry, set on the long one).
///
/// The classbook groups [`CLASS_GROUP_DIMS`] partitions per §8.6.2
/// classword (radix-packed); its seed lengths are uniform and the
/// encode path retrains them occupancy-optimal for the final plans.
/// One §4.2.4 mode of the produced stream: the setup entry (block
/// size) it maps to and the §4.3.5 coupling steps its mapping carries.
/// A stream declares one mode per distinct `(entry, steps)` pair the
/// per-packet coupling election uses, so a packet selects its
/// coupling by selecting its mode.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ModeSpec {
    entry: usize,
    coupling: Vec<MappingCouplingStep>,
}

fn build_setup(
    floor_headers: Vec<crate::setup::Floor1Header>,
    ladder: ResidueLadder,
    half_ns: &[u32],
    residue_ends: &[u32],
    mode_specs: &[ModeSpec],
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
    // One mapping + mode per spec: the mapping carries the spec's
    // coupling steps under a single submap over the entry's floor and
    // residue; the mode selects the entry's block size.
    let mappings = mode_specs
        .iter()
        .map(|spec| MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: spec.coupling.clone(),
            // §4.2.4: the mux table is only present when submaps > 1;
            // with one submap every channel implicitly maps to it.
            mux: Vec::new(),
            submap_configs: vec![MappingSubmap {
                time_placeholder: 0,
                floor: spec.entry as u8,
                residue: spec.entry as u8,
            }],
        })
        .collect();
    let modes = mode_specs
        .iter()
        .enumerate()
        .map(|(m, spec)| ModeHeader {
            blockflag: switching && spec.entry == 1,
            windowtype: 0,
            transformtype: 0,
            mapping: m as u8,
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
/// At the top of the quality knob
/// ([`EncoderTuning::adaptive_margin_headroom_db`] `> 0`, i.e.
/// `q > 0.75`) a multichannel encode is **fidelity-balanced** by
/// measurement: the encoder own-decodes its first pass, and when a
/// channel's measured SNR trails the best channel by more than 3 dB it
/// re-encodes once with a deeper per-channel masking margin for
/// exactly the trailing channels (up to the headroom, scaled by the
/// deficit) — waveform coding where it measurably pays. The retry is
/// kept only when the worst channel actually improves; otherwise the
/// first pass stands. Mono streams and the knob at or below `q = 0.75`
/// are untouched (single pass, byte-identical to the ungated encoder).
///
/// # Errors
///
/// As [`encode_pcm_to_ogg`].
pub fn encode_pcm_to_packets(
    pcm: &[Vec<f32>],
    config: &StreamEncoderConfig,
) -> Result<EncodedVorbisStream, OggFileError> {
    if let Some(bits_per_second) = config.target_bitrate {
        return encode_pcm_to_packets_abr(pcm, config, bits_per_second);
    }
    let tuning = EncoderTuning::from_quality(config.quality)?;
    let first = encode_pcm_to_packets_margined(pcm, config, &[])?;
    let headroom = tuning.adaptive_margin_headroom_db;
    if headroom <= 0.0 || pcm.len() < 2 {
        return Ok(first);
    }
    // Measure where the first pass actually lands per channel.
    let snrs = decoded_per_channel_snr(&first, pcm)?;
    let best = snrs
        .iter()
        .copied()
        .filter(|s| s.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !best.is_finite() {
        return Ok(first);
    }
    // A channel earns extra margin in proportion to its measured
    // deficit against the best channel: nothing within 3 dB, the full
    // headroom at 12 dB behind.
    let extras: Vec<f32> = snrs
        .iter()
        .map(|&s| {
            let deficit = (best - s) as f32;
            headroom * ((deficit - 3.0) / 9.0).clamp(0.0, 1.0)
        })
        .collect();
    if extras.iter().all(|&e| e <= 0.01) {
        return Ok(first);
    }
    let retry = encode_pcm_to_packets_margined(pcm, config, &extras)?;
    let retry_snrs = decoded_per_channel_snr(&retry, pcm)?;
    let min_of = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    // Keep the retry only when the measured worst channel genuinely
    // improves — "deepens only where waveform coding pays" is checked
    // against the stream itself, not assumed.
    Ok(if min_of(&retry_snrs) > min_of(&snrs) + 0.5 {
        retry
    } else {
        first
    })
}

/// The ABR entry under [`encode_pcm_to_packets`] (see
/// [`StreamEncoderConfig::target_bitrate`]): bisect the residue
/// Lagrangian over its quality law with
/// [`crate::quality::solve_lambda_for_bits`], measuring each probe as
/// a full quality-mode encode's audio-packet bits, and return the
/// highest-fidelity stream measured within the budget (the cheapest
/// probe when even `q = 0` overshoots). The mapping `q(λ)` inverts
/// the [`EncoderTuning::from_quality`] lambda law, so every other
/// lever follows the probe's quality coherently.
fn encode_pcm_to_packets_abr(
    pcm: &[Vec<f32>],
    config: &StreamEncoderConfig,
    bits_per_second: u32,
) -> Result<EncodedVorbisStream, OggFileError> {
    if pcm.is_empty() || pcm[0].is_empty() || config.sample_rate == 0 {
        // Let the quality-mode validation produce the precise error.
        let mut probe = config.clone();
        probe.target_bitrate = None;
        return encode_pcm_to_packets(pcm, &probe);
    }
    let seconds = pcm[0].len() as f64 / f64::from(config.sample_rate);
    let target_bits = (f64::from(bits_per_second) * seconds) as u64;
    // The from_quality lambda law, invertible: λ = 10^(−1.4 − 2.6 q).
    let q_of = |lambda: f64| ((-1.4 - lambda.log10()) / 2.6).clamp(0.0, 1.0) as f32;
    let lambda_of = |q: f64| 10f64.powf(-1.4 - 2.6 * q);
    // Every probe is kept so the returned stream is exactly the one
    // the solver measured — no re-encode, no drift.
    let mut probes: Vec<(u64, EncodedVorbisStream)> = Vec::new();
    let mut encode_at = |lambda: f64| -> Result<u64, OggFileError> {
        let mut probe = config.clone();
        probe.target_bitrate = None;
        probe.quality = q_of(lambda);
        let stream = encode_pcm_to_packets(pcm, &probe)?;
        let bits = 8 * stream
            .audio
            .iter()
            .map(|(p, _)| p.len() as u64)
            .sum::<u64>();
        probes.push((bits, stream));
        Ok(bits)
    };
    // λ(q=1) is the expensive end, λ(q=0) the cheap end; six halvings
    // resolve the knob to ~1.5 % of its range.
    let solution = crate::quality::solve_lambda_for_bits(
        target_bits,
        lambda_of(1.0),
        lambda_of(0.0),
        6,
        &mut encode_at,
    )
    .map_err(|e| match e {
        crate::quality::LambdaSolveError::Rate(inner) => inner,
        // The bracket and iteration count are fixed above.
        other => OggFileError::Header(other.to_string()),
    })?;
    let position = probes
        .iter()
        .position(|(bits, _)| *bits == solution.bits)
        .expect("the solution's rate was measured on one of the probes");
    let mut stream = probes.swap_remove(position).1;
    // Stamp the target as the nominal bitrate (§4.2.2: purely
    // informational fields).
    let id = write_identification_header(&VorbisIdentificationHeader {
        vorbis_version: 0,
        audio_channels: config.channels,
        audio_sample_rate: config.sample_rate,
        bitrate_maximum: 0,
        bitrate_nominal: bits_per_second as i32,
        bitrate_minimum: 0,
        blocksize_0: stream.short_blocksize as u16,
        blocksize_1: stream.blocksize as u16,
    })?;
    stream.identification = id;
    Ok(stream)
}

/// The single-margin-set encode under [`encode_pcm_to_packets`]:
/// `extra_margins[c]` (empty = all zero) deepens channel `c`'s masking
/// margin past the tuning's global `threshold_offset_db`. Runs the
/// top-band geometry race when the knob sits past the lattice fine
/// ladder's coverage cap.
fn encode_pcm_to_packets_margined(
    pcm: &[Vec<f32>],
    config: &StreamEncoderConfig,
    extra_margins: &[f32],
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
        let scalar = encode_pcm_to_packets_geometry(pcm, config, &tuning, false, extra_margins)?;
        let joint = encode_pcm_to_packets_geometry(pcm, config, &seam_tuning, true, extra_margins)?;
        let scalar_snr = decoded_stream_snr(&scalar, pcm)?;
        let joint_snr = decoded_stream_snr(&joint, pcm)?;
        let scalar_bytes: usize = scalar.audio.iter().map(|(p, _)| p.len()).sum();
        let joint_bytes: usize = joint.audio.iter().map(|(p, _)| p.len()).sum();
        let keep_scalar =
            scalar_snr > joint_snr || (scalar_snr == joint_snr && scalar_bytes <= joint_bytes);
        return Ok(if keep_scalar { scalar } else { joint });
    }
    encode_pcm_to_packets_geometry(pcm, config, &tuning, config.vq_dims > 1, extra_margins)
}

/// Own-decode a packet stream back to per-channel PCM rows — the
/// measurement half of the encoder's self-checks (the top-band
/// geometry selector and the adaptive-margin balance pass).
fn own_decode_stream(stream: &EncodedVorbisStream) -> Result<Vec<Vec<f32>>, OggFileError> {
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
    Ok(decoded)
}

/// `10·log10(Σ signal² / Σ error²)` between one reference row and its
/// decode (over the overlapping prefix). `+inf` for a bit-exact (or
/// all-zero) row.
fn row_snr_db(reference: &[f32], out: &[f32]) -> f64 {
    let n = reference.len().min(out.len());
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&r, &d) in reference[..n].iter().zip(&out[..n]) {
        sig += f64::from(r) * f64::from(r);
        let e = f64::from(r) - f64::from(d);
        err += e * e;
    }
    if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sig / err).log10()
    }
}

/// Own-decode a packet stream and report the whole-stream SNR (dB)
/// against the input PCM (`10·log10(Σ signal² / Σ error²)` across all
/// channels) — the top-band geometry selector's ground truth.
fn decoded_stream_snr(stream: &EncodedVorbisStream, pcm: &[Vec<f32>]) -> Result<f64, OggFileError> {
    let decoded = own_decode_stream(stream)?;
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

/// Per-channel variant of [`decoded_stream_snr`] — the adaptive-margin
/// balance pass's measurement.
fn decoded_per_channel_snr(
    stream: &EncodedVorbisStream,
    pcm: &[Vec<f32>],
) -> Result<Vec<f64>, OggFileError> {
    let decoded = own_decode_stream(stream)?;
    Ok(pcm
        .iter()
        .zip(&decoded)
        .map(|(reference, out)| row_snr_db(reference, out))
        .collect())
}

/// Fit one frame's floor-1 posts to `envelope` so the **rendered** curve
/// covers it: the plain post fit ([`plan_floor1_envelope`]) samples the
/// envelope at the post x-coordinates only, and between posts the
/// dB-linear segment can run well under a spectral line the posts do
/// not sit on — the post next to a masked-region neighbour rides the
/// masking threshold, so the segment through the peak bin drops
/// with the threshold. That undershoot is what set the residue
/// ladders' span: measured on the tonal battery, a +1.2 dB threshold
/// margin step scaled every residue target by 1.5× (the peak bins'
/// `X / floor` blew up as their floor segment sank) and cost 5 dB of
/// whole-stream SNR while spending *more* bits — a cliff in the
/// quality knob. The covering fit re-renders after the sample fit
/// and, wherever the envelope still exceeds the curve by more than
/// [`FLOOR_COVER_TOLERANCE_DB`], lifts the two posts bounding that bin
/// by the deficit (in §10.1 ladder steps), iterating to a fixed point
/// (at most [`FLOOR_COVER_PASSES`] passes; the lifts are monotone, so
/// the loop terminates at the post range if nothing else). Lifting a
/// post can only shrink the residue targets on its segment, so the
/// covered fit never widens the ladder and the quiet neighbours of a
/// peak quantise toward zero a little more cheaply.
///
/// Returns the packed `floor1_y` posts and the rendered curve.
fn fit_covering_floor(
    envelope: &[f32],
    header: &crate::setup::Floor1Header,
    decoder: &Floor1Decoder,
    half: usize,
) -> Result<(Vec<u32>, Vec<f32>), OggFileError> {
    let mut posts = plan_floor1_envelope(envelope, header)?;
    let x_list = crate::floor1_encode::full_x_list(header);
    // Posts in ascending-x order (the header's list is partition-tiled;
    // `full_x_list` is endpoints first, then the header order).
    let mut order: Vec<usize> = (0..x_list.len()).collect();
    order.sort_by_key(|&i| x_list[i]);
    let range = crate::floor1_envelope::floor1_post_range(header.multiplier) as i32;
    let multiplier = header.multiplier as i32;
    let mut floor1_y = plan_floor1_y(&posts, header)?;
    let mut rendered = decoder.render_curve(&floor1_y, half);
    let tolerance = 10.0f32.powf(FLOOR_COVER_TOLERANCE_DB / 20.0);
    for _ in 0..FLOOR_COVER_PASSES {
        let mut lifted = false;
        for seg in order.windows(2) {
            let (i0, i1) = (seg[0], seg[1]);
            let x0 = x_list[i0] as usize;
            let x1 = (x_list[i1] as usize).min(half);
            if x0 >= x1 {
                continue;
            }
            // The worst deficit on this segment, in §10.1 ladder steps
            // of the post grid (a post step is `multiplier` ladder
            // steps): the lift both bounding posts need.
            let mut need = 0i32;
            for x in x0..x1.min(envelope.len()) {
                let env = envelope[x];
                let cur = rendered[x];
                if env > cur * tolerance && cur > 0.0 {
                    let want = i32::from(crate::floor1_envelope::invert_inverse_db(env));
                    let have = i32::from(crate::floor1_envelope::invert_inverse_db(cur));
                    need = need.max((want - have + multiplier - 1) / multiplier);
                }
            }
            if need > 0 {
                for &i in &[i0, i1] {
                    let lifted_post = (posts[i] + need).min(range - 1);
                    if lifted_post != posts[i] {
                        posts[i] = lifted_post;
                        lifted = true;
                    }
                }
            }
        }
        if !lifted {
            break;
        }
        floor1_y = plan_floor1_y(&posts, header)?;
        rendered = decoder.render_curve(&floor1_y, half);
    }
    Ok((floor1_y, rendered))
}

/// How far (dB) the rendered floor may sit under the envelope before
/// the covering fit lifts the bounding posts.
const FLOOR_COVER_TOLERANCE_DB: f32 = 0.5;

/// Maximum lift-and-re-render passes of the covering floor fit.
const FLOOR_COVER_PASSES: usize = 4;

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
    extra_margins: &[f32],
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
    // energy-envelope transient detector run **per channel** and
    // OR-merged (`plan_block_sequence_multi`): the packet's blockflag
    // is shared by every channel, so it goes short when any channel's
    // lookahead region is transient. (The old channel-mixdown detector
    // missed a burst confined to one channel: a loud steady sibling
    // dilutes the mix's peak-to-mean concentration, and
    // anti-correlated content cancels in the mix outright.) The
    // granule positions come from the §4.3.8 walk `(n_prev + n_cur)/4`
    // per packet. A single-blocksize stream is the uniform walk: the
    // priming packet plus one packet per n/2 finished samples.
    let (flags, granules) = if switching {
        let rows: Vec<&[f32]> = pcm.iter().map(|row| row.as_slice()).collect();
        let plan = plan_block_sequence_perceptual(&rows, n0, n1, config.sample_rate)?;
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
    // (`mode_number` is finalised once the per-packet coupling election
    // below has fixed the stream's mode list.)
    let mut headers: Vec<AudioPacketHeader> = (0..frames)
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
                                                                      // Per-frame, per-channel phase-predictability figure for the
                                                                      // masking model's tonality estimate (`psy::unpredictability`):
                                                                      // the windowed block's complex spectrum against the linear
                                                                      // extrapolation of the two preceding frames' — defined only where
                                                                      // the last three frames share one block size (across a §4.3.1
                                                                      // switch the model falls back to spectral flatness). Only a
                                                                      // two-deep complex history per channel is kept.
    let mut unpred: Vec<Vec<Option<Vec<f32>>>> = Vec::with_capacity(frames);
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
        // (prev1, prev2) complex spectra per channel.
        type History = (Option<Vec<Complex>>, Option<Vec<Complex>>);
        let mut history: Vec<History> = vec![(None, None); ch];
        for f in 0..frames {
            let mut per_ch = Vec::with_capacity(ch);
            let mut u_row = Vec::with_capacity(ch);
            for (splitter, hist) in splitters.iter_mut().zip(history.iter_mut()) {
                // Apply the pending §4.3.8 stride between differing
                // block sizes, then slice; take_frame applies the
                // §4.3.1 analysis window, so the bare kernel follows.
                splitter.advance_pending_stride(sizes[f]);
                let block = splitter.take_frame(sizes[f], &windows[window_of[f]])?;
                per_ch.push(mdct_vec(&block, 4.0 / sizes[f] as f32)?);
                let cplx = complex_spectrum(&block)?;
                let u = match (&hist.0, &hist.1) {
                    (Some(p1), Some(p2)) if p1.len() == cplx.len() && p2.len() == cplx.len() => {
                        Some(unpredictability(&cplx, p1, p2)?)
                    }
                    _ => None,
                };
                u_row.push(u);
                hist.1 = hist.0.take();
                hist.0 = Some(cplx);
            }
            spectra.push(per_ch);
            unpred.push(u_row);
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
    // A channel may carry an extra content-adaptive margin (the
    // measured second-pass grant — see `encode_pcm_to_packets`); the
    // common case is no extras and every channel sharing one config.
    let psy_config_for = |c: usize| PsyConfig {
        threshold_offset_db: tuning.threshold_offset_db
            + extra_margins.get(c).copied().unwrap_or(0.0),
        ..PsyConfig::new(config.sample_rate)
    };
    let uniform_sizes = sizes.windows(2).all(|w| w[0] == w[1]);
    let mut maskings: Vec<Vec<MaskingAnalysis>> = vec![Vec::with_capacity(ch); frames];
    if uniform_sizes {
        for c in 0..ch {
            let psy_config = psy_config_for(c);
            let mut temporal =
                TemporalMasking::new(&TemporalMaskingConfig::new(sizes[0] / 2), &psy_config)?;
            let mut emitted = 0usize;
            for (per_ch, u_row) in spectra.iter().zip(&unpred) {
                if let Some(analysis) = temporal.push_frame_with_predictability(
                    &per_ch[c],
                    u_row[c].as_deref(),
                    &psy_config,
                )? {
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
            for (c, x) in per_ch.iter().enumerate() {
                maskings[f].push(compute_masking_with_predictability(
                    x,
                    unpred[f][c].as_deref(),
                    &psy_config_for(c),
                )?);
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
    // §8.6.1 `[residue_end]` per setup entry: the whole spectrum. The
    // old 20 kHz coded-band fence is gone — it was measured as a hard
    // 12 dB SNR ceiling on wideband noise (6 % of a 44.1 kHz
    // spectrum's bins sit above 20 kHz, and wideband-noise energy
    // there is real signal energy the black-box reference encoder
    // carries at the same rates: removing the fence alone took the
    // white-noise battery from 12.0 dB to 32.2 dB at mid quality).
    // What the fence saved is saved better by the rate-distortion
    // chooser itself: partitions the masking model prices at or
    // under the threshold in quiet go to the silence class for a
    // couple of grouped-classword bits.
    let residue_ends: Vec<usize> = (0..n_entries).map(entry_half).collect();
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
    // Per-bin audibility weights for the masking-weighted VQ
    // selection (the per-partition rows above are their means: the
    // trainer and the band-adoption tallies work per partition).
    let mut bin_weights: Vec<Vec<Vec<f64>>> = Vec::with_capacity(frames);
    // ---- §4.3.5 channel coupling: per-packet, masking-driven ----
    // The §4.2.4 mapping fixes the coupling steps for every packet
    // using it, so a per-packet choice needs one mapping (and mode)
    // per distinct step set the stream uses: the election here picks
    // each frame's step set, and the setup header declares exactly
    // the `(block size, step set)` pairs that occur — a stream that
    // never couples carries no coupled mapping, one that always does
    // carries no uncoupled one.
    //
    // A coupled pair codes under **one shared floor** (the two
    // channels' envelopes' maximum, fitted once and carried by both):
    // the §4.3.5 transform acts on the floor-normalised residues, so
    // under two independently fitted floors an identical signal in
    // both channels still leaves a difference vector — the floors'
    // mismatch — and the pair reads decorrelated (measured: a mid +
    // small-side tone pair elected coupling on 6 of 23 frames under
    // its own floors). Under the shared floor the difference vector
    // is the true side signal.
    //
    // Each candidate pair `(2p, 2p+1)` is elected per frame on the
    // **audible energy to code**: the bin-weighted energy of the pair's
    // own-floor residue targets left uncoupled, `Σ_k w_L·L² + w_R·R²`,
    // against the shared-floor targets coupled, `Σ_k w·(M² + s_k·A²)`
    // with `w = max(w_L, w_R)` (error in either coupled vector reaches
    // both outputs) and `s_k` the angle audibility scale
    // ([`angle_weight_scale`]: fine interaural phase is inaudible
    // above ~1.5 kHz, so error in the difference vector weighs less up
    // there — the point-stereo discount, faded out toward the top of
    // the knob). A correlated pair couples (the angle quantises toward
    // zero cheaply); an independent or anti-phase pair stays
    // uncoupled (coupling would only move energy around).
    let pairs: Vec<(usize, usize)> = if config.coupling && ch >= 2 {
        (0..ch / 2).map(|p| (2 * p, 2 * p + 1)).collect()
    } else {
        Vec::new()
    };
    let mut frame_steps: Vec<Vec<MappingCouplingStep>> = Vec::with_capacity(frames);
    for f in 0..frames {
        let e = entry_of(f);
        let half = sizes[f] / 2;
        let end = frame_res_end(f);
        let ps = frame_ps(f);
        let mut y_row = Vec::with_capacity(ch);
        let mut t_row = Vec::with_capacity(ch);
        let mut w_row = Vec::with_capacity(ch);
        let mut bw_row = Vec::with_capacity(ch);
        for c in 0..ch {
            let (floor1_y, rendered) = fit_covering_floor(
                &envelopes[f][c],
                &floor_headers[e],
                &floor_decoders[e],
                half,
            )?;
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
            let bw = residue_bin_weights(&rendered, &maskings[f][c], 0, frame_res_end(f))?;
            y_row.push(floor1_y);
            t_row.push(target);
            w_row.push(w);
            bw_row.push(bw);
        }
        let mut steps = Vec::new();
        for &(mag, ang) in &pairs {
            let shared_env: Vec<f32> = envelopes[f][mag]
                .iter()
                .zip(&envelopes[f][ang])
                .map(|(&a, &b)| a.max(b))
                .collect();
            let (y_s, rendered_s) =
                fit_covering_floor(&shared_env, &floor_headers[e], &floor_decoders[e], half)?;
            let target_s = |c: usize| -> Vec<f32> {
                spectra[f][c]
                    .iter()
                    .zip(&rendered_s)
                    .map(|(&xv, &fv)| xv / fv)
                    .collect()
            };
            let (t_mag, t_ang) = (target_s(mag), target_s(ang));
            let bw_mag = residue_bin_weights(&rendered_s, &maskings[f][mag], 0, end)?;
            let bw_ang = residue_bin_weights(&rendered_s, &maskings[f][ang], 0, end)?;
            // Bits-like measure: `log2(1 + w·t²)` per bin — the rate a
            // bin needs to bring its weighted error under the mask
            // grows with the log of its audible energy, and a sum of
            // logs counts every bin instead of letting the few most
            // audible bins (weights reach 10⁴) decide alone. (Measured:
            // under a plain weighted-energy sum a mid + small-side pair
            // read *uncoupled* — the side tone's anti-phase bins cost
            // 2.5× under coupling and outweighed the hundreds of
            // identical mid bins that halve.)
            let bits = |w: f64, t: f32| (1.0 + w * f64::from(t * t)).log2();
            let mut uncoupled = 0.0f64;
            let mut coupled = 0.0f64;
            for k in 0..end {
                let (l, r) = (t_row[mag][k], t_row[ang][k]);
                uncoupled += bits(bw_row[mag][k], l) + bits(bw_row[ang][k], r);
                let (m, a) = crate::synthesis::forward_couple_scalar(t_mag[k], t_ang[k]);
                let scale = angle_weight_scale(
                    (k as f32 + 0.5) * config.sample_rate as f32 / (2.0 * half as f32),
                    tuning,
                );
                let w = bw_mag[k].max(bw_ang[k]);
                coupled += bits(w, m) + bits(w * f64::from(scale), a);
            }
            if coupled <= uncoupled {
                steps.push(MappingCouplingStep {
                    magnitude_channel: mag as u8,
                    angle_channel: ang as u8,
                });
                let means = |bw: &[f64]| -> Vec<f64> {
                    bw.chunks(ps)
                        .map(|chunk| chunk.iter().sum::<f64>() / chunk.len() as f64)
                        .collect()
                };
                w_row[mag] = means(&bw_mag);
                w_row[ang] = means(&bw_ang);
                bw_row[mag] = bw_mag;
                bw_row[ang] = bw_ang;
                t_row[mag] = t_mag;
                t_row[ang] = t_ang;
                y_row[mag] = y_s.clone();
                y_row[ang] = y_s;
            }
        }
        frame_steps.push(steps);
        floor_ys.push(y_row);
        targets.push(t_row);
        weights.push(w_row);
        bin_weights.push(bw_row);
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
    // Elected steps are forward-coupled here — the residue planner
    // below quantises magnitude/angle vectors — and the decoder's
    // §4.3.5 inverse coupling undoes the transform after residue
    // decode, before the §4.3.6 floor multiply.
    // Per frame, per channel: is the channel an angle vector?
    let mut is_angle: Vec<Vec<bool>> = vec![vec![false; ch]; frames];
    for (f, ((t_row, w_row), bw_row)) in targets
        .iter_mut()
        .zip(weights.iter_mut())
        .zip(bin_weights.iter_mut())
        .enumerate()
    {
        let half = sizes[f] / 2;
        let steps = &frame_steps[f];
        if !steps.is_empty() {
            forward_couple_all(t_row, steps)
                .expect("coupling steps are constructed in range with distinct channels");
            // Merge each coupled pair's NMR weights (per partition and
            // per bin) to the element-wise max: quantisation error in
            // either coupled vector spreads into both output channels
            // through the inverse coupling, so the more sensitive
            // channel's audibility bound governs both — the angle
            // vector's additionally scaled by the interaural-phase
            // audibility of its frequency.
            for step in steps {
                let (mag, ang) = (step.magnitude_channel as usize, step.angle_channel as usize);
                is_angle[f][ang] = true;
                let ps = frame_ps(f);
                for k in 0..bw_row[mag].len() {
                    let w = bw_row[mag][k].max(bw_row[ang][k]);
                    let scale = angle_weight_scale(
                        (k as f32 + 0.5) * config.sample_rate as f32 / (2.0 * half as f32),
                        tuning,
                    );
                    bw_row[mag][k] = w;
                    bw_row[ang][k] = w * f64::from(scale);
                }
                for p in 0..w_row[mag].len() {
                    let lo = p * ps;
                    let hi = (lo + ps).min(bw_row[mag].len());
                    w_row[mag][p] = bw_row[mag][lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
                    w_row[ang][p] = bw_row[ang][lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
                }
            }
        }
    }
    // The stream's mode list: one `(entry, steps)` per distinct pair
    // used, entries first, the most-coupled variant of each entry
    // first. Every entry keeps at least one mode (an unused block
    // size still needs a legal mode).
    let mut mode_specs: Vec<ModeSpec> = Vec::new();
    for e in 0..n_entries {
        let mut variants: Vec<Vec<MappingCouplingStep>> = Vec::new();
        for (f, steps) in frame_steps.iter().enumerate() {
            if entry_of(f) == e && !variants.contains(steps) {
                variants.push(steps.clone());
            }
        }
        if variants.is_empty() {
            variants.push(Vec::new());
        }
        let key = |steps: &Vec<MappingCouplingStep>| -> Vec<(u8, u8)> {
            steps
                .iter()
                .map(|s| (s.magnitude_channel, s.angle_channel))
                .collect()
        };
        variants.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| key(a).cmp(&key(b))));
        for coupling in variants {
            mode_specs.push(ModeSpec { entry: e, coupling });
        }
    }
    for (f, header) in headers.iter_mut().enumerate() {
        let spec = ModeSpec {
            entry: entry_of(f),
            coupling: frame_steps[f].clone(),
        };
        header.mode_number = mode_specs
            .iter()
            .position(|s| *s == spec)
            .expect("every frame's (entry, steps) pair is declared")
            as u32;
    }
    let any_coupling = mode_specs.iter().any(|s| !s.coupling.is_empty());

    // ---- per-partition peak statistics over the coded band ----
    // Everything downstream — the ladder spans, the amplitude-band
    // split, the band design corpora — is driven by the peak |target|
    // of each §8.6 partition inside the coded band (bins past
    // `residue_end` are never coded, so a loud ultrasonic bin must not
    // widen any ladder).
    // Angle channels are left out of the span statistics: the §4.3.5
    // angle vector is the *difference* of two floor-normalised
    // residues, so where a pair is anti-phase it reaches 2× the
    // magnitude vector's range by construction (a side component the
    // covering floor pins at exactly ±1 in each channel reads ±2 in
    // the angle). Spanning the shared ladders to that doubles every
    // step for every channel — measured as a 6 dB coupled-vs-dual-mono
    // fidelity loss on a mid + side tone pair. Spanned to the
    // magnitude / uncoupled vectors instead, an anti-phase angle
    // partition clips against the coarse reach (`±1.33·span`, an error
    // the rate-distortion chooser prices) while everything else keeps
    // its grid.
    let mut max_abs = 0.0f32;
    let mut partition_peaks: Vec<f32> = Vec::new();
    let mut span_peaks: Vec<f32> = Vec::new();
    for (f, t_row) in targets.iter().enumerate() {
        let (end, ps) = (frame_res_end(f), frame_ps(f));
        for (c, target) in t_row.iter().enumerate() {
            for part in target[..end].chunks_exact(ps) {
                let peak = part.iter().fold(0.0f32, |m, &t| m.max(t.abs()));
                partition_peaks.push(peak);
                if !is_angle[f][c] {
                    span_peaks.push(peak);
                }
                max_abs = max_abs.max(peak);
            }
        }
    }
    // The ladder span: a high quantile of the partition peaks, not
    // the absolute maximum. A stream's very loudest targets are
    // floor-fit undershoots (a spectral line the post budget could
    // not pin between posts) — a handful of outliers up to 6× the
    // 99.9th-percentile peak on the noise battery — and sizing every
    // ladder to them coarsens every step in the stream. At the
    // 99.9th percentile the rare outlier partition clips against the
    // coarse book's reach (the rate-distortion chooser prices that
    // clip like any other error) while every other partition is
    // quantised on a materially finer grid: measured +1…+5 dB
    // whole-stream at equal-or-lower rate across the staged corpus,
    // no fixture worse.
    if !span_peaks.is_empty() {
        span_peaks.sort_unstable_by(f32::total_cmp);
        max_abs = span_peaks[((span_peaks.len() - 1) as f64 * LADDER_SPAN_QUANTILE) as usize];
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
    let (coarse, fine, half_geometry, rungs) = if joint_geometry {
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
        type DesignedPair = (VorbisCodebook, VorbisCodebook, Vec<VorbisCodebook>);
        let design_pair = |corpus: Vec<f32>, span: f32| -> Result<DesignedPair, OggFileError> {
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
            // The refinement rungs: second-stage lattices over the same
            // coarse leftover at the rung steps ([`REFINEMENT_RUNGS`]).
            let mut rungs = Vec::with_capacity(REFINEMENT_RUNGS.len());
            if config.residue_bands {
                for &(divisor, levels) in &REFINEMENT_RUNGS {
                    let step = crate::book_design::pack_nearest(coarse_step / divisor as f32);
                    let ladder = crate::book_design::uniform_value_ladder(
                        -(levels as f32 / 2.0) * step,
                        step,
                        levels,
                        8,
                    )?;
                    rungs.push(
                        crate::book_design::design_lattice_vq_codebook(
                            &leftovers,
                            config.vq_dims,
                            &ladder,
                            VQ_DESIGN_MAX_CODEWORD_LEN,
                            true,
                        )?
                        .codebook,
                    );
                }
            }
            Ok((coarse, fine, rungs))
        };
        let (coarse, fine, rungs) = design_pair(raw, max_abs)?;
        // The half-span tier's geometry (designed below, once the band
        // designer exists): the coarse step and level count.
        let half_geometry = config.residue_bands.then(|| {
            (
                crate::book_design::pack_nearest(8.0 * max_abs / (3.0 * lv as f32)),
                lv,
            )
        });
        (coarse, fine, half_geometry, rungs)
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
            None,
            Vec::new(),
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
    let design_band_book =
        |dims: u16, levels: u32, step: f32| -> Result<VorbisCodebook, OggFileError> {
            let d = dims as usize;
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
                dims,
                &ladder,
                VQ_DESIGN_MAX_CODEWORD_LEN,
                true,
            )?
            .codebook)
        };
    let noise = design_band_book(
        NOISE_BOOK_DIMS,
        NOISE_BOOK_LEVELS,
        crate::book_design::pack_nearest(max_abs / 48.0),
    )?;
    // The mid tier: same joint dimensionality, wider ladder — reach
    // `mid_span` (2 of its 5 levels), covering the band between the
    // noise book's reach and the median coarse-class partition.
    let mid = mid_span
        .map(|span| {
            design_band_book(
                NOISE_BOOK_DIMS,
                MID_BOOK_LEVELS,
                crate::book_design::pack_nearest(span / (MID_BOOK_LEVELS / 2) as f32),
            )
        })
        .transpose()?;
    // The deep 8-D tiers (§8.6.2 partition/dimension interplay: 8
    // divides both partition sizes, so one codeword covers eight
    // contiguous §8.6.4 bins — twice the noise class's joint span).
    // A ternary 8-D grid is the only full-lattice shape that fits
    // §3.2.1's entry space at this dimensionality (3⁸ = 6561 entries;
    // five levels would need 5⁸ ≈ 391 k): each tier trades amplitude
    // resolution for joint dimensionality one step further than its
    // 4-D sibling, which pays exactly where partitions are *textured
    // but patterned* — the trained joint occupancy prices the
    // stream's actual eight-bin patterns, and the sparse
    // final-emission retrain keeps the setup table proportional to
    // the cells actually used. Both tiers are candidates only; the
    // Lagrangian adoption below drops any tier that cannot buy its
    // own setup table + classword-alphabet growth.
    let noise8 = if config.residue_bands {
        Some(design_band_book(
            BAND8_BOOK_DIMS,
            NOISE_BOOK_LEVELS,
            crate::book_design::pack_nearest(max_abs / 48.0),
        )?)
    } else {
        None
    };
    let mid8 = if config.residue_bands {
        mid_span
            .map(|span| {
                design_band_book(
                    BAND8_BOOK_DIMS,
                    NOISE_BOOK_LEVELS,
                    crate::book_design::pack_nearest(span),
                )
            })
            .transpose()?
    } else {
        None
    };

    // ---- the class ladder + setup header ----
    // The appended band-class candidates, in ladder order after the
    // four base classes. Candidacy is cheap: each candidate is
    // adopted only if the exact post-training serialisation cost
    // (setup table + classwords + value codewords) measures smaller
    // with it than without it — see the adoption loop below.
    // `(book, cascaded)`: a single-pass band class, or (cascaded) the
    // book as pass 0 with the base fine book as pass 1.
    let mut band_candidates: Vec<(VorbisCodebook, BandShape)> = Vec::new();
    if let Some(book) = mid {
        band_candidates.push((book, BandShape::Single));
    }
    if let Some(book) = noise8 {
        band_candidates.push((book, BandShape::Single));
    }
    if let Some(book) = mid8 {
        band_candidates.push((book, BandShape::Single));
    }
    // The coarse cascade's refinement rungs (see [`REFINEMENT_RUNGS`]).
    for book in rungs {
        band_candidates.push((book, BandShape::AfterCoarse));
    }
    // The half-span tier: the coarse geometry (same dimensionality and
    // level count) over half the span, so a partition whose peak fits
    // inside `±0.66·span` is quantised on a grid twice as fine as the
    // coarse book's for the same codeword alphabet. It fills the gap
    // the two-stage cascade leaves between the coarse class (measured
    // 17 dB at 3.4 bits/bin on white noise) and coarse + fine (43 dB
    // at 7.9 bits/bin): with no intermediate operating point the
    // rate-distortion chooser mixes the two and the low half of the
    // knob lands a third of a noise stream's partitions at 17 dB. A
    // candidate only — adopted below if it pays (measured +2.4 dB on
    // the white-noise battery and +2.5 dB on correlated stereo at mid
    // quality, measured out on the tonal corpora). Quarter- and
    // eighth-span tiers were measured and rejected (never adopted).
    if let Some((coarse_step, levels)) = half_geometry {
        band_candidates.push((
            design_band_book(
                config.vq_dims,
                levels,
                crate::book_design::pack_nearest(coarse_step / 2.0),
            )?,
            BandShape::Single,
        ));
        // The wide cascade tier, offered on coupled streams: the
        // coarse geometry over **twice** the span as pass 0, the base
        // fine book as pass 1. The §4.3.5 angle vector reaches 2× the
        // magnitude range where a pair is anti-phase (a side component
        // pinned at ±1 by each channel's covering floor reads ±2 in
        // the angle), and the ladders are spanned to the magnitude /
        // uncoupled vectors so every other partition keeps its grid —
        // this class is where the anti-phase angle partitions go at
        // full fine resolution instead of clipping against the coarse
        // reach. (The wide book's step is twice the coarse step, so
        // its leftover fits exactly inside the fine ladder's
        // ±coarse-step span.) Adoption keeps it only if it pays.
        if any_coupling {
            band_candidates.push((
                design_band_book(
                    config.vq_dims,
                    levels,
                    crate::book_design::pack_nearest(coarse_step * 2.0),
                )?,
                BandShape::PlusFine,
            ));
        }
    }
    let classifications = 4 + band_candidates.len() as u32;
    let half_ns: Vec<u32> = (0..n_entries).map(|e| entry_half(e) as u32).collect();
    let residue_ends_u32: Vec<u32> = residue_ends.iter().map(|&e| e as u32).collect();
    let build = |coarse: &VorbisCodebook,
                 fine: &VorbisCodebook,
                 noise: &VorbisCodebook,
                 bands: &[(VorbisCodebook, BandShape)]|
     -> VorbisSetupHeader {
        let mut ladder = ResidueLadder::base(coarse.clone(), fine.clone(), noise.clone());
        for (book, shape) in bands {
            ladder.push_band(book.clone(), *shape);
        }
        build_setup(
            floor_headers.clone(),
            ladder,
            &half_ns,
            &residue_ends_u32,
            &mode_specs,
            switching,
        )
    };
    let mut setup = build(&coarse, &fine, &noise, &band_candidates);

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

    // ---- §8.6.2 residue planning + band-class adoption ----
    // `evaluate` runs the whole planning tail for one candidate band
    // set: classword-aware rate-distortion planning under the trained
    // books, the exact emission tallies, the occupancy-optimal length
    // redesigns (floor-post + classbook dense — the planner picks
    // classes without consulting availability, so every symbol keeps
    // a codeword; the ladder value books **sparse** — the plans are
    // final here, every emitted entry is tallied, and pruning the
    // never-emitted cells is what keeps a deep band book's setup
    // table proportional to its actual use), and the exact
    // serialisation cost of everything the band choice can move: the
    // setup header packet plus the classword and residue-value
    // codewords (floor emissions are identical across sets).
    let trained: Vec<VorbisCodebook> = setup.codebooks[2..].to_vec();
    type Evaluated = (f64, VorbisSetupHeader, Vec<Vec<ResidueVectorPlan>>);
    // A candidate set is a list of indices into the *trained* band
    // books (`trained[3..]`), so every evaluation prices the exact
    // codeword lengths the closed-loop trainer settled on.
    let evaluate = |band_idx: &[usize]| -> Result<Evaluated, OggFileError> {
        let bands: Vec<(VorbisCodebook, BandShape)> = band_idx
            .iter()
            .map(|&i| (trained[3 + i].clone(), band_candidates[i].1))
            .collect();
        let mut setup = build(&trained[0], &trained[1], &trained[2], &bands);
        let classifications = 4 + bands.len() as u32;
        // The per-class value-book rows are resolved generically from
        // the setup header's own §8.6.1 `books` table (all entries
        // share the class rows — only `residue_end` / partition size
        // differ), so the rate-distortion chooser prices exactly the
        // ladder the header declares. Each partition's classword is
        // thereby a per-band value-book assignment, priced per
        // partition per pass.
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
        // Plans one full pass and also accumulates the plans' **weighted
        // distortion** `Σ w[p]·error²[p]` — the same noise-to-mask
        // objective every chooser above optimises — so the adoption
        // loop below can compare candidate band sets on the whole
        // Lagrangian, not on bits alone (a band class that buys real
        // fidelity must never be dropped for a byte win).
        type Planned = (Vec<Vec<ResidueVectorPlan>>, f64);
        let plan_all = |bias: Option<&[f64]>| -> Result<Planned, OggFileError> {
            let mut frame_plans: Vec<Vec<ResidueVectorPlan>> = Vec::with_capacity(frames);
            let mut weighted_error = 0.0f64;
            for f in 0..frames {
                let end = frame_res_end(f);
                let mut plans = Vec::with_capacity(ch);
                for c in 0..ch {
                    // The per-bin masking-weighted chooser: every VQ
                    // read is selected under its own elements'
                    // audibility weights and the Lagrangian charges the
                    // weighted error directly.
                    let choices = plan_vector_classifications_rd_bin_weighted(
                        &targets[f][c][..end],
                        &value_rows,
                        1,
                        frame_ps(f) as u32,
                        tuning.lambda,
                        &bin_weights[f][c],
                        bias,
                    )?;
                    let mut classifications = Vec::with_capacity(choices.len());
                    let mut partition_entries = Vec::with_capacity(choices.len());
                    for choice in choices {
                        weighted_error += choice.error_sq;
                        classifications.push(choice.classification);
                        partition_entries.push(choice.entries);
                    }
                    plans.push(ResidueVectorPlan {
                        classifications,
                        partition_entries,
                    });
                }
                frame_plans.push(plans);
            }
            Ok((frame_plans, weighted_error))
        };
        // Classword-aware planning: pass 1 prices value bits alone;
        // each refinement pass then prices every class's **marginal
        // classword bits** from the previous pass's class histogram
        // (`-log2 p(c)` — the per-partition share of an
        // entropy-optimal classword under an independence model,
        // which the dense occupancy retrain below approaches) and
        // re-plans under the biased chooser. Without this, a class
        // adopted for a marginal value-bit win can inflate the
        // classword entropy by more than it saves — the mispricing
        // that made a naive richer class ladder spend *more* audio
        // bytes at identical fidelity. Alternating plan ↔ re-price
        // converges like entropy-constrained quantiser design; the
        // loop stops early at a plan fixed point.
        let (mut frame_plans, mut weighted_error) = plan_all(None)?;
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
                    // Unseen classes are floored at one count
                    // (adopting one costs a fresh, long classword
                    // codeword), and the price is capped so a rare
                    // class stays *expensive* rather than
                    // unreachable.
                    let p = h.max(1) as f64 / total as f64;
                    (-p.log2()).clamp(0.0, f64::from(CLASSWORD_PRICE_CAP_BITS))
                })
                .collect();
            let (replanned, replanned_error) = plan_all(Some(&bias))?;
            if replanned == frame_plans {
                break;
            }
            frame_plans = replanned;
            weighted_error = replanned_error;
        }

        // Exact emission tallies for the final plans (the writer's
        // own §8.6.2 grouping via tally_residue_plans; the §7.2.3
        // floor-post emissions via tally_floor1_packet). Codeword
        // lengths carry no values, so every redesign below leaves the
        // packets decoding to bit-identical PCM; they only serialise
        // into fewer bits.
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
        for book in 2..setup.codebooks.len() {
            if let Some(freqs) = tallies.counts(book) {
                if freqs.iter().any(|&f| f > 0) {
                    setup.codebooks[book] = crate::book_design::redesign_codebook(
                        &setup.codebooks[book],
                        freqs,
                        VQ_DESIGN_MAX_CODEWORD_LEN,
                        false,
                    )?;
                }
            }
        }
        // The exact serialised cost the band choice can move — the
        // setup header packet plus every classword / residue-value
        // codeword bit — folded with the plans' weighted distortion
        // into the same `Σ w·error² + λ·bits` Lagrangian every
        // chooser above optimises. Setup bits ride the same λ: a
        // band book must buy its own table.
        let setup_packet = write_setup_header(&setup, config.channels)?;
        let mut bits = 8 * setup_packet.len() as u64;
        for book in 1..setup.codebooks.len() {
            if let Some(freqs) = tallies.counts(book) {
                bits += freqs
                    .iter()
                    .zip(&setup.codebooks[book].codeword_lengths)
                    .map(|(&f, &l)| f * u64::from(l))
                    .sum::<u64>();
            }
        }
        let objective = weighted_error + tuning.lambda * bits as f64;
        Ok((objective, setup, frame_plans))
    };

    // Greedy adopt-if-improved over the appended band classes: start
    // from the full candidate ladder the books were trained on and
    // drop any band whose removal measures a strictly smaller
    // Lagrangian. This is what keeps candidacy safe: a band the
    // corpus wants pays its way through shorter classwords + value
    // codewords or lower masked distortion; a band the plans barely
    // touch cannot cover its own setup table (or the classword
    // alphabet growth) and is measured out — the header never
    // carries a dead book.
    let mut kept: Vec<usize> = (0..band_candidates.len()).collect();
    let mut best = evaluate(&kept)?;
    let mut i = 0;
    while i < kept.len() {
        let mut reduced = kept.clone();
        reduced.remove(i);
        let candidate = evaluate(&reduced)?;
        if candidate.0 < best.0 {
            kept = reduced;
            best = candidate;
        } else {
            i += 1;
        }
    }
    let (_, setup, frame_plans) = best;
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

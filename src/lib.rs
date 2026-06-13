//! # oxideav-vorbis
//!
//! Pure-Rust Vorbis I codec — round-1 clean-room scaffold.
//!
//! ## Status
//!
//! The crate is being rebuilt clean-room under the workspace policy
//! against the Vorbis I Specification (Xiph.Org, 2020-07-04 revision)
//! and the corpus traces under `docs/audio/vorbis/`.
//!
//! Round 1 landed the identification-header parser (Vorbis I §4.2.2);
//! see [`identification`] for details. Round 2 adds the comment-header
//! parser (Vorbis I §5); see [`comment`]. Round 3 adds the
//! codebook-header parser (Vorbis I §3.2.1); see [`codebook`]. Round 4
//! adds the canonical Huffman tree builder + entry decoder; see
//! [`huffman`]. Round 5 adds the setup-header outer walker covering
//! codebooks, time-domain placeholders, floor headers and residue
//! headers (Vorbis I §4.2.4); see [`setup`]. Round 6 closes the
//! setup-header walk: mapping configurations (§4.2.4 "Mappings"),
//! mode configurations (§4.2.4 "Modes"), and the trailing framing
//! flag are now parsed. Round 7 lifts decoded Huffman entries into
//! VQ vectors per §3.2.1 "VQ lookup table vector representation" +
//! §3.3 "Use of the codebook abstraction"; see [`vq`]. Round 8 lands
//! the per-packet residue decode (§8.6.2 packet decode + §8.6.3/4/5
//! format 0/1/2 specifics); see [`residue`]. Round 9 lands the floor
//! type 1 per-packet decode + curve computation (§7.2.3 packet decode +
//! §7.2.4 curve computation); see [`floor1`]. Round 10 lands the
//! floor type 0 per-packet decode + LSP curve computation (§6.2.2
//! packet decode + §6.2.3 curve computation); see [`floor0`]. Round 11
//! lands the audio-packet synthesis primitives — the Vorbis window
//! (§1.3.2 / §4.3.1) and inverse channel coupling (§4.3.5); see
//! [`synthesis`]. Round 12 lands the two fully-specified, IMDCT-independent
//! audio-packet driver stages: §4.3.3 nonzero-vector propagate and §4.3.6
//! floor/residue dot product; see [`packet`]. Round 13 adds the §4.3.1
//! audio-packet prelude reader (`[packet_type]` / `[mode_number]` /
//! blocksize resolution / window flags); see [`packet::read_packet_header`].
//! Round 14 lands the top-level audio-packet driver covering §4.3.2
//! through §4.3.6: per-channel floor decode routed through the
//! mapping's submap table, §4.3.3 nonzero propagate, submap-routed
//! residue decode, §4.3.5 inverse coupling, and §4.3.6 dot product.
//! The driver stops cleanly at the §4.3.7 inverse-MDCT boundary; see
//! [`audio`]. The top-level [`decode_packet`] still returns at the
//! §4.3.7 boundary because wiring the IMDCT into the per-packet
//! driver depends on pinning the Vorbis-specific normalization
//! factor (see [`imdct`] doc) — a follow-up round once the fixture
//! traces under `docs/audio/vorbis/fixtures/` extend through the
//! post-IMDCT trace point. Round 15 lands the §4.3.8 overlap-add
//! primitive as a standalone, IMDCT-independent math module ready to
//! consume any windowed time-domain frame; see [`overlap`].
//! Round 16 lands the §4.3.7 inverse-MDCT cosine-summation kernel as
//! a standalone primitive — the direct O(N²) reference form derived
//! from the in-repo clean-room cross-reference document
//! `docs/audio/vorbis/imdct-cross-reference.md`; see [`imdct`]. Round 17
//! wires the §4.3.7 IMDCT and §4.3.6 windowing into the per-packet
//! driver: [`decode_audio_packet_windowed`] and the convenience
//! [`decode_one_packet_windowed`] entry point return per-channel
//! length-`n` windowed time-domain frames ready to feed into per-channel
//! [`overlap::OverlapAdd::push_frame`] instances. Only the Vorbis-specific
//! IMDCT normalization scalar (a deferred-fixture concern) is still
//! pinned via an explicit `imdct_scale: f32` knob; the full §4.3
//! pipeline-up-to-overlap-add is now reachable from a parsed packet.
//! Round 18 lands the multi-channel streaming PCM driver:
//! [`StreamingDecoder`] holds one [`overlap::OverlapAdd`] instance per
//! channel and stitches consecutive per-packet windowed outcomes into
//! finished PCM samples (§4.3.8 across packets), closing the last
//! composition step from a parsed audio-packet bitstream to per-channel
//! PCM; see [`streaming`].
//! Round 19 adds the §4.2.1 / §4.3.1 packet-kind classifier and the
//! unified header-packet dispatcher: [`classify_packet`] resolves a
//! raw packet payload to one of the four [`PacketKind`] variants
//! (`Identification` / `Comment` / `Setup` / `Audio`) by inspecting
//! the §4.2.1 common-header prelude or, for audio packets, the
//! §4.3.1 step-1 `[packet_type]` bit; [`parse_header_packet`] then
//! delegates to the matching per-header sub-parser and returns the
//! parsed result in a [`HeaderPacket`] sum. See [`packet_kind`].
//!
//! Round 20 (umbrella round 195) lands the first concrete encoder-
//! side primitive: a pair of header-packet WRITE functions. See
//! [`encoder`]. [`write_identification_header`] serialises a
//! [`VorbisIdentificationHeader`] to the fixed 30-byte §4.2.2 packet
//! shape, and [`write_comment_header`] serialises a
//! [`VorbisCommentHeader`] to the variable-length §5.2.1 packet
//! shape. Both functions validate the same spec-mandated invariants
//! their corresponding parsers enforce on input, so the bit-exact
//! roundtrip property
//! `parse_(...)_header(&write_(...)_header(&x)?)? == x` is
//! guaranteed for every legal input.
//!
//! Round 21 (umbrella round 201) lands the first nested-block writer:
//! [`write_codebook`] serialises a [`VorbisCodebook`] to the §3.2.1
//! codebook-header bitstream shape, plus the encoder-side §9.2.2
//! companion [`float32_pack`] that inverts the existing
//! [`float32_unpack`]. The writer picks the densest legal length
//! encoding (ordered / dense unordered / sparse unordered) per the
//! codebook's content and validates every §3.2.1 invariant
//! (`entries > 0`, length table well-sized, used lengths in `1..=32`,
//! `value_bits ∈ 1..=16`, multiplicand fits its field, ordered
//! requires no unused entries + non-decreasing lengths, lookup-type-1
//! count matches `lookup1_values()`, lookup-type-2 count matches
//! `entries × dimensions`, packed-float representability). The
//! round-trip property
//! `parse_codebook(&mut BitReaderLsb::new(&write_codebook(&book)?))? == book`
//! holds for every legal input across all three length encodings and
//! all three lookup types. Audio-packet WRITE and floor / residue /
//! mapping / mode WRITE primitives plus the setup-header splice are
//! explicit followups.
//!
//! Round 22 (umbrella round 206) lands the next nested-block writer:
//! [`write_floor1_header`] serialises a
//! [`crate::setup::Floor1Header`] to the §7.2.2 floor-type-1
//! header bitstream shape. Every §7.2.2 structural invariant is
//! validated before any bits are emitted; the writer fails closed
//! with a structured [`WriteFloor1Error`] (thirteen variants covering
//! partitions / class-list / class-count / dimensions / subclasses /
//! masterbook presence / subclass-book count / subclass-book overflow
//! / multiplier / rangebits / x-list length / x-list value-overflow
//! invariants). The bit-exact roundtrip property
//! `parse_floor1_header(&mut BitReaderLsb::new(&write_floor1_header(&h)?))? == h`
//! holds for every legal input. The umbrella [`WriteError`] grows a
//! [`WriteError::Floor1`] variant with the matching `From` glue.
//! Floor 0 WRITE, residue WRITE, mapping / mode WRITE, audio-packet
//! WRITE, and the setup-header splice remain explicit followups.
//!
//! Round 29 (umbrella round 243) lands the §4.3.7 forward-MDCT
//! cosine-summation kernel as a standalone primitive — the
//! encoder-side counterpart to the round-16 inverse-MDCT kernel.
//! See [`mdct`]. The forward kernel is the linear matrix-transpose
//! of the IMDCT kernel; the two directions share one cosine matrix
//! and the derived identity `mdct(imdct(X)) == (N/2) · X` holds at
//! every legal blocksize. The derivation is laid out in the
//! [`mdct`] module documentation and uses only the IMDCT formula
//! already in `docs/audio/vorbis/imdct-cross-reference.md`.
//!
//! Round 33 (umbrella round 255) lands the §4.3.5 forward channel
//! coupling primitives — the encoder counterpart of the round-11
//! decoder-side [`synthesis::inverse_couple`] / [`synthesis::inverse_couple_all`].
//! [`synthesis::forward_couple_scalar`] applies the algebraic inverse
//! of the §4.3.5 step-3 four-quadrant rule to a single Cartesian
//! `(L, R)` pair, returning the square-polar `(M, A)` such that
//! `couple_scalar(M, A) == (L, R)` bit-exactly.
//! [`synthesis::forward_couple`] is the in-place vector wrapper;
//! [`synthesis::forward_couple_all`] is the per-mapping driver that
//! runs every coupling step **in ascending order** (the reverse of
//! the §4.3.5 decoder loop's descending direction), producing the
//! square-polar channels the residue encoder will quantise. The
//! round-trip property
//! `inverse_couple_all(forward_couple_all(x)) == x` holds for every
//! legal input.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod audio;
pub mod codebook;
pub mod comment;
pub mod encoder;
pub mod floor0;
pub mod floor1;
pub mod framing;
pub mod huffman;
pub mod identification;
pub mod imdct;
pub mod mdct;
pub mod overlap;
pub mod packet;
pub mod packet_kind;
pub mod residue;
pub mod setup;
pub mod streaming;
pub mod synthesis;
pub mod vq;

pub use audio::{
    apply_imdct_and_window, decode_audio_packet_pre_imdct, decode_audio_packet_windowed,
    decode_one_packet, decode_one_packet_windowed, AudioDecoderState, AudioPacketError,
    AudioPacketOutcome, WindowedPacketOutcome,
};
pub use codebook::{
    float32_pack, float32_unpack, ilog, lookup1_values, parse_codebook,
    ParseError as CodebookParseError, VorbisCodebook, VqLookup, UNUSED_ENTRY,
};
pub use comment::{
    parse_comment_header, split_key_value, ParseError as CommentParseError, VorbisCommentHeader,
};
pub use encoder::{
    pack_residue_classification_groups, pack_residue_classifications, residue_body_shape,
    residue_partition_codeword_count, write_audio_packet_header, write_codebook,
    write_comment_header, write_floor0_header, write_floor0_packet, write_floor1_header,
    write_floor1_packet, write_identification_header, write_mapping_header, write_mode_header,
    write_residue_body, write_residue_header, write_residue_partition, Floor0Packet, Floor1Packet,
    PackResidueClassError, PackResidueClassGroupsError, ResidueBodyShape, ResidueVectorPlan,
    WriteAudioPacketHeaderError, WriteCodebookError, WriteError, WriteFloor0Error,
    WriteFloor0PacketError, WriteFloor1Error, WriteFloor1PacketError, WriteMappingError,
    WriteModeError, WriteResidueBodyError, WriteResidueError, WriteResiduePartitionError,
};
pub use floor0::{bark as floor0_bark, Floor0Curve, Floor0Decoder, Floor0Error};
pub use floor1::{
    high_neighbor, low_neighbor, render_line, render_point, Floor1Decoder, Floor1Error, FloorCurve,
};
pub use framing::{FrameSplitter, FramingError};
pub use huffman::{
    BuildError as HuffmanBuildError, DecodeError as HuffmanDecodeError, HuffmanNode, HuffmanTree,
};
pub use identification::{
    parse_identification_header, ParseError as IdentificationParseError, VorbisIdentificationHeader,
};
pub use imdct::{imdct_naive, imdct_naive_vec, ImdctError};
pub use mdct::{
    apply_window_and_mdct, apply_window_and_mdct_vec, mdct_naive, mdct_naive_vec,
    ApplyWindowAndMdctError, MdctError,
};
pub use overlap::{OverlapAdd, OverlapError};
pub use packet::{
    dot_product, dot_product_all, nonzero_propagate, read_packet_header, AudioPacketHeader,
    PacketError, PacketHeaderStage, VectorKind,
};
pub use packet_kind::{
    classify_packet, parse_header_packet, ClassifyError, HeaderDispatchError, HeaderPacket,
    PacketKind,
};
pub use residue::{ResidueDecoder, ResidueError};
pub use setup::{
    parse_setup_header, parse_setup_header_body, Floor0Header, Floor1Class, Floor1Header,
    FloorHeader, FloorKind, MappingCouplingStep, MappingHeader, MappingSubmap, ModeHeader,
    ParseError as SetupParseError, ResidueHeader, VorbisSetupHeader, SETUP_PACKET_MAGIC,
    SETUP_PACKET_TYPE,
};
pub use streaming::{StreamingDecoder, StreamingError, StreamingFrame};
pub use synthesis::{
    couple_scalar, forward_couple, forward_couple_all, forward_couple_scalar, inverse_couple,
    inverse_couple_all, slope, vorbis_window, window_premultiply, CouplingError, WindowError,
    WindowPremultiplyError,
};
pub use vq::{unpack_vector, UnpackError as VqUnpackError};

/// Crate-local error type for the in-progress clean-room rebuild.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A code path is reachable in the public surface but its
    /// implementation has not yet landed in any round.
    NotImplemented,
    /// The identification header (Vorbis I §4.2.2) failed to parse.
    Identification(IdentificationParseError),
    /// The comment header (Vorbis I §5) failed to parse.
    Comment(CommentParseError),
    /// A codebook header (Vorbis I §3.2.1) failed to parse.
    Codebook(CodebookParseError),
    /// A Huffman tree (Vorbis I §3.2.1) failed to build from a parsed
    /// codebook's `codeword_lengths`.
    HuffmanBuild(HuffmanBuildError),
    /// A Huffman codeword could not be decoded from the packet
    /// bitstream (Vorbis I §3.3).
    HuffmanDecode(HuffmanDecodeError),
    /// The setup-header walker (Vorbis I §4.2.4) failed to parse.
    Setup(SetupParseError),
    /// A VQ vector unpack (Vorbis I §3.2.1 / §3.3) failed.
    Vq(VqUnpackError),
    /// A residue decode (Vorbis I §8.6) failed to prepare or run.
    Residue(ResidueError),
    /// A floor type 1 decode (Vorbis I §7.2) failed to prepare or run.
    Floor1(Floor1Error),
    /// A floor type 0 decode (Vorbis I §6.2) failed to prepare or run.
    Floor0(Floor0Error),
    /// A Vorbis window could not be built (Vorbis I §1.3.2 / §4.3.1).
    Window(WindowError),
    /// Inverse channel coupling (Vorbis I §4.3.5) failed.
    Coupling(CouplingError),
    /// An audio-packet driver stage (Vorbis I §4.3.3 nonzero-vector
    /// propagate / §4.3.6 dot product) failed.
    Packet(PacketError),
    /// The §4.3.8 overlap-add primitive rejected a frame as malformed
    /// (length not a positive power of two, or below the spec minimum).
    Overlap(OverlapError),
    /// The §4.3.8 encoder-side framing-inverse primitive rejected a
    /// frame request — invalid length, mismatched analysis window
    /// length, or a buffer that does not yet hold enough PCM samples.
    Framing(FramingError),
    /// The §4.3.7 inverse-MDCT cosine-summation kernel rejected its
    /// input — either an invalid spectrum length, or a mismatched
    /// output buffer length.
    Imdct(ImdctError),
    /// The §4.3.7 forward-MDCT cosine-summation kernel rejected its
    /// input — either an invalid block length, or a mismatched
    /// output buffer length.
    Mdct(MdctError),
    /// The top-level §4.3 audio-packet driver failed at the §4.3.2
    /// through §4.3.6 stages (mapping/submap routing, floor or residue
    /// decode failure, inverse coupling, dot product), or stopped at the
    /// §4.3.7 inverse-MDCT docs-gap boundary.
    AudioPacket(AudioPacketError),
    /// The multi-channel streaming driver
    /// ([`crate::streaming::StreamingDecoder`]) failed at either the
    /// per-packet stage, the per-channel §4.3.8 overlap-add stage, or
    /// the defensive channel-count mismatch check.
    Streaming(StreamingError),
    /// The §4.2.1 / §4.3.1 packet-kind classifier
    /// ([`crate::packet_kind::classify_packet`]) failed to identify
    /// a packet from its byte-0 / magic prelude.
    Classify(ClassifyError),
    /// The unified header-packet dispatcher
    /// ([`crate::packet_kind::parse_header_packet`]) failed at the
    /// classification step, on an unexpected audio packet, or in one
    /// of the three header sub-parsers.
    HeaderDispatch(HeaderDispatchError),
    /// A header-packet, nested-block, or audio-packet-body writer
    /// ([`crate::encoder::write_identification_header`],
    /// [`crate::encoder::write_comment_header`],
    /// [`crate::encoder::write_codebook`],
    /// [`crate::encoder::write_floor1_header`],
    /// [`crate::encoder::write_floor0_header`],
    /// [`crate::encoder::write_residue_header`],
    /// [`crate::encoder::write_mapping_header`],
    /// [`crate::encoder::write_floor1_packet`],
    /// [`crate::encoder::pack_residue_classifications`],
    /// [`crate::encoder::write_residue_partition`], or
    /// [`crate::encoder::write_residue_body`]) rejected its
    /// input because the supplied struct fails one of the §4.2.2 /
    /// §5.2.1 / §3.2.1 / §6.2.1 / §7.2.2 / §7.2.3 / §8.6.1 / §8.6.2 /
    /// §4.2.4 invariants
    /// the encoder is contracted to refuse rather than emit a
    /// malformed packet.
    Write(WriteError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NotImplemented => write!(
                f,
                "oxideav-vorbis: code path not yet implemented in this round"
            ),
            Error::Identification(e) => write!(f, "{e}"),
            Error::Comment(e) => write!(f, "{e}"),
            Error::Codebook(e) => write!(f, "{e}"),
            Error::HuffmanBuild(e) => write!(f, "{e}"),
            Error::HuffmanDecode(e) => write!(f, "{e}"),
            Error::Setup(e) => write!(f, "{e}"),
            Error::Vq(e) => write!(f, "{e}"),
            Error::Residue(e) => write!(f, "{e}"),
            Error::Floor1(e) => write!(f, "{e}"),
            Error::Floor0(e) => write!(f, "{e}"),
            Error::Window(e) => write!(f, "{e}"),
            Error::Coupling(e) => write!(f, "{e}"),
            Error::Packet(e) => write!(f, "{e}"),
            Error::Overlap(e) => write!(f, "{e}"),
            Error::Framing(e) => write!(f, "{e}"),
            Error::Imdct(e) => write!(f, "{e}"),
            Error::Mdct(e) => write!(f, "{e}"),
            Error::AudioPacket(e) => write!(f, "{e}"),
            Error::Streaming(e) => write!(f, "{e}"),
            Error::Classify(e) => write!(f, "{e}"),
            Error::HeaderDispatch(e) => write!(f, "{e}"),
            Error::Write(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Identification(e) => Some(e),
            Error::Comment(e) => Some(e),
            Error::Codebook(e) => Some(e),
            Error::HuffmanBuild(e) => Some(e),
            Error::HuffmanDecode(e) => Some(e),
            Error::Setup(e) => Some(e),
            Error::Vq(e) => Some(e),
            Error::Residue(e) => Some(e),
            Error::Floor1(e) => Some(e),
            Error::Floor0(e) => Some(e),
            Error::Window(e) => Some(e),
            Error::Coupling(e) => Some(e),
            Error::Packet(e) => Some(e),
            Error::Overlap(e) => Some(e),
            Error::Framing(e) => Some(e),
            Error::Imdct(e) => Some(e),
            Error::Mdct(e) => Some(e),
            Error::AudioPacket(e) => Some(e),
            Error::Streaming(e) => Some(e),
            Error::Classify(e) => Some(e),
            Error::HeaderDispatch(e) => Some(e),
            Error::Write(e) => Some(e),
            Error::NotImplemented => None,
        }
    }
}

impl From<IdentificationParseError> for Error {
    fn from(value: IdentificationParseError) -> Self {
        Error::Identification(value)
    }
}

impl From<CommentParseError> for Error {
    fn from(value: CommentParseError) -> Self {
        Error::Comment(value)
    }
}

impl From<CodebookParseError> for Error {
    fn from(value: CodebookParseError) -> Self {
        Error::Codebook(value)
    }
}

impl From<HuffmanBuildError> for Error {
    fn from(value: HuffmanBuildError) -> Self {
        Error::HuffmanBuild(value)
    }
}

impl From<HuffmanDecodeError> for Error {
    fn from(value: HuffmanDecodeError) -> Self {
        Error::HuffmanDecode(value)
    }
}

impl From<SetupParseError> for Error {
    fn from(value: SetupParseError) -> Self {
        Error::Setup(value)
    }
}

impl From<VqUnpackError> for Error {
    fn from(value: VqUnpackError) -> Self {
        Error::Vq(value)
    }
}

impl From<ResidueError> for Error {
    fn from(value: ResidueError) -> Self {
        Error::Residue(value)
    }
}

impl From<Floor1Error> for Error {
    fn from(value: Floor1Error) -> Self {
        Error::Floor1(value)
    }
}

impl From<Floor0Error> for Error {
    fn from(value: Floor0Error) -> Self {
        Error::Floor0(value)
    }
}

impl From<WindowError> for Error {
    fn from(value: WindowError) -> Self {
        Error::Window(value)
    }
}

impl From<CouplingError> for Error {
    fn from(value: CouplingError) -> Self {
        Error::Coupling(value)
    }
}

impl From<PacketError> for Error {
    fn from(value: PacketError) -> Self {
        Error::Packet(value)
    }
}

impl From<OverlapError> for Error {
    fn from(value: OverlapError) -> Self {
        Error::Overlap(value)
    }
}

impl From<FramingError> for Error {
    fn from(value: FramingError) -> Self {
        Error::Framing(value)
    }
}

impl From<ImdctError> for Error {
    fn from(value: ImdctError) -> Self {
        Error::Imdct(value)
    }
}

impl From<MdctError> for Error {
    fn from(value: MdctError) -> Self {
        Error::Mdct(value)
    }
}

impl From<AudioPacketError> for Error {
    fn from(value: AudioPacketError) -> Self {
        Error::AudioPacket(value)
    }
}

impl From<StreamingError> for Error {
    fn from(value: StreamingError) -> Self {
        Error::Streaming(value)
    }
}

impl From<ClassifyError> for Error {
    fn from(value: ClassifyError) -> Self {
        Error::Classify(value)
    }
}

impl From<HeaderDispatchError> for Error {
    fn from(value: HeaderDispatchError) -> Self {
        Error::HeaderDispatch(value)
    }
}

impl From<WriteError> for Error {
    fn from(value: WriteError) -> Self {
        Error::Write(value)
    }
}

/// Top-level audio-packet decode entry point. Drives the §4.3.2 through
/// §4.3.6 pipeline ([`audio::decode_one_packet`]) and stops cleanly at
/// the §4.3.7 inverse-MDCT boundary, currently a documented docs gap
/// (the Vorbis I spec defers the MDCT definition entirely to external
/// reference `[1]`, which the workspace clean-room policy bars).
///
/// * `packet` — the audio-packet bitstream payload (Ogg framing
///   stripped, page-coalesced).
/// * `setup` — the stream's parsed setup header.
/// * `state` — the per-stream decoder cache built by
///   [`AudioDecoderState::new`].
/// * `audio_channels` / `blocksize_0` / `blocksize_1` — the matching
///   identification-header fields.
///
/// Returns [`Error::AudioPacket(AudioPacketError::ImdctStage)`](AudioPacketError::ImdctStage)
/// on every successful drive through §4.3.6. Earlier-stage failures
/// surface as the corresponding [`AudioPacketError`] variant inside
/// [`Error::AudioPacket`]. The function never returns `Ok(_)` in this
/// round; the signature is shaped to absorb the IMDCT-round change
/// without breaking callers.
pub fn decode_packet(
    packet: &[u8],
    setup: &setup::VorbisSetupHeader,
    state: &AudioDecoderState,
    audio_channels: u8,
    blocksize_0: usize,
    blocksize_1: usize,
) -> Result<(), Error> {
    let mut reader = oxideav_core::bits::BitReaderLsb::new(packet);
    audio::decode_one_packet(
        &mut reader,
        setup,
        state,
        audio_channels,
        blocksize_0,
        blocksize_1,
    )?;
    // decode_one_packet always returns Err in this round (Ok is
    // unreachable until IMDCT lands), so this branch is unreachable.
    // Kept as the explicit "shape for the future" return.
    Err(Error::NotImplemented)
}

/// No-op codec registration — the round-1 scaffold does not yet
/// register a [`oxideav_core::Decoder`] or [`oxideav_core::Encoder`]
/// because the audio-packet pipeline is not yet implemented.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("vorbis", register);

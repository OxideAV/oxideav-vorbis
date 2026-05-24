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
//! The full audio-packet decode (§4.3) is still pending — §4.3.7 inverse
//! MDCT defers its definition to a barred external reference — and
//! [`decode_packet`] currently returns [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod codebook;
pub mod comment;
pub mod floor0;
pub mod floor1;
pub mod huffman;
pub mod identification;
pub mod packet;
pub mod residue;
pub mod setup;
pub mod synthesis;
pub mod vq;

pub use codebook::{
    parse_codebook, ParseError as CodebookParseError, VorbisCodebook, VqLookup, UNUSED_ENTRY,
};
pub use comment::{
    parse_comment_header, split_key_value, ParseError as CommentParseError, VorbisCommentHeader,
};
pub use floor0::{bark as floor0_bark, Floor0Curve, Floor0Decoder, Floor0Error};
pub use floor1::{
    high_neighbor, low_neighbor, render_line, render_point, Floor1Decoder, Floor1Error, FloorCurve,
};
pub use huffman::{
    BuildError as HuffmanBuildError, DecodeError as HuffmanDecodeError, HuffmanNode, HuffmanTree,
};
pub use identification::{
    parse_identification_header, ParseError as IdentificationParseError, VorbisIdentificationHeader,
};
pub use packet::{
    dot_product, dot_product_all, nonzero_propagate, read_packet_header, AudioPacketHeader,
    PacketError, PacketHeaderStage, VectorKind,
};
pub use residue::{ResidueDecoder, ResidueError};
pub use setup::{
    parse_setup_header, parse_setup_header_body, Floor0Header, Floor1Class, Floor1Header,
    FloorHeader, FloorKind, MappingCouplingStep, MappingHeader, MappingSubmap, ModeHeader,
    ParseError as SetupParseError, ResidueHeader, VorbisSetupHeader, SETUP_PACKET_MAGIC,
    SETUP_PACKET_TYPE,
};
pub use synthesis::{
    couple_scalar, inverse_couple, inverse_couple_all, slope, vorbis_window, CouplingError,
    WindowError,
};
pub use vq::{unpack_vector, UnpackError as VqUnpackError};

/// Crate-local error type for the in-progress clean-room rebuild.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Placeholder audio-packet decode entry point. The setup header
/// and audio-packet decode pipeline are not yet wired up; this
/// function always returns [`Error::NotImplemented`].
pub fn decode_packet(_packet: &[u8]) -> Result<(), Error> {
    Err(Error::NotImplemented)
}

/// No-op codec registration — the round-1 scaffold does not yet
/// register a [`oxideav_core::Decoder`] or [`oxideav_core::Encoder`]
/// because the audio-packet pipeline is not yet implemented.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("vorbis", register);

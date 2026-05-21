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
//! codebook-header parser (Vorbis I §3.2.1); see [`codebook`]. The
//! enclosing setup-header walker (§4.2.4) and the audio-packet decode
//! (§4.3) are still pending and [`decode_packet`] currently returns
//! [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod codebook;
pub mod comment;
pub mod identification;

pub use codebook::{
    parse_codebook, ParseError as CodebookParseError, VorbisCodebook, VqLookup, UNUSED_ENTRY,
};
pub use comment::{
    parse_comment_header, split_key_value, ParseError as CommentParseError, VorbisCommentHeader,
};
pub use identification::{
    parse_identification_header, ParseError as IdentificationParseError, VorbisIdentificationHeader,
};

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
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Identification(e) => Some(e),
            Error::Comment(e) => Some(e),
            Error::Codebook(e) => Some(e),
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

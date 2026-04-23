// Parallel-array index loops are idiomatic in audio codec code; let clippy
// nag elsewhere.
#![allow(clippy::needless_range_loop)]
#![cfg_attr(feature = "nightly", feature(portable_simd))]

//! Vorbis I audio codec.
//!
//! Decoder is feature-complete for real-world file shapes: floor types
//! 0 (LSP) + 1, residue types 0/1/2 with cascade books, mapping type 0
//! with any number of submaps and channel coupling steps, up to 255
//! channels, full asymmetric long↔short MDCT windows and overlap-add.
//! Matches libvorbis / lewton output within float rounding on the
//! fixture suite.
//!
//! Encoder handles 1 or 2 channels at any sample rate with sum/difference
//! channel coupling (Vorbis I §1.3.3), ATH-scaled floor1, a single
//! 128-entry residue VQ covering {-5..+5}², and transient-driven
//! short-block switching (asymmetric long↔short windows per §1.3.2 /
//! §4.3.1). Output decodes through both this crate's decoder and
//! ffmpeg's libvorbis. See the `encoder.rs` module-level doc for the
//! known bitrate trade-offs relative to libvorbis (point-stereo,
//! Annex-B reference books, floor0 emission).

pub mod audio_packet;
pub mod bits_ext;
pub mod codebook;
pub mod dbtable;
pub mod decoder;
pub mod encoder;
pub mod floor;
pub mod identification;
pub mod imdct;
pub mod libvorbis_setup;
pub mod residue;
pub mod setup;
pub mod setup_writer;
pub mod simd;

use oxideav_codec::{CodecInfo, CodecRegistry, Decoder, Encoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, CodecTag, Result};

pub const CODEC_ID_STR: &str = "vorbis";

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("vorbis_sw")
        .with_lossy(true)
        .with_max_channels(255);
    // AVI / WAVEFORMATEX tags — six values have been stamped on Vorbis
    // AVI streams historically, differing in how the setup / codebook
    // headers are packed into extradata. All decode through the same
    // path here.
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder)
            .tags([
                CodecTag::wave_format(0x674F),
                CodecTag::wave_format(0x6750),
                CodecTag::wave_format(0x6751),
                CodecTag::wave_format(0x676F),
                CodecTag::wave_format(0x6770),
                CodecTag::wave_format(0x6771),
            ]),
    );
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    encoder::make_encoder(params)
}

pub use identification::{parse_identification_header, Identification};

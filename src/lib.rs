// Parallel-array index loops are idiomatic in audio codec code; let clippy
// nag elsewhere.
#![allow(clippy::needless_range_loop)]

//! Vorbis I audio codec.
//!
//! Decoder is feature-complete for real-world file shapes: floors type 1,
//! residue types 0/1/2 with cascade books, mapping type 0 with any number
//! of submaps and channel coupling steps, up to 255 channels, full
//! asymmetric long↔short MDCT windows and overlap-add. Matches libvorbis
//! / lewton output within float rounding on the fixture suite. Floor type
//! 0 (LSP) is rejected with `Error::Unsupported` — no modern encoder
//! produces it.
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
pub mod bitreader;
pub mod bitwriter;
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

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

pub const CODEC_ID_STR: &str = "vorbis";

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::audio("vorbis_sw")
        .with_lossy(true)
        .with_max_channels(255);
    reg.register_decoder_impl(cid.clone(), caps.clone(), make_decoder);
    reg.register_encoder_impl(cid, caps, make_encoder);
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    encoder::make_encoder(params)
}

pub use identification::{parse_identification_header, Identification};

//! Vorbis-specific bit-reader extensions (LSB-first).
//!
//! The generic LSB-first bit reader + writer live in
//! [`oxideav_core::bits`] as `BitReaderLsb` / `BitWriterLsb`. Vorbis
//! adds one codec-specific primitive — the IEEE-like 32-bit
//! "Vorbis float" (§9.2.2), which has a unique mantissa + biased
//! exponent layout unrelated to IEEE 754. Pull it in via
//! `use oxideav_vorbis::bits_ext::BitReaderExt;`.

use oxideav_core::{bits::BitReaderLsb, Result};

pub trait BitReaderExt {
    /// Read a 32-bit Vorbis float (§9.2.2): 21-bit mantissa (signed),
    /// 1 bit sign, 10-bit biased exponent (bias 788). Returns the
    /// decoded value as `f32`.
    fn read_vorbis_float(&mut self) -> Result<f32>;
}

impl BitReaderExt for BitReaderLsb<'_> {
    fn read_vorbis_float(&mut self) -> Result<f32> {
        let raw = self.read_u32(32)?;
        let mantissa_raw = raw & 0x001FFFFF;
        let sign = raw & 0x80000000;
        let exponent = ((raw & 0x7FE00000) >> 21) as i32;
        let mantissa = if sign != 0 {
            -(mantissa_raw as f64)
        } else {
            mantissa_raw as f64
        };
        Ok((mantissa * 2f64.powi(exponent - 788)) as f32)
    }
}

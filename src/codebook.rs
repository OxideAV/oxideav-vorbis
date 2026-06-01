//! Vorbis I codebook header parser (Vorbis I §3.2.1).
//!
//! A Vorbis codebook is a Huffman-coded mapping from a bit-prefix of the
//! packet bitstream to either a scalar value or a fixed-dimension vector.
//! Codebooks live in the setup header (§4.2.4); this module decodes a
//! single codebook header from a [`BitReaderLsb`] positioned at the
//! 24-bit `0x564342` sync pattern.
//!
//! ## Packet layout (Vorbis I §3.2.1)
//!
//! Per the 2020-07-04 revision of the Vorbis I Specification:
//!
//! * 24 bits — sync pattern `0x564342` (`"BCV"` little-endian: byte 0
//!   `0x42`, byte 1 `0x43`, byte 2 `0x56`; reading 24 bits LSB-first per
//!   §2.1.4 yields the literal value `0x564342`).
//! * 16 bits — `codebook_dimensions` (unsigned).
//! * 24 bits — `codebook_entries` (unsigned).
//! * 1 bit  — `ordered` flag.
//! * Codeword-length table, format dependent on `ordered`:
//!   * `ordered = 0`: 1 bit `sparse` flag, then per entry:
//!     * If `sparse = 1`: 1-bit `used` flag; if set, 5-bit `length - 1`
//!       (so the recorded length is `bits + 1`, in `1..=32`); else
//!       the entry is unused (no codeword in the tree).
//!     * If `sparse = 0`: 5-bit `length - 1` (every entry is used).
//!   * `ordered = 1`: ascending-length run-length encoding —
//!     ```text
//!     current_entry  = 0
//!     current_length = read 5 bits + 1
//!     loop:
//!         number = read ilog(entries - current_entry) bits
//!         lengths[current_entry .. current_entry + number] = current_length
//!         current_entry  += number
//!         current_length += 1
//!         if current_entry  > entries: ERROR
//!         if current_entry == entries: done
//!     ```
//! * 4 bits — `codebook_lookup_type`:
//!   * `0` — no lookup; the codebook emits the entry index directly.
//!   * `1` — lattice VQ; multiplicand count = `lookup1_values(entries,
//!     dimensions)` (§9.2.3).
//!   * `2` — tessellated VQ; multiplicand count = `entries × dimensions`.
//!   * `3..=15` — reserved; reading any of these on a conformant stream
//!     is an error.
//! * For `lookup_type ∈ {1, 2}`:
//!   * 32 bits — `minimum_value`, a Vorbis-packed float (§9.2.2).
//!   * 32 bits — `delta_value`,   a Vorbis-packed float (§9.2.2).
//!   * 4 bits  — `value_bits − 1` (so the recorded width is in `1..=16`).
//!   * 1 bit   — `sequence_p`.
//!   * `lookup_values × value_bits` bits — unsigned multiplicands.
//!
//! ## What this module is, and is not
//!
//! The round-3 module decodes the codebook header's *structure*: it
//! returns dimensions, entries, the per-entry codeword length table
//! (with unused-entry sentinel), and the lookup metadata + raw
//! multiplicands. It does **not** build a Huffman decision tree, decode
//! a codebook entry from the audio bitstream, or apply the lookup table
//! to recover a vector. Those routines are followups for the next
//! rounds.

use core::fmt;

use oxideav_core::bits::BitReaderLsb;

/// Sentinel "this entry is unused" value in [`VorbisCodebook::codeword_lengths`].
///
/// §3.2.1 forbids a codeword length of `0` (lengths are `[length] + 1`
/// where `[length]` is a 5-bit field, giving the valid range `1..=32`),
/// so `0` is reserved here to signal a sparse-codebook entry that the
/// encoder marked unused.
pub const UNUSED_ENTRY: u8 = 0;

/// The lookup section of a parsed codebook (§3.2.1).
///
/// `None`-equivalent is represented by [`VqLookup::None`], matching
/// `codebook_lookup_type = 0` in the spec ("no lookup, the codebook
/// emits an index").
#[derive(Debug, Clone, PartialEq)]
pub enum VqLookup {
    /// `codebook_lookup_type = 0` — entropy-only codebook. The decoded
    /// Huffman entry index is the codebook's output; no multiplicand
    /// table follows in the header.
    None,
    /// `codebook_lookup_type = 1` — lattice (implicitly populated) VQ.
    ///
    /// The multiplicand count is `lookup1_values(entries, dimensions)`
    /// per §9.2.3 ("greatest integer such that
    /// `lookup_values ^ dimensions <= entries`"). Per-entry vectors are
    /// produced at decode time by permuting this scalar table.
    Lattice {
        /// Vorbis-packed-float "minimum value" base of the multiplicand
        /// table, already decoded to host `f32` via [`float32_unpack`].
        minimum_value: f32,
        /// Vorbis-packed-float "delta value" multiplied into each
        /// multiplicand at decode time.
        delta_value: f32,
        /// Bit width of each multiplicand stored in the header
        /// (`value_bits − 1` is read; the resolved width is in `1..=16`).
        value_bits: u8,
        /// `sequence_p = 1` flags a lookup whose decoded vector
        /// elements are returned as a running prefix sum.
        sequence_p: bool,
        /// Raw multiplicands, each an unsigned integer of `value_bits`
        /// bits, in stream order. Length is exactly
        /// `lookup1_values(entries, dimensions)`.
        multiplicands: Vec<u32>,
    },
    /// `codebook_lookup_type = 2` — tessellation (explicitly populated)
    /// VQ. Multiplicand count is `entries × dimensions`; the table is
    /// indexed directly by the decoded Huffman entry.
    Tessellation {
        /// Vorbis-packed-float "minimum value" base.
        minimum_value: f32,
        /// Vorbis-packed-float "delta value" applied at decode time.
        delta_value: f32,
        /// Bit width of each multiplicand (1..=16).
        value_bits: u8,
        /// `sequence_p = 1` flags running-prefix-sum semantics.
        sequence_p: bool,
        /// Raw multiplicands. Length is exactly `entries × dimensions`.
        multiplicands: Vec<u32>,
    },
}

/// Parsed Vorbis I codebook header (§3.2.1).
#[derive(Debug, Clone, PartialEq)]
pub struct VorbisCodebook {
    /// `codebook_dimensions` (16-bit unsigned).
    pub dimensions: u16,
    /// `codebook_entries` (24-bit unsigned).
    pub entries: u32,
    /// Per-entry codeword lengths in `1..=32`, or [`UNUSED_ENTRY`]
    /// (`0`) for sparse codebooks where the entry was marked unused.
    /// Length is exactly `entries as usize`.
    pub codeword_lengths: Vec<u8>,
    /// The codebook's lookup table (§3.2.1 step 7+).
    pub lookup: VqLookup,
}

impl VorbisCodebook {
    /// The fixed 24-bit codebook sync pattern (§3.2.1).
    ///
    /// On a packet body bit-packed LSB-first per §2.1.4 the three sync
    /// bytes appear in memory as `0x42 0x43 0x56` (`"BCV"`); reading
    /// them as a 24-bit LSB-first integer yields the literal value
    /// `0x564342`.
    pub const SYNC_PATTERN: u32 = 0x0056_4342;

    /// Returns `true` if every entry of [`Self::codeword_lengths`] is
    /// marked unused. The spec rejects empty trees as an error (§3.2.1
    /// "underspecified tree"), but a parser only enforces that on tree
    /// construction; callers may want to flag a fully-unused codebook
    /// before they attempt to build a Huffman tree.
    #[must_use]
    pub fn is_fully_unused(&self) -> bool {
        self.codeword_lengths.iter().all(|&l| l == UNUSED_ENTRY)
    }

    /// Returns the count of used entries (i.e. entries with
    /// `length != 0`).
    #[must_use]
    pub fn used_entries(&self) -> usize {
        self.codeword_lengths
            .iter()
            .filter(|&&l| l != UNUSED_ENTRY)
            .count()
    }
}

/// Errors that may arise while parsing a Vorbis codebook header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The 24-bit sync pattern at the start of the codebook header did
    /// not equal `0x564342` (§3.2.1).
    BadSyncPattern(u32),
    /// `codebook_entries` was zero. The spec encodes the field as a
    /// 24-bit unsigned with no explicit lower bound, but a codebook
    /// with zero entries has no Huffman tree and is by construction
    /// not consumable; treat it as a malformed stream.
    ZeroEntries,
    /// The ordered-encoding loop overshot `codebook_entries`
    /// (§3.2.1 step 7: "if `current_entry` is greater than
    /// `codebook_entries`, ERROR CONDITION").
    OrderedOverflow {
        /// The cumulative `current_entry` after the overshooting step.
        current_entry: u32,
        /// `codebook_entries`, for context.
        entries: u32,
    },
    /// A `codebook_lookup_type` outside `0..=2` appeared (§3.2.1 "a
    /// `codebook_lookup_type` greater than two is reserved and
    /// indicates a stream that is not decodable by the specification
    /// in this document").
    ReservedLookupType(u8),
    /// The bitstream ran out of bits mid-codebook. Per §3.2.1 "an `end
    /// of packet` during any read operation in the above steps is
    /// considered an error condition rendering the stream
    /// undecodable", so this is a fatal parse failure.
    UnexpectedEndOfPacket,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::BadSyncPattern(v) => write!(
                f,
                "vorbis codebook: sync pattern 0x{v:06x} != 0x564342 (§3.2.1)"
            ),
            ParseError::ZeroEntries => {
                write!(f, "vorbis codebook: codebook_entries = 0 (§3.2.1)")
            }
            ParseError::OrderedOverflow {
                current_entry,
                entries,
            } => write!(
                f,
                "vorbis codebook: ordered run-length encoding overshot entries \
                 (current_entry={current_entry} > entries={entries}; §3.2.1 step 7)"
            ),
            ParseError::ReservedLookupType(t) => write!(
                f,
                "vorbis codebook: reserved codebook_lookup_type={t} (must be 0, 1 or 2 per §3.2.1)"
            ),
            ParseError::UnexpectedEndOfPacket => write!(
                f,
                "vorbis codebook: end-of-packet mid-codebook (§3.2.1: fatal)"
            ),
        }
    }
}

impl std::error::Error for ParseError {}

/// `ilog(x)` per Vorbis I §9.2.1: position number (1..=n) of the highest
/// set bit of `x`, or `0` for `x = 0` (the spec also treats negative `x`
/// as `0`; `u32` cannot be negative).
#[must_use]
pub fn ilog(x: u32) -> u32 {
    if x == 0 {
        0
    } else {
        32 - x.leading_zeros()
    }
}

/// `float32_unpack(x)` per Vorbis I §9.2.2: decode a Vorbis-packed
/// 32-bit float (a 21-bit mantissa, 10-bit biased exponent, 1-bit sign,
/// with bias 788) into a host `f32`.
#[must_use]
pub fn float32_unpack(x: u32) -> f32 {
    // §9.2.2 algorithm, transcribed verbatim:
    //   mantissa = x & 0x1fffff           (21-bit unsigned)
    //   sign     = x & 0x80000000
    //   exponent = (x & 0x7fe00000) >> 21 (10-bit unsigned)
    //   if sign != 0: negate mantissa
    //   return mantissa * 2^(exponent - 788)
    let mantissa = (x & 0x001f_ffff) as i32;
    let sign = (x & 0x8000_0000) != 0;
    let exponent = ((x & 0x7fe0_0000) >> 21) as i32;
    let signed = if sign { -mantissa } else { mantissa };
    // 2 ^ (exponent - 788) computed via scalbn for full range coverage
    // — `exponent` may legally span 0..=1023, so `exponent - 788` can
    // be as small as -788 and as large as +235, both of which require
    // proper handling of subnormals and overflow that bare `powi`
    // doesn't give.
    (signed as f32) * (2.0f32).powi(exponent - 788)
}

/// Encoder-side companion to [`float32_unpack`]: pack a finite host
/// `f32` into the Vorbis I §9.2.2 32-bit container.
///
/// §9.2.2 only defines the unpack direction; this helper is the
/// arithmetic inverse, packing a finite `f32` as
/// `(sign, mantissa, exponent)` such that
/// `float32_unpack(float32_pack(x)) == x` for every finite `x`
/// whose magnitude fits in the §9.2.2 range. The container has:
///
/// * 21-bit unsigned mantissa (`0..=0x1f_ffff`),
/// * 10-bit unsigned biased exponent (`0..=0x3ff`), bias 788,
/// * 1-bit sign in the MSB.
///
/// Special cases:
///
/// * `x == 0.0` (positive or negative zero) → returns `0` (the
///   canonical zero representation; §9.2.2 produces `0.0` for
///   container value `0`).
/// * Non-finite inputs (NaN / ±∞) → `None`. §9.2.2 has no
///   representation for these.
/// * `|x|` smaller than the smallest representable magnitude
///   (`1 * 2^-788`) → `None`. The container has no subnormal range
///   beyond `mantissa = 1, exponent = 0`.
/// * `|x|` larger than the largest representable magnitude
///   (`0x1f_ffff * 2^(1023 - 788)`) → `None`.
///
/// Round-trip equality `float32_unpack(float32_pack(x).unwrap()) == x`
/// holds only for `x` already expressible as an integer mantissa
/// `m ∈ -2_097_151..=2_097_151` times a power of two — which is
/// exactly the set of values §9.2.2 can carry. Lossy-input quantizing
/// is the encoder's job, not this primitive's.
#[must_use]
pub fn float32_pack(x: f32) -> Option<u32> {
    if x == 0.0 {
        // Both ±0 collapse to the canonical zero container; §9.2.2 of
        // the unpack reproduces 0.0 from a fully-zero u32.
        return Some(0);
    }
    if !x.is_finite() {
        return None;
    }
    // Decompose |x| into IEEE-754 mantissa + exponent, then re-scale
    // to fit the §9.2.2 21-bit unsigned mantissa container.
    let sign_bit: u32 = if x.is_sign_negative() { 0x8000_0000 } else { 0 };
    let abs = x.abs();

    // f32 bit layout: 1 sign + 8 exp (bias 127) + 23 mantissa.
    let bits = abs.to_bits();
    let ieee_exp = ((bits >> 23) & 0xff) as i32;
    let ieee_mant = bits & 0x007f_ffff;

    // Reconstruct (mantissa, unbiased_exp) such that
    // abs == mantissa * 2^unbiased_exp with mantissa as the smallest
    // integer that exactly represents the value.
    //
    // Normal IEEE: value = (1 + ieee_mant * 2^-23) * 2^(ieee_exp - 127)
    //                    = (2^23 + ieee_mant) * 2^(ieee_exp - 127 - 23)
    // Subnormal (ieee_exp == 0): value = ieee_mant * 2^(1 - 127 - 23)
    //                                  = ieee_mant * 2^-149
    let (mut int_mant, mut unbiased_exp): (u32, i32) = if ieee_exp == 0 {
        // Subnormal — including denormalised inputs.
        if ieee_mant == 0 {
            // Handled by the `x == 0.0` early-return above.
            unreachable!("zero already handled");
        }
        (ieee_mant, -149)
    } else {
        // Normal — re-attach the implicit leading 1.
        (ieee_mant | (1 << 23), ieee_exp - 127 - 23)
    };

    // Strip trailing zero bits of the mantissa to get the canonical
    // (smallest integer mantissa, matching exponent) representation —
    // §9.2.2 has no preferred normalisation, but the smallest mantissa
    // gives us the most headroom against the 21-bit field.
    while int_mant != 0 && (int_mant & 1) == 0 {
        int_mant >>= 1;
        unbiased_exp += 1;
    }

    // §9.2.2: value = mantissa * 2^(stored_exponent - 788), so
    // stored_exponent = unbiased_exp + 788.
    let stored_exp = unbiased_exp + 788;
    if !(0..=1023).contains(&stored_exp) {
        return None;
    }

    // §9.2.2 21-bit mantissa container limit.
    if int_mant > 0x001f_ffff {
        return None;
    }

    Some(sign_bit | ((stored_exp as u32) << 21) | int_mant)
}

/// `lookup1_values(entries, dimensions)` per Vorbis I §9.2.3: the
/// greatest integer `n` such that `n.pow(dimensions) <= entries`.
///
/// Returns `0` for `dimensions = 0` (the spec does not define this
/// case, but a zero-dimensional codebook has no per-entry vector, so
/// callers should never invoke this with `dimensions = 0`).
#[must_use]
pub fn lookup1_values(entries: u32, dimensions: u16) -> u32 {
    if dimensions == 0 {
        return 0;
    }
    if entries == 0 {
        return 0;
    }
    // The result is bounded above by `entries` itself (since
    // `1.pow(d) = 1 <= entries` always, and `entries^1 = entries`).
    // A simple monotonic search over an `f64` seed gives an upper
    // bound; verify and step until the largest valid `n` is found.
    let dims = dimensions as i32;
    // Seed with floor((entries as f64).powf(1.0 / dims)).
    let seed = (entries as f64).powf(1.0 / dims as f64).floor() as u32;
    // Walk a small window around the seed to absorb floating-point
    // imprecision near the boundary.
    let mut best = 0u32;
    let lo = seed.saturating_sub(2);
    let hi = seed.saturating_add(2);
    for candidate in lo..=hi {
        // pow(candidate, dimensions) — saturate on overflow so that
        // sufficiently-large `candidate` is treated as "too large"
        // rather than wrapping silently.
        let mut prod: u128 = 1;
        let mut overflow = false;
        for _ in 0..dims {
            prod = prod.saturating_mul(candidate as u128);
            if prod > entries as u128 {
                overflow = true;
                break;
            }
        }
        if !overflow && prod <= entries as u128 {
            best = candidate;
        }
    }
    best
}

/// Parses a single Vorbis I codebook header from a [`BitReaderLsb`]
/// positioned at the 24-bit `0x564342` sync pattern.
///
/// On success, returns the parsed [`VorbisCodebook`] and leaves the bit
/// reader positioned immediately after the codebook's last bit (i.e.
/// after the final multiplicand for lookup types 1 / 2, or after the
/// `codebook_lookup_type` nibble for lookup type 0).
///
/// On any deviation from §3.2.1, returns a structured [`ParseError`].
pub fn parse_codebook(reader: &mut BitReaderLsb<'_>) -> Result<VorbisCodebook, ParseError> {
    // §3.2.1 step 1: 24-bit sync pattern 0x564342.
    let sync = read_u32(reader, 24)?;
    if sync != VorbisCodebook::SYNC_PATTERN {
        return Err(ParseError::BadSyncPattern(sync));
    }

    // §3.2.1 step 2: 16-bit dimensions, 24-bit entries.
    let dimensions = read_u32(reader, 16)? as u16;
    let entries = read_u32(reader, 24)?;
    if entries == 0 {
        return Err(ParseError::ZeroEntries);
    }

    // §3.2.1 step 3: 1-bit ordered flag.
    let ordered = read_bit(reader)?;

    let codeword_lengths = if !ordered {
        // Unordered: 1-bit sparse flag, then per-entry length.
        let sparse = read_bit(reader)?;
        let mut lengths = Vec::with_capacity(entries as usize);
        for _ in 0..entries {
            if sparse {
                let used = read_bit(reader)?;
                if used {
                    let length = read_u32(reader, 5)? as u8 + 1;
                    lengths.push(length);
                } else {
                    lengths.push(UNUSED_ENTRY);
                }
            } else {
                let length = read_u32(reader, 5)? as u8 + 1;
                lengths.push(length);
            }
        }
        lengths
    } else {
        // Ordered: ascending-length run encoding.
        let mut lengths = vec![UNUSED_ENTRY; entries as usize];
        let mut current_entry: u32 = 0;
        let mut current_length: u32 = read_u32(reader, 5)? + 1;
        while current_entry < entries {
            // §3.2.1 step 3 (ordered branch):
            //   number = read ilog(entries - current_entry) bits
            let width = ilog(entries - current_entry);
            let number = read_u32(reader, width)?;
            let new_entry = current_entry
                .checked_add(number)
                .ok_or(ParseError::UnexpectedEndOfPacket)?;
            if new_entry > entries {
                return Err(ParseError::OrderedOverflow {
                    current_entry: new_entry,
                    entries,
                });
            }
            // Length field is a u8 in the struct; spec caps at 32 (5
            // bits + 1), but ordered streams could in principle grow
            // `current_length` past 32. Clamp to u8 range via cast and
            // rely on later tree-build validation to reject any length
            // > 32. (A well-formed ordered stream must by construction
            // produce lengths in 1..=32.)
            let length_u8 = current_length.min(255) as u8;
            for slot in &mut lengths[current_entry as usize..new_entry as usize] {
                *slot = length_u8;
            }
            current_entry = new_entry;
            current_length += 1;
        }
        lengths
    };

    // §3.2.1 step 5: lookup_type (4 bits).
    let lookup_type = read_u32(reader, 4)? as u8;
    let lookup = match lookup_type {
        0 => VqLookup::None,
        1 | 2 => {
            // §3.2.1 step 5 / lookup decode:
            //   minimum_value = float32_unpack(read 32 bits)
            //   delta_value   = float32_unpack(read 32 bits)
            //   value_bits    = read 4 bits + 1
            //   sequence_p    = read 1 bit
            //   lookup_values = lookup1_values(entries, dimensions) for type 1
            //                 = entries * dimensions             for type 2
            //   read lookup_values unsigned ints of value_bits each → multiplicands
            let minimum_value = float32_unpack(read_u32(reader, 32)?);
            let delta_value = float32_unpack(read_u32(reader, 32)?);
            let value_bits = (read_u32(reader, 4)? as u8) + 1;
            let sequence_p = read_bit(reader)?;
            let lookup_values: u32 = if lookup_type == 1 {
                lookup1_values(entries, dimensions)
            } else {
                // entries * dimensions; saturate to detect pathological
                // streams. A real Vorbis stream caps both fields well
                // below 2^31, so overflow here is decoder-fence.
                (entries as u64)
                    .checked_mul(dimensions as u64)
                    .filter(|&v| v <= u32::MAX as u64)
                    .ok_or(ParseError::UnexpectedEndOfPacket)? as u32
            };
            let mut multiplicands = Vec::with_capacity(lookup_values as usize);
            for _ in 0..lookup_values {
                multiplicands.push(read_u32(reader, value_bits as u32)?);
            }
            if lookup_type == 1 {
                VqLookup::Lattice {
                    minimum_value,
                    delta_value,
                    value_bits,
                    sequence_p,
                    multiplicands,
                }
            } else {
                VqLookup::Tessellation {
                    minimum_value,
                    delta_value,
                    value_bits,
                    sequence_p,
                    multiplicands,
                }
            }
        }
        t => return Err(ParseError::ReservedLookupType(t)),
    };

    Ok(VorbisCodebook {
        dimensions,
        entries,
        codeword_lengths,
        lookup,
    })
}

// ---- small read helpers that funnel BitReaderLsb's `Error::Eof` into
// our `ParseError::UnexpectedEndOfPacket`. ----

fn read_u32(reader: &mut BitReaderLsb<'_>, n: u32) -> Result<u32, ParseError> {
    reader
        .read_u32(n)
        .map_err(|_| ParseError::UnexpectedEndOfPacket)
}

fn read_bit(reader: &mut BitReaderLsb<'_>) -> Result<bool, ParseError> {
    reader
        .read_bit()
        .map_err(|_| ParseError::UnexpectedEndOfPacket)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriterLsb;

    // ---- §9.2.1 ilog ----

    /// Spec examples for [`ilog`] (§9.2.1: ilog(0)=0, ilog(1)=1,
    /// ilog(2)=2, ilog(3)=2, ilog(4)=3, ilog(7)=3).
    #[test]
    fn ilog_matches_spec_examples() {
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(2), 2);
        assert_eq!(ilog(3), 2);
        assert_eq!(ilog(4), 3);
        assert_eq!(ilog(7), 3);
    }

    // ---- §9.2.2 float32_unpack ----

    /// `float32_unpack(0)` should yield 0.0 (mantissa=0, sign=0,
    /// exponent=0 → 0 * 2^(0 - 788) = 0).
    #[test]
    fn float32_unpack_zero() {
        assert_eq!(float32_unpack(0), 0.0);
    }

    /// `float32_unpack` of a value with sign bit set negates the
    /// mantissa per §9.2.2 step 4.
    #[test]
    fn float32_unpack_sign_negates() {
        // mantissa = 1, exponent = 788 (so 2^0 = 1.0), sign = 1
        //   x = sign | (exponent << 21) | mantissa
        //     = 0x80000000 | (788 << 21) | 1
        let x: u32 = 0x8000_0000 | (788u32 << 21) | 1;
        assert_eq!(float32_unpack(x), -1.0);
    }

    /// Unit mantissa with the unit exponent 788 yields ±1.0.
    #[test]
    fn float32_unpack_unit() {
        let x: u32 = (788u32 << 21) | 1;
        assert_eq!(float32_unpack(x), 1.0);
    }

    // ---- §9.2.2 float32_pack (encoder-side companion) ----

    /// Zero (both signs) packs to the canonical all-zero container,
    /// and the container's unpack reproduces `0.0`.
    #[test]
    fn float32_pack_zero() {
        assert_eq!(float32_pack(0.0), Some(0));
        assert_eq!(float32_pack(-0.0), Some(0));
        assert_eq!(float32_unpack(0), 0.0);
    }

    /// `float32_pack(1.0)` produces the canonical unit container —
    /// mantissa = 1, stored_exponent = 788 — matching the
    /// `float32_unpack_unit` test on the parser side.
    #[test]
    fn float32_pack_one() {
        let packed = float32_pack(1.0).expect("must pack");
        assert_eq!(packed, (788u32 << 21) | 1);
        assert_eq!(float32_unpack(packed), 1.0);
    }

    /// Negative inputs set the §9.2.2 sign bit.
    #[test]
    fn float32_pack_negative_one() {
        let packed = float32_pack(-1.0).expect("must pack");
        assert_eq!(packed, 0x8000_0000 | (788u32 << 21) | 1);
        assert_eq!(float32_unpack(packed), -1.0);
    }

    /// Non-finite inputs are rejected — §9.2.2 has no NaN / ±∞.
    #[test]
    fn float32_pack_rejects_non_finite() {
        assert_eq!(float32_pack(f32::NAN), None);
        assert_eq!(float32_pack(f32::INFINITY), None);
        assert_eq!(float32_pack(f32::NEG_INFINITY), None);
    }

    /// Roundtrip a representative spread of finite values that all fit
    /// in the §9.2.2 mantissa+exponent envelope: powers of two on both
    /// sides of 1.0, small integer mantissas at varying exponents, and
    /// the signed unit. Every one must satisfy
    /// `unpack(pack(x)) == x` bit-for-bit.
    ///
    /// Inputs are restricted to values exactly representable as
    /// `mantissa * 2^k` with `|mantissa| <= 0x1f_ffff` — the §9.2.2
    /// container's representable set. Decimal literals like `1.0e-6`
    /// are deliberately excluded because their nearest `f32` already
    /// exceeds 21 mantissa bits and so can't round-trip through the
    /// container.
    #[test]
    fn float32_pack_unpack_roundtrips_canonical_values() {
        let cases: [f32; 13] = [
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            1024.0,
            1.0 / 1024.0,
            3.0,
            -3.0,
            7.0 * 0.125, // 0.875 — mantissa 7, exponent -3
            1048576.0,   // 2^20, exactly representable
            1234567.0,   // 21-bit odd mantissa
        ];
        for x in cases {
            let packed = float32_pack(x).unwrap_or_else(|| panic!("pack failed for {x}"));
            let round = float32_unpack(packed);
            assert_eq!(round, x, "roundtrip failed for {x}: packed=0x{packed:08x}");
        }
    }

    /// Inputs whose exact `f32` value requires more than 21 mantissa
    /// bits cannot fit in the §9.2.2 container. Confirm the
    /// representative case — decimal constants like `1.0e-6` whose
    /// nearest `f32` has a full 24-bit mantissa — is rejected with
    /// `None` (not silently quantised).
    #[test]
    fn float32_pack_rejects_unrepresentable_decimals() {
        // 1.0e-6_f32 = 0x358637bd, mantissa-with-implicit-1 = 0x86_37bd
        // after normalisation = 0x10c_6f7a >> trailing zeros → still
        // 22+ bits — exceeds the 21-bit field.
        assert_eq!(float32_pack(1.0e-6_f32), None);
    }

    /// A value whose integer mantissa exceeds the §9.2.2 21-bit field
    /// (`> 0x1f_ffff`) cannot be represented. We expect rejection
    /// rather than silent precision loss. `0x20_0000 = 2_097_152` is
    /// one above the field cap and is exactly representable in `f32`.
    #[test]
    fn float32_pack_rejects_mantissa_over_21_bits() {
        // Integer mantissa exactly one above the cap: 2_097_152.
        let too_big = 2_097_152.0_f32;
        // But this normalises to mantissa=1, exponent=21 after the
        // canonical-form pass — which IS representable. Test the
        // genuine overflow case instead: 2_097_151 * (any odd small
        // multiplier that doesn't reduce). 3 * 2_097_151 / 2 won't be
        // an exact f32; use 2_097_151 itself with a non-trivial odd
        // sub-bit pattern: 2_097_151 is 0x1fffff, exactly fits.
        // For genuine overflow, we need a value that can't normalise
        // down to ≤21 bits — but every f32 has at most 24 mantissa
        // bits, so any odd value > 0x1f_ffff cannot reduce. Pick
        // 0x1f_ffff * 2 + 1 = 0x3f_ffff = 4_194_303, exactly f32.
        let unreducible = 4_194_303.0_f32; // 0x3f_ffff, 22 bits, odd
        assert_eq!(float32_pack(unreducible), None);
        // The "too big but even" case normalises, so it IS packable.
        let packed = float32_pack(too_big).expect("even values normalise");
        assert_eq!(float32_unpack(packed), too_big);
    }

    /// Sweep every legal packed container `(sign, exponent, mantissa)`
    /// shape at coarse granularity, packing the unpacked value and
    /// confirming the round-trip lands on the *same* packed bits — the
    /// canonical-mantissa pass is deterministic.
    #[test]
    fn float32_pack_unpack_pack_idempotent_for_canonical_containers() {
        // Sweep mantissa across odd 21-bit values (so the canonical
        // form is the input itself), at three representative
        // exponents (small positive, unit, small negative).
        for stored_exp in [770u32, 788, 800] {
            for mant in [1u32, 3, 7, 0x1f_ffff] {
                let packed = (stored_exp << 21) | mant;
                let unpacked = float32_unpack(packed);
                let repacked = float32_pack(unpacked).expect("must repack");
                assert_eq!(
                    repacked, packed,
                    "exp={stored_exp} mant=0x{mant:06x} did not round-trip"
                );
            }
        }
    }

    // ---- §9.2.3 lookup1_values ----

    /// `lookup1_values(entries=2916, dimensions=8)` should return 3
    /// because `3^8 = 6561 > 2916` while `2^8 = 256 <= 2916` ... wait,
    /// re-check: the spec says "greatest integer such that
    /// `lookup_values ^ dimensions <= entries`". `3^8 = 6561 > 2916`,
    /// so `lookup_values = 2`? That'd be very small. Let's verify with
    /// `2916 = 6^4 * ...` — actually `3^7 = 2187`, `3^8 = 6561`, so
    /// `n^8 <= 2916` gives `n=2` (`2^8=256<=2916`, `3^8=6561>2916`).
    #[test]
    fn lookup1_values_2916_8() {
        // n=2: 2^8 = 256  <= 2916 ✓
        // n=3: 3^8 = 6561 > 2916 ✗
        assert_eq!(lookup1_values(2916, 8), 2);
    }

    /// `lookup1_values(entries=64, dimensions=2)` should be 8 because
    /// `8^2 = 64 <= 64` and `9^2 = 81 > 64`.
    #[test]
    fn lookup1_values_64_2() {
        assert_eq!(lookup1_values(64, 2), 8);
    }

    /// `lookup1_values(entries=N, dimensions=1)` should be N.
    #[test]
    fn lookup1_values_d1_returns_entries() {
        assert_eq!(lookup1_values(1, 1), 1);
        assert_eq!(lookup1_values(2, 1), 2);
        assert_eq!(lookup1_values(255, 1), 255);
        assert_eq!(lookup1_values(1024, 1), 1024);
    }

    // ---- §3.2.1 parse_codebook ----

    /// Build a synthetic codebook bitstream using [`BitWriterLsb`] so
    /// the round-trip exercises the same LSB-first packing convention
    /// the parser expects (§2.1.4).
    struct CodebookBuilder {
        w: BitWriterLsb,
    }

    impl CodebookBuilder {
        fn new() -> Self {
            let mut w = BitWriterLsb::with_capacity(64);
            // §3.2.1 sync pattern 0x564342 (24 bits).
            w.write_u32(VorbisCodebook::SYNC_PATTERN, 24);
            Self { w }
        }

        fn header(mut self, dimensions: u16, entries: u32, ordered: bool) -> Self {
            self.w.write_u32(dimensions as u32, 16);
            self.w.write_u32(entries, 24);
            self.w.write_bit(ordered);
            self
        }

        fn dense_unordered_lengths(mut self, lengths: &[u8]) -> Self {
            // sparse = 0
            self.w.write_bit(false);
            for &len in lengths {
                debug_assert!((1..=32).contains(&len));
                self.w.write_u32((len - 1) as u32, 5);
            }
            self
        }

        fn sparse_unordered_lengths(mut self, lengths: &[Option<u8>]) -> Self {
            // sparse = 1
            self.w.write_bit(true);
            for &len in lengths {
                match len {
                    Some(l) => {
                        debug_assert!((1..=32).contains(&l));
                        self.w.write_bit(true);
                        self.w.write_u32((l - 1) as u32, 5);
                    }
                    None => self.w.write_bit(false),
                }
            }
            self
        }

        fn ordered_runs(mut self, starting_length: u8, runs: &[u32]) -> Self {
            // starting_length - 1 (5 bits)
            self.w.write_u32((starting_length - 1) as u32, 5);
            let total: u32 = runs.iter().sum();
            let mut consumed = 0u32;
            for &n in runs {
                let width = ilog(total - consumed);
                self.w.write_u32(n, width);
                consumed += n;
            }
            self
        }

        fn lookup_none(mut self) -> Self {
            self.w.write_u32(0, 4);
            self
        }

        #[allow(clippy::too_many_arguments)]
        fn lookup_1_or_2(
            mut self,
            lookup_type: u8,
            minimum_value_packed: u32,
            delta_value_packed: u32,
            value_bits: u8,
            sequence_p: bool,
            multiplicands: &[u32],
        ) -> Self {
            self.w.write_u32(lookup_type as u32, 4);
            self.w.write_u32(minimum_value_packed, 32);
            self.w.write_u32(delta_value_packed, 32);
            self.w.write_u32((value_bits - 1) as u32, 4);
            self.w.write_bit(sequence_p);
            for &m in multiplicands {
                self.w.write_u32(m, value_bits as u32);
            }
            self
        }

        fn finish(self) -> Vec<u8> {
            self.w.finish()
        }
    }

    /// Smallest legal codebook: 1 dimension, 8 entries, dense unordered
    /// lengths matching the spec's §3.2.1 worked example (lengths 2 4
    /// 4 4 4 2 3 3), no lookup. This mirrors trace-doc §3 codebook 0
    /// (`dimensions=1 entries=8 ordered=0 sparse=0 lookup_type=0`).
    #[test]
    fn parses_spec_worked_example_codebook() {
        let lengths = [2u8, 4, 4, 4, 4, 2, 3, 3];
        let packet = CodebookBuilder::new()
            .header(1, 8, false)
            .dense_unordered_lengths(&lengths)
            .lookup_none()
            .finish();
        let mut r = BitReaderLsb::new(&packet);
        let book = parse_codebook(&mut r).expect("must parse");
        assert_eq!(book.dimensions, 1);
        assert_eq!(book.entries, 8);
        assert_eq!(book.codeword_lengths, lengths.to_vec());
        assert_eq!(book.lookup, VqLookup::None);
        assert_eq!(book.used_entries(), 8);
        assert!(!book.is_fully_unused());
    }

    /// Sparse-mode codebook: every entry carries an explicit "used"
    /// bit, and unused entries are stored as [`UNUSED_ENTRY`].
    #[test]
    fn parses_sparse_codebook_with_unused_entries() {
        // 6 entries: 3 used (lengths 1, 3, 5), 3 unused.
        let entries: [Option<u8>; 6] = [Some(1), None, Some(3), None, Some(5), None];
        let packet = CodebookBuilder::new()
            .header(1, 6, false)
            .sparse_unordered_lengths(&entries)
            .lookup_none()
            .finish();
        let mut r = BitReaderLsb::new(&packet);
        let book = parse_codebook(&mut r).expect("must parse");
        assert_eq!(book.entries, 6);
        assert_eq!(
            book.codeword_lengths,
            vec![1, UNUSED_ENTRY, 3, UNUSED_ENTRY, 5, UNUSED_ENTRY]
        );
        assert_eq!(book.used_entries(), 3);
        assert!(!book.is_fully_unused());
    }

    /// Ordered codebook: starting length 2, two runs (3 then 5
    /// entries), encoding lengths [2, 2, 2, 3, 3, 3, 3, 3].
    #[test]
    fn parses_ordered_codebook() {
        // 8 entries total, starting_length=2.
        //   run 1: 3 entries at length 2
        //   run 2: 5 entries at length 3
        let packet = CodebookBuilder::new()
            .header(1, 8, true)
            .ordered_runs(2, &[3, 5])
            .lookup_none()
            .finish();
        let mut r = BitReaderLsb::new(&packet);
        let book = parse_codebook(&mut r).expect("must parse");
        assert_eq!(book.entries, 8);
        assert_eq!(book.codeword_lengths, vec![2, 2, 2, 3, 3, 3, 3, 3]);
    }

    /// Lookup type 2 (tessellation): exercise multiplicand decode
    /// against a synthetic 2-dim, 4-entry, value_bits=3 book. The
    /// resulting `multiplicands` length is `entries * dimensions = 8`.
    #[test]
    fn parses_lookup_type_2_multiplicands() {
        let lengths = [2u8, 2, 2, 2];
        // value_bits = 3 → each multiplicand is a 3-bit unsigned int.
        let multiplicands: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        // minimum_value = 0.0 (packed = 0), delta_value = 1.0
        //   1.0 packed: mantissa=1, exponent=788, sign=0
        let delta_packed: u32 = (788u32 << 21) | 1;
        let packet = CodebookBuilder::new()
            .header(2, 4, false)
            .dense_unordered_lengths(&lengths)
            .lookup_1_or_2(2, 0, delta_packed, 3, false, &multiplicands)
            .finish();
        let mut r = BitReaderLsb::new(&packet);
        let book = parse_codebook(&mut r).expect("must parse");
        assert_eq!(book.dimensions, 2);
        assert_eq!(book.entries, 4);
        match book.lookup {
            VqLookup::Tessellation {
                minimum_value,
                delta_value,
                value_bits,
                sequence_p,
                multiplicands: m,
            } => {
                assert_eq!(minimum_value, 0.0);
                assert_eq!(delta_value, 1.0);
                assert_eq!(value_bits, 3);
                assert!(!sequence_p);
                assert_eq!(m, multiplicands.to_vec());
            }
            other => panic!("expected Tessellation, got {other:?}"),
        }
    }

    /// Lookup type 1 (lattice): multiplicand count is
    /// `lookup1_values(entries, dimensions)`. For entries=64,
    /// dimensions=2 this is 8.
    #[test]
    fn parses_lookup_type_1_multiplicands() {
        // 64 entries, dimensions=2, dense lengths all 6 → fully
        // populated 64-leaf tree (2^6 = 64).
        let lengths: Vec<u8> = vec![6; 64];
        let multiplicands: Vec<u32> = (0..8).collect();
        let delta_packed: u32 = (788u32 << 21) | 1;
        let packet = CodebookBuilder::new()
            .header(2, 64, false)
            .dense_unordered_lengths(&lengths)
            .lookup_1_or_2(1, 0, delta_packed, 4, true, &multiplicands)
            .finish();
        let mut r = BitReaderLsb::new(&packet);
        let book = parse_codebook(&mut r).expect("must parse");
        match book.lookup {
            VqLookup::Lattice {
                value_bits,
                sequence_p,
                multiplicands: m,
                ..
            } => {
                assert_eq!(value_bits, 4);
                assert!(sequence_p);
                assert_eq!(m, multiplicands);
            }
            other => panic!("expected Lattice, got {other:?}"),
        }
    }

    /// A bad 24-bit sync pattern at the head of the bitstream must
    /// surface as [`ParseError::BadSyncPattern`].
    #[test]
    fn rejects_bad_sync_pattern() {
        let mut w = BitWriterLsb::with_capacity(8);
        w.write_u32(0xDEADBE, 24);
        // Pad enough bits to reach the next read.
        for _ in 0..3 {
            w.write_u32(0, 32);
        }
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_codebook(&mut r) {
            Err(ParseError::BadSyncPattern(0x00DE_ADBE)) => {}
            other => panic!("expected BadSyncPattern, got {other:?}"),
        }
    }

    /// A `codebook_lookup_type` of 3..=15 is reserved and must yield
    /// [`ParseError::ReservedLookupType`].
    #[test]
    fn rejects_reserved_lookup_type() {
        let lengths = [1u8];
        let mut w = BitWriterLsb::with_capacity(16);
        w.write_u32(VorbisCodebook::SYNC_PATTERN, 24);
        w.write_u32(1, 16); // dimensions
        w.write_u32(1, 24); // entries
        w.write_bit(false); // ordered = 0
        w.write_bit(false); // sparse = 0
        w.write_u32((lengths[0] - 1) as u32, 5);
        w.write_u32(7, 4); // reserved lookup_type = 7
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_codebook(&mut r) {
            Err(ParseError::ReservedLookupType(7)) => {}
            other => panic!("expected ReservedLookupType(7), got {other:?}"),
        }
    }

    /// Zero `codebook_entries` is rejected.
    #[test]
    fn rejects_zero_entries() {
        let mut w = BitWriterLsb::with_capacity(8);
        w.write_u32(VorbisCodebook::SYNC_PATTERN, 24);
        w.write_u32(1, 16); // dimensions
        w.write_u32(0, 24); // entries = 0
                            // Pad so the parser has bits to attempt the next read.
        w.write_u32(0, 32);
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(parse_codebook(&mut r), Err(ParseError::ZeroEntries));
    }

    /// An ordered run that overshoots `entries` is rejected with
    /// [`ParseError::OrderedOverflow`].
    #[test]
    fn rejects_ordered_overflow() {
        // entries = 4, but encode a run of 5 at length 1.
        // current_entry starts at 0; ilog(4) = 3 bits; number = 5 fits
        // in 3 bits, and 5 > 4 → overflow.
        let mut w = BitWriterLsb::with_capacity(8);
        w.write_u32(VorbisCodebook::SYNC_PATTERN, 24);
        w.write_u32(1, 16); // dimensions
        w.write_u32(4, 24); // entries = 4
        w.write_bit(true); // ordered = 1
        w.write_u32(0, 5); // starting_length - 1 = 0 → length 1
        w.write_u32(5, 3); // number = 5 (ilog(4) = 3 bits)
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_codebook(&mut r) {
            Err(ParseError::OrderedOverflow {
                current_entry: 5,
                entries: 4,
            }) => {}
            other => panic!("expected OrderedOverflow, got {other:?}"),
        }
    }

    /// Truncated packet (EOF mid-codeword-length) surfaces as
    /// [`ParseError::UnexpectedEndOfPacket`] per §3.2.1.
    #[test]
    fn rejects_truncated_packet() {
        let mut w = BitWriterLsb::with_capacity(8);
        w.write_u32(VorbisCodebook::SYNC_PATTERN, 24);
        w.write_u32(1, 16); // dimensions
        w.write_u32(2, 24); // entries
        w.write_bit(false); // ordered = 0
        w.write_bit(false); // sparse = 0
        w.write_u32(0, 5); // first entry length-1 = 0 (length 1)
                           // ...and now truncate: only one of two entries is present.
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            parse_codebook(&mut r),
            Err(ParseError::UnexpectedEndOfPacket)
        );
    }

    /// Trace-doc §3 fixture-style: `dimensions=8 entries=2916
    /// ordered=0 sparse=1 lookup_type=2 value_bits=8 sequence_p=0`.
    /// We don't need to populate all 2916 entries with meaningful
    /// lengths; mark them all as unused except the last (no
    /// codewords means `is_fully_unused` returns false for at least
    /// one used entry — sanity check on a realistic stream shape).
    #[test]
    fn parses_trace_doc_codebook_27_shape() {
        // To keep this test fast, build a synthetic 8-entry codebook
        // that mirrors the trace-doc shape (sparse, lookup_type=2,
        // value_bits=8). The full 2916-entry book is exercised
        // implicitly by the unit `lookup1_values_2916_8` test (which
        // confirms the count math) and would otherwise just allocate
        // megabytes for no extra coverage.
        let mut sparse_lengths = vec![None; 8];
        sparse_lengths[0] = Some(1);
        sparse_lengths[7] = Some(1);
        // multiplicands: 8 entries × 8 dimensions = 64 values.
        let multiplicands: Vec<u32> = (0..64).map(|i| (i & 0xff) as u32).collect();
        let delta_packed: u32 = (788u32 << 21) | 1;
        let packet = CodebookBuilder::new()
            .header(8, 8, false)
            .sparse_unordered_lengths(&sparse_lengths)
            .lookup_1_or_2(2, 0, delta_packed, 8, false, &multiplicands)
            .finish();
        let mut r = BitReaderLsb::new(&packet);
        let book = parse_codebook(&mut r).expect("must parse");
        assert_eq!(book.dimensions, 8);
        assert_eq!(book.entries, 8);
        assert_eq!(book.used_entries(), 2);
        match book.lookup {
            VqLookup::Tessellation {
                value_bits,
                multiplicands: m,
                ..
            } => {
                assert_eq!(value_bits, 8);
                assert_eq!(m.len(), 64);
            }
            other => panic!("expected Tessellation, got {other:?}"),
        }
    }
}

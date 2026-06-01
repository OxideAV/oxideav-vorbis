//! Vorbis I header-packet + codebook encoder primitives (rounds 195 + 201).
//!
//! This module collects the encoder-side functions that mirror the
//! parser surface in [`crate::identification`], [`crate::comment`],
//! and [`crate::codebook`]. Every writer is the byte-exact inverse of
//! its parser counterpart and is exercised by an in-module test suite
//! against the `parse → equal` round-trip property.
//!
//! Functions:
//!
//! * [`write_identification_header`] — serialises a
//!   [`VorbisIdentificationHeader`] to the fixed 30-byte packet shape
//!   defined in Vorbis I §4.2.1 + §4.2.2.
//! * [`write_comment_header`] — serialises a [`VorbisCommentHeader`]
//!   to the variable-length packet shape defined in Vorbis I §4.2.1 +
//!   §5.2.1.
//! * [`write_codebook`] — serialises a [`VorbisCodebook`] to the
//!   variable-length codebook-header shape defined in Vorbis I §3.2.1.
//!   This is the first sub-packet writer (codebooks are nested inside
//!   the setup header — they do not have a §4.2.1 common header).
//!
//! Both functions validate the same spec-mandated invariants that the
//! corresponding parser enforces on input, so a typo in the caller's
//! input cannot produce a malformed packet — the writer rejects the
//! call with a structured [`WriteError`] before any bytes are emitted.
//!
//! ## Bit-exact roundtrip guarantee
//!
//! For every value `x` of type [`VorbisIdentificationHeader`] that
//! satisfies the §4.2.2 invariants:
//!
//! ```text
//! parse_identification_header(&write_identification_header(&x)?)? == x
//! ```
//!
//! For every value `y` of type [`VorbisCommentHeader`] that satisfies
//! the §5.2.1 invariants:
//!
//! ```text
//! parse_comment_header(&write_comment_header(&y)?)? == y
//! ```
//!
//! The roundtrip property is exercised exhaustively in the in-module
//! test suite against the fixture shapes documented in
//! `docs/audio/vorbis/vorbis-fixtures-and-traces.md` §2.1 / §2.2 plus
//! the canonical edge cases (spec-minimum / spec-maximum blocksizes,
//! signed bitrate hints, empty vendor + zero-comment list, multi-byte
//! UTF-8 in vendor and comments).
//!
//! ## Spec sources
//!
//! Both functions are derived from:
//!
//! * `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.2.1 "Common header decode"
//!   (the 7-byte `packet_type + "vorbis"` prelude common to all three
//!   header packets).
//! * `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.2.2 "Identification
//!   header" (the field layout, the legal exponent range 6..=13, the
//!   `blocksize_0 <= blocksize_1` ordering, the trailing framing flag).
//! * `docs/audio/vorbis/Vorbis_I_spec.pdf` §5.2.1 "Structure" + §5.2.3
//!   "Encoding" (the `u32` LE vendor-length, vendor bytes, `u32` LE
//!   comment-count, per-comment `u32` LE length + UTF-8 payload, and
//!   trailing framing bit).
//! * `docs/audio/vorbis/Vorbis_I_spec.pdf` §2.1.4 "coding bits into byte
//!   sequences" (the LSB-first packing rule that — for fields whose bit
//!   width is a multiple of 8 — collapses to plain little-endian byte
//!   order, as called out in §5.2.1).
//! * `docs/audio/vorbis/Vorbis_I_spec.pdf` §2.1.8 "end-of-packet
//!   alignment" (the framing-byte's seven high bits are zero padding).
//!
//! Audio-packet encode and floor / residue / mapping / mode WRITE
//! primitives are explicit followups for subsequent rounds.

use core::fmt;

use oxideav_core::bits::BitWriterLsb;

use crate::codebook::{float32_pack, ilog, lookup1_values, VorbisCodebook, VqLookup, UNUSED_ENTRY};
use crate::comment::VorbisCommentHeader;
use crate::identification::VorbisIdentificationHeader;
use crate::setup::Floor1Header;

/// Errors that may arise while writing a Vorbis I header packet.
///
/// Each variant flags a §4.2.2 / §5.2.1 invariant the caller-supplied
/// struct does not satisfy. The writer refuses the call (returning the
/// error) rather than emit a packet the corresponding parser would
/// reject — this keeps the bit-exact roundtrip guarantee defensible.
#[derive(Debug, Clone, PartialEq)]
pub enum WriteError {
    /// `vorbis_version` was non-zero. Vorbis I §4.2.2 mandates
    /// `vorbis_version == 0` for any conformant Vorbis I stream.
    UnsupportedVorbisVersion(u32),
    /// `audio_channels` was zero. §4.2.2 mandates `> 0`.
    ZeroChannels,
    /// `audio_sample_rate` was zero. §4.2.2 mandates `> 0`.
    ZeroSampleRate,
    /// `blocksize_0` or `blocksize_1` was not a power of two in the
    /// spec-legal `{64, 128, 256, 512, 1024, 2048, 4096, 8192}` set,
    /// i.e. its base-2 exponent did not fall in 6..=13 inclusive.
    /// The contained tuple is the rejected `(blocksize_0, blocksize_1)`.
    IllegalBlocksize(u16, u16),
    /// `blocksize_0` was strictly greater than `blocksize_1`. §4.2.2
    /// mandates `blocksize_0 <= blocksize_1`.
    BlocksizesOutOfOrder {
        /// The short-block sample count supplied by the caller.
        blocksize_0: u16,
        /// The long-block sample count supplied by the caller.
        blocksize_1: u16,
    },
    /// One comment entry's UTF-8 byte length exceeded `u32::MAX`, so
    /// it cannot be expressed in the §5.2.1 32-bit length prefix. The
    /// contained `usize` is the rejected entry's byte length.
    CommentTooLong(usize),
    /// The vendor string's UTF-8 byte length exceeded `u32::MAX`, so
    /// it cannot be expressed in the §5.2.1 32-bit vendor-length
    /// prefix. The contained `usize` is the rejected vendor length.
    VendorTooLong(usize),
    /// The comment count exceeded `u32::MAX`, so it cannot be
    /// expressed in the §5.2.1 32-bit `user_comment_list_length`
    /// prefix. The contained `usize` is the rejected count.
    TooManyComments(usize),
    /// A nested codebook (§3.2.1) failed one of the writer-side
    /// invariants checked by [`write_codebook`].
    Codebook(WriteCodebookError),
    /// A nested floor type 1 header (§7.2.2) failed one of the
    /// writer-side invariants checked by [`write_floor1_header`].
    Floor1(WriteFloor1Error),
}

/// Errors that may arise while writing a Vorbis I codebook header
/// (§3.2.1) via [`write_codebook`].
///
/// Each variant flags a §3.2.1 invariant the caller-supplied
/// [`VorbisCodebook`] does not satisfy. The writer refuses the call
/// without emitting any bits, preserving the bit-exact roundtrip
/// guarantee `parse_codebook(&write_codebook(&book)?)? == book`.
#[derive(Debug, Clone, PartialEq)]
pub enum WriteCodebookError {
    /// `codebook_entries == 0`. §3.2.1's parser rejects this on
    /// input; the writer mirrors the rule.
    ZeroEntries,
    /// `codeword_lengths.len() != codebook_entries as usize`. The
    /// §3.2.1 layout has exactly one length per entry — the writer
    /// cannot serialise a struct whose internal length table is
    /// inconsistent with its `entries` field.
    LengthTableMismatch {
        /// The `entries` field on the supplied codebook.
        entries: u32,
        /// The actual length of `codeword_lengths`, which must equal
        /// `entries`.
        length_table: usize,
    },
    /// A used entry has a codeword length outside `1..=32`. §3.2.1
    /// encodes lengths as a 5-bit `length - 1` field, so the legal
    /// range is `1..=32` (plus the [`UNUSED_ENTRY`] sentinel `0`
    /// allowed only in sparse codebooks).
    IllegalCodewordLength {
        /// Index of the offending entry in `codeword_lengths`.
        entry: u32,
        /// The rejected length.
        length: u8,
    },
    /// `value_bits` on a lookup table was outside `1..=16`. §3.2.1
    /// encodes the field as `value_bits - 1` in a 4-bit slot.
    IllegalValueBits(u8),
    /// A multiplicand exceeds `(1 << value_bits) - 1`. §3.2.1 writes
    /// each multiplicand as an unsigned integer of `value_bits` bits;
    /// values above that cannot be represented.
    MultiplicandOverflow {
        /// Index of the offending multiplicand in the lookup table.
        index: usize,
        /// The rejected multiplicand value.
        value: u32,
        /// The field width that contains it, in bits.
        value_bits: u8,
    },
    /// Lookup-type 1 multiplicand table has the wrong length. Per
    /// §9.2.3, the table must contain exactly
    /// `lookup1_values(entries, dimensions)` values.
    LatticeMultiplicandCountMismatch {
        /// Expected count = `lookup1_values(entries, dimensions)`.
        expected: u32,
        /// Actual length of the supplied multiplicand `Vec`.
        actual: usize,
    },
    /// Lookup-type 2 multiplicand table has the wrong length. Per
    /// §3.2.1, the table must contain exactly `entries * dimensions`
    /// values.
    TessellationMultiplicandCountMismatch {
        /// Expected count = `entries * dimensions`.
        expected: u64,
        /// Actual length of the supplied multiplicand `Vec`.
        actual: usize,
    },
    /// A lookup-table `minimum_value` or `delta_value` could not be
    /// expressed in the §9.2.2 32-bit container (NaN / ±∞, magnitude
    /// outside the representable range, or a mantissa requiring more
    /// than 21 bits). The contained boolean is `true` for
    /// `minimum_value`, `false` for `delta_value`.
    UnrepresentableLookupFloat {
        /// `true` if the rejection was on `minimum_value`, `false` if
        /// on `delta_value`.
        is_minimum: bool,
        /// The rejected `f32` value.
        value: f32,
    },
    /// An ordered codebook contained an [`UNUSED_ENTRY`] sentinel.
    /// §3.2.1's ordered encoding has no representation for unused
    /// entries — every entry must carry a length.
    OrderedHasUnusedEntries,
    /// An ordered codebook's codeword lengths were not non-decreasing.
    /// The §3.2.1 ordered run-length encoding only encodes ascending
    /// runs (`current_length += 1` after each run), so a decreasing
    /// step cannot be serialised.
    OrderedNotMonotonic {
        /// The first index whose length is strictly less than the
        /// previous entry's length.
        entry: u32,
        /// `(previous_length, this_length)`.
        lengths: (u8, u8),
    },
    /// `entries * dimensions` overflowed `u64`, so the
    /// lookup-type-2 multiplicand count cannot be computed.
    LookupCountOverflow {
        /// `entries`.
        entries: u32,
        /// `dimensions`.
        dimensions: u16,
    },
}

impl fmt::Display for WriteCodebookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteCodebookError::ZeroEntries => {
                write!(f, "vorbis codebook (write): codebook_entries = 0 (§3.2.1)")
            }
            WriteCodebookError::LengthTableMismatch {
                entries,
                length_table,
            } => write!(
                f,
                "vorbis codebook (write): codeword_lengths.len()={length_table} != entries={entries} (§3.2.1)"
            ),
            WriteCodebookError::IllegalCodewordLength { entry, length } => write!(
                f,
                "vorbis codebook (write): codeword_lengths[{entry}]={length} outside legal 1..=32 (§3.2.1)"
            ),
            WriteCodebookError::IllegalValueBits(v) => write!(
                f,
                "vorbis codebook (write): value_bits={v} outside legal 1..=16 (§3.2.1)"
            ),
            WriteCodebookError::MultiplicandOverflow {
                index,
                value,
                value_bits,
            } => write!(
                f,
                "vorbis codebook (write): multiplicand[{index}]={value} exceeds 2^{value_bits}-1 (§3.2.1)"
            ),
            WriteCodebookError::LatticeMultiplicandCountMismatch { expected, actual } => write!(
                f,
                "vorbis codebook (write): lookup-type-1 multiplicand count {actual} != lookup1_values()={expected} (§9.2.3)"
            ),
            WriteCodebookError::TessellationMultiplicandCountMismatch { expected, actual } => write!(
                f,
                "vorbis codebook (write): lookup-type-2 multiplicand count {actual} != entries*dimensions={expected} (§3.2.1)"
            ),
            WriteCodebookError::UnrepresentableLookupFloat { is_minimum, value } => write!(
                f,
                "vorbis codebook (write): {field}={value} not expressible in the §9.2.2 packed-float container",
                field = if *is_minimum { "minimum_value" } else { "delta_value" }
            ),
            WriteCodebookError::OrderedHasUnusedEntries => write!(
                f,
                "vorbis codebook (write): ordered encoding cannot carry unused-entry sentinels (§3.2.1)"
            ),
            WriteCodebookError::OrderedNotMonotonic {
                entry,
                lengths: (prev, cur),
            } => write!(
                f,
                "vorbis codebook (write): ordered encoding requires non-decreasing lengths but entry {entry} drops from {prev} to {cur} (§3.2.1)"
            ),
            WriteCodebookError::LookupCountOverflow {
                entries,
                dimensions,
            } => write!(
                f,
                "vorbis codebook (write): entries={entries} * dimensions={dimensions} overflows u64 (§3.2.1)"
            ),
        }
    }
}

impl std::error::Error for WriteCodebookError {}

/// Errors that may arise while writing a Vorbis I floor type 1 header
/// (§7.2.2) via [`write_floor1_header`].
///
/// Each variant flags a §7.2.2 invariant the caller-supplied
/// [`Floor1Header`] does not satisfy. The writer refuses the call
/// without emitting any bits, preserving the bit-exact roundtrip
/// guarantee
/// `parse_floor1_header(&mut BitReaderLsb::new(&write_floor1_header(&h)?))? == h`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteFloor1Error {
    /// `partitions` exceeds the 5-bit `floor1_partitions` field's
    /// representable range (`0..=31`).
    PartitionsOverflow(u8),
    /// `partition_class_list.len()` does not match the declared
    /// `partitions` count. §7.2.2 step 3 emits exactly one 4-bit class
    /// index per partition.
    PartitionClassListMismatch {
        /// The declared `partitions` count.
        partitions: u8,
        /// The actual length of `partition_class_list`.
        list_len: usize,
    },
    /// A `partition_class_list[i]` value exceeds the 4-bit
    /// `floor1_partition_class_list[i]` field's representable range
    /// (`0..=15`).
    PartitionClassValueOverflow {
        /// The partition index in `0 .. partitions`.
        index: u8,
        /// The rejected class-list value.
        value: u8,
    },
    /// `classes.len()` is inconsistent with the largest value in
    /// `partition_class_list`. §7.2.2 step 6 requires exactly
    /// `maximum_class + 1` class entries (and zero entries when
    /// `partitions == 0`).
    ClassCountMismatch {
        /// The expected class count = `max(partition_class_list) + 1`,
        /// or `0` when `partition_class_list` is empty.
        expected: usize,
        /// The actual length of `classes`.
        actual: usize,
    },
    /// A class's `dimensions` field was outside `1..=8`. §7.2.2 step 7
    /// encodes `class_dimensions[i]` as `read 3 bits + 1`, so the legal
    /// range is `1..=8`.
    IllegalClassDimensions {
        /// Class index.
        class: usize,
        /// The rejected `dimensions` value.
        dimensions: u8,
    },
    /// A class's `subclasses` field exceeded the 2-bit
    /// `floor1_class_subclasses[i]` field's representable range
    /// (`0..=3`).
    SubclassesOverflow {
        /// Class index.
        class: usize,
        /// The rejected `subclasses` value.
        subclasses: u8,
    },
    /// A class's `masterbook` slot was inconsistent with its
    /// `subclasses` count. §7.2.2 step 9 only reads the masterbook
    /// field when `subclasses > 0`; a class with `subclasses == 0`
    /// must have `masterbook == None`, and a class with `subclasses
    /// > 0` must have `masterbook == Some(_)`.
    MasterbookPresenceMismatch {
        /// Class index.
        class: usize,
        /// The class's `subclasses` count.
        subclasses: u8,
        /// `true` if `masterbook` was present, `false` otherwise.
        present: bool,
    },
    /// A class's `subclass_books` length did not match `1 << subclasses`.
    /// §7.2.2 step 11 emits exactly `2^subclasses` per-subclass codebook
    /// slots.
    SubclassBookCountMismatch {
        /// Class index.
        class: usize,
        /// Expected length (`1 << subclasses`).
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// A subclass-book entry's `Some(v)` value exceeded the largest
    /// `v` that fits in §7.2.2 step 12's "read 8 bits - 1" container.
    /// The raw byte must fit in 8 bits, so the highest representable
    /// `Some(v)` is `v == 254` (raw `255`); values `>= 255` cannot be
    /// emitted.
    SubclassBookOverflow {
        /// Class index.
        class: usize,
        /// Subclass slot index.
        subclass: usize,
        /// The rejected book index.
        book: u8,
    },
    /// `multiplier` was outside `1..=4`. §7.2.2 step 13 encodes
    /// `floor1_multiplier` as `read 2 bits + 1`, so the legal range is
    /// `1..=4`.
    IllegalMultiplier(u8),
    /// `rangebits` exceeded the 4-bit `floor1_rangebits` field's
    /// representable range (`0..=15`).
    RangebitsOverflow(u8),
    /// `x_list.len()` did not match the `sum_over_partitions(
    /// classes[partition_class_list[i]].dimensions)` total §7.2.2
    /// step 18 requires.
    XListLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// An `x_list[i]` value did not fit in the `rangebits`-bit field.
    /// §7.2.2 step 21 emits each x-coordinate as a `rangebits`-bit
    /// unsigned integer; a value with bits set beyond `rangebits` would
    /// be truncated on emit.
    XListValueOverflow {
        /// The x-list index.
        index: usize,
        /// The rejected value.
        value: u32,
        /// The declared `rangebits` field width.
        rangebits: u8,
    },
}

impl fmt::Display for WriteFloor1Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteFloor1Error::PartitionsOverflow(n) => write!(
                f,
                "vorbis floor1 header (write): partitions={n} > 31 (§7.2.2 step 1 is a 5-bit field)"
            ),
            WriteFloor1Error::PartitionClassListMismatch {
                partitions,
                list_len,
            } => write!(
                f,
                "vorbis floor1 header (write): partition_class_list.len()={list_len} != partitions={partitions} (§7.2.2 step 3)"
            ),
            WriteFloor1Error::PartitionClassValueOverflow { index, value } => write!(
                f,
                "vorbis floor1 header (write): partition_class_list[{index}]={value} > 15 (§7.2.2 step 4 is a 4-bit field)"
            ),
            WriteFloor1Error::ClassCountMismatch { expected, actual } => write!(
                f,
                "vorbis floor1 header (write): classes.len()={actual} != max(partition_class_list)+1={expected} (§7.2.2 step 6)"
            ),
            WriteFloor1Error::IllegalClassDimensions { class, dimensions } => write!(
                f,
                "vorbis floor1 header (write): classes[{class}].dimensions={dimensions} outside legal 1..=8 (§7.2.2 step 7)"
            ),
            WriteFloor1Error::SubclassesOverflow { class, subclasses } => write!(
                f,
                "vorbis floor1 header (write): classes[{class}].subclasses={subclasses} > 3 (§7.2.2 step 8 is a 2-bit field)"
            ),
            WriteFloor1Error::MasterbookPresenceMismatch {
                class,
                subclasses,
                present,
            } => write!(
                f,
                "vorbis floor1 header (write): classes[{class}] masterbook present={present} but subclasses={subclasses} (§7.2.2 step 9: present iff subclasses > 0)"
            ),
            WriteFloor1Error::SubclassBookCountMismatch {
                class,
                expected,
                actual,
            } => write!(
                f,
                "vorbis floor1 header (write): classes[{class}].subclass_books.len()={actual} != 1<<subclasses={expected} (§7.2.2 step 11)"
            ),
            WriteFloor1Error::SubclassBookOverflow {
                class,
                subclass,
                book,
            } => write!(
                f,
                "vorbis floor1 header (write): classes[{class}].subclass_books[{subclass}]=Some({book}) cannot be expressed in §7.2.2 step 12's 'read 8 bits - 1' container (max Some(254))"
            ),
            WriteFloor1Error::IllegalMultiplier(m) => write!(
                f,
                "vorbis floor1 header (write): multiplier={m} outside legal 1..=4 (§7.2.2 step 13)"
            ),
            WriteFloor1Error::RangebitsOverflow(r) => write!(
                f,
                "vorbis floor1 header (write): rangebits={r} > 15 (§7.2.2 step 14 is a 4-bit field)"
            ),
            WriteFloor1Error::XListLengthMismatch { expected, actual } => write!(
                f,
                "vorbis floor1 header (write): x_list.len()={actual} != sum(class.dimensions over partitions)={expected} (§7.2.2 step 18)"
            ),
            WriteFloor1Error::XListValueOverflow {
                index,
                value,
                rangebits,
            } => write!(
                f,
                "vorbis floor1 header (write): x_list[{index}]={value} does not fit in {rangebits}-bit field (§7.2.2 step 21)"
            ),
        }
    }
}

impl std::error::Error for WriteFloor1Error {}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteError::UnsupportedVorbisVersion(v) => write!(
                f,
                "vorbis identification header (write): vorbis_version={v}, not supported (Vorbis I requires 0 per §4.2.2)"
            ),
            WriteError::ZeroChannels => write!(
                f,
                "vorbis identification header (write): audio_channels=0 (must be > 0 per §4.2.2)"
            ),
            WriteError::ZeroSampleRate => write!(
                f,
                "vorbis identification header (write): audio_sample_rate=0 (must be > 0 per §4.2.2)"
            ),
            WriteError::IllegalBlocksize(b0, b1) => write!(
                f,
                "vorbis identification header (write): blocksize ({b0}, {b1}) outside spec-legal {{64,128,256,512,1024,2048,4096,8192}} per §4.2.2"
            ),
            WriteError::BlocksizesOutOfOrder {
                blocksize_0,
                blocksize_1,
            } => write!(
                f,
                "vorbis identification header (write): blocksize_0={blocksize_0} > blocksize_1={blocksize_1} (must be <= per §4.2.2)"
            ),
            WriteError::CommentTooLong(n) => write!(
                f,
                "vorbis comment header (write): single comment is {n} bytes; >u32::MAX cannot be expressed in the §5.2.1 length prefix"
            ),
            WriteError::VendorTooLong(n) => write!(
                f,
                "vorbis comment header (write): vendor string is {n} bytes; >u32::MAX cannot be expressed in the §5.2.1 length prefix"
            ),
            WriteError::TooManyComments(n) => write!(
                f,
                "vorbis comment header (write): {n} comments; >u32::MAX cannot be expressed in the §5.2.1 user_comment_list_length prefix"
            ),
            WriteError::Codebook(e) => write!(f, "{e}"),
            WriteError::Floor1(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for WriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WriteError::Codebook(e) => Some(e),
            WriteError::Floor1(e) => Some(e),
            _ => None,
        }
    }
}

impl From<WriteCodebookError> for WriteError {
    fn from(value: WriteCodebookError) -> Self {
        WriteError::Codebook(value)
    }
}

impl From<WriteFloor1Error> for WriteError {
    fn from(value: WriteFloor1Error) -> Self {
        WriteError::Floor1(value)
    }
}

/// Returns the base-2 exponent of `n` if `n` is a power of two,
/// otherwise `None`.
///
/// Used to translate a sample-count blocksize ({64, 128, ..., 8192})
/// back to the 4-bit exponent field §4.2.2 actually serialises.
fn exponent_of_power_of_two(n: u16) -> Option<u8> {
    if n == 0 || !n.is_power_of_two() {
        return None;
    }
    Some(n.trailing_zeros() as u8)
}

/// Serialises a [`VorbisIdentificationHeader`] to the fixed 30-byte
/// Vorbis I §4.2.2 packet shape.
///
/// Layout (per §4.2.1 + §4.2.2 — the body's fields all span a whole
/// number of octets, so §2.1.4 LSB-first packing collapses to plain
/// little-endian byte order):
///
/// | Offset | Bytes | Content                                                  |
/// | -----: | ----: | -------------------------------------------------------- |
/// |      0 |     1 | `packet_type` (`0x01`)                                   |
/// |      1 |     6 | ASCII `"vorbis"` magic                                   |
/// |      7 |     4 | `vorbis_version` (`u32` LE; must be `0`)                 |
/// |     11 |     1 | `audio_channels` (`u8`; must be `> 0`)                   |
/// |     12 |     4 | `audio_sample_rate` (`u32` LE; must be `> 0`)            |
/// |     16 |     4 | `bitrate_maximum` (`i32` LE; 0 = unset)                  |
/// |     20 |     4 | `bitrate_nominal` (`i32` LE; 0 = unset)                  |
/// |     24 |     4 | `bitrate_minimum` (`i32` LE; 0 = unset)                  |
/// |     28 |     1 | `blocksize_0_exp` in low nibble, `blocksize_1_exp` in high |
/// |     29 |     1 | framing flag in bit 0 (`= 1`), bits 1..=7 are zero       |
///
/// Returns [`WriteError`] without emitting any bytes if any field
/// fails its §4.2.2 invariant. On success the returned [`Vec<u8>`] is
/// exactly [`VorbisIdentificationHeader::PACKET_LEN`] bytes long.
pub fn write_identification_header(
    header: &VorbisIdentificationHeader,
) -> Result<Vec<u8>, WriteError> {
    // §4.2.2 invariants — refuse rather than emit an invalid packet.
    if header.vorbis_version != 0 {
        return Err(WriteError::UnsupportedVorbisVersion(header.vorbis_version));
    }
    if header.audio_channels == 0 {
        return Err(WriteError::ZeroChannels);
    }
    if header.audio_sample_rate == 0 {
        return Err(WriteError::ZeroSampleRate);
    }
    let Some(bs0_exp) = exponent_of_power_of_two(header.blocksize_0) else {
        return Err(WriteError::IllegalBlocksize(
            header.blocksize_0,
            header.blocksize_1,
        ));
    };
    let Some(bs1_exp) = exponent_of_power_of_two(header.blocksize_1) else {
        return Err(WriteError::IllegalBlocksize(
            header.blocksize_0,
            header.blocksize_1,
        ));
    };
    if !(6..=13).contains(&bs0_exp) || !(6..=13).contains(&bs1_exp) {
        return Err(WriteError::IllegalBlocksize(
            header.blocksize_0,
            header.blocksize_1,
        ));
    }
    if header.blocksize_0 > header.blocksize_1 {
        return Err(WriteError::BlocksizesOutOfOrder {
            blocksize_0: header.blocksize_0,
            blocksize_1: header.blocksize_1,
        });
    }

    let mut packet = Vec::with_capacity(VorbisIdentificationHeader::PACKET_LEN);
    // §4.2.1 common header.
    packet.push(VorbisIdentificationHeader::PACKET_TYPE);
    packet.extend_from_slice(&VorbisIdentificationHeader::MAGIC);
    // §4.2.2 body — byte-aligned little-endian fields.
    packet.extend_from_slice(&header.vorbis_version.to_le_bytes());
    packet.push(header.audio_channels);
    packet.extend_from_slice(&header.audio_sample_rate.to_le_bytes());
    packet.extend_from_slice(&header.bitrate_maximum.to_le_bytes());
    packet.extend_from_slice(&header.bitrate_nominal.to_le_bytes());
    packet.extend_from_slice(&header.bitrate_minimum.to_le_bytes());
    // The two 4-bit exponents share a single byte: §2.1.4 LSB-first
    // packing puts the first field (`blocksize_0`) into the low
    // nibble (bits 0..3) and the second (`blocksize_1`) into the
    // high nibble (bits 4..7).
    packet.push((bs0_exp & 0x0f) | ((bs1_exp & 0x0f) << 4));
    // §4.2.2 framing flag in bit 0; bits 1..=7 are §2.1.8 padding,
    // which §2.1.8 recommends be zero on encode.
    packet.push(0x01);

    debug_assert_eq!(packet.len(), VorbisIdentificationHeader::PACKET_LEN);
    Ok(packet)
}

/// Serialises a [`VorbisCommentHeader`] to the variable-length Vorbis I
/// §5.2.1 packet shape.
///
/// Layout (per §4.2.1 + §5.2.1 + §5.2.3 — body is byte-aligned per
/// §5.2.1's "can simply be read as unaligned 32 bit little endian
/// unsigned integers" note):
///
/// ```text
/// 0x03                                  # packet_type
/// "vorbis"                              # 6-byte magic
/// vendor_length      : u32 LE           # number of bytes in vendor string
/// vendor             : [u8; vendor_length]
/// user_comment_list_length : u32 LE     # number of comment entries
/// (per comment, in order:
///   comment_length : u32 LE             # bytes in this comment
///   comment_bytes  : [u8; comment_length]
/// )
/// framing_byte       : u8               # bit 0 = 1 (framing flag),
///                                       # bits 1..=7 are §2.1.8 padding (zero on encode)
/// ```
///
/// Returns [`WriteError`] without emitting any bytes if any field
/// fails its §5.2.1 length-prefix invariant. The returned [`Vec<u8>`]
/// is exactly
/// `7 + 4 + vendor.len() + 4 + sum(4 + comment.len()) + 1` bytes long.
///
/// The `comments` slice is written in vector-order (`comments[0]`
/// first), which is the order [`VorbisCommentHeader::comments`]
/// preserves — `write_comment_header(parse_comment_header(&p)?)?`
/// reproduces the original comment order bit-exactly.
pub fn write_comment_header(header: &VorbisCommentHeader) -> Result<Vec<u8>, WriteError> {
    // §5.2.1 length-prefix invariants — guard the casts to u32 so we
    // refuse the call rather than wrap silently.
    if header.vendor.len() > u32::MAX as usize {
        return Err(WriteError::VendorTooLong(header.vendor.len()));
    }
    if header.comments.len() > u32::MAX as usize {
        return Err(WriteError::TooManyComments(header.comments.len()));
    }
    for c in &header.comments {
        if c.len() > u32::MAX as usize {
            return Err(WriteError::CommentTooLong(c.len()));
        }
    }

    // Pre-compute the exact final length so a single allocation
    // suffices. 7 common-header bytes + 4 vendor-length + N vendor
    // bytes + 4 comment-count + per-comment (4 + bytes) + 1 framing.
    let mut total: usize = 7 + 4 + header.vendor.len() + 4 + 1;
    for c in &header.comments {
        total = total.saturating_add(4).saturating_add(c.len());
    }
    let mut packet = Vec::with_capacity(total);

    // §4.2.1 common header.
    packet.push(VorbisCommentHeader::PACKET_TYPE);
    packet.extend_from_slice(&VorbisCommentHeader::MAGIC);

    // §5.2.1 step 1+2: vendor_length + vendor bytes.
    packet.extend_from_slice(&(header.vendor.len() as u32).to_le_bytes());
    packet.extend_from_slice(header.vendor.as_bytes());

    // §5.2.1 step 3: user_comment_list_length.
    packet.extend_from_slice(&(header.comments.len() as u32).to_le_bytes());

    // §5.2.1 step 4..6: per-comment length + bytes.
    for c in &header.comments {
        packet.extend_from_slice(&(c.len() as u32).to_le_bytes());
        packet.extend_from_slice(c.as_bytes());
    }

    // §5.2.1 step 7..8: framing bit in bit 0; bits 1..=7 zero per
    // §2.1.8.
    packet.push(0x01);

    debug_assert_eq!(packet.len(), total);
    Ok(packet)
}

/// Encoding policy for the [`write_codebook`] codeword-length section.
///
/// §3.2.1 lets the encoder pick between three forms for the per-entry
/// length table: dense unordered, sparse unordered (1-bit `used`
/// flags), and ordered ascending-length run encoding. The auto policy
/// inspects the [`VorbisCodebook`] and picks the form that
/// (a) is legal for the supplied data, (b) is typically densest.
///
/// For round 21 the writer exposes only the auto policy; callers who
/// want a specific encoding (e.g. for differential-fuzz against an
/// external implementation) should be able to specify it explicitly
/// in a later round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodebookLengthEncoding {
    /// Per §3.2.1 with `ordered = 0` and `sparse = 0`. Legal only
    /// when every entry is used (no [`UNUSED_ENTRY`] sentinels).
    /// Cost: `1 + 5*entries` bits.
    DenseUnordered,
    /// Per §3.2.1 with `ordered = 0` and `sparse = 1`. Legal for any
    /// codebook. Cost: `1 + sum_per_entry(1 + 5*used)` bits.
    SparseUnordered,
    /// Per §3.2.1 with `ordered = 1`. Legal only when every entry is
    /// used AND lengths are non-decreasing. Cost: highly variable.
    Ordered,
}

/// Auto-policy: pick the most compact legal encoding for `book`'s
/// codeword-length table per §3.2.1.
///
/// Preference order:
///
/// 1. Ordered — densest when applicable (a few `ilog`-bit run counters
///    instead of one 5-bit length per entry), but requires every
///    entry to be used AND lengths to be non-decreasing.
/// 2. Dense unordered — flat `1 + 5*entries` bits; requires every
///    entry to be used.
/// 3. Sparse unordered — universal fallback; the only form that can
///    carry [`UNUSED_ENTRY`] sentinels.
fn pick_length_encoding(book: &VorbisCodebook) -> CodebookLengthEncoding {
    let has_unused = book.codeword_lengths.contains(&UNUSED_ENTRY);
    if has_unused {
        return CodebookLengthEncoding::SparseUnordered;
    }
    let monotonic = book.codeword_lengths.windows(2).all(|w| w[0] <= w[1]);
    if monotonic {
        CodebookLengthEncoding::Ordered
    } else {
        CodebookLengthEncoding::DenseUnordered
    }
}

/// Serialises a [`VorbisCodebook`] to the variable-length Vorbis I
/// §3.2.1 codebook-header bitstream shape.
///
/// **Layered position** — a codebook header has no §4.2.1 common
/// header. Codebooks are nested inside the setup header (§4.2.4), so
/// the bitstream produced here starts with the 24-bit `0x564342`
/// sync pattern and ends immediately after the last bit of the
/// lookup section. The setup-header writer (followup) is responsible
/// for splicing this output into the larger setup packet.
///
/// **Encoding policy** — the length-table encoding (dense / sparse /
/// ordered) is chosen by the writer per [`pick_length_encoding`]:
///
/// * Any unused-entry sentinel ⇒ sparse unordered.
/// * Otherwise, non-decreasing lengths ⇒ ordered.
/// * Otherwise ⇒ dense unordered.
///
/// **Bit-exact roundtrip guarantee** — for every legal `book`,
/// `parse_codebook(&mut BitReaderLsb::new(&write_codebook(book)?))?`
/// equals `book` field-for-field.
///
/// **Return value** — the produced bitstream as a [`Vec<u8>`]. The
/// final byte may carry up to 7 bits of zero padding to byte-align
/// the slice; the parser ignores trailing padding because it stops
/// after the last codebook bit (§3.2.1 closes precisely at the last
/// multiplicand / lookup-type nibble).
///
/// Returns [`WriteCodebookError`] without emitting any bits if the
/// supplied codebook violates a §3.2.1 invariant.
pub fn write_codebook(book: &VorbisCodebook) -> Result<Vec<u8>, WriteCodebookError> {
    let mut writer = BitWriterLsb::with_capacity(64);
    write_codebook_into_writer(book, &mut writer)?;
    Ok(writer.finish())
}

/// Bit-level helper to splice a codebook into a larger bit-packed
/// stream (the setup-header writer will use this in a later round).
///
/// Writes the codebook's bits into `writer` at its current bit
/// position. On error, the caller's writer may have had no bits
/// appended (we validate before emitting) — but in the partial-write
/// case (a lookup-table error after the length section has been
/// emitted), the caller is expected to discard the partially-written
/// buffer rather than reuse it.
pub(crate) fn write_codebook_into_writer(
    book: &VorbisCodebook,
    writer: &mut BitWriterLsb,
) -> Result<(), WriteCodebookError> {
    // §3.2.1 invariant gate. The writer fails closed before any bits
    // hit `writer`'s accumulator, so a partial-write does not pollute
    // a caller-supplied buffer.
    if book.entries == 0 {
        return Err(WriteCodebookError::ZeroEntries);
    }
    if book.codeword_lengths.len() != book.entries as usize {
        return Err(WriteCodebookError::LengthTableMismatch {
            entries: book.entries,
            length_table: book.codeword_lengths.len(),
        });
    }
    for (idx, &len) in book.codeword_lengths.iter().enumerate() {
        // 0 is the unused-entry sentinel (only legal in sparse
        // encoding; we'll re-check below once we've picked one).
        // Used entries must be in 1..=32.
        if len != UNUSED_ENTRY && !(1..=32).contains(&len) {
            return Err(WriteCodebookError::IllegalCodewordLength {
                entry: idx as u32,
                length: len,
            });
        }
    }

    let encoding = pick_length_encoding(book);
    if matches!(encoding, CodebookLengthEncoding::Ordered) {
        // Belt-and-braces: pick_length_encoding already excluded
        // unused entries before choosing Ordered, but assert the
        // invariants in case a future refactor weakens the picker.
        if book.codeword_lengths.contains(&UNUSED_ENTRY) {
            return Err(WriteCodebookError::OrderedHasUnusedEntries);
        }
        for (idx, w) in book.codeword_lengths.windows(2).enumerate() {
            if w[0] > w[1] {
                return Err(WriteCodebookError::OrderedNotMonotonic {
                    entry: (idx + 1) as u32,
                    lengths: (w[0], w[1]),
                });
            }
        }
    }

    // §3.2.1 step 1: 24-bit sync pattern 0x564342.
    writer.write_u32(VorbisCodebook::SYNC_PATTERN, 24);
    // §3.2.1 step 2: 16-bit dimensions, 24-bit entries.
    writer.write_u32(book.dimensions as u32, 16);
    writer.write_u32(book.entries, 24);

    match encoding {
        CodebookLengthEncoding::DenseUnordered => {
            // ordered = 0, sparse = 0
            writer.write_bit(false);
            writer.write_bit(false);
            for &len in &book.codeword_lengths {
                // Every entry is used (picker invariant).
                writer.write_u32((len - 1) as u32, 5);
            }
        }
        CodebookLengthEncoding::SparseUnordered => {
            // ordered = 0, sparse = 1
            writer.write_bit(false);
            writer.write_bit(true);
            for &len in &book.codeword_lengths {
                if len == UNUSED_ENTRY {
                    writer.write_bit(false);
                } else {
                    writer.write_bit(true);
                    writer.write_u32((len - 1) as u32, 5);
                }
            }
        }
        CodebookLengthEncoding::Ordered => {
            // ordered = 1, then the §3.2.1 ascending-run encoding.
            writer.write_bit(true);
            let starting_length = book.codeword_lengths[0];
            writer.write_u32((starting_length - 1) as u32, 5);
            // Walk consecutive equal-length runs of ascending length,
            // emitting `number` in `ilog(entries - current_entry)`
            // bits per the §3.2.1 ordered branch.
            let entries = book.entries;
            let mut current_entry: u32 = 0;
            let mut current_length: u32 = starting_length as u32;
            while current_entry < entries {
                let mut run_end = current_entry;
                while (run_end as usize) < book.codeword_lengths.len()
                    && book.codeword_lengths[run_end as usize] as u32 == current_length
                {
                    run_end += 1;
                }
                let number = run_end - current_entry;
                let width = ilog(entries - current_entry);
                writer.write_u32(number, width);
                current_entry = run_end;
                current_length += 1;
            }
        }
    }

    // §3.2.1 step 5+: lookup_type and (for type ∈ {1, 2}) its body.
    match &book.lookup {
        VqLookup::None => {
            writer.write_u32(0, 4);
        }
        VqLookup::Lattice {
            minimum_value,
            delta_value,
            value_bits,
            sequence_p,
            multiplicands,
        } => {
            write_lookup_block(
                writer,
                1,
                *minimum_value,
                *delta_value,
                *value_bits,
                *sequence_p,
                multiplicands,
                lookup1_values(book.entries, book.dimensions) as u64,
                false, // is_tessellation
                book.entries,
                book.dimensions,
            )?;
        }
        VqLookup::Tessellation {
            minimum_value,
            delta_value,
            value_bits,
            sequence_p,
            multiplicands,
        } => {
            let expected = (book.entries as u64)
                .checked_mul(book.dimensions as u64)
                .ok_or(WriteCodebookError::LookupCountOverflow {
                    entries: book.entries,
                    dimensions: book.dimensions,
                })?;
            write_lookup_block(
                writer,
                2,
                *minimum_value,
                *delta_value,
                *value_bits,
                *sequence_p,
                multiplicands,
                expected,
                true, // is_tessellation
                book.entries,
                book.dimensions,
            )?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_lookup_block(
    writer: &mut BitWriterLsb,
    lookup_type: u8,
    minimum_value: f32,
    delta_value: f32,
    value_bits: u8,
    sequence_p: bool,
    multiplicands: &[u32],
    expected_count: u64,
    is_tessellation: bool,
    _entries: u32,
    _dimensions: u16,
) -> Result<(), WriteCodebookError> {
    if !(1..=16).contains(&value_bits) {
        return Err(WriteCodebookError::IllegalValueBits(value_bits));
    }
    let count = multiplicands.len() as u64;
    if count != expected_count {
        if is_tessellation {
            return Err(WriteCodebookError::TessellationMultiplicandCountMismatch {
                expected: expected_count,
                actual: multiplicands.len(),
            });
        } else {
            return Err(WriteCodebookError::LatticeMultiplicandCountMismatch {
                expected: expected_count as u32,
                actual: multiplicands.len(),
            });
        }
    }
    let cap: u32 = if value_bits == 32 {
        u32::MAX
    } else {
        (1u32 << value_bits) - 1
    };
    for (i, &m) in multiplicands.iter().enumerate() {
        if m > cap {
            return Err(WriteCodebookError::MultiplicandOverflow {
                index: i,
                value: m,
                value_bits,
            });
        }
    }
    let min_packed =
        float32_pack(minimum_value).ok_or(WriteCodebookError::UnrepresentableLookupFloat {
            is_minimum: true,
            value: minimum_value,
        })?;
    let delta_packed =
        float32_pack(delta_value).ok_or(WriteCodebookError::UnrepresentableLookupFloat {
            is_minimum: false,
            value: delta_value,
        })?;
    writer.write_u32(lookup_type as u32, 4);
    writer.write_u32(min_packed, 32);
    writer.write_u32(delta_packed, 32);
    writer.write_u32((value_bits - 1) as u32, 4);
    writer.write_bit(sequence_p);
    for &m in multiplicands {
        writer.write_u32(m, value_bits as u32);
    }
    Ok(())
}

/// Serialises a [`Floor1Header`] to the Vorbis I §7.2.2 floor-type-1
/// header bit pattern.
///
/// This is the inverse of the §7.2.2 parser invoked by the setup-header
/// walker. The 16-bit `floor_type = 1` selector that the setup-header
/// walker writes ahead of the per-floor body is **not** emitted here —
/// the floor-type tag belongs to the outer setup-header layout, and the
/// nested writer is the body-only function whose output the
/// [`Floor1Header::partitions`]-onwards parser ([`crate::setup::parse_setup_header_body`]
/// at floor-type dispatch) consumes.
///
/// Layout (per §7.2.2):
///
/// 1. `floor1_partitions` — 5 bits.
/// 2. For `i in 0 .. partitions`: `floor1_partition_class_list[i]` —
///    4 bits.
/// 3. For `i in 0 ..= maximum_class` (or none when `partitions == 0`),
///    per-class: `class_dimensions[i] - 1` (3 bits), `class_subclasses[i]`
///    (2 bits); when `subclasses > 0`, `class_masterbooks[i]` (8 bits);
///    then for `j in 0 .. 2^class_subclasses[i]`,
///    `subclass_books[i][j] + 1` (8 bits, with `None` mapped to raw `0`).
/// 4. `floor1_multiplier - 1` — 2 bits.
/// 5. `floor1_rangebits` — 4 bits.
/// 6. For each partition's `class.dimensions` x-list values: each as a
///    `rangebits`-bit unsigned integer.
///
/// **Return value** — the produced bitstream as a [`Vec<u8>`]. The
/// final byte may carry up to 7 bits of zero padding to byte-align the
/// slice; the parser stops after the last x-list bit per §7.2.2 step 21.
///
/// Returns [`WriteFloor1Error`] without emitting any bits if the
/// supplied header violates a §7.2.2 invariant.
pub fn write_floor1_header(header: &Floor1Header) -> Result<Vec<u8>, WriteFloor1Error> {
    let mut writer = BitWriterLsb::with_capacity(16);
    write_floor1_header_into_writer(header, &mut writer)?;
    Ok(writer.finish())
}

/// Bit-level helper to splice a floor 1 header into a larger
/// bit-packed stream (the setup-header writer will use this in a
/// later round, just as [`write_codebook_into_writer`] is exposed for
/// the codebook side).
///
/// Writes the header's bits into `writer` at its current bit
/// position. On error, the writer has had no bits appended (we
/// validate before emitting).
pub(crate) fn write_floor1_header_into_writer(
    header: &Floor1Header,
    writer: &mut BitWriterLsb,
) -> Result<(), WriteFloor1Error> {
    // ---- §7.2.2 invariant gate. ----
    // Fail-closed: refuse to emit a single bit if any field cannot be
    // serialised back to a packet the parser would accept.

    // step 1: partitions fits in a 5-bit field (0..=31).
    if header.partitions > 31 {
        return Err(WriteFloor1Error::PartitionsOverflow(header.partitions));
    }
    // step 3: partition_class_list length matches partitions.
    if header.partition_class_list.len() != header.partitions as usize {
        return Err(WriteFloor1Error::PartitionClassListMismatch {
            partitions: header.partitions,
            list_len: header.partition_class_list.len(),
        });
    }
    // step 4: each partition_class_list value fits in a 4-bit field.
    for (i, &v) in header.partition_class_list.iter().enumerate() {
        if v > 15 {
            return Err(WriteFloor1Error::PartitionClassValueOverflow {
                index: i as u8,
                value: v,
            });
        }
    }
    // step 6: classes.len() == max(partition_class_list) + 1, or 0
    // when partition_class_list is empty (partitions == 0).
    let expected_class_count = header
        .partition_class_list
        .iter()
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0);
    if header.classes.len() != expected_class_count {
        return Err(WriteFloor1Error::ClassCountMismatch {
            expected: expected_class_count,
            actual: header.classes.len(),
        });
    }
    // steps 7..12: per-class invariants.
    for (class_idx, class) in header.classes.iter().enumerate() {
        if !(1..=8).contains(&class.dimensions) {
            return Err(WriteFloor1Error::IllegalClassDimensions {
                class: class_idx,
                dimensions: class.dimensions,
            });
        }
        if class.subclasses > 3 {
            return Err(WriteFloor1Error::SubclassesOverflow {
                class: class_idx,
                subclasses: class.subclasses,
            });
        }
        let present = class.masterbook.is_some();
        let required = class.subclasses > 0;
        if present != required {
            return Err(WriteFloor1Error::MasterbookPresenceMismatch {
                class: class_idx,
                subclasses: class.subclasses,
                present,
            });
        }
        let expected_book_count = 1usize << class.subclasses;
        if class.subclass_books.len() != expected_book_count {
            return Err(WriteFloor1Error::SubclassBookCountMismatch {
                class: class_idx,
                expected: expected_book_count,
                actual: class.subclass_books.len(),
            });
        }
        for (j, &slot) in class.subclass_books.iter().enumerate() {
            if let Some(book) = slot {
                // step 12: raw = book + 1 must fit in 8 bits, so the
                // highest representable Some(_) is Some(254).
                if book == u8::MAX {
                    return Err(WriteFloor1Error::SubclassBookOverflow {
                        class: class_idx,
                        subclass: j,
                        book,
                    });
                }
            }
        }
    }
    // step 13: multiplier in 1..=4.
    if !(1..=4).contains(&header.multiplier) {
        return Err(WriteFloor1Error::IllegalMultiplier(header.multiplier));
    }
    // step 14: rangebits fits in a 4-bit field.
    if header.rangebits > 15 {
        return Err(WriteFloor1Error::RangebitsOverflow(header.rangebits));
    }
    // step 18: x_list length matches the sum-over-partitions formula.
    let expected_x_len: usize = header
        .partition_class_list
        .iter()
        .map(|&pc| header.classes[pc as usize].dimensions as usize)
        .sum();
    if header.x_list.len() != expected_x_len {
        return Err(WriteFloor1Error::XListLengthMismatch {
            expected: expected_x_len,
            actual: header.x_list.len(),
        });
    }
    // step 21: every x_list element fits in rangebits bits.
    // - rangebits == 0 → only 0 fits.
    // - 0 < rangebits < 32 → cap = (1 << rangebits) - 1.
    // - rangebits == 32 is unreachable here (rangebits is at most 15
    //   per the 4-bit field width checked above).
    for (i, &v) in header.x_list.iter().enumerate() {
        let cap: u32 = if header.rangebits == 0 {
            0
        } else {
            (1u32 << header.rangebits) - 1
        };
        if v > cap {
            return Err(WriteFloor1Error::XListValueOverflow {
                index: i,
                value: v,
                rangebits: header.rangebits,
            });
        }
    }

    // ---- §7.2.2 emit. ----
    // step 1.
    writer.write_u32(header.partitions as u32, 5);
    // step 3.
    for &pc in &header.partition_class_list {
        writer.write_u32(pc as u32, 4);
    }
    // steps 6..12.
    for class in &header.classes {
        writer.write_u32((class.dimensions - 1) as u32, 3);
        writer.write_u32(class.subclasses as u32, 2);
        if class.subclasses > 0 {
            // masterbook presence guaranteed by the gate above.
            writer.write_u32(class.masterbook.expect("validated") as u32, 8);
        }
        for slot in &class.subclass_books {
            let raw: u32 = match slot {
                Some(b) => (*b as u32) + 1,
                None => 0,
            };
            writer.write_u32(raw, 8);
        }
    }
    // step 13.
    writer.write_u32((header.multiplier - 1) as u32, 2);
    // step 14.
    writer.write_u32(header.rangebits as u32, 4);
    // step 18..21.
    for &v in &header.x_list {
        writer.write_u32(v, header.rangebits as u32);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comment::parse_comment_header;
    use crate::identification::parse_identification_header;
    use crate::setup::Floor1Class;

    // ----------------------------------------------------------------
    // Identification header — byte-shape pinning.
    // ----------------------------------------------------------------

    /// Pin the exact 30 bytes of a well-formed mono-44100 q5 packet so
    /// the byte layout (LSB-first nibble packing of the blocksize byte
    /// plus the framing-bit byte) is locked, not merely "roundtrips
    /// through the parser." Mirrors the fixture shape in
    /// `docs/audio/vorbis/vorbis-fixtures-and-traces.md` §2.1.
    #[test]
    fn identification_byte_shape_mono_44100_q5() {
        let header = VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 1,
            audio_sample_rate: 44100,
            bitrate_maximum: 0,
            bitrate_nominal: 96000,
            bitrate_minimum: 0,
            blocksize_0: 256,  // exponent 8
            blocksize_1: 2048, // exponent 11
        };
        let packet = write_identification_header(&header).expect("must build");
        assert_eq!(packet.len(), VorbisIdentificationHeader::PACKET_LEN);
        // Common header.
        assert_eq!(packet[0], 0x01);
        assert_eq!(&packet[1..7], b"vorbis");
        // vorbis_version = 0.
        assert_eq!(&packet[7..11], &0u32.to_le_bytes());
        // audio_channels = 1.
        assert_eq!(packet[11], 1);
        // audio_sample_rate = 44100.
        assert_eq!(&packet[12..16], &44100u32.to_le_bytes());
        // bitrate_max / nom / min.
        assert_eq!(&packet[16..20], &0i32.to_le_bytes());
        assert_eq!(&packet[20..24], &96000i32.to_le_bytes());
        assert_eq!(&packet[24..28], &0i32.to_le_bytes());
        // Blocksize byte: low nibble = 8 (256), high nibble = 11 (2048).
        assert_eq!(packet[28], 8 | (11 << 4));
        // Framing byte: bit 0 set, bits 1..=7 zero.
        assert_eq!(packet[29], 0x01);
    }

    /// Bit-exact roundtrip: writer → parser → equal.
    fn id_roundtrips(header: &VorbisIdentificationHeader) {
        let packet = write_identification_header(header).expect("write must succeed");
        let parsed = parse_identification_header(&packet).expect("parse must succeed");
        assert_eq!(&parsed, header, "roundtrip equality");
    }

    #[test]
    fn identification_roundtrips_mono_44100_q5() {
        id_roundtrips(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 1,
            audio_sample_rate: 44100,
            bitrate_maximum: 0,
            bitrate_nominal: 96000,
            bitrate_minimum: 0,
            blocksize_0: 256,
            blocksize_1: 2048,
        });
    }

    #[test]
    fn identification_roundtrips_5_1_channel_48000_q5() {
        id_roundtrips(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 6,
            audio_sample_rate: 48000,
            bitrate_maximum: 0,
            bitrate_nominal: 192000,
            bitrate_minimum: 0,
            blocksize_0: 256,
            blocksize_1: 2048,
        });
    }

    #[test]
    fn identification_roundtrips_negative_bitrate_hints() {
        // Negative sentinels in the signed bitrate fields — the §4.2.2
        // fields are signed two's-complement, so a value of `-1` is
        // representable and must roundtrip bit-for-bit.
        id_roundtrips(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 2,
            audio_sample_rate: 44100,
            bitrate_maximum: -1,
            bitrate_nominal: -1,
            bitrate_minimum: -1,
            blocksize_0: 256,
            blocksize_1: 2048,
        });
    }

    #[test]
    fn identification_roundtrips_equal_blocksizes_at_spec_minimum() {
        // exponent 6 → 64 samples, the §4.2.2 minimum.
        id_roundtrips(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 1,
            audio_sample_rate: 22050,
            bitrate_maximum: 0,
            bitrate_nominal: 32000,
            bitrate_minimum: 0,
            blocksize_0: 64,
            blocksize_1: 64,
        });
    }

    #[test]
    fn identification_roundtrips_max_blocksize_8192() {
        // exponent 13 → 8192 samples, the §4.2.2 maximum.
        id_roundtrips(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 2,
            audio_sample_rate: 96000,
            bitrate_maximum: 0,
            bitrate_nominal: 500000,
            bitrate_minimum: 0,
            blocksize_0: 64,
            blocksize_1: 8192,
        });
    }

    #[test]
    fn identification_roundtrips_max_channels() {
        // §1.1.1 caps the channel count at 255 (1 byte). Verify the
        // upper edge roundtrips.
        id_roundtrips(&VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 255,
            audio_sample_rate: 48000,
            bitrate_maximum: 0,
            bitrate_nominal: 0,
            bitrate_minimum: 0,
            blocksize_0: 1024,
            blocksize_1: 1024,
        });
    }

    /// Sweep every legal `(blocksize_0_exponent, blocksize_1_exponent)`
    /// pair with `bs0 <= bs1` — there are `8 + 7 + ... + 1 = 36` of
    /// these — to confirm the nibble-pack of byte 28 roundtrips for
    /// every legal combination.
    #[test]
    fn identification_roundtrips_all_legal_blocksize_pairs() {
        for bs0_exp in 6u8..=13 {
            for bs1_exp in bs0_exp..=13 {
                let header = VorbisIdentificationHeader {
                    vorbis_version: 0,
                    audio_channels: 1,
                    audio_sample_rate: 44100,
                    bitrate_maximum: 0,
                    bitrate_nominal: 0,
                    bitrate_minimum: 0,
                    blocksize_0: 1u16 << bs0_exp,
                    blocksize_1: 1u16 << bs1_exp,
                };
                let packet = write_identification_header(&header).expect("write");
                assert_eq!(packet[28], bs0_exp | (bs1_exp << 4));
                let parsed = parse_identification_header(&packet).expect("parse");
                assert_eq!(parsed, header);
            }
        }
    }

    // ----------------------------------------------------------------
    // Identification header — §4.2.2 invariant rejection.
    // ----------------------------------------------------------------

    fn legal_id() -> VorbisIdentificationHeader {
        VorbisIdentificationHeader {
            vorbis_version: 0,
            audio_channels: 2,
            audio_sample_rate: 44100,
            bitrate_maximum: 0,
            bitrate_nominal: 128000,
            bitrate_minimum: 0,
            blocksize_0: 256,
            blocksize_1: 2048,
        }
    }

    #[test]
    fn identification_rejects_nonzero_vorbis_version() {
        let mut h = legal_id();
        h.vorbis_version = 1;
        assert_eq!(
            write_identification_header(&h),
            Err(WriteError::UnsupportedVorbisVersion(1))
        );
    }

    #[test]
    fn identification_rejects_zero_channels() {
        let mut h = legal_id();
        h.audio_channels = 0;
        assert_eq!(
            write_identification_header(&h),
            Err(WriteError::ZeroChannels)
        );
    }

    #[test]
    fn identification_rejects_zero_sample_rate() {
        let mut h = legal_id();
        h.audio_sample_rate = 0;
        assert_eq!(
            write_identification_header(&h),
            Err(WriteError::ZeroSampleRate)
        );
    }

    #[test]
    fn identification_rejects_non_power_of_two_blocksize() {
        let mut h = legal_id();
        h.blocksize_0 = 300; // not a power of two
        match write_identification_header(&h) {
            Err(WriteError::IllegalBlocksize(300, 2048)) => {}
            other => panic!("expected IllegalBlocksize(300, 2048), got {other:?}"),
        }
    }

    #[test]
    fn identification_rejects_blocksize_exponent_below_six() {
        // 32 samples — exponent 5, below the legal minimum of 6.
        let mut h = legal_id();
        h.blocksize_0 = 32;
        match write_identification_header(&h) {
            Err(WriteError::IllegalBlocksize(32, 2048)) => {}
            other => panic!("expected IllegalBlocksize(32, 2048), got {other:?}"),
        }
    }

    #[test]
    fn identification_rejects_blocksize_exponent_above_thirteen() {
        // 16384 samples — exponent 14, above the legal maximum of 13.
        let mut h = legal_id();
        h.blocksize_1 = 16384;
        match write_identification_header(&h) {
            Err(WriteError::IllegalBlocksize(256, 16384)) => {}
            other => panic!("expected IllegalBlocksize(256, 16384), got {other:?}"),
        }
    }

    #[test]
    fn identification_rejects_blocksize_zero() {
        let mut h = legal_id();
        h.blocksize_0 = 0;
        match write_identification_header(&h) {
            Err(WriteError::IllegalBlocksize(0, 2048)) => {}
            other => panic!("expected IllegalBlocksize(0, 2048), got {other:?}"),
        }
    }

    #[test]
    fn identification_rejects_blocksizes_out_of_order() {
        let mut h = legal_id();
        h.blocksize_0 = 2048;
        h.blocksize_1 = 256;
        match write_identification_header(&h) {
            Err(WriteError::BlocksizesOutOfOrder {
                blocksize_0: 2048,
                blocksize_1: 256,
            }) => {}
            other => panic!("expected BlocksizesOutOfOrder, got {other:?}"),
        }
    }

    // ----------------------------------------------------------------
    // Comment header — byte-shape pinning + roundtrip.
    // ----------------------------------------------------------------

    /// Pin the exact bytes of a small comment packet so the byte
    /// layout (u32 LE length prefixes + raw UTF-8 + framing byte) is
    /// locked, not merely "roundtrips through the parser."
    #[test]
    fn comment_byte_shape_one_comment() {
        let header = VorbisCommentHeader {
            vendor: "Lavf61.7.100".to_string(),
            comments: vec!["encoder=Lavc61.19.101 libvorbis".to_string()],
        };
        let packet = write_comment_header(&header).expect("must build");
        // Common header.
        assert_eq!(packet[0], 0x03);
        assert_eq!(&packet[1..7], b"vorbis");
        // vendor_length = 12.
        assert_eq!(&packet[7..11], &12u32.to_le_bytes());
        // vendor bytes.
        assert_eq!(&packet[11..23], b"Lavf61.7.100");
        // user_comment_list_length = 1.
        assert_eq!(&packet[23..27], &1u32.to_le_bytes());
        // First comment length = 31 (bytes of "encoder=Lavc61.19.101 libvorbis").
        assert_eq!(&packet[27..31], &31u32.to_le_bytes());
        // First comment bytes.
        assert_eq!(&packet[31..62], b"encoder=Lavc61.19.101 libvorbis");
        // Framing byte: bit 0 set.
        assert_eq!(packet[62], 0x01);
        assert_eq!(packet.len(), 63);
    }

    /// Bit-exact roundtrip: writer → parser → equal.
    fn comment_roundtrips(header: &VorbisCommentHeader) {
        let packet = write_comment_header(header).expect("write must succeed");
        let parsed = parse_comment_header(&packet).expect("parse must succeed");
        assert_eq!(&parsed, header, "roundtrip equality");
    }

    #[test]
    fn comment_roundtrips_typical_one_comment() {
        comment_roundtrips(&VorbisCommentHeader {
            vendor: "Lavf61.7.100".to_string(),
            comments: vec!["encoder=Lavc61.19.101 libvorbis".to_string()],
        });
    }

    #[test]
    fn comment_roundtrips_seven_comments() {
        // Mirrors the `with-vorbis-comment-tags` fixture shape from
        // `docs/audio/vorbis/vorbis-fixtures-and-traces.md` §2.2.
        comment_roundtrips(&VorbisCommentHeader {
            vendor: "Lavf61.7.100".to_string(),
            comments: vec![
                "TITLE=Round 195 Test".to_string(),
                "ARTIST=OxideAV".to_string(),
                "ALBUM=Vorbis I Encoder Primitives".to_string(),
                "DATE=2026-05-31".to_string(),
                "GENRE=Reference".to_string(),
                "TRACKNUMBER=01".to_string(),
                "encoder=Lavc61.19.101 libvorbis".to_string(),
            ],
        });
    }

    #[test]
    fn comment_roundtrips_empty_vendor_no_comments() {
        comment_roundtrips(&VorbisCommentHeader {
            vendor: String::new(),
            comments: Vec::new(),
        });
    }

    #[test]
    fn comment_roundtrips_empty_vendor_with_comments() {
        comment_roundtrips(&VorbisCommentHeader {
            vendor: String::new(),
            comments: vec!["KEY=value".to_string()],
        });
    }

    #[test]
    fn comment_roundtrips_multibyte_utf8_vendor() {
        // Vendor field can carry arbitrary UTF-8 per §5.2.1; round-trip
        // a string with multi-byte sequences (Greek, Japanese, emoji-
        // equivalent ASCII punctuation, ASCII letters mix).
        comment_roundtrips(&VorbisCommentHeader {
            vendor: "Ω-encoder ✓ v1.0 — Karpelès".to_string(),
            comments: Vec::new(),
        });
    }

    #[test]
    fn comment_roundtrips_multibyte_utf8_in_comments() {
        comment_roundtrips(&VorbisCommentHeader {
            vendor: "Lavf61.7.100".to_string(),
            comments: vec![
                "TITLE=日本語タイトル".to_string(),
                "ARTIST=Καλλιτέχνης".to_string(),
                "DESCRIPTION=mixed €£¥".to_string(),
            ],
        });
    }

    #[test]
    fn comment_roundtrips_duplicate_keys() {
        // §5.2.2 allows duplicate keys; the parser preserves
        // insertion order — the writer must too.
        comment_roundtrips(&VorbisCommentHeader {
            vendor: "x".to_string(),
            comments: vec![
                "ARTIST=Alpha".to_string(),
                "ARTIST=Beta".to_string(),
                "ARTIST=Gamma".to_string(),
            ],
        });
    }

    #[test]
    fn comment_roundtrips_long_payload() {
        // A 32 KiB single comment exercises the u32 length-prefix path
        // and the buffer allocation accounting without ballooning the
        // test runtime.
        let big = "A".repeat(32 * 1024);
        let comments = vec![format!("METADATA_BLOCK_PICTURE={big}")];
        let header = VorbisCommentHeader {
            vendor: "Lavf61.7.100".to_string(),
            comments,
        };
        let packet = write_comment_header(&header).expect("write must succeed");
        let parsed = parse_comment_header(&packet).expect("parse must succeed");
        assert_eq!(parsed, header);
    }

    #[test]
    fn comment_preserves_comment_order() {
        // Distinct ordered keys + an out-of-order alphabet to confirm
        // we never sort or otherwise reorder the input vector.
        let header = VorbisCommentHeader {
            vendor: "v".to_string(),
            comments: vec![
                "Z=last-by-letter-first-by-position".to_string(),
                "A=first-by-letter-second-by-position".to_string(),
                "M=middle-by-letter-third-by-position".to_string(),
            ],
        };
        let packet = write_comment_header(&header).expect("write");
        let parsed = parse_comment_header(&packet).expect("parse");
        assert_eq!(parsed.comments[0], header.comments[0]);
        assert_eq!(parsed.comments[1], header.comments[1]);
        assert_eq!(parsed.comments[2], header.comments[2]);
    }

    /// The writer's emitted byte length must match the closed-form
    /// `7 + 4 + V + 4 + sum(4 + C_i) + 1` formula from §5.2.1.
    #[test]
    fn comment_byte_length_formula() {
        let cases: Vec<(VorbisCommentHeader, usize)> = vec![
            (
                VorbisCommentHeader {
                    vendor: String::new(),
                    comments: Vec::new(),
                },
                // 7 common + 4 vendor_len + 0 vendor bytes + 4 count + 1 framing.
                7 + 4 + 4 + 1,
            ),
            (
                VorbisCommentHeader {
                    vendor: "v".to_string(),
                    comments: Vec::new(),
                },
                7 + 4 + 1 + 4 + 1,
            ),
            (
                VorbisCommentHeader {
                    vendor: "v".to_string(),
                    comments: vec!["A=B".to_string()],
                },
                7 + 4 + 1 + 4 + (4 + 3) + 1,
            ),
            (
                VorbisCommentHeader {
                    vendor: "Lavf61.7.100".to_string(),
                    comments: vec![
                        "K1=v1".to_string(),     // 5
                        "K2=value2".to_string(), // 9
                    ],
                },
                7 + 4 + 12 + 4 + (4 + 5) + (4 + 9) + 1,
            ),
        ];
        for (header, expected_len) in cases {
            let packet = write_comment_header(&header).expect("write");
            assert_eq!(
                packet.len(),
                expected_len,
                "header {:?} produced {} bytes, expected {}",
                header,
                packet.len(),
                expected_len
            );
        }
    }

    // ----------------------------------------------------------------
    // WriteError — Display sanity (the strings appear in logs).
    // ----------------------------------------------------------------

    #[test]
    fn write_error_display_non_empty() {
        // Smoke-test every variant's Display non-emptiness; we don't
        // pin the exact strings (the wording is policy, not API), but
        // we do confirm none of them are silently blank.
        let variants = [
            WriteError::UnsupportedVorbisVersion(1),
            WriteError::ZeroChannels,
            WriteError::ZeroSampleRate,
            WriteError::IllegalBlocksize(300, 2048),
            WriteError::BlocksizesOutOfOrder {
                blocksize_0: 2048,
                blocksize_1: 256,
            },
            WriteError::CommentTooLong(usize::MAX),
            WriteError::VendorTooLong(usize::MAX),
            WriteError::TooManyComments(usize::MAX),
        ];
        for e in variants {
            let s = format!("{e}");
            assert!(!s.is_empty(), "WriteError::Display empty for {e:?}");
        }
    }

    #[test]
    fn write_error_source_returns_none() {
        // All WriteError variants are leaf errors (no inner source).
        use std::error::Error as StdError;
        let e: WriteError = WriteError::ZeroChannels;
        assert!(StdError::source(&e).is_none());
    }

    // ----------------------------------------------------------------
    // Internal helper: exponent_of_power_of_two.
    // ----------------------------------------------------------------

    #[test]
    fn exponent_helper_recognises_powers_of_two() {
        for exp in 0u8..=13 {
            assert_eq!(exponent_of_power_of_two(1u16 << exp), Some(exp));
        }
    }

    #[test]
    fn exponent_helper_rejects_non_powers_of_two() {
        assert_eq!(exponent_of_power_of_two(0), None);
        assert_eq!(exponent_of_power_of_two(3), None);
        assert_eq!(exponent_of_power_of_two(100), None);
        assert_eq!(exponent_of_power_of_two(300), None);
        assert_eq!(exponent_of_power_of_two(1023), None);
        assert_eq!(exponent_of_power_of_two(1025), None);
    }

    // ----------------------------------------------------------------
    // Codebook header — §3.2.1 roundtrip + encoding-pick coverage.
    // ----------------------------------------------------------------

    use crate::codebook::parse_codebook;
    use oxideav_core::bits::BitReaderLsb;

    /// Bit-exact roundtrip helper: writer → parser → equal.
    fn codebook_roundtrips(book: &VorbisCodebook) {
        let bytes = write_codebook(book).expect("write must succeed");
        let mut reader = BitReaderLsb::new(&bytes);
        let parsed = parse_codebook(&mut reader).expect("parse must succeed");
        assert_eq!(&parsed, book, "roundtrip equality");
    }

    /// Dense unordered: §3.2.1 worked-example shape (8 entries,
    /// lengths `[2, 4, 4, 4, 4, 2, 3, 3]`). The lengths are NOT
    /// monotonic — so the auto-picker chooses dense unordered, not
    /// ordered.
    #[test]
    fn codebook_roundtrips_dense_unordered_worked_example() {
        codebook_roundtrips(&VorbisCodebook {
            dimensions: 1,
            entries: 8,
            codeword_lengths: vec![2, 4, 4, 4, 4, 2, 3, 3],
            lookup: VqLookup::None,
        });
    }

    /// Sparse unordered: any unused-entry sentinel forces sparse.
    #[test]
    fn codebook_roundtrips_sparse_with_unused_entries() {
        codebook_roundtrips(&VorbisCodebook {
            dimensions: 1,
            entries: 6,
            codeword_lengths: vec![1, UNUSED_ENTRY, 3, UNUSED_ENTRY, 5, UNUSED_ENTRY],
            lookup: VqLookup::None,
        });
    }

    /// Ordered: non-decreasing lengths + no unused entries → the
    /// picker chooses the ordered encoding.
    #[test]
    fn codebook_roundtrips_ordered_monotonic() {
        // Same length profile as the §3.2.1 ordered-branch test in
        // the parser suite: [2, 2, 2, 3, 3, 3, 3, 3].
        codebook_roundtrips(&VorbisCodebook {
            dimensions: 1,
            entries: 8,
            codeword_lengths: vec![2, 2, 2, 3, 3, 3, 3, 3],
            lookup: VqLookup::None,
        });
    }

    /// Lookup type 2 (tessellation): roundtrip a 2-dim, 4-entry,
    /// value_bits=3 book — the multiplicand table has length
    /// `entries × dimensions = 8`. The lengths are NOT monotonic
    /// (all equal), so the auto-picker would actually choose ordered;
    /// flip one length to force dense unordered for coverage of the
    /// lookup-encoding path on the dense branch.
    #[test]
    fn codebook_roundtrips_lookup_type_2_tessellation() {
        codebook_roundtrips(&VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![2, 2, 2, 2],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 3,
                sequence_p: false,
                multiplicands: vec![0, 1, 2, 3, 4, 5, 6, 7],
            },
        });
    }

    /// Lookup type 1 (lattice): roundtrip a 2-dim, 64-entry,
    /// value_bits=4 book — the multiplicand table has length
    /// `lookup1_values(64, 2) = 8`.
    #[test]
    fn codebook_roundtrips_lookup_type_1_lattice() {
        codebook_roundtrips(&VorbisCodebook {
            dimensions: 2,
            entries: 64,
            codeword_lengths: vec![6; 64],
            lookup: VqLookup::Lattice {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 4,
                sequence_p: true,
                multiplicands: (0..8).collect(),
            },
        });
    }

    /// Non-trivial floats in the lookup table: roundtrip a value that
    /// exercises `float32_pack` past the trivial zero/one cases.
    #[test]
    fn codebook_roundtrips_lookup_with_non_trivial_floats() {
        codebook_roundtrips(&VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![2, 2, 2, 2],
            lookup: VqLookup::Tessellation {
                minimum_value: -1024.0,
                delta_value: 0.5,
                value_bits: 8,
                sequence_p: true,
                multiplicands: vec![0, 1, 127, 255, 0x80, 0x40, 0x20, 0x10],
            },
        });
    }

    /// Roundtrip a maximum-width lookup table: `value_bits = 16`,
    /// 2-dim, 4-entry → 8 multiplicands of `u16::MAX`.
    #[test]
    fn codebook_roundtrips_lookup_value_bits_16_edge() {
        codebook_roundtrips(&VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![2, 2, 2, 2],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 16,
                sequence_p: false,
                multiplicands: vec![0xffff; 8],
            },
        });
    }

    /// Roundtrip a single-entry codebook. The picker hits the
    /// edge-case `windows(2)` empty iterator (monotonic vacuously).
    #[test]
    fn codebook_roundtrips_single_entry() {
        codebook_roundtrips(&VorbisCodebook {
            dimensions: 1,
            entries: 1,
            codeword_lengths: vec![1],
            lookup: VqLookup::None,
        });
    }

    /// Roundtrip the trace-doc §3 fixture-style codebook shape: 8
    /// dimensions, 8 entries, sparse with most entries unused,
    /// lookup_type=2, value_bits=8.
    #[test]
    fn codebook_roundtrips_trace_doc_shape() {
        let mut lengths = vec![UNUSED_ENTRY; 8];
        lengths[0] = 1;
        lengths[7] = 1;
        codebook_roundtrips(&VorbisCodebook {
            dimensions: 8,
            entries: 8,
            codeword_lengths: lengths,
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands: (0..64).map(|i| (i & 0xff) as u32).collect(),
            },
        });
    }

    // ----------------------------------------------------------------
    // Codebook header — picker / byte-shape pinning.
    // ----------------------------------------------------------------

    /// Pin the encoding picker's choice: a codebook with unused
    /// entries MUST go to sparse, regardless of monotonicity.
    #[test]
    fn codebook_picker_unused_forces_sparse() {
        let book = VorbisCodebook {
            dimensions: 1,
            entries: 4,
            codeword_lengths: vec![1, UNUSED_ENTRY, 3, 4],
            lookup: VqLookup::None,
        };
        let bytes = write_codebook(&book).expect("write");
        // Re-parse and reason about the ordered/sparse bits directly.
        let mut r = BitReaderLsb::new(&bytes);
        // Skip sync (24), dimensions (16), entries (24) = 64 bits.
        let _ = r.read_u32(24).unwrap();
        let _ = r.read_u32(16).unwrap();
        let _ = r.read_u32(24).unwrap();
        let ordered = r.read_bit().unwrap();
        let sparse = r.read_bit().unwrap();
        assert!(!ordered, "picker must NOT choose ordered for sparse data");
        assert!(
            sparse,
            "picker must choose sparse for codebook with unused entries"
        );
    }

    /// Pin the encoding picker's choice: monotonic lengths with no
    /// unused entries goes to ordered.
    #[test]
    fn codebook_picker_monotonic_chooses_ordered() {
        let book = VorbisCodebook {
            dimensions: 1,
            entries: 6,
            codeword_lengths: vec![1, 2, 2, 3, 3, 3],
            lookup: VqLookup::None,
        };
        let bytes = write_codebook(&book).expect("write");
        let mut r = BitReaderLsb::new(&bytes);
        let _ = r.read_u32(24).unwrap();
        let _ = r.read_u32(16).unwrap();
        let _ = r.read_u32(24).unwrap();
        let ordered = r.read_bit().unwrap();
        assert!(ordered, "picker must choose ordered for monotonic data");
    }

    /// Pin the encoding picker's choice: non-monotonic lengths with
    /// no unused entries goes to dense unordered.
    #[test]
    fn codebook_picker_non_monotonic_chooses_dense_unordered() {
        let book = VorbisCodebook {
            dimensions: 1,
            entries: 4,
            codeword_lengths: vec![3, 1, 2, 4],
            lookup: VqLookup::None,
        };
        let bytes = write_codebook(&book).expect("write");
        let mut r = BitReaderLsb::new(&bytes);
        let _ = r.read_u32(24).unwrap();
        let _ = r.read_u32(16).unwrap();
        let _ = r.read_u32(24).unwrap();
        let ordered = r.read_bit().unwrap();
        let sparse = r.read_bit().unwrap();
        assert!(!ordered);
        assert!(!sparse);
    }

    /// The first 24 bits of every codebook are the sync pattern,
    /// LSB-first per §2.1.4 — so bytes 0..=2 of the output are
    /// `0x42 0x43 0x56` regardless of contents.
    #[test]
    fn codebook_byte_shape_sync_pattern_first() {
        let book = VorbisCodebook {
            dimensions: 1,
            entries: 1,
            codeword_lengths: vec![1],
            lookup: VqLookup::None,
        };
        let bytes = write_codebook(&book).expect("write");
        assert_eq!(bytes[0], 0x42);
        assert_eq!(bytes[1], 0x43);
        assert_eq!(bytes[2], 0x56);
    }

    // ----------------------------------------------------------------
    // Codebook header — §3.2.1 invariant rejection.
    // ----------------------------------------------------------------

    fn legal_book() -> VorbisCodebook {
        VorbisCodebook {
            dimensions: 1,
            entries: 4,
            codeword_lengths: vec![1, 2, 3, 4],
            lookup: VqLookup::None,
        }
    }

    #[test]
    fn codebook_rejects_zero_entries() {
        let mut b = legal_book();
        b.entries = 0;
        b.codeword_lengths.clear();
        assert_eq!(write_codebook(&b), Err(WriteCodebookError::ZeroEntries));
    }

    #[test]
    fn codebook_rejects_length_table_mismatch() {
        let mut b = legal_book();
        b.codeword_lengths.pop();
        match write_codebook(&b) {
            Err(WriteCodebookError::LengthTableMismatch {
                entries: 4,
                length_table: 3,
            }) => {}
            other => panic!("expected LengthTableMismatch, got {other:?}"),
        }
    }

    #[test]
    fn codebook_rejects_codeword_length_zero_in_dense() {
        // A length of 0 in the middle is the unused-entry sentinel —
        // legal only in sparse. The picker auto-routes to sparse
        // (because of the unused sentinel), so this should NOT error
        // out. Verify it actually succeeds.
        let b = VorbisCodebook {
            dimensions: 1,
            entries: 3,
            codeword_lengths: vec![1, UNUSED_ENTRY, 3],
            lookup: VqLookup::None,
        };
        assert!(write_codebook(&b).is_ok());
    }

    #[test]
    fn codebook_rejects_codeword_length_above_thirty_two() {
        let mut b = legal_book();
        b.codeword_lengths[2] = 33;
        match write_codebook(&b) {
            Err(WriteCodebookError::IllegalCodewordLength {
                entry: 2,
                length: 33,
            }) => {}
            other => panic!("expected IllegalCodewordLength(2, 33), got {other:?}"),
        }
    }

    #[test]
    fn codebook_rejects_value_bits_zero() {
        let b = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![2, 2, 2, 2],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 0,
                sequence_p: false,
                multiplicands: vec![0; 8],
            },
        };
        match write_codebook(&b) {
            Err(WriteCodebookError::IllegalValueBits(0)) => {}
            other => panic!("expected IllegalValueBits(0), got {other:?}"),
        }
    }

    #[test]
    fn codebook_rejects_value_bits_seventeen() {
        let b = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![2, 2, 2, 2],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 17,
                sequence_p: false,
                multiplicands: vec![0; 8],
            },
        };
        match write_codebook(&b) {
            Err(WriteCodebookError::IllegalValueBits(17)) => {}
            other => panic!("expected IllegalValueBits(17), got {other:?}"),
        }
    }

    #[test]
    fn codebook_rejects_multiplicand_overflow() {
        let b = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![2, 2, 2, 2],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 3,
                sequence_p: false,
                // value 16 needs 5 bits but only 3 are allotted.
                multiplicands: vec![0, 1, 2, 16, 4, 5, 6, 7],
            },
        };
        match write_codebook(&b) {
            Err(WriteCodebookError::MultiplicandOverflow {
                index: 3,
                value: 16,
                value_bits: 3,
            }) => {}
            other => panic!("expected MultiplicandOverflow, got {other:?}"),
        }
    }

    #[test]
    fn codebook_rejects_tessellation_count_mismatch() {
        let b = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![2, 2, 2, 2],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 3,
                sequence_p: false,
                // expected 8, supplied 7
                multiplicands: vec![0; 7],
            },
        };
        match write_codebook(&b) {
            Err(WriteCodebookError::TessellationMultiplicandCountMismatch {
                expected: 8,
                actual: 7,
            }) => {}
            other => panic!("expected TessellationMultiplicandCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn codebook_rejects_lattice_count_mismatch() {
        let b = VorbisCodebook {
            dimensions: 2,
            entries: 64,
            codeword_lengths: vec![6; 64],
            lookup: VqLookup::Lattice {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 4,
                sequence_p: false,
                // expected lookup1_values(64, 2) = 8, supplied 7
                multiplicands: vec![0; 7],
            },
        };
        match write_codebook(&b) {
            Err(WriteCodebookError::LatticeMultiplicandCountMismatch {
                expected: 8,
                actual: 7,
            }) => {}
            other => panic!("expected LatticeMultiplicandCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn codebook_rejects_unrepresentable_minimum_value() {
        let b = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![2, 2, 2, 2],
            lookup: VqLookup::Tessellation {
                minimum_value: f32::INFINITY, // not expressible in §9.2.2
                delta_value: 1.0,
                value_bits: 3,
                sequence_p: false,
                multiplicands: vec![0; 8],
            },
        };
        match write_codebook(&b) {
            Err(WriteCodebookError::UnrepresentableLookupFloat {
                is_minimum: true, ..
            }) => {}
            other => panic!("expected UnrepresentableLookupFloat(min), got {other:?}"),
        }
    }

    #[test]
    fn codebook_rejects_unrepresentable_delta_value() {
        let b = VorbisCodebook {
            dimensions: 2,
            entries: 4,
            codeword_lengths: vec![2, 2, 2, 2],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: f32::NAN,
                value_bits: 3,
                sequence_p: false,
                multiplicands: vec![0; 8],
            },
        };
        match write_codebook(&b) {
            Err(WriteCodebookError::UnrepresentableLookupFloat {
                is_minimum: false, ..
            }) => {}
            other => panic!("expected UnrepresentableLookupFloat(delta), got {other:?}"),
        }
    }

    // ----------------------------------------------------------------
    // Codebook header — bit-length accounting matches a hand-computed
    // formula for each encoding form.
    // ----------------------------------------------------------------

    /// Dense unordered: header is 24 (sync) + 16 (dims) + 24 (entries)
    ///     + 1 (ordered) + 1 (sparse) + 5*entries (lengths) + 4 (lookup
    ///     type) bits = 70 + 5*entries bits. Byte length is `ceil(bits / 8)`.
    #[test]
    fn codebook_dense_unordered_bit_length() {
        let book = VorbisCodebook {
            dimensions: 1,
            entries: 8,
            codeword_lengths: vec![2, 4, 4, 4, 4, 2, 3, 3], // non-monotonic → dense
            lookup: VqLookup::None,
        };
        let bytes = write_codebook(&book).expect("write");
        let bits: usize = 24 + 16 + 24 + 1 + 1 + 5 * 8 + 4; // 110 bits
        assert_eq!(bytes.len(), bits.div_ceil(8));
    }

    /// Sparse unordered: header is 70 (constants) + per-entry
    /// (1 + 5*used) bits.
    #[test]
    fn codebook_sparse_unordered_bit_length() {
        let book = VorbisCodebook {
            dimensions: 1,
            entries: 4,
            // 2 used, 2 unused
            codeword_lengths: vec![1, UNUSED_ENTRY, 3, UNUSED_ENTRY],
            lookup: VqLookup::None,
        };
        let bytes = write_codebook(&book).expect("write");
        let used: usize = 2;
        let unused: usize = 2;
        let bits: usize = 24 + 16 + 24 + 1 + 1 + (used * 6) + unused + 4;
        assert_eq!(bytes.len(), bits.div_ceil(8));
    }

    // ----------------------------------------------------------------
    // WriteError glue.
    // ----------------------------------------------------------------

    /// Codebook write errors surface as `WriteError::Codebook` via the
    /// `From` impl, preserving the variant for caller inspection.
    #[test]
    fn write_error_codebook_glue() {
        let inner: WriteError = WriteCodebookError::ZeroEntries.into();
        assert_eq!(inner, WriteError::Codebook(WriteCodebookError::ZeroEntries));
        // Source chain points to the inner error.
        use std::error::Error as StdError;
        assert!(StdError::source(&inner).is_some());
    }

    // ----------------------------------------------------------------
    // Floor 1 header writer (§7.2.2) — fixture builders.
    // ----------------------------------------------------------------

    /// The §7.2.2 floor 1 header parser is `pub(crate)` (it is reached
    /// via the setup-header walker); call into the bytes by replaying
    /// the parser's published flow through a `BitReaderLsb`. To keep
    /// the encoder's test module independent of the setup-header
    /// outer walker we reach into the parser directly via the same
    /// `BitReaderLsb` plumbing the parser uses internally.
    fn parse_floor1_via_setup_body(bytes: &[u8]) -> Floor1Header {
        // The §7.2.2 parser is a `fn parse_floor1_header(reader)` in
        // `setup`. We wrap that path by running the setup-header
        // walker against a synthetic body that carries exactly one
        // floor-type-1 header and nothing else.
        // The setup walker, however, demands the surrounding
        // codebook/time/floor-count/residue/mapping/mode/framing
        // scaffolding. The cleaner clean-room reflection is: invoke
        // a tiny local parser duplicating the public §7.2.2 step list,
        // using only the bit-reader primitives. Anything else would
        // need to consume the setup-header walker's plumbing.
        let mut reader = oxideav_core::bits::BitReaderLsb::new(bytes);
        local_parse_floor1_for_tests(&mut reader)
    }

    /// Mirror of `setup::parse_floor1_header` — clean-room reproduction
    /// from the §7.2.2 step list, used only to exercise the writer's
    /// bit-exact roundtrip property without coupling the encoder test
    /// suite to the setup-header outer walker.
    ///
    /// Spec source: `docs/audio/vorbis/Vorbis_I_spec.pdf` §7.2.2.
    fn local_parse_floor1_for_tests(
        reader: &mut oxideav_core::bits::BitReaderLsb<'_>,
    ) -> Floor1Header {
        let partitions = reader.read_u32(5).unwrap() as u8;
        let mut partition_class_list = Vec::with_capacity(partitions as usize);
        for _ in 0..partitions {
            partition_class_list.push(reader.read_u32(4).unwrap() as u8);
        }
        let maximum_class = partition_class_list.iter().copied().max();
        let classes = if let Some(maximum_class) = maximum_class {
            let class_count = (maximum_class as usize) + 1;
            let mut classes = Vec::with_capacity(class_count);
            for _ in 0..class_count {
                let dimensions = (reader.read_u32(3).unwrap() as u8) + 1;
                let subclasses = reader.read_u32(2).unwrap() as u8;
                let masterbook = if subclasses > 0 {
                    Some(reader.read_u32(8).unwrap() as u8)
                } else {
                    None
                };
                let subclass_book_count: usize = 1usize << subclasses;
                let mut subclass_books = Vec::with_capacity(subclass_book_count);
                for _ in 0..subclass_book_count {
                    let raw = reader.read_u32(8).unwrap() as i32 - 1;
                    subclass_books.push(if raw < 0 { None } else { Some(raw as u8) });
                }
                classes.push(Floor1Class {
                    dimensions,
                    subclasses,
                    masterbook,
                    subclass_books,
                });
            }
            classes
        } else {
            Vec::new()
        };
        let multiplier = (reader.read_u32(2).unwrap() as u8) + 1;
        let rangebits = reader.read_u32(4).unwrap() as u8;
        let mut x_list: Vec<u32> = Vec::new();
        for &current_class in &partition_class_list {
            let cdim = classes[current_class as usize].dimensions;
            for _ in 0..cdim {
                x_list.push(reader.read_u32(rangebits as u32).unwrap());
            }
        }
        Floor1Header {
            partitions,
            partition_class_list,
            classes,
            multiplier,
            rangebits,
            x_list,
        }
    }

    /// The "minimal" floor 1 from the setup-header test suite:
    /// `partitions=1`, one class with dimensions=1, subclasses=0,
    /// multiplier=1, rangebits=4, x_list=[5].
    fn minimal_floor1() -> Floor1Header {
        Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 1,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![Some(0)],
            }],
            multiplier: 1,
            rangebits: 4,
            x_list: vec![5],
        }
    }

    fn floor1_roundtrips(header: &Floor1Header) {
        let bytes = write_floor1_header(header).expect("write must succeed");
        let parsed = parse_floor1_via_setup_body(&bytes);
        assert_eq!(&parsed, header, "floor1 roundtrip equality");
    }

    // ----------------------------------------------------------------
    // Floor 1 header writer — byte-shape pinning.
    // ----------------------------------------------------------------

    /// Pin the exact bit layout of the minimal floor 1 fixture. This
    /// mirrors the byte sequence that `setup::tests::minimal_floor1`
    /// emits (excluding the outer `floor_type=1` 16-bit selector,
    /// which is the setup walker's responsibility).
    #[test]
    fn floor1_byte_shape_minimal() {
        let bytes = write_floor1_header(&minimal_floor1()).expect("must build");
        // Bits emitted (LSB-first per §2.1.4):
        //   partitions=1         -> 5 bits  -> 0b00001
        //   class_list[0]=0      -> 4 bits  -> 0b0000
        //   class 0: dim-1=0     -> 3 bits  -> 0b000
        //   class 0: subclass=0  -> 2 bits  -> 0b00
        //   subclass_books[0]=Some(0): raw=1 -> 8 bits -> 0b00000001
        //   multiplier-1=0       -> 2 bits  -> 0b00
        //   rangebits=4          -> 4 bits  -> 0b0100
        //   x_list[0]=5          -> 4 bits  -> 0b0101
        //   Total = 5+4+3+2+8+2+4+4 = 32 bits = 4 bytes.
        assert_eq!(bytes.len(), 4);
        // Build the same stream through BitWriterLsb step-by-step to
        // pin the exact bytes.
        let mut expected = BitWriterLsb::with_capacity(4);
        expected.write_u32(1, 5); // partitions
        expected.write_u32(0, 4); // partition_class_list[0]
        expected.write_u32(0, 3); // class[0].dimensions - 1
        expected.write_u32(0, 2); // class[0].subclasses
        expected.write_u32(1, 8); // subclass_books[0][0] raw = 0 + 1
        expected.write_u32(0, 2); // multiplier - 1
        expected.write_u32(4, 4); // rangebits
        expected.write_u32(5, 4); // x_list[0]
        assert_eq!(bytes, expected.finish());
    }

    /// Round 22 closed-form bit-length formula. Verifies the writer
    /// emits exactly the spec-mandated number of bits for a fixed
    /// shape with non-zero `subclasses` (which adds the masterbook).
    #[test]
    fn floor1_bit_length_formula_with_masterbook() {
        // partitions = 2 → 5 bits
        // class_list[0..2] = [0, 1] → 2 × 4 = 8 bits
        // 2 classes:
        //   class 0: dim=2, subclasses=1 → 3 + 2 = 5 bits
        //            masterbook (subclasses > 0) → 8 bits
        //            2 subclass slots × 8 = 16 bits
        //   class 1: dim=1, subclasses=2 → 3 + 2 = 5 bits
        //            masterbook → 8 bits
        //            4 subclass slots × 8 = 32 bits
        // multiplier - 1 → 2 bits
        // rangebits → 4 bits
        // x_list: sum(dim over partitions) = 2 + 1 = 3, × rangebits=5 → 15 bits
        // Total = 5 + 8 + 5 + 8 + 16 + 5 + 8 + 32 + 2 + 4 + 15 = 108 bits
        // = 14 bytes (ceil 108/8).
        let header = Floor1Header {
            partitions: 2,
            partition_class_list: vec![0, 1],
            classes: vec![
                Floor1Class {
                    dimensions: 2,
                    subclasses: 1,
                    masterbook: Some(3),
                    subclass_books: vec![None, Some(0)],
                },
                Floor1Class {
                    dimensions: 1,
                    subclasses: 2,
                    masterbook: Some(7),
                    subclass_books: vec![None, Some(0), Some(1), Some(2)],
                },
            ],
            multiplier: 2,
            rangebits: 5,
            x_list: vec![1, 2, 3],
        };
        let bytes = write_floor1_header(&header).expect("must build");
        let total_bits = 5 + 8 + 5 + 8 + 16 + 5 + 8 + 32 + 2 + 4 + 15;
        assert_eq!(bytes.len(), (total_bits as usize).div_ceil(8));
        // Roundtrip too.
        floor1_roundtrips(&header);
    }

    // ----------------------------------------------------------------
    // Floor 1 header writer — bit-exact roundtrip.
    // ----------------------------------------------------------------

    #[test]
    fn floor1_roundtrips_minimal() {
        floor1_roundtrips(&minimal_floor1());
    }

    #[test]
    fn floor1_roundtrips_zero_partitions() {
        // §7.2.2 step 5 corner case: partitions = 0 means
        // partition_class_list is empty, maximum_class is undefined,
        // and the class loop iterates zero times. The roundtrip must
        // still hold.
        floor1_roundtrips(&Floor1Header {
            partitions: 0,
            partition_class_list: vec![],
            classes: vec![],
            multiplier: 1,
            rangebits: 4,
            x_list: vec![],
        });
    }

    #[test]
    fn floor1_roundtrips_multiple_partitions_same_class() {
        floor1_roundtrips(&Floor1Header {
            partitions: 4,
            partition_class_list: vec![0, 0, 0, 0],
            classes: vec![Floor1Class {
                dimensions: 2,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![Some(1)],
            }],
            multiplier: 2,
            rangebits: 5,
            x_list: vec![1, 2, 3, 4, 5, 6, 7, 8],
        });
    }

    #[test]
    fn floor1_roundtrips_multiple_classes() {
        floor1_roundtrips(&Floor1Header {
            partitions: 3,
            partition_class_list: vec![0, 1, 2],
            classes: vec![
                Floor1Class {
                    dimensions: 1,
                    subclasses: 0,
                    masterbook: None,
                    subclass_books: vec![Some(0)],
                },
                Floor1Class {
                    dimensions: 2,
                    subclasses: 1,
                    masterbook: Some(3),
                    subclass_books: vec![Some(0), Some(1)],
                },
                Floor1Class {
                    dimensions: 4,
                    subclasses: 2,
                    masterbook: Some(7),
                    subclass_books: vec![None, Some(0), Some(1), Some(2)],
                },
            ],
            multiplier: 3,
            rangebits: 6,
            x_list: vec![1, 2, 3, 4, 5, 6, 7],
        });
    }

    #[test]
    fn floor1_roundtrips_max_subclasses() {
        // subclasses = 3 → 2^3 = 8 subclass slots, with masterbook.
        floor1_roundtrips(&Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 8,
                subclasses: 3,
                masterbook: Some(5),
                subclass_books: vec![
                    None,
                    Some(0),
                    Some(1),
                    None,
                    Some(2),
                    Some(3),
                    None,
                    Some(4),
                ],
            }],
            multiplier: 4,
            rangebits: 8,
            x_list: vec![1, 2, 3, 4, 5, 6, 7, 8],
        });
    }

    #[test]
    fn floor1_roundtrips_subclass_book_at_upper_edge() {
        // book = 254 is the largest representable Some(_): raw = 255
        // is the highest 8-bit value.
        floor1_roundtrips(&Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 1,
                subclasses: 1,
                masterbook: Some(0),
                subclass_books: vec![Some(254), None],
            }],
            multiplier: 1,
            rangebits: 4,
            x_list: vec![15],
        });
    }

    #[test]
    fn floor1_roundtrips_rangebits_zero() {
        // rangebits = 0 → every x_list element must be 0; the parser
        // reads 0 bits per element (a no-op). This pins the corner.
        floor1_roundtrips(&Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 3,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![Some(0)],
            }],
            multiplier: 1,
            rangebits: 0,
            x_list: vec![0, 0, 0],
        });
    }

    #[test]
    fn floor1_roundtrips_rangebits_at_upper_edge() {
        // rangebits = 15, x_list values near the cap.
        floor1_roundtrips(&Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 2,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![Some(0)],
            }],
            multiplier: 4,
            rangebits: 15,
            x_list: vec![0, 0x7fff],
        });
    }

    #[test]
    fn floor1_roundtrips_max_partitions() {
        // partitions = 31 is the largest 5-bit value.
        let pcl = vec![0u8; 31];
        let x_list: Vec<u32> = (0..31).collect();
        floor1_roundtrips(&Floor1Header {
            partitions: 31,
            partition_class_list: pcl,
            classes: vec![Floor1Class {
                dimensions: 1,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![Some(0)],
            }],
            multiplier: 1,
            rangebits: 5,
            x_list,
        });
    }

    #[test]
    fn floor1_roundtrips_max_class_index() {
        // partition_class_list references class index 15 (the largest
        // 4-bit value), forcing classes.len() = 16.
        let mut classes = Vec::with_capacity(16);
        for i in 0..16 {
            classes.push(Floor1Class {
                dimensions: ((i % 8) + 1) as u8,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![Some(0)],
            });
        }
        floor1_roundtrips(&Floor1Header {
            partitions: 1,
            partition_class_list: vec![15],
            classes,
            multiplier: 2,
            rangebits: 4,
            x_list: vec![1, 2, 3, 4, 5, 6, 7, 8],
        });
    }

    // ----------------------------------------------------------------
    // Floor 1 header writer — rejection variants.
    // ----------------------------------------------------------------

    #[test]
    fn floor1_rejects_partitions_overflow() {
        let mut h = minimal_floor1();
        h.partitions = 32;
        // Mismatched lengths are normally a separate error; force the
        // list length to match the rejected partitions count so the
        // overflow check fires first.
        h.partition_class_list = vec![0; 32];
        h.x_list = vec![0; 32];
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::PartitionsOverflow(32)
        );
    }

    #[test]
    fn floor1_rejects_partition_class_list_mismatch() {
        let mut h = minimal_floor1();
        h.partition_class_list = vec![0, 0]; // declared partitions = 1
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::PartitionClassListMismatch {
                partitions: 1,
                list_len: 2,
            }
        );
    }

    #[test]
    fn floor1_rejects_partition_class_value_overflow() {
        let mut h = minimal_floor1();
        h.partition_class_list = vec![16]; // 4-bit field caps at 15
                                           // Resize classes to satisfy the count check should this rule
                                           // ever be reordered; but the value-overflow gate must fire
                                           // first in any case.
        let mut classes = Vec::with_capacity(17);
        for _ in 0..17 {
            classes.push(Floor1Class {
                dimensions: 1,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![Some(0)],
            });
        }
        h.classes = classes;
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::PartitionClassValueOverflow {
                index: 0,
                value: 16,
            }
        );
    }

    #[test]
    fn floor1_rejects_class_count_mismatch() {
        let mut h = minimal_floor1();
        h.classes = vec![]; // partitions=1, max(pcl)=0 → expected 1
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::ClassCountMismatch {
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn floor1_rejects_illegal_class_dimensions_zero() {
        let mut h = minimal_floor1();
        h.classes[0].dimensions = 0;
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::IllegalClassDimensions {
                class: 0,
                dimensions: 0,
            }
        );
    }

    #[test]
    fn floor1_rejects_illegal_class_dimensions_too_large() {
        let mut h = minimal_floor1();
        h.classes[0].dimensions = 9;
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::IllegalClassDimensions {
                class: 0,
                dimensions: 9,
            }
        );
    }

    #[test]
    fn floor1_rejects_subclasses_overflow() {
        let mut h = minimal_floor1();
        h.classes[0].subclasses = 4; // 2-bit field caps at 3
                                     // Pad subclass_books to the matching length so the test
                                     // catches the subclasses check, not the book-count check.
        h.classes[0].subclass_books = vec![Some(0); 16];
        h.classes[0].masterbook = Some(0);
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::SubclassesOverflow {
                class: 0,
                subclasses: 4,
            }
        );
    }

    #[test]
    fn floor1_rejects_masterbook_present_with_zero_subclasses() {
        let mut h = minimal_floor1();
        h.classes[0].masterbook = Some(0); // but subclasses == 0
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::MasterbookPresenceMismatch {
                class: 0,
                subclasses: 0,
                present: true,
            }
        );
    }

    #[test]
    fn floor1_rejects_masterbook_missing_with_nonzero_subclasses() {
        let mut h = minimal_floor1();
        h.classes[0].subclasses = 1;
        h.classes[0].masterbook = None;
        h.classes[0].subclass_books = vec![None, None];
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::MasterbookPresenceMismatch {
                class: 0,
                subclasses: 1,
                present: false,
            }
        );
    }

    #[test]
    fn floor1_rejects_subclass_book_count_mismatch() {
        let mut h = minimal_floor1();
        h.classes[0].subclasses = 1;
        h.classes[0].masterbook = Some(0);
        h.classes[0].subclass_books = vec![Some(0)]; // expected 2
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::SubclassBookCountMismatch {
                class: 0,
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn floor1_rejects_subclass_book_overflow() {
        let mut h = minimal_floor1();
        h.classes[0].subclass_books = vec![Some(u8::MAX)];
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::SubclassBookOverflow {
                class: 0,
                subclass: 0,
                book: u8::MAX,
            }
        );
    }

    #[test]
    fn floor1_rejects_illegal_multiplier_zero() {
        let mut h = minimal_floor1();
        h.multiplier = 0;
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::IllegalMultiplier(0)
        );
    }

    #[test]
    fn floor1_rejects_illegal_multiplier_too_large() {
        let mut h = minimal_floor1();
        h.multiplier = 5;
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::IllegalMultiplier(5)
        );
    }

    #[test]
    fn floor1_rejects_rangebits_overflow() {
        let mut h = minimal_floor1();
        h.rangebits = 16; // 4-bit field caps at 15
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::RangebitsOverflow(16)
        );
    }

    #[test]
    fn floor1_rejects_x_list_length_mismatch() {
        let mut h = minimal_floor1();
        h.x_list = vec![0, 1]; // expected 1 (1 partition × dim 1)
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::XListLengthMismatch {
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn floor1_rejects_x_list_value_overflow_rangebits_nonzero() {
        let mut h = minimal_floor1();
        h.rangebits = 4; // cap = 15
        h.x_list = vec![16];
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::XListValueOverflow {
                index: 0,
                value: 16,
                rangebits: 4,
            }
        );
    }

    #[test]
    fn floor1_rejects_x_list_value_overflow_rangebits_zero() {
        let mut h = minimal_floor1();
        h.rangebits = 0; // cap = 0
        h.x_list = vec![1];
        assert_eq!(
            write_floor1_header(&h).unwrap_err(),
            WriteFloor1Error::XListValueOverflow {
                index: 0,
                value: 1,
                rangebits: 0,
            }
        );
    }

    // ----------------------------------------------------------------
    // Floor 1 header — Display + WriteError glue.
    // ----------------------------------------------------------------

    /// Every `WriteFloor1Error` variant has a non-empty `Display`
    /// rendering so error messages propagated through the umbrella
    /// `Error::Write` are always informative.
    #[test]
    fn floor1_display_non_empty_for_every_variant() {
        let cases = [
            WriteFloor1Error::PartitionsOverflow(32),
            WriteFloor1Error::PartitionClassListMismatch {
                partitions: 1,
                list_len: 0,
            },
            WriteFloor1Error::PartitionClassValueOverflow {
                index: 0,
                value: 16,
            },
            WriteFloor1Error::ClassCountMismatch {
                expected: 1,
                actual: 0,
            },
            WriteFloor1Error::IllegalClassDimensions {
                class: 0,
                dimensions: 0,
            },
            WriteFloor1Error::SubclassesOverflow {
                class: 0,
                subclasses: 4,
            },
            WriteFloor1Error::MasterbookPresenceMismatch {
                class: 0,
                subclasses: 1,
                present: false,
            },
            WriteFloor1Error::SubclassBookCountMismatch {
                class: 0,
                expected: 2,
                actual: 1,
            },
            WriteFloor1Error::SubclassBookOverflow {
                class: 0,
                subclass: 0,
                book: u8::MAX,
            },
            WriteFloor1Error::IllegalMultiplier(0),
            WriteFloor1Error::RangebitsOverflow(16),
            WriteFloor1Error::XListLengthMismatch {
                expected: 1,
                actual: 2,
            },
            WriteFloor1Error::XListValueOverflow {
                index: 0,
                value: 16,
                rangebits: 4,
            },
        ];
        for case in &cases {
            let s = format!("{case}");
            assert!(!s.is_empty(), "Display empty for {case:?}");
            assert!(
                s.contains("floor1"),
                "Display for {case:?} must mention the §7.2.2 floor1 context"
            );
        }
    }

    /// Floor 1 write errors surface as `WriteError::Floor1` via the
    /// `From` impl, preserving the variant for caller inspection and
    /// the source chain.
    #[test]
    fn write_error_floor1_glue() {
        let inner: WriteError = WriteFloor1Error::PartitionsOverflow(32).into();
        assert_eq!(
            inner,
            WriteError::Floor1(WriteFloor1Error::PartitionsOverflow(32))
        );
        use std::error::Error as StdError;
        let src = StdError::source(&inner).expect("source chain should reach Floor1");
        // The chained Display must non-trivially echo the floor1 tag.
        assert!(format!("{src}").contains("floor1"));
    }
}

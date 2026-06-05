//! Vorbis I header-packet + codebook encoder primitives (rounds 195 + 201 + 206 + 212 + 218 + 228 + 240).
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
//! * [`write_floor1_header`] — serialises a [`Floor1Header`] to the
//!   §7.2.2 floor-type-1 setup-header bit pattern.
//! * [`write_floor0_header`] — serialises a [`Floor0Header`] to the
//!   §6.2.1 floor-type-0 setup-header bit pattern. Sibling of
//!   [`write_floor1_header`] and the second per-floor encoder
//!   primitive.
//! * [`write_residue_header`] — serialises a [`ResidueHeader`] to the
//!   §8.6.1 residue-header bit pattern. Common to all three residue
//!   types (0, 1, 2); the outer 16-bit `residue_type` selector is the
//!   setup walker's responsibility, mirroring the floor 0 / floor 1
//!   convention. Round 218.
//! * [`write_mode_header`] — serialises a [`ModeHeader`] to the
//!   §4.2.4 "Modes" body bit pattern. Fixed-width 41-bit body
//!   (1-bit `blockflag`, 16-bit `windowtype`, 16-bit `transformtype`,
//!   8-bit `mapping`). The writer is the byte-exact inverse of the
//!   round-5 mode parser and refuses any header whose `mapping`
//!   index falls outside the supplied `mapping_count`. Round 228.
//! * [`write_audio_packet_header`] — serialises an [`AudioPacketHeader`]
//!   to the §4.3.1 audio-packet prelude bit pattern. The byte-exact
//!   inverse of [`crate::packet::read_packet_header`]: 1-bit
//!   `packet_type`, `ilog([vorbis_mode_count] - 1)`-bit `mode_number`,
//!   then two 1-bit window flags on long blocks. The writer cross-
//!   checks the cached `(blockflag, n)` pair against the §4.3.1 step-3
//!   blocksize selection so a malformed struct cannot silently
//!   round-trip. Round 240 — the first audio-packet WRITE primitive.
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
//! Round 27 (umbrella round 234) landed [`write_setup_header`]: the
//! wrapping §4.2.4 setup-header WRITE primitive that splices the six
//! nested-block writers (codebook / floor 0 / floor 1 / residue /
//! mapping / mode) into a single byte-aligned packet matching the
//! round-5 [`crate::setup::parse_setup_header`] reader.
//!
//! Round 240 lands [`write_audio_packet_header`]: the first audio-
//! packet WRITE primitive — the §4.3.1 prelude. The rest of the §4.3
//! audio packet (floor / residue / spectrum / inverse couple / IMDCT)
//! and a wrapping §4.3 audio-packet writer that splices the prelude
//! into the per-channel payload remain explicit followups for
//! subsequent rounds.

use core::fmt;

use oxideav_core::bits::BitWriterLsb;

use crate::codebook::{float32_pack, ilog, lookup1_values, VorbisCodebook, VqLookup, UNUSED_ENTRY};
use crate::comment::VorbisCommentHeader;
use crate::identification::VorbisIdentificationHeader;
use crate::packet::AudioPacketHeader;
use crate::setup::{
    Floor0Header, Floor1Header, FloorKind, MappingHeader, ModeHeader, ResidueHeader,
    VorbisSetupHeader, SETUP_PACKET_MAGIC, SETUP_PACKET_TYPE,
};

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
    /// A nested floor type 0 header (§6.2.1) failed one of the
    /// writer-side invariants checked by [`write_floor0_header`].
    Floor0(WriteFloor0Error),
    /// A nested residue header (§8.6.1) failed one of the writer-side
    /// invariants checked by [`write_residue_header`].
    Residue(WriteResidueError),
    /// A nested mapping configuration (§4.2.4 "Mappings") failed one
    /// of the writer-side invariants checked by [`write_mapping_header`].
    Mapping(WriteMappingError),
    /// A nested mode configuration (§4.2.4 "Modes") failed one of the
    /// writer-side invariants checked by [`write_mode_header`].
    Mode(WriteModeError),
    /// A wrapping setup-header (§4.2.4) failed one of the writer-side
    /// invariants checked by [`write_setup_header`].
    Setup(WriteSetupError),
    /// An audio-packet prelude (§4.3.1) failed one of the writer-side
    /// invariants checked by [`write_audio_packet_header`].
    AudioPacket(WriteAudioPacketHeaderError),
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

/// Errors that may arise while writing a Vorbis I floor type 0 header
/// (§6.2.1) via [`write_floor0_header`].
///
/// Each variant flags a §6.2.1 invariant the caller-supplied
/// [`Floor0Header`] does not satisfy. The writer refuses the call
/// without emitting any bits, preserving the bit-exact roundtrip
/// guarantee
/// `parse_floor0_header(&mut BitReaderLsb::new(&write_floor0_header(&h)?))? == h`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteFloor0Error {
    /// `amplitude_bits` exceeded the 6-bit `floor0_amplitude_bits`
    /// field's representable range (`0..=63`).
    AmplitudeBitsOverflow(u8),
    /// `book_list` was empty. §6.2.1 encodes
    /// `[floor0_number_of_books] = read 4 bits + 1`, so the smallest
    /// representable book list has length `1`. The writer refuses
    /// rather than emit a header whose round-trip would conjure a
    /// non-existent book index.
    EmptyBookList,
    /// `book_list.len()` exceeded the largest count representable by
    /// the 4-bit `floor0_number_of_books - 1` field (`16`).
    BookListTooLong(usize),
}

impl fmt::Display for WriteFloor0Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteFloor0Error::AmplitudeBitsOverflow(v) => write!(
                f,
                "vorbis floor0 header (write): amplitude_bits={v} > 63 (§6.2.1 step 4 is a 6-bit field)"
            ),
            WriteFloor0Error::EmptyBookList => write!(
                f,
                "vorbis floor0 header (write): book_list is empty (§6.2.1 step 6 encodes the count as `read 4 bits + 1`, so the minimum length is 1)"
            ),
            WriteFloor0Error::BookListTooLong(n) => write!(
                f,
                "vorbis floor0 header (write): book_list.len()={n} > 16 (§6.2.1 step 6 is a 4-bit + 1 field, so the maximum length is 16)"
            ),
        }
    }
}

impl std::error::Error for WriteFloor0Error {}

/// Errors that may arise while writing a Vorbis I residue header
/// (§8.6.1) via [`write_residue_header`].
///
/// Each variant flags a §8.6.1 invariant the caller-supplied
/// [`ResidueHeader`] does not satisfy. The writer refuses the call
/// without emitting any bits, preserving the bit-exact roundtrip
/// guarantee
/// `parse_residue_header(&write_residue_header(&h)?, h.residue_type)? == h`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteResidueError {
    /// `residue_type` was a value outside `{0, 1, 2}`. §8.6 only
    /// defines those three formats; the §4.2.4 setup walker rejects
    /// `> 2` on input and the writer mirrors the rule — even though
    /// the §8.6.1 layout itself does not serialise `residue_type` (the
    /// 16-bit selector is the setup walker's responsibility), the
    /// writer refuses the call so a stale field cannot quietly persist.
    UnsupportedResidueType(u16),
    /// `residue_begin` exceeded the 24-bit `[residue_begin]` field's
    /// representable range (`0..=0xFF_FFFF`).
    ResidueBeginOverflow(u32),
    /// `residue_end` exceeded the 24-bit `[residue_end]` field's
    /// representable range (`0..=0xFF_FFFF`).
    ResidueEndOverflow(u32),
    /// `partition_size` was `0`, or exceeded the encodable cap. §8.6.1
    /// stores the field as `read 24 bits + 1`, so the legal range is
    /// `1..=2^24`. The writer refuses both ends.
    PartitionSizeOutOfRange(u32),
    /// `classifications` was `0`, or exceeded the encodable cap. §8.6.1
    /// stores the field as `read 6 bits + 1`, so the legal range is
    /// `1..=64`. The writer refuses both ends.
    ClassificationsOutOfRange(u8),
    /// `cascade.len()` did not equal `classifications`. The §8.6.1
    /// layout has exactly one cascade byte per classification.
    CascadeLengthMismatch {
        /// `classifications` from the header.
        classifications: u8,
        /// Actual length of the supplied `cascade` vector.
        actual: usize,
    },
    /// `books.len()` did not equal `classifications`. The §8.6.1
    /// layout has exactly one 8-slot row per classification.
    BooksLengthMismatch {
        /// `classifications` from the header.
        classifications: u8,
        /// Actual length of the supplied `books` vector.
        actual: usize,
    },
    /// A `books[class][stage]` slot was `Some(_)` but the matching
    /// `cascade[class]` bit was unset, or the slot was `None` but the
    /// matching cascade bit was set. The §8.6.1 layout reads each
    /// 8-bit `residue_books[class][stage]` *iff* `cascade[class]` bit
    /// `stage` is set — the writer refuses any inconsistency rather
    /// than emit a header whose round-trip would silently differ.
    BooksCascadeMismatch {
        /// Classification index.
        class: usize,
        /// Cascade stage index `0..=7`.
        stage: usize,
        /// `true` if a codebook was supplied but the cascade bit is
        /// clear; `false` if no codebook was supplied but the cascade
        /// bit is set.
        book_present: bool,
    },
}

impl fmt::Display for WriteResidueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteResidueError::UnsupportedResidueType(t) => write!(
                f,
                "vorbis residue header (write): residue_type={t} not in {{0,1,2}} (§4.2.4 step 2c / §8.6)"
            ),
            WriteResidueError::ResidueBeginOverflow(v) => write!(
                f,
                "vorbis residue header (write): residue_begin={v} > 0xFFFFFF (§8.6.1 is a 24-bit field)"
            ),
            WriteResidueError::ResidueEndOverflow(v) => write!(
                f,
                "vorbis residue header (write): residue_end={v} > 0xFFFFFF (§8.6.1 is a 24-bit field)"
            ),
            WriteResidueError::PartitionSizeOutOfRange(v) => write!(
                f,
                "vorbis residue header (write): partition_size={v} outside legal 1..=2^24 (§8.6.1 stores `read 24 bits + 1`)"
            ),
            WriteResidueError::ClassificationsOutOfRange(v) => write!(
                f,
                "vorbis residue header (write): classifications={v} outside legal 1..=64 (§8.6.1 stores `read 6 bits + 1`)"
            ),
            WriteResidueError::CascadeLengthMismatch {
                classifications,
                actual,
            } => write!(
                f,
                "vorbis residue header (write): cascade.len()={actual} != classifications={classifications} (§8.6.1)"
            ),
            WriteResidueError::BooksLengthMismatch {
                classifications,
                actual,
            } => write!(
                f,
                "vorbis residue header (write): books.len()={actual} != classifications={classifications} (§8.6.1)"
            ),
            WriteResidueError::BooksCascadeMismatch {
                class,
                stage,
                book_present,
            } => write!(
                f,
                "vorbis residue header (write): books[{class}][{stage}] {} but cascade[{class}] bit {stage} {} (§8.6.1)",
                if *book_present { "is Some" } else { "is None" },
                if *book_present { "is clear" } else { "is set" },
            ),
        }
    }
}

impl std::error::Error for WriteResidueError {}

/// Errors that may arise while writing a Vorbis I mapping configuration
/// (§4.2.4 "Mappings") via [`write_mapping_header`].
///
/// Each variant flags an invariant the caller-supplied
/// [`MappingHeader`] does not satisfy with respect to either the
/// mapping body itself (§4.2.4 steps 2a–2c.v) or the surrounding
/// context (`audio_channels`, `floor_count`, `residue_count`) the
/// round-5 parser is fed. The writer refuses the call without
/// emitting any bits, preserving the bit-exact roundtrip guarantee
/// against [`crate::setup::MappingHeader`]-shaped input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteMappingError {
    /// `mapping_type` was nonzero. §4.2.4 step 2b only defines
    /// `mapping_type = 0` in Vorbis I and the round-5 parser rejects
    /// any other value with `ParseError::UnsupportedMappingType`.
    UnsupportedMappingType(u16),
    /// `audio_channels` (the context value supplied to the writer)
    /// was `0`. §4.2.2 requires `audio_channels > 0`; the §4.2.4
    /// "Mappings" decode steps assume the value is positive.
    ZeroAudioChannels,
    /// `submaps` (the value stored in [`MappingHeader::submaps`])
    /// fell outside the legal `1..=16` range encodable in the
    /// `read 4 bits + 1` field at §4.2.4 step 2c.i.
    SubmapsOutOfRange(u8),
    /// `coupling.len()` exceeded the encodable cap. §4.2.4 step
    /// 2c.ii stores the count as `read 8 bits + 1`, so the legal
    /// range is `1..=256`. The writer refuses any larger length.
    CouplingStepsOverflow(usize),
    /// A coupling step's magnitude/angle channel violated the
    /// §4.2.4 step 2c.ii "magnitude != angle, both <
    /// audio_channels" constraint.
    BadCouplingChannels {
        /// Index of the offending coupling step.
        step_index: usize,
        /// Magnitude-channel value supplied by the caller.
        magnitude_channel: u8,
        /// Angle-channel value supplied by the caller.
        angle_channel: u8,
        /// Context-supplied `audio_channels`.
        audio_channels: u8,
    },
    /// A coupling step's magnitude or angle channel did not fit in
    /// the `ilog(audio_channels - 1)`-bit field width prescribed by
    /// §4.2.4 step 2c.ii.A. This is a tighter check than
    /// [`Self::BadCouplingChannels`] for callers that supplied a
    /// value `< audio_channels` but somehow `>= 1 << channel_bits`
    /// (only reachable when context disagrees, but checked for
    /// safety).
    CouplingChannelOverflow {
        /// Index of the offending coupling step.
        step_index: usize,
        /// Channel value that did not fit in `channel_bits`.
        channel: u8,
        /// `ilog(audio_channels - 1)` width the field was checked
        /// against.
        channel_bits: u32,
    },
    /// `mux.len()` did not equal `audio_channels` when `submaps > 1`,
    /// or `mux` was non-empty when `submaps == 1` (§4.2.4 step 2c.iv
    /// only reads `mux[ch]` for each of the `audio_channels` channels
    /// when `submaps > 1`; the field is otherwise absent).
    MuxLengthMismatch {
        /// Expected length (`audio_channels` when `submaps > 1`, else
        /// `0`).
        expected: usize,
        /// Actual length of the supplied `mux` vector.
        actual: usize,
    },
    /// A `mux[ch]` value was `>= submaps`, i.e. it would dereference
    /// past the end of the per-submap configuration list.
    BadMuxValue {
        /// Channel index whose `mux[ch]` value is illegal.
        channel_index: usize,
        /// The illegal `mux` value.
        mux: u8,
        /// `submaps` from the header.
        submaps: u8,
    },
    /// `submap_configs.len()` did not equal `submaps`. §4.2.4 step
    /// 2c.v always reads one `(time_placeholder, floor, residue)`
    /// triple per submap.
    SubmapCountMismatch {
        /// `submaps` from the header.
        submaps: u8,
        /// Actual length of the supplied `submap_configs` vector.
        actual: usize,
    },
    /// A per-submap `floor` index was `>= floor_count`. §4.2.4 step
    /// 2c.v.B range-checks each floor index at parse time; the
    /// writer mirrors that rule.
    BadSubmapFloor {
        /// Submap index whose `floor` value is illegal.
        submap_index: usize,
        /// The illegal `floor` value.
        floor: u8,
        /// Context-supplied `floor_count`.
        floor_count: usize,
    },
    /// A per-submap `residue` index was `>= residue_count`. §4.2.4
    /// step 2c.v.C range-checks each residue index at parse time;
    /// the writer mirrors that rule.
    BadSubmapResidue {
        /// Submap index whose `residue` value is illegal.
        submap_index: usize,
        /// The illegal `residue` value.
        residue: u8,
        /// Context-supplied `residue_count`.
        residue_count: usize,
    },
}

impl fmt::Display for WriteMappingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteMappingError::UnsupportedMappingType(t) => write!(
                f,
                "vorbis mapping header (write): mapping_type={t} != 0 (Vorbis I §4.2.4 step 2b defines only type 0)"
            ),
            WriteMappingError::ZeroAudioChannels => write!(
                f,
                "vorbis mapping header (write): audio_channels=0 in writer context (§4.2.2 requires > 0)"
            ),
            WriteMappingError::SubmapsOutOfRange(v) => write!(
                f,
                "vorbis mapping header (write): submaps={v} outside legal 1..=16 (§4.2.4 step 2c.i stores `read 4 bits + 1`)"
            ),
            WriteMappingError::CouplingStepsOverflow(n) => write!(
                f,
                "vorbis mapping header (write): coupling.len()={n} > 256 (§4.2.4 step 2c.ii stores `read 8 bits + 1`)"
            ),
            WriteMappingError::BadCouplingChannels {
                step_index,
                magnitude_channel,
                angle_channel,
                audio_channels,
            } => write!(
                f,
                "vorbis mapping header (write): coupling[{step_index}] magnitude={magnitude_channel} angle={angle_channel} violates §4.2.4 step 2c.ii (need magnitude != angle, both < audio_channels={audio_channels})"
            ),
            WriteMappingError::CouplingChannelOverflow {
                step_index,
                channel,
                channel_bits,
            } => write!(
                f,
                "vorbis mapping header (write): coupling[{step_index}] channel={channel} does not fit in channel_bits={channel_bits} field (§4.2.4 step 2c.ii.A)"
            ),
            WriteMappingError::MuxLengthMismatch { expected, actual } => write!(
                f,
                "vorbis mapping header (write): mux.len()={actual} != expected={expected} (§4.2.4 step 2c.iv)"
            ),
            WriteMappingError::BadMuxValue {
                channel_index,
                mux,
                submaps,
            } => write!(
                f,
                "vorbis mapping header (write): mux[{channel_index}]={mux} >= submaps={submaps} (§4.2.4 step 2c.iv)"
            ),
            WriteMappingError::SubmapCountMismatch { submaps, actual } => write!(
                f,
                "vorbis mapping header (write): submap_configs.len()={actual} != submaps={submaps} (§4.2.4 step 2c.v)"
            ),
            WriteMappingError::BadSubmapFloor {
                submap_index,
                floor,
                floor_count,
            } => write!(
                f,
                "vorbis mapping header (write): submap_configs[{submap_index}].floor={floor} >= floor_count={floor_count} (§4.2.4 step 2c.v.B)"
            ),
            WriteMappingError::BadSubmapResidue {
                submap_index,
                residue,
                residue_count,
            } => write!(
                f,
                "vorbis mapping header (write): submap_configs[{submap_index}].residue={residue} >= residue_count={residue_count} (§4.2.4 step 2c.v.C)"
            ),
        }
    }
}

impl std::error::Error for WriteMappingError {}

/// Errors that may arise while writing a Vorbis I mode configuration
/// (§4.2.4 "Modes") via [`write_mode_header`].
///
/// Each variant flags an invariant the caller-supplied [`ModeHeader`]
/// does not satisfy with respect to either the mode body itself
/// (§4.2.4 step 2e) or the surrounding context (`mapping_count`) the
/// round-5 parser is fed. The writer refuses the call without
/// emitting any bits, preserving the bit-exact roundtrip guarantee
/// against [`crate::setup::ModeHeader`]-shaped input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteModeError {
    /// `windowtype` was nonzero. §4.2.4 step 2e specifies "zero is
    /// the only legal value in Vorbis I for [vorbis_mode_windowtype]";
    /// the round-5 parser rejects any other value with
    /// `ParseError::NonZeroModeWindowType`.
    NonZeroWindowType(u16),
    /// `transformtype` was nonzero. §4.2.4 step 2e specifies "zero is
    /// the only legal value in Vorbis I for
    /// [vorbis_mode_transformtype]"; the round-5 parser rejects any
    /// other value with `ParseError::NonZeroModeTransformType`.
    NonZeroTransformType(u16),
    /// `mapping` (the value stored in [`ModeHeader::mapping`]) was
    /// `>= mapping_count`, i.e. it would dereference past the end of
    /// the setup header's mapping table. §4.2.4 step 2e specifies
    /// "vorbis_mode_mapping must not be greater than the highest
    /// number mapping in use"; the round-5 parser mirrors that as
    /// `ParseError::BadModeMapping`.
    BadMapping {
        /// The illegal `mapping` value.
        mapping: u8,
        /// Context-supplied `mapping_count`.
        mapping_count: usize,
    },
}

impl fmt::Display for WriteModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteModeError::NonZeroWindowType(v) => write!(
                f,
                "vorbis mode header (write): windowtype={v} != 0 (Vorbis I §4.2.4 step 2e: zero is the only legal value)"
            ),
            WriteModeError::NonZeroTransformType(v) => write!(
                f,
                "vorbis mode header (write): transformtype={v} != 0 (Vorbis I §4.2.4 step 2e: zero is the only legal value)"
            ),
            WriteModeError::BadMapping {
                mapping,
                mapping_count,
            } => write!(
                f,
                "vorbis mode header (write): mapping={mapping} >= mapping_count={mapping_count} (§4.2.4 step 2e)"
            ),
        }
    }
}

impl std::error::Error for WriteModeError {}

/// Errors that may arise while writing a wrapping Vorbis I setup-header
/// packet (Vorbis I §4.2.4) via [`write_setup_header`].
///
/// Each variant flags a §4.2.4 invariant the caller-supplied
/// [`VorbisSetupHeader`] does not satisfy. The writer refuses the call
/// (returning the error) without emitting any bytes — keeping the
/// bit-exact roundtrip guarantee
/// `parse_setup_header(&write_setup_header(&h, audio_channels)?, audio_channels)? == h`
/// defensible for every legal input.
///
/// Nested codebook / floor / residue / mapping / mode failures are
/// re-exported as their existing dedicated error types (the writer
/// returns them through the umbrella [`WriteError`] when invoked via
/// [`write_setup_header`]).
#[derive(Debug, Clone, PartialEq)]
pub enum WriteSetupError {
    /// `audio_channels` (the value sourced from the identification
    /// header and supplied to [`write_setup_header`]) was zero. §4.2.2
    /// mandates `audio_channels > 0` for any conformant Vorbis I
    /// stream, and the round-5 setup parser rejects zero with
    /// `ParseError::ZeroAudioChannels`.
    ZeroAudioChannels,
    /// `codebooks` was empty, i.e. the §4.2.4 step "Codebooks" would
    /// have to encode `vorbis_codebook_count = 0`. The 8-bit
    /// `read 8 bits + 1` count container only encodes `1..=256`; the
    /// round-5 parser would never observe a zero-length codebook list.
    EmptyCodebooks,
    /// `codebooks.len()` exceeded the 8-bit `vorbis_codebook_count - 1`
    /// container (legal range `1..=256`).
    CodebookCountOverflow(usize),
    /// `time_placeholders` was empty. The 6-bit `read 6 bits + 1`
    /// count container only encodes `1..=64`; the round-5 parser
    /// would never observe a zero-length time-placeholder list.
    EmptyTimePlaceholders,
    /// `time_placeholders.len()` exceeded the 6-bit
    /// `vorbis_time_count - 1` container (legal range `1..=64`).
    TimeCountOverflow(usize),
    /// A `time_placeholders[i]` value was nonzero. §4.2.4 step 2
    /// mandates every time-domain transform value be zero on the wire;
    /// the round-5 parser rejects any nonzero entry with
    /// `ParseError::NonZeroTimePlaceholder`.
    NonZeroTimePlaceholder {
        /// The offending entry index in `0 .. time_placeholders.len()`.
        index: usize,
        /// The rejected 16-bit value.
        value: u16,
    },
    /// `floors` was empty. The 6-bit `read 6 bits + 1` count container
    /// only encodes `1..=64`.
    EmptyFloors,
    /// `floors.len()` exceeded the 6-bit `vorbis_floor_count - 1`
    /// container (legal range `1..=64`).
    FloorCountOverflow(usize),
    /// A `floors[i]` entry's `floor_type` was strictly greater than 1.
    /// §4.2.4 step 2d mandates `floor_type ∈ {0, 1}`; the round-5
    /// parser rejects any other value with
    /// `ParseError::UnsupportedFloorType`.
    UnsupportedFloorType {
        /// The offending entry index in `0 .. floors.len()`.
        index: usize,
        /// The rejected 16-bit `floor_type` value.
        floor_type: u16,
    },
    /// A `floors[i]` entry's `floor_type` field disagreed with its
    /// `kind` discriminant — e.g. `floor_type = 0` paired with a
    /// `FloorKind::Type1(_)` payload (or vice versa). The struct does
    /// not serialise to a packet the round-5 parser would round-trip
    /// to the same [`FloorHeader`].
    FloorTypeKindMismatch {
        /// The offending entry index in `0 .. floors.len()`.
        index: usize,
        /// The struct's stored `floor_type` field.
        floor_type: u16,
        /// `0` for `FloorKind::Type0(_)`, `1` for `FloorKind::Type1(_)`.
        kind_discriminant: u16,
    },
    /// `residues` was empty. The 6-bit `read 6 bits + 1` count
    /// container only encodes `1..=64`.
    EmptyResidues,
    /// `residues.len()` exceeded the 6-bit `vorbis_residue_count - 1`
    /// container (legal range `1..=64`).
    ResidueCountOverflow(usize),
    /// `mappings` was empty. The 6-bit `read 6 bits + 1` count
    /// container only encodes `1..=64`.
    EmptyMappings,
    /// `mappings.len()` exceeded the 6-bit `vorbis_mapping_count - 1`
    /// container (legal range `1..=64`).
    MappingCountOverflow(usize),
    /// `modes` was empty. The 6-bit `read 6 bits + 1` count container
    /// only encodes `1..=64`.
    EmptyModes,
    /// `modes.len()` exceeded the 6-bit `vorbis_mode_count - 1`
    /// container (legal range `1..=64`).
    ModeCountOverflow(usize),
    /// `framing_flag` was `false`. §4.2.4 step 3 mandates the trailing
    /// framing bit be set; the round-5 parser rejects `false` with
    /// `ParseError::BadFramingFlag`.
    BadFramingFlag,
}

impl fmt::Display for WriteSetupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteSetupError::ZeroAudioChannels => write!(
                f,
                "vorbis setup header (write): audio_channels = 0 (§4.2.2 mandates > 0)"
            ),
            WriteSetupError::EmptyCodebooks => write!(
                f,
                "vorbis setup header (write): codebooks is empty (§4.2.4 step \"Codebooks\" encodes count - 1 in 8 bits, range 1..=256)"
            ),
            WriteSetupError::CodebookCountOverflow(n) => write!(
                f,
                "vorbis setup header (write): codebook count {n} > 256 (§4.2.4 step \"Codebooks\" 8-bit count - 1 field)"
            ),
            WriteSetupError::EmptyTimePlaceholders => write!(
                f,
                "vorbis setup header (write): time_placeholders is empty (§4.2.4 step \"Time domain transforms\" encodes count - 1 in 6 bits, range 1..=64)"
            ),
            WriteSetupError::TimeCountOverflow(n) => write!(
                f,
                "vorbis setup header (write): time-placeholder count {n} > 64 (§4.2.4 step \"Time domain transforms\" 6-bit count - 1 field)"
            ),
            WriteSetupError::NonZeroTimePlaceholder { index, value } => write!(
                f,
                "vorbis setup header (write): time_placeholders[{index}] = {value} (§4.2.4 step 2 mandates every value be zero)"
            ),
            WriteSetupError::EmptyFloors => write!(
                f,
                "vorbis setup header (write): floors is empty (§4.2.4 step \"Floors\" encodes count - 1 in 6 bits, range 1..=64)"
            ),
            WriteSetupError::FloorCountOverflow(n) => write!(
                f,
                "vorbis setup header (write): floor count {n} > 64 (§4.2.4 step \"Floors\" 6-bit count - 1 field)"
            ),
            WriteSetupError::UnsupportedFloorType { index, floor_type } => write!(
                f,
                "vorbis setup header (write): floors[{index}].floor_type = {floor_type} not in {{0, 1}} (§4.2.4 step 2d)"
            ),
            WriteSetupError::FloorTypeKindMismatch {
                index,
                floor_type,
                kind_discriminant,
            } => write!(
                f,
                "vorbis setup header (write): floors[{index}].floor_type = {floor_type} disagrees with kind discriminant {kind_discriminant}"
            ),
            WriteSetupError::EmptyResidues => write!(
                f,
                "vorbis setup header (write): residues is empty (§4.2.4 step \"Residues\" encodes count - 1 in 6 bits, range 1..=64)"
            ),
            WriteSetupError::ResidueCountOverflow(n) => write!(
                f,
                "vorbis setup header (write): residue count {n} > 64 (§4.2.4 step \"Residues\" 6-bit count - 1 field)"
            ),
            WriteSetupError::EmptyMappings => write!(
                f,
                "vorbis setup header (write): mappings is empty (§4.2.4 step \"Mappings\" encodes count - 1 in 6 bits, range 1..=64)"
            ),
            WriteSetupError::MappingCountOverflow(n) => write!(
                f,
                "vorbis setup header (write): mapping count {n} > 64 (§4.2.4 step \"Mappings\" 6-bit count - 1 field)"
            ),
            WriteSetupError::EmptyModes => write!(
                f,
                "vorbis setup header (write): modes is empty (§4.2.4 step \"Modes\" encodes count - 1 in 6 bits, range 1..=64)"
            ),
            WriteSetupError::ModeCountOverflow(n) => write!(
                f,
                "vorbis setup header (write): mode count {n} > 64 (§4.2.4 step \"Modes\" 6-bit count - 1 field)"
            ),
            WriteSetupError::BadFramingFlag => write!(
                f,
                "vorbis setup header (write): framing_flag = false (§4.2.4 step 3 mandates the trailing framing bit be set)"
            ),
        }
    }
}

impl std::error::Error for WriteSetupError {}

/// Errors that may arise while writing a §4.3.1 audio-packet prelude
/// via [`write_audio_packet_header`].
///
/// Each variant flags a §4.3.1 invariant the caller-supplied
/// [`AudioPacketHeader`] does not satisfy together with the supplied
/// `(setup, blocksize_0, blocksize_1)` context. The writer refuses the
/// call without emitting any bits, preserving the bit-exact roundtrip
/// guarantee
/// `read_packet_header(&write_audio_packet_header(&h, &setup, b0, b1)?, &setup, b0, b1)? == h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteAudioPacketHeaderError {
    /// The supplied [`crate::setup::VorbisSetupHeader::modes`] list was
    /// empty. A well-formed Vorbis I stream always has at least one mode
    /// (§4.2.4 mode count is `read 6 bits + 1`, so the minimum is 1);
    /// this is a defensive caller-bug guard mirroring
    /// `PacketError::EmptyModeList` on the parser side.
    EmptyModeList,
    /// `header.mode_number` indexed past the supplied setup header's
    /// mode list. §4.3.1 step 2 cannot serialise an out-of-range
    /// `mode_number`; the parser would reject it with
    /// `PacketError::BadModeNumber`.
    BadModeNumber {
        /// The offending mode-number value supplied by the caller.
        mode_number: u32,
        /// `vorbis_mode_count` from the supplied setup header.
        mode_count: usize,
    },
    /// `header.blockflag` disagreed with the selected mode's
    /// [`ModeHeader::blockflag`]. §4.3.1 step 3 reads `blockflag` from
    /// the mode entry, not from the packet — a packet whose cached
    /// `blockflag` disagrees with `setup.modes[mode_number].blockflag`
    /// is internally inconsistent (the parser ignores the cached value
    /// on the wire because it is not transmitted; the writer cross-
    /// checks rather than silently emit the mode's value).
    BlockflagMismatch {
        /// `header.blockflag` as supplied by the caller.
        header_blockflag: bool,
        /// `setup.modes[mode_number].blockflag`.
        mode_blockflag: bool,
    },
    /// `header.n` disagreed with the §4.3.1 step-3 blocksize selection
    /// (`blocksize_0` when `blockflag` is clear, otherwise
    /// `blocksize_1`). Like [`Self::BlockflagMismatch`], `n` is not on
    /// the wire — the writer cross-checks it rather than silently emit
    /// a header whose cached `n` disagrees with the spec-derived value
    /// the parser will recompute.
    BlocksizeMismatch {
        /// `header.n` as supplied by the caller.
        header_n: usize,
        /// The spec-derived `n` value: `blocksize_0` for short blocks,
        /// `blocksize_1` for long blocks.
        expected_n: usize,
    },
    /// On a short block (`blockflag == false`), the §4.3.1 step 4b
    /// path is taken and the window flags are not transmitted on the
    /// wire. The reader returns `(false, false)` placeholders on a
    /// short block; the writer mirrors the rule and refuses to emit a
    /// header whose `previous_window_flag` or `next_window_flag` is set
    /// on a short block, because no equivalent bit pattern would round-
    /// trip to the same struct.
    ShortBlockHasWindowFlag {
        /// `header.previous_window_flag`.
        previous_window_flag: bool,
        /// `header.next_window_flag`.
        next_window_flag: bool,
    },
}

impl fmt::Display for WriteAudioPacketHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteAudioPacketHeaderError::EmptyModeList => write!(
                f,
                "vorbis audio packet (write): setup.modes is empty (§4.2.4 mode count >= 1)"
            ),
            WriteAudioPacketHeaderError::BadModeNumber {
                mode_number,
                mode_count,
            } => write!(
                f,
                "vorbis audio packet (write): mode_number={mode_number} >= mode_count={mode_count} (§4.3.1 step 2)"
            ),
            WriteAudioPacketHeaderError::BlockflagMismatch {
                header_blockflag,
                mode_blockflag,
            } => write!(
                f,
                "vorbis audio packet (write): header.blockflag={header_blockflag} disagrees with setup.modes[mode_number].blockflag={mode_blockflag} (§4.3.1 step 3)"
            ),
            WriteAudioPacketHeaderError::BlocksizeMismatch {
                header_n,
                expected_n,
            } => write!(
                f,
                "vorbis audio packet (write): header.n={header_n} disagrees with §4.3.1 step 3 blocksize selection (expected {expected_n})"
            ),
            WriteAudioPacketHeaderError::ShortBlockHasWindowFlag {
                previous_window_flag,
                next_window_flag,
            } => write!(
                f,
                "vorbis audio packet (write): short block carries previous_window_flag={previous_window_flag} / next_window_flag={next_window_flag}; §4.3.1 step 4b transmits no window flags on a short block"
            ),
        }
    }
}

impl std::error::Error for WriteAudioPacketHeaderError {}

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
            WriteError::Floor0(e) => write!(f, "{e}"),
            WriteError::Residue(e) => write!(f, "{e}"),
            WriteError::Mapping(e) => write!(f, "{e}"),
            WriteError::Mode(e) => write!(f, "{e}"),
            WriteError::Setup(e) => write!(f, "{e}"),
            WriteError::AudioPacket(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for WriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WriteError::Codebook(e) => Some(e),
            WriteError::Floor1(e) => Some(e),
            WriteError::Floor0(e) => Some(e),
            WriteError::Residue(e) => Some(e),
            WriteError::Mapping(e) => Some(e),
            WriteError::Mode(e) => Some(e),
            WriteError::Setup(e) => Some(e),
            WriteError::AudioPacket(e) => Some(e),
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

impl From<WriteFloor0Error> for WriteError {
    fn from(value: WriteFloor0Error) -> Self {
        WriteError::Floor0(value)
    }
}

impl From<WriteResidueError> for WriteError {
    fn from(value: WriteResidueError) -> Self {
        WriteError::Residue(value)
    }
}

impl From<WriteMappingError> for WriteError {
    fn from(value: WriteMappingError) -> Self {
        WriteError::Mapping(value)
    }
}

impl From<WriteModeError> for WriteError {
    fn from(value: WriteModeError) -> Self {
        WriteError::Mode(value)
    }
}

impl From<WriteSetupError> for WriteError {
    fn from(value: WriteSetupError) -> Self {
        WriteError::Setup(value)
    }
}

impl From<WriteAudioPacketHeaderError> for WriteError {
    fn from(value: WriteAudioPacketHeaderError) -> Self {
        WriteError::AudioPacket(value)
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

/// Serialises a [`Floor0Header`] to the §6.2.1 floor-type-0
/// setup-header bit pattern.
///
/// §6.2.1 fields (each value is unsigned, written LSB-first into the
/// bit stream per §2.1.4):
///
/// | Step | Field                       | Width  | Notes                                |
/// | ---: | --------------------------- | -----: | ------------------------------------ |
/// |    1 | `floor0_order`              |   8 b  | Raw `u8`.                            |
/// |    2 | `floor0_rate`               |  16 b  | Raw `u16`.                           |
/// |    3 | `floor0_bark_map_size`      |  16 b  | Raw `u16`.                           |
/// |    4 | `floor0_amplitude_bits`     |   6 b  | Range `0..=63`.                      |
/// |    5 | `floor0_amplitude_offset`   |   8 b  | Raw `u8`.                            |
/// |    6 | `floor0_number_of_books - 1`|   4 b  | `book_list.len()` in `1..=16`.       |
/// |    7 | `floor0_book_list[i]`       |   8 b  | One byte per entry.                  |
///
/// The header is emitted **without** the outer 16-bit `floor_type`
/// selector — that field is the setup-header walker's responsibility
/// (mirroring the [`write_floor1_header`] convention). The companion
/// [`write_floor0_header_into_writer`] splice point is shaped to slot
/// into the setup-header writer when that lands in a later round.
///
/// **Return value** — the produced bitstream as a [`Vec<u8>`]. The
/// final byte may carry up to 7 bits of zero padding to byte-align
/// the slice; the parser stops after the last book-list entry per
/// §6.2.1 step 7.
///
/// Returns [`WriteFloor0Error`] without emitting any bits if the
/// supplied header violates a §6.2.1 invariant.
pub fn write_floor0_header(header: &Floor0Header) -> Result<Vec<u8>, WriteFloor0Error> {
    let mut writer = BitWriterLsb::with_capacity(16);
    write_floor0_header_into_writer(header, &mut writer)?;
    Ok(writer.finish())
}

/// Bit-level helper to splice a floor 0 header into a larger
/// bit-packed stream. The setup-header writer will use this in a
/// later round, mirroring [`write_floor1_header_into_writer`] /
/// [`write_codebook_into_writer`].
///
/// Writes the header's bits into `writer` at its current bit
/// position. On error, the writer has had no bits appended (we
/// validate before emitting).
pub(crate) fn write_floor0_header_into_writer(
    header: &Floor0Header,
    writer: &mut BitWriterLsb,
) -> Result<(), WriteFloor0Error> {
    // ---- §6.2.1 invariant gate. ----
    // Fail-closed: refuse to emit a single bit if any field cannot be
    // serialised back to a packet the parser would accept.

    // step 4: amplitude_bits fits in a 6-bit field.
    if header.amplitude_bits > 63 {
        return Err(WriteFloor0Error::AmplitudeBitsOverflow(
            header.amplitude_bits,
        ));
    }
    // step 6: book_list.len() in 1..=16 (encoded as len - 1 in 4 bits).
    if header.book_list.is_empty() {
        return Err(WriteFloor0Error::EmptyBookList);
    }
    if header.book_list.len() > 16 {
        return Err(WriteFloor0Error::BookListTooLong(header.book_list.len()));
    }
    // Note: `order`, `rate`, `bark_map_size`, `amplitude_offset` are
    // raw u8/u16 values — every value of those types fits its field
    // by construction (8/16/16/8 bits), so no further bound checks
    // are required.

    // ---- §6.2.1 emit. ----
    // step 1.
    writer.write_u32(header.order as u32, 8);
    // step 2.
    writer.write_u32(header.rate as u32, 16);
    // step 3.
    writer.write_u32(header.bark_map_size as u32, 16);
    // step 4.
    writer.write_u32(header.amplitude_bits as u32, 6);
    // step 5.
    writer.write_u32(header.amplitude_offset as u32, 8);
    // step 6: number_of_books - 1.
    writer.write_u32((header.book_list.len() - 1) as u32, 4);
    // step 7.
    for &book in &header.book_list {
        writer.write_u32(book as u32, 8);
    }
    Ok(())
}

/// Serialises a [`ResidueHeader`] to the §8.6.1 residue-header bit
/// pattern.
///
/// §8.6.1 fields (each value is unsigned, written LSB-first into the
/// bit stream per §2.1.4):
///
/// | Step | Field                              | Width  | Notes                                            |
/// | ---: | ---------------------------------- | -----: | ------------------------------------------------ |
/// |    1 | `residue_begin`                    |  24 b  | Raw `u32` in `0..=0xFFFFFF`.                     |
/// |    2 | `residue_end`                      |  24 b  | Raw `u32` in `0..=0xFFFFFF`.                     |
/// |    3 | `residue_partition_size - 1`       |  24 b  | `partition_size` in `1..=2^24`.                  |
/// |    4 | `residue_classifications - 1`      |   6 b  | `classifications` in `1..=64`.                   |
/// |    5 | `residue_classbook`                |   8 b  | Raw `u8` index into the codebook table.          |
/// |  6.a | per-classification: `low_bits`     |   3 b  | `cascade[i] & 0x07`.                             |
/// |  6.b | per-classification: `bitflag`      |   1 b  | `1` iff `cascade[i] >> 3 != 0`, else `0`.        |
/// |  6.c | per-classification: `high_bits`    |   5 b  | Only when `bitflag == 1`; `cascade[i] >> 3`.     |
/// |    7 | per-classification × stage 0..=7   |   8 b  | `books[i][j]` iff `cascade[i]` bit `j` is set.   |
///
/// The header is emitted **without** the outer 16-bit `residue_type`
/// selector — that field is the setup-header walker's responsibility
/// (mirroring the [`write_floor1_header`] / [`write_floor0_header`]
/// convention). The companion [`write_residue_header_into_writer`]
/// splice point is shaped to slot into the setup-header writer when
/// that lands in a later round.
///
/// **Return value** — the produced bitstream as a [`Vec<u8>`]. The
/// final byte may carry up to 7 bits of zero padding to byte-align
/// the slice; the parser stops after the last `residue_books[i][7]`
/// per §8.6.1.
///
/// Returns [`WriteResidueError`] without emitting any bits if the
/// supplied header violates a §8.6.1 invariant.
///
/// ## Spec source
///
/// `docs/audio/vorbis/Vorbis_I_spec.pdf` §8.6.1 ("Residue setup
/// header decode" — the residue-header field list common to all three
/// formats), §4.2.4 step 2c (the `residue_type ∈ {0, 1, 2}`
/// constraint), and §2.1.4 (LSB-first packing).
pub fn write_residue_header(header: &ResidueHeader) -> Result<Vec<u8>, WriteResidueError> {
    let mut writer = BitWriterLsb::with_capacity(16);
    write_residue_header_into_writer(header, &mut writer)?;
    Ok(writer.finish())
}

/// Bit-level helper to splice a residue header into a larger
/// bit-packed stream. The setup-header writer will use this in a
/// later round, mirroring
/// [`write_floor0_header_into_writer`] / [`write_floor1_header_into_writer`]
/// / [`write_codebook_into_writer`].
///
/// Writes the header's bits into `writer` at its current bit
/// position. On error, the writer has had no bits appended (we
/// validate before emitting).
pub(crate) fn write_residue_header_into_writer(
    header: &ResidueHeader,
    writer: &mut BitWriterLsb,
) -> Result<(), WriteResidueError> {
    // ---- §8.6.1 invariant gate. ----
    // Fail-closed: refuse to emit a single bit if any field cannot be
    // serialised back to a packet the parser would accept.

    // §4.2.4 step 2c gate (mirrors the setup-walker rejection).
    if header.residue_type > 2 {
        return Err(WriteResidueError::UnsupportedResidueType(
            header.residue_type,
        ));
    }
    // step 1: residue_begin fits in a 24-bit field.
    if header.residue_begin > 0x00FF_FFFF {
        return Err(WriteResidueError::ResidueBeginOverflow(
            header.residue_begin,
        ));
    }
    // step 2: residue_end fits in a 24-bit field.
    if header.residue_end > 0x00FF_FFFF {
        return Err(WriteResidueError::ResidueEndOverflow(header.residue_end));
    }
    // step 3: partition_size in 1..=2^24 (stored as `read 24 bits + 1`).
    if header.partition_size == 0 || header.partition_size > (1u32 << 24) {
        return Err(WriteResidueError::PartitionSizeOutOfRange(
            header.partition_size,
        ));
    }
    // step 4: classifications in 1..=64 (stored as `read 6 bits + 1`).
    if header.classifications == 0 || header.classifications > 64 {
        return Err(WriteResidueError::ClassificationsOutOfRange(
            header.classifications,
        ));
    }
    // The cascade and books tables must each carry exactly one entry
    // per classification per the §8.6.1 outer loops.
    if header.cascade.len() != header.classifications as usize {
        return Err(WriteResidueError::CascadeLengthMismatch {
            classifications: header.classifications,
            actual: header.cascade.len(),
        });
    }
    if header.books.len() != header.classifications as usize {
        return Err(WriteResidueError::BooksLengthMismatch {
            classifications: header.classifications,
            actual: header.books.len(),
        });
    }
    // Each `books[class][stage]` slot must match the cascade bit
    // (Some(_) iff cascade bit set).
    for (class_idx, (&cas, row)) in header.cascade.iter().zip(header.books.iter()).enumerate() {
        for (stage, slot) in row.iter().enumerate() {
            let cascade_bit_set = (cas >> stage) & 1 == 1;
            match (slot, cascade_bit_set) {
                (Some(_), false) => {
                    return Err(WriteResidueError::BooksCascadeMismatch {
                        class: class_idx,
                        stage,
                        book_present: true,
                    });
                }
                (None, true) => {
                    return Err(WriteResidueError::BooksCascadeMismatch {
                        class: class_idx,
                        stage,
                        book_present: false,
                    });
                }
                _ => {}
            }
        }
    }
    // Note: `residue_classbook` is a raw `u8` whose 8-bit field width
    // accepts every u8 value; no further bound check is required.

    // ---- §8.6.1 emit. ----
    // step 1.
    writer.write_u32(header.residue_begin, 24);
    // step 2.
    writer.write_u32(header.residue_end, 24);
    // step 3.
    writer.write_u32(header.partition_size - 1, 24);
    // step 4.
    writer.write_u32((header.classifications - 1) as u32, 6);
    // step 5.
    writer.write_u32(header.classbook as u32, 8);
    // step 6: cascade — split each byte into 3 low bits + bitflag +
    // optional 5 high bits. The parser computes
    // `cascade[i] = high_bits * 8 + low_bits`; the inverse is
    // `low_bits = byte & 7`, `high_bits = byte >> 3`. We elide the
    // 5-bit high read iff `high_bits == 0`, which matches the parser's
    // `bitflag` branch.
    for &cas in &header.cascade {
        let low_bits = (cas & 0x07) as u32;
        let high_bits = (cas >> 3) as u32;
        writer.write_u32(low_bits, 3);
        if high_bits == 0 {
            writer.write_u32(0, 1);
        } else {
            writer.write_u32(1, 1);
            writer.write_u32(high_bits, 5);
        }
    }
    // step 7: per-class × per-stage: write 8-bit book iff cascade bit
    // set. The invariant gate above guarantees Some(_) ↔ cascade-set.
    for (cas, row) in header.cascade.iter().zip(header.books.iter()) {
        for (stage, slot) in row.iter().enumerate() {
            if (cas >> stage) & 1 == 1 {
                let book = slot.expect("invariant-gate-validated");
                writer.write_u32(book as u32, 8);
            }
        }
    }
    Ok(())
}

/// Serialises a [`MappingHeader`] to the §4.2.4 "Mappings" body bit
/// pattern.
///
/// The writer is the bit-exact inverse of the round-5 mapping parser
/// (`setup::parse_mapping_header`). Given the same context tuple the
/// parser is supplied with —
/// `(mapping_type, audio_channels, floor_count, residue_count)` — the
/// round-trip property
///
/// ```text
/// local_parse_mapping_for_tests(
///     &mut BitReaderLsb::new(
///         &write_mapping_header(&h, audio_channels, floor_count, residue_count)?,
///     ),
///     h.mapping_type, audio_channels,
/// ) == h
/// ```
///
/// holds for every legal [`MappingHeader`].
///
/// ## §4.2.4 emit order
///
/// | Step | Field                                       | Width | Notes |
/// | ---: | :------------------------------------------ | :---: | :---- |
/// |   2a | `mapping_type`                              |  16 b | Must be `0`. |
/// |  2c.i.a | `submaps_flag`                           |   1 b | `0` when `submaps == 1`, else `1`. |
/// |  2c.i.b | `submaps - 1` (only when flag set)       |   4 b | `submaps` is `1..=16`. |
/// |  2c.ii.a | `square_polar_flag`                     |   1 b | `0` when `coupling` is empty, else `1`. |
/// |  2c.ii.b | `coupling_steps - 1` (only when flag set) |   8 b | `coupling_steps` is `1..=256`. |
/// |  2c.ii.A | per-step magnitude / angle              | `2 × ilog(audio_channels - 1)` b | Each pair, in order. |
/// |  2c.iii | reserved                                  |   2 b | Always `0`. |
/// |  2c.iv | per-channel `mux[ch]` (only when `submaps > 1`) | 4 b each | One per `audio_channels`. |
/// |  2c.v.A | per-submap `time_placeholder`            |   8 b | Verbatim from the header. |
/// |  2c.v.B | per-submap `floor`                        |   8 b | Range-checked against `floor_count`. |
/// |  2c.v.C | per-submap `residue`                      |   8 b | Range-checked against `residue_count`. |
///
/// The §4.2.4 "Mappings" body is emitted **without** the surrounding
/// `vorbis_mapping_count - 1` 6-bit field (that is the setup-header
/// walker's responsibility). The companion
/// [`write_mapping_header_into_writer`] splice point is shaped to slot
/// into the setup-header writer when that lands, matching the existing
/// `write_codebook_into_writer` / `write_floor1_header_into_writer` /
/// `write_floor0_header_into_writer` / `write_residue_header_into_writer`
/// splice points.
///
/// ## Encoding-form selection
///
/// §4.2.4 step 2c.i and step 2c.ii each gate an optional body behind a
/// 1-bit flag. The writer always selects the densest legal encoding:
///
/// * When `submaps == 1` the `submaps_flag` is emitted as `0` and the
///   4-bit `submaps - 1` body is elided — exactly the form the parser
///   defaults to when the flag is unset.
/// * When `coupling.is_empty()` the `square_polar_flag` is emitted as
///   `0` and the 8-bit coupling-step body + per-step channel-number
///   bodies are elided — exactly the form the parser defaults to when
///   the flag is unset.
///
/// `audio_channels` / `floor_count` / `residue_count` are the context
/// values the §4.2.4 walker sources from the identification header
/// (channels) and the in-progress setup-walker state (floor and
/// residue counts). They are used:
///
/// * `audio_channels` — to pick the per-coupling-step channel-number
///   width via `ilog(audio_channels - 1)` (§4.2.4 step 2c.ii.A) and to
///   validate the §4.2.4 step 2c.ii "magnitude != angle, both <
///   audio_channels" constraint;
/// * `audio_channels` — to validate the per-channel `mux[ch]` slot
///   count (§4.2.4 step 2c.iv);
/// * `floor_count` / `residue_count` — to validate the per-submap
///   `floor` / `residue` indices (§4.2.4 step 2c.v.B / step 2c.v.C).
///
/// **Return value** — the produced bitstream as a [`Vec<u8>`]. The
/// final byte may carry up to 7 bits of zero padding to byte-align
/// the slice; the parser stops after the last per-submap residue
/// index per §4.2.4 step 2c.v.
///
/// Returns [`WriteMappingError`] without emitting any bits if the
/// supplied header (or the context) violates a §4.2.4 invariant.
///
/// ## Spec source
///
/// `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.2.4 "Mappings" (the
/// mapping-body step list including the optional `submaps_flag` and
/// `square_polar_flag` gates, the per-coupling-step magnitude/angle
/// "magnitude != angle" rule, the 2-bit reserved field, the per-channel
/// `mux[ch]` slot, and the per-submap `(time_placeholder, floor,
/// residue)` triples), §9.2.1 "ilog" (the per-channel-number field
/// width), and §2.1.4 (LSB-first packing).
pub fn write_mapping_header(
    header: &MappingHeader,
    audio_channels: u8,
    floor_count: usize,
    residue_count: usize,
) -> Result<Vec<u8>, WriteMappingError> {
    let mut writer = BitWriterLsb::with_capacity(16);
    write_mapping_header_into_writer(
        header,
        audio_channels,
        floor_count,
        residue_count,
        &mut writer,
    )?;
    Ok(writer.finish())
}

/// Bit-level helper to splice a mapping body into a larger bit-packed
/// stream. The setup-header writer will use this in a later round,
/// mirroring [`write_codebook_into_writer`] /
/// [`write_floor1_header_into_writer`] /
/// [`write_floor0_header_into_writer`] /
/// [`write_residue_header_into_writer`].
///
/// Writes the body's bits into `writer` at its current bit position.
/// On error, the writer has had no bits appended (we validate before
/// emitting).
pub(crate) fn write_mapping_header_into_writer(
    header: &MappingHeader,
    audio_channels: u8,
    floor_count: usize,
    residue_count: usize,
    writer: &mut BitWriterLsb,
) -> Result<(), WriteMappingError> {
    // ---- §4.2.4 invariant gate. ----
    // Fail-closed: refuse to emit a single bit if any field cannot be
    // serialised back to a body the parser would accept.

    // step 2b: mapping_type must be 0 (only type defined in Vorbis I).
    if header.mapping_type != 0 {
        return Err(WriteMappingError::UnsupportedMappingType(
            header.mapping_type,
        ));
    }
    // The §4.2.4 mapping algorithm assumes audio_channels > 0; the
    // identification-header parser enforces this on the way in.
    if audio_channels == 0 {
        return Err(WriteMappingError::ZeroAudioChannels);
    }
    // step 2c.i: submaps in 1..=16 (stored as `read 4 bits + 1`).
    if header.submaps == 0 || header.submaps > 16 {
        return Err(WriteMappingError::SubmapsOutOfRange(header.submaps));
    }
    // step 2c.ii: coupling.len() in 0..=256 (0 = "no square-polar
    // body emitted", 1..=256 = `read 8 bits + 1`).
    if header.coupling.len() > 256 {
        return Err(WriteMappingError::CouplingStepsOverflow(
            header.coupling.len(),
        ));
    }
    // step 2c.ii inner: per-step magnitude != angle, both <
    // audio_channels; also bound by the per-step channel-number
    // field width.
    let channel_bits = ilog((audio_channels as u32).saturating_sub(1));
    let channel_field_cap: u32 = if channel_bits >= 32 {
        u32::MAX
    } else {
        (1u32 << channel_bits).saturating_sub(1)
    };
    for (step_index, step) in header.coupling.iter().enumerate() {
        if step.magnitude_channel == step.angle_channel
            || step.magnitude_channel >= audio_channels
            || step.angle_channel >= audio_channels
        {
            return Err(WriteMappingError::BadCouplingChannels {
                step_index,
                magnitude_channel: step.magnitude_channel,
                angle_channel: step.angle_channel,
                audio_channels,
            });
        }
        // Defensive: channel_bits is derived from audio_channels, so a
        // value `< audio_channels` always fits. Verify all the same.
        if (step.magnitude_channel as u32) > channel_field_cap {
            return Err(WriteMappingError::CouplingChannelOverflow {
                step_index,
                channel: step.magnitude_channel,
                channel_bits,
            });
        }
        if (step.angle_channel as u32) > channel_field_cap {
            return Err(WriteMappingError::CouplingChannelOverflow {
                step_index,
                channel: step.angle_channel,
                channel_bits,
            });
        }
    }
    // step 2c.iv: mux is present iff submaps > 1; when present it has
    // exactly audio_channels entries, each strictly less than submaps.
    let expected_mux_len = if header.submaps > 1 {
        audio_channels as usize
    } else {
        0
    };
    if header.mux.len() != expected_mux_len {
        return Err(WriteMappingError::MuxLengthMismatch {
            expected: expected_mux_len,
            actual: header.mux.len(),
        });
    }
    for (channel_index, &mux) in header.mux.iter().enumerate() {
        if mux >= header.submaps {
            return Err(WriteMappingError::BadMuxValue {
                channel_index,
                mux,
                submaps: header.submaps,
            });
        }
    }
    // step 2c.v: exactly one (time_placeholder, floor, residue) triple
    // per submap; floor / residue are range-checked against the
    // walker-supplied counts.
    if header.submap_configs.len() != header.submaps as usize {
        return Err(WriteMappingError::SubmapCountMismatch {
            submaps: header.submaps,
            actual: header.submap_configs.len(),
        });
    }
    for (submap_index, submap) in header.submap_configs.iter().enumerate() {
        if (submap.floor as usize) >= floor_count {
            return Err(WriteMappingError::BadSubmapFloor {
                submap_index,
                floor: submap.floor,
                floor_count,
            });
        }
        if (submap.residue as usize) >= residue_count {
            return Err(WriteMappingError::BadSubmapResidue {
                submap_index,
                residue: submap.residue,
                residue_count,
            });
        }
    }

    // ---- §4.2.4 emit. ----
    // step 2a: 16-bit mapping_type (always 0 here, by the gate above).
    writer.write_u32(header.mapping_type as u32, 16);

    // step 2c.i: optional submaps_flag + 4-bit body. Pin the densest
    // encoding: submaps_flag = 0 when submaps == 1.
    if header.submaps == 1 {
        writer.write_u32(0, 1);
    } else {
        writer.write_u32(1, 1);
        writer.write_u32((header.submaps - 1) as u32, 4);
    }

    // step 2c.ii: optional square_polar_flag + 8-bit body + per-step
    // magnitude/angle channel numbers at `channel_bits` width each.
    if header.coupling.is_empty() {
        writer.write_u32(0, 1);
    } else {
        writer.write_u32(1, 1);
        // `coupling.len()` is in 1..=256 (gated above), so `len() - 1`
        // fits in 8 bits.
        writer.write_u32((header.coupling.len() - 1) as u32, 8);
        for step in &header.coupling {
            writer.write_u32(step.magnitude_channel as u32, channel_bits);
            writer.write_u32(step.angle_channel as u32, channel_bits);
        }
    }

    // step 2c.iii: 2-bit reserved, must be 0.
    writer.write_u32(0, 2);

    // step 2c.iv: per-channel mux (only present when submaps > 1, by
    // the gate above).
    for &mux in &header.mux {
        writer.write_u32(mux as u32, 4);
    }

    // step 2c.v: per-submap (time_placeholder, floor, residue).
    for submap in &header.submap_configs {
        writer.write_u32(submap.time_placeholder as u32, 8);
        writer.write_u32(submap.floor as u32, 8);
        writer.write_u32(submap.residue as u32, 8);
    }

    Ok(())
}

/// Serialises a [`ModeHeader`] to the §4.2.4 "Modes" body bit pattern.
///
/// The §4.2.4 mode body is a single fixed-width 41-bit record:
///
/// | Step | Field             | Width | Notes                                                 |
/// | ---: | ----------------- | ----: | ----------------------------------------------------- |
/// |   2a | `blockflag`       |   1 b | `false` selects `blocksize_0`, `true` selects `blocksize_1`. |
/// |   2b | `windowtype`      |  16 b | Vorbis I §4.2.4 step 2e: only `0` is legal.          |
/// |   2c | `transformtype`   |  16 b | Vorbis I §4.2.4 step 2e: only `0` is legal.          |
/// |   2d | `mapping`         |   8 b | Range-checked against `mapping_count`.               |
///
/// The 41-bit body is emitted **without** the surrounding
/// `vorbis_mode_count - 1` 6-bit count field and **without** the
/// trailing 1-bit framing flag (both are the setup-header walker's
/// responsibility). The companion [`write_mode_header_into_writer`]
/// splice point is shaped to slot into the setup-header writer when
/// that lands, matching the existing
/// `write_codebook_into_writer` / `write_floor1_header_into_writer` /
/// `write_floor0_header_into_writer` / `write_residue_header_into_writer` /
/// `write_mapping_header_into_writer` splice points.
///
/// `mapping_count` is the context value the §4.2.4 walker sources
/// from the in-progress setup-walker state (i.e. the length of the
/// mapping list parsed earlier in the setup body). It is used to
/// validate the §4.2.4 step 2e "vorbis_mode_mapping must not be
/// greater than the highest number mapping in use" constraint.
///
/// **Return value** — the produced bitstream as a [`Vec<u8>`]. The
/// final byte carries 7 bits of zero padding to byte-align the
/// 41-bit body; the parser stops after the 8-bit mapping index per
/// §4.2.4 step 2d.
///
/// Returns [`WriteModeError`] without emitting any bits if the
/// supplied header (or the context) violates a §4.2.4 invariant.
///
/// ## Bit-exact roundtrip guarantee
///
/// For every value `h: ModeHeader` and matching context
/// `mapping_count` for which the §4.2.4 invariants hold:
///
/// ```text
/// local_parse_mode_for_tests(
///     &mut BitReaderLsb::new(&write_mode_header(&h, mapping_count)?),
///     mapping_count,
/// ) == h
/// ```
///
/// ## Spec source
///
/// `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.2.4 "Modes" (the four-step
/// fixed-width body — `blockflag`, `windowtype`, `transformtype`,
/// `mapping`), with the "zero is the only legal value" rule on the
/// 16-bit window and transform fields and the
/// `mapping < vorbis_mapping_count` constraint at step 2e; §2.1.4
/// (LSB-first packing).
pub fn write_mode_header(
    header: &ModeHeader,
    mapping_count: usize,
) -> Result<Vec<u8>, WriteModeError> {
    let mut writer = BitWriterLsb::with_capacity(8);
    write_mode_header_into_writer(header, mapping_count, &mut writer)?;
    Ok(writer.finish())
}

/// Bit-level helper to splice a mode body into a larger bit-packed
/// stream. The setup-header writer will use this in a later round,
/// mirroring [`write_codebook_into_writer`] /
/// [`write_floor1_header_into_writer`] /
/// [`write_floor0_header_into_writer`] /
/// [`write_residue_header_into_writer`] /
/// [`write_mapping_header_into_writer`].
///
/// Writes the body's bits into `writer` at its current bit position.
/// On error, the writer has had no bits appended (we validate before
/// emitting).
pub(crate) fn write_mode_header_into_writer(
    header: &ModeHeader,
    mapping_count: usize,
    writer: &mut BitWriterLsb,
) -> Result<(), WriteModeError> {
    // ---- §4.2.4 invariant gate. ----
    // Fail-closed: refuse to emit a single bit if any field cannot be
    // serialised back to a body the parser would accept.

    // step 2e: windowtype must be 0.
    if header.windowtype != 0 {
        return Err(WriteModeError::NonZeroWindowType(header.windowtype));
    }
    // step 2e: transformtype must be 0.
    if header.transformtype != 0 {
        return Err(WriteModeError::NonZeroTransformType(header.transformtype));
    }
    // step 2e: mapping < mapping_count.
    if (header.mapping as usize) >= mapping_count {
        return Err(WriteModeError::BadMapping {
            mapping: header.mapping,
            mapping_count,
        });
    }

    // ---- §4.2.4 emit. ----
    // step 2a: 1-bit blockflag.
    writer.write_bit(header.blockflag);
    // step 2b: 16-bit windowtype (always 0, by the gate above).
    writer.write_u32(header.windowtype as u32, 16);
    // step 2c: 16-bit transformtype (always 0, by the gate above).
    writer.write_u32(header.transformtype as u32, 16);
    // step 2d: 8-bit mapping (range-checked above).
    writer.write_u32(header.mapping as u32, 8);

    Ok(())
}

/// Serialises a [`VorbisSetupHeader`] to the full §4.2.4 packet shape
/// — the third Vorbis I header (after identification and comment).
///
/// This is the wrapping setup-header WRITE primitive that stitches the
/// six nested-block writers (the crate-private `write_codebook_into_writer`,
/// `write_floor0_header_into_writer`, `write_floor1_header_into_writer`,
/// `write_residue_header_into_writer`, `write_mapping_header_into_writer`,
/// and `write_mode_header_into_writer` splice points exposed alongside the
/// public per-block writers [`write_codebook`], [`write_floor0_header`],
/// [`write_floor1_header`], [`write_residue_header`], [`write_mapping_header`],
/// [`write_mode_header`]) into a single byte-aligned packet matching the
/// round-5 [`crate::setup::parse_setup_header`] reader.
///
/// Layout (per `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.2.1 + §4.2.4):
///
/// ```text
/// 0x05                                  # packet_type
/// "vorbis"                              # 6-byte magic
/// vorbis_codebook_count - 1   : 8 bits  # then codebook_count codebook bodies (§3.2.1)
/// vorbis_time_count - 1       : 6 bits  # then time_count * 16-bit zero placeholders
/// vorbis_floor_count - 1      : 6 bits  # then floor_count * (16-bit floor_type + body)
/// vorbis_residue_count - 1    : 6 bits  # then residue_count * (16-bit residue_type + body)
/// vorbis_mapping_count - 1    : 6 bits  # then mapping_count mapping bodies (§4.2.4 "Mappings")
/// vorbis_mode_count - 1       : 6 bits  # then mode_count * 41-bit mode bodies (§4.2.4 "Modes")
/// framing_flag                : 1 bit   # § 4.2.4 step 3: must be 1
/// # final byte may carry up to 7 bits of §2.1.8 zero padding to byte-align.
/// ```
///
/// `audio_channels` must equal the `audio_channels` field of the
/// identification header parsed earlier in the same logical stream
/// (Vorbis I §4.2.2). It is needed for the
/// `ilog(audio_channels - 1)`-bit magnitude/angle channel reads in
/// §4.2.4 "Mappings" and for the per-channel `mux[ch]` reads when a
/// mapping declares `submaps > 1` — i.e. it is the context the nested
/// `write_mapping_header_into_writer` splice point already consumes.
///
/// The nested mapping writer consumes `(audio_channels, floor_count,
/// residue_count)` as its context tuple; the nested mode writer
/// consumes `mapping_count`. Both context values are sourced from the
/// in-progress setup-walker state (`header.floors.len()`,
/// `header.residues.len()`, `header.mappings.len()`), so the
/// setup-header writer is the single entry point that wires the
/// context up exactly as the §4.2.4 walker would.
///
/// **Return value** — the produced bitstream as a [`Vec<u8>`]. The
/// final byte may carry up to 7 bits of zero padding to byte-align the
/// slice; the parser stops after the 1-bit framing flag per §4.2.4
/// step 3.
///
/// Returns [`WriteError`] without emitting any bytes if any field
/// (or any nested block) fails a §4.2.4 invariant.
///
/// ## Bit-exact roundtrip guarantee
///
/// For every value `h: VorbisSetupHeader` and matching `audio_channels`
/// for which the §4.2.4 invariants hold:
///
/// ```text
/// parse_setup_header(
///     &write_setup_header(&h, audio_channels)?,
///     audio_channels,
/// ) == h
/// ```
///
/// ## Spec source
///
/// `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.2.1 (the 7-byte common
/// header — packet type `0x05` + ASCII `"vorbis"` magic), §4.2.4 (the
/// bit-packed body — codebook / time / floor / residue / mapping / mode
/// blocks plus the trailing framing flag); §2.1.4 (LSB-first packing);
/// §2.1.8 (end-of-packet alignment — the final byte's trailing bits are
/// zero padding).
pub fn write_setup_header(
    header: &VorbisSetupHeader,
    audio_channels: u8,
) -> Result<Vec<u8>, WriteError> {
    // ---- §4.2.4 invariant gate — fail-closed: refuse to emit any
    // bytes if any field fails its container width / range invariant.
    // The nested-block writers gate their own §-specific invariants
    // when invoked further down; this layer gates only the wrapping
    // structure (counts, type selectors, framing flag).

    if audio_channels == 0 {
        return Err(WriteSetupError::ZeroAudioChannels.into());
    }

    // §4.2.4 "Codebooks": 8-bit `read 8 bits + 1`, so range 1..=256.
    if header.codebooks.is_empty() {
        return Err(WriteSetupError::EmptyCodebooks.into());
    }
    if header.codebooks.len() > 256 {
        return Err(WriteSetupError::CodebookCountOverflow(header.codebooks.len()).into());
    }

    // §4.2.4 "Time domain transforms": 6-bit `read 6 bits + 1`, range
    // 1..=64. Every value must be zero.
    if header.time_placeholders.is_empty() {
        return Err(WriteSetupError::EmptyTimePlaceholders.into());
    }
    if header.time_placeholders.len() > 64 {
        return Err(WriteSetupError::TimeCountOverflow(header.time_placeholders.len()).into());
    }
    for (index, &value) in header.time_placeholders.iter().enumerate() {
        if value != 0 {
            return Err(WriteSetupError::NonZeroTimePlaceholder { index, value }.into());
        }
    }

    // §4.2.4 "Floors": 6-bit `read 6 bits + 1`, range 1..=64. Each
    // entry's `floor_type` ∈ {0, 1} and must agree with the kind
    // discriminant on the payload.
    if header.floors.is_empty() {
        return Err(WriteSetupError::EmptyFloors.into());
    }
    if header.floors.len() > 64 {
        return Err(WriteSetupError::FloorCountOverflow(header.floors.len()).into());
    }
    for (index, floor) in header.floors.iter().enumerate() {
        if floor.floor_type > 1 {
            return Err(WriteSetupError::UnsupportedFloorType {
                index,
                floor_type: floor.floor_type,
            }
            .into());
        }
        let kind_discriminant: u16 = match floor.kind {
            FloorKind::Type0(_) => 0,
            FloorKind::Type1(_) => 1,
        };
        if floor.floor_type != kind_discriminant {
            return Err(WriteSetupError::FloorTypeKindMismatch {
                index,
                floor_type: floor.floor_type,
                kind_discriminant,
            }
            .into());
        }
    }

    // §4.2.4 "Residues": 6-bit `read 6 bits + 1`, range 1..=64. The
    // nested writer's own gate validates `residue_type ∈ {0, 1, 2}`.
    if header.residues.is_empty() {
        return Err(WriteSetupError::EmptyResidues.into());
    }
    if header.residues.len() > 64 {
        return Err(WriteSetupError::ResidueCountOverflow(header.residues.len()).into());
    }

    // §4.2.4 "Mappings": 6-bit `read 6 bits + 1`, range 1..=64. The
    // nested writer's own gate validates the rest.
    if header.mappings.is_empty() {
        return Err(WriteSetupError::EmptyMappings.into());
    }
    if header.mappings.len() > 64 {
        return Err(WriteSetupError::MappingCountOverflow(header.mappings.len()).into());
    }

    // §4.2.4 "Modes": 6-bit `read 6 bits + 1`, range 1..=64. The
    // nested writer's own gate validates the rest.
    if header.modes.is_empty() {
        return Err(WriteSetupError::EmptyModes.into());
    }
    if header.modes.len() > 64 {
        return Err(WriteSetupError::ModeCountOverflow(header.modes.len()).into());
    }

    // §4.2.4 step 3: the trailing framing flag must be set.
    if !header.framing_flag {
        return Err(WriteSetupError::BadFramingFlag.into());
    }

    // ---- §4.2.4 emit. ----
    // Pre-allocate with an inexpensive lower bound — the codebook
    // and floor-1 bodies dwarf the wrapping structure so the bit
    // packer will grow regardless.
    let mut writer = BitWriterLsb::with_capacity(256);

    // §4.2.1: 7-byte common header (`packet_type` + "vorbis" magic).
    // The writer starts byte-aligned; emit each byte as an 8-bit field.
    writer.write_u32(SETUP_PACKET_TYPE as u32, 8);
    for &b in SETUP_PACKET_MAGIC.iter() {
        writer.write_u32(b as u32, 8);
    }

    // §4.2.4 "Codebooks": 8-bit count - 1, then each codebook body.
    writer.write_u32((header.codebooks.len() - 1) as u32, 8);
    for book in &header.codebooks {
        write_codebook_into_writer(book, &mut writer)?;
    }

    // §4.2.4 "Time domain transforms": 6-bit count - 1, then each
    // 16-bit zero placeholder.
    writer.write_u32((header.time_placeholders.len() - 1) as u32, 6);
    for &value in &header.time_placeholders {
        // The gate above pinned every value to 0; emit verbatim.
        writer.write_u32(value as u32, 16);
    }

    // §4.2.4 "Floors": 6-bit count - 1, then per floor:
    //   16-bit `floor_type` + per-type body.
    writer.write_u32((header.floors.len() - 1) as u32, 6);
    for floor in &header.floors {
        writer.write_u32(floor.floor_type as u32, 16);
        match &floor.kind {
            FloorKind::Type0(body) => write_floor0_header_into_writer(body, &mut writer)?,
            FloorKind::Type1(body) => write_floor1_header_into_writer(body, &mut writer)?,
        }
    }

    // §4.2.4 "Residues": 6-bit count - 1, then per residue:
    //   16-bit `residue_type` + body.
    writer.write_u32((header.residues.len() - 1) as u32, 6);
    for residue in &header.residues {
        writer.write_u32(residue.residue_type as u32, 16);
        write_residue_header_into_writer(residue, &mut writer)?;
    }

    // §4.2.4 "Mappings": 6-bit count - 1, then per mapping body. The
    // nested writer consumes (audio_channels, floor_count,
    // residue_count) as its context tuple.
    let floor_count = header.floors.len();
    let residue_count = header.residues.len();
    writer.write_u32((header.mappings.len() - 1) as u32, 6);
    for mapping in &header.mappings {
        write_mapping_header_into_writer(
            mapping,
            audio_channels,
            floor_count,
            residue_count,
            &mut writer,
        )?;
    }

    // §4.2.4 "Modes": 6-bit count - 1, then per mode body. The nested
    // writer consumes `mapping_count` as its context.
    let mapping_count = header.mappings.len();
    writer.write_u32((header.modes.len() - 1) as u32, 6);
    for mode in &header.modes {
        write_mode_header_into_writer(mode, mapping_count, &mut writer)?;
    }

    // §4.2.4 step 3: trailing 1-bit framing flag. The gate above pinned
    // this to `true`; emit verbatim.
    writer.write_bit(header.framing_flag);

    // §2.1.8: the final byte's trailing bits are zero padding so the
    // packet is delivered byte-aligned to the container layer.
    Ok(writer.finish())
}

/// Serialises a [`AudioPacketHeader`] to the §4.3.1 audio-packet prelude
/// bit pattern — the partial audio-packet WRITE primitive that emits the
/// prelude bits a §4.3 audio packet starts with.
///
/// This is the first audio-packet writer (after the three header-packet
/// writers and the six setup-header sub-block writers). The prelude is
/// the §4.3.1 step 1..4 region — `packet_type`, `mode_number`, and the
/// long-block `previous_window_flag` / `next_window_flag` — i.e. the
/// bits the parser side [`crate::packet::read_packet_header`] consumes. The audio
/// floor / residue / spectrum payload that follows is the subject of
/// further followup writers.
///
/// Layout (per `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.3.1):
///
/// ```text
/// packet_type           : 1 bit                              # step 1: must be 0
/// mode_number           : ilog([vorbis_mode_count] - 1) bits # step 2
/// # step 3: blocksize resolution is not on the wire; the parser
/// #         recomputes n = blocksize_0 or blocksize_1 from
/// #         setup.modes[mode_number].blockflag. The writer cross-
/// #         checks `header.n` against the spec-derived value.
/// # step 4: short block (blockflag == false) → no further bits.
/// #         long  block (blockflag == true ) → two more 1-bit reads:
/// previous_window_flag  : 1 bit   # step 4a.i  (long block only)
/// next_window_flag      : 1 bit   # step 4a.ii (long block only)
/// # final byte: §2.1.8 zero padding to byte-align the slice.
/// ```
///
/// `setup` supplies `setup.modes` — its length sizes the
/// `ilog([vorbis_mode_count] - 1)`-bit `mode_number` field per §4.3.1
/// step 2 (using [`ilog`] / [`crate::codebook::ilog`]) and the selected
/// mode's [`crate::setup::ModeHeader::blockflag`] resolves the
/// blocksize. The other setup-header lists (codebooks, floors,
/// residues, mappings) are ignored, matching the reader.
///
/// `blocksize_0` / `blocksize_1` come from the parsed identification
/// header (§4.2.2). They are not on the §4.3.1 wire — the writer uses
/// them only to cross-check `header.n` against the spec-derived
/// blocksize selection so a malformed `(blockflag, n)` cached pair
/// cannot silently be emitted.
///
/// **Return value** — the produced bitstream as a [`Vec<u8>`]. The
/// final byte may carry up to 7 bits of §2.1.8 zero padding to byte-
/// align the slice; the parser stops after the last consumed bit
/// (§4.3.1 step 4 closing position).
///
/// Returns [`WriteAudioPacketHeaderError`] without emitting any bytes
/// if any field fails a §4.3.1 invariant.
///
/// # Bit-exact roundtrip guarantee
///
/// For every value `h: AudioPacketHeader` and matching context
/// `(setup, blocksize_0, blocksize_1)` for which the §4.3.1 invariants
/// hold:
///
/// ```text
/// read_packet_header(
///     &mut BitReaderLsb::new(&write_audio_packet_header(&h, &setup, blocksize_0, blocksize_1)?),
///     &setup,
///     blocksize_0,
///     blocksize_1,
/// ) == Ok(h)
/// ```
///
/// # Spec source
///
/// `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.3.1 "packet type, mode and
/// window decode" (steps 1..4 — `packet_type`, `mode_number`, blocksize
/// resolution, and the long-block window flags); §4.2.4 "Modes" (the
/// mode list whose length sizes the step-2 read); §9.2.1 "ilog" (the
/// bit-width formula for the step-2 read width — single-mode degenerate
/// case `ilog(0) == 0` reads zero bits and resolves to mode 0
/// unconditionally); §2.1.4 "coding bits into byte sequences" (LSB-
/// first packing); §2.1.8 "end-of-packet alignment" (final byte zero
/// padding).
///
/// # Errors
///
/// [`WriteAudioPacketHeaderError::EmptyModeList`] if `setup.modes` is
/// empty. [`WriteAudioPacketHeaderError::BadModeNumber`] if
/// `header.mode_number >= setup.modes.len()`.
/// [`WriteAudioPacketHeaderError::BlockflagMismatch`] if
/// `header.blockflag` disagrees with the selected mode's
/// `blockflag`. [`WriteAudioPacketHeaderError::BlocksizeMismatch`] if
/// `header.n` disagrees with the §4.3.1 step-3 blocksize selection.
/// [`WriteAudioPacketHeaderError::ShortBlockHasWindowFlag`] if
/// `header.blockflag` is `false` but a window flag is set (no bit
/// pattern would round-trip to that struct on the short-block path).
pub fn write_audio_packet_header(
    header: &AudioPacketHeader,
    setup: &VorbisSetupHeader,
    blocksize_0: usize,
    blocksize_1: usize,
) -> Result<Vec<u8>, WriteAudioPacketHeaderError> {
    let mut writer = BitWriterLsb::with_capacity(1);
    write_audio_packet_header_into_writer(header, setup, blocksize_0, blocksize_1, &mut writer)?;
    Ok(writer.finish())
}

/// Bit-level helper to splice an audio-packet prelude into a larger
/// bit-packed stream. A wrapping audio-packet writer (the §4.3 packet
/// builder, an explicit followup) will use this to thread the prelude
/// into the per-channel floor / residue / spectrum payload, mirroring
/// the existing `write_codebook_into_writer` /
/// `write_floor0_header_into_writer` / `write_floor1_header_into_writer` /
/// `write_residue_header_into_writer` / `write_mapping_header_into_writer` /
/// `write_mode_header_into_writer` splice points.
///
/// Writes the prelude's bits into `writer` at its current bit position.
/// On error, the writer has had no bits appended (we validate before
/// emitting).
pub(crate) fn write_audio_packet_header_into_writer(
    header: &AudioPacketHeader,
    setup: &VorbisSetupHeader,
    blocksize_0: usize,
    blocksize_1: usize,
    writer: &mut BitWriterLsb,
) -> Result<(), WriteAudioPacketHeaderError> {
    // ---- §4.3.1 invariant gate. ----
    // Fail-closed: refuse to emit a single bit if any field cannot be
    // serialised back to a prelude the parser would accept.

    if setup.modes.is_empty() {
        return Err(WriteAudioPacketHeaderError::EmptyModeList);
    }
    let mode_count = setup.modes.len();
    if (header.mode_number as usize) >= mode_count {
        return Err(WriteAudioPacketHeaderError::BadModeNumber {
            mode_number: header.mode_number,
            mode_count,
        });
    }

    // §4.3.1 step 3: blockflag is sourced from the selected mode; the
    // parser does not transmit `blockflag` on the wire but caches the
    // mode's value into `AudioPacketHeader`. Cross-check rather than
    // silently emit the mode's value when the caller's struct disagrees.
    let mode = setup.modes[header.mode_number as usize];
    let mode_blockflag = mode.blockflag;
    if header.blockflag != mode_blockflag {
        return Err(WriteAudioPacketHeaderError::BlockflagMismatch {
            header_blockflag: header.blockflag,
            mode_blockflag,
        });
    }

    // §4.3.1 step 3: n = blocksize_0 (short) or blocksize_1 (long).
    let expected_n = if mode_blockflag {
        blocksize_1
    } else {
        blocksize_0
    };
    if header.n != expected_n {
        return Err(WriteAudioPacketHeaderError::BlocksizeMismatch {
            header_n: header.n,
            expected_n,
        });
    }

    // §4.3.1 step 4b: short block does not transmit window flags. The
    // parser returns (false, false) placeholders on the short-block
    // path; refuse a struct whose flags would silently disappear in
    // the round-trip.
    if !mode_blockflag && (header.previous_window_flag || header.next_window_flag) {
        return Err(WriteAudioPacketHeaderError::ShortBlockHasWindowFlag {
            previous_window_flag: header.previous_window_flag,
            next_window_flag: header.next_window_flag,
        });
    }

    // ---- §4.3.1 emit. ----
    // step 1: 1-bit packet_type = 0 (the §4.3 "is this an audio packet?"
    // discriminant; non-audio packets are rejected by the parser, so the
    // writer always emits zero).
    writer.write_bit(false);
    // step 2: ilog([vorbis_mode_count] - 1) bits, LSB-first. §9.2.1
    // `ilog(0) == 0` collapses to a zero-bit read in the single-mode
    // degenerate case — write_u32(v, 0) emits nothing per the bit
    // packer's contract, matching the reader.
    let mode_bits = ilog((mode_count as u32).saturating_sub(1));
    writer.write_u32(header.mode_number, mode_bits);
    // step 4: long block adds two 1-bit fields; short block emits no
    // further bits.
    if mode_blockflag {
        writer.write_bit(header.previous_window_flag);
        writer.write_bit(header.next_window_flag);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comment::parse_comment_header;
    use crate::identification::parse_identification_header;
    use crate::setup::{Floor1Class, MappingCouplingStep};

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

    // ================================================================
    // Floor 0 header writer (§6.2.1) — fixture builders + tests
    // ================================================================

    /// Clean-room reproduction of [`setup::parse_floor0_header`] from the
    /// §6.2.1 step list, used only to exercise the writer's bit-exact
    /// roundtrip property without coupling the encoder test suite to
    /// the setup-header outer walker.
    ///
    /// Spec source: `docs/audio/vorbis/Vorbis_I_spec.pdf` §6.2.1.
    fn local_parse_floor0_for_tests(
        reader: &mut oxideav_core::bits::BitReaderLsb<'_>,
    ) -> Floor0Header {
        let order = reader.read_u32(8).unwrap() as u8;
        let rate = reader.read_u32(16).unwrap() as u16;
        let bark_map_size = reader.read_u32(16).unwrap() as u16;
        let amplitude_bits = reader.read_u32(6).unwrap() as u8;
        let amplitude_offset = reader.read_u32(8).unwrap() as u8;
        let number_of_books = (reader.read_u32(4).unwrap() as usize) + 1;
        let mut book_list = Vec::with_capacity(number_of_books);
        for _ in 0..number_of_books {
            book_list.push(reader.read_u32(8).unwrap() as u8);
        }
        Floor0Header {
            order,
            rate,
            bark_map_size,
            amplitude_bits,
            amplitude_offset,
            book_list,
        }
    }

    /// The "minimal" floor 0 from the setup-header test suite:
    /// `order=4`, `rate=44100`, `bark_map_size=64`, `amplitude_bits=8`,
    /// `amplitude_offset=100`, `book_list=[0]`. Mirrors
    /// `setup::tests::SetupHeaderBuilder::minimal_floor0`.
    fn minimal_floor0() -> Floor0Header {
        Floor0Header {
            order: 4,
            rate: 44100,
            bark_map_size: 64,
            amplitude_bits: 8,
            amplitude_offset: 100,
            book_list: vec![0],
        }
    }

    fn floor0_roundtrips(header: &Floor0Header) {
        let bytes = write_floor0_header(header).expect("write must succeed");
        let mut reader = oxideav_core::bits::BitReaderLsb::new(&bytes);
        let parsed = local_parse_floor0_for_tests(&mut reader);
        assert_eq!(&parsed, header, "floor0 roundtrip equality");
    }

    /// Pin the exact bit layout of the minimal floor 0 fixture. This
    /// mirrors the byte sequence that `setup::tests::minimal_floor0`
    /// emits (excluding the outer `floor_type=0` 16-bit selector,
    /// which is the setup walker's responsibility).
    ///
    /// The `unusual_byte_groupings` lint is locally disabled because
    /// each binary literal below is deliberately split along the
    /// §6.2.1 sub-byte field boundary it visualises (e.g. `0b00_001000`
    /// = the high 2 bits of amplitude_offset spliced above the 6 bits
    /// of amplitude_bits).
    #[test]
    #[allow(clippy::unusual_byte_groupings)]
    fn floor0_byte_shape_minimal() {
        let bytes = write_floor0_header(&minimal_floor0()).expect("must build");
        // Bits emitted (LSB-first per §2.1.4):
        //   order=4              -> 8 bits  -> 0b00000100
        //   rate=44100           -> 16 bits -> 0xAC44 -> LE bytes 0x44 0xAC
        //   bark_map_size=64     -> 16 bits -> 0x0040 -> LE bytes 0x40 0x00
        //   amplitude_bits=8     -> 6 bits  -> 0b001000
        //   amplitude_offset=100 -> 8 bits  -> 0b01100100
        //   number_of_books-1=0  -> 4 bits  -> 0b0000
        //   book_list[0]=0       -> 8 bits  -> 0b00000000
        //
        // The first three fields cover whole-byte spans so they pack
        // as plain LE bytes. The amplitude_bits (6 b) + first 2 bits
        // of amplitude_offset share a byte; the remaining 6 bits of
        // amplitude_offset share the next byte with the 2 low bits of
        // (number_of_books - 1); the high 2 bits of (number_of_books
        // - 1 = 0) plus the 6 low bits of book_list[0] share the next
        // byte; the high 2 bits of book_list[0] occupy the LSBs of
        // the final byte with the rest as zero-padding.
        //
        // Total bits = 8 + 16 + 16 + 6 + 8 + 4 + 8 = 66 bits ->
        // 9 bytes (with 6 bits of zero padding).
        assert_eq!(bytes.len(), 9);
        assert_eq!(bytes[0], 0x04); // order
        assert_eq!(bytes[1], 0x44); // rate low byte
        assert_eq!(bytes[2], 0xAC); // rate high byte
        assert_eq!(bytes[3], 0x40); // bark_map_size low byte
        assert_eq!(bytes[4], 0x00); // bark_map_size high byte
                                    // Byte 5: amplitude_bits=8 in low 6 bits; high 2 bits =
                                    // low 2 bits of amplitude_offset=100=0b01100100, i.e. 0b00.
        assert_eq!(bytes[5], 0b00_001000);
        // Byte 6: high 6 bits of amplitude_offset=0b011001 in low 6
        // bits; high 2 bits = low 2 bits of (number_of_books - 1)=0.
        assert_eq!(bytes[6], 0b00_011001);
        // Byte 7: high 2 bits of (number_of_books - 1)=0 in low 2
        // bits; high 6 bits = low 6 bits of book_list[0]=0.
        assert_eq!(bytes[7], 0b000000_00);
        // Byte 8: high 2 bits of book_list[0]=0 in low 2 bits; rest zero.
        assert_eq!(bytes[8], 0b000000_00);
        // Roundtrip check.
        floor0_roundtrips(&minimal_floor0());
    }

    /// Hand-computed bit-length formula for floor 0:
    /// 8 + 16 + 16 + 6 + 8 + 4 + 8 × number_of_books bits.
    /// Verified on the minimal (1 book → 66 bits → 9 bytes) and a
    /// max-length (16 books → 8+16+16+6+8+4+128 = 186 bits → 24 bytes)
    /// shape.
    #[test]
    fn floor0_bit_length_formula() {
        // Minimal: 66 bits → 9 bytes.
        let minimal = write_floor0_header(&minimal_floor0()).expect("write");
        assert_eq!(minimal.len(), 9);

        // Max-length book list = 16 entries.
        let max_books = Floor0Header {
            order: 32,
            rate: 48000,
            bark_map_size: 256,
            amplitude_bits: 6,
            amplitude_offset: 200,
            book_list: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        };
        let max_bytes = write_floor0_header(&max_books).expect("write");
        // 8 + 16 + 16 + 6 + 8 + 4 + 16 × 8 = 186 bits -> 24 bytes.
        assert_eq!(max_bytes.len(), 24);
        floor0_roundtrips(&max_books);
    }

    #[test]
    fn floor0_roundtrip_minimal() {
        floor0_roundtrips(&minimal_floor0());
    }

    /// Spec-realistic floor 0: high-order LSP filter from a 48k q5
    /// pipeline.
    #[test]
    fn floor0_roundtrip_spec_realistic_48k() {
        floor0_roundtrips(&Floor0Header {
            order: 16,
            rate: 48000,
            bark_map_size: 128,
            amplitude_bits: 8,
            amplitude_offset: 192,
            book_list: vec![0, 2, 5, 7],
        });
    }

    /// `amplitude_bits` at the 6-bit field upper edge.
    #[test]
    fn floor0_roundtrip_amplitude_bits_max() {
        floor0_roundtrips(&Floor0Header {
            order: 8,
            rate: 22050,
            bark_map_size: 64,
            amplitude_bits: 63, // 6-bit field upper edge
            amplitude_offset: 0,
            book_list: vec![3],
        });
    }

    /// All u8 / u16 field corners at their representable maxima.
    #[test]
    fn floor0_roundtrip_field_extremes() {
        floor0_roundtrips(&Floor0Header {
            order: u8::MAX,
            rate: u16::MAX,
            bark_map_size: u16::MAX,
            amplitude_bits: 1, // smallest non-zero amplitude (writer-level)
            amplitude_offset: u8::MAX,
            book_list: vec![u8::MAX, 0, u8::MAX, 0],
        });
    }

    /// All-zero floor 0 (spec-legal as a writer-side input shape; the
    /// runtime [`Floor0Decoder::new`] additionally rejects zero
    /// `order` / `bark_map_size` / `amplitude_bits`, but that is a
    /// decoder-time gate, not a §6.2.1 writer-time invariant).
    #[test]
    fn floor0_roundtrip_all_zero_minimal_book_list() {
        floor0_roundtrips(&Floor0Header {
            order: 0,
            rate: 0,
            bark_map_size: 0,
            amplitude_bits: 0,
            amplitude_offset: 0,
            book_list: vec![0],
        });
    }

    /// Book list at the 16-entry upper edge.
    #[test]
    fn floor0_roundtrip_book_list_max() {
        floor0_roundtrips(&Floor0Header {
            order: 4,
            rate: 44100,
            bark_map_size: 64,
            amplitude_bits: 8,
            amplitude_offset: 100,
            book_list: (0u8..16).collect(),
        });
    }

    /// `amplitude_bits` above the 6-bit field width is rejected.
    #[test]
    fn floor0_rejects_amplitude_bits_overflow() {
        let mut h = minimal_floor0();
        h.amplitude_bits = 64;
        assert_eq!(
            write_floor0_header(&h),
            Err(WriteFloor0Error::AmplitudeBitsOverflow(64))
        );
    }

    /// An empty book list cannot be expressed by the `number_of_books
    /// - 1` 4-bit field — the smallest representable count is 1.
    #[test]
    fn floor0_rejects_empty_book_list() {
        let mut h = minimal_floor0();
        h.book_list.clear();
        assert_eq!(
            write_floor0_header(&h),
            Err(WriteFloor0Error::EmptyBookList)
        );
    }

    /// A book list exceeding 16 entries cannot be expressed by the
    /// 4-bit count field.
    #[test]
    fn floor0_rejects_book_list_too_long() {
        let mut h = minimal_floor0();
        h.book_list = vec![0; 17];
        assert_eq!(
            write_floor0_header(&h),
            Err(WriteFloor0Error::BookListTooLong(17))
        );
    }

    /// `WriteFloor0Error::Display` is non-empty for every variant and
    /// echoes the §6.2.1 tag.
    #[test]
    fn floor0_error_display_smoke() {
        let cases = [
            WriteFloor0Error::AmplitudeBitsOverflow(64),
            WriteFloor0Error::EmptyBookList,
            WriteFloor0Error::BookListTooLong(17),
        ];
        for case in cases {
            let s = format!("{case}");
            assert!(!s.is_empty(), "Display empty for {case:?}");
            assert!(
                s.contains("floor0"),
                "Display for {case:?} must mention the §6.2.1 floor0 context"
            );
        }
    }

    /// Floor 0 write errors surface as `WriteError::Floor0` via the
    /// `From` impl, preserving the variant for caller inspection and
    /// the source chain.
    #[test]
    fn write_error_floor0_glue() {
        let inner: WriteError = WriteFloor0Error::AmplitudeBitsOverflow(64).into();
        assert_eq!(
            inner,
            WriteError::Floor0(WriteFloor0Error::AmplitudeBitsOverflow(64))
        );
        use std::error::Error as StdError;
        let src = StdError::source(&inner).expect("source chain should reach Floor0");
        // The chained Display must non-trivially echo the floor0 tag.
        assert!(format!("{src}").contains("floor0"));
    }

    /// The `write_floor0_header_into_writer` splice point appends to
    /// an in-progress writer at the current bit offset, leaving
    /// pre-existing bits intact. This pins the shape the setup-header
    /// writer will splice the floor 0 body into.
    #[test]
    fn floor0_into_writer_splice_appends_after_existing_bits() {
        let mut w = BitWriterLsb::with_capacity(4);
        // Seed: write 7 bits (sub-byte) before splicing.
        w.write_u32(0b1010101, 7);
        write_floor0_header_into_writer(&minimal_floor0(), &mut w).expect("splice");
        let with_splice = w.finish();

        // Standalone floor 0 bytes, for comparison.
        let standalone = write_floor0_header(&minimal_floor0()).expect("standalone");

        // Spliced output must be strictly longer than the seed-only
        // and standalone slices.
        assert!(with_splice.len() >= standalone.len());

        // Re-decode the spliced output: consume the 7 seed bits then
        // run the floor 0 parser at the resumed position.
        let mut r = oxideav_core::bits::BitReaderLsb::new(&with_splice);
        let seed = r.read_u32(7).unwrap();
        assert_eq!(seed, 0b1010101);
        let parsed = local_parse_floor0_for_tests(&mut r);
        assert_eq!(parsed, minimal_floor0());
    }

    /// The fail-closed gate: when the writer rejects a header, no
    /// bits are appended to the supplied splice writer. This pins the
    /// "validate before emit" contract documented on
    /// [`write_floor0_header_into_writer`].
    #[test]
    fn floor0_into_writer_splice_emits_no_bits_on_error() {
        let mut w = BitWriterLsb::with_capacity(4);
        // Seed 3 bits to give the writer some pre-existing state.
        w.write_u32(0b101, 3);
        let before = w.bit_position();

        let mut bad = minimal_floor0();
        bad.amplitude_bits = 64; // 6-bit field overflow

        let err = write_floor0_header_into_writer(&bad, &mut w).expect_err("must reject overflow");
        assert_eq!(err, WriteFloor0Error::AmplitudeBitsOverflow(64));

        // bit_position unchanged: the gate ran before any write call.
        assert_eq!(w.bit_position(), before);
    }

    // ================================================================
    // Residue header writer (§8.6.1) — fixture builders + tests
    // ================================================================

    /// Clean-room reproduction of `setup::parse_residue_header` from
    /// the §8.6.1 step list, used only to exercise the writer's
    /// bit-exact roundtrip property without coupling the encoder test
    /// suite to the setup-header outer walker.
    ///
    /// Spec source: `docs/audio/vorbis/Vorbis_I_spec.pdf` §8.6.1.
    fn local_parse_residue_for_tests(
        reader: &mut oxideav_core::bits::BitReaderLsb<'_>,
        residue_type: u16,
    ) -> ResidueHeader {
        let residue_begin = reader.read_u32(24).unwrap();
        let residue_end = reader.read_u32(24).unwrap();
        let partition_size = reader.read_u32(24).unwrap() + 1;
        let classifications = (reader.read_u32(6).unwrap() as u8) + 1;
        let classbook = reader.read_u32(8).unwrap() as u8;

        let mut cascade = Vec::with_capacity(classifications as usize);
        for _ in 0..classifications {
            let low_bits = reader.read_u32(3).unwrap() as u8;
            let bitflag = reader.read_u32(1).unwrap();
            let high_bits = if bitflag == 1 {
                reader.read_u32(5).unwrap() as u8
            } else {
                0
            };
            cascade.push(high_bits.wrapping_mul(8).wrapping_add(low_bits));
        }

        let mut books = Vec::with_capacity(classifications as usize);
        for &cas in &cascade {
            let mut row: [Option<u8>; 8] = [None; 8];
            for (j, slot) in row.iter_mut().enumerate() {
                if (cas >> j) & 1 == 1 {
                    *slot = Some(reader.read_u32(8).unwrap() as u8);
                }
            }
            books.push(row);
        }

        ResidueHeader {
            residue_type,
            residue_begin,
            residue_end,
            partition_size,
            classifications,
            classbook,
            cascade,
            books,
        }
    }

    /// The "minimal" residue header: residue_type=2, residue_begin=0,
    /// residue_end=128, partition_size=32, classifications=1,
    /// classbook=0, cascade=[1] (only stage 0 present),
    /// books=[[Some(0),None,None,None,None,None,None,None]].
    /// Mirrors the `minimal_residue_type2` setup-builder shape.
    fn minimal_residue() -> ResidueHeader {
        ResidueHeader {
            residue_type: 2,
            residue_begin: 0,
            residue_end: 128,
            partition_size: 32,
            classifications: 1,
            classbook: 0,
            cascade: vec![1],
            books: vec![[Some(0), None, None, None, None, None, None, None]],
        }
    }

    fn residue_roundtrips(header: &ResidueHeader) {
        let bytes = write_residue_header(header).expect("write must succeed");
        let mut reader = oxideav_core::bits::BitReaderLsb::new(&bytes);
        let parsed = local_parse_residue_for_tests(&mut reader, header.residue_type);
        assert_eq!(&parsed, header, "residue roundtrip equality");
    }

    // ----------------------------------------------------------------
    // Residue header writer — byte-shape pinning.
    // ----------------------------------------------------------------

    /// Pin the exact bit layout of the minimal residue fixture. This
    /// reproduces a single hand-rolled `BitWriterLsb` stream against
    /// the actual writer output, so the §8.6.1 field order + widths
    /// are locked, not merely "roundtrips through the parser."
    #[test]
    fn residue_byte_shape_minimal() {
        let bytes = write_residue_header(&minimal_residue()).expect("must build");
        // Bits emitted (LSB-first per §2.1.4):
        //   residue_begin=0          -> 24 bits
        //   residue_end=128          -> 24 bits
        //   partition_size-1=31      -> 24 bits
        //   classifications-1=0      ->  6 bits
        //   classbook=0              ->  8 bits
        //   cascade[0]=1: low=1 bf=0 ->  3 + 1 = 4 bits
        //   books[0][0]=Some(0)      ->  8 bits
        // Total = 24+24+24+6+8+4+8 = 98 bits.
        let total_bits = 24 + 24 + 24 + 6 + 8 + 4 + 8;
        assert_eq!(bytes.len(), (total_bits as usize).div_ceil(8));

        let mut expected = BitWriterLsb::with_capacity(16);
        expected.write_u32(0, 24); // residue_begin
        expected.write_u32(128, 24); // residue_end
        expected.write_u32(31, 24); // partition_size - 1
        expected.write_u32(0, 6); // classifications - 1
        expected.write_u32(0, 8); // classbook
        expected.write_u32(1, 3); // cascade[0] low_bits
        expected.write_u32(0, 1); // cascade[0] bitflag (high == 0)
        expected.write_u32(0, 8); // books[0][0]
        assert_eq!(bytes, expected.finish());
    }

    /// Closed-form bit-length formula on a non-trivial fixture: two
    /// classifications, the first cascade-byte uses both halves
    /// (high_bits > 0 → bitflag=1 + 5 more bits), the second is
    /// low-only (high_bits == 0 → bitflag=0).
    ///
    /// Per-cascade-byte bits =
    ///   3 (low) + 1 (bitflag) + (bitflag ? 5 : 0)
    /// Total header bits =
    ///   24 (begin) + 24 (end) + 24 (psize-1) + 6 (class-1) + 8 (classbook)
    ///   + Σ per-class (3 + 1 + maybe 5) + 8 × popcount(cascade[i])
    #[test]
    fn residue_bit_length_formula_two_classes_mixed_cascades() {
        let header = ResidueHeader {
            residue_type: 1,
            residue_begin: 0,
            residue_end: 1024,
            partition_size: 32,
            classifications: 2,
            classbook: 3,
            cascade: vec![
                0b0010_0001, // 33: low=1, high=4 (bitflag=1, +5 bits); 2 bits set → 2 books
                0b0000_0010, // 2:  low=2, high=0 (bitflag=0, no extra);  1 bit set → 1 book
            ],
            books: vec![
                [Some(7), None, None, None, None, Some(11), None, None],
                [None, Some(2), None, None, None, None, None, None],
            ],
        };
        let bytes = write_residue_header(&header).expect("write");
        // Header constants + per-class cascade emission + per-book emission.
        let cascade_bits = (3 + 1 + 5) + (3 + 1); // class 0 high>0, class 1 high==0
        let book_bits = 8 * (2 + 1);
        let total_bits = 24 + 24 + 24 + 6 + 8 + cascade_bits + book_bits;
        assert_eq!(bytes.len(), (total_bits as usize).div_ceil(8));
        residue_roundtrips(&header);
    }

    // ----------------------------------------------------------------
    // Residue header writer — bit-exact roundtrip.
    // ----------------------------------------------------------------

    #[test]
    fn residue_roundtrips_minimal() {
        residue_roundtrips(&minimal_residue());
    }

    /// Residue type 0 is the format-0 (interleaved) classifier. The
    /// header layout is identical to types 1 and 2 — only the runtime
    /// decode differs.
    #[test]
    fn residue_roundtrips_type_0() {
        let mut h = minimal_residue();
        h.residue_type = 0;
        residue_roundtrips(&h);
    }

    /// Residue type 1 (contiguous layout).
    #[test]
    fn residue_roundtrips_type_1() {
        let mut h = minimal_residue();
        h.residue_type = 1;
        residue_roundtrips(&h);
    }

    /// Begin/end at the 24-bit field upper edge.
    #[test]
    fn residue_roundtrips_begin_end_at_24bit_edge() {
        residue_roundtrips(&ResidueHeader {
            residue_type: 2,
            residue_begin: 0x00FF_FFFE,
            residue_end: 0x00FF_FFFF,
            partition_size: 1,
            classifications: 1,
            classbook: 255,
            cascade: vec![0],
            books: vec![[None; 8]],
        });
    }

    /// partition_size at the upper edge (2^24, encoded as 24-bit all-ones).
    #[test]
    fn residue_roundtrips_partition_size_upper_edge() {
        residue_roundtrips(&ResidueHeader {
            residue_type: 2,
            residue_begin: 0,
            residue_end: 0,
            partition_size: 1u32 << 24,
            classifications: 1,
            classbook: 0,
            cascade: vec![0],
            books: vec![[None; 8]],
        });
    }

    /// classifications at the upper edge (64, encoded as 6-bit all-ones).
    /// Mixed cascade bytes to also exercise the high-bits branch.
    #[test]
    fn residue_roundtrips_classifications_upper_edge() {
        let cascade: Vec<u8> = (0u8..64).collect();
        let mut books = Vec::with_capacity(64);
        for &cas in &cascade {
            let mut row: [Option<u8>; 8] = [None; 8];
            for (j, slot) in row.iter_mut().enumerate() {
                if (cas >> j) & 1 == 1 {
                    *slot = Some((j as u8).wrapping_mul(7).wrapping_add(cas));
                }
            }
            books.push(row);
        }
        residue_roundtrips(&ResidueHeader {
            residue_type: 2,
            residue_begin: 16,
            residue_end: 4096,
            partition_size: 128,
            classifications: 64,
            classbook: 17,
            cascade,
            books,
        });
    }

    /// Cascade byte 0xFF (all 8 stages present) — exercises every
    /// stage's 8-bit book read.
    #[test]
    fn residue_roundtrips_cascade_all_stages_set() {
        residue_roundtrips(&ResidueHeader {
            residue_type: 0,
            residue_begin: 0,
            residue_end: 256,
            partition_size: 32,
            classifications: 1,
            classbook: 0,
            cascade: vec![0xFF],
            books: vec![[
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
            ]],
        });
    }

    /// Cascade byte 0x00 (no stages) — no per-stage book bytes are
    /// emitted; the roundtrip still must hold.
    #[test]
    fn residue_roundtrips_cascade_all_stages_clear() {
        residue_roundtrips(&ResidueHeader {
            residue_type: 0,
            residue_begin: 0,
            residue_end: 256,
            partition_size: 32,
            classifications: 1,
            classbook: 0,
            cascade: vec![0x00],
            books: vec![[None; 8]],
        });
    }

    /// Cascade byte with high_bits at the 5-bit upper edge (31) and
    /// low_bits at the 3-bit upper edge (7) — packs as 0xFF, the same
    /// numeric byte the all-stages-set test uses but reached via the
    /// (high*8 + low) accounting.
    #[test]
    fn residue_roundtrips_cascade_high_bits_upper_edge() {
        residue_roundtrips(&ResidueHeader {
            residue_type: 1,
            residue_begin: 0,
            residue_end: 128,
            partition_size: 8,
            classifications: 1,
            classbook: 42,
            // 31 * 8 + 7 = 255 = 0xFF.
            cascade: vec![31u8.wrapping_mul(8).wrapping_add(7)],
            books: vec![[
                Some(10),
                Some(11),
                Some(12),
                Some(13),
                Some(14),
                Some(15),
                Some(16),
                Some(17),
            ]],
        });
    }

    /// Mix of bitflag=0 and bitflag=1 cascades across consecutive
    /// classifications — the parser's bitflag branch is per-class, so
    /// transitions across the classification loop must be bit-exact.
    #[test]
    fn residue_roundtrips_alternating_bitflag_classes() {
        // 4 classifications: low-only, high>0, low-only, high>0.
        let cascade: Vec<u8> = vec![
            0b0000_0011, // low=3, high=0
            0b0010_0001, // low=1, high=4
            0b0000_0010, // low=2, high=0
            0b0001_0100, // low=4, high=2
        ];
        let mut books = Vec::with_capacity(4);
        for (i, &cas) in cascade.iter().enumerate() {
            let mut row: [Option<u8>; 8] = [None; 8];
            for (j, slot) in row.iter_mut().enumerate() {
                if (cas >> j) & 1 == 1 {
                    *slot = Some(((i * 8 + j) % 250) as u8);
                }
            }
            books.push(row);
        }
        residue_roundtrips(&ResidueHeader {
            residue_type: 2,
            residue_begin: 4,
            residue_end: 512,
            partition_size: 16,
            classifications: 4,
            classbook: 9,
            cascade,
            books,
        });
    }

    /// classbook = 255 (upper edge of the 8-bit field).
    #[test]
    fn residue_roundtrips_classbook_at_upper_edge() {
        let mut h = minimal_residue();
        h.classbook = u8::MAX;
        residue_roundtrips(&h);
    }

    // ----------------------------------------------------------------
    // Residue header writer — rejection variants.
    // ----------------------------------------------------------------

    #[test]
    fn residue_rejects_unsupported_residue_type() {
        let mut h = minimal_residue();
        h.residue_type = 3; // §4.2.4 step 2c cap is 2
        assert_eq!(
            write_residue_header(&h),
            Err(WriteResidueError::UnsupportedResidueType(3))
        );
    }

    #[test]
    fn residue_rejects_residue_begin_overflow() {
        let mut h = minimal_residue();
        h.residue_begin = 0x0100_0000; // 24-bit field caps at 0x00FF_FFFF
        assert_eq!(
            write_residue_header(&h),
            Err(WriteResidueError::ResidueBeginOverflow(0x0100_0000))
        );
    }

    #[test]
    fn residue_rejects_residue_end_overflow() {
        let mut h = minimal_residue();
        h.residue_end = 0x0100_0000;
        assert_eq!(
            write_residue_header(&h),
            Err(WriteResidueError::ResidueEndOverflow(0x0100_0000))
        );
    }

    #[test]
    fn residue_rejects_partition_size_zero() {
        let mut h = minimal_residue();
        h.partition_size = 0;
        assert_eq!(
            write_residue_header(&h),
            Err(WriteResidueError::PartitionSizeOutOfRange(0))
        );
    }

    #[test]
    fn residue_rejects_partition_size_above_cap() {
        let mut h = minimal_residue();
        h.partition_size = (1u32 << 24) + 1; // one past the cap
        assert_eq!(
            write_residue_header(&h),
            Err(WriteResidueError::PartitionSizeOutOfRange((1u32 << 24) + 1))
        );
    }

    #[test]
    fn residue_rejects_classifications_zero() {
        let mut h = minimal_residue();
        h.classifications = 0;
        // cascade/books length checks fire later, so this rejection
        // gates first.
        assert_eq!(
            write_residue_header(&h),
            Err(WriteResidueError::ClassificationsOutOfRange(0))
        );
    }

    #[test]
    fn residue_rejects_classifications_above_cap() {
        let mut h = minimal_residue();
        h.classifications = 65;
        h.cascade = vec![0; 65];
        h.books = vec![[None; 8]; 65];
        assert_eq!(
            write_residue_header(&h),
            Err(WriteResidueError::ClassificationsOutOfRange(65))
        );
    }

    #[test]
    fn residue_rejects_cascade_length_mismatch() {
        let mut h = minimal_residue();
        // classifications=1 but cascade has 2 entries.
        h.cascade = vec![0, 0];
        h.books = vec![[None; 8], [None; 8]];
        assert_eq!(
            write_residue_header(&h),
            Err(WriteResidueError::CascadeLengthMismatch {
                classifications: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn residue_rejects_books_length_mismatch() {
        let mut h = minimal_residue();
        // classifications=1, cascade length=1, but books length=2.
        h.books = vec![
            [Some(0), None, None, None, None, None, None, None],
            [None; 8],
        ];
        assert_eq!(
            write_residue_header(&h),
            Err(WriteResidueError::BooksLengthMismatch {
                classifications: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn residue_rejects_books_present_but_cascade_clear() {
        let mut h = minimal_residue();
        // cascade=1 (bit 0 set), books[0][0]=Some, books[0][1]=Some
        // but cascade bit 1 is clear.
        h.cascade = vec![1];
        h.books = vec![[Some(0), Some(7), None, None, None, None, None, None]];
        assert_eq!(
            write_residue_header(&h),
            Err(WriteResidueError::BooksCascadeMismatch {
                class: 0,
                stage: 1,
                book_present: true,
            })
        );
    }

    #[test]
    fn residue_rejects_books_absent_but_cascade_set() {
        let mut h = minimal_residue();
        // cascade=3 (bits 0 and 1 set), books only fills stage 0.
        h.cascade = vec![3];
        h.books = vec![[Some(0), None, None, None, None, None, None, None]];
        assert_eq!(
            write_residue_header(&h),
            Err(WriteResidueError::BooksCascadeMismatch {
                class: 0,
                stage: 1,
                book_present: false,
            })
        );
    }

    /// `WriteResidueError::Display` is non-empty for every variant
    /// and echoes the §8.6.1 / residue context tag.
    #[test]
    fn residue_error_display_smoke() {
        let cases = [
            WriteResidueError::UnsupportedResidueType(5),
            WriteResidueError::ResidueBeginOverflow(0x0100_0000),
            WriteResidueError::ResidueEndOverflow(0x0100_0000),
            WriteResidueError::PartitionSizeOutOfRange(0),
            WriteResidueError::ClassificationsOutOfRange(0),
            WriteResidueError::CascadeLengthMismatch {
                classifications: 1,
                actual: 2,
            },
            WriteResidueError::BooksLengthMismatch {
                classifications: 1,
                actual: 2,
            },
            WriteResidueError::BooksCascadeMismatch {
                class: 0,
                stage: 1,
                book_present: true,
            },
            WriteResidueError::BooksCascadeMismatch {
                class: 0,
                stage: 1,
                book_present: false,
            },
        ];
        for case in cases {
            let s = format!("{case}");
            assert!(!s.is_empty(), "Display empty for {case:?}");
            assert!(
                s.contains("residue"),
                "Display for {case:?} must mention residue context"
            );
        }
    }

    // ----------------------------------------------------------------
    // Residue header writer — WriteError glue.
    // ----------------------------------------------------------------

    /// Residue write errors surface as `WriteError::Residue` via the
    /// `From` impl, preserving the variant for caller inspection and
    /// the source chain.
    #[test]
    fn write_error_residue_glue() {
        let inner: WriteError = WriteResidueError::UnsupportedResidueType(7).into();
        assert_eq!(
            inner,
            WriteError::Residue(WriteResidueError::UnsupportedResidueType(7))
        );
        use std::error::Error as StdError;
        let src = StdError::source(&inner).expect("source chain should reach Residue");
        // The chained Display must non-trivially echo the residue tag.
        assert!(format!("{src}").contains("residue"));
    }

    // ----------------------------------------------------------------
    // Residue header writer — splice point.
    // ----------------------------------------------------------------

    /// The `write_residue_header_into_writer` splice point appends to
    /// an in-progress writer at the current bit offset, leaving
    /// pre-existing bits intact. This pins the shape the setup-header
    /// writer will splice the residue body into.
    #[test]
    fn residue_into_writer_splice_appends_after_existing_bits() {
        let mut w = BitWriterLsb::with_capacity(16);
        // Seed: write 7 bits (sub-byte) before splicing.
        w.write_u32(0b1010101, 7);
        write_residue_header_into_writer(&minimal_residue(), &mut w).expect("splice");
        let with_splice = w.finish();

        // Standalone residue bytes, for comparison.
        let standalone = write_residue_header(&minimal_residue()).expect("standalone");

        // Spliced output must be at least as long as the standalone
        // slice (because the seed bits push the splice forward, and
        // rounding pushes a partial last byte up).
        assert!(with_splice.len() >= standalone.len());

        // Re-decode the spliced output: consume the 7 seed bits then
        // run the residue parser at the resumed position.
        let mut r = oxideav_core::bits::BitReaderLsb::new(&with_splice);
        let seed = r.read_u32(7).unwrap();
        assert_eq!(seed, 0b1010101);
        let parsed = local_parse_residue_for_tests(&mut r, 2);
        assert_eq!(parsed, minimal_residue());
    }

    /// The fail-closed gate: when the writer rejects a header, no
    /// bits are appended to the supplied splice writer. This pins the
    /// "validate before emit" contract documented on
    /// [`write_residue_header_into_writer`].
    #[test]
    fn residue_into_writer_splice_emits_no_bits_on_error() {
        let mut w = BitWriterLsb::with_capacity(16);
        // Seed 3 bits to give the writer some pre-existing state.
        w.write_u32(0b101, 3);
        let before = w.bit_position();

        let mut bad = minimal_residue();
        bad.residue_type = 7; // §4.2.4 step 2c cap is 2

        let err = write_residue_header_into_writer(&bad, &mut w)
            .expect_err("must reject unsupported residue_type");
        assert_eq!(err, WriteResidueError::UnsupportedResidueType(7));

        // bit_position unchanged: the gate ran before any write call.
        assert_eq!(w.bit_position(), before);
    }

    // ================================================================
    // Mapping header writer (§4.2.4 "Mappings") — fixture builders + tests
    // ================================================================

    /// Clean-room reproduction of `setup::parse_mapping_header` from
    /// the §4.2.4 "Mappings" step list, used only to exercise the
    /// writer's bit-exact roundtrip property without coupling the
    /// encoder test suite to the setup-header outer walker. Mirrors
    /// the writer's encoding choices (densest legal form).
    ///
    /// Spec source: `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.2.4
    /// "Mappings" + §9.2.1 (`ilog`).
    fn local_parse_mapping_for_tests(
        reader: &mut oxideav_core::bits::BitReaderLsb<'_>,
        audio_channels: u8,
    ) -> MappingHeader {
        let mapping_type = reader.read_u32(16).unwrap() as u16;
        let submaps_flag = reader.read_u32(1).unwrap() == 1;
        let submaps = if submaps_flag {
            (reader.read_u32(4).unwrap() as u8) + 1
        } else {
            1
        };
        let square_polar_flag = reader.read_u32(1).unwrap() == 1;
        let coupling = if square_polar_flag {
            let coupling_steps = (reader.read_u32(8).unwrap() as usize) + 1;
            let channel_bits = ilog((audio_channels as u32).saturating_sub(1));
            let mut steps = Vec::with_capacity(coupling_steps);
            for _ in 0..coupling_steps {
                let magnitude_channel = reader.read_u32(channel_bits).unwrap() as u8;
                let angle_channel = reader.read_u32(channel_bits).unwrap() as u8;
                steps.push(MappingCouplingStep {
                    magnitude_channel,
                    angle_channel,
                });
            }
            steps
        } else {
            Vec::new()
        };
        let _reserved = reader.read_u32(2).unwrap();
        let mux = if submaps > 1 {
            let mut mux = Vec::with_capacity(audio_channels as usize);
            for _ in 0..(audio_channels as usize) {
                mux.push(reader.read_u32(4).unwrap() as u8);
            }
            mux
        } else {
            Vec::new()
        };
        let mut submap_configs = Vec::with_capacity(submaps as usize);
        for _ in 0..(submaps as usize) {
            let time_placeholder = reader.read_u32(8).unwrap() as u8;
            let floor = reader.read_u32(8).unwrap() as u8;
            let residue = reader.read_u32(8).unwrap() as u8;
            submap_configs.push(crate::setup::MappingSubmap {
                time_placeholder,
                floor,
                residue,
            });
        }
        MappingHeader {
            mapping_type,
            submaps,
            coupling,
            mux,
            submap_configs,
        }
    }

    /// The "minimal" mapping configuration: mono, single submap, no
    /// coupling, floor=0, residue=0. Pins the densest §4.2.4 encoding
    /// (both 1-bit flags emitted as zero).
    fn minimal_mono_mapping() -> MappingHeader {
        MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: Vec::new(),
            mux: Vec::new(),
            submap_configs: vec![crate::setup::MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        }
    }

    /// Stereo, single submap, one coupling step (magnitude=0,
    /// angle=1). Pins the §4.2.4 stereo-coupled layout at submaps=1
    /// and one coupling step.
    fn stereo_coupled_mapping() -> MappingHeader {
        MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: vec![MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            }],
            mux: Vec::new(),
            submap_configs: vec![crate::setup::MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        }
    }

    fn mapping_roundtrips(
        header: &MappingHeader,
        audio_channels: u8,
        floor_count: usize,
        residue_count: usize,
    ) {
        let bytes = write_mapping_header(header, audio_channels, floor_count, residue_count)
            .expect("write must succeed");
        let mut reader = oxideav_core::bits::BitReaderLsb::new(&bytes);
        let parsed = local_parse_mapping_for_tests(&mut reader, audio_channels);
        assert_eq!(&parsed, header, "mapping roundtrip equality");
    }

    // ----------------------------------------------------------------
    // Mapping header writer — byte-shape pinning.
    // ----------------------------------------------------------------

    /// Pin the exact bit layout of the minimal mono fixture. Locks
    /// the §4.2.4 emit order at the densest-encoding defaults
    /// (`submaps_flag = 0`, `square_polar_flag = 0`).
    #[test]
    fn mapping_byte_shape_minimal_mono() {
        let bytes = write_mapping_header(&minimal_mono_mapping(), 1, 1, 1).expect("must build");
        // Bits emitted (LSB-first per §2.1.4):
        //   mapping_type=0           -> 16 bits
        //   submaps_flag=0           ->  1 bit
        //   square_polar_flag=0      ->  1 bit
        //   reserved=0               ->  2 bits
        //   submap[0].time=0         ->  8 bits
        //   submap[0].floor=0        ->  8 bits
        //   submap[0].residue=0      ->  8 bits
        // Total = 16+1+1+2+8+8+8 = 44 bits = 6 bytes (with 4 bits of
        // §2.1.8 zero padding in the final byte).
        let total_bits = 16 + 1 + 1 + 2 + 8 + 8 + 8;
        assert_eq!(bytes.len(), (total_bits as usize).div_ceil(8));

        let mut expected = BitWriterLsb::with_capacity(16);
        expected.write_u32(0, 16); // mapping_type
        expected.write_u32(0, 1); // submaps_flag = 0 (submaps == 1)
        expected.write_u32(0, 1); // square_polar_flag = 0 (no coupling)
        expected.write_u32(0, 2); // reserved
        expected.write_u32(0, 8); // submap[0].time_placeholder
        expected.write_u32(0, 8); // submap[0].floor
        expected.write_u32(0, 8); // submap[0].residue
        assert_eq!(bytes, expected.finish());
    }

    /// Pin the §4.2.4 bit layout for stereo coupling at packed-densest
    /// defaults (submaps=1, single coupling step magnitude=0/angle=1).
    /// `channel_bits = ilog(audio_channels - 1) = ilog(1) = 1`, so
    /// each magnitude/angle is exactly 1 bit.
    #[test]
    fn mapping_byte_shape_stereo_coupled() {
        let bytes = write_mapping_header(&stereo_coupled_mapping(), 2, 1, 1).expect("must build");
        // Bits emitted:
        //   mapping_type=0                     -> 16 b
        //   submaps_flag=0                     ->  1 b
        //   square_polar_flag=1                ->  1 b
        //   coupling_steps - 1 = 0             ->  8 b
        //   step[0].magnitude=0                ->  1 b
        //   step[0].angle=1                    ->  1 b
        //   reserved=0                         ->  2 b
        //   submap[0].time=0                   ->  8 b
        //   submap[0].floor=0                  ->  8 b
        //   submap[0].residue=0                ->  8 b
        // Total = 16+1+1+8+1+1+2+8+8+8 = 54 bits = 7 bytes.
        let total_bits = 16 + 1 + 1 + 8 + 1 + 1 + 2 + 8 + 8 + 8;
        assert_eq!(bytes.len(), (total_bits as usize).div_ceil(8));

        let mut expected = BitWriterLsb::with_capacity(16);
        expected.write_u32(0, 16); // mapping_type
        expected.write_u32(0, 1); // submaps_flag = 0
        expected.write_u32(1, 1); // square_polar_flag = 1
        expected.write_u32(0, 8); // coupling_steps - 1
        expected.write_u32(0, 1); // step[0].magnitude
        expected.write_u32(1, 1); // step[0].angle
        expected.write_u32(0, 2); // reserved
        expected.write_u32(0, 8); // submap[0].time
        expected.write_u32(0, 8); // submap[0].floor
        expected.write_u32(0, 8); // submap[0].residue
        assert_eq!(bytes, expected.finish());
    }

    /// Closed-form bit-length formula on a non-trivial fixture:
    /// audio_channels = 4, submaps = 2 (so submaps_flag = 1 and
    /// per-channel mux is emitted), one coupling step (so
    /// square_polar_flag = 1).
    ///
    /// Per-field bits =
    ///   16 (mapping_type)
    ///   + 1 (submaps_flag) + 4 (submaps - 1)
    ///   + 1 (square_polar_flag) + 8 (coupling_steps - 1)
    ///   + 2 * ilog(audio_channels - 1) per coupling step
    ///   + 2 (reserved)
    ///   + 4 * audio_channels (mux)
    ///   + 24 * submaps (per-submap triple)
    ///     = 16 + 5 + 9 + 2*ilog(3) + 2 + 4*4 + 24*2
    ///     = 16 + 5 + 9 + 4 + 2 + 16 + 48 = 100 bits.
    #[test]
    fn mapping_bit_length_formula_two_submaps_one_coupling_4ch() {
        let header = MappingHeader {
            mapping_type: 0,
            submaps: 2,
            coupling: vec![MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            }],
            mux: vec![0, 1, 0, 1],
            submap_configs: vec![
                crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                },
                crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 1,
                    residue: 1,
                },
            ],
        };
        let bytes = write_mapping_header(&header, 4, 2, 2).expect("write");
        let channel_bits = ilog(3) as usize;
        let coupling_bits = 1 + 8 + 2 * channel_bits;
        let mux_bits = 4 * 4;
        let submap_bits = 24 * 2;
        let total_bits = 16 + 5 + coupling_bits + 2 + mux_bits + submap_bits;
        assert_eq!(bytes.len(), total_bits.div_ceil(8));
        mapping_roundtrips(&header, 4, 2, 2);
    }

    // ----------------------------------------------------------------
    // Mapping header writer — encoding-form selection pinning.
    // ----------------------------------------------------------------

    /// When `submaps == 1` the writer must elide the 4-bit body and
    /// emit `submaps_flag = 0`. Checked via the leading bit after
    /// `mapping_type`.
    #[test]
    fn mapping_picks_dense_submaps_form_when_submaps_eq_1() {
        let bytes =
            write_mapping_header(&minimal_mono_mapping(), 1, 1, 1).expect("write must succeed");
        let mut reader = oxideav_core::bits::BitReaderLsb::new(&bytes);
        let _mapping_type = reader.read_u32(16).unwrap();
        let submaps_flag = reader.read_u32(1).unwrap();
        assert_eq!(
            submaps_flag, 0,
            "submaps=1 must encode as submaps_flag=0 (dense form)"
        );
    }

    /// When `coupling.is_empty()` the writer must elide the 8-bit
    /// coupling-steps body + per-step body and emit
    /// `square_polar_flag = 0`.
    #[test]
    fn mapping_picks_dense_coupling_form_when_coupling_empty() {
        let bytes =
            write_mapping_header(&minimal_mono_mapping(), 1, 1, 1).expect("write must succeed");
        let mut reader = oxideav_core::bits::BitReaderLsb::new(&bytes);
        let _mapping_type = reader.read_u32(16).unwrap();
        let _submaps_flag = reader.read_u32(1).unwrap();
        let square_polar_flag = reader.read_u32(1).unwrap();
        assert_eq!(
            square_polar_flag, 0,
            "empty coupling must encode as square_polar_flag=0 (dense form)"
        );
    }

    /// When `submaps > 1`, the writer must emit `submaps_flag = 1`
    /// followed by `submaps - 1` in 4 bits.
    #[test]
    fn mapping_picks_explicit_submaps_form_when_submaps_gt_1() {
        let header = MappingHeader {
            mapping_type: 0,
            submaps: 3,
            coupling: Vec::new(),
            mux: vec![0, 1, 2],
            submap_configs: vec![
                crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                },
                crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                },
                crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                },
            ],
        };
        let bytes = write_mapping_header(&header, 3, 1, 1).expect("write");
        let mut reader = oxideav_core::bits::BitReaderLsb::new(&bytes);
        let _mapping_type = reader.read_u32(16).unwrap();
        let submaps_flag = reader.read_u32(1).unwrap();
        assert_eq!(submaps_flag, 1);
        let submaps_minus_one = reader.read_u32(4).unwrap();
        assert_eq!(submaps_minus_one, 2); // 3 - 1
    }

    // ----------------------------------------------------------------
    // Mapping header writer — bit-exact roundtrip fixtures.
    // ----------------------------------------------------------------

    #[test]
    fn mapping_roundtrips_minimal_mono() {
        mapping_roundtrips(&minimal_mono_mapping(), 1, 1, 1);
    }

    #[test]
    fn mapping_roundtrips_stereo_coupled() {
        mapping_roundtrips(&stereo_coupled_mapping(), 2, 1, 1);
    }

    /// Stereo, no coupling — both flags emit zero.
    #[test]
    fn mapping_roundtrips_stereo_no_coupling() {
        mapping_roundtrips(
            &MappingHeader {
                mapping_type: 0,
                submaps: 1,
                coupling: Vec::new(),
                mux: Vec::new(),
                submap_configs: vec![crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                }],
            },
            2,
            1,
            1,
        );
    }

    /// 5.1-channel (6 channels), two submaps, three coupling steps.
    /// `channel_bits = ilog(5) = 3` per channel field.
    #[test]
    fn mapping_roundtrips_5_1_channels_two_submaps_three_couplings() {
        mapping_roundtrips(
            &MappingHeader {
                mapping_type: 0,
                submaps: 2,
                coupling: vec![
                    MappingCouplingStep {
                        magnitude_channel: 0,
                        angle_channel: 1,
                    },
                    MappingCouplingStep {
                        magnitude_channel: 2,
                        angle_channel: 3,
                    },
                    MappingCouplingStep {
                        magnitude_channel: 4,
                        angle_channel: 5,
                    },
                ],
                mux: vec![0, 0, 1, 1, 0, 1],
                submap_configs: vec![
                    crate::setup::MappingSubmap {
                        time_placeholder: 1,
                        floor: 0,
                        residue: 0,
                    },
                    crate::setup::MappingSubmap {
                        time_placeholder: 2,
                        floor: 1,
                        residue: 1,
                    },
                ],
            },
            6,
            2,
            2,
        );
    }

    /// submaps at the 16 upper edge (the 4-bit `submaps - 1` field's
    /// cap). Requires 16-channel context to allow the per-channel mux
    /// to legally reference every submap.
    #[test]
    fn mapping_roundtrips_submaps_at_upper_edge() {
        // 16 channels, 16 submaps; each channel maps to its own submap.
        let mux: Vec<u8> = (0u8..16).collect();
        let submap_configs: Vec<_> = (0u8..16)
            .map(|i| crate::setup::MappingSubmap {
                time_placeholder: i,
                floor: 0,
                residue: 0,
            })
            .collect();
        mapping_roundtrips(
            &MappingHeader {
                mapping_type: 0,
                submaps: 16,
                coupling: Vec::new(),
                mux,
                submap_configs,
            },
            16,
            1,
            1,
        );
    }

    /// coupling_steps at the 256 upper edge (the 8-bit
    /// `coupling_steps - 1` field's cap). Requires ≥ 2 audio channels
    /// for the magnitude/angle distinctness constraint to hold.
    #[test]
    fn mapping_roundtrips_coupling_steps_at_upper_edge() {
        let coupling: Vec<MappingCouplingStep> = (0..256)
            .map(|_| MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            })
            .collect();
        mapping_roundtrips(
            &MappingHeader {
                mapping_type: 0,
                submaps: 1,
                coupling,
                mux: Vec::new(),
                submap_configs: vec![crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                }],
            },
            2,
            1,
            1,
        );
    }

    /// Submap floor/residue indices at the 255 upper edge with the
    /// matching 256-entry floor/residue counts.
    #[test]
    fn mapping_roundtrips_submap_indices_at_upper_edge() {
        mapping_roundtrips(
            &MappingHeader {
                mapping_type: 0,
                submaps: 1,
                coupling: Vec::new(),
                mux: Vec::new(),
                submap_configs: vec![crate::setup::MappingSubmap {
                    time_placeholder: 255,
                    floor: 255,
                    residue: 255,
                }],
            },
            1,
            256,
            256,
        );
    }

    /// `time_placeholder` sweep across an interesting bit-pattern set:
    /// the §4.2.4 step 2c.v.A field is "read and discard", so the
    /// writer must preserve any 8-bit pattern verbatim.
    #[test]
    fn mapping_roundtrips_time_placeholder_sweep() {
        for tp in [0u8, 1, 7, 0x55, 0xAA, 0xFF] {
            mapping_roundtrips(
                &MappingHeader {
                    mapping_type: 0,
                    submaps: 1,
                    coupling: Vec::new(),
                    mux: Vec::new(),
                    submap_configs: vec![crate::setup::MappingSubmap {
                        time_placeholder: tp,
                        floor: 0,
                        residue: 0,
                    }],
                },
                1,
                1,
                1,
            );
        }
    }

    /// 8 audio channels — `channel_bits = ilog(7) = 3`. Coupling
    /// magnitude/angle fields each take 3 bits.
    #[test]
    fn mapping_roundtrips_8ch_coupling_width() {
        mapping_roundtrips(
            &MappingHeader {
                mapping_type: 0,
                submaps: 1,
                coupling: vec![
                    MappingCouplingStep {
                        magnitude_channel: 2,
                        angle_channel: 5,
                    },
                    MappingCouplingStep {
                        magnitude_channel: 7,
                        angle_channel: 0,
                    },
                ],
                mux: Vec::new(),
                submap_configs: vec![crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                }],
            },
            8,
            1,
            1,
        );
    }

    /// 3 audio channels — `channel_bits = ilog(2) = 2`. Pins the
    /// non-power-of-two channel-bits computation.
    #[test]
    fn mapping_roundtrips_3ch_coupling_width() {
        mapping_roundtrips(
            &MappingHeader {
                mapping_type: 0,
                submaps: 1,
                coupling: vec![MappingCouplingStep {
                    magnitude_channel: 1,
                    angle_channel: 2,
                }],
                mux: Vec::new(),
                submap_configs: vec![crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                }],
            },
            3,
            1,
            1,
        );
    }

    /// 255 audio channels — `channel_bits = ilog(254) = 8` (8-bit
    /// upper edge before the next power-of-two boundary).
    #[test]
    fn mapping_roundtrips_255ch_coupling_width() {
        mapping_roundtrips(
            &MappingHeader {
                mapping_type: 0,
                submaps: 1,
                coupling: vec![MappingCouplingStep {
                    magnitude_channel: 254,
                    angle_channel: 0,
                }],
                mux: Vec::new(),
                submap_configs: vec![crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                }],
            },
            255,
            1,
            1,
        );
    }

    /// 4 audio channels, 2 submaps with a mux pattern that cycles
    /// through both submaps. Exercises the mux-emission loop.
    #[test]
    fn mapping_roundtrips_4ch_2submaps_mux_cycle() {
        mapping_roundtrips(
            &MappingHeader {
                mapping_type: 0,
                submaps: 2,
                coupling: Vec::new(),
                mux: vec![0, 1, 0, 1],
                submap_configs: vec![
                    crate::setup::MappingSubmap {
                        time_placeholder: 0,
                        floor: 0,
                        residue: 0,
                    },
                    crate::setup::MappingSubmap {
                        time_placeholder: 0,
                        floor: 1,
                        residue: 0,
                    },
                ],
            },
            4,
            2,
            1,
        );
    }

    // ----------------------------------------------------------------
    // Mapping header writer — rejection variants.
    // ----------------------------------------------------------------

    #[test]
    fn mapping_rejects_unsupported_mapping_type() {
        let mut h = minimal_mono_mapping();
        h.mapping_type = 1;
        assert_eq!(
            write_mapping_header(&h, 1, 1, 1),
            Err(WriteMappingError::UnsupportedMappingType(1))
        );
    }

    #[test]
    fn mapping_rejects_zero_audio_channels() {
        let h = minimal_mono_mapping();
        assert_eq!(
            write_mapping_header(&h, 0, 1, 1),
            Err(WriteMappingError::ZeroAudioChannels)
        );
    }

    #[test]
    fn mapping_rejects_submaps_zero() {
        let mut h = minimal_mono_mapping();
        h.submaps = 0;
        h.submap_configs.clear();
        assert_eq!(
            write_mapping_header(&h, 1, 1, 1),
            Err(WriteMappingError::SubmapsOutOfRange(0))
        );
    }

    #[test]
    fn mapping_rejects_submaps_above_cap() {
        let mut h = minimal_mono_mapping();
        h.submaps = 17;
        assert_eq!(
            write_mapping_header(&h, 1, 1, 1),
            Err(WriteMappingError::SubmapsOutOfRange(17))
        );
    }

    #[test]
    fn mapping_rejects_coupling_steps_overflow() {
        let coupling: Vec<MappingCouplingStep> = (0..257)
            .map(|_| MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            })
            .collect();
        let h = MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling,
            mux: Vec::new(),
            submap_configs: vec![crate::setup::MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        };
        assert_eq!(
            write_mapping_header(&h, 2, 1, 1),
            Err(WriteMappingError::CouplingStepsOverflow(257))
        );
    }

    #[test]
    fn mapping_rejects_bad_coupling_magnitude_eq_angle() {
        let h = MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: vec![MappingCouplingStep {
                magnitude_channel: 1,
                angle_channel: 1,
            }],
            mux: Vec::new(),
            submap_configs: vec![crate::setup::MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        };
        assert_eq!(
            write_mapping_header(&h, 2, 1, 1),
            Err(WriteMappingError::BadCouplingChannels {
                step_index: 0,
                magnitude_channel: 1,
                angle_channel: 1,
                audio_channels: 2,
            })
        );
    }

    #[test]
    fn mapping_rejects_bad_coupling_channel_out_of_range() {
        let h = MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: vec![MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 2, // >= audio_channels = 2
            }],
            mux: Vec::new(),
            submap_configs: vec![crate::setup::MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        };
        assert_eq!(
            write_mapping_header(&h, 2, 1, 1),
            Err(WriteMappingError::BadCouplingChannels {
                step_index: 0,
                magnitude_channel: 0,
                angle_channel: 2,
                audio_channels: 2,
            })
        );
    }

    /// audio_channels == 1 ⇒ channel_bits = 0; any coupling step
    /// triggers magnitude == angle == 0 which the per-step validator
    /// catches. This pins the "no coupling legal on mono" path.
    #[test]
    fn mapping_rejects_coupling_on_mono_audio_channels() {
        let h = MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: vec![MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 0,
            }],
            mux: Vec::new(),
            submap_configs: vec![crate::setup::MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        };
        assert_eq!(
            write_mapping_header(&h, 1, 1, 1),
            Err(WriteMappingError::BadCouplingChannels {
                step_index: 0,
                magnitude_channel: 0,
                angle_channel: 0,
                audio_channels: 1,
            })
        );
    }

    #[test]
    fn mapping_rejects_mux_length_mismatch_when_submaps_gt_1() {
        let h = MappingHeader {
            mapping_type: 0,
            submaps: 2,
            coupling: Vec::new(),
            // expected len = audio_channels = 2, supplied len = 3.
            mux: vec![0, 1, 0],
            submap_configs: vec![
                crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                },
                crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                },
            ],
        };
        assert_eq!(
            write_mapping_header(&h, 2, 1, 1),
            Err(WriteMappingError::MuxLengthMismatch {
                expected: 2,
                actual: 3,
            })
        );
    }

    #[test]
    fn mapping_rejects_mux_present_when_submaps_eq_1() {
        let h = MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: Vec::new(),
            mux: vec![0], // must be empty when submaps == 1
            submap_configs: vec![crate::setup::MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        };
        assert_eq!(
            write_mapping_header(&h, 1, 1, 1),
            Err(WriteMappingError::MuxLengthMismatch {
                expected: 0,
                actual: 1,
            })
        );
    }

    #[test]
    fn mapping_rejects_bad_mux_value() {
        let h = MappingHeader {
            mapping_type: 0,
            submaps: 2,
            coupling: Vec::new(),
            mux: vec![0, 2], // 2 >= submaps=2
            submap_configs: vec![
                crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                },
                crate::setup::MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                },
            ],
        };
        assert_eq!(
            write_mapping_header(&h, 2, 1, 1),
            Err(WriteMappingError::BadMuxValue {
                channel_index: 1,
                mux: 2,
                submaps: 2,
            })
        );
    }

    #[test]
    fn mapping_rejects_submap_count_mismatch() {
        let h = MappingHeader {
            mapping_type: 0,
            submaps: 2,
            coupling: Vec::new(),
            mux: vec![0, 1],
            // Only one submap configuration for submaps=2.
            submap_configs: vec![crate::setup::MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        };
        assert_eq!(
            write_mapping_header(&h, 2, 1, 1),
            Err(WriteMappingError::SubmapCountMismatch {
                submaps: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn mapping_rejects_bad_submap_floor() {
        let mut h = minimal_mono_mapping();
        h.submap_configs[0].floor = 1; // floor_count = 1 → 1 is illegal
        assert_eq!(
            write_mapping_header(&h, 1, 1, 1),
            Err(WriteMappingError::BadSubmapFloor {
                submap_index: 0,
                floor: 1,
                floor_count: 1,
            })
        );
    }

    #[test]
    fn mapping_rejects_bad_submap_residue() {
        let mut h = minimal_mono_mapping();
        h.submap_configs[0].residue = 5; // residue_count = 1 → 5 is illegal
        assert_eq!(
            write_mapping_header(&h, 1, 1, 1),
            Err(WriteMappingError::BadSubmapResidue {
                submap_index: 0,
                residue: 5,
                residue_count: 1,
            })
        );
    }

    /// `WriteMappingError::Display` is non-empty for every variant
    /// and echoes the §4.2.4 / mapping context tag.
    #[test]
    fn mapping_error_display_smoke() {
        let cases = [
            WriteMappingError::UnsupportedMappingType(7),
            WriteMappingError::ZeroAudioChannels,
            WriteMappingError::SubmapsOutOfRange(0),
            WriteMappingError::SubmapsOutOfRange(17),
            WriteMappingError::CouplingStepsOverflow(300),
            WriteMappingError::BadCouplingChannels {
                step_index: 0,
                magnitude_channel: 1,
                angle_channel: 1,
                audio_channels: 2,
            },
            WriteMappingError::CouplingChannelOverflow {
                step_index: 0,
                channel: 8,
                channel_bits: 3,
            },
            WriteMappingError::MuxLengthMismatch {
                expected: 2,
                actual: 3,
            },
            WriteMappingError::BadMuxValue {
                channel_index: 0,
                mux: 5,
                submaps: 2,
            },
            WriteMappingError::SubmapCountMismatch {
                submaps: 2,
                actual: 1,
            },
            WriteMappingError::BadSubmapFloor {
                submap_index: 0,
                floor: 1,
                floor_count: 1,
            },
            WriteMappingError::BadSubmapResidue {
                submap_index: 0,
                residue: 5,
                residue_count: 1,
            },
        ];
        for case in cases {
            let s = format!("{case}");
            assert!(!s.is_empty(), "Display empty for {case:?}");
            assert!(
                s.contains("mapping"),
                "Display for {case:?} must mention mapping context"
            );
        }
    }

    // ----------------------------------------------------------------
    // Mapping header writer — WriteError glue.
    // ----------------------------------------------------------------

    /// Mapping write errors surface as `WriteError::Mapping` via the
    /// `From` impl, preserving the variant for caller inspection and
    /// the source chain.
    #[test]
    fn write_error_mapping_glue() {
        let inner: WriteError = WriteMappingError::UnsupportedMappingType(3).into();
        assert_eq!(
            inner,
            WriteError::Mapping(WriteMappingError::UnsupportedMappingType(3))
        );
        use std::error::Error as StdError;
        let src = StdError::source(&inner).expect("source chain should reach Mapping");
        // The chained Display must non-trivially echo the mapping tag.
        assert!(format!("{src}").contains("mapping"));
    }

    // ----------------------------------------------------------------
    // Mapping header writer — splice point.
    // ----------------------------------------------------------------

    /// The `write_mapping_header_into_writer` splice point appends to
    /// an in-progress writer at the current bit offset, leaving
    /// pre-existing bits intact. Pins the shape the setup-header
    /// writer will splice the mapping body into.
    #[test]
    fn mapping_into_writer_splice_appends_after_existing_bits() {
        let mut w = BitWriterLsb::with_capacity(16);
        // Seed: write 7 bits (sub-byte) before splicing.
        w.write_u32(0b1010101, 7);
        write_mapping_header_into_writer(&minimal_mono_mapping(), 1, 1, 1, &mut w).expect("splice");
        let with_splice = w.finish();

        // Standalone bytes, for comparison.
        let standalone =
            write_mapping_header(&minimal_mono_mapping(), 1, 1, 1).expect("standalone");
        assert!(with_splice.len() >= standalone.len());

        // Re-decode the spliced output: consume the 7 seed bits then
        // run the mapping parser at the resumed position.
        let mut r = oxideav_core::bits::BitReaderLsb::new(&with_splice);
        let seed = r.read_u32(7).unwrap();
        assert_eq!(seed, 0b1010101);
        let parsed = local_parse_mapping_for_tests(&mut r, 1);
        assert_eq!(parsed, minimal_mono_mapping());
    }

    /// The fail-closed gate: when the writer rejects a header, no
    /// bits are appended to the supplied splice writer. Pins the
    /// "validate before emit" contract documented on
    /// [`write_mapping_header_into_writer`].
    #[test]
    fn mapping_into_writer_splice_emits_no_bits_on_error() {
        let mut w = BitWriterLsb::with_capacity(16);
        // Seed 3 bits to give the writer some pre-existing state.
        w.write_u32(0b101, 3);
        let before = w.bit_position();

        let mut bad = minimal_mono_mapping();
        bad.mapping_type = 9; // §4.2.4 step 2b only defines 0

        let err = write_mapping_header_into_writer(&bad, 1, 1, 1, &mut w)
            .expect_err("must reject unsupported mapping_type");
        assert_eq!(err, WriteMappingError::UnsupportedMappingType(9));

        // bit_position unchanged: the gate ran before any write call.
        assert_eq!(w.bit_position(), before);
    }

    // ================================================================
    // Mode header writer — §4.2.4 "Modes" (round 26, umbrella round 228).
    // ================================================================

    /// Stand-alone bit-level mode reader, exercising the §4.2.4 step
    /// 2a–2d field order without dragging in the wider
    /// `parse_setup_header` walker. Mirrors the splice-point + roundtrip
    /// pattern used for codebook / floor / residue / mapping.
    fn local_parse_mode_for_tests(
        reader: &mut oxideav_core::bits::BitReaderLsb<'_>,
        _mapping_count: usize,
    ) -> ModeHeader {
        let blockflag = reader.read_bit().unwrap();
        let windowtype = reader.read_u32(16).unwrap() as u16;
        let transformtype = reader.read_u32(16).unwrap() as u16;
        let mapping = reader.read_u32(8).unwrap() as u8;
        ModeHeader {
            blockflag,
            windowtype,
            transformtype,
            mapping,
        }
    }

    /// The "minimal" mode configuration: short block, mapping index 0.
    /// Pins the densest §4.2.4 emit (every gate at its all-zeros legal
    /// value).
    fn minimal_short_mode() -> ModeHeader {
        ModeHeader {
            blockflag: false,
            windowtype: 0,
            transformtype: 0,
            mapping: 0,
        }
    }

    /// The companion "long-block" mode at mapping index 1. Used to pin
    /// the `blockflag = 1` bit position and the `mapping` byte position.
    fn minimal_long_mode() -> ModeHeader {
        ModeHeader {
            blockflag: true,
            windowtype: 0,
            transformtype: 0,
            mapping: 1,
        }
    }

    fn mode_roundtrips(header: &ModeHeader, mapping_count: usize) {
        let bytes = write_mode_header(header, mapping_count).expect("write must succeed");
        let mut reader = oxideav_core::bits::BitReaderLsb::new(&bytes);
        let parsed = local_parse_mode_for_tests(&mut reader, mapping_count);
        assert_eq!(&parsed, header, "mode roundtrip equality");
    }

    /// Pin the §4.2.4 byte layout for the minimal short-block fixture.
    /// 41 bits total → 6 bytes with 7 bits of §2.1.8 zero padding in
    /// the final byte.
    #[test]
    fn mode_byte_shape_short_block_minimal() {
        let bytes = write_mode_header(&minimal_short_mode(), 1).expect("must build");
        // Bits emitted (LSB-first per §2.1.4):
        //   blockflag=0      ->  1 bit
        //   windowtype=0     -> 16 bits
        //   transformtype=0  -> 16 bits
        //   mapping=0        ->  8 bits
        // Total = 1 + 16 + 16 + 8 = 41 bits = 6 bytes (with 7 bits of
        // §2.1.8 zero padding in the final byte).
        let total_bits = 1 + 16 + 16 + 8;
        assert_eq!(bytes.len(), (total_bits as usize).div_ceil(8));

        let mut expected = BitWriterLsb::with_capacity(8);
        expected.write_bit(false); // blockflag
        expected.write_u32(0, 16); // windowtype
        expected.write_u32(0, 16); // transformtype
        expected.write_u32(0, 8); // mapping
        assert_eq!(bytes, expected.finish());
    }

    /// Pin the §4.2.4 byte layout for the long-block fixture at
    /// `mapping = 1`. The `blockflag = 1` shows up in the LSB of the
    /// first byte; the 8-bit `mapping = 1` shows up at the well-defined
    /// bit offset 33.
    #[test]
    fn mode_byte_shape_long_block_mapping1() {
        let bytes = write_mode_header(&minimal_long_mode(), 2).expect("must build");
        let total_bits = 1 + 16 + 16 + 8;
        assert_eq!(bytes.len(), (total_bits as usize).div_ceil(8));

        // Re-decode the body and confirm both the per-field structure
        // and the LSB-first packing.
        let mut reader = oxideav_core::bits::BitReaderLsb::new(&bytes);
        assert!(reader.read_bit().unwrap(), "blockflag bit must be set");
        assert_eq!(reader.read_u32(16).unwrap(), 0, "windowtype must be 0");
        assert_eq!(reader.read_u32(16).unwrap(), 0, "transformtype must be 0");
        assert_eq!(reader.read_u32(8).unwrap(), 1, "mapping must be 1");
    }

    /// Fixed-width body: 41 bits regardless of mapping-table size.
    /// Cross-check that the byte length does not vary with
    /// `mapping_count`.
    #[test]
    fn mode_bit_length_is_constant_41_bits() {
        for mapping_count in [1usize, 2, 7, 32, 255] {
            let h = ModeHeader {
                blockflag: false,
                windowtype: 0,
                transformtype: 0,
                mapping: (mapping_count - 1) as u8,
            };
            let bytes = write_mode_header(&h, mapping_count).expect("write must succeed");
            assert_eq!(
                bytes.len(),
                6,
                "41 bits → 6 bytes for mapping_count={mapping_count}"
            );
        }
    }

    #[test]
    fn mode_roundtrips_short_block() {
        mode_roundtrips(&minimal_short_mode(), 1);
    }

    #[test]
    fn mode_roundtrips_long_block_mapping1() {
        mode_roundtrips(&minimal_long_mode(), 2);
    }

    #[test]
    fn mode_roundtrips_full_legal_mapping_range() {
        // 8-bit mapping field is exactly 256 possible values; exercise
        // the boundary at the table's high end.
        let h = ModeHeader {
            blockflag: true,
            windowtype: 0,
            transformtype: 0,
            mapping: 255,
        };
        mode_roundtrips(&h, 256);
    }

    #[test]
    fn mode_rejects_nonzero_windowtype() {
        let h = ModeHeader {
            blockflag: false,
            windowtype: 1,
            transformtype: 0,
            mapping: 0,
        };
        assert_eq!(
            write_mode_header(&h, 1),
            Err(WriteModeError::NonZeroWindowType(1))
        );
    }

    #[test]
    fn mode_rejects_nonzero_transformtype() {
        let h = ModeHeader {
            blockflag: false,
            windowtype: 0,
            transformtype: 2,
            mapping: 0,
        };
        assert_eq!(
            write_mode_header(&h, 1),
            Err(WriteModeError::NonZeroTransformType(2))
        );
    }

    #[test]
    fn mode_rejects_mapping_equal_to_count() {
        // mapping == mapping_count is the first illegal value.
        let h = ModeHeader {
            blockflag: false,
            windowtype: 0,
            transformtype: 0,
            mapping: 1,
        };
        assert_eq!(
            write_mode_header(&h, 1),
            Err(WriteModeError::BadMapping {
                mapping: 1,
                mapping_count: 1,
            })
        );
    }

    #[test]
    fn mode_rejects_mapping_far_above_count() {
        let h = ModeHeader {
            blockflag: false,
            windowtype: 0,
            transformtype: 0,
            mapping: 200,
        };
        assert_eq!(
            write_mode_header(&h, 4),
            Err(WriteModeError::BadMapping {
                mapping: 200,
                mapping_count: 4,
            })
        );
    }

    /// `WriteError::Mode(e)` carries the inner enum through the
    /// `From` impl, preserving the variant for caller inspection and
    /// the source chain.
    #[test]
    fn write_error_mode_glue() {
        let inner: WriteError = WriteModeError::NonZeroWindowType(5).into();
        assert_eq!(
            inner,
            WriteError::Mode(WriteModeError::NonZeroWindowType(5))
        );
        use std::error::Error as StdError;
        let src = StdError::source(&inner).expect("source chain should reach Mode");
        // The chained Display must non-trivially echo the mode tag.
        assert!(format!("{src}").contains("mode"));
    }

    /// `WriteError::Mode` Display goes through the inner enum.
    #[test]
    fn write_error_mode_display_forwards_inner() {
        let err: WriteError = WriteModeError::BadMapping {
            mapping: 9,
            mapping_count: 2,
        }
        .into();
        let s = format!("{err}");
        assert!(s.contains("mapping=9"));
        assert!(s.contains("mapping_count=2"));
    }

    // ----------------------------------------------------------------
    // Mode header writer — splice point.
    // ----------------------------------------------------------------

    /// The `write_mode_header_into_writer` splice point appends to an
    /// in-progress writer at the current bit offset, leaving
    /// pre-existing bits intact. Pins the shape the setup-header
    /// writer will splice the mode body into.
    #[test]
    fn mode_into_writer_splice_appends_after_existing_bits() {
        let mut w = BitWriterLsb::with_capacity(8);
        // Seed: write 11 bits (sub-byte, crosses byte boundary) before
        // splicing.
        w.write_u32(0b101_1010_0101, 11);
        write_mode_header_into_writer(&minimal_long_mode(), 2, &mut w).expect("splice");
        let with_splice = w.finish();

        // Standalone bytes, for comparison.
        let standalone = write_mode_header(&minimal_long_mode(), 2).expect("standalone");
        assert!(with_splice.len() >= standalone.len());

        // Re-decode the spliced output: consume the 11 seed bits then
        // run the mode parser at the resumed position.
        let mut r = oxideav_core::bits::BitReaderLsb::new(&with_splice);
        let seed = r.read_u32(11).unwrap();
        assert_eq!(seed, 0b101_1010_0101);
        let parsed = local_parse_mode_for_tests(&mut r, 2);
        assert_eq!(parsed, minimal_long_mode());
    }

    /// The fail-closed gate: when the writer rejects a header, no
    /// bits are appended to the supplied splice writer. Pins the
    /// "validate before emit" contract documented on
    /// [`write_mode_header_into_writer`].
    #[test]
    fn mode_into_writer_splice_emits_no_bits_on_error() {
        let mut w = BitWriterLsb::with_capacity(8);
        // Seed 5 bits to give the writer some pre-existing state.
        w.write_u32(0b1_0101, 5);
        let before = w.bit_position();

        let bad = ModeHeader {
            blockflag: false,
            windowtype: 7, // §4.2.4 step 2e only defines 0
            transformtype: 0,
            mapping: 0,
        };

        let err = write_mode_header_into_writer(&bad, 1, &mut w)
            .expect_err("must reject nonzero windowtype");
        assert_eq!(err, WriteModeError::NonZeroWindowType(7));

        // bit_position unchanged: the gate ran before any write call.
        assert_eq!(w.bit_position(), before);
    }

    /// The splice point also rejects `mapping >= mapping_count` without
    /// emitting any bits, the same fail-closed semantics as the
    /// stand-alone writer.
    #[test]
    fn mode_into_writer_splice_emits_no_bits_on_bad_mapping() {
        let mut w = BitWriterLsb::with_capacity(8);
        w.write_u32(0b11, 2);
        let before = w.bit_position();

        let bad = ModeHeader {
            blockflag: false,
            windowtype: 0,
            transformtype: 0,
            mapping: 4,
        };

        let err = write_mode_header_into_writer(&bad, 4, &mut w)
            .expect_err("must reject mapping == mapping_count");
        assert_eq!(
            err,
            WriteModeError::BadMapping {
                mapping: 4,
                mapping_count: 4,
            }
        );
        assert_eq!(w.bit_position(), before);
    }

    // ================================================================
    // Setup header (round 27) — wrapping §4.2.4 packet writer.
    // ================================================================

    use crate::setup::{
        parse_setup_header, FloorHeader, MappingSubmap, ParseError as SetupParseError,
    };

    /// The smallest valid mono setup header that satisfies every
    /// §4.2.4 invariant: exactly one codebook, one zero
    /// time-placeholder, one floor-1 entry, one residue-2 entry, one
    /// mapping (mono, no coupling, single submap), one short-block
    /// mode pointing at the single mapping, and the framing flag set.
    fn minimal_mono_setup() -> VorbisSetupHeader {
        VorbisSetupHeader {
            codebooks: vec![VorbisCodebook {
                dimensions: 1,
                entries: 8,
                codeword_lengths: vec![2, 4, 4, 4, 4, 2, 3, 3],
                lookup: VqLookup::None,
            }],
            time_placeholders: vec![0],
            floors: vec![FloorHeader {
                floor_type: 1,
                kind: FloorKind::Type1(Floor1Header {
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
                }),
            }],
            residues: vec![ResidueHeader {
                residue_type: 2,
                residue_begin: 0,
                residue_end: 128,
                partition_size: 32,
                classifications: 1,
                classbook: 0,
                cascade: vec![1],
                books: vec![[Some(0), None, None, None, None, None, None, None]],
            }],
            mappings: vec![MappingHeader {
                mapping_type: 0,
                submaps: 1,
                coupling: Vec::new(),
                mux: Vec::new(),
                submap_configs: vec![MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                }],
            }],
            modes: vec![ModeHeader {
                blockflag: false,
                windowtype: 0,
                transformtype: 0,
                mapping: 0,
            }],
            framing_flag: true,
        }
    }

    /// Bit-exact roundtrip helper: writer → full parser → equal.
    fn setup_roundtrips(header: &VorbisSetupHeader, audio_channels: u8) {
        let packet = write_setup_header(header, audio_channels).expect("write must succeed");
        let parsed = parse_setup_header(&packet, audio_channels).expect("parse must succeed");
        assert_eq!(&parsed, header, "setup roundtrip equality");
    }

    /// Pin the 7-byte common-header prefix at the head of every setup
    /// packet the writer emits, mirroring the §4.2.1 fixture shape
    /// `parse_setup_header` validates.
    #[test]
    fn setup_byte_shape_common_header_prefix() {
        let packet = write_setup_header(&minimal_mono_setup(), 1).expect("must build minimal mono");
        assert!(packet.len() >= 7, "packet shorter than §4.2.1 header");
        assert_eq!(packet[0], 0x05, "packet_type must be SETUP_PACKET_TYPE");
        assert_eq!(&packet[1..7], b"vorbis", "magic must be ASCII 'vorbis'");
    }

    /// Pin the minimal-mono fixture: writing then parsing recovers the
    /// exact same setup-header struct (codebooks, time placeholders,
    /// floors, residues, mappings, modes, framing flag).
    #[test]
    fn setup_roundtrips_minimal_mono() {
        setup_roundtrips(&minimal_mono_setup(), 1);
    }

    /// Two-codebook variant: the writer must emit `count - 1` = 1 in
    /// the 8-bit codebook-count field, then both codebook bodies in
    /// order. Roundtrips the order through the parser.
    #[test]
    fn setup_roundtrips_two_codebooks() {
        let mut header = minimal_mono_setup();
        header.codebooks.push(VorbisCodebook {
            dimensions: 1,
            entries: 4,
            codeword_lengths: vec![1, 2, 3, 3],
            lookup: VqLookup::None,
        });
        setup_roundtrips(&header, 1);
    }

    /// Time-placeholder count at the 6-bit upper edge (64 entries).
    /// The §4.2.4 "Time domain transforms" container is `read 6 bits
    /// + 1`, so 64 entries means a wire value of 63.
    #[test]
    fn setup_roundtrips_max_time_placeholders() {
        let mut header = minimal_mono_setup();
        header.time_placeholders = vec![0u16; 64];
        setup_roundtrips(&header, 1);
    }

    /// Both floor kinds: a type-0 floor followed by a type-1 floor.
    /// Exercises the `floor_type` 16-bit selector branching inside
    /// the writer's per-floor loop.
    #[test]
    fn setup_roundtrips_mixed_floor_kinds() {
        let mut header = minimal_mono_setup();
        header.floors.push(FloorHeader {
            floor_type: 0,
            kind: FloorKind::Type0(Floor0Header {
                order: 4,
                rate: 44100,
                bark_map_size: 64,
                amplitude_bits: 8,
                amplitude_offset: 100,
                book_list: vec![0],
            }),
        });
        // The minimal_mono mapping points at floor 0 (type 1); both
        // floor entries remain reachable through the floor-count
        // accumulator. Add a second mapping that points at floor 1
        // (type 0) so the round-trip exercises both branches.
        header.mappings.push(MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: Vec::new(),
            mux: Vec::new(),
            submap_configs: vec![MappingSubmap {
                time_placeholder: 0,
                floor: 1,
                residue: 0,
            }],
        });
        header.modes.push(ModeHeader {
            blockflag: true,
            windowtype: 0,
            transformtype: 0,
            mapping: 1,
        });
        setup_roundtrips(&header, 1);
    }

    /// All three residue types in a single packet — exercises the
    /// 16-bit `residue_type` selector across the {0, 1, 2} space.
    #[test]
    fn setup_roundtrips_mixed_residue_types() {
        let mut header = minimal_mono_setup();
        for rtype in [0u16, 1u16] {
            header.residues.push(ResidueHeader {
                residue_type: rtype,
                residue_begin: 0,
                residue_end: 128,
                partition_size: 32,
                classifications: 1,
                classbook: 0,
                cascade: vec![1],
                books: vec![[Some(0), None, None, None, None, None, None, None]],
            });
        }
        setup_roundtrips(&header, 1);
    }

    /// Stereo-coupled mapping at `audio_channels = 2`: exercises the
    /// `ilog(audio_channels - 1)` per-coupling-step field width that
    /// the nested mapping writer's context tuple consumes.
    #[test]
    fn setup_roundtrips_stereo_coupled_mapping() {
        let mut header = minimal_mono_setup();
        header.mappings[0] = MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: vec![MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            }],
            mux: Vec::new(),
            submap_configs: vec![MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        };
        setup_roundtrips(&header, 2);
    }

    /// Two-mode variant (one short, one long) — pins the mode-count
    /// field + the per-mode `mapping_count` context being passed
    /// through to the splice writer.
    #[test]
    fn setup_roundtrips_two_modes() {
        let mut header = minimal_mono_setup();
        header.modes.push(ModeHeader {
            blockflag: true,
            windowtype: 0,
            transformtype: 0,
            mapping: 0,
        });
        setup_roundtrips(&header, 1);
    }

    /// The writer rejects the call rather than emit a packet the
    /// parser would reject. Each error variant below pins a §4.2.4
    /// invariant the writer enforces fail-closed.
    #[test]
    fn setup_rejects_zero_audio_channels() {
        let err = write_setup_header(&minimal_mono_setup(), 0)
            .expect_err("must reject audio_channels = 0");
        assert_eq!(err, WriteError::Setup(WriteSetupError::ZeroAudioChannels));
    }

    #[test]
    fn setup_rejects_empty_codebooks() {
        let mut header = minimal_mono_setup();
        header.codebooks.clear();
        let err = write_setup_header(&header, 1).expect_err("must reject empty codebooks");
        assert_eq!(err, WriteError::Setup(WriteSetupError::EmptyCodebooks));
    }

    #[test]
    fn setup_rejects_codebook_count_overflow() {
        // 257 codebooks would need to encode count - 1 = 256, which
        // exceeds the 8-bit container.
        let mut header = minimal_mono_setup();
        let template = header.codebooks[0].clone();
        header.codebooks = vec![template; 257];
        let err = write_setup_header(&header, 1).expect_err("must reject codebook count = 257");
        assert_eq!(
            err,
            WriteError::Setup(WriteSetupError::CodebookCountOverflow(257))
        );
    }

    #[test]
    fn setup_rejects_empty_time_placeholders() {
        let mut header = minimal_mono_setup();
        header.time_placeholders.clear();
        let err = write_setup_header(&header, 1).expect_err("must reject empty time placeholders");
        assert_eq!(
            err,
            WriteError::Setup(WriteSetupError::EmptyTimePlaceholders)
        );
    }

    #[test]
    fn setup_rejects_time_count_overflow() {
        let mut header = minimal_mono_setup();
        header.time_placeholders = vec![0u16; 65];
        let err = write_setup_header(&header, 1).expect_err("must reject time count = 65");
        assert_eq!(
            err,
            WriteError::Setup(WriteSetupError::TimeCountOverflow(65))
        );
    }

    #[test]
    fn setup_rejects_nonzero_time_placeholder() {
        let mut header = minimal_mono_setup();
        header.time_placeholders = vec![0, 7];
        let err = write_setup_header(&header, 1).expect_err("must reject nonzero time placeholder");
        assert_eq!(
            err,
            WriteError::Setup(WriteSetupError::NonZeroTimePlaceholder { index: 1, value: 7 })
        );
    }

    #[test]
    fn setup_rejects_empty_floors() {
        let mut header = minimal_mono_setup();
        header.floors.clear();
        let err = write_setup_header(&header, 1).expect_err("must reject empty floors");
        assert_eq!(err, WriteError::Setup(WriteSetupError::EmptyFloors));
    }

    #[test]
    fn setup_rejects_floor_count_overflow() {
        let mut header = minimal_mono_setup();
        let template = header.floors[0].clone();
        header.floors = vec![template; 65];
        let err = write_setup_header(&header, 1).expect_err("must reject floor count = 65");
        assert_eq!(
            err,
            WriteError::Setup(WriteSetupError::FloorCountOverflow(65))
        );
    }

    #[test]
    fn setup_rejects_unsupported_floor_type() {
        let mut header = minimal_mono_setup();
        header.floors[0].floor_type = 2;
        let err = write_setup_header(&header, 1).expect_err("must reject floor_type = 2");
        assert_eq!(
            err,
            WriteError::Setup(WriteSetupError::UnsupportedFloorType {
                index: 0,
                floor_type: 2
            })
        );
    }

    #[test]
    fn setup_rejects_floor_type_kind_mismatch() {
        let mut header = minimal_mono_setup();
        // `floor_type = 0` paired with a `FloorKind::Type1(_)` payload.
        header.floors[0].floor_type = 0;
        let err =
            write_setup_header(&header, 1).expect_err("must reject floor_type/kind disagreement");
        assert_eq!(
            err,
            WriteError::Setup(WriteSetupError::FloorTypeKindMismatch {
                index: 0,
                floor_type: 0,
                kind_discriminant: 1,
            })
        );
    }

    #[test]
    fn setup_rejects_empty_residues() {
        let mut header = minimal_mono_setup();
        header.residues.clear();
        let err = write_setup_header(&header, 1).expect_err("must reject empty residues");
        assert_eq!(err, WriteError::Setup(WriteSetupError::EmptyResidues));
    }

    #[test]
    fn setup_rejects_empty_mappings() {
        let mut header = minimal_mono_setup();
        header.mappings.clear();
        let err = write_setup_header(&header, 1).expect_err("must reject empty mappings");
        assert_eq!(err, WriteError::Setup(WriteSetupError::EmptyMappings));
    }

    #[test]
    fn setup_rejects_empty_modes() {
        let mut header = minimal_mono_setup();
        header.modes.clear();
        let err = write_setup_header(&header, 1).expect_err("must reject empty modes");
        assert_eq!(err, WriteError::Setup(WriteSetupError::EmptyModes));
    }

    #[test]
    fn setup_rejects_bad_framing_flag() {
        let mut header = minimal_mono_setup();
        header.framing_flag = false;
        let err = write_setup_header(&header, 1).expect_err("must reject framing_flag = false");
        assert_eq!(err, WriteError::Setup(WriteSetupError::BadFramingFlag));
    }

    /// A nested-block failure (e.g. a mode whose mapping index points
    /// past the mappings list) propagates up through the umbrella
    /// `WriteError` as the corresponding `WriteError::Mode(_)`
    /// variant. Confirms the `?` chain wires the nested writer's
    /// fail-closed semantics into the wrapping writer's contract.
    #[test]
    fn setup_propagates_nested_mode_error() {
        let mut header = minimal_mono_setup();
        // One mapping, but mode points at mapping index 5 → out of
        // range. The mode-body writer rejects with
        // `WriteModeError::BadMapping`.
        header.modes[0].mapping = 5;
        let err =
            write_setup_header(&header, 1).expect_err("must reject out-of-range mode mapping");
        match err {
            WriteError::Mode(WriteModeError::BadMapping {
                mapping,
                mapping_count,
            }) => {
                assert_eq!(mapping, 5);
                assert_eq!(mapping_count, 1);
            }
            _ => panic!("expected WriteError::Mode(BadMapping), got {err:?}"),
        }
    }

    /// A nested floor-1 failure (e.g. a subclass-book overflow) also
    /// propagates as the corresponding `WriteError::Floor1(_)`
    /// variant. Confirms the `?` chain wires every nested writer's
    /// fail-closed semantics into the wrapping writer's contract.
    #[test]
    fn setup_propagates_nested_floor1_error() {
        let mut header = minimal_mono_setup();
        // Mutate the floor-1 partitions field past its 5-bit range.
        if let FloorKind::Type1(ref mut body) = header.floors[0].kind {
            body.partitions = 32; // 5-bit field maxes at 31
        }
        let err = write_setup_header(&header, 1).expect_err("must reject partitions = 32");
        assert!(
            matches!(err, WriteError::Floor1(_)),
            "expected WriteError::Floor1, got {err:?}",
        );
    }

    /// The parser must round-trip every byte the writer emitted —
    /// confirms there is no slack at either end of the wire shape.
    /// Uses `parse_setup_header` (the full common-header + body
    /// path) rather than `parse_setup_header_body`.
    #[test]
    fn setup_parser_consumes_full_packet() {
        let header = minimal_mono_setup();
        let packet = write_setup_header(&header, 1).expect("must build");
        let parsed = parse_setup_header(&packet, 1).expect("must parse");
        assert_eq!(parsed, header);
    }

    /// The framing-bit `BadFramingFlag` rejection is gated before any
    /// emit so the writer never returns a partial byte stream.
    /// Confirmed by re-checking that the rejection produces zero
    /// bytes — there is no "partial packet" leak path.
    #[test]
    fn setup_bad_framing_flag_emits_nothing() {
        let mut header = minimal_mono_setup();
        header.framing_flag = false;
        let err = write_setup_header(&header, 1).expect_err("must reject");
        assert_eq!(err, WriteError::Setup(WriteSetupError::BadFramingFlag));
    }

    /// `WriteSetupError::Display` non-emptiness across every variant —
    /// the messages should each cite the §4.2.4 invariant they enforce.
    #[test]
    fn setup_error_display_nonempty_all_variants() {
        let cases = [
            WriteSetupError::ZeroAudioChannels,
            WriteSetupError::EmptyCodebooks,
            WriteSetupError::CodebookCountOverflow(300),
            WriteSetupError::EmptyTimePlaceholders,
            WriteSetupError::TimeCountOverflow(65),
            WriteSetupError::NonZeroTimePlaceholder { index: 0, value: 1 },
            WriteSetupError::EmptyFloors,
            WriteSetupError::FloorCountOverflow(65),
            WriteSetupError::UnsupportedFloorType {
                index: 0,
                floor_type: 2,
            },
            WriteSetupError::FloorTypeKindMismatch {
                index: 0,
                floor_type: 1,
                kind_discriminant: 0,
            },
            WriteSetupError::EmptyResidues,
            WriteSetupError::ResidueCountOverflow(65),
            WriteSetupError::EmptyMappings,
            WriteSetupError::MappingCountOverflow(65),
            WriteSetupError::EmptyModes,
            WriteSetupError::ModeCountOverflow(65),
            WriteSetupError::BadFramingFlag,
        ];
        for variant in &cases {
            let s = format!("{variant}");
            assert!(!s.is_empty(), "Display empty for {variant:?}");
            assert!(
                s.contains("vorbis setup header"),
                "Display did not name the setup writer: {s}"
            );
        }
    }

    /// `WriteError::Setup` `From` glue + `source()` chain — the
    /// umbrella `WriteError` delegates `Display` and `Error::source`
    /// to the inner enum for the new variant just like every other
    /// nested-writer error.
    #[test]
    fn setup_umbrella_write_error_glue() {
        let inner = WriteSetupError::BadFramingFlag;
        let umbrella: WriteError = inner.clone().into();
        assert_eq!(umbrella, WriteError::Setup(inner.clone()));
        assert_eq!(format!("{umbrella}"), format!("{inner}"));
        use std::error::Error as _;
        let src = umbrella.source().expect("source must be Some");
        // The source must downcast back to the inner enum.
        assert!(
            src.downcast_ref::<WriteSetupError>().is_some(),
            "source should downcast to WriteSetupError"
        );
    }

    /// Spot-check: the parser also rejects the packet shapes the writer
    /// refuses. Confirms the writer's invariant gate aligns with the
    /// round-5 parser's invariant gate (no silent disagreement between
    /// the two layers).
    #[test]
    fn setup_writer_invariants_align_with_parser() {
        // Hand-roll a packet with a nonzero time placeholder; the
        // parser must reject it with the matching variant.
        // (We construct the packet by writing a valid one and
        // patching the 16-bit time-placeholder field; the offset is
        // determined by the encoder layout.)
        let header = minimal_mono_setup();
        let packet = write_setup_header(&header, 1).expect("must build");
        // The parser rejects this same valid packet successfully.
        let parsed = parse_setup_header(&packet, 1).expect("baseline must parse");
        assert_eq!(parsed, header);
        // Now construct a fixture whose writer would refuse, and
        // verify the writer refuses it — i.e. there is no path
        // through which an invalid-shaped struct is silently emitted.
        let mut bad = header.clone();
        bad.time_placeholders[0] = 1;
        let bad_err = write_setup_header(&bad, 1).expect_err("writer must reject");
        match bad_err {
            WriteError::Setup(WriteSetupError::NonZeroTimePlaceholder { index, value }) => {
                assert_eq!(index, 0);
                assert_eq!(value, 1);
            }
            other => panic!("expected NonZeroTimePlaceholder rejection, got {other:?}"),
        }
        // Suppress the unused-import lint when no test in the module
        // mentions `SetupParseError` by name; it remains imported for
        // future fixture-specific assertions.
        let _ = std::marker::PhantomData::<SetupParseError>;
    }

    // ----------------------------------------------------------------
    // §4.3.1 audio-packet header — WRITE primitive.
    // ----------------------------------------------------------------

    fn ap_mode(blockflag: bool, mapping: u8) -> ModeHeader {
        ModeHeader {
            blockflag,
            windowtype: 0,
            transformtype: 0,
            mapping,
        }
    }

    fn ap_setup(modes: Vec<ModeHeader>) -> VorbisSetupHeader {
        VorbisSetupHeader {
            codebooks: Vec::new(),
            time_placeholders: Vec::new(),
            floors: Vec::new(),
            residues: Vec::new(),
            mappings: Vec::new(),
            modes,
            framing_flag: true,
        }
    }

    /// Round-trip: emit the prelude, parse it back through
    /// `read_packet_header`, struct must equal input. Single-mode short
    /// block — `ilog(0) == 0` collapses the mode-number field to zero
    /// bits, so total bits emitted = 1.
    #[test]
    fn audio_packet_header_single_mode_short_roundtrips() {
        use crate::packet::read_packet_header;
        use oxideav_core::bits::BitReaderLsb;

        let setup = ap_setup(vec![ap_mode(false, 0)]);
        let h = AudioPacketHeader {
            mode_number: 0,
            blockflag: false,
            n: 64,
            previous_window_flag: false,
            next_window_flag: false,
        };
        let bytes = write_audio_packet_header(&h, &setup, 64, 1024).unwrap();
        // §4.3.1 emits a single `packet_type` bit; §2.1.8 zero-pads
        // the final byte. The total slice is therefore one byte.
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0x00);

        let mut r = BitReaderLsb::new(&bytes);
        let parsed = read_packet_header(&mut r, &setup, 64, 1024).unwrap();
        assert_eq!(parsed, h);
        // Parser consumed exactly the one bit we emitted.
        assert_eq!(r.bit_position(), 1);
    }

    /// Round-trip: single-mode long block. `ilog(0)==0` collapses the
    /// mode-number field; the long block adds two 1-bit window flags.
    /// Total bits = 1 (packet_type) + 2 (window flags) = 3.
    #[test]
    fn audio_packet_header_single_mode_long_roundtrips() {
        use crate::packet::read_packet_header;
        use oxideav_core::bits::BitReaderLsb;

        let setup = ap_setup(vec![ap_mode(true, 0)]);
        let h = AudioPacketHeader {
            mode_number: 0,
            blockflag: true,
            n: 1024,
            previous_window_flag: true,
            next_window_flag: false,
        };
        let bytes = write_audio_packet_header(&h, &setup, 64, 1024).unwrap();
        assert_eq!(bytes.len(), 1);
        // bit 0: packet_type = 0
        // bit 1: previous_window_flag = 1
        // bit 2: next_window_flag     = 0
        // bits 3..8: §2.1.8 zero padding.
        // Packed LSB-first: 0b00000_010 = 0x02.
        assert_eq!(bytes[0], 0b0000_0010);

        let mut r = BitReaderLsb::new(&bytes);
        let parsed = read_packet_header(&mut r, &setup, 64, 1024).unwrap();
        assert_eq!(parsed, h);
        assert_eq!(r.bit_position(), 3);
    }

    /// Round-trip: two modes (`ilog(1) == 1`-bit `mode_number`).
    /// Selecting mode 1 (short) → 1 + 1 = 2 bits emitted.
    #[test]
    fn audio_packet_header_two_modes_short_roundtrips() {
        use crate::packet::read_packet_header;
        use oxideav_core::bits::BitReaderLsb;

        let setup = ap_setup(vec![ap_mode(true, 0), ap_mode(false, 0)]);
        let h = AudioPacketHeader {
            mode_number: 1,
            blockflag: false,
            n: 256,
            previous_window_flag: false,
            next_window_flag: false,
        };
        let bytes = write_audio_packet_header(&h, &setup, 256, 2048).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let parsed = read_packet_header(&mut r, &setup, 256, 2048).unwrap();
        assert_eq!(parsed, h);
        assert_eq!(r.bit_position(), 2);
    }

    /// Round-trip: two modes (`ilog(1) == 1`), pick mode 0 (long).
    /// Total bits = 1 + 1 + 2 = 4.
    #[test]
    fn audio_packet_header_two_modes_long_roundtrips() {
        use crate::packet::read_packet_header;
        use oxideav_core::bits::BitReaderLsb;

        let setup = ap_setup(vec![ap_mode(true, 0), ap_mode(false, 0)]);
        let h = AudioPacketHeader {
            mode_number: 0,
            blockflag: true,
            n: 2048,
            previous_window_flag: true,
            next_window_flag: true,
        };
        let bytes = write_audio_packet_header(&h, &setup, 256, 2048).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let parsed = read_packet_header(&mut r, &setup, 256, 2048).unwrap();
        assert_eq!(parsed, h);
        assert_eq!(r.bit_position(), 4);
    }

    /// Round-trip: three modes — `ilog(2) == 2`-bit `mode_number`.
    /// Pick mode 2 (long). Total bits = 1 + 2 + 2 = 5.
    #[test]
    fn audio_packet_header_three_modes_long_roundtrips() {
        use crate::packet::read_packet_header;
        use oxideav_core::bits::BitReaderLsb;

        let setup = ap_setup(vec![ap_mode(false, 0), ap_mode(false, 0), ap_mode(true, 0)]);
        let h = AudioPacketHeader {
            mode_number: 2,
            blockflag: true,
            n: 1024,
            previous_window_flag: false,
            next_window_flag: true,
        };
        let bytes = write_audio_packet_header(&h, &setup, 64, 1024).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let parsed = read_packet_header(&mut r, &setup, 64, 1024).unwrap();
        assert_eq!(parsed, h);
        assert_eq!(r.bit_position(), 5);
    }

    /// Byte-level pin: long block, 4 bits emitted, LSB-first packing.
    /// Two modes (1-bit mode_number), mode 0 (long).
    ///   bit 0: packet_type      = 0
    ///   bit 1: mode_number      = 0
    ///   bit 2: previous_window_flag = 1
    ///   bit 3: next_window_flag     = 1
    /// Byte (LSB-first): 0b0000_1100 = 0x0c.
    #[test]
    fn audio_packet_header_byte_shape_long_block_two_modes() {
        let setup = ap_setup(vec![ap_mode(true, 0), ap_mode(false, 0)]);
        let h = AudioPacketHeader {
            mode_number: 0,
            blockflag: true,
            n: 2048,
            previous_window_flag: true,
            next_window_flag: true,
        };
        let bytes = write_audio_packet_header(&h, &setup, 256, 2048).unwrap();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0b0000_1100);
    }

    /// Cross-verify against the parser-side test fixture pattern:
    /// `read_packet_header` test `packet_header_three_modes_uses_two_mode_bits`
    /// wrote `(packet_type, mode_number=2, prev=0, next=0)` and parsed
    /// to `(mode_number=2, blockflag=true, n=1024)`. The writer fed the
    /// same struct must emit the same bytes.
    #[test]
    fn audio_packet_header_matches_parser_fixture_three_modes() {
        use oxideav_core::bits::BitWriterLsb;

        let setup = ap_setup(vec![ap_mode(false, 0), ap_mode(false, 0), ap_mode(true, 0)]);
        let h = AudioPacketHeader {
            mode_number: 2,
            blockflag: true,
            n: 1024,
            previous_window_flag: false,
            next_window_flag: false,
        };
        let writer_bytes = write_audio_packet_header(&h, &setup, 64, 1024).unwrap();

        // The hand-rolled writer in the parser test:
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        w.write_u32(2, 2); // mode_number
        w.write_u32(0, 1); // previous_window_flag
        w.write_u32(0, 1); // next_window_flag
        let expected = w.finish();

        assert_eq!(writer_bytes, expected);
    }

    /// Reject: empty modes list.
    #[test]
    fn audio_packet_header_rejects_empty_mode_list() {
        let setup = ap_setup(Vec::new());
        let h = AudioPacketHeader {
            mode_number: 0,
            blockflag: false,
            n: 64,
            previous_window_flag: false,
            next_window_flag: false,
        };
        assert_eq!(
            write_audio_packet_header(&h, &setup, 64, 1024),
            Err(WriteAudioPacketHeaderError::EmptyModeList)
        );
    }

    /// Reject: mode_number indexes past the mode list.
    #[test]
    fn audio_packet_header_rejects_bad_mode_number() {
        let setup = ap_setup(vec![ap_mode(false, 0), ap_mode(false, 0)]);
        let h = AudioPacketHeader {
            mode_number: 2,
            blockflag: false,
            n: 64,
            previous_window_flag: false,
            next_window_flag: false,
        };
        assert_eq!(
            write_audio_packet_header(&h, &setup, 64, 1024),
            Err(WriteAudioPacketHeaderError::BadModeNumber {
                mode_number: 2,
                mode_count: 2,
            })
        );
    }

    /// Reject: cached blockflag disagrees with the selected mode.
    #[test]
    fn audio_packet_header_rejects_blockflag_mismatch() {
        let setup = ap_setup(vec![ap_mode(true, 0)]);
        // Caller says short block but the mode is long.
        let h = AudioPacketHeader {
            mode_number: 0,
            blockflag: false,
            n: 64,
            previous_window_flag: false,
            next_window_flag: false,
        };
        assert_eq!(
            write_audio_packet_header(&h, &setup, 64, 1024),
            Err(WriteAudioPacketHeaderError::BlockflagMismatch {
                header_blockflag: false,
                mode_blockflag: true,
            })
        );
    }

    /// Reject: cached `n` disagrees with `blocksize_0` on a short block.
    #[test]
    fn audio_packet_header_rejects_blocksize_mismatch_short() {
        let setup = ap_setup(vec![ap_mode(false, 0)]);
        let h = AudioPacketHeader {
            mode_number: 0,
            blockflag: false,
            n: 1024, // wrong: should be blocksize_0 = 64
            previous_window_flag: false,
            next_window_flag: false,
        };
        assert_eq!(
            write_audio_packet_header(&h, &setup, 64, 1024),
            Err(WriteAudioPacketHeaderError::BlocksizeMismatch {
                header_n: 1024,
                expected_n: 64,
            })
        );
    }

    /// Reject: cached `n` disagrees with `blocksize_1` on a long block.
    #[test]
    fn audio_packet_header_rejects_blocksize_mismatch_long() {
        let setup = ap_setup(vec![ap_mode(true, 0)]);
        let h = AudioPacketHeader {
            mode_number: 0,
            blockflag: true,
            n: 64, // wrong: should be blocksize_1 = 1024
            previous_window_flag: false,
            next_window_flag: false,
        };
        assert_eq!(
            write_audio_packet_header(&h, &setup, 64, 1024),
            Err(WriteAudioPacketHeaderError::BlocksizeMismatch {
                header_n: 64,
                expected_n: 1024,
            })
        );
    }

    /// Reject: short block carries a previous_window_flag (no on-wire
    /// bit pattern round-trips to a short block + a set window flag).
    #[test]
    fn audio_packet_header_rejects_short_block_previous_flag() {
        let setup = ap_setup(vec![ap_mode(false, 0)]);
        let h = AudioPacketHeader {
            mode_number: 0,
            blockflag: false,
            n: 64,
            previous_window_flag: true,
            next_window_flag: false,
        };
        assert_eq!(
            write_audio_packet_header(&h, &setup, 64, 1024),
            Err(WriteAudioPacketHeaderError::ShortBlockHasWindowFlag {
                previous_window_flag: true,
                next_window_flag: false,
            })
        );
    }

    /// Reject: short block carries a next_window_flag.
    #[test]
    fn audio_packet_header_rejects_short_block_next_flag() {
        let setup = ap_setup(vec![ap_mode(false, 0)]);
        let h = AudioPacketHeader {
            mode_number: 0,
            blockflag: false,
            n: 64,
            previous_window_flag: false,
            next_window_flag: true,
        };
        assert_eq!(
            write_audio_packet_header(&h, &setup, 64, 1024),
            Err(WriteAudioPacketHeaderError::ShortBlockHasWindowFlag {
                previous_window_flag: false,
                next_window_flag: true,
            })
        );
    }

    /// Confirm a short block with both window flags set is also
    /// rejected (single variant fires regardless of which flag the
    /// caller mis-cached).
    #[test]
    fn audio_packet_header_rejects_short_block_both_flags() {
        let setup = ap_setup(vec![ap_mode(false, 0)]);
        let h = AudioPacketHeader {
            mode_number: 0,
            blockflag: false,
            n: 64,
            previous_window_flag: true,
            next_window_flag: true,
        };
        assert_eq!(
            write_audio_packet_header(&h, &setup, 64, 1024),
            Err(WriteAudioPacketHeaderError::ShortBlockHasWindowFlag {
                previous_window_flag: true,
                next_window_flag: true,
            })
        );
    }

    /// Exhaustive roundtrip across the four (blockflag, prev, next)
    /// combinations on a single-mode long-block stream.
    #[test]
    fn audio_packet_header_exhaustive_long_block_window_flag_combinations() {
        use crate::packet::read_packet_header;
        use oxideav_core::bits::BitReaderLsb;

        let setup = ap_setup(vec![ap_mode(true, 0)]);
        for prev in [false, true] {
            for next in [false, true] {
                let h = AudioPacketHeader {
                    mode_number: 0,
                    blockflag: true,
                    n: 1024,
                    previous_window_flag: prev,
                    next_window_flag: next,
                };
                let bytes = write_audio_packet_header(&h, &setup, 64, 1024).expect("must build");
                let mut r = BitReaderLsb::new(&bytes);
                let parsed = read_packet_header(&mut r, &setup, 64, 1024).expect("must parse");
                assert_eq!(
                    parsed, h,
                    "round-trip failed for (prev={prev}, next={next})"
                );
            }
        }
    }

    /// Round-trip across all mode-count values 1..=64 with the maximum
    /// `mode_number` selected. Confirms the `ilog(mode_count - 1)` bit
    /// width matches the reader for every value of the 6-bit setup-
    /// header `mode_count - 1` field.
    #[test]
    fn audio_packet_header_roundtrips_for_all_mode_counts() {
        use crate::packet::read_packet_header;
        use oxideav_core::bits::BitReaderLsb;

        for mode_count in 1..=64usize {
            // Build a setup where the last mode is the chosen one and
            // is short (blockflag = false). All others are also short
            // so we don't have to track per-mode blockflags.
            let modes: Vec<ModeHeader> = (0..mode_count).map(|_| ap_mode(false, 0)).collect();
            let setup = ap_setup(modes);
            let h = AudioPacketHeader {
                mode_number: (mode_count - 1) as u32,
                blockflag: false,
                n: 64,
                previous_window_flag: false,
                next_window_flag: false,
            };
            let bytes = write_audio_packet_header(&h, &setup, 64, 1024).unwrap();
            let mut r = BitReaderLsb::new(&bytes);
            let parsed = read_packet_header(&mut r, &setup, 64, 1024).unwrap();
            assert_eq!(parsed, h, "round-trip failed at mode_count={mode_count}");
        }
    }

    /// `WriteError::AudioPacket` glue: a writer-side rejection bubbles
    /// up through the umbrella `WriteError` via the `From` impl with
    /// the right `source()` chain.
    #[test]
    fn audio_packet_header_umbrella_write_error_glue() {
        use std::error::Error as StdError;

        let inner = WriteAudioPacketHeaderError::EmptyModeList;
        let wrapped: WriteError = inner.into();
        match wrapped {
            WriteError::AudioPacket(WriteAudioPacketHeaderError::EmptyModeList) => {}
            other => panic!("expected WriteError::AudioPacket(EmptyModeList), got {other:?}"),
        }
        let displayed = format!("{wrapped}");
        assert!(displayed.contains("setup.modes is empty"));
        let src = wrapped.source().expect("source must be set");
        assert!(format!("{src}").contains("setup.modes is empty"));
    }

    /// `WriteAudioPacketHeaderError::Display` non-emptiness across all
    /// five variants — the §-prefixed strings are part of the public
    /// diagnostic surface.
    #[test]
    fn audio_packet_header_error_display_nonempty_all_variants() {
        let variants = [
            WriteAudioPacketHeaderError::EmptyModeList,
            WriteAudioPacketHeaderError::BadModeNumber {
                mode_number: 2,
                mode_count: 2,
            },
            WriteAudioPacketHeaderError::BlockflagMismatch {
                header_blockflag: false,
                mode_blockflag: true,
            },
            WriteAudioPacketHeaderError::BlocksizeMismatch {
                header_n: 64,
                expected_n: 1024,
            },
            WriteAudioPacketHeaderError::ShortBlockHasWindowFlag {
                previous_window_flag: true,
                next_window_flag: false,
            },
        ];
        for v in variants {
            let s = format!("{v}");
            assert!(s.starts_with("vorbis audio packet (write):"));
            assert!(!s.is_empty());
        }
    }

    /// Writer invariants align with the parser side: a struct the
    /// writer refuses (out-of-range mode_number) — when we hand-roll
    /// the equivalent bit pattern — is also refused by the parser
    /// with the matching `PacketError` variant.
    #[test]
    fn audio_packet_header_writer_invariants_align_with_parser() {
        use crate::packet::{read_packet_header, PacketError};
        use oxideav_core::bits::{BitReaderLsb, BitWriterLsb};

        let setup = ap_setup(vec![
            ap_mode(false, 0),
            ap_mode(false, 0),
            ap_mode(false, 0),
        ]);
        // The writer refuses this:
        let bad = AudioPacketHeader {
            mode_number: 3,
            blockflag: false,
            n: 64,
            previous_window_flag: false,
            next_window_flag: false,
        };
        assert!(matches!(
            write_audio_packet_header(&bad, &setup, 64, 1024),
            Err(WriteAudioPacketHeaderError::BadModeNumber {
                mode_number: 3,
                mode_count: 3,
            })
        ));

        // Hand-roll the equivalent bits the writer would have emitted
        // had it ignored the gate (1-bit packet_type + 2-bit mode_number
        // = 3). The parser side rejects it for the same reason.
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        w.write_u32(3, 2); // mode_number = 3 (>= mode_count = 3)
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            read_packet_header(&mut r, &setup, 64, 1024),
            Err(PacketError::BadModeNumber {
                mode_number: 3,
                mode_count: 3,
            })
        );
    }

    /// `write_audio_packet_header_into_writer` (the splice helper)
    /// preserves the byte-aligned-output property of the public
    /// wrapper: append into a writer that is already byte-aligned, the
    /// resulting bytes equal the public function's output.
    #[test]
    fn audio_packet_header_splice_matches_public_writer() {
        use oxideav_core::bits::BitWriterLsb;

        let setup = ap_setup(vec![ap_mode(true, 0), ap_mode(false, 0)]);
        let h = AudioPacketHeader {
            mode_number: 0,
            blockflag: true,
            n: 1024,
            previous_window_flag: true,
            next_window_flag: false,
        };
        let public = write_audio_packet_header(&h, &setup, 64, 1024).unwrap();

        let mut w = BitWriterLsb::with_capacity(1);
        write_audio_packet_header_into_writer(&h, &setup, 64, 1024, &mut w).unwrap();
        let splice = w.finish();

        assert_eq!(splice, public);
    }
}

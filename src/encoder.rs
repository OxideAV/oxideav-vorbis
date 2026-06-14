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
//! * [`plan_residue_bundles`] — the §4.3.3 + §4.3.4 residue-bundle
//!   planning primitive: applies §4.3.3 nonzero-vector propagation to a
//!   per-channel `no_residue` vector, then gathers the channels per
//!   submap in ascending channel order with their per-bundle
//!   `do_not_decode` flags ([`SubmapResidueBundle`]). This is the
//!   inverse-mapping layer a wrapping §4.3 audio-packet writer threads
//!   between its floor choices and the per-submap [`write_residue_body`]
//!   calls. Round 293.
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
//!
//! Round 35 (umbrella round 267) lands
//! [`pack_residue_classifications`]: the §8.6.2 step-9..12 residue
//! classification packing primitive — the exact arithmetic inverse of
//! the residue decoder's classbook *unpack*. It packs one group of
//! `classwords_per_codeword` classification indices into the single
//! classbook entry index the residue-body writer will Huffman-code,
//! the first piece of the §8.6.2 residue-body WRITE path. The
//! VQ-codeword body emission (§8.6.3/§8.6.4/§8.6.5) and the wrapping
//! §8.6.2 residue-body writer remain explicit followups.
//!
//! Round 36 (umbrella round 274) lands
//! [`pack_residue_classification_groups`]: the grouping layer directly
//! above the per-group packer. It slices a full per-vector
//! classification array into consecutive groups of
//! `classwords_per_codeword`, right-pads the final partial group with
//! classification index `0` (the digits the decoder reads-and-discards),
//! packs each group, and returns one classbook entry per group in
//! stream order — the §8.6.2 step-6..9 decode loop's structural
//! inverse. The classbook Huffman emission of each returned entry
//! (§3.2.1 [`crate::huffman::HuffmanTree::encode_entry`]) and the
//! VQ-codeword body emission (§8.6.3/§8.6.4/§8.6.5) remain explicit
//! followups.
//!
//! Round 37 (umbrella round 278) lands [`write_residue_partition`]:
//! the §8.6.3/§8.6.4/§8.6.5 per-partition value-codeword WRITE
//! primitive — the value half of the §8.6.2 residue-body writer (the
//! round-35/36 classification packers are the classification half).
//! Given the per-partition sequence of VQ codebook **entry indices**
//! (the encoder's quantisation choice, kept explicit like the floor 1
//! packet writer's `partition_cvals` knob), it Huffman-codes each
//! entry with the partition's value book in the exact order the
//! decoder's partition decode reads them back: `n / dims` codewords
//! for format 0 (§8.6.3 step 1's `[step]`), `ceil(n / dims)` for
//! formats 1 and 2 (§8.6.4's read-while-`[i] < [n]` loop; §8.6.5 is
//! reducible to format 1). [`residue_partition_codeword_count`]
//! exposes that count so callers can size their entry lists. The
//! wrapping §8.6.2 residue-body writer that interleaves classbook
//! codewords with these partition bodies across the pass/partition/
//! vector loops remains an explicit followup.
//!
//! Round 38 (umbrella round 281) lands [`write_residue_body`]: the
//! wrapping §8.6.2 residue-body WRITE primitive. It runs the §8.6.2
//! step-3..21 pass/partition/vector loop in the *write* direction,
//! interleaving the classbook codewords (the round-35/36 packers +
//! [`crate::huffman::HuffmanTree::encode_entry`]) with the round-37
//! per-partition value-codeword bodies in the exact stream order the
//! residue decoder reads them back: on pass 0 each group of
//! `classwords_per_codeword` partitions is preceded by one classbook
//! codeword per decoded vector, and on every pass each (partition,
//! vector) pair whose cascade stage holds a value book emits its
//! partition body. The caller describes one decode vector per
//! [`ResidueVectorPlan`] (the per-partition classifications plus the
//! per-(partition, pass) entry-index lists); [`residue_body_shape`]
//! exposes the §8.6.2 step-1..5 derived `(vectors,
//! partitions_to_read)` pair — including the §8.6.5 format-2
//! single-interleaved-vector reduction and its all-'do not decode'
//! no-decode shortcut — so callers can size the plan before
//! quantising. The remaining residue-side followup is the VQ-encode
//! stage that picks the classifications and entry indices from real
//! residue scalars.

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
    /// A floor 1 audio-packet body (§7.2.3) failed one of the
    /// writer-side invariants checked by [`write_floor1_packet`].
    Floor1Packet(WriteFloor1PacketError),
    /// A residue classification group (§8.6.2 steps 9..12) failed one
    /// of the writer-side invariants checked by
    /// [`pack_residue_classifications`].
    ResidueClassification(PackResidueClassError),
    /// A residue partition body (§8.6.3/§8.6.4/§8.6.5) failed one of
    /// the writer-side invariants checked by
    /// [`write_residue_partition`].
    ResiduePartition(WriteResiduePartitionError),
    /// A full residue body (§8.6.2) failed one of the writer-side
    /// invariants checked by [`write_residue_body`].
    ResidueBody(WriteResidueBodyError),
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

/// A floor 1 audio-packet body (§7.2.3), in the shape the writer
/// emits.
///
/// `nonzero == false` represents the "this channel carried no audio
/// energy this frame" path: the packet is a single `0` bit and the
/// other fields are ignored. `nonzero == true` represents the full
/// per-partition emission: the two endpoint amplitudes plus per-class
/// master / sub-book codewords reconstructing the [`floor1_y`] vector
/// the §4.3.2 floor decoder will read back.
///
/// The struct round-trips with [`crate::floor1::Floor1Decoder::decode`]
/// up to the [`crate::floor1::FloorCurve`] return value when the
/// supplied `partition_cvals` and `floor1_y` are consistent with the
/// supplied `Floor1Header` + codebook table — see
/// [`write_floor1_packet`] for the consistency rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Floor1Packet {
    /// `[nonzero]` flag (§7.2.3 step 1). Clear when the channel is
    /// unused this frame, set when an endpoint + per-partition body
    /// follows.
    pub nonzero: bool,
    /// `[floor1_Y]` (§7.2.3): the full Y vector including the two
    /// endpoint amplitudes at positions 0 and 1, then per-partition
    /// per-dimension Y values. Length must equal
    /// `Floor1Header::x_list.len() + 2` (= the decoder's
    /// `floor1_values`). Only consulted when `nonzero == true`.
    pub floor1_y: Vec<u32>,
    /// Per-partition master-selector value (§7.2.3 step 12), one entry
    /// per partition (length must equal
    /// `Floor1Header::partition_class_list.len()`). Each entry's
    /// `subclasses > 0` partitions emit `cval` as a master-book
    /// codeword; `subclasses == 0` partitions ignore the value
    /// (per §7.2.3 step 10 [cval] is initialised to 0 and the
    /// masterbook read is skipped). Only consulted when
    /// `nonzero == true`.
    pub partition_cvals: Vec<u32>,
}

/// Errors that may arise while writing a §7.2.3 floor 1 audio-packet
/// body via [`write_floor1_packet`].
///
/// Each variant flags a §7.2.3 invariant the caller-supplied
/// [`Floor1Packet`] / [`Floor1Header`] / codebook table tuple does
/// not satisfy. The writer refuses the call without emitting any
/// bits, preserving the round-trip guarantee that the floor 1 decoder
/// reads the same `[floor1_Y]` back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteFloor1PacketError {
    /// `floor1_y.len()` does not match the decoder's `floor1_values`
    /// count (= `header.x_list.len() + 2`). §7.2.3 reads exactly that
    /// many Y values — two endpoints plus
    /// `sum(class.dimensions over partitions)` per-dimension values.
    YLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// `partition_cvals.len()` does not match the partition count
    /// (= `header.partition_class_list.len()`). §7.2.3 step 5 iterates
    /// over exactly `partitions` partitions; the writer needs one
    /// `cval` per partition.
    CvalListLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// `header.multiplier` was outside `1..=4`. §7.2.3 step 1 selects
    /// `[range]` from the four-element table indexed by `multiplier - 1`,
    /// so the legal range is `1..=4`.
    IllegalMultiplier(u8),
    /// An endpoint amplitude (`floor1_y[0]` or `floor1_y[1]`) was
    /// outside the §7.2.3 step 2/3 emission range
    /// (`0..(range)`, expressible in `ilog(range - 1)` bits).
    EndpointOverflow {
        /// `0` for `floor1_y[0]`, `1` for `floor1_y[1]`.
        index: usize,
        /// The rejected value.
        value: u32,
        /// `range` for the current `multiplier`.
        range: u32,
    },
    /// A partition referenced a class index outside `header.classes`.
    /// §7.2.3 step 6 indexes `header.classes` by the per-partition
    /// class number; the writer refuses to consult a non-existent
    /// class entry.
    BadClassIndex {
        /// Partition index in `0..partitions`.
        partition: usize,
        /// The rejected class index.
        class: u8,
        /// `header.classes.len()`.
        class_count: usize,
    },
    /// A class's `masterbook` referenced an out-of-range codebook.
    /// §7.2.2 already validates this on the header parser side, but
    /// the packet writer needs to look it up to encode the master
    /// codeword, so the caller-side codebook table must contain it.
    MasterbookOutOfRange {
        /// Class index.
        class: usize,
        /// The rejected codebook index.
        book: u8,
        /// `codebooks.len()`.
        codebook_count: usize,
    },
    /// A sub-book referenced an out-of-range codebook. Symmetric to
    /// [`Self::MasterbookOutOfRange`].
    SubclassBookOutOfRange {
        /// Class index.
        class: usize,
        /// Subclass slot index.
        subclass: usize,
        /// The rejected codebook index.
        book: u8,
        /// `codebooks.len()`.
        codebook_count: usize,
    },
    /// Building the Huffman tree for a referenced codebook failed.
    /// Mirrors [`crate::huffman::BuildError`] surfacing through the
    /// packet writer.
    Huffman(crate::huffman::BuildError),
    /// A sub-book emission was asked to encode a Y value the book
    /// cannot represent (the entry is not in the codebook's used set,
    /// or its index is past `entries`). §7.2.3 step 17 reads each Y
    /// value as a codebook entry, so the writer needs the value's
    /// entry to be a valid leaf of the Huffman tree.
    UnencodableY {
        /// Partition index.
        partition: usize,
        /// Dimension index within the partition.
        dimension: usize,
        /// The Y value the writer was asked to emit.
        y_value: u32,
        /// The codebook index the sub-book pointed at.
        book: u8,
    },
    /// A partition with a `None` sub-book (encoded `-1`, "no codebook
    /// for this subclass") was asked to encode a non-zero Y value.
    /// §7.2.3 step 18 forces the Y value to 0 when the sub-book is
    /// negative; a non-zero Y in this slot has no on-wire
    /// representation that round-trips.
    NoneBookNonzeroY {
        /// Partition index.
        partition: usize,
        /// Dimension index within the partition.
        dimension: usize,
        /// The non-zero Y value the writer was refusing to encode.
        y_value: u32,
    },
    /// A master-book emission was asked to encode a `cval` index that
    /// is not in the master codebook's used set. Symmetric to
    /// [`Self::UnencodableY`] for the master selector.
    UnencodableCval {
        /// Partition index.
        partition: usize,
        /// The `cval` value the writer was asked to emit.
        cval: u32,
        /// The codebook index the masterbook pointed at.
        book: u8,
    },
}

impl fmt::Display for WriteFloor1PacketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteFloor1PacketError::YLengthMismatch { expected, actual } => write!(
                f,
                "vorbis floor1 packet (write): floor1_y.len()={actual} != floor1_values={expected} (§7.2.3)"
            ),
            WriteFloor1PacketError::CvalListLengthMismatch { expected, actual } => write!(
                f,
                "vorbis floor1 packet (write): partition_cvals.len()={actual} != partitions={expected} (§7.2.3 step 5)"
            ),
            WriteFloor1PacketError::IllegalMultiplier(m) => write!(
                f,
                "vorbis floor1 packet (write): multiplier={m} outside 1..=4 (§7.2.3 step 1)"
            ),
            WriteFloor1PacketError::EndpointOverflow {
                index,
                value,
                range,
            } => write!(
                f,
                "vorbis floor1 packet (write): floor1_y[{index}]={value} >= range={range} (§7.2.3 step {})",
                index + 2
            ),
            WriteFloor1PacketError::BadClassIndex {
                partition,
                class,
                class_count,
            } => write!(
                f,
                "vorbis floor1 packet (write): partition {partition} class index {class} outside header.classes (count={class_count}) (§7.2.3 step 6)"
            ),
            WriteFloor1PacketError::MasterbookOutOfRange {
                class,
                book,
                codebook_count,
            } => write!(
                f,
                "vorbis floor1 packet (write): classes[{class}].masterbook={book} >= codebooks.len()={codebook_count} (§7.2.2)"
            ),
            WriteFloor1PacketError::SubclassBookOutOfRange {
                class,
                subclass,
                book,
                codebook_count,
            } => write!(
                f,
                "vorbis floor1 packet (write): classes[{class}].subclass_books[{subclass}]={book} >= codebooks.len()={codebook_count} (§7.2.2)"
            ),
            WriteFloor1PacketError::Huffman(e) => write!(
                f,
                "vorbis floor1 packet (write): Huffman build error: {e}"
            ),
            WriteFloor1PacketError::UnencodableY {
                partition,
                dimension,
                y_value,
                book,
            } => write!(
                f,
                "vorbis floor1 packet (write): partition {partition} dimension {dimension} Y={y_value} not encodable by codebook {book} (§7.2.3 step 17)"
            ),
            WriteFloor1PacketError::NoneBookNonzeroY {
                partition,
                dimension,
                y_value,
            } => write!(
                f,
                "vorbis floor1 packet (write): partition {partition} dimension {dimension} Y={y_value} != 0 but its sub-book is None (§7.2.3 step 18 forces 0)"
            ),
            WriteFloor1PacketError::UnencodableCval {
                partition,
                cval,
                book,
            } => write!(
                f,
                "vorbis floor1 packet (write): partition {partition} cval={cval} not encodable by master codebook {book} (§7.2.3 step 12)"
            ),
        }
    }
}

impl std::error::Error for WriteFloor1PacketError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WriteFloor1PacketError::Huffman(e) => Some(e),
            _ => None,
        }
    }
}

impl From<crate::huffman::BuildError> for WriteFloor1PacketError {
    fn from(value: crate::huffman::BuildError) -> Self {
        WriteFloor1PacketError::Huffman(value)
    }
}

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
            WriteError::Floor1Packet(e) => write!(f, "{e}"),
            WriteError::ResidueClassification(e) => write!(f, "{e}"),
            WriteError::ResiduePartition(e) => write!(f, "{e}"),
            WriteError::ResidueBody(e) => write!(f, "{e}"),
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
            WriteError::Floor1Packet(e) => Some(e),
            WriteError::ResidueClassification(e) => Some(e),
            WriteError::ResiduePartition(e) => Some(e),
            WriteError::ResidueBody(e) => Some(e),
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

impl From<WriteFloor1PacketError> for WriteError {
    fn from(value: WriteFloor1PacketError) -> Self {
        WriteError::Floor1Packet(value)
    }
}

impl From<PackResidueClassError> for WriteError {
    fn from(value: PackResidueClassError) -> Self {
        WriteError::ResidueClassification(value)
    }
}

impl From<WriteResiduePartitionError> for WriteError {
    fn from(value: WriteResiduePartitionError) -> Self {
        WriteError::ResiduePartition(value)
    }
}

impl From<WriteResidueBodyError> for WriteError {
    fn from(value: WriteResidueBodyError) -> Self {
        WriteError::ResidueBody(value)
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

/// Serialises a [`Floor1Packet`] to the §7.2.3 floor 1 audio-packet
/// body bitstream shape.
///
/// This is the encoder-side counterpart of
/// [`crate::floor1::Floor1Decoder::decode`]: given the same
/// `(header, codebooks)` context the decoder was built with, the
/// emitted packet decodes back to the `floor1_y` vector the caller
/// supplied (modulo §7.2.4's `[floor1_final_Y]` derivation, which is
/// the curve-computation stage).
///
/// Layout per §7.2.3:
///
/// | Step | Field             | Bits                              |
/// | ---: | ----------------- | --------------------------------- |
/// |    1 | `[nonzero]`       | 1                                 |
/// |  2-3 | endpoints         | `2 × ilog(range - 1)` (if nonzero)|
/// |    5 | per partition i { | iterate over `partition_class_list`|
/// |   12 |   master `cval`   | masterbook codeword (if cbits > 0)|
/// |   13 |   per dimension j { | iterate over `class.dimensions`  |
/// |   17 |     sub-book Y    | sub-book codeword (if Some)       |
/// |   18 |     (forced 0)    | 0 bits (if None)                  |
/// |      |   }               |                                   |
/// |      | }                 |                                   |
///
/// `[range]` is the `{256, 128, 86, 64}` entry indexed by
/// `multiplier - 1` (§7.2.3 step 1). Each codeword is emitted MSb-first
/// via [`crate::huffman::HuffmanTree::encode_entry`], matching the
/// §3.2.1 canonical-codeword convention the decoder reads.
///
/// # Errors
///
/// Returns [`WriteFloor1PacketError`] without emitting any bits if any
/// of the per-variant invariants is violated — see the type's variant
/// documentation. The roundtrip property holds for every valid call:
///
/// ```text
/// // For every legal Floor1Packet `p`, header `h`, codebooks `cb`:
/// let bytes = write_floor1_packet(&p, &h, &cb).unwrap();
/// let dec = Floor1Decoder::new(&h, &cb).unwrap();
/// let curve = dec.decode(&mut BitReaderLsb::new(&bytes), n);
/// // when nonzero: curve == Curve(expected_curve_from_p.floor1_y)
/// // when !nonzero: curve == Unused
/// ```
pub fn write_floor1_packet(
    packet: &Floor1Packet,
    header: &crate::setup::Floor1Header,
    codebooks: &[VorbisCodebook],
) -> Result<Vec<u8>, WriteFloor1PacketError> {
    let mut writer = BitWriterLsb::with_capacity(2);
    write_floor1_packet_into_writer(packet, header, codebooks, &mut writer)?;
    Ok(writer.finish())
}

/// Bit-level helper to splice a §7.2.3 floor 1 audio-packet body into a
/// larger bit-packed stream. The wrapping §4.3 audio-packet writer
/// (explicit followup) will use this to thread the per-channel floor 1
/// body between the §4.3.1 prelude and the §4.3.4 residue body,
/// mirroring the existing per-header `_into_writer` splice points.
///
/// Writes the body's bits into `writer` at its current bit position.
/// On error, no bits are emitted (validation precedes emission).
pub(crate) fn write_floor1_packet_into_writer(
    packet: &Floor1Packet,
    header: &crate::setup::Floor1Header,
    codebooks: &[VorbisCodebook],
    writer: &mut BitWriterLsb,
) -> Result<(), WriteFloor1PacketError> {
    // ---- §7.2.3 step 1: unused short-circuit. ----
    if !packet.nonzero {
        // Emit only the `[nonzero] = 0` bit; the decoder returns
        // FloorCurve::Unused without reading further. The other packet
        // fields are intentionally ignored to keep "build an Unused
        // packet without populating floor1_y / partition_cvals"
        // ergonomic.
        writer.write_bit(false);
        return Ok(());
    }

    // ---- §7.2.3 fail-closed invariant gate. ----
    // multiplier in 1..=4 (§7.2.3 step 1).
    if !(1..=4).contains(&header.multiplier) {
        return Err(WriteFloor1PacketError::IllegalMultiplier(header.multiplier));
    }
    let range = RANGE_TABLE_FLOOR1[(header.multiplier - 1) as usize];
    let amp_bits = ilog(range - 1);

    // floor1_y length = floor1_values (x_list.len() + 2 implicit endpoints).
    let expected_y_len = header.x_list.len() + 2;
    if packet.floor1_y.len() != expected_y_len {
        return Err(WriteFloor1PacketError::YLengthMismatch {
            expected: expected_y_len,
            actual: packet.floor1_y.len(),
        });
    }

    // partition_cvals length = partition_class_list.len() (§7.2.3 step 5).
    let partitions = header.partition_class_list.len();
    if packet.partition_cvals.len() != partitions {
        return Err(WriteFloor1PacketError::CvalListLengthMismatch {
            expected: partitions,
            actual: packet.partition_cvals.len(),
        });
    }

    // Endpoint ranges (§7.2.3 step 2/3): each Y must fit in `amp_bits`,
    // i.e. be strictly less than `range`.
    for idx in 0..2 {
        let v = packet.floor1_y[idx];
        if v >= range {
            return Err(WriteFloor1PacketError::EndpointOverflow {
                index: idx,
                value: v,
                range,
            });
        }
    }

    // Pre-resolve every partition's class + its master/sub-book Huffman
    // trees in one pass so we fail fast on a bad class index or a
    // dangling codebook reference *before* we start emitting bits. The
    // partition_class_list / class-list out-of-range cases are then
    // gated upstream; on success we have a parallel `Vec<ClassEnc>`
    // ready for the emit pass.
    struct ClassEnc {
        dimensions: usize,
        subclasses: u8,
        masterbook_tree: Option<crate::huffman::HuffmanTree>,
        subclass_trees: Vec<Option<crate::huffman::HuffmanTree>>,
    }

    let mut per_partition: Vec<ClassEnc> = Vec::with_capacity(partitions);
    for (partition_idx, &class_no) in header.partition_class_list.iter().enumerate() {
        let class =
            header
                .classes
                .get(class_no as usize)
                .ok_or(WriteFloor1PacketError::BadClassIndex {
                    partition: partition_idx,
                    class: class_no,
                    class_count: header.classes.len(),
                })?;

        let masterbook_tree = if class.subclasses > 0 {
            match class.masterbook {
                Some(book) => {
                    let cb = codebooks.get(book as usize).ok_or(
                        WriteFloor1PacketError::MasterbookOutOfRange {
                            class: class_no as usize,
                            book,
                            codebook_count: codebooks.len(),
                        },
                    )?;
                    Some(crate::huffman::HuffmanTree::from_codebook(cb)?)
                }
                // (header parser already rejects subclasses > 0 with no
                // masterbook; defensive None here means the decoder
                // wouldn't read the masterbook either, so emit nothing.)
                None => None,
            }
        } else {
            None
        };

        let mut subclass_trees: Vec<Option<crate::huffman::HuffmanTree>> =
            Vec::with_capacity(class.subclass_books.len());
        for (sub_idx, slot) in class.subclass_books.iter().enumerate() {
            match slot {
                None => subclass_trees.push(None),
                Some(book) => {
                    let cb = codebooks.get(*book as usize).ok_or(
                        WriteFloor1PacketError::SubclassBookOutOfRange {
                            class: class_no as usize,
                            subclass: sub_idx,
                            book: *book,
                            codebook_count: codebooks.len(),
                        },
                    )?;
                    subclass_trees.push(Some(crate::huffman::HuffmanTree::from_codebook(cb)?));
                }
            }
        }

        per_partition.push(ClassEnc {
            dimensions: class.dimensions as usize,
            subclasses: class.subclasses,
            masterbook_tree,
            subclass_trees,
        });
    }

    // Cross-check the implied Y length against `expected_y_len`: the
    // sum of per-partition dimensions plus the two endpoint slots must
    // equal x_list.len() + 2. (Header-side parser enforces this on
    // input; we re-check defensively because a hand-built Floor1Header
    // could disagree.)
    let dims_sum: usize = per_partition.iter().map(|c| c.dimensions).sum();
    let implied_y_len = dims_sum + 2;
    if implied_y_len != expected_y_len {
        return Err(WriteFloor1PacketError::YLengthMismatch {
            expected: implied_y_len,
            actual: expected_y_len,
        });
    }

    // Per-partition encodability pre-check: walk `floor1_y` from
    // offset 2, comparing each Y value against the corresponding
    // sub-book's used-set. We need to do this BEFORE emitting any
    // bits — see the fail-closed contract. We also resolve which
    // sub-book each dimension uses by replaying the same `cval & csub
    // → cval >>= cbits` decode logic on the encoder side.
    //
    // The check also catches the `cval` overflow case implicitly: if a
    // class has cbits > 0 but the caller supplied a cval too large to
    // emit through the master book (i.e. cval is not in the master
    // book's used set), encode_entry will refuse — surfaced through
    // UnencodableCval.

    let mut offset = 2usize;
    for (partition_idx, class) in per_partition.iter().enumerate() {
        let cbits = class.subclasses;
        let csub: u32 = (1u32 << cbits).saturating_sub(1);
        let mut cval = packet.partition_cvals[partition_idx];

        // Master selector: encodable only if a master tree exists and
        // `cval` is a leaf.
        if cbits > 0 {
            if let Some(tree) = &class.masterbook_tree {
                let mut dummy = BitWriterLsb::new();
                tree.encode_entry(cval, &mut dummy).map_err(|_| {
                    WriteFloor1PacketError::UnencodableCval {
                        partition: partition_idx,
                        cval,
                        book: header.classes[header.partition_class_list[partition_idx] as usize]
                            .masterbook
                            .unwrap_or(0),
                    }
                })?;
            }
        }

        for dim_idx in 0..class.dimensions {
            let sub_idx = (cval & csub) as usize;
            cval >>= cbits;
            let y = packet.floor1_y[offset + dim_idx];
            match class.subclass_trees.get(sub_idx).and_then(|t| t.as_ref()) {
                Some(tree) => {
                    let mut dummy = BitWriterLsb::new();
                    tree.encode_entry(y, &mut dummy).map_err(|_| {
                        // Resolve the codebook index for the error
                        // message via the same path the decoder would
                        // take.
                        let class_no = header.partition_class_list[partition_idx] as usize;
                        let book = header.classes[class_no].subclass_books[sub_idx].unwrap_or(0);
                        WriteFloor1PacketError::UnencodableY {
                            partition: partition_idx,
                            dimension: dim_idx,
                            y_value: y,
                            book,
                        }
                    })?;
                }
                // §7.2.3 step 18: `None` sub-book forces Y = 0; a
                // non-zero Y here has no on-wire round-trip.
                None => {
                    if y != 0 {
                        return Err(WriteFloor1PacketError::NoneBookNonzeroY {
                            partition: partition_idx,
                            dimension: dim_idx,
                            y_value: y,
                        });
                    }
                }
            }
        }
        offset += class.dimensions;
    }

    // ---- §7.2.3 emit. ----
    // step 1: [nonzero] = 1.
    writer.write_bit(true);
    // steps 2..3: endpoints, each `amp_bits` wide.
    writer.write_u32(packet.floor1_y[0], amp_bits);
    writer.write_u32(packet.floor1_y[1], amp_bits);

    // step 5: iterate partitions.
    let mut offset = 2usize;
    for (partition_idx, class) in per_partition.iter().enumerate() {
        let cbits = class.subclasses;
        let csub: u32 = (1u32 << cbits).saturating_sub(1);
        let mut cval = packet.partition_cvals[partition_idx];

        // step 12: master selector (only when cbits > 0).
        if cbits > 0 {
            if let Some(tree) = &class.masterbook_tree {
                // `encode_entry` is infallible here because the
                // pre-check verified `cval` is a leaf.
                tree.encode_entry(cval, writer)
                    .expect("encode_entry must succeed after pre-check; cval already validated");
            }
        }

        // step 13: per-dimension Y emission.
        for dim_idx in 0..class.dimensions {
            let sub_idx = (cval & csub) as usize;
            cval >>= cbits;
            let y = packet.floor1_y[offset + dim_idx];
            match class.subclass_trees.get(sub_idx).and_then(|t| t.as_ref()) {
                Some(tree) => {
                    tree.encode_entry(y, writer)
                        .expect("encode_entry must succeed after pre-check; Y already validated");
                }
                None => {
                    // §7.2.3 step 18: Y is forced to 0; no bits emitted.
                }
            }
        }
        offset += class.dimensions;
    }

    // step 20: done.
    Ok(())
}

/// `[range]` table for floor 1 (§7.2.3 step 1, also §7.2.4 step-1 step 1):
/// `{ 256, 128, 86, 64 }` indexed by `[floor1_multiplier] - 1`.
const RANGE_TABLE_FLOOR1: [u32; 4] = [256, 128, 86, 64];

/// The encoder's description of one §6.2.2 floor 0 audio-packet body —
/// the input to [`write_floor0_packet`].
///
/// A floor 0 packet is either *unused* (the per-packet `[amplitude]`
/// field read zero, so the channel carries no energy this frame and the
/// decoder emits an all-zero curve) or a *curve* (a nonzero amplitude
/// plus a value-book selector and a run of VQ codewords that rebuild the
/// LSP coefficient list). [`Floor0Packet::Unused`] emits only the
/// `floor0_amplitude_bits`-wide zero amplitude field — exactly the bits
/// the §6.2.2 step-2 zero-amplitude short-circuit reads before returning
/// `'unused'`. [`Floor0Packet::Curve`] emits the amplitude, the
/// `ilog(floor0_number_of_books)`-wide `[booknumber]` selector, then one
/// canonical §3.2.1 codeword per supplied VQ entry.
///
/// The `entries` member is the encoder's explicit quantisation choice —
/// the same knob philosophy as [`Floor1Packet::partition_cvals`] and
/// [`ResidueVectorPlan`]: the writer serialises exactly the value-book
/// entry indices it is handed, bit-exact by construction, and a future
/// VQ-encode stage picks the entries that nearest-match a target LSP
/// curve. Each entry decodes (through the selected book's
/// [`crate::vq::unpack_vector`]) to `book.dimensions` coefficients; the
/// writer requires exactly `ceil(order / dimensions)` entries — the
/// count the §6.2.2 step-7..11 loop reads to fill `[coefficients]` to
/// `floor0_order` scalars (the decoder stops the *frame* once
/// `len(coefficients) >= order`, so a trailing partial vector is read in
/// full and its surplus scalars discarded).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Floor0Packet {
    /// The channel carried no audio energy this frame: the `[amplitude]`
    /// field is zero and the §6.2.2 step-2 short-circuit returns
    /// `'unused'` without reading the `[booknumber]` or any VQ codewords.
    Unused,
    /// A nonzero-amplitude curve: the amplitude, the value-book selector,
    /// and the VQ entry run that rebuilds the LSP coefficients.
    Curve {
        /// `[amplitude]` (§6.2.2 step 1), emitted in
        /// `floor0_amplitude_bits` bits. Must be `> 0` (a zero amplitude
        /// is [`Floor0Packet::Unused`], not a `Curve`) and must fit the
        /// `floor0_amplitude_bits`-wide field.
        amplitude: u32,
        /// `[booknumber]` (§6.2.2 step 4) — a *position* in
        /// `floor0_book_list`, emitted in
        /// `ilog(floor0_number_of_books)` bits. Selects the value book
        /// `codebooks[book_list[booknumber]]` used for the VQ decode.
        booknumber: u32,
        /// The value-book **entry indices** the §6.2.2 step-7 loop
        /// decodes, in stream order. Length must equal
        /// `ceil(floor0_order / book.dimensions)`.
        entries: Vec<u32>,
    },
}

/// Errors that may arise while writing a §6.2.2 floor 0 audio-packet
/// body via [`write_floor0_packet`].
///
/// Each variant flags a §6.2.2 invariant the caller-supplied
/// [`Floor0Packet`] / [`Floor0Header`] / codebook table tuple does not
/// satisfy. The writer refuses the call without emitting any bits,
/// preserving the round-trip guarantee that the floor 0 decoder reads
/// the same amplitude / booknumber / coefficients back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteFloor0PacketError {
    /// `floor0_amplitude_bits` was zero. §6.2.1 stores it as a `read 6
    /// bits` field; a zero width makes the `[amplitude]` read always
    /// yield 0, so the decoder constructor (`Floor0Decoder::new`) rejects
    /// such a header outright and no nonzero-amplitude packet could ever
    /// round-trip. The writer mirrors that fail-closed gate.
    ZeroAmplitudeBits,
    /// `floor0_amplitude_bits` exceeded 63 (the §6.2.1 6-bit field's
    /// maximum). Symmetric to [`WriteFloor0Error::AmplitudeBitsOverflow`].
    AmplitudeBitsOverflow(u8),
    /// `floor0_order` was zero. §6.2.1 stores it as a `read 8 bits`
    /// field, but `Floor0Decoder::new` rejects a zero order (the §6.2.2
    /// step-7 loop would read zero vectors and the §6.2.3 curve would be
    /// empty). The writer mirrors that gate.
    ZeroOrder,
    /// `floor0_book_list` was empty. §6.2.1 stores `floor0_number_of_books`
    /// as `read 4 bits + 1`, so the legal minimum is 1; an empty list has
    /// no `[booknumber]` field width and no value books to select.
    EmptyBookList,
    /// `floor0_book_list.len()` exceeded 16 (the §6.2.1 `read 4 bits + 1`
    /// field's maximum). Symmetric to [`WriteFloor0Error::BookListTooLong`].
    BookListTooLong(usize),
    /// A `Curve` packet carried `amplitude == 0`. §6.2.2 step 2 treats a
    /// zero amplitude as `'unused'`; a zero-amplitude `Curve` has no
    /// on-wire representation that round-trips back to a `Curve` (the
    /// decoder would return `'unused'` and never read the booknumber or
    /// coefficients). Use [`Floor0Packet::Unused`] instead.
    ZeroAmplitudeCurve,
    /// A `Curve` packet's `amplitude` did not fit the
    /// `floor0_amplitude_bits`-wide field (`amplitude >= 1 <<
    /// amplitude_bits`). §6.2.2 step 1 reads exactly `amplitude_bits`
    /// bits, so a wider value cannot be recovered.
    AmplitudeOverflow {
        /// The rejected amplitude.
        amplitude: u32,
        /// `floor0_amplitude_bits`.
        amplitude_bits: u8,
    },
    /// A `Curve` packet's `booknumber` was `>= floor0_number_of_books`.
    /// §6.2.2 step 5 maps an out-of-range book selector to `'unused'`
    /// (the reserved values correspond to no value book), so it cannot
    /// round-trip back to a `Curve`.
    BooknumberOutOfRange {
        /// The rejected `booknumber`.
        booknumber: u32,
        /// `floor0_number_of_books` (= `book_list.len()`).
        number_of_books: usize,
    },
    /// The value book a `Curve` selected referenced an out-of-range
    /// codebook. `floor0_book_list[booknumber]` indexes the codebook
    /// table; the writer needs the codebook to encode the VQ codewords.
    ValueBookOutOfRange {
        /// The `book_list` position (= `booknumber`).
        position: usize,
        /// `book_list[booknumber]` — the rejected codebook index.
        book: u8,
        /// `codebooks.len()`.
        codebook_count: usize,
    },
    /// The value book a `Curve` selected has `lookup_type == 0` (no VQ
    /// value mapping). §3.3 forbids a VQ-context decode against a
    /// scalar-only codebook; `Floor0Decoder::new` rejects such a book at
    /// construction time. The writer mirrors that gate.
    ValueBookHasNoLookup {
        /// The `book_list` position (= `booknumber`).
        position: usize,
        /// `book_list[booknumber]` — the offending codebook index.
        book: u8,
    },
    /// The value book a `Curve` selected has `dimensions == 0`. The
    /// §6.2.2 step-7 loop adds `dimensions` coefficients per vector; a
    /// zero-dimension book would loop forever, so the decoder treats it
    /// as undecodable and the writer refuses it.
    ZeroDimensionBook {
        /// The `book_list` position (= `booknumber`).
        position: usize,
        /// `book_list[booknumber]` — the offending codebook index.
        book: u8,
    },
    /// A `Curve` packet's `entries.len()` did not match the count the
    /// §6.2.2 step-7..11 loop reads. The decoder reads vectors until
    /// `len(coefficients) >= order`, i.e. exactly
    /// `ceil(order / dimensions)` vectors; the writer needs one entry per
    /// vector so the decode count matches.
    EntryCountMismatch {
        /// `ceil(order / dimensions)` — the count the decoder reads.
        expected: usize,
        /// The supplied `entries.len()`.
        actual: usize,
    },
    /// A VQ entry index was `>= book.entries` (out of the value book's
    /// range). §6.2.2 step 7 decodes a codeword whose entry must be a
    /// valid index into the book's value-mapping table.
    EntryOutOfRange {
        /// Index into `entries`.
        index: usize,
        /// The rejected entry.
        entry: u32,
        /// `book.entries`.
        entries: u32,
    },
    /// A VQ entry index is not in the value book's *used* set (no
    /// canonical §3.2.1 codeword maps to it). §6.2.2 step 7 decodes each
    /// codeword through the book's Huffman tree, so every emitted entry
    /// must be a leaf of that tree.
    UnencodableEntry {
        /// Index into `entries`.
        index: usize,
        /// The rejected entry.
        entry: u32,
        /// The number of used entries in the book's tree.
        used_count: u32,
    },
    /// Building the Huffman tree for the selected value book failed.
    /// Mirrors [`crate::huffman::BuildError`] surfacing through the
    /// packet writer.
    Huffman(crate::huffman::BuildError),
}

impl fmt::Display for WriteFloor0PacketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteFloor0PacketError::ZeroAmplitudeBits => write!(
                f,
                "vorbis floor0 packet (write): floor0_amplitude_bits is zero — no curve packet can round-trip (§6.2.1)"
            ),
            WriteFloor0PacketError::AmplitudeBitsOverflow(b) => write!(
                f,
                "vorbis floor0 packet (write): floor0_amplitude_bits={b} > 63 (§6.2.1 6-bit field)"
            ),
            WriteFloor0PacketError::ZeroOrder => write!(
                f,
                "vorbis floor0 packet (write): floor0_order is zero (§6.2.1)"
            ),
            WriteFloor0PacketError::EmptyBookList => write!(
                f,
                "vorbis floor0 packet (write): floor0_book_list is empty (§6.2.1 number_of_books >= 1)"
            ),
            WriteFloor0PacketError::BookListTooLong(n) => write!(
                f,
                "vorbis floor0 packet (write): floor0_book_list.len()={n} > 16 (§6.2.1 read 4 bits + 1)"
            ),
            WriteFloor0PacketError::ZeroAmplitudeCurve => write!(
                f,
                "vorbis floor0 packet (write): Curve packet with amplitude=0 — use Floor0Packet::Unused (§6.2.2 step 2)"
            ),
            WriteFloor0PacketError::AmplitudeOverflow {
                amplitude,
                amplitude_bits,
            } => write!(
                f,
                "vorbis floor0 packet (write): amplitude={amplitude} does not fit floor0_amplitude_bits={amplitude_bits} (§6.2.2 step 1)"
            ),
            WriteFloor0PacketError::BooknumberOutOfRange {
                booknumber,
                number_of_books,
            } => write!(
                f,
                "vorbis floor0 packet (write): booknumber={booknumber} >= floor0_number_of_books={number_of_books} (§6.2.2 step 5)"
            ),
            WriteFloor0PacketError::ValueBookOutOfRange {
                position,
                book,
                codebook_count,
            } => write!(
                f,
                "vorbis floor0 packet (write): book_list[{position}]={book} >= codebooks.len()={codebook_count} (§6.2.1)"
            ),
            WriteFloor0PacketError::ValueBookHasNoLookup { position, book } => write!(
                f,
                "vorbis floor0 packet (write): value book at position {position} (codebook {book}) has lookup_type=0 (§3.3)"
            ),
            WriteFloor0PacketError::ZeroDimensionBook { position, book } => write!(
                f,
                "vorbis floor0 packet (write): value book at position {position} (codebook {book}) has dimensions=0 (§6.2.2 step 7 would loop)"
            ),
            WriteFloor0PacketError::EntryCountMismatch { expected, actual } => write!(
                f,
                "vorbis floor0 packet (write): entries.len()={actual} != ceil(order/dimensions)={expected} (§6.2.2 step 7..11)"
            ),
            WriteFloor0PacketError::EntryOutOfRange {
                index,
                entry,
                entries,
            } => write!(
                f,
                "vorbis floor0 packet (write): entries[{index}]={entry} >= book.entries={entries} (§6.2.2 step 7)"
            ),
            WriteFloor0PacketError::UnencodableEntry {
                index,
                entry,
                used_count,
            } => write!(
                f,
                "vorbis floor0 packet (write): entries[{index}]={entry} not encodable (book tree has {used_count} used entries) (§6.2.2 step 7)"
            ),
            WriteFloor0PacketError::Huffman(e) => write!(
                f,
                "vorbis floor0 packet (write): Huffman build error: {e}"
            ),
        }
    }
}

impl std::error::Error for WriteFloor0PacketError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WriteFloor0PacketError::Huffman(e) => Some(e),
            _ => None,
        }
    }
}

impl From<crate::huffman::BuildError> for WriteFloor0PacketError {
    fn from(value: crate::huffman::BuildError) -> Self {
        WriteFloor0PacketError::Huffman(value)
    }
}

/// Serialises a §6.2.2 floor 0 audio-packet body to the bitstream the
/// floor 0 decoder ([`crate::floor0::Floor0Decoder::decode`]) reads
/// back.
///
/// This is the inverse of the §6.2.2 packet decode: an
/// [`Floor0Packet::Unused`] emits only the `floor0_amplitude_bits`-wide
/// zero amplitude; an [`Floor0Packet::Curve`] emits the amplitude, the
/// `ilog(floor0_number_of_books)`-wide `[booknumber]` selector, then one
/// canonical §3.2.1 codeword per VQ entry. The returned [`Vec<u8>`] is
/// the body's bits LSB-first (§2.1.4), with up to 7 bits of zero padding
/// in the final byte to byte-align the slice.
///
/// On error the call is refused without emitting a single bit (validation
/// precedes emission), preserving the round-trip guarantee that the
/// decoder reads the same amplitude / booknumber / coefficients back.
///
/// ## Spec source
///
/// `docs/audio/vorbis/Vorbis_I_spec.pdf` §6.2.2 (the floor 0 packet
/// decode — the amplitude / booknumber / VQ-vector read loop), §6.2.1
/// (the header field bounds the writer mirrors), §3.2.1 (canonical
/// Huffman codewords), §3.3 (the VQ-context lookup_type gate), §2.1.4
/// (LSB-first packing).
pub fn write_floor0_packet(
    packet: &Floor0Packet,
    header: &Floor0Header,
    codebooks: &[VorbisCodebook],
) -> Result<Vec<u8>, WriteFloor0PacketError> {
    let mut writer = BitWriterLsb::with_capacity(2);
    write_floor0_packet_into_writer(packet, header, codebooks, &mut writer)?;
    Ok(writer.finish())
}

/// Bit-level helper to splice a §6.2.2 floor 0 audio-packet body into a
/// larger bit-packed stream. The wrapping §4.3 audio-packet writer
/// (explicit followup) will use this to thread the per-channel floor 0
/// body between the §4.3.1 prelude and the §4.3.4 residue body,
/// mirroring [`write_floor1_packet_into_writer`] and the existing
/// per-header `_into_writer` splice points.
///
/// Writes the body's bits into `writer` at its current bit position.
/// On error, no bits are emitted (validation precedes emission).
pub(crate) fn write_floor0_packet_into_writer(
    packet: &Floor0Packet,
    header: &Floor0Header,
    codebooks: &[VorbisCodebook],
    writer: &mut BitWriterLsb,
) -> Result<(), WriteFloor0PacketError> {
    // ---- §6.2.1 header invariant gate (mirrors Floor0Decoder::new). ----
    // These bound the field widths the packet body uses; a header that
    // the decoder constructor would reject can never produce a
    // round-tripping packet, so we fail closed before touching the
    // packet contents.
    if header.amplitude_bits == 0 {
        return Err(WriteFloor0PacketError::ZeroAmplitudeBits);
    }
    if header.amplitude_bits > 63 {
        return Err(WriteFloor0PacketError::AmplitudeBitsOverflow(
            header.amplitude_bits,
        ));
    }
    if header.order == 0 {
        return Err(WriteFloor0PacketError::ZeroOrder);
    }
    if header.book_list.is_empty() {
        return Err(WriteFloor0PacketError::EmptyBookList);
    }
    if header.book_list.len() > 16 {
        return Err(WriteFloor0PacketError::BookListTooLong(
            header.book_list.len(),
        ));
    }

    let amplitude_bits = header.amplitude_bits as u32;

    // ---- §6.2.2 step 1/2: the unused short-circuit. ----
    let (amplitude, booknumber, entries) = match packet {
        Floor0Packet::Unused => {
            // Emit only the zero `[amplitude]` field; the decoder's
            // step-2 short-circuit returns 'unused' without reading
            // further. (Validation above already passed; nothing in the
            // packet payload to check.)
            writer.write_u32(0, amplitude_bits);
            return Ok(());
        }
        Floor0Packet::Curve {
            amplitude,
            booknumber,
            entries,
        } => (*amplitude, *booknumber, entries),
    };

    // ---- §6.2.2 Curve invariant gate (no bits emitted on error). ----
    // (1) amplitude must be nonzero (else it round-trips as 'unused').
    if amplitude == 0 {
        return Err(WriteFloor0PacketError::ZeroAmplitudeCurve);
    }
    // (2) amplitude must fit the amplitude_bits-wide field.
    //     (amplitude_bits is 1..=63 here, so `1u64 << amplitude_bits`
    //     never overflows a u64.)
    if (amplitude as u64) >= (1u64 << amplitude_bits) {
        return Err(WriteFloor0PacketError::AmplitudeOverflow {
            amplitude,
            amplitude_bits: header.amplitude_bits,
        });
    }

    // (3) §6.2.2 step 4/5: booknumber selects a position in book_list.
    //     The decoder reads ilog(number_of_books) bits and rejects an
    //     out-of-range selector as 'unused'; that cannot round-trip to a
    //     Curve, so we refuse it.
    let number_of_books = header.book_list.len();
    if (booknumber as usize) >= number_of_books {
        return Err(WriteFloor0PacketError::BooknumberOutOfRange {
            booknumber,
            number_of_books,
        });
    }
    let position = booknumber as usize;
    let book_index = header.book_list[position];

    // (4) §6.2.1: the selected book must exist, carry a VQ value mapping,
    //     and have nonzero dimensions (else the decoder loops). These
    //     mirror Floor0Decoder::new's per-book construction checks.
    let book =
        codebooks
            .get(book_index as usize)
            .ok_or(WriteFloor0PacketError::ValueBookOutOfRange {
                position,
                book: book_index,
                codebook_count: codebooks.len(),
            })?;
    if matches!(book.lookup, VqLookup::None) {
        return Err(WriteFloor0PacketError::ValueBookHasNoLookup {
            position,
            book: book_index,
        });
    }
    let dimensions = book.dimensions as usize;
    if dimensions == 0 {
        return Err(WriteFloor0PacketError::ZeroDimensionBook {
            position,
            book: book_index,
        });
    }

    // (5) §6.2.2 step 7..11: the decoder reads vectors until
    //     len(coefficients) >= order, i.e. ceil(order / dimensions)
    //     vectors. The encoder must supply exactly that many entries.
    let expected = (header.order as usize).div_ceil(dimensions);
    if entries.len() != expected {
        return Err(WriteFloor0PacketError::EntryCountMismatch {
            expected,
            actual: entries.len(),
        });
    }

    // (6) Every entry must be a valid, encodable leaf of the book's
    //     §3.2.1 Huffman tree. The encodability pre-check runs against a
    //     scratch writer so the caller's stream is untouched if any entry
    //     is refused (mirrors write_residue_partition_into_writer).
    let tree = crate::huffman::HuffmanTree::from_codebook(book)?;
    for (index, &entry) in entries.iter().enumerate() {
        if entry >= book.entries {
            return Err(WriteFloor0PacketError::EntryOutOfRange {
                index,
                entry,
                entries: book.entries,
            });
        }
        let mut scratch = BitWriterLsb::new();
        tree.encode_entry(entry, &mut scratch).map_err(|e| {
            let crate::huffman::EncodeError::UnknownEntry { used_count, .. } = e;
            WriteFloor0PacketError::UnencodableEntry {
                index,
                entry,
                used_count,
            }
        })?;
    }

    // ---- §6.2.2 emit (all validation passed). ----
    // step 1: [amplitude] in amplitude_bits bits.
    writer.write_u32(amplitude, amplitude_bits);
    // step 4: [booknumber] in ilog(number_of_books) bits.
    let book_index_bits = ilog(number_of_books as u32);
    writer.write_u32(booknumber, book_index_bits);
    // step 7: one canonical codeword per VQ entry, in stream order.
    for &entry in entries {
        tree.encode_entry(entry, writer)
            .expect("encode_entry must succeed after pre-check; entry already validated");
    }
    Ok(())
}

/// Errors that may arise while packing a residue classification group
/// (§8.6.2 steps 9..12) via [`pack_residue_classifications`].
///
/// The packer is the exact arithmetic inverse of the §8.6.2 step-10..12
/// classbook *unpack* the residue decoder performs (see
/// [`crate::residue::ResidueDecoder`]). Each variant flags an input the
/// packer cannot serialise back to a classbook entry index the unpack
/// loop would reproduce, so it refuses the call rather than emit a value
/// that fails to round-trip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackResidueClassError {
    /// `num_classifications` was zero. §8.6.1 stores
    /// `residue_classifications` as a `read 6 bits + 1` field, so the
    /// legal range is `1..=64`; a zero base has no meaning for the
    /// positional `temp % C` / `temp /= C` digit extraction.
    ZeroClassifications,
    /// `num_classifications` exceeded 64. §8.6.1 caps
    /// `residue_classifications` at 64 (the `read 6 bits + 1` upper
    /// edge). The contained value is the rejected base.
    ClassificationsTooLarge(u32),
    /// The classification group was empty. §8.6.2 packs
    /// `classwords_per_codeword` classifications into one classbook
    /// entry, and that count (the classbook's `dimensions`) is `>= 1`
    /// for any decodable residue; a zero-length group cannot be the
    /// inverse of a real classbook read.
    EmptyGroup,
    /// The group held more than 32 classifications. A classbook entry
    /// index is a `u32`, and `C^32 >= 2^32` for every legal base
    /// `C >= 2`, so a group longer than 32 cannot fit its packed value
    /// in the `u32` the unpack loop reads in scalar context. The
    /// contained value is the rejected group length. (In practice
    /// `classwords_per_codeword` is the classbook's `dimensions`, which
    /// is far smaller; this is a defensive upper bound, not a §8.6
    /// limit.)
    GroupTooLong(usize),
    /// A classification index was `>= num_classifications`. The unpack
    /// loop produces `temp % C`, which is always in `0..C`, so any
    /// index `>= C` has no on-wire round-trip.
    ClassificationOutOfRange {
        /// Position of the offending classification within the group
        /// (0 = the least-significant base-`C` digit).
        position: usize,
        /// The rejected classification index.
        classification: u32,
        /// `num_classifications` — the legal exclusive upper bound.
        num_classifications: u32,
    },
    /// The packed classbook entry index overflowed `u32`. Even with
    /// every digit in range, `Σ class[i]·C^i` can exceed `u32::MAX`
    /// for a large base and group length; the classbook read is a
    /// scalar `u32` decode, so an index above `u32::MAX` is
    /// unrepresentable on the wire.
    PackedValueOverflow {
        /// `num_classifications` (the digit base `C`).
        num_classifications: u32,
        /// The group length (the number of base-`C` digits).
        group_len: usize,
    },
}

impl fmt::Display for PackResidueClassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackResidueClassError::ZeroClassifications => write!(
                f,
                "vorbis residue classification (write): num_classifications=0 (must be 1..=64 per §8.6.1)"
            ),
            PackResidueClassError::ClassificationsTooLarge(c) => write!(
                f,
                "vorbis residue classification (write): num_classifications={c} > 64 (§8.6.1 caps at 64)"
            ),
            PackResidueClassError::EmptyGroup => write!(
                f,
                "vorbis residue classification (write): empty classification group (classwords_per_codeword must be >= 1 per §8.6.2)"
            ),
            PackResidueClassError::GroupTooLong(n) => write!(
                f,
                "vorbis residue classification (write): group length {n} > 32; packed index would exceed u32 (§8.6.2 scalar classbook read)"
            ),
            PackResidueClassError::ClassificationOutOfRange {
                position,
                classification,
                num_classifications,
            } => write!(
                f,
                "vorbis residue classification (write): classification[{position}]={classification} >= num_classifications={num_classifications} (no §8.6.2 unpack round-trip)"
            ),
            PackResidueClassError::PackedValueOverflow {
                num_classifications,
                group_len,
            } => write!(
                f,
                "vorbis residue classification (write): packed classbook index overflows u32 for base {num_classifications}, {group_len} digits (§8.6.2 scalar classbook read)"
            ),
        }
    }
}

impl std::error::Error for PackResidueClassError {}

/// Pack one residue classification group into the single classbook
/// **entry index** the §8.6.2 audio-packet residue decode reads in
/// scalar context (the `[temp] = [classbook] read` of step 9).
///
/// This is the exact arithmetic inverse of the §8.6.2 step-10..12
/// classbook *unpack* the residue decoder performs. The decoder, after
/// reading one classbook entry `temp`, recovers `classwords_per_codeword`
/// classifications by the descending loop
///
/// ```text
/// for i in (0 .. classwords).rev() {
///     classification[i] = temp % num_classifications;
///     temp /= num_classifications;
/// }
/// ```
///
/// The descending `(0..classwords).rev()` index means the *last*
/// iteration writes `classification[0]` from the lowest base-`C` digit,
/// so group position 0 is the **most**-significant digit and position
/// `classwords - 1` is the least-significant. With `C =
/// num_classifications` and `L = classwords`:
///
/// ```text
/// temp = Σ_{i=0}^{L-1} classification[i] · C^(L-1-i)
/// ```
///
/// `pack_residue_classifications` computes that sum. `classifications`
/// is one group of exactly `classwords_per_codeword` classification
/// indices in the decoder's array order (position 0 = the
/// most-significant base-`C` digit), and the returned `u32` is the
/// classbook entry index a residue-body writer then Huffman-codes with
/// the classbook (§8.6.2 step 9's inverse).
///
/// The round-trip property
///
/// ```text
/// unpack(pack_residue_classifications(group, C)?, group.len(), C) == group
/// ```
///
/// holds for every legal `group` and base `C` (where `unpack` is the
/// decoder's step-10..12 loop above).
///
/// # Errors
///
/// Returns a [`PackResidueClassError`] if `num_classifications` is
/// outside `1..=64`, the group is empty or longer than 32, any
/// classification index is `>= num_classifications`, or the packed
/// index would exceed `u32::MAX`. Validation precedes any arithmetic;
/// the function is a pure value-to-value transform with no side effects.
pub fn pack_residue_classifications(
    classifications: &[u32],
    num_classifications: u32,
) -> Result<u32, PackResidueClassError> {
    // §8.6.1: residue_classifications is a `read 6 bits + 1` field.
    if num_classifications == 0 {
        return Err(PackResidueClassError::ZeroClassifications);
    }
    if num_classifications > 64 {
        return Err(PackResidueClassError::ClassificationsTooLarge(
            num_classifications,
        ));
    }
    // §8.6.2: classwords_per_codeword (= classbook dimensions) is >= 1.
    if classifications.is_empty() {
        return Err(PackResidueClassError::EmptyGroup);
    }
    if classifications.len() > 32 {
        return Err(PackResidueClassError::GroupTooLong(classifications.len()));
    }

    // Validate every digit is in 0..num_classifications before packing
    // (the unpack loop only ever produces `temp % C`, which is < C).
    for (position, &classification) in classifications.iter().enumerate() {
        if classification >= num_classifications {
            return Err(PackResidueClassError::ClassificationOutOfRange {
                position,
                classification,
                num_classifications,
            });
        }
    }

    // temp = Σ class[i] · C^(L-1-i): group position 0 is the
    // most-significant digit, so Horner processes the group front-to-back
    // (`packed = packed·C + class[i]`). Every step is checked against
    // u32 overflow so an out-of-range packed index is refused rather than
    // wrapping silently.
    let base = num_classifications;
    let mut packed: u32 = 0;
    for &classification in classifications.iter() {
        packed = packed
            .checked_mul(base)
            .and_then(|p| p.checked_add(classification))
            .ok_or(PackResidueClassError::PackedValueOverflow {
                num_classifications,
                group_len: classifications.len(),
            })?;
    }
    Ok(packed)
}

/// Errors that may arise while grouping a full per-vector residue
/// classification array into the sequence of classbook entry indices the
/// §8.6.2 audio-packet residue decode reads (one classbook entry per
/// group of `classwords_per_codeword` partitions), via
/// [`pack_residue_classification_groups`].
///
/// The grouper is the structural inverse of the §8.6.2 step-6..9 decode
/// loop, which — on pass 0 — reads one classbook entry per `classwords`
/// partitions and unpacks it back into that many classifications. Each
/// variant flags an input the grouper cannot serialise to a classbook
/// entry sequence the decode loop would reproduce, so it refuses the
/// call rather than emit a value that fails to round-trip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackResidueClassGroupsError {
    /// `classwords_per_codeword` was zero. §8.6.2 derives it from the
    /// classbook's `dimensions`, which is `>= 1` for any decodable
    /// residue; a zero group width would read no classifications yet
    /// still consume a classbook codeword, which the spec's
    /// `classwords_per_codeword > 0` invariant forbids.
    ZeroClasswords,
    /// One classification group failed [`pack_residue_classifications`].
    /// The contained `group` index identifies which group (0-based, in
    /// stream order) raised the error, and `source` carries the inner
    /// per-group failure verbatim.
    Pack {
        /// The 0-based group index (in stream order) that failed.
        group: usize,
        /// The per-group packing error.
        source: PackResidueClassError,
    },
}

impl fmt::Display for PackResidueClassGroupsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackResidueClassGroupsError::ZeroClasswords => write!(
                f,
                "vorbis residue classification groups (write): classwords_per_codeword=0 (must be >= 1 per §8.6.2)"
            ),
            PackResidueClassGroupsError::Pack { group, source } => write!(
                f,
                "vorbis residue classification groups (write): group {group} failed to pack: {source}"
            ),
        }
    }
}

impl std::error::Error for PackResidueClassGroupsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PackResidueClassGroupsError::ZeroClasswords => None,
            PackResidueClassGroupsError::Pack { source, .. } => Some(source),
        }
    }
}

/// Group a full per-vector residue classification array into the sequence
/// of classbook **entry indices** the §8.6.2 audio-packet residue decode
/// reads — one classbook entry per group of `classwords_per_codeword`
/// partitions (§8.6.2 step 9's structural inverse).
///
/// This sits directly above [`pack_residue_classifications`]: the decoder
/// (§8.6.2 step 6..12), on pass 0, walks the partitions in
/// `classwords_per_codeword`-sized strides; at each stride it reads ONE
/// classbook entry in scalar context and unpacks it into that many
/// classifications (only storing those whose partition index is
/// `< partitions_to_read`). This function performs the inverse walk: it
/// slices `classifications` into consecutive groups of
/// `classwords_per_codeword`, packs each group with
/// [`pack_residue_classifications`], and returns one classbook entry per
/// group in stream order. A residue-body writer then Huffman-codes each
/// returned entry with the classbook.
///
/// # The final partial group
///
/// When `classifications.len()` is not a multiple of
/// `classwords_per_codeword`, the last group is padded on the **right**
/// (the least-significant base-`C` digits, positions
/// `len % classwords .. classwords`) with classification index `0`. This
/// mirrors the decoder exactly: its step-10..12 unpack loop reads all
/// `classwords` digits but discards any whose partition index is
/// `>= partitions_to_read` (`if slot < partitions_to_read`), so the
/// padding digits are read-and-thrown-away. Zero is the canonical pad —
/// it yields the smallest classbook entry index that round-trips every
/// meaningful (kept) classification.
///
/// `classifications` is the per-vector classification array in partition
/// order (`classifications[partition]`), exactly the `classifications[j]`
/// row the decoder builds for one decode vector `j`. An empty array
/// yields an empty result (no classbook entries — the decode loop reads
/// none when `partitions_to_read == 0`).
///
/// The round-trip property: for every group `g`, unpacking the returned
/// `entries[g]` with the decoder's step-10..12 loop reproduces
/// `classifications[g*classwords .. (g+1)*classwords]` (the final group's
/// kept prefix matches; its padded tail unpacks back to the `0` pads).
///
/// # Errors
///
/// Returns [`PackResidueClassGroupsError::ZeroClasswords`] if
/// `classwords_per_codeword` is zero, or
/// [`PackResidueClassGroupsError::Pack`] (tagged with the offending group
/// index) if any group fails [`pack_residue_classifications`] (an
/// out-of-range classification, an oversized base, or a packed-index
/// overflow). Validation precedes any output allocation growth past the
/// failing group; the function is a pure value-to-value transform with no
/// side effects.
pub fn pack_residue_classification_groups(
    classifications: &[u32],
    num_classifications: u32,
    classwords_per_codeword: usize,
) -> Result<Vec<u32>, PackResidueClassGroupsError> {
    // §8.6.2: classwords_per_codeword (= classbook dimensions) is >= 1.
    if classwords_per_codeword == 0 {
        return Err(PackResidueClassGroupsError::ZeroClasswords);
    }

    // Empty classification array => no classbook entries (the decode loop
    // runs zero groups when partitions_to_read == 0).
    if classifications.is_empty() {
        return Ok(Vec::new());
    }

    // Number of classbook entries = ceil(len / classwords). The final
    // group is right-padded with classification index 0 to a full width
    // of `classwords_per_codeword`.
    let total = classifications.len();
    let num_groups = total.div_ceil(classwords_per_codeword);
    let mut entries = Vec::with_capacity(num_groups);

    // Scratch buffer reused for each (possibly padded) group so the pad
    // path allocates nothing extra in the hot loop.
    let mut group = vec![0u32; classwords_per_codeword];
    for g in 0..num_groups {
        let start = g * classwords_per_codeword;
        let end = (start + classwords_per_codeword).min(total);
        let kept = end - start;
        // Copy the kept classifications into the most-significant
        // positions (0..kept); zero-pad the least-significant tail
        // (kept..classwords) — the decoder discards those digits.
        group[..kept].copy_from_slice(&classifications[start..end]);
        for slot in group.iter_mut().take(classwords_per_codeword).skip(kept) {
            *slot = 0;
        }
        let packed = pack_residue_classifications(&group, num_classifications)
            .map_err(|source| PackResidueClassGroupsError::Pack { group: g, source })?;
        entries.push(packed);
    }
    Ok(entries)
}

/// Errors that may arise while writing one residue partition body
/// (§8.6.3/§8.6.4/§8.6.5) via [`write_residue_partition`] or while
/// sizing one via [`residue_partition_codeword_count`].
///
/// The writer is the on-wire inverse of the residue decoder's
/// per-partition decode (§8.6.2 step 19's "decode partition ... using
/// codebook number `[vqbook]` in VQ context"). Each variant flags an
/// input that cannot serialise to a partition body the decoder would
/// read back, so the call is refused before any bits are emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteResiduePartitionError {
    /// `residue_type` was a value other than 0, 1, or 2. §8.6 defines
    /// only the three formats.
    UnsupportedResidueType(u16),
    /// `partition_size` was zero. §8.6.1 stores
    /// `residue_partition_size` as a `read 24 bits + 1` field, so the
    /// legal range starts at 1; a zero-size partition reads no
    /// codewords and has no on-wire body.
    ZeroPartitionSize,
    /// The value codebook's `dimensions` was zero. §8.6.3 divides the
    /// partition size by `[codebook_dimensions]` and §8.6.4 advances
    /// its read cursor by one codebook dimension per decoded element;
    /// a zero-dimension book can never cover a partition.
    ZeroDimensions,
    /// The value codebook has no value mapping (`codebook_lookup_type`
    /// 0). §8.6.1: a codebook used in VQ context must carry a vector
    /// lookup; a scalar (entropy-only) book cannot yield the
    /// `[entry_temp]` vector §8.6.3/§8.6.4 read.
    ScalarValueBook,
    /// Format 0 requires the codebook dimensions to evenly divide the
    /// partition size (§8.6.3 step 1 computes
    /// `[step] = [n] / [codebook_dimensions]` and steps 2..5 cover
    /// exactly `step × dimensions = n` scalars).
    Format0NotDivisible {
        /// `residue_partition_size` (the §8.6.3 `[n]`).
        partition_size: u32,
        /// The value codebook's `dimensions`.
        dimensions: u16,
    },
    /// The supplied entry list's length disagrees with the codeword
    /// count the decoder will read for this `(residue_type,
    /// partition_size, dimensions)` triple — `n / dims` for format 0,
    /// `ceil(n / dims)` for formats 1 and 2 (see
    /// [`residue_partition_codeword_count`]).
    EntryCountMismatch {
        /// The codeword count the decoder will read.
        expected: usize,
        /// The supplied `entries.len()`.
        actual: usize,
    },
    /// Building the value codebook's Huffman tree failed (the book's
    /// `codeword_lengths` do not describe a valid §3.2.1 canonical
    /// tree).
    Huffman(crate::huffman::BuildError),
    /// An entry index has no canonical codeword in the value codebook
    /// — it is out of range or marked unused (§3.2.1). No on-wire bit
    /// pattern decodes to it, so the body cannot round-trip.
    UnencodableEntry {
        /// Position of the offending entry within the supplied list.
        index: usize,
        /// The rejected entry index.
        entry: u32,
        /// Number of used (leaf) entries the book's tree holds.
        used_count: u32,
    },
}

impl fmt::Display for WriteResiduePartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteResiduePartitionError::UnsupportedResidueType(t) => write!(
                f,
                "vorbis residue partition (write): unsupported residue_type {t} (§8.6 defines 0, 1, 2)"
            ),
            WriteResiduePartitionError::ZeroPartitionSize => write!(
                f,
                "vorbis residue partition (write): partition_size=0 (§8.6.1 stores it as read-24-bits + 1, so >= 1)"
            ),
            WriteResiduePartitionError::ZeroDimensions => write!(
                f,
                "vorbis residue partition (write): value codebook dimensions=0 (§8.6.3/§8.6.4 advance by one dimension per scalar)"
            ),
            WriteResiduePartitionError::ScalarValueBook => write!(
                f,
                "vorbis residue partition (write): value codebook has lookup_type 0 (§8.6.1 requires a value mapping in VQ context)"
            ),
            WriteResiduePartitionError::Format0NotDivisible {
                partition_size,
                dimensions,
            } => write!(
                f,
                "vorbis residue partition (write): format-0 partition_size {partition_size} not divisible by codebook dimensions {dimensions} (§8.6.3 step 1)"
            ),
            WriteResiduePartitionError::EntryCountMismatch { expected, actual } => write!(
                f,
                "vorbis residue partition (write): {actual} entries supplied but the decoder reads {expected} codewords for this partition (§8.6.3/§8.6.4)"
            ),
            WriteResiduePartitionError::Huffman(e) => write!(
                f,
                "vorbis residue partition (write): Huffman build error: {e}"
            ),
            WriteResiduePartitionError::UnencodableEntry {
                index,
                entry,
                used_count,
            } => write!(
                f,
                "vorbis residue partition (write): entries[{index}]={entry} has no canonical codeword (tree has {used_count} used entries, §3.2.1)"
            ),
        }
    }
}

impl std::error::Error for WriteResiduePartitionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WriteResiduePartitionError::Huffman(e) => Some(e),
            _ => None,
        }
    }
}

impl From<crate::huffman::BuildError> for WriteResiduePartitionError {
    fn from(value: crate::huffman::BuildError) -> Self {
        WriteResiduePartitionError::Huffman(value)
    }
}

/// The number of VQ codewords the residue decoder reads for one
/// partition body, given the residue format, the partition size, and
/// the value codebook's dimensions.
///
/// * **Format 0 (§8.6.3).** Step 1 computes
///   `[step] = [n] / [codebook_dimensions]` and step 2 reads exactly
///   `[step]` VQ vectors (the dimensions must evenly divide the
///   partition size — [`WriteResiduePartitionError::Format0NotDivisible`]
///   otherwise).
/// * **Formats 1 and 2 (§8.6.4 / §8.6.5).** The decode loop reads one
///   VQ vector, appends up to `dimensions` scalars, and continues while
///   `[i] < [n]` — i.e. `ceil(n / dims)` reads, with any surplus
///   elements of the final vector discarded when `dims` does not divide
///   `n`. Format 2 is "reducible to format 1" (§8.6.5) and reads with
///   the same rule over the interleaved vector.
///
/// This is the count [`write_residue_partition`] requires its `entries`
/// list to match; it is exposed so callers can size their per-partition
/// entry lists before quantising.
///
/// # Errors
///
/// Returns a [`WriteResiduePartitionError`] for a `residue_type`
/// outside {0, 1, 2}, a zero `partition_size`, zero `dimensions`, or a
/// format-0 divisibility failure.
pub fn residue_partition_codeword_count(
    residue_type: u16,
    partition_size: u32,
    dimensions: u16,
) -> Result<usize, WriteResiduePartitionError> {
    if residue_type > 2 {
        return Err(WriteResiduePartitionError::UnsupportedResidueType(
            residue_type,
        ));
    }
    if partition_size == 0 {
        return Err(WriteResiduePartitionError::ZeroPartitionSize);
    }
    if dimensions == 0 {
        return Err(WriteResiduePartitionError::ZeroDimensions);
    }
    let n = partition_size as usize;
    let dims = dimensions as usize;
    if residue_type == 0 {
        // §8.6.3 step 1: [step] = [n] / [codebook_dimensions]; steps
        // 2..5 read exactly [step] VQ vectors covering step×dims = n
        // scalars, so dims must divide n.
        if n % dims != 0 {
            return Err(WriteResiduePartitionError::Format0NotDivisible {
                partition_size,
                dimensions,
            });
        }
        Ok(n / dims)
    } else {
        // §8.6.4 (and §8.6.5 via "reducible to format 1"): read one VQ
        // vector per `dims` scalars while [i] < [n] → ceil(n / dims).
        Ok(n.div_ceil(dims))
    }
}

/// Serialise one residue partition body (§8.6.3/§8.6.4/§8.6.5) to a
/// byte-aligned slice — the per-partition value-codeword WRITE
/// primitive, the value half of the §8.6.2 residue-body writer (the
/// classification half is [`pack_residue_classifications`] /
/// [`pack_residue_classification_groups`]).
///
/// `entries` is the partition's sequence of value-codebook **entry
/// indices** in stream order, exactly the entries the decoder's
/// partition decode walks the book's Huffman tree to: `n / dims`
/// entries for format 0 (§8.6.3's `[step]` reads), `ceil(n / dims)`
/// for formats 1 and 2 (§8.6.4's read-while-`[i] < [n]` loop) — see
/// [`residue_partition_codeword_count`]. Which entry best quantises a
/// given run of residue scalars is the encoder's psychoacoustic
/// choice; keeping the entry indices explicit (like the floor 1
/// packet writer's `partition_cvals` knob) lets a future VQ-encode
/// stage pick them without the writer guessing, and makes the
/// emission bit-exact by construction.
///
/// The emission itself is format-independent: every format reads the
/// partition's codewords one after another with the same book in VQ
/// context; the §8.6.3 scatter (`[offset]+[i]+[j]*[step]`) versus the
/// §8.6.4 contiguous append is decode-side *addressing* of the
/// unpacked scalars, not an on-wire difference. The formats differ
/// on the wire only in how many codewords the decoder reads, which is
/// what the fail-closed count gate pins.
///
/// The round-trip property: splicing the returned bits into a §8.6.2
/// packet stream after the partition's classbook codeword yields a
/// body the residue decoder reads back as exactly
/// `Σ unpack_vector(book, entries[k])` scattered per the format rule.
///
/// # Errors
///
/// Returns a [`WriteResiduePartitionError`] for a bad
/// `(residue_type, partition_size, dimensions)` triple (see
/// [`residue_partition_codeword_count`]), a value book without a
/// vector lookup (§8.6.1), an entry-count mismatch, a Huffman build
/// failure, or an entry with no canonical codeword. Validation
/// precedes emission; on error no bits are written.
pub fn write_residue_partition(
    entries: &[u32],
    book: &VorbisCodebook,
    residue_type: u16,
    partition_size: u32,
) -> Result<Vec<u8>, WriteResiduePartitionError> {
    let mut writer = BitWriterLsb::with_capacity(entries.len().div_ceil(2).max(1));
    write_residue_partition_into_writer(entries, book, residue_type, partition_size, &mut writer)?;
    Ok(writer.finish())
}

/// Bit-level helper to splice a §8.6.3/§8.6.4/§8.6.5 residue partition
/// body into a larger bit-packed stream. The wrapping §8.6.2
/// residue-body writer (explicit followup) will use this to thread
/// each partition body between the classbook codewords across the
/// pass/partition/vector loops, mirroring the existing per-header and
/// floor-1-packet `_into_writer` splice points.
///
/// Writes the body's bits into `writer` at its current bit position.
/// On error, no bits are emitted (validation precedes emission).
pub(crate) fn write_residue_partition_into_writer(
    entries: &[u32],
    book: &VorbisCodebook,
    residue_type: u16,
    partition_size: u32,
    writer: &mut BitWriterLsb,
) -> Result<(), WriteResiduePartitionError> {
    // ---- fail-closed invariant gate (no bits emitted on error). ----
    // (1) The (residue_type, partition_size, dimensions) triple pins the
    // codeword count the decoder will read for this partition.
    let expected = residue_partition_codeword_count(residue_type, partition_size, book.dimensions)?;

    // (2) §8.6.1: a book used in VQ context must carry a value mapping.
    if matches!(book.lookup, VqLookup::None) {
        return Err(WriteResiduePartitionError::ScalarValueBook);
    }

    // (3) The supplied entry list must match the decoder's read count.
    if entries.len() != expected {
        return Err(WriteResiduePartitionError::EntryCountMismatch {
            expected,
            actual: entries.len(),
        });
    }

    // (4) The book's lengths must build a canonical §3.2.1 tree, and
    // every entry must be one of its leaves. The encodability pre-check
    // runs against a scratch writer so the caller's stream is untouched
    // if any entry is refused.
    let tree = crate::huffman::HuffmanTree::from_codebook(book)?;
    for (index, &entry) in entries.iter().enumerate() {
        let mut scratch = BitWriterLsb::new();
        tree.encode_entry(entry, &mut scratch).map_err(|e| {
            let crate::huffman::EncodeError::UnknownEntry { used_count, .. } = e;
            WriteResiduePartitionError::UnencodableEntry {
                index,
                entry,
                used_count,
            }
        })?;
    }

    // ---- emit: one canonical codeword per entry, in stream order. ----
    for &entry in entries {
        tree.encode_entry(entry, writer)
            .expect("encode_entry must succeed after pre-check; entry already validated");
    }
    Ok(())
}

/// The encoder's description of one §8.6.2 decode vector's residue
/// content — the per-vector input to [`write_residue_body`].
///
/// A *decode vector* is what the §8.6.2 loop iterates `[j]` over: one
/// per channel in the submap bundle for formats 0 and 1, or the single
/// interleaved vector for format 2 (§8.6.5 reduces all channels to one
/// format-1 decode). [`residue_body_shape`] reports how many decode
/// vectors a given `(header, blocksize, do_not_decode)` triple expects
/// and how many partitions each spans.
///
/// Both members are the encoder's explicit quantisation choices —
/// the same knob philosophy as the floor 1 packet writer's
/// `partition_cvals` and the round-37 partition writer's `entries`:
/// the writer serialises exactly what it is told, bit-exact by
/// construction, and a future VQ-encode stage picks the values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResidueVectorPlan {
    /// `classifications[partition]` for this decode vector — exactly
    /// the `[classifications]` row the §8.6.2 step-11 unpack rebuilds.
    /// Length must equal [`ResidueBodyShape::partitions_to_read`] for
    /// a decoded vector, and 0 for a 'do not decode' vector (the
    /// decoder reads nothing for those).
    pub classifications: Vec<u32>,
    /// `partition_entries[partition][pass]` — the per-partition,
    /// per-cascade-stage value-codebook entry-index list (the round-37
    /// [`write_residue_partition`] `entries` argument). Must be `Some`
    /// exactly where the header's `residue_books[class][pass]` holds a
    /// book for this partition's classification (§8.6.2 step 18 skips
    /// 'unused' stages), with the list length pinned by
    /// [`residue_partition_codeword_count`]. Outer length mirrors
    /// [`Self::classifications`].
    pub partition_entries: Vec<[Option<Vec<u32>>; 8]>,
}

/// The §8.6.2 step-1..5 derived shape of one residue body — how many
/// decode vectors [`write_residue_body`] expects plans for, and how
/// many partitions each plan spans. Returned by [`residue_body_shape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidueBodyShape {
    /// The number of §8.6.2 decode vectors: `do_not_decode.len()` for
    /// formats 0 and 1 (one per channel, 'do not decode' ones
    /// included), 1 for format 2 (the single interleaved vector), or 0
    /// for format 2 when every channel is marked 'do not decode'
    /// (§8.6.5: "if all vectors are marked 'do not decode', no decode
    /// occurs").
    pub vectors: usize,
    /// `[partitions_to_read] = [n_to_read] / [residue_partition_size]`
    /// after the §8.6.2 step-1..5 begin/end limiting — the required
    /// length of each decoded vector's [`ResidueVectorPlan`] rows.
    pub partitions_to_read: usize,
}

/// Errors that may arise while writing a full §8.6.2 residue body via
/// [`write_residue_body`] or while sizing one via
/// [`residue_body_shape`].
///
/// The writer is the on-wire inverse of the §8.6.2 packet-decode loop.
/// Each variant flags an input that cannot serialise to a body the
/// residue decoder would read back, so the call is refused before any
/// bits are emitted (validation precedes emission in full).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteResidueBodyError {
    /// `residue_type` was a value other than 0, 1, or 2 (§8.6).
    UnsupportedResidueType(u16),
    /// `partition_size` was zero. §8.6.1 stores it as a
    /// `read 24 bits + 1` field, so the legal range starts at 1.
    ZeroPartitionSize,
    /// The `residue_classbook` index points outside the supplied
    /// codebook table (§8.6.1's undecodability clause — mirrored from
    /// the decoder's construction-time gate).
    ClassbookOutOfRange {
        /// The offending `residue_classbook` index.
        classbook: u8,
        /// The number of codebooks available.
        codebook_count: usize,
    },
    /// The classbook's `dimensions` is zero. §8.6.2 derives
    /// `classwords_per_codeword` from it; a zero group width cannot
    /// describe a classbook codeword (mirrors the decoder's gate).
    ZeroClasswordsPerCodeword,
    /// A `residue_books[class][stage]` index points outside the
    /// supplied codebook table (§8.6.1).
    ValueBookOutOfRange {
        /// Classification index.
        class: usize,
        /// Cascade stage index `0..=7`.
        stage: usize,
        /// The offending codebook index.
        book: u8,
        /// The number of codebooks available.
        codebook_count: usize,
    },
    /// A `residue_books[class][stage]` codebook has `lookup_type` 0.
    /// §8.6.1: a book used in VQ context must carry a value mapping.
    ValueBookHasNoLookup {
        /// Classification index.
        class: usize,
        /// Cascade stage index `0..=7`.
        stage: usize,
        /// The offending codebook index.
        book: u8,
    },
    /// `plans.len()` disagrees with the decode-vector count derived by
    /// [`residue_body_shape`] (one plan per channel for formats 0/1,
    /// one for the interleaved vector for format 2, zero for format
    /// 2's all-'do not decode' shortcut).
    PlanCountMismatch {
        /// The decode-vector count the §8.6.2 loop iterates.
        expected: usize,
        /// The supplied `plans.len()`.
        actual: usize,
    },
    /// A 'do not decode' vector's plan carried classifications or
    /// partition entries. The decoder reads nothing for such a vector
    /// (§8.6.2 step 8 / step 15 skip it), so non-empty plan content
    /// can never reach the wire — the writer refuses rather than
    /// silently drop it.
    DoNotDecodePlanNotEmpty {
        /// The offending decode-vector index.
        vector: usize,
    },
    /// A decoded vector's `classifications` length disagrees with
    /// `partitions_to_read`.
    ClassificationCountMismatch {
        /// The offending decode-vector index.
        vector: usize,
        /// `partitions_to_read`.
        expected: usize,
        /// The supplied row length.
        actual: usize,
    },
    /// A decoded vector's `partition_entries` length disagrees with
    /// `partitions_to_read`.
    PartitionEntriesCountMismatch {
        /// The offending decode-vector index.
        vector: usize,
        /// `partitions_to_read`.
        expected: usize,
        /// The supplied row length.
        actual: usize,
    },
    /// A classification index was `>= residue_classifications`. The
    /// §8.6.2 step-11 unpack (`temp % residue_classifications`) can
    /// never reproduce it, and it cannot index `residue_books`.
    ClassificationOutOfRange {
        /// The offending decode-vector index.
        vector: usize,
        /// The offending partition index.
        partition: usize,
        /// The rejected classification.
        classification: u32,
        /// `residue_classifications` — the exclusive upper bound.
        num_classifications: u32,
    },
    /// The cascade holds a value book for this (partition, pass) pair
    /// but the plan supplies no entry list — the decoder would read a
    /// partition body the writer has nothing to emit for.
    MissingPartitionEntries {
        /// The offending decode-vector index.
        vector: usize,
        /// The offending partition index.
        partition: usize,
        /// The cascade stage (pass) index `0..=7`.
        pass: usize,
    },
    /// The plan supplies an entry list for a (partition, pass) pair
    /// whose cascade stage is 'unused' — the decoder reads nothing
    /// there (§8.6.2 step 18), so the list can never reach the wire.
    UnexpectedPartitionEntries {
        /// The offending decode-vector index.
        vector: usize,
        /// The offending partition index.
        partition: usize,
        /// The cascade stage (pass) index `0..=7`.
        pass: usize,
    },
    /// Packing a vector's classifications into classbook entry indices
    /// failed ([`pack_residue_classification_groups`] — an
    /// out-of-range base or a packed-index overflow).
    ClassificationPack {
        /// The offending decode-vector index.
        vector: usize,
        /// The per-group packing error (carries the group index).
        source: PackResidueClassGroupsError,
    },
    /// A packed classbook entry index has no canonical codeword in the
    /// classbook — out of range or marked unused (§3.2.1). No on-wire
    /// bit pattern decodes to it.
    UnencodableClassbookEntry {
        /// The offending decode-vector index.
        vector: usize,
        /// The offending classification group index (in stream order).
        group: usize,
        /// The rejected classbook entry index.
        entry: u32,
        /// Number of used (leaf) entries the classbook's tree holds.
        used_count: u32,
    },
    /// Building the classbook's Huffman tree failed (§3.2.1).
    Huffman(crate::huffman::BuildError),
    /// One partition body failed the round-37 per-partition writer's
    /// validation ([`write_residue_partition`] — entry-count mismatch,
    /// scalar book, unencodable entry, …).
    Partition {
        /// The offending decode-vector index.
        vector: usize,
        /// The offending partition index.
        partition: usize,
        /// The cascade stage (pass) index `0..=7`.
        pass: usize,
        /// The per-partition writer's error, verbatim.
        source: WriteResiduePartitionError,
    },
}

impl fmt::Display for WriteResidueBodyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteResidueBodyError::UnsupportedResidueType(t) => write!(
                f,
                "vorbis residue body (write): unsupported residue_type {t} (§8.6 defines 0, 1, 2)"
            ),
            WriteResidueBodyError::ZeroPartitionSize => write!(
                f,
                "vorbis residue body (write): partition_size=0 (§8.6.1 stores it as read-24-bits + 1, so >= 1)"
            ),
            WriteResidueBodyError::ClassbookOutOfRange {
                classbook,
                codebook_count,
            } => write!(
                f,
                "vorbis residue body (write): classbook index {classbook} >= codebook count {codebook_count} (§8.6.1)"
            ),
            WriteResidueBodyError::ZeroClasswordsPerCodeword => write!(
                f,
                "vorbis residue body (write): classbook dimensions=0 (§8.6.2 derives classwords_per_codeword from it)"
            ),
            WriteResidueBodyError::ValueBookOutOfRange {
                class,
                stage,
                book,
                codebook_count,
            } => write!(
                f,
                "vorbis residue body (write): residue_books[{class}][{stage}]={book} >= codebook count {codebook_count} (§8.6.1)"
            ),
            WriteResidueBodyError::ValueBookHasNoLookup { class, stage, book } => write!(
                f,
                "vorbis residue body (write): residue_books[{class}][{stage}]={book} has lookup_type 0 (§8.6.1 requires a value mapping in VQ context)"
            ),
            WriteResidueBodyError::PlanCountMismatch { expected, actual } => write!(
                f,
                "vorbis residue body (write): {actual} vector plans supplied but the §8.6.2 loop decodes {expected} vectors"
            ),
            WriteResidueBodyError::DoNotDecodePlanNotEmpty { vector } => write!(
                f,
                "vorbis residue body (write): vector {vector} is marked 'do not decode' but its plan is non-empty (§8.6.2 reads nothing for it)"
            ),
            WriteResidueBodyError::ClassificationCountMismatch {
                vector,
                expected,
                actual,
            } => write!(
                f,
                "vorbis residue body (write): vector {vector} supplies {actual} classifications but partitions_to_read is {expected} (§8.6.2)"
            ),
            WriteResidueBodyError::PartitionEntriesCountMismatch {
                vector,
                expected,
                actual,
            } => write!(
                f,
                "vorbis residue body (write): vector {vector} supplies {actual} partition-entry rows but partitions_to_read is {expected} (§8.6.2)"
            ),
            WriteResidueBodyError::ClassificationOutOfRange {
                vector,
                partition,
                classification,
                num_classifications,
            } => write!(
                f,
                "vorbis residue body (write): vector {vector} partition {partition} classification {classification} >= residue_classifications {num_classifications} (§8.6.2)"
            ),
            WriteResidueBodyError::MissingPartitionEntries {
                vector,
                partition,
                pass,
            } => write!(
                f,
                "vorbis residue body (write): vector {vector} partition {partition} pass {pass} has a cascade book but no entry list was supplied (§8.6.2 step 19)"
            ),
            WriteResidueBodyError::UnexpectedPartitionEntries {
                vector,
                partition,
                pass,
            } => write!(
                f,
                "vorbis residue body (write): vector {vector} partition {partition} pass {pass} supplies entries but the cascade stage is unused (§8.6.2 step 18)"
            ),
            WriteResidueBodyError::ClassificationPack { vector, source } => write!(
                f,
                "vorbis residue body (write): vector {vector} classification packing failed: {source}"
            ),
            WriteResidueBodyError::UnencodableClassbookEntry {
                vector,
                group,
                entry,
                used_count,
            } => write!(
                f,
                "vorbis residue body (write): vector {vector} group {group} packs to classbook entry {entry} which has no canonical codeword (tree has {used_count} used entries, §3.2.1)"
            ),
            WriteResidueBodyError::Huffman(e) => write!(
                f,
                "vorbis residue body (write): classbook Huffman build error: {e}"
            ),
            WriteResidueBodyError::Partition {
                vector,
                partition,
                pass,
                source,
            } => write!(
                f,
                "vorbis residue body (write): vector {vector} partition {partition} pass {pass}: {source}"
            ),
        }
    }
}

impl std::error::Error for WriteResidueBodyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WriteResidueBodyError::ClassificationPack { source, .. } => Some(source),
            WriteResidueBodyError::Partition { source, .. } => Some(source),
            WriteResidueBodyError::Huffman(e) => Some(e),
            _ => None,
        }
    }
}

impl From<crate::huffman::BuildError> for WriteResidueBodyError {
    fn from(value: crate::huffman::BuildError) -> Self {
        WriteResidueBodyError::Huffman(value)
    }
}

/// The §8.6.2 step-1..5 derived shape of one residue body: how many
/// decode vectors the §8.6.2 loop iterates (= the number of
/// [`ResidueVectorPlan`]s [`write_residue_body`] expects) and how many
/// partitions each spans (= the required plan row lengths).
///
/// The shape mirrors the decoder's setup arithmetic exactly:
///
/// 1. `[actual_size] = blocksize / 2`, multiplied by the channel count
///    for format 2 (§8.6.2 steps 1..3 — format 2 decodes one
///    interleaved vector spanning all channels).
/// 2. `[limit_residue_begin/end] = min(residue_begin/end,
///    [actual_size])` (§8.6.2 steps 4..5).
/// 3. `[partitions_to_read] = ([limit_residue_end] -
///    [limit_residue_begin]) / [residue_partition_size]`.
///
/// The decode-vector count is `do_not_decode.len()` for formats 0 and
/// 1 (every channel gets a plan; 'do not decode' channels must supply
/// an empty one), and for format 2 it is 1 — or 0 when every channel
/// is marked 'do not decode' (§8.6.5: no decode occurs at all).
///
/// # Errors
///
/// Returns a [`WriteResidueBodyError`] for a `residue_type` outside
/// {0, 1, 2} or a zero `partition_size`.
pub fn residue_body_shape(
    header: &ResidueHeader,
    blocksize: usize,
    do_not_decode: &[bool],
) -> Result<ResidueBodyShape, WriteResidueBodyError> {
    if header.residue_type > 2 {
        return Err(WriteResidueBodyError::UnsupportedResidueType(
            header.residue_type,
        ));
    }
    if header.partition_size == 0 {
        return Err(WriteResidueBodyError::ZeroPartitionSize);
    }
    let ch = do_not_decode.len();
    // §8.6.5: format 2 with every vector marked 'do not decode' (which
    // includes the degenerate zero-channel bundle) performs no decode —
    // there is no decode vector and no partition.
    if header.residue_type == 2 && do_not_decode.iter().all(|&dnd| dnd) {
        return Ok(ResidueBodyShape {
            vectors: 0,
            partitions_to_read: 0,
        });
    }
    // §8.6.2 steps 1..5: limit begin/end to the actual vector size
    // (`blocksize/2`, times `ch` for format 2's interleaved vector).
    let per_channel_size = (blocksize / 2) as u32;
    let actual_size = if header.residue_type == 2 {
        per_channel_size.saturating_mul(ch as u32)
    } else {
        per_channel_size
    };
    let limit_begin = header.residue_begin.min(actual_size);
    let limit_end = header.residue_end.min(actual_size);
    let n_to_read = limit_end.saturating_sub(limit_begin);
    let partitions_to_read = (n_to_read / header.partition_size) as usize;
    let vectors = if header.residue_type == 2 { 1 } else { ch };
    Ok(ResidueBodyShape {
        vectors,
        partitions_to_read,
    })
}

/// Serialise one full §8.6.2 residue body to a byte-aligned slice —
/// the wrapping residue-body WRITE primitive that interleaves the
/// classbook codewords ([`pack_residue_classification_groups`] +
/// [`crate::huffman::HuffmanTree::encode_entry`]) with the round-37
/// per-partition value-codeword bodies ([`write_residue_partition`])
/// across the §8.6.2 pass/partition/vector loops.
///
/// The emission order is the exact inverse of the decoder's §8.6.2
/// step-3..21 read order:
///
/// * Passes run 0..=7 (step 3).
/// * On pass 0, each stride of `classwords_per_codeword` partitions is
///   preceded by ONE classbook codeword per decoded vector (steps
///   6..12) — the codeword packs the stride's classifications, with
///   the final partial stride right-padded by zero digits the decoder
///   reads-and-discards.
/// * On every pass, each (partition, vector) pair whose classification
///   selects a cascade stage holding a value book emits that
///   partition's body (steps 13..20); 'unused' stages and 'do not
///   decode' vectors emit nothing (steps 15 / 18).
///
/// `plans` holds one [`ResidueVectorPlan`] per §8.6.2 decode vector —
/// `do_not_decode.len()` of them for formats 0 and 1 (the 'do not
/// decode' ones empty), a single interleaved-vector plan for format 2,
/// or none at all for format 2's all-'do not decode' shortcut
/// (§8.6.5). [`residue_body_shape`] reports both the expected plan
/// count and the required row length. `blocksize` and `do_not_decode`
/// are the same per-packet context the decoder's `decode` receives.
///
/// The round-trip property: feeding the returned bytes to the residue
/// decoder built from the same `(header, codebooks)` with the same
/// `(blocksize, do_not_decode)` reproduces exactly the vectors implied
/// by the plans (each partition body accumulating
/// `Σ unpack_vector(book, entries[k])` per the format's addressing
/// rule).
///
/// # Errors
///
/// Returns a [`WriteResidueBodyError`] for any §8.6.1 header/codebook
/// inconsistency (mirroring the decoder's construction-time gates), a
/// plan whose shape disagrees with the §8.6.2-derived geometry, a
/// classification with no unpack round-trip, an entry list present or
/// absent against the cascade, or any per-partition failure (carried
/// verbatim). Validation precedes emission in full; on error nothing
/// is written.
pub fn write_residue_body(
    plans: &[ResidueVectorPlan],
    header: &ResidueHeader,
    codebooks: &[VorbisCodebook],
    blocksize: usize,
    do_not_decode: &[bool],
) -> Result<Vec<u8>, WriteResidueBodyError> {
    let mut writer = BitWriterLsb::new();
    write_residue_body_into_writer(
        plans,
        header,
        codebooks,
        blocksize,
        do_not_decode,
        &mut writer,
    )?;
    Ok(writer.finish())
}

/// Bit-level helper to splice a §8.6.2 residue body into a larger
/// bit-packed stream. The wrapping §4.3 audio-packet writer (explicit
/// followup) will use this to thread the residue body between the
/// per-channel floor bodies and end-of-packet, mirroring the existing
/// per-header and floor-1-packet `_into_writer` splice points.
///
/// Writes the body's bits into `writer` at its current bit position.
/// On error, no bits are emitted (validation precedes emission).
pub(crate) fn write_residue_body_into_writer(
    plans: &[ResidueVectorPlan],
    header: &ResidueHeader,
    codebooks: &[VorbisCodebook],
    blocksize: usize,
    do_not_decode: &[bool],
    writer: &mut BitWriterLsb,
) -> Result<(), WriteResidueBodyError> {
    // ---- fail-closed validation (no bits emitted on any error). ----
    // (1) §8.6.2 steps 1..5: derive the decode-vector count and
    // partitions_to_read (also gates residue_type / partition_size).
    let shape = residue_body_shape(header, blocksize, do_not_decode)?;
    let partitions_to_read = shape.partitions_to_read;

    // (2) §8.6.1/§8.6.2: the classbook must exist, have dimensions
    // >= 1, and build a canonical §3.2.1 tree.
    let classbook = codebooks.get(header.classbook as usize).ok_or({
        WriteResidueBodyError::ClassbookOutOfRange {
            classbook: header.classbook,
            codebook_count: codebooks.len(),
        }
    })?;
    let classwords = classbook.dimensions as usize;
    if classwords == 0 {
        return Err(WriteResidueBodyError::ZeroClasswordsPerCodeword);
    }
    let classbook_tree = crate::huffman::HuffmanTree::from_codebook(classbook)?;
    let num_classifications = header.classifications as u32;

    // (3) §8.6.1: resolve and validate every configured (class, stage)
    // value book, mirroring the decoder's construction-time checks.
    let class_count = header.classifications as usize;
    let mut value_books: Vec<[Option<&VorbisCodebook>; 8]> = Vec::with_capacity(class_count);
    for (class, stage_books) in header.books.iter().take(class_count).enumerate() {
        let mut row: [Option<&VorbisCodebook>; 8] = [None; 8];
        for (stage, slot) in stage_books.iter().enumerate() {
            let Some(book_idx) = slot else { continue };
            let book = codebooks.get(*book_idx as usize).ok_or({
                WriteResidueBodyError::ValueBookOutOfRange {
                    class,
                    stage,
                    book: *book_idx,
                    codebook_count: codebooks.len(),
                }
            })?;
            if matches!(book.lookup, VqLookup::None) {
                return Err(WriteResidueBodyError::ValueBookHasNoLookup {
                    class,
                    stage,
                    book: *book_idx,
                });
            }
            row[stage] = Some(book);
        }
        value_books.push(row);
    }
    // A malformed header may carry fewer book rows than
    // classifications; missing rows are all-'unused' (mirrors the
    // decoder's padding).
    while value_books.len() < class_count {
        value_books.push([None; 8]);
    }

    // (4) One plan per §8.6.2 decode vector.
    if plans.len() != shape.vectors {
        return Err(WriteResidueBodyError::PlanCountMismatch {
            expected: shape.vectors,
            actual: plans.len(),
        });
    }
    // Per-decode-vector 'do not decode' flags: per channel for formats
    // 0/1; format 2's single interleaved vector is always decoded
    // (the all-'do not decode' case has zero vectors).
    let vector_dnd: Vec<bool> = if header.residue_type == 2 {
        vec![false; shape.vectors]
    } else {
        do_not_decode.to_vec()
    };

    // (5) Per-vector plan validation + classbook-entry precomputation.
    let mut group_entries: Vec<Vec<u32>> = Vec::with_capacity(plans.len());
    for (j, plan) in plans.iter().enumerate() {
        if vector_dnd[j] {
            // §8.6.2 steps 8 / 15: nothing is read for this vector, so
            // nothing can be written — refuse non-empty content.
            if !plan.classifications.is_empty() || !plan.partition_entries.is_empty() {
                return Err(WriteResidueBodyError::DoNotDecodePlanNotEmpty { vector: j });
            }
            group_entries.push(Vec::new());
            continue;
        }
        if plan.classifications.len() != partitions_to_read {
            return Err(WriteResidueBodyError::ClassificationCountMismatch {
                vector: j,
                expected: partitions_to_read,
                actual: plan.classifications.len(),
            });
        }
        if plan.partition_entries.len() != partitions_to_read {
            return Err(WriteResidueBodyError::PartitionEntriesCountMismatch {
                vector: j,
                expected: partitions_to_read,
                actual: plan.partition_entries.len(),
            });
        }
        for (partition, (&class, stages)) in plan
            .classifications
            .iter()
            .zip(plan.partition_entries.iter())
            .enumerate()
        {
            if class >= num_classifications {
                return Err(WriteResidueBodyError::ClassificationOutOfRange {
                    vector: j,
                    partition,
                    classification: class,
                    num_classifications,
                });
            }
            let row = &value_books[class as usize];
            for (pass, (book, supplied)) in row.iter().zip(stages.iter()).enumerate() {
                match (book, supplied) {
                    // §8.6.2 step 19: this (partition, pass) pair emits
                    // a partition body — validate it via the round-37
                    // per-partition writer against a scratch stream.
                    (Some(book), Some(entries)) => {
                        let mut scratch = BitWriterLsb::new();
                        write_residue_partition_into_writer(
                            entries,
                            book,
                            header.residue_type,
                            header.partition_size,
                            &mut scratch,
                        )
                        .map_err(|source| {
                            WriteResidueBodyError::Partition {
                                vector: j,
                                partition,
                                pass,
                                source,
                            }
                        })?;
                    }
                    (Some(_), None) => {
                        return Err(WriteResidueBodyError::MissingPartitionEntries {
                            vector: j,
                            partition,
                            pass,
                        });
                    }
                    (None, Some(_)) => {
                        return Err(WriteResidueBodyError::UnexpectedPartitionEntries {
                            vector: j,
                            partition,
                            pass,
                        });
                    }
                    // §8.6.2 step 18: 'unused' stage — nothing on wire.
                    (None, None) => {}
                }
            }
        }
        // §8.6.2 steps 9..12 inverse: pack this vector's classification
        // array into stream-order classbook entries (final partial
        // group right-padded with the digits the decoder discards).
        let entries = pack_residue_classification_groups(
            &plan.classifications,
            num_classifications,
            classwords,
        )
        .map_err(|source| WriteResidueBodyError::ClassificationPack { vector: j, source })?;
        // Each packed entry must hold a canonical classbook codeword.
        for (group, &entry) in entries.iter().enumerate() {
            let mut scratch = BitWriterLsb::new();
            classbook_tree
                .encode_entry(entry, &mut scratch)
                .map_err(|e| {
                    let crate::huffman::EncodeError::UnknownEntry { used_count, .. } = e;
                    WriteResidueBodyError::UnencodableClassbookEntry {
                        vector: j,
                        group,
                        entry,
                        used_count,
                    }
                })?;
        }
        group_entries.push(entries);
    }

    // ---- emission: the §8.6.2 step-3..21 loop, write direction.  ----
    // Every value below is pre-validated; no failure path remains.
    if partitions_to_read == 0 {
        // §8.6.2 step 2: no residue to decode — the body is empty.
        return Ok(());
    }
    // §8.6.2 step 3: iterate passes 0..=7. The pass index addresses
    // the stage column of BOTH `value_books` and each plan's
    // `partition_entries` row inside the partition walk, so the
    // §8.6.2 loop shape is kept verbatim rather than rewritten as a
    // single-container iterator.
    #[allow(clippy::needless_range_loop)]
    for pass in 0..8usize {
        // §8.6.2 step 4.
        let mut partition_count = 0usize;
        // §8.6.2 step 5.
        while partition_count < partitions_to_read {
            // §8.6.2 step 6: on pass 0, one classbook codeword per
            // decoded vector precedes each stride of `classwords`
            // partitions (steps 7..12).
            if pass == 0 {
                let group = partition_count / classwords;
                for (j, dnd) in vector_dnd.iter().enumerate() {
                    if *dnd {
                        continue;
                    }
                    classbook_tree
                        .encode_entry(group_entries[j][group], writer)
                        .expect("classbook entry pre-validated against the tree");
                }
            }
            // §8.6.2 step 13: decode (here: emit) this stride.
            let mut i = 0usize;
            while i < classwords && partition_count < partitions_to_read {
                // §8.6.2 step 14: iterate the vectors.
                for (j, dnd) in vector_dnd.iter().enumerate() {
                    if *dnd {
                        continue;
                    }
                    // §8.6.2 steps 16..18: classification → stage book.
                    let class = plans[j].classifications[partition_count] as usize;
                    if let Some(book) = value_books[class][pass] {
                        // §8.6.2 step 19 inverse: the partition body.
                        let entries = plans[j].partition_entries[partition_count][pass]
                            .as_ref()
                            .expect("entry-list presence pre-validated against the cascade");
                        write_residue_partition_into_writer(
                            entries,
                            book,
                            header.residue_type,
                            header.partition_size,
                            writer,
                        )
                        .expect("partition body pre-validated");
                    }
                }
                // §8.6.2 step 20.
                partition_count += 1;
                i += 1;
            }
        }
    }
    Ok(())
}

/// One submap's residue bundle — the §4.3.4 step-2/step-7 gather of the
/// channels assigned to a submap, with the per-bundle `do_not_decode`
/// flags the submap's residue decode (and the encoder-side
/// [`write_residue_body`]) consumes.
///
/// The decoder builds this implicitly inside the §4.3.4 submap loop:
/// "for each channel `j` in order `0 .. audio_channels-1`, if `mux[j] ==
/// i` then `do_not_decode_flag[ch] = no_residue[j]`; increment `ch`".
/// [`SubmapResidueBundle`] captures the same bundle so an encoder can
/// build one `do_not_decode` slice per submap to feed
/// [`write_residue_body`], and the §4.3.4 step-7 inverse scatter
/// ("residue vector for channel `j` is set to decoded residue vector
/// `ch`") is recoverable from [`Self::channels`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmapResidueBundle {
    /// The submap index (`i` in §4.3.4) this bundle decodes.
    pub submap: usize,
    /// The channels assigned to this submap, in ascending channel
    /// order (§4.3.4 step 2's `j` walk). Element `ch` of the bundle is
    /// channel `channels[ch]`; this is the §4.3.4 step-7 scatter map.
    pub channels: Vec<usize>,
    /// `do_not_decode_flag[ch]` for the bundle — the post-§4.3.3
    /// `no_residue` flag of channel `channels[ch]` (§4.3.4 step 2.i).
    /// Same length as [`Self::channels`]; this is the slice the
    /// submap's [`write_residue_body`] / residue decode receives.
    pub do_not_decode: Vec<bool>,
}

/// The full §4.3.3 + §4.3.4 residue-bundle plan for one audio packet:
/// the post-coupling `no_residue` vector and one
/// [`SubmapResidueBundle`] per submap, in submap order.
///
/// This is the inverse-mapping layer between the per-channel floor
/// decode (each channel's §4.3.2 step-6 `no_residue` flag, set when its
/// floor decoded 'unused') and the per-submap residue body writer: a
/// wrapping §4.3 audio-packet writer derives the raw `no_residue` flags
/// from its floor choices, calls [`plan_residue_bundles`], then threads
/// one [`write_residue_body`] per [`Self::bundles`] entry (in submap
/// order, §4.3.4) using that bundle's [`SubmapResidueBundle::channels`]
/// to gather the per-channel residue plans and
/// [`SubmapResidueBundle::do_not_decode`] as the body's flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidueBundlePlan {
    /// `no_residue` after §4.3.3 nonzero-vector propagation. Index `i`
    /// is channel `i`; `true` means the channel's residue is not coded
    /// in the stream (its floor was 'unused' and no coupling step
    /// pulled it back in). Length equals the channel count passed in.
    pub no_residue: Vec<bool>,
    /// One bundle per submap, in submap order `0 .. submaps-1`
    /// (§4.3.4's outer loop). A submap with no channels assigned still
    /// gets an (empty) bundle so the index lines up with
    /// `mapping.submap_configs`.
    pub bundles: Vec<SubmapResidueBundle>,
}

/// Errors from [`plan_residue_bundles`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanResidueBundlesError {
    /// The mapping declared zero submaps. A well-formed Vorbis I
    /// mapping has `submaps >= 1` (the `submaps_flag`-unset path pins
    /// it to exactly 1), so a zero here cannot describe a real packet.
    ZeroSubmaps,
    /// A coupling step named a channel at or past the channel count.
    /// Mirrors the §4.3.3 [`crate::packet::nonzero_propagate`] bounds
    /// check.
    CouplingChannelOutOfRange {
        /// The §4.3.3 coupling-step index that named the channel.
        step: usize,
        /// The out-of-range channel index.
        channel: usize,
        /// The packet's channel count (`no_residue.len()`).
        channels: usize,
    },
    /// A channel's `mux` entry selected a submap at or past
    /// `mapping.submaps`. Mirrors the decoder's submap-bounds gate.
    SubmapOutOfRange {
        /// The channel whose `mux` entry was out of range.
        channel: usize,
        /// The out-of-range submap index `mux[channel]`.
        submap: usize,
        /// The mapping's declared submap count.
        submaps: usize,
    },
    /// The mapping declared `submaps > 1` but its `mux` table is too
    /// short to cover every channel. §4.2.4 writes one `mux` entry per
    /// `audio_channels`; a multi-submap mapping with a short table
    /// cannot route every channel to a submap.
    MuxTooShort {
        /// The channel with no `mux` entry.
        channel: usize,
        /// The length of `mapping.mux`.
        mux_len: usize,
    },
}

impl core::fmt::Display for PlanResidueBundlesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PlanResidueBundlesError::ZeroSubmaps => {
                write!(
                    f,
                    "vorbis §4.3.4 residue-bundle plan: mapping declared zero submaps"
                )
            }
            PlanResidueBundlesError::CouplingChannelOutOfRange {
                step,
                channel,
                channels,
            } => write!(
                f,
                "vorbis §4.3.3 residue-bundle plan: coupling step {step} named channel \
                 {channel} but the packet has {channels} channel(s)"
            ),
            PlanResidueBundlesError::SubmapOutOfRange {
                channel,
                submap,
                submaps,
            } => write!(
                f,
                "vorbis §4.3.4 residue-bundle plan: channel {channel} mux selected submap \
                 {submap} but the mapping declares {submaps} submap(s)"
            ),
            PlanResidueBundlesError::MuxTooShort { channel, mux_len } => write!(
                f,
                "vorbis §4.3.4 residue-bundle plan: multi-submap mapping has no mux entry \
                 for channel {channel} (mux table length {mux_len})"
            ),
        }
    }
}

impl std::error::Error for PlanResidueBundlesError {}

/// Build the §4.3.3 + §4.3.4 residue-bundle plan for one audio packet.
///
/// `no_residue` is the per-channel §4.3.2 step-6 flag *before* §4.3.3
/// propagation — i.e. exactly what the floor decode produced: element
/// `i` is `true` when channel `i`'s floor decoded to 'unused'. Its
/// length fixes the packet's channel count.
///
/// The function:
///
/// 1. Applies §4.3.3 nonzero-vector propagation over `mapping.coupling`
///    (the identical rule [`crate::packet::nonzero_propagate`] runs on
///    decode: if either partner of a coupling step is used, both become
///    used),
///    producing [`ResidueBundlePlan::no_residue`].
/// 2. For each submap `i` in order (§4.3.4 outer loop), gathers the
///    channels with `submap_for_channel == i` in ascending channel
///    order and copies their propagated `no_residue` flags into the
///    bundle's `do_not_decode` (§4.3.4 step 2). The single-submap case
///    (`submaps == 1`) routes every channel to submap 0 regardless of
///    `mux`, matching the decoder's implicit-zero path.
///
/// The result lets a wrapping §4.3 audio-packet writer thread one
/// [`write_residue_body`] per bundle in submap order, then recover the
/// §4.3.4 step-7 channel scatter from each
/// [`SubmapResidueBundle::channels`].
///
/// # Errors
///
/// * [`PlanResidueBundlesError::ZeroSubmaps`] — `mapping.submaps == 0`.
/// * [`PlanResidueBundlesError::CouplingChannelOutOfRange`] — a §4.3.3
///   coupling step named a channel outside `0 .. no_residue.len()`.
/// * [`PlanResidueBundlesError::SubmapOutOfRange`] /
///   [`PlanResidueBundlesError::MuxTooShort`] — a multi-submap
///   mapping's `mux` entry is missing or selects a submap `>= submaps`.
///
/// On any error nothing is returned; the function is pure (no side
/// effects on its inputs — `no_residue` is taken by value and the
/// propagated copy is returned inside the plan).
pub fn plan_residue_bundles(
    mapping: &MappingHeader,
    no_residue: &[bool],
) -> Result<ResidueBundlePlan, PlanResidueBundlesError> {
    let submaps = mapping.submaps as usize;
    if submaps == 0 {
        return Err(PlanResidueBundlesError::ZeroSubmaps);
    }
    let channels = no_residue.len();

    // §4.3.3 nonzero-vector propagate. Reuse the decoder's exact rule
    // (over a local copy so the caller's slice is untouched) rather
    // than re-deriving it; this keeps the encode/decode bit-for-bit
    // agreement on which channels are coded.
    let mut propagated = no_residue.to_vec();
    crate::packet::nonzero_propagate(&mut propagated, &mapping.coupling).map_err(|e| match e {
        crate::packet::PacketError::ChannelOutOfRange {
            step,
            channel,
            channels,
        } => PlanResidueBundlesError::CouplingChannelOutOfRange {
            step,
            channel,
            channels,
        },
        // nonzero_propagate only emits ChannelOutOfRange; any other
        // variant would be a contract break in that function.
        _ => unreachable!("nonzero_propagate emits only ChannelOutOfRange"),
    })?;

    // §4.3.4 step 2 / step 7: bundle the channels per submap in
    // ascending channel order. `submap_for(j)` mirrors the decoder's
    // `submap_for_channel`: implicit zero when `submaps == 1`, else
    // `mux[j]` with a bounds gate.
    let submap_for = |channel: usize| -> Result<usize, PlanResidueBundlesError> {
        if submaps <= 1 {
            return Ok(0);
        }
        let raw = *mapping.mux.get(channel).ok_or({
            PlanResidueBundlesError::MuxTooShort {
                channel,
                mux_len: mapping.mux.len(),
            }
        })? as usize;
        if raw >= submaps {
            return Err(PlanResidueBundlesError::SubmapOutOfRange {
                channel,
                submap: raw,
                submaps,
            });
        }
        Ok(raw)
    };

    // Resolve every channel's submap once (fail-closed before any
    // bundle is built), so the plan is all-or-nothing.
    let mut channel_submap = Vec::with_capacity(channels);
    for channel in 0..channels {
        channel_submap.push(submap_for(channel)?);
    }

    let mut bundles: Vec<SubmapResidueBundle> = (0..submaps)
        .map(|submap| SubmapResidueBundle {
            submap,
            channels: Vec::new(),
            do_not_decode: Vec::new(),
        })
        .collect();
    for (channel, &submap) in channel_submap.iter().enumerate() {
        let bundle = &mut bundles[submap];
        bundle.channels.push(channel);
        bundle.do_not_decode.push(propagated[channel]);
    }

    Ok(ResidueBundlePlan {
        no_residue: propagated,
        bundles,
    })
}

/// A channel's floor header + supplied packet body, paired and
/// type-checked, ready for emission. Internal to the §4.3 driver.
enum ResolvedFloor<'a> {
    Type0(&'a Floor0Header, &'a Floor0Packet),
    Type1(&'a Floor1Header, &'a Floor1Packet),
}

/// One channel's floor audio-packet body for the wrapping §4.3 audio-
/// packet writer ([`write_audio_packet`]).
///
/// §4.3.2 decodes one floor curve per channel in channel order, the
/// channel's floor selected by its submap's `floor` index. The variant
/// MUST match the [`FloorKind`] of that resolved floor header
/// ([`FloorKind::Type0`] ↔ [`AudioChannelFloor::Type0`],
/// [`FloorKind::Type1`] ↔ [`AudioChannelFloor::Type1`]); a mismatch is
/// rejected with [`WriteAudioPacketError::FloorTypeMismatch`] before any
/// bits are emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioChannelFloor {
    /// A §6.2.2 floor 0 packet body, written by
    /// [`write_floor0_packet_into_writer`].
    Type0(Floor0Packet),
    /// A §7.2.3 floor 1 packet body, written by
    /// [`write_floor1_packet_into_writer`].
    Type1(Floor1Packet),
}

impl AudioChannelFloor {
    /// `true` when this channel's floor body round-trips to `'unused'`
    /// (§4.3.2 step 6 sets the channel's `no_residue` flag). A floor 0
    /// `Unused` packet and a floor 1 packet with `nonzero == false` both
    /// decode to an unused curve.
    fn is_unused(&self) -> bool {
        match self {
            AudioChannelFloor::Type0(p) => matches!(p, Floor0Packet::Unused),
            AudioChannelFloor::Type1(p) => !p.nonzero,
        }
    }
}

/// Errors that may arise while writing a full §4.3 audio packet via
/// [`write_audio_packet`].
///
/// The driver fails closed: it validates the prelude, every channel's
/// floor body, the residue-bundle plan, and every submap's residue body
/// before emitting a single bit, so a caller's `writer` is bit-exactly
/// untouched on every error path.
#[derive(Debug)]
pub enum WriteAudioPacketError {
    /// The §4.3.1 prelude failed a [`write_audio_packet_header`]
    /// invariant.
    Header(WriteAudioPacketHeaderError),
    /// `floors.len()` did not match the channel count the caller
    /// declared via `audio_channels`. §4.3.2 decodes exactly one floor
    /// per channel.
    FloorCountMismatch {
        /// The declared `audio_channels`.
        audio_channels: usize,
        /// The number of supplied per-channel floor bodies.
        floors: usize,
    },
    /// `audio_channels` was zero. A Vorbis I stream carries at least one
    /// channel (§4.2.2 rejects a zero channel count); a zero-channel
    /// audio packet has no floor or residue body to write.
    ZeroAudioChannels,
    /// The §4.3.1 mode selected a mapping index outside
    /// `setup.mappings`. Mirrors the decoder's
    /// [`AudioPacketError::BadModeMapping`](crate::audio::AudioPacketError).
    BadModeMapping {
        /// The selected `[mode_number]`.
        mode_number: u32,
        /// The mode's `mapping` index.
        mapping: u8,
        /// The number of mappings in the setup header.
        mapping_count: usize,
    },
    /// A submap's `floor` index was outside `setup.floors`.
    SubmapFloorOutOfRange {
        /// The submap whose `floor` index was out of range.
        submap: usize,
        /// The out-of-range floor index.
        floor: u8,
        /// The number of floors in the setup header.
        floor_count: usize,
    },
    /// A submap's `residue` index was outside `setup.residues`.
    SubmapResidueOutOfRange {
        /// The submap whose `residue` index was out of range.
        submap: usize,
        /// The out-of-range residue index.
        residue: u8,
        /// The number of residues in the setup header.
        residue_count: usize,
    },
    /// A channel's [`AudioChannelFloor`] variant did not match the
    /// [`FloorKind`] of the floor header its submap selected.
    FloorTypeMismatch {
        /// The channel whose floor body had the wrong type.
        channel: usize,
        /// The floor header's type (`0` or `1`).
        header_type: u8,
        /// The supplied packet's type (`0` or `1`).
        packet_type: u8,
    },
    /// [`plan_residue_bundles`] rejected the mapping + `no_residue`
    /// vector (carried verbatim).
    Plan(PlanResidueBundlesError),
    /// `residue_plans.len()` did not match `mapping.submaps`. §4.3.4
    /// runs one residue body per submap, in submap order.
    ResiduePlanCountMismatch {
        /// The mapping's declared submap count.
        submaps: usize,
        /// The number of supplied per-submap residue-plan lists.
        plans: usize,
    },
    /// Writing a channel's floor 0 body failed (carried verbatim).
    Floor0 {
        /// The channel whose floor 0 body failed.
        channel: usize,
        /// The underlying floor 0 packet-writer error.
        source: WriteFloor0PacketError,
    },
    /// Writing a channel's floor 1 body failed (carried verbatim).
    Floor1 {
        /// The channel whose floor 1 body failed.
        channel: usize,
        /// The underlying floor 1 packet-writer error.
        source: WriteFloor1PacketError,
    },
    /// Writing a submap's §8.6.2 residue body failed (carried verbatim).
    Residue {
        /// The submap whose residue body failed.
        submap: usize,
        /// The underlying residue-body-writer error.
        source: WriteResidueBodyError,
    },
}

impl fmt::Display for WriteAudioPacketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteAudioPacketError::Header(e) => {
                write!(f, "vorbis §4.3 audio packet: prelude: {e}")
            }
            WriteAudioPacketError::FloorCountMismatch {
                audio_channels,
                floors,
            } => write!(
                f,
                "vorbis §4.3.2 audio packet: floor body count {floors} != audio_channels \
                 {audio_channels}"
            ),
            WriteAudioPacketError::ZeroAudioChannels => {
                write!(f, "vorbis §4.3 audio packet: audio_channels is zero")
            }
            WriteAudioPacketError::BadModeMapping {
                mode_number,
                mapping,
                mapping_count,
            } => write!(
                f,
                "vorbis §4.3.1 audio packet: mode_number {mode_number} maps to mapping \
                 {mapping} but only {mapping_count} mappings exist"
            ),
            WriteAudioPacketError::SubmapFloorOutOfRange {
                submap,
                floor,
                floor_count,
            } => write!(
                f,
                "vorbis §4.3.2 audio packet: submap {submap} floor index {floor} >= \
                 {floor_count} floors"
            ),
            WriteAudioPacketError::SubmapResidueOutOfRange {
                submap,
                residue,
                residue_count,
            } => write!(
                f,
                "vorbis §4.3.4 audio packet: submap {submap} residue index {residue} >= \
                 {residue_count} residues"
            ),
            WriteAudioPacketError::FloorTypeMismatch {
                channel,
                header_type,
                packet_type,
            } => write!(
                f,
                "vorbis §4.3.2 audio packet: channel {channel} floor packet type \
                 {packet_type} != header floor type {header_type}"
            ),
            WriteAudioPacketError::Plan(e) => {
                write!(f, "vorbis §4.3.3/§4.3.4 audio packet: {e}")
            }
            WriteAudioPacketError::ResiduePlanCountMismatch { submaps, plans } => write!(
                f,
                "vorbis §4.3.4 audio packet: residue plan list count {plans} != submaps \
                 {submaps}"
            ),
            WriteAudioPacketError::Floor0 { channel, source } => {
                write!(
                    f,
                    "vorbis §4.3.2 audio packet: channel {channel} floor 0: {source}"
                )
            }
            WriteAudioPacketError::Floor1 { channel, source } => {
                write!(
                    f,
                    "vorbis §4.3.2 audio packet: channel {channel} floor 1: {source}"
                )
            }
            WriteAudioPacketError::Residue { submap, source } => {
                write!(
                    f,
                    "vorbis §4.3.4 audio packet: submap {submap} residue: {source}"
                )
            }
        }
    }
}

impl std::error::Error for WriteAudioPacketError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WriteAudioPacketError::Header(e) => Some(e),
            WriteAudioPacketError::Plan(e) => Some(e),
            WriteAudioPacketError::Floor0 { source, .. } => Some(source),
            WriteAudioPacketError::Floor1 { source, .. } => Some(source),
            WriteAudioPacketError::Residue { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Serialise one full §4.3 audio packet — the wrapping audio-packet
/// WRITE driver that splices the §4.3.1 prelude, the §4.3.2 per-channel
/// floor bodies (channel order), and the §4.3.4 per-submap residue
/// bodies (submap order) into one byte-aligned packet.
///
/// This is the composition layer over the four `_into_writer` splice
/// primitives ([`write_audio_packet_header_into_writer`],
/// [`write_floor0_packet_into_writer`],
/// [`write_floor1_packet_into_writer`],
/// [`write_residue_body_into_writer`]) and the §4.3.3/§4.3.4
/// inverse-mapping layer ([`plan_residue_bundles`]). The emission order
/// is the exact inverse of [`crate::audio::decode_audio_packet_pre_imdct`]'s
/// read order:
///
/// 1. **§4.3.1 prelude** — `packet_type`, `mode_number`, and (long
///    blocks only) the two window flags.
/// 2. **§4.3.2 floors** — one floor body per channel, in channel order.
///    Each channel's floor type/header is resolved through its submap's
///    `floor` index (`mux[ch]` when the mapping has `submaps > 1`, else
///    submap 0).
/// 3. **§4.3.4 residues** — one residue body per submap, in submap
///    order. The per-submap `do_not_decode` flags come from
///    [`plan_residue_bundles`] (which applies §4.3.3 nonzero-vector
///    propagation over the mapping's coupling steps first), so a channel
///    pulled back in by a coupling partner is coded even though its own
///    floor was `'unused'`.
///
/// `floors` carries one [`AudioChannelFloor`] per channel in channel
/// order; its variant must match each channel's resolved floor header
/// type. `residue_plans` carries one `Vec<ResidueVectorPlan>` per submap
/// in submap order — exactly the `plans` slice
/// [`write_residue_body`] consumes for that submap, with the
/// `do_not_decode` flags supplied by the bundle plan.
///
/// The round-trip property: feeding the returned bytes to
/// [`crate::audio::decode_audio_packet_pre_imdct`] built from the same
/// setup reproduces the floor curves and residue vectors implied by the
/// inputs.
///
/// # Errors
///
/// Returns a [`WriteAudioPacketError`] for a prelude failure, a channel
/// count or floor-type mismatch, an out-of-range mapping/floor/residue
/// index, a bundle-plan rejection, a residue-plan count mismatch, or any
/// per-channel floor / per-submap residue body failure (each carried
/// verbatim). Validation precedes emission in full; on error nothing is
/// written.
#[allow(clippy::too_many_arguments)]
pub fn write_audio_packet(
    header: &AudioPacketHeader,
    setup: &VorbisSetupHeader,
    blocksize_0: usize,
    blocksize_1: usize,
    audio_channels: u8,
    floors: &[AudioChannelFloor],
    residue_plans: &[Vec<ResidueVectorPlan>],
) -> Result<Vec<u8>, WriteAudioPacketError> {
    let mut writer = BitWriterLsb::with_capacity(64);
    write_audio_packet_into_writer(
        header,
        setup,
        blocksize_0,
        blocksize_1,
        audio_channels,
        floors,
        residue_plans,
        &mut writer,
    )?;
    Ok(writer.finish())
}

/// Bit-level helper to splice a full §4.3 audio packet into a larger
/// bit-packed stream. Writes the packet's bits into `writer` at its
/// current bit position; on error no bits are emitted (validation
/// precedes emission in full).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_audio_packet_into_writer(
    header: &AudioPacketHeader,
    setup: &VorbisSetupHeader,
    blocksize_0: usize,
    blocksize_1: usize,
    audio_channels: u8,
    floors: &[AudioChannelFloor],
    residue_plans: &[Vec<ResidueVectorPlan>],
    writer: &mut BitWriterLsb,
) -> Result<(), WriteAudioPacketError> {
    // ---- fail-closed validation (no bits emitted on any error). ----
    let channels = audio_channels as usize;
    if channels == 0 {
        return Err(WriteAudioPacketError::ZeroAudioChannels);
    }
    if floors.len() != channels {
        return Err(WriteAudioPacketError::FloorCountMismatch {
            audio_channels: channels,
            floors: floors.len(),
        });
    }

    // §4.3.1 prelude validation is delegated to the prelude writer's
    // gate (mode number / blockflag / blocksize / window-flag checks).
    // We validate it here (into a throwaway writer) so the whole packet
    // is checked before any bits land in the caller's writer, then emit
    // it for real once everything passes.

    // §4.3.2 — resolve the mode's mapping.
    let mode = setup.modes[header.mode_number as usize];
    let mapping_index = mode.mapping as usize;
    let mapping =
        setup
            .mappings
            .get(mapping_index)
            .ok_or(WriteAudioPacketError::BadModeMapping {
                mode_number: header.mode_number,
                mapping: mode.mapping,
                mapping_count: setup.mappings.len(),
            })?;

    // §4.3.2 step 6: each channel's floor body sets its `no_residue`
    // flag. Build the raw (pre-§4.3.3) flags + the residue-bundle plan,
    // which range-checks the mux/coupling and bundles channels per
    // submap (§4.3.3 propagate + §4.3.4 gather).
    let no_residue: Vec<bool> = floors.iter().map(AudioChannelFloor::is_unused).collect();
    let plan = plan_residue_bundles(mapping, &no_residue).map_err(WriteAudioPacketError::Plan)?;

    if residue_plans.len() != mapping.submaps as usize {
        return Err(WriteAudioPacketError::ResiduePlanCountMismatch {
            submaps: mapping.submaps as usize,
            plans: residue_plans.len(),
        });
    }

    // Invert the bundle plan to a per-channel submap lookup so floor
    // bodies emit in channel order (§4.3.2) while residue bodies emit in
    // submap order (§4.3.4). The plan resolved every channel exactly
    // once, so this covers `0..channels`.
    let mut channel_submap = vec![0usize; channels];
    for bundle in &plan.bundles {
        for &ch in &bundle.channels {
            channel_submap[ch] = bundle.submap;
        }
    }

    // Pre-resolve each channel's floor header (via its submap) and
    // cross-check the supplied floor-body variant matches the header's
    // type, before any emission.
    let mut resolved_floors: Vec<ResolvedFloor<'_>> = Vec::with_capacity(channels);
    for (ch, chan_floor) in floors.iter().enumerate() {
        let submap = channel_submap[ch];
        // The bundle plan guarantees `submap < mapping.submaps`; the
        // setup parser pins `submap_configs.len() == submaps`, so this
        // index is in range, but resolve defensively.
        let submap_config = mapping.submap_configs.get(submap).ok_or(
            WriteAudioPacketError::SubmapFloorOutOfRange {
                submap,
                floor: 0,
                floor_count: setup.floors.len(),
            },
        )?;
        let floor_idx = submap_config.floor as usize;
        let floor_header =
            setup
                .floors
                .get(floor_idx)
                .ok_or(WriteAudioPacketError::SubmapFloorOutOfRange {
                    submap,
                    floor: submap_config.floor,
                    floor_count: setup.floors.len(),
                })?;
        let resolved = match (&floor_header.kind, chan_floor) {
            (FloorKind::Type0(h), AudioChannelFloor::Type0(p)) => ResolvedFloor::Type0(h, p),
            (FloorKind::Type1(h), AudioChannelFloor::Type1(p)) => ResolvedFloor::Type1(h, p),
            (FloorKind::Type0(_), AudioChannelFloor::Type1(_)) => {
                return Err(WriteAudioPacketError::FloorTypeMismatch {
                    channel: ch,
                    header_type: 0,
                    packet_type: 1,
                });
            }
            (FloorKind::Type1(_), AudioChannelFloor::Type0(_)) => {
                return Err(WriteAudioPacketError::FloorTypeMismatch {
                    channel: ch,
                    header_type: 1,
                    packet_type: 0,
                });
            }
        };
        resolved_floors.push(resolved);
    }

    // Pre-resolve each submap's residue header (via the bundle plan, in
    // submap order) before emission.
    let mut resolved_residues: Vec<(
        &ResidueHeader,
        &SubmapResidueBundle,
        &Vec<ResidueVectorPlan>,
    )> = Vec::with_capacity(plan.bundles.len());
    for bundle in &plan.bundles {
        let submap = bundle.submap;
        let submap_config = &mapping.submap_configs[submap];
        let residue_idx = submap_config.residue as usize;
        let residue_header = setup.residues.get(residue_idx).ok_or(
            WriteAudioPacketError::SubmapResidueOutOfRange {
                submap,
                residue: submap_config.residue,
                residue_count: setup.residues.len(),
            },
        )?;
        // residue_plans is index-aligned with submaps (count checked
        // above); bundle.submap is in `0..submaps`.
        resolved_residues.push((residue_header, bundle, &residue_plans[submap]));
    }

    // Probe pass: emit the whole packet into a throwaway writer so the
    // floor- and residue-body writers' own fail-closed gates run BEFORE
    // a single bit lands in the caller's writer. A body gate failing
    // here leaves only the probe writer dirty; the caller's writer is
    // untouched. The emit pass below is then guaranteed to succeed (the
    // body writers are pure functions of the same inputs).
    {
        let mut probe = BitWriterLsb::with_capacity(64);
        emit_audio_packet_bodies(
            header,
            setup,
            blocksize_0,
            blocksize_1,
            &resolved_floors,
            &resolved_residues,
            &mut probe,
        )?;
    }

    // ---- emit (validation complete). ----
    emit_audio_packet_bodies(
        header,
        setup,
        blocksize_0,
        blocksize_1,
        &resolved_floors,
        &resolved_residues,
        writer,
    )
}

/// Emit the §4.3.1 prelude + §4.3.2 floor bodies + §4.3.4 residue bodies
/// into `writer`, in spec order. Factored out of
/// [`write_audio_packet_into_writer`] so the same emission runs once
/// against a probe writer (gate check) and once against the real writer.
#[allow(clippy::too_many_arguments)]
fn emit_audio_packet_bodies(
    header: &AudioPacketHeader,
    setup: &VorbisSetupHeader,
    blocksize_0: usize,
    blocksize_1: usize,
    resolved_floors: &[ResolvedFloor<'_>],
    resolved_residues: &[(
        &ResidueHeader,
        &SubmapResidueBundle,
        &Vec<ResidueVectorPlan>,
    )],
    writer: &mut BitWriterLsb,
) -> Result<(), WriteAudioPacketError> {
    write_audio_packet_header_into_writer(header, setup, blocksize_0, blocksize_1, writer)
        .map_err(WriteAudioPacketError::Header)?;

    // §4.3.2 — floor bodies in channel order.
    for (ch, resolved) in resolved_floors.iter().enumerate() {
        match resolved {
            ResolvedFloor::Type0(h, p) => {
                write_floor0_packet_into_writer(p, h, &setup.codebooks, writer).map_err(
                    |source| WriteAudioPacketError::Floor0 {
                        channel: ch,
                        source,
                    },
                )?;
            }
            ResolvedFloor::Type1(h, p) => {
                write_floor1_packet_into_writer(p, h, &setup.codebooks, writer).map_err(
                    |source| WriteAudioPacketError::Floor1 {
                        channel: ch,
                        source,
                    },
                )?;
            }
        }
    }

    // §4.3.4 — residue bodies in submap order.
    for (residue_header, bundle, plans) in resolved_residues {
        write_residue_body_into_writer(
            plans,
            residue_header,
            &setup.codebooks,
            header.n,
            &bundle.do_not_decode,
            writer,
        )
        .map_err(|source| WriteAudioPacketError::Residue {
            submap: bundle.submap,
            source,
        })?;
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

    use crate::setup::{parse_setup_header, MappingSubmap, ParseError as SetupParseError};

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

    // ----------------------------------------------------------------
    // Floor 1 audio-packet body (§7.2.3) — round 250.
    // ----------------------------------------------------------------

    use crate::floor1::{Floor1Decoder, FloorCurve};

    /// Minimal scalar codebook with `entries` length-1 entries (so a
    /// 2-entry book assigns entry 0 → '0', entry 1 → '1'). Mirrors the
    /// helper in `floor1` tests so the §7.2.3 packet shape exercised
    /// here matches what `Floor1Decoder::decode` reads.
    fn fp_scalar_book(entries: u32) -> VorbisCodebook {
        VorbisCodebook {
            dimensions: 1,
            entries,
            codeword_lengths: vec![1u8; entries as usize],
            lookup: VqLookup::None,
        }
    }

    /// One-partition fixture matching the `floor1::tests::header_one_partition`
    /// configuration: 1 partition of class 0; class 0 dim 2, subclasses 0,
    /// one subclass book (index 0). multiplier 2, rangebits 4.
    fn fp_header_one_partition() -> crate::setup::Floor1Header {
        crate::setup::Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 2,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![Some(0)],
            }],
            multiplier: 2,
            rangebits: 4,
            x_list: vec![4, 8],
        }
    }

    /// `[nonzero] = 0` packet is exactly one zero bit; decoder returns
    /// `Unused` regardless of x_list / floor1_values.
    #[test]
    fn floor1_packet_unused_is_single_zero_bit() {
        let header = fp_header_one_partition();
        let codebooks = [fp_scalar_book(2)];
        let packet = Floor1Packet {
            nonzero: false,
            floor1_y: vec![], // ignored when nonzero == false
            partition_cvals: vec![],
        };
        let bytes = write_floor1_packet(&packet, &header, &codebooks).expect("must encode");
        // Single zero bit packs to one byte of value 0.
        assert_eq!(bytes, vec![0u8]);

        // Decoder roundtrip: FloorCurve::Unused.
        let dec = Floor1Decoder::new(&header, &codebooks).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(dec.decode(&mut r, 16), FloorCurve::Unused);
    }

    /// Roundtrip a `[nonzero] = 1` packet through the §7.2.3 decoder
    /// and confirm the recovered curve matches the floor1_packet_full_curve_round_trip
    /// fixture from the floor1 tests.
    #[test]
    fn floor1_packet_full_body_round_trips_against_decoder() {
        let header = fp_header_one_partition();
        let codebooks = [fp_scalar_book(2)];

        // Y values from the floor1 hand-trace test: endpoints 40, 20;
        // interior values 1, 0 (entry 1 then entry 0 of the 2-entry book).
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 1, 0],
            partition_cvals: vec![0], // unused (class.subclasses == 0)
        };
        let bytes = write_floor1_packet(&packet, &header, &codebooks).expect("must encode");

        // Decode and check curve == the hand-trace expected.
        let dec = Floor1Decoder::new(&header, &codebooks).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let expected_int: [usize; 16] = [
            80, 77, 74, 71, 68, 66, 64, 61, 59, 57, 54, 52, 50, 47, 45, 43,
        ];
        let expected: Vec<f32> = expected_int
            .iter()
            .map(|&i| crate::floor1::INVERSE_DB_TABLE[i])
            .collect();
        match dec.decode(&mut r, 16) {
            FloorCurve::Curve(c) => assert_eq!(c, expected),
            FloorCurve::Unused => panic!("expected a curve, got Unused"),
        }
    }

    /// Byte-shape pinning: the one-partition non-zero packet is a fixed
    /// bit pattern.
    ///   [nonzero=1] + [ep0=40,7b] + [ep1=20,7b] + [Y=1,1b] + [Y=0,1b]
    ///   = 1 + 7 + 7 + 1 + 1 = 17 bits → 3 bytes.
    /// LSB-first packing assembles those bits into:
    ///   bit 0 (LSb of byte 0): nonzero = 1
    ///   bits 1..=7 (rest of byte 0 + bit 0 of byte 1): ep0 = 40 LSb-first
    ///   ...
    /// We compute the expected bytes via BitWriterLsb directly so the
    /// test pins the bit ordering rather than guessing.
    #[test]
    fn floor1_packet_byte_shape_one_partition() {
        let header = fp_header_one_partition();
        let codebooks = [fp_scalar_book(2)];
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 1, 0],
            partition_cvals: vec![0],
        };
        let bytes = write_floor1_packet(&packet, &header, &codebooks).unwrap();

        // Build the same bit sequence with the raw bit writer.
        let mut expected_w = BitWriterLsb::with_capacity(3);
        expected_w.write_bit(true); // nonzero
        expected_w.write_u32(40, 7); // ep0 (ilog(127) = 7 bits)
        expected_w.write_u32(20, 7); // ep1
                                     // Class subclasses == 0: no master codeword. csub = 0,
                                     // cval = 0 → sub_idx = 0 for both dimensions, reads
                                     // 2-entry scalar book whose codewords are '0' and '1'.
        expected_w.write_bit(true); // Y[2] = 1 → entry 1 → codeword '1'
        expected_w.write_bit(false); // Y[3] = 0 → entry 0 → codeword '0'
        let expected = expected_w.finish();
        assert_eq!(bytes, expected);
    }

    /// Master/sub-cascade roundtrip: a class with subclasses > 0 reads
    /// a master selector then per-dim sub-book codewords. The packet
    /// writer must emit both, threading `cval` through `cval & csub`
    /// then `cval >>= cbits` on each dimension exactly like the
    /// decoder.
    #[test]
    fn floor1_packet_master_subclass_cascade_round_trips() {
        // Mirror of `floor1::tests::packet_decode_master_subclass_cascade`.
        let header = crate::setup::Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 2,
                subclasses: 1,
                masterbook: Some(0),
                subclass_books: vec![Some(1), Some(2)],
            }],
            multiplier: 2,
            rangebits: 4,
            x_list: vec![4, 8],
        };
        // 4-entry master book; lengths [2,2,2,2] yield codewords
        //   entry 0 = 00, entry 1 = 01, entry 2 = 10, entry 3 = 11.
        let master = VorbisCodebook {
            dimensions: 1,
            entries: 4,
            codeword_lengths: vec![2, 2, 2, 2],
            lookup: VqLookup::None,
        };
        let codebooks = vec![master, fp_scalar_book(2), fp_scalar_book(2)];

        // Pick cval = 2 (binary 10):
        //   dim 0: sub_idx = cval & 1 = 0 → subbook A (index 1); cval >>= 1 → 1.
        //   dim 1: sub_idx = cval & 1 = 1 → subbook B (index 2); cval >>= 1 → 0.
        // So Y[2] is encoded via subbook A; Y[3] via subbook B.
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 1, 0],
            partition_cvals: vec![2],
        };
        let bytes = write_floor1_packet(&packet, &header, &codebooks).unwrap();

        let dec = Floor1Decoder::new(&header, &codebooks).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let expected_int: [usize; 16] = [
            80, 77, 74, 71, 68, 66, 64, 61, 59, 57, 54, 52, 50, 47, 45, 43,
        ];
        let expected: Vec<f32> = expected_int
            .iter()
            .map(|&i| crate::floor1::INVERSE_DB_TABLE[i])
            .collect();
        match dec.decode(&mut r, 16) {
            FloorCurve::Curve(c) => assert_eq!(c, expected),
            FloorCurve::Unused => panic!("expected a curve, got Unused"),
        }
    }

    /// A `None` sub-book (encoded `-1`) forces the Y value to 0 and
    /// emits zero bits for that dimension. The writer accepts Y = 0
    /// and rejects any non-zero Y in that slot.
    #[test]
    fn floor1_packet_none_subclass_book_accepts_zero_rejects_nonzero() {
        let header = crate::setup::Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 2,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![None], // forces Y = 0
            }],
            multiplier: 2,
            rangebits: 4,
            x_list: vec![4, 8],
        };
        let codebooks = [fp_scalar_book(2)];

        // Zero interior Ys: encodes cleanly; decoder yields a curve.
        let ok = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 0, 0],
            partition_cvals: vec![0],
        };
        let bytes = write_floor1_packet(&ok, &header, &codebooks).unwrap();
        let dec = Floor1Decoder::new(&header, &codebooks).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        match dec.decode(&mut r, 16) {
            FloorCurve::Curve(c) => assert_eq!(c.len(), 16),
            FloorCurve::Unused => panic!("expected a curve"),
        }

        // Non-zero in a `None` slot is rejected.
        let bad = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 7, 0],
            partition_cvals: vec![0],
        };
        match write_floor1_packet(&bad, &header, &codebooks).unwrap_err() {
            WriteFloor1PacketError::NoneBookNonzeroY {
                partition: 0,
                dimension: 0,
                y_value: 7,
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// `floor1_y.len()` must equal `floor1_values` (= x_list.len() + 2).
    #[test]
    fn floor1_packet_rejects_y_length_mismatch() {
        let header = fp_header_one_partition();
        let codebooks = [fp_scalar_book(2)];
        // expected 4 (2 endpoints + 2 from one partition of dim 2);
        // give 3 → reject.
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 1],
            partition_cvals: vec![0],
        };
        match write_floor1_packet(&packet, &header, &codebooks).unwrap_err() {
            WriteFloor1PacketError::YLengthMismatch {
                expected: 4,
                actual: 3,
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// `partition_cvals.len()` must equal `partitions`.
    #[test]
    fn floor1_packet_rejects_cval_list_length_mismatch() {
        let header = fp_header_one_partition();
        let codebooks = [fp_scalar_book(2)];
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 1, 0],
            partition_cvals: vec![0, 1], // 2 != partitions (1)
        };
        match write_floor1_packet(&packet, &header, &codebooks).unwrap_err() {
            WriteFloor1PacketError::CvalListLengthMismatch {
                expected: 1,
                actual: 2,
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// `multiplier` outside `1..=4` is rejected.
    #[test]
    fn floor1_packet_rejects_illegal_multiplier() {
        let mut header = fp_header_one_partition();
        header.multiplier = 0;
        let codebooks = [fp_scalar_book(2)];
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 1, 0],
            partition_cvals: vec![0],
        };
        match write_floor1_packet(&packet, &header, &codebooks).unwrap_err() {
            WriteFloor1PacketError::IllegalMultiplier(0) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// Endpoint amplitude `>= range` is rejected; multiplier 2 → range
    /// 128, so a value of 128 is the smallest illegal endpoint.
    #[test]
    fn floor1_packet_rejects_endpoint_overflow() {
        let header = fp_header_one_partition();
        let codebooks = [fp_scalar_book(2)];
        // ep0 = 128 = range for multiplier 2.
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![128, 20, 1, 0],
            partition_cvals: vec![0],
        };
        match write_floor1_packet(&packet, &header, &codebooks).unwrap_err() {
            WriteFloor1PacketError::EndpointOverflow {
                index: 0,
                value: 128,
                range: 128,
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// `partition_class_list[i]` pointing at a missing class entry
    /// is rejected.
    #[test]
    fn floor1_packet_rejects_bad_class_index() {
        let mut header = fp_header_one_partition();
        // Point at class 5 which does not exist.
        header.partition_class_list = vec![5];
        let codebooks = [fp_scalar_book(2)];
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 1, 0],
            partition_cvals: vec![0],
        };
        match write_floor1_packet(&packet, &header, &codebooks).unwrap_err() {
            WriteFloor1PacketError::BadClassIndex {
                partition: 0,
                class: 5,
                class_count: 1,
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// `masterbook` pointing past the codebook table is rejected.
    #[test]
    fn floor1_packet_rejects_masterbook_out_of_range() {
        let header = crate::setup::Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 2,
                subclasses: 1,
                masterbook: Some(9), // out of range
                subclass_books: vec![Some(0), Some(0)],
            }],
            multiplier: 2,
            rangebits: 4,
            x_list: vec![4, 8],
        };
        let codebooks = [fp_scalar_book(2)];
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 1, 0],
            partition_cvals: vec![0],
        };
        match write_floor1_packet(&packet, &header, &codebooks).unwrap_err() {
            WriteFloor1PacketError::MasterbookOutOfRange {
                class: 0,
                book: 9,
                codebook_count: 1,
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// A subclass book pointing past the codebook table is rejected.
    #[test]
    fn floor1_packet_rejects_subclass_book_out_of_range() {
        let header = crate::setup::Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 2,
                subclasses: 0,
                masterbook: None,
                subclass_books: vec![Some(9)], // out of range
            }],
            multiplier: 2,
            rangebits: 4,
            x_list: vec![4, 8],
        };
        let codebooks = [fp_scalar_book(2)];
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 1, 0],
            partition_cvals: vec![0],
        };
        match write_floor1_packet(&packet, &header, &codebooks).unwrap_err() {
            WriteFloor1PacketError::SubclassBookOutOfRange {
                class: 0,
                subclass: 0,
                book: 9,
                codebook_count: 1,
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// A Y value not present in the sub-book's used set is rejected
    /// with `UnencodableY`.
    #[test]
    fn floor1_packet_rejects_unencodable_y() {
        let header = fp_header_one_partition();
        // 2-entry book → only Y values 0 and 1 are encodable.
        let codebooks = [fp_scalar_book(2)];
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 7, 0], // Y=7 doesn't fit the book
            partition_cvals: vec![0],
        };
        match write_floor1_packet(&packet, &header, &codebooks).unwrap_err() {
            WriteFloor1PacketError::UnencodableY {
                partition: 0,
                dimension: 0,
                y_value: 7,
                book: 0,
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// A `cval` value larger than the master book's used set is
    /// rejected with `UnencodableCval`.
    #[test]
    fn floor1_packet_rejects_unencodable_cval() {
        let header = crate::setup::Floor1Header {
            partitions: 1,
            partition_class_list: vec![0],
            classes: vec![Floor1Class {
                dimensions: 2,
                subclasses: 1,
                masterbook: Some(0),
                subclass_books: vec![Some(1), Some(2)],
            }],
            multiplier: 2,
            rangebits: 4,
            x_list: vec![4, 8],
        };
        // Master book has only 4 entries (lengths [2,2,2,2]).
        let master = VorbisCodebook {
            dimensions: 1,
            entries: 4,
            codeword_lengths: vec![2, 2, 2, 2],
            lookup: VqLookup::None,
        };
        let codebooks = vec![master, fp_scalar_book(2), fp_scalar_book(2)];
        // cval = 9 is past the 4 entries → unencodable.
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 0, 0],
            partition_cvals: vec![9],
        };
        match write_floor1_packet(&packet, &header, &codebooks).unwrap_err() {
            WriteFloor1PacketError::UnencodableCval {
                partition: 0,
                cval: 9,
                book: 0,
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// Roundtrip across the four multiplier values (1..=4) on the
    /// one-partition fixture. Each multiplier yields a different
    /// `[range]`; the writer derives `amp_bits = ilog(range - 1)` and
    /// the decoder must read back the same endpoint values.
    #[test]
    fn floor1_packet_round_trips_across_all_multipliers() {
        let mut header = fp_header_one_partition();
        let codebooks = [fp_scalar_book(2)];
        for multiplier in 1u8..=4 {
            header.multiplier = multiplier;
            let range = RANGE_TABLE_FLOOR1[(multiplier - 1) as usize];
            // Use endpoint values inside `range`.
            let ep0 = range / 4;
            let ep1 = range / 8;
            let packet = Floor1Packet {
                nonzero: true,
                floor1_y: vec![ep0, ep1, 0, 0],
                partition_cvals: vec![0],
            };
            let bytes = write_floor1_packet(&packet, &header, &codebooks).unwrap();
            let dec = Floor1Decoder::new(&header, &codebooks).unwrap();
            let mut r = BitReaderLsb::new(&bytes);
            // For this fixture (interior Ys = 0) decode yields a
            // FloorCurve::Curve.
            match dec.decode(&mut r, 16) {
                FloorCurve::Curve(c) => assert_eq!(c.len(), 16),
                FloorCurve::Unused => {
                    panic!("multiplier {multiplier}: expected a curve, got Unused")
                }
            }
        }
    }

    /// The splice helper `write_floor1_packet_into_writer` produces the
    /// same bytes as the public wrapper when the writer starts byte-
    /// aligned at position 0.
    #[test]
    fn floor1_packet_splice_matches_public_writer() {
        let header = fp_header_one_partition();
        let codebooks = [fp_scalar_book(2)];
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![40, 20, 1, 0],
            partition_cvals: vec![0],
        };
        let public = write_floor1_packet(&packet, &header, &codebooks).unwrap();

        let mut w = BitWriterLsb::with_capacity(1);
        write_floor1_packet_into_writer(&packet, &header, &codebooks, &mut w).unwrap();
        let splice = w.finish();

        assert_eq!(splice, public);
    }

    /// The unused-path splice helper writes exactly one zero bit.
    /// When the writer is already byte-aligned, this packs to a single
    /// 0x00 byte.
    #[test]
    fn floor1_packet_unused_splice_writes_one_zero_bit() {
        let header = fp_header_one_partition();
        let codebooks = [fp_scalar_book(2)];
        let packet = Floor1Packet {
            nonzero: false,
            floor1_y: vec![],
            partition_cvals: vec![],
        };
        let mut w = BitWriterLsb::with_capacity(1);
        // Establish the bit position is 0 → byte boundary.
        let pos_before = w.bit_position();
        write_floor1_packet_into_writer(&packet, &header, &codebooks, &mut w).unwrap();
        let pos_after = w.bit_position();
        assert_eq!(pos_after - pos_before, 1, "unused emits exactly one bit");
        let bytes = w.finish();
        assert_eq!(bytes, vec![0u8]);
    }

    /// The umbrella `WriteError` From-glue forwards
    /// `WriteFloor1PacketError` into `WriteError::Floor1Packet` and
    /// preserves `source()` chaining for the wrapping error.
    #[test]
    fn floor1_packet_umbrella_write_error_glue() {
        let err: WriteError = WriteFloor1PacketError::IllegalMultiplier(5).into();
        match &err {
            WriteError::Floor1Packet(WriteFloor1PacketError::IllegalMultiplier(5)) => {}
            other => panic!("From glue wrong variant: {other:?}"),
        }
        let source = std::error::Error::source(&err);
        assert!(source.is_some(), "Floor1Packet must chain its source");

        let crate_err: crate::Error = err.into();
        match &crate_err {
            crate::Error::Write(WriteError::Floor1Packet(
                WriteFloor1PacketError::IllegalMultiplier(5),
            )) => {}
            other => panic!("crate::Error glue wrong variant: {other:?}"),
        }
    }

    /// `Display` of every `WriteFloor1PacketError` variant is non-empty
    /// and (where applicable) contains the offending number.
    #[test]
    fn floor1_packet_error_displays_are_informative() {
        let cases: Vec<WriteFloor1PacketError> = vec![
            WriteFloor1PacketError::YLengthMismatch {
                expected: 4,
                actual: 3,
            },
            WriteFloor1PacketError::CvalListLengthMismatch {
                expected: 1,
                actual: 2,
            },
            WriteFloor1PacketError::IllegalMultiplier(5),
            WriteFloor1PacketError::EndpointOverflow {
                index: 0,
                value: 128,
                range: 128,
            },
            WriteFloor1PacketError::BadClassIndex {
                partition: 0,
                class: 5,
                class_count: 1,
            },
            WriteFloor1PacketError::MasterbookOutOfRange {
                class: 0,
                book: 9,
                codebook_count: 1,
            },
            WriteFloor1PacketError::SubclassBookOutOfRange {
                class: 0,
                subclass: 0,
                book: 9,
                codebook_count: 1,
            },
            WriteFloor1PacketError::UnencodableY {
                partition: 0,
                dimension: 1,
                y_value: 42,
                book: 3,
            },
            WriteFloor1PacketError::NoneBookNonzeroY {
                partition: 0,
                dimension: 0,
                y_value: 7,
            },
            WriteFloor1PacketError::UnencodableCval {
                partition: 0,
                cval: 9,
                book: 0,
            },
        ];
        for e in &cases {
            let s = format!("{e}");
            assert!(
                !s.is_empty(),
                "Display for {e:?} must produce non-empty output"
            );
            assert!(
                s.contains("floor1") || s.contains("vorbis"),
                "Display for {e:?} should be grep-able: {s}"
            );
        }
    }

    // ---- §8.6.2 residue classification packing (round 35) ----

    /// Faithful re-statement of the §8.6.2 step-10..12 classbook
    /// *unpack* the residue decoder performs after reading one classbook
    /// entry `temp`. Recovers `group_len` classifications in ascending
    /// group order (position 0 = least-significant base-`C` digit).
    /// This is the exact loop `ResidueDecoder::decode_core` runs; the
    /// `pack_residue_classifications` round-trip is checked against it.
    fn unpack_residue_classifications(mut temp: u32, group_len: usize, base: u32) -> Vec<u32> {
        let mut out = vec![0u32; group_len];
        for i in (0..group_len).rev() {
            out[i] = temp % base;
            temp /= base;
        }
        out
    }

    #[test]
    fn pack_residue_classifications_single_digit_is_identity() {
        // A length-1 group packs to the digit itself for any base.
        for base in 1..=64u32 {
            for d in 0..base {
                assert_eq!(pack_residue_classifications(&[d], base), Ok(d));
            }
        }
    }

    #[test]
    fn pack_residue_classifications_positional_weights() {
        // Position 0 is the most-significant digit (matching the
        // decoder's descending unpack). base = 3, group [0, 1, 2]:
        // 0·3^2 + 1·3^1 + 2·3^0 = 3 + 2 = 5.
        assert_eq!(pack_residue_classifications(&[0, 1, 2], 3), Ok(5));
        // base = 10, group [7, 2, 4]: 700 + 20 + 4 = 724.
        assert_eq!(pack_residue_classifications(&[7, 2, 4], 10), Ok(724));
        // base = 2, group [1, 1, 0, 1]: 8 + 4 + 0 + 1 = 13.
        assert_eq!(pack_residue_classifications(&[1, 1, 0, 1], 2), Ok(13));
    }

    #[test]
    fn pack_residue_classifications_position_zero_is_most_significant() {
        // The last group position is the least-significant digit: bumping
        // it changes the result by exactly 1 (weight C^0).
        let base = 5;
        let a = pack_residue_classifications(&[2, 3, 0], base).unwrap();
        let b = pack_residue_classifications(&[2, 3, 1], base).unwrap();
        assert_eq!(b - a, 1);
        // Bumping position 0 changes it by C^(len-1) per increment.
        let c = pack_residue_classifications(&[4, 3, 0], base).unwrap();
        assert_eq!(c - a, (4 - 2) * base.pow(2));
    }

    #[test]
    fn pack_residue_classifications_round_trips_against_unpack() {
        // Exhaustive over small bases and group lengths: pack then run
        // the decoder's own unpack loop; the group must be recovered.
        for base in 1..=6u32 {
            for group_len in 1..=4usize {
                // Enumerate every legal group (each digit in 0..base).
                let total = base.pow(group_len as u32);
                for n in 0..total {
                    let group = unpack_residue_classifications(n, group_len, base);
                    let packed = pack_residue_classifications(&group, base).unwrap();
                    assert_eq!(packed, n, "pack must reproduce the classbook entry");
                    let recovered = unpack_residue_classifications(packed, group_len, base);
                    assert_eq!(recovered, group, "unpack(pack(group)) must equal group");
                }
            }
        }
    }

    #[test]
    fn pack_residue_classifications_round_trips_at_base_64() {
        // The §8.6.1 maximum base (residue_classifications = 64). A
        // 5-digit group at base 64 still fits a u32 (64^5 = 2^30).
        let base = 64;
        let group = [63, 0, 17, 63, 5];
        let packed = pack_residue_classifications(&group, base).unwrap();
        let recovered = unpack_residue_classifications(packed, group.len(), base);
        assert_eq!(recovered, group.to_vec());
    }

    #[test]
    fn pack_residue_classifications_rejects_zero_base() {
        assert_eq!(
            pack_residue_classifications(&[0], 0),
            Err(PackResidueClassError::ZeroClassifications)
        );
    }

    #[test]
    fn pack_residue_classifications_rejects_base_above_64() {
        assert_eq!(
            pack_residue_classifications(&[0], 65),
            Err(PackResidueClassError::ClassificationsTooLarge(65))
        );
    }

    #[test]
    fn pack_residue_classifications_rejects_empty_group() {
        assert_eq!(
            pack_residue_classifications(&[], 4),
            Err(PackResidueClassError::EmptyGroup)
        );
    }

    #[test]
    fn pack_residue_classifications_rejects_group_longer_than_32() {
        let group = vec![0u32; 33];
        assert_eq!(
            pack_residue_classifications(&group, 2),
            Err(PackResidueClassError::GroupTooLong(33))
        );
        // A length-32 group is accepted (the boundary).
        let group32 = vec![0u32; 32];
        assert!(pack_residue_classifications(&group32, 2).is_ok());
    }

    #[test]
    fn pack_residue_classifications_rejects_out_of_range_digit() {
        // Digit at position 1 equals the base → out of range.
        assert_eq!(
            pack_residue_classifications(&[2, 4, 1], 4),
            Err(PackResidueClassError::ClassificationOutOfRange {
                position: 1,
                classification: 4,
                num_classifications: 4,
            })
        );
    }

    #[test]
    fn pack_residue_classifications_rejects_packed_overflow() {
        // base 64, 33 digits would overflow — but length is gated first,
        // so build a case that passes the length gate yet overflows: base
        // 64, 8 digits all = 63. 64^8 = 2^48 >> u32::MAX, so even the
        // top digit's weight overflows.
        let group = vec![63u32; 8];
        assert_eq!(
            pack_residue_classifications(&group, 64),
            Err(PackResidueClassError::PackedValueOverflow {
                num_classifications: 64,
                group_len: 8,
            })
        );
    }

    #[test]
    fn pack_residue_classifications_validation_precedes_overflow() {
        // An out-of-range digit is reported even when the group would
        // also overflow — validation runs before the packing arithmetic.
        let mut group = vec![63u32; 8];
        group[2] = 64; // out of range for base 64
        assert_eq!(
            pack_residue_classifications(&group, 64),
            Err(PackResidueClassError::ClassificationOutOfRange {
                position: 2,
                classification: 64,
                num_classifications: 64,
            })
        );
    }

    #[test]
    fn pack_residue_classifications_umbrella_write_error_glue() {
        let err: WriteError = PackResidueClassError::ZeroClassifications.into();
        match &err {
            WriteError::ResidueClassification(PackResidueClassError::ZeroClassifications) => {}
            other => panic!("From glue wrong variant: {other:?}"),
        }
        assert!(
            std::error::Error::source(&err).is_some(),
            "ResidueClassification must chain its source"
        );
        let crate_err: crate::Error = err.into();
        match &crate_err {
            crate::Error::Write(WriteError::ResidueClassification(
                PackResidueClassError::ZeroClassifications,
            )) => {}
            other => panic!("crate::Error glue wrong variant: {other:?}"),
        }
    }

    #[test]
    fn pack_residue_class_error_displays_are_informative() {
        let cases: Vec<PackResidueClassError> = vec![
            PackResidueClassError::ZeroClassifications,
            PackResidueClassError::ClassificationsTooLarge(65),
            PackResidueClassError::EmptyGroup,
            PackResidueClassError::GroupTooLong(33),
            PackResidueClassError::ClassificationOutOfRange {
                position: 1,
                classification: 4,
                num_classifications: 4,
            },
            PackResidueClassError::PackedValueOverflow {
                num_classifications: 64,
                group_len: 8,
            },
        ];
        for e in &cases {
            let s = format!("{e}");
            assert!(!s.is_empty(), "Display for {e:?} must be non-empty");
            assert!(
                s.contains("vorbis") && s.contains("residue"),
                "Display for {e:?} should be grep-able: {s}"
            );
        }
    }

    // ---- §8.6.2 residue classification grouping (round 36) ----

    /// The structural inverse of the §8.6.2 step-6..9 decode walk: with
    /// `classwords == 1`, every partition is its own group, so each
    /// classbook entry is just that partition's classification index.
    #[test]
    fn pack_residue_class_groups_classwords_one_is_per_partition_identity() {
        let class = [0u32, 3, 1, 9, 2, 7];
        let base = 10;
        let entries = pack_residue_classification_groups(&class, base, 1).unwrap();
        assert_eq!(entries, class.to_vec());
    }

    /// An empty classification array yields no classbook entries (the
    /// decode loop runs zero groups when `partitions_to_read == 0`).
    #[test]
    fn pack_residue_class_groups_empty_yields_empty() {
        assert_eq!(
            pack_residue_classification_groups(&[], 8, 4),
            Ok(Vec::new())
        );
        // …even with a width that doesn't divide an (absent) length.
        assert_eq!(
            pack_residue_classification_groups(&[], 64, 7),
            Ok(Vec::new())
        );
    }

    /// A length that is an exact multiple of `classwords` splits into
    /// full groups, each packed by the per-group primitive. Hand-check
    /// the packed values against `pack_residue_classifications`.
    #[test]
    fn pack_residue_class_groups_exact_multiple_matches_per_group_pack() {
        let class = [1u32, 2, 3, 0, 1, 2];
        let base = 4;
        let classwords = 3;
        let entries = pack_residue_classification_groups(&class, base, classwords).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0],
            pack_residue_classifications(&[1, 2, 3], base).unwrap()
        );
        assert_eq!(
            entries[1],
            pack_residue_classifications(&[0, 1, 2], base).unwrap()
        );
    }

    /// A length that is NOT a multiple of `classwords` right-pads the
    /// final group with classification index 0 (the least-significant
    /// digits the decoder reads-and-discards). The final entry must equal
    /// the per-group pack of the kept prefix zero-padded to full width.
    #[test]
    fn pack_residue_class_groups_partial_final_group_pads_with_zero() {
        // 5 partitions, classwords = 3 => 2 groups; the second holds the
        // kept [4, 2] plus one zero pad => [4, 2, 0].
        let class = [4u32, 2, 5, 4, 2];
        let base = 6;
        let entries = pack_residue_classification_groups(&class, base, 3).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0],
            pack_residue_classifications(&[4, 2, 5], base).unwrap()
        );
        assert_eq!(
            entries[1],
            pack_residue_classifications(&[4, 2, 0], base).unwrap()
        );
    }

    /// End-to-end round-trip against the decoder's own step-10..12 unpack
    /// loop: group a full classification array, unpack each returned
    /// entry, concatenate (truncating the final group to the kept count),
    /// and confirm the original array is recovered. Exhaustive over a
    /// grid of bases, widths, and lengths (including non-multiples).
    #[test]
    fn pack_residue_class_groups_round_trips_against_unpack() {
        for base in 1..=5u32 {
            for classwords in 1..=4usize {
                for len in 0..=11usize {
                    // Build a deterministic classification array in 0..base.
                    let class: Vec<u32> = (0..len).map(|p| (p as u32 * 7 + 1) % base).collect();
                    let entries =
                        pack_residue_classification_groups(&class, base, classwords).unwrap();
                    // The decoder unpacks each entry into `classwords`
                    // classifications, keeping only partition indices
                    // `< len`. Reconstruct the kept array.
                    let mut recovered = Vec::with_capacity(len);
                    for &entry in &entries {
                        let group = unpack_residue_classifications(entry, classwords, base);
                        for d in group {
                            if recovered.len() < len {
                                recovered.push(d);
                            }
                        }
                    }
                    assert_eq!(
                        recovered, class,
                        "round-trip failed for base={base} classwords={classwords} len={len}"
                    );
                }
            }
        }
    }

    /// `classwords_per_codeword == 0` is refused (the §8.6.2 group width
    /// is the classbook dimensions, which is >= 1).
    #[test]
    fn pack_residue_class_groups_rejects_zero_classwords() {
        assert_eq!(
            pack_residue_classification_groups(&[0, 1, 2], 4, 0),
            Err(PackResidueClassGroupsError::ZeroClasswords)
        );
        // Even an empty array is refused on a zero width (the width is an
        // invariant of the residue config, not of the data length).
        assert_eq!(
            pack_residue_classification_groups(&[], 4, 0),
            Err(PackResidueClassGroupsError::ZeroClasswords)
        );
    }

    /// An out-of-range classification surfaces the per-group error tagged
    /// with the group index that contained it (here group 1).
    #[test]
    fn pack_residue_class_groups_tags_failing_group_index() {
        // base = 4 (legal indices 0..4); group 1 holds the bad index 9.
        let class = [0u32, 1, 2, 9, 1, 2];
        let err = pack_residue_classification_groups(&class, 4, 3).unwrap_err();
        match err {
            PackResidueClassGroupsError::Pack { group, source } => {
                assert_eq!(group, 1, "the second group (index 1) holds the bad digit");
                assert_eq!(
                    source,
                    PackResidueClassError::ClassificationOutOfRange {
                        position: 0,
                        classification: 9,
                        num_classifications: 4,
                    }
                );
            }
            other => panic!("expected Pack error, got {other:?}"),
        }
    }

    /// An oversized base propagates from the per-group packer on group 0.
    #[test]
    fn pack_residue_class_groups_propagates_base_error() {
        let err = pack_residue_classification_groups(&[0, 1], 65, 2).unwrap_err();
        assert_eq!(
            err,
            PackResidueClassGroupsError::Pack {
                group: 0,
                source: PackResidueClassError::ClassificationsTooLarge(65),
            }
        );
    }

    /// `Error::source()` chains through to the inner per-group error so
    /// callers can walk the cause chain; `ZeroClasswords` has no source.
    #[test]
    fn pack_residue_class_groups_error_source_chains() {
        use std::error::Error as _;
        let pack = PackResidueClassGroupsError::Pack {
            group: 2,
            source: PackResidueClassError::EmptyGroup,
        };
        assert!(
            pack.source().is_some(),
            "Pack must chain to the inner error"
        );
        assert!(
            PackResidueClassGroupsError::ZeroClasswords
                .source()
                .is_none(),
            "ZeroClasswords has no source"
        );
    }

    /// Both `Display` strings are non-empty and grep-friendly, and the
    /// `Pack` variant embeds the offending group index and the inner
    /// message.
    #[test]
    fn pack_residue_class_groups_error_displays_are_informative() {
        let zero = format!("{}", PackResidueClassGroupsError::ZeroClasswords);
        assert!(zero.contains("vorbis") && zero.contains("classwords_per_codeword"));

        let pack = format!(
            "{}",
            PackResidueClassGroupsError::Pack {
                group: 3,
                source: PackResidueClassError::ZeroClassifications,
            }
        );
        assert!(
            pack.contains("group 3") && pack.contains("vorbis"),
            "Pack Display should name the group and be grep-able: {pack}"
        );
    }

    // ===== §8.6.3/§8.6.4/§8.6.5 residue partition body WRITE (round 37) =====

    /// A scalar (lookup_type 0) codebook — used as a classbook in the
    /// roundtrip fixtures and as the §8.6.1 rejection input.
    fn rp_scalar_book(dimensions: u16, lengths: Vec<u8>) -> VorbisCodebook {
        VorbisCodebook {
            dimensions,
            entries: lengths.len() as u32,
            codeword_lengths: lengths,
            lookup: VqLookup::None,
        }
    }

    /// A tessellation (lookup_type 2) VQ value book with `delta = 1`,
    /// `min = 0`, no sequence flag — entry `e`'s vector is the
    /// multiplicand row `multiplicands[e*dims .. (e+1)*dims]` verbatim.
    fn rp_vq_book(dimensions: u16, lengths: Vec<u8>, multiplicands: Vec<u32>) -> VorbisCodebook {
        VorbisCodebook {
            dimensions,
            entries: lengths.len() as u32,
            codeword_lengths: lengths,
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands,
            },
        }
    }

    /// A residue header whose cascade bitmap is derived from `books`.
    fn rp_residue_header(
        residue_type: u16,
        begin: u32,
        end: u32,
        partition_size: u32,
        books: Vec<[Option<u8>; 8]>,
    ) -> crate::setup::ResidueHeader {
        let cascade = books
            .iter()
            .map(|row| {
                let mut bits = 0u8;
                for (j, slot) in row.iter().enumerate() {
                    if slot.is_some() {
                        bits |= 1 << j;
                    }
                }
                bits
            })
            .collect();
        crate::setup::ResidueHeader {
            residue_type,
            residue_begin: begin,
            residue_end: end,
            partition_size,
            classifications: books.len() as u8,
            classbook: 0,
            cascade,
            books,
        }
    }

    /// §8.6.3 step 1: format 0 reads exactly `n / dims` codewords.
    #[test]
    fn residue_partition_codeword_count_format0_exact_division() {
        assert_eq!(residue_partition_codeword_count(0, 8, 2), Ok(4));
        assert_eq!(residue_partition_codeword_count(0, 8, 8), Ok(1));
        assert_eq!(residue_partition_codeword_count(0, 6, 3), Ok(2));
    }

    /// §8.6.4 (and §8.6.5 via "reducible to format 1"): formats 1 and 2
    /// read `ceil(n / dims)` codewords — the read-while-`[i] < [n]`
    /// loop reads a final partial vector when dims does not divide n.
    #[test]
    fn residue_partition_codeword_count_format1_and_2_use_ceil() {
        assert_eq!(residue_partition_codeword_count(1, 4, 1), Ok(4));
        assert_eq!(residue_partition_codeword_count(1, 5, 2), Ok(3));
        assert_eq!(residue_partition_codeword_count(1, 4, 8), Ok(1));
        assert_eq!(residue_partition_codeword_count(2, 5, 2), Ok(3));
        assert_eq!(residue_partition_codeword_count(2, 4, 4), Ok(1));
    }

    /// The count helper carries the four structural rejections shared
    /// with the writer: bad format, zero partition size, zero
    /// dimensions, format-0 divisibility.
    #[test]
    fn residue_partition_codeword_count_rejects_invalid_inputs() {
        assert_eq!(
            residue_partition_codeword_count(3, 4, 2),
            Err(WriteResiduePartitionError::UnsupportedResidueType(3))
        );
        assert_eq!(
            residue_partition_codeword_count(1, 0, 2),
            Err(WriteResiduePartitionError::ZeroPartitionSize)
        );
        assert_eq!(
            residue_partition_codeword_count(1, 4, 0),
            Err(WriteResiduePartitionError::ZeroDimensions)
        );
        assert_eq!(
            residue_partition_codeword_count(0, 5, 2),
            Err(WriteResiduePartitionError::Format0NotDivisible {
                partition_size: 5,
                dimensions: 2,
            })
        );
    }

    /// Byte-shape pin: a 2-entry length-[1,1] book assigns canonical
    /// 1-bit codewords `0` / `1`; entries [0, 1, 1, 0] emit bits
    /// 0,1,1,0 which pack LSb-first (§2.1.4) into the single byte
    /// 0b0000_0110 = 0x06.
    #[test]
    fn write_residue_partition_format1_byte_shape() {
        let book = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let bytes = write_residue_partition(&[0, 1, 1, 0], &book, 1, 4).unwrap();
        assert_eq!(bytes, vec![0x06]);
    }

    /// End-to-end §8.6.4 roundtrip: classbook codeword + partition body
    /// composed by hand (the wrapping §8.6.2 writer is a followup),
    /// then read back through the full residue decoder. Mirrors the
    /// decoder-side `format1_single_channel_two_partitions` fixture:
    /// two partitions of two scalars each over a 1-D value book whose
    /// entries unpack to [3] and [5].
    #[test]
    fn write_residue_partition_roundtrips_against_decoder_format1() {
        use oxideav_core::bits::BitReaderLsb;

        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);

        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 4, 2, vec![row]);
        let codebooks = vec![classbook.clone(), valbook.clone()];
        let dec = crate::residue::ResidueDecoder::new(&header, &codebooks).unwrap();

        // classifications: both partitions are class 0 (the only class);
        // classwords_per_codeword = classbook dims = 1, so each partition
        // gets its own classbook entry.
        let cls_entries = pack_residue_classification_groups(&[0, 0], 1, 1).unwrap();
        assert_eq!(cls_entries, vec![0, 0]);
        let cls_tree = crate::huffman::HuffmanTree::from_codebook(&classbook).unwrap();

        // §8.6.2 pass-0 stream order with classwords = 1:
        //   classbook codeword (partition 0) → partition 0 body
        //   classbook codeword (partition 1) → partition 1 body
        let mut w = BitWriterLsb::new();
        cls_tree.encode_entry(cls_entries[0], &mut w).unwrap();
        write_residue_partition_into_writer(&[0, 1], &valbook, 1, 2, &mut w).unwrap();
        cls_tree.encode_entry(cls_entries[1], &mut w).unwrap();
        write_residue_partition_into_writer(&[1, 0], &valbook, 1, 2, &mut w).unwrap();
        let bytes = w.finish();

        let mut r = BitReaderLsb::new(&bytes);
        let out = dec.decode(&mut r, 8, &[false]).unwrap();
        assert_eq!(out, vec![vec![3.0, 5.0, 5.0, 3.0]]);
    }

    /// End-to-end §8.6.3 roundtrip: format 0's interleaved scatter
    /// (`v[offset + i + j*step] += entry_temp[j]`) reorders the decoded
    /// scalars, but the on-wire body is the same flat codeword
    /// sequence. With a 2-D book and step = 2, entries [e0, e1] land as
    /// [e0[0], e1[0], e0[1], e1[1]].
    #[test]
    fn write_residue_partition_roundtrips_against_decoder_format0_scatter() {
        use oxideav_core::bits::BitReaderLsb;

        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(2, vec![1, 1], vec![1, 2, 3, 4]);
        let e0 = crate::vq::unpack_vector(&valbook, 0).unwrap();
        let e1 = crate::vq::unpack_vector(&valbook, 1).unwrap();

        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(0, 0, 4, 4, vec![row]);
        let codebooks = vec![classbook.clone(), valbook.clone()];
        let dec = crate::residue::ResidueDecoder::new(&header, &codebooks).unwrap();

        // One partition (n_to_read = 4, partition_size = 4); format 0
        // reads n / dims = 2 codewords.
        let cls_tree = crate::huffman::HuffmanTree::from_codebook(&classbook).unwrap();
        let mut w = BitWriterLsb::new();
        cls_tree.encode_entry(0, &mut w).unwrap();
        write_residue_partition_into_writer(&[0, 1], &valbook, 0, 4, &mut w).unwrap();
        let bytes = w.finish();

        let mut r = BitReaderLsb::new(&bytes);
        let out = dec.decode(&mut r, 8, &[false]).unwrap();
        assert_eq!(out, vec![vec![e0[0], e1[0], e0[1], e1[1]]]);
    }

    /// End-to-end §8.6.5 roundtrip: format 2 decodes one interleaved
    /// vector with the format-1 rule then de-interleaves
    /// (`output[j][i] = v[i*ch + j]`). The partition body is written
    /// once for the interleaved vector.
    #[test]
    fn write_residue_partition_roundtrips_against_decoder_format2_interleaved() {
        use oxideav_core::bits::BitReaderLsb;

        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);

        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(2, 0, 4, 4, vec![row]);
        let codebooks = vec![classbook.clone(), valbook.clone()];
        let dec = crate::residue::ResidueDecoder::new(&header, &codebooks).unwrap();

        // blocksize 4, 2 channels → per-channel size 2, interleaved
        // length 4; one partition of 4 scalars → ceil(4/1) = 4 codewords.
        let cls_tree = crate::huffman::HuffmanTree::from_codebook(&classbook).unwrap();
        let mut w = BitWriterLsb::new();
        cls_tree.encode_entry(0, &mut w).unwrap();
        write_residue_partition_into_writer(&[0, 1, 1, 0], &valbook, 2, 4, &mut w).unwrap();
        let bytes = w.finish();

        let mut r = BitReaderLsb::new(&bytes);
        let out = dec.decode(&mut r, 4, &[false, false]).unwrap();
        // interleaved [3, 5, 5, 3] → ch0 = [v[0], v[2]] = [3, 5],
        // ch1 = [v[1], v[3]] = [5, 3].
        assert_eq!(out, vec![vec![3.0, 5.0], vec![5.0, 3.0]]);
    }

    /// §8.6.4 partial-final-vector roundtrip: with dims = 2 and
    /// partition_size = 3, the decoder reads ceil(3/2) = 2 codewords
    /// and discards the final vector's surplus element.
    #[test]
    fn write_residue_partition_format1_partial_final_vector_roundtrip() {
        use oxideav_core::bits::BitReaderLsb;

        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(2, vec![1, 1], vec![1, 2, 3, 4]);
        let e0 = crate::vq::unpack_vector(&valbook, 0).unwrap();
        let e1 = crate::vq::unpack_vector(&valbook, 1).unwrap();

        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 3, 3, vec![row]);
        let codebooks = vec![classbook.clone(), valbook.clone()];
        let dec = crate::residue::ResidueDecoder::new(&header, &codebooks).unwrap();

        let cls_tree = crate::huffman::HuffmanTree::from_codebook(&classbook).unwrap();
        let mut w = BitWriterLsb::new();
        cls_tree.encode_entry(0, &mut w).unwrap();
        write_residue_partition_into_writer(&[0, 1], &valbook, 1, 3, &mut w).unwrap();
        let bytes = w.finish();

        let mut r = BitReaderLsb::new(&bytes);
        let out = dec.decode(&mut r, 8, &[false]).unwrap();
        // [e0[0], e0[1], e1[0]] land; e1[1] is discarded (i reaches n);
        // index 3 stays at the §8.6.2 step-1 zero fill.
        assert_eq!(out, vec![vec![e0[0], e0[1], e1[0], 0.0]]);
    }

    /// The entry-count gate fires in both directions (too few / too
    /// many) before any bits are emitted.
    #[test]
    fn write_residue_partition_rejects_entry_count_mismatch() {
        let book = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        assert_eq!(
            write_residue_partition(&[0], &book, 1, 4),
            Err(WriteResiduePartitionError::EntryCountMismatch {
                expected: 4,
                actual: 1,
            })
        );
        assert_eq!(
            write_residue_partition(&[0, 1, 0, 1, 0], &book, 1, 4),
            Err(WriteResiduePartitionError::EntryCountMismatch {
                expected: 4,
                actual: 5,
            })
        );
    }

    /// §8.6.1: a book used in VQ context must carry a value mapping; a
    /// scalar (lookup_type 0) book is refused — mirroring the decoder's
    /// `ValueBookHasNoLookup` construction-time rejection.
    #[test]
    fn write_residue_partition_rejects_scalar_book() {
        let book = rp_scalar_book(1, vec![1, 1]);
        assert_eq!(
            write_residue_partition(&[0, 1], &book, 1, 2),
            Err(WriteResiduePartitionError::ScalarValueBook)
        );
    }

    /// An entry with no canonical codeword — out of range, or marked
    /// unused in a sparse book — is refused with its list position.
    #[test]
    fn write_residue_partition_rejects_unencodable_entry() {
        // Out-of-range entry in a dense 2-entry book.
        let dense = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        assert_eq!(
            write_residue_partition(&[0, 7], &dense, 1, 2),
            Err(WriteResiduePartitionError::UnencodableEntry {
                index: 1,
                entry: 7,
                used_count: 2,
            })
        );
        // Unused (sparse) entry: lengths [1, UNUSED, 1] build a full
        // 1-bit tree over entries {0, 2}; entry 1 has no codeword.
        let sparse = rp_vq_book(1, vec![1, UNUSED_ENTRY, 1], vec![3, 9, 5]);
        assert_eq!(
            write_residue_partition(&[0, 1], &sparse, 1, 2),
            Err(WriteResiduePartitionError::UnencodableEntry {
                index: 1,
                entry: 1,
                used_count: 2,
            })
        );
    }

    /// A book whose codeword lengths cannot build a canonical §3.2.1
    /// tree surfaces as the `Huffman` variant (with `source()`
    /// chaining to the build error).
    #[test]
    fn write_residue_partition_propagates_huffman_build_error() {
        use std::error::Error as _;
        // Three 1-bit codewords over-subscribe a binary tree.
        let bad = rp_vq_book(1, vec![1, 1, 1], vec![3, 5, 7]);
        let err = write_residue_partition(&[0, 1], &bad, 1, 2).unwrap_err();
        assert!(
            matches!(err, WriteResiduePartitionError::Huffman(_)),
            "expected Huffman build error, got {err:?}"
        );
        assert!(err.source().is_some(), "Huffman must chain its source");
    }

    /// Fail-closed splice contract: on error the caller's writer is
    /// left bit-exactly as seeded.
    #[test]
    fn write_residue_partition_splice_emits_no_bits_on_error() {
        let book = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut w = BitWriterLsb::new();
        w.write_bit(true);
        w.write_bit(false);
        w.write_bit(true);
        let err = write_residue_partition_into_writer(&[0], &book, 1, 4, &mut w).unwrap_err();
        assert!(matches!(
            err,
            WriteResiduePartitionError::EntryCountMismatch { .. }
        ));
        // Only the three seeded bits (1, 0, 1 LSb-first → 0x05).
        assert_eq!(w.finish(), vec![0x05]);
    }

    /// The public byte-aligned wrapper and the splice helper emit the
    /// same bits at byte alignment.
    #[test]
    fn write_residue_partition_public_matches_splice() {
        let book = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let public = write_residue_partition(&[0, 1, 1, 0], &book, 1, 4).unwrap();
        let mut w = BitWriterLsb::new();
        write_residue_partition_into_writer(&[0, 1, 1, 0], &book, 1, 4, &mut w).unwrap();
        assert_eq!(public, w.finish());
    }

    /// Every `Display` string is non-empty and grep-able.
    #[test]
    fn write_residue_partition_error_displays_are_informative() {
        let huffman_err = crate::huffman::HuffmanTree::from_lengths(&[1, 1, 1]).unwrap_err();
        let cases: Vec<WriteResiduePartitionError> = vec![
            WriteResiduePartitionError::UnsupportedResidueType(3),
            WriteResiduePartitionError::ZeroPartitionSize,
            WriteResiduePartitionError::ZeroDimensions,
            WriteResiduePartitionError::ScalarValueBook,
            WriteResiduePartitionError::Format0NotDivisible {
                partition_size: 5,
                dimensions: 2,
            },
            WriteResiduePartitionError::EntryCountMismatch {
                expected: 4,
                actual: 1,
            },
            WriteResiduePartitionError::Huffman(huffman_err),
            WriteResiduePartitionError::UnencodableEntry {
                index: 1,
                entry: 7,
                used_count: 2,
            },
        ];
        for case in cases {
            let msg = format!("{case}");
            assert!(
                msg.contains("vorbis residue partition (write)"),
                "Display should be grep-able: {msg}"
            );
        }
    }

    /// Umbrella glue: `WriteError::ResiduePartition` From + `source()`
    /// chain, then the crate-level `Error::Write` chain on top.
    #[test]
    fn write_residue_partition_umbrella_write_error_glue() {
        let err: WriteError = WriteResiduePartitionError::ZeroPartitionSize.into();
        match &err {
            WriteError::ResiduePartition(WriteResiduePartitionError::ZeroPartitionSize) => {}
            other => panic!("From glue wrong variant: {other:?}"),
        }
        assert!(
            std::error::Error::source(&err).is_some(),
            "ResiduePartition must chain its source"
        );
        let crate_err: crate::Error = err.into();
        match &crate_err {
            crate::Error::Write(WriteError::ResiduePartition(
                WriteResiduePartitionError::ZeroPartitionSize,
            )) => {}
            other => panic!("crate::Error glue wrong variant: {other:?}"),
        }
    }

    // ===== §8.6.2 residue body WRITE (round 38) =====

    /// Build a `[Option<Vec<u32>>; 8]` stage row with one entry list at
    /// the given pass.
    fn rb_stage_row(pass: usize, entries: Vec<u32>) -> [Option<Vec<u32>>; 8] {
        let mut row: [Option<Vec<u32>>; 8] = Default::default();
        row[pass] = Some(entries);
        row
    }

    /// §8.6.2 steps 1..5 for formats 0/1: one decode vector per
    /// channel, partitions from the begin/end-limited range.
    #[test]
    fn residue_body_shape_format1_basics() {
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 4, 2, vec![row]);
        assert_eq!(
            residue_body_shape(&header, 8, &[false]),
            Ok(ResidueBodyShape {
                vectors: 1,
                partitions_to_read: 2,
            })
        );
        // residue_end beyond the actual vector size is limited to it
        // (§8.6.2 steps 4..5): blocksize 8 → actual_size 4.
        let big_end = rp_residue_header(1, 0, 100, 2, vec![row]);
        assert_eq!(
            residue_body_shape(&big_end, 8, &[false, true]),
            Ok(ResidueBodyShape {
                vectors: 2,
                partitions_to_read: 2,
            })
        );
        // begin == end → n_to_read = 0 → no partitions.
        let empty = rp_residue_header(1, 2, 2, 2, vec![row]);
        assert_eq!(
            residue_body_shape(&empty, 8, &[false]),
            Ok(ResidueBodyShape {
                vectors: 1,
                partitions_to_read: 0,
            })
        );
    }

    /// §8.6.5 format 2: one interleaved decode vector spanning all
    /// channels (`actual_size *= ch`), or none at all when every
    /// channel is marked 'do not decode'.
    #[test]
    fn residue_body_shape_format2_interleaves_and_shortcuts() {
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(2, 0, 4, 4, vec![row]);
        // blocksize 4, 2 channels → per-channel 2, interleaved 4.
        assert_eq!(
            residue_body_shape(&header, 4, &[false, false]),
            Ok(ResidueBodyShape {
                vectors: 1,
                partitions_to_read: 1,
            })
        );
        // All 'do not decode' → no decode occurs (§8.6.5 step 1).
        assert_eq!(
            residue_body_shape(&header, 4, &[true, true]),
            Ok(ResidueBodyShape {
                vectors: 0,
                partitions_to_read: 0,
            })
        );
    }

    /// The shape helper carries the two structural rejections.
    #[test]
    fn residue_body_shape_rejects_invalid_inputs() {
        let mut row = [None; 8];
        row[0] = Some(1);
        let bad_type = rp_residue_header(3, 0, 4, 2, vec![row]);
        assert_eq!(
            residue_body_shape(&bad_type, 8, &[false]),
            Err(WriteResidueBodyError::UnsupportedResidueType(3))
        );
        let zero_psize = rp_residue_header(1, 0, 4, 0, vec![row]);
        assert_eq!(
            residue_body_shape(&zero_psize, 8, &[false]),
            Err(WriteResidueBodyError::ZeroPartitionSize)
        );
    }

    /// End-to-end §8.6.2 roundtrip, format 1, single channel, two
    /// partitions: the writer's bytes equal the hand-composed
    /// classbook-codeword + partition-body stream the round-37 test
    /// pinned, and the residue decoder reads them back to the expected
    /// vector.
    #[test]
    fn write_residue_body_format1_byte_shape_and_roundtrip() {
        use oxideav_core::bits::BitReaderLsb;

        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 4, 2, vec![row]);
        let codebooks = vec![classbook.clone(), valbook.clone()];

        let plan = ResidueVectorPlan {
            classifications: vec![0, 0],
            partition_entries: vec![rb_stage_row(0, vec![0, 1]), rb_stage_row(0, vec![1, 0])],
        };
        let bytes = write_residue_body(&[plan], &header, &codebooks, 8, &[false]).unwrap();

        // Hand-composed expectation (§8.6.2 pass-0 order, classwords=1):
        // cls(p0) → body(p0) → cls(p1) → body(p1).
        let cls_tree = crate::huffman::HuffmanTree::from_codebook(&classbook).unwrap();
        let mut w = BitWriterLsb::new();
        cls_tree.encode_entry(0, &mut w).unwrap();
        write_residue_partition_into_writer(&[0, 1], &valbook, 1, 2, &mut w).unwrap();
        cls_tree.encode_entry(0, &mut w).unwrap();
        write_residue_partition_into_writer(&[1, 0], &valbook, 1, 2, &mut w).unwrap();
        assert_eq!(bytes, w.finish());

        let dec = crate::residue::ResidueDecoder::new(&header, &codebooks).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let out = dec.decode(&mut r, 8, &[false]).unwrap();
        assert_eq!(out, vec![vec![3.0, 5.0, 5.0, 3.0]]);
    }

    /// §8.6.2 step-14 vector interleave: with two decoded channels the
    /// stream alternates per-vector classbook codewords and per-vector
    /// partition bodies. Pinned against a hand-composed stream and the
    /// decoder.
    #[test]
    fn write_residue_body_format1_two_channels_interleave() {
        use oxideav_core::bits::BitReaderLsb;

        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 2, 2, vec![row]);
        let codebooks = vec![classbook.clone(), valbook.clone()];

        let plan0 = ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![rb_stage_row(0, vec![0, 1])],
        };
        let plan1 = ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![rb_stage_row(0, vec![1, 0])],
        };
        let bytes =
            write_residue_body(&[plan0, plan1], &header, &codebooks, 4, &[false, false]).unwrap();

        // §8.6.2 order: cls(j=0), cls(j=1), body(j=0), body(j=1).
        let cls_tree = crate::huffman::HuffmanTree::from_codebook(&classbook).unwrap();
        let mut w = BitWriterLsb::new();
        cls_tree.encode_entry(0, &mut w).unwrap();
        cls_tree.encode_entry(0, &mut w).unwrap();
        write_residue_partition_into_writer(&[0, 1], &valbook, 1, 2, &mut w).unwrap();
        write_residue_partition_into_writer(&[1, 0], &valbook, 1, 2, &mut w).unwrap();
        assert_eq!(bytes, w.finish());

        let dec = crate::residue::ResidueDecoder::new(&header, &codebooks).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let out = dec.decode(&mut r, 4, &[false, false]).unwrap();
        assert_eq!(out, vec![vec![3.0, 5.0], vec![5.0, 3.0]]);
    }

    /// A 'do not decode' channel contributes no bits; its plan must be
    /// empty and the decoder returns it zeroed.
    #[test]
    fn write_residue_body_skips_do_not_decode_channel() {
        use oxideav_core::bits::BitReaderLsb;

        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 2, 2, vec![row]);
        let codebooks = vec![classbook.clone(), valbook.clone()];

        let plan0 = ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![rb_stage_row(0, vec![0, 1])],
        };
        let bytes = write_residue_body(
            &[plan0, ResidueVectorPlan::default()],
            &header,
            &codebooks,
            4,
            &[false, true],
        )
        .unwrap();

        // Identical to a single-channel emission of plan0.
        let cls_tree = crate::huffman::HuffmanTree::from_codebook(&classbook).unwrap();
        let mut w = BitWriterLsb::new();
        cls_tree.encode_entry(0, &mut w).unwrap();
        write_residue_partition_into_writer(&[0, 1], &valbook, 1, 2, &mut w).unwrap();
        assert_eq!(bytes, w.finish());

        let dec = crate::residue::ResidueDecoder::new(&header, &codebooks).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let out = dec.decode(&mut r, 4, &[false, true]).unwrap();
        assert_eq!(out, vec![vec![3.0, 5.0], vec![0.0, 0.0]]);
    }

    /// End-to-end §8.6.3 roundtrip: format 0's scatter is decode-side
    /// addressing; the body writes the same flat codeword sequence.
    #[test]
    fn write_residue_body_format0_scatter_roundtrip() {
        use oxideav_core::bits::BitReaderLsb;

        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(2, vec![1, 1], vec![1, 2, 3, 4]);
        let e0 = crate::vq::unpack_vector(&valbook, 0).unwrap();
        let e1 = crate::vq::unpack_vector(&valbook, 1).unwrap();
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(0, 0, 4, 4, vec![row]);
        let codebooks = vec![classbook, valbook];

        let plan = ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![rb_stage_row(0, vec![0, 1])],
        };
        let bytes = write_residue_body(&[plan], &header, &codebooks, 8, &[false]).unwrap();

        let dec = crate::residue::ResidueDecoder::new(&header, &codebooks).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let out = dec.decode(&mut r, 8, &[false]).unwrap();
        assert_eq!(out, vec![vec![e0[0], e1[0], e0[1], e1[1]]]);
    }

    /// End-to-end §8.6.5 roundtrip: format 2 takes ONE interleaved-
    /// vector plan; the decoder de-interleaves into both channels.
    #[test]
    fn write_residue_body_format2_interleaved_roundtrip() {
        use oxideav_core::bits::BitReaderLsb;

        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(2, 0, 4, 4, vec![row]);
        let codebooks = vec![classbook, valbook];

        let plan = ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![rb_stage_row(0, vec![0, 1, 1, 0])],
        };
        let bytes = write_residue_body(&[plan], &header, &codebooks, 4, &[false, false]).unwrap();

        let dec = crate::residue::ResidueDecoder::new(&header, &codebooks).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let out = dec.decode(&mut r, 4, &[false, false]).unwrap();
        // interleaved [3, 5, 5, 3] → ch0 [3, 5], ch1 [5, 3].
        assert_eq!(out, vec![vec![3.0, 5.0], vec![5.0, 3.0]]);
    }

    /// §8.6.5's all-'do not decode' shortcut: zero plans, zero bytes.
    #[test]
    fn write_residue_body_format2_all_do_not_decode_is_empty() {
        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(2, 0, 4, 4, vec![row]);
        let codebooks = vec![classbook, valbook];
        let bytes = write_residue_body(&[], &header, &codebooks, 4, &[true, true]).unwrap();
        assert!(bytes.is_empty());
    }

    /// §8.6.2 step-2 empty body: begin == end reads nothing, so plan
    /// rows are empty and no bits are emitted.
    #[test]
    fn write_residue_body_empty_range_emits_nothing() {
        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 2, 2, 2, vec![row]);
        let codebooks = vec![classbook, valbook];
        let bytes = write_residue_body(
            &[ResidueVectorPlan::default()],
            &header,
            &codebooks,
            8,
            &[false],
        )
        .unwrap();
        assert!(bytes.is_empty());
    }

    /// Multi-pass cascade: a class with books at stages 0 AND 1 emits
    /// the pass-1 partition body after the whole pass-0 walk, with no
    /// further classbook codewords (§8.6.2 step 6 gates them to pass
    /// 0). The decoder accumulates both bodies (`+=`).
    #[test]
    fn write_residue_body_multi_pass_cascade_accumulates() {
        use oxideav_core::bits::BitReaderLsb;

        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        row[1] = Some(1);
        let header = rp_residue_header(1, 0, 2, 2, vec![row]);
        let codebooks = vec![classbook.clone(), valbook.clone()];

        let mut stages = rb_stage_row(0, vec![0, 1]);
        stages[1] = Some(vec![1, 1]);
        let plan = ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![stages],
        };
        let bytes = write_residue_body(&[plan], &header, &codebooks, 4, &[false]).unwrap();

        // Hand-composed: pass 0 → cls + body [0,1]; pass 1 → body [1,1].
        let cls_tree = crate::huffman::HuffmanTree::from_codebook(&classbook).unwrap();
        let mut w = BitWriterLsb::new();
        cls_tree.encode_entry(0, &mut w).unwrap();
        write_residue_partition_into_writer(&[0, 1], &valbook, 1, 2, &mut w).unwrap();
        write_residue_partition_into_writer(&[1, 1], &valbook, 1, 2, &mut w).unwrap();
        assert_eq!(bytes, w.finish());

        let dec = crate::residue::ResidueDecoder::new(&header, &codebooks).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let out = dec.decode(&mut r, 4, &[false]).unwrap();
        // [3, 5] from pass 0 plus [5, 5] from pass 1.
        assert_eq!(out, vec![vec![8.0, 10.0]]);
    }

    /// Multi-group classification stream: classwords = 2 over 4
    /// partitions and 2 classes — each stride of 2 partitions is
    /// preceded by ONE classbook codeword packing both classifications,
    /// and each partition's body uses its class's own value book.
    #[test]
    fn write_residue_body_multi_group_classifications_roundtrip() {
        use oxideav_core::bits::BitReaderLsb;

        // Classbook: dims 2, 4 entries (base-2 packed pairs), 2-bit
        // codewords each.
        let classbook = rp_scalar_book(2, vec![2, 2, 2, 2]);
        let val_a = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let val_b = rp_vq_book(1, vec![1, 1], vec![7, 9]);
        let mut row_a = [None; 8];
        row_a[0] = Some(1);
        let mut row_b = [None; 8];
        row_b[0] = Some(2);
        let header = rp_residue_header(1, 0, 8, 2, vec![row_a, row_b]);
        let codebooks = vec![classbook.clone(), val_a.clone(), val_b.clone()];

        // classes [0, 1, 1, 0] → groups [0,1] → entry 1, [1,0] → entry 2.
        let plan = ResidueVectorPlan {
            classifications: vec![0, 1, 1, 0],
            partition_entries: vec![
                rb_stage_row(0, vec![0, 1]), // class 0 → val_a → [3, 5]
                rb_stage_row(0, vec![1, 0]), // class 1 → val_b → [9, 7]
                rb_stage_row(0, vec![0, 0]), // class 1 → val_b → [7, 7]
                rb_stage_row(0, vec![1, 1]), // class 0 → val_a → [5, 5]
            ],
        };
        let bytes = write_residue_body(&[plan], &header, &codebooks, 16, &[false]).unwrap();

        // Hand-composed: cls(group 0)=1, bodies p0+p1, cls(group 1)=2,
        // bodies p2+p3.
        let cls_tree = crate::huffman::HuffmanTree::from_codebook(&classbook).unwrap();
        let mut w = BitWriterLsb::new();
        cls_tree.encode_entry(1, &mut w).unwrap();
        write_residue_partition_into_writer(&[0, 1], &val_a, 1, 2, &mut w).unwrap();
        write_residue_partition_into_writer(&[1, 0], &val_b, 1, 2, &mut w).unwrap();
        cls_tree.encode_entry(2, &mut w).unwrap();
        write_residue_partition_into_writer(&[0, 0], &val_b, 1, 2, &mut w).unwrap();
        write_residue_partition_into_writer(&[1, 1], &val_a, 1, 2, &mut w).unwrap();
        assert_eq!(bytes, w.finish());

        let dec = crate::residue::ResidueDecoder::new(&header, &codebooks).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let out = dec.decode(&mut r, 16, &[false]).unwrap();
        assert_eq!(out, vec![vec![3.0, 5.0, 9.0, 7.0, 7.0, 7.0, 5.0, 5.0]]);
    }

    /// Partial final classification group: 3 partitions with classwords
    /// = 2 — the second classbook codeword carries one real digit plus
    /// the zero pad the decoder reads-and-discards.
    #[test]
    fn write_residue_body_partial_final_group_roundtrip() {
        use oxideav_core::bits::BitReaderLsb;

        let classbook = rp_scalar_book(2, vec![2, 2, 2, 2]);
        let val_a = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let val_b = rp_vq_book(1, vec![1, 1], vec![7, 9]);
        let mut row_a = [None; 8];
        row_a[0] = Some(1);
        let mut row_b = [None; 8];
        row_b[0] = Some(2);
        let header = rp_residue_header(1, 0, 6, 2, vec![row_a, row_b]);
        let codebooks = vec![classbook, val_a, val_b];

        // classes [1, 0, 1] → groups [1,0] → 2, [1,(pad 0)] → 2.
        let plan = ResidueVectorPlan {
            classifications: vec![1, 0, 1],
            partition_entries: vec![
                rb_stage_row(0, vec![0, 1]), // val_b → [7, 9]
                rb_stage_row(0, vec![1, 0]), // val_a → [5, 3]
                rb_stage_row(0, vec![1, 1]), // val_b → [9, 9]
            ],
        };
        let bytes = write_residue_body(&[plan], &header, &codebooks, 16, &[false]).unwrap();

        let dec = crate::residue::ResidueDecoder::new(&header, &codebooks).unwrap();
        let mut r = BitReaderLsb::new(&bytes);
        let out = dec.decode(&mut r, 16, &[false]).unwrap();
        assert_eq!(out, vec![vec![7.0, 9.0, 5.0, 3.0, 9.0, 9.0, 0.0, 0.0]]);
    }

    /// The plan-count gate fires for both too-few and too-many plans.
    #[test]
    fn write_residue_body_rejects_plan_count_mismatch() {
        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 2, 2, vec![row]);
        let codebooks = vec![classbook, valbook];
        assert_eq!(
            write_residue_body(&[], &header, &codebooks, 4, &[false, false]),
            Err(WriteResidueBodyError::PlanCountMismatch {
                expected: 2,
                actual: 0,
            })
        );
    }

    /// A non-empty plan on a 'do not decode' vector is refused — its
    /// content can never reach the wire.
    #[test]
    fn write_residue_body_rejects_nonempty_do_not_decode_plan() {
        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 2, 2, vec![row]);
        let codebooks = vec![classbook, valbook];
        let stray = ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![rb_stage_row(0, vec![0, 1])],
        };
        assert_eq!(
            write_residue_body(&[stray], &header, &codebooks, 4, &[true]),
            Err(WriteResidueBodyError::DoNotDecodePlanNotEmpty { vector: 0 })
        );
    }

    /// Both row-length gates fire with the offending vector index.
    #[test]
    fn write_residue_body_rejects_row_length_mismatches() {
        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 4, 2, vec![row]);
        let codebooks = vec![classbook, valbook];

        let short_cls = ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![rb_stage_row(0, vec![0, 1]), rb_stage_row(0, vec![1, 0])],
        };
        assert_eq!(
            write_residue_body(&[short_cls], &header, &codebooks, 8, &[false]),
            Err(WriteResidueBodyError::ClassificationCountMismatch {
                vector: 0,
                expected: 2,
                actual: 1,
            })
        );

        let short_entries = ResidueVectorPlan {
            classifications: vec![0, 0],
            partition_entries: vec![rb_stage_row(0, vec![0, 1])],
        };
        assert_eq!(
            write_residue_body(&[short_entries], &header, &codebooks, 8, &[false]),
            Err(WriteResidueBodyError::PartitionEntriesCountMismatch {
                vector: 0,
                expected: 2,
                actual: 1,
            })
        );
    }

    /// A classification at or above `residue_classifications` has no
    /// §8.6.2 unpack round-trip and cannot index `residue_books`.
    #[test]
    fn write_residue_body_rejects_classification_out_of_range() {
        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 2, 2, vec![row]);
        let codebooks = vec![classbook, valbook];
        let plan = ResidueVectorPlan {
            classifications: vec![1],
            partition_entries: vec![rb_stage_row(0, vec![0, 1])],
        };
        assert_eq!(
            write_residue_body(&[plan], &header, &codebooks, 4, &[false]),
            Err(WriteResidueBodyError::ClassificationOutOfRange {
                vector: 0,
                partition: 0,
                classification: 1,
                num_classifications: 1,
            })
        );
    }

    /// Cascade-presence gates: a missing entry list where the cascade
    /// holds a book, and a stray entry list where it does not.
    #[test]
    fn write_residue_body_rejects_cascade_presence_mismatches() {
        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 2, 2, vec![row]);
        let codebooks = vec![classbook, valbook];

        let missing = ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![Default::default()],
        };
        assert_eq!(
            write_residue_body(&[missing], &header, &codebooks, 4, &[false]),
            Err(WriteResidueBodyError::MissingPartitionEntries {
                vector: 0,
                partition: 0,
                pass: 0,
            })
        );

        let mut stray_stages = rb_stage_row(0, vec![0, 1]);
        stray_stages[3] = Some(vec![0, 1]);
        let unexpected = ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![stray_stages],
        };
        assert_eq!(
            write_residue_body(&[unexpected], &header, &codebooks, 4, &[false]),
            Err(WriteResidueBodyError::UnexpectedPartitionEntries {
                vector: 0,
                partition: 0,
                pass: 3,
            })
        );
    }

    /// §8.6.1 header/codebook gates mirror the decoder's construction-
    /// time checks: classbook out of range, zero classwords, value
    /// book out of range, value book without a value mapping.
    #[test]
    fn write_residue_body_rejects_header_codebook_inconsistencies() {
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);

        // classbook index 0 with an empty codebook table.
        let header = rp_residue_header(1, 0, 2, 2, vec![row]);
        assert_eq!(
            write_residue_body(&[ResidueVectorPlan::default()], &header, &[], 4, &[false]),
            Err(WriteResidueBodyError::ClassbookOutOfRange {
                classbook: 0,
                codebook_count: 0,
            })
        );

        // Zero-dimension classbook.
        let zero_dims = rp_scalar_book(0, vec![1, 1]);
        assert_eq!(
            write_residue_body(
                &[ResidueVectorPlan::default()],
                &header,
                &[zero_dims, valbook.clone()],
                4,
                &[false],
            ),
            Err(WriteResidueBodyError::ZeroClasswordsPerCodeword)
        );

        // Value book index beyond the table.
        let classbook = rp_scalar_book(1, vec![1, 1]);
        let mut far_row = [None; 8];
        far_row[2] = Some(9);
        let far_header = rp_residue_header(1, 0, 2, 2, vec![far_row]);
        assert_eq!(
            write_residue_body(
                &[ResidueVectorPlan::default()],
                &far_header,
                &[classbook.clone(), valbook],
                4,
                &[false],
            ),
            Err(WriteResidueBodyError::ValueBookOutOfRange {
                class: 0,
                stage: 2,
                book: 9,
                codebook_count: 2,
            })
        );

        // Value book without a value mapping (scalar in VQ context).
        let scalar_val = rp_scalar_book(1, vec![1, 1]);
        assert_eq!(
            write_residue_body(
                &[ResidueVectorPlan::default()],
                &header,
                &[classbook, scalar_val],
                4,
                &[false],
            ),
            Err(WriteResidueBodyError::ValueBookHasNoLookup {
                class: 0,
                stage: 0,
                book: 1,
            })
        );
    }

    /// A per-partition failure surfaces as `Partition` with the
    /// (vector, partition, pass) coordinates and the round-37 error
    /// verbatim, `source()`-chained.
    #[test]
    fn write_residue_body_wraps_partition_errors() {
        use std::error::Error as _;
        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 2, 2, vec![row]);
        let codebooks = vec![classbook, valbook];
        let plan = ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![rb_stage_row(0, vec![0])], // needs 2
        };
        let err = write_residue_body(&[plan], &header, &codebooks, 4, &[false]).unwrap_err();
        assert_eq!(
            err,
            WriteResidueBodyError::Partition {
                vector: 0,
                partition: 0,
                pass: 0,
                source: WriteResiduePartitionError::EntryCountMismatch {
                    expected: 2,
                    actual: 1,
                },
            }
        );
        assert!(err.source().is_some(), "Partition must chain its source");
    }

    /// A packed classbook entry with no canonical codeword (sparse
    /// classbook) is refused with its group coordinates.
    #[test]
    fn write_residue_body_rejects_unencodable_classbook_entry() {
        // Two classes; the sparse classbook's entry 1 is unused, so
        // classification 1 (packing to entry 1) has no codeword.
        let classbook = rp_scalar_book(1, vec![1, UNUSED_ENTRY, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 2, 2, vec![row, row]);
        let codebooks = vec![classbook, valbook];
        let plan = ResidueVectorPlan {
            classifications: vec![1],
            partition_entries: vec![rb_stage_row(0, vec![0, 1])],
        };
        let err = write_residue_body(&[plan], &header, &codebooks, 4, &[false]).unwrap_err();
        assert_eq!(
            err,
            WriteResidueBodyError::UnencodableClassbookEntry {
                vector: 0,
                group: 0,
                entry: 1,
                used_count: 2,
            }
        );
    }

    /// A classification-packing failure (packed index overflowing the
    /// u32 classbook read) surfaces as `ClassificationPack` with the
    /// inner groups error verbatim, `source()`-chained.
    #[test]
    fn write_residue_body_wraps_classification_pack_errors() {
        use std::error::Error as _;
        // Base 64 with classwords 6: 64^6 - 1 > u32::MAX.
        let classbook = rp_scalar_book(6, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let rows: Vec<[Option<u8>; 8]> = vec![row; 64];
        let header = rp_residue_header(1, 0, 12, 2, rows);
        let codebooks = vec![classbook, valbook];
        let plan = ResidueVectorPlan {
            classifications: vec![63; 6],
            partition_entries: vec![rb_stage_row(0, vec![0, 1]); 6],
        };
        let err = write_residue_body(&[plan], &header, &codebooks, 32, &[false]).unwrap_err();
        match &err {
            WriteResidueBodyError::ClassificationPack { vector: 0, source } => {
                assert!(
                    matches!(
                        source,
                        PackResidueClassGroupsError::Pack {
                            group: 0,
                            source: PackResidueClassError::PackedValueOverflow { .. },
                        }
                    ),
                    "inner error should be the overflow: {source:?}"
                );
            }
            other => panic!("expected ClassificationPack, got {other:?}"),
        }
        assert!(err.source().is_some(), "must chain to the groups error");
    }

    /// Fail-closed splice contract: on error the caller's writer is
    /// left bit-exactly as seeded, even for an error found late in
    /// validation (after earlier partitions validated clean).
    #[test]
    fn write_residue_body_splice_emits_no_bits_on_error() {
        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 4, 2, vec![row]);
        let codebooks = vec![classbook, valbook];
        // Partition 0 is valid; partition 1 has a bad entry count.
        let plan = ResidueVectorPlan {
            classifications: vec![0, 0],
            partition_entries: vec![rb_stage_row(0, vec![0, 1]), rb_stage_row(0, vec![1])],
        };
        let mut w = BitWriterLsb::new();
        w.write_bit(true);
        w.write_bit(false);
        w.write_bit(true);
        let err = write_residue_body_into_writer(&[plan], &header, &codebooks, 8, &[false], &mut w)
            .unwrap_err();
        assert!(matches!(
            err,
            WriteResidueBodyError::Partition { partition: 1, .. }
        ));
        // Only the three seeded bits (1, 0, 1 LSb-first → 0x05).
        assert_eq!(w.finish(), vec![0x05]);
    }

    /// The public byte-aligned wrapper and the splice helper emit the
    /// same bits at byte alignment, and the splice appends after
    /// pre-existing bits without disturbing them.
    #[test]
    fn write_residue_body_public_matches_splice() {
        let classbook = rp_scalar_book(1, vec![1, 1]);
        let valbook = rp_vq_book(1, vec![1, 1], vec![3, 5]);
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = rp_residue_header(1, 0, 4, 2, vec![row]);
        let codebooks = vec![classbook.clone(), valbook.clone()];
        let plan = ResidueVectorPlan {
            classifications: vec![0, 0],
            partition_entries: vec![rb_stage_row(0, vec![0, 1]), rb_stage_row(0, vec![1, 0])],
        };
        let public = write_residue_body(
            std::slice::from_ref(&plan),
            &header,
            &codebooks,
            8,
            &[false],
        )
        .unwrap();
        let mut w = BitWriterLsb::new();
        write_residue_body_into_writer(
            std::slice::from_ref(&plan),
            &header,
            &codebooks,
            8,
            &[false],
            &mut w,
        )
        .unwrap();
        assert_eq!(public, w.finish());

        // Seeded splice: same body bits after a 3-bit prefix.
        let mut seeded = BitWriterLsb::new();
        seeded.write_bit(true);
        seeded.write_bit(false);
        seeded.write_bit(true);
        write_residue_body_into_writer(&[plan], &header, &codebooks, 8, &[false], &mut seeded)
            .unwrap();
        let mut expected = BitWriterLsb::new();
        expected.write_bit(true);
        expected.write_bit(false);
        expected.write_bit(true);
        let cls_tree = crate::huffman::HuffmanTree::from_codebook(&classbook).unwrap();
        cls_tree.encode_entry(0, &mut expected).unwrap();
        write_residue_partition_into_writer(&[0, 1], &valbook, 1, 2, &mut expected).unwrap();
        cls_tree.encode_entry(0, &mut expected).unwrap();
        write_residue_partition_into_writer(&[1, 0], &valbook, 1, 2, &mut expected).unwrap();
        assert_eq!(seeded.finish(), expected.finish());
    }

    /// Every `Display` string is non-empty and grep-able.
    #[test]
    fn write_residue_body_error_displays_are_informative() {
        let huffman_err = crate::huffman::HuffmanTree::from_lengths(&[1, 1, 1]).unwrap_err();
        let cases: Vec<WriteResidueBodyError> = vec![
            WriteResidueBodyError::UnsupportedResidueType(3),
            WriteResidueBodyError::ZeroPartitionSize,
            WriteResidueBodyError::ClassbookOutOfRange {
                classbook: 7,
                codebook_count: 2,
            },
            WriteResidueBodyError::ZeroClasswordsPerCodeword,
            WriteResidueBodyError::ValueBookOutOfRange {
                class: 1,
                stage: 2,
                book: 9,
                codebook_count: 3,
            },
            WriteResidueBodyError::ValueBookHasNoLookup {
                class: 1,
                stage: 2,
                book: 9,
            },
            WriteResidueBodyError::PlanCountMismatch {
                expected: 2,
                actual: 0,
            },
            WriteResidueBodyError::DoNotDecodePlanNotEmpty { vector: 1 },
            WriteResidueBodyError::ClassificationCountMismatch {
                vector: 0,
                expected: 2,
                actual: 1,
            },
            WriteResidueBodyError::PartitionEntriesCountMismatch {
                vector: 0,
                expected: 2,
                actual: 1,
            },
            WriteResidueBodyError::ClassificationOutOfRange {
                vector: 0,
                partition: 1,
                classification: 4,
                num_classifications: 2,
            },
            WriteResidueBodyError::MissingPartitionEntries {
                vector: 0,
                partition: 1,
                pass: 2,
            },
            WriteResidueBodyError::UnexpectedPartitionEntries {
                vector: 0,
                partition: 1,
                pass: 2,
            },
            WriteResidueBodyError::ClassificationPack {
                vector: 0,
                source: PackResidueClassGroupsError::ZeroClasswords,
            },
            WriteResidueBodyError::UnencodableClassbookEntry {
                vector: 0,
                group: 1,
                entry: 5,
                used_count: 2,
            },
            WriteResidueBodyError::Huffman(huffman_err),
            WriteResidueBodyError::Partition {
                vector: 0,
                partition: 1,
                pass: 2,
                source: WriteResiduePartitionError::ZeroPartitionSize,
            },
        ];
        for case in cases {
            let msg = format!("{case}");
            assert!(
                msg.contains("vorbis residue body (write)"),
                "Display should be grep-able: {msg}"
            );
        }
    }

    /// Umbrella glue: `WriteError::ResidueBody` From + `source()`
    /// chain, then the crate-level `Error::Write` chain on top.
    #[test]
    fn write_residue_body_umbrella_write_error_glue() {
        let err: WriteError = WriteResidueBodyError::ZeroPartitionSize.into();
        match &err {
            WriteError::ResidueBody(WriteResidueBodyError::ZeroPartitionSize) => {}
            other => panic!("From glue wrong variant: {other:?}"),
        }
        assert!(
            std::error::Error::source(&err).is_some(),
            "ResidueBody must chain its source"
        );
        let crate_err: crate::Error = err.into();
        match &crate_err {
            crate::Error::Write(WriteError::ResidueBody(
                WriteResidueBodyError::ZeroPartitionSize,
            )) => {}
            other => panic!("crate::Error glue wrong variant: {other:?}"),
        }
    }

    // ===== §6.2.2 floor 0 packet body WRITE (round 39) =====

    use crate::floor0::{Floor0Curve, Floor0Decoder};
    use crate::setup::Floor0Header;

    /// A tessellation (lookup_type 2) VQ value book with `delta = 1`,
    /// `min = 0`, no sequence flag — entry `e`'s vector is the
    /// multiplicand row `multiplicands[e*dims .. (e+1)*dims]` verbatim.
    /// Mirrors the round-37 `rp_vq_book` helper.
    fn f0_vq_book(dimensions: u16, lengths: Vec<u8>, multiplicands: Vec<u32>) -> VorbisCodebook {
        VorbisCodebook {
            dimensions,
            entries: lengths.len() as u32,
            codeword_lengths: lengths,
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 1.0,
                value_bits: 8,
                sequence_p: false,
                multiplicands,
            },
        }
    }

    /// A minimal floor 0 header selecting `book_list` value books.
    fn f0_header(order: u8, amplitude_bits: u8, book_list: Vec<u8>) -> Floor0Header {
        Floor0Header {
            order,
            rate: 44_100,
            bark_map_size: 256,
            amplitude_bits,
            amplitude_offset: 0,
            book_list,
        }
    }

    /// Re-read amplitude / booknumber / entries straight off the bytes
    /// the writer produced, in the §6.2.2 decoder read order, so a test
    /// can pin the on-wire content exactly.
    fn f0_read_back(
        bytes: &[u8],
        header: &Floor0Header,
        codebooks: &[VorbisCodebook],
    ) -> (u32, u32, Vec<u32>) {
        use oxideav_core::bits::BitReaderLsb;
        let mut r = BitReaderLsb::new(bytes);
        let amplitude = r.read_u32(header.amplitude_bits as u32).unwrap();
        if amplitude == 0 {
            return (0, 0, Vec::new());
        }
        let book_index_bits = ilog(header.book_list.len() as u32);
        let booknumber = r.read_u32(book_index_bits).unwrap();
        let book = &codebooks[header.book_list[booknumber as usize] as usize];
        let tree = crate::huffman::HuffmanTree::from_codebook(book).unwrap();
        let dims = book.dimensions as usize;
        let count = (header.order as usize).div_ceil(dims);
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            entries.push(tree.decode_entry(&mut r).unwrap());
        }
        (amplitude, booknumber, entries)
    }

    #[test]
    fn f0_unused_emits_only_zero_amplitude() {
        // amplitude_bits = 5 → one 5-bit zero field, no booknumber.
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        let header = f0_header(4, 5, vec![0]);
        let books = vec![book];
        let bytes = write_floor0_packet(&Floor0Packet::Unused, &header, &books).unwrap();
        // 5 bits of zero → one byte (byte-aligned), all zero.
        assert_eq!(bytes, vec![0x00]);
        // Decoder reads it back as Unused.
        let dec = Floor0Decoder::new(&header, &books).unwrap();
        let mut r = oxideav_core::bits::BitReaderLsb::new(&bytes);
        assert_eq!(dec.decode(&mut r, 128), Floor0Curve::Unused);
    }

    #[test]
    fn f0_curve_single_vector_roundtrip() {
        // order = 2, dims = 2 → exactly one VQ vector (one entry).
        let book = f0_vq_book(2, vec![1, 1], vec![10, 20, 30, 40]);
        let header = f0_header(2, 6, vec![0]);
        let packet = Floor0Packet::Curve {
            amplitude: 17,
            booknumber: 0,
            entries: vec![1],
        };
        let books = vec![book];
        let bytes = write_floor0_packet(&packet, &header, &books).unwrap();
        let (amp, bn, entries) = f0_read_back(&bytes, &header, &books);
        assert_eq!(amp, 17);
        assert_eq!(bn, 0);
        assert_eq!(entries, vec![1]);
        // And the real decoder produces a nonzero-length curve, not Unused.
        let dec = Floor0Decoder::new(&header, &books).unwrap();
        let mut r = oxideav_core::bits::BitReaderLsb::new(&bytes);
        match dec.decode(&mut r, 64) {
            Floor0Curve::Curve(c) => assert_eq!(c.len(), 64),
            other => panic!("expected Curve, got {other:?}"),
        }
    }

    #[test]
    fn f0_curve_multi_vector_count() {
        // order = 5, dims = 2 → ceil(5/2) = 3 vectors (3 entries).
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        let header = f0_header(5, 4, vec![0]);
        let packet = Floor0Packet::Curve {
            amplitude: 9,
            booknumber: 0,
            entries: vec![0, 1, 0],
        };
        let books = vec![book];
        let bytes = write_floor0_packet(&packet, &header, &books).unwrap();
        let (amp, bn, entries) = f0_read_back(&bytes, &header, &books);
        assert_eq!(amp, 9);
        assert_eq!(bn, 0);
        assert_eq!(entries, vec![0, 1, 0]);
    }

    #[test]
    fn f0_curve_second_book_selected() {
        // Two books in book_list; booknumber = 1 selects the second.
        // book_list.len() = 2 → ilog(2) = 2-bit booknumber field.
        let book0 = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        let book1 = f0_vq_book(1, vec![1, 1], vec![5, 6]);
        let codebooks = vec![book0, book1];
        // book_list maps positions 0,1 → codebook indices 0,1.
        let header = f0_header(3, 6, vec![0, 1]);
        // booknumber 1 → book_list[1] = codebook 1 (dims 1) → 3 entries.
        let packet = Floor0Packet::Curve {
            amplitude: 33,
            booknumber: 1,
            entries: vec![0, 1, 0],
        };
        let bytes = write_floor0_packet(&packet, &header, &codebooks).unwrap();
        let (amp, bn, entries) = f0_read_back(&bytes, &header, &codebooks);
        assert_eq!(amp, 33);
        assert_eq!(bn, 1);
        assert_eq!(entries, vec![0, 1, 0]);
    }

    #[test]
    fn f0_into_writer_splice_no_bits_on_error() {
        // A seeded writer must be byte-identical after a refused call.
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        let header = f0_header(2, 6, vec![0]);
        // entries.len() mismatch (expected 1, got 2) → late error.
        let bad = Floor0Packet::Curve {
            amplitude: 1,
            booknumber: 0,
            entries: vec![0, 1],
        };
        let mut w = BitWriterLsb::new();
        w.write_u32(0b101, 3); // seed 3 bits
        let before = {
            let mut probe = BitWriterLsb::new();
            probe.write_u32(0b101, 3);
            probe.finish()
        };
        let err = write_floor0_packet_into_writer(&bad, &header, &[book], &mut w).unwrap_err();
        assert!(matches!(
            err,
            WriteFloor0PacketError::EntryCountMismatch {
                expected: 1,
                actual: 2
            }
        ));
        assert_eq!(w.finish(), before, "no bits emitted on error");
    }

    #[test]
    fn f0_public_matches_splice() {
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        let header = f0_header(2, 6, vec![0]);
        let packet = Floor0Packet::Curve {
            amplitude: 5,
            booknumber: 0,
            entries: vec![1],
        };
        let books = vec![book];
        let pubbytes = write_floor0_packet(&packet, &header, &books).unwrap();
        let mut w = BitWriterLsb::new();
        write_floor0_packet_into_writer(&packet, &header, &books, &mut w).unwrap();
        assert_eq!(pubbytes, w.finish());
    }

    #[test]
    fn f0_rejects_zero_amplitude_bits_header() {
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        let header = f0_header(2, 0, vec![0]);
        let err = write_floor0_packet(&Floor0Packet::Unused, &header, &[book]).unwrap_err();
        assert_eq!(err, WriteFloor0PacketError::ZeroAmplitudeBits);
    }

    #[test]
    fn f0_rejects_amplitude_bits_overflow() {
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        let header = f0_header(2, 64, vec![0]);
        let err = write_floor0_packet(&Floor0Packet::Unused, &header, &[book]).unwrap_err();
        assert_eq!(err, WriteFloor0PacketError::AmplitudeBitsOverflow(64));
    }

    #[test]
    fn f0_rejects_zero_order() {
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        let header = f0_header(0, 6, vec![0]);
        let err = write_floor0_packet(&Floor0Packet::Unused, &header, &[book]).unwrap_err();
        assert_eq!(err, WriteFloor0PacketError::ZeroOrder);
    }

    #[test]
    fn f0_rejects_empty_book_list() {
        let header = f0_header(2, 6, vec![]);
        let err = write_floor0_packet(&Floor0Packet::Unused, &header, &[]).unwrap_err();
        assert_eq!(err, WriteFloor0PacketError::EmptyBookList);
    }

    #[test]
    fn f0_rejects_book_list_too_long() {
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        let header = f0_header(2, 6, vec![0; 17]);
        let err = write_floor0_packet(&Floor0Packet::Unused, &header, &[book]).unwrap_err();
        assert_eq!(err, WriteFloor0PacketError::BookListTooLong(17));
    }

    #[test]
    fn f0_rejects_zero_amplitude_curve() {
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        let header = f0_header(2, 6, vec![0]);
        let bad = Floor0Packet::Curve {
            amplitude: 0,
            booknumber: 0,
            entries: vec![0],
        };
        let err = write_floor0_packet(&bad, &header, &[book]).unwrap_err();
        assert_eq!(err, WriteFloor0PacketError::ZeroAmplitudeCurve);
    }

    #[test]
    fn f0_rejects_amplitude_overflow() {
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        // amplitude_bits = 4 → max value 15; 16 overflows.
        let header = f0_header(2, 4, vec![0]);
        let bad = Floor0Packet::Curve {
            amplitude: 16,
            booknumber: 0,
            entries: vec![0],
        };
        let err = write_floor0_packet(&bad, &header, &[book]).unwrap_err();
        assert_eq!(
            err,
            WriteFloor0PacketError::AmplitudeOverflow {
                amplitude: 16,
                amplitude_bits: 4
            }
        );
    }

    #[test]
    fn f0_rejects_booknumber_out_of_range() {
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        let header = f0_header(2, 6, vec![0]); // number_of_books = 1
        let bad = Floor0Packet::Curve {
            amplitude: 1,
            booknumber: 1,
            entries: vec![0],
        };
        let err = write_floor0_packet(&bad, &header, &[book]).unwrap_err();
        assert_eq!(
            err,
            WriteFloor0PacketError::BooknumberOutOfRange {
                booknumber: 1,
                number_of_books: 1
            }
        );
    }

    #[test]
    fn f0_rejects_value_book_out_of_range() {
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        // book_list points position 0 at codebook index 5 (only 1 exists).
        let header = f0_header(2, 6, vec![5]);
        let bad = Floor0Packet::Curve {
            amplitude: 1,
            booknumber: 0,
            entries: vec![0],
        };
        let err = write_floor0_packet(&bad, &header, &[book]).unwrap_err();
        assert_eq!(
            err,
            WriteFloor0PacketError::ValueBookOutOfRange {
                position: 0,
                book: 5,
                codebook_count: 1
            }
        );
    }

    #[test]
    fn f0_rejects_scalar_value_book() {
        // A lookup_type 0 book cannot serve in a VQ context (§3.3).
        let scalar = VorbisCodebook {
            dimensions: 2,
            entries: 2,
            codeword_lengths: vec![1, 1],
            lookup: VqLookup::None,
        };
        let header = f0_header(2, 6, vec![0]);
        let bad = Floor0Packet::Curve {
            amplitude: 1,
            booknumber: 0,
            entries: vec![0],
        };
        let err = write_floor0_packet(&bad, &header, &[scalar]).unwrap_err();
        assert_eq!(
            err,
            WriteFloor0PacketError::ValueBookHasNoLookup {
                position: 0,
                book: 0
            }
        );
    }

    #[test]
    fn f0_rejects_entry_count_mismatch() {
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]);
        let header = f0_header(2, 6, vec![0]); // expects 1 entry
        let bad = Floor0Packet::Curve {
            amplitude: 1,
            booknumber: 0,
            entries: vec![0, 1, 0],
        };
        let err = write_floor0_packet(&bad, &header, &[book]).unwrap_err();
        assert_eq!(
            err,
            WriteFloor0PacketError::EntryCountMismatch {
                expected: 1,
                actual: 3
            }
        );
    }

    #[test]
    fn f0_rejects_entry_out_of_range() {
        let book = f0_vq_book(2, vec![1, 1], vec![0, 1, 2, 3]); // entries = 2
        let header = f0_header(2, 6, vec![0]);
        let bad = Floor0Packet::Curve {
            amplitude: 1,
            booknumber: 0,
            entries: vec![7],
        };
        let err = write_floor0_packet(&bad, &header, &[book]).unwrap_err();
        assert_eq!(
            err,
            WriteFloor0PacketError::EntryOutOfRange {
                index: 0,
                entry: 7,
                entries: 2
            }
        );
    }

    #[test]
    fn f0_rejects_unencodable_entry() {
        // Entry 1 is marked unused (length 0) → not a tree leaf.
        let book = f0_vq_book(2, vec![1, 0], vec![0, 1, 2, 3]);
        let header = f0_header(2, 6, vec![0]);
        let bad = Floor0Packet::Curve {
            amplitude: 1,
            booknumber: 0,
            entries: vec![1],
        };
        let err = write_floor0_packet(&bad, &header, &[book]).unwrap_err();
        assert!(matches!(
            err,
            WriteFloor0PacketError::UnencodableEntry {
                index: 0,
                entry: 1,
                ..
            }
        ));
    }

    #[test]
    fn f0_error_display_is_grepable() {
        let cases: Vec<WriteFloor0PacketError> = vec![
            WriteFloor0PacketError::ZeroAmplitudeBits,
            WriteFloor0PacketError::AmplitudeBitsOverflow(64),
            WriteFloor0PacketError::ZeroOrder,
            WriteFloor0PacketError::EmptyBookList,
            WriteFloor0PacketError::BookListTooLong(17),
            WriteFloor0PacketError::ZeroAmplitudeCurve,
            WriteFloor0PacketError::AmplitudeOverflow {
                amplitude: 16,
                amplitude_bits: 4,
            },
            WriteFloor0PacketError::BooknumberOutOfRange {
                booknumber: 1,
                number_of_books: 1,
            },
            WriteFloor0PacketError::ValueBookOutOfRange {
                position: 0,
                book: 5,
                codebook_count: 1,
            },
            WriteFloor0PacketError::ValueBookHasNoLookup {
                position: 0,
                book: 0,
            },
            WriteFloor0PacketError::ZeroDimensionBook {
                position: 0,
                book: 0,
            },
            WriteFloor0PacketError::EntryCountMismatch {
                expected: 1,
                actual: 3,
            },
            WriteFloor0PacketError::EntryOutOfRange {
                index: 0,
                entry: 7,
                entries: 2,
            },
            WriteFloor0PacketError::UnencodableEntry {
                index: 0,
                entry: 1,
                used_count: 1,
            },
        ];
        for c in &cases {
            let s = c.to_string();
            assert!(
                s.contains("vorbis floor0 packet (write)"),
                "Display must be grep-able: {s}"
            );
        }
    }

    // ---- §4.3.3 + §4.3.4 residue-bundle plan (plan_residue_bundles). ----

    fn submap_cfg(floor: u8, residue: u8) -> crate::setup::MappingSubmap {
        crate::setup::MappingSubmap {
            time_placeholder: 0,
            floor,
            residue,
        }
    }

    /// Mono, single submap, no coupling: one bundle holding channel 0,
    /// flags pass through untouched.
    #[test]
    fn plan_bundles_mono_single_submap() {
        let mapping = MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: Vec::new(),
            mux: Vec::new(),
            submap_configs: vec![submap_cfg(0, 0)],
        };
        // Used channel.
        let plan = plan_residue_bundles(&mapping, &[false]).unwrap();
        assert_eq!(plan.no_residue, vec![false]);
        assert_eq!(plan.bundles.len(), 1);
        assert_eq!(plan.bundles[0].submap, 0);
        assert_eq!(plan.bundles[0].channels, vec![0]);
        assert_eq!(plan.bundles[0].do_not_decode, vec![false]);

        // Unused (floor 'unused') channel — no coupling, stays unused.
        let plan = plan_residue_bundles(&mapping, &[true]).unwrap();
        assert_eq!(plan.no_residue, vec![true]);
        assert_eq!(plan.bundles[0].do_not_decode, vec![true]);
    }

    /// Stereo, single submap, coupling (mag=0, ang=1). §4.3.3: if either
    /// partner is used, both become used. So an unused angle gets pulled
    /// back in, and the single bundle holds both channels in ascending
    /// order with the propagated flags.
    #[test]
    fn plan_bundles_stereo_coupling_propagates() {
        let mapping = MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: vec![MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            }],
            mux: Vec::new(),
            submap_configs: vec![submap_cfg(0, 0)],
        };
        // Channel 0 used, channel 1 'unused' → §4.3.3 pulls channel 1
        // back in: both coded.
        let plan = plan_residue_bundles(&mapping, &[false, true]).unwrap();
        assert_eq!(plan.no_residue, vec![false, false]);
        assert_eq!(plan.bundles.len(), 1);
        assert_eq!(plan.bundles[0].channels, vec![0, 1]);
        assert_eq!(plan.bundles[0].do_not_decode, vec![false, false]);

        // Both 'unused' and no partner used → both stay unused.
        let plan = plan_residue_bundles(&mapping, &[true, true]).unwrap();
        assert_eq!(plan.no_residue, vec![true, true]);
        assert_eq!(plan.bundles[0].do_not_decode, vec![true, true]);
    }

    /// Four channels, two submaps, mux = [0, 1, 0, 1]. §4.3.4 gathers
    /// each submap's channels in ascending channel order; the bundle's
    /// `do_not_decode` mirrors each gathered channel's flag.
    #[test]
    fn plan_bundles_two_submaps_interleaved_mux() {
        let mapping = MappingHeader {
            mapping_type: 0,
            submaps: 2,
            coupling: Vec::new(),
            mux: vec![0, 1, 0, 1],
            submap_configs: vec![submap_cfg(0, 0), submap_cfg(0, 1)],
        };
        // Channels: 0 used, 1 unused, 2 unused, 3 used.
        let no_residue = [false, true, true, false];
        let plan = plan_residue_bundles(&mapping, &no_residue).unwrap();
        assert_eq!(plan.no_residue, no_residue.to_vec());
        assert_eq!(plan.bundles.len(), 2);

        // Submap 0 gathers channels 0 and 2 (ascending).
        assert_eq!(plan.bundles[0].submap, 0);
        assert_eq!(plan.bundles[0].channels, vec![0, 2]);
        assert_eq!(plan.bundles[0].do_not_decode, vec![false, true]);

        // Submap 1 gathers channels 1 and 3 (ascending).
        assert_eq!(plan.bundles[1].submap, 1);
        assert_eq!(plan.bundles[1].channels, vec![1, 3]);
        assert_eq!(plan.bundles[1].do_not_decode, vec![true, false]);
    }

    /// A submap with no channels assigned still gets an empty bundle so
    /// the bundle index lines up with `submap_configs`.
    #[test]
    fn plan_bundles_empty_submap_kept() {
        let mapping = MappingHeader {
            mapping_type: 0,
            submaps: 2,
            coupling: Vec::new(),
            // Every channel routes to submap 0; submap 1 is unused.
            mux: vec![0, 0],
            submap_configs: vec![submap_cfg(0, 0), submap_cfg(0, 1)],
        };
        let plan = plan_residue_bundles(&mapping, &[false, false]).unwrap();
        assert_eq!(plan.bundles.len(), 2);
        assert_eq!(plan.bundles[0].channels, vec![0, 1]);
        assert!(plan.bundles[1].channels.is_empty());
        assert!(plan.bundles[1].do_not_decode.is_empty());
    }

    /// The plan's per-submap `do_not_decode` slice is the exact slice
    /// the decoder's §4.3.4 step-2 loop builds. Cross-check against the
    /// decoder helper [`crate::packet::nonzero_propagate`] applied
    /// independently.
    #[test]
    fn plan_bundles_matches_independent_propagation() {
        let mapping = MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: vec![
                MappingCouplingStep {
                    magnitude_channel: 0,
                    angle_channel: 1,
                },
                MappingCouplingStep {
                    magnitude_channel: 2,
                    angle_channel: 3,
                },
            ],
            mux: Vec::new(),
            submap_configs: vec![submap_cfg(0, 0)],
        };
        let raw = [true, false, true, true];
        let plan = plan_residue_bundles(&mapping, &raw).unwrap();

        let mut expected = raw.to_vec();
        crate::packet::nonzero_propagate(&mut expected, &mapping.coupling).unwrap();
        assert_eq!(plan.no_residue, expected);
        // Single submap → all channels in one bundle, flags == expected.
        assert_eq!(plan.bundles[0].do_not_decode, expected);
        assert_eq!(plan.bundles[0].channels, vec![0, 1, 2, 3]);
    }

    #[test]
    fn plan_bundles_rejects_zero_submaps() {
        let mapping = MappingHeader {
            mapping_type: 0,
            submaps: 0,
            coupling: Vec::new(),
            mux: Vec::new(),
            submap_configs: Vec::new(),
        };
        assert_eq!(
            plan_residue_bundles(&mapping, &[false]),
            Err(PlanResidueBundlesError::ZeroSubmaps)
        );
    }

    #[test]
    fn plan_bundles_rejects_coupling_out_of_range() {
        let mapping = MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: vec![MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 5,
            }],
            mux: Vec::new(),
            submap_configs: vec![submap_cfg(0, 0)],
        };
        assert_eq!(
            plan_residue_bundles(&mapping, &[false, true]),
            Err(PlanResidueBundlesError::CouplingChannelOutOfRange {
                step: 0,
                channel: 5,
                channels: 2,
            })
        );
    }

    #[test]
    fn plan_bundles_rejects_submap_out_of_range() {
        let mapping = MappingHeader {
            mapping_type: 0,
            submaps: 2,
            coupling: Vec::new(),
            // Channel 1 routes to submap 7, which does not exist.
            mux: vec![0, 7],
            submap_configs: vec![submap_cfg(0, 0), submap_cfg(0, 1)],
        };
        assert_eq!(
            plan_residue_bundles(&mapping, &[false, false]),
            Err(PlanResidueBundlesError::SubmapOutOfRange {
                channel: 1,
                submap: 7,
                submaps: 2,
            })
        );
    }

    #[test]
    fn plan_bundles_rejects_short_mux() {
        let mapping = MappingHeader {
            mapping_type: 0,
            submaps: 2,
            coupling: Vec::new(),
            // Two channels but only one mux entry.
            mux: vec![0],
            submap_configs: vec![submap_cfg(0, 0), submap_cfg(0, 1)],
        };
        assert_eq!(
            plan_residue_bundles(&mapping, &[false, false]),
            Err(PlanResidueBundlesError::MuxTooShort {
                channel: 1,
                mux_len: 1,
            })
        );
    }

    /// Single-submap mapping ignores `mux` entirely (implicit-zero
    /// path), matching the decoder's `submap_for_channel`.
    #[test]
    fn plan_bundles_single_submap_ignores_mux() {
        let mapping = MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: Vec::new(),
            // A stray (out-of-range-looking) mux table that must be
            // ignored because submaps == 1.
            mux: vec![9, 9, 9],
            submap_configs: vec![submap_cfg(0, 0)],
        };
        let plan = plan_residue_bundles(&mapping, &[false, true, false]).unwrap();
        assert_eq!(plan.bundles.len(), 1);
        assert_eq!(plan.bundles[0].channels, vec![0, 1, 2]);
        assert_eq!(plan.bundles[0].do_not_decode, vec![false, true, false]);
    }

    /// Zero channels: every submap is empty, no propagation needed.
    #[test]
    fn plan_bundles_zero_channels() {
        let mapping = MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: Vec::new(),
            mux: Vec::new(),
            submap_configs: vec![submap_cfg(0, 0)],
        };
        let plan = plan_residue_bundles(&mapping, &[]).unwrap();
        assert!(plan.no_residue.is_empty());
        assert_eq!(plan.bundles.len(), 1);
        assert!(plan.bundles[0].channels.is_empty());
    }

    #[test]
    fn plan_bundles_error_display_grepable() {
        let cases = [
            PlanResidueBundlesError::ZeroSubmaps,
            PlanResidueBundlesError::CouplingChannelOutOfRange {
                step: 0,
                channel: 5,
                channels: 2,
            },
            PlanResidueBundlesError::SubmapOutOfRange {
                channel: 1,
                submap: 7,
                submaps: 2,
            },
            PlanResidueBundlesError::MuxTooShort {
                channel: 1,
                mux_len: 1,
            },
        ];
        for c in &cases {
            let s = c.to_string();
            assert!(
                s.contains("vorbis §4.3"),
                "Display must be grep-able by spec section: {s}"
            );
        }
    }

    // ---- §4.3 wrapping audio-packet writer (`write_audio_packet`). ----
    //
    // These tests mirror the trivial all-zero setup the decoder-side
    // tests in `audio.rs` use (a 0-partition floor 1 + a begin==end==0
    // residue that decodes to all-zero), then prove the wrapping driver
    // produces a byte-identical packet AND that the bytes round-trip
    // back through the real `decode_audio_packet_pre_imdct`.

    use crate::audio::{decode_audio_packet_pre_imdct, AudioDecoderState, AudioPacketOutcome};
    use crate::setup::{FloorHeader, MappingCouplingStep as ApMappingCouplingStep};

    fn ap_floor1_all_zero_header() -> Floor1Header {
        // multiplier 4 → range 7 → amp_bits = ilog(6) = 3.
        Floor1Header {
            partitions: 0,
            partition_class_list: Vec::new(),
            classes: Vec::new(),
            multiplier: 4,
            rangebits: 4,
            x_list: Vec::new(),
        }
    }

    fn ap_floor_type1() -> FloorHeader {
        FloorHeader {
            floor_type: 1,
            kind: FloorKind::Type1(ap_floor1_all_zero_header()),
        }
    }

    fn ap_scalar_classbook() -> VorbisCodebook {
        VorbisCodebook {
            dimensions: 1,
            entries: 1,
            codeword_lengths: vec![1],
            lookup: VqLookup::None,
        }
    }

    fn ap_zero_lookup_codebook() -> VorbisCodebook {
        VorbisCodebook {
            dimensions: 1,
            entries: 1,
            codeword_lengths: vec![1],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 0.0,
                value_bits: 1,
                sequence_p: false,
                multiplicands: vec![0],
            },
        }
    }

    fn ap_residue_zero_header() -> ResidueHeader {
        ResidueHeader {
            residue_type: 0,
            residue_begin: 0,
            residue_end: 0,
            partition_size: 1,
            classifications: 1,
            classbook: 0,
            cascade: vec![0],
            books: vec![std::array::from_fn(|_| None)],
        }
    }

    fn ap_mode_short() -> ModeHeader {
        ModeHeader {
            blockflag: false,
            windowtype: 0,
            transformtype: 0,
            mapping: 0,
        }
    }

    fn ap_mono_setup() -> VorbisSetupHeader {
        VorbisSetupHeader {
            codebooks: vec![ap_scalar_classbook(), ap_zero_lookup_codebook()],
            time_placeholders: Vec::new(),
            floors: vec![ap_floor_type1()],
            residues: vec![ap_residue_zero_header()],
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
            modes: vec![ap_mode_short()],
            framing_flag: true,
        }
    }

    fn ap_stereo_coupled_setup() -> VorbisSetupHeader {
        VorbisSetupHeader {
            codebooks: vec![ap_scalar_classbook(), ap_zero_lookup_codebook()],
            time_placeholders: Vec::new(),
            floors: vec![ap_floor_type1()],
            residues: vec![ap_residue_zero_header()],
            mappings: vec![MappingHeader {
                mapping_type: 0,
                submaps: 1,
                coupling: vec![ApMappingCouplingStep {
                    magnitude_channel: 0,
                    angle_channel: 1,
                }],
                mux: Vec::new(),
                submap_configs: vec![MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                }],
            }],
            modes: vec![ap_mode_short()],
            framing_flag: true,
        }
    }

    fn ap_header(n: usize) -> AudioPacketHeader {
        AudioPacketHeader {
            mode_number: 0,
            blockflag: false,
            n,
            previous_window_flag: false,
            next_window_flag: false,
        }
    }

    /// A floor 1 "used" body carrying two zero endpoints (the
    /// `floor1_all_zero` header has no partitions, so `floor1_y` is just
    /// the two endpoints).
    fn ap_floor1_used() -> AudioChannelFloor {
        AudioChannelFloor::Type1(Floor1Packet {
            nonzero: true,
            floor1_y: vec![0, 0],
            partition_cvals: Vec::new(),
        })
    }

    fn ap_floor1_unused() -> AudioChannelFloor {
        AudioChannelFloor::Type1(Floor1Packet {
            nonzero: false,
            floor1_y: Vec::new(),
            partition_cvals: Vec::new(),
        })
    }

    /// One submap's residue plan for the begin==end==0 residue: zero
    /// partitions → one empty `ResidueVectorPlan` per non-'do not decode'
    /// vector. Format 0, so one plan per channel (decoded ones empty).
    fn ap_residue_plans_for(do_not_decode: &[bool]) -> Vec<ResidueVectorPlan> {
        do_not_decode
            .iter()
            .map(|_| ResidueVectorPlan {
                classifications: Vec::new(),
                partition_entries: Vec::new(),
            })
            .collect()
    }

    #[test]
    fn write_audio_packet_mono_used_matches_handwritten_and_roundtrips() {
        let setup = ap_mono_setup();
        // §4.3.1 prelude (1 bit) + floor1 [nonzero]=1 (1 bit) + two
        // endpoints. multiplier 4 → range 64 → amp_bits = ilog(63) = 6
        // bits each (§7.2.3). Residue begin==end==0 → 0 bits. Total
        // = 1 + 1 + 6 + 6 = 14 bits → 2 bytes. The packet is byte-exact
        // and fully spec-explicit (unlike the EOF-padded hand-written
        // fixture in audio.rs, which relies on end-of-packet zero-fill
        // for the high endpoint bits).
        let floors = vec![ap_floor1_used()];
        // mono, single submap, channel 0 used → do_not_decode = [false].
        let residue_plans = vec![ap_residue_plans_for(&[false])];
        let bytes =
            write_audio_packet(&ap_header(64), &setup, 64, 1024, 1, &floors, &residue_plans)
                .unwrap();
        // bit 0: packet_type=0; bit 1: nonzero=1; bits 2..14: zero
        // endpoints; bits 14..16: §2.1.8 zero pad. LSB-first byte 0 holds
        // bits 0..8: 0b0000_0010 = 0x02; byte 1 holds bits 8..16 = 0x00.
        assert_eq!(bytes, vec![0x02, 0x00], "byte-exact §4.3 packet");

        // Round-trip through the real decoder.
        let state = AudioDecoderState::new(&setup).unwrap();
        let mut r = oxideav_core::bits::BitReaderLsb::new(&bytes);
        let outcome = decode_audio_packet_pre_imdct(&mut r, &setup, &state, 1, 64, 1024).unwrap();
        match outcome {
            AudioPacketOutcome::PreImdct { n, spectra, .. } => {
                assert_eq!(n, 64);
                assert_eq!(spectra.len(), 1);
                assert_eq!(spectra[0].len(), 32);
                for &s in &spectra[0] {
                    assert_eq!(s, 0.0);
                }
            }
            other => panic!("expected PreImdct, got {other:?}"),
        }
    }

    #[test]
    fn write_audio_packet_mono_unused_matches_handwritten_and_roundtrips() {
        let setup = ap_mono_setup();
        // prelude (1) + floor1 [nonzero]=0 = 2 bits → one byte 0x00.
        let floors = vec![ap_floor1_unused()];
        // Channel unused → do_not_decode = [true].
        let residue_plans = vec![ap_residue_plans_for(&[true])];
        let bytes =
            write_audio_packet(&ap_header(64), &setup, 64, 1024, 1, &floors, &residue_plans)
                .unwrap();
        assert_eq!(bytes, vec![0x00]);

        let state = AudioDecoderState::new(&setup).unwrap();
        let mut r = oxideav_core::bits::BitReaderLsb::new(&bytes);
        let outcome = decode_audio_packet_pre_imdct(&mut r, &setup, &state, 1, 64, 1024).unwrap();
        match outcome {
            AudioPacketOutcome::PreImdct { spectra, .. } => {
                assert_eq!(spectra.len(), 1);
                for &s in &spectra[0] {
                    assert_eq!(s, 0.0);
                }
            }
            other => panic!("expected PreImdct, got {other:?}"),
        }
    }

    #[test]
    fn write_audio_packet_stereo_coupled_roundtrips() {
        let setup = ap_stereo_coupled_setup();
        // Both channels used → §4.3.3 propagate keeps both used.
        let floors = vec![ap_floor1_used(), ap_floor1_used()];
        let residue_plans = vec![ap_residue_plans_for(&[false, false])];
        let bytes =
            write_audio_packet(&ap_header(64), &setup, 64, 1024, 2, &floors, &residue_plans)
                .unwrap();
        // prelude(1) + ch0(1+6+6) + ch1(1+6+6) = 27 bits → 4 bytes
        // (amp_bits = 6 per the §7.2.3 range table for multiplier 4).
        assert_eq!(bytes.len(), 4);

        let state = AudioDecoderState::new(&setup).unwrap();
        let mut r = oxideav_core::bits::BitReaderLsb::new(&bytes);
        let outcome = decode_audio_packet_pre_imdct(&mut r, &setup, &state, 2, 64, 1024).unwrap();
        match outcome {
            AudioPacketOutcome::PreImdct { spectra, .. } => {
                assert_eq!(spectra.len(), 2);
                for ch in &spectra {
                    assert_eq!(ch.len(), 32);
                    for &s in ch {
                        assert_eq!(s, 0.0);
                    }
                }
            }
            other => panic!("expected PreImdct, got {other:?}"),
        }
    }

    #[test]
    fn write_audio_packet_stereo_coupled_one_unused_floor_still_codes_both() {
        // §4.3.3: channel 1's floor is 'unused' but the coupling step
        // (0,1) pulls it back in, so the residue bundle's do_not_decode
        // is [false, false] — both channels coded. We assert the bundle
        // plan the driver derives matches that, and the packet
        // round-trips.
        let setup = ap_stereo_coupled_setup();
        let floors = vec![ap_floor1_used(), ap_floor1_unused()];
        // Both still coded after propagation.
        let residue_plans = vec![ap_residue_plans_for(&[false, false])];
        let bytes =
            write_audio_packet(&ap_header(64), &setup, 64, 1024, 2, &floors, &residue_plans)
                .unwrap();

        // Cross-check: the propagated no_residue is [false, false].
        let mapping = &setup.mappings[0];
        let no_residue = vec![false, true]; // ch0 used, ch1 floor unused
        let plan = plan_residue_bundles(mapping, &no_residue).unwrap();
        assert_eq!(plan.no_residue, vec![false, false]);
        assert_eq!(plan.bundles[0].do_not_decode, vec![false, false]);

        let state = AudioDecoderState::new(&setup).unwrap();
        let mut r = oxideav_core::bits::BitReaderLsb::new(&bytes);
        let outcome = decode_audio_packet_pre_imdct(&mut r, &setup, &state, 2, 64, 1024).unwrap();
        assert!(matches!(outcome, AudioPacketOutcome::PreImdct { .. }));
    }

    #[test]
    fn write_audio_packet_rejects_floor_count_mismatch() {
        let setup = ap_mono_setup();
        let floors = vec![ap_floor1_used(), ap_floor1_used()]; // 2 != 1
        let residue_plans = vec![ap_residue_plans_for(&[false])];
        match write_audio_packet(&ap_header(64), &setup, 64, 1024, 1, &floors, &residue_plans) {
            Err(WriteAudioPacketError::FloorCountMismatch {
                audio_channels,
                floors,
            }) => {
                assert_eq!(audio_channels, 1);
                assert_eq!(floors, 2);
            }
            other => panic!("expected FloorCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn write_audio_packet_rejects_zero_channels() {
        let setup = ap_mono_setup();
        match write_audio_packet(&ap_header(64), &setup, 64, 1024, 0, &[], &[]) {
            Err(WriteAudioPacketError::ZeroAudioChannels) => {}
            other => panic!("expected ZeroAudioChannels, got {other:?}"),
        }
    }

    #[test]
    fn write_audio_packet_rejects_floor_type_mismatch() {
        let setup = ap_mono_setup(); // floor 0 in setup is Type1.
                                     // Supply a Type0 body for a channel whose header is Type1.
        let floors = vec![AudioChannelFloor::Type0(Floor0Packet::Unused)];
        let residue_plans = vec![ap_residue_plans_for(&[true])];
        match write_audio_packet(&ap_header(64), &setup, 64, 1024, 1, &floors, &residue_plans) {
            Err(WriteAudioPacketError::FloorTypeMismatch {
                channel,
                header_type,
                packet_type,
            }) => {
                assert_eq!(channel, 0);
                assert_eq!(header_type, 1);
                assert_eq!(packet_type, 0);
            }
            other => panic!("expected FloorTypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn write_audio_packet_rejects_residue_plan_count_mismatch() {
        let setup = ap_mono_setup(); // submaps == 1
        let floors = vec![ap_floor1_used()];
        let residue_plans: Vec<Vec<ResidueVectorPlan>> = Vec::new(); // 0 != 1
        match write_audio_packet(&ap_header(64), &setup, 64, 1024, 1, &floors, &residue_plans) {
            Err(WriteAudioPacketError::ResiduePlanCountMismatch { submaps, plans }) => {
                assert_eq!(submaps, 1);
                assert_eq!(plans, 0);
            }
            other => panic!("expected ResiduePlanCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn write_audio_packet_rejects_bad_mode_mapping() {
        let mut setup = ap_mono_setup();
        // Point the only mode at a nonexistent mapping.
        setup.modes[0].mapping = 5;
        let floors = vec![ap_floor1_used()];
        let residue_plans = vec![ap_residue_plans_for(&[false])];
        match write_audio_packet(&ap_header(64), &setup, 64, 1024, 1, &floors, &residue_plans) {
            Err(WriteAudioPacketError::BadModeMapping {
                mode_number,
                mapping,
                mapping_count,
            }) => {
                assert_eq!(mode_number, 0);
                assert_eq!(mapping, 5);
                assert_eq!(mapping_count, 1);
            }
            other => panic!("expected BadModeMapping, got {other:?}"),
        }
    }

    #[test]
    fn write_audio_packet_into_writer_emits_no_bits_on_error() {
        // A validation error must leave the caller's writer untouched.
        let setup = ap_mono_setup();
        let floors = vec![ap_floor1_used(), ap_floor1_used()]; // count mismatch
        let residue_plans = vec![ap_residue_plans_for(&[false])];
        let mut w = BitWriterLsb::new();
        w.write_u32(0b101, 3); // three sentinel bits already in the writer
        let before = w.bit_position();
        let res = write_audio_packet_into_writer(
            &ap_header(64),
            &setup,
            64,
            1024,
            1,
            &floors,
            &residue_plans,
            &mut w,
        );
        assert!(res.is_err());
        assert_eq!(w.bit_position(), before, "no bits emitted on error");
    }

    #[test]
    fn write_audio_packet_error_display_is_grepable() {
        let cases: Vec<WriteAudioPacketError> = vec![
            WriteAudioPacketError::ZeroAudioChannels,
            WriteAudioPacketError::FloorCountMismatch {
                audio_channels: 1,
                floors: 2,
            },
            WriteAudioPacketError::BadModeMapping {
                mode_number: 0,
                mapping: 5,
                mapping_count: 1,
            },
            WriteAudioPacketError::FloorTypeMismatch {
                channel: 0,
                header_type: 1,
                packet_type: 0,
            },
            WriteAudioPacketError::ResiduePlanCountMismatch {
                submaps: 1,
                plans: 0,
            },
        ];
        for c in &cases {
            let s = c.to_string();
            assert!(s.contains("vorbis §4.3"), "grep-able by spec section: {s}");
        }
    }
}

//! Vorbis I setup-header outer walker (Vorbis I §4.2.4).
//!
//! The setup header is the third of three required Vorbis header packets
//! (§4.2.1). It carries the codec configuration the audio packets need:
//! the list of codebooks (§3.2.1), placeholder time-domain transforms
//! (§4.2.4 "Time domain transforms"), the floor configurations (§6, §7),
//! the residue configurations (§8), the mappings (§4.2.4 "Mappings"),
//! and the modes (§4.2.4 "Modes"). The packet ends with a 1-bit framing
//! flag (§4.2.4 "Modes" step 3).
//!
//! ## Round-6 scope
//!
//! This module parses the **entire** setup header — the six sub-lists
//! plus the trailing framing flag:
//!
//! 1. **Codebooks** (`u8 + 1` codebooks, each per §3.2.1 — delegated to
//!    [`crate::codebook::parse_codebook`]).
//! 2. **Time-domain transform placeholders** (`u6 + 1` 16-bit values,
//!    each spec-mandated to equal zero per §4.2.4 step 2).
//! 3. **Floors** (`u6 + 1` floors, each prefixed by a 16-bit
//!    `floor_type`; for floor type 1 this round reads the structural
//!    fields of §7.2.2 — partitions, partition classes, multiplier,
//!    rangebits, x-list — but **does not** evaluate the per-packet
//!    floor curve, which lives in audio-packet decode and is a later
//!    round). For floor type 0 (LSP), the §6.2.1 setup fields are
//!    read; libvorbis encoders do not produce floor 0 streams but the
//!    spec mandates the parser handle them.
//! 4. **Residues** (`u6 + 1` residues, each prefixed by a 16-bit
//!    `residue_type` ∈ {0, 1, 2}; the structural header per §8.6.1 —
//!    begin/end/partition_size/classifications/classbook + the
//!    cascade bitmap + per-classification book list — is read).
//! 5. **Mappings** (`u6 + 1` mappings, each prefixed by a 16-bit
//!    `mapping_type` which must be 0; the §4.2.4 "Mappings" structural
//!    fields are read — optional submap count, optional coupling steps,
//!    2-bit reserved field, optional `mux[ch]` table, then per-submap
//!    placeholder + floor index + residue index).
//! 6. **Modes** (`u6 + 1` modes; each `blockflag + windowtype +
//!    transformtype + mapping` per §4.2.4 "Modes"; the spec requires
//!    `windowtype == 0` and `transformtype == 0`).
//! 7. **Framing flag** (1 bit, must be set per §4.2.4 "Modes" step 3).
//!
//! Because the mapping decode requires `ilog(audio_channels - 1)` bits
//! for each magnitude / angle channel number (§4.2.4 "Mappings" step
//! 2c.ii.A.) and the channel count is carried in a separately-parsed
//! identification header, [`parse_setup_header_body`] and
//! [`parse_setup_header`] take an `audio_channels: u8` parameter that
//! callers must thread through from
//! [`crate::identification::VorbisIdentificationHeader::audio_channels`].
//! The same channel count is also needed for the per-channel `mux[ch]`
//! reads when `submaps > 1`.
//!
//! The trace-doc §2.4 shape — `codebook_count`, `time_count`,
//! `floor_count`, `residue_count`, `mapping_count`, `mode_count`,
//! `framing_flag` — matches this layout exactly, and round-6 emits all
//! seven.
//!
//! ## Common-header prefix
//!
//! The setup-header packet starts with a 7-byte common header
//! (`packet_type = 0x05`, then ASCII `"vorbis"` per §4.2.1). This module
//! exposes both [`parse_setup_header`] (which validates the common
//! header from a `&[u8]` packet body) and [`parse_setup_header_body`]
//! (which expects the caller to have already advanced past the common
//! header and operates directly on a [`BitReaderLsb`] positioned at the
//! `codebook_count_minus_1` byte). The body form is what the
//! eventual audio-packet decoder will consume after Ogg framing strips
//! the per-page lacing.

use core::fmt;

use oxideav_core::bits::BitReaderLsb;

use crate::codebook::{self, ilog, parse_codebook, VorbisCodebook};

/// Common-header packet-type byte for the setup header (Vorbis I §4.2.1).
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub const SETUP_PACKET_TYPE: u8 = 0x05;

/// The six magic bytes that follow the packet-type byte in every Vorbis
/// header packet (Vorbis I §4.2.1).
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub const SETUP_PACKET_MAGIC: [u8; 6] = *b"vorbis";

/// Parsed structural shell of a Vorbis I setup header (Vorbis I §4.2.4)
/// for the **round-6 scope**: every sub-list of the setup header is
/// populated — codebooks, time-domain placeholders, floor headers,
/// residue headers, mapping configurations, mode configurations — plus
/// the trailing framing flag.
#[derive(Debug, Clone, PartialEq)]
pub struct VorbisSetupHeader {
    /// All `vorbis_codebook_count` codebook configurations in stream
    /// order (§4.2.4 "Codebooks"; per-codebook structure per §3.2.1).
    pub codebooks: Vec<VorbisCodebook>,
    /// Per §4.2.4 "Time domain transforms", the `vorbis_time_count`
    /// 16-bit placeholder values that the encoder writes — each
    /// spec-mandated to equal zero. Stored as `u16` so callers can
    /// inspect the raw value if they want to relax the zero check;
    /// [`parse_setup_header_body`] rejects any nonzero value at parse
    /// time per §4.2.4 step 2 ("If any value is nonzero, this is an
    /// error condition and the stream is undecodable").
    pub time_placeholders: Vec<u16>,
    /// Floor configurations in stream order (§4.2.4 "Floors"; type 0
    /// per §6.2.1, type 1 per §7.2.2). Round-5 reads the structural
    /// header only — no per-packet curve decode.
    pub floors: Vec<FloorHeader>,
    /// Residue configurations in stream order (§4.2.4 "Residues",
    /// per-residue structural shell per §8.6.1; all three residue
    /// types share the same header layout).
    pub residues: Vec<ResidueHeader>,
    /// Mapping configurations in stream order (§4.2.4 "Mappings"; only
    /// `mapping_type = 0` is defined in Vorbis I, so each entry's
    /// `mapping_type` is always 0).
    pub mappings: Vec<MappingHeader>,
    /// Mode configurations in stream order (§4.2.4 "Modes"; `blockflag
    /// + windowtype + transformtype + mapping`).
    pub modes: Vec<ModeHeader>,
    /// The 1-bit framing flag at the tail of the setup-header packet
    /// (§4.2.4 "Modes" step 3). Always `true` for a well-formed packet;
    /// `false` triggers [`ParseError::BadFramingFlag`] at parse time
    /// (this field is therefore informational only, but is retained for
    /// callers wanting to confirm parse-time invariants downstream).
    pub framing_flag: bool,
}

/// A parsed mapping configuration (§4.2.4 "Mappings").
///
/// Vorbis I defines exactly one mapping type, `0`, so the 16-bit
/// `mapping_type` field is always `0` after a successful parse. Any
/// nonzero value is rejected per step 2b of the spec algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub struct MappingHeader {
    /// The 16-bit `mapping_type` field that the encoder wrote. Always
    /// `0` for a well-formed Vorbis I stream.
    pub mapping_type: u16,
    /// `vorbis_mapping_submaps`. `1` when the `submaps_flag` bit is
    /// unset, otherwise `read 4 bits + 1` (so `1..=16`).
    pub submaps: u8,
    /// Per-coupling-step `(magnitude_channel, angle_channel)` pairs
    /// from §4.2.4 "Mappings" step 2c.ii (the square-polar coupling
    /// branch). Empty when `square_polar_flag` is unset.
    pub coupling: Vec<MappingCouplingStep>,
    /// `vorbis_mapping_mux[ch]` for each of the stream's `audio_channels`
    /// channels, identifying which submap that channel belongs to. Only
    /// read when [`Self::submaps`] > 1; otherwise empty (every channel
    /// implicitly maps to submap 0).
    pub mux: Vec<u8>,
    /// Per-submap configuration (length [`Self::submaps`]): the unused
    /// 8-bit time-config placeholder + the 8-bit floor index + the
    /// 8-bit residue index.
    pub submap_configs: Vec<MappingSubmap>,
}

/// One coupling step of a mapping configuration (§4.2.4 "Mappings"
/// step 2c.ii.A).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub struct MappingCouplingStep {
    /// `vorbis_mapping_magnitude[j]` — channel index used as the
    /// magnitude channel for coupling step `j`.
    pub magnitude_channel: u8,
    /// `vorbis_mapping_angle[j]` — channel index used as the angle
    /// channel for coupling step `j`.
    pub angle_channel: u8,
}

/// Per-submap fields of a mapping configuration (§4.2.4 "Mappings"
/// step 2c.v).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub struct MappingSubmap {
    /// `vorbis_mapping_submap_time_placeholder[j]` — 8 bits the spec
    /// instructs the decoder to "read and discard" (the unused time
    /// configuration placeholder, step 2c.v.A). Stored verbatim for
    /// callers that want round-trip inspection; not otherwise used.
    pub time_placeholder: u8,
    /// `vorbis_mapping_submap_floor[j]` — index into the setup
    /// header's [`VorbisSetupHeader::floors`] list. Range-checked at
    /// parse time to be < `floors.len()`.
    pub floor: u8,
    /// `vorbis_mapping_submap_residue[j]` — index into the setup
    /// header's [`VorbisSetupHeader::residues`] list. Range-checked at
    /// parse time to be < `residues.len()`.
    pub residue: u8,
}

/// A parsed mode configuration (§4.2.4 "Modes").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub struct ModeHeader {
    /// `vorbis_mode_blockflag` — 1 bit. `false` selects `blocksize_0`
    /// (short block), `true` selects `blocksize_1` (long block).
    pub blockflag: bool,
    /// `vorbis_mode_windowtype` — 16 bits. Must be 0 per §4.2.4
    /// "Modes" step 2e; any nonzero value is rejected with
    /// [`ParseError::NonZeroModeWindowType`].
    pub windowtype: u16,
    /// `vorbis_mode_transformtype` — 16 bits. Must be 0 per §4.2.4
    /// "Modes" step 2e; any nonzero value is rejected with
    /// [`ParseError::NonZeroModeTransformType`].
    pub transformtype: u16,
    /// `vorbis_mode_mapping` — 8 bits. Index into the setup header's
    /// [`VorbisSetupHeader::mappings`] list. Range-checked at parse
    /// time to be < `mappings.len()`.
    pub mapping: u8,
}

/// A parsed floor header (§6.2.1 for type 0, §7.2.2 for type 1).
///
/// Floor type is recorded both as a 16-bit `floor_type` field (matching
/// the trace-doc field shape) and as the discriminant on the enum's
/// `kind` payload. A floor type strictly greater than 1 is rejected at
/// parse time per §4.2.4 step 2d.
#[derive(Debug, Clone, PartialEq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub struct FloorHeader {
    /// The 16-bit `floor_type` field that the encoder wrote.
    pub floor_type: u16,
    /// Per-type structural payload.
    pub kind: FloorKind,
}

/// Per-type floor structural fields.
#[derive(Debug, Clone, PartialEq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub enum FloorKind {
    /// `floor_type = 0` — line-spectrum-pair representation (§6.2.1).
    /// libvorbis does not emit this type; a conformant parser must
    /// still accept it.
    Type0(Floor0Header),
    /// `floor_type = 1` — piecewise-linear representation (§7.2.2).
    /// The dominant format; every fixture in
    /// `docs/audio/vorbis/fixtures/` exercises type 1.
    Type1(Floor1Header),
}

/// Floor 0 setup header fields per §6.2.1.
#[derive(Debug, Clone, PartialEq, Eq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub struct Floor0Header {
    /// `floor0_order` — LSP filter order (8-bit unsigned).
    pub order: u8,
    /// `floor0_rate` — sample-rate hint for the floor curve (16-bit).
    pub rate: u16,
    /// `floor0_bark_map_size` — size of the integer map of frequency
    /// bins to Bark-scale bins (16-bit).
    pub bark_map_size: u16,
    /// `floor0_amplitude_bits` — width in bits of the per-packet
    /// amplitude field (6 bits in the header).
    pub amplitude_bits: u8,
    /// `floor0_amplitude_offset` — offset added to amplitude post-decode
    /// (8-bit unsigned).
    pub amplitude_offset: u8,
    /// `floor0_book_list` — list of codebook indices the floor 0
    /// per-packet decoder may use (each 8 bits). Length is the parsed
    /// `floor0_number_of_books`, which is `read 4 bits + 1` per §6.2.1.
    pub book_list: Vec<u8>,
}

/// Floor 1 setup header fields per §7.2.2.
#[derive(Debug, Clone, PartialEq, Eq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub struct Floor1Header {
    /// `floor1_partitions` — number of partitions in the x-coordinate
    /// list (5 bits in the header).
    pub partitions: u8,
    /// `floor1_partition_class_list` — partition-class index for each
    /// partition (each 4 bits). Length is exactly [`Self::partitions`].
    pub partition_class_list: Vec<u8>,
    /// One entry per partition class (`0 ..= maximum_class`).
    pub classes: Vec<Floor1Class>,
    /// `floor1_multiplier` (2 bits in the header `+ 1`, so 1..=4).
    pub multiplier: u8,
    /// `rangebits` (4 bits) — bit width of each explicit x-list element.
    pub rangebits: u8,
    /// `floor1_X_list` — the explicit per-partition x-coordinates read
    /// from the header (excluding the implicit endpoints `0` and
    /// `2^rangebits` which the curve decoder injects). Length is
    /// `sum_over_partitions(class.dimensions)`.
    pub x_list: Vec<u32>,
}

/// One partition-class entry of a floor 1 header (§7.2.2 steps 7..12).
#[derive(Debug, Clone, PartialEq, Eq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub struct Floor1Class {
    /// `floor1_class_dimensions[i]` — number of Y values this class
    /// encodes at once (read 3 bits + 1, so 1..=8).
    pub dimensions: u8,
    /// `floor1_class_subclasses[i]` — power-of-two count of alternate
    /// codebooks (`2^subclasses` possible alternatives, read 2 bits).
    pub subclasses: u8,
    /// `floor1_class_masterbooks[i]` — codebook used to select among
    /// the subclass books; only present (8 bits) when `subclasses > 0`.
    pub masterbook: Option<u8>,
    /// `floor1_subclass_books[i][j]` for `j` in `0..2^subclasses` —
    /// the per-subclass codebooks for this class. Each is stored as
    /// `read 8 bits - 1`, so the sentinel `-1` (encoded `0`) means
    /// "this subclass has no codebook"; stored as [`Option`] for
    /// clarity. Length is exactly `1 << subclasses`.
    pub subclass_books: Vec<Option<u8>>,
}

/// A parsed residue header (§8.6.1 — common across all three residue
/// types).
#[derive(Debug, Clone, PartialEq, Eq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub struct ResidueHeader {
    /// The 16-bit `residue_type` field that the encoder wrote. Must be
    /// in {0, 1, 2} per §4.2.4 step 2c.
    pub residue_type: u16,
    /// `residue_begin` (24 bits).
    pub residue_begin: u32,
    /// `residue_end` (24 bits).
    pub residue_end: u32,
    /// `residue_partition_size` = read 24 bits + 1.
    pub partition_size: u32,
    /// `residue_classifications` = read 6 bits + 1 (so 1..=64).
    pub classifications: u8,
    /// `residue_classbook` = read 8 bits (index into the codebook
    /// table).
    pub classbook: u8,
    /// `residue_cascade[i]` per-classification cascade bitmap
    /// (8 bits each, conceptually): bit `j` set means stage `j`'s
    /// codebook is present in [`Self::books`] for that classification.
    /// Length is exactly [`Self::classifications`].
    pub cascade: Vec<u8>,
    /// `residue_books[i][j]` — the codebook used at stage `j` for
    /// classification `i`. Outer length is [`Self::classifications`];
    /// inner length is always 8. `None` indicates the corresponding
    /// cascade bit was unset (no codebook for that stage).
    pub books: Vec<[Option<u8>; 8]>,
}

/// Errors that may arise while walking the round-5 portion of a Vorbis
/// I setup header.
#[derive(Debug, Clone, PartialEq, Eq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub enum ParseError {
    /// The supplied packet was shorter than the 7-byte common header.
    PacketTooShort(usize),
    /// The 1-byte common-header packet type did not equal
    /// [`SETUP_PACKET_TYPE`].
    WrongPacketType(u8),
    /// The six bytes following the packet type were not the ASCII
    /// magic string `"vorbis"`.
    BadMagic,
    /// A nested codebook header (§3.2.1) failed to parse.
    Codebook {
        /// Codebook index in `0 .. vorbis_codebook_count`.
        index: usize,
        /// The inner parse error from [`crate::codebook::parse_codebook`].
        source: codebook::ParseError,
    },
    /// A time-domain transform placeholder (§4.2.4 step 2) was nonzero;
    /// the spec explicitly rejects this.
    NonZeroTimePlaceholder {
        /// Time-placeholder index in `0 .. vorbis_time_count`.
        index: usize,
        /// The (nonzero) 16-bit value that was actually read.
        value: u16,
    },
    /// A floor type strictly greater than 1 was read (§4.2.4 step 2d).
    UnsupportedFloorType {
        /// Floor index in `0 .. vorbis_floor_count`.
        index: usize,
        /// The 16-bit floor type that the encoder wrote.
        floor_type: u16,
    },
    /// A residue type strictly greater than 2 was read (§4.2.4 step 2c).
    UnsupportedResidueType {
        /// Residue index in `0 .. vorbis_residue_count`.
        index: usize,
        /// The 16-bit residue type that the encoder wrote.
        residue_type: u16,
    },
    /// A mapping type other than 0 was read (§4.2.4 "Mappings" step 2b:
    /// "If the mapping type is nonzero, the stream is undecodable").
    UnsupportedMappingType {
        /// Mapping index in `0 .. vorbis_mapping_count`.
        index: usize,
        /// The 16-bit mapping type that the encoder wrote.
        mapping_type: u16,
    },
    /// A coupling step referenced a magnitude- or angle-channel index
    /// outside `0..audio_channels`, or set the two equal (§4.2.4
    /// "Mappings" step 2c.ii — "If for any coupling step the angle
    /// channel number equals the magnitude channel number, the
    /// magnitude channel number is greater than [audio_channels]-1, or
    /// the angle channel is greater than [audio_channels]-1, the
    /// stream is undecodable").
    BadCouplingChannels {
        /// Mapping index in `0 .. vorbis_mapping_count`.
        mapping_index: usize,
        /// Coupling-step index in `0 .. vorbis_mapping_coupling_steps`.
        step_index: usize,
        /// The magnitude channel index as read.
        magnitude_channel: u8,
        /// The angle channel index as read.
        angle_channel: u8,
        /// The stream's `audio_channels` value (from the identification
        /// header).
        audio_channels: u8,
    },
    /// The 2-bit mapping reserved field (§4.2.4 "Mappings" step 2c.iii)
    /// was nonzero. The spec marks any nonzero value as undecodable.
    NonZeroMappingReserved {
        /// Mapping index in `0 .. vorbis_mapping_count`.
        index: usize,
        /// The 2-bit value that was read.
        value: u8,
    },
    /// A `mux[ch]` value referenced a submap outside `0..submaps`
    /// (§4.2.4 "Mappings" step 2c.iv.B).
    BadMuxValue {
        /// Mapping index in `0 .. vorbis_mapping_count`.
        mapping_index: usize,
        /// Channel index in `0 .. audio_channels`.
        channel_index: usize,
        /// The `mux[ch]` value as read.
        mux: u8,
        /// The mapping's `vorbis_mapping_submaps`.
        submaps: u8,
    },
    /// A submap floor index was outside `0..floors.len()` (§4.2.4
    /// "Mappings" step 2c.v.C).
    BadSubmapFloor {
        /// Mapping index in `0 .. vorbis_mapping_count`.
        mapping_index: usize,
        /// Submap index in `0 .. submaps`.
        submap_index: usize,
        /// The floor index as read.
        floor: u8,
        /// The setup header's `vorbis_floor_count`.
        floor_count: usize,
    },
    /// A submap residue index was outside `0..residues.len()` (§4.2.4
    /// "Mappings" step 2c.v.E).
    BadSubmapResidue {
        /// Mapping index in `0 .. vorbis_mapping_count`.
        mapping_index: usize,
        /// Submap index in `0 .. submaps`.
        submap_index: usize,
        /// The residue index as read.
        residue: u8,
        /// The setup header's `vorbis_residue_count`.
        residue_count: usize,
    },
    /// A mode's `vorbis_mode_windowtype` was not 0 (§4.2.4 "Modes"
    /// step 2e: "zero is the only legal value in Vorbis I for
    /// [vorbis_mode_windowtype]").
    NonZeroModeWindowType {
        /// Mode index in `0 .. vorbis_mode_count`.
        index: usize,
        /// The 16-bit window type that the encoder wrote.
        windowtype: u16,
    },
    /// A mode's `vorbis_mode_transformtype` was not 0 (§4.2.4 "Modes"
    /// step 2e: "zero is the only legal value in Vorbis I for
    /// [vorbis_mode_transformtype]").
    NonZeroModeTransformType {
        /// Mode index in `0 .. vorbis_mode_count`.
        index: usize,
        /// The 16-bit transform type that the encoder wrote.
        transformtype: u16,
    },
    /// A mode's `vorbis_mode_mapping` was outside `0..mappings.len()`
    /// (§4.2.4 "Modes" step 2e: "vorbis_mode_mapping must not be
    /// greater than the highest number mapping in use").
    BadModeMapping {
        /// Mode index in `0 .. vorbis_mode_count`.
        index: usize,
        /// The mapping index as read.
        mapping: u8,
        /// The setup header's `vorbis_mapping_count`.
        mapping_count: usize,
    },
    /// The trailing 1-bit framing flag was 0 (§4.2.4 "Modes" step 3:
    /// "If unset, a framing error occurred and the stream is not
    /// decodable").
    BadFramingFlag,
    /// `audio_channels = 0` was passed to [`parse_setup_header`] or
    /// [`parse_setup_header_body`]. The identification header (§4.2.2)
    /// already guarantees `audio_channels > 0`; this is a caller bug
    /// surfaced here as a structured error rather than a panic.
    ZeroAudioChannels,
    /// The bitstream ran out of bits mid-setup-header. Per §4.2.4
    /// "An end-of-packet condition during setup header decode renders
    /// the stream undecodable" (echoed by §3.2.1, §6.2.1, §7.2.2, and
    /// §8.6.1), so this is a fatal parse failure.
    UnexpectedEndOfPacket,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::PacketTooShort(n) => write!(
                f,
                "vorbis setup header: packet too short ({n} bytes; need at least 7 for the common header per §4.2.1)"
            ),
            ParseError::WrongPacketType(t) => write!(
                f,
                "vorbis setup header: wrong packet_type byte 0x{t:02x} (expected 0x05 per §4.2.1)"
            ),
            ParseError::BadMagic => write!(
                f,
                "vorbis setup header: missing 'vorbis' magic per §4.2.1"
            ),
            ParseError::Codebook { index, source } => write!(
                f,
                "vorbis setup header: codebook[{index}] parse failed: {source}"
            ),
            ParseError::NonZeroTimePlaceholder { index, value } => write!(
                f,
                "vorbis setup header: time_placeholders[{index}]={value} (must be 0 per §4.2.4 step 2)"
            ),
            ParseError::UnsupportedFloorType { index, floor_type } => write!(
                f,
                "vorbis setup header: floors[{index}].floor_type={floor_type} (must be 0 or 1 per §4.2.4 step 2d)"
            ),
            ParseError::UnsupportedResidueType {
                index,
                residue_type,
            } => write!(
                f,
                "vorbis setup header: residues[{index}].residue_type={residue_type} (must be 0, 1 or 2 per §4.2.4 step 2c)"
            ),
            ParseError::UnsupportedMappingType {
                index,
                mapping_type,
            } => write!(
                f,
                "vorbis setup header: mappings[{index}].mapping_type={mapping_type} (must be 0 per §4.2.4 step 2b)"
            ),
            ParseError::BadCouplingChannels {
                mapping_index,
                step_index,
                magnitude_channel,
                angle_channel,
                audio_channels,
            } => write!(
                f,
                "vorbis setup header: mappings[{mapping_index}] coupling step {step_index}: \
                 magnitude={magnitude_channel} angle={angle_channel} audio_channels={audio_channels} \
                 (per §4.2.4 step 2c.ii: indices must differ and each be < audio_channels)"
            ),
            ParseError::NonZeroMappingReserved { index, value } => write!(
                f,
                "vorbis setup header: mappings[{index}] reserved field = {value} (must be 0 per §4.2.4 step 2c.iii)"
            ),
            ParseError::BadMuxValue {
                mapping_index,
                channel_index,
                mux,
                submaps,
            } => write!(
                f,
                "vorbis setup header: mappings[{mapping_index}] mux[{channel_index}]={mux} (must be < submaps={submaps} per §4.2.4 step 2c.iv.B)"
            ),
            ParseError::BadSubmapFloor {
                mapping_index,
                submap_index,
                floor,
                floor_count,
            } => write!(
                f,
                "vorbis setup header: mappings[{mapping_index}] submap {submap_index} floor={floor} (must be < floor_count={floor_count} per §4.2.4 step 2c.v.C)"
            ),
            ParseError::BadSubmapResidue {
                mapping_index,
                submap_index,
                residue,
                residue_count,
            } => write!(
                f,
                "vorbis setup header: mappings[{mapping_index}] submap {submap_index} residue={residue} (must be < residue_count={residue_count} per §4.2.4 step 2c.v.E)"
            ),
            ParseError::NonZeroModeWindowType { index, windowtype } => write!(
                f,
                "vorbis setup header: modes[{index}].windowtype={windowtype} (must be 0 per §4.2.4 \"Modes\" step 2e)"
            ),
            ParseError::NonZeroModeTransformType {
                index,
                transformtype,
            } => write!(
                f,
                "vorbis setup header: modes[{index}].transformtype={transformtype} (must be 0 per §4.2.4 \"Modes\" step 2e)"
            ),
            ParseError::BadModeMapping {
                index,
                mapping,
                mapping_count,
            } => write!(
                f,
                "vorbis setup header: modes[{index}].mapping={mapping} (must be < mapping_count={mapping_count} per §4.2.4 \"Modes\" step 2e)"
            ),
            ParseError::BadFramingFlag => write!(
                f,
                "vorbis setup header: trailing framing flag was unset (§4.2.4 \"Modes\" step 3: stream not decodable)"
            ),
            ParseError::ZeroAudioChannels => write!(
                f,
                "vorbis setup header: caller passed audio_channels=0; identification header guarantees > 0"
            ),
            ParseError::UnexpectedEndOfPacket => write!(
                f,
                "vorbis setup header: end-of-packet mid-setup (§4.2.4: fatal)"
            ),
        }
    }
}

impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ParseError::Codebook { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Parse the full Vorbis I setup-header packet (Vorbis I §4.2.4).
///
/// The supplied `packet` must contain the full setup-header packet
/// (common header + bit-packed body). `audio_channels` must equal the
/// `audio_channels` field of the identification header parsed earlier
/// in the same logical stream (Vorbis I §4.2.2) — it is needed for the
/// `ilog(audio_channels - 1)`-bit magnitude/angle channel reads in
/// §4.2.4 "Mappings" and for the per-channel `mux[ch]` reads when a
/// mapping declares `submaps > 1`.
///
/// The function validates the 7-byte common header per §4.2.1 then
/// delegates to [`parse_setup_header_body`] for the bit-packed body.
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub fn parse_setup_header(
    packet: &[u8],
    audio_channels: u8,
) -> Result<VorbisSetupHeader, ParseError> {
    // Common header (§4.2.1): 0x05 + "vorbis".
    if packet.len() < 7 {
        return Err(ParseError::PacketTooShort(packet.len()));
    }
    if packet[0] != SETUP_PACKET_TYPE {
        return Err(ParseError::WrongPacketType(packet[0]));
    }
    if packet[1..7] != SETUP_PACKET_MAGIC {
        return Err(ParseError::BadMagic);
    }

    let mut reader = BitReaderLsb::new(&packet[7..]);
    parse_setup_header_body(&mut reader, audio_channels)
}

/// Parse the full Vorbis I setup-header **body** — i.e. the bit-packed
/// payload that follows the 7-byte common header (§4.2.1). The caller
/// must have positioned `reader` at the `codebook_count_minus_1` byte.
///
/// `audio_channels` is the stream's channel count from the
/// identification header; see [`parse_setup_header`] for details.
///
/// On successful return, the reader is positioned immediately after the
/// trailing 1-bit framing flag (so the bit cursor is byte-aligned only
/// by coincidence — the spec does not pad).
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub fn parse_setup_header_body(
    reader: &mut BitReaderLsb<'_>,
    audio_channels: u8,
) -> Result<VorbisSetupHeader, ParseError> {
    if audio_channels == 0 {
        return Err(ParseError::ZeroAudioChannels);
    }

    // ----- Codebooks (§4.2.4 "Codebooks") -----
    let codebook_count = (read_u32(reader, 8)? as usize) + 1;
    let mut codebooks = Vec::with_capacity(codebook_count);
    for index in 0..codebook_count {
        let book =
            parse_codebook(reader).map_err(|source| ParseError::Codebook { index, source })?;
        codebooks.push(book);
    }

    // ----- Time-domain transform placeholders (§4.2.4 "Time domain
    // transforms") -----
    let time_count = (read_u32(reader, 6)? as usize) + 1;
    let mut time_placeholders = Vec::with_capacity(time_count);
    for index in 0..time_count {
        let value = read_u32(reader, 16)? as u16;
        if value != 0 {
            return Err(ParseError::NonZeroTimePlaceholder { index, value });
        }
        time_placeholders.push(value);
    }

    // ----- Floors (§4.2.4 "Floors") -----
    let floor_count = (read_u32(reader, 6)? as usize) + 1;
    let mut floors = Vec::with_capacity(floor_count);
    for index in 0..floor_count {
        let floor_type = read_u32(reader, 16)? as u16;
        let kind = match floor_type {
            0 => FloorKind::Type0(parse_floor0_header(reader)?),
            1 => FloorKind::Type1(parse_floor1_header(reader)?),
            t => {
                return Err(ParseError::UnsupportedFloorType {
                    index,
                    floor_type: t,
                })
            }
        };
        floors.push(FloorHeader { floor_type, kind });
    }

    // ----- Residues (§4.2.4 "Residues") -----
    let residue_count = (read_u32(reader, 6)? as usize) + 1;
    let mut residues = Vec::with_capacity(residue_count);
    for index in 0..residue_count {
        let residue_type = read_u32(reader, 16)? as u16;
        if residue_type > 2 {
            return Err(ParseError::UnsupportedResidueType {
                index,
                residue_type,
            });
        }
        residues.push(parse_residue_header(reader, residue_type)?);
    }

    // ----- Mappings (§4.2.4 "Mappings") -----
    let mapping_count = (read_u32(reader, 6)? as usize) + 1;
    let mut mappings = Vec::with_capacity(mapping_count);
    for index in 0..mapping_count {
        mappings.push(parse_mapping_header(
            reader,
            index,
            audio_channels,
            floors.len(),
            residues.len(),
        )?);
    }

    // ----- Modes (§4.2.4 "Modes") -----
    let mode_count = (read_u32(reader, 6)? as usize) + 1;
    let mut modes = Vec::with_capacity(mode_count);
    for index in 0..mode_count {
        modes.push(parse_mode_header(reader, index, mappings.len())?);
    }

    // ----- Framing flag (§4.2.4 "Modes" step 3) -----
    let framing_flag = read_bit(reader)?;
    if !framing_flag {
        return Err(ParseError::BadFramingFlag);
    }

    Ok(VorbisSetupHeader {
        codebooks,
        time_placeholders,
        floors,
        residues,
        mappings,
        modes,
        framing_flag,
    })
}

/// Read a single floor type-0 setup-header (§6.2.1).
fn parse_floor0_header(reader: &mut BitReaderLsb<'_>) -> Result<Floor0Header, ParseError> {
    // 1) [floor0_order]            = read 8 bits
    // 2) [floor0_rate]             = read 16 bits
    // 3) [floor0_bark_map_size]    = read 16 bits
    // 4) [floor0_amplitude_bits]   = read 6 bits
    // 5) [floor0_amplitude_offset] = read 8 bits
    // 6) [floor0_number_of_books]  = read 4 bits + 1
    // 7) [floor0_book_list]        = [number_of_books] × 8 bits
    let order = read_u32(reader, 8)? as u8;
    let rate = read_u32(reader, 16)? as u16;
    let bark_map_size = read_u32(reader, 16)? as u16;
    let amplitude_bits = read_u32(reader, 6)? as u8;
    let amplitude_offset = read_u32(reader, 8)? as u8;
    let number_of_books = (read_u32(reader, 4)? as usize) + 1;
    let mut book_list = Vec::with_capacity(number_of_books);
    for _ in 0..number_of_books {
        book_list.push(read_u32(reader, 8)? as u8);
    }
    Ok(Floor0Header {
        order,
        rate,
        bark_map_size,
        amplitude_bits,
        amplitude_offset,
        book_list,
    })
}

/// Read a single floor type-1 setup-header (§7.2.2).
fn parse_floor1_header(reader: &mut BitReaderLsb<'_>) -> Result<Floor1Header, ParseError> {
    //  1) [floor1_partitions]       = read 5 bits
    //  3) iterate i over 0..partitions {
    //  4)     class_list[i] = read 4 bits
    //     }
    //  5) maximum_class = max(class_list[*])
    //  6) iterate i over 0..=maximum_class {
    //  7)     class_dimensions[i] = read 3 bits + 1
    //  8)     class_subclasses[i] = read 2 bits
    //  9)     if class_subclasses[i] > 0:
    //          class_masterbooks[i] = read 8 bits
    // 11)     for j in 0..(1 << class_subclasses[i]):
    // 12)         subclass_books[i][j] = read 8 bits - 1  (so 0 → -1 sentinel)
    //     }
    // 13) multiplier = read 2 bits + 1
    // 14) rangebits  = read 4 bits
    // 18) for i in 0..partitions:
    //         current_class = partition_class_list[i]
    // 20)     for j in 0..class_dimensions[current_class]:
    // 21)         x_list[..] = read rangebits bits
    let partitions = read_u32(reader, 5)? as u8;
    let mut partition_class_list = Vec::with_capacity(partitions as usize);
    for _ in 0..partitions {
        partition_class_list.push(read_u32(reader, 4)? as u8);
    }
    // §7.2.2 step 5: maximum_class is the largest scalar value in
    // partition_class_list. With a 4-bit field maximum_class is in
    // 0..=15. The loop iterates 0..=maximum_class, so when
    // partition_class_list is empty (partitions = 0) there are no
    // classes — the spec's range is technically -1..=-1 (i.e. empty)
    // in that corner case.
    let maximum_class = partition_class_list.iter().copied().max();
    let classes = if let Some(maximum_class) = maximum_class {
        let class_count = (maximum_class as usize) + 1;
        let mut classes = Vec::with_capacity(class_count);
        for _ in 0..class_count {
            let dimensions = (read_u32(reader, 3)? as u8) + 1;
            let subclasses = read_u32(reader, 2)? as u8;
            let masterbook = if subclasses > 0 {
                Some(read_u32(reader, 8)? as u8)
            } else {
                None
            };
            let subclass_book_count: usize = 1usize << subclasses;
            let mut subclass_books = Vec::with_capacity(subclass_book_count);
            for _ in 0..subclass_book_count {
                // §7.2.2 step 12: "read 8 bits as unsigned integer and
                // subtract one". A raw value of 0 therefore becomes -1,
                // which is the spec's sentinel for "no codebook
                // configured for this subclass slot" — represented as
                // `None` here.
                let raw = read_u32(reader, 8)? as i32 - 1;
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

    let multiplier = (read_u32(reader, 2)? as u8) + 1;
    let rangebits = read_u32(reader, 4)? as u8;

    // §7.2.2 step 18: read x_list with `class_dimensions[current_class]`
    // values per partition.
    let mut x_list: Vec<u32> = Vec::new();
    for &current_class in &partition_class_list {
        // partition_class_list values index into `classes` (which has
        // length maximum_class + 1), so this lookup is in range by
        // construction.
        let cdim = classes[current_class as usize].dimensions;
        for _ in 0..cdim {
            x_list.push(read_u32(reader, rangebits as u32)?);
        }
    }

    Ok(Floor1Header {
        partitions,
        partition_class_list,
        classes,
        multiplier,
        rangebits,
        x_list,
    })
}

/// Read a single residue header (§8.6.1; identical across types 0/1/2).
fn parse_residue_header(
    reader: &mut BitReaderLsb<'_>,
    residue_type: u16,
) -> Result<ResidueHeader, ParseError> {
    // §8.6.1 layout (common to all three residue types):
    //   1) [residue_begin]            = read 24 bits
    //   2) [residue_end]              = read 24 bits
    //   3) [residue_partition_size]   = read 24 bits + 1
    //   4) [residue_classifications]  = read 6 bits + 1
    //   5) [residue_classbook]        = read 8 bits
    //   iterate i over 0..classifications {
    //       low_bits  = read 3 bits
    //       bitflag   = read 1 bit
    //       high_bits = bitflag ? read 5 bits : 0
    //       cascade[i] = high_bits*8 + low_bits
    //   }
    //   iterate i over 0..classifications {
    //       iterate j over 0..8 {
    //           if cascade[i] bit j set:
    //               books[i][j] = read 8 bits
    //           else:
    //               books[i][j] = unused
    //       }
    //   }
    let residue_begin = read_u32(reader, 24)?;
    let residue_end = read_u32(reader, 24)?;
    let partition_size = read_u32(reader, 24)? + 1;
    let classifications = (read_u32(reader, 6)? as u8) + 1;
    let classbook = read_u32(reader, 8)? as u8;

    let mut cascade = Vec::with_capacity(classifications as usize);
    for _ in 0..classifications {
        let low_bits = read_u32(reader, 3)? as u8;
        let bitflag = read_bit(reader)?;
        let high_bits = if bitflag {
            read_u32(reader, 5)? as u8
        } else {
            0
        };
        // §8.6.1: cascade[i] = high_bits * 8 + low_bits. With
        // high_bits ∈ 0..=31 and low_bits ∈ 0..=7, the product fits in
        // a u8 (max 31*8 + 7 = 255).
        cascade.push(high_bits.wrapping_mul(8).wrapping_add(low_bits));
    }

    let mut books = Vec::with_capacity(classifications as usize);
    for &cas in &cascade {
        let mut row: [Option<u8>; 8] = [None; 8];
        for (j, slot) in row.iter_mut().enumerate() {
            if (cas >> j) & 1 == 1 {
                *slot = Some(read_u32(reader, 8)? as u8);
            }
        }
        books.push(row);
    }

    Ok(ResidueHeader {
        residue_type,
        residue_begin,
        residue_end,
        partition_size,
        classifications,
        classbook,
        cascade,
        books,
    })
}

/// Read a single mapping header (§4.2.4 "Mappings").
///
/// Only `mapping_type = 0` is defined in Vorbis I; any other value is
/// rejected with [`ParseError::UnsupportedMappingType`]. `audio_channels`,
/// `floor_count`, and `residue_count` are validation thresholds for the
/// magnitude/angle channel reads, the `mux[ch]` reads, and the per-submap
/// floor/residue index reads, respectively.
fn parse_mapping_header(
    reader: &mut BitReaderLsb<'_>,
    mapping_index: usize,
    audio_channels: u8,
    floor_count: usize,
    residue_count: usize,
) -> Result<MappingHeader, ParseError> {
    // §4.2.4 "Mappings" step 2a: 16-bit mapping_type.
    let mapping_type = read_u32(reader, 16)? as u16;
    if mapping_type != 0 {
        return Err(ParseError::UnsupportedMappingType {
            index: mapping_index,
            mapping_type,
        });
    }

    // step 2c.i: optional `submaps_flag`. If set, read 4 bits + 1 (so
    // 1..=16); otherwise default to 1.
    let submaps_flag = read_bit(reader)?;
    let submaps = if submaps_flag {
        (read_u32(reader, 4)? as u8) + 1
    } else {
        1
    };

    // step 2c.ii: optional `square_polar_flag`. If set, read 8 bits + 1
    // coupling steps and then magnitude/angle channel numbers each at
    // `ilog(audio_channels - 1)` bits.
    let square_polar_flag = read_bit(reader)?;
    let coupling = if square_polar_flag {
        let coupling_steps = (read_u32(reader, 8)? as usize) + 1;
        // §4.2.4 step 2c.ii.A: per-channel-number width.
        // audio_channels was guaranteed nonzero by the outer entry; for
        // audio_channels == 1, ilog(0) == 0, so each magnitude/angle
        // read would consume 0 bits and the resulting channel numbers
        // would both be 0 — which immediately fails the "magnitude !=
        // angle" check in step 2c.ii. The well-formed mono path
        // therefore goes through `square_polar_flag = 0`; if we get
        // here with channels == 1 and coupling_steps > 0 the per-step
        // validation will reject it. We still read 0 bits for the
        // pair so the bit cursor stays aligned.
        let channel_bits = ilog((audio_channels as u32).saturating_sub(1));
        let mut steps = Vec::with_capacity(coupling_steps);
        for step_index in 0..coupling_steps {
            let magnitude_channel = read_u32(reader, channel_bits)? as u8;
            let angle_channel = read_u32(reader, channel_bits)? as u8;
            // §4.2.4 step 2c.ii (the "If for any coupling step..."
            // paragraph): magnitude != angle, both < audio_channels.
            if magnitude_channel == angle_channel
                || magnitude_channel >= audio_channels
                || angle_channel >= audio_channels
            {
                return Err(ParseError::BadCouplingChannels {
                    mapping_index,
                    step_index,
                    magnitude_channel,
                    angle_channel,
                    audio_channels,
                });
            }
            steps.push(MappingCouplingStep {
                magnitude_channel,
                angle_channel,
            });
        }
        steps
    } else {
        Vec::new()
    };

    // step 2c.iii: 2-bit reserved field, must be 0.
    let reserved = read_u32(reader, 2)? as u8;
    if reserved != 0 {
        return Err(ParseError::NonZeroMappingReserved {
            index: mapping_index,
            value: reserved,
        });
    }

    // step 2c.iv: when submaps > 1, read mux[ch] for each channel.
    let mux = if submaps > 1 {
        let mut mux = Vec::with_capacity(audio_channels as usize);
        for channel_index in 0..(audio_channels as usize) {
            let value = read_u32(reader, 4)? as u8;
            if value >= submaps {
                return Err(ParseError::BadMuxValue {
                    mapping_index,
                    channel_index,
                    mux: value,
                    submaps,
                });
            }
            mux.push(value);
        }
        mux
    } else {
        Vec::new()
    };

    // step 2c.v: per-submap (time-placeholder, floor, residue).
    let mut submap_configs = Vec::with_capacity(submaps as usize);
    for submap_index in 0..(submaps as usize) {
        let time_placeholder = read_u32(reader, 8)? as u8;
        let floor = read_u32(reader, 8)? as u8;
        if (floor as usize) >= floor_count {
            return Err(ParseError::BadSubmapFloor {
                mapping_index,
                submap_index,
                floor,
                floor_count,
            });
        }
        let residue = read_u32(reader, 8)? as u8;
        if (residue as usize) >= residue_count {
            return Err(ParseError::BadSubmapResidue {
                mapping_index,
                submap_index,
                residue,
                residue_count,
            });
        }
        submap_configs.push(MappingSubmap {
            time_placeholder,
            floor,
            residue,
        });
    }

    Ok(MappingHeader {
        mapping_type,
        submaps,
        coupling,
        mux,
        submap_configs,
    })
}

/// Read a single mode header (§4.2.4 "Modes").
fn parse_mode_header(
    reader: &mut BitReaderLsb<'_>,
    mode_index: usize,
    mapping_count: usize,
) -> Result<ModeHeader, ParseError> {
    let blockflag = read_bit(reader)?;
    let windowtype = read_u32(reader, 16)? as u16;
    if windowtype != 0 {
        return Err(ParseError::NonZeroModeWindowType {
            index: mode_index,
            windowtype,
        });
    }
    let transformtype = read_u32(reader, 16)? as u16;
    if transformtype != 0 {
        return Err(ParseError::NonZeroModeTransformType {
            index: mode_index,
            transformtype,
        });
    }
    let mapping = read_u32(reader, 8)? as u8;
    if (mapping as usize) >= mapping_count {
        return Err(ParseError::BadModeMapping {
            index: mode_index,
            mapping,
            mapping_count,
        });
    }
    Ok(ModeHeader {
        blockflag,
        windowtype,
        transformtype,
        mapping,
    })
}

// ---- small read helpers funnelling BitReaderLsb's Error::Eof into
// our ParseError::UnexpectedEndOfPacket. ----

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
    use crate::codebook::VorbisCodebook;
    use oxideav_core::bits::BitWriterLsb;

    /// Builder for the bit-packed body of a tiny well-formed setup
    /// header. Each helper appends a section of the body; the standard
    /// tail of the body — a single trivial mono mapping, a single
    /// short-block mode pointing at it, and the framing flag — is
    /// emitted by [`Self::minimal_tail`].
    struct SetupBuilder {
        w: BitWriterLsb,
    }

    impl SetupBuilder {
        fn new() -> Self {
            Self {
                w: BitWriterLsb::with_capacity(128),
            }
        }

        fn codebook_count_minus_1(mut self, n: u8) -> Self {
            self.w.write_u32(n as u32, 8);
            self
        }

        /// Append a trivial one-entry codebook (`dimensions=1
        /// entries=1 ordered=0 sparse=0 length=1 lookup_type=0`). The
        /// resulting tree builder is fine because errata 20150226
        /// allows a single-entry length-1 codebook.
        fn trivial_codebook(mut self) -> Self {
            self.w.write_u32(VorbisCodebook::SYNC_PATTERN, 24); // sync
            self.w.write_u32(1, 16); // dimensions
            self.w.write_u32(1, 24); // entries
            self.w.write_bit(false); // ordered = 0
            self.w.write_bit(false); // sparse = 0
            self.w.write_u32(0, 5); // length - 1 = 0 → length 1
            self.w.write_u32(0, 4); // lookup_type = 0
            self
        }

        fn time_count_minus_1(mut self, n: u8) -> Self {
            self.w.write_u32(n as u32, 6);
            self
        }

        fn time_placeholder(mut self, value: u16) -> Self {
            self.w.write_u32(value as u32, 16);
            self
        }

        fn floor_count_minus_1(mut self, n: u8) -> Self {
            self.w.write_u32(n as u32, 6);
            self
        }

        /// A minimal floor 1 with `partitions=1`, one class of
        /// dimensions=1, subclasses=0, multiplier=1, rangebits=4, and
        /// one x_list element.
        fn minimal_floor1(mut self) -> Self {
            self.w.write_u32(1, 16); // floor_type = 1
            self.w.write_u32(1, 5); // partitions = 1
            self.w.write_u32(0, 4); // partition_class_list[0] = 0
                                    // one class (maximum_class = 0):
            self.w.write_u32(0, 3); // dimensions - 1 = 0 → 1
            self.w.write_u32(0, 2); // subclasses = 0
                                    // subclasses=0 → no masterbook; 2^0=1 subclass slot:
            self.w.write_u32(1, 8); // subclass_books[0][0] = 0 (after −1 → −1 sentinel; encode raw=1 → stored value 0)
            self.w.write_u32(0, 2); // multiplier - 1 = 0 → 1
            self.w.write_u32(4, 4); // rangebits = 4
                                    // x_list: partitions × class.dimensions = 1 × 1 = 1 read:
            self.w.write_u32(5, 4); // x_list[0] = 5
            self
        }

        /// A minimal floor 0: order=4, rate=44100, bark_map_size=64,
        /// amplitude_bits=8, amplitude_offset=100, book_list=[0].
        fn minimal_floor0(mut self) -> Self {
            self.w.write_u32(0, 16); // floor_type = 0
            self.w.write_u32(4, 8); // order
            self.w.write_u32(44100, 16); // rate
            self.w.write_u32(64, 16); // bark_map_size
            self.w.write_u32(8, 6); // amplitude_bits
            self.w.write_u32(100, 8); // amplitude_offset
            self.w.write_u32(0, 4); // number_of_books - 1 = 0 → 1
            self.w.write_u32(0, 8); // book_list[0]
            self
        }

        fn residue_count_minus_1(mut self, n: u8) -> Self {
            self.w.write_u32(n as u32, 6);
            self
        }

        /// A minimal residue: type 2, begin=0, end=128,
        /// partition_size=16, classifications=1, classbook=0,
        /// cascade[0]=0b0000_0001 (just stage 0), books[0][0]=0.
        fn minimal_residue_type2(mut self) -> Self {
            self.w.write_u32(2, 16); // residue_type
            self.w.write_u32(0, 24); // begin
            self.w.write_u32(128, 24); // end
            self.w.write_u32(15, 24); // partition_size - 1 = 15 → 16
            self.w.write_u32(0, 6); // classifications - 1 = 0 → 1
            self.w.write_u32(0, 8); // classbook
                                    // cascade[0]: low_bits=1, bitflag=0 → high_bits=0 → cascade=1
            self.w.write_u32(1, 3); // low_bits
            self.w.write_bit(false); // bitflag = 0
                                     // books[0][0] = 0 (only stage 0 is set in cascade)
            self.w.write_u32(0, 8);
            self
        }

        fn mapping_count_minus_1(mut self, n: u8) -> Self {
            self.w.write_u32(n as u32, 6);
            self
        }

        /// Append a minimal mono mapping: `mapping_type=0`,
        /// `submaps_flag=0` (so submaps=1), `square_polar_flag=0` (no
        /// coupling), 2-bit reserved=0, no `mux[]` (submaps==1), one
        /// submap with `(time=0, floor=0, residue=0)`. Suitable for
        /// stream(s) that already declare at least one floor + residue
        /// indexed at slot 0.
        fn minimal_mono_mapping(mut self) -> Self {
            self.w.write_u32(0, 16); // mapping_type = 0
            self.w.write_bit(false); // submaps_flag = 0 → submaps = 1
            self.w.write_bit(false); // square_polar_flag = 0 → coupling_steps = 0
            self.w.write_u32(0, 2); // reserved
                                    // submaps == 1: no mux[]
                                    // one submap: 8 bits placeholder + 8 bits floor + 8 bits residue
            self.w.write_u32(0, 8); // time placeholder
            self.w.write_u32(0, 8); // floor index
            self.w.write_u32(0, 8); // residue index
            self
        }

        fn mode_count_minus_1(mut self, n: u8) -> Self {
            self.w.write_u32(n as u32, 6);
            self
        }

        /// Append a single mode: blockflag=0, windowtype=0,
        /// transformtype=0, mapping=0.
        fn minimal_short_mode(mut self) -> Self {
            self.w.write_bit(false); // blockflag = 0 (short)
            self.w.write_u32(0, 16); // windowtype
            self.w.write_u32(0, 16); // transformtype
            self.w.write_u32(0, 8); // mapping
            self
        }

        /// Append the final framing flag = 1 (well-formed).
        fn framing_flag(mut self, set: bool) -> Self {
            self.w.write_bit(set);
            self
        }

        /// Append the standard well-formed tail: one trivial mono
        /// mapping, one short-block mode pointing at it, framing flag
        /// set. Suitable for tests whose body already contains at
        /// least one codebook, one time placeholder, one floor, one
        /// residue.
        fn minimal_tail(self) -> Self {
            self.mapping_count_minus_1(0)
                .minimal_mono_mapping()
                .mode_count_minus_1(0)
                .minimal_short_mode()
                .framing_flag(true)
        }

        fn finish(self) -> Vec<u8> {
            self.w.finish()
        }
    }

    #[test]
    fn parses_minimal_setup_body() {
        let bytes = SetupBuilder::new()
            .codebook_count_minus_1(0) // 1 codebook
            .trivial_codebook()
            .time_count_minus_1(0) // 1 time placeholder
            .time_placeholder(0)
            .floor_count_minus_1(0) // 1 floor
            .minimal_floor1()
            .residue_count_minus_1(0) // 1 residue
            .minimal_residue_type2()
            .minimal_tail() // 1 mapping + 1 mode + framing
            .finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r, 1).expect("must parse");
        assert_eq!(setup.codebooks.len(), 1);
        assert_eq!(setup.codebooks[0].entries, 1);
        assert_eq!(setup.time_placeholders, vec![0]);
        assert_eq!(setup.floors.len(), 1);
        assert_eq!(setup.floors[0].floor_type, 1);
        match &setup.floors[0].kind {
            FloorKind::Type1(f1) => {
                assert_eq!(f1.partitions, 1);
                assert_eq!(f1.partition_class_list, vec![0]);
                assert_eq!(f1.classes.len(), 1);
                assert_eq!(f1.classes[0].dimensions, 1);
                assert_eq!(f1.classes[0].subclasses, 0);
                assert_eq!(f1.classes[0].masterbook, None);
                assert_eq!(f1.classes[0].subclass_books, vec![Some(0)]);
                assert_eq!(f1.multiplier, 1);
                assert_eq!(f1.rangebits, 4);
                assert_eq!(f1.x_list, vec![5]);
            }
            other => panic!("expected Type1, got {other:?}"),
        }
        assert_eq!(setup.residues.len(), 1);
        let r = &setup.residues[0];
        assert_eq!(r.residue_type, 2);
        assert_eq!(r.residue_begin, 0);
        assert_eq!(r.residue_end, 128);
        assert_eq!(r.partition_size, 16);
        assert_eq!(r.classifications, 1);
        assert_eq!(r.classbook, 0);
        assert_eq!(r.cascade, vec![1]);
        assert_eq!(r.books[0][0], Some(0));
        for slot in &r.books[0][1..] {
            assert_eq!(*slot, None);
        }
        // round-6 tail
        assert_eq!(setup.mappings.len(), 1);
        assert_eq!(setup.mappings[0].mapping_type, 0);
        assert_eq!(setup.mappings[0].submaps, 1);
        assert!(setup.mappings[0].coupling.is_empty());
        assert!(setup.mappings[0].mux.is_empty());
        assert_eq!(setup.mappings[0].submap_configs.len(), 1);
        assert_eq!(setup.mappings[0].submap_configs[0].floor, 0);
        assert_eq!(setup.mappings[0].submap_configs[0].residue, 0);
        assert_eq!(setup.modes.len(), 1);
        assert!(!setup.modes[0].blockflag);
        assert_eq!(setup.modes[0].windowtype, 0);
        assert_eq!(setup.modes[0].transformtype, 0);
        assert_eq!(setup.modes[0].mapping, 0);
        assert!(setup.framing_flag);
    }

    /// Common-header validation: 7-byte prefix `0x05 "vorbis"`.
    #[test]
    fn parses_full_setup_packet_with_common_header() {
        // Build a packet body, then prepend the 7-byte common header.
        let body = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .minimal_tail()
            .finish();
        let mut packet = Vec::with_capacity(7 + body.len());
        packet.push(SETUP_PACKET_TYPE);
        packet.extend_from_slice(&SETUP_PACKET_MAGIC);
        packet.extend_from_slice(&body);
        let setup = parse_setup_header(&packet, 1).expect("must parse");
        assert_eq!(setup.codebooks.len(), 1);
        assert_eq!(setup.mappings.len(), 1);
        assert_eq!(setup.modes.len(), 1);
        assert!(setup.framing_flag);
    }

    #[test]
    fn rejects_short_packet() {
        let packet = [0u8; 3];
        match parse_setup_header(&packet, 1) {
            Err(ParseError::PacketTooShort(3)) => {}
            other => panic!("expected PacketTooShort(3), got {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_packet_type() {
        let mut packet = vec![0u8; 7];
        packet[0] = 0x01; // identification header byte, not 0x05
        packet[1..7].copy_from_slice(&SETUP_PACKET_MAGIC);
        match parse_setup_header(&packet, 1) {
            Err(ParseError::WrongPacketType(0x01)) => {}
            other => panic!("expected WrongPacketType(0x01), got {other:?}"),
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut packet = vec![0u8; 7];
        packet[0] = SETUP_PACKET_TYPE;
        packet[1..7].copy_from_slice(b"Vorbis"); // capitalised
        assert_eq!(parse_setup_header(&packet, 1), Err(ParseError::BadMagic));
    }

    /// §4.2.4 step 2: a nonzero time-placeholder is fatal.
    #[test]
    fn rejects_nonzero_time_placeholder() {
        let bytes = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0x1234) // nonzero!
            .finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 1) {
            Err(ParseError::NonZeroTimePlaceholder {
                index: 0,
                value: 0x1234,
            }) => {}
            other => panic!("expected NonZeroTimePlaceholder, got {other:?}"),
        }
    }

    /// §4.2.4 step 2d: a floor type > 1 is fatal.
    #[test]
    fn rejects_unsupported_floor_type() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0);
        b.w.write_u32(7, 16); // floor_type = 7 — reserved
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 1) {
            Err(ParseError::UnsupportedFloorType {
                index: 0,
                floor_type: 7,
            }) => {}
            other => panic!("expected UnsupportedFloorType(0, 7), got {other:?}"),
        }
    }

    /// §4.2.4 step 2c: a residue type > 2 is fatal.
    #[test]
    fn rejects_unsupported_residue_type() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0);
        b.w.write_u32(3, 16); // residue_type = 3 — reserved
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 1) {
            Err(ParseError::UnsupportedResidueType {
                index: 0,
                residue_type: 3,
            }) => {}
            other => panic!("expected UnsupportedResidueType(0, 3), got {other:?}"),
        }
    }

    /// A truncated packet (bits run out mid-codebook) surfaces as
    /// `UnexpectedEndOfPacket`.
    #[test]
    fn rejects_truncated_packet() {
        // codebook_count_minus_1 = 0 → 1 codebook expected, but
        // immediately follow with a single zero byte so the codebook
        // sync read runs out of bits.
        let bytes = vec![0u8, 0u8];
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 1) {
            Err(ParseError::Codebook {
                index: 0,
                source: _,
            }) => {
                // Inner error is some codebook ParseError — likely
                // BadSyncPattern or UnexpectedEndOfPacket depending on
                // exactly when bits run out. Either is fine.
            }
            other => panic!("expected Codebook error wrapping inner, got {other:?}"),
        }
    }

    /// A floor 1 with `partitions = 0` is legal (zero classes), since
    /// the spec's iteration of `0..floor1_partitions` and
    /// `0..=maximum_class` both degenerate to empty.
    #[test]
    fn parses_floor1_with_zero_partitions() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0);
        // floor_type=1, partitions=0, then no partition_class_list, no
        // classes, multiplier and rangebits, then no x_list.
        b.w.write_u32(1, 16);
        b.w.write_u32(0, 5); // partitions = 0
        b.w.write_u32(0, 2); // multiplier - 1 = 0 → 1
        b.w.write_u32(4, 4); // rangebits = 4 (any value works; no x_list to read)
        let mut b2 = b;
        b2 = b2
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .minimal_tail();
        let bytes = b2.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r, 1).expect("must parse");
        match &setup.floors[0].kind {
            FloorKind::Type1(f1) => {
                assert_eq!(f1.partitions, 0);
                assert!(f1.classes.is_empty());
                assert!(f1.x_list.is_empty());
                assert_eq!(f1.rangebits, 4);
            }
            other => panic!("expected Type1, got {other:?}"),
        }
    }

    /// Floor 0 (LSP) path — exercised even though libvorbis does not
    /// produce it. §6.2.1 setup must still be parsable.
    #[test]
    fn parses_floor0_setup() {
        let bytes = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor0()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .minimal_tail()
            .finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r, 1).expect("must parse");
        assert_eq!(setup.floors[0].floor_type, 0);
        match &setup.floors[0].kind {
            FloorKind::Type0(f0) => {
                assert_eq!(f0.order, 4);
                assert_eq!(f0.rate, 44100);
                assert_eq!(f0.bark_map_size, 64);
                assert_eq!(f0.amplitude_bits, 8);
                assert_eq!(f0.amplitude_offset, 100);
                assert_eq!(f0.book_list, vec![0]);
            }
            other => panic!("expected Type0, got {other:?}"),
        }
    }

    /// Multi-codebook + multi-floor + multi-residue setup, mirroring
    /// the trace-doc §2.4 stereo-q5 shape (`codebook_count` plural,
    /// `time_count=1`, `floor_count=2`, `residue_count=2`). Round 6
    /// adds the two-mapping + two-mode tail and asserts the mode
    /// blockflags + mapping references are read in order.
    #[test]
    fn parses_two_floors_two_residues() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(1) // 2 codebooks
            .trivial_codebook()
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(1) // 2 floors
            .minimal_floor1()
            .minimal_floor1()
            .residue_count_minus_1(1) // 2 residues
            .minimal_residue_type2()
            .minimal_residue_type2();
        // Two stereo-style mappings: each is the same mono-shape mapping
        // (no coupling) but routes its single submap to a different
        // (floor, residue) pair. mapping[0] = (floor 0, residue 0);
        // mapping[1] = (floor 1, residue 1).
        b = b.mapping_count_minus_1(1);
        // mapping[0]
        b.w.write_u32(0, 16); // mapping_type
        b.w.write_bit(false); // submaps_flag
        b.w.write_bit(false); // square_polar_flag
        b.w.write_u32(0, 2); // reserved
        b.w.write_u32(0, 8); // time placeholder
        b.w.write_u32(0, 8); // floor 0
        b.w.write_u32(0, 8); // residue 0
                             // mapping[1]
        b.w.write_u32(0, 16);
        b.w.write_bit(false);
        b.w.write_bit(false);
        b.w.write_u32(0, 2);
        b.w.write_u32(0, 8);
        b.w.write_u32(1, 8); // floor 1
        b.w.write_u32(1, 8); // residue 1
                             // 2 modes: short → mapping 0, long → mapping 1
        b = b.mode_count_minus_1(1);
        b.w.write_bit(false); // mode[0].blockflag = 0
        b.w.write_u32(0, 16);
        b.w.write_u32(0, 16);
        b.w.write_u32(0, 8); // mapping[0]
        b.w.write_bit(true); // mode[1].blockflag = 1
        b.w.write_u32(0, 16);
        b.w.write_u32(0, 16);
        b.w.write_u32(1, 8); // mapping[1]
        let b = b.framing_flag(true);
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r, 1).expect("must parse");
        assert_eq!(setup.codebooks.len(), 2);
        assert_eq!(setup.floors.len(), 2);
        assert_eq!(setup.residues.len(), 2);
        for floor in &setup.floors {
            assert_eq!(floor.floor_type, 1);
        }
        for residue in &setup.residues {
            assert_eq!(residue.residue_type, 2);
            assert_eq!(residue.partition_size, 16);
        }
        assert_eq!(setup.mappings.len(), 2);
        assert_eq!(setup.mappings[0].submap_configs[0].floor, 0);
        assert_eq!(setup.mappings[0].submap_configs[0].residue, 0);
        assert_eq!(setup.mappings[1].submap_configs[0].floor, 1);
        assert_eq!(setup.mappings[1].submap_configs[0].residue, 1);
        assert_eq!(setup.modes.len(), 2);
        assert!(!setup.modes[0].blockflag);
        assert!(setup.modes[1].blockflag);
        assert_eq!(setup.modes[0].mapping, 0);
        assert_eq!(setup.modes[1].mapping, 1);
        assert!(setup.framing_flag);
    }

    /// Residue cascade exercise: classification 0 has a cascade
    /// bitmap of `0b00010011` (= 19 = 16 + 3 = bits 0, 1, 4 set), so
    /// three book reads should follow.
    #[test]
    fn parses_residue_with_multistage_cascade() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0);
        b.w.write_u32(1, 16); // residue_type = 1
        b.w.write_u32(0, 24); // begin
        b.w.write_u32(64, 24); // end
        b.w.write_u32(7, 24); // partition_size - 1 = 7 → 8
        b.w.write_u32(0, 6); // classifications - 1 = 0 → 1
        b.w.write_u32(0, 8); // classbook
                             // cascade[0]: low_bits=3, bitflag=1, high_bits=2 → cascade = 2*8 + 3 = 19 = 0b00010011
        b.w.write_u32(3, 3); // low_bits = 3 (bits 0, 1 set)
        b.w.write_bit(true); // bitflag
        b.w.write_u32(2, 5); // high_bits = 2 (so bit 4 set in cascade)
                             // books[0][...]: bits 0, 1, 4 set → 3 reads
        b.w.write_u32(0, 8);
        b.w.write_u32(1, 8);
        b.w.write_u32(2, 8);
        let b = b.minimal_tail();
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r, 1).expect("must parse");
        let res = &setup.residues[0];
        assert_eq!(res.residue_type, 1);
        assert_eq!(res.cascade, vec![0b00010011]);
        assert_eq!(res.books[0][0], Some(0));
        assert_eq!(res.books[0][1], Some(1));
        assert_eq!(res.books[0][2], None);
        assert_eq!(res.books[0][3], None);
        assert_eq!(res.books[0][4], Some(2));
        for slot in &res.books[0][5..] {
            assert_eq!(*slot, None);
        }
    }

    /// Residue type 0 path (no libvorbis fixture exercises it, but
    /// §4.2.4 step 2c requires the parser to accept it).
    #[test]
    fn parses_residue_type_0() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0);
        b.w.write_u32(0, 16); // residue_type = 0
        b.w.write_u32(0, 24); // begin
        b.w.write_u32(32, 24); // end
        b.w.write_u32(7, 24); // partition_size - 1 = 7 → 8
        b.w.write_u32(0, 6); // classifications - 1 = 0 → 1
        b.w.write_u32(0, 8); // classbook
                             // cascade[0] = bit 0 only (low_bits=1, bitflag=0, high_bits=0)
        b.w.write_u32(1, 3);
        b.w.write_bit(false);
        b.w.write_u32(0, 8); // books[0][0]
        let b = b.minimal_tail();
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r, 1).expect("must parse");
        assert_eq!(setup.residues[0].residue_type, 0);
    }

    // ===== round-6 mapping + mode coverage =====

    /// A stereo mapping with a single coupling step (magnitude=0,
    /// angle=1) — the trace-doc §6 "stereo libvorbis output always
    /// uses one coupling step (magnitude=L, angle=R)" shape.
    #[test]
    fn parses_stereo_coupling_mapping() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .mapping_count_minus_1(0);
        // mapping[0]: type=0, submaps_flag=0, square_polar_flag=1,
        // coupling_steps=1, magnitude=0, angle=1, reserved=0, no mux,
        // one submap (floor=0, residue=0).
        b.w.write_u32(0, 16); // mapping_type
        b.w.write_bit(false); // submaps_flag → submaps = 1
        b.w.write_bit(true); // square_polar_flag
        b.w.write_u32(0, 8); // coupling_steps - 1 = 0 → 1
                             // audio_channels = 2 → ilog(1) = 1 bit per channel index
        b.w.write_u32(0, 1); // magnitude = 0
        b.w.write_u32(1, 1); // angle = 1
        b.w.write_u32(0, 2); // reserved
                             // no mux (submaps == 1)
        b.w.write_u32(0, 8); // time placeholder
        b.w.write_u32(0, 8); // floor
        b.w.write_u32(0, 8); // residue
                             // 1 mode + framing
        let b = b
            .mode_count_minus_1(0)
            .minimal_short_mode()
            .framing_flag(true);
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r, 2).expect("must parse");
        assert_eq!(setup.mappings.len(), 1);
        let m = &setup.mappings[0];
        assert_eq!(m.coupling.len(), 1);
        assert_eq!(m.coupling[0].magnitude_channel, 0);
        assert_eq!(m.coupling[0].angle_channel, 1);
        assert_eq!(m.submaps, 1);
        assert!(m.mux.is_empty());
    }

    /// A 5.1 multi-submap mapping (trace-doc §6: "5.1 streams use
    /// submaps=2 with mux=[0,0,0,0,0,1] routing the LFE on its own
    /// submap"). audio_channels=6.
    #[test]
    fn parses_multi_submap_mapping() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(1) // 2 floors
            .minimal_floor1()
            .minimal_floor1()
            .residue_count_minus_1(1) // 2 residues
            .minimal_residue_type2()
            .minimal_residue_type2()
            .mapping_count_minus_1(0);
        // mapping[0]: type=0, submaps_flag=1 (submaps=2),
        // square_polar_flag=0, reserved=0, mux=[0,0,0,0,0,1] for the
        // 6 channels, two submaps each carrying a distinct floor/residue.
        b.w.write_u32(0, 16); // mapping_type
        b.w.write_bit(true); // submaps_flag
        b.w.write_u32(1, 4); // submaps - 1 = 1 → 2 submaps
        b.w.write_bit(false); // square_polar_flag = 0 (5.1 has no coupling per fixture trace)
        b.w.write_u32(0, 2); // reserved
                             // mux for 6 channels
        for &v in &[0u32, 0, 0, 0, 0, 1] {
            b.w.write_u32(v, 4);
        }
        // submap 0: (placeholder, floor 0, residue 0)
        b.w.write_u32(0, 8);
        b.w.write_u32(0, 8);
        b.w.write_u32(0, 8);
        // submap 1: (placeholder, floor 1, residue 1)
        b.w.write_u32(0, 8);
        b.w.write_u32(1, 8);
        b.w.write_u32(1, 8);
        let b = b
            .mode_count_minus_1(0)
            .minimal_short_mode()
            .framing_flag(true);
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r, 6).expect("must parse");
        let m = &setup.mappings[0];
        assert_eq!(m.submaps, 2);
        assert_eq!(m.mux, vec![0, 0, 0, 0, 0, 1]);
        assert_eq!(m.submap_configs.len(), 2);
        assert_eq!(m.submap_configs[0].floor, 0);
        assert_eq!(m.submap_configs[0].residue, 0);
        assert_eq!(m.submap_configs[1].floor, 1);
        assert_eq!(m.submap_configs[1].residue, 1);
    }

    /// A `mapping_type != 0` is fatal (§4.2.4 step 2b).
    #[test]
    fn rejects_unsupported_mapping_type() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .mapping_count_minus_1(0);
        b.w.write_u32(1, 16); // mapping_type = 1 — reserved
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 1) {
            Err(ParseError::UnsupportedMappingType {
                index: 0,
                mapping_type: 1,
            }) => {}
            other => panic!("expected UnsupportedMappingType(0, 1), got {other:?}"),
        }
    }

    /// Coupling step with magnitude_channel == angle_channel is fatal
    /// per §4.2.4 step 2c.ii.
    #[test]
    fn rejects_coupling_with_equal_channels() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .mapping_count_minus_1(0);
        b.w.write_u32(0, 16);
        b.w.write_bit(false); // submaps_flag → 1
        b.w.write_bit(true); // square_polar_flag
        b.w.write_u32(0, 8); // coupling_steps - 1 = 0 → 1
                             // audio_channels=2 → ilog(1)=1 bit per channel
        b.w.write_u32(0, 1); // magnitude = 0
        b.w.write_u32(0, 1); // angle = 0 (== magnitude → reject)
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 2) {
            Err(ParseError::BadCouplingChannels {
                mapping_index: 0,
                step_index: 0,
                magnitude_channel: 0,
                angle_channel: 0,
                audio_channels: 2,
            }) => {}
            other => panic!("expected BadCouplingChannels(0,0,0=0,ch=2), got {other:?}"),
        }
    }

    /// Coupling step with magnitude_channel >= audio_channels is fatal.
    #[test]
    fn rejects_coupling_with_oob_magnitude() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .mapping_count_minus_1(0);
        b.w.write_u32(0, 16);
        b.w.write_bit(false); // submaps_flag → 1
        b.w.write_bit(true); // square_polar_flag
        b.w.write_u32(0, 8); // coupling_steps - 1 = 0 → 1
                             // audio_channels=3 → ilog(2)=2 bits per channel
        b.w.write_u32(3, 2); // magnitude = 3 (out of range; channels are 0..=2)
        b.w.write_u32(0, 2); // angle = 0
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 3) {
            Err(ParseError::BadCouplingChannels {
                magnitude_channel: 3,
                angle_channel: 0,
                audio_channels: 3,
                ..
            }) => {}
            other => panic!("expected BadCouplingChannels mag=3 oob, got {other:?}"),
        }
    }

    /// Mapping reserved field nonzero → reject per §4.2.4 step 2c.iii.
    #[test]
    fn rejects_nonzero_mapping_reserved() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .mapping_count_minus_1(0);
        b.w.write_u32(0, 16); // mapping_type
        b.w.write_bit(false); // submaps_flag
        b.w.write_bit(false); // square_polar_flag
        b.w.write_u32(3, 2); // reserved = 3 (nonzero!)
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 1) {
            Err(ParseError::NonZeroMappingReserved { index: 0, value: 3 }) => {}
            other => panic!("expected NonZeroMappingReserved(0, 3), got {other:?}"),
        }
    }

    /// `mux[ch] >= submaps` → reject per §4.2.4 step 2c.iv.B.
    #[test]
    fn rejects_oob_mux_value() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .mapping_count_minus_1(0);
        b.w.write_u32(0, 16);
        b.w.write_bit(true); // submaps_flag
        b.w.write_u32(1, 4); // submaps - 1 = 1 → 2 submaps
        b.w.write_bit(false); // no coupling
        b.w.write_u32(0, 2); // reserved
                             // mux for 2 channels: [0, 5] — second value 5 is out of range
        b.w.write_u32(0, 4);
        b.w.write_u32(5, 4);
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 2) {
            Err(ParseError::BadMuxValue {
                mapping_index: 0,
                channel_index: 1,
                mux: 5,
                submaps: 2,
            }) => {}
            other => panic!("expected BadMuxValue(0,1,5,2), got {other:?}"),
        }
    }

    /// Submap floor index out of range → reject per §4.2.4 step 2c.v.C.
    #[test]
    fn rejects_oob_submap_floor() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0) // 1 floor (index 0 only)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .mapping_count_minus_1(0);
        b.w.write_u32(0, 16);
        b.w.write_bit(false); // submaps = 1
        b.w.write_bit(false); // no coupling
        b.w.write_u32(0, 2); // reserved
        b.w.write_u32(0, 8); // time placeholder
        b.w.write_u32(2, 8); // floor = 2 (> floor_count - 1 = 0)
        b.w.write_u32(0, 8); // residue
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 1) {
            Err(ParseError::BadSubmapFloor {
                mapping_index: 0,
                submap_index: 0,
                floor: 2,
                floor_count: 1,
            }) => {}
            other => panic!("expected BadSubmapFloor(0,0,2,1), got {other:?}"),
        }
    }

    /// Submap residue index out of range → reject per §4.2.4 step 2c.v.E.
    #[test]
    fn rejects_oob_submap_residue() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0) // 1 residue (index 0 only)
            .minimal_residue_type2()
            .mapping_count_minus_1(0);
        b.w.write_u32(0, 16);
        b.w.write_bit(false);
        b.w.write_bit(false);
        b.w.write_u32(0, 2);
        b.w.write_u32(0, 8); // time placeholder
        b.w.write_u32(0, 8); // floor
        b.w.write_u32(7, 8); // residue = 7 (> residue_count - 1 = 0)
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 1) {
            Err(ParseError::BadSubmapResidue {
                mapping_index: 0,
                submap_index: 0,
                residue: 7,
                residue_count: 1,
            }) => {}
            other => panic!("expected BadSubmapResidue(0,0,7,1), got {other:?}"),
        }
    }

    /// Mode windowtype != 0 → reject per §4.2.4 "Modes" step 2e.
    #[test]
    fn rejects_nonzero_mode_windowtype() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .mapping_count_minus_1(0)
            .minimal_mono_mapping()
            .mode_count_minus_1(0);
        b.w.write_bit(false); // blockflag
        b.w.write_u32(1, 16); // windowtype = 1 — reserved
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 1) {
            Err(ParseError::NonZeroModeWindowType {
                index: 0,
                windowtype: 1,
            }) => {}
            other => panic!("expected NonZeroModeWindowType(0, 1), got {other:?}"),
        }
    }

    /// Mode transformtype != 0 → reject per §4.2.4 "Modes" step 2e.
    #[test]
    fn rejects_nonzero_mode_transformtype() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .mapping_count_minus_1(0)
            .minimal_mono_mapping()
            .mode_count_minus_1(0);
        b.w.write_bit(false); // blockflag
        b.w.write_u32(0, 16); // windowtype
        b.w.write_u32(2, 16); // transformtype = 2 — reserved
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 1) {
            Err(ParseError::NonZeroModeTransformType {
                index: 0,
                transformtype: 2,
            }) => {}
            other => panic!("expected NonZeroModeTransformType(0, 2), got {other:?}"),
        }
    }

    /// Mode mapping index out of range → reject per §4.2.4 "Modes"
    /// step 2e.
    #[test]
    fn rejects_oob_mode_mapping() {
        let mut b = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .mapping_count_minus_1(0) // 1 mapping (index 0 only)
            .minimal_mono_mapping()
            .mode_count_minus_1(0);
        b.w.write_bit(false);
        b.w.write_u32(0, 16);
        b.w.write_u32(0, 16);
        b.w.write_u32(4, 8); // mapping = 4 (> mapping_count - 1 = 0)
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match parse_setup_header_body(&mut r, 1) {
            Err(ParseError::BadModeMapping {
                index: 0,
                mapping: 4,
                mapping_count: 1,
            }) => {}
            other => panic!("expected BadModeMapping(0,4,1), got {other:?}"),
        }
    }

    /// Trailing framing flag = 0 → reject per §4.2.4 "Modes" step 3.
    #[test]
    fn rejects_unset_framing_flag() {
        let bytes = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .mapping_count_minus_1(0)
            .minimal_mono_mapping()
            .mode_count_minus_1(0)
            .minimal_short_mode()
            .framing_flag(false) // 0 → reject
            .finish();
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            parse_setup_header_body(&mut r, 1),
            Err(ParseError::BadFramingFlag)
        );
    }

    /// `audio_channels = 0` is a caller bug; reported as a structured
    /// error rather than panicking.
    #[test]
    fn rejects_zero_audio_channels() {
        let bytes = SetupBuilder::new()
            .codebook_count_minus_1(0)
            .trivial_codebook()
            .time_count_minus_1(0)
            .time_placeholder(0)
            .floor_count_minus_1(0)
            .minimal_floor1()
            .residue_count_minus_1(0)
            .minimal_residue_type2()
            .minimal_tail()
            .finish();
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            parse_setup_header_body(&mut r, 0),
            Err(ParseError::ZeroAudioChannels)
        );
    }
}

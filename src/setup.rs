//! Vorbis I setup-header outer walker (Vorbis I §4.2.4).
//!
//! The setup header is the third of three required Vorbis header packets
//! (§4.2.1). It carries the codec configuration the audio packets need:
//! the list of codebooks (§3.2.1), placeholder time-domain transforms
//! (§4.2.4 "Time domain transforms"), the floor configurations (§6, §7),
//! the residue configurations (§8), the mappings (§4.3), and the modes
//! (§4.3.1). The packet ends with a 1-bit framing flag (§4.2.4 "Modes"
//! step 3).
//!
//! ## Round-5 scope
//!
//! This module currently parses the **first four** sub-lists of the
//! setup header:
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
//!
//! **Mappings (§4.3) and modes (§4.3.1) are deferred to round 6.** This
//! means the bit reader is left positioned immediately after the last
//! residue's book list; the framing flag is *not* read in this round,
//! because mapping + mode bits still sit between us and it. Callers must
//! therefore treat the resulting [`VorbisSetupHeader`] as a *partial*
//! parse: the codebook list is enough to drive Huffman decode, but
//! audio-packet decode (which needs the modes) is not yet wired up.
//!
//! The trace-doc §2.4 shape — `codebook_count`, `time_count`,
//! `floor_count`, `residue_count`, then `mapping_count` + `mode_count` —
//! matches this layout exactly; the round-5 walker emits the first four
//! counts and stops there.
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

use crate::codebook::{self, parse_codebook, VorbisCodebook};

/// Common-header packet-type byte for the setup header (Vorbis I §4.2.1).
pub const SETUP_PACKET_TYPE: u8 = 0x05;

/// The six magic bytes that follow the packet-type byte in every Vorbis
/// header packet (Vorbis I §4.2.1).
pub const SETUP_PACKET_MAGIC: [u8; 6] = *b"vorbis";

/// Parsed structural shell of a Vorbis I setup header (Vorbis I §4.2.4)
/// for the **round-5 scope**: codebooks, time-domain placeholders, floor
/// headers, and residue headers. Mappings + modes + framing flag remain
/// to be wired in subsequent rounds.
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
}

/// A parsed floor header (§6.2.1 for type 0, §7.2.2 for type 1).
///
/// Floor type is recorded both as a 16-bit `floor_type` field (matching
/// the trace-doc field shape) and as the discriminant on the enum's
/// `kind` payload. A floor type strictly greater than 1 is rejected at
/// parse time per §4.2.4 step 2d.
#[derive(Debug, Clone, PartialEq)]
pub struct FloorHeader {
    /// The 16-bit `floor_type` field that the encoder wrote.
    pub floor_type: u16,
    /// Per-type structural payload.
    pub kind: FloorKind,
}

/// Per-type floor structural fields.
#[derive(Debug, Clone, PartialEq)]
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

/// Parse the round-5 portion of a Vorbis I setup-header packet.
///
/// The supplied `packet` must contain the full setup-header packet
/// (common header + bit-packed body). The function validates the
/// 7-byte common header per §4.2.1 then delegates to
/// [`parse_setup_header_body`] for the bit-packed body.
///
/// Returns a *partial* [`VorbisSetupHeader`] populated with codebooks,
/// time placeholders, floors, and residues. The mapping list, mode list,
/// and framing flag are **not** consumed — that lives behind round 6.
pub fn parse_setup_header(packet: &[u8]) -> Result<VorbisSetupHeader, ParseError> {
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
    parse_setup_header_body(&mut reader)
}

/// Parse the round-5 portion of a Vorbis I setup-header **body** —
/// i.e. the bit-packed payload that follows the 7-byte common header
/// (§4.2.1). The caller must have positioned `reader` at the
/// `codebook_count_minus_1` byte.
///
/// On return, `reader` is positioned immediately after the last residue
/// header's per-classification book list. The caller can then proceed
/// (in a future round) to read the mapping list, mode list, and
/// framing flag.
pub fn parse_setup_header_body(
    reader: &mut BitReaderLsb<'_>,
) -> Result<VorbisSetupHeader, ParseError> {
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

    Ok(VorbisSetupHeader {
        codebooks,
        time_placeholders,
        floors,
        residues,
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

    /// Build the bit-packed body of a tiny well-formed setup header:
    /// one trivial codebook, one zero-valued time placeholder, one
    /// floor (type-tagged), one residue. Mappings + modes are NOT
    /// appended because round 5 stops after residues.
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
            .finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r).expect("must parse");
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
            .finish();
        let mut packet = Vec::with_capacity(7 + body.len());
        packet.push(SETUP_PACKET_TYPE);
        packet.extend_from_slice(&SETUP_PACKET_MAGIC);
        packet.extend_from_slice(&body);
        let setup = parse_setup_header(&packet).expect("must parse");
        assert_eq!(setup.codebooks.len(), 1);
    }

    #[test]
    fn rejects_short_packet() {
        let packet = [0u8; 3];
        match parse_setup_header(&packet) {
            Err(ParseError::PacketTooShort(3)) => {}
            other => panic!("expected PacketTooShort(3), got {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_packet_type() {
        let mut packet = vec![0u8; 7];
        packet[0] = 0x01; // identification header byte, not 0x05
        packet[1..7].copy_from_slice(&SETUP_PACKET_MAGIC);
        match parse_setup_header(&packet) {
            Err(ParseError::WrongPacketType(0x01)) => {}
            other => panic!("expected WrongPacketType(0x01), got {other:?}"),
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut packet = vec![0u8; 7];
        packet[0] = SETUP_PACKET_TYPE;
        packet[1..7].copy_from_slice(b"Vorbis"); // capitalised
        assert_eq!(parse_setup_header(&packet), Err(ParseError::BadMagic));
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
        match parse_setup_header_body(&mut r) {
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
        match parse_setup_header_body(&mut r) {
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
        match parse_setup_header_body(&mut r) {
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
        match parse_setup_header_body(&mut r) {
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
        b2 = b2.residue_count_minus_1(0).minimal_residue_type2();
        let bytes = b2.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r).expect("must parse");
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
            .finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r).expect("must parse");
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
    /// `time_count=1`, `floor_count=2`, `residue_count=2`).
    #[test]
    fn parses_two_floors_two_residues() {
        let bytes = SetupBuilder::new()
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
            .minimal_residue_type2()
            .finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r).expect("must parse");
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
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r).expect("must parse");
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
        let bytes = b.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let setup = parse_setup_header_body(&mut r).expect("must parse");
        assert_eq!(setup.residues[0].residue_type, 0);
    }
}

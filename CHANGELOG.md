# Changelog

All notable changes to `oxideav-vorbis` are recorded here.

## [Unreleased]

### Added

* **Vorbis I floor type 1 per-packet decode + curve computation (Vorbis
  I §7.2.3 "packet decode" + §7.2.4 "curve computation").** New module
  `floor1` exporting `Floor1Decoder`, `FloorCurve`, `Floor1Error`, and
  the integer geometry helpers `low_neighbor` / `high_neighbor` (§9.2.4 /
  §9.2.5), `render_point` (§9.2.6) and `render_line` (§9.2.7).
  `Floor1Decoder::new(&Floor1Header, &[VorbisCodebook])` reconstructs the
  full `[floor1_X_list]` (prepending the two implicit endpoints `0` and
  `2^rangebits` that §7.2.2 injects before the per-partition read loop),
  validates the §7.2.2 undecodability clauses (multiplier in `1..=4`,
  `[floor1_values]` ≤ 65, x-list uniqueness, master/sub-book indices in
  range) and pre-builds every referenced codebook's Huffman tree once.
  `Floor1Decoder::decode(&mut BitReaderLsb, n)` runs the two-stage
  decode: (1) §7.2.3 packet decode reads the `[nonzero]` flag (returning
  `FloorCurve::Unused` if clear), the two `ilog([range]-1)`-bit endpoint
  amplitudes, and per partition the master-book selector (only when
  `[cbits] > 0`) followed by the per-dimension sub-book scalar
  amplitudes (a negative/`None` sub-book yields a `0` Y with no bits
  read), producing `[floor1_Y]`; (2) §7.2.4 step 1 unwraps the positive
  `[floor1_Y]` differences through iterative `render_point` line
  prediction into `[floor1_final_Y]` + `[floor1_step2_flag]` (with the
  suggested `[0, range)` clamp), and step 2 sorts the `(X, final_Y,
  flag)` triples by ascending X, renders the contiguous integer line
  segments via `render_line`, and substitutes each integer floor sample
  through the §10.1 `floor1_inverse_dB_table` (transcribed verbatim as
  the 256-element `INVERSE_DB_TABLE`) to produce a linear-domain
  envelope of length `n`. An end-of-packet condition anywhere in §7.2.3
  is the spec's nominal occurrence: decode returns `FloorCurve::Unused`
  exactly as if `[nonzero]` had been clear. New error enum `Floor1Error`
  (`BookOutOfRange`, `BadMultiplier`, `TooManyValues`, `NonUniqueXList`,
  `Huffman`). Crate root re-exports `Floor1Decoder`, `FloorCurve`,
  `Floor1Error`, `low_neighbor`, `high_neighbor`, `render_point`,
  `render_line`; the unified `Error` grows a `Floor1(Floor1Error)`
  variant with `From` glue. 18 new unit tests cover the §9.2.6/§9.2.7
  render-point/render-line spec geometry (up/down/flat segments,
  endpoint-not-written convention), the §9.2.4/§9.2.5 neighbor lookups,
  all four §7.2.2 construction-validation rejections, a hand-traced full
  `curve_computation` (x_list `[0,16,4,8]`, Y `[40,20,1,0]` → final_Y
  `[40,20,34,30]`), a full packet→curve round trip, the master/subclass
  cascade selector path, the `nonzero` unset path, two end-of-packet
  nominal-`Unused` paths (mid-amplitude + mid-codeword), the
  negative-sub-book zero-Y path, and the §10.1 table endpoints. Test
  count: 146 total (128 → 146).

* **Vorbis I per-packet residue decode (Vorbis I §8.6.2 "packet decode"
  + §8.6.3/§8.6.4/§8.6.5 "format 0/1/2 specifics").** New module
  `residue` exporting `ResidueDecoder`. `ResidueDecoder::new(&ResidueHeader,
  &[VorbisCodebook])` validates the §8.6.1 undecodability clauses
  (classbook / value-book indices in range, value books carry a value
  mapping, classbook dimensions nonzero) and pre-builds the classbook +
  value-book Huffman trees once. `ResidueDecoder::decode(&mut
  BitReaderLsb, blocksize, &[bool])` runs the §8.6.2 packet decode: it
  caps `[residue_begin]`/`[residue_end]` to the per-format vector size
  (`blocksize/2` for format 0/1, `blocksize/2 * ch` for format 2),
  derives `classwords_per_codeword`/`n_to_read`/`partitions_to_read`,
  reads the per-partition classifications on pass 0 from the classbook in
  scalar context (descending integer-divide / integer-modulo unpack),
  and over passes 0..=7 decodes each partition's stage-`pass` value book
  in VQ context — *accumulating* (`+=`) into the output. Format 0
  (§8.6.3) scatters element `j` to `offset + i + j*step`; format 1
  (§8.6.4) appends contiguously; format 2 (§8.6.5) decodes one
  interleaved vector of length `ch * blocksize/2` as a format-1 decode
  then de-interleaves `v[i*ch + j] -> output[j][i]`, with the all-`do
  not decode` short-circuit. End-of-packet during decode is treated as
  the §8.6.2 nominal occurrence: decode stops and returns the
  vectors-so-far rather than erroring. New error enum `ResidueError`
  (`UnsupportedFormat`, `ClassbookOutOfRange`, `ValueBookOutOfRange`,
  `ValueBookHasNoLookup`, `ZeroClasswordsPerCodeword`,
  `Format0PartitionNotDivisible`, `Huffman`, `Vq`). Crate root re-exports
  `ResidueDecoder` + `ResidueError`; the unified `Error` grows a
  `Residue(ResidueError)` variant with `From` glue. 15 new unit tests
  cover all four §8.6.1 construction-validation rejections, the
  `n_to_read == 0` short-circuit, format-1 mono two-partition decode,
  `do not decode` zeroing/skip, format-0 interleaved scatter, format-2
  interleave/de-interleave + the all-`do not decode` no-op, two
  end-of-packet nominal-occurrence paths (empty buffer + mid-codeword
  overrun), the multi-classword descending classification unpack, and
  cascade-stage accumulation across passes. Test count: 128 total (113 →
  128).

* **Vorbis I VQ vector unpack (Vorbis I §3.2.1 "VQ lookup table vector
  representation" + §3.3 "Use of the codebook abstraction").** New
  module `vq` exporting `unpack_vector(&VorbisCodebook, lookup_offset:
  u32) -> Result<Vec<f32>, VqUnpackError>`. Implements the §3.2.1
  "Vector value decode: Lookup type 1" mixed-base permutation
  (`multiplicand_offset = (lookup_offset / index_divisor) mod
  codebook_lookup_values`, with `index_divisor` multiplied by
  `codebook_lookup_values` between iterations) and the §3.2.1 "Vector
  value decode: Lookup type 2" one-to-one direct slice
  (`multiplicand_offset = lookup_offset * codebook_dimensions`,
  incremented per iteration). Both paths honour `codebook_sequence_p`
  cumulatively: when set, `[last]` carries the full prior
  `value_vector[i]` (post-`minimum_value`, post-`delta_value`,
  post-`last`) forward, making the output a prefix-sum; when clear,
  `[last]` stays at `0.0` and each element is independent. New error
  enum `VqUnpackError` with variants `EntryOutOfRange`,
  `NoVectorForType0` (per §3.3 "requesting decode using a codebook of
  lookup type 0 in any context expecting a vector return value … is
  forbidden"), `ZeroDimensions`, and `MultiplicandShapeMismatch`. Crate
  root re-exports `unpack_vector` and `VqUnpackError`; the unified
  `Error` grows a `Vq(VqUnpackError)` variant with `From` glue. 16 new
  unit tests cover both lookup-type paths, both `sequence_p` modes,
  per-codebook §3.3 round-trips (an 8-entry / 8-dim tessellation and a
  9-entry / 2-dim lattice exhaustively check every entry against
  hand-computed reference vectors), the type-0 vector-rejection path,
  out-of-range lookup-offset rejection, zero-dimensions rejection, and
  multiplicand-shape mismatch detection for both lattice and
  tessellation. Test count: 113 total (97 → 113).

* **Vorbis I setup-header mapping + mode + framing-flag parse (Vorbis I
  §4.2.4 "Mappings" / "Modes").** `parse_setup_header` /
  `parse_setup_header_body` now walk the entire setup header, not just
  the codebook/time/floor/residue prefix. Both entry points now take an
  `audio_channels: u8` parameter (the value carried in the
  identification header per §4.2.2), required for the
  `ilog(audio_channels - 1)`-bit magnitude/angle channel reads in the
  square-polar coupling subblock and for the per-channel `mux[ch]`
  reads when `submaps > 1`. New types: `MappingHeader { mapping_type,
  submaps, coupling: Vec<MappingCouplingStep>, mux: Vec<u8>,
  submap_configs: Vec<MappingSubmap> }`, `MappingCouplingStep
  { magnitude_channel, angle_channel }`, `MappingSubmap
  { time_placeholder, floor, residue }`, `ModeHeader { blockflag,
  windowtype, transformtype, mapping }`. `VorbisSetupHeader` grows
  three fields: `mappings`, `modes`, `framing_flag`. New
  `setup::ParseError` variants enforcing the §4.2.4 invariants:
  `UnsupportedMappingType` (`mapping_type != 0` per step 2b),
  `BadCouplingChannels` (magnitude == angle, or either >=
  audio_channels — step 2c.ii), `NonZeroMappingReserved` (2-bit
  reserved — step 2c.iii), `BadMuxValue` (`mux[ch] >= submaps` —
  step 2c.iv.B), `BadSubmapFloor` (floor index >= `floor_count` —
  step 2c.v.C), `BadSubmapResidue` (residue index >= `residue_count` —
  step 2c.v.E), `NonZeroModeWindowType` / `NonZeroModeTransformType` /
  `BadModeMapping` (mode-section step 2e), `BadFramingFlag` (trailing
  flag unset — "Modes" step 3), `ZeroAudioChannels` (caller-side
  guard — §4.2.2 already guarantees > 0). Re-exports at the crate
  root: `MappingHeader`, `MappingCouplingStep`, `MappingSubmap`,
  `ModeHeader`. 14 new unit tests cover the trace-doc §6 / §7 shapes:
  stereo mapping with one coupling step (magnitude=0, angle=1), 5.1
  multi-submap mapping with `mux=[0,0,0,0,0,1]` and a per-submap
  (floor 0, residue 0) / (floor 1, residue 1) split, two
  one-mode-per-block setups, the §4.2.4 "Modes" `windowtype != 0` /
  `transformtype != 0` / OOB-`mapping` rejections, the
  `audio_channels = 0` caller-bug guard, and the unset-framing-flag
  reject (97 tests total, up from 83).

* **Vorbis I setup-header outer walker (Vorbis I §4.2.4).**
  `parse_setup_header(&[u8])` validates the 7-byte common header
  (`0x05 "vorbis"` per §4.2.1) and `parse_setup_header_body(&mut
  BitReaderLsb)` consumes the bit-packed body that follows. The
  round-5 walker reads the first four sub-lists of the setup header:
  (1) `vorbis_codebook_count` codebook configurations via
  `parse_codebook` (§3.2.1); (2) `vorbis_time_count` 16-bit
  time-domain transform placeholders, each spec-mandated to equal zero
  (§4.2.4 step 2); (3) `vorbis_floor_count` floor headers, each
  prefixed by a 16-bit `floor_type` that dispatches to the §6.2.1
  (LSP, type 0) or §7.2.2 (piecewise-linear, type 1) structural-field
  reader — no per-packet curve decode; (4) `vorbis_residue_count`
  residue headers (§8.6.1 common layout: `residue_begin`,
  `residue_end`, `residue_partition_size`, `residue_classifications`,
  `residue_classbook`, the per-classification cascade bitmap with the
  `low_bits / bitflag / high_bits` encoding, and the conditional
  per-stage book reads). Mappings (§4.3), modes (§4.3.1), and the
  trailing framing flag are deferred to round 6; the bit reader is
  left positioned immediately after the last residue's book list.
  Spec-mandated errors surface as `setup::ParseError` variants
  (`PacketTooShort`, `WrongPacketType`, `BadMagic`, `Codebook` —
  wrapping the inner codebook parse error — `NonZeroTimePlaceholder`,
  `UnsupportedFloorType`, `UnsupportedResidueType`,
  `UnexpectedEndOfPacket`). Re-exports at the crate root:
  `parse_setup_header`, `parse_setup_header_body`,
  `VorbisSetupHeader`, `FloorHeader`, `FloorKind`, `Floor0Header`,
  `Floor1Header`, `Floor1Class`, `ResidueHeader`, `SetupParseError`,
  `SETUP_PACKET_TYPE`, `SETUP_PACKET_MAGIC`. 14 new unit tests cover
  the minimal one-of-each setup body, a full setup packet with the
  7-byte common header prefix, packet-length / packet-type / magic
  rejections, nonzero time-placeholder rejection, floor-type > 1
  rejection, residue-type > 2 rejection, truncated-packet EOF
  surfaced through `Codebook` wrapping, floor 1 with `partitions = 0`
  (empty-classes corner case), floor 0 (LSP) setup, two-floor /
  two-residue stereo shape mirroring trace-doc §2.4, multi-stage
  residue cascade exercising `bitflag = 1` high bits, and residue
  type 0 (83 tests total, up from 69).

* **Vorbis I canonical Huffman tree builder + entry decoder (Vorbis I
  §3.2.1).** `HuffmanTree::from_codebook(&VorbisCodebook)` /
  `HuffmanTree::from_lengths(&[u8])` builds a canonical Vorbis decision
  tree from the per-entry codeword-length table returned by
  `parse_codebook`. Construction uses an open-position deque
  (left-to-right) that pops the leftmost slot, splits it down to the
  required depth, and places a leaf — realising the §3.2.1 "lowest
  valued unused canonical codeword" assignment without ever
  materialising codeword integers. The spec's worked example (lengths
  `[2 4 4 4 4 2 3 3]` → codewords `00 0100 0101 0110 0111 10 110 111`)
  round-trips bit-exactly. `HuffmanTree::decode_entry(&mut
  BitReaderLsb)` walks the tree against an LSb-first packet bitstream
  (first bit = MSb of canonical codeword) and returns the codebook
  entry index — the underlying primitive for §3.3 scalar codebook reads
  and §3.4 / §8.6 VQ codebook reads. Errata 20150226 single-entry
  codebooks: a codebook with exactly one used entry whose recorded
  length is `1` yields a 2-node tree (`root.left == root.right == sole
  leaf`) so `decode_entry` returns the sole entry and sinks one bit
  for either `0` or `1` (per the errata "decoders should tolerate that
  the bit read from the stream be '1' instead of '0'"); single-entry
  codebooks with `length != 1` are rejected. Underspecified /
  overspecified trees (§3.2.1) surface as
  `HuffmanBuildError::{UnderspecifiedTree, OverspecifiedTree}`;
  out-of-range lengths (`1..=32`) and zero-used-entries surface as
  `InvalidLength` / `EmptyTree`. End-of-packet mid-codeword surfaces
  as `HuffmanDecodeError::UnexpectedEndOfPacket` per §3.3 "reading
  past the end of a packet propagates the 'end-of-stream' condition
  to the decoder". 13 new unit tests cover the spec worked example,
  concatenated stream decode, sparse codebook with entry index
  `0xFF`, single-entry both-bits tolerance, single-entry length≠1
  rejection, fully-unused rejection, underspecified / overspecified /
  invalid-length rejection, EOF on truncated input, balanced 16-entry
  length-4 round-trip, max-length-32 entries, and `from_codebook` ↔
  `from_lengths` equivalence (69 tests total, up from 56).

* **Vorbis I codebook-header parser (Vorbis I §3.2.1).**
  `parse_codebook(&mut BitReaderLsb)` decodes a single Vorbis I
  codebook header — the most foundational sub-structure of the
  setup header (§4.2.4). The function reads the 24-bit `0x564342`
  sync pattern, 16-bit `dimensions`, 24-bit `entries`, 1-bit
  `ordered` flag, then the codeword-length table (dense, sparse,
  or ordered run-length encoded per §3.2.1), the 4-bit
  `lookup_type`, and for `lookup_type ∈ {1, 2}` the 32-bit
  `minimum_value` + 32-bit `delta_value` (both passed through
  `float32_unpack` per §9.2.2), `value_bits - 1` (4 bits),
  `sequence_p` (1 bit), and the raw multiplicand table (size
  `lookup1_values(entries, dimensions)` for type 1 per §9.2.3 or
  `entries × dimensions` for type 2). Returns
  `VorbisCodebook { dimensions, entries, codeword_lengths,
  lookup: VqLookup }`. Sparse unused entries are stored as the
  `UNUSED_ENTRY` sentinel (0); spec-legal lengths are 1..=32.
  Spec-mandated errors surface as `codebook::ParseError` variants
  (`BadSyncPattern`, `ZeroEntries`, `OrderedOverflow`,
  `ReservedLookupType`, `UnexpectedEndOfPacket`). 18 unit tests
  cover the spec's worked example (§3.2.1 lengths 2 4 4 4 4 2 3
  3), sparse encoding with unused entries, ordered run-length
  encoding, lookup types 0 / 1 / 2 with synthetic multiplicand
  tables, the `lookup1_values(2916, 8) = 2` corner from
  trace-doc §3, the spec examples for `ilog` (§9.2.1) and
  `float32_unpack` (§9.2.2), and every documented `ParseError`
  failure mode. The §9.2.1 / §9.2.2 / §9.2.3 helpers are also
  exposed at the module root for downstream use by the upcoming
  Huffman-tree builder and the floor / residue parsers.

* **Vorbis I comment-header parser (Vorbis I §5).**
  `parse_comment_header(&[u8])` decodes a Vorbis I comment-header
  packet (common header `0x03 "vorbis"` + §5.2.1 body) and returns a
  `VorbisCommentHeader { vendor: String, comments: Vec<String> }`.
  The body is read per §5.2.1 (byte-aligned `u32` LE
  `vendor_length` + UTF-8 vendor + `u32` LE
  `user_comment_list_length` + per-comment `u32` LE length + UTF-8
  bytes + 1-bit framing flag). All spec-mandated checks are
  enforced and surface as `comment::ParseError` variants
  (`PacketTooShort`, `WrongPacketType`, `BadMagic`,
  `UnexpectedEndOfPacket` — see §4.2 for the comment-header's
  "non-fatal" relaxation —, `LengthOverflow`, `InvalidVendorUtf8`,
  `InvalidCommentUtf8`, `BadFramingFlag`). Helpers
  `VorbisCommentHeader::key_value_iter` and `split_key_value` parse
  the §5.2.2 `KEY=value` shape. 22 unit tests cover the
  `mono-44100-q5-typical` (1-entry encoder tag) and
  `with-vorbis-comment-tags` (7-entry title/artist/album/date/genre/
  tracknumber + encoder) fixture shapes, the historical
  `Xiph.Org libVorbis I 20020717` vendor string, empty vendor + empty
  comment list, multi-byte UTF-8 in vendor and comment values,
  duplicate-key entries per §5.2.2, a 64 KiB
  `METADATA_BLOCK_PICTURE`-sized payload, framing-byte padding per
  §2.1.8, and every documented `ParseError` failure mode.

* **Vorbis I identification-header parser (Vorbis I §4.2.2).**
  `parse_identification_header(&[u8])` decodes a 30-byte
  identification-header packet and returns a
  `VorbisIdentificationHeader` struct exposing `vorbis_version`,
  `audio_channels`, `audio_sample_rate`, the three signed bitrate
  hints, and the two resolved block sizes. All spec-mandated
  invariants (packet type byte, `"vorbis"` magic per §4.2.1,
  `vorbis_version == 0`, nonzero channels and sample rate, blocksize
  exponents in 6..=13, `blocksize_0 <= blocksize_1`, framing flag
  nonzero) are enforced and surface as `ParseError` variants. 16 unit
  tests cover both the happy-path field-shape equivalents of the
  trace-doc fixtures (`mono-44100-q5-typical`,
  `5.1-channel-48000-q5`, spec-minimum and -maximum blocksizes,
  signed bitrate-hint encoding) and every documented `ParseError`
  failure mode.

### Changed

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule. Orphan-master rebuild per workspace
  policy; no `old` branch retained. License also reset to clean MIT.

  Round 1 lands the identification-header parse only. Comment header
  (§5), setup header (§4.2.4), codebook/floor/residue/mapping/mode
  decode (§3, §6, §7, §8, §4.3) and audio-packet decode (§4.3.2) are
  pending in later rounds; `decode_packet` returns
  `Error::NotImplemented`. The Ogg framing layer (RFC 3533 + Vorbis
  I §A) is also not yet wired up.

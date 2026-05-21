# oxideav-vorbis

Pure-Rust Vorbis I audio codec — clean-room rebuild, round 6.

## Status — 2026-05-22

**Round 6 landed: the full setup-header walker now closes — mapping
configurations (§4.2.4 "Mappings", `mapping_type = 0` only), mode
configurations (§4.2.4 "Modes"), and the trailing framing flag are
parsed.** Round 5 landed the setup-header outer walker (codebooks +
time-domain placeholders + floor headers + residue headers); round 4
the canonical Huffman tree builder + entry decoder (§3.2.1); round 3
the codebook-header parser (§3.2.1); round 2 the comment-header parser
(§5); round 1 the identification-header parser (§4.2.2) on 2026-05-20.

The crate's prior implementation was retired under the workspace
clean-room policy because module-level docstrings and inline comments
referenced libvorbis internals as their provenance source. The current
master is an orphan rebuild that started from a `NotImplemented`
scaffold; round 1 lands the identification header, round 2 the
comment header.

### What works

* [`parse_identification_header`] reads a 30-byte Vorbis I
  identification-header packet (Vorbis I §4.2.2) and returns a
  [`VorbisIdentificationHeader`] struct exposing:
  * `vorbis_version` (u32; must be 0 for Vorbis I)
  * `audio_channels` (u8; must be > 0)
  * `audio_sample_rate` (u32, Hz; must be > 0)
  * `bitrate_maximum` / `bitrate_nominal` / `bitrate_minimum`
    (signed i32 hints; 0 means "unset" per §4.2.2)
  * `blocksize_0` / `blocksize_1` (u16, sample counts; one of
    {64, 128, 256, 512, 1024, 2048, 4096, 8192} per §4.2.2, with
    `blocksize_0 <= blocksize_1`)
* All spec-mandated validity checks from §4.2.2 are enforced: packet
  length, common-header magic + packet type, nonzero
  channels/sample_rate, blocksize exponents in 6..=13, blocksize
  ordering, framing flag nonzero.
* [`parse_comment_header`] reads a Vorbis I comment-header packet
  (§5.2.1 / §5.2.3) and returns a [`VorbisCommentHeader`] exposing:
  * `vendor` (UTF-8 String; the encoder's vendor identification, e.g.
    `"Lavf61.7.100"` or `"Xiph.Org libVorbis I 20020717"`)
  * `comments` (Vec<String>; raw UTF-8 entries in `KEY=value` form
    per §5.2.2, preserving insertion order)
* All spec-mandated invariants from §5.2.1 are enforced: common-header
  packet type (`0x03`) + `"vorbis"` magic, UTF-8 validation of the
  vendor and every comment entry, length-prefix overflow guard,
  truncation reported as a structured `UnexpectedEndOfPacket` so
  callers can apply the §4.2 "non-fatal" relaxation if desired,
  framing-bit check.
* Helpers: [`split_key_value`] cuts an entry on the first `=` octet
  per §5.2.2; [`VorbisCommentHeader::key_value_iter`] yields
  `(key, value)` pairs and skips malformed (no-`=`) entries.
* [`parse_codebook`] reads a Vorbis I codebook header (§3.2.1) from
  an LSB-first [`oxideav_core::bits::BitReaderLsb`] positioned at
  the 24-bit `0x564342` sync pattern. Returns a [`VorbisCodebook`]
  exposing:
  * `dimensions` (u16; 16-bit unsigned)
  * `entries` (u32; 24-bit unsigned)
  * `codeword_lengths` (Vec<u8>; per-entry length in `1..=32`, or
    [`UNUSED_ENTRY`]`= 0` for sparse-codebook unused entries)
  * `lookup` ([`VqLookup`]; `None` for `lookup_type=0`, `Lattice`
    for `lookup_type=1`, `Tessellation` for `lookup_type=2`,
    carrying the unpacked `minimum_value` / `delta_value` floats
    (§9.2.2), `value_bits` in `1..=16`, `sequence_p` flag, and the
    raw multiplicand table)
* Codebook helpers exposed at the crate root: [`ilog`] (§9.2.1),
  [`float32_unpack`] (§9.2.2), [`lookup1_values`] (§9.2.3). The
  ordered length-encoding branch (§3.2.1 step 3 ordered subcase)
  reuses `ilog(entries - current_entry)` for the per-run width.
* All §3.2.1 invariants are enforced and surface as
  `codebook::ParseError` variants (`BadSyncPattern`, `ZeroEntries`,
  `OrderedOverflow`, `ReservedLookupType`, `UnexpectedEndOfPacket`).
* [`HuffmanTree::from_codebook`] / [`HuffmanTree::from_lengths`]
  builds a canonical Vorbis I Huffman decision tree (§3.2.1) from the
  per-entry `codeword_lengths` table. Construction uses a left-to-right
  open-position deque: each used entry pops the leftmost open slot and
  either places a leaf there or splits it down to the entry's recorded
  depth (allocating internal nodes + pushing both new children to the
  deque front). The spec's worked example (lengths `[2 4 4 4 4 2 3 3]`
  → codewords `00 0100 0101 0110 0111 10 110 111`) round-trips
  exactly.
* [`HuffmanTree::decode_entry`] walks the tree against an LSb-first
  [`BitReaderLsb`]: each read bit selects `left` (0) or `right` (1)
  until a leaf is hit, then returns the leaf's entry index. The first
  bit read is the MSb of the canonical codeword per the §3.2.1 "the
  leftmost bit is the MSb" convention; under §2.1.4 LSb-first packing
  this means the encoder writes codewords with their MSb going into
  the LSb of the next stream byte.
* Errata 20150226 single-entry codebooks: a codebook with exactly one
  used entry whose recorded length is `1` builds a synthetic tree
  whose root's `left` and `right` both point at the same leaf, so
  `decode_entry` returns that entry and sinks one bit on either a `0`
  or a `1` (per the errata "decoders should tolerate that the bit
  read from the stream be '1' instead of '0'"). Single-entry
  codebooks with `length != 1` are rejected with
  [`HuffmanBuildError::SingleEntryWrongLength`].
* Underspecified / overspecified detection (§3.2.1 errata 20150226):
  any leftover open deque positions after every used entry has been
  placed → [`HuffmanBuildError::UnderspecifiedTree`]; popping the
  deque dry before all entries are placed →
  [`HuffmanBuildError::OverspecifiedTree`]. Out-of-range lengths
  (anything outside `1..=32`) surface as
  [`HuffmanBuildError::InvalidLength`], and zero used entries as
  [`HuffmanBuildError::EmptyTree`].
* [`parse_setup_header`] / [`parse_setup_header_body`] now walk the
  **entire** Vorbis I setup-header packet (§4.2.4). Both entry points
  take the stream's `audio_channels` (from the identification header)
  because the mapping decode reads channel-number widths of
  `ilog(audio_channels - 1)` bits:
  * `vorbis_codebook_count` codebook configurations (delegated to
    [`parse_codebook`]; §3.2.1).
  * `vorbis_time_count` 16-bit time-domain transform placeholders —
    each spec-mandated to equal zero (§4.2.4 step 2; any nonzero value
    is rejected with `SetupParseError::NonZeroTimePlaceholder`).
  * `vorbis_floor_count` floor headers. Each carries a 16-bit
    `floor_type`; type 0 (§6.2.1) and type 1 (§7.2.2) decode their
    structural fields (no per-packet curve decode), type > 1 is
    rejected. [`Floor1Header`] exposes `partitions`,
    `partition_class_list`, `classes` (per-class `dimensions`,
    `subclasses`, optional `masterbook`, `subclass_books`),
    `multiplier`, `rangebits`, `x_list`. [`Floor0Header`] exposes
    `order`, `rate`, `bark_map_size`, `amplitude_bits`,
    `amplitude_offset`, `book_list`.
  * `vorbis_residue_count` residue headers (§8.6.1 — common header
    layout across types 0/1/2). [`ResidueHeader`] exposes
    `residue_type` (0/1/2; >2 is rejected), `residue_begin`,
    `residue_end`, `partition_size`, `classifications`, `classbook`,
    `cascade[classifications]`, and `books[classifications][8]` with
    `None` entries for cascade-bit-unset stages.
  * `vorbis_mapping_count` mapping configurations (§4.2.4 "Mappings",
    `mapping_type = 0` only — any other type is rejected per step 2b).
    [`MappingHeader`] exposes `mapping_type`, `submaps` (1 or 4-bit
    encoded), `coupling` (per-step `(magnitude_channel, angle_channel)`
    pairs read at `ilog(audio_channels - 1)` bits each, with the spec's
    "magnitude != angle and both < audio_channels" validation), `mux`
    (per-channel submap routing; only present when `submaps > 1`), and
    `submap_configs` (the 8-bit time placeholder + the 8-bit floor /
    residue indices, both range-checked against the prior counts).
  * `vorbis_mode_count` mode configurations (§4.2.4 "Modes").
    [`ModeHeader`] exposes `blockflag`, `windowtype` (forced to 0 per
    step 2e), `transformtype` (forced to 0 per step 2e), and `mapping`
    (range-checked against `mapping_count`).
  * The trailing 1-bit framing flag (§4.2.4 "Modes" step 3) is read
    and required to be set; the flag value is also surfaced as
    `VorbisSetupHeader::framing_flag` for downstream inspection.
* 97 unit tests in total: 16 cover §4.2.2, 22 cover §5, 18 cover §3
  codebook-header parse, 13 cover §3.2.1 Huffman tree, and **28 cover
  §4.2.4 setup-header walker** — the round-5 set (minimal one-of-each
  setup body, full packet with common header, packet-length /
  packet-type / magic rejection, nonzero time-placeholder rejection,
  floor-type > 1 rejection, residue-type > 2 rejection, truncated
  packet, floor 1 with `partitions=0`, floor 0 LSP setup, two-floor /
  two-residue shape, multi-stage residue cascade, residue type 0) plus
  the **14 new round-6 tests** (stereo coupling mapping, 5.1 multi-submap
  mapping with `mux=[0,0,0,0,0,1]`, `mapping_type != 0` rejection,
  coupling-step magnitude==angle rejection, coupling-step OOB magnitude
  rejection, 2-bit reserved-field rejection, `mux` OOB rejection, submap
  floor / residue OOB rejections, mode `windowtype != 0` /
  `transformtype != 0` / OOB-`mapping` rejections, unset framing-flag
  rejection, `audio_channels = 0` caller-bug guard).

### What does not yet work

* Vector-decode wrapper applying the lookup table to a Huffman entry
  (§3.3 "VQ context": entry → vector via `multiplicands` +
  `minimum_value` / `delta_value` / `sequence_p`). The round-4 walker
  returns the raw entry index; vector unpacking is a followup.
* Floor curve / residue decode runtime (§7.2.3, §7.2.4, §8.6.2..§8.6.5),
  mapping submap channel routing at packet time, mode windowing.
* Audio-packet decode (§4.3): mode / window decode, floor curve,
  residue, channel coupling, inverse MDCT, overlap-add.
* Ogg framing (RFC 3533 + Vorbis I §A) — the parsers are currently
  bring-your-own-packet. Consuming an Ogg-encapsulated stream needs
  to be wired up via `oxideav-ogg`.
* `METADATA_BLOCK_PICTURE` base64 + FLAC-PICTURE block decoding (see
  trace-doc §2.3) — the comment parser returns the raw base64 string
  as a comment value; FLAC-PICTURE decoding belongs in a higher-level
  consumer.
* No [`oxideav_core::Decoder`] / [`oxideav_core::Encoder`] is
  registered yet; `decode_packet` returns `Error::NotImplemented`.

## Clean-room sources

Rounds 1 — 6 were implemented against, and only against:

* `docs/audio/vorbis/Vorbis_I_spec.pdf` — Xiph.Org Vorbis I
  Specification, 2020-07-04 revision. Round 1 used §2 Bitpacking
  Convention, §4.2.1 Common header decode, §4.2.2 Identification
  header. Round 2 used §4.2 (end-of-packet handling), §4.2.1, §4.2.3,
  §5.1, §5.2.1 (structure / decoder pseudocode), §5.2.2 (content
  vector format), §5.2.3 (encoder-side recap). Round 3 used §3.1
  (codebook overview), §3.2.1 (codebook decode algorithm), §9.2.1
  (`ilog`), §9.2.2 (`float32_unpack`), §9.2.3 (`lookup1_values`).
  Round 4 used §3.2.1 "Huffman decision tree representation" (the
  worked-example codeword table for lengths `[2 4 4 4 4 2 3 3]` plus
  the underspecified / overspecified discussion), the §3.2.1 errata
  20150226 "Single entry codebooks" addendum, and §3.3 "Use of the
  codebook abstraction" (decode-time bit-walking semantics + the
  end-of-packet condition). Round 5 used §4.2.4 "Setup header"
  (Codebooks / Time domain transforms / Floors / Residues subsections,
  the "Time domain transforms" must-be-zero rejection, the floor-type
  > 1 and residue-type > 2 rejections, and the in-spec floor 0 / floor
  1 / residue branches), §6.2.1 "Floor 0 header decode", §7.2.2 "Floor
  1 header decode" (steps 1..23), and §8.6.1 "Residue header decode"
  (the begin/end/partition_size/classifications/classbook fields, the
  per-classification cascade bitmap with `high_bits = read 5 bits if
  bitflag`, and the conditional per-stage book reads). Round 6 used
  §4.2.4 "Mappings" (steps 1..2 inclusive of the `mapping_type != 0`
  rejection, the optional submaps and square-polar coupling subblocks,
  the `ilog(audio_channels - 1)`-bit magnitude/angle channel reads,
  the "magnitude != angle and both < audio_channels" validation
  paragraph, the 2-bit reserved field, the `mux[ch]` reads gated on
  `submaps > 1` with the OOB check, and the per-submap placeholder +
  floor + residue index reads with their respective OOB checks),
  §4.2.4 "Modes" (the per-mode blockflag / windowtype / transformtype /
  mapping reads plus step 2e's `windowtype == 0`, `transformtype == 0`,
  and `mapping < mapping_count` enforcement), and §4.2.4 "Modes"
  step 3 (the trailing framing-flag requirement). §9.2.1 `ilog` is
  reused via [`crate::codebook::ilog`].
* `docs/audio/vorbis/vorbis-fixtures-and-traces.md` — clean-room
  trace-corpus document. Round 2 referenced §2.2 (`mono-44100-q5-typical`
  and `with-vorbis-comment-tags` `VORBIS_HEADER_COMMENT` /
  `VORBIS_COMMENT_ENTRY` shapes), §2.3 (`METADATA_BLOCK_PICTURE`
  convention), §9 (trace event vocabulary). Round 3 referenced §3
  (the `CODEBOOK` event shape — `book_idx dimensions entries
  ordered sparse lookup_type value_bits sequence_p` field set).
  Round 5 referenced §2.4 (`VORBIS_HEADER_SETUP` event shape —
  `codebook_count`, `time_count`, `floor_count`, `residue_count`,
  `mapping_count`, `mode_count`, `framing_flag`), §4 (`FLOOR_CONFIG`
  field shape — `floor_idx`, `type`, plus `partitions`,
  `multiplier`, `rangebits`, `x_list_count` for type 1), and §5
  (`RESIDUE_CONFIG` field shape — `residue_idx`, `type`, `begin`,
  `end`, `partition_size`, `classifications`, `classbook`). Round 6
  referenced §6 (`MAPPING_CONFIG` shape — `mapping_idx`, `type`,
  `submaps`, `coupling_steps`, `magnitude`, `angle`, `floor`,
  `residue`; including the §6 narrative confirming "stereo libvorbis
  output always uses one coupling step `(magnitude=L, angle=R)`,
  mono streams have `coupling_steps=0` and `submaps=1`, 5.1 streams
  use `submaps=2` with `mux=[0,0,0,0,0,1]` routing the LFE on its
  own submap") and §7 (`MODE_CONFIG` shape — `mode_idx`,
  `blockflag`, `windowtype`, `transformtype`, `mapping`, plus the
  trace narrative that mode 0 is the short-block mode and mode 1
  is the long-block mode in every libvorbis stream).
* `docs/audio/vorbis/fixtures/{mono-44100-q5-typical,with-vorbis-comment-tags}/trace.txt`
  — only as the source for the field-level shape of the test
  fixtures.

No external library source was consulted, quoted, paraphrased, or
used as a cross-check oracle, in accordance with the workspace
clean-room policy. The forbidden-list for this crate includes every
existing Vorbis implementation (the Xiph reference encoder/decoder,
Tremor, lewton, FFmpeg's Vorbis decoder, every third-party Rust
crate that wraps or implements the format).

[`parse_identification_header`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/identification/fn.parse_identification_header.html
[`VorbisIdentificationHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/identification/struct.VorbisIdentificationHeader.html
[`parse_comment_header`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/comment/fn.parse_comment_header.html
[`VorbisCommentHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/comment/struct.VorbisCommentHeader.html
[`split_key_value`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/comment/fn.split_key_value.html
[`VorbisCommentHeader::key_value_iter`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/comment/struct.VorbisCommentHeader.html#method.key_value_iter
[`parse_codebook`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/fn.parse_codebook.html
[`VorbisCodebook`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/struct.VorbisCodebook.html
[`VqLookup`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/enum.VqLookup.html
[`UNUSED_ENTRY`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/constant.UNUSED_ENTRY.html
[`ilog`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/fn.ilog.html
[`float32_unpack`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/fn.float32_unpack.html
[`lookup1_values`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/fn.lookup1_values.html
[`HuffmanTree::from_codebook`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/struct.HuffmanTree.html#method.from_codebook
[`HuffmanTree::from_lengths`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/struct.HuffmanTree.html#method.from_lengths
[`HuffmanTree::decode_entry`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/struct.HuffmanTree.html#method.decode_entry
[`HuffmanBuildError::SingleEntryWrongLength`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/enum.BuildError.html#variant.SingleEntryWrongLength
[`HuffmanBuildError::UnderspecifiedTree`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/enum.BuildError.html#variant.UnderspecifiedTree
[`HuffmanBuildError::OverspecifiedTree`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/enum.BuildError.html#variant.OverspecifiedTree
[`HuffmanBuildError::InvalidLength`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/enum.BuildError.html#variant.InvalidLength
[`HuffmanBuildError::EmptyTree`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/enum.BuildError.html#variant.EmptyTree
[`BitReaderLsb`]: https://docs.rs/oxideav-core/latest/oxideav_core/bits/struct.BitReaderLsb.html
[`oxideav_core::bits::BitReaderLsb`]: https://docs.rs/oxideav-core/latest/oxideav_core/bits/struct.BitReaderLsb.html
[`oxideav_core::Decoder`]: https://docs.rs/oxideav-core/latest/oxideav_core/trait.Decoder.html
[`oxideav_core::Encoder`]: https://docs.rs/oxideav-core/latest/oxideav_core/trait.Encoder.html
[`parse_setup_header`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/fn.parse_setup_header.html
[`parse_setup_header_body`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/fn.parse_setup_header_body.html
[`Floor0Header`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/struct.Floor0Header.html
[`Floor1Header`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/struct.Floor1Header.html
[`ResidueHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/struct.ResidueHeader.html
[`MappingHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/struct.MappingHeader.html
[`ModeHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/struct.ModeHeader.html

## License

MIT. See `LICENSE`.

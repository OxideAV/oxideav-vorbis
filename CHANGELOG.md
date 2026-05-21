# Changelog

All notable changes to `oxideav-vorbis` are recorded here.

## [Unreleased]

### Added

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

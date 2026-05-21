# Changelog

All notable changes to `oxideav-vorbis` are recorded here.

## [Unreleased]

### Added

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

# Changelog

All notable changes to `oxideav-vorbis` are recorded here.

## [Unreleased]

### Added

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

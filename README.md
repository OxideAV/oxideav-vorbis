# oxideav-vorbis

Pure-Rust Vorbis I audio codec — clean-room rebuild, round 2.

## Status — 2026-05-21

**Round 2 landed: comment-header parser (Vorbis I §5).** Round 1
landed the identification-header parser (Vorbis I §4.2.2) on
2026-05-20; that work remains the only audio-relevant decode on
master so far.

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
* 38 unit tests in total: 16 cover §4.2.2 (identification header),
  22 cover §5 (comment header) including the
  `mono-44100-q5-typical` and `with-vorbis-comment-tags` corpus
  shapes, historical libvorbis vendor, empty/UTF-8/duplicate-key
  cases, large (64 KiB) `METADATA_BLOCK_PICTURE`-sized payloads, and
  every documented `ParseError` variant.

### What does not yet work

* Setup header (§4.2.4) — codebooks (§3), floors (§6, §7), residues
  (§8), mappings (§4.3), modes (§4.3.1).
* Audio-packet decode (§4.3.2): mode / window decode, floor curve,
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

Rounds 1 and 2 were implemented against, and only against:

* `docs/audio/vorbis/Vorbis_I_spec.pdf` — Xiph.Org Vorbis I
  Specification, 2020-07-04 revision. Round 1 used §2 Bitpacking
  Convention, §4.2.1 Common header decode, §4.2.2 Identification
  header. Round 2 used §4.2 (end-of-packet handling), §4.2.1, §4.2.3,
  §5.1, §5.2.1 (structure / decoder pseudocode), §5.2.2 (content
  vector format), §5.2.3 (encoder-side recap).
* `docs/audio/vorbis/vorbis-fixtures-and-traces.md` — clean-room
  trace-corpus document. Round 2 referenced §2.2 (`mono-44100-q5-typical`
  and `with-vorbis-comment-tags` `VORBIS_HEADER_COMMENT` /
  `VORBIS_COMMENT_ENTRY` shapes), §2.3 (`METADATA_BLOCK_PICTURE`
  convention), §9 (trace event vocabulary).
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
[`oxideav_core::Decoder`]: https://docs.rs/oxideav-core/latest/oxideav_core/trait.Decoder.html
[`oxideav_core::Encoder`]: https://docs.rs/oxideav-core/latest/oxideav_core/trait.Encoder.html

## License

MIT. See `LICENSE`.

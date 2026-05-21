# oxideav-vorbis

Pure-Rust Vorbis I audio codec — clean-room rebuild, round 3.

## Status — 2026-05-21

**Round 3 landed: codebook-header parser (Vorbis I §3.2.1).** Round 2
landed the comment-header parser (§5) on 2026-05-21; round 1 landed
the identification-header parser (§4.2.2) on 2026-05-20.

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
* 56 unit tests in total: 16 cover §4.2.2, 22 cover §5, 18 cover §3
  (sync, dense / sparse / ordered lengths, lookup types 0 / 1 / 2,
  multiplicand decode, `ilog` / `float32_unpack` / `lookup1_values`
  spec examples, and every documented `ParseError` failure mode).

### What does not yet work

* Setup-header outer walker (§4.2.4) — wires multiple codebooks
  together plus floors / residues / mappings / modes.
* Huffman tree construction from `VorbisCodebook::codeword_lengths`
  and codeword decode from the audio packet bitstream (§3.2.1
  "Huffman decision tree representation" + §3.3 "Use of the codebook
  abstraction"). The round-3 parser returns the raw length table; a
  tree builder + walker is the round-4 followup.
* Floors (§6, §7), residues (§8), mappings (§4.3), modes (§4.3.1).
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

Rounds 1, 2 and 3 were implemented against, and only against:

* `docs/audio/vorbis/Vorbis_I_spec.pdf` — Xiph.Org Vorbis I
  Specification, 2020-07-04 revision. Round 1 used §2 Bitpacking
  Convention, §4.2.1 Common header decode, §4.2.2 Identification
  header. Round 2 used §4.2 (end-of-packet handling), §4.2.1, §4.2.3,
  §5.1, §5.2.1 (structure / decoder pseudocode), §5.2.2 (content
  vector format), §5.2.3 (encoder-side recap). Round 3 used §3.1
  (codebook overview), §3.2.1 (codebook decode algorithm), §9.2.1
  (`ilog`), §9.2.2 (`float32_unpack`), §9.2.3 (`lookup1_values`).
* `docs/audio/vorbis/vorbis-fixtures-and-traces.md` — clean-room
  trace-corpus document. Round 2 referenced §2.2 (`mono-44100-q5-typical`
  and `with-vorbis-comment-tags` `VORBIS_HEADER_COMMENT` /
  `VORBIS_COMMENT_ENTRY` shapes), §2.3 (`METADATA_BLOCK_PICTURE`
  convention), §9 (trace event vocabulary). Round 3 referenced §3
  (the `CODEBOOK` event shape — `book_idx dimensions entries
  ordered sparse lookup_type value_bits sequence_p` field set).
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
[`oxideav_core::bits::BitReaderLsb`]: https://docs.rs/oxideav-core/latest/oxideav_core/bits/struct.BitReaderLsb.html
[`oxideav_core::Decoder`]: https://docs.rs/oxideav-core/latest/oxideav_core/trait.Decoder.html
[`oxideav_core::Encoder`]: https://docs.rs/oxideav-core/latest/oxideav_core/trait.Encoder.html

## License

MIT. See `LICENSE`.

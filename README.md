# oxideav-vorbis

Pure-Rust Vorbis I audio codec — clean-room rebuild, round 1.

## Status — 2026-05-20

**Round 1 landed: identification-header parser (Vorbis I §4.2.2).**

The crate's prior implementation was retired under the workspace
clean-room policy because module-level docstrings and inline comments
referenced libvorbis internals as their provenance source. The current
master is an orphan rebuild that started from a `NotImplemented`
scaffold; round 1 lands the first piece of the bitstream.

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
* 16 unit tests cover happy-path parses against shape-equivalents of
  the corpus fixtures `mono-44100-q5-typical`, `5.1-channel-48000-q5`,
  spec-minimum (`64/64`) and spec-maximum (`8192`) blocksizes, signed
  bitrate-hint encoding, and every documented `ParseError` failure
  mode.

### What does not yet work

* Comment header (Vorbis I §5).
* Setup header (§4.2.4) — codebooks (§3), floors (§6, §7), residues
  (§8), mappings (§4.3), modes (§4.3.1).
* Audio-packet decode (§4.3.2): mode / window decode, floor curve,
  residue, channel coupling, inverse MDCT, overlap-add.
* Ogg framing (RFC 3533 + Vorbis I §A) — the parser is currently
  bring-your-own-packet. Consuming an Ogg-encapsulated stream needs
  to be wired up via `oxideav-ogg`.
* No [`oxideav_core::Decoder`] / [`oxideav_core::Encoder`] is
  registered yet; `decode_packet` returns `Error::NotImplemented`.

## Clean-room sources

Round 1 was implemented against, and only against:

* `docs/audio/vorbis/Vorbis_I_spec.pdf` — Xiph.Org Vorbis I
  Specification, 2020-07-04 revision (§2 Bitpacking Convention,
  §4.2.1 Common header decode, §4.2.2 Identification header).
* `docs/audio/vorbis/vorbis-fixtures-and-traces.md` — clean-room
  trace-corpus document for the field-level value shapes of the
  fixture set.

No external library source was consulted, quoted, paraphrased, or
used as a cross-check oracle, in accordance with the workspace
clean-room policy. The forbidden-list for this crate includes every
existing Vorbis implementation (the Xiph reference encoder/decoder,
Tremor, lewton, FFmpeg's Vorbis decoder, every third-party Rust
crate that wraps or implements the format).

[`parse_identification_header`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/identification/fn.parse_identification_header.html
[`VorbisIdentificationHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/identification/struct.VorbisIdentificationHeader.html
[`oxideav_core::Decoder`]: https://docs.rs/oxideav-core/latest/oxideav_core/trait.Decoder.html
[`oxideav_core::Encoder`]: https://docs.rs/oxideav-core/latest/oxideav_core/trait.Encoder.html

## License

MIT. See `LICENSE`.

# oxideav-vorbis

Vorbis audio codec for oxideav — full decoder and tier-2 encoder.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — a
100% pure Rust media transcoding and streaming stack. No C libraries, no FFI
wrappers, no `*-sys` crates.

## Status

**Decoder**: feature-complete for common Vorbis I files (q3–q10 shapes), matches
libvorbis / lewton output within float rounding on the fixture suite.

**Encoder**: mono / stereo at any sample rate, with:

- ATH-scaled floor1 analysis and a 128-entry residue VQ.
- Sum/difference channel coupling (Vorbis I §1.3.3, lossless).
- Transient-driven short-block switching with full asymmetric long↔short
  window transitions (§1.3.2 / §4.3.1). Short blocks confine transient
  energy to ~`bs0/2` samples, reducing pre- and post-echo relative to a
  long-only encode (see `roundtrip_click_short_beats_long_only_baseline`
  for a concrete measurement).
- Output decodes through both our own decoder and ffmpeg's libvorbis.

Deferred: point-stereo coupling, libvorbis Annex B reference codebooks,
floor type 0 (LSP). See the module-level doc in `src/encoder.rs`.

## Usage

```toml
[dependencies]
oxideav-vorbis = "0.0"
```

## License

MIT — see [LICENSE](LICENSE).

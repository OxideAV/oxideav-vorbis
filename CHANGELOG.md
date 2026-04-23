# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Multichannel encoder support (3..=8 channels) with channel coupling per
  the Vorbis I standard channel mappings (L/C/R, 4ch quad, 5.1, 7.1).
  Round-trip tests exercise sine signals per-channel on 3ch through 7.1,
  plus B-format quad noise. Mean SNR on a 5.1 sine bed is ~5 dB.
- Multi-codebook / cascade residue (residue type 2) with per-partition
  class selection. The encoder setup now emits four codebooks (floor Y,
  variable-length `[1, 2, 3, 3]` classbook, 128-entry main VQ,
  16-entry fine-correction VQ) and a two-class / two-cascade-pass
  residue layout: class 0 partitions emit no VQ bits, class 1
  partitions cascade stage-1 (main book) then stage-2 (fine book) to
  quantise residue and residual-of-residue. For sparse-in-frequency
  signals (pure tones) this gives ~60% residue bitrate reduction vs
  the prior single-book single-classification path; for dense signals
  the cost is roughly neutral. SNR on pure tones is unchanged or
  slightly improved (stage-2 halves residue quantisation error).

## [0.0.5](https://github.com/OxideAV/oxideav-vorbis/compare/v0.0.4...v0.0.5) - 2026-04-19

### Other

- add SIMD-accelerated IMDCT and Criterion benches
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"

## [0.0.4](https://github.com/OxideAV/oxideav-vorbis/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- claim WAVEFORMATEX tags via oxideav-codec CodecTag registry
- bump oxideav-core to 0.0.5
- migrate to oxideav_core::bits LSB variants + local float ext
- add public-API encode→decode roundtrip integration tests
- rewrite README/lib docs to match real coverage
- add short-block (transient) encoder

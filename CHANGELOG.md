# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Codebook lookup-type optimiser** (`codebook_optimizer` module,
  task #173). New `detect_lookup1` / `optimise` / `optimise_setup`
  functions inspect VQ codebooks at lookup_type 2 (flat per-entry
  table) and promote them to lookup_type 1 (Cartesian-grid lookup)
  when the per-entry data lies on a small shared grid. The detector
  is exact-bit-equivalent: every entry's decoded f32 vector is
  preserved through the rewrite (verified by round-trip tests
  against `Codebook::vq_lookup`). Heuristic ceiling
  `MAX_GRID_PER_DIM = 32` distinct values per dim — beyond that the
  rewrite stops being worthwhile vs. the lookup_type-2 form. The
  libvorbis q3 reference setup (`LIBVORBIS_SETUP_MONO_48K_Q3`)
  consists entirely of lookup_type-0 (25 books) + lookup_type-1
  (10 books), so the optimiser is a no-op there; on a synthetic
  64-entry dim-2 trained-book-shaped lookup_type-2 input the
  multiplicand-bit budget shrinks 93.8% (from 512 bits to 32 bits).
  The in-tree LBG-trained books in `trained_books.rs` are encoder-
  side only (not in the bitstream) and their unconstrained f32
  centroids do not lie on a Cartesian grid; the detector correctly
  reports `NotPossible` for that input shape — see the test
  `real_trained_book_quantised_does_not_promote`. 13 new unit
  + integration tests cover the detector heuristic, the round-trip
  decode equivalence, and the libvorbis-setup no-op behaviour.
- **Point-stereo channel coupling** above a configurable crossover
  frequency (default 4 kHz, see `DEFAULT_POINT_STEREO_FREQ`). For
  bins above the threshold, every coupled pair `(magnitude, angle)`
  is encoded as `(sign(dominant) * sqrt((L²+R²)/2), 0)` instead of
  the lossless sum/difference form. The decoder reconstructs
  `L = R = m` for those bins, monoising high-frequency stereo
  content where inter-aural phase is below human auditory
  resolution. Verified by `point_stereo_*` test suite — quadrature
  6 kHz stereo input has L−R MSE drop from 16.5M to 0.19M (87×) on
  the decoded output. Bitrate impact is mildly negative (~3% smaller
  on dense high-band content) thanks to tighter VQ matches once
  the angle channel is forced to zero.
- **Floor1 envelope tuning**: per-post target magnitude is now a
  blend `0.7 * peak + 0.3 * RMS` over the band centred on the post,
  instead of pure peak. Y values are picked via a binary search of
  the inverse-dB lookup table (was a linear scan over 128
  candidates) and pass through a forward+backward smearing pass
  that bounds inter-post drops at `12 * multiplier` dB so the
  Bresenham-rendered floor doesn't undercut spectral envelope
  ridges. SNR on a 1 kHz mono sine through our decoder went from
  3.80 dB to 4.20 dB; ffmpeg's libvorbis cross-decode reaches
  4.29 dB on the same input.
- **ffmpeg cross-decode tests**. The encoder's output bitstream is
  now smoke-tested by piping through `ffmpeg -i pipe:0 -f s16le -`
  with a hand-rolled minimal Ogg muxer. Both mono 1 kHz sine and
  stereo 6 kHz quadrature-phase tones (point-coupled) decode
  cleanly with the expected target-frequency dominance.

## [0.0.6](https://github.com/OxideAV/oxideav-vorbis/compare/v0.0.5...v0.0.6) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- bump criterion 0.5 → 0.8
- cascade + multi-class residue (type 2) cuts bitrate ~60%
- update README + encoder docs for 1..=8 channel encode
- multichannel encoder round-trip tests + CHANGELOG
- generalise encoder setup coupling to 1..=8 channels
- add floor type 0 (LSP) decoder
- add NOTICES with libvorbis BSD-3 attribution

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

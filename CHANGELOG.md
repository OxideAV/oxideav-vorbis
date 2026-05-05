# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Tail-aware mu-law quantiser for lookup_type=1 main residue book**
  (`tail_quantiser` module, task #478). New `BitrateTarget::HighTail`
  variant ships a non-uniform mu-law-companded multiplicand axis on
  the main residue VQ book — denser quantisation steps near zero
  where the post-floor residue distribution lives, sparser at the
  tails. Same per-frame codeword budget as `Medium` (11×11 / 7-bit
  codeword), only the bitstream-side `value_bits` widens from 4 to 5
  (+11 setup-header bits, < 0.001 % of a typical residue payload).
  Quality lever closes the round-1 task #463 followup investigation
  ("lookup_type=1 axis-grid path … requires tail-aware quantiser
  design"): pure k-means clusters near zero and under-serves the
  residue distribution's tails, while a closed-form mu-law grid
  (mu = 8, the textbook G.711 mid-value) puts grid resolution
  exactly where the residue lives. Quantiser-only quality gate:
  +1.5 dB SNR vs uniform on a Laplacian source (gated by
  `tail_quantiser::tests::mu_law_grid_outperforms_uniform_on_laplacian_source`,
  6.32 dB vs 4.81 dB measured). End-to-end PCM-domain delta on
  natural-content-like 1.5 kHz tone + Laplacian noise: HighTail
  delivers +0.18 dB vs Medium (5.68 vs 5.50 dB measured) — the
  smaller delta reflects the floor1 stage absorbing much of the
  spectral envelope before residue ever reaches the VQ search.
  ffmpeg cross-decode: HighTail's non-uniform multiplicands are
  wire-format-legal per Vorbis I §3.2.1 and decode through ffmpeg's
  libvorbis without complaint (gated by
  `encoder::tests::ffmpeg_cross_decode_high_tail`). Medium remains
  byte-stable (uniform `0..10` multiplicands at value_bits=4,
  min=-5.0, delta=1.0 — gated by
  `encoder::tests::medium_setup_unchanged_after_high_tail_landed`).
  New public API: `BitrateTarget::HighTail` plus
  `tail_quantiser::{mu_law_grid, MuLawGrid, HIGH_TAIL_MAIN_MULTIPLICANDS}`.
  10 new unit + integration tests (4 in `tail_quantiser`, 6 in
  `encoder::tests`) gate the mu-law grid construction, the bank's
  spec-canonical invariant, the bitstream-side multiplicand emission,
  the SNR-floor preservation vs Medium, the heavy-tailed source
  improvement, and ffmpeg cross-decode. **Followups not in scope
  for this round**: per-frame floor0/floor1 selection (task #478
  secondary item — needs 4 floors + 4 modes in setup header,
  per-block tonality picker; in scope for a future round); LBG-
  trained per-target axis grids (would be a corpus-driven refinement
  on top of the closed-form mu-law shape, gated on a larger training
  corpus than the current LibriVox + Musopen mix).
- **Per-`BitrateTarget` point-stereo crossover frequency** (task #463).
  `BitrateTarget::point_stereo_freq_hz` now returns a per-target
  default that the encoder constructor wires into the per-block
  point-stereo coupling decision: `Low` → 3 kHz (more spectrum
  monoised, smaller bitrate on speech-band content), `Medium` → 4 kHz
  (matches `DEFAULT_POINT_STEREO_FREQ` so existing fixtures stay
  byte-stable), `High` → 6 kHz (preserves more HF stereo image at
  higher rate). On a 4-block stereo HF-decorrelated mix the bank
  produces 10589 B (Low) / 11447 B (Medium) / 14026 B (High) — a
  monotone bitrate ladder that respects each target's intent. ffmpeg
  cross-decodes all three targets unchanged. 3 new unit tests gate
  the monotone-crossover invariant, the Medium byte-stability
  preservation vs `DEFAULT_POINT_STEREO_FREQ`, and the per-target
  bitrate monotonicity on stereo HF-decorrelated content.

  **Followup investigation: trained main-VQ centroids per
  `BitrateTarget`** (task #463 round-1, NOT landed). Replacing the
  grid-based main residue VQ with LBG-trained centroids would require
  either (a) `lookup_type = 2` per-entry vector storage in the setup
  header — but **modern ffmpeg's libvorbis decoder explicitly rejects
  `lookup_type >= 2`** ("Codebook lookup type not supported" in
  `vorbis_dec.c`), making lookup_type=2 incompatible with ffmpeg
  cross-decode; or (b) `lookup_type = 1` with non-uniform per-axis
  multiplicands trained via 1-D LBG on the residue distribution —
  empirically this regresses SNR vs the uniform grid because pure
  k-means centroids cluster near zero and under-serve the residue
  distribution's tails, which tail-aware quantiser design (mu-law /
  Lloyd-Max with range constraint) would address. Both alternatives
  shipped behind paths in this round were reverted on the grounds
  above; the encoder docs now document the ffmpeg-interop
  constraint to save future agents the same investigation cycle.

- **Bitstream-resident residue codebook bank** (`codebook_bank` module).
  New `BitrateTarget::{Low, Medium, High}` enum drives selection of a
  curated dim-2 grid VQ pair (main + fine) at encoder construction
  time; the chosen books are emitted into the setup header so the
  audio packets index bitstream-resident codebooks through the
  `mapping → submap → residue` chain. Each variant is a perfect-fill
  canonical-Huffman tree with the spec-canonical
  `lookup1_values(entries, 2) == values_per_dim` invariant
  (Vorbis I §9.2.3) — Low at codeword_len 6 / 8×8 grid, Medium at 7 /
  11×11 (the historical default; byte-stable for existing fixtures),
  High at 8 / 16×16. New public API:
  `encoder::make_encoder_with_bitrate(&params, BitrateTarget)` plus
  `encoder::build_encoder_setup_header_with_target(channels, target)`.
  On a 2 s 1 kHz + low-noise mono mix Low encodes 47.8 kB / 2.94 dB
  SNR vs Medium 51.6 kB / 4.20 dB (~7% bitrate saving) and High
  63.0 kB / 6.05 dB (+22% bytes for +1.85 dB SNR). All three
  variants ffmpeg cross-decode cleanly with the expected
  target-frequency dominance. 10 new unit tests cover the bank's
  spec-invariant validation, the Medium byte-stability gate, the
  monotone-in-bitrate ordering, end-to-end round-trip per target,
  per-target SNR floor, and ffmpeg cross-decode per target.
- **Floor type 0 (LSP) encoder** (`floor0_encoder` module, task #181).
  New `make_encoder_floor0` constructor produces a Vorbis encoder that
  emits floor0 packets end-to-end via LPC analysis + LSP conversion.
  Pipeline per Vorbis I §6: windowed autocorrelation → Levinson-
  Durbin recursion (`order = 16`) → LPC → LSP conversion via
  symmetric / antisymmetric polynomial root-finding (Chebyshev grid
  scan + bisection) → cosine quantisation against a 256-entry
  `[-1, 1]²` VQ codebook → bitstream emission of `amplitude` +
  book number + VQ codewords. The setup writer
  `build_encoder_setup_header_floor0` produces a self-contained 2-
  codebook setup (LSP VQ + constant-1.0 residue book) that decodes
  through the existing `crate::decoder` path. Round-trip on a 440 +
  880 Hz mono fixture (4 long blocks, 48 kHz) preserves the
  fundamental-vs-off-band energy ratio. The
  `floor0_encoder::should_use_floor0` helper exposes a per-frame
  prediction-gain heuristic (Levinson-Durbin order 4 → ratio ≥ 4×)
  for hybrid encoders that want to pick floor0 on tonal blocks and
  floor1 on noise blocks. 8 new unit + integration tests cover the
  codebook builder, LSP analysis, quantiser round-trip, setup parser
  acceptance, and end-to-end decode of an encoded fixture.
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

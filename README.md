# oxideav-vorbis

Pure-Rust **Vorbis I** audio codec — full decoder and 1..=8 channel
encoder. Zero C dependencies, no FFI, no `*-sys` crates.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-vorbis = "0.0"
```

Vorbis always ships three header packets (identification, comment,
setup) in the container's codec-private blob. The decoder expects
them as a Xiph-laced concatenation — the format `oxideav-ogg` and
`oxideav-mkv` already produce for Vorbis tracks.

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{Frame, MediaType};

let mut codecs = CodecRegistry::new();
oxideav_vorbis::register(&mut codecs);

let mut dec = codecs.make_decoder(&stream_params)?;
dec.send_packet(&packet)?;
while let Ok(Frame::Audio(af)) = dec.receive_frame() {
    // af.format == SampleFormat::S16, interleaved in af.data[0]
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Decoder coverage

The decoder is feature-complete for the file shapes real-world
encoders produce (libvorbis / aoTuV across q0–q10):

- Mono, stereo, and multichannel streams up to 255 channels.
- Floor type 1 (piecewise linear, Bresenham-rendered) with any
  multiplier, rangebits, or partition layout.
- Residue types 0, 1, and 2 (per-channel interleave, concatenated,
  and cross-channel interleave) with cascade books.
- Mapping type 0, any number of submaps, any number of channel
  coupling steps. Inverse coupling (Vorbis I §1.3.3) handles both
  the lossless sum/difference and the lossy "point stereo"
  encoder convention — the bitstream is the same for both.
- Full asymmetric MDCT windows (short/long, per §1.3.2 / §4.3.1)
  and overlap-add across block-size transitions.

Floor type 0 (LSP) is supported on the decode side and exercised by
both the in-tree fixture suite and the floor0 encoder's round-trip
tests (`floor0_encoder::tests::round_trip_tonal_fixture_via_our_decoder`).
Output matches libvorbis / lewton within float rounding on the
fixture suite.

## Encoder

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, SampleFormat};

let mut codecs = CodecRegistry::new();
oxideav_vorbis::register(&mut codecs);

let mut params = CodecParameters::audio(CodecId::new("vorbis"));
params.channels = Some(2);
params.sample_rate = Some(48_000);
params.sample_format = Some(SampleFormat::S16);

let mut enc = codecs.make_encoder(&params)?;
enc.send_frame(&frame)?;   // S16 or F32 interleaved
let pkt = enc.receive_packet()?;
let extradata = enc.output_params().extradata.clone();
# Ok::<(), Box<dyn std::error::Error>>(())
```

The encoder accepts 1..=8 channels at any sample rate. Output
decodes through both this crate's decoder and ffmpeg's libvorbis.

What's implemented:

- Short (256) and long (2048) block sizes with a transient-driven
  switcher and full asymmetric long↔short window transitions
  (§1.3.2 / §4.3.1). Short blocks confine click energy to
  `bs0/2` post-echo — see the
  `roundtrip_click_short_beats_long_only_baseline` test for a
  measurement.
- Floor1 analysis with peak/RMS-blended envelope tracking,
  binary-search dB-domain quantisation, and a
  smearing pass that prevents inter-post Bresenham dips below
  `12 * multiplier` dB (§7.2.4 step1 / step2). 8 posts on short
  blocks, 32 on long. Mono 1 kHz SNR is ~4.2 dB through both our
  decoder and ffmpeg's libvorbis (vs ~3.8 dB pre-tuning).
- Sum/difference channel coupling (§1.3.3) for L-R pairs below the
  point-stereo crossover (~4 kHz, configurable via
  `DEFAULT_POINT_STEREO_FREQ`); above the crossover the encoder
  splits the spectrum into critical-band sub-bands (4-6, 6-9, 9-13,
  13-Nyquist kHz, see `POINT_STEREO_BAND_THRESHOLDS`) and applies
  point-stereo per-band (`a = 0`, `m = sign(dom) * sqrt((L²+R²)/2)`)
  only when the band's L/R correlation exceeds the band's threshold
  (0.60 → 0.50 → 0.40 → 0.35 from low to high — HF bands are more
  permissive because masking grows with frequency). Decorrelated
  bands fall back to lossless sum/difference so phase information
  is preserved where it matters perceptually. The decoder
  reconstructs `L = R = m` for point-coupled bins. Multichannel
  streams use the standard Vorbis I mappings (L/C/R, 4ch quad, 5.1,
  7.1) with coupled L-R, back-L/back-R, and side-L/side-R pairs;
  center and LFE channels stay uncoupled.
- A 4-codebook residue setup: floor1 Y book (128 × 7-bit), a 4-entry
  variable-length classbook (`[1, 2, 3, 3]`), main VQ + fine correction
  VQ from a curated **bitstream-resident codebook bank**
  ([`src/codebook_bank.rs`](src/codebook_bank.rs)) with three
  bitrate-target variants — `Low` (8×8 main + 4×4 fine, 6+4 bit
  codewords), `Medium` (11×11 + 4×4, 7+4 bit, the historical default)
  and `High` (16×16 + 8×8, 8+6 bit). Each is a perfect-fill
  canonical-Huffman tree with the spec-canonical
  `lookup1_values(entries, 2) = values_per_dim` relation so libvorbis /
  ffmpeg accept all three variants. Picked at construction via
  `make_encoder_with_bitrate(&params, BitrateTarget::Low | Medium |
  High)`; the chosen books go into the setup header and the audio
  packets index them through `mapping → submap → residue`. On a 2-second
  1 kHz mono mix Low saves ~7% bytes vs Medium at slightly lower SNR
  (2.94 vs 4.20 dB); High costs ~22% more bytes for ~1.85 dB SNR gain
  (6.05 dB). See `bank_targets_are_monotone_in_bitrate` for the
  measurement.
- Residue type 2 (interleaved across channels) for both block sizes
  with two-class per-partition selection (silent vs active cascade).
- **Trained-VQ partition classifier** (task #93). The silent/active
  decision per residue partition is driven by the median 2-bin slice
  L2 of four LBG-trained 256-entry codebooks — see
  [`src/trained_books.rs`](src/trained_books.rs), trained from
  ~15 minutes of mixed LibriVox PD speech + Musopen Chopin (CC0)
  via [`scripts/fetch-vq-corpus.sh`](scripts/fetch-vq-corpus.sh) and
  [`src/bin/vq-train.rs`](src/bin/vq-train.rs). Training is offline;
  the trained centroids are baked into the crate as a `const` table.
  On a 5-second sine + voice-band noise mix the trained classifier
  saves ~11% bytes vs the prior hard-coded threshold at identical
  SNR — see the
  `encoder::tests::trained_vs_legacy_classifier_bitrate_5s_mix`
  fixture. The bitstream layout is unchanged; the trained books
  inform encoder choices only, not the wire format.
- Xiph-laced 3-packet `extradata` in `output_params()`, ready to
  hand to a container muxer. ffmpeg's libvorbis decodes the output
  bit-cleanly — see the `ffmpeg_cross_decode_*` tests.

Known limitations, relative to libvorbis, that affect bitrate but
not bitstream conformance:

- **Bank tuning is grid-derived, not LBG-trained.** The three
  `BitrateTarget` entries in
  [`src/codebook_bank.rs`](src/codebook_bank.rs) span a uniform
  Cartesian grid per `(values_per_dim, codeword_len)` shape. They
  give monotone-in-bitrate behaviour and ffmpeg-accepted bitstreams
  but don't carry the corpus-trained centroid placement that
  libvorbis's per-quality books do. Future work: replace the grid
  with LBG centroids (per-bitrate-target re-training of the
  trained-book corpus) for a tighter-fitting main VQ at each rate.
- **Floor type 0 (LSP) emission.** The default `make_encoder`
  constructor still always declares floor1. A separate
  `floor0_encoder::make_encoder_floor0` constructor (task #181) emits
  floor0 packets end-to-end via LPC analysis (autocorrelation +
  Levinson-Durbin) followed by LPC→LSP conversion; the
  `floor0_encoder::should_use_floor0` helper provides the
  prediction-gain heuristic for hybrid encoders that want to switch
  per-frame. The default constructor doesn't yet wire those together
  — picking floor0 vs floor1 from a single setup requires plumbing
  both into the same mapping, tracked as future work.

## Performance

The hot kernels (IMDCT, forward MDCT, window overlap-add, residue
accumulation, floor-curve multiply) live in [`src/simd/`](src/simd/)
with three parallel implementations:

- `simd::scalar` — reference, always available, used as the bit-exact
  oracle for the test suite.
- `simd::chunked` — stable-Rust "manual SIMD" over fixed 8-lane `[f32; 8]`
  chunks. LLVM reliably lowers this to `vfmadd231ps` on AVX2 and to
  paired NEON Q-reg ops on ARMv8. This is the default on stable.
- `simd::portable` — `std::simd::f32x8` using fused multiply-add
  (`StdFloat::mul_add`). Gated behind the `nightly` feature flag; needs
  a nightly Rust toolchain.

The biggest single win came from changing the IMDCT / forward MDCT from
the textbook `x[n] = Σ_k X[k] · cos(...)` inner loop to a precomputed
cosine-matrix dot product. The matrix is cached per-blocksize in a
`OnceLock<HashMap>` and lives for the rest of the process, so the
cosine computation happens exactly once per `(short, long)` pair per
run. Runtime CPU feature detection lives in
[`src/simd/dispatch.rs`](src/simd/dispatch.rs) (AVX2/FMA on x86_64,
NEON on aarch64) and is wired through `simd::dispatch::detect()` for
future use — the chunked path already reaches AVX2 throughput via
auto-vectorisation.

### Microbenchmarks

Measured on x86_64 (Gentoo, rustc 1.93.1 stable, release build):

| Bench                       | Reference (scalar f64) | Optimised | Speedup |
|----------------------------|------------------------|-----------|---------|
| `imdct_1024`               | 3.60 ms                | 129 µs    | 27.8×   |
| `imdct_2048`               | 12.3 ms                | 558 µs    | 22.0×   |
| `forward_mdct_2048`        | 13.3 ms                | 537 µs    | 24.8×   |
| `window_overlap_add_2048`  | 201 ns                 | 181 ns    | 1.11×   |
| `add_inplace_1024`         | 51.4 ns                | 46.8 ns   | 1.10×   |

End-to-end (1 s of input → bitstream → PCM):

| Bench                       | Time    | Real-time factor |
|----------------------------|---------|------------------|
| `decode_1s_mono_48k`       | 32.1 ms | ~31×             |
| `decode_1s_stereo_44k1`    | 57.8 ms | ~17×             |
| `encode_1s_stereo_44k1`    | 111 ms  | ~9×              |

Run `cargo bench --bench vorbis` to reproduce. Each paired kernel is
verified bit-exact against the scalar reference within a square-root-N
ε (see `imdct::tests::imdct_simd_matches_reference`).

### Enabling the `nightly` feature

```bash
cargo +nightly build --release --features nightly
cargo +nightly bench --bench vorbis --features nightly
```

The `nightly` feature flips `simd::mul_inplace` / `add_inplace` /
`overlap_add` / `mat_vec_mul` from the chunked path to the
`std::simd::f32x8` path. In practice the performance delta is small
(LLVM's auto-vectorisation of the chunked path is already tight); the
flag is primarily there so the `std::simd` API can evolve without
breaking the stable build.

## Codec ID

- Codec: `"vorbis"`; output sample format `S16`, accepted input
  `S16` or `F32`.
- Headers are produced as a 3-packet Xiph-laced `extradata` blob.

## License

MIT — see [LICENSE](LICENSE).

This crate additionally incorporates the floor1 inverse-dB table from
libvorbis (© 2002-2020 Xiph.Org Foundation, BSD-3-Clause). The upstream
notice and license text are in [NOTICES](NOTICES) and must be preserved
in redistributions.

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

Floor type 0 (LSP) is not implemented: no modern encoder produces
it. The decoder rejects such streams with `Error::Unsupported`.
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
  switches to point-stereo (`a = 0`, `m = sign(dom) *
  sqrt((L²+R²)/2)`) to monoise inter-aural phase that the human
  ear cannot resolve. The decoder reconstructs `L = R = m` for
  point-coupled bins. Multichannel streams use the standard
  Vorbis I mappings (L/C/R, 4ch quad, 5.1, 7.1) with coupled L-R,
  back-L/back-R, and side-L/side-R pairs; center and LFE channels
  stay uncoupled.
- A 4-codebook residue setup: floor1 Y book (128 × 7-bit), a 4-entry
  variable-length classbook (`[1, 2, 3, 3]`), 128-entry main VQ
  ({-5..+5}²) and 16-entry fine correction VQ ({-0.6..+0.6}²) with
  cascade — exhaustive nearest-neighbour search per partition.
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

- **Trained-book bitstream codebooks.** The encoder's residue
  cascade still emits via the 4-codebook setup (floor1 Y, classbook,
  main VQ, fine VQ) — the LBG-trained books in
  [`src/trained_books.rs`](src/trained_books.rs) inform partition
  classification but do not yet replace the bitstream codebooks
  themselves. Doing so would let the encoder reach libvorbis's
  per-genre book tuning (dozens of dim-16 trained books per
  quality tier instead of one global main VQ); that's a setup-header
  change tracked under future work.
- **Per-band point-stereo thresholds.** We use a single global
  crossover frequency. Libvorbis's `iiPST` config carries a per-
  band threshold list so the crossover can adapt by frequency band.
  Our scheme matches Libvorbis's qN ≤ 2 channel-folding shape but
  not its qN ≥ 4 multi-band tuning.
- **Floor type 0 (LSP) emission.** Setup always declares floor1.

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

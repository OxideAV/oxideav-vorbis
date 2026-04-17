# oxideav-vorbis

Pure-Rust **Vorbis I** audio codec — full decoder and mono/stereo
encoder. Zero C dependencies, no FFI, no `*-sys` crates.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.0"
oxideav-codec = "0.0"
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

The encoder accepts 1 or 2 channels at any sample rate. Output
decodes through both this crate's decoder and ffmpeg's libvorbis.

What's implemented:

- Short (256) and long (2048) block sizes with a transient-driven
  switcher and full asymmetric long↔short window transitions
  (§1.3.2 / §4.3.1). Short blocks confine click energy to
  `bs0/2` post-echo — see the
  `roundtrip_click_short_beats_long_only_baseline` test for a
  measurement.
- ATH-scaled floor1 analysis (8 posts short, 32 posts long).
- Sum/difference channel coupling (§1.3.3) for stereo — lossless,
  round-trips via the decoder's inverse step.
- A single 128-entry, dim-2 VQ residue book covering {-5..+5}²,
  with exhaustive nearest-neighbour search per partition.
- Residue type 1 (concatenated per-channel) for both block sizes.
- Xiph-laced 3-packet `extradata` in `output_params()`, ready to
  hand to a container muxer.

Known limitations, relative to libvorbis, that affect bitrate but
not bitstream conformance:

- **More than 2 channels on encode.** The decoder handles up to
  255; the encoder setup builder only wires mono and stereo
  mappings.
- **Point-stereo encoding.** Stereo always emits lossless
  sum/difference coupling. Libvorbis switches to lossy point
  stereo above a threshold frequency to roughly halve the angle
  channel's residue cost. The decoder already handles the
  bitstream unchanged — this is purely an encoder-side refinement.
- **Libvorbis Annex B reference codebooks.** Our single residue
  book serves every quality. Libvorbis ships dozens of books per
  quality tier plus energy-classifying master books, so its files
  compress tighter for comparable quality.
- **Floor type 0 (LSP) emission.** Setup always declares floor1.

## Codec ID

- Codec: `"vorbis"`; output sample format `S16`, accepted input
  `S16` or `F32`.
- Headers are produced as a 3-packet Xiph-laced `extradata` blob.

## License

MIT — see [LICENSE](LICENSE).

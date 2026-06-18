# oxideav-vorbis

Pure-Rust Vorbis I audio codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
Clean-room implementation against the Vorbis I Specification
(Xiph.Org) and the corpus traces under `docs/audio/vorbis/`.

## Status

**Clean-room rebuild in progress.** The crate implements the full
Vorbis I decode pipeline at the codec layer plus a growing set of
encoder write-path primitives. It is codec-only: Ogg container
framing and `oxideav-core` registration are not yet wired —
`register()` is currently a no-op.

### Decode

- **Header parsers** — the three Vorbis I header packets:
  identification (§4.2.2, `parse_identification_header`), comment
  (§5, `parse_comment_header`), and setup (§4.2.4,
  `parse_setup_header`, including the codebook §3.2.1, floor 0 §6.2.1,
  floor 1 §7.2.2, residue §8.6.1, mapping, and mode sub-blocks). A
  packet-kind classifier (`packet_kind::classify_packet` /
  `parse_header_packet`) resolves a raw packet to its kind without
  parsing the body.
- **Per-packet audio decode** — the §4.3 pipeline end to end: floor 0
  / floor 1 curve computation, residue formats 0 / 1 / 2, §4.3.3
  nonzero propagation, §4.3.5 inverse coupling, §4.3.6 dot product
  (`audio::decode_audio_packet_pre_imdct`), the §4.3.7 inverse-MDCT
  cosine-summation kernel (`imdct`), the §4.3.1 / §1.3.2 Vorbis window
  (`synthesis::window_premultiply`), and the §4.3.8 overlap-add
  (`overlap::OverlapAdd`).
- **§4.3.9 output channel order** — `channel_order::speaker_layout` /
  `speaker_at` map each encoded-stream channel index to its mapping-type-0
  physical speaker location for the 1..=8-channel layouts the spec fixes
  (mono; stereo L/R; the 5.1 / 6.1 / 7.1 surround orderings, where 6.1
  and 7.1 use the rev-16781 side-pair + rear-center / rear-pair forms).
  Counts above eight report `Speaker::Unspecified` (application-defined).
  Decode keeps emitting channels in bitstream order; this is the
  documented layout a consumer uses to reorder.
- **Streaming decoder** — `streaming::StreamingDecoder` stitches the
  per-packet driver to per-channel overlap-add across consecutive
  packets, emitting finished `StreamingFrame::Pcm` samples per channel.
  It is the integrated bitstream → PCM path.
- **Fixture-anchored silence decode** — an integration test
  (`tests/silence_stream_decode.rs`) drives the
  `docs/audio/vorbis/fixtures/silence-stream/` packet geometry (mono,
  `blocksize_0 = 256`, `blocksize_1 = 2048`, two modes; every audio
  packet `packet_bytes=1`) through the full public
  `StreamingDecoder::push_packet` path and asserts pure-silence PCM. Per
  §4.3.2 step 6 the `'unused'` floor short-circuits to the all-zero
  spectrum, so the result is provably independent of the still-deferred
  IMDCT normalization scalar (`0 · α = 0`) — making silence the one
  end-to-end PCM target pinnable while the post-IMDCT trace point that
  would fix the scalar is missing from the staged traces.

### Encode

A set of bit-exact WRITE primitives that invert the parsers, each
validated against its decode counterpart by a roundtrip property:
the identification / comment / setup header writers (`encoder::write_*`),
all six setup sub-block writers (codebook / floor 0 / floor 1 /
residue / mapping / mode), the §4.3.1 audio-packet prelude writer, the
floor 0 / floor 1 packet-body writers, the §8.6.2 residue-body writer
(classification packing + per-partition value codewords), the §4.3
wrapping audio-packet writer, the §4.3.7 forward-MDCT kernel (`mdct`),
the §4.3.5 forward coupling, the §4.3.6 spectrum-factoring inverse, the
§4.3.8 encoder-side framing splitter (`framing::FrameSplitter`), and the
§3.2.1 VQ-encode quantiser (`vq::quantize_vector`) — the encode-side
inverse of `unpack_vector`: it picks the **used** codebook entry whose
§3.2.1-decoded vector lies nearest a target (squared-Euclidean, ties to
the lowest index, sparse-codebook unused entries skipped) and returns the
entry index plus its decoded reconstruction and the squared-distance
residual for residue-stage (§8.6.2) cascading.

The **residue VQ-encode cascade planner** (`residue_encode` module:
`plan_partition_cascade` + `plan_vector_partition_entries`) is the glue
that sits between `vq::quantize_vector` and the residue WRITE path. Given
a partition's real spectral residual it walks the §8.6.2 cascade in the
write direction — gathering each VQ read's sub-vector from the running
residual at the exact positions the decoder scatters into (§8.6.3 strided
for format 0, §8.6.4 contiguous for formats 1/2, format 2 reduced to
format 1 per §8.6.5), quantising each via `quantize_vector`, then
subtracting the chosen entry's reconstruction so the next cascade stage
refines the leftover error (the inverse of §8.6.2 step 19's additive
`+=` accumulation). It produces exactly the per-`(partition, pass)`
entry-index lists `encoder::ResidueVectorPlan::partition_entries` /
`write_residue_body` already consume; an independent decode-reconstruct
oracle pins the round-trip across both addressing formats and multi-stage
refinement. The encoder WRITE primitives no longer need the residue
entry indices supplied by hand.

The **floor-0 VQ-encode glue** (`floor0_encode` module:
`plan_floor0_coefficients` + `floor0_vector_count`) is the analogous glue
for floor 0. Given a target LSP `[coefficients]` list and the value book a
packet's `[booknumber]` selects, it walks the §6.2.2 step-7..11 vector
schedule in the write direction: for each of the `ceil(order / dimensions)`
vectors the decoder reads, it **un-offsets** the target sub-vector by the
running `[last]` accumulator (the inverse of step 8's additive offset),
quantises the raw target via `vq::quantize_vector`, then advances `[last]`
from the chosen entry's **reconstruction** (not the target) so every
subsequent vector un-offsets against the value the decoder will carry. It
produces exactly the entry run `encoder::Floor0Packet::Curve::entries` /
`write_floor0_packet` already consume; an independent decode-reconstruct
oracle pins the cumulative-`[last]` round-trip across single-vector,
multi-vector, partial-final-vector, and lossy-quantisation cases. The
floor-0 WRITE primitive no longer needs the entry indices supplied by hand.

### Not yet supported / known gaps

- **No `Decoder` / `Encoder` registration** and no Ogg container
  layer — the crate exposes the codec primitives, not a wired codec.
- **The Vorbis-specific MDCT normalization scalar** is a caller-supplied
  `imdct_scale` knob, not yet pinned to a constant. The Vorbis I spec
  defers the MDCT definition to an external reference barred by the
  clean-room policy; the in-repo IMDCT cross-reference
  (`docs/audio/vorbis/imdct-cross-reference.md`) reconstructs the
  kernel, but the normalization scalar "falls out of matching the
  fixture traces" and the staged traces under
  `docs/audio/vorbis/fixtures/<case>/` do not yet log post-MDCT
  samples. Pinning it is the remaining gap before sample-exact PCM.
- **Deriving the target floor-0 LSP `[coefficients]`** — the VQ-encode
  glue mapping a target coefficient list onto the §6.2.2 entry run now
  exists (`floor0_encode::plan_floor0_coefficients`, see above), so the
  floor-0 WRITE primitive no longer needs hand-supplied entry indices.
  Inverting §6.2.3 curve computation to *produce* that target coefficient
  list (and choosing the per-packet `amplitude` / `booknumber`) from a
  desired floor envelope is the remaining floor-0 encode followup.

## Clean-room provenance

The implementation is derived entirely from the Vorbis I Specification
PDF and the OxideAV clean-room companion documents and corpus traces
under `docs/audio/vorbis/`. No third-party Vorbis implementation has
been consulted at any stage. The crate's prior implementation was
retired under the workspace clean-room policy and rebuilt from a
`NotImplemented` scaffold.

## License

MIT. See `LICENSE`.

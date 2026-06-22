# oxideav-vorbis

Pure-Rust Vorbis I audio codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
Clean-room implementation against the Vorbis I Specification
(Xiph.Org) and the corpus traces under `docs/audio/vorbis/`.

## Status

**Clean-room rebuild in progress.** The crate implements the full
Vorbis I decode pipeline at the codec layer — **decoding real Vorbis
audio packets to sample-exact PCM** — plus a growing set of encoder
write-path primitives. It is codec-only: Ogg container framing and
`oxideav-core` registration are not yet wired — `register()` is
currently a no-op.

The §4.3 decode chain is sample-exact end to end: twelve staged fixtures
(`docs/audio/vorbis/fixtures/*` — mono / stereo / 5.1, q−1 through q10,
CBR, 22.05–96 kHz, floor-1-only, full-residue noise, all three residue
formats, and the short↔long transient block-size switch) decode through
the public `StreamingDecoder::push_packet` path to PCM that matches each
fixture's `expected.wav` reference dump within the documented ±1 s16
lossy tolerance (`tests/fixture_pcm_decode.rs`). A separate
`tests/chained_stream_decode.rs` drives the chained-Ogg fixture (two
concatenated logical bitstreams, RFC 3533 §5): the first stream decodes
sample-exact against `expected.wav`, and the second logical bitstream
re-parses its own three Vorbis headers and decodes independently —
exercising the per-stream reset + re-parse + decode cycle across a
stream boundary. A third, `tests/comment_header_decode.rs`, drives the
two metadata fixtures (`with-vorbis-comment-tags`,
`with-attached-picture`): the §5.2 VORBIS_COMMENT parse recovers the
canonical TITLE / ARTIST / ALBUM / DATE / GENRE / TRACKNUMBER tags
(case-insensitively per §5.2.2) and the base64 `METADATA_BLOCK_PICTURE`
cover-art blob, while the audio decodes sample-exact — proving a large
comment block does not perturb the §4.3 decode (both fixtures carry the
identical audio and decode bit-for-bit alike). The §4.3.7 IMDCT
normalization scalar — the last
deferred decode unknown — is pinned to **1.0**: the bare cosine-summation
kernel plus the §4.3.6 window and §4.3.8 overlap-add need no extra
Vorbis-specific scaling.

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
- **Decode robustness** — `tests/decode_robustness.rs` pins the §4.3.1
  "discard this packet" recovery contract: a real fixture packet
  truncated at every byte length, header-type packets (first bit set)
  routed into the audio driver, and empty / single-byte / pseudo-random
  packet bodies all return a typed `StreamingError` or decode cleanly —
  never a panic.
- **Per-packet trace conformance** — `tests/audio_packet_trace_conformance.rs`
  validates the §4.3.1 *structural decisions* the PCM-level fixture test
  doesn't reach. Each fixture's `trace.txt` is the documented load-bearing
  reference for the per-packet `mode_number` / `blockflag` / `prev_window`
  / `next_window` / `block_size` selections every conformant decoder must
  reproduce; the suite drives every fixture's audio packets through the
  public §4.3.1 parser (`read_packet_header`) and asserts each parsed
  header matches the trace **line-for-line** — **505 audio-packet decisions
  across all 16 staged fixtures**, the chained two-stream fixture included
  (a serial-aware de-framer separates its logical streams so each is
  validated against its own `stream_idx` trace records, proving the §4.3.1
  decode is per-stream independent across the chaining boundary). It is a
  pure header-decision oracle (no floor / residue / IMDCT runs), isolating
  a regression in the mode-bit width, the short-vs-long window-flag gating,
  or the blocksize resolution from the PCM-level test where such a bug would
  surface only as garbled audio. The trace's `packet_idx` is honoured as the
  true bitstream index — long streams log packets 0..=31 then the final
  end-trim packet at its real index, so the granule-position end-trim packet
  is validated too — and each de-framed packet body length is cross-checked
  against the trace's `packet_bytes`.
- **Setup-header structural conformance** —
  `tests/setup_header_trace_conformance.rs` validates the *setup-time*
  structural configuration the headers establish (the "same setup-header
  counts" the fixture notes list as load-bearing). It parses each fixture's
  identification (§4.2.2) and setup (§4.2.4) headers and asserts the parsed
  structures reproduce, field-for-field, every `VORBIS_HEADER_ID` /
  `VORBIS_HEADER_SETUP` / `CODEBOOK` / `FLOOR_CONFIG` / `RESIDUE_CONFIG` /
  `MAPPING_CONFIG` / `MODE_CONFIG` trace event: identification
  channels/rate/bitrates/blocksizes; the five setup counts; per-codebook
  dimensions/entries/lookup-type/value-bits/sequence-p; per-floor type and
  (floor-1) partitions/multiplier/rangebits/x-list-count (with the two
  implicit endpoint posts); per-residue
  type/begin/end/partition-size/classifications/classbook; per-mapping
  type/submaps/coupling-steps and the magnitude/angle/per-submap
  floor/residue index arrays; and per-mode
  blockflag/windowtype/transformtype/mapping — **842 structural events
  across all 16 staged fixtures**, the chained two-stream fixture validated
  per-logical-stream. Together with the per-packet suite this pins the
  **entire** structural decode of every staged stream against the reference
  trace, leaving only the lossy sample values to the ±1-s16 PCM test.
- **Overlap-add output geometry** — `tests/overlap_add_geometry.rs` pins the
  §4.3.8 windowing / overlap-add-into-PCM *geometry* as it runs inside
  `StreamingDecoder`. Driving every fixture's whole audio stream through the
  public `push_packet` path, it asserts for every emitted frame that the
  priming step lands only on the first packet, that each subsequent frame's
  per-channel PCM length equals the §4.3.8 `prev_n/4 + cur_n/4` lap (with
  `prev_n` from the previous packet's reported `n`, so the contract is
  checked across **all** packets — including the ones the trace does not
  log), that every channel of a frame has identical length, and that the
  streaming path reports the same `mode_number` / `blockflag` /
  `block_size` the trace logged. **654 PCM frames across all 16 staged
  fixtures.** It is a geometry + dispatch oracle (no sample-value re-check),
  isolating a §4.3.8 lap-length / priming / mode-dispatch regression from a
  numeric IMDCT/floor/residue one.

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

The **floor-1 amplitude-unwrap glue** (`floor1_encode` module:
`plan_floor1_y`) is the floor-1 analogue. Given a target *reconstructed*
post list (`[floor1_final_Y]`, the integer amplitudes the §7.2.4 step-2
curve synthesis draws) and the floor's `[Floor1Header]`, it inverts the
§7.2.4 step-1 amplitude synthesis to produce the always-non-negative
packet-domain `[floor1_Y]` values `encoder::Floor1Packet::floor1_y` /
`write_floor1_packet` already consume. It walks the same strict
left-to-right schedule the decoder uses — for each post computing the
identical `render_point` line prediction from the **already-reconstructed**
neighbour posts (`low_neighbor` / `high_neighbor` only look backward), then
the `highroom` / `lowroom` / `room` window — and selects the unique packet
value whose decode reproduces the target: `val = 0` for an on-prediction
(unflagged) post, the zig-zag even/odd candidate (`2·d` / `−2·d−1`) inside
`room`, or the single linear extension (upper when `highroom > lowroom`,
lower otherwise) past it. A target the §7.2.4 map cannot reach in
`[0, range)` is rejected (`UnreachablePost`); everything reachable
round-trips **losslessly** (floor-1 post coding is exact — the lossy choice
is which posts to target). An independent from-spec decode oracle pins the
round-trip across the zig-zag region, both linear extensions, a full
`[0, range)` interior sweep over all four multipliers, and the endpoint-only
degenerate floor. The floor-1 WRITE primitive no longer needs the
`[floor1_Y]` values supplied by hand.

The **floor-1 envelope-fit glue** (`floor1_envelope` module:
`plan_floor1_envelope`, `invert_inverse_db`) closes the front of the chain:
the §7.2.4 step-2 / §10.1 dB-table **inverse**. Given a desired
linear-domain floor envelope — one magnitude per spectral bin, the domain
the forward MDCT produces — it fits the integer `[floor1_final_Y]` post
vector `plan_floor1_y` consumes. For each post it samples the envelope at
the post's `x` (posts past the floor length sample the last bin, matching
the decoder's flat tail render), inverts the strictly-increasing 256-entry
`floor1_inverse_dB_table` to the nearest ladder index (monotone search,
ties to the lower index), then divides by the multiplier (round-to-nearest)
and clamps into `[0, range)`. `invert_inverse_db` is the standalone
dB-table inverse, pinned to recover every exact table value to its own
index. The whole floor-1 encode pipeline now runs end to end: a forward
**windowed MDCT** analysis frame → smoothed magnitude envelope →
`plan_floor1_envelope` → `plan_floor1_y` → `write_floor1_packet` → the
crate's own decoder. The `tests/floor1_envelope_roundtrip.rs` integration
test drives exactly that chain on a synthetic PCM block and pins both
**post-exact reconstruction** (every rendered post returns the nearest
representable envelope value) and a **23 dB log-domain (dB-index) SNR**
across the whole reconstructed curve — the first end-to-end
encode→decode spectral round-trip in the crate.

Two further integration tests close the encode→decode loop on the residue
and the time domain. `tests/residue_cascade_roundtrip.rs` drives a real
signed, non-flat spectral residual through the full residue encode chain
(`plan_partition_cascade` → `write_residue_body`) and back through
`ResidueDecoder`, pinning three §8.6.2 properties: the decoded residual is
the nearest-entry ladder quantisation of the target, the decoded vector
equals bin-for-bin the **sum of the chosen entries' reconstructions**
(proving the entry-index ↔ codeword round-trip is exact), and a **two-stage
cascade is strictly closer to the target than a one-stage cascade** (and
lifts the spectral SNR by ≥6 dB) — the additive cascade-refinement the
§8.6.2 `+=` accumulation provides. `tests/pcm_packet_roundtrip.rs` is the
crate's first **full §4.3 PCM → encode → decode → PCM time-domain
round-trip**: a synthetic analysis frame is windowed + forward-MDCT'd,
fitted with a flat floor-1 (`F = 1.0` at post 255) plus a two-stage residue
cascade carrying `X/F`, serialised by `write_audio_packet`, then decoded by
`decode_audio_packet_windowed` (the §4.3.2–§4.3.6 driver + the §4.3.7 IMDCT
+ the §4.3.6 window) back to a length-`N` windowed frame. The decoded frame
is compared against `window ⊙ IMDCT(X)` — the exact frame the decoder's own
IMDCT+window produce from the un-quantised analysis spectrum — clearing a
pinned **30 dB** PCM-domain SNR (≈44.7 dB measured), and the round-trip is
shown geometry-robust across block sizes 64 / 256 / 1024. This jointly
proves the floor render, §4.3.6 dot product, §4.3.7 IMDCT and §4.3.6 window
are correct end to end.

Three further integration suites cover decode paths that **no staged
fixture exercises** — every `docs/audio/vorbis/fixtures/*` stream is floor
type 1 with residue formats 1/2 only, so the floor-0 LSP path and the
residue format-0 strided-scatter layout had no end-to-end coverage despite
both being wired into the decode driver.
`tests/floor0_curve_roundtrip.rs` drives the floor-0
plan→write→decode→curve loop (`plan_floor0_coefficients` →
`write_floor0_packet` → `Floor0Decoder::decode`) and asserts the §6.2.3
LSP→envelope curve **bit-for-bit** against an independent in-test
recomputation of the Bark-map + LSP-product + `exp` synthesis — covering
both order parities, dim-1/dim-2 value books, a partial-final-vector
surplus discard, the per-`n` Bark-map recompute across block sizes, and the
§6.2.2 zero-amplitude unused short-circuit. `tests/floor0_audio_packet_decode.rs`
then drives a real floor-0 audio packet through the **full
`decode_audio_packet_pre_imdct` driver** (the §4.3.2 dispatch landing on
`FloorDecoder::Type0`, the §6.2.2-body → §4.3.4-residue bit hand-off, the
§4.3.6 dot product over a floor-0 curve), cross-checked against a
standalone `Floor0Decoder` of the same body, plus the §4.3.2 step-6 unused
path. `tests/residue_cascade_roundtrip.rs` gains **residue format-0**
coverage: a §8.6.3 strided-scatter (`read i, element j → i + j·step`)
encode→decode round-trip whose decoded residual is checked against the
hand-scattered entry reconstructions — with a cross-check that the
contiguous (format-1) interpretation of the same entry run would *not*
match, proving the encode-side gather is the exact inverse of the decode
scatter — plus a two-stage format-0 cascade refinement.

### Not yet supported / known gaps

- **No `Decoder` / `Encoder` registration** and no Ogg container
  layer — the crate exposes the codec primitives, not a wired codec. The
  integration test carries a minimal RFC-3533 page de-framer as private
  test scaffolding to feed real `input.ogg` packets to the decoder;
  production Ogg demuxing belongs in `oxideav-ogg`.
- **Deriving the target floor-0 LSP `[coefficients]`** — the VQ-encode
  glue mapping a target coefficient list onto the §6.2.2 entry run now
  exists (`floor0_encode::plan_floor0_coefficients`, see above), so the
  floor-0 WRITE primitive no longer needs hand-supplied entry indices.
  Inverting §6.2.3 curve computation to *produce* that target coefficient
  list (and choosing the per-packet `amplitude` / `booknumber`) from a
  desired floor envelope is the remaining floor-0 encode followup.
- **Choosing floor-1 `partition_cvals` master-selectors / class books** —
  fitting the per-post integer amplitudes from a linear-domain floor
  envelope now exists end to end (`floor1_envelope::plan_floor1_envelope` →
  `floor1_encode::plan_floor1_y`, see above), and the
  `tests/floor1_envelope_roundtrip.rs` chain round-trips a forward-MDCT
  envelope through encode + decode. The remaining floor-1 encode decision is
  the **partition packing**: choosing the per-partition master-selector
  `cval` values and the class / sub-book assignment that Huffman-pack the
  fitted `[floor1_Y]` most compactly. The single-partition, `subclasses = 0`
  path is exercised today (each post coded directly through one value book);
  the multi-subclass master-book selection is the open optimisation.

## Clean-room provenance

The implementation is derived entirely from the Vorbis I Specification
PDF and the OxideAV clean-room companion documents and corpus traces
under `docs/audio/vorbis/`. No third-party Vorbis implementation has
been consulted at any stage. The crate's prior implementation was
retired under the workspace clean-room policy and rebuilt from a
`NotImplemented` scaffold.

## License

MIT. See `LICENSE`.

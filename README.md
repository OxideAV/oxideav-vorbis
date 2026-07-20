# oxideav-vorbis

[![CI](https://github.com/OxideAV/oxideav-vorbis/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-vorbis/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-vorbis.svg)](https://crates.io/crates/oxideav-vorbis) [![docs.rs](https://docs.rs/oxideav-vorbis/badge.svg)](https://docs.rs/oxideav-vorbis) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust Vorbis I audio codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
Clean-room implementation against the Vorbis I Specification
(Xiph.Org) and the corpus traces under `docs/audio/vorbis/`.

## Status

**Clean-room rebuild in progress.** The crate implements the full
Vorbis I decode pipeline — **decoding real Vorbis audio packets to
sample-exact PCM** — and a complete, playable **`PCM → .ogg`
encoder** (`encode_pcm_to_ogg`): the §A.2 Ogg/Vorbis encapsulation
rules over the [`oxideav-ogg`](https://github.com/OxideAV/oxideav-ogg)
container crate's RFC 3533 page transport, and the whole
psychoacoustic / floor / residue encode stack — **§4.3.1 short/long
block switching** (transient-scheduled short blocks, per-size setup
entries, hybrid window edges) and gated **§4.3.5 square-polar channel
coupling** (a correlated stereo pair encodes −33 % vs dual-mono at
equal SNR) included — behind one quality scalar, with the wire-format
entropy (residue classwords, floor-post codewords, value books)
trained per stream, an **amplitude-banded residue class ladder**
(silence / joint 4-D noise book / a corpus-gated 4-D **mid band
book** / coarse / coarse + fine — per-partition, per-pass value-book
assignment priced by the classword-aware rate-distortion planner),
the §8.6.1 **coded-band cap** (`residue_end` stops at the 20 kHz
ATH bound — 960 of 1024 long-block bins at 44.1 kHz, the reference
streams' own cap), and the cascade value books **corpus-designed as
2-D joint lattices by default** (the scalar ladders take over past
the lattice fine ladder's coverage cap via a top-of-knob candidate
race) — black-box verified: swept encoder outputs (mono/stereo,
real-audio re-encodes with block switching,
`q ∈ {0.4, 0.7, 0.85, 0.9, 1.0}`) decode through ffmpeg to the
exact declared frame counts at SNRs matching the crate's own decoder
to 0.01 dB (the staged mono-q5 corpus re-encodes from ~2.9 kB /
27.9 dB at `q = 0.4` through ~7.0 kB / 47.9 dB at the default to
~11 kB / 55.4 dB at `q = 1`, against the 6.1 kB reference stream;
the banded ladder lifts the stereo corpus +2.9 dB at the default
and +4.5 dB at −2 % bytes at `q = 1`). `register()` installs the
codec — decoder and encoder factories plus the Matroska `A_VORBIS`
tag — into `oxideav_core::RuntimeContext`, and the dual-API
endpoints `decoder::make_decoder` / `encoder::make_encoder` are
directly callable without a registry.

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

The **residue classification-selection layer** (`residue_encode` module:
`plan_vector_classifications` + `plan_vector_residue`, over the scored
primitive `plan_partition_cascade_scored`) sits above the cascade planner
and closes the last hand-supplied residue knob. `plan_vector_partition_entries`
takes the per-partition classifications as a given; the chooser instead
*derives* them from the spectrum. For each partition it tries **every**
candidate classification, plans its cascade, and keeps the one whose
reconstruction minimises the squared distortion (`plan_partition_cascade_scored`
reports the leftover residual's norm plus a populated-stage bit-cost
proxy) — ties broken toward fewer stages (cheaper) then the lower index.
`plan_vector_residue` is the top of the stack: raw spectral residual in,
the index-aligned `classifications` + `partition_entries` a
`ResidueVectorPlan` holds out, with **no hand-supplied classifications**.
A full PCM→encode→decode round-trip (`tests/pcm_adaptive_residue_roundtrip.rs`)
splits a real `X / rendered_floor` residue into many partitions, lets the
chooser pick each partition's classification, and clears a fixed
single-coarse-class baseline by ≈ 17.5 dB PCM-domain SNR — the encoder
now plans both the residue classification and the cascade from spectrum.

The **rate-distortion residue stack** (`residue_encode` module:
`plan_vector_classifications_rd`, `plan_vector_residue_rd`,
`select_residue_config`) makes the residue choice *bit-budget aware*. The
distortion-only chooser always prefers the densest cascade that
reconstructs best; the scored primitive now also reports the exact
value-codeword bit cost (`ScoredPartitionCascade::bit_cost`, the sum of the
chosen entries' codeword lengths), and the rate-distortion chooser
minimises the Lagrangian `error_sq + lambda · bit_cost` per partition —
`lambda == 0` reduces exactly to the distortion chooser, larger `lambda`
trades distortion for fewer bits. `select_residue_config` lifts this to the
whole-vector configuration choice: given several candidate residue configs
(differing residue type, partition size, value-book table, classbook
width) it scores each candidate's rate-distortion plan and keeps the one
minimising `total_error_sq + lambda · (value_bits + classword_bits)`, the
§8.6.2 classword cost folded in at the config level. A PCM round-trip
(`tests/rate_distortion_residue_roundtrip.rs`) proves the `lambda` sweep
monotonically reduces bit cost, every rate point still decodes to finite
PCM, and the config selector flips from the fine to the cheap config as
`lambda` rises.

The **stereo coupling decision** (`synthesis` module: `coupling_energy`,
`should_couple`, `prune_coupling_steps`) gives the encoder the §4.3.5
*whether-to-couple* choice the unconditional `forward_couple*` path lacked.
`coupling_energy` measures the square-polar magnitude/angle energy split a
forward coupling would produce **without** mutating the channels;
`CouplingEnergy::angle_ratio` (`angle_energy / magnitude_energy`) is low for
a correlated pair (coupling pays off — the angle residue quantises toward
zero) and high for an anti-correlated one (ratio 4 — coupling buys nothing).
`prune_coupling_steps` drives that gate over a candidate coupling-step list
in `forward_couple_all` order, returning the subset worth applying; the
kept set round-trips cleanly through `forward_couple_all` →
`inverse_couple_all`.

The **long/short block-size decision** (`blocksize` module:
`detect_transient`, `choose_blocksize`, `plan_block_sequence`) is the
§1.3.2 / §4.3.1 encode-side selection that drives a mode's `blockflag`. A
clean-room energy-envelope transient detector splits a PCM block into
sub-frames, measures each sub-frame's energy, and flags a transient by the
peak-to-mean concentration ratio; `choose_blocksize` picks the **short**
block for a transient (to confine quantisation noise around the attack and
avoid pre-echo) and the **long** block otherwise, with the ratio threshold
as the caller's quality/bit-rate lever. `plan_block_sequence` lifts the
per-block decision to a whole stream: it walks the §4.3.8 granule
recurrence (`(n_prev + n_cur)/4` per packet) forward, deciding each
packet's flag over the lookahead a candidate long frame would smear noise
across, and returns the `blockflags` + granule walk the integrated encoder
emits verbatim. Two independent criteria call a region transient:
within-window peak-to-mean **concentration** (a sharp attack), and an
**energy rise** against the previous decision region (a sustained
loudness step — a noise burst over a tone bed — that is flat within the
window and invisible to concentration; the staged
`transient-blocksize-switch` fixture is exactly this shape, measuring
concentration ≤ 2.7 in every window).

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

The **floor-0 envelope-fit chain** (`floor0_envelope` + `floor0_lsp`
modules) closes the front of the floor-0 encode path — the §6.2.3 curve
**inverse**, the floor-0 analogue of `plan_floor1_envelope`. The §6.2.3
curve is `exp(K·amplitude·g(ω) − C)` with `g(ω) = 1/sqrt(p+q)` (the
amplitude-independent LSP **shape**, identical to `1/|A(e^jω)|` for an
all-pole LPC model whose Line-Spectral-Pair frequencies are the §6.2.3
`coefficients` — pinned to 1e-6). So fitting a desired linear-domain
envelope is the classic speech-coding chain, derived purely from the
§6.2.3 curve definition plus standard DSP identities:
`plan_floor0_lsp` folds the target's *shifted-log* envelope (the curve is
exponential in `g`, so `g` must track `ln(envelope)`, not the envelope)
onto the §6.2.3 Bark-bucket grid, inverse-DFTs it to autocorrelation lags
over the renderer's exact `ω = π·m/bark_map_size` grid, Levinson-Durbin
solves the all-pole model, and a parity-aware P/Q deflation + dense-grid
root bracketing extracts the LSP angles. `fit_floor0_amplitude` then solves
the integer per-packet `[amplitude]` in closed form (`Σ(g·t)/Σ(g²)`,
clamped into `[1, 2^bits−1]`). `plan_floor0_packet` is the **one-call**
composition — a desired envelope → `plan_floor0_lsp` → `fit_floor0_amplitude`
→ `plan_floor0_coefficients` → a write-ready `Floor0Packet::Curve`, with
neither LSP `coefficients`, `amplitude`, nor entry run supplied by hand.
`tests/floor0_envelope_roundtrip.rs` pins the planned packet round-trips
bit-for-bit through `write_floor0_packet` → `Floor0Decoder` (even and odd
order) and clears a log-domain shape SNR bar, and
`tests/floor0_pcm_roundtrip.rs` drives the whole §4.3 audio packet with an
LSP floor-0 floor end to end (PCM → encode → decode → PCM, ≥30 dB at order
14). No reference encoder emits floor 0, so self-consistency against the
crate's own decoder is the ground truth.

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

The **floor-1 partition-packing planner** (`floor1_encode` module:
`plan_floor1_partition_cvals`) closes the last hand-supplied floor-1 packet
knob: the per-partition master-selector `cval` values
(`encoder::Floor1Packet::partition_cvals`). Given the fitted packet-domain
`[floor1_Y]` vector and the stream codebooks, it walks §7.2.3 steps 5..19 in
the **write** direction. For a `subclasses == 0` class no master book is read,
so every dimension uses sub-book slot 0 and it emits `cval = 0` after
verifying slot 0 carries each target (a §7.2.3-step-18 negative book accepts
only `Y = 0`). For `subclasses > 0` it searches the master book's **used**
entries ascending for the smallest `cval` whose per-dimension slices
(`(cval >> j·cbits) & csub`) all land on sub-books that can encode the
corresponding target — the exact inverse of the decoder's steps 14/15
`cval & csub → cval >>= cbits` walk; smallest-first keeps the master codeword
short and the choice deterministic. `NoEncodableCval` is returned when no
reachable selector covers the partition's targets. With this in place,
`plan_floor1_packet` is the **one-call** floor-1 encode: a desired
linear-domain envelope → a write-ready `Floor1Packet`, composing
`plan_floor1_envelope` → `plan_floor1_y` → `plan_floor1_partition_cvals`
(its `Floor1PacketPlanError` unions the three stages). An envelope → packet →
`write_floor1_packet` → decode round-trip pins the decoded curve bit-for-bit
against `render_curve` over the planned `[floor1_Y]`, and a master/subclass
selection suite proves the chosen `cval` slices into the books that carry
each fitted amplitude (even/odd selectors, sparse master books, and the
unreachable-cval refusal). The floor-1 WRITE path now needs **neither**
`floor1_y` **nor** `partition_cvals` supplied by hand.

The **floor-1 setup-header geometry designer** (`floor1_layout` module:
`plan_floor1_x_list`, `plan_floor1_partition_layout`, `design_floor1_header`,
`min_rangebits`) closes the header-*design* gap upstream of the per-packet
chain. The §7.2.4 step-2 render draws straight integer line segments between
posts in the dB-ladder domain, so a good x-list is one whose piecewise-linear
interpolation through the chosen posts tracks the desired envelope mapped into
that ladder domain (via the §10.1 dB-table inverse). `plan_floor1_x_list`
places the explicit posts by **adaptive refinement** — from the two implicit
endpoints it repeatedly inserts the interior bin furthest from the current
reconstruction, stopping at a post budget or a worst-case ladder-error
tolerance. `plan_floor1_partition_layout` tiles a chosen post count into the
fewest §7.2.2 partitions over a catalogue of class dimensions via exact
dynamic-programming tiling (greedy descending alone dead-ends, e.g. dims {2,3}
posts 4 — the DP finds 2+2), honouring the 5-bit partition / 4-bit class-index
ceilings. `design_floor1_header` composes the two with `min_rangebits` into a
write-ready `Floor1Header` from a representative envelope and a caller-supplied
class catalogue. `tests/floor1_designed_header_roundtrip.rs` drives the full
§4.3 packet with a header **designed from the spectrum** (not hand-built) —
PCM → forward MDCT → envelope → `design_floor1_header` → fit → residue against
the rendered floor → `write_audio_packet` → decode — clearing a pinned **35 dB**
PCM-domain SNR across a 128/256/1024 block-size sweep. The floor-1 setup
header's geometry + partition grouping + carriage are now planned from
spectrum; only the value-codebook *contents* (bit-allocation) design remains.

The **floor-0 order selector** (`floor0_layout` module: `select_floor0_order`,
`select_floor0_order_rd`, `score_floor0_orders`, `suggest_floor0_params`) is
the floor-0 analogue: choosing `floor0_order`, the number of §6.2.3 LSP poles.
It sweeps a candidate order range, fits each order's LSP shape + amplitude,
renders the §6.2.3 curve the decoder would draw, and scores its log-domain
fidelity (the curve is exponential, so the natural metric is `ln`-space).
`select_floor0_order` returns the smallest order meeting an SNR target;
`select_floor0_order_rd` minimises `distortion + lambda · order` (the order a
monotone proxy for the per-pole coefficient bit cost). `suggest_floor0_params`
offers spec-grounded defaults for the surrounding header fields. No reference
encoder emits floor 0, so fidelity is the crate's own decoder render.

The **codebook content design + training stack** (`book_design` module)
closes the codebook-*content* followups both floor paths and the residue
named — a Vorbis codebook's entropy content is fully determined by its
per-entry codeword-**length** list (§3.2.1's canonical
lowest-valued-codeword rule implies the codewords), so content design
reduces to optimal length assignment plus VQ value placement:

- **Optimal codeword lengths** (`design_codeword_lengths` /
  `_dense`, `stream_cost_bits`) — the classic package-merge
  (coin-collector) construction for length-limited prefix codes:
  bit-cost-optimal lengths for a symbol-frequency table under the
  §3.2.1 constraints (lengths `1..=32`, Kraft sum exactly 1 so the tree
  is fully populated, sparse `UNUSED_ENTRY` slots, the errata-20150226
  single-entry book). Pinned against an exhaustive small-case
  brute-force oracle.
- **Book assembly + retraining** (`design_entropy_codebook`,
  `redesign_codebook`, `BookTallies`) — write-ready books from
  frequencies, and *retraining* an existing book around a measured
  distribution while preserving its shape and VQ lookup: every entry
  still unpacks to the identical §3.2.1 vector, so packets referencing
  the same entry indices decode **bit-identically** while serialising
  into fewer bits.
- **Per-subsystem emission tallies** — `tally_floor1_packet` (the
  §7.2.3 master-cval + sub-book-Y walk, mirroring
  `write_floor1_packet`'s emission exactly), `tally_residue_plans`
  (§8.6.2 classwords through the classbook via the writer's own
  grouping primitive, per-stage entries through the cascade's value
  books), and `tally_floor0_packet` (the §6.2.2 VQ entry run against
  the `[booknumber]`-selected book). All commit atomically
  (`BookTallies::record_all`). Three integration suites
  (`tests/floor1_trained_books.rs`, `tests/residue_trained_books.rs`,
  `tests/floor0_trained_books.rs`) pin the shared contract per
  subsystem: trained books decode the training corpus **bit-identically**
  while it serialises into **strictly fewer bytes**, stay carriage-legal
  through the §3.2.1 codebook writer/parser, and the sparse policy
  prunes + re-plans cleanly.
- **Closed-loop rate-aware training** (`train_residue_books_rd`) —
  alternates the §8.6.2 rate-distortion planner (which charges exact
  codeword lengths in its Lagrangian) with the sparse retrainer;
  classic alternating minimisation with a provably monotone
  non-increasing Lagrangian and fixed-point convergence detection.
- **VQ value-ladder design** (`design_value_ladder`) — the value-side
  half: 1-D Lloyd centroids for a training-scalar set, snapped to a
  `value_bits`-wide multiplicand grid whose `minimum` / `delta` are
  rounded to §9.2.2-packable floats, wrapped as a write-ready
  tessellation lookup.
- **Multi-dimensional VQ codebook design** (`design_vq_codebook`) —
  the joint-quantisation upgrade: a `dims`-dimensional lookup-type-2
  book from a flat corpus of residue sub-vectors by the classic
  split-and-refine construction (deterministic highest-distortion
  cell splitting, nearest-neighbour/centroid fixed-point refinement
  with starved-cell re-seeding), centroids snapped onto one shared
  §9.2.2-packable grid, **sparse occupancy-optimal codeword lengths**
  from the final cell populations. Per-component MSE is reported
  against the snapped book; on a correlated-pair corpus the dim-2
  design at equal bits/component clears half the scalar ladder's MSE.
  Composes with the closed-loop trainer (whose centroid ladder step
  is dimension-generic).
- **Interoperable lattice variant**
  (`design_lattice_vq_codebook` + `uniform_value_ladder`) — the same
  joint-occupancy length training over a §3.2.1 **lookup-type-1**
  product grid (a caller-supplied shared scalar ladder), the codebook
  shape widely deployed decoders accept: black-box, a common
  reference decoder binary rejects lookup type 2 outright, so this is
  the form the integrated encoder carries on the wire. Sparse or
  dense length policy (dense for subsample-designed seeds, so no
  grid cell the full stream reaches is ever pruned).
- **Whole-stream capstone** (`tests/trained_stream_roundtrip.rs`) — a
  20-frame mono PCM corpus through the full §4.3 audio-packet writer,
  floor + residue emissions tallied together, the entire codebook
  table retrained in one pass: every retrained §4.3 packet decodes to
  the bit-identical windowed PCM frame, the audio corpus shrinks, and
  the trained setup header round-trips whole through
  `write_setup_header` → `parse_setup_header` (§4.2.4 carriage).

The **psychoacoustic masking model** (`psy` module) is the encoder's
what-is-audible layer — clean-room encoder territory (the spec defines
only decode) built from textbook psychoacoustics over the spec's own
§6.2.3 Bark scale. `compute_masking` turns one frame's MDCT magnitudes
into a per-bin linear-amplitude **masking threshold**: 1-Bark
critical-band analysis, spectral-flatness tonality (the tone-masking-
noise `14.5 + z` dB vs noise-masking-tone `5.5` dB offset asymmetry),
asymmetric spreading (−27 dB/Bark down, −10 dB/Bark up) evaluated at
each bin's *continuous* Bark coordinate (no box-edge cliffs), and the
standard analytic threshold-in-quiet floor calibrated by
`PsyConfig::full_scale_db`. Two glue routines drive the encode stack
from it: `plan_psy_floor_envelope` (a floor-1 envelope target riding
`max(peak-held |X|, threshold)` — the floor shapes residue noise under
the masking curve, since the decoder multiplies the residue by the
floor per §4.3.6) and `residue_partition_weights` (per-partition
`(floor/threshold)²` noise-to-mask factors on the raw audibility
scale, each bin capped at a 40 dB excess). The **weighted residue RD
chooser** (`plan_vector_classifications_rd_weighted` /
`plan_vector_residue_rd_weighted`) charges `weights[p] · error_sq +
lambda · bit_cost`, making the Lagrangian an NMR-vs-bits trade;
all-`1.0` weights reproduce the unweighted chooser bit-for-bit.
`tests/psy_encode_roundtrip.rs` measures the payoff on a
tones + borderline-pedestal + ATH-masked-hash spectrum: the psy floor
+ weighted chooser encode 293 B against the naive magnitude-envelope
encode's 413 B (−29%) at equal transparent NMR; at a rate-starved
lambda the unweighted chooser drops the tones (NMR 17.3) while the
weighted one keeps them transparent (NMR 0.0004); the
`threshold_offset_db` margin sweeps rate 101→293 B against NMR
0.49→0.0009, monotone.

The **distortion-aware ladder step in the closed training loop**
(`train_residue_books_rd_ladder`, over
`residue_encode::replay_partition_cascade`) extends the length-only
alternating descent to the value books' *reconstruction values*:
the replay re-walks each planned §8.6.2 cascade deterministically to
recover the exact target sub-vector every entry quantised, each
exercised tessellation book's entries move to the centroid of their
targets (the classic VQ codebook update), re-expressed on a fresh
§9.2.2-packable grid. Cascade stages interact — a joint update from
stale targets can regress — so the joint and each single-book
candidate are evaluated by fresh plan passes and only the best strict
improvement is adopted: the Lagrangian is monotone non-increasing by
construction and multi-stage ladders converge stage-by-stage. A ±5
corpus against a `[-2, 1.5]` seed ladder (unreachable by re-pricing
alone) at least halves the final Lagrangian.

**Quality targeting** (`quality` module) ties the levers to one
scalar: `EncoderTuning::from_quality(q ∈ [0, 1])` expands to the
residue `lambda` (log-linear `10⁻¹·⁴ → 10⁻⁴`, recalibrated for the
four-class ladder — pricing bits in noise-to-mask units under the
weighted chooser), the psy margin (−12 dB rising to a measured
**+6 dB cap**: past it, the threshold-riding floor envelope drags
onto `|X|` across the quiet bins and the full-scale residue targets
quantise poorly — SNR stops rising while bytes climb), the floor
post budget (8 → 32), and the **fine value-ladder divisor** (192
through `q ≤ 0.7`, log-linear to 768 at `q = 1` — the top of the
knob lowers the encoder's reconstruction noise floor instead of
buying saturated-SNR density), all monotone by construction;
`solve_lambda_for_bits` bisects any monotone `rate(lambda)`
measurement to the cheapest lambda inside a bit budget.
`tests/quality_rate_curve.rs` measures the lever stack on an 8-frame
stream: rate 704 → 2048 → 2312 → 2528 → 2600 B, spectral SNR 32.7 →
35.6 → 36.6 → 37.1 → 37.3 dB, and mean NMR 0.62 → 0.013 → 0.0006 →
0.0005 → 0.0005 across `q ∈ {0, ¼, ½, ¾, 1}` — monotone in every
metric, transparent at the top; the budget solver lands the real
stream's halfway byte budget, and the ladder trainer composes with
the psy stack. `tests/encoder_quality_headroom.rs` pins the
whole-encoder knob: rate and SNR monotone across
`q ∈ {0.5, 0.7, 0.85, 1}` and `q = 1` clearing the default by
≥ 6 dB — the regression guard for the two root causes above (the
fixed fine ladder and the uncapped margin each made the top of the
knob wobble around a saturated ceiling while bytes tripled).

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

`tests/nonflat_floor_pcm_roundtrip.rs` closes the harder, more
representative encode case the flat-floor round-trip leaves open: a
**non-flat** floor-1 (seven interior posts) fitted to a smoothed `|X|`
magnitude envelope. The fidelity hinge is the encoder-side
`Floor1Decoder::render_curve` primitive (§7.2.4): because §7.2.4 step 2
draws **integer line segments** between posts, the reconstructed floor bows
away from the desired envelope *between* posts, so a faithful encoder must
carry residue against `X[k] / render_curve(floor1_y)[k]` — the exact per-bin
floor the decoder multiplies back in (§4.3.6) — not against the envelope
sampled at posts. The suite drives PCM → forward MDCT → envelope fit →
render → residue-against-the-rendered-floor → `write_audio_packet` →
`decode_audio_packet_windowed` and clears **≥35 dB** PCM-domain SNR (≈44 dB
measured) across short and long blocks. A control variant pins *why* the
rendered-floor divide is the correct one: dividing by the desired envelope
instead collapses the reconstruction to **<1 dB** SNR (a ≥10 dB gap is
asserted). A third assertion confirms the fitted floor is genuinely
non-flat (≥2× dynamic range across the band), so the test is no easier than
the flat-floor case. `render_curve` itself is unit-tested bit-identical to
both the private curve-computation and the decode-path curve.

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

`tests/stereo_coupling_roundtrip.rs` is the first **encode→decode** exercise
of §4.3.5 channel coupling (every other encode round-trip is mono with empty
coupling; §4.3.5 was previously only decode-tested via the `mode-stereo`
fixture). Stereo PCM → two windowed forward MDCTs → `forward_couple_all`
(the §4.3.5 square-polar magnitude/angle transform) → residue against a flat
floor → `write_audio_packet` (one submap carrying the coupled pair) →
`decode_audio_packet_windowed` (which runs the §4.3.5 inverse coupling) →
two channels, each clearing **≥25 dB** PCM-domain SNR against its own
`window ⊙ IMDCT(X*)` reference across a 128/256/512 block-size sweep, with a
control proving the coupling transform actually ran — the encoder's forward
coupling and the decoder's inverse coupling are proven to compose to the
identity on the L/R signal.

`tests/residue_format2_roundtrip.rs` closes the §8.6.5 **residue format-2**
encode→decode gap (previously only the decoder's interleave/de-interleave
was unit-tested; every packet round-trip used a per-channel format-1
residue). Two-channel PCM → two windowed MDCTs → flat floor → the two
per-channel residue targets **interleaved** into one virtual vector
(`interleaved[i·ch + j] = channel[j][i]`, the exact inverse of the §8.6.5
step-3 de-interleave) → `plan_partition_cascade` over the single interleaved
vector → `write_audio_packet` with a `residue_type: 2` header (one residue
plan) → `decode_audio_packet_windowed` (which interleaves, format-1-decodes,
and de-interleaves) → two channels, each clearing **≥30 dB** PCM-domain SNR
(≥25 dB across a 128/256/512 block-size sweep) — the §8.6.5 interleave is
proven the exact inverse of the decode-side de-interleave end to end.
`tests/residue_format0_roundtrip.rs` does the same for **format 0** (§8.6.3
strided scatter, `read i, element j → i + j·step`): mono PCM → flat floor →
`plan_partition_cascade` with `residue_type = 0` (the encode-side strided
**gather**) over a 2-D value book → `write_audio_packet` (`residue_type: 0`)
→ `decode_audio_packet_windowed` (the strided scatter) → PCM, clearing
**≥25 dB** PCM-domain SNR — the first audio-packet-level format-0 round-trip
(format 0 previously had only isolated residue-body coverage). All three
residue formats now have an end-to-end §4.3 packet round-trip.

### Ogg carriage + the wired codec

The **RFC 3533 Ogg page transport is the
[`oxideav-ogg`](https://github.com/OxideAV/oxideav-ogg) container
crate's job** — the in-crate page layer (`ogg` module) and the
`oggmux` muxer were removed in its favour. What stays here is the
Vorbis-specific §A.2 mapping, in `oggfile`: `mux_vorbis_stream`
(same signature as before) verifies the §4.2.1 header ordering and
the non-decreasing granule positions, packages the three headers as
the Xiph-laced codec-private blob (`lace_vorbis_headers`) on the
container `StreamInfo`, carries each audio packet's end-PCM granule
on `Packet::pts`, and drives the page-boundary policy (RFC 3533
soft 4 kB audio page target, plus a forced break before the final
packet so the EOS page carries the end-trim granule with an exact
blocksize-walk anchor on the page before it) through the container
crate's `unit_boundary` flag; `ogg_packets` is the inverse
de-framing convenience over `oxideav_ogg::page::Page`. The §A.2
page rules still hold on the wire: identification header alone on
the 58-byte BOS first page, comment + setup from page 1 with audio
beginning fresh, header pages at granule 0, audio pages stamped
with the end-PCM position of the last packet completed on the page
(`-1` for spanned pages), EOS on the final page with the end-trim
granule. `tests/ogg_framing.rs` pins the dependency against the
corpus (every page of all 16 staged real-world streams parses with
a verifying CRC and re-serializes **byte-for-byte**);
`tests/ogg_vorbis_remux.rs` de-frames all 15 single-stream
fixtures, recomputes every packet's granule from the §4.3.1
blocksize walk, remuxes, and asserts the identical packet sequence
plus every §A.2 page rule. Black-box, all 15 remuxes decode
through ffmpeg to **exactly their declared final-granule length**,
with ffmpeg's decode of each original a bit-identical prefix (three
byte-identical; on the rest ffmpeg under-reads the original's own
pagination by 128–350 samples that the remux recovers).

The **whole-stream encoder** (`oggfile` module) is the integrated
`encode(pcm) → .ogg` path: `encode_pcm_to_ogg` (and its packet-level
form `encode_pcm_to_packets`) composes the §4.3.1 block-size schedule
(default `2048/256` — the staged corpus geometry), the §4.3.8-inverse
framing splitter (per-frame §4.3.1 windows, hybrid edges included),
the bare §4.3.7 forward MDCT at the derived `4/n`
unity-reconstruction scale, per-channel psychoacoustics, per-size
floor-1 design/fit, the gated §4.3.5 forward channel coupling, and
NMR-weighted RD residue planning over an **amplitude-banded §8.6.1
class ladder** per partition: silence, a joint **4-dimensional
ternary noise book** (one codeword per four bins — near-threshold
texture at a fraction of the scalar cost; designed per stream from
its own quiet partitions), a corpus-gated **4-D 625-entry mid band
book** (five levels per dimension, ladder reach at the median
above-noise partition peak — the same joint-dimensionality rate
mechanism one amplitude tier up; carried only when the corpus
separates, so a uniform-loudness stream keeps the base four
classes), coarse-only, and the coarse + fine two-stage cascade.
Class choice is **priced against the codebook cost model end to
end**: the chooser charges each candidate's exact value-codeword
bits *plus* a per-class marginal classword price (`-log2 p(class)`
from a first planning pass, re-planned once — without it, a class
adopted for a marginal value-bit win inflates the trained classword
entropy by more than it saves). The §8.6.1 `residue_end` caps the
coded band at the 20 kHz ATH bound (960 of 1024 long bins at
44.1 kHz — the reference streams' own cap; uncapped when Nyquist
sits below the cutoff), and every design/training/planning stage
follows the coded band.
The residue partitions scale with the spectrum (16 short / 32 long);
the value ladders are §9.2.2-packable with the fine ladder's step
following the quality knob (see below), and the cascade books are
**corpus-designed 2-D lookup-type-1 lattices by default**
(`vq_dims = 2`): one trained codeword jointly codes two neighbouring
bins, which on the staged mono corpus at the default quality spends
−22 % audio bytes at +6.3 dB SNR against the scalar ladders. The
joint geometry rules the low and middle of the knob; past the
lattice fine ladder's 2× **coverage cap** (`q > 0.85` — the
per-dimension level count is pinned by the entry ceiling, so the
joint books' resolution saturates there) the encoder races the
scalar-ladder geometry at the requested quality against the joint
geometry frozen at its cap point and keeps the higher own-decoded
SNR, so the knob stays monotone on every staged corpus (`vq_dims =
1` forces the scalar geometry throughout). The three §4.2 header
writers plus §A.2 encapsulation carry the mixed-size
`(n_prev + n_cur)/4` granule walk end-trimmed to the exact input
length — all behind the one `quality ∈ [0, 1]` scalar.
The wire-format entropy is trained per stream: the residue classbook
groups **four partitions per §8.6.2 classword** and, after the final
rate-distortion plans are made, both the classword lengths and the
§7.2.3 **floor-post book lengths** are retrained occupancy-optimal
for the exact emissions the packets make (dense policy —
decode-identical PCM, strictly fewer bits).
`decode_ogg_to_pcm` is the inverse convenience (de-frame, header
parse, streaming decode, end-trim). `tests/ogg_encode_roundtrip.rs`
pins the §A.2 structure of the produced stream and the round-trip
fidelity (knob spread 2.1 kB / 16.6 dB at `q = 0.2` → 5.4 kB /
34.1 dB at `q = 0.9` on its tonal corpus; the weighted closed-loop
codebook training cuts a stream 9.9 → 8.4 kB at identical SNR — the
designed lattice seeds already sit near the trained optimum, where
the old scalar seeds took a 15.2 → 8.1 kB cut); black-box, the swept fixture re-encodes
decode through ffmpeg to their exact declared frame counts at SNRs
matching the crate's own decoder to 0.01 dB.

**§4.3.1 block switching** is wired through that whole chain
(`tests/ogg_block_switching.rs`): `blocksize::plan_block_sequence`
walks the §4.3.8 granule recurrence forward, deciding each packet's
`blockflag` with the energy-envelope transient detector over the
`long_n` lookahead a candidate long frame's quantisation noise would
smear across; the setup header carries a floor / residue / mapping /
mode set per block size, every long packet's window flags mirror its
neighbours' blockflags (§4.3.1 hybrid edges at each long↔short
transition — the encoder-side `FrameSplitter` grew the short→long
negative-stride zero-fill this needs, pinned by a mixed-size
FrameSplitter → OverlapAdd chain that reconstructs the input exactly),
and codebook training chains the short- and long-block residue corpora
over the shared ladders. Measured: on an attack-after-silence corpus,
switching cuts the pre-attack noise energy beyond the short block's
intrinsic `n0/2` reach by **~180×** against a forced-long encode at
equal quality; black-box, ffmpeg decodes the switched fixture
re-encodes to their exact declared lengths at SNRs matching the
crate's own decoder to 0.01 dB. The real-audio corpus re-encodes
cleanly (`tests/fixture_reencode.rs`): the
`transient-blocksize-switch` fixture schedules shorts at its
noise-burst onset (the energy-rise criterion), steady music stays
all-long at 47.9 dB at the default quality (55.6 dB at `q = 1`), and
the decorrelated stereo fixture is correctly left uncoupled — all
end-trim-exact through the crate's own decoder, with **two-sided
regression gates** (an audio-byte ceiling AND an SNR floor) at the
pinned quality points.

**§4.3.5 channel coupling** is likewise wired
(`tests/ogg_coupled_stream.rs`): adjacent channel pairs are gated on
the whole stream's square-polar energy split (angle ≤ half the
magnitude energy, accumulated over every frame's residue targets),
kept steps land in every mapping and are forward-coupled over the
residue targets (`X / rendered_floor`, the exact vectors the decoder
inverse-couples), and each coupled pair's per-partition NMR weights
merge to the element-wise max. Measured: a correlated stereo corpus at
`q = 0.7` encodes to **12.0 kB coupled vs 18.0 kB dual-mono (−33 %)**
at equal per-channel SNR (32.8 dB / 32.7 dB); an anti-correlated pair
fails the gate and stays uncoupled; a 4-channel stream gates each pair
independently.

**Registration + dual API**: `register()` installs the codec
(decoder + encoder factories, Matroska `A_VORBIS` tag) into
`oxideav_core::RuntimeContext`. `decoder::VorbisDecoder` is the
packet-to-frame `oxideav_core::Decoder` (in-band order-checked §4.2
headers, §4.3 audio through the streaming pipeline, planar-f32
frames with sample-granularity PTS, seek-safe `reset()`);
`encoder::VorbisStreamEncoder` is the frame-to-packet
`oxideav_core::Encoder` (buffers F32/F32P frames, two-pass encode at
`flush()`, header-flagged §4.2 packets then timestamped audio
packets; `"quality"` / `"blocksize"` / `"short_blocksize"` /
`"coupling"` options).
`tests/registry_wiring.rs` round-trips PCM through boxed trait
objects on both the registry and the direct dual-API path.

The **temporal masking extension** (`psy::TemporalMasking`) adds the
cross-frame component the per-frame model lacked: post-masking (a
masker's threshold elevation decays across following frames at a
configurable dB/ms rate, default 0.5) and pre-masking (a loud onset
lifts the previous frame's threshold via one frame of lookahead,
attenuated 12 dB). Pinned structural properties: never below the
per-frame model, identical on steady-state signals, monotone decay
back after a burst. `tests/psy_temporal_masking.rs` validates the
payoff NMR-style at equal transparency: on a burst-then-low-comb
transient corpus the temporal encode spends **−14 %** bytes at fixed
lambda, both encodes transparent under their own model, and on a
steady corpus the two models are byte-identical. The whole-stream
encoder runs its thresholds through this pipeline per channel on
uniform-blocksize streams; a genuinely switched stream has a variable
frame hop the temporal model does not define, so it uses the per-frame
model (pre-echo control there rests on the short blocks themselves —
the §1.3.2 mechanism).

### Not yet supported / known gaps

- **The psy tonality estimate is band-level spectral flatness** (no
  phase-predictability tracking), the perceptual weighting enters the
  residue chooser per *partition* (the VQ entry selection inside
  `vq::quantize_vector` is unweighted Euclidean per read), and the
  coupling gate is energy-driven, not masking-driven (and whole-stream:
  one coupling decision per pair, since the §4.2.4 mapping fixes the
  steps for every packet using it — per-packet coupling choice would
  need a second uncoupled mapping per block size).
- **The transient detector is two global thresholds** (peak-to-mean
  concentration + energy rise over a 16-sub-frame lookahead); it is not
  loudness-adaptive, and the block schedule is decided on a channel
  mixdown, so a transient confined to one channel of an uncoupled pair
  still switches both.
- **The re-encoded audio rate still trails the reference corpus at
  the knee.** On audio-packet bytes the default-quality re-encode
  spends ~2.4× the reference stream's audio bytes (mono-q5: ~4.9 kB
  vs 2.1 kB — down from 6.1 kB scalar) at a measured 47.9 dB, with
  audio-parity near `q ≈ 0.55` at ~35 dB. The amplitude-band mid
  class and the coded-band cap landed (the banded ladder pays mostly
  above the knee: stereo default +2.9 dB, `q = 1` +1.7…+4.5 dB at
  fewer bytes; the knee itself trades ≤ +4 % bytes at equal SNR);
  the remaining structural gap versus the reference's class set
  (10 classes, per-band books up to 8-D) is **deeper band tiers**
  (an 8-D near-silence tier, per-band coarse dimensionality) and
  cross-frame entropy. Measured and rejected across r416–r420: a
  third-stage ultra lattice (+36 % audio bytes for +0.1 dB), an
  in-ladder scalar/joint hybrid class (routing collapse under the
  unweighted trainer; still byte-dominated under the weighted one),
  and a same-dimensionality narrower-span per-band coarse + fine
  pair (+4…+13 % audio bytes at identical SNR — occupancy-trained
  lengths already price amplitude statistics inside one book; the
  per-band win must come from joint dimensionality, which is what
  the mid band book does).
- **The stereo quiet-channel top end trades against the +6 dB margin
  cap.** The old uncapped margin (+12 dB at `q = 1`) pushed the psy
  floor onto `|X|`, switching the encoder into waveform coding — on
  the decorrelated stereo fixture that lifted the quiet channel's
  `q = 1` SNR to ~56 dB at +37 % bytes; with the cap the min channel
  reads ~31 dB (mean 43 dB, −30 % bytes vs the old encoder's top).
  A per-channel (or content-adaptive) margin that deepens only where
  waveform coding pays is the follow-up.

## Clean-room provenance

The implementation is derived entirely from the Vorbis I Specification
PDF and the OxideAV clean-room companion documents and corpus traces
under `docs/audio/vorbis/`. No third-party Vorbis implementation has
been consulted at any stage. The crate's prior implementation was
retired under the workspace clean-room policy and rebuilt from a
`NotImplemented` scaffold.

## License

MIT. See `LICENSE`.

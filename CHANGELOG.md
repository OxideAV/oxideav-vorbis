# Changelog

All notable changes to `oxideav-vorbis` are recorded here.

## [Unreleased]

### Fixed

- **Closed-loop trainers could prune a classword the planner later
  emits.** `train_residue_books_rd` / `train_residue_books_rd_ladder`
  length-retrained *every* book with the sparse policy, including the
  residue **classbook** — but the rate-distortion planner's
  per-partition class choice ranges over the header's whole class set
  without consulting classword availability, so a class no plan
  picked in one iteration lost its codeword and a later iteration (or
  the final packets) that selected it failed with
  `UnusedSymbolHasFrequency`. Reachable with any residue whose class
  set offers a candidate the early iterations never pick (e.g. a
  fine-only class whose book span starts smaller than the targets).
  The classbook now retrains with the **dense** policy — every class
  keeps a (long) codeword, still occupancy-shaped — while value books
  keep the exactly-optimal sparse policy; regression-pinned by
  `trainer_keeps_every_classword_encodable`.

### Added

- **Occupancy-trained floor-post codeword lengths in the integrated
  encoder.** The shared §7.2.3 floor post book shipped uniform
  8-bit codewords for its 256 entries; the encode path now tallies
  the exact fitted `floor1_y` emissions (`tally_floor1_packet`, the
  writer's own §7.2.3 walk) alongside the residue classwords and
  retrains the post book's lengths occupancy-optimal (dense policy —
  every post amplitude stays encodable). Pure entropy win: packets
  decode to bit-identical PCM. Measured on the staged real-audio
  corpus at the default configuration (on top of the classword win
  below): mono-q5 8310 → 7763 B, stereo-q5 14 767 → 13 486 B,
  transient 8844 → 8137 B — cumulatively **−8.7 / −10.8 / −10.0 %**
  against the previous round's encoder at identical PCM, closing the
  re-encode ratio to ×1.27 / ×1.47 / ×1.23 of the reference streams.

- **Grouped, occupancy-trained residue classwords in the integrated
  encoder.** The residue classbook now packs four partitions per
  §8.6.2 classword (`dimensions = 4`, radix-packed — the classword
  form the reference corpus streams carry) and, after the final
  rate-distortion plans are made, its codeword lengths are retrained
  occupancy-optimal for the exact classwords the packets emit (dense
  policy via the writer's own `tally_residue_plans` grouping, so
  every class group stays encodable). A common run — e.g. four
  consecutive silent high-band partitions — now costs a couple of
  bits instead of four separate codewords. Measured on the staged
  real-audio corpus at the default configuration: −2.2 to −2.4 %
  stream bytes at **bit-identical audio packets' PCM** (same plans,
  same SNR — a pure classword-entropy win): mono-q5 8503 → 8310 B,
  stereo-q5 15 126 → 14 767 B, transient 9043 → 8844 B. The encode
  path now plans all frames first, trains the classbook, then writes
  the setup + audio packets.

- **Multi-dimensional residue value books in the integrated encoder**
  (`StreamEncoderConfig::vq_dims`, default `1`): at `vq_dims > 1` the
  whole-stream encoder designs its two §8.6.2 cascade value books
  from the stream's own residue corpus via `design_vq_codebook` —
  `vq_dims`-dimensional §3.2.1 lookup-type-2 tessellation books, the
  coarse book over the raw dims-element residue sub-vectors and the
  fine book over the post-coarse leftovers (exactly the second
  stage's targets, since the cascade planner subtracts the chosen
  entry's decoded reconstruction), on a bounded deterministic
  stride-subsample of the corpus. Every legal dimensionality
  (`1 | 2 | 4 | 8 | 16` — a power of two dividing the partition
  size; anything else is refused with the new
  `OggFileError::BadVqDims`) encodes and decodes end-trim-exact
  through the crate's own decoder with coupling + block switching
  live, and the closed-loop trainer refines the designed books
  unchanged. Measured (`tests/multidim_residue_books.rs`): at
  `q = 0.7` on the synthetic tones + noise corpus the dim-2 designed
  books lift the reconstruction ceiling ≈ 28 → 32 dB SNR at ≈ 1.3×
  the bytes — corpus-fitted joint reconstruction points the generic
  per-scalar ladders cannot reach — while real-audio re-encodes keep
  the scalar default (a fixed entry budget spreads over
  `entries^(1/dims)` levels per scalar, so high-rate content favours
  `vq_dims = 1` until the residue class set can adapt per partition).

- **Multi-dimensional VQ residue codebook designer**
  (`book_design::design_vq_codebook` + `VqCodebookDesign`): designs a
  `dims`-dimensional §3.2.1 lookup-type-2 value book from a flat
  corpus of residue sub-vectors by the classic split-and-refine
  vector-quantiser construction — deterministic highest-distortion
  cell splitting, nearest-neighbour/centroid fixed-point refinement
  with starved-cell re-seeding, converged centroids snapped onto one
  shared §9.2.2-packable `minimum/delta` multiplicand grid, and
  **sparse occupancy-optimal codeword lengths**
  (`design_codeword_lengths` over the final cell populations, empty
  cells left `UNUSED_ENTRY`). The per-component MSE is reported
  against the snapped book, directly comparable to
  `design_value_ladder`; on a correlated-pair corpus the dim-2 design
  at equal bits/component clears half the scalar ladder's MSE. New
  `BookDesignError::{ZeroVqDimensions, ZeroEntries,
  TrainingNotVectorAligned}` variants type the new validation. The
  designed books are carriage-legal (`write_codebook` →
  `parse_codebook` identical), quantise through `vq::quantize_vector`
  to their exact §3.2.1 unpacks, and compose with the closed-loop
  `train_residue_books_rd_ladder` trainer (its centroid ladder step
  was already dimension-generic).

- **Energy-rise transient criterion + fixture-corpus re-encode.**
  `plan_block_sequence` gains a second, independent transient
  criterion (new `energy_rise_threshold` parameter): the lookahead's
  peak sub-frame energy against the previous decision region's mean —
  catching a sustained loudness step (a noise burst over a tone bed)
  that is flat *within* the window and therefore invisible to the
  peak-to-mean concentration ratio, exactly the shape the staged
  `transient-blocksize-switch` fixture carries (measured concentration
  ≤ 2.7 across its windows). The integrated encoder passes `4.0`
  (+6 dB over one lookahead). New `tests/fixture_reencode.rs` drives
  the staged **real-audio** corpus back through the default encoder:
  the transient fixture now schedules short blocks at the burst onset
  and round-trips end-trim-exact; steady music stays all-long at
  47.1 dB; and the genuinely decorrelated stereo fixture
  (angle/magnitude energy ratio ≈ 1.7) is correctly left uncoupled by
  the profitability gate.

- **Depth tests over the two new subsystems**: the closed-loop
  ladder trainer composes with coupling + switching — retraining on
  the coupled magnitude/angle residue corpus (chained per block size)
  cuts a coupled+switched stereo stream **15 228 → 12 161 B (−20 %)**
  at identical per-channel SNR
  (`tests/ogg_coupled_stream.rs`); a dense-impulse corpus schedules
  predominantly short blocks (long only on the silent stream tail)
  and exercises the unused-size floor design via envelope resampling
  (`tests/ogg_block_switching.rs`).

- **Registry encoder options for the new subsystems**: the
  `make_encoder` factory grows a `"short_blocksize"` option (power of
  two `<=` the long size; equal disables §4.3.1 block switching) next
  to the existing `"blocksize"`, with a lone `"blocksize"` below the
  default short size clamping the short size down (so a small
  single-blocksize request stays legal), plus shape/order guards; the
  `"coupling"` boolean option (added with the coupling subsystem)
  rejects non-boolean values. `tests/registry_wiring.rs` pins the
  option matrix.

- **§4.3.1 block switching in the integrated encoder.**
  `encode_pcm_to_ogg` / `encode_pcm_to_packets` now run short/long
  block switching — new `StreamEncoderConfig::short_blocksize` knob
  (default `256` against the long `1024`; setting it equal to
  `blocksize` disables switching). New pub
  `blocksize::plan_block_sequence` (+ `BlockSequencePlan`) walks the
  §4.3.8 granule recurrence forward, deciding each packet's
  `blockflag` with the energy-envelope transient detector over the
  `long_n` lookahead a candidate long frame's quantisation noise
  would smear across. The stream carries per-size floors / residues /
  mappings / modes (the setup header grows a second entry set), each
  long packet's window flags mirror its neighbours' blockflags
  (§4.3.1 hybrid edges at every long↔short transition), and the
  §A.2 granules follow the mixed-size walk. A genuinely switched
  stream uses the per-frame psy model (the temporal pipeline needs a
  fixed hop); uniform streams keep the temporal path. Codebook
  training chains the short- and long-block residue corpora over the
  shared ladders. Measured (`tests/ogg_block_switching.rs`):
  short blocks cut the pre-attack noise energy beyond the short
  block's intrinsic reach by **220×** (1.7 × 10⁻⁶ vs 3.7 × 10⁻⁴)
  against a forced-long encode at equal quality and equal whole-stream
  SNR; black-box, switched mono and switched+coupled stereo 1 s
  encodes decode through ffmpeg to exactly 44 100 frames, ffmpeg
  agreeing with the crate's own decoder to **134 dB** (max sample
  delta 2.1 × 10⁻⁷).
- **`FrameSplitter` short→long geometry fix.** The §4.3.8-inverse
  splitter mis-placed a frame following a smaller predecessor (the
  negative-stride case): the frame's global start precedes the
  buffered head by `cur_n/4 − prev_n/4`, which the old code ignored,
  drifting the walk. Those positions fall entirely inside the long
  block's §4.3.1 zero lead-in, so `take_frame` now zero-fills them
  and aligns the buffered samples at the window's rising edge,
  draining `cur_n/2 − lead` to keep the head-at-center invariant. A
  new mixed-size chain test drives a short/long flag sequence with
  the real hybrid windows through FrameSplitter → OverlapAdd and
  reconstructs the input exactly (the cross-transition w² TDAC
  identity).

- **§4.3.5 channel coupling in the integrated encoder.**
  `encode_pcm_to_ogg` / `encode_pcm_to_packets` now offer square-polar
  coupling on adjacent channel pairs `(0, 1)`, `(2, 3)`, … — new
  `StreamEncoderConfig::coupling` knob (default `true`) and a
  `"coupling"` option on the registry encoder factory. Each candidate
  pair is gated on the whole stream's coupling-energy split
  (`synthesis::coupling_energy` accumulated over every frame's residue
  targets; kept only when the angle energy stays under half the
  magnitude energy), kept steps are recorded as mapping coupling steps
  in the setup header and forward-coupled over the residue targets
  (`X / rendered_floor`, the exact vectors the decoder's §4.3.5
  inverse coupling recovers), and each coupled pair's per-partition
  NMR weights merge to the element-wise max (error in either coupled
  vector spreads into both output channels). Measured on a correlated
  stereo corpus at `q = 0.7`: **14 024 B coupled vs 20 556 B
  dual-mono (−32 %) at equal per-channel SNR** (32.2 dB vs 32.2 dB);
  black-box, a 1 s coupled stereo `q = 0.8` encode decodes through
  ffmpeg to exactly 44 100 frames at 46.3 dB SNR. An anti-correlated
  pair fails the gate and stays uncoupled
  (`tests/ogg_coupled_stream.rs`).

### Changed

- **Ogg carriage now rides the `oxideav-ogg` container crate** (new
  dependency, `0.1`, unconditional). The Vorbis-specific §A.2
  mapping stays in-crate in `oggfile`: `mux_vorbis_stream` keeps its
  signature and the §4.2.1 header-order + non-decreasing-granule
  checks, hands the three headers to the container layer as the
  Xiph-laced codec-private blob (new pub `lace_vorbis_headers`),
  carries audio granules on `Packet::pts`, and signals the page
  policy (soft 4 kB audio page target + a forced break before the
  final packet so the EOS end-trim granule has an exact
  blocksize-walk anchor on the previous page) via
  `PacketFlags::unit_boundary`; new pub `ogg_packets` de-frames a
  physical stream through `oxideav_ogg::page::Page`.
  `decode_ogg_to_pcm` / `encode_pcm_to_ogg` keep their signatures
  and behaviour (pagination differs from the old in-crate muxer;
  the §A.2 page rules and the end-trim are unchanged). Black-box:
  the 1 s stereo `q = 0.8` encode decodes through ffmpeg to exactly
  44100 frames (agreeing with the crate's own decoder to 117 dB,
  max sample delta 1.3 × 10⁻⁶), and all 15 single-stream fixture
  remuxes decode through ffmpeg to exactly their declared
  final-granule length with the original's decode a bit-identical
  prefix.
- `MuxError` moved from the removed `oggmux` module into `oggfile`
  (variants: `HeaderOrder` / `Classify` / `NonMonotoneGranule` /
  `Container`); `OggFileError::Ogg` now carries the container page
  parser's rendered message as a `String`.

### Removed

- The in-crate RFC 3533 page layer (`ogg` module: `OggPage`,
  `parse_pages`, `pages_to_packets`, `PacketAssembler`,
  `PageWriter`, `ogg_crc32`, `MAX_PAGE_SEGMENTS`,
  `OGG_CAPTURE_PATTERN`, `PAGE_HEADER_LEN`) and the `oggmux` module
  (`VorbisOggMuxer`) — superseded by the `oxideav-ogg` dependency;
  the fixture framing-conformance gate (`tests/ogg_framing.rs`) now
  pins the dependency byte-for-byte against the corpus instead.

### Added

- **Closed-loop codebook training wired into the integrated encoder**
  — `StreamEncoderConfig::training_iterations` (default 4) runs
  `train_residue_books_rd_ladder` over the stream's own residue
  targets before packet planning, replacing the generic seed ladders
  with usage-priced, centroid-placed, §9.2.2-packable trained books
  in the emitted setup header. The ladder *value* step now also
  covers **1-D lattice books** (lookup type 1 with
  `dimensions == 1`, where the scalar table is directly
  entry-indexed — the interoperable shape the integrated encoder
  emits), rebuilding them in their own lookup type; multi-dimensional
  lattices stay untouched (their entries are permutations of a shared
  table). The flat 2-entry classbook is kept as seeded — the trainer
  plans unweighted and could starve a class the NMR-weighted final
  plan selects. Measured: a 16 k mono stream drops 16 760 → 8 433 B
  at *better* SNR (36.8 → 37.7 dB); the registry round-trip stream
  drops 13.8 kB → 6.0 kB at identical SNR; black-box, the 1 s stereo
  `q = 0.8` encode drops 64.4 kB → 28.3 kB (−56 %) and still decodes
  through ffmpeg to exactly 44100 frames at 30.3 dB.
- **Framework registration + dual API** — `register()` is no longer a
  no-op: it installs one `"vorbis"` codec into
  `oxideav_core::RuntimeContext` with both factories and the Matroska
  `A_VORBIS` tag claim. `decoder::make_decoder` builds
  `VorbisDecoder` (a packet-to-frame `oxideav_core::Decoder`: the
  three §4.2 header packets in-band and order-checked, §4.3 audio
  packets through the streaming pipeline, planar-f32 `AudioFrame`s
  with sample-granularity PTS, `reset()` clearing the §4.3.8 overlap
  state). `encoder::make_encoder` builds `VorbisStreamEncoder` (a
  frame-to-packet `oxideav_core::Encoder` over the new
  `encode_pcm_to_packets`: buffers planar/interleaved F32 frames,
  runs the two-pass whole-stream encode at `flush()`, and emits the
  three header packets flagged `header` followed by audio packets
  with sample timestamps; `"quality"` / `"blocksize"` options).
  `encode_pcm_to_ogg` is now a thin §A.2 mux over
  `encode_pcm_to_packets` / `EncodedVorbisStream`.
  `tests/registry_wiring.rs` round-trips PCM through boxed trait
  objects on both the registry path and the direct dual-API path
  (23.5 dB at strict-margin quality on a tonal corpus), checks the
  tag resolution, header flagging, PTS bookkeeping, factory option
  guards, and out-of-order header rejection.
- **Temporal masking (`psy::TemporalMasking`)** — the cross-frame
  extension of the r387 Bark-domain model. Post-masking: a masker's
  threshold elevation decays across following frames at a configurable
  dB/ms rate (default 0.5 dB/ms — a 60 dB elevation erased in
  ~120 ms) via `eff[f] = max(fresh[f], eff[f−1]·10^(−decay/20))`.
  Pre-masking: one frame of lookahead projects the next frame's fresh
  threshold backwards under a 12 dB attenuation, a pre-echo-tolerant
  discount just ahead of an onset — the driver is a one-frame
  lookahead pipeline (`push_frame` emits the previous frame,
  `finish()` drains). Three structural properties are pinned: the
  effective threshold never sits below the per-frame model, it is
  *identical* on steady-state signals, and the post-masking tail
  decays monotonically back to the per-frame threshold.
  `tests/psy_temporal_masking.rs` validates the payoff NMR-style at
  equal transparency: on a burst-then-low-comb transient corpus the
  temporal encode spends 717 B vs the per-frame model's 837 B
  (−14%), both transparent under their own model (mean NMR 0.034 /
  0.003), the temporal encode bounded under the nominal model; on a
  steady corpus the two models produce byte-identical streams. The
  whole-stream encoder now runs its thresholds through this pipeline
  per channel (lookahead is free in a whole-stream encode).
- **Whole-stream encoder + decoder entry points (`oggfile` module)**
  — `encode_pcm_to_ogg(pcm, &StreamEncoderConfig) → Vec<u8>` is the
  crate's first integrated `PCM → playable .ogg` encoder: §4.3.8-
  inverse framing split (single-blocksize geometry, `n/2` zero
  pre-roll, tail pad), bare §4.3.7 forward MDCT at the derived `4/n`
  unity-reconstruction scale (kernel identity `mdct(imdct) = n/2·`
  times the windowed-TDAC ½), per-frame/channel psychoacoustic
  masking + psy floor envelope, floor-1 header design from the
  corpus-max envelope + per-frame post fit, rendered-floor residue
  targets with NMR partition weights, the perceptually weighted RD
  residue chooser, §9.2.2-packable signed lattice value ladders
  (lookup type 1 for black-box interoperability), the three §4.2
  header writers, and §A.2 encapsulation with `f·n/2` granule
  positions end-trimmed to the exact input length. Any channel count,
  uncoupled. One `quality ∈ [0,1]` scalar drives margin, post budget
  and λ. `decode_ogg_to_pcm` is the inverse convenience (de-frame,
  header parse, §4.3 streaming decode, §A.2 end-trim).
  `tests/ogg_encode_roundtrip.rs` pins §A.2 structure on the produced
  stream, mono 27.6 dB / stereo 29.7 + 22.7 dB time-domain SNR at
  `q = 0.7`, the rate/fidelity quality trade on the real .ogg
  (1.25 kB / 0.9 dB at `q = 0.2` → 9.4 kB / 6.0+ dB at `q = 0.9` on
  an 8k corpus), sub-block and silent streams, n = 256 geometry, and
  the shape guards. Black-box (local): a 1-second stereo `q = 0.8`
  encode decodes through ffmpeg to exactly 44100 frames at
  **30.8 dB SNR** against the input — `encode(pcm) → .ogg` plays
  through an independent decoder end to end.
- **Vorbis I §A.2 Ogg encapsulation (`oggmux` module)** —
  `VorbisOggMuxer` / `mux_vorbis_stream` apply the Vorbis mapping
  rules on top of the RFC 3533 page writer: identification header
  alone on the 58-byte BOS first page, comment + setup from page 1
  with the setup header finishing its page, audio beginning fresh,
  header pages at granule 0, audio pages stamped with the end-PCM
  position of the last packet completed (−1 when spanned), EOS on the
  final page with the §A.2 end-trim granule passed through. Packet
  order is enforced with the §4.2.1 classifier and granule positions
  must be non-decreasing. `tests/ogg_vorbis_remux.rs` de-frames all
  15 single-stream fixtures, recomputes each packet's granule from
  the §4.3.1 blocksize walk, remuxes, and asserts the identical
  packet sequence, every §A.2 page rule, and per-page granule
  bookkeeping; black-box: a remuxed fixture decodes through ffmpeg to
  the full declared duration, sample-identical (±1) to the original
  decode.
- **RFC 3533 Ogg page framing (`ogg` module)** — the transport layer
  under the Vorbis I §A encapsulation, both directions. Read side:
  `OggPage::parse` / `parse_pages` (capture pattern, version, CRC
  verify) and `PacketAssembler` (lacing→packet coalescing across
  `continued` page boundaries, serial-locked). Write side:
  `OggPage::serialize` and `PageWriter` (packet→255-byte lacing
  segmentation, auto page emit on a full segment table with the
  `continued` flag on the follow-on page, last-completed-packet
  granule stamping with `-1` for spanned pages, BOS on the first
  page, EOS on `finish()` — including re-stamping an already-flushed
  final page in place with a recomputed CRC). The §6 item 7 CRC
  (polynomial `0x04c11db7`, MSB-first, zero init, no final XOR) is
  pinned by `tests/ogg_framing.rs`: every page of every staged
  real-world fixture stream parses with a verifying CRC and the
  parsed pages re-serialize **byte-for-byte** to each `input.ogg`,
  chained streams included; every logical stream's first packet
  parses as a §4.2.2 identification header, and the §A.2 58-byte
  first-page shape is asserted.

- **Psychoacoustic masking model (`psy` module)** — the encoder-side
  perceptual layer the README named as the last quality gap. The
  Vorbis I spec defines only decode, so this is clean-room encoder
  territory built from textbook psychoacoustics over the spec's own
  §6.2.3 Bark scale: 1-Bark critical-band analysis of the MDCT
  magnitudes, per-band tonality from spectral flatness (tone- vs
  noise-masker offsets `14.5 + z` dB / `5.5` dB), asymmetric Bark-axis
  spreading (−27 dB/Bark down, −10 dB/Bark up, max-combined), and the
  standard analytic absolute-threshold-of-hearing floor calibrated by
  `PsyConfig::full_scale_db`. `compute_masking` returns a per-bin
  linear-amplitude masking threshold plus band tonality; two glue
  routines feed it into the existing encode stack:
  `plan_psy_floor_envelope` (a floor-1 envelope target that tracks
  the peak-held signal where audible and rides the threshold where
  masked, clamped to the §10.1 dB-ladder range) and
  `residue_partition_weights` (per-partition `(floor/threshold)²`
  noise-to-mask weights on the raw audibility scale, each bin's ratio
  capped at a 40 dB excess so one near-silent-threshold bin cannot
  monopolise the bit budget). Tests pin the ATH dip at 3–4 kHz, the
  silence→threshold-in-quiet reduction, masker-local threshold lift,
  the upward-spread asymmetry, threshold monotonicity in masker
  level, tone/noise tonality separation, the
  tonal-masks-less-than-noise offset ordering, the
  `threshold_offset_db` lever, envelope max/peak-hold behaviour, and
  the weight normalisation + guard surface. Re-exported at the crate
  root.
- **Perceptually weighted residue rate-distortion chooser
  (`residue_encode::plan_vector_classifications_rd_weighted`,
  `plan_vector_residue_rd_weighted`)** — the §8.6.2 encode-direction
  Lagrangian gains a per-partition distortion weight:
  `weights[p] · error_sq + lambda · bit_cost`. Equal residue-domain
  error is not equally audible (the decoder multiplies the residue by
  the rendered floor, §4.3.6), so charging each partition's squared
  error with `psy::residue_partition_weights`'s capped
  `(floor/threshold)²` factor turns the trade into an approximate
  noise-to-mask-ratio-vs-bits descent: audible partitions attract
  denser cascades, masked partitions surrender bits first. All-`1.0`
  weights reproduce the unweighted chooser bit-for-bit (shared core,
  identical arithmetic); a `0.0` weight makes a partition rate-only.
  New guard variants `WeightLengthMismatch` / `BadWeight`. Tests pin
  the all-ones equivalence across lambdas (choices and scored
  totals), the weight-buys-density flip at fixed lambda, the
  zero-weight cheapest-cascade rule, per-partition weight
  independence, and the guard surface. Re-exported at the crate root.
- **Distortion-aware value-ladder step in the closed RD training loop
  (`book_design::train_residue_books_rd_ladder`,
  `residue_encode::replay_partition_cascade`)** — closes the README's
  "ladder is not yet inside the closed loop" followup. The
  length-only trainer holds every VQ lookup fixed (old packets decode
  bit-identically), which caps the descent when the seed ladder does
  not span the corpus. The extended trainer adds a per-iteration
  value step: `replay_partition_cascade` re-walks each planned §8.6.2
  cascade deterministically to recover the exact target sub-vector
  every entry quantised, each exercised tessellation book's entries
  move to the centroid of their targets (the classic VQ codebook
  update), and the result is re-expressed on a fresh §9.2.2-packable
  `minimum/delta` grid. Because cascade stages interact (improving an
  early stage shrinks a later stage's targets), the joint update and
  each single-book update are evaluated by fresh plan passes and only
  the best strict improvement is adopted — the recorded Lagrangian is
  monotone non-increasing by construction, and multi-stage ladders
  converge stage-by-stage against re-derived targets. `sequence_p`
  and lattice books are never touched. Tests pin the replay's
  target/entry reproduction (formats 0 and 1, shape/unpack guards), a
  ±5 corpus against a `[-2, 1.5]` seed ladder (length-only training
  cannot reach it; the ladder trainer at least halves the final
  Lagrangian, ≥1 accepted update, monotone descent), §3.2.1 carriage
  legality + final-plan/final-book consistency, the ideal-ladder
  no-op equivalence, and the guard surface. Re-exported at the crate
  root.
- **Quality targeting (`quality` module: `EncoderTuning`,
  `solve_lambda_for_bits`)** — one scalar for the encode stack's
  quality levers, plus the bit-budget inverse.
  `EncoderTuning::from_quality(q ∈ [0, 1])` expands to a coherent
  lever set: the residue RD `lambda` falls log-linearly (`1.0` →
  `10⁻⁴`, pricing bits in noise-to-mask units under the weighted
  chooser), the psy margin `threshold_offset_db` rises linearly
  (−12 → +12 dB), and the floor-1 post budget grows (8 → 32) —
  monotone by construction. `solve_lambda_for_bits` bisects any
  caller-supplied monotone non-increasing `rate(lambda)` measurement
  (a plan's value bits, a packet's size, a stream's) to the cheapest
  `lambda` fitting a bit budget, returning the measured point (no
  interpolation), the loose-budget fidelity end, or the flagged
  cheapest end when the budget is unreachable. Tests pin endpoint
  values, monotonicity across the knob, bracket/iteration guards,
  loose/unreachable budget behaviour, near-target landing (within 2%
  under budget on a synthetic curve), budget-monotone solutions, and
  rate-error propagation. Re-exported at the crate root.
- **Measured quality → rate/fidelity curve
  (`tests/quality_rate_curve.rs`)** — the capstone over the round's
  psychoacoustic + quality stack. An 8-frame corpus (breathing tones,
  re-rolled borderline pedestal + masked hash) is encoded at
  `q ∈ {0, 0.25, 0.5, 0.75, 1}` through the full chain — per-tuning
  masking → psy floor envelope → `design_floor1_header` at the
  tuning's post budget → per-frame post fit → NMR-weighted residue RD
  at the tuning's lambda → `write_audio_packet` →
  `decode_audio_packet_pre_imdct` — and the measured curve is pinned
  monotone: rate 488 → 776 → 2264 → 2480 → 2600 B, spectral SNR
  7.2 → 32.8 → 36.4 → 36.9 → 37.3 dB, mean nominal-threshold NMR
  5.8 → 0.58 → 0.002 → 0.0005 → 0.0005 (q = 1 transparent under the
  model). A second test drives `solve_lambda_for_bits` over the
  *real* stream rate–lambda curve: a halfway byte budget is landed
  exactly (12 672 / 12 672 bits, `within_budget`), and a tighter
  budget never receives a smaller lambda.
- **Psy × training composition capstone
  (`tests/quality_rate_curve.rs::ladder_trained_books_cut_the_rate_of_the_psy_stream`)**
  — the psy floor + quality tuning fix the residue targets of the
  8-frame stream; `train_residue_books_rd_ladder` then retrains the
  seed value books on those targets. Measured: the trained stream
  serialises into 1 473 B against the seed books' 2 288 B (−36%) at
  unchanged NMR (0.00118), the descent is monotone, and the trained
  setup header round-trips whole through `write_setup_header` →
  `parse_setup_header` (§4.2.4 carriage).
- **VQ value-ladder design (`book_design::design_value_ladder`,
  `ValueLadderDesign`)** — the *value*-side half of codebook training
  (the codeword-length half is `design_codeword_lengths`). A
  lookup-type-1/2 codebook reconstructs every scalar as
  `multiplicand · delta + minimum` (§3.2.1), so designing the ladder
  means choosing the reconstruction points that minimise the training
  set's quantisation error and expressing them in that grid form. The
  optimiser is the classic 1-D Lloyd iteration (nearest-level
  assignment ↔ centroid update, deterministic quantile
  initialisation, monotone MSE descent); the converged centroids snap
  to a `value_bits`-wide multiplicand grid whose `minimum` / `delta`
  are rounded to **§9.2.2-packable** floats (a 21-bit-significand
  `pack_nearest` helper), so `write_codebook` carries the designed
  ladder exactly. `ValueLadderDesign` reports the snapped parameters,
  per-level reconstruction values, the final-grid MSE, and wraps
  itself as a `VqLookup::Tessellation` (`into_tessellation_lookup`).
  Tests pin cluster recovery (two-cluster data beats the uniform
  ladder decisively), full §3.2.1 carriage + `unpack_vector` /
  `quantize_vector` agreement with the designed levels, the
  degenerate constant-corpus ladder (`delta = 0`, zero error),
  packability across 40 pseudo-random corpora, and the new guard
  surface (`EmptyTraining` / `NonFiniteTraining` / `ZeroLevels` /
  `InvalidValueBits` / `LevelsExceedValueBits`). Re-exported at the
  crate root.
- **Whole-stream trained-books round-trip
  (`tests/trained_stream_roundtrip.rs`)** — the capstone over the
  codebook-content design stack. A 20-frame mono PCM corpus (drifting
  harmonic mix) is encoded through the full §4.3 audio-packet writer
  against a real `VorbisSetupHeader` (floor-1 + three-class residue +
  four codebooks); **both** subsystems' emissions are tallied into one
  `BookTallies` (`tally_floor1_packet` + `tally_residue_plans` per
  packet), the whole codebook table is retrained in one pass, and the
  same packets re-serialise under a setup header carrying the trained
  books. Pins: every retrained §4.3 packet decodes to the
  **bit-identical** windowed PCM frame (the §4.3.2–§4.3.7 numeric
  pipeline sees identical inputs — floor-1 post coding is lossless and
  every VQ lookup is preserved); the audio corpus serialises into
  **strictly fewer bytes**; every book in the stream is exercised; and
  the trained setup header itself round-trips **whole** through
  `write_setup_header` → `parse_setup_header` field-for-field (§4.2.4
  carriage, with the mandatory time-transform placeholder), so the
  trained stream is carriage-complete — headers and audio.
- **Closed-loop rate-aware residue training
  (`book_design::train_residue_books_rd`,
  `ResidueRdTrainingOutcome`)** — couples the trainer to the
  rate-distortion planner. A single tally→retrain pass re-prices
  codewords for plans chosen under the *old* prices;
  `plan_vector_residue_rd` charges exact codeword lengths in its
  Lagrangian, so re-planning under retrained books shifts choices
  toward the now-cheaper symbols, justifying another retrain. The loop
  is classic alternating minimisation over the shared objective
  `Σ error_sq + λ · value_bits`: the plan step is per-partition
  optimal given the books, and the **sparse** retrain step is exactly
  optimal for the observed frequencies (dense smoothing would break
  the descent guarantee — the policy note documents this, and every
  entry the current plans use keeps its codeword so the previous
  plans stay feasible). The outcome reports the per-iteration
  Lagrangian (provably monotone non-increasing), the per-iteration
  total codeword bits (value + classwords, priced through
  `stream_cost_bits`), the final plans, the trained books, and a
  fixed-point convergence flag (identical plans ⇒ identical tallies ⇒
  identical retrain ⇒ stop). Integration tests drive the loop over
  the residue corpus: monotone descent + convergence + strictly fewer
  total bits than the flat-book first pass + final plans round-trip
  through the real §8.6.2 writer/decoder under the trained books +
  carriage legality; a `λ = 0` case pins the distortion-only
  reduction (price-independent choices stabilise after one retrain,
  which still pays off on the wire). Unit tests pin the
  `ZeroIterations` / non-finite-λ guards and small-corpus
  convergence. `BookDesignError` drops its `Eq` derive (it now
  carries `ResidueEncodeError`, which is `PartialEq` only).
- **Floor-0 value-codebook-content trainer
  (`book_design::tally_floor0_packet` +
  `tests/floor0_trained_books.rs`)** — closes the "floor-0
  value-codebook *contents*" followup the README named, completing the
  trainer triad (floor-1 / residue / floor-0). A floor-0 packet's only
  codewords are the §6.2.2 step-7 VQ entry run; the tally records each
  entry against the value book the packet's `[booknumber]` selects
  through `floor0_book_list` (§6.2.2 step 5), skipping the raw
  fixed-width `[amplitude]` / `[booknumber]` fields and the `Unused`
  short-circuit, committing atomically via `record_all`. The
  integration suite drives the training loop over a 32-envelope
  formant corpus with drifting resonance centres (LSP coefficients of
  similar spectra cluster on the value ladder — the test asserts a
  strict minority of the 256 entries is used, so the statistical
  structure is real): plan (`plan_floor0_packet`) → tally → `retrain`
  → re-write, pinning bit-identical §6.2.3 curves under the trained
  book (VQ ladder preserved), a strictly smaller corpus on the wire,
  and §3.2.1 writer/parser carriage legality (the test book's ladder
  is dyadic, `−0.5 + e/64`, so it is exactly §9.2.2-packable). A
  second test pins the sparse policy: pruned ladder entries, a
  successful re-plan (the §3.2.1 quantiser only ever selects used
  entries), the same reconstructed curve, and no extra cost on the
  training corpus. Unit tests pin the booknumber routing and the
  atomic error surface (`Floor0BooknumberOutOfRange`, out-of-range
  entries recording nothing).
- **Residue codebook-content trainer (`book_design::tally_residue_plans`
  + `BookTallies::record_all` + `tests/residue_trained_books.rs`)** —
  the residue analogue of the floor-1 trainer. The tally mirrors
  `write_residue_body`'s §8.6.2 emission exactly: each stride of
  `classwords_per_codeword` (= classbook `dimensions`) classifications
  packs into one classbook entry via the same
  `pack_residue_classification_groups` primitive the writer uses
  (steps 6..12, final partial stride right-padded), and each
  `(partition, pass)` whose cascade holds a book records that stage's
  entry list (step 19; `None` stages and 'do not decode' vectors emit
  nothing). Both tally walks now commit **atomically** through the new
  `BookTallies::record_all` (a rejected packet/plan never leaves a
  partial tally). The integration suite drives the full training loop
  over a 40-vector corpus sectioned to exercise all three classes of an
  unused/coarse/coarse+fine header (silence → class 0; values exactly
  on the coarse ladder → the fewer-stages tie-break picks class 1;
  off-grid oscillation → class 2), pinning that trained books decode
  every body to the **bit-identical** §8.6.2 vector (retraining
  preserves the VQ lookups) while the corpus serialises into
  **strictly fewer bytes**, that trained books stay carriage-legal
  through the §3.2.1 writer/parser, and the same contract with a
  'do not decode' channel in the bundle. Unit tests pin the
  classword/value-entry routing and the new error surface
  (`ResiduePlanShapeMismatch` / `ResidueClassificationOutOfRange` /
  `ResiduePlanCascadeMismatch` / `ResidueClassPack`).
- **Floor-1 codebook-content trainer (`book_design::tally_floor1_packet`
  + `tests/floor1_trained_books.rs`)** — closes the "floor-1 master/sub
  codebook *contents*" followup the README named. The tally walks a
  planned `Floor1Packet` in `write_floor1_packet`'s exact §7.2.3
  emission order — the master selector `cval` into the class masterbook
  (step 12, only when `subclasses > 0`), each packet-domain Y into the
  sub-book its `cval` slice selects (steps 14/15), skipping step-18
  `None` slots and the raw (non-codeword) endpoint/`[nonzero]` fields —
  accumulating per-book symbol frequencies in `BookTallies`. The
  integration suite drives the full training loop over a 48-envelope
  corpus against a header exercising both class shapes (a
  `subclasses = 0` class and a `subclasses = 1` class whose master book
  routes between a coarse `Y < 96` sub-book and a full-range one, with
  the corpus straddling the boundary so the master path is
  load-bearing): plan → tally → `retrain` → re-write, pinning that the
  trained books decode every packet to the **bit-identical** §7.2.4
  curve (floor-1 post coding is lossless; retraining only re-prices the
  codewords) while the corpus serialises into **strictly fewer bytes**,
  and that the trained books stay carriage-legal through the §3.2.1
  codebook writer/parser. A second test pins the sparse policy:
  never-emitted entries are pruned, re-*planning* against the sparse
  books succeeds (the planner only selects encodable cvals), the fitted
  `floor1_y` is book-independent, and the sparse corpus costs no more
  than the dense one. Unit tests pin the tally walk (master + sub-book
  routing, unused-packet no-op) and its shape-gate error surface
  (`Floor1YLengthMismatch` / `Floor1CvalLengthMismatch` /
  `Floor1ClassOutOfRange`).
- **Codebook assembly + usage-driven retraining (`book_design` module:
  `design_entropy_codebook`, `redesign_codebook`, `BookTallies`)** —
  the layer that turns designed codeword lengths into write-ready
  codebooks and closes the training loop. `design_entropy_codebook`
  builds a complete lookup-type-0 `VorbisCodebook` from a
  symbol-frequency table (accepted by `write_codebook`, reproduced by
  `parse_codebook`, tree-buildable). `redesign_codebook` retrains an
  *existing* book around a measured distribution while preserving its
  shape and VQ lookup — every entry still unpacks to the identical
  §3.2.1 vector, so packets referencing the same entry indices decode
  **bit-identically** while serialising into fewer bits.
  `BookTallies` is the per-stream accumulator (one frequency row per
  codebook): the encoder records every codeword it plans, then
  `retrain` redesigns exactly the exercised books and clones the rest
  unchanged. Tests pin the write→parse round-trip, a measured on-wire
  win over the flat book (emitted bytes counted through
  `HuffmanTree::encode_entry`, agreeing with `stream_cost_bits`
  pricing), VQ-semantics preservation entry-for-entry, the
  only-exercised-books retraining contract, and the new shape /
  out-of-range error surface. Re-exported at the crate root.
- **Codebook content design — optimal codeword-length assignment
  (`book_design` module: `design_codeword_lengths`,
  `design_codeword_lengths_dense`, `stream_cost_bits`,
  `MAX_CODEWORD_LEN`, `BookDesignError`)** — the foundation of the
  codebook-*content* design followup both floor paths named. A Vorbis
  codebook's entropy content is fully determined by its per-entry
  codeword-**length** list (§3.2.1's canonical "lowest valued unused
  codeword" rule implies the codewords), so designing a book reduces to
  choosing the length list that minimises `Σ freq·length` subject to
  §3.2.1 legality: lengths in `1..=32` (the 5-bit `length − 1` field),
  a **fully populated** decision tree (Kraft sum exactly 1 — §3.2.1
  rejects both under- and over-specified trees), sparse
  `UNUSED_ENTRY` slots for never-emitted entries, and the errata-20150226
  single-used-entry book (sole entry at length 1). The optimiser is the
  classic package-merge (coin-collector) construction for
  length-limited prefix codes, realised with the sorted-prefix property
  (per level only the count of taken symbol coins is tracked). Two
  sparse policies: `design_codeword_lengths` prunes zero-frequency
  entries (cheapest, book cannot encode them), `_dense` smooths them to
  frequency 1 (book keeps its whole entry range encodable).
  `stream_cost_bits` prices a symbol-frequency table against a length
  list exactly. 16 new unit tests: balanced/dyadic recovery, an
  exhaustive small-case **brute-force optimality oracle** (every
  Kraft-complete capped length multiset enumerated), a binding
  length-cap case (Fibonacci frequencies) staying legal, Kraft equality
  + `HuffmanTree` acceptance across 200 pseudo-random tables,
  never-worse-than-flat, deterministic tie-breaks, sparse/dense/single
  policies, the full error surface, and a 4096-entry Zipf book. All
  re-exported at the crate root.
- **Floor-1 rate-distortion post-budget selection (`floor1_layout` module:
  `select_floor1_post_budget`, `floor1_x_list_distortion`)** — makes the
  floor-1 post count *bit-budget aware*, the geometry analogue of the
  residue RD stack. `floor1_x_list_distortion` exposes the ladder-domain
  sum-squared error of a placement (the objective `plan_floor1_x_list`
  greedily minimises). `select_floor1_post_budget` sweeps post counts
  `1..=max_posts`, plans each placement, and returns the one minimising
  `distortion + lambda · posts` (the post count a direct proxy for the
  x-list + Y bit cost). `lambda == 0` spends the whole budget (densest,
  lowest distortion); a punishing `lambda` strips to the endpoint-only floor.
  Tests pin that distortion drops monotonically as posts are added, that the
  lambda sweep moves the chosen budget from full to empty, and the
  bad-envelope guard. Re-exported at the crate root.
- **Floor-0 order design (`floor0_layout` module: `select_floor0_order`,
  `select_floor0_order_rd`, `score_floor0_orders`, `suggest_floor0_params`,
  `Floor0OrderFit`)** — the floor-0 analogue of the floor-1 post-budget
  choice: choosing `floor0_order`, the number of §6.2.3 LSP poles. The
  selector sweeps a caller-bounded order range, fits each candidate's LSP
  shape + amplitude (`plan_floor0_lsp` → `fit_floor0_amplitude`), renders
  the §6.2.3 curve the decoder would draw (`Floor0Decoder::render_curve`),
  and scores its **log-domain** fidelity (the curve is exponential, so the
  natural error metric is in `ln` space). `select_floor0_order` returns the
  smallest order meeting an SNR target (cheapest "good enough" model);
  `select_floor0_order_rd` minimises `distortion + lambda · order` (the
  order is a monotone proxy for the per-pole coefficient bit cost), ties to
  the lower order. `suggest_floor0_params` offers spec-grounded defaults for
  the surrounding header fields (`bark_map_size` = bin count, full 6-bit
  amplitude width, a non-zero gain offset). Because no reference encoder
  emits floor 0, fidelity is measured against the crate's own decoder
  render. Tests pin that a higher order fits a multi-resonance envelope no
  worse, that the smallest-order/RD selectors pick correctly across the
  lambda sweep, and the error-surface guards. Re-exported at the crate root.
- **Designed-floor-1-header PCM round-trip
  (`tests/floor1_designed_header_roundtrip.rs`)** — the integration suite
  that drives the §4.3 audio packet with a floor-1 header **designed from
  the spectrum** (not hand-built). Synthetic PCM → forward MDCT → smoothed
  `|X|` envelope → `design_floor1_header` (adaptive posts + DP partition
  tiling over a `{1, 2, 4}`-dimension class catalogue) → `plan_floor1_envelope`
  → `plan_floor1_y` → residue against the rendered floor → `write_audio_packet`
  → `decode_audio_packet_windowed`, clearing a pinned **35 dB** PCM-domain
  SNR against `window ⊙ IMDCT(X)` across a 128/256/1024 block-size sweep. It
  also pins the designed header's self-consistency (partitions tile the
  x-list; a tilted spectrum designs interior posts) and that a larger post
  budget does not regress fidelity. The floor-1 encode path is now closed
  from raw spectrum through a self-designed setup header.
- **Floor-1 one-call setup-header designer (`floor1_layout` module:
  `design_floor1_header`)** — the composition that ties the layout module
  to the existing per-packet floor-1 encode chain
  (`floor1_encode::plan_floor1_packet`). From a representative envelope, a
  post budget, a fit tolerance, the multiplier, and a caller-supplied
  `Floor1Class` catalogue, it places the explicit x-coordinates
  (`plan_floor1_x_list`), tiles them into partitions over the classes'
  dimensions (`plan_floor1_partition_layout`), picks the smallest covering
  `rangebits` (`min_rangebits`), and assembles a write-ready
  `Floor1Header`. A flat-enough envelope yields a legal endpoint-only
  header (`partitions = 0`). Tests pin that a designed header is
  structurally valid (builds a `Floor1Decoder`), tiles its x-list exactly,
  reconstructs a peaky envelope no worse than uniform spacing at equal
  budget, degenerates correctly on a flat envelope, and rejects an empty
  catalogue. The geometry + partition + carriage of the floor-1
  setup-header is now planned from spectrum; codebook-content
  (bit-allocation) design remains the open followup. Re-exported at the
  crate root.
- **Floor-1 partition layout design (`floor1_layout` module:
  `plan_floor1_partition_layout`, `Floor1PartitionLayout`)** — the
  partition-grouping half of the floor-1 setup-header design. Each §7.2.2
  partition draws `class.dimensions` Y-values, and the partitions'
  dimensions must sum to exactly the explicit-post count. Given the
  catalogue of available class dimensions (each `1..=8`; the codebook a
  class carries fixes its dimension), the planner tiles the post count into
  the fewest partitions via exact dynamic-programming tiling (greedy
  descending alone can dead-end, e.g. dims {2,3} posts 4 — the DP finds
  2+2). It returns the `floor1_partition_class_list` indexing into the
  caller's class order plus the partition count, honouring the §7.2.2 5-bit
  partition ceiling (31) and 4-bit class-index ceiling (15). Tests pin
  exact tiling across 1..=40 posts, minimum-partition-count selection, the
  non-greedy DP case, the §7.2.2 ceilings, and the untileable/illegal-dim
  guards. Re-exported at the crate root.
- **Floor-1 x-list (post-placement) design (`floor1_layout` module:
  `plan_floor1_x_list`, `min_rangebits`)** — the first piece of the
  floor-1 *setup-header* design the README named as the open followup.
  The §7.2.4 step-2 render draws straight integer line segments between
  posts in the dB-ladder domain, so a good x-list is one whose
  piecewise-linear interpolation through the chosen posts tracks the
  desired envelope mapped into that ladder domain. `plan_floor1_x_list`
  derives the explicit x-coordinates by **adaptive refinement**: starting
  from the two implicit endpoints (`0` and the floor length) it repeatedly
  inserts the interior bin whose ladder value is furthest from the current
  reconstruction, until a post budget is met or the worst-case ladder error
  falls below a caller tolerance. Error is measured in §10.1 dB-ladder
  indices (via `invert_inverse_db`), the domain the line is actually drawn
  in. `min_rangebits` returns the smallest 4-bit `rangebits` whose implicit
  upper endpoint `2^rangebits` covers a given floor length. Tests pin the
  placement is sorted/unique/interior, that a flat envelope needs no
  interior posts, that adaptive placement beats uniform spacing at equal
  budget on a peaky envelope (full envelope → posts → decode → curve SSE),
  and the error-surface guards (empty/non-finite envelope, zero/over-budget
  post counts). Re-exported at the crate root.
- **Long/short block-size decision (`blocksize` module: `detect_transient`,
  `choose_blocksize`)** — the encode-side §1.3.2 / §4.3.1 block-size
  selection that drives a mode's `blockflag`. The spec fixes the bitstream
  mechanics of block switching but leaves the *analysis* to the encoder;
  this is a clean-room energy-envelope transient detector. `detect_transient`
  splits a time-domain PCM block into equal sub-frames (the final one
  absorbs a non-dividing remainder, conserving energy), measures each
  sub-frame's `Σ x²`, and reports the envelope with its peak-to-mean
  concentration ratio (`≈ 1.0` flat, large for a single attack).
  `choose_blocksize` turns that into the `blockflag`: a ratio above the
  caller's threshold means a transient → **short** block (`blockflag false`,
  confining quantisation noise around the attack to avoid pre-echo);
  otherwise the **long** block (`blockflag true`) for finer frequency
  resolution. The threshold is the caller's quality/bit-rate lever. Error
  surface guards empty blocks, zero/over-count sub-frames, and non-finite
  samples. Re-exported at the crate root.
- **Stereo coupling decision heuristic (`coupling_energy`, `should_couple`,
  `CouplingEnergy`)** — an encoder-side §4.3.5 lever that decides whether a
  Cartesian `(left, right)` channel pair is worth square-polar coupling.
  `coupling_energy` measures the magnitude/angle energy split the forward
  coupling would produce **without** mutating either channel (the
  non-committing analogue of `forward_couple`), and reports it alongside the
  uncoupled `L² + R²` baseline. `CouplingEnergy::angle_ratio` is the
  `angle_energy / magnitude_energy` figure a per-region gate keys off: a
  correlated pair (`L ≈ R`) couples to a near-zero angle vector (low ratio,
  coupling pays off — the angle residue quantises toward zero), while an
  anti-correlated pair (`L ≈ −R`) couples to a large angle vector (ratio 4,
  coupling buys nothing). `should_couple` gates on a caller-chosen
  `max_angle_ratio` threshold, refusing a non-finite or negative gate. All
  re-exported at the crate root. Tests prove the measurement matches a
  committed `forward_couple`, leaves inputs untouched, and that the ratio
  separates correlated from anti-correlated pairs (exactly 0 vs 4).
- **Coupling-step pruning driver (`prune_coupling_steps`)** — the
  channel-pair-level §4.3.5 decision built on `should_couple`: given the
  per-channel Cartesian spectra and a *candidate* coupling-step list (e.g.
  couple every adjacent pair), it returns the subset of steps worth
  applying. The decision is sequential and order-faithful to
  `forward_couple_all`: steps are visited in ascending order on a working
  *copy* of the spectra, a kept step forward-couples its pair in the copy so
  later steps that reference the same channel see its square-polar result,
  and a dropped step leaves its channels Cartesian. The input spectra are
  never mutated. The kept-step list threads back through
  `forward_couple_all` → `inverse_couple_all` to reconstruct the original
  spectra (proven by a round-trip test), so pruning yields a self-consistent
  encode/decode coupling set. Same channel-range / same-channel validation
  as `forward_couple_all`, applied to every candidate step. Re-exported at
  the crate root.
- **Residue cascade rate term (`ScoredPartitionCascade::bit_cost`)** — the
  scored cascade planner (`plan_partition_cascade_scored`) now reports the
  exact value-codeword bit cost it emits for a partition: the sum of
  `book.codeword_lengths[entry]` over every entry the cascade chose, across
  all populated stages (Vorbis I §8.6.2). The quantiser only ever selects a
  'used' entry, so every charged length is in `1..=32` and the sum is the
  precise number of bits the write path packs for the partition's value
  codewords — the rate term a rate-distortion classification chooser trades
  against `error_sq`. The §8.6.2 classword is amortised at the vector level
  (one classword can cover several partitions in formats 1 / 2) and is
  therefore scored separately, not charged here.
- **Rate-distortion residue classification chooser
  (`plan_vector_classifications_rd`)** — the rate-aware sibling of
  `plan_vector_classifications`. The existing chooser minimises
  reconstruction distortion alone (with a stage-count tie-break), which
  always prefers the densest cascade that reconstructs best and never
  trades a little distortion for a cheaper encoding. The new chooser
  minimises the Lagrangian cost `error_sq + lambda · bit_cost` per
  partition: `lambda == 0` reduces *exactly* to the distortion chooser
  (proven by a bit-for-bit equality test), and a larger `lambda` pulls the
  choice toward cheaper (fewer-bit) classifications — the encoder's
  response to a tighter bit budget (Vorbis I §8.6.2). `PartitionClassChoice`
  gains a `bit_cost` field carrying the chosen cascade's value-codeword
  cost, populated by both choosers. A NaN or negative `lambda` is rejected
  with the new `ResidueEncodeError::NonFiniteLambda`. Re-exported at the
  crate root.
- **Whole-vector rate-distortion residue planning + configuration selection
  (`plan_vector_residue_rd`, `select_residue_config`)** — the top of the
  rate-aware residue stack. `plan_vector_residue_rd` chooses every
  partition's classification by the Lagrangian criterion and returns the
  assembled `classifications` + `partition_entries` together with the
  aggregate figures (`ScoredVectorResidue { total_error_sq,
  total_value_bits }`). `select_residue_config` sits one level up: given the
  same target residual and several candidate residue *configurations*
  (`ResidueConfigCandidate` — differing `residue_type`, `partition_size`,
  value-book table, and classbook width), it scores each candidate's
  rate-distortion plan and keeps the one minimising `total_error_sq +
  lambda · (value_bits + classword_bits)`. The §8.6.2 classword cost — one
  classword per `partitions_per_classword` partitions — is folded in here
  because it is a property of the residue header (its classbook), constant
  across a candidate's partitions but different between candidates.
  Tie-break on cheaper total bits then lower index. `SelectedResidueConfig`
  reports the winning index, plan, classword bits, and Lagrangian cost. New
  `ResidueEncodeError::ZeroPartitionsPerClassword` guards a malformed
  classbook dimension. All re-exported at the crate root.
- **Rate-distortion residue PCM round-trip suite**
  (`tests/rate_distortion_residue_roundtrip.rs`) — the rate-aware
  counterpart to `tests/pcm_adaptive_residue_roundtrip.rs`, driving the same
  synthetic tilted-spectrum PCM through `plan_vector_residue_rd` and
  `select_residue_config` and decoding back through the real §4.3 decoder.
  It proves: (1) at `lambda == 0` the rate-distortion plan round-trips to
  the same pinned ≥20 dB PCM-domain SNR as the distortion plan and is
  content-adaptive; (2) a monotone-increasing `lambda` sweep never
  *increases* the plan's total value-codeword bit cost and the cheapest
  point spends strictly fewer bits than `lambda == 0` (the rate knob bites);
  (3) every rate point still round-trips to finite PCM (rate reduction
  trades fidelity, it does not corrupt the bitstream — a higher `lambda`
  yields lower SNR but a valid stream); and (4) `select_residue_config` over
  a fine vs coarse-only candidate pair picks the fine config at tiny
  `lambda` and the narrower-classbook coarse config at large `lambda`, with
  both winners round-tripping.
- **Residue format-0 strided-scatter PCM round-trip**
  (`tests/residue_format0_roundtrip.rs`) — the first *audio-packet-level*
  exercise of §8.6.3 (read `i`, element `j` → `i + j·step`); format 0 had
  only isolated residue-body coverage, and every packet round-trip used a
  contiguous format-1/2 residue. Mono PCM → windowed MDCT → flat floor →
  `plan_partition_cascade` with `residue_type = 0` (the encode-side strided
  **gather** that inverts the §8.6.3 scatter) over a 2-D value book (its
  `dimensions = 2` divides the partition) → `write_audio_packet`
  (`residue_type: 0`) → `decode_audio_packet_windowed` → PCM, clearing
  ≥25 dB PCM-domain SNR (≥20 dB across a 128/256/512 block-size sweep).
- **Residue format-2 multi-channel PCM round-trip**
  (`tests/residue_format2_roundtrip.rs`) — the first *encode→decode* exercise
  of §8.6.5 (the "interleave all channels into one virtual vector,
  format-1-decode, de-interleave" residue mode); previously only the
  decoder's interleave/de-interleave was unit-tested, and every packet
  round-trip used a per-channel format-1 residue. Two-channel PCM → two
  windowed MDCTs → flat floor → **interleave** the per-channel targets
  (`interleaved[i·ch + j] = channel[j][i]`, the inverse of the §8.6.5 step-3
  de-interleave) → `plan_partition_cascade` the single interleaved vector →
  `write_audio_packet` (`residue_type: 2`, one residue plan) →
  `decode_audio_packet_windowed`, with each decoded channel clearing ≥30 dB
  PCM-domain SNR (≥25 dB across a 128/256/512 block-size sweep).
- **Stereo channel-coupling PCM round-trip** (`tests/stereo_coupling_roundtrip.rs`)
  — the first *encode→decode* exercise of §4.3.5 channel coupling (every
  prior encode round-trip is mono with empty coupling; §4.3.5 was only
  decode-tested). Stereo PCM → two windowed forward MDCTs →
  `forward_couple_all` (square-polar magnitude/angle) → residue against a
  flat floor → `write_audio_packet` (one submap, the coupled pair) →
  `decode_audio_packet_windowed` (which runs the §4.3.5 inverse coupling) →
  two channels, each clearing ≥25 dB PCM-domain SNR against its own
  `window ⊙ IMDCT(X*)` reference, plus a 128/256/512 block-size sweep and a
  control proving the coupling transform actually ran. Proves the encoder's
  forward coupling and the decoder's inverse coupling compose to reproduce
  the original L/R signal.
- **Floor-0 envelope-fit chain** (`floor0_envelope` + `floor0_lsp` modules)
  — the §6.2.3 curve **inverse**, the floor-0 analogue of
  `plan_floor1_envelope`. `floor0_lsp` carries the generic DSP:
  `autocorrelation_from_angles` (midpoint-quadrature inverse-DFT over a
  non-uniform Bark grid), `levinson_durbin` (all-pole Yule-Walker solve),
  and `lpc_to_lsp` (parity-aware symmetric/antisymmetric P/Q deflation +
  dense-grid root bracketing for the LSP angles). The identity
  `1/sqrt(p+q) == 1/|A(e^jω)|` is pinned to 1e-6. `floor0_envelope` carries
  the Vorbis composition: `plan_floor0_lsp` (fold the target's *shifted-log*
  envelope onto the §6.2.3 Bark-bucket grid — the curve is exponential in
  the LSP shape, so `g` must track `ln(envelope)` — then run the DSP chain),
  `fit_floor0_amplitude` (closed-form integer `[amplitude]` via
  `Σ(g·t)/Σ(g²)`), and the one-call `plan_floor0_packet` (envelope →
  write-ready `Floor0Packet::Curve`, neither LSP coefficients nor amplitude
  nor entry run supplied by hand). New `Floor0EnvelopeError`,
  `Floor0LspError`, `Floor0PacketPlanError`. Closes the floor-0 per-packet
  encode chain end to end.
- **Floor-0 envelope → packet → decode round-trip**
  (`tests/floor0_envelope_roundtrip.rs`) — a desired envelope →
  `plan_floor0_packet` → `write_floor0_packet` → `Floor0Decoder` round-trips
  bit-for-bit against an independent §6.2.3 render of the rebuilt
  coefficients (even order 14 and odd order 13), clears a log-domain shape
  SNR bar, and rejects out-of-range book selectors.
- **Floor-0 PCM → encode → decode → PCM full-packet round-trip**
  (`tests/floor0_pcm_roundtrip.rs`) — the first audio-packet-level floor-0
  round-trip, the floor-0 analogue of `nonflat_floor_pcm_roundtrip`: PCM →
  window+MDCT → |X| envelope → `plan_floor0_packet` → residue against the
  rendered §6.2.3 curve → `write_audio_packet` (`AudioChannelFloor::Type0`)
  → `decode_audio_packet_windowed`, clearing ≥30 dB PCM-domain SNR at order
  14, a 128/256/512 block-size sweep, and the odd-parity order-13 LSP
  branch. No reference encoder emits floor 0, so self-consistency against
  the crate's own decoder is the ground truth.

- **Floor-1 partition-packing planner** (`floor1_encode::plan_floor1_partition_cvals`)
  — derives each partition's master-selector `cval`
  (`encoder::Floor1Packet::partition_cvals`) from the fitted packet-domain
  `[floor1_Y]` vector, walking §7.2.3 steps 5..19 in the write direction.
  `subclasses == 0` classes emit `cval = 0` (every dimension uses sub-book
  slot 0); `subclasses > 0` classes search the master book's used entries
  ascending for the smallest selector whose per-dimension slices
  (`(cval >> j·cbits) & csub`) all land on sub-books that can encode the
  targets — the inverse of the decoder's steps 14/15. New `Floor1CvalError`
  covers length / class-index / book-resolution / no-reachable-cval
  failures. Closes the last hand-supplied floor-1 packet knob.
- **One-call floor-1 packet planner** (`floor1_encode::plan_floor1_packet`)
  — composes `plan_floor1_envelope` → `plan_floor1_y` →
  `plan_floor1_partition_cvals` to turn a desired linear-domain floor
  envelope directly into a write-ready `Floor1Packet` (no hand-supplied
  `floor1_y` or `partition_cvals`). New `Floor1PacketPlanError` unions the
  three stages. Unit coverage: cbits-0 emit/reject, negative-book Y=0
  constraint, master/subclass selection (even & odd cval, sparse master),
  validation gates, a `plan_floor1_y` → cval → `write_floor1_packet` →
  decode roundtrip, and an envelope → packet → decode roundtrip pinning the
  decoded curve against `render_curve` over the planned `[floor1_Y]`.
- Adaptive-classification **PCM → encode → decode → PCM** round-trip
  (`tests/pcm_adaptive_residue_roundtrip.rs`) — the first full §4.3
  time-domain round-trip whose residue classifications are chosen **from
  the spectrum** rather than hand-supplied. The existing
  `tests/nonflat_floor_pcm_roundtrip.rs` codes the whole spectrum as one
  residue partition with `classifications: vec![0]`; this suite splits the
  residue window into many partitions and lets `plan_vector_residue` pick
  each partition's classification (unused / coarse single-stage /
  coarse+fine two-stage) before `write_audio_packet` serialises the §4.3
  packet and `decode_audio_packet_windowed` decodes it back to a windowed
  frame. The chain is PCM → §4.3.1 window → §4.3.7 forward MDCT → non-flat
  floor-1 fit → `X / rendered_floor` residue target → **adaptive
  per-partition plan** → packet → decode → IMDCT+window, asserting the
  decoded frame tracks `window ⊙ IMDCT(X)` to ≥ 20 dB PCM-domain SNR, that
  the adaptive plan matches an explicit-classification replan
  **bit-for-bit** (selection ↔ entry round-trip exact), that the chosen
  classifications are content-adaptive (not constant on a tilted
  spectrum), and that adaptive selection clears a fixed single-coarse-class
  baseline by **≥ 2 dB** (≈ 46.7 dB adaptive vs ≈ 29.2 dB fixed measured —
  an ≈ 17.5 dB PCM-domain gain). A robustness case sweeps partition counts
  4 / 8 / 16 / 32, each round-tripping above a 15 dB floor.
- From-spectrum residue classification-selection integration coverage
  (`tests/residue_cascade_roundtrip.rs`). Two new end-to-end tests drive
  a **multi-classification** format-1 residue body through the
  full from-spectrum encode path (`plan_vector_residue` →
  `ResidueVectorPlan` → `write_residue_body` → `ResidueDecoder`) — the
  first integration coverage of the classification chooser.
  `from_spectrum_classification_selection_round_trips_and_beats_fixed_class`
  builds a three-classification residue (unused / coarse single-stage /
  coarse+fine two-stage cascade) over a sharply block-varying residual
  (quiet partitions then loud partitions) and pins: (1) the planner
  reaches for the high-fidelity two-stage class on the loud partitions
  (content-adaptive, not constant); (2) the from-spectrum plan and an
  explicit-classification replan reconstruct **bit-identically** (the
  selection ↔ entry-list round-trip is exact); and (3) adaptive selection
  clears the fixed-coarse-class baseline by **≥ 3 dB** spectral SNR
  (≈ 43.1 dB adaptive vs ≈ 24.4 dB fixed measured — an ≈ 18.6 dB gain,
  the loud partitions gaining the fine refinement stage the fixed coarse
  class cannot reach). `from_spectrum_silent_partitions_choose_unused_class`
  pins that an all-zero vector is coded with the cheapest ('unused')
  classification on the distortion tie and decodes back to exact silence.
- §8.6.2 residue **classification-selection** layer (`residue_encode`
  module: `plan_vector_classifications`, `plan_vector_residue`,
  `PartitionClassChoice`, plus the scored cascade primitive
  `plan_partition_cascade_scored` / `ScoredPartitionCascade`). The
  residue encode stack already turned a vector's residual into the
  per-partition value-codeword entry lists *given the classifications*
  (`plan_vector_partition_entries`); choosing those classifications was
  the open optimisation the README named. This closes it from the
  distortion side. For each partition `plan_vector_classifications` tries
  **every** candidate classification in the `value_books` table, plans
  its cascade with the new `plan_partition_cascade_scored` (which keeps
  the leftover residual and reports its squared norm as `error_sq` plus a
  `populated_stages` bit-cost proxy), and keeps the classification whose
  cascade reconstructs the partition's target most closely — ties broken
  toward fewer populated stages (cheaper encoding) then the lower
  classification index (deterministic). `plan_vector_residue` is the
  top-of-stack splitter: raw spectral residual in, the index-aligned
  `classifications` + `partition_entries` arrays a `ResidueVectorPlan`
  holds out, with **no hand-supplied classifications**. A new
  `ResidueEncodeError::NoClassifications` rejects an empty `value_books`.
  The residue WRITE path no longer needs the per-partition
  classifications supplied by hand — the residue encoder now plans both
  the classification and the cascade from spectrum. `plan_partition_cascade`
  is unchanged on the wire (it delegates to the scored primitive and
  discards the score). 18 new in-module unit tests (crate lib suite
  **850 → 878**): the scored cascade's exact-hit zero error /
  quantisation residual / unused-cascade target-norm /
  entries-match-unscored properties; classification selection picking the
  lower-distortion class, both tie-breaks (fewer stages, lower index),
  per-partition independence, the empty-table + non-multiple-length +
  cascade-error-propagation rejections, a full `plan_vector_residue`
  round-trip through the independent decode-reconstruct oracle, and an
  exhaustive 121-point target sweep asserting the chosen class's
  distortion is `<=` every other class's. Spec source:
  `docs/audio/vorbis/Vorbis_I_spec.pdf` §8.6.2 (the cascade /
  classification decode loop), §8.6.1 (`residue_classifications`),
  §8.6.3 / §8.6.4 / §8.6.5 (the format addressing).
- `Floor0Decoder::render_curve` — the encoder-side §6.2.3 floor-0
  LSP-curve-synthesis primitive (the floor-0 twin of
  `Floor1Decoder::render_curve`). Given a per-packet `amplitude` and the
  post-§6.2.2 LSP `coefficients` vector, it renders the exact linear-domain
  floor the decoder reconstructs without reading a bitstream, so an encoder
  can plan residue against the rendered floor-0 envelope (`X[k] /
  render_curve(...)[k]`, the §4.3.6 per-bin multiplier) on a non-flat
  floor-0. Returns the nominal all-zero `'unused'` curve for `amplitude ==
  0` or a coefficients vector shorter than `floor0_order`. Unit-tested
  bit-identical to the decode-path curve, and cross-checked in
  `tests/floor0_audio_packet_decode.rs` against the §6.2.3 curve the
  **full §4.3 driver** reconstructs inside the §4.3.6 product — proving the
  encoder-side primitive matches the per-bin floor the decoder reapplies.
- Non-flat floor-1 PCM round-trip fidelity suite
  (`tests/nonflat_floor_pcm_roundtrip.rs`). The existing
  `tests/pcm_packet_roundtrip.rs` round-trip used a *flat* floor (constant
  `F = 1.0`) so residue carried the analysis spectrum directly; this suite
  closes the representative case of a **non-flat** floor-1 fitted to the
  spectral shape. It drives synthetic PCM → §4.3.7 forward MDCT → smoothed
  magnitude-envelope fit → seven-interior-post floor-1 →
  `Floor1Decoder::render_curve` → residue-against-the-rendered-floor cascade
  → `write_audio_packet` → `decode_audio_packet_windowed` → IMDCT+window,
  and pins the decoded frame to `window ⊙ IMDCT(X)` at ≥ 35 dB PCM-domain
  SNR (≈ 44 dB achieved) across short and long blocks. A control variant
  proves the fidelity hinge: dividing the spectrum by the *desired envelope*
  (sampled at posts) instead of the **rendered** floor collapses the
  reconstruction to < 1 dB SNR — because §7.2.4 step 2 draws integer line
  segments between posts, so the rendered floor bows away from the envelope
  between them and only the rendered-floor divide hands residue the per-bin
  floor the decoder reapplies. A third assertion confirms the fitted floor
  is genuinely non-flat (≥ 2× dynamic range across the band), so the test is
  no easier than the flat-floor case.
- `Floor1Decoder::render_curve` — the encoder-side §7.2.4 floor-1
  curve-synthesis primitive. Given a packet-domain `[floor1_Y]` post
  vector (the same vector `plan_floor1_y` produces and `write_floor1_packet`
  serialises) it renders the exact linear-domain floor the decoder will
  reconstruct, without reading a bitstream. This is the floor a faithful
  encoder must plan residue against on a **non-flat** floor: the decoder
  computes the final spectrum as `floor[k] · residue[k]` (§4.3.6), and
  §7.2.4 step 2 draws integer line segments between posts, so the rendered
  floor bows away from a curved target between posts. Dividing the target
  spectrum by `render_curve` output (rather than by the desired envelope
  sampled at posts) gives residue the exact per-bin floor the decoder
  multiplies back in. Unit-tested bit-identical to both the private
  curve-computation and the decode-path curve.
- §4.3.8 overlap-add output-geometry conformance over the full staged
  fixture decode (`tests/overlap_add_geometry.rs`). The PCM fixture test
  validates the decoded sample *values*; this suite pins the windowing /
  overlap-add-into-PCM *geometry* as it runs inside `StreamingDecoder`.
  Driving every fixture's whole audio stream through the public
  `push_packet` path, it asserts for every emitted frame: the priming step
  lands only on the first packet; each subsequent frame's per-channel PCM
  length equals the §4.3.8 `prev_n/4 + cur_n/4` lap (with `prev_n` from the
  previous packet's reported `n`, so the contract is checked across **all**
  packets, including the ones the trace does not log); every channel of a
  frame has identical length; and the streaming path reports the same
  `mode_number` / `blockflag` / `block_size` the trace logged (indexed by
  the trace's true bitstream `packet_idx`). 654 PCM frames across the 16
  staged fixtures (17 logical streams). It is a geometry + dispatch oracle
  (no sample-value re-check), isolating a §4.3.8 lap-length / priming /
  mode-dispatch regression from a numeric IMDCT/floor/residue one.
- Setup-/identification-header structural conformance against the staged
  fixture traces (`tests/setup_header_trace_conformance.rs`). Each
  `trace.txt` logs structured `VORBIS_HEADER_ID`, `VORBIS_HEADER_SETUP`,
  `CODEBOOK`, `FLOOR_CONFIG`, `RESIDUE_CONFIG`, `MAPPING_CONFIG` and
  `MODE_CONFIG` events — the "same setup-header counts" the fixture notes
  list as load-bearing reference data. The new suite parses each fixture's
  identification (§4.2.2) and setup (§4.2.4) headers and asserts the parsed
  structures reproduce, field-for-field, every one of those events:
  identification channels / sample-rate / bitrates / blocksizes; the five
  setup counts; per-codebook `dimensions` / `entries` / `lookup_type`
  (None=0 / Lattice=1 / Tessellation=2) / `value_bits` / `sequence_p`;
  per-floor type and (floor-1) `partitions` / `multiplier` / `rangebits` /
  `x_list_count` (honouring the two implicit endpoint posts); per-residue
  type / `begin` / `end` / `partition_size` / `classifications` /
  `classbook`; per-mapping type / `submaps` / `coupling_steps` and the
  `magnitude` / `angle` / per-submap `floor` / `residue` index arrays; and
  per-mode `blockflag` / `windowtype` / `transformtype` / `mapping`. 842
  structural events across the 16 staged fixtures (chained two-stream
  fixture validated per-logical-stream). Together with the audio-packet
  suite this pins the **entire** structural decode of every staged stream —
  every setup decision and every per-packet decision — against the trace,
  leaving only the lossy sample values to the ±1-s16 PCM test.
- Per-packet §4.3.1 header-decision conformance against the staged
  fixture traces (`tests/audio_packet_trace_conformance.rs`). Each
  `docs/audio/vorbis/fixtures/*/trace.txt` logs, per audio packet, the
  load-bearing structural decisions every conformant decoder must
  reproduce (`mode_number` / `blockflag` / `prev_window` / `next_window` /
  `block_size`). The PCM-level fixture test only validated the *final*
  bytes; this new suite drives every fixture's audio packets through the
  public §4.3.1 parser (`read_packet_header`) and asserts each parsed
  header matches the trace **line-for-line** — 505 audio-packet decisions
  across all 16 staged fixtures (including the chained two-stream fixture,
  walked per-logical-stream by a serial-aware de-framer). It is a pure
  header-decision oracle (no floor/residue/IMDCT), so a regression in the
  mode-bit width, the short-vs-long window-flag gating, or the blocksize
  resolution is caught in isolation. The trace's `packet_idx` is honoured
  as the true bitstream index (long streams log packets 0..=31 then the
  final end-trim packet at its real index), and each de-framed packet's
  body length is cross-checked against the trace's `packet_bytes`.
- Floor-0 (LSP) end-to-end coverage for the decode paths no staged fixture
  exercises (every `docs/audio/vorbis/fixtures/*` stream is floor type 1):
  - `tests/floor0_curve_roundtrip.rs` — the floor-0 plan→write→decode→curve
    loop (`plan_floor0_coefficients` → `write_floor0_packet` →
    `Floor0Decoder::decode`), asserting the §6.2.3 LSP→envelope curve
    bit-for-bit against an independent in-test recomputation of the
    Bark-map + LSP-product + `exp` synthesis. Covers both order parities,
    dim-1/dim-2 value books, a partial-final-vector surplus discard, the
    per-`n` Bark-map recompute across block sizes, and the §6.2.2
    zero-amplitude unused short-circuit.
  - `tests/floor0_audio_packet_decode.rs` — a real floor-0 audio packet
    decoded through the full `decode_audio_packet_pre_imdct` driver (the
    §4.3.2 dispatch landing on `FloorDecoder::Type0`, the §6.2.2-body →
    §4.3.4-residue bit hand-off, the §4.3.6 dot product over a floor-0
    curve), cross-checked against a standalone `Floor0Decoder` of the same
    body, plus the §4.3.2 step-6 unused path.
- Residue **format-0** (§8.6.3 strided scatter) encode→decode round-trip in
  `tests/residue_cascade_roundtrip.rs` — real encoders emit only formats
  1/2, so the `read i, element j → i + j·step` scatter had only a
  unit-level decode test. The new tests prove the encode-side gather is the
  exact inverse of the decode scatter (decoded residual == hand-scattered
  entry reconstructions, with a cross-check that the contiguous format-1
  interpretation of the same entry run would *not* match), and that the
  §8.6.2 additive cascade refines strictly under the format-0 layout.

### Fixed

- §3.2.1 canonical Huffman tree construction now assigns codewords by the
  spec's "lowest valued unused binary Huffman codeword" rule against the
  tree directly, instead of by a left-to-right open-slot deque. The deque
  assigned the leftmost *tree* position at each entry's depth, which only
  matches the canonical lowest-valued codeword when codeword lengths are
  non-decreasing. Real-world floor / residue classification books emit
  interleaved lengths (e.g. `[2,3,3,3,3,4,3,4]`, a fully-populated Kraft-1
  book), and the deque left dangling capacity on them — spuriously
  rejecting the codebook as `UnderspecifiedTree` and blocking every
  audio-packet decode against a real stream. Construction now descends the
  partially-built tree preferring the `0` child, materialising the lowest
  free codeword of each entry's length; over/under-specified detection is
  preserved (dead-end descent → `OverspecifiedTree`; any dangling child
  after all entries placed → `UnderspecifiedTree`). Regression tests pin
  the canonical codewords for two non-monotonic books and a full
  encode→decode round-trip. This was the last bug between the §4.3 decode
  chain and sample-exact PCM against the staged fixtures.

### Added

- Full §4.3 PCM → encode → decode → PCM time-domain round-trip integration
  test (`tests/pcm_packet_roundtrip.rs`) — the crate's first end-to-end
  audio-packet round-trip that returns to the time domain. A synthetic PCM
  analysis frame is windowed + forward-MDCT'd to an analysis spectrum `X`,
  fitted with a flat floor-1 (`F = 1.0` at post 255) plus a two-stage
  §8.6.2 residue cascade carrying `X/F`, serialised by `write_audio_packet`,
  then decoded by `decode_audio_packet_windowed` (the §4.3.2–§4.3.6 driver
  + the §4.3.7 IMDCT + the §4.3.6 window) back to a length-`N` windowed
  frame. The decoded frame is compared against `window ⊙ IMDCT(X)` — the
  exact frame the decoder's own IMDCT+window produce from the un-quantised
  analysis spectrum — clearing a pinned 30 dB PCM-domain SNR (≈44.7 dB
  measured) and shown geometry-robust across block sizes 64 / 256 / 1024.
  Jointly proves the floor render, §4.3.6 dot product, §4.3.7 IMDCT and
  §4.3.6 window are correct end to end.
- Residue VQ-encode cascade → §8.6.2 body write → residue decode spectral
  round-trip test (`tests/residue_cascade_roundtrip.rs`). Drives a real
  signed, non-flat spectral residual through `plan_partition_cascade` →
  `write_residue_body` and back through `ResidueDecoder`, pinning that the
  decoded residual is the nearest-entry ladder quantisation of the target,
  equals bin-for-bin the sum of the chosen entries' reconstructions (the
  entry-index ↔ codeword round-trip is exact), and that a two-stage cascade
  is strictly closer to the target than a one-stage cascade (and lifts the
  spectral SNR by ≥6 dB) — the §8.6.2 additive cascade-refinement property.
- Floor-1 envelope-fit glue (`floor1_envelope` module:
  `plan_floor1_envelope`, `invert_inverse_db`) — the §7.2.4 step-2 / §10.1
  dB-table **inverse**, encode direction. Given a desired linear-domain
  floor envelope (one magnitude per spectral bin, the domain the forward
  MDCT magnitude lives in) it fits the integer `[floor1_final_Y]` post
  vector `floor1_encode::plan_floor1_y` consumes: for each post it samples
  the envelope at the post's `x`, inverts the strictly-increasing 256-entry
  `floor1_inverse_dB_table` to the nearest ladder index, then divides by
  the multiplier (round-to-nearest) and clamps into `[0, range)`. This was
  the remaining floor-1 encode followup the README named — the
  `plan_floor1_envelope → plan_floor1_y → write_floor1_packet → decode`
  chain now reconstructs a floor matching the target envelope at every post
  to within the multiplier-grid + dB-ladder quantisation. The standalone
  `invert_inverse_db` helper is pinned to recover every exact table value
  to its own index and to break ties toward the lower (smaller-amplitude)
  index.
- Forward-MDCT → floor-1 envelope-fit → encode → decode round-trip
  integration test (`tests/floor1_envelope_roundtrip.rs`) — the first
  end-to-end encode→decode spectral round-trip in the crate. A synthetic PCM
  analysis frame is windowed and forward-MDCT'd (`apply_window_and_mdct_vec`)
  to a magnitude spectrum; a smoothed envelope is fit to floor-1 posts
  (`plan_floor1_envelope` → `plan_floor1_y`), serialised
  (`write_floor1_packet`), and decoded (`Floor1Decoder`). The test pins
  post-exact reconstruction at every rendered post and a 23 dB log-domain
  (dB-index) SNR across the whole reconstructed curve.
- §4.3 decode-driver robustness coverage (`tests/decode_robustness.rs`).
  The per-packet driver must reject malformed or truncated audio packets
  with a typed `StreamingError` and never panic (the §4.3.1 closing note
  makes an in-packet end-of-packet a recoverable "discard this packet"
  condition, not a crash). The test sweeps a real fixture packet truncated
  at every byte length, routes header-type packets (first bit set, §4.3.1
  step 1) into the audio driver and asserts a `NonAudioPacketType`
  rejection, and feeds empty / single-byte / 512 pseudo-random packet
  bodies — confirming panic-freedom across every path.
- Metadata-fixture parse + decode coverage
  (`tests/comment_header_decode.rs`). The two staged corpus members that
  carry a populated VORBIS_COMMENT header — `with-vorbis-comment-tags`
  (canonical TITLE / ARTIST / ALBUM / DATE / GENRE / TRACKNUMBER fields)
  and `with-attached-picture` (a >1 KB base64 `METADATA_BLOCK_PICTURE`
  cover-art blob, the FLAC-borrowed convention Ogg/Vorbis uses) — are now
  exercised end to end: `parse_comment_header` recovers the vendor string
  and every `KEY=value` entry (looked up case-insensitively per §5.2.2),
  the picture blob base64-decodes to a FLAC-PICTURE front-cover PNG
  header, and each fixture's audio decodes sample-exact against its
  `expected.wav`. A cross-check confirms the two fixtures — identical
  audio, different metadata — decode bit-for-bit alike, proving a large
  comment block does not shift the setup/audio packet boundaries or
  perturb the §4.3 decode.
- Chained-Ogg decode coverage (`tests/chained_stream_decode.rs`). The
  `chained-streams` fixture — two independent Ogg/Vorbis logical
  bitstreams concatenated byte-for-byte (RFC 3533 §5), each with its own
  `BOS`/`EOS` page flags and `bitstream_serial` cookie — is now decoded
  end to end through the public `StreamingDecoder` path. A serial-aware
  RFC-3533 page de-framer (private test scaffolding) splits the input into
  its two logical bitstreams without cross-boundary packet bleed; the
  first stream decodes sample-exact against `expected.wav` (the fixture's
  reference is the first stream only, per its `notes.md`), and the second
  stream re-parses its own three Vorbis headers and decodes independently,
  validating the per-stream reset + re-parse + decode cycle across a
  stream boundary. This was the one staged corpus member
  `tests/fixture_pcm_decode.rs` explicitly did not consume.
- §4.3 end-to-end sample-exact PCM decode + IMDCT normalization scalar
  pinned (`tests/fixture_pcm_decode.rs`). Twelve staged fixtures
  (`docs/audio/vorbis/fixtures/*` — mono / stereo / 5.1, q−1..q10, CBR,
  22.05–96 kHz, floor-1-only, full-residue noise, all three residue
  formats, and the short↔long transient block-size switch) decode through
  the public `StreamingDecoder::push_packet` bitstream → PCM path and
  match each fixture's `expected.wav` reference dump within the documented
  ±1 s16 lossy tolerance. This pins the previously-deferred §4.3.7 IMDCT
  normalization scalar to **1.0**: the bare cosine-summation kernel, the
  §4.3.6 window, and the §4.3.8 overlap-add (whose §1.3.2 squared-overlap
  property carries the reconstruction normalization) reproduce the
  reference PCM with no extra Vorbis-specific scaling. The 5.1 fixture
  additionally validates the §4.3.9 mapping-type-0 channel order
  (bitstream `[FL,FC,FR,RL,RR,LFE]` reordered to the WAV interleave layout
  before comparison). The `imdct.rs` / `streaming.rs` docs are updated to
  record the pin. The test carries a minimal RFC-3533 Ogg page de-framer
  as private test scaffolding (container framing belongs in `oxideav-ogg`;
  this avoids a cross-crate dev-dependency) and consumes `expected.wav` as
  opaque black-box-validator output — no reference decoder source is read.

- §7.2.4 step-1 floor-1 amplitude-unwrap glue (`floor1_encode` module:
  `plan_floor1_y`, `Floor1EncodeError`). The floor-1 analogue of the
  floor-0 `plan_floor0_coefficients` glue: it inverts the §7.2.4 step-1
  amplitude synthesis, turning a target reconstructed-post list
  (`[floor1_final_Y]`) plus the floor's `Floor1Header` into the
  always-non-negative packet-domain `[floor1_Y]` values
  `encoder::Floor1Packet::floor1_y` / `write_floor1_packet` consume.
  It walks the decoder's strict left-to-right schedule — recomputing
  each post's `render_point` prediction from the already-reconstructed
  backward neighbours, then the `highroom` / `lowroom` / `room` window —
  and chooses the unique packet value whose decode reproduces the target
  (`0` for an on-prediction post, the zig-zag even/odd candidate inside
  `room`, or the single linear extension past it). Unreachable targets
  (a deviation the geometry has no linear room for, or one outside
  `[0, range)`) are rejected. An independent from-spec decode oracle
  pins the lossless round-trip across the zig-zag region, both linear
  extensions, a full `[0, range)` interior sweep over all four
  multipliers, and the endpoint-only degenerate floor. The floor-1
  WRITE primitive no longer needs its `[floor1_Y]` values supplied by
  hand.

- §8.6.2 residue VQ-encode cascade planner (`residue_encode` module:
  `plan_partition_cascade`, `plan_vector_partition_entries`,
  `ResidueEncodeError`). The glue between the `vq::quantize_vector` leaf
  and the residue WRITE path: given a partition's real spectral residual
  it walks the §8.6.2 cascade in the write direction — gathering each VQ
  read's sub-vector from the running residual at the positions the
  decoder scatters into (§8.6.3 strided for format 0, §8.6.4 contiguous
  for formats 1/2, format 2 reduced to format 1 per §8.6.5), quantising
  each with `quantize_vector`, then subtracting the chosen entry's
  reconstruction so the next cascade stage refines the leftover error
  (the inverse of §8.6.2 step 19's additive accumulation). It produces
  exactly the per-`(partition, pass)` entry-index lists
  `encoder::ResidueVectorPlan::partition_entries` / `write_residue_body`
  consume, so the residue WRITE primitives no longer need their entry
  indices supplied by hand. An independent decode-reconstruct oracle
  pins the round-trip across both addressing formats, the non-divisible
  format-1 tail, and multi-stage cascade refinement.
- §3.2.1 VQ-encode quantiser (`vq::quantize_vector` + `QuantizedEntry` /
  `QuantizeError`): the encode-side inverse of `unpack_vector`. Given a
  target vector it scans the codebook's **used** entries (skipping
  sparse-codebook entries with codeword length `0`, which the decoder
  can never reach), decodes each through the same §3.2.1 lattice /
  tessellation transform, and returns the entry minimising the
  squared-Euclidean distance (ties → lowest index, deterministic). The
  returned `vector` is bit-identical to `unpack_vector` for the chosen
  entry, and `distance_sq` is the residual a cascaded residue stage
  (§8.6.2) would quantise next. This is the leaf the residue / floor 0
  WRITE primitives call to turn real scalars into the explicit entry
  indices they already serialise — the previously-named remaining
  residue-side encode followup.
- §4.3.9 output channel order (`channel_order` module): `speaker_layout`
  / `speaker_at` resolve each encoded-stream channel index to its
  mapping-type-0 physical speaker location across the 1..=8-channel
  layouts the Vorbis I spec fixes (mono; stereo L/R; 3..=5 surround; 5.1
  / 6.1 / 7.1, with the rev-16781 side-pair + rear-center / rear-pair
  forms for 6.1 and 7.1). Counts above eight report
  `Speaker::Unspecified` (the spec leaves them application-defined) and a
  zero count yields no layout. Decode still emits channels in bitstream
  order; this is the documented layout for consumer-side reordering.
- §4.3 fixture-anchored end-to-end silence-decode integration test
  (`tests/silence_stream_decode.rs`): drives the
  `docs/audio/vorbis/fixtures/silence-stream/` packet geometry through
  the full public `StreamingDecoder::push_packet` bitstream → PCM path
  and asserts pure-silence output. Exercises the §4.3.2-step-6 unused-
  floor short-circuit, which is provably invariant to the still-deferred
  IMDCT normalization scalar (`0 · α = 0`), so it is the one end-to-end
  PCM assertion available before the post-IMDCT trace point lands.

## [0.0.11](https://github.com/OxideAV/oxideav-vorbis/compare/v0.0.10...v0.0.11) - 2026-06-15

### Other

- vorbis §4.3.6 spectrum-factoring primitive (encoder-side dot-product inverse)
- §4.3 wrapping audio-packet WRITE driver (round 41)
- scrub pre-existing impl-naming from floor0 module header
- §4.3.3/§4.3.4 residue-bundle planning primitive (round 40)
- §6.2.2 floor 0 packet-body WRITE primitive (round 39)
- §8.6.2 wrapping residue-body WRITE primitive (round 38)
- §8.6.3/§8.6.4/§8.6.5 per-partition value-codeword WRITE primitive (round 37)
- §8.6.2 residue classification grouping layer (pack_residue_classification_groups)
- §8.6.2 residue classification packing primitive (round 35)
- §4.3.6/§4.3.7 encoder window + forward-MDCT composition primitive (round 34)
- §4.3.5 forward channel coupling primitives (round 33)
- §4.3.8 encoder-side framing-inverse primitive (round 32)
- §7.2.3 floor 1 audio-packet body WRITE primitive (round 31)
- drop release-plz.toml — use release-plz defaults across the workspace
- §4.3.6 window pre-multiplication primitive (round 30)
- §4.3.7 forward-MDCT cosine-summation kernel (round 29)
- §4.3.1 audio-packet header WRITE primitive (round 28)
- scrub rustdoc private-link warnings introduced by r27
- §4.2.4 setup-header WRITE primitive (round 27)
- §4.2.4 mode header WRITE primitive (round 26)
- §4.2.4 mapping header WRITE primitive (round 25)
- §8.6.1 residue header WRITE primitive (round 24)
- §6.2.1 floor type 0 header WRITE primitive (round 23)
- §7.2.2 floor type 1 header WRITE primitive (round 22)
- §3.2.1 codebook WRITE primitive + §9.2.2 float32_pack (round 21)
- write_identification_header + write_comment_header — first WRITE primitives (round 20)

### Added

* **Vorbis I §4.3.6 spectrum-factoring primitive — the encoder-side
  inverse of the §4.3.6 dot product (round 42).** `synthesis::factor_spectrum`
  / `synthesis::factor_spectrum_all` recover the residue vector(s) a
  target audio spectrum implies given the encoder's chosen floor curve,
  the algebraic inverse of the decode-side `packet::dot_product` /
  `packet::dot_product_all` element-wise floor × residue product:
  `residue[i] = spectrum[i] / floor[i]`. The round-trip
  `dot_product(floor, factor_spectrum(spectrum, floor)) == spectrum`
  holds bit-exactly wherever the floor is finite and nonzero. A zero
  floor bin is unconstrained (the decode product is zero for any
  residue), so the canonical `0.0` residue is emitted there and a target
  whose spectrum is nonzero over a zero-floor bin is rejected — no
  finite residue could reproduce it. A `None` floor (an `'unused'`
  channel, §4.3.2 step 6 / §4.3.3) yields an empty residue and requires
  an all-zero target. New public surface (re-exported at the crate
  root): `synthesis::factor_spectrum`, `synthesis::factor_spectrum_all`,
  and `synthesis::FactorSpectrumError` (`LengthMismatch`,
  `ChannelCountMismatch`, `NonFiniteFloor`,
  `NonzeroSpectrumOverZeroFloor`). 15 new in-module unit tests bring the
  crate suite from 764 → 779.

* **Vorbis I §4.3 wrapping audio-packet WRITE driver (round 41).**
  `encoder::write_audio_packet` is the composition layer over the four
  per-stage `_into_writer` splice primitives (the §4.3.1 prelude, the
  §6.2.2 floor 0 / §7.2.3 floor 1 packet bodies, and the §8.6.2 residue
  body) plus the §4.3.3/§4.3.4 inverse-mapping layer
  (`plan_residue_bundles`). It serialises one full audio packet in the
  exact inverse of `audio::decode_audio_packet_pre_imdct`'s read order:
  the §4.3.1 prelude, then one floor body per channel in channel order
  (each channel's floor type/header resolved through its submap's
  `floor` index), then one residue body per submap in submap order
  (the per-submap `do_not_decode` flags derived from the bundle plan,
  so §4.3.3 coupling-propagation re-codes a channel pulled back in by a
  coupling partner even though its own floor was `'unused'`). New
  public surface (re-exported at the crate root):

  - `encoder::AudioChannelFloor` (`Type0(Floor0Packet)` /
    `Type1(Floor1Packet)`) — one channel's floor body; the variant must
    match the resolved floor header's `FloorKind`.
  - `encoder::WriteAudioPacketError` — fail-closed gates: prelude
    failure, floor/residue count or floor-type mismatch, out-of-range
    mapping/floor/residue index, bundle-plan rejection, and per-channel
    floor / per-submap residue body failures (each carried verbatim
    with `source()` chaining). Validation precedes emission in full; the
    caller's writer is bit-exactly untouched on every error path.

  Round-trip proven: a mono used/unused packet, a stereo coupled
  packet, and a stereo packet where one channel's floor is `'unused'`
  but coupling re-codes it all decode cleanly through
  `decode_audio_packet_pre_imdct` to the expected per-channel spectra.

* **Vorbis I §4.3.3 + §4.3.4 residue-bundle planning primitive
  (round 40, umbrella round 293).** `encoder::plan_residue_bundles`
  builds the inverse-mapping layer a wrapping §4.3 audio-packet writer
  needs to thread its per-submap residue bodies. Given a `MappingHeader`
  and the per-channel `no_residue` vector the floor decode produced
  (§4.3.2 step 6 — `true` where a channel's floor decoded `'unused'`),
  it (1) applies §4.3.3 nonzero-vector propagation over the mapping's
  coupling steps (the identical rule `packet::nonzero_propagate` runs
  on decode: if either partner of a coupling step is used, both become
  used), then (2) gathers, per submap in submap order, the channels
  with that submap's mux index in ascending channel order plus their
  propagated `no_residue` flags as the per-bundle `do_not_decode`
  slice (§4.3.4 step 2). New public surface (re-exported at the crate
  root):

  - `encoder::ResidueBundlePlan { no_residue, bundles }` — the
    post-§4.3.3 `no_residue` vector plus one `SubmapResidueBundle` per
    submap.
  - `encoder::SubmapResidueBundle { submap, channels, do_not_decode }`
    — one submap's bundle: the channels (ascending; the §4.3.4 step-7
    scatter map) and the matching `do_not_decode` flags the submap's
    `write_residue_body` consumes. Empty submaps still get an
    index-aligned empty bundle.
  - `encoder::PlanResidueBundlesError` — `ZeroSubmaps`,
    `CouplingChannelOutOfRange` (mirrors the §4.3.3 propagate bounds
    gate), `SubmapOutOfRange`, and `MuxTooShort` (a multi-submap
    mapping whose `mux` table is short or selects a nonexistent
    submap). The single-submap case ignores `mux` entirely (the
    implicit-zero path), matching the decoder's `submap_for_channel`.
    Validation is all-or-nothing — every channel's submap resolves
    before any bundle is built.

  12 in-module unit tests bring the crate suite from **741 → 753
  (+12)**: mono single-submap pass-through, stereo coupling pulling an
  unused angle back in, two-submap interleaved-mux gather, empty-submap
  retention, an independent cross-check against `nonzero_propagate`,
  single-submap mux-ignoring, zero-channel degenerate, all four error
  gates, and grep-able `Display` strings keyed by spec section.

* **Vorbis I §6.2.2 floor 0 packet-body WRITE primitive (round 39,
  umbrella round 288).** The inverse of the floor 0 packet decode —
  the amplitude + value-book selector + per-vector VQ codeword run.
  New public surface (re-exported at the crate root):

  - `encoder::write_floor0_packet(packet, header, codebooks)`
    serialises one §6.2.2 floor 0 audio-packet body. `Floor0Packet::Unused`
    emits only the `floor0_amplitude_bits`-wide zero `[amplitude]`
    field — exactly the bits the §6.2.2 step-2 zero-amplitude
    short-circuit reads before returning `'unused'`.
    `Floor0Packet::Curve { amplitude, booknumber, entries }` emits the
    nonzero amplitude (step 1), the
    `ilog(floor0_number_of_books)`-wide `[booknumber]` selector
    (step 4 — a *position* in `floor0_book_list`), then one canonical
    §3.2.1 codeword per VQ entry (steps 7..11). The writer requires
    exactly `ceil(floor0_order / book.dimensions)` entries — the count
    the decode loop reads to fill `[coefficients]` to `floor0_order`
    scalars. A crate-private `write_floor0_packet_into_writer` splice
    helper keeps the established `_into_writer` contract (validation
    precedes emission in full; the caller's writer is bit-exactly
    untouched on every error path), ready for the wrapping §4.3
    audio-packet writer to thread the per-channel floor 0 body.
  - `encoder::Floor0Packet` enum (`Unused` / `Curve`) — the encoder's
    explicit quantisation choices, same knob philosophy as the floor 1
    packet writer's `Floor1Packet` and the residue writer's
    `ResidueVectorPlan`.
  - `encoder::WriteFloor0PacketError` enumerates fourteen fail-closed
    invariants: the §6.2.1 header gates mirrored from
    `Floor0Decoder::new` (`ZeroAmplitudeBits`, `AmplitudeBitsOverflow`,
    `ZeroOrder`, `EmptyBookList`, `BookListTooLong`), the curve gates
    (`ZeroAmplitudeCurve`, `AmplitudeOverflow`, `BooknumberOutOfRange`,
    `ValueBookOutOfRange`, `ValueBookHasNoLookup`, `ZeroDimensionBook`,
    `EntryCountMismatch`), and the per-entry gates (`EntryOutOfRange`,
    `UnencodableEntry`, `Huffman`), with `source()` chaining on the
    wrapping variant.

  20 new in-module unit tests (crate suite **721 → 741, +20**): the
  unused short-circuit's exact one-byte output + decode-back to
  `Floor0Curve::Unused`; three `Curve` roundtrips (single vector,
  multi-vector `ceil(order/dims)` count, second-book selection with a
  2-bit booknumber field) re-read straight off the produced bytes in
  decoder read order; one `Curve` decoded through the real
  `Floor0Decoder` to a 64-bin curve; the public-vs-splice byte
  equality; the seeded-splice no-bits-on-error contract; ten
  error-path rejections; and the grep-able `Display` content for all
  fourteen variants. Spec source:
  `docs/audio/vorbis/Vorbis_I_spec.pdf` §6.2.2 (the floor 0 packet
  decode loop), §6.2.1 (header field bounds), §3.2.1 (canonical
  Huffman codewords), §3.3 (the VQ-context lookup_type gate), §2.1.4
  (LSB-first packing).

* **Vorbis I §8.6.2 residue-body WRITE primitive (round 38, umbrella
  round 281).** The wrapping writer that runs the §8.6.2 step-3..21
  pass/partition/vector loop in the write direction, interleaving the
  classbook codewords (round-35/36 `pack_residue_classifications` /
  `pack_residue_classification_groups` + §3.2.1
  `HuffmanTree::encode_entry`) with the round-37
  `write_residue_partition` bodies in the exact stream order the
  residue decoder reads them back. New public surface (re-exported at
  the crate root):

  - `encoder::write_residue_body(plans, header, codebooks, blocksize,
    do_not_decode)` serialises one full residue body. On pass 0 each
    stride of `classwords_per_codeword` partitions is preceded by one
    classbook codeword per decoded vector (§8.6.2 steps 6..12, final
    partial stride right-padded with the digits the decoder
    reads-and-discards); on every pass each (partition, vector) pair
    whose classification's cascade stage holds a value book emits its
    partition body (steps 13..20). 'Do not decode' vectors and
    'unused' stages emit nothing (steps 15 / 18). A crate-private
    `write_residue_body_into_writer` splice helper keeps the
    established `_into_writer` contract (validation precedes emission
    in full; the caller's writer is untouched on every error path).
  - `encoder::ResidueVectorPlan` describes one §8.6.2 decode vector:
    the per-partition `classifications` row plus the per-(partition,
    pass) `partition_entries` lists — the encoder's explicit
    quantisation choices, same knob philosophy as the floor 1 packet
    writer's `partition_cvals`.
  - `encoder::residue_body_shape(header, blocksize, do_not_decode)`
    returns the §8.6.2 step-1..5 derived `ResidueBodyShape { vectors,
    partitions_to_read }`, including the §8.6.5 format-2
    single-interleaved-vector reduction (`actual_size *= ch`) and its
    all-'do not decode' no-decode shortcut (zero vectors), so callers
    can size plans before quantising.

  A new `WriteResidueBodyError` enumerates seventeen fail-closed
  invariants: the §8.6.1 header/codebook gates mirrored from the
  decoder's construction-time checks (`ClassbookOutOfRange`,
  `ZeroClasswordsPerCodeword`, `ValueBookOutOfRange`,
  `ValueBookHasNoLookup`), the shape gates (`UnsupportedResidueType`,
  `ZeroPartitionSize`, `PlanCountMismatch`, `DoNotDecodePlanNotEmpty`,
  `ClassificationCountMismatch`, `PartitionEntriesCountMismatch`), the
  per-partition gates (`ClassificationOutOfRange`,
  `MissingPartitionEntries`, `UnexpectedPartitionEntries`,
  `Partition` wrapping the round-37 error with (vector, partition,
  pass) coordinates), and the classbook-emission gates
  (`ClassificationPack`, `UnencodableClassbookEntry`, `Huffman`) —
  with `source()` chaining on the three wrapping variants. The
  umbrella `WriteError` grows a `ResidueBody` variant with `From` glue
  and `source()` chain.

  26 new in-module unit tests bring the crate suite from 695 → 721
  (+26): the three shape behaviours (format 0/1 per-channel vectors +
  begin/end limiting, format 2 interleave + all-'do not decode'
  shortcut, the two structural rejections); five end-to-end roundtrips
  through the real `ResidueDecoder` (format 1 single-channel
  byte-shape pin, format 1 two-channel §8.6.2 step-14 interleave,
  format 0 scatter, format 2 de-interleave, multi-pass cascade
  accumulation) plus the multi-group classbook-codeword mid-stream
  interleave and the partial-final-group pad — each pinned against a
  hand-composed stream where order is load-bearing; the 'do not
  decode' channel skip; the two empty-body shapes (format 2 all-dnd,
  begin == end); ten error-path rejections; the splice
  no-bits-on-error contract (error found late in validation);
  public-vs-splice + seeded-splice byte equality; grep-able `Display`
  content for all seventeen variants; and the umbrella `WriteError`
  From + `source()` + crate-level `Error::Write` chain.

  Spec source: `docs/audio/vorbis/Vorbis_I_spec.pdf` §8.6.2 (the
  packet-decode loop, steps 1..21), §8.6.5 (format-2 reduction +
  all-'do not decode' shortcut), §8.6.1, §3.2.1.

* **Vorbis I §8.6.3/§8.6.4/§8.6.5 per-partition value-codeword WRITE
  primitive (round 37, umbrella round 278).** The value half of the
  §8.6.2 residue-body writer (the round-35/36 classification packers
  are the classification half). Two new public functions (re-exported
  at the crate root):

  - `encoder::write_residue_partition(entries, book, residue_type,
    partition_size)` serialises one residue partition body — the
    sequence of value-codebook **entry indices** (the encoder's
    quantisation choice, kept explicit like the floor 1 packet
    writer's `partition_cvals` knob) Huffman-coded in the exact order
    the decoder's §8.6.2 step-19 partition decode reads them back.
    The emission is format-independent: §8.6.3's interleaved scatter
    versus §8.6.4's contiguous append is decode-side addressing, not
    an on-wire difference; the formats differ on the wire only in the
    codeword count. A crate-private
    `write_residue_partition_into_writer` splice helper matches the
    established `_into_writer` splice-point contract (validation
    precedes emission; the caller's writer is untouched on error).
  - `encoder::residue_partition_codeword_count(residue_type,
    partition_size, dimensions)` pins that count: `n / dims` for
    format 0 (§8.6.3 step 1's `[step]`, with the divisibility
    requirement), `ceil(n / dims)` for formats 1 and 2 (§8.6.4's
    read-while-`[i] < [n]` loop; §8.6.5 is reducible to format 1 over
    the interleaved vector).

  A new `WriteResiduePartitionError` enumerates eight fail-closed
  invariants (`UnsupportedResidueType`, `ZeroPartitionSize`,
  `ZeroDimensions`, `ScalarValueBook` — §8.6.1's VQ-context
  value-mapping requirement, mirroring the decoder's
  `ValueBookHasNoLookup` gate —, `Format0NotDivisible`,
  `EntryCountMismatch`, `Huffman` with `source()` chaining,
  `UnencodableEntry`); the umbrella `WriteError` grows a
  `ResiduePartition` variant with `From` glue and `source()` chain.

  16 new in-module unit tests bring the crate suite from 679 → 695
  (+16): the codeword-count behaviours (format-0 exact division,
  format-1/2 ceil, all four structural rejections), a byte-shape pin,
  four end-to-end roundtrips through the real `ResidueDecoder`
  composing classbook codeword + partition bodies by hand (format 1
  two-partition, format 0 scatter, format 2 two-channel
  de-interleave, format-1 partial-final-vector discard), both
  entry-count mismatch directions, the scalar-book rejection, the
  out-of-range and sparse-unused `UnencodableEntry` shapes, the
  Huffman build-error propagation, the splice-emits-no-bits-on-error
  contract, public-vs-splice byte equality, grep-able `Display`
  content for all eight variants, and the umbrella `WriteError` From
  + `source()` + crate-level `Error::Write` chain.

  Spec source: `docs/audio/vorbis/Vorbis_I_spec.pdf` §8.6.2 step 19,
  §8.6.3, §8.6.4, §8.6.5, §8.6.1, §3.2.1.

* **Vorbis I §8.6.2 residue classification grouping layer (round 36,
  umbrella round 274).** New public function
  `encoder::pack_residue_classification_groups(classifications,
  num_classifications, classwords_per_codeword)` (re-exported at the
  crate root) — the structural inverse of the §8.6.2 step-6..9 decode
  walk, sitting directly above the round-35 per-group packer. It slices
  a full per-vector classification array into consecutive groups of
  `classwords_per_codeword` partitions, packs each group via
  `pack_residue_classifications`, and returns one classbook entry index
  per group in stream order — the sequence a residue-body writer then
  Huffman-codes with the classbook.

  When the array length is not a multiple of `classwords_per_codeword`,
  the final partial group is right-padded (the least-significant
  base-`C` positions) with classification index `0`. This mirrors the
  decoder exactly: its step-10..12 unpack loop reads all `classwords`
  digits but discards any whose partition index is
  `>= partitions_to_read` (`if slot < partitions_to_read`), so the pad
  digits are read-and-thrown-away; `0` is the canonical pad (the
  smallest classbook entry that round-trips every kept classification).
  An empty array yields an empty result (the decode loop runs zero
  groups when `partitions_to_read == 0`).

  A new `PackResidueClassGroupsError` enum carries two fail-closed
  paths: `ZeroClasswords` (the §8.6.2 group width = classbook
  `dimensions`, which is `>= 1`) and `Pack { group, source }` (a
  per-group failure tagged with the offending 0-based group index, with
  `Error::source()` chaining through to the inner
  `PackResidueClassError`).

  10 new in-module unit tests bring the crate suite from 669 → 679
  (+10): the `classwords == 1` per-partition identity, the empty-array
  no-op (two widths), an exact-multiple split hand-checked against the
  per-group packer, the partial-final-group zero-pad, an exhaustive
  `group → decoder-unpack → equal` round-trip over a grid of bases
  (1..=5), widths (1..=4), and lengths (0..=11, including non-multiples),
  `ZeroClasswords` rejection (with and without data), the failing-group
  index tag on an out-of-range digit, base-error propagation on group 0,
  `Error::source()` chaining for both variants, and grep-able `Display`
  content.

* **Vorbis I §8.6.2 residue classification packing primitive (round 35,
  umbrella round 267).** New public function
  `encoder::pack_residue_classifications(classifications,
  num_classifications)` (re-exported at the crate root) — the exact
  arithmetic inverse of the §8.6.2 step-10..12 classbook *unpack* the
  residue decoder runs after reading one classbook entry. It packs one
  group of `classwords_per_codeword` classification indices into the
  single classbook entry index a residue-body writer will then
  Huffman-code (§8.6.2 step 9's inverse), the first piece of the
  §8.6.2 residue-body WRITE path.

  The decoder recovers classifications by the descending loop
  `for i in (0..classwords).rev() { class[i] = temp % C; temp /= C }`,
  so group position 0 is the most-significant base-`C` digit and the
  packer computes `temp = Σ class[i]·C^(L-1-i)` by overflow-checked
  Horner accumulation. A new `PackResidueClassError` enum enumerates
  six fail-closed invariants (`ZeroClassifications`,
  `ClassificationsTooLarge`, `EmptyGroup`, `GroupTooLong`,
  `ClassificationOutOfRange`, `PackedValueOverflow`); the umbrella
  `WriteError` grows a `ResidueClassification` variant with `From`
  glue and `source()` chaining.

  14 new in-module unit tests bring the crate suite from 655 → 669
  (+14): single-digit identity across every legal base, hand-computed
  positional weights, the most-significant-position semantics, an
  exhaustive `pack → decoder-unpack → equal` round-trip over every
  group at bases 1..=6 and lengths 1..=4, a base-64 (§8.6.1 maximum)
  round-trip, every error path (zero/oversized base, empty/over-long
  group, out-of-range digit, packed-value overflow), validation-
  precedes-overflow ordering, the umbrella `WriteError` From glue +
  `source()` chain, and grep-able `Display` content for all six
  variants.

* **Vorbis I §4.3.6 / §4.3.7 encoder-side window + forward-MDCT
  composition primitive (round 34, umbrella round 259).** Closes the
  encoder-side mirror of the decoder's `audio::apply_imdct_and_window`
  flow. The forward MDCT (`mdct::mdct_naive`, round 29) and the
  §4.3.6 point-wise window pre-multiplication
  (`synthesis::window_premultiply`, round 30) already shipped as
  independent primitives. This round bundles them into the single
  encoder transform stage. Two new public functions in `mdct` (also
  re-exported at the crate root):

  - `apply_window_and_mdct(block, window, output, scale)` —
    in-place multiplies a length-`N` time-domain block by a length-`N`
    §4.3.1-built window, then runs the §4.3.7 forward MDCT into a
    caller-allocated length-`N/2` spectrum buffer. The block-side
    window product mirrors `synthesis::window_premultiply` exactly
    (`a *= w`); inlining the kernel here keeps `mdct.rs` self-contained
    instead of pulling in a cross-module dependency for one
    multiplication.
  - `apply_window_and_mdct_vec(block, window, scale)` — the spectrum
    `Vec<f32>` allocating wrapper.

  A new public `ApplyWindowAndMdctError` enum unifies the two
  underlying failure modes: a `WindowLengthMismatch` arm for the
  pre-MDCT length check and a `Mdct(MdctError)` arm that forwards the
  forward MDCT's existing errors. `From<MdctError>` is provided so
  the composition can `?` through the bare-MDCT call.

  Tests pin the algebraic content end to end:
  - Length-mismatch rejection happens before any in-place mutation
    (the block is verified unchanged on the error path).
  - Block-length validity and output-slice sizing are propagated
    verbatim from `mdct_naive` via the wrapping `Mdct` arm.
  - The zero window collapses both the block and the spectrum to
    all-zero; the unity window leaves the composed result equal to
    the bare forward MDCT bit-exactly; an alternating non-trivial
    window verifies the composed primitive matches a hand-rolled
    "in-place multiply, then `mdct_naive`" recipe element-wise.
  - Linearity in the block input is exercised with a non-trivial
    window at three blocksizes (64, 256, 1024), confirming the two
    underlying linear maps compose into a still-linear map.
  - The `scale` argument is pinned as a pure post-MDCT multiplier
    (no leakage into the window product or into the cosine sum).
  - `Display` and `Error::source` are checked for both arms — the
    length-mismatch arm has `source() == None`, the `Mdct` arm
    chains to the wrapped `MdctError`.

  The encoder transform stage's mathematical content is now
  end-to-end composable: a §4.3 encoder driver lands a windowed
  spectrum with `apply_window_and_mdct`, the decode-side
  `audio::apply_imdct_and_window` reconstructs the same windowed
  time-domain frame, and the §4.3.8 overlap-add primitive
  (`overlap::OverlapAdd`) closes the round trip for consecutive
  blocks at the TDAC complementarity boundary `w[i]² + w[i+n/2]² == 1`
  that the round-15 `overlap` module already pins.

  Provenance: clean-room implementation grounded entirely in
  `docs/audio/vorbis/Vorbis_I_spec.pdf` §§4.3.1, 4.3.6, 4.3.7 and the
  staged OxideAV-original `docs/audio/vorbis/imdct-cross-reference.md`.
  Each test asserts a mathematical property derivable from the cosine
  summation alone (linearity, zero-input, unity-window identity,
  scale linearity) or from the two existing primitives' algebraic
  composition (manual two-step equivalence) — no fixture data and no
  external reference is consulted.

* **Vorbis I §4.3.5 forward channel coupling primitives
  (round 33, umbrella round 255).** The encoder counterpart of the
  round-11 decoder-side `synthesis::inverse_couple` /
  `synthesis::inverse_couple_all`. Three new public functions in
  `synthesis`:

  - `forward_couple_scalar(l: f32, r: f32) -> (f32, f32)` — applies
    the algebraic inverse of the §4.3.5 step-3 four-quadrant rule to
    a single Cartesian `(L, R)` pair. The four cases are derived
    from the inverse table by inversion:
    - `L > 0 AND L > R` → `M = L`, `A = L - R` (mirrors `M > 0, A > 0`)
    - `R > 0 AND L ≤ R` → `M = R`, `A = L - R` (mirrors `M > 0, A ≤ 0`)
    - `L ≤ 0 AND R > L` → `M = L`, `A = R - L` (mirrors `M ≤ 0, A > 0`)
    - `R ≤ 0 AND R ≤ L` → `M = R`, `A = R - L` (mirrors `M ≤ 0, A ≤ 0`)

    The four conditions are mutually exclusive and exhaustive; the
    boundary ties (`L == 0`, `R == 0`, `L == R`) are absorbed by the
    existing `> 0` / `≤ 0` splits the inverse uses on `M` and `A`.
  - `forward_couple(left, right)` — the in-place vector wrapper, the
    encoder counterpart of `inverse_couple`. Panics on length mismatch
    matching the existing inverse convention.
  - `forward_couple_all(channels, coupling)` — the per-mapping driver.
    Runs every coupling step **in ascending order** (the reverse of
    the §4.3.5 decoder loop's descending direction), producing the
    square-polar channels the residue encoder will quantise. Reuses
    the existing `CouplingError` error type (matching variants and
    messages) and the same `lo`/`hi` slice-split pattern as the
    inverse-side driver.

  The round-trip property
  `inverse_couple_all(forward_couple_all(x)) == x` holds bit-exactly
  for every legal input across single-step coupling, multi-step
  coupling, both magnitude-vs-angle channel orderings (`mag < ang`
  and `mag > ang`), and chained coupling where one step's output is
  another step's input. Uncoupled channels are left untouched.

  The umbrella `crate::Error` already had a `Coupling` variant from
  round 11; no new error variants are introduced. The `synthesis`
  module documentation grows a "Forward channel coupling" section
  with the four-case derivation table.

  19 new in-module unit tests bring the suite from **646 → 665
  (+19)**:

  - `forward_couple_scalar_all_four_quadrants` — fires the four
    sign-of-`M`/sign-of-`A` cases by feeding the `(L, R)` pairs the
    existing inverse test produces.
  - `forward_couple_scalar_handles_boundary_ties` — `L == R` for
    positive, zero, and negative ties.
  - `forward_couple_scalar_handles_zeros` — every axis-zero input.
  - `forward_then_inverse_couple_scalar_is_identity_on_grid` —
    exhaustive `(L, R)` round-trip on the `[-10, 10] × [-10, 10]`
    integer grid (441 pairs), covering every quadrant and every
    `L`/`R` sign comparison.
  - `forward_then_inverse_couple_scalar_is_identity_on_floats` —
    ten non-integer / non-tie probes including very small and very
    large magnitudes.
  - `forward_couple_pointwise_matches_scalar_function` — the vector
    wrapper applies the scalar function element-by-element with no
    cross-talk.
  - `forward_then_inverse_couple_is_identity_on_vectors` — end-to-end
    vector round-trip over eight diverse `(L, R)` pairs.
  - `forward_couple_panics_on_length_mismatch` — defensive panic on
    length mismatch.
  - `forward_couple_all_single_step_low_high` and
    `forward_couple_all_single_step_high_low` — the `mag < ang` and
    `mag > ang` slice-split branches of the driver.
  - `forward_couple_all_rejects_out_of_range_channel`,
    `forward_couple_all_rejects_out_of_range_magnitude_channel`,
    `forward_couple_all_rejects_same_channel` — the three error
    paths.
  - `forward_couple_all_empty_coupling_is_noop`.
  - `forward_then_inverse_couple_all_is_identity_single_step` —
    decoder-then-encoder round-trip on a two-channel stereo
    fixture-style vector pair.
  - `forward_then_inverse_couple_all_is_identity_multi_step` —
    two-step coupling on a four-channel stream.
  - `forward_then_inverse_couple_all_is_identity_with_reversed_channel_order`
    — exercises the `mag > ang` driver branch through the full
    round-trip.
  - `forward_couple_all_leaves_uncoupled_channels_alone` — a 5.1
    layout where one coupling step touches only channels 0/1.
  - `forward_couple_all_step_order_matters_for_chained_coupling` —
    two chained coupling steps that share channel 1; the encoder's
    ascending order undoes the decoder's descending order
    end-to-end.

  Spec source: `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.3.5 (the
  decoder-side step-3 four-quadrant rule; the encoder-side rule is
  derived from it by algebraic inversion as documented in the
  `synthesis` module header).

* **Vorbis I §4.3.8 encoder-side framing-inverse primitive
  (round 32, umbrella round 253).** First encoder-side DSP framing
  primitive — the inverse of the round-15 decoder-side
  [`crate::overlap::OverlapAdd`]. New module `framing` exports
  `FrameSplitter::take_frame(cur_n, analysis_window)` which slices
  the next length-`cur_n` windowed time-domain block from an internal
  PCM buffer, applies the §4.3.1 analysis window pointwise, and
  advances the read base per the §4.3.8 recurrence
  `g_{N+1} = g_N + prev_n*3/4 - cur_n/4` (the same alignment rule
  the round-15 `OverlapAdd` reverses on the decoder side). A new
  `FramingError` enumerates four structural rejections —
  `NotPowerOfTwo`, `FrameTooSmall`, `NeedMoreInput { shortfall }`,
  `WindowLengthMismatch` — and surfaces through `Error::Framing` in
  the umbrella with the matching `From` glue and `source()` chain.
  Two `From` impls (`OverlapError -> FramingError`,
  `WindowPremultiplyError -> FramingError`) let callers driving both
  the decoder-side overlap-add and the encoder-side splitter through
  a shared shape normalise on one error type.

  The splitter's internal model keeps the previous frame's right
  half in the buffer between calls (mirroring the decoder
  `OverlapAdd::prev_right_half` storage) so the overlap region of
  the next frame is already buffered; a separate
  `FrameSplitter::advance_pending_stride(cur_n)` method applies the
  signed `prev_n/4 - cur_n/4` stride before the next slice. On the
  priming frame, the caller is expected to push zero-padded left-
  half PCM (the spec's "data is not returned from the first frame"
  priming step's encoder counterpart).

  The reconstruction property is exercised end-to-end:
  `splitter_then_overlap_add_round_trips_constant` and
  `splitter_then_overlap_add_round_trips_ramp` push a continuous
  PCM stream through `FrameSplitter`, feed each frame through a
  second window multiplication standing in for the MDCT/IMDCT
  round-trip, and overlap-add back to PCM; the squared-window
  identity `w[i]² + w[i + n/2]² = 1` from §1.3.2 makes the round-
  trip a per-sample identity inside every non-priming overlap-add
  return-range, modulo `f32` tolerance.

  23 new in-module unit tests bring the suite from 604 to 627:
  four error-path rejections (non-power-of-two, zero-length,
  too-small, window-length-mismatch), a `NeedMoreInput` shortfall
  test that validates the recovery path, the two `From` conversions,
  priming-state checks, first-frame buffer-zero slicing, reset back
  to priming, window-application pointwise correctness, the §4.3.1
  hybrid-window lead-in/tail zeroing, three stride/read-base
  geometry tests (second frame starts at previous center, third
  frame advances per the recurrence, long-then-short positive
  stride drops 48 samples), short-then-long negative-stride no-drop,
  the priming left-half zero-padding convention, the two end-to-end
  round-trip reconstructions (constant and ramp), `push_pcm` /
  `buffered` sanity, `frame_required_samples` returns `cur_n`, and
  `advance_pending_stride` is idempotent in the priming state.

  Spec source: `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.3.8
  (overlap-add alignment rule + return-length formula), §4.3.1
  (Vorbis window), §1.3.2 (squared-overlap reconstruction
  property), §4.2.2 (blocksize set).

* **Vorbis I §7.2.3 floor 1 audio-packet body WRITE primitive
  (round 31, umbrella round 250).** First audio-packet *body* WRITE
  primitive — the next step in the §4.3 audio-packet writer after the
  round-28 §4.3.1 prelude. New public function
  `encoder::write_floor1_packet(&Floor1Packet, &Floor1Header,
  &[VorbisCodebook])` serialises the §7.2.3 floor 1 per-channel
  payload: the `[nonzero]=0` unused short-circuit (single zero bit),
  or the full `[nonzero]=1` body — two endpoint amplitudes
  (`ilog([range]-1)` bits each) followed by per-partition emissions
  threading `cval` through `cval & csub` → `cval >>= cbits` exactly
  like the round-9 decoder reads them back. Each codeword is emitted
  MSb-first via the new `HuffmanTree::encode_entry` helper, matching
  the §3.2.1 canonical-codeword convention.
  A `Floor1Packet` struct (`nonzero`, `floor1_y`, `partition_cvals`)
  describes the on-wire shape; a `WriteFloor1PacketError` enum
  enumerates ten §7.2.3 invariant violations
  (`YLengthMismatch`, `CvalListLengthMismatch`, `IllegalMultiplier`,
  `EndpointOverflow`, `BadClassIndex`, `MasterbookOutOfRange`,
  `SubclassBookOutOfRange`, `Huffman`, `UnencodableY`,
  `NoneBookNonzeroY`, `UnencodableCval`). The umbrella `WriteError`
  grows a matching `WriteError::Floor1Packet` variant with the
  `From` glue and `source()` chain. The encoder-side Huffman helper
  `HuffmanTree::encode_entry(entry, &mut BitWriterLsb)` performs the
  byte-exact inverse of `decode_entry`: a DFS path from the root to
  the requested leaf records the canonical codeword, emitted
  MSb-first; an `EncodeError::UnknownEntry` variant flags entries
  not present in the tree's used set. A crate-private
  `write_floor1_packet_into_writer` splice helper is shaped to
  thread the body between the §4.3.1 prelude and the §4.3.4
  residue body in the wrapping audio-packet writer (followup).
  26 new in-module unit tests bring the suite from 578 to 604: the
  unused short-circuit byte shape, the full-body byte shape, the
  hand-trace `floor1_packet_full_body_round_trips_against_decoder`
  matching the round-9 floor1 decoder fixture, the master/sub-cascade
  roundtrip with a 4-entry master book + two sub-books, the `None`
  sub-book Y=0 acceptance + nonzero rejection, length-mismatch
  rejections on both `floor1_y` and `partition_cvals`, illegal-
  multiplier rejection, endpoint-overflow rejection, bad-class-index
  rejection, masterbook / subclass-book out-of-range rejections,
  unencodable-Y / unencodable-cval rejections, roundtrip across all
  four multiplier values, splice helper matches public writer, splice
  in unused path writes exactly one bit, the umbrella `WriteError`
  From glue + `crate::Error` glue, and `Display` informativeness for
  every error variant. The Huffman side adds seven more tests
  exercising `encode_entry`: the §3.2.1 worked-example codeword
  emission, concatenated encode→decode roundtrip across the worked
  example, balanced 16-entry length-4 tree roundtrip, sparse codebook
  rejection of unused entries, single-entry codebook emits one zero
  bit + non-sole entry rejection, generic encode→decode roundtrip on
  a mixed-length tree, `EncodeError` `Display` content. Spec source:
  `docs/audio/vorbis/Vorbis_I_spec.pdf` §7.2.3, §7.2.2, §3.2.1,
  §9.2.1 (`ilog`), §2.1.4 (LSB-first packing).

* **Vorbis I §4.3.6 window pre-multiplication primitive (round 30,
  umbrella round 246).** New public function
  `synthesis::window_premultiply(time_frame, window)` that applies
  the §4.3.7 closing window-application step (the IMDCT output
  multiplied element-wise by the §4.3.1-built window) in place on a
  caller-owned slice. The pipeline-side step previously inlined
  inside `audio::apply_imdct_and_window` as a four-line `zip` loop
  is now a discrete, named primitive with a structured error type
  (`WindowPremultiplyError::LengthMismatch`) and a matching
  `AudioPacketError::WindowPremultiply` surface, mirroring the
  §4.3.5 inverse-coupling primitive's
  (`inverse_couple` / `CouplingError`) shape. Twelve unit tests pin
  the pointwise product, the in-place semantics, the zero / unity /
  empty-slice edge cases, the lead-in / tail / plateau interaction
  with a hybrid `vorbis_window`, sign preservation, fail-closed
  behaviour on length mismatch, and the error `Display` format.
  Derived from `docs/audio/vorbis/Vorbis_I_spec.pdf` §4.3.7 +
  §1.3.2 and `docs/audio/vorbis/imdct-cross-reference.md`
  §"Window-function equivalence".

* **Vorbis I §4.3.7 forward-MDCT cosine-summation kernel (round 29,
  umbrella round 243).** New `mdct` module containing public
  `mdct::mdct_naive` + `mdct::mdct_naive_vec` functions that take one
  channel's length-`N` time-domain block and return the length-`N/2`
  audio-spectrum vector — the encode-side counterpart to the
  round-16 inverse-MDCT kernel in `crate::imdct`. The direct-form
  O(N²) cosine summation it implements is derived from the IMDCT
  formula pinned in `docs/audio/vorbis/imdct-cross-reference.md` via
  a single mathematical step: the forward kernel is the linear
  matrix-transpose of the IMDCT kernel, so the two directions share
  one cosine matrix. The derived identity `mdct(imdct(X)) == (N/2) · X`
  follows directly from computing `Cᵀ · C` with the cosine
  product-to-sum identity (every off-diagonal entry sums to zero
  because its cosine arguments form a non-zero rational multiple of
  `2π/N` integrated over the full `N`-term sum). The full
  derivation is laid out in the module documentation and uses only
  the IMDCT formula already in the in-repo cross-reference document
  plus the standard cosine product-to-sum identity and the
  closed-form sum of a cosine sampled at integer multiples of `2π/N`.
  Like the inverse direction the kernel is bare (un-normalized) and
  takes the same `scale: f32` knob; the Vorbis-specific
  normalization scalar remains a deferred-fixture concern documented
  by the IMDCT cross-reference document. A new `MdctError` enum
  enumerates two structural invariants (`BlockNotPowerOfTwo`,
  `OutputLenMismatch`), matching the `ImdctError` shape, and the
  umbrella `Error` grows a matching `Error::Mdct(MdctError)` variant
  with the corresponding `From` glue and `source()` chain. 14 new
  in-module unit tests bring the crate-wide test count from 555 to
  569: six error-path tests, linearity + zero-input properties at
  three blocksizes, the `mdct(imdct(X)) == (N/2) · X` round-trip
  identity at three blocksizes plus a companion test confirming
  that `scale = 2/N` recovers the spectrum directly, three
  hand-computed N=4 impulse tests pinning the cosine argument
  formula at three distinct input indices, a full-basis N=64
  matrix-transpose cross-check confirming `mdct[n][k] == imdct[n][k]`
  entry-wise, the `scale`-is-pure-multiplier guard, and the
  `MdctError::Display` smoke test. Encoder-side §4.3.6 window
  pre-multiplication, the §4.3.8 overlap-add inversion / framing on
  the encode side, an FFT-decomposed forward-MDCT fast path
  validating against this kernel's output, the §4.3 audio-packet
  writer wrapping mode + floor + residue + spectrum + MDCT encode,
  and pinning the Vorbis-specific normalization scalar are explicit
  followups.

* **Vorbis I §4.3.1 audio-packet header WRITE primitive (round 28,
  umbrella round 240).** New public function
  `encoder::write_audio_packet_header` serialises an
  `crate::packet::AudioPacketHeader` to the §4.3.1 audio-packet
  prelude bit pattern — the first audio-packet WRITE primitive (after
  the three header-packet writers and the six nested setup-header
  sub-block writers + the round-27 wrapping setup-header writer). The
  prelude is the §4.3.1 step 1..4 region: a 1-bit `packet_type` (must
  be 0; §4.3 reject path is not on the writer's emission set), an
  `ilog([vorbis_mode_count] - 1)`-bit `mode_number` (using the
  existing `codebook::ilog` helper — collapses to a zero-bit read in
  the single-mode degenerate case), and then on a long block, two
  1-bit window flags (`previous_window_flag`, `next_window_flag`);
  the short-block path emits no further bits per §4.3.1 step 4b. The
  writer is the byte-exact inverse of `crate::packet::read_packet_header`:
  given a matching `(setup, blocksize_0, blocksize_1)` context the
  roundtrip property
  `read_packet_header(&mut BitReaderLsb::new(&write_audio_packet_header(&h, &setup, b0, b1)?), &setup, b0, b1) == Ok(h)`
  holds for every legal `AudioPacketHeader`. A new
  `WriteAudioPacketHeaderError` enum enumerates five §4.3.1 invariant
  violations — `EmptyModeList`, `BadModeNumber`, `BlockflagMismatch`,
  `BlocksizeMismatch`, `ShortBlockHasWindowFlag` — that the writer
  refuses with a fail-closed gate rather than emit a prelude the
  parser would reject. The crate-private
  `write_audio_packet_header_into_writer` companion is shaped to
  splice the prelude into the (still-followup) wrapping §4.3 audio-
  packet writer, matching the existing `write_codebook_into_writer` /
  `write_floor0_header_into_writer` /
  `write_floor1_header_into_writer` /
  `write_residue_header_into_writer` /
  `write_mapping_header_into_writer` / `write_mode_header_into_writer`
  splice points. The umbrella `WriteError` grows a
  `WriteError::AudioPacket(WriteAudioPacketHeaderError)` variant with
  the matching `From` glue and `source()` chain. 21 new in-module
  unit tests cover the 1-bit short-block emission, the 3-bit long-
  block emission, byte-shape pinning for the long-block two-mode
  fixture, parser-side fixture cross-verification, an exhaustive
  scan across all four (`previous_window_flag`, `next_window_flag`)
  combinations on a long-block single-mode stream, a roundtrip
  across every `mode_count` value in `1..=64` (the 6-bit setup-header
  `mode_count - 1` range), five rejection tests covering each new
  error variant, two short-block-with-flag rejection variants,
  the `WriteError::AudioPacket` `From` glue + `source()` chain
  pinning, a `Display` non-emptiness smoke test across all five
  variants, a writer-vs-parser invariant cross-check that confirms
  a struct the writer refuses is also refused by the parser when the
  equivalent bits are hand-rolled, and a splice-vs-public-writer
  byte equality check.

* **Vorbis I §4.2.4 wrapping setup-header WRITE primitive (round 27,
  umbrella round 234).** New public function
  `encoder::write_setup_header` serialises a complete
  `crate::setup::VorbisSetupHeader` into a single byte-aligned
  §4.2.4 packet — the third Vorbis I header, after identification
  and comment. The wrapping writer stitches the six nested-block
  writers (codebook, floor 0, floor 1, residue, mapping, mode that
  landed across rounds 21..26) into one entry point, threading the
  per-block context tuples through the `?` chain exactly as the
  §4.2.4 walker reads them: codebook bodies emitted back-to-back,
  floor entries prefixed with a 16-bit `floor_type` selector then
  dispatched to the per-type splice writer, residue entries prefixed
  with a 16-bit `residue_type` selector then dispatched to the
  residue splice writer, mapping bodies passing the
  `(audio_channels, floor_count, residue_count)` context tuple, and
  mode bodies passing the `mapping_count` context. The §4.2.1 7-byte
  common header (packet type `0x05` + ASCII `"vorbis"` magic) is
  emitted before the bit-packed body, and the trailing 1-bit framing
  flag is emitted before the final §2.1.8 zero-padding tail. Given
  the identification header's `audio_channels` parameter, the
  bit-exact roundtrip property
  `parse_setup_header(&write_setup_header(&h, audio_channels)?, audio_channels) == h`
  holds for every legal `VorbisSetupHeader`. A new
  `WriteSetupError` enum enumerates 17 §4.2.4 invariant violations
  the wrapping layer enforces itself (zero audio channels, empty /
  oversized codebook / time / floor / residue / mapping / mode lists,
  nonzero time placeholder, unsupported floor type, floor
  type/kind disagreement, false framing flag); nested-block
  failures continue to surface as the existing dedicated error
  types (`WriteCodebookError`, `WriteFloor0Error`,
  `WriteFloor1Error`, `WriteResidueError`, `WriteMappingError`,
  `WriteModeError`) through the umbrella `WriteError`'s `?` chain.
  The umbrella `WriteError` enum grows a
  `WriteError::Setup(WriteSetupError)` variant with the matching
  `From` glue and `source()` chain. 29 new tests bring the in-module
  suite to 532: the 7-byte common-header prefix is pinned, six
  positive round-trip fixtures exercise the layout's main branches
  (two-codebook, time-count at the 6-bit upper edge of 64, mixed
  floor 0 + floor 1 kinds, mixed residue types 0 + 1 + 2,
  stereo-coupled mapping at `audio_channels = 2`, two modes), 14
  rejection tests cover every new `WriteSetupError` variant, two
  propagation tests confirm nested `WriteMode` / `WriteFloor1`
  failures surface as the matching umbrella variant, the
  `WriteSetupError::Display` non-emptiness smoke test covers all 17
  variants, the `WriteError::Setup` `From` glue + `source()` chain
  is checked end-to-end, and an alignment test confirms the
  writer's fail-closed gate refuses a struct the round-5 parser
  would also reject.

* **Vorbis I §4.2.4 mode header WRITE primitive (round 26,
  umbrella round 228).** New public function
  `encoder::write_mode_header` serialises a `crate::setup::ModeHeader`
  to the §4.2.4 "Modes" body bit pattern. The function is the
  bit-exact inverse of the round-5 mode parser
  (`setup::parse_mode_header`): given the same context value the
  parser is supplied with — `mapping_count` (the length of the
  mapping list earlier in the setup body) — the round-trip property
  `local_parse_mode_for_tests(&mut BitReaderLsb::new(&write_mode_header(&h, mapping_count)?), mapping_count) == h`
  holds for every legal `ModeHeader`. The §4.2.4 "Modes" body is a
  single fixed-width 41-bit record: 1 bit `blockflag`, 16 bit
  `windowtype` (must be 0 per §4.2.4 step 2e), 16 bit `transformtype`
  (must be 0 per §4.2.4 step 2e), 8 bit `mapping` (range-checked
  against `mapping_count`). The body is emitted **without** the
  surrounding `vorbis_mode_count - 1` 6-bit count field and
  **without** the trailing 1-bit framing flag (both are the
  setup-header walker's responsibility). The crate-private
  `write_mode_header_into_writer` companion mirrors the splice
  point shape established by `write_codebook_into_writer`,
  `write_floor1_header_into_writer`,
  `write_floor0_header_into_writer`,
  `write_residue_header_into_writer`, and
  `write_mapping_header_into_writer`, allowing the still-pending
  setup-header writer to slot the mode body into a wider
  bit-packed stream. A new `WriteModeError` enum enumerates three
  §4.2.4 step-2e invariant violations (`NonZeroWindowType`,
  `NonZeroTransformType`, `BadMapping`); each refuses the call
  without emitting any bits rather than serialise a header the
  parser would reject. The umbrella `WriteError` enum grows a
  `WriteError::Mode(WriteModeError)` variant with the matching
  `From` glue and `source()` chain. 15 new tests pin the byte
  shape on the short-block fixture (41-bit body, 6 bytes), the
  long-block fixture at `mapping = 1` (re-decoded to confirm both
  LSB-first packing and the bit positions of the four fields),
  the constant-41-bit-body length check across five `mapping_count`
  sweep values (`{1, 2, 7, 32, 255}`), three bit-exact roundtrip
  fixtures (short-block minimal, long-block at mapping index 1,
  `mapping = 255` against `mapping_count = 256` at the 8-bit upper
  edge), every `WriteModeError` rejection variant
  (`NonZeroWindowType(1)`, `NonZeroTransformType(2)`, plus two
  `BadMapping` shapes: `mapping == mapping_count` boundary and
  `mapping = 200` against `mapping_count = 4`), the
  `WriteError::Mode(_)` `From` + `source()` chain, the umbrella
  Display forwarding through to the inner enum, and three splice-point
  tests (appends-after-existing-bits across a sub-byte 11-bit seed,
  emits-no-bits on `NonZeroWindowType`, emits-no-bits on the
  `mapping == mapping_count` boundary). The §4.2.4 setup-header
  splice that stitches all six nested-block writers (plus the
  leading `audio_channels` glue and the trailing framing bit) and
  the audio-packet WRITE primitives remain explicit followups.

* **Vorbis I §4.2.4 mapping header WRITE primitive (round 25,
  umbrella round 223).** New public function
  `encoder::write_mapping_header` serialises a
  `crate::setup::MappingHeader` to the §4.2.4 "Mappings" body bit
  pattern. The function is the bit-exact inverse of the round-5
  mapping parser (`setup::parse_mapping_header`): given the same
  context tuple the parser is supplied with —
  `(mapping_type, audio_channels, floor_count, residue_count)` — the
  round-trip property
  `local_parse_mapping_for_tests(&mut BitReaderLsb::new(&write_mapping_header(&h, audio_channels, floor_count, residue_count)?), audio_channels) == h`
  holds for every legal `MappingHeader`. The crate-private
  `write_mapping_header_into_writer` companion mirrors the splice
  point shape established by `write_codebook_into_writer`,
  `write_floor1_header_into_writer`,
  `write_floor0_header_into_writer`, and
  `write_residue_header_into_writer`, allowing the still-pending
  setup-header writer to slot the mapping body into a wider
  bit-packed stream. A new `WriteMappingError` enum enumerates
  eleven §4.2.4 invariant violations (`UnsupportedMappingType`,
  `ZeroAudioChannels`, `SubmapsOutOfRange`, `CouplingStepsOverflow`,
  `BadCouplingChannels`, `CouplingChannelOverflow`,
  `MuxLengthMismatch`, `BadMuxValue`, `SubmapCountMismatch`,
  `BadSubmapFloor`, `BadSubmapResidue`); each refuses the call
  without emitting any bits rather than serialise a header the
  parser would reject. The umbrella `WriteError` enum grows a
  `WriteError::Mapping(WriteMappingError)` variant with the matching
  `From` glue and `source()` chain. 36 new tests pin the byte shape
  on the minimal-mono fixture (44-bit body), the stereo-coupled
  fixture at `audio_channels = 2` (54-bit body), the closed-form
  bit-length formula on a multi-submap multi-channel shape with
  coupling (100 bits), eleven bit-exact roundtrip fixtures
  (minimal-mono, stereo-coupled, stereo no-coupling, 5.1-channel
  with three coupling steps and two submaps, submaps at the 16
  upper edge, coupling_steps at the 256 upper edge, submap floor
  and residue indices at the 255 upper edge against 256-entry
  counts, time_placeholder sweep, 8-channel coupling width
  (`channel_bits = 3`), 3-channel coupling width
  (`channel_bits = 2`), 255-channel coupling width
  (`channel_bits = 8`), 4-channel 2-submap mux cycle), three
  encoding-form selection tests (dense submaps form when
  `submaps == 1`, dense coupling form when `coupling.is_empty()`,
  explicit submaps form when `submaps > 1`), every
  `WriteMappingError` rejection variant (including
  `magnitude == angle`, channel out-of-range, coupling-on-mono,
  mux-length mismatch in both directions, bad mux value, submap
  count mismatch, bad submap floor, bad submap residue), the
  `WriteMappingError::Display` non-emptiness smoke test across
  twelve enumerated cases, the `WriteError::Mapping` `From` +
  `source()` chain, and two splice-point tests
  (appends-after-existing-bits + emits-no-bits-on-error). Mode
  WRITE, audio-packet WRITE, and the setup-header splice that
  stitches all six nested-block writers together remain explicit
  followups.

* **Vorbis I §8.6.1 residue header WRITE primitive (round 24,
  umbrella round 218).** New public function
  `encoder::write_residue_header` serialises a
  `crate::setup::ResidueHeader` to the §8.6.1 residue-header bit
  pattern common to all three residue formats (0, 1, 2). The function
  is the bit-exact inverse of the round-5 residue header parser
  (`setup::parse_residue_header`): the property
  `parse_residue_header_via_local_helper(&write_residue_header(&h)?, h.residue_type) == h`
  holds for every legal `ResidueHeader`. The crate-private
  `write_residue_header_into_writer` companion mirrors the splice
  point shape established by `write_codebook_into_writer`,
  `write_floor1_header_into_writer`, and
  `write_floor0_header_into_writer`, allowing the still-pending
  setup-header writer to slot the residue body into a wider
  bit-packed stream. A new `WriteResidueError` enum enumerates eight
  §8.6.1 invariant violations (`UnsupportedResidueType`,
  `ResidueBeginOverflow`, `ResidueEndOverflow`,
  `PartitionSizeOutOfRange`, `ClassificationsOutOfRange`,
  `CascadeLengthMismatch`, `BooksLengthMismatch`,
  `BooksCascadeMismatch`); each refuses the call without emitting any
  bits rather than serialise a header the parser would reject. The
  umbrella `WriteError` enum grows a
  `WriteError::Residue(WriteResidueError)` variant with the matching
  `From` glue and `source()` chain. Twenty-eight new tests pin the
  byte shape on the residue-type-2 minimal fixture (98-bit packet),
  the closed-form bit-length formula on a two-class mixed-cascade
  shape, eleven roundtrip fixtures (type-0 / type-1 / type-2; begin
  and end at the 24-bit upper edge; partition_size at the 2^24 upper
  edge; classifications at the 64 upper edge with a 64-byte cascade
  sweep; cascade=0xFF (all eight stages) and cascade=0x00 (no
  stages); cascade high-bits=31/low-bits=7 upper edge; alternating
  bitflag classes across consecutive entries; classbook=255 upper
  edge), every `WriteResidueError` rejection variant (with
  `BooksCascadeMismatch` in both `book_present` directions), the
  `WriteResidueError::Display` non-emptiness smoke test, the
  `WriteError::Residue` `From` + `source()` chain, and two
  splice-point tests (appends-after-existing-bits + emits-no-bits-on-
  error). The crate's public surface also gains
  `encoder::write_floor0_header` and `encoder::WriteFloor0Error`
  re-exports retroactively for round 23 (they were already
  implemented but missed the lib.rs re-export list).

* **Vorbis I §6.2.1 floor type 0 header WRITE primitive (round 23,
  umbrella round 212).** New public function
  `encoder::write_floor0_header` serialises a
  `crate::setup::Floor0Header` to the §6.2.1 floor-type-0 header bit
  pattern. The function is the bit-exact inverse of the round-7 floor 0
  parser: the property
  `parse_floor0_header(&mut BitReaderLsb::new(&write_floor0_header(&h)?))? == h`
  holds for every legal `Floor0Header`. The crate-private
  `write_floor0_header_into_writer` companion mirrors the §7.2.2 floor 1
  splice point shape established in round 22, allowing the still-pending
  setup-header writer to splice the floor 0 body into a wider
  bit-packed stream. A new `WriteFloor0Error` enum enumerates three
  §6.2.1 invariant violations (`AmplitudeBitsOverflow`,
  `EmptyBookList`, `BookListTooLong`); each refuses the call without
  emitting any bits rather than serialise a header the parser would
  reject. The umbrella `WriteError` enum grows a
  `WriteError::Floor0(WriteFloor0Error)` variant with the matching
  `From` glue and `source()` chain. Sixteen new tests pin the byte
  shape on the minimal `setup::tests::minimal_floor0` fixture, the
  parametric bit-length formula
  `8 + 16 + 16 + 6 + 8 + 4 + 8 × number_of_books` on the 1-book
  minimum and 16-book maximum, six roundtrip variants covering field
  extremes (`u8::MAX` / `u16::MAX` corners, `amplitude_bits = 63`,
  16-book maximum, all-zero spec-legal writer-input), each rejection
  path, the splice-point append behaviour, and the fail-closed
  validate-before-emit contract on the `into_writer` helper.

* **Vorbis I §7.2.2 floor type 1 header WRITE primitive (round 22,
  umbrella round 206).** New public function
  `encoder::write_floor1_header` serialises a
  `crate::setup::Floor1Header` to the §7.2.2 floor-type-1 header bit
  pattern. The function is the bit-exact inverse of the round-9 floor 1
  parser: the property
  `parse_floor1_header(&mut BitReaderLsb::new(&write_floor1_header(&h)?))? == h`
  holds for every legal `Floor1Header`. The crate-private
  `write_floor1_header_into_writer` companion is shaped to splice the
  floor 1 body into the surrounding setup-header writer (still a
  followup), matching the `write_codebook_into_writer` splice point
  established in round 21. A new `WriteFloor1Error` enum enumerates
  thirteen §7.2.2 invariant violations (`PartitionsOverflow`,
  `PartitionClassListMismatch`, `PartitionClassValueOverflow`,
  `ClassCountMismatch`, `IllegalClassDimensions`,
  `SubclassesOverflow`, `MasterbookPresenceMismatch`,
  `SubclassBookCountMismatch`, `SubclassBookOverflow`,
  `IllegalMultiplier`, `RangebitsOverflow`, `XListLengthMismatch`,
  `XListValueOverflow`); each refuses the call without emitting any
  bits rather than serialise a header the parser would reject. The
  umbrella `WriteError` enum grows a `WriteError::Floor1(WriteFloor1Error)`
  variant with the matching `From` glue and `source()` chain; the
  module-level `Error` enum's `Write` docstring is updated to mention
  the new §7.2.2 invariant gate. 31 new tests cover the byte-shape of
  the minimal §7.2.2 fixture (32-bit packet built explicitly through
  `BitWriterLsb` step-by-step to pin the exact bytes), the closed-form
  bit-length formula on a non-trivial two-class shape with masterbooks
  (108 bits → 14 bytes), nine bit-exact roundtrip fixtures (minimal,
  zero-partitions corner, multiple-partitions-same-class,
  multiple-classes, max-subclasses=3 with full eight-slot
  subclass-book table, subclass-book at the `Some(254)` upper edge,
  rangebits=0 corner with all-zero x_list, rangebits=15 at the upper
  edge, partitions=31 max-5-bit-field, and the max-class-index=15
  max-4-bit-field shape), every `WriteFloor1Error` rejection variant
  (each of the thirteen), the `WriteFloor1Error::Display` non-emptiness
  smoke test across every variant, and the `WriteError::Floor1`
  `From` + `source()` chain. Test count: **409 total (378 → 409, +31)**.
  Floor 0 WRITE, residue WRITE, mapping / mode WRITE, audio-packet
  WRITE, and the setup-header splice are explicit followups.

* **Vorbis I §3.2.1 codebook WRITE primitive + §9.2.2 `float32_pack`
  encoder-side companion (round 21, umbrella round 201).** New public
  function `encoder::write_codebook` serialises a `VorbisCodebook`
  (the round-3 parser's output type) to the §3.2.1 codebook-header
  bitstream shape. The writer picks the densest legal length
  encoding from the codebook's content per the auto policy:
  any `UNUSED_ENTRY` sentinel forces sparse unordered, otherwise
  non-decreasing lengths choose ordered, otherwise dense unordered.
  The bit-exact roundtrip property
  `parse_codebook(&mut BitReaderLsb::new(&write_codebook(&book)?))?
  == book` holds for every legal input across all three length
  encodings and all three lookup types (None / Lattice / Tessellation).
  `WriteCodebookError` enumerates eleven §3.2.1 / §9.2.2 invariant
  violations (`ZeroEntries`, `LengthTableMismatch`,
  `IllegalCodewordLength`, `IllegalValueBits`, `MultiplicandOverflow`,
  `LatticeMultiplicandCountMismatch`,
  `TessellationMultiplicandCountMismatch`,
  `UnrepresentableLookupFloat`, `OrderedHasUnusedEntries`,
  `OrderedNotMonotonic`, `LookupCountOverflow`); each refuses the
  call without emitting any bits rather than serialise a packet the
  parser would reject. The `codebook` module gains the §9.2.2 encoder
  companion `float32_pack(f32) -> Option<u32>` that inverts the
  existing `float32_unpack`: zero packs to the canonical all-zero
  container, non-finite inputs are rejected, mantissa is canonicalised
  by stripping trailing zero bits, and the function returns `None` for
  values whose 21-bit mantissa or 10-bit biased exponent would
  overflow. The umbrella `WriteError` enum grows a
  `WriteError::Codebook(WriteCodebookError)` variant with the matching
  `From` glue; the `codebook` module now re-exports `float32_pack`,
  `float32_unpack`, `ilog`, and `lookup1_values` so the encoder side
  can reach them without a private path. 35 new tests cover the
  §9.2.2 pack helper (zero, ±1, sign bit, non-finite rejection,
  spread of canonical roundtrip values, unrepresentable-decimal
  rejection, mantissa-overflow rejection, repack idempotence),
  bit-exact roundtrip on nine codebook shapes (dense unordered §3.2.1
  worked example, sparse with unused entries, ordered monotonic,
  lookup-type-2 tessellation, lookup-type-1 lattice, non-trivial
  floats in the lookup table, `value_bits = 16` edge, single-entry
  edge, trace-doc §3 fixture shape), three picker pinning tests
  (sparse forced by unused, monotonic forced to ordered, non-monotonic
  forced to dense unordered), the byte-shape sync-pattern-first
  invariant, every `WriteCodebookError` rejection variant
  (`ZeroEntries`, `LengthTableMismatch`, `IllegalCodewordLength`,
  `IllegalValueBits` at both ends, `MultiplicandOverflow`,
  `TessellationMultiplicandCountMismatch`,
  `LatticeMultiplicandCountMismatch`, `UnrepresentableLookupFloat` on
  both `minimum_value` and `delta_value`), two bit-length
  hand-computed formula checks (dense + sparse), and the
  `WriteError::Codebook` `From` glue with `source()` chaining. The
  module-level lib.rs `Error` enum loses its `Eq` bound (now
  `Clone + PartialEq` only) because
  `WriteCodebookError::UnrepresentableLookupFloat` carries an `f32`;
  no public-API consumers within the crate or downstream depend on
  `Eq` for this type. Test count: **378 total (343 → 378, +35)**.
  Audio-packet WRITE and floor / residue / mapping / mode WRITE
  primitives plus the setup-header splice are explicit followups.

* **Vorbis I header-packet WRITE primitives — first encoder-side
  functions (round 20, umbrella round 195).** New module `encoder`
  exposes `write_identification_header` and `write_comment_header`.
  Each is the byte-exact inverse of its round-1 / round-2 parser
  counterpart and honours the bit-exact roundtrip property
  `parse_(...)_header(&write_(...)_header(&x)?)? == x` for every
  legal input. Both functions validate the same §4.2.2 / §5.2.1
  invariants their parser counterparts enforce on input
  (`vorbis_version == 0`, nonzero channels and sample rate, blocksize
  exponents in 6..=13, `blocksize_0 <= blocksize_1`,
  `u32`-representable vendor / comment lengths and count) and refuse
  the call with a structured `WriteError` (eight variants:
  `UnsupportedVorbisVersion`, `ZeroChannels`, `ZeroSampleRate`,
  `IllegalBlocksize`, `BlocksizesOutOfOrder`, `CommentTooLong`,
  `VendorTooLong`, `TooManyComments`) rather than emit a malformed
  packet. Layout sources: `docs/audio/vorbis/Vorbis_I_spec.pdf`
  §4.2.1 (common header), §4.2.2 (identification layout), §5.2.1 /
  §5.2.3 (comment encoding), §2.1.4 (LSB-first packing collapsing to
  little-endian byte order for octet-aligned fields), §2.1.8 (framing
  byte's seven high bits are zero padding). 31 new unit tests cover
  byte-shape pinning for `mono-44100-q5-typical` and the 63-byte
  one-comment packet shape, bit-exact roundtrip on five canonical
  identification fixtures (mono-44100 q5, 5.1-channel 48000 q5,
  negative bitrate hints, equal-blocksize spec-minimum 64-sample,
  spec-maximum 8192-sample, 255-channel upper edge), an exhaustive
  sweep of every legal `(blocksize_0_exp, blocksize_1_exp)` pair (36
  combinations) confirming the byte-28 nibble pack roundtrips,
  bit-exact roundtrip on seven canonical comment fixtures (typical
  one-comment, seven-comment `with-vorbis-comment-tags` shape, empty
  vendor + zero comments, empty vendor + one comment, multi-byte
  UTF-8 vendor, multi-byte UTF-8 in comments, duplicate keys, 32 KiB
  long payload), comment-ordering preservation, the closed-form
  `7 + 4 + V + 4 + sum(4 + C_i) + 1` byte-length formula across four
  shapes, every `WriteError` rejection variant, the
  `WriteError::Display` non-emptiness smoke test for all eight
  variants, the `std::error::Error::source` leaf check, and the
  `exponent_of_power_of_two` helper on every legal exponent and on
  six non-power-of-two cases. The umbrella `Error` enum grows an
  `Error::Write(WriteError)` variant with `From<WriteError> for Error`
  glue. Test count: **343 total (312 → 343)**. Audio-packet WRITE
  and codebook / floor / residue / mapping / mode WRITE primitives
  are explicit followups for subsequent rounds.

## [0.0.10](https://github.com/OxideAV/oxideav-vorbis/compare/v0.0.9...v0.0.10) - 2026-05-30

### Other

- §4.2.1/§4.3.1 classifier + unified header dispatcher (round 19)
- add multi-channel §4.3.8 PCM driver (round 18)

### Added

* **Vorbis I §4.2.1 / §4.3.1 packet-kind classifier + unified header
  dispatcher (round 19).** New module `packet_kind` exposes
  `classify_packet`, a cheap byte-0 / six-byte-magic inspection that
  resolves a raw Vorbis packet payload to one of the four `PacketKind`
  variants (`Identification` / `Comment` / `Setup` / `Audio`) without
  parsing the body; header packets are recognised by the §4.2.1 common
  header (`0x01` / `0x03` / `0x05` followed by ASCII `"vorbis"`),
  audio packets by the §4.3.1 step-1 `[packet_type]` bit (LSB of byte 0
  == 0). The companion `parse_header_packet` dispatcher classifies and
  then delegates to `parse_identification_header` /
  `parse_comment_header` / `parse_setup_header`, returning the parsed
  result in a `HeaderPacket` sum with `identification()` / `comment()` /
  `setup()` borrow accessors. New error types `ClassifyError` (empty,
  unknown-odd, too-short-for-magic, bad-magic) and
  `HeaderDispatchError` (classify + expected-header-got-audio +
  sub-parser pass-through) are wired into the umbrella
  `Error::Classify` / `Error::HeaderDispatch` variants. 24 new unit
  tests bring the total to **312 (288 → 312)**.

## [0.0.9](https://github.com/OxideAV/oxideav-vorbis/compare/v0.0.8...v0.0.9) - 2026-05-29

### Other

- wire §4.3.7 IMDCT + §4.3.6 window into per-packet driver (round 17)
- add §4.3.7 inverse-MDCT cosine-summation kernel (Vorbis I)
- add §4.3.8 overlap-add primitive (Vorbis I)
- round 14: top-level §4.3 audio-packet driver (§4.3.2 through §4.3.6)
- add §4.3.1 audio-packet prelude reader (Vorbis I)
- add §4.3.3 nonzero-vector propagate + §4.3.6 dot product (Vorbis I)
- add Vorbis window + inverse channel coupling (Vorbis I §1.3.2/§4.3.1 + §4.3.5)
- per-packet decode + LSP curve computation (Vorbis I §6.2.2 + §6.2.3)
- per-packet decode + curve computation (Vorbis I §7.2.3 + §7.2.4)
- add §8.6.2 packet decode + §8.6.3/4/5 format 0/1/2 (round 8)
- add §3.2.1 / §3.3 VQ vector unpack (round 7)
- add §4.2.4 mapping + mode + framing-flag parse (round 6)
- add §4.2.4 setup-header outer walker — codebooks/time/floors/residues (round 5)
- add §3.2.1 canonical Huffman tree builder + entry decoder (round 4)
- add §3.2.1 codebook-header parser (round 3)
- round 2: comment-header parser (Vorbis I §5)
- round 1: identification-header parser (Vorbis I §4.2.2)
- orphan rebuild: clean-room scaffold post 2026-05-20 audit

### Added

* **Vorbis I multi-channel streaming PCM driver (round 18).** New module
  `streaming` exposes `StreamingDecoder` — a per-stream state machine
  holding one `crate::overlap::OverlapAdd` instance per channel — that
  stitches the round-17 `decode_audio_packet_windowed` per-packet driver
  to the §4.3.8 overlap-add primitive across consecutive packets. The
  decoder is constructed from the identification-header fields
  (`audio_channels` / `blocksize_0` / `blocksize_1`) plus the deferred
  `imdct_scale`, then driven one packet at a time via
  `push_packet(reader, setup, state)` (or `push_windowed(outcome)` for
  callers that already hold a `WindowedPacketOutcome`). The first packet
  primes every per-channel overlap-add state (§4.3.8 priming step) and
  returns `StreamingFrame::Primed`; from the second packet on the
  engine emits `StreamingFrame::Pcm { mode_number, blockflag, n,
  per_channel_pcm }` carrying the `prev_n / 4 + cur_n / 4` finished PCM
  samples per channel (in bitstream channel order — §4.3.9
  rearrangement is left to the consumer). `reset()` returns every
  per-channel state to priming (e.g. after a seek); `finish()` drains
  the last frame's right-half tail (`n / 2` samples per channel) for
  callers flushing a finite encoded clip. New error type
  `StreamingError` with three variants (`Packet(AudioPacketError)` /
  `Overlap { channel, source: OverlapError }` / `ChannelCountMismatch
  { expected, got }`) wires the underlying failures up; umbrella
  `Error::Streaming(StreamingError)` + `From<StreamingError> for Error`
  surface them at the top level. 14 new unit tests cover the
  construction accessors, the §4.3.8 priming step, the spec-formula
  return length (`prev_n / 4 + cur_n / 4`) for equal-sized and
  mixed-sized block transitions, multi-channel routing with a
  channel-ratio invariant on synthetic per-channel ramps, the
  `ChannelCountMismatch` defensive check, the `ZeroedWindowed` packet
  propagation (preserves geometry through priming, emits per-channel
  zero PCM with the previous-frame plateau preserved on a zeroed-after-
  normal transition), `reset()` returning to priming, `finish()`
  emitting per-channel right-half tails (or `None` on an unprimed
  engine), and the `StreamingError::Display` strings for both the
  channel-count-mismatch variant and the per-channel overlap-add
  failure. Per-packet driver failures surface verbatim via
  `StreamingError::Packet`. With this round the entire §4.3 pipeline
  from a parsed audio-packet bitstream to PCM is reachable end-to-end
  as a single `StreamingDecoder::push_packet` call per packet — the
  last composition step the round-17 wiring named.
* **Vorbis I §4.3.7 IMDCT + §4.3.6 windowing wired into the per-packet
  driver (round 17).** New entry points `decode_audio_packet_windowed`
  (drives §4.3.2..§4.3.6 via `decode_audio_packet_pre_imdct`, runs the
  §4.3.7 `imdct::imdct_naive` cosine-summation kernel per channel, and
  element-wise multiplies by the §4.3.6 / §1.3.2 Vorbis window built
  once per packet via `AudioPacketHeader::build_window`) and the
  convenience `decode_one_packet_windowed` that returns per-channel
  length-`n` windowed time-domain frames ready to feed straight into
  per-channel `crate::overlap::OverlapAdd::push_frame` instances. New
  pure transform `apply_imdct_and_window(outcome, blocksize_0,
  imdct_scale)` for callers that already hold a parsed
  `AudioPacketOutcome` (e.g. from a buffered decode) and only need the
  §4.3.7-then-§4.3.6 stage. New outcome enum `WindowedPacketOutcome`
  with two variants (`Windowed` for a normal frame, `ZeroedWindowed`
  for the §4.3.2 short-circuit) plus `header()` and `frames()`
  accessors. The `imdct_scale: f32` argument is the deferred-
  normalization knob — the Vorbis-specific IMDCT normalization scalar
  the cross-reference notes "falls out of matching the fixture traces"
  is still pinned to caller-supplied; passing `1.0` returns the bare
  un-normalized kernel output × window. Two new error variants on
  `AudioPacketError`: `Window(WindowError)` for a window-builder
  rejection and `Imdct(ImdctError)` for an IMDCT-kernel rejection;
  both surface verbatim with `§4.3.6` / `§4.3.7` prefixes in their
  `Display`. The legacy `decode_one_packet` entry point still returns
  `AudioPacketError::ImdctStage` so callers depending on the pre-IMDCT
  stop remain unbroken. 11 new tests cover: windowed driver on the
  trivial mono packet (one length-`n` frame per channel; geometry
  pinned), `apply_imdct_and_window` round-trip on a hand-built outcome
  with a long-block window (lead-in / tail regions are exactly zero by
  window-edge construction), the §4.3.2 short-circuit (`ZeroedWindowed`
  returns per-channel all-zero length-`n` frames), the `imdct_scale`
  linearity property (scaling by α scales every output sample by α),
  the IMDCT-then-window composition matching the direct
  `imdct_naive_vec` × `vorbis_window` path bit-for-bit, end-to-end
  integration with `OverlapAdd::push_frame` (first call primes,
  second emits the §4.3.8 finished-PCM range), legacy
  `decode_one_packet` ImdctStage preservation, `decode_one_packet_windowed`
  parity with `decode_audio_packet_windowed`, `header()` / `frames()`
  accessor checks, and `AudioPacketError::Window` / `Imdct` Display
  strings. Test count: 274 total (263 → 274).

* **Vorbis I §4.3.7 inverse MDCT — direct cosine-summation kernel.**
  New module `imdct` exporting `imdct_naive(spectrum, output, scale)`,
  `imdct_naive_vec(spectrum, scale)`, and the `ImdctError` enum
  (`SpectrumNotPowerOfTwo`, `OutputLenMismatch`). The kernel implements
  the bare cosine summation
  `x[n] = sum_k X[k] · cos[ (π/N) · (2n + 1 + N/2) · (2k + 1) / 2 ]`
  from `docs/audio/vorbis/imdct-cross-reference.md` — the OxideAV
  clean-room companion that closes Vorbis I §4.3.7's deferral to
  external reference `[1]` by observing that the IMDCT mathematical
  kernel is restated in three other adjacent in-repo specs (ATSC A/52
  §7.9.4, ISO/IEC 14496-3 §4.6.x, IETF RFC 6716 §4.3.7). The
  implementation is the O(N²) direct form, deliberately chosen as the
  *reference* implementation that is provably correct by inspection
  against the cosine summation; an FFT-decomposed fast path can land in
  a later round and validate against this kernel's output. Working
  precision is `f64` accumulators with `f32` output to match the
  spectral pipeline. The `scale` argument is a pure output multiplier
  — the Vorbis-specific normalization scalar that maps the bare
  kernel to oggdec-bit-equivalent PCM is deliberately **not** pinned
  in this round (see Followups). Crate root re-exports `imdct_naive`,
  `imdct_naive_vec`, `ImdctError`; the unified `Error` grows an
  `Imdct(ImdctError)` variant with `From` glue. 12 new unit tests
  cover the error paths (empty / non-power-of-two spectrum, mismatched
  output length, vec-wrapper rejection), mathematical properties
  derivable directly from the cosine summation (zero input → zero
  output; linearity in the spectrum; left-half anti-symmetry
  `x[i] + x[N/2 - 1 - i] = 0`; right-half symmetry
  `x[N/2 + i] = x[N - 1 - i]` — the two TDAC half-rules that overlap-add
  cancellation builds on), the `scale` argument's linearity property,
  two hand-computed N = 4 fixtures (impulse on DC bin, impulse on
  k = 1 bin — these pin the exact cosine arguments and would catch
  any off-by-one in the `(2n + 1 + N/2) · (2k + 1) / 2` form), and the
  `Display` strings. Test count: 263 total (251 → 263). The top-level
  `decode_packet` still returns at the §4.3.7 boundary because wiring
  the IMDCT into the per-packet driver needs the normalization factor
  to be pinned first — a future-round task once fixture traces extend
  through the post-IMDCT trace point.

* **Vorbis I §4.3.8 overlap-add primitive (Vorbis I §4.3.8 "overlap
  add").** New module `overlap` exporting `OverlapAdd` (a one-channel
  stateful overlap-add engine) and `OverlapError`. `OverlapAdd::new`
  creates a priming state; `OverlapAdd::push_frame(windowed)` consumes
  the next windowed time-domain frame and returns `Ok(None)` on the
  first call (the spec's "data is not returned from the first frame"
  priming step) and `Ok(Some(samples))` on every subsequent call,
  emitting the §4.3.8 return range of
  `previous_window_blocksize / 4 + current_window_blocksize / 4`
  samples spanning the previous-window center (`windowsize / 2`) to
  the current-window center (`windowsize / 2 - 1`, inclusive). The
  3/4-vs-1/4 alignment geometry (`cur_global_start = prev_n * 3 / 4 -
  cur_n / 4`) is reproduced from §4.3.8 verbatim, with signed-isize
  arithmetic so the short→long case (where the current frame begins
  BEFORE the previous-window center) is handled exactly as the spec
  text "much of the returned range does not actually overlap"
  describes. Per-frame state is the previous-frame right half
  (length `prev_n / 2`), so memory is proportional to block size
  rather than full frame. `OverlapAdd::reset` returns to priming;
  `OverlapAdd::finish` drains the stored tail for stream-end finishing
  (the symmetric counterpart of the first-frame priming). Helper
  accessors `is_priming`, `stored_tail_len`, and `next_output_len`
  expose the §4.3.8 return-length formula without requiring a push.
  New error enum `OverlapError` (`NotPowerOfTwo`, `FrameTooSmall`)
  guards against malformed frame lengths (defensive — §4.2.2 already
  pins `blocksize >= 64`); `From<OverlapError> for WindowError` maps
  into the umbrella window-error type. Crate root re-exports
  `OverlapAdd`, `OverlapError`; the unified `Error` grows an
  `Overlap(OverlapError)` variant with `From` glue. 20 new unit tests
  cover both error paths, the priming → output → reset cycle,
  `next_output_len` against the spec formula, the equal-sized case
  (one-half block return, fully overlapped), the long→short and
  short→long mixed-size cases (return-length formulas + which range
  carries each frame's contribution), the §1.3.2 squared-window
  perfect-reconstruction property (`w[i]² + w[i + n/2]² == 1`),
  a three-frame chained sequence, the finish/priming corner cases,
  and the spec's "from element windowsize/2 of previous to element
  windowsize/2 - 1 of current" return-range indexing invariant.
  Test count: 251 total (231 → 251). With this round every §4.3.x
  stage the Vorbis I spec body defines in its own text is implemented
  as a standalone primitive; the only remaining gap is the §4.3.7
  inverse MDCT itself, still gated on the spec's externally-cited
  reference `[1]`.

* **Vorbis I top-level §4.3 audio-packet driver (§4.3.2 through §4.3.6),
  stopping cleanly at the §4.3.7 inverse-MDCT boundary.** New module
  `audio` exporting `decode_audio_packet_pre_imdct`, `decode_one_packet`,
  `AudioDecoderState`, `AudioPacketOutcome`, `AudioPacketError`. The
  driver walks every §4.3 stage the spec defines in its own text:
  §4.3.1 packet header (via `read_packet_header`), §4.3.2 floor decode
  in channel order with `mapping.mux[ch]` (or implicit submap-0 when
  `submaps == 1`) routing per channel to its submap's floor, §4.3.3
  nonzero propagate, §4.3.4 residue decode in submap order with a
  channels-in-submap gather + `do_not_decode_flag` build + post-decode
  scatter, §4.3.5 inverse coupling (descending step order), and §4.3.6
  dot product. Returns `AudioPacketOutcome::PreImdct { mode_number,
  blockflag, n, previous_window_flag, next_window_flag, spectra }` with
  one length-`n/2` audio spectrum per channel, ready for the (still
  pending) §4.3.7 IMDCT. `AudioDecoderState::new(setup)` builds the
  per-stream cache of `Floor0Decoder` / `Floor1Decoder` / `ResidueDecoder`
  decoders once up front (so per-packet decode is allocation-light) and
  surfaces per-floor / per-residue construction errors as
  `AudioPacketError::Floor0Build { index, source }` /
  `Floor1Build { index, source }` / `ResidueBuild { index, source }`.
  Top-level `decode_one_packet` (and the umbrella `decode_packet`)
  return `AudioPacketError::ImdctStage` cleanly at the IMDCT docs-gap
  boundary; the variant's `Display` impl spells out the docs-gap
  reason. Defensive `AudioPacketError::BadModeMapping` /
  `BadSubmapIndex` / `BadSubmapFloor` / `BadSubmapResidue` /
  `MuxOutOfRange` variants guard against hand-built or corrupted
  setup state (the setup parser already range-checks these). 12 new
  unit tests: cache-build success, Floor1 build error propagation,
  Residue build error propagation, mono used-floor packet, mono
  unused-floor short-circuit, stereo-coupled used packet,
  decode_one_packet IMDCT-stop, ImdctStage Display contains docs-gap
  text, non-audio header reject, single-submap `submap_for_channel`
  always 0, multi-submap mux indexing + OOB, and a trivial-classbook
  Huffman build sanity check. Crate root re-exports
  `decode_audio_packet_pre_imdct`, `decode_one_packet`,
  `AudioDecoderState`, `AudioPacketOutcome`, `AudioPacketError`; the
  umbrella `Error` enum gains `Error::AudioPacket(AudioPacketError)`
  with `From<AudioPacketError>`. `decode_packet`'s signature now takes
  `(packet, setup, state, audio_channels, blocksize_0, blocksize_1)`
  and returns the IMDCT-stop error variant. Test count: 231 total (219
  → 231).

* **Vorbis I audio-packet §4.3.1 "packet type, mode and window decode".**
  New `packet::read_packet_header(&mut BitReaderLsb, &VorbisSetupHeader,
  blocksize_0, blocksize_1) -> Result<AudioPacketHeader, PacketError>` reads
  the fixed-shape audio-packet prelude: (1) 1-bit `[packet_type]` with the
  §4.3 "must ignore" reject path for `packet_type != 0`; (2) the
  `ilog([vorbis_mode_count] - 1)`-bit `[mode_number]` with OOB validation
  (§9.2.1 `ilog` returns `0` for `mode_count == 1` → zero mode bits, the
  single-mode degenerate case); (3) blocksize resolution `n = blockflag ?
  blocksize_1 : blocksize_0` (§4.3.1 step 3); (4) long-block-only
  `[previous_window_flag]` + `[next_window_flag]` (step 4a.i/ii) — short
  blocks always reuse the symmetric short shape (step 4b) and do NOT
  transmit these bits. `AudioPacketHeader::build_window(blocksize_0)`
  drives the existing `synthesis::vorbis_window` builder with the resolved
  fields. End-of-packet anywhere in §4.3.1 is fatal (§4.3.1 closing note:
  "should be considered an error that discards this packet"), surfaced as
  `PacketError::UnexpectedEndOfPacket { stage: PacketHeaderStage }` with
  per-sub-step granularity (`PacketType`, `ModeNumber`,
  `PreviousWindowFlag`, `NextWindowFlag`). Additional `PacketError` variants
  `NonAudioPacketType { packet_type }`, `BadModeNumber { mode_number,
  mode_count }`, `EmptyModeList`. Crate root re-exports
  `read_packet_header`, `AudioPacketHeader`, `PacketHeaderStage`. 16 new
  unit tests cover: single-mode short block (zero mode bits, 1 bit total),
  single-mode long block (zero mode bits + 2 window flags), two-mode
  one-bit `mode_number` short and long paths, three-mode two-bit
  `mode_number` long path, non-audio packet_type reject, out-of-range
  mode_number reject, empty-mode-list defensive guard, EOF on packet_type
  (empty stream), EOF on mode_number (130 modes → 8-bit read past 7
  remaining bits), EOF on previous_window_flag (65 modes → 7-bit read
  consumes the byte), EOF on next_window_flag (33 modes → 6-bit read
  leaves 1 bit), `build_window` short-block matches direct
  `vorbis_window` call, `build_window` long-block hybrid-left matches +
  confirms zero lead-in, `WindowError` propagation through
  `build_window` (non-power-of-two `n`), and the mode-blockflag-driven
  blocksize selection across mode 0 short / mode 1 long. Test count: 219
  total (203 → 219).

* **Vorbis I audio-packet driver stages §4.3.3 "nonzero vector propagate"
  and §4.3.6 "dot product".** New module `packet` exporting
  `nonzero_propagate`, `dot_product`, `dot_product_all`, `PacketError`,
  and `VectorKind`. `nonzero_propagate(&mut [bool], &[MappingCouplingStep])`
  runs the §4.3.3 ascending coupling-step loop that pulls an `'unused'`
  channel back into the active set when its coupling partner is used,
  preventing coupling from silently zeroing a half-pair. `dot_product(&[f32],
  &[f32], &mut [f32])` is the bare §4.3.6 element-wise floor × residue
  product into a caller-provided spectrum buffer; `dot_product_all(&[Option<Vec<f32>>],
  &[Vec<f32>], n/2)` is the per-channel driver that emits the all-zero
  spectrum for an unused channel (per §4.3.3 "that final output vector is
  all-zero values") and the element-wise product for every used channel.
  These two stages are the fully-specified, IMDCT-independent halves of
  the §4.3 audio-packet pipeline; the §4.3.7 inverse MDCT remains pending
  on a documented docs gap (Vorbis I spec §4.3.7 defers the MDCT
  definition entirely to external reference `[1]`, T. Sporer / K.
  Brandenburg / B. Edler, which the workspace clean-room policy bars).
* 19 new unit tests cover §4.3.3 (no-coupling no-op, magnitude-pulls-in,
  angle-pulls-in, both-unused-stays-unused, both-used-stays-used, chained
  ascending cascade, isolated unused channel survives, two
  out-of-range-channel rejections) and §4.3.6 (element-wise product,
  signed operands, two length-mismatch panic cases, two-channel
  per-driver case, unused-channel zero short-circuit, all-unused case,
  channel-count mismatch, short floor curve rejection, short residue
  vector rejection).

* **Vorbis I audio-packet synthesis primitives: the Vorbis window
  (§1.3.2 / §4.3.1 "packet type, mode and window decode") and inverse
  channel coupling (§4.3.5 "inverse coupling").** New module `synthesis`
  exporting `vorbis_window`, `slope`, `WindowError`, `couple_scalar`,
  `inverse_couple`, `inverse_couple_all`, and `CouplingError`.
  `slope(x, n)` is the bare Vorbis slope function
  `y = sin(π/2·sin²((x+0.5)/n·π))` (§1.3.2). `vorbis_window(n,
  blocksize_0, blockflag, previous_window_flag, next_window_flag)`
  builds the length-`n` window per the §4.3.1 eight-step generation
  procedure: a zero lead-in, a rising edge using the per-edge `…·π/2`
  argument over `left_n`, a plateau of ones, a `+π/2`-phase-shifted
  falling edge over `right_n`, and a zero tail — with the
  `n/4 ± blocksize_0/4` hybrid ramps selected when a long block laps a
  short neighbour (`blockflag` set and the matching window flag clear),
  and the full-half-block ramps otherwise. Short blocks (`blockflag`
  clear) always get the plain symmetric shape, ignoring the flags
  (§4.3.1 step 4b). `inverse_couple_all(&mut [Vec<f32>],
  &[MappingCouplingStep])` runs the §4.3.5 inverse-coupling loop over a
  residue-vector bundle **in descending coupling-step order**,
  decoupling each `(magnitude, angle)` pair in place via `couple_scalar`
  (the four-quadrant square-polar → Cartesian rule of §4.3.5 step 3).
  New error enums `WindowError` (`NotPowerOfTwo`, `ShortBlockTooLarge`)
  and `CouplingError` (`ChannelOutOfRange`, `SameChannel`), both wired
  into the crate `Error` enum via `From`. 20 new tests cover the slope
  endpoints/symmetry, the short/long/hybrid window shapes with their
  squared-overlap unity-power reconstruction property, the long+short
  adjacent-window lapping, the four coupling quadrants plus the
  zero-magnitude else branch, in-place pairwise decoupling, the
  descending-order loop observability, the magnitude-below-angle index
  branch, and the out-of-range / same-channel / empty-list driver
  cases.

* **Vorbis I floor type 0 per-packet decode + LSP curve computation
  (Vorbis I §6.2.2 "packet decode" + §6.2.3 "curve computation").**
  New module `floor0` exporting `Floor0Decoder`, `Floor0Curve`,
  `Floor0Error`, and the §6.2.3 post-errata Bark scale helper `bark`.
  `Floor0Decoder::new(&Floor0Header, &[VorbisCodebook])` validates the
  §6.2.1 / §6.2.3 undecodability clauses (nonzero `order` /
  `bark_map_size` / `amplitude_bits`, non-empty `book_list`, every book
  index in range, every referenced book carries a VQ lookup table per
  §3.3) and pre-builds each value codebook's Huffman decision tree once.
  `Floor0Decoder::decode(&mut BitReaderLsb, n)` runs the two-stage
  decode: (1) §6.2.2 packet decode reads `[amplitude]` in
  `floor0_amplitude_bits` bits (returning `Floor0Curve::Unused` if
  zero), then `[booknumber]` in `ilog([floor0_number_of_books])` bits
  (mapping `booknumber >= floor0_book_list.len()` reserved values to
  `Unused` per the nominal-occurrence rule), then loops reading VQ
  vectors from `floor0_book_list[booknumber]` and concatenating them
  into `[coefficients]` until the vector reaches `floor0_order`
  elements, with the running `[last]` accumulator added before
  concatenation (§6.2.2 steps 6..9) and reset to the last scalar of
  each just-decoded vector (the spec explicitly permits over-reading
  past `floor0_order`); (2) §6.2.3 curve computation builds a
  Bark-scale `map[i]` per the post-errata `bark(x) = 13.1·atan(.00074x)
  + 2.24·atan(.0000000185x²) + .0001x` formula, then synthesises the
  LSP curve via the order-parity `[p]`/`[q]` product at every
  `[ω] = π·map[i]/bark_map_size` and applies the
  `exp(.11512925·(amplitude·offset/((2^bits - 1)·sqrt(p+q)) - offset))`
  log→linear transform, replicating each synthesis value across all
  consecutive output bins whose `map[i]` matches the current synthesis
  bin (§6.2.3 step 8 `[iteration_condition]` chaining). An
  end-of-packet condition anywhere in §6.2.2 is the spec's nominal
  occurrence: decode returns `Floor0Curve::Unused`. New error enum
  `Floor0Error` (`BookOutOfRange`, `EmptyBookList`, `ZeroOrder`,
  `ZeroBarkMapSize`, `ZeroAmplitudeBits`, `ValueBookHasNoLookup`,
  `Huffman`). Crate root re-exports `Floor0Decoder`, `Floor0Curve`,
  `Floor0Error`, and `bark` as `floor0_bark`; the unified `Error`
  grows a `Floor0(Floor0Error)` variant with `From` glue. 18 new unit
  tests cover the §6.2.3 post-errata Bark formula at zero plus its
  monotonicity on the audible range, all six construction-validation
  rejections, the `amplitude == 0` and EOF-during-amplitude `Unused`
  paths, an EOF-during-coefficients `Unused` path, a hand-traced packet
  decode producing the expected `(amplitude, coefficients)` pair with
  `[last]` accumulating across vectors, a dim-1-book multi-codeword
  fill exercising the `[last]` carry across every iteration, a curve-
  computation length-and-finiteness check on a hand-picked LSP set, the
  `[iteration_condition]` chaining replication assertion, an
  `amplitude = 0` direct `curve_computation` returning the all-zero
  length-`n` vector, the reserved-`booknumber` → `Unused` path, and a
  full packet → `Floor0Curve` end-to-end round trip. Test count: 164
  total (146 → 164). No real-world encoder emits floor 0 (production
  encoders use floor 1 exclusively) but a conformant Vorbis I decoder
  must implement the codepath because the spec defines it; this is the
  missing-piece half of the floor decode pipeline (round 9 covered
  floor 1).

* **Vorbis I floor type 1 per-packet decode + curve computation (Vorbis
  I §7.2.3 "packet decode" + §7.2.4 "curve computation").** New module
  `floor1` exporting `Floor1Decoder`, `FloorCurve`, `Floor1Error`, and
  the integer geometry helpers `low_neighbor` / `high_neighbor` (§9.2.4 /
  §9.2.5), `render_point` (§9.2.6) and `render_line` (§9.2.7).
  `Floor1Decoder::new(&Floor1Header, &[VorbisCodebook])` reconstructs the
  full `[floor1_X_list]` (prepending the two implicit endpoints `0` and
  `2^rangebits` that §7.2.2 injects before the per-partition read loop),
  validates the §7.2.2 undecodability clauses (multiplier in `1..=4`,
  `[floor1_values]` ≤ 65, x-list uniqueness, master/sub-book indices in
  range) and pre-builds every referenced codebook's Huffman tree once.
  `Floor1Decoder::decode(&mut BitReaderLsb, n)` runs the two-stage
  decode: (1) §7.2.3 packet decode reads the `[nonzero]` flag (returning
  `FloorCurve::Unused` if clear), the two `ilog([range]-1)`-bit endpoint
  amplitudes, and per partition the master-book selector (only when
  `[cbits] > 0`) followed by the per-dimension sub-book scalar
  amplitudes (a negative/`None` sub-book yields a `0` Y with no bits
  read), producing `[floor1_Y]`; (2) §7.2.4 step 1 unwraps the positive
  `[floor1_Y]` differences through iterative `render_point` line
  prediction into `[floor1_final_Y]` + `[floor1_step2_flag]` (with the
  suggested `[0, range)` clamp), and step 2 sorts the `(X, final_Y,
  flag)` triples by ascending X, renders the contiguous integer line
  segments via `render_line`, and substitutes each integer floor sample
  through the §10.1 `floor1_inverse_dB_table` (transcribed verbatim as
  the 256-element `INVERSE_DB_TABLE`) to produce a linear-domain
  envelope of length `n`. An end-of-packet condition anywhere in §7.2.3
  is the spec's nominal occurrence: decode returns `FloorCurve::Unused`
  exactly as if `[nonzero]` had been clear. New error enum `Floor1Error`
  (`BookOutOfRange`, `BadMultiplier`, `TooManyValues`, `NonUniqueXList`,
  `Huffman`). Crate root re-exports `Floor1Decoder`, `FloorCurve`,
  `Floor1Error`, `low_neighbor`, `high_neighbor`, `render_point`,
  `render_line`; the unified `Error` grows a `Floor1(Floor1Error)`
  variant with `From` glue. 18 new unit tests cover the §9.2.6/§9.2.7
  render-point/render-line spec geometry (up/down/flat segments,
  endpoint-not-written convention), the §9.2.4/§9.2.5 neighbor lookups,
  all four §7.2.2 construction-validation rejections, a hand-traced full
  `curve_computation` (x_list `[0,16,4,8]`, Y `[40,20,1,0]` → final_Y
  `[40,20,34,30]`), a full packet→curve round trip, the master/subclass
  cascade selector path, the `nonzero` unset path, two end-of-packet
  nominal-`Unused` paths (mid-amplitude + mid-codeword), the
  negative-sub-book zero-Y path, and the §10.1 table endpoints. Test
  count: 146 total (128 → 146).

* **Vorbis I per-packet residue decode (Vorbis I §8.6.2 "packet decode"
  + §8.6.3/§8.6.4/§8.6.5 "format 0/1/2 specifics").** New module
  `residue` exporting `ResidueDecoder`. `ResidueDecoder::new(&ResidueHeader,
  &[VorbisCodebook])` validates the §8.6.1 undecodability clauses
  (classbook / value-book indices in range, value books carry a value
  mapping, classbook dimensions nonzero) and pre-builds the classbook +
  value-book Huffman trees once. `ResidueDecoder::decode(&mut
  BitReaderLsb, blocksize, &[bool])` runs the §8.6.2 packet decode: it
  caps `[residue_begin]`/`[residue_end]` to the per-format vector size
  (`blocksize/2` for format 0/1, `blocksize/2 * ch` for format 2),
  derives `classwords_per_codeword`/`n_to_read`/`partitions_to_read`,
  reads the per-partition classifications on pass 0 from the classbook in
  scalar context (descending integer-divide / integer-modulo unpack),
  and over passes 0..=7 decodes each partition's stage-`pass` value book
  in VQ context — *accumulating* (`+=`) into the output. Format 0
  (§8.6.3) scatters element `j` to `offset + i + j*step`; format 1
  (§8.6.4) appends contiguously; format 2 (§8.6.5) decodes one
  interleaved vector of length `ch * blocksize/2` as a format-1 decode
  then de-interleaves `v[i*ch + j] -> output[j][i]`, with the all-`do
  not decode` short-circuit. End-of-packet during decode is treated as
  the §8.6.2 nominal occurrence: decode stops and returns the
  vectors-so-far rather than erroring. New error enum `ResidueError`
  (`UnsupportedFormat`, `ClassbookOutOfRange`, `ValueBookOutOfRange`,
  `ValueBookHasNoLookup`, `ZeroClasswordsPerCodeword`,
  `Format0PartitionNotDivisible`, `Huffman`, `Vq`). Crate root re-exports
  `ResidueDecoder` + `ResidueError`; the unified `Error` grows a
  `Residue(ResidueError)` variant with `From` glue. 15 new unit tests
  cover all four §8.6.1 construction-validation rejections, the
  `n_to_read == 0` short-circuit, format-1 mono two-partition decode,
  `do not decode` zeroing/skip, format-0 interleaved scatter, format-2
  interleave/de-interleave + the all-`do not decode` no-op, two
  end-of-packet nominal-occurrence paths (empty buffer + mid-codeword
  overrun), the multi-classword descending classification unpack, and
  cascade-stage accumulation across passes. Test count: 128 total (113 →
  128).

* **Vorbis I VQ vector unpack (Vorbis I §3.2.1 "VQ lookup table vector
  representation" + §3.3 "Use of the codebook abstraction").** New
  module `vq` exporting `unpack_vector(&VorbisCodebook, lookup_offset:
  u32) -> Result<Vec<f32>, VqUnpackError>`. Implements the §3.2.1
  "Vector value decode: Lookup type 1" mixed-base permutation
  (`multiplicand_offset = (lookup_offset / index_divisor) mod
  codebook_lookup_values`, with `index_divisor` multiplied by
  `codebook_lookup_values` between iterations) and the §3.2.1 "Vector
  value decode: Lookup type 2" one-to-one direct slice
  (`multiplicand_offset = lookup_offset * codebook_dimensions`,
  incremented per iteration). Both paths honour `codebook_sequence_p`
  cumulatively: when set, `[last]` carries the full prior
  `value_vector[i]` (post-`minimum_value`, post-`delta_value`,
  post-`last`) forward, making the output a prefix-sum; when clear,
  `[last]` stays at `0.0` and each element is independent. New error
  enum `VqUnpackError` with variants `EntryOutOfRange`,
  `NoVectorForType0` (per §3.3 "requesting decode using a codebook of
  lookup type 0 in any context expecting a vector return value … is
  forbidden"), `ZeroDimensions`, and `MultiplicandShapeMismatch`. Crate
  root re-exports `unpack_vector` and `VqUnpackError`; the unified
  `Error` grows a `Vq(VqUnpackError)` variant with `From` glue. 16 new
  unit tests cover both lookup-type paths, both `sequence_p` modes,
  per-codebook §3.3 round-trips (an 8-entry / 8-dim tessellation and a
  9-entry / 2-dim lattice exhaustively check every entry against
  hand-computed reference vectors), the type-0 vector-rejection path,
  out-of-range lookup-offset rejection, zero-dimensions rejection, and
  multiplicand-shape mismatch detection for both lattice and
  tessellation. Test count: 113 total (97 → 113).

* **Vorbis I setup-header mapping + mode + framing-flag parse (Vorbis I
  §4.2.4 "Mappings" / "Modes").** `parse_setup_header` /
  `parse_setup_header_body` now walk the entire setup header, not just
  the codebook/time/floor/residue prefix. Both entry points now take an
  `audio_channels: u8` parameter (the value carried in the
  identification header per §4.2.2), required for the
  `ilog(audio_channels - 1)`-bit magnitude/angle channel reads in the
  square-polar coupling subblock and for the per-channel `mux[ch]`
  reads when `submaps > 1`. New types: `MappingHeader { mapping_type,
  submaps, coupling: Vec<MappingCouplingStep>, mux: Vec<u8>,
  submap_configs: Vec<MappingSubmap> }`, `MappingCouplingStep
  { magnitude_channel, angle_channel }`, `MappingSubmap
  { time_placeholder, floor, residue }`, `ModeHeader { blockflag,
  windowtype, transformtype, mapping }`. `VorbisSetupHeader` grows
  three fields: `mappings`, `modes`, `framing_flag`. New
  `setup::ParseError` variants enforcing the §4.2.4 invariants:
  `UnsupportedMappingType` (`mapping_type != 0` per step 2b),
  `BadCouplingChannels` (magnitude == angle, or either >=
  audio_channels — step 2c.ii), `NonZeroMappingReserved` (2-bit
  reserved — step 2c.iii), `BadMuxValue` (`mux[ch] >= submaps` —
  step 2c.iv.B), `BadSubmapFloor` (floor index >= `floor_count` —
  step 2c.v.C), `BadSubmapResidue` (residue index >= `residue_count` —
  step 2c.v.E), `NonZeroModeWindowType` / `NonZeroModeTransformType` /
  `BadModeMapping` (mode-section step 2e), `BadFramingFlag` (trailing
  flag unset — "Modes" step 3), `ZeroAudioChannels` (caller-side
  guard — §4.2.2 already guarantees > 0). Re-exports at the crate
  root: `MappingHeader`, `MappingCouplingStep`, `MappingSubmap`,
  `ModeHeader`. 14 new unit tests cover the trace-doc §6 / §7 shapes:
  stereo mapping with one coupling step (magnitude=0, angle=1), 5.1
  multi-submap mapping with `mux=[0,0,0,0,0,1]` and a per-submap
  (floor 0, residue 0) / (floor 1, residue 1) split, two
  one-mode-per-block setups, the §4.2.4 "Modes" `windowtype != 0` /
  `transformtype != 0` / OOB-`mapping` rejections, the
  `audio_channels = 0` caller-bug guard, and the unset-framing-flag
  reject (97 tests total, up from 83).

* **Vorbis I setup-header outer walker (Vorbis I §4.2.4).**
  `parse_setup_header(&[u8])` validates the 7-byte common header
  (`0x05 "vorbis"` per §4.2.1) and `parse_setup_header_body(&mut
  BitReaderLsb)` consumes the bit-packed body that follows. The
  round-5 walker reads the first four sub-lists of the setup header:
  (1) `vorbis_codebook_count` codebook configurations via
  `parse_codebook` (§3.2.1); (2) `vorbis_time_count` 16-bit
  time-domain transform placeholders, each spec-mandated to equal zero
  (§4.2.4 step 2); (3) `vorbis_floor_count` floor headers, each
  prefixed by a 16-bit `floor_type` that dispatches to the §6.2.1
  (LSP, type 0) or §7.2.2 (piecewise-linear, type 1) structural-field
  reader — no per-packet curve decode; (4) `vorbis_residue_count`
  residue headers (§8.6.1 common layout: `residue_begin`,
  `residue_end`, `residue_partition_size`, `residue_classifications`,
  `residue_classbook`, the per-classification cascade bitmap with the
  `low_bits / bitflag / high_bits` encoding, and the conditional
  per-stage book reads). Mappings (§4.3), modes (§4.3.1), and the
  trailing framing flag are deferred to round 6; the bit reader is
  left positioned immediately after the last residue's book list.
  Spec-mandated errors surface as `setup::ParseError` variants
  (`PacketTooShort`, `WrongPacketType`, `BadMagic`, `Codebook` —
  wrapping the inner codebook parse error — `NonZeroTimePlaceholder`,
  `UnsupportedFloorType`, `UnsupportedResidueType`,
  `UnexpectedEndOfPacket`). Re-exports at the crate root:
  `parse_setup_header`, `parse_setup_header_body`,
  `VorbisSetupHeader`, `FloorHeader`, `FloorKind`, `Floor0Header`,
  `Floor1Header`, `Floor1Class`, `ResidueHeader`, `SetupParseError`,
  `SETUP_PACKET_TYPE`, `SETUP_PACKET_MAGIC`. 14 new unit tests cover
  the minimal one-of-each setup body, a full setup packet with the
  7-byte common header prefix, packet-length / packet-type / magic
  rejections, nonzero time-placeholder rejection, floor-type > 1
  rejection, residue-type > 2 rejection, truncated-packet EOF
  surfaced through `Codebook` wrapping, floor 1 with `partitions = 0`
  (empty-classes corner case), floor 0 (LSP) setup, two-floor /
  two-residue stereo shape mirroring trace-doc §2.4, multi-stage
  residue cascade exercising `bitflag = 1` high bits, and residue
  type 0 (83 tests total, up from 69).

* **Vorbis I canonical Huffman tree builder + entry decoder (Vorbis I
  §3.2.1).** `HuffmanTree::from_codebook(&VorbisCodebook)` /
  `HuffmanTree::from_lengths(&[u8])` builds a canonical Vorbis decision
  tree from the per-entry codeword-length table returned by
  `parse_codebook`. Construction uses an open-position deque
  (left-to-right) that pops the leftmost slot, splits it down to the
  required depth, and places a leaf — realising the §3.2.1 "lowest
  valued unused canonical codeword" assignment without ever
  materialising codeword integers. The spec's worked example (lengths
  `[2 4 4 4 4 2 3 3]` → codewords `00 0100 0101 0110 0111 10 110 111`)
  round-trips bit-exactly. `HuffmanTree::decode_entry(&mut
  BitReaderLsb)` walks the tree against an LSb-first packet bitstream
  (first bit = MSb of canonical codeword) and returns the codebook
  entry index — the underlying primitive for §3.3 scalar codebook reads
  and §3.4 / §8.6 VQ codebook reads. Errata 20150226 single-entry
  codebooks: a codebook with exactly one used entry whose recorded
  length is `1` yields a 2-node tree (`root.left == root.right == sole
  leaf`) so `decode_entry` returns the sole entry and sinks one bit
  for either `0` or `1` (per the errata "decoders should tolerate that
  the bit read from the stream be '1' instead of '0'"); single-entry
  codebooks with `length != 1` are rejected. Underspecified /
  overspecified trees (§3.2.1) surface as
  `HuffmanBuildError::{UnderspecifiedTree, OverspecifiedTree}`;
  out-of-range lengths (`1..=32`) and zero-used-entries surface as
  `InvalidLength` / `EmptyTree`. End-of-packet mid-codeword surfaces
  as `HuffmanDecodeError::UnexpectedEndOfPacket` per §3.3 "reading
  past the end of a packet propagates the 'end-of-stream' condition
  to the decoder". 13 new unit tests cover the spec worked example,
  concatenated stream decode, sparse codebook with entry index
  `0xFF`, single-entry both-bits tolerance, single-entry length≠1
  rejection, fully-unused rejection, underspecified / overspecified /
  invalid-length rejection, EOF on truncated input, balanced 16-entry
  length-4 round-trip, max-length-32 entries, and `from_codebook` ↔
  `from_lengths` equivalence (69 tests total, up from 56).

* **Vorbis I codebook-header parser (Vorbis I §3.2.1).**
  `parse_codebook(&mut BitReaderLsb)` decodes a single Vorbis I
  codebook header — the most foundational sub-structure of the
  setup header (§4.2.4). The function reads the 24-bit `0x564342`
  sync pattern, 16-bit `dimensions`, 24-bit `entries`, 1-bit
  `ordered` flag, then the codeword-length table (dense, sparse,
  or ordered run-length encoded per §3.2.1), the 4-bit
  `lookup_type`, and for `lookup_type ∈ {1, 2}` the 32-bit
  `minimum_value` + 32-bit `delta_value` (both passed through
  `float32_unpack` per §9.2.2), `value_bits - 1` (4 bits),
  `sequence_p` (1 bit), and the raw multiplicand table (size
  `lookup1_values(entries, dimensions)` for type 1 per §9.2.3 or
  `entries × dimensions` for type 2). Returns
  `VorbisCodebook { dimensions, entries, codeword_lengths,
  lookup: VqLookup }`. Sparse unused entries are stored as the
  `UNUSED_ENTRY` sentinel (0); spec-legal lengths are 1..=32.
  Spec-mandated errors surface as `codebook::ParseError` variants
  (`BadSyncPattern`, `ZeroEntries`, `OrderedOverflow`,
  `ReservedLookupType`, `UnexpectedEndOfPacket`). 18 unit tests
  cover the spec's worked example (§3.2.1 lengths 2 4 4 4 4 2 3
  3), sparse encoding with unused entries, ordered run-length
  encoding, lookup types 0 / 1 / 2 with synthetic multiplicand
  tables, the `lookup1_values(2916, 8) = 2` corner from
  trace-doc §3, the spec examples for `ilog` (§9.2.1) and
  `float32_unpack` (§9.2.2), and every documented `ParseError`
  failure mode. The §9.2.1 / §9.2.2 / §9.2.3 helpers are also
  exposed at the module root for downstream use by the upcoming
  Huffman-tree builder and the floor / residue parsers.

* **Vorbis I comment-header parser (Vorbis I §5).**
  `parse_comment_header(&[u8])` decodes a Vorbis I comment-header
  packet (common header `0x03 "vorbis"` + §5.2.1 body) and returns a
  `VorbisCommentHeader { vendor: String, comments: Vec<String> }`.
  The body is read per §5.2.1 (byte-aligned `u32` LE
  `vendor_length` + UTF-8 vendor + `u32` LE
  `user_comment_list_length` + per-comment `u32` LE length + UTF-8
  bytes + 1-bit framing flag). All spec-mandated checks are
  enforced and surface as `comment::ParseError` variants
  (`PacketTooShort`, `WrongPacketType`, `BadMagic`,
  `UnexpectedEndOfPacket` — see §4.2 for the comment-header's
  "non-fatal" relaxation —, `LengthOverflow`, `InvalidVendorUtf8`,
  `InvalidCommentUtf8`, `BadFramingFlag`). Helpers
  `VorbisCommentHeader::key_value_iter` and `split_key_value` parse
  the §5.2.2 `KEY=value` shape. 22 unit tests cover the
  `mono-44100-q5-typical` (1-entry encoder tag) and
  `with-vorbis-comment-tags` (7-entry title/artist/album/date/genre/
  tracknumber + encoder) fixture shapes, the historical
  `Xiph.Org libVorbis I 20020717` vendor string, empty vendor + empty
  comment list, multi-byte UTF-8 in vendor and comment values,
  duplicate-key entries per §5.2.2, a 64 KiB
  `METADATA_BLOCK_PICTURE`-sized payload, framing-byte padding per
  §2.1.8, and every documented `ParseError` failure mode.

* **Vorbis I identification-header parser (Vorbis I §4.2.2).**
  `parse_identification_header(&[u8])` decodes a 30-byte
  identification-header packet and returns a
  `VorbisIdentificationHeader` struct exposing `vorbis_version`,
  `audio_channels`, `audio_sample_rate`, the three signed bitrate
  hints, and the two resolved block sizes. All spec-mandated
  invariants (packet type byte, `"vorbis"` magic per §4.2.1,
  `vorbis_version == 0`, nonzero channels and sample rate, blocksize
  exponents in 6..=13, `blocksize_0 <= blocksize_1`, framing flag
  nonzero) are enforced and surface as `ParseError` variants. 16 unit
  tests cover both the happy-path field-shape equivalents of the
  trace-doc fixtures (`mono-44100-q5-typical`,
  `5.1-channel-48000-q5`, spec-minimum and -maximum blocksizes,
  signed bitrate-hint encoding) and every documented `ParseError`
  failure mode.

### Changed

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule. Orphan-master rebuild per workspace
  policy; no `old` branch retained. License also reset to clean MIT.

  Round 1 lands the identification-header parse only. Comment header
  (§5), setup header (§4.2.4), codebook/floor/residue/mapping/mode
  decode (§3, §6, §7, §8, §4.3) and audio-packet decode (§4.3.2) are
  pending in later rounds; `decode_packet` returns
  `Error::NotImplemented`. The Ogg framing layer (RFC 3533 + Vorbis
  I §A) is also not yet wired up.

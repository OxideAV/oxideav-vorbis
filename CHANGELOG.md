# Changelog

All notable changes to `oxideav-vorbis` are recorded here.

## [Unreleased]

### Added

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
  total (146 → 164). libvorbis never emits floor 0 (every reference
  encoder uses floor 1 exclusively) but a conformant Vorbis I decoder
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

# oxideav-vorbis

Pure-Rust Vorbis I audio codec — clean-room rebuild, round 18.

## Status — 2026-05-29 (round 18)

**Round 18 landed: multi-channel streaming PCM driver.** New module
[`streaming`] exposes [`StreamingDecoder`] — a per-stream state machine
holding one [`overlap::OverlapAdd`] instance per channel — that stitches
the round-17 [`decode_audio_packet_windowed`] per-packet driver to the
§4.3.8 overlap-add primitive across consecutive packets. The decoder is
constructed from the identification-header fields (`audio_channels` /
`blocksize_0` / `blocksize_1`) plus the deferred `imdct_scale`, then
driven one packet at a time via
[`StreamingDecoder::push_packet`]`(reader, setup, state)` (or
[`StreamingDecoder::push_windowed`]`(outcome)` for callers that already
hold a [`WindowedPacketOutcome`]). The first packet primes every
per-channel overlap-add state (§4.3.8 priming step) and returns
[`StreamingFrame::Primed`]; from the second packet on the engine emits
[`StreamingFrame::Pcm`] holding `prev_n / 4 + cur_n / 4` finished PCM
samples per channel (bitstream channel order — §4.3.9 rearrangement is
a presentation concern handled above this module).
[`StreamingDecoder::reset`] returns every per-channel state to priming
(e.g. after a seek); [`StreamingDecoder::finish`] drains the last
frame's right-half tail (`n / 2` samples per channel) for callers
flushing a finite encoded clip. New error type [`StreamingError`] with
three variants (`Packet` / `Overlap { channel, source }` /
`ChannelCountMismatch { expected, got }`) surfaces the underlying
failures; the umbrella [`Error::Streaming`] variant + `From` impl wire
them up at the crate root.

14 new unit tests cover construction accessors, the §4.3.8 priming step
on a long block, the `prev_n / 4 + cur_n / 4` spec-formula return length
for equal-sized and mixed-sized block transitions, two-channel routing
with a per-channel ratio invariant on synthetic ramps, the defensive
`ChannelCountMismatch` check, the `ZeroedWindowed` packet propagating
cleanly through priming and emitting the previous-frame plateau on a
zeroed-after-normal transition, `reset()` returning to priming, the
`finish()` per-channel tail drain (or `None` on an unprimed engine),
the per-packet failure surface via `StreamingError::Packet`, and the
two non-trivial `Display` strings. Test count: **288 total
(274 → 288)**. With this round the entire §4.3 pipeline from a parsed
audio-packet bitstream to PCM is reachable end-to-end as a single
[`StreamingDecoder::push_packet`] call per packet — the last
composition step the round-17 wiring named. Only the Vorbis-specific
IMDCT normalization scalar `imdct_scale` is still caller-supplied; the
staged fixture traces under `docs/audio/vorbis/fixtures/<case>/trace.txt`
do not yet log post-IMDCT samples, so pinning that constant remains a
documented docs gap.

## Status — 2026-05-29 (round 17)

**Round 17 landed: §4.3.7 IMDCT + §4.3.6 window wired into the per-packet
driver.** New entry point [`decode_audio_packet_windowed`] drives
§4.3.2..§4.3.6 via the existing [`decode_audio_packet_pre_imdct`], runs
the §4.3.7 [`imdct::imdct_naive`] cosine-summation kernel on each
channel's length-`n/2` spectrum to obtain the length-`n` time-domain
frame, then element-wise multiplies by the §4.3.6 / §1.3.2 Vorbis
window built once per packet via [`AudioPacketHeader::build_window`].
The result is one length-`n` windowed time-domain frame per channel —
exactly the input the §4.3.8 [`overlap::OverlapAdd::push_frame`]
primitive expects. The convenience [`decode_one_packet_windowed`] is
the same call shaped for "drop in to the streaming pipeline." New pure
transform [`apply_imdct_and_window(outcome, blocksize_0, imdct_scale)`]
lifts §4.3.7-then-§4.3.6 over an already-parsed
[`AudioPacketOutcome`] for callers that hold one (e.g. from a buffered
decode). New outcome enum [`WindowedPacketOutcome`] holds either
`Windowed { frames, … }` for a normal packet or `ZeroedWindowed { frames,
… }` for the §4.3.2 short-circuit (IMDCT of zero is zero × any window is
still zero); both expose `header()` and `frames()` accessors.

The `imdct_scale: f32` parameter on every new entry point is the
**deferred-normalization knob** the IMDCT cross-reference document
(`docs/audio/vorbis/imdct-cross-reference.md` §"Vorbis-specific
parameters" item 5) names — the Vorbis-specific normalization scalar
that maps the bare kernel to oggdec-bit-equivalent PCM. The
cross-reference notes the scalar "falls out of matching the fixture
traces," and the staged fixture traces under
`docs/audio/vorbis/fixtures/<case>/trace.txt` do not yet log post-IMDCT
samples; the scalar is therefore still pinned to caller-supplied. By
linearity of the IMDCT kernel `imdct_scale` is a pure output multiplier:
scaling it by α scales every returned sample by α, so a future round
can land the fixture-derived value as a constant without changing call
sites.

11 new tests cover the windowed driver on the trivial mono synthetic
packet (one length-`n` frame per channel; geometry pinned), the pure
[`apply_imdct_and_window`] transform on a hand-built outcome with a
long-block window (lead-in / tail regions exactly zero by window-edge
construction), the §4.3.2 short-circuit (`ZeroedWindowed` returns
per-channel all-zero length-`n` frames), the `imdct_scale` linearity
property, the IMDCT-then-window composition matching the direct
`imdct_naive_vec` × `vorbis_window` path bit-for-bit, end-to-end
integration with `OverlapAdd::push_frame` (first call primes the
overlap-add state, second emits the §4.3.8 finished-PCM range), legacy
[`decode_one_packet`] preservation of the `ImdctStage` stop,
[`decode_one_packet_windowed`] parity with
[`decode_audio_packet_windowed`], accessor returns, and the new
[`AudioPacketError::Window`] / [`AudioPacketError::Imdct`] Display
strings. Two new error variants land on [`AudioPacketError`]
(`Window(WindowError)` / `Imdct(ImdctError)`); both surface verbatim
with §4.3.6 / §4.3.7 prefixes. Legacy [`decode_one_packet`] is
preserved unchanged so callers depending on the pre-IMDCT stop are not
broken. Test count: **274 total (263 → 274)**. With this round the
entire §4.3 pipeline from a parsed audio-packet bitstream to PCM is
reachable in code: only the fixture-derived `imdct_scale` constant
remains as a documented docs gap.

## Status — 2026-05-29

**Round 16 landed: the §4.3.7 inverse-MDCT direct cosine-summation kernel
as a standalone math primitive.** New module [`imdct`] exposes
[`imdct_naive(spectrum, output, scale)`] and its allocating
[`imdct_naive_vec(spectrum, scale)`] convenience wrapper. The kernel
implements the bare cosine summation
`x[n] = sum_k X[k] · cos[ (π/N) · (2n + 1 + N/2) · (2k + 1) / 2 ]`
verbatim from the OxideAV clean-room companion document
`docs/audio/vorbis/imdct-cross-reference.md`. That document closes the
Vorbis I §4.3.7 spec-deferral to external reference `[1]`
(Sporer/Brandenburg/Edler, barred by workspace policy) by observing
that the IMDCT mathematical kernel is **also** restated in three
adjacent in-repo specs (ATSC A/52 §7.9.4, ISO/IEC 14496-3 §4.6.x,
IETF RFC 6716 §4.3.7) and giving the canonical formula. The
implementation is the O(N²) direct form, chosen as the *reference*
implementation that is provably correct by inspection against the
cosine summation; a future round can land an FFT-decomposed fast path
and validate byte-for-byte against this kernel. Working precision is
`f64` accumulators with `f32` output to match the spectral pipeline.
12 new unit tests cover the error paths, the mathematical properties
derivable directly from the cosine summation (zero input → zero
output; linearity; left-half anti-symmetry `x[i] = -x[N/2 - 1 - i]`;
right-half symmetry `x[N/2 + i] = x[N - 1 - i]` — the two TDAC
half-rules), the `scale` argument's linearity property, and two
hand-computed N = 4 fixtures (impulse on DC bin / impulse on k = 1)
that pin the exact cosine arguments against any off-by-one in the
`(2n + 1 + N/2) · (2k + 1) / 2` form. The Vorbis-specific
normalization scalar that would make the kernel output
oggdec-bit-equivalent is **deliberately not pinned in this round** —
`imdct-cross-reference.md` notes it "falls out of matching the
fixture traces," and the staged fixtures under
`docs/audio/vorbis/fixtures/<case>/trace.txt` do not yet log
post-IMDCT samples. The `scale` argument is a `f32` multiplier the
caller can plug a tentative factor into; a follow-up round pins its
value when those traces extend. The top-level [`decode_packet`]
still stops at the §4.3.7 boundary for the same reason. Test count:
263 total (251 → 263). With this round, every §4.3.x stage the
Vorbis I spec body or the clean-room IMDCT cross-reference defines
exists as a standalone primitive; the remaining work is the
fixture-derived normalization and the per-packet driver glue.

## Status — 2026-05-25

**Round 15 landed: the §4.3.8 overlap-add primitive as a standalone,
IMDCT-independent math module.** A new [`overlap`] module exposes
[`OverlapAdd`] — a one-channel stateful overlap-add engine driven by
[`OverlapAdd::push_frame(windowed_frame)`]. It consumes the windowed
time-domain frame (the §4.3.7 IMDCT output multiplied by the §1.3.2 /
§4.3.1 Vorbis window) and returns the finished PCM samples for the
previous → current frame transition per §4.3.8: `prev_n / 4 +
cur_n / 4` samples spanning the previous-window center
(`windowsize / 2`) to the current-window center (`windowsize / 2 - 1`,
inclusive). The 3/4-vs-1/4 alignment geometry is reproduced from
§4.3.8 verbatim; the spec's first-frame priming step (no output) is
modelled as `Ok(None)` on the first [`push_frame`] call. All four
mixed-size combinations (long→long, long→short, short→long,
short→short) are exercised against the §4.3.8 spec text, including
the spec's note that a short→long boundary lets the current frame
begin BEFORE the previous-frame center (negative offset, handled
with signed arithmetic). The squared-window perfect-reconstruction
property of §1.3.2 (`w[i]² + w[i + n/2]² == 1`) is verified
end-to-end: pushing the same windowed unit-signal through two
consecutive frames returns constant `1.0` across the entire overlap
region. With this round, every §4.3.x stage the Vorbis I spec body
defines in its own text is now implemented as a standalone primitive;
the only remaining gap is the §4.3.7 inverse MDCT itself, still
gated on the spec's externally-cited reference `[1]`. When the IMDCT
round lands, the audio-packet driver glue is: per channel, run IMDCT
on the `PreImdct` spectrum, multiply by the §4.3.1 window, hand the
result to that channel's [`OverlapAdd`] instance, and emit whatever
PCM it returns.

**Round 14 landed: the top-level §4.3 audio-packet driver covering
§4.3.2 floor decode (per-channel, submap-routed) through §4.3.6 dot
product, stopping cleanly at the §4.3.7 inverse-MDCT boundary.**
[`audio::decode_audio_packet_pre_imdct`] takes an LSB-first bit reader
positioned at an audio packet, the parsed setup header, a per-stream
[`AudioDecoderState`] decoder cache (built once via
[`AudioDecoderState::new(setup)`] — every floor and residue decoder
constructed up front so per-packet decode is allocation-light), the
stream's `audio_channels`, and the two blocksizes, and runs every §4.3
stage whose definition the spec gives in its own text:

1. §4.3.1 packet header via the existing [`read_packet_header`].
2. §4.3.2 floor decode in channel order: for each channel, look up its
   submap via `mapping.mux[channel]` (or always `0` when the mapping
   declared `submaps == 1`), pick the submap's floor index, and run the
   matching [`Floor0Decoder`] / [`Floor1Decoder`] for a length-`n/2`
   curve. The §4.3.2 step-6 `[no_residue]` flag is set per the floor's
   `Unused` return; the floor implementations already collapse EOF to
   `Unused`, matching §4.3.2's nominal-EOF rule.
3. §4.3.3 nonzero propagate via the existing [`nonzero_propagate`].
4. §4.3.4 residue decode in submap order: for each submap, gather the
   channels whose `mux[channel] == submap_idx` (preserving channel
   order), pack their `no_residue` flags into the `do_not_decode_flag`
   bundle, dispatch the submap's [`ResidueDecoder::decode`], and scatter
   the per-channel vectors back into the global per-channel array
   (§4.3.4 step 7).
5. §4.3.5 inverse coupling via the existing [`inverse_couple_all`]
   (descending step order).
6. §4.3.6 dot product via [`dot_product_all`]: every used channel emits
   the element-wise floor × residue product; every unused channel emits
   the all-zero spectrum per §4.3.3's closing note.

The driver returns an [`AudioPacketOutcome::PreImdct { mode_number,
blockflag, n, previous_window_flag, next_window_flag, spectra }`]
carrying one length-`n/2` audio spectrum vector per channel, ready for
the §4.3.7 IMDCT (still pending on a documented docs gap — the spec
defers the MDCT to external reference `[1]`, barred by the workspace
clean-room policy). The top-level [`decode_packet`] /
[`audio::decode_one_packet`] entry points stop at the IMDCT boundary
and return [`AudioPacketError::ImdctStage`] (`§4.3.7 inverse MDCT is a
documented docs gap …`) cleanly. The (still-pending) IMDCT-round
landing is a one-function plug-in: take the `spectra` from `PreImdct`,
run the IMDCT + window + overlap-add, and replace the `ImdctStage`
return with the resulting PCM. No callable surface changes beyond
that.

**Round 13 landed: the §4.3.1 "packet type, mode and window decode"
packet-prelude reader — the audio-packet driver's entry point that the
remaining §4.3 stages feed off.** [`read_packet_header`] takes an LSB-first
[`BitReaderLsb`] positioned at the first bit of an audio packet, the parsed
setup header (only `setup.modes` is consulted), and the stream's two
blocksizes; it reads (1) the 1-bit `[packet_type]` and rejects any
non-audio packet per the §4.3 "must ignore" rule, (2) the
`ilog([vorbis_mode_count] - 1)`-bit `[mode_number]` (zero bits when
`mode_count == 1`, the single-mode degenerate case), (3) resolves the
per-frame blocksize `[n]` from the selected mode's `blockflag`
(§4.3.1 step 3), and (4) reads the two long-block-only window flags
`[previous_window_flag]` / `[next_window_flag]` (step 4a.i/ii) or skips
them for a short block (step 4b: short blocks always reuse the symmetric
short shape and do not transmit these bits). The returned
[`AudioPacketHeader`] carries the resolved `(mode_number, blockflag, n,
previous_window_flag, next_window_flag)`; [`AudioPacketHeader::build_window`]
hands them off to the existing [`vorbis_window`] builder for a length-`n`
Vorbis window. End-of-packet anywhere in §4.3.1 is fatal (§4.3.1 closing
note: "An end-of-packet condition up to this point should be considered an
error that discards this packet from the stream"), surfaced as
[`PacketError::UnexpectedEndOfPacket`] with per-sub-step granularity via
[`PacketHeaderStage`]. With this round every §4.3.x stage whose definition
the spec text gives in its own body is implemented; the only remaining
piece is the §4.3.7 inverse MDCT, still gated on the spec's externally-cited
reference (T. Sporer / K. Brandenburg / B. Edler).

**Round 12 landed: the two fully-specified, IMDCT-independent stages of
the §4.3 audio-packet decode driver — §4.3.3 "nonzero vector propagate"
and §4.3.6 "dot product".** [`nonzero_propagate`] runs the §4.3.3
ascending coupling-step loop over a per-channel `no_residue` flag slice:
for each `(magnitude_channel, angle_channel)` pair, if *either* member is
used (`no_residue` false) §4.3.3 forces *both* members used, because
coupling can mix a zeroed and nonzeroed vector to produce two nonzeroed
ones. [`dot_product`] computes the §4.3.6 element-wise (Hadamard)
product of one channel's length-`n/2` floor curve and residue vector
into a caller-provided spectrum buffer; [`dot_product_all`] is the
per-channel driver that emits the all-zero spectrum for an unused channel
(`floors[ch] = None`, per §4.3.3 "that final output vector is all-zero
values") and the element-wise product for every used channel. These two
stages sit at §4.3 steps 4 (between residue decode and inverse coupling)
and 6 (between inverse coupling and the inverse MDCT); they complete every
§4.3 stage whose definition the Vorbis I spec gives in its own text. The
remaining §4.3.7 inverse MDCT is a **documented docs gap**: the spec
defers the MDCT definition entirely to external reference `[1]` (T.
Sporer, K. Brandenburg, B. Edler, *The use of multirate filter banks for
coding of high quality digital audio*), which the workspace clean-room
policy bars. The Vorbis-specific MDCT normalization / sign convention
that would make a decoder byte-exact against `oggdec` is not stated
anywhere in `docs/audio/vorbis/`.

**Round 11 landed: audio-packet synthesis primitives — the Vorbis window
(Vorbis I §1.3.2 / §4.3.1 "packet type, mode and window decode") and
inverse channel coupling (§4.3.5 "inverse coupling").** [`vorbis_window`]
builds the length-`n` window for one audio frame per the §4.3.1
eight-step generation procedure: a zero lead-in, a rising edge, a plateau
of ones, a `+π/2`-phase-shifted falling edge, and a zero tail. Every edge
uses the Vorbis slope function `y = sin(π/2·sin²((x+0.5)/n·π))` (§1.3.2),
exposed bare as [`slope`]. A long block (`blockflag` set) reads two flag
bits and selects the `n/4 ± blocksize_0/4` hybrid ramp on either side
whose neighbour is short (the matching window flag clear), preserving the
50%-overlap squared-power reconstruction across dissimilar lapping;
otherwise the full-half-block ramp is used, and a short block always gets
the plain symmetric shape (§4.3.1 step 4b). [`inverse_couple_all`] runs
the §4.3.5 inverse-coupling loop over a per-channel residue-vector bundle
**in descending coupling-step order**, decoupling each `(magnitude,
angle)` pair in place via [`couple_scalar`] — the four-quadrant
square-polar → Cartesian rule of §4.3.5 step 3. These are the pure,
stateless transforms of the §4.3 pipeline that sit between residue decode
and the inverse MDCT; the packet-level driver tying floor + residue +
window + coupling + MDCT together is still pending.

**Round 10 landed: floor type 0 per-packet decode + LSP curve
computation (Vorbis I §6.2.2 "packet decode" + §6.2.3 "curve
computation").** [`Floor0Decoder`] turns the floor payload of an audio
packet into a linear-domain spectral envelope of length `n` (=
`blocksize/2`). It reads the `[amplitude]` field (`floor0_amplitude_bits`
bits, returning `Floor0Curve::Unused` when zero), then the `[booknumber]`
(`ilog([floor0_number_of_books])` bits), then loops reading VQ vectors
from `floor0_book_list[booknumber]` and concatenating them into
`[coefficients]` until the vector reaches at least `floor0_order`
elements — each new vector has the running `[last]` accumulator added
before concatenation (§6.2.2 steps 6..10), and `[last]` is reset to the
last scalar of the just-decoded vector after each iteration. The spec
explicitly permits the loop to over-read past `floor0_order`; the curve
synthesis slices to the first `order` coefficients. The curve
computation builds a Bark-scale frequency map per §6.2.3 (using the
post-errata `bark(x) = 13.1·atan(.00074x) + 2.24·atan(.0000000185x²) +
.0001x` formula), then iterates the order-parity-dependent `[p]`/`[q]`
LSP product through the `exp(.11512925 · …)` log→linear amplitude
transform at every angle `[ω] = π · map[i] / floor0_bark_map_size`,
replicating the value across every consecutive output bin whose `map[i]`
matches the current synthesis bin (the `[iteration_condition]` chaining
of §6.2.3 step 8). An end-of-packet condition anywhere in §6.2.2 is the
spec's nominal occurrence: decode returns `Floor0Curve::Unused` exactly
as if `[amplitude]` had read zero. Reserved-`booknumber` values
(`booknumber >= floor0_book_list.len()` because the spec stores the
field in `ilog([floor0_number_of_books])` bits with the high values
reserved) also map to `Unused`. libvorbis never produces floor 0 but a
conformant Vorbis I decoder must implement the codepath; this is the
missing-piece half of the floor decode pipeline (round 9 covered floor
1).

**Round 9 landed: floor type 1 per-packet decode + curve computation
(Vorbis I §7.2.3 "packet decode" + §7.2.4 "curve computation").**
[`Floor1Decoder`] turns the floor payload of an audio packet into a
linear-domain spectral envelope of length `n` (= `blocksize/2`). It reads
the `[nonzero]` flag (returning `Unused` if clear), the two
`ilog([range]-1)`-bit endpoint amplitudes, and per partition the
master-book selector (only when `[cbits] > 0`) followed by the
per-dimension sub-book scalar amplitudes, yielding `[floor1_Y]`; then
unwraps those positive differences through iterative `render_point` line
prediction (§7.2.4 step 1) and renders the sorted contiguous line
segments via `render_line` (§7.2.4 step 2) before substituting through
the §10.1 inverse-dB table. End-of-packet mid-decode is the §7.2.3
nominal occurrence (return `Unused`). This is the floor half of the
§4.3.2 audio-packet pipeline that complements round 8's residue half.

**Round 8 landed: per-packet residue decode (Vorbis I §8.6.2 "packet
decode" + §8.6.3/§8.6.4/§8.6.5 "format 0/1/2 specifics").**
[`ResidueDecoder`] turns the residue payload of an audio packet into one
`Vec<f32>` per channel: it caps the begin/end range to the per-format
vector size, reads each partition's classification from the classbook in
scalar context (pass 0), and over passes 0..=7 accumulates the
stage-`pass` value codebook's VQ vectors into the output — format 0
interleaved-scatter, format 1 contiguous, format 2 interleave→format-1→
de-interleave. End-of-packet mid-decode is the §8.6.2 nominal occurrence
(stop and return work-so-far). Round 7 landed VQ vector unpack
([`unpack_vector`]), lifting a decoded Huffman entry index into a fixed
`codebook.dimensions`-element `Vec<f32>` by walking the spec's
mixed-base permutation (`lookup_type = 1`, lattice) or one-to-one
slice (`lookup_type = 2`, tessellation) of the codebook's multiplicand
table, honouring `sequence_p` as a running prefix-sum of the per-element
`multiplicand × delta + minimum` term. Round 6 landed the rest of the
setup-header walker — mapping configurations (§4.2.4 "Mappings",
`mapping_type = 0` only), mode configurations (§4.2.4 "Modes"), and the
trailing framing flag. Round 5 landed the setup-header outer walker
(codebooks + time-domain placeholders + floor headers + residue
headers); round 4 the canonical Huffman tree builder + entry decoder
(§3.2.1); round 3 the codebook-header parser (§3.2.1); round 2 the
comment-header parser (§5); round 1 the identification-header parser
(§4.2.2) on 2026-05-20.

The crate's prior implementation was retired under the workspace
clean-room policy because module-level docstrings and inline comments
referenced libvorbis internals as their provenance source. The current
master is an orphan rebuild that started from a `NotImplemented`
scaffold; round 1 lands the identification header, round 2 the
comment header.

### What works

* [`parse_identification_header`] reads a 30-byte Vorbis I
  identification-header packet (Vorbis I §4.2.2) and returns a
  [`VorbisIdentificationHeader`] struct exposing:
  * `vorbis_version` (u32; must be 0 for Vorbis I)
  * `audio_channels` (u8; must be > 0)
  * `audio_sample_rate` (u32, Hz; must be > 0)
  * `bitrate_maximum` / `bitrate_nominal` / `bitrate_minimum`
    (signed i32 hints; 0 means "unset" per §4.2.2)
  * `blocksize_0` / `blocksize_1` (u16, sample counts; one of
    {64, 128, 256, 512, 1024, 2048, 4096, 8192} per §4.2.2, with
    `blocksize_0 <= blocksize_1`)
* All spec-mandated validity checks from §4.2.2 are enforced: packet
  length, common-header magic + packet type, nonzero
  channels/sample_rate, blocksize exponents in 6..=13, blocksize
  ordering, framing flag nonzero.
* [`parse_comment_header`] reads a Vorbis I comment-header packet
  (§5.2.1 / §5.2.3) and returns a [`VorbisCommentHeader`] exposing:
  * `vendor` (UTF-8 String; the encoder's vendor identification, e.g.
    `"Lavf61.7.100"` or `"Xiph.Org libVorbis I 20020717"`)
  * `comments` (Vec<String>; raw UTF-8 entries in `KEY=value` form
    per §5.2.2, preserving insertion order)
* All spec-mandated invariants from §5.2.1 are enforced: common-header
  packet type (`0x03`) + `"vorbis"` magic, UTF-8 validation of the
  vendor and every comment entry, length-prefix overflow guard,
  truncation reported as a structured `UnexpectedEndOfPacket` so
  callers can apply the §4.2 "non-fatal" relaxation if desired,
  framing-bit check.
* Helpers: [`split_key_value`] cuts an entry on the first `=` octet
  per §5.2.2; [`VorbisCommentHeader::key_value_iter`] yields
  `(key, value)` pairs and skips malformed (no-`=`) entries.
* [`parse_codebook`] reads a Vorbis I codebook header (§3.2.1) from
  an LSB-first [`oxideav_core::bits::BitReaderLsb`] positioned at
  the 24-bit `0x564342` sync pattern. Returns a [`VorbisCodebook`]
  exposing:
  * `dimensions` (u16; 16-bit unsigned)
  * `entries` (u32; 24-bit unsigned)
  * `codeword_lengths` (Vec<u8>; per-entry length in `1..=32`, or
    [`UNUSED_ENTRY`]`= 0` for sparse-codebook unused entries)
  * `lookup` ([`VqLookup`]; `None` for `lookup_type=0`, `Lattice`
    for `lookup_type=1`, `Tessellation` for `lookup_type=2`,
    carrying the unpacked `minimum_value` / `delta_value` floats
    (§9.2.2), `value_bits` in `1..=16`, `sequence_p` flag, and the
    raw multiplicand table)
* Codebook helpers exposed at the crate root: [`ilog`] (§9.2.1),
  [`float32_unpack`] (§9.2.2), [`lookup1_values`] (§9.2.3). The
  ordered length-encoding branch (§3.2.1 step 3 ordered subcase)
  reuses `ilog(entries - current_entry)` for the per-run width.
* All §3.2.1 invariants are enforced and surface as
  `codebook::ParseError` variants (`BadSyncPattern`, `ZeroEntries`,
  `OrderedOverflow`, `ReservedLookupType`, `UnexpectedEndOfPacket`).
* [`HuffmanTree::from_codebook`] / [`HuffmanTree::from_lengths`]
  builds a canonical Vorbis I Huffman decision tree (§3.2.1) from the
  per-entry `codeword_lengths` table. Construction uses a left-to-right
  open-position deque: each used entry pops the leftmost open slot and
  either places a leaf there or splits it down to the entry's recorded
  depth (allocating internal nodes + pushing both new children to the
  deque front). The spec's worked example (lengths `[2 4 4 4 4 2 3 3]`
  → codewords `00 0100 0101 0110 0111 10 110 111`) round-trips
  exactly.
* [`HuffmanTree::decode_entry`] walks the tree against an LSb-first
  [`BitReaderLsb`]: each read bit selects `left` (0) or `right` (1)
  until a leaf is hit, then returns the leaf's entry index. The first
  bit read is the MSb of the canonical codeword per the §3.2.1 "the
  leftmost bit is the MSb" convention; under §2.1.4 LSb-first packing
  this means the encoder writes codewords with their MSb going into
  the LSb of the next stream byte.
* Errata 20150226 single-entry codebooks: a codebook with exactly one
  used entry whose recorded length is `1` builds a synthetic tree
  whose root's `left` and `right` both point at the same leaf, so
  `decode_entry` returns that entry and sinks one bit on either a `0`
  or a `1` (per the errata "decoders should tolerate that the bit
  read from the stream be '1' instead of '0'"). Single-entry
  codebooks with `length != 1` are rejected with
  [`HuffmanBuildError::SingleEntryWrongLength`].
* Underspecified / overspecified detection (§3.2.1 errata 20150226):
  any leftover open deque positions after every used entry has been
  placed → [`HuffmanBuildError::UnderspecifiedTree`]; popping the
  deque dry before all entries are placed →
  [`HuffmanBuildError::OverspecifiedTree`]. Out-of-range lengths
  (anything outside `1..=32`) surface as
  [`HuffmanBuildError::InvalidLength`], and zero used entries as
  [`HuffmanBuildError::EmptyTree`].
* [`parse_setup_header`] / [`parse_setup_header_body`] now walk the
  **entire** Vorbis I setup-header packet (§4.2.4). Both entry points
  take the stream's `audio_channels` (from the identification header)
  because the mapping decode reads channel-number widths of
  `ilog(audio_channels - 1)` bits:
  * `vorbis_codebook_count` codebook configurations (delegated to
    [`parse_codebook`]; §3.2.1).
  * `vorbis_time_count` 16-bit time-domain transform placeholders —
    each spec-mandated to equal zero (§4.2.4 step 2; any nonzero value
    is rejected with `SetupParseError::NonZeroTimePlaceholder`).
  * `vorbis_floor_count` floor headers. Each carries a 16-bit
    `floor_type`; type 0 (§6.2.1) and type 1 (§7.2.2) decode their
    structural fields (no per-packet curve decode), type > 1 is
    rejected. [`Floor1Header`] exposes `partitions`,
    `partition_class_list`, `classes` (per-class `dimensions`,
    `subclasses`, optional `masterbook`, `subclass_books`),
    `multiplier`, `rangebits`, `x_list`. [`Floor0Header`] exposes
    `order`, `rate`, `bark_map_size`, `amplitude_bits`,
    `amplitude_offset`, `book_list`.
  * `vorbis_residue_count` residue headers (§8.6.1 — common header
    layout across types 0/1/2). [`ResidueHeader`] exposes
    `residue_type` (0/1/2; >2 is rejected), `residue_begin`,
    `residue_end`, `partition_size`, `classifications`, `classbook`,
    `cascade[classifications]`, and `books[classifications][8]` with
    `None` entries for cascade-bit-unset stages.
  * `vorbis_mapping_count` mapping configurations (§4.2.4 "Mappings",
    `mapping_type = 0` only — any other type is rejected per step 2b).
    [`MappingHeader`] exposes `mapping_type`, `submaps` (1 or 4-bit
    encoded), `coupling` (per-step `(magnitude_channel, angle_channel)`
    pairs read at `ilog(audio_channels - 1)` bits each, with the spec's
    "magnitude != angle and both < audio_channels" validation), `mux`
    (per-channel submap routing; only present when `submaps > 1`), and
    `submap_configs` (the 8-bit time placeholder + the 8-bit floor /
    residue indices, both range-checked against the prior counts).
  * `vorbis_mode_count` mode configurations (§4.2.4 "Modes").
    [`ModeHeader`] exposes `blockflag`, `windowtype` (forced to 0 per
    step 2e), `transformtype` (forced to 0 per step 2e), and `mapping`
    (range-checked against `mapping_count`).
  * The trailing 1-bit framing flag (§4.2.4 "Modes" step 3) is read
    and required to be set; the flag value is also surfaced as
    `VorbisSetupHeader::framing_flag` for downstream inspection.
* [`unpack_vector`] applies a codebook's `codebook_lookup_type` to a
  Huffman entry index `lookup_offset` to recover the per-entry VQ
  vector (Vorbis I §3.2.1 + §3.3). Returns a `Vec<f32>` of length
  exactly `codebook.dimensions`. Three branches:
  * `VqLookup::None` → [`VqUnpackError::NoVectorForType0`] (§3.3:
    "requesting decode using a codebook of lookup type 0 in any
    context expecting a vector return value … is forbidden").
  * `VqLookup::Lattice` (lookup type 1) → mixed-base permutation per
    §3.2.1 "Vector value decode: Lookup type 1" — `multiplicand_offset
    = (lookup_offset / index_divisor) mod codebook_lookup_values` for
    each dimension `i`, with `index_divisor` multiplied by
    `codebook_lookup_values` between iterations.
  * `VqLookup::Tessellation` (lookup type 2) → direct one-to-one slice
    per §3.2.1 "Vector value decode: Lookup type 2" —
    `multiplicand_offset = lookup_offset * codebook_dimensions`,
    then increment per iteration.
  * Both honour `sequence_p`: when set, `[last]` carries forward the
    full prior `value_vector[i]` (post-`min`, post-`delta`,
    post-`last`), making the output a prefix-sum; when clear, `[last]`
    stays at `0.0`. Structured `VqUnpackError` variants:
    `EntryOutOfRange`, `NoVectorForType0`, `ZeroDimensions`,
    `MultiplicandShapeMismatch`.
* [`ResidueDecoder`] decodes the per-packet residue payload (Vorbis I
  §8.6.2 + §8.6.3/§8.6.4/§8.6.5) into one `Vec<f32>` per channel.
  [`ResidueDecoder::new`] validates the §8.6.1 undecodability clauses
  (classbook + value-book indices in range, value books carry a value
  mapping, classbook dimensions nonzero) and pre-builds the classbook +
  value-book Huffman trees once. [`ResidueDecoder::decode`] runs the
  §8.6.2 packet decode:
  * caps `[residue_begin]`/`[residue_end]` to the actual vector size —
    `blocksize/2` for format 0/1, `blocksize/2 × ch` for format 2 (the
    interleaved-vector cap of §8.6.2 step 3);
  * derives `classwords_per_codeword` (= classbook dimensions),
    `n_to_read`, `partitions_to_read`, returning the zeroed vectors when
    `n_to_read == 0`;
  * on pass 0 reads each partition's classification from the classbook
    in scalar context, unpacking `classwords_per_codeword`
    classifications per codeword by the descending integer-divide /
    integer-modulo by `residue_classifications` (§8.6.2 steps 9..12);
  * over passes 0..=7 looks up each partition's stage-`pass` value book
    and, if not `unused`, decodes the partition into the output in VQ
    context, *accumulating* (`+=`) per §8.6.2 step 19 so cascade stages
    stack;
  * format 0 (§8.6.3) scatters element `j` to `offset + i + j×step`;
    format 1 (§8.6.4) appends contiguously; format 2 (§8.6.5) decodes a
    single interleaved vector of length `ch × blocksize/2` as a format-1
    decode then de-interleaves `v[i×ch + j] → output[j][i]`, with the
    all-`do not decode` short-circuit;
  * treats end-of-packet mid-decode as the §8.6.2 nominal occurrence —
    decode stops and returns the vectors-so-far instead of erroring.
  Structured [`ResidueError`] variants (`UnsupportedFormat`,
  `ClassbookOutOfRange`, `ValueBookOutOfRange`, `ValueBookHasNoLookup`,
  `ZeroClasswordsPerCodeword`, `Format0PartitionNotDivisible`,
  `Huffman`, `Vq`).
* [`Floor1Decoder`] decodes the per-packet floor type 1 payload (Vorbis
  I §7.2.3 + §7.2.4) into a linear-domain spectral envelope.
  [`Floor1Decoder::new`] reconstructs the full `[floor1_X_list]` —
  prepending the implicit endpoints `0` and `2^rangebits` — validates the
  §7.2.2 undecodability clauses (multiplier in `1..=4`, `[floor1_values]`
  ≤ 65, x-list uniqueness, master/sub-book indices in range) and
  pre-builds every referenced codebook's Huffman tree once.
  [`Floor1Decoder::decode`] runs:
  * §7.2.3 packet decode — reads the `[nonzero]` flag (returning
    `FloorCurve::Unused` if clear), the two `ilog([range]-1)`-bit
    endpoint amplitudes, and per partition the master-book selector (only
    when `[cbits] > 0`) then the per-dimension sub-book scalar
    amplitudes, with a negative/`None` sub-book forcing a `0` Y with no
    bits consumed (§7.2.3 step 16/18);
  * §7.2.4 step 1 — unwraps the positive `[floor1_Y]` differences through
    iterative `render_point` line prediction into `[floor1_final_Y]` +
    `[floor1_step2_flag]`, with the suggested `[0, range)` clamp;
  * §7.2.4 step 2 — sorts the `(X, final_Y, flag)` triples by ascending
    X, renders the contiguous integer line segments via `render_line`,
    and substitutes each integer floor sample through the §10.1
    `floor1_inverse_dB_table` (`INVERSE_DB_TABLE`) for a length-`n`
    linear envelope;
  * treats end-of-packet mid-decode as the §7.2.3 nominal occurrence —
    returns `FloorCurve::Unused` as if `[nonzero]` had been clear.
  The integer geometry helpers [`low_neighbor`] / [`high_neighbor`]
  (§9.2.4 / §9.2.5), [`render_point`] (§9.2.6) and [`render_line`]
  (§9.2.7) are public. Structured [`Floor1Error`] variants
  (`BookOutOfRange`, `BadMultiplier`, `TooManyValues`, `NonUniqueXList`,
  `Huffman`).
* [`Floor0Decoder`] decodes the per-packet floor type 0 payload (Vorbis
  I §6.2.2 + §6.2.3) into a linear-domain spectral envelope. The
  `Floor0Decoder::new` constructor validates the §6.2.1 / §6.2.3
  undecodability clauses (nonzero `order` / `bark_map_size` /
  `amplitude_bits`, non-empty `book_list`, every book index in range,
  every referenced book carries a VQ lookup table per §3.3) and
  pre-builds each value codebook's Huffman decision tree once.
  `Floor0Decoder::decode` runs:
  * §6.2.2 packet decode — reads `[amplitude]` (returning
    `Floor0Curve::Unused` if zero), reads `[booknumber]` in
    `ilog([floor0_number_of_books])` bits (mapping
    `booknumber >= floor0_book_list.len()` to `Unused` per the
    nominal-occurrence rule on reserved values), then loops decoding VQ
    vectors from the selected value book — each vector has the running
    `[last]` accumulator added before concatenation (§6.2.2 steps 6..9)
    and `[last]` is then reset to the just-decoded vector's tail.
  * §6.2.3 curve computation — builds a Bark-scale `map[i]` per the
    post-errata `bark(x) = 13.1·atan(.00074x) + 2.24·atan(.0000000185x²) +
    .0001x` formula, then synthesises the LSP curve via the order-parity
    `[p]`/`[q]` product at `[ω] = π·map[i]/bark_map_size` and applies the
    `exp(.11512925 · …)` log→linear amplitude transform, replicating each
    synthesis value across all consecutive output bins whose `map[i]`
    matches the current synthesis bin (§6.2.3 step 8 `[iteration_condition]`
    chaining).
  * end-of-packet anywhere in §6.2.2 → `Floor0Curve::Unused`.
  The Bark-scale formula helper [`bark`] (re-exported at the crate root
  as `floor0_bark`) is public. Structured [`Floor0Error`] variants
  (`BookOutOfRange`, `EmptyBookList`, `ZeroOrder`, `ZeroBarkMapSize`,
  `ZeroAmplitudeBits`, `ValueBookHasNoLookup`, `Huffman`).
* [`vorbis_window`] builds the length-`n` Vorbis window for one audio
  frame (Vorbis I §1.3.2 / §4.3.1). It validates `n` is a positive power
  of two and (for long blocks) that `blocksize_0 <= n` is a power of two,
  computes `window_center = n/2`, then selects each edge's
  `(start, end, n)` per §4.3.1 steps 2..3 — the `n/4 ± blocksize_0/4`
  hybrid ramp when `blockflag` is set and the matching neighbour flag is
  clear, else the full-half-block `0..center` / `center..n` ramp — and
  fills the zero lead-in (step 4), the rising slope edge (step 5), the
  ones plateau (step 6), the `+π/2`-shifted falling slope edge (step 7),
  and the zero tail (step 8). The bare slope `y =
  sin(π/2·sin²((x+0.5)/n·π))` is exposed as [`slope`]; per-edge fills use
  the §4.3.1 `…·π/2` quarter-period argument directly. Structured
  [`WindowError`] variants (`NotPowerOfTwo`, `ShortBlockTooLarge`).
* [`inverse_couple_all`] runs the §4.3.5 inverse-coupling loop over a
  slice of per-channel residue vectors, applying every coupling step in
  **descending** order (`coupling_steps-1 … 0`). Each step decouples its
  `(magnitude_channel, angle_channel)` pair in place via
  [`inverse_couple`], which applies the [`couple_scalar`] four-quadrant
  square-polar → Cartesian rule (§4.3.5 step 3) element by element. The
  driver range-checks each step's channel indices against the
  residue-vector count and rejects a step that names the same channel for
  both magnitude and angle. Structured [`CouplingError`] variants
  (`ChannelOutOfRange`, `SameChannel`).
* [`nonzero_propagate`] runs the §4.3.3 "nonzero vector propagate" loop in
  **ascending** coupling-step order: for each `(magnitude_channel,
  angle_channel)` pair, if either member's `no_residue` flag is `false`
  (used) it forces *both* members' flags `false`. Ascending vs descending
  is immaterial here (the loop body only ever clears flags and a cleared
  flag stays cleared) but the spec text is ascending so we follow it. The
  driver range-checks each step's channel indices against the
  `no_residue` slice length. Structured [`PacketError`] variants
  (`ChannelOutOfRange`, plus the dot-product-driver variants below).
* [`dot_product`] computes one channel's §4.3.6 element-wise
  (Hadamard / "dot product" in the spec's terminology) product of a
  length-`n/2` floor curve and a length-`n/2` residue vector into a
  caller-supplied length-`n/2` spectrum buffer. Length mismatches panic
  per the §4.3.6 invariant that every per-channel vector is exactly `n/2`
  long. [`dot_product_all`] is the per-channel driver: it accepts
  `floors: &[Option<Vec<f32>>]` where `None` marks a channel whose floor
  returned `'unused'` (and whose §4.3.3 `no_residue` survived coupling
  propagation), emits the all-zero spectrum of length `n/2` for those
  channels per §4.3.3, and runs [`dot_product`] for the used channels.
  Structured [`PacketError`] variants (`ChannelCountMismatch`,
  `VectorLength` carrying [`VectorKind::Floor`] / [`VectorKind::Residue`]).
* [`read_packet_header`] reads the §4.3.1 audio-packet prelude from an
  LSB-first [`BitReaderLsb`] given the stream's parsed
  [`VorbisSetupHeader`] (only `setup.modes` is consulted) and the two
  blocksizes (`blocksize_0` / `blocksize_1`) from the identification
  header (§4.2.2). It validates the 1-bit `[packet_type]` == 0 reject
  path (§4.3 "must ignore" rule, surfaced as
  [`PacketError::NonAudioPacketType`]), reads the
  `ilog([vorbis_mode_count] - 1)`-bit `[mode_number]` with OOB validation
  (§9.2.1 `ilog`: zero bits when `mode_count == 1`), resolves the
  per-frame blocksize `[n]` from the selected mode's `blockflag`, and —
  for long blocks only — reads the two 1-bit window flags
  `[previous_window_flag]` and `[next_window_flag]` (§4.3.1 step 4a.i/ii;
  short blocks always reuse the symmetric short shape per step 4b and do
  not transmit these bits). Returns an [`AudioPacketHeader`] carrying the
  resolved `(mode_number, blockflag, n, previous_window_flag,
  next_window_flag)`. [`AudioPacketHeader::build_window`] then drives the
  existing [`vorbis_window`] builder with the resolved fields. EOF anywhere
  in §4.3.1 is the spec-mandated fatal path (§4.3.1 closing note),
  surfaced as [`PacketError::UnexpectedEndOfPacket`] with per-sub-step
  granularity via [`PacketHeaderStage`] (`PacketType` / `ModeNumber` /
  `PreviousWindowFlag` / `NextWindowFlag`). Additional `PacketError`
  variants: [`PacketError::BadModeNumber`] (mode_number ≥ mode_count),
  [`PacketError::EmptyModeList`] (defensive caller-bug guard for a setup
  header with zero modes).
* 219 unit tests in total: 16 cover §4.2.2, 22 cover §5, 18 cover §3
  codebook-header parse, 13 cover §3.2.1 Huffman tree, 28 cover
  §4.2.4 setup-header walker, 16 cover §3.2.1 / §3.3 VQ unpack, 15 cover
  §8.6 residue decode, 18 cover §7.2 floor 1 decode, 18 cover §6.2
  floor 0 decode, 20 cover §1.3.2 / §4.3.1 window generation +
  §4.3.5 inverse coupling, 19 cover §4.3.3 nonzero-vector propagate +
  §4.3.6 dot product, and **16 new round-13 tests cover §4.3.1 packet-prelude
  reading**: single-mode short-block (zero mode bits, 1 bit total),
  single-mode long-block (zero mode bits + 2 window flags),
  two-mode-one-bit-`mode_number` short and long paths, three-mode-two-bit
  `mode_number` long path, non-audio-packet_type reject, out-of-range
  `mode_number` reject, empty-mode-list defensive guard, EOF on
  `packet_type` (empty stream), EOF on `mode_number` (130 modes →
  ilog(129)=8-bit read past 7 remaining bits), EOF on
  `previous_window_flag` (65 modes → ilog(64)=7-bit read consumes the
  byte), EOF on `next_window_flag` (33 modes → ilog(32)=6-bit read +
  prev-flag leaves 0 bits), `build_window` short-block matches direct
  `vorbis_window` call, `build_window` long-block hybrid-left matches +
  confirms zero lead-in, `WindowError` propagation through `build_window`
  (non-power-of-two `n`), and the mode-blockflag-driven blocksize
  selection across mode 0 short / mode 1 long.

### What does not yet work

* **§4.3.7 IMDCT normalization scalar — documented docs gap.** The
  Vorbis-specific IMDCT normalization scalar that maps the bare cosine-
  summation kernel to oggdec-bit-equivalent PCM is the only piece of
  the §4.3 pipeline still pinned to a deferred-fixture knob. As of
  round 17 the §4.3.7 kernel itself and the §4.3.6 window
  multiplication are both wired into the per-packet driver via
  [`decode_audio_packet_windowed`] and [`apply_imdct_and_window`]; the
  Vorbis-specific normalization constant is exposed as an
  `imdct_scale: f32` argument that defaults to caller-supplied. The
  IMDCT cross-reference document
  (`docs/audio/vorbis/imdct-cross-reference.md` §"Vorbis-specific
  parameters" item 5) notes the constant "falls out of matching the
  fixture traces" — pinning it requires the staged fixture traces under
  `docs/audio/vorbis/fixtures/<case>/trace.txt` to extend through the
  post-IMDCT trace point. Until then a caller passing `1.0` gets the
  bare un-normalized kernel output × window; passing the
  fixture-derived constant once it exists is a one-line site change.
* **§4.3.8 overlap-add — landed round 15, integrated round 17.**
  [`OverlapAdd`] is the IMDCT-independent standalone primitive. The
  round-17 [`decode_audio_packet_windowed`] / [`apply_imdct_and_window`]
  entry points now produce exactly the windowed time-domain frames
  this primitive expects, so a per-channel
  [`OverlapAdd::push_frame(frame)`] call chain takes the round-17
  output the rest of the way to PCM.
* Mapping submap channel routing at packet time (which channels feed
  which residue / floor via `[vorbis_mapping_mux]`) — **landed round
  14**: [`audio::decode_audio_packet_pre_imdct`] walks `mapping.mux[ch]`
  for the §4.3.2 floor iteration and gathers per-submap channel bundles
  for the §4.3.4 residue iteration.
* Top-level audio-packet decode (§4.3.2..§4.3.9) tying floor + residue +
  window + coupling + dot-product + MDCT + overlap-add together. **Round
  14 closed every stage up to §4.3.6** via
  [`audio::decode_audio_packet_pre_imdct`] (per-channel length-`n/2`
  pre-IMDCT spectra). **Round 17 closes §4.3.7 + §4.3.6 windowing** via
  [`audio::decode_audio_packet_windowed`] / [`decode_one_packet_windowed`]
  / [`apply_imdct_and_window`] (per-channel length-`n` windowed
  time-domain frames). §4.3.8 [`OverlapAdd::push_frame`] consumes each
  windowed frame and emits PCM. §4.3.9 channel-order rearrangement is a
  presentation concern handled above the codec. The legacy
  [`decode_packet`] / [`audio::decode_one_packet`] entry points are
  preserved with the §4.3.7 boundary stop (`AudioPacketError::ImdctStage`)
  for callers that depend on it. Only the fixture-derived
  `imdct_scale` constant (deferred-normalization knob) remains.
* Ogg framing (RFC 3533 + Vorbis I §A) — the parsers are currently
  bring-your-own-packet. Consuming an Ogg-encapsulated stream needs
  to be wired up via `oxideav-ogg`.
* `METADATA_BLOCK_PICTURE` base64 + FLAC-PICTURE block decoding (see
  trace-doc §2.3) — the comment parser returns the raw base64 string
  as a comment value; FLAC-PICTURE decoding belongs in a higher-level
  consumer.
* No [`oxideav_core::Decoder`] / [`oxideav_core::Encoder`] is
  registered yet; the top-level [`decode_packet`] drives §4.3.2..§4.3.6
  and then returns [`AudioPacketError::ImdctStage`] at the IMDCT
  boundary.

## Clean-room sources

Rounds 1 — 14 were implemented against, and only against:

* `docs/audio/vorbis/Vorbis_I_spec.pdf` — Xiph.Org Vorbis I
  Specification, 2020-07-04 revision. Round 1 used §2 Bitpacking
  Convention, §4.2.1 Common header decode, §4.2.2 Identification
  header. Round 2 used §4.2 (end-of-packet handling), §4.2.1, §4.2.3,
  §5.1, §5.2.1 (structure / decoder pseudocode), §5.2.2 (content
  vector format), §5.2.3 (encoder-side recap). Round 3 used §3.1
  (codebook overview), §3.2.1 (codebook decode algorithm), §9.2.1
  (`ilog`), §9.2.2 (`float32_unpack`), §9.2.3 (`lookup1_values`).
  Round 4 used §3.2.1 "Huffman decision tree representation" (the
  worked-example codeword table for lengths `[2 4 4 4 4 2 3 3]` plus
  the underspecified / overspecified discussion), the §3.2.1 errata
  20150226 "Single entry codebooks" addendum, and §3.3 "Use of the
  codebook abstraction" (decode-time bit-walking semantics + the
  end-of-packet condition). Round 5 used §4.2.4 "Setup header"
  (Codebooks / Time domain transforms / Floors / Residues subsections,
  the "Time domain transforms" must-be-zero rejection, the floor-type
  > 1 and residue-type > 2 rejections, and the in-spec floor 0 / floor
  1 / residue branches), §6.2.1 "Floor 0 header decode", §7.2.2 "Floor
  1 header decode" (steps 1..23), and §8.6.1 "Residue header decode"
  (the begin/end/partition_size/classifications/classbook fields, the
  per-classification cascade bitmap with `high_bits = read 5 bits if
  bitflag`, and the conditional per-stage book reads). Round 6 used
  §4.2.4 "Mappings" (steps 1..2 inclusive of the `mapping_type != 0`
  rejection, the optional submaps and square-polar coupling subblocks,
  the `ilog(audio_channels - 1)`-bit magnitude/angle channel reads,
  the "magnitude != angle and both < audio_channels" validation
  paragraph, the 2-bit reserved field, the `mux[ch]` reads gated on
  `submaps > 1` with the OOB check, and the per-submap placeholder +
  floor + residue index reads with their respective OOB checks),
  §4.2.4 "Modes" (the per-mode blockflag / windowtype / transformtype /
  mapping reads plus step 2e's `windowtype == 0`, `transformtype == 0`,
  and `mapping < mapping_count` enforcement), and §4.2.4 "Modes"
  step 3 (the trailing framing-flag requirement). §9.2.1 `ilog` is
  reused via [`crate::codebook::ilog`]. Round 7 used §3.2.1 "VQ
  lookup table vector representation" introductory paragraph (the
  eight values consumed by the unpack: `codebook_multiplicands`,
  `codebook_minimum_value`, `codebook_delta_value`,
  `codebook_sequence_p`, `codebook_lookup_type`, `codebook_entries`,
  `codebook_dimensions`, `codebook_lookup_values`), §3.2.1 "Vector
  value decode: Lookup type 1" (lattice mixed-base permutation
  pseudocode, steps 1..8), §3.2.1 "Vector value decode: Lookup type
  2" (tessellation one-to-one pseudocode, steps 1..7), and §3.3
  "Use of the codebook abstraction" — the explicit prohibition on
  requesting a VQ value out of a `lookup_type = 0` codebook ("even
  in a case where a vector of dimension one … is an error condition
  rendering the packet undecodable"), and the entry-index → VQ
  vector hand-off after a successful tree walk. §9.2.3
  `lookup1_values` is reused via [`crate::codebook::lookup1_values`]
  for the type-1 shape cross-check. Round 8 used §8.6.2 "packet decode"
  (the begin/end limiting steps 1..5 including the format-2 `actual_size
  = actual_size * ch` rule, the `classwords_per_codeword` / `n_to_read`
  / `partitions_to_read` convenience values, the `n_to_read == 0`
  early-out, the pass 0..=7 loop, the pass-0 classbook scalar read with
  the descending modulo/divide classification unpack at steps 9..12, the
  step 13..20 per-partition VQ decode loop with the `vqbook` "unused"
  skip, and the "end-of-packet … is to be considered a nominal
  occurrence" clause), §8.6.3 "format 0 specifics" (`step = n /
  codebook_dimensions`, the `offset + i + j*step` scatter), §8.6.4
  "format 1 specifics" (the contiguous `offset + i` append loop), and
  §8.6.5 "format 2 specifics" (the all-`do not decode` short-circuit,
  the single `ch*n` interleaved format-1 decode, and the `v[i*ch + j] →
  output[j][i]` de-interleave). §8.6.1's undecodability clauses ("any
  codebook number greater than the maximum numbered codebook … renders
  the stream undecodable" and "all codebooks in array [residue books]
  are required to have a value mapping") were used for the construction
  validation. §3.2.1 / §3.3 are reused via [`crate::vq::unpack_vector`]
  and [`crate::huffman::HuffmanTree`] for the VQ-context partition reads
  and the classbook scalar reads. Round 9 used §7.2.3 "packet decode"
  (the `[nonzero]` flag at step 1, the `[range]` table lookup, the two
  `ilog([range]-1)`-bit endpoint amplitudes at steps 2..3, the
  per-partition steps 5..19 master-book selector + `cval & csub` /
  `cval >>= cbits` sub-book cascade + the negative-book zero-Y branch,
  and the closing "end-of-packet … nominal occurrence → return unused"
  note), §7.2.4 "curve computation" (step 1 amplitude synthesis with the
  `render_point` prediction, `highroom`/`lowroom`/`room` wrap arithmetic,
  the val-vs-room and odd/even branches, the suggested `[0, range)`
  clamp; step 2 curve synthesis with the ascending-X sort, the
  `render_line` segment chaining, the `hx < n` tail extension, and the
  `floor1_inverse_dB_table` substitution), §7.2.1 "model" (the
  iterative-prediction narrative), §9.2.4 "low neighbor", §9.2.5 "high
  neighbor", §9.2.6 "render point" (the integer line-solve), §9.2.7
  "render line" (the Bresenham-style integer line drawing with
  toward-zero division), and §10.1 "floor1 inverse dB table" (the
  256-element static table transcribed verbatim). §9.2.1 `ilog` is
  reused via [`crate::codebook::ilog`]; the scalar codebook reads use
  [`crate::huffman::HuffmanTree`]. Round 10 used §6.1 "Overview" (the
  LSP narrative), §6.2.1 "header decode" (the seven structural fields
  including the closing note that "any element of the array
  `[floor0_book_list]` that is greater than the maximum codebook number
  for this bitstream is an error condition that also renders the stream
  undecodable"), §6.2.2 "packet decode" (step 1 `[amplitude]` =
  `[floor0_amplitude_bits]` bits, step 2 `amplitude > 0` gating, step 4
  `[booknumber]` = `ilog([floor0_number_of_books])` bits with the
  alternative `ilog([floor0_number_of_books] - 1)` storage note
  explicitly declined in favour of the spec-literal reading, step 5
  reserved-value → undecodable, steps 6..11 the `[last]` carry / VQ
  vector concat loop, step 12 done, the "extra values are not used and
  may be ignored or discarded" over-read clause, the `[amplitude] == 0`
  ⇒ 'unused' rule, and the closing "end-of-packet condition during
  decode should be considered a nominal occurrence" note), §6.2.3
  "curve computation" (the `amplitude == 0` ⇒ all-zero shortcut, the
  `map[i] = min(bark_map_size - 1, foobar)` computation with
  `foobar = floor(bark(rate·i/(2n)) · bark_map_size / bark(.5·rate))`,
  the order-odd `[p]` / `[q]` formulas with the `(1 - cos²ω)` / `0.25`
  lead factors, the order-even formulas with the `(1 ± cos ω)/2` lead
  factors, step 4 `[linear_floor_value] = exp(.11512925 · (amplitude ·
  offset / ((2^bits - 1)·sqrt(p + q)) - offset))`, and steps 5..9 the
  `[iteration_condition]` chaining that replicates `[linear_floor_value]`
  across all consecutive output bins whose `map[i]` matches), and the
  §6.2.3 errata 20150227 "Bark scale computation" parenthesis-
  misplacement correction `bark(x) = 13.1·atan(.00074x) +
  2.24·atan(.0000000185x²) + .0001x`. §9.2.1 `ilog` is reused via
  [`crate::codebook::ilog`]; the VQ-context value-book reads use
  [`crate::vq::unpack_vector`] and [`crate::huffman::HuffmanTree`].
  Round 11 used §1.3.2 "Decode Procedure" (the decode-step narrative
  enumerating window-shape decode, the inverse-coupling step, the inverse
  monolithic MDCT, and the overlap/add stage) and its "Window shape
  decode (long windows only)" subsection (the slope function `y =
  sin(.5·π·sin²((x + .5)/n·π))`, the equal-sized and long/short overlap
  illustrations, and the `previous_window_flag`/`next_window_flag`
  redundancy paragraph), §4.3 "Audio packet decode and synthesis"
  (intro + the per-stage narrative: floor decode, residue decode, inverse
  channel coupling "converting square polar … back to Cartesian", the
  floor/residue dot product, the inverse MDCT, overlap/add "3/4 point of
  the previous window aligned with the 1/4 point of the current window",
  cache-right-hand-data, and the
  `window_blocksize(prev)/4 + window_blocksize(cur)/4` return-length
  formula), §4.3.1 "packet type, mode and window decode" (steps 1..4
  including the long-block `previous_window_flag`/`next_window_flag`
  reads at step 4a.i/ii, the short-block always-same-shape rule at step
  4b, and the eight-step window-generation procedure: `window_center =
  n/2`; the step-2 left-edge `(left_window_start, left_window_end,
  left_n)` selection between the `n/4 ± blocksize_0/4` hybrid ramp and
  the `0..window_center`/`n/2` full ramp; the symmetric step-3 right-edge
  selection between `n*3/4 ± blocksize_0/4` and `window_center..n`; the
  step-4 zero lead-in; the step-5 rising-edge fill `window([i]) =
  sin(π/2·sin²(([i]-left_window_start+0.5)/left_n·π/2))`; the step-6
  ones plateau; the step-7 falling-edge fill with the `… + π/2` phase
  shift; and the step-8 zero tail; plus the end-of-packet
  error-vs-nominal-occurrence note), and §4.3.5 "inverse coupling" (the
  descending `[vorbis_mapping_coupling_steps]-1 … 0` loop over
  magnitude/angle vector pairs and the step-3 four-quadrant
  `[new_M]`/`[new_A]` square-polar → Cartesian rule). The
  [`MappingCouplingStep`] `(magnitude_channel, angle_channel)` pairs from
  §4.2.4 (round 6) feed the inverse-coupling driver. Round 12 used §4.3
  "Audio packet decode and synthesis" intro narrative (the per-stage
  enumeration: floor decode → nonzero-vector propagation → residue decode
  → inverse coupling → floor/residue dot product → inverse MDCT →
  overlap/add), §4.3.3 "nonzero vector propagate" (the full
  for-each-`[i]`-from-`0...[vorbis_mapping_coupling_steps]-1` ascending
  loop body: "if either `[no_residue]` entry for channel
  (`[vorbis_mapping_magnitude]` element `[i]`) or channel
  (`[vorbis_mapping_angle]` element `[i]`) are set to false, then both
  must be set to false"; plus the §4.3.2 step 6 narrative that establishes
  the per-channel `[no_residue]` flag from the floor decode's
  `'unused'`/non-`'unused'` return), and §4.3.6 "dot product" (the
  per-channel element-wise "multiply each element of the floor curve by
  each element of that channel's residue vector" rule and its closing
  sentence "the produced vectors are the length `[n]/2` audio spectrum
  for each channel" — making explicit that this is element-wise / Hadamard
  product, not a scalar inner product, despite the spec's "dot product"
  naming). §4.3.7 "inverse MDCT" was **read and explicitly NOT
  implemented**: the section defers the MDCT definition entirely to
  external reference `[1]` ("A detailed description of the MDCT is
  available in [1]"), which the workspace clean-room policy bars. No
  external MDCT formula, normalization convention, or reference
  implementation was consulted; the docs gap is recorded in the "What
  does not yet work" section above. The [`MappingCouplingStep`]
  `(magnitude_channel, angle_channel)` pairs from §4.2.4 (round 6) feed
  the §4.3.3 propagation driver, and the per-channel `Option<Vec<f32>>`
  representation of floor curves is the natural lift of [`FloorCurve`] /
  [`Floor0Curve`]'s `Unused` / `Curve(Vec<f32>)` distinction (rounds
  9 / 10). Round 13 used §4.3 introductory text (the "First step of audio
  packet decode is to read and verify the packet type. A non-audio packet
  when audio is expected indicates stream corruption or a non-compliant
  stream. The decoder must ignore the packet and not attempt decoding it
  to audio" reject rule) and §4.3.1 "packet type, mode and window decode"
  steps 1..4 plus its closing note: step 1's 1-bit `[packet_type]` with the
  `== 0` audio check; step 2's `ilog([vorbis_mode_count] - 1)`-bit
  `[mode_number]`; step 3's blocksize resolution (`n = blockflag ?
  blocksize_1 : blocksize_0`); step 4a's long-block `[previous_window_flag]`
  + `[next_window_flag]` reads; step 4b's "if this is a short window, the
  window is always the same short-window shape" rule that skips the window
  flags; and the closing note "An end-of-packet condition up to this point
  should be considered an error that discards this packet from the stream
  … An end of packet condition past this point is to be considered a
  possible nominal occurrence" which makes §4.3.1 the only §4.3 stage with
  fatal EOF semantics. §9.2.1 `ilog` is reused via
  [`crate::codebook::ilog`]; the resolved window-flag + blockflag + `n`
  fields feed [`crate::synthesis::vorbis_window`] (round 11) through
  [`AudioPacketHeader::build_window`]. Round 14 wired §4.3 itself — the
  "Audio packet decode and synthesis" outer prose plus §4.3.2 "floor
  curve decode" (the channel-order iteration, the `submap_number =
  mux[i]` / `floor_number = vorbis_submap_floor[submap_number]` lookup
  per step 1/2, the type-0 → §6.2.2 / type-1 → §7.2.3 dispatch per
  step 3/4, the step-5 "save the needed decoded floor information for
  later synthesis" hand-off into the §4.3.6 dot-product, the step-6
  `no_residue` flag toggle, and the closing-note "An end-of-packet
  condition during floor decode shall result in packet decode zeroing
  all channel output vectors and skipping to the add/overlap output
  stage" surfaced as [`AudioPacketOutcome::Zeroed`]); §4.3.4 "residue
  decode" (the submap-order iteration plus the steps 1..7 channel-gather
  / `do_not_decode_flag` build / scatter loop); and §4.3.7's "[1]"
  reference deferral (the documented docs gap, surfaced as
  [`AudioPacketError::ImdctStage`]).
* `docs/audio/vorbis/vorbis-fixtures-and-traces.md` — clean-room
  trace-corpus document. Round 2 referenced §2.2 (`mono-44100-q5-typical`
  and `with-vorbis-comment-tags` `VORBIS_HEADER_COMMENT` /
  `VORBIS_COMMENT_ENTRY` shapes), §2.3 (`METADATA_BLOCK_PICTURE`
  convention), §9 (trace event vocabulary). Round 3 referenced §3
  (the `CODEBOOK` event shape — `book_idx dimensions entries
  ordered sparse lookup_type value_bits sequence_p` field set).
  Round 5 referenced §2.4 (`VORBIS_HEADER_SETUP` event shape —
  `codebook_count`, `time_count`, `floor_count`, `residue_count`,
  `mapping_count`, `mode_count`, `framing_flag`), §4 (`FLOOR_CONFIG`
  field shape — `floor_idx`, `type`, plus `partitions`,
  `multiplier`, `rangebits`, `x_list_count` for type 1), and §5
  (`RESIDUE_CONFIG` field shape — `residue_idx`, `type`, `begin`,
  `end`, `partition_size`, `classifications`, `classbook`). Round 6
  referenced §6 (`MAPPING_CONFIG` shape — `mapping_idx`, `type`,
  `submaps`, `coupling_steps`, `magnitude`, `angle`, `floor`,
  `residue`; including the §6 narrative confirming "stereo libvorbis
  output always uses one coupling step `(magnitude=L, angle=R)`,
  mono streams have `coupling_steps=0` and `submaps=1`, 5.1 streams
  use `submaps=2` with `mux=[0,0,0,0,0,1]` routing the LFE on its
  own submap") and §7 (`MODE_CONFIG` shape — `mode_idx`,
  `blockflag`, `windowtype`, `transformtype`, `mapping`, plus the
  trace narrative that mode 0 is the short-block mode and mode 1
  is the long-block mode in every libvorbis stream).
* `docs/audio/vorbis/fixtures/{mono-44100-q5-typical,with-vorbis-comment-tags}/trace.txt`
  — only as the source for the field-level shape of the test
  fixtures.

No external library source was consulted, quoted, paraphrased, or
used as a cross-check oracle, in accordance with the workspace
clean-room policy. The forbidden-list for this crate includes every
existing Vorbis implementation (the Xiph reference encoder/decoder,
Tremor, lewton, FFmpeg's Vorbis decoder, every third-party Rust
crate that wraps or implements the format).

[`parse_identification_header`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/identification/fn.parse_identification_header.html
[`VorbisIdentificationHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/identification/struct.VorbisIdentificationHeader.html
[`parse_comment_header`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/comment/fn.parse_comment_header.html
[`VorbisCommentHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/comment/struct.VorbisCommentHeader.html
[`split_key_value`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/comment/fn.split_key_value.html
[`VorbisCommentHeader::key_value_iter`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/comment/struct.VorbisCommentHeader.html#method.key_value_iter
[`parse_codebook`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/fn.parse_codebook.html
[`VorbisCodebook`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/struct.VorbisCodebook.html
[`VqLookup`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/enum.VqLookup.html
[`UNUSED_ENTRY`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/constant.UNUSED_ENTRY.html
[`ilog`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/fn.ilog.html
[`float32_unpack`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/fn.float32_unpack.html
[`lookup1_values`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/codebook/fn.lookup1_values.html
[`HuffmanTree::from_codebook`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/struct.HuffmanTree.html#method.from_codebook
[`HuffmanTree::from_lengths`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/struct.HuffmanTree.html#method.from_lengths
[`HuffmanTree::decode_entry`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/struct.HuffmanTree.html#method.decode_entry
[`HuffmanBuildError::SingleEntryWrongLength`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/enum.BuildError.html#variant.SingleEntryWrongLength
[`HuffmanBuildError::UnderspecifiedTree`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/enum.BuildError.html#variant.UnderspecifiedTree
[`HuffmanBuildError::OverspecifiedTree`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/enum.BuildError.html#variant.OverspecifiedTree
[`HuffmanBuildError::InvalidLength`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/enum.BuildError.html#variant.InvalidLength
[`HuffmanBuildError::EmptyTree`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/huffman/enum.BuildError.html#variant.EmptyTree
[`BitReaderLsb`]: https://docs.rs/oxideav-core/latest/oxideav_core/bits/struct.BitReaderLsb.html
[`oxideav_core::bits::BitReaderLsb`]: https://docs.rs/oxideav-core/latest/oxideav_core/bits/struct.BitReaderLsb.html
[`oxideav_core::Decoder`]: https://docs.rs/oxideav-core/latest/oxideav_core/trait.Decoder.html
[`oxideav_core::Encoder`]: https://docs.rs/oxideav-core/latest/oxideav_core/trait.Encoder.html
[`parse_setup_header`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/fn.parse_setup_header.html
[`parse_setup_header_body`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/fn.parse_setup_header_body.html
[`Floor0Header`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/struct.Floor0Header.html
[`Floor1Header`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/struct.Floor1Header.html
[`ResidueHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/struct.ResidueHeader.html
[`MappingHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/struct.MappingHeader.html
[`ModeHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/struct.ModeHeader.html
[`unpack_vector`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/vq/fn.unpack_vector.html
[`VqUnpackError::NoVectorForType0`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/vq/enum.UnpackError.html#variant.NoVectorForType0
[`ResidueDecoder`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/residue/struct.ResidueDecoder.html
[`ResidueDecoder::new`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/residue/struct.ResidueDecoder.html#method.new
[`ResidueDecoder::decode`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/residue/struct.ResidueDecoder.html#method.decode
[`ResidueError`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/residue/enum.ResidueError.html
[`Floor1Decoder`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/floor1/struct.Floor1Decoder.html
[`Floor1Decoder::new`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/floor1/struct.Floor1Decoder.html#method.new
[`Floor1Decoder::decode`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/floor1/struct.Floor1Decoder.html#method.decode
[`Floor1Error`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/floor1/enum.Floor1Error.html
[`low_neighbor`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/floor1/fn.low_neighbor.html
[`high_neighbor`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/floor1/fn.high_neighbor.html
[`render_point`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/floor1/fn.render_point.html
[`render_line`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/floor1/fn.render_line.html
[`Floor0Decoder`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/floor0/struct.Floor0Decoder.html
[`Floor0Error`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/floor0/enum.Floor0Error.html
[`bark`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/floor0/fn.bark.html
[`vorbis_window`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/synthesis/fn.vorbis_window.html
[`slope`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/synthesis/fn.slope.html
[`WindowError`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/synthesis/enum.WindowError.html
[`inverse_couple_all`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/synthesis/fn.inverse_couple_all.html
[`inverse_couple`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/synthesis/fn.inverse_couple.html
[`couple_scalar`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/synthesis/fn.couple_scalar.html
[`CouplingError`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/synthesis/enum.CouplingError.html
[`MappingCouplingStep`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/struct.MappingCouplingStep.html
[`nonzero_propagate`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/fn.nonzero_propagate.html
[`dot_product`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/fn.dot_product.html
[`dot_product_all`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/fn.dot_product_all.html
[`PacketError`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/enum.PacketError.html
[`PacketError::NonAudioPacketType`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/enum.PacketError.html#variant.NonAudioPacketType
[`PacketError::BadModeNumber`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/enum.PacketError.html#variant.BadModeNumber
[`PacketError::EmptyModeList`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/enum.PacketError.html#variant.EmptyModeList
[`PacketError::UnexpectedEndOfPacket`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/enum.PacketError.html#variant.UnexpectedEndOfPacket
[`PacketHeaderStage`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/enum.PacketHeaderStage.html
[`VectorKind::Floor`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/enum.VectorKind.html#variant.Floor
[`VectorKind::Residue`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/enum.VectorKind.html#variant.Residue
[`read_packet_header`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/fn.read_packet_header.html
[`AudioPacketHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/struct.AudioPacketHeader.html
[`AudioPacketHeader::build_window`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/packet/struct.AudioPacketHeader.html#method.build_window
[`VorbisSetupHeader`]: https://docs.rs/oxideav-vorbis/latest/oxideav_vorbis/setup/struct.VorbisSetupHeader.html

## License

MIT. See `LICENSE`.

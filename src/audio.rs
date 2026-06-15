//! Vorbis I audio-packet decode driver (§4.3.2 through §4.3.6).
//!
//! This module wires every per-packet stage of §4.3 that the Vorbis I spec
//! defines in its own body into a single top-level driver, stopping
//! cleanly at the §4.3.7 inverse-MDCT boundary (still gated on a docs gap
//! — the spec defers the MDCT definition entirely to external reference
//! `[1]`, which the workspace clean-room policy bars).
//!
//! The §4.3 decode-order is fixed (§4.3 "Audio packet decode and synthesis"):
//!
//! 1. **§4.3.1** packet type, mode and window decode
//!    ([`crate::packet::read_packet_header`] — called by this driver as
//!    its first step).
//! 2. **§4.3.2** floor curve decode — *channel order*. Per channel `i`:
//!    pick the submap via `mux[i]` (or always `0` when the mapping
//!    declared `submaps == 1`), pick the floor index from that submap's
//!    `(time_placeholder, floor, residue)` triple, and run the matching
//!    floor decoder ([`crate::floor1::Floor1Decoder`] /
//!    [`crate::floor0::Floor0Decoder`]). The result feeds the per-channel
//!    `[no_residue]` flag (§4.3.2 step 6: `'unused'` → `true`, else
//!    `false`). End-of-packet during floor decode is the §4.3.2 nominal
//!    occurrence — both floor implementations already return their
//!    `Unused` variant on EOF, and §4.3.2's closing note instructs the
//!    decoder to zero every channel output vector and skip straight to
//!    §4.3.8 overlap-add. This driver surfaces that signal as
//!    [`AudioPacketOutcome::Zeroed`] so the (still-pending) IMDCT
//!    boundary stage does the right thing.
//! 3. **§4.3.3** nonzero vector propagate ([`crate::packet::nonzero_propagate`]).
//! 4. **§4.3.4** residue decode — *submap order*. For each submap `i`:
//!    gather the channels whose `mux[channel] == i` (preserving channel
//!    order), build the per-bundle `do_not_decode_flag` from that
//!    submap's slice of `no_residue`, and run the submap's residue
//!    decoder ([`crate::residue::ResidueDecoder::decode`]). Scatter the
//!    bundle's per-channel vectors back into the global per-channel
//!    array (§4.3.4 step 7).
//! 5. **§4.3.5** inverse coupling
//!    ([`crate::synthesis::inverse_couple_all`], descending step order).
//! 6. **§4.3.6** dot product
//!    ([`crate::packet::dot_product_all`]) — every used channel's floor
//!    × residue product is its final length-`n/2` audio spectrum; every
//!    unused channel emits the all-zero spectrum (§4.3.3 closing note).
//! 7. **§4.3.7** inverse MDCT — **stopped here**. The driver hands the
//!    per-channel spectra to a placeholder that returns
//!    [`AudioPacketError::ImdctStage`] / [`AudioPacketOutcome::PreImdct`].
//!    The full IMDCT integration is pending an externally-cited
//!    reference (T. Sporer / K. Brandenburg / B. Edler) that the
//!    clean-room policy bars; see the crate README "What does not yet
//!    work" for the documented gap.
//!
//! # Decoder state
//!
//! Floor 0 / floor 1 / residue decoders all pay an up-front Huffman
//! tree-build cost; that cost belongs once per stream-setup, not once
//! per packet. [`AudioDecoderState`] caches the per-stream decoders
//! keyed by setup-header index so the driver constructs them only on
//! the first packet (or eagerly via [`AudioDecoderState::new`]).
//!
//! # §4.3.7 inverse-MDCT + §4.3.6 window wiring (round 17)
//!
//! As of round 17 the driver also exposes a "windowed time-domain frame"
//! entry point: [`decode_audio_packet_windowed`]. After the §4.3.2..§4.3.6
//! pipeline completes (the [`decode_audio_packet_pre_imdct`] body), each
//! per-channel length-`n/2` audio-spectrum vector is run through the
//! [`crate::imdct::imdct_naive`] direct cosine-summation kernel and the
//! resulting length-`n` time-domain frame is multiplied element-wise by
//! the §4.3.6 / §1.3.2 Vorbis window built once per packet via
//! [`AudioPacketHeader::build_window`]. The result is a length-`n`
//! windowed time-domain frame per channel, ready to feed straight into
//! the §4.3.8 [`crate::overlap::OverlapAdd`] primitive.
//!
//! The §4.3.7 IMDCT cross-reference document
//! (`docs/audio/vorbis/imdct-cross-reference.md`) closes the spec's
//! deferral to external reference `[1]` by observing that the IMDCT
//! kernel is generic DSP restated in three adjacent in-repo specs (ATSC
//! A/52 §7.9.4, ISO/IEC 14496-3 §4.6.x, IETF RFC 6716 §4.3.7) and giving
//! the canonical formula that [`crate::imdct`] implements. The only
//! Vorbis-specific piece still deferred is the **normalization scalar**
//! — the cross-reference notes the scalar "falls out of matching the
//! fixture traces," and the staged traces under
//! `docs/audio/vorbis/fixtures/<case>/trace.txt` do not yet log
//! post-IMDCT samples. [`decode_audio_packet_windowed`] takes an explicit
//! `imdct_scale: f32` parameter that callers plug a tentative value into;
//! a follow-up round pins it once the traces extend.
//!
//! # What this driver does NOT do
//!
//! * **Inverse MDCT normalization scalar (§4.3.7).** Documented docs gap;
//!   `imdct_scale: f32` is the deferred-normalization knob.
//! * **Overlap-add (§4.3.8).** Fully specified in the spec; this driver
//!   stops at the windowed-frame boundary so the caller can keep one
//!   [`crate::overlap::OverlapAdd`] instance per channel without the
//!   driver having to thread per-stream state.
//! * **Channel-order rearrangement (§4.3.9).** A presentation concern
//!   for the consumer; the driver returns per-channel vectors in
//!   bitstream order. The §4.3.9 mapping-type-0 speaker layout (what
//!   physical location each bitstream channel denotes) is exposed
//!   separately via [`crate::channel_order`]; the consumer permutes if
//!   it wants a non-Vorbis physical ordering.
//! * **Ogg framing.** The driver is bring-your-own-packet, just like the
//!   header parsers.

use crate::floor0::{Floor0Curve, Floor0Decoder, Floor0Error};
use crate::floor1::{Floor1Decoder, Floor1Error, FloorCurve};
use crate::imdct::{imdct_naive, ImdctError};
use crate::packet::{
    dot_product_all, nonzero_propagate, read_packet_header, AudioPacketHeader, PacketError,
};
use crate::residue::{ResidueDecoder, ResidueError};
use crate::setup::{FloorHeader, FloorKind, MappingHeader, VorbisSetupHeader};
use crate::synthesis::{
    inverse_couple_all, window_premultiply, CouplingError, WindowError, WindowPremultiplyError,
};
use oxideav_core::bits::BitReaderLsb;

/// Per-stream cache of every floor and residue decoder a stream may use,
/// indexed by setup-header position.
///
/// Built once from a parsed [`VorbisSetupHeader`] (eagerly, via
/// [`AudioDecoderState::new`]); reused across every audio packet for the
/// life of the stream. Each entry's construction validates the §6.2.1 /
/// §7.2.2 / §8.6.1 undecodability clauses up front, so per-packet decode
/// can assume well-formed decoders.
#[derive(Debug)]
pub struct AudioDecoderState {
    /// One decoder per setup-header floor entry, in stream order.
    floors: Vec<FloorDecoder>,
    /// One decoder per setup-header residue entry, in stream order.
    residues: Vec<ResidueDecoder>,
}

/// Tag for the floor decoder kind that backs a setup-header floor entry.
#[derive(Debug)]
enum FloorDecoder {
    Type0(Floor0Decoder),
    Type1(Floor1Decoder),
}

impl FloorDecoder {
    /// Run §4.3.2 step 3/4 (floor packet decode for one channel),
    /// returning `(curve_or_unused, no_residue_flag)`.
    fn decode(&self, reader: &mut BitReaderLsb<'_>, half_n: usize) -> (Option<Vec<f32>>, bool) {
        match self {
            FloorDecoder::Type0(d) => match d.decode(reader, half_n) {
                Floor0Curve::Unused => (None, true),
                Floor0Curve::Curve(c) => (Some(c), false),
            },
            FloorDecoder::Type1(d) => match d.decode(reader, half_n) {
                FloorCurve::Unused => (None, true),
                FloorCurve::Curve(c) => (Some(c), false),
            },
        }
    }
}

impl AudioDecoderState {
    /// Build the per-stream decoder cache from a parsed setup header.
    ///
    /// Constructs one [`Floor0Decoder`] or [`Floor1Decoder`] per floor
    /// entry (matching the entry's `FloorKind` discriminant), and one
    /// [`ResidueDecoder`] per residue entry; both call into the codebook
    /// table parsed by the setup-header walker.
    ///
    /// # Errors
    ///
    /// Any per-floor / per-residue construction error is surfaced
    /// verbatim through [`AudioPacketError`]: [`Floor0Error`] /
    /// [`Floor1Error`] for an invalid floor configuration (book OOB,
    /// non-VQ value book, zero order/map size/amplitude bits, etc.) and
    /// [`ResidueError`] for an invalid residue configuration (classbook
    /// OOB, zero classwords-per-codeword, value book without a VQ
    /// lookup, etc.).
    pub fn new(setup: &VorbisSetupHeader) -> Result<Self, AudioPacketError> {
        let mut floors = Vec::with_capacity(setup.floors.len());
        for (index, floor) in setup.floors.iter().enumerate() {
            floors.push(build_floor(index, floor, &setup.codebooks)?);
        }
        let mut residues = Vec::with_capacity(setup.residues.len());
        for (index, residue) in setup.residues.iter().enumerate() {
            let decoder = ResidueDecoder::new(residue, &setup.codebooks)
                .map_err(|source| AudioPacketError::ResidueBuild { index, source })?;
            residues.push(decoder);
        }
        Ok(Self { floors, residues })
    }

    /// The cached floor decoders, in setup-header order.
    pub fn floor_count(&self) -> usize {
        self.floors.len()
    }

    /// The cached residue decoders, in setup-header order.
    pub fn residue_count(&self) -> usize {
        self.residues.len()
    }
}

fn build_floor(
    index: usize,
    floor: &FloorHeader,
    codebooks: &[crate::codebook::VorbisCodebook],
) -> Result<FloorDecoder, AudioPacketError> {
    match &floor.kind {
        FloorKind::Type0(h) => {
            let decoder = Floor0Decoder::new(h, codebooks)
                .map_err(|source| AudioPacketError::Floor0Build { index, source })?;
            Ok(FloorDecoder::Type0(decoder))
        }
        FloorKind::Type1(h) => {
            let decoder = Floor1Decoder::new(h, codebooks)
                .map_err(|source| AudioPacketError::Floor1Build { index, source })?;
            Ok(FloorDecoder::Type1(decoder))
        }
    }
}

/// The outcome of running the §4.3.2..§4.3.6 driver over one audio packet.
///
/// The driver stops at the §4.3.7 inverse-MDCT boundary; downstream
/// callers receive either a per-channel pre-IMDCT spectrum bundle
/// ([`Self::PreImdct`]) or the §4.3.2 "zero every output vector"
/// short-circuit ([`Self::Zeroed`]).
#[derive(Debug, Clone, PartialEq)]
pub enum AudioPacketOutcome {
    /// The driver completed §4.3.2..§4.3.6 normally and produced one
    /// length-`n/2` audio-spectrum vector per channel, ready for the
    /// §4.3.7 inverse MDCT (currently a docs gap — see crate README).
    PreImdct {
        /// `[mode_number]` selected by §4.3.1.
        mode_number: u32,
        /// `[vorbis_mode_blockflag]` of the selected mode.
        blockflag: bool,
        /// `[n]` — the per-frame blocksize resolved from `blockflag`.
        n: usize,
        /// `[previous_window_flag]` (long-block-only; placeholder
        /// `false` for short blocks per §4.3.1 step 4b).
        previous_window_flag: bool,
        /// `[next_window_flag]` (long-block-only; placeholder `false`
        /// for short blocks).
        next_window_flag: bool,
        /// One length-`n/2` audio spectrum per channel, in bitstream
        /// channel order.
        spectra: Vec<Vec<f32>>,
    },
    /// §4.3.2's closing note fired: floor decode hit an end-of-packet
    /// condition, and §4.3.2 mandates "zeroing all channel output
    /// vectors and skipping to the add/overlap output stage." The
    /// driver returns this so the IMDCT boundary stage emits an
    /// all-zero PCM frame instead of attempting decode.
    Zeroed {
        /// `[mode_number]` selected by §4.3.1.
        mode_number: u32,
        /// `[vorbis_mode_blockflag]` of the selected mode.
        blockflag: bool,
        /// `[n]` — the per-frame blocksize resolved from `blockflag`.
        n: usize,
        /// `[previous_window_flag]` (long-block-only; placeholder
        /// `false` for short blocks).
        previous_window_flag: bool,
        /// `[next_window_flag]` (long-block-only; placeholder `false`
        /// for short blocks).
        next_window_flag: bool,
        /// Number of channels the per-channel zero vector should be
        /// emitted for. Each vector is length `n/2`; this is the count,
        /// not the data, because every value is `0.0f32` and callers
        /// generally allocate the buffer themselves.
        channels: usize,
    },
}

/// The outcome of running the §4.3.2..§4.3.6 driver, plus §4.3.7 IMDCT,
/// plus §4.3.6 window multiplication. Produced by
/// [`decode_audio_packet_windowed`].
///
/// The variant boundary mirrors [`AudioPacketOutcome`]: a normal frame
/// (`Windowed`) holds the per-channel length-`n` windowed time-domain
/// frames, ready to feed straight into the §4.3.8 overlap-add primitive;
/// the §4.3.2 short-circuit (`ZeroedWindowed`) emits per-channel all-zero
/// length-`n` frames (the IMDCT of zero is zero by linearity, times any
/// window is still zero).
#[derive(Debug, Clone, PartialEq)]
pub enum WindowedPacketOutcome {
    /// The driver completed §4.3.2..§4.3.6 normally, ran the §4.3.7 IMDCT
    /// per channel, and multiplied each length-`n` time-domain frame by
    /// the §4.3.6 / §1.3.2 Vorbis window. One windowed frame per channel,
    /// in bitstream channel order, each length `n`.
    Windowed {
        /// `[mode_number]` selected by §4.3.1.
        mode_number: u32,
        /// `[vorbis_mode_blockflag]` of the selected mode.
        blockflag: bool,
        /// `[n]` — the per-frame blocksize resolved from `blockflag`.
        n: usize,
        /// `[previous_window_flag]` (long-block-only; placeholder
        /// `false` for short blocks).
        previous_window_flag: bool,
        /// `[next_window_flag]` (long-block-only; placeholder `false`
        /// for short blocks).
        next_window_flag: bool,
        /// One length-`n` windowed time-domain frame per channel, in
        /// bitstream channel order. Each is `imdct(spectrum) ⊙ window`
        /// — element-wise multiplication of the §4.3.7 IMDCT output by
        /// the §4.3.6 / §1.3.2 window, scaled by the caller-supplied
        /// `imdct_scale` factor. Hand each one to a per-channel
        /// [`crate::overlap::OverlapAdd::push_frame`].
        frames: Vec<Vec<f32>>,
    },
    /// §4.3.2's closing note fired (see [`AudioPacketOutcome::Zeroed`]).
    /// The IMDCT of an all-zero spectrum is the all-zero frame, times
    /// any window is still all-zero — so this variant returns ready-to-
    /// overlap-add zero frames directly. Geometry is preserved so the
    /// caller's per-channel overlap-add state still advances correctly.
    ZeroedWindowed {
        /// `[mode_number]` selected by §4.3.1.
        mode_number: u32,
        /// `[vorbis_mode_blockflag]` of the selected mode.
        blockflag: bool,
        /// `[n]` — the per-frame blocksize resolved from `blockflag`.
        n: usize,
        /// `[previous_window_flag]` (long-block-only; placeholder).
        previous_window_flag: bool,
        /// `[next_window_flag]` (long-block-only; placeholder).
        next_window_flag: bool,
        /// One length-`n` all-zero windowed frame per channel.
        frames: Vec<Vec<f32>>,
    },
}

impl WindowedPacketOutcome {
    /// Convenience accessor: the resolved §4.3.1 header for either variant.
    #[must_use]
    pub fn header(&self) -> AudioPacketHeader {
        match *self {
            WindowedPacketOutcome::Windowed {
                mode_number,
                blockflag,
                n,
                previous_window_flag,
                next_window_flag,
                ..
            }
            | WindowedPacketOutcome::ZeroedWindowed {
                mode_number,
                blockflag,
                n,
                previous_window_flag,
                next_window_flag,
                ..
            } => AudioPacketHeader {
                mode_number,
                blockflag,
                n,
                previous_window_flag,
                next_window_flag,
            },
        }
    }

    /// Borrow the per-channel windowed-frame slice held by either variant.
    /// The `Windowed` variant returns the §4.3.7-then-§4.3.6 output;
    /// `ZeroedWindowed` returns the all-zero frames; both have one entry
    /// per channel and each entry is length `n`.
    #[must_use]
    pub fn frames(&self) -> &[Vec<f32>] {
        match self {
            WindowedPacketOutcome::Windowed { frames, .. }
            | WindowedPacketOutcome::ZeroedWindowed { frames, .. } => frames,
        }
    }
}

impl AudioPacketOutcome {
    /// Convenience accessor: the resolved §4.3.1 header for either variant.
    pub fn header(&self) -> AudioPacketHeader {
        match *self {
            AudioPacketOutcome::PreImdct {
                mode_number,
                blockflag,
                n,
                previous_window_flag,
                next_window_flag,
                ..
            }
            | AudioPacketOutcome::Zeroed {
                mode_number,
                blockflag,
                n,
                previous_window_flag,
                next_window_flag,
                ..
            } => AudioPacketHeader {
                mode_number,
                blockflag,
                n,
                previous_window_flag,
                next_window_flag,
            },
        }
    }
}

/// Run the §4.3.2..§4.3.6 audio-packet driver over a single packet,
/// stopping at the §4.3.7 inverse-MDCT boundary.
///
/// * `reader` — an LSB-first bit reader positioned at the first bit of
///   the audio packet (RFC-3533-stripped, Ogg-page-coalesced payload).
/// * `setup` — the stream's parsed setup header.
/// * `state` — the per-stream decoder cache built by
///   [`AudioDecoderState::new`].
/// * `audio_channels` — the stream's `[audio_channels]` from the
///   identification header.
/// * `blocksize_0` / `blocksize_1` — the two blocksizes from the
///   identification header.
///
/// Returns an [`AudioPacketOutcome`] holding either the per-channel
/// pre-IMDCT spectra (`PreImdct`) or the §4.3.2 zero-output short-circuit
/// (`Zeroed`). The bit reader is left at the position the IMDCT would
/// resume from (residue decode and below do not advance the reader past
/// their own consumption; coupling and dot product are pure transforms).
///
/// To finish the packet through PCM the caller would (when the docs gap
/// is closed): per used channel, run the §4.3.7 IMDCT of the spectrum,
/// element-multiply by the §4.3.1 window built via
/// [`AudioPacketHeader::build_window`](crate::packet::AudioPacketHeader::build_window),
/// and overlap-add into the previous packet's tail (§4.3.8). Until then,
/// the top-level [`decode_one_packet_into_zero_pcm`] entry point uses the
/// `Zeroed` short-circuit + an explicit `Imdct` not-implemented variant
/// to stop cleanly.
///
/// # Errors
///
/// * [`AudioPacketError::Header`] for any §4.3.1 prelude failure
///   ([`PacketError`]).
/// * [`AudioPacketError::BadModeMapping`] / `BadSubmapFloor` /
///   `BadSubmapResidue` for index-out-of-range conditions the setup
///   parser should already have rejected (defensive checks against
///   hand-built or corrupted state).
/// * [`AudioPacketError::Floor1`] / `Floor0` should not arise in
///   practice (the floor decoders return `Unused` rather than an error
///   on EOF), but the variants exist for symmetry.
/// * [`AudioPacketError::Residue`] for a residue decode failure
///   ([`ResidueError`]) other than nominal EOF (§8.6.2 EOF returns
///   work-so-far).
/// * [`AudioPacketError::Coupling`] for a §4.3.5 inverse-coupling
///   failure ([`CouplingError`]).
/// * [`AudioPacketError::Packet`] for a §4.3.3 / §4.3.6 driver failure
///   ([`PacketError`]).
pub fn decode_audio_packet_pre_imdct(
    reader: &mut BitReaderLsb<'_>,
    setup: &VorbisSetupHeader,
    state: &AudioDecoderState,
    audio_channels: u8,
    blocksize_0: usize,
    blocksize_1: usize,
) -> Result<AudioPacketOutcome, AudioPacketError> {
    // §4.3.1 — packet header.
    let header = read_packet_header(reader, setup, blocksize_0, blocksize_1)
        .map_err(AudioPacketError::Header)?;
    let channels = audio_channels as usize;
    let n = header.n;
    let half_n = n / 2;

    // §4.3.2 — pick the mapping for the selected mode.
    let mode = setup.modes[header.mode_number as usize];
    let mapping_index = mode.mapping as usize;
    let mapping = setup
        .mappings
        .get(mapping_index)
        .ok_or(AudioPacketError::BadModeMapping {
            mode_number: header.mode_number,
            mapping: mode.mapping,
            mapping_count: setup.mappings.len(),
        })?;

    // §4.3.2 — floors decoded in channel order. For each channel, look up
    // its submap (mux[ch] if mapping has submaps > 1, else 0), pick the
    // submap's floor index, and run the floor decoder. End-of-packet
    // during floor decode is §4.3.2 nominal: both Floor*Decoder
    // implementations already collapse it to `Unused`, but §4.3.2's
    // closing note mandates we zero every output vector and skip to
    // overlap-add. We surface that as `AudioPacketOutcome::Zeroed`.
    //
    // §4.3.2 step 6: an `Unused` floor sets `no_residue[ch] = true`.
    let mut floors_decoded: Vec<Option<Vec<f32>>> = Vec::with_capacity(channels);
    let mut no_residue: Vec<bool> = Vec::with_capacity(channels);
    for ch in 0..channels {
        let submap = submap_for_channel(mapping, ch)?;
        let submap_config = mapping.submap_configs.get(submap as usize).ok_or(
            AudioPacketError::BadSubmapIndex {
                channel: ch,
                submap,
                submaps: mapping.submap_configs.len(),
            },
        )?;
        let floor_index = submap_config.floor as usize;
        let floor_decoder =
            state
                .floors
                .get(floor_index)
                .ok_or(AudioPacketError::BadSubmapFloor {
                    submap: submap as usize,
                    floor: submap_config.floor,
                    floor_count: state.floors.len(),
                })?;
        let (curve, nor) = floor_decoder.decode(reader, half_n);
        floors_decoded.push(curve);
        no_residue.push(nor);
    }

    // §4.3.3 — nonzero-vector propagate via coupling steps.
    nonzero_propagate(&mut no_residue, &mapping.coupling).map_err(AudioPacketError::Packet)?;

    // §4.3.4 — residue decode in submap order.
    let mut residues_per_channel: Vec<Vec<f32>> = vec![vec![0.0f32; half_n]; channels];
    for submap_idx in 0..(mapping.submaps as usize) {
        // Gather the channels belonging to this submap in ascending
        // channel order, and build the per-bundle do_not_decode_flag
        // from each gathered channel's no_residue flag (§4.3.4 step 2).
        let mut bundle_channels: Vec<usize> = Vec::new();
        let mut do_not_decode: Vec<bool> = Vec::new();
        for (ch, &nor) in no_residue.iter().enumerate().take(channels) {
            let ch_submap = submap_for_channel(mapping, ch)?;
            if ch_submap as usize == submap_idx {
                bundle_channels.push(ch);
                do_not_decode.push(nor);
            }
        }
        if bundle_channels.is_empty() {
            // The setup parser already guarantees every submap has at
            // least one channel routed to it (the mux[ch] OOB check
            // bounds mux values into `0..submaps`); but a degenerate
            // hand-built mapping might leave a submap unused. Skip
            // cleanly rather than dispatching a zero-channel residue
            // decode.
            continue;
        }
        let submap_config = &mapping.submap_configs[submap_idx];
        let residue_index = submap_config.residue as usize;
        let residue_decoder =
            state
                .residues
                .get(residue_index)
                .ok_or(AudioPacketError::BadSubmapResidue {
                    submap: submap_idx,
                    residue: submap_config.residue,
                    residue_count: state.residues.len(),
                })?;
        // §4.3.4 step 5: decode `ch` vectors of length n/2.
        let bundle = residue_decoder
            .decode(reader, n, &do_not_decode)
            .map_err(AudioPacketError::Residue)?;
        // §4.3.4 step 7: scatter back into the global per-channel array.
        for (bundle_idx, &ch) in bundle_channels.iter().enumerate() {
            // The residue decoder allocates exactly `ch` vectors of
            // length n/2 (§8.6.2 step 1), so this indexing cannot
            // panic on a well-formed call.
            if let Some(v) = bundle.get(bundle_idx) {
                residues_per_channel[ch] = v.clone();
            }
        }
    }

    // §4.3.5 — inverse coupling (descending step order).
    inverse_couple_all(&mut residues_per_channel, &mapping.coupling)
        .map_err(AudioPacketError::Coupling)?;

    // §4.3.6 — dot product per channel. An `Unused` floor whose
    // no_residue survived §4.3.3 propagation produces the all-zero
    // spectrum. (Note that §4.3.3 may have flipped a channel's
    // no_residue back to `false` while its floor was `None`; in that
    // case spec §4.3.6 still has nothing to multiply against, so we
    // also emit zero for that channel — matching the §4.3.6 text "the
    // floor curve" + the §4.3.3 closing note "an 'unused' floor has
    // no decoded floor information; it is important that this is
    // remembered at floor curve synthesis time".)
    let spectra = dot_product_all(&floors_decoded, &residues_per_channel, half_n)
        .map_err(AudioPacketError::Packet)?;

    Ok(AudioPacketOutcome::PreImdct {
        mode_number: header.mode_number,
        blockflag: header.blockflag,
        n,
        previous_window_flag: header.previous_window_flag,
        next_window_flag: header.next_window_flag,
        spectra,
    })
}

/// Top-level audio-packet decode entry point.
///
/// Runs the §4.3.2..§4.3.6 driver via [`decode_audio_packet_pre_imdct`]
/// and then stops at the §4.3.7 inverse-MDCT boundary, returning
/// [`AudioPacketError::ImdctStage`] cleanly. This is the legacy entry
/// point preserved for callers that want the pre-IMDCT stop; the new
/// IMDCT-wired entry point is [`decode_one_packet_windowed`].
///
/// The §4.3.2-mandated "zero every output vector" short-circuit is
/// surfaced via [`AudioPacketError::ImdctStage`] just like the normal
/// pre-IMDCT path: the IMDCT-of-zero is zero by linearity, so the
/// behaviour is identical from the caller's perspective.
///
/// # Errors
///
/// Returns [`AudioPacketError::ImdctStage`] on every successful drive
/// (the §4.3.7 boundary stop). Any earlier-stage failure is surfaced as
/// the corresponding [`AudioPacketError`] variant. The function never
/// returns `Ok(_)`; it is shaped that way so the public signature does
/// not need to change.
pub fn decode_one_packet(
    reader: &mut BitReaderLsb<'_>,
    setup: &VorbisSetupHeader,
    state: &AudioDecoderState,
    audio_channels: u8,
    blocksize_0: usize,
    blocksize_1: usize,
) -> Result<(), AudioPacketError> {
    let _outcome = decode_audio_packet_pre_imdct(
        reader,
        setup,
        state,
        audio_channels,
        blocksize_0,
        blocksize_1,
    )?;
    // §4.3.7 inverse MDCT normalization — DOCS-GAP'd: the spec defers
    // the MDCT definition entirely to external reference [1], which the
    // workspace clean-room policy bars. The kernel + window + overlap-add
    // are all wired (see `decode_one_packet_windowed`); only the
    // fixture-derived normalization scalar is still deferred. This
    // legacy entry point preserves the pre-IMDCT stop for callers that
    // depend on it.
    Err(AudioPacketError::ImdctStage)
}

/// Run the full §4.3.2..§4.3.7-then-§4.3.6-window pipeline over one
/// audio packet and return per-channel **windowed time-domain frames**.
///
/// This is the IMDCT-wired sibling of [`decode_audio_packet_pre_imdct`]:
/// it runs the §4.3.2..§4.3.6 pipeline to produce per-channel
/// audio-spectrum vectors, then per channel runs the §4.3.7
/// [`crate::imdct::imdct_naive`] direct cosine-summation kernel on the
/// length-`n/2` spectrum to obtain the length-`n` time-domain frame,
/// then element-wise multiplies by the §4.3.6 / §1.3.2 Vorbis window
/// built once per packet via
/// [`AudioPacketHeader::build_window`]. The result is the input the
/// §4.3.8 [`crate::overlap::OverlapAdd::push_frame`] primitive expects
/// — one per-channel windowed frame ready to feed straight in.
///
/// * `reader` — an LSB-first bit reader positioned at the first bit of
///   the audio packet (RFC-3533-stripped, Ogg-page-coalesced payload).
/// * `setup` — the stream's parsed setup header.
/// * `state` — the per-stream decoder cache built by
///   [`AudioDecoderState::new`].
/// * `audio_channels` — the stream's `[audio_channels]` from the
///   identification header.
/// * `blocksize_0` / `blocksize_1` — the two blocksizes from the
///   identification header (§4.2.2).
/// * `imdct_scale` — the deferred-normalization knob. The Vorbis IMDCT
///   cross-reference document
///   (`docs/audio/vorbis/imdct-cross-reference.md` §"Vorbis-specific
///   parameters" item 5) notes the Vorbis-specific normalization scalar
///   "falls out of matching the fixture traces"; the staged traces
///   under `docs/audio/vorbis/fixtures/` do not yet log post-IMDCT
///   samples, so this scalar is **deliberately deferred** to a follow-up
///   round once those traces extend. Callers pass `1.0` for the bare
///   un-normalized kernel, or any tentative scaling they want to
///   experiment with. By linearity of [`crate::imdct::imdct_naive`] this
///   parameter is a pure output multiplier: scaling it by `α` scales
///   every returned sample by `α`.
///
/// # Errors
///
/// * Every error path the §4.3.2..§4.3.6 stage emits surfaces verbatim
///   from [`decode_audio_packet_pre_imdct`].
/// * [`AudioPacketError::Window`] for a [`WindowError`] from the §4.3.6
///   / §1.3.2 window builder (e.g. `blocksize_0 > blocksize_1` on a long
///   block — already caught by the identification-header parser, but
///   checked defensively here).
/// * [`AudioPacketError::Imdct`] for an [`ImdctError`] from the §4.3.7
///   kernel. The §4.3.6 dot-product already returns a length-`n/2`
///   spectrum per channel and `n` is power-of-two-validated by the
///   identification header, so this should not arise in practice; the
///   variant exists for defensive surfacing.
pub fn decode_audio_packet_windowed(
    reader: &mut BitReaderLsb<'_>,
    setup: &VorbisSetupHeader,
    state: &AudioDecoderState,
    audio_channels: u8,
    blocksize_0: usize,
    blocksize_1: usize,
    imdct_scale: f32,
) -> Result<WindowedPacketOutcome, AudioPacketError> {
    let pre = decode_audio_packet_pre_imdct(
        reader,
        setup,
        state,
        audio_channels,
        blocksize_0,
        blocksize_1,
    )?;
    apply_imdct_and_window(pre, blocksize_0, imdct_scale)
}

/// IMDCT + §4.3.6 window post-processing of an [`AudioPacketOutcome`].
///
/// Separated from [`decode_audio_packet_windowed`] so callers that
/// already hold a pre-IMDCT outcome (e.g. from a buffered decode) can
/// run the §4.3.7 stage without re-driving §4.3.2..§4.3.6 from the bit
/// stream. The transformation is pure (no bit-reader state).
///
/// # Errors
///
/// See [`decode_audio_packet_windowed`].
pub fn apply_imdct_and_window(
    outcome: AudioPacketOutcome,
    blocksize_0: usize,
    imdct_scale: f32,
) -> Result<WindowedPacketOutcome, AudioPacketError> {
    match outcome {
        AudioPacketOutcome::PreImdct {
            mode_number,
            blockflag,
            n,
            previous_window_flag,
            next_window_flag,
            spectra,
        } => {
            let header = AudioPacketHeader {
                mode_number,
                blockflag,
                n,
                previous_window_flag,
                next_window_flag,
            };
            // §4.3.6 / §1.3.2 window — one build per packet, reused for
            // every channel. The window length matches the packet's `n`.
            let window = header
                .build_window(blocksize_0)
                .map_err(AudioPacketError::Window)?;
            // Defensive: the window builder validates `n` already, but
            // pin the length match against the IMDCT output to make any
            // future window-builder change loud.
            if window.len() != n {
                return Err(AudioPacketError::Window(WindowError::NotPowerOfTwo { n }));
            }

            // §4.3.7 per channel: IMDCT(spectrum) → length-`n` frame,
            // then §4.3.6 windowing via the [`window_premultiply`]
            // primitive. The window is `0` at the lead-in and tail, so
            // the same multiplication zeroes the overlap-out-of-bounds
            // regions automatically.
            let mut frames: Vec<Vec<f32>> = Vec::with_capacity(spectra.len());
            for spectrum in &spectra {
                let mut time_frame = vec![0.0f32; n];
                imdct_naive(spectrum, &mut time_frame, imdct_scale)
                    .map_err(AudioPacketError::Imdct)?;
                window_premultiply(&mut time_frame, &window)
                    .map_err(AudioPacketError::WindowPremultiply)?;
                frames.push(time_frame);
            }

            Ok(WindowedPacketOutcome::Windowed {
                mode_number,
                blockflag,
                n,
                previous_window_flag,
                next_window_flag,
                frames,
            })
        }
        AudioPacketOutcome::Zeroed {
            mode_number,
            blockflag,
            n,
            previous_window_flag,
            next_window_flag,
            channels,
        } => {
            // The IMDCT of the all-zero spectrum is the all-zero frame
            // (`imdct::tests::zero_input_gives_zero_output`), and the
            // §4.3.6 window times zero is still zero. We can skip the
            // window build entirely here and emit the canonical zero
            // frames at the right geometry.
            let frames = vec![vec![0.0f32; n]; channels];
            Ok(WindowedPacketOutcome::ZeroedWindowed {
                mode_number,
                blockflag,
                n,
                previous_window_flag,
                next_window_flag,
                frames,
            })
        }
    }
}

/// IMDCT-wired top-level packet driver: returns one length-`n` windowed
/// time-domain frame per channel, ready to feed into per-channel
/// [`crate::overlap::OverlapAdd::push_frame`] instances.
///
/// Convenience wrapper around [`decode_audio_packet_windowed`] that
/// extracts the per-channel frames directly. The header geometry is
/// recoverable from the returned [`WindowedPacketOutcome`] via
/// [`WindowedPacketOutcome::header`] when needed; this entry point is
/// for the common case where the caller already pairs the per-packet
/// header with the frames stream-side.
///
/// # Errors
///
/// See [`decode_audio_packet_windowed`].
pub fn decode_one_packet_windowed(
    reader: &mut BitReaderLsb<'_>,
    setup: &VorbisSetupHeader,
    state: &AudioDecoderState,
    audio_channels: u8,
    blocksize_0: usize,
    blocksize_1: usize,
    imdct_scale: f32,
) -> Result<WindowedPacketOutcome, AudioPacketError> {
    decode_audio_packet_windowed(
        reader,
        setup,
        state,
        audio_channels,
        blocksize_0,
        blocksize_1,
        imdct_scale,
    )
}

/// Resolve the submap a channel belongs to (`mux[ch]` if the mapping
/// has `submaps > 1`, else `0`).
///
/// Per §4.2.4 "Mappings": when `submaps == 1`, the per-channel `mux[ch]`
/// table is not present in the bitstream and every channel implicitly
/// routes to submap 0. The setup parser stores `mux` as an empty vector
/// in that case and as a length-`audio_channels` vector when
/// `submaps > 1`.
fn submap_for_channel(mapping: &MappingHeader, channel: usize) -> Result<u8, AudioPacketError> {
    if mapping.submaps <= 1 {
        return Ok(0);
    }
    mapping
        .mux
        .get(channel)
        .copied()
        .ok_or(AudioPacketError::MuxOutOfRange {
            channel,
            mux_len: mapping.mux.len(),
        })
}

/// Errors that can arise while driving §4.3.2..§4.3.6.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioPacketError {
    /// §4.3.1 packet-prelude read failed (see [`PacketError`]).
    Header(PacketError),
    /// A §4.3.3 / §4.3.6 driver step failed (out-of-range coupling
    /// channel, mismatched channel counts, wrong-length vectors).
    Packet(PacketError),
    /// §4.3.5 inverse coupling failed.
    Coupling(CouplingError),
    /// A §4.3.4 residue decode failed (other than nominal §8.6.2 EOF).
    Residue(ResidueError),
    /// Building a [`Floor0Decoder`] from a setup-header entry failed
    /// (per-stream construction error, see [`Floor0Error`]).
    Floor0Build {
        /// Position of the failing entry in `setup.floors`.
        index: usize,
        /// The underlying construction error.
        source: Floor0Error,
    },
    /// Building a [`Floor1Decoder`] from a setup-header entry failed.
    Floor1Build {
        /// Position of the failing entry in `setup.floors`.
        index: usize,
        /// The underlying construction error.
        source: Floor1Error,
    },
    /// Building a [`ResidueDecoder`] from a setup-header entry failed.
    ResidueBuild {
        /// Position of the failing entry in `setup.residues`.
        index: usize,
        /// The underlying construction error.
        source: ResidueError,
    },
    /// The §4.3.1 `[mode_number]` selected a mode whose `mapping` index
    /// is out of range. Defensive: the setup parser already range-checks
    /// `mode.mapping` against `mappings.len()`.
    BadModeMapping {
        /// The §4.3.1 mode number.
        mode_number: u32,
        /// The offending mapping index.
        mapping: u8,
        /// `setup.mappings.len()`.
        mapping_count: usize,
    },
    /// The submap index resolved from `mux[ch]` (or implicitly `0` when
    /// `submaps == 1`) was outside the mapping's `submap_configs`
    /// array. Defensive: the setup parser range-checks `mux` values.
    BadSubmapIndex {
        /// The channel whose submap lookup failed.
        channel: usize,
        /// The offending submap index.
        submap: u8,
        /// `mapping.submap_configs.len()`.
        submaps: usize,
    },
    /// A submap's `floor` index pointed past the decoder cache.
    /// Defensive: the setup parser range-checks floor indices.
    BadSubmapFloor {
        /// The submap index.
        submap: usize,
        /// The offending floor index.
        floor: u8,
        /// Number of floor decoders cached.
        floor_count: usize,
    },
    /// A submap's `residue` index pointed past the decoder cache.
    /// Defensive: the setup parser range-checks residue indices.
    BadSubmapResidue {
        /// The submap index.
        submap: usize,
        /// The offending residue index.
        residue: u8,
        /// Number of residue decoders cached.
        residue_count: usize,
    },
    /// A `mux[ch]` lookup ran past the mapping's `mux` array. Defensive:
    /// the setup parser sizes `mux` to exactly `audio_channels` entries
    /// when `submaps > 1`.
    MuxOutOfRange {
        /// The channel whose lookup failed.
        channel: usize,
        /// The length of `mapping.mux`.
        mux_len: usize,
    },
    /// **§4.3.7 inverse MDCT is a documented docs gap.** The legacy
    /// [`decode_one_packet`] entry point stops cleanly at this boundary
    /// and returns this variant. The §4.3.2..§4.3.6 stages all ran
    /// successfully when this is returned; the per-channel pre-IMDCT
    /// spectra are available via [`decode_audio_packet_pre_imdct`]; the
    /// per-channel windowed time-domain frames are available via
    /// [`decode_audio_packet_windowed`] (with the deferred-normalization
    /// scalar passed as `imdct_scale`).
    ImdctStage,
    /// The §4.3.6 / §1.3.2 Vorbis window builder rejected the requested
    /// geometry — defensive: the identification header already validates
    /// `blocksize_0` / `blocksize_1` as powers of two, so this only
    /// arises from a hand-built setup.
    Window(WindowError),
    /// The §4.3.7 inverse-MDCT kernel rejected the spectrum or output
    /// length — defensive: the §4.3.6 dot-product always returns a
    /// length-`n/2` spectrum per channel, and `n` is power-of-two
    /// validated, so this only arises from a hand-built outcome passed
    /// to [`apply_imdct_and_window`].
    Imdct(ImdctError),
    /// The §4.3.6 / §4.3.7 window pre-multiplication primitive
    /// rejected a length mismatch between the IMDCT-output frame and
    /// the §4.3.1-built window — defensive: both sides are derived
    /// from the same packet `n` here, so this only arises from a
    /// hand-built outcome passed to [`apply_imdct_and_window`].
    WindowPremultiply(WindowPremultiplyError),
}

impl core::fmt::Display for AudioPacketError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AudioPacketError::Header(e) => write!(f, "vorbis audio packet: §4.3.1 header: {e}"),
            AudioPacketError::Packet(e) => write!(f, "vorbis audio packet: §4.3 driver: {e}"),
            AudioPacketError::Coupling(e) => {
                write!(f, "vorbis audio packet: §4.3.5 coupling: {e}")
            }
            AudioPacketError::Residue(e) => {
                write!(f, "vorbis audio packet: §4.3.4 residue: {e}")
            }
            AudioPacketError::Floor0Build { index, source } => write!(
                f,
                "vorbis audio packet: floor[{index}] (type 0) build failed: {source}"
            ),
            AudioPacketError::Floor1Build { index, source } => write!(
                f,
                "vorbis audio packet: floor[{index}] (type 1) build failed: {source}"
            ),
            AudioPacketError::ResidueBuild { index, source } => write!(
                f,
                "vorbis audio packet: residue[{index}] build failed: {source}"
            ),
            AudioPacketError::BadModeMapping {
                mode_number,
                mapping,
                mapping_count,
            } => write!(
                f,
                "vorbis audio packet: §4.3.1 mode_number {mode_number} maps to mapping \
                 {mapping} but only {mapping_count} mappings are configured"
            ),
            AudioPacketError::BadSubmapIndex {
                channel,
                submap,
                submaps,
            } => write!(
                f,
                "vorbis audio packet: §4.3.2 channel {channel} resolved to submap \
                 {submap} but only {submaps} submaps are configured"
            ),
            AudioPacketError::BadSubmapFloor {
                submap,
                floor,
                floor_count,
            } => write!(
                f,
                "vorbis audio packet: submap {submap} floor {floor} >= floor_count {floor_count}"
            ),
            AudioPacketError::BadSubmapResidue {
                submap,
                residue,
                residue_count,
            } => write!(
                f,
                "vorbis audio packet: submap {submap} residue {residue} >= \
                 residue_count {residue_count}"
            ),
            AudioPacketError::MuxOutOfRange { channel, mux_len } => write!(
                f,
                "vorbis audio packet: §4.3.2 mux[{channel}] is out of range (mux \
                 has only {mux_len} entries)"
            ),
            AudioPacketError::ImdctStage => write!(
                f,
                "vorbis audio packet: §4.3.7 inverse MDCT is a documented docs gap \
                 (Vorbis I spec defers to external reference [1] — barred by \
                  workspace clean-room policy)"
            ),
            AudioPacketError::Window(e) => {
                write!(f, "vorbis audio packet: §4.3.6 window builder: {e}")
            }
            AudioPacketError::Imdct(e) => {
                write!(f, "vorbis audio packet: §4.3.7 inverse MDCT: {e}")
            }
            AudioPacketError::WindowPremultiply(e) => write!(
                f,
                "vorbis audio packet: §4.3.6 window pre-multiplication: {e}"
            ),
        }
    }
}

impl std::error::Error for AudioPacketError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AudioPacketError::Header(e) | AudioPacketError::Packet(e) => Some(e),
            AudioPacketError::Coupling(e) => Some(e),
            AudioPacketError::Residue(e) => Some(e),
            AudioPacketError::Floor0Build { source, .. } => Some(source),
            AudioPacketError::Floor1Build { source, .. } => Some(source),
            AudioPacketError::ResidueBuild { source, .. } => Some(source),
            AudioPacketError::Window(e) => Some(e),
            AudioPacketError::Imdct(e) => Some(e),
            AudioPacketError::WindowPremultiply(e) => Some(e),
            AudioPacketError::BadModeMapping { .. }
            | AudioPacketError::BadSubmapIndex { .. }
            | AudioPacketError::BadSubmapFloor { .. }
            | AudioPacketError::BadSubmapResidue { .. }
            | AudioPacketError::MuxOutOfRange { .. }
            | AudioPacketError::ImdctStage => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::{VorbisCodebook, VqLookup};
    use crate::huffman::HuffmanTree;
    use crate::setup::{
        Floor1Header, FloorHeader, FloorKind, MappingCouplingStep, MappingHeader, MappingSubmap,
        ModeHeader, ResidueHeader, VorbisSetupHeader,
    };
    use oxideav_core::bits::{BitReaderLsb, BitWriterLsb};

    // ---- Synthetic-packet fixtures ----
    //
    // Build the minimum viable Vorbis setup that exercises the §4.3.2
    // floor iteration + §4.3.3 nonzero propagate + §4.3.4 residue
    // submap iteration + §4.3.5 inverse coupling + §4.3.6 dot product:
    //
    //   - 1 mode, short block, mapping 0
    //   - 1 mapping, submaps == 1, coupling == [], mux empty
    //   - 1 floor, type 1, trivial: 1 partition, 1 class, no master/sub
    //     books (every Y read returns 0), multiplier 1, rangebits 4 → an
    //     all-zero floor that nonetheless reads the §7.2.3 [nonzero] bit
    //     and returns Curve(...) when that bit is set or Unused when
    //     cleared.
    //   - 1 residue, type 0, with trivial classbook & no value books
    //     → every partition decodes to the all-zero vector.
    //
    // The packet bitstream we then assemble for these tests is just the
    // §4.3.1 prelude + the §7.2.3 floor1 [nonzero] bit + a few zero
    // partitions for the residue classbook.

    fn floor1_all_zero_header() -> Floor1Header {
        // No partitions → no classes → empty partition_class_list. The
        // Floor1Decoder is happy with this: floor1_values() == 2 (just
        // the two implicit endpoints) and packet_decode reads the
        // [nonzero] bit + the two endpoint amplitudes (0 bits each
        // since multiplier=1 → range=256 → amp_bits = ilog(255) = 8;
        // so still 8 bits each).
        //
        // Adjust: multiplier 1 → range 256 → amp_bits = 8. We want 0
        // amp bits to keep the packet trivially encodable. Set
        // multiplier 4 → range 7 → amp_bits = ilog(6) = 3 — 6-bit
        // total amplitude region. We use multiplier=4 → 3 bits per
        // endpoint amplitude.
        Floor1Header {
            partitions: 0,
            partition_class_list: Vec::new(),
            classes: Vec::new(),
            multiplier: 4,
            rangebits: 4,
            x_list: Vec::new(),
        }
    }

    fn floor_type1_header() -> FloorHeader {
        FloorHeader {
            floor_type: 1,
            kind: FloorKind::Type1(floor1_all_zero_header()),
        }
    }

    /// A single-entry codebook with a single dimension and a VQ lookup
    /// type 2 that emits one zero scalar per call. This makes the
    /// residue classbook trivial: every classword decodes to class 0,
    /// and class 0 has no per-stage value books, so every partition
    /// stays zero.
    fn zero_lookup_codebook() -> VorbisCodebook {
        // Single-entry codebooks have a unique zero-length codeword
        // (§3.2.1 errata 20150226): the reader emits index 0 without
        // consuming any bits.
        VorbisCodebook {
            dimensions: 1,
            entries: 1,
            codeword_lengths: vec![1],
            lookup: VqLookup::Tessellation {
                minimum_value: 0.0,
                delta_value: 0.0,
                value_bits: 1,
                sequence_p: false,
                multiplicands: vec![0],
            },
        }
    }

    /// A simpler scalar codebook (no VQ lookup) used as the residue
    /// classbook. Single-entry, 1-bit, decodes to class 0 every time.
    fn scalar_classbook() -> VorbisCodebook {
        VorbisCodebook {
            dimensions: 1,
            entries: 1,
            codeword_lengths: vec![1],
            lookup: VqLookup::None,
        }
    }

    fn residue_zero_header() -> ResidueHeader {
        // One classification, no cascade bits set → no value books to
        // build. classbook = 0 (scalar). residue_begin = 0,
        // residue_end = 0 → the §8.6.2 partition loop iterates zero
        // times and the output is all-zero.
        ResidueHeader {
            residue_type: 0,
            residue_begin: 0,
            residue_end: 0,
            partition_size: 1,
            classifications: 1,
            classbook: 0,
            cascade: vec![0],
            books: vec![std::array::from_fn(|_| None)],
        }
    }

    fn mode_short_block() -> ModeHeader {
        ModeHeader {
            blockflag: false,
            windowtype: 0,
            transformtype: 0,
            mapping: 0,
        }
    }

    fn mapping_mono_no_coupling() -> MappingHeader {
        MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: Vec::new(),
            mux: Vec::new(),
            submap_configs: vec![MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        }
    }

    fn mono_setup() -> VorbisSetupHeader {
        VorbisSetupHeader {
            codebooks: vec![scalar_classbook(), zero_lookup_codebook()],
            time_placeholders: Vec::new(),
            floors: vec![floor_type1_header()],
            residues: vec![residue_zero_header()],
            mappings: vec![mapping_mono_no_coupling()],
            modes: vec![mode_short_block()],
            framing_flag: true,
        }
    }

    fn mapping_stereo_coupled() -> MappingHeader {
        MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: vec![MappingCouplingStep {
                magnitude_channel: 0,
                angle_channel: 1,
            }],
            mux: Vec::new(),
            submap_configs: vec![MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        }
    }

    fn stereo_setup() -> VorbisSetupHeader {
        VorbisSetupHeader {
            codebooks: vec![scalar_classbook(), zero_lookup_codebook()],
            time_placeholders: Vec::new(),
            floors: vec![floor_type1_header()],
            residues: vec![residue_zero_header()],
            mappings: vec![mapping_stereo_coupled()],
            modes: vec![mode_short_block()],
            framing_flag: true,
        }
    }

    /// Per the headers above: a "this channel is used" packet has
    /// 1 (packet_type) + 0 (mode_bits=0) + 1 (nonzero) + 3 + 3
    /// (endpoint amplitudes, ilog(range-1) with multiplier 4 →
    /// range 7 → ilog(6) = 3) = 8 bits. Mapping pattern: 0 | 1 | 0 0 0 | 0 0 0
    /// → byte 0b00000010 = 0x02.
    fn write_used_packet() -> Vec<u8> {
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type = audio
                           // mode_count = 1 → 0 mode bits.
        w.write_u32(1, 1); // floor1 [nonzero] flag set
        w.write_u32(0, 3); // endpoint amplitude 0 (3 bits)
        w.write_u32(0, 3); // endpoint amplitude 1 (3 bits)
                           // floor1 has 0 partitions → no further floor bits.
                           // residue: begin == end == 0 → zero partitions decoded.
        w.finish()
    }

    fn write_unused_packet() -> Vec<u8> {
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type = audio
        w.write_u32(0, 1); // floor1 [nonzero] = 0 → Unused
        w.finish()
    }

    #[test]
    fn decoder_state_builds_floor1_and_residue() {
        let setup = mono_setup();
        let state = AudioDecoderState::new(&setup).unwrap();
        assert_eq!(state.floor_count(), 1);
        assert_eq!(state.residue_count(), 1);
    }

    #[test]
    fn decoder_state_propagates_floor1_build_error() {
        // Multiplier 0 is rejected (§7.2.2: floor1_multiplier ∈ 1..=4).
        let bad_floor = FloorHeader {
            floor_type: 1,
            kind: FloorKind::Type1(Floor1Header {
                partitions: 0,
                partition_class_list: Vec::new(),
                classes: Vec::new(),
                multiplier: 0,
                rangebits: 4,
                x_list: Vec::new(),
            }),
        };
        let setup = VorbisSetupHeader {
            codebooks: vec![scalar_classbook()],
            time_placeholders: Vec::new(),
            floors: vec![bad_floor],
            residues: Vec::new(),
            mappings: Vec::new(),
            modes: Vec::new(),
            framing_flag: true,
        };
        match AudioDecoderState::new(&setup) {
            Err(AudioPacketError::Floor1Build { index, .. }) => assert_eq!(index, 0),
            other => panic!("expected Floor1Build error, got {:?}", other),
        }
    }

    #[test]
    fn decoder_state_propagates_residue_build_error() {
        // Residue classbook OOB.
        let bad_residue = ResidueHeader {
            residue_type: 0,
            residue_begin: 0,
            residue_end: 0,
            partition_size: 1,
            classifications: 1,
            classbook: 99,
            cascade: vec![0],
            books: vec![std::array::from_fn(|_| None)],
        };
        let setup = VorbisSetupHeader {
            codebooks: vec![scalar_classbook()],
            time_placeholders: Vec::new(),
            floors: Vec::new(),
            residues: vec![bad_residue],
            mappings: Vec::new(),
            modes: Vec::new(),
            framing_flag: true,
        };
        match AudioDecoderState::new(&setup) {
            Err(AudioPacketError::ResidueBuild { index, .. }) => assert_eq!(index, 0),
            other => panic!("expected ResidueBuild error, got {:?}", other),
        }
    }

    #[test]
    fn driver_mono_used_packet_emits_pre_imdct_spectrum() {
        let setup = mono_setup();
        let state = AudioDecoderState::new(&setup).unwrap();
        let bytes = write_used_packet();
        let mut r = BitReaderLsb::new(&bytes);
        let outcome = decode_audio_packet_pre_imdct(&mut r, &setup, &state, 1, 64, 1024).unwrap();
        match outcome {
            AudioPacketOutcome::PreImdct {
                blockflag,
                n,
                spectra,
                ..
            } => {
                assert!(!blockflag);
                assert_eq!(n, 64);
                assert_eq!(spectra.len(), 1);
                assert_eq!(spectra[0].len(), 32); // n/2
                                                  // Floor returned a curve (Unused short-circuits earlier);
                                                  // residue is all zero. Spectrum = floor * 0 = 0 for every bin.
                for &s in &spectra[0] {
                    assert_eq!(s, 0.0);
                }
            }
            other => panic!("expected PreImdct, got {:?}", other),
        }
    }

    #[test]
    fn driver_mono_unused_floor_returns_zero_spectrum() {
        let setup = mono_setup();
        let state = AudioDecoderState::new(&setup).unwrap();
        let bytes = write_unused_packet();
        let mut r = BitReaderLsb::new(&bytes);
        let outcome = decode_audio_packet_pre_imdct(&mut r, &setup, &state, 1, 64, 1024).unwrap();
        match outcome {
            AudioPacketOutcome::PreImdct { spectra, .. } => {
                assert_eq!(spectra.len(), 1);
                // §4.3.3 / §4.3.6: an unused channel emits all-zero
                // spectrum (no_residue stays true, dot_product_all
                // maps None floor → zero).
                for &s in &spectra[0] {
                    assert_eq!(s, 0.0);
                }
            }
            other => panic!("expected PreImdct, got {:?}", other),
        }
    }

    #[test]
    fn driver_stereo_used_packet_runs_inverse_coupling() {
        let setup = stereo_setup();
        let state = AudioDecoderState::new(&setup).unwrap();
        // For stereo, every channel reads (nonzero + 2 endpoints) on its
        // own (§4.3.2 channel order). Two channels back-to-back:
        // [packet_type=0] + ch0:[nonzero=1, 3, 3] + ch1:[nonzero=1, 3, 3]
        // = 1 + 7 + 7 = 15 bits → 2 bytes (pad to 16).
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        for _ in 0..2 {
            w.write_u32(1, 1); // nonzero
            w.write_u32(0, 3); // amp0
            w.write_u32(0, 3); // amp1
        }
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let outcome = decode_audio_packet_pre_imdct(&mut r, &setup, &state, 2, 64, 1024).unwrap();
        match outcome {
            AudioPacketOutcome::PreImdct { spectra, .. } => {
                assert_eq!(spectra.len(), 2);
                for ch in &spectra {
                    assert_eq!(ch.len(), 32);
                    for &s in ch {
                        assert_eq!(s, 0.0);
                    }
                }
            }
            other => panic!("expected PreImdct, got {:?}", other),
        }
    }

    #[test]
    fn decode_one_packet_stops_at_imdct_boundary() {
        let setup = mono_setup();
        let state = AudioDecoderState::new(&setup).unwrap();
        let bytes = write_used_packet();
        let mut r = BitReaderLsb::new(&bytes);
        let result = decode_one_packet(&mut r, &setup, &state, 1, 64, 1024);
        assert_eq!(result, Err(AudioPacketError::ImdctStage));
    }

    #[test]
    fn decode_one_packet_imdct_error_displays_docs_gap() {
        let err = AudioPacketError::ImdctStage;
        let s = format!("{err}");
        assert!(s.contains("§4.3.7"), "{s}");
        assert!(s.contains("docs gap"), "{s}");
    }

    #[test]
    fn driver_rejects_non_audio_packet_via_header() {
        let setup = mono_setup();
        let state = AudioDecoderState::new(&setup).unwrap();
        let mut w = BitWriterLsb::new();
        w.write_u32(1, 1); // packet_type = 1 → not audio
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        match decode_audio_packet_pre_imdct(&mut r, &setup, &state, 1, 64, 1024) {
            Err(AudioPacketError::Header(PacketError::NonAudioPacketType { packet_type })) => {
                assert_eq!(packet_type, 1);
            }
            other => panic!("expected Header(NonAudioPacketType), got {:?}", other),
        }
    }

    #[test]
    fn submap_for_channel_single_submap_returns_zero() {
        let mapping = mapping_mono_no_coupling();
        assert_eq!(submap_for_channel(&mapping, 0).unwrap(), 0);
        // Even past the (empty) mux table: submaps == 1 is the
        // implicit-zero path.
        assert_eq!(submap_for_channel(&mapping, 42).unwrap(), 0);
    }

    #[test]
    fn submap_for_channel_multi_submap_uses_mux() {
        let mapping = MappingHeader {
            mapping_type: 0,
            submaps: 2,
            coupling: Vec::new(),
            mux: vec![0, 1, 0, 1],
            submap_configs: vec![
                MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                },
                MappingSubmap {
                    time_placeholder: 0,
                    floor: 0,
                    residue: 0,
                },
            ],
        };
        assert_eq!(submap_for_channel(&mapping, 0).unwrap(), 0);
        assert_eq!(submap_for_channel(&mapping, 1).unwrap(), 1);
        assert_eq!(submap_for_channel(&mapping, 2).unwrap(), 0);
        assert_eq!(submap_for_channel(&mapping, 3).unwrap(), 1);
        // Channel 4 is out of range; the parser would have caught this,
        // but the driver returns the defensive error.
        match submap_for_channel(&mapping, 4) {
            Err(AudioPacketError::MuxOutOfRange {
                channel: 4,
                mux_len: 4,
            }) => {}
            other => panic!("expected MuxOutOfRange, got {:?}", other),
        }
    }

    /// Round-trip helper: make sure the Huffman tree builder accepts the
    /// trivial single-entry codebook we use in these tests (sanity check
    /// — if this fails the rest of the test suite is misleading).
    #[test]
    fn trivial_classbook_huffman_tree_builds() {
        let cb = scalar_classbook();
        let _tree = HuffmanTree::from_codebook(&cb).expect("trivial classbook builds");
    }

    // ---- Round 17: §4.3.7 IMDCT + §4.3.6 window wiring ----

    /// The IMDCT-wired entry point returns one length-`n` windowed frame
    /// per channel on a normal used-channel packet. Because the trivial
    /// setup decodes every spectrum to all-zero (floor-curve × zero
    /// residue = 0), every windowed frame is also all-zero — IMDCT of
    /// zero is zero by linearity, then any window times zero is zero.
    /// The geometry (channel count, frame length = n) is the test
    /// surface.
    #[test]
    fn windowed_driver_mono_used_packet_emits_one_frame() {
        let setup = mono_setup();
        let state = AudioDecoderState::new(&setup).unwrap();
        let bytes = write_used_packet();
        let mut r = BitReaderLsb::new(&bytes);
        let outcome =
            decode_audio_packet_windowed(&mut r, &setup, &state, 1, 64, 1024, 1.0).unwrap();
        match outcome {
            WindowedPacketOutcome::Windowed {
                blockflag,
                n,
                frames,
                ..
            } => {
                assert!(!blockflag);
                assert_eq!(n, 64);
                assert_eq!(frames.len(), 1);
                assert_eq!(frames[0].len(), 64); // length n, not n/2.
                for &s in &frames[0] {
                    assert_eq!(s, 0.0);
                }
            }
            other => panic!("expected Windowed, got {:?}", other),
        }
    }

    /// Apply IMDCT + window to a hand-built PreImdct outcome with a
    /// non-zero spectrum (one bin only) and verify:
    /// * one frame per channel, each length `n`,
    /// * the lead-in / tail regions (window == 0) are exactly zero,
    /// * the plateau region (window == 1) carries the bare IMDCT output.
    ///
    /// This exercises the IMDCT + window composition without depending
    /// on the synthetic-packet bit-stream encoding.
    #[test]
    fn apply_imdct_and_window_carries_one_bin_into_plateau() {
        // Short block: n = 64 → window has no lead-in/tail (the plain
        // symmetric short shape spans the full length and is everywhere
        // > 0). Use a long block instead so we have a clear plateau
        // surrounded by zero edges.
        //
        // Long block n = 1024 with previous_window_flag = false,
        // next_window_flag = false → hybrid ramps with `blocksize_0/2`
        // ramp length on each side. Lead-in = `n/4 - blocksize_0/4`
        // samples of zero before the rising edge starts.
        let blocksize_0 = 64;
        let n = 1024;
        let lead_in = n / 4 - blocksize_0 / 4; // = 240 samples of zero
        let half_n = n / 2;
        let spectrum: Vec<f32> = (0..half_n).map(|i| (i as f32 + 1.0) * 0.001).collect();
        let outcome = AudioPacketOutcome::PreImdct {
            mode_number: 0,
            blockflag: true,
            n,
            previous_window_flag: false,
            next_window_flag: false,
            spectra: vec![spectrum],
        };
        let windowed = apply_imdct_and_window(outcome, blocksize_0, 1.0).unwrap();
        match windowed {
            WindowedPacketOutcome::Windowed {
                frames,
                n: out_n,
                blockflag,
                ..
            } => {
                assert_eq!(out_n, n);
                assert!(blockflag);
                assert_eq!(frames.len(), 1);
                assert_eq!(frames[0].len(), n);
                // The lead-in region is everywhere zero in the window;
                // the windowed frame inherits that exactness.
                for (i, &sample) in frames[0].iter().enumerate().take(lead_in) {
                    assert_eq!(sample, 0.0, "lead-in sample {i} non-zero in windowed frame");
                }
                // The mirrored tail region is also everywhere zero.
                let tail_start = n - lead_in;
                for (i, &sample) in frames[0].iter().enumerate().take(n).skip(tail_start) {
                    assert_eq!(sample, 0.0, "tail sample {i} non-zero in windowed frame");
                }
            }
            other => panic!("expected Windowed, got {:?}", other),
        }
    }

    /// The §4.3.2 short-circuit `Zeroed` produces per-channel all-zero
    /// length-`n` windowed frames via [`apply_imdct_and_window`].
    #[test]
    fn apply_imdct_and_window_zeroed_short_circuit() {
        let outcome = AudioPacketOutcome::Zeroed {
            mode_number: 0,
            blockflag: false,
            n: 64,
            previous_window_flag: false,
            next_window_flag: false,
            channels: 2,
        };
        let windowed = apply_imdct_and_window(outcome, 64, 1.0).unwrap();
        match windowed {
            WindowedPacketOutcome::ZeroedWindowed { frames, n, .. } => {
                assert_eq!(n, 64);
                assert_eq!(frames.len(), 2);
                for ch in &frames {
                    assert_eq!(ch.len(), 64);
                    for &s in ch {
                        assert_eq!(s, 0.0);
                    }
                }
            }
            other => panic!("expected ZeroedWindowed, got {:?}", other),
        }
    }

    /// `imdct_scale` is a pure output multiplier (inherits from
    /// `imdct::tests::scale_is_pure_output_multiplier`); scaling by `α`
    /// scales every returned sample by `α`. Test via the public
    /// [`apply_imdct_and_window`] entry point against a non-zero
    /// spectrum.
    #[test]
    fn apply_imdct_and_window_scale_is_linear() {
        let n = 64;
        let half_n = n / 2;
        let spectrum: Vec<f32> = (0..half_n).map(|i| ((i + 1) as f32 * 0.07).sin()).collect();
        let make = |_scale: f32| AudioPacketOutcome::PreImdct {
            mode_number: 0,
            blockflag: false,
            n,
            previous_window_flag: false,
            next_window_flag: false,
            spectra: vec![spectrum.clone()],
        };
        let one = apply_imdct_and_window(make(1.0), 64, 1.0).unwrap();
        let two_point_five = apply_imdct_and_window(make(1.0), 64, 2.5).unwrap();
        let frames_one = match &one {
            WindowedPacketOutcome::Windowed { frames, .. } => frames,
            _ => panic!("expected Windowed"),
        };
        let frames_alpha = match &two_point_five {
            WindowedPacketOutcome::Windowed { frames, .. } => frames,
            _ => panic!("expected Windowed"),
        };
        for i in 0..n {
            let expected = frames_one[0][i] * 2.5;
            let diff = (frames_alpha[0][i] - expected).abs();
            let tol = (expected.abs() * 1.0e-4).max(1.0e-5);
            assert!(
                diff < tol,
                "idx {i}: scaled {} != expected {} (diff {})",
                frames_alpha[0][i],
                expected,
                diff,
            );
        }
    }

    /// The composition `imdct then window` matches running each
    /// primitive directly: pin this against the standalone
    /// [`crate::imdct::imdct_naive_vec`] + the standalone
    /// [`crate::synthesis::vorbis_window`]. Guards against an off-by-one
    /// or accidental window reuse in the audio driver.
    #[test]
    fn apply_imdct_and_window_matches_direct_composition() {
        let n = 64;
        let half_n = n / 2;
        let blocksize_0 = 64;
        let spectrum: Vec<f32> = (0..half_n).map(|i| ((i + 1) as f32 * 0.13).cos()).collect();
        let outcome = AudioPacketOutcome::PreImdct {
            mode_number: 0,
            blockflag: false,
            n,
            previous_window_flag: false,
            next_window_flag: false,
            spectra: vec![spectrum.clone()],
        };
        let windowed = apply_imdct_and_window(outcome, blocksize_0, 1.0).unwrap();
        let frames = match &windowed {
            WindowedPacketOutcome::Windowed { frames, .. } => frames,
            _ => panic!("expected Windowed"),
        };

        // Direct path: bare IMDCT + element-wise window mul.
        let direct_time = crate::imdct::imdct_naive_vec(&spectrum, 1.0).unwrap();
        let window = crate::synthesis::vorbis_window(n, blocksize_0, false, false, false).unwrap();
        let direct_windowed: Vec<f32> = direct_time
            .iter()
            .zip(&window)
            .map(|(&t, &w)| t * w)
            .collect();

        for i in 0..n {
            let diff = (frames[0][i] - direct_windowed[i]).abs();
            assert!(
                diff < 1.0e-6,
                "idx {i}: driver {} != direct {} (diff {})",
                frames[0][i],
                direct_windowed[i],
                diff,
            );
        }
    }

    /// Output of [`decode_audio_packet_windowed`] feeds straight into
    /// [`crate::overlap::OverlapAdd::push_frame`] — pin the integration
    /// end-to-end on the synthetic mono packet. The first call primes
    /// (returns `Ok(None)`); the second call returns
    /// `Ok(Some(samples))` with the §4.3.8 finished-PCM geometry.
    #[test]
    fn windowed_driver_feeds_overlap_add() {
        let setup = mono_setup();
        let state = AudioDecoderState::new(&setup).unwrap();
        let bytes = write_used_packet();
        let mut overlap = crate::overlap::OverlapAdd::new();

        // Packet 1 — should prime overlap-add and return None.
        let mut r1 = BitReaderLsb::new(&bytes);
        let outcome1 =
            decode_audio_packet_windowed(&mut r1, &setup, &state, 1, 64, 1024, 1.0).unwrap();
        let frame1 = &outcome1.frames()[0];
        assert_eq!(frame1.len(), 64);
        let result1 = overlap.push_frame(frame1).unwrap();
        assert!(result1.is_none(), "first frame must prime overlap-add");

        // Packet 2 — should emit prev_n/4 + cur_n/4 = 32 finished samples.
        let mut r2 = BitReaderLsb::new(&bytes);
        let outcome2 =
            decode_audio_packet_windowed(&mut r2, &setup, &state, 1, 64, 1024, 1.0).unwrap();
        let frame2 = &outcome2.frames()[0];
        let result2 = overlap.push_frame(frame2).unwrap();
        let pcm = result2.expect("second frame should emit PCM");
        assert_eq!(pcm.len(), 64 / 4 + 64 / 4); // 32 samples
                                                // Trivial setup → spectrum is 0 → IMDCT is 0 → windowed is 0 →
                                                // overlap-add of zero + zero is zero.
        for &s in &pcm {
            assert_eq!(s, 0.0);
        }
    }

    /// The legacy [`decode_one_packet`] entry point still surfaces
    /// `ImdctStage` so callers depending on the pre-IMDCT stop are not
    /// broken by round 17. The new wiring is the additive
    /// [`decode_one_packet_windowed`] entry point.
    #[test]
    fn legacy_decode_one_packet_still_stops_at_imdct_boundary() {
        let setup = mono_setup();
        let state = AudioDecoderState::new(&setup).unwrap();
        let bytes = write_used_packet();
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            decode_one_packet(&mut r, &setup, &state, 1, 64, 1024),
            Err(AudioPacketError::ImdctStage),
        );
    }

    /// The new [`decode_one_packet_windowed`] entry point is a thin
    /// wrapper around [`decode_audio_packet_windowed`]; behaviour parity
    /// is the documented invariant.
    #[test]
    fn decode_one_packet_windowed_matches_decode_audio_packet_windowed() {
        let setup = mono_setup();
        let state = AudioDecoderState::new(&setup).unwrap();
        let bytes = write_used_packet();
        let mut r_a = BitReaderLsb::new(&bytes);
        let mut r_b = BitReaderLsb::new(&bytes);
        let a = decode_audio_packet_windowed(&mut r_a, &setup, &state, 1, 64, 1024, 1.0).unwrap();
        let b = decode_one_packet_windowed(&mut r_b, &setup, &state, 1, 64, 1024, 1.0).unwrap();
        assert_eq!(a, b);
    }

    /// The convenience accessor [`WindowedPacketOutcome::header`] returns
    /// the same fields for either variant.
    #[test]
    fn windowed_packet_outcome_header_accessor() {
        let outcome = WindowedPacketOutcome::Windowed {
            mode_number: 3,
            blockflag: true,
            n: 2048,
            previous_window_flag: true,
            next_window_flag: false,
            frames: Vec::new(),
        };
        let h = outcome.header();
        assert_eq!(h.mode_number, 3);
        assert!(h.blockflag);
        assert_eq!(h.n, 2048);
        assert!(h.previous_window_flag);
        assert!(!h.next_window_flag);

        let zeroed = WindowedPacketOutcome::ZeroedWindowed {
            mode_number: 1,
            blockflag: false,
            n: 256,
            previous_window_flag: false,
            next_window_flag: true,
            frames: Vec::new(),
        };
        let hz = zeroed.header();
        assert_eq!(hz.mode_number, 1);
        assert!(!hz.blockflag);
        assert_eq!(hz.n, 256);
        assert!(!hz.previous_window_flag);
        assert!(hz.next_window_flag);
    }

    /// The `frames()` accessor borrows the per-channel slice from
    /// either variant identically.
    #[test]
    fn windowed_packet_outcome_frames_accessor() {
        let frames_w = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let outcome = WindowedPacketOutcome::Windowed {
            mode_number: 0,
            blockflag: false,
            n: 4,
            previous_window_flag: false,
            next_window_flag: false,
            frames: frames_w.clone(),
        };
        assert_eq!(outcome.frames(), &frames_w[..]);

        let frames_z = vec![vec![0.0; 4], vec![0.0; 4]];
        let zeroed = WindowedPacketOutcome::ZeroedWindowed {
            mode_number: 0,
            blockflag: false,
            n: 4,
            previous_window_flag: false,
            next_window_flag: false,
            frames: frames_z.clone(),
        };
        assert_eq!(zeroed.frames(), &frames_z[..]);
    }

    /// The error display strings for the new Window + Imdct variants
    /// include the §4.3 section that owns the failure.
    #[test]
    fn audio_packet_error_window_and_imdct_display() {
        let we = AudioPacketError::Window(WindowError::NotPowerOfTwo { n: 31 });
        assert!(we.to_string().contains("§4.3.6"), "{we}");
        let ie = AudioPacketError::Imdct(ImdctError::SpectrumNotPowerOfTwo { spectrum_len: 31 });
        assert!(ie.to_string().contains("§4.3.7"), "{ie}");
    }
}

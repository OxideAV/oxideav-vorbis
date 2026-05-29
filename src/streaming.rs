//! Vorbis I multi-channel streaming PCM driver — packets in, finished
//! PCM out (§4.3 end-to-end glue).
//!
//! # Scope
//!
//! Rounds 1..17 of the clean-room rebuild landed every individual §4.3
//! stage as a standalone primitive (§4.3.1 packet header,
//! §4.3.2..§4.3.6 spectrum reconstruction via
//! [`crate::audio::decode_audio_packet_pre_imdct`], §4.3.7 IMDCT via
//! [`crate::imdct::imdct_naive`], §4.3.6 / §1.3.2 windowing via
//! [`crate::packet::AudioPacketHeader::build_window`], and §4.3.8
//! single-channel overlap-add via [`crate::overlap::OverlapAdd`]).
//! Round 17 then composed the first six of those into
//! [`crate::audio::decode_audio_packet_windowed`], which returns per-channel
//! **windowed time-domain frames** ready to feed into a §4.3.8 overlap-add
//! engine.
//!
//! This module closes the last composition step: it stitches a
//! per-channel [`crate::overlap::OverlapAdd`] instance per stream into a
//! single multi-channel state machine — [`StreamingDecoder`] — that the
//! caller drives one packet at a time. After the first packet the engine
//! is primed (no PCM emitted, per §4.3.8); from the second packet on, it
//! emits a [`StreamingFrame::Pcm`] holding `prev_n / 4 + cur_n / 4`
//! finished PCM samples per channel (in bitstream channel order). On
//! stream end the caller can call [`StreamingDecoder::finish`] to drain
//! the last frame's right-half tail.
//!
//! # Decode order
//!
//! Per packet the driver runs (via
//! [`crate::audio::decode_audio_packet_windowed`]):
//!
//! 1. §4.3.1 packet prelude (mode, blocksize, window flags).
//! 2. §4.3.2..§4.3.6 spectrum reconstruction (floor + residue + inverse
//!    coupling + dot product).
//! 3. §4.3.7 IMDCT of each per-channel spectrum.
//! 4. §4.3.6 / §1.3.2 windowing.
//!
//! Then per channel:
//!
//! 5. §4.3.8 overlap-add into the per-channel state.
//!
//! The result is exactly the data flow Vorbis I §4.3 prescribes from a
//! parsed audio packet to PCM, modulo the still-deferred IMDCT
//! normalization scalar (see [`crate::imdct`] for the docs-gap details).
//! The `imdct_scale: f32` knob propagates through unchanged: by linearity
//! of the IMDCT kernel, scaling it by `α` scales every output PCM sample
//! by `α`.
//!
//! # What this module is NOT
//!
//! * **Not an Ogg demuxer.** Packets come in already RFC-3533-stripped
//!   and page-coalesced; sourcing them is the demuxer's job. The
//!   [`StreamingDecoder`] is bring-your-own-packet, matching every
//!   other entry point in this crate.
//! * **Not a §4.3.9 channel-order rearrangement.** §4.3.9 is a
//!   presentation concern handled by the consumer above this module;
//!   the engine emits per-channel PCM in bitstream channel order.
//! * **Not an IMDCT normalization pin.** `imdct_scale` is still a
//!   caller-supplied knob (documented docs gap).

use crate::audio::{
    decode_audio_packet_windowed, AudioDecoderState, AudioPacketError, WindowedPacketOutcome,
};
use crate::overlap::{OverlapAdd, OverlapError};
use crate::setup::VorbisSetupHeader;
use oxideav_core::bits::BitReaderLsb;

/// Errors the multi-channel streaming driver can surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamingError {
    /// The per-packet driver
    /// ([`crate::audio::decode_audio_packet_windowed`]) failed at any
    /// §4.3.1..§4.3.7 stage.
    Packet(AudioPacketError),
    /// Per-channel overlap-add failed on a windowed frame — defensive,
    /// since the per-packet driver always produces a length-`n` power-of-two
    /// frame with `n >= 64`, which the §4.3.8 primitive accepts unconditionally.
    Overlap {
        /// Bitstream channel index whose [`OverlapAdd`] rejected the frame.
        channel: usize,
        /// The underlying overlap-add failure.
        source: OverlapError,
    },
    /// The packet's resolved channel count does not match the
    /// stream-configured channel count. The Vorbis I bitstream nominally
    /// pins the channel count at the identification header; this
    /// defensive variant fires only if a hand-built setup or a corrupted
    /// stream emits a packet with a different shape.
    ChannelCountMismatch {
        /// The channel count the [`StreamingDecoder`] was built with.
        expected: usize,
        /// The channel count the per-packet driver returned.
        got: usize,
    },
}

impl core::fmt::Display for StreamingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StreamingError::Packet(e) => {
                write!(f, "vorbis streaming: per-packet driver: {e}")
            }
            StreamingError::Overlap { channel, source } => write!(
                f,
                "vorbis streaming: overlap-add for channel {channel} rejected the frame: {source}",
            ),
            StreamingError::ChannelCountMismatch { expected, got } => write!(
                f,
                "vorbis streaming: packet emitted {got} channels but the decoder \
                 was configured for {expected}",
            ),
        }
    }
}

impl std::error::Error for StreamingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StreamingError::Packet(e) => Some(e),
            StreamingError::Overlap { source, .. } => Some(source),
            StreamingError::ChannelCountMismatch { .. } => None,
        }
    }
}

impl From<AudioPacketError> for StreamingError {
    fn from(value: AudioPacketError) -> Self {
        StreamingError::Packet(value)
    }
}

/// Outcome of a single [`StreamingDecoder::push_packet`] call.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingFrame {
    /// The first packet primed the per-channel overlap-add state and
    /// emitted no PCM (§4.3.8 priming step). The header geometry of the
    /// primed-against packet is reported so the caller can still inspect
    /// it (useful for granule-position tracking and seek-bisection state).
    Primed {
        /// `[mode_number]` of the primed-against packet.
        mode_number: u32,
        /// `[vorbis_mode_blockflag]` of the primed-against packet.
        blockflag: bool,
        /// `[n]` of the primed-against packet (the right-half tail length
        /// stored is `n / 2`).
        n: usize,
    },
    /// A normal subsequent packet emitted finished PCM samples for the
    /// previous → current packet transition. Each entry of
    /// `per_channel_pcm` holds the same number of samples,
    /// `prev_n / 4 + cur_n / 4` per §4.3.8.
    Pcm {
        /// `[mode_number]` of the packet just consumed.
        mode_number: u32,
        /// `[vorbis_mode_blockflag]` of the packet just consumed.
        blockflag: bool,
        /// `[n]` of the packet just consumed.
        n: usize,
        /// Per-channel finished PCM samples, in bitstream channel order.
        /// Every entry has the same length, `prev_n / 4 + cur_n / 4`.
        per_channel_pcm: Vec<Vec<f32>>,
    },
}

impl StreamingFrame {
    /// `true` if this is the priming-step variant (no PCM emitted).
    #[must_use]
    pub fn is_primed(&self) -> bool {
        matches!(self, StreamingFrame::Primed { .. })
    }

    /// Borrow the per-channel PCM if this is the [`StreamingFrame::Pcm`]
    /// variant; returns `None` for the priming step.
    #[must_use]
    pub fn pcm(&self) -> Option<&[Vec<f32>]> {
        match self {
            StreamingFrame::Pcm {
                per_channel_pcm, ..
            } => Some(per_channel_pcm),
            StreamingFrame::Primed { .. } => None,
        }
    }

    /// `[n]` of the packet this outcome corresponds to.
    #[must_use]
    pub fn n(&self) -> usize {
        match *self {
            StreamingFrame::Primed { n, .. } | StreamingFrame::Pcm { n, .. } => n,
        }
    }
}

/// Multi-channel streaming Vorbis PCM driver.
///
/// Owns one [`OverlapAdd`] instance per channel and the stream-level
/// configuration ([`audio_channels`](Self::audio_channels), blocksizes,
/// `imdct_scale`). Drive it with [`push_packet`](Self::push_packet) once per
/// audio packet; the engine returns either a priming-step
/// [`StreamingFrame::Primed`] (first packet only) or a
/// [`StreamingFrame::Pcm`] holding the finished PCM samples for the
/// previous → current packet transition.
///
/// The engine is bring-your-own-packet — Ogg demuxing is the caller's
/// responsibility. [`reset`](Self::reset) returns every per-channel
/// overlap-add state to priming, e.g. after a seek.
#[derive(Debug, Clone)]
pub struct StreamingDecoder {
    /// Identification-header `[audio_channels]` (§4.2.2).
    audio_channels: u8,
    /// Identification-header `[blocksize_0]` (§4.2.2).
    blocksize_0: usize,
    /// Identification-header `[blocksize_1]` (§4.2.2).
    blocksize_1: usize,
    /// Vorbis-specific IMDCT normalization scalar — caller-supplied,
    /// because the staged fixture traces do not yet log post-IMDCT samples
    /// (see [`crate::imdct`] for the docs-gap details).
    imdct_scale: f32,
    /// One §4.3.8 overlap-add state per bitstream channel, in bitstream
    /// channel order (§4.3.9 rearrangement is a presentation concern
    /// handled above this module).
    overlap: Vec<OverlapAdd>,
}

impl StreamingDecoder {
    /// Build a fresh streaming decoder for an `audio_channels`-channel
    /// stream with the given identification-header geometry and IMDCT
    /// normalization scalar.
    ///
    /// `audio_channels` must match the stream's identification-header
    /// `[audio_channels]`; `blocksize_0` / `blocksize_1` must match
    /// `[blocksize_0]` / `[blocksize_1]`. `imdct_scale` is the
    /// caller-supplied deferred-normalization knob (see [`crate::imdct`]).
    #[must_use]
    pub fn new(
        audio_channels: u8,
        blocksize_0: usize,
        blocksize_1: usize,
        imdct_scale: f32,
    ) -> Self {
        let overlap = (0..audio_channels as usize)
            .map(|_| OverlapAdd::new())
            .collect();
        Self {
            audio_channels,
            blocksize_0,
            blocksize_1,
            imdct_scale,
            overlap,
        }
    }

    /// The stream's identification-header `[audio_channels]`.
    #[must_use]
    pub fn audio_channels(&self) -> u8 {
        self.audio_channels
    }

    /// The stream's identification-header `[blocksize_0]`.
    #[must_use]
    pub fn blocksize_0(&self) -> usize {
        self.blocksize_0
    }

    /// The stream's identification-header `[blocksize_1]`.
    #[must_use]
    pub fn blocksize_1(&self) -> usize {
        self.blocksize_1
    }

    /// The caller-supplied IMDCT normalization scalar.
    #[must_use]
    pub fn imdct_scale(&self) -> f32 {
        self.imdct_scale
    }

    /// Override the IMDCT normalization scalar on an already-constructed
    /// decoder. Useful when a callsite wants to experiment with several
    /// candidate values against the same fixture without re-allocating
    /// the per-channel state.
    pub fn set_imdct_scale(&mut self, scale: f32) {
        self.imdct_scale = scale;
    }

    /// `true` when every per-channel overlap-add state is still in the
    /// priming phase (no packet has been pushed since construction or the
    /// last [`reset`](Self::reset)).
    #[must_use]
    pub fn is_priming(&self) -> bool {
        self.overlap.iter().all(OverlapAdd::is_priming)
    }

    /// Reset every per-channel overlap-add state to the priming phase
    /// (§4.3.8 priming step). The next [`push_packet`](Self::push_packet)
    /// call will return [`StreamingFrame::Primed`].
    ///
    /// Use this after a seek or stream restart: the §4.3.8 overlap-add
    /// arithmetic is contextual to the previous packet's right-half tail,
    /// and that tail is no longer valid after a discontinuity.
    pub fn reset(&mut self) {
        for state in &mut self.overlap {
            state.reset();
        }
    }

    /// Drive one audio packet through the §4.3.1..§4.3.8 pipeline and
    /// return either [`StreamingFrame::Primed`] (first packet after
    /// construction or reset) or [`StreamingFrame::Pcm`] (every subsequent
    /// packet).
    ///
    /// * `reader` — an LSB-first bit reader positioned at the first bit
    ///   of an audio packet's bitstream (RFC-3533-stripped,
    ///   page-coalesced).
    /// * `setup` — the stream's parsed setup header.
    /// * `state` — the per-stream decoder cache built by
    ///   [`AudioDecoderState::new`].
    ///
    /// # Errors
    ///
    /// * [`StreamingError::Packet`] for any §4.3.1..§4.3.7 per-packet
    ///   driver failure ([`AudioPacketError`]).
    /// * [`StreamingError::Overlap`] for a §4.3.8 frame rejection
    ///   ([`OverlapError`]). Defensive: the per-packet driver always
    ///   produces a length-`n` frame with `n` a power of two `>= 64`,
    ///   which [`OverlapAdd::push_frame`] accepts unconditionally.
    /// * [`StreamingError::ChannelCountMismatch`] if the packet's
    ///   per-channel frame count differs from the decoder's configured
    ///   channel count. Defensive against hand-built setups.
    pub fn push_packet(
        &mut self,
        reader: &mut BitReaderLsb<'_>,
        setup: &VorbisSetupHeader,
        state: &AudioDecoderState,
    ) -> Result<StreamingFrame, StreamingError> {
        let outcome = decode_audio_packet_windowed(
            reader,
            setup,
            state,
            self.audio_channels,
            self.blocksize_0,
            self.blocksize_1,
            self.imdct_scale,
        )?;
        self.consume_outcome(outcome)
    }

    /// Drive one already-windowed packet outcome through the §4.3.8
    /// per-channel overlap-add state. Equivalent to the §4.3.8 tail of
    /// [`push_packet`](Self::push_packet); separated so callers that hold
    /// a [`WindowedPacketOutcome`] (e.g. from a buffered decode) can
    /// continue the streaming pipeline without re-driving the bit reader.
    ///
    /// # Errors
    ///
    /// See [`push_packet`](Self::push_packet).
    pub fn push_windowed(
        &mut self,
        outcome: WindowedPacketOutcome,
    ) -> Result<StreamingFrame, StreamingError> {
        self.consume_outcome(outcome)
    }

    fn consume_outcome(
        &mut self,
        outcome: WindowedPacketOutcome,
    ) -> Result<StreamingFrame, StreamingError> {
        let header = outcome.header();
        let frames = match outcome {
            WindowedPacketOutcome::Windowed { frames, .. }
            | WindowedPacketOutcome::ZeroedWindowed { frames, .. } => frames,
        };
        if frames.len() != self.overlap.len() {
            return Err(StreamingError::ChannelCountMismatch {
                expected: self.overlap.len(),
                got: frames.len(),
            });
        }

        // §4.3.8 priming: if any one channel is priming, every channel is
        // priming (we always push the same packet to every channel
        // simultaneously). Snapshot the state up front so the per-channel
        // loop below can answer "primed or not" without re-borrowing
        // `self.overlap`.
        let was_priming = self.overlap.iter().all(OverlapAdd::is_priming);

        let mut per_channel_pcm: Vec<Vec<f32>> = Vec::with_capacity(self.overlap.len());
        for (channel_idx, (state, frame)) in self.overlap.iter_mut().zip(frames.iter()).enumerate()
        {
            let pcm = state
                .push_frame(frame)
                .map_err(|source| StreamingError::Overlap {
                    channel: channel_idx,
                    source,
                })?;
            match pcm {
                None => {
                    // Priming-step: empty PCM for this channel. Push an
                    // empty vector so the per-channel index alignment is
                    // preserved; the caller-facing variant collapses to
                    // `Primed` below.
                    per_channel_pcm.push(Vec::new());
                }
                Some(samples) => {
                    per_channel_pcm.push(samples);
                }
            }
        }

        if was_priming {
            // Every per-channel overlap-add primed; emit the dedicated
            // variant rather than a length-zero Pcm.
            Ok(StreamingFrame::Primed {
                mode_number: header.mode_number,
                blockflag: header.blockflag,
                n: header.n,
            })
        } else {
            Ok(StreamingFrame::Pcm {
                mode_number: header.mode_number,
                blockflag: header.blockflag,
                n: header.n,
                per_channel_pcm,
            })
        }
    }

    /// Drain the right-half tail of the last pushed packet's window
    /// (§4.3.8 stream-end finishing). One [`OverlapAdd::finish`] call per
    /// channel, returning `None` if no packet has been pushed (or the
    /// engine was reset), or `Some(per_channel_tail)` where each entry is
    /// the channel's stored right half (length `n / 2` of the last pushed
    /// packet).
    ///
    /// §4.3.8 normally **discards** this tail: it would only finalize on
    /// the next packet's overlap-add, and on stream end there is no next
    /// packet. Some applications (e.g. flushing a finite encoded clip to
    /// PCM) still want it; this method exposes it. After the call the
    /// engine is back to priming.
    pub fn finish(&mut self) -> Option<Vec<Vec<f32>>> {
        let mut any_tail = false;
        let tails: Vec<Vec<f32>> = self
            .overlap
            .iter_mut()
            .map(|state| {
                let tail = state.finish().unwrap_or_default();
                if !tail.is_empty() {
                    any_tail = true;
                }
                tail
            })
            .collect();
        if any_tail {
            Some(tails)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::vorbis_window;

    /// Helper: build a length-`n` windowed unit ramp matching the §1.3.2
    /// window for a long block with both window flags false. The plateau
    /// is `1.0` so the IMDCT-then-window pipeline of a uniform spectrum
    /// becomes a known triangular shape; the windowed frame here is
    /// hand-built independently of the IMDCT module so the test exercises
    /// just the overlap-add stitching.
    fn synthetic_windowed_frame(n: usize, blocksize_0: usize, value: f32) -> Vec<f32> {
        let window =
            vorbis_window(n, blocksize_0, true, false, false).expect("hand-built long window");
        window.into_iter().map(|w| w * value).collect()
    }

    /// A minimum-viable `WindowedPacketOutcome::Windowed` with one
    /// channel — sidesteps the bit-stream synthesis so the test surface
    /// is just the streaming engine.
    fn hand_built_windowed(channels: usize, n: usize, blocksize_0: usize) -> WindowedPacketOutcome {
        let frames = (0..channels)
            .map(|ch| synthetic_windowed_frame(n, blocksize_0, ch as f32 + 1.0))
            .collect();
        WindowedPacketOutcome::Windowed {
            mode_number: 0,
            blockflag: true,
            n,
            previous_window_flag: false,
            next_window_flag: false,
            frames,
        }
    }

    /// Hand-built zeroed outcome — the §4.3.2 short-circuit shape.
    fn hand_built_zeroed(channels: usize, n: usize) -> WindowedPacketOutcome {
        WindowedPacketOutcome::ZeroedWindowed {
            mode_number: 1,
            blockflag: false,
            n,
            previous_window_flag: false,
            next_window_flag: false,
            frames: vec![vec![0.0f32; n]; channels],
        }
    }

    // ---- construction / accessors ----

    #[test]
    fn new_starts_in_priming_state() {
        let dec = StreamingDecoder::new(2, 64, 1024, 1.0);
        assert_eq!(dec.audio_channels(), 2);
        assert_eq!(dec.blocksize_0(), 64);
        assert_eq!(dec.blocksize_1(), 1024);
        assert_eq!(dec.imdct_scale(), 1.0);
        assert!(dec.is_priming());
    }

    #[test]
    fn set_imdct_scale_updates_field() {
        let mut dec = StreamingDecoder::new(1, 64, 1024, 1.0);
        dec.set_imdct_scale(0.5);
        assert_eq!(dec.imdct_scale(), 0.5);
    }

    // ---- priming + first PCM emission ----

    #[test]
    fn first_packet_primes_and_emits_no_pcm() {
        let mut dec = StreamingDecoder::new(1, 64, 1024, 1.0);
        let outcome = hand_built_windowed(1, 1024, 64);
        let frame = dec.push_windowed(outcome).unwrap();
        assert!(frame.is_primed(), "first push should prime");
        assert_eq!(frame.n(), 1024);
        match frame {
            StreamingFrame::Primed {
                mode_number,
                blockflag,
                n,
            } => {
                assert_eq!(mode_number, 0);
                assert!(blockflag);
                assert_eq!(n, 1024);
            }
            other => panic!("expected Primed, got {:?}", other),
        }
        assert!(!dec.is_priming(), "decoder should no longer be priming");
    }

    #[test]
    fn second_packet_emits_pcm_with_spec_return_length() {
        let mut dec = StreamingDecoder::new(1, 64, 1024, 1.0);
        // Prime with a long block.
        let _ = dec.push_windowed(hand_built_windowed(1, 1024, 64)).unwrap();
        // Push another long block; expected return length is
        // prev_n/4 + cur_n/4 = 256 + 256 = 512 (§4.3.8).
        let frame = dec.push_windowed(hand_built_windowed(1, 1024, 64)).unwrap();
        assert!(!frame.is_primed());
        let pcm = frame.pcm().unwrap();
        assert_eq!(pcm.len(), 1);
        assert_eq!(pcm[0].len(), 512);
    }

    #[test]
    fn mixed_block_sizes_use_spec_return_formula() {
        let mut dec = StreamingDecoder::new(1, 64, 1024, 1.0);
        // Prev = short (64), cur = long (1024) → 64/4 + 1024/4 = 16 + 256 = 272.
        let _ = dec.push_windowed(hand_built_windowed(1, 64, 64)).unwrap();
        let frame = dec.push_windowed(hand_built_windowed(1, 1024, 64)).unwrap();
        let pcm = frame.pcm().unwrap();
        assert_eq!(pcm[0].len(), 272);
    }

    // ---- multi-channel routing ----

    #[test]
    fn two_channel_decoder_emits_two_pcm_vectors() {
        let mut dec = StreamingDecoder::new(2, 64, 1024, 1.0);
        let _ = dec.push_windowed(hand_built_windowed(2, 1024, 64)).unwrap();
        let frame = dec.push_windowed(hand_built_windowed(2, 1024, 64)).unwrap();
        let pcm = frame.pcm().unwrap();
        assert_eq!(pcm.len(), 2);
        assert_eq!(pcm[0].len(), 512);
        assert_eq!(pcm[1].len(), 512);
        // The per-channel synthetic ramps used `value = channel + 1`, so
        // the second channel's PCM should be (asymptotically) twice the
        // first channel's PCM at every aligned position. The plateau
        // region carries both contributions multiplicatively in the same
        // ratio.
        for (a, b) in pcm[0].iter().zip(pcm[1].iter()) {
            if a.abs() > 1e-6 {
                let ratio = b / a;
                let diff = (ratio - 2.0).abs();
                assert!(
                    diff < 1e-4,
                    "channel ratio {ratio} not 2.0 (a={a}, b={b}, diff={diff})",
                );
            }
        }
    }

    #[test]
    fn channel_count_mismatch_rejected() {
        let mut dec = StreamingDecoder::new(1, 64, 1024, 1.0);
        // Hand-built outcome with two channels but a one-channel decoder.
        let outcome = hand_built_windowed(2, 1024, 64);
        match dec.push_windowed(outcome) {
            Err(StreamingError::ChannelCountMismatch {
                expected: 1,
                got: 2,
            }) => {}
            other => panic!("expected ChannelCountMismatch, got {:?}", other),
        }
    }

    // ---- zeroed packets propagate cleanly ----

    #[test]
    fn zeroed_packet_primes_with_n_pinned() {
        let mut dec = StreamingDecoder::new(1, 64, 1024, 1.0);
        let frame = dec.push_windowed(hand_built_zeroed(1, 64)).unwrap();
        assert!(frame.is_primed());
        assert_eq!(frame.n(), 64);
        match frame {
            StreamingFrame::Primed { blockflag, n, .. } => {
                assert!(!blockflag);
                assert_eq!(n, 64);
            }
            other => panic!("expected Primed, got {:?}", other),
        }
    }

    #[test]
    fn zeroed_after_normal_emits_zero_pcm() {
        let mut dec = StreamingDecoder::new(1, 64, 1024, 1.0);
        // Prime with a normal long block, then push a zeroed long block:
        // the right-half tail of the first frame at the plateau is `1.0`
        // (window value 1.0 × spectrum value 1.0), but the zeroed frame
        // contributes 0 at every index. PCM at the right boundary should
        // therefore carry just the previous-frame contribution.
        let _ = dec.push_windowed(hand_built_windowed(1, 1024, 64)).unwrap();
        let frame = dec.push_windowed(hand_built_zeroed(1, 1024)).unwrap();
        let pcm = frame.pcm().unwrap();
        assert_eq!(pcm[0].len(), 512);
        // At the plateau of the previous window the tail is constant 1.0;
        // the §4.3.8 return spans prev_center..cur_center exclusive of
        // the cur side that's still in the zeroed-frame's lead-in. Pick
        // an interior index well inside the plateau to verify the
        // previous contribution is preserved.
        // The previous window's plateau on the right half spans roughly
        // 512 + (n/4 - blocksize_0/4) = 512 + 240 ≤ idx ≤ 768 (long
        // block, n=1024, blocksize_0=64). The tail stored is the prev
        // window's right half (indices 512..1024); index 0 of the return
        // corresponds to prev local 512, etc. The plateau in the right
        // half persists until local 768, so return index 0..256
        // (= 768 - 512) is the plateau range. Indices well into it (e.g.
        // 0..100) should be ~1.0.
        for &v in pcm[0].iter().take(100) {
            let diff = (v - 1.0).abs();
            assert!(diff < 1e-4, "plateau sample {v} not 1.0 (diff {diff})");
        }
    }

    // ---- reset ----

    #[test]
    fn reset_returns_to_priming() {
        let mut dec = StreamingDecoder::new(1, 64, 1024, 1.0);
        let _ = dec.push_windowed(hand_built_windowed(1, 1024, 64)).unwrap();
        assert!(!dec.is_priming());
        dec.reset();
        assert!(dec.is_priming());
        // After reset, the next push should prime again.
        let frame = dec.push_windowed(hand_built_windowed(1, 1024, 64)).unwrap();
        assert!(frame.is_primed());
    }

    // ---- finish ----

    #[test]
    fn finish_on_unprimed_engine_returns_none() {
        let mut dec = StreamingDecoder::new(2, 64, 1024, 1.0);
        assert!(dec.finish().is_none());
    }

    #[test]
    fn finish_drains_per_channel_right_half_tail() {
        let mut dec = StreamingDecoder::new(2, 64, 1024, 1.0);
        let _ = dec.push_windowed(hand_built_windowed(2, 1024, 64)).unwrap();
        let tails = dec.finish().expect("primed engine has a tail");
        assert_eq!(tails.len(), 2);
        // Tail length is n/2 = 512.
        assert_eq!(tails[0].len(), 512);
        assert_eq!(tails[1].len(), 512);
        // After finish, the engine is back to priming.
        assert!(dec.is_priming());
    }

    // ---- error path: per-packet driver failure surfacing ----

    #[test]
    fn packet_driver_failure_surfaces_as_packet_variant() {
        // A hand-built setup with a packet missing the §4.3.1 header
        // exercises [`decode_audio_packet_windowed`]'s error path. We
        // re-use the existing `crate::audio::tests::write_used_packet` —
        // but those helpers live in audio.rs's tests module which is
        // private. Instead, push a malformed bitstream that the
        // per-packet driver will reject at the §4.3.1 header (mode list
        // empty).
        use crate::setup::VorbisSetupHeader;
        let setup = VorbisSetupHeader {
            codebooks: Vec::new(),
            time_placeholders: Vec::new(),
            floors: Vec::new(),
            residues: Vec::new(),
            mappings: Vec::new(),
            modes: Vec::new(),
            framing_flag: true,
        };
        let state = AudioDecoderState::new(&setup).unwrap();
        let mut dec = StreamingDecoder::new(1, 64, 1024, 1.0);
        let buf = [0u8; 4];
        let mut reader = BitReaderLsb::new(&buf);
        let err = dec.push_packet(&mut reader, &setup, &state).unwrap_err();
        match err {
            StreamingError::Packet(_) => {}
            other => panic!("expected StreamingError::Packet, got {:?}", other),
        }
    }

    // ---- Display ----

    #[test]
    fn error_display_pins_strings() {
        let s1 = format!(
            "{}",
            StreamingError::ChannelCountMismatch {
                expected: 2,
                got: 5,
            }
        );
        assert!(s1.contains("5 channels"), "{s1}");
        assert!(s1.contains("configured for 2"), "{s1}");

        let s2 = format!(
            "{}",
            StreamingError::Overlap {
                channel: 3,
                source: OverlapError::NotPowerOfTwo { n: 7 },
            }
        );
        assert!(s2.contains("channel 3"), "{s2}");
        assert!(s2.contains("not a positive power of two"), "{s2}");
    }
}

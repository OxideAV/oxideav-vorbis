//! Vorbis I encoder-side §4.3.8 framing-inverse primitive.
//!
//! This module is the encoder mirror of [`crate::overlap::OverlapAdd`].
//! Where the decoder consumes a sequence of overlapped-and-added
//! windowed frames and emits a continuous PCM stream, the encoder must
//! take a continuous PCM input stream and slice it into the same
//! sequence of overlapping windowed frames the §4.3.7 forward MDCT
//! consumes. The MDCT + IMDCT round-trip + decoder-side overlap-add
//! then reconstructs (a delayed copy of) the original PCM, by the
//! squared-sum-equals-one property of the §1.3.2 / §4.3.1 Vorbis
//! window pinned in the round-15 [`crate::overlap`] test
//! `squared_window_overlap_add_reconstructs_constant_signal`.
//!
//! # Spec geometry (§4.3.8)
//!
//! Per §4.3.8, the decoder aligns consecutive frames such that "the
//! 3/4 point of the previous window is aligned with the 1/4 point of
//! the current window." From this rule the
//! [`crate::overlap`] module derives the global-PCM placement of
//! frame N+1 relative to frame N:
//!
//! ```text
//! g_{N+1} = g_N + prev_n * 3/4 - cur_n / 4
//! ```
//!
//! where `prev_n = block(N)` and `cur_n = block(N+1)`. Setting frame 0
//! at `g_0 = 0`, this is a forward recurrence that places every frame
//! on the global timeline as a function of the chosen blocksize
//! sequence.
//!
//! The encoder framing primitive runs that recurrence forward:
//! each [`FrameSplitter::take_frame`] call reads `cur_n` PCM samples
//! out of an internal buffer, starting at the buffer-relative position
//! `g_N - read_base`, multiplies them by the supplied analysis window
//! (the same §4.3.1 window the decoder applies after IMDCT), and
//! returns the resulting length-`n` windowed time-domain block ready
//! to feed into the §4.3.7 forward MDCT
//! ([`crate::mdct::mdct_naive`]).
//!
//! # Reconstruction property
//!
//! With the symmetric §4.3.1 window's squared-overlap property — the
//! same property the round-15 decoder-side overlap-add reconstruction
//! test pins — the round-trip
//!
//! ```text
//!     pcm ──► FrameSplitter ──► windowed_frames
//!                                      │
//!                                      ▼
//!                       (identity per-frame stand-in for
//!                         MDCT→IMDCT→window_premultiply)
//!                                      │
//!                                      ▼
//!     pcm_out ◄── OverlapAdd ◄── windowed_frames
//! ```
//!
//! reproduces the input PCM exactly inside the return-range of every
//! non-priming `OverlapAdd::push_frame` call, modulo `f32` arithmetic.
//! This is exercised end-to-end by the
//! `splitter_then_overlap_add_round_trips_constant` and
//! `splitter_then_overlap_add_round_trips_ramp` tests in this module.
//!
//! Note that the round-trip absorbs both the analysis window (applied
//! here) and the synthesis window (applied by the decoder after
//! IMDCT) into the `w[i]² + w[i + n/2]² = 1` identity §4.3.8 relies
//! on. The pure-DSP framing primitive in this module covers the
//! analysis side; the decoder side already exists in
//! [`crate::synthesis::window_premultiply`].
//!
//! # First-frame priming
//!
//! [`crate::overlap::OverlapAdd::push_frame`] returns `None` on the
//! first call (the §4.3.8 "data is not returned from the first
//! frame" priming step). The encoder side mirrors this by allowing
//! the first frame to be requested when the input buffer holds only
//! the right half of the frame (the left half corresponds to
//! pre-stream silence and is conceptually padded with zeros). For
//! simplicity the public API requires the caller to supply zero-
//! padded "pre-roll" PCM on stream start; the
//! `splitter_first_frame_left_half_zero_padded` test pins the
//! geometry.

use crate::overlap::OverlapError;
use crate::synthesis::WindowPremultiplyError;

/// Errors that can arise from the encoder-side framing-inverse
/// primitive.
///
/// The driver above this module is the §4.3 audio-packet encoder
/// (a future round); these variants are the structural rejections the
/// driver will surface to its callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FramingError {
    /// The requested frame length `cur_n` was not a positive power of
    /// two, or was below the §4.3.8 minimum of 4. The Vorbis I §4.2.2
    /// constraint pins blocksizes to `{64, 128, 256, …, 8192}` so this
    /// is a defensive check on hand-built calls.
    NotPowerOfTwo {
        /// The offending frame length.
        n: usize,
    },
    /// The requested frame length was below the §4.3.8 minimum of 4.
    /// §4.3.8's "windowsize / 4" arithmetic requires `n >= 4`; §4.2.2
    /// already pins `blocksize >= 64`, so this is a defensive check.
    FrameTooSmall {
        /// The offending frame length.
        n: usize,
    },
    /// The internal PCM buffer does not yet hold enough samples to
    /// produce the requested frame. The caller should
    /// [`FrameSplitter::push_pcm`] more input and retry.
    NeedMoreInput {
        /// The smallest number of additional PCM samples that would
        /// satisfy the request.
        shortfall: usize,
    },
    /// The supplied analysis window length disagreed with the
    /// requested `cur_n`. The §4.3.1 window builder produces a length-
    /// `n` window per its `n` argument; a mismatch means the caller
    /// built the window from a different `n` than they asked the
    /// splitter to slice.
    WindowLengthMismatch {
        /// The requested frame length.
        frame_len: usize,
        /// The supplied window length.
        window_len: usize,
    },
}

impl core::fmt::Display for FramingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FramingError::NotPowerOfTwo { n } => write!(
                f,
                "vorbis §4.3.8 framing-inverse: frame length {n} is not a positive power of two"
            ),
            FramingError::FrameTooSmall { n } => write!(
                f,
                "vorbis §4.3.8 framing-inverse: frame length {n} is below the §4.3.8 minimum of 4"
            ),
            FramingError::NeedMoreInput { shortfall } => write!(
                f,
                "vorbis §4.3.8 framing-inverse: need {shortfall} more PCM sample(s) to produce \
                 the requested frame"
            ),
            FramingError::WindowLengthMismatch {
                frame_len,
                window_len,
            } => write!(
                f,
                "vorbis §4.3.8 framing-inverse: window length {window_len} disagrees with frame \
                 length {frame_len}"
            ),
        }
    }
}

impl std::error::Error for FramingError {}

impl From<OverlapError> for FramingError {
    /// Convenience: [`OverlapError::NotPowerOfTwo`] maps to
    /// [`FramingError::NotPowerOfTwo`] so callers running both the
    /// decoder-side overlap-add and the encoder-side framing splitter
    /// through a shared shape can normalise on one error type.
    fn from(value: OverlapError) -> Self {
        match value {
            OverlapError::NotPowerOfTwo { n } => FramingError::NotPowerOfTwo { n },
            OverlapError::FrameTooSmall { n } => FramingError::FrameTooSmall { n },
        }
    }
}

impl From<WindowPremultiplyError> for FramingError {
    /// Convenience: surface the length-mismatch check from the shared
    /// [`crate::synthesis::window_premultiply`] primitive — used by
    /// the splitter to apply the analysis window — under the
    /// splitter's own error type.
    fn from(value: WindowPremultiplyError) -> Self {
        match value {
            WindowPremultiplyError::LengthMismatch {
                frame_len,
                window_len,
            } => FramingError::WindowLengthMismatch {
                frame_len,
                window_len,
            },
        }
    }
}

/// One-channel §4.3.8 encoder-side framing splitter — the inverse of
/// [`crate::overlap::OverlapAdd`].
///
/// Internally holds a sliding PCM buffer plus the previous frame's
/// length. Successive [`take_frame`](Self::take_frame) calls produce
/// the next windowed time-domain block to feed into the §4.3.7
/// forward MDCT, advancing the buffer's read base per the §4.3.8
/// alignment recurrence
/// `g_{N+1} = g_N + prev_n * 3/4 - cur_n / 4`.
///
/// One instance per audio channel. The §4.3.9 channel layout is a
/// higher-level concern; this primitive operates on raw scalar
/// streams.
#[derive(Debug, Default, Clone)]
pub struct FrameSplitter {
    /// PCM samples available for the next frame, starting at the
    /// in-buffer position the current frame's global start maps to.
    /// Old samples below the current frame's start are dropped on
    /// every [`take_frame`](Self::take_frame) call.
    buffer: Vec<f32>,
    /// Length of the previous frame, or `None` if no frame has been
    /// produced yet (the priming state). Tracked separately from the
    /// buffer because the §4.3.8 stride uses `prev_n / 4` and only
    /// the splitter knows the previous frame's size.
    prev_n: Option<usize>,
}

impl FrameSplitter {
    /// Construct an empty (priming) framing splitter. Equivalent to
    /// [`FrameSplitter::default`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            prev_n: None,
        }
    }

    /// Reset to the priming state. Use this on a seek or stream
    /// restart; the next [`take_frame`](Self::take_frame) call will
    /// again treat its frame as the priming frame.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.prev_n = None;
    }

    /// `true` when no frame has been produced yet. The next
    /// [`take_frame`](Self::take_frame) call will be the §4.3.8
    /// priming frame: it reads `cur_n` samples starting at buffer
    /// offset 0 (callers must zero-pad the left half so the analysis
    /// window's lead-in slope lands on pre-stream silence).
    #[must_use]
    pub fn is_priming(&self) -> bool {
        self.prev_n.is_none()
    }

    /// Number of PCM samples currently buffered (not yet consumed).
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Append PCM samples to the internal buffer. The samples are
    /// appended at the end; previously-appended samples that have
    /// not yet been consumed by a [`take_frame`](Self::take_frame)
    /// call are retained.
    pub fn push_pcm(&mut self, pcm: &[f32]) {
        self.buffer.extend_from_slice(pcm);
    }

    /// Number of PCM samples the splitter needs at the buffer start
    /// to produce a frame of length `cur_n`. The splitter consumes
    /// `cur_n` samples per frame (the frame's full length) but
    /// advances its read base by the §4.3.8 stride
    /// `prev_n / 4 + cur_n / 4` on a non-priming frame and `cur_n / 2`
    /// on the priming frame (the priming step places `g_0 = 0` and
    /// the next frame at `g_1 = prev_n * 3/4 - cur_next/4`; the
    /// buffer-advance step on frame 0 is therefore `g_1 - g_0 -
    /// prev_n/2` plus the half-block stride — see the test block
    /// below).
    ///
    /// Always equals `cur_n`. Exposed as a method so callers can
    /// budget input buffer sizing without inspecting the geometry.
    #[must_use]
    pub fn frame_required_samples(&self, cur_n: usize) -> usize {
        cur_n
    }

    /// Produce the next windowed time-domain frame.
    ///
    /// `cur_n` is the requested frame length (must be a positive power
    /// of two `>= 4`). `analysis_window` is the §4.3.1 window for
    /// this frame, built via [`crate::synthesis::vorbis_window`] with
    /// the same `n` and the same `(blocksize_0, blockflag,
    /// previous_window_flag, next_window_flag)` parameters the
    /// decoder will use to reconstruct it from the audio-packet header
    /// (§4.3.1 step 4).
    ///
    /// Returns a length-`cur_n` vector of windowed PCM samples ready
    /// to feed into the §4.3.7 forward MDCT
    /// ([`crate::mdct::mdct_naive`]). On success, the splitter
    /// advances its read base by `prev_n / 4 + cur_n / 4` on a
    /// non-priming call (matching the §4.3.8 stride) and by `cur_n / 2`
    /// on the priming call (the §4.3.8 priming step's "no PCM is
    /// returned" half-block consumption).
    ///
    /// # Errors
    ///
    /// * [`FramingError::NotPowerOfTwo`] if `cur_n` is not a positive
    ///   power of two.
    /// * [`FramingError::FrameTooSmall`] if `cur_n < 4`.
    /// * [`FramingError::WindowLengthMismatch`] if
    ///   `analysis_window.len() != cur_n`.
    /// * [`FramingError::NeedMoreInput`] if the buffer holds fewer
    ///   than `cur_n` samples at the current read base. Callers
    ///   should [`push_pcm`](Self::push_pcm) more input and retry.
    pub fn take_frame(
        &mut self,
        cur_n: usize,
        analysis_window: &[f32],
    ) -> Result<Vec<f32>, FramingError> {
        if cur_n == 0 || !cur_n.is_power_of_two() {
            return Err(FramingError::NotPowerOfTwo { n: cur_n });
        }
        if cur_n < 4 {
            return Err(FramingError::FrameTooSmall { n: cur_n });
        }
        if analysis_window.len() != cur_n {
            return Err(FramingError::WindowLengthMismatch {
                frame_len: cur_n,
                window_len: analysis_window.len(),
            });
        }

        // Negative pending stride (a short→long transition): the
        // current frame's §4.3.8 global start precedes the buffered
        // head — which sits at the previous frame's center — by
        // `lead = cur_n/4 - prev_n/4`. Those samples were consumed by
        // the previous frame's left-half drop, but every one of them
        // falls inside the current frame's zero lead-in: §4.3.1 step 2
        // places the long block's hybrid rising edge at
        // `left_window_start = cur_n/4 - blocksize_0/4`, exactly
        // `lead` when the previous block was the short size (the only
        // §4.2.2 configuration a smaller previous block can have).
        // The windowed frame is therefore exact with the first `lead`
        // positions zero-filled — the multiplication by the window's
        // zero lead-in would have produced zeros regardless of the
        // sample values there.
        let lead = match self.prev_n {
            Some(prev_n) if cur_n > prev_n => cur_n / 4 - prev_n / 4,
            _ => 0,
        };
        let need = cur_n - lead;
        if self.buffer.len() < need {
            return Err(FramingError::NeedMoreInput {
                shortfall: need - self.buffer.len(),
            });
        }

        // §4.3.7 close: slice the frame's buffered samples (offset by
        // `lead` within the frame) and multiply pointwise by the
        // analysis window. The window's lead-in and tail zeros
        // (§4.3.1 step 4 / step 8) mean PCM samples outside the
        // analysis support are zeroed by the multiplication.
        let mut frame = vec![0.0f32; cur_n];
        for ((sample, &pcm), &w) in frame[lead..]
            .iter_mut()
            .zip(&self.buffer[..need])
            .zip(&analysis_window[lead..])
        {
            *sample = pcm * w;
        }

        // §4.3.8 advance: drop the consumed PCM. The stride is the
        // global advance of the next frame's start relative to the
        // current frame's start, i.e. `cur_n * 3 / 4 - next_n / 4`.
        // We don't know `next_n` yet — but the advance of the read
        // base is determined by the current frame alone if we split
        // it into "always drop cur_n/2 here; the next call's
        // priming-vs-non-priming branch consumes the remaining
        // `(prev_n - cur_n) / 4` (signed) inside the next-frame
        // arithmetic." That bookkeeping is messy.
        //
        // Cleaner formulation: keep the previous frame's right half
        // (length `cur_n / 2`) in the buffer for the next call to
        // overlap with. Drop only `cur_n / 2` samples — the current
        // frame's left half. This mirrors the decoder-side
        // `OverlapAdd::prev_right_half` storage exactly and makes
        // the next-call read base "the current frame's center
        // (windowsize/2)" — the §4.3.8 "from the center of the
        // previous window" anchor.
        //
        // The next call then advances by `next_n / 2` after slicing,
        // and the geometry composes: across calls N → N+1 the read
        // base moves by `prev_n / 2 + next_n / 2 - prev_n / 2 = next_n / 2`,
        // wait, that misplaces it. Re-derive below.
        //
        // Per §4.3.8, frame N+1's global start is
        // g_{N+1} = g_N + prev_n*3/4 - cur_n_{next}/4. If we keep
        // (frame N's right half) in the buffer, the buffer's logical
        // start is g_N + cur_n/2. The next call wants to slice cur_n_{next}
        // samples starting at g_{N+1} = g_N + prev_n*3/4 - cur_n_{next}/4
        // = (buffer_start_global) + prev_n/4 - cur_n_{next}/4.
        //
        // So the next call must drop `prev_n/4 - cur_n_{next}/4` from
        // the buffer start. That's a SIGNED quantity (negative when
        // next is larger), so we cannot do that here — we don't know
        // cur_n_{next} yet. We defer the post-overlap drop to the
        // next-call entry and do only the always-positive part
        // (drop `cur_n/2`, the current frame's left half) here.
        //
        // To make this work cleanly we record `prev_n` and let the
        // next call do its own `prev_n/4 - cur_n/4` adjustment
        // BEFORE slicing.
        //
        // Implementation: drop cur_n/2 samples here; on the next
        // call, before the buffer-length check, advance the read
        // base by `prev_n/4 - cur_n/4` (signed). On the priming
        // frame, prev_n is None and we treat the read base as
        // unchanged — the next call's adjustment will be exactly
        // (prev_n=cur_n)/4 - next_n/4.
        //
        // Done above already? No — we have to subtract the cur_n/2
        // here, then on the next entry do the prev_n/4 - cur_n/4
        // adjustment. Let's record prev_n before draining.
        //
        // Actually the cleanest split: in this function, after
        // emitting the windowed frame, drop EXACTLY `cur_n / 2`
        // samples from the buffer front. Then before the NEXT
        // call's slice, do the signed `prev_n/4 - cur_n/4` drop
        // (positive ⇒ drop; negative ⇒ would have to prepend, but
        // that means the previous frame's right half overlaps into
        // the next frame's left, which is exactly the overlap-add
        // input pattern — that's WHY the encoder needs the same
        // input twice and why a "drop only the unique part"
        // accounting is the right model).
        //
        // Note: when `next_n > prev_n`, `prev_n/4 - cur_n/4` is
        // negative, meaning we'd need samples already consumed.
        // That's exactly why we drop only `cur_n/2` here — keeping
        // the right half makes those samples available for the
        // overlap geometry of the next frame.
        let prev_n_was = self.prev_n;
        // Restore the head-at-center invariant: the frame's global
        // start is `lead` before the old head, so its center is
        // `cur_n/2 - lead` past it.
        let half = cur_n / 2 - lead;
        self.buffer.drain(..half);
        self.prev_n = Some(cur_n);

        // On a non-priming call, also apply the pending signed
        // adjustment from the previous-to-current transition: drop
        // `prev_n/4 - cur_n/4` more samples (clamped at the buffer
        // length; underflow is the negative-stride case where the
        // current frame's left half already overlaps the previous
        // frame's right half, which is the buffered region — no
        // additional drop is needed because the slice already
        // started inside the kept right half).
        //
        // Wait — the adjustment was supposed to happen BEFORE the
        // slice, not after. Re-examining: we sliced the frame from
        // buffer position 0. The frame's window has its analysis
        // peak at `cur_n/2`. We dropped `cur_n/2` after the slice,
        // landing the buffer's new position-0 at the frame's center
        // (windowsize/2) — the §4.3.8 "previous center" anchor.
        // The §4.3.8 stride from "previous center" to "next start"
        // is `prev_n/4 - next_n/4` (signed) — POSITIVE when
        // prev_n > next_n. That's the adjustment the next call needs
        // before its own slice.
        //
        // So: do nothing more here, and have the next call apply
        // the `prev_n/4 - cur_n/4` (with the now-stored prev_n) at
        // its entry. We need to keep `prev_n_was` for one more
        // wrinkle: the very first call's `prev_n_was` is `None`,
        // and the priming frame has no pending stride to apply on
        // its entry. The check is handled by `prev_n_was` being
        // `None` on entry, which we already store as `Some(cur_n)`
        // on the way out.
        let _ = prev_n_was; // priming-vs-non-priming reflected in self.prev_n already

        Ok(frame)
    }

    /// Drop the pending §4.3.8 stride adjustment from the previous
    /// take-frame call. Idempotent if no previous frame is recorded.
    ///
    /// The internal model:
    ///
    /// * On entry to a non-priming [`take_frame`](Self::take_frame)
    ///   call, the buffer's position 0 is the previous frame's
    ///   center (windowsize/2).
    /// * The current frame's global start is offset from the
    ///   previous center by `prev_n/4 - cur_n/4` (signed).
    /// * For `prev_n >= cur_n` this is non-negative and we drop
    ///   that many samples from the buffer start before slicing.
    /// * For `prev_n < cur_n` it is negative: the current frame's
    ///   left half overlaps into the buffered right half of the
    ///   previous frame. The slice still starts at buffer position
    ///   0 — the overlap is intrinsic and is what the windowed
    ///   reconstruction relies on.
    ///
    /// This method exposes the "drop the pending adjustment"
    /// operation as a public method so callers that want to skip
    /// the next frame (e.g. seeking) can advance the buffer without
    /// producing a windowed frame. Idempotent in the priming state.
    pub fn advance_pending_stride(&mut self, cur_n: usize) {
        if let Some(prev_n) = self.prev_n {
            let signed_drop = (prev_n as isize) / 4 - (cur_n as isize) / 4;
            if signed_drop > 0 {
                let to_drop = (signed_drop as usize).min(self.buffer.len());
                self.buffer.drain(..to_drop);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlap::OverlapAdd;
    use crate::synthesis::vorbis_window;

    // ---- error paths ----

    #[test]
    fn take_frame_rejects_non_power_of_two() {
        let mut splitter = FrameSplitter::new();
        splitter.push_pcm(&vec![0.5f32; 200]);
        let window = vec![1.0f32; 100];
        assert_eq!(
            splitter.take_frame(100, &window),
            Err(FramingError::NotPowerOfTwo { n: 100 })
        );
    }

    #[test]
    fn take_frame_rejects_zero_length() {
        let mut splitter = FrameSplitter::new();
        splitter.push_pcm(&[0.5f32; 16]);
        let window: Vec<f32> = Vec::new();
        assert_eq!(
            splitter.take_frame(0, &window),
            Err(FramingError::NotPowerOfTwo { n: 0 })
        );
    }

    #[test]
    fn take_frame_rejects_frame_smaller_than_four() {
        let mut splitter = FrameSplitter::new();
        splitter.push_pcm(&[0.5f32; 16]);
        let window = vec![1.0f32; 2];
        assert_eq!(
            splitter.take_frame(2, &window),
            Err(FramingError::FrameTooSmall { n: 2 })
        );
    }

    #[test]
    fn take_frame_rejects_window_length_mismatch() {
        let mut splitter = FrameSplitter::new();
        splitter.push_pcm(&vec![0.5f32; 64]);
        let window = vec![1.0f32; 32];
        assert_eq!(
            splitter.take_frame(64, &window),
            Err(FramingError::WindowLengthMismatch {
                frame_len: 64,
                window_len: 32,
            })
        );
    }

    #[test]
    fn take_frame_reports_need_more_input_shortfall() {
        let mut splitter = FrameSplitter::new();
        splitter.push_pcm(&[0.5f32; 40]);
        let window = vec![1.0f32; 64];
        assert_eq!(
            splitter.take_frame(64, &window),
            Err(FramingError::NeedMoreInput { shortfall: 24 })
        );
        // After pushing the missing samples, the next call must succeed.
        splitter.push_pcm(&[0.5f32; 24]);
        assert!(splitter.take_frame(64, &window).is_ok());
    }

    #[test]
    fn framing_error_display_strings_are_grep_friendly() {
        let e1 = FramingError::NotPowerOfTwo { n: 100 };
        let e2 = FramingError::FrameTooSmall { n: 2 };
        let e3 = FramingError::NeedMoreInput { shortfall: 5 };
        let e4 = FramingError::WindowLengthMismatch {
            frame_len: 64,
            window_len: 32,
        };
        for e in [&e1, &e2, &e3, &e4] {
            let s = format!("{e}");
            assert!(s.contains("framing-inverse"), "missing tag: {s}");
        }
        assert!(format!("{e1}").contains("100"));
        assert!(format!("{e2}").contains("2"));
        assert!(format!("{e3}").contains("5"));
        let s4 = format!("{e4}");
        assert!(s4.contains("64") && s4.contains("32"));
    }

    #[test]
    fn overlap_error_to_framing_error_conversion() {
        assert_eq!(
            FramingError::from(OverlapError::NotPowerOfTwo { n: 7 }),
            FramingError::NotPowerOfTwo { n: 7 }
        );
        assert_eq!(
            FramingError::from(OverlapError::FrameTooSmall { n: 2 }),
            FramingError::FrameTooSmall { n: 2 }
        );
    }

    #[test]
    fn window_premultiply_error_to_framing_error_conversion() {
        let from = FramingError::from(WindowPremultiplyError::LengthMismatch {
            frame_len: 64,
            window_len: 32,
        });
        assert_eq!(
            from,
            FramingError::WindowLengthMismatch {
                frame_len: 64,
                window_len: 32,
            }
        );
    }

    // ---- priming ----

    #[test]
    fn new_is_priming() {
        let splitter = FrameSplitter::new();
        assert!(splitter.is_priming());
        assert_eq!(splitter.buffered(), 0);
    }

    #[test]
    fn first_frame_takes_buffer_from_position_zero() {
        // On the priming frame, the splitter slices cur_n samples
        // starting at buffer position 0. The buffer is then drained of
        // cur_n/2 samples (the frame's left half).
        let mut splitter = FrameSplitter::new();
        let n = 64;
        // Push exactly one frame's worth.
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        splitter.push_pcm(&input);
        let window = vec![1.0f32; n];
        let frame = splitter.take_frame(n, &window).unwrap();
        assert_eq!(frame.len(), n);
        for (i, v) in frame.iter().enumerate() {
            assert!((v - i as f32).abs() < 1e-6, "frame[{i}] = {v}, want {i}");
        }
        assert!(!splitter.is_priming());
        // n/2 samples drained ⇒ n/2 left.
        assert_eq!(splitter.buffered(), n / 2);
    }

    #[test]
    fn reset_returns_to_priming() {
        let mut splitter = FrameSplitter::new();
        splitter.push_pcm(&vec![0.5f32; 128]);
        let window = vec![1.0f32; 64];
        splitter.take_frame(64, &window).unwrap();
        assert!(!splitter.is_priming());
        splitter.reset();
        assert!(splitter.is_priming());
        assert_eq!(splitter.buffered(), 0);
    }

    // ---- window application ----

    #[test]
    fn take_frame_applies_analysis_window() {
        let mut splitter = FrameSplitter::new();
        let n = 64;
        splitter.push_pcm(&vec![2.0f32; n]);
        let window: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let frame = splitter.take_frame(n, &window).unwrap();
        for (i, v) in frame.iter().enumerate() {
            let expected = 2.0 * (i as f32 * 0.01);
            assert!(
                (v - expected).abs() < 1e-6,
                "frame[{i}] = {v}, want {expected}"
            );
        }
    }

    #[test]
    fn take_frame_zeros_lead_in_and_tail_via_window() {
        // The §4.3.1 window has zero lead-in / tail (for hybrid blocks);
        // the multiplication absorbs the §4.3.8 "out of support" zeros.
        let mut splitter = FrameSplitter::new();
        let n = 256;
        splitter.push_pcm(&vec![5.0f32; n]);
        // Hybrid window: long block, short neighbours.
        let window = vorbis_window(n, 64, true, false, false).unwrap();
        let frame = splitter.take_frame(n, &window).unwrap();
        // Lead-in `0..n/4 - blocksize_0/4` = `0..48` is zero.
        for (i, v) in frame.iter().enumerate().take(48) {
            assert!(v.abs() < 1e-6, "lead-in frame[{i}] = {v}");
        }
        // Tail `n*3/4 + blocksize_0/4 .. n` = `208..256` is zero.
        for (i, v) in frame.iter().enumerate().skip(208) {
            assert!(v.abs() < 1e-6, "tail frame[{i}] = {v}");
        }
    }

    // ---- stride / read-base bookkeeping ----

    #[test]
    fn second_frame_starts_at_previous_center() {
        // After the priming frame of size 64, the buffer's logical
        // position 0 is the previous frame's center (windowsize/2 = 32).
        // Pushing a second equal-size frame's worth and slicing it should
        // pick up at sample 32 of the global stream.
        let mut splitter = FrameSplitter::new();
        let n = 64;
        // Push two frames' worth (positions 0..96, since after the
        // priming frame n/2=32 are drained, we need another 32 for a
        // second frame of size 64 → buffer needs to hold 64 at slice
        // time = original samples 32..96).
        let input: Vec<f32> = (0..96).map(|i| i as f32).collect();
        splitter.push_pcm(&input);
        let window = vec![1.0f32; n];
        let frame0 = splitter.take_frame(n, &window).unwrap();
        assert_eq!(frame0[0], 0.0);
        assert_eq!(frame0[63], 63.0);
        // Buffer now holds samples 32..96 (length 64).
        assert_eq!(splitter.buffered(), n);
        let frame1 = splitter.take_frame(n, &window).unwrap();
        // First sample of frame1 is global position 32.
        assert!((frame1[0] - 32.0).abs() < 1e-6);
        assert!((frame1[63] - 95.0).abs() < 1e-6);
    }

    #[test]
    fn third_frame_advances_per_spec_recurrence() {
        // Chain of three equal-size frames. The §4.3.8 advance from
        // frame N's center to frame N+1's start is prev_n/4 - cur_n/4.
        // For equal sizes that's zero: frame N+1 starts exactly at
        // frame N's center. So with input 0..N+N/2+N/2 = 0..2N, we get
        // frame0 from 0..N, frame1 from N/2..N/2+N, frame2 from
        // N..N+N.
        let mut splitter = FrameSplitter::new();
        let n = 64;
        let total = 2 * n;
        let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
        splitter.push_pcm(&input);
        let window = vec![1.0f32; n];
        let f0 = splitter.take_frame(n, &window).unwrap();
        let f1 = splitter.take_frame(n, &window).unwrap();
        let f2 = splitter.take_frame(n, &window).unwrap();
        // Frame 0: 0..64
        assert!((f0[0] - 0.0).abs() < 1e-6);
        assert!((f0[63] - 63.0).abs() < 1e-6);
        // Frame 1: 32..96
        assert!((f1[0] - 32.0).abs() < 1e-6);
        assert!((f1[63] - 95.0).abs() < 1e-6);
        // Frame 2: 64..128
        assert!((f2[0] - 64.0).abs() < 1e-6);
        assert!((f2[63] - 127.0).abs() < 1e-6);
    }

    #[test]
    fn stride_after_long_then_short_drops_extra_samples() {
        // prev_n=256 (long), cur_n=64 (short). §4.3.8 stride from
        // previous center to current start is prev_n/4 - cur_n/4 =
        // 64 - 16 = 48 (positive — drop 48 samples).
        //
        // Buffer model after frame 0 (long, n=256):
        //   * sliced 0..256 windowed
        //   * dropped 128 samples
        //   * buffer holds samples 128..(whatever pushed length is).
        //
        // For frame 1 to start at global position 256*3/4 - 64/4 =
        // 192 - 16 = 176, we need buffer position 0 → global 128 to
        // become buffer position 0 → global 176, i.e. drop 48 samples
        // first via advance_pending_stride.
        let mut splitter = FrameSplitter::new();
        let n0 = 256;
        let n1 = 64;
        let input: Vec<f32> = (0..300).map(|i| i as f32).collect();
        splitter.push_pcm(&input);
        let window0 = vec![1.0f32; n0];
        let window1 = vec![1.0f32; n1];
        let _f0 = splitter.take_frame(n0, &window0).unwrap();
        assert_eq!(splitter.buffered(), 300 - 128);
        splitter.advance_pending_stride(n1);
        // After the pending-stride drop, buffer holds samples
        // 176..300 (length 124).
        assert_eq!(splitter.buffered(), 300 - 128 - 48);
        let f1 = splitter.take_frame(n1, &window1).unwrap();
        // f1[0] is global position 176.
        assert!((f1[0] - 176.0).abs() < 1e-6, "f1[0]={}", f1[0]);
        assert!((f1[63] - (176.0 + 63.0)).abs() < 1e-6, "f1[63]={}", f1[63]);
    }

    #[test]
    fn stride_after_short_then_long_zero_fills_the_lead_in() {
        // prev_n=64, cur_n=256. §4.3.8 stride from previous center to
        // current start = 16 - 64 = -48 (negative): the long frame's
        // global start precedes the buffered head (the previous
        // frame's center) by `lead = 256/4 - 64/4 = 48` samples. Those
        // samples were consumed by the previous frame's left-half
        // drop, but they fall entirely inside the long frame's §4.3.1
        // zero lead-in (`left_window_start = 256/4 - 64/4 = 48` for a
        // long block lapping a short predecessor), so the splitter
        // zero-fills positions 0..48 and aligns the buffered samples
        // at position 48 — where the window's rising edge begins.
        let mut splitter = FrameSplitter::new();
        let n0 = 64;
        let n1 = 256;
        let input: Vec<f32> = (0..400).map(|i| i as f32).collect();
        splitter.push_pcm(&input);
        let window0 = vec![1.0f32; n0];
        // The real §4.3.1 hybrid window (long block, short previous
        // neighbour): zero lead-in over 0..48, rising edge 48..112.
        let window1 = vorbis_window(n1, n0, true, false, true).unwrap();
        let _f0 = splitter.take_frame(n0, &window0).unwrap();
        // After frame 0, buffer holds 400 - 32 = 368 samples (samples
        // 32..400).
        assert_eq!(splitter.buffered(), 368);
        let buffered_before = splitter.buffered();
        splitter.advance_pending_stride(n1);
        // Negative stride ⇒ nothing for the pending-stride drop.
        assert_eq!(splitter.buffered(), buffered_before);
        let f1 = splitter.take_frame(n1, &window1).unwrap();
        // Zero-filled lead-in.
        for (i, v) in f1.iter().enumerate().take(48) {
            assert_eq!(*v, 0.0, "lead-in f1[{i}] = {v}");
        }
        // Position 48 carries global sample 32 (the previous center)
        // weighted by the window's first rising-edge value: the frame's
        // §4.3.8 global start is 32 - 48 = -16, so index k holds
        // global sample k - 16.
        for (k, v) in f1.iter().enumerate().skip(48) {
            let expected = (k as f32 - 16.0) * window1[k];
            assert!(
                (v - expected).abs() < 1e-4,
                "f1[{k}] = {v}, want {expected}"
            );
        }
        // The head-at-center invariant: the frame's center is at
        // global -16 + 128 = 112, i.e. 80 samples past the old head —
        // the drain is cur_n/2 - lead = 128 - 48 = 80.
        assert_eq!(splitter.buffered(), buffered_before - 80);
    }

    #[test]
    fn mixed_blocksize_chain_round_trips_through_overlap_add() {
        // The full §4.3.8 mixed-size contract: a short/long blockflag
        // sequence with the real §4.3.1 hybrid windows, driven through
        // FrameSplitter → (identity transform stand-in) → second
        // window multiply → OverlapAdd, must reconstruct the input
        // inside every returned range — the w² TDAC identity holds
        // across long↔short transitions because the hybrid edges pair
        // slopes of equal length.
        let n0 = 64;
        let n1 = 256;
        let flags = [false, false, true, true, false, false, true, false];
        let pcm: Vec<f32> = (0..1024).map(|i| 0.5 + (i as f32 * 0.013).sin()).collect();
        let mut splitter = FrameSplitter::new();
        splitter.push_pcm(&pcm);
        let mut adder = OverlapAdd::new();
        // Global position of the first returned sample: frame 0's
        // center (no pre-roll in this test, mirroring the equal-size
        // round-trip tests above).
        let mut out_pos = n0 / 2;
        let mut returned = 0usize;
        for (f, &flag) in flags.iter().enumerate() {
            let n = if flag { n1 } else { n0 };
            let prev = flag && (f == 0 || flags[f - 1]);
            let next = flag && (f + 1 == flags.len() || flags[f + 1]);
            let window = vorbis_window(n, n0, flag, prev, next).unwrap();
            splitter.advance_pending_stride(n);
            let frame = splitter.take_frame(n, &window).unwrap();
            let windowed: Vec<f32> = frame.iter().zip(&window).map(|(s, w)| s * w).collect();
            if let Some(out) = adder.push_frame(&windowed).unwrap() {
                for (k, v) in out.iter().enumerate() {
                    let expected = pcm[out_pos + k];
                    assert!(
                        (v - expected).abs() < 1e-4,
                        "frame {f} out[{k}] (global {}) = {v}, want {expected}",
                        out_pos + k
                    );
                }
                out_pos += out.len();
                returned += out.len();
            }
        }
        // Every §4.3.8 lap must have been returned: Σ (n_prev+n_cur)/4.
        let mut expect = 0usize;
        for f in 1..flags.len() {
            let np = if flags[f - 1] { n1 } else { n0 };
            let nc = if flags[f] { n1 } else { n0 };
            expect += (np + nc) / 4;
        }
        assert_eq!(returned, expect);
    }

    // ---- end-to-end round-trip with OverlapAdd ----

    #[test]
    fn splitter_first_frame_left_half_zero_padded() {
        // §4.3.8 priming convention: the very first frame's left half
        // corresponds to pre-stream silence and must be zero-padded
        // by the caller. The decoder's `OverlapAdd::push_frame(frame0)`
        // call returns `None` (no PCM emitted), so the "lost" pre-
        // stream samples never reach the output and the priming
        // simply matches the encoder's left-half zero padding.
        //
        // This test pins the convention: caller pushes a zero-padded
        // priming PCM block, takes a windowed frame, and the result
        // has the expected left-half-zero shape.
        let n = 64;
        let mut left_half_zeros = vec![0.0f32; n / 2];
        let right_half: Vec<f32> = (0..(n / 2)).map(|i| (i + 1) as f32).collect();
        left_half_zeros.extend_from_slice(&right_half);
        let mut splitter = FrameSplitter::new();
        splitter.push_pcm(&left_half_zeros);
        let window = vec![1.0f32; n];
        let frame = splitter.take_frame(n, &window).unwrap();
        for v in frame.iter().take(n / 2) {
            assert!(v.abs() < 1e-6);
        }
        for (i, v) in frame.iter().skip(n / 2).enumerate() {
            let expected = (i + 1) as f32;
            assert!((v - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn splitter_then_overlap_add_round_trips_constant() {
        // End-to-end pipeline: PCM → FrameSplitter → (identity stand-in
        // for MDCT/IMDCT) → window_premultiply → OverlapAdd → PCM.
        // With the symmetric §4.3.1 window, the squared-overlap
        // identity w[i]² + w[i + n/2]² = 1 means a constant input
        // stream reconstructs to the same constant inside the
        // OverlapAdd return-range.
        let n = 128;
        let window = vorbis_window(n, 64, false, true, true).unwrap();
        let mut splitter = FrameSplitter::new();
        let mut adder = OverlapAdd::new();
        // Generate enough PCM to drive three frames through the
        // splitter (need 2n samples for three equal-size frames).
        let pcm = vec![1.0f32; 2 * n];
        splitter.push_pcm(&pcm);
        let f0 = splitter.take_frame(n, &window).unwrap();
        let f1 = splitter.take_frame(n, &window).unwrap();
        let f2 = splitter.take_frame(n, &window).unwrap();
        // Each frame is `pcm * w`. The decoder side applies a second
        // multiplication by `w`, producing `pcm * w²`. Overlap-add
        // sums w[i]² (current left) and w[i + n/2]² (previous right)
        // ⇒ constant 1.0 × pcm.
        let weight_frame = |frame: Vec<f32>| -> Vec<f32> {
            frame
                .iter()
                .zip(window.iter())
                .map(|(s, w)| s * w)
                .collect()
        };
        let wf0 = weight_frame(f0);
        let wf1 = weight_frame(f1);
        let wf2 = weight_frame(f2);
        adder.push_frame(&wf0).unwrap();
        let out01 = adder.push_frame(&wf1).unwrap().unwrap();
        let out12 = adder.push_frame(&wf2).unwrap().unwrap();
        // The returned PCM samples should all be 1.0 (the constant
        // input) to within f32 tolerance.
        for v in out01.iter().chain(out12.iter()) {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "reconstruction failed: out={v}, want 1.0"
            );
        }
    }

    #[test]
    fn splitter_then_overlap_add_round_trips_ramp() {
        // Same pipeline as the constant case but with a ramp input.
        // The squared-overlap identity is signal-independent, so the
        // ramp must also be reconstructed inside the OverlapAdd
        // return-range.
        let n = 128;
        let window = vorbis_window(n, 64, false, true, true).unwrap();
        let mut splitter = FrameSplitter::new();
        let mut adder = OverlapAdd::new();
        let pcm: Vec<f32> = (0..(2 * n)).map(|i| i as f32 * 0.01).collect();
        splitter.push_pcm(&pcm);
        let f0 = splitter.take_frame(n, &window).unwrap();
        let f1 = splitter.take_frame(n, &window).unwrap();
        let f2 = splitter.take_frame(n, &window).unwrap();
        let weight = |frame: Vec<f32>| -> Vec<f32> {
            frame
                .iter()
                .zip(window.iter())
                .map(|(s, w)| s * w)
                .collect()
        };
        let wf0 = weight(f0);
        let wf1 = weight(f1);
        let wf2 = weight(f2);
        adder.push_frame(&wf0).unwrap();
        let out01 = adder.push_frame(&wf1).unwrap().unwrap();
        let out12 = adder.push_frame(&wf2).unwrap().unwrap();
        // OverlapAdd's first non-priming output covers global
        // [n/2 .. n) of the input (the "previous center to current
        // center" range). Verify per-sample reconstruction.
        for (k, v) in out01.iter().enumerate() {
            let expected = (n / 2 + k) as f32 * 0.01;
            assert!(
                (v - expected).abs() < 1e-4,
                "out01[{k}] = {v}, want {expected}"
            );
        }
        for (k, v) in out12.iter().enumerate() {
            let expected = (n + k) as f32 * 0.01;
            assert!(
                (v - expected).abs() < 1e-4,
                "out12[{k}] = {v}, want {expected}"
            );
        }
    }

    // ---- buffered() / push_pcm() sanity ----

    #[test]
    fn push_pcm_appends() {
        let mut splitter = FrameSplitter::new();
        splitter.push_pcm(&[1.0, 2.0, 3.0]);
        assert_eq!(splitter.buffered(), 3);
        splitter.push_pcm(&[4.0, 5.0]);
        assert_eq!(splitter.buffered(), 5);
    }

    #[test]
    fn frame_required_samples_returns_cur_n() {
        let splitter = FrameSplitter::new();
        assert_eq!(splitter.frame_required_samples(64), 64);
        assert_eq!(splitter.frame_required_samples(256), 256);
        assert_eq!(splitter.frame_required_samples(8192), 8192);
    }

    #[test]
    fn advance_pending_stride_is_idempotent_in_priming() {
        let mut splitter = FrameSplitter::new();
        splitter.push_pcm(&vec![0.0f32; 64]);
        assert_eq!(splitter.buffered(), 64);
        splitter.advance_pending_stride(64);
        // Priming ⇒ no-op.
        assert_eq!(splitter.buffered(), 64);
        splitter.advance_pending_stride(256);
        assert_eq!(splitter.buffered(), 64);
    }
}

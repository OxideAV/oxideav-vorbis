//! Vorbis I overlap-add primitive (§4.3.8 "overlap add").
//!
//! This module is the §4.3.8 stage of the §4.3 audio-packet decode
//! pipeline. The pipeline immediately upstream of §4.3.8 is:
//!
//! 1. §4.3.2 floor decode (per channel, submap-routed)
//! 2. §4.3.3 nonzero-vector propagate
//! 3. §4.3.4 residue decode (per submap)
//! 4. §4.3.5 inverse coupling
//! 5. §4.3.6 dot product → per-channel **audio spectrum** of length `n/2`
//! 6. §4.3.7 **inverse MDCT** → per-channel time-domain frame of length `n`
//! 7. §4.3.7 windowing — multiply the IMDCT output by the §1.3.2 / §4.3.1
//!    Vorbis window of length `n`
//! 8. **§4.3.8 overlap-add** (this module) — combine the new windowed
//!    frame with the right-hand tail of the previous windowed frame to
//!    produce the finished PCM samples for output, and store this
//!    frame's right-hand tail for the next frame
//!
//! §4.3.7 (the inverse MDCT itself) is gated on the Vorbis I spec's
//! externally-cited reference `[1]` (T. Sporer / K. Brandenburg /
//! B. Edler), barred by the workspace clean-room policy. §4.3.8 by
//! contrast is fully self-contained in the Vorbis I spec body
//! (§4.3.8 + §1.3.2 "Window shape decode") and is the kind of pure
//! DSP primitive that has no MDCT-specific shape: it just adds two
//! time-domain signals with a defined alignment. Implementing it as
//! a standalone module ahead of the §4.3.7 round means the IMDCT-
//! round agent only has to plug in a transform and a multiplication;
//! the surrounding pipeline shape, the bookkeeping, and the return-
//! range arithmetic are all done.
//!
//! # Spec text (Vorbis I §4.3.8, verbatim)
//!
//! > "Windowed MDCT output is overlapped and added with the right hand
//! > data of the previous window such that the 3/4 point of the previous
//! > window is aligned with the 1/4 point of the current window (as
//! > illustrated in section 1.3.2, 'Window shape decode (long windows
//! > only)'). The overlapped portion produced from overlapping the
//! > previous and current frame data is finished data to be returned by
//! > the decoder. This data spans from the center of the previous window
//! > to the center of the current window. In the case of same-sized
//! > windows, the amount of data to return is one-half block consisting
//! > of and only of the overlapped portions. When overlapping a short
//! > and long window, much of the returned range does not actually
//! > overlap. This does not damage transform orthogonality. Pay
//! > attention however to returning the correct data range; the amount
//! > of data to be returned is:
//! >
//! > ```text
//! > window_blocksize(previous_window)/4 +
//! >     window_blocksize(current_window)/4
//! > ```
//! >
//! > from the center (element windowsize/2) of the previous window to
//! > the center (element windowsize/2-1, inclusive) of the current
//! > window.
//! >
//! > Data is not returned from the first frame; it must be used to
//! > 'prime' the decode engine. The encoder accounts for this priming
//! > when calculating PCM offsets; after the first frame, the proper
//! > PCM output offset is '0' (as no data has been returned yet)."
//!
//! # Geometry
//!
//! The spec's "3/4 point aligned with 1/4 point" rule fixes the relative
//! offset of two consecutive frames in the global PCM timeline. Take a
//! previous frame of size `prev_n` and a current frame of size `cur_n`,
//! both indexed `0 .. n` locally. If the previous frame starts at global
//! position 0:
//!
//! * The previous-frame `3/4` point is at global position `prev_n * 3 / 4`.
//! * The current-frame `1/4` point is at local index `cur_n / 4`.
//! * Aligning them sets the current frame to start at global position
//!   `prev_n * 3 / 4 - cur_n / 4` (signed — negative when a short block
//!   is followed by a long block).
//!
//! From this alignment, the spec's "return from previous center to
//! current center" range is:
//!
//! * Return-range start (global) = `prev_n / 2` (previous center).
//! * Return-range end (global, exclusive) = `cur_global_start +
//!   cur_n / 2` (one past the current center).
//! * Return-range length = `prev_n / 4 + cur_n / 4` (a quarter of each
//!   block — matches the spec formula exactly).
//!
//! For each return-range position `k` (`0 .. return_len`), the global
//! position is `prev_n / 2 + k`. From that:
//!
//! * Previous frame contributes when `global < prev_n`, i.e.
//!   `k < prev_n / 2`. The previous-frame local index is
//!   `prev_n / 2 + k`. (Note: this is always inside the previous
//!   frame's right half, `[prev_n / 2 .. prev_n)`.)
//! * Current frame contributes when `global >= cur_global_start`,
//!   i.e. `k >= prev_n / 4 - cur_n / 4` (a negative threshold means
//!   the current frame already started; the contribution is
//!   immediate). The current-frame local index is
//!   `global - cur_global_start = k - (prev_n / 4 - cur_n / 4)`.
//!
//! The overlap-add output at each `k` is the sum of those two
//! contributions (either 0 if the corresponding frame doesn't cover
//! that global position).
//!
//! Because every previous-contribution position is in the previous
//! frame's right half, the per-frame state this module needs is **just
//! the previous frame's right half** (length `prev_n / 2`). Storing only
//! the right half keeps the state proportional to the block size rather
//! than the full frame.
//!
//! # First-frame priming
//!
//! Per the spec's final paragraph, "Data is not returned from the first
//! frame; it must be used to 'prime' the decode engine." [`OverlapAdd`]
//! reflects this: the first call to [`OverlapAdd::push_frame`] stores
//! the frame's right half and returns `None`. Every subsequent call
//! returns the finished PCM range for the previous → current pair.

use crate::synthesis::WindowError;

/// Errors that can arise from the §4.3.8 overlap-add primitive.
///
/// The driver above this module already validates frame lengths through
/// the §4.3.1 `[n]` reader (which proves `n` is a positive power of two
/// per §4.2.2) and the [`crate::synthesis::vorbis_window`] generator
/// (same validation). These error variants are defensive checks against
/// hand-built calls where those upstream invariants are not in place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlapError {
    /// The pushed frame's length was not a positive power of two. Vorbis
    /// blocksizes are constrained to `{64, 128, 256, … , 8192}` (§4.2.2),
    /// and §4.3.8's "previous_window_blocksize / 4 + current_window_
    /// blocksize / 4" formula requires both sides to be divisible by 4
    /// (and thus a positive power of two with `n >= 4`).
    NotPowerOfTwo {
        /// The offending frame length.
        n: usize,
    },
    /// The pushed frame's length was less than 4. The §4.3.8 return-
    /// range arithmetic divides by 4 unconditionally; a frame smaller
    /// than 4 cannot satisfy the spec's "windowsize / 4" division.
    /// §4.2.2 already pins `blocksize >= 64`, so this is a defensive
    /// check.
    FrameTooSmall {
        /// The offending frame length.
        n: usize,
    },
}

impl core::fmt::Display for OverlapError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OverlapError::NotPowerOfTwo { n } => {
                write!(
                    f,
                    "vorbis overlap-add: frame length {n} is not a positive power of two"
                )
            }
            OverlapError::FrameTooSmall { n } => write!(
                f,
                "vorbis overlap-add: frame length {n} is below the §4.3.8 minimum of 4"
            ),
        }
    }
}

impl std::error::Error for OverlapError {}

impl From<OverlapError> for WindowError {
    /// Convenience: [`OverlapError::NotPowerOfTwo`] maps to
    /// [`WindowError::NotPowerOfTwo`] when callers want to surface
    /// overlap-add failures through the umbrella window-error type.
    fn from(value: OverlapError) -> Self {
        match value {
            OverlapError::NotPowerOfTwo { n } | OverlapError::FrameTooSmall { n } => {
                WindowError::NotPowerOfTwo { n }
            }
        }
    }
}

/// One-channel §4.3.8 overlap-add state.
///
/// Holds the **right half** of the previous windowed frame
/// (length `prev_n / 2`) so the next call can add it into the new
/// frame's left half. The first call after construction primes the
/// state and returns no PCM; every subsequent call returns
/// `prev_n / 4 + cur_n / 4` finished samples (§4.3.8 return-range
/// formula).
///
/// One instance per audio channel. The §4.3.9 channel ordering is a
/// presentation concern handled by the consumer above this module; this
/// primitive operates on raw scalar streams and does not know which
/// physical channel a given instance represents.
#[derive(Debug, Default, Clone)]
pub struct OverlapAdd {
    /// Right half of the previous windowed frame, or `None` if no frame
    /// has been pushed yet (the priming state).
    ///
    /// Length is the previous frame's `n / 2`; the previous frame's full
    /// length `n` is `prev_right_half.len() * 2`.
    prev_right_half: Option<Vec<f32>>,
}

impl OverlapAdd {
    /// Construct an empty (priming) overlap-add state. Equivalent to
    /// [`OverlapAdd::default`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            prev_right_half: None,
        }
    }

    /// Reset to the priming state. Use this on a seek or stream restart;
    /// the next [`push_frame`](Self::push_frame) call will again return
    /// `None`.
    pub fn reset(&mut self) {
        self.prev_right_half = None;
    }

    /// `true` when the state has not yet received any frame. The next
    /// [`push_frame`](Self::push_frame) call will prime and return
    /// `None`.
    #[must_use]
    pub fn is_priming(&self) -> bool {
        self.prev_right_half.is_none()
    }

    /// Length the next [`push_frame`](Self::push_frame) call will return,
    /// given the new frame's length `cur_n`. Returns `None` in the
    /// priming state (no output).
    ///
    /// Per §4.3.8 the return length is
    /// `previous_window_blocksize / 4 + current_window_blocksize / 4`.
    #[must_use]
    pub fn next_output_len(&self, cur_n: usize) -> Option<usize> {
        let prev_half = self.prev_right_half.as_ref()?;
        let prev_n = prev_half.len() * 2;
        Some(prev_n / 4 + cur_n / 4)
    }

    /// Length of the right-hand tail this state currently holds. Returns
    /// `0` in the priming state.
    #[must_use]
    pub fn stored_tail_len(&self) -> usize {
        self.prev_right_half.as_ref().map_or(0, Vec::len)
    }

    /// Push the next windowed time-domain frame and produce the finished
    /// PCM samples for the previous → current transition.
    ///
    /// `windowed` must be the IMDCT output (length `n`) **already
    /// multiplied** by the Vorbis window of the same length (the §4.3.7
    /// step right before §4.3.8 — see §1.3.2 step 9 in the decode
    /// procedure). `n` must be a positive power of two `>= 4`.
    ///
    /// Returns `Ok(None)` on the first call (the spec's priming step),
    /// `Ok(Some(samples))` on every subsequent call with the
    /// `prev_n / 4 + cur_n / 4` finished PCM values per §4.3.8.
    ///
    /// # Errors
    ///
    /// [`OverlapError::NotPowerOfTwo`] if `windowed.len()` is not a
    /// positive power of two, or [`OverlapError::FrameTooSmall`] if it
    /// is below the spec minimum of 4.
    pub fn push_frame(&mut self, windowed: &[f32]) -> Result<Option<Vec<f32>>, OverlapError> {
        let cur_n = windowed.len();
        if cur_n == 0 || !cur_n.is_power_of_two() {
            return Err(OverlapError::NotPowerOfTwo { n: cur_n });
        }
        if cur_n < 4 {
            return Err(OverlapError::FrameTooSmall { n: cur_n });
        }

        let cur_half = cur_n / 2;

        // Build the next-call right-half tail from this frame's upper
        // half before any borrow of `self.prev_right_half`.
        let new_right_half = windowed[cur_half..].to_vec();

        let output = match self.prev_right_half.take() {
            // §4.3.8 priming: first frame produces no output.
            None => None,
            Some(prev_right_half) => {
                let prev_n = prev_right_half.len() * 2;
                let return_len = prev_n / 4 + cur_n / 4;
                let mut out = Vec::with_capacity(return_len);

                // §4.3.8: cur_global_start = prev_n*3/4 - cur_n/4. The
                // sign of the offset matters; use signed arithmetic to
                // keep the "current already started" (negative offset)
                // case clean.
                let prev_half = prev_n / 2;
                let prev_quarter = prev_n / 4;
                let cur_quarter = cur_n / 4;
                // current_offset_in_return = cur_global_start - prev_n/2
                //                          = prev_n*3/4 - cur_n/4 - prev_n/2
                //                          = prev_n/4 - cur_n/4
                // Signed: this can be negative when prev_n < cur_n.
                let cur_offset_in_return: isize = prev_quarter as isize - cur_quarter as isize;

                // The loop body indexes BOTH `prev_right_half` (at `k`)
                // and `windowed` (at a `k`-derived offset). Rewriting it
                // as an iterator over one of the two arrays would
                // obscure the §4.3.8 alignment arithmetic; the loop
                // index `k` is itself the global-PCM-position-relative
                // coordinate the spec uses.
                #[allow(clippy::needless_range_loop)]
                for k in 0..return_len {
                    // Previous contribution: present iff k < prev_n/2.
                    let prev_contrib = if k < prev_half {
                        // prev_right_half is indexed 0 .. prev_n/2, where
                        // entry j corresponds to previous-frame local
                        // index prev_n/2 + j.
                        prev_right_half[k]
                    } else {
                        0.0
                    };

                    // Current contribution: present iff
                    // global >= cur_global_start, i.e.
                    // k >= cur_offset_in_return.
                    let cur_contrib = {
                        let signed_k = k as isize;
                        if signed_k >= cur_offset_in_return {
                            let cur_idx = (signed_k - cur_offset_in_return) as usize;
                            // The current frame is length cur_n; the only
                            // way cur_idx could exceed it is if return_len
                            // were inconsistent with the geometry. We hold
                            // to the spec formula, so cur_idx is always in
                            // [0, cur_n/2 + ... ).
                            //
                            // Concretely, the maximum k is return_len - 1
                            // = prev_n/4 + cur_n/4 - 1, so max cur_idx is
                            // (prev_n/4 + cur_n/4 - 1) - (prev_n/4 -
                            // cur_n/4) = cur_n/2 - 1, exactly at the
                            // current center (windowsize/2 - 1) as the
                            // spec mandates.
                            windowed[cur_idx]
                        } else {
                            0.0
                        }
                    };

                    out.push(prev_contrib + cur_contrib);
                }

                Some(out)
            }
        };

        self.prev_right_half = Some(new_right_half);
        Ok(output)
    }

    /// Drain the right-hand tail of the last frame as the stream-tail
    /// finishing samples.
    ///
    /// §4.3.8 mandates that "data is not returned from the first frame"
    /// (the priming step). Symmetrically, the *last* frame's right-half
    /// tail is normally discarded on stream end, because there is no
    /// "next" frame to overlap-add it against. For some applications
    /// (e.g. flushing a finite encoded clip to PCM) the caller may still
    /// want to emit the unoverlapped tail; this method exposes it.
    ///
    /// Returns the stored right-half tail (length `prev_n / 2`) on the
    /// last frame, or `None` if no frame has been pushed. After the call
    /// the state is back to priming.
    pub fn finish(&mut self) -> Option<Vec<f32>> {
        self.prev_right_half.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::vorbis_window;

    // ---- error paths ----

    #[test]
    fn push_rejects_non_power_of_two() {
        let mut state = OverlapAdd::new();
        let bad = vec![0.0f32; 100];
        assert_eq!(
            state.push_frame(&bad),
            Err(OverlapError::NotPowerOfTwo { n: 100 })
        );
    }

    #[test]
    fn push_rejects_zero_length() {
        let mut state = OverlapAdd::new();
        let bad: Vec<f32> = Vec::new();
        assert_eq!(
            state.push_frame(&bad),
            Err(OverlapError::NotPowerOfTwo { n: 0 })
        );
    }

    #[test]
    fn push_rejects_frame_smaller_than_four() {
        let mut state = OverlapAdd::new();
        let too_small = vec![0.5f32, 0.5];
        assert_eq!(
            state.push_frame(&too_small),
            Err(OverlapError::FrameTooSmall { n: 2 })
        );
    }

    #[test]
    fn overlap_error_to_window_error_conversion() {
        assert_eq!(
            WindowError::from(OverlapError::NotPowerOfTwo { n: 7 }),
            WindowError::NotPowerOfTwo { n: 7 }
        );
        assert_eq!(
            WindowError::from(OverlapError::FrameTooSmall { n: 2 }),
            WindowError::NotPowerOfTwo { n: 2 }
        );
    }

    // ---- priming ----

    #[test]
    fn first_frame_primes_and_returns_none() {
        let mut state = OverlapAdd::new();
        assert!(state.is_priming());
        let frame = vec![1.0f32; 64];
        assert_eq!(state.push_frame(&frame), Ok(None));
        assert!(!state.is_priming());
        assert_eq!(state.stored_tail_len(), 32);
    }

    #[test]
    fn reset_returns_to_priming() {
        let mut state = OverlapAdd::new();
        state.push_frame(&vec![0.5f32; 64]).unwrap();
        assert!(!state.is_priming());
        state.reset();
        assert!(state.is_priming());
        assert_eq!(state.stored_tail_len(), 0);
        assert_eq!(state.push_frame(&vec![0.5f32; 64]), Ok(None));
    }

    #[test]
    fn next_output_len_reports_spec_formula() {
        let mut state = OverlapAdd::new();
        assert_eq!(state.next_output_len(64), None); // priming
        state.push_frame(&vec![0.0f32; 256]).unwrap();
        // prev_n=256, cur_n=64 → 256/4 + 64/4 = 64 + 16 = 80
        assert_eq!(state.next_output_len(64), Some(80));
        // prev_n=256, cur_n=256 → 64 + 64 = 128
        assert_eq!(state.next_output_len(256), Some(128));
    }

    // ---- equal-sized overlap geometry ----

    #[test]
    fn equal_size_returns_one_half_block() {
        // §4.3.8: "In the case of same-sized windows, the amount of data
        // to return is one-half block consisting of and only of the
        // overlapped portions." Half of 256 = 128, which equals
        // 256/4 + 256/4.
        let mut state = OverlapAdd::new();
        let frame = vec![1.0f32; 256];
        state.push_frame(&frame).unwrap();
        let out = state.push_frame(&frame).unwrap().unwrap();
        assert_eq!(out.len(), 128);
    }

    #[test]
    fn equal_size_overlap_add_sums_components() {
        // Two frames with constant magnitudes. The return range is fully
        // overlapped, so every output = prev[i] + cur[i].
        let prev = vec![1.0f32; 64];
        let cur = vec![0.25f32; 64];
        let mut state = OverlapAdd::new();
        state.push_frame(&prev).unwrap();
        let out = state.push_frame(&cur).unwrap().unwrap();
        assert_eq!(out.len(), 16 + 16);
        for v in &out {
            assert!(
                (*v - 1.25).abs() < 1e-6,
                "equal-size sum should be 1.25 everywhere, got {v}"
            );
        }
    }

    #[test]
    fn equal_size_overlap_uses_correct_indices() {
        // Build a "ramp" previous frame whose right half is monotonically
        // identifiable, and a zero current frame. The output should equal
        // the ramp's [n/2 .. n) range exactly (current contributes 0).
        let n = 64;
        let prev: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let cur = vec![0.0f32; n];
        let mut state = OverlapAdd::new();
        state.push_frame(&prev).unwrap();
        let out = state.push_frame(&cur).unwrap().unwrap();
        assert_eq!(out.len(), 32);
        // out[k] = prev[n/2 + k] + cur[k - 0] = prev[n/2 + k] + 0
        for (k, &v) in out.iter().enumerate() {
            assert!(
                (v - (32.0 + k as f32)).abs() < 1e-6,
                "ramp test k={k}: expected {} got {v}",
                32 + k
            );
        }
    }

    #[test]
    fn equal_size_overlap_zero_previous_uses_current_left_half() {
        // Zero previous, ramp current. The output should equal current's
        // [0 .. n/2) (the left half).
        let n = 64;
        let prev = vec![0.0f32; n];
        let cur: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut state = OverlapAdd::new();
        state.push_frame(&prev).unwrap();
        let out = state.push_frame(&cur).unwrap().unwrap();
        assert_eq!(out.len(), 32);
        // cur_global_start = prev_n*3/4 - cur_n/4 = 48 - 16 = 32 = prev_n/2.
        // So cur_offset_in_return = 0, cur_idx = k.
        for (k, &v) in out.iter().enumerate() {
            assert!((v - k as f32).abs() < 1e-6, "k={k}: expected {k} got {v}");
        }
    }

    // ---- mixed-size overlap geometry ----

    #[test]
    fn long_then_short_returns_correct_length() {
        // prev=256, cur=64 → return = 64 + 16 = 80
        let mut state = OverlapAdd::new();
        state.push_frame(&vec![0.0f32; 256]).unwrap();
        let out = state.push_frame(&vec![0.0f32; 64]).unwrap().unwrap();
        assert_eq!(out.len(), 80);
    }

    #[test]
    fn short_then_long_returns_correct_length() {
        // prev=64, cur=256 → return = 16 + 64 = 80
        let mut state = OverlapAdd::new();
        state.push_frame(&vec![0.0f32; 64]).unwrap();
        let out = state.push_frame(&vec![0.0f32; 256]).unwrap().unwrap();
        assert_eq!(out.len(), 80);
    }

    #[test]
    fn long_then_short_only_overlaps_short_window_at_tail() {
        // §4.3.8: "When overlapping a short and long window, much of the
        // returned range does not actually overlap." Build a long
        // previous (constant 1.0) and a short current (constant 0.5);
        // the overlap region only covers the last quarter+quarter slice
        // where both frames intersect.
        //
        // prev_n=256, cur_n=64. cur_global_start = 192 - 16 = 176.
        // return range global = [128, 208).
        // - prev covers [0, 256) → fully covers [128, 208).
        // - cur covers [176, 240) → covers [176, 208) of the return.
        //   Within the return [128, 208) (length 80), positions
        //   k=48..80 (i.e. the last 32 samples) carry current.
        let mut state = OverlapAdd::new();
        state.push_frame(&vec![1.0f32; 256]).unwrap();
        let out = state.push_frame(&vec![0.5f32; 64]).unwrap().unwrap();
        assert_eq!(out.len(), 80);
        // First 48: only previous contributes → 1.0
        for (k, &v) in out.iter().enumerate().take(48) {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "k={k}: prev-only should be 1.0, got {v}"
            );
        }
        // Last 32: both contribute → 1.5
        for (k, &v) in out.iter().enumerate().skip(48) {
            assert!((v - 1.5).abs() < 1e-6, "k={k}: sum should be 1.5, got {v}");
        }
    }

    #[test]
    fn short_then_long_current_starts_before_previous_ends() {
        // Mirror of the previous test. prev_n=64 (cur_global_start =
        // 48 - 64 = -16, negative — current already started before
        // previous-center).
        // return range global = [32, 112).
        // - prev covers [0, 64) → covers [32, 64), i.e. first 32 of the 80.
        // - cur covers [-16, 240) → fully covers [32, 112).
        //   cur_offset_in_return = 16 - 64 = -48 → cur_idx = k + 48,
        //   so cur_idx runs 48..128, exactly current's [48 .. 128) i.e.
        //   the back half of current.
        let prev: Vec<f32> = vec![1.0f32; 64];
        let cur: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let mut state = OverlapAdd::new();
        state.push_frame(&prev).unwrap();
        let out = state.push_frame(&cur).unwrap().unwrap();
        assert_eq!(out.len(), 80);
        // First 32: prev=1.0 + cur[k+48]
        for (k, &v) in out.iter().enumerate().take(32) {
            let expected = 1.0 + (k as f32 + 48.0);
            assert!(
                (v - expected).abs() < 1e-6,
                "k={k}: expected {expected} got {v}"
            );
        }
        // Last 48: prev=0 + cur[k+48]
        for (k, &v) in out.iter().enumerate().skip(32) {
            let expected = k as f32 + 48.0;
            assert!(
                (v - expected).abs() < 1e-6,
                "k={k}: expected {expected} got {v}"
            );
        }
    }

    // ---- repeated equal frames lap to unity power (perfect reconstruction) ----

    #[test]
    fn squared_window_overlap_add_reconstructs_constant_signal() {
        // Vorbis windows have the squared-sum-equals-one property: when
        // the same signal is fed through two equal-sized windowed frames
        // shifted by half a block and added back, the steady-state
        // overlap-add output equals the original. Synthesise that test:
        // window a flat unit signal twice with the symmetric window,
        // push two frames, the returned PCM in the overlap region should
        // reconstruct constant 1.0 because the squared-window overlap
        // gives w[i]² + w[i + n/2]² = 1.
        let n = 128;
        let w = vorbis_window(n, 64, false, true, true).unwrap();
        // Frame 1: signal is constant 1.0, windowed = w[i] * 1.0.
        // After MDCT round-trip you'd get w[i] back; modelling that
        // directly, windowed1[i] = w[i] * w[i] (because the synthesis
        // window is applied a second time in §4.3.7's "windowing"
        // step on top of the analysis-window already in the encoded
        // signal — Vorbis uses the same window both sides).
        // For overlap-add to reconstruct, both frames are the same.
        let windowed: Vec<f32> = w.iter().map(|&v| v * v).collect();
        let mut state = OverlapAdd::new();
        state.push_frame(&windowed).unwrap();
        let out = state.push_frame(&windowed).unwrap().unwrap();
        assert_eq!(out.len(), n / 2);
        // Expected: out[k] = w[n/2 + k]² + w[k]² == 1 by the squared-
        // window reconstruction property of §1.3.2.
        for (k, &v) in out.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "k={k}: squared overlap-add reconstruction = {v} (should be 1.0)"
            );
        }
    }

    // ---- multi-frame sequence ----

    #[test]
    fn multi_frame_stream_chains_output_correctly() {
        // Three equal-sized frames with distinguishable constant levels.
        // After priming on frame 0, frame-1 push produces overlap of (0,1),
        // frame-2 push produces overlap of (1,2). State carries forward
        // across calls without leaks.
        let n = 64;
        let frame0 = vec![0.25f32; n];
        let frame1 = vec![0.5f32; n];
        let frame2 = vec![1.0f32; n];

        let mut state = OverlapAdd::new();
        assert_eq!(state.push_frame(&frame0), Ok(None));
        let out01 = state.push_frame(&frame1).unwrap().unwrap();
        let out12 = state.push_frame(&frame2).unwrap().unwrap();
        assert_eq!(out01.len(), 32);
        assert_eq!(out12.len(), 32);
        for &v in &out01 {
            assert!((v - 0.75).abs() < 1e-6, "out01: {v} != 0.75");
        }
        for &v in &out12 {
            assert!((v - 1.5).abs() < 1e-6, "out12: {v} != 1.5");
        }
    }

    // ---- finish() ----

    #[test]
    fn finish_returns_last_right_half() {
        let mut state = OverlapAdd::new();
        state.push_frame(&vec![0.0f32; 64]).unwrap();
        state.push_frame(&vec![1.0f32; 64]).unwrap();
        let tail = state.finish().unwrap();
        assert_eq!(tail.len(), 32);
        for &v in &tail {
            assert!((v - 1.0).abs() < 1e-6);
        }
        assert!(state.is_priming());
    }

    #[test]
    fn finish_in_priming_returns_none() {
        let mut state = OverlapAdd::new();
        assert!(state.finish().is_none());
    }

    // ---- spec range invariants ----

    #[test]
    fn return_range_indices_match_spec_for_long_long() {
        // Spec: "the amount of data to return is window_blocksize(
        // previous)/4 + window_blocksize(current)/4 ... from the center
        // (element windowsize/2) of the previous window to the center
        // (element windowsize/2-1, inclusive) of the current window."
        //
        // Construct a previous frame whose right half is [0, 1, 2, ...]
        // and a current frame whose values are 0. The return is exactly
        // prev_right_half (because cur=0 contributes 0), so the FIRST
        // returned sample == prev[n/2] (the spec's "element windowsize/2
        // of the previous window") and the LAST returned sample ==
        // prev[n-1] (the last sample before n/2 of the next-virtual
        // window, mirroring "element windowsize/2-1" geometry).
        let n = 256;
        let mut prev = vec![0.0f32; n];
        for (i, v) in prev[n / 2..].iter_mut().enumerate() {
            *v = i as f32 + 1.0; // 1..=n/2
        }
        let cur = vec![0.0f32; n];
        let mut state = OverlapAdd::new();
        state.push_frame(&prev).unwrap();
        let out = state.push_frame(&cur).unwrap().unwrap();
        assert_eq!(out.len(), n / 2);
        assert_eq!(
            out[0], 1.0,
            "first return sample should be prev[n/2] (the spec's 'element windowsize/2')"
        );
        assert_eq!(
            out[n / 2 - 1],
            (n / 2) as f32,
            "last return sample should be prev[n-1]"
        );
    }
}

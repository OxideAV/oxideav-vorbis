//! Long/short block-size decision (Vorbis I §1.3.2, encode direction).
//!
//! A Vorbis stream carries two block sizes (`blocksize_0` short,
//! `blocksize_1` long; §4.2.4) and selects per packet which to use via the
//! mode's `blockflag` (§4.3.1). The decoder is told the choice in the
//! bitstream; the **encoder** must *make* it. The spec fixes the bitstream
//! mechanics of block-size switching (the window-overlap geometry of
//! §4.3.8 / §1.3.2, the `previous_window_flag` / `next_window_flag` edge
//! handling) but deliberately leaves the *analysis* that drives the choice
//! to the encoder: the format does not mandate a particular transient
//! detector.
//!
//! ## Why short blocks exist
//!
//! The long block's fine frequency resolution comes at the cost of poor
//! time resolution: a sharp transient (a drum hit, a plosive) smears its
//! quantisation noise across the whole long window, audible as **pre-echo**
//! — noise *before* the attack. Splitting that block into several short
//! blocks confines the noise to the short window containing the attack.
//! So the rule of thumb is: **use a short block when the block contains a
//! transient, a long block otherwise.**
//!
//! ## The heuristic
//!
//! This module's [`detect_transient`] is a clean-room energy-envelope
//! detector: split the time-domain block into equal sub-frames, measure
//! each sub-frame's energy, and flag a transient when the **ratio of the
//! peak sub-frame energy to the mean sub-frame energy** exceeds a
//! threshold — i.e. when energy is concentrated in one sub-frame rather
//! than spread evenly (a flat block has ratio ≈ 1; a single-attack block
//! has a high ratio). [`choose_blocksize`] wraps that into the long/short
//! `blockflag` the mode selection needs. The threshold is the caller's
//! lever (a bit-rate / quality trade): a lower threshold switches to short
//! blocks more eagerly.
//!
//! The detector is pure time-domain analysis of one channel's PCM. It
//! does not read or write any bitstream; it produces the `blockflag` an
//! encoder feeds into [`crate::encoder::write_audio_packet_header`].

/// Errors from the block-size decision heuristic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlocksizeError {
    /// The PCM block was empty. Energy analysis needs at least one sample.
    EmptyBlock,
    /// `subframes` was zero. The block must be split into at least one
    /// sub-frame to measure an energy envelope (one sub-frame trivially
    /// reports no transient).
    ZeroSubframes,
    /// `subframes` exceeded the block length, so a sub-frame would be
    /// empty. The detector requires `subframes <= block.len()`.
    TooManySubframes {
        /// The requested sub-frame count.
        subframes: usize,
        /// The block length it exceeded.
        block_len: usize,
    },
    /// A PCM sample was not finite (`NaN` / `±∞`); an energy sum over it
    /// would poison the ratio. Carries the offending sample index.
    NonFiniteSample(usize),
}

impl core::fmt::Display for BlocksizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BlocksizeError::EmptyBlock => {
                write!(f, "vorbis blocksize: empty PCM block")
            }
            BlocksizeError::ZeroSubframes => {
                write!(f, "vorbis blocksize: subframes=0 (need >= 1)")
            }
            BlocksizeError::TooManySubframes {
                subframes,
                block_len,
            } => write!(
                f,
                "vorbis blocksize: subframes {subframes} exceeds block length {block_len}"
            ),
            BlocksizeError::NonFiniteSample(i) => {
                write!(f, "vorbis blocksize: non-finite PCM sample at index {i}")
            }
        }
    }
}

impl std::error::Error for BlocksizeError {}

/// The energy-envelope figures [`detect_transient`] computes — exposed so
/// a caller can audit or re-threshold the decision without recomputing.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientAnalysis {
    /// Per-sub-frame energy `Σ x²`, in sub-frame order. Length equals the
    /// requested `subframes`.
    pub subframe_energy: Vec<f64>,
    /// The peak sub-frame energy.
    pub peak_energy: f64,
    /// The mean sub-frame energy.
    pub mean_energy: f64,
    /// `peak_energy / mean_energy` — the concentration ratio the transient
    /// gate keys off. `1.0` for perfectly flat energy; large when one
    /// sub-frame dominates. Defined as `1.0` for a fully-silent block
    /// (both peak and mean zero → no transient).
    pub peak_to_mean: f64,
}

/// Analyse a time-domain PCM block's energy envelope for a transient
/// (Vorbis I §1.3.2, encode-side block-size analysis).
///
/// Splits `block` into `subframes` contiguous sub-frames (the final
/// sub-frame absorbs the remainder when `subframes` does not divide the
/// length), measures each sub-frame's energy `Σ x²`, and returns the
/// envelope together with its peak-to-mean concentration ratio. A flat
/// block has every sub-frame at roughly the mean energy, so the ratio is
/// near `1.0`; a block with a single sharp attack has one dominant
/// sub-frame, so the ratio is large.
///
/// # Errors
///
/// * [`BlocksizeError::EmptyBlock`] — `block` is empty.
/// * [`BlocksizeError::ZeroSubframes`] — `subframes == 0`.
/// * [`BlocksizeError::TooManySubframes`] — `subframes > block.len()`.
/// * [`BlocksizeError::NonFiniteSample`] — a sample is `NaN` / `±∞`.
pub fn detect_transient(
    block: &[f32],
    subframes: usize,
) -> Result<TransientAnalysis, BlocksizeError> {
    if block.is_empty() {
        return Err(BlocksizeError::EmptyBlock);
    }
    if subframes == 0 {
        return Err(BlocksizeError::ZeroSubframes);
    }
    if subframes > block.len() {
        return Err(BlocksizeError::TooManySubframes {
            subframes,
            block_len: block.len(),
        });
    }
    for (i, &s) in block.iter().enumerate() {
        if !s.is_finite() {
            return Err(BlocksizeError::NonFiniteSample(i));
        }
    }

    // Equal sub-frames; the final one takes the remainder so every sample
    // is counted exactly once (energy is conserved across the split).
    let base = block.len() / subframes;
    let mut subframe_energy = Vec::with_capacity(subframes);
    let mut start = 0usize;
    for sf in 0..subframes {
        // Last sub-frame runs to the end to absorb a non-dividing tail.
        let end = if sf + 1 == subframes {
            block.len()
        } else {
            start + base
        };
        let mut e = 0.0f64;
        for &x in &block[start..end] {
            e += f64::from(x) * f64::from(x);
        }
        subframe_energy.push(e);
        start = end;
    }

    let peak_energy = subframe_energy
        .iter()
        .copied()
        .fold(0.0f64, |m, e| m.max(e));
    let mean_energy = subframe_energy.iter().sum::<f64>() / subframes as f64;
    let peak_to_mean = if mean_energy == 0.0 {
        // A fully-silent block: no transient.
        1.0
    } else {
        peak_energy / mean_energy
    };

    Ok(TransientAnalysis {
        subframe_energy,
        peak_energy,
        mean_energy,
        peak_to_mean,
    })
}

/// Choose the per-packet block size (Vorbis I §1.3.2 / §4.3.1, encode
/// direction): return the mode `blockflag` a packet header carries —
/// `false` for a **short** block (`blocksize_0`), `true` for a **long**
/// block (`blocksize_1`).
///
/// Wraps [`detect_transient`]: when the block's peak-to-mean energy ratio
/// exceeds `peak_to_mean_threshold` the block holds a transient and the
/// encoder picks the **short** block (`blockflag == false`) to confine the
/// quantisation noise around the attack and avoid pre-echo; otherwise the
/// long block (`blockflag == true`) gives better frequency resolution.
///
/// `peak_to_mean_threshold` is the caller's quality/bit-rate lever: a lower
/// threshold switches to short blocks more eagerly (more pre-echo
/// protection, finer time resolution, coarser frequency resolution). A
/// `threshold <= 1.0` always selects short (every block's ratio is `>=
/// 1.0`); a very large threshold always selects long.
///
/// Returns the chosen `blockflag` together with the [`TransientAnalysis`]
/// that drove it, so the caller can log or re-decide.
///
/// # Errors
///
/// Identical to [`detect_transient`], plus the threshold is taken as-is
/// (a non-finite threshold compares `false` against any finite ratio, so a
/// `NaN` threshold selects long — the safe default — rather than erroring).
pub fn choose_blocksize(
    block: &[f32],
    subframes: usize,
    peak_to_mean_threshold: f64,
) -> Result<(bool, TransientAnalysis), BlocksizeError> {
    let analysis = detect_transient(block, subframes)?;
    // Transient ⇒ short block (blockflag false). `>` (not `>=`) so a flat
    // block at exactly the threshold stays long.
    let is_transient = analysis.peak_to_mean > peak_to_mean_threshold;
    let blockflag = !is_transient;
    Ok((blockflag, analysis))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_block_has_peak_to_mean_near_one() {
        // Constant-energy block: every sub-frame has equal energy → ratio 1.
        let block = vec![0.5f32; 64];
        let a = detect_transient(&block, 8).unwrap();
        assert_eq!(a.subframe_energy.len(), 8);
        assert!((a.peak_to_mean - 1.0).abs() < 1e-9);
    }

    #[test]
    fn single_attack_block_has_high_peak_to_mean() {
        // Silent except a burst in one sub-frame → ratio == subframes.
        let mut block = vec![0.0f32; 64];
        for s in block.iter_mut().take(8) {
            *s = 1.0; // first sub-frame (of 8) carries all the energy
        }
        let a = detect_transient(&block, 8).unwrap();
        // peak = 8 (one sub-frame, 8 samples of 1²); mean = 8/8 = 1.
        assert!((a.peak_to_mean - 8.0).abs() < 1e-9);
    }

    #[test]
    fn energy_is_conserved_across_non_dividing_split() {
        // 65 samples into 8 sub-frames → last sub-frame absorbs the
        // remainder; total sub-frame energy must equal the block energy.
        let block: Vec<f32> = (0..65).map(|i| (i as f32 * 0.01).sin()).collect();
        let a = detect_transient(&block, 8).unwrap();
        let total: f64 = a.subframe_energy.iter().sum();
        let direct: f64 = block.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        assert!((total - direct).abs() < 1e-9);
    }

    #[test]
    fn silent_block_reports_no_transient() {
        let block = vec![0.0f32; 32];
        let a = detect_transient(&block, 4).unwrap();
        assert_eq!(a.peak_energy, 0.0);
        assert_eq!(a.mean_energy, 0.0);
        assert_eq!(a.peak_to_mean, 1.0);
    }

    #[test]
    fn choose_blocksize_picks_short_for_transient_long_for_flat() {
        // Transient block → short (blockflag false).
        let mut transient = vec![0.0f32; 64];
        for s in transient.iter_mut().take(8) {
            *s = 1.0;
        }
        let (flag, a) = choose_blocksize(&transient, 8, 2.0).unwrap();
        assert!(!flag, "transient block must select short");
        assert!(a.peak_to_mean > 2.0);

        // Flat block → long (blockflag true).
        let flat = vec![0.3f32; 64];
        let (flag, a) = choose_blocksize(&flat, 8, 2.0).unwrap();
        assert!(flag, "flat block must select long");
        assert!(a.peak_to_mean <= 2.0);
    }

    #[test]
    fn choose_blocksize_threshold_below_one_always_short() {
        let flat = vec![0.3f32; 64];
        let (flag, _) = choose_blocksize(&flat, 8, 0.5).unwrap();
        assert!(!flag, "ratio >= 1 always exceeds a sub-1 threshold → short");
    }

    #[test]
    fn choose_blocksize_huge_threshold_always_long() {
        let mut transient = vec![0.0f32; 64];
        transient[0] = 10.0;
        let (flag, _) = choose_blocksize(&transient, 8, 1e9).unwrap();
        assert!(flag, "no finite ratio beats a 1e9 threshold → long");
    }

    #[test]
    fn errors_on_empty_zero_subframes_overflow_and_nonfinite() {
        assert_eq!(detect_transient(&[], 4), Err(BlocksizeError::EmptyBlock));
        assert_eq!(
            detect_transient(&[1.0, 2.0], 0),
            Err(BlocksizeError::ZeroSubframes)
        );
        assert_eq!(
            detect_transient(&[1.0, 2.0], 3),
            Err(BlocksizeError::TooManySubframes {
                subframes: 3,
                block_len: 2,
            })
        );
        assert_eq!(
            detect_transient(&[1.0, f32::NAN, 3.0], 1),
            Err(BlocksizeError::NonFiniteSample(1))
        );
    }
}

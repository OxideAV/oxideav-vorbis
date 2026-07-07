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
    /// A block-sequence blocksize was not a power of two in `64..=8192`
    /// (§4.2.2), or the short size exceeded the long size (§4.2.2:
    /// `blocksize_0` must be `<= blocksize_1`).
    BadBlocksizePair {
        /// The short (`blocksize_0`) candidate.
        short_n: usize,
        /// The long (`blocksize_1`) candidate.
        long_n: usize,
    },
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
            BlocksizeError::BadBlocksizePair { short_n, long_n } => write!(
                f,
                "vorbis blocksize: illegal blocksize pair ({short_n}, {long_n}) — each must be \
                 a power of two in 64..=8192 with short <= long"
            ),
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

/// A whole-stream block-size schedule (Vorbis I §4.3.1 / §4.3.8, encode
/// direction): the per-packet `blockflag` sequence plus the granule walk
/// it implies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSequencePlan {
    /// Per-packet `blockflag`s: `false` selects the short block
    /// (`blocksize_0`), `true` the long block (`blocksize_1`). One entry
    /// per audio packet, the §4.3.8 priming packet included.
    pub blockflags: Vec<bool>,
    /// Cumulative granule position after each packet — the absolute
    /// end-PCM-sample count the §4.3.8 lapping rule finishes: packet 0
    /// (priming) finishes 0 samples, packet `f` then adds
    /// `(n_{f-1} + n_f) / 4`. The final entry is `>=` the input length
    /// (the §A.2 end-trim lowers it at mux time).
    pub granules: Vec<u64>,
}

/// Plan a whole stream's block-size sequence from its PCM (Vorbis I
/// §1.3.2 / §4.3.1, encode direction) — the schedule that drives the
/// per-packet mode selection of a two-blocksize stream.
///
/// Walks the §4.3.8 granule recurrence forward: packet 0 is the priming
/// frame (finishes no PCM), and each subsequent packet `f` finishes
/// `(n_{f-1} + n_f) / 4` samples, where each `n` is `short_n` or
/// `long_n` per the packet's `blockflag`. At every step the flag is
/// decided by [`choose_blocksize`] over the *lookahead* region a
/// candidate **long** frame's quantisation noise would smear across —
/// the `long_n` samples from the current granule position (a long
/// frame scheduled at granule `g` is centred at most `g + long_n / 2`
/// and spans a full `long_n` window, so an attack anywhere inside
/// `[g, g + long_n)` could pick up its pre-echo; the region is clamped
/// to the input end, a region shorter than `subframes` shrinks the
/// sub-frame count, and an empty region — pure zero-padding — is
/// always long). The walk stops once the cumulative granule covers the
/// whole input, so the returned plan is exactly the packet sequence a
/// whole-stream encoder emits.
///
/// `pcm` is one channel's (or a mixdown's) PCM; `subframes` and
/// `peak_to_mean_threshold` are the [`detect_transient`] levers.
/// `short_n == long_n` is legal and degenerates to an all-`false`
/// single-blocksize plan.
///
/// Two independent criteria call a region transient (either fires):
///
/// * **within-window concentration** — [`detect_transient`]'s
///   peak-to-mean ratio exceeds `peak_to_mean_threshold`: a sharp
///   attack inside an otherwise quieter window;
/// * **energy rise against context** — the region's peak sub-frame
///   energy exceeds `energy_rise_threshold ×` the *previous* decision
///   region's mean sub-frame energy: a sustained loudness step (e.g. a
///   noise burst over a tone bed) that is flat *within* the window —
///   invisible to the concentration ratio — but whose onset would
///   still smear pre-echo across a long block reaching back into the
///   quieter context. A silent previous region floors at the smallest
///   positive energy, so a true silence→sound onset always fires.
///
/// # Errors
///
/// * [`BlocksizeError::EmptyBlock`] — `pcm` is empty.
/// * [`BlocksizeError::BadBlocksizePair`] — a blocksize is not a power
///   of two in `64..=8192`, or `short_n > long_n` (§4.2.2).
/// * [`BlocksizeError::ZeroSubframes`] — `subframes == 0`.
pub fn plan_block_sequence(
    pcm: &[f32],
    short_n: usize,
    long_n: usize,
    subframes: usize,
    peak_to_mean_threshold: f64,
    energy_rise_threshold: f64,
) -> Result<BlockSequencePlan, BlocksizeError> {
    if pcm.is_empty() {
        return Err(BlocksizeError::EmptyBlock);
    }
    let legal = |n: usize| n.is_power_of_two() && (64..=8192).contains(&n);
    if !legal(short_n) || !legal(long_n) || short_n > long_n {
        return Err(BlocksizeError::BadBlocksizePair { short_n, long_n });
    }
    if subframes == 0 {
        return Err(BlocksizeError::ZeroSubframes);
    }

    let lookahead = long_n;
    // Mean sub-frame energy of the previous decision region — the
    // context the energy-rise criterion compares against. `None` until
    // the first non-empty region is analysed. (Packet 0 and packet 1
    // decide at the same granule; the second call's context is then
    // that same region's own mean, degenerating the rise criterion to
    // the concentration criterion — harmless.)
    let mut prev_mean: Option<f64> = None;
    let mut decide = |g: usize| -> Result<bool, BlocksizeError> {
        if short_n == long_n {
            // Degenerate single-blocksize stream: the flag is
            // meaningless; report the short/only block.
            return Ok(false);
        }
        let end = (g + lookahead).min(pcm.len());
        if g >= end {
            // The frame covers only zero-padding past the input: no
            // transient there — long block.
            return Ok(true);
        }
        let region = &pcm[g..end];
        let sf = subframes.min(region.len());
        let analysis = detect_transient(region, sf)?;
        let concentrated = analysis.peak_to_mean > peak_to_mean_threshold;
        let rise = match prev_mean {
            Some(pm) => analysis.peak_energy > energy_rise_threshold * pm.max(f64::MIN_POSITIVE),
            None => false,
        };
        prev_mean = Some(analysis.mean_energy);
        Ok(!(concentrated || rise))
    };

    let size = |flag: bool| if flag { long_n } else { short_n };
    let mut blockflags = vec![decide(0)?];
    let mut granules = vec![0u64];
    let mut g = 0usize;
    while g < pcm.len() {
        let flag = decide(g)?;
        let n_prev = size(*blockflags.last().expect("non-empty"));
        // §4.3.8: packet f finishes (n_{f-1} + n_f) / 4 samples.
        g += (n_prev + size(flag)) / 4;
        blockflags.push(flag);
        granules.push(g as u64);
    }
    Ok(BlockSequencePlan {
        blockflags,
        granules,
    })
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

    // ---- plan_block_sequence ----

    /// The §4.3.8 granule walk the plan reports must match the flags it
    /// chose: packet 0 at granule 0, then `(n_prev + n_cur) / 4` steps.
    fn assert_walk_consistent(plan: &BlockSequencePlan, short_n: usize, long_n: usize) {
        assert_eq!(plan.blockflags.len(), plan.granules.len());
        assert_eq!(plan.granules[0], 0);
        let size = |flag: bool| if flag { long_n } else { short_n } as u64;
        for f in 1..plan.blockflags.len() {
            let step = (size(plan.blockflags[f - 1]) + size(plan.blockflags[f])) / 4;
            assert_eq!(
                plan.granules[f],
                plan.granules[f - 1] + step,
                "granule walk broken at packet {f}"
            );
        }
    }

    #[test]
    fn steady_signal_plans_all_long() {
        let pcm: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let plan = plan_block_sequence(&pcm, 256, 1024, 8, 4.0, 4.0).unwrap();
        assert_walk_consistent(&plan, 256, 1024);
        assert!(plan.blockflags.iter().all(|&f| f), "steady ⇒ all long");
        // The walk covers the input and stops as soon as it does.
        let last = *plan.granules.last().unwrap() as usize;
        assert!(last >= pcm.len());
        assert!(last - 512 < pcm.len(), "walk overshot by a whole packet");
    }

    #[test]
    fn attack_forces_short_blocks_around_it_only() {
        // Silence, then a burst at sample 2000, then steady tone.
        let mut pcm = vec![0.0f32; 6000];
        for (i, s) in pcm.iter_mut().enumerate().skip(2000).take(64) {
            *s = if i % 2 == 0 { 0.9 } else { -0.9 };
        }
        for (i, s) in pcm.iter_mut().enumerate().skip(3000) {
            *s = (i as f32 * 0.05).sin() * 0.4;
        }
        let plan = plan_block_sequence(&pcm, 256, 1024, 8, 4.0, 4.0).unwrap();
        assert_walk_consistent(&plan, 256, 1024);
        assert!(
            plan.blockflags.iter().any(|&f| !f),
            "the attack must force short blocks"
        );
        assert!(
            plan.blockflags.iter().any(|&f| f),
            "the steady regions must stay long"
        );
        // Every short block's decision region [g, g + long_n) must
        // overlap a genuine attack — the burst at 2000..2064 or the
        // silence→tone onset at 3000 — so shorts cluster at the
        // transients, not elsewhere.
        for f in 0..plan.blockflags.len() {
            if !plan.blockflags[f] {
                let g = if f == 0 {
                    0
                } else {
                    plan.granules[f - 1] as usize
                };
                let covers_burst = g < 2064 && g + 1024 > 2000;
                let covers_tone_onset = g <= 3000 && g + 1024 > 3000;
                assert!(
                    covers_burst || covers_tone_onset,
                    "short block at packet {f} (granule {g}) covers no transient"
                );
            }
        }
    }

    #[test]
    fn equal_blocksizes_degenerate_to_all_short_flags() {
        let pcm = vec![0.25f32; 3000];
        let plan = plan_block_sequence(&pcm, 512, 512, 8, 4.0, 4.0).unwrap();
        assert!(plan.blockflags.iter().all(|&f| !f));
        assert_walk_consistent(&plan, 512, 512);
        // Uniform walk: every packet after priming finishes 256.
        assert_eq!(plan.granules[1], 256);
    }

    #[test]
    fn sustained_burst_fires_the_energy_rise_criterion() {
        // A tone bed, then a 2000-sample loud pseudo-noise burst, then
        // the bed again — the burst is FLAT within any lookahead
        // window (concentration ratio ≈ 1), so only the energy-rise
        // criterion can catch its onset. Concentration is disabled
        // (threshold 1e9) to prove the rise criterion alone fires.
        let mut pcm: Vec<f32> = (0..12_000)
            .map(|i| 0.3 * (i as f32 * 0.031).sin())
            .collect();
        for (i, s) in pcm.iter_mut().enumerate().skip(5000).take(2000) {
            let h = (i as u32).wrapping_mul(2_654_435_761) >> 8;
            *s = ((h & 0xffff) as f32 / 32768.0 - 1.0) * 0.9;
        }
        let plan = plan_block_sequence(&pcm, 256, 1024, 16, 1e9, 4.0).unwrap();
        assert_walk_consistent(&plan, 256, 1024);
        assert!(
            plan.blockflags.iter().any(|&f| !f),
            "the burst onset must fire the rise criterion"
        );
        assert!(
            plan.blockflags.iter().any(|&f| f),
            "the tone beds must stay long"
        );
        // Every short block's decision region overlaps the burst onset
        // (sample 5000): rises fire at the step, not inside the flat
        // burst or on the fall back to the bed.
        for f in 0..plan.blockflags.len() {
            if !plan.blockflags[f] {
                let g = if f == 0 {
                    0
                } else {
                    plan.granules[f - 1] as usize
                };
                assert!(
                    g <= 5000 && g + 1024 > 5000,
                    "short block at packet {f} (granule {g}) does not cover the burst onset"
                );
            }
        }
        // With the rise criterion disabled too, the plan is all-long —
        // the burst is invisible to concentration alone.
        let blind = plan_block_sequence(&pcm, 256, 1024, 16, 1e9, f64::INFINITY).unwrap();
        assert!(blind.blockflags.iter().all(|&f| f));
    }

    #[test]
    fn plan_rejects_bad_shapes() {
        assert_eq!(
            plan_block_sequence(&[], 256, 1024, 8, 4.0, 4.0),
            Err(BlocksizeError::EmptyBlock)
        );
        assert_eq!(
            plan_block_sequence(&[0.0; 100], 100, 1024, 8, 4.0, 4.0),
            Err(BlocksizeError::BadBlocksizePair {
                short_n: 100,
                long_n: 1024
            })
        );
        assert_eq!(
            plan_block_sequence(&[0.0; 100], 1024, 256, 8, 4.0, 4.0),
            Err(BlocksizeError::BadBlocksizePair {
                short_n: 1024,
                long_n: 256
            })
        );
        assert_eq!(
            plan_block_sequence(&[0.0; 100], 256, 16384, 8, 4.0, 4.0),
            Err(BlocksizeError::BadBlocksizePair {
                short_n: 256,
                long_n: 16384
            })
        );
        assert_eq!(
            plan_block_sequence(&[0.0; 100], 256, 1024, 0, 4.0, 4.0),
            Err(BlocksizeError::ZeroSubframes)
        );
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

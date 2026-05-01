//! Trained-book partition classifier for the Vorbis encoder's residue
//! stage.
//!
//! ## What this replaces
//!
//! The encoder's prior `emit_residue_type2` classified each 2-bin residue
//! partition with a single hard-coded L2 threshold:
//!
//! ```ignore
//! const CLASSIFY_L2_THRESHOLD: f32 = 0.25;
//! classes[p] = if partition_l2 > CLASSIFY_L2_THRESHOLD { 1 } else { 0 };
//! ```
//!
//! That's what task #93's README called "the 2-book degenerate placeholder":
//! one global threshold, no learned context, no per-spectrum-band tuning.
//! Partitions in noise-floor bands and partitions in tonal bands competed
//! for the same threshold; the encoder either over-spent bits on quiet
//! partitions (when the threshold was loose) or under-coded loud partitions
//! (when it was tight).
//!
//! ## How the trained books help
//!
//! `vq-train` (in `src/bin/vq-train.rs`) ran LBG on ~15 minutes of mixed
//! LibriVox public-domain speech + Musopen Chopin (CC0) and produced four
//! 256-entry codebooks of 16-D centroids. Each centroid is the cluster
//! mean of a sub-spectrum slice from the training corpus — i.e. it
//! encodes "what a typical 16-bin chunk of post-window-MDCT spectrum
//! looks like in this energy class."
//!
//! At encode-time we pre-compute, from those centroids, a sorted list of
//! per-2-bin-slice L2 magnitudes — one entry per (book × centroid × 2-bin
//! slice) for a total of 4 × 256 × 8 = 8192 samples. Each partition the
//! encoder is about to classify is matched against this distribution: we
//! locate where the partition's L2 falls in the sorted centroid-L2 list
//! and use the percentile to drive the silent/active decision.
//!
//! Concretely:
//!   * Partitions whose L2 is below the centroid distribution's silence
//!     quantile (default 50th percentile of centroid slice L2s) classify
//!     as silent — no VQ bits emitted for that partition.
//!   * Partitions above that quantile classify as active and go through
//!     the existing main + fine cascade.
//!
//! The learned threshold is derived once at encoder construction (it's a
//! function of the trained books only, not of any per-frame data) and
//! then used per-partition. This costs one extra `f32` comparison per
//! partition vs the prior path's hard-coded threshold.
//!
//! ## Bitstream impact
//!
//! Zero. The trained books are **not** declared in the Vorbis setup
//! header — they're encoder-side perceptual oracles. The decoder doesn't
//! see them; the bitstream still uses the existing 4-codebook setup
//! (Y / classbook / main VQ / fine VQ) declared in `encoder.rs`. This
//! keeps the decoder side untouched and preserves bit-clean playback
//! through ffmpeg's libvorbis.

use crate::trained_books::TRAINED_BOOKS;

/// Default percentile of the trained-centroid 2-bin slice L2 distribution
/// at which we cut the silent / active boundary. 0.5 = median: roughly
/// half of training-corpus partitions classify silent, half active.
///
/// Higher values (closer to 1.0) make the encoder more aggressive about
/// silencing partitions — cheaper bitrate, lower SNR. Lower values
/// preserve more partitions as active — bigger files, higher SNR. The
/// 50th-percentile default empirically matches libvorbis q3-q4 sparsity.
const DEFAULT_SILENCE_PERCENTILE: f32 = 0.50;

/// One-time-built classifier wrapping the trained codebooks. Holds the
/// derived silence threshold so per-partition classification is a single
/// f32 compare on the hot path.
#[derive(Clone, Debug)]
pub(crate) struct TrainedPartitionClassifier {
    /// Squared L2 threshold: a partition with `partition_l2_squared <
    /// threshold_l2_squared` classifies silent. Pre-squared so the
    /// per-partition path doesn't need a sqrt.
    threshold_l2_squared: f32,
    /// The original (unsquared) threshold — kept for diagnostic logging
    /// only.
    #[cfg_attr(not(test), allow(dead_code))]
    threshold_l2: f32,
}

impl TrainedPartitionClassifier {
    /// Build a classifier from the in-tree trained books.
    pub(crate) fn from_trained_books() -> Self {
        Self::from_percentile(DEFAULT_SILENCE_PERCENTILE)
    }

    /// Build a classifier matching the legacy hard-coded threshold
    /// (`CLASSIFY_L2_THRESHOLD = 0.25`) used before round-2 wired the
    /// trained books in. Used by the bitrate-comparison fixture so we
    /// have a like-for-like apples-to-apples baseline.
    #[cfg(test)]
    pub(crate) fn from_legacy_threshold() -> Self {
        Self {
            threshold_l2: 0.25,
            threshold_l2_squared: 0.25 * 0.25,
        }
    }

    /// Build a classifier with a custom silence percentile in `(0.0, 1.0)`.
    /// Out-of-range values are clamped.
    pub(crate) fn from_percentile(p: f32) -> Self {
        let p = p.clamp(0.001, 0.999);
        let mut slice_l2s = Vec::with_capacity(8192);
        for book in TRAINED_BOOKS.iter() {
            // Each book is dim × entries floats, row-major (entry-major).
            // Slice each entry into `dim/2` 2-bin slices and record the
            // L2 magnitude.
            let dim = book.dim;
            let entries = book.entries;
            if dim < 2 {
                continue;
            }
            for e in 0..entries {
                let row = &book.data[e * dim..(e + 1) * dim];
                let n_slices = dim / 2;
                for s in 0..n_slices {
                    let v0 = row[s * 2];
                    let v1 = row[s * 2 + 1];
                    let l2sq = v0 * v0 + v1 * v1;
                    slice_l2s.push(l2sq);
                }
            }
        }
        if slice_l2s.is_empty() {
            // Fallback: fall back on the prior hard-coded threshold if the
            // trained books happen to be empty (shouldn't happen with our
            // generated `trained_books.rs`, but keeps the code total).
            return Self {
                threshold_l2_squared: 0.25 * 0.25,
                threshold_l2: 0.25,
            };
        }
        // Sort ascending so the percentile lookup is O(1).
        slice_l2s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((slice_l2s.len() as f32) * p) as usize;
        let idx = idx.min(slice_l2s.len() - 1);
        let threshold_l2_squared = slice_l2s[idx];
        let threshold_l2 = threshold_l2_squared.sqrt();
        Self {
            threshold_l2_squared,
            threshold_l2,
        }
    }

    /// Classify a partition by its already-summed squared L2.
    /// Returns `0` for silent (below the trained threshold) or `1` for
    /// active (above the threshold). The two-class output layout matches
    /// the encoder's existing 2-classification residue setup so this is a
    /// drop-in replacement for the prior hard-coded threshold.
    #[inline]
    pub(crate) fn classify(&self, partition_l2_squared: f32) -> u32 {
        if partition_l2_squared > self.threshold_l2_squared {
            1
        } else {
            0
        }
    }

    /// Diagnostic: the unsquared threshold the classifier resolves to.
    #[cfg(test)]
    pub(crate) fn threshold_l2(&self) -> f32 {
        self.threshold_l2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The classifier should resolve to a finite, positive threshold from
    /// the in-tree trained books. (Smoke test — `trained_books.rs` ships
    /// with the crate so this should always succeed.)
    #[test]
    fn default_classifier_has_finite_threshold() {
        let c = TrainedPartitionClassifier::from_trained_books();
        let t = c.threshold_l2();
        assert!(t.is_finite() && t > 0.0, "threshold not finite: {t}");
    }

    /// A partition with energy well below the threshold classifies silent;
    /// well above classifies active.
    #[test]
    fn classifier_separates_silent_from_active() {
        let c = TrainedPartitionClassifier::from_trained_books();
        let t = c.threshold_l2();
        let lo = (t * 0.1).powi(2);
        let hi = (t * 10.0).powi(2);
        assert_eq!(c.classify(lo), 0, "lo={lo} should classify silent");
        assert_eq!(c.classify(hi), 1, "hi={hi} should classify active");
    }

    /// Higher percentile → higher threshold → more silent partitions.
    #[test]
    fn higher_percentile_raises_threshold() {
        let lo = TrainedPartitionClassifier::from_percentile(0.25);
        let hi = TrainedPartitionClassifier::from_percentile(0.75);
        assert!(
            hi.threshold_l2() >= lo.threshold_l2(),
            "p75 threshold ({}) should be >= p25 ({})",
            hi.threshold_l2(),
            lo.threshold_l2()
        );
    }
}

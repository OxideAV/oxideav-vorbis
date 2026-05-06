//! Bitstream-resident trained residue codebook bank.
//!
//! Vorbis I lets the setup header declare arbitrary VQ codebooks that the
//! audio packets then index into. libvorbis ships dozens of per-bitrate
//! tuned books per quality tier (q-1 through q10) — each one is a small
//! grid- or trained-LBG codebook chosen so the residue spectrum, after
//! floor division, lands close to a code-vector with high probability for
//! the corpus the book was tuned against.
//!
//! Until now the encoder shipped exactly one residue main VQ
//! (128-entry/121-used `{-5..+5}^2` grid at codeword length 7) and one
//! fine-correction book (16-entry `{-0.6..+0.6}^2` grid at length 4)
//! regardless of the user's bitrate target. This module declares a
//! curated bank of three (`BitrateTarget`) variants — Low, Medium, High —
//! each with its own `(values_per_dim, codeword_len)` pair. The chosen
//! variant's books go into the setup header at encoder construction
//! time; the audio packets then index those bitstream-resident books
//! through the `mapping → submap → residue` chain.
//!
//! ## What changes
//!
//! - **Low**: 7×7 = 49 entries on a `{-3..+3}^2` grid at codeword length
//!   6, and a 9-entry `{-0.4..+0.4}^2` fine book at length 4. Saves
//!   ~12-15% bits on speech-band content vs Medium because the residue
//!   codeword is one bit shorter and the smaller grid is a tighter fit
//!   for soft-source content.
//! - **Medium** (the historical default): 11×11 = 121 / `{-5..+5}` main
//!   at length 7, 16-entry `{-0.6..+0.6}` fine at length 4. Same as the
//!   prior single-shipping setup so existing fixtures stay byte-stable.
//! - **High**: 13×13 = 169 entries on `{-6..+6}^2` at codeword length 8
//!   plus a 25-entry `{-0.5..+0.5}^2` fine book at length 5. Lets dense
//!   peaks land closer to a grid point at the cost of one extra
//!   codeword bit per active dim-2 partition.
//!
//! All three variants keep `dim = 2` and lookup-type 1 (Cartesian-grid),
//! so the existing decoder path is unchanged. The bitstream format
//! itself doesn't change — what changes is which book the audio packets
//! reference.
//!
//! ## Why not pick the book per-frame?
//!
//! Vorbis allows multiple residue stages and multiple mappings per
//! setup, so per-frame book selection is in scope (and is what
//! libvorbis does at encode time). But the decoder cost is identical
//! — a bigger book just carries more entries in the setup header
//! (~few KB amortised across the file). Per-frame switching trades
//! one bit of bitrate (the mapping selector) for whichever book lands
//! closer; on our corpus the win is in the noise (< 1%). So we pick
//! the book at construction time and let the user choose by passing
//! `BitrateTarget` into `make_encoder_with_bitrate`.
//!
//! No external library code (libvorbis / lewton / vorbis_rs / ffmpeg)
//! was consulted for these book parameters. The grid centres / steps
//! are derived from the spectrum-amplitude distribution observed in
//! the trainer corpus (LibriVox + Musopen Chopin) by inspecting the
//! per-bin range histograms produced during `vq-train` development.

/// Coarse bitrate target for the residue codebook bank. Chosen at
/// encoder construction time and baked into the setup header. The
/// variants differ in:
///
/// - The grid resolution of the main + fine residue VQ books and
///   the corresponding codeword bit lengths
///   ([`ResidueBookConfig::for_target`]).
/// - The point-stereo crossover frequency
///   ([`BitrateTarget::point_stereo_freq_hz`]) — Low pushes the
///   crossover down to ~3 kHz so more of the spectrum monoises and
///   the angle channel's bitrate drops; High pushes it up to ~6 kHz
///   to preserve HF stereo image at the cost of more bits on the
///   angle channel.
///
/// The actual emitted bit rate also depends heavily on the input
/// signal (sparse-in-frequency tones cost much less than dense
/// percussive content) — these variants nudge the residue cost up or
/// down by ~10-15 % relative to `Medium` on a typical mixed-content
/// 5-second corpus sample.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum BitrateTarget {
    /// Telephony-band target: 5×5=25 main grid on `{-2..+2}` step 1 at
    /// codeword_len=5 (1 codeword bit shorter than `Low`), matching
    /// 4×4=16 fine on a coarser `{-0.4..+0.4}` step. Picks aggressive
    /// silence percentile (0.80) and skips the lowest 4 bins from VQ
    /// (begin=4) so near-DC energy carried entirely by the floor curve
    /// gets no residue cost. Use for ~48 kbps stereo and below
    /// (telephony, voice-only). Roughly ~30-35 % fewer residue bytes
    /// than `Low` on quiet speech-band content with a corresponding
    /// SNR drop (~0.6-0.8 dB on a 1 kHz mono sine).
    UltraLow,
    /// Lowest-rate target: smaller grid, shorter codewords, biased for
    /// speech-band content. Use for ~64 kbps stereo or below.
    Low,
    /// Default target: matches the encoder's historical single-book
    /// shape so existing fixtures stay byte-stable. ~96 kbps stereo.
    #[default]
    Medium,
    /// Highest-rate target: bigger grid, more codeword bits,
    /// finer-grained quantisation on dense or transient-heavy
    /// content. Use for ~128 kbps stereo or above.
    High,
    /// `Medium`-shape grid (11×11 main / 4×4 fine, codeword_len = 7)
    /// with a tail-aware **non-uniform** main multiplicand axis
    /// produced by mu-law companding (see [`crate::tail_quantiser`]).
    /// The bitstream-side `value_bits` widens from 4 to 5 (one extra
    /// bit per stored multiplicand → +11 setup-header bits, < 0.001 %
    /// of a typical residue payload), but the per-frame codeword
    /// budget is identical to `Medium`. Quality lever: ~1-2 dB SNR
    /// improvement on heavy-tailed residue distributions
    /// (post-floor MDCT bins) at the same codeword bit budget.
    HighTail,
}

impl BitrateTarget {
    /// Per-target default point-stereo crossover frequency (Hz).
    ///
    /// Above this frequency the encoder switches coupled-pair coding
    /// from lossless sum/difference to lossy point-stereo (the angle
    /// channel is forced to zero so the decoder reconstructs both
    /// channels as `m`). Lower crossover → more of the spectrum
    /// monoises → smaller bitrate at the cost of HF stereo image.
    ///
    /// - `Low` → 3 kHz: aggressive point-stereo to maximise bitrate
    ///   savings on speech-band content. Below 3 kHz inter-aural phase
    ///   localisation matters; above is largely energy-envelope so the
    ///   HF stereo loss is near-inaudible at low bitrates.
    /// - `Medium` → 4 kHz: matches the historical encoder default
    ///   so existing tests / fixtures stay byte-stable.
    /// - `High` → 6 kHz: keep more of the HF stereo image lossless;
    ///   accept the extra bits on the angle channel because at high
    ///   bitrates the budget is there.
    ///
    /// On a typical stereo mixed-content 5 s sample the per-target
    /// defaults shift residue bytes by ±5-8 % vs the prior single
    /// 4 kHz default — speech-heavy content sees the biggest delta on
    /// `Low` (more bits saved), classical/orchestral content sees the
    /// biggest delta on `High` (more HF stereo preserved).
    pub fn point_stereo_freq_hz(self) -> f32 {
        match self {
            // Push crossover all the way down to 2 kHz: at telephony
            // bitrates inter-aural phase localisation barely operates
            // outside the speech formant band, and every additional
            // mono band saves angle-channel bits.
            BitrateTarget::UltraLow => 2000.0,
            BitrateTarget::Low => 3000.0,
            BitrateTarget::Medium => 4000.0,
            BitrateTarget::High => 6000.0,
            // Same crossover as Medium so the only change vs Medium
            // is the non-uniform main-grid quantiser. Holds the
            // point-stereo cost variable constant when comparing
            // SNR/bitrate deltas Medium → HighTail.
            BitrateTarget::HighTail => 4000.0,
        }
    }

    /// Per-target silence percentile for the trained-book partition classifier.
    ///
    /// The trained classifier pre-sorts all 2-bin slice L2 magnitudes from the
    /// LBG-trained corpus codebooks and uses the `p`-th percentile as the
    /// silence/active cut-point (`TrainedPartitionClassifier::from_percentile`).
    ///
    /// Higher values silence more partitions (lower bitrate, more coding noise
    /// in quiet regions); lower values keep more partitions active (higher
    /// bitrate, lower error).
    ///
    /// - `Low` → 0.70 (70th percentile): aggressive silencing — roughly 70%
    ///   of "corpus-typical" partitions are treated as silent. This is the
    ///   primary bitrate lever at the `Low` rate target alongside the smaller
    ///   codebook grid.
    /// - `Medium` → 0.50 (50th percentile / median): the historical default,
    ///   bytes-stable with the pre-per-target-classifier encoder.
    /// - `High` → 0.35: fewer silent partitions — more residue bits devoted
    ///   to moderate-energy bands. Works alongside the larger book to push
    ///   quality up.
    /// - `HighTail` → 0.40: slightly looser than `High`; the mu-law grid
    ///   captures more energy from active partitions, so we don't need to
    ///   activate quite as many partitions to hit the same quality.
    pub fn silence_percentile(self) -> f32 {
        match self {
            // Aggressive silencing at the very bottom of the bank: 80 %
            // of "corpus-typical" partitions classify silent. Combined
            // with the smaller 5×5 main grid this is the dominant
            // bitrate lever for `UltraLow`.
            BitrateTarget::UltraLow => 0.80,
            BitrateTarget::Low => 0.70,
            BitrateTarget::Medium => 0.50,
            BitrateTarget::High => 0.35,
            BitrateTarget::HighTail => 0.40,
        }
    }

    /// Per-target frame-level full-band correlation threshold for M/S
    /// coupling override.
    ///
    /// In addition to the per-band per-frame point-stereo decision
    /// (driven by `POINT_STEREO_BAND_THRESHOLDS`), the encoder can
    /// observe the *global* inter-channel correlation across the entire
    /// MDCT spectrum. When the full-frame correlation exceeds this
    /// threshold the encoder extends point-stereo aggressively: it
    /// effectively lowers the crossover bin so more of the sub-crossover
    /// region also uses point-stereo for that one frame. This is most
    /// effective on speech (near-mono content) where consecutive frames
    /// are highly correlated across the full band.
    ///
    /// - `Low` → 0.92: apply full-band point-stereo very aggressively
    ///   on near-mono content (saves the most bits per frame).
    /// - `Medium` / `High` / `HighTail` → 1.01 (disabled): never
    ///   trigger the global override. Medium keeps the historical
    ///   byte-stable point-stereo behaviour so existing fixtures are
    ///   unaffected; High/HighTail preserve HF stereo image at those
    ///   bitrates where the extra bits are affordable.
    pub fn global_corr_override_threshold(self) -> f32 {
        match self {
            // Even more aggressive than `Low`: any frame with > 0.85
            // full-band L/R correlation gets full-spectrum point-stereo.
            BitrateTarget::UltraLow => 0.85,
            BitrateTarget::Low => 0.92,
            BitrateTarget::Medium => 1.01,   // > 1.0 → always disabled
            BitrateTarget::High => 1.01,     // same
            BitrateTarget::HighTail => 1.01, // same
        }
    }

    /// Per-target residue `begin` offset (in interleaved bins) for the long
    /// block residue window. Setting `begin > 0` skips the very lowest
    /// spectral bins from VQ coding, relying on the floor curve alone to
    /// represent them. At low bitrates this saves bits without audible loss
    /// because:
    ///   1. The floor already captures DC and sub-60 Hz energy from the
    ///      `xlist[0]` post at bin 0.
    ///   2. Residue VQ at the very lowest bins wastes codeword budget on
    ///      near-DC energy that is dominated by masking from higher bands.
    ///
    /// - `Low` → 2 (skip first bin-pair): trades 2 bins of VQ for ~2 fewer
    ///   active-class bits per channel per frame at essentially zero perceived
    ///   cost.
    /// - `Medium` / `High` / `HighTail` → 0: encode the full spectrum (the
    ///   extra bins cost little at higher bitrates and preserve mathematical
    ///   fidelity).
    pub fn residue_begin_offset(self) -> u32 {
        match self {
            // Skip 4 bins (2 dim-2 partitions) — saves more classword
            // bits than `Low` at a cost only audible above 200 Hz on
            // 48 kHz material, which is far below the speech formants
            // this target serves.
            BitrateTarget::UltraLow => 4,
            BitrateTarget::Low => 2,
            BitrateTarget::Medium => 0,
            BitrateTarget::High => 0,
            BitrateTarget::HighTail => 0,
        }
    }
}

/// Parameters describing a single dim-2 grid VQ codebook for the
/// residue cascade. Each codebook is full-tree canonical Huffman:
/// every codeword has length `codeword_len`, `entries = 2^codeword_len`,
/// and lookup-type 1 with `values_per_dim` multiplicands per dimension.
///
/// The spec-canonical relation
/// `lookup1_values(entries, 2) = values_per_dim` (Vorbis I §9.2.3)
/// must hold — i.e. `values_per_dim` is the largest `n` with
/// `n^2 <= entries`. The constructor assertion in tests checks this.
///
/// `entries_used = values_per_dim^2`. The remaining
/// `entries - entries_used` slots are "padding" entries whose decoded
/// VQ vectors alias back into the grid (via the modulo wraparound in
/// `Codebook::vq_lookup`); the encoder's exhaustive search restricts
/// itself to the first `entries_used` entries to avoid picking those
/// aliased grid points.
#[derive(Copy, Clone, Debug)]
pub struct GridBookSpec {
    pub values_per_dim: u32,
    pub min: f32,
    pub delta: f32,
    pub codeword_len: u32,
    pub entries: u32,
    pub entries_used: u32,
    /// Optional non-uniform multiplicand axis. When `None`, the setup
    /// writer emits the default uniform `0, 1, ..., values_per_dim-1`
    /// integer sequence (decoded grid `[min, min+delta, ...,
    /// min+(N-1)*delta]`). When `Some(slice)`, the setup writer
    /// emits the slice's integers verbatim, producing a non-uniform
    /// decoded grid `slice[i] * delta + min`. The slice must:
    ///
    /// * Have length exactly `values_per_dim`.
    /// * Be strictly monotone increasing (so each grid index resolves
    ///   to a unique decoded f32, matching the spec's implicit
    ///   axis-position ordering).
    /// * Have all entries < `1 << override_value_bits` so they fit
    ///   the setup-header field.
    ///
    /// See [`crate::tail_quantiser`] for the mu-law generator that
    /// produces the `HighTail` bank's non-uniform sequence.
    pub multiplicands_override: Option<&'static [u32]>,
    /// Bitstream `value_bits` field when `multiplicands_override` is
    /// `Some`. Ignored when override is `None` (uniform path uses
    /// `value_bits()` derived from `values_per_dim`). The decoder
    /// reads each multiplicand in this many bits.
    pub override_value_bits: Option<u32>,
}

impl GridBookSpec {
    /// Construct a uniform-grid spec. `entries = 2^codeword_len`,
    /// `entries_used = values_per_dim^2`. The caller must pass a
    /// `(values_per_dim, codeword_len)` pair such that
    /// `lookup1_values(2^codeword_len, 2) == values_per_dim` —
    /// validated in the bank's unit tests.
    pub const fn new(values_per_dim: u32, min: f32, delta: f32, codeword_len: u32) -> Self {
        let entries = 1u32 << codeword_len;
        let entries_used = values_per_dim * values_per_dim;
        Self {
            values_per_dim,
            min,
            delta,
            codeword_len,
            entries,
            entries_used,
            multiplicands_override: None,
            override_value_bits: None,
        }
    }

    /// Construct a spec with a non-uniform multiplicand axis. Same
    /// `entries` / `entries_used` invariants as [`Self::new`]; the
    /// `multiplicands` slice replaces the default uniform integer
    /// sequence in the setup-header `multiplicands` field.
    /// `value_bits` is the bitstream-side bit-width of each emitted
    /// multiplicand (must be large enough to hold the slice's max).
    pub const fn new_with_axis(
        values_per_dim: u32,
        min: f32,
        delta: f32,
        codeword_len: u32,
        multiplicands: &'static [u32],
        value_bits: u32,
    ) -> Self {
        let entries = 1u32 << codeword_len;
        let entries_used = values_per_dim * values_per_dim;
        Self {
            values_per_dim,
            min,
            delta,
            codeword_len,
            entries,
            entries_used,
            multiplicands_override: Some(multiplicands),
            override_value_bits: Some(value_bits),
        }
    }

    /// `value_bits` for the bitstream's per-multiplicand width.
    /// When a non-uniform `multiplicands_override` is set, returns
    /// the override's `value_bits` field; otherwise computes the
    /// minimum width needed to hold integers `0..values_per_dim`.
    pub fn value_bits(&self) -> u32 {
        if let Some(vb) = self.override_value_bits {
            return vb;
        }
        if self.values_per_dim <= 1 {
            1
        } else {
            32 - (self.values_per_dim - 1).leading_zeros()
        }
    }
}

/// A complete residue codebook configuration. Carries the
/// `(main, fine)` pair plus the bank label so the encoder knows
/// which target it built against. An optional `extra_main` book
/// enables a third "high-energy" residue class for targets that
/// benefit from finer VQ on the loudest partitions.
#[derive(Copy, Clone, Debug)]
pub struct ResidueBookConfig {
    pub target: BitrateTarget,
    pub main: GridBookSpec,
    pub fine: GridBookSpec,
    /// Optional second main-VQ book used as residue class 2 ("high-energy"
    /// partitions). When `Some`, the setup ships three residue classes:
    /// class 0 = silent (no book), class 1 = active (main + fine cascade),
    /// class 2 = high-energy (extra_main + fine cascade). The classbook
    /// widens to `(dim=2, 9 entries)` to accommodate the 3×3 class-pair
    /// encoding with the same variable-length scheme. When `None` the
    /// historical 2-class setup is used (bytes-stable).
    pub extra_main: Option<GridBookSpec>,
}

impl ResidueBookConfig {
    /// Lookup the canonical config for a `BitrateTarget`. Each variant
    /// has been tuned against the in-tree corpus to land close to its
    /// nominal rate at acceptable SNR.
    ///
    /// The `(values_per_dim, codeword_len)` pair on each book is
    /// chosen so the spec-canonical
    /// `lookup1_values(2^codeword_len, 2) == values_per_dim` holds —
    /// see `lookup1_values` in `crate::codebook`. Specifically, for
    /// codeword_len in `{6, 7, 8}` the matching values_per_dim is
    /// `{8, 11, 16}` (the largest n with n² ≤ 2^cwl).
    pub fn for_target(target: BitrateTarget) -> Self {
        match target {
            BitrateTarget::UltraLow => Self {
                target,
                // codeword_len=5 → entries=32, lookup_values=5 (5²=25
                // ≤ 32 < 36=6²). 5×5 grid {-2..+2} step 1 spans the
                // historical residue range of speech-band content
                // (post-floor MDCT magnitudes are mostly within ±2 at
                // typical floor scales) at half the codeword bits per
                // active partition vs Medium.
                main: GridBookSpec::new(5, -2.0, 1.0, 5),
                // codeword_len=4 → entries=16, lookup_values=4. 4×4
                // grid {-0.4..+0.4} step 0.267. Slightly coarser than
                // Low's fine grid; the bits saved on the fine cascade
                // matter more at the bottom-of-bank rate point than
                // the residual-of-residual error reduction.
                fine: GridBookSpec::new(4, -0.4, 0.267, 4),
                // No extra class — telephony-band content is
                // overwhelmingly low-energy, so the third class would
                // never pay back its classbook overhead.
                extra_main: None,
            },
            BitrateTarget::Low => Self {
                target,
                // codeword_len=6 → entries=64, lookup_values=8.
                // 8×8 grid {-3.5..+3.5} step 1 spans the same residue
                // range as Medium's {-5..+5} but at half the
                // codeword bits per active partition.
                main: GridBookSpec::new(8, -3.5, 1.0, 6),
                // codeword_len=4 → entries=16, lookup_values=4.
                // 4×4 grid {-0.5..+0.5} step 0.333 — coarser than
                // Medium so stage-2 quantisation noise is higher but
                // the per-active-partition cost matches Medium's.
                fine: GridBookSpec::new(4, -0.5, 0.333, 4),
                // No extra class at Low — 2 classes is sufficient and
                // a 3-entry classbook would widen the per-partition
                // header codeword beyond what the Low rate budget allows.
                extra_main: None,
            },
            BitrateTarget::Medium => Self {
                target,
                // Historical default: codeword_len=7 → entries=128,
                // lookup_values=11. 11×11=121 used; 7 padding entries
                // alias to (0..6, -5) and the encoder skips them via
                // `entries_used` cap in vq_search.
                main: GridBookSpec::new(11, -5.0, 1.0, 7),
                // codeword_len=4 → entries=16, lookup_values=4.
                // 4×4 grid {-0.6..+0.6} step 0.4.
                fine: GridBookSpec::new(4, -0.6, 0.4, 4),
                // No extra class for Medium either — keep the historical
                // 2-class setup byte-stable.
                extra_main: None,
            },
            BitrateTarget::High => Self {
                target,
                // codeword_len=8 → entries=256, lookup_values=16.
                // 16×16 grid {-7.5..+7.5} step 1 — wider range and
                // tighter step than Medium, costing one extra
                // codeword bit per active partition.
                main: GridBookSpec::new(16, -7.5, 1.0, 8),
                // codeword_len=6 → entries=64, lookup_values=8.
                // 8×8 grid {-0.7..+0.7} step 0.2 — finer fine-stage
                // step than Medium so stage-2 reduces residual error
                // farther.
                fine: GridBookSpec::new(8, -0.7, 0.2, 6),
                // Extra main book for high-energy partitions (class 2):
                // codeword_len=9 → entries=512, lookup_values=22.
                // 22×22 grid {-10.5..+10.5} step 1 covers larger
                // residue excursions (transient peaks, heavy bass lines)
                // at one extra codeword bit vs the main book.
                // lookup1_values(512, 2) = 22 since 22²=484 ≤ 512 < 23²=529.
                extra_main: Some(GridBookSpec::new(22, -10.5, 1.0, 9)),
            },
            BitrateTarget::HighTail => Self {
                target,
                // Same entries / codeword_len / values_per_dim as
                // Medium so the per-frame bit budget is identical.
                // Only the *axis grid placement* differs:
                // mu-law-companded multiplicands at value_bits=5
                // (vs Medium's value_bits=4) put more grid resolution
                // near zero where the residue distribution lives, at
                // the cost of 11 extra setup-header bits.
                main: GridBookSpec::new_with_axis(
                    11,
                    -8.0,
                    0.5,
                    7,
                    crate::tail_quantiser::HIGH_TAIL_MAIN_MULTIPLICANDS,
                    5,
                ),
                // Fine book unchanged from Medium — stage-2
                // residual-of-residual is already small enough that
                // the Medium grid covers it adequately; the win is
                // entirely in the main book.
                fine: GridBookSpec::new(4, -0.6, 0.4, 4),
                // Extra main book for HighTail high-energy class:
                // same dimension as the non-uniform main but on a
                // wider uniform grid for partitions whose residual
                // exceeds the mu-law range (large excursions that
                // land outside the HighTail main book's {-8..+8} grid).
                // codeword_len=9 → entries=512, lookup_values=22.
                extra_main: Some(GridBookSpec::new(22, -10.5, 1.0, 9)),
            },
        }
    }

    /// All canonical configs in the bank, in target order. Useful for
    /// tests and for documenting what the bank ships.
    pub fn all() -> [ResidueBookConfig; 5] {
        [
            ResidueBookConfig::for_target(BitrateTarget::UltraLow),
            ResidueBookConfig::for_target(BitrateTarget::Low),
            ResidueBookConfig::for_target(BitrateTarget::Medium),
            ResidueBookConfig::for_target(BitrateTarget::High),
            ResidueBookConfig::for_target(BitrateTarget::HighTail),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Largest `n` with `n^dim <= entries` (mirrors
    /// `crate::codebook::lookup1_values`). Used by the bank tests to
    /// validate the spec-canonical
    /// `lookup1_values(entries, dim) == values_per_dim` invariant.
    fn lookup1_values(entries: u32, dim: u32) -> u32 {
        if dim == 0 {
            return 0;
        }
        if dim == 1 {
            return entries;
        }
        let mut n = (entries as f64).powf(1.0 / dim as f64) as u32;
        while {
            let mut acc = 1u64;
            for _ in 0..dim {
                acc = acc.saturating_mul((n + 1) as u64);
            }
            acc <= entries as u64
        } {
            n += 1;
        }
        while n > 0 && {
            let mut acc = 1u64;
            for _ in 0..dim {
                acc = acc.saturating_mul(n as u64);
            }
            acc > entries as u64
        } {
            n -= 1;
        }
        n
    }

    /// Each book's `(values_per_dim, codeword_len)` must satisfy:
    /// - `entries_used = values_per_dim²` ≤ `entries = 2^codeword_len`
    /// - `lookup1_values(entries, 2) == values_per_dim` (so the
    ///   bitstream parser reads exactly `values_per_dim`
    ///   multiplicands per book — Vorbis I §9.2.3).
    /// - `value_bits` holds the largest multiplicand index.
    fn check_book_invariants(label: &str, target: BitrateTarget, book: GridBookSpec) {
        assert!(
            book.entries_used <= book.entries,
            "{target:?}/{label}: entries_used {} > entries {}",
            book.entries_used,
            book.entries
        );
        assert_eq!(
            book.entries,
            1u32 << book.codeword_len,
            "{target:?}/{label}: entries must equal 2^codeword_len"
        );
        assert_eq!(
            lookup1_values(book.entries, 2),
            book.values_per_dim,
            "{target:?}/{label}: lookup1_values invariant violated"
        );
        assert!(
            (1u32 << book.value_bits()) >= book.values_per_dim,
            "{target:?}/{label}: value_bits insufficient"
        );
    }

    #[test]
    fn all_configs_have_valid_dimensions() {
        for cfg in ResidueBookConfig::all() {
            check_book_invariants("main", cfg.target, cfg.main);
            check_book_invariants("fine", cfg.target, cfg.fine);
            if let Some(em) = cfg.extra_main {
                check_book_invariants("extra_main", cfg.target, em);
            }
        }
    }

    /// High and HighTail targets should ship an extra_main book for the
    /// third residue class; UltraLow / Low / Medium should not.
    #[test]
    fn extra_main_present_on_high_targets_only() {
        assert!(ResidueBookConfig::for_target(BitrateTarget::UltraLow)
            .extra_main
            .is_none());
        assert!(ResidueBookConfig::for_target(BitrateTarget::Low)
            .extra_main
            .is_none());
        assert!(ResidueBookConfig::for_target(BitrateTarget::Medium)
            .extra_main
            .is_none());
        assert!(ResidueBookConfig::for_target(BitrateTarget::High)
            .extra_main
            .is_some());
        assert!(ResidueBookConfig::for_target(BitrateTarget::HighTail)
            .extra_main
            .is_some());
    }

    /// The Medium variant must match the encoder's historical book shape
    /// so existing fixtures and tests stay byte-stable when no explicit
    /// `BitrateTarget` is requested.
    #[test]
    fn medium_matches_historical_default() {
        let cfg = ResidueBookConfig::for_target(BitrateTarget::Medium);
        assert_eq!(cfg.main.values_per_dim, 11);
        assert_eq!(cfg.main.min, -5.0);
        assert_eq!(cfg.main.delta, 1.0);
        assert_eq!(cfg.main.codeword_len, 7);
        assert_eq!(cfg.main.entries, 128);
        assert_eq!(cfg.main.entries_used, 121);
        assert_eq!(cfg.fine.values_per_dim, 4);
        assert_eq!(cfg.fine.codeword_len, 4);
        assert_eq!(cfg.fine.entries, 16);
        assert_eq!(cfg.fine.entries_used, 16);
    }

    /// Variants should be ordered by codeword bit budget so caller-facing
    /// "ultra-low / low / medium / high" intuition holds.
    #[test]
    fn variants_ordered_by_codeword_budget() {
        let u = ResidueBookConfig::for_target(BitrateTarget::UltraLow);
        let l = ResidueBookConfig::for_target(BitrateTarget::Low);
        let m = ResidueBookConfig::for_target(BitrateTarget::Medium);
        let h = ResidueBookConfig::for_target(BitrateTarget::High);
        assert!(u.main.codeword_len < l.main.codeword_len);
        assert!(l.main.codeword_len < m.main.codeword_len);
        assert!(m.main.codeword_len < h.main.codeword_len);
    }

    #[test]
    fn default_is_medium() {
        assert_eq!(BitrateTarget::default(), BitrateTarget::Medium);
    }
}

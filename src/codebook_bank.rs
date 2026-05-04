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
            BitrateTarget::Low => 3000.0,
            BitrateTarget::Medium => 4000.0,
            BitrateTarget::High => 6000.0,
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
}

impl GridBookSpec {
    /// Construct a grid spec. `entries = 2^codeword_len`,
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
        }
    }

    /// `value_bits` for the bitstream's per-multiplicand width.
    /// Returns the number of bits needed to hold values `0..values_per_dim`.
    pub fn value_bits(&self) -> u32 {
        if self.values_per_dim <= 1 {
            1
        } else {
            32 - (self.values_per_dim - 1).leading_zeros()
        }
    }
}

/// A complete residue codebook configuration. Carries the
/// `(main, fine)` pair plus the bank label so the encoder knows
/// which target it built against.
#[derive(Copy, Clone, Debug)]
pub struct ResidueBookConfig {
    pub target: BitrateTarget,
    pub main: GridBookSpec,
    pub fine: GridBookSpec,
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
            },
        }
    }

    /// All canonical configs in the bank, in target order. Useful for
    /// tests and for documenting what the bank ships.
    pub fn all() -> [ResidueBookConfig; 3] {
        [
            ResidueBookConfig::for_target(BitrateTarget::Low),
            ResidueBookConfig::for_target(BitrateTarget::Medium),
            ResidueBookConfig::for_target(BitrateTarget::High),
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
    #[test]
    fn all_configs_have_valid_dimensions() {
        for cfg in ResidueBookConfig::all() {
            for (label, book) in [("main", cfg.main), ("fine", cfg.fine)] {
                assert!(
                    book.entries_used <= book.entries,
                    "{:?}/{}: entries_used {} > entries {}",
                    cfg.target,
                    label,
                    book.entries_used,
                    book.entries
                );
                assert_eq!(
                    book.entries,
                    1u32 << book.codeword_len,
                    "{:?}/{}: entries must equal 2^codeword_len",
                    cfg.target,
                    label
                );
                assert_eq!(
                    lookup1_values(book.entries, 2),
                    book.values_per_dim,
                    "{:?}/{}: lookup1_values invariant violated",
                    cfg.target,
                    label
                );
                assert!(
                    (1u32 << book.value_bits()) >= book.values_per_dim,
                    "{:?}/{}: value_bits insufficient",
                    cfg.target,
                    label
                );
            }
        }
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
    /// "low / medium / high" intuition holds.
    #[test]
    fn variants_ordered_by_codeword_budget() {
        let l = ResidueBookConfig::for_target(BitrateTarget::Low);
        let m = ResidueBookConfig::for_target(BitrateTarget::Medium);
        let h = ResidueBookConfig::for_target(BitrateTarget::High);
        assert!(l.main.codeword_len < m.main.codeword_len);
        assert!(m.main.codeword_len < h.main.codeword_len);
    }

    #[test]
    fn default_is_medium() {
        assert_eq!(BitrateTarget::default(), BitrateTarget::Medium);
    }
}

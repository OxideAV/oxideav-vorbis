//! Codebook *content* design: optimal codeword-length assignment from
//! symbol statistics (Vorbis I §3.2.1).
//!
//! Everything a Vorbis codebook communicates about its entropy code is
//! the per-entry codeword-*length* list — the canonical codewords
//! themselves are implied by §3.2.1's "lowest valued unused binary
//! Huffman codeword" rule ([`crate::huffman::HuffmanTree`] realises
//! it). So *designing* a codebook's entropy content reduces to choosing
//! the length list that packs the encoder's actual symbol distribution
//! into the fewest bits, subject to the §3.2.1 legality constraints:
//!
//! * every used entry's length is in `1..=32` (§3.2.1 packs
//!   `length − 1` as a 5-bit field, capping codewords at 32 bits);
//! * the length list must describe a **fully populated** decision tree
//!   — §3.2.1 rejects both underspecified and overspecified trees, so
//!   the Kraft sum over used entries must equal exactly 1 (with the
//!   errata-20150226 single-used-entry book, whose sole entry must
//!   record length 1, as the one special case);
//! * unused entries carry [`UNUSED_ENTRY`] (`0`) — the sparse-codebook
//!   form §3.2.1 admits for entries the encoder never emits.
//!
//! [`design_codeword_lengths`] solves this optimally: given the
//! frequency with which the encoder emits each entry, it returns the
//! length list minimising the total emitted bits `Σ freq[i] ·
//! length[i]` among all §3.2.1-legal length lists honouring a maximum
//! codeword length. The optimiser is the classic *package-merge*
//! (coin-collector) construction for length-limited prefix codes —
//! textbook algorithmics: build, for each depth level from the cap up
//! to the root, the merged list of symbol "coins" and pairwise
//! packages from the level below, then take the `2·n − 2` cheapest
//! items of the top list; each time a symbol's coin is taken its
//! codeword grows one bit. Taking exactly `2·n − 2` items makes the
//! Kraft sum land on exactly 1, i.e. the §3.2.1 fully-populated tree.
//! With the length cap at 32 and fewer than `2^32` used entries the
//! cap never makes the problem infeasible for a real codebook.
//!
//! Two entry points cover the two sparse policies:
//!
//! * [`design_codeword_lengths`] — zero-frequency entries become
//!   [`UNUSED_ENTRY`] (a sparse book). Cheapest on the wire, but the
//!   resulting book *cannot encode* the pruned entries at all.
//! * [`design_codeword_lengths_dense`] — zero-frequency entries are
//!   smoothed to frequency 1 so every entry keeps a codeword. The
//!   book stays able to encode its full entry range (the safe choice
//!   when future packets may emit symbols the training corpus never
//!   did), at a small cost in optimality for the observed corpus.
//!
//! [`stream_cost_bits`] prices a frequency table against a length
//! list — the exact number of codeword bits a stream with those symbol
//! counts pays — so callers can measure the saving a redesigned book
//! buys before committing to it.

use core::fmt;

use crate::codebook::UNUSED_ENTRY;

/// The §3.2.1 maximum codeword length: the codebook header stores
/// `length − 1` in 5 bits, so lengths span `1..=32`.
pub const MAX_CODEWORD_LEN: u8 = 32;

/// Errors raised by the codeword-length designers and the
/// [`stream_cost_bits`] pricing helper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BookDesignError {
    /// Every entry's frequency was zero (or the table was empty): there
    /// is no used symbol to assign a codeword to. §3.2.1 rejects a
    /// fully-unused codebook at tree-build time ("underspecified");
    /// designing one is refused up front.
    NoUsedSymbols,
    /// `max_len` was outside `1..=32` (§3.2.1's 5-bit `length − 1`
    /// field caps codewords at 32 bits).
    InvalidMaxLength {
        /// The rejected cap.
        max_len: u8,
    },
    /// More used symbols than `2^max_len` distinct codewords exist at
    /// the requested cap — no prefix code can host them.
    TooManySymbols {
        /// Count of used (nonzero-frequency) symbols.
        used: usize,
        /// The requested length cap.
        max_len: u8,
    },
    /// [`stream_cost_bits`]: the length list and the frequency table
    /// disagree on the entry count.
    LengthMismatch {
        /// `lengths.len()`.
        lengths: usize,
        /// `freqs.len()`.
        freqs: usize,
    },
    /// [`stream_cost_bits`]: a symbol with a nonzero frequency maps to
    /// an [`UNUSED_ENTRY`] length — the book cannot encode a stream
    /// that emits it, so the cost is undefined.
    UnusedSymbolHasFrequency {
        /// The offending entry index.
        entry: usize,
    },
    /// [`stream_cost_bits`]: a used entry's length was outside the
    /// §3.2.1-legal `1..=32` range.
    InvalidLength {
        /// The offending entry index.
        entry: usize,
        /// The recorded length.
        length: u8,
    },
}

impl fmt::Display for BookDesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BookDesignError::NoUsedSymbols => write!(
                f,
                "vorbis book design: no symbol has a nonzero frequency (§3.2.1 rejects a fully-unused book)"
            ),
            BookDesignError::InvalidMaxLength { max_len } => write!(
                f,
                "vorbis book design: max codeword length {max_len} outside 1..=32 (§3.2.1)"
            ),
            BookDesignError::TooManySymbols { used, max_len } => write!(
                f,
                "vorbis book design: {used} used symbols cannot fit in 2^{max_len} codewords"
            ),
            BookDesignError::LengthMismatch { lengths, freqs } => write!(
                f,
                "vorbis book design: length list has {lengths} entries but frequency table has {freqs}"
            ),
            BookDesignError::UnusedSymbolHasFrequency { entry } => write!(
                f,
                "vorbis book design: entry {entry} has nonzero frequency but no codeword (unused)"
            ),
            BookDesignError::InvalidLength { entry, length } => write!(
                f,
                "vorbis book design: entry {entry} has invalid codeword length {length} (must be 1..=32 per §3.2.1)"
            ),
        }
    }
}

impl std::error::Error for BookDesignError {}

/// Design the bit-cost-optimal §3.2.1 codeword-length list for a
/// symbol-frequency table, **sparse** policy: zero-frequency entries
/// are marked [`UNUSED_ENTRY`] and get no codeword.
///
/// `freqs[i]` is the number of times the encoder expects to emit entry
/// `i` (e.g. tallied from a training corpus). The returned list is the
/// same length as `freqs`, with `0` for unused entries and a length in
/// `1..=max_len` for used ones, and it minimises
/// `Σ freqs[i] · lengths[i]` over all §3.2.1-legal assignments:
///
/// * the Kraft sum over used entries is exactly 1 (fully populated
///   decision tree — [`crate::huffman::HuffmanTree::from_lengths`]
///   accepts it, neither under- nor over-specified);
/// * except for a **single**-used-entry table, which per errata
///   20150226 must (and does) record length 1 for its sole entry.
///
/// Ties between equal-frequency symbols resolve deterministically:
/// the lower entry index never gets the longer codeword.
///
/// # Errors
///
/// * [`BookDesignError::NoUsedSymbols`] — all frequencies zero.
/// * [`BookDesignError::InvalidMaxLength`] — `max_len` outside
///   `1..=32`.
/// * [`BookDesignError::TooManySymbols`] — more used symbols than
///   `2^max_len` codewords.
pub fn design_codeword_lengths(freqs: &[u64], max_len: u8) -> Result<Vec<u8>, BookDesignError> {
    if !(1..=MAX_CODEWORD_LEN).contains(&max_len) {
        return Err(BookDesignError::InvalidMaxLength { max_len });
    }

    // Gather the used symbols (nonzero frequency), remembering their
    // original entry indices for the scatter back.
    let used: Vec<(usize, u64)> = freqs
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, f)| f > 0)
        .collect();

    if used.is_empty() {
        return Err(BookDesignError::NoUsedSymbols);
    }

    let mut out = vec![UNUSED_ENTRY; freqs.len()];

    // Errata 20150226: a single-used-entry codebook records length 1
    // for its sole entry ("decoder implementations shall reject a
    // codebook if it contains only one used entry and the encoded
    // codeword_length of that entry is not 1").
    if used.len() == 1 {
        out[used[0].0] = 1;
        return Ok(out);
    }

    // Feasibility: `used` symbols need `used` distinct codewords of at
    // most `max_len` bits. (Only reachable for tiny caps — a real
    // codebook's 24-bit entry count always fits under 2^32.)
    if max_len < 63 && (used.len() as u64) > (1u64 << max_len) {
        return Err(BookDesignError::TooManySymbols {
            used: used.len(),
            max_len,
        });
    }

    // Sort ascending by frequency. Package-merge takes level prefixes
    // off this order, so the smallest-frequency symbols (the *front*
    // of the sorted list) collect the most length increments. On a
    // frequency tie the **higher** entry index sorts first — putting
    // it inside every prefix its tied partner is inside — so the lower
    // entry index never ends up with the longer codeword.
    let mut sorted = used.clone();
    sorted.sort_by(|&(ia, fa), &(ib, fb)| fa.cmp(&fb).then(ib.cmp(&ia)));
    let sorted_freqs: Vec<u64> = sorted.iter().map(|&(_, f)| f).collect();

    let lengths_sorted = package_merge(&sorted_freqs, max_len);

    for (&(idx, _), &len) in sorted.iter().zip(lengths_sorted.iter()) {
        out[idx] = len;
    }
    Ok(out)
}

/// Design the codeword-length list with the **dense** policy: every
/// entry keeps a codeword, zero-frequency entries smoothed to
/// frequency 1.
///
/// This trades a little corpus-optimality for coverage: the resulting
/// book can still encode symbols the training corpus never emitted
/// (they simply get the longest codewords), so an encoder that trains
/// on one corpus and then codes new material never finds itself
/// holding a symbol the book cannot express. An empty table is
/// rejected with [`BookDesignError::NoUsedSymbols`].
pub fn design_codeword_lengths_dense(
    freqs: &[u64],
    max_len: u8,
) -> Result<Vec<u8>, BookDesignError> {
    if freqs.is_empty() {
        return Err(BookDesignError::NoUsedSymbols);
    }
    let smoothed: Vec<u64> = freqs.iter().map(|&f| f.max(1)).collect();
    design_codeword_lengths(&smoothed, max_len)
}

/// Price a symbol stream against a codeword-length list: the exact
/// total number of codeword bits a stream emitting entry `i`
/// `freqs[i]` times pays, `Σ freqs[i] · lengths[i]`.
///
/// # Errors
///
/// * [`BookDesignError::LengthMismatch`] — table sizes differ.
/// * [`BookDesignError::UnusedSymbolHasFrequency`] — a symbol the
///   stream emits has no codeword.
/// * [`BookDesignError::InvalidLength`] — a used length outside
///   `1..=32`.
pub fn stream_cost_bits(lengths: &[u8], freqs: &[u64]) -> Result<u64, BookDesignError> {
    if lengths.len() != freqs.len() {
        return Err(BookDesignError::LengthMismatch {
            lengths: lengths.len(),
            freqs: freqs.len(),
        });
    }
    let mut total: u64 = 0;
    for (entry, (&len, &freq)) in lengths.iter().zip(freqs.iter()).enumerate() {
        if freq == 0 {
            continue;
        }
        if len == UNUSED_ENTRY {
            return Err(BookDesignError::UnusedSymbolHasFrequency { entry });
        }
        if len > MAX_CODEWORD_LEN {
            return Err(BookDesignError::InvalidLength { entry, length: len });
        }
        total = total.saturating_add(freq.saturating_mul(len as u64));
    }
    Ok(total)
}

/// Package-merge (coin-collector) core: optimal length-limited prefix
/// code lengths for `n >= 2` symbols whose frequencies arrive sorted
/// ascending. Returns one length per symbol, aligned to the input
/// order, each in `1..=max_len`, with Kraft sum exactly 1.
///
/// The construction builds, for each depth `d` from `max_len` down to
/// `1`, the merged list of the symbol coins (every symbol appears at
/// every depth) and the pairwise packages of the depth-`d+1` list,
/// ordered by weight. The optimum takes the `2·n − 2` cheapest items
/// of the depth-1 list; a taken package recursively takes its two
/// constituents from the level below, and every taken *coin* adds one
/// bit to its symbol's codeword length.
///
/// Because the coins at every level are the same ascending-sorted
/// frequency list, the coins taken at any level form a *prefix* of
/// that list — so the walk below only tracks, per level, *how many*
/// coins were taken, and symbol `i`'s length is the number of levels
/// whose prefix covers it. (On a weight tie between a coin and a
/// package the merge prefers the coin, keeping the tie-break
/// deterministic; any tie order yields the same total cost.)
fn package_merge(sorted_freqs: &[u64], max_len: u8) -> Vec<u8> {
    let n = sorted_freqs.len();
    debug_assert!(n >= 2);
    debug_assert!(sorted_freqs.windows(2).all(|w| w[0] <= w[1]));

    // A depth level, in merged (ascending weight) order: the weight of
    // each item plus whether it is a symbol coin (`true`) or a package
    // (`false`). Levels are indexed 0 → depth `max_len` (the deepest)
    // up to `max_len − 1` → depth 1 (the top).
    struct Level {
        weights: Vec<u64>,
        is_coin: Vec<bool>,
    }

    let levels_count = max_len as usize;
    let mut levels: Vec<Level> = Vec::with_capacity(levels_count);

    // Deepest level: coins only.
    levels.push(Level {
        weights: sorted_freqs.to_vec(),
        is_coin: vec![true; n],
    });

    for _ in 1..levels_count {
        let prev = levels.last().expect("at least the deepest level exists");
        // Pairwise packages of the previous (deeper) level.
        let pkg_count = prev.weights.len() / 2;
        let mut pkg_weights = Vec::with_capacity(pkg_count);
        for p in 0..pkg_count {
            pkg_weights.push(prev.weights[2 * p].saturating_add(prev.weights[2 * p + 1]));
        }
        // Merge coins (ascending) with packages (ascending); prefer the
        // coin on a tie.
        let mut weights = Vec::with_capacity(n + pkg_count);
        let mut is_coin = Vec::with_capacity(n + pkg_count);
        let (mut ci, mut pi) = (0usize, 0usize);
        while ci < n || pi < pkg_count {
            let take_coin = if ci >= n {
                false
            } else if pi >= pkg_count {
                true
            } else {
                sorted_freqs[ci] <= pkg_weights[pi]
            };
            if take_coin {
                weights.push(sorted_freqs[ci]);
                is_coin.push(true);
                ci += 1;
            } else {
                weights.push(pkg_weights[pi]);
                is_coin.push(false);
                pi += 1;
            }
        }
        levels.push(Level { weights, is_coin });
    }

    // Walk from the top level down, converting "take the first `m`
    // items of this level" into coin-prefix counts. `coins_taken[d]`
    // is how many of the (ascending-sorted) symbol coins level `d`
    // contributes to the solution.
    let mut coins_taken = vec![0usize; levels_count];
    let mut take = 2 * n - 2; // items to take from the current level
    for level_idx in (0..levels_count).rev() {
        if take == 0 {
            break;
        }
        let level = &levels[level_idx];
        let take_here = take.min(level.is_coin.len());
        debug_assert_eq!(
            take_here, take,
            "package-merge level {level_idx} cannot supply {take} items"
        );
        let coins = level.is_coin[..take_here].iter().filter(|&&c| c).count();
        coins_taken[level_idx] = coins;
        // Each taken package expands to two items of the level below.
        take = (take_here - coins) * 2;
    }
    debug_assert_eq!(
        take, 0,
        "package-merge bottomed out with items left to take"
    );

    // Symbol `i`'s codeword length = number of levels whose taken coin
    // prefix covers it.
    let mut lengths = vec![0u8; n];
    for &coins in &coins_taken {
        for length in lengths.iter_mut().take(coins) {
            *length += 1;
        }
    }
    debug_assert!(lengths.iter().all(|&l| l >= 1 && l <= max_len));
    lengths
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman::HuffmanTree;

    /// Kraft sum over used entries, in units of 2^-32 (so a fully
    /// populated tree sums to exactly `1 << 32`).
    fn kraft_sum_q32(lengths: &[u8]) -> u128 {
        lengths
            .iter()
            .filter(|&&l| l != UNUSED_ENTRY)
            .map(|&l| 1u128 << (32 - l as u32))
            .sum()
    }

    /// Deterministic pseudo-random generator (xorshift64*) so tests
    /// need no external crates.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// Equal frequencies over a power-of-two count yield the balanced
    /// code: eight equal symbols → all length 3.
    #[test]
    fn equal_frequencies_yield_balanced_code() {
        let freqs = [7u64; 8];
        let lengths = design_codeword_lengths(&freqs, 32).expect("designs");
        assert_eq!(lengths, vec![3u8; 8]);
    }

    /// A strongly skewed distribution assigns the shortest codeword to
    /// the most frequent symbol and monotonically non-shorter codewords
    /// to rarer ones.
    #[test]
    fn skewed_distribution_orders_lengths_inversely_to_frequency() {
        let freqs = [1000u64, 500, 250, 125, 60, 30, 15, 15];
        let lengths = design_codeword_lengths(&freqs, 32).expect("designs");
        for w in lengths.windows(2) {
            assert!(
                w[0] <= w[1],
                "lengths must be non-decreasing as frequency falls: {lengths:?}"
            );
        }
        // The classic dyadic distribution recovers the textbook depths.
        assert_eq!(lengths[0], 1);
        assert_eq!(lengths[1], 2);
        assert_eq!(lengths[2], 3);
        // Kraft equality (fully populated §3.2.1 tree).
        assert_eq!(kraft_sum_q32(&lengths), 1u128 << 32);
    }

    /// The design always builds a valid canonical tree: neither
    /// underspecified nor overspecified per §3.2.1, across a sweep of
    /// pseudo-random frequency tables.
    #[test]
    fn designed_lengths_always_build_a_valid_tree() {
        let mut rng = Rng(0x0DDB_1A5E_5BAD_5EED);
        for trial in 0..200 {
            let n = 2 + (rng.next() % 40) as usize;
            let freqs: Vec<u64> = (0..n).map(|_| rng.next() % 1000).collect();
            if freqs.iter().all(|&f| f == 0) {
                continue;
            }
            let lengths = design_codeword_lengths(&freqs, 32).expect("designs");
            let used = lengths.iter().filter(|&&l| l != UNUSED_ENTRY).count();
            let tree = HuffmanTree::from_lengths(&lengths)
                .unwrap_or_else(|e| panic!("trial {trial}: tree must build, got {e:?}"));
            assert_eq!(tree.used_count() as usize, used);
            if used >= 2 {
                assert_eq!(
                    kraft_sum_q32(&lengths),
                    1u128 << 32,
                    "trial {trial}: Kraft sum must be exactly 1: {lengths:?}"
                );
            }
        }
    }

    /// Zero-frequency entries become sparse [`UNUSED_ENTRY`] slots and
    /// the used entries still form a complete tree.
    #[test]
    fn zero_frequency_entries_are_sparse() {
        let freqs = [10u64, 0, 20, 0, 30, 0, 40, 0];
        let lengths = design_codeword_lengths(&freqs, 32).expect("designs");
        for (i, &l) in lengths.iter().enumerate() {
            if freqs[i] == 0 {
                assert_eq!(l, UNUSED_ENTRY, "entry {i} must be unused");
            } else {
                assert!(l >= 1, "entry {i} must be used");
            }
        }
        assert_eq!(kraft_sum_q32(&lengths), 1u128 << 32);
        HuffmanTree::from_lengths(&lengths).expect("sparse book builds");
    }

    /// The dense policy keeps every entry encodable, giving the
    /// zero-frequency entries the longest codewords.
    #[test]
    fn dense_policy_keeps_every_entry_used() {
        let freqs = [100u64, 0, 50, 0];
        let lengths = design_codeword_lengths_dense(&freqs, 32).expect("designs");
        assert!(lengths.iter().all(|&l| l != UNUSED_ENTRY));
        let max = *lengths.iter().max().unwrap();
        assert_eq!(lengths[1], max, "smoothed entry gets a longest codeword");
        assert_eq!(lengths[3], max, "smoothed entry gets a longest codeword");
        HuffmanTree::from_lengths(&lengths).expect("dense book builds");
    }

    /// Errata 20150226: a single used symbol records length 1 (and the
    /// crate's tree builder accepts it as the single-entry book).
    #[test]
    fn single_used_symbol_records_length_one() {
        let freqs = [0u64, 42, 0];
        let lengths = design_codeword_lengths(&freqs, 32).expect("designs");
        assert_eq!(lengths, vec![0u8, 1, 0]);
        let tree = HuffmanTree::from_lengths(&lengths).expect("builds");
        assert!(tree.is_single_entry());
    }

    /// An all-zero frequency table is refused.
    #[test]
    fn all_zero_frequencies_are_rejected() {
        assert_eq!(
            design_codeword_lengths(&[0u64; 4], 32),
            Err(BookDesignError::NoUsedSymbols)
        );
        assert_eq!(
            design_codeword_lengths(&[], 32),
            Err(BookDesignError::NoUsedSymbols)
        );
        assert_eq!(
            design_codeword_lengths_dense(&[], 32),
            Err(BookDesignError::NoUsedSymbols)
        );
    }

    /// Cap validation: 0 and 33 are rejected.
    #[test]
    fn invalid_max_length_is_rejected() {
        assert_eq!(
            design_codeword_lengths(&[1u64, 1], 0),
            Err(BookDesignError::InvalidMaxLength { max_len: 0 })
        );
        assert_eq!(
            design_codeword_lengths(&[1u64, 1], 33),
            Err(BookDesignError::InvalidMaxLength { max_len: 33 })
        );
    }

    /// Infeasible cap: five symbols cannot fit in 2^2 codewords.
    #[test]
    fn too_many_symbols_for_cap_is_rejected() {
        assert_eq!(
            design_codeword_lengths(&[1u64; 5], 2),
            Err(BookDesignError::TooManySymbols {
                used: 5,
                max_len: 2
            })
        );
        // Exactly 2^2 symbols do fit (all at length 2).
        let lengths = design_codeword_lengths(&[1u64; 4], 2).expect("designs");
        assert_eq!(lengths, vec![2u8; 4]);
    }

    /// The length cap binds: a Fibonacci-like frequency table whose
    /// unlimited Huffman code would exceed the cap still designs a
    /// legal capped code with Kraft equality, and the capped cost is
    /// never below the unlimited cost.
    #[test]
    fn length_cap_binds_and_stays_legal() {
        // Fibonacci frequencies force a maximally skewed Huffman tree
        // (depth n−1 uncapped).
        let mut freqs = vec![1u64, 1];
        while freqs.len() < 12 {
            let n = freqs.len();
            freqs.push(freqs[n - 1] + freqs[n - 2]);
        }
        let freqs: Vec<u64> = freqs.into_iter().rev().collect(); // descending
        let unlimited = design_codeword_lengths(&freqs, 32).expect("designs");
        assert!(
            unlimited.iter().any(|&l| l > 6),
            "premise: the unlimited design must exceed the cap we test: {unlimited:?}"
        );
        let capped = design_codeword_lengths(&freqs, 6).expect("designs");
        assert!(capped.iter().all(|&l| (1..=6).contains(&l)), "{capped:?}");
        assert_eq!(kraft_sum_q32(&capped), 1u128 << 32);
        HuffmanTree::from_lengths(&capped).expect("capped book builds");
        let cost_unlimited = stream_cost_bits(&unlimited, &freqs).unwrap();
        let cost_capped = stream_cost_bits(&capped, &freqs).unwrap();
        assert!(
            cost_capped >= cost_unlimited,
            "capping cannot beat the unconstrained optimum"
        );
    }

    /// Brute-force optimality oracle. Enumerate every non-decreasing
    /// length multiset with Kraft sum exactly 1 and lengths within the
    /// cap; the minimum cost assigns shorter lengths to more frequent
    /// symbols. The designer must match that minimum exactly.
    fn brute_force_min_cost(freqs_desc: &[u64], max_len: u8) -> u64 {
        fn rec(
            n_left: usize,
            min_len: u8,
            max_len: u8,
            budget: u64, // remaining Kraft budget in units of 2^-max_len
            lengths: &mut Vec<u8>,
            freqs_desc: &[u64],
            best: &mut u64,
        ) {
            if n_left == 0 {
                if budget == 0 {
                    let cost: u64 = lengths
                        .iter()
                        .zip(freqs_desc.iter())
                        .map(|(&l, &f)| l as u64 * f)
                        .sum();
                    *best = (*best).min(cost);
                }
                return;
            }
            for l in min_len..=max_len {
                let unit = 1u64 << (max_len - l);
                // Everything must still be payable: remaining n_left−1
                // symbols each cost at least 2^-max_len (1 unit).
                if unit > budget || budget - unit < (n_left as u64 - 1) {
                    continue;
                }
                lengths.push(l);
                rec(
                    n_left - 1,
                    l,
                    max_len,
                    budget - unit,
                    lengths,
                    freqs_desc,
                    best,
                );
                lengths.pop();
            }
        }
        let mut best = u64::MAX;
        let mut lengths = Vec::new();
        // Non-decreasing lengths paired with descending frequencies is
        // always an optimal pairing (a swap argument: exchanging two
        // lengths against the frequency order never lowers the cost).
        rec(
            freqs_desc.len(),
            1,
            max_len,
            1u64 << max_len,
            &mut lengths,
            freqs_desc,
            &mut best,
        );
        best
    }

    /// Package-merge matches the brute-force optimum across exhaustive
    /// small cases (with and without a binding cap).
    #[test]
    fn matches_brute_force_optimum_on_small_cases() {
        let mut rng = Rng(0xC0DE_B00C_5EED_0001);
        for trial in 0..60 {
            let n = 2 + (rng.next() % 5) as usize; // 2..=6 symbols
            let mut freqs: Vec<u64> = (0..n).map(|_| 1 + rng.next() % 100).collect();
            freqs.sort_unstable_by(|a, b| b.cmp(a)); // descending
            for &cap in &[3u8, 4, 6] {
                if (n as u64) > (1u64 << cap) {
                    continue;
                }
                let designed = design_codeword_lengths(&freqs, cap)
                    .unwrap_or_else(|e| panic!("trial {trial} cap {cap}: {e:?}"));
                let got = stream_cost_bits(&designed, &freqs).unwrap();
                let want = brute_force_min_cost(&freqs, cap);
                assert_eq!(
                    got, want,
                    "trial {trial} cap {cap}: designed cost {got} != brute-force optimum {want} \
                     (freqs {freqs:?}, lengths {designed:?})"
                );
            }
        }
    }

    /// The designed code never costs more than the flat (balanced)
    /// assignment, whatever the distribution.
    #[test]
    fn never_worse_than_flat_code() {
        let mut rng = Rng(0xFEED_FACE_CAFE_0002);
        for _ in 0..100 {
            let n = 2 + (rng.next() % 30) as usize;
            let freqs: Vec<u64> = (0..n).map(|_| 1 + rng.next() % 10_000).collect();
            let designed = design_codeword_lengths(&freqs, 32).expect("designs");
            let flat_len = (usize::BITS - (n - 1).leading_zeros()) as u8; // ceil(log2 n)
            let total: u64 = freqs.iter().sum();
            let flat_cost = total * flat_len as u64;
            let designed_cost = stream_cost_bits(&designed, &freqs).unwrap();
            assert!(
                designed_cost <= flat_cost,
                "designed {designed_cost} > flat {flat_cost} for freqs {freqs:?}"
            );
        }
    }

    /// Equal-frequency ties resolve toward the lower entry index: the
    /// earlier symbol never carries the longer codeword.
    #[test]
    fn frequency_ties_break_toward_lower_entry_index() {
        // Three symbols of equal frequency: lengths must be {1,2,2} up
        // to assignment — and the lower indices take the shorter ones.
        let freqs = [5u64, 5, 5];
        let lengths = design_codeword_lengths(&freqs, 32).expect("designs");
        assert_eq!(lengths, vec![1u8, 2, 2]);
    }

    /// `stream_cost_bits` error surface: mismatched tables, an emitted
    /// symbol without a codeword, and an illegal used length.
    #[test]
    fn stream_cost_bits_error_surface() {
        assert_eq!(
            stream_cost_bits(&[1u8, 2], &[1u64]),
            Err(BookDesignError::LengthMismatch {
                lengths: 2,
                freqs: 1
            })
        );
        assert_eq!(
            stream_cost_bits(&[1u8, 0], &[1u64, 1]),
            Err(BookDesignError::UnusedSymbolHasFrequency { entry: 1 })
        );
        assert_eq!(
            stream_cost_bits(&[1u8, 33], &[1u64, 1]),
            Err(BookDesignError::InvalidLength {
                entry: 1,
                length: 33
            })
        );
        // Zero-frequency entries may be unused without error.
        assert_eq!(stream_cost_bits(&[1u8, 0], &[3u64, 0]), Ok(3));
    }

    /// Cost accounting is exact: `Σ freq·len`.
    #[test]
    fn stream_cost_bits_is_exact() {
        let lengths = [1u8, 2, 3, 3];
        let freqs = [10u64, 20, 30, 40];
        assert_eq!(
            stream_cost_bits(&lengths, &freqs).unwrap(),
            10 + 40 + 90 + 120
        );
    }

    /// Large books stay well-behaved: 4096 entries with a Zipf-ish
    /// distribution design in one pass, build a valid tree, and beat
    /// the flat code.
    #[test]
    fn large_book_designs_and_beats_flat() {
        let n = 4096usize;
        let freqs: Vec<u64> = (0..n).map(|i| 1 + (100_000 / (i as u64 + 1))).collect();
        let lengths = design_codeword_lengths(&freqs, 32).expect("designs");
        assert_eq!(kraft_sum_q32(&lengths), 1u128 << 32);
        HuffmanTree::from_lengths(&lengths).expect("builds");
        let designed_cost = stream_cost_bits(&lengths, &freqs).unwrap();
        let total: u64 = freqs.iter().sum();
        assert!(designed_cost < total * 12, "must beat the flat 12-bit code");
    }
}

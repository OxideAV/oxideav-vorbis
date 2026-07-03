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

use crate::codebook::{VorbisCodebook, VqLookup, UNUSED_ENTRY};

/// The §3.2.1 maximum codeword length: the codebook header stores
/// `length − 1` in 5 bits, so lengths span `1..=32`.
pub const MAX_CODEWORD_LEN: u8 = 32;

/// Errors raised by the codeword-length designers and the
/// [`stream_cost_bits`] pricing helper.
#[derive(Debug, Clone, PartialEq)]
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
    /// [`design_entropy_codebook`] / [`redesign_codebook`]: the
    /// frequency table's length disagrees with the codebook's declared
    /// `entries` count — every entry needs a frequency slot (zero for
    /// never-emitted entries).
    EntryCountMismatch {
        /// The codebook's `entries`.
        entries: u32,
        /// `freqs.len()`.
        freqs: usize,
    },
    /// [`BookTallies::record`]: the codebook index is outside the
    /// stream's codebook table.
    BookIndexOutOfRange {
        /// The offending codebook index.
        book: usize,
        /// The codebook table length the tally was built from.
        books: usize,
    },
    /// [`BookTallies::record`]: the recorded entry index is outside the
    /// named codebook's entry range.
    EntryOutOfRange {
        /// The codebook index.
        book: usize,
        /// The offending entry.
        entry: u32,
        /// The codebook's `entries`.
        entries: u32,
    },
    /// [`tally_floor1_packet`]: `floor1_y` does not carry the two
    /// endpoints plus one Y value per partition dimension the header
    /// implies (§7.2.3 reads exactly that many).
    Floor1YLengthMismatch {
        /// Expected `floor1_y` length.
        expected: usize,
        /// Actual `floor1_y` length.
        actual: usize,
    },
    /// [`tally_floor1_packet`]: `partition_cvals` does not carry one
    /// selector per §7.2.3 partition.
    Floor1CvalLengthMismatch {
        /// Expected `partition_cvals` length.
        expected: usize,
        /// Actual `partition_cvals` length.
        actual: usize,
    },
    /// [`tally_floor1_packet`]: a partition names a class outside the
    /// header's class list.
    Floor1ClassOutOfRange {
        /// The partition index.
        partition: usize,
        /// The rejected class index.
        class: u8,
        /// `header.classes.len()`.
        class_count: usize,
    },
    /// [`tally_residue_plans`]: a plan's `partition_entries` row count
    /// disagrees with its `classifications` count.
    ResiduePlanShapeMismatch {
        /// `plan.classifications.len()`.
        classifications: usize,
        /// `plan.partition_entries.len()`.
        partition_entries: usize,
    },
    /// [`tally_residue_plans`]: a partition's classification indexes
    /// outside the residue header's per-class book table.
    ResidueClassificationOutOfRange {
        /// The partition index.
        partition: usize,
        /// The rejected classification.
        classification: u32,
        /// `header.books.len()`.
        classifications: usize,
    },
    /// [`tally_residue_plans`]: an entry list is present where the
    /// header's cascade holds no book for that `(classification, pass)`
    /// stage, or absent where it holds one — the plan and the header
    /// disagree on what §8.6.2 step 18/19 puts on the wire.
    ResiduePlanCascadeMismatch {
        /// The partition index.
        partition: usize,
        /// The cascade stage.
        pass: usize,
    },
    /// [`tally_residue_plans`]: packing a classification stride into
    /// its classbook entry failed (carried verbatim from the §8.6.2
    /// grouping primitive).
    ResidueClassPack(crate::encoder::PackResidueClassGroupsError),
    /// [`tally_floor0_packet`]: the packet's `[booknumber]` selects a
    /// position outside the header's `floor0_book_list` (§6.2.2 step
    /// 5: "if `[booknumber]` is greater than the highest number decode
    /// codebook, this packet is undecodable").
    Floor0BooknumberOutOfRange {
        /// The rejected `[booknumber]`.
        booknumber: u32,
        /// `header.book_list.len()`.
        books: usize,
    },
    /// [`train_residue_books_rd`]: the rate-distortion planner rejected
    /// a corpus vector (carried verbatim).
    ResidueEncode(crate::residue_encode::ResidueEncodeError),
    /// [`train_residue_books_rd`]: `max_iterations` was zero — the
    /// trainer must run at least one plan→retrain pass.
    ZeroIterations,
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
            BookDesignError::EntryCountMismatch { entries, freqs } => write!(
                f,
                "vorbis book design: codebook has {entries} entries but frequency table has {freqs}"
            ),
            BookDesignError::BookIndexOutOfRange { book, books } => write!(
                f,
                "vorbis book design: codebook index {book} outside the stream's {books}-book table"
            ),
            BookDesignError::EntryOutOfRange {
                book,
                entry,
                entries,
            } => write!(
                f,
                "vorbis book design: entry {entry} outside codebook {book}'s {entries} entries"
            ),
            BookDesignError::Floor1YLengthMismatch { expected, actual } => write!(
                f,
                "vorbis book design: floor1_y carries {actual} values, header implies {expected} (§7.2.3)"
            ),
            BookDesignError::Floor1CvalLengthMismatch { expected, actual } => write!(
                f,
                "vorbis book design: partition_cvals carries {actual} selectors, header has {expected} partitions (§7.2.3)"
            ),
            BookDesignError::Floor1ClassOutOfRange {
                partition,
                class,
                class_count,
            } => write!(
                f,
                "vorbis book design: partition {partition} names class {class} outside the header's {class_count} classes (§7.2.3)"
            ),
            BookDesignError::ResiduePlanShapeMismatch {
                classifications,
                partition_entries,
            } => write!(
                f,
                "vorbis book design: residue plan carries {partition_entries} partition-entry rows for {classifications} classifications (§8.6.2)"
            ),
            BookDesignError::ResidueClassificationOutOfRange {
                partition,
                classification,
                classifications,
            } => write!(
                f,
                "vorbis book design: partition {partition} classification {classification} outside the header's {classifications} classes (§8.6.2)"
            ),
            BookDesignError::ResiduePlanCascadeMismatch { partition, pass } => write!(
                f,
                "vorbis book design: partition {partition} pass {pass}: plan entries disagree with the header cascade (§8.6.2 steps 18/19)"
            ),
            BookDesignError::ResidueClassPack(source) => write!(
                f,
                "vorbis book design: classification stride does not pack to a classbook entry: {source} (§8.6.2)"
            ),
            BookDesignError::Floor0BooknumberOutOfRange { booknumber, books } => write!(
                f,
                "vorbis book design: booknumber {booknumber} outside the header's {books}-book floor0_book_list (§6.2.2)"
            ),
            BookDesignError::ResidueEncode(source) => write!(
                f,
                "vorbis book design: rate-distortion residue planning failed: {source}"
            ),
            BookDesignError::ZeroIterations => write!(
                f,
                "vorbis book design: max_iterations must be at least 1"
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

/// Design a complete entropy-only (`codebook_lookup_type = 0`)
/// [`VorbisCodebook`] from a symbol-frequency table.
///
/// The returned book carries the given shape (`entries`,
/// `dimensions`), [`VqLookup::None`], and the bit-cost-optimal
/// codeword lengths [`design_codeword_lengths`] /
/// [`design_codeword_lengths_dense`] assign (per the `dense` policy
/// flag). It is write-ready: `crate::encoder::write_codebook` accepts
/// it and `crate::codebook::parse_codebook` reproduces it.
///
/// `freqs` must have exactly `entries` slots (zero for entries the
/// encoder never emits).
pub fn design_entropy_codebook(
    entries: u32,
    dimensions: u16,
    freqs: &[u64],
    max_len: u8,
    dense: bool,
) -> Result<VorbisCodebook, BookDesignError> {
    if freqs.len() != entries as usize {
        return Err(BookDesignError::EntryCountMismatch {
            entries,
            freqs: freqs.len(),
        });
    }
    let codeword_lengths = if dense {
        design_codeword_lengths_dense(freqs, max_len)?
    } else {
        design_codeword_lengths(freqs, max_len)?
    };
    Ok(VorbisCodebook {
        dimensions,
        entries,
        codeword_lengths,
        lookup: VqLookup::None,
    })
}

/// Redesign an existing codebook's entropy content around a new
/// symbol-frequency table, leaving its shape and VQ lookup untouched.
///
/// This is the retraining step: the returned book has the same
/// `entries`, `dimensions`, and (for lookup types 1 / 2) the same
/// multiplicand table — so every entry still decodes to the identical
/// §3.2.1 VQ vector, and a packet that references the same entry
/// indices decodes to **bit-identical** spectra — but its codeword
/// lengths are re-optimised for the measured distribution, so those
/// same packets serialise into fewer bits. With `dense == true` every
/// entry the original book *uses* stays encodable (never-observed
/// entries keep a long codeword); with `dense == false` never-observed
/// entries are pruned to sparse [`UNUSED_ENTRY`] slots.
///
/// Note the sparse caveat for scalar-context books (the floor-1
/// master / sub-books, whose decoded *entry index* is the value): a
/// pruned entry makes that value unrepresentable in future packets.
/// The dense policy is the safe default when the training corpus may
/// not cover the value range.
pub fn redesign_codebook(
    book: &VorbisCodebook,
    freqs: &[u64],
    max_len: u8,
    dense: bool,
) -> Result<VorbisCodebook, BookDesignError> {
    if freqs.len() != book.entries as usize {
        return Err(BookDesignError::EntryCountMismatch {
            entries: book.entries,
            freqs: freqs.len(),
        });
    }
    let codeword_lengths = if dense {
        design_codeword_lengths_dense(freqs, max_len)?
    } else {
        design_codeword_lengths(freqs, max_len)?
    };
    Ok(VorbisCodebook {
        dimensions: book.dimensions,
        entries: book.entries,
        codeword_lengths,
        lookup: book.lookup.clone(),
    })
}

/// Per-codebook symbol-usage accumulator: one frequency slot per entry
/// of every codebook in a stream's setup-header table.
///
/// An encoder tallies every codeword it emits (or plans to emit)
/// against the book it emits it through, then [`BookTallies::retrain`]
/// redesigns exactly the books the corpus exercised — the training
/// loop behind the floor / residue codebook-content designers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookTallies {
    counts: Vec<Vec<u64>>,
}

impl BookTallies {
    /// Build an all-zero tally table shaped after a stream's codebook
    /// list (one row per book, one slot per entry).
    #[must_use]
    pub fn new(codebooks: &[VorbisCodebook]) -> Self {
        Self {
            counts: codebooks
                .iter()
                .map(|b| vec![0u64; b.entries as usize])
                .collect(),
        }
    }

    /// Record one emission of `entry` through codebook `book`.
    pub fn record(&mut self, book: usize, entry: u32) -> Result<(), BookDesignError> {
        let books = self.counts.len();
        let row = self
            .counts
            .get_mut(book)
            .ok_or(BookDesignError::BookIndexOutOfRange { book, books })?;
        let entries = row.len() as u32;
        let slot = row
            .get_mut(entry as usize)
            .ok_or(BookDesignError::EntryOutOfRange {
                book,
                entry,
                entries,
            })?;
        *slot = slot.saturating_add(1);
        Ok(())
    }

    /// Record a batch of `(book, entry)` emissions **atomically**:
    /// every pair is validated against the tally shape first, and on
    /// any failure nothing is recorded. The packet-level tally walks
    /// ([`tally_floor1_packet`], [`tally_residue_plans`]) commit
    /// through this so a rejected packet never leaves a partial tally
    /// behind.
    pub fn record_all(&mut self, emissions: &[(usize, u32)]) -> Result<(), BookDesignError> {
        for &(book, entry) in emissions {
            let books = self.counts.len();
            let row = self
                .counts
                .get(book)
                .ok_or(BookDesignError::BookIndexOutOfRange { book, books })?;
            if entry as usize >= row.len() {
                return Err(BookDesignError::EntryOutOfRange {
                    book,
                    entry,
                    entries: row.len() as u32,
                });
            }
        }
        for &(book, entry) in emissions {
            let slot = &mut self.counts[book][entry as usize];
            *slot = slot.saturating_add(1);
        }
        Ok(())
    }

    /// The accumulated frequency row for codebook `book`, if in range.
    #[must_use]
    pub fn counts(&self, book: usize) -> Option<&[u64]> {
        self.counts.get(book).map(Vec::as_slice)
    }

    /// Total number of symbol emissions recorded against `book`.
    #[must_use]
    pub fn total(&self, book: usize) -> u64 {
        self.counts
            .get(book)
            .map(|row| row.iter().sum())
            .unwrap_or(0)
    }

    /// Redesign every codebook the tally exercised, leaving untouched
    /// books (no recorded emission) exactly as they were.
    ///
    /// The returned table is index-aligned with `codebooks`: retrained
    /// books keep their shape and VQ lookup ([`redesign_codebook`]) so
    /// existing entry-index plans decode bit-identically, while their
    /// codeword lengths are re-optimised for the recorded
    /// distribution; a book with zero recorded emissions is cloned
    /// unchanged (there is no evidence to retrain it on — and §3.2.1
    /// has no zero-used-entry book to express "never used" with).
    pub fn retrain(
        &self,
        codebooks: &[VorbisCodebook],
        max_len: u8,
        dense: bool,
    ) -> Result<Vec<VorbisCodebook>, BookDesignError> {
        if codebooks.len() != self.counts.len() {
            return Err(BookDesignError::BookIndexOutOfRange {
                book: codebooks.len(),
                books: self.counts.len(),
            });
        }
        codebooks
            .iter()
            .zip(self.counts.iter())
            .map(|(book, freqs)| {
                if freqs.iter().all(|&f| f == 0) {
                    Ok(book.clone())
                } else {
                    redesign_codebook(book, freqs, max_len, dense)
                }
            })
            .collect()
    }
}

/// Tally every codeword emission a §7.2.3 floor-1 packet makes into a
/// per-book frequency table — the statistics-collection step of the
/// floor-1 codebook-*content* trainer.
///
/// The walk mirrors `crate::encoder::write_floor1_packet`'s emission
/// order exactly: for each partition, the class's master selector
/// `cval` is recorded against the class masterbook (only when
/// `subclasses > 0` and a masterbook is present — §7.2.3 step 12),
/// then each dimension's packet-domain Y value is recorded against
/// the sub-book its `cval` slice selects (`(cval >> j·cbits) & csub` —
/// §7.2.3 steps 14/15), skipping the §7.2.3 step-18 `None` sub-book
/// slots (which emit no bits and force `Y = 0`). The two endpoint
/// amplitudes and the `[nonzero]` flag are raw fixed-width fields, not
/// codewords, so they are not tallied; an unused packet
/// (`nonzero == false`) tallies nothing.
///
/// Feeding a corpus of planned packets (e.g. from
/// `crate::floor1_encode::plan_floor1_packet`) through this and then
/// calling [`BookTallies::retrain`] yields floor-1 master / sub-books
/// whose codeword lengths are bit-cost-optimal for that corpus while
/// decoding the same packets to bit-identical curves.
pub fn tally_floor1_packet(
    tallies: &mut BookTallies,
    packet: &crate::encoder::Floor1Packet,
    header: &crate::setup::Floor1Header,
) -> Result<(), BookDesignError> {
    // §7.2.3 step 1: an unused floor emits only the raw `[nonzero]`
    // bit — no codewords to tally.
    if !packet.nonzero {
        return Ok(());
    }

    // Shape gates, mirroring the packet writer's fail-closed checks.
    let expected_y = header.x_list.len() + 2;
    if packet.floor1_y.len() != expected_y {
        return Err(BookDesignError::Floor1YLengthMismatch {
            expected: expected_y,
            actual: packet.floor1_y.len(),
        });
    }
    let partitions = header.partition_class_list.len();
    if packet.partition_cvals.len() != partitions {
        return Err(BookDesignError::Floor1CvalLengthMismatch {
            expected: partitions,
            actual: packet.partition_cvals.len(),
        });
    }
    // The per-partition dimensions must tile floor1_y exactly (two
    // implicit endpoint slots + one Y per dimension).
    let mut dims_sum = 0usize;
    for (partition, &class_no) in header.partition_class_list.iter().enumerate() {
        let class = header.classes.get(class_no as usize).ok_or({
            BookDesignError::Floor1ClassOutOfRange {
                partition,
                class: class_no,
                class_count: header.classes.len(),
            }
        })?;
        dims_sum += class.dimensions as usize;
    }
    if dims_sum + 2 != expected_y {
        return Err(BookDesignError::Floor1YLengthMismatch {
            expected: dims_sum + 2,
            actual: expected_y,
        });
    }

    // Emission walk (§7.2.3 steps 5..19, write direction). Emissions
    // are gathered first and committed atomically at the end, so a
    // rejected packet never leaves a partial tally behind.
    let mut emissions: Vec<(usize, u32)> = Vec::new();
    let mut offset = 2usize;
    for (partition, &class_no) in header.partition_class_list.iter().enumerate() {
        let class = &header.classes[class_no as usize];
        let cbits = class.subclasses;
        let csub: u32 = (1u32 << cbits).saturating_sub(1);
        let mut cval = packet.partition_cvals[partition];

        // Step 12: master selector codeword.
        if cbits > 0 {
            if let Some(book) = class.masterbook {
                emissions.push((book as usize, cval));
            }
        }

        // Steps 13..19: per-dimension Y codewords.
        for dim in 0..class.dimensions as usize {
            let sub_idx = (cval & csub) as usize;
            cval >>= cbits;
            let y = packet.floor1_y[offset + dim];
            if let Some(Some(book)) = class.subclass_books.get(sub_idx) {
                emissions.push((*book as usize, y));
            }
            // A `None` sub-book slot emits nothing (§7.2.3 step 18).
        }
        offset += class.dimensions as usize;
    }
    tallies.record_all(&emissions)
}

/// Tally every codeword emission a §8.6.2 residue body makes into a
/// per-book frequency table — the statistics-collection step of the
/// residue codebook-content trainer, the residue analogue of
/// [`tally_floor1_packet`].
///
/// `plans` holds one `ResidueVectorPlan` per §8.6.2 decode vector,
/// exactly as handed to `crate::encoder::write_residue_body`. Two
/// codeword families are tallied, mirroring the writer's emission:
///
/// * **classwords** — each stride of `classwords_per_codeword`
///   (= the classbook's `dimensions`) partition classifications packs
///   into one classbook entry (§8.6.2 steps 6..12; the final partial
///   stride right-padded with the zero digits the decoder discards),
///   recorded against `header.classbook` via the same
///   `pack_residue_classification_groups` primitive the writer uses;
/// * **value codewords** — each `(partition, pass)` whose
///   classification's cascade holds a book records that stage's entry
///   list against it (§8.6.2 step 19); `None` stages emit nothing
///   (step 18).
///
/// A 'do not decode' vector's plan (empty `classifications` +
/// `partition_entries`) tallies nothing, matching the decoder reading
/// nothing for it. Feeding a corpus of plans through this and calling
/// [`BookTallies::retrain`] yields a classbook + value books whose
/// codeword lengths are bit-cost-optimal for the corpus while the
/// same plans decode to bit-identical residue vectors (retraining
/// preserves each book's VQ lookup, so every entry still unpacks to
/// the identical §3.2.1 vector).
pub fn tally_residue_plans(
    tallies: &mut BookTallies,
    plans: &[crate::encoder::ResidueVectorPlan],
    header: &crate::setup::ResidueHeader,
    codebooks: &[VorbisCodebook],
) -> Result<(), BookDesignError> {
    let classbook_idx = header.classbook as usize;
    let classbook = codebooks
        .get(classbook_idx)
        .ok_or(BookDesignError::BookIndexOutOfRange {
            book: classbook_idx,
            books: codebooks.len(),
        })?;
    let classwords = classbook.dimensions as usize;
    let num_classifications = header.classifications as u32;

    // Emissions are gathered first and committed atomically at the
    // end, so a rejected plan set never leaves a partial tally behind.
    let mut emissions: Vec<(usize, u32)> = Vec::new();
    for plan in plans {
        // A 'do not decode' vector reads nothing (§8.6.2 step 15).
        if plan.classifications.is_empty() && plan.partition_entries.is_empty() {
            continue;
        }
        if plan.partition_entries.len() != plan.classifications.len() {
            return Err(BookDesignError::ResiduePlanShapeMismatch {
                classifications: plan.classifications.len(),
                partition_entries: plan.partition_entries.len(),
            });
        }

        // Classwords: identical grouping to the writer (§8.6.2 steps
        // 6..12 inverse, final partial stride right-padded).
        let groups = crate::encoder::pack_residue_classification_groups(
            &plan.classifications,
            num_classifications,
            classwords,
        )
        .map_err(BookDesignError::ResidueClassPack)?;
        for entry in groups {
            emissions.push((classbook_idx, entry));
        }

        // Value codewords: per (partition, pass) against the cascade.
        for (partition, (&class, stages)) in plan
            .classifications
            .iter()
            .zip(plan.partition_entries.iter())
            .enumerate()
        {
            let row = header.books.get(class as usize).ok_or(
                BookDesignError::ResidueClassificationOutOfRange {
                    partition,
                    classification: class,
                    classifications: header.books.len(),
                },
            )?;
            for (pass, (book, supplied)) in row.iter().zip(stages.iter()).enumerate() {
                match (book, supplied) {
                    (Some(book), Some(entries)) => {
                        for &entry in entries {
                            emissions.push((*book as usize, entry));
                        }
                    }
                    // §8.6.2 step 18: 'unused' stage — nothing on wire.
                    (None, None) => {}
                    _ => {
                        return Err(BookDesignError::ResiduePlanCascadeMismatch {
                            partition,
                            pass,
                        });
                    }
                }
            }
        }
    }
    tallies.record_all(&emissions)
}

/// Tally every codeword emission a §6.2.2 floor-0 packet makes into a
/// per-book frequency table — the statistics-collection step of the
/// floor-0 value-codebook-content trainer, the floor-0 analogue of
/// [`tally_floor1_packet`].
///
/// A floor-0 packet's only codewords are the §6.2.2 step-7 VQ entry
/// run: each entry is recorded against the value book the packet's
/// `[booknumber]` selects through `floor0_book_list` (§6.2.2 step 5).
/// The `[amplitude]` and `[booknumber]` fields themselves are raw
/// fixed-width reads (`floor0_amplitude_bits` /
/// `ilog(floor0_number_of_books)` wide), not codewords, so they are
/// not tallied; an [`Floor0Packet::Unused`] packet
/// (`crate::encoder::Floor0Packet::Unused`) emits only the zero
/// amplitude field and tallies nothing.
///
/// Feeding a corpus of planned packets (e.g. from
/// `crate::floor0_envelope::plan_floor0_packet`) through this and
/// calling [`BookTallies::retrain`] yields a floor-0 value book whose
/// codeword lengths are bit-cost-optimal for that corpus while the
/// same packets decode to bit-identical §6.2.3 curves (retraining
/// preserves the book's VQ lookup, so every entry still unpacks to the
/// identical LSP coefficient sub-vector). This closes the
/// statistics-collection half of the floor-0 codebook-content design
/// followup; on error nothing is recorded.
pub fn tally_floor0_packet(
    tallies: &mut BookTallies,
    packet: &crate::encoder::Floor0Packet,
    header: &crate::setup::Floor0Header,
) -> Result<(), BookDesignError> {
    match packet {
        // §6.2.2 step 2: a zero amplitude short-circuits to 'unused'
        // before any codeword — nothing to tally.
        crate::encoder::Floor0Packet::Unused => Ok(()),
        crate::encoder::Floor0Packet::Curve {
            booknumber,
            entries,
            ..
        } => {
            let book_idx = *header.book_list.get(*booknumber as usize).ok_or(
                BookDesignError::Floor0BooknumberOutOfRange {
                    booknumber: *booknumber,
                    books: header.book_list.len(),
                },
            )? as usize;
            let emissions: Vec<(usize, u32)> =
                entries.iter().map(|&entry| (book_idx, entry)).collect();
            tallies.record_all(&emissions)
        }
    }
}

/// The outcome of a [`train_residue_books_rd`] closed-loop training
/// run.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidueRdTrainingOutcome {
    /// The trained codebook table (index-aligned with the input;
    /// books the corpus never exercised are unchanged). Trained books
    /// are **sparse** — see [`train_residue_books_rd`]'s policy note.
    pub codebooks: Vec<VorbisCodebook>,
    /// The final per-corpus-vector plans, chosen by the
    /// rate-distortion planner under [`Self::codebooks`].
    pub plans: Vec<crate::encoder::ResidueVectorPlan>,
    /// Per iteration: the corpus Lagrangian
    /// `Σ error_sq + lambda · value_bits`, measured at the plan step
    /// under that iteration's books. Alternating minimisation makes
    /// this monotonically non-increasing: the plan step minimises it
    /// per partition given the books, and the (sparse) retrain step
    /// minimises the bit term given the plans.
    pub lagrangian_per_iteration: Vec<f64>,
    /// Per iteration: the corpus' total codeword bits (value codewords
    /// plus classwords) under that iteration's books.
    pub total_bits_per_iteration: Vec<u64>,
    /// `true` if the loop stopped because an iteration reproduced the
    /// previous iteration's plans exactly (a fixed point — further
    /// passes cannot change anything), `false` if it ran out of
    /// `max_iterations`.
    pub converged: bool,
}

/// Closed-loop rate-aware residue book training: alternate the §8.6.2
/// rate-distortion planner and the codebook retrainer until the pair
/// reaches a fixed point.
///
/// A single tally→retrain pass (as in [`tally_residue_plans`] +
/// [`BookTallies::retrain`]) re-prices the codewords for plans chosen
/// under the *old* prices. But
/// `crate::residue_encode::plan_vector_residue_rd` charges the exact
/// codeword lengths in its Lagrangian `error_sq + lambda · bit_cost`,
/// so re-planning under the retrained books can shift choices toward
/// the now-cheaper symbols — which changes the statistics — which
/// justifies another retrain. This is classic alternating
/// minimisation, and it descends a shared objective:
///
/// * **plan step** — given books `b`, choose plans minimising
///   `Σ error_sq + lambda · value_bits(p, b)` (per-partition optimal);
/// * **train step** — given plans `p`, choose codeword lengths
///   minimising `value_bits(p, b) (+ classword bits)` (the optimal
///   length-limited code for the observed frequencies).
///
/// Each step can only lower (or hold) the Lagrangian, so
/// [`ResidueRdTrainingOutcome::lagrangian_per_iteration`] is monotone
/// non-increasing, and the loop terminates at a fixed point or after
/// `max_iterations`.
///
/// **Sparse retrain policy.** The train step uses the sparse policy
/// (`dense == false`): the descent argument needs the retrained books
/// to be *exactly* optimal for the observed frequencies, which dense
/// smoothing (ghost `freq = 1` entries) is not. Every entry the
/// current plans use keeps a codeword, so the previous plans stay
/// feasible — the invariant the monotonicity proof rests on. Callers
/// who need every entry to stay encodable for unseen material can
/// re-run [`BookTallies::retrain`] with `dense == true` over the
/// final plans afterwards.
///
/// `residuals` holds one target residual vector per corpus member
/// (each is planned as one §8.6.2 decode vector against `header`'s
/// type / partitioning). The initial `codebooks` seed the first plan
/// pass; `lambda` is the rate-distortion multiplier (`>= 0`, finite).
pub fn train_residue_books_rd(
    residuals: &[Vec<f32>],
    header: &crate::setup::ResidueHeader,
    codebooks: &[VorbisCodebook],
    lambda: f64,
    max_iterations: usize,
) -> Result<ResidueRdTrainingOutcome, BookDesignError> {
    if max_iterations == 0 {
        return Err(BookDesignError::ZeroIterations);
    }
    let classbook_idx = header.classbook as usize;

    let mut books: Vec<VorbisCodebook> = codebooks.to_vec();
    let mut prev_plans: Option<Vec<crate::encoder::ResidueVectorPlan>> = None;
    let mut lagrangians: Vec<f64> = Vec::new();
    let mut totals: Vec<u64> = Vec::new();
    let mut converged = false;

    for _ in 0..max_iterations {
        // Resolve the per-class value-book rows against the current
        // book table (§8.6.1's books[class][pass] indirection).
        let mut rows: Vec<[Option<&VorbisCodebook>; 8]> = Vec::with_capacity(header.books.len());
        for row in &header.books {
            let mut resolved: [Option<&VorbisCodebook>; 8] = Default::default();
            for (pass, slot) in row.iter().enumerate() {
                if let Some(book) = slot {
                    resolved[pass] = Some(books.get(*book as usize).ok_or(
                        BookDesignError::BookIndexOutOfRange {
                            book: *book as usize,
                            books: books.len(),
                        },
                    )?);
                }
            }
            rows.push(resolved);
        }

        // Plan step: rate-distortion planning under the current books.
        let mut plans: Vec<crate::encoder::ResidueVectorPlan> = Vec::with_capacity(residuals.len());
        let mut lagrangian = 0.0f64;
        let mut value_bits = 0u64;
        for residual in residuals {
            let scored = crate::residue_encode::plan_vector_residue_rd(
                residual,
                &rows,
                header.residue_type,
                header.partition_size,
                lambda,
            )
            .map_err(BookDesignError::ResidueEncode)?;
            lagrangian += scored.total_error_sq + lambda * scored.total_value_bits as f64;
            value_bits += scored.total_value_bits;
            plans.push(crate::encoder::ResidueVectorPlan {
                classifications: scored.classifications,
                partition_entries: scored.partition_entries,
            });
        }

        // Measure the classword bits of these plans under the current
        // classbook (the tally routes classwords through it, so the
        // trained classbook prices them too).
        let mut tallies = BookTallies::new(&books);
        tally_residue_plans(&mut tallies, &plans, header, &books)?;
        let classword_bits = tallies
            .counts(classbook_idx)
            .map(|freqs| stream_cost_bits(&books[classbook_idx].codeword_lengths, freqs))
            .transpose()?
            .unwrap_or(0);
        lagrangians.push(lagrangian);
        totals.push(value_bits + classword_bits);

        // Fixed point: identical plans re-tally to identical
        // frequencies, so the retrained books cannot change either.
        if prev_plans.as_ref() == Some(&plans) {
            converged = true;
            prev_plans = Some(plans);
            break;
        }

        // Train step: sparse retrain (exactly optimal for the
        // observed frequencies — see the policy note above).
        books = tallies.retrain(&books, MAX_CODEWORD_LEN, false)?;
        prev_plans = Some(plans);
    }

    Ok(ResidueRdTrainingOutcome {
        codebooks: books,
        plans: prev_plans.expect("max_iterations >= 1 ran at least one plan pass"),
        lagrangian_per_iteration: lagrangians,
        total_bits_per_iteration: totals,
        converged,
    })
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

    /// A designed entropy codebook is write-ready: it serialises via
    /// the §3.2.1 codebook writer, parses back field-for-field, and its
    /// tree builds.
    #[test]
    fn designed_entropy_codebook_write_parse_round_trips() {
        use crate::encoder::write_codebook;
        use oxideav_core::bits::BitReaderLsb;

        let freqs = [400u64, 120, 87, 87, 40, 12, 4, 1];
        let book = design_entropy_codebook(8, 1, &freqs, 32, false).expect("designs");
        assert_eq!(book.entries, 8);
        assert_eq!(book.lookup, VqLookup::None);
        let bytes = write_codebook(&book).expect("writes");
        let mut r = BitReaderLsb::new(&bytes);
        let parsed = crate::codebook::parse_codebook(&mut r).expect("parses");
        assert_eq!(parsed, book);
        HuffmanTree::from_codebook(&parsed).expect("tree builds");
    }

    /// Measured on the wire: coding a skewed symbol stream through the
    /// designed book costs strictly fewer bits than through the flat
    /// (balanced) book of the same entry count.
    #[test]
    fn designed_book_beats_flat_book_on_the_wire() {
        use oxideav_core::bits::BitWriterLsb;

        // Zipf-ish 16-symbol distribution.
        let freqs: Vec<u64> = (0..16u64).map(|i| 600 / (i + 1)).collect();
        let designed = design_entropy_codebook(16, 1, &freqs, 32, false).expect("designs");
        let flat = VorbisCodebook {
            dimensions: 1,
            entries: 16,
            codeword_lengths: vec![4u8; 16],
            lookup: VqLookup::None,
        };
        let designed_tree = HuffmanTree::from_codebook(&designed).unwrap();
        let flat_tree = HuffmanTree::from_codebook(&flat).unwrap();

        // Emit the whole training stream through both books.
        let mut w_designed = BitWriterLsb::new();
        let mut w_flat = BitWriterLsb::new();
        for (entry, &f) in freqs.iter().enumerate() {
            for _ in 0..f {
                designed_tree
                    .encode_entry(entry as u32, &mut w_designed)
                    .unwrap();
                flat_tree.encode_entry(entry as u32, &mut w_flat).unwrap();
            }
        }
        let designed_bytes = w_designed.finish().len();
        let flat_bytes = w_flat.finish().len();
        assert!(
            designed_bytes < flat_bytes,
            "designed book must beat the flat book: {designed_bytes} vs {flat_bytes} bytes"
        );
        // And the a-priori pricing agrees with the emitted size.
        let priced = stream_cost_bits(&designed.codeword_lengths, &freqs).unwrap();
        assert_eq!(designed_bytes, priced.div_ceil(8) as usize);
    }

    /// Retraining a VQ (lattice) book rewrites only the codeword
    /// lengths: every entry still unpacks to the identical §3.2.1
    /// vector, so existing entry-index plans decode bit-identically.
    #[test]
    fn redesign_preserves_vq_lookup_semantics() {
        use crate::vq::unpack_vector;

        // 2-D lattice, 9 entries (lookup1_values = 3), multiplicands
        // {0,1,2} → grid {-1.0, 0.0, 1.0}.
        let original = VorbisCodebook {
            dimensions: 2,
            entries: 9,
            codeword_lengths: vec![4u8, 4, 4, 4, 4, 4, 4, 4, 4],
            lookup: VqLookup::Lattice {
                minimum_value: -1.0,
                delta_value: 1.0,
                value_bits: 2,
                sequence_p: false,
                multiplicands: vec![0, 1, 2],
            },
        };
        // Fix the Kraft slack of the hand-rolled table: 9 entries at
        // length 4 leave capacity; use the designer itself to build a
        // legal starting book instead.
        let flat_freqs = vec![1u64; 9];
        let original = redesign_codebook(&original, &flat_freqs, 32, false).expect("legalises");

        let skewed: Vec<u64> = (0..9u64).map(|i| 1 + 1000 / (1 + i * i)).collect();
        let retrained = redesign_codebook(&original, &skewed, 32, true).expect("retrains");
        assert_eq!(retrained.entries, original.entries);
        assert_eq!(retrained.dimensions, original.dimensions);
        assert_eq!(retrained.lookup, original.lookup);
        assert_ne!(
            retrained.codeword_lengths, original.codeword_lengths,
            "a skewed distribution must actually change the lengths"
        );
        for entry in 0..9u32 {
            assert_eq!(
                unpack_vector(&original, entry).unwrap(),
                unpack_vector(&retrained, entry).unwrap(),
                "entry {entry} must decode to the identical VQ vector"
            );
        }
        // The retrained book is cheaper on its own training stream.
        let before = stream_cost_bits(&original.codeword_lengths, &skewed).unwrap();
        let after = stream_cost_bits(&retrained.codeword_lengths, &skewed).unwrap();
        assert!(after < before, "retrained {after} must beat flat {before}");
    }

    /// Shape validation: the frequency table must match the entry
    /// count exactly.
    #[test]
    fn entry_count_mismatch_is_rejected() {
        assert_eq!(
            design_entropy_codebook(8, 1, &[1u64; 7], 32, false),
            Err(BookDesignError::EntryCountMismatch {
                entries: 8,
                freqs: 7
            })
        );
        let book = design_entropy_codebook(4, 1, &[1u64; 4], 32, false).unwrap();
        assert_eq!(
            redesign_codebook(&book, &[1u64; 5], 32, false),
            Err(BookDesignError::EntryCountMismatch {
                entries: 4,
                freqs: 5
            })
        );
    }

    /// BookTallies: record / out-of-range surface / totals.
    #[test]
    fn book_tallies_record_and_error_surface() {
        let books = vec![
            design_entropy_codebook(4, 1, &[1u64; 4], 32, false).unwrap(),
            design_entropy_codebook(2, 1, &[1u64; 2], 32, false).unwrap(),
        ];
        let mut tallies = BookTallies::new(&books);
        tallies.record(0, 3).unwrap();
        tallies.record(0, 3).unwrap();
        tallies.record(1, 0).unwrap();
        assert_eq!(tallies.counts(0).unwrap(), &[0, 0, 0, 2]);
        assert_eq!(tallies.total(0), 2);
        assert_eq!(tallies.total(1), 1);
        assert_eq!(tallies.total(9), 0);
        assert_eq!(
            tallies.record(2, 0),
            Err(BookDesignError::BookIndexOutOfRange { book: 2, books: 2 })
        );
        assert_eq!(
            tallies.record(1, 2),
            Err(BookDesignError::EntryOutOfRange {
                book: 1,
                entry: 2,
                entries: 2
            })
        );
    }

    /// Retraining through tallies touches exactly the exercised books:
    /// untallied books come back unchanged, tallied ones re-optimised.
    #[test]
    fn tallies_retrain_only_exercised_books() {
        let flat8 = design_entropy_codebook(8, 1, &[1u64; 8], 32, false).unwrap();
        let flat4 = design_entropy_codebook(4, 1, &[1u64; 4], 32, false).unwrap();
        let books = vec![flat8.clone(), flat4.clone()];
        let mut tallies = BookTallies::new(&books);
        // Exercise only book 0, heavily skewed toward entry 0.
        for _ in 0..1000 {
            tallies.record(0, 0).unwrap();
        }
        for e in 1..8u32 {
            tallies.record(0, e).unwrap();
        }
        let retrained = tallies.retrain(&books, 32, true).expect("retrains");
        assert_eq!(retrained.len(), 2);
        assert_eq!(retrained[1], flat4, "untallied book must be unchanged");
        assert_ne!(
            retrained[0].codeword_lengths, flat8.codeword_lengths,
            "tallied book must be re-optimised"
        );
        assert_eq!(
            retrained[0].codeword_lengths[0], 1,
            "dominant symbol gets 1 bit"
        );
        // Dense: every original entry stays encodable.
        assert!(retrained[0]
            .codeword_lengths
            .iter()
            .all(|&l| l != UNUSED_ENTRY));
        // Table-shape mismatch is rejected.
        assert_eq!(
            tallies.retrain(&books[..1], 32, true),
            Err(BookDesignError::BookIndexOutOfRange { book: 1, books: 2 })
        );
    }

    // ----------------------------------------------------------------
    // tally_floor1_packet — the §7.2.3 emission tally.
    // ----------------------------------------------------------------

    /// A two-partition floor-1 header exercising both class shapes:
    /// partition 0 uses class 0 (`subclasses = 0`, slot 0 → book 0),
    /// partition 1 uses class 1 (`subclasses = 1`, masterbook 1,
    /// slots → books 0 / 2, dimensions 2).
    fn tally_test_header() -> crate::setup::Floor1Header {
        use crate::setup::{Floor1Class, Floor1Header};
        Floor1Header {
            partitions: 2,
            partition_class_list: vec![0, 1],
            classes: vec![
                Floor1Class {
                    dimensions: 2,
                    subclasses: 0,
                    masterbook: None,
                    subclass_books: vec![Some(0)],
                },
                Floor1Class {
                    dimensions: 2,
                    subclasses: 1,
                    masterbook: Some(1),
                    subclass_books: vec![Some(0), Some(2)],
                },
            ],
            multiplier: 2,
            rangebits: 7,
            x_list: vec![16, 32, 64, 96],
        }
    }

    fn tally_test_books() -> Vec<VorbisCodebook> {
        vec![
            design_entropy_codebook(128, 1, &[1u64; 128], 32, false).unwrap(),
            design_entropy_codebook(4, 1, &[1u64; 4], 32, false).unwrap(),
            design_entropy_codebook(64, 1, &[1u64; 64], 32, false).unwrap(),
        ]
    }

    /// The tally mirrors the §7.2.3 emission walk: master cval into the
    /// masterbook, each Y into the sub-book its cval slice selects.
    #[test]
    fn floor1_tally_records_master_and_subbook_symbols() {
        use crate::encoder::Floor1Packet;
        let header = tally_test_header();
        let books = tally_test_books();
        let mut tallies = BookTallies::new(&books);
        // cval = 0b10 for partition 1: dim 0 → slot 0 (book 0), dim 1 →
        // slot 1 (book 2).
        let packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![5, 9, 20, 30, 40, 50],
            partition_cvals: vec![0, 2],
        };
        tally_floor1_packet(&mut tallies, &packet, &header).expect("tallies");
        // Partition 0 (class 0, subclasses 0): both dims through book 0.
        // Partition 1 (class 1): master cval 2 through book 1; dim 0
        // (Y = 40) slot 0 → book 0, dim 1 (Y = 50) slot 1 → book 2.
        assert_eq!(tallies.total(0), 3, "book 0: Y 20, 30, 40");
        assert_eq!(tallies.counts(0).unwrap()[20], 1);
        assert_eq!(tallies.counts(0).unwrap()[30], 1);
        assert_eq!(tallies.counts(0).unwrap()[40], 1);
        assert_eq!(tallies.counts(1).unwrap()[2], 1, "master cval 2");
        assert_eq!(tallies.total(1), 1);
        assert_eq!(tallies.counts(2).unwrap()[50], 1, "fine book Y 50");
        assert_eq!(tallies.total(2), 1);
        // Endpoints (5, 9) are raw fields — never tallied.
        assert_eq!(tallies.counts(0).unwrap()[5], 0);
        assert_eq!(tallies.counts(0).unwrap()[9], 0);
    }

    /// An unused packet (`nonzero == false`) tallies nothing.
    #[test]
    fn floor1_tally_unused_packet_records_nothing() {
        use crate::encoder::Floor1Packet;
        let header = tally_test_header();
        let books = tally_test_books();
        let mut tallies = BookTallies::new(&books);
        let packet = Floor1Packet {
            nonzero: false,
            floor1_y: vec![],
            partition_cvals: vec![],
        };
        tally_floor1_packet(&mut tallies, &packet, &header).expect("tallies");
        for b in 0..3 {
            assert_eq!(tallies.total(b), 0);
        }
    }

    /// Shape-gate error surface: wrong Y length, wrong cval count, and
    /// an out-of-range class index.
    #[test]
    fn floor1_tally_error_surface() {
        use crate::encoder::Floor1Packet;
        let header = tally_test_header();
        let books = tally_test_books();
        let mut tallies = BookTallies::new(&books);
        let bad_y = Floor1Packet {
            nonzero: true,
            floor1_y: vec![0; 5],
            partition_cvals: vec![0, 0],
        };
        assert_eq!(
            tally_floor1_packet(&mut tallies, &bad_y, &header),
            Err(BookDesignError::Floor1YLengthMismatch {
                expected: 6,
                actual: 5
            })
        );
        let bad_cvals = Floor1Packet {
            nonzero: true,
            floor1_y: vec![0; 6],
            partition_cvals: vec![0],
        };
        assert_eq!(
            tally_floor1_packet(&mut tallies, &bad_cvals, &header),
            Err(BookDesignError::Floor1CvalLengthMismatch {
                expected: 2,
                actual: 1
            })
        );
        let mut bad_class_header = header.clone();
        bad_class_header.partition_class_list[1] = 7;
        let ok_packet = Floor1Packet {
            nonzero: true,
            floor1_y: vec![0; 6],
            partition_cvals: vec![0, 0],
        };
        assert_eq!(
            tally_floor1_packet(&mut tallies, &ok_packet, &bad_class_header),
            Err(BookDesignError::Floor1ClassOutOfRange {
                partition: 1,
                class: 7,
                class_count: 2
            })
        );
        // Nothing was recorded by any failing call.
        for b in 0..3 {
            assert_eq!(tallies.total(b), 0);
        }
    }

    // ----------------------------------------------------------------
    // tally_residue_plans — the §8.6.2 emission tally.
    // ----------------------------------------------------------------

    /// A format-1 residue header with two classes: class 0 'unused'
    /// (no books anywhere in its cascade), class 1 carrying book 1 at
    /// pass 0 and book 2 at pass 1. Classbook 0 has dimensions 2
    /// (`classwords_per_codeword = 2`) and 4 entries (2 classes ^ 2).
    fn residue_tally_header() -> crate::setup::ResidueHeader {
        let unused: [Option<u8>; 8] = Default::default();
        let mut two_stage: [Option<u8>; 8] = Default::default();
        two_stage[0] = Some(1);
        two_stage[1] = Some(2);
        crate::setup::ResidueHeader {
            residue_type: 1,
            residue_begin: 0,
            residue_end: 64,
            partition_size: 8,
            classifications: 2,
            classbook: 0,
            cascade: vec![0, 0b11],
            books: vec![unused, two_stage],
        }
    }

    fn residue_tally_books() -> Vec<VorbisCodebook> {
        vec![
            design_entropy_codebook(4, 2, &[1u64; 4], 32, false).unwrap(),
            design_entropy_codebook(8, 1, &[1u64; 8], 32, false).unwrap(),
            design_entropy_codebook(8, 1, &[1u64; 8], 32, false).unwrap(),
        ]
    }

    /// The tally mirrors the §8.6.2 emission: classification strides
    /// pack into classbook entries, per-stage entry lists land on the
    /// cascade's books, 'unused' classes and stages emit nothing.
    #[test]
    fn residue_tally_records_classwords_and_value_entries() {
        use crate::encoder::ResidueVectorPlan;
        let header = residue_tally_header();
        let books = residue_tally_books();
        let mut tallies = BookTallies::new(&books);

        // Three partitions: class 1, class 0 (unused), class 1. With
        // classwords = 2 the strides are [1, 0] and [1, <pad 0>]:
        // packed entries 1·2+0 = 2 and 1·2+0 = 2.
        let mut p0: [Option<Vec<u32>>; 8] = Default::default();
        p0[0] = Some(vec![3, 3]);
        p0[1] = Some(vec![5]);
        let p1: [Option<Vec<u32>>; 8] = Default::default();
        let mut p2: [Option<Vec<u32>>; 8] = Default::default();
        p2[0] = Some(vec![3, 7]);
        p2[1] = Some(vec![5]);
        let plan = ResidueVectorPlan {
            classifications: vec![1, 0, 1],
            partition_entries: vec![p0, p1, p2],
        };
        tally_residue_plans(&mut tallies, &[plan], &header, &books).expect("tallies");

        // Classbook: two strides, both packing to entry 2.
        assert_eq!(tallies.counts(0).unwrap(), &[0, 0, 2, 0]);
        // Pass-0 book: entries 3, 3, 3, 7.
        assert_eq!(tallies.counts(1).unwrap()[3], 3);
        assert_eq!(tallies.counts(1).unwrap()[7], 1);
        assert_eq!(tallies.total(1), 4);
        // Pass-1 book: entry 5 twice.
        assert_eq!(tallies.counts(2).unwrap()[5], 2);
        assert_eq!(tallies.total(2), 2);
    }

    /// A 'do not decode' vector's empty plan tallies nothing.
    #[test]
    fn residue_tally_skips_do_not_decode_plans() {
        use crate::encoder::ResidueVectorPlan;
        let header = residue_tally_header();
        let books = residue_tally_books();
        let mut tallies = BookTallies::new(&books);
        let dnd = ResidueVectorPlan {
            classifications: vec![],
            partition_entries: vec![],
        };
        tally_residue_plans(&mut tallies, &[dnd], &header, &books).expect("tallies");
        for b in 0..3 {
            assert_eq!(tallies.total(b), 0);
        }
    }

    /// Error surface: plan shape mismatch, cascade disagreement, and a
    /// classification outside the header's class table.
    #[test]
    fn residue_tally_error_surface() {
        use crate::encoder::ResidueVectorPlan;
        let header = residue_tally_header();
        let books = residue_tally_books();
        let mut tallies = BookTallies::new(&books);

        let shape = ResidueVectorPlan {
            classifications: vec![0, 0],
            partition_entries: vec![Default::default()],
        };
        assert_eq!(
            tally_residue_plans(&mut tallies, &[shape], &header, &books),
            Err(BookDesignError::ResiduePlanShapeMismatch {
                classifications: 2,
                partition_entries: 1
            })
        );

        // Class 1's cascade holds a pass-0 book, but the plan omits it.
        let cascade = ResidueVectorPlan {
            classifications: vec![1],
            partition_entries: vec![Default::default()],
        };
        assert_eq!(
            tally_residue_plans(&mut tallies, &[cascade], &header, &books),
            Err(BookDesignError::ResiduePlanCascadeMismatch {
                partition: 0,
                pass: 0
            })
        );

        // Classification 5 is outside the two-class header. Packing
        // catches it first (the §8.6.2 grouping primitive validates
        // digits against `num_classifications`).
        let class_oor = ResidueVectorPlan {
            classifications: vec![5],
            partition_entries: vec![Default::default()],
        };
        assert!(matches!(
            tally_residue_plans(&mut tallies, &[class_oor], &header, &books),
            Err(BookDesignError::ResidueClassPack(_))
        ));

        // Classbook index outside the codebook table.
        let mut bad_header = residue_tally_header();
        bad_header.classbook = 9;
        let ok = ResidueVectorPlan {
            classifications: vec![0],
            partition_entries: vec![Default::default()],
        };
        assert_eq!(
            tally_residue_plans(&mut tallies, &[ok], &bad_header, &books),
            Err(BookDesignError::BookIndexOutOfRange { book: 9, books: 3 })
        );
        for b in 0..3 {
            assert_eq!(tallies.total(b), 0, "failing calls record nothing");
        }
    }

    // ----------------------------------------------------------------
    // tally_floor0_packet — the §6.2.2 emission tally.
    // ----------------------------------------------------------------

    fn floor0_tally_header() -> crate::setup::Floor0Header {
        crate::setup::Floor0Header {
            order: 8,
            rate: 44_100,
            bark_map_size: 128,
            amplitude_bits: 10,
            amplitude_offset: 32,
            // booknumber 0 → codebook 2, booknumber 1 → codebook 0.
            book_list: vec![2, 0],
        }
    }

    /// The VQ entry run is recorded against the book `[booknumber]`
    /// selects through `floor0_book_list`; amplitude / booknumber are
    /// raw fields and tally nothing.
    #[test]
    fn floor0_tally_records_entries_against_selected_book() {
        use crate::encoder::Floor0Packet;
        let header = floor0_tally_header();
        let books = vec![
            design_entropy_codebook(16, 1, &[1u64; 16], 32, false).unwrap(),
            design_entropy_codebook(4, 1, &[1u64; 4], 32, false).unwrap(),
            design_entropy_codebook(32, 2, &[1u64; 32], 32, false).unwrap(),
        ];
        let mut tallies = BookTallies::new(&books);
        let packet = Floor0Packet::Curve {
            amplitude: 500,
            booknumber: 0, // → book_list[0] = codebook 2
            entries: vec![7, 7, 30, 1],
        };
        tally_floor0_packet(&mut tallies, &packet, &header).expect("tallies");
        assert_eq!(tallies.total(2), 4);
        assert_eq!(tallies.counts(2).unwrap()[7], 2);
        assert_eq!(tallies.counts(2).unwrap()[30], 1);
        assert_eq!(tallies.counts(2).unwrap()[1], 1);
        assert_eq!(tallies.total(0), 0);
        assert_eq!(tallies.total(1), 0);

        // booknumber 1 routes to codebook 0.
        let second = Floor0Packet::Curve {
            amplitude: 12,
            booknumber: 1,
            entries: vec![3],
        };
        tally_floor0_packet(&mut tallies, &second, &header).expect("tallies");
        assert_eq!(tallies.counts(0).unwrap()[3], 1);
    }

    /// An unused floor-0 packet tallies nothing; error surface covers
    /// the out-of-range booknumber and an entry outside the selected
    /// book (atomically — nothing recorded).
    #[test]
    fn floor0_tally_unused_and_error_surface() {
        use crate::encoder::Floor0Packet;
        let header = floor0_tally_header();
        let books = vec![
            design_entropy_codebook(16, 1, &[1u64; 16], 32, false).unwrap(),
            design_entropy_codebook(4, 1, &[1u64; 4], 32, false).unwrap(),
            design_entropy_codebook(32, 2, &[1u64; 32], 32, false).unwrap(),
        ];
        let mut tallies = BookTallies::new(&books);
        tally_floor0_packet(&mut tallies, &Floor0Packet::Unused, &header).expect("tallies");
        for b in 0..3 {
            assert_eq!(tallies.total(b), 0);
        }
        assert_eq!(
            tally_floor0_packet(
                &mut tallies,
                &Floor0Packet::Curve {
                    amplitude: 1,
                    booknumber: 2,
                    entries: vec![0],
                },
                &header,
            ),
            Err(BookDesignError::Floor0BooknumberOutOfRange {
                booknumber: 2,
                books: 2
            })
        );
        // Entry 40 is outside codebook 2's 32 entries: the batch is
        // rejected atomically, so the in-range entry 5 before it is
        // not recorded either.
        assert_eq!(
            tally_floor0_packet(
                &mut tallies,
                &Floor0Packet::Curve {
                    amplitude: 1,
                    booknumber: 0,
                    entries: vec![5, 40],
                },
                &header,
            ),
            Err(BookDesignError::EntryOutOfRange {
                book: 2,
                entry: 40,
                entries: 32
            })
        );
        for b in 0..3 {
            assert_eq!(tallies.total(b), 0, "failing calls record nothing");
        }
    }

    // ----------------------------------------------------------------
    // train_residue_books_rd — guard-rail cases (the descent itself is
    // pinned by tests/residue_trained_books.rs against real bodies).
    // ----------------------------------------------------------------

    /// Zero iterations and a non-finite lambda are refused.
    #[test]
    fn rd_training_guards() {
        let header = residue_tally_header();
        let books = residue_tally_books();
        let residuals = vec![vec![0.25f32; 16]];
        assert_eq!(
            train_residue_books_rd(&residuals, &header, &books, 0.5, 0),
            Err(BookDesignError::ZeroIterations)
        );
        assert!(matches!(
            train_residue_books_rd(&residuals, &header, &books, f64::NAN, 2),
            Err(BookDesignError::ResidueEncode(_))
        ));
    }

    /// A trivially-stable corpus converges: identical plans on the
    /// second pass stop the loop with `converged = true`, and the
    /// Lagrangian never rises.
    #[test]
    fn rd_training_converges_on_stable_corpus() {
        let header = residue_tally_header();
        let books = residue_tally_books();
        // Residue books here are entropy-only (no VQ lookup), which the
        // planner rejects for value reads — build VQ-capable books.
        let mut books = books;
        for book in books.iter_mut().skip(1) {
            book.lookup = VqLookup::Tessellation {
                minimum_value: -2.0,
                delta_value: 0.5,
                value_bits: 8,
                sequence_p: false,
                multiplicands: (0..book.entries).collect(),
            };
        }
        let residuals: Vec<Vec<f32>> = (0..6)
            .map(|k| (0..16).map(|i| 0.5 * ((i + k) % 5) as f32 - 1.0).collect())
            .collect();
        let outcome =
            train_residue_books_rd(&residuals, &header, &books, 0.25, 10).expect("trains");
        assert!(outcome.converged, "stable corpus must reach a fixed point");
        assert!(outcome.lagrangian_per_iteration.len() >= 2);
        for w in outcome.lagrangian_per_iteration.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-9,
                "Lagrangian must never rise: {:?}",
                outcome.lagrangian_per_iteration
            );
        }
        assert_eq!(outcome.plans.len(), residuals.len());
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

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
    /// [`design_value_ladder`]: the training sample set was empty.
    EmptyTraining,
    /// [`design_value_ladder`]: a training sample was NaN or infinite.
    NonFiniteTraining {
        /// The offending sample index.
        index: usize,
    },
    /// [`design_value_ladder`]: `levels` was zero — a ladder needs at
    /// least one rung.
    ZeroLevels,
    /// [`design_value_ladder`]: `value_bits` outside the §3.2.1-legal
    /// `1..=16` range (the codebook header stores `value_bits − 1` in
    /// 4 bits).
    InvalidValueBits {
        /// The rejected width.
        value_bits: u8,
    },
    /// [`design_value_ladder`]: more ladder levels than a
    /// `value_bits`-wide multiplicand can address.
    LevelsExceedValueBits {
        /// The requested level count.
        levels: u32,
        /// The multiplicand width.
        value_bits: u8,
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
            BookDesignError::EmptyTraining => {
                write!(f, "vorbis book design: empty training sample set")
            }
            BookDesignError::NonFiniteTraining { index } => write!(
                f,
                "vorbis book design: training sample {index} is not finite"
            ),
            BookDesignError::ZeroLevels => write!(
                f,
                "vorbis book design: a value ladder needs at least one level"
            ),
            BookDesignError::InvalidValueBits { value_bits } => write!(
                f,
                "vorbis book design: value_bits {value_bits} outside 1..=16 (§3.2.1)"
            ),
            BookDesignError::LevelsExceedValueBits { levels, value_bits } => write!(
                f,
                "vorbis book design: {levels} levels cannot be addressed by {value_bits}-bit multiplicands"
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
        // Plan step: rate-distortion planning under the current books.
        let (plans, lagrangian, value_bits) = plan_corpus(residuals, header, &books, lambda)?;

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

/// The outcome of a [`train_residue_books_rd_ladder`] closed-loop
/// training run — [`ResidueRdTrainingOutcome`] extended with the
/// ladder-step bookkeeping.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidueLadderTrainingOutcome {
    /// The trained codebook table (index-aligned with the input).
    /// Value books the ladder step touched carry **new reconstruction
    /// values**, so — unlike [`train_residue_books_rd`] — packets
    /// planned under earlier iterations are *not* bit-identical under
    /// the final books; the final [`Self::plans`] are the matching
    /// plan set.
    pub codebooks: Vec<VorbisCodebook>,
    /// The final per-corpus-vector plans, chosen by the
    /// rate-distortion planner under [`Self::codebooks`].
    pub plans: Vec<crate::encoder::ResidueVectorPlan>,
    /// Per iteration: the corpus Lagrangian
    /// `Σ error_sq + lambda · value_bits`, measured at the plan step
    /// under that iteration's books. Monotone non-increasing: the
    /// plan and length-retrain steps descend as in
    /// [`train_residue_books_rd`], and the ladder step is
    /// accept-if-improved (a candidate that does not lower the
    /// re-planned Lagrangian is discarded).
    pub lagrangian_per_iteration: Vec<f64>,
    /// Per iteration: the corpus' total codeword bits (value
    /// codewords plus classwords) under that iteration's books.
    pub total_bits_per_iteration: Vec<u64>,
    /// `true` if the loop stopped at a plan fixed point.
    pub converged: bool,
    /// How many iterations' ladder candidates were adopted (they
    /// lowered, or matched, the re-planned Lagrangian).
    pub ladder_updates_accepted: usize,
    /// How many iterations' ladder candidates were discarded.
    pub ladder_updates_rejected: usize,
}

/// One corpus plan pass: rate-distortion plans for every residual
/// under the given books, plus the summed Lagrangian
/// `Σ error_sq + lambda · value_bits` and the value-bit total.
type CorpusPlan = (Vec<crate::encoder::ResidueVectorPlan>, f64, u64);

fn plan_corpus(
    residuals: &[Vec<f32>],
    header: &crate::setup::ResidueHeader,
    books: &[VorbisCodebook],
    lambda: f64,
) -> Result<CorpusPlan, BookDesignError> {
    // Resolve the per-class value-book rows against the book table
    // (§8.6.1's books[class][pass] indirection).
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

    let mut plans = Vec::with_capacity(residuals.len());
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
    Ok((plans, lagrangian, value_bits))
}

/// Build the ladder-update candidate: for every tessellation value
/// book the plans exercised, move each entry's reconstruction vector
/// to the **centroid of the target sub-vectors that selected it**
/// (recovered exactly by [`crate::residue_encode::replay_partition_cascade`]),
/// then re-express every entry on a fresh §9.2.2-packable
/// `minimum/delta` grid at the book's `value_bits`. Entries the corpus
/// never selected keep their old values (snapped to the new grid);
/// `sequence_p` books and lattice (lookup-type-1) books are left
/// untouched — their per-entry values are not independently free.
fn ladder_update_candidate(
    books: &[VorbisCodebook],
    plans: &[crate::encoder::ResidueVectorPlan],
    residuals: &[Vec<f32>],
    header: &crate::setup::ResidueHeader,
) -> Result<Vec<VorbisCodebook>, BookDesignError> {
    let ps = header.partition_size as usize;

    // Per-book accumulation: (Σ target per entry-component, count per entry).
    let mut acc: Vec<Option<(Vec<f64>, Vec<u64>)>> = books.iter().map(|_| None).collect();

    for (plan, residual) in plans.iter().zip(residuals) {
        if plan.classifications.is_empty() && plan.partition_entries.is_empty() {
            continue;
        }
        if plan.partition_entries.len() != plan.classifications.len() {
            return Err(BookDesignError::ResiduePlanShapeMismatch {
                classifications: plan.classifications.len(),
                partition_entries: plan.partition_entries.len(),
            });
        }
        let need = plan.classifications.len() * ps;
        if residual.len() != need {
            return Err(BookDesignError::ResidueEncode(
                crate::residue_encode::ResidueEncodeError::ResidualLengthMismatch {
                    expected: need,
                    actual: residual.len(),
                },
            ));
        }

        for (p, (&class, stages)) in plan
            .classifications
            .iter()
            .zip(plan.partition_entries.iter())
            .enumerate()
        {
            // A classification with no configured row decodes nothing
            // (mirrors the planner's padding of short book tables).
            let row = header.books.get(class as usize);
            let mut stage_books: [Option<&VorbisCodebook>; 8] = [None; 8];
            let mut stage_idx: [usize; 8] = [usize::MAX; 8];
            if let Some(row) = row {
                for (pass, slot) in row.iter().enumerate() {
                    if let Some(book) = slot {
                        stage_books[pass] = Some(books.get(*book as usize).ok_or(
                            BookDesignError::BookIndexOutOfRange {
                                book: *book as usize,
                                books: books.len(),
                            },
                        )?);
                        stage_idx[pass] = *book as usize;
                    }
                }
            }
            let scalars = &residual[p * ps..(p + 1) * ps];
            crate::residue_encode::replay_partition_cascade(
                scalars,
                stages,
                &stage_books,
                header.residue_type,
                header.partition_size,
                |pass, entry, target| {
                    // replay only reports populated stages, so the
                    // index is always resolved.
                    let idx = stage_idx[pass];
                    let book = &books[idx];
                    let dims = book.dimensions as usize;
                    let (sums, counts) = acc[idx].get_or_insert_with(|| {
                        (
                            vec![0.0f64; book.entries as usize * dims],
                            vec![0u64; book.entries as usize],
                        )
                    });
                    let base = entry as usize * dims;
                    for (d, &t) in target.iter().enumerate() {
                        sums[base + d] += f64::from(t);
                    }
                    counts[entry as usize] += 1;
                },
            )
            .map_err(BookDesignError::ResidueEncode)?;
        }
    }

    // Rebuild each exercised tessellation book around its centroids.
    let mut out = books.to_vec();
    for (idx, cell) in acc.iter().enumerate() {
        let Some((sums, counts)) = cell else { continue };
        let book = &books[idx];
        let VqLookup::Tessellation {
            minimum_value,
            delta_value,
            value_bits,
            sequence_p,
            multiplicands,
        } = &book.lookup
        else {
            continue;
        };
        if *sequence_p {
            continue;
        }
        let dims = book.dimensions as usize;
        let entries = book.entries as usize;
        if multiplicands.len() != entries * dims {
            continue; // malformed hand-built book: leave untouched
        }

        // Desired per-component values: centroids where observed, the
        // old decoded values elsewhere (unused entries keep meaning).
        let mut values = vec![0.0f64; entries * dims];
        for e in 0..entries {
            for d in 0..dims {
                values[e * dims + d] = if counts[e] > 0 {
                    sums[e * dims + d] / counts[e] as f64
                } else {
                    f64::from(multiplicands[e * dims + d]) * f64::from(*delta_value)
                        + f64::from(*minimum_value)
                };
            }
        }

        // Snap onto a fresh packable grid spanning the value range.
        let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let addressable = 1u64 << *value_bits;
        let (new_min, new_delta) = if hi - lo <= 0.0 {
            (pack_nearest(lo as f32), 0.0f32)
        } else {
            let raw = ((hi - lo) / (addressable - 1) as f64) as f32;
            (
                pack_nearest(lo as f32),
                pack_nearest(raw).max(f32::MIN_POSITIVE),
            )
        };
        let new_mults: Vec<u32> = values
            .iter()
            .map(|&v| {
                if new_delta == 0.0 {
                    0
                } else {
                    let m = ((v - f64::from(new_min)) / f64::from(new_delta)).round();
                    (m.max(0.0) as u64).min(addressable - 1) as u32
                }
            })
            .collect();
        out[idx].lookup = VqLookup::Tessellation {
            minimum_value: new_min,
            delta_value: new_delta,
            value_bits: *value_bits,
            sequence_p: false,
            multiplicands: new_mults,
        };
    }
    Ok(out)
}

/// Closed-loop rate-aware residue book training with a
/// **distortion-aware value-ladder step**: the
/// [`train_residue_books_rd`] alternating descent extended to update
/// the value books' *reconstruction values*, not just their codeword
/// prices.
///
/// The length-only trainer holds every book's VQ lookup fixed — a
/// deliberate invariant (old plans decode bit-identically) that also
/// caps how far the descent can go: if the initial ladder does not
/// span the corpus (or wastes rungs where no residual ever lands), no
/// amount of re-pricing helps. This trainer adds, per iteration, a
/// third step:
///
/// 1. **plan** — rate-distortion plans under the current books
///    (identical to [`train_residue_books_rd`]);
/// 2. **length retrain** — optimal sparse codeword lengths for the
///    plans' emission statistics (identical);
/// 3. **ladder step** — each exercised tessellation value book's
///    entries move to the centroid of the target sub-vectors that
///    chose them (the classic VQ codebook update, targets recovered
///    exactly by [`crate::residue_encode::replay_partition_cascade`]),
///    snapped to a fresh §9.2.2-packable grid. Because new values
///    change every subsequent plan — and because cascade stages
///    interact (improving an early stage shrinks a later stage's
///    targets, so a joint update from stale targets can regress) —
///    the step is **accept-if-improved over candidates**: the joint
///    update and each single-book update are each evaluated by a
///    fresh plan pass, and the best *strict* improvement over the
///    length-retrained books is adopted (none ⇒ the values stay). The
///    recorded Lagrangian is therefore monotone non-increasing
///    whether or not any candidate is accepted, and multi-stage
///    ladders converge stage-by-stage against re-derived targets.
///
/// Unlike the length-only trainer, an adopted ladder step changes
/// reconstruction values, so earlier packets are **not** bit-identical
/// under the final books — the final [`ResidueLadderTrainingOutcome::plans`]
/// are the matching plan set to serialise. `sequence_p` and lattice
/// books are never touched (their entry values are not independently
/// free); books the corpus never exercises are unchanged.
///
/// # Errors
///
/// As [`train_residue_books_rd`]: [`BookDesignError::ZeroIterations`],
/// a book index outside the table, or any planner/tally error.
pub fn train_residue_books_rd_ladder(
    residuals: &[Vec<f32>],
    header: &crate::setup::ResidueHeader,
    codebooks: &[VorbisCodebook],
    lambda: f64,
    max_iterations: usize,
) -> Result<ResidueLadderTrainingOutcome, BookDesignError> {
    if max_iterations == 0 {
        return Err(BookDesignError::ZeroIterations);
    }
    let classbook_idx = header.classbook as usize;

    let mut books: Vec<VorbisCodebook> = codebooks.to_vec();
    let mut prev_plans: Option<Vec<crate::encoder::ResidueVectorPlan>> = None;
    let mut carried: Option<CorpusPlan> = None;
    let mut lagrangians: Vec<f64> = Vec::new();
    let mut totals: Vec<u64> = Vec::new();
    let mut converged = false;
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for _ in 0..max_iterations {
        // Plan step (reusing the evaluation pass that chose the
        // current books, when one exists).
        let (plans, lagrangian, value_bits) = match carried.take() {
            Some(t) => t,
            None => plan_corpus(residuals, header, &books, lambda)?,
        };

        // Measure the classword bits under the current classbook.
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
        // frequencies and identical centroids.
        if prev_plans.as_ref() == Some(&plans) {
            converged = true;
            prev_plans = Some(plans);
            break;
        }

        // Length retrain (sparse — exactly optimal for the plans).
        let books_len = tallies.retrain(&books, MAX_CODEWORD_LEN, false)?;
        // Ladder candidate values from the same plans. The length
        // retrain preserves every lookup, so the replayed
        // reconstructions are exactly the ones the plans were made
        // against.
        let books_ladder = ladder_update_candidate(&books_len, &plans, residuals, header)?;
        let touched: Vec<usize> = (0..books_len.len())
            .filter(|&i| books_ladder[i].lookup != books_len[i].lookup)
            .collect();

        // Accept-if-improved, evaluated by fresh plan passes. Cascade
        // stages interact — improving an early stage shrinks a later
        // stage's targets, so a joint update computed from the *old*
        // targets can regress even when each book's own move is sound.
        // Try the joint candidate AND each single-book candidate; keep
        // the best strict improvement over the length-only books (the
        // remaining books catch up on later iterations, against
        // re-derived targets).
        let eval_len = plan_corpus(residuals, header, &books_len, lambda)?;
        let mut best_books = books_len;
        let mut best_eval = eval_len;
        let mut adopted = false;
        let consider = |cand: Vec<VorbisCodebook>,
                        best_books: &mut Vec<VorbisCodebook>,
                        best_eval: &mut CorpusPlan,
                        adopted: &mut bool|
         -> Result<(), BookDesignError> {
            let eval = plan_corpus(residuals, header, &cand, lambda)?;
            if eval.1 < best_eval.1 {
                *best_books = cand;
                *best_eval = eval;
                *adopted = true;
            }
            Ok(())
        };
        if touched.len() > 1 {
            consider(
                books_ladder.clone(),
                &mut best_books,
                &mut best_eval,
                &mut adopted,
            )?;
        }
        for &i in &touched {
            let mut single = best_books.clone();
            // Candidate: only book `i`'s values move (relative to the
            // length-retrained table the loop otherwise keeps).
            single[i] = books_ladder[i].clone();
            if single[i].lookup == best_books[i].lookup {
                continue; // already adopted via the joint candidate
            }
            consider(single, &mut best_books, &mut best_eval, &mut adopted)?;
        }
        if adopted {
            accepted += 1;
        } else {
            rejected += 1;
        }
        books = best_books;
        carried = Some(best_eval);
        prev_plans = Some(plans);
    }

    // On convergence the last measured plans were planned under the
    // final books (no update ran after them). On a max_iterations
    // stop, the winner's evaluation pass — planned under the final
    // (possibly ladder-updated) books — is the matching set.
    let final_plans = match carried {
        Some((plans, _, _)) if !converged => plans,
        _ => prev_plans.expect("max_iterations >= 1 ran at least one plan pass"),
    };

    Ok(ResidueLadderTrainingOutcome {
        codebooks: books,
        plans: final_plans,
        lagrangian_per_iteration: lagrangians,
        total_bits_per_iteration: totals,
        converged,
        ladder_updates_accepted: accepted,
        ladder_updates_rejected: rejected,
    })
}

/// A designed §3.2.1 VQ value ladder — the `minimum_value` /
/// `delta_value` / `multiplicands` triple a lookup-type-1/2 codebook
/// carries, produced by [`design_value_ladder`].
#[derive(Debug, Clone, PartialEq)]
pub struct ValueLadderDesign {
    /// The §9.2.2-packable ladder base (`float32_pack` accepts it, so
    /// `crate::encoder::write_codebook` can carry the ladder).
    pub minimum_value: f32,
    /// The §9.2.2-packable ladder step.
    pub delta_value: f32,
    /// One multiplicand per designed level, ascending, each strictly
    /// below `2^value_bits`. Level `i` decodes (with `sequence_p`
    /// clear) to `multiplicands[i] · delta_value + minimum_value`.
    pub multiplicands: Vec<u32>,
    /// The multiplicand bit width the design honoured.
    pub value_bits: u8,
    /// The mean squared quantisation error of the training samples
    /// against the final (grid-snapped) ladder.
    pub mse: f64,
}

impl ValueLadderDesign {
    /// The decoded scalar value of ladder level `i` (with `sequence_p`
    /// clear): `multiplicands[i] · delta + minimum`, exactly the
    /// §3.2.1 lookup arithmetic.
    #[must_use]
    pub fn level_value(&self, i: usize) -> Option<f32> {
        self.multiplicands
            .get(i)
            .map(|&m| m as f32 * self.delta_value + self.minimum_value)
    }

    /// Wrap the ladder as a 1-D [`VqLookup::Tessellation`] table for a
    /// book of `multiplicands.len()` entries — entry `i` decodes to
    /// [`Self::level_value`]`(i)`.
    #[must_use]
    pub fn into_tessellation_lookup(self) -> VqLookup {
        VqLookup::Tessellation {
            minimum_value: self.minimum_value,
            delta_value: self.delta_value,
            value_bits: self.value_bits,
            sequence_p: false,
            multiplicands: self.multiplicands,
        }
    }
}

/// Round a finite `f32` to the nearest §9.2.2-packable value (21-bit
/// mantissa × power of two): the largest-magnitude ladder parameters a
/// codebook header can carry exactly.
pub(crate) fn pack_nearest(x: f32) -> f32 {
    if crate::codebook::float32_pack(x).is_some() {
        return x;
    }
    // An f32 has a 24-bit significand; §9.2.2 carries 21 bits. Round
    // the significand to 21 bits (ties away from zero — a half-ulp
    // choice invisible at ladder scale) and rebuild.
    let bits = x.abs().to_bits();
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0 {
        // Subnormal f32s are far below any audio ladder scale; snap
        // to zero (packable).
        return 0.0;
    }
    let full = mant | (1 << 23); // 24-bit significand
    let rounded = (full + 4) >> 3; // 21 bits, round-half-up
    let value = rounded as f32 * (2.0f32).powi(exp - 127 - 20);
    if x.is_sign_negative() {
        -value
    } else {
        value
    }
}

/// Design a §3.2.1 VQ value ladder from training scalars: the
/// *value*-side half of codebook training (the codeword-length half is
/// [`design_codeword_lengths`]).
///
/// A lookup-type-1/2 codebook reconstructs every scalar as
/// `multiplicand · delta + minimum`, so designing the ladder means
/// choosing `levels` reconstruction points that minimise the training
/// set's quantisation error, then expressing them in the §3.2.1 grid
/// form. The optimiser is the classic 1-D Lloyd iteration
/// (nearest-level assignment ↔ level-at-centroid update, initialised
/// at the sorted sample quantiles — deterministic, no randomness),
/// which monotonically reduces the MSE; the converged centroids are
/// then snapped to the multiplicand grid: `delta` spans the centroid
/// range across the `value_bits`-wide multiplicand space, `minimum` /
/// `delta` are rounded to §9.2.2-packable floats ([`float32_pack`]
/// accepts them, so `crate::encoder::write_codebook` carries the
/// ladder exactly), and each centroid becomes its nearest grid rung.
///
/// The returned [`ValueLadderDesign::mse`] is measured against the
/// **final snapped** ladder — the reconstruction points a decoder will
/// actually produce — not the ideal centroids.
///
/// [`float32_pack`]: crate::codebook::float32_pack
pub fn design_value_ladder(
    samples: &[f32],
    levels: u32,
    value_bits: u8,
) -> Result<ValueLadderDesign, BookDesignError> {
    if samples.is_empty() {
        return Err(BookDesignError::EmptyTraining);
    }
    if let Some(index) = samples.iter().position(|v| !v.is_finite()) {
        return Err(BookDesignError::NonFiniteTraining { index });
    }
    if levels == 0 {
        return Err(BookDesignError::ZeroLevels);
    }
    if !(1..=16).contains(&value_bits) {
        return Err(BookDesignError::InvalidValueBits { value_bits });
    }
    let addressable = 1u64 << value_bits;
    if levels as u64 > addressable {
        return Err(BookDesignError::LevelsExceedValueBits { levels, value_bits });
    }
    let k = levels as usize;

    // ---- Lloyd iteration in 1-D over the sorted samples. ----
    let mut sorted: Vec<f32> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finiteness checked above"));

    // Quantile initialisation: centroid i at the midpoint of its
    // equal-population slice.
    let n = sorted.len();
    let mut centers: Vec<f64> = (0..k)
        .map(|i| {
            let idx = ((2 * i + 1) * n / (2 * k)).min(n - 1);
            sorted[idx] as f64
        })
        .collect();

    for _ in 0..64 {
        // Assignment: samples are sorted and centers are ascending, so
        // each center's cell is a contiguous run bounded by midpoints.
        let mut sums = vec![0.0f64; k];
        let mut counts = vec![0usize; k];
        let mut c = 0usize;
        for &s in &sorted {
            let s = s as f64;
            while c + 1 < k && (centers[c + 1] + centers[c]) / 2.0 < s {
                c += 1;
            }
            // `c` is the first cell whose upper midpoint bound is at or
            // above `s` — the nearest center for ascending input.
            sums[c] += s;
            counts[c] += 1;
        }
        // Update: move each populated center to its cell centroid.
        let mut changed = false;
        for i in 0..k {
            if counts[i] > 0 {
                let next = sums[i] / counts[i] as f64;
                if (next - centers[i]).abs() > 1e-12 {
                    centers[i] = next;
                    changed = true;
                }
            }
        }
        centers.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        if !changed {
            break;
        }
    }

    // ---- Snap the centroids to the §3.2.1 multiplicand grid. ----
    let lo = centers[0];
    let hi = centers[k - 1];
    let span = hi - lo;
    let (minimum_value, delta_value) = if span <= 0.0 {
        // Degenerate single-point ladder: every level reconstructs the
        // same value; delta 0 keeps the arithmetic exact.
        (pack_nearest(lo as f32), 0.0f32)
    } else {
        let raw_delta = (span / (addressable - 1) as f64) as f32;
        let delta = pack_nearest(raw_delta).max(f32::MIN_POSITIVE);
        (pack_nearest(lo as f32), delta)
    };

    let multiplicands: Vec<u32> = centers
        .iter()
        .map(|&c| {
            if delta_value == 0.0 {
                0
            } else {
                let m = ((c - minimum_value as f64) / delta_value as f64).round();
                (m.max(0.0) as u64).min(addressable - 1) as u32
            }
        })
        .collect();

    // ---- Final MSE against the snapped reconstruction points. ----
    let points: Vec<f64> = multiplicands
        .iter()
        .map(|&m| m as f64 * delta_value as f64 + minimum_value as f64)
        .collect();
    let mut err = 0.0f64;
    for &s in &sorted {
        let s = s as f64;
        let best = points
            .iter()
            .map(|&p| (s - p) * (s - p))
            .fold(f64::INFINITY, f64::min);
        err += best;
    }
    let mse = err / n as f64;

    debug_assert!(crate::codebook::float32_pack(minimum_value).is_some());
    debug_assert!(crate::codebook::float32_pack(delta_value).is_some());

    Ok(ValueLadderDesign {
        minimum_value,
        delta_value,
        multiplicands,
        value_bits,
        mse,
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

    // ----------------------------------------------------------------
    // train_residue_books_rd_ladder — the distortion-aware value step.
    // ----------------------------------------------------------------

    /// VQ-capable variant of the residue tally books: dim-1 8-entry
    /// tessellation value books whose ladder spans `[-2, 1.5]`.
    fn ladder_test_books() -> Vec<VorbisCodebook> {
        let mut books = residue_tally_books();
        for book in books.iter_mut().skip(1) {
            book.lookup = VqLookup::Tessellation {
                minimum_value: -2.0,
                delta_value: 0.5,
                value_bits: 8,
                sequence_p: false,
                multiplicands: (0..book.entries).collect(),
            };
        }
        books
    }

    /// A corpus whose residuals cluster at ±5 — far outside the seed
    /// ladder's `[-2, 1.5]` span. Length-only retraining can never
    /// reach them (it must not move reconstruction values); the ladder
    /// step must, cutting the corpus distortion decisively.
    #[test]
    fn ladder_training_fixes_a_mismatched_ladder() {
        let header = residue_tally_header();
        let books = ladder_test_books();
        let residuals: Vec<Vec<f32>> = (0..6)
            .map(|k| {
                (0..16)
                    .map(|i| {
                        let sign = if (i + k) % 2 == 0 { 1.0f32 } else { -1.0 };
                        sign * (5.0 + 0.02 * ((i * 7 + k) % 5) as f32)
                    })
                    .collect()
            })
            .collect();
        let lambda = 0.25;

        let length_only =
            train_residue_books_rd(&residuals, &header, &books, lambda, 12).expect("trains");
        let with_ladder =
            train_residue_books_rd_ladder(&residuals, &header, &books, lambda, 12).expect("trains");

        // The ladder trainer descends monotonically…
        for w in with_ladder.lagrangian_per_iteration.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-9,
                "Lagrangian must never rise: {:?}",
                with_ladder.lagrangian_per_iteration
            );
        }
        // …accepts at least one value update…
        assert!(
            with_ladder.ladder_updates_accepted >= 1,
            "a mismatched ladder must be updated (accepted {}, rejected {})",
            with_ladder.ladder_updates_accepted,
            with_ladder.ladder_updates_rejected
        );
        // …and lands far below what re-pricing alone can reach.
        let final_len = *length_only.lagrangian_per_iteration.last().unwrap();
        let final_ladder = *with_ladder.lagrangian_per_iteration.last().unwrap();
        assert!(
            final_ladder < 0.5 * final_len,
            "ladder {final_ladder} must decisively beat length-only {final_len}"
        );
    }

    /// The final books stay §3.2.1 carriage-legal after value updates:
    /// every trained book (skipping never-emitted sparse books that
    /// retrain to all-unused) serialises through `write_codebook` and
    /// parses back field-for-field, and the final plans replay against
    /// the final books (shape-consistent entry lists).
    #[test]
    fn ladder_trained_books_stay_carriage_legal() {
        use crate::encoder::write_codebook;
        use oxideav_core::bits::BitReaderLsb;

        let header = residue_tally_header();
        let books = ladder_test_books();
        let residuals: Vec<Vec<f32>> = (0..4)
            .map(|k| (0..16).map(|i| ((i + k) % 7) as f32 - 3.0).collect())
            .collect();
        let outcome =
            train_residue_books_rd_ladder(&residuals, &header, &books, 0.25, 8).expect("trains");

        for (i, book) in outcome.codebooks.iter().enumerate() {
            if book.codeword_lengths.iter().all(|&l| l == UNUSED_ENTRY) {
                continue; // never-emitted → no carriage to check
            }
            let bytes = write_codebook(book).unwrap_or_else(|e| panic!("book {i} writes: {e}"));
            let mut reader = BitReaderLsb::new(&bytes);
            let parsed =
                crate::codebook::parse_codebook(&mut reader).expect("trained book parses back");
            assert_eq!(&parsed, book, "book {i} round-trips");
        }

        // The stored plans belong to the final books: replaying each
        // partition's cascade against them succeeds shape-exactly.
        for (plan, residual) in outcome.plans.iter().zip(&residuals) {
            for (p, (&class, stages)) in plan
                .classifications
                .iter()
                .zip(plan.partition_entries.iter())
                .enumerate()
            {
                let mut stage_books: [Option<&VorbisCodebook>; 8] = [None; 8];
                if let Some(row) = header.books.get(class as usize) {
                    for (pass, slot) in row.iter().enumerate() {
                        if let Some(b) = slot {
                            stage_books[pass] = Some(&outcome.codebooks[*b as usize]);
                        }
                    }
                }
                let ps = header.partition_size as usize;
                crate::residue_encode::replay_partition_cascade(
                    &residual[p * ps..(p + 1) * ps],
                    stages,
                    &stage_books,
                    header.residue_type,
                    header.partition_size,
                    |_, _, _| {},
                )
                .expect("final plans replay against final books");
            }
        }
    }

    /// When the seed ladder already carries the corpus exactly, the
    /// value step has nothing to win: the trainer converges and its
    /// final Lagrangian matches the length-only trainer's.
    #[test]
    fn ladder_training_matches_length_only_on_an_ideal_ladder() {
        let header = residue_tally_header();
        let books = ladder_test_books();
        // Residual values drawn exactly from the ladder rungs.
        let residuals: Vec<Vec<f32>> = (0..4)
            .map(|k| (0..16).map(|i| -2.0 + 0.5 * ((i + k) % 8) as f32).collect())
            .collect();
        let lambda = 0.25;
        let length_only =
            train_residue_books_rd(&residuals, &header, &books, lambda, 8).expect("trains");
        let with_ladder =
            train_residue_books_rd_ladder(&residuals, &header, &books, lambda, 8).expect("trains");
        assert!(with_ladder.converged, "ideal corpus reaches a fixed point");
        let a = *length_only.lagrangian_per_iteration.last().unwrap();
        let b = *with_ladder.lagrangian_per_iteration.last().unwrap();
        assert!(
            (a - b).abs() <= 1e-9 * a.abs().max(1.0),
            "ideal ladder: length-only {a} vs ladder {b} must agree"
        );
    }

    /// Guards mirror the length-only trainer's.
    #[test]
    fn ladder_training_guards() {
        let header = residue_tally_header();
        let books = ladder_test_books();
        let residuals = vec![vec![0.25f32; 16]];
        assert_eq!(
            train_residue_books_rd_ladder(&residuals, &header, &books, 0.5, 0),
            Err(BookDesignError::ZeroIterations)
        );
        assert!(matches!(
            train_residue_books_rd_ladder(&residuals, &header, &books, f64::NAN, 2),
            Err(BookDesignError::ResidueEncode(_))
        ));
    }

    // ----------------------------------------------------------------
    // design_value_ladder — the VQ value-side designer.
    // ----------------------------------------------------------------

    /// Two tight clusters, two levels: the designed ladder lands one
    /// reconstruction point near each cluster mean and beats the
    /// uniform (range-spanning) ladder's MSE decisively.
    #[test]
    fn value_ladder_finds_clusters_and_beats_uniform() {
        let mut samples = Vec::new();
        for i in 0..50 {
            samples.push(-1.0 + 0.001 * (i % 7) as f32);
            samples.push(2.0 + 0.001 * (i % 5) as f32);
        }
        let design = design_value_ladder(&samples, 2, 8).expect("designs");
        assert_eq!(design.multiplicands.len(), 2);
        let l0 = design.level_value(0).unwrap();
        let l1 = design.level_value(1).unwrap();
        assert!((l0 - (-1.0)).abs() < 0.05, "level 0 near cluster: {l0}");
        assert!((l1 - 2.0).abs() < 0.05, "level 1 near cluster: {l1}");

        // Uniform 2-level ladder over the same range: endpoints only.
        let lo = -1.0f64;
        let hi = 2.004f64;
        let uniform_mse: f64 = samples
            .iter()
            .map(|&s| {
                let s = s as f64;
                (s - lo).powi(2).min((s - hi).powi(2))
            })
            .sum::<f64>()
            / samples.len() as f64;
        assert!(
            design.mse <= uniform_mse,
            "designed {} must not exceed uniform {}",
            design.mse,
            uniform_mse
        );
        assert!(design.mse < 1e-4, "clusters are tight: {}", design.mse);
    }

    /// The designed ladder is carriage-exact: a book carrying it
    /// serialises through the §3.2.1 writer, parses back
    /// field-for-field, and each entry unpacks to the design's level
    /// value; the §3.2.1 quantiser picks the nearest designed level.
    #[test]
    fn value_ladder_book_carries_and_quantises() {
        use crate::encoder::write_codebook;
        use crate::vq::{quantize_vector, unpack_vector};
        use oxideav_core::bits::BitReaderLsb;

        let samples: Vec<f32> = (0..200)
            .map(|i| ((i * 37) % 101) as f32 * 0.02 - 1.0)
            .collect();
        let design = design_value_ladder(&samples, 8, 8).expect("designs");
        let levels: Vec<f32> = (0..8).map(|i| design.level_value(i).unwrap()).collect();
        let lengths = design_codeword_lengths(&[1u64; 8], 32).unwrap();
        let book = VorbisCodebook {
            dimensions: 1,
            entries: 8,
            codeword_lengths: lengths,
            lookup: design.clone().into_tessellation_lookup(),
        };
        let bytes = write_codebook(&book).expect("writes");
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            crate::codebook::parse_codebook(&mut r).expect("parses"),
            book
        );
        for (i, &lv) in levels.iter().enumerate() {
            assert_eq!(unpack_vector(&book, i as u32).unwrap(), vec![lv]);
        }
        // Quantising a value near level 3 picks level 3.
        let target = levels[3] + 0.001;
        let q = quantize_vector(&book, &[target]).expect("quantises");
        assert_eq!(q.entry, 3);
    }

    /// Degenerate constant training data yields the exact single-point
    /// ladder (delta 0, zero error), and the guard-rail error surface
    /// fires.
    #[test]
    fn value_ladder_degenerate_and_error_surface() {
        let design = design_value_ladder(&[0.25; 30], 4, 8).expect("designs");
        assert_eq!(design.delta_value, 0.0);
        assert!(design.multiplicands.iter().all(|&m| m == 0));
        assert!(design.mse < 1e-12);
        for i in 0..4 {
            assert_eq!(design.level_value(i), Some(design.minimum_value));
        }

        assert_eq!(
            design_value_ladder(&[], 2, 8),
            Err(BookDesignError::EmptyTraining)
        );
        assert_eq!(
            design_value_ladder(&[1.0, f32::NAN], 2, 8),
            Err(BookDesignError::NonFiniteTraining { index: 1 })
        );
        assert_eq!(
            design_value_ladder(&[1.0], 0, 8),
            Err(BookDesignError::ZeroLevels)
        );
        assert_eq!(
            design_value_ladder(&[1.0], 2, 0),
            Err(BookDesignError::InvalidValueBits { value_bits: 0 })
        );
        assert_eq!(
            design_value_ladder(&[1.0], 2, 17),
            Err(BookDesignError::InvalidValueBits { value_bits: 17 })
        );
        assert_eq!(
            design_value_ladder(&[1.0], 5, 2),
            Err(BookDesignError::LevelsExceedValueBits {
                levels: 5,
                value_bits: 2
            })
        );
    }

    /// The ladder parameters are always §9.2.2-packable — including
    /// when the centroid span produces a non-dyadic raw delta.
    #[test]
    fn value_ladder_parameters_are_always_packable() {
        use crate::codebook::float32_pack;
        let mut rng = Rng(0x1ADD_E12D_E51F_0001);
        for _ in 0..40 {
            let n = 20 + (rng.next() % 200) as usize;
            let scale = 0.001 + (rng.next() % 1000) as f32 * 0.01;
            let samples: Vec<f32> = (0..n)
                .map(|_| ((rng.next() % 2001) as f32 - 1000.0) * scale / 1000.0)
                .collect();
            let levels = 2 + (rng.next() % 15) as u32;
            let design = design_value_ladder(&samples, levels, 8).expect("designs");
            assert!(float32_pack(design.minimum_value).is_some());
            assert!(float32_pack(design.delta_value).is_some());
            assert!(design
                .multiplicands
                .iter()
                .all(|&m| m < (1u32 << design.value_bits)));
            assert!(design.mse.is_finite());
        }
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

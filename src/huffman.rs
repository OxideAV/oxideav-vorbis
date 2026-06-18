//! Vorbis I canonical Huffman tree builder + entry decoder (§3.2.1).
//!
//! This module turns the per-entry codeword-length table produced by
//! [`crate::codebook::parse_codebook`] into a Huffman decision tree that
//! [`HuffmanTree::decode_entry`] can walk against an LSb-first packet
//! bitstream.
//!
//! ## Codeword assignment (Vorbis I §3.2.1)
//!
//! Vorbis Huffman codewords are *canonical*: "each used codebook entry is
//! assigned, in order, the lowest valued unused binary Huffman codeword
//! possible." The spec calls out the bit order explicitly:
//!
//! > Note: Unlike most binary numerical values in this document, we intend
//! > the above codewords to be read and used bit by bit from left to right,
//! > thus the codeword '001' is the bit string 'zero, zero, one'. When
//! > determining 'lowest possible value' in the assignment definition
//! > above, the leftmost bit is the MSb.
//!
//! The spec then gives the worked example for lengths `[2 4 4 4 4 2 3 3]`:
//!
//! ```text
//! entry 0: length 2 codeword 00
//! entry 1: length 4 codeword 0100
//! entry 2: length 4 codeword 0101
//! entry 3: length 4 codeword 0110
//! entry 4: length 4 codeword 0111
//! entry 5: length 2 codeword 10
//! entry 6: length 3 codeword 110
//! entry 7: length 3 codeword 111
//! ```
//!
//! Construction places each used entry, in entry order, at the *lowest
//! valued unused codeword* of that entry's length. The lowest free
//! codeword of length `L` is found by descending the partially-built
//! decision tree from the root, preferring the `0` (left) child at every
//! level and falling back to `1` only when the `0` subtree can no longer
//! host an `L`-deep leaf (because a shorter codeword already terminates
//! there, or it is saturated). Interior nodes are materialised lazily
//! along the chosen path. This realises the spec's MSb-first
//! "lowest valued unused canonical codeword" rule directly against the
//! tree, which is the unambiguous source of truth even when codeword
//! lengths are *not* non-decreasing — e.g. `[2 3 3 3 3 4 3 4]`, where a
//! shorter codeword follows a longer one. (An earlier left-to-right
//! "open-slot deque" assigned the leftmost *tree* position at the target
//! depth, which coincides with the lowest-valued codeword only for
//! non-decreasing lengths; on the interleaved books that real-stream
//! floor / residue classification produces it diverged and spuriously
//! reported the book underspecified.)
//!
//! The convention used in this module: a node is a [`HuffmanNode`] enum
//! that is either a `Leaf(entry_index)` or an `Internal { left, right }`.
//! At decode time, reading a single bit `0` selects `left`, a `1` selects
//! `right`; the first bit read is interpreted as the *MSb* of the
//! canonical codeword (matching the spec's "leftmost bit is the MSb").
//!
//! Under the workspace's LSb-first [`BitReaderLsb`] packing convention
//! (§2.1.4: "values are packed from the least-significant bit upward"),
//! the first bit read from the stream is the LSb of the next byte. So
//! the decoder simply reads one bit at a time and walks the tree —
//! integer values of codewords as "MSb-first" need never be materialised
//! at decode time; they only matter when *constructing* the tree, which
//! happens at setup once per codebook.
//!
//! ## Underspecified / overspecified detection (errata 20150226)
//!
//! §3.2.1 closes with:
//!
//! > It is clear that the codeword length list represents a Huffman
//! > decision tree with the entry numbers equivalent to the leaves
//! > numbered left-to-right. […] Both underspecified and overspecified
//! > trees are an error condition rendering the stream undecodable.
//!
//! The construction algorithm detects both:
//!
//! * **Overspecified** — no free codeword of an entry's length remains
//!   (the descent dead-ends); the length list demands more codewords
//!   than the tree can host. Surfaces as
//!   [`BuildError::OverspecifiedTree`].
//! * **Underspecified** — after every used entry is placed, some
//!   internal node still has a dangling child: the codeword-length list
//!   does not fully populate the decision tree. Surfaces as
//!   [`BuildError::UnderspecifiedTree`].
//!
//! ## Single-entry codebook (errata 20150226)
//!
//! The 2015-02-26 erratum to Vorbis I §3.2.1 special-cases a codebook
//! with exactly one used entry:
//!
//! > Decoder implementations shall reject a codebook if it contains only
//! > one used entry and the encoded `[codeword_length]` of that entry is
//! > not 1. 'Reading' a value from single-entry codebook always returns
//! > the single used codeword value and sinks one bit. Decoders should
//! > tolerate that the bit read from the stream be '1' instead of '0';
//! > both values shall return the single used codeword.
//!
//! This module:
//! * Rejects a single-used-entry codebook whose recorded length is not
//!   `1` (returns [`BuildError::SingleEntryWrongLength`]).
//! * Builds a synthetic tree whose root is *both* children pointing to a
//!   leaf containing the sole used entry, so `decode_entry` always
//!   returns that entry regardless of whether the bit read is 0 or 1.
//!
//! ## Fully-unused codebook
//!
//! A codebook with zero used entries cannot be walked (the spec
//! [`UNUSED_ENTRY`](crate::codebook::UNUSED_ENTRY) sentinel marks entries
//! that the encoder explicitly left out). Building one is rejected with
//! [`BuildError::EmptyTree`] — there is no leaf to return.

use core::fmt;

use oxideav_core::bits::{BitReaderLsb, BitWriterLsb};

use crate::codebook::{VorbisCodebook, UNUSED_ENTRY};

/// One node of a canonical Vorbis Huffman tree.
///
/// `Internal` nodes carry indices into the tree's owning [`HuffmanTree::nodes`]
/// arena rather than `Box<Self>` so a single decode-time `match` over the
/// node array avoids repeated heap indirection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HuffmanNode {
    /// A leaf node: decoding stops here and the inner value is the codebook
    /// entry index returned to the caller of [`HuffmanTree::decode_entry`].
    Leaf(u32),
    /// An internal node: `left` is taken on a `0` bit, `right` on a `1`
    /// bit. Both child indices refer into the tree's `nodes` arena.
    Internal {
        /// Child index taken when the next bit read is `0`.
        left: u32,
        /// Child index taken when the next bit read is `1`.
        right: u32,
    },
}

/// A canonical Vorbis Huffman decision tree (Vorbis I §3.2.1).
///
/// Construct one with [`HuffmanTree::from_codebook`] (or directly from a
/// length slice with [`HuffmanTree::from_lengths`]) and decode codewords
/// against it with [`HuffmanTree::decode_entry`].
///
/// The tree owns its node arena flat in a [`Vec<HuffmanNode>`]; index 0
/// is always the root.
#[derive(Debug, Clone)]
pub struct HuffmanTree {
    nodes: Vec<HuffmanNode>,
    /// Number of used entries (equivalently, the leaf count); zero is
    /// rejected by the constructor.
    used_count: u32,
    /// `Some(entry)` iff the tree was built from a single-used-entry
    /// codebook (errata 20150226). `decode_entry` short-circuits in
    /// this case to always return `entry` while sinking one bit.
    single_entry: Option<u32>,
}

/// Errors raised by [`HuffmanTree::from_codebook`] /
/// [`HuffmanTree::from_lengths`] when the length list cannot define a
/// valid Huffman tree per Vorbis I §3.2.1 (with errata 20150226).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// Every entry in the length table is [`UNUSED_ENTRY`]; the tree
    /// would have no leaves.
    EmptyTree,
    /// A codeword length was outside the spec-legal range
    /// `1..=32`. (The codebook parser emits `length - 1` as a 5-bit
    /// field, so lengths up to 32 are reachable; anything else means
    /// the caller fed a hand-rolled length table.)
    InvalidLength {
        /// The offending entry index in the source length list.
        entry: u32,
        /// The recorded length value (in the same encoding as
        /// [`VorbisCodebook::codeword_lengths`]: `0` for unused,
        /// `1..=32` for used).
        length: u8,
    },
    /// The greedy canonical assignment ran out of codeword space at the
    /// requested length — the tree is over-populated. Per §3.2.1: "the
    /// tree is fully populated and a ninth codeword is impossible. […]
    /// Both underspecified and overspecified trees are an error
    /// condition rendering the stream undecodable."
    OverspecifiedTree {
        /// The offending entry index.
        entry: u32,
        /// Its recorded codeword length.
        length: u8,
    },
    /// Kraft-equality check failed at the end of construction: the
    /// codeword lengths leave dangling capacity. §3.2.1: "in the above
    /// example, if codeword seven were eliminated, it’s clear that the
    /// tree is unfinished."
    UnderspecifiedTree,
    /// A codebook with exactly one used entry whose recorded length was
    /// not `1`. Per errata 20150226: "decoder implementations shall
    /// reject a codebook if it contains only one used entry and the
    /// encoded `[codeword_length]` of that entry is not 1."
    SingleEntryWrongLength {
        /// The single used entry's recorded length.
        length: u8,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::EmptyTree => {
                write!(f, "vorbis huffman: codebook has zero used entries (§3.2.1)")
            }
            BuildError::InvalidLength { entry, length } => write!(
                f,
                "vorbis huffman: entry {entry} has invalid codeword length {length} (must be 1..=32 per §3.2.1)"
            ),
            BuildError::OverspecifiedTree { entry, length } => write!(
                f,
                "vorbis huffman: overspecified tree — entry {entry} length {length} has no canonical codeword left (§3.2.1)"
            ),
            BuildError::UnderspecifiedTree => write!(
                f,
                "vorbis huffman: underspecified tree — codeword-length list does not fully populate the decision tree (§3.2.1)"
            ),
            BuildError::SingleEntryWrongLength { length } => write!(
                f,
                "vorbis huffman: single-entry codebook with length {length} != 1 (errata 20150226)"
            ),
        }
    }
}

impl std::error::Error for BuildError {}

/// Errors raised by [`HuffmanTree::decode_entry`] while walking the tree
/// against a packet bitstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The bit reader ran out of bits mid-codeword. Per §3.3: "reading
    /// past the end of a packet propagates the 'end-of-stream' condition
    /// to the decoder."
    UnexpectedEndOfPacket,
}

/// Errors raised by [`HuffmanTree::encode_entry`] while emitting the
/// canonical codeword for a codebook entry to a bitstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// The requested entry index is outside the tree's leaf set. A
    /// well-formed call only ever asks for an entry index that the tree
    /// has a leaf for; an out-of-range index either points past the
    /// codebook's `entries` field or names an entry whose codeword
    /// length was [`UNUSED_ENTRY`] at build time.
    UnknownEntry {
        /// The offending entry index.
        entry: u32,
        /// Number of used (leaf) entries the tree was built with.
        used_count: u32,
    },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::UnknownEntry { entry, used_count } => write!(
                f,
                "vorbis huffman: no canonical codeword for entry {entry} (tree has {used_count} used entries)"
            ),
        }
    }
}

impl std::error::Error for EncodeError {}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::UnexpectedEndOfPacket => {
                write!(f, "vorbis huffman: end-of-packet mid-codeword (§3.3)")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

impl HuffmanTree {
    /// Convenience: build the tree directly from a parsed codebook
    /// header. Delegates to [`HuffmanTree::from_lengths`] using
    /// `codebook.codeword_lengths`.
    pub fn from_codebook(codebook: &VorbisCodebook) -> Result<Self, BuildError> {
        Self::from_lengths(&codebook.codeword_lengths)
    }

    /// Build a canonical Vorbis Huffman tree from a per-entry codeword
    /// length list.
    ///
    /// Each list slot is either [`UNUSED_ENTRY`] (`0`) for a
    /// sparse-codebook unused entry, or a length in `1..=32`. Entry
    /// indices are implicit (the position in the slice).
    pub fn from_lengths(lengths: &[u8]) -> Result<Self, BuildError> {
        // Validate range + tally used entries.
        let mut used_count: u32 = 0;
        for (idx, &len) in lengths.iter().enumerate() {
            if len == UNUSED_ENTRY {
                continue;
            }
            if !(1..=32).contains(&len) {
                return Err(BuildError::InvalidLength {
                    entry: idx as u32,
                    length: len,
                });
            }
            used_count += 1;
        }
        if used_count == 0 {
            return Err(BuildError::EmptyTree);
        }

        // ---- single-entry codebook (errata 20150226) ----
        if used_count == 1 {
            // Find the sole used entry.
            let (entry, &len) = lengths
                .iter()
                .enumerate()
                .find(|(_, &l)| l != UNUSED_ENTRY)
                .expect("used_count == 1 implies a used entry");
            if len != 1 {
                return Err(BuildError::SingleEntryWrongLength { length: len });
            }
            // Build a synthetic 3-node tree: root has both children →
            // the same leaf, so reading a 0 or a 1 both return the
            // single entry while sinking exactly one bit.
            let leaf = HuffmanNode::Leaf(entry as u32);
            let nodes = vec![HuffmanNode::Internal { left: 1, right: 1 }, leaf];
            return Ok(Self {
                nodes,
                used_count: 1,
                single_entry: Some(entry as u32),
            });
        }

        // ---- general case: canonical codeword assignment ----
        //
        // §3.2.1: "the first codeword entry is assigned, in order, the
        // lowest valued unused binary Huffman codeword." That is the
        // textbook *canonical* Huffman assignment: an entry of length
        // `L` takes the numerically-lowest `L`-bit codeword that is
        // neither already used nor a prefix-extension of (nor prefixed
        // by) an already-assigned codeword.
        //
        // The earlier left-to-right "open-slot deque" build assigned the
        // leftmost *tree* position at the target depth, which only
        // coincides with the lowest-valued codeword when the codeword
        // lengths are non-decreasing. Real codebooks (e.g. the floor /
        // residue books real streams emit) interleave shorter codewords
        // after longer ones — `[2,3,3,3,3,4,3,4]` — so the two orders
        // diverge and the deque left dangling capacity, spuriously
        // reporting `UnderspecifiedTree` on a perfectly-populated book.
        //
        // We assign each codeword directly against the partially-built
        // decision tree, which is the unambiguous source of truth: the
        // lowest free `L`-bit codeword is found by descending the tree
        // from the root preferring the `0` child, treating any branch
        // that already terminates in a leaf (a used shorter prefix) or
        // is fully populated as unavailable. Codewords are MSb-first on
        // the wire (the `decode_entry` walk reads the high bit first),
        // so codeword `c` of length `L` is the root-to-leaf path whose
        // bit `i` (`i = 0..L`) is `(c >> (L - 1 - i)) & 1`.
        //
        // This is generic canonical-Huffman / entropy-coding bookkeeping
        // (the construction the §3.2.1 worked example walks by hand), not
        // specific to any implementation.

        let mut nodes: Vec<HuffmanNode> = Vec::with_capacity(used_count as usize * 2);
        nodes.push(HuffmanNode::Internal {
            left: u32::MAX,
            right: u32::MAX,
        });

        for (idx, &len) in lengths.iter().enumerate() {
            if len == UNUSED_ENTRY {
                continue;
            }

            // Find the lowest free `len`-bit codeword by descending the
            // partially-built tree, preferring the `0` (left) child at
            // every level. A child is "free to descend into" if it is an
            // existing internal node or an empty slot (`u32::MAX`); a
            // leaf child means that prefix is already taken (try the `1`
            // sibling instead). We materialise the path as we go.
            //
            // At each level we hold the current node index `cur` and the
            // bit we must take. We greedily take `0` whenever its subtree
            // still has room; otherwise `1`. "Has room" at the final
            // level means the slot is empty; at an interior level it
            // means the child is empty or an internal node (never a leaf,
            // and never a fully-saturated subtree — but a subtree filled
            // by exactly `2^(len-depth)` shorter+equal codewords cannot
            // accept a new `len`-bit leaf). We detect saturation lazily:
            // if the greedy `0`-then-`1` descent dead-ends, the tree is
            // over-populated for this length.
            let placed = place_lowest(&mut nodes, idx as u32, len);
            if !placed {
                return Err(BuildError::OverspecifiedTree {
                    entry: idx as u32,
                    length: len,
                });
            }
        }

        // Underspecified check: any internal node with a dangling child
        // (`u32::MAX`) means the length list leaves capacity unused.
        for n in &nodes {
            if let HuffmanNode::Internal { left, right } = *n {
                if left == u32::MAX || right == u32::MAX {
                    return Err(BuildError::UnderspecifiedTree);
                }
            }
        }

        Ok(Self {
            nodes,
            used_count,
            single_entry: None,
        })
    }

    /// Returns the number of used codebook entries (leaves) the tree
    /// was built from.
    #[must_use]
    pub fn used_count(&self) -> u32 {
        self.used_count
    }

    /// Returns `true` if the tree was built from a single-used-entry
    /// codebook (errata 20150226). Such a tree always returns its sole
    /// entry from [`decode_entry`] while sinking one bit.
    #[must_use]
    pub fn is_single_entry(&self) -> bool {
        self.single_entry.is_some()
    }

    /// Returns the underlying node arena. Node `0` is the root; each
    /// `Internal` node's `left` / `right` are indices into the same
    /// arena. Exposed for debugging / inspection.
    #[must_use]
    pub fn nodes(&self) -> &[HuffmanNode] {
        &self.nodes
    }

    /// Decode a single codebook entry index from an LSb-first packet
    /// bitstream.
    ///
    /// Reads bits one at a time, walking the tree from the root: a `0`
    /// bit takes the `left` child, a `1` bit takes the `right` child.
    /// Stops at the first leaf hit and returns its entry index. If the
    /// bit reader runs out of bits mid-walk, returns
    /// [`DecodeError::UnexpectedEndOfPacket`].
    pub fn decode_entry(&self, reader: &mut BitReaderLsb<'_>) -> Result<u32, DecodeError> {
        // Single-entry codebook fast path: sink one bit, return the
        // sole entry. Errata 20150226: "decoders should tolerate that
        // the bit read from the stream be '1' instead of '0'; both
        // values shall return the single used codeword."
        if let Some(entry) = self.single_entry {
            reader
                .read_bit()
                .map_err(|_| DecodeError::UnexpectedEndOfPacket)?;
            return Ok(entry);
        }

        let mut cur: u32 = 0;
        loop {
            match self.nodes[cur as usize] {
                HuffmanNode::Leaf(entry) => return Ok(entry),
                HuffmanNode::Internal { left, right } => {
                    let bit = reader
                        .read_bit()
                        .map_err(|_| DecodeError::UnexpectedEndOfPacket)?;
                    cur = if bit { right } else { left };
                }
            }
        }
    }

    /// Encode a single codebook entry index to an LSb-first packet
    /// bitstream as its canonical Huffman codeword, emitting MSb-first.
    ///
    /// This is the byte-exact inverse of [`HuffmanTree::decode_entry`]:
    /// for every entry index `e` the tree has a leaf for, calling
    /// `encode_entry(e, &mut writer)` followed by `decode_entry` on a
    /// reader over `writer.finish()` returns `e`. The first bit written
    /// is the MSb of the canonical codeword (matching §3.2.1's
    /// "leftmost bit is the MSb" convention); subsequent bits stream
    /// the lower bits of the codeword down to the LSb at codeword
    /// length `L`.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::UnknownEntry`] if `entry` does not name a
    /// leaf of the tree — either because it points past the codebook's
    /// `entries` field or because the entry's codeword length was
    /// [`UNUSED_ENTRY`] at build time. The writer is not mutated on
    /// failure (validation happens via a single tree walk that emits
    /// nothing until it has determined the codeword).
    ///
    /// # Single-entry codebook fast path
    ///
    /// Per errata 20150226 a single-entry codebook always returns its
    /// sole entry from `decode_entry` while sinking one bit. The encode
    /// counterpart emits one zero bit for the sole entry (the spec also
    /// permits a `1` for decoder tolerance, but the writer always picks
    /// `0` so the round-trip is deterministic). Any other entry index
    /// is rejected.
    pub fn encode_entry(&self, entry: u32, writer: &mut BitWriterLsb) -> Result<(), EncodeError> {
        // Single-entry codebook fast path (errata 20150226): the tree
        // has exactly one leaf reached by either bit. Emit `0` as the
        // canonical "first bit" for round-trip determinism.
        if let Some(only) = self.single_entry {
            if entry != only {
                return Err(EncodeError::UnknownEntry {
                    entry,
                    used_count: 1,
                });
            }
            writer.write_bit(false);
            return Ok(());
        }

        // Walk the tree from the root, recording the bit (0 = left,
        // 1 = right) taken at each internal node, until we hit the
        // requested leaf. The recorded `Vec<bool>` is the canonical
        // codeword for `entry`, MSb-first. Tree depth is bounded by
        // §3.2.1's 32-bit hard limit on codeword length, so the
        // recursion budget is tiny.
        let mut codeword: Vec<bool> = Vec::with_capacity(32);
        if !self.walk_to_leaf(0, entry, &mut codeword) {
            return Err(EncodeError::UnknownEntry {
                entry,
                used_count: self.used_count,
            });
        }
        // Emit the codeword bits MSb-first: walk_to_leaf appends bits
        // in root-to-leaf order, which is already MSb-first.
        for bit in &codeword {
            writer.write_bit(*bit);
        }
        Ok(())
    }

    /// DFS helper for [`Self::encode_entry`]: walk from `node_idx`
    /// looking for `Leaf(entry)`, appending the chosen bit (false for
    /// `left`, true for `right`) at each internal node and back-tracking
    /// on miss. Returns `true` iff the leaf was found.
    fn walk_to_leaf(&self, node_idx: u32, entry: u32, path: &mut Vec<bool>) -> bool {
        match self.nodes[node_idx as usize] {
            HuffmanNode::Leaf(found) => found == entry,
            HuffmanNode::Internal { left, right } => {
                path.push(false);
                if self.walk_to_leaf(left, entry, path) {
                    return true;
                }
                path.pop();
                path.push(true);
                if self.walk_to_leaf(right, entry, path) {
                    return true;
                }
                path.pop();
                false
            }
        }
    }
}

/// Mutate the `bit` (0 or 1) child of the node at `cur` in the arena.
///
/// Pulled out as a free function to side-step a borrow-checker conflict
/// when the parent loop is iterating + mutating the arena in the same
/// scope.
/// Place a leaf for `entry` at the lowest free codeword of exactly
/// `len` bits, descending from the root (node 0) and preferring the `0`
/// (left) child at every level. Returns `true` on success, `false` if
/// no `len`-bit codeword is free (the tree is over-populated for that
/// length).
///
/// "Lowest valued unused binary Huffman codeword" (§3.2.1) is realised
/// by always trying the `0` branch first and only falling back to `1`
/// when the `0` subtree can no longer host a `len`-depth leaf. A subtree
/// can host one iff some root-to-depth path through it ends in an empty
/// slot at the target depth without passing through an existing leaf.
fn place_lowest(nodes: &mut Vec<HuffmanNode>, entry: u32, len: u8) -> bool {
    place_rec(nodes, 0, len)
        .map(|leaf_slot| {
            // `leaf_slot` is `(parent, bit)` of the empty position the
            // descent chose; install the leaf there.
            let leaf_idx = nodes.len() as u32;
            nodes.push(HuffmanNode::Leaf(entry));
            set_child(nodes, leaf_slot.0, leaf_slot.1, leaf_idx);
        })
        .is_some()
}

/// Recursive descent for [`place_lowest`]. `cur` is the current internal
/// node index; `remaining` is how many more levels down the leaf must
/// sit. Returns `Some((parent, bit))` naming the empty child slot to
/// receive the leaf, materialising interior nodes along the chosen path,
/// or `None` if this subtree cannot host a leaf at the target depth.
fn place_rec(nodes: &mut Vec<HuffmanNode>, cur: u32, remaining: u8) -> Option<(u32, u8)> {
    debug_assert!(remaining >= 1);
    for bit in [0u8, 1u8] {
        let child = match nodes[cur as usize] {
            HuffmanNode::Internal { left, right } => {
                if bit == 0 {
                    left
                } else {
                    right
                }
            }
            HuffmanNode::Leaf(_) => return None,
        };
        if remaining == 1 {
            // We need an empty slot exactly here.
            if child == u32::MAX {
                return Some((cur, bit));
            }
            // Occupied (leaf or internal) → this bit is taken; try `1`.
            continue;
        }
        // remaining > 1: descend into (or create) the child subtree.
        let descend_idx = if child == u32::MAX {
            // Empty slot: create a fresh internal node, descend into it.
            let new_idx = nodes.len() as u32;
            nodes.push(HuffmanNode::Internal {
                left: u32::MAX,
                right: u32::MAX,
            });
            set_child(nodes, cur, bit, new_idx);
            new_idx
        } else {
            match nodes[child as usize] {
                HuffmanNode::Internal { .. } => child,
                // A leaf already terminates this prefix; the whole
                // subtree is unavailable for a deeper codeword.
                HuffmanNode::Leaf(_) => continue,
            }
        };
        if let Some(slot) = place_rec(nodes, descend_idx, remaining - 1) {
            return Some(slot);
        }
        // The descent into an *existing* child subtree failed (it is
        // saturated for this depth); try the `1` sibling. A freshly
        // created empty internal node can always host a leaf at any
        // remaining depth, so `child == u32::MAX` never reaches this
        // line — only a pre-existing internal subtree does.
    }
    None
}

fn set_child(nodes: &mut [HuffmanNode], cur: u32, bit: u8, child: u32) {
    if let HuffmanNode::Internal { left, right } = &mut nodes[cur as usize] {
        if bit == 0 {
            *left = child;
        } else {
            *right = child;
        }
    } else {
        // Caller guarantees `cur` is internal at this point; reaching
        // this branch indicates a bug in the construction loop.
        unreachable!("set_child called on a leaf node");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::{BitReaderLsb, BitWriterLsb};

    /// Build a packet that contains the given canonical codewords back
    /// to back. Each `(code, len)` writes `len` bits with the *MSb* of
    /// `code` going to the stream first (so when [`BitReaderLsb`] reads
    /// the first bit, it sees the MSb of the codeword — matching the
    /// §3.2.1 "leftmost bit is the MSb" convention).
    fn pack_codewords(codewords: &[(u32, u8)]) -> Vec<u8> {
        let mut w = BitWriterLsb::with_capacity(8);
        for &(code, len) in codewords {
            for bit_pos in (0..len).rev() {
                w.write_bit(((code >> bit_pos) & 1) != 0);
            }
        }
        w.finish()
    }

    /// §3.2.1 worked example: lengths `[2, 4, 4, 4, 4, 2, 3, 3]` yield
    /// canonical codewords:
    ///   entry 0: 00
    ///   entry 1: 0100
    ///   entry 2: 0101
    ///   entry 3: 0110
    ///   entry 4: 0111
    ///   entry 5: 10
    ///   entry 6: 110
    ///   entry 7: 111
    #[test]
    fn builds_spec_worked_example_tree() {
        let lengths = [2u8, 4, 4, 4, 4, 2, 3, 3];
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        assert_eq!(tree.used_count(), 8);
        assert!(!tree.is_single_entry());

        // Decode each codeword and confirm we get the expected entry.
        let cases: [(u32, u8, u32); 8] = [
            (0b00, 2, 0),
            (0b0100, 4, 1),
            (0b0101, 4, 2),
            (0b0110, 4, 3),
            (0b0111, 4, 4),
            (0b10, 2, 5),
            (0b110, 3, 6),
            (0b111, 3, 7),
        ];
        for &(code, len, expected_entry) in &cases {
            let bytes = pack_codewords(&[(code, len)]);
            let mut r = BitReaderLsb::new(&bytes);
            let got = tree.decode_entry(&mut r).expect("must decode");
            assert_eq!(
                got,
                expected_entry,
                "codeword {code:0width$b} (len {len}) should decode to entry {expected_entry}",
                width = len as usize,
            );
        }
    }

    /// Concatenating every spec-example codeword into one stream and
    /// decoding sequentially yields exactly `0..=7`.
    #[test]
    fn decodes_concatenated_worked_example_stream() {
        let lengths = [2u8, 4, 4, 4, 4, 2, 3, 3];
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        let codes: [(u32, u8); 8] = [
            (0b00, 2),
            (0b0100, 4),
            (0b0101, 4),
            (0b0110, 4),
            (0b0111, 4),
            (0b10, 2),
            (0b110, 3),
            (0b111, 3),
        ];
        let bytes = pack_codewords(&codes);
        let mut r = BitReaderLsb::new(&bytes);
        for expected in 0u32..8 {
            assert_eq!(tree.decode_entry(&mut r).unwrap(), expected);
        }
    }

    /// A sparse codebook (entry 0xFF mapping): used entries at irregular
    /// positions with [`UNUSED_ENTRY`] holes between them. Confirm the
    /// returned entry indices are the original positions (not a
    /// compacted index over used-only).
    #[test]
    fn maps_sparse_codebook_entry_indices_correctly() {
        // 6-entry sparse codebook: only entries 1, 3, 5 are used, all
        // length 2. Canonical codewords: 00, 01, 10. (The fourth length-2
        // codeword 11 is left dangling → underspecified — so add a
        // length-1 entry... no: a length-1 entry would burn 2 codewords
        // of length 2. The simplest balanced choice is four used entries
        // at length 2.)
        //
        // Use: 0xFF (255) as one of the entry indices to confirm large
        // entry numbers round-trip without truncation.
        let mut lengths = vec![UNUSED_ENTRY; 256];
        lengths[1] = 2;
        lengths[3] = 2;
        lengths[5] = 2;
        lengths[0xFF] = 2;
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        assert_eq!(tree.used_count(), 4);

        // Canonical assignment: entries placed in entry-index order, so
        //   entry 1   → 00
        //   entry 3   → 01
        //   entry 5   → 10
        //   entry 255 → 11
        let cases: [(u32, u8, u32); 4] =
            [(0b00, 2, 1), (0b01, 2, 3), (0b10, 2, 5), (0b11, 2, 0xFF)];
        for &(code, len, expected_entry) in &cases {
            let bytes = pack_codewords(&[(code, len)]);
            let mut r = BitReaderLsb::new(&bytes);
            assert_eq!(tree.decode_entry(&mut r).unwrap(), expected_entry);
        }
    }

    /// Single-entry codebook (errata 20150226): one used entry whose
    /// length is exactly 1. `decode_entry` sinks one bit and returns
    /// that entry's index, accepting both `0` and `1` per the errata
    /// (decoders shall tolerate either).
    #[test]
    fn single_entry_codebook_returns_sole_entry_for_either_bit() {
        // Sparse 4-entry codebook with only entry 2 marked used at
        // length 1.
        let lengths = [UNUSED_ENTRY, UNUSED_ENTRY, 1u8, UNUSED_ENTRY];
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        assert_eq!(tree.used_count(), 1);
        assert!(tree.is_single_entry());

        for bit_value in [0u8, 1u8] {
            let bytes = pack_codewords(&[(bit_value as u32, 1)]);
            let mut r = BitReaderLsb::new(&bytes);
            assert_eq!(tree.decode_entry(&mut r).unwrap(), 2);
            // The bit must have been sunk: one bit of input → zero
            // bits remaining (the byte still has 7 padding bits, but
            // the bit reader has read one of them).
            assert_eq!(r.bit_position(), 1);
        }
    }

    /// Single-entry codebook with an incorrect (non-1) length is
    /// rejected per the errata.
    #[test]
    fn single_entry_codebook_with_wrong_length_is_rejected() {
        // Sparse codebook: only entry 0 used, length 2.
        let lengths = [2u8, UNUSED_ENTRY];
        match HuffmanTree::from_lengths(&lengths) {
            Err(BuildError::SingleEntryWrongLength { length: 2 }) => {}
            other => panic!("expected SingleEntryWrongLength{{length:2}}, got {other:?}"),
        }
    }

    /// A codebook with zero used entries is rejected.
    #[test]
    fn fully_unused_codebook_is_rejected() {
        let lengths = [UNUSED_ENTRY; 4];
        match HuffmanTree::from_lengths(&lengths) {
            Err(BuildError::EmptyTree) => {}
            other => panic!("expected EmptyTree, got {other:?}"),
        }
    }

    /// Underspecified tree: §3.2.1 example with entry 7 eliminated —
    /// the resulting length list does not fill the canonical decision
    /// tree.
    #[test]
    fn rejects_underspecified_tree() {
        // Spec worked example minus its last length-3 entry (entry 7).
        let lengths = [2u8, 4, 4, 4, 4, 2, 3];
        match HuffmanTree::from_lengths(&lengths) {
            Err(BuildError::UnderspecifiedTree) => {}
            other => panic!("expected UnderspecifiedTree, got {other:?}"),
        }
    }

    /// Overspecified tree: every codeword fits in 2 bits (00, 01, 10,
    /// 11 → 4 entries), but we list a fifth length-2 entry. There's no
    /// canonical 5th codeword at length 2.
    #[test]
    fn rejects_overspecified_tree() {
        let lengths = [2u8, 2, 2, 2, 2];
        match HuffmanTree::from_lengths(&lengths) {
            Err(BuildError::OverspecifiedTree {
                entry: 4,
                length: 2,
            }) => {}
            other => panic!("expected OverspecifiedTree{{entry:4,length:2}}, got {other:?}"),
        }
    }

    /// Length outside `1..=32` is rejected with [`BuildError::InvalidLength`].
    #[test]
    fn rejects_invalid_length() {
        let lengths = [1u8, 33, 1];
        match HuffmanTree::from_lengths(&lengths) {
            Err(BuildError::InvalidLength {
                entry: 1,
                length: 33,
            }) => {}
            other => panic!("expected InvalidLength{{entry:1,length:33}}, got {other:?}"),
        }
    }

    /// Truncated packet (EOF mid-codeword) surfaces as
    /// [`DecodeError::UnexpectedEndOfPacket`].
    #[test]
    fn decode_truncated_packet_surfaces_eof() {
        // Build a length-4 tree (so we know we'll need at least 4 bits
        // for some codewords) and feed it an empty byte slice.
        let lengths = [2u8, 4, 4, 4, 4, 2, 3, 3];
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        let bytes: [u8; 0] = [];
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            tree.decode_entry(&mut r),
            Err(DecodeError::UnexpectedEndOfPacket)
        );

        // Half-codeword: feed 2 bits of `0110` so the reader gets a
        // start of entry 3 (`0110`), then exhausts.
        let mut w = BitWriterLsb::with_capacity(1);
        w.write_bit(false); // MSb of 0110 → bit 0 of read
        w.write_bit(true); // bit 2 of 0110 → bit 1 of read
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        // We have 2 bits in the byte (then 6 padding zero bits). Reading
        // 0, then 1 walks into the length-3/length-4 prefix branch
        // (000... is entry 0's prefix? no — entry 0 is `00` length 2;
        // we read 0, 1 — that's into the `01` prefix, an internal
        // node). Then padding `0`s carry us to entry 1 (`0100`):
        assert_eq!(tree.decode_entry(&mut r).unwrap(), 1);
    }

    /// Building from a [`VorbisCodebook`] via [`HuffmanTree::from_codebook`]
    /// matches building from the lengths slice directly.
    #[test]
    fn from_codebook_matches_from_lengths() {
        use crate::codebook::{VorbisCodebook, VqLookup};
        let lengths = vec![2u8, 4, 4, 4, 4, 2, 3, 3];
        let cb = VorbisCodebook {
            dimensions: 1,
            entries: 8,
            codeword_lengths: lengths.clone(),
            lookup: VqLookup::None,
        };
        let from_cb = HuffmanTree::from_codebook(&cb).expect("must build");
        let from_lens = HuffmanTree::from_lengths(&lengths).expect("must build");
        assert_eq!(from_cb.used_count(), from_lens.used_count());
        assert_eq!(from_cb.nodes(), from_lens.nodes());
    }

    /// A fully-balanced 16-entry codebook (every entry at length 4)
    /// roundtrips: assigned codewords are 0000..=1111, and entry `i`
    /// decodes back to `i` when its canonical codeword is fed.
    #[test]
    fn balanced_16_entry_length4_tree_roundtrips() {
        let lengths = vec![4u8; 16];
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        assert_eq!(tree.used_count(), 16);
        for entry in 0u32..16 {
            // Canonical codeword for entry i is just i, MSb-first, len 4.
            let bytes = pack_codewords(&[(entry, 4)]);
            let mut r = BitReaderLsb::new(&bytes);
            assert_eq!(tree.decode_entry(&mut r).unwrap(), entry);
        }
    }

    /// A maximum-length entry (length 32) is honoured: build a tree with
    /// one length-1 entry, one length-2 entry, ..., and finally one
    /// length-32 entry plus a second length-32 entry that fills the
    /// tree (the Kraft sum is `1/2 + 1/4 + ... + 1/2^31 + 2/2^32 = 1`).
    #[test]
    fn handles_max_length_32_entries() {
        // lengths: [1, 2, 3, ..., 31, 32, 32]
        let mut lengths: Vec<u8> = (1..=31).collect();
        lengths.push(32);
        lengths.push(32);
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        assert_eq!(tree.used_count(), 33);
        // Sanity: first entry (length 1, codeword "0") decodes.
        let bytes = pack_codewords(&[(0, 1)]);
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(tree.decode_entry(&mut r).unwrap(), 0);
    }

    // ----------------------------------------------------------------
    // encode_entry tests (round 250) — encoder-side canonical codeword
    // emission, the byte-exact inverse of decode_entry.
    // ----------------------------------------------------------------

    /// §3.2.1 worked example: encoding each entry emits exactly the
    /// canonical codeword bits the spec lists for it, MSb-first.
    #[test]
    fn encodes_spec_worked_example_codewords() {
        let lengths = [2u8, 4, 4, 4, 4, 2, 3, 3];
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        let expected: [(u32, u8); 8] = [
            (0b00, 2),
            (0b0100, 4),
            (0b0101, 4),
            (0b0110, 4),
            (0b0111, 4),
            (0b10, 2),
            (0b110, 3),
            (0b111, 3),
        ];
        for (entry, &(code, len)) in expected.iter().enumerate() {
            let mut w = BitWriterLsb::with_capacity(2);
            tree.encode_entry(entry as u32, &mut w)
                .expect("encode must succeed");
            let bytes = w.finish();
            let expected_bytes = pack_codewords(&[(code, len)]);
            assert_eq!(
                bytes,
                expected_bytes,
                "entry {entry} should emit codeword {code:0width$b}",
                width = len as usize,
            );
        }
    }

    /// Concatenated encode → decode roundtrip across the spec example.
    /// Encodes every entry into one buffer and decodes back; recovers
    /// the original 0..8 sequence.
    #[test]
    fn concatenated_encode_decode_round_trips_worked_example() {
        let lengths = [2u8, 4, 4, 4, 4, 2, 3, 3];
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        let mut w = BitWriterLsb::with_capacity(8);
        for entry in 0u32..8 {
            tree.encode_entry(entry, &mut w).unwrap();
        }
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        for expected in 0u32..8 {
            assert_eq!(tree.decode_entry(&mut r).unwrap(), expected);
        }
    }

    /// Balanced 16-entry length-4 tree: every entry encodes to its own
    /// codeword bits (i for entry i, length 4).
    #[test]
    fn balanced_16_entry_length4_tree_encode_decode_roundtrips() {
        let lengths = vec![4u8; 16];
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        for entry in 0u32..16 {
            let mut w = BitWriterLsb::with_capacity(1);
            tree.encode_entry(entry, &mut w).unwrap();
            let bytes = w.finish();
            // Canonical codeword for entry i is just i, MSb-first, len 4.
            let expected = pack_codewords(&[(entry, 4)]);
            assert_eq!(bytes, expected, "entry {entry} canonical codeword");
            // And decode round-trips.
            let mut r = BitReaderLsb::new(&bytes);
            assert_eq!(tree.decode_entry(&mut r).unwrap(), entry);
        }
    }

    /// A sparse codebook (only entries 0, 2, 4, 6 used at length 2)
    /// encodes only those entries; other indices return
    /// [`EncodeError::UnknownEntry`].
    #[test]
    fn sparse_codebook_encode_rejects_unused_entries() {
        // 8-slot length table with only entries 0, 2, 4, 6 used at
        // length 2 each (Kraft 4/4 = 1, so the tree is well-formed).
        let lengths = [2u8, 0, 2, 0, 2, 0, 2, 0];
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        assert_eq!(tree.used_count(), 4);
        // Used entries (0, 2, 4, 6) encode successfully.
        for &used in &[0u32, 2, 4, 6] {
            let mut w = BitWriterLsb::with_capacity(1);
            tree.encode_entry(used, &mut w).expect("used entry encodes");
        }
        // Unused entries are rejected.
        for &unused in &[1u32, 3, 5, 7, 8, 100] {
            let mut w = BitWriterLsb::with_capacity(1);
            let err = tree
                .encode_entry(unused, &mut w)
                .expect_err("unused entry rejected");
            match err {
                EncodeError::UnknownEntry {
                    entry,
                    used_count: 4,
                } => assert_eq!(entry, unused),
                _ => panic!("unexpected error: {err:?}"),
            }
        }
    }

    /// Single-entry codebook (errata 20150226): encode emits one `0`
    /// bit; any non-sole entry is rejected.
    #[test]
    fn single_entry_codebook_encode_emits_one_zero_bit() {
        // 4-slot table with only entry 2 used at length 1.
        let lengths = [0u8, 0, 1, 0];
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        assert!(tree.is_single_entry());
        let mut w = BitWriterLsb::with_capacity(1);
        tree.encode_entry(2, &mut w).expect("sole entry encodes");
        let bytes = w.finish();
        assert_eq!(bytes, vec![0u8]);
        // Decoder tolerates both 0 and 1; verify our `0` round-trips.
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(tree.decode_entry(&mut r).unwrap(), 2);
        // Other entries rejected.
        let mut w = BitWriterLsb::with_capacity(1);
        assert_eq!(
            tree.encode_entry(0, &mut w).unwrap_err(),
            EncodeError::UnknownEntry {
                entry: 0,
                used_count: 1
            }
        );
    }

    /// Generic encode-decode roundtrip on a hand-built mixed-length tree.
    /// Lengths `[2,2,3,3,3,3]` form a complete tree (Kraft 2/4 + 4/8 = 1).
    #[test]
    fn encode_then_decode_recovers_every_entry() {
        let lengths = [2u8, 2, 3, 3, 3, 3];
        let tree = HuffmanTree::from_lengths(&lengths).expect("must build");
        for entry in 0u32..6 {
            let mut w = BitWriterLsb::with_capacity(1);
            tree.encode_entry(entry, &mut w).unwrap();
            let bytes = w.finish();
            let mut r = BitReaderLsb::new(&bytes);
            assert_eq!(tree.decode_entry(&mut r).unwrap(), entry);
        }
    }

    /// `EncodeError::UnknownEntry` Display contains both numbers.
    #[test]
    fn encode_error_display_contains_numbers() {
        let err = EncodeError::UnknownEntry {
            entry: 42,
            used_count: 8,
        };
        let s = format!("{err}");
        assert!(s.contains("42"), "Display should contain entry index: {s}");
        assert!(s.contains('8'), "Display should contain used count: {s}");
    }
}

#[cfg(test)]
mod r338_regression {
    use super::*;
    use oxideav_core::bits::{BitReaderLsb, BitWriterLsb};

    /// Regression: non-monotonic codeword lengths must still build a
    /// fully-populated canonical tree. The previous left-to-right
    /// open-slot deque assigned the leftmost *tree* position at the
    /// target depth, which diverges from the §3.2.1 "lowest valued
    /// unused codeword" rule once lengths are not non-decreasing — it
    /// spuriously reported `UnderspecifiedTree` on these books (the
    /// floor / residue class books real streams routinely carry).
    ///
    /// Lengths `[2,3,3,3,3,4,3,4]` (Kraft sum = 1/4 + 5·1/8 + 2·1/16 = 1,
    /// exactly populated). Canonical assignment in entry order:
    ///   e0 len2 → 00, e1 → 010, e2 → 011, e3 → 100, e4 → 101,
    ///   e5 len4 → 1100, e6 len3 → 111, e7 len4 → 1101.
    #[test]
    fn builds_non_monotonic_length_book() {
        let lengths = [2u8, 3, 3, 3, 3, 4, 3, 4];
        let tree = HuffmanTree::from_lengths(&lengths).expect("fully populated book builds");
        assert_eq!(tree.used_count(), 8);

        // Pin the canonical codewords by encoding each entry and reading
        // the emitted bits back MSb-first.
        let expected: [(u32, u32, u8); 8] = [
            (0, 0b00, 2),
            (1, 0b010, 3),
            (2, 0b011, 3),
            (3, 0b100, 3),
            (4, 0b101, 3),
            (5, 0b1100, 4),
            (6, 0b111, 3),
            (7, 0b1101, 4),
        ];
        for (entry, code, len) in expected {
            let mut w = BitWriterLsb::with_capacity(1);
            tree.encode_entry(entry, &mut w).expect("entry encodes");
            let bytes = w.finish();
            // Re-decode the emitted codeword to confirm it round-trips to
            // the same entry, and check the bit pattern matches `code`.
            let mut r = BitReaderLsb::new(&bytes);
            assert_eq!(
                tree.decode_entry(&mut r).unwrap(),
                entry,
                "entry {entry} round-trips"
            );
            // Verify the on-wire bits equal the canonical codeword.
            let mut r2 = BitReaderLsb::new(&bytes);
            let mut got = 0u32;
            for _ in 0..len {
                got = (got << 1) | r2.read_bit().unwrap() as u32;
            }
            assert_eq!(
                got,
                code,
                "entry {entry} codeword {got:0width$b} != expected {code:0width$b}",
                width = len as usize
            );
        }
    }

    /// Every entry of a non-monotonic book round-trips through a
    /// concatenated stream — exercises the decode walk on the tree built
    /// from out-of-order lengths.
    #[test]
    fn non_monotonic_book_stream_round_trips() {
        let lengths = [3u8, 2, 4, 3, 4, 2, 3]; // Kraft: 2·1/4 + 3·1/8 + 2·1/16 = 1
        let tree = HuffmanTree::from_lengths(&lengths).expect("builds");
        let mut w = BitWriterLsb::with_capacity(4);
        for e in 0u32..7 {
            tree.encode_entry(e, &mut w).expect("encode");
        }
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        for e in 0u32..7 {
            assert_eq!(tree.decode_entry(&mut r).unwrap(), e, "entry {e}");
        }
    }
}

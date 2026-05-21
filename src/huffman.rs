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
//! Construction maintains a deque of "open" tree positions in
//! left-to-right order. Each used entry pops the leftmost open position
//! and either places a leaf there (if it sits at the entry's depth) or
//! splits it down to that depth — allocating a fresh internal node and
//! pushing the new node's two children to the *front* of the deque so
//! they remain leftmost. The deque-front-pop order matches the spec's
//! "lowest valued unused canonical codeword" assignment one-to-one,
//! so no codeword integers ever need to be materialised at build time.
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
//! * **Overspecified** — the deque empties before every used entry
//!   has been placed; no open position remains to host the next leaf.
//!   Surfaces as [`BuildError::OverspecifiedTree`].
//! * **Underspecified** — the deque still has open positions after
//!   every used entry is placed: the codeword-length list does not
//!   fully populate the decision tree. Surfaces as
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

use oxideav_core::bits::BitReaderLsb;

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

        // ---- general case: canonical-assignment greedy build ----
        //
        // We maintain a flat node arena `nodes` (root is node 0) plus a
        // single deque of "open" leaf positions, in left-to-right tree
        // order. Each open position carries (parent_idx, bit, depth):
        // the parent arena index whose `left` (bit=0) or `right`
        // (bit=1) pointer is the open position, and the depth of that
        // pointer below the root.
        //
        // Initially the root contributes two open positions, both at
        // depth 1: (root, bit=0) and (root, bit=1).
        //
        // For each used entry of length L:
        //   1. Pop the front of the deque (the leftmost open
        //      position). Call its depth D.
        //   2. While D < L, split it: allocate a fresh internal node at
        //      the popped position, push its two child positions to
        //      the *front* of the deque (so they remain leftmost), and
        //      pop the new front. D ← D + 1.
        //   3. When D == L, place a `Leaf(entry)` at the popped
        //      position and wire it into the parent's child slot.
        //   4. If the deque empties before D == L, the tree is
        //      overspecified.
        //
        // After all entries are placed, any remaining open position
        // means the tree is underspecified.
        //
        // This realises the spec's canonical "lowest-valued unused
        // codeword" without materialising codeword integers, because
        // a deque popped front-to-back yields positions in the same
        // left-to-right order in which canonical codewords are
        // assigned. The §3.2.1 worked example follows directly:
        // entry 0 at depth 2 picks `00`, entry 5 at depth 2 (after
        // entries 1-4 burn the `01xx` subtree) picks `10`, etc.

        #[derive(Clone, Copy)]
        struct OpenSlot {
            parent: u32,
            // 0 = parent's `left`, 1 = parent's `right`.
            bit: u8,
            depth: u8,
        }

        let mut nodes: Vec<HuffmanNode> = Vec::with_capacity(used_count as usize * 2);
        nodes.push(HuffmanNode::Internal {
            left: u32::MAX,
            right: u32::MAX,
        });

        // VecDeque-like usage via Vec — we push to front (`insert(0,..)`)
        // and pop from front (`remove(0)`). The deque depth is bounded
        // by ~2 * max_codeword_length (at most ~64), so the linear
        // shifts are cheap; we'd reach for `VecDeque` if profiling
        // showed it mattering.
        let mut deque: Vec<OpenSlot> = Vec::with_capacity(64);
        deque.push(OpenSlot {
            parent: 0,
            bit: 0,
            depth: 1,
        });
        deque.push(OpenSlot {
            parent: 0,
            bit: 1,
            depth: 1,
        });

        for (idx, &len) in lengths.iter().enumerate() {
            if len == UNUSED_ENTRY {
                continue;
            }

            // Pop the leftmost open position; bail if empty.
            if deque.is_empty() {
                return Err(BuildError::OverspecifiedTree {
                    entry: idx as u32,
                    length: len,
                });
            }
            let mut slot = deque.remove(0);

            // Walk down by splitting until we reach the target depth.
            while slot.depth < len {
                // Allocate a fresh internal node at this position.
                let new_idx = nodes.len() as u32;
                nodes.push(HuffmanNode::Internal {
                    left: u32::MAX,
                    right: u32::MAX,
                });
                set_child(&mut nodes, slot.parent, slot.bit, new_idx);
                // Push the new node's right child to the deque front
                // first, then the left child — `insert(0, ..)` pushes
                // to front, so the left ends up *before* the right
                // (i.e. left is the new leftmost).
                deque.insert(
                    0,
                    OpenSlot {
                        parent: new_idx,
                        bit: 1,
                        depth: slot.depth + 1,
                    },
                );
                deque.insert(
                    0,
                    OpenSlot {
                        parent: new_idx,
                        bit: 0,
                        depth: slot.depth + 1,
                    },
                );
                slot = deque.remove(0);
            }

            // slot.depth == len; place the leaf.
            let leaf_idx = nodes.len() as u32;
            nodes.push(HuffmanNode::Leaf(idx as u32));
            set_child(&mut nodes, slot.parent, slot.bit, leaf_idx);
        }

        // Underspecified check: any leftover open position means
        // dangling capacity.
        if !deque.is_empty() {
            return Err(BuildError::UnderspecifiedTree);
        }

        // Defensive: every internal node now has both children
        // populated.
        for n in &nodes {
            if let HuffmanNode::Internal { left, right } = *n {
                debug_assert_ne!(left, u32::MAX);
                debug_assert_ne!(right, u32::MAX);
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
}

/// Mutate the `bit` (0 or 1) child of the node at `cur` in the arena.
///
/// Pulled out as a free function to side-step a borrow-checker conflict
/// when the parent loop is iterating + mutating the arena in the same
/// scope.
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
}

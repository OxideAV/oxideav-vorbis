//! Vorbis I per-packet residue decode (Vorbis I §8.6.2 "packet decode",
//! §8.6.3 "format 0 specifics", §8.6.4 "format 1 specifics", §8.6.5
//! "format 2 specifics").
//!
//! A *residue* encodes the per-channel spectral residual that, summed
//! with the floor envelope, yields the pre-IMDCT spectrum. The residue
//! header (§8.6.1) is parsed at setup time by [`crate::setup`]; this
//! module performs the runtime per-packet decode that turns the residue
//! payload of an audio packet into one float vector per channel.
//!
//! # The common decode infrastructure (§8.6.2)
//!
//! All three residue formats share the same skeleton. The decoder is
//! handed the number of vectors in the submap bundle (`ch`) and a
//! "do not decode" flag per vector. Even `do not decode` vectors are
//! allocated and zeroed.
//!
//! 1. Limit `[residue_begin]` / `[residue_end]` to the maximum vector
//!    size. For format 0/1 the cap is `blocksize/2`; for format 2 the
//!    cap is `blocksize/2 * ch` because format 2 works on a single
//!    interleaved vector (§8.6.2 steps 1..5).
//! 2. Derive the convenience values: `classwords_per_codeword =
//!    classbook.dimensions`, `n_to_read = limit_residue_end -
//!    limit_residue_begin`, `partitions_to_read = n_to_read /
//!    partition_size` (§8.6.2 step list).
//! 3. If `n_to_read == 0`, there is no residue to decode — return the
//!    zeroed vectors.
//! 4. Otherwise iterate `[pass]` over `0 ... 7`. On pass 0, read the
//!    partition classifications from the classbook in scalar context
//!    (one classbook read encodes `classwords_per_codeword`
//!    classifications, unpacked by repeated integer-divide /
//!    integer-modulo by `residue_classifications`). On every pass, walk
//!    each partition: look up its classification, find the stage-`pass`
//!    codebook from `[residue_books]`, and if that book is not
//!    `unused`, decode the partition into the output vector in VQ
//!    context (§8.6.2 step list).
//!
//! # End-of-packet handling
//!
//! Per §8.6.2: "An end-of-packet condition during packet decode is to
//! be considered a nominal occurrence. Decode returns the result of
//! vector decode up to that point." So unlike header decode (where EOF
//! is fatal), residue packet decode that runs the bit reader dry simply
//! stops and returns the partially-accumulated vectors. This is
//! signalled internally by the classbook / VQ codebook walk returning
//! [`crate::huffman::DecodeError::UnexpectedEndOfPacket`]; this module
//! catches it and returns the work-so-far.
//!
//! # Format specifics
//!
//! * **Format 0 (§8.6.3).** A partition of `n` scalars is decoded by
//!   reading `n / codebook.dimensions` VQ vectors and scattering each
//!   vector's `j`-th element to `offset + i + j*step` (interleaved
//!   layout).
//! * **Format 1 (§8.6.4).** A partition of `n` scalars is decoded by
//!   reading VQ vectors back-to-back, appending each vector's elements
//!   contiguously from `offset` until `n` scalars have been written
//!   (contiguous layout).
//! * **Format 2 (§8.6.5).** Reducible to format 1: all channels are
//!   interleaved into one virtual vector of length `ch * (blocksize/2)`,
//!   that vector is format-1 decoded, then de-interleaved back into the
//!   per-channel vectors (`output[j][i] = v[i*ch + j]`). If *every*
//!   vector is marked `do not decode`, no decode runs and zeroed
//!   vectors are returned.

use crate::codebook::{VorbisCodebook, VqLookup};
use crate::huffman::{BuildError, DecodeError, HuffmanTree};
use crate::setup::ResidueHeader;
use crate::vq::{unpack_vector, UnpackError};
use oxideav_core::bits::BitReaderLsb;

/// Errors that can arise while preparing or running a residue decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidueError {
    /// `residue_type` was a value other than 0, 1, or 2. §8.6 only
    /// defines those three formats; the setup parser already rejects
    /// `> 2`, but [`ResidueDecoder::new`] re-checks because it may be
    /// constructed from a hand-built [`ResidueHeader`].
    UnsupportedFormat(u16),
    /// The `residue_classbook` index points outside the supplied
    /// codebook table. §8.6.1: "any codebook number greater than the
    /// maximum numbered codebook set up in this stream … renders the
    /// stream undecodable."
    ClassbookOutOfRange {
        /// The offending `residue_classbook` index.
        classbook: u8,
        /// The number of codebooks available.
        codebook_count: usize,
    },
    /// A `residue_books[class][stage]` index points outside the supplied
    /// codebook table (§8.6.1, same clause as
    /// [`Self::ClassbookOutOfRange`]).
    ValueBookOutOfRange {
        /// Classification index.
        class: usize,
        /// Cascade stage index `0..=7`.
        stage: usize,
        /// The offending codebook index.
        book: u8,
        /// The number of codebooks available.
        codebook_count: usize,
    },
    /// The residue classbook's `dimensions` is zero. §8.6.2 derives
    /// `classwords_per_codeword = classbook.dimensions`; a zero value
    /// would make the classification loop read no classifications yet
    /// still consume a classbook codeword, which the spec's
    /// "overdetermines" clause never contemplates. Treat it as
    /// undecodable.
    ZeroClasswordsPerCodeword,
    /// A value codebook used in VQ context has `lookup_type = 0`
    /// (entropy-only) and so cannot yield a vector. §8.6.1: "All
    /// codebooks in array [residue books] are required to have a value
    /// mapping. The presence of codebook … without a value mapping
    /// (maptype equals zero) renders the stream undecodable."
    ValueBookHasNoLookup {
        /// Classification index.
        class: usize,
        /// Cascade stage index `0..=7`.
        stage: usize,
        /// The offending codebook index.
        book: u8,
    },
    /// A residue value codebook's `dimensions` does not divide its
    /// partition's element count for a format-0 decode (§8.6.3 derives
    /// `step = n / codebook.dimensions`, an exact integer division). A
    /// non-dividing pairing is malformed.
    Format0PartitionNotDivisible {
        /// `residue_partition_size`.
        partition_size: u32,
        /// Value codebook `dimensions`.
        dimensions: u16,
    },
    /// Building a Huffman tree for the classbook or a value book failed
    /// (§3.2.1). Carries the underlying [`BuildError`].
    Huffman(BuildError),
    /// A VQ vector unpack failed (§3.2.1 / §3.3). Carries the
    /// underlying [`UnpackError`].
    Vq(UnpackError),
}

impl core::fmt::Display for ResidueError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ResidueError::UnsupportedFormat(t) => {
                write!(f, "vorbis residue: unsupported residue_type {t} (§8.6)")
            }
            ResidueError::ClassbookOutOfRange {
                classbook,
                codebook_count,
            } => write!(
                f,
                "vorbis residue: classbook index {classbook} >= codebook count {codebook_count} (§8.6.1)"
            ),
            ResidueError::ValueBookOutOfRange {
                class,
                stage,
                book,
                codebook_count,
            } => write!(
                f,
                "vorbis residue: residue_books[{class}][{stage}] = {book} >= codebook count {codebook_count} (§8.6.1)"
            ),
            ResidueError::ZeroClasswordsPerCodeword => write!(
                f,
                "vorbis residue: classbook dimensions = 0 (§8.6.2)"
            ),
            ResidueError::ValueBookHasNoLookup {
                class,
                stage,
                book,
            } => write!(
                f,
                "vorbis residue: residue_books[{class}][{stage}] = {book} has lookup_type 0 (§8.6.1)"
            ),
            ResidueError::Format0PartitionNotDivisible {
                partition_size,
                dimensions,
            } => write!(
                f,
                "vorbis residue: format-0 partition_size {partition_size} not divisible by codebook dimensions {dimensions} (§8.6.3)"
            ),
            ResidueError::Huffman(e) => write!(f, "vorbis residue: {e}"),
            ResidueError::Vq(e) => write!(f, "vorbis residue: {e}"),
        }
    }
}

impl std::error::Error for ResidueError {}

impl From<BuildError> for ResidueError {
    fn from(value: BuildError) -> Self {
        ResidueError::Huffman(value)
    }
}

impl From<UnpackError> for ResidueError {
    fn from(value: UnpackError) -> Self {
        ResidueError::Vq(value)
    }
}

/// A residue value codebook prepared for runtime VQ decode: its Huffman
/// decision tree plus a clone of the codebook itself (so `unpack_vector`
/// can read the multiplicand table during decode).
#[derive(Debug)]
struct ValueBook {
    tree: HuffmanTree,
    codebook: VorbisCodebook,
}

/// A residue decoder built from a [`ResidueHeader`] and the stream's
/// codebook table (Vorbis I §8.6).
///
/// Construction (`ResidueDecoder::new`) validates the header against the
/// codebook table (§8.6.1's undecodability clauses) and pre-builds the
/// classbook + value-book Huffman trees once, so [`ResidueDecoder::decode`]
/// can be called per audio packet without re-parsing setup data.
#[derive(Debug)]
pub struct ResidueDecoder {
    /// `residue_type` (0, 1, or 2).
    residue_type: u16,
    /// `residue_begin` (header field, pre-limit).
    residue_begin: u32,
    /// `residue_end` (header field, pre-limit).
    residue_end: u32,
    /// `residue_partition_size` (already `+1`-decoded by the parser).
    partition_size: u32,
    /// `residue_classifications` (1..=64).
    classifications: u32,
    /// `classwords_per_codeword` = classbook dimensions (§8.6.2).
    classwords_per_codeword: u32,
    /// The classbook's Huffman decision tree (used in scalar context).
    classbook_tree: HuffmanTree,
    /// `residue_books[class][stage]`: `Some(ValueBook)` where the cascade
    /// bit was set, `None` otherwise. Outer length =
    /// `classifications`, inner length = 8.
    value_books: Vec<[Option<ValueBook>; 8]>,
}

impl ResidueDecoder {
    /// Build a residue decoder from a parsed [`ResidueHeader`] and the
    /// stream's full codebook table.
    ///
    /// `codebooks` is the slice of codebooks parsed from the setup
    /// header (in stream order); `residue_classbook` and every entry of
    /// `residue_books` index into it. Returns a structured
    /// [`ResidueError`] for any §8.6.1 undecodability condition or any
    /// Huffman build failure.
    pub fn new(header: &ResidueHeader, codebooks: &[VorbisCodebook]) -> Result<Self, ResidueError> {
        if header.residue_type > 2 {
            return Err(ResidueError::UnsupportedFormat(header.residue_type));
        }

        // §8.6.1: classbook must index a real codebook.
        let classbook_idx = header.classbook as usize;
        let classbook = codebooks.get(classbook_idx).ok_or({
            ResidueError::ClassbookOutOfRange {
                classbook: header.classbook,
                codebook_count: codebooks.len(),
            }
        })?;
        let classwords_per_codeword = classbook.dimensions as u32;
        if classwords_per_codeword == 0 {
            return Err(ResidueError::ZeroClasswordsPerCodeword);
        }
        let classbook_tree = HuffmanTree::from_codebook(classbook)?;

        // §8.6.1: build the per-(class, stage) value books, validating
        // that each referenced codebook exists and has a value mapping.
        let classifications = header.classifications as usize;
        let mut value_books: Vec<[Option<ValueBook>; 8]> = Vec::with_capacity(classifications);
        for (class, stage_books) in header.books.iter().take(classifications).enumerate() {
            let mut row: [Option<ValueBook>; 8] = Default::default();
            for (stage, slot) in stage_books.iter().enumerate() {
                let Some(book_idx) = slot else { continue };
                let cb_idx = *book_idx as usize;
                let codebook = codebooks.get(cb_idx).ok_or({
                    ResidueError::ValueBookOutOfRange {
                        class,
                        stage,
                        book: *book_idx,
                        codebook_count: codebooks.len(),
                    }
                })?;
                // §8.6.1: the codebook must have a value mapping
                // (lookup_type != 0), since it is used in VQ context.
                if matches!(codebook.lookup, VqLookup::None) {
                    return Err(ResidueError::ValueBookHasNoLookup {
                        class,
                        stage,
                        book: *book_idx,
                    });
                }
                let tree = HuffmanTree::from_codebook(codebook)?;
                row[stage] = Some(ValueBook {
                    tree,
                    codebook: codebook.clone(),
                });
            }
            value_books.push(row);
        }
        // Pad in case `header.books` is shorter than `classifications`
        // (a malformed header); a missing row contributes all-`None`
        // stages, which the decode loop treats as `unused`.
        while value_books.len() < classifications {
            value_books.push(Default::default());
        }

        Ok(Self {
            residue_type: header.residue_type,
            residue_begin: header.residue_begin,
            residue_end: header.residue_end,
            partition_size: header.partition_size,
            classifications: header.classifications as u32,
            classwords_per_codeword,
            classbook_tree,
            value_books,
        })
    }

    /// Decode the residue payload of one audio packet into `ch` float
    /// vectors (Vorbis I §8.6.2 + the format-specific §8.6.3/4/5).
    ///
    /// * `reader` is positioned at the start of this residue's payload
    ///   within the audio packet.
    /// * `blocksize` is the *current packet's* block size (the IMDCT
    ///   length `N`; vectors hold `N/2` spectral coefficients).
    /// * `do_not_decode[j] == true` marks output vector `j` as not to be
    ///   decoded; it is still allocated and zeroed (§8.6.2). `ch` is
    ///   `do_not_decode.len()`.
    ///
    /// Returns one `Vec<f32>` per channel, each of length `blocksize/2`.
    /// An end-of-packet during decode is nominal (§8.6.2): the routine
    /// stops and returns the vectors decoded up to that point.
    pub fn decode(
        &self,
        reader: &mut BitReaderLsb<'_>,
        blocksize: usize,
        do_not_decode: &[bool],
    ) -> Result<Vec<Vec<f32>>, ResidueError> {
        let ch = do_not_decode.len();
        let per_channel_size = blocksize / 2;

        // §8.6.2 step 1: allocate and zero all returned vectors.
        let mut output: Vec<Vec<f32>> = vec![vec![0.0f32; per_channel_size]; ch];

        if self.residue_type == 2 {
            self.decode_format2(reader, per_channel_size, do_not_decode, &mut output)?;
        } else {
            self.decode_format01(reader, per_channel_size, do_not_decode, &mut output)?;
        }
        Ok(output)
    }

    /// §8.6.2 + §8.6.3/§8.6.4 — the format 0 / format 1 path, decoding
    /// `ch` independent per-channel vectors directly into `output`.
    fn decode_format01(
        &self,
        reader: &mut BitReaderLsb<'_>,
        per_channel_size: usize,
        do_not_decode: &[bool],
        output: &mut [Vec<f32>],
    ) -> Result<(), ResidueError> {
        // §8.6.2 steps 1..5: limit begin/end to the actual vector size.
        // For format 0/1 the cap is `blocksize/2` (per_channel_size).
        let actual_size = per_channel_size as u32;
        let limit_begin = self.residue_begin.min(actual_size);
        let limit_end = self.residue_end.min(actual_size);

        self.decode_core(reader, limit_begin, limit_end, do_not_decode, output)
    }

    /// §8.6.5 — format 2: interleave all `ch` channels into one virtual
    /// vector of length `ch * per_channel_size`, format-1 decode it, then
    /// de-interleave back into `output`.
    fn decode_format2(
        &self,
        reader: &mut BitReaderLsb<'_>,
        per_channel_size: usize,
        do_not_decode: &[bool],
        output: &mut [Vec<f32>],
    ) -> Result<(), ResidueError> {
        let ch = do_not_decode.len();

        // §8.6.5 step 1: if every vector is 'do not decode', no decode
        // occurs — `output` is already zeroed, nothing to do.
        if do_not_decode.iter().all(|&dnd| dnd) {
            return Ok(());
        }

        // §8.6.2 steps 1..5: for format 2, the cap is `blocksize/2 * ch`
        // because decode operates on the single interleaved vector.
        let actual_size = (per_channel_size as u32).saturating_mul(ch as u32);
        let limit_begin = self.residue_begin.min(actual_size);
        let limit_end = self.residue_end.min(actual_size);

        // §8.6.5 step 2: decode a single vector of length `ch * n`.
        // The interleaved decode treats this as a single-channel format-1
        // decode (one output vector, never 'do not decode' since at least
        // one real channel is decoded).
        let mut interleaved = [vec![0.0f32; per_channel_size * ch]];
        let single_dnd = [false];
        self.decode_core(
            reader,
            limit_begin,
            limit_end,
            &single_dnd,
            &mut interleaved,
        )?;

        // §8.6.5 step 3: de-interleave `v[i*ch + j]` -> output[j][i].
        let interleaved = &interleaved[0];
        for i in 0..per_channel_size {
            for (j, out) in output.iter_mut().enumerate() {
                out[i] = interleaved[i * ch + j];
            }
        }
        Ok(())
    }

    /// §8.6.2 packet-decode core, shared by format 0/1 (called with one
    /// per-channel vector each) and format 2 (called with a single
    /// interleaved vector). `vectors[j]` is decode vector `j`; its length
    /// is `do_not_decode.len()`.
    fn decode_core(
        &self,
        reader: &mut BitReaderLsb<'_>,
        limit_begin: u32,
        limit_end: u32,
        do_not_decode: &[bool],
        vectors: &mut [Vec<f32>],
    ) -> Result<(), ResidueError> {
        let vch = vectors.len();
        // §8.6.2 step list: convenience values.
        let n_to_read = limit_end.saturating_sub(limit_begin);
        // §8.6.2 step 2: nothing to decode.
        if n_to_read == 0 {
            return Ok(());
        }
        let partitions_to_read = (n_to_read / self.partition_size) as usize;
        if partitions_to_read == 0 {
            return Ok(());
        }
        let classwords = self.classwords_per_codeword as usize;

        // `classifications[j][partition]` — only the entries actually
        // referenced (`< partitions_to_read`) are populated. Stored as
        // one flat row per vector for cache friendliness.
        let mut classifications = vec![vec![0u32; partitions_to_read]; vch];

        // §8.6.2 step 3: iterate passes 0..=7.
        for pass in 0..8usize {
            // §8.6.2 step 5: process every partition.
            let mut partition_count = 0usize;
            while partition_count < partitions_to_read {
                // §8.6.2 step 6: on pass 0, refill classifications for
                // the next `classwords` partitions.
                if pass == 0 {
                    // §8.6.2 step 7: iterate over the vectors.
                    for (j, dnd) in do_not_decode.iter().enumerate() {
                        if *dnd {
                            continue;
                        }
                        // §8.6.2 step 9: read classbook in scalar context.
                        let temp = match self.classbook_tree.decode_entry(reader) {
                            Ok(t) => t,
                            // §8.6.2: EOF is nominal — stop and keep work.
                            Err(DecodeError::UnexpectedEndOfPacket) => return Ok(()),
                        };
                        // §8.6.2 steps 10..12: unpack `classwords`
                        // classifications, descending.
                        let mut temp = temp;
                        for i in (0..classwords).rev() {
                            let slot = i + partition_count;
                            if slot < partitions_to_read {
                                classifications[j][slot] = temp % self.classifications;
                            }
                            temp /= self.classifications;
                        }
                    }
                }

                // §8.6.2 step 13: decode this group of partitions.
                let mut i = 0usize;
                while i < classwords && partition_count < partitions_to_read {
                    // §8.6.2 step 14: iterate over the vectors.
                    for (j, dnd) in do_not_decode.iter().enumerate() {
                        if *dnd {
                            continue;
                        }
                        // §8.6.2 step 16: this partition's classification.
                        let vqclass = classifications[j][partition_count] as usize;
                        // §8.6.2 step 17: the stage-`pass` value codebook.
                        let vqbook = self.value_books[vqclass][pass].as_ref();
                        // §8.6.2 step 18: skip 'unused' stages.
                        if let Some(book) = vqbook {
                            // §8.6.2 step 19: decode partition into output
                            // vector `j` at offset
                            // limit_begin + partition_count*partition_size.
                            let offset = (limit_begin as usize)
                                + partition_count * self.partition_size as usize;
                            let out = &mut vectors[j];
                            match self.decode_partition(reader, book, out, offset) {
                                Ok(()) => {}
                                // §8.6.2: EOF is nominal — stop, keep work.
                                Err(PartitionStop::Eof) => return Ok(()),
                                Err(PartitionStop::Err(e)) => return Err(e),
                            }
                        }
                    }
                    // §8.6.2 step 20: advance.
                    partition_count += 1;
                    i += 1;
                }
            }
        }
        Ok(())
    }

    /// Decode one partition of `residue_partition_size` scalars into
    /// `v` starting at `offset`, dispatching on the residue format.
    ///
    /// `residue_type` 0 → §8.6.3, `residue_type` 1 or 2 → §8.6.4 (format
    /// 2's interleaved vector is decoded with the format-1 partition
    /// layout, per §8.6.5 "reducible to format 1").
    fn decode_partition(
        &self,
        reader: &mut BitReaderLsb<'_>,
        book: &ValueBook,
        v: &mut [f32],
        offset: usize,
    ) -> Result<(), PartitionStop> {
        let n = self.partition_size as usize;
        let dims = book.codebook.dimensions as usize;
        if self.residue_type == 0 {
            // §8.6.3 format 0 specifics.
            if dims == 0 || n % dims != 0 {
                return Err(PartitionStop::Err(
                    ResidueError::Format0PartitionNotDivisible {
                        partition_size: self.partition_size,
                        dimensions: book.codebook.dimensions,
                    },
                ));
            }
            // §8.6.3 step 1: step = n / codebook_dimensions.
            let step = n / dims;
            // §8.6.3 step 2: iterate i over 0 .. step-1.
            for i in 0..step {
                // §8.6.3 step 3: read a VQ vector.
                let entry_temp = self.read_vq(reader, book)?;
                // §8.6.3 steps 4..5: scatter element j to offset+i+j*step.
                for (j, &val) in entry_temp.iter().enumerate() {
                    let idx = offset + i + j * step;
                    if idx < v.len() {
                        v[idx] += val;
                    }
                }
            }
        } else {
            // §8.6.4 format 1 specifics (also format 2's inner decode).
            // §8.6.4 step 1: i = 0.
            let mut i = 0usize;
            // §8.6.4 step 6: loop while i < n.
            while i < n {
                // §8.6.4 step 2: read a VQ vector.
                let entry_temp = self.read_vq(reader, book)?;
                // §8.6.4 steps 3..5: append elements contiguously.
                for &val in entry_temp.iter() {
                    let idx = offset + i;
                    if idx < v.len() {
                        v[idx] += val;
                    }
                    i += 1;
                    if i >= n {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// Walk a value codebook's Huffman tree to an entry index, then
    /// unpack that entry into a VQ vector (§3.3 hand-off → §3.2.1
    /// unpack). EOF mid-codeword is mapped to [`PartitionStop::Eof`]
    /// so the §8.6.2 caller can treat it as the nominal end-of-packet.
    fn read_vq(
        &self,
        reader: &mut BitReaderLsb<'_>,
        book: &ValueBook,
    ) -> Result<Vec<f32>, PartitionStop> {
        let entry = match book.tree.decode_entry(reader) {
            Ok(e) => e,
            Err(DecodeError::UnexpectedEndOfPacket) => return Err(PartitionStop::Eof),
        };
        unpack_vector(&book.codebook, entry).map_err(|e| PartitionStop::Err(ResidueError::Vq(e)))
    }
}

/// Internal control-flow for partition decode: separates the nominal
/// end-of-packet (§8.6.2) from a genuine error.
enum PartitionStop {
    /// End-of-packet reached mid-decode — nominal per §8.6.2.
    Eof,
    /// A genuine decode error.
    Err(ResidueError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::VqLookup;
    use crate::setup::ResidueHeader;
    use oxideav_core::bits::BitWriterLsb;

    /// Build a scalar (lookup_type 0) codebook with the given codeword
    /// lengths. Used as a classbook.
    fn scalar_book(dimensions: u16, lengths: Vec<u8>) -> VorbisCodebook {
        VorbisCodebook {
            dimensions,
            entries: lengths.len() as u32,
            codeword_lengths: lengths,
            lookup: VqLookup::None,
        }
    }

    /// Build a tessellation (lookup_type 2) VQ codebook with the given
    /// codeword lengths and multiplicand table.
    fn vq_book(
        dimensions: u16,
        lengths: Vec<u8>,
        multiplicands: Vec<u32>,
        delta: f32,
        min: f32,
    ) -> VorbisCodebook {
        VorbisCodebook {
            dimensions,
            entries: lengths.len() as u32,
            codeword_lengths: lengths,
            lookup: VqLookup::Tessellation {
                minimum_value: min,
                delta_value: delta,
                value_bits: 8,
                sequence_p: false,
                multiplicands,
            },
        }
    }

    /// A residue header builder for tests.
    fn residue_header(
        residue_type: u16,
        begin: u32,
        end: u32,
        partition_size: u32,
        classifications: u8,
        classbook: u8,
        books: Vec<[Option<u8>; 8]>,
    ) -> ResidueHeader {
        let cascade = books
            .iter()
            .map(|row| {
                let mut bits = 0u8;
                for (j, slot) in row.iter().enumerate() {
                    if slot.is_some() {
                        bits |= 1 << j;
                    }
                }
                bits
            })
            .collect();
        ResidueHeader {
            residue_type,
            residue_begin: begin,
            residue_end: end,
            partition_size,
            classifications,
            classbook,
            cascade,
            books,
        }
    }

    /// Encode a single-bit codeword stream by writing `bits` LSb-first.
    /// We control the canonical codewords by choosing codeword lengths,
    /// so for a 2-entry length-1 book, entry 0 = codeword `0`, entry 1 =
    /// codeword `1`. The Huffman walk reads MSb-first, but for length-1
    /// codewords a single bit selects the entry directly.
    fn encode_bits(bits: &[u8]) -> Vec<u8> {
        let mut w = BitWriterLsb::new();
        for &b in bits {
            w.write_bit(b != 0);
        }
        // Pad to a byte boundary; trailing zero bits are harmless.
        w.finish()
    }

    // ---------- construction validation ----------

    #[test]
    fn classbook_out_of_range_is_rejected() {
        let books = vec![[None; 8]];
        let header = residue_header(1, 0, 8, 4, 1, 5, books);
        let codebooks = vec![scalar_book(2, vec![1, 1])];
        assert_eq!(
            ResidueDecoder::new(&header, &codebooks).unwrap_err(),
            ResidueError::ClassbookOutOfRange {
                classbook: 5,
                codebook_count: 1,
            }
        );
    }

    #[test]
    fn value_book_out_of_range_is_rejected() {
        let mut row = [None; 8];
        row[0] = Some(9);
        let header = residue_header(1, 0, 8, 4, 1, 0, vec![row]);
        let codebooks = vec![scalar_book(1, vec![1, 1])];
        assert_eq!(
            ResidueDecoder::new(&header, &codebooks).unwrap_err(),
            ResidueError::ValueBookOutOfRange {
                class: 0,
                stage: 0,
                book: 9,
                codebook_count: 1,
            }
        );
    }

    #[test]
    fn value_book_without_lookup_is_rejected() {
        // classbook = book 0 (scalar), value book = book 1 (also scalar,
        // lookup_type 0) — illegal per §8.6.1.
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = residue_header(1, 0, 8, 4, 1, 0, vec![row]);
        let codebooks = vec![scalar_book(1, vec![1, 1]), scalar_book(2, vec![1, 1])];
        assert_eq!(
            ResidueDecoder::new(&header, &codebooks).unwrap_err(),
            ResidueError::ValueBookHasNoLookup {
                class: 0,
                stage: 0,
                book: 1,
            }
        );
    }

    #[test]
    fn zero_classwords_per_codeword_is_rejected() {
        let header = residue_header(1, 0, 8, 4, 1, 0, vec![[None; 8]]);
        let codebooks = vec![scalar_book(0, vec![1, 1])];
        assert_eq!(
            ResidueDecoder::new(&header, &codebooks).unwrap_err(),
            ResidueError::ZeroClasswordsPerCodeword
        );
    }

    #[test]
    fn unsupported_format_is_rejected() {
        let header = residue_header(3, 0, 8, 4, 1, 0, vec![[None; 8]]);
        let codebooks = vec![scalar_book(1, vec![1, 1])];
        assert_eq!(
            ResidueDecoder::new(&header, &codebooks).unwrap_err(),
            ResidueError::UnsupportedFormat(3)
        );
    }

    // ---------- n_to_read == 0 ----------

    #[test]
    fn empty_range_returns_zeroed_vectors() {
        // begin == end → n_to_read == 0 → no decode (§8.6.2 step 2).
        let header = residue_header(1, 4, 4, 4, 1, 0, vec![[None; 8]]);
        let codebooks = vec![scalar_book(1, vec![1, 1])];
        let dec = ResidueDecoder::new(&header, &codebooks).unwrap();
        let data = encode_bits(&[]);
        let mut r = BitReaderLsb::new(&data);
        let out = dec.decode(&mut r, 16, &[false]).unwrap();
        assert_eq!(out, vec![vec![0.0f32; 8]]);
    }

    // ---------- format 1 single-channel decode ----------

    /// A minimal format-1 mono residue. classbook has dimensions 1
    /// (1 classword per codeword) and 1 classification; the single
    /// classification's stage-0 value book is a 1-D VQ book. Every
    /// partition reads one classbook codeword (always class 0) then one
    /// value codeword per scalar.
    #[test]
    fn format1_single_channel_two_partitions() {
        // classbook: 2 entries, dim 1, lengths [1,1] → entry0 = bit 0,
        // entry1 = bit 1. We only ever need class 0, so we feed bit 0.
        let classbook = scalar_book(1, vec![1, 1]);
        // value book: 1-D tessellation, 2 entries, lengths [1,1].
        // multiplicands [3, 5], delta 1, min 0 → entry0 → [3], entry1 → [5].
        let valbook = vq_book(1, vec![1, 1], vec![3, 5], 1.0, 0.0);
        let codebooks = vec![classbook, valbook];

        let mut row = [None; 8];
        row[0] = Some(1); // stage 0 → value book index 1
        let header = residue_header(1, 0, 4, 2, 1, 0, vec![row]);
        let dec = ResidueDecoder::new(&header, &codebooks).unwrap();

        // n_to_read = 4, partition_size = 2 → 2 partitions.
        // classwords_per_codeword = 1, so on pass 0 each partition needs
        // its own classbook codeword (class 0 = bit 0).
        // Decode order (pass 0):
        //   partition_count 0: read classbook (bit 0 → class 0), then in
        //     the step-13 group decode value codewords for 2 scalars.
        //   partition_count 1: read classbook (bit 0 → class 0), decode 2.
        //
        // Bit stream (LSb-first, MSb-of-codeword first per the tree walk):
        //   classbook entry 0 → bit 0
        //   partition 0 scalars: entry0 ([3]), entry1 ([5]) → bits 0,1
        //   classbook entry 0 → bit 0
        //   partition 1 scalars: entry1 ([5]), entry0 ([3]) → bits 1,0
        let data = encode_bits(&[0, 0, 1, 0, 1, 0]);
        let mut r = BitReaderLsb::new(&data);
        let out = dec.decode(&mut r, 8, &[false]).unwrap();
        // per_channel_size = blocksize/2 = 4.
        // partition 0 (offset 0): [3, 5]; partition 1 (offset 2): [5, 3].
        assert_eq!(out, vec![vec![3.0, 5.0, 5.0, 3.0]]);
    }

    /// `do_not_decode` channel is allocated, zeroed, and skipped.
    #[test]
    fn do_not_decode_channel_is_zeroed_and_skipped() {
        let classbook = scalar_book(1, vec![1, 1]);
        let valbook = vq_book(1, vec![1, 1], vec![3, 5], 1.0, 0.0);
        let codebooks = vec![classbook, valbook];
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = residue_header(1, 0, 4, 2, 1, 0, vec![row]);
        let dec = ResidueDecoder::new(&header, &codebooks).unwrap();

        // Two channels: channel 0 decoded, channel 1 'do not decode'.
        // Only channel 0 consumes bits, identical to the single-channel
        // case above.
        let data = encode_bits(&[0, 0, 1, 0, 1, 0]);
        let mut r = BitReaderLsb::new(&data);
        let out = dec.decode(&mut r, 8, &[false, true]).unwrap();
        assert_eq!(out[0], vec![3.0, 5.0, 5.0, 3.0]);
        assert_eq!(out[1], vec![0.0, 0.0, 0.0, 0.0]);
    }

    // ---------- format 0 decode ----------

    /// Format 0 scatters vector element j to offset + i + j*step. With a
    /// 2-D value book, partition_size 4 → step = 4/2 = 2. Reading two VQ
    /// vectors [a0,a1] then [b0,b1] fills offset+{0,1,2,3} as
    /// [a0, b0, a1, b1].
    #[test]
    fn format0_interleaved_scatter() {
        let classbook = scalar_book(1, vec![1, 1]);
        // 2-D value book: 2 entries, lengths [1,1].
        // multiplicands (entries×dims = 4): entry0 → [1,2], entry1 → [4,8].
        let valbook = vq_book(2, vec![1, 1], vec![1, 2, 4, 8], 1.0, 0.0);
        let codebooks = vec![classbook, valbook];
        let mut row = [None; 8];
        row[0] = Some(1);
        // format 0, single partition of size 4 → step 2 → 2 reads.
        let header = residue_header(0, 0, 4, 4, 1, 0, vec![row]);
        let dec = ResidueDecoder::new(&header, &codebooks).unwrap();

        // pass 0: classbook (bit 0 → class 0), then partition decode:
        //   i=0 read entry0 [1,2]; i=1 read entry1 [4,8].
        // scatter: v[0+0+0*2]=1, v[0+0+1*2]=2, v[0+1+0*2]=4, v[0+1+1*2]=8
        //   → [1, 4, 2, 8]
        // bits: classbook 0; value entry0 (bit 0), entry1 (bit 1).
        let data = encode_bits(&[0, 0, 1]);
        let mut r = BitReaderLsb::new(&data);
        let out = dec.decode(&mut r, 8, &[false]).unwrap();
        assert_eq!(out, vec![vec![1.0, 4.0, 2.0, 8.0]]);
    }

    // ---------- format 2 interleave/deinterleave ----------

    /// Format 2 decodes a single interleaved vector of length ch*n then
    /// de-interleaves v[i*ch + j] → output[j][i]. With ch=2, n=4, the
    /// interleaved vector [a,b,c,d,e,f,g,h] de-interleaves to
    /// channel 0 = [a,c,e,g], channel 1 = [b,d,f,h].
    #[test]
    fn format2_interleave_deinterleave() {
        let classbook = scalar_book(1, vec![1, 1]);
        // 1-D value book with 2 entries: entry0 → [1], entry1 → [10].
        let valbook = vq_book(1, vec![1, 1], vec![1, 10], 1.0, 0.0);
        let codebooks = vec![classbook, valbook];
        let mut row = [None; 8];
        row[0] = Some(1);
        // format 2, ch=2, per_channel_size=4 → interleaved length 8.
        // begin 0, end 8 (the interleaved size), partition_size 8 →
        // 1 partition, classwords_per_codeword 1.
        let header = residue_header(2, 0, 8, 8, 1, 0, vec![row]);
        let dec = ResidueDecoder::new(&header, &codebooks).unwrap();

        // The interleaved decode is a single format-1 vector decode:
        //   classbook (bit 0 → class 0); then 8 scalars from the value
        //   book: choose entries 0,1,0,1,0,1,0,1 →
        //   interleaved = [1,10,1,10,1,10,1,10]
        // de-interleave (ch=2): ch0 = idx 0,2,4,6 = [1,1,1,1];
        //   ch1 = idx 1,3,5,7 = [10,10,10,10].
        let data = encode_bits(&[0, /*scalars*/ 0, 1, 0, 1, 0, 1, 0, 1]);
        let mut r = BitReaderLsb::new(&data);
        let out = dec.decode(&mut r, 8, &[false, false]).unwrap();
        assert_eq!(out[0], vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(out[1], vec![10.0, 10.0, 10.0, 10.0]);
    }

    /// Format 2 with all channels 'do not decode' → no bits consumed,
    /// zeroed output (§8.6.5 step 1).
    #[test]
    fn format2_all_do_not_decode_is_noop() {
        let classbook = scalar_book(1, vec![1, 1]);
        let valbook = vq_book(1, vec![1, 1], vec![1, 10], 1.0, 0.0);
        let codebooks = vec![classbook, valbook];
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = residue_header(2, 0, 8, 8, 1, 0, vec![row]);
        let dec = ResidueDecoder::new(&header, &codebooks).unwrap();
        let data = encode_bits(&[1, 1, 1, 1]); // would-be bits, ignored
        let mut r = BitReaderLsb::new(&data);
        let out = dec.decode(&mut r, 8, &[true, true]).unwrap();
        assert_eq!(out[0], vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(out[1], vec![0.0, 0.0, 0.0, 0.0]);
        // No bits should have been consumed.
        assert_eq!(r.bit_position(), 0);
    }

    // ---------- end-of-packet is nominal ----------

    /// An empty packet body (the bit reader is dry at the very first
    /// classbook read) is a nominal end-of-packet (§8.6.2): decode
    /// returns the zeroed vectors instead of erroring.
    #[test]
    fn eof_at_first_read_is_nominal() {
        let classbook = scalar_book(1, vec![1, 1]);
        let valbook = vq_book(1, vec![1, 1], vec![3, 5], 1.0, 0.0);
        let codebooks = vec![classbook, valbook];
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = residue_header(1, 0, 4, 2, 1, 0, vec![row]);
        let dec = ResidueDecoder::new(&header, &codebooks).unwrap();

        // No bits at all — the first classbook read hits EOF. §8.6.2
        // makes this nominal: return the zeroed vectors, not an error.
        let data: Vec<u8> = Vec::new();
        let mut r = BitReaderLsb::new(&data);
        let out = dec.decode(&mut r, 8, &[false]).unwrap();
        assert_eq!(out, vec![vec![0.0, 0.0, 0.0, 0.0]]);
    }

    /// Running the bit reader dry mid-decode returns the work so far
    /// (§8.6.2 "nominal occurrence"), not an error. We craft an exact
    /// byte-boundary overrun using a *complete* 4-entry depth-2 value
    /// book (lengths [2,2,2,2]; entry 0 = "00"). Each value read costs 2
    /// bits and each classbook read 1 bit, so a single byte (8 bits)
    /// covers: partition 0 = cb + 2 values (5 bits), partition 1 = cb +
    /// 1 value (3 bits) = 8 bits exactly; partition 1's *second* value
    /// read then overruns the buffer → nominal EOF. The output's first
    /// three scalars are decoded; the fourth stays zero.
    #[test]
    fn eof_mid_codeword_is_nominal() {
        let classbook = scalar_book(1, vec![1, 1]);
        // value book: 4 entries, lengths [2,2,2,2] → complete depth-2
        // tree. entry 0 = "00". multiplicands (entries×dims = 4):
        // entry0 → [3], entry1 → [5], entry2 → [7], entry3 → [9].
        let valbook = vq_book(1, vec![2, 2, 2, 2], vec![3, 5, 7, 9], 1.0, 0.0);
        let codebooks = vec![classbook, valbook];
        let mut row = [None; 8];
        row[0] = Some(1);
        let header = residue_header(1, 0, 4, 2, 1, 0, vec![row]);
        let dec = ResidueDecoder::new(&header, &codebooks).unwrap();

        // 8 bits, all zero (one byte):
        //   partition 0: cb "0" (class 0); val "00" → entry0 [3];
        //                val "00" → entry0 [3]  (5 bits, ends at bit 5)
        //   partition 1: cb "0" (class 0); val "00" → entry0 [3]
        //                (bits 5,6,7 → ends at bit 8 = buffer end)
        //                second val read overruns → EOF.
        // → output [3, 3, 3, 0].
        let data = vec![0u8]; // 8 zero bits
        let mut r = BitReaderLsb::new(&data);
        let out = dec.decode(&mut r, 8, &[false]).unwrap();
        assert_eq!(out, vec![vec![3.0, 3.0, 3.0, 0.0]]);
    }

    // ---------- multi-classword classbook ----------

    /// A classbook with dimensions 2 packs 2 classifications per
    /// codeword. Entry value `temp` is unpacked descending:
    /// class[i+pc] = temp % classifications; temp /= classifications.
    /// With classifications = 3 and a classbook entry value of 7:
    ///   i=1: class[1] = 7 % 3 = 1; temp = 2
    ///   i=0: class[0] = 2 % 3 = 2; temp = 0
    /// So partition 0 → class 2, partition 1 → class 1.
    #[test]
    fn multi_classword_unpack_is_descending() {
        // classbook dim 2, 8 entries lengths all 3 → balanced; entry e
        // has codeword = e in 3 bits (MSb first). We want entry 7 = 0b111.
        let classbook = scalar_book(2, vec![3, 3, 3, 3, 3, 3, 3, 3]);
        // value books for classes 1 and 2 only (class 0 unused here).
        // class 1 stage 0 → book idx 1; class 2 stage 0 → book idx 1.
        let valbook = vq_book(1, vec![1, 1], vec![100, 200], 1.0, 0.0);
        let codebooks = vec![classbook, valbook];
        let r0 = [None; 8]; // class 0: no books
        let mut r1 = [None; 8];
        r1[0] = Some(1); // class 1
        let mut r2 = [None; 8];
        r2[0] = Some(1); // class 2
                         // 3 classifications, classbook idx 0.
                         // partition_size 1, begin 0, end 2 → 2 partitions; classwords 2,
                         // so a single classbook codeword covers both partitions.
        let header = residue_header(1, 0, 2, 1, 3, 0, vec![r0, r1, r2]);
        let dec = ResidueDecoder::new(&header, &codebooks).unwrap();

        // pass 0: read classbook entry 7 (bits 1,1,1 MSb-first → tree
        // walk: each '1' goes right). With a balanced 8-entry length-3
        // book, codeword for entry 7 is 111.
        //   → class[0] = 2, class[1] = 1.
        // Then decode group: partition 0 (class 2) reads value book →
        // 1 scalar; partition 1 (class 1) reads value book → 1 scalar.
        // value entries: partition 0 → entry0 ([100]); partition 1 →
        // entry1 ([200]).
        // bits: classbook 1,1,1; value 0; value 1.
        let data = encode_bits(&[1, 1, 1, 0, 1]);
        let mut r = BitReaderLsb::new(&data);
        let out = dec.decode(&mut r, 4, &[false]).unwrap();
        // per_channel_size = 2. partition 0 offset 0 → 100; partition 1
        // offset 1 → 200.
        assert_eq!(out, vec![vec![100.0, 200.0]]);
    }

    // ---------- cascade: multiple stages accumulate ----------

    /// Two cascade stages for the same classification accumulate (add)
    /// into the same output positions across passes (§8.6.2 step 19
    /// "decode partition into output vector … using codebook … in VQ
    /// context" — the format decoders *add* into `v`).
    #[test]
    fn cascade_stages_accumulate() {
        let classbook = scalar_book(1, vec![1, 1]);
        // stage 0 book: entry0 → [1]; stage 1 book: entry0 → [10].
        let stage0 = vq_book(1, vec![1, 1], vec![1, 2], 1.0, 0.0);
        let stage1 = vq_book(1, vec![1, 1], vec![10, 20], 1.0, 0.0);
        let codebooks = vec![classbook, stage0, stage1];
        let mut row = [None; 8];
        row[0] = Some(1); // stage 0 → book 1
        row[1] = Some(2); // stage 1 → book 2
                          // 1 partition of size 1, begin 0 end 1.
        let header = residue_header(1, 0, 1, 1, 1, 0, vec![row]);
        let dec = ResidueDecoder::new(&header, &codebooks).unwrap();

        // pass 0: classbook (bit 0 → class 0); stage-0 book read entry0
        //   → v[0] += 1.
        // pass 1: stage-1 book read entry0 → v[0] += 10. (classbook is
        //   only read on pass 0.)
        // passes 2..7: class 0 has no stage book → unused, nothing read.
        // bits: pass0 classbook 0, stage0 value 0; pass1 stage1 value 0.
        let data = encode_bits(&[0, 0, 0]);
        let mut r = BitReaderLsb::new(&data);
        let out = dec.decode(&mut r, 2, &[false]).unwrap();
        // v[0] = 1 + 10 = 11.
        assert_eq!(out, vec![vec![11.0]]);
    }
}

//! Vorbis I audio-packet decode driver stages that are fully specified by
//! the Vorbis I bitstream specification and require no inverse MDCT:
//! §4.3.1 "packet type, mode and window decode", §4.3.3 "nonzero vector
//! propagate" and §4.3.6 "dot product".
//!
//! The §4.3 audio-packet decode pipeline is a fixed sequence of stages
//! (§4.3 "Audio packet decode and synthesis"):
//!
//! 1. **§4.3.1 packet type, mode and window decode** ([`read_packet_header`]
//!    / [`AudioPacketHeader::build_window`], this module). Drives the
//!    window builder [`crate::synthesis::vorbis_window`].
//! 2. §4.3.2 floor curve decode (per channel:
//!    [`crate::floor1::Floor1Decoder`] / [`crate::floor0::Floor0Decoder`]).
//! 3. **§4.3.3 nonzero vector propagate** ([`nonzero_propagate`], this
//!    module).
//! 4. §4.3.4 residue decode ([`crate::residue::ResidueDecoder`]).
//! 5. §4.3.5 inverse coupling ([`crate::synthesis::inverse_couple_all`]).
//! 6. **§4.3.6 dot product** ([`dot_product`] / [`dot_product_all`], this
//!    module).
//! 7. §4.3.7 inverse MDCT — **not implementable from this spec**: §4.3.7
//!    defers the MDCT definition entirely to external reference `[1]`
//!    (T. Sporer, K. Brandenburg, B. Edler), which the workspace
//!    clean-room policy bars. See the crate README "What does not yet
//!    work" for the documented docs gap.
//! 8. §4.3.8 overlap/add.
//!
//! This module lands the three stages that are completely specified in the
//! Vorbis I spec text itself (no external citation) and that depend only
//! on the already-implemented setup-header walker, floor and residue
//! decoders: the §4.3.1 packet-prelude reader (packet type, mode number,
//! blocksize selection, long-block window flags), the §4.3.3 coupling-aware
//! "this channel really is used after all" propagation, and the §4.3.6
//! element-wise floor × residue spectrum product.
//!
//! # §4.3.1 packet type, mode and window decode
//!
//! Every audio packet's bitstream begins with this fixed-shape prelude
//! (§4.3.1):
//!
//! 1. 1 bit `[packet_type]`; verify == 0 (audio).
//! 2. `ilog([vorbis_mode_count] - 1)` bits `[mode_number]` indexing the
//!    setup header's [`crate::setup::ModeHeader`] list.
//! 3. Resolve the per-frame blocksize `[n]`: `blocksize_0` if the mode's
//!    `blockflag` is clear, else `blocksize_1`.
//! 4. For a long block (`blockflag` set) only, read 1 bit
//!    `[previous_window_flag]` and 1 bit `[next_window_flag]`. A short
//!    block does not carry these and reuses the same symmetric short shape
//!    every frame (§4.3.1 step 4b).
//!
//! End-of-packet anywhere in §4.3.1 is an error that "discards this packet
//! from the stream" (§4.3.1 closing note) — this is the only §4.3 stage
//! where EOF is fatal rather than nominal, because the decoder cannot
//! even pick a window without these bits.
//!
//! [`read_packet_header`] returns an [`AudioPacketHeader`] holding the
//! resolved `(mode_number, blockflag, n, previous_window_flag,
//! next_window_flag)`. [`AudioPacketHeader::build_window`] then drives the
//! existing [`crate::synthesis::vorbis_window`] builder with the resolved
//! fields, producing the length-`n` window the inverse MDCT will eventually
//! consume.
//!
//! # §4.3.3 nonzero vector propagate
//!
//! Floor curve decode (§4.3.2 step 6) sets a per-channel `[no_residue]`
//! flag: `true` when the floor returned `'unused'` (the channel carried no
//! energy this frame and its residue is not coded in the stream). But
//! channel coupling can mix a zeroed vector with a nonzeroed one and
//! produce two nonzeroed vectors, so §4.3.3 clears `[no_residue]` for both
//! members of any coupling step where *either* member is used:
//!
//! > for each `[i]` from `0 ... [vorbis_mapping_coupling_steps]-1`: if
//! > either `[no_residue]` entry for channel
//! > (`[vorbis_mapping_magnitude]` element `[i]`) or channel
//! > (`[vorbis_mapping_angle]` element `[i]`) are set to false, then both
//! > must be set to false.
//!
//! [`nonzero_propagate`] runs that loop in **ascending** step order as the
//! spec writes it. (Ascending vs descending is immaterial here — the loop
//! body only ever clears flags and a cleared flag stays cleared — but the
//! spec text is ascending so we follow it.)
//!
//! # §4.3.6 dot product
//!
//! After inverse coupling, each channel holds its length-`n/2` residue
//! vector. §4.3.6 multiplies the floor curve element-wise by the residue
//! vector to produce the length-`n/2` audio spectrum:
//!
//! > For each channel, multiply each element of the floor curve by each
//! > element of that channel's residue vector. The result is the dot
//! > product of the floor and residue vectors for each channel; the
//! > produced vectors are the length `[n]/2` audio spectrum for each
//! > channel.
//!
//! ("dot product" here is the spec's term for the element-wise — Hadamard
//! — product, not the scalar inner product; the result is a length-`n/2`
//! vector, not a scalar, as the closing sentence makes explicit.)
//!
//! A channel whose floor was `'unused'` (and whose `[no_residue]` survived
//! §4.3.3, i.e. it was not pulled back in by coupling) produces an
//! all-zero output vector (§4.3.3 "that final output vector is all-zero
//! values (and the floor is zero)"). [`dot_product_all`] models an unused
//! channel's floor curve as [`None`] and emits the zero vector for it.

/// The resolved §4.3.1 packet-prelude fields for one audio frame.
///
/// Produced by [`read_packet_header`] from the raw packet's bit stream + a
/// parsed [`crate::setup::VorbisSetupHeader`] + the stream's two
/// blocksizes from the identification header (§4.2.2). The fields are the
/// minimum needed to drive the rest of §4.3 (floor decode picks the
/// mode's mapping, residue decode reuses the resolved `n`, and the window
/// builder needs `blockflag` + the two window flags).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioPacketHeader {
    /// `[mode_number]` — index into the setup header's
    /// [`crate::setup::VorbisSetupHeader::modes`] list, read in
    /// `ilog([vorbis_mode_count] - 1)` bits (§4.3.1 step 2).
    pub mode_number: u32,
    /// `[vorbis_mode_blockflag]` of the selected mode — `false` for a
    /// short block (blocksize `blocksize_0`), `true` for a long block
    /// (blocksize `blocksize_1`). Cached here so callers don't have to
    /// re-index the mode list (§4.3.1 step 3).
    pub blockflag: bool,
    /// `[n]` — the resolved per-frame blocksize: `blocksize_0` if
    /// [`Self::blockflag`] is `false`, else `blocksize_1` (§4.3.1
    /// step 3).
    pub n: usize,
    /// `[previous_window_flag]` — read only for long blocks (§4.3.1 step
    /// 4a.i). For short blocks the spec does not transmit this bit;
    /// `false` here is a placeholder and is ignored by the window
    /// builder, which always uses the symmetric short shape (§4.3.1
    /// step 4b).
    pub previous_window_flag: bool,
    /// `[next_window_flag]` — read only for long blocks (§4.3.1 step
    /// 4a.ii). For short blocks the spec does not transmit this bit;
    /// `false` here is a placeholder and is ignored by the window
    /// builder.
    pub next_window_flag: bool,
}

impl AudioPacketHeader {
    /// Build the length-`n` Vorbis window for this packet by driving the
    /// existing [`crate::synthesis::vorbis_window`] builder with the
    /// resolved fields (§4.3.1 step 4 + §1.3.2 generation procedure).
    ///
    /// `blocksize_0` is the stream's short blocksize from the
    /// identification header (§4.2.2), needed for the hybrid ramps when a
    /// long block laps a short neighbour.
    ///
    /// # Errors
    ///
    /// Returns [`crate::synthesis::WindowError`] for the same conditions
    /// the bare window builder rejects (non-power-of-two `n`, or
    /// `blocksize_0 > n` on a long block).
    pub fn build_window(
        &self,
        blocksize_0: usize,
    ) -> Result<Vec<f32>, crate::synthesis::WindowError> {
        crate::synthesis::vorbis_window(
            self.n,
            blocksize_0,
            self.blockflag,
            self.previous_window_flag,
            self.next_window_flag,
        )
    }
}

/// Read the §4.3.1 packet prelude from an audio packet's bit stream.
///
/// * `reader` — an LSB-first bit reader positioned at the first bit of an
///   audio packet's bitstream (RFC-3533-stripped, Ogg-page-coalesced
///   payload; the parser is intentionally bring-your-own-packet).
/// * `setup` — the stream's parsed setup header. Only [`crate::setup::VorbisSetupHeader::modes`]
///   is consulted: its length sizes the `[mode_number]` read width per
///   `ilog([vorbis_mode_count] - 1)`, and the selected mode's `blockflag`
///   resolves the per-frame blocksize.
/// * `blocksize_0` / `blocksize_1` — the two blocksizes from the
///   identification header (§4.2.2). `n` resolves to `blocksize_0` when
///   the selected mode's `blockflag` is clear and to `blocksize_1`
///   otherwise.
///
/// Returns an [`AudioPacketHeader`] holding the resolved fields. The
/// reader is left positioned immediately after the last consumed bit:
/// after the §4.3.1 step-4a.ii `[next_window_flag]` for a long block, or
/// after the §4.3.1 step-3 blocksize resolution for a short block.
///
/// # Errors
///
/// [`PacketError::NonAudioPacketType`] if the §4.3.1 step-1 `[packet_type]`
/// bit is `1` (the decoder "must ignore the packet and not attempt
/// decoding it to audio" per §4.3). [`PacketError::BadModeNumber`] if the
/// §4.3.1 step-2 `[mode_number]` indexes past the setup header's mode
/// list. [`PacketError::EmptyModeList`] if `setup.modes` is empty (a
/// well-formed setup header always has `mode_count >= 1`, but the spec
/// makes no explicit invariant statement, so we check defensively).
/// [`PacketError::UnexpectedEndOfPacket`] if the bit reader runs out
/// anywhere in §4.3.1 — this stage is the **only** §4.3 stage where the
/// spec explicitly says "An end-of-packet condition up to this point
/// should be considered an error that discards this packet" (§4.3.1
/// closing note).
pub fn read_packet_header(
    reader: &mut oxideav_core::bits::BitReaderLsb<'_>,
    setup: &crate::setup::VorbisSetupHeader,
    blocksize_0: usize,
    blocksize_1: usize,
) -> Result<AudioPacketHeader, PacketError> {
    if setup.modes.is_empty() {
        return Err(PacketError::EmptyModeList);
    }
    // §4.3.1 step 1.
    let packet_type = reader
        .read_u32(1)
        .map_err(|_| PacketError::UnexpectedEndOfPacket {
            stage: PacketHeaderStage::PacketType,
        })?;
    if packet_type != 0 {
        return Err(PacketError::NonAudioPacketType {
            packet_type: packet_type as u8,
        });
    }
    // §4.3.1 step 2: `ilog([vorbis_mode_count] - 1)` bits. §9.2.1 `ilog`
    // returns 0 for input 0 (the single-mode degenerate case reads zero
    // bits and resolves to mode 0 unconditionally — the only legal
    // `mode_number` value when `mode_count == 1`).
    let mode_count = setup.modes.len();
    let mode_bits = crate::codebook::ilog((mode_count as u32).saturating_sub(1));
    let mode_number =
        reader
            .read_u32(mode_bits)
            .map_err(|_| PacketError::UnexpectedEndOfPacket {
                stage: PacketHeaderStage::ModeNumber,
            })?;
    if (mode_number as usize) >= mode_count {
        return Err(PacketError::BadModeNumber {
            mode_number,
            mode_count,
        });
    }
    let mode = setup.modes[mode_number as usize];
    let blockflag = mode.blockflag;
    // §4.3.1 step 3.
    let n = if blockflag { blocksize_1 } else { blocksize_0 };
    // §4.3.1 step 4.
    let (previous_window_flag, next_window_flag) = if blockflag {
        // step 4a.i / 4a.ii — both bits are required, EOF is fatal.
        let prev = reader
            .read_bit()
            .map_err(|_| PacketError::UnexpectedEndOfPacket {
                stage: PacketHeaderStage::PreviousWindowFlag,
            })?;
        let next = reader
            .read_bit()
            .map_err(|_| PacketError::UnexpectedEndOfPacket {
                stage: PacketHeaderStage::NextWindowFlag,
            })?;
        (prev, next)
    } else {
        // step 4b — no bits read, flags are placeholders.
        (false, false)
    };
    Ok(AudioPacketHeader {
        mode_number,
        blockflag,
        n,
        previous_window_flag,
        next_window_flag,
    })
}

/// Run the §4.3.3 nonzero-vector-propagate pass over a per-channel
/// `no_residue` flag slice and a mapping's coupling step list.
///
/// `no_residue[channel]` is `true` when channel `channel`'s floor returned
/// `'unused'` at §4.3.2 step 6. For every coupling step
/// `(magnitude_channel, angle_channel)`, if *either* member is used
/// (`no_residue` false), §4.3.3 forces *both* members used (`no_residue`
/// cleared), because coupling can turn a zeroed + nonzeroed pair into two
/// nonzeroed vectors.
///
/// The flags are modified in place. Returns [`PacketError::ChannelOutOfRange`]
/// if a coupling step names a channel index `>= no_residue.len()`; the
/// setup parser already range-checks coupling channels against
/// `audio_channels` (§4.2.4 "Mappings"), so this only triggers when the
/// caller passes a flag slice that is not one-per-channel.
///
/// # Errors
///
/// [`PacketError::ChannelOutOfRange`] if a coupling step references a
/// channel index outside `no_residue`.
pub fn nonzero_propagate(
    no_residue: &mut [bool],
    coupling: &[crate::setup::MappingCouplingStep],
) -> Result<(), PacketError> {
    let channels = no_residue.len();
    // §4.3.3: "for each [i] from 0 ... [vorbis_mapping_coupling_steps]-1".
    for (step, cs) in coupling.iter().enumerate() {
        let mag = cs.magnitude_channel as usize;
        let ang = cs.angle_channel as usize;
        if mag >= channels {
            return Err(PacketError::ChannelOutOfRange {
                step,
                channel: mag,
                channels,
            });
        }
        if ang >= channels {
            return Err(PacketError::ChannelOutOfRange {
                step,
                channel: ang,
                channels,
            });
        }
        // "if either [no_residue] entry for channel (magnitude) or channel
        // (angle) are set to false, then both must be set to false."
        if !no_residue[mag] || !no_residue[ang] {
            no_residue[mag] = false;
            no_residue[ang] = false;
        }
    }
    Ok(())
}

/// Compute the §4.3.6 dot product (element-wise / Hadamard product) of one
/// channel's floor curve and residue vector, writing the length-`n/2`
/// audio spectrum into `spectrum`.
///
/// All three slices must be the same length (`n/2`, the §4.3.6 floor and
/// residue synthesis length). The operation is `spectrum[i] = floor[i] *
/// residue[i]`.
///
/// # Panics
///
/// Panics if `floor`, `residue` and `spectrum` are not all the same
/// length; per §4.3.6 every per-channel vector in an audio packet is
/// exactly `n/2` long, so a mismatch indicates a caller bug rather than
/// stream data.
pub fn dot_product(floor: &[f32], residue: &[f32], spectrum: &mut [f32]) {
    assert_eq!(
        floor.len(),
        residue.len(),
        "dot_product: floor/residue length mismatch"
    );
    assert_eq!(
        floor.len(),
        spectrum.len(),
        "dot_product: floor/spectrum length mismatch"
    );
    for ((s, &f), &r) in spectrum.iter_mut().zip(floor.iter()).zip(residue.iter()) {
        *s = f * r;
    }
}

/// Run the §4.3.6 dot product across every channel, returning one
/// length-`n/2` audio spectrum vector per channel.
///
/// * `floors[channel]` is the channel's decoded floor curve, or [`None`]
///   if the floor returned `'unused'` (§4.3.2 step 6) *and* the channel
///   was not pulled back in by §4.3.3 coupling propagation. An unused
///   channel produces an all-zero spectrum of length `half_n` (§4.3.3
///   "that final output vector is all-zero values").
/// * `residues[channel]` is the channel's post-coupling residue vector
///   (§4.3.5 output). It is only read for channels with a `Some` floor.
/// * `half_n` is `n/2`, the per-channel spectrum length (§4.3.6).
///
/// # Errors
///
/// [`PacketError::ChannelCountMismatch`] if `floors` and `residues` have
/// different lengths. [`PacketError::VectorLength`] if any `Some` floor
/// curve or its paired residue vector is not exactly `half_n` long.
pub fn dot_product_all(
    floors: &[Option<Vec<f32>>],
    residues: &[Vec<f32>],
    half_n: usize,
) -> Result<Vec<Vec<f32>>, PacketError> {
    if floors.len() != residues.len() {
        return Err(PacketError::ChannelCountMismatch {
            floors: floors.len(),
            residues: residues.len(),
        });
    }
    let mut out = Vec::with_capacity(floors.len());
    for (channel, floor) in floors.iter().enumerate() {
        match floor {
            // §4.3.3: an 'unused' channel's final output vector is all-zero.
            None => out.push(vec![0.0f32; half_n]),
            Some(curve) => {
                if curve.len() != half_n {
                    return Err(PacketError::VectorLength {
                        channel,
                        which: VectorKind::Floor,
                        expected: half_n,
                        actual: curve.len(),
                    });
                }
                let residue = &residues[channel];
                if residue.len() != half_n {
                    return Err(PacketError::VectorLength {
                        channel,
                        which: VectorKind::Residue,
                        expected: half_n,
                        actual: residue.len(),
                    });
                }
                let mut spectrum = vec![0.0f32; half_n];
                dot_product(curve, residue, &mut spectrum);
                out.push(spectrum);
            }
        }
    }
    Ok(out)
}

/// Which per-channel vector triggered a [`PacketError::VectorLength`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorKind {
    /// The floor curve vector (§4.3.2 / §4.3.6).
    Floor,
    /// The post-coupling residue vector (§4.3.5 / §4.3.6).
    Residue,
}

impl core::fmt::Display for VectorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VectorKind::Floor => write!(f, "floor curve"),
            VectorKind::Residue => write!(f, "residue vector"),
        }
    }
}

/// Which sub-step of §4.3.1 ran out of bits in [`read_packet_header`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketHeaderStage {
    /// §4.3.1 step 1 — the 1-bit `[packet_type]`.
    PacketType,
    /// §4.3.1 step 2 — the `ilog([vorbis_mode_count] - 1)`-bit
    /// `[mode_number]`.
    ModeNumber,
    /// §4.3.1 step 4a.i — the 1-bit `[previous_window_flag]` (long block
    /// only).
    PreviousWindowFlag,
    /// §4.3.1 step 4a.ii — the 1-bit `[next_window_flag]` (long block
    /// only).
    NextWindowFlag,
}

impl core::fmt::Display for PacketHeaderStage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PacketHeaderStage::PacketType => write!(f, "packet_type"),
            PacketHeaderStage::ModeNumber => write!(f, "mode_number"),
            PacketHeaderStage::PreviousWindowFlag => write!(f, "previous_window_flag"),
            PacketHeaderStage::NextWindowFlag => write!(f, "next_window_flag"),
        }
    }
}

/// Errors that can arise while driving the §4.3.1 / §4.3.3 / §4.3.6
/// packet stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PacketError {
    /// §4.3.1 step 1 — the audio-packet prelude's first bit was `1`, not
    /// `0`. Per §4.3 the decoder "must ignore the packet and not attempt
    /// decoding it to audio".
    NonAudioPacketType {
        /// The offending packet-type bit value (always 1 in this
        /// context; included for symmetry with the header packets'
        /// 8-bit type byte).
        packet_type: u8,
    },
    /// §4.3.1 step 2 — `[mode_number]` indexed past the setup header's
    /// mode list. Indicates stream corruption (the encoder is forbidden
    /// from emitting an out-of-range mode index).
    BadModeNumber {
        /// The decoded mode-number value.
        mode_number: u32,
        /// `vorbis_mode_count` from the setup header.
        mode_count: usize,
    },
    /// The supplied [`crate::setup::VorbisSetupHeader::modes`] list was
    /// empty. A well-formed Vorbis I stream always has at least one mode;
    /// this is a defensive caller-bug guard.
    EmptyModeList,
    /// §4.3.1 — the bit reader ran out of bits before the prelude
    /// completed. The spec's closing note: "An end-of-packet condition
    /// up to this point should be considered an error that discards this
    /// packet from the stream." (Unlike §4.3.2 onwards, where EOF is
    /// nominal.)
    UnexpectedEndOfPacket {
        /// Which §4.3.1 sub-step ran out of bits.
        stage: PacketHeaderStage,
    },
    /// A coupling step's magnitude- or angle-channel index pointed past
    /// the supplied `no_residue` flag slice. The setup parser already
    /// range-checks these against `audio_channels` (§4.2.4 "Mappings"); a
    /// violation here means the flag slice is not one entry per channel.
    ChannelOutOfRange {
        /// The coupling step index (ascending §4.3.3 loop order).
        step: usize,
        /// The offending channel index.
        channel: usize,
        /// The number of `no_residue` flags available.
        channels: usize,
    },
    /// `floors` and `residues` had different channel counts in
    /// [`dot_product_all`].
    ChannelCountMismatch {
        /// The number of floor curves supplied.
        floors: usize,
        /// The number of residue vectors supplied.
        residues: usize,
    },
    /// A `Some` floor curve or its paired residue vector was not exactly
    /// `n/2` long (§4.3.6 mandates length `n/2` for every per-channel
    /// vector).
    VectorLength {
        /// The channel whose vector had the wrong length.
        channel: usize,
        /// Whether the offending vector was the floor or the residue.
        which: VectorKind,
        /// The expected length (`n/2`).
        expected: usize,
        /// The actual length supplied.
        actual: usize,
    },
}

impl core::fmt::Display for PacketError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PacketError::NonAudioPacketType { packet_type } => write!(
                f,
                "vorbis packet: §4.3.1 packet_type {packet_type} is not an audio packet"
            ),
            PacketError::BadModeNumber {
                mode_number,
                mode_count,
            } => write!(
                f,
                "vorbis packet: §4.3.1 mode_number {mode_number} >= mode_count {mode_count}"
            ),
            PacketError::EmptyModeList => write!(
                f,
                "vorbis packet: setup header carries no modes — cannot decode any audio packet"
            ),
            PacketError::UnexpectedEndOfPacket { stage } => write!(
                f,
                "vorbis packet: §4.3.1 end-of-packet while reading {stage} (fatal per §4.3.1 closing note)"
            ),
            PacketError::ChannelOutOfRange {
                step,
                channel,
                channels,
            } => write!(
                f,
                "vorbis packet: §4.3.3 coupling step {step} channel {channel} out of range \
                 (have {channels} channels)"
            ),
            PacketError::ChannelCountMismatch { floors, residues } => write!(
                f,
                "vorbis packet: §4.3.6 floor/residue channel count mismatch \
                 ({floors} floors vs {residues} residues)"
            ),
            PacketError::VectorLength {
                channel,
                which,
                expected,
                actual,
            } => write!(
                f,
                "vorbis packet: §4.3.6 channel {channel} {which} length {actual} != n/2 {expected}"
            ),
        }
    }
}

impl std::error::Error for PacketError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::{MappingCouplingStep, ModeHeader, VorbisSetupHeader};
    use oxideav_core::bits::{BitReaderLsb, BitWriterLsb};

    fn step(magnitude_channel: u8, angle_channel: u8) -> MappingCouplingStep {
        MappingCouplingStep {
            magnitude_channel,
            angle_channel,
        }
    }

    /// Build a minimal `VorbisSetupHeader` carrying just the supplied modes.
    /// The other lists are empty — the §4.3.1 reader only consults
    /// `setup.modes`, so the rest is irrelevant for these tests.
    fn setup_with_modes(modes: Vec<ModeHeader>) -> VorbisSetupHeader {
        VorbisSetupHeader {
            codebooks: Vec::new(),
            time_placeholders: Vec::new(),
            floors: Vec::new(),
            residues: Vec::new(),
            mappings: Vec::new(),
            modes,
            framing_flag: true,
        }
    }

    fn mode(blockflag: bool, mapping: u8) -> ModeHeader {
        ModeHeader {
            blockflag,
            windowtype: 0,
            transformtype: 0,
            mapping,
        }
    }

    // ---- §4.3.1 packet type, mode and window decode ----

    #[test]
    fn packet_header_single_mode_short_block_reads_packet_type_only() {
        // mode_count == 1: ilog(0) == 0, so step 2 reads ZERO bits and
        // mode_number resolves to 0. blockflag is false → short block, no
        // window flags read. Total bits read = 1 (just packet_type).
        let setup = setup_with_modes(vec![mode(false, 0)]);
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type = 0 (audio)
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let h = read_packet_header(&mut r, &setup, 64, 1024).unwrap();
        assert_eq!(h.mode_number, 0);
        assert!(!h.blockflag);
        assert_eq!(h.n, 64); // blocksize_0 chosen
        assert!(!h.previous_window_flag);
        assert!(!h.next_window_flag);
        // The reader is positioned right after the single bit we wrote.
        assert_eq!(r.bit_position(), 1);
    }

    #[test]
    fn packet_header_single_mode_long_block_still_reads_window_flags() {
        // Even with mode_count == 1 (zero mode-number bits), the long
        // block still reads previous_window_flag + next_window_flag.
        let setup = setup_with_modes(vec![mode(true, 0)]);
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type = 0
        w.write_u32(1, 1); // previous_window_flag = 1
        w.write_u32(0, 1); // next_window_flag = 0
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let h = read_packet_header(&mut r, &setup, 64, 1024).unwrap();
        assert_eq!(h.mode_number, 0);
        assert!(h.blockflag);
        assert_eq!(h.n, 1024); // blocksize_1 chosen
        assert!(h.previous_window_flag);
        assert!(!h.next_window_flag);
        assert_eq!(r.bit_position(), 3);
    }

    #[test]
    fn packet_header_two_modes_reads_one_mode_bit_short() {
        // mode_count == 2: ilog(1) == 1, so mode_number is 1 bit.
        // Select mode 1 (a short mode for this test).
        let setup = setup_with_modes(vec![mode(true, 0), mode(false, 0)]);
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        w.write_u32(1, 1); // mode_number = 1
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let h = read_packet_header(&mut r, &setup, 256, 2048).unwrap();
        assert_eq!(h.mode_number, 1);
        assert!(!h.blockflag);
        assert_eq!(h.n, 256);
        assert_eq!(r.bit_position(), 2);
    }

    #[test]
    fn packet_header_two_modes_long_reads_one_mode_bit_plus_window_flags() {
        // Pick mode 0 (long); should read packet_type + mode + 2 window flags = 4 bits.
        let setup = setup_with_modes(vec![mode(true, 0), mode(false, 0)]);
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        w.write_u32(0, 1); // mode_number = 0 → long
        w.write_u32(1, 1); // previous_window_flag = 1
        w.write_u32(1, 1); // next_window_flag = 1
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let h = read_packet_header(&mut r, &setup, 256, 2048).unwrap();
        assert_eq!(h.mode_number, 0);
        assert!(h.blockflag);
        assert_eq!(h.n, 2048);
        assert!(h.previous_window_flag);
        assert!(h.next_window_flag);
        assert_eq!(r.bit_position(), 4);
    }

    #[test]
    fn packet_header_three_modes_uses_two_mode_bits() {
        // mode_count == 3: ilog(2) == 2. Select mode 2 (a long mode).
        let setup = setup_with_modes(vec![mode(false, 0), mode(false, 0), mode(true, 0)]);
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        w.write_u32(2, 2); // mode_number = 2 (LSb-first: 10 reads as 2)
        w.write_u32(0, 1); // previous_window_flag = 0
        w.write_u32(0, 1); // next_window_flag = 0
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let h = read_packet_header(&mut r, &setup, 64, 1024).unwrap();
        assert_eq!(h.mode_number, 2);
        assert!(h.blockflag);
        assert_eq!(h.n, 1024);
        assert!(!h.previous_window_flag);
        assert!(!h.next_window_flag);
    }

    #[test]
    fn packet_header_rejects_non_audio_packet_type() {
        // packet_type bit == 1 is the §4.3 reject path; the decoder must
        // not attempt audio decode.
        let setup = setup_with_modes(vec![mode(false, 0)]);
        let mut w = BitWriterLsb::new();
        w.write_u32(1, 1); // packet_type = 1 (NOT audio)
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            read_packet_header(&mut r, &setup, 64, 1024),
            Err(PacketError::NonAudioPacketType { packet_type: 1 })
        );
    }

    #[test]
    fn packet_header_rejects_out_of_range_mode_number() {
        // mode_count == 3, ilog(2) == 2 bits, but mode_number bits = 11 → 3
        // (out of range).
        let setup = setup_with_modes(vec![mode(false, 0), mode(false, 0), mode(false, 0)]);
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        w.write_u32(3, 2); // mode_number = 3 (>= mode_count)
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            read_packet_header(&mut r, &setup, 64, 1024),
            Err(PacketError::BadModeNumber {
                mode_number: 3,
                mode_count: 3,
            })
        );
    }

    #[test]
    fn packet_header_rejects_empty_mode_list() {
        let setup = setup_with_modes(Vec::new());
        let bytes = [0u8; 1];
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            read_packet_header(&mut r, &setup, 64, 1024),
            Err(PacketError::EmptyModeList)
        );
    }

    #[test]
    fn packet_header_eof_on_packet_type() {
        // Empty packet — even step 1 can't run.
        let setup = setup_with_modes(vec![mode(false, 0)]);
        let bytes: [u8; 0] = [];
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            read_packet_header(&mut r, &setup, 64, 1024),
            Err(PacketError::UnexpectedEndOfPacket {
                stage: PacketHeaderStage::PacketType,
            })
        );
    }

    #[test]
    fn packet_header_eof_on_mode_number() {
        // BitWriterLsb pads to a byte boundary, so we cannot produce a
        // sub-byte stream. Force EOF by sizing mode_count so the
        // mode_number read width exceeds the bits remaining in a
        // single byte after the packet_type bit.
        //
        // mode_count == 130 → ilog(129) == 8 bits. After 1 packet_type
        // bit there are 7 bits left in the byte → EOF on the 8-bit
        // mode_number read.
        let many: Vec<_> = (0..130).map(|_| mode(false, 0)).collect();
        let setup_many = setup_with_modes(many);
        let bytes_one = [0x00u8];
        let mut r = BitReaderLsb::new(&bytes_one);
        assert_eq!(
            read_packet_header(&mut r, &setup_many, 64, 1024),
            Err(PacketError::UnexpectedEndOfPacket {
                stage: PacketHeaderStage::ModeNumber,
            })
        );
    }

    #[test]
    fn packet_header_eof_on_previous_window_flag() {
        // Single long mode, mode_bits == 0. Stream carries only the
        // packet_type bit; previous_window_flag read EOFs.
        // BitWriterLsb pads to a byte boundary, so we must use a
        // crafted byte-aligned stream.
        // Trick: use 9 modes → ilog(8) == 4 bits. After packet_type (1)
        // + mode (4) = 5 bits consumed of a 1-byte stream, leaving 3
        // bits; that's enough for 2 window flags. So instead use 17
        // modes → ilog(16) == 5 bits → 1 + 5 = 6 consumed, 2 left, just
        // enough for prev+next. Use 33 modes → ilog(32) == 6 → 1+6=7
        // consumed, 1 left → next_window_flag EOFs (covered by next
        // test). For prev EOF specifically: 65 modes → ilog(64) == 7 →
        // 1+7=8 consumed, 0 left → previous_window_flag EOFs.
        let mut many = Vec::new();
        // Pick mode 0 = long so we read window flags; need 65 modes.
        many.push(mode(true, 0));
        for _ in 1..65 {
            many.push(mode(false, 0));
        }
        let setup_many = setup_with_modes(many);
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        w.write_u32(0, 7); // mode_number = 0
                           // No further bits; finish() pads — but the bit reader knows
                           // about all 8 bits of the byte. There are exactly 0 bits left
                           // after the 8 consumed → previous_window_flag EOFs.
        let bytes = w.finish();
        // Sanity-check the writer produced exactly 1 byte.
        assert_eq!(bytes.len(), 1);
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            read_packet_header(&mut r, &setup_many, 64, 1024),
            Err(PacketError::UnexpectedEndOfPacket {
                stage: PacketHeaderStage::PreviousWindowFlag,
            })
        );
    }

    #[test]
    fn packet_header_eof_on_next_window_flag() {
        // 33 modes → ilog(32) == 6 mode bits. After packet_type (1) +
        // mode (6) + prev (1) = 8 bits consumed of one byte; next read
        // EOFs.
        let mut many = Vec::new();
        many.push(mode(true, 0));
        for _ in 1..33 {
            many.push(mode(false, 0));
        }
        let setup_many = setup_with_modes(many);
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        w.write_u32(0, 6); // mode_number = 0
        w.write_u32(1, 1); // previous_window_flag = 1
        let bytes = w.finish();
        assert_eq!(bytes.len(), 1);
        let mut r = BitReaderLsb::new(&bytes);
        assert_eq!(
            read_packet_header(&mut r, &setup_many, 64, 1024),
            Err(PacketError::UnexpectedEndOfPacket {
                stage: PacketHeaderStage::NextWindowFlag,
            })
        );
    }

    #[test]
    fn packet_header_build_window_short_block() {
        // Short block → vorbis_window is called with blockflag=false
        // and returns the symmetric short shape, independent of the
        // (ignored) window flags.
        let setup = setup_with_modes(vec![mode(false, 0)]);
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let h = read_packet_header(&mut r, &setup, 64, 1024).unwrap();
        let win = h.build_window(64).unwrap();
        let reference = crate::synthesis::vorbis_window(64, 64, false, false, false).unwrap();
        assert_eq!(win, reference);
    }

    #[test]
    fn packet_header_build_window_long_block_hybrid_left() {
        // Long block with previous_window_flag clear → left edge is the
        // 64-wide hybrid ramp. Confirm the driven window matches the
        // direct vorbis_window call with the same parameters.
        let setup = setup_with_modes(vec![mode(true, 0)]);
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        w.write_u32(0, 1); // previous_window_flag = 0
        w.write_u32(1, 1); // next_window_flag = 1
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let h = read_packet_header(&mut r, &setup, 64, 256).unwrap();
        assert!(h.blockflag);
        assert_eq!(h.n, 256);
        let win = h.build_window(64).unwrap();
        let reference = crate::synthesis::vorbis_window(256, 64, true, false, true).unwrap();
        assert_eq!(win, reference);
        // The first 48 bins are the zero lead-in (n/4 - blocksize_0/4 = 48).
        for &v in win.iter().take(48) {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn packet_header_build_window_propagates_window_error() {
        // Non-power-of-two blocksize_1 should propagate
        // WindowError::NotPowerOfTwo through build_window.
        let setup = setup_with_modes(vec![mode(true, 0)]);
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        w.write_u32(1, 1); // prev flag
        w.write_u32(1, 1); // next flag
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let h = read_packet_header(&mut r, &setup, 64, 100).unwrap();
        assert_eq!(h.n, 100);
        assert_eq!(
            h.build_window(64),
            Err(crate::synthesis::WindowError::NotPowerOfTwo { n: 100 })
        );
    }

    #[test]
    fn packet_header_blocksize_selection_follows_mode_blockflag() {
        // Build a setup where mode 0 is short and mode 1 is long; pick
        // each and check the resolved n.
        let setup = setup_with_modes(vec![mode(false, 0), mode(true, 0)]);
        // mode 0 → short → n == blocksize_0
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1);
        w.write_u32(0, 1);
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let h0 = read_packet_header(&mut r, &setup, 128, 1024).unwrap();
        assert_eq!(h0.n, 128);
        assert!(!h0.blockflag);

        // mode 1 → long → n == blocksize_1
        let mut w = BitWriterLsb::new();
        w.write_u32(0, 1); // packet_type
        w.write_u32(1, 1); // mode_number = 1
        w.write_u32(1, 1); // prev flag
        w.write_u32(1, 1); // next flag
        let bytes = w.finish();
        let mut r = BitReaderLsb::new(&bytes);
        let h1 = read_packet_header(&mut r, &setup, 128, 1024).unwrap();
        assert_eq!(h1.n, 1024);
        assert!(h1.blockflag);
    }

    // ---- §4.3.3 nonzero vector propagate ----

    #[test]
    fn nonzero_propagate_no_coupling_is_noop() {
        // Mono / no coupling: flags are untouched.
        let mut no_residue = vec![true, false];
        let before = no_residue.clone();
        nonzero_propagate(&mut no_residue, &[]).unwrap();
        assert_eq!(no_residue, before);
    }

    #[test]
    fn nonzero_propagate_pulls_unused_partner_back_in() {
        // Stereo coupling (mag=0, ang=1). Channel 0 used (false), channel 1
        // unused (true). §4.3.3: either used → both used. Channel 1 is
        // pulled back in.
        let mut no_residue = vec![false, true];
        nonzero_propagate(&mut no_residue, &[step(0, 1)]).unwrap();
        assert_eq!(no_residue, vec![false, false]);
    }

    #[test]
    fn nonzero_propagate_used_angle_pulls_in_magnitude() {
        // Symmetric: magnitude unused, angle used → both used.
        let mut no_residue = vec![true, false];
        nonzero_propagate(&mut no_residue, &[step(0, 1)]).unwrap();
        assert_eq!(no_residue, vec![false, false]);
    }

    #[test]
    fn nonzero_propagate_both_unused_stays_unused() {
        // Both members unused → neither "false" → flags untouched (both
        // stay true / unused).
        let mut no_residue = vec![true, true];
        nonzero_propagate(&mut no_residue, &[step(0, 1)]).unwrap();
        assert_eq!(no_residue, vec![true, true]);
    }

    #[test]
    fn nonzero_propagate_both_used_stays_used() {
        let mut no_residue = vec![false, false];
        nonzero_propagate(&mut no_residue, &[step(0, 1)]).unwrap();
        assert_eq!(no_residue, vec![false, false]);
    }

    #[test]
    fn nonzero_propagate_chained_steps_cascade() {
        // Two coupling steps sharing channel 1: (0,1) then (1,2). Channel 0
        // is used; everything else unused. Step (0,1) pulls in 1; the loop
        // then sees step (1,2) with channel 1 now used (false) → pulls in
        // channel 2. The cascade reaches channel 2 through ascending order.
        let mut no_residue = vec![false, true, true];
        nonzero_propagate(&mut no_residue, &[step(0, 1), step(1, 2)]).unwrap();
        assert_eq!(no_residue, vec![false, false, false]);
    }

    #[test]
    fn nonzero_propagate_isolated_unused_channel_survives() {
        // 5.1-style: channels 0/1 coupled and used, channel 2 (LFE on its
        // own submap, no coupling) is unused and stays unused.
        let mut no_residue = vec![false, true, true];
        nonzero_propagate(&mut no_residue, &[step(0, 1)]).unwrap();
        assert_eq!(no_residue, vec![false, false, true]);
    }

    #[test]
    fn nonzero_propagate_rejects_out_of_range_channel() {
        let mut no_residue = vec![false];
        assert_eq!(
            nonzero_propagate(&mut no_residue, &[step(0, 1)]),
            Err(PacketError::ChannelOutOfRange {
                step: 0,
                channel: 1,
                channels: 1,
            })
        );
    }

    #[test]
    fn nonzero_propagate_rejects_out_of_range_magnitude() {
        let mut no_residue = vec![false, false];
        assert_eq!(
            nonzero_propagate(&mut no_residue, &[step(5, 1)]),
            Err(PacketError::ChannelOutOfRange {
                step: 0,
                channel: 5,
                channels: 2,
            })
        );
    }

    // ---- §4.3.6 dot product ----

    #[test]
    fn dot_product_is_elementwise() {
        let floor = [2.0, 3.0, 4.0, 0.5];
        let residue = [10.0, -1.0, 0.0, 8.0];
        let mut spectrum = [0.0; 4];
        dot_product(&floor, &residue, &mut spectrum);
        assert_eq!(spectrum, [20.0, -3.0, 0.0, 4.0]);
    }

    #[test]
    fn dot_product_handles_negative_floor_and_residue() {
        // Floor values are linear-domain magnitudes (always >= 0 from the
        // inverse-dB table) but the function is a pure product, so it
        // honours sign on either operand.
        let floor = [1.5, 2.0];
        let residue = [-4.0, -2.0];
        let mut spectrum = [0.0; 2];
        dot_product(&floor, &residue, &mut spectrum);
        assert_eq!(spectrum, [-6.0, -4.0]);
    }

    #[test]
    #[should_panic(expected = "floor/residue length mismatch")]
    fn dot_product_rejects_residue_length_mismatch() {
        let floor = [1.0, 2.0];
        let residue = [1.0];
        let mut spectrum = [0.0; 2];
        dot_product(&floor, &residue, &mut spectrum);
    }

    #[test]
    #[should_panic(expected = "floor/spectrum length mismatch")]
    fn dot_product_rejects_spectrum_length_mismatch() {
        let floor = [1.0, 2.0];
        let residue = [1.0, 2.0];
        let mut spectrum = [0.0; 3];
        dot_product(&floor, &residue, &mut spectrum);
    }

    // ---- §4.3.6 driver across channels ----

    #[test]
    fn dot_product_all_two_used_channels() {
        let floors = vec![Some(vec![2.0, 3.0]), Some(vec![1.0, 4.0])];
        let residues = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let out = dot_product_all(&floors, &residues, 2).unwrap();
        assert_eq!(out, vec![vec![10.0, 18.0], vec![7.0, 32.0]]);
    }

    #[test]
    fn dot_product_all_unused_channel_is_zeroed() {
        // Channel 1's floor is None (unused, not pulled in by coupling) →
        // its spectrum is the all-zero vector regardless of any residue
        // content, and the residue vector for that channel is not read.
        let floors = vec![Some(vec![2.0, 3.0]), None];
        let residues = vec![vec![5.0, 6.0], vec![99.0, 99.0]];
        let out = dot_product_all(&floors, &residues, 2).unwrap();
        assert_eq!(out, vec![vec![10.0, 18.0], vec![0.0, 0.0]]);
    }

    #[test]
    fn dot_product_all_all_unused_returns_all_zero() {
        let floors = vec![None, None];
        let residues = vec![vec![], vec![]];
        let out = dot_product_all(&floors, &residues, 3).unwrap();
        assert_eq!(out, vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]);
    }

    #[test]
    fn dot_product_all_rejects_channel_count_mismatch() {
        let floors = vec![Some(vec![1.0])];
        let residues = vec![vec![1.0], vec![2.0]];
        assert_eq!(
            dot_product_all(&floors, &residues, 1),
            Err(PacketError::ChannelCountMismatch {
                floors: 1,
                residues: 2,
            })
        );
    }

    #[test]
    fn dot_product_all_rejects_short_floor_curve() {
        let floors = vec![Some(vec![1.0, 2.0])];
        let residues = vec![vec![1.0, 2.0, 3.0]];
        assert_eq!(
            dot_product_all(&floors, &residues, 3),
            Err(PacketError::VectorLength {
                channel: 0,
                which: VectorKind::Floor,
                expected: 3,
                actual: 2,
            })
        );
    }

    #[test]
    fn dot_product_all_rejects_short_residue_vector() {
        let floors = vec![Some(vec![1.0, 2.0, 3.0])];
        let residues = vec![vec![1.0, 2.0]];
        assert_eq!(
            dot_product_all(&floors, &residues, 3),
            Err(PacketError::VectorLength {
                channel: 0,
                which: VectorKind::Residue,
                expected: 3,
                actual: 2,
            })
        );
    }
}

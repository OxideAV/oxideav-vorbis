//! Vorbis I audio-packet decode driver stages that are fully specified by
//! the Vorbis I bitstream specification and require no inverse MDCT:
//! §4.3.3 "nonzero vector propagate" and §4.3.6 "dot product".
//!
//! The §4.3 audio-packet decode pipeline is a fixed sequence of stages
//! (§4.3 "Audio packet decode and synthesis"):
//!
//! 1. §4.3.1 packet type, mode and window decode (window builder:
//!    [`crate::synthesis::vorbis_window`]).
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
//! This module lands the two stages that are completely specified in the
//! Vorbis I spec text itself (no external citation) and that depend only
//! on the already-implemented floor and residue decoders: the §4.3.3
//! coupling-aware "this channel really is used after all" propagation, and
//! the §4.3.6 element-wise floor × residue spectrum product.
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

/// Errors that can arise while driving the §4.3.3 / §4.3.6 packet stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PacketError {
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
    use crate::setup::MappingCouplingStep;

    fn step(magnitude_channel: u8, angle_channel: u8) -> MappingCouplingStep {
        MappingCouplingStep {
            magnitude_channel,
            angle_channel,
        }
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

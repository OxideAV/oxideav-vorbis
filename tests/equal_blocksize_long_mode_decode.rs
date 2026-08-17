//! §4.3.1 equal-blocksize (`blocksize_0 == blocksize_1`) long-mode
//! decode — the geometry the r447 black-box differential campaign
//! found in the wild that no staged fixture covers.
//!
//! §4.2.2 requires only `blocksize_0 <= blocksize_1`, and real encoder
//! dialects exist that set them **equal** while still carrying two
//! modes — one with `blockflag = 0` and one with `blockflag = 1` — so
//! every long-mode packet reads the §4.3.1 step-4 window flags even
//! though there is no smaller block size to lap against. (The r447
//! campaign validated the crate's decode of such a real stream against
//! a reference decoder to ≤0.01 s16 LSB; this suite pins the same
//! geometry natively so CI holds it without the external binary.)
//!
//! The load-bearing §4.3.1/§1.3.2 property: a long block's hybrid
//! window edge replaces the slope with the *short* slope over the
//! short size — when `n0 == n1` that "short" slope **is** the full
//! slope, so all four `(previous_window_flag, next_window_flag)`
//! combinations must produce the identical window, and a long-mode
//! stream must decode **bit-identically** to the same spectra sent
//! through the short mode. This suite drives both streams through the
//! public [`StreamingDecoder::push_packet`] path and asserts exactly
//! that, plus the §4.3.8 lap geometry.
//!
//! Fully synthetic: no Ogg framing, no `docs/` fixtures, so it runs in
//! standalone CI.

use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::setup::{
    Floor1Header, FloorHeader, FloorKind, MappingHeader, MappingSubmap, ModeHeader, ResidueHeader,
    VorbisSetupHeader,
};
use oxideav_vorbis::streaming::{StreamingDecoder, StreamingFrame};
use oxideav_vorbis::{
    plan_partition_cascade, write_audio_packet, AudioChannelFloor, AudioDecoderState,
    AudioPacketHeader, Floor1Packet, ResidueVectorPlan,
};

use oxideav_core::bits::BitReaderLsb;

const N: usize = 512;
const HALF_N: usize = N / 2;
const PACKETS: usize = 6;

/// A Kraft-complete 1-D tessellation VQ value book: `2^length` entries
/// all at codeword length `length`, ladder `(e − half)·step`.
fn signed_value_book(length: u8, step: f32) -> VorbisCodebook {
    let entries: u32 = 1u32 << length;
    let half = entries / 2;
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::Tessellation {
            minimum_value: -(half as f32) * step,
            delta_value: step,
            value_bits: 8,
            sequence_p: false,
            multiplicands: (0..entries).collect(),
        },
    }
}

/// A balanced 1-D scalar classbook (no VQ lookup).
fn classbook(entries: u32, length: u8) -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
}

/// The two-mode equal-blocksize mono setup: flat floor-1, one format-1
/// residue over the whole spectrum, one mapping, and — the point of
/// the suite — `modes[0].blockflag = false`, `modes[1].blockflag =
/// true`, both on the same mapping, exactly the shape the campaign's
/// wild stream used.
fn equal_blocksize_setup(
    cb: VorbisCodebook,
    coarse: VorbisCodebook,
    fine: VorbisCodebook,
) -> VorbisSetupHeader {
    let floor = FloorHeader {
        floor_type: 1,
        kind: FloorKind::Type1(Floor1Header {
            partitions: 0,
            partition_class_list: Vec::new(),
            classes: Vec::new(),
            multiplier: 1,
            rangebits: 4,
            x_list: Vec::new(),
        }),
    };
    let mut stages: [Option<u8>; 8] = Default::default();
    stages[0] = Some(1);
    stages[1] = Some(2);
    let residue = ResidueHeader {
        residue_type: 1,
        residue_begin: 0,
        residue_end: HALF_N as u32,
        partition_size: HALF_N as u32,
        classifications: 1,
        classbook: 0,
        cascade: vec![(1 << 0) | (1 << 1)],
        books: vec![stages],
    };
    VorbisSetupHeader {
        codebooks: vec![cb, coarse, fine],
        time_placeholders: Vec::new(),
        floors: vec![floor],
        residues: vec![residue],
        mappings: vec![MappingHeader {
            mapping_type: 0,
            submaps: 1,
            coupling: Vec::new(),
            mux: Vec::new(),
            submap_configs: vec![MappingSubmap {
                time_placeholder: 0,
                floor: 0,
                residue: 0,
            }],
        }],
        modes: vec![
            ModeHeader {
                blockflag: false,
                windowtype: 0,
                transformtype: 0,
                mapping: 0,
            },
            ModeHeader {
                blockflag: true,
                windowtype: 0,
                transformtype: 0,
                mapping: 0,
            },
        ],
        framing_flag: true,
    }
}

/// Per-packet synthetic spectrum: a few tones whose bins and
/// amplitudes shift with the packet index so consecutive frames
/// genuinely differ across the overlap-add.
fn packet_spectrum(p: usize) -> Vec<f32> {
    (0..HALF_N)
        .map(|k| {
            let mut v = 0.0f32;
            if k == 3 + p {
                v += 0.9 - 0.07 * p as f32;
            }
            if k == 20 + 2 * p {
                v -= 0.5;
            }
            if k == 100 {
                v += 0.25 + 0.05 * p as f32;
            }
            v
        })
        .collect()
}

/// Serialise the `PACKETS` test spectra as §4.3 audio packets, all
/// through the given mode. For the long mode the §4.3.1 window flags
/// cycle through all four combinations across the stream.
fn build_packets(setup: &VorbisSetupHeader, mode_number: u32, blockflag: bool) -> Vec<Vec<u8>> {
    // Flat floor at table index 255 → F = 1.0 (multiplier 1), so the
    // residue carries the spectrum verbatim.
    let coarse = match &setup.codebooks[1].lookup {
        VqLookup::Tessellation { delta_value, .. } => *delta_value,
        _ => unreachable!(),
    };
    let fine = match &setup.codebooks[2].lookup {
        VqLookup::Tessellation { delta_value, .. } => *delta_value,
        _ => unreachable!(),
    };
    assert!(coarse > fine, "cascade steps sized coarse over fine");

    (0..PACKETS)
        .map(|p| {
            let x = packet_spectrum(p);
            let mut refs: [Option<&VorbisCodebook>; 8] = Default::default();
            refs[0] = Some(&setup.codebooks[1]);
            refs[1] = Some(&setup.codebooks[2]);
            let entries =
                plan_partition_cascade(&x, &refs, 1, HALF_N as u32).expect("residue cascade plans");
            let floors = vec![AudioChannelFloor::Type1(Floor1Packet {
                nonzero: true,
                floor1_y: vec![255, 255],
                partition_cvals: Vec::new(),
            })];
            let residue_plans = vec![vec![ResidueVectorPlan {
                classifications: vec![0],
                partition_entries: vec![entries],
            }]];
            let header = AudioPacketHeader {
                mode_number,
                blockflag,
                n: N,
                // Cycle all four §4.3.1 window-flag combinations across
                // the long-mode stream (ignored by the writer for the
                // short mode).
                previous_window_flag: blockflag && (p % 2 == 1),
                next_window_flag: blockflag && (p % 4 >= 2),
            };
            write_audio_packet(&header, setup, N, N, 1, &floors, &residue_plans)
                .expect("audio packet serialises")
        })
        .collect()
}

/// Decode a packet stream through the public streaming path, asserting
/// the §4.3.8 lap geometry, and return the concatenated mono PCM.
fn decode_stream(setup: &VorbisSetupHeader, packets: &[Vec<u8>]) -> Vec<f32> {
    let state = AudioDecoderState::new(setup).expect("decoder state builds");
    let mut decoder = StreamingDecoder::new(1, N, N, 1.0);
    let mut pcm = Vec::new();
    for (i, packet) in packets.iter().enumerate() {
        let mut reader = BitReaderLsb::new(packet);
        match decoder
            .push_packet(&mut reader, setup, &state)
            .expect("packet decodes")
        {
            StreamingFrame::Primed { .. } => {
                assert_eq!(i, 0, "only the first packet may prime");
            }
            StreamingFrame::Pcm {
                per_channel_pcm, ..
            } => {
                assert!(i > 0, "the first packet must prime, not emit");
                assert_eq!(per_channel_pcm.len(), 1);
                // §4.3.8: prev_n/4 + cur_n/4 with all blocks length N.
                assert_eq!(
                    per_channel_pcm[0].len(),
                    N / 2,
                    "equal-blocksize lap must be N/2"
                );
                pcm.extend_from_slice(&per_channel_pcm[0]);
            }
        }
    }
    pcm
}

/// The long-mode equal-blocksize stream decodes bit-identically to the
/// same spectra sent through the short mode: with `n0 == n1` the
/// §1.3.2 hybrid slopes coincide with the full slope for every
/// window-flag combination, so `blockflag` must not change one sample.
#[test]
fn long_mode_equals_short_mode_when_blocksizes_are_equal() {
    let cb = classbook(2, 1);
    let coarse = signed_value_book(6, 0.05);
    let fine = signed_value_book(6, 0.05 / 8.0);
    let setup = equal_blocksize_setup(cb, coarse, fine);

    let short_packets = build_packets(&setup, 0, false);
    let long_packets = build_packets(&setup, 1, true);

    let short_pcm = decode_stream(&setup, &short_packets);
    let long_pcm = decode_stream(&setup, &long_packets);

    assert_eq!(short_pcm.len(), (PACKETS - 1) * (N / 2));
    assert_eq!(short_pcm.len(), long_pcm.len());
    let energy: f64 = short_pcm.iter().map(|&v| (v as f64) * (v as f64)).sum();
    assert!(energy > 1.0e-3, "decode must produce nonzero audio");
    for (i, (s, l)) in short_pcm.iter().zip(&long_pcm).enumerate() {
        assert!(
            s.to_bits() == l.to_bits(),
            "sample {i}: short-mode {s} != long-mode {l} — equal-blocksize \
             long mode must decode bit-identically",
        );
    }
}

/// The long-mode packets are genuinely longer on the wire (the §4.3.1
/// prelude carries the two window flags), so the equality above is not
/// vacuous — the two streams differ as bitstreams and only converge in
/// PCM.
#[test]
fn long_mode_packets_carry_the_window_flag_bits() {
    let cb = classbook(2, 1);
    let coarse = signed_value_book(6, 0.05);
    let fine = signed_value_book(6, 0.05 / 8.0);
    let setup = equal_blocksize_setup(cb, coarse, fine);

    let short_packets = build_packets(&setup, 0, false);
    let long_packets = build_packets(&setup, 1, true);
    for (p, (s, l)) in short_packets.iter().zip(&long_packets).enumerate() {
        assert_ne!(
            s, l,
            "packet {p}: long-mode bitstream must differ from short-mode"
        );
    }
}

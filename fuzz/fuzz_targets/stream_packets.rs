//! §4.3 audio-packet robustness behind a valid setup header.
//!
//! Whole-stream fuzzing (`decode` target) rarely threads the §4.2
//! header gates with random bytes, so the §4.3 entropy decode sees
//! little of the budget there. This target fixes a small valid
//! two-mode setup (built once with the crate's own setup types) and
//! feeds *arbitrary* packet bodies through the public
//! [`StreamingDecoder::push_packet`] path — driving the §4.3.1
//! prelude, floor-1 body decode, §8.6.2 residue entropy decode, the
//! §4.3.7 IMDCT and the §4.3.8 overlap-add with attacker-controlled
//! bits. Contract: a typed `StreamingError` or a decoded frame —
//! never a panic.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::setup::{
    Floor1Header, FloorHeader, FloorKind, MappingHeader, MappingSubmap, ModeHeader, ResidueHeader,
    VorbisSetupHeader,
};
use oxideav_vorbis::streaming::StreamingDecoder;
use oxideav_vorbis::AudioDecoderState;
use std::sync::OnceLock;

const N0: usize = 256;
const N1: usize = 2048;

fn value_book(length: u8, step: f32) -> VorbisCodebook {
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

fn setup() -> &'static (VorbisSetupHeader, AudioDecoderState) {
    static SETUP: OnceLock<(VorbisSetupHeader, AudioDecoderState)> = OnceLock::new();
    SETUP.get_or_init(|| {
        let classbook = VorbisCodebook {
            dimensions: 1,
            entries: 2,
            codeword_lengths: vec![1, 1],
            lookup: VqLookup::None,
        };
        let mut stages: [Option<u8>; 8] = Default::default();
        stages[0] = Some(1);
        stages[1] = Some(2);
        let residue = ResidueHeader {
            residue_type: 1,
            residue_begin: 0,
            residue_end: (N0 / 2) as u32,
            partition_size: 32,
            classifications: 2,
            classbook: 0,
            cascade: vec![1, (1 << 0) | (1 << 1)],
            books: vec![stages, stages],
        };
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
        let header = VorbisSetupHeader {
            codebooks: vec![classbook, value_book(6, 0.05), value_book(6, 0.00625)],
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
        };
        let state = AudioDecoderState::new(&header).expect("fuzz setup state builds");
        (header, state)
    })
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > 1 << 14 {
        return;
    }
    let (header, state) = setup();
    // Split the input into up to 8 packet bodies: the first byte picks
    // the split geometry, the rest is payload.
    let n_packets = 1 + (data[0] & 7) as usize;
    let body = &data[1..];
    let chunk = body.len().div_ceil(n_packets).max(1);
    let mut decoder = StreamingDecoder::new(1, N0, N1, 1.0);
    for packet in body.chunks(chunk) {
        let mut reader = oxideav_core::bits::BitReaderLsb::new(packet);
        // Errors are expected constantly; panics never. A decoder that
        // errored keeps accepting further packets (the §4.3.1 contract
        // is per-packet discard).
        let _ = decoder.push_packet(&mut reader, header, state);
    }
});

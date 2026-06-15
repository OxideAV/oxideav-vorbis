//! End-to-end silence-decode integration test against the staged
//! `docs/audio/vorbis/fixtures/silence-stream/` corpus geometry.
//!
//! # Why this test exists
//!
//! The unit suites in `src/` validate each §4.3 stage in isolation and
//! the `src/streaming.rs` tests drive the [`StreamingDecoder`] with
//! hand-built *windowed* outcomes (`push_windowed`). Neither exercises
//! the full public bitstream → PCM path
//! ([`StreamingDecoder::push_packet`], which runs §4.3.1..§4.3.8 from raw
//! packet bits) against a packet sequence taken from a staged fixture.
//! This integration test closes that gap for the one fixture whose PCM
//! target is *fully determined independent of the still-deferred IMDCT
//! normalization scalar*: a pure-silence stream.
//!
//! # Why silence is the right anchor
//!
//! `docs/audio/vorbis/fixtures/silence-stream/` is
//! `anullsrc` (pure silence) encoded at q3 mono 44100 Hz (see that
//! fixture's `notes.md`). Its `trace.txt` shows **every** audio packet
//! with `packet_bytes=1`: the packet carries only the §4.3.1 header plus
//! the floor's leading `[nonzero]` flag, with no residue payload. Per
//! §4.3.2 step 6 an `'unused'` floor (the `[nonzero]` flag clear) sets
//! `no_residue[ch] = true`; §4.3.6 then yields the all-zero spectrum;
//! §4.3.7's IMDCT of the all-zero spectrum is the all-zero frame
//! (`imdct(0) = 0` for *any* scale — see `imdct.rs`); the §4.3.6 window
//! times zero stays zero; and §4.3.8 overlap-add of zero frames stays
//! zero. Every emitted PCM sample is therefore exactly `0.0`, matching
//! the fixture's silent `expected.wav`, and the result does **not**
//! depend on the unpinned normalization scalar (`imdct_scale`): `0 * α`
//! is `0` for every `α`. That makes silence the strongest end-to-end
//! assertion available while the post-IMDCT trace point that would pin
//! the scalar is still missing from the staged traces.
//!
//! # Fixture geometry anchored here
//!
//! The trace records the silence stream as mono, `blocksize_0 = 256`,
//! `blocksize_1 = 2048`, `mode_count = 2` (mode 0 = short / blockflag 0;
//! mode 1 = long / blockflag 1), with the first audio packet selecting
//! mode 0 and the steady-state packets selecting mode 1 with both window
//! flags set (`prev_window = next_window = 1`). The setup body built here
//! reproduces that geometry; the floor / residue configurations are kept
//! minimal-but-valid because the silence path never reads codebook
//! contents (the floor short-circuits before any residue codeword is
//! consumed).

use oxideav_core::bits::BitWriterLsb;
use oxideav_vorbis::{
    AudioDecoderState, Floor1Header, FloorHeader, FloorKind, MappingHeader, MappingSubmap,
    ModeHeader, ResidueHeader, StreamingDecoder, StreamingFrame, VorbisCodebook, VorbisSetupHeader,
    VqLookup,
};

const BLOCKSIZE_0: usize = 256;
const BLOCKSIZE_1: usize = 2048;

/// Single-entry scalar codebook used as the residue classbook. It is
/// never actually consulted on the silence path (the floor short-circuit
/// fires before residue decode reads anything), but the setup must hold a
/// structurally valid classbook for [`AudioDecoderState::new`] to build.
fn scalar_classbook() -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries: 1,
        codeword_lengths: vec![1],
        lookup: VqLookup::None,
    }
}

/// A floor type 1 header with zero partitions: floor1_values() == 2 (the
/// two implicit endpoints). On the silence path the leading `[nonzero]`
/// flag is clear, so the decoder returns `Unused` before reading any
/// endpoint amplitudes (§7.3 / §4.3.2 step 6).
fn floor_type1_unused_header() -> FloorHeader {
    FloorHeader {
        floor_type: 1,
        kind: FloorKind::Type1(Floor1Header {
            partitions: 0,
            partition_class_list: Vec::new(),
            classes: Vec::new(),
            multiplier: 4,
            rangebits: 4,
            x_list: Vec::new(),
        }),
    }
}

/// Residue with `begin == end == 0`: the §8.6.2 partition loop iterates
/// zero times, so even if it were reached the output would be all-zero.
fn residue_zero_header() -> ResidueHeader {
    ResidueHeader {
        residue_type: 0,
        residue_begin: 0,
        residue_end: 0,
        partition_size: 1,
        classifications: 1,
        classbook: 0,
        cascade: vec![0],
        books: vec![std::array::from_fn(|_| None)],
    }
}

fn mapping_mono_no_coupling(floor: u8, residue: u8) -> MappingHeader {
    MappingHeader {
        mapping_type: 0,
        submaps: 1,
        coupling: Vec::new(),
        mux: Vec::new(),
        submap_configs: vec![MappingSubmap {
            time_placeholder: 0,
            floor,
            residue,
        }],
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

/// The two-mode mono setup the silence trace records: mode 0 short
/// (blockflag 0), mode 1 long (blockflag 1), each routed through a single
/// submap with the unused floor + zero residue above.
fn silence_setup() -> VorbisSetupHeader {
    VorbisSetupHeader {
        codebooks: vec![scalar_classbook()],
        time_placeholders: Vec::new(),
        floors: vec![floor_type1_unused_header()],
        residues: vec![residue_zero_header()],
        mappings: vec![mapping_mono_no_coupling(0, 0)],
        modes: vec![mode(false, 0), mode(true, 0)],
        framing_flag: true,
    }
}

/// Build the single byte of a short-block (mode 0) silence packet.
///
/// Bit layout (LSB-first, mode_count == 2 → `ilog(1) == 1` mode bit):
/// `[packet_type=0]` `[mode_number=0]` `[floor nonzero=0]` → 3 bits, all
/// zero. A short block reads no window flags (§4.3.1 step 4b).
fn short_silence_packet() -> Vec<u8> {
    let mut w = BitWriterLsb::new();
    w.write_u32(0, 1); // packet_type = audio
    w.write_u32(0, 1); // mode_number = 0 (short)
    w.write_u32(0, 1); // floor type 1 [nonzero] = 0 → Unused
    w.finish()
}

/// Build the single byte of a long-block (mode 1) silence packet.
///
/// Bit layout (LSB-first): `[packet_type=0]` `[mode_number=1]`
/// `[previous_window_flag=1]` `[next_window_flag=1]` `[floor nonzero=0]`
/// → 5 bits. A long block reads both window flags (§4.3.1 step 4a).
fn long_silence_packet() -> Vec<u8> {
    let mut w = BitWriterLsb::new();
    w.write_u32(0, 1); // packet_type = audio
    w.write_u32(1, 1); // mode_number = 1 (long)
    w.write_u32(1, 1); // previous_window_flag = 1
    w.write_u32(1, 1); // next_window_flag = 1
    w.write_u32(0, 1); // floor type 1 [nonzero] = 0 → Unused
    w.finish()
}

/// Drive the documented silence packet sequence (one short priming
/// packet followed by long steady-state packets) through the full public
/// [`StreamingDecoder::push_packet`] path and assert every emitted PCM
/// sample is exactly silence, for any normalization scalar.
fn assert_silence_for_scale(imdct_scale: f32) {
    let setup = silence_setup();
    let state = AudioDecoderState::new(&setup).expect("silence setup builds decoder state");
    let mut dec = StreamingDecoder::new(1, BLOCKSIZE_0, BLOCKSIZE_1, imdct_scale);

    let short = short_silence_packet();
    let long = long_silence_packet();
    // The fixture's first audio packet is mode 0 (short); the rest are
    // mode 1 (long). One short priming packet then several long packets
    // exercises both block geometries and the short→long overlap-add
    // transition.
    let sequence: [&[u8]; 5] = [&short, &long, &long, &long, &long];

    let mut emitted_pcm_frames = 0usize;
    let mut total_samples = 0usize;

    for (idx, packet) in sequence.iter().enumerate() {
        let mut reader = oxideav_core::bits::BitReaderLsb::new(packet);
        let frame = dec
            .push_packet(&mut reader, &setup, &state)
            .unwrap_or_else(|e| panic!("packet {idx} decode failed at scale {imdct_scale}: {e}"));
        match frame {
            StreamingFrame::Primed { .. } => {
                assert_eq!(idx, 0, "only the first packet should prime");
            }
            StreamingFrame::Pcm {
                per_channel_pcm, ..
            } => {
                assert_eq!(per_channel_pcm.len(), 1, "mono stream emits one channel");
                emitted_pcm_frames += 1;
                for (ch, samples) in per_channel_pcm.iter().enumerate() {
                    total_samples += samples.len();
                    for (i, &s) in samples.iter().enumerate() {
                        assert_eq!(
                            s, 0.0,
                            "scale {imdct_scale} packet {idx} channel {ch} sample {i} \
                             expected silence, got {s}",
                        );
                    }
                }
            }
        }
    }

    // Sanity: the sequence must have produced finished PCM (otherwise the
    // assertion above is vacuously true). Four packets after priming each
    // emit one PCM frame.
    assert_eq!(
        emitted_pcm_frames, 4,
        "expected four PCM-emitting packets after priming",
    );
    assert!(
        total_samples > 0,
        "expected non-empty PCM output to assert silence against",
    );

    // Drain the stream-end tail; it too must be pure silence.
    if let Some(tails) = dec.finish() {
        for (ch, tail) in tails.iter().enumerate() {
            for (i, &s) in tail.iter().enumerate() {
                assert_eq!(
                    s, 0.0,
                    "scale {imdct_scale} finish tail channel {ch} sample {i} \
                     expected silence, got {s}",
                );
            }
        }
    }
}

#[test]
fn silence_stream_decodes_to_pure_silence() {
    // Unit scale — the bare kernel path.
    assert_silence_for_scale(1.0);
}

#[test]
fn silence_is_invariant_to_normalization_scalar() {
    // The whole point: silence is zero regardless of the still-deferred
    // IMDCT normalization scalar. Exercise several candidate values an
    // experimenter might try; all must yield identical (silent) output.
    for &scale in &[0.5f32, 2.0, 1.0 / 2048.0, 64.0, 1e-3] {
        assert_silence_for_scale(scale);
    }
}

#[test]
fn silence_pcm_return_lengths_follow_spec_4_3_8() {
    // Cross-check the §4.3.8 return-length formula (prev_n/4 + cur_n/4)
    // on the documented short→long→long… sequence, independent of the
    // sample values. This pins the geometry the silence assertion runs
    // over so a future overlap-add refactor can't silently shrink it.
    let setup = silence_setup();
    let state = AudioDecoderState::new(&setup).unwrap();
    let mut dec = StreamingDecoder::new(1, BLOCKSIZE_0, BLOCKSIZE_1, 1.0);

    let short = short_silence_packet();
    let long = long_silence_packet();

    // Packet 0: short, primes (no PCM).
    let mut r = oxideav_core::bits::BitReaderLsb::new(&short);
    assert!(dec.push_packet(&mut r, &setup, &state).unwrap().is_primed());

    // Packet 1: long after short → prev_n/4 + cur_n/4 = 256/4 + 2048/4
    // = 64 + 512 = 576.
    let mut r = oxideav_core::bits::BitReaderLsb::new(&long);
    let f = dec.push_packet(&mut r, &setup, &state).unwrap();
    assert_eq!(f.pcm().unwrap()[0].len(), BLOCKSIZE_0 / 4 + BLOCKSIZE_1 / 4);

    // Packet 2: long after long → 2048/4 + 2048/4 = 1024.
    let mut r = oxideav_core::bits::BitReaderLsb::new(&long);
    let f = dec.push_packet(&mut r, &setup, &state).unwrap();
    assert_eq!(f.pcm().unwrap()[0].len(), BLOCKSIZE_1 / 4 + BLOCKSIZE_1 / 4);
}

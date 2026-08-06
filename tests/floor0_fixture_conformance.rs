//! Structural conformance for the `mode-floor0-lsp` fixture — the
//! trace-substitute suite.
//!
//! Every other staged fixture ships a `trace.txt` that the
//! `audio_packet_trace_conformance` / `setup_header_trace_conformance`
//! suites replay event-for-event. The floor-0 fixture ships none: its
//! `notes.md` instead tabulates the whole stream layout exhaustively
//! and gives the audio packets as **closed forms** in the packet index
//! `k` ("Stream layout" / "Audio packet schedule" — the documented
//! reproduction recipe). This suite pins the crate's structural decode
//! against that documentation with the same granularity the trace
//! suites provide elsewhere:
//!
//! * the §4.2.2 identification and §4.2.4 setup headers,
//!   field-for-field (all five codebooks down to their multiplicand
//!   tables, the floor-0 config, the residue / mapping / mode tables);
//! * the RFC 3533 page table (header types, granule positions,
//!   segment counts, body sizes);
//! * every audio packet's §4.3.1 header decisions **and** its §6.2.2
//!   floor-0 payload (amplitude, booknumber, LSP entry run) and
//!   §8.6.2 residue symbol stream (classword entries, value-codeword
//!   entries), read bit-by-bit against the closed forms;
//! * the §6.2.3 curve: the crate's `Floor0Decoder::decode` of the real
//!   packet bits must reproduce, bit-for-bit, `render_curve` over the
//!   coefficients *predicted* by the closed form (entry run →
//!   `unpack_vector` → §6.2.2 `[last]` accumulation) — so a mismatch
//!   anywhere in the codebook/LSP pipeline fails loudly.
//!
//! Every codebook in the stream has `entries == 2^len` with uniform
//! codeword lengths, so the §3.2.1 canonical assignment maps entry `i`
//! to codeword `i` (MSb-first) — which is what lets this suite read
//! Huffman symbols with plain bit reads.
//!
//! # Standalone-CI skip
//!
//! As with every fixture suite: skips (without `#[ignore]`) when the
//! umbrella `docs/` submodule is absent.

use oxideav_core::bits::BitReaderLsb;
use oxideav_vorbis::floor0::{Floor0Curve, Floor0Decoder};
use oxideav_vorbis::packet::read_packet_header;
use oxideav_vorbis::setup::{FloorKind, VorbisSetupHeader};
use oxideav_vorbis::vq::unpack_vector;
use oxideav_vorbis::{parse_identification_header, parse_setup_header, VqLookup};

fn fixture_dir() -> String {
    format!(
        "{}/../../docs/audio/vorbis/fixtures/mode-floor0-lsp",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn fixture_available() -> bool {
    std::path::Path::new(&fixture_dir()).is_dir()
}

/// One parsed page header row of the notes.md "Ogg pages" table:
/// `(header_type, granule_position, page_sequence_no, segment_count,
/// body_len)`.
type PageRow = (u8, u64, u32, usize, usize);

/// Minimal RFC 3533 page-header walk over the raw file bytes,
/// returning one [`PageRow`] per page plus the reassembled packets.
fn pages_and_packets(data: &[u8]) -> (Vec<PageRow>, Vec<Vec<u8>>) {
    let mut pages = Vec::new();
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut pending: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    while pos + 27 <= data.len() {
        assert_eq!(&data[pos..pos + 4], b"OggS", "page sync at byte {pos}");
        let header_type = data[pos + 5];
        let granule = u64::from_le_bytes(data[pos + 6..pos + 14].try_into().unwrap());
        let seq = u32::from_le_bytes(data[pos + 18..pos + 22].try_into().unwrap());
        let seg_count = data[pos + 26] as usize;
        let seg_table = &data[pos + 27..pos + 27 + seg_count];
        let body_len: usize = seg_table.iter().map(|&l| l as usize).sum();
        pages.push((header_type, granule, seq, seg_count, body_len));
        let mut body = pos + 27 + seg_count;
        for &lace in seg_table {
            let l = lace as usize;
            pending.extend_from_slice(&data[body..body + l]);
            body += l;
            if l < 255 {
                packets.push(std::mem::take(&mut pending));
            }
        }
        pos = body;
    }
    assert!(pending.is_empty(), "no packet may span past the last page");
    (pages, packets)
}

/// Long-block predicate from the documented schedule: long iff
/// `(k / 4) % 2 == 1`, except packets 0 and 63 forced short.
fn is_long(k: usize) -> bool {
    k != 0 && k != 63 && (k / 4) % 2 == 1
}

/// Read one canonical codeword of `len` bits (uniform-length exactly
/// populated book: entry `i` ↔ codeword `i`, MSb-first).
fn read_codeword(r: &mut BitReaderLsb<'_>, len: u32) -> u32 {
    let mut e = 0u32;
    for _ in 0..len {
        e = (e << 1) | r.read_u32(1).expect("codeword bits available");
    }
    e
}

fn load() -> Option<(Vec<PageRow>, Vec<Vec<u8>>)> {
    if !fixture_available() {
        eprintln!("SKIP: docs/ fixtures submodule not checked out (standalone CI)");
        return None;
    }
    let ogg = std::fs::read(format!("{}/input.ogg", fixture_dir())).expect("input.ogg present");
    Some(pages_and_packets(&ogg))
}

fn parsed_setup(packets: &[Vec<u8>]) -> VorbisSetupHeader {
    let id = parse_identification_header(&packets[0]).expect("id header parses");
    parse_setup_header(&packets[2], id.audio_channels).expect("setup header parses")
}

#[test]
fn page_table_matches_the_documented_layout() {
    let Some((pages, packets)) = load() else {
        return;
    };
    // notes.md "Ogg pages": 6 pages — BOS id / comment+setup / 4 audio
    // pages of 16 packets each, EOS on the last.
    let expected: [(u8, u64, u32, usize, usize); 6] = [
        (0x02, 0, 0, 1, 30),
        (0x00, 0, 1, 2, 296),
        (0x00, 8640, 2, 16, 584),
        (0x00, 17856, 3, 16, 584),
        (0x00, 27072, 4, 16, 584),
        (0x04, 35840, 5, 16, 583),
    ];
    assert_eq!(pages.len(), expected.len(), "page count");
    for (i, (page, want)) in pages.iter().zip(&expected).enumerate() {
        assert_eq!(page, want, "page {i} (type, granule, seq, segs, body)");
    }
    assert_eq!(packets.len(), 3 + 64, "3 headers + 64 audio packets");
    for (i, pkt) in packets[3..].iter().enumerate() {
        assert!(
            pkt.len() == 36 || pkt.len() == 37,
            "audio packet {i}: {} bytes, documented 36 or 37",
            pkt.len()
        );
    }
}

#[test]
fn headers_match_the_documented_layout() {
    let Some((_, packets)) = load() else {
        return;
    };

    // §4.2.2 identification header table.
    let id = parse_identification_header(&packets[0]).expect("id header parses");
    assert_eq!(id.audio_channels, 1);
    assert_eq!(id.audio_sample_rate, 44_100);
    assert_eq!(id.bitrate_nominal, 32_000);
    assert_eq!(id.bitrate_maximum, 0);
    assert_eq!(id.bitrate_minimum, 0);
    assert_eq!(id.blocksize_0, 256);
    assert_eq!(id.blocksize_1, 2048);

    // Setup header: 189 bytes on page 1, five codebooks.
    assert_eq!(packets[2].len(), 189, "setup header byte size");
    let setup = parsed_setup(&packets);
    assert_eq!(setup.codebooks.len(), 5);
    assert_eq!(setup.time_placeholders, vec![0u16]);
    assert_eq!(setup.floors.len(), 1);
    assert_eq!(setup.residues.len(), 1);
    assert_eq!(setup.mappings.len(), 1);
    assert_eq!(setup.modes.len(), 2);
    assert!(setup.framing_flag);

    // Per-codebook table: (dims, entries, len, lookup fields).
    // Books 0/2/3/4 are lattice VQ over four multiplicands; book 1 is
    // the scalar residue classbook.
    struct Lattice {
        minimum: f32,
        delta: f32,
        sequence_p: bool,
        multiplicands: [u32; 4],
    }
    let lattices: [(usize, Lattice); 4] = [
        (
            0,
            Lattice {
                minimum: 0.0,
                delta: 1.0 / 64.0,
                sequence_p: true,
                multiplicands: [19, 21, 23, 25],
            },
        ),
        (
            2,
            Lattice {
                minimum: -0.5,
                delta: 1.0 / 32.0,
                sequence_p: false,
                multiplicands: [5, 13, 19, 27],
            },
        ),
        (
            3,
            Lattice {
                minimum: -0.5,
                delta: 1.0 / 32.0,
                sequence_p: false,
                multiplicands: [2, 11, 21, 30],
            },
        ),
        (
            4,
            Lattice {
                minimum: 0.0,
                delta: 1.0 / 64.0,
                sequence_p: true,
                multiplicands: [20, 22, 24, 25],
            },
        ),
    ];
    for (idx, want) in &lattices {
        let book = &setup.codebooks[*idx];
        assert_eq!(book.dimensions, 2, "book {idx} dims");
        assert_eq!(book.entries, 16, "book {idx} entries");
        assert!(
            book.codeword_lengths.iter().all(|&l| l == 4),
            "book {idx}: uniform 4-bit lengths"
        );
        match &book.lookup {
            VqLookup::Lattice {
                minimum_value,
                delta_value,
                value_bits,
                sequence_p,
                multiplicands,
            } => {
                assert_eq!(*minimum_value, want.minimum, "book {idx} minimum");
                assert_eq!(*delta_value, want.delta, "book {idx} delta");
                assert_eq!(*value_bits, 5, "book {idx} value_bits");
                assert_eq!(*sequence_p, want.sequence_p, "book {idx} sequence_p");
                assert_eq!(multiplicands, &want.multiplicands, "book {idx} scalars");
            }
            other => panic!("book {idx}: expected lattice lookup, got {other:?}"),
        }
    }
    let classbook = &setup.codebooks[1];
    assert_eq!(classbook.dimensions, 2);
    assert_eq!(classbook.entries, 4);
    assert!(classbook.codeword_lengths.iter().all(|&l| l == 2));
    assert!(matches!(classbook.lookup, VqLookup::None));

    // Floor 0 config (§6.2.1).
    assert_eq!(setup.floors[0].floor_type, 0);
    let FloorKind::Type0(f0) = &setup.floors[0].kind else {
        panic!("floor 0 expected");
    };
    assert_eq!(f0.order, 8);
    assert_eq!(f0.rate, 44_100);
    assert_eq!(f0.bark_map_size, 128);
    assert_eq!(f0.amplitude_bits, 8);
    assert_eq!(f0.amplitude_offset, 48);
    assert_eq!(f0.book_list, vec![0, 4, 0], "three slots, books 0/4/0");

    // Residue 0 config (§8.6.1): type 1, one-pass two-class cascade.
    let res = &setup.residues[0];
    assert_eq!(res.residue_type, 1);
    assert_eq!(res.residue_begin, 0);
    assert_eq!(res.residue_end, 128);
    assert_eq!(res.partition_size, 32);
    assert_eq!(res.classifications, 2);
    assert_eq!(res.classbook, 1);
    assert_eq!(res.cascade, vec![1, 1], "only cascade bit 0 set");
    assert_eq!(res.books[0][0], Some(2));
    assert_eq!(res.books[1][0], Some(3));
    for class in 0..2 {
        for pass in 1..8 {
            assert_eq!(res.books[class][pass], None, "no book past pass 0");
        }
    }

    // Mapping: type 0, one submap, no coupling, submap 0 → floor 0,
    // residue 0.
    let map = &setup.mappings[0];
    assert_eq!(map.mapping_type, 0);
    assert_eq!(map.submaps, 1);
    assert!(map.coupling.is_empty());
    // `submaps == 1` ⇒ no `mux[ch]` table on the wire (§4.2.4): every
    // channel implicitly uses submap 0.
    assert!(map.mux.is_empty());
    assert_eq!(map.submap_configs[0].floor, 0);
    assert_eq!(map.submap_configs[0].residue, 0);

    // Modes: 0 short / 1 long, both mapping 0, window/transform 0.
    for (i, mode) in setup.modes.iter().enumerate() {
        assert_eq!(mode.blockflag, i == 1, "mode {i} blockflag");
        assert_eq!(mode.windowtype, 0);
        assert_eq!(mode.transformtype, 0);
        assert_eq!(mode.mapping, 0);
    }
}

#[test]
fn audio_packets_follow_the_closed_form_schedule() {
    let Some((_, packets)) = load() else {
        return;
    };
    let setup = parsed_setup(&packets);
    let FloorKind::Type0(f0_header) = &setup.floors[0].kind else {
        panic!("floor 0 expected");
    };
    let floor0 =
        Floor0Decoder::new(f0_header, &setup.codebooks).expect("floor-0 decoder constructs");

    let audio = &packets[3..];
    assert_eq!(audio.len(), 64);
    for (k, pkt) in audio.iter().enumerate() {
        // --- §4.3.1 packet header: the documented schedule. ---
        let mut r = BitReaderLsb::new(pkt);
        let hdr = read_packet_header(&mut r, &setup, 256, 2048).expect("packet header parses");
        assert_eq!(hdr.blockflag, is_long(k), "packet {k} blockflag");
        assert_eq!(hdr.mode_number, u32::from(is_long(k)), "packet {k} mode");
        assert_eq!(hdr.n, if is_long(k) { 2048 } else { 256 }, "packet {k} n");
        if is_long(k) {
            assert_eq!(
                hdr.previous_window_flag,
                is_long(k - 1),
                "packet {k} prev_window"
            );
            assert_eq!(
                hdr.next_window_flag,
                is_long(k + 1),
                "packet {k} next_window"
            );
        }

        // --- §6.2.2 floor-0 payload: amplitude, booknumber, LSP run. ---
        let amplitude = r.read_u32(8).expect("amplitude bits");
        assert_eq!(
            amplitude,
            (60 + (k as u32 * 37) % 22),
            "packet {k} amplitude"
        );
        let booknumber = r.read_u32(2).expect("booknumber bits");
        assert_eq!(booknumber, (k % 3) as u32, "packet {k} booknumber");
        let lsp_book = &setup.codebooks[f0_header.book_list[booknumber as usize] as usize];
        // Four vectors of dimension 2 = exactly floor0_order scalars,
        // with the §6.2.2 [last] accumulation across vectors.
        let mut coefficients: Vec<f32> = Vec::with_capacity(8);
        let mut last = 0.0f32;
        for i in 0..4 {
            let entry = read_codeword(&mut r, 4);
            assert_eq!(entry, ((k * 3 + i * 5) % 16) as u32, "packet {k} LSP {i}");
            let mut vec = unpack_vector(lsp_book, entry).expect("entry unpacks");
            for v in &mut vec {
                *v += last;
            }
            last = *vec.last().expect("dimension 2");
            coefficients.extend_from_slice(&vec);
        }

        // --- §8.6.2 residue symbol stream (type 1, one pass, four
        // partitions of 32, two classwords covering two partitions
        // each, 16 two-dimensional value codewords per partition). ---
        for half in 0..2usize {
            let classword = read_codeword(&mut r, 2);
            let want = if half == 0 { k % 4 } else { (k + 2) % 4 };
            assert_eq!(classword, want as u32, "packet {k} classword {half}");
            for p in (2 * half)..(2 * half + 2) {
                for j in 0..16usize {
                    let entry = read_codeword(&mut r, 4);
                    assert_eq!(
                        entry,
                        ((k * 7 + p * 5 + j * 3) % 16) as u32,
                        "packet {k} residue partition {p} vector {j}"
                    );
                }
            }
        }

        // --- §6.2.3 curve: the crate's decode of the real bits must
        // equal render_curve over the closed-form coefficients. ---
        let mut r2 = BitReaderLsb::new(pkt);
        let _ = read_packet_header(&mut r2, &setup, 256, 2048).expect("re-parse");
        let decoded = floor0.decode(&mut r2, hdr.n / 2);
        let Floor0Curve::Curve(curve) = decoded else {
            panic!("packet {k}: non-zero amplitude must yield a curve");
        };
        let rendered = floor0.render_curve(amplitude, &coefficients, hdr.n / 2);
        assert_eq!(
            curve, rendered,
            "packet {k}: decoded §6.2.3 curve must equal the closed-form render"
        );
    }
}

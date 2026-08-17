//! Panic-freedom of the whole-stream decode entry and the direct §4.2
//! header parsers on arbitrary bytes.
//!
//! Contract: every call below returns a `Result` (or a classification)
//! for malformed input — no panic, no abort, no unbounded allocation
//! from attacker-controlled length fields.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Cap the pipeline input so a single iteration stays cheap; the
    // parsers below see the uncapped slice (they are O(len)).
    if data.len() <= 1 << 16 {
        let _ = oxideav_vorbis::oggfile::decode_ogg_to_pcm(data);
        let _ = oxideav_vorbis::oggfile::ogg_packets(data);
    }
    let _ = oxideav_vorbis::packet_kind::classify_packet(data);
    let _ = oxideav_vorbis::identification::parse_identification_header(data);
    let _ = oxideav_vorbis::comment::parse_comment_header(data);
    // The setup parser needs the channel count from the (separate)
    // identification header; sweep representative and hostile values.
    for ch in [1u8, 2, 6, 8, 255] {
        let _ = oxideav_vorbis::setup::parse_setup_header(data, ch);
    }
});

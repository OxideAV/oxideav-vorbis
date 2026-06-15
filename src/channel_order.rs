//! Vorbis I §4.3.9 — output channel order (mapping type 0).
//!
//! The §4.3 audio-packet driver ([`crate::audio`]) produces one
//! time-domain PCM vector per channel **in encoded-bitstream order**:
//! channel `0` is the first channel in the stream, channel `1` the
//! second, and so on. §4.3.9 fixes what physical speaker location each
//! of those bitstream positions denotes, for the standard mapping
//! type 0 layouts the Vorbis I specification defines for 1..=8
//! channels.
//!
//! This module turns the bare channel count into that documented
//! layout. It is a *presentation* concern — the decode driver itself
//! never reorders, exactly as §4.3.9 notes: *"These channel orderings
//! refer to order within the encoded stream. It is naturally possible
//! for a decoder to produce output with channels in any order. Any
//! such decoder should explicitly document channel reordering
//! behavior."* A consumer that wants a different physical ordering
//! (e.g. WAVE/`WAVEFORMATEXTENSIBLE` order, which differs from the
//! Vorbis order for >2 channels) looks each Vorbis-stream channel up
//! through [`speaker_layout`] and permutes accordingly.
//!
//! # The mapping (§4.3.9, mapping type 0)
//!
//! Verbatim speaker assignment from the Vorbis I spec, where the index
//! into each slice is the encoded-stream channel index:
//!
//! | channels | stream channel order |
//! |----------|----------------------|
//! | 1        | mono |
//! | 2        | left, right |
//! | 3        | left, center, right |
//! | 4        | front left, front right, rear left, rear right |
//! | 5        | front left, center, front right, rear left, rear right |
//! | 6 (5.1)  | front left, center, front right, rear left, rear right, LFE |
//! | 7 (6.1)  | front left, center, front right, side left, side right, rear center, LFE |
//! | 8 (7.1)  | front left, center, front right, side left, side right, rear left, rear right, LFE |
//!
//! For counts greater than eight the spec leaves channel use and order
//! "defined by the application"; this module reports
//! [`Speaker::Unspecified`] for every position in that case rather than
//! inventing an ordering. A channel count of zero is invalid per
//! §4.2.2 and yields an empty layout.

/// A physical speaker location named by Vorbis I §4.3.9 mapping type 0.
///
/// The variants cover exactly the speaker locations the spec assigns
/// across its 1..=8-channel layouts. [`Speaker::Unspecified`] is used
/// for channel counts above eight, which §4.3.9 defines as
/// application-specified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Speaker {
    /// The single channel of a monophonic stream.
    Mono,
    /// Front-left speaker (also the "left" of a stereo pair).
    FrontLeft,
    /// Front-center speaker.
    FrontCenter,
    /// Front-right speaker (also the "right" of a stereo pair).
    FrontRight,
    /// Rear-left speaker (quad/5.x "rear left").
    RearLeft,
    /// Rear-right speaker (quad/5.x "rear right").
    RearRight,
    /// Rear-center speaker (6.1 only).
    RearCenter,
    /// Side-left speaker (6.1 / 7.1).
    SideLeft,
    /// Side-right speaker (6.1 / 7.1).
    SideRight,
    /// Low-frequency-effects (subwoofer) channel.
    Lfe,
    /// Channel count > 8: §4.3.9 leaves ordering to the application.
    Unspecified,
}

impl Speaker {
    /// Human-readable label matching the §4.3.9 wording.
    pub const fn label(self) -> &'static str {
        match self {
            Speaker::Mono => "mono",
            Speaker::FrontLeft => "front left",
            Speaker::FrontCenter => "center",
            Speaker::FrontRight => "front right",
            Speaker::RearLeft => "rear left",
            Speaker::RearRight => "rear right",
            Speaker::RearCenter => "rear center",
            Speaker::SideLeft => "side left",
            Speaker::SideRight => "side right",
            Speaker::Lfe => "LFE",
            Speaker::Unspecified => "application-defined",
        }
    }
}

// §4.3.9 mapping-type-0 layouts, indexed by `channels - 1` for
// `channels` in 1..=8. Each inner slice is the encoded-stream channel
// order: position `i` is the speaker carried by bitstream channel `i`.
const LAYOUT_1: &[Speaker] = &[Speaker::Mono];
const LAYOUT_2: &[Speaker] = &[Speaker::FrontLeft, Speaker::FrontRight];
const LAYOUT_3: &[Speaker] = &[
    Speaker::FrontLeft,
    Speaker::FrontCenter,
    Speaker::FrontRight,
];
const LAYOUT_4: &[Speaker] = &[
    Speaker::FrontLeft,
    Speaker::FrontRight,
    Speaker::RearLeft,
    Speaker::RearRight,
];
const LAYOUT_5: &[Speaker] = &[
    Speaker::FrontLeft,
    Speaker::FrontCenter,
    Speaker::FrontRight,
    Speaker::RearLeft,
    Speaker::RearRight,
];
const LAYOUT_6: &[Speaker] = &[
    Speaker::FrontLeft,
    Speaker::FrontCenter,
    Speaker::FrontRight,
    Speaker::RearLeft,
    Speaker::RearRight,
    Speaker::Lfe,
];
const LAYOUT_7: &[Speaker] = &[
    Speaker::FrontLeft,
    Speaker::FrontCenter,
    Speaker::FrontRight,
    Speaker::SideLeft,
    Speaker::SideRight,
    Speaker::RearCenter,
    Speaker::Lfe,
];
const LAYOUT_8: &[Speaker] = &[
    Speaker::FrontLeft,
    Speaker::FrontCenter,
    Speaker::FrontRight,
    Speaker::SideLeft,
    Speaker::SideRight,
    Speaker::RearLeft,
    Speaker::RearRight,
    Speaker::Lfe,
];

/// Return the §4.3.9 mapping-type-0 speaker layout for `channels`
/// encoded channels, as a slice in **encoded-stream channel order**.
///
/// `layout[i]` is the speaker location that encoded-stream channel `i`
/// carries. For `channels` in 1..=8 this is the spec-defined static
/// layout; for `channels == 0` (invalid per §4.2.2) the slice is empty;
/// for `channels > 8` the spec defers to the application, so this
/// returns [`None`] rather than guessing.
///
/// The borrowed-slice form is allocation-free and covers every
/// spec-fixed layout. Use [`speaker_at`] for a single position or when
/// the count may exceed eight.
pub fn speaker_layout(channels: u8) -> Option<&'static [Speaker]> {
    Some(match channels {
        1 => LAYOUT_1,
        2 => LAYOUT_2,
        3 => LAYOUT_3,
        4 => LAYOUT_4,
        5 => LAYOUT_5,
        6 => LAYOUT_6,
        7 => LAYOUT_7,
        8 => LAYOUT_8,
        _ => return None,
    })
}

/// Return the speaker location carried by encoded-stream channel
/// `index` of a `channels`-channel mapping-type-0 stream.
///
/// * For `channels` in 1..=8 and `index < channels`, this is the
///   spec-fixed §4.3.9 location.
/// * For `channels > 8` (application-defined) every in-range index is
///   [`Speaker::Unspecified`].
/// * If `index >= channels` (out of range) the result is [`None`].
pub fn speaker_at(channels: u8, index: u8) -> Option<Speaker> {
    if channels == 0 || index >= channels {
        return None;
    }
    match speaker_layout(channels) {
        Some(layout) => Some(layout[index as usize]),
        // channels > 8: in range but application-defined.
        None => Some(Speaker::Unspecified),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_layout() {
        assert_eq!(speaker_layout(1), Some(LAYOUT_1));
        assert_eq!(speaker_at(1, 0), Some(Speaker::Mono));
        assert_eq!(speaker_at(1, 1), None);
    }

    #[test]
    fn stereo_is_left_right() {
        let l = speaker_layout(2).unwrap();
        assert_eq!(l, &[Speaker::FrontLeft, Speaker::FrontRight]);
        assert_eq!(speaker_at(2, 0), Some(Speaker::FrontLeft));
        assert_eq!(speaker_at(2, 1), Some(Speaker::FrontRight));
    }

    #[test]
    fn three_channel_has_center_in_middle() {
        // §4.3.9: left, center, right — note center is index 1, NOT a
        // trailing channel as some other codecs order it.
        let l = speaker_layout(3).unwrap();
        assert_eq!(
            l,
            &[
                Speaker::FrontLeft,
                Speaker::FrontCenter,
                Speaker::FrontRight
            ]
        );
    }

    #[test]
    fn quad_has_no_center_no_lfe() {
        let l = speaker_layout(4).unwrap();
        assert_eq!(
            l,
            &[
                Speaker::FrontLeft,
                Speaker::FrontRight,
                Speaker::RearLeft,
                Speaker::RearRight
            ]
        );
        assert!(!l.contains(&Speaker::FrontCenter));
        assert!(!l.contains(&Speaker::Lfe));
    }

    #[test]
    fn five_one_is_six_channels_lfe_last() {
        let l = speaker_layout(6).unwrap();
        assert_eq!(l.len(), 6);
        assert_eq!(l[5], Speaker::Lfe);
        assert_eq!(l[1], Speaker::FrontCenter);
        // 5.1 uses rear (not side) for the surround pair.
        assert_eq!(l[3], Speaker::RearLeft);
        assert_eq!(l[4], Speaker::RearRight);
    }

    #[test]
    fn six_one_uses_side_pair_and_rear_center() {
        // §4.3.9 (rev 16781): 6.1 = FL, C, FR, SL, SR, rear center, LFE.
        let l = speaker_layout(7).unwrap();
        assert_eq!(
            l,
            &[
                Speaker::FrontLeft,
                Speaker::FrontCenter,
                Speaker::FrontRight,
                Speaker::SideLeft,
                Speaker::SideRight,
                Speaker::RearCenter,
                Speaker::Lfe,
            ]
        );
    }

    #[test]
    fn seven_one_uses_side_and_rear_pairs() {
        // §4.3.9 (rev 16781): 7.1 = FL, C, FR, SL, SR, RL, RR, LFE.
        let l = speaker_layout(8).unwrap();
        assert_eq!(
            l,
            &[
                Speaker::FrontLeft,
                Speaker::FrontCenter,
                Speaker::FrontRight,
                Speaker::SideLeft,
                Speaker::SideRight,
                Speaker::RearLeft,
                Speaker::RearRight,
                Speaker::Lfe,
            ]
        );
    }

    #[test]
    fn every_fixed_layout_length_matches_count() {
        for ch in 1u8..=8 {
            let l = speaker_layout(ch).expect("1..=8 are fixed layouts");
            assert_eq!(l.len(), ch as usize, "layout length must equal channels");
        }
    }

    #[test]
    fn five_one_has_exactly_one_lfe_and_one_center() {
        let l = speaker_layout(6).unwrap();
        assert_eq!(l.iter().filter(|s| **s == Speaker::Lfe).count(), 1);
        assert_eq!(l.iter().filter(|s| **s == Speaker::FrontCenter).count(), 1);
    }

    #[test]
    fn above_eight_is_application_defined() {
        assert_eq!(speaker_layout(9), None);
        assert_eq!(speaker_layout(255), None);
        // In-range positions report Unspecified rather than guessing.
        assert_eq!(speaker_at(9, 0), Some(Speaker::Unspecified));
        assert_eq!(speaker_at(255, 254), Some(Speaker::Unspecified));
        assert_eq!(speaker_at(9, 9), None);
    }

    #[test]
    fn zero_channels_is_empty() {
        // §4.2.2 forbids zero channels; the layout helpers stay total.
        assert_eq!(speaker_layout(0), None);
        assert_eq!(speaker_at(0, 0), None);
    }

    #[test]
    fn labels_are_nonempty() {
        for ch in 1u8..=8 {
            for &s in speaker_layout(ch).unwrap() {
                assert!(!s.label().is_empty());
            }
        }
        assert_eq!(Speaker::Unspecified.label(), "application-defined");
    }
}

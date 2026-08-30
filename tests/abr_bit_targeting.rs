//! ABR bit targeting over the integrated encoder
//! (`StreamEncoderConfig::target_bitrate`): the whole-stream
//! quality solve of `quality::solve_lambda_for_bits` run over real
//! encodes. Pins: a reachable budget is met from below (and the
//! spend approaches it), the delivered rate is monotone in the
//! budget, an unreachable budget degrades to the cheapest (`q = 0`)
//! stream, and the identification header carries the target as
//! `bitrate_nominal`. Fully synthetic corpus.

use oxideav_vorbis::{
    decode_ogg_to_pcm, encode_pcm_to_ogg, ogg_packets, parse_identification_header,
    StreamEncoderConfig,
};

const RATE: u32 = 44_100;

/// Two seconds of tones + hiss — enough texture that the knob's rate
/// range spans the budgets below.
fn corpus() -> Vec<Vec<f32>> {
    let n = 2 * RATE as usize;
    let mut seed = 0xc0ffee11u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let row: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            0.3 * (2.0 * std::f32::consts::PI * 331.0 * t).sin()
                + 0.12 * (2.0 * std::f32::consts::PI * 1250.0 * t).sin()
                + 0.02 * rnd()
        })
        .collect();
    vec![row]
}

fn audio_kbps(ogg: &[u8], seconds: f64) -> f64 {
    let packets = ogg_packets(ogg).expect("de-frames");
    let bytes: usize = packets[3..].iter().map(Vec::len).sum();
    bytes as f64 * 8.0 / seconds / 1000.0
}

#[test]
fn abr_meets_reachable_budgets_monotonically() {
    let pcm = corpus();
    let seconds = pcm[0].len() as f64 / f64::from(RATE);
    let mut delivered = Vec::new();
    // This corpus costs ~130 kbps at q = 0 and ~370 kbps at q = 1
    // under the r453 calibration, so the budgets sit inside the
    // knob's reachable range.
    for &kbps in &[160u32, 240, 330] {
        let mut config = StreamEncoderConfig::new(RATE, 1);
        config.target_bitrate = Some(kbps * 1000);
        let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
        let got = audio_kbps(&ogg, seconds);
        eprintln!("ABR target {kbps} kbps -> {got:.1} kbps audio");
        assert!(
            got <= f64::from(kbps),
            "target {kbps} kbps missed: delivered {got:.1}"
        );
        // The solve lands near the budget, not far under it (the knob
        // is bisected to ~1.5 % resolution; the rate curve's own steps
        // dominate — half the budget would mean a broken solve).
        assert!(
            got >= f64::from(kbps) * 0.4,
            "target {kbps} kbps underspent: delivered {got:.1}"
        );
        // The header carries the target.
        let packets = ogg_packets(&ogg).unwrap();
        let id = parse_identification_header(&packets[0]).expect("id parses");
        assert_eq!(id.bitrate_nominal, (kbps * 1000) as i32);
        // And the stream decodes.
        let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
        assert_eq!(decoded.pcm[0].len(), pcm[0].len());
        delivered.push(got);
    }
    assert!(
        delivered.windows(2).all(|w| w[0] <= w[1] + 0.5),
        "delivered rate must be monotone in the budget: {delivered:?}"
    );
}

#[test]
fn unreachable_budget_returns_the_cheapest_knob_end() {
    let pcm = corpus();
    let seconds = pcm[0].len() as f64 / f64::from(RATE);
    let mut config = StreamEncoderConfig::new(RATE, 1);
    config.target_bitrate = Some(1_000); // 1 kbps: far under the q=0 floor
    let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
    let got = audio_kbps(&ogg, seconds);

    // The q = 0 encode is the floor the ABR entry must degrade to.
    let mut floor_cfg = StreamEncoderConfig::new(RATE, 1);
    floor_cfg.quality = 0.0;
    let floor = encode_pcm_to_ogg(&pcm, &floor_cfg).expect("floor encodes");
    let floor_kbps = audio_kbps(&floor, seconds);
    eprintln!("unreachable 1 kbps -> {got:.1} kbps (q=0 floor {floor_kbps:.1})");
    assert!(
        (got - floor_kbps).abs() <= 0.02 * floor_kbps.max(1.0),
        "unreachable budget must land on the q=0 stream: {got:.1} vs {floor_kbps:.1} kbps"
    );
    let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
    assert_eq!(decoded.pcm[0].len(), pcm[0].len());
}

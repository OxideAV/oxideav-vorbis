//! Residue VQ-encode cascade → §8.6.2 body write → residue decode
//! round-trip, measured as a spectral-domain signal-to-noise ratio.
//!
//! This is the milestone integration test for the residue **encode**
//! chain: it drives a real (signed, non-flat) spectral residual through
//! the whole residue-side pipeline and confirms the crate's own residue
//! decoder reconstructs the *nearest-entry cascade approximation* of that
//! residual — bit-exactly on the bitstream, and to a pinned SNR in the
//! spectral domain.
//!
//! The chain, end to end:
//!
//!  1. **Target.** A synthetic length-`P` spectral residual (one residue
//!     partition) is generated with a non-trivial signed shape.
//!  2. **Plan (the code under test).**
//!     [`oxideav_vorbis::plan_partition_cascade`] walks the §8.6.2 cascade
//!     in the write direction, gathering each VQ read's sub-vector,
//!     quantising it with the nearest codebook entry, and subtracting the
//!     reconstruction so each later stage refines the leftover error.
//!  3. **Encode.** The per-partition entry-index lists are wrapped into a
//!     [`oxideav_vorbis::ResidueVectorPlan`] and serialised to a §8.6.2
//!     residue body by [`oxideav_vorbis::write_residue_body`].
//!  4. **Decode.** [`oxideav_vorbis::ResidueDecoder`] reads the body back
//!     and accumulates the cascade (`v[idx] += val`) into the residual
//!     vector — the exact inverse of step 2.
//!  5. **Assert.** The decoded residual equals the sum of the chosen
//!     entries' reconstructions (an exact bitstream round-trip), the
//!     cascade approximates the target to a pinned SNR, and a two-stage
//!     cascade is *strictly closer* to the target than a one-stage cascade
//!     (the refinement property §8.6.2's additive accumulation provides).
//!
//! No Ogg framing, no `docs/` fixtures: the test is fully synthetic and
//! self-contained, so it runs in standalone CI.

use oxideav_vorbis::codebook::{VorbisCodebook, VqLookup};
use oxideav_vorbis::setup::ResidueHeader;
use oxideav_vorbis::{
    plan_partition_cascade, write_residue_body, ResidueDecoder, ResidueVectorPlan,
};

use oxideav_core::bits::BitReaderLsb;

/// A 1-D tessellation (lookup-type-2) VQ value book over a signed integer
/// ladder with `delta = step`, built Kraft-complete: it carries exactly
/// `2^length` entries all at codeword length `length` (so `Σ 2⁻ˡ = 1` and
/// every entry is reachable). Entry `e`'s reconstructed scalar is
/// `minimum_value + e*step`; with `minimum_value = -(entries/2)*step` the
/// ladder spans `[-(entries/2)·step, (entries/2 − 1)·step]` in steps of
/// `step` — a signed quantiser centred near zero. [`book_half`] returns
/// `entries/2` so a caller can map an entry index back to its value.
fn signed_value_book(length: u8, step: f32) -> VorbisCodebook {
    let entries: u32 = 1u32 << length;
    let half = entries / 2;
    let multiplicands: Vec<u32> = (0..entries).collect();
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::Tessellation {
            minimum_value: -(half as f32) * step,
            delta_value: step,
            value_bits: 8,
            sequence_p: false,
            multiplicands,
        },
    }
}

/// The `entries/2` offset that maps a `signed_value_book(length, _)` entry
/// index `e` back to its value `(e − half) * step`.
fn book_half(length: u8) -> i32 {
    (1i32 << length) / 2
}

/// A balanced 1-D scalar classbook with `entries` length-`length`
/// codewords (no VQ lookup) — the §8.6.2 classbook that codes each
/// partition's classification. With one classification we only need a
/// single reachable entry, but a balanced multi-entry book keeps the
/// construction Kraft-complete.
fn classbook(entries: u32, length: u8) -> VorbisCodebook {
    VorbisCodebook {
        dimensions: 1,
        entries,
        codeword_lengths: vec![length; entries as usize],
        lookup: VqLookup::None,
    }
}

/// A single-classification format-1 residue header covering
/// `[0, partition_size)` of a length-`partition_size` decode vector, with
/// the cascade stages `stages` (each `Some(book_index)` for a populated
/// pass). `classbook` is codebook index 0; value books follow.
fn one_partition_header(partition_size: u32, stages: [Option<u8>; 8]) -> ResidueHeader {
    let mut cascade_bits = 0u8;
    for (j, slot) in stages.iter().enumerate() {
        if slot.is_some() {
            cascade_bits |= 1 << j;
        }
    }
    ResidueHeader {
        residue_type: 1,
        residue_begin: 0,
        residue_end: partition_size,
        partition_size,
        classifications: 1,
        classbook: 0,
        cascade: vec![cascade_bits],
        books: vec![stages],
    }
}

/// 10·log10 of the energy ratio between a target and the per-bin error.
fn snr_db(target: &[f32], got: &[f32]) -> f32 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&t, &g) in target.iter().zip(got.iter()) {
        sig += (t as f64) * (t as f64);
        let e = (t - g) as f64;
        err += e * e;
    }
    if err == 0.0 {
        return f32::INFINITY;
    }
    (10.0 * (sig / err).log10()) as f32
}

/// Sum of squared per-bin error between two equal-length vectors.
fn sse(target: &[f32], got: &[f32]) -> f64 {
    target
        .iter()
        .zip(got.iter())
        .map(|(&t, &g)| {
            let e = (t - g) as f64;
            e * e
        })
        .sum()
}

/// A non-flat signed synthetic residual of length `p`: a couple of
/// sinusoids plus a slow tilt, scaled into roughly `±10`.
fn synthetic_residual(p: usize) -> Vec<f32> {
    (0..p)
        .map(|i| {
            let t = i as f32;
            6.0 * (2.0 * std::f32::consts::PI * 2.0 * t / p as f32).sin()
                + 3.0 * (2.0 * std::f32::consts::PI * 7.0 * t / p as f32).cos()
                + (t / p as f32 - 0.5) * 4.0
        })
        .collect()
}

/// Plan a one-partition cascade, encode it, decode it back, and return
/// the reconstructed residual.
fn cascade_roundtrip(
    residual: &[f32],
    header: &ResidueHeader,
    codebooks: &[VorbisCodebook],
    value_book_refs: &[Option<&VorbisCodebook>; 8],
) -> Vec<f32> {
    let p = residual.len() as u32;

    // Plan the cascade (the code under test).
    let entries = plan_partition_cascade(residual, value_book_refs, header.residue_type, p)
        .expect("partition cascade plans");

    let plan = ResidueVectorPlan {
        classifications: vec![0],
        partition_entries: vec![entries],
    };

    // Encode the residue body. blocksize = 2 * partition_size so the
    // per-channel vector length (blocksize/2) equals the partition,
    // and residue_end covers it exactly.
    let blocksize = 2 * residual.len();
    let body = write_residue_body(
        std::slice::from_ref(&plan),
        header,
        codebooks,
        blocksize,
        &[false],
    )
    .expect("residue body serialises");

    // Decode it back.
    let dec = ResidueDecoder::new(header, codebooks).expect("residue decoder builds");
    let mut reader = BitReaderLsb::new(&body);
    let out = dec
        .decode(&mut reader, blocksize, &[false])
        .expect("residue body decodes");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), residual.len());
    out.into_iter().next().unwrap()
}

#[test]
fn single_stage_cascade_roundtrips_and_quantises_to_the_ladder() {
    let p = 32usize;
    let residual = synthetic_residual(p);

    // One stage, length 6 → 64 entries, step 0.25: ladder spans
    // [-8.0, +7.75] in 0.25 increments — comfortably covers the ~±10
    // residual's bulk and quantises each scalar to the nearest 0.25.
    let vbook = signed_value_book(6, 0.25);
    let span_lo = -(book_half(6) as f32) * 0.25; // -8.0
    let span_hi = (book_half(6) as f32 - 1.0) * 0.25; // +7.75
    let cbook = classbook(2, 1);
    let codebooks = vec![cbook, vbook.clone()];
    let mut stages: [Option<u8>; 8] = Default::default();
    stages[0] = Some(1);
    let header = one_partition_header(p as u32, stages);

    let mut refs: [Option<&VorbisCodebook>; 8] = Default::default();
    refs[0] = Some(&vbook);

    let got = cascade_roundtrip(&residual, &header, &codebooks, &refs);

    // The decoded residual is the nearest-ladder approximation: every
    // sample is a multiple of the step within the book's span. (The
    // synthetic residual exceeds the span at its peaks, so this is a
    // genuine clamping quantiser, not a no-op.)
    for &v in &got {
        let q = (v / 0.25).round() * 0.25;
        assert!(
            (v - q).abs() <= 1e-4,
            "decoded {v} is not on the 0.25 ladder"
        );
        assert!(
            v >= span_lo - 1e-4 && v <= span_hi + 1e-4,
            "decoded {v} outside book span [{span_lo}, {span_hi}]"
        );
    }

    // Spectral SNR over the whole partition clears a healthy floor for a
    // 0.25-step ladder applied to a ~±10 signal.
    let snr = snr_db(&residual, &got);
    assert!(
        snr >= 6.0,
        "single-stage cascade SNR {snr} dB below pinned 6 dB"
    );
}

#[test]
fn two_stage_cascade_is_strictly_closer_than_one_stage() {
    let p = 48usize;
    let residual = synthetic_residual(p);

    // Stage 0: coarse ladder (length 6 → 64 entries, step 2.0, span
    // [-64, +62] — covers the whole signal). Stage 1: fine ladder
    // (length 5 → 32 entries, step 0.25, span [-4.0, +3.75]) refines the
    // stage-0 quantisation error, which lies within ±1.0 (half a coarse
    // step).
    let coarse = signed_value_book(6, 2.0);
    let fine = signed_value_book(5, 0.25);
    let cbook = classbook(2, 1);

    // --- one-stage reference: coarse only. ---
    let codebooks_1 = vec![cbook.clone(), coarse.clone()];
    let mut stages_1: [Option<u8>; 8] = Default::default();
    stages_1[0] = Some(1);
    let header_1 = one_partition_header(p as u32, stages_1);
    let mut refs_1: [Option<&VorbisCodebook>; 8] = Default::default();
    refs_1[0] = Some(&coarse);
    let got_1 = cascade_roundtrip(&residual, &header_1, &codebooks_1, &refs_1);

    // --- two-stage: coarse then fine. ---
    let codebooks_2 = vec![cbook, coarse.clone(), fine.clone()];
    let mut stages_2: [Option<u8>; 8] = Default::default();
    stages_2[0] = Some(1);
    stages_2[1] = Some(2);
    let header_2 = one_partition_header(p as u32, stages_2);
    let mut refs_2: [Option<&VorbisCodebook>; 8] = Default::default();
    refs_2[0] = Some(&coarse);
    refs_2[1] = Some(&fine);
    let got_2 = cascade_roundtrip(&residual, &header_2, &codebooks_2, &refs_2);

    // The refinement stage cannot make the approximation worse, and on a
    // genuinely non-ladder-aligned signal it makes it strictly better.
    let sse_1 = sse(&residual, &got_1);
    let sse_2 = sse(&residual, &got_2);
    assert!(
        sse_2 < sse_1,
        "two-stage SSE {sse_2} not strictly below one-stage SSE {sse_1}"
    );

    // Quantitatively, the second 0.25-step stage drives the SNR well above
    // the coarse-only baseline.
    let snr_1 = snr_db(&residual, &got_1);
    let snr_2 = snr_db(&residual, &got_2);
    assert!(
        snr_2 >= snr_1 + 6.0,
        "two-stage SNR {snr_2} dB did not improve coarse-only {snr_1} dB by ≥6 dB"
    );
}

#[test]
fn cascade_reconstruction_equals_sum_of_chosen_entries() {
    // The decoded residual must equal, bin for bin, the sum of the
    // reconstructions of the entries the planner chose — proving the
    // bitstream round-trip (entry indices ↔ codewords) is exact and the
    // only loss is the planner's nearest-entry quantisation.
    let p = 24usize;
    let residual = synthetic_residual(p);

    let coarse = signed_value_book(6, 2.0);
    let fine = signed_value_book(5, 0.25);
    let cbook = classbook(2, 1);
    let codebooks = vec![cbook, coarse.clone(), fine.clone()];

    let mut stages: [Option<u8>; 8] = Default::default();
    stages[0] = Some(1);
    stages[1] = Some(2);
    let header = one_partition_header(p as u32, stages);

    let mut refs: [Option<&VorbisCodebook>; 8] = Default::default();
    refs[0] = Some(&coarse);
    refs[1] = Some(&fine);

    // Plan and reconstruct independently from the chosen entries: for a
    // 1-D contiguous (format 1) cascade, read `k` covers position `k`, so
    // entry list[k] reconstructs to `(entry - half)*step` at bin `k`,
    // summed across stages.
    let entries = plan_partition_cascade(&residual, &refs, 1, p as u32).unwrap();
    let mut expected = vec![0.0f32; p];
    // Stage 0 (coarse, half = 32, step 2.0).
    for (k, &e) in entries[0].as_ref().unwrap().iter().enumerate() {
        expected[k] += (e as f32 - book_half(6) as f32) * 2.0;
    }
    // Stage 1 (fine, half = 16, step 0.25).
    for (k, &e) in entries[1].as_ref().unwrap().iter().enumerate() {
        expected[k] += (e as f32 - book_half(5) as f32) * 0.25;
    }

    let got = cascade_roundtrip(&residual, &header, &codebooks, &refs);
    for (bin, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-4,
            "bin {bin}: decoded {g} != sum-of-entries {e}"
        );
    }
}

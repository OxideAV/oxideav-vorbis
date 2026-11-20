//! Encoder measurement harness: a self-generated signal battery, the
//! encode / decode entry points with timing, and the objective
//! measures the crate README quotes (whole-stream SNR, short-segment
//! SNR and a Bark-band noise-to-mask ratio).
//!
//! ```text
//! quality_battery gen  <dir>                    write the battery as 16-bit WAVs
//! quality_battery enc  <in.wav> <out.ogg> q=<0..1> | kbps=<n>
//! quality_battery dec  <in.ogg> <out.wav>       this crate's decoder
//! quality_battery rate <in.ogg>                 audio-packet kbps
//! quality_battery cmp  <ref.wav> <decoded.wav>  SNR / segSNR / NMR
//! ```
//!
//! Everything is deterministic (a fixed linear-congruential noise
//! source), so two runs of `gen` produce identical files. The `cmp`
//! measures are computed by this file alone — none of the encoder's
//! own psychoacoustic code — so they score any decoder's output.
//!
//! * **SNR** — `10·log10(Σ ref² / Σ (ref − dec)²)` over all channels.
//! * **segSNR** — the mean of per-segment SNRs over 256-sample
//!   segments whose reference level is above −70 dBFS, each clamped
//!   to `[-10, 60]` dB. Sensitive to noise spread ahead of an attack.
//! * **NMR** — per 2048-sample Hann frame (hop 1024) the error power
//!   in each of 25 Bark bands against a masked threshold built from
//!   the reference's band powers (spread −27 dB/Bark downward and
//!   −10 dB/Bark upward, 12 dB under the masker, floored at the
//!   threshold in quiet for a 96 dB SPL full scale). Reported as the
//!   mean band NMR in dB and the percentage of band-frames whose noise
//!   exceeds the threshold.

use std::f64::consts::PI;
use std::time::Instant;

use oxideav_vorbis::{StreamEncoderConfig, decode_ogg_to_pcm, encode_pcm_to_ogg, ogg_packets};

const RATE: u32 = 44_100;
const SECONDS: f64 = 6.0;

/// Deterministic uniform noise in `[-1, 1)`.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn total_samples() -> usize {
    (SECONDS * f64::from(RATE)) as usize
}

/// Add one struck-string note: inharmonic partials with
/// frequency-dependent decay plus a short broadband strike.
fn add_struck_note(
    out: &mut [Vec<f64>],
    start: usize,
    f0: f64,
    gain: f64,
    pan: f64,
    noise: &mut Noise,
) {
    let n = out[0].len();
    let sr = f64::from(RATE);
    for i in start..n {
        let t = (i - start) as f64 / sr;
        if t > 3.0 {
            break;
        }
        let mut v = 0.0;
        for h in 1..=24 {
            let hf = f64::from(h);
            let f = f0 * hf * (1.0 + 0.0005 * hf * hf).sqrt();
            if f > 0.45 * sr {
                break;
            }
            v += (2.0 * PI * f * t).sin() * (-t * (1.2 + 0.5 * hf)).exp() / hf.powf(1.2);
        }
        // The strike: 3 ms of noise under a fast decay.
        let strike = if t < 0.02 {
            noise.next() * (-t * 400.0).exp() * 0.5
        } else {
            0.0
        };
        let attack = (t / 0.002).min(1.0);
        let s = gain * (v * attack + strike);
        out[0][i] += s * (1.0 - pan);
        out[1][i] += s * pan;
    }
}

fn piano() -> Vec<Vec<f64>> {
    let n = total_samples();
    let mut out = vec![vec![0.0; n]; 2];
    let mut noise = Noise(11);
    let notes = [
        220.0, 277.2, 329.6, 440.0, 392.0, 261.6, 523.3, 196.0, 349.2, 293.7, 659.3, 246.9, 130.8,
        415.3, 587.3, 174.6,
    ];
    for (k, &f0) in notes.iter().enumerate() {
        let start = (k as f64 * 0.36 * f64::from(RATE)) as usize;
        let pan = 0.3 + 0.4 * ((k * 7) % 5) as f64 / 4.0;
        add_struck_note(&mut out, start, f0, 0.22, pan, &mut noise);
    }
    out
}

/// Plucked strings: a recirculating delay line with a two-point
/// average in the loop, excited by a noise burst.
fn plucked() -> Vec<Vec<f64>> {
    let n = total_samples();
    let mut out = vec![vec![0.0; n]; 2];
    let mut noise = Noise(23);
    let notes = [
        110.0, 164.8, 196.0, 246.9, 329.6, 146.8, 220.0, 293.7, 392.0, 130.8, 174.6, 261.6,
    ];
    for (k, &f0) in notes.iter().enumerate() {
        let start = (k as f64 * 0.47 * f64::from(RATE)) as usize;
        let period = (f64::from(RATE) / f0).round() as usize;
        let mut line: Vec<f64> = (0..period).map(|_| noise.next()).collect();
        let pan = if k % 2 == 0 { 0.35 } else { 0.65 };
        let mut pos = 0usize;
        for i in start..n {
            let cur = line[pos];
            let next = line[(pos + 1) % period];
            line[pos] = 0.498 * (cur + next);
            pos = (pos + 1) % period;
            out[0][i] += 0.25 * cur * (1.0 - pan);
            out[1][i] += 0.25 * cur * pan;
        }
    }
    out
}

/// Castanet-like clicks: 2–7 kHz resonant noise bursts with a ~4 ms
/// decay at irregular spacing, out of silence.
fn clicks() -> Vec<Vec<f64>> {
    let n = total_samples();
    let mut out = vec![vec![0.0; n]; 2];
    let mut noise = Noise(37);
    let sr = f64::from(RATE);
    let mut t0 = 0.1;
    let mut k = 0usize;
    while t0 < SECONDS - 0.1 {
        let start = (t0 * sr) as usize;
        let fc = 2500.0 + 900.0 * ((k * 5) % 6) as f64;
        // Two-pole resonator on noise.
        let r = 0.93;
        let (a1, a2) = (-2.0 * r * (2.0 * PI * fc / sr).cos(), r * r);
        let (mut y1, mut y2) = (0.0, 0.0);
        let pan = 0.25 + 0.5 * ((k * 3) % 4) as f64 / 3.0;
        for i in start..(start + 2200).min(n) {
            let t = (i - start) as f64 / sr;
            let x = noise.next() * (-t * 260.0).exp();
            let y = x - a1 * y1 - a2 * y2;
            y2 = y1;
            y1 = y;
            out[0][i] += 0.12 * y * (1.0 - pan);
            out[1][i] += 0.12 * y * pan;
        }
        t0 += 0.11 + 0.23 * ((k * 7) % 5) as f64 / 4.0;
        k += 1;
    }
    out
}

/// Speech-like: a glottal pulse train with a gliding pitch through
/// three moving formant resonators, alternating with fricative noise
/// and pauses.
fn speech() -> Vec<Vec<f64>> {
    let n = total_samples();
    let mut mono = vec![0.0; n];
    let mut noise = Noise(51);
    let sr = f64::from(RATE);
    let mut phase = 0.0;
    let mut state = [[0.0f64; 2]; 3];
    let mut hp = 0.0;
    for (i, slot) in mono.iter_mut().enumerate() {
        let t = i as f64 / sr;
        let syll = t / 0.28;
        let frac = syll.fract();
        let idx = syll as usize;
        let voiced = idx % 4 != 3;
        let env = if frac < 0.75 {
            (frac / 0.06).min(1.0) * ((0.75 - frac) / 0.1).min(1.0)
        } else {
            0.0
        };
        let f0 = 110.0 + 35.0 * (2.0 * PI * 0.7 * t).sin() + 12.0 * ((idx * 3) % 5) as f64;
        phase += f0 / sr;
        let mut src = 0.0;
        if phase >= 1.0 {
            phase -= 1.0;
            src = 1.0;
        }
        let formants = [
            300.0 + 450.0 * (((idx * 5) % 7) as f64 / 6.0),
            900.0 + 1400.0 * (((idx * 3) % 5) as f64 / 4.0),
            2500.0 + 500.0 * (((idx * 2) % 3) as f64 / 2.0),
        ];
        let v = if voiced {
            let mut acc = 0.0;
            for (st, (&fc, &g)) in state.iter_mut().zip(formants.iter().zip(&[1.0, 0.6, 0.3])) {
                let r = 0.985f64;
                let y = src - (-2.0 * r * (2.0 * PI * fc / sr).cos()) * st[0] - r * r * st[1];
                st[1] = st[0];
                st[0] = y;
                acc += g * y;
            }
            0.035 * acc
        } else {
            // Fricative: differenced noise.
            let x = noise.next();
            let y = x - hp;
            hp = x;
            0.05 * y
        };
        *slot = v * env;
    }
    let right: Vec<f64> = mono.iter().map(|&v| 0.9 * v).collect();
    vec![mono, right]
}

/// Sustained tonal: an organ-like chord with slow vibrato and a
/// slightly different partial balance per channel.
fn sustained() -> Vec<Vec<f64>> {
    let n = total_samples();
    let mut out = vec![vec![0.0; n]; 2];
    let sr = f64::from(RATE);
    let chord = [130.8, 196.0, 261.6, 329.6, 392.0];
    for (c, row) in out.iter_mut().enumerate() {
        for (i, slot) in row.iter_mut().enumerate() {
            let t = i as f64 / sr;
            let fade = (t / 0.2).min(1.0) * ((SECONDS - t) / 0.2).min(1.0);
            let mut v = 0.0;
            for (k, &f0) in chord.iter().enumerate() {
                let vib = 1.0 + 0.003 * (2.0 * PI * (4.5 + 0.3 * k as f64) * t).sin();
                for h in 1..=10 {
                    let hf = f64::from(h);
                    let g = if (h + c as i32) % 2 == 0 { 0.7 } else { 1.0 };
                    v += g * (2.0 * PI * f0 * hf * vib * t + 0.4 * hf * c as f64).sin()
                        / hf.powf(1.4);
                }
            }
            *slot = 0.05 * v * fade;
        }
    }
    out
}

/// A mix: sustained bed, struck notes and clicks together.
fn mix() -> Vec<Vec<f64>> {
    let (a, b, c) = (sustained(), piano(), clicks());
    (0..2)
        .map(|ch| {
            (0..total_samples())
                .map(|i| 0.6 * a[ch][i] + 0.7 * b[ch][i] + 0.6 * c[ch][i])
                .collect()
        })
        .collect()
}

type Material = (&'static str, fn() -> Vec<Vec<f64>>);

fn battery() -> Vec<Material> {
    vec![
        ("piano", piano as fn() -> Vec<Vec<f64>>),
        ("plucked", plucked),
        ("clicks", clicks),
        ("speech", speech),
        ("sustained", sustained),
        ("mix", mix),
    ]
}

// ---- WAV I/O (16-bit PCM) ----

fn write_wav(path: &str, pcm: &[Vec<f32>], rate: u32) -> std::io::Result<()> {
    let ch = pcm.len() as u16;
    let n = pcm[0].len();
    let data_len = (n * ch as usize * 2) as u32;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&ch.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * u32::from(ch) * 2).to_le_bytes());
    b.extend_from_slice(&(ch * 2).to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..n {
        for row in pcm {
            let v = (f64::from(row[i]) * 32767.0)
                .round()
                .clamp(-32768.0, 32767.0) as i16;
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, b)
}

fn read_wav(path: &str) -> (Vec<Vec<f32>>, u32) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(
        &b[..4] == b"RIFF" && &b[8..12] == b"WAVE",
        "{path}: not a WAV"
    );
    let mut pos = 12;
    let (mut ch, mut rate, mut bits) = (0usize, 0u32, 0u16);
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let len = u32::from_le_bytes(b[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            ch = u16::from_le_bytes(b[body + 2..body + 4].try_into().unwrap()) as usize;
            rate = u32::from_le_bytes(b[body + 4..body + 8].try_into().unwrap());
            bits = u16::from_le_bytes(b[body + 14..body + 16].try_into().unwrap());
        } else if id == b"data" {
            assert_eq!(bits, 16, "{path}: 16-bit PCM expected");
            let end = (body + len).min(b.len());
            let frames = (end - body) / (2 * ch);
            let mut pcm = vec![Vec::with_capacity(frames); ch];
            for f in 0..frames {
                for (c, row) in pcm.iter_mut().enumerate() {
                    let o = body + 2 * (f * ch + c);
                    row.push(f32::from(i16::from_le_bytes([b[o], b[o + 1]])) / 32767.0);
                }
            }
            return (pcm, rate);
        }
        pos = body + len + (len & 1);
    }
    panic!("{path}: no data chunk");
}

// ---- measures ----

/// In-place radix-2 complex FFT (`re`/`im` of power-of-two length).
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (ang * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * c - im[b] * s, re[b] * s + im[b] * c);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

fn bark(f: f64) -> f64 {
    13.0 * (0.00076 * f).atan() + 3.5 * ((f / 7500.0) * (f / 7500.0)).atan()
}

/// Threshold in quiet (dB SPL) at `f` Hz.
fn quiet_db(f: f64) -> f64 {
    let k = (f / 1000.0).max(0.02);
    3.64 * k.powf(-0.8) - 6.5 * (-0.6 * (k - 3.3) * (k - 3.3)).exp() + 1e-3 * k.powi(4)
}

const NMR_FRAME: usize = 2048;
const NMR_BANDS: usize = 25;

/// `(mean band NMR in dB, percentage of band-frames above threshold)`.
fn noise_to_mask(reference: &[Vec<f32>], decoded: &[Vec<f32>], rate: u32) -> (f64, f64) {
    let half = NMR_FRAME / 2;
    let window: Vec<f64> = (0..NMR_FRAME)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * (i as f64 + 0.5) / NMR_FRAME as f64).cos())
        .collect();
    let band_of: Vec<usize> = (0..half)
        .map(|k| {
            let f = (k as f64 + 0.5) * f64::from(rate) / NMR_FRAME as f64;
            (bark(f) as usize).min(NMR_BANDS - 1)
        })
        .collect();
    // Threshold in quiet per band as a power per bin on the frame's
    // scale: a full-scale sine reads 96 dB SPL.
    let full_scale_power = (NMR_FRAME as f64 / 4.0).powi(2);
    let quiet: Vec<f64> = (0..NMR_BANDS)
        .map(|b| {
            let centre = (0..half)
                .filter(|&k| band_of[k] == b)
                .map(|k| (k as f64 + 0.5) * f64::from(rate) / NMR_FRAME as f64)
                .fold(f64::INFINITY, f64::min);
            if centre.is_finite() {
                full_scale_power * 10f64.powf((quiet_db(centre.max(50.0)) - 96.0) / 10.0)
            } else {
                0.0
            }
        })
        .collect();
    let (mut sum_db, mut above, mut count) = (0.0f64, 0usize, 0usize);
    for (r, d) in reference.iter().zip(decoded) {
        let n = r.len().min(d.len());
        let mut start = 0;
        while start + NMR_FRAME <= n {
            let mut sr = vec![0.0; NMR_FRAME];
            let mut si = vec![0.0; NMR_FRAME];
            let mut er = vec![0.0; NMR_FRAME];
            let mut ei = vec![0.0; NMR_FRAME];
            for i in 0..NMR_FRAME {
                let (a, b) = (f64::from(r[start + i]), f64::from(d[start + i]));
                sr[i] = a * window[i];
                er[i] = (b - a) * window[i];
            }
            fft(&mut sr, &mut si);
            fft(&mut er, &mut ei);
            let mut sig = [0.0f64; NMR_BANDS];
            let mut err = [0.0f64; NMR_BANDS];
            for k in 0..half {
                sig[band_of[k]] += sr[k] * sr[k] + si[k] * si[k];
                err[band_of[k]] += er[k] * er[k] + ei[k] * ei[k];
            }
            let frame_power: f64 = sig.iter().sum();
            if frame_power > full_scale_power * 1e-7 {
                for b in 0..NMR_BANDS {
                    let mut mask = quiet[b];
                    for (j, &s) in sig.iter().enumerate() {
                        let dz = b as f64 - j as f64;
                        let slope = if dz >= 0.0 { -10.0 * dz } else { 27.0 * dz };
                        mask += s * 10f64.powf((slope - 12.0) / 10.0);
                    }
                    let nmr = 10.0 * ((err[b] + 1e-30) / (mask + 1e-30)).log10();
                    sum_db += nmr.max(-40.0);
                    above += usize::from(nmr > 0.0);
                    count += 1;
                }
            }
            start += half;
        }
    }
    if count == 0 {
        return (0.0, 0.0);
    }
    (sum_db / count as f64, 100.0 * above as f64 / count as f64)
}

fn snr_db(reference: &[Vec<f32>], decoded: &[Vec<f32>]) -> f64 {
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (r, d) in reference.iter().zip(decoded) {
        for (&a, &b) in r.iter().zip(d) {
            sig += f64::from(a) * f64::from(a);
            err += (f64::from(a) - f64::from(b)).powi(2);
        }
    }
    10.0 * (sig / err.max(1e-30)).log10()
}

fn segmental_snr_db(reference: &[Vec<f32>], decoded: &[Vec<f32>]) -> f64 {
    const SEG: usize = 256;
    let (mut acc, mut count) = (0.0f64, 0usize);
    for (r, d) in reference.iter().zip(decoded) {
        let n = r.len().min(d.len());
        for (rs, ds) in r[..n]
            .as_chunks::<SEG>()
            .0
            .iter()
            .zip(d[..n].as_chunks::<SEG>().0)
        {
            let (mut sig, mut err) = (0.0f64, 0.0f64);
            for (&a, &b) in rs.iter().zip(ds) {
                sig += f64::from(a) * f64::from(a);
                err += (f64::from(a) - f64::from(b)).powi(2);
            }
            if sig / SEG as f64 > 1e-7 {
                acc += (10.0 * (sig / err.max(1e-30)).log10()).clamp(-10.0, 60.0);
                count += 1;
            }
        }
    }
    if count == 0 { 0.0 } else { acc / count as f64 }
}

fn audio_kbps(ogg: &[u8], seconds: f64) -> f64 {
    let packets = ogg_packets(ogg).expect("de-frames");
    let bytes: usize = packets.iter().skip(3).map(Vec::len).sum();
    bytes as f64 * 8.0 / seconds / 1000.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: quality_battery gen|enc|dec|rate|cmp ... (see the file header)";
    match args.get(1).map(String::as_str) {
        Some("gen") => {
            let dir = args.get(2).expect(usage);
            std::fs::create_dir_all(dir).expect("create dir");
            for (name, make) in battery() {
                let pcm: Vec<Vec<f32>> = make()
                    .into_iter()
                    .map(|row| row.into_iter().map(|v| v as f32).collect())
                    .collect();
                let peak = pcm.iter().flatten().fold(0.0f32, |m, &v| m.max(v.abs()));
                write_wav(&format!("{dir}/{name}.wav"), &pcm, RATE).expect("write wav");
                println!("{name}: peak {peak:.3}");
            }
        }
        Some("enc") => {
            let (pcm, rate) = read_wav(args.get(2).expect(usage));
            let mut config = StreamEncoderConfig::new(rate, pcm.len() as u8);
            let knob = args.get(4).expect(usage);
            if let Some(q) = knob.strip_prefix("q=") {
                config.quality = q.parse().expect("q");
            } else if let Some(k) = knob.strip_prefix("kbps=") {
                let kbps: f64 = k.parse().expect("kbps");
                config.target_bitrate = Some((kbps * 1000.0) as u32);
            } else {
                panic!("{usage}");
            }
            let seconds = pcm[0].len() as f64 / f64::from(rate);
            let t = Instant::now();
            let ogg = encode_pcm_to_ogg(&pcm, &config).expect("encodes");
            let dt = t.elapsed().as_secs_f64();
            std::fs::write(args.get(3).expect(usage), &ogg).expect("write ogg");
            println!(
                "kbps {:.2} enc_s {dt:.3} xrt {:.2}",
                audio_kbps(&ogg, seconds),
                seconds / dt
            );
        }
        Some("dec") => {
            let ogg = std::fs::read(args.get(2).expect(usage)).expect("read ogg");
            let t = Instant::now();
            let decoded = decode_ogg_to_pcm(&ogg).expect("decodes");
            let dt = t.elapsed().as_secs_f64();
            let seconds = decoded.pcm[0].len() as f64 / f64::from(decoded.sample_rate);
            write_wav(args.get(3).expect(usage), &decoded.pcm, decoded.sample_rate)
                .expect("write wav");
            println!("dec_s {dt:.4} xrt {:.1}", seconds / dt);
        }
        Some("rate") => {
            let ogg = std::fs::read(args.get(2).expect(usage)).expect("read ogg");
            let seconds: f64 = args
                .get(3)
                .map(|s| s.parse().expect("seconds"))
                .unwrap_or(SECONDS);
            println!("{:.2}", audio_kbps(&ogg, seconds));
        }
        Some("cmp") => {
            let (reference, rate) = read_wav(args.get(2).expect(usage));
            let (decoded, _) = read_wav(args.get(3).expect(usage));
            let (nmr, above) = noise_to_mask(&reference, &decoded, rate);
            println!(
                "snr {:.2} seg {:.2} nmr {nmr:.2} above {above:.1}",
                snr_db(&reference, &decoded),
                segmental_snr_db(&reference, &decoded),
            );
        }
        _ => eprintln!("{usage}"),
    }
}

# VQ training corpus options

The Vorbis encoder currently ships a 2-book degenerate residue setup
(see `src/encoder.rs` — 128-entry main VQ on `{-5..+5}^2` plus a
16-entry fine-correction VQ on `{-0.6..+0.6}^2`). Round 1 of task #93
adds an in-tree LBG trainer (`src/bin/vq-train.rs`) but does not yet
choose a corpus. This document surveys three license-clean options so
the user can pick one for round 2.

## Hard requirements

- License must be CC0, CC-BY, BSD, MIT, or Apache. **No** Vorbis
  reference encoder source, **no** libvorbis-derived `.vqh`, **no**
  third-party Vorbis training scripts (lewton, stb_vorbis, libvorbis
  itself).
- Total ~1-2 hours of mixed material at 44.1 or 48 kHz, signed 16-bit
  little-endian PCM (the trainer's input format). Mix ratio target:
  ~40% music, ~30% speech, ~30% ambient/foley.
- Stable downloadable URL (mirror-able). Per-file licensing trivially
  inspectable (single license tag for the whole pack, not per-file).

## Option A — LibriVox CC0 public-domain audiobook chunks

| Field         | Value                                                          |
|---------------|----------------------------------------------------------------|
| License       | Public Domain (Creative Commons CC0 equivalent in US)          |
| Sample rate   | 44.1 kHz native (most recordings)                              |
| Channels      | Mostly mono                                                    |
| Total size    | ~50 MB / hour at 44.1 kHz mono → 100 MB for 2 hours speech     |
| URL           | https://librivox.org/ (per-book; e.g. catalog ID 0001)         |
| Stability     | Excellent — Internet Archive mirrors every recording           |

**Pros:** Completely unencumbered (pre-1929 books, US public domain).
Speech content directly stresses the formant + sibilant partitions
that VQ training tends to under-represent. LibriVox files are
volunteer-recorded so the timbral variety is high (many readers, many
mic chains).

**Cons:** Mono only — for stereo VQ training we'd need to either
duplicate to L=R (defeating the point of joint coding) or pair
unrelated recordings (artificial). Speech only — no music, no
ambient. The trainer would need a separate music + ambient corpus to
balance partitions out.

**Mitigation:** use LibriVox for the speech 30% and pair with one of
the music corpora below.

## Option B — Free Music Archive CC-BY curated set

| Field         | Value                                                          |
|---------------|----------------------------------------------------------------|
| License       | CC-BY (attribution required in NOTICES)                        |
| Sample rate   | Mostly 44.1 kHz, occasional 48 kHz                             |
| Channels      | Stereo                                                         |
| Total size    | ~700 MB / hour at 44.1 kHz stereo 16-bit                       |
| URL           | https://freemusicarchive.org/ (per-album; e.g. FMA-small set)  |
| Stability     | Good — FMA-small (8 GB, 8000 tracks across 8 genres) is        |
|               | distributed for ML research and mirrored on archive.org        |

**Pros:** Stereo, broad genre coverage (electronic, hip-hop, folk,
rock, instrumental, jazz, classical, experimental — directly
relevant to libvorbis's per-genre book tuning). FMA-small is the
canonical music-information-retrieval (MIR) benchmark dataset, so
provenance is well-documented and the pre-cut 30-second per-track
clips are perfect for training (no need to chop ourselves).

**Cons:** Attribution requirement: NOTICES would have to enumerate
every track used (can collapse to one URL pointing at the FMA
metadata CSV). 8 GB full pack is too big — pick a 30-track subset
spanning all 8 genres = ~15 minutes total. Some FMA-small tracks
are CC-BY-NC (non-commercial), which we'd need to filter out — the
metadata CSV has the per-track license, so a one-shot `awk` filter
suffices.

**Mitigation:** start from `fma_metadata.csv`, keep only rows where
`track_license` matches `^CC BY( |$)|^CC0|^Public Domain`. ~60% of
FMA-small passes this filter.

## Option C — Wikimedia Commons Featured Sounds (mixed content)

| Field         | Value                                                          |
|---------------|----------------------------------------------------------------|
| License       | Mostly CC-BY-SA, some CC0, some PD-old                         |
| Sample rate   | Variable: 44.1 / 48 kHz, occasionally 22.05 kHz                |
| Channels      | Mixed mono/stereo                                              |
| Total size    | ~200 MB for a 1-hour cross-section                             |
| URL           | https://commons.wikimedia.org/wiki/Commons:Featured_sounds     |
| Stability     | Excellent — Wikimedia Foundation mirrors                       |

**Pros:** Single repository, single license metadata format
(MediaWiki templates are scriptable). Genuinely mixed content:
classical recordings, animal calls, spoken word, environmental
ambient, instrument samples — exactly the cross-section we want for
a general-purpose codebook. Featured-sounds set is hand-curated for
audio quality (no clipped or noisy uploads).

**Cons:** CC-BY-SA "share-alike" is the most aggressive licence in
this trio — it propagates to derived works. The trained codebook
tables themselves arguably aren't a derivative of the audio (LBG
clusters are aggregated statistics, not samples), but the question
is novel and would need a license review. A CC0/CC-BY-only filter
would leave maybe 30% of the corpus, which is then comparable in
size to option A. Audio is delivered as Ogg Vorbis or FLAC, so the
trainer would need a decode step to get raw PCM (out of round-1
scope — currently we accept raw PCM only).

**Mitigation:** filter to CC0 + CC-BY tracks only, decode upstream
into raw 16-bit PCM with ffmpeg before feeding the trainer (the
trainer stays format-agnostic: PCM in).

## Recommendation outline (final pick deferred to round 2)

| Aspect              | Best option       | Reason                              |
|---------------------|-------------------|-------------------------------------|
| Licence simplicity  | A (LibriVox PD)   | No attribution, no SA               |
| Stereo coverage     | B (FMA CC-BY)     | Native stereo                       |
| Genre balance       | C (Commons mixed) | Hand-curated cross-section          |
| Total bytes         | A (~100 MB)       | Smallest                            |
| Tooling effort      | B (FMA CC-BY)     | Pre-cut 30-s clips, no chopping     |

**Likely pick:** A + B, with LibriVox covering speech (30 minutes
mono) and a CC-BY-only subset of FMA-small covering music + ambient
(60 minutes stereo). NOTICES would gain one URL for LibriVox plus a
checked-in list of FMA track IDs. Total raw PCM ~600 MB downloaded
once, never committed (a `scripts/fetch-vq-corpus.sh` would do it).

The user should pick before round 2; the trainer (round 1) is
corpus-agnostic.

#!/usr/bin/env bash
# fetch-vq-corpus.sh — download and decode the VQ training corpus for
# task #93 round 2 of oxideav-vorbis.
#
# Pulls ~30 minutes of LibriVox public-domain speech and ~60 minutes of
# Musopen CC0 / archive.org Featured Sounds music, decodes everything
# to raw 16-bit signed little-endian PCM at 44.1 kHz mono, and writes
# the result to two files in the cache directory:
#
#   $CORPUS_DIR/speech.pcm   — concatenated LibriVox chapters
#   $CORPUS_DIR/music.pcm    — concatenated Musopen / Commons tracks
#
# CORPUS_DIR defaults to ~/.cache/oxideav-vq-corpus/ . Override with the
# environment variable CORPUS_DIR if you want a different location.
#
# These files are NOT committed to the repository — they're large
# (~60 MB speech + ~315 MB music at 44.1 kHz S16 mono) and the audio
# itself isn't part of our deliverable. The TRAINED codebooks
# (`src/trained_books.rs`, generated downstream by `vq-train`) are the
# committed artifact.
#
# Requires: curl, ffmpeg.
#
# Source URLs and licences:
#
#   LibriVox / Internet Archive (US Public Domain):
#     - Alice's Adventures in Wonderland (Carroll, version 8)
#       https://archive.org/details/aliceinwonderland_2106_librivox
#     - Pride and Prejudice (Austen, version 3 / Karen Savage), chapters 1-2
#       https://archive.org/details/pride_prejudice_krs_librivox
#     - The Adventures of Sherlock Holmes (Doyle, version 2 / Ruth Golding)
#       https://archive.org/details/adventures_sherlock_holmes_rg_librivox
#
#   Musopen — The Complete Chopin Collection (CC0 1.0 Universal):
#     https://archive.org/details/musopen-chopin
#     Several piano works picked for tonal variety (etudes / nocturnes /
#     ballades) — single CC0 licence covers everything.
#
# Total raw download size: ~250 MB compressed (mp3/ogg). After decode
# to 44.1 kHz S16 mono: ~375 MB raw PCM.

set -euo pipefail

CORPUS_DIR="${CORPUS_DIR:-$HOME/.cache/oxideav-vq-corpus}"
mkdir -p "$CORPUS_DIR"
RAW_DIR="$CORPUS_DIR/raw"
mkdir -p "$RAW_DIR"

if ! command -v curl >/dev/null 2>&1; then
    echo "fetch-vq-corpus: curl not found, aborting" >&2
    exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "fetch-vq-corpus: ffmpeg not found, aborting" >&2
    exit 1
fi

# Each line: <output-name> <source-url>
SPEECH_URLS=(
    "alice_ch01.mp3 https://archive.org/download/aliceinwonderland_2106_librivox/alicesadventuresinwonderland_01_carroll.mp3"
    "alice_ch02.mp3 https://archive.org/download/aliceinwonderland_2106_librivox/alicesadventuresinwonderland_02_carroll.mp3"
    "alice_ch03.mp3 https://archive.org/download/aliceinwonderland_2106_librivox/alicesadventuresinwonderland_03_carroll.mp3"
    "pride_ch01.mp3 https://archive.org/download/pride_prejudice_krs_librivox/pride_and_prejudice_01_austen.mp3"
    "pride_ch02.mp3 https://archive.org/download/pride_prejudice_krs_librivox/pride_and_prejudice_02_austen.mp3"
    "sherlock_ch01.mp3 https://archive.org/download/adventures_sherlock_holmes_rg_librivox/adventuresholmes_01_doyle.mp3"
)

# Music: Chopin Complete Collection (CC0). File names verified against the
# musopen-chopin item; Internet Archive serves the exact filenames listed
# in its `metadata` JSON. We use the .ogg derivatives for compactness —
# ffmpeg decodes them losslessly (well, as losslessly as you can decode
# Vorbis…) into 16-bit PCM for the trainer.
MUSIC_URLS=(
    "chopin_nocturne_op9_2.ogg https://archive.org/download/musopen-chopin/Nocturne%20Op.%209%20no.%202%20in%20E%20flat%20major.ogg"
    "chopin_nocturne_op27_1.ogg https://archive.org/download/musopen-chopin/Nocturne%20Op.%2027%20no.%201%20in%20C%20sharp%20minor.ogg"
    "chopin_nocturne_op48_1.ogg https://archive.org/download/musopen-chopin/Nocturne%20Op.%2048%20no.%201%20in%20C%20minor.ogg"
    "chopin_ballade_op23.ogg https://archive.org/download/musopen-chopin/Ballade%20no.%201%20-%20Op.%2023.ogg"
    "chopin_ballade_op38.ogg https://archive.org/download/musopen-chopin/Ballade%20no.%202%20-%20Op.%2038.ogg"
    "chopin_fantasie_imp_op66.ogg https://archive.org/download/musopen-chopin/Fantasie%20Impromptu%20Op.%2066.ogg"
)

fetch_one() {
    local name="$1"
    local url="$2"
    local out="$RAW_DIR/$name"
    if [ -f "$out" ] && [ -s "$out" ]; then
        echo "[skip] $name already cached ($(du -h "$out" | cut -f1))"
        return
    fi
    echo "[fetch] $name <- $url"
    # -f → fail on HTTP errors, -L → follow redirects, --retry for flaky CDN
    curl -fL --retry 3 --retry-delay 2 --connect-timeout 30 --max-time 600 \
        -o "$out.tmp" "$url"
    mv "$out.tmp" "$out"
}

echo "[fetch-vq-corpus] cache dir: $CORPUS_DIR"
echo "[fetch-vq-corpus] downloading speech (LibriVox, public domain)..."
for entry in "${SPEECH_URLS[@]}"; do
    name="${entry%% *}"
    url="${entry#* }"
    fetch_one "$name" "$url"
done

echo "[fetch-vq-corpus] downloading music (Musopen Chopin, CC0)..."
for entry in "${MUSIC_URLS[@]}"; do
    name="${entry%% *}"
    url="${entry#* }"
    fetch_one "$name" "$url"
done

decode_concat() {
    local out="$1"
    shift
    local list="$CORPUS_DIR/.tmp_concat.txt"
    : > "$list"
    for f in "$@"; do
        # ffmpeg concat-demuxer requires absolute or relative-to-list paths;
        # we use absolute and shell-quote (filenames are ASCII so no escaping).
        echo "file '$f'" >> "$list"
    done
    echo "[decode] $(basename "$out") <- $(wc -l < "$list") inputs"
    # -f s16le -ac 1 -ar 44100 → raw 16-bit signed LE, mono, 44.1 kHz
    # -loglevel error so the script's stderr stays readable; ffmpeg errors
    # still propagate via the non-zero exit code.
    ffmpeg -y -loglevel error -f concat -safe 0 -i "$list" \
        -f s16le -ac 1 -ar 44100 "$out"
    rm -f "$list"
}

# Build absolute paths for ffmpeg concat-demuxer.
SPEECH_FILES=()
for entry in "${SPEECH_URLS[@]}"; do
    name="${entry%% *}"
    SPEECH_FILES+=("$RAW_DIR/$name")
done
MUSIC_FILES=()
for entry in "${MUSIC_URLS[@]}"; do
    name="${entry%% *}"
    MUSIC_FILES+=("$RAW_DIR/$name")
done

decode_concat "$CORPUS_DIR/speech.pcm" "${SPEECH_FILES[@]}"
decode_concat "$CORPUS_DIR/music.pcm" "${MUSIC_FILES[@]}"

# Produce a combined corpus too — the trainer uses one PCM stream as
# input, so the typical invocation is:
#   cat $CORPUS_DIR/speech.pcm $CORPUS_DIR/music.pcm | vq-train ...
# This concatenated file just makes the command line shorter.
echo "[concat] $(basename "$CORPUS_DIR/corpus.pcm") <- speech.pcm + music.pcm"
cat "$CORPUS_DIR/speech.pcm" "$CORPUS_DIR/music.pcm" > "$CORPUS_DIR/corpus.pcm"

ls -lh "$CORPUS_DIR"/*.pcm
echo "[fetch-vq-corpus] done — feed corpus.pcm to vq-train --input <path>"

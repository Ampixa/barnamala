#!/usr/bin/env bash
# Download the Devanagari Handwritten Character Dataset (DHCD).
# Source: Acharya, Pant & Gyawali 2015 (UCI ML Repository id 389).
set -euo pipefail
cd "$(dirname "$0")/.."

RAW=data/raw
EXTRACTED=data/extracted
URL="https://archive.ics.uci.edu/static/public/389/devanagari+handwritten+character+dataset.zip"

mkdir -p "$RAW" "$EXTRACTED"
if [ ! -f "$RAW/dhcd.zip" ]; then
    curl -L --fail -o "$RAW/dhcd.zip" "$URL"
fi
unzip -qo "$RAW/dhcd.zip" -d "$EXTRACTED"

TRAIN_DIR="$EXTRACTED/DevanagariHandwrittenCharacterDataset/Train"
TEST_DIR="$EXTRACTED/DevanagariHandwrittenCharacterDataset/Test"

n_train=$(find "$TRAIN_DIR" -name '*.png' | wc -l)
n_test=$(find "$TEST_DIR" -name '*.png' | wc -l)
n_classes=$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo "train images: $n_train (expect 78200)"
echo "test images:  $n_test (expect 13800)"
echo "classes:      $n_classes (expect 46)"
[ "$n_train" -eq 78200 ] && [ "$n_test" -eq 13800 ] && [ "$n_classes" -eq 46 ] \
    && echo "OK: DHCD verified" || { echo "FAIL: counts do not match"; exit 1; }

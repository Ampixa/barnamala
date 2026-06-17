#!/usr/bin/env bash
# Download CMATERdb 3.2.1 Devanagari handwritten numerals (10 classes, 32x32 RGB).
#
# Source: CMATER lab, Jadavpur University. Underlying data CC-BY-4.0
#   (Das et al., "A Statistical-topological Feature Combination for Recognition
#    of Handwritten Numerals", Appl. Soft Comput. 12(8):2486-2495, 2012).
# We fetch the NumPy-packaged mirror (Apache-2.0 packaging) from prabhuomkar/CMATERdb;
# 2500 train + 500 test images, 300/class. Independent provenance from DHCD
# (Acharya & Pant, UCI 389) — the basis for the distribution-shift transfer test.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=data/cmaterdb/devanagari-numerals
BASE=https://github.com/prabhuomkar/CMATERdb/raw/master/datasets/devanagari-numerals

mkdir -p "$DEST"
for f in training-images.npz testing-images.npz; do
    if [ ! -f "$DEST/$f" ]; then
        curl -L --fail -o "$DEST/$f" "$BASE/$f"
    fi
done
echo "CMATERdb 3.2.1 Devanagari numerals -> $DEST"
ls -l "$DEST"

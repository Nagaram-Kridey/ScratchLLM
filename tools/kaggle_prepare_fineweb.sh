#!/usr/bin/env bash
# Run from the repo root in a Kaggle notebook (CPU is enough). Downloads the first four
# FineWeb-Edu sample-10BT files at a pinned revision, records their sizes and hashes,
# trains the 32,000-entry tokenizer and writes token files to /kaggle/working.
set -euo pipefail
REV=87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
URL=https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/$REV/sample/10BT
RAW=/tmp/fineweb-edu
OUT=/kaggle/working
SEP="<|endoftext|>"
mkdir -p "$RAW" "$OUT"
df -h /tmp "$OUT"
for i in 000 001 002 003; do
  curl -fsSL --retry 5 -o "$RAW/${i}_00000.parquet" "$URL/${i}_00000.parquet"
done
(cd "$RAW" && stat -c '%n %s' *.parquet && sha256sum *.parquet) | tee "$OUT/fineweb-edu-files.txt"
python tools/parquet_to_text.py "$RAW/fineweb-edu.txt" "$SEP" "$RAW"/*.parquet
rm "$RAW"/*.parquet
time python -m scratchllm tokenizer-train --input "$RAW/fineweb-edu.txt" --separator "$SEP" \
  --vocab 32000 --output "$OUT/tokenizer.json"
time python -m scratchllm prepare --input "$RAW/fineweb-edu.txt" --separator "$SEP" \
  --tokenizer "$OUT/tokenizer.json" --out "$OUT/data"
ls -l "$OUT/data"

#!/usr/bin/env bash
# One Kaggle GPU session of `small` training. Run from the repo root.
# Inputs attached to the notebook: the scratchllm-data output (tokenizer.json, data/) and,
# from the second session on, this notebook's previous output (run/ckpt.pt, run/log.jsonl).
# Each session trains up to STEPS more steps, but stops after HOURS of training whatever the speed,
# so it always ends inside Kaggle's 12-hour limit with its output saved. Checkpoints are written
# atomically every 250 steps, so a stop loses at most the steps since the last one.
set -euo pipefail
STEPS=${STEPS:-3500}
HOURS=${HOURS:-10.5}
RUN=/kaggle/working/run
DATA=$(dirname "$(find /kaggle/input -path '*data/train.bin' | head -1)")
TOKENIZER=$(find /kaggle/input -name tokenizer.json | head -1)
PREVIOUS=$(find /kaggle/input -path '*run/ckpt.pt' | head -1 || true)
echo "data: $DATA  tokenizer: $TOKENIZER  previous: ${PREVIOUS:-none}"
# Training reads 128 random windows per step; /kaggle/input is slow at random reads, so copy
# the token files to local disk first (about 6 GB).
time cp -r "$DATA" /tmp/data
DATA=/tmp/data
nvidia-smi --query-gpu=name,memory.total --format=csv
# Loss lines only appear every 250 steps (about 40 minutes), so show GPU use every 5 minutes.
(while sleep 300; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; done) &
mkdir -p "$RUN"
cp "$TOKENIZER" /kaggle/working/tokenizer.json
if [ -n "$PREVIOUS" ]; then
  cp "$(dirname "$PREVIOUS")"/ckpt.pt "$(dirname "$PREVIOUS")"/log.jsonl "$RUN"/
  DONE=$(python -c "import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])['step'])" "$RUN/log.jsonl")
  RESUME=--resume
else
  DONE=0
  RESUME=
fi
echo "steps done: $DONE, this session stops at step $((DONE + STEPS)) or after $HOURS hours"
STATUS=0
CUDA_VISIBLE_DEVICES=0 timeout --signal=INT "${HOURS}h" python -m scratchllm train \
  --config configs/small.toml --data "$DATA" --run "$RUN" $RESUME --stop-at $((DONE + STEPS)) || STATUS=$?
if [ "$STATUS" -eq 124 ]; then echo "stopped after $HOURS hours"; elif [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi
tail -3 "$RUN/log.jsonl"

#!/usr/bin/env bash
# One Kaggle GPU session of `small` training. Run from the repo root.
# Inputs attached to the notebook: the scratchllm-data output (tokenizer.json, data/) and,
# from the second session on, this notebook's previous output (run/ckpt.pt, run/log.jsonl).
# Each session trains STEPS more steps (about 10 hours on a T4) and leaves run/ in /kaggle/working.
set -euo pipefail
STEPS=${STEPS:-3500}
RUN=/kaggle/working/run
DATA=$(dirname "$(find /kaggle/input -path '*data/train.bin' | head -1)")
TOKENIZER=$(find /kaggle/input -name tokenizer.json | head -1)
PREVIOUS=$(find /kaggle/input -path '*run/ckpt.pt' | head -1 || true)
echo "data: $DATA  tokenizer: $TOKENIZER  previous: ${PREVIOUS:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv
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
echo "steps done: $DONE, this session stops at $((DONE + STEPS))"
CUDA_VISIBLE_DEVICES=0 python -m scratchllm train --config configs/small.toml --data "$DATA" \
  --run "$RUN" $RESUME --stop-at $((DONE + STEPS))
tail -3 "$RUN/log.jsonl"

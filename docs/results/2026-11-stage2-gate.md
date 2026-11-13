# Stage 2 gate: speed of `target` and the cost of Stage 3

Measured on 13 November 2026. Prices were read from https://nebius.com/prices the same day.

## What was measured

| | |
|---|---|
| Where | Kaggle notebook, free tier, one NVIDIA Tesla T4 (16 GB) |
| Model | `target`, 512,851,200 parameters, context 2048, vocabulary 32,000 |
| Config | `configs/target-speedtest.toml`: 1 sequence × 64 accumulated = 131,072 tokens per step, same as `target.toml` |
| Precision | float16 with loss scaling (the T4 has no native bfloat16) |
| Data | random token ids; speed does not depend on the text |
| Result | **2,453 tokens/s** over steps 1–10, **2,418 tokens/s** over steps 11–20 |

The loss stayed finite in float16, and the model fits in 16 GB at one sequence per micro-batch.

## What Stage 3 needs

`target.toml` plans 100,000 steps × 131,072 tokens = **13.1 billion tokens**.

- At the T4 speed this is about 1,500 GPU-hours, or about 50 weeks of Kaggle's 30 free hours a week. The free tier cannot do Stage 3.
- The cost at the T4 speed works out to about 9.3 TFLOPS achieved, from roughly 3.8 GFLOP per token (6 × parameters, plus attention).

## Estimate for a rented H100 (not measured)

Assumption: the same code reaches 25–35% of the H100's 989 bfloat16 TFLOPS. That gives 65,000–90,000 tokens/s and about 40–56 hours for pretraining, plus about 10% for chat tuning: **44–62 GPU-hours**.

| Nebius H100, price on 13 Nov 2026 | One run (44–62 h) | With one full retry (88–124 h) |
|---|---|---|
| On-demand, $4.50/h | $198–$279 | $396–$558 |
| Preemptible, from $0.79/h (changes every 15 minutes) | from $35–$49 | from $70–$98 |

A preemptible machine can be stopped by Nebius at any time. Atomic checkpoints and `--resume` cover that.

## Before money is spent

1. **Measure on the real GPU.** The H100 figure above is an estimate. Thirty minutes on a preemptible H100 costs about $0.40 and replaces it with a measurement.
2. **Data.** TinyStories has 534 million tokens. 13.1 billion tokens of general English needs a larger corpus, licence-checked like the Stage 1 sets, and about 26 GB of token files.
3. **Owner's choice at the gate:** approve a fixed amount, choose a smaller model, or stop.

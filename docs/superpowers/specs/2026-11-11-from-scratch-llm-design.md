# From-scratch English chat LLM: design

11 November 2026. Status: draft for owner review.

## 1. Purpose

Build an original language model from scratch: own tokenizer, own model code, weights starting from random. The owner asked for this on 9 November 2026 as a personal project, with a target of about 512M parameters. It is separate from Derek, derek-core and Kryam.

Success means both of these:

1. The finished model holds a general English conversation: it answers in turns, follows simple instructions and answers common questions.
2. Every part is written in this repo. PyTorch supplies tensors, automatic gradients and GPU maths only. No Hugging Face model, tokenizer or trainer classes.

The model will be clearly weaker than Qwen3-4B and will make factual mistakes. That is accepted.

## 2. Decisions from the owner

| Topic | Decision |
|---|---|
| Goal | General English chat, every part built in this repo |
| Compute | Staged: laptop first, GPU later |
| Depth | Own code on PyTorch |
| Budget | ₹0 until the gate; then a costed quote and a fixed amount approved by the owner |
| Release | Code and weights public under Apache-2.0 |
| Design | Modern decoder-only transformer |

## 3. Constraints

- Laptop: i7-13620H, 15.7 GB RAM, no CUDA GPU, Windows 11.
- Python 3.11, started with `py -V:Astral/CPython3.11.15`. Virtual environment in `.venv`, git-ignored.
- Dependencies: `torch` (CPU build on the laptop) and `pytest`. Nothing else without a stated reason.
- Local git only. No GitHub repository until the owner asks.
- Package name `scratchllm` is a working name; the owner can rename the project.

## 4. Out of scope

Indian languages, tool calling, JSON tasks, Derek integration, retrieval, a web interface, quantised export, and safety tuning beyond choosing clean data. Each can be a later project.

## 5. Components

Each unit has one job and can be tested alone.

### 5.1 Tokenizer (`scratchllm/tokenizer.py`)

- Byte-level BPE. Any text encodes without unknown tokens and decodes back to the same bytes.
- `train(text_iter, vocab_size) -> Tokenizer`, `encode(str) -> list[int]`, `decode(list[int]) -> str`, `save(path)`, `load(path)`.
- The trainer runs on a text sample (up to 100 MB), not the whole corpus.
- Special tokens, never produced by encoding ordinary text: `<|bos|>`, `<|eos|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>`.
- Vocabulary: 8,192 for the laptop model, 32,000 for the target model.

### 5.2 Model (`scratchllm/model.py`)

One decoder-only transformer, sized by a config:

- token embedding, shared with the output layer;
- per layer: RMSNorm → causal self-attention with rotary position encoding → RMSNorm → SwiGLU feed-forward with hidden width 8/3 × model width; residual connections around both;
- final RMSNorm, then output logits;
- no bias terms.

Attention is written out explicitly. Training may use PyTorch's fused attention kernel for speed; a test proves both give the same output.

| Config | Layers | Width | Heads | Vocabulary | Context | Parameters |
|---|---|---|---|---|---|---|
| `tiny` | 8 | 384 | 6 | 8,192 | 512 | about 17M |
| `target` | 24 | 1280 | 20 | 32,000 | 2,048 | about 513M |

Interface: `Model(config)`, `forward(tokens, targets=None) -> (logits, loss)`, `count_parameters()`.

### 5.3 Data (`scratchllm/data.py`, `scratchllm/registry.py`)

- `data/registry.toml` lists every dataset: source URL, pinned revision, licence, licence URL, file sizes and SHA-256 hashes. This follows the registry-and-verifier pattern from derek-core, re-implemented here because the repos are separate.
- `LICENSES.md` is generated from the registry.
- A dataset is used only if its licence allows an Apache-2.0 release of code and weights. The licence is read from its primary page before download. Anything unclear is left out.
- `prepare` verifies hashes, tokenizes, and writes `train.bin` and `val.bin` as flat arrays of 16-bit token ids.
- The loader memory-maps those files and returns random windows of `context + 1` tokens.

Candidate datasets, each to be licence-checked before use:

- Stage 1: TinyStories, Simple English Wikipedia.
- Stage 3: full English Wikipedia, public-domain books, permissively licensed conversation data.

### 5.4 Training (`scratchllm/train.py`)

- AdamW, linear warm-up then cosine decay, gradient clipping at 1.0, gradient accumulation.
- Mixed precision on GPU; full precision on CPU.
- A checkpoint holds model weights, optimizer state, step, random-generator state and config. It is written to a temporary file and then renamed, so a cut-off session never leaves a broken checkpoint.
- `--resume` continues from the latest checkpoint.
- Each evaluation interval logs training loss, held-out loss and tokens per second to a JSON-lines file.
- A non-finite loss stops the run and keeps the last good checkpoint.

### 5.5 Chat tuning (`scratchllm/chat_format.py`, reusing `train.py`)

- Format: `<|bos|><|user|>text<|end|><|assistant|>text<|end|>`, repeated per turn.
- Loss is counted on assistant tokens only.
- Stage 1 fallback, if no conversation dataset passes the licence check: question-and-answer pairs built from Simple English Wikipedia ("What is X?" answered by the article's first sentence).

### 5.6 Sampling and chat (`scratchllm/sample.py`, `scratchllm/cli.py`)

- Temperature and top-p sampling with a seed. Generation stops at `<|end|>` or a token limit.
- Commands: `tokenizer-train`, `prepare`, `train`, `sample`, `chat`, `eval`, `verify`, `licences`.

### 5.7 Evals (`scratchllm/evals.py`)

- Held-out loss and perplexity.
- A fixed set of 20 prompts, sampled with a fixed seed at every checkpoint and saved for reading.
- A chat set of 50 questions, each with a list of accepted answer strings. The score is the share of replies that contain an accepted string. The set is written before training and never used as training data.
- Results are saved as one JSON file per checkpoint.

## 6. Stages and gates

### Stage 1: laptop, ₹0

Build every component and train `tiny` end to end.

Done when all of these hold:

- all tests pass;
- held-out loss falls steadily and ends below the loss of a model that only knows word frequencies;
- stopping and resuming a run gives the same weights as an uninterrupted run with the same seed;
- the chat-tuned `tiny` model replies in the turn format and stops with `<|end|>`;
- the owner reads the fixed-prompt samples and judges them grammatical simple English.

### Stage 2: gate

- Run the `target` config for about 30 minutes on a free cloud GPU and record tokens per second.
- Quote = planned training tokens ÷ measured speed × the rental price read that day, plus chat tuning and one retry.
- The owner approves a fixed amount, or a smaller model, or stops. Nothing is spent before this approval.

### Stage 3: GPU run

Pretrain at the approved size, chat-tune, run the evals, then publish code, weights, the licence table and a write-up with the measured results.

## 7. Testing

Test-first for every unit. Key tests:

- tokenizer round-trip on arbitrary bytes; special tokens never produced from ordinary text;
- parameter counts match the table for both configs;
- changing a later token never changes an earlier position's output (causality);
- explicit and fused attention agree;
- a small model drives loss near zero on one repeated batch;
- checkpoint, resume and compare weights;
- chat loss mask covers assistant tokens only;
- sampling with the same seed gives the same text;
- registry rejects a file with a wrong hash or a missing licence.

## 8. Repository layout

```
scratchllm/        tokenizer.py model.py data.py registry.py train.py
                   chat_format.py sample.py evals.py cli.py
configs/           tiny.toml target.toml
data/registry.toml
evals/             prompts.txt chat_set.jsonl
tests/
docs/superpowers/  specs/ plans/
LICENSE            Apache-2.0
LICENSES.md        generated
```

Datasets, token files and checkpoints live in git-ignored `data/raw/`, `data/bin/` and `runs/`.

## 9. Risks

- **512M may cost more than the owner wants.** The gate handles this with a measured quote and the option to shrink.
- **Open-licence conversation data may be thin.** The Wikipedia question-and-answer fallback keeps the pipeline working; chat quality would be lower.
- **A pure-Python BPE trainer is slow.** It trains on a sample; encoding the full corpus runs once and is cached.
- **Free GPU sessions cut off.** Atomic checkpoints and resume cover this.

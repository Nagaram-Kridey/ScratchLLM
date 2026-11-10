# ScratchLLM Stage 1 (Laptop Pipeline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build every component of the from-scratch chat LLM and train the `tiny` (about 17M parameter) model end to end on the laptop at ₹0.

**Architecture:** One Python package, `scratchllm`, with one file per component: tokenizer, model, registry, data, training, chat format, sampling, evals, CLI. Model size comes only from a TOML config, so the same code later runs the 513M `target` config on a GPU. Stages 2 (gate) and 3 (GPU run) are not in this plan.

**Tech Stack:** Python 3.11, PyTorch (CPU build), pytest. Standard library for everything else.

**Spec:** `docs/superpowers/specs/2026-11-11-from-scratch-llm-design.md`

## Global Constraints

- Python is started with `py -V:Astral/CPython3.11.15`. The virtual environment is `.venv` (git-ignored). Run everything as `.venv/Scripts/python.exe -m ...`.
- Dependencies: `torch` and `pytest` only. No `numpy`, `transformers`, `tokenizers`, `datasets` or `sentencepiece` imports.
- PyTorch supplies tensors, automatic gradients and GPU maths only. No `nn.Transformer*`, `nn.MultiheadAttention` or `nn.LayerNorm`.
- Write each test before its code, watch it fail, then make it pass.
- Commit locally after each task. `origin` is `https://github.com/Nagaram-Kridey/ScratchLLM.git`; push only when the owner says so.
- No dataset is downloaded until its licence is read from its primary page, recorded in `data/registry.toml`, and found to allow an Apache-2.0 release of code and weights.
- `data/raw/`, `data/bin/`, `runs/` and `.venv/` are git-ignored.
- ₹0 spend in this plan.

## Review Focus

1. **Prompt longer than the context** in `sample` or `chat`: generation uses the last `context` tokens and does not crash. Test in Task 7.
2. **User types a special-token string** such as `<|end|>` in chat: it is encoded as ordinary text, never as the special token. Test in Task 2, used in Task 8.
3. **Token file shorter than `context + 1`:** the loader raises `ValueError` naming the file and the needed length. Test in Task 5.
4. **`--resume` with no checkpoint in the run folder:** `FileNotFoundError` naming the folder; no silent fresh start. Test in Task 6.
5. **Dataset text with invalid UTF-8 or a decode that splits a multi-byte character:** reading and `decode` use `errors="replace"` and do not raise. Tests in Tasks 2 and 5.

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml`, `.gitignore`, `LICENSE` | Project metadata, ignores, Apache-2.0 text |
| `configs/tiny.toml`, `configs/target.toml` | Model and training sizes |
| `scratchllm/config.py` | Config dataclasses and loader |
| `scratchllm/tokenizer.py` | Byte-level BPE |
| `scratchllm/model.py` | Transformer |
| `scratchllm/registry.py` | Dataset registry, hash check, licence table |
| `scratchllm/data.py` | Text readers, token files, batch loader |
| `scratchllm/train.py` | Training loop, checkpoints, resume |
| `scratchllm/chat_format.py` | Chat encoding, assistant mask, chat token files |
| `scratchllm/wiki_qa.py` | Question-and-answer pairs from a Wikipedia dump (fallback) |
| `scratchllm/sample.py` | Generation |
| `scratchllm/evals.py` | Held-out loss, baseline, prompt samples, chat score |
| `scratchllm/cli.py`, `scratchllm/__main__.py` | Commands |
| `evals/prompts.txt`, `evals/chat_set.jsonl` | Fixed eval inputs |

---

### Task 1: Project setup and configs

**Files:** Create `pyproject.toml`, `.gitignore`, `LICENSE`, `configs/tiny.toml`, `configs/target.toml`, `scratchllm/__init__.py`, `scratchllm/config.py`. Test: `tests/test_config.py`.

**Interfaces — Produces:**

```python
@dataclass(frozen=True)
class ModelConfig: n_layers: int; d_model: int; n_heads: int; vocab_size: int; context: int
@dataclass(frozen=True)
class TrainConfig:
    batch_size: int; grad_accum: int; lr: float; warmup_steps: int; max_steps: int
    weight_decay: float; eval_interval: int; eval_batches: int; seed: int
def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]
```

TOML has `[model]` and `[train]` tables with exactly those keys.

| | tiny | target |
|---|---|---|
| n_layers / d_model / n_heads | 8 / 384 / 6 | 24 / 1280 / 20 |
| vocab_size / context | 8192 / 512 | 32000 / 2048 |
| batch_size / grad_accum | 16 / 2 | 8 / 8 |
| lr / warmup_steps / max_steps | 6e-4 / 200 / 5000 | 3e-4 / 2000 / 100000 |
| weight_decay / eval_interval / eval_batches / seed | 0.1 / 250 / 20 / 1337 | 0.1 / 1000 / 50 / 1337 |

- [ ] **Step 1:** Create the venv and install: `py -V:Astral/CPython3.11.15 -m venv .venv`, then `.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install pytest`. Expected: `python -c "import torch; print(torch.__version__)"` prints a version ending `+cpu`.
- [ ] **Step 2:** Write `tests/test_config.py`: `load_config("configs/tiny.toml")` returns `ModelConfig(8, 384, 6, 8192, 512)` and `TrainConfig.max_steps == 5000`; `target` returns `ModelConfig(24, 1280, 20, 32000, 2048)`; a TOML with a missing or unknown key raises `ValueError` naming the key.
- [ ] **Step 3:** Run `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`. Expected: FAIL, module not found.
- [ ] **Step 4:** Write the files. `.gitignore`: `.venv/`, `data/raw/`, `data/bin/`, `runs/`, `__pycache__/`, `.pytest_cache/`. `LICENSE`: the standard Apache-2.0 text. `pyproject.toml`: name `scratchllm`, version `0.1.0`, `requires-python = ">=3.11"`, dependency `torch`, pytest `testpaths = ["tests"]`.
- [ ] **Step 5:** Run the test. Expected: PASS.
- [ ] **Step 6:** Commit: `feat: project setup and configs`.

### Task 2: Tokenizer

**Files:** Create `scratchllm/tokenizer.py`. Test: `tests/test_tokenizer.py`.

**Interfaces — Produces:**

```python
SPECIALS = ["<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|end|>", "<|pad|>"]
class Tokenizer:
    vocab_size: int
    special: dict[str, int]          # "<|bos|>" -> 256 ... "<|pad|>" -> 261
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "Tokenizer"
    def encode(self, text: str) -> list[int]          # never emits special ids
    def decode(self, ids: list[int]) -> str           # specials decode to their strings
    def save(self, path) -> None                      # JSON
    @classmethod
    def load(cls, path) -> "Tokenizer"
```

Fixed decisions: ids 0–255 are bytes, 256–261 are the specials in the order above, merges start at 262. Text is split with the regex ` ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+` and merges never cross a split. Training counts each distinct chunk once with its frequency and picks the most frequent pair, ties broken by the lower pair of ids. `decode` joins bytes and decodes UTF-8 with `errors="replace"`. `train` raises `ValueError` if `vocab_size < 262`.

- [ ] **Step 1:** Write tests:
  - `test_roundtrip`: for `"Hello, world! 123\n\ttabs"`, `"naïve café 日本語 🙂"` and `""`, `decode(encode(s)) == s`.
  - `test_vocab_size`: training on a repeated paragraph with `vocab_size=300` gives `tok.vocab_size == 300` and every id from `encode` is `< 300`.
  - `test_merges_compress`: after training on `["the cat sat on the mat "] * 200` with `vocab_size=280`, `len(encode("the cat")) < len("the cat".encode())`.
  - `test_specials_never_from_text`: `encode("<|end|> <|user|>")` contains no id in 256–261 and round-trips.
  - `test_special_ids`: `tok.special["<|bos|>"] == 256` and `tok.special["<|pad|>"] == 261`; `decode([tok.special["<|end|>"]]) == "<|end|>"`.
  - `test_decode_partial_utf8`: `decode([0xE6])` returns a string and does not raise.
  - `test_save_load`: a loaded tokenizer encodes a sample identically.
  - `test_deterministic`: training twice on the same text gives equal saved files.
  - `test_too_small_vocab`: `train([...], 100)` raises `ValueError`.
- [ ] **Step 2:** Run `pytest tests/test_tokenizer.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `Tokenizer`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: byte-level BPE tokenizer`.

### Task 3: Model

**Files:** Create `scratchllm/model.py`. Test: `tests/test_model.py`.

**Interfaces — Consumes:** `ModelConfig`. **Produces:**

```python
class Model(torch.nn.Module):
    use_fused: bool = True        # False = explicit attention maths
    def __init__(self, cfg: ModelConfig)
    def forward(self, tokens: LongTensor[B, T], targets: LongTensor[B, T] | None = None
                ) -> tuple[Tensor[B, T, vocab], Tensor | None]
    def count_parameters(self) -> int
```

Fixed decisions: layer order as in spec section 5.2. Feed-forward hidden width is `int(8 * d_model / 3)`; SwiGLU is `w_out(silu(w_gate(x)) * w_up(x))`. Rotary encoding is applied to queries and keys with base 10000. No bias anywhere. Output layer shares the embedding weight. Weights start from a normal distribution with standard deviation 0.02. Loss is cross-entropy that ignores targets equal to `-1`. `forward` raises `ValueError` if `T > cfg.context`. The fused path is `F.scaled_dot_product_attention(..., is_causal=True)`.

- [ ] **Step 1:** Write tests (use a small config `ModelConfig(2, 48, 4, 100, 16)` unless stated):
  - `test_param_counts`: `Model(tiny).count_parameters() == 17_308_032`; `Model(target).count_parameters() == 512_851_200` (construct `target` on CPU; skip if memory allocation fails).
  - `test_shapes`: logits are `[B, T, 100]`; loss is `None` without targets and a scalar with them.
  - `test_causal`: changing the token at position 10 leaves logits at positions 0–9 unchanged (`atol=1e-5`).
  - `test_fused_matches_explicit`: same weights, `use_fused` True and False, logits agree (`atol=1e-4`).
  - `test_ignore_index`: targets of all `-1` except one position give the same loss as that single position's cross-entropy.
  - `test_overfits_one_batch`: 200 AdamW steps at `lr=1e-2` on one fixed batch bring loss below 0.1.
  - `test_too_long_raises`: 17 tokens with `context=16` raises `ValueError`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement `RMSNorm`, rotary helper, `Attention`, `FeedForward`, `Block`, `Model`.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: decoder-only transformer`.

### Task 4: Dataset registry and licence table

**Files:** Create `scratchllm/registry.py`, `data/registry.toml` (empty list for now). Test: `tests/test_registry.py`.

**Interfaces — Produces:**

```python
@dataclass(frozen=True)
class DataFile: name: str; size: int; sha256: str
@dataclass(frozen=True)
class Dataset: id: str; url: str; revision: str; licence: str; licence_url: str; files: tuple[DataFile, ...]
class RegistryError(Exception)
def load_registry(path) -> list[Dataset]
def verify(ds: Dataset, raw_dir) -> None        # files at raw_dir/<ds.id>/<file.name>
def sha256_file(path) -> str
def render_licences(datasets: list[Dataset]) -> str   # markdown table: id, licence, licence URL, source, revision
```

TOML shape: `[[dataset]]` tables with `[[dataset.file]]` sub-tables.

- [ ] **Step 1:** Write tests: a valid registry loads; an entry with empty `licence`, `licence_url` or `revision` raises `RegistryError` naming the id and field; duplicate ids raise; `verify` passes on a matching temp file; `verify` raises `RegistryError` on wrong size, wrong hash and missing file, each message naming the file; `render_licences` output contains every id and licence and is identical on a second call.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement with `tomllib` and `hashlib`, reading files in 1 MB chunks.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: dataset registry and licence table`.

### Task 5: Token files and batch loader

**Files:** Create `scratchllm/data.py`. Test: `tests/test_data.py`.

**Interfaces — Consumes:** `Tokenizer`. **Produces:**

```python
def read_text_docs(path, separator: str) -> Iterator[str]     # utf-8, errors="replace"; skips empty docs
def prepare(docs: Iterable[str], tok: Tokenizer, out_dir, val_fraction: float = 0.01) -> dict
    # writes out_dir/train.bin and val.bin; returns {"train_tokens": int, "val_tokens": int}
def get_batch(bin_path, batch_size: int, context: int, gen: torch.Generator
              ) -> tuple[LongTensor[B, context], LongTensor[B, context]]
```

Fixed decisions: token files are little-endian signed 16-bit (vocabulary is at most 32,000). Each document is written as its tokens followed by `<|eos|>`. Every 100th document goes to `val.bin` when `val_fraction=0.01` (document index modulo `round(1 / val_fraction)` equals 0), so the split is deterministic. `get_batch` memory-maps with `mmap` and `torch.frombuffer`; `y` is `x` shifted by one.

- [ ] **Step 1:** Write tests: `read_text_docs` splits on the separator, skips blanks, and survives a file containing byte `0xFF`; `prepare` on 300 short docs writes both files, token counts match the returned dict, and decoding `train.bin` reproduces the documents with `<|eos|>` between them; `get_batch` returns shapes `[4, 8]`, dtype `int64`, `y[:, :-1] == x[:, 1:]`, and the same generator seed gives the same batch; a file with 5 tokens and `context=8` raises `ValueError` containing the path and `9`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: token files and batch loader`.

### Task 6: Training loop, checkpoints and resume

**Files:** Create `scratchllm/train.py`. Test: `tests/test_train.py`.

**Interfaces — Consumes:** `Model`, `ModelConfig`, `TrainConfig`, `get_batch`, `assistant_mask` (Task 8; imported lazily when `chat=True`). **Produces:**

```python
class TrainingDiverged(Exception)
def lr_at(step: int, cfg: TrainConfig) -> float
def train(model_cfg: ModelConfig, train_cfg: TrainConfig, data_dir, run_dir, *,
          resume: bool = False, init_from=None, chat: bool = False,
          special: dict[str, int] | None = None) -> None
def load_model(ckpt_path, device="cpu") -> Model       # rebuilds from the stored config
```

Fixed decisions:
- `lr_at`: linear from 0 to `lr` over `warmup_steps`, then cosine to `0.1 * lr` at `max_steps`.
- AdamW betas `(0.9, 0.95)`; weight decay on tensors with two or more dimensions only; gradient clipping at 1.0; one optimizer step per `grad_accum` batches.
- Device is `cuda` when available, else `cpu`. On `cuda`, use `torch.autocast` with `bfloat16`.
- Checkpoint `run_dir/ckpt.pt` holds `model`, `optimizer`, `step`, `generator` (the batch generator's state), `model_cfg`, `train_cfg`. It is saved to `ckpt.pt.tmp` and moved with `os.replace`, at every `eval_interval` and at the end.
- `run_dir/log.jsonl` gets one line per eval: `step`, `train_loss`, `val_loss`, `lr`, `tokens_per_sec`.
- `val_loss` is the mean over `eval_batches` batches drawn with a fixed generator seeded `seed + 1`.
- A non-finite loss raises `TrainingDiverged`; the last saved checkpoint is left untouched.
- `resume=True` with no `ckpt.pt` raises `FileNotFoundError` naming `run_dir`.
- `init_from` loads model weights only from another checkpoint.
- `chat=True` sets `y` to `-1` wherever `assistant_mask` is False.

- [ ] **Step 1:** Write tests with a tiny config (`ModelConfig(2, 32, 2, 300, 16)`, `max_steps=20`, `eval_interval=10`, `batch_size=4`, `grad_accum=1`) on a synthetic token file:
  - `test_lr_schedule`: `lr_at(0) == 0`, `lr_at(warmup_steps) == lr`, `lr_at(max_steps) == pytest.approx(0.1 * lr)`.
  - `test_loss_falls`: the last `val_loss` in `log.jsonl` is lower than the first, on a file made of one repeated 20-token pattern with `max_steps=100`.
  - `test_resume_matches_uninterrupted`: run A does 20 steps; run B uses the same `TrainConfig` with `stop_at=10`, then resumes to 20. Model weights are exactly equal.
  - `test_resume_without_checkpoint`: raises `FileNotFoundError`.
  - `test_atomic_save`: after training, `ckpt.pt.tmp` does not exist.
  - `test_diverged`: with `lr=1e9` the run raises `TrainingDiverged` and any existing `ckpt.pt` still loads.
  - `test_load_model`: `load_model(ckpt)` returns a model whose logits match the trained one.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Add keyword `stop_at: int | None = None` to `train` (stop after that step, saving a checkpoint) for the resume test and for timed runs.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: training loop with atomic checkpoints and resume`.

### Task 7: Sampling

**Files:** Create `scratchllm/sample.py`. Test: `tests/test_sample.py`.

**Interfaces — Consumes:** `Model`. **Produces:**

```python
def generate(model: Model, ids: list[int], max_new_tokens: int, *, temperature: float = 0.8,
             top_p: float = 0.95, seed: int = 0, stop_id: int | None = None) -> list[int]
    # returns only the new tokens; stop_id is not included
```

Fixed decisions: `temperature == 0` means take the highest-scoring token. Top-p keeps the smallest set of tokens whose probability sum reaches `top_p`. Only the last `model.cfg.context` tokens are fed in. Runs under `torch.no_grad()` in eval mode. Empty `ids` raises `ValueError`.

- [ ] **Step 1:** Write tests: same seed gives the same output and a different seed gives a different one (random-weight model, 30 tokens); `temperature=0` ignores the seed; with a model whose logits are patched to always favour token 7, `stop_id=7` returns `[]`; `top_p=1e-9` equals greedy; a prompt of `context + 50` tokens generates 5 tokens without error; `ids=[]` raises `ValueError`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: temperature and top-p sampling`.

### Task 8: Chat format and chat token files

**Files:** Create `scratchllm/chat_format.py`. Test: `tests/test_chat_format.py`.

**Interfaces — Consumes:** `Tokenizer`. **Produces:**

```python
def encode_chat(turns: list[tuple[str, str]], tok: Tokenizer) -> list[int]
    # <|bos|> then, per turn, <|user|> user <|end|> <|assistant|> assistant <|end|>
def encode_prompt(history: list[tuple[str, str]], user: str, tok: Tokenizer) -> list[int]
    # same, ending with <|user|> user <|end|> <|assistant|>
def assistant_mask(ids: LongTensor[B, T], assistant_id: int, end_id: int) -> BoolTensor[B, T]
    # True for tokens after an <|assistant|> up to and including its <|end|>
def prepare_chat(jsonl_path, tok: Tokenizer, out_dir, exclude_questions: set[str] = frozenset()) -> dict
    # input lines: {"turns": [["user text", "assistant text"], ...]}; writes train.bin and val.bin
def normalise_question(text: str) -> str    # lower-case, strip, collapse spaces, drop trailing "?"
```

Fixed decisions: user and assistant text go through `tok.encode`, so special strings typed by a user stay ordinary text. `prepare_chat` drops any conversation whose first user text, normalised, is in `exclude_questions`. Split is the same every-100th rule as Task 5. A window that starts in the middle of an assistant reply leaves that partial reply unmasked.

- [ ] **Step 1:** Write tests: `encode_chat([("hi", "hello")])` equals `[bos, user, *enc("hi"), end, assistant, *enc("hello"), end]`; `encode_prompt` ends with the `assistant` id; user text `"<|end|>"` adds no special ids beyond the structural ones; `assistant_mask` on that sequence is True exactly on `enc("hello")` and the final `end`, and False on the `assistant` marker itself; a two-turn conversation masks both replies; `prepare_chat` with `exclude_questions={"what is a cat"}` drops `"What is a cat?"` and keeps the rest; a line without `turns` raises `ValueError` with the line number.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. `assistant_mask` is vectorised with cumulative sums, no Python loop over tokens.
- [ ] **Step 4:** Add `test_chat_training_masks` to `tests/test_train.py`: with `chat=True`, a batch's targets are `-1` outside the mask (check through a small helper `apply_chat_mask(x, y, special)` exported from `train.py`).
- [ ] **Step 5:** Run both test files. Expected: PASS.
- [ ] **Step 6:** Commit: `feat: chat format and assistant-only loss`.

### Task 9: Evals

**Files:** Create `scratchllm/evals.py`, `evals/prompts.txt`, `evals/chat_set.jsonl`. Test: `tests/test_evals.py`.

**Interfaces — Consumes:** `Model`, `Tokenizer`, `get_batch`, `generate`, `encode_prompt`, `normalise_question`. **Produces:**

```python
def heldout_loss(model, bin_path, batches: int, batch_size: int, seed: int) -> float
def unigram_loss(train_bin, val_bin, vocab_size: int) -> float      # add-one smoothed, nats per token
def run_prompts(model, tok, prompts: list[str], seed: int, max_new_tokens: int = 120) -> list[str]
def load_chat_set(path) -> list[dict]      # {"question": str, "accepted": [str, ...]}
def chat_score(model, tok, chat_set: list[dict], seed: int) -> tuple[float, list[dict]]
    # reply counts if it contains any accepted string, case-insensitive
def chat_set_questions(path) -> set[str]   # normalised, for prepare_chat's exclude_questions
def write_result(path, result: dict) -> None    # JSON, sorted keys
```

- [ ] **Step 1:** Write `evals/prompts.txt`: exactly 20 story openings, one per line, in simple English (for example `Once upon a time, there was a little dog named`).
- [ ] **Step 2:** Write `evals/chat_set.jsonl`: exactly 50 lines, 10 in each group: animals ("What sound does a dog make?"), colours and everyday objects, counting and sums below 20, opposites ("What is the opposite of hot?"), simple definitions ("What is a river?"). Each has 1–4 short accepted strings.
- [ ] **Step 3:** Write tests: `unigram_loss` on a file of one repeated token is below 0.1, and on uniformly random tokens over 50 ids is within 0.1 of `ln(50)` (pass `vocab_size=50`); `heldout_loss` is deterministic for a seed; `chat_score` with a stubbed generator returning `"A dog says WOOF."` scores a `["woof"]` item as correct and a `["meow"]` item as wrong; the files have exactly 20 and 50 entries, every chat item has a non-empty `accepted` list, and no question repeats after `normalise_question`; `write_result` output loads back equal.
- [ ] **Step 4:** Run. Expected: FAIL.
- [ ] **Step 5:** Implement.
- [ ] **Step 6:** Run. Expected: PASS.
- [ ] **Step 7:** Commit: `feat: evals and fixed eval sets`.

### Task 10: Command line

**Files:** Create `scratchllm/cli.py`, `scratchllm/__main__.py`. Test: `tests/test_cli.py`.

**Interfaces — Consumes:** everything above. **Produces** `python -m scratchllm <command>`:

| Command | Arguments | Does |
|---|---|---|
| `verify` | `--registry data/registry.toml --raw data/raw [--id ID]` | Hash-checks files; exit code 1 on failure |
| `licences` | `--registry ... --output LICENSES.md` | Writes the table |
| `tokenizer-train` | `--input FILE --separator S --vocab N --sample-mb 100 --output tok.json` | Trains on the first `sample-mb` of text |
| `prepare` | `--input FILE --separator S --tokenizer tok.json --out DIR` | Writes token files |
| `prepare-chat` | `--input FILE.jsonl --tokenizer tok.json --out DIR --exclude evals/chat_set.jsonl` | Writes chat token files |
| `train` | `--config C --data DIR --run DIR [--resume] [--init CKPT] [--chat --tokenizer tok.json] [--stop-at N]` | Trains |
| `sample` | `--ckpt C --tokenizer T --prompt P [--max-new 200 --temperature 0.8 --top-p 0.95 --seed 0]` | Prints text |
| `chat` | `--ckpt C --tokenizer T` | Terminal chat; empty line asks again; `/quit` exits |
| `eval` | `--ckpt C --tokenizer T --data DIR --out FILE.json [--chat]` | Writes `val_loss`, `perplexity`, `unigram_loss`, samples, and `chat_score` with `--chat` |

- [ ] **Step 1:** Write tests that call `cli.main([...])` on temp folders: the chain `tokenizer-train` → `prepare` → `train` (20 steps, tiny test config) → `sample` → `eval` runs and each output file exists; `verify` returns 1 for a bad hash; `chat` with stdin `"\nhello\n/quit\n"` prints one reply and exits 0; an unknown command exits with code 2.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement with `argparse` subcommands. `chat` keeps history and uses `encode_prompt` with `stop_id=<|end|>`.
- [ ] **Step 4:** Run the full suite: `.venv/Scripts/python.exe -m pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: command line`.

### Task 11: Data licence check and download

**Files:** Modify `data/registry.toml`. Create `LICENSES.md` (generated). Create `scratchllm/wiki_qa.py` and `tests/test_wiki_qa.py` only if the fallback below is needed.

- [ ] **Step 1:** For TinyStories and for each candidate open conversation dataset, open the primary page, read the licence, and record source URL, pinned revision, licence name and licence URL. Report the findings to the owner before downloading anything.
- [ ] **Step 2:** For each dataset that passes, download the plain-text files to `data/raw/<id>/`, record size and SHA-256 in the registry, and run `python -m scratchllm verify`. Expected: exit code 0.
- [ ] **Step 3:** Run `python -m scratchllm licences --output LICENSES.md`.
- [ ] **Step 4 (only if no conversation dataset passes):** build the Wikipedia fallback. `wiki_qa.py` produces `iter_first_sentences(dump_path) -> Iterator[tuple[str, str]]` (title, first sentence; reads the `.xml.bz2` dump with `bz2` and `xml.etree.ElementTree.iterparse`; skips redirects, non-article namespaces and pages whose cleaned text is empty) and `write_qa_jsonl(pairs, out_path) -> int` (one line per page: `{"turns": [["What is <title>?", "<first sentence>"]]}`). Cleaning removes `{{...}}` templates (nested), `<ref>...</ref>`, turns `[[a|b]]` into `b` and `[[a]]` into `a`, and strips `'''` and `''`. Tests feed a small hand-written dump covering each rule. The Simple English Wikipedia dump goes through Steps 1–3 first.
- [ ] **Step 5:** Commit: `data: registry entries and licence table`.

### Task 12: Train `tiny` and check the Stage 1 criteria

**Files:** Create `docs/results/2026-11-stage1.md`.

- [ ] **Step 1:** `tokenizer-train --vocab 8192` on the pretraining text, then `prepare` into `data/bin/pretrain/`.
- [ ] **Step 2:** Measure speed: `train --config configs/tiny.toml --run runs/speed --stop-at 30` and read `tokens_per_sec`. Set `max_steps` in `configs/tiny.toml` to `floor(8 * 3600 * tokens_per_sec / (16 * 2 * 512))`, so the full run takes about eight hours. Tell the owner the figure before starting.
- [ ] **Step 3:** Pretrain: `train --config configs/tiny.toml --data data/bin/pretrain --run runs/tiny-pretrain`, in the background. Stop it once at an arbitrary point and restart with `--resume` to prove resume on the real run.
- [ ] **Step 4:** `prepare-chat` into `data/bin/chat/`, then chat-tune: `train --config configs/tiny.toml --data data/bin/chat --run runs/tiny-chat --init runs/tiny-pretrain/ckpt.pt --chat`, with `max_steps` set to one tenth of the pretraining value and `lr` 1e-4 in a copy `configs/tiny-chat.toml`.
- [ ] **Step 5:** Run `eval` on both checkpoints (`--chat` on the second).
- [ ] **Step 6:** Check each Stage 1 criterion from spec section 6 and write `docs/results/2026-11-stage1.md` with: the full test run output, final `val_loss` against `unigram_loss`, the loss curve as a table from `log.jsonl`, the resume evidence, ten chat exchanges showing turn format and `<|end|>` stops, the chat score, and the 20 prompt samples. State plainly any criterion that is not met.
- [ ] **Step 7:** Give the owner the samples to read. The owner's judgement on grammatical simple English is the last criterion.
- [ ] **Step 8:** Commit: `docs: stage 1 results`.

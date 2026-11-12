"""Evals: held-out loss, the word-frequency baseline, fixed prompts and the fixed chat set."""

import json
from pathlib import Path

import torch

from .chat_format import chat_reply, normalise_question
from .data import read_tokens
from .sample import generate
from .train import mean_loss as heldout_loss  # noqa: F401  (same seeded batches as training's val loss)

_COUNT_CHUNK = 1 << 24


def _counts(bin_path, vocab_size: int) -> torch.Tensor:
    data = read_tokens(bin_path)
    counts = torch.zeros(vocab_size, dtype=torch.float64)
    for start in range(0, len(data), _COUNT_CHUNK):
        counts += torch.bincount(data[start : start + _COUNT_CHUNK].long(), minlength=vocab_size)
    return counts


def unigram_loss(train_bin, val_bin, vocab_size: int) -> float:
    """Loss in nats per token of guessing from token frequencies alone (add-one smoothed)."""
    train = _counts(train_bin, vocab_size) + 1
    val = _counts(val_bin, vocab_size)
    return float(-(val * (train / train.sum()).log()).sum() / val.sum())


def load_prompts(path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run_prompts(model, tok, prompts: list[str], seed: int, max_new_tokens: int = 120) -> list[str]:
    """Each prompt followed by the model's continuation, stopping at <|eos|>."""
    eos = tok.special["<|eos|>"]
    return [
        prompt + tok.decode(generate(model, tok.encode(prompt), max_new_tokens, seed=seed + i, stop_id=eos))
        for i, prompt in enumerate(prompts)
    ]


def load_chat_set(path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                question, accepted = item["question"], item["accepted"]
                if not isinstance(question, str) or not accepted or not all(isinstance(a, str) for a in accepted):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                raise ValueError(
                    f'{path}: line {number} is not {{"question": text, "accepted": [text, ...]}}'
                ) from None
            items.append({"question": question, "accepted": accepted})
    return items


def chat_set_questions(path) -> set[str]:
    """The chat set's questions, normalised, so prepare_chat can keep them out of training."""
    return {normalise_question(item["question"]) for item in load_chat_set(path)}


def chat_score(model, tok, chat_set: list[dict]) -> tuple[float, list[dict]]:
    """Fraction of questions whose most likely reply contains an accepted string, and every exchange."""
    details = []
    for item in chat_set:
        reply, stopped = chat_reply(model, tok, [], item["question"], temperature=0)
        correct = any(accepted.lower() in reply.lower() for accepted in item["accepted"])
        details.append({"question": item["question"], "reply": reply, "correct": correct, "stopped": stopped})
    return sum(d["correct"] for d in details) / len(details), details


def write_result(path, result: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

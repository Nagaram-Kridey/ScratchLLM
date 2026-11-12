import array
import json
import math
from pathlib import Path

import pytest
import torch

from scratchllm import evals
from scratchllm.chat_format import normalise_question
from scratchllm.config import ModelConfig
from scratchllm.model import Model
from scratchllm.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent


def _bin(path, ids):
    with open(path, "wb") as f:
        array.array("h", ids).tofile(f)
    return path


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.train(["once upon a time there was a little dog. "] * 50, 300)


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    return Model(ModelConfig(2, 32, 2, tok.vocab_size, 16)).eval()


def test_unigram_loss_one_repeated_token(tmp_path):
    train, val = _bin(tmp_path / "train.bin", [7] * 10_000), _bin(tmp_path / "val.bin", [7] * 1_000)
    assert evals.unigram_loss(train, val, 50) < 0.1


def test_unigram_loss_uniform_tokens(tmp_path):
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 50, (120_000,), generator=gen).tolist()
    train, val = _bin(tmp_path / "train.bin", ids[:100_000]), _bin(tmp_path / "val.bin", ids[100_000:])
    assert evals.unigram_loss(train, val, 50) == pytest.approx(math.log(50), abs=0.1)


def test_unigram_loss_counts_in_chunks(tmp_path, monkeypatch):
    ids = [i % 50 for i in range(5_000)]
    train, val = _bin(tmp_path / "train.bin", ids), _bin(tmp_path / "val.bin", ids)
    whole = evals.unigram_loss(train, val, 50)
    monkeypatch.setattr(evals, "_COUNT_CHUNK", 333)
    assert evals.unigram_loss(train, val, 50) == pytest.approx(whole)


def test_heldout_loss_is_deterministic(model, tmp_path):
    path = _bin(tmp_path / "val.bin", [i % 200 for i in range(2_000)])
    a = evals.heldout_loss(model, path, 3, 4, seed=5)
    assert a == evals.heldout_loss(model, path, 3, 4, seed=5)
    assert a != evals.heldout_loss(model, path, 3, 4, seed=6)


def test_run_prompts_returns_prompt_and_continuation(model, tok):
    out = evals.run_prompts(model, tok, ["once upon", "a little dog"], seed=1, max_new_tokens=5)
    assert len(out) == 2
    assert out[0].startswith("once upon") and out[1].startswith("a little dog")
    assert out == evals.run_prompts(model, tok, ["once upon", "a little dog"], seed=1, max_new_tokens=5)


def test_chat_score(monkeypatch):
    monkeypatch.setattr(evals, "chat_reply", lambda *a, **k: ("A dog says WOOF.", True))
    chat_set = [
        {"question": "What sound does a dog make?", "accepted": ["woof"]},
        {"question": "What sound does a cat make?", "accepted": ["meow"]},
    ]
    score, details = evals.chat_score(None, None, chat_set)
    assert score == 0.5
    assert [d["correct"] for d in details] == [True, False]
    assert details[0] == {
        "question": "What sound does a dog make?", "reply": "A dog says WOOF.", "correct": True, "stopped": True,
    }


def test_eval_files():
    prompts = evals.load_prompts(ROOT / "evals" / "prompts.txt")
    assert len(prompts) == 20 and len(set(prompts)) == 20
    chat_set = evals.load_chat_set(ROOT / "evals" / "chat_set.jsonl")
    assert len(chat_set) == 50
    for item in chat_set:
        assert 1 <= len(item["accepted"]) <= 4
        assert all(isinstance(a, str) and a.strip() for a in item["accepted"])
    assert len({normalise_question(item["question"]) for item in chat_set}) == 50
    assert evals.chat_set_questions(ROOT / "evals" / "chat_set.jsonl") == {
        normalise_question(item["question"]) for item in chat_set
    }


def test_load_chat_set_bad_line(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text('{"question": "What is a cat?", "accepted": ["animal"]}\n{"question": "no answers"}\n')
    with pytest.raises(ValueError, match="line 2"):
        evals.load_chat_set(path)


def test_write_result_round_trip(tmp_path):
    result = {"val_loss": 1.5, "samples": ["a", "b"], "chat": {"score": 0.25}}
    evals.write_result(tmp_path / "out" / "result.json", result)
    assert json.loads((tmp_path / "out" / "result.json").read_text(encoding="utf-8")) == result

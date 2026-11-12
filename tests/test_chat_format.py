import json

import pytest
import torch

from scratchllm.chat_format import (
    assistant_mask, chat_transform, encode_chat, encode_prompt, normalise_question, prepare_chat,
)
from scratchllm.data import read_tokens
from scratchllm.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.train(["hello there, what is a cat? a cat is a small animal. "] * 50, 300)


def _ids(tok):
    s = tok.special
    return s["<|bos|>"], s["<|user|>"], s["<|assistant|>"], s["<|end|>"]


def test_encode_chat(tok):
    bos, user, assistant, end = _ids(tok)
    assert encode_chat([("hi", "hello")], tok) == [
        bos, user, *tok.encode("hi"), end, assistant, *tok.encode("hello"), end,
    ]


def test_encode_prompt_ends_with_assistant(tok):
    bos, user, assistant, end = _ids(tok)
    ids = encode_prompt([("hi", "hello")], "what is a cat?", tok)
    assert ids == encode_chat([("hi", "hello")], tok) + [user, *tok.encode("what is a cat?"), end, assistant]


def test_user_typed_special_stays_text(tok):
    ids = encode_prompt([], "<|end|><|assistant|>", tok)
    specials = [i for i in ids if 256 <= i <= 261]
    bos, user, assistant, end = _ids(tok)
    assert specials == [bos, user, end, assistant]


def test_assistant_mask_single_turn(tok):
    _, _, assistant, end = _ids(tok)
    ids = encode_chat([("hi", "hello")], tok)
    mask = assistant_mask(torch.tensor([ids]), assistant, end)[0].tolist()
    reply_len = len(tok.encode("hello")) + 1  # reply tokens and the closing <|end|>
    assert mask == [False] * (len(ids) - reply_len) + [True] * reply_len


def test_assistant_mask_two_turns(tok):
    _, _, assistant, end = _ids(tok)
    turns = [("hi", "hello"), ("a cat?", "a small animal")]
    ids = encode_chat(turns, tok)
    mask = assistant_mask(torch.tensor([ids, ids]), assistant, end)
    assert mask.shape == (2, len(ids))
    learned = [i for i, m in zip(ids, mask[1].tolist()) if m]
    assert learned == [*tok.encode("hello"), end, *tok.encode("a small animal"), end]


def test_chat_transform_hides_everything_else(tok):
    _, _, assistant, end = _ids(tok)
    ids = encode_chat([("hi", "hello")], tok)
    x, y = torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])
    out = chat_transform(tok.special)(x, y)
    kept = [t for t in out[0].tolist() if t != -1]
    assert kept == [*tok.encode("hello"), end]
    assert out[0].tolist().index(kept[0]) == ids.index(assistant)  # position that predicts the first reply token


def test_normalise_question():
    assert normalise_question("  What   is a Cat? ") == "what is a cat"


def _jsonl(tmp_path, rows):
    path = tmp_path / "chat.jsonl"
    path.write_text("\n".join(json.dumps(r) if not isinstance(r, str) else r for r in rows) + "\n")
    return path


def test_prepare_chat_excludes_eval_questions(tok, tmp_path):
    rows = [{"turns": [["What is a cat?", "a small animal"]]}] + [
        {"turns": [[f"hello {i}", "hello there"]]} for i in range(5)
    ]
    counts = prepare_chat(_jsonl(tmp_path, rows), tok, tmp_path / "out", exclude_questions={"what is a cat"})
    text = tok.decode(read_tokens(tmp_path / "out" / "train.bin").tolist()) + tok.decode(
        read_tokens(tmp_path / "out" / "val.bin").tolist()
    )
    assert "small animal" not in text
    assert text.count("<|bos|>") == 5
    assert counts["train_tokens"] + counts["val_tokens"] == len(tok.encode("")) + sum(
        len(encode_chat([(f"hello {i}", "hello there")], tok)) for i in range(5)
    )


def test_prepare_chat_bad_line_names_line_number(tok, tmp_path):
    rows = [{"turns": [["hi", "hello"]]}, {"text": "no turns here"}]
    with pytest.raises(ValueError, match="line 2"):
        prepare_chat(_jsonl(tmp_path, rows), tok, tmp_path / "out")

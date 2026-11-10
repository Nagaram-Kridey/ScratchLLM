import array

import pytest
import torch

from scratchllm.data import get_batch, prepare, read_text_docs, read_tokens
from scratchllm.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.train(["the cat sat on the mat. the dog ran to the park. "] * 50, 300)


def _write_tokens(path, ids):
    with open(path, "wb") as f:
        array.array("h", ids).tofile(f)
    return path


def test_read_text_docs(tmp_path):
    path = tmp_path / "docs.txt"
    path.write_bytes(b"first doc\n<|endoftext|>\n\n<|endoftext|>second \xff doc<|endoftext|>")
    docs = list(read_text_docs(path, "<|endoftext|>"))
    assert len(docs) == 2
    assert docs[0] == "first doc"
    assert docs[1].startswith("second") and docs[1].endswith("doc")


def test_read_text_docs_separator_across_read_boundary(tmp_path):
    path = tmp_path / "docs.txt"
    path.write_text("a" * 10 + "<SEP>" + "b" * 10)
    assert list(read_text_docs(path, "<SEP>", chunk_chars=12)) == ["a" * 10, "b" * 10]


def test_prepare_writes_both_files(tmp_path, tok):
    docs = [f"the cat sat {i}" for i in range(300)]
    counts = prepare(docs, tok, tmp_path)
    train = read_tokens(tmp_path / "train.bin").tolist()
    val = read_tokens(tmp_path / "val.bin").tolist()
    assert counts == {"train_tokens": len(train), "val_tokens": len(val)}
    eos = tok.special["<|eos|>"]
    assert val.count(eos) == 3 and train.count(eos) == 297
    expected_val = "<|eos|>".join(docs[i] for i in (0, 100, 200)) + "<|eos|>"
    assert tok.decode(val) == expected_val
    assert tok.decode(train).startswith("the cat sat 1<|eos|>the cat sat 2<|eos|>")


def test_get_batch(tmp_path):
    path = _write_tokens(tmp_path / "t.bin", list(range(1000)))
    x, y = get_batch(path, 4, 8, torch.Generator().manual_seed(0))
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y[:, -1], x[:, -1] + 1)
    x2, _ = get_batch(path, 4, 8, torch.Generator().manual_seed(0))
    assert torch.equal(x, x2)


def test_get_batch_reaches_both_ends(tmp_path):
    path = _write_tokens(tmp_path / "t.bin", list(range(9)))
    x, y = get_batch(path, 2, 8, torch.Generator().manual_seed(0))
    assert x[0].tolist() == list(range(8)) and y[0].tolist() == list(range(1, 9))


def test_get_batch_file_too_short(tmp_path):
    path = _write_tokens(tmp_path / "short.bin", [1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match=r"short\.bin.*9"):
        get_batch(path, 2, 8, torch.Generator().manual_seed(0))

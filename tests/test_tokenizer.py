import pytest

from scratchllm.tokenizer import SPECIALS, Tokenizer

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "A little girl named Lily went to the park with her dog, and they played all day. "
)


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.train([PARAGRAPH] * 50, 300)


@pytest.mark.parametrize("text", ["Hello, world! 123\n\ttabs", "naïve café 日本語 🙂", ""])
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_vocab_size(tok):
    assert tok.vocab_size == 300
    assert all(i < 300 for i in tok.encode(PARAGRAPH))


def test_merges_compress():
    tok = Tokenizer.train(["the cat sat on the mat "] * 200, 280)
    assert len(tok.encode("the cat")) < len("the cat".encode())


def test_specials_never_from_text(tok):
    text = "<|end|> <|user|>"
    ids = tok.encode(text)
    assert not any(256 <= i <= 261 for i in ids)
    assert tok.decode(ids) == text


def test_special_ids(tok):
    assert SPECIALS[0] == "<|bos|>"
    assert tok.special["<|bos|>"] == 256
    assert tok.special["<|pad|>"] == 261
    assert tok.decode([tok.special["<|end|>"]]) == "<|end|>"


def test_decode_partial_utf8(tok):
    assert isinstance(tok.decode([0xE6]), str)


def test_save_load(tok, tmp_path):
    tok.save(tmp_path / "tok.json")
    loaded = Tokenizer.load(tmp_path / "tok.json")
    assert loaded.encode(PARAGRAPH) == tok.encode(PARAGRAPH)
    assert loaded.vocab_size == tok.vocab_size


def test_deterministic(tmp_path):
    for name in ("a.json", "b.json"):
        Tokenizer.train([PARAGRAPH] * 20, 290).save(tmp_path / name)
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_too_small_vocab():
    with pytest.raises(ValueError):
        Tokenizer.train([PARAGRAPH], 100)

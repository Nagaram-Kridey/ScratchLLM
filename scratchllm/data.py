"""Text readers, token files and the batch loader.

Token files are flat little-endian signed 16-bit ids (the vocabulary is at most 32,000).
"""

import array
import mmap
import sys
import warnings
from pathlib import Path
from typing import Iterable, Iterator

import torch

from .tokenizer import Tokenizer

_FLUSH_TOKENS = 1 << 20


def read_text_docs(path, separator: str, chunk_chars: int = 1 << 20) -> Iterator[str]:
    """Yield the non-empty documents of a text file, split on separator, without loading it whole."""
    pending = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while chunk := f.read(chunk_chars):
            *docs, pending = (pending + chunk).split(separator)
            for doc in docs:
                if doc := doc.strip():
                    yield doc
    if pending := pending.strip():
        yield pending


class TokenWriter:
    """Appends token ids to a file as 16-bit values."""

    def __init__(self, path):
        self._file = open(path, "wb")
        self._buffer = array.array("h")
        self.count = 0

    def write(self, ids: list[int]) -> None:
        self._buffer.extend(ids)
        self.count += len(ids)
        if len(self._buffer) >= _FLUSH_TOKENS:
            self._flush()

    def _flush(self) -> None:
        if sys.byteorder == "big":
            self._buffer.byteswap()
        self._buffer.tofile(self._file)
        self._buffer = array.array("h")

    def close(self) -> None:
        self._flush()
        self._file.close()


def write_split(sequences: Iterable[list[int]], out_dir, val_fraction: float = 0.01) -> dict:
    """Write token sequences to train.bin and val.bin. Every Nth sequence goes to val."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    every = round(1 / val_fraction)
    train, val = TokenWriter(out_dir / "train.bin"), TokenWriter(out_dir / "val.bin")
    try:
        for index, ids in enumerate(sequences):
            (val if index % every == 0 else train).write(ids)
    finally:
        train.close()
        val.close()
    return {"train_tokens": train.count, "val_tokens": val.count}


def prepare(docs: Iterable[str], tok: Tokenizer, out_dir, val_fraction: float = 0.01) -> dict:
    """Tokenize documents, each followed by <|eos|>, into train.bin and val.bin."""
    eos = tok.special["<|eos|>"]
    return write_split((tok.encode(doc) + [eos] for doc in docs), out_dir, val_fraction)


def read_tokens(path) -> torch.Tensor:
    """Memory-map a token file as a 1-D int16 tensor."""
    with open(path, "rb") as f:
        if f.seek(0, 2) == 0:
            return torch.empty(0, dtype=torch.int16)
        mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    with warnings.catch_warnings():
        # torch warns that a read-only buffer must not be written to; nothing here writes to it.
        warnings.simplefilter("ignore", UserWarning)
        return torch.frombuffer(mapped, dtype=torch.int16)


def get_batch(bin_path, batch_size: int, context: int, gen: torch.Generator):
    """Random windows from a token file: x is [B, context], y is x shifted by one."""
    data = read_tokens(bin_path)
    if len(data) < context + 1:
        raise ValueError(
            f"{bin_path} holds {len(data)} tokens; a context of {context} needs at least {context + 1}"
        )
    starts = torch.randint(0, len(data) - context, (batch_size,), generator=gen).tolist()
    x = torch.stack([data[s : s + context] for s in starts]).long()
    y = torch.stack([data[s + 1 : s + context + 1] for s in starts]).long()
    return x, y

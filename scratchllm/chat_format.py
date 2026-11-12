"""Chat turn format, assistant-only loss mask and chat token files.

One turn is  <|user|> text <|end|> <|assistant|> text <|end|>  and a conversation starts with <|bos|>.
"""

import json
import re
from typing import Iterator

import torch

from .data import write_split
from .tokenizer import Tokenizer


def encode_chat(turns, tok: Tokenizer) -> list[int]:
    """Token ids of a whole conversation given as (user text, assistant text) pairs."""
    s = tok.special
    ids = [s["<|bos|>"]]
    for user, assistant in turns:
        ids += [s["<|user|>"], *tok.encode(user), s["<|end|>"]]
        ids += [s["<|assistant|>"], *tok.encode(assistant), s["<|end|>"]]
    return ids


def encode_prompt(history, user: str, tok: Tokenizer) -> list[int]:
    """Token ids that end ready for the model to write the next reply."""
    s = tok.special
    return encode_chat(history, tok) + [s["<|user|>"], *tok.encode(user), s["<|end|>"], s["<|assistant|>"]]


def _reply_open(ids: torch.Tensor, assistant_id: int, end_id: int) -> torch.Tensor:
    """[B, T] Bool: True where the latest marker at or before the position is <|assistant|>."""
    index = torch.arange(ids.shape[-1], device=ids.device).expand_as(ids)
    none = torch.full_like(index, -1)
    last_assistant = torch.where(ids == assistant_id, index, none).cummax(dim=-1).values
    last_end = torch.where(ids == end_id, index, none).cummax(dim=-1).values
    return last_assistant > last_end


def assistant_mask(ids: torch.Tensor, assistant_id: int, end_id: int) -> torch.Tensor:
    """[B, T] Bool: True on reply tokens and the <|end|> that closes each reply."""
    is_open = _reply_open(ids, assistant_id, end_id)
    mask = torch.zeros_like(is_open)
    mask[:, 1:] = is_open[:, :-1]
    return mask


def chat_transform(special: dict):
    """Return transform(x, y) -> y that hides every target that is not part of a reply.

    A window that starts in the middle of a reply has no marker before those tokens, so they are hidden too.
    """
    assistant_id, end_id = special["<|assistant|>"], special["<|end|>"]

    def transform(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y.masked_fill(~_reply_open(x, assistant_id, end_id), -1)

    return transform


def normalise_question(text: str) -> str:
    """Lower case, letters and digits only, single spaces: used to match questions loosely."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _read_conversations(path) -> Iterator[list]:
    with open(path, encoding="utf-8", errors="replace") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                turns = json.loads(line)["turns"]
                if not turns or any(len(turn) != 2 or not all(isinstance(t, str) for t in turn) for turn in turns):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                raise ValueError(
                    f'{path}: line {number} is not {{"turns": [[user text, assistant text], ...]}}'
                ) from None
            yield turns


def prepare_chat(path, tok: Tokenizer, out_dir, exclude_questions=frozenset(), val_fraction: float = 0.01) -> dict:
    """Write a JSON-lines chat file to train.bin and val.bin, leaving out conversations
    whose first question (normalised) is in exclude_questions."""
    sequences = (
        encode_chat(turns, tok)
        for turns in _read_conversations(path)
        if normalise_question(turns[0][0]) not in exclude_questions
    )
    return write_split(sequences, out_dir, val_fraction)

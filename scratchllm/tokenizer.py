"""Byte-level BPE tokenizer.

ids 0-255 are raw bytes, 256-261 are the special tokens, merges start at 262.
"""

import heapq
import json
import re
from collections import Counter, defaultdict
from typing import Iterable

SPECIALS = ["<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|end|>", "<|pad|>"]
FIRST_MERGE_ID = 256 + len(SPECIALS)
# Merges never cross these chunks, so a word keeps its leading space.
_SPLIT = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")
_CACHE_LIMIT = 1_000_000


def _merge(symbols: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i, n = [], 0, len(symbols)
    while i < n:
        if i + 1 < n and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(m) for m in merges]
        self.vocab_size = FIRST_MERGE_ID + len(self.merges)
        self.special = {name: 256 + i for i, name in enumerate(SPECIALS)}
        self._rank = {pair: FIRST_MERGE_ID + i for i, pair in enumerate(self.merges)}
        self._bytes = [bytes([i]) for i in range(256)] + [s.encode() for s in SPECIALS]
        for a, b in self.merges:
            self._bytes.append(self._bytes[a] + self._bytes[b])
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "Tokenizer":
        if vocab_size < FIRST_MERGE_ID:
            raise ValueError(f"vocab_size must be at least {FIRST_MERGE_ID}, got {vocab_size}")
        chunk_counts: Counter[str] = Counter()
        for text in texts:
            chunk_counts.update(_SPLIT.findall(text))
        words = [list(chunk.encode()) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for w, symbols in enumerate(words):
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += freqs[w]
                where[pair].add(w)
        # Most frequent first; ties go to the lower pair of ids.
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < vocab_size - FIRST_MERGE_ID and heap:
            neg, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if count != -neg:  # stale entry: re-queue with the current count
                if count > 0:
                    heapq.heappush(heap, (-count, pair))
                continue
            new_id = FIRST_MERGE_ID + len(merges)
            merges.append(pair)
            touched = set()
            for w in where.pop(pair):
                old = words[w]
                for p in zip(old, old[1:]):
                    pair_counts[p] -= freqs[w]
                    touched.add(p)
                new = words[w] = _merge(old, pair, new_id)
                for p in zip(new, new[1:]):
                    pair_counts[p] += freqs[w]
                    where[p].add(w)
                    touched.add(p)
            pair_counts.pop(pair, None)
            for p in touched:
                if pair_counts.get(p, 0) > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        symbols = list(chunk.encode())
        while len(symbols) > 1:
            best = min(zip(symbols, symbols[1:]), key=lambda p: self._rank.get(p, 1 << 30))
            if best not in self._rank:
                break
            symbols = _merge(symbols, best, self._rank[best])
        return symbols

    def encode(self, text: str) -> list[int]:
        """Encode ordinary text. Never emits a special id, whatever the text says."""
        ids: list[int] = []
        cache = self._cache
        for chunk in _SPLIT.findall(text):
            found = cache.get(chunk)
            if found is None:
                found = self._encode_chunk(chunk)
                if len(cache) < _CACHE_LIMIT:
                    cache[chunk] = found
            ids.extend(found)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._bytes[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "merges": self.merges}, f)

    @classmethod
    def load(cls, path) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["merges"])

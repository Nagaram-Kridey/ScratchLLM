import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from parquet_to_text import convert  # noqa: E402

from scratchllm.data import read_text_docs

SEP = "<|endoftext|>"


def test_parquet_to_text_round_trip(tmp_path):
    texts = ["First page.\nTwo lines.", "", "bad <|endoftext|> doc", "Second page.", None, "Third."]
    pq.write_table(pa.table({"text": texts[:3], "id": [1, 2, 3]}), tmp_path / "a.parquet")
    pq.write_table(pa.table({"text": texts[3:], "id": [4, 5, 6]}), tmp_path / "b.parquet")
    written, skipped = convert(tmp_path / "out.txt", SEP, [tmp_path / "a.parquet", tmp_path / "b.parquet"])
    assert (written, skipped) == (3, 3)
    assert list(read_text_docs(tmp_path / "out.txt", SEP)) == ["First page.\nTwo lines.", "Second page.", "Third."]

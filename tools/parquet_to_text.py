"""Write the `text` column of Parquet files to one text file, documents joined by a separator,
for `tokenizer-train` and `prepare`. Reads in batches, so files larger than memory are fine.

Needs pyarrow, which the model code itself does not use:
    python tools/parquet_to_text.py out.txt "<|endoftext|>" in1.parquet [in2.parquet ...]
"""

import sys

import pyarrow.parquet as pq


def convert(out_path, separator, parquet_paths, column="text"):
    """Returns the number of documents written. A document containing the separator is skipped."""
    written = skipped = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for path in parquet_paths:
            for batch in pq.ParquetFile(path).iter_batches(columns=[column], batch_size=10_000):
                for text in batch.column(0).to_pylist():
                    if not text or not text.strip() or separator in text:
                        skipped += 1
                        continue
                    if written:
                        out.write(separator)
                    out.write(text)
                    written += 1
    return written, skipped


if __name__ == "__main__":
    out, sep, *inputs = sys.argv[1:]
    written, skipped = convert(out, sep, inputs)
    print(f"{written} documents written to {out}, {skipped} skipped")

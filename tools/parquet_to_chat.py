"""Turn Parquet files with a `messages` column (role/content) into the chat
JSON-lines format of `prepare-chat`: {"turns": [[user text, assistant text], ...]}.

Needs pyarrow, which the model code itself does not use:
    python tools/parquet_to_chat.py out.jsonl in1.parquet [in2.parquet ...]
"""

import json
import sys

import pyarrow.parquet as pq


def turns_of(messages):
    """Pairs of (user, assistant) text, or None if the roles do not alternate.
    A last user message with no reply is dropped."""
    if len(messages) % 2 and messages[-1]["role"] == "user":
        messages = messages[:-1]
    roles = [m["role"] for m in messages]
    if not messages or len(messages) % 2 or roles != ["user", "assistant"] * (len(messages) // 2):
        return None
    texts = [m["content"].strip() for m in messages]
    if not all(texts):
        return None
    return [texts[i:i + 2] for i in range(0, len(texts), 2)]


def main(out_path, *parquet_paths):
    kept = skipped = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for path in parquet_paths:
            for messages in pq.read_table(path, columns=["messages"]).column("messages").to_pylist():
                turns = turns_of(messages)
                if turns is None:
                    skipped += 1
                    continue
                out.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")
                kept += 1
    print(f"{kept} conversations written to {out_path}, {skipped} skipped")


if __name__ == "__main__":
    main(*sys.argv[1:])

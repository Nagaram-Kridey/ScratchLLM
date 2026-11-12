import hashlib
import io
import json
from pathlib import Path

import pytest

from scratchllm import cli
from scratchllm.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SEP = "<|endoftext|>"

CONFIG = """
[model]
n_layers = 2
d_model = 32
n_heads = 2
vocab_size = {vocab}
context = 16

[train]
batch_size = 4
grad_accum = 1
lr = 1e-3
warmup_steps = 2
max_steps = 20
weight_decay = 0.1
eval_interval = 10
eval_batches = 2
seed = 1
"""


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Runs tokenizer-train -> prepare -> train once; later tests use the outputs."""
    folder = tmp_path_factory.mktemp("cli")
    stories = [f"once upon a time there was a little dog number {i}. the dog was happy." for i in range(300)]
    (folder / "stories.txt").write_text(SEP.join(stories), encoding="utf-8")
    tok, data, run = folder / "tok.json", folder / "bin", folder / "run"
    assert cli.main([
        "tokenizer-train", "--input", str(folder / "stories.txt"), "--separator", SEP,
        "--vocab", "300", "--output", str(tok),
    ]) == 0
    assert cli.main([
        "prepare", "--input", str(folder / "stories.txt"), "--separator", SEP,
        "--tokenizer", str(tok), "--out", str(data),
    ]) == 0
    (folder / "config.toml").write_text(CONFIG.format(vocab=Tokenizer.load(tok).vocab_size))
    assert cli.main(["train", "--config", str(folder / "config.toml"), "--data", str(data), "--run", str(run)]) == 0
    return folder


def test_chain_writes_every_output(work):
    for name in ("tok.json", "bin/train.bin", "bin/val.bin", "run/ckpt.pt", "run/log.jsonl"):
        assert (work / name).is_file(), name


def test_sample_prints_prompt_and_more(work, capsys):
    code = cli.main([
        "sample", "--ckpt", str(work / "run" / "ckpt.pt"), "--tokenizer", str(work / "tok.json"),
        "--prompt", "once upon", "--max-new", "8",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("once upon") and len(out.strip()) > len("once upon")


def test_eval_writes_result(work):
    out = work / "results" / "eval.json"
    assert cli.main([
        "eval", "--ckpt", str(work / "run" / "ckpt.pt"), "--tokenizer", str(work / "tok.json"),
        "--data", str(work / "bin"), "--out", str(out), "--prompts", str(ROOT / "evals" / "prompts.txt"),
    ]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert set(result) == {"val_loss", "perplexity", "unigram_loss", "samples"}
    assert len(result["samples"]) == 20


def test_chat_tune_and_chat_eval(work):
    rows = [{"turns": [[f"what is dog number {i}?", "a happy little dog."]]} for i in range(300)]
    (work / "chat.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (work / "set.jsonl").write_text(
        json.dumps({"question": "What is dog number 7?", "accepted": ["dog"]}) + "\n", encoding="utf-8"
    )
    tok = str(work / "tok.json")
    assert cli.main([
        "prepare-chat", "--input", str(work / "chat.jsonl"), "--tokenizer", tok,
        "--out", str(work / "chatbin"), "--exclude", str(work / "set.jsonl"),
    ]) == 0
    assert "number 7?" not in Tokenizer.load(tok).decode(
        list((work / "chatbin" / "train.bin").read_bytes()[::2])  # ids below 256 are their own byte
    )
    assert cli.main([
        "train", "--config", str(work / "config.toml"), "--data", str(work / "chatbin"),
        "--run", str(work / "chatrun"), "--init", str(work / "run" / "ckpt.pt"), "--chat", "--tokenizer", tok,
    ]) == 0
    out = work / "chat-eval.json"
    assert cli.main([
        "eval", "--ckpt", str(work / "chatrun" / "ckpt.pt"), "--tokenizer", tok, "--data", str(work / "chatbin"),
        "--out", str(out), "--prompts", str(ROOT / "evals" / "prompts.txt"),
        "--chat", "--chat-set", str(work / "set.jsonl"),
    ]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert 0.0 <= result["chat_score"] <= 1.0
    assert len(result["chat"]) == 1 and result["chat"][0]["question"] == "What is dog number 7?"


def test_chat_flag_needs_tokenizer(work, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["train", "--config", str(work / "config.toml"), "--data", "x", "--run", "y", "--chat"])
    assert stop.value.code == 2


def test_chat_loop(work, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("\nhello\n/quit\n"))
    code = cli.main(["chat", "--ckpt", str(work / "run" / "ckpt.pt"), "--tokenizer", str(work / "tok.json")])
    assert code == 0
    assert capsys.readouterr().out.count("model> ") == 1


def test_chat_ends_when_input_ends(work, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("hello\n"))
    assert cli.main(["chat", "--ckpt", str(work / "run" / "ckpt.pt"), "--tokenizer", str(work / "tok.json")]) == 0


def test_resume_without_checkpoint_is_a_clear_error(work, capsys):
    code = cli.main([
        "train", "--config", str(work / "config.toml"), "--data", str(work / "bin"),
        "--run", str(work / "nothing-here"), "--resume",
    ])
    assert code == 1
    assert "nothing-here" in capsys.readouterr().err


def test_tokenizer_from_another_model_is_refused(work, tmp_path, capsys):
    other = tmp_path / "other.json"
    Tokenizer([]).save(other)
    code = cli.main(["sample", "--ckpt", str(work / "run" / "ckpt.pt"), "--tokenizer", str(other), "--prompt", "hi"])
    assert code == 1
    assert "vocabulary" in capsys.readouterr().err


REGISTRY = """
[[dataset]]
id = "stories"
url = "https://example.org/stories"
revision = "abc123"
licence = "MIT"
licence_url = "https://example.org/licence"

[[dataset.file]]
name = "stories.txt"
size = {size}
sha256 = "{sha}"
"""


def _registry(tmp_path, content, sha=None):
    (tmp_path / "raw" / "stories").mkdir(parents=True)
    (tmp_path / "raw" / "stories" / "stories.txt").write_bytes(content)
    path = tmp_path / "registry.toml"
    path.write_text(REGISTRY.format(size=len(content), sha=sha or hashlib.sha256(content).hexdigest()))
    return ["--registry", str(path)]


def test_verify_good_and_bad_hash(tmp_path, capsys):
    good = _registry(tmp_path / "good", b"once upon a time\n")
    assert cli.main(["verify", *good, "--raw", str(tmp_path / "good" / "raw")]) == 0
    bad = _registry(tmp_path / "bad", b"once upon a time\n", sha="0" * 64)
    assert cli.main(["verify", *bad, "--raw", str(tmp_path / "bad" / "raw")]) == 1
    assert "sha256" in capsys.readouterr().err
    assert cli.main(["verify", *good, "--raw", str(tmp_path / "good" / "raw"), "--id", "nope"]) == 1


def test_licences_writes_table(tmp_path):
    registry = _registry(tmp_path, b"x")
    assert cli.main(["licences", *registry, "--output", str(tmp_path / "LICENSES.md")]) == 0
    assert "| stories | MIT |" in (tmp_path / "LICENSES.md").read_text(encoding="utf-8")


def test_unknown_command_exits_2():
    with pytest.raises(SystemExit) as stop:
        cli.main(["no-such-command"])
    assert stop.value.code == 2

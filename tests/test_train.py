import array
import json
from dataclasses import replace

import pytest
import torch

from scratchllm.config import ModelConfig, TrainConfig
from scratchllm.train import TrainingDiverged, load_model, lr_at, train

MODEL = ModelConfig(2, 32, 2, 300, 16)
TRAIN = TrainConfig(
    batch_size=4, grad_accum=1, lr=1e-3, warmup_steps=2, max_steps=20,
    weight_decay=0.1, eval_interval=10, eval_batches=2, seed=1,
)


@pytest.fixture
def data_dir(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    pattern = list(range(10, 30))
    for name, repeats in (("train.bin", 50), ("val.bin", 10)):
        with open(folder / name, "wb") as f:
            array.array("h", pattern * repeats).tofile(f)
    return folder


def _log(run_dir):
    return [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]


def _weights(run_dir):
    return torch.load(run_dir / "ckpt.pt", weights_only=False)["model"]


def test_lr_schedule():
    cfg = replace(TRAIN, warmup_steps=10, max_steps=100)
    assert lr_at(0, cfg) == 0
    assert lr_at(5, cfg) == pytest.approx(0.5 * cfg.lr)
    assert lr_at(10, cfg) == pytest.approx(cfg.lr)
    assert lr_at(100, cfg) == pytest.approx(0.1 * cfg.lr)
    assert cfg.lr > lr_at(55, cfg) > 0.1 * cfg.lr


def test_loss_falls_and_log_is_written(data_dir, tmp_path):
    run = tmp_path / "run"
    train(MODEL, replace(TRAIN, max_steps=100, lr=1e-2), data_dir, run)
    log = _log(run)
    assert [entry["step"] for entry in log] == list(range(10, 101, 10))
    assert set(log[0]) == {"step", "train_loss", "val_loss", "lr", "tokens_per_sec"}
    assert log[-1]["val_loss"] < log[0]["val_loss"]
    # ln(20) = 3.0 is what knowing only token frequencies gives; lower means it learned the order.
    assert log[-1]["val_loss"] < 2.0


def test_resume_matches_uninterrupted(data_dir, tmp_path):
    train(MODEL, TRAIN, data_dir, tmp_path / "a")
    train(MODEL, TRAIN, data_dir, tmp_path / "b", stop_at=10)
    assert torch.load(tmp_path / "b" / "ckpt.pt", weights_only=False)["step"] == 10
    train(MODEL, TRAIN, data_dir, tmp_path / "b", resume=True)
    a, b = _weights(tmp_path / "a"), _weights(tmp_path / "b")
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), name


def test_resume_without_checkpoint(data_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="empty-run"):
        train(MODEL, TRAIN, data_dir, tmp_path / "empty-run", resume=True)


def test_atomic_save(data_dir, tmp_path):
    train(MODEL, TRAIN, data_dir, tmp_path / "run")
    assert (tmp_path / "run" / "ckpt.pt").exists()
    assert not (tmp_path / "run" / "ckpt.pt.tmp").exists()


def test_diverged_keeps_last_checkpoint(data_dir, tmp_path):
    run = tmp_path / "run"
    train(MODEL, TRAIN, data_dir, run, stop_at=10)
    before = (run / "ckpt.pt").read_bytes()
    with pytest.raises(TrainingDiverged):
        train(MODEL, replace(TRAIN, lr=1e30, warmup_steps=0), data_dir, run, resume=True)
    assert (run / "ckpt.pt").read_bytes() == before
    load_model(run / "ckpt.pt")


def test_init_from_loads_weights_only(data_dir, tmp_path):
    train(MODEL, TRAIN, data_dir, tmp_path / "pre")
    train(MODEL, replace(TRAIN, lr=0.0), data_dir, tmp_path / "tuned", init_from=tmp_path / "pre" / "ckpt.pt")
    pre, tuned = _weights(tmp_path / "pre"), _weights(tmp_path / "tuned")
    for name in pre:
        assert torch.equal(pre[name], tuned[name]), name
    assert torch.load(tmp_path / "tuned" / "ckpt.pt", weights_only=False)["step"] == 20


def test_load_model(data_dir, tmp_path):
    train(MODEL, TRAIN, data_dir, tmp_path / "run")
    model = load_model(tmp_path / "run" / "ckpt.pt")
    assert model.cfg == MODEL
    assert not model.training
    state = _weights(tmp_path / "run")
    assert torch.equal(model.embed.weight, state["embed.weight"])

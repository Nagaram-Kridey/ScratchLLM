"""Model and training sizes, loaded from a TOML file."""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    d_model: int
    n_heads: int
    vocab_size: int
    context: int


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    grad_accum: int
    lr: float
    warmup_steps: int
    max_steps: int
    weight_decay: float
    eval_interval: int
    eval_batches: int
    seed: int


def _build(cls, table: dict, name: str):
    wanted = {f.name for f in fields(cls)}
    missing = wanted - table.keys()
    unknown = table.keys() - wanted
    if missing:
        raise ValueError(f"[{name}] is missing key(s): {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"[{name}] has unknown key(s): {', '.join(sorted(unknown))}")
    return cls(**table)


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return (
        _build(ModelConfig, raw.get("model", {}), "model"),
        _build(TrainConfig, raw.get("train", {}), "train"),
    )

"""Training loop with atomic checkpoints and exact resume."""

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ModelConfig, TrainConfig
from .data import get_batch
from .model import Model


class TrainingDiverged(Exception):
    pass


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warm-up from 0 to lr, then cosine decay to 0.1 * lr at max_steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def amp_dtype(device: str):
    """The mixed-precision type for a device: bfloat16 where the GPU has it natively,
    float16 on older GPUs (such as P100 and T4), none on CPU."""
    if device != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def _autocast(device: str):
    dtype = amp_dtype(device)
    return torch.autocast("cuda", dtype=dtype or torch.bfloat16, enabled=dtype is not None)


def _optimizer(model: Model, cfg: TrainConfig) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def _save(path: Path, model, optimizer, scaler, step, gen, model_cfg, train_cfg) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "generator": gen.get_state(),
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(train_cfg),
        },
        tmp,
    )
    os.replace(tmp, path)  # a cut-off save never leaves a broken ckpt.pt


def _load(path, device="cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def load_model(ckpt_path, device: str = "cpu") -> Model:
    """Rebuild a model from a checkpoint, in eval mode."""
    ckpt = _load(ckpt_path, device)
    model = Model(ModelConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


@torch.no_grad()
def mean_loss(model: Model, bin_path, batches: int, batch_size: int, seed: int, transform=None) -> float:
    """Mean loss over a fixed, seeded set of batches."""
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(bin_path, batch_size, model.cfg.context, gen)
        if transform:
            y = transform(x, y)
        with _autocast(device.type):
            _, loss = model(x.to(device), y.to(device))
        total += loss.item()
    model.train(was_training)
    return total / batches


def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_dir,
    run_dir,
    *,
    resume: bool = False,
    init_from=None,
    stop_at: int | None = None,
    transform=None,
) -> None:
    """Train and write run_dir/ckpt.pt and run_dir/log.jsonl.

    transform(x, y) -> y, if given, rewrites the targets of every batch (used for chat tuning).
    """
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    ckpt_path, log_path = run_dir / "ckpt.pt", run_dir / "log.jsonl"
    device = pick_device()

    torch.manual_seed(train_cfg.seed)
    model = Model(model_cfg).to(device)
    optimizer = _optimizer(model, train_cfg)
    # float16 needs loss scaling so small gradients do not round to zero; elsewhere this does nothing.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype(device) == torch.float16)
    gen = torch.Generator().manual_seed(train_cfg.seed)
    step = 0

    if resume:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"cannot resume: no checkpoint in {run_dir}")
        ckpt = _load(ckpt_path, device)
        if ModelConfig(**ckpt["model_cfg"]) != model_cfg:
            raise ValueError(f"the checkpoint in {run_dir} was trained with a different model config")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        # _load maps every tensor to the training device, but the batch generator lives on the CPU.
        gen.set_state(ckpt["generator"].cpu())
        step = ckpt["step"]
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
        if init_from is not None:
            model.load_state_dict(_load(init_from, device)["model"])

    last_step = min(train_cfg.max_steps, stop_at or train_cfg.max_steps)
    tokens_per_step = train_cfg.batch_size * train_cfg.grad_accum * model_cfg.context
    model.train()
    interval_loss, interval_steps, interval_start = 0.0, 0, time.perf_counter()

    while step < last_step:
        step += 1
        lr = lr_at(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(train_cfg.grad_accum):
            x, y = get_batch(data_dir / "train.bin", train_cfg.batch_size, model_cfg.context, gen)
            if transform:
                y = transform(x, y)
            with _autocast(device):
                _, loss = model(x.to(device), y.to(device))
            if not torch.isfinite(loss):
                raise TrainingDiverged(
                    f"loss became {loss.item()} at step {step}; the last saved checkpoint is untouched"
                )
            scaler.scale(loss / train_cfg.grad_accum).backward()
            step_loss += loss.item() / train_cfg.grad_accum
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)  # skips the update if float16 gradients overflowed
        scaler.update()
        interval_loss += step_loss
        interval_steps += 1

        if step % train_cfg.eval_interval == 0 or step == last_step:
            elapsed = time.perf_counter() - interval_start
            entry = {
                "step": step,
                "train_loss": interval_loss / interval_steps,
                "val_loss": mean_loss(
                    model, data_dir / "val.bin", train_cfg.eval_batches,
                    train_cfg.batch_size, train_cfg.seed + 1, transform,
                ),
                "lr": lr,
                "tokens_per_sec": interval_steps * tokens_per_step / elapsed,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
            _save(ckpt_path, model, optimizer, scaler, step, gen, model_cfg, train_cfg)
            interval_loss, interval_steps, interval_start = 0.0, 0, time.perf_counter()

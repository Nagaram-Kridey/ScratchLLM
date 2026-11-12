import pytest

from scratchllm.config import ModelConfig, load_config


def test_tiny_config():
    model, train = load_config("configs/tiny.toml")
    assert model == ModelConfig(8, 384, 6, 8192, 512)
    assert train.max_steps == 1784  # eight hours at the measured laptop speed


def test_target_config():
    model, _ = load_config("configs/target.toml")
    assert model == ModelConfig(24, 1280, 20, 32000, 2048)


def _write(tmp_path, model_extra="", drop=""):
    text = (
        "[model]\nn_layers = 2\nd_model = 32\nn_heads = 2\nvocab_size = 300\ncontext = 16\n"
        + model_extra
        + "[train]\nbatch_size = 4\ngrad_accum = 1\nlr = 0.001\nwarmup_steps = 2\n"
        "max_steps = 20\nweight_decay = 0.1\neval_interval = 10\neval_batches = 2\nseed = 1\n"
    ).replace(drop, "")
    path = tmp_path / "c.toml"
    path.write_text(text)
    return path


def test_missing_key_named(tmp_path):
    with pytest.raises(ValueError, match="context"):
        load_config(_write(tmp_path, drop="context = 16\n"))


def test_unknown_key_named(tmp_path):
    with pytest.raises(ValueError, match="dropout"):
        load_config(_write(tmp_path, model_extra="dropout = 0.1\n"))

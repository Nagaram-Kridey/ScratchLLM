import pytest
import torch
import torch.nn.functional as F

from scratchllm.config import ModelConfig, load_config
from scratchllm.model import Model

SMALL = ModelConfig(2, 48, 4, 100, 16)


def _tokens(batch=2, length=16, seed=0):
    return torch.randint(0, 100, (batch, length), generator=torch.Generator().manual_seed(seed))


def test_tiny_param_count():
    cfg, _ = load_config("configs/tiny.toml")
    assert Model(cfg).count_parameters() == 17_308_032


def test_target_param_count():
    cfg, _ = load_config("configs/target.toml")
    try:
        model = Model(cfg)
    except (RuntimeError, MemoryError):
        pytest.skip("not enough memory to build the target model")
    assert model.count_parameters() == 512_851_200


def test_shapes():
    model = Model(SMALL)
    x = _tokens()
    logits, loss = model(x)
    assert logits.shape == (2, 16, 100)
    assert loss is None
    _, loss = model(x, x)
    assert loss.shape == ()


@pytest.mark.parametrize("fused", [True, False])
def test_causal(fused):
    torch.manual_seed(0)
    model = Model(SMALL).eval()
    model.use_fused = fused
    a = _tokens(1)
    b = a.clone()
    b[0, 10] = (b[0, 10] + 1) % 100
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-5)
    assert not torch.allclose(la[0, 10:], lb[0, 10:], atol=1e-5)


def test_fused_matches_explicit():
    torch.manual_seed(0)
    model = Model(SMALL).eval()
    x = _tokens()
    model.use_fused = True
    fused, _ = model(x)
    model.use_fused = False
    explicit, _ = model(x)
    assert torch.allclose(fused, explicit, atol=1e-4)


def test_position_matters():
    # Rotary encoding must make the same token read differently at different places.
    torch.manual_seed(0)
    model = Model(SMALL).eval()
    x = torch.tensor([[5, 9, 5, 9, 5, 9]])
    logits, _ = model(x)
    assert not torch.allclose(logits[0, 1], logits[0, 3], atol=1e-5)


def test_ignore_index():
    torch.manual_seed(0)
    model = Model(SMALL).eval()
    x = _tokens(1)
    targets = torch.full_like(x, -1)
    targets[0, 5] = 42
    logits, loss = model(x, targets)
    expected = F.cross_entropy(logits[0, 5:6], torch.tensor([42]))
    assert torch.allclose(loss, expected, atol=1e-6)


def test_overfits_one_batch():
    torch.manual_seed(0)
    model = Model(SMALL)
    x = _tokens(4)
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_too_long_raises():
    with pytest.raises(ValueError):
        Model(SMALL)(_tokens(1, 17))

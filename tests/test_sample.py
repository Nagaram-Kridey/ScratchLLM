import pytest
import torch

from scratchllm.config import ModelConfig
from scratchllm.model import Model
from scratchllm.sample import generate

CFG = ModelConfig(2, 32, 2, 50, 16)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Model(CFG).eval()


class Favours(torch.nn.Module):
    """Stand-in model whose logits always favour one token."""

    def __init__(self, token):
        super().__init__()
        self.cfg, self.token = CFG, token
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, tokens, targets=None):
        logits = torch.zeros(*tokens.shape, CFG.vocab_size)
        logits[..., self.token] = 10.0
        return logits, None


def test_same_seed_same_output(model):
    a = generate(model, [1, 2, 3], 30, seed=5)
    assert len(a) == 30
    assert a == generate(model, [1, 2, 3], 30, seed=5)
    assert a != generate(model, [1, 2, 3], 30, seed=6)


def test_temperature_zero_ignores_seed(model):
    assert generate(model, [1, 2, 3], 10, temperature=0, seed=1) == generate(
        model, [1, 2, 3], 10, temperature=0, seed=2
    )


def test_tiny_top_p_equals_greedy(model):
    assert generate(model, [1, 2, 3], 10, top_p=1e-9, seed=3) == generate(model, [1, 2, 3], 10, temperature=0)


def test_stop_id_is_not_returned():
    assert generate(Favours(7), [1], 20, temperature=0, stop_id=7) == []
    assert generate(Favours(7), [1], 4, temperature=0) == [7, 7, 7, 7]


def test_prompt_longer_than_context(model):
    prompt = [i % 50 for i in range(CFG.context + 50)]
    assert len(generate(model, prompt, 5)) == 5


def test_empty_prompt_raises(model):
    with pytest.raises(ValueError):
        generate(model, [], 5)


def test_training_mode_is_restored():
    torch.manual_seed(0)
    model = Model(CFG).train()
    generate(model, [1], 2)
    assert model.training

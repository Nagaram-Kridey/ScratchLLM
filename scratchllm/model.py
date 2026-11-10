"""Decoder-only transformer: RMSNorm, rotary positions, SwiGLU, shared embedding."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotary_tables(head_dim: int, context: int, base: float = 10000.0):
    """cos and sin of position * frequency, shape [context, head_dim / 2]."""
    freqs = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(context).float(), freqs)
    return angles.cos(), angles.sin()


def apply_rotary(x, cos, sin):
    """Rotate each (even, odd) pair of x [B, heads, T, head_dim] by its position's angle."""
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads or (cfg.d_model // cfg.n_heads) % 2:
            raise ValueError("d_model / n_heads must be a whole, even number")
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, fused: bool):
        B, T, D = x.shape
        q, k, v = (
            t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
            for t in self.qkv(x).chunk(3, dim=-1)
        )
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        if fused:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            y = scores.masked_fill(future, float("-inf")).softmax(-1) @ v
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3)
        self.w_gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_out = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_out(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x, cos, sin, fused: bool):
        x = x + self.attn(self.norm1(x), cos, sin, fused)
        return x + self.ff(self.norm2(x))


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.use_fused = True  # False = explicit attention maths
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        cos, sin = rotary_tables(cfg.d_model // cfg.n_heads, cfg.context)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        T = tokens.size(1)
        if T > self.cfg.context:
            raise ValueError(f"sequence of {T} tokens is longer than the context of {self.cfg.context}")
        x = self.embed(tokens)
        cos, sin = self.cos[:T], self.sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin, self.use_fused)
        logits = F.linear(self.norm(x), self.embed.weight)  # output shares the embedding
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=-1)
        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

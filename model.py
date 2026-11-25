"""A small, fast GPT: RoPE + RMSNorm + flash attention (SDPA) + tied embeddings."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 18
    block_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 160


def rope_cache(seq_len: int, head_dim: int) -> torch.Tensor:
    """(seq_len, head_dim/2, 2) table of cos/sin rotation angles."""
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), freqs)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Rotate channel pairs (i, i + D/2) of x (B, H, T, D) by position-dependent angles."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = rope[: x.shape[-2], :, 0], rope[: x.shape[-2], :, 1]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, rope, return_attn=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, rope), apply_rope(k, rope)
        attn = None
        if return_attn:  # slow explicit path, only used for visualisation
            w = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            attn = w.masked_fill(causal, float("-inf")).softmax(-1)
            y = attn @ v
        else:  # flash attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C)), attn


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
        )

    def forward(self, x, rope, return_attn=False):
        a, attn = self.attn(self.norm1(x), rope, return_attn)
        x = x + a
        return x + self.mlp(self.norm2(x)), attn


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # weight tying
        self.register_buffer("rope", rope_cache(cfg.block_size, cfg.n_embd // cfg.n_head), persistent=False)
        self.apply(self._init)
        for b in self.blocks:  # residual branches start as identity
            nn.init.zeros_(b.attn.proj.weight)
            nn.init.zeros_(b.mlp[2].weight)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, return_internals=False):
        """Returns (logits, loss). With return_internals=True returns
        (logits, loss, attn[list of (B,H,T,T)], hidden[list of (B,T,C)]) for visualisation."""
        x = self.wte(idx)
        attns, hiddens = [], []
        for block in self.blocks:
            x, attn = block(x, self.rope, return_internals)
            if return_internals:
                attns.append(attn)
                hiddens.append(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=-100)
        return (logits, loss, attns, hiddens) if return_internals else (logits, loss)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, stop_token: int | None = None):
        """Greedy decoding for a batch of equal-length prompts."""
        done = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_token is not None:
                done |= nxt.squeeze(1) == stop_token
                if done.all():
                    break
        return idx

"""General-purpose causal NeoGPT vs. standard RoPE GPT for multi-step expressions.

ZERO task-specific arithmetic length formulas: every positional/structural cue is
100% strictly causal (depends only on tokens <= t) and applies to any nested
expression, code, or structured reasoning trace.

Two matched architectures sharing the exact same RMSNorm + QK-RMSNorm + SwiGLU +
tied-embedding + depth-scaled-residual backbone:
  * `arch="neo"` (General Causal NeoGPT):
      1. Pure NoPE Q/K (zero periodic sinusoidal RoPE).
      2. Clipped local token bias (`min(i - j, 4)`) + ALiBi-spaced monotonic
         logarithmic token distance decay (`-softplus(alpha) * log(1 + (i - j))`).
      3. Strictly causal parenthesis-depth relative bias: running depth
         `d(t)` where `(` enters the inner scope and `)` closes it after `)`,
         resetting to 0 on `\\n` / `<` (clamped >= 0). Relative bucket
         `clamp(d_j - d_i, -2, 2) + 2` + monotonic `-softplus(beta) * |d_i - d_j|`.
      4. Token-Order Forgetting Gate (FoX, Lin et al. ICLR 2025): learned
         per-token non-negative forget rate `g_{k,h} = softplus(W_g h_k + b_g)`
         accumulated causally `-(G_i - G_j)` so the model can reset/forget
         obsolete CoT lines at `\\n` boundaries.
      5. Bias-Aware SSMax: shift-invariant effective bias fan-in
         `log N_eff(i, h) = logsumexp(B_i) - max(B_i)` scaling Q by
         `max(0.25, 1 + s_h * log N_eff)` before SDPA.
  * `arch="rope"` (Standard 1D RoPE GPT baseline):
      Full 1D RoPE (`base=10000`, optional train-time `rope_scale`) with the exact
      same backbone and parameter budget.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from expr_data import BOS_ID, EXPR_CHARS, LP_ID, NL_ID, RP_ID


@dataclass
class NeoExprConfig:
    vocab_size: int = len(EXPR_CHARS)
    arch: str = "neo"          # "neo" | "rope"
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 128
    local_clip: int = 4        # local token bias buckets: 0, 1, 2, 3, >=4
    use_depth: bool = True     # causal parenthesis-depth bias (neo only)
    use_fox: bool = True       # Token-FoX data-dependent forget gate (neo only)
    use_ssmax: bool = True     # Bias-Aware SSMax query scaling (neo only)
    rope_base: float = 10000.0


def causal_paren_depth(idx: torch.Tensor) -> torch.Tensor:
    """Strictly causal parenthesis nesting depth reset at each line boundary (`\\n` or `<`).

    `(` increments depth at `(` itself; `)` decrements depth AFTER `)` so `(` and `)`
    share the inner scope's depth. Clamped >= 0 so malformed greedy text stays safe.
    """
    B, T = idx.shape
    dev = idx.device
    is_open = (idx == LP_ID).long()
    is_close = (idx == RP_ID).long()
    # Close takes effect on the NEXT token
    close_prev = F.pad(is_close[:, :-1], (1, 0), value=0)
    delta = is_open - close_prev
    # Reset running sum at each newline or BOS so every CoT line starts cleanly at depth 0
    is_reset = (idx == NL_ID) | (idx == BOS_ID)
    # Segment-wise cumsum via cummax reset trick (exact, O(T), MPS-native!)
    raw = delta.cumsum(dim=1)
    # At reset positions, subtract the running sum *including* any close_prev at reset
    reset_val = torch.where(is_reset, raw, torch.zeros_like(raw))
    # Carry forward the most recent reset_val using index cummax
    pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
    last_reset_pos = torch.where(is_reset, pos, torch.zeros_like(pos)).cummax(dim=1).values
    base = raw.gather(1, last_reset_pos)
    has_reset = is_reset.cumsum(dim=1) > 0
    depth = torch.where(has_reset, raw - base, raw)
    return depth.clamp(min=0)


def apply_rope_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        hidden = ((int(8 * n_embd / 3) + 31) // 32) * 32
        self.w_gate_up = nn.Linear(n_embd, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_out(F.silu(gate) * up)


class ExprAttention(nn.Module):
    def __init__(self, cfg: NeoExprConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.arch = cfg.arch
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        if cfg.arch == "neo":
            self.local_bias = nn.Parameter(torch.zeros(cfg.n_head, cfg.local_clip + 1))
            init_rates = torch.logspace(math.log10(0.05), math.log10(1.5), cfg.n_head)
            self.dist_log_slope = nn.Parameter(torch.log(torch.expm1(init_rates)))

            if cfg.use_depth:
                self.depth_rel_bias = nn.Parameter(torch.zeros(cfg.n_head, 5))
                self.depth_log_slope = nn.Parameter(torch.full((cfg.n_head,), -1.5))
            else:
                self.depth_rel_bias = self.depth_log_slope = None

            self.fox_proj = nn.Linear(cfg.n_embd, cfg.n_head, bias=True) if cfg.use_fox else None
            self.ssmax_scale = nn.Parameter(torch.zeros(cfg.n_head)) if cfg.use_ssmax else None
        else:
            self.local_bias = self.dist_log_slope = None
            self.depth_rel_bias = self.depth_log_slope = None
            self.fox_proj = self.ssmax_scale = None

    def forward(self, x: torch.Tensor, geom: tuple, kv_cache: dict | None = None):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        q, k, v = self.qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.arch == "rope":
            cos, sin = geom
            q = apply_rope_1d(q, cos, sin)
            k = apply_rope_1d(k, cos, sin)
            if kv_cache is not None:
                if "k" in kv_cache:
                    k = torch.cat([kv_cache["k"], k], dim=2)
                    v = torch.cat([kv_cache["v"], v], dim=2)
                kv_cache["k"], kv_cache["v"] = k, v
                y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.proj(y.transpose(1, 2).reshape(B, T, C))

        # General Causal NeoGPT path
        local_oh, log_dist, depth_info, causal_mask = geom
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v

        l_bias = (local_oh @ self.local_bias.T).permute(2, 0, 1).unsqueeze(0)  # (1, H, T_q, T_k)
        slope = -F.softplus(self.dist_log_slope).view(1, H, 1, 1)
        bias_finite = l_bias + slope * log_dist

        if self.depth_rel_bias is not None:
            # Exact low-rank factorization over K = max_depth + 1 (<=6) discrete depths:
            # builds tiny (H, K, K) matrix M[h, u, v] instead of giant (B, T, T, 5) one-hot!
            d_q_oh, d_k_oh, d_bucket, d_abs = depth_info  # (B, T_q, K), (B, T_k, K), (K, K), (K, K)
            K = d_q_oh.shape[-1]
            d_slope = -F.softplus(self.depth_log_slope).view(H, 1, 1)
            M = self.depth_rel_bias[:, d_bucket] + d_slope * d_abs.unsqueeze(0)  # (H, K, K)
            # (B, T_q, K) @ (K, H*K) -> (B, H, T_q, K) @ (B, 1, K, T_k) -> (B, H, T_q, T_k)
            M_flat = M.permute(1, 0, 2).reshape(K, H * K)
            U = (d_q_oh @ M_flat).view(B, T, H, K).permute(0, 2, 1, 3)
            bias_finite = bias_finite + (U @ d_k_oh.unsqueeze(1).transpose(-2, -1))

        if self.fox_proj is not None:
            # Token-FoX: cumulative non-negative forget rate G_i - G_j along causal token order
            g = F.softplus(self.fox_proj(x)).transpose(1, 2)  # (B, H, T_q) >= 0
            if kv_cache is not None and "G" in kv_cache:
                G_prev = kv_cache["G"]
                G_new = G_prev[:, :, -1:] + g.cumsum(dim=-1)
                G_all = torch.cat([G_prev, G_new], dim=-1)
                kv_cache["G"] = G_all
                G_q, G_k = G_new, G_all
            else:
                G_all = g.cumsum(dim=-1)
                if kv_cache is not None:
                    kv_cache["G"] = G_all
                G_q, G_k = G_all, G_all
            fox_decay = (G_q[:, :, :, None] - G_k[:, :, None, :]).clamp(min=0.0)
            bias_finite = bias_finite - fox_decay

        bias = bias_finite.masked_fill(causal_mask, float("-inf")) if causal_mask is not None else bias_finite

        if self.ssmax_scale is not None:
            log_neff = (
                torch.logsumexp(bias, dim=-1, keepdim=True) - bias.amax(dim=-1, keepdim=True)
            ).clamp(min=0.0).detach()
            tau = (1.0 + self.ssmax_scale.view(1, H, 1, 1) * log_neff).clamp(min=0.25)
            q = q * tau

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class ExprBlock(nn.Module):
    def __init__(self, cfg: NeoExprConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.n_embd)
        self.attn = ExprAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg.n_embd)

    def forward(self, x: torch.Tensor, geom: tuple, kv_cache: dict | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), geom, kv_cache=kv_cache)
        return x + self.mlp(self.norm2(x))


class NeoExprGPT(nn.Module):
    def __init__(self, cfg: NeoExprConfig):
        super().__init__()
        assert cfg.arch in ("neo", "rope")
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(ExprBlock(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight

        if cfg.arch == "rope":
            hd = cfg.n_embd // cfg.n_head
            inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, hd, 2).float() / hd))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            self.inv_freq = None

        self.apply(self._init)
        res_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for b in self.blocks:
            nn.init.normal_(b.attn.proj.weight, std=res_std)
            nn.init.normal_(b.mlp.w_out.weight, std=res_std)
            if b.attn.fox_proj is not None:
                nn.init.normal_(b.attn.fox_proj.weight, std=0.01)
                nn.init.constant_(b.attn.fox_proj.bias, -3.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _geometry(
        self,
        idx: torch.Tensor,
        rope_scale: torch.Tensor | None = None,
        pos_offset: int = 0,
        depth_all: torch.Tensor | None = None,
    ) -> tuple:
        B, T_q = idx.shape
        T_k = pos_offset + T_q
        dev = idx.device
        ii = torch.arange(pos_offset, T_k, device=dev)
        jj = torch.arange(T_k, device=dev)

        if self.cfg.arch == "rope":
            pos = ii.float().unsqueeze(0).expand(B, T_q)
            if rope_scale is not None:
                pos = pos * rope_scale
            ang = pos[:, None, :, None] * self.inv_freq
            return ang.cos(), ang.sin()

        diff = (ii[:, None] - jj[None, :]).clamp(min=0)
        local_oh = F.one_hot(diff.clamp(max=self.cfg.local_clip), self.cfg.local_clip + 1).float()
        log_dist = torch.log1p(diff.float())[None, None, :, :]
        causal_mask = (jj[None, :] > ii[:, None]) if T_q > 1 else None

        if self.cfg.use_depth:
            d_k = causal_paren_depth(idx) if depth_all is None else depth_all
            d_q = d_k[:, pos_offset:]
            K = int(d_k.max().item()) + 1
            d_k_oh = F.one_hot(d_k, K).float()
            d_q_oh = d_k_oh[:, pos_offset:] if depth_all is None else F.one_hot(d_q, K).float()
            u = torch.arange(K, device=dev)
            d_diff_kk = u[None, :] - u[:, None]  # v - u (key_depth - query_depth)
            d_bucket = d_diff_kk.clamp(-2, 2) + 2
            d_abs = d_diff_kk.abs().float()
            depth_info = (d_q_oh, d_k_oh, d_bucket, d_abs)
        else:
            depth_info = None
        return local_oh, log_dist, depth_info, causal_mask

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        rope_scale: torch.Tensor | None = None,
    ):
        x = self.wte(idx)
        geom = self._geometry(idx, rope_scale=rope_scale)
        for blk in self.blocks:
            x = blk(x, geom)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def _decode_step(
        self,
        idx_step: torch.Tensor,
        caches: list[dict],
        pos_offset: int,
        depth_all: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.wte(idx_step)
        geom = self._geometry(idx_step, pos_offset=pos_offset, depth_all=depth_all)
        for blk, cache in zip(self.blocks, caches):
            x = blk(x, geom, kv_cache=cache)
        return self.head(self.norm(x[:, -1]))

    @torch.no_grad()
    def generate_batch(
        self,
        prompts_ids: list[list[int]],
        max_new_tokens: int,
        stop_token: int,
        pad_token: int,
    ) -> list[list[int]]:
        """Length-bucketed KV-cached greedy decoding (exact parity with uncached forward)."""
        dev = self.wte.weight.device
        groups: dict[int, list[int]] = {}
        for i, p in enumerate(prompts_ids):
            groups.setdefault(len(p), []).append(i)

        results: list[list[int]] = [[] for _ in prompts_ids]
        for L, idxs in groups.items():
            for off in range(0, len(idxs), 64):
                chunk = idxs[off : off + 64]
                cur = torch.tensor([prompts_ids[i] for i in chunk], dtype=torch.long, device=dev)
                done = torch.zeros(len(chunk), dtype=torch.bool, device=dev)
                caches: list[dict] = [{} for _ in self.blocks]

                # Prefill full prompt once
                depth_all = causal_paren_depth(cur) if (self.cfg.arch == "neo" and self.cfg.use_depth) else None
                logits_last = self._decode_step(cur, caches, pos_offset=0, depth_all=depth_all)
                nxt = logits_last.argmax(-1)
                cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
                done |= nxt == stop_token

                for step_i in range(1, max_new_tokens):
                    if done.all():
                        break
                    pos_off = L + step_i - 1
                    tok_in = nxt.unsqueeze(1)
                    if depth_all is not None:
                        # Update running parenthesis depth in O(1) per token (exact match to causal_paren_depth)
                        prev_tok = cur[:, -2]
                        prev_d = depth_all[:, -1]
                        is_reset = (nxt == NL_ID) | (nxt == BOS_ID)
                        d_new = torch.where(
                            is_reset,
                            torch.zeros_like(prev_d),
                            (prev_d + (nxt == LP_ID).long() - (prev_tok == RP_ID).long()).clamp(min=0),
                        )
                        depth_all = torch.cat([depth_all, d_new.unsqueeze(1)], dim=1)
                    logits_last = self._decode_step(tok_in, caches, pos_offset=pos_off, depth_all=depth_all)
                    nxt = logits_last.argmax(-1)
                    nxt = torch.where(done, torch.full_like(nxt, pad_token), nxt)
                    cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
                    done |= nxt == stop_token

                for i, seq in zip(chunk, cur.tolist()):
                    results[i] = seq
        return results

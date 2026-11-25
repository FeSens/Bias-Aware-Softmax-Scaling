"""NeoGPT: a simple, strictly length-invariant Transformer for algorithmic generalization.

Default configuration is NeoGPT v2 (alias-free):
  1. Pure NoPE Q/K (zero sinusoidal RoPE, zero learned positional lookup table).
  2. 6-bucket monotonic relative-place bias {none, <=-2, -1, 0, +1, >=+2} plus a non-positive
     logarithmic distance decay `-softplus(alpha) * log(1 + |Delta pid|)` with ALiBi-spaced
     initial slopes across heads.
  3. Clipped local token bias (`min(i - j, 2)` -> {self, prev, >=2 back}).
  4. QK-RMSNorm, SwiGLU MLP, depth-scaled residual init (`0.02 / sqrt(2 * n_layer)`),
     tied input/output embeddings, and no fixed `block_size` limit.

Optional v1 / v1.1 ablation mode (`rope_frac > 0`, e.g. `0.5` with `n_place_buckets=4` and
train-time `place_rope_scale` passed to `forward`): restores decoupled Partial Place-RoPE so
the effect of random Place-RoPE scaling can be measured 1-factor against v1 and v2.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS = "0123456789 =+-*/.\n"
EQ_ID = CHARS.index("=")
PAD_ID = CHARS.index("\n")


@dataclass
class NeoGPTConfig:
    vocab_size: int = len(CHARS)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 160
    local_clip: int = 2               # local token distance buckets: 0 (self), 1 (prev), >=2 (far)
    n_place_buckets: int = 6          # 6 = {none, <=-2, -1, 0, +1, >=+2} (v2); 4 = {none, <0, 0, >0} (v1)
    rope_frac: float = 0.0            # 0.0 = pure NoPE (v2 default); 0.5 = decoupled Place-RoPE (v1 / v1.1)
    place_w_max: float = 2.5          # fastest Place-RoPE rad/place when rope_frac > 0
    place_w_min: float = 0.05         # slowest Place-RoPE rad/place when rope_frac > 0
    uniform_slope: float | None = None  # None = ALiBi geometric slopes [0.15..2.5] (v2); float = uniform raw init (v1)
    bias_ssmax: bool = False          # Bias-Aware SSMax: scale Q by (1 + s_h * log N_eff(bias))
    vanilla_ssmax: bool = False       # Vanilla SSMax control: scale Q by (1 + s_h * log(i + 1))
    barrier_fox: bool = False         # Place-FoX: data-dependent non-negative barrier decay along place order
    barrier_gate: str = "softplus"    # "softplus" | "relu" (dead-zone gate allowing exact 0.0 on propagate cols)


def place_ids(idx: torch.Tensor) -> torch.Tensor:
    """Causal digit place IDs computed from tokens alone (zero answer leakage).

    Operand digit at 10^p -> p + 1 (1 = units).
    Answer token that emits answer digit j (0 = space after '=') -> max(0, n + 1 - j),
    where n = max(digits(a), digits(b)). All non-digit / post-answer tokens -> 0.
    """
    eqc = (idx == EQ_ID).long().cumsum(1)
    opd = ((idx < 10) & (eqc == 0)).flip(1)
    r = opd.long().cumsum(1)
    run = ((r - torch.where(opd, 0, r).cummax(1).values) * opd).flip(1)
    n = run.max(1, keepdim=True).values
    j = eqc.cumsum(1) - 2
    return run + (n + 1 - j).clamp(min=0) * (j >= 0)


def apply_partial_place_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor, d_rope: int
) -> torch.Tensor:
    """Leave first (D - d_rope) channels unrotated (NoPE); rotate & mask last d_rope channels."""
    x_nope, x_rope = x[..., :-d_rope], x[..., -d_rope:]
    r1, r2 = x_rope.chunk(2, dim=-1)
    rot = torch.cat([r1 * cos - r2 * sin, r1 * sin + r2 * cos], dim=-1) * mask
    return torch.cat([x_nope, rot], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        hidden = ((int(8 * n_embd / 3) + 31) // 32) * 32
        self.w_gate_up = nn.Linear(n_embd, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_out(F.silu(gate) * up)


class NeoAttention(nn.Module):
    def __init__(self, cfg: NeoGPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_place_buckets in (4, 6)
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.d_rope = int(round(self.head_dim * cfg.rope_frac)) // 2 * 2

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        self.local_bias = nn.Parameter(torch.zeros(cfg.n_head, cfg.local_clip + 1))
        self.place_rel_bias = nn.Parameter(torch.zeros(cfg.n_head, cfg.n_place_buckets))
        if cfg.uniform_slope is not None:
            init_raw = torch.full((cfg.n_head,), cfg.uniform_slope)
        else:
            init_rates = torch.logspace(math.log10(0.15), math.log10(2.5), cfg.n_head)
            init_raw = torch.log(torch.expm1(init_rates))  # inverse softplus
        self.place_log_slope = nn.Parameter(init_raw)
        assert not (cfg.bias_ssmax and cfg.vanilla_ssmax), "Choose either bias_ssmax or vanilla_ssmax"
        assert cfg.barrier_gate in ("softplus", "relu")
        self.ssmax_mode = "bias" if cfg.bias_ssmax else ("vanilla" if cfg.vanilla_ssmax else "none")
        self.ssmax_scale = nn.Parameter(torch.zeros(cfg.n_head)) if self.ssmax_mode != "none" else None
        self.barrier_gate = cfg.barrier_gate
        self.barrier_proj = nn.Linear(cfg.n_embd, cfg.n_head, bias=True) if cfg.barrier_fox else None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        pid_mask: torch.Tensor | None,
        rel_oh: torch.Tensor,
        log_dist: torch.Tensor,
        local_oh: torch.Tensor,
        causal_mask: torch.Tensor,
        barrier_geom: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        return_attn: bool = False,
        partition_stretch: torch.Tensor | None = None,
        real_pairs: torch.Tensor | None = None,
    ):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        q, k, v = self.qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.d_rope > 0:
            q = apply_partial_place_rope(q, cos, sin, pid_mask, self.d_rope)
            k = apply_partial_place_rope(k, cos, sin, pid_mask, self.d_rope)

        p_bias = (rel_oh @ self.place_rel_bias.T).permute(0, 3, 1, 2)
        slope = -F.softplus(self.place_log_slope).view(1, H, 1, 1)
        l_bias = (local_oh @ self.local_bias.T).permute(2, 0, 1).unsqueeze(0)
        bias_finite = p_bias + slope * log_dist.unsqueeze(1) + l_bias
        if self.barrier_proj is not None:
            pid_oh, idx_hi, idx_lo, between_mask = barrier_geom
            raw_g = self.barrier_proj(x)
            g = (F.relu(raw_g) if self.barrier_gate == "relu" else F.softplus(raw_g)).transpose(1, 2)
            S = (pid_oh[:, None, :, :] * g[:, :, :, None]).cumsum(dim=2)   # causal sum per place: (B, H, T, P)
            C_pref = S.cumsum(dim=3)                                        # prefix sum across places: (B, H, T, P)
            barrier = (
                C_pref.gather(3, idx_hi.expand(B, H, T, T))
                - C_pref.gather(3, idx_lo.expand(B, H, T, T))
            ).clamp(min=0.0) * between_mask
            bias_finite = bias_finite - barrier
        bias = bias_finite.masked_fill(causal_mask, float("-inf"))

        if self.ssmax_scale is not None:
            if self.ssmax_mode == "bias":
                log_scale = (torch.logsumexp(bias, dim=-1, keepdim=True) - bias.amax(dim=-1, keepdim=True)).clamp(min=0.0).detach()
            else:
                log_scale = torch.log(torch.arange(1, T + 1, device=x.device, dtype=q.dtype)).view(1, 1, T, 1)
            tau = (1.0 + self.ssmax_scale.view(1, H, 1, 1) * log_scale).clamp(min=0.25)
            q = q * tau

        score_l2 = None
        need_explicit = return_attn or (partition_stretch is not None) or (real_pairs is not None)
        if need_explicit:
            scores_finite = (q @ k.transpose(-2, -1)) * (D ** -0.5) + bias_finite
            if real_pairs is not None:
                denom = (real_pairs.sum() * H).clamp(min=1)
                score_l2 = scores_finite.square().masked_fill(~real_pairs[:, None], 0.0).sum() / denom
            scores = scores_finite.masked_fill(causal_mask, float("-inf"))
            if partition_stretch is not None:
                valid_f = (~causal_mask).to(scores_finite.dtype)
                n_keys = torch.arange(1, T + 1, device=x.device, dtype=scores_finite.dtype).view(1, 1, T, 1)
                z_bar = ((scores_finite * valid_f).sum(dim=-1, keepdim=True) / n_keys).detach()
                n_prior = torch.arange(T, device=x.device, dtype=scores_finite.dtype).view(1, 1, T, 1)
                beta = (partition_stretch - 1.0).clamp(min=0.0) * n_prior
                z_virt = torch.where(beta > 0, z_bar + beta.clamp(min=1e-30).log(), float("-inf"))
                attn_w = torch.cat([scores, z_virt], dim=-1).softmax(dim=-1)[..., :-1]
            else:
                attn_w = scores.softmax(dim=-1)
            y = attn_w @ v
            attn = attn_w if return_attn else None
        else:
            attn = None
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(y.transpose(1, 2).reshape(B, T, C)), attn, score_l2


class NeoBlock(nn.Module):
    def __init__(self, cfg: NeoGPTConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.n_embd)
        self.attn = NeoAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg.n_embd)

    def forward(
        self,
        x,
        cos,
        sin,
        pid_mask,
        rel_oh,
        log_dist,
        local_oh,
        causal_mask,
        barrier_geom=None,
        return_attn=False,
        partition_stretch=None,
        real_pairs=None,
    ):
        a, attn, score_l2 = self.attn(
            self.norm1(x), cos, sin, pid_mask, rel_oh, log_dist, local_oh, causal_mask,
            barrier_geom=barrier_geom, return_attn=return_attn,
            partition_stretch=partition_stretch, real_pairs=real_pairs,
        )
        x = x + a
        return x + self.mlp(self.norm2(x)), attn, score_l2


class NeoGPT(nn.Module):
    def __init__(self, cfg: NeoGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(NeoBlock(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # tied embeddings

        d_rope = int(round((cfg.n_embd // cfg.n_head) * cfg.rope_frac)) // 2 * 2
        if d_rope > 0:
            n_freq = d_rope // 2
            freqs = cfg.place_w_max * (cfg.place_w_min / cfg.place_w_max) ** torch.linspace(0, 1, n_freq)
            self.register_buffer("place_freqs", freqs, persistent=False)
        else:
            self.place_freqs = None

        self.apply(self._init)
        res_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for b in self.blocks:
            nn.init.normal_(b.attn.proj.weight, std=res_std)
            nn.init.normal_(b.mlp.w_out.weight, std=res_std)
            if b.attn.barrier_proj is not None:
                if cfg.barrier_gate == "relu":
                    nn.init.normal_(b.attn.barrier_proj.weight, std=0.02)
                    nn.init.constant_(b.attn.barrier_proj.bias, 0.05)
                else:
                    nn.init.normal_(b.attn.barrier_proj.weight, std=0.01)
                    nn.init.constant_(b.attn.barrier_proj.bias, -3.0)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _geometry(
        self,
        idx: torch.Tensor,
        pid_override: torch.Tensor | None = None,
        place_rope_scale: torch.Tensor | None = None,
    ):
        B, T = idx.shape
        dev = idx.device
        pid = place_ids(idx) if pid_override is None else pid_override
        has_pid = pid > 0

        if self.place_freqs is not None:
            pid_mask = has_pid.float()[:, None, :, None]
            pid_f = pid.float() if place_rope_scale is None else pid.float() * place_rope_scale
            angles = pid_f[:, None, :, None] * self.place_freqs
            cos, sin = angles.cos(), angles.sin()
        else:
            cos = sin = pid_mask = None

        both = has_pid[:, :, None] & has_pid[:, None, :]
        diff = pid[:, None, :] - pid[:, :, None]
        if self.cfg.n_place_buckets == 6:
            rel_idx = torch.where(both, diff.clamp(-2, 2) + 3, 0)
        else:
            rel_idx = torch.where(both, diff.clamp(-1, 1) + 2, 0)
        rel_oh = F.one_hot(rel_idx, self.cfg.n_place_buckets).float()
        log_dist = torch.log1p(diff.abs().float()) * both.float()

        ii = torch.arange(T, device=dev)
        local_idx = (ii[:, None] - ii[None, :]).clamp(0, self.cfg.local_clip)
        local_oh = F.one_hot(local_idx, self.cfg.local_clip + 1).float()
        causal_mask = ii[None, :] > ii[:, None]
        barrier_geom = None
        if self.cfg.barrier_fox:
            P = int(pid.max().item()) + 1
            pid_oh = F.one_hot(pid, P).float()
            pid_oh[..., 0] = 0.0
            lo = torch.minimum(pid[:, :, None], pid[:, None, :])
            hi = torch.maximum(pid[:, :, None], pid[:, None, :])
            between_mask = (both & ((hi - lo) >= 2)).unsqueeze(1).float()
            idx_hi = (hi - 1).clamp(min=0).unsqueeze(1)
            idx_lo = lo.unsqueeze(1)
            barrier_geom = (pid_oh, idx_hi, idx_lo, between_mask)
        return cos, sin, pid_mask, rel_oh, log_dist, local_oh, causal_mask, barrier_geom

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_internals: bool = False,
        pid_override: torch.Tensor | None = None,
        place_rope_scale: torch.Tensor | None = None,
        partition_stretch: torch.Tensor | None = None,
        attn_decay: float = 0.0,
    ):
        """Returns (logits, loss), or (logits, loss, attns, hiddens) when return_internals=True."""
        x = self.wte(idx)
        geom = self._geometry(idx, pid_override=pid_override, place_rope_scale=place_rope_scale)
        real_pairs = None
        if attn_decay > 0.0:
            real = idx != PAD_ID
            real[:, 0] = True
            real_pairs = real[:, :, None] & real[:, None, :] & (~geom[6])
        attns, hiddens, score_l2s = [], [], []
        for block in self.blocks:
            x, attn, sl2 = block(
                x, *geom,
                return_attn=return_internals,
                partition_stretch=partition_stretch,
                real_pairs=real_pairs,
            )
            if return_internals:
                attns.append(attn)
                hiddens.append(x)
            if sl2 is not None:
                score_l2s.append(sl2)
        logits = self.head(self.norm(x))
        loss = None
        self.last_score_l2 = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=-100)
            if score_l2s:
                sc = sum(score_l2s) / len(score_l2s)
                loss = loss + attn_decay * sc
                self.last_score_l2 = sc.detach()
        return (logits, loss, attns, hiddens) if return_internals else (logits, loss)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, stop_token: int | None = None):
        """Greedy decoding for a batch of equal-length prompts (no block_size limit)."""
        done = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_token is not None:
                done |= nxt.squeeze(1) == stop_token
                if done.all():
                    break
        return idx

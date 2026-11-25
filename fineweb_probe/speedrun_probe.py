"""Faithful modded-nanogpt ("gpt-speedrun") small probe on official FineWeb-10B GPT-2 tokens.

Tests whether our two general-purpose innovations translate to real language-modeling
speedrun gains (FineWeb val loss AND wall-clock ms/step) on top of a modern
modded-nanogpt stack:
  * 8L 6H d=384 (head_dim=64, ~33.5M params, tied embeddings, bf16)
  * Pre-Norm RMSNorm + QK-RMSNorm + RoPE + ReLU^2 MLP + 30*tanh logit soft-capping
  * Muon (5 Newton-Schulz steps) on 2D hidden block weights + AdamW on embeddings/scalars
  * FlexAttention with exact <|endoftext|> document masking + alternating short/long
    sliding windows + dynamic attention window warmup (128->512 short, 256->2048 long)

5 controlled 1-factor arms (`--arm`):
  1. base            : modded-nanogpt small probe baseline
  2. vssmax          : + Vanilla SSMax (1 + s_h * log(pos_in_seq + 1))
  3. bssmax          : + Window/Doc-Aware 1D Bias-Aware SSMax (1 + s_h * log(min(pos_in_doc, W_layer)))
                         [exact O(T) analytic N_vis, ZERO FlexAttention score_mod overhead!]
  4. relu_fox        : + Dead-Zone ReLU Token-FoX in FlexAttention score_mod
                         [g = ReLU(W_g h + b_g), G = cumsum(g), score - (G[q] - G[kv])]
  5. bssmax_relu_fox : + Both Window/Doc-Aware Bias-Aware SSMax and Dead-Zone ReLU Token-FoX
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EOT_ID = 50256
VOCAB_SIZE = 50304  # padded multiple of 64 (exact modded-nanogpt)


def load_fineweb_shard(path: Path) -> torch.Tensor:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(1024), dtype=np.int32)
        assert int(header[0]) == 20240520, f"Bad magic in {path}: {header[0]}"
        ntok = int(header[2])
        toks = np.frombuffer(f.read(), dtype=np.uint16)
        assert len(toks) == ntok, f"Token count mismatch in {path}: {len(toks)} vs {ntok}"
    return torch.from_numpy(toks.copy())  # uint16 on CPU


def zeropower_ns5(G: torch.Tensor) -> torch.Tensor:
    """5-step Newton-Schulz orthogonalization in bfloat16 (exact Muon)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(5):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Nesterov-momentum orthogonalized Muon optimizer for 2D hidden matrices."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, wd: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, wd=wd))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, mom, wd = g["lr"], g["momentum"], g["wd"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(grad)
                buf = st["buf"]
                buf.lerp_(grad, 1.0 - mom)
                upd = grad.lerp(buf, mom)
                upd = zeropower_ns5(upd)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(upd.to(p.dtype), alpha=-lr * scale)


@dataclass
class ProbeConfig:
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 384
    arm: str = "base"  # base | vssmax | bssmax | relu_fox | bssmax_relu_fox
    rope_base: float = 10000.0


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class ProbeAttention(nn.Module):
    def __init__(self, cfg: ProbeConfig, is_long: bool):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.is_long = is_long
        self.arm = cfg.arm
        self.use_ssmax = cfg.arm in ("vssmax", "bssmax", "bssmax_relu_fox")
        self.use_fox = cfg.arm in ("relu_fox", "bssmax_relu_fox")

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.ssmax_scale = nn.Parameter(torch.zeros(cfg.n_head)) if self.use_ssmax else None
        self.fox_proj = nn.Linear(cfg.n_embd, cfg.n_head, bias=True) if self.use_fox else None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_mask,
        log_n_scale: torch.Tensor | None,
        flex_fn,
    ) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        q, k, v = self.qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)

        # QK-RMSNorm (exact modded-nanogpt)
        q = F.rms_norm(q, (D,))
        k = F.rms_norm(k, (D,))
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Zero-overhead 1D SSMax query scaling BEFORE FlexAttention
        if self.ssmax_scale is not None:
            tau = (1.0 + self.ssmax_scale.view(1, H, 1, 1) * log_n_scale).clamp(min=0.25)
            q = (q * tau).to(v.dtype)

        if flex_fn is not None:
            flex_plain, flex_fox = flex_fn
            if self.fox_proj is not None:
                # Dead-zone ReLU barrier gate; cumsum in fp32 so doc-offset cancels cleanly in G[q]-G[kv]
                g = F.relu(self.fox_proj(x)).transpose(1, 2).float()  # (B, H, T) >= 0
                G_q = g.cumsum(dim=-1).contiguous()
                G_k = G_q.clone()
                y = flex_fox(q, k, v, block_mask, G_q, G_k)
            else:
                y = flex_plain(q, k, v, block_mask)
        else:
            # CPU / fallback path for local unit testing without CUDA FlexAttention
            mask_bias = block_mask  # additive (B, 1, T, T) float mask
            if self.fox_proj is not None:
                g = F.relu(self.fox_proj(x)).transpose(1, 2).float()
                G = g.cumsum(dim=-1)
                fox_decay = (G[:, :, :, None] - G[:, :, None, :]).clamp(min=0.0)
                mask_bias = mask_bias - fox_decay.to(q.dtype)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_bias)

        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class ProbeMLP(nn.Module):
    """ReLU^2 MLP (exact modded-nanogpt)."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.fc(x)).square())


class ProbeBlock(nn.Module):
    def __init__(self, cfg: ProbeConfig, is_long: bool):
        super().__init__()
        self.attn = ProbeAttention(cfg, is_long=is_long)
        self.mlp = ProbeMLP(cfg.n_embd)
        self.is_long = is_long

    def forward(self, x, cos, sin, block_mask, log_n_scale, flex_fn):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)), cos, sin, block_mask, log_n_scale, flex_fn)
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


class SpeedrunProbeGPT(nn.Module):
    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Alternating short / long sliding-window layers, last layer always long (exact modded-nanogpt pattern)
        self.blocks = nn.ModuleList(
            ProbeBlock(cfg, is_long=(i % 2 == 1 or i == cfg.n_layer - 1)) for i in range(cfg.n_layer)
        )
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight

        hd = cfg.n_embd // cfg.n_head
        inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, hd, 2).float() / hd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.apply(self._init)
        for b in self.blocks:
            nn.init.zeros_(b.attn.proj.weight)
            nn.init.zeros_(b.mlp.proj.weight)
            if b.attn.fox_proj is not None:
                nn.init.normal_(b.attn.fox_proj.weight, std=0.02)
                nn.init.constant_(b.attn.fox_proj.bias, 0.05)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None,
        short_mask,
        long_mask,
        short_w: int,
        long_w: int,
        flex_fn,
    ):
        B, T = idx.shape
        dev = idx.device
        pos = torch.arange(T, device=dev)
        ang = pos.float()[:, None] * self.inv_freq[None, :]
        cos = ang.cos()[None, None, :, :].to(self.wte.weight.dtype)
        sin = ang.sin()[None, None, :, :].to(self.wte.weight.dtype)

        # Precompute 1D O(T) SSMax log-scales (ZERO O(T^2) cost!)
        log_short = log_long = None
        if self.cfg.arm == "vssmax":
            log_seq = torch.log(pos.float() + 1.0).view(1, 1, T, 1)
            log_short = log_long = log_seq
        elif self.cfg.arm in ("bssmax", "bssmax_relu_fox"):
            # Exact visible key count N_vis(i, layer) = min(pos_in_doc(i) + 1, W_layer)
            is_bos = idx == EOT_ID
            is_bos[:, 0] = True
            p2d = pos.unsqueeze(0).expand(B, T)
            doc_start = torch.where(is_bos, p2d, torch.zeros_like(p2d)).cummax(dim=1).values
            pos_in_doc = (p2d - doc_start + 1).float()  # (B, T)
            log_short = torch.log(pos_in_doc.clamp(max=float(short_w))).view(B, 1, T, 1)
            log_long = torch.log(pos_in_doc.clamp(max=float(long_w))).view(B, 1, T, 1)

        x = F.rms_norm(self.wte(idx), (self.cfg.n_embd,))
        for blk in self.blocks:
            mask = long_mask if blk.is_long else short_mask
            lscale = log_long if blk.is_long else log_short
            x = blk(x, cos, sin, mask, lscale, flex_fn)

        x = F.rms_norm(x, (self.cfg.n_embd,))
        logits = self.head(x).float()
        # 30 * tanh logit soft-capping (exact modded-nanogpt)
        logits = 30.0 * torch.tanh(logits / 30.0)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def window_schedule(step: int, total_steps: int, seq_len: int) -> tuple[int, int]:
    """Dynamic attention window warmup quantized to 128-token blocks (modded-nanogpt style)."""
    frac = step / max(1, total_steps)
    if frac < 0.25:
        short_w, long_w = 128, 256
    elif frac < 0.50:
        short_w, long_w = 256, 512
    elif frac < 0.75:
        short_w, long_w = 256, 1024
    else:
        short_w, long_w = 512, min(2048, seq_len)
    return min(short_w, seq_len), min(long_w, seq_len)


def build_masks(idx: torch.Tensor, short_w: int, long_w: int, use_flex: bool):
    """Build exact document-boundary + sliding-window causal masks (FlexAttention or dense fallback)."""
    B, T = idx.shape
    dev = idx.device
    is_bos = idx == EOT_ID
    is_bos[:, 0] = True
    docs = is_bos.cumsum(dim=1)  # (B, T)

    if use_flex:
        from torch.nn.attention.flex_attention import create_block_mask

        def make_bm(w: int):
            def mask_mod(b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                doc_ok = docs[b, q_idx] == docs[b, kv_idx]
                win_ok = (q_idx - kv_idx) < w
                return causal & doc_ok & win_ok

            return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=dev)

        return make_bm(short_w), make_bm(long_w)

    # Dense fallback for CPU testing
    ii = torch.arange(T, device=dev)
    diff = ii[:, None] - ii[None, :]
    causal = diff >= 0
    doc_ok = docs[:, :, None] == docs[:, None, :]
    short_ok = causal.unsqueeze(0) & doc_ok & (diff < short_w).unsqueeze(0)
    long_ok = causal.unsqueeze(0) & doc_ok & (diff < long_w).unsqueeze(0)
    neg = torch.tensor(float("-inf"), device=dev)
    return (
        torch.where(short_ok, 0.0, neg).unsqueeze(1),
        torch.where(long_ok, 0.0, neg).unsqueeze(1),
    )


@torch.no_grad()
def evaluate_val(
    model: SpeedrunProbeGPT,
    val_toks: torch.Tensor,
    dev: torch.device,
    seq_len: int,
    batch_size: int,
    val_batches: int,
    short_w: int,
    long_w: int,
    use_flex: bool,
    flex_fn,
    amp_ctx,
) -> float:
    model.eval()
    total_loss = 0.0
    stride = batch_size * seq_len
    for bi in range(val_batches):
        off = bi * stride
        chunk = val_toks[off : off + stride + 1].to(torch.long)
        x = chunk[:-1].view(batch_size, seq_len).to(dev, non_blocking=True)
        y = chunk[1:].view(batch_size, seq_len).to(dev, non_blocking=True)
        sm, lm = build_masks(x, short_w, long_w, use_flex)
        with amp_ctx:
            _, loss = model(x, y, sm, lm, short_w, long_w, flex_fn)
        total_loss += loss.item()
    model.train()
    return total_loss / val_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="base", choices=["base", "vssmax", "bssmax", "relu_fox", "bssmax_relu_fox"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro_batch", type=int, default=4, help="micro-batch per forward (grad_accum = batch // micro_batch)")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--val_batches", type=int, default=64, help="64 * 4 * 2048 = 524,288 val tokens")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--adam_lr", type=float, default=0.008)
    ap.add_argument("--scalar_lr", type=float, default=0.004)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = dev.type == "cuda"
    use_flex = use_cuda and not args.no_compile
    flex_fn = None
    if use_flex:
        import torch._dynamo as _dynamo
        from torch.nn.attention.flex_attention import flex_attention

        _dynamo.config.cache_size_limit = 64
        if hasattr(_dynamo.config, "recompile_limit"):
            _dynamo.config.recompile_limit = 64

        @torch.compile(dynamic=False)
        def flex_plain(q, k, v, block_mask):
            return flex_attention(q, k, v, block_mask=block_mask)

        @torch.compile(dynamic=False)
        def flex_fox(q, k, v, block_mask, G_q, G_k):
            def score_mod(score, b, h, q_idx, kv_idx):
                return score - (G_q[b, h, q_idx] - G_k[b, h, kv_idx]).to(score.dtype)

            return flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)

        flex_fn = (flex_plain, flex_fox)

    data_dir = Path(args.data_dir)
    train_toks = load_fineweb_shard(data_dir / "fineweb_train_000001.bin")
    val_toks = load_fineweb_shard(data_dir / "fineweb_val_000000.bin")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    cfg = ProbeConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, arm=args.arm)
    model = SpeedrunProbeGPT(cfg).to(dev)
    if use_cuda:
        model = model.to(torch.bfloat16)

    # Exact modded-nanogpt optimizer split: Muon on 2D block weights, AdamW on embeddings & scalars
    muon_params, emb_params, scalar_params = [], [], []
    for name, p in model.named_parameters():
        if name == "wte.weight":
            emb_params.append(p)
        elif p.ndim == 2 and "blocks." in name and "fox_proj" not in name:
            muon_params.append(p)
        else:
            scalar_params.append(p)

    opt_muon = Muon(muon_params, lr=args.muon_lr, momentum=0.95)
    opt_adam = torch.optim.AdamW(
        [
            {"params": emb_params, "lr": args.adam_lr, "weight_decay": 0.0},
            {"params": scalar_params, "lr": args.scalar_lr, "weight_decay": 0.0},
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
    )

    # Trapezoidal warmup (10%) + steady + linear cooldown (35%) schedule (exact modded-nanogpt!)
    warmup = max(1, int(0.05 * args.steps))
    cooldown = max(1, int(0.35 * args.steps))

    def lr_mult(s: int) -> float:
        if s < warmup:
            return (s + 1) / warmup
        if s < args.steps - cooldown:
            return 1.0
        return max(0.05, (args.steps - s) / cooldown)

    from contextlib import nullcontext

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()

    assert args.batch % args.micro_batch == 0
    grad_accum = args.batch // args.micro_batch
    mb = args.micro_batch
    mb_stride = mb * args.seq_len
    stride = args.batch * args.seq_len
    need_toks = (args.steps + 5) * stride + 1
    max_start = max(1, len(train_toks) - need_toks)
    rng_off = np.random.default_rng(1000 + args.seed)
    base_off = int(rng_off.integers(0, max_start))

    # Warm up Triton FlexAttention compilation (2 micro-steps, untimed!)
    print(f"START arm={args.arm} s{args.seed} | {model.num_params():,} params | dev={dev} | flex={use_flex} | mb={mb}x{grad_accum}", flush=True)
    for _ in range(2 if use_flex else 0):
        sw, lw = window_schedule(0, args.steps, args.seq_len)
        chunk = train_toks[base_off : base_off + mb_stride + 1].to(torch.long)
        x = chunk[:-1].view(mb, args.seq_len).to(dev)
        y = chunk[1:].view(mb, args.seq_len).to(dev)
        sm, lm = build_masks(x, sw, lw, use_flex)
        with amp_ctx:
            _, loss = model(x, y, sm, lm, sw, lw, flex_fn)
        loss.backward()
        model.zero_grad(set_to_none=True)
    if use_cuda:
        torch.cuda.synchronize()

    # Timed training loop (tracks exact train_time_s excluding val time, exact modded-nanogpt style!)
    train_time_s = 0.0
    curve = []
    for step in range(1, args.steps + 1):
        mult = lr_mult(step - 1)
        for g in opt_muon.param_groups:
            g["lr"] = args.muon_lr * mult
        opt_adam.param_groups[0]["lr"] = args.adam_lr * mult
        opt_adam.param_groups[1]["lr"] = args.scalar_lr * mult
        # Momentum warmup for Muon (0.85 -> 0.95 over first 150 steps)
        frac_m = min(step / 150.0, 1.0)
        for g in opt_muon.param_groups:
            g["momentum"] = (1 - frac_m) * 0.85 + frac_m * 0.95

        sw, lw = window_schedule(step - 1, args.steps, args.seq_len)
        step_off = base_off + (step - 1) * stride

        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        step_loss = 0.0
        for acc_i in range(grad_accum):
            off = step_off + acc_i * mb_stride
            chunk = train_toks[off : off + mb_stride + 1].to(torch.long)
            x = chunk[:-1].view(mb, args.seq_len).to(dev, non_blocking=True)
            y = chunk[1:].view(mb, args.seq_len).to(dev, non_blocking=True)
            sm, lm = build_masks(x, sw, lw, use_flex)
            with amp_ctx:
                _, loss = model(x, y, sm, lm, sw, lw, flex_fn)
                loss_scaled = loss / grad_accum
            loss_scaled.backward()
            step_loss += loss.item() / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_muon.step()
        opt_adam.step()

        if use_cuda:
            torch.cuda.synchronize()
        train_time_s += time.perf_counter() - t0

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate_val(
                model, val_toks, dev, args.seq_len, mb, args.val_batches,
                short_w=min(512, args.seq_len), long_w=args.seq_len,
                use_flex=use_flex, flex_fn=flex_fn, amp_ctx=amp_ctx,
            )
            ms_step = 1000.0 * train_time_s / step
            snap = {
                "step": step,
                "train_time_s": round(train_time_s, 2),
                "ms_per_step": round(ms_step, 2),
                "train_loss": round(step_loss, 4),
                "val_loss": round(val_loss, 4),
            }
            curve.append(snap)
            print(
                f"arm={args.arm:15s} s{args.seed} step {step:4d} | time {train_time_s:6.1f}s ({ms_step:5.1f}ms/st) "
                f"| tr {step_loss:.4f} | val {val_loss:.4f}",
                flush=True,
            )

    # Also evaluate 2x OOD context length (seq_len * 2 = 4096) on FineWeb val!
    ood_seq = args.seq_len * 2
    ood_mb = max(1, mb // 2)
    ood_val = evaluate_val(
        model, val_toks, dev, ood_seq, ood_mb, max(1, args.val_batches // 2),
        short_w=min(512, ood_seq), long_w=ood_seq,
        use_flex=use_flex, flex_fn=flex_fn, amp_ctx=amp_ctx,
    )

    # Collect learned SSMax / ReLU-FoX telemetry
    telemetry = {}
    if model.blocks[0].attn.ssmax_scale is not None:
        telemetry["ssmax_by_layer"] = [
            [round(v, 3) for v in b.attn.ssmax_scale.detach().float().cpu().tolist()]
            for b in model.blocks
        ]
    if model.blocks[0].attn.fox_proj is not None:
        # Measure actual ReLU sparsity (% exact zero!) on one val micro-batch
        chunk = val_toks[: mb_stride + 1].to(torch.long)
        x = chunk[:-1].view(mb, args.seq_len).to(dev)
        with torch.no_grad(), amp_ctx:
            h = F.rms_norm(model.wte(x), (cfg.n_embd,))
            zero_fracs, pos_means = [], []
            sm, lm = build_masks(x, min(512, args.seq_len), args.seq_len, use_flex)
            pos = torch.arange(args.seq_len, device=dev)
            ang = pos.float()[:, None] * model.inv_freq[None, :]
            cos = ang.cos()[None, None, :, :].to(h.dtype)
            sin = ang.sin()[None, None, :, :].to(h.dtype)
            log_s = log_l = None
            if cfg.arm == "bssmax_relu_fox":
                is_bos = x == EOT_ID; is_bos[:, 0] = True
                p2d = pos.unsqueeze(0).expand(mb, args.seq_len)
                ds = torch.where(is_bos, p2d, torch.zeros_like(p2d)).cummax(dim=1).values
                pid = (p2d - ds + 1).float()
                log_s = torch.log(pid.clamp(max=512.0)).view(mb, 1, args.seq_len, 1)
                log_l = torch.log(pid.clamp(max=float(args.seq_len))).view(mb, 1, args.seq_len, 1)
            for b in model.blocks:
                g = F.relu(b.attn.fox_proj(F.rms_norm(h, (cfg.n_embd,)))).float()
                zero_fracs.append(round(float((g == 0).float().mean().item()), 3))
                pos_means.append(round(float(g[g > 0].mean().item()) if (g > 0).any() else 0.0, 3))
                h = b(h, cos, sin, lm if b.is_long else sm, log_l if b.is_long else log_s, flex_fn)
        telemetry["fox_zero_frac"] = zero_fracs
        telemetry["fox_pos_mean"] = pos_means

    rec = {
        "arm": args.arm,
        "seed": args.seed,
        "params": model.num_params(),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        "steps": args.steps,
        "tokens_trained": args.steps * stride,
        "train_time_s": round(train_time_s, 2),
        "ms_per_step": round(1000.0 * train_time_s / args.steps, 2),
        "val_loss_2048": round(curve[-1]["val_loss"], 4),
        "val_loss_4096": round(ood_val, 4),
        "curve": curve,
        "telemetry": telemetry,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(rec, indent=1))
    print(f"RESULT_JSON {json.dumps(rec)}", flush=True)


if __name__ == "__main__":
    main()

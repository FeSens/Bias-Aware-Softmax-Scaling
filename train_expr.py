"""Train and evaluate General Causal NeoGPT vs. 1D RoPE GPT on multi-step exact-rational expressions.

    python train_expr.py --arch neo --name expr_neo_mps --seed 0
    python train_expr.py --arch rope --name expr_rope_mps --seed 0

Logs full TensorBoard dashboard (train loss, token acc, teacher-forced & greedy
exact-match across 4 splits: in_dist, ood_steps [5-7 ops], ood_depth [depth 3-4],
ood_scale [bigger nums], plus live text rollouts & NeoGPT FoX/SSMax/depth telemetry).
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from expr_data import (
    EOS_ID,
    ITOS,
    PAD_ID,
    build_eval_suite,
    encode_prompts,
    encode_sequences,
    extract_final_fraction,
    parse_expr_str,
    eval_tokens,
    sample_problem,
)
from neo_expr import NeoExprConfig, NeoExprGPT

EXPR_LAYOUT = {
    "Multi-Step Rational Expressions": {
        "greedy final answer accuracy": ["Multiline", ["eval_final/.*"]],
        "greedy exact trace accuracy": ["Multiline", ["eval_trace/.*"]],
        "teacher-forced token accuracy": ["Multiline", ["eval_tok/.*"]],
    },
    "NeoExpr Telemetry": {
        "Bias-Aware SSMax scale (mean)": ["Multiline", ["neo/ssmax_s_L.*"]],
        "Token-FoX bias (mean)": ["Multiline", ["neo/fox_bias_L.*"]],
    },
}


@torch.no_grad()
def eval_split_teacher(model: NeoExprGPT, items: list[tuple[str, list[str], str]], dev: torch.device, batch: int = 64):
    """Fast teacher-forced token accuracy and exact-sequence match."""
    tok_hit = tok_tot = seq_hit = 0
    for i in range(0, len(items), batch):
        seqs = [it[2] for it in items[i : i + batch]]
        x, y = encode_sequences(seqs, dev)
        logits, _ = model(x)
        preds = logits.argmax(-1)
        ok = y >= 0
        hit = (preds == y) & ok
        tok_hit += int(hit.sum().item())
        tok_tot += int(ok.sum().item())
        seq_hit += int((hit | ~ok).all(dim=1).sum().item())
    return {
        "tok_acc": tok_hit / max(1, tok_tot),
        "tf_exact": seq_hit / max(1, len(items)),
    }


@torch.no_grad()
def eval_split_greedy(
    model: NeoExprGPT,
    items: list[tuple[str, list[str], str]],
    max_new_tokens: int = 360,
):
    """True autoregressive greedy rollout: checks exact trace match AND exact Fraction final value."""
    prompts = [it[0] for it in items]
    p_ids = encode_prompts(prompts)
    outs = model.generate_batch(p_ids, max_new_tokens=max_new_tokens, stop_token=EOS_ID, pad_token=PAD_ID)

    trace_ok = final_ok = valid_parse = 0
    samples: list[str] = []
    for i, (seq_ids, (prompt, target_lines, full_target)) in enumerate(zip(outs, items)):
        txt = "".join(ITOS[t] for t in seq_ids)
        if ">" in txt:
            txt_cut = txt.split(">", 1)[0] + ">"
        else:
            txt_cut = txt
        if txt_cut == full_target:
            trace_ok += 1
        pred_frac = extract_final_fraction(txt)
        true_frac = eval_tokens(parse_expr_str(prompt))
        if pred_frac is not None:
            valid_parse += 1
            if pred_frac == true_frac:
                final_ok += 1
        if i < 3:
            mark = "OK" if pred_frac == true_frac else "ERR"
            samples.append(f"[{mark}] true={target_lines[-1]} pred={pred_frac}\nTARGET:\n{full_target}\nPRED:\n{txt_cut}")
    n = max(1, len(items))
    return {
        "trace_exact": trace_ok / n,
        "final_exact": final_ok / n,
        "parse_valid": valid_parse / n,
        "samples": samples,
    }


@torch.no_grad()
def log_telemetry(writer: SummaryWriter, model: NeoExprGPT, step: int):
    if model.cfg.arch != "neo":
        return
    for li, blk in enumerate(model.blocks):
        if blk.attn.ssmax_scale is not None:
            sc = blk.attn.ssmax_scale.detach().cpu()
            writer.add_scalar(f"neo/ssmax_s_L{li + 1}_mean", sc.mean().item(), step)
        if blk.attn.fox_proj is not None:
            fb = torch.nn.functional.softplus(blk.attn.fox_proj.bias.detach().cpu())
            fw = blk.attn.fox_proj.weight.detach().cpu().norm(dim=-1)
            writer.add_scalar(f"neo/fox_bias_L{li + 1}_mean", fb.mean().item(), step)
            writer.add_scalar(f"neo/fox_wnorm_L{li + 1}_mean", fw.mean().item(), step)
        if blk.attn.depth_log_slope is not None:
            ds = torch.nn.functional.softplus(blk.attn.depth_log_slope.detach().cpu())
            writer.add_scalar(f"neo/depth_slope_L{li + 1}_mean", ds.mean().item(), step)


def link_tb_views(tb_dir: Path):
    for view_name in ("expr_bench", "b1_fixes"):
        view_dir = ROOT / "research" / "tb_views" / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        link = view_dir / tb_dir.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(tb_dir.resolve())


class ExprStream:
    """Online curriculum stream with length-sorted micro-batching (eliminates padding waste)."""

    def __init__(self, seed: int = 0, pool_mult: int = 8):
        self.rng = random.Random(seed)
        self.pool_mult = pool_mult
        self.p = [0.25, 0.25, 0.25, 0.25]
        self.ema_acc = [0.0, 0.0, 0.0, 0.0]
        self._queue: list[tuple[list[str], list[int]]] = []

    def _refill(self, batch_size: int):
        items: list[tuple[str, int]] = []
        for _ in range(batch_size * self.pool_mult):
            b = self.rng.choices([0, 1, 2, 3], weights=self.p, k=1)[0]
            n_ops = b + 1
            max_d = min(2, n_ops)
            _, _, full = sample_problem(
                self.rng,
                n_ops_range=(n_ops, n_ops),
                max_depth=max_d,
                add_max=200,
                mul_max=25,
                max_num=9999,
                max_den=99,
                max_seq_len=240,
                holdout_filter="train",
            )
            items.append((full, b))
        items.sort(key=lambda it: len(it[0]))
        batches = [
            ([s for s, _ in items[i : i + batch_size]], [bk for _, bk in items[i : i + batch_size]])
            for i in range(0, len(items), batch_size)
        ]
        self.rng.shuffle(batches)
        self._queue = batches

    def sample_batch(self, batch_size: int) -> tuple[list[str], list[int]]:
        if not self._queue:
            self._refill(batch_size)
        return self._queue.pop()

    def update_curriculum(self, buckets: list[int], exact: list[float]):
        for b in range(4):
            vals = [e for bk, e in zip(buckets, exact) if bk == b]
            if vals:
                self.ema_acc[b] = 0.9 * self.ema_acc[b] + 0.1 * (sum(vals) / len(vals))
        w = [max(0.08, 1.0 - a) for a in self.ema_acc]
        s = sum(w)
        self.p = [v / s for v in w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="expr_neo_mps")
    ap.add_argument("--arch", default="neo", choices=["neo", "rope"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--bias_lr_mult", type=float, default=5.0)
    ap.add_argument("--no_depth", action="store_true")
    ap.add_argument("--no_fox", action="store_true")
    ap.add_argument("--no_ssmax", action="store_true")
    ap.add_argument("--rope_scale", type=float, nargs=2, default=None)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_n", type=int, default=64, help="problems per eval split")
    ap.add_argument("--no_record", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    eval_suite = build_eval_suite(seed=4242, per_split=args.eval_n)
    stream = ExprStream(seed=1000 + args.seed)

    torch.manual_seed(args.seed)
    cfg = NeoExprConfig(
        arch=args.arch,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        use_depth=not args.no_depth,
        use_fox=not args.no_fox,
        use_ssmax=not args.no_ssmax,
    )
    model = NeoExprGPT(cfg).to(dev)
    ema = copy.deepcopy(model).requires_grad_(False)

    bias_suffixes = ("local_bias", "dist_log_slope", "depth_rel_bias", "depth_log_slope")
    bias_params = [p for n, p in model.named_parameters() if n.endswith(bias_suffixes)]
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and not n.endswith(bias_suffixes)]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 and not n.endswith(bias_suffixes)]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.wd, "lr_mult": 1.0},
            {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0},
            {"params": bias_params, "weight_decay": 0.0, "lr_mult": args.bias_lr_mult},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    warmup = min(300, max(1, args.steps // 10))
    lr_at = lambda s: args.lr * (
        s / warmup
        if s < warmup
        else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, args.steps - warmup)))
    )
    params = list(model.parameters())

    ts = time.strftime("%m%d_%H%M%S")
    tb_dir = Path("/tmp/expr_smoke") if args.no_record else (ROOT / "runs" / "expr" / f"{ts}_{args.name}_s{args.seed}")
    writer = SummaryWriter(tb_dir, flush_secs=10)
    writer.add_custom_scalars(EXPR_LAYOUT)
    writer.add_text("config", f"`{vars(args)}`\n\n`{vars(cfg)}`\n\n{model.num_params():,} params on `{dev}`")
    if not args.no_record:
        link_tb_views(tb_dir)

    print(f"START {args.name} s{args.seed} | arch={args.arch} | {model.num_params():,} params | dev={dev} | tb={tb_dir}", flush=True)

    t0, t_log, curve = time.time(), time.time(), []
    for step in range(1, args.steps + 1):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_mult"]

        seqs, buckets = stream.sample_batch(args.batch)
        x, y = encode_sequences(seqs, dev)
        rs = None
        if args.arch == "rope" and args.rope_scale is not None:
            lo, hi = args.rope_scale
            rs = lo + (hi - lo) * torch.rand(x.shape[0], 1, device=dev)

        opt.zero_grad(set_to_none=True)
        logits, loss = model(x, y, rope_scale=rs)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), params):
                pe.lerp_(p, 1 - d)

        if step % 20 == 0:
            ok = y >= 0
            hit = (logits.argmax(-1) == y) & ok
            seq_ex = (hit | ~ok).all(dim=1).float().cpu().tolist()
            stream.update_curriculum(buckets, seq_ex)
            dt, t_log = time.time() - t_log, time.time()
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("accuracy/token_train", (hit.sum() / ok.sum().clamp(min=1)).item(), step)
            writer.add_scalar("accuracy/seq_train", sum(seq_ex) / len(seq_ex), step)
            writer.add_scalar("optim/lr", lr, step)
            writer.add_scalar("optim/grad_norm", gnorm.item(), step)
            writer.add_scalar("perf/steps_per_sec", 20 / max(dt, 1e-6), step)

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            log_telemetry(writer, model, step)
            snap = {"step": step, "loss": round(loss.item(), 4)}
            parts = []
            for split_name, items in eval_suite.items():
                tf = eval_split_teacher(model, items, dev)
                gr = eval_split_greedy(model, items, max_new_tokens=360 if split_name != "in_dist" else 260)
                writer.add_scalar(f"eval_tok/{split_name}", tf["tok_acc"], step)
                writer.add_scalar(f"eval_tf/{split_name}", tf["tf_exact"], step)
                writer.add_scalar(f"eval_trace/{split_name}", gr["trace_exact"], step)
                writer.add_scalar(f"eval_final/{split_name}", gr["final_exact"], step)
                writer.add_scalar(f"eval_valid/{split_name}", gr["parse_valid"], step)
                if split_name in ("in_dist", "ood_steps"):
                    writer.add_text(f"samples/{split_name}", "\n\n---\n\n".join(gr["samples"]), step)
                snap[split_name] = {
                    "tok": round(tf["tok_acc"], 4),
                    "tf_exact": round(tf["tf_exact"], 4),
                    "trace": round(gr["trace_exact"], 4),
                    "final": round(gr["final_exact"], 4),
                }
                parts.append(f"{split_name}:{100*gr['final_exact']:4.1f}%(tr {100*gr['trace_exact']:4.1f}%)")
            curve.append(snap)
            print(f"{args.name} s{args.seed} step {step:4d} | loss {loss.item():.4f} | " + " | ".join(parts), flush=True)
            model.train()
            t_log = time.time()

    # Final EMA evaluation
    ema.eval()
    final_ema = {}
    for split_name, items in eval_suite.items():
        tf = eval_split_teacher(ema, items, dev)
        gr = eval_split_greedy(ema, items, max_new_tokens=360 if split_name != "in_dist" else 260)
        final_ema[split_name] = {
            "tok": round(tf["tok_acc"], 4),
            "tf_exact": round(tf["tf_exact"], 4),
            "trace": round(gr["trace_exact"], 4),
            "final": round(gr["final_exact"], 4),
        }
    writer.close()

    rec = {
        "name": args.name,
        "seed": args.seed,
        "steps": args.steps,
        "minutes": round((time.time() - t0) / 60, 2),
        "task": "rational_expr_cot",
        "cfg": vars(cfg),
        "args": {k: v for k, v in vars(args).items() if k not in ("name", "seed", "steps")},
        "final": curve[-1],
        "final_ema": final_ema,
        "curve": curve,
    }
    if not args.no_record:
        out = ROOT / "research" / "runs" / f"{args.name}_s{args.seed}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(rec, indent=1))
        torch.save(model.state_dict(), out / "model.pt")
        with open(ROOT / "research" / "expr_results.jsonl", "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    ema_parts = [f"{k}:{100*v['final']:4.1f}%" for k, v in final_ema.items()]
    print(f"DONE {args.name} s{args.seed} ({rec['minutes']}m) | EMA final: " + " | ".join(ema_parts), flush=True)


if __name__ == "__main__":
    main()

"""Train and evaluate NeoGPT under the exact B1 / OOD-lab protocol with full TensorBoard tracking.

    python train_neo.py --seed 0

Protocol (matches research/ood_lab.py B1 runs apples-to-apples):
  * Trained on 1, 2, 3, and 5-digit addition (`--skip_digits 4`), 5,000 steps, batch 256,
    AdamW (lr=1e-3, wd=0.1, betas=(0.9, 0.95), 200-step warmup + cosine decay to 0.05x, clip 1.0).
  * NO train-time RoPE stretch, NO random place offset, NO attention delay, NO score decay,
    NO answer-padding hack — pure NeoGPT architecture.
  * Identical train/held-out hash split (seed=0) and fixed exam (seed=4242, 1024/bucket).
  * Full TensorBoard dashboard: all train.py scalars, error maps (1-6d & 1-8d), curriculum map,
    attention maps, in-dist & OOD logit lens, token embeddings, greedy samples, weight histograms,
    extended length accuracy (2d..10d), gradient-agreement probes, and NeoGPT bias telemetry.
  * Saves checkpoints + metrics to `research/runs/<name>_s<seed>/`, appends to `research/results.jsonl`,
    and symlinks TensorBoard dirs into `research/tb_views/neo_gpt/` (and `research/tb_views/b1_fixes/`).
"""
import argparse
import copy
import fcntl
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import viz
from neo_gpt import NeoGPT, NeoGPTConfig
from train import CHARS, D_MAX, LAYOUT, AdditionStream, encode, evaluate, log_eval

STEPS, BATCH, TRAIN_D = 5000, 256, 5

NEO_LAYOUT = {
    **LAYOUT,
    "NeoGPT extrapolation": {
        "exact match by equal operand length (2d-10d)": ["Multiline", ["length/exact_.*"]],
        "learned place log-distance slopes": ["Multiline", ["neo/place_slope_L.*"]],
    },
}


@torch.no_grad()
def by_length(model: NeoGPT, max_n: int = 10, N: int = 512) -> list[float]:
    """Exact match (%) with both operands n digits, n = 2..max_n, on N fixed random problems per length.
    Uses the exact same generator seeds (n * 11) as research/ood_lab.py for n=2..8.
    Encodes on CPU to avoid MPS int64 edge cases above 2^24."""
    dev = model.wte.weight.device
    out = []
    for n in range(2, max_n + 1):
        g = torch.Generator().manual_seed(n * 11)
        a = torch.randint(10 ** (n - 1), 10 ** n, (8 * N,), generator=g)[:N]
        b = torch.randint(10 ** (n - 1), 10 ** n, (8 * N,), generator=g)[:N]
        x, y, _, _ = encode(a, b, max(8, n))
        lg = model(x.to(dev))[0].cpu()
        acc = ((lg.argmax(-1) == y) | (y < 0)).all(1).float().mean().item()
        out.append(round(100 * acc, 1))
    return out


@torch.no_grad()
def far_error_map(model: NeoGPT, step: int, n_max: int = 8, per_band: int = 24):
    """Error heatmap extended to 8-digit operands (bands 6-8 never trained). Own RNG: no side effects."""
    dev = model.wte.weight.device
    g = torch.Generator().manual_seed(1)
    v = torch.cat([
        torch.randint(0 if d == 1 else 10 ** (d - 1), 10 ** d, (per_band,), generator=g).sort().values
        for d in range(1, n_max + 1)
    ])
    a, b = v.repeat_interleave(len(v)), v.repeat(len(v))
    exact, p_wrong = [], []
    for i in range(0, len(a), 4096):
        x, y, _, _ = encode(a[i : i + 4096], b[i : i + 4096], n_max)
        logits = model(x.to(dev))[0].float().cpu()
        nll = -F.log_softmax(logits, -1).gather(2, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * (y >= 0)
        exact.append(((logits.argmax(-1) == y) | (y < 0)).all(1))
        p_wrong.append(1 - torch.exp(-nll.sum(1)))
    exact, p_wrong = torch.cat(exact).float(), torch.cat(p_wrong)
    band = torch.arange(len(v)) // per_band
    ood = (band.repeat_interleave(len(v)) >= TRAIN_D) | (band.repeat(len(v)) >= TRAIN_D)
    return viz.error_map(
        p_wrong.view(len(v), len(v)).numpy(), exact[~ood].mean().item(), exact[ood].mean().item(), step, n_max, TRAIN_D
    )


@torch.no_grad()
def score(model: NeoGPT, exam):
    a, b, bucket, ood = exam
    exact, _, hit, ok, _ = evaluate(model, a, b)
    ood, bucket, ex = ood.cpu(), bucket.cpu(), exact.float()
    tok = lambda m: ((hit & ok)[m].sum() / ok[m].sum()).item()
    return {
        "ood_exact": ex[ood].mean().item(),
        "ood_token": tok(ood),
        "in_exact": ex[~ood].mean().item(),
        "in_token": tok(~ood),
        "by_bucket": [round(ex[bucket == k].mean().item(), 4) for k in range(D_MAX ** 2)],
    }


@torch.no_grad()
def log_neo_telemetry(writer: SummaryWriter, model: NeoGPT, step: int):
    """Track NeoGPT's learned relative-place slopes and local/place bias magnitudes."""
    for li, blk in enumerate(model.blocks):
        slopes = F.softplus(blk.attn.place_log_slope).detach().cpu()
        writer.add_scalar(f"neo/place_slope_L{li + 1}_mean", slopes.mean().item(), step)
        writer.add_scalar(f"neo/place_slope_L{li + 1}_max", slopes.max().item(), step)
        lb = blk.attn.local_bias.detach().cpu()
        writer.add_scalar(f"neo/local_prev_vs_far_L{li + 1}", (lb[:, 1] - lb[:, 2]).abs().mean().item(), step)
        pb = blk.attn.place_rel_bias.detach().cpu()
        if pb.shape[1] == 6:
            writer.add_scalar(f"neo/place_same_vs_m1_L{li + 1}", (pb[:, 3] - pb[:, 2]).abs().mean().item(), step)
            writer.add_scalar(f"neo/place_m1_vs_farR_L{li + 1}", (pb[:, 2] - pb[:, 1]).abs().mean().item(), step)
        else:
            writer.add_scalar(f"neo/place_same_vs_right_L{li + 1}", (pb[:, 2] - pb[:, 1]).abs().mean().item(), step)
        if blk.attn.ssmax_scale is not None:
            sc = blk.attn.ssmax_scale.detach().cpu()
            writer.add_scalar(f"neo/ssmax_s_L{li + 1}_mean", sc.mean().item(), step)
            writer.add_scalar(f"neo/ssmax_s_L{li + 1}_max", sc.max().item(), step)
        if blk.attn.barrier_proj is not None:
            raw_b = blk.attn.barrier_proj.bias.detach().cpu()
            gb = F.relu(raw_b) if blk.attn.barrier_gate == "relu" else F.softplus(raw_b)
            gw = blk.attn.barrier_proj.weight.detach().cpu().norm(dim=-1)
            writer.add_scalar(f"neo/barrier_bias_L{li + 1}_mean", gb.mean().item(), step)
            writer.add_scalar(f"neo/barrier_wnorm_L{li + 1}_mean", gw.mean().item(), step)


def link_tb_views(tb_dir: Path, run_tag: str):
    """Symlink the TensorBoard run into both dedicated neo_gpt view and b1_fixes comparison view."""
    for view_name in ("neo_gpt", "b1_fixes"):
        view_dir = ROOT / "research" / "tb_views" / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        link = view_dir / tb_dir.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(tb_dir.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="neo_gpt_v2_mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--skip_digits", type=int, nargs="*", default=[4])
    ap.add_argument("--rope_frac", type=float, default=0.0, help="0.0 = v2 pure NoPE; 0.5 = v1/v1.1 decoupled Place-RoPE")
    ap.add_argument("--n_place_buckets", type=int, default=6, choices=[4, 6])
    ap.add_argument("--uniform_slope", type=float, default=None, help="None = ALiBi slopes (v2); float = uniform raw init (v1)")
    ap.add_argument("--bias_lr_mult", type=float, default=10.0)
    ap.add_argument("--place_rope_scale", type=float, nargs=2, default=None, help="lo hi: train-time random scale U(lo,hi) on Place-RoPE pids (v1.1)")
    ap.add_argument("--bias_ssmax", action="store_true", help="Bias-Aware SSMax: scale Q by (1 + s_h * log N_eff(bias))")
    ap.add_argument("--vanilla_ssmax", action="store_true", help="Vanilla SSMax control: scale Q by (1 + s_h * log(i + 1))")
    ap.add_argument("--barrier_fox", action="store_true", help="Place-FoX: data-dependent non-negative barrier decay along place order")
    ap.add_argument("--barrier_gate", default="softplus", choices=["softplus", "relu"], help="Barrier gate nonlinearity")
    ap.add_argument("--partition_stretch", type=float, nargs=2, default=None, help="lo hi: train-only random partition-function stretch gamma ~ U(lo,hi)")
    ap.add_argument("--attn_decay", type=float, default=0.0, help="L2 penalty on mean squared pre-softmax causal attention scores (ood_lab formulation)")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--no_record", action="store_true", help="skip writing results.jsonl (for smoke tests)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    stream = AdditionStream(dev, TRAIN_D, seed=0)  # identical train/held-out split to ood_lab.py
    torch.manual_seed(4242)
    exam = stream.exam(per_bucket=1024)            # identical exam to ood_lab.py

    torch.manual_seed(args.seed)
    cfg = NeoGPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_place_buckets=args.n_place_buckets,
        rope_frac=args.rope_frac,
        uniform_slope=args.uniform_slope,
        bias_ssmax=args.bias_ssmax,
        vanilla_ssmax=args.vanilla_ssmax,
        barrier_fox=args.barrier_fox,
        barrier_gate=args.barrier_gate,
    )
    model = NeoGPT(cfg).to(dev)
    ema = copy.deepcopy(model).requires_grad_(False)

    skip = torch.tensor(
        [(k // TRAIN_D + 1 in args.skip_digits) or (k % TRAIN_D + 1 in args.skip_digits) for k in range(TRAIN_D ** 2)],
        device=dev,
    )

    def mask_curriculum():
        if skip.any():
            stream.p = stream.p.masked_fill(skip, 0.0)
            stream.p = stream.p / stream.p.sum()

    mask_curriculum()

    # Weight decay on 2D matrices only; scalar place/local biases get bias_lr_mult (ssmax_scale stays at 1.0x lr, 0 wd)
    bias_suffixes = ("local_bias", "place_rel_bias", "place_log_slope")
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
        betas=(0.9, args.beta2),
    )
    warmup = min(200, max(1, args.steps // 10))
    lr_at = lambda s: args.lr * (
        s / warmup
        if s < warmup
        else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, args.steps - warmup)))
    )
    params = list(model.parameters())

    ts = time.strftime("%m%d_%H%M%S")
    tb_dir = Path("/tmp/neo_smoke") if args.no_record else (ROOT / "runs" / "lab" / f"{ts}_{args.name}_s{args.seed}")
    writer = SummaryWriter(tb_dir, flush_secs=10)
    writer.add_custom_scalars(NEO_LAYOUT)
    writer.add_text(
        "config",
        f"`{vars(args)}`\n\n`{vars(cfg)}`\n\n{model.num_params():,} parameters on `{dev}`",
    )
    if not args.no_record:
        link_tb_views(tb_dir, f"{args.name}_s{args.seed}")

    print(f"START {args.name} s{args.seed} | {model.num_params():,} params | dev={dev} | tb={tb_dir}", flush=True)
    log_eval(writer, model, stream, exam, 0)
    log_neo_telemetry(writer, model, 0)

    t0, t_log, curve = time.time(), time.time(), []
    for step in range(1, args.steps + 1):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_mult"]

        a, b, bucket = stream.sample(args.batch)
        x, y, _, _ = encode(a, b, TRAIN_D)

        opt.zero_grad(set_to_none=True)
        prs = None
        if args.place_rope_scale is not None:
            lo, hi = args.place_rope_scale
            prs = lo + (hi - lo) * torch.rand(x.shape[0], 1, device=dev)
        pstretch = None
        if args.partition_stretch is not None:
            lo, hi = args.partition_stretch
            pstretch = lo + (hi - lo) * torch.rand(x.shape[0], 1, 1, 1, device=dev)
        logits, loss = model(x, y, place_rope_scale=prs, partition_stretch=pstretch, attn_decay=args.attn_decay)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        with torch.no_grad():
            d = min(args.ema, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), params):
                pe.lerp_(p, 1 - d)

        if step % 10 == 0:
            exact = ((logits.argmax(-1) == y) | (y < 0)).all(1).float()
            stream.update_curriculum(bucket, exact)
            mask_curriculum()
            dt, t_log = time.time() - t_log, time.time()
            writer.add_scalar("loss/train", loss.item(), step)
            if model.last_score_l2 is not None:
                writer.add_scalar("loss/attn_score_l2", model.last_score_l2.item(), step)
            writer.add_scalar("accuracy/train", exact.mean().item(), step)
            writer.add_scalar("optim/lr", lr, step)
            writer.add_scalar("optim/grad_norm", gnorm.item(), step)
            writer.add_scalar("perf/problems_per_sec", 10 * args.batch / max(dt, 1e-6), step)

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            log_eval(writer, model, stream, exam, step)
            log_neo_telemetry(writer, model, step)
            bl = by_length(model, max_n=10)
            for n, v in zip(range(2, 11), bl):
                writer.add_scalar(f"length/exact_{n}_digits", v / 100, step)
            if step % (2 * args.eval_every) == 0 or step == args.steps:
                writer.add_figure("1_error_map/far_to_8_digits", far_error_map(model, step), step)
            model.train()
            t_log = time.time()

        if step % 1000 == 0 or step == args.steps:
            model.eval()
            s = score(model, exam)
            bl10 = by_length(model, max_n=10)
            model.train()
            curve.append({
                "step": step,
                "loss": round(loss.item(), 4),
                **{k: round(s[k], 4) for k in ("ood_exact", "ood_token", "in_exact")},
                "by_length_2to10": bl10,
            })
            print(
                f"{args.name} s{args.seed} step {step:4d} | loss {loss.item():.4f} | in {100*s['in_exact']:5.1f}% "
                f"| ood {100*s['ood_exact']:5.1f}% | 4d..10d {bl10[2:]}",
                flush=True,
            )

    model.eval()
    final_s = score(model, exam)
    ema_s = score(ema, exam)
    bl_all = by_length(model, max_n=10)
    bl_ema_all = by_length(ema, max_n=10)

    # High-res 1-8d error map + final summary text in TensorBoard
    writer.add_figure("1_error_map/final_high_res_to_8d", far_error_map(model, args.steps, n_max=8, per_band=32), args.steps)
    summary = (
        f"NeoGPT ({model.num_params():,} params) | held-out 1-5d (incl. unseen 4d): {100*final_s['in_exact']:.2f}% "
        f"| mixed 6d exam: {100*final_s['ood_exact']:.2f}% (EMA {100*ema_s['ood_exact']:.2f}%) "
        f"| equal-length 2d..10d: {bl_all}"
    )
    writer.add_text("final", summary, args.steps)
    writer.close()

    rec = {
        "name": args.name,
        "seed": args.seed,
        "steps": args.steps,
        "minutes": round((time.time() - t0) / 60, 2),
        "arch": "NeoGPT",
        "cfg": vars(cfg),
        "args": {k: v for k, v in vars(args).items() if k not in ("name", "seed", "steps")},
        "final": final_s,
        "final_ema": ema_s,
        "by_length": bl_all[:7],           # 2d..8d (exact ood_lab.py schema!)
        "by_length_ema": bl_ema_all[:7],
        "by_length_2to10": bl_all,         # extended through 10d!
        "by_length_ema_2to10": bl_ema_all,
        "curve": curve,
    }
    if not args.no_record:
        out = ROOT / "research" / "runs" / f"{args.name}_s{args.seed}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(rec, indent=1))
        torch.save(model.state_dict(), out / "model.pt")
        if args.steps == STEPS:
            with open(ROOT / "research" / "results.jsonl", "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    print(
        f"DONE {args.name} s{args.seed} | mixed 6d {100*final_s['ood_exact']:.1f}% (EMA {100*ema_s['ood_exact']:.1f}%) "
        f"| held-out {100*final_s['in_exact']:.1f}% | 2d..10d {bl_all} | {rec['minutes']} min",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Teach a tiny GPT 1-5 digit addition from an infinite, GPU-generated data stream,
then examine it on 6-digit sums it has never seen (out of distribution).

    python train.py                 # trains, launches TensorBoard, opens the browser

Data is never stored: every batch is sampled and tokenised directly on the GPU.
  * 10% of all (a, b) pairs are held out by a hash and NEVER trained on -> honest in-distribution metric.
  * Operands are sampled per digit-length bucket (so short numbers are not ~0% of the data),
    and an adaptive curriculum shifts the batch towards the buckets the model still gets wrong.
  * Loss is only applied to the answer tokens (the operands are unpredictable noise).
  * Any problem with an operand longer than --train_digits is out of distribution: eval only.
"""
import argparse
import math
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import viz
from model import GPT, GPTConfig

sys.stdout.reconfigure(line_buffering=True)  # live progress even when piped to a log

# ----------------------------------------------------------------------------- tokens
CHARS = "0123456789 =+-*/.\n"  # digit tokens have id == digit value; "\n" is BOS / EOS / pad
STOI = {c: i for i, c in enumerate(CHARS)}
EOS = STOI["\n"]
D_MAX = 6  # longest operand we ever evaluate (training uses --train_digits <= D_MAX)


def layout(D):
    """Slots for D-digit operands: \\n a*D _ + _ b*D _ = _ c*(D+1) \\n -> (n_slots, first answer slot)."""
    return 3 * D + 9, 2 * D + 7


def decode(ids) -> str:
    return "".join(CHARS[i] for i in ids)


def encode(a: torch.Tensor, b: torch.Tensor, D: int = D_MAX):
    """Vectorised "a + b = c" tokeniser, entirely on a's device.

    Lays every problem out in fixed slots (zero-padded digits), then squeezes out the
    leading zeros with a cumsum + scatter. Returns x, y (y = -100 outside the answer),
    plus pos/keep, which map slots -> sequence positions (for per-digit-place metrics).
    """
    B, dev = a.shape[0], a.device
    L, ans0 = layout(D)
    c = a + b
    col = lambda ch: torch.full((B, 1), STOI[ch], device=dev)
    one = torch.ones(B, 1, dtype=torch.bool, device=dev)
    digs = lambda n, k: torch.stack([(n // 10**i) % 10 for i in range(k - 1, -1, -1)], 1)
    sig = lambda n, k: torch.cat([torch.stack([n >= 10**i for i in range(k - 1, 0, -1)], 1), one], 1)

    slots = torch.cat([col("\n"), digs(a, D), col(" "), col("+"), col(" "), digs(b, D),
                       col(" "), col("="), col(" "), digs(c, D + 1), col("\n")], 1)
    keep = torch.cat([one, sig(a, D), one, one, one, sig(b, D), one, one, one, sig(c, D + 1), one], 1)
    pos = torch.where(keep, keep.long().cumsum(1) - 1, L)  # dropped slots go to a dump column
    seq = torch.full((B, L + 1), EOS, device=dev).scatter_(1, pos, slots)
    is_ans = (torch.arange(L, device=dev) >= ans0).long().expand(B, L).contiguous()
    ans = torch.zeros(B, L + 1, dtype=torch.long, device=dev).scatter_(1, pos, is_ans).bool()
    x = seq[:, : L - 1]
    y = torch.where(ans[:, 1:L], seq[:, 1:L], -100)
    return x, y, pos, keep


# ----------------------------------------------------------------------------- data stream
class AdditionStream:
    """Infinite on-device problem generator with a hashed held-out set and an adaptive curriculum."""

    def __init__(self, device, train_digits=5, seed=0):
        self.n, self.seed, self.device = train_digits, seed, device
        self.lo = torch.tensor([0] + [10**k for k in range(1, D_MAX)], device=device)
        self.span = torch.tensor([10] + [9 * 10**k for k in range(1, D_MAX)], device=device)
        self.acc = torch.zeros(self.n**2, device=device)               # EMA exact-match per (len_a, len_b)
        self.p = torch.full((self.n**2,), 1 / self.n**2, device=device)  # sampling distribution over buckets

    def is_heldout(self, a, b):
        """Deterministic pseudo-random 10% of all pairs (Lehmer hash; no table, scales to any size)."""
        k = (a * 1000003 + b + 7919 * self.seed) % 2147483647
        return (k * 48271 % 2147483647) // 1000 % 10 == 0

    def _draw(self, n_digits):
        u = torch.rand(n_digits.shape[0], device=self.device)
        return self.lo[n_digits] + (u * self.span[n_digits]).long()

    def draw(self, la, lb, heldout=None):
        """Operands with the given digit lengths (0-indexed). heldout=True/False restricts the split."""
        a, b = self._draw(la), self._draw(lb)
        while heldout is not None and (bad := self.is_heldout(a, b) != heldout).any():  # rejection-resample
            a, b = torch.where(bad, self._draw(la), a), torch.where(bad, self._draw(lb), b)
        return a, b

    def sample(self, B):
        """A training batch: follows the curriculum and excludes every held-out pair."""
        bucket = torch.multinomial(self.p, B, replacement=True)
        a, b = self.draw(bucket // self.n, bucket % self.n, heldout=False)
        return a, b, bucket

    def update_curriculum(self, bucket, exact, ema=0.9, floor=0.05):
        n = torch.zeros_like(self.acc).index_add_(0, bucket, torch.ones_like(exact))
        hit = torch.zeros_like(self.acc).index_add_(0, bucket, exact)
        seen = n > 0
        self.acc[seen] = ema * self.acc[seen] + (1 - ema) * (hit[seen] / n[seen])
        w = (1 - self.acc) + floor  # sample in proportion to the error rate
        self.p = w / w.sum()

    def exam(self, per_bucket):
        """Never-trained problems for every (len_a, len_b) up to D_MAX: held-out pairs inside the
        training range, anything at all outside it. Returns a, b, bucket (0..D_MAX^2-1), is_ood."""
        k = torch.arange(D_MAX**2, device=self.device).repeat_interleave(per_bucket)
        la, lb = k // D_MAX, k % D_MAX
        ood = (la >= self.n) | (lb >= self.n)
        a, b = self.draw(la, lb)
        ai, bi = self.draw(la[~ood], lb[~ood], heldout=True)
        a[~ood], b[~ood] = ai, bi
        return a, b, k, ood

    def map_values(self, per_band, seed=1):
        """Sorted sample values per digit-length band: the axes of the error map."""
        g = torch.Generator().manual_seed(seed)
        lo, span = self.lo.cpu(), self.span.cpu()
        v = [(lo[k] + (torch.rand(per_band, generator=g) * span[k]).long()).sort().values for k in range(D_MAX)]
        return torch.cat(v).to(self.device)


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, a, b, bs=8192):
    """Teacher-forced scoring (exact match here == greedy-decoding exact match).
    Returns CPU tensors: exact (N,), answer nll (N,), per-place hit / valid (N, D_MAX+1), n answer tokens (N,)."""
    L, ans0 = layout(D_MAX)
    out = []
    for i in range(0, len(a), bs):
        x, y, pos, keep = encode(a[i:i + bs], b[i:i + bs])
        logits = model(x)[0].float()
        nll = -F.log_softmax(logits, -1).gather(2, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * (y >= 0)
        hit = (logits.argmax(-1) == y) | (y < 0)
        place = (pos[:, ans0:ans0 + D_MAX + 1] - 1).clamp(0, L - 2)
        out.append((hit.all(1), nll.sum(1), hit.gather(1, place), keep[:, ans0:ans0 + D_MAX + 1], (y >= 0).sum(1)))
    return [torch.cat(t).cpu() for t in zip(*out)]


def n_carries(a, b):
    return sum(((a % 10**k + b % 10**k) >= 10**k).long() for k in range(1, D_MAX + 1))


def error_map(model, stream, per_band, step):
    v = stream.map_values(per_band)
    ga, gb = v.repeat_interleave(len(v)), v.repeat(len(v))
    exact, nll, *_ = evaluate(model, ga, gb)
    band = torch.arange(len(v)) // per_band
    ood = ((band.repeat_interleave(len(v)) >= stream.n) | (band.repeat(len(v)) >= stream.n))
    ex = exact.float()
    acc_ood = ex[ood].mean().item() if ood.any() else float("nan")
    return viz.error_map((1 - torch.exp(-nll)).view(len(v), len(v)).numpy(), ex[~ood].mean().item(), acc_ood,
                         step, D_MAX, stream.n)


@torch.no_grad()
def log_lens(writer, model, tag, a, b, step, with_attention=False):
    """Decode the residual stream after every layer on one showcase problem."""
    dev = model.wte.weight.device
    x, y, _, _ = encode(torch.tensor([a], device=dev), torch.tensor([b], device=dev))
    n_tok = int((y[0] >= 0).nonzero().max()) + 1  # trim the padding
    x, y = x[:, :n_tok], y[:, :n_tok]
    _, _, attns, hiddens = model(x, return_internals=True)
    if with_attention:
        writer.add_figure("3_attention/in_distribution", viz.attention(attns, list(decode(x[0].tolist())), step), step)
    ans_pos = (y[0] >= 0).nonzero().squeeze(1)
    probs = torch.stack([model.head(model.norm(h))[0, ans_pos].float().softmax(-1) for h in hiddens])  # (layer, tok, vocab)
    target = y[0, ans_pos]
    p_correct = probs.gather(2, target.expand(len(hiddens), -1).unsqueeze(-1)).squeeze(-1).cpu().numpy()
    top = [[CHARS[i] for i in row] for row in probs.argmax(-1).tolist()]
    writer.add_figure(tag, viz.logit_lens(p_correct, top, list(decode(target.tolist())), step, f"{a} + {b}"), step)


@torch.no_grad()
def log_eval(writer, model, stream, exam, step, map_per_band=24):
    """Everything that makes the dashboard: scalars, heat maps, attention, lens, samples."""
    model.eval()
    ea, eb, bucket, ood = exam
    exact, nll, place_hit, place_ok, n_tok = evaluate(model, ea, eb)
    bucket, ood, exact_f = bucket.cpu(), ood.cpu(), exact.float()
    carries, max_len = n_carries(ea, eb).cpu(), torch.maximum(bucket // D_MAX, bucket % D_MAX)
    acc_in = exact_f[~ood].mean().item()
    acc_ood = exact_f[ood].mean().item() if ood.any() else float("nan")
    writer.add_scalar("accuracy/holdout", acc_in, step)
    writer.add_scalar("loss/holdout", (nll[~ood].sum() / n_tok[~ood].sum()).item(), step)
    if ood.any():
        writer.add_scalar("accuracy/ood", acc_ood, step)
        writer.add_scalar("loss/ood", (nll[ood].sum() / n_tok[ood].sum()).item(), step)
    for k in range(D_MAX):
        writer.add_scalar(f"acc_by_max_digits/{k + 1}d", exact_f[max_len == k].mean().item(), step)
    for k in range(stream.n + 1):
        sel = ~ood & (carries == k)
        if sel.any():
            writer.add_scalar(f"acc_by_carries/{k}_carries", exact_f[sel].mean().item(), step)
    for k in range(D_MAX + 1):  # answer digit places, most significant first
        for name, sel in (("acc_by_place", ~ood), ("ood_by_place", ood)):
            ok = place_ok[:, k] & sel
            if ok.any():
                writer.add_scalar(f"{name}/10^{D_MAX - k}", place_hit[:, k][ok].float().mean().item(), step)

    writer.add_figure("1_error_map/all_lengths", error_map(model, stream, map_per_band, step), step)
    bucket_acc = torch.stack([exact_f[bucket == k].mean() for k in range(D_MAX**2)]).view(D_MAX, D_MAX)
    p = torch.full((D_MAX, D_MAX), float("nan"))
    p[: stream.n, : stream.n] = stream.p.view(stream.n, stream.n).cpu()
    writer.add_figure("2_curriculum/accuracy_and_sampling", viz.digit_buckets(bucket_acc.numpy(), p.numpy(), step, stream.n), step)

    # showcase problems with a carry in every column, in and out of distribution
    sa, sb = "587674", "465358"
    log_lens(writer, model, "4_logit_lens/in_distribution", int(sa[: stream.n]), int(sb[: stream.n]), step, with_attention=True)
    if stream.n < D_MAX:
        log_lens(writer, model, "4_logit_lens/out_of_distribution", int(sa[: stream.n + 1]), int(sb[: stream.n + 1]), step)
    writer.add_figure("5_embeddings/tokens", viz.embeddings(model.wte.weight, CHARS, step), step)

    # real greedy generation on fixed never-trained problems
    rows = ["| split | prompt | model says | truth | |", "|---|---|---|---|---|"]
    i_in, i_ood = (~ood).nonzero().squeeze(1), ood.nonzero().squeeze(1)
    picks = i_in[7::max(1, len(i_in) // 8)][:8].tolist() + i_ood[7::max(1, len(i_ood) // 8)][:8].tolist()
    for i in picks:
        a, b = ea[i].item(), eb[i].item()
        prompt = f"\n{a} + {b} = "
        ids = model.generate(torch.tensor([[STOI[c] for c in prompt]], device=ea.device), D_MAX + 2, stop_token=EOS)
        said = decode(ids[0, len(prompt):].tolist()).split("\n")[0]
        rows.append(f"| {'OOD' if ood[i] else 'held-out'} | `{a} + {b} =` | **{said}** | {a + b} | {'✅' if said == str(a + b) else '❌'} |")
    writer.add_text("samples/greedy_generation", "\n".join(rows), step)

    for name, w in model.named_parameters():
        writer.add_histogram("weights/" + name.replace(".", "/"), w.detach().float().cpu(), step)
    model.train()
    return acc_in, acc_ood


# ----------------------------------------------------------------------------- tensorboard
LAYOUT = {"Addition": {
    "loss": ["Multiline", ["loss/train", "loss/holdout", "loss/ood"]],
    "exact-match accuracy": ["Multiline", ["accuracy/train", "accuracy/holdout", "accuracy/ood"]],
    "never-trained accuracy by longest operand": ["Multiline", ["acc_by_max_digits/.*"]],
    "held-out accuracy by number of carries": ["Multiline", ["acc_by_carries/.*"]],
    "held-out accuracy by answer digit place": ["Multiline", ["acc_by_place/.*"]],
    "OOD accuracy by answer digit place": ["Multiline", ["ood_by_place/.*"]],
}}


def launch_tensorboard(logdir, port):
    """Start TensorBoard (unless something already serves the port) and open the browser."""
    def up():
        with socket.socket() as s:
            s.settimeout(0.2)
            return s.connect_ex(("127.0.0.1", port)) == 0

    if not up():
        subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--port", str(port),
             "--reload_interval", "5", "--samples_per_plugin", "images=1000,text=1000"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(100):
            if up():
                break
            time.sleep(0.2)
    url = f"http://localhost:{port}/?darkMode=true#timeseries"
    webbrowser.open(url)
    return url


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_digits", type=int, default=5, help=f"longest operand trained on (<= {D_MAX})")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=160)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--logdir", default="runs")
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_browser", action="store_true")
    ap.add_argument("--no_curriculum", action="store_true")
    args = ap.parse_args()
    assert 1 <= args.train_digits <= D_MAX

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    model = GPT(GPTConfig(vocab_size=len(CHARS), n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda")
    lr_at = lambda s: args.lr * (s / args.warmup if s < args.warmup else
                                 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, args.steps - args.warmup))))

    stream = AdditionStream(device, args.train_digits, seed=args.seed)
    exam = stream.exam(per_bucket=384)  # fixed set of never-trained problems, uniform over length buckets

    run = Path(args.logdir) / time.strftime(f"%m%d_%H%M%S_add{args.train_digits}d")
    writer = SummaryWriter(run)
    writer.add_custom_scalars(LAYOUT)
    writer.add_text("config", f"`{vars(args)}`\n\n{model.num_params():,} parameters on `{device}`")
    print(f"model: {model.num_params() / 1e6:.2f}M params | device: {device} | run: {run}")
    if not args.no_browser:
        print("tensorboard:", launch_tensorboard(args.logdir, args.port))

    log_eval(writer, model, stream, exam, 0)
    t0, acc, ood_acc, best = time.time(), 0.0, 0.0, 0.0
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        a, b, bucket = stream.sample(args.batch)
        x, y, _, _ = encode(a, b, args.train_digits)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            exact = ((logits.argmax(-1) == y) | (y < 0)).all(1).float()
            if not args.no_curriculum:
                stream.update_curriculum(bucket, exact)
            dt, t0 = time.time() - t0, time.time()
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("accuracy/train", exact.mean().item(), step)
            writer.add_scalar("optim/lr", lr_at(step), step)
            writer.add_scalar("optim/grad_norm", gnorm.item(), step)
            writer.add_scalar("perf/problems_per_sec", args.log_every * args.batch / dt, step)
            if step % 100 == 0:
                print(f"step {step:5d} | loss {loss.item():.4f} | train acc {exact.mean().item():6.1%} | held-out {acc:6.1%} "
                      f"| OOD {ood_acc:6.1%} | {args.log_every * args.batch / dt:,.0f} problems/s")

        if step % args.eval_every == 0 or step == args.steps:
            acc, ood_acc = log_eval(writer, model, stream, exam, step)
            t0 = time.time()
            if acc >= best:
                best = acc
                torch.save({"model": model.state_dict(), "config": model.cfg, "chars": CHARS, "step": step}, run / "ckpt.pt")

    # final exam: a large fresh sample of never-trained problems + a high-resolution map
    model.eval()
    fa, fb, _, f_ood = stream.exam(per_bucket=4096)
    f_exact = evaluate(model, fa, fb)[0].float()
    f_ood = f_ood.cpu()
    in_acc, out_acc = f_exact[~f_ood].mean().item(), (f_exact[f_ood].mean().item() if f_ood.any() else float("nan"))
    writer.add_figure("1_error_map/final_high_res", error_map(model, stream, 64, args.steps), args.steps)
    summary = (f"held-out 1-{args.train_digits} digit problems: {in_acc:.3%} exact ({int((~f_ood).sum()):,} problems) | "
               f"out-of-distribution (an operand with > {args.train_digits} digits): {out_acc:.3%} ({int(f_ood.sum()):,} problems)")
    writer.add_text("final", summary.replace(" | ", "  \n"), args.steps)
    writer.close()
    print(f"final | {summary} | checkpoint: {run / 'ckpt.pt'}")


if __name__ == "__main__":
    main()

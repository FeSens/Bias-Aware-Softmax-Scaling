"""Matplotlib figures for TensorBoard: one dark visual system, colour assigned by job.

cool ramp  = "how right" (accuracy, probability, attention mass)
warm ramp  = "how wrong" (error glow, where the curriculum is pushing)
diverging  = signed similarity, neutral midpoint
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, MUTED, GRID = "#0f1117", "#e8eaed", "#8b909a", "#262a33"
COOL = LinearSegmentedColormap.from_list("cool", [SURFACE, "#143a5c", "#2a7fb8", "#7cc8f0", "#eaf7ff"])
WARM = LinearSegmentedColormap.from_list("warm", [SURFACE, "#5a1a12", "#c4441c", "#f59a3a", "#fff1c2"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2f8fd8", "#2a2d36", "#f08a3c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "font.family": "DejaVu Sans", "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
})


def show(ch: str) -> str:
    return {" ": "·", "\n": "↵"}.get(ch, ch)


def _ink_for(value: float) -> str:
    """Readable annotation ink on a ramp cell (ramps go dark -> light)."""
    return SURFACE if value > 0.62 else INK


def _colorbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, color=MUTED)
    cb.outline.set_visible(False)
    return cb


def _bands(ax, n, n_train):
    """Shared axes for digit-length grids: one band per operand length, OOD bands fenced off."""
    names = [f"{k}d" for k in range(1, n + 1)]
    ax.set_xlabel("digits in b"), ax.set_ylabel("digits in a")
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return names


def error_map(p_wrong: np.ndarray, acc_in: float, acc_ood: float, step: int, n: int, n_train: int, op: str = "+"):
    """a x b map across every operand length. Each band holds sorted sample values of that length;
    every pixel is one addition problem. Bands past n_train were never seen in training."""
    fig, ax = plt.subplots(figsize=(8.8, 7.8))
    im = ax.imshow(p_wrong, cmap=WARM, vmin=0, vmax=1, origin="lower", extent=(0, n, 0, n), interpolation="nearest")
    names = _bands(ax, n, n_train)
    ax.set_xticks(np.arange(n) + 0.5, names), ax.set_yticks(np.arange(n) + 0.5, names)
    for k in range(1, n):
        ood = k == n_train
        ax.axvline(k, color=INK if ood else MUTED, lw=1.2 if ood else 0.5, alpha=0.9 if ood else 0.45)
        ax.axhline(k, color=INK if ood else MUTED, lw=1.2 if ood else 0.5, alpha=0.9 if ood else 0.45)
    if n > n_train:
        ax.text(n - 0.03, n + 0.04, "never trained →", ha="right", va="bottom", color=INK, fontsize=8)
    ax.set_title(f"Where the model is still wrong · step {step:,}\n"
                 f"{acc_in:.1%} solved in-distribution · {acc_ood:.1%} out-of-distribution", pad=16)
    _colorbar(fig, im, ax, "1 − P(correct answer)   (dark = solved)")
    ax.text(0, -0.11, f"each pixel = one problem a {op} b · {p_wrong.shape[0] // n} sorted sample values per band",
            transform=ax.transAxes, color=MUTED, fontsize=8)
    fig.tight_layout()
    return fig


def digit_buckets(acc: np.ndarray, p_sample: np.ndarray, step: int, n_train: int):
    """Exact-match accuracy by operand length (never-trained problems), next to the curriculum."""
    n = acc.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.9))
    panels = [(acc, COOL, 1.0, "Accuracy on never-trained problems"),
              (p_sample, WARM, max(0.12, float(np.nanmax(p_sample))), "Curriculum: share of training batch")]
    for ax, (m, cmap, vmax, title) in zip(axes, panels):
        ax.imshow(np.nan_to_num(m), cmap=cmap, vmin=0, vmax=vmax)
        for i in range(n):
            for j in range(n):
                txt = "—" if np.isnan(m[i, j]) else f"{m[i, j]:.0%}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=11, fontweight="bold",
                        color=MUTED if np.isnan(m[i, j]) else _ink_for(m[i, j] / vmax))
        names = _bands(ax, n, n_train)
        ax.set_xticks(range(n), names), ax.set_yticks(range(n), names)
        ax.set_xticks(np.arange(-0.5, n), minor=True), ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)  # 2px surface gap between cells
        ax.tick_params(which="both", length=0)
        if n > n_train:  # fence off the out-of-distribution L
            ax.plot([n_train - 0.5, n_train - 0.5, -0.5], [-0.5, n_train - 0.5, n_train - 0.5], color=INK, lw=1.4)
            ax.text(n - 0.5, -0.62, "never trained →", ha="right", va="bottom", color=INK, fontsize=8)
        ax.set_title(title, pad=14)
    fig.suptitle(f"step {step:,}", x=0.99, ha="right", color=MUTED, fontsize=9)
    fig.tight_layout()
    return fig


def attention(attns, tokens: list[str], step: int):
    """Every head of every layer on one showcase problem. Rows attend to columns."""
    n_layer, n_head = len(attns), attns[0].shape[1]
    T = len(tokens)
    fig, axes = plt.subplots(n_layer, n_head, figsize=(3.5 * n_head, 3.5 * n_layer + 0.5), squeeze=False)
    labels = [show(t) for t in tokens]
    for l in range(n_layer):
        for h in range(n_head):
            ax = axes[l][h]
            ax.imshow(attns[l][0, h].float().cpu().numpy(), cmap=COOL, vmin=0, vmax=1)
            ax.set_xticks(range(T), labels, fontsize=7)
            ax.set_yticks(range(T), labels, fontsize=7)
            ax.tick_params(length=0, pad=2)
            ax.set_title(f"layer {l + 1} · head {h + 1}", fontsize=9, color=MUTED, fontweight="normal")
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle(f"Attention on “{''.join(labels)}”  ·  step {step:,}  ·  row (query) → column (key)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def logit_lens(p_correct: np.ndarray, top_chars: list[list[str]], answer: list[str], step: int, problem: str):
    """Decode the residual stream after each layer: when does the answer appear?"""
    n_layer, n_tok = p_correct.shape
    fig, ax = plt.subplots(figsize=(1.15 * n_tok + 2.6, 0.75 * n_layer + 1.9))
    im = ax.imshow(p_correct, cmap=COOL, vmin=0, vmax=1, aspect="auto")
    for l in range(n_layer):
        for t in range(n_tok):
            ax.text(t, l, show(top_chars[l][t]), ha="center", va="center", fontsize=14,
                    fontweight="bold", color=_ink_for(p_correct[l, t]))
    ax.set_xticks(range(n_tok), [f"target {show(c)}" for c in answer])
    ax.set_yticks(range(n_layer), [f"after layer {l + 1}" for l in range(n_layer)])
    ax.set_xticks(np.arange(-0.5, n_tok), minor=True), ax.set_yticks(np.arange(-0.5, n_layer), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Logit lens on {problem} · step {step:,} · glyph = layer's top guess")
    _colorbar(fig, im, ax, "P(correct token)")
    fig.tight_layout()
    return fig


def embeddings(wte: torch.Tensor, chars: str, step: int):
    """Token geometry: cosine similarity, and digits projected on their top-2 principal axes."""
    w = wte.detach().float().cpu()
    wn = torch.nn.functional.normalize(w, dim=1)
    sim = (wn @ wn.T).numpy()
    labels = [show(c) for c in chars]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.8))

    im = ax1.imshow(sim, cmap=DIVERGING, vmin=-1, vmax=1)
    ax1.set_xticks(range(len(chars)), labels), ax1.set_yticks(range(len(chars)), labels)
    ax1.tick_params(length=0)
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.set_title("Token embedding cosine similarity")
    _colorbar(fig, im, ax1, "cosine similarity")

    d = w[:10] - w[:10].mean(0)
    _, S, V = torch.linalg.svd(d, full_matrices=False)
    xy = (d @ V[:2].T).numpy()
    var = (S[:2] ** 2 / (S ** 2).sum()).numpy()
    ax2.plot(xy[:, 0], xy[:, 1], color=MUTED, lw=1, alpha=0.6, zorder=1)
    ax2.scatter(xy[:, 0], xy[:, 1], c=np.arange(10), cmap=COOL, vmin=-4, vmax=9, s=340,
                edgecolors=SURFACE, linewidths=2, zorder=2)
    for i in range(10):
        ax2.text(xy[i, 0], xy[i, 1], str(i), ha="center", va="center", fontsize=11, fontweight="bold",
                 color=_ink_for((i + 4) / 13), zorder=3)
    ax2.set_xlabel(f"PC1 ({var[0]:.0%} of variance)"), ax2.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax2.grid(color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.margins(0.12)
    ax2.set_title("Digit embeddings · line follows 0 → 9")
    fig.suptitle(f"step {step:,}", x=0.99, ha="right", color=MUTED, fontsize=9)
    fig.tight_layout()
    return fig


def op_buckets(accs: dict, trained: dict, step: int):
    """Exact-match accuracy by operand length for each operation; the fence marks what that operation trained on."""
    fig, axes = plt.subplots(1, len(accs), figsize=(7.2 * len(accs), 6.9), squeeze=False)
    for ax, (op, acc) in zip(axes[0], accs.items()):
        n, nt = acc.shape[0], trained[op]
        ax.imshow(acc, cmap=COOL, vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{acc[i, j]:.0%}", ha="center", va="center", fontsize=9, fontweight="bold",
                        color=_ink_for(acc[i, j]))
        names = [f"{k}d" for k in range(1, n + 1)]
        ax.set_xticks(range(n), names), ax.set_yticks(range(n), names)
        ax.set_xticks(np.arange(-0.5, n), minor=True), ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        if 0 < nt < n:
            ax.plot([nt - 0.5, nt - 0.5, -0.5], [-0.5, nt - 0.5, nt - 0.5], color=INK, lw=1.4)
        ax.set_xlabel("digits in b"), ax.set_ylabel("digits in a")
        ax.set_title(f"a {op} b · trained up to {nt} digits" if nt else f"a {op} b · never trained", pad=12)
    fig.suptitle(f"never-trained problems · step {step:,}", x=0.99, ha="right", color=MUTED, fontsize=9)
    fig.tight_layout()
    return fig

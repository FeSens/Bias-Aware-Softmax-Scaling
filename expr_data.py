"""Multi-step exact-rational arithmetic expression generator & canonical CoT rewriter.

Generates nested expressions with +, -, *, /, parentheses, and exact Python Fraction
semantics — e.g.:
    <(17 * 3) / (4 + 8) - 5
    = 51 / (4 + 8) - 5
    = 51 / 12 - 5
    = 17/4 - 5
    = (17 - 5 * 4) / 4
    = (17 - 20) / 4
    = -3 / 4
    = -3/4>

Every step applies ONE deterministic canonical rewrite:
  1. Select the deepest leftmost parenthesized group (or top-level if none).
  2. Pick the leftmost highest-precedence binary op (*, / before +, -).
  3. If both operands are integers, evaluate to exact reduced Fraction.
  4. If either operand is a non-integer fraction (den > 1), expand symbolically
     into an exact integer-operand expression (preserving precedence) so every
     arithmetic step stays atomic (one integer +, -, *, or / reduction).
  5. Strip redundant parentheses around a single atom.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import random
from typing import Sequence

import torch

EXPR_CHARS = "0123456789 ()+-*/=\n<>"
STOI = {c: i for i, c in enumerate(EXPR_CHARS)}
ITOS = {i: c for i, c in enumerate(EXPR_CHARS)}
BOS_ID = STOI["<"]
EOS_ID = STOI[">"]
PAD_ID = EOS_ID
NL_ID = STOI["\n"]
LP_ID = STOI["("]
RP_ID = STOI[")"]

OPS = ("+", "-", "*", "/")


def fmt_frac(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def format_tokens(toks: Sequence[Fraction | str]) -> str:
    out: list[str] = []
    for i, t in enumerate(toks):
        s = fmt_frac(t) if isinstance(t, Fraction) else t
        if i == 0:
            out.append(s)
        elif s == ")":
            out.append(s)
        elif out[-1].endswith("("):
            out.append(s)
        else:
            out.append(" " + s)
    return "".join(out)


def parse_expr_str(s: str) -> list[Fraction | str]:
    """Parse canonical expression string back into token list (for verification & eval)."""
    toks: list[Fraction | str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == " ":
            i += 1
            continue
        if c in "()":
            toks.append(c)
            i += 1
            continue
        if c in "+*/":
            toks.append(c)
            i += 1
            continue
        if c == "-" and (i + 1 >= n or s[i + 1] == " "):
            toks.append("-")
            i += 1
            continue
        # Signed integer or tight fraction literal (e.g. -3/4, 17/6, 42)
        j = i
        if s[j] == "-":
            j += 1
        start_num = j
        while j < n and s[j].isdigit():
            j += 1
        if j == start_num:
            raise ValueError(f"Unexpected char at {i}: {s!r}")
        if j < n and s[j] == "/" and (j + 1 < n and s[j + 1].isdigit()):
            j += 1
            while j < n and s[j].isdigit():
                j += 1
            toks.append(Fraction(s[i:j]))
        else:
            toks.append(Fraction(int(s[i:j]), 1))
        i = j
    return toks


def eval_tokens(toks: Sequence[Fraction | str]) -> Fraction:
    """Exact recursive-descent evaluator for token lists."""
    pos = 0

    def parse_add() -> Fraction:
        nonlocal pos
        v = parse_mul()
        while pos < len(toks) and toks[pos] in ("+", "-"):
            op = toks[pos]
            pos += 1
            r = parse_mul()
            v = v + r if op == "+" else v - r
        return v

    def parse_mul() -> Fraction:
        nonlocal pos
        v = parse_atom()
        while pos < len(toks) and toks[pos] in ("*", "/"):
            op = toks[pos]
            pos += 1
            r = parse_atom()
            if op == "*":
                v = v * r
            else:
                if r == 0:
                    raise ZeroDivisionError("division by zero")
                v = v / r
        return v

    def parse_atom() -> Fraction:
        nonlocal pos
        t = toks[pos]
        if t == "(":
            pos += 1
            v = parse_add()
            if pos >= len(toks) or toks[pos] != ")":
                raise ValueError("missing ')'")
            pos += 1
            return v
        if isinstance(t, Fraction):
            pos += 1
            return t
        raise ValueError(f"unexpected token {t!r}")

    val = parse_add()
    if pos != len(toks):
        raise ValueError(f"trailing tokens at {pos}: {toks[pos:]}")
    return val


def _apply_int_op(a: Fraction, op: str, b: Fraction) -> Fraction:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b


def _expand_frac_op(a: Fraction, op: str, b: Fraction) -> list[Fraction | str]:
    """Expand a binary op where at least one operand has denominator > 1 into integer ops."""
    p, q = a.numerator, a.denominator
    r, s = b.numerator, b.denominator
    F = Fraction
    if op in ("+", "-"):
        if q == s:
            return ["(", F(p), op, F(r), ")", "/", F(q)]
        if s == 1:
            return ["(", F(p), op, F(r), "*", F(q), ")", "/", F(q)]
        if q == 1:
            return ["(", F(p), "*", F(s), op, F(r), ")", "/", F(s)]
        return ["(", F(p), "*", F(s), op, F(r), "*", F(q), ")", "/", "(", F(q), "*", F(s), ")"]
    if op == "*":
        if s == 1:
            return ["(", F(p), "*", F(r), ")", "/", F(q)]
        if q == 1:
            return ["(", F(p), "*", F(r), ")", "/", F(s)]
        return ["(", F(p), "*", F(r), ")", "/", "(", F(q), "*", F(s), ")"]
    # op == "/"
    if r == 0:
        raise ZeroDivisionError("division by zero")
    if s == 1:
        return [F(p), "/", "(", F(q), "*", F(r), ")"]
    if q == 1:
        return ["(", F(p), "*", F(s), ")", "/", F(r)]
    return ["(", F(p), "*", F(s), ")", "/", "(", F(q), "*", F(r), ")"]


def step_once(toks: list[Fraction | str]) -> list[Fraction | str] | None:
    """Perform one canonical reduction step. Returns None when already a single Fraction atom."""
    if len(toks) == 1 and isinstance(toks[0], Fraction):
        return None

    # 1. Find deepest leftmost parenthesized group (contains no inner parens)
    depth, best_d, best_span = 0, 0, None
    stack: list[int] = []
    for i, t in enumerate(toks):
        if t == "(":
            depth += 1
            stack.append(i)
        elif t == ")":
            l = stack.pop()
            if depth > best_d:
                best_d = depth
                best_span = (l, i)
            depth -= 1

    if best_span is not None:
        l, r = best_span
        sub = toks[l + 1 : r]
        if len(sub) == 1:
            return toks[:l] + sub + toks[r + 1 :]
    else:
        l, r = -1, len(toks)
        sub = list(toks)

    # 2. Find leftmost highest-precedence operator inside flat `sub`
    op_idx = -1
    for i, t in enumerate(sub):
        if t in ("*", "/"):
            op_idx = i
            break
    if op_idx < 0:
        for i, t in enumerate(sub):
            if t in ("+", "-"):
                op_idx = i
                break
    if op_idx < 0:
        raise ValueError(f"Malformed subexpression: {sub!r}")

    a, op, b = sub[op_idx - 1], sub[op_idx], sub[op_idx + 1]
    assert isinstance(a, Fraction) and isinstance(b, Fraction) and isinstance(op, str)

    if a.denominator == 1 and b.denominator == 1:
        repl: list[Fraction | str] = [_apply_int_op(a, op, b)]
    else:
        repl = _expand_frac_op(a, op, b)

    new_sub = sub[: op_idx - 1] + repl + sub[op_idx + 2 :]
    if best_span is not None:
        # Strip outer parentheses if the group reduced to a single atom
        if len(new_sub) == 1 and isinstance(new_sub[0], Fraction):
            return toks[:l] + new_sub + toks[r + 1 :]
        return toks[: l + 1] + new_sub + toks[r:]
    return new_sub


def solve_trace(
    toks: list[Fraction | str],
    max_steps: int = 40,
    max_num: int = 10**7,
    max_den: int = 10**4,
) -> list[str] | None:
    """Return full list of formatted lines [prompt_expr, step1, ..., final_ans], or None if bounds exceeded."""
    cur = list(toks)
    lines = [format_tokens(cur)]
    for _ in range(max_steps):
        nxt = step_once(cur)
        if nxt is None:
            return lines
        for t in nxt:
            if isinstance(t, Fraction):
                if abs(t.numerator) > max_num or t.denominator > max_den:
                    return None
        lines.append(format_tokens(nxt))
        cur = nxt
    return None


def _build_random_tree(
    rng: random.Random,
    n_ops: int,
    max_depth: int,
    cur_depth: int,
    add_max: int,
    mul_max: int,
    force_depth: int = 0,
    op_weights: tuple[float, float, float, float] = (0.35, 0.30, 0.20, 0.15),
) -> tuple[list[Fraction | str], int]:
    """Recursively build a random expression token list with exact `n_ops` binary operators."""
    if n_ops == 0:
        return [Fraction(rng.randint(1, add_max), 1)], cur_depth

    op = rng.choices(OPS, weights=op_weights, k=1)[0]
    # Split remaining (n_ops - 1) operators between left and right subtrees
    if force_depth > cur_depth and n_ops >= 2:
        # Bias toward putting enough ops on one side to reach force_depth
        side = rng.randint(0, 1)
        need = force_depth - cur_depth
        if side == 0:
            l_ops = rng.randint(min(need, n_ops - 1), n_ops - 1)
            r_ops = n_ops - 1 - l_ops
        else:
            r_ops = rng.randint(min(need, n_ops - 1), n_ops - 1)
            l_ops = n_ops - 1 - r_ops
    else:
        l_ops = rng.randint(0, n_ops - 1)
        r_ops = n_ops - 1 - l_ops

    # Decide whether to wrap left/right children in parentheses
    can_wrap = cur_depth < max_depth
    l_wrap = can_wrap and l_ops > 0 and (
        (force_depth > cur_depth and l_ops >= r_ops) or rng.random() < 0.45
    )
    r_wrap = can_wrap and r_ops > 0 and (
        (force_depth > cur_depth and r_ops > l_ops and not l_wrap) or rng.random() < 0.45
    )

    child_add_max = mul_max if op in ("*", "/") else add_max
    l_toks, l_d = _build_random_tree(
        rng, l_ops, max_depth, cur_depth + (1 if l_wrap else 0),
        child_add_max if l_ops == 0 else add_max, mul_max,
        force_depth=force_depth if l_wrap else 0, op_weights=op_weights,
    )
    r_toks, r_d = _build_random_tree(
        rng, r_ops, max_depth, cur_depth + (1 if r_wrap else 0),
        child_add_max if r_ops == 0 else add_max, mul_max,
        force_depth=force_depth if r_wrap else 0, op_weights=op_weights,
    )

    # Ensure parentheses whenever required by operator precedence or wrapping choice
    l_has_add = any(t in ("+", "-") for t in l_toks) and not (l_toks[0] == "(" and l_toks[-1] == ")")
    r_has_add = any(t in ("+", "-") for t in r_toks) and not (r_toks[0] == "(" and r_toks[-1] == ")")
    if op in ("*", "/"):
        if l_has_add and cur_depth < max_depth:
            l_wrap = True
        if r_has_add and cur_depth < max_depth:
            r_wrap = True
    if op == "/" and r_ops > 0 and cur_depth < max_depth:
        r_wrap = True
    if op == "-" and r_has_add and cur_depth < max_depth:
        r_wrap = True

    if l_wrap and l_ops > 0:
        l_toks = ["("] + l_toks + [")"]
        l_d = max(l_d, cur_depth + 1)
    if r_wrap and r_ops > 0:
        r_toks = ["("] + r_toks + [")"]
        r_d = max(r_d, cur_depth + 1)

    return l_toks + [op] + r_toks, max(l_d, r_d)


def max_paren_depth(toks: Sequence[Fraction | str]) -> int:
    d = m = 0
    for t in toks:
        if t == "(":
            d += 1
            if d > m:
                m = d
        elif t == ")":
            d -= 1
    return m


def _is_heldout(prompt: str, mod: int = 20) -> bool:
    """Deterministic 5% hash holdout on prompt strings (zero train/test overlap)."""
    return (int(hashlib.blake2b(prompt.encode(), digest_size=4).hexdigest(), 16) % mod) == 0


def sample_problem(
    rng: random.Random,
    n_ops_range: tuple[int, int] = (1, 4),
    max_depth: int = 2,
    min_depth: int = 0,
    add_max: int = 200,
    mul_max: int = 25,
    max_num: int = 9999,
    max_den: int = 99,
    max_seq_len: int = 320,
    max_steps: int = 24,
    holdout_filter: str | None = None,
) -> tuple[str, list[str], str]:
    """Sample a valid exact-rational problem returning (prompt_str, lines, full_seq)."""
    for _ in range(500):
        n_ops = rng.randint(n_ops_range[0], n_ops_range[1])
        # Occasionally include larger multi-digit integers inside + / - chains
        a_max = add_max if rng.random() < 0.7 else max(add_max, min(max_num // 4, add_max * 10))
        toks, _ = _build_random_tree(
            rng, n_ops, max_depth, 0, add_max=a_max, mul_max=mul_max, force_depth=min_depth
        )
        d = max_paren_depth(toks)
        if d < min_depth or d > max_depth:
            continue
        prompt = format_tokens(toks)
        if holdout_filter == "train" and _is_heldout(prompt):
            continue
        if holdout_filter == "eval" and not _is_heldout(prompt):
            continue
        try:
            lines = solve_trace(toks, max_steps=max_steps, max_num=max_num, max_den=max_den)
        except ZeroDivisionError:
            continue
        if lines is None or len(lines) < 2:
            continue
        full = "<" + "\n= ".join(lines) + ">"
        if len(full) > max_seq_len:
            continue
        return prompt, lines, full
    raise RuntimeError("Failed to sample valid expression within 500 tries")


def encode_sequences(full_seqs: Sequence[str], dev: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Encode `<expr\\n= step1\\n= ...\\n= ans>` into padded (x, y) tensors.

    Supervision (`y >= 0`) begins at the first token AFTER the initial `\\n= `
    and includes the stopping token `>`; prompt tokens and trailing pads get -100.
    """
    B = len(full_seqs)
    max_l = max(len(s) for s in full_seqs) - 1
    x = torch.full((B, max_l), PAD_ID, dtype=torch.long)
    y = torch.full((B, max_l), -100, dtype=torch.long)
    for i, s in enumerate(full_seqs):
        ids = [STOI[c] for c in s]
        L = len(ids) - 1
        x[i, :L] = torch.tensor(ids[:-1], dtype=torch.long)
        # First "\n= " is the prompt/CoT boundary; supervise predicting step1..'>'
        sep = s.index("\n= ")
        start_y = sep + 2  # x[sep+2] is ' ' after '=', predicting first char of step1
        y[i, start_y:L] = torch.tensor(ids[start_y + 1 :], dtype=torch.long)
    return x.to(dev), y.to(dev)


def encode_prompts(prompts: Sequence[str], dev: torch.device | str = "cpu") -> list[list[int]]:
    """Encode prompt prefixes `<expr\\n= ` for autoregressive generation."""
    return [[STOI[c] for c in f"<{p}\n= "] for p in prompts]


def extract_final_fraction(gen_text: str) -> Fraction | None:
    """Parse the final line before `>` as an exact Fraction, returning None on parse failure."""
    if ">" in gen_text:
        gen_text = gen_text.split(">", 1)[0]
    lines = [ln.strip() for ln in gen_text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    if last.startswith("="):
        last = last[1:].strip()
    try:
        toks = parse_expr_str(last)
        if len(toks) == 1 and isinstance(toks[0], Fraction):
            return toks[0]
        return eval_tokens(toks)
    except Exception:
        return None


def build_eval_suite(seed: int = 4242, per_split: int = 128) -> dict[str, list[tuple[str, list[str], str]]]:
    """Fixed deterministic eval suite testing 4 distinct generalization axes."""
    rng = random.Random(seed)
    return {
        "in_dist": [
            sample_problem(rng, n_ops_range=(1, 4), max_depth=2, add_max=200, mul_max=25,
                           max_num=9999, max_den=99, max_seq_len=320, holdout_filter="eval")
            for _ in range(per_split)
        ],
        "ood_steps": [
            sample_problem(rng, n_ops_range=(5, 7), max_depth=2, add_max=200, mul_max=25,
                           max_num=99999, max_den=199, max_seq_len=520, max_steps=36)
            for _ in range(per_split)
        ],
        "ood_depth": [
            sample_problem(rng, n_ops_range=(3, 5), max_depth=4, min_depth=3, add_max=200, mul_max=25,
                           max_num=99999, max_den=199, max_seq_len=480, max_steps=32)
            for _ in range(per_split)
        ],
        "ood_scale": [
            sample_problem(rng, n_ops_range=(2, 4), max_depth=2, add_max=9999, mul_max=45,
                           max_num=999999, max_den=499, max_seq_len=420, max_steps=28)
            for _ in range(per_split)
        ],
    }

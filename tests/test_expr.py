"""Unit, exact-rational invariance, causality, and architecture tests for multi-step expressions."""
from fractions import Fraction
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from expr_data import (
    EOS_ID,
    PAD_ID,
    STOI,
    build_eval_suite,
    encode_prompts,
    encode_sequences,
    eval_tokens,
    extract_final_fraction,
    parse_expr_str,
    sample_problem,
    solve_trace,
)
from neo_expr import NeoExprConfig, NeoExprGPT, causal_paren_depth


def test_user_example_exact_rational_trace():
    """Verify exact step-by-step rational reduction on the user's motivating expression."""
    expr = "(17 * 22) / (18 + 321 + 4122451) - 123 + 34 / 12"
    toks = parse_expr_str(expr)
    true_val = eval_tokens(toks)
    lines = solve_trace(toks, max_steps=60, max_num=10**15, max_den=10**12)
    assert lines is not None
    # Every single step must preserve exact Fraction value!
    for ln in lines:
        assert eval_tokens(parse_expr_str(ln)) == true_val, f"Value drift at step: {ln}"
    assert extract_final_fraction("<" + "\n= ".join(lines) + ">") == true_val
    print(f"PASS test_user_example_exact_rational_trace ({len(lines) - 1} steps -> {lines[-1]})")


def test_random_traces_exact_and_no_overlap():
    """Verify 200 sampled problems across all 4 eval splits preserve exact Fraction value at every step."""
    suite = build_eval_suite(seed=7, per_split=50)
    rng = random.Random(99)
    train_prompts = {
        sample_problem(rng, holdout_filter="train")[0] for _ in range(100)
    }
    in_prompts = {it[0] for it in suite["in_dist"]}
    assert train_prompts.isdisjoint(in_prompts), "Train/eval prompt overlap detected!"

    for split, items in suite.items():
        for prompt, lines, full in items:
            tv = eval_tokens(parse_expr_str(prompt))
            for ln in lines:
                assert eval_tokens(parse_expr_str(ln)) == tv, f"Step drift in {split}: {ln}"
            assert extract_final_fraction(full) == tv
    print("PASS test_random_traces_exact_and_no_overlap (200 traces verified)")


def test_causal_paren_depth():
    """Verify running parenthesis depth matches expected values and resets at newlines."""
    s = "<((1 + 2) * 3)\n= (3 * 3)\n= 9>"
    idx = torch.tensor([[STOI[c] for c in s]])
    d = causal_paren_depth(idx)[0].tolist()
    # Map each char to expected depth
    expected = []
    cur = 0
    for c in s:
        if c in ("\n", "<"):
            cur = 0
            expected.append(0)
        elif c == "(":
            cur += 1
            expected.append(cur)
        elif c == ")":
            expected.append(cur)
            cur = max(0, cur - 1)
        else:
            expected.append(cur)
    assert d == expected, f"Depth mismatch:\n  got: {d}\n  exp: {expected}"
    print("PASS test_causal_paren_depth")


def test_neo_and_rope_causality_and_grads():
    """Verify strict causality, backward gradients, and batched greedy generation for both archs."""
    suite = build_eval_suite(seed=11, per_split=4)
    seqs = [it[2] for it in suite["in_dist"]]
    x, y = encode_sequences(seqs)

    for arch in ("neo", "rope"):
        torch.manual_seed(0)
        m = NeoExprGPT(NeoExprConfig(arch=arch, n_layer=4, n_head=4, n_embd=64)).eval()
        lg1, _ = m(x)
        x2 = x.clone()
        t_mut = x.shape[1] // 2
        x2[:, t_mut] = STOI["("]  # even mutating a paren must not leak backward!
        lg2, _ = m(x2)
        diff = (lg1[:, :t_mut] - lg2[:, :t_mut]).abs().max().item()
        assert diff < 1e-5, f"{arch} causality leak before t={t_mut}: {diff}"

        m.train()
        _, loss = m(x, y)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

        m.eval()
        p_ids = encode_prompts([it[0] for it in suite["in_dist"]])
        outs = m.generate_batch(p_ids, max_new_tokens=16, stop_token=EOS_ID, pad_token=PAD_ID)
        assert len(outs) == 4 and all(len(o) > len(p) for o, p in zip(outs, p_ids))

        # Verify exact KV-cached vs uncached decoding parity on a single sequence
        cur_ref = torch.tensor([p_ids[0]], dtype=torch.long)
        for _ in range(16):
            lg_ref, _ = m(cur_ref)
            nxt_ref = lg_ref[:, -1].argmax(-1, keepdim=True)
            cur_ref = torch.cat([cur_ref, nxt_ref], dim=1)
            if int(nxt_ref.item()) == EOS_ID:
                break
        assert outs[0] == cur_ref[0].tolist(), f"{arch} KV-cache mismatch:\n  cache={outs[0]}\n  ref  ={cur_ref[0].tolist()}"
        print(f"PASS test_neo_and_rope_causality_and_grads (arch={arch}, params={m.num_params():,}, KV-cache exact)")


if __name__ == "__main__":
    test_user_example_exact_rational_trace()
    test_random_traces_exact_and_no_overlap()
    test_causal_paren_depth()
    test_neo_and_rope_causality_and_grads()

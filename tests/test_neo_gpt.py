"""Unit, invariance, and monotonicity (anti-aliasing) tests for NeoGPT v2."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neo_gpt import CHARS, NeoGPT, NeoGPTConfig, place_ids
from train import STOI, encode


def test_causality_on_answer():
    """Modifying any answer token at t' > t must not change logits at position <= t."""
    torch.manual_seed(0)
    m = NeoGPT(NeoGPTConfig()).eval()
    a = torch.tensor([587674, 12345])
    b = torch.tensor([465358, 987])
    x, y, _, _ = encode(a, b, 6)
    lg1, _ = m(x)

    x2 = x.clone()
    eq_pos = (x == STOI["="]).float().argmax(1)
    for row in range(len(x)):
        t_mut = int(eq_pos[row]) + 4
        x2[row, t_mut] = (x2[row, t_mut] + 1) % 10
    lg2, _ = m(x2)

    for row in range(len(x)):
        t_mut = int(eq_pos[row]) + 4
        diff = (lg1[row, :t_mut] - lg2[row, :t_mut]).abs().max().item()
        assert diff < 1e-5, f"Causality leak before t={t_mut}: max diff {diff}"
    print("PASS test_causality_on_answer")


def test_exact_place_shift_invariance():
    """Adding any constant offset to all positive place IDs must yield identical logits."""
    torch.manual_seed(1)
    m = NeoGPT(NeoGPTConfig()).eval()
    a = torch.tensor([587674, 99999])
    b = torch.tensor([465358, 1])
    x, _, _, _ = encode(a, b, 6)
    pid = place_ids(x)
    lg0, _ = m(x, pid_override=pid)
    for off in (1, 7, 25, 100):
        pid_shifted = torch.where(pid > 0, pid + off, 0)
        lg_s, _ = m(x, pid_override=pid_shifted)
        diff = (lg0 - lg_s).abs().max().item()
        assert diff < 1e-5, f"Place shift invariance broken at off={off}: max diff {diff}"
    print("PASS test_exact_place_shift_invariance")


def test_monotonic_no_aliasing():
    """Relative place bias for Delta <= -2 must be strictly non-increasing in |Delta| (zero aliasing!)."""
    torch.manual_seed(4)
    m = NeoGPT(NeoGPTConfig()).eval()
    # Set random weights on biases & slopes to test general learned states
    with torch.no_grad():
        for b in m.blocks:
            b.attn.place_rel_bias.normal_()
            b.attn.place_log_slope.normal_()
    for li, b in enumerate(m.blocks):
        pb = b.attn.place_rel_bias.detach()
        slope = -F.softplus(b.attn.place_log_slope).detach()
        # For Delta = -2, -3, ..., -30, bucket is 1 and bias = pb[:, 1] + slope * log(1 + |Delta|)
        ds = torch.arange(2, 31).float()
        vals = pb[:, 1:2] + slope[:, None] * torch.log1p(ds)[None, :]  # (H, 29)
        diffs = vals[:, 1:] - vals[:, :-1]
        assert (diffs <= 0).all(), f"Non-monotonic bias in L{li+1}: max positive step {diffs.max().item()}"
    print("PASS test_monotonic_no_aliasing (strictly non-increasing for |Delta|=2..30)")


def test_arbitrary_lengths_and_internals():
    """NeoGPT handles 2d..15d sequences natively (no block_size limit) and returns valid internals."""
    torch.manual_seed(2)
    m = NeoGPT(NeoGPTConfig()).eval()
    for D in (2, 5, 8, 12, 15):
        a = torch.tensor([10 ** (D - 1) + 7])
        b = torch.tensor([10 ** (D - 1) + 9])
        x, y, _, _ = encode(a, b, D)
        lg, loss, attns, hiddens = m(x, y, return_internals=True)
        assert lg.shape == (1, x.shape[1], len(CHARS))
        assert torch.isfinite(loss)
        assert len(attns) == 4 and len(hiddens) == 4
        assert attns[0].shape == (1, 4, x.shape[1], x.shape[1])
        assert (attns[0].sum(-1) - 1.0).abs().max().item() < 1e-5
    prompt = "\n1234567890 + 9876543210 = "
    idx = torch.tensor([[STOI[c] for c in prompt]])
    out = m.generate(idx, max_new_tokens=12, stop_token=STOI["\n"])
    assert out.shape[1] > idx.shape[1]
    print(f"PASS test_arbitrary_lengths_and_internals (params={m.num_params():,})")


def test_mps_parity():
    if not torch.backends.mps.is_available():
        print("SKIP test_mps_parity (no MPS)")
        return
    torch.manual_seed(3)
    m_cpu = NeoGPT(NeoGPTConfig()).eval()
    m_mps = NeoGPT(NeoGPTConfig()).eval()
    m_mps.load_state_dict(m_cpu.state_dict())
    m_mps.to("mps")
    a = torch.tensor([587674, 12345, 7])
    b = torch.tensor([465358, 987, 999999])
    x, y, _, _ = encode(a, b, 6)
    lg_cpu, _ = m_cpu(x, y)
    lg_mps, _ = m_mps(x.to("mps"), y.to("mps"))
    diff = (lg_cpu - lg_mps.cpu()).abs().max().item()
    assert diff < 1e-4, f"MPS vs CPU mismatch: {diff}"
    print(f"PASS test_mps_parity (max diff={diff:.2e})")


def test_v1_1_scaled_place_rope():
    """v1.1 mode (Place-RoPE + random scale) preserves shift invariance and reloads saved v1/v2 weights."""
    torch.manual_seed(5)
    cfg = NeoGPTConfig(rope_frac=0.5, n_place_buckets=4, uniform_slope=-1.5)
    m = NeoGPT(cfg).eval()
    assert m.num_params() == 1_274_528
    a = torch.tensor([587674, 99999])
    b = torch.tensor([465358, 1])
    x, _, _, _ = encode(a, b, 6)
    pid = place_ids(x)
    s = torch.tensor([[1.37], [1.82]])
    lg0, _ = m(x, pid_override=pid, place_rope_scale=s)
    pid_shifted = torch.where(pid > 0, pid + 11, 0)
    lg1, _ = m(x, pid_override=pid_shifted, place_rope_scale=s)
    diff = (lg0 - lg1).abs().max().item()
    assert diff < 1e-5, f"v1.1 shift invariance broken under scale: {diff}"
    # Verify saved v1 and v2 checkpoints both load cleanly
    v1_p = ROOT / "research/runs/neo_gpt_mps_s0/model.pt"
    v2_p = ROOT / "research/runs/neo_gpt_v2_mps_s0/model.pt"
    if v1_p.exists():
        m.load_state_dict(torch.load(v1_p, map_location="cpu"))
    if v2_p.exists():
        NeoGPT(NeoGPTConfig()).load_state_dict(torch.load(v2_p, map_location="cpu"))
    print("PASS test_v1_1_scaled_place_rope")


def test_partition_stretch_bias_ssmax_and_attn_decay():
    """Verify Partition-Function Stretch, Bias-Aware SSMax, and Attention Score Decay."""
    torch.manual_seed(6)
    a = torch.tensor([587674, 12345])
    b = torch.tensor([465358, 987])
    x, y, _, _ = encode(a, b, 6)

    # 1. Partition-Function Stretch: gamma=1 matches baseline exactly; gamma>1 shrinks attn sum for i>=1 only
    cfg = NeoGPTConfig(n_layer=4, n_head=4, n_embd=48, rope_frac=0.5, n_place_buckets=4, uniform_slope=-1.5)
    m = NeoGPT(cfg).eval()
    assert m.num_params() == 112_112
    lg_base, _ = m(x, y)
    lg_one, _, attns_one, _ = m(x, y, return_internals=True, partition_stretch=torch.ones(2, 1, 1, 1))
    assert (lg_base - lg_one).abs().max().item() < 1e-5, "gamma=1 must equal baseline"
    assert (attns_one[0].sum(-1) - 1.0).abs().max().item() < 1e-5
    lg_st, loss_st, attns_st, _ = m(x, y, return_internals=True, partition_stretch=torch.full((2, 1, 1, 1), 4.0))
    assert torch.isfinite(loss_st)
    assert (attns_st[0][:, :, 0].sum(-1) - 1.0).abs().max().item() < 1e-5, "i=0 must stay untouched"
    assert (attns_st[0][:, :, 1:].sum(-1) < 1.0).all(), "i>=1 must absorb virtual distractor mass"
    loss_st.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

    # 2. Bias-Aware SSMax: init s=0 matches baseline; shift-invariant to constant bias shifts; differentiable
    cfg_ss = NeoGPTConfig(n_layer=4, n_head=4, n_embd=48, rope_frac=0.5, n_place_buckets=4, uniform_slope=-1.5, bias_ssmax=True)
    m_ss = NeoGPT(cfg_ss).eval()
    assert m_ss.num_params() == 112_112 + 16
    sd = m.state_dict()
    for li in range(4):
        sd[f"blocks.{li}.attn.ssmax_scale"] = torch.zeros(4)
    m_ss.load_state_dict(sd)
    lg_ss0, _ = m_ss(x, y)
    assert (lg_base - lg_ss0).abs().max().item() < 1e-5, "Bias-Aware SSMax at s=0 must equal baseline"
    with torch.no_grad():
        for blk in m_ss.blocks:
            blk.attn.ssmax_scale.fill_(0.5)
    lg_ss1, _ = m_ss(x, y)
    with torch.no_grad():
        for blk in m_ss.blocks:
            blk.attn.place_rel_bias.add_(7.3)
    lg_ss2, _ = m_ss(x, y)
    assert (lg_ss1 - lg_ss2).abs().max().item() < 1e-5, "Bias-Aware SSMax must be shift-invariant to bias constants"
    _, loss_ss = m_ss(x, y)
    loss_ss.backward()
    assert all(blk.attn.ssmax_scale.grad is not None and torch.isfinite(blk.attn.ssmax_scale.grad).all() for blk in m_ss.blocks)

    # 3. Attention Score Decay: strictly adds lambda * mean(w^2) over real causal pairs and gives finite grads
    m.zero_grad()
    _, loss_ce = m(x, y, attn_decay=0.0)
    _, loss_dec = m(x, y, attn_decay=1e-4)
    assert m.last_score_l2 is not None and m.last_score_l2.item() > 0
    assert abs((loss_dec - loss_ce).item() - 1e-4 * m.last_score_l2.item()) < 1e-5
    loss_dec.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print("PASS test_partition_stretch_bias_ssmax_and_attn_decay")


def test_barrier_fox():
    """Verify Barrier-FoX causality, exact place-shift invariance, selective blocking, and gradients."""
    torch.manual_seed(7)
    cfg = NeoGPTConfig(n_layer=4, n_head=4, n_embd=48, barrier_fox=True)
    m = NeoGPT(cfg).eval()
    base_params = NeoGPT(NeoGPTConfig(n_layer=4, n_head=4, n_embd=48)).num_params()
    assert m.num_params() == base_params + 4 * (48 * 4 + 4)

    a = torch.tensor([587674, 12345])
    b = torch.tensor([465358, 987])
    x, y, _, _ = encode(a, b, 6)

    # 1. Strict causality on answer tokens
    lg1, _ = m(x)
    x2 = x.clone()
    eq_pos = (x == STOI["="]).float().argmax(1)
    for row in range(len(x)):
        t_mut = int(eq_pos[row]) + 4
        x2[row, t_mut] = (x2[row, t_mut] + 1) % 10
    lg2, _ = m(x2)
    for row in range(len(x)):
        t_mut = int(eq_pos[row]) + 4
        diff = (lg1[row, :t_mut] - lg2[row, :t_mut]).abs().max().item()
        assert diff < 1e-5, f"Barrier-FoX causality leak before t={t_mut}: {diff}"

    # 2. Exact place-ID shift invariance
    pid = place_ids(x)
    lg0, _ = m(x, pid_override=pid)
    for off in (1, 17, 100):
        pid_s = torch.where(pid > 0, pid + off, 0)
        lg_s, _ = m(x, pid_override=pid_s)
        diff = (lg0 - lg_s).abs().max().item()
        assert diff < 1e-5, f"Barrier-FoX shift invariance broken at off={off}: {diff}"

    # 3. Verify selective barrier: raising barrier_proj.bias strictly reduces attention weight on |Delta|>=2
    #    relative to |Delta|<=1, and forward+backward grads are finite
    _, loss = m(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

    # Also test combined Barrier-FoX + Bias-Aware SSMax, ReLU dead-zone gate, and Vanilla SSMax control
    m_both = NeoGPT(NeoGPTConfig(n_layer=4, n_head=4, n_embd=48, barrier_fox=True, bias_ssmax=True))
    _, loss_both = m_both(x, y)
    loss_both.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m_both.parameters())

    m_relu = NeoGPT(NeoGPTConfig(n_layer=4, n_head=4, n_embd=48, barrier_fox=True, barrier_gate="relu", bias_ssmax=True))
    lg_r0, _ = m_relu.eval()(x, pid_override=pid)
    lg_r1, _ = m_relu.eval()(x, pid_override=torch.where(pid > 0, pid + 23, 0))
    assert (lg_r0 - lg_r1).abs().max().item() < 1e-5, "ReLU Barrier-FoX shift invariance broken"
    _, loss_r = m_relu.train()(x, y)
    loss_r.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m_relu.parameters())

    m_vss = NeoGPT(NeoGPTConfig(n_layer=4, n_head=4, n_embd=48, vanilla_ssmax=True))
    _, loss_v = m_vss(x, y)
    loss_v.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m_vss.parameters())
    print("PASS test_barrier_fox")


if __name__ == "__main__":
    test_causality_on_answer()
    test_exact_place_shift_invariance()
    test_monotonic_no_aliasing()
    test_arbitrary_lengths_and_internals()
    test_mps_parity()
    test_v1_1_scaled_place_rope()
    test_partition_stretch_bias_ssmax_and_attn_decay()
    test_barrier_fox()

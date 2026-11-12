"""
Tests for model/memory_v2.py.  Run:  python tests/test_memory_v2.py   (CPU, ~15 s)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import torch
import torch.nn.functional as F

from model.memory_v2 import MemoryV2, delta_step, shift_values, future_values
from model.titans_memory import TitansMemory
from model.transformer import SLM
from tests.test_model import tiny_cfg, influence

D = 32


def v2_cfg(variant, **mem):
    cfg = tiny_cfg(d_model=D, n_heads=4, chunk_size=8, attn_window=4)
    cfg.memory = {"variant": variant, "version": 2, "insert_after_layer": 1,
                  "n_heads": 4, "d_k": 8, "d_v": 8, "sub_chunk": 4, **mem}
    return cfg


def test_delta_step_matches_sequential():
    torch.manual_seed(0)
    B, H, L, dk, dv = 2, 3, 7, 8, 5
    M0 = torch.randn(B, H, dk, dv)
    k = F.normalize(torch.randn(B, H, L, dk), dim=-1)
    v = torch.randn(B, H, L, dv)
    beta = torch.rand(B, H, L)
    M = M0.clone()
    for t in range(L):                                    # textbook per-token delta rule
        kt, vt = k[:, :, t:t + 1], v[:, :, t:t + 1]
        M = M + beta[:, :, t, None, None] * kt.transpose(-1, -2) @ (vt - kt @ M)
    dM = delta_step(M0, k, v - k @ M0, beta)
    assert torch.allclose(M0 + dM, M, atol=1e-5), (M0 + dM - M).abs().max()
    print("ok  parallel delta step == sequential per-token delta rule")


def test_shift_values():
    v = torch.arange(6.).view(1, 1, 6, 1)
    tgt, valid = shift_values(v, 2)
    assert tgt.flatten().tolist() == [1.5, 2.5, 3.5, 4.5, 5.0, 5.0]
    assert valid.flatten().tolist() == [1, 1, 1, 1, 1, 0]
    print("ok  values = mean of the next s tokens, last token masked")


def test_future_values_keep_order():
    torch.manual_seed(0)
    H, T, dv, s = 2, 6, 4, 3
    v = torch.randn(1, H, T, dv)
    v = v * torch.rsqrt((v ** 2).mean(-1, keepdim=True))
    mean_taps = torch.full((H, dv, s), 1.0 / s)
    tgt, valid = future_values(v, mean_taps)
    ref, _ = shift_values(v, s)
    ref = ref * torch.rsqrt((ref ** 2).mean(-1, keepdim=True))
    assert torch.allclose(tgt[:, :, :T - 1], ref[:, :, :T - 1], atol=1e-5), "init should be the (normalised) mean"
    assert valid.flatten().tolist() == [1, 1, 1, 1, 1, 0]
    taps = torch.randn(H, dv, s)                       # learned, unequal weights
    swapped = v.clone(); swapped[:, :, [1, 2]] = v[:, :, [2, 1]]
    a, _ = future_values(v, taps); b, _ = future_values(swapped, taps)
    assert not torch.allclose(a[:, :, 0], b[:, :, 0]), "order of the following tokens is lost"
    m1, _ = shift_values(v, s); m2, _ = shift_values(swapped, s)
    assert torch.allclose(m1[:, :, 0], m2[:, :, 0]), "plain mean should be order-blind (sanity)"
    print("ok  learned value taps start as the mean but can tell 'v1 v2' from 'v2 v1'")


def test_saturated_gates_stay_finite():
    """C's NaN on Kaggle: huge gate inputs must not produce inf/NaN (forget = -softplus)."""
    for variant in ("B", "C"):
        torch.manual_seed(0)
        model = SLM(v2_cfg(variant))
        torch.nn.init.normal_(model.memory.W_O.weight, std=0.1)
        mem = model.memory
        with torch.no_grad():
            if variant == "C":
                for lin in (mem.f, mem.u, mem.w_r):
                    lin.bias.fill_(60.0)                   # sigmoid == 1.0 exactly in fp32
            else:
                mem.forget_raw.fill_(60.0)
        x, y = torch.randint(0, 64, (2, 32)), torch.randint(0, 64, (2, 32))
        loss = model(x * 1, y)[1]
        loss.backward()
        assert torch.isfinite(loss)
        for n, p in model.named_parameters():
            assert p.grad is None or torch.isfinite(p.grad).all(), (variant, n)
    print("ok  saturated gates (sigmoid = 1.0): loss and every gradient stay finite")


def test_registry():
    cfg = tiny_cfg(d_model=D, n_heads=4)
    for variant in ("B", "C"):
        cfg.memory = {"variant": variant, "insert_after_layer": 1, "n_heads": 4, "d_k": 8, "d_v": 8}
        assert isinstance(SLM(cfg).memory, TitansMemory)              # default stays v1
        assert isinstance(SLM(v2_cfg(variant)).memory, MemoryV2)
    print("ok  version 1 stays the default; version 2 builds MemoryV2")


def test_causal():
    for variant in ("B", "C"):
        torch.manual_seed(0)
        model = SLM(v2_cfg(variant)).eval()
        torch.nn.init.normal_(model.memory.W_O.weight, std=0.5)
        for src in (0, 3, 7, 8, 13, 20):
            changed = influence(model, 40, src)
            assert min(changed) == src, f"{variant} leaks the future from {src}: {changed}"
        assert max(influence(model, 40, 2)) > 2 + 2 * 3, "memory carries nothing forward"
    print("ok  B and C v2 are causal (also across chunk borders) and carry information forward")


def test_step0_equals_A():
    torch.manual_seed(3)
    a = SLM(tiny_cfg(d_model=D, n_heads=4, chunk_size=8, attn_window=4)).eval()
    for variant in ("B", "C"):
        m = SLM(v2_cfg(variant)).eval()
        m.load_state_dict(a.state_dict(), strict=False)
        x = torch.randint(0, 64, (2, 40))
        with torch.no_grad():
            assert torch.allclose(a(x)[0], m(x)[0], atol=1e-6), variant
    print("ok  B and C v2 equal A at step 0 (W_O zero-init)")


def test_recall_capacity():
    """Store 16 random key->value pairs in one head (dk=dv=32, under capacity), then recall.
    Uses the module's own write() so it covers sub-steps, shift and decay too."""
    torch.manual_seed(0)
    cfg = tiny_cfg(d_model=32, n_heads=4, chunk_size=48)
    cfg.memory = {"variant": "B", "version": 2, "n_heads": 1, "d_k": 32, "d_v": 32,
                  "conv": 1, "value_shift": 1, "sub_chunk": 16, "theta_init": 0.999,
                  "momentum_init": 1e-4, "decay_init": 1e-6}
    m = MemoryV2(cfg, "B")
    with torch.no_grad():
        m.W_K.weight.copy_(torch.eye(32)); m.W_V.weight.copy_(torch.eye(32))
    h = torch.randn(1, 17, 32)                            # 16 pairs: key at t, value at t+1
    st = m.write(h, m.init_state(1, "cpu"))
    keys = F.normalize(h[0, :16], dim=-1)
    r = keys @ st.M[0, 0]
    cos = F.cosine_similarity(r, h[0, 1:17], dim=-1)
    # newest facts are exact; older ones fade a little where random keys overlap
    assert cos[-1] > 0.99 and cos.min() > 0.6 and cos.mean() > 0.8, cos
    print(f"ok  16 facts written in one chunk, recalled by key: mean cos {cos.mean():.2f}, "
          f"worst {cos.min():.2f}")


def test_long_run_stable():
    for variant in ("B", "C"):
        torch.manual_seed(0)
        model = SLM(v2_cfg(variant)).eval()
        torch.nn.init.normal_(model.memory.W_O.weight, std=0.1)
        with torch.no_grad():
            _, _, st = model(torch.randint(0, 64, (2, 512)), return_state=True)
        n = st.M.flatten(2).norm(dim=-1)
        assert torch.isfinite(n).all() and n.max() < 1e3, n.max()
        assert set(model.memory.stats()) >= {"M_fro", "write_rate", "read_gate", "forget_per_256"}
    print("ok  64 chunks: memory stays finite and bounded; stats() reports")


def test_gradients():
    for variant in ("B", "C"):
        torch.manual_seed(0)
        model = SLM(v2_cfg(variant))
        torch.nn.init.normal_(model.memory.W_O.weight, std=0.1)
        x, y = torch.randint(0, 64, (2, 32)), torch.randint(0, 64, (2, 32))
        model(x, y)[1].backward()
        names = ["W_K", "W_Q", "W_V", "W_O", "conv_k", "conv_q", "theta_raw", "eta_raw"]
        names += ["f", "u", "w_r"] if variant == "C" else ["forget_raw"]
        names += ["v_taps"]
        for n in names:
            p = getattr(model.memory, n)
            g = (p.weight if hasattr(p, "weight") else p).grad
            assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, (variant, n)
    print("ok  finite non-zero gradients into every memory parameter")


def test_state_across_calls():
    torch.manual_seed(0)
    model = SLM(v2_cfg("B")).eval()
    torch.nn.init.normal_(model.memory.W_O.weight, std=0.3)
    x = torch.randint(0, 64, (1, 32))
    with torch.no_grad():
        _, _, st = model(x[:, :16], return_state=True)
        second, _ = model(x[:, 16:], memory_state=st)
        fresh, _ = model(x[:, 16:])
    assert not torch.allclose(second, fresh)
    print("ok  memory state (incl. conv tail) carries across calls")


def test_detach_chunks():
    model = SLM(v2_cfg("C", detach_chunks=True))
    x, y = torch.randint(0, 64, (2, 32)), torch.randint(0, 64, (2, 32))
    model(x, y)[1].backward()
    print("ok  detach_chunks=True trains")


if __name__ == "__main__":
    test_delta_step_matches_sequential()
    test_shift_values()
    test_future_values_keep_order()
    test_saturated_gates_stay_finite()
    test_registry()
    test_causal()
    test_step0_equals_A()
    test_recall_capacity()
    test_long_run_stable()
    test_gradients()
    test_state_across_calls()
    test_detach_chunks()
    print("\nall memory v2 tests passed")

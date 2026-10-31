"""
Unit tests for model/titans_memory.py (spec section 8).
Run from the repo root:  python -m pytest tests/test_titans_memory.py -q
CPU is enough.
"""
import torch
import torch.nn.functional as F
from config import ModelConfig
from model.memory import NullMemory
from model.titans_memory import TitansMemory, MemoryState, delta_write, retrieve

torch.manual_seed(0)
B, C, D = 2, 8, 64


def make(variant="B", **mem):
    cfg = ModelConfig(d_model=D, n_heads=4,
                      memory={"variant": variant, "n_heads": 4, "d_k": 16, "d_v": 16, **mem})
    return TitansMemory(cfg, variant)


def randomize_out(m):
    torch.nn.init.normal_(m.W_O.weight, std=0.1)   # undo zero-init for tests that need output


def run_chunks(m, hs):
    st = m.init_state(hs[0].shape[0], hs[0].device)
    outs, states = [], []
    for h in hs:
        outs.append(m.read(h, st))
        st = m.write(h, st)
        states.append(st)
    return outs, states


# 1. shapes -----------------------------------------------------------------
def test_shapes():
    for v in ("B", "C"):
        m = make(v)
        st = m.init_state(B, "cpu")
        h = torch.randn(B, C, D)
        assert m.read(h, st).shape == (B, C, D)
        st2 = m.write(h, st)
        assert st2.M.shape == st.M.shape == (B, 4, 16, 16)


# 2. recall -----------------------------------------------------------------
def test_recall():
    k = F.normalize(torch.randn(1, 1, 1, 16), dim=-1)
    v = torch.randn(1, 1, 1, 16)
    M = torch.zeros(1, 1, 16, 16)
    M, _, _ = delta_write(M, k, v, torch.ones(1, 1, 1), torch.ones(1), decay=0.0)
    r = retrieve(k, M)
    assert F.cosine_similarity(r.flatten(), v.flatten(), dim=0) > 0.95


# 3. overwrite --------------------------------------------------------------
def test_overwrite():
    k = F.normalize(torch.randn(1, 1, 1, 16), dim=-1)
    v1, v2 = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    M = torch.zeros(1, 1, 16, 16)
    one, th = torch.ones(1, 1, 1), torch.ones(1)
    M, _, _ = delta_write(M, k, v1, one, th, 0.0)
    for _ in range(3):
        M, _, _ = delta_write(M, k, v2, one, th, 0.0)
    r = retrieve(k, M).flatten()
    assert torch.allclose(r, v2.flatten(), atol=1e-5)          # new value, not v1
    assert F.cosine_similarity(r, v1.flatten(), dim=0) < 0.95


# 4. causality ---------------------------------------------------------------
def test_causality():
    for v in ("B", "C"):
        m = make(v)
        randomize_out(m)
        hs = [torch.randn(B, C, D) for _ in range(4)]
        outs, states = run_chunks(m, hs)

        # a) perturbing chunk 2 leaves chunks 0,1 outputs and the state they produced untouched
        hs2 = [h.clone() for h in hs]
        hs2[2] = hs2[2] + torch.randn_like(hs2[2])
        outs2, states2 = run_chunks(m, hs2)
        for t in (0, 1):
            assert torch.allclose(outs[t], outs2[t])
            assert torch.allclose(states[t].M, states2[t].M)

        # b) no leakage inside a chunk: the read at token i ignores the other tokens
        hs3 = [h.clone() for h in hs]
        hs3[1][:, 5] += torch.randn(B, D)
        outs3, _ = run_chunks(m, hs3)
        keep = [i for i in range(C) if i != 5]
        assert torch.allclose(outs[1][:, keep], outs3[1][:, keep], atol=1e-6)


# 5. zero-init: B and C equal A at step 0 -------------------------------------
def test_zero_init():
    h = torch.randn(B, C, D)
    null = NullMemory()
    for v in ("B", "C"):
        m = make(v)
        st = m.init_state(B, "cpu")
        st = m.write(h, st)                                    # non-empty memory
        assert torch.equal(m.read(h, st), null.read(h, None))  # zeros


# 6. gate sanity (variant C) ---------------------------------------------------
def test_gate_sanity():
    m = make("C", decay=0.0)
    torch.nn.init.zeros_(m.u.weight)
    torch.nn.init.zeros_(m.u.bias)                             # gate driven by surprise only
    with torch.no_grad():
        m.theta_raw.fill_(10.0)                                # theta ~ 2, fastest stable step
    h = torch.randn(B, C, D)
    st = m.init_state(B, "cpu")
    st = m.write(h, st)
    first = m.stats()["write_rate"]
    for _ in range(100):
        st = m.write(h, st)                                    # same pair again and again
    repeated = m.stats()["write_rate"]
    st_new = m.write(torch.randn(B, C, D), st)                 # novel pair
    novel = m.stats()["write_rate"]
    assert first > 0.7 and repeated < 0.35 and novel > repeated + 0.2, (first, repeated, novel)


# 7. stability -------------------------------------------------------------------
def test_stability():
    for v in ("B", "C"):
        m = make(v)
        st = m.init_state(B, "cpu")
        for _ in range(16):
            h = (torch.randn(B, C, D) * 3).half()              # fp16 backbone input
            r = m.read(h, st)
            st = m.write(h, st)
            assert torch.isfinite(r.float()).all() and torch.isfinite(st.M).all()
        assert st.M.abs().max() < 1e3


# 8. gradients flow across chunks ------------------------------------------------
def test_gradients():
    m = make("C")
    randomize_out(m)
    h0, h1 = torch.randn(B, C, D), torch.randn(B, C, D)
    st = m.init_state(B, "cpu")
    st = m.write(h0, st)
    m.read(h1, st).sum().backward()
    for name in ("W_K", "W_V", "W_Q", "W_O", "w_r"):
        g = getattr(m, name).weight.grad
        assert g is not None and g.abs().sum() > 0, name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

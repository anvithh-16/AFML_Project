"""
Backbone tests. Run:  python tests/test_model.py
(CPU, ~10 s; no GPU, data or pytest needed.)

Person B: once your memory exists, add it to MEMORIES_TO_CHECK at the bottom;
test_memory_causality then checks your module never leaks future tokens.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig  # noqa: E402
from model.memory import MemoryModule  # noqa: E402
from model.transformer import SLM, sliding_window_mask  # noqa: E402


def tiny_cfg(**kw):
    base = dict(vocab_size=64, n_layers=2, d_model=32, n_heads=4, ffn_hidden=64,
                train_seq_len=32, attn_window=4, chunk_size=4, eval_max_seq_len=64)
    base.update(kw)
    cfg = ModelConfig(**base)
    cfg.memory = {"variant": "A", "insert_after_layer": base["n_layers"] // 2}
    return cfg


def influence(model, T, src, seed=0):
    """Which output positions change when input token `src` changes."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, model.cfg.vocab_size, (1, T), generator=g)
    b = a.clone()
    b[0, src] = (a[0, src] + 1) % model.cfg.vocab_size
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    return ((la - lb).abs().amax(-1)[0] > 1e-6).nonzero().flatten().tolist()


def test_mask():
    m = sliding_window_mask(6, 3, "cpu").int().tolist()
    assert m[5] == [0, 0, 0, 1, 1, 1] and m[0] == [1, 0, 0, 0, 0, 0], m
    print("ok  sliding-window mask")


def test_causal_and_receptive_field():
    torch.manual_seed(0)
    cfg = tiny_cfg(n_layers=2, attn_window=4)
    model = SLM(cfg).eval()
    T, src = 32, 5
    changed = influence(model, T, src)
    reach = cfg.n_layers * (cfg.attn_window - 1)          # 2 * 3 = 6
    assert min(changed) == src, f"leak into the past: {changed}"
    assert max(changed) == src + reach, f"receptive field {max(changed) - src}, expected {reach}"
    print(f"ok  causal; receptive field = L*(W-1) = {reach} tokens (full config: 8*255 = 2040)")


def test_param_count():
    model = SLM(ModelConfig())
    n, est = model.num_params(), ModelConfig().approx_params()
    assert abs(n - est) / est < 0.01, (n, est)
    assert model.lm_head.weight is model.tok_emb.weight
    print(f"ok  full config: {n / 1e6:.2f}M params, embeddings tied")


def test_loss_at_init():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = SLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.randint(0, cfg.vocab_size, (4, 32))   # not x: tied embeddings favour copying the input at init
    _, loss = model(x, y)
    expect = torch.log(torch.tensor(float(cfg.vocab_size))).item()
    assert abs(loss.item() - expect) < 0.3, (loss.item(), expect)
    print(f"ok  initial loss {loss.item():.3f} ~ ln(V) = {expect:.3f}")


def test_long_eval_length():
    cfg = tiny_cfg(eval_max_seq_len=16)
    model = SLM(cfg).eval()
    with torch.no_grad():
        logits, _ = model(torch.randint(0, 64, (1, 40)))     # longer than the RoPE table
    assert logits.shape == (1, 40, 64)
    print("ok  runs past the precomputed RoPE length")


# ---- memory hook -----------------------------------------------------------
class ToyMemory(MemoryModule):
    """Stand-in memory: running mean of written chunks, read through a linear map.
    Only for testing the hook; Person B replaces it."""

    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.calls = []

    def init_state(self, B, device, dtype):
        return {"sum": torch.zeros(B, self.proj.in_features, device=device, dtype=dtype), "n": 0}

    def read(self, h, state):
        self.calls.append(("read", h.shape[1]))
        mean = state["sum"] / max(state["n"], 1)
        return self.proj(mean)[:, None, :].expand_as(h)

    def write(self, h, state):
        self.calls.append(("write", h.shape[1]))
        return {"sum": state["sum"] + h.sum(1), "n": state["n"] + h.shape[1]}


def with_memory(cfg, mem_cls):
    model = SLM(cfg)
    model.memory = mem_cls(cfg)
    return model.eval()


def test_memory_hook_order():
    cfg = tiny_cfg(attn_window=4, chunk_size=4)
    model = with_memory(cfg, ToyMemory)
    with torch.no_grad():
        model(torch.randint(0, 64, (2, 10)))                  # 10 tokens -> chunks 4,4,2
    assert model.memory.calls == [("read", 4), ("write", 4), ("read", 4),
                                  ("write", 4), ("read", 2), ("write", 2)], model.memory.calls
    print("ok  memory called read->write per chunk, last chunk may be short")


def test_memory_causality(mem_cls=ToyMemory):
    torch.manual_seed(0)
    cfg = tiny_cfg(n_layers=2, attn_window=4, chunk_size=4)
    model = with_memory(cfg, mem_cls)
    T, src = 32, 9
    changed = influence(model, T, src)
    assert min(changed) == src, f"{mem_cls.__name__} leaks the future: {changed}"
    # memory should now carry information further than the backbone alone (src + 6)
    print(f"ok  {mem_cls.__name__}: causal; token {src} now reaches position {max(changed)} "
          f"(backbone alone: {src + 6})")


def test_state_carries_across_calls():
    cfg = tiny_cfg()
    model = with_memory(cfg, ToyMemory)
    x = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        full, _ = model(x)
        _, _, st = model(x[:, :8], return_state=True)
        second, _ = model(x[:, 8:], memory_state=st)
    # second half sees memory of the first half (not identical to full: no attention across the cut)
    fresh, _ = model(x[:, 8:])
    assert not torch.allclose(second, fresh), "state passed between calls had no effect"
    print("ok  memory_state can be carried across calls (for long eval documents)")


MEMORIES_TO_CHECK = [ToyMemory]

if __name__ == "__main__":
    test_mask()
    test_causal_and_receptive_field()
    test_param_count()
    test_loss_at_init()
    test_long_eval_length()
    test_memory_hook_order()
    for m in MEMORIES_TO_CHECK:
        test_memory_causality(m)
    test_state_carries_across_calls()
    print("\nall backbone tests passed")

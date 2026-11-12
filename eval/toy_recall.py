"""
Toy recall bench: can a memory learn to store a fact and bring it back from beyond
the attention window? A mechanism check only (tiny models, toy tokens), run BEFORE any
GPU time on real data. Not training data for the real models.

Each sequence (T tokens, all random filler except):
    n_facts facts   [FACT k1 k2 SEP v1 v2]   placed in the first part of the sequence
    one question    [QUERY k1 k2 SEP] -> v1 v2   at the very end, about one of the facts
Every fact sits further back than the backbone's whole receptive field
(layers * (window-1)), so Variant A can only guess (1/n_values per token).
Loss is on the two answer tokens only.

    python eval/toy_recall.py --variants A B1 B2 C2 --steps 1500           # CPU ok, ~10 min each
    python eval/toy_recall.py --variants A B2 C2 --device cuda               # Kaggle: ~1 min each

Variant names: A = no memory, B1/C1 = memory v1, B2/C2 = memory v2.
Half the questions have their fact copied just before the question (inside the window),
so every model can learn the format; the other half only memory can answer.
Pass: everyone solves "near"; on "far", B2/C2 clearly above chance while A and v1 stay near it.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig  # noqa: E402
from model.transformer import SLM  # noqa: E402

FACT, QUERY, SEP = 1, 2, 3


def make_batch(B, T, n_keys, n_vals, n_facts, layers, window, gen, device, p_near=0.5):
    """Returns idx (B,T), targets (B,T) with -100 except the two answer positions, and
    near (B,) bool. With prob p_near the asked fact is moved to just before the question
    (inside the attention window): every model can learn the task format from those.
    Otherwise every fact is beyond the receptive field and only memory can answer."""
    k0, v0 = 4, 4 + n_keys
    f0 = v0 + n_vals
    vocab = f0 + 64
    reach = layers * (window - 1)
    last_fact_end = T - 6 - reach - 8            # every fact strictly beyond the receptive field
    assert last_fact_end > n_facts * 8, "sequence too short for this many facts"
    idx = torch.randint(f0, vocab, (B, T), generator=gen)
    tgt = torch.full((B, T), -100)
    near = torch.rand(B, generator=gen) < p_near
    for b in range(B):
        keys = torch.randperm(n_keys * n_keys, generator=gen)[:n_facts]       # distinct 2-token keys
        vals = torch.randint(0, n_vals, (n_facts, 2), generator=gen)
        slots = torch.randperm(last_fact_end // 8, generator=gen)[:n_facts] * 8
        for (key, val, p) in zip(keys.tolist(), vals.tolist(), slots.tolist()):
            idx[b, p:p + 6] = torch.tensor([FACT, k0 + key // n_keys, k0 + key % n_keys, SEP,
                                            v0 + val[0], v0 + val[1]])
        j = int(torch.randint(0, n_facts, (1,), generator=gen))
        key, val = keys[j].item(), vals[j].tolist()
        q = T - 6
        if near[b]:                                   # copy the asked fact right before the question
            p = q - 6 - int(torch.randint(0, max(1, window - 13), (1,), generator=gen))
            idx[b, p:p + 6] = torch.tensor([FACT, k0 + key // n_keys, k0 + key % n_keys, SEP,
                                            v0 + val[0], v0 + val[1]])
        idx[b, q:q + 6] = torch.tensor([QUERY, k0 + key // n_keys, k0 + key % n_keys, SEP,
                                        v0 + val[0], v0 + val[1]])
        tgt[b, q + 3], tgt[b, q + 4] = v0 + val[0], v0 + val[1]              # predict v1 at SEP, v2 at v1
    return idx.to(device), tgt.to(device), near.to(device)


def build(variant, a):
    cfg = ModelConfig(vocab_size=4 + a.n_keys + a.n_vals + 64, n_layers=a.layers, d_model=a.d_model,
                      n_heads=4, ffn_hidden=a.d_model * 11 // 4, train_seq_len=a.T,
                      attn_window=a.window, chunk_size=a.window, eval_max_seq_len=a.T)
    mem = {"variant": variant[0], "insert_after_layer": a.layers // 2}
    if variant[0] != "A":
        mem.update(n_heads=4, d_k=a.d_model // 4, d_v=a.d_model // 4)
        if variant.endswith("2"):
            mem.update(version=2, sub_chunk=min(64, a.window))
    cfg.memory = mem
    return SLM(cfg)


def evaluate(model, a, device, batches=8):
    """Answer-token accuracy on a fixed held-out set, split into near (fact inside the
    window) and far (every fact beyond the receptive field)."""
    gen = torch.Generator().manual_seed(999)
    model.eval()
    right = {True: 0, False: 0}
    count = {True: 0, False: 0}
    with torch.no_grad():
        for _ in range(batches):
            x, y, near = make_batch(a.batch, a.T, a.n_keys, a.n_vals, a.n_facts, a.layers, a.window,
                                    gen, device)
            logits, _ = model(x)
            mask = y != -100
            ok = ((logits.argmax(-1) == y) & mask).sum(1)
            for b in range(x.shape[0]):
                right[bool(near[b])] += ok[b].item()
                count[bool(near[b])] += 2
    model.train()
    return right[True] / max(1, count[True]), right[False] / max(1, count[False])


def run(variant, a, device):
    torch.manual_seed(a.seed)
    model = build(variant, a).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(a.seed)
    t0, log = time.time(), []
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, _ = make_batch(a.batch, a.T, a.n_keys, a.n_vals, a.n_facts, a.layers, a.window, gen, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            near, far = evaluate(model, a, device)
            stats = {k: round(v, 3) for k, v in model.memory.stats().items()} if not model.memory.is_null else {}
            log.append(dict(step=step, loss=round(loss.item(), 3), near=round(near, 3), far=round(far, 3)))
            print(f"[{variant}] step {step:5d} | loss {loss.item():.3f} | near acc {near:.3f} | "
                  f"FAR acc {far:.3f} | {time.time() - t0:.0f}s {stats}", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["A", "B1", "B2", "C2"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_keys", type=int, default=16)
    ap.add_argument("--n_vals", type=int, default=16)
    ap.add_argument("--n_facts", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="write results JSON here")
    a = ap.parse_args()
    print(f"chance = {1 / a.n_vals:.3f} per answer token | far facts are > {a.layers * (a.window - 1)} "
          f"tokens back (beyond the receptive field); near facts are inside the window")
    results = {v: run(v, a, a.device) for v in a.variants}
    print("\nfinal (held-out):")
    for v, log in results.items():
        print(f"  {v:3s} near {log[-1]['near']:.3f} | FAR {log[-1]['far']:.3f}")
    if a.out:
        json.dump({"args": vars(a), "results": results}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

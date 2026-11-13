"""
Is the memory used at all? Runs each B/C checkpoint twice on the same inputs, once
normally and once with the memory's read switched off (it returns zeros, so the model
behaves like its backbone alone), and reports what changes.

    1. size     : |memory output| / |hidden state| per token (how loud the memory is),
                  |W_O| (the memory's output weights start at 0), and memory.stats()
                  (C: read gate, write rate, forget rate)
    2. PG-19    : far-back-word loss, memory ON vs OFF (pretrained checkpoints)
    3. BABILong : accuracy, log-prob of the right answer, and margin over the best wrong
                  answer, memory ON vs OFF (instruction-tuned checkpoints)

If ON == OFF the memory is not used; the gap ON - OFF is exactly what memory contributes.

    python eval/memory_usage.py --long_dir .../pg19mix --babilong_dir .../instruct_babilong \
        --tokenizer .../tokenizer.json \
        --pretrained B=<...>/variant-B_pretrain.pt C=<...>/variant-C_pretrain.pt \
        --instruct   B=<...>/variant-B_instruct.pt C=<...>/variant-C_instruct.pt
"""
import argparse
import contextlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESPONSE_TEMPLATE  # noqa: E402
from eval.babilong_eval import choice_logprob  # noqa: E402
from eval.far_recall_loss import load, score  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402


@contextlib.contextmanager
def memory_off(model):
    """Read returns zeros: the model runs as its backbone alone (writes still happen, unused)."""
    orig = model.memory.read
    model.memory.read = lambda h, state: torch.zeros_like(h)
    try:
        yield
    finally:
        model.memory.read = orig


@contextlib.contextmanager
def measure_size(model, sink):
    """Records |read output| / |h| per token while the model runs normally."""
    orig = model.memory.read

    def read(h, state):
        r = orig(h, state)
        sink.append(float((r.float().norm(dim=-1) / (h.float().norm(dim=-1) + 1e-6)).mean()))
        return r
    model.memory.read = read
    try:
        yield
    finally:
        model.memory.read = orig


def load_books(long_dir, n, max_tokens):
    data = np.memmap(os.path.join(long_dir, "test_books.bin"), dtype=np.uint16, mode="r")
    offs = np.load(os.path.join(long_dir, "test_offsets.npy"))
    return [np.asarray(data[offs[i]:min(offs[i + 1], offs[i] + max_tokens)]) for i in range(min(n, len(offs) - 1))]


def babilong_scores(model, test, cand, idx, device):
    out = []
    for i in idx:
        r = test["raw"][i]
        prompt = test["ids"][i][:test["prompt_len"][i]]
        lps = [choice_logprob(model, prompt, c, device) for c in cand[r["task"]]]
        k = test["choices"][r["task"]].index(r["response"])
        wrong = max(lp for j, lp in enumerate(lps) if j != k)
        out.append((int(lps[k] > wrong), lps[k], lps[k] - wrong))
    a = np.array(out)
    return {"acc": float(a[:, 0].mean()), "logp_right": float(a[:, 1].mean()), "margin": float(a[:, 2].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long_dir", default=None, help="needed only with --pretrained")
    ap.add_argument("--babilong_dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pretrained", nargs="*", default=[], help="NAME=path (B, C); empty = skip the PG-19 part")
    ap.add_argument("--instruct", nargs="+", required=True, help="NAME=path (B, C)")
    ap.add_argument("--books", type=int, default=30)
    ap.add_argument("--book_tokens", type=int, default=16384)
    ap.add_argument("--per_cell", type=int, default=60, help="BABILong questions per (task, length)")
    ap.add_argument("--seg_len", type=int, default=2048)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"pretrained": {}, "instruct": {}}

    # ---- 1 + 2: size and PG-19 far-back-word loss, pretrained -------------------
    books = load_books(a.long_dir, a.books, a.book_tokens) if a.pretrained else []
    if a.pretrained:
        print(f"PG-19: {len(books)} test books x <= {a.book_tokens} tokens\n")
    for spec in a.pretrained:
        name, path = spec.split("=", 1)
        model = load(path, a.device)
        sizes = []
        with measure_size(model, sizes):
            on = score(model, books, a.seg_len, a.window, a.device)
        stats = model.memory.stats()
        with memory_off(model):
            off = score(model, books, a.seg_len, a.window, a.device)
        wo = float(model.memory.W_O.weight.float().norm())
        res["pretrained"][name] = {"read_over_h": float(np.mean(sizes)), "W_O_norm": wo, "stats": stats,
                                   "on": on, "off": off}
        print(f"[{name} pretrained] memory output = {np.mean(sizes):.4f} x |h| | |W_O| = {wo:.3f} | "
              + ", ".join(f"{k} {v:.3f}" for k, v in stats.items()))
        for b in ("all", "near", "far", "new"):
            print(f"    {b:4s} loss  ON {on[b]['loss']:.4f}  OFF {off[b]['loss']:.4f}  "
                  f"memory helps by {off[b]['loss'] - on[b]['loss']:+.4f}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- 3: BABILong ON vs OFF, instruction-tuned ----------------------------------
    tok = Tokenizer.from_file(a.tokenizer)
    test = torch.load(os.path.join(a.babilong_dir, "test.pt"))
    cand = {t: [tok.encode(RESPONSE_TEMPLATE.format(response=c)).ids for c in cs] for t, cs in test["choices"].items()}
    cells = defaultdict(list)
    for i, r in enumerate(test["raw"]):
        if len(cells[(r["task"], r["length"])]) < a.per_cell:
            cells[(r["task"], r["length"])].append(i)
    print(f"\nBABILong: {a.per_cell} questions per cell, {len(cells)} cells")
    for spec in a.instruct:
        name, path = spec.split("=", 1)
        model = load(path, a.device)
        res["instruct"][name] = {}
        print(f"[{name} instruct]  cell       acc ON/OFF     log p(right) ON/OFF     margin ON/OFF")
        for (task, length), idx in sorted(cells.items()):
            on = babilong_scores(model, test, cand, idx, a.device)
            with memory_off(model):
                off = babilong_scores(model, test, cand, idx, a.device)
            res["instruct"][name][f"{task}/{length}"] = {"on": on, "off": off}
            print(f"    {task}/{length:3s}  {on['acc']:.3f} / {off['acc']:.3f}    {on['logp_right']:7.3f} / "
                  f"{off['logp_right']:7.3f}    {on['margin']:+.3f} / {off['margin']:+.3f}", flush=True)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\nHow to read it: ON = normal model, OFF = memory output zeroed. 'memory helps by' > 0 and ON above OFF "
          "mean the memory carries something; ON == OFF means the model ignores its memory.")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

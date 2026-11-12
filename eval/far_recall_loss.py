"""
Far-back-word loss on PG-19 test books: does memory help predict words whose last
appearance is beyond the attention window? Natural text, no synthetic data.

Each test book (first --max_book_tokens) is read in segments of --seg_len tokens with the
memory state carried from segment to segment (A has none). Every predicted token is put in
one bucket by where the SAME token id last appeared before it:
    near : 1..256 tokens back   (inside the window: attention can see it)
    far  : more than 256 back   (only memory can carry it)  <- the number that matters
    new  : never before in the book
Only positions with a full 256-token window inside their segment are scored (plus the
book's first segment), so A is never penalised for the cut between segments.

    python eval/far_recall_loss.py --data_dir <AFML_Project_PG19Mix>/artifacts/pg19mix \
        --ckpt A=<...>/variant-A_pretrain.pt B=<...>/variant-B_pretrain.pt C=<...>/variant-C_pretrain.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig  # noqa: E402
from model.transformer import SLM  # noqa: E402

BUCKETS = ("near", "far", "new")


def buckets_for(book, window):
    """bucket id (0 near, 1 far, 2 new) for predicting book[i], i >= 1."""
    last, out = {}, np.full(len(book), 2, np.int8)
    for i, t in enumerate(book.tolist()):
        j = last.get(t)
        if j is not None:
            out[i] = 0 if i - j <= window else 1
        last[t] = i
    return out


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SLM(ModelConfig(**ck["model_config"]))
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


@torch.no_grad()
def score(model, books, seg_len, window, device):
    use_amp = str(device).startswith("cuda")
    tot = {b: 0.0 for b in BUCKETS}
    cnt = {b: 0 for b in BUCKETS}
    for book in books:
        bk = buckets_for(book, window)
        state = None
        for s in range(0, len(book) - 1, seg_len):
            x = torch.from_numpy(book[s:s + seg_len].astype(np.int64))[None].to(device)
            y = torch.from_numpy(book[s + 1:s + seg_len + 1].astype(np.int64))[None].to(device)
            x = x[:, :y.shape[1]]
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits, _, state = model(x, memory_state=state, return_state=True)
            nll = F.cross_entropy(logits.float()[0], y[0], reduction="none").cpu().numpy()
            first = 0 if s == 0 else window                         # full window inside the segment
            b = bk[s + 1:s + 1 + len(nll)]
            for k, name in enumerate(BUCKETS):
                sel = (b[first:] == k)
                tot[name] += float(nll[first:][sel].sum())
                cnt[name] += int(sel.sum())
    out = {name: {"loss": tot[name] / max(cnt[name], 1), "tokens": cnt[name]} for name in BUCKETS}
    n = sum(cnt.values())
    out["all"] = {"loss": sum(tot.values()) / max(n, 1), "tokens": n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="output of data/prepare_pg19.py")
    ap.add_argument("--ckpt", nargs="+", required=True, help="NAME=path, e.g. A=variant-A.pt")
    ap.add_argument("--max_books", type=int, default=100)
    ap.add_argument("--max_book_tokens", type=int, default=16384)
    ap.add_argument("--seg_len", type=int, default=2048)
    ap.add_argument("--window", type=int, default=256, help="the attention window")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = np.memmap(os.path.join(a.data_dir, "test_books.bin"), dtype=np.uint16, mode="r")
    offs = np.load(os.path.join(a.data_dir, "test_offsets.npy"))
    books = [np.asarray(data[offs[i]:min(offs[i + 1], offs[i] + a.max_book_tokens)])
             for i in range(min(a.max_books, len(offs) - 1))]
    print(f"{len(books)} PG-19 test books, {sum(map(len, books)):,} tokens, segments of {a.seg_len}, "
          f"window {a.window}")
    results = {}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        results[name] = score(load(path, a.device), books, a.seg_len, a.window, a.device)
        r = results[name]
        print(f"{name}: all {r['all']['loss']:.4f} | near {r['near']['loss']:.4f} | "
              f"FAR {r['far']['loss']:.4f} ({r['far']['tokens']:,} tok) | new {r['new']['loss']:.4f}")
    base = next(iter(results))
    print(f"\nchange vs {base} (negative = better):")
    for name, r in results.items():
        print(f"  {name}: " + " | ".join(f"{b} {r[b]['loss'] - results[base][b]['loss']:+.4f}"
                                         for b in ("all", "near", "far", "new")))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

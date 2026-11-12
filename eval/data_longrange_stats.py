"""
How much long-range information is in the training data? Measured, not assumed.
Run after data/prepare_pg19.py and data/prepare_babilong.py (CPU, a few minutes).

For a sample of training rows, every token is put in one bucket by where the SAME token id
last appeared earlier in the same document (the row; a FineWeb row is cut at <|endoftext|>):
    near : within the attention window (<= 256 back)   -> attention can use it
    far  : beyond the window                           -> only memory can use it
    new  : not earlier in the document
"far" is the share of tokens where memory COULD help. Content words (tokens outside the
most frequent 1,000, i.e. names, rare nouns) are reported separately, because "the" and "."
repeat everywhere and say little about long-range dependencies.

Compared: v2 PG-19 rows, v2 FineWeb rows (2,048), v1 FineWeb windows (1,024, as v1 trained).
BABILong: how far back the answer word was last mentioned before the question.

    python eval/data_longrange_stats.py --v2_dir .../pg19mix --v1_dir .../artifacts/data \
        --babilong_dir .../instruct_babilong --tokenizer .../tokenizer.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EOT = 0
EDGES = (256, 512, 1024, 2048)


def scan(rows, window, common):
    """Counts over a list of 1-D token arrays (each one document piece)."""
    c = Counter()
    for row in rows:
        last = {}
        doc_start = 0
        for i, t in enumerate(row.tolist()):
            if t == EOT:                         # document boundary inside a row
                last, doc_start = {}, i + 1
                continue
            content = t not in common
            c["tok"] += 1
            c["content"] += content
            if i - doc_start >= window:
                c["has_history_beyond_window"] += 1
            j = last.get(t)
            if j is None:
                c["new"] += 1
            elif i - j <= window:
                c["near"] += 1
            else:
                c["far"] += 1
                c["far_content"] += content
                d = i - j
                for lo, hi in zip(EDGES, EDGES[1:]):
                    if lo < d <= hi:
                        c[f"far_{lo + 1}_{hi}"] += 1
            last[t] = i
    return c


def report(name, c):
    t = max(c["tok"], 1)
    out = {
        "tokens_scanned": c["tok"],
        "far_pct": 100 * c["far"] / t,
        "near_pct": 100 * c["near"] / t,
        "new_pct": 100 * c["new"] / t,
        "far_content_pct_of_content": 100 * c["far_content"] / max(c["content"], 1),
        "positions_with_history_beyond_window_pct": 100 * c["has_history_beyond_window"] / t,
    }
    for lo, hi in zip(EDGES, EDGES[1:]):
        out[f"far_{lo + 1}_{hi}_pct"] = 100 * c[f"far_{lo + 1}_{hi}"] / t
    print(f"{name:28s} far {out['far_pct']:5.1f}% | content words far {out['far_content_pct_of_content']:5.1f}% | "
          f"near {out['near_pct']:5.1f}% | new {out['new_pct']:5.1f}% | "
          f"positions with >{EDGES[0]} tokens of same-doc history {out['positions_with_history_beyond_window_pct']:5.1f}%")
    print(" " * 29 + "far by distance: " + " | ".join(
        f"{lo + 1}-{hi}: {out[f'far_{lo + 1}_{hi}_pct']:.1f}%" for lo, hi in zip(EDGES, EDGES[1:])))
    return out


def babilong_distances(data_dir, tokenizer, window, per_cell=200):
    """BABILong train examples: how many tokens lie between the last mention of the answer
    word in the story and the answer itself (the distance memory has to bridge)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer)
    d = torch.load(os.path.join(data_dir, "train.pt"))
    cells = {}
    for r in d["raw"]:
        cat = str(r.get("category", ""))
        if cat.startswith("babilong") and len(cells.setdefault(cat, [])) < per_cell:
            ctx, ans = r["context"], r["response"].strip()
            k = ctx.lower().rfind(ans.lower(), 0, ctx.rfind("Question:"))
            cells[cat].append(None if k < 0 else len(tok.encode(ctx[k:]).ids))
    res = {}
    for cat, ds in sorted(cells.items()):
        found = [x for x in ds if x is not None]
        res[cat] = {"examples": len(ds), "answer_in_story_pct": 100 * len(found) / max(len(ds), 1),
                    "median_distance": float(np.median(found)) if found else None,
                    "beyond_window_pct": 100 * sum(x > window for x in found) / max(len(found), 1)}
        print(f"{cat:22s} answer word last in the story a median {res[cat]['median_distance']:>6} tokens "
              f"before the answer | beyond the {window}-token window: {res[cat]['beyond_window_pct']:3.0f}% "
              f"(sample {len(ds)})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_dir", required=True)
    ap.add_argument("--v1_dir", default=None)
    ap.add_argument("--babilong_dir", default=None)
    ap.add_argument("--tokenizer", default=None, help="needed for the BABILong distances")
    ap.add_argument("--rows", type=int, default=3000, help="rows sampled per source")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--common", type=int, default=1000, help="most frequent token ids treated as function words")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    meta = json.load(open(os.path.join(a.v2_dir, "meta.json")))
    R = meta["row_len"]
    train = np.memmap(os.path.join(a.v2_dir, "train.bin"), dtype=np.uint16, mode="r").reshape(-1, R)
    src = np.load(os.path.join(a.v2_dir, "train_source.npy"))
    pg_idx = rng.choice(np.nonzero(src == 0)[0], size=min(a.rows, int((src == 0).sum())), replace=False)
    fw_idx = rng.choice(np.nonzero(src == 1)[0], size=min(a.rows, int((src == 1).sum())), replace=False)
    pg_rows = [np.asarray(train[i]) for i in pg_idx]
    fw_rows = [np.asarray(train[i]) for i in fw_idx]
    freq = Counter(np.concatenate(pg_rows + fw_rows).tolist())
    common = {t for t, _ in freq.most_common(a.common)}

    print(f"window {a.window} | {len(pg_rows)} PG-19 rows, {len(fw_rows)} FineWeb rows of {R} tokens | "
          f"content words = all but the {a.common} most frequent token ids\n")
    out = {"v2_pg19_2048": report("v2 PG-19 rows (2,048)", scan(pg_rows, a.window, common)),
           "v2_fineweb_2048": report("v2 FineWeb rows (2,048)", scan(fw_rows, a.window, common))}
    if a.v1_dir:
        v1 = np.memmap(os.path.join(a.v1_dir, "train.bin"), dtype=np.uint16, mode="r")
        starts = rng.integers(0, len(v1) - 1025, size=a.rows)
        out["v1_fineweb_1024"] = report("v1 FineWeb windows (1,024)",
                                        scan([np.asarray(v1[s:s + 1025]) for s in starts], a.window, common))
    if a.babilong_dir and a.tokenizer:
        print()
        out["babilong"] = babilong_distances(a.babilong_dir, a.tokenizer, a.window)
    m = meta["rows"]
    print(f"\nv2 mix: {m['train_pg19']:,} PG-19 + {m['train_fineweb']:,} FineWeb rows = {meta['tokens']['train'] / 1e6:.1f}M tokens")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

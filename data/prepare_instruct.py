"""
Prepare Dolly-15k for instruction tuning, once, for all three variants.

Kaggle (Internet ON, CPU is enough, ~1 min):
    python data/prepare_instruct.py --tokenizer <AFML_Project_Data>/artifacts/tokenizer/tokenizer.json \
        --out_dir /kaggle/working/artifacts/instruct
Local test without internet:
    python data/prepare_instruct.py --local_jsonl dolly.jsonl --tokenizer ... --out_dir ...

Output (save as a Kaggle dataset, like the pretraining data):
    train.pt, val.pt   {"ids": [[...]], "prompt_len": [...], "raw": [{instruction, context, response, category}]}
    meta.json          counts, dropped examples, token stats, tokenizer_sha, split seed

The split is a seeded shuffle, so it is identical on every machine; val holds
--val_size examples that no variant ever trains on. Examples longer than
--max_len tokens (the training sequence length) are dropped, not truncated,
so no response is ever cut off mid-answer.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig  # noqa: E402
from data.instruct_format import encode_example  # noqa: E402
from data.prepare_pretrain import file_sha  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

DOLLY = "databricks/databricks-dolly-15k"


def load_rows(args):
    if args.local_jsonl:
        with open(args.local_jsonl, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    from datasets import load_dataset
    return list(load_dataset(DOLLY, split="train"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--out_dir", default="artifacts/instruct")
    ap.add_argument("--local_jsonl", default=None)
    ap.add_argument("--val_size", type=int, default=500)
    ap.add_argument("--max_len", type=int, default=ModelConfig().train_seq_len)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    rows = load_rows(args)
    order = np.random.default_rng(args.seed).permutation(len(rows))
    splits = {"val": order[:args.val_size], "train": order[args.val_size:]}

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {"source": args.local_jsonl or DOLLY, "seed": args.seed, "max_len": args.max_len,
            "tokenizer_sha": file_sha(args.tokenizer), "splits": {}}
    for name, idx in splits.items():
        out = {"ids": [], "prompt_len": [], "raw": []}
        dropped_long = dropped_empty = 0
        for i in idx:
            r = rows[int(i)]
            if not str(r.get("response", "")).strip():
                dropped_empty += 1
                continue
            ids, plen = encode_example(tok, r["instruction"], r.get("context", ""), r["response"])
            if len(ids) > args.max_len:
                dropped_long += 1
                continue
            out["ids"].append(ids)
            out["prompt_len"].append(plen)
            out["raw"].append({k: r.get(k, "") for k in ("instruction", "context", "response", "category")})
        torch.save(out, os.path.join(args.out_dir, f"{name}.pt"))
        lens = np.array([len(x) for x in out["ids"]])
        resp = int(sum(len(x) - p for x, p in zip(out["ids"], out["prompt_len"])))
        meta["splits"][name] = {"examples": len(out["ids"]), "dropped_too_long": dropped_long,
                                "dropped_empty": dropped_empty, "total_tokens": int(lens.sum()),
                                "response_tokens": resp, "mean_len": round(float(lens.mean()), 1),
                                "p95_len": int(np.percentile(lens, 95)), "max_len": int(lens.max())}
        print(f"{name}: {len(out['ids']):,} examples kept, {dropped_long} too long, {dropped_empty} empty | "
              f"mean {lens.mean():.0f} tok, p95 {np.percentile(lens, 95):.0f}, max {lens.max()}")

    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    ex = torch.load(os.path.join(args.out_dir, "val.pt"))
    print("\nexample (prompt | response):")
    ids, p = ex["ids"][0], ex["prompt_len"][0]
    print(repr(tok.decode(ids[:p], skip_special_tokens=False)), "|", repr(tok.decode(ids[p:], skip_special_tokens=False)))
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # HF datasets can leave threads that hang a Kaggle commit

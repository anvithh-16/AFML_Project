"""
Tokenize the pretraining corpus into flat uint16 binaries (nanoGPT style).

Output (in artifacts/data/):
    train.bin, val.bin   flat token streams, docs separated by <|endoftext|>
    meta.json            token counts, source, tokenizer hash

Colab / Kaggle:
    python data/prepare_pretrain.py                       # 600M tokens
    python data/prepare_pretrain.py --target_tokens 20_000_000   # quick slice
Local smoke test:
    python data/prepare_pretrain.py --local_text some.txt

The split is by document (every Nth doc -> val), so no document is cut across
train and val. Writing is append-only and resumable via --resume.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EOT, DataConfig, TokenizerConfig  # noqa: E402
from tokenizer.train_tokenizer import local_text_iterator  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402


def doc_stream(args, cfg: DataConfig, skip: int):
    if args.local_text:
        for i, t in enumerate(local_text_iterator(args.local_text)):
            if i >= skip:
                yield i, t
        return
    from datasets import load_dataset
    ds = load_dataset(cfg.hf_dataset, name=cfg.hf_subset, split="train", streaming=True)
    if skip:
        ds = ds.skip(skip)
    for i, ex in enumerate(ds, start=skip):
        yield i, ex["text"]


def file_sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def main():
    dcfg, tcfg = DataConfig(), TokenizerConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=int, default=dcfg.target_tokens)
    ap.add_argument("--tokenizer", default=tcfg.path)
    ap.add_argument("--out_dir", default=dcfg.out_dir)
    ap.add_argument("--local_text", default=None)
    ap.add_argument("--batch_docs", type=int, default=1024)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eot_id = tok.token_to_id(EOT)
    assert eot_id == 0 and tok.get_vocab_size() < 2**16
    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "meta.json")
    paths = {s: os.path.join(args.out_dir, f"{s}.bin") for s in ("train", "val")}

    counts, docs_seen = {"train": 0, "val": 0}, 0
    if args.resume and os.path.exists(meta_path):
        m = json.load(open(meta_path))
        counts, docs_seen = m["tokens"], m["docs_seen"]
        # truncate any partial batch written after the last meta save
        for s in paths:
            with open(paths[s], "r+b") as f:
                f.truncate(counts[s] * 2)
        print(f"resuming at doc {docs_seen}, train tokens {counts['train']:,}")
    else:
        for p in paths.values():
            open(p, "wb").close()

    files = {s: open(p, "ab") for s, p in paths.items()}

    def save_meta():
        json.dump({
            "tokens": counts, "docs_seen": docs_seen,
            "source": args.local_text or f"{dcfg.hf_dataset}/{dcfg.hf_subset}",
            "val_every_n_docs": dcfg.val_every_n_docs, "dtype": dcfg.dtype,
            "vocab_size": tok.get_vocab_size(), "tokenizer_sha": file_sha(args.tokenizer),
        }, open(meta_path, "w"), indent=2)

    def flush(batch):
        nonlocal docs_seen
        if not batch:
            return
        encs = tok.encode_batch([t for _, t in batch])
        out = {"train": [], "val": []}
        for (i, _), e in zip(batch, encs):
            split = "val" if i % dcfg.val_every_n_docs == 0 else "train"
            out[split].append(np.asarray(e.ids + [eot_id], dtype=np.uint16))
        for s in out:
            if out[s]:
                arr = np.concatenate(out[s])
                files[s].write(arr.tobytes())
                counts[s] += int(arr.size)
        for f in files.values():
            f.flush()
        docs_seen = batch[-1][0] + 1
        save_meta()

    t0, batch = time.time(), []
    for i, text in doc_stream(args, dcfg, docs_seen):
        batch.append((i, text))
        if len(batch) >= args.batch_docs:
            flush(batch)
            batch = []
            rate = counts["train"] / max(time.time() - t0, 1e-6)
            print(f"\rdocs {docs_seen:,}  train {counts['train']/1e6:.1f}M  "
                  f"val {counts['val']/1e6:.2f}M  ({rate/1e3:.0f}k tok/s)", end="", flush=True)
            if counts["train"] >= args.target_tokens:
                break
    flush(batch)
    for f in files.values():
        f.close()
    print(f"\ndone: train={counts['train']:,} val={counts['val']:,} tokens, docs={docs_seen:,}")


if __name__ == "__main__":
    main()

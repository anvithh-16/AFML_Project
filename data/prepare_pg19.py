"""
Build the long-context pretraining set: PG-19 books + a FineWeb-Edu share, cut into
fixed rows of seq_len+1 tokens. Each PG-19 row is ONE continuous stretch of ONE book,
so the memory sees real text continue across all 8 chunks of a 2,048-token row.
Made once, used identically by Variants A, B and C.

Kaggle (Internet ON, CPU is enough; ~30-60 min for 200M tokens):
    python data/prepare_pg19.py \
        --tokenizer  /kaggle/input/<AFML_Project_Data>/artifacts/tokenizer/tokenizer.json \
        --fineweb_dir /kaggle/input/<AFML_Project_Data>/artifacts/data \
        --out_dir /kaggle/working/artifacts/pg19mix
Offline test:
    python data/prepare_pg19.py --local_books books/ ...   (folder with train/ validation/ test/ *.txt)

Output (save as ONE Kaggle dataset; every variant loads this same version):
    train.bin        rows of row_len uint16 tokens, shuffled with --seed (75% PG-19, 25% FineWeb)
    train_source.npy one byte per train row: 0 = PG-19, 1 = FineWeb-Edu
    val.bin          rows from the PG-19 validation books (pure PG-19, never trained on)
    test_books.bin   the PG-19 test books, whole, for eval/far_recall_loss.py
    test_offsets.npy start of each test book in test_books.bin (+ end)
    meta.json        format "rows", row_len, token/row counts, book lists, tokenizer_sha, seed

data/loader.py sees format "rows" in meta.json and samples whole rows only.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EOT  # noqa: E402
from data.prepare_pretrain import file_sha  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

PG19 = "emozilla/pg19"     # parquet copy of deepmind/pg19 (28,602 / 50 / 100 books)


def book_stream(args, split):
    """Yields (title, text) for one PG-19 split."""
    if args.local_books:
        for p in sorted(glob.glob(os.path.join(args.local_books, split, "*.txt"))):
            yield os.path.basename(p), open(p, encoding="utf-8").read()
        return
    from datasets import load_dataset
    for ex in load_dataset(PG19, split=split, streaming=True):
        yield ex["short_book_title"], ex["text"]


def book_rows(ids, row_len, max_rows):
    """Consecutive, non-overlapping rows from one book; the tail shorter than a row is dropped."""
    n = min(len(ids) // row_len, max_rows)
    if n == 0:
        return np.zeros((0, row_len), np.uint16)
    return np.asarray(ids[:n * row_len], dtype=np.uint16).reshape(n, row_len)


def tokenize_books(tok, stream, eot, batch_books=16):
    batch = []
    for title, text in stream:
        batch.append((title, text))
        if len(batch) == batch_books:
            for (t, _), e in zip(batch, tok.encode_batch([x for _, x in batch])):
                yield t, e.ids + [eot]
            batch = []
    if batch:
        for (t, _), e in zip(batch, tok.encode_batch([x for _, x in batch])):
            yield t, e.ids + [eot]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--fineweb_dir", required=True, help="existing AFML_Project_Data artifacts/data (train.bin)")
    ap.add_argument("--out_dir", default="artifacts/pg19mix")
    ap.add_argument("--local_books", default=None)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--train_tokens", type=int, default=200_000_000, help="total, PG-19 + FineWeb")
    ap.add_argument("--fineweb_frac", type=float, default=0.25)
    ap.add_argument("--max_rows_per_book", type=int, default=100,
                    help="cap so a few very long books cannot dominate (100 rows ~ 205k tokens)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id(EOT)
    assert eot == 0 and tok.get_vocab_size() < 2 ** 16
    sha = file_sha(args.tokenizer)
    fw_meta = json.load(open(os.path.join(args.fineweb_dir, "meta.json")))
    assert fw_meta["tokenizer_sha"] == sha, "FineWeb data was made with a different tokenizer"
    row_len = args.seq_len + 1
    n_rows = args.train_tokens // row_len
    n_fw = int(round(n_rows * args.fineweb_frac))
    n_pg = n_rows - n_fw
    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)                       # never leave a meta that disagrees with the .bin files
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ---- PG-19 train rows -------------------------------------------------------
    pg_rows, titles, pg_tokens_read = [], [], 0
    have = 0
    for title, ids in tokenize_books(tok, book_stream(args, "train"), eot):
        r = book_rows(ids, row_len, args.max_rows_per_book)
        if len(r) == 0:
            continue
        r = r[:n_pg - have]
        pg_rows.append(r)
        titles.append(title)
        have += len(r)
        pg_tokens_read += len(ids)
        print(f"\rPG-19 train: {len(titles):,} books, {have:,}/{n_pg:,} rows "
              f"({time.time() - t0:.0f}s)", end="", flush=True)
        if have >= n_pg:
            break
    print()
    pg = np.concatenate(pg_rows) if pg_rows else np.zeros((0, row_len), np.uint16)
    if len(pg) < n_pg:
        print(f"warning: only {len(pg):,} PG-19 rows available (asked {n_pg:,})")

    # ---- FineWeb-Edu rows: random non-overlapping windows of the existing train.bin
    fw_data = np.memmap(os.path.join(args.fineweb_dir, "train.bin"), dtype=np.uint16, mode="r")
    n_slots = len(fw_data) // row_len
    assert n_fw <= n_slots, "not enough FineWeb tokens"
    slots = np.sort(rng.choice(n_slots, size=n_fw, replace=False))
    fw = np.stack([fw_data[s * row_len:(s + 1) * row_len] for s in slots]) if n_fw else \
        np.zeros((0, row_len), np.uint16)

    # ---- mix + shuffle (fixed seed) ---------------------------------------------
    rows = np.concatenate([pg, fw])
    source = np.concatenate([np.zeros(len(pg), np.uint8), np.ones(len(fw), np.uint8)])
    perm = rng.permutation(len(rows))
    rows, source = rows[perm], source[perm]
    rows.tofile(os.path.join(args.out_dir, "train.bin"))
    np.save(os.path.join(args.out_dir, "train_source.npy"), source)

    # ---- PG-19 validation rows (pure PG-19) -------------------------------------
    val_rows, val_titles = [], []
    for title, ids in tokenize_books(tok, book_stream(args, "validation"), eot):
        r = book_rows(ids, row_len, args.max_rows_per_book)
        if len(r):
            val_rows.append(r)
            val_titles.append(title)
    val = np.concatenate(val_rows)
    val.tofile(os.path.join(args.out_dir, "val.bin"))

    # ---- PG-19 test books, whole (far-recall eval) -------------------------------
    offsets, test_titles, total = [0], [], 0
    with open(os.path.join(args.out_dir, "test_books.bin"), "wb") as f:
        for title, ids in tokenize_books(tok, book_stream(args, "test"), eot):
            a = np.asarray(ids, dtype=np.uint16)
            f.write(a.tobytes())
            total += len(a)
            offsets.append(total)
            test_titles.append(title)
    np.save(os.path.join(args.out_dir, "test_offsets.npy"), np.asarray(offsets, dtype=np.int64))

    meta = {
        "format": "rows", "row_len": row_len, "seq_len": args.seq_len,
        "tokens": {"train": int(rows.size), "val": int(val.size)},
        "rows": {"train": int(len(rows)), "val": int(len(val)), "train_pg19": int(len(pg)),
                 "train_fineweb": int(len(fw))},
        "test_books": len(test_titles), "test_tokens": int(total),
        "sources": {"pg19": args.local_books or PG19, "fineweb": fw_meta.get("source", "")},
        "fineweb_frac": args.fineweb_frac, "max_rows_per_book": args.max_rows_per_book,
        "pg19_train_books": titles, "pg19_val_books": val_titles, "pg19_test_books": test_titles,
        "vocab_size": tok.get_vocab_size(), "tokenizer_sha": sha, "seed": args.seed, "dtype": "uint16",
    }
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s: train {len(rows):,} rows ({len(pg):,} PG-19 from "
          f"{len(titles):,} books + {len(fw):,} FineWeb) = {rows.size / 1e6:.1f}M tokens | "
          f"val {len(val):,} rows from {len(val_titles)} books | test {len(test_titles)} books, "
          f"{total / 1e6:.1f}M tokens -> {args.out_dir}")


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # HF streaming can leave threads that hang a Kaggle commit

"""
BABILong for the instruction-tuning stage (train) and for evaluation (test), in the
same format as data/prepare_instruct.py, so train/instruct.py needs no change.
Identical for Variants A, B and C.

    story (PG-19 text with the facts hidden in it)  ->  question  ->  one-word answer

Sources (Hugging Face):
    train: RMT-team/babilong-train-5k-samples   configs 0k..32k, splits qa1..qa10, 5k each
    test:  RMT-team/babilong-1k-samples          configs 0k..128k, splits qa1..qa5, ~1k each
Test examples whose story also appears in the train pick are removed.

Kaggle (Internet ON, CPU, a few minutes):
    python data/prepare_babilong.py --tokenizer <...>/tokenizer.json \
        --dolly_dir /kaggle/input/<AFML_Project_Instruct>/artifacts/instruct \
        --out_dir /kaggle/working/artifacts/instruct_babilong

Output (one Kaggle dataset):
    train.pt, val.pt  Dolly + BABILong mixed, {"ids", "prompt_len", "raw"} like Dolly's files
                      (val: Dolly val first, so instruct.py's samples stay Dolly questions)
    test.pt           BABILong test, by task and length, with each task's answer choices
    meta.json         counts per (task, length), dropped-too-long, tokenizer_sha, seed

Prompt (data/instruct_format.py, unchanged): the story is the context and the question goes
at the END of it, right before the answer, as a reader would see it:
    <|user|> Read the story, then answer the question at the end with one word.
    <|context|> {story}\\n\\nQuestion: {question}
    <|assistant|> {answer}<|endoftext|>
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.instruct_format import encode_example  # noqa: E402
from data.prepare_pretrain import file_sha  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

TRAIN_REPO = "RMT-team/babilong-train-5k-samples"
TEST_REPO = "RMT-team/babilong-1k-samples"
INSTRUCTION = "Read the story, then answer the question at the end with one word."


def context_of(story, question):
    return f"{story.strip()}\n\nQuestion: {question.strip()}"


def load_split(repo, length, task, local_dir):
    """List of {input, question, target}."""
    if local_dir:
        p = os.path.join(local_dir, os.path.basename(repo), length, f"{task}.jsonl")
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    from datasets import load_dataset
    return list(load_dataset(repo, length, split=task))


def story_hash(s):
    return hashlib.sha1(s.strip().encode()).hexdigest()


def encode(tok, ex):
    return encode_example(tok, INSTRUCTION, context_of(ex["input"], ex["question"]), ex["target"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dolly_dir", required=True, help="output of data/prepare_instruct.py (train.pt, val.pt)")
    ap.add_argument("--out_dir", default="artifacts/instruct_babilong")
    ap.add_argument("--local_dir", default=None, help="offline: <dir>/<repo name>/<length>/<task>.jsonl")
    ap.add_argument("--tasks", nargs="+", default=["qa1", "qa2", "qa3"])
    ap.add_argument("--train_lengths", nargs="+", default=["0k", "1k", "2k", "4k"])
    ap.add_argument("--test_lengths", nargs="+", default=["1k", "2k", "4k"])
    ap.add_argument("--train_per_cell", type=int, default=750, help="examples per (task, length)")
    ap.add_argument("--val_per_cell", type=int, default=25)
    ap.add_argument("--test_per_cell", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=4096, help="train/val examples longer than this are dropped")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    sha = file_sha(args.tokenizer)
    dmeta = json.load(open(os.path.join(args.dolly_dir, "meta.json")))
    assert dmeta["tokenizer_sha"] == sha, "Dolly data was made with a different tokenizer"
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    meta = {"train_repo": TRAIN_REPO, "test_repo": TEST_REPO, "instruction": INSTRUCTION,
            "tasks": args.tasks, "train_lengths": args.train_lengths, "test_lengths": args.test_lengths,
            "max_len": args.max_len, "seed": args.seed, "tokenizer_sha": sha, "cells": {}}

    out = {s: {"ids": [], "prompt_len": [], "raw": []} for s in ("train", "val")}
    train_stories = set()
    for task in args.tasks:
        for length in args.train_lengths:
            rows = load_split(TRAIN_REPO, length, task, args.local_dir)
            order = rng.permutation(len(rows))
            picks = {"train": order[:args.train_per_cell],
                     "val": order[args.train_per_cell:args.train_per_cell + args.val_per_cell]}
            cell = {}
            for split, idx in picks.items():
                kept = dropped = 0
                for i in idx:
                    ex = rows[int(i)]
                    ids, plen = encode(tok, ex)
                    if len(ids) > args.max_len:
                        dropped += 1
                        continue
                    out[split]["ids"].append(ids)
                    out[split]["prompt_len"].append(plen)
                    out[split]["raw"].append({"instruction": INSTRUCTION,
                                              "context": context_of(ex["input"], ex["question"]),
                                              "response": ex["target"], "category": f"babilong_{task}_{length}"})
                    train_stories.add(story_hash(ex["input"]))
                    kept += 1
                cell[split] = {"kept": kept, "dropped_too_long": dropped}
            meta["cells"][f"{task}/{length}"] = cell
            print(f"train {task}/{length}: kept {cell['train']['kept']}, too long {cell['train']['dropped_too_long']}")

    # ---- merge with Dolly (Dolly val first) -------------------------------------
    n_bl = {s: len(out[s]["ids"]) for s in out}
    for split in ("train", "val"):
        d = torch.load(os.path.join(args.dolly_dir, f"{split}.pt"))
        merged = {k: list(d[k]) + out[split][k] for k in ("ids", "prompt_len", "raw")}
        if split == "train":                                  # one fixed shuffle
            perm = rng.permutation(len(merged["ids"]))
            merged = {k: [merged[k][i] for i in perm] for k in merged}
        torch.save(merged, os.path.join(args.out_dir, f"{split}.pt"))
        meta[f"{split}_examples"] = {"dolly": len(d["ids"]), "babilong": n_bl[split], "total": len(merged["ids"])}
        print(f"{split}: {len(d['ids']):,} Dolly + {n_bl[split]:,} BABILong = {len(merged['ids']):,}")

    # ---- test (separate repo, deduplicated against train) -----------------------
    test = {"ids": [], "prompt_len": [], "raw": [], "choices": {}}
    meta["test_cells"] = {}
    for task in args.tasks:
        answers = set()
        for length in args.test_lengths:
            rows = load_split(TEST_REPO, length, task, args.local_dir)
            answers |= {r["target"].strip() for r in rows}
            kept = dup = 0
            for i in rng.permutation(len(rows)):
                if kept == args.test_per_cell:
                    break
                ex = rows[int(i)]
                if story_hash(ex["input"]) in train_stories:
                    dup += 1
                    continue
                ids, plen = encode(tok, ex)
                test["ids"].append(ids)
                test["prompt_len"].append(plen)
                test["raw"].append({"instruction": INSTRUCTION, "context": context_of(ex["input"], ex["question"]),
                                    "response": ex["target"].strip(), "task": task, "length": length,
                                    "n_tokens": len(ids)})
                kept += 1
            meta["test_cells"][f"{task}/{length}"] = {"kept": kept, "removed_seen_in_train": dup}
            print(f"test {task}/{length}: {kept} (removed {dup} seen in train)")
        test["choices"][task] = sorted(answers)
    torch.save(test, os.path.join(args.out_dir, "test.pt"))
    meta["test_choices"] = test["choices"]
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=1)

    ex = test["raw"][0]
    print(f"\nanswer choices: {test['choices']}\nexample test prompt tail: ...{ex['context'][-120:]!r} -> {ex['response']!r}")
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)

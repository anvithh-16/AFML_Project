"""
Score any checkpoint (A, B or C; pretrained or instruction-tuned) on the frozen
eval set. See docs/EVAL_SPEC.md.

    python eval/run_eval.py --ckpt <path/to/variant-A_instruct_xxx.pt> \
        --tokenizer artifacts/tokenizer/tokenizer.json --eval_dir artifacts/eval \
        --out_dir results

    # sanity check: an UNTRAINED model must score ~25% everywhere
    python eval/run_eval.py --random A --tokenizer ... --eval_dir ... --out_dir results

Scoring (multiple choice by likelihood):
    prompt = build_prompt(instruction, context)          # same template as instruction tuning
    for each of the 4 choices:  score = sum log p(choice tokens | prompt)
    pick = argmax score.  All 4 choices have the same token length, so sums are fair.
The 4 choices are run as one batch of 4 sequences (one forward pass per example).
Memory chunking happens inside model.forward, so this file never touches memory.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig  # noqa: E402
from data.instruct_format import encode_prompt  # noqa: E402
from model.transformer import SLM  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

EXPECTED_SHA = "7f469545b846"


def load_model(args, device):
    if args.random:                                   # untrained model, for the leak check
        cfg = ModelConfig()
        cfg.memory = {**cfg.memory, "variant": args.random}
        torch.manual_seed(0)
        info = {"variant": args.random, "stage": "random", "step": 0}
        return SLM(cfg).to(device).eval(), info
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sha = ck.get("tokenizer_sha")
    if sha != EXPECTED_SHA and not args.allow_other_tokenizer:
        sys.exit(f"checkpoint tokenizer_sha {sha} != {EXPECTED_SHA}: trained on different data")
    model = SLM(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    info = {"variant": ck.get("variant", "?"), "stage": ck.get("stage", "?"), "step": ck.get("step")}
    return model.to(device).eval(), info


@torch.no_grad()
def score_choices(model, prompt_ids, choice_ids, device, amp):
    """Returns the summed log-prob of each choice given the prompt."""
    seqs = torch.tensor([prompt_ids + c for c in choice_ids], device=device)     # (4, P+R)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        logits, _ = model(seqs)
    logp = F.log_softmax(logits.float(), dim=-1)
    P, R = len(prompt_ids), len(choice_ids[0])
    # logits at position t predict token t+1, so the response tokens P..P+R-1
    # are predicted by positions P-1..P+R-2
    pred = logp[:, P - 1:P + R - 1, :]
    target = seqs[:, P:P + R]
    return pred.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum(-1).tolist()


@torch.no_grad()
def greedy_answer(model, tok, prompt_ids, device, amp, max_new=16):
    idx = torch.tensor([prompt_ids], device=device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model.generate(idx, max_new_tokens=max_new, temperature=0.0, eos_id=0)
    return tok.decode(out[0, len(prompt_ids):].tolist()).strip()


def norm(s):
    return " ".join(s.lower().replace(".", " ").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--random", choices=["A", "B", "C"], default=None, help="untrained model (leak check)")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--eval_dir", default="artifacts/eval")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--tests", default=None, help="default: every test file in eval_dir")
    ap.add_argument("--limit", type=int, default=None, help="N rows per test, spread over all bins (quick runs)")
    ap.add_argument("--greedy", action="store_true", help="also do greedy exact match (slower)")
    ap.add_argument("--tag", default=None, help="name for the output files")
    ap.add_argument("--allow_other_tokenizer", action="store_true", help="local testing only")
    args = ap.parse_args()
    assert args.ckpt or args.random, "give --ckpt or --random"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    tok = Tokenizer.from_file(args.tokenizer)
    eval_meta = json.load(open(os.path.join(args.eval_dir, "eval_meta.json")))
    model, info = load_model(args, device)
    if eval_meta["tokenizer_sha"] != EXPECTED_SHA and not args.allow_other_tokenizer:
        sys.exit("eval set was built with a different tokenizer")
    tag = args.tag or f"variant-{info['variant']}_{info['stage']}"
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{tag}.csv")
    print(f"{tag} | step {info['step']} | {device} | params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    fields = ["id", "test", "bin", "regime", "distance", "relation", "variant_info", "answer", "picked",
              "correct", "picked_stale", "picked_distractor", "picked_injected", "greedy", "greedy_correct",
              "logprobs", "choices"]
    summary = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        tests = args.tests.split(",") if args.tests else [
            t for t in ["t1c", "t1", "t2", "t3", "t4", "t5"]
            if os.path.exists(os.path.join(args.eval_dir, f"{t}.jsonl"))]
        for test in tests:
            rows = [json.loads(line) for line in open(os.path.join(args.eval_dir, f"{test}.jsonl"))]
            if args.limit:                            # spread over all bins, not just the first
                rows = rows[::max(1, len(rows) // args.limit)][:args.limit]
            for i, r in enumerate(rows):
                m = r["meta"]
                prompt_ids = encode_prompt(tok, r["instruction"], r["context"])
                choice_ids = [tok.encode(c).ids for c in r["choices"]]
                assert len({len(c) for c in choice_ids}) == 1, f"{r['id']}: choices differ in token length"
                lp = score_choices(model, prompt_ids, choice_ids, device, amp)
                picked = r["choices"][max(range(4), key=lambda j: lp[j])]
                out = {
                    "id": r["id"], "test": test, "bin": r["bin"], "regime": r["regime"],
                    "distance": m["distance"], "relation": m.get("relation", "multi_hop"),
                    "variant_info": m.get("n_distractors", m.get("corruption", m.get("far_hop", ""))),
                    "answer": r["answer"], "picked": picked, "correct": int(picked == r["answer"]),
                    "picked_stale": int(picked == m.get("stale_value")),
                    "picked_distractor": int(picked in m.get("distractor_values", [])),
                    "picked_injected": int(picked == m.get("injected_value")),
                    "greedy": "", "greedy_correct": "",
                    "logprobs": json.dumps([round(x, 3) for x in lp]), "choices": json.dumps(r["choices"]),
                }
                if args.greedy:
                    g = greedy_answer(model, tok, prompt_ids, device, amp)
                    out["greedy"], out["greedy_correct"] = g, int(norm(g).startswith(norm(r["answer"])))
                w.writerow(out)
                summary[(test, r["regime"])]["acc"].append(out["correct"])
                if (i + 1) % 100 == 0:
                    print(f"  {test} {i + 1}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)

    # --- short summary: accuracy per test and regime (chance = 25%) ---
    res = {"tag": tag, **info, "chance": 0.25, "accuracy": {}}
    print(f"\n{'test':<5}{'regime':<8}{'n':>6}{'acc':>8}")
    for (test, regime), d in sorted(summary.items()):
        acc = sum(d["acc"]) / len(d["acc"])
        res["accuracy"][f"{test}_{regime}"] = {"n": len(d["acc"]), "acc": round(acc, 4)}
        print(f"{test:<5}{regime:<8}{len(d['acc']):>6}{acc:>8.3f}")
    with open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out_csv}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

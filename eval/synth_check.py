"""
Does the memory carry facts at all? A diagnostic on the PRETRAINED checkpoints.

Builds fresh questions in the exact format of data/make_synth.py (same templates,
a new seed, filler from val.bin, which no model trained on) and scores A, B and C,
plus B and C a second time with memory switched OFF (read returns zeros, so the
model is its own backbone). Comparing memory-on vs memory-off on the SAME weights
isolates what the memory contributes, independent of B/C vs A training noise.

    python eval/synth_check.py --ckpts A.pt,B.pt,C.pt \
        --data_dir <...>/artifacts/data --tokenizer <...>/artifacts/tokenizer/tokenizer.json \
        --out_dir /kaggle/working/synth_check

Each question:  [lead filler] FACT [filler, maybe distractor facts, maybe a CORRECTION] QUESTION
4 choices, scored by the summed log-probability of the answer tokens (" " + value).
All values are 5-digit codes, so every choice has the same token length (digits are
single tokens) and no choice is favoured by length. Wrong choices are, in order of
preference: the stale value (if a correction was inserted), distractor keys' values
(if present), then random codes. Chance = 25%.

Distances (tokens between the fact and the question) per bin:
  ~150         inside the 256-token window (control: everything should be high)
  300 … 900    the range the synthetic pretraining examples covered
  1500, 2500   longer than any training example
Two tables are printed: all questions, and the "clean" subset with no correction
(where reading the nearest matching value cannot give the right answer).
"""
import argparse
import csv
import json
import os
import sys
import time
import types
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig  # noqa: E402
from data.make_synth import ENTITIES, FACT_TEMPLATES, QUESTION_TEMPLATES, UPDATE_TEMPLATES  # noqa: E402
from data.prepare_pretrain import file_sha  # noqa: E402
from model.transformer import SLM  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402


# ---------------------------------------------------------------- questions
def build_questions(tok, val, n_per_bin, bins, seed, ctrl_range=(100, 200)):
    rng = np.random.default_rng(seed)
    enc = lambda s: tok.encode(s).ids  # noqa: E731
    code = lambda: str(int(rng.integers(10000, 100000)))  # noqa: E731

    def key():
        return f"{rng.choice(ENTITIES)} {int(rng.integers(1, 1000))}"

    def filler(n):
        if n <= 0:
            return []
        s = int(rng.integers(0, len(val) - n - 1))
        return val[s:s + n].tolist()

    qs = []
    for b in bins:
        for _ in range(n_per_bin):
            dist = int(rng.integers(*ctrl_range)) if b == "ctrl" else int(b)
            k, v = key(), code()
            fact = enc("\n" + str(rng.choice(FACT_TEMPLATES)).format(key=k, val=v) + "\n")
            middle, distractor_vals, stale = [], [], None
            if rng.random() < 0.5:                                    # 1-3 distractor facts
                for _ in range(int(rng.integers(1, 4))):
                    k2 = key()
                    while k2 == k:
                        k2 = key()
                    v2 = code()
                    distractor_vals.append(v2)
                    middle.append(enc("\n" + str(rng.choice(FACT_TEMPLATES)).format(key=k2, val=v2) + "\n"))
            if rng.random() < 0.25:                                   # later correction
                stale, v = v, code()
                middle.append(enc("\n" + str(rng.choice(UPDATE_TEMPLATES)).format(key=k, val=v) + "\n"))
            gap_fill = dist - sum(len(m) for m in middle)
            if gap_fill < 0:
                continue
            cuts = sorted(rng.integers(0, gap_fill + 1, size=len(middle)).tolist())
            fill, gap, prev = filler(gap_fill), [], 0
            for c, m in zip(cuts, middle):
                gap += fill[prev:c] + m
                prev = c
            gap += fill[prev:]
            question = enc("\n" + str(rng.choice(QUESTION_TEMPLATES)).format(key=k))
            ctx = filler(int(rng.integers(0, 65))) + fact + gap + question

            wrong = ([stale] if stale else []) + distractor_vals
            wrong = [w for w in dict.fromkeys(wrong) if w != v][:3]
            kinds = {w: ("stale" if w == stale else "distractor") for w in wrong}
            while len(wrong) < 3:
                w = code()
                if w != v and w not in wrong:
                    wrong.append(w)
                    kinds[w] = "random"
            choices = [v] + wrong
            order = rng.permutation(4)
            choices = [choices[i] for i in order]
            qs.append({"bin": "ctrl" if b == "ctrl" else int(b), "distance": dist, "ctx": ctx,
                       "choices": choices, "answer": choices.index(v),
                       "kinds": [("correct" if c == v else kinds[c]) for c in choices],
                       "has_correction": stale is not None, "n_distractors": len(distractor_vals),
                       "len": len(ctx)})
    return qs


# ---------------------------------------------------------------- scoring
@torch.no_grad()
def score(model, tok, qs, device, use_amp):
    """Pick for each question the choice with the highest summed log-prob of its tokens."""
    picks = []
    for q in qs:
        cands = [tok.encode(" " + c).ids for c in q["choices"]]
        L = len(cands[0])
        assert all(len(c) == L for c in cands), "choices must have equal token length"
        x = torch.tensor([q["ctx"] + c for c in cands], device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits, _ = model(x[:, :-1])
        lp = F.log_softmax(logits[:, -L:].float(), dim=-1)            # positions predicting the choice
        tgt = x[:, -L:]
        s = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(-1)
        picks.append(int(s.argmax()))
    return picks


def memory_off(model):
    """Same weights, memory read switched off: the model runs as its own backbone."""
    if model.memory.is_null:
        return None
    model.memory.read = types.MethodType(lambda self, h, state: torch.zeros_like(h), model.memory)
    return model


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SLM(ModelConfig(**ck["model_config"]))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths (pretrained A,B,C)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", default="synth_check")
    ap.add_argument("--n_per_bin", type=int, default=100)
    ap.add_argument("--bins", default="ctrl,300,500,700,900,1500,2500")
    ap.add_argument("--seed", type=int, default=7777)                 # make_synth used 2026
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    use_amp = args.device.startswith("cuda")

    tok = Tokenizer.from_file(args.tokenizer)
    sha = file_sha(args.tokenizer)
    val = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    bins = [b if b == "ctrl" else int(b) for b in args.bins.split(",")]
    qs = build_questions(tok, val, args.n_per_bin, bins, args.seed)
    print(f"{len(qs)} questions | bins {args.bins} | max length {max(q['len'] for q in qs)} tokens | "
          f"{sum(q['has_correction'] for q in qs)} with correction")

    runs = []                                                         # (label, model)
    for p in args.ckpts.split(","):
        m, ck = load(p.strip(), args.device)
        assert ck.get("tokenizer_sha") == sha, f"{p}: tokenizer {ck.get('tokenizer_sha')} != {sha}"
        v = ck.get("variant", "?")
        if ck.get("stage") != "pretrain":
            print(f"note: {os.path.basename(p)} is stage={ck.get('stage')!r}, not 'pretrain'")
        runs.append((v if m.memory.is_null else f"{v} mem on", m, p))
        if not m.memory.is_null:
            m_off, _ = load(p.strip(), args.device)
            runs.append((f"{v} mem off", memory_off(m_off), p))

    results = {}
    for label, m, p in runs:
        t0 = time.time()
        results[label] = score(m, tok, qs, args.device, use_amp)
        print(f"  scored {label:10s} in {time.time() - t0:.0f}s")
        del m
        if use_amp:
            torch.cuda.empty_cache()

    # ---- CSV: one row per question per run
    with open(os.path.join(args.out_dir, "synth_check.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "bin", "distance", "has_correction", "n_distractors", "correct", "picked_kind"])
        for label, picks in results.items():
            for q, pk in zip(qs, picks):
                w.writerow([label, q["bin"], q["distance"], int(q["has_correction"]), q["n_distractors"],
                            int(pk == q["answer"]), q["kinds"][pk]])

    # ---- tables
    labels = list(results)

    def table(title, keep):
        print(f"\n{title}  (accuracy %, chance = 25)")
        print(f"{'bin':>8} {'n':>4} " + " ".join(f"{l:>11}" for l in labels))
        rows = {}
        for b in bins:
            idx = [i for i, q in enumerate(qs) if q["bin"] == b and keep(q)]
            if not idx:
                continue
            accs = [100 * np.mean([results[l][i] == qs[i]["answer"] for i in idx]) for l in labels]
            rows[str(b)] = dict(zip(labels, accs))
            name = "100-200" if b == "ctrl" else str(b)
            print(f"{name:>8} {len(idx):>4} " + " ".join(f"{a:>11.1f}" for a in accs))
        return rows

    out = {"all": table("ALL QUESTIONS", lambda q: True),
           "clean": table("CLEAN (no correction: the nearest matching value is wrong or absent)",
                          lambda q: not q["has_correction"])}

    print("\nWRONG ANSWERS by kind (% of all questions at 300+ tokens)")
    far = [i for i, q in enumerate(qs) if q["bin"] != "ctrl"]
    print(f"{'run':>11} {'stale':>7} {'distractor':>11} {'random':>7}")
    for l in labels:
        kinds = defaultdict(int)
        for i in far:
            k = qs[i]["kinds"][results[l][i]]
            if k != "correct":
                kinds[k] += 1
        print(f"{l:>11} " + " ".join(f"{100 * kinds[k] / len(far):>{n}.1f}"
                                     for k, n in (("stale", 7), ("distractor", 11), ("random", 7))))
    json.dump({"tables": out, "n_per_bin": args.n_per_bin, "seed": args.seed,
               "ckpts": args.ckpts.split(",")}, open(os.path.join(args.out_dir, "synth_check.json"), "w"), indent=2)
    print(f"\nsaved -> {args.out_dir}/synth_check.csv, synth_check.json")


if __name__ == "__main__":
    main()

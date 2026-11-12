"""
BABILong test accuracy by task and length (test split from data/prepare_babilong.py;
never trained on). Each answer choice of the task (e.g. the 6 rooms) is scored by the
summed log-prob of "<choice><|endoftext|>" after the prompt; the highest wins. Same idea as
the frozen exam's choice scoring. The whole prompt goes through the model (no truncation),
so memory sees the full story.

    python eval/babilong_eval.py --data_dir <AFML_Project_InstructBabilong>/artifacts/instruct_babilong \
        --tokenizer <...>/tokenizer.json \
        --ckpt A=<...>/variant-A_instruct.pt B=<...>/variant-B_instruct.pt C=<...>/variant-C_instruct.pt
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESPONSE_TEMPLATE  # noqa: E402
from eval.far_recall_loss import load  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402


@torch.no_grad()
def choice_logprob(model, prompt, cand, device):
    ids = torch.tensor([prompt + cand], device=device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
        logits, _ = model(ids[:, :-1])
    lp = F.log_softmax(logits.float()[0, len(prompt) - 1:], -1)
    return float(lp.gather(1, torch.tensor(cand, device=device)[:, None]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, help="NAME=path")
    ap.add_argument("--max_per_cell", type=int, default=None, help="quick runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    test = torch.load(os.path.join(a.data_dir, "test.pt"))
    cand = {t: [tok.encode(RESPONSE_TEMPLATE.format(response=c)).ids for c in cs]
            for t, cs in test["choices"].items()}
    cells = defaultdict(list)
    for i, r in enumerate(test["raw"]):
        cells[(r["task"], r["length"])].append(i)
    if a.max_per_cell:
        cells = {k: v[:a.max_per_cell] for k, v in cells.items()}
    print("chance per task: " + ", ".join(f"{t} {1 / len(c):.2f}" for t, c in test["choices"].items()))

    results = {}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        model = load(path, a.device)
        res = {}
        for (task, length), idx in sorted(cells.items()):
            right = 0
            for i in idx:
                prompt = test["ids"][i][:test["prompt_len"][i]]
                scores = [choice_logprob(model, prompt, c, a.device) for c in cand[task]]
                right += test["choices"][task][max(range(len(scores)), key=scores.__getitem__)] == \
                    test["raw"][i]["response"]
            res[f"{task}/{length}"] = right / len(idx)
            print(f"{name} {task}/{length}: {right}/{len(idx)} = {right / len(idx):.3f}", flush=True)
        results[name] = res
    keys = sorted({k for r in results.values() for k in r})
    print("\n" + "cell".ljust(10) + "".join(n.rjust(8) for n in results))
    for k in keys:
        print(k.ljust(10) + "".join(f"{results[n].get(k, float('nan')):8.3f}" for n in results))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

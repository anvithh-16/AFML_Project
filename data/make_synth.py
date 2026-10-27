"""
Synthetic long-range examples mixed into pretraining (team decision, Oct 27).

Why: ordinary web text rarely needs a fact from hundreds of tokens back, so
Variants B and C could learn to ignore their memory. These examples force a
real long-range dependency. ALL THREE variants get the same mix, so the
comparison stays fair.

Each example (built directly in token space; filler = real FineWeb text):

    [lead filler 0-64] FACT(key -> value) [filler, distractor facts, maybe a CORRECTION] QUESTION(key) ANSWER <|endoftext|>
                       |<---------- 256 to 900 tokens between fact and question ---------->|

  * 50% of examples have 1-3 distractor facts (other keys) inside the filler.
  * 25% have a later CORRECTION for the same key; the answer is the new value.
  * Values are 4-6 digit codes or single words.

TRAINING TEMPLATES ONLY. Person C's eval must use different wording, keys and
value formats (listed in docs/ARCHITECTURE_SPEC.md and in synth_meta.json),
otherwise the eval measures template memorisation, not memory.

Kaggle (CPU is enough, Internet off, ~2-4 min):
    python data/make_synth.py --data_dir <AFML_Project_Data>/artifacts/data \
        --tokenizer <AFML_Project_Data>/artifacts/tokenizer/tokenizer.json \
        --out_dir /kaggle/working/artifacts/synth
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DataConfig, ModelConfig  # noqa: E402
from data.prepare_pretrain import file_sha  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

ENTITIES = ["locker", "storage unit", "server rack", "safe", "vault", "mailbox", "bike lock",
            "garage", "filing cabinet", "warehouse door", "staff room", "lab freezer"]
WORDS = ["amber", "falcon", "cedar", "orbit", "maple", "harbor", "pixel", "quartz", "tundra",
         "violet", "willow", "zephyr", "basalt", "comet", "delta", "ember", "glacier", "indigo"]
FACT_TEMPLATES = [
    "Remember this: the combination for {key} is {val}.",
    "For the record, {key} uses the code {val}.",
    "Memo: {key} -> {val}.",
    "Note to self: {key} opens with {val}.",
]
UPDATE_TEMPLATES = [
    "Correction: {key} now uses {val}.",
    "Update: the code for {key} was changed to {val}.",
]
QUESTION_TEMPLATES = [
    "Q: What is the combination for {key}?\nA:",
    "Which code does {key} use? Answer:",
    "Recall the value for {key}:",
]


def make_value(rng):
    if rng.random() < 0.7:
        digits = int(rng.integers(4, 7))                       # 4-6 digit code
        return str(rng.integers(10 ** (digits - 1), 10 ** digits))
    return str(rng.choice(WORDS))


def make_key(rng):
    return f"{rng.choice(ENTITIES)} {int(rng.integers(1, 1000))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DataConfig().out_dir)
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--out_dir", default="artifacts/synth")
    ap.add_argument("--n_examples", type=int, default=60_000)
    ap.add_argument("--max_len", type=int, default=ModelConfig().train_seq_len)
    ap.add_argument("--min_dist", type=int, default=ModelConfig().attn_window)   # > one window back
    ap.add_argument("--max_dist", type=int, default=900)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    data_meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    sha = file_sha(args.tokenizer)
    assert sha == data_meta["tokenizer_sha"], "tokenizer does not match the pretraining data"
    train = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    eot = tok.token_to_id("<|endoftext|>")
    rng = np.random.default_rng(args.seed)
    enc = lambda s: tok.encode(s).ids  # noqa: E731

    def filler(n):
        if n <= 0:
            return []
        s = int(rng.integers(0, len(train) - n - 1))
        return train[s:s + n].tolist()

    chunks, offsets = [], [0]
    stats = {"distance": [], "length": [], "distractors": 0, "corrections": 0, "word_values": 0}
    budget_extra = 40                                   # rough room for question + answer
    for _ in range(args.n_examples):
        key, val = make_key(rng), make_value(rng)
        fact = enc("\n" + rng.choice(FACT_TEMPLATES).format(key=key, val=val) + "\n")
        lead = filler(int(rng.integers(0, 65)))

        middle = []                                     # distractors / correction, placed in the gap
        if rng.random() < 0.5:
            for _ in range(int(rng.integers(1, 4))):
                k2 = make_key(rng)
                while k2 == key:
                    k2 = make_key(rng)
                middle.append(enc("\n" + rng.choice(FACT_TEMPLATES).format(key=k2, val=make_value(rng)) + "\n"))
            stats["distractors"] += 1
        if rng.random() < 0.25:
            new = make_value(rng)
            while new == val:
                new = make_value(rng)
            val = new
            middle.append(enc("\n" + rng.choice(UPDATE_TEMPLATES).format(key=key, val=val) + "\n"))
            stats["corrections"] += 1
        stats["word_values"] += not val.isdigit()

        question = enc("\n" + rng.choice(QUESTION_TEMPLATES).format(key=key))
        answer = enc(" " + val) + [eot]
        room = args.max_len - len(lead) - len(fact) - len(question) - len(answer)
        dist = int(min(rng.integers(args.min_dist, args.max_dist + 1), room))
        gap_filler = dist - sum(len(m) for m in middle)
        if gap_filler < 0:                              # too many inserts for the room: drop them
            middle, gap_filler = [], dist
        # split the filler around the inserted facts at random points
        cuts = sorted(rng.integers(0, gap_filler + 1, size=len(middle)).tolist())
        fill = filler(gap_filler)
        gap, prev = [], 0
        for c, m in zip(cuts, middle):
            gap += fill[prev:c] + m
            prev = c
        gap += fill[prev:]

        ex = lead + fact + gap + question + answer
        assert len(ex) <= args.max_len + 1 and len(gap) == dist, (len(ex), len(gap), dist)
        chunks.append(np.asarray(ex, dtype=np.uint16))
        offsets.append(offsets[-1] + len(ex))
        stats["distance"].append(dist)
        stats["length"].append(len(ex))

    os.makedirs(args.out_dir, exist_ok=True)
    flat = np.concatenate(chunks)
    flat.tofile(os.path.join(args.out_dir, "synth.bin"))
    np.save(os.path.join(args.out_dir, "synth_offsets.npy"), np.asarray(offsets, dtype=np.int64))
    d, L = np.array(stats["distance"]), np.array(stats["length"])
    meta = {
        "n_examples": args.n_examples, "tokens": int(flat.size), "seed": args.seed,
        "tokenizer_sha": sha, "synth_sha": hashlib.sha256(flat.tobytes()).hexdigest()[:12],
        "distance_tokens": {"min": int(d.min()), "mean": round(float(d.mean()), 1), "max": int(d.max())},
        "length_tokens": {"mean": round(float(L.mean()), 1), "max": int(L.max())},
        "frac_with_distractors": round(stats["distractors"] / args.n_examples, 3),
        "frac_with_correction": round(stats["corrections"] / args.n_examples, 3),
        "frac_word_values": round(stats["word_values"] / args.n_examples, 3),
        "training_templates": {"entities": ENTITIES, "word_values": WORDS, "facts": FACT_TEMPLATES,
                               "updates": UPDATE_TEMPLATES, "questions": QUESTION_TEMPLATES},
    }
    json.dump(meta, open(os.path.join(args.out_dir, "synth_meta.json"), "w"), indent=2)
    print(f"{args.n_examples:,} examples, {flat.size / 1e6:.1f}M tokens | distance {d.min()}-{d.max()} "
          f"(mean {d.mean():.0f}) | length mean {L.mean():.0f}, max {L.max()} | synth_sha {meta['synth_sha']}")
    print("\nexample (filler shortened):")
    ex = chunks[0].tolist()
    text = tok.decode(ex, skip_special_tokens=False)
    print(text[:200].replace("\n", "\\n"), " ... ", text[-200:].replace("\n", "\\n"))
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()

"""
Build the frozen long-context eval set (Person C). See docs/EVAL_SPEC.md.

    python eval/generate.py --tokenizer artifacts/tokenizer/tokenizer.json \
        --val_bin artifacts/data/val.bin --meta artifacts/data/meta.json \
        --synth_meta artifacts/synth/synth_meta.json --out_dir artifacts/eval

Writes t1..t5 .jsonl (one example per line) + eval_meta.json.

How one example is built (all tests share this):

    [lead filler] FACT [filler ... other facts ... filler] \n\nQuestion: ...
                       |<----------- distance (tokens) ----------->| answer

Distances are measured on the FINAL token sequence the model sees
(build_prompt -> tokenizer), from the last token of the fact to the first
answer token, so "2500 tokens back" really means 2500 tokens.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.instruct_format import build_prompt  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------
EXPECTED_SHA = "7f469545b846"
INSTRUCTION = "Read the document and answer the question at the end."
WEAK_BINS = [300, 600, 1000, 1500, 2000]      # A can partly reach (< 2,040)
BLIND_BINS = [2500, 3000, 3500, 3900]         # only memory can reach
BINS = WEAK_BINS + BLIND_BINS
MAX_TOKENS = 4096
TOLERANCE = 0.03                              # actual distance within 3% of the bin

# ---------------------------------------------------------------------------
# 2. Vocabulary pools. Deliberately disjoint from the training synth (spec 4b):
#    keys are people/projects/stations/ships, values are cities, dates,
#    surnames, 3-digit and 7-digit numbers (training used 4-6 digits + words).
# ---------------------------------------------------------------------------
FIRST = ["Ilona", "Tomas", "Priya", "Kwame", "Sofia", "Mateo", "Yuki", "Anders", "Leila",
         "Ravi", "Noor", "Emil", "Hana", "Diego", "Ines", "Farid", "Greta", "Omar", "Lucia",
         "Jonas", "Mei", "Pavel", "Amara", "Viktor", "Elena", "Kofi", "Saanvi", "Bruno",
         "Ingrid", "Tariq"]
LAST = ["Varga", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Haddad", "Novak", "Castillo",
        "Brennan", "Iyer", "Kowalski", "Mensah", "Petrov", "Sato", "Alvarez", "Dubois",
        "Fischer", "Nakamura", "Osei", "Romano", "Svensson", "Yilmaz", "Becker", "Duarte",
        "Halloran", "Kuznetsov", "Mbeki", "Ortega", "Quinlan", "Strand"]
TITLES = ["Dr.", "Professor", "Ms.", "Mr."]
PROJECTS = ["Kestrel", "Heron", "Lynx", "Osprey", "Marten", "Ibis", "Puffin", "Stoat", "Gannet",
            "Plover", "Wren", "Bittern", "Shrike", "Egret", "Sable", "Curlew", "Tern", "Ocelot",
            "Kittiwake", "Merlin", "Dunlin", "Bison", "Pika", "Starling"]
STATIONS = ["Halvard", "Nordkapp", "Sorrel", "Brightwater", "Esker", "Tamsin", "Corrin",
            "Ardent", "Fellmoor", "Galdur", "Kessock", "Lowmere", "Ravensgill", "Skerry",
            "Thornbury", "Westerly", "Aldmere", "Calder", "Drumlin", "Kelso"]
SHIPS = ["Marisol", "Northern Lantern", "Albatross Grey", "Santa Ines", "Morning Tide",
         "Corvina", "Iron Wren", "Silver Kelpie", "Tamarind", "Halcyon Drift", "Seraphine",
         "Bluefin Rose", "Oriana", "Saltmarsh", "Cormorant Star", "Lodestar Bay"]
CITIES = ["Tallinn", "Porto", "Riga", "Ghent", "Bergen", "Lyon", "Krakow", "Seville", "Turin",
          "Leipzig", "Utrecht", "Malmo", "Galway", "Brno", "Graz", "Bilbao", "Aarhus", "Ljubljana",
          "Vilnius", "Nantes", "Bologna", "Tampere", "Cork", "Basel", "Split", "Plovdiv",
          "Coimbra", "Salzburg", "Tartu", "Lille", "Genoa", "Dresden", "Uppsala", "Zadar"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
LIBRARY_NAMES = ["Kreutz", "Abernathy", "Solvang", "Pellegrini", "Hartwell", "Marchetti",
                 "Ostrander", "Vandermeer", "Laskaris", "Delacroix", "Whitcombe", "Ferreira",
                 "Grunewald", "Ashdown", "Bellamy", "Rasmussen", "Calloway", "Montague"]


def make_person(rng):
    return f"{rng.choice(TITLES)} {rng.choice(FIRST)} {rng.choice(LAST)}"


def make_date(rng):
    return f"{int(rng.integers(10, 29))} {rng.choice(MONTHS)} {int(rng.integers(1950, 2020))}"


# ---------------------------------------------------------------------------
# 3. Relations: one key type -> one value type, with fact / update / question
#    wording. Update wording is used by T3 (contradiction).
# ---------------------------------------------------------------------------
RELATIONS = {
    "lives_in": dict(
        key=make_person, val=lambda rng: str(rng.choice(CITIES)),
        fact=["{k} lives in {v}.", "These days {k} makes a home in {v}."],
        update=["{k} has since moved to {v}.", "Last year {k} relocated to {v}."],
        q="In which city does {k} live?"),
    "launch_date": dict(
        key=lambda rng: f"Project {rng.choice(PROJECTS)}", val=make_date,
        fact=["{k} is scheduled to launch on {v}.", "The team set the start of {k} for {v}."],
        update=["{k} has been rescheduled to launch on {v}.", "After delays, {k} now starts on {v}."],
        q="When does {k} launch?"),
    "staff_count": dict(
        key=lambda rng: f"the research station {rng.choice(STATIONS)}",
        val=lambda rng: str(int(rng.integers(100, 1000))),          # 3 digits
        fact=["{K} is staffed by {v} people.", "Around {v} people work at {k}."],
        update=["{K} now has a staff of {v}.", "After the expansion, {v} people work at {k}."],
        q="How many people work at {k}?"),
    "captain": dict(
        key=lambda rng: f"the ship {rng.choice(SHIPS)}", val=lambda rng: str(rng.choice(LAST)),
        fact=["{K} is commanded by Captain {v}.", "Captain {v} is in charge of {k}."],
        update=["Command of {k} has passed to Captain {v}.", "Captain {v} has taken over {k}."],
        q="What is the surname of the captain of {k}?"),
    "office_number": dict(
        key=make_person, val=lambda rng: str(int(rng.integers(1_000_000, 10_000_000))),  # 7 digits
        fact=["The office line of {k} is {v}.", "You can reach {k} at extension {v}."],
        update=["The office line of {k} changed to {v}.", "{k} can now be reached at extension {v}."],
        q="What is the office line of {k}?"),
}


def fill(template, k, v=None):
    """{k} = key as-is, {K} = key with a capital first letter (sentence start)."""
    return template.format(k=k, K=k[0].upper() + k[1:], v=v)


# ---------------------------------------------------------------------------
# 4. Generator
# ---------------------------------------------------------------------------
class Builder:
    def __init__(self, tok, val_tokens, seed):
        self.tok, self.val = tok, val_tokens
        self.rng = np.random.default_rng(seed)

    def n_tok(self, text):
        return len(self.tok.encode(text).ids)

    # --- filler: a run of real val text of about n tokens, cut at sentence ends ---
    def filler(self, n):
        if n <= 0:
            return ""
        s = int(self.rng.integers(0, len(self.val) - 2 * n - 200))
        ids = [i for i in self.val[s:s + 2 * n + 200].tolist() if i != 0]   # drop <|endoftext|>
        text = self.tok.decode(ids)
        m = re.search(r"[.!?]\s+", text)                 # start at a sentence start
        text = text[m.end():] if m else text
        ids = self.tok.encode(text).ids[:n]
        text = self.tok.decode(ids)
        cut = max(text.rfind(". "), text.rfind(".\n"))   # end at a sentence end
        return text[:cut + 1] if cut > len(text) // 2 else text

    # --- a new value of the same kind AND the same token length as `ref` ---
    def val_like(self, make_val, ref, exclude=()):
        n = self.n_tok(ref)
        for _ in range(500):
            x = make_val(self.rng)
            if x != ref and x not in exclude and self.n_tok(x) == n:
                return x
        return None

    # --- choices: correct value + 3 distinct distractors of the same token length ---
    def choices(self, answer, must_include, make_val):
        target_len = self.n_tok(answer)
        out = [answer] + [c for c in dict.fromkeys(must_include) if c != answer][:3]
        tries = 0
        while len(out) < 4:
            c = make_val(self.rng)
            tries += 1
            if c not in out and self.n_tok(c) == target_len:
                out.append(c)
            if tries > 2000:
                return None
        self.rng.shuffle(out)
        return out

    # --- assemble pieces, measure the real distance, fix filler until it fits ---
    def assemble(self, segments, anchor, question, answer, target, adjust=True):
        """segments: list of text pieces; strings starting with '@FILL:<n>' are filler
        placeholders. anchor: index of the segment whose END the distance is measured from.
        adjust=False: measure only (used by T5 so matched pairs keep the same filler)."""
        texts = [self.filler(int(s[6:])) if s.startswith("@FILL:") else s for s in segments]
        for _ in range(6 if adjust else 1):
            texts = [t.strip() for t in texts]
            context = " ".join(t for t in texts if t) + f"\n\nQuestion: {question}"
            prompt = build_prompt(INSTRUCTION, context)
            enc = self.tok.encode(prompt)
            # character offset where the anchor segment ends inside the prompt
            before = " ".join(t for t in texts[:anchor + 1] if t)
            char_end = prompt.index(context) + len(before)
            fact_end_tok = max(i for i, (a, b) in enumerate(enc.offsets) if b <= char_end and b > 0)
            distance = len(enc.ids) - 1 - fact_end_tok      # first answer token is at len(prompt)
            err = target - distance
            if abs(err) <= TOLERANCE * target:
                break
            if not adjust:
                return None
            # adjust the last filler piece after the anchor
            j = max(i for i, s in enumerate(segments) if s.startswith("@FILL:") and i > anchor)
            texts[j] = self.filler(max(0, self.n_tok(texts[j]) + err))
        else:
            return None
        n_total = len(enc.ids) + self.n_tok(answer) + 1
        if n_total > MAX_TOKENS:
            return None
        return context, distance, n_total, texts

    def row(self, test, idx, bin_, assembled, answer, choices, **meta):
        if choices is None:
            return None
        context, distance, n_total = assembled[:3]
        return {"id": f"{test}_{bin_}_{idx:04d}", "test": test, "bin": bin_,
                "regime": "weak" if bin_ in WEAK_BINS else "blind",
                "instruction": INSTRUCTION, "context": context, "answer": answer,
                "choices": choices,
                "meta": {"distance": distance, "n_tokens": n_total, **meta}}

    def lead(self):
        return f"@FILL:{int(self.rng.integers(32, 97))}"

    def question_tokens(self, q):
        return self.n_tok(f"\n\nQuestion: {q}") + 4     # + "\n<|assistant|>\n" overhead

    def pick_relation(self, names=None):
        name = str(self.rng.choice(names or list(RELATIONS)))
        return name, RELATIONS[name]

    def other_keys(self, rel, key, n):
        keys = []
        while len(keys) < n:
            k = rel["key"](self.rng)
            if k != key and k not in keys:
                keys.append(k)
        return keys

    # ------------------------------------------------------------------ T1
    def t1(self, idx, d, variant=0):
        name, rel = self.pick_relation()
        k, v = rel["key"](self.rng), rel["val"](self.rng)
        q = fill(rel["q"], k)
        fact = fill(str(self.rng.choice(rel["fact"])), k, v)
        segs = [self.lead(), fact, f"@FILL:{d - self.question_tokens(q)}"]
        a = self.assemble(segs, 1, q, v, d)
        return a and self.row("t1", idx, d, a, v, self.choices(v, [], rel["val"]), relation=name)

    # ------------------------------------------------------------------ T2
    def t2(self, idx, d, variant=0):
        name, rel = self.pick_relation()
        k, v = rel["key"](self.rng), rel["val"](self.rng)
        n_dis = [2, 5, 10][variant]
        q = fill(rel["q"], k)
        dis_keys = self.other_keys(rel, k, n_dis)
        dis_vals = []
        for _ in dis_keys:
            x = self.val_like(rel["val"], v, dis_vals)
            if x is None:
                return None
            dis_vals.append(x)
        dis = [fill(str(self.rng.choice(rel["fact"])), dk, dv) for dk, dv in zip(dis_keys, dis_vals)]
        # target sits among the distractors at a random rank; all facts lie before the question
        order = list(range(n_dis + 1))
        self.rng.shuffle(order)
        facts = [fill(str(self.rng.choice(rel["fact"])), k, v) if i == n_dis else dis[i] for i in order]
        t_pos = order.index(n_dis)
        gap = d - self.question_tokens(q)
        after = n_dis - t_pos                                  # facts after the target
        per = max(20, gap // (after + 1) - 15)
        segs, anchor = [self.lead()], None
        for i, f in enumerate(facts):
            segs.append(f)
            if i == t_pos:
                anchor = len(segs) - 1
            gap_here = per if i >= t_pos else int(self.rng.integers(20, 120))
            segs.append(f"@FILL:{gap_here}")
        a = self.assemble(segs, anchor, q, v, d)
        return a and self.row("t2", idx, d, a, v, self.choices(v, dis_vals, rel["val"]),
                              relation=name, n_distractors=n_dis, distractor_values=dis_vals)

    # ------------------------------------------------------------------ T3
    def t3(self, idx, d, variant=0):
        name, rel = self.pick_relation()
        k, old = rel["key"](self.rng), rel["val"](self.rng)
        new = self.val_like(rel["val"], old)
        if new is None:
            return None
        q = fill(rel["q"], k)
        gap = d - self.question_tokens(q)
        frac = float(self.rng.uniform(0.2, 0.8))               # where the update sits in the gap
        first = int(gap * frac)
        segs = [self.lead(), fill(str(self.rng.choice(rel["fact"])), k, old), f"@FILL:{first}",
                fill(str(self.rng.choice(rel["update"])), k, new), f"@FILL:{gap - first}"]
        a = self.assemble(segs, 1, q, new, d)
        return a and self.row("t3", idx, d, a, new, self.choices(new, [old], rel["val"]),
                              relation=name, stale_value=old, update_frac=round(frac, 2))

    # ------------------------------------------------------------------ T4
    def t4(self, idx, d, variant=0):
        p = make_person(self.rng)
        cities = list(self.rng.choice(CITIES, size=3, replace=False))
        first_lib = str(self.rng.choice(LIBRARY_NAMES))
        libs = [first_lib]
        for _ in range(2):
            x = self.val_like(lambda r: str(r.choice(LIBRARY_NAMES)), first_lib, libs)
            if x is None:
                return None
            libs.append(x)
        city, lib = str(cities[0]), f"{libs[0]} Library"
        hop1 = fill(str(self.rng.choice(RELATIONS["lives_in"]["fact"])), p, city)
        hop2s = [f"The oldest library in {c} is the {lb} Library." for c, lb in zip(cities, libs)]
        q = f"What is the oldest library in the city where {p} lives?"
        gap = d - self.question_tokens(q)
        # the FARTHER hop is at distance d; the other hop and 2 decoy hop2s sit inside the gap
        first_is_hop1 = bool(self.rng.random() < 0.5)
        far = hop1 if first_is_hop1 else hop2s[0]
        inside = [hop2s[0], hop2s[1], hop2s[2]] if first_is_hop1 else [hop1, hop2s[1], hop2s[2]]
        self.rng.shuffle(inside)
        per = max(20, gap // 4 - 15)
        segs = [self.lead(), far, f"@FILL:{per}"]
        for x in inside:
            segs += [x, f"@FILL:{per}"]
        a = self.assemble(segs, 1, q, lib, d)
        decoys = [f"{lb} Library" for lb in libs[1:]]
        make_lib = lambda r: f"{r.choice(LIBRARY_NAMES)} Library"  # noqa: E731
        return a and self.row("t4", idx, d, a, lib, self.choices(lib, decoys, make_lib),
                              far_hop="hop1" if first_is_hop1 else "hop2")

    # ------------------------------------------------------------------ T5
    def t5(self, base_idx, d, variant=0):
        """Returns 3 matched rows from one base example: clean, irrelevant, near-miss.
        The corrupted copies reuse the clean example's filler; the filler right after
        the injection is shortened by the injection's length, so the distance stays
        (almost) the same and the ONLY real difference is the injected text."""
        name, rel = self.pick_relation(["lives_in", "office_number"])
        title, first, last = str(self.rng.choice(TITLES)), str(self.rng.choice(FIRST)), str(self.rng.choice(LAST))
        k = f"{title} {first} {last}"
        v = rel["val"](self.rng)
        q = fill(rel["q"], k)
        fact = fill(str(self.rng.choice(rel["fact"])), k, v)
        gap = d - self.question_tokens(q)
        cut = int(gap * float(self.rng.uniform(0.3, 0.7)))
        clean = self.assemble([self.lead(), fact, f"@FILL:{cut}", f"@FILL:{gap - cut}"], 1, q, v, d)
        if not clean:
            return []
        lead, _, f1, f2 = clean[3]

        # near-miss: same title + surname, different first name, different value
        nm_first = str(self.rng.choice([f for f in FIRST if f != first]))
        nm_val = self.val_like(rel["val"], v)
        if nm_val is None:
            return []
        near_miss = fill(str(self.rng.choice(rel["fact"])), f"{title} {nm_first} {last}", nm_val)
        n_irr = [2, 5, 10][variant]
        irr = []
        for _ in range(n_irr):
            _, orel = self.pick_relation([r for r in RELATIONS if r != name])
            irr.append(fill(str(orel["fact"][0]), orel["key"](self.rng), orel["val"](self.rng)))

        choices = self.choices(v, [nm_val], rel["val"])
        if choices is None:
            return []
        rows = []
        for kind, inject in [("clean", []), ("irrelevant", irr), ("near_miss", [near_miss])]:
            if inject:
                drop = self.n_tok(" ".join(inject)) + 1
                ids = self.tok.encode(f2).ids[drop:]
                f2_short = self.tok.decode(ids)
                m = re.search(r"[.!?]\s+", f2_short)
                f2_short = f2_short[m.end():] if m and m.end() < 200 else f2_short
                a = self.assemble([lead, fact, f1, " ".join(inject), f2_short], 1, q, v, d,
                                  adjust=False)
            else:
                a = clean
            if not a:
                return []
            rows.append(self.row("t5", base_idx * 3 + len(rows), d, a, v, choices, relation=name,
                                 pair_id=f"t5_{d}_{base_idx:04d}", corruption=kind,
                                 n_injected=len(inject),
                                 injected_value=nm_val if kind == "near_miss" else None))
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--val_bin", default="artifacts/data/val.bin")
    ap.add_argument("--meta", default="artifacts/data/meta.json")
    ap.add_argument("--synth_meta", default=None, help="synth_meta.json, to check no template overlap")
    ap.add_argument("--out_dir", default="artifacts/eval")
    ap.add_argument("--per_bin", type=int, default=100)
    ap.add_argument("--tests", default="t1,t2,t3,t4,t5")
    ap.add_argument("--seed", type=int, default=777)          # different from training synth (2026)
    ap.add_argument("--allow_other_tokenizer", action="store_true", help="for local testing only")
    args = ap.parse_args()

    with open(args.tokenizer, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:12]
    if sha != EXPECTED_SHA and not args.allow_other_tokenizer:
        sys.exit(f"tokenizer sha {sha} != {EXPECTED_SHA}: wrong tokenizer file")
    if args.synth_meta:
        check_no_overlap(args.synth_meta)

    tok = Tokenizer.from_file(args.tokenizer)
    val = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    os.makedirs(args.out_dir, exist_ok=True)
    meta = {"tokenizer_sha": sha, "seed": args.seed, "bins": BINS, "weak_bins": WEAK_BINS,
            "blind_bins": BLIND_BINS, "instruction": INSTRUCTION, "files": {}}

    for ti, test in enumerate(args.tests.split(",")):
        b = Builder(tok, val, seed=args.seed * 100 + ti)     # each test has its own fixed stream
        rows, skipped = [], 0
        for d in BINS:
            n_here = 0
            idx = 0
            while n_here < args.per_bin:
                variant = (n_here // (3 if test == "t5" else 1)) % 3   # even mix of 2/5/10
                if test == "t5":
                    got = b.t5(idx, d, variant)
                    n_add = len(got)
                    rows += got
                else:
                    r = getattr(b, test)(idx, d, variant)
                    got, n_add = r, int(bool(r))
                    if r:
                        rows.append(r)
                skipped += 0 if got else 1
                n_here += n_add
                idx += 1
        path = os.path.join(args.out_dir, f"{test}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(path, "rb") as f:
            meta["files"][f"{test}.jsonl"] = {"rows": len(rows), "skipped": skipped,
                                              "sha": hashlib.sha256(f.read()).hexdigest()[:12]}
        dists = [r["meta"]["distance"] / r["bin"] for r in rows]
        print(f"{test}: {len(rows)} rows, skipped {skipped}, distance/bin in "
              f"[{min(dists):.3f}, {max(dists):.3f}], max tokens {max(r['meta']['n_tokens'] for r in rows)}")
    with open(os.path.join(args.out_dir, "eval_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", args.out_dir)


def check_no_overlap(synth_meta_path):
    """Fail loudly if any training-synth key word, value word or template phrase
    appears in our pools or templates."""
    sm = json.load(open(synth_meta_path))
    strings = []

    def walk(x):
        if isinstance(x, str):
            strings.append(x)
        elif isinstance(x, dict):
            for vv in x.values():
                walk(vv)
        elif isinstance(x, list):
            for vv in x:
                walk(vv)
    walk(sm.get("training_templates", sm))
    ours_text = " ".join(
        [t for r in RELATIONS.values() for t in r["fact"] + r["update"] + [r["q"]]]
        + FIRST + LAST + PROJECTS + STATIONS + SHIPS + CITIES + LIBRARY_NAMES + [INSTRUCTION]
    ).lower()
    clashes = []
    for s in strings:
        plain = re.sub(r"\{[^}]*\}", "", s).strip().lower()
        if "{" in s:     # template: compare its fixed words (e.g. "the combination for")
            for part in re.split(r"\s*[-:>?.,\n]+\s*", plain):
                if len(part.split()) >= 3 and part in ours_text:
                    clashes.append(s)
        elif plain and re.search(rf"\b{re.escape(plain)}\b", ours_text):
            clashes.append(s)
    if clashes:
        sys.exit(f"eval overlaps training synth templates: {sorted(set(clashes))}")
    print(f"overlap check vs {synth_meta_path}: OK ({len(strings)} training strings checked)")


if __name__ == "__main__":
    main()

"""
Is the fact IN the memory, and does the model USE it? A probe inside B and C.

Uses the synthetic question format (same templates as data/make_synth.py, a new seed,
filler from val.bin) and keeps only questions where the fact sits in an EARLIER chunk
than the question, so only memory can deliver it. For each one it runs the model
chunk by chunk, keeps the memory matrix M, and measures:

  STORAGE   (is THIS code in M, distinguishable from the other codes?)
    Binding score for the fact's digit tokens t, against the 3 distractor codes' digits d:
      cos(k_t·M, v_t) − mean cos(k_t·M, v_d)
      (0 = M cannot tell this code from the others; >0 = stored and addressable by its own key)
    store_ideal   M = only the fact sentences, written perfectly with the model's own k/v
                  (no filler, no averaging, no decay): the best this representation allows
    store_fresh   M right after the fact's chunk was written
    store_now     M as it is when the question is read
    -> store_ideal ≈ 0: k/v do not encode which code belongs to which key; no writing rule
       can fix that (representation/learning problem)
    -> store_ideal > 0 but store_fresh ≈ 0: bindable, but drowned by filler (capacity problem)
    -> store_fresh > 0 but store_now ≈ 0: they get in, then are overwritten before the question

  RETRIEVAL (does the question's query find it?)
    r = M_now · q  at the question; compare r with the value vectors of 4 candidates:
    the true code and the 3 distractor codes (all out of the window). Chance = 25%.
    rank1_ideal   question-span query on the IDEAL memory (upper bound for retrieval)
    rank1_last    true fact ranked first, query = last token (the one predicting the answer)
    rank1_qspan   same, query = mean over the question's tokens
    rank1_rand    same, query = a random filler token (sanity: should be ~25%)
    -> rank1 ≈ 25%: memory cannot be addressed by the question (key mismatch)

  USAGE (does the model listen?)
    usage   |memory output| / |h| at the question's last token
    gate_q  C only: read gate at the question's last token; gate_avg: mean over the chunk

    python eval/memory_probe.py --ckpts B.pt,C.pt --data_dir <..>/data --tokenizer <..>/tokenizer.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig  # noqa: E402
from data.make_synth import ENTITIES, FACT_TEMPLATES, QUESTION_TEMPLATES  # noqa: E402
from data.prepare_pretrain import file_sha  # noqa: E402
from model.titans_memory import retrieve  # noqa: E402
from model.transformer import SLM, sliding_window_mask  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402


def find_sub(seq, sub, start=0):
    for i in range(start, len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


def build(tok, val, n_per_bin, bins, seed):
    """Clean questions (no correction), each with 3 distractor codes >= 300 tokens before the
    question, with token spans recorded."""
    rng = np.random.default_rng(seed)
    enc = lambda s: tok.encode(s).ids  # noqa: E731
    code = lambda: str(int(rng.integers(10000, 100000)))  # noqa: E731
    key = lambda: f"{rng.choice(ENTITIES)} {int(rng.integers(1, 1000))}"  # noqa: E731

    def filler(n):
        s = int(rng.integers(0, len(val) - n - 1))
        return val[s:s + n].tolist() if n > 0 else []

    def fact_tokens(k, v):
        t = enc("\n" + str(rng.choice(FACT_TEMPLATES)).format(key=k, val=v) + "\n")
        d = find_sub(t, enc(v))
        assert d >= 0
        return t, (d, d + len(enc(v)))

    qs = []
    for b in bins:
        for _ in range(n_per_bin):
            k, v = key(), code()
            fact, (d0, d1) = fact_tokens(k, v)
            lead = filler(int(rng.integers(0, 65)))
            distr = []
            for _ in range(3):                                   # always 3 distractor codes
                k2 = key()
                while k2 == k:
                    k2 = key()
                distr.append(fact_tokens(k2, code()))
            gap_fill = b - sum(len(t) for t, _ in distr)
            if gap_fill < 0:
                continue
            # keep distractors >= 300 tokens before the question (out of the window, like the fact)
            hi = max(0, gap_fill - 300)
            cuts = sorted(rng.integers(0, hi + 1, size=3).tolist())
            fill = filler(gap_fill)
            pos = len(lead) + len(fact)
            gap, prev, dspans, special = [], 0, [], set()
            for c, (t, (a0, a1)) in zip(cuts, distr):
                gap += fill[prev:c]
                pos += c - prev
                dspans.append((pos + a0, pos + a1))
                special.update(range(pos, pos + len(t)))
                gap += t
                pos += len(t)
                prev = c
            gap += fill[prev:]
            question = enc("\n" + str(rng.choice(QUESTION_TEMPLATES)).format(key=k))
            ctx = lead + fact + gap + question
            f0 = len(lead)
            special.update(range(f0, f0 + len(fact)))
            qs.append({"bin": b, "ctx": ctx, "value_span": (f0 + d0, f0 + d1), "fact_span": (f0, f0 + len(fact)),
                       "distractor_spans": dspans, "special": special,
                       "q_span": (len(ctx) - len(question), len(ctx)), "lead": len(lead)})
    return qs


@torch.no_grad()
def probe_one(model, q, rng, device):
    mem, C = model.memory, model.cfg.chunk_size
    ids = torch.tensor([q["ctx"]], device=device)
    L = ids.shape[1]
    cos, sin = model._rope(L)
    mask = sliding_window_mask(L, model.cfg.attn_window, device)
    h = model.tok_emb(ids)
    for blk in model.lower:
        h = blk(h, cos, sin, mask)
    H, dk, dv = mem.H, mem.dk, mem.dv
    hf = h.float()
    k = F.normalize(mem.W_K(hf).view(1, L, H, dk), dim=-1)
    v = mem.W_V(hf).view(1, L, H, dv)
    qv = F.normalize(mem.W_Q(hf).view(1, L, H, dk), dim=-1)

    c_fact, c_q = (q["fact_span"][1] - 1) // C, (L - 1) // C
    if c_fact >= c_q:
        return None
    state = mem.init_state(1, device)
    M_before, out_last, gate_q, gate_avg = {}, None, None, None
    for ci, s in enumerate(range(0, L, C)):
        M_before[ci] = state.M.clone()
        h_t = h[:, s:s + C]
        if ci == c_q:
            out = mem.read(h_t, state)
            out_last = out[0, (L - 1) - s].float()
            g = mem._raw.get("read_gate")
            if g is not None:
                gate_q, gate_avg = float(g[0, (L - 1) - s].mean()), float(g.mean())
        state = mem.write(h_t, state)
    M_now, M_fresh = M_before[c_q], M_before[c_fact + 1]

    v0, v1 = q["value_span"]
    n = v1 - v0
    dsp = [(a, b) for a, b in q["distractor_spans"] if b <= c_q * C and b - a == n]
    if len(dsp) < 3:
        return None
    fact_toks = list(range(v0, v1))
    other_toks = [t for a, b in dsp[:3] for t in range(a, b)]

    def store(Mx):
        """Binding: does k_t·M point to THIS code's digit vectors rather than the other codes'?
        cos(k_t·M, v_t) − mean cos(k_t·M, v_d) over the distractor codes' digits d.
        0 = M cannot tell this code from the others; >0 = this code is stored and addressable."""
        tt, oo = torch.tensor(fact_toks, device=device), torch.tensor(other_toks, device=device)
        r = F.normalize(retrieve(k[:, tt], Mx)[0], dim=-1)               # (n,H,dv)
        vt = F.normalize(v[0, tt], dim=-1)
        vo = F.normalize(v[0, oo], dim=-1)
        return float((r * vt).sum(-1).mean() - torch.einsum("nhd,mhd->nmh", r, vo).mean())

    # IDEAL memory from the model's OWN keys/values: only fact sentences written (target + 3 distractors),
    # summed, no filler, no averaging, no decay. An upper bound on what this k/v representation allows.
    ideal = torch.tensor(sorted(t for t in q["special"] if t < c_q * C), device=device)
    M_ideal = torch.einsum("bthk,bthv->bhkv", k[:, ideal], v[:, ideal])
    res = {"store_ideal": store(M_ideal), "store_fresh": store(M_fresh), "store_now": store(M_now)}
    cands = [fact_toks] + [list(range(a, b)) for a, b in dsp[:3]]       # true code + 3 other codes
    cvec = torch.stack([v[0, c].mean(0).reshape(-1) for c in cands])     # (4, H*dv)

    def rank1(query, Mx=None):                                           # query (H,dk)
        r = retrieve(query.view(1, 1, H, dk), M_now if Mx is None else Mx).reshape(-1)
        return float(F.cosine_similarity(r.unsqueeze(0), cvec, dim=-1).argmax() == 0)

    pool = [t for t in range(q["lead"], c_q * C) if t not in q["special"]]
    qa, qb = q["q_span"]
    res["rank1_ideal"] = rank1(F.normalize(qv[0, qa:qb].mean(0), dim=-1), M_ideal)
    res["rank1_last"] = rank1(qv[0, L - 1])
    res["rank1_qspan"] = rank1(F.normalize(qv[0, qa:qb].mean(0), dim=-1))
    rt = int(rng.choice(pool))
    res["rank1_rand"] = rank1(qv[0, rt])
    res["usage"] = float(out_last.norm() / h[0, L - 1].float().norm())
    res["gate_q"], res["gate_avg"] = gate_q, gate_avg
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-separated B and C checkpoints (pretrained)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", default="memory_probe")
    ap.add_argument("--n_per_bin", type=int, default=100)
    ap.add_argument("--bins", default="300,500,700,900,1500,2500")
    ap.add_argument("--seed", type=int, default=9191)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tok = Tokenizer.from_file(args.tokenizer)
    val = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    bins = [int(b) for b in args.bins.split(",")]
    qs = build(tok, val, args.n_per_bin, bins, args.seed)
    print(f"{len(qs)} clean questions | bins {args.bins}")
    cols = ["store_ideal", "store_fresh", "store_now", "rank1_ideal", "rank1_last", "rank1_qspan", "rank1_rand", "usage",
            "gate_q", "gate_avg"]
    allres = {}
    for p in args.ckpts.split(","):
        ck = torch.load(p.strip(), map_location="cpu", weights_only=False)
        assert ck.get("tokenizer_sha") == file_sha(args.tokenizer), "tokenizer mismatch"
        m = SLM(ModelConfig(**ck["model_config"]))
        m.load_state_dict(ck["model"])
        m = m.to(args.device).float().eval()
        if m.memory.is_null:
            print(f"skip {os.path.basename(p)}: no memory")
            continue
        label = ck.get("variant", "?")
        rng = np.random.default_rng(args.seed + 1)
        rows = {b: [] for b in bins}
        for q in qs:
            r = probe_one(m, q, rng, args.device)
            if r is not None:
                rows[q["bin"]].append(r)
        print(f"\nVARIANT {label}  (cos: 1 = perfect recall, 0 = none | rank1: chance 25% | usage: memory/|h|)")
        print(f"{'bin':>6} {'n':>4} " + " ".join(f"{c:>12}" for c in cols))
        table = {}
        for b in bins:
            if not rows[b]:
                continue
            agg = {}
            for c in cols:
                vals = [r[c] for r in rows[b] if r[c] is not None and not np.isnan(r[c])]
                agg[c] = (100 * np.mean(vals) if c.startswith("rank1") else np.mean(vals)) if vals else float("nan")
            table[b] = agg
            print(f"{b:>6} {len(rows[b]):>4} " + " ".join(
                f"{agg[c]:>11.1f}%" if c.startswith("rank1") else f"{agg[c]:>12.3f}" for c in cols))
        allres[label] = {str(b): t for b, t in table.items()}
        del m
    json.dump(allres, open(os.path.join(args.out_dir, "memory_probe.json"), "w"), indent=2)
    print(f"\nsaved -> {args.out_dir}/memory_probe.json")


if __name__ == "__main__":
    main()

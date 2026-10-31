"""
T6 — Efficiency: what does memory cost, compared with simply widening attention?
See docs/EVAL_SPEC.md section 3 (T6).

    python eval/efficiency.py --out_dir results                       # A, B, C (whatever is implemented) + full-attention reference
    python eval/efficiency.py --ckpts A=ckpt_A.pt B=ckpt_B.pt ...      # same, but with real weights (speed doesn't depend on them)

Measured per (model, seq_len, batch):
    tokens_per_sec   forward-pass throughput
    ms_per_seq       latency for one sequence
    peak_mem_mb      peak GPU memory during the forward pass
Theoretical (computed, not timed):
    attn_gflops      attention score + mix FLOPs per sequence, counting only the keys
                     each query may see (window W, or all earlier tokens)
    kv_cache_mb      keys/values that must be kept to generate the NEXT token at
                     position T (fp16). Local attention keeps only the last W.
    mem_state_mb     size of the memory module's state (B/C), fp16

WHY BOTH: the backbone implements the sliding window as a MASK over full
attention (scaled_dot_product_attention + attn_mask), so it still computes all
T x T scores and the measured speed of A equals full attention. The theoretical
columns show what local attention + memory costs with a kernel that skips the
masked blocks (e.g. FlashAttention's window option). Report both, and say which
is which.

"full_attn_4096" = the same backbone with attn_window = 4096 (every token sees
every earlier token). It is untrained: it is a COST reference only. Quality
comes from T1-T5.
Run on the same GPU type for every model (Kaggle T4), nothing else running.
"""
import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig  # noqa: E402
from model.transformer import SLM  # noqa: E402


def theoretical(model, T):
    cfg = model.cfg
    L, d = cfg.n_layers, cfg.d_model
    W = min(cfg.attn_window, T)
    keys_seen = sum(min(i + 1, W) for i in range(T))           # total (query, key) pairs
    attn_gflops = 4 * L * d * keys_seen / 1e9                  # QK^T and AV, 2 FLOPs per MAC
    kv_cache_mb = 2 * L * min(W, T) * d * 2 / 2**20            # K and V, fp16
    mem = 0.0
    if not model.memory.is_null:
        state = model.memory.init_state(1, "cpu", torch.float16)
        tensors = [state] if torch.is_tensor(state) else (
            list(state.values()) if isinstance(state, dict) else list(state))
        mem = sum(t.numel() for t in tensors if torch.is_tensor(t)) * 2 / 2**20
    return {"attn_gflops": round(attn_gflops, 2), "kv_cache_mb": round(kv_cache_mb, 2),
            "mem_state_mb": round(mem, 2)}


def build(name, ckpt_path):
    if ckpt_path:                                    # real checkpoint
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = SLM(ModelConfig(**ck["model_config"]))
        m.load_state_dict(ck["model"])
        return m
    cfg = ModelConfig()
    if name == "full_attn_4096":
        cfg.attn_window = 4096                       # no sliding window, no memory
    else:
        cfg.memory = {**cfg.memory, "variant": name}
    return SLM(cfg)


@torch.no_grad()
def measure(model, T, B, device, reps):
    x = torch.randint(5, 8192, (B, T), device=device)
    amp = device == "cuda"
    for _ in range(2):                               # warm-up (kernel selection, caches)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            model(x)
    if amp:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            model(x)
    if amp:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    peak = torch.cuda.max_memory_allocated() / 2**20 if amp else float("nan")
    return {"tokens_per_sec": round(B * T / dt), "ms_per_seq": round(1000 * dt / B, 2),
            "peak_mem_mb": round(peak, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="A,B,C,full_attn_4096")
    ap.add_argument("--ckpts", nargs="*", default=[], help="NAME=path, e.g. A=variant-A.pt")
    ap.add_argument("--lengths", default="1024,2048,4096")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU, numbers are only for testing the script (memory is not measured)")
    ckpts = dict(c.split("=", 1) for c in args.ckpts)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "efficiency.csv")
    gpu = torch.cuda.get_device_name() if device == "cuda" else "cpu"

    rows = []
    for name in args.models.split(","):
        try:
            model = build(name, ckpts.get(name)).to(device).eval()
        except NotImplementedError as e:             # B/C not plugged in yet
            print(f"skip {name}: {e}")
            continue
        n_params = sum(p.numel() for p in model.parameters())
        for T in map(int, args.lengths.split(",")):
            for B in map(int, args.batches.split(",")):
                try:
                    r = measure(model, T, B, device, args.reps)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    r = {"tokens_per_sec": "OOM", "ms_per_seq": "OOM", "peak_mem_mb": "OOM"}
                row = {"model": name, "params_m": round(n_params / 1e6, 2), "seq_len": T, "batch": B,
                       "device": gpu, **r, **theoretical(model, T)}
                rows.append(row)
                print(row, flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()

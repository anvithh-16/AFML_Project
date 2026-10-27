"""
Pretrain one variant (A, B or C) on the shared FineWeb-Edu tokens.

Kaggle (GPU T4, Internet off is fine):
    python train/pretrain.py --variant A \
        --data_dir /kaggle/input/<AFML_Project_Data>/artifacts/data \
        --out_dir /kaggle/working/ckpt_A --time_limit_min 690

Resume in a NEW session (attach the previous run's output as an input first):
    python train/pretrain.py --variant A --data_dir ... --out_dir /kaggle/working/ckpt_A \
        --resume /kaggle/input/<previous-output>/ckpt_A/latest.pt

Local smoke test (CPU, tiny model, ~1 min):
    python train/pretrain.py --tiny --data_dir <small data> --out_dir /tmp/ckpt --max_steps 60

What it does
  * AdamW (decay on matrices only), linear warmup then cosine decay to min_lr.
  * fp16 autocast + GradScaler on CUDA (the T4 has no fast bf16).
  * Every --eval_every steps: val loss on a FIXED set of val windows, so curves
    from A, B and C are directly comparable.
  * Every --ckpt_every steps: latest.pt (model + optimizer + scaler + RNG +
    loader state), written atomically. Resume continues the exact same run.
  * --time_limit_min: saves and exits cleanly before Kaggle kills the session.
  * At the end: variant-<X>_pretrain_<date>.pt = weights + config only
    (the team's checkpoint naming rule), plus log.csv.
"""
import argparse
import csv
import datetime
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DataConfig, ModelConfig, PretrainConfig  # noqa: E402
from data.loader import TokenLoader  # noqa: E402
from model.transformer import SLM  # noqa: E402


def parse_args():
    p, m = PretrainConfig(), ModelConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A", choices=["A", "B", "C"])
    ap.add_argument("--data_dir", default=DataConfig().out_dir)
    ap.add_argument("--synth_dir", default=None,
                    help="synthetic long-range examples (data/make_synth.py); team default: use them")
    ap.add_argument("--synth_frac", type=float, default=DataConfig().synth_frac,
                    help="fraction of training rows that start with a synthetic example")
    ap.add_argument("--out_dir", default="checkpoints/pretrain_A")
    ap.add_argument("--resume", default="auto",
                    help="'auto' = out_dir/latest.pt if present, 'none', or a path to latest.pt")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="default: target_tokens / tokens_per_step (~4,577)")
    ap.add_argument("--batch_size", type=int, default=p.batch_size)
    ap.add_argument("--grad_accum", type=int, default=p.grad_accum)
    ap.add_argument("--seq_len", type=int, default=m.train_seq_len)
    ap.add_argument("--lr", type=float, default=p.lr)
    ap.add_argument("--min_lr", type=float, default=p.min_lr)
    ap.add_argument("--warmup_steps", type=int, default=p.warmup_steps)
    ap.add_argument("--weight_decay", type=float, default=p.weight_decay)
    ap.add_argument("--grad_clip", type=float, default=p.grad_clip)
    ap.add_argument("--eval_every", type=int, default=p.eval_every)
    ap.add_argument("--eval_batches", type=int, default=40)
    ap.add_argument("--ckpt_every", type=int, default=p.ckpt_every)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--time_limit_min", type=float, default=None)
    ap.add_argument("--stop_after_steps", type=int, default=None,
                    help="save and exit after N steps in THIS session (testing / splitting runs)")
    ap.add_argument("--seed", type=int, default=p.seed)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compile", action="store_true", help="torch.compile (faster on GPU; slow first step)")
    ap.add_argument("--tiny", action="store_true", help="tiny model for smoke tests")
    return ap.parse_args()


def model_config(args, data_vocab):
    cfg = ModelConfig()
    if args.tiny:
        cfg = ModelConfig(vocab_size=data_vocab, n_layers=2, d_model=64, n_heads=4, ffn_hidden=192,
                          train_seq_len=args.seq_len, attn_window=64, chunk_size=64,
                          eval_max_seq_len=max(256, args.seq_len))
    else:
        assert data_vocab == cfg.vocab_size, f"data vocab {data_vocab} != model vocab {cfg.vocab_size}"
        cfg.train_seq_len = args.seq_len
    cfg.memory = dict(cfg.memory, variant=args.variant)
    if args.tiny:
        cfg.memory["insert_after_layer"] = cfg.n_layers // 2
    return cfg


def lr_at(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = min(1.0, (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model, args, device):
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=args.lr, betas=PretrainConfig().betas,
                             fused=device.startswith("cuda"))


@torch.no_grad()
def evaluate(model, loader, n_batches, amp_ctx):
    model.eval()
    losses = []
    for x, y in loader.val_batches(max_batches=n_batches):
        with amp_ctx():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = args.device
    use_amp = device.startswith("cuda")
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True

    loader = TokenLoader(args.data_dir, args.seq_len, args.batch_size, device=device, seed=args.seed,
                         synth_dir=args.synth_dir, synth_frac=args.synth_frac if args.synth_dir else 0.0)
    meta = loader.meta
    synth_sha = loader.synth_meta["synth_sha"] if loader.synth_meta else None
    cfg = model_config(args, meta["vocab_size"])
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    if args.max_steps is None:
        args.max_steps = min(DataConfig().target_tokens, meta["tokens"]["train"]) // tokens_per_step

    model = SLM(cfg).to(device)
    opt = make_optimizer(model, args, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def amp_ctx():
        return torch.autocast("cuda", dtype=torch.float16) if use_amp else torch.autocast("cpu", enabled=False)

    # ---- resume -----------------------------------------------------------
    step, best_val, log_rows = 0, float("inf"), []
    resume_path = None
    if args.resume == "auto" and os.path.exists(os.path.join(args.out_dir, "latest.pt")):
        resume_path = os.path.join(args.out_dir, "latest.pt")
    elif args.resume not in ("auto", "none"):
        resume_path = args.resume
    if resume_path:
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        assert ck["tokenizer_sha"] == meta["tokenizer_sha"], "checkpoint was trained on different data"
        assert ck["model_config"] == asdict(cfg), "model config changed since this checkpoint"
        assert ck.get("synth_sha") == synth_sha and ck.get("synth_frac", 0.0) == loader.synth_frac, \
            "synthetic mix differs from this checkpoint's run (check --synth_dir / --synth_frac)"
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, best_val = ck["step"], ck["best_val"]
        loader.rng.bit_generator.state = ck["loader_rng"]
        torch.set_rng_state(ck["torch_rng"])
        if use_amp and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"])
        log_rows = ck.get("log", [])
        print(f"resumed from {resume_path} at step {step}")

    train_model = torch.compile(model) if args.compile else model

    def checkpoint(tag="latest"):
        save_atomic({
            "model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
            "step": step, "best_val": best_val, "model_config": asdict(cfg), "args": vars(args),
            "tokenizer_sha": meta["tokenizer_sha"], "data_tokens": meta["tokens"],
            "synth_sha": synth_sha, "synth_frac": loader.synth_frac,
            "loader_rng": loader.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if use_amp else None, "log": log_rows,
        }, os.path.join(args.out_dir, f"{tag}.pt"))
        with open(os.path.join(args.out_dir, "log.csv"), "w", newline="") as f:
            if log_rows:
                w = csv.DictWriter(f, fieldnames=list(log_rows[-1].keys()))
                w.writeheader()
                w.writerows(log_rows)

    print(f"variant {args.variant} | {model.num_params() / 1e6:.2f}M params | device {device} | "
          f"{tokens_per_step:,} tok/step | {args.max_steps:,} steps | data sha {meta['tokenizer_sha']} | "
          + (f"synth {loader.synth_frac:.0%} of rows (sha {synth_sha})" if synth_sha else "synth OFF"))

    # ---- train ------------------------------------------------------------
    t_start = t_last = time.time()
    steps_this_session, running = 0, []
    model.train()
    while step < args.max_steps:
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x, y = loader.get_batch("train")
            with amp_ctx():
                _, loss = train_model(x, y)
            scaler.scale(loss / args.grad_accum).backward()
            loss_acc += loss.item() / args.grad_accum
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        scaler.step(opt)
        scaler.update()
        step += 1
        steps_this_session += 1
        running.append(loss_acc)

        if not math.isfinite(loss_acc):
            print(f"step {step}: loss is {loss_acc}; stopping (check lr / data)")
            checkpoint("crash")
            sys.exit(1)

        do_eval = step % args.eval_every == 0 or step == args.max_steps
        if step % args.log_every == 0 or do_eval:
            now = time.time()
            n_steps = args.log_every if step % args.log_every == 0 else len(running)
            tps = n_steps * tokens_per_step / max(now - t_last, 1e-9)
            t_last = now
            row = {"step": step, "tokens": step * tokens_per_step, "train_loss": round(float(np.mean(running)), 4),
                   "val_loss": "", "lr": f"{lr:.2e}", "grad_norm": round(gnorm, 3),
                   "tok_per_s": int(tps), "elapsed_min": round((now - t_start) / 60, 1)}
            row.update({f"mem_{k}": v for k, v in model.memory.stats().items()})
            running = []
            if do_eval:
                row["val_loss"] = round(evaluate(model, loader, args.eval_batches, amp_ctx), 4)
                if row["val_loss"] < best_val:
                    best_val = row["val_loss"]
            log_rows.append(row)
            eta = (args.max_steps - step) * tokens_per_step / max(tps, 1) / 60
            print(f"step {step:>5}/{args.max_steps} | loss {row['train_loss']:.4f}"
                  + (f" | val {row['val_loss']:.4f}" if do_eval else "")
                  + f" | lr {lr:.1e} | gnorm {gnorm:.2f} | {tps / 1e3:.1f}k tok/s | eta {eta:.0f} min",
                  flush=True)

        if step % args.ckpt_every == 0:
            checkpoint()
        out_of_time = args.time_limit_min and (time.time() - t_start) / 60 > args.time_limit_min
        if out_of_time or (args.stop_after_steps and steps_this_session >= args.stop_after_steps):
            checkpoint()
            print(f"stopping at step {step} ({'time limit' if out_of_time else 'stop_after_steps'}); "
                  f"resume with --resume {os.path.join(args.out_dir, 'latest.pt')}")
            return

    checkpoint()
    final = os.path.join(args.out_dir, f"variant-{args.variant}_pretrain_{datetime.date.today():%Y%m%d}.pt")
    save_atomic({"model": model.state_dict(), "model_config": asdict(cfg), "variant": args.variant,
                 "stage": "pretrain", "step": step, "best_val": best_val,
                 "tokenizer_sha": meta["tokenizer_sha"], "synth_sha": synth_sha,
                 "synth_frac": loader.synth_frac}, final)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"variant": args.variant, "steps": step, "tokens": step * tokens_per_step,
                   "final_val_loss": log_rows[-1]["val_loss"] if log_rows else None,
                   "best_val_loss": best_val, "checkpoint": os.path.basename(final)}, f, indent=2)
    print(f"done: {step} steps, best val {best_val:.4f} -> {final}")


if __name__ == "__main__":
    main()

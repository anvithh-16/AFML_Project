"""
Instruction-tune a pretrained variant on Dolly-15k. Same settings for A, B, C.

Kaggle (GPU T4; inputs: code, AFML_Project_Data, the instruct data, the pretrain checkpoint):
    python train/instruct.py --init <...>/variant-A_pretrain_<date>.pt \
        --data_dir <...>/artifacts/instruct --out_dir /kaggle/working/instruct_A

What it does
  * Loads the pretrained weights; refuses if the tokenizer differs.
  * Batches are padded on the right with <|pad|>; padding and prompt tokens get
    label -100, so the loss is computed ONLY on the response + <|endoftext|>.
    (The model still reads the prompt; it just isn't graded on predicting it.)
  * AdamW, warmup then cosine decay, fp16 on GPU. ~3 epochs over ~14.5k examples
    takes roughly 10-20 min on a T4.
  * Val loss (token-weighted over all response tokens) every --eval_every steps;
    the best checkpoint is kept as variant-<X>_instruct_<date>.pt.
  * samples.md: greedy answers to the same val prompts BEFORE and AFTER tuning
    (handy for the report).
"""
import argparse
import csv
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig  # noqa: E402
from data.instruct_format import encode_prompt  # noqa: E402
from model.transformer import SLM  # noqa: E402
from train.pretrain import save_atomic  # noqa: E402

PAD_ID, EOT_ID = 1, 0   # fixed by config.SPECIAL_TOKENS (asserted when the tokenizer is trained)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="variant-X_pretrain_<date>.pt")
    ap.add_argument("--data_dir", default="artifacts/instruct")
    ap.add_argument("--tokenizer", default=None, help="only needed for samples.md")
    ap.add_argument("--out_dir", default="checkpoints/instruct")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def make_labels(ids, prompt_len):
    return [-100] * prompt_len + ids[prompt_len:]


def collate(ids_list, plen_list, device):
    """Right-pad to the longest example. x = ids[:-1], y = labels[1:]."""
    T = max(len(ids) for ids in ids_list) - 1
    x = torch.full((len(ids_list), T), PAD_ID, dtype=torch.long)
    y = torch.full((len(ids_list), T), -100, dtype=torch.long)
    for i, (ids, p) in enumerate(zip(ids_list, plen_list)):
        lab = make_labels(ids, p)
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1])
        y[i, :n] = torch.tensor(lab[1:])
    return x.to(device), y.to(device)


def batches(data, batch_size, rng):
    order = rng.permutation(len(data["ids"]))
    for s in range(0, len(order), batch_size):
        b = order[s:s + batch_size]
        yield [data["ids"][i] for i in b], [data["prompt_len"][i] for i in b]


@torch.no_grad()
def evaluate(model, data, batch_size, device, amp_ctx):
    """Token-weighted mean loss over every response token in val."""
    model.eval()
    total, count = 0.0, 0
    ids, plens = data["ids"], data["prompt_len"]
    order = np.argsort([len(x) for x in ids])           # similar lengths together = less padding
    for s in range(0, len(order), batch_size):
        b = order[s:s + batch_size]
        x, y = collate([ids[i] for i in b], [plens[i] for i in b], device)
        with amp_ctx():
            logits, _ = model(x)
        total += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                                 ignore_index=-100, reduction="sum").item()
        count += (y != -100).sum().item()
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def sample_answers(model, tok, raws, device, max_new_tokens):
    model.eval()
    outs = []
    for r in raws:
        p = torch.tensor([encode_prompt(tok, r["instruction"], r.get("context", ""))], device=device)
        gen = model.generate(p, max_new_tokens, temperature=0.0, eos_id=EOT_ID)[0, p.shape[1]:].tolist()
        if EOT_ID in gen:
            gen = gen[:gen.index(EOT_ID)]
        outs.append(tok.decode(gen).strip())
    model.train()
    return outs


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device, use_amp = args.device, args.device.startswith("cuda")

    def amp_ctx():
        return torch.autocast("cuda", dtype=torch.float16) if use_amp else torch.autocast("cpu", enabled=False)

    # ---- load ----------------------------------------------------------------
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    assert ck["tokenizer_sha"] == meta["tokenizer_sha"], \
        f"pretrain tokenizer {ck['tokenizer_sha']} != instruct data tokenizer {meta['tokenizer_sha']}"
    cfg = ModelConfig(**ck["model_config"])
    variant = ck.get("variant", cfg.memory.get("variant", "A"))
    model = SLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    train = torch.load(os.path.join(args.data_dir, "train.pt"))
    val = torch.load(os.path.join(args.data_dir, "val.pt"))

    tok = None
    tok_path = args.tokenizer or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.data_dir))),
                                              "tokenizer", "tokenizer.json")
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_path)
    else:
        print(f"(no tokenizer at {tok_path}; pass --tokenizer to get samples.md)")

    steps_per_epoch = math.ceil(len(train["ids"]) / (args.batch_size * args.grad_accum))
    max_steps = int(steps_per_epoch * args.epochs)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        prog = min(1.0, (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))

    print(f"variant {variant} | init {os.path.basename(args.init)} | train {len(train['ids']):,} / "
          f"val {len(val['ids']):,} examples | {steps_per_epoch} steps/epoch x {args.epochs} = {max_steps} steps")

    sample_raws = val["raw"][:args.n_samples]
    before = sample_answers(model, tok, sample_raws, device, args.max_new_tokens) if tok else None
    best_val = evaluate(model, val, args.batch_size * 2, device, amp_ctx)
    print(f"val loss before tuning: {best_val:.4f}")
    log_rows = [{"step": 0, "epoch": 0, "train_loss": "", "val_loss": round(best_val, 4), "lr": "", "elapsed_min": 0}]
    best_val = float("inf")   # keep the best AFTER tuning starts
    final = os.path.join(args.out_dir, f"variant-{variant}_instruct_{datetime.date.today():%Y%m%d}.pt")

    # ---- train ---------------------------------------------------------------
    step, t0, running = 0, time.time(), []
    model.train()
    stream = iter(())
    epoch = 0
    while step < max_steps:
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.grad_accum):
            try:
                ids, plens = next(stream)
            except StopIteration:
                epoch += 1
                stream = batches(train, args.batch_size, rng)
                ids, plens = next(stream)
            x, y = collate(ids, plens, device)
            with amp_ctx():
                _, loss = model(x, y)
            scaler.scale(loss / args.grad_accum).backward()
            acc += loss.item() / args.grad_accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1
        running.append(acc)

        do_eval = step % args.eval_every == 0 or step == max_steps
        if step % args.log_every == 0 or do_eval:
            row = {"step": step, "epoch": round(step / steps_per_epoch, 2),
                   "train_loss": round(float(np.mean(running)), 4), "val_loss": "",
                   "lr": f"{lr:.2e}", "elapsed_min": round((time.time() - t0) / 60, 1)}
            running = []
            if do_eval:
                row["val_loss"] = round(evaluate(model, val, args.batch_size * 2, device, amp_ctx), 4)
                if row["val_loss"] < best_val:
                    best_val = row["val_loss"]
                    save_atomic({"model": model.state_dict(), "model_config": ck["model_config"],
                                 "variant": variant, "stage": "instruct", "step": step, "val_loss": best_val,
                                 "tokenizer_sha": meta["tokenizer_sha"], "init": os.path.basename(args.init),
                                 "args": vars(args)}, final)
            log_rows.append(row)
            print(f"step {step:>4}/{max_steps} | ep {row['epoch']:.2f} | loss {row['train_loss']:.4f}"
                  + (f" | val {row['val_loss']:.4f}" + (" *best" if row["val_loss"] == best_val else "")
                     if do_eval else "") + f" | lr {lr:.1e}", flush=True)

    with open(os.path.join(args.out_dir, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[-1].keys()))
        w.writeheader()
        w.writerows(log_rows)

    # ---- before/after samples from the best checkpoint ----------------------
    if tok:
        model.load_state_dict(torch.load(final, map_location=device, weights_only=False)["model"])
        after = sample_answers(model, tok, sample_raws, device, args.max_new_tokens)
        with open(os.path.join(args.out_dir, "samples.md"), "w") as f:
            f.write(f"# Variant {variant}: greedy answers on held-out Dolly prompts\n\n")
            for r, b, a in zip(sample_raws, before, after):
                f.write(f"## {r['instruction']}\n\n")
                if r.get("context"):
                    f.write(f"Context: {r['context'][:300]}{'...' if len(r['context']) > 300 else ''}\n\n")
                f.write(f"Reference: {r['response']}\n\nBefore tuning: {b}\n\nAfter tuning: {a}\n\n")
        print(f"\nsample after tuning:\n  Q: {sample_raws[0]['instruction']}\n  A: {after[0][:300]}")

    json.dump({"variant": variant, "steps": max_steps, "best_val_loss": best_val,
               "val_loss_before": log_rows[0]["val_loss"], "checkpoint": os.path.basename(final)},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print(f"done: best val {best_val:.4f} (before tuning {log_rows[0]['val_loss']}) -> {final}")


if __name__ == "__main__":
    main()

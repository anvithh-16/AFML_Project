"""
Train the shared byte-level BPE tokenizer (used unchanged by Variants A, B, C).

Colab / Kaggle:
    python tokenizer/train_tokenizer.py
Local smoke test (no internet):
    python tokenizer/train_tokenizer.py --local_text some.txt --vocab_size 1024

Design choices (worth explaining in the report):
  * Byte-level BPE: no unknown tokens, handles any Dolly / eval text.
  * 8192 vocab: a 30M model can't afford a 50k vocab (50k*512 = 25M params in
    embeddings alone). 8k keeps embeddings ~14% of parameters.
  * Digits split individually: needle-in-context tests ask the model to copy
    exact numbers; per-digit tokens make copying a uniform per-token skill
    instead of memorising thousands of multi-digit tokens.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPECIAL_TOKENS, TokenizerConfig, DataConfig  # noqa: E402

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402


def hf_text_iterator(n_docs: int, cfg: DataConfig):
    from datasets import load_dataset
    ds = load_dataset(cfg.hf_dataset, name=cfg.hf_subset, split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= n_docs:
            break
        # Skip the val docs so the tokenizer never sees validation text
        if i % cfg.val_every_n_docs == 0:
            continue
        yield ex["text"]


def local_text_iterator(path: str):
    with open(path, encoding="utf-8") as f:
        buf = []
        for line in f:
            if line.strip() == "" and buf:
                yield "".join(buf)
                buf = []
            else:
                buf.append(line)
        if buf:
            yield "".join(buf)


def build_tokenizer(vocab_size: int, min_frequency: int, individual_digits: bool):
    tok = Tokenizer(models.BPE())
    pre = [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)]
    if individual_digits:
        pre.insert(0, pre_tokenizers.Digits(individual_digits=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence(pre)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,          # ids 0..4 in this exact order
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    return tok, trainer


def main():
    tcfg, dcfg = TokenizerConfig(), DataConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=tcfg.vocab_size)
    ap.add_argument("--train_docs", type=int, default=tcfg.train_docs)
    ap.add_argument("--local_text", type=str, default=None)
    ap.add_argument("--out", type=str, default=tcfg.path)
    args = ap.parse_args()

    tok, trainer = build_tokenizer(args.vocab_size, tcfg.min_frequency, tcfg.individual_digits)
    it = local_text_iterator(args.local_text) if args.local_text else hf_text_iterator(args.train_docs, dcfg)
    # Download first, train second: feeding the HF streaming reader straight
    # into the Rust trainer can crash silently, so materialise the docs.
    print("[1/3] collecting docs...", flush=True)
    texts = list(it)
    print(f"      got {len(texts):,} docs, {sum(map(len, texts)) / 1e6:.1f}M chars", flush=True)
    print("[2/3] training BPE...", flush=True)
    tok.train_from_iterator(texts, trainer=trainer, length=len(texts))
    del texts
    print("[3/3] trained, checking and saving", flush=True)

    # Sanity: special tokens must sit at fixed ids so every lane can hardcode them
    for i, s in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(s) == i, f"{s} got id {tok.token_to_id(s)}, expected {i}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tok.save(args.out)
    with open(os.path.join(os.path.dirname(args.out), "tokenizer_meta.json"), "w") as f:
        json.dump({"vocab_size": tok.get_vocab_size(), "special_tokens": SPECIAL_TOKENS,
                   "individual_digits": tcfg.individual_digits,
                   "source": args.local_text or f"{dcfg.hf_dataset}/{dcfg.hf_subset}",
                   "train_docs": None if args.local_text else args.train_docs}, f, indent=2)

    sample = "The secret code for Room 12 is 4821.\n<|user|>\nWhat is it?\n<|assistant|>\n4821<|endoftext|>"
    enc = tok.encode(sample)
    print(f"vocab={tok.get_vocab_size()}  sample -> {len(enc.ids)} tokens")
    print(enc.tokens)
    assert tok.decode(enc.ids, skip_special_tokens=False) == sample, "round-trip failed"
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
    # Exit at once: HF streaming leaves a prefetch thread that can keep the
    # process alive after the work is saved (would hang a Kaggle commit).
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)

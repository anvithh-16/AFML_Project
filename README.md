# AFML Team 30 — Memory-augmented SLM

Lane A (data + backbone) files so far:

| File | What it does |
|---|---|
| `config.py` | Shared hyperparameters for A/B/C (run it to print size and FLOPs) |
| `tokenizer/train_tokenizer.py` | Trains the shared 8k byte-level BPE |
| `data/prepare_pretrain.py` | Streams FineWeb-Edu into `train.bin` / `val.bin` (resumable) |
| `data/loader.py` | Memmap batch loader used by every training script |
| `docs/ARCHITECTURE_SPEC.md` | Handoff to Person B: dims, context, memory plug-in interface |

## Run on Colab / Kaggle

```bash
pip install -r requirements.txt
python tokenizer/train_tokenizer.py              # ~10-20 min, 200k docs
python data/prepare_pretrain.py                  # 600M tokens, ~1.2 GB; add --resume if the session dies
cp -r artifacts /content/drive/MyDrive/afml30/   # always keep a Drive copy
```

Quick slice for debugging: `python data/prepare_pretrain.py --target_tokens 20_000_000`.

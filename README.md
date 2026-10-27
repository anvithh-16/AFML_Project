# AFML Team 30 — Memory-augmented SLM

| File | What it does |
|---|---|
| `config.py` | Shared hyperparameters for A/B/C (run it to print size and FLOPs) |
| `tokenizer/train_tokenizer.py` | Trains the shared 8k byte-level BPE |
| `data/prepare_pretrain.py` | Streams FineWeb-Edu into `train.bin` / `val.bin` (resumable) |
| `data/loader.py` | Memmap batch loader used by every training script |
| `model/transformer.py` | Decoder backbone: RMSNorm, RoPE, SwiGLU, sliding-window attention, memory hook |
| `model/memory.py` | Memory interface + `NullMemory` (Variant A); Person B adds B and C here |
| `train/pretrain.py` | Pretraining with fp16, cosine LR, val loss, exact resume, Kaggle time limit |
| `tests/test_model.py` | Causality, receptive field, param count, memory-hook tests (CPU, ~10 s) |
| `docs/ARCHITECTURE_SPEC.md` | Dims, context, memory interface, shared settings |

Data is already built: Kaggle dataset **AFML_Project_Data** (`tokenizer_sha 7f469545b846`).

## Pretrain on Kaggle (Accelerator: GPU T4, Internet can be off)

Attach two inputs: the code dataset and `AFML_Project_Data`. Then:

```python
!find /kaggle/input -maxdepth 6 -name "config.py" -o -maxdepth 6 -name "meta.json"
CODE = "/kaggle/input/.../afml-slm"            # from find
DATA = "/kaggle/input/.../artifacts/data"      # folder holding meta.json
!cp -r {CODE} /kaggle/working/afml-slm
%cd /kaggle/working/afml-slm
!python tests/test_model.py
```

1. **Throughput check (~3 min):** measures tok/s and confirms fp16 works on the GPU.
   ```python
   !python train/pretrain.py --variant A --data_dir {DATA} --out_dir /kaggle/working/bench \
       --max_steps 30 --eval_every 30 --log_every 10 --resume none
   ```
2. **Full run** as a background commit (Save Version → Save & Run All):
   ```python
   !python train/pretrain.py --variant A --data_dir {DATA} --out_dir /kaggle/working/ckpt_A --time_limit_min 690
   ```
3. If it stops on the time limit, attach that version's output as an input in a new
   session and add `--resume /kaggle/input/<that output>/ckpt_A/latest.pt`.

Outputs in `ckpt_A/`: `latest.pt` (full state), `variant-A_pretrain_<date>.pt`
(weights + config, for instruction tuning and eval), `log.csv`, `summary.json`.

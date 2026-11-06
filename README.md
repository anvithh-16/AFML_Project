# AFML Team 30 — Memory-augmented SLM

| File | What it does |
|---|---|
| `config.py` | Shared hyperparameters for A/B/C (run it to print size and FLOPs) |
| `tokenizer/train_tokenizer.py` | Trains the shared 8k byte-level BPE |
| `data/prepare_pretrain.py` | Streams FineWeb-Edu into `train.bin` / `val.bin` (resumable) |
| `data/make_synth.py` | Synthetic long-range examples (5% of pretraining rows) |
| `data/loader.py` | Memmap batch loader, with the synthetic mix |
| `data/instruct_format.py` | The one prompt template, shared by instruction tuning and eval |
| `data/prepare_instruct.py` | Dolly-15k → fixed train/val split |
| `model/transformer.py` | Decoder backbone: RMSNorm, RoPE, SwiGLU, sliding-window attention, memory hook |
| `model/memory.py` | Memory interface, `NullMemory` (Variant A), `build_memory` |
| `model/titans_memory.py` | Titans-style memory for Variants B (always-on) and C (gated) |
| `train/pretrain.py` | Pretraining: fp16, cosine LR, val loss, exact resume, Kaggle time limit |
| `train/instruct.py` | Instruction tuning: response-only loss, best-val checkpoint, samples |
| `eval/` | Test generator, runner (choice log-prob), efficiency test, plots |
| `tests/` | Backbone, memory and instruction-data tests (CPU, seconds) |
| `docs/` | Architecture, memory and eval specs; training log |
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

0. **Synthetic long-range data (once, CPU, ~3 min)**, then save `/kaggle/working/artifacts/synth`
   as the Kaggle dataset `AFML_Project_Synth` and attach it to every pretraining notebook:
   ```python
   !python data/make_synth.py --data_dir {DATA} --tokenizer {DATA}/../tokenizer/tokenizer.json \
       --out_dir /kaggle/working/artifacts/synth
   ```
   Every pretraining command below then adds `--synth_dir <.../artifacts/synth>`.

1. **Throughput check (~3 min):** measures tok/s and confirms fp16 works on the GPU.
   ```python
   !python train/pretrain.py --variant A --data_dir {DATA} --out_dir /kaggle/working/bench \
       --max_steps 30 --eval_every 30 --log_every 10 --resume none
   ```
2. **Full run** as a background commit (Save Version → Save & Run All):
   ```python
   !python train/pretrain.py --variant A --data_dir {DATA} --synth_dir {SYNTH} --out_dir /kaggle/working/ckpt_A --time_limit_min 690
   ```
3. If it stops on the time limit, attach that version's output as an input in a new
   session and add `--resume /kaggle/input/<that output>/ckpt_A/latest.pt`.

Outputs in `ckpt_A/`: `latest.pt` (full state), `variant-A_pretrain_<date>.pt`
(weights + config, for instruction tuning and eval), `log.csv`, `summary.json`.

## Instruction tuning (Dolly-15k)

**Prepare once (CPU, Internet ON, ~1 min)** and save the output as a Kaggle dataset
`AFML_Project_Instruct`, shared by all three variants:

```python
TOK = "/kaggle/input/.../artifacts/tokenizer/tokenizer.json"
!python tests/test_instruct.py {TOK}
!python data/prepare_instruct.py --tokenizer {TOK} --out_dir /kaggle/working/artifacts/instruct
```

**Tune a variant (GPU T4, ~10-20 min)**, inputs: code, `AFML_Project_Data`,
`AFML_Project_Instruct`, and that variant's pretrain checkpoint:

```python
!python train/instruct.py --init /kaggle/input/.../variant-A_pretrain_<date>.pt \
    --data_dir /kaggle/input/.../artifacts/instruct --tokenizer {TOK} --out_dir /kaggle/working/instruct_A
```

Outputs in `instruct_A/`: `variant-A_instruct_<date>.pt` (best val loss),
`log.csv`, `summary.json`, `samples.md` (answers before vs after tuning).

| File | What it does |
|---|---|
| `data/instruct_format.py` | The only place prompts are built; eval must import it too |
| `data/prepare_instruct.py` | Dolly-15k → fixed split, tokenized, `meta.json` |
| `train/instruct.py` | Response-only loss fine-tuning, best-checkpoint saving, samples |
| `tests/test_instruct.py` | Template, prompt-token match, loss-mask tests |

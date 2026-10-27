# Architecture Spec — AFML Team 30 (handoff A → B, due Oct 28)

Source of truth for numbers is `config.py`. This doc explains them and fixes the
interface Person B plugs memory into. Anything here changes only at a check-in.

## 1. Backbone (identical for Variants A, B, C)

| Setting | Value | Why |
|---|---|---|
| Layers | 8 | Depth over width at this scale; 8 blocks keeps a T4 run under ~4 h |
| Hidden size `d_model` | 512 | ~30M params total, fits free-tier compute |
| Heads | 8 (head_dim 64) | Standard head_dim; RoPE works cleanly on 64 |
| FFN | SwiGLU, hidden 1408 | ≈ (8/3)·d, so params match a 4·d GELU MLP |
| Norm | RMSNorm, pre-norm, eps 1e-5 | Cheaper than LayerNorm, stable in fp16 |
| Positions | RoPE, theta 10,000 | Relative; positions inside a window only |
| Vocab | 8192, byte-level BPE, tied embeddings | Embeddings stay ~14% of params |
| Dropout | 0.0 | Single pass over data, no need |
| **Params** | **≈ 29.9M** | 25.7M blocks + 4.2M embedding |

## 2. Context handling

| Setting | Value |
|---|---|
| Training sequence length | 1024 tokens |
| Attention | Local causal, sliding window **W = 256** |
| Memory chunk size **C** | 256 tokens (= W) |
| Eval sequence length | up to 4096 tokens |

Consequences everyone must be able to explain:

- **Receptive field (corrected Oct 27, verified in `tests/test_model.py`).**
  One layer sees 256 tokens back, but stacked windows let information hop back
  one window per layer: the backbone's indirect reach is
  L·(W−1) = 8·255 = **2,040 tokens**. Each hop is lossy, so recall from 300–2,040
  tokens back is possible but weak; **beyond 2,040 tokens Variant A is strictly
  blind** and only memory can help. Eval should place needles in both regimes
  (≈300–2,000 tokens back and > 2,040 tokens back, up to 4,096) and report them
  separately.
- A 1024-token training sequence = 4 chunks, so memory is exercised during
  pretraining (3 write→read hand-offs per sequence). At eval, 4096 = 16 chunks.
- RoPE never sees offsets > 256, so evaluating at 4096 does not stress
  position extrapolation; only memory can carry information that far.

## 3. Memory plug-in point (Person B)

Memory sits **after block 4 of 8** (`memory.insert_after_layer = 4`): lower
blocks build token features, upper blocks use what memory returns.

Implemented in `model/transformer.py` (`SLM.run_memory`). The lower and upper
blocks run over the whole sequence in parallel, so their attention windows
cross chunk borders; only the memory step runs chunk by chunk (`B × C × d`):

```
h = blocks[0:4](x)                             # whole sequence, sliding-window attention
for each chunk t:
    r_t     = memory.read(h_t, state_{t-1})    # retrieve using chunk-t queries
    h_t     = h_t + r_t                        # residual add (B: always; C: gated inside read)
    state_t = memory.write(h_t, state_{t-1})   # update AFTER reading (no leakage of chunk t into its own read)
y = blocks[4:8](concat(h_t))                   # whole sequence again
```

Interface the backbone calls (`model/memory.py`; Person B subclasses
`MemoryModule`, registers B and C in `build_memory`, and adds the class to
`MEMORIES_TO_CHECK` in `tests/test_model.py`, which checks causality):

```python
class MemoryModule(nn.Module):
    def init_state(self, batch_size: int, device, dtype) -> MemoryState: ...
    def read(self, h: Tensor, state: MemoryState) -> Tensor:          # (B,C,d) -> (B,C,d)
    def write(self, h: Tensor, state: MemoryState) -> MemoryState: ...
    def stats(self) -> dict: ...   # optional: write rate, gate means, norms (for logging)
```

- Variant A uses a `NullMemory` (read returns zeros, write returns state).
  Same code path for all three, so A/B/C differ only in this module.
- Read-before-write within a chunk keeps causality: chunk t's tokens never
  retrieve information written from chunk t itself (local attention already
  covers that).
- Gradients flow through `state` across chunks inside one 1024-token
  sequence; state resets at each new training sequence. At eval, state
  persists across all chunks of the document.
- `MemoryState` is whatever Person B needs (e.g. a fast-weight matrix
  `B × d_k × d_v` for Titans-style, or slots). Must be detachable for
  truncated backprop if memory gets tight.

## 4. Shared training settings (do not vary across variants)

- Tokenizer: `artifacts/tokenizer/tokenizer.json` (special ids:
  0 `<|endoftext|>`, 1 `<|pad|>`, 2 `<|user|>`, 3 `<|assistant|>`, 4 `<|context|>`).
- Data: FineWeb-Edu `sample-10BT`, first ~600M train tokens, every 200th doc to val.
  **Built Oct 27 — Kaggle dataset `AFML_Project_Data`:** 600,734,248 train /
  3,103,788 val tokens from 481,280 docs; `tokenizer_sha 7f469545b846`.
  Everyone loads this dataset; nobody re-tokenizes. If your `meta.json` shows a
  different `tokenizer_sha`, you are on the wrong data.
- Optimiser: AdamW (0.9, 0.95), wd 0.1, lr 6e-4 → 6e-5 cosine, 500 warmup,
  clip 1.0, fp16 + GradScaler, 131k tokens/step (32 × 4 accum × 1024), seed 1337.
- Instruction tuning (`train/instruct.py`): Dolly-15k with the template in
  `config.py`, formatted only through `data/instruct_format.py` (eval uses the
  same module). Fixed split: 500 val examples, seed 1337; examples over 1,024
  tokens dropped, never truncated. Loss only on response tokens + `<|endoftext|>`.
  AdamW lr 2e-4 → 2e-5 cosine, 50 warmup, wd 0.01, batch 16 × 2 accum, 3 epochs,
  best checkpoint by val loss. Identical for A, B and C.

## 4b. Synthetic long-range data in pretraining (decided Oct 27: yes)

- `data/make_synth.py` builds 60,000 examples once; Kaggle dataset
  `AFML_Project_Synth`. Each: a key→value FACT, then 256–900 tokens of real
  FineWeb filler (50% contain 1–3 distractor facts, 25% a later CORRECTION
  of the same key), then a QUESTION and the ANSWER.
- `--synth_frac 0.05`: each training row starts with a synthetic example with
  probability 5%; FineWeb fills the rest of the row. Identical for A, B, C.
  Val loss is never mixed (pure FineWeb), so it stays comparable.
- Checkpoints record `synth_sha`; resuming with a different mix is refused.
- Training distances stop at ~900 tokens (one 1,024-token sequence), so
  eval distances of 1,000–4,096 also test generalisation beyond training.

**Person C: eval must NOT reuse these training templates** (full lists in
`synth_meta.json` → `training_templates`):

| Part | Training uses (avoid in eval) |
|---|---|
| Keys | locker, storage unit, server rack, safe, vault, mailbox, bike lock, garage, filing cabinet, warehouse door, staff room, lab freezer + a number 1–999 |
| Values | 4–6 digit codes; words amber, falcon, cedar, orbit, maple, harbor, pixel, quartz, tundra, violet, willow, zephyr, basalt, comet, delta, ember, glacier, indigo |
| Fact phrasing | "Remember this: the combination for X is V." · "For the record, X uses the code V." · "Memo: X -> V." · "Note to self: X opens with V." |
| Correction phrasing | "Correction: X now uses V." · "Update: the code for X was changed to V." |
| Question phrasing | "Q: What is the combination for X?\nA:" · "Which code does X use? Answer:" · "Recall the value for X:" |

Suggested eval instead: people/cities/project names as keys, other value
types (dates, names, 3-digit or 7+ digit numbers), different sentence shapes.
Eval prompts for instruction-tuned models use `data/instruct_format.py`.

## 5. Compute estimate

6 · 29.9M · 600M ≈ 1.1e17 FLOPs ≈ 3–4 h on one T4 at realistic utilisation
for Variant A (~4,580 steps). B and C will be slower; Person B should
report tokens/sec early so we can cut `target_tokens` for all three together
if needed (same budget for every variant).

## 6. Open points for the Oct 28 check-in

1. **Long-range signal in pretraining.** FineWeb-Edu rarely needs facts from
   > 256 tokens back in a way that rewards memory. Proposal: mix ~5% synthetic
   long-range examples (from Person C's generator, held-out templates from eval)
   into pretraining for **all three variants** equally. Otherwise B/C may learn
   to ignore memory and the ablation shows nothing.
2. Memory position (after block 4) is a guess; one ablation could move it to
   after block 2 or 6 if time allows.
3. Confirm everyone has Kaggle + Colab access before Nov 1 launch.

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

- **Variant A cannot see beyond 256 tokens back.** Any fact further away is
  invisible to it. This is the gap memory is meant to close, and why tests use
  contexts of 1k–4k with the needle placed > 256 tokens before the question.
- A 1024-token training sequence = 4 chunks, so memory is exercised during
  pretraining (3 write→read hand-offs per sequence). At eval, 4096 = 16 chunks.
- RoPE never sees offsets > 256, so evaluating at 4096 does not stress
  position extrapolation; only memory can carry information that far.

## 3. Memory plug-in point (Person B)

Memory sits **after block 4 of 8** (`memory.insert_after_layer = 4`): lower
blocks build token features, upper blocks use what memory returns.

Processing is chunk-sequential. For chunk `t` (shape `B × C × d`):

```
h_t   = blocks[0:4](x_t)                       # local attention only
r_t   = memory.read(h_t, state_{t-1})          # retrieve using chunk-t queries
h_t   = h_t + r_t                              # residual add (B: always; C: gated inside read)
state_t = memory.write(h_t, state_{t-1})       # update AFTER reading (no leakage of chunk t into its own read)
y_t   = blocks[4:8](h_t)
```

Interface the backbone will call (Person B implements, keeps this signature):

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
- Optimiser: AdamW (0.9, 0.95), wd 0.1, lr 6e-4 → 6e-5 cosine, 500 warmup,
  clip 1.0, fp16 + GradScaler, 131k tokens/step (32 × 4 accum × 1024), seed 1337.
- Instruction tuning: Dolly-15k with the template in `config.py`, loss only on
  response tokens.

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

# Training log

All runs: 600M FineWeb-Edu tokens (tokenizer_sha 7f469545b846), 5% synthetic
long-range rows (synth_sha 894281983d09), 4,577 steps × 131,072 tokens, one Kaggle T4.

## Pretraining

| Variant | Params | Val loss | Perplexity | Speed | Time | Checkpoint |
|---|---|---|---|---|---|---|
| A (no memory) | 29.89M | 3.165 | ~23.7 | 32.3k tok/s | 5.2 h | variant-A_pretrain_20261027.pt |
| B (always-on memory) | 30.94M | 3.228 | ~25.2 | 32.7k tok/s | 5.4 h | variant-B_pretrain_20261105.pt |
| C (gated memory) | 30.95M | 3.165 | ~23.7 | 27.9k tok/s | 6.3 h | variant-C_pretrain_20261106.pt |

B's memory is active and stable at the end of training: write rate 1.0, read gate 1.0,
memory norm ~27.5 (flat), retrieval norm ~18. B's val loss is ~0.06 higher than A's on
ordinary text, where memory rarely helps; C's read gate is meant to remove this cost.

C's write gate stayed selective (write rate ~0.58, mean surprise ~0.82) with a far smaller
memory norm (~1.7 vs B's ~27.5). Its read gate fell from 0.50 to ~0.04 on average, so on
ordinary text C almost stops reading memory, which is why its val loss matches A's (3.165)
and avoids B's ~0.06 cost. Whether the gate opens when a question needs a far-back fact
is what the long-range eval decides.

## Instruction tuning (Dolly-15k, 3 epochs, best val kept)

| Variant | Answer val loss before → best | Best step | Checkpoint |
|---|---|---|---|
| A | 2.954 → 2.782 | 800 | variant-A_instruct_20261028.pt |
| B | 3.017 → 2.820 | 800 | variant-B_instruct_20261106.pt |
| C | 2.962 → 2.783 | 800 | variant-C_instruct_20261109.pt |

## Eval baseline: Variant A (instruction-tuned)

- t1c control (fact 60–200 tokens back): 95%
- Every test from 300 tokens back: ~25% (chance, 4 choices)
- T2: picks the distractor 71%; T3: 42% only because the update sits closer
- Untrained model ~25% (no leak); re-runs identical

# Training log

All runs: 600M FineWeb-Edu tokens (tokenizer_sha 7f469545b846), 5% synthetic
long-range rows (synth_sha 894281983d09), 4,577 steps × 131,072 tokens, one Kaggle T4.

## Pretraining

| Variant | Params | Val loss | Perplexity | Speed | Time | Checkpoint |
|---|---|---|---|---|---|---|
| A (no memory) | 29.89M | 3.165 | ~23.7 | 32.3k tok/s | 5.2 h | variant-A_pretrain_20261027.pt |
| B (always-on memory) | 30.94M | 3.228 | ~25.2 | 32.7k tok/s | 5.4 h | variant-B_pretrain_20261105.pt |
| C (gated memory) | 30.95M | – | – | – | – | – |

B's memory is active and stable at the end of training: write rate 1.0, read gate 1.0,
memory norm ~27.5 (flat), retrieval norm ~18. B's val loss is ~0.06 higher than A's on
ordinary text, where memory rarely helps; C's read gate is meant to remove this cost.

## Instruction tuning (Dolly-15k, 3 epochs, best val kept)

| Variant | Answer val loss before → best | Best step | Checkpoint |
|---|---|---|---|
| A | 2.954 → 2.782 | 800 | variant-A_instruct_20261028.pt |
| B | – | – | – |
| C | – | – | – |

## Eval baseline: Variant A (instruction-tuned)

- t1c control (fact 60–200 tokens back): 95%
- Every test from 300 tokens back: ~25% (chance, 4 choices)
- T2: picks the distractor 71%; T3: 42% only because the update sits closer
- Untrained model ~25% (no leak); re-runs identical
# Results

All three variants were evaluated on the same frozen exam: 4,818 four-option questions in six files. Chance is 25%. Scoring is choice log-probability, and every checkpoint is the instruction-tuned one at its best step (800). Distances are measured in tokens from the end of the fact to the answer. Results are split into three ranges:

- **Control:** 60–200 tokens back, inside the 256-token window.
- **Weak:** 300–2,000 tokens back, inside the 2,040-token theoretical relay reach.
- **Blind:** 2,500–3,900 tokens back, where only memory can help.

With 400–500 questions per cell, the 95% interval is about ±4 points.

## Sanity checks

- An untrained model scores 15–32% on every test, around chance. So the exam does not give the answer away.
- Re-running Variant A gives identical numbers, so the evaluation is deterministic.
- Inside the window, all three models answer almost perfectly: **A 95.3%, B 98.0%, C 96.0%**. The models can read and answer; the format works.

## Main result: no variant recalls facts beyond its attention window

| Test | Range | A | B | C |
|---|---|---|---|---|
| T1 control | 60–200 | **95.3%** | **98.0%** | **96.0%** |
| T1 Needle | weak | 25.2% | 26.8% | 27.6% |
| T1 Needle | blind | 23.5% | 21.5% | 25.8% |
| T2 Selective memory | weak | 17.6% | 19.6% | 17.6% |
| T2 Selective memory | blind | 26.5% | 19.5% | 23.8% |
| T3 Contradiction | weak | 41.8% | 45.4% | 43.8% |
| T3 Contradiction | blind | 26.5% | 25.2% | 27.0% |
| T4 Multi-hop | weak | 25.4% | 22.8% | 21.6% |
| T4 Multi-hop | blind | 24.2% | 26.2% | 25.8% |
| T5 Corruption | weak | 20.8% | 19.6% | 22.9% |
| T5 Corruption | blind | 24.3% | 20.6% | 18.4% |
| **All tests, 300+ tokens** | | **25.6%** | **24.9%** | **25.5%** |

Accuracy falls from about 95% at 200 tokens to chance at 300 tokens, for all three models (Figure 1). It stays at chance out to 3,900 tokens. No memory variant beats A in any range by more than the ±4-point interval.

Two consequences:

1. **The effective reach of the backbone is the 256-token window, not the theoretical 2,040.** Information can in principle relay back one window per layer, but in practice a specific fact does not survive the hops.
2. **Neither memory design closes the gap on this exam.** B (always on) and C (learned gates) score like A both where attention cannot reach (blind) and where the synthetic training examples did reach (300–900 tokens).

The three models are not identical: on the same questions they pick the same answer only 58% of the time. But they are equally at chance, so the memory changes *which* wrong answer is chosen, not whether the right one is.

## How the models fail: they copy the nearest matching value

The wrong answers on this exam are chosen on purpose, so they show *why* a model fails.

| Behaviour | A | B | C |
|---|---|---|---|
| T2: picks another key's value (distractor) | 71.2% | 73.2% | 71.9% |
| T2 at 300 tokens: correct | 6% | 9% | 8% |
| T3 weak: correct (newest value) | 41.8% | 45.4% | 43.8% |
| T3: picks the old, overwritten value | 22.8% | 19.6% | 20.4% |
| T5: picks the misleading near-miss value | 37.9% | 42.2% | 39.2% |

- **T2 (selective memory).** When other facts of the same kind sit closer to the question, every model picks one of them. At 300 tokens the distractors are inside the window and the target is not, so accuracy drops to 6–9%: far *below* chance.
- **T3 (contradiction).** The above-chance "weak" score comes entirely from short distances. At 300 tokens the update sits inside the window, and all models answer 95–98% correctly. By 600 tokens it is 41–46%, and from 1,000 tokens on it is chance. The models do not track updates through memory; they read whichever value they can still see.
- **T5 (corruption).** A misleading sentence about a similar name, placed nearer the question, is picked about 40% of the time.
- **T4 (multi-hop)** is at chance for all variants at every distance.

The pattern is the same for all three models: **when the right fact is out of view, they answer with the most recent matching value in view.** This is the behaviour memory was meant to fix, and neither B nor C changes it.

## Why the memory did not help (what we know)

- **C barely reads its memory.** Its learned read gate averaged about 0.04 during pretraining, and C matches A's validation loss exactly (3.165, and 2.783 vs 2.782 after tuning). Both are consistent with C learning to mostly switch memory off.
- **B always reads, but the reads do not carry the fact.** B pays a small loss cost (pretrain val 3.228 vs 3.165) without any gain in recall.
- **B learned the synthetic training templates, not general recall.** Before tuning, B produced the training templates verbatim ("Memo: lab freezer 411 -> 5023"). The exam deliberately uses different keys, values and sentences, and there B is at chance. This suggests the 5% synthetic data taught surface patterns more than a reusable "store a fact, retrieve it later" skill.
- **Not tested:** whether instruction tuning on short Dolly examples (which never need memory) weakened memory use. Evaluating the *pretrained* B and C checkpoints would answer this. It is future work.

# Trade-offs

## What memory costs

| | A | B | C | Full attention (4k window) |
|---|---|---|---|---|
| Parameters | 29.9M | 30.9M | 30.9M | 29.9M |
| Memory state per sequence | none | 128 KiB | 128 KiB | none |
| Training speed (T4) | 32.3k tok/s | 32.7k tok/s | 27.9k tok/s | n/a |
| Pretrain / instruct val loss | 3.165 / 2.782 | 3.228 / 2.820 | 3.165 / 2.783 | n/a |
| Latency at 4,096 tokens (batch 1) | 75 ms | 82 ms | 82 ms | 78 ms |
| Peak GPU memory at 4,096 (batch 8) | 877 MB | 891 MB | 891 MB | 900 MB |

- **The memory is cheap.** It adds about 1M parameters (+3.5%) and a fixed 128 KiB state that does not grow with document length.
- **Speed cost:** about 9% more inference latency at 4k tokens. In training, B ran as fast as A, while C was about 14% slower because of its gates.
- **Quality cost:** B paid a small language-modelling cost. C did not, but C also used its memory very little.

## What it would save, in theory and in practice

| At 4,096 tokens | Local window (A/B/C) | Full attention | Ratio |
|---|---|---|---|
| Attention compute | 16.7 GFLOPs | 137.5 GFLOPs | 8.3× less |
| KV cache to generate the next token | 4 MB | 64 MB | 16× less |
| Measured latency | 75–82 ms | 78 ms | about the same |

In theory, local attention plus a fixed memory scales much better than full attention: 8× less attention compute and a 16× smaller cache at 4k tokens, and the gap grows with length. In this implementation the saving is not realised. The sliding window is applied as a mask over full attention, so all scores are still computed, and measured speed and memory match full attention (Figure 6). We therefore report the efficiency gain as theoretical, and list a real sliding-window kernel as future work.

## The trade-off in one line

Memory added about 3.5% parameters, 128 KiB of state and up to about 14% training time. In return, on this exam, it gave no measurable gain in long-range recall. The architecture's efficiency advantage is real on paper but needs (a) a model that actually learns to use its memory and (b) a true sliding-window kernel before it pays off.

## Future work suggested by these results

- Evaluate the pretrained (pre-Dolly) B and C checkpoints, to see whether instruction tuning washed out memory use.
- Train with more varied long-range examples (more keys, phrasings and distances beyond 900 tokens) so memory learns recall rather than templates.
- Log C's read-gate value on exam questions, to see whether it opens when a fact is needed.
- Implement a real sliding-window attention kernel, so the theoretical savings can be measured.

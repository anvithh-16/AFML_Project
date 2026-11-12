# Memory v2 spec (B and C), AFML Team 30

Code: `model/memory_v2.py` · tests: `tests/test_memory_v2.py` · toy check: `eval/toy_recall.py`.
Selected with `memory.version = 2` (`--memory_version 2`). v1 (`model/titans_memory.py`) stays
the default, so every v1 checkpoint and result is unchanged and reported as the "before".

## 1. Why a v2

v1 B and C gave no long-range gain (`docs/Results_and_Tradeoffs.md`). `eval/memory_probe.py`
showed why, inside the memory itself:

| Finding (v1) | Cause |
|---|---|
| Even an ideal memory, written from the fact sentences only, could not tell the true code from distractors (binding score ~0) | Each token's key was stored with **its own** value, so a key never points to the fact that follows it |
| Retrieval did not depend on the query | Same cause, plus 256 tokens averaged into one update |
| B's memory output was ~27% of the hidden state but carried no information | Every read was rescaled to full size, so "nothing found" was as loud as a hit |
| C's read gate closed (~0.04) and stayed closed | Starting half-open with nothing useful to read, it learned to switch memory off |

## 2. Design

Unchanged from v1: plug-in point (after block 4), chunk 256, read before write, write from
pre-read h, 8 heads x 64 x 64 fp32 state (128 KiB per sequence), `W_O` zero-init (B and C
start exactly equal to A).

| Part | v2 | Source / reason |
|---|---|---|
| Keys and queries | Linear projection, then a 4-token causal depthwise conv (identity at init), unit norm per head. The last 3 tokens of the previous chunk are carried in the state | Short convs as in Titans and other linear-memory models: a key stands for a phrase ("locker 12") |
| Values | Key t is stored with tokens t+1..t+4 (`value_shift`), combined with a learned weight per offset and channel (`value_taps`, init = plain mean), unit RMS | Fixes the v1 binding failure; Memory Mosaics also pairs keys with future tokens. The taps keep order ("48213" vs "84213"), which a plain mean loses |
| Write rule | Exact per-token delta rule `M <- M + beta_t k_t^T (v_t - k_t M)`, computed in parallel per 64-token step (UT transform, as in DeltaNet). Stable for beta in [0,1] | v1 averaged 256 tokens into one update (~4x over capacity) |
| Write strength | `beta_t = theta_h * w_t`, theta learned per head (init 0.5) | A kept token is written at full strength; v1 divided by the chunk length |
| Forgetting | Retention per step = prod_t (1 - f_t), init 2% per 256 tokens | Titans: removing forgetting hurt most |
| Momentum | `S <- eta S + dM`, `M <- M + S`, eta learned per head (init 0.05, max 0.5) | Titans' momentum on surprise |
| Read | `r = q M` straight to `W_O` (no rescaling) | Irrelevant or empty memory gives a small read |

### The only difference between B and C

| | Write weight w_t | Read | Forget f_t |
|---|---|---|---|
| **B** (always on) | 1 | always | one learned rate per head |
| **C** (learned policy) | `sigmoid(beta (surprise - tau) + u.h + b_w)`, **b_w = +2 so it starts open** | gate `sigmoid(w_r.h + c log(1+|r|) + b)`, **b = +2 so it starts open** | `sigmoid(f.h + b_f)` per token |

Surprise = `|v - kM|^2 / |v|^2` per token (detached), measured before each 64-token step.
Extra parameters over A: ~1.05M (B), ~1.06M (C).

## 3. Tests (`python tests/test_memory_v2.py`, CPU)

- Parallel delta step equals the textbook per-token loop (to 1e-5).
- Causal for B and C, also across chunk borders (conv tail); memory carries information forward.
- B and C equal A at step 0.
- 16 facts written in one chunk are recalled by key (mean cos ~0.86; newest exact, oldest ~0.66).
- 64 chunks: state stays finite and bounded. Gradients reach every memory parameter.
- State, including the conv tail, carries across calls (long eval documents). `version` 1 stays the default.

## 4. Gate before GPU time: toy recall bench

`eval/toy_recall.py`: tiny models (4 layers, window 16) learn `[FACT k1 k2 SEP v1 v2]` placed
beyond the receptive field, then `[QUERY k1 k2 SEP] -> v1 v2`. A can only guess (1/16 per
token). **Pass: v2 B/C clearly above chance while A and v1 stay near it.** A mechanism check
only; never training data for the real models.

Half the questions have their fact copied just inside the window ("near"), so every model
learns the format; "far" questions can only be answered from memory.

**Result (Nov 12, CPU, 2,000 steps, d_model 64, window 16, T 256, seed 0; chance 0.062):**

| Variant | Near acc | **Far acc** | Memory behaviour |
|---|---|---|---|
| A (no memory) | 1.000 | **0.056** | n/a |
| B v1 | 1.000 | **0.065** | large read (~14) that carries nothing |
| B v2 | 1.000 | **0.169** | read ~1.5, writes 94% of tokens |
| C v2 | 1.000 | **0.185** | writes 17% of tokens, forgets 0.5%/256, read gate stays open (0.95) |

Gate passed: v2 recalls facts from beyond the receptive field at ~3x chance and ~3x A,
v1 does not. Absolute recall is still low at this tiny size and budget; one seed only.

**Kaggle GPU, 10,000 steps, d_model 128, window 32, T 512 (Nov 12):** A far 0.035-0.110
(noise around chance), B v2 far 0.16 -> 0.23, levelling off near 0.20 after ~6k steps,
1.5x A's step time. C v2 diverged to NaN before step 1,000.

**Fixes after that run:**
- C's NaN: the forget gate used `log(1 - sigmoid(x))`. Once the gate saturates (sigmoid = 1.0
  in fp32) that is log 0 with an infinite gradient, and one step turns every weight NaN
  (reproduced in `test_saturated_gates_stay_finite`). Now `-softplus(x)`: the same value,
  always finite. The read-gate confidence uses a norm with a finite gradient at 0.
- B's plateau near 20%: a plain mean of the next 4 tokens is order-blind, so "v1 v2" and
  "v2 v1" were stored the same. Now learned per-offset taps (`value_taps`, init = mean).
- The bench now stops a variant at its first non-finite loss or gradient and says where.

**Rerun with the fixes (Kaggle, 10,000 steps):** B v2 far **0.461** (from ~0.20; climbing to
~9k steps), C v2 far **0.259**, no NaN. Step time vs A: B 1.65x, C 1.9x.
C's write gate closed mechanically as memory filled (surprise ~0.4 gives
sigmoid(4(0.4-0.5)) ~ 0.4) and dropped fact tokens with filler; in this toy surprise cannot
tell them apart. Now `write_gate_bias = +2`: C starts like B (gate ~0.83 with a full
memory) and has to learn to skip tokens, the same principle as the open read gate.

## 5. Training protocol (identical for A, B, C except the memory)

- Same data file, mix, seed, sequence length (2,048 = 8 chunks), window 256, chunk 256,
  tokens per step (~131k via batch x accum), optimiser, steps.
- Same start for all three: either all from scratch, or all from Variant A's checkpoint
  (`--init_from`; B and C get a fresh memory on top, A just continues).
- Same instruction tuning and long-range fine-tune stage; eval on held-out splits and the frozen exam.
- If B/C run out of GPU memory: `memory.detach_chunks = True` for both B and C.

## 6. Not in v2 (future work)

Deep MLP memory (Titans' best variant), a per-layer memory (Infini-attention), a real
sliding-window kernel for measured (not theoretical) efficiency.

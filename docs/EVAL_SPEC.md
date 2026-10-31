# Evaluation Spec — AFML Team 30 (Person C, lane-c)

Every variant (A, B, C) is scored by the same harness on the same frozen eval
files. Distances are in **tokens** (shared tokenizer, `tokenizer_sha 7f469545b846`).

## 1. Shared rules

**Filler.** Real FineWeb-Edu text taken from `val.bin` (never trained on).

**Distance.** This is the number of tokens from the end of the fact to the first
answer token. The answer token is the position where the model has to retrieve
the fact.

**Distance bins.** Results are reported separately for each regime:

| Regime | Bins (tokens) | Why |
|---|---|---|
| Weak (A can partly reach) | 300, 600, 1000, 1500, 2000 | Under the 2,040 receptive field |
| Blind (only memory helps) | 2500, 3000, 3500, 3900 | Beyond 2,040, so A must be at chance |

Training synth stops at about 900 tokens, so every bin of 1000 or more also tests
generalisation beyond the distances seen in training.

**Prompt.** It is built with `data/instruct_format.py` (the same template as
instruction tuning):

- Instruction: `Read the document and answer the question at the end.`
- Context: `<filler + facts>\n\nQuestion: <question>`
- Response being scored: `<answer>`

The question goes at the end of the context, not in the instruction slot.
Otherwise it would sit thousands of tokens before the answer.

**Scoring.**

- *Primary metric: choice accuracy.* There are 4 candidate answers of the same
  type with equal token length. Each candidate is scored by the summed log-prob
  of its tokens as the response, and the highest score is the model's pick.
  Chance is 25%. This works on pretrained checkpoints too.
- *Secondary metric: greedy exact match.* The model generates up to 16 tokens,
  which are compared to the answer after normalisation.

**Size.** 100 examples per (test, bin), fixed seeds. The eval files are frozen
once shared.

**Sanity check.** A randomly initialised model must score about 25%. If it scores
clearly higher, the test leaks the answer.

## 2. Templates (must differ from the training synth, spec 4b)

| Part | Eval uses |
|---|---|
| Keys | People ("Dr. Ilona Varga"), projects ("Project Kestrel"), ships, research stations |
| Values | Dates ("14 March 1987"), cities, person names, 3-digit or 7-digit numbers |
| Fact phrasing | "Dr. Ilona Varga was born in Tallinn." · "Project Kestrel launched on 14 March 1987." · "The research station Halvard is staffed by 742 people." |
| Update phrasing | "Dr. Varga later moved her lab to Porto." · "Project Kestrel's launch was postponed to 2 June 1991." |
| Question phrasing | "Where was Dr. Ilona Varga born?" · "On what date did Project Kestrel launch?" |

None of the training keys, values, words or sentence shapes from spec 4b may appear.

## 3. The six tests

### T1 — Needle in long context
- **Measures:** whether the model can recall one fact after a long gap.
- **Input:** lead filler → fact → filler (distance d) → question.
- **Varies:** d over all 9 bins.
- **Reported:** accuracy vs d, weak and blind regimes separately.

### T1c — Control: needle inside the window (added Oct 31)
- **Measures:** whether the model and the scoring work at all, separate from memory.
- **Input:** same as T1, but the fact is only 60, 120 or 200 tokens back, inside the
  256-token attention window. 100 examples per bin, file `t1c.jsonl`.
- **Reported:** accuracy per bin, as the "control" regime.
- **Expected:** high for every variant. Variant A instruct scored 94–96% (Oct 31).
  A variant that fails here has a model or format problem, not a memory problem.
- **Why it was added:** A scored about 25% from 300 tokens out on every test. The
  control shows the drop is a real limit of the model (its effective reach is the
  256-token window, not the 2,040-token theoretical reach), not a broken test.

### T2 — Selective memory
- **Measures:** whether the model stores the useful fact instead of everything.
- **Input:** the target fact plus k distractor facts of the same type about other
  keys, inside filler, followed by a question about the target. The distractor
  values are among the 4 choices.
- **Varies:** k ∈ {2, 5, 10}, with d spread over the bins.
- **Reported:** accuracy, and the *confusion rate* (the model picked another
  key's value).

### T3 — Contradiction / update
- **Measures:** whether memory overwrites a stale fact.
- **Input:** fact (old value) → filler → update (new value) → filler → question.
  The correct answer is the new value. The old value is always one of the choices.
- **Varies:** the distance from the old fact and from the update.
- **Reported:** accuracy, and the *stale rate* (the model picked the old value).

### T4 — Multi-hop
- **Measures:** whether the model can combine two facts that are far apart.
- **Input:** hop 1 ("Dr. Varga was born in Tallinn.") … filler … hop 2 ("Tallinn's
  oldest library is the Kreutz Library.") … question ("What is the oldest library
  in Dr. Varga's birth city?").
- **Varies:** the distance of each hop from the question.
- **Reported:** accuracy vs the distance of the farther hop.

### T5 — Memory corruption
- **Measures:** how much injected noise changes the answer.
- **Input:** T1 examples paired with copies that contain injected text:
  - (a) *irrelevant*: n ∈ {2, 5, 10} unrelated facts;
  - (b) *near-miss*: a similar key with a different value ("Dr. Ilona Vargas was
    born in Riga."). The near-miss value is one of the choices.
- **Reported:** accuracy drop (clean − corrupted) on matched pairs, and the rate
  at which the model picks the injected value.

### T6 — Efficiency
- **Measures:** cost compared with simply widening attention.
- **Setup:** A, B and C, plus a reference model with the same backbone but
  `attn_window = 4096` (full attention). The reference uses untrained weights and
  is used for cost only.
- **Varies:** sequence length 1k, 2k and 4k tokens, at batch size 1 and 8.
- **Reported:** tokens/sec, latency per sequence, and peak GPU memory.
  Quality comes from T1–T5.
- **Also reported, as theoretical cost:** attention GFLOPs (counting only the keys
  each query may see), KV-cache size for generating the next token, and the size
  of the memory state.
- **Why both kinds of cost:** the backbone builds the sliding window as a *mask*
  over full attention (`scaled_dot_product_attention` + `attn_mask`). It still
  computes all T×T scores, so measured speed of A equals full attention. The
  theoretical columns show the cost with a kernel that actually skips the masked
  blocks. The report must say which numbers are which.

## 4. Files and handoff

- `eval/generate.py` → `artifacts/eval/{t1..t5}.jsonl`, plus `eval_meta.json`
  (seeds, tokenizer_sha, and a hash of every file).
- `eval/run_eval.py --ckpt <path>` → `results/<variant>_<stage>.csv` with one row
  per example: `test, bin, correct, picked, stale/confused flags, logprobs`.
- `eval/efficiency.py` → `results/efficiency.csv`.
- `eval/plots.py` → the figures used in the report.

**Checkpoint format accepted.** It is the existing one:
`SLM(ModelConfig(**ck["model_config"]))`, load `ck["model"]`, and assert
`ck["tokenizer_sha"] == "7f469545b846"`. One forward pass over at most 4,096
tokens; memory chunking happens inside the model.

**JSONL row**

```json
{"id": "t1_2500_0007", "test": "t1", "bin": 2500, "regime": "blind",
 "instruction": "...", "context": "...", "answer": "Tallinn",
 "choices": ["Tallinn", "Porto", "Riga", "Ghent"],
 "meta": {"distance": 2503, "n_tokens": 2780, "seed": 7}}
```

**Fallback** (from the schedule): if time runs out, cut T5 and T2 first.

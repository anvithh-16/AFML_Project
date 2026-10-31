# Memory Design Spec — AFML Team 30 (Person B, due Oct 28)

Builds on `docs/ARCHITECTURE_SPEC.md` and `config.py`. Fixes what is stored, how it is
written, how it is read, and which knobs the ablations turn. Variants B and C share
**one code path**; they differ only in the two gates.

## 1. Design in one paragraph

A fast-weight **linear associative memory**, one small matrix per head, lives after
block 4. Each chunk first **reads** (queries retrieve from the state written by earlier
chunks), then **writes** (one error-correcting gradient step on `||k·M − v||²`, applied
analytically so it is differentiable). The memory's parameters are updated while the
sequence is being read, which is the test-time-learning idea from Titans. Associative
key→value storage with similarity-based retrieval follows Memory Mosaics. Overwriting
comes free from the delta rule: writing a new value for a key already stored first
subtracts the old prediction, so stale facts are replaced instead of piling up.

## 2. Store

| Item | Value |
|---|---|
| State `M` | `(B, H, d_k, d_v)` = `(B, 8, 64, 64)`, fp32, zero at sequence start |
| Size per sequence | 8·64·64 = 32,768 floats = **128 KiB**, constant in context length |
| For comparison | KV cache of a 4096-token window, all 8 layers, fp16: 64 MiB |
| Reset | new training sequence → zeros; at eval, persists across all chunks of a document |
| Learned params | `W_K, W_Q, W_V: 512→512`, `W_O: 512→512` (zero-init) ≈ 1.05M, ~3.5% on top of the 29.9M backbone |

Report the +3.5% parameter overhead of B and C versus A in the write-up.

## 3. Write (per chunk t, from the PRE-read h; team decision Oct 29)

```
k = L2norm(W_K h)            (B,C,H,d_k)     keys are unit length
v = W_V h                    (B,C,H,d_v)
err = v - k @ M_prev         prediction error against the state BEFORE this write
s   = |err|² / (|v|² + eps)  surprise, per token per head, detached

w = 1                                         Variant B  (write everything)
w = sigmoid(β_h·(s − τ) + u_h·h + b_w)        Variant C  (learned write gate)

ΔM    = (θ_h / C) · kᵀ (w ⊙ err)             theta_h = 2·sigmoid(raw_h), init 1
M_new = (1 − α)·M_prev + ΔM
```

- Mean over the chunk with `θ ∈ (0,2)` guarantees the step is stable
  (λmax(kᵀWk)/C ≤ 1 for unit keys), so no exploding state in fp16 runs.
- **Variant C behaviours the gate must learn:** a repeated, already-predicted token has
  low `s`, so `w` drops and it is not rewritten (selective memory). A changed value for a
  known key has high `s`, so `w` rises and the delta rule overwrites (contradiction test).
- Decay `α` is a constant, **identical in B and C** (default 0.02). C therefore differs
  from B only in learning when to write and read, which is the brief's controlled question.
- Momentum on the surprise is supported (`momentum=η`) but **off by default**: with 4
  chunks per training sequence it adds state and little else.

## 4. Read (per chunk t, before write)

```
q  = L2norm(W_Q h)
r  = q @ M_prev                      (B,C,H,d_v); zero in chunk 0
r̂  = RMSNorm_per_head(r)             parameter-free; makes retrieval scale-invariant to θ
g  = 1                                         Variant B  (always on)
g  = sigmoid(w_h·h + c_h·log(1+|r|) + b_r)     Variant C  (need gate)
out = λ_r · W_O( concat_heads( g ⊙ r̂ ) )       returned as the residual added to h
```

- `W_O` is zero-initialised, so at step 0 B and C compute exactly what A computes; the
  memory path grows in only if it lowers the loss.
- The need gate sees `h` (block-4 features, which carry whether local context suffices)
  and the retrieval confidence `|r|`. Init `b_r = 0` (gate starts at 0.5).
- `λ_r` is the "retrieval strength" ablation knob (default 1.0), usable at eval time.

## 5. Causality and training

- Read-before-write: chunk t only retrieves from chunks < t, never from itself.
- Gradients flow through `M` across the 4 chunks of a 1024-token sequence; `detach_every`
  option for truncated backprop if memory gets tight (off by default).
- Memory math runs in fp32 with autocast disabled inside the module (fp16 backbone).
- Gate collapse risk (C degenerates to B by keeping `w`≈1): log mean write rate; if it
  stays > 0.9, enable the optional aux loss `λ_w·(mean(w) − ρ)²`, ρ ≈ 0.2.
- Training data includes the team's 5% synthetic long-range mix, same for A, B, C.

## 6. Interface (matches ARCHITECTURE_SPEC §3)

```python
@dataclass
class MemoryState:
    M: Tensor            # (B, H, d_k, d_v) fp32
    S: Tensor | None     # momentum buffer, None unless momentum > 0

class MemoryModule(nn.Module):
    def __init__(self, cfg, variant: str):                   # "B" or "C"
    def init_state(self, batch_size, device, dtype) -> MemoryState
    def read(self, h, state) -> Tensor                       # (B,C,d) -> (B,C,d), the residual
    def write(self, h, state) -> MemoryState
    def stats(self) -> dict   # write_rate, mean_w, mean_g, mean_surprise, |M|_F, |r|
```

Config additions under `memory`: `n_heads=8, d_k=64, d_v=64, decay=0.02, tau=0.5,
read_strength=1.0, momentum=0.0, detach_every=None`.

## 7. Ablations (Nov 9–15)

| # | Knob | Default | Sweep | Needs retraining? |
|---|---|---|---|---|
| 1 | Memory size `d_k` | 64 | 32, 64, 128 | Yes, short matched runs |
| 2a | Write threshold `τ` (C) | 0.5 | 0.25, 0.5, 0.75 | No, eval-time |
| 2b | Decay `α` | 0.02 | 0, 0.02, 0.1 | No, eval-time |
| 3 | Chunk length `C` | 256 | 128, 256, 512 | No, eval-time |
| – | Retrieval strength `λ_r` | 1.0 | 0.5, 1, 2 | No, eval-time |

Only ablation 1 costs compute, so ablations 2 and 3 fit the Nov 13–15 window.

## 8. Unit tests (Oct 29–31, toy tensors)

1. Shapes: `read` and `write` preserve `(B,C,d)` and `(B,H,d_k,d_v)`.
2. Recall: write one `(k,v)` into zero memory, read with `q=k`, cosine(`r`, `v`) > 0.95.
3. Overwrite: write `(k,v1)`, then `(k,v2)` repeatedly; read approaches `v2`, not `v1`.
4. Causality: perturbing chunk t tokens leaves chunk t's `read` unchanged.
5. Zero-init: with `W_O=0`, output equals `NullMemory` output (A ≡ B ≡ C at step 0).
6. Gate sanity (C): repeated identical pair gives a falling write gate; novel pair high.
7. Stability: 16 chunks of random fp16 input, no NaN, `|M|_F` bounded.

## 9. Questions and decisions

1. **Write from pre-read `h`?** DECIDED (Oct 29, Anvith): yes. The backbone writes
   chunk t's own pre-read features, so retrieved content is never stored back.
2. **Parameter budget:** accept +3.5% for B and C, or shrink `d_k` to 32 to narrow it?
3. **Throughput:** B and C will be slower than A. I will report tokens/sec on the first
   30 steps so `target_tokens` can be cut for all three together if needed.
4. Memory position (after block 4) stays; moving it to block 2 or 6 is a stretch ablation.

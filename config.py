"""
Shared configuration for all three variants (A: no memory, B: always-on memory,
C: learned gated memory).

RULE: Variants A, B and C must use the SAME tokenizer, data, prompt template and
loss masking. Only the `memory` block below may differ between variants.
Changing anything else here after Oct 28 needs a team check-in.
"""
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Tokenizer + prompt template (frozen once the tokenizer is trained)
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = [
    "<|endoftext|>",   # id 0 — document separator, also EOS for instruct
    "<|pad|>",         # id 1 — padding (never trained on)
    "<|user|>",        # id 2 — start of instruction turn
    "<|assistant|>",   # id 3 — start of response turn (loss starts AFTER this)
    "<|context|>",     # id 4 — optional context block (Dolly 'context' field)
]
EOT, PAD, USER, ASSISTANT, CONTEXT = SPECIAL_TOKENS

# Instruction template. Loss is computed only on tokens after <|assistant|>,
# including the closing <|endoftext|>.
PROMPT_TEMPLATE = "<|user|>\n{instruction}\n{context_block}<|assistant|>\n"
CONTEXT_BLOCK_TEMPLATE = "<|context|>\n{context}\n"   # empty string if no context
RESPONSE_TEMPLATE = "{response}<|endoftext|>"


@dataclass
class TokenizerConfig:
    vocab_size: int = 8192          # small vocab keeps embeddings ~4M of ~30M params
    min_frequency: int = 2
    individual_digits: bool = True  # "4821" -> 4 tokens; helps exact recall in needle tests
    train_docs: int = 200_000       # FineWeb-Edu docs used to fit BPE (~0.5-1 GB text)
    path: str = "artifacts/tokenizer/tokenizer.json"


@dataclass
class DataConfig:
    hf_dataset: str = "HuggingFaceFW/fineweb-edu"
    hf_subset: str = "sample-10BT"
    target_tokens: int = 600_000_000   # ~20 tokens/param (Chinchilla-ish) for ~30M params
    val_every_n_docs: int = 200        # every 200th doc -> val (~0.5%)
    out_dir: str = "artifacts/data"
    dtype: str = "uint16"              # vocab < 65536


@dataclass
class ModelConfig:
    # --- fixed backbone (identical for A, B, C) ---
    vocab_size: int = 8192
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8                  # head_dim = 64
    ffn_hidden: int = 1408            # SwiGLU: ~(8/3)*d_model, rounded to multiple of 64
    norm_eps: float = 1e-5            # RMSNorm
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    # --- context handling ---
    train_seq_len: int = 1024         # tokens per training sequence
    attn_window: int = 256            # local causal (sliding-window) attention span
    chunk_size: int = 256             # memory read/write granularity (== attn_window)
    eval_max_seq_len: int = 4096      # long-context tests go up to here

    # --- memory (Person B owns this block) ---
    memory: dict = field(default_factory=lambda: {
        "variant": "A",               # "A" none | "B" always-on | "C" learned gating
        "insert_after_layer": 4,      # memory read/write happens after this block (1-indexed)
    })

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def approx_params(self) -> int:
        d, f, L, V = self.d_model, self.ffn_hidden, self.n_layers, self.vocab_size
        per_layer = 4 * d * d + 3 * d * f + 2 * d     # attn qkvo + SwiGLU + 2 RMSNorm
        emb = V * d * (1 if self.tie_embeddings else 2)
        return L * per_layer + emb + d


@dataclass
class PretrainConfig:
    batch_size: int = 32              # sequences per micro-batch on a T4 (fp16)
    grad_accum: int = 4               # effective batch = 32*4*1024 = 131k tokens/step
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    precision: str = "fp16"           # T4 has no fast bf16; use GradScaler
    eval_every: int = 250
    ckpt_every: int = 500
    seed: int = 1337


if __name__ == "__main__":
    m = ModelConfig()
    p = PretrainConfig()
    tokens_per_step = p.batch_size * p.grad_accum * m.train_seq_len
    print(f"approx params     : {m.approx_params() / 1e6:.1f}M")
    print(f"head_dim          : {m.head_dim}")
    print(f"tokens per step   : {tokens_per_step:,}")
    print(f"steps for target  : {DataConfig().target_tokens // tokens_per_step:,}")
    print(f"pretrain FLOPs    : {6 * m.approx_params() * DataConfig().target_tokens:.2e}")

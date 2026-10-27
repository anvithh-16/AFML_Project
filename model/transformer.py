"""
Decoder-only Transformer backbone shared by Variants A, B and C.

    x -> embed -> blocks[0:k] -> memory (chunk by chunk) -> blocks[k:L] -> RMSNorm -> logits

Each block: pre-norm, sliding-window causal self-attention with RoPE, then a
SwiGLU MLP, each added back through a residual connection.

Design choices (for the report):
  * RMSNorm: rescales by the root-mean-square only (no mean subtraction, no
    bias). Cheaper than LayerNorm and works as well in LLaMA-style models.
  * RoPE: rotates query/key pairs by an angle proportional to position, so
    the q.k score depends only on the RELATIVE distance i - j. No learned
    position table, and positions beyond training length need no new params.
  * SwiGLU: down( silu(gate(x)) * up(x) ). The gate lets the MLP switch
    features on and off per token; hidden size 1408 ~ (8/3)*512 keeps the
    parameter count of a 4x GELU MLP.
  * Sliding window W=256: token i attends to tokens i-255..i. Cost per layer is
    O(T*W) instead of O(T^2). NOTE: stacking L windowed layers gives an
    indirect receptive field of L*(W-1) tokens (8*255 = 2040 here), so
    information can hop backwards one window per layer, but each hop is lossy.
    Beyond 2040 tokens the backbone is strictly blind; only memory can help.
  * Memory sits between the lower and upper blocks. Lower blocks run over the
    whole sequence in parallel (their windows cross chunk borders freely), the
    memory loop runs chunk by chunk, then upper blocks run in parallel again.
"""
import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.memory import build_memory


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute in fp32 for stability under fp16 autocast
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).type_as(x)


def rope_tables(head_dim, max_len, theta, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # (T, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)          # (T, D)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (T, D)
    d = x.shape[-1] // 2
    rotated = torch.cat([-x[..., d:], x[..., :d]], dim=-1)
    return (x * cos + rotated * sin).type_as(x)


_mask_cache = {}


def sliding_window_mask(T, window, device):
    """Bool (T, T): True where query i may attend key j, i.e. 0 <= i - j < window."""
    key = (T, window, str(device))
    if key not in _mask_cache:
        i = torch.arange(T, device=device)[:, None]
        j = torch.arange(T, device=device)[None, :]
        _mask_cache[key] = (j <= i) & (i - j < window)
    return _mask_cache[key]


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        return x + self.mlp(self.norm2(x))


class SLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        k = cfg.memory.get("insert_after_layer", cfg.n_layers // 2)
        assert 0 <= k <= cfg.n_layers
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.lower = nn.ModuleList([Block(cfg) for _ in range(k)])
        self.memory = build_memory(cfg)
        self.upper = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers - k)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = rope_tables(cfg.head_dim, max(cfg.eval_max_seq_len, cfg.train_seq_len), cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style: shrink the residual-branch output projections by 1/sqrt(2L)
        for name, p in self.named_parameters():
            if name.endswith("attn.out.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, exclude_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        return n - self.tok_emb.weight.numel() if exclude_embedding else n

    def _rope(self, T):
        if T > self.rope_cos.shape[0]:
            cos, sin = rope_tables(self.cfg.head_dim, T, self.cfg.rope_theta, self.rope_cos.device)
            self.rope_cos, self.rope_sin = cos, sin
        return self.rope_cos[:T], self.rope_sin[:T]

    def run_memory(self, h, state=None):
        """Chunk-sequential memory pass. Returns (h_with_memory, final_state)."""
        if self.memory.is_null:
            return h, state
        B, T, _ = h.shape
        C = self.cfg.chunk_size
        if state is None:
            state = self.memory.init_state(B, h.device, h.dtype)
        outs = []
        for s in range(0, T, C):
            h_t = h[:, s:s + C]
            h_t = h_t + self.memory.read(h_t, state)
            state = self.memory.write(h_t, state)
            outs.append(h_t)
        return torch.cat(outs, dim=1), state

    def forward(self, idx, targets=None, memory_state=None, return_state=False):
        B, T = idx.shape
        cos, sin = self._rope(T)
        mask = sliding_window_mask(T, self.cfg.attn_window, idx.device)

        x = self.tok_emb(idx)
        for blk in self.lower:
            x = blk(x, cos, sin, mask)
        x, state = self.run_memory(x, memory_state)
        for blk in self.upper:
            x = blk(x, cos, sin, mask)
        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-100)
        if return_state:
            return logits, loss, state
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None, eos_id=None):
        """Simple full-recompute sampling (fine for short eval answers at this size).
        temperature=0 -> greedy."""
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.eval_max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]
            if temperature == 0.0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx

    def config_dict(self):
        return asdict(self.cfg)

"""
model/memory_v2.py: Memory v2 for Variants B (always-on) and C (learned gates).
Selected with  cfg.memory["version"] = 2  (v1 = model/titans_memory.py stays the default,
so every v1 checkpoint and result is unchanged). Same plug-in interface as v1:

    r     = memory.read(h_t, state)     # chunk t reads what chunks < t wrote
    state = memory.write(h_t, state)    # then stores chunk t (pre-read h)
    out_t = h_t + r

What changed from v1 and why (docs/MEMORY_V2_SPEC.md has the full story):

  1. Phrase keys. Keys and queries go through a short causal depthwise conv (4 tokens),
     so a key stands for "locker 12", not just "12". The last 3 tokens of the previous
     chunk are carried in the state, so the conv also works across chunk borders.
  2. Next-token values. Key t is stored with the values of tokens t+1..t+4
     ("is 48213"), not with token t itself; a learned weight per offset and channel
     (init = plain mean) keeps their ORDER, so "48213" and "84213" differ. v1 associated each token with itself and
     could not bind a key to its fact even in the ideal case (memory_probe: ~0).
  3. Exact delta rule in small steps. A chunk is written in steps of 64 tokens. Inside
     a step the update equals the sequential per-token delta rule
         M <- M + beta_t * k_t^T (v_t - k_t M)
     computed in parallel (UT transform, as in DeltaNet). Each token corrects what the
     previous ones wrote, and with beta in [0,1] and unit keys the update cannot blow up.
     v1 averaged 256 tokens into one update, ~4x over a 64x64 memory's capacity.
  4. Strength per token. beta_t = theta_h * w_t: a kept token is written at full
     strength however few tokens are kept (v1 divided by the chunk length).
  5. Read keeps its size. Values are stored at unit RMS (keys at unit norm), and
     r = q M goes straight to W_O: a real hit reads back at about full size, an empty or
     irrelevant memory reads near zero (v1 rescaled every read to full size).
  6. Forgetting and momentum (Titans). Retention per step = prod_t (1 - f_t);
     B learns one forget rate per head, C a per-token forget gate. A momentum buffer
     S <- eta S + dM, M <- M + S carries recent updates (eta learned, capped at 0.5).

B vs C (the only difference between them):
                   write weight w_t          read                 forget f_t
    B              1                          always               learned constant per head
    C              sigmoid(beta*(surprise-tau) + u.h + b_w), b_w=+2   gate g = sigmoid(w_r.h + c*conf + b), b=+2 (opens at init)
                                                                    sigmoid(f.h + b_f) per token

W_O is zero-initialised, so B and C start exactly equal to Variant A.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.memory import MemoryModule

DEFAULTS = dict(
    n_heads=8, d_k=64, d_v=64,
    conv=4,               # causal conv width for keys and queries (1 = off)
    value_shift=4,        # key t stored with the next s tokens' values (0 = own value)
    value_taps=True,      # True: learned weight per offset and channel (keeps order), init = mean
    sub_chunk=64,         # tokens per exact delta-rule step
    theta_init=0.5,       # initial write strength per head, in (0, 1)
    decay_init=0.02,      # initial forgetting per 256 tokens (v1 used 0.02 per chunk)
    momentum_init=0.05,   # initial eta
    momentum_max=0.5,     # eta = momentum_max * sigmoid(raw)
    tau=0.5,              # C: surprise threshold of the write gate
    read_gate_bias=2.0,   # C: read gate starts open, sigmoid(2) = 0.88
    write_gate_bias=2.0,  # C: write gate starts open (~0.83 even once memory is full), so C
                          #    starts like B and has to LEARN to skip tokens
    read_strength=1.0,
    detach_chunks=False,  # True = cut gradients between chunks (saves memory)
)
EPS = 1e-6


def _logit(p):
    return math.log(p / (1.0 - p))


@dataclass
class MemoryStateV2:
    M: torch.Tensor       # (B, H, dk, dv) fp32 associative memory
    S: torch.Tensor       # (B, H, dk, dv) fp32 momentum buffer
    tail: torch.Tensor    # (B, conv-1, d_model) last pre-read h of the previous chunk

    def detach(self):
        return MemoryStateV2(self.M.detach(), self.S.detach(), self.tail.detach())


# ---------------------------------------------------------------------------
# Core maths as free functions (unit-tested on toy tensors)
# ---------------------------------------------------------------------------
def delta_step(M, k, err, beta):
    """Exact sequential delta rule over L tokens, computed in parallel.
    M (B,H,dk,dv), k (B,H,L,dk) unit norm, err = v - k M (B,H,L,dv), beta (B,H,L) in [0,1].
    Same result as:  for t in range(L): M = M + beta_t * k_t^T (v_t - k_t M)
    Returns dM, the total change to M."""
    L = k.shape[2]
    kk = k @ k.transpose(-1, -2)                                     # (B,H,L,L)
    A = torch.tril(beta.unsqueeze(-1) * kk, diagonal=-1)
    A = A + torch.eye(L, device=k.device, dtype=k.dtype)
    U = torch.linalg.solve_triangular(A, beta.unsqueeze(-1) * err, upper=False,
                                      unitriangular=True)            # per-token corrections
    return k.transpose(-1, -2) @ U


def shift_values(v, s):
    """v (B,H,T,dv) -> (v_tgt, valid). v_tgt[..., t, :] = mean of v at t+1..t+s inside the
    chunk; valid (1,1,T) = 1 where at least one later token exists. Invalid positions keep
    their own v (so surprise stays finite) and get write weight 0."""
    T = v.shape[2]
    if s <= 0:
        return v, v.new_ones(1, 1, T)
    cs = torch.cat([torch.zeros_like(v[:, :, :1]), v.cumsum(2)], dim=2)   # (B,H,T+1,dv)
    t = torch.arange(T, device=v.device)
    lo, hi = torch.clamp(t + 1, max=T), torch.clamp(t + 1 + s, max=T)
    cnt = (hi - lo).to(v.dtype)
    mean = (cs[:, :, hi] - cs[:, :, lo]) / cnt.clamp(min=1).view(1, 1, T, 1)
    valid = (cnt > 0).to(v.dtype).view(1, 1, T)
    return torch.where(valid.unsqueeze(-1) > 0, mean, v), valid


def future_values(v, taps):
    """Order-aware targets. v (B,H,T,dv) unit-RMS, taps (H,dv,s): weight of token t+j for
    each head/channel. v_tgt[t] = sum_j taps[..., j-1] * v[t+j] over the tokens that exist
    inside the chunk, re-normalised to unit RMS. Unlike a plain mean, swapping two following
    tokens changes the target, so "v1 then v2" and "v2 then v1" are stored differently.
    Returns (v_tgt, valid) like shift_values."""
    B, H, T, dv = v.shape
    s = taps.shape[-1]
    vp = torch.cat([v, v.new_zeros(B, H, s, dv)], dim=2)                   # zeros past the chunk end
    tgt = sum(taps[:, :, j].view(1, H, 1, dv) * vp[:, :, j + 1:j + 1 + T] for j in range(s))
    t = torch.arange(T, device=v.device)
    valid = (t + 1 < T).to(v.dtype).view(1, 1, T)
    tgt = tgt * torch.rsqrt((tgt ** 2).mean(-1, keepdim=True) + EPS)
    return torch.where(valid.unsqueeze(-1) > 0, tgt, v), valid


def surprise(err, v):
    """Normalised prediction error per token, detached. (B,H,L)"""
    return ((err ** 2).sum(-1) / ((v ** 2).sum(-1) + EPS)).detach()


def _lin(lin, x):
    """Projection in the ambient precision (fp16 under autocast), fp32 result."""
    if not torch.is_autocast_enabled():
        x = x.float()
    return lin(x).float()


class MemoryV2(MemoryModule):
    is_null = False

    def __init__(self, cfg, variant: str = "B"):
        super().__init__()
        assert variant in ("B", "C"), "variant A is NullMemory in model/memory.py"
        m = {**DEFAULTS, **cfg.memory}
        self.variant = variant
        self.H, self.dk, self.dv = m["n_heads"], m["d_k"], m["d_v"]
        self.K = int(m["conv"])
        self.shift, self.sub = int(m["value_shift"]), int(m["sub_chunk"])
        self.tau, self.read_strength = m["tau"], m["read_strength"]
        self.momentum_max, self.detach_chunks = m["momentum_max"], bool(m["detach_chunks"])
        assert self.K >= 1 and self.sub >= 1 and self.shift >= 0
        d, H = cfg.d_model, self.H
        self.d_model = d

        self.W_K = nn.Linear(d, H * self.dk, bias=False)
        self.W_Q = nn.Linear(d, H * self.dk, bias=False)
        self.W_V = nn.Linear(d, H * self.dv, bias=False)
        self.W_O = nn.Linear(H * self.dv, d, bias=False)
        nn.init.zeros_(self.W_O.weight)                                   # B, C == A at step 0
        for lin in (self.W_K, self.W_Q, self.W_V):
            nn.init.normal_(lin.weight, std=0.02)
        # depthwise causal convs, initialised to identity (last tap = 1)
        self.conv_k = nn.Conv1d(H * self.dk, H * self.dk, self.K, groups=H * self.dk, bias=False)
        self.conv_q = nn.Conv1d(H * self.dk, H * self.dk, self.K, groups=H * self.dk, bias=False)
        for c in (self.conv_k, self.conv_q):
            nn.init.zeros_(c.weight)
            with torch.no_grad():
                c.weight[:, 0, -1] = 1.0

        self.value_taps = bool(m["value_taps"]) and self.shift > 0
        if self.value_taps:
            self.v_taps = nn.Parameter(torch.full((H, self.dv, self.shift), 1.0 / self.shift))
        self.theta_raw = nn.Parameter(torch.full((H,), _logit(m["theta_init"])))
        self.eta_raw = nn.Parameter(torch.full((H,), _logit(m["momentum_init"] / self.momentum_max)))
        f0 = 1.0 - (1.0 - m["decay_init"]) ** (1.0 / 256)                  # per-token forget rate
        if variant == "B":
            self.forget_raw = nn.Parameter(torch.full((H,), _logit(f0)))
        else:
            self.f = nn.Linear(d, H)                                       # forget gate
            self.u = nn.Linear(d, H)                                       # write gate
            self.w_r = nn.Linear(d, H)                                     # read gate
            self.beta = nn.Parameter(torch.full((H,), 4.0))
            self.c = nn.Parameter(torch.zeros(H))
            for lin in (self.f, self.u, self.w_r):
                nn.init.normal_(lin.weight, std=0.02)
            nn.init.constant_(self.f.bias, _logit(f0))
            nn.init.constant_(self.u.bias, m["write_gate_bias"])
            nn.init.constant_(self.w_r.bias, m["read_gate_bias"])
        self._raw = {}

    # -- state ---------------------------------------------------------------
    def init_state(self, batch_size, device, dtype=None):
        M = torch.zeros(batch_size, self.H, self.dk, self.dv, device=device, dtype=torch.float32)
        tail = torch.zeros(batch_size, self.K - 1, self.d_model, device=device,
                           dtype=dtype or torch.float32)
        return MemoryStateV2(M, torch.zeros_like(M), tail)

    def _keys(self, lin, conv, h, tail):
        """Phrase keys/queries: project [tail; h], causal depthwise conv, unit norm per head.
        Returns (B,H,T,dk) fp32."""
        B, T, _ = h.shape
        x = torch.cat([tail.to(h.dtype), h], dim=1) if self.K > 1 else h
        p = _lin(lin, x)                                                   # (B,T+K-1,H*dk)
        if self.K > 1:
            p = F.conv1d(p.transpose(1, 2), conv.weight.float(), groups=p.shape[-1]).transpose(1, 2)
        return F.normalize(p.view(B, T, self.H, self.dk), dim=-1).permute(0, 2, 1, 3)

    def _new_tail(self, h, tail):
        if self.K == 1:
            return tail
        return torch.cat([tail.to(h.dtype), h], dim=1)[:, -(self.K - 1):]

    # -- read ----------------------------------------------------------------
    def read(self, h, state):
        B, T, _ = h.shape
        q = self._keys(self.W_Q, self.conv_q, h, state.tail)              # (B,H,T,dk)
        with torch.autocast(device_type=h.device.type, enabled=False):
            r = q @ state.M                                                # (B,H,T,dv)
            if self.variant == "C":
                conf = torch.log1p(torch.sqrt((r ** 2).sum(-1) + EPS))     # (B,H,T), finite grad at r=0
                g = torch.sigmoid(_lin(self.w_r, h).transpose(1, 2) + self.c.view(1, -1, 1) * conf)
                r = g.unsqueeze(-1) * r
            else:
                g = None
        out = self.read_strength * _lin(self.W_O, r.permute(0, 2, 1, 3).reshape(B, T, -1))
        self._raw["read_gate"], self._raw["r"] = (None if g is None else g.detach()), r.detach()
        return out.to(h.dtype)

    # -- write ---------------------------------------------------------------
    def write(self, h, state):
        if self.detach_chunks:
            state = state.detach()
        B, T, _ = h.shape
        k = self._keys(self.W_K, self.conv_k, h, state.tail)              # (B,H,T,dk)
        v = _lin(self.W_V, h).view(B, T, self.H, self.dv).permute(0, 2, 1, 3)
        v = v * torch.rsqrt((v ** 2).mean(-1, keepdim=True) + EPS)        # unit-RMS values
        if self.variant == "C":
            u = _lin(self.u, h).transpose(1, 2)                            # (B,H,T)
            f_logit = _lin(self.f, h).transpose(1, 2)                      # (B,H,T)
        with torch.autocast(device_type=h.device.type, enabled=False):
            if self.value_taps:
                v, valid = future_values(v, self.v_taps.float())
            else:
                v, valid = shift_values(v, self.shift)
            theta = torch.sigmoid(self.theta_raw).view(1, -1, 1)
            eta = (self.momentum_max * torch.sigmoid(self.eta_raw)).view(1, -1, 1, 1)
            if self.variant == "B":
                # log(1 - sigmoid(x)) == -softplus(x): same value, never log(0) or an inf gradient
                log_keep_tok = -F.softplus(self.forget_raw).view(1, -1)                # (1,H)
            M, S = state.M, state.S
            ws, ss = [], []
            for a in range(0, T, self.sub):
                ks, vs, val = k[:, :, a:a + self.sub], v[:, :, a:a + self.sub], valid[:, :, a:a + self.sub]
                n = ks.shape[2]
                if self.variant == "B":
                    keep = torch.exp(log_keep_tok * n)                       # (1,H)
                else:
                    keep = torch.exp(-F.softplus(f_logit[:, :, a:a + n]).sum(-1))  # (B,H)
                M = keep[..., None, None] * M
                err = vs - ks @ M
                if self.variant == "C":
                    s = surprise(err, vs)
                    w = torch.sigmoid(self.beta.view(1, -1, 1) * (s - self.tau) + u[:, :, a:a + n]) * val
                    ss.append(s)
                else:
                    w = val.expand(B, self.H, n)
                ws.append(w)
                dM = delta_step(M, ks, err, theta * w)
                S = eta * S + dM
                M = M + S
        self._raw.update(w=torch.cat(ws, 2).detach(), M=M.detach(),
                         s=torch.cat(ss, 2) if ss else None,
                         f=(torch.sigmoid(f_logit).detach() if self.variant == "C" else None))
        return MemoryStateV2(M, S, self._new_tail(h, state.tail))

    @torch.no_grad()
    def stats(self):
        raw, out = self._raw, {}
        if "M" in raw:
            out["M_fro"] = float(raw["M"].flatten(2).norm(dim=-1).mean())
            out["write_rate"] = float(raw["w"].mean())
            out["theta"] = float(torch.sigmoid(self.theta_raw).mean())
            out["eta"] = float((self.momentum_max * torch.sigmoid(self.eta_raw)).mean())
            fr = torch.sigmoid(self.forget_raw) if self.variant == "B" else raw["f"]
            out["forget_per_256"] = float(1 - (1 - fr.mean()) ** 256)
        if raw.get("s") is not None:
            out["mean_surprise"] = float(raw["s"].mean())
        if "r" in raw:
            out["read_norm"] = float(raw["r"].norm(dim=-1).mean())
            out["read_gate"] = 1.0 if raw["read_gate"] is None else float(raw["read_gate"].mean())
        return out

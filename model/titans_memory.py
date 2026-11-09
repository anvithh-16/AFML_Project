"""
model/titans_memory.py: Variants B (always-on) and C (learned gating).
Implements docs/MEMORY_SPEC.md on top of the base class in model/memory.py.

Per chunk the backbone does (team decision Oct 29: write from PRE-read h):
    r     = memory.read(h, state)
    state = memory.write(h, state)
    out   = h + r
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.memory import MemoryModule

DEFAULTS = dict(
    n_heads=8, d_k=64, d_v=64,
    decay=0.02,          # alpha, same for B and C
    tau=0.5,             # write threshold (variant C)
    read_strength=1.0,   # lambda_r
    momentum=0.0,        # 0 = off
)
EPS = 1e-8


@dataclass
class MemoryState:
    M: torch.Tensor                  # (B, H, d_k, d_v) fp32
    S: torch.Tensor | None = None    # momentum buffer, None unless momentum > 0

    def detach(self):
        return MemoryState(self.M.detach(), None if self.S is None else self.S.detach())


# ---------------------------------------------------------------------------
# Core math, kept as free functions so it can be unit-tested on toy tensors
# ---------------------------------------------------------------------------
def retrieve(q, M):
    """q: (B,T,H,dk) unit keys, M: (B,H,dk,dv) -> (B,T,H,dv)"""
    return torch.einsum("bthk,bhkv->bthv", q, M)


def delta_write(M, k, v, w, theta, decay, S=None, momentum=0.0, err=None):
    """
    One error-correcting step on ||k M - v||^2, mean over the T tokens of the chunk.
    M (B,H,dk,dv), k (B,T,H,dk) unit norm, v (B,T,H,dv), w (B,T,H) in [0,1] or None
    (= write everything), theta (H,) in (0,2). err = v - k M may be passed in to save a
    matmul. Returns (M_new, S_new, err).
    """
    T = k.shape[1]
    if err is None:
        err = v - retrieve(k, M)                               # (B,T,H,dv)
    werr = err if w is None else w.unsqueeze(-1) * err
    dM = torch.einsum("bthk,bthv->bhkv", k, werr)
    dM = dM * (theta.view(1, -1, 1, 1) / T)
    if momentum > 0.0:
        S = dM if S is None else momentum * S + dM
        upd = S
    else:
        upd = dM
    return (1.0 - decay) * M + upd, S, err


def surprise(err, v):
    """Normalised prediction error per token per head, detached. (B,T,H)"""
    return ((err ** 2).sum(-1) / ((v ** 2).sum(-1) + EPS)).detach()



def _lin(lin, h):
    """Run a projection in the ambient precision (fp16 under CUDA autocast, so it uses
    tensor cores), return fp32 for the state maths. Without autocast, run in fp32."""
    if not torch.is_autocast_enabled():
        h = h.float()
    return lin(h).float()


class TitansMemory(MemoryModule):
    is_null = False

    def __init__(self, cfg, variant: str = "B"):
        super().__init__()
        assert variant in ("B", "C"), "variant A is NullMemory in model/memory.py"
        m = {**DEFAULTS, **cfg.memory}
        self.variant = variant
        self.H, self.dk, self.dv = m["n_heads"], m["d_k"], m["d_v"]
        self.decay, self.tau = m["decay"], m["tau"]
        self.read_strength, self.momentum = m["read_strength"], m["momentum"]
        d = cfg.d_model

        self.W_K = nn.Linear(d, self.H * self.dk, bias=False)
        self.W_Q = nn.Linear(d, self.H * self.dk, bias=False)
        self.W_V = nn.Linear(d, self.H * self.dv, bias=False)
        self.W_O = nn.Linear(self.H * self.dv, d, bias=False)
        nn.init.zeros_(self.W_O.weight)                       # B, C == A at step 0
        self.theta_raw = nn.Parameter(torch.zeros(self.H))    # theta = 2*sigmoid -> 1.0

        if variant == "C":
            self.u = nn.Linear(d, self.H)                     # write gate: u_h . h + b_w
            self.beta = nn.Parameter(torch.full((self.H,), 4.0))
            self.w_r = nn.Linear(d, self.H)                   # need gate: w_h . h + b_r
            self.c = nn.Parameter(torch.zeros(self.H))
            for lin in (self.u, self.w_r):
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)

        self._raw = {}

    # -- state -------------------------------------------------------------
    def init_state(self, batch_size, device, dtype=None):
        M = torch.zeros(batch_size, self.H, self.dk, self.dv, device=device, dtype=torch.float32)
        S = torch.zeros_like(M) if self.momentum > 0 else None
        return MemoryState(M, S)

    def _theta(self):
        return 2.0 * torch.sigmoid(self.theta_raw)

    # -- read --------------------------------------------------------------
    def read(self, h, state):
        B, T, _ = h.shape
        q = F.normalize(_lin(self.W_Q, h).view(B, T, self.H, self.dk), dim=-1)
        with torch.autocast(device_type=h.device.type, enabled=False):
            r = retrieve(q, state.M)                                   # (B,T,H,dv)
            r_hat = r * torch.rsqrt((r ** 2).mean(-1, keepdim=True) + EPS)
            if self.variant == "C":
                conf = torch.log1p(r.norm(dim=-1))                     # (B,T,H)
                g = torch.sigmoid(_lin(self.w_r, h) + self.c * conf)
                r_hat = g.unsqueeze(-1) * r_hat
            else:
                g = None                                               # always on
        out = self.read_strength * _lin(self.W_O, r_hat.reshape(B, T, -1))
        self._raw["read_gate"], self._raw["r"] = g, r.detach()          # reduced lazily in stats()
        return out.to(h.dtype)

    # -- write -------------------------------------------------------------
    def write(self, h, state):
        B, T, _ = h.shape
        k = F.normalize(_lin(self.W_K, h).view(B, T, self.H, self.dk), dim=-1)
        v = _lin(self.W_V, h).view(B, T, self.H, self.dv)
        with torch.autocast(device_type=h.device.type, enabled=False):
            err = v - retrieve(k, state.M)
            if self.variant == "C":
                s = surprise(err, v)
                w = torch.sigmoid(self.beta * (s - self.tau) + _lin(self.u, h))
            else:
                s, w = None, None                                      # B: no gate, no surprise
            M_new, S_new, _ = delta_write(state.M, k, v, w, self._theta(), self.decay,
                                          state.S, self.momentum, err=err)
        self._raw["w"] = None if w is None else w.detach()
        self._raw["s"] = s
        self._raw["M"] = M_new.detach()
        return MemoryState(M_new, S_new)

    def stats(self):
        """Reductions happen here (when logging), not on every forward step."""
        raw, out = self._raw, {}
        if "M" in raw:
            out["M_fro"] = float(raw["M"].flatten(2).norm(dim=-1).mean())
            out["write_rate"] = 1.0 if raw["w"] is None else float(raw["w"].mean())
        if raw.get("s") is not None:
            out["mean_surprise"] = float(raw["s"].mean())
        if "r" in raw:
            out["read_norm"] = float(raw["r"].norm(dim=-1).mean())
            out["read_gate"] = 1.0 if raw["read_gate"] is None else float(raw["read_gate"].detach().mean())
        return out


class AlwaysOnMemory(TitansMemory):
    """Variant B: write every token, always read."""
    def __init__(self, cfg):
        super().__init__(cfg, "B")


class GatedMemory(TitansMemory):
    """Variant C: learned write (surprise) gate and need gate."""
    def __init__(self, cfg):
        super().__init__(cfg, "C")

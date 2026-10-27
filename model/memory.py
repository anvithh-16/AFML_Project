"""
Memory plug-in interface (Person B owns everything below NullMemory).

The backbone calls a memory module between block `insert_after_layer` and the
next block, one chunk of `chunk_size` tokens at a time, in order:

    state = memory.init_state(B, device, dtype)
    for each chunk t:
        r     = memory.read(h_t, state)      # uses only chunks < t  (causal)
        h_t   = h_t + r                      # residual add
        state = memory.write(h_t, state)     # store chunk t for later chunks

Rules a memory module must keep (tests/test_model.py checks them on any module):
  * read() must not depend on chunk t itself, only on `state`.
  * read() returns a tensor shaped like h: (B, C, d_model). The last chunk of a
    sequence may be shorter than chunk_size.
  * State resets at each training sequence; at eval it persists across all
    chunks of one document.

Variant A = NullMemory, so A, B and C share one code path.
"""
from typing import Any

import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    """Base class. Subclass this for Variant B (always-on) and C (gated)."""

    is_null = False

    def init_state(self, batch_size: int, device, dtype) -> Any:
        raise NotImplementedError

    def read(self, h: torch.Tensor, state: Any) -> torch.Tensor:
        raise NotImplementedError

    def write(self, h: torch.Tensor, state: Any) -> Any:
        raise NotImplementedError

    def stats(self) -> dict:
        """Optional scalars to log (write rate, gate means, state norm...)."""
        return {}


class NullMemory(MemoryModule):
    """Variant A: no memory. The backbone skips the chunk loop entirely."""

    is_null = True

    def init_state(self, batch_size, device, dtype):
        return None

    def read(self, h, state):
        return torch.zeros_like(h)

    def write(self, h, state):
        return state


def build_memory(model_cfg) -> MemoryModule:
    variant = model_cfg.memory.get("variant", "A")
    if variant == "A":
        return NullMemory()
    # Person B: register Variant B / C here, e.g.
    #   from model.titans_memory import AlwaysOnMemory, GatedMemory
    #   if variant == "B": return AlwaysOnMemory(model_cfg)
    #   if variant == "C": return GatedMemory(model_cfg)
    raise NotImplementedError(f"memory variant {variant!r} is not implemented yet (Person B)")

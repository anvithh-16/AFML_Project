"""
Batch loader over the uint16 token binaries. Shared by all variants.

    from data.loader import TokenLoader
    loader = TokenLoader("artifacts/data", seq_len=1024, batch_size=32, device="cuda")
    x, y = loader.get_batch("train")        # (B, T) int64, y = x shifted by 1
    for x, y in loader.val_batches(max_batches=50): ...

Train batches sample random windows (fast, no epoch bookkeeping; at 600M tokens
we see the corpus about once). Val batches are fixed, non-overlapping windows,
so val loss is comparable across variants and across runs.
"""
import json
import os

import numpy as np
import torch


class TokenLoader:
    def __init__(self, data_dir, seq_len, batch_size, device="cpu", seed=1337):
        self.seq_len, self.batch_size, self.device = seq_len, batch_size, device
        self.meta = json.load(open(os.path.join(data_dir, "meta.json")))
        self.paths = {s: os.path.join(data_dir, f"{s}.bin") for s in ("train", "val")}
        self.rng = np.random.default_rng(seed)
        for s, p in self.paths.items():
            n = os.path.getsize(p) // 2
            expected = self.meta["tokens"][s]
            assert n == expected, (f"{s}.bin has {n:,} tokens but meta.json says {expected:,}; "
                                   "data prep was interrupted or re-run. Re-run prepare_pretrain.py.")
            assert n > seq_len + 1, f"{s}.bin has only {n} tokens (< seq_len+1)"

    def _mm(self, split):
        # re-open each call: avoids memmap memory growth on long runs
        return np.memmap(self.paths[split], dtype=np.uint16, mode="r")

    def _to_device(self, x, y):
        x, y = torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))
        if str(self.device).startswith("cuda"):
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device)

    def get_batch(self, split="train"):
        data, T = self._mm(split), self.seq_len
        ix = self.rng.integers(0, len(data) - T - 1, size=self.batch_size)
        x = np.stack([data[i:i + T] for i in ix])
        y = np.stack([data[i + 1:i + 1 + T] for i in ix])
        return self._to_device(x, y)

    def val_batches(self, max_batches=None):
        data, T, B = self._mm("val"), self.seq_len, self.batch_size
        n_windows = (len(data) - 1) // T
        starts = np.arange(n_windows) * T
        nb = len(starts) // B if max_batches is None else min(max_batches, len(starts) // B)
        for b in range(nb):
            ix = starts[b * B:(b + 1) * B]
            x = np.stack([data[i:i + T] for i in ix])
            y = np.stack([data[i + 1:i + 1 + T] for i in ix])
            yield self._to_device(x, y)

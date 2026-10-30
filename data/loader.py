"""
Batch loader over the uint16 token binaries. Shared by all variants.

    from data.loader import TokenLoader
    loader = TokenLoader("artifacts/data", seq_len=1024, batch_size=32, device="cuda",
                         synth_dir="artifacts/synth", synth_frac=0.05)
    x, y = loader.get_batch("train")        # (B, T) int64, y = x shifted by 1
    for x, y in loader.val_batches(max_batches=50): ...

Train batches sample random windows (fast, no epoch bookkeeping; at 600M tokens
we see the corpus about once). Val batches are fixed, non-overlapping windows,
so val loss is comparable across variants and across runs.

Synthetic mixing (optional): each TRAIN row is, with probability synth_frac, a
synthetic long-range example placed at the start of the row, with a random
FineWeb window filling the rest. Val is never mixed, so val loss stays pure
FineWeb and comparable with or without mixing.
"""
import json
import os

import numpy as np
import torch


class TokenLoader:
    def __init__(self, data_dir, seq_len, batch_size, device="cpu", seed=1337,
                 synth_dir=None, synth_frac=0.0):
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

        self.synth_frac, self.synth_meta = 0.0, None
        if synth_dir and synth_frac > 0:
            self.synth_meta = json.load(open(os.path.join(synth_dir, "synth_meta.json")))
            assert self.synth_meta["tokenizer_sha"] == self.meta["tokenizer_sha"], \
                "synthetic data was made with a different tokenizer"
            self.synth_path = os.path.join(synth_dir, "synth.bin")
            self.synth_offsets = np.load(os.path.join(synth_dir, "synth_offsets.npy"))
            assert os.path.getsize(self.synth_path) // 2 == self.synth_offsets[-1] == self.synth_meta["tokens"]
            assert np.diff(self.synth_offsets).max() <= seq_len + 1, "a synthetic example is longer than seq_len"
            self.synth_frac = synth_frac

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
        rows = np.stack([data[i:i + T + 1] for i in ix])
        if split == "train" and self.synth_frac > 0:
            synth = np.memmap(self.synth_path, dtype=np.uint16, mode="r")
            for r in np.nonzero(self.rng.random(self.batch_size) < self.synth_frac)[0]:
                j = int(self.rng.integers(0, len(self.synth_offsets) - 1))
                ex = synth[self.synth_offsets[j]:self.synth_offsets[j + 1]]
                rows[r, :len(ex)] = ex          # example first; the original FineWeb window fills the rest
        return self._to_device(rows[:, :-1], rows[:, 1:])

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

from __future__ import annotations
from dataclasses import dataclass
import random
import torch
import torch.nn.functional as F


@dataclass
class MemoryItem:
    sample: dict
    score: float


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float(F.cosine_similarity(a[None], b[None]).item())


class GradientReplayBuffer:
    """
    Implements the pseudo-code shown on slide 19.

    c = max_i cosine(g, G_i) + 1   # make score positive
    if memory full and c < 1 (negative cosine before +1):
        i ~ P(i)=C_i/sum_j C_j
        r~U(0,1)
        if r < C_i/(C_i+c):
            replace M_i with new sample and C_i <- c
    else if not full:
        append new sample, score c

    The slide's Update(x,y,M) step is handled by the caller's normal optimizer.
    """
    def __init__(self, capacity: int, probe_size: int = 8, seed: int = 0):
        self.capacity = capacity
        self.probe_size = probe_size
        self.rng = random.Random(seed)
        self.items: list[MemoryItem] = []

    def __len__(self):
        return len(self.items)

    def random_subset(self):
        if not self.items:
            return []
        n = min(self.probe_size, len(self.items))
        return self.rng.sample(self.items, n)

    def update(
        self,
        sample: dict,
        current_grad: torch.Tensor,
        probe_grads: list[torch.Tensor],
    ):
        if probe_grads:
            c = max(_cosine(current_grad, g) for g in probe_grads) + 1.0
        else:
            c = 1.0

        if len(self.items) >= self.capacity:
            if c < 1.0:
                scores = torch.tensor(
                    [max(it.score, 1e-8) for it in self.items],
                    dtype=torch.float32,
                )
                probs = scores / scores.sum()
                idx = int(torch.multinomial(probs, 1))
                ci = self.items[idx].score
                if self.rng.random() < ci / max(ci + c, 1e-8):
                    copied = {k: v.detach().cpu().clone() for k, v in sample.items()}
                    self.items[idx] = MemoryItem(copied, c)
                    return True
            return False

        copied = {k: v.detach().cpu().clone() for k, v in sample.items()}
        self.items.append(MemoryItem(copied, c))
        return True

    def sample_batch(self, n: int, device):
        if not self.items:
            return None
        chosen = self.rng.sample(self.items, min(n, len(self.items)))
        keys = chosen[0].sample.keys()
        return {
            k: torch.stack([it.sample[k] for it in chosen]).to(device)
            for k in keys
        }

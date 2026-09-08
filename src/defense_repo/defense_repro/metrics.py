from __future__ import annotations
import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean())


@torch.no_grad()
def macro_f1(logits: torch.Tensor, y: torch.Tensor, n_classes: int) -> float:
    pred = logits.argmax(dim=-1)
    values = []
    for c in range(n_classes):
        tp = ((pred == c) & (y == c)).sum().float()
        fp = ((pred == c) & (y != c)).sum().float()
        fn = ((pred != c) & (y == c)).sum().float()
        denom = 2 * tp + fp + fn
        values.append(torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(denom)))
    return float(torch.stack(values).mean())


@torch.no_grad()
def mae_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    # Treat ordered class index as sentiment score for demo purposes.
    pred = logits.argmax(dim=-1).float()
    return float((pred - y.float()).abs().mean())


@torch.no_grad()
def corr_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1).float()
    target = y.float()
    pred = pred - pred.mean()
    target = target - target.mean()
    den = torch.sqrt((pred.square().sum() * target.square().sum()).clamp_min(1e-12))
    return float((pred * target).sum() / den)

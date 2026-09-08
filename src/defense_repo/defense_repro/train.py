from __future__ import annotations
import torch
import torch.nn.functional as F
from .models.text_baselines import multiclass_hinge_loss


def train_epoch(model, loader, optimizer, device, loss_kind="ce"):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        if hasattr(model, "loss") and model.__class__.__name__ == "PromptImageTextClassifier":
            loss = model.loss(batch)
        else:
            logits = model(batch)
            if loss_kind == "hinge":
                loss = multiclass_hinge_loss(logits, batch["label"])
            else:
                loss = F.cross_entropy(logits, batch["label"])

        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * batch["label"].shape[0]
        n += batch["label"].shape[0]
    return total / max(n, 1)


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits.append(model(batch).cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(logits), torch.cat(labels)

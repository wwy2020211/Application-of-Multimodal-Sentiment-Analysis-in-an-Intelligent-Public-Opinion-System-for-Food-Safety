from __future__ import annotations
import torch
import torch.nn.functional as F


class RobustnessQueryStrategy:
    """
    Reproduces slides 24-25:
      unlabeled sample + perturbed unlabeled sample -> model
      if result changes -> query expert/manual labeling.

    Exact perturbation type is not specified in the defense slides.
    This implementation perturbs continuous modalities with Gaussian noise and
    randomly replaces a small fraction of text tokens.

    Selection priority:
      1) prediction changed (exact slide criterion)
      2) among changed samples, higher Jensen-Shannon divergence first
    """
    def __init__(self, text_replace_prob=0.08, noise_std=0.10, vocab_size=512):
        self.text_replace_prob = text_replace_prob
        self.noise_std = noise_std
        self.vocab_size = vocab_size

    @torch.no_grad()
    def perturb(self, batch):
        out = {k: v.clone() for k, v in batch.items()}
        if "text" in out:
            mask = torch.rand_like(out["text"].float()) < self.text_replace_prob
            repl = torch.randint(
                0, self.vocab_size, out["text"].shape, device=out["text"].device
            )
            out["text"][mask] = repl[mask]
        for k in ("vision", "audio"):
            if k in out:
                out[k] = out[k] + self.noise_std * torch.randn_like(out[k])
        return out

    @torch.no_grad()
    def score(self, model, batch):
        clean = model(batch)
        noisy_batch = self.perturb(batch)
        noisy = model(noisy_batch)

        pc = F.softmax(clean, dim=-1)
        pn = F.softmax(noisy, dim=-1)
        m = 0.5 * (pc + pn)
        js = 0.5 * (
            F.kl_div(m.log(), pc, reduction="none").sum(dim=-1)
            + F.kl_div(m.log(), pn, reduction="none").sum(dim=-1)
        )
        changed = clean.argmax(dim=-1) != noisy.argmax(dim=-1)
        return changed, js

    @torch.no_grad()
    def query(self, model, batch, budget: int):
        changed, js = self.score(model, batch)
        # Large bonus enforces slide criterion: changed predictions come first.
        rank_score = js + changed.float() * (js.max().detach() + 1.0)
        return torch.topk(rank_score, k=min(budget, len(rank_score))).indices

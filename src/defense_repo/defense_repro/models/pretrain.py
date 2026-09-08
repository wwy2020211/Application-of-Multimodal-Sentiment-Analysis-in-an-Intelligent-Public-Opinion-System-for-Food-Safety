from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMLMNSP(nn.Module):
    """
    Small self-contained MLM + NSP pretraining model.

    The defense slides cite MLM/NSP as generic pretraining tasks but do not
    provide an exact text backbone, so this is a compact runnable equivalent.
    """
    def __init__(self, vocab_size=512, d_model=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, d_model)  # last ID is [MASK]
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.mlm = nn.Linear(d_model, vocab_size)
        self.nsp = nn.Linear(d_model, 2)

    def forward(self, tokens):
        return self.encoder(self.embed(tokens))

    def loss(self, a, b, is_next, mask_prob=0.15):
        # Concatenate two segments; use first token pooled for NSP.
        tokens = torch.cat([a, b], dim=1)
        target = tokens.clone()
        mask = torch.rand_like(tokens.float()) < mask_prob
        masked = tokens.clone()
        masked[mask] = self.vocab_size

        h = self(masked)
        mlm_loss = F.cross_entropy(self.mlm(h)[mask], target[mask]) if mask.any() else 0.0
        nsp_loss = F.cross_entropy(self.nsp(h[:, 0]), is_next)
        return mlm_loss + nsp_loss

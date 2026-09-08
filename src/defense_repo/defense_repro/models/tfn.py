import torch
import torch.nn as nn
from .common import TextEncoder, SeqEncoder


class TensorFusionNetwork(nn.Module):
    """
    TFN baseline with explicit tensor outer product.

    h_t'=[1,h_t], h_v'=[1,h_v], h_a'=[1,h_a]
    fusion = h_t' ⊗ h_v' ⊗ h_a'

    torch.einsum performs the fusion in one expression.
    """
    def __init__(
        self,
        vocab_size=512,
        d_model=24,
        vision_dim=24,
        audio_dim=20,
        hidden=12,
        n_classes=3,
    ):
        super().__init__()
        self.text = TextEncoder(vocab_size, d_model, hidden)
        self.vision = SeqEncoder(vision_dim, hidden)
        self.audio = SeqEncoder(audio_dim, hidden)
        fusion_dim = (hidden + 1) ** 3
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, batch):
        _, t = self.text(batch["text"])
        _, v = self.vision(batch["vision"])
        _, a = self.audio(batch["audio"])

        one = torch.ones((t.shape[0], 1), device=t.device, dtype=t.dtype)
        t = torch.cat([one, t], dim=-1)
        v = torch.cat([one, v], dim=-1)
        a = torch.cat([one, a], dim=-1)

        z = torch.einsum("bi,bj,bk->bijk", t, v, a).flatten(1)
        return self.head(z)

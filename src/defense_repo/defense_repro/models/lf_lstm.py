import torch
import torch.nn as nn
from .common import TextEncoder, SeqEncoder


class LFLSTM(nn.Module):
    """
    Late-fusion LSTM:
      one recurrent encoder per modality -> concatenate final states -> classifier.
    """
    def __init__(
        self,
        vocab_size=512,
        d_model=24,
        vision_dim=24,
        audio_dim=20,
        hidden=32,
        n_classes=3,
    ):
        super().__init__()
        self.text = TextEncoder(vocab_size, d_model, hidden)
        self.vision = SeqEncoder(vision_dim, hidden)
        self.audio = SeqEncoder(audio_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, batch):
        _, ht = self.text(batch["text"])
        _, hv = self.vision(batch["vision"])
        _, ha = self.audio(batch["audio"])
        return self.head(torch.cat([ht, hv, ha], dim=-1))

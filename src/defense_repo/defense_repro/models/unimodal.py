import torch
import torch.nn as nn
from .common import TextEncoder, SeqEncoder


class UnimodalClassifier(nn.Module):
    """only-language / only-vision / only-audio ablations."""
    def __init__(
        self,
        modality: str,
        vocab_size=512,
        d_model=24,
        vision_dim=24,
        audio_dim=20,
        hidden=32,
        n_classes=3,
    ):
        super().__init__()
        self.modality = modality
        if modality == "language":
            self.encoder = TextEncoder(vocab_size, d_model, hidden)
        elif modality == "vision":
            self.encoder = SeqEncoder(vision_dim, hidden)
        elif modality == "audio":
            self.encoder = SeqEncoder(audio_dim, hidden)
        else:
            raise ValueError("modality must be language/vision/audio")
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, batch):
        key = "text" if self.modality == "language" else self.modality
        _, h = self.encoder(batch[key])
        return self.head(h)

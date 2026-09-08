from __future__ import annotations
import torch
from torch.utils.data import Dataset
from .config import SyntheticConfig


class SyntheticMultimodalDataset(Dataset):
    """
    Random but learnable multimodal sentiment data.

    The defense slides describe positive / negative / neutral labels and
    text-image-audio inputs, but do not publish the original enterprise data.
    This dataset is only an offline reproducibility scaffold.
    """
    def __init__(self, cfg: SyntheticConfig):
        super().__init__()
        g = torch.Generator().manual_seed(cfg.seed)
        self.cfg = cfg

        y = torch.randint(0, cfg.n_classes, (cfg.n_samples,), generator=g)

        # Class-dependent token regions create learnable text signal.
        bucket = max(cfg.vocab_size // cfg.n_classes, 2)
        base = y[:, None] * bucket
        text = base + torch.randint(
            0, bucket, (cfg.n_samples, cfg.text_len), generator=g
        )
        text %= cfg.vocab_size

        # Shared latent class signal for continuous modalities.
        latent = torch.randn(cfg.n_classes, 8, generator=g)
        z = latent[y]

        wv = torch.randn(8, cfg.vision_dim, generator=g) / (8 ** 0.5)
        wa = torch.randn(8, cfg.audio_dim, generator=g) / (8 ** 0.5)

        vision_center = z @ wv
        audio_center = z @ wa

        vision = vision_center[:, None, :] + 0.7 * torch.randn(
            cfg.n_samples, cfg.vision_len, cfg.vision_dim, generator=g
        )
        audio = audio_center[:, None, :] + 0.8 * torch.randn(
            cfg.n_samples, cfg.audio_len, cfg.audio_dim, generator=g
        )

        self.text = text.long()
        self.vision = vision.float()
        self.audio = audio.float()
        self.labels = y.long()

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, idx):
        return {
            "text": self.text[idx],
            "vision": self.vision[idx],
            "audio": self.audio[idx],
            "label": self.labels[idx],
        }


def batch_to_device(batch, device):
    return {
        k: v.to(device)
        for k, v in batch.items()
    }

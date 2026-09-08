import torch
import torch.nn as nn


class EFLSTM(nn.Module):
    """
    Early-fusion LSTM baseline:
      encode text tokens -> concatenate text/image/audio at each time step
      -> one shared LSTM -> classifier.

    This follows the conventional EF-LSTM meaning used in multimodal
    sentiment baselines; the defense slides only list its name/results.
    """
    def __init__(
        self,
        vocab_size=512,
        text_dim=16,
        vision_dim=24,
        audio_dim=20,
        hidden=32,
        n_classes=3,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, text_dim)
        self.rnn = nn.LSTM(
            text_dim + vision_dim + audio_dim,
            hidden,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, batch):
        t = self.text_embed(batch["text"])
        v = batch["vision"]
        a = batch["audio"]
        L = min(t.shape[1], v.shape[1], a.shape[1])
        x = torch.cat([t[:, :L], v[:, :L], a[:, :L]], dim=-1)
        _, (h, _) = self.rnn(x)
        return self.head(h[-1])

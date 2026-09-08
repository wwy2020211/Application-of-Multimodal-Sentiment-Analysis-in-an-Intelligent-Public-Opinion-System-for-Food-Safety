import torch
import torch.nn as nn
import torch.nn.functional as F


class TextLSTM(nn.Module):
    def __init__(self, vocab_size=512, d_model=32, hidden=32, n_classes=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.LSTM(d_model, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, batch):
        x = self.embed(batch["text"])
        _, (h, _) = self.rnn(x)
        return self.head(h[-1])


class TextCNN(nn.Module):
    def __init__(self, vocab_size=512, d_model=32, channels=32, n_classes=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, channels, k, padding=k // 2)
            for k in (3, 5, 7)
        ])
        self.head = nn.Linear(channels * 3, n_classes)

    def forward(self, batch):
        x = self.embed(batch["text"]).transpose(1, 2)
        feats = [torch.relu(c(x)).amax(dim=-1) for c in self.convs]
        return self.head(torch.cat(feats, dim=-1))


class LinearSVM(nn.Module):
    """
    Multiclass linear SVM operating on bag-of-words normalized token counts.
    """
    def __init__(self, vocab_size=512, n_classes=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(vocab_size, n_classes)

    def features(self, tokens):
        x = torch.zeros(
            (tokens.shape[0], self.vocab_size),
            device=tokens.device,
            dtype=torch.float32,
        )
        x.scatter_add_(
            1,
            tokens,
            torch.ones_like(tokens, dtype=torch.float32),
        )
        return x / x.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(self, batch):
        return self.linear(self.features(batch["text"]))


def multiclass_hinge_loss(scores, y, margin=1.0):
    true = scores.gather(1, y[:, None])
    margins = (scores - true + margin).clamp_min(0)
    margins.scatter_(1, y[:, None], 0.0)
    return margins.sum(dim=1).mean()

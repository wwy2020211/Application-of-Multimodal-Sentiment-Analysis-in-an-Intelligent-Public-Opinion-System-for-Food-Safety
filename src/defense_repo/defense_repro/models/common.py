from __future__ import annotations
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.LSTM(d_model, hidden, batch_first=True)

    def forward(self, tokens):
        x = self.embed(tokens)
        seq, (h, _) = self.rnn(x)
        return seq, h[-1]


class SeqEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True)

    def forward(self, x):
        x = torch.tanh(self.proj(x))
        seq, (h, _) = self.rnn(x)
        return seq, h[-1]

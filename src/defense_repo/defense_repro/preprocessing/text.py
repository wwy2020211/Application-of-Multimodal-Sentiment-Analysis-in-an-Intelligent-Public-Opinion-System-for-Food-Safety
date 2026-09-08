from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_skipgram_pairs(tokens: torch.Tensor, window: int = 2):
    """
    Turn token IDs into center-context pairs for a small Word2Vec/Skip-Gram demo.
    tokens: [N,T]
    """
    centers, contexts = [], []
    for row in tokens.tolist():
        for i, c in enumerate(row):
            lo, hi = max(0, i-window), min(len(row), i+window+1)
            for j in range(lo, hi):
                if j != i:
                    centers.append(c)
                    contexts.append(row[j])
    return torch.tensor(centers), torch.tensor(contexts)


class SkipGramWord2Vec(nn.Module):
    """
    Minimal Word2Vec Skip-Gram implementation.
    The slides only specify one-hot -> word2vec, not exact training settings.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 32):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, center_ids):
        return self.out(self.in_embed(center_ids))

    def loss(self, center_ids, context_ids):
        return F.cross_entropy(self(center_ids), context_ids)

    def embeddings(self):
        return self.in_embed.weight.detach()

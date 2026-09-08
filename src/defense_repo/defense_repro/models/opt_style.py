from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class OPTStyleTriModal(nn.Module):
    """
    Lightweight tri-modal Transformer inspired by the slide's OPT diagram:
      text / vision / audio encoders
        -> cross-modal Transformer
        -> masked language / vision / audio pretraining heads
        -> downstream sentiment finetuning

    This is a self-contained architecture-level reproduction, not the original
    pretrained OPT checkpoint (the defense slides do not specify checkpoint,
    exact dimensions, tokenizer, or pretraining corpus).
    """
    def __init__(
        self,
        vocab_size=512,
        vision_dim=24,
        audio_dim=20,
        d_model=32,
        n_classes=3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)

        self.modality_embed = nn.Embedding(3, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mask_text = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_vision = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_audio = nn.Parameter(torch.zeros(1, 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=0.1,
        )
        self.cross = nn.TransformerEncoder(layer, num_layers=2)

        self.classifier = nn.Linear(d_model, n_classes)
        self.text_decoder = nn.Linear(d_model, vocab_size)
        self.vision_decoder = nn.Linear(d_model, vision_dim)
        self.audio_decoder = nn.Linear(d_model, audio_dim)

    def _tokens(self, batch):
        t = self.text_embed(batch["text"])
        v = self.vision_proj(batch["vision"])
        a = self.audio_proj(batch["audio"])

        t = t + self.modality_embed.weight[0]
        v = v + self.modality_embed.weight[1]
        a = a + self.modality_embed.weight[2]
        return t, v, a

    def forward(self, batch):
        t, v, a = self._tokens(batch)
        cls = self.cls.expand(t.shape[0], -1, -1)
        x = torch.cat([cls, t, v, a], dim=1)
        h = self.cross(x)
        return self.classifier(h[:, 0])

    def masked_pretrain_loss(self, batch, mask_prob=0.15):
        t, v, a = self._tokens(batch)
        B, Lt, _ = t.shape
        Lv, La = v.shape[1], a.shape[1]
        device = t.device

        mt = torch.rand(B, Lt, device=device) < mask_prob
        mv = torch.rand(B, Lv, device=device) < mask_prob
        ma = torch.rand(B, La, device=device) < mask_prob

        tm = torch.where(mt[..., None], self.mask_text, t)
        vm = torch.where(mv[..., None], self.mask_vision, v)
        am = torch.where(ma[..., None], self.mask_audio, a)

        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tm, vm, am], dim=1)
        h = self.cross(x)

        ht = h[:, 1:1+Lt]
        hv = h[:, 1+Lt:1+Lt+Lv]
        ha = h[:, 1+Lt+Lv:]

        loss = torch.zeros((), device=device)

        if mt.any():
            loss = loss + F.cross_entropy(
                self.text_decoder(ht)[mt],
                batch["text"][mt],
            )
        if mv.any():
            loss = loss + F.mse_loss(
                self.vision_decoder(hv)[mv],
                batch["vision"][mv],
            )
        if ma.any():
            loss = loss + F.mse_loss(
                self.audio_decoder(ha)[ma],
                batch["audio"][ma],
            )
        return loss

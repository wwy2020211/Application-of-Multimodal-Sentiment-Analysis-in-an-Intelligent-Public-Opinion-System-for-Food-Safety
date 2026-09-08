from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptImageTextClassifier(nn.Module):
    """
    Lightweight offline reconstruction of the slide's image-text prompt model.

    Slide structure:
      text + prompt "This message is [EMO]" -> TXT encoder -> [EMO] -> softmax
      image -> IMG encoder
      txt_feature + img_feature -> auxiliary softmax loss

    This module supports:
      use_prompt=True/False
      finetune_encoders=True/False

    Those switches reproduce the slide's CLIP-prompt / without-prompt /
    without-finetune ablation logic, but this is NOT a pretrained CLIP
    checkpoint because the defense file does not specify the exact checkpoint.
    """
    def __init__(
        self,
        vocab_size=512,
        d_model=32,
        vision_dim=24,
        n_classes=3,
        use_prompt=True,
        finetune_encoders=True,
    ):
        super().__init__()
        self.use_prompt = use_prompt
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.emo_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True,
            dim_feedforward=d_model * 2,
        )
        self.text_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.image_encoder = nn.Sequential(
            nn.Linear(vision_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.prompt_head = nn.Linear(d_model, n_classes)
        self.fusion_head = nn.Linear(d_model * 2, n_classes)

        if not finetune_encoders:
            for module in [self.text_embed, self.text_encoder, self.image_encoder]:
                for p in module.parameters():
                    p.requires_grad = False

    def encode(self, batch):
        t = self.text_embed(batch["text"])
        if self.use_prompt:
            emo = self.emo_token.expand(t.shape[0], -1, -1)
            seq = torch.cat([t, emo], dim=1)
            h = self.text_encoder(seq)
            emo_feature = h[:, -1]
            txt_feature = h[:, :-1].mean(dim=1)
        else:
            h = self.text_encoder(t)
            txt_feature = h.mean(dim=1)
            emo_feature = txt_feature

        img_feature = self.image_encoder(batch["vision"]).mean(dim=1)
        return emo_feature, txt_feature, img_feature

    def forward(self, batch):
        emo_feature, txt_feature, img_feature = self.encode(batch)
        return self.prompt_head(emo_feature)

    def loss(self, batch):
        emo_feature, txt_feature, img_feature = self.encode(batch)
        y = batch["label"]
        main = F.cross_entropy(self.prompt_head(emo_feature), y)
        aux = F.cross_entropy(
            self.fusion_head(torch.cat([txt_feature, img_feature], dim=-1)),
            y,
        )
        return main + 0.5 * aux

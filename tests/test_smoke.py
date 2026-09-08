import torch

from defense_repro.config import SyntheticConfig
from defense_repro.data import SyntheticMultimodalDataset
from defense_repro.models import (
    EFLSTM, LFLSTM, TensorFusionNetwork,
    PromptImageTextClassifier, OPTStyleTriModal,
)
from defense_repro.active import RobustnessQueryStrategy


def make_batch():
    cfg = SyntheticConfig(n_samples=8)
    ds = SyntheticMultimodalDataset(cfg)
    keys = ["text", "vision", "audio", "label"]
    batch = {k: torch.stack([ds[i][k] for i in range(8)]) for k in keys}
    return cfg, batch


def test_models_forward():
    cfg, batch = make_batch()
    models = [
        EFLSTM(vocab_size=cfg.vocab_size, vision_dim=cfg.vision_dim, audio_dim=cfg.audio_dim),
        LFLSTM(vocab_size=cfg.vocab_size, vision_dim=cfg.vision_dim, audio_dim=cfg.audio_dim),
        TensorFusionNetwork(vocab_size=cfg.vocab_size, vision_dim=cfg.vision_dim, audio_dim=cfg.audio_dim),
        PromptImageTextClassifier(vocab_size=cfg.vocab_size, vision_dim=cfg.vision_dim),
        OPTStyleTriModal(vocab_size=cfg.vocab_size, vision_dim=cfg.vision_dim, audio_dim=cfg.audio_dim),
    ]
    for m in models:
        y = m(batch)
        assert y.shape == (8, 3)


def test_active_query():
    cfg, batch = make_batch()
    m = EFLSTM(vocab_size=cfg.vocab_size, vision_dim=cfg.vision_dim, audio_dim=cfg.audio_dim)
    q = RobustnessQueryStrategy(vocab_size=cfg.vocab_size)
    idx = q.query(m, batch, 3)
    assert idx.numel() == 3

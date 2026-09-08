from dataclasses import dataclass


@dataclass
class SyntheticConfig:
    n_samples: int = 256
    n_classes: int = 3
    vocab_size: int = 512
    text_len: int = 16
    vision_len: int = 16
    audio_len: int = 16
    vision_dim: int = 24
    audio_dim: int = 20
    seed: int = 2022


@dataclass
class ModelConfig:
    n_classes: int = 3
    vocab_size: int = 512
    d_model: int = 32
    vision_dim: int = 24
    audio_dim: int = 20
    hidden_dim: int = 32
    dropout: float = 0.1

import torch
from torch.utils.data import DataLoader
from defense_repro.config import SyntheticConfig
from defense_repro.data import SyntheticMultimodalDataset
from defense_repro.models import EFLSTM
from defense_repro.active import RobustnessQueryStrategy


def main():
    cfg = SyntheticConfig(n_samples=64)
    ds = SyntheticMultimodalDataset(cfg)
    loader = DataLoader(ds, batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EFLSTM(
        vocab_size=cfg.vocab_size,
        vision_dim=cfg.vision_dim,
        audio_dim=cfg.audio_dim,
        n_classes=cfg.n_classes,
    ).to(device)

    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    query = RobustnessQueryStrategy(vocab_size=cfg.vocab_size)
    idx = query.query(model, batch, budget=8)
    changed, js = query.score(model, batch)
    print("queried indices:", idx.tolist())
    print("changed predictions:", int(changed.sum()))
    print("max JS:", float(js.max()))


if __name__ == "__main__":
    main()

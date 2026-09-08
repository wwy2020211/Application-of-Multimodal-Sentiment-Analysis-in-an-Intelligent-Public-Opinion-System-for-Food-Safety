import torch
from torch.utils.data import DataLoader
from defense_repro.config import SyntheticConfig
from defense_repro.data import SyntheticMultimodalDataset
from defense_repro.models import OPTStyleTriModal


def main():
    cfg = SyntheticConfig(n_samples=64)
    ds = SyntheticMultimodalDataset(cfg)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OPTStyleTriModal(
        vocab_size=cfg.vocab_size,
        vision_dim=cfg.vision_dim,
        audio_dim=cfg.audio_dim,
        n_classes=cfg.n_classes,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    loss = model.masked_pretrain_loss(batch)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    print("masked tri-modal pretrain loss:", float(loss))


if __name__ == "__main__":
    main()

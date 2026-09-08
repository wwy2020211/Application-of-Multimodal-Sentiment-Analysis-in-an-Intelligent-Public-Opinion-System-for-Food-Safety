import argparse
import torch
from torch.utils.data import DataLoader, random_split

from defense_repro.config import SyntheticConfig
from defense_repro.data import SyntheticMultimodalDataset
from defense_repro.models import (
    EFLSTM, LFLSTM, TensorFusionNetwork,
    UnimodalClassifier, TextLSTM, TextCNN, LinearSVM,
    PromptImageTextClassifier, OPTStyleTriModal,
)
from defense_repro.train import train_epoch, collect_logits
from defense_repro.metrics import accuracy, macro_f1, mae_from_logits, corr_from_logits


def build_model(name, cfg):
    kw = dict(
        vocab_size=cfg.vocab_size,
        vision_dim=cfg.vision_dim,
        audio_dim=cfg.audio_dim,
        n_classes=cfg.n_classes,
    )
    if name == "ef_lstm":
        return EFLSTM(**kw)
    if name == "lf_lstm":
        return LFLSTM(**kw)
    if name == "tfn":
        return TensorFusionNetwork(**kw)
    if name == "only_vision":
        return UnimodalClassifier("vision", **kw)
    if name == "only_audio":
        return UnimodalClassifier("audio", **kw)
    if name == "only_language":
        return UnimodalClassifier("language", **kw)
    if name == "text_lstm":
        return TextLSTM(cfg.vocab_size, n_classes=cfg.n_classes)
    if name == "text_cnn":
        return TextCNN(cfg.vocab_size, n_classes=cfg.n_classes)
    if name == "svm":
        return LinearSVM(cfg.vocab_size, cfg.n_classes)
    if name == "clip_prompt":
        return PromptImageTextClassifier(
            vocab_size=cfg.vocab_size,
            vision_dim=cfg.vision_dim,
            n_classes=cfg.n_classes,
            use_prompt=True,
            finetune_encoders=True,
        )
    if name == "without_prompt":
        return PromptImageTextClassifier(
            vocab_size=cfg.vocab_size,
            vision_dim=cfg.vision_dim,
            n_classes=cfg.n_classes,
            use_prompt=False,
            finetune_encoders=True,
        )
    if name == "without_finetune":
        return PromptImageTextClassifier(
            vocab_size=cfg.vocab_size,
            vision_dim=cfg.vision_dim,
            n_classes=cfg.n_classes,
            use_prompt=True,
            finetune_encoders=False,
        )
    if name == "opt_finetune":
        return OPTStyleTriModal(**kw)
    raise KeyError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ef_lstm")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--samples", type=int, default=192)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    cfg = SyntheticConfig(n_samples=args.samples)
    ds = SyntheticMultimodalDataset(cfg)
    n_train = int(len(ds) * 0.8)
    train_ds, test_ds = random_split(
        ds, [n_train, len(ds) - n_train],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = build_model(args.model, cfg).to(device)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-3,
    )
    loss_kind = "hinge" if args.model == "svm" else "ce"

    for e in range(args.epochs):
        loss = train_epoch(model, train_loader, opt, device, loss_kind)
        print(f"epoch {e+1}: loss={loss:.4f}")

    logits, y = collect_logits(model, test_loader, device)
    print({
        "ACC": round(accuracy(logits, y), 4),
        "F1": round(macro_f1(logits, y, cfg.n_classes), 4),
        "MAE": round(mae_from_logits(logits, y), 4),
        "CORR": round(corr_from_logits(logits, y), 4),
    })


if __name__ == "__main__":
    main()

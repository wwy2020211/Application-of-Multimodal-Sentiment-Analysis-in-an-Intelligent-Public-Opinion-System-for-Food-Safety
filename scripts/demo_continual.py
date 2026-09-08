"""
Small demo of the slide-19 gradient replay buffer.

For runtime simplicity we use a text classifier and treat its embedding,
recurrent encoder, and head as the three gradient groups.
"""
import torch
import torch.nn.functional as F
from defense_repro.models import TextLSTM
from defense_repro.continual import GradientReplayBuffer, grouped_gradient_signature


def one_sample(seed, vocab=128, length=12, classes=3):
    g = torch.Generator().manual_seed(seed)
    return {
        "text": torch.randint(0, vocab, (length,), generator=g),
        "label": torch.randint(0, classes, (), generator=g),
    }


def grad_signature(model, sample, device):
    batch = {
        "text": sample["text"][None].to(device),
        "label": sample["label"][None].to(device),
    }
    logits = model(batch)
    loss = F.cross_entropy(logits, batch["label"])
    return grouped_gradient_signature(
        loss,
        model.embed.parameters(),
        model.rnn.parameters(),
        model.head.parameters(),
        weights=(0.3, 0.4, 0.3),
    ).detach()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextLSTM(vocab_size=128, d_model=16, hidden=16, n_classes=3).to(device)
    buf = GradientReplayBuffer(capacity=12, probe_size=4, seed=0)

    grad_cache = []
    for step in range(24):
        sample = one_sample(step)
        g = grad_signature(model, sample, device)

        probes = []
        for item in buf.random_subset():
            probes.append(grad_signature(model, item.sample, device))

        buf.update(sample, g, probes)

    print("memory size:", len(buf))
    print("scores:", [round(x.score, 4) for x in buf.items])


if __name__ == "__main__":
    main()

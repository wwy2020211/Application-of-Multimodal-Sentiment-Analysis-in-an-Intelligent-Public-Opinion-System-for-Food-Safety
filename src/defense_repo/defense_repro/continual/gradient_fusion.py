from __future__ import annotations
import torch


def _flatten(grads):
    vec = [g.reshape(-1) for g in grads if g is not None]
    if not vec:
        return torch.empty(0)
    return torch.cat(vec)


def grouped_gradient_signature(
    loss,
    text_params,
    image_params,
    downstream_params,
    weights=(1.0, 1.0, 1.0),
):
    """
    Reproduces the slide's:
      G_multimodal = a G_textencoder + b G_imageencoder + c G_downstreamnet

    Since the parameter groups live in different coordinate blocks, the stable
    implementation is weighted concatenation of the three gradient blocks.
    Cosine similarity on this signature is then used by replay selection.
    """
    groups = [list(text_params), list(image_params), list(downstream_params)]
    parts = []
    for gi, (params, w) in enumerate(zip(groups, weights)):
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        flat = _flatten(grads)
        if flat.numel():
            parts.append(float(w) * flat)
    return torch.cat(parts) if parts else torch.empty(0, device=loss.device)

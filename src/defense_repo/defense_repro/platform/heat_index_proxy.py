import torch


def propagation_heat_proxy(
    reposts: torch.Tensor,
    comments: torch.Tensor,
    likes: torch.Tensor,
    source_weight: torch.Tensor | float = 1.0,
    weights=(0.45, 0.35, 0.20),
):
    """
    PROXY ONLY.

    The defense slides mention a "网络传播热度指数算法" but provide no formula.
    This configurable log-count index is included only so the project has a
    runnable platform component; it is NOT claimed as the thesis's exact heat
    formula.
    """
    wr, wc, wl = weights
    sw = torch.as_tensor(source_weight, device=reposts.device, dtype=torch.float32)
    return sw * (
        wr * torch.log1p(reposts.float())
        + wc * torch.log1p(comments.float())
        + wl * torch.log1p(likes.float())
    )

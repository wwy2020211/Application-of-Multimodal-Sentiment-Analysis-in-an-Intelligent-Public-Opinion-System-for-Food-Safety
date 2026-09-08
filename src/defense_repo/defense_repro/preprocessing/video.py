import torch
import torch.nn.functional as F


def grayscale(images: torch.Tensor) -> torch.Tensor:
    """
    RGB -> grayscale.
    images: [B,3,H,W] or [3,H,W]
    """
    squeeze = images.ndim == 3
    if squeeze:
        images = images.unsqueeze(0)
    if images.shape[1] != 3:
        raise ValueError("expected RGB channel dimension=3")
    weights = torch.tensor(
        [0.299, 0.587, 0.114],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    out = (images * weights).sum(dim=1, keepdim=True)
    return out.squeeze(0) if squeeze else out


def smooth_image(gray: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Mean smoothing, matching the slide's generic '图像平滑' requirement.
    """
    squeeze = gray.ndim == 3
    if squeeze:
        gray = gray.unsqueeze(0)
    if gray.shape[1] != 1:
        raise ValueError("expected grayscale [B,1,H,W]")
    k = kernel_size
    kernel = torch.ones((1, 1, k, k), device=gray.device, dtype=gray.dtype) / (k * k)
    out = F.conv2d(gray, kernel, padding=k // 2)
    return out.squeeze(0) if squeeze else out

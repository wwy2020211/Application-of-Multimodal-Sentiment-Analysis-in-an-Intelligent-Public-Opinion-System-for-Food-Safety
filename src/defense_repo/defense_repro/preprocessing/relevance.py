from __future__ import annotations
import torch


@torch.no_grad()
def kmeans_binary_relevance(
    x: torch.Tensor,
    iters: int = 30,
    seed: int = 0,
):
    """
    Two-cluster K-means for the slide's "用聚类法进行食品相关/不相关二分类".

    The defense slides do not specify which clustering algorithm was used;
    K-means is an explicit reconstruction choice.

    x: [N,D]
    return labels [N], centers [2,D]
    """
    g = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g, device=x.device)[:2]
    centers = x[idx].clone()

    for _ in range(iters):
        dist = torch.cdist(x.float(), centers.float())
        labels = dist.argmin(dim=1)
        new = []
        for k in range(2):
            members = x[labels == k]
            new.append(members.float().mean(dim=0) if len(members) else centers[k])
        new_centers = torch.stack(new)
        if torch.allclose(new_centers, centers.float(), atol=1e-5):
            break
        centers = new_centers

    return labels, centers

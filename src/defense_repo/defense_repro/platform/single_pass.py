from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def single_pass_cluster(
    x: torch.Tensor,
    cosine_threshold: float = 0.75,
):
    """
    Single-pass online clustering for news similarity.

    Each incoming vector joins the most similar existing centroid if similarity
    exceeds threshold; otherwise it starts a new cluster.
    """
    x = F.normalize(x.float(), dim=-1)
    centroids = []
    counts = []
    labels = []

    for v in x:
        if not centroids:
            centroids.append(v.clone())
            counts.append(1)
            labels.append(0)
            continue

        C = F.normalize(torch.stack(centroids), dim=-1)
        sim = C @ v
        best = int(sim.argmax())

        if float(sim[best]) >= cosine_threshold:
            n = counts[best]
            centroids[best] = (centroids[best] * n + v) / (n + 1)
            counts[best] += 1
            labels.append(best)
        else:
            centroids.append(v.clone())
            counts.append(1)
            labels.append(len(centroids) - 1)

    return torch.tensor(labels, device=x.device), torch.stack(centroids)

"""
Adaptive K-means using silhouette coefficient bounds (PyTorch implementation)

Main idea:
- Run K-means with an initial K (k_init).
- After convergence (or each few iterations), compute per-sample silhouette scores.
- Compute mean silhouette per cluster.
- If a cluster's mean silhouette < lower_thresh -> merge it with its nearest neighbor cluster (centroid distance).
- If cluster's mean silhouette >= upper_thresh -> consider splitting (optional: not implemented here to keep stable).
- Repeat until no cluster needs merging or until a minimum K is reached.

This implementation is vectorized in PyTorch and should run on CPU or GPU.

Usage example at bottom demonstrates on synthetic data.
"""

from typing import Tuple, Optional
from collections import OrderedDict
import torch


def kmeans_torch(X: torch.Tensor, K: int, n_iters: int = 100, tol: float = 1e-4, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple K-means in PyTorch.

    Args:
        X: (N, D) data tensor.
        K: number of clusters.
        n_iters: max iterations.
        tol: tolerance on centroid movement for early stop.
        device: torch.device or None.

    Returns:
        centers: (K, D) cluster centroids.
        labels: (N,) cluster assignments.
    """
    if device is None:
        device = X.device
    X = X.to(device)
    N, D = X.shape

    # initialize centers by sampling K points
    indices = torch.randperm(N, device=device)[:K]
    centers = X[indices].clone()

    labels = torch.zeros(N, dtype=torch.long, device=device)
    for it in range(n_iters):
        # compute distances (N, K)
        dists = torch.cdist(X, centers, p=2)  # Euclidean
        new_labels = torch.argmin(dists, dim=1)

        # update centers
        new_centers = torch.zeros_like(centers)
        for k in range(K):
            members = (new_labels == k)
            if members.any():
                new_centers[k] = X[members].mean(dim=0)
            else:
                # reinitialize empty cluster to a random point
                new_centers[k] = X[torch.randint(0, N, (1,), device=device)]

        shift = torch.norm(new_centers - centers, dim=1).max()
        centers = new_centers
        labels = new_labels
        if shift < tol:
            break
    return centers, labels


def silhouette_scores_torch(X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute silhouette score per sample.

    Args:
        X: (N, D)
        labels: (N,) ints from 0..K-1

    Returns:
        scores: (N,) silhouette coefficient in [-1, 1]
    """
    N = X.shape[0]
    device = X.device
    labels = labels.to(device)
    unique_labels = torch.unique(labels)
    K = unique_labels.shape[0]

    # precompute pairwise distances
    dists = torch.cdist(X, X, p=2)  # (N, N)

    # compute a_i: mean intra-cluster distance for each sample
    a = torch.zeros(N, device=device)
    b = torch.zeros(N, device=device)

    for lab in unique_labels:
        mask = (labels == lab)
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        # distances among members
        if idx.numel() == 1:
            a[idx] = 0.0
        else:
            # for each member, mean distance to other members
            sub = dists[idx][:, idx]  # (m, m)
            # exclude self (zero)
            # sum then divide by (m-1)
            m = idx.numel()
            a[idx] = (sub.sum(dim=1) / (m - 1))

    # compute b_i: minimum mean distance to points in other clusters
    # for each cluster, get mean distance from every sample to that cluster
    # we compute mean distances to each cluster, then take min over clusters != own
    mean_dists_to_cluster = torch.zeros((N, K), device=device)
    for j, lab in enumerate(unique_labels):
        mask = (labels == lab)
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            mean_dists_to_cluster[:, j] = float('inf')
        else:
            # mean distance from each point to points in cluster j
            sub = dists[:, idx]  # (N, m_j)
            mean_dists_to_cluster[:, j] = sub.mean(dim=1)

    # for each sample, b is min over clusters excluding its own cluster column
    # find column index of its own cluster in unique_labels
    label_to_col = {int(l.item()): i for i, l in enumerate(unique_labels)}
    own_cols = torch.tensor([label_to_col[int(l.item())] for l in labels], device=device)

    # set own cluster mean to +inf temporarily so min excludes it
    rows = torch.arange(N, device=device)
    mean_dists_masked = mean_dists_to_cluster.clone()
    mean_dists_masked[rows, own_cols] = float('inf')
    b_vals, _ = mean_dists_masked.min(dim=1)
    b = b_vals

    # silhouette s = (b - a) / max(a, b)
    denom = torch.maximum(a, b)
    # handle denom == 0
    zero_mask = (denom == 0)
    s = torch.zeros(N, device=device)
    nonzero = ~zero_mask
    s[nonzero] = (b[nonzero] - a[nonzero]) / denom[nonzero]
    # if denom==0, silhouette is set to 0
    return s


def adaptive_kmeans_silhouette_bounds(X: torch.Tensor,
                                      k_init: int,
                                      lower_thresh: float = 0.25,
                                      upper_thresh: float = 0.75,
                                      min_k: int = 2,
                                      max_iter: int = 20,
                                      kmeans_inner_iters: int = 100,
                                      device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adaptive K-means that merges clusters whose mean silhouette < lower_thresh.

    Args:
        X: (N, D) data tensor.
        k_init: initial K.
        lower_thresh: clusters with mean silhouette < lower_thresh will be merged.
        upper_thresh: reserved for potential splits (not implemented here).
        min_k: minimum allowed number of clusters.
        max_iter: maximum number of outer adapt iterations.
        kmeans_inner_iters: iterations for inner K-means.
        device: torch device.

    Returns:
        centers: final centers (K_final, D)
        labels: final labels (N,)
    """
    if device is None:
        device = X.device
    X = X.to(device)

    K = k_init
    centers, labels = kmeans_torch(X, K, n_iters=kmeans_inner_iters, device=device)

    for adapt_it in range(max_iter):
        scores = silhouette_scores_torch(X, labels)
        unique = torch.unique(labels)
        cluster_mean_scores = {}
        for lab in unique:
            mask = (labels == lab)
            if mask.sum() == 0:
                cluster_mean_scores[int(lab.item())] = float('inf')
            else:
                cluster_mean_scores[int(lab.item())] = float(scores[mask].mean().item())

        # find clusters to merge (those below lower_thresh)
        to_merge = [lab for lab, sc in cluster_mean_scores.items() if sc < lower_thresh]
        if not to_merge or K <= min_k:
            # nothing to merge
            break

        # we'll merge each such cluster into its nearest other centroid
        # compute centroid distances
        centers = centers.cpu()
        lab_to_idx = {int(l.item()): i for i, l in enumerate(unique)}
        keep_centers = centers.clone()
        for lab in to_merge:
            # index of this lab in centers: find matching center by label order
            if lab not in lab_to_idx:
                continue
            idx = lab_to_idx[lab]
            # distances to other centers
            c = centers[idx:idx+1]  # (1, D)
            other_idx = [i for i in range(centers.shape[0]) if i != idx]
            if len(other_idx) == 0:
                continue
            other = centers[other_idx]
            d = torch.cdist(c, other).squeeze(0)
            nearest_pos = d.argmin().item()
            nearest_center_idx = other_idx[nearest_pos]

            # reassign points of `lab` to nearest_center_idx
            # we need to map labels back: unique contains actual labels ordering
            actual_label = unique[idx].item()
            actual_nearest_label = unique[nearest_center_idx].item()
            labels[labels == actual_label] = actual_nearest_label

        # After reassignment, compress labels to contiguous 0..K'-1 and recompute centers via kmeans
        unique = torch.unique(labels)
        mapping = {int(l.item()): i for i, l in enumerate(unique)}
        labels_mapped = labels.clone()
        for old, new in mapping.items():
            labels_mapped[labels == old] = new
        K = len(unique)
        # recompute centers as mean of assigned points
        new_centers = torch.zeros((K, X.shape[1]), device=device)
        for i in range(K):
            mask = (labels_mapped == i)
            if mask.sum() == 0:
                new_centers[i] = X[torch.randint(0, X.shape[0], (1,), device=device)]
            else:
                new_centers[i] = X[mask].mean(dim=0)

        centers = new_centers
        labels = labels_mapped

        # run a few Kmeans iterations to refine
        centers, labels = kmeans_torch(X, K, n_iters=kmeans_inner_iters, tol=1e-4, device=device)

    return centers, labels

def auto_kmeans_by_silhouette(
    X,
    k_min=2,
    k_max=20,
    lower_thresh=0.4,
    upper_thresh=0.6,
    max_outer_iters=10,
    device=None
):
    X = X.to(device or X.device)
    K = k_min
    best_K, best_score = None, -1
    best_centers, best_labels = None,None
    for _ in range(max_outer_iters):
        centers, labels = kmeans_torch(X, K, n_iters=100, device=device)
        scores = silhouette_scores_torch(X, labels)
        mean_s = scores.mean().item()

        print(f"K={K}, mean silhouette={mean_s:.3f}")

        # 保存最优
        if mean_s > best_score:
            best_score, best_K = mean_s, K
            best_centers, best_labels = centers.clone(), labels.clone()

        if lower_thresh <= mean_s <= upper_thresh:
            break
        elif mean_s < lower_thresh and K < k_max:
            K += 1
        elif mean_s > upper_thresh and K > k_min:
            K -= 1
        else:
            break

    clusters = OrderedDict()
    for i in best_labels:
        clusters[int(i)] = []
    for i in range(len(X)):
        clusters[int(best_labels[i])].append(i)
    print(f"Selected K={best_K}, silhouette={best_score:.3f}")
    return clusters, best_centers, best_labels


if __name__ == '__main__':
    # demo on synthetic data
    torch.manual_seed(42)
    N = 1500
    D = 2
    # create three gaussian blobs
    x1 = torch.randn(N // 3, D) * 0.4 + torch.tensor([0.0, 0.0])
    x2 = torch.randn(N // 3, D) * 0.5 + torch.tensor([3.0, 3.0])
    x3 = torch.randn(N // 3, D) * 0.3 + torch.tensor([0.0, 4.0])
    X = torch.cat([x1, x2, x3], dim=0)

    centers, labels = adaptive_kmeans_silhouette_bounds(X, k_init=6, lower_thresh=0.2, upper_thresh=0.8, min_k=2)
    print('Final K =', centers.shape[0])
    # print centers
    print(centers)

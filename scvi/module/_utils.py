import torch
import numpy as np
from typing import Optional

from scvi.nn import one_hot


def iterate(obj, func):
    """Iterates over an object and applies a function to each element."""
    t = type(obj)
    if t is list or t is tuple:
        return t([iterate(o, func) for o in obj])
    else:
        return func(obj) if obj is not None else None


def broadcast_labels(y, *o, n_broadcast=-1):
    """Utility for the semi-supervised setting.

    If y is defined(labelled batch) then one-hot encode the labels (no broadcasting needed)
    If y is undefined (unlabelled batch) then generate all possible labels (and broadcast other arguments if not None)
    """
    if not len(o):
        raise ValueError("Broadcast must have at least one reference argument")
    if y is None:
        ys = enumerate_discrete(o[0], n_broadcast)
        new_o = iterate(
            o,
            lambda x: x.repeat(n_broadcast, 1)
            if len(x.size()) == 2
            else x.repeat(n_broadcast),
        )
    else:
        ys = one_hot(y, n_broadcast)
        new_o = o
    return (ys,) + new_o


def enumerate_discrete(x, y_dim):
    """Enumerate discrete variables."""

    def batch(batch_size, label):
        labels = torch.ones(batch_size, 1, device=x.device, dtype=torch.long) * label
        return one_hot(labels, y_dim)

    batch_size = x.size(0)
    return torch.cat([batch(batch_size, i) for i in range(y_dim)])


def masked_softmax(weights, mask, dim=-1, eps=1e-30):
    """Computes a softmax of ``weights`` along ``dim`` where ``mask is True``.

    Adds a small ``eps`` term in the numerator and denominator to avoid zero division.
    Taken from: https://discuss.pytorch.org/t/apply-mask-softmax/14212/15.
    Pytorch issue tracked at: https://github.com/pytorch/pytorch/issues/55056.
    """
    weight_exps = torch.exp(weights)
    masked_exps = weight_exps.masked_fill(mask == 0, eps)
    masked_sums = masked_exps.sum(dim, keepdim=True) + eps
    return masked_exps / masked_sums


def create_random_proportion(
    n_classes: int, n_non_zero: Optional[int] = None
) -> np.ndarray:
    """Create a random proportion vector of size n_classes.

    The n_non_zero parameter allows to set the number
    of non-zero components of the random discrete density vector.
    """
    if n_non_zero is None:
        n_non_zero = n_classes

    proportion_vector = np.zeros(
        n_classes,
    )

    proportion_vector[:n_non_zero] = np.random.rand(n_non_zero)

    proportion_vector = proportion_vector / proportion_vector.sum()
    return np.random.permutation(proportion_vector)


def compute_ground_truth_proportions(y_pseudobulk, n_labels, n_cells_per_pseudobulk):
    """Compute the ground truth cell type proportions for each pseudobulk of the batch."""
    all_proportions = []
    for ground_truth in y_pseudobulk:
        unique_indices, counts = ground_truth.unique(return_counts=True)
        if len(counts) < n_labels:
            # then not all labels are present in pseudobulk, but we need to specify these proportions of 0
            unique_indices = unique_indices.int().tolist()
            counts =  [counts[unique_indices.index(j)].item() if j in unique_indices else 0 for j in range(n_labels)]
            counts = torch.tensor(counts).to(device="cuda")
        proportions = counts.float() / n_cells_per_pseudobulk
        all_proportions.append(proportions)
    return all_proportions


def compute_signature(y, x_):
    """Compute the signature matrix and signature matrix mask of the batch."""
    x_signature_mask = []
    unique_indices, counts = y.unique(return_counts=True)
    for cell_type in unique_indices:
        idx = (y == cell_type).flatten()
        x_signature_mask.append(idx.tolist())
    x_signature_mask = torch.Tensor(x_signature_mask).to(device="cuda")
    x_signature_ = torch.matmul(x_signature_mask, x_)/counts.unsqueeze(-1)
    return counts, x_signature_mask, x_signature_


def get_mean_pearsonr_torch(x, y):
    """
    Mimics `scipy.stats.pearsonr`
    Rewritten to adapt to 2D tensors, to compute the mean of the 1D correlations along the first axis.
    Arguments
    ---------
    x : 2D torch.Tensor
    y : 2D torch.Tensor
    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y

    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x, axis=1).unsqueeze(dim=1)
    mean_y = torch.mean(y, axis=1).unsqueeze(dim=1)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = (xm*ym).sum(dim=1)
    r_den = torch.norm(xm, p=2, dim=1) * torch.norm(ym, p=2, dim=1)
    r_val = r_num / r_den
    return torch.mean(r_val)
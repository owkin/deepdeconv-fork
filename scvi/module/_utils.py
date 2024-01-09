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


def get_pearsonr_torch(x, y):
    """
    Mimics `scipy.stats.pearsonr`
    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor
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
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def run_incompatible_value_checks(
    pseudo_bulk, loss_computation, use_batch_norm, mixup_penalty, gene_likelihood
):
    """Check the values of the categorical variables to run MixUpVI are compatible.
    The first 4 checks will only be relevant when pseudobulk will not be computed both
    in encoder and decoder (right now, computed in both). Until then, use_batch_norm
    should be None.
    """
    if (
        pseudo_bulk == "pre_encoded"
        and loss_computation == "latent_space"
        and use_batch_norm in ["encoder", "both"]
    ):
        raise ValueError(
            "MixUpVI cannot use batch normalization there, as the batch size of pseudobulk is 1."
        )
    elif (
        pseudo_bulk == "pre_encoded"
        and loss_computation == "reconstructed_space"
        and use_batch_norm != "none"
    ):
        raise ValueError(
            "MixUpVI cannot use batch normalization there, as the batch size of pseudobulk is 1."
        )
    elif pseudo_bulk == "post_inference" and loss_computation == "latent_space":
        raise ValueError(
            "Pseudo bulk needs to be pre-encoded to compute the MixUp loss in the latent space."
        )
    elif (
        pseudo_bulk == "post_inference"
        and loss_computation == "reconstructed_space"
        and use_batch_norm in ["decoder", "both"]
    ):
        raise ValueError(
            "MixUpVI cannot use batch normalization there, as the batch size of pseudobulk is 1."
        )
    if (
        mixup_penalty == "kl"
        and loss_computation != "latent_space"
        and gene_likelihood == "zinb"
    ):
        raise NotImplementedError(
            "The KL divergence between ZINB distributions for the MixUp loss is not "
            "implemented."
        )


def run_categorical_value_checks(
    encode_covariates,
    encode_cont_covariates,
    use_batch_norm,
    signature_type,
    loss_computation,
    pseudo_bulk,
    mixup_penalty,
    dispersion,
    gene_likelihood,
):
    """Check the values and types of the categorical variables to run MixUpVI."""
    assert isinstance(
        encode_covariates, bool
    ), "ENCODE_COVARIATES should be of type bool"
    assert isinstance(
        encode_cont_covariates, bool
    ), "ENCODE_CONT_COVARIATES should be of type bool"
    assert isinstance(use_batch_norm, str), "BATCH_NORM should be of type string"
    assert isinstance(signature_type, str), "SIGNATURE_TYPE should be of type string"
    assert isinstance(
        loss_computation, str
    ), "LOSS_COMPUTATION should be of type string"
    assert isinstance(pseudo_bulk, str), "PSEUDO_BULK should be of type string"
    assert isinstance(mixup_penalty, str), "MIXUP_PENALTY should be of type string"
    assert isinstance(dispersion, str), "DISPERSION should be of type string"
    assert isinstance(gene_likelihood, str), "GENE_LIKELIHOOD should be of type string"
    if encode_covariates:
        raise NotImplementedError(
            "For now, MixUpVI only uses cell types as categorical covariates without encoding them."
        )
    if use_batch_norm not in ["encoder", "decoder", "none", "both"]:
        raise ValueError(
            "Batch normalization can only be part of ['encoder', 'decoder', 'none', 'both']."
        )
    if signature_type not in ["pre_encoded", "post_inference"]:
        raise ValueError(
            "Signature type can only be part of ['pre_encoded', 'post_inference']."
        )
    if loss_computation not in ["latent_space", "reconstructed_space"]:
        raise ValueError(
            "Loss computation can only be part of ['latent_space', 'reconstructed_space']."
        )
    if pseudo_bulk not in ["pre_encoded", "post_inference"]:
        raise ValueError(
            "Pseudo bulk computation can only be part of ['pre_encoded', 'post_inference']."
        )
    if mixup_penalty not in ["l2", "kl"]:
        raise ValueError("Mixup penalty can only be part of ['l2', 'kl'].")
    if dispersion not in ["gene", "gene_cell"]:
        raise ValueError(
            "The dispersion parameter can only be part of ['gene', 'gene_cell'], "
            "not gene-label nor gene-batch because categorical covariates don't make "
            "sense for pseudobulk."
        )
    if gene_likelihood not in ["zinb", "nb", "poisson"]:
        raise ValueError(
            "The dispersion parameter can only be part of ['zinb', 'nb', 'poisson']."
        )
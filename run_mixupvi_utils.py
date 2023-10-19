"""MixUpVI utils."""

import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl


def run_categorical_value_checks(
    cell_group,
    cat_cov,
    cont_cov,
    encode_covariates,
    encode_cont_covariates,
    use_batch_norm,
    signature_type,
    loss_computation,
    pseudo_bulk,
    pseudobulk_loss,
    dispersion,
):
    """Check the values and types of the categorical variables to run MixUpVI."""
    assert isinstance(cell_group, str), "CELL_GROUP should be of type string"
    assert isinstance(cat_cov, list), "CAT_COV should be of type list"
    assert (
        isinstance(cont_cov, list) or cont_cov == None
    ), "CONT_COV should be None or type list"
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
    assert isinstance(pseudobulk_loss, str), "PSEUDOBULK_LOSS should be of type string"
    assert isinstance(dispersion, str), "DISPERSION should be of type string"
    if cell_group not in [
        "primary_groups",
        "precise_groups",
        "updated_granular_groups",
    ]:
        raise NotImplementedError(
            "For now, only these cell category granularities are implemented."
        )
    if len(cat_cov) > 1:
        raise NotImplementedError(
            "For now, MixUpVI works with only the cell type as categorical covariate."
        )
    if encode_covariates:
        raise NotImplementedError(
            "For now, MixUpVI only uses cell types as categorical covariates without encoding them."
        )
    if use_batch_norm not in ["encoder", "decoder", "none", "both"]:
        raise ValueError(
            "Batch normalization can only be part of ['encoder', 'decoder', 'none', 'both']."
        )
    if signature_type not in ["pre_encoded", "latent_space"]:
        raise ValueError(
            "Signature type can only be part of ['pre_encoded', 'latent_space']."
        )
    if loss_computation not in ["latent_space", "reconstructed_space", "both"]:
        raise ValueError(
            "Loss computation can only be part of ['latent_space', 'reconstructed_space', 'both']."
        )
    if pseudo_bulk not in ["pre_encoded", "latent_space"]:
        raise ValueError(
            "Pseudo bulk computation can only be part of ['pre_encoded', 'latent_space']."
        )
    if pseudobulk_loss not in ["l2", "kl"]:
        raise ValueError("Pseudobulk loss can only be part of ['l2', 'kl'].")
    if dispersion not in ["gene", "gene_cell"]:
        raise ValueError(
            "The dispersion parameter can only be part of ['gene', 'gene_cell'], "
            "not gene-label nor gene-batch because categorical covariates don't make "
            "sense for pseudobulk."
        )


def run_incompatible_value_checks(
    pseudo_bulk, loss_computation, use_batch_norm, pseudobulk_loss
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
        and loss_computation in ["reconstructed_space", "both"]
        and use_batch_norm != "none"
    ):
        raise ValueError(
            "MixUpVI cannot use batch normalization there, as the batch size of pseudobulk is 1."
        )
    elif pseudo_bulk == "latent_space" and loss_computation != "reconstructed_space":
        raise ValueError(
            "Pseudo bulk needs to be pre-encoded to compute the MixUp loss in the latent space."
        )
    elif (
        pseudo_bulk == "latent_space"
        and loss_computation == "reconstructed_space"
        and use_batch_norm in ["decoder", "both"]
    ):
        raise ValueError(
            "MixUpVI cannot use batch normalization there, as the batch size of pseudobulk is 1."
        )
    if pseudobulk_loss == "kl" and loss_computation != "latent_space":
        raise NotImplementedError(
            "The KL divergence between ZINB distributions for the MixUp loss is implemented."
        )  # what are the parameters of average of independant ZINB ?


def pearsonr_torch(x, y):
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


def compute_l2_mixup_loss(single_cells, pseudobulk):
    """
    Compute L2 loss between average of single cells and pseudobulk
    """
    mean_single_cells = torch.mean(single_cells, axis=0)
    pseudobulk_loss = torch.sum((pseudobulk - mean_single_cells) ** 2)
    return pseudobulk_loss, mean_single_cells


def compute_kl_mixup_loss(single_cells_distrib, pseudobulk_distrib):
    """
    Compute KL divergence between average of single cells distrib and pseudobulk distrib
    """
    mean_averaged_cells = single_cells_distrib.mean.mean(axis=0)
    std_averaged_cells = single_cells_distrib.variance.sum(axis=0).sqrt() / len(
        single_cells_distrib.loc
    )
    averaged_cells_distrib = Normal(mean_averaged_cells, std_averaged_cells)
    pseudobulk_loss = kl(averaged_cells_distrib, pseudobulk_distrib).sum(dim=-1)
    return pseudobulk_loss

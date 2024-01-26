"""Deconvolution benchmark utilities."""

import anndata as ad
import pandas as pd
import numpy as np
from typing import Optional, Union
import scvi
from scipy.special import softmax

from typing import Optional
import scipy.stats
from sklearn.linear_model import LinearRegression

import torch


def perform_nnls(signature: pd.DataFrame,
                 adata_pseudobulk: ad.AnnData) -> pd.DataFrame:
    """Perform deconvolution using the nnls method.
    It will be performed as many times as there are samples in averaged_data.

    Paramaters
    ----------
    signature: pd.DataFrame
        Signature matrix of shape (n_genes, n_cell_types)
    adata_pseudobulk: ad.AnnData
        AnnData object of shape (n_samples, n_genes) | relative counts

    Returns
    -------
    """
    deconv = LinearRegression(positive=True).fit(
        signature, adata_pseudobulk.layers["counts"].T
    )
    deconv_results = pd.DataFrame(
        deconv.coef_, index=adata_pseudobulk.obs_names, columns=signature.columns
    )
    deconv_results = deconv_results.div(
        deconv_results.sum(axis=1), axis=0
    )  # to sum up to 1
    return deconv_results


def perform_latent_deconv(adata_pseudobulk: ad.AnnData,
                          adata_latent_signature: ad.AnnData,
                          model: Optional[Union[scvi.model.SCVI,
                                                scvi.model.MixUpVI,
                                                scvi.model.CondSCVI]],
                          use_nnls: bool = False,
                          use_softmax: bool = False) -> pd.DataFrame:
    """Perform deconvolution in latent space using the nnls method.

    Parameters
    ----------
    adata_pseudobulk: ad.AnnData
        Pseudobulk AnnData object of shape (n_samples, n_genes) | counts
    adata_latent_signature: ad.AnnData
        Latent signature AnnData object of shape (n_genes, n_cell_types)
    model:
        Generative model to use for latent space deconvolution
    use_nnls: bool
        Whether to use nnls or not
    use_softmax: bool
        Whether to use softmax or not

    Returns
    -------
    deconv_results: pd.DataFrame
        Deconvolution results of shape (n_samples, n_cell_types)
    """
    # with torch.no_grad():
    adata_pseudobulk = ad.AnnData(X=adata_pseudobulk.layers["counts"],
                                  obs=adata_pseudobulk.obs,
                                  var=adata_pseudobulk.var)
    adata_pseudobulk.layers["counts"] = adata_pseudobulk.X.copy()

    latent_pseudobulk = model.get_latent_representation(adata_pseudobulk)

    if use_nnls:
        deconv = LinearRegression(positive=True).fit(adata_latent_signature.X.T,
                                                    latent_pseudobulk.T)
        deconv_results = pd.DataFrame(
            deconv.coef_,
            index=adata_pseudobulk.obs_names,
            columns=list(adata_latent_signature.obs["cell type"].values)
        )
        deconv_results = deconv_results.div(
            deconv_results.sum(axis=1), axis=0
        )  # to sum up to 1
    else:
        deconv = LinearRegression().fit(adata_latent_signature.X.T,
                                        latent_pseudobulk.T)
        if use_softmax:
            deconv_results = softmax(deconv.coef_, axis=1)
            deconv_results = pd.DataFrame(
                deconv_results,
                index=adata_pseudobulk.obs_names,
                columns=list(adata_latent_signature.obs["cell type"].values)
            )
        else:
            deconv_results = pd.DataFrame(
                np.abs(deconv.coef_),
                index=adata_pseudobulk.obs_names,
                columns=list(adata_latent_signature.obs["cell type"].values)
            )
            deconv_results = deconv_results.div(
                deconv_results.sum(axis=1), axis=0
            )  # to sum up to 1
    return deconv_results


def compute_correlations(deconv_results, ground_truth_fractions):
    """Compute n_sample pairwise correlations between the deconvolution results and the
    ground truth fractions of the n_groups (here n cell types).
    """
    deconv_results = deconv_results[
        ground_truth_fractions.columns
    ]  # to align order of columns
    correlations = [
        scipy.stats.pearsonr(
            ground_truth_fractions.iloc[i], deconv_results.iloc[i]
        ).statistic
        for i in range(len(deconv_results))
    ]
    correlations = pd.DataFrame({"correlations": correlations})
    return correlations


def compute_group_correlations(deconv_results, ground_truth_fractions):
    """Compute n_groups (here n cell types) pairwise correlations between the
    deconvolution results and ground truth fractions of the n_samples.
    """
    deconv_results = deconv_results[
        ground_truth_fractions.columns
    ]  # to align order of columns
    correlations = [
        scipy.stats.pearsonr(
            ground_truth_fractions.T.iloc[i], deconv_results.T.iloc[i]
        ).statistic
        for i in range(len(deconv_results.T))
    ]
    correlations = pd.DataFrame({"correlations": correlations})
    return correlations


def create_random_proportion(
    n_classes: int, n_non_zero: Optional[int] = None
) -> np.ndarray:
    """Create a random proportion vector of size n_classes."""
    if n_non_zero is None:
        n_non_zero = n_classes

    proportion_vector = np.zeros(
        n_classes,
    )

    proportion_vector[:n_non_zero] = np.random.rand(n_non_zero)

    proportion_vector = proportion_vector / proportion_vector.sum()
    return np.random.permutation(proportion_vector)

"""Utilities for pseudobulk dataset creation."""

import anndata as ad
import pandas as pd
import numpy as np
import random
from loguru import logger

from .run_benchmark_constants import (
    initialize_func,
    EVALUATION_PSEUDOBULK_SAMPLINGS
) 


def launch_evaluation_pseudobulk_samplings(
    evaluation_pseudobulk_sampling: list,
    all_data: dict,
    evaluation_dataset: str,
    granularity: str,
    n_cells_per_evaluation_pseudobulk: int,
    n_samples_evaluation_pseudobulk: int,
):
    """General function to create pseudobulks from different sampling methods.

    Parameters
    ----------
    evaluation_datasets: list
        The datasets to evaluate the deconvolution methods on.
    train_dataset: str | None
        The dataset to train some deconvolution methods on.
    n_variable_genes: int | None
        The number of most variable genes to keep.

    Return
    ------
    data: dict
        The train and evaluation datasets.
    """
    evaluation_pseudobulk_samplings_func, kwargs = initialize_func(
        EVALUATION_PSEUDOBULK_SAMPLINGS[evaluation_pseudobulk_sampling]
    )
    all_test_dset = all_data["datasets"][evaluation_dataset]
    test_dset = all_test_dset["dataset"][
        all_test_dset[granularity]["Test index"]
    ]
    kwargs["adata"] = test_dset
    if "cell_type_group" in kwargs:
        kwargs["adata"].obs = kwargs["adata"].obs.rename(
            {f"cell_types_grouped_{granularity}": "cell_types_grouped"},
            axis = 1
        )
    if "n_cells" in kwargs and "n_sample" in kwargs:
        kwargs["n_cells"] = n_cells_per_evaluation_pseudobulk
        kwargs["n_sample"] = n_samples_evaluation_pseudobulk
        message = (
            f"Creating pseudobulks composed of {n_samples_evaluation_pseudobulk}"
            f" samples with {n_cells_per_evaluation_pseudobulk} cells using the "
            f"{evaluation_pseudobulk_sampling} method..."
        )
    else:
        message = (
            f"Creating pseudobulks using the {evaluation_pseudobulk_sampling} method..."
        )
    logger.debug(message)

    pseudobulks = evaluation_pseudobulk_samplings_func(**kwargs)

    return pseudobulks


def create_anndata_pseudobulk(
    adata_obs: pd.DataFrame, adata_var_names: list, x: np.array
) -> ad.AnnData:
    """Creates an anndata object from a pseudobulk sample.

    Parameters
    ----------
    adata_obs: pd.DataFrame
        Obs dataframe from anndata object storing training set
    adata_var_names: list
        Gene names from the anndata object
    x: np.array
        pseudobulk sample

    Return
    ------
    ad.AnnData
        Anndata object storing the pseudobulk array
    """
    df_obs = pd.DataFrame.from_dict(
        [{col: adata_obs[col].value_counts().index[0] for col in adata_obs.columns}]
    )
    if len(x.shape) > 1 and x.shape[0] > 1:
        # several pseudobulks, so duplicate df_obs row
        df_obs = df_obs.loc[df_obs.index.repeat(x.shape[0])].reset_index(drop=True)
        df_obs.index = [f"sample_{idx}" for idx in df_obs.index]
    adata_pseudobulk = ad.AnnData(X=x, obs=df_obs)
    adata_pseudobulk.var_names = adata_var_names
    adata_pseudobulk.layers["counts"] = np.copy(x)
    adata_pseudobulk.raw = adata_pseudobulk

    return adata_pseudobulk


def create_purified_pseudobulk_dataset(
    adata: ad.AnnData,
    cell_type_group: str = "cell_types_grouped",
    aggregation_method : str = "mean",
):
    """Create pseudobulk dataset from single-cell RNA data, purified by cell types.
    There will thus be as many deconvolutions as there are cell types, each one of them
    only asked to infer that there is only one cell type in the pseudobulk it is trying
    to deconvolve. This task is thus supposed to be very easy.
    """
    logger.info("Creating purified pseudobulk dataset...")
    grouped = adata.obs.groupby(cell_type_group)
    averaged_data, group = {"relative_counts": [], "counts": []}, []
    for group_key, group_indices in grouped.groups.items():
        if aggregation_method == "mean":
            averaged_data["relative_counts"].append(adata[group_indices].layers["relative_counts"].mean(axis=0).tolist()[0])
            averaged_data["counts"].append(adata[group_indices].layers["counts"].mean(axis=0).tolist()[0])
        else:
            averaged_data["relative_counts"].append(adata[group_indices].layers["relative_counts"].sum(axis=0).tolist()[0])
            averaged_data["counts"].append(adata[group_indices].layers["counts"].sum(axis=0).tolist()[0])
        group.append(group_key)

    # pseudobulk dataset
    adata_pseudobulk_rc = create_anndata_pseudobulk(adata.obs, adata.var_names,
                                                    np.array(averaged_data["relative_counts"])
                                                    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(adata.obs, adata.var_names,
                                                    np.array(averaged_data["counts"])
                                                    )
    adata_pseudobulk_rc.obs_names = group
    adata_pseudobulk_counts.obs_names = group
    groundtruth_fractions = pd.DataFrame(
        np.eye(len(group)), index=group, columns=group
    )
    groundtruth_fractions.columns.name = cell_type_group

    pseudobulks = {
        "adata_pseudobulk_test_counts": adata_pseudobulk_counts,
        "adata_pseudobulk_test_rc": adata_pseudobulk_rc,
        "df_proportions_test": groundtruth_fractions,
    }

    return pseudobulks


def create_uniform_pseudobulk_dataset(
    adata: ad.AnnData,
    n_sample: int = 300,
    n_cells: int = 2000,
    cell_type_group: str = "cell_types_grouped",
    aggregation_method : str = "mean",
):
    """Create pseudobulk dataset from single-cell RNA data, randomly sampled.
    This deconvolution task is not too hard because the pseudo-bulk have the same cell
    fractions than the training dataset on which was created the signature matrix. Plus,
    when using a high n_cells (e.g. the default 2000) to create the pseudo-bulks, all
    n_sample pseudo-bulks will have the same cell fractions because of the high number
    of cells.
    """
    logger.info("Creating uniform pseudobulk dataset...")
    random.seed(random.randint(0, 1000))
    averaged_data, group = {"relative_counts": [], "counts": []}, []
    groundtruth_fractions = []
    for _ in range(n_sample):
        cell_sample = random.sample(list(adata.obs_names), n_cells)
        adata_sample = adata[cell_sample, :]
        groundtruth_frac = adata_sample.obs[cell_type_group].value_counts() / n_cells
        groundtruth_fractions.append(groundtruth_frac)
        if aggregation_method == "mean":
            averaged_data["relative_counts"].append(adata_sample.layers["relative_counts"].mean(axis=0).tolist()[0])
            averaged_data["counts"].append(adata_sample.layers["counts"].mean(axis=0).tolist()[0])
        else:
            averaged_data["relative_counts"].append(adata_sample.layers["relative_counts"].sum(axis=0).tolist()[0])
            averaged_data["counts"].append(adata_sample.layers["counts"].sum(axis=0).tolist()[0])

    # pseudobulk dataset
    adata_pseudobulk_rc = create_anndata_pseudobulk(adata.obs, adata.var_names,
                                                    np.array(averaged_data["relative_counts"])
                                                    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(adata.obs, adata.var_names,
                                                    np.array(averaged_data["counts"])
                                                    )

    # ground truth fractions
    groundtruth_fractions = pd.DataFrame(
        groundtruth_fractions,
        index=adata_pseudobulk_counts.obs_names,
        columns=groundtruth_fractions[0].index
    )
    groundtruth_fractions = groundtruth_fractions.fillna(
        0
    )  # the Nan are cells not sampled

    pseudobulks = {
        "adata_pseudobulk_test_counts": adata_pseudobulk_counts,
        "adata_pseudobulk_test_rc": adata_pseudobulk_rc,
        "df_proportions_test": groundtruth_fractions,
    }     
    return pseudobulks


def create_dirichlet_pseudobulk_dataset(
    adata: ad.AnnData,
    prior_alphas: np.array = None,
    n_sample: int = 300,
    cell_type_group: str = "cell_types_grouped",
    aggregation_method : str = "mean",
    n_cells : int = 1000,
    is_n_cells_random : bool = False,
    add_sparsity : bool = False,
):
    """Create pseudobulk dataset from single-cell RNA data, sampled from a dirichlet
    distribution. If a prior belief on the cell fractions (e.g. prior knowledge from
    specific tissue), then it can be incorporated. Otherwise, it will just be a non-
    informative prior. Then, compute dirichlet posteriors to sample cells - dirichlet is
    conjugate to the multinomial distribution, thus giving an easy posterior
    calculation.
    """
    # logger.info("Creating dirichlet pseudobulk dataset...")
    seed = random.randint(0, 1000)
    random_state = np.random.RandomState(seed=seed)
    cell_types = adata.obs[cell_type_group].value_counts()
    if prior_alphas is None:
        prior_alphas = np.ones(len(cell_types))  # non-informative prior
    likelihood_alphas = cell_types / adata.n_obs  # multinomial likelihood
    alpha_posterior = prior_alphas + likelihood_alphas
    posterior_dirichlet = random_state.dirichlet(alpha_posterior, n_sample)
    if is_n_cells_random:
        n_cells = np.random.randint(50, 1001, size=posterior_dirichlet.shape[0])
        posterior_dirichlet = np.round(np.multiply(posterior_dirichlet, n_cells))
    else:
        posterior_dirichlet = np.round(posterior_dirichlet * n_cells)
    posterior_dirichlet = posterior_dirichlet.astype(np.int64)  # number of cells to sample
    groundtruth_fractions = posterior_dirichlet / posterior_dirichlet.sum(
        axis=1, keepdims=True
    )

    random.seed(seed)
    averaged_data, group = {"relative_counts": [], "counts": []}, []
    all_adata_samples = []
    for i in range(n_sample):
        sample_data = []
        for j, cell_type in enumerate(likelihood_alphas.index):
            # If sample larger than cell population, sample with replacement
            if posterior_dirichlet[i][j] > cell_types[cell_type]:
                cell_sample = random.choices(
                    list(adata.obs.loc[adata.obs.cell_types_grouped == cell_type].index),
                    k=posterior_dirichlet[i][j],
                )
            else:
                cell_sample = random.sample(
                    list(adata.obs.loc[adata.obs.cell_types_grouped == cell_type].index),
                    posterior_dirichlet[i][j],
                )
            sample_data.extend(cell_sample)
        adata_sample = adata[sample_data]
        if aggregation_method == "mean":
            averaged_data["relative_counts"].append(adata_sample.layers["relative_counts"].mean(axis=0).tolist()[0])
            X = np.array(adata_sample.layers["counts"].mean(axis=0).tolist()[0])
            # TODO: For now, we remove the possibility to add sparsity, as all_adata_samples would not be affected
            # if add_sparsity:
            #     X = random_state.binomial(1, 0.2, X.shape[0]) * X
            averaged_data["counts"].append(X)
        # TODO: For now, we remove the possibility to aggregate by sum, as all_adata_samples would not be affected
        # else:
        #     averaged_data["relative_counts"].append(adata_sample.layers["relative_counts"].sum(axis=0).tolist()[0])
        #     X = np.array(adata_sample.layers["counts"].mean(axis=0).tolist()[0])
        #     if add_sparsity:
        #         X = random_state.binomial(1, 0.2, X.shape[0]) * X
        #     averaged_data["counts"].append(X)
        all_adata_samples.append(adata_sample)

    # pseudobulk dataset
    adata_pseudobulk_rc = create_anndata_pseudobulk(adata.obs, adata.var_names,
                                                    np.array(averaged_data["relative_counts"])
                                                    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(adata.obs, adata.var_names,
                                                    np.array(averaged_data["counts"])
                                                    )

    # ground truth fractions
    groundtruth_fractions = pd.DataFrame(
        groundtruth_fractions,
        index=adata_pseudobulk_counts.obs_names,
        columns=list(cell_types.index)
    )
    groundtruth_fractions = groundtruth_fractions.fillna(
        0
    )  # The Nan are cells not sampled

    pseudobulks = {
        "all_adata_samples_test": all_adata_samples,
        "adata_pseudobulk_test_counts": adata_pseudobulk_counts,
        "adata_pseudobulk_test_rc": adata_pseudobulk_rc,
        "df_proportions_test": groundtruth_fractions,
    }     
    return pseudobulks

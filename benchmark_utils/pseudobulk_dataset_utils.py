"""Utilities for pseudobulk dataset creation."""

import random

import anndata as ad
import numpy as np
import pandas as pd
from loguru import logger

from .run_benchmark_constants import EVALUATION_PSEUDOBULK_SAMPLINGS, initialize_func


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
    test_dset = all_test_dset["dataset"][all_test_dset[granularity]["Test index"]]
    kwargs["adata"] = test_dset
    if "cell_type_group" in kwargs:
        kwargs["adata"].obs = kwargs["adata"].obs.rename(
            {f"cell_types_grouped_{granularity}": "cell_types_grouped"}, axis=1
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

    # TODO: Add a check to see if the pseudobulks have to be created with mean or sum!!!!!
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
    aggregation_method: str = "mean",
):
    """Create pseudobulk dataset from single-cell RNA data, purified by cell types.

    There will thus be as many deconvolutions as there are cell types, each one of them
    only asked to infer that there is only one cell type in the pseudobulk it is trying
    to deconvolve. This task is supposed to be very easy.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to create the pseudobulk dataset from
    cell_type_group : str
        The cell type group to use for the pseudobulk dataset
    aggregation_method : str
        The aggregation method to use for the pseudobulk dataset (default "mean",
        can also be "sum")
    """
    logger.info("Creating purified pseudobulk dataset...")
    grouped = adata.obs.groupby(cell_type_group)
    averaged_data, group = {"relative_counts": [], "counts": []}, []
    for group_key, group_indices in grouped.groups.items():
        if aggregation_method == "mean":
            averaged_data["relative_counts"].append(
                adata[group_indices].layers["relative_counts"].mean(axis=0).tolist()[0]
            )
            averaged_data["counts"].append(
                adata[group_indices].layers["counts"].mean(axis=0).tolist()[0]
            )
        else:
            averaged_data["relative_counts"].append(
                adata[group_indices].layers["relative_counts"].sum(axis=0).tolist()[0]
            )
            averaged_data["counts"].append(
                adata[group_indices].layers["counts"].sum(axis=0).tolist()[0]
            )
        group.append(group_key)

    # pseudobulk dataset
    adata_pseudobulk_rc = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["relative_counts"])
    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["counts"])
    )
    adata_pseudobulk_rc.obs_names = group
    adata_pseudobulk_counts.obs_names = group
    groundtruth_fractions = pd.DataFrame(np.eye(len(group)), index=group, columns=group)
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
    aggregation_method: str = "mean",
):
    """Create pseudobulk dataset from single-cell RNA data, randomly sampled.

    This deconvolution task is not too hard because the pseudo-bulk have the same cell
    fractions than the training dataset on which was created the signature matrix. Plus,
    when using a high n_cells (e.g. the default 2000) to create the pseudo-bulks, all
    n_sample pseudo-bulks will have the same cell fractions because of the high number
    of cells.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to create the pseudobulk dataset from
    n_sample : int
        The number of pseudobulks to create
    n_cells : int
        The number of cells to sample for each pseudobulk
    cell_type_group : str
        The cell type group to use for the pseudobulk dataset
    aggregation_method : str
        The aggregation method to use for the pseudobulk dataset (default "mean",
        can also be "sum")
    """
    logger.info("Creating uniform pseudobulk dataset...")
    random.seed(random.randint(0, 1000))
    averaged_data = {"relative_counts": [], "counts": []}
    groundtruth_fractions = []
    for _ in range(n_sample):
        cell_sample = random.sample(list(adata.obs_names), n_cells)
        adata_sample = adata[cell_sample, :]
        groundtruth_frac = adata_sample.obs[cell_type_group].value_counts() / n_cells
        groundtruth_fractions.append(groundtruth_frac)
        if aggregation_method == "mean":
            averaged_data["relative_counts"].append(
                adata_sample.layers["relative_counts"].mean(axis=0).tolist()[0]
            )
            averaged_data["counts"].append(
                adata_sample.layers["counts"].mean(axis=0).tolist()[0]
            )
        else:
            averaged_data["relative_counts"].append(
                adata_sample.layers["relative_counts"].sum(axis=0).tolist()[0]
            )
            averaged_data["counts"].append(
                adata_sample.layers["counts"].sum(axis=0).tolist()[0]
            )

    # pseudobulk dataset
    adata_pseudobulk_rc = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["relative_counts"])
    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["counts"])
    )

    # ground truth fractions
    groundtruth_fractions = pd.DataFrame(
        groundtruth_fractions,
        index=adata_pseudobulk_counts.obs_names,
        columns=groundtruth_fractions[0].index,
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
    aggregation_method: str = "mean",
    n_cells: int = 1000,
    is_n_cells_random: bool = False,
    add_sparsity: bool = False,
):
    """Create pseudobulk dataset from single-cell RNA data, sampled from a dirichlet distribution.

    If a prior belief on the cell fractions (e.g. prior knowledge from
    specific tissue), then it can be incorporated. Otherwise, it will just be a non-
    informative prior. Then, compute dirichlet posteriors to sample cells - dirichlet is
    conjugate to the multinomial distribution, thus giving an easy posterior
    calculation.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to create the pseudobulk dataset from
    prior_alphas : np.array
        The prior alphas to use for the pseudobulk dataset
    n_sample : int
        The number of pseudobulks to create
    cell_type_group : str
        The cell type group to use for the pseudobulk dataset
    aggregation_method : str
        The aggregation method to use for the pseudobulk dataset (default "mean",
        can also be "sum")
    n_cells : int
        The number of cells to sample for each pseudobulk
    is_n_cells_random : bool
        Whether to sample the number of cells for each pseudobulk randomly
    add_sparsity : bool
        Whether to add sparsity to the pseudobulk dataset
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
    posterior_dirichlet = posterior_dirichlet.astype(
        np.int64
    )  # number of cells to sample
    groundtruth_fractions = posterior_dirichlet / posterior_dirichlet.sum(
        axis=1, keepdims=True
    )

    random.seed(seed)
    averaged_data = {"relative_counts": [], "counts": []}
    all_adata_samples = []
    for i in range(n_sample):
        sample_data = []
        for j, cell_type in enumerate(likelihood_alphas.index):
            # If sample larger than cell population, sample with replacement
            if posterior_dirichlet[i][j] > cell_types[cell_type]:
                cell_sample = random.choices(
                    list(adata.obs.loc[adata.obs[cell_type_group] == cell_type].index),
                    k=posterior_dirichlet[i][j],
                )
            else:
                cell_sample = random.sample(
                    list(adata.obs.loc[adata.obs[cell_type_group] == cell_type].index),
                    posterior_dirichlet[i][j],
                )
            sample_data.extend(cell_sample)
        adata_sample = adata[sample_data]
        if aggregation_method == "mean":
            averaged_data["relative_counts"].append(
                adata_sample.layers["relative_counts"].mean(axis=0).tolist()[0]
            )
            X = np.array(adata_sample.layers["counts"].mean(axis=0).tolist()[0])
            # TODO: For now, we remove the possibility to add sparsity, as all_adata_samples would not be affected
            # if add_sparsity:
            #     X = random_state.binomial(1, 0.2, X.shape[0]) * X
            averaged_data["counts"].append(X)
        # TODO: For now, we remove the possibility to aggregate by sum, as all_adata_samples would not be affected
        elif aggregation_method == "sum":
            averaged_data["relative_counts"].append(
                adata_sample.layers["relative_counts"].sum(axis=0).tolist()[0]
            )
            X = np.array(adata_sample.layers["counts"].sum(axis=0).tolist()[0])
            # if add_sparsity:
            #     X = random_state.binomial(1, 0.2, X.shape[0]) * X
            averaged_data["counts"].append(X)
        else:
            raise ValueError(f"Aggregation method {aggregation_method} not supported")
        all_adata_samples.append(adata_sample)

    # pseudobulk dataset
    adata_pseudobulk_rc = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["relative_counts"])
    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["counts"])
    )

    # ground truth fractions
    groundtruth_fractions = pd.DataFrame(
        groundtruth_fractions,
        index=adata_pseudobulk_counts.obs_names,
        columns=list(cell_types.index),
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


def create_purified_50_50_pseudobulk_dataset(
    adata,
    cell_type_1,
    cell_type_2,
    n_sample,
    n_cells_per_pseudobulk,
    cell_type_group="cell_types_grouped",
    aggregation_method="mean",
    random_state=None,
):
    """Creates multiple pseudobulk datasets by mixing two cell types in equal proportions.

    Args:
        adata: AnnData object containing single-cell data
        cell_type_1: First cell type to mix
        cell_type_2: Second cell type to mix
        n_sample: Number of pseudobulk samples to generate
        n_cells_per_pseudobulk: Total number of cells per pseudobulk
        cell_type_group: Column name in adata.obs containing cell type labels
        aggregation_method: Method to aggregate cells, either "mean" or "sum"
        random_state: Random state for reproducibility

    Returns
    -------
        Dictionary containing:
            - all_adata_samples_test: List of AnnData objects for each pseudobulk sample
            - adata_pseudobulk_test_counts: AnnData with raw counts
            - adata_pseudobulk_test_rc: AnnData with relative counts
            - df_proportions_test: DataFrame with ground truth proportions
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Get cells for each type
    cells_type1 = adata[adata.obs[cell_type_group] == cell_type_1].copy()
    cells_type2 = adata[adata.obs[cell_type_group] == cell_type_2].copy()

    # Check if we have enough cells
    cells_per_type = n_cells_per_pseudobulk // 2
    if cells_type1.n_obs < cells_per_type or cells_type2.n_obs < cells_per_type:
        raise ValueError(
            f"Not enough cells available. Need {cells_per_type} cells per type, "
            f"but have {cells_type1.n_obs} cells for type 1 and {cells_type2.n_obs} for type 2"
        )

    averaged_data = {"relative_counts": [], "counts": []}
    all_adata_samples = []

    for _ in range(n_sample):
        # Sample cells for each type
        sample1 = cells_type1[
            np.random.choice(cells_type1.n_obs, cells_per_type, replace=False)
        ]
        sample2 = cells_type2[
            np.random.choice(cells_type2.n_obs, cells_per_type, replace=False)
        ]

        # Combine samples
        adata_sample = adata.concatenate(sample1, sample2)
        all_adata_samples.append(adata_sample)

        if aggregation_method == "mean":
            averaged_data["relative_counts"].append(
                adata_sample.layers["relative_counts"].mean(axis=0).tolist()[0]
            )
            averaged_data["counts"].append(
                np.array(adata_sample.layers["counts"].mean(axis=0).tolist()[0])
            )
        elif aggregation_method == "sum":
            averaged_data["relative_counts"].append(
                adata_sample.layers["relative_counts"].sum(axis=0).tolist()[0]
            )
            averaged_data["counts"].append(
                np.array(adata_sample.layers["counts"].sum(axis=0).tolist()[0])
            )
        else:
            raise ValueError(f"Aggregation method {aggregation_method} not supported")

    # Create pseudobulk datasets
    adata_pseudobulk_rc = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["relative_counts"])
    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["counts"])
    )

    # Create ground truth proportions
    cell_types = pd.Series(adata.obs[cell_type_group].unique())
    groundtruth_fractions = pd.DataFrame(
        0,
        index=adata_pseudobulk_counts.obs_names,
        columns=list(cell_types),
    )
    groundtruth_fractions[cell_type_1] = 0.5
    groundtruth_fractions[cell_type_2] = 0.5

    pseudobulks = {
        "all_adata_samples_test": all_adata_samples,
        "adata_pseudobulk_test_counts": adata_pseudobulk_counts,
        "adata_pseudobulk_test_rc": adata_pseudobulk_rc,
        "df_proportions_test": groundtruth_fractions,
    }
    return pseudobulks

"""Utilities for creating and preprocessing single-cell RNA datasets for deconvolution benchmarking."""
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import random
from loguru import logger
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

from constants import GROUPS


def preprocess_scrna(
    adata: ad.AnnData, keep_genes: int = 2000, log: bool = False, batch_key: Optional[str] = None
):
    """Preprocess single-cell RNA data for deconvolution benchmarking."""
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy()  # preserve counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    if log:
        sc.pp.log1p(adata)
    adata.layers["relative_counts"] = adata.X.copy()  # preserve counts
    adata.raw = adata  # freeze the state in `.raw`
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=keep_genes,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key=batch_key,
    )
    #TODO: add the filtering / QC steps that they perform in Servier


def split_dataset(
    adata: ad.AnnData,
    grouping_choice: str = "updated_granular_groups",
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Split single-cell RNA data into train/test sets for deconvolution."""
    # create cell types
    groups = GROUPS[grouping_choice]
    group_correspondence = {}
    for k, v in groups.items():
        for cell_type in v:
            group_correspondence[cell_type] = k
    adata.obs["cell_types_grouped"] = [
        group_correspondence[cell_type] for cell_type in adata.obs.Manually_curated_celltype
    ]
    # remove some cell types: you need more than 15GB memory to run that
    index_to_keep = adata.obs.loc[adata.obs["cell_types_grouped"] != "To remove"].index
    adata = adata[index_to_keep]
    # build signature on train set and apply deconvo on the test set
    cell_types_train, cell_types_test = train_test_split(
        adata.obs_names,
        test_size=0.5,
        stratify=adata.obs["cell_types_grouped"],
        random_state=42,
    )
    return index_to_keep, cell_types_train, cell_types_test


def add_cell_types_grouped(
    adata: ad.AnnData, group: str = "primary_groups"
) -> ad.AnnData:
    """Add the cell types grouped columns in Anndata according to the grouping choice.
    It uses and returns the train_test_index csv file created for the signature matrix.
    """
    if group == "primary_groups":
        train_test_index = pd.read_csv("/home/owkin/project/train_test_index_matrix_common.csv", index_col=1).iloc[:,1:]
        col_name = "primary_groups"
    elif group == "precise_groups":
        train_test_index = pd.read_csv("/home/owkin/project/train_test_index_matrix_granular.csv", index_col=1).iloc[:,1:]
        col_name = "precise_groups"
    elif group == "updated_granular_groups":
        train_test_index = pd.read_csv("/home/owkin/project/train_test_index_matrix_granular_updated.csv", index_col=0)
        col_name = "precise_groups_updated"
    adata.obs["cell_types_grouped"] = train_test_index[col_name]
    return adata, train_test_index


def create_anndata_pseudobulk(adata: ad.AnnData, x: np.array) -> ad.AnnData:
    """Creates an anndata object from a pseudobulk sample.

    Parameters
    ----------
    adata: ad.AnnData
        AnnData aobject storing training set
    x: np.array
        pseudobulk sample

    Return
    ------
    ad.AnnData
        Anndata object storing the pseudobulk array
    """
    df_obs = pd.DataFrame.from_dict(
        [{col: adata.obs[col].value_counts().index[0] for col in adata.obs.columns}]
    )
    if len(x.shape) > 1 and x.shape[0] > 1:
        # several pseudobulks, so duplicate df_obs row
        df_obs = df_obs.loc[df_obs.index.repeat(x.shape[0])].reset_index(drop=True)
        df_obs.index = [f"sample_{idx}" for idx in df_obs.index]
    adata_pseudobulk = ad.AnnData(X=x, obs=df_obs)
    adata_pseudobulk.var_names = adata.var_names
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
    adata_pseudobulk_rc = create_anndata_pseudobulk(adata,
                                                    np.array(averaged_data["relative_counts"])
                                                    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(adata,
                                                    np.array(averaged_data["counts"])
                                                    )
    adata_pseudobulk_rc.obs_names = group
    adata_pseudobulk_counts.obs_names = group

    return adata_pseudobulk_counts, adata_pseudobulk_rc


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
    adata_pseudobulk_rc = create_anndata_pseudobulk(adata,
                                                    np.array(averaged_data["relative_counts"])
                                                    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(adata,
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

    return adata_pseudobulk_counts, adata_pseudobulk_rc, groundtruth_fractions


def create_dirichlet_pseudobulk_dataset(
    adata: ad.AnnData,
    prior_alphas: np.array = None,
    n_sample: int = 300,
    cell_type_group: str = "cell_types_grouped",
    aggregation_method : str = "mean",
):
    """Create pseudobulk dataset from single-cell RNA data, sampled from a dirichlet
    distribution. If a prior belief on the cell fractions (e.g. prior knowledge from
    specific tissue), then it can be incorporated. Otherwise, it will just be a non-
    informative prior. Then, compute dirichlet posteriors to sample cells - dirichlet is
    conjugate to the multinomial distribution, thus giving an easy posterior
    calculation.
    """
    logger.info("Creating dirichlet pseudobulk dataset...")
    seed = random.randint(0, 1000)
    random_state = np.random.RandomState(seed=seed)
    cell_types = adata.obs[cell_type_group].value_counts()
    if prior_alphas is None:
        prior_alphas = np.ones(len(cell_types))  # non-informative prior
    likelihood_alphas = cell_types / adata.n_obs  # multinomial likelihood
    alpha_posterior = prior_alphas + likelihood_alphas
    posterior_dirichlet = random_state.dirichlet(alpha_posterior, n_sample)
    posterior_dirichlet = np.round(posterior_dirichlet * 1000)
    posterior_dirichlet = posterior_dirichlet.astype(np.int64)  # number of cells to sample
    groundtruth_fractions = posterior_dirichlet / posterior_dirichlet.sum(
        axis=1, keepdims=True
    )

    random.seed(seed)
    averaged_data, group = {"relative_counts": [], "counts": []}, []
    for i in range(n_sample):
        sample_data = []
        for j, cell_type in enumerate(likelihood_alphas.index):
            cell_sample = random.sample(
                list(adata.obs.loc[adata.obs.cell_types_grouped == cell_type].index),
                posterior_dirichlet[i][j],
            )
            sample_data.extend(cell_sample)
        adata_sample = adata[sample_data]
        if aggregation_method == "mean":
            averaged_data["relative_counts"].append(adata_sample.layers["relative_counts"].mean(axis=0).tolist()[0])
            averaged_data["counts"].append(adata_sample.layers["counts"].mean(axis=0).tolist()[0])
        else:
            averaged_data["relative_counts"].append(adata_sample.layers["relative_counts"].sum(axis=0).tolist()[0])
            averaged_data["counts"].append(adata_sample.layers["counts"].sum(axis=0).tolist()[0])

    # pseudobulk dataset
    adata_pseudobulk_rc = create_anndata_pseudobulk(adata,
                                                    np.array(averaged_data["relative_counts"])
                                                    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(adata,
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
    )  # the Nan are cells not sampled

    return adata_pseudobulk_counts, adata_pseudobulk_rc, groundtruth_fractions

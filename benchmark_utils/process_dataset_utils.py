"""Utilities for creating and preprocessing single-cell RNA datasets for deconvolution benchmarking."""

import scanpy as sc
import anndata as ad
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

from constants import GROUPS


def preprocess_scrna(
    adata: ad.AnnData, keep_genes: int = 2000, batch_key: Optional[str] = None
):
    """Preprocess single-cell RNA data for deconvolution benchmarking.

    * in adata.X, the normalized log1p counts are saved
    * in adata.layers["counts"], raw counts are saved
    * in adata.layers["relative_counts"], the relative counts are saved
    => The highly variable genes can be found in adata.var["highly_variable"]

    """
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy()  # preserve counts, used for training
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.layers["relative_counts"] = adata.X.copy()  # preserve counts, used for baselines
    sc.pp.log1p(adata)
    adata.raw = adata  # freeze the state in `.raw`
    if keep_genes is not None:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=keep_genes,
            layer="counts",
            flavor="seurat_v3",
            batch_key=batch_key,
            subset=False,
            inplace=True,
        )
    #TODO: add the filtering / QC steps that they perform in Servier

    return adata


def split_dataset(
    adata: ad.AnnData,
    grouping_choice: str = "2nd_level_granularity",
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


def create_new_granularity_index(grouping_choice = "2nd_level_granularity"):
    """Create new granularity index file used to create signature matrices."""
    adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
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
    adata2 = adata.copy()
    adata2 = adata2[index_to_keep]
    # build signature on train set and apply deconvo on the test set
    cell_types_train, cell_types_test = train_test_split(
        adata2.obs_names,
        test_size=0.5,
        stratify=adata2.obs["cell_types_grouped"],
        random_state=42,
    )
    index_df = pd.DataFrame(adata.obs.index, columns=["index"])
    index_df["to keep"] = [True if index in adata2.obs.index else False for index in index_df["index"]]
    index_df["Train index"] = [True if index in cell_types_train else False for index in index_df["index"]]
    index_df["Test index"] = [True if index in cell_types_test else False for index in index_df["index"]]
    index_df["grouping"] = adata.obs["cell_types_grouped"].values
    return index_df


def add_cell_types_grouped(
    adata: ad.AnnData, group: str = "1st_level_granularity"
) -> ad.AnnData:
    """Add the cell types grouped columns in Anndata according to the grouping choice.
    It uses and returns the train_test_index csv file created for the signature matrix.
    """
    if group == "1st_level_granularity":
        train_test_index = pd.read_csv("/home/owkin/project/train_test_index_dataframes/train_test_index_matrix_common.csv", index_col=1).iloc[:,1:]
        col_name = "primary_groups"
    elif group == "2nd_level_granularity":
        train_test_index = pd.read_csv("/home/owkin/project/train_test_index_dataframes/train_test_index_matrix_granular_updated.csv", index_col=0)
        col_name = "precise_groups_updated"
    elif group == "3rd_level_granularity":
        train_test_index = pd.read_csv("/home/owkin/project/train_test_index_dataframes/train_test_index_3rd_level.csv", index_col=1).iloc[:,1:]
        col_name = "grouping"
    elif group == "4th_level_granularity":
        train_test_index = pd.read_csv("/home/owkin/project/train_test_index_dataframes/train_test_index_4th_level.csv", index_col=1).iloc[:,1:]
        col_name = "grouping"
    elif group == "FACS_1st_level_granularity":
        train_test_index = pd.read_csv("/home/owkin/project/train_test_index_dataframes/train_test_index_facs_1st_level.csv", index_col=1).iloc[:,1:]
        col_name = "grouping"
    adata.obs[f"cell_types_grouped_{group}"] = train_test_index[col_name]
    return adata, train_test_index

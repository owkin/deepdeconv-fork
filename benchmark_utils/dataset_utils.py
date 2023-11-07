"""Utilities for creating and preprocessing single-cell RNA datasets for deconvolution benchmarking."""
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


def preprocess_scrna(
    adata: ad.AnnData, keep_genes: int = 2000, batch_key: Optional[str] = None
):
    """Preprocess single-cell RNA data for deconvolution benchmarking."""
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy()  # preserve counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # freeze the state in `.raw`
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=keep_genes,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key=batch_key,
    )


def split_dataset(
    adata: ad.AnnData,
    stratify: Optional[str] = "cell_types_grouped",
    random_state: int = 42,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Split single-cell RNA data into train/test sets for deconvolution."""
    cell_types_train, cell_types_test = train_test_split(
        adata.obs_names,
        test_size=0.5,
        stratify=adata.obs[stratify],
        random_state=random_state,
    )

    adata_train = adata[cell_types_train, :]
    adata_test = adata[cell_types_test, :]

    return adata_train, adata_test


def create_pseudobulk_dataset(
    adata: ad.AnnData,
    n_sample: int = 300,
    cell_type_group: str = "cell_types_grouped",
    aggregation_method : str = "mean",
):
    """Create pseudobulk dataset from single-cell RNA data."""
    random.seed(random.randint(0, 1000))
    averaged_data = []
    ground_truth_fractions = []
    n_cells = 2000
    for i in range(n_sample):
        cell_sample = random.sample(list(adata.obs_names), n_cells)
        adata_sample = adata[cell_sample, :]
        ground_truth_frac = adata_sample.obs[cell_type_group].value_counts() / n_cells
        ground_truth_fractions.append(ground_truth_frac)
        if aggregation_method == "mean":
            averaged_data.append(adata_sample.raw.X.mean(axis=0).tolist()[0])
        else:
            averaged_data.append(adata_sample.raw.X.sum(axis=0).tolist()[0])


    averaged_data = pd.DataFrame(
        averaged_data, index=list(range(n_sample)), columns=adata_sample.raw.var_names
    )
    # pseudobulk dataset
    adata_pseudobulk = ad.AnnData(X=averaged_data.values)
    adata_pseudobulk.obs_names = [f"sample_{idx}" for idx in list(averaged_data.index)]
    adata_pseudobulk.var_names = list(averaged_data.columns)
    adata_pseudobulk.layers["counts"] = adata_pseudobulk.X.copy()
    # adata_pseudobulk.obsm["spatial"] = adata_pseudobulk.obsm["location"]
    sc.pp.normalize_total(adata_pseudobulk, target_sum=1e4)
    adata_pseudobulk.layers["relative_counts"] = adata_pseudobulk.X.copy()
    sc.pp.log1p(adata_pseudobulk)
    adata_pseudobulk.raw = adata_pseudobulk
    # filter genes to be the same on the pseudobulk data
    intersect = np.intersect1d(adata_pseudobulk.var_names, adata.var_names)
    adata_pseudobulk = adata_pseudobulk[:, intersect].copy()
    adata_pseudobulk.obs[cell_type_group] = "B"
    G = len(intersect)
    return adata_pseudobulk, ground_truth_fractions

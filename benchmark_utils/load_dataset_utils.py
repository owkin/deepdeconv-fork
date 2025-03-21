"""Utilities to load single cell and bulk/facs datasets."""

from __future__ import annotations

import pandas as pd
import scanpy as sc
from loguru import logger

from .process_dataset_utils import preprocess_scrna
from .run_benchmark_constants import (
    initialize_func,
    DATASETS,
)


def load_preprocessed_datasets(
    evaluation_datasets: list,
    train_dataset: str | None = None,
    n_variable_genes: int | None = None,
) -> dict:
    """General function to load and preprocess train and evaluation datasets.

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
    data = {"datasets": {}}
    for evaluation_dataset in evaluation_datasets:
        logger.info(f"Loading dataset: {evaluation_dataset}...")
        data["datasets"][evaluation_dataset] = {}
        dataset_config = DATASETS[evaluation_dataset]
        initialized_func, kwargs = initialize_func(dataset_config)
        kwargs["n_variable_genes"] = n_variable_genes
        data["datasets"][evaluation_dataset] = initialized_func(**kwargs)

    if train_dataset is not None and train_dataset not in evaluation_datasets:
        logger.info(f"Loading train dataset: {train_dataset}...")
        data["datasets"][train_dataset] = {}
        dataset_config = DATASETS[train_dataset]
        initialized_func, kwargs = initialize_func(dataset_config)
        kwargs["n_variable_genes"] = n_variable_genes
        data["datasets"][train_dataset] = initialized_func(**kwargs)
    
    # In case one of the evaluation dataset is BULK_FACS, intersect the bulk and CTI gene lists
    if "BULK_FACS" in evaluation_datasets:
        logger.warning(
            "BULK_FACS is provided as one evaluation_dataset, therefore CTI will be intersected "
            "with the common genes between both datasets. To prevent this intersection for "
            "pseudobulk evaluations, remove BULK_FACS as evaluation_dataset."
        )
        # TODO (related to the warning above): The code prevents CTI pseudobulk evaluation to train on non-intersected genes, 
        # so the intersection should be done at training time
        # TODO: It should be done in an automatized way no matter the training single cell dataset (not CTI hard-coded like here)
        bulk_gene_list = data["datasets"]["BULK_FACS"]["dataset"].index
        cti_gene_list = data["datasets"]["CTI"]["dataset"].var_names
        cti_bulk_genes_intersection = list(set(bulk_gene_list).intersection(set(cti_gene_list)))
        data["datasets"]["CTI"]["dataset"] = data["datasets"]["CTI"]["dataset"][:, cti_bulk_genes_intersection]
        data["datasets"]["BULK_FACS"]["dataset"] = data["datasets"]["BULK_FACS"]["dataset"].loc[cti_bulk_genes_intersection]

    return data


def load_cti(n_variable_genes: int, **kwargs) -> dict:
    """Load and preprocess the CTI scRNAseq dataset.

    Parameters
    ----------
    n_variable_genes: int | None
        The number of most variable genes to keep.

    Return
    ------
    data: dict
        The preprocessed CTI dataset.
    """
    adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
    adata = preprocess_scrna(adata,
                     keep_genes=n_variable_genes,
                     batch_key="donor_id")
    data = {"dataset": adata}
    return data


def load_bulk_facs(**kwargs) -> dict:
    """Load and preprocess the Bulk/FACS dataset.

    Return
    ------
    data: dict
        The preprocessed Bulk/FACS dataset.
    """
    # Load data
    facs_results = pd.read_csv(
        "/home/owkin/project/bulk_facs/240214_majorCelltypes.csv", index_col=0
    ).drop(["No.B.Cells.in.Live.Cells","NKT.Cells.in.Live.Cells"],axis=1).set_index(
        "Sample"
    )
    facs_results = facs_results.rename(
        {
            "B.Cells.in.Live.Cells":"B",
            "NK.Cells.in.Live.Cells":"NK",
            "T.Cells.in.Live.Cells":"T",
            "Monocytes.in.Live.Cells":"Mono",
            "Dendritic.Cells.in.Live.Cells":"DC",
        }, axis=1
    )
    facs_results = facs_results.dropna()
    bulk_data = pd.read_csv(
        (
        "/home/owkin/project/bulk_facs/"
        "gene_counts_batchs1-5_raw.csv"
        # "gene_counts20230103_batch1-5_all_cleaned-TPMnorm-allpatients.tsv"
        ), 
        index_col=0,
        # sep="\t"
    ).T
    common_samples = pd.read_csv(
        "/home/owkin/project/bulk_facs/RNA-FACS_common-samples.csv", index_col=0
    )
    # Align bulk and facs samples
    common_facs = common_samples.set_index("FACS.ID")["Patient"]
    facs_results = facs_results.loc[facs_results.index.isin(common_facs.keys())]
    facs_results = facs_results.rename(index=common_facs)
    common_bulk = common_samples.set_index("RNAseq_ID")["Patient"]
    bulk_data = bulk_data.loc[bulk_data.index.isin(common_bulk.keys())]
    bulk_data = bulk_data.rename(index=common_bulk)
    bulk_data = bulk_data.loc[facs_results.index]
    bulk_data.index = bulk_data.index.astype(str)
    facs_results.index = facs_results.index.astype(str)

    data = {"dataset": bulk_data.T, "ground_truth": facs_results}

    return data
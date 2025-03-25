"""Deconvolution benchmark utilities."""

from __future__ import annotations

import numpy as np
import os
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression
from typing import Optional

from .pseudobulk_dataset_utils import launch_evaluation_pseudobulk_samplings
from .run_benchmark_constants import (
    initialize_func,
    DECONV_METHODS,
    MODEL_TO_FIT,
    SIGNATURE_MATRIX_MODELS,
    SIGNATURE_TO_GRANULARITY,
)


def initialize_deconv_methods(
    deconv_methods: list,
    all_data: dict,
    granularity: str,
    train_dataset: str,
    signature_matrices: list,
):
    """General function to initialize and fit deconvolution methods in the benchmark.

    Parameters
    ----------
    deconv_methods: list
        The list of deconvolution methods to initialize and, if appropriate, fit.
    all_data: dict
        The data dictionary contraining the training dataset and signature matrices.
    granularity: str
        The cell type granularity for deconvolution to use.
    train_dataset: str
        The dataset to train some deconvolution methods on.
    signature_matrices: list
        The signature matrices to use for some of the methods.

    Return
    ------
    deconv_methods_initialized: dict
        The initialized and fitted deconvolution methods.
    """
    deconv_methods_initialized = {}
    for deconv_method in deconv_methods:
        deconv_method_func, kwargs = initialize_func(DECONV_METHODS[deconv_method])
        if (deconv_method in MODEL_TO_FIT)==(deconv_method in SIGNATURE_MATRIX_MODELS):
            message = (
                "The codebase is not formatted yet to have a deconvolution method "
                "needing both to be fit and a user-provided signature matrix, or none "
                "of these two options. It needs one of these options only."
            )
            logger.error(message)
            raise NotImplementedError(message)
        if deconv_method in MODEL_TO_FIT:
            logger.debug(f"Training deconvolution method {deconv_method}...")
            all_train_dset = all_data["datasets"][train_dataset]
            train_dset = all_train_dset["dataset"][
                all_train_dset[granularity]["Train index"]
            ]
            kwargs["adata_train"] = train_dset
            kwargs["adata_train"].obs = kwargs["adata_train"].obs.rename(
                {f"cell_types_grouped_{granularity}": "cell_types_grouped"},
                axis = 1
            )
            if "adata_pseudobulk" in kwargs:
                train_pseudobulks = launch_evaluation_pseudobulk_samplings(
                    evaluation_pseudobulk_sampling="DIRICHLET", # "UNIFORM"
                    all_data=all_data,
                    evaluation_dataset=train_dataset,
                    granularity=granularity,
                    n_cells_per_evaluation_pseudobulk=100,
                    n_samples_evaluation_pseudobulk=500,
                )
                kwargs["adata_pseudobulk"] = train_pseudobulks["adata_pseudobulk_test_counts"]
            deconv_method_initialized = deconv_method_func(**kwargs)
            deconv_methods_initialized[deconv_method] = deconv_method_initialized
        elif deconv_method in SIGNATURE_MATRIX_MODELS:
            for signature_matrix in signature_matrices:
                if SIGNATURE_TO_GRANULARITY[signature_matrix]==granularity:
                    kwargs["signature_matrix_name"] = signature_matrix
                    kwargs["signature_matrix"] = all_data["signature_matrices"][
                        signature_matrix
                    ]
                    deconv_method_initialized = deconv_method_func(**kwargs)
                    deconv_methods_initialized[
                        f"{deconv_method}_{signature_matrix}"
                    ] = deconv_method_initialized
    
    logger.debug("Initialization of the deconvolution methods complete.")
    return deconv_methods_initialized


def use_nnls_method(to_deconvolve: pd.DataFrame, signature_matrix: pd.DataFrame):
    """Apply NNLS on data to deconvolve.
    """
    # Gene intersection with signature matrix
    gene_intersection = signature_matrix.index.intersection(to_deconvolve.index)
    to_deconvolve = to_deconvolve.loc[gene_intersection]
    signature_matrix = signature_matrix.loc[gene_intersection]
    # Run NNLS
    deconv = LinearRegression(positive=True).fit(
        signature_matrix, to_deconvolve
    )
    deconv_results = pd.DataFrame(
        deconv.coef_, index=to_deconvolve.columns, columns=signature_matrix.columns
    )
    deconv_results = deconv_results.div(
        deconv_results.sum(axis=1), axis=0
    )  # to sum up to 1
    
    return deconv_results


def save_deconvolution_results(deconv_results: dict, experiment_path):
    """
    """
    for granularity, sub_dict in deconv_results.items():
        granularity_dir = os.path.join(experiment_path, str(granularity))
        os.makedirs(granularity_dir, exist_ok=True)
        
        for key, value in sub_dict.items():
            if isinstance(value, dict):
                save_deconvolution_results(value, os.path.join(granularity_dir, key))
            else:                
                value.to_csv(os.path.join(granularity_dir, f"{key}.csv"))
    

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
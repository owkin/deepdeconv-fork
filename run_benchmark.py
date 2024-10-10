"""Pseudobulk benchmark."""

import argparse
import os
import scanpy as sc
import pandas as pd
import anndata as ad
import warnings
from loguru import logger
from sklearn.linear_model import LinearRegression
from typing import Optional

from benchmark_utils import add_cell_types_grouped, create_signature
from run_benchmark_help import initialise_deconv_methods, load_preprocessed_datasets
from run_benchmark_config_dataclass import (
    RunBenchmarkConfig,
    GRANULARITY_TO_DATASET,
    SINGLE_CELL_DATASETS,
) 

def run_benchmark(
    deconv_methods: list,
    evaluation_datasets: list,
    granularities: list,
    evaluation_pseudobulk_samplings: Optional[list],
    signature_matrices: Optional[list],
    train_dataset: Optional[str],
    n_variable_genes: Optional[int],
    save: bool,
    experiment_name: Optional[str],
):
    """Run the deconvolution benchmark pipeline.

    The arguments are defined in a config yaml file passed to the RunBenchmarkConfig
    dataclass.
    """
    all_data = load_preprocessed_datasets(
        evaluation_datasets,
        train_dataset,
        n_variable_genes,
    )

    if signature_matrices is not None:
        all_data["signature_matrices"] = {}
        for signature_matrix in signature_matrices:
            logger.info(f"Loading signature matrix: {signature_matrix}...")
            all_data["signature_matrices"][signature_matrix] = create_signature(
                signature_matrix
            )

    # Will there be a problem for differentiation of FACS vs SC ? 
    for granularity in granularities:
            logger.info(
                f"Loading train/test index for granularity: {granularity}..."
            )
            for dataset in all_data["datasets"]:
                    if GRANULARITY_TO_DATASET[granularity] == dataset:
                        all_data["datasets"][dataset]["dataset"], train_test_index = \
                            add_cell_types_grouped(
                                all_data["datasets"][dataset]["dataset"], 
                                granularity
                            )
                        all_data["datasets"][dataset][granularity] = train_test_index


    logger.info(
        "All the data are now loaded."
    )

    for granularity in granularities:
        logger.info(
            f"Launching the deconvolution experiments for granularity: {granularity}..."
        )
        deconv_methods_initialized = initialise_deconv_methods(
            deconv_methods=deconv_methods,
            all_data=all_data,
            granularity=granularity,
            train_dataset=train_dataset,
            signature_matrices=signature_matrices,
        )
        evaluation_dataset = GRANULARITY_TO_DATASET[granularity]
        logger.info(f"Running evaluation on {evaluation_dataset}...")
        if evaluation_dataset in SINGLE_CELL_DATASETS:
            for evaluation_pseudobulk_sampling in evaluation_pseudobulk_samplings:
                # TODO: TAKE from here, we also need the N_CELLS argument and to reformat the sanity checks to make them more readable (which was the whole point of this PR at first)
                logger.info(
                    f"Creating pseudobulks with {evaluation_pseudobulk_sampling} "
                    "method..."
                )
                for deconv_method_initialized in deconv_methods_initialized:
                    pass
        else:
            # direct pred
            pass




    # for deconv_method_name in deconv_methods:
    #         logger.info(f"Running deconvolution method: {deconv_method_name}")
    #         if not os.path.exists(f"{experiment_path}/{deconv_method_name}"):
    #             os.mkdir(f"{experiment_path}/{deconv_method_name}")
    #         # Instantiate deconvolution method
    #         method = deconv_methods[deconv_method_name]
    #         if method in FIT_SINGLE_CELL:
    #             # Fit deconvolution method
    #             method.fit(adata_train[:,filtered_genes].copy())
    #         if method in CREATE_LATENT_SIGNATURE:
    #             # Create the signature matrix inside the latent space
    #             method.create_latent_signature(adata_train[:,filtered_genes])

    #         for sanity_check_name in sanity_checks:
    #             sanity_check_type = sanity_check_name.split("_")[3]
    #             logger.info(f"Run following sanity check: {sanity_check_name}")
    #             if not os.path.exists(f"{experiment_path}/{deconv_method_name}/{sanity_check_type}"):
    #                 os.mkdir(f"{experiment_path}/{deconv_method_name}/{sanity_check_type}")

    #             # Create the pseudobulk (or bulk) test and true simulated (or facs) proportions
    #             pseudobulk_test, df_proportions_test = sanity_check.create_test_data(adata_test.copy())

    #             # Perform test deconvolution
    #             deconv_results = method(pseudobulk_test, df_proportions_test)
    #             saving_path = f"{experiment_path}/{deconv_method_name}/{sanity_check_type}/{sanity_check_name}.csv"
    #             deconv_results.to_csv(saving_path)
    #             logger.info(f"The deconvolution results are saved inside the following directory: {saving_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    config_dict = RunBenchmarkConfig.from_config_yaml(config_path=args.config)

    run_benchmark(**config_dict)
    # constants should be used for the methods HPs for now, but saved + logged as full configs
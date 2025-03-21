"""Run pseudobulk benchmark."""

import argparse
import pandas as pd
from loguru import logger
from typing import Optional

from benchmark_utils import (
    add_cell_types_grouped,
    compute_benchmark_correlations,
    create_signature,
    initialize_deconv_methods,
    launch_evaluation_pseudobulk_samplings,
    load_preprocessed_datasets,
    plot_benchmark_correlations,
    save_deconvolution_results,
    DECONV_METHOD_TO_EVALUATION_PSEUDOBULK,
    GRANULARITY_TO_TRAINING_DATASET,
    GRANULARITY_TO_EVALUATION_DATASET,
    SINGLE_CELL_DATASETS,
)
from run_benchmark_config_dataclass import RunBenchmarkConfig

def run_benchmark(
    deconv_methods: list,
    evaluation_datasets: list,
    granularities: list,
    evaluation_pseudobulk_samplings: Optional[list],
    n_samples_evaluation_pseudobulk: int,
    n_cells_per_evaluation_pseudobulk: Optional[list],
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
    # Loading datasets
    all_data = load_preprocessed_datasets(
        evaluation_datasets=evaluation_datasets,
        train_dataset=train_dataset,
        n_variable_genes=n_variable_genes,
    )

    # Loading signature matrices
    if signature_matrices is not None:
        all_data["signature_matrices"] = {}
        for signature_matrix in signature_matrices:
            logger.info(f"Loading signature matrix: {signature_matrix}...")
            all_data["signature_matrices"][signature_matrix] = create_signature(
                signature_matrix
            )

    # Loading train/test indexes and cell type groupings
    for granularity in granularities:
            logger.info(
                f"Loading train/test index for granularity: {granularity}..."
            )
            for dataset in all_data["datasets"]:
                    if GRANULARITY_TO_TRAINING_DATASET[granularity] == dataset:
                        all_data["datasets"][dataset]["dataset"], train_test_index = \
                            add_cell_types_grouped(
                                all_data["datasets"][dataset]["dataset"], 
                                granularity
                            )
                        all_data["datasets"][dataset][granularity] = train_test_index


    logger.info("All the data is now loaded.")

    # Deconvolution training and inference
    all_data["deconv_results"] = {}
    for granularity in granularities:
        logger.info(
            f"Launching the deconvolution experiments for granularity: {granularity}..."
        )
        all_data["deconv_results"][granularity] = {}
        deconv_methods_initialized = initialize_deconv_methods(
            deconv_methods=deconv_methods,
            all_data=all_data,
            granularity=granularity,
            train_dataset=train_dataset,
            signature_matrices=signature_matrices,
        )
        evaluation_dataset = GRANULARITY_TO_EVALUATION_DATASET[granularity]
        logger.debug(f"Running evaluation on {evaluation_dataset}...")
        if evaluation_dataset in SINGLE_CELL_DATASETS:
            # Inference on scRNAseq-derived pseudobulks
            for evaluation_pseudobulk_sampling in evaluation_pseudobulk_samplings:
                all_data["deconv_results"][granularity][evaluation_pseudobulk_sampling] = {}
                for n_cells in n_cells_per_evaluation_pseudobulk:
                    all_data["deconv_results"][granularity][evaluation_pseudobulk_sampling][n_cells] = {}
                    evaluation_pseudobulks = launch_evaluation_pseudobulk_samplings(
                        evaluation_pseudobulk_sampling=evaluation_pseudobulk_sampling,
                        all_data=all_data,
                        evaluation_dataset=evaluation_dataset,
                        granularity=granularity,
                        n_cells_per_evaluation_pseudobulk=n_cells,
                        n_samples_evaluation_pseudobulk=n_samples_evaluation_pseudobulk,
                    )
                    for deconv_method_initialized_key, deconv_method_initialized in deconv_methods_initialized.items():
                        if hasattr(deconv_method_initialized, "signature_matrix_name"):
                            signature_matrix_name = deconv_method_initialized.signature_matrix_name
                            deconv_method_key = deconv_method_initialized_key.split(f"_{signature_matrix_name}")[0]
                            var_to_deconvolve = DECONV_METHOD_TO_EVALUATION_PSEUDOBULK[deconv_method_key]
                        else:
                            var_to_deconvolve = DECONV_METHOD_TO_EVALUATION_PSEUDOBULK[deconv_method_initialized_key]
                        deconv_results = deconv_method_initialized.apply_deconvolution(to_deconvolve=evaluation_pseudobulks[var_to_deconvolve])
                        all_data["deconv_results"][granularity][evaluation_pseudobulk_sampling][n_cells][deconv_method_initialized_key] = {}
                        all_data["deconv_results"][granularity][evaluation_pseudobulk_sampling][n_cells][deconv_method_initialized_key]["deconvolution_results"] = deconv_results
                        all_data["deconv_results"][granularity][evaluation_pseudobulk_sampling][n_cells][deconv_method_initialized_key]["ground_truth"] = evaluation_pseudobulks["df_proportions_test"]
        else:
            # Direct inference on Bulk data
            # TODO: should we allow inference on scRNAseq-derived pseudobulks, from bulk granularities ?
            for deconv_method_initialized_key, deconv_method_initialized in deconv_methods_initialized.items():
                deconv_results = deconv_method_initialized.apply_deconvolution(to_deconvolve=all_data["datasets"][evaluation_dataset]["dataset"])
                all_data["deconv_results"][granularity][deconv_method_initialized_key] = {}
                all_data["deconv_results"][granularity][deconv_method_initialized_key]["ground_truth"] = all_data["datasets"][evaluation_dataset]["ground_truth"]
                all_data["deconv_results"][granularity][deconv_method_initialized_key]["deconvolution_results"] = deconv_results

    logger.info("Deconvolution inference is now complete.")

    if save:
        save_deconvolution_results(all_data["deconv_results"], experiment_path=experiment_name)
        logger.debug("Saved deconvolution results and ground truths.")

    # Compute basic correlations
    logger.info("Computing correlations.")
    df_samples_correlation = compute_benchmark_correlations(all_data["deconv_results"], correlation_type="sample_wise_correlation")
    df_cell_type_correlation = compute_benchmark_correlations(all_data["deconv_results"], correlation_type="cell_type_wise_correlation")
    df_all_correlations = pd.concat([df_samples_correlation, df_cell_type_correlation], ignore_index=True)
    if save:
        df_all_correlations.to_csv(experiment_name + "/df_all_correlations.csv")
        logger.debug(f"Saved correlation results in {experiment_name}/df_all_correlations.csv")
    logger.info("Correlations computed.")

    # Basic plotting
    plot_benchmark_correlations(df_all_correlations, save_path=experiment_name)
    logger.debug(f"Saved plots.")

    open(f"{experiment_name}/experiment_over.txt", "w").close() # Finish experiment
    logger.info("Experiment over.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    config_dict = RunBenchmarkConfig.from_config_yaml(config_path=args.config)

    run_benchmark(**config_dict)
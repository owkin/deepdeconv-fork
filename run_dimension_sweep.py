"""Run MixUpVI dimension sweep experiments to analyze performance vs input dimension."""

import os

import scanpy as sc
from loguru import logger

import ray
from benchmark_utils import add_cell_types_grouped, preprocess_scrna, tune_mixupvi
from constants import (
    TRAINING_CELL_TYPE_GROUP,
    TRAINING_DATASET,
)
from tuning_configs import (
    ADDITIONAL_METRICS,
    METRIC,
    NUM_SAMPLES,
    SEARCH_SPACE,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Configuration for the sweep
DIMENSION_RANGE = [4000, 10000]  # 1k to 10k in steps of 1k


def run_single_experiment(adata, ranked_genes, n_genes, cell_type):
    """Run a single experiment with given dimension and seed."""
    # Subset to top n_genes from ranked genes
    genes_subset = ranked_genes[:n_genes]
    adata_subset = adata[:, genes_subset].copy()

    # Fit model
    _ = tune_mixupvi(
        adata_subset,
        cell_type_group=cell_type,
        search_space=SEARCH_SPACE,
        metric=METRIC,
        additional_metrics=ADDITIONAL_METRICS,
        num_samples=NUM_SAMPLES,
        training_dataset=TRAINING_DATASET,
        experiment_name=f"tune_mixupvi_dimension_sweep_2_{n_genes}",
    )


def main():
    """Run the complete dimension sweep experiment."""
    logger.info(f"Loading single-cell dataset: {TRAINING_DATASET} ...")

    # Load and preprocess data
    if TRAINING_DATASET == "CTI":
        adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
        preprocess_scrna(adata, keep_genes=max(DIMENSION_RANGE), batch_key="donor_id")
        cell_type = f"cell_types_grouped_{TRAINING_CELL_TYPE_GROUP}"
    else:
        raise ValueError(
            f"Dataset {TRAINING_DATASET} not supported for dimension sweep"
        )

    adata, train_test_index = add_cell_types_grouped(adata, TRAINING_CELL_TYPE_GROUP)
    adata_train = adata[train_test_index["Train index"]]
    ranked_genes = adata_train.var["highly_variable_rank"].sort_values().index.tolist()

    # Run experiments
    for n_genes in DIMENSION_RANGE:
        logger.info(f"Starting experiments with {n_genes} genes...")
        try:
            run_single_experiment(adata_train, ranked_genes, n_genes, cell_type)
        except (ValueError, IndexError) as e:
            logger.error(f"Data processing error (n_genes={n_genes}): {str(e)}")
            continue
        except RuntimeError as e:
            logger.error(f"Model training error (n_genes={n_genes}): {str(e)}")
            continue
        ray.shutdown()


if __name__ == "__main__":
    main()

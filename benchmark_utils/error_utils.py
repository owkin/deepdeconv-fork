"""Utilities to compute error metrics from deconvolution outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from .run_benchmark_constants import (
    ERROR_FUNCTIONS,
    GRANULARITY_TO_EVALUATION_DATASET,
    SINGLE_CELL_DATASETS,
    initialize_func,
)


def compute_benchmark_errors(deconv_results, error_type: str) -> pd.DataFrame:
    """Compute different benchmark errors.

    Parameters
    ----------
    deconv_results : dict
        The deconvolution results to evaluate
    error_type : str
        The type of error to compute

    Returns
    -------
    all_results : pd.DataFrame
        The computed errors
    """
    compute_error_fn, _ = initialize_func(ERROR_FUNCTIONS[error_type])
    all_results = []
    for granularity, level1 in deconv_results.items():
        evaluation_dataset = GRANULARITY_TO_EVALUATION_DATASET[granularity]
        if evaluation_dataset in SINGLE_CELL_DATASETS:
            for sampling_method, level2 in level1.items():
                for num_cells, level3 in level2.items():
                    for deconv_method, level4 in level3.items():
                        if level4["deconvolution_results"].isna().any().any():
                            logger.warning(
                                f"Deconv results for the {deconv_method} method for the "
                                f"granularity {granularity} for the sampling method {sampling_method} "
                                f"for the number of cells {num_cells} contains NaN values, so "
                                "error computation will be skipped there."
                            )
                        else:
                            df_error = compute_error_fn(
                                level4["deconvolution_results"], level4["ground_truth"]
                            )
                            df_error["deconv_method"] = deconv_method
                            df_error["num_cells"] = num_cells
                            df_error["sampling_method"] = sampling_method
                            df_error["granularity"] = granularity
                            df_error["error_type"] = error_type
                            all_results.append(df_error)
        else:
            for deconv_method, level2 in level1.items():
                if level2["deconvolution_results"].isna().any().any():
                    logger.warning(
                        f"Deconv results for the {deconv_method} method for the "
                        f"granularity {granularity} contains NaN values, so "
                        "error computation will be skipped there."
                    )
                else:
                    # Normalize ground truth to sum to 1 for each row, here we are discarding the other cell types for bulk data!
                    # TODO: check if this is the correct way to do it or we should just divide by 100 for the bulk data (this way there would be a systematic error in the computation of the errors)
                    level2["ground_truth"] = level2["ground_truth"].div(
                        level2["ground_truth"].sum(axis=1), axis=0
                    )
                    df_error = compute_error_fn(
                        level2["deconvolution_results"], level2["ground_truth"]
                    )
                    df_error["deconv_method"] = deconv_method
                    df_error["granularity"] = granularity
                    df_error["error_type"] = error_type
                    all_results.append(df_error)

    all_results = pd.concat(all_results, ignore_index=True)

    return all_results


def compute_rmse(deconv_results, ground_truth_fractions):
    """Compute Root Mean Squared Error (RMSE) between deconvolution results and ground truth fractions.

    Parameters
    ----------
    deconv_results : DataFrame
        The deconvolution results to evaluate
    ground_truth_fractions : DataFrame
        The ground truth fractions to compare against
    """
    deconv_results = deconv_results[ground_truth_fractions.columns]  # align columns
    rmse = ((deconv_results - ground_truth_fractions) ** 2).mean(axis=1)
    rmse = np.sqrt(rmse)
    rmse = pd.DataFrame({"errors": rmse, "sample_id": deconv_results.index})
    return rmse


def compute_mae(deconv_results, ground_truth_fractions):
    """Compute Mean Absolute Error (MAE) between deconvolution results and ground truth fractions.

    Parameters
    ----------
    deconv_results : DataFrame
        The deconvolution results to evaluate
    ground_truth_fractions : DataFrame
        The ground truth fractions to compare against
    """
    deconv_results = deconv_results[ground_truth_fractions.columns]  # align columns
    mae = (deconv_results - ground_truth_fractions).abs().mean(axis=1)
    mae = pd.DataFrame({"errors": mae, "sample_id": deconv_results.index})
    return mae


def compute_mape(deconv_results, ground_truth_fractions):
    """Compute Mean Absolute Percentage Error (MAPE) between deconvolution results and ground truth fractions.

    Parameters
    ----------
    deconv_results : DataFrame
        The deconvolution results to evaluate
    ground_truth_fractions : DataFrame
        The ground truth fractions to compare against
    """
    deconv_results = deconv_results[ground_truth_fractions.columns]  # align columns
    # Avoid division by zero by replacing zeros with np.nan, then fill with 0 after calculation
    denominator = ground_truth_fractions.replace(0, np.nan)
    mape = ((deconv_results - ground_truth_fractions).abs() / denominator).mean(
        axis=1
    ) * 100
    mape = mape.fillna(0)
    mape = pd.DataFrame({"errors": mape, "sample_id": deconv_results.index})
    return mape


def compute_group_rmse(deconv_results, ground_truth_fractions):
    """Compute cell type RMSE between deconvolution results and ground truth fractions."""
    deconv_results = deconv_results[ground_truth_fractions.columns]
    rmse = ((deconv_results - ground_truth_fractions) ** 2).mean(axis=0)
    rmse = np.sqrt(rmse)
    rmse = pd.DataFrame({"errors": rmse, "cell_types": deconv_results.columns})
    return rmse


def compute_group_mae(deconv_results, ground_truth_fractions):
    """Compute cell type MAE between deconvolution results and ground truth fractions."""
    deconv_results = deconv_results[ground_truth_fractions.columns]
    mae = (deconv_results - ground_truth_fractions).abs().mean(axis=0)
    mae = pd.DataFrame({"errors": mae, "cell_types": deconv_results.columns})
    return mae


def compute_group_mape(deconv_results, ground_truth_fractions):
    """Compute cell type MAPE between deconvolution results and ground truth fractions."""
    deconv_results = deconv_results[ground_truth_fractions.columns]
    denominator = ground_truth_fractions.replace(0, 1e-10)
    mape = ((deconv_results - ground_truth_fractions).abs() / denominator).mean(
        axis=0
    ) * 100
    mape = mape.fillna(0)
    mape = pd.DataFrame({"errors": mape, "cell_types": deconv_results.columns})
    return mape

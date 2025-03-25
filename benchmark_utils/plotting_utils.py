import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from typing import Dict
from datetime import datetime
from loguru import logger


def plot_benchmark_correlations(df_all_correlations, save_path: str = "", save: bool = True):
    """General function to plot benchmark correlations, and save them by default."""
    def _get_groups(df, groupby_col):
        """Returns grouped DataFrames if groupby_col exists and is not empty, else returns a list with the original df."""
        if groupby_col in df.columns and df[groupby_col].notna().any():
            return [group for _, group in df.groupby(groupby_col)]
        return [df]
    
    plot_func_map = {
        "sample_wise_correlation": plot_deconv_results,
        "cell_type_wise_correlation": plot_deconv_results_group
    }

    for granularity in df_all_correlations.granularity.unique():
        df_all_correlations_temp = df_all_correlations[df_all_correlations["granularity"] == granularity]
        if "num_cells" in df_all_correlations_temp.columns and df_all_correlations_temp.num_cells.dropna().nunique() > 1:
            # Multiple num_cells were computed
            df_to_plot = df_all_correlations_temp[df_all_correlations_temp["correlation_type"] == "sample_wise_correlation"]
            for group in _get_groups(df_to_plot, "sampling_method"):
                plot_deconv_lineplot(group, save=save, save_path=save_path)
        else:
            # One pseudobulk num_cells or bulk
            for correlation_type in df_all_correlations_temp["correlation_type"].unique():
                df_to_plot = df_all_correlations_temp[df_all_correlations_temp["correlation_type"] == correlation_type]
                plot_func = plot_func_map.get(correlation_type, plot_deconv_results)
                for group in _get_groups(df_to_plot, "sampling_method"):
                    plot_func(group, save=save, save_path=save_path)


def plot_purified_deconv_results(deconv_results, only_fit_one_baseline, more_details=False, save=False, filename="test"):
    """Plot the deconv results from sanity check 1"""
    if not more_details:
        if only_fit_one_baseline:
            deconv_results = deconv_results.loc[
                deconv_results["Cell type predicted"] == deconv_results["Cell type"]
            ].copy()
            deconv_results["Method"] = "NNLS"
        hue = "Method"
    else:
        hue = "Cell type predicted"

    plt.clf()
    sns.set_style("whitegrid")
    sns.stripplot(
        data=deconv_results, x="Cell type", y="Estimated Fraction", hue=hue
    )
    plt.show()
    if save:
        plt.savefig(f"/home/owkin/project/plots/{filename}.png", dpi=300)


def plot_deconv_results(correlations, save=False, save_path="", filename=""):
    """Plot the deconv correlation results from sanity checks 2 and 3."""
    if filename == "":
        granularity = correlations["granularity"].unique()[0]
        if "sampling_method" in correlations.columns and isinstance(correlations["sampling_method"].unique()[0],str):
            sampling_method = correlations["sampling_method"].unique()[0]
            filename = f"{granularity}_{sampling_method}_sampling_correlation_boxplot"
        else:
            filename = f"{granularity}_correlation_boxplot"

    correlations = correlations[["correlations","deconv_method"]]
    plt.clf()
    sns.set_style("whitegrid")
    # Boxplot
    boxplot = sns.boxplot(correlations, x="deconv_method", y="correlations", hue="deconv_method")
    plt.xticks(rotation=90)
    # Plot the medians
    x_categories = [t.get_text() for t in boxplot.get_xticklabels()] # order of categories
    medians = (
        correlations.groupby("deconv_method")["correlations"]
        .median()
        .reindex(x_categories)  # ensure same order as x-axis
        .round(4)
    )
    y_range = correlations["correlations"].max() - correlations["correlations"].min()
    vertical_offset = y_range * 0.0005
    y_position = medians + vertical_offset
    for xtick, method in enumerate(x_categories):
        median_value = medians.loc[method]
        if np.isfinite(median_value):
            boxplot.text(
                xtick,
                y_position.loc[method],
                f"{median_value:.4f}",
                size="x-small",
                color="black",
                ha="center",
                weight="semibold",
            )
    if save:
        plt.savefig(f"{save_path}/{filename}.png", dpi=300)

def plot_deconv_results_group(correlations_group, save=False, save_path="", filename=""):
    """Plot the deconv correlation results from sanity checks 2 and 3.
    per cell type.
    """
    if filename == "":
        granularity = correlations_group["granularity"].unique()[0]
        if "sampling_method" in correlations_group.columns and isinstance(correlations_group["sampling_method"].unique()[0], str):
            sampling_method = correlations_group["sampling_method"].unique()[0]
            filename = f"{granularity}_{sampling_method}_sampling_cell_type_correlation_plot"
        else:
            filename = f"{granularity}_cell_type_plot"

    df = correlations_group[["correlations","deconv_method", "cell_types"]]
    df = df.fillna(0) # Replace NaN with zeros
    plt.clf()
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="cell_types", y="correlations", hue="deconv_method", data=df)
    plt.legend()
    plt.xlabel("Cell Type")
    plt.ylabel("Correlation")
    plt.title("Bar Plot of Correlations by Cell Type and Model")
    plt.show()
    if save:
        plt.savefig(f"{save_path}/{filename}.png", dpi=300)

def plot_deconv_lineplot(results: Dict[int, pd.DataFrame],
                         save=False,
                         save_path="",
                         filename=""):
    if filename == "":
        granularity = results["granularity"].unique()[0]
        sampling_method = results["sampling_method"].unique()[0]
        filename = f"{granularity}_{sampling_method}_sampling_numcells_lineplot"

    results = results[["correlations","deconv_method", "num_cells"]]
    plt.clf()
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results,
                 x="num_cells",
                 y="correlations",
                 hue="deconv_method")

    plt.title("Pearson correlation coefficient (vs) # sampled cells")
    plt.xlabel("Number of cells per pseudobulk")
    plt.ylabel("Correlation")
    plt.show()

    if save:
        path = f"/home/owkin/project/plots/{filename}.png"
        if os.path.isfile(path):
            new_path = f"/home/owkin/project/plots/{filename}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.png"
            logger.warning(f"{path} already exists. Saving file on this path instead: {new_path}")
            path = new_path
        plt.savefig(f"{save_path}/{filename}.png", dpi=300)
        logger.debug(f"Plot saved to the following path: {save_path}/{filename}.png")


def plot_metrics(model_history, train: bool = True, n_epochs: int = 100):
    """Plot the train or val metrics from training."""
    if train:
        suffix = "train"
    else:
        suffix = "validation"
    plt.clf()
    plt.plot(
        range(n_epochs),
        model_history[f"pearson_coeff_{suffix}"],
        label="Latent space Pearson coefficient",
    )
    plt.plot(
        range(n_epochs),
        model_history[f"pearson_coeff_deconv_{suffix}"],
        label="Deconv pearson coefficient",
    )
    plt.plot(
        range(n_epochs),
        model_history[f"cosine_similarity_{suffix}"],
        label="Deconv cosine similarity",
    )
    plt.legend()
    plt.title(f"{suffix} metrics")
    plt.show()

def plot_mse_mae_deconv(model_history, train: bool = True, n_epochs: int = 100):
    """Plot the train or val MSE and MAE deconv errors from training."""
    if train:
        suffix = "train"
    else:
        suffix = "validation"
    plt.clf()
    plt.plot(
        range(n_epochs), 
        model_history[f"mse_deconv_{suffix}"], 
        label="Deconv MSE error",
    )
    plt.plot(
        range(n_epochs), 
        model_history[f"mae_deconv_{suffix}"], 
        label="Deconv MAE error",
    )
    plt.legend()
    plt.title(f"{suffix} errors")
    plt.show()

def plot_loss(model_history, n_epochs: int = 100):
    """Plot the train and val loss from training."""
    plt.clf()
    plt.plot(range(n_epochs), model_history["train_loss_epoch"], label="Train")
    plt.plot(
        range(n_epochs),
        model_history["validation_loss"],
        label="Validation",
    )
    plt.legend()
    plt.title("Loss epochs")
    plt.show()

def plot_mixup_loss(model_history, n_epochs: int = 100):
    """Plot the train and val mixup loss from training."""
    plt.clf()
    plt.plot(range(n_epochs), model_history["mixup_penalty_train"], label="Train")
    plt.plot(
        range(n_epochs),
        model_history["mixup_penalty_validation"],
        label="Validation",
    )
    plt.legend()
    plt.title("Mixup loss epochs")
    plt.show()

def plot_reconstruction_loss(model_history, n_epochs: int = 100):
    """Plot the train and val reconstruction loss from training."""
    plt.clf()
    plt.plot(range(n_epochs), model_history["reconstruction_loss_train"], label="Train")
    plt.plot(
        range(n_epochs),
        model_history["reconstruction_loss_validation"],
        label="Validation",
    )
    plt.legend()
    plt.title("Reconstruction loss epochs")
    plt.show()

def plot_kl_loss(model_history, n_epochs: int = 100):
    """Plot the train and val KL loss from training."""
    plt.clf()
    plt.plot(range(n_epochs), model_history["kl_local_train"], label="Train")
    plt.plot(
        range(n_epochs),
        model_history["kl_local_validation"],
        label="Validation",
    )
    plt.legend()
    plt.title("KL loss epochs")
    plt.show()

def plot_pearson_random(model_history, train: bool = True, n_epochs: int = 100):
    """Plot the train or val random vs normal pearson deconv metrics from training."""
    if train:
        suffix = "train"
    else:
        suffix = "validation"
    plt.clf()
    plt.plot(
        range(n_epochs),
        model_history[f"pearson_coeff_deconv_{suffix}"],
        label="Deconv pearson coefficient",
    )
    plt.plot(
        range(n_epochs),
        model_history[f"pearson_coeff_random_{suffix}"],
        label="Deconv pearson coefficient random vector",
    )
    plt.legend()
    plt.title(f"{suffix} metrics")
    plt.show()

def compare_tuning_results(
      all_results, variable_to_plot: str, variable_tuned: str, n_epochs: int = 100, hp_index_to_plot: list = None
):
    """Plot the train or val losses for a selection of hyperparameters."""
    all_hp = all_results[variable_tuned].unique()
    if all_hp.dtype != "0":
        all_hp.sort()
    if hp_index_to_plot is not None:
        hp_to_plot = all_hp[hp_index_to_plot]
        all_results = all_results.loc[all_results[variable_tuned].isin(hp_to_plot)]
    
    custom_palette = sns.color_palette("husl", n_colors=len(all_results[variable_tuned].unique()))
    all_results["epoch"] = all_results.index
    if (n_nan := all_results[variable_to_plot].isna().sum()) > 0:
        print(
            f"There are {n_nan} missing values in the variable to plot ({variable_to_plot})."
            "Filling them with the next row values."
        )
        all_results[variable_to_plot] = all_results[variable_to_plot].fillna(method='bfill')
    sns.set_theme(style="darkgrid")
    sns.lineplot(x="epoch", y=variable_to_plot, hue=variable_tuned, ci="sd", data=all_results, err_style="bars", palette=custom_palette)
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from typing import Dict

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


def plot_deconv_results(correlations, save=False, filename="test"):
    """Plot the deconv correlation results from sanity checks 2 and 3."""
    plt.clf()
    sns.set_style("whitegrid")
    boxplot = sns.boxplot(correlations)
    medians = correlations.median(numeric_only=True).round(4)
    vertical_offset = (
        correlations.median() * 0.0005
    )  # for non-overlapping display
    y_position = medians + vertical_offset
    for xtick in range(len(medians)):  # show the median value
        if np.isfinite(y_position[xtick]):
            boxplot.text(
                xtick,
                y_position[xtick],
                medians[xtick],
                size="x-small",
                color="w",
                weight="semibold",
            )
    plt.show()
    if save:
        plt.savefig(f"/home/owkin/project/plots/{filename}.png", dpi=300)

def plot_deconv_results_group(correlations_group, save=False, filename="test_group"):
    """Plot the deconv correlation results from sanity checks 2 and 3.
    per cell type.
    """
    plt.clf()
    sns.set_style("whitegrid")
    df = correlations_group.copy()
    df["group"] = df.index.to_list()
    df.set_index('group', inplace=True)
    # Replace NaN with zeros
    df = df.fillna(0)

    # Bar plot
    plt.figure(figsize=(10, 6))

    # Create a long-form DataFrame
    df_melted = pd.melt(df.reset_index(), id_vars='group', var_name='Model', value_name='Correlation')

    # Create a bar plot using seaborn
    sns.barplot(x='group', y='Correlation', hue='Model', data=df_melted)

    # Add legend
    plt.legend()

    # Set plot labels and title
    plt.xlabel('Cell Type')
    plt.ylabel('Correlation')
    plt.title('Bar Plot of Correlations by Cell Type and Model')
    plt.show()
    if save:
        plt.savefig(f"/home/owkin/project/plots/{filename}.png", dpi=300)

def plot_deconv_lineplot(results: Dict[int, pd.DataFrame],
                         save=False,
                         filename="sim_pseudobulk_lineplot"):
    for key, df in results.items():
        df = pd.melt(df,
                    var_name='method',
                    value_name='pcc',
        )
        df["n_cells"] = key
        results[key] = df
    df_results = pd.concat(results.values(), axis=0)

    sns.lineplot(data=df_results,
                 x="n_cells",
                 y="pcc",
                 hue="method")

    plt.title("Pearson correlation coefficient (vs) # sampled cells")
    plt.xlabel('Cell Type')
    plt.ylabel('Correlation')
    plt.show()

    if save:
        plt.savefig(f"/home/owkin/project/plots/{filename}.png", dpi=300)

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
    sns.set_theme(style="darkgrid")
    sns.lineplot(x="epoch", y=variable_to_plot, hue=variable_tuned, ci="sd", data=all_results, err_style="bars", palette=custom_palette)
    plt.show()

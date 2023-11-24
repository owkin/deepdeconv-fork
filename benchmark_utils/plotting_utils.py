import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_purified_deconv_results(deconv_results, only_fit_baseline_nnls, more_details=False, save=False, filename="test"):
    """Plot the deconv results from sanity check 1"""
    if not more_details:
        if only_fit_baseline_nnls:
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
        plt.savefig(f"/home/owkin/project/sanity_checks/{filename}.png", dpi=300)


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
        plt.savefig(f"/home/owkin/project/sanity_checks/{filename}.png", dpi=300)


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

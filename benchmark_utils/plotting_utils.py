import matplotlib.pyplot as plt
import seaborn as sns


def plot_deconv_results(correlations):
    sns.set_style("whitegrid")
    boxplot = sns.boxplot(correlations, y="correlations", x="Method")
    medians = correlations.groupby(["Method"])["correlations"].median().round(4)
    vertical_offset = (
        correlations["correlations"].median() * 0.0005
    )  # for non-overlapping display
    for xtick in boxplot.get_xticks():  # show the median value
        boxplot.text(
            xtick,
            medians[xtick] + vertical_offset,
            medians[xtick],
            horizontalalignment="center",
            size="x-small",
            color="w",
            weight="semibold",
        )
    plt.show()


def plot_metrics(model_history, train: bool = True, n_epochs: int = 100):
    """Plot the train or val metrics from training."""
    if train:
        suffix = "train"
    else:
        suffix = "validation"
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

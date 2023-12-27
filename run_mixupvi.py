"""Run MixUpVI experiments with the right sanity checks."""

# %%
import scanpy as sc
import scvi
from loguru import logger
import warnings

from benchmark_utils import (
    fit_mixupvi,
    tune_mixupvi,
    preprocess_scrna,
    add_cell_types_grouped,
    plot_metrics,
    plot_loss,
    plot_mixup_loss,
    plot_reconstruction_loss,
    plot_kl_loss,
    plot_pearson_random,
    compare_tuning_results,
    read_tuning_results,
    read_search_space,
)
from constants import (
    TUNE_MIXUPVI,
    SAVE_MODEL,
    PATH,
    TRAINING_DATASET,
    TRAINING_LOG,
    TRAINING_CELL_TYPE_GROUP,
)
from tuning_configs import (
    SEARCH_SPACE, TUNED_VARIABLES, NUM_SAMPLES, METRIC, ADDITIONAL_METRICS,
)


# %% Load scRNAseq dataset
logger.info(f"Loading single-cell dataset: {TRAINING_DATASET} ...")
cell_type = "cell_types_grouped"
if TRAINING_DATASET == "TOY":
    adata_train = scvi.data.heart_cell_atlas_subsampled()
    preprocess_scrna(adata_train, keep_genes=1200, log=TRAINING_LOG)
    cell_type = "cell_type"
elif TRAINING_DATASET == "CTI":
    # Load processed for speed-up (already filtered, normalised, etc.)
    adata = sc.read("/home/owkin/cti_data/processed/cti_processed_2500.h5ad")
    # adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
    # preprocess_scrna(adata,
    #                  keep_genes=2500,
    #                  log=TRAINING_LOG,
    #                  batch_key="donor_id")
elif TRAINING_DATASET == "CTI_RAW":
    warnings.warn("The raw data of this adata is on adata.raw.X, but the normalised "
                  "adata.X will be used here")
    adata = sc.read("/home/owkin/data/cross-tissue/omics/raw/local.h5ad")
    preprocess_scrna(adata,
                     keep_genes=2500,
                     log=TRAINING_LOG,
                     batch_key="donor_id",
    )
elif TRAINING_DATASET == "CTI_PROCESSED":
    adata = sc.read("/home/owkin/cti_data/processed/cti_processed_2500.h5ad")
    # adata = sc.read("/home/owkin/cti_data/processed/cti_processed_batch.h5ad")


# %% Add cell types groups and split train/test
if TRAINING_DATASET != "TOY":
    adata, train_test_index = add_cell_types_grouped(adata, TRAINING_CELL_TYPE_GROUP)
    adata_train = adata[train_test_index["Train index"]]
    adata_test = adata[train_test_index["Test index"]]


# %% Fit MixUpVI with hyperparameters defined in constants.py
adata_train = adata_train.copy()
if TUNE_MIXUPVI:
    all_results, best_hp, tuning_path, search_path = tune_mixupvi(
        adata_train,
        cell_type_group=cell_type,
        search_space=SEARCH_SPACE,
        metric=METRIC,
        additional_metrics=ADDITIONAL_METRICS,
        num_samples=NUM_SAMPLES,
        training_dataset=TRAINING_DATASET,
    )
    model_history = all_results.loc[all_results.hyperparameter == best_hp] # plots for the best hp found by tuning
else:
    model = fit_mixupvi(
        adata_train,
        model_path=PATH,
        cell_type_group=cell_type,
        save_model=SAVE_MODEL
    )
    model_history = model.history


# %% Load model / results: Uncomment if not running previous cells
# if TUNE_MIXUPVI:
#     # path = "/home/owkin/project/mixupvi_tuning/n_latent/TOY_dataset_tune_mixupvi_2023-12-15-11:15:32"
#     # path = "/home/owkin/project/mixupvi_tuning/batch_size/CTI_dataset_tune_mixupvi_2023-12-20-12:35:19"
#     path = "/home/owkin/project/mixupvi_tuning/signature_type/CTI_dataset_tune_mixupvi_2023-12-20-12:35:44"
#     path = "/home/owkin/project/mixupvi_tuning/pseudo_bulk/CTI_dataset_tune_mixupvi_2023-12-20-12:35:19"
#     all_results = read_tuning_results(f"{path}/tuning_results.csv")
#     search_space = read_search_space(f"{path}/search_space.pkl")
#     best_hp = search_space["best_hp"]
#     model_history = all_results.loc[all_results.hyperparameter == "pre"] # plots for the best hp found by tuning
# else:
#     import torch
#     path = "/home/owkin/project/scvi_models/models/toy_100_epochs"
#     model = torch.load(f"{path}/model.pt")
#     model_history = model["attr_dict"]["history_"]


# %% Plots for a given model
n_epochs = len(model_history["train_loss_epoch"])
plot_metrics(model_history, train=True, n_epochs=n_epochs)
plot_metrics(model_history, train=False, n_epochs=n_epochs)
plot_loss(model_history, n_epochs=n_epochs)
plot_mixup_loss(model_history, n_epochs=n_epochs)
plot_reconstruction_loss(model_history, n_epochs=n_epochs)
plot_kl_loss(model_history, n_epochs=n_epochs)
plot_pearson_random(model_history, train=True, n_epochs=n_epochs)
plot_pearson_random(model_history, train=False, n_epochs=n_epochs)


# %% Plots to compare HPs
if TUNE_MIXUPVI:
    n_epochs = len(model_history["train_loss_epoch"])
    tuned_variable = TUNED_VARIABLES[0]
    hp_index_to_plot = [0, 1, 3, 4, 5] # only these index (of the HPs tried) will be plotted, for clearer visualisation
    compare_tuning_results(
        all_results, variable_to_plot="validation_loss", variable_tuned=tuned_variable,
        n_epochs=n_epochs, hp_index_to_plot=hp_index_to_plot,
    )
    compare_tuning_results( # latent space pearson coeff
        all_results, variable_to_plot="pearson_coeff_validation", variable_tuned=tuned_variable,
        n_epochs=n_epochs, hp_index_to_plot=hp_index_to_plot,
    )
    compare_tuning_results( # deconv pearson coefficient
        all_results, variable_to_plot="pearson_coeff_deconv_validation", variable_tuned=tuned_variable,
        n_epochs=n_epochs, hp_index_to_plot=hp_index_to_plot,
    )
    compare_tuning_results( # deconv cosine similarity
        all_results, variable_to_plot="cosine_similarity_validation", variable_tuned=tuned_variable,
        n_epochs=n_epochs, hp_index_to_plot=hp_index_to_plot,
    )
    compare_tuning_results(
        all_results, variable_to_plot="mixup_penalty_validation", variable_tuned=tuned_variable,
        n_epochs=n_epochs, hp_index_to_plot=hp_index_to_plot,
    )
    compare_tuning_results(
        all_results, variable_to_plot="reconstruction_loss_validation", variable_tuned=tuned_variable,
        n_epochs=n_epochs, hp_index_to_plot=hp_index_to_plot,
    )
    compare_tuning_results(
        all_results, variable_to_plot="kl_local_validation", variable_tuned=tuned_variable,
        n_epochs=n_epochs, hp_index_to_plot=hp_index_to_plot,
    )
# %%

"""Run MixUpVI experiments with the right sanity checks."""

# %%
import scanpy as sc
import scvi
from loguru import logger
import warnings

from benchmark_utils import (
    fit_mixupvi,
    preprocess_scrna,
    add_cell_types_grouped,
    plot_metrics,
    plot_loss,
    plot_mixup_loss,
    plot_reconstruction_loss,
    plot_kl_loss,
    plot_pearson_random,
)
from constants import (
    SAVE_MODEL,
    PATH,
    TRAINING_DATASET,
    TRAINING_LOG,
    TRAINING_CELL_TYPE_GROUP,
)


# %% Load scRNAseq dataset
logger.info(f"Loading single-cell dataset: {TRAINING_DATASET} ...")
if TRAINING_DATASET == "TOY":
    adata_train = scvi.data.heart_cell_atlas_subsampled()
    preprocess_scrna(adata_train, keep_genes=1200, log=TRAINING_LOG)
    CAT_COV = ["cell_type"]
elif TRAINING_DATASET == "CTI":
    adata = sc.read("/home/owkin/deepdeconv/data/cti_adata.h5ad")
    preprocess_scrna(adata,
                     keep_genes=2500,
                     log=TRAINING_LOG,
                     batch_key="donor_id")
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
    adata = sc.read("/home/owkin/cti_data/processed/cti_processed.h5ad")
    # adata = sc.read("/home/owkin/cti_data/processed/cti_processed_batch.h5ad")


# %% Add cell types groups and split train/test
if TRAINING_DATASET != "TOY":
    adata, train_test_index = add_cell_types_grouped(adata, TRAINING_CELL_TYPE_GROUP)
    adata_train = adata[train_test_index["Train index"]]
    adata_test = adata[train_test_index["Test index"]]
    CAT_COV = ["cell_types_grouped"]


# %% Fit MixUpVI with hyperparameters defined in constants.py
adata_train = adata_train.copy()
model = fit_mixupvi(
    adata_train, 
    model_path=PATH, 
    cell_type_group="cell_types_grouped", 
    save_model=SAVE_MODEL
)


# %%
# TODO: add option to load model
n_epochs = len(model.history["train_loss_epoch"])
plot_metrics(model.history, train=True, n_epochs=n_epochs)
plot_metrics(model.history, train=False, n_epochs=n_epochs)
plot_loss(model.history, n_epochs=n_epochs)
plot_mixup_loss(model.history, n_epochs=n_epochs)
plot_reconstruction_loss(model.history, n_epochs=n_epochs)
plot_kl_loss(model.history, n_epochs=n_epochs)
plot_pearson_random(model.history, train=True, n_epochs=n_epochs)
plot_pearson_random(model.history, train=False, n_epochs=n_epochs)
# %%

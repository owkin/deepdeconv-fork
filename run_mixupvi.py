"""Run MixUpVI experiments with the right sanity checks."""

# %%
import scanpy as sc
import scvi
from loguru import logger
import warnings

from benchmark_utils import (
    run_categorical_value_checks,
    run_incompatible_value_checks,
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
    MODEL_SAVE,
    PATH,
    TRAINING_DATASET,
    TRAINING_LOG,
    MAX_EPOCHS,
    BATCH_SIZE,
    TRAIN_SIZE,
    TRAINING_CELL_TYPE_GROUP,
    CONT_COV,
    ENCODE_COVARIATES,
    ENCODE_CONT_COVARIATES,
    SIGNATURE_TYPE,
    USE_BATCH_NORM,
    LOSS_COMPUTATION,
    PSEUDO_BULK,
    MIXUP_PENALTY,
    DISPERSION,
    GENE_LIKELIHOOD,
)
if TRAIN_SIZE == 1:
    TRAIN_SIZE = 0.9 # to have validation

# Run sanity checks
run_categorical_value_checks(
    cell_group=TRAINING_CELL_TYPE_GROUP,
    cont_cov=CONT_COV,
    encode_covariates=ENCODE_COVARIATES,
    encode_cont_covariates=ENCODE_CONT_COVARIATES,
    use_batch_norm=USE_BATCH_NORM,
    signature_type=SIGNATURE_TYPE,
    loss_computation=LOSS_COMPUTATION,
    pseudo_bulk=PSEUDO_BULK,
    mixup_penalty=MIXUP_PENALTY,
    dispersion=DISPERSION,
    gene_likelihood=GENE_LIKELIHOOD,
)
run_incompatible_value_checks(
    pseudo_bulk=PSEUDO_BULK,
    loss_computation=LOSS_COMPUTATION,
    use_batch_norm=USE_BATCH_NORM,
    mixup_penalty=MIXUP_PENALTY,
    gene_likelihood=GENE_LIKELIHOOD,
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


# %% add cell types groups and split train/test
if TRAINING_DATASET != "TOY":
    adata, train_test_index = add_cell_types_grouped(adata, TRAINING_CELL_TYPE_GROUP)
    adata_train = adata[train_test_index["Train index"]]
    adata_test = adata[train_test_index["Test index"]]
    CAT_COV = ["cell_types_grouped"]


# %% Train mixupVI
adata_train = adata_train.copy()
scvi.model.MixUpVI.setup_anndata(
    adata_train,
    layer="counts",
    categorical_covariate_keys=CAT_COV,  # only cell types for now
    # continuous_covariate_keys=["percent_mito", "percent_ribo"],
)
model = scvi.model.MixUpVI(
    adata_train,
    use_batch_norm=USE_BATCH_NORM,
    signature_type=SIGNATURE_TYPE,
    loss_computation=LOSS_COMPUTATION,
    pseudo_bulk=PSEUDO_BULK,
    encode_covariates=ENCODE_COVARIATES,  # always False for now, because cat covariates is only cell types
    encode_cont_covariates=ENCODE_CONT_COVARIATES,  # if want to encode continuous covariates
    mixup_penalty=MIXUP_PENALTY,
    dispersion=DISPERSION,
    gene_likelihood=GENE_LIKELIHOOD,
)
model.view_anndata_setup()
model.train(
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    train_size=TRAIN_SIZE,
    check_val_every_n_epoch=1,
)

## Save model
if MODEL_SAVE:
    model.save(PATH)


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

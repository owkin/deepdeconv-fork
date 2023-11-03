"""Run MixUpVI experiments with the right sanity checks."""

# %%
import scanpy as sc
import scvi
import pandas as pd

from run_mixupvi_utils import (
    run_categorical_value_checks,
    run_incompatible_value_checks,
    plot_metrics,
    plot_loss,
    plot_mixup_loss,
    plot_reconstruction_loss,
    plot_kl_loss,
    plot_pearson_random,
)
from notebooks.sanity_checks_utils import GROUPS

# general constants
MODEL_SAVE = False
PATH = "/home/owkin/project/scvi_models/models/cti_linear_test"
USE_CTI_DATA = False  # if False, use toy dataset
MAX_EPOCHS = 100
BATCH_SIZE = 1024
TRAIN_SIZE = 0.7
# MixUpVI specific constants and constraints
CELL_GROUP = (
    "primary_groups"  # ["primary_groups", "precise_groups", "updated_granular_groups"]
)
CAT_COV = [CELL_GROUP]  # for now, only works with cell groups as categorical covariate
CONT_COV = None  # list of continuous covariates to include
ENCODE_COVARIATES = False  # should be always False for now, we don't encode cat covar
ENCODE_CONT_COVARIATES = False  # True or False, whether to include cont covar
SIGNATURE_TYPE = "pre_encoded"  # ["pre_encoded", "post_inference"]
USE_BATCH_NORM = "none"  # ["encoder", "decoder", "none", "both"]
LOSS_COMPUTATION = "latent_space"  # ["latent_space", "reconstructed_space"]
PSEUDO_BULK = "pre_encoded"  # ["pre_encoded", "post_inference"]
MIXUP_PENALTY = "l2"  # ["l2", "kl"]
DISPERSION = "gene"  # ["gene", "gene_cell"]
GENE_LIKELIHOOD = "zinb"  # ["zinb", "nb", "poisson"]

## Run sanity checks
run_categorical_value_checks(
    cell_group=CELL_GROUP,
    cat_cov=CAT_COV,
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


# %% Run MixUpVI experiment
## Cross-immune
if USE_CTI_DATA:
    adata = sc.read("/home/owkin/data/cross-tissue/omics/raw/local.h5ad")
    adata.layers["counts"] = adata.raw.X.copy()
    adata.X = adata.raw.X.copy()  # copy counts
    batch_key = "assay"
else:
    adata = scvi.data.heart_cell_atlas_subsampled()
    adata.layers["counts"] = adata.X.copy()  # preserve counts
    batch_key = "cell_source"
    CAT_COV = ["cell_type"]

## Preprocess
sc.pp.filter_genes(adata, min_counts=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata  # freeze the state in `.raw`
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=1200,  # to tune
    subset=True,
    layer="counts",
    flavor="seurat_v3",
    batch_key=batch_key,
)

if USE_CTI_DATA:
    ## Create the cell type categories
    groups = GROUPS[CELL_GROUP]
    group_correspondence = {}
    for k, v in groups.items():
        for cell_type in v:
            group_correspondence[cell_type] = k
    adata.obs[CELL_GROUP] = [
        group_correspondence[cell_type]
        for cell_type in adata.obs.Manually_curated_celltype
    ]
    # remove some cell types: you need more than 15GB memory to run that
    index_to_keep = adata.obs.loc[adata.obs[CELL_GROUP] != "To remove"].index
    ## Train test split
    train_test_index_matrix_common = pd.read_csv(
        "/home/owkin/project/train_test_index_matrix_common.csv", index_col=1
    )
    adata_train = adata[train_test_index_matrix_common["Train index"]]
    adata_test = adata[train_test_index_matrix_common["Test index"]]
else:
    adata_train = adata.copy()

## Train mixupVI
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

"""Pseudobulk benchmark."""
# %%
import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
from loguru import logger
import anndata as ad

from .utils import (
    perform_nnls,
    compute_correlations,
    create_signature,
    fit_scvi,
    fit_destvi,
    #fit_mixupvi,
    preprocess_scrna,
    create_pseudobulk_dataset,
    split_dataset,
    run_categorical_value_checks
)

# %% params
DATASET = "TOY" # "CTI"
SIGNATURE_TYPE = "almudena"  # ["laughney", "almudena", "crosstissue_general", "crosstissue_granular_updated"]
CELL_TYPE_GROUP = "primary_groups"  # ["primary_groups", "precise_groups"]
# %%
logger.info("Loading single-cell dataset ...")

if DATASET == "TOY":
    adata = scvi.data.heart_cell_atlas_subsampled()
elif DATASET == "CTI":
    adata = sc.read("/home/owkin/data/cross-tissue/omics/raw/local.h5ad")

#%% load signature
signature = create_signature(adata,
                             signature_type=SIGNATURE_TYPE,
                             group=CELL_TYPE_GROUP)
# %% split train/test
adata_train, adata_test = split_dataset(adata)

# %% Create pseudobulk dataset
logger.info("Creating pseudobulk dataset...")
adata_pseudobulk, ground_truth_fractions = create_pseudobulk_dataset(adata_test)

# %% ground truth cell type fractions
ground_truth_fractions = pd.DataFrame(ground_truth_fractions, index=adata_pseudobulk.obs_names)
ground_truth_fractions = ground_truth_fractions.fillna(0)  # the Nan are cells not sampled

# %%
# Create and train models

### %% 1. scVI
adata_train = adata_train.copy()
adata_test = adata_test.copy()

logger.info("Fit scVI ...")
model_path = f"models/{DATASET}_scvi.pkl"
scvi_model = fit_scvi(adata_train, model_path)

#### %% 2. DestVI
logger.info("Fit DestVI ...")
model_path_1 = f"models/{DATASET}_condscvi.pkl"
model_path_2 = f"models/{DATASET}_destvi.pkl"
condscvi_model , destvi_model= fit_destvi(adata_train=adata_train,
                                          adata_pseudobulk=adata_pseudobulk,
                                          model_path_1=model_path_1,
                                          model_path_2=model_path_2)

#### %% 3. MixupVI
logger.info("Train mixupVI ...")
# scvi.model.MixUpVI.setup_anndata(
#     adata,
#     layer="counts",
#     categorical_covariate_keys=["cell_type"],  # no other cat covariate for now
#     # continuous_covariate_keys=["percent_mito", "percent_ribo"],
# )
# model = scvi.model.MixUpVI(
#     adata, signature_type="post_encoded", loss_computation="reconstructed_space"
# )
# model.view_anndata_setup()
# model.train(max_epochs=100, batch_size=1024, train_size=0.9, check_val_every_n_epoch=1)


#
adata_pseudobulk.obsm["proportions"] = destvi_model.get_proportions()
#

# Deconv
# `n_sample`` different deconvolutions with n_marker_genes observations each
deconv_results = perform_nnls(signature, averaged_data)

# compute correlations
correlations = compute_correlations(deconv_results, ground_truth_fractions)

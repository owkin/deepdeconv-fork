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

from .utils.dataset_utils import preprocess_scrna, split_dataset, create_pseudobulk_dataset
from .utils.benchmark_utils import perform_nnls, perform_latent_deconv, compute_correlations, create_random_proportion
from .utils.latent_signature_utils import create_latent_signature
from .utils.training_utils import fit_scvi, fit_destvi, fit_mixupvi
from .utils.signature_utils import create_signature, add_cell_types_grouped
from .utils.plotting_utils import plot_correlations

# %% params
DATASET = "TOY" # "CTI"
SIGNATURE_CHOICE = "almudena"  # ["laughney", "almudena", "crosstissue_general", "crosstissue_granular_updated"]
CELL_TYPE_GROUP = "primary_groups"  # ["primary_groups", "precise_groups"]

# %% Load scRNAseq dataset
logger.info("Loading single-cell dataset ...")

if DATASET == "TOY":
    adata = scvi.data.heart_cell_atlas_subsampled()
    preprocess_scrna(adata, keep_genes=1200)
elif DATASET == "CTI":
    adata = sc.read("/home/owkin/data/cross-tissue/omics/raw/local.h5ad")
    preprocess_scrna(adata, keep_genes=2500)


#%% load signature
signature = create_signature(adata,
                             signature_type=SIGNATURE_CHOICE,
                             group=CELL_TYPE_GROUP)
add_cell_types_grouped(adata)
# %% split train/test
adata_train, adata_test = split_dataset(adata)

# %% Create pseudobulk dataset
logger.info("Creating pseudobulk dataset...")
adata_pseudobulk_train, pseudobulk_train, proportions_train = create_pseudobulk_dataset(adata_train)
adata_pseudobulk_test, pseudobulk_test, proportions_test = create_pseudobulk_dataset(adata_test)

# %% ground truth cell type fractions
df_proportions_train = pd.DataFrame(proportions_train,
                                    index=adata_pseudobulk_train.obs_names)
df_proportions_test = pd.DataFrame(proportions_test,
                                    index=adata_pseudobulk_train.obs_names)
df_proportions_train = df_proportions_train.fillna(0)  # the Nan are cells not sampled
df_proportions_test = df_proportions_test.fillna(0)  # the Nan are cells not sampled
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
condscvi_model , destvi_model= fit_destvi(adata_train,
                                          adata_pseudobulk_train,
                                          model_path_1,
                                          model_path_2)

#### %% 3. MixupVI
logger.info("Train mixupVI ...")
model_path = f"models/{DATASET}_mixupvi.pkl"
mixupvi_model = fit_mixupvi(adata_train, model_path)
#

# %% Deconvolution
df_predicted_proportions = pd.DataFrame(index=adata_pseudobulk_test.obs_names,
                              columns=["scVI", "DestVI", "MixupVI", "Random", "NNLS"])
df_test_correlations = pd.DataFrame(index=adata_pseudobulk_test.obs_names,
                              columns=["scVI", "DestVI", "MixupVI", "Random", "NNLS"])
### % Random proportions
# TO DO:
#

### % Linear regression (NNLS)
deconv_results = perform_nnls(signature, pseudobulk_test)
correlations = compute_correlations(deconv_results, df_proportions_test)
df_predicted_proportions.loc[:, "NNLS"] = deconv_results.values
df_test_correlations.loc[:, "NNLS"] = correlations.values

### % scVI
latent_signature = create_latent_signature(adata_test, scvi_model, sc_per_pseudobulk=1000)
deconv_results = perform_latent_deconv(adata_pseudobulk_test, scvi_model, latent_signature)
correlations = compute_correlations(deconv_results, df_proportions_test)
df_predicted_proportions.loc[:, "scVI"] = deconv_results.values
df_test_correlations.loc[:, "scVI"] = correlations.values

### % MixupVI
latent_signature = create_latent_signature(adata_test, mixupvi_model, sc_per_pseudobulk=1000)
deconv_results = perform_latent_deconv(adata_pseudobulk_test, scvi_model, latent_signature)
df_predicted_proportions.loc[:, "MixupVI"] = deconv_results.values
df_test_correlations.loc[:, "MixupVI"] = correlations.values

### % DestVI
deconv_results = destvi_model.get_proportions(adata_pseudobulk_test, deterministic=True)
correlations = compute_correlations(deconv_results, df_proportions_test)
df_predicted_proportions.loc[:, "DestVI"] = deconv_results
df_test_correlations.loc[:, "DestVI"] = correlations.values

### % Plots
plot_correlations(df_test_correlations, "test")

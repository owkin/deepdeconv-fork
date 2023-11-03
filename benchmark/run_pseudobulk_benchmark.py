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

from sklearn.model_selection import train_test_split

# from ... import compute_signature_matrix
from scipy.optimize import perform_nnls
# %% params
DATASET = "TOY"

# %%
logger.info("Loading single-cell dataset ...")
adata = scvi.data.heart_cell_atlas_subsampled()

# %%
# train/test split
cell_types_train, cell_types_test = train_test_split(
    adata.obs_names,
    test_size=0.5,
    stratify=adata.obs.cell_types_grouped,
    random_state=42,
)

adata_train = adata[cell_types_train, :]
adata_test = adata[cell_types_test, :]

# %% Create pseudobulk dataset
logger.info("Creating pseudobulk dataset...")

n_sample = 500
random.seed(42)
averaged_data = []
ground_truth_fractions = []
for i in range(n_sample):
    cell_sample = random.sample(list(adata_test.obs_names), 1000)
    adata_sample = adata_test[cell_sample, :]
    ground_truth_frac = adata_sample.obs.cell_types_grouped.value_counts() / 1000
    ground_truth_fractions.append(ground_truth_frac)
    averaged_data.append(adata_sample.X.mean(axis=0).tolist()[0])

averaged_data = pd.DataFrame(
    averaged_data,
    index=range(n_sample),
    columns=adata.var_names
)
# pseudobulk dataset
adata_pseudobulk = ad.AnnData(averaged_data)
adata_pseudobulk.layers["counts"] = adata_pseudobulk.X.copy()
# adata_pseudobulk.obsm["spatial"] = adata_pseudobulk.obsm["location"]
sc.pp.normalize_total(adata_pseudobulk, target_sum=10e4)
sc.pp.log1p(adata_pseudobulk)
adata_pseudobulk.raw = adata_pseudobulk
# filter genes to be the same on the pseudobulk data
intersect = np.intersect1d(adata_pseudobulk.var_names, adata_test.var_names)
adata_pseudobulk = adata_pseudobulk[:, intersect].copy()
adata_pseudobulk = adata_pseudobulk[:, intersect].copy()
G = len(intersect)

# %% ground truth cell type fractions
ground_truth_fractions = pd.DataFrame(ground_truth_fractions, index=range(n_sample))
ground_truth_fractions = ground_truth_fractions.fillna(
    0
)  # the Nan are cells not sampled

# intersection between all genes and marker genes
averaged_data = averaged_data[intersection]

# copy
adata_train = adata_train.copy()
adata_test = adata_test.copy()

# %%
# Create and train models

### %% 1. scVI
logger.info("Fit scVI ...")

# check if model is already saved
model_path = f"models/{DATASET}_scvi.pkl"
if os.path.exists(model_path.exists()):
    loger.info(f"Model fitted, saved in path:{model_path}, loading scVI...")
    scvi_model = scvi.model.scVI.load(model_path)
else:
  scvi.model.scVI.setup_anndata(
    adata,
    layer="counts",
    categorical_covariate_keys=["cell_type"],
    batch_index="batch_key", # no other cat covariate for now
    # continuous_covariate_keys=["percent_mito", "percent_ribo"],
  )
  scvi_model = scvi.model.scVI(adata)
  scvi_model.view_anndata_setup()
  scvi_model.train(max_epochs=300, batch_size=128, train_size=1.0, check_val_every_n_epoch=1)
  scvi_model.save(model_path)

#### %% 2. DestVI
logger.info("Fit DestVI ...")

# condscVI
model_path = f"models/{DATASET}_condscvi.pkl"
if os.path.exists(model_path.exists()):
    logger.info(f"Model fitted, saved in path:{model_path}, loading condscVI...")
    scvi_model = scvi.model.condSCVI.load(model_path)
else:
  scvi.mode.CondSCVI.setup_anndata(
      adata_train,
      layer="counts",
      labels_key="cell_type"
  )
  condscvi_model = CondSCVI(adata_test, weight_obs=False)
  condscvi_model.view_anndata_setup()
  condscvi_model.train()
  condscvi_model.save(model_path)
# DestVI
model_path = f"models/{DATASET}_destvi.pkl"
if os.path.exists(model_path.exists()):
    logger.info(f"Model fitted, saved in path:{model_path}, loading DestVI...")
    destvi_model = scvi.model.scVI.load(model_path)
else:
    scvi.model.DestVI.setup_anndata(
        adata_pseudobulk,
        layer="counts"
        )
    destvi_model = DestVI.from_rna_model(adata_pseudobulk, condscvi_model)
    destvi_model.view_anndata_setup()
    destvi_model.train(max_epochs=2500)

####

# logger.info("Train mixupVI ...")cond
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

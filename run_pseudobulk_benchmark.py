"""Pseudobulk benchmark."""
# %%
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from loguru import logger

from benchmark_utils import (
    preprocess_scrna,
    split_dataset,
    create_pseudobulk_dataset,
    create_latent_signature,
    fit_scvi,
    fit_destvi,
    fit_mixupvi,
    create_signature,
    add_cell_types_grouped,
    perform_nnls,
    perform_latent_deconv,
    compute_correlations,
    compute_group_correlations,
    plot_deconv_results,
)

# %% params
DATASET = "CTI"  # "TOY"
SIGNATURE_CHOICE = "crosstissue_general"  # ["laughney", "crosstissue_granular_updated"]
CELL_TYPE_GROUP = "primary_groups" #"updated_granular_groups"  # ["primary_groups", "precise_groups"]

# %% Load scRNAseq dataset
logger.info(f"Loading single-cell dataset: {DATASET} ...")
if DATASET == "TOY":
    adata = scvi.data.heart_cell_atlas_subsampled()
    preprocess_scrna(adata, keep_genes=1200)
elif DATASET == "CTI":
    adata = sc.read("/home/owkin/deepdeconv/data/cti_adata.h5ad")
    preprocess_scrna(adata,
                     keep_genes=2500,
                     batch_key="donor_id")
# elif DATASET == "CTI_PROCESSED":
# adata_processed = sc.read("/home/owkin/cti_data/processed/cti_processed.h5ad")
# adata = sc.read("/home/owkin/cti_data/processed/cti_processed_batch.h5ad")

# %% load signature
logger.info(f"Loading signature matrix: {SIGNATURE_CHOICE} | {CELL_TYPE_GROUP}...")
adata = add_cell_types_grouped(adata, CELL_TYPE_GROUP)
signature = create_signature(
    adata,
    signature_type=SIGNATURE_CHOICE,
    group=CELL_TYPE_GROUP
)

# %% split train/test
adata_train, adata_test = split_dataset(adata, stratify=CELL_TYPE_GROUP)

# %% Create pseudobulk dataset
logger.info("Creating pseudobulk dataset...")
adata_pseudobulk_train, proportions_train = create_pseudobulk_dataset(
    adata_train, cell_type_group=CELL_TYPE_GROUP
)
adata_pseudobulk_test, proportions_test = create_pseudobulk_dataset(
    adata_test, cell_type_group=CELL_TYPE_GROUP
)

# %% ground truth cell type fractions
df_proportions_train = pd.DataFrame(
    np.stack([proportions_train[i].values for i in range(len(proportions_train))]),
    index=adata_pseudobulk_train.obs_names,
    columns=list(proportions_train[0].index),
)
df_proportions_test = pd.DataFrame(
    np.stack([proportions_train[i].values for i in range(len(proportions_test))]),
    index=adata_pseudobulk_test.obs_names,
    columns=list(proportions_test[0].index),
)
df_proportions_train = df_proportions_train.fillna(0)  # the Nan are cells not sampled
df_proportions_test = df_proportions_test.fillna(0)  # the Nan are cells not sampled
# %%
# Create and train models
adata_train = adata_train.copy()
adata_test = adata_test.copy()

### %% 1. scVI
logger.info("Fit scVI ...")
model_path = f"models/{DATASET}_scvi.pkl"
scvi_model = fit_scvi(adata_train, model_path)

#### %% 2. DestVI
# logger.info("Fit DestVI ...")
# model_path_1 = f"models/{DATASET}_condscvi.pkl"
# model_path_2 = f"models/{DATASET}_destvi.pkl"
# condscvi_model , destvi_model= fit_destvi(adata_train,
#                                           adata_pseudobulk_train,
#                                           model_path_1,
#                                           model_path_2,
#                                           cell_type_key=CELL_TYPE_GROUP)

#### %% 3. MixupVI
logger.info("Train mixupVI ...")
model_path = f"models/{DATASET}_mixupvi.pkl"
mixupvi_model = fit_mixupvi(adata_train, model_path, cell_type_group=CELL_TYPE_GROUP)
#

# %% Deconvolution
df_test_correlations = pd.DataFrame(
    index=adata_pseudobulk_test.obs_names, columns=["scVI", "MixupVI", "NNLS"]
)  # "Random", "DestVI"]
df_test_group_correlations = pd.DataFrame(
    index=df_proportions_test.columns, columns=["scVI", "MixupVI", "NNLS"]
)
### % Random proportions
# TO DO:
#

### % Linear regression (NNLS)
deconv_results = perform_nnls(signature, adata_pseudobulk_test)
correlations = compute_correlations(deconv_results, df_proportions_test)
group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
df_test_correlations.loc[:, "NNLS"] = correlations.values
df_test_group_correlations.loc[:, "NNLS"] = group_correlations.values

### % scVI
adata_latent_signature = create_latent_signature(
    adata=adata_test,
    model=scvi_model,
    sc_per_pseudobulk=2000,
    cell_type_column=CELL_TYPE_GROUP,
)
deconv_results = perform_latent_deconv(
    adata_pseudobulk=adata_pseudobulk_test,
    adata_latent_signature=adata_latent_signature,
    model=scvi_model,
)
correlations = compute_correlations(deconv_results, df_proportions_test)
group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
df_test_correlations.loc[:, "scVI"] = correlations.values
df_test_group_correlations.loc[:, "scVI"] = group_correlations.values

### % MixupVI
adata_latent_signature = create_latent_signature(
    adata=adata_test,
    model=mixupvi_model,
    sc_per_pseudobulk=2000,
    cell_type_column=CELL_TYPE_GROUP,
)
deconv_results = perform_latent_deconv(
    adata_pseudobulk=adata_pseudobulk_test,
    adata_latent_signature=adata_latent_signature,
    model=mixupvi_model
)
correlations = compute_correlations(deconv_results, df_proportions_test)
group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
df_test_correlations.loc[:, "MixupVI"] = correlations.values
df_test_group_correlations.loc[:, "MixupVI"] = group_correlations.values

### % DestVI
# deconv_results = destvi_model.get_proportions(adata_pseudobulk_test, deterministic=True)
# correlations = compute_correlations(deconv_results, df_proportions_test)
# group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
# df_predicted_proportions.loc[:, "DestVI"] = deconv_results
# df_test_correlations.loc[:, "DestVI"] = correlations.values
# df_test_group_correlations.loc[:, "DestVI"] = group_correlations.values

### % Plots
plot_deconv_results(df_test_correlations, "test")
plot_deconv_results(df_test_group_correlations, "cell_type_test")

# %% [markdown]
# **scVI sanity checks on toy dataset**

# In this notebook, we go through several sanity checks to test if scVI has linear propreties


#  %%
# General imports
import os
import scanpy as sc
import scvi
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy

sc.set_figure_params(figsize=(4, 4))

# %%
## 1. Load toy dataset
dataset = "toy"

if dataset == "cti":
    adata = ad.read_h5ad("/home/owkin/deepdeconv/notebooks/data/adata_cti_5000.h5ad")
else:
    adata = scvi.data.heart_cell_atlas_subsampled()
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy()  # preserve counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # freeze the state in `.raw`
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=1200,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key="cell_source",
    )

# %%
## 2. Load trained dataset
dir_path = "/home/owkin/deepdeconv/notebooks/models/"
params = ["100", "200", "300", "400"]

models = {}

for param in params:
    model_name = f"{dataset}_{param}_epochs"
    model = scvi.model.SCVI.load(dir_path=os.path.join(dir_path, model_name),
                                adata=adata,
                                use_gpu=True
                                )
    models[param] = model

# %%
## 3. Linearity sanity checks

import tqdm
from scvi_sanity_checks_utils import sanity_checks_metrics_latent, sanity_checks_metrics_feature

if dataset == "toy":
    batch_size = [128, 256, 512, 1024, 2048] #, 4096] #, 8192
else:
    batch_size = [512, 1024, 2048, 4096, 8192, 16384]

latent_space_metrics = {}
feature_space_metrics = {}

# params = [str(x) for x in (100, 200, 300, 400)]
params = [str(x) for x in (100, 200, 300, 400)]

for param in tqdm.tqdm(params):
    # latent
    latent_space_metrics[param] = {}
    latent_metrics, latent_errors = sanity_checks_metrics_latent(models[param],
                                            adata,
                                            batch_sizes=batch_size,
                                            n_repeats=100,
                                            use_get_latent=True)
    latent_space_metrics[param]["corr"] = latent_metrics["corr"]
    latent_space_metrics[param]["error"] = latent_errors["corr"]
    # feature
    feature_space_metrics[param] = {}
    feature_metrics, feature_errors = sanity_checks_metrics_feature(models[param],
                                            adata,
                                            batch_sizes=batch_size,
                                            n_repeats=100,
                                            use_get_normalized=True)
    feature_space_metrics[param]["corr"] = feature_metrics["corr"]
    feature_space_metrics[param]["error"] = latent_errors["corr"]


# %%
## 4. plot

def plot_metrics(dict_metrics, params, title, type):
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))

    for param in params:
            plt.errorbar(batch_size,
                    dict_metrics[param]["corr"],
                    yerr=dict_metrics[param]["error"],
                    fmt='o-',
                    capsize=5,
                    linestyle="--", marker="+",
                    label=f"{param}_epochs")


    plt.legend()
    plt.xlabel("Batch size")
    plt.ylabel("Pearson correaltion")
    plt.xticks(batch_size)

    plt.title(title)
    plt.savefig(f"toy_linearity_check_{str(type)}.png")

plot_metrics(latent_space_metrics, params, "Sanity check 1: Interpolation in latent space", 1)
plot_metrics(feature_space_metrics, params, "Sanity check 2: Interpolation in feature space", 2)

"""scVI trainer debugger."""
# %%
import scanpy as sc
import scvi
import matplotlib.pyplot as plt

# %%
# load toy dataset
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
# Create and train model
adata = adata.copy()
scvi.model.MixUpVI.setup_anndata(
    adata,
    layer="counts",
    categorical_covariate_keys=["cell_type"],  # no other cat covariate for now
    # continuous_covariate_keys=["percent_mito", "percent_ribo"],
)
model = scvi.model.MixUpVI(adata, signature_type="post_encoded")
model.view_anndata_setup()
model.train(max_epochs=100, batch_size=512, validation_size=0.1,early_stopping=True)

print("ok")


# %%
# Plot coefficients
plt.plot(
    range(100),
    model.history["pearson_coeff_train"],
    label="Latent space Pearson coefficient",
)
plt.plot(
    range(100),
    model.history["pearson_coeff_deconv_train"],
    label="Deconv pearson coefficient",
)
plt.plot(
    range(100),
    model.history["cosine_similarity_train"],
    label="Deconv cosine similarity",
)
plt.legend()


# %%
# Plot train loss epoch
plt.plot(range(100), model.history["train_loss_epoch"], label="Train loss epoch")
plt.plot(range(100), model.history["elbo_train"], label="Train elbo")
plt.legend()

# %%

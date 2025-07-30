# %%
from benchmark_utils.pseudobulk_dataset_utils import prepare_mixupvi_v2_data
from benchmark_utils import preprocess_scrna, add_cell_types_grouped

import pickle as pkl
import numpy as np
import scanpy as sc
import scvi

with open("project/highest_r2_genes_FACS_1st_gran.pkl", "rb") as f:
    filtered_genes = pkl.load(f)

BASE_MODEL_PATH = "project/base_mixupvi_FACS_1st_gran"
MIXUPVI_V2_MODEL_PATH = "project/mixupvi_v2_FACS_extra_alignment_prior_bulk_deconvolution"

# %%
cti_adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
cti_adata = preprocess_scrna(cti_adata, keep_genes=None)
cell_type = f"cell_types_grouped_FACS_1st_level_granularity"
cti_adata, train_test_index = add_cell_types_grouped(cti_adata, "FACS_1st_level_granularity")
cti_adata.obs["cell_types_grouped"] = cti_adata.obs[cell_type]
cti_adata = cti_adata[train_test_index["Train index"]]
cti_adata = cti_adata[:, filtered_genes]
cti_adata.obs["source"] = "pseudobulk"
cti_adata.obsm["ground_truth"] = np.empty((len(cti_adata), 5))

# %%
base_model = scvi.model.MixUpVI.load(BASE_MODEL_PATH, adata=cti_adata.copy())
# %%
cti_adata.obsm["latent_sc"] = base_model.get_latent_representation(give_mean=True, return_dist=False)

# %%
final_adata = prepare_mixupvi_v2_data(cti_adata, base_model, n_pseudobulk_samples=200)

# %%
new_model = scvi.model.MixUpVI_v2.load(MIXUPVI_V2_MODEL_PATH, final_adata)

# %%
n_cells_per_type = 50 # Number of cells to sample from each type
cell_types = cti_adata.obs[cell_type].unique()

sampled_indices = []
for ct in cell_types:
    ct_indices = cti_adata.obs[cti_adata.obs[cell_type] == ct].index
    if len(ct_indices) >= n_cells_per_type:
        sampled = np.random.choice(ct_indices, n_cells_per_type, replace=False)
    else:
        # If not enough cells, sample with replacement
        sampled = np.random.choice(ct_indices, n_cells_per_type, replace=True)
    sampled_indices.extend(sampled)

small_sc_dataset = cti_adata[sampled_indices].copy()
# %%
latent_sc = new_model.get_latent_representation(small_sc_dataset)
latent_pseudobulk = new_model.get_latent_representation(final_adata[final_adata.obs["source"] == "pseudobulk"])
latent_bulk = new_model.get_latent_representation(final_adata[final_adata.obs["source"] == "bulk"])
latent_signature = new_model.module.latent_signature_matrix

# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt


all_latent = np.concatenate([
    latent_sc,
    latent_pseudobulk,
    latent_bulk,
    latent_signature,
])

labels = np.concatenate([
    small_sc_dataset.obs["cell_types_grouped"],
    np.array(["Pseudobulk"]*len(latent_pseudobulk)),
    np.array(["Bulk"]*len(latent_bulk)),
    final_adata.uns["bulk_cell_types_order"],
    #[f"Signature {type}" for type in bulk_adata.uns["cell_types_order"]],
])


pca = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=42)
umap = UMAP(n_components=2, random_state=42)

latent_2d_pca = pca.fit_transform(all_latent)
latent_2d_tsne = tsne.fit_transform(all_latent)
latent_2d_umap = umap.fit_transform(all_latent)

# %%
plt.figure(figsize=(24, 6), dpi=300)

# Create a color map for unique labels
unique_labels = np.unique(labels)
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
color_dict = dict(zip(unique_labels, colors))
point_colors = np.array([color_dict[label] for label in labels])

# Plot PCA
plt.subplot(131)
plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=point_colors, alpha=0.6)
plt.title('PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Plot t-SNE
plt.subplot(132)
plt.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], c=point_colors, alpha=0.6)
plt.title('t-SNE')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

# Plot UMAP
plt.subplot(133)
plt.scatter(latent_2d_umap[:, 0], latent_2d_umap[:, 1], c=point_colors, alpha=0.6)
plt.title('UMAP')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# Add legend
legend_elements = [plt.scatter([], [], c=[color_dict[label]], label=label) for label in unique_labels]
plt.figlegend(handles=legend_elements, labels=list(unique_labels), bbox_to_anchor=(1.05, 0.5), loc='center left')

plt.tight_layout()
plt.show()
# %%

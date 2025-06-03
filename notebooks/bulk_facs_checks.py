# %%
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import integrate, sparse
from tqdm import tqdm

from benchmark_utils import (
    add_cell_types_grouped,
    create_dirichlet_pseudobulk_dataset,
    create_uniform_pseudobulk_dataset,
    load_bulk_facs,
    load_bulk_facs_tpm,
    load_cti,
    preprocess_scrna,
)

# %%
# --- Load datasets ---
cti_data = load_cti(n_variable_genes=None)
cti_data = preprocess_scrna(cti_data["dataset"], keep_genes=None, batch_key="donor_id")
cti_data, _ = add_cell_types_grouped(cti_data, "FACS_1st_level_granularity")
single_cell_data = cti_data.to_df()

bulk_facs_data = load_bulk_facs()["dataset"].T
bulk_facs_tpm_data = load_bulk_facs_tpm()["dataset"].T

# %%
# --- Common genes across datasets ---
common_genes = list(
    set(bulk_facs_data.columns)
    & set(bulk_facs_tpm_data.columns)
    & set(single_cell_data.columns)
)


# --- Helper functions ---
def compute_variance(df, genes):
    """Compute the variance of the genes in the dataframe."""
    return pd.Series(np.var(df[genes].values, axis=0, ddof=1), index=genes)


def compute_umi_stats(df, genes):
    """Compute the UMI stats of the genes in the dataframe."""
    return df[genes].sum(axis=1)


def compute_sample_variance(df, genes):
    """Compute the sample variance of the genes in the dataframe."""
    return pd.Series(np.var(df[genes].values, axis=1, ddof=1), index=df.index)


def summarize_variance(series):
    """Summarize the variance of the genes in the dataframe."""
    return {"mean": series.mean(), "max": series.max(), "min": series.min()}


def summarize_umi(series):
    """Summarize the UMI stats of the genes in the dataframe."""
    return {
        "mean": series.mean(),
        "max": series.max(),
        "min": series.min(),
        "median": series.median(),
    }


def calculate_cumulative_overlaps(sorted_list1, sorted_list2):
    """Calculate the cumulative overlaps of the two lists."""
    seen_list1, seen_list2, matched = set(), set(), set()
    overlaps, n_genes = [], []
    for i in range(len(sorted_list1)):
        gene1 = sorted_list1[i]
        gene2 = sorted_list2[i]
        seen_list1.add(gene1)
        if gene1 in seen_list2:
            matched.add(gene1)
        seen_list2.add(gene2)
        if gene2 in seen_list1:
            matched.add(gene2)
        overlaps.append(len(matched))
        n_genes.append(i + 1)
    return overlaps, n_genes


# --- Plot overlap percentages ---
def percentages(overlaps, n_genes):
    """Calculate the percentages of the overlaps."""
    return [o / n for o, n in zip(overlaps, n_genes)]


# %%
# --- Pseudobulk generation settings ---
sampling_methods = ["uniform", "dirichlet"]
cell_numbers = [10000, 1024, 100]
aggregation_methods = ["sum", "mean"]

# %%
# --- Initialize data holders ---
pseudobulk_data = {}
variance_vectors = {}
umi_vectors = {}
sample_variance_vectors = {}

# %%
# --- Single-cell and Bulk calculations (cached) ---
variance_vectors["single_cell"] = compute_variance(single_cell_data, common_genes)
umi_vectors["single_cell"] = compute_umi_stats(single_cell_data, common_genes)
sample_variance_vectors["single_cell"] = compute_sample_variance(
    single_cell_data, common_genes
)

variance_vectors["bulk_facs"] = compute_variance(bulk_facs_data, common_genes)
umi_vectors["bulk_facs"] = compute_umi_stats(bulk_facs_data, common_genes)
sample_variance_vectors["bulk_facs"] = compute_sample_variance(
    bulk_facs_data, common_genes
)

variance_vectors["bulk_facs_tpm"] = compute_variance(bulk_facs_tpm_data, common_genes)
umi_vectors["bulk_facs_tpm"] = compute_umi_stats(bulk_facs_tpm_data, common_genes)
sample_variance_vectors["bulk_facs_tpm"] = compute_sample_variance(
    bulk_facs_tpm_data, common_genes
)

# %%
# --- Generate pseudobulk datasets ---
total_iterations = len(sampling_methods) * len(cell_numbers) * len(aggregation_methods)
with tqdm(total=total_iterations, desc="Generating pseudobulk datasets") as pbar:
    for sampling in sampling_methods:
        for n_cells in cell_numbers:
            for aggregation in aggregation_methods:
                key = f"{sampling}_pb_{n_cells}_{aggregation}"
                creator = (
                    create_uniform_pseudobulk_dataset
                    if sampling == "uniform"
                    else create_dirichlet_pseudobulk_dataset
                )
                pbar.set_description(f"Processing {key}")
                dataset = creator(
                    cti_data,
                    n_sample=206,
                    n_cells=n_cells,
                    cell_type_group="cell_types_grouped_FACS_1st_level_granularity",
                    aggregation_method=aggregation,
                )
                adata_df = dataset["adata_pseudobulk_test_counts"].to_df()
                pseudobulk_data[key] = adata_df
                variance_vectors[key] = compute_variance(adata_df, common_genes)
                umi_vectors[key] = compute_umi_stats(adata_df, common_genes)
                sample_variance_vectors[key] = compute_sample_variance(
                    adata_df, common_genes
                )
                pbar.update(1)


# %%
# --- Compute final statistics ---
variance_stats = {k: summarize_variance(v) for k, v in variance_vectors.items()}
umi_stats = {k: summarize_umi(v) for k, v in umi_vectors.items()}
sample_variance_stats = {
    k: summarize_variance(v) for k, v in sample_variance_vectors.items()
}

variance_stats_df = pd.DataFrame(variance_stats).T
umi_stats_df = pd.DataFrame(umi_stats).T
sample_variance_stats_df = pd.DataFrame(sample_variance_stats).T

# %%
# --- Output variance summary ---
print("\nVariance statistics:")
print(variance_stats_df.round(2))

# %%
# --- Output UMI summary ---
print("\nUMI statistics:")
print(umi_stats_df.round(2))

# %%
# --- Output Sample Variance summary ---
print("\nSample Variance statistics:")
print(sample_variance_stats_df.round(2))

# %%
# --- Create boxplots to compare UMI counts across all datasets ---
plt.figure(figsize=(16, 8), dpi=300)
boxplot_data = (
    [umi_vectors["single_cell"]]
    + [umi_vectors[k] for k in pseudobulk_data.keys()]
    + [umi_vectors["bulk_facs"], umi_vectors["bulk_facs_tpm"]]
)
labels = ["Single-cell"] + list(pseudobulk_data.keys()) + ["Bulk FACS", "Bulk FACS TPM"]
plt.boxplot(boxplot_data, labels=labels)
plt.yscale("log")
plt.xticks(rotation=90)
plt.ylabel("UMI Counts (log scale)")
plt.title("Distribution of UMI Counts Across Genes")
plt.tight_layout()
plt.show()

# %%
# --- Create boxplots to compare Sample Variances across all datasets ---
plt.figure(figsize=(16, 8), dpi=300)
boxplot_data = (
    [sample_variance_vectors["single_cell"]]
    + [sample_variance_vectors[k] for k in pseudobulk_data.keys()]
    + [sample_variance_vectors["bulk_facs"], sample_variance_vectors["bulk_facs_tpm"]]
)
labels = ["Single-cell"] + list(pseudobulk_data.keys()) + ["Bulk FACS", "Bulk FACS TPM"]
plt.boxplot(boxplot_data, labels=labels)
plt.yscale("log")
plt.xticks(rotation=90)
plt.ylabel("Sample Variance (log scale)")
plt.title("Distribution of Sample Variances Across Genes")
plt.tight_layout()
plt.show()

# %%
# --- Overlap analysis between top variant genes ---
# Prepare sorted indices for all datasets
sorted_variances = {
    k: v[common_genes].sort_values(ascending=False).index
    for k, v in variance_vectors.items()
}

# Calculate cumulative overlaps
overlap_results_bulk = {}
overlap_results_bulk_tpm = {}

for key in sorted_variances:
    if key in ["bulk_facs", "bulk_facs_tpm"]:
        continue
    overlap_bulk, _ = calculate_cumulative_overlaps(
        sorted_variances[key], sorted_variances["bulk_facs"]
    )
    overlap_bulk_tpm, _ = calculate_cumulative_overlaps(
        sorted_variances[key], sorted_variances["bulk_facs_tpm"]
    )
    overlap_results_bulk[key] = overlap_bulk
    overlap_results_bulk_tpm[key] = overlap_bulk_tpm

# Random overlaps
random_genes1 = common_genes.copy()
random_genes2 = common_genes.copy()
random.shuffle(random_genes1)
random.shuffle(random_genes2)
overlaps_random, n_genes_random = calculate_cumulative_overlaps(
    random_genes1, random_genes2
)

n_genes = list(range(1, len(common_genes) + 1))

# %%
# --- Plot overlaps ---
plt.figure(figsize=(20, 20), dpi=300)
for key, overlaps in overlap_results_bulk.items():
    plt.plot(n_genes, overlaps, label=f"{key} vs Bulk", alpha=0.7)
for key, overlaps in overlap_results_bulk_tpm.items():
    plt.plot(n_genes, overlaps, label=f"{key} vs Bulk TPM", linestyle="--", alpha=0.7)
plt.plot(
    n_genes_random, overlaps_random, label="Random overlap", linestyle=":", alpha=0.7
)
plt.plot(n_genes, n_genes, "--", label="Perfect overlap (x=y)", alpha=0.7)
plt.xlabel("Number of top variant genes considered")
plt.ylabel("Number of overlapping genes")
plt.title("Overlap with Bulk FACS and Bulk FACS TPM")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# --- Plot overlap percentages ---
plt.figure(figsize=(20, 20), dpi=300)
for key, overlaps in overlap_results_bulk.items():
    plt.plot(n_genes, percentages(overlaps, n_genes), label=f"{key} vs Bulk", alpha=0.7)
for key, overlaps in overlap_results_bulk_tpm.items():
    plt.plot(
        n_genes,
        percentages(overlaps, n_genes),
        label=f"{key} vs Bulk TPM",
        linestyle="--",
        alpha=0.7,
    )
plt.plot(
    n_genes_random,
    percentages(overlaps_random, n_genes_random),
    label="Random overlap",
    linestyle=":",
    alpha=0.7,
)
plt.hlines(
    y=1.0,
    xmin=0,
    xmax=len(n_genes),
    color="g",
    linestyle="--",
    alpha=0.8,
    label="Perfect overlap (100%)",
)
plt.vlines(x=0, ymin=0, ymax=1, color="g", linestyle="--", alpha=0.8)
plt.xlabel("Number of top variant genes considered")
plt.ylabel("Percentage of overlapping genes")
plt.title("Overlap percentage with Bulk FACS and Bulk FACS TPM")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# --- AUC Calculations ---
x_norm = [x / max(n_genes) for x in n_genes]
x_norm_random = [x / max(n_genes_random) for x in n_genes_random]

print("\nArea Under Curve (AUC) Results:")
for key, overlaps in overlap_results_bulk.items():
    auc = integrate.trapz(percentages(overlaps, n_genes), x_norm)
    print(f"{key} vs Bulk FACS: {auc:.4f}")
for key, overlaps in overlap_results_bulk_tpm.items():
    auc = integrate.trapz(percentages(overlaps, n_genes), x_norm)
    print(f"{key} vs Bulk FACS TPM: {auc:.4f}")
auc_random = integrate.trapz(
    percentages(overlaps_random, n_genes_random), x_norm_random
)
print(f"Random overlap: {auc_random:.3f}")

# %%
# --- Plot overlap percentages in 0 to 5000 range ---
plt.figure(figsize=(20, 20), dpi=300)
for key, overlaps in overlap_results_bulk.items():
    plt.plot(n_genes, percentages(overlaps, n_genes), label=f"{key} vs Bulk", alpha=0.7)
for key, overlaps in overlap_results_bulk_tpm.items():
    plt.plot(
        n_genes,
        percentages(overlaps, n_genes),
        label=f"{key} vs Bulk TPM",
        linestyle="--",
        alpha=0.7,
    )
plt.plot(
    n_genes_random,
    percentages(overlaps_random, n_genes_random),
    label="Random overlap",
    linestyle=":",
    alpha=0.7,
)
plt.hlines(
    y=1.0,
    xmin=0,
    xmax=len(n_genes),
    color="g",
    linestyle="--",
    alpha=0.8,
    label="Perfect overlap (100%)",
)
plt.vlines(x=0, ymin=0, ymax=1, color="g", linestyle="--", alpha=0.8)
plt.xlim(0, 5000)
plt.ylim(0, 1)
plt.xlabel("Number of top variant genes considered")
plt.ylabel("Percentage of overlapping genes")
plt.title("Overlap percentage with Bulk FACS and Bulk FACS TPM")
plt.grid(True)
plt.legend()

# %%
# --- AUC Calculations in 0 to 5000 range ---
x_norm = [x / max(n_genes[0:5000]) for x in n_genes[0:5000]]
x_norm_random = [x / max(n_genes_random[0:5000]) for x in n_genes_random[0:5000]]

print("\nArea Under Curve (AUC) Results:")
for key, overlaps in overlap_results_bulk.items():
    auc = integrate.trapz(percentages(overlaps[0:5000], n_genes[0:5000]), x_norm)
    print(f"{key} vs Bulk FACS: {auc:.4f}")
for key, overlaps in overlap_results_bulk_tpm.items():
    auc = integrate.trapz(percentages(overlaps[0:5000], n_genes[0:5000]), x_norm)
    print(f"{key} vs Bulk FACS TPM: {auc:.4f}")
auc_random = integrate.trapz(
    percentages(overlaps_random[0:5000], n_genes_random[0:5000]), x_norm_random
)


# %%
# --- Computing coefficient of variation (CV) for each gene ---
# Using numpy operations for better performance
cv_sc = single_cell_data[common_genes].values.std(axis=0) / single_cell_data[
    common_genes
].values.mean(axis=0)
cv_bulk = bulk_facs_data[common_genes].values.std(axis=0) / bulk_facs_data[
    common_genes
].values.mean(axis=0)
cv_bulk_tpm = bulk_facs_tpm_data[common_genes].values.std(axis=0) / bulk_facs_tpm_data[
    common_genes
].values.mean(axis=0)

# Sort genes by CV
cv_sorted_sc = pd.Series(cv_sc, index=common_genes).sort_values()
cv_sorted_bulk = pd.Series(cv_bulk, index=common_genes).sort_values()
cv_sorted_bulk_tpm = pd.Series(cv_bulk_tpm, index=common_genes).sort_values()

print("\nTop 10 genes by coefficient of variation (low to high):")
print("\nSingle cell:")
print(cv_sorted_sc.head(10))
print("\nBulk FACS:")
print(cv_sorted_bulk.head(10))
print("\nBulk FACS TPM:")
print(cv_sorted_bulk_tpm.head(10))

# %%
# -- Computing coefficient of variation for pseudobulk datasets ---
cv_pseudobulk = {}
for key in tqdm(pseudobulk_data):
    cv_pseudobulk[key] = pseudobulk_data[key][common_genes].values.std(
        axis=0
    ) / pseudobulk_data[key][common_genes].values.mean(axis=0)

cv_pseudobulk_sorted = {
    k: pd.Series(v, index=common_genes).sort_values() for k, v in cv_pseudobulk.items()
}
# %%
# --- Calculating cumulative overlaps between CV-sorted genes ---
overlap_sc_bulk, n_genes_sc_bulk = calculate_cumulative_overlaps(
    cv_sorted_sc.index, cv_sorted_bulk.index
)
overlap_sc_bulk_tpm, n_genes_sc_bulk_tpm = calculate_cumulative_overlaps(
    cv_sorted_sc.index, cv_sorted_bulk_tpm.index
)

overlap_pseudobulk_list = []
overlap_pseudobulk_tpm_list = []
n_genes_pseudobulk_list = []
n_genes_pseudobulk_tpm_list = []

for key in cv_pseudobulk_sorted:
    overlap_pseudobulk, n_genes_pseudobulk = calculate_cumulative_overlaps(
        cv_pseudobulk_sorted[key].index, cv_sorted_bulk.index
    )
    overlap_pseudobulk_tpm, n_genes_pseudobulk_tpm = calculate_cumulative_overlaps(
        cv_pseudobulk_sorted[key].index, cv_sorted_bulk_tpm.index
    )
    overlap_pseudobulk_list.append(overlap_pseudobulk)
    overlap_pseudobulk_tpm_list.append(overlap_pseudobulk_tpm)
    n_genes_pseudobulk_list.append(n_genes_pseudobulk)
    n_genes_pseudobulk_tpm_list.append(n_genes_pseudobulk_tpm)


# Plot percentage overlap curve for CV-sorted genes
plt.figure(figsize=(10, 10), dpi=300)
plt.plot(
    n_genes_sc_bulk,
    percentages(overlap_sc_bulk, n_genes_sc_bulk),
    label="Single Cell vs Bulk FACS",
)
plt.plot(
    n_genes_sc_bulk_tpm,
    percentages(overlap_sc_bulk_tpm, n_genes_sc_bulk_tpm),
    label="Single Cell vs Bulk FACS TPM",
)
for i, key in enumerate(cv_pseudobulk_sorted):
    plt.plot(
        n_genes_pseudobulk_list[i],
        percentages(overlap_pseudobulk_list[i], n_genes_pseudobulk_list[i]),
        label=f"{key} vs Bulk FACS",
    )
    plt.plot(
        n_genes_pseudobulk_tpm_list[i],
        percentages(overlap_pseudobulk_tpm_list[i], n_genes_pseudobulk_tpm_list[i]),
        label=f"{key} vs Bulk FACS TPM",
        linestyle="--",
    )

plt.hlines(
    y=1.0,
    xmin=0,
    xmax=len(n_genes_sc_bulk),
    color="g",
    linestyle="--",
    alpha=0.8,
    label="Perfect overlap (100%)",
)
plt.vlines(x=0, ymin=0, ymax=1, color="g", linestyle="--", alpha=0.8)

plt.xlabel("Number of genes")
plt.ylabel("Overlap percentage")
plt.title("Overlap percentage with CV-sorted genes")
plt.grid(True)
plt.legend()

# %%
# --- Create interactive plot for CV-sorted gene overlaps ---
fig = go.Figure()

# Add single cell vs bulk traces
fig.add_trace(
    go.Scatter(
        x=n_genes_sc_bulk,
        y=percentages(overlap_sc_bulk, n_genes_sc_bulk),
        name="Single Cell vs Bulk FACS",
        mode="lines",
        hovertemplate="Single Cell vs Bulk FACS<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=n_genes_sc_bulk_tpm,
        y=percentages(overlap_sc_bulk_tpm, n_genes_sc_bulk_tpm),
        name="Single Cell vs Bulk FACS TPM",
        mode="lines",
        hovertemplate="Single Cell vs Bulk FACS TPM<extra></extra>",
    )
)

# Add pseudobulk traces
for i, key in enumerate(cv_pseudobulk_sorted):
    fig.add_trace(
        go.Scatter(
            x=n_genes_pseudobulk_list[i],
            y=percentages(overlap_pseudobulk_list[i], n_genes_pseudobulk_list[i]),
            name=f"{key} vs Bulk FACS",
            mode="lines",
            hovertemplate=f"{key} vs Bulk FACS<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=n_genes_pseudobulk_tpm_list[i],
            y=percentages(
                overlap_pseudobulk_tpm_list[i], n_genes_pseudobulk_tpm_list[i]
            ),
            name=f"{key} vs Bulk FACS TPM",
            line={"dash": "dash"},
            mode="lines",
            hovertemplate=f"{key} vs Bulk FACS TPM<extra></extra>",
        )
    )

# Add perfect overlap line
fig.add_trace(
    go.Scatter(
        x=[0, len(n_genes_sc_bulk)],
        y=[1.0, 1.0],
        name="Perfect overlap (100%)",
        line={"color": "green", "dash": "dash"},
        opacity=0.8,
        mode="lines",
        hovertemplate="Perfect overlap (100%)<extra></extra>",
    )
)

# Update layout
fig.update_layout(
    title="Overlap percentage with CV-sorted genes (Interactive)",
    xaxis_title="Number of genes",
    yaxis_title="Overlap percentage",
    showlegend=True,
    width=800,
    height=800,
    hovermode="closest",
)

# Add grid
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

fig.show()

# %%
# Save first 2000 sorted genes for dirichlet pseudobulks with 100 cells and sum aggregation
dirichlet_100_sum_genes = list(
    cv_pseudobulk_sorted["dirichlet_pb_100_sum"][:2000].index
)

# Save to file
# with open('project/filtered_genes_low_CV.pkl', 'wb') as f:
#     pickle.dump(dirichlet_100_sum_genes, f)

# %%
# --- Calculate Between/Within Cell Type Variance Ratio ---
# Calculating sigma²_between and sigma²_within as described in the metric
# sigma²_within: variance of genes inside cell types
# sigma²_between: variance between cell types


def between_within_fraction(adata, gene_list, cell_type_key):
    """Fast fraction of variance explained by cell-type (ρ) for each gene.

    ρ_g = σ²_between / (σ²_between + σ²_within)

    Parameters
    ----------
    adata : AnnData
        Must contain normalised / log-transformed expression in adata.X
    gene_list : sequence of str
        Gene names (must be in adata.var_names)
    cell_type_key : str
        Column in adata.obs with cell-type labels

    Returns
    -------
    pandas.Series (index = gene_list, values = ρ)
    """
    # ---------- 1. pull out the expression block ----------
    X = adata[:, gene_list].X  # cells × genes
    n_cells, n_genes = X.shape

    # ---------- 2. encode cell types as integers 0..K-1 ----------
    group_codes, groups = pd.factorize(
        adata.obs[cell_type_key].values, sort=True
    )  # reproducible order
    K = len(groups)
    n_k = np.bincount(group_codes)  # (K,)

    # ---------- 3. build a sparse 1-hot matrix: cells × K ----------
    # row i, col k is 1 if cell i belongs to group k
    indptr = np.arange(n_cells + 1, dtype=np.intp)
    indicator = sparse.csr_matrix(
        (np.ones(n_cells, dtype=np.float32), group_codes, indptr), shape=(n_cells, K)
    )

    # ---------- 4. per-group means and global mean ----------
    group_sums = indicator.T @ X  # (K × genes)
    group_means = group_sums / n_k[:, None]  # (K × genes)
    global_mean = np.asarray(X.mean(axis=0)).ravel()  # (genes,)

    # ---------- 5. between-group variance ----------
    mu_diff = group_means - global_mean  # broadcast
    ss_between = (n_k[:, None] * mu_diff**2).sum(axis=0)
    var_between = ss_between / (K - 1)

    # ---------- 6. within-group variance -------------
    # sum of squares inside each group: Σ(x²) - n_k * μ_k²
    # first term:
    group_sq_sums = indicator.T @ X.multiply(X)  # (K × genes)
    ss_within = (group_sq_sums - n_k[:, None] * group_means**2).sum(axis=0)
    var_within = ss_within / (n_cells - K)

    # ---------- 7. fraction of variance explained ----
    rho = var_between / (var_between + var_within + 1e-8)  # ε avoids 0/0

    return pd.Series(rho, index=gene_list)


# Calculate the metric for single-cell data
print("Calculating between/within variance ratio for single-cell data...")
bw_variance_ratio_sc = between_within_fraction(
    cti_data,
    common_genes,
    cell_type_key="cell_types_grouped_FACS_1st_level_granularity",
)

# Sort genes by the between/within variance ratio (decreasing) for single-cell data
bw_sorted_genes_sc = bw_variance_ratio_sc.sort_values(ascending=False)

# Print top genes according to this metric
print("\nTop 50 genes with highest between/within variance ratio (single-cell):")
print(bw_sorted_genes_sc.head(50))

# %%
# --- Save top genes based on between/within variance ratio ---
# Save top 2000 genes sorted by between/within variance ratio for single-cell data
bw_top_genes_sc = list(bw_sorted_genes_sc.index[:2000])

# Save to file
with open("project/filtered_genes_bw_ratio_sc.pkl", "wb") as f:
    pickle.dump(bw_top_genes_sc, f)
    print("Saved top genes to project/filtered_genes_bw_ratio_sc.pkl")

# %%

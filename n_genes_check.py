"""Run MixUpVI experiments with the right sanity checks."""

# %%

import scanpy as sc
from loguru import logger

from benchmark_utils import (
    preprocess_scrna,
)
from constants import (
    N_GENES,
    TRAINING_CELL_TYPE_GROUP,
    TRAINING_DATASET,
)

# %% Load scRNAseq dataset
logger.info(f"Loading single-cell dataset: {TRAINING_DATASET} ...")

if TRAINING_DATASET == "CTI":
    adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
    # preprocess_scrna(adata, keep_genes=N_GENES, batch_key="donor_id")
    cell_type = f"cell_types_grouped_{TRAINING_CELL_TYPE_GROUP}"
else:
    raise ValueError(f"Invalid training dataset: {TRAINING_DATASET}")


# %%
# Check if gene selection methods give same result
logger.info("Checking gene selection methods...")

# Method 1: Direct selection of N_GENES
adata_direct = adata.copy()
preprocess_scrna(adata_direct, keep_genes=N_GENES)

direct_ranks = adata_direct.var["highly_variable_rank"].sort_values().tolist()
direct_ranks_indexes = (
    adata_direct.var["highly_variable_rank"].sort_values().index.tolist()
)

# %%
# Method 2: Select 10k then filter to N_GENES by rank
adata_ranked = adata.copy()
preprocess_scrna(adata_ranked, keep_genes=10000)

ranks = adata_ranked.var["highly_variable_rank"].sort_values().tolist()
ranks_indexes = adata_ranked.var["highly_variable_rank"].sort_values().index.tolist()

# %%

# Compare results
dir_filter = set(direct_ranks_indexes[:4000])
normal_filter = set(ranks_indexes[:4000])
genes_match = dir_filter == normal_filter
logger.info(f"Gene sets match: {genes_match}")
if not genes_match:
    n_different = len(set(normal_filter).symmetric_difference(set(dir_filter)))
    logger.info(f"Number of different genes: {n_different}")

from benchmark_utils.training_utils import fit_mixupvi_v2
from benchmark_utils import preprocess_scrna, add_cell_types_grouped
import scanpy as sc
import pickle as pkl

from loguru import logger

with open("project/highest_r2_genes_FACS_1st_gran.pkl", "rb") as f:
    filtered_genes = pkl.load(f)

logger.info("Loading data...")
cti_adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
cti_adata = preprocess_scrna(cti_adata, keep_genes=None)
cell_type = f"cell_types_grouped_FACS_1st_level_granularity"
cti_adata, train_test_index = add_cell_types_grouped(cti_adata, "FACS_1st_level_granularity")
cti_adata.obs["cell_types_grouped"] = cti_adata.obs[cell_type]
cti_adata = cti_adata[train_test_index["Train index"]]
cti_adata = cti_adata[:, filtered_genes]

logger.info("Fitting whole pipeline...")
new_model = fit_mixupvi_v2(cti_adata, "project/base_mixupvi_FACS_1st_gran_batch_v2", "project/mixupvi_v2_FACS_1st_gran_mixup_base_batch_extra_alignment_v2", cell_type_group="cell_types_grouped", save_model=True)

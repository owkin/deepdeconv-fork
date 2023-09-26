## Imports
import scanpy as sc
import scvi
import anndata as ad
import pandas as pd

MODEL_SAVE = False
PATH = "/home/owkin/project/scvi_models/models/cti_linear_test"


## Cross-immune 
adata = sc.read("/home/owkin/data/cross-tissue/omics/raw/local.h5ad")
adata.layers["counts"] = adata.raw.X.copy() 
adata.X = adata.raw.X.copy() # copy counts


## Preprocess
sc.pp.filter_genes(adata, min_counts=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata  # freeze the state in `.raw`
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=5000,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
    batch_key="assay",
)


## Train test split
train_test_index_matrix_common = pd.read_csv("/home/owkin/project/train_test_index_matrix_common.csv", index_col=1)
adata_train = adata[train_test_index_matrix_common["Train index"]]
adata_test = adata[train_test_index_matrix_common["Test index"]]


## Train scVI
scvi.model.SCVI.setup_anndata(
    adata_train,
    layer="counts",
    categorical_covariate_keys=["assay", "donor_id"],
    # continuous_covariate_keys=["percent_mito", "percent_ribo"],
)
model = scvi.model.SCVI(adata_train)
model.train(max_epochs=100)


## Save model
if MODEL_SAVE:
    model.save(PATH)


import anndata as ad
import pandas as pd

from .pseudobulk_dataset_utils import create_dirichlet_pseudobulk_dataset_v2
from .signature_utils import create_signature
from .latent_signature_utils import create_latent_signature
from .load_dataset_utils import load_bulk_facs
from .deconv_utils import use_nnls_method

from constants import N_CELLS_PER_PSEUDOBULK

def prepare_mixupvi_v2_data_prior_deconvolution(
    adata: ad.AnnData,
    base_model,
    n_pseudobulk_samples: int = 10000,
    n_cells_per_pseudobulk: int = None,
    seed: int = None,
):
    if n_cells_per_pseudobulk is None:
        n_cells_per_pseudobulk = N_CELLS_PER_PSEUDOBULK
    
    # Add latent representation to single-cell data
    adata.obsm["latent_sc"] = base_model.get_latent_representation(give_mean=True, return_dist=False)
    
    # Create pseudobulk dataset
    adata_pb = create_dirichlet_pseudobulk_dataset_v2(
        adata, 
        n_sample=n_pseudobulk_samples, 
        n_cells=n_cells_per_pseudobulk,
        seed=seed
    )
    genes = adata_pb["adata_pseudobulk_counts"].var_names.tolist()
    
    # Load and process bulk data
    bulk_dataset = load_bulk_facs()
    adata_bulk = bulk_dataset["dataset"]
    adata_bulk = adata_bulk.loc[genes]
    adata_bulk = adata_bulk.T
    
    adata_bulk = ad.AnnData(
        X=adata_bulk.values,
        var=adata_pb["adata_pseudobulk_counts"].var,
    )
    
    # Process ground truth proportions
    bulk_dataset["ground_truth"] = bulk_dataset["ground_truth"] / 100
    bulk_dataset["ground_truth"] = bulk_dataset["ground_truth"].div(
        bulk_dataset["ground_truth"].sum(axis=1), axis=0
    )
    bulk_dataset["ground_truth"] = bulk_dataset["ground_truth"][adata_pb["df_proportions"].columns]
    
    signature_matrix = create_signature(signature_type="FACS_1st_level_granularity")
    bulk_to_deconvolve = adata_bulk.to_df().T
    # Add latent representation for bulk data coming from a prior deconvolution
    prior_deconvolution = use_nnls_method(bulk_to_deconvolve, signature_matrix)
    latent_signature_matrix = create_latent_signature(
        adata,
        model=base_model,
        use_mixupvi=False,
        average_all_cells=True,
    )
    adata_bulk.obsm["latent_sc"] = prior_deconvolution.values @ latent_signature_matrix.X
    
    adata_bulk.obsm["ground_truth"] = bulk_dataset["ground_truth"].values
    adata_bulk.uns["cell_types_order"] = bulk_dataset["ground_truth"].columns.tolist()
    
    # Add metadata flags
    adata_bulk.obs["has_latent"] = False
    adata_pb["adata_pseudobulk_counts"].obs["has_latent"] = True
    adata_pb["adata_pseudobulk_counts"].obsm["ground_truth"] = adata_pb["df_proportions"].values
    adata_pb["adata_pseudobulk_counts"].uns["cell_types_order"] = adata_pb["df_proportions"].columns
    
    # Concatenate pseudobulk and bulk data
    final_adata = ad.concat(
        [adata_pb["adata_pseudobulk_counts"], adata_bulk],
        join="outer",
        merge="same",
        label="source",
        keys=["pseudobulk", "bulk"],
    )

    final_adata.uns["bulk_cell_types_order"] = adata_bulk.uns["cell_types_order"]
    return final_adata

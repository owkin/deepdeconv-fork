"""Utilities to create latent signatures from scvi-like (deep generative) models."""
from typing import Optional, Tupple
import scvi
import anndata as ad
import numpy as np
import pandas as pd
import random

def create_anndata_pseudobulk(adata: ad.AnnData,
                              x: np.array) -> ad.AnnData:
    """Creates an anndata object from a pseudobulk sample.

    Parameters
    ----------
    adata: ad.AnnData
        AnnData aobject storing training set
    x: np.array
        pseudobulk sample

    Return
    ------
    ad.AnnData
        Anndata object storing the pseudobulk array
    """
    df_obs = pd.DataFrame.from_dict([{col: adata.obs[col].value_counts().index[0] for col in adata.obs.columns}])
    adata_pseudobulk = ad.AnnData(X=x, obs=df_obs)
    adata_pseudobulk.layers["counts"] = np.copy(x)

    return adata_pseudobulk

def create_latent_signature(
    adata: ad.AnnData,
    sc_per_pseudobulk: int,
    repeats: int = 1,
    signature_type: str = "pre-encoded",
    cell_type_column: str = "cell_type",
    count_key: Optional[str] = "counts",
    representation_key: Optional[str] = "X_scvi",
    model: Optional[scvi.model.SCVI] = None,
) -> ad.AnnData:
    """Make cell type representations from a single cell dataset represented with scvi.


    From an annotated single cell dataset (adata), for each cell type, (found in the
    cell_type column of obs in adata), we create "repeats" representation in the
    following way.

    - We sample sc_per_pseudobulk single cells of the desired cell type with replacement
    - We then create the corresponding cell type representation, in one of the
    two following ways.
    - Option 1)
        If we choose to aggregate before embedding (aggregate_before_embedding flag),
        we construct a pseudobulk of these single cells (all of the same cell type)
        forming a "pure" pseudobulk of the given cell type.
        We then take the scvi model (model) latent representation of this purified
        pseudobulk.
    - Option 2)
        If we choose to aggregate after embedding, we get the corresponding
        embeddings from the adata.obsm[(representation_key)] field of the ann data,
        (This assumes that we have encoded the ann data with scvi) and then average them.

    We then output an AnnData object, containing all these representations
    (n_repeats representation per cell type), whose obs field contains the repeat number
    in the "repeat" column, and the cell type in the "cell type" column.


    Parameters
    ----------
    adata: ad.AnnData
        The single cell dataset, with a cell_type_column, and a representation_key in
        the obsm if one wants to aggregate after embedding.
    sc_per_pseudobulk: int
        The number of single cells used to construct the purified pseudobulks.
    repeats: int
        The number of representations computed randomly for a given cell type.
    aggregate_before_embedding: bool
        Perform the aggregation (average) before embedding the cell-type specific
        pseudobulk. If false, we aggregate the representations.
    cell_type_column: str
        The field of the ann data obs containing the cell type partition of interest.
    count_key: Optional[str]
        The layer containing counts, mandatory if aggregating before embedding.
    representation_key: Optional[str]
        The field of obsm containing the pre-computed scvi representation, used only
        if aggregation takes place after representation.
    model: Optional[scvi.model.SCVI]
        The trained scvi model, used only if aggregation is before representation.

    Returns
    -------
    ad.AnnData
        The output annotated dataset of cell type specific representations,
        containing "repeats" random examples.

    """
    cell_type_list = []
    representation_list: list[np.ndarray] = []
    repeat_list = []
    for cell_type in adata.obs[cell_type_column].unique():
        for repeat in range(repeats):
            seed = random.seed()
            # Sample cells
            sampled_cells = (
                adata.obs[adata.obs[cell_type_column] == cell_type]
                .sample(n=sc_per_pseudobulk, random_state=seed, replace=True)
                .index
            )
            adata_sampled = adata[sampled_cells]

            if signature_type == "pre-encoded":
                assert (
                    model is not None,
                    "If representing a purified pseudo bulk (aggregate before embedding",
                    "), must give a model",
                )
                assert (
                    count_key is not None
                ), "Must give a count key if aggregating before embedding."

                pseudobulk = (
                    adata_sampled.layers[count_key].mean(axis=0).reshape(1, -1)
                )  # .astype(int).astype(numpy.float32)
                adata_pseudobulk = create_anndata_pseudobulk(adata_sampled, pseudobulk)
                result = model.get_latent_representation(adata_pseudobulk).reshape(-1)
            else:
                raise ValueError("Only pre-encoded signatures are supported for now.")
            repeat_list.append(repeat)
            representation_list.append(result)
            cell_type_list.append(cell_type)
    full_rpz = np.stack(representation_list, axis=0)
    obs = pd.DataFrame(pd.Series(cell_type_list, name="cell type"))
    obs["repeat"] = repeat_list
    adata_signature = ad.AnnData(X=full_rpz, obs=obs)
    return adata_signature.X.T, list(adata_signature.obs["cell type"])

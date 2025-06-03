"""Utilities to create latent signatures from scvi-like (deep generative) models."""
import random
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

import scvi
from constants import SIGNATURE_TYPE

from .pseudobulk_dataset_utils import create_anndata_pseudobulk


def create_latent_signature(
    adata: ad.AnnData,
    use_mixupvi: bool = False,
    repeats: int = 1,
    average_all_cells: bool = True,
    sc_per_pseudobulk: int = 3000,
    cell_type_column: str = "cell_types_grouped",
    count_key: Optional[str] = "counts",
    representation_key: Optional[str] = "X_scvi",
    model: Optional[Union[scvi.model.SCVI, scvi.model.MixUpVI]] = None,
) -> ad.AnnData:
    """Make cell type representations from a single cell dataset represented with scvi.

    From an annotated single cell dataset (adata), for each cell type, (found in the
    cell_type column of obs in adata), we create "repeats" representation in the
    following way.

    - We sample sc_per_pseudobulk single cells of the desired cell type with replacement
      or all cells of the given cell type if average_all_cells == True.
    - We then create the corresponding cell type representation, in one of the
    two following ways.
    - Option 1)
        If we choose to aggregate before embedding (aggregate_before_embedding flag),
        we construct a pseudobulk of these single cells (all of the same cell type)
        forming a "pure" pseudobulk of the given cell type.
        We then take the model latent representation of this purified pseudobulk.
    - Option 2)
        If we choose to aggregate after embedding, we get the corresponding
        embeddings from the adata.obsm[(representation_key)] field of the ann data,
        (This assumes that we have encoded the ann data with scvi) and then average them.

    The aggregation method (mean or sum) is determined by the model's pseudo_bulk_aggregation
    attribute if present, defaulting to mean otherwise.

    Parameters
    ----------
    adata: ad.AnnData
        The single cell dataset, with a cell_type_column, and a representation_key in
        the obsm if one wants to aggregate after embedding.
    use_mixupvi: bool
        Whether to use mixupvi model.
    average_all_cells: bool
        If True, then average all cells per given cell type. sc_per_pseudobulk will
        not be used.
    sc_per_pseudobulk: int
        The number of single cells used to construct the purified pseudobulks.
        Won't be used if average_all_cells = True.
    repeats: int
        The number of representations computed randomly for a given cell type. If
        average_all_cells is True, all repeats will be the same.
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

    # Determine aggregation method based on model's attribute if available
    aggregation_method = "mean"
    if model is not None and hasattr(model.module, "pseudo_bulk_aggregation"):
        aggregation_method = model.module.pseudo_bulk_aggregation

    with torch.no_grad():
        for cell_type in adata.obs[cell_type_column].unique():
            for repeat in range(repeats):
                # Sample cells
                sampled_cells = adata.obs[adata.obs[cell_type_column] == cell_type]
                if average_all_cells:
                    adata_sampled = adata[sampled_cells.index]
                else:
                    seed = random.seed()
                    sampled_cells = sampled_cells.sample(
                        n=sc_per_pseudobulk, random_state=seed, replace=True
                    ).index
                    adata_sampled = adata[sampled_cells]

                assert model is not None, (
                    "If representing a purified pseudo bulk (aggregate before embedding), "
                    "must give a model"
                )
                assert (
                    count_key is not None
                ), "Must give a count key if aggregating before embedding."

                if use_mixupvi:
                    # TODO: in this case, n_cells sampled will be equal to self.n_cells_per_pseudobulk by mixupvae
                    # so change that to being equal to either all cells (if average_all_cells) or sc_per_pseudobulk
                    result = model.get_latent_representation(
                        adata_sampled, get_pseudobulk=True
                    )[
                        0
                    ]  # take first pseudobulk
                else:
                    if SIGNATURE_TYPE == "pre_encoded":
                        # Use appropriate aggregation method based on the model's configuration
                        if aggregation_method == "sum":
                            pseudobulk = (
                                adata_sampled.layers[count_key]
                                .sum(axis=0)
                                .reshape(1, -1)
                            )
                        else:  # default to mean
                            pseudobulk = (
                                adata_sampled.layers[count_key]
                                .mean(axis=0)
                                .reshape(1, -1)
                            )
                        adata_sampled = create_anndata_pseudobulk(
                            adata_sampled.obs, adata_sampled.var_names, pseudobulk
                        )
                    result = model.get_latent_representation(
                        adata_sampled,
                    )
                    if SIGNATURE_TYPE == "pre_encoded":
                        result = result.reshape(-1)
                    elif SIGNATURE_TYPE == "post_inference":
                        if aggregation_method == "sum":
                            result = result.sum(axis=0)
                        else:  # default to mean
                            result = result.mean(axis=0)
                repeat_list.append(repeat)
                representation_list.append(result)
                cell_type_list.append(cell_type)
    full_rpz = np.stack(representation_list, axis=0)
    obs = pd.DataFrame(pd.Series(cell_type_list, name="cell type"))
    obs["repeat"] = repeat_list
    adata_signature = ad.AnnData(X=full_rpz, obs=obs)
    return adata_signature


def create_latent_signature_pca(
    adata: ad.AnnData,
    model: PCA = None,
    average_all_cells: bool = True,
    sc_per_pseudobulk: int = 3000,
    repeats: int = 1,
    cell_type_column: str = "cell_types_grouped",
    count_key: Optional[str] = "counts",
    representation_key: Optional[str] = "X_scvi",
) -> ad.AnnData:
    """Create a latent signature from a single cell dataset using PCA."""
    cell_type_list = []
    representation_list: list[np.ndarray] = []
    repeat_list = []

    for cell_type in adata.obs[cell_type_column].unique():
        for repeat in range(repeats):
            # Sample cells
            sampled_cells = adata.obs[adata.obs[cell_type_column] == cell_type]
            if average_all_cells:
                adata_sampled = adata[sampled_cells.index]
            else:
                seed = random.seed()
                sampled_cells = sampled_cells.sample(
                    n=sc_per_pseudobulk, random_state=seed, replace=True
                ).index
                adata_sampled = adata[sampled_cells]

            if SIGNATURE_TYPE == "pre_encoded":
                pseudobulk = adata_sampled.layers[count_key].mean(axis=0).reshape(1, -1)
                adata_sampled = create_anndata_pseudobulk(
                    adata_sampled.obs, adata_sampled.var_names, pseudobulk
                )

            result = model.transform(adata_sampled.X.toarray())

            if SIGNATURE_TYPE == "pre_encoded":
                result = result.reshape(-1)
            elif SIGNATURE_TYPE == "post_inference":
                result = result.mean(axis=0)

            repeat_list.append(repeat)
            representation_list.append(result)
            cell_type_list.append(cell_type)

    full_rpz = np.stack(representation_list, axis=0)
    obs = pd.DataFrame(pd.Series(cell_type_list, name="cell type"))
    obs["repeat"] = repeat_list
    adata_signature = ad.AnnData(X=full_rpz, obs=obs)
    return adata_signature

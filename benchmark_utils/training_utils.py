"""Utilities for training deep generative models"""

import os
from typing import List, Tuple

import anndata as ad
from loguru import logger

import ray
import scvi
import numpy as np
from constants import (
    BATCH_SIZE,
    CAT_COV,
    CHECK_VAL_EVERY_N_EPOCH,
    CONT_COV,
    DISPERSION,
    ENCODE_COVARIATES,
    GENE_LIKELIHOOD,
    LATENT_SIZE,
    LOSS_COMPUTATION,
    MAX_EPOCHS,
    MIXUP_PENALTY,
    N_CELLS_PER_PSEUDOBULK,
    N_HIDDEN,
    N_PSEUDOBULKS,
    PSEUDO_BULK,
    PSEUDO_BULK_AGGREGATION,
    SEED,
    SIGNATURE_TYPE,
    TRAIN_SIZE,
    USE_BATCH_NORM,
)
from scvi import autotune
from tuning_configs import TUNED_VARIABLES

from .training_callbacks import LatentSpaceVisualizationCallback
from .tuning_utils import format_and_save_tuning_results
from .pseudobulk_dataset_utils import prepare_mixupvi_v2_data
from .appendix_utils import prepare_mixupvi_v2_data_prior_deconvolution
from .load_dataset_utils import load_bulk_facs
from .latent_signature_utils import create_latent_signature
from ._init_utils import transfer_weights_selective



def tune_mixupvi(
    adata: ad.AnnData,
    cell_type_group: str,
    search_space: dict,
    metric: str,
    additional_metrics: list[str],
    num_samples: int,
    training_dataset: str,
):
    """Tune the MixUpVI model.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to tune the MixUpVI model on
    cell_type_group : str
        The cell type group to use for the MixUpVI model
    search_space : dict
        The search space to use for the MixUpVI model
    metric : str
        The metric to use for the MixUpVI model
    additional_metrics : list[str]
        The additional metrics to use for the MixUpVI model
    num_samples : int
        The number of samples to use for the MixUpVI model
    training_dataset : str
        The training dataset to use for the MixUpVI model
    """
    mixupvi_model = scvi.model.MixUpVI
    mixupvi_model.setup_anndata(
        adata,
        layer="counts",
        labels_key=cell_type_group,
        categorical_covariate_keys=CAT_COV,
        continuous_covariate_keys=CONT_COV,
    )
    scvi_tuner = autotune.ModelTuner(mixupvi_model)
    # scvi_tuner.info() # to look at all the HP/metrics tunable
    ray.init(log_to_driver=False)
    tuning_results = scvi_tuner.fit(
        adata,
        metric=metric,
        additional_metrics=additional_metrics,
        search_space=search_space,
        num_samples=num_samples,  # will randomly num_samples samples (with replacement) among the HP cominations specified
        max_epochs=MAX_EPOCHS,
        resources={"cpu": 6, "gpu": 0.5},
    )

    all_results, best_hp, tuning_path, search_path = format_and_save_tuning_results(
        tuning_results,
        variables=TUNED_VARIABLES,
        training_dataset=training_dataset,
        cat_cov=CAT_COV,
        cont_cov=CONT_COV,
    )

    return all_results, best_hp, tuning_path, search_path


def fit_mixupvi(
    adata: ad.AnnData,
    model_path: str,
    cell_type_group: str,
    save_model: bool = True,
    cat_cov: List[str] = CAT_COV,
    cont_cov: List[str] = CONT_COV,
    encode_covariates: bool = ENCODE_COVARIATES,
    latent_space_visualizer: bool = False,
):
    """Fit the MixUpVI model.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to fit the MixUpVI model on
    model_path : str
        The path to save the MixUpVI model
    cell_type_group : str
        The cell type group to use for the MixUpVI model
    save_model : bool
        Whether to save the MixUpVI model
    cat_cov : List[str]
        The categorical covariates to use for the MixUpVI model
    cont_cov : List[str]
        The continuous covariates to use for the MixUpVI model
    encode_covariates : bool
        Whether to encode the covariates
    """
    if os.path.exists(model_path):
        logger.info(f"Model fitted, saved in path:{model_path}, loading MixupVI...")
        mixupvi_model = scvi.model.MixUpVI.load(model_path, adata)
    else:
        scvi.model.MixUpVI.setup_anndata(
            adata,
            layer="counts",
            labels_key=cell_type_group,
            batch_key=None,
            categorical_covariate_keys=cat_cov,
            continuous_covariate_keys=cont_cov,
        )
        mixupvi_model = scvi.model.MixUpVI(
            adata,
            seed=SEED,
            n_pseudobulks=N_PSEUDOBULKS,
            n_cells_per_pseudobulk=N_CELLS_PER_PSEUDOBULK,
            n_latent=LATENT_SIZE,
            n_hidden=N_HIDDEN,
            use_batch_norm=USE_BATCH_NORM,
            signature_type=SIGNATURE_TYPE,
            loss_computation=LOSS_COMPUTATION,
            pseudo_bulk=PSEUDO_BULK,
            pseudo_bulk_aggregation=PSEUDO_BULK_AGGREGATION,
            encode_covariates=encode_covariates,
            mixup_penalty=MIXUP_PENALTY,
            dispersion=DISPERSION,
            gene_likelihood=GENE_LIKELIHOOD,
        )
        mixupvi_model.view_anndata_setup()

        latent_space_visualizer_callback = []
        if latent_space_visualizer:
            latent_space_visualizer_callback = [LatentSpaceVisualizationCallback(adata, cell_type_group, visualization_frequency=10, path_to_save_figures=model_path)]

        mixupvi_model.train(
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            train_size=TRAIN_SIZE,
            check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
            callbacks=latent_space_visualizer_callback,
        )
        if save_model:
            mixupvi_model.save(model_path)

    return mixupvi_model


def fit_mixupvi_v2(
    adata: ad.AnnData,
    base_model_path: str,
    model_path: str,
    cell_type_group: str,
    save_model: bool = True,
):
    """Fit the MixUpVI_v2 model.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to fit the MixUpVI_v2 model on, this has to be single-cell 
    base_model_path : str
        The path to save the base model
    model_path : str
        The path to save the MixUpVI_v2 model
    cell_type_group : str
        The cell type group to use for the MixUpVI_v2 model
    save_model : bool
        Whether to save the MixUpVI_v2 model
    """

    if os.path.exists(base_model_path):
        logger.info(f"Base model fitted, saved in path:{base_model_path}, loading MixUpVI...")
        base_model = scvi.model.MixUpVI.load(base_model_path, adata.copy())
    else:
        logger.info(f"Base model not fitted, fitting MixUpVI...")
        #Here we can interchangebly use the scvi model or the mixupvi model as base model
        base_model = fit_mixupvi(adata.copy(), base_model_path, cell_type_group=cell_type_group, save_model=save_model)

    # Prepare data for MixUpVI_v2 using the dedicated function
    final_adata = prepare_mixupvi_v2_data_prior_deconvolution(
        adata=adata,
        base_model=base_model,
        n_pseudobulk_samples=10000,
        n_cells_per_pseudobulk=N_CELLS_PER_PSEUDOBULK,
        seed=SEED,
    )

    if os.path.exists(model_path):
        logger.info(f"Model fitted, saved in path:{model_path}, loading MixUpVI_v2...")
        mixupvi_v2_model = scvi.model.MixUpVI_v2.load(model_path, final_adata)
    else:

        latent_signature_matrix = create_latent_signature(
            adata,
            model=base_model,
            use_mixupvi=False,
            average_all_cells=True,
        )

        ordered_indices = [np.where(latent_signature_matrix.obs["cell type"] == cell_type)[0][0] for cell_type in final_adata.uns["bulk_cell_types_order"]]
        ordered_latent_signature = latent_signature_matrix.X[ordered_indices]

        # We need to add the batch effect here when using it in the MixUpVI_v2 model
        scvi.model.MixUpVI_v2.setup_anndata(
            final_adata,
            batch_key="source",
        )

        # Filter out any extra params that MixUpVI_v2 doesn't need
        mixupvi_params = {
            k: v for k, v in base_model.init_params_["non_kwargs"].items() 
            if k in scvi.model.MixUpVI_v2.__init__.__code__.co_varnames
        }
        model_kwargs = {
            k: v for k, v in base_model.init_params_["kwargs"]["model_kwargs"].items()
            if k in scvi.model.MixUpVI_v2.__init__.__code__.co_varnames
        }
        
        # We need to add the batch effect also here when using it in the MixUpVI_v2 model
        mixupvi_params["dispersion"] = "gene-batch"
        # mixupvi_params["gene_likelihood"] = "nb"

        mixupvi_v2_model = scvi.model.MixUpVI_v2(
            final_adata,
            **mixupvi_params,
            **model_kwargs,
            latent_signature_matrix=ordered_latent_signature,
        )
        mixupvi_v2_model.view_anndata_setup()

        transfer_weights_selective(base_model, mixupvi_v2_model)

        mixupvi_v2_model.train(
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            train_size=TRAIN_SIZE,
            check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
        )
        
        if save_model:
            mixupvi_v2_model.save(model_path)

    return mixupvi_v2_model


def fit_scvi(
    adata: ad.AnnData,
    model_path: str,
    save_model: bool = True,
) -> scvi.model.SCVI:
    """Fit scVI model to single-cell RNA data."""
    if os.path.exists(model_path):
        logger.info(f"Model fitted, saved in path:{model_path}, loading scVI...")
        scvi_model = scvi.model.SCVI.load(model_path, adata)
    else:
        scvi.model.SCVI.setup_anndata(
            adata,
            layer="counts",
            categorical_covariate_keys=CAT_COV,
            continuous_covariate_keys=CONT_COV,
        )
        scvi_model = scvi.model.SCVI(
            adata,
            n_latent=LATENT_SIZE,
            n_hidden=N_HIDDEN,
            use_batch_norm=USE_BATCH_NORM,
            dispersion=DISPERSION,
            gene_likelihood=GENE_LIKELIHOOD,
        )
        scvi_model.view_anndata_setup()
        scvi_model.train(
            max_epochs=MAX_EPOCHS,
            batch_size=128,
            train_size=TRAIN_SIZE,
        )
        if save_model:
            scvi_model.save(model_path)

    return scvi_model


def fit_destvi(
    adata: ad.AnnData,
    adata_pseudobulk: ad.AnnData,
    model_path_1: str,
    model_path_2: str,
    cell_type_group: str = "cell_types_grouped",
    save_model: bool = True,
) -> Tuple[scvi.model.CondSCVI, scvi.model.DestVI]:
    """Fit CondSCVI and DestVI model to paired single-cell/pseudoulk datasets."""
    # condscVI
    if os.path.exists(model_path_1):
        logger.info(f"Model fitted, saved in path:{model_path_1}, loading condscVI...")
        condscvi_model = scvi.model.CondSCVI.load(model_path_1, adata)
    else:
        scvi.model.CondSCVI.setup_anndata(
            adata, layer="counts", labels_key=cell_type_group
        )
        condscvi_model = scvi.model.CondSCVI(adata, weight_obs=False)
        condscvi_model.view_anndata_setup()
        condscvi_model.train(max_epochs=MAX_EPOCHS, train_size=TRAIN_SIZE)
        if save_model:
            condscvi_model.save(model_path_1)
    # DestVI
    if os.path.exists(model_path_2):
        logger.info(f"Model fitted, saved in path:{model_path_2}, loading DestVI...")
        destvi_model = scvi.model.DestVI.load(model_path_2, adata_pseudobulk)
    else:
        scvi.model.DestVI.setup_anndata(adata_pseudobulk, layer="counts")
        destvi_model = scvi.model.DestVI.from_rna_model(
            adata_pseudobulk, condscvi_model
        )
        destvi_model.view_anndata_setup()
        destvi_model.train(max_epochs=2500)
        if save_model:
            destvi_model.save(model_path_2)

    return condscvi_model, destvi_model

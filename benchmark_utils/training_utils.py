"""Utilities for training deep generative models"""

from loguru import logger
import anndata as ad
import scvi
import ray
from scvi import autotune
import os

from typing import Tuple, List
from .tuning_utils import format_and_save_tuning_results

from tuning_configs import TUNED_VARIABLES
from constants import (
    MAX_EPOCHS,
    BATCH_SIZE,
    LATENT_SIZE,
    N_PSEUDOBULKS,
    N_CELLS_PER_PSEUDOBULK,
    TRAIN_SIZE,
    CHECK_VAL_EVERY_N_EPOCH,
    CONT_COV,
    CAT_COV,
    ENCODE_COVARIATES,
    SIGNATURE_TYPE,
    USE_BATCH_NORM,
    LOSS_COMPUTATION,
    PSEUDO_BULK,
    MIXUP_PENALTY,
    DISPERSION,
    GENE_LIKELIHOOD,
    MIXUP_PENATLY_AGGREGATION,
    AVERAGE_VARIABLES_MIXUP_PENALTY,
    SEED,
)

def tune_mixupvi(adata: ad.AnnData,
                  cell_type_group: str,
                  search_space: dict,
                  metric: str,
                  additional_metrics: list[str],
                  num_samples: int,
                  training_dataset: str,
):
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
        num_samples=num_samples, # will randomly num_samples samples (with replacement) among the HP cominations specified
        max_epochs=MAX_EPOCHS,
        resources={"cpu": 10, "gpu": 0.5},
    )

    all_results, best_hp, tuning_path, search_path = format_and_save_tuning_results(
        tuning_results, variables=TUNED_VARIABLES, training_dataset=training_dataset,
    )

    return all_results, best_hp, tuning_path, search_path

def fit_mixupvi(adata: ad.AnnData,
                model_path: str,
                cell_type_group: str,
                save_model: bool = True,
                cat_cov: List[str] = CAT_COV,
                cont_cov: List[str] = CONT_COV,
                encode_covariates: bool = ENCODE_COVARIATES,
):
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
                use_batch_norm=USE_BATCH_NORM,
                signature_type=SIGNATURE_TYPE,
                loss_computation=LOSS_COMPUTATION,
                pseudo_bulk=PSEUDO_BULK,
                encode_covariates=encode_covariates,
                mixup_penalty=MIXUP_PENALTY,
                dispersion=DISPERSION,
                gene_likelihood=GENE_LIKELIHOOD,
                mixup_penalty_aggregation=MIXUP_PENATLY_AGGREGATION,
                average_variables_mixup_penalty=AVERAGE_VARIABLES_MIXUP_PENALTY,
            )
            mixupvi_model.view_anndata_setup()
            mixupvi_model.train(
                max_epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                train_size=TRAIN_SIZE,
                check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
            )
            if save_model:
                mixupvi_model.save(model_path)

    return mixupvi_model

def fit_scvi(adata: ad.AnnData,
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
            scvi_model = scvi.model.SCVI(adata)
            scvi_model.view_anndata_setup()
            scvi_model.train(max_epochs=MAX_EPOCHS,
                             batch_size=128,
                             train_size=TRAIN_SIZE,
                            )
            if save_model:
                scvi_model.save(model_path)

    return scvi_model

def fit_destvi(adata: ad.AnnData,
              adata_pseudobulk: ad.AnnData,
              model_path_1: str,
              model_path_2: str,
              cell_type_key: str = "cell_types_grouped",
              save_model: bool = True,
              ) -> Tuple[scvi.model.CondSCVI, scvi.model.DestVI]:
  """Fit CondSCVI and DestVI model to paired single-cell/pseudoulk datasets."""
  # condscVI
  if os.path.exists(model_path_1):
        logger.info(f"Model fitted, saved in path:{model_path_1}, loading condscVI...")
        condscvi_model = scvi.model.CondSCVI.load(model_path_1, adata)
  else:
        scvi.model.CondSCVI.setup_anndata(
            adata,
            layer="counts",
            labels_key=cell_type_key
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
        scvi.model.DestVI.setup_anndata(
            adata_pseudobulk,
            layer="counts"
            )
        destvi_model = scvi.model.DestVI.from_rna_model(adata_pseudobulk, condscvi_model)
        destvi_model.view_anndata_setup()
        destvi_model.train(max_epochs=2500)
        if save_model:
            destvi_model.save(model_path_2)

  return condscvi_model, destvi_model




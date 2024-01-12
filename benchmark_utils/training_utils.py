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
    N_LATENT,
    TRAIN_SIZE,
    CHECK_VAL_EVERY_N_EPOCH,
    # BENCHMARK_CELL_TYPE_GROUP,
    # CONT_COV,
    ENCODE_COVARIATES,
    ENCODE_CONT_COVARIATES,
    SIGNATURE_TYPE,
    USE_BATCH_NORM,
    LOSS_COMPUTATION,
    PSEUDO_BULK,
    MIXUP_PENALTY,
    DISPERSION,
    GENE_LIKELIHOOD,
)

def tune_mixupvi(adata: ad.AnnData,
                  cell_type_group: str,
                  search_space: dict,
                  metric: str,
                  additional_metrics: list[str],
                  num_samples: int,
                  training_dataset: str,
):
    CAT_COV = [cell_type_group]
    ## check missing for cell_group = BENCHMARK_CELL_TYPE_GROUP, cont_cov=CONT_COV,
    mixupvi_model = scvi.model.MixUpVI
    mixupvi_model.setup_anndata(
        adata,
        layer="counts",
        categorical_covariate_keys=CAT_COV,  # only cell types for now
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
        tuning_results, variable=TUNED_VARIABLES[0], training_dataset=training_dataset,
    )

    return all_results, best_hp, tuning_path, search_path

def fit_mixupvi(adata: ad.AnnData,
                model_path: str,
                cell_type_group: str,
                save_model: bool = True,
):
    if os.path.exists(model_path):
            logger.info(f"Model fitted, saved in path:{model_path}, loading MixupVI...")
            mixupvi_model = scvi.model.MixUpVI.load(model_path, adata)
    else:
            CAT_COV = [cell_type_group]
            ## check missing for cell_group = BENCHMARK_CELL_TYPE_GROUP, cont_cov=CONT_COV,
            scvi.model.MixUpVI.setup_anndata(
                adata,
                layer="counts",
                categorical_covariate_keys=CAT_COV,  # only cell types for now
            )
            mixupvi_model = scvi.model.MixUpVI(
                adata,
                n_latent=N_LATENT,
                use_batch_norm=USE_BATCH_NORM,
                signature_type=SIGNATURE_TYPE,
                loss_computation=LOSS_COMPUTATION,
                pseudo_bulk=PSEUDO_BULK,
                encode_covariates=ENCODE_COVARIATES,  # always False for now, because cat covariates is only cell types
                encode_cont_covariates=ENCODE_CONT_COVARIATES,  # if want to encode continuous covariates
                mixup_penalty=MIXUP_PENALTY,
                dispersion=DISPERSION,
                gene_likelihood=GENE_LIKELIHOOD,
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
             batch_key: List[str] = ["donor_id"],
             ) -> scvi.model.SCVI:

    """Fit scVI model to single-cell RNA data."""
    if os.path.exists(model_path):
            logger.info(f"Model fitted, saved in path:{model_path}, loading scVI...")
            scvi_model = scvi.model.SCVI.load(model_path, adata)
    else:
            scvi.model.SCVI.setup_anndata(
            adata,
            layer="counts",
            categorical_covariate_keys=batch_key
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




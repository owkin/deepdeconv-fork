"""Utilities for training deep generative models"""

from loguru import logger
import anndata as ad
import scvi
import os

from typing import Optional, Tuple
from .sanity_checks_utils import run_categorical_value_checks, run_incompatible_value_checks

from constants import (
    MAX_EPOCHS,
    BATCH_SIZE,
    TRAIN_SIZE,
    BENCHMARK_CELL_TYPE_GROUP,
    CONT_COV,
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

def fit_mixupvi(adata: ad.AnnData,
                model_path: str,
                cell_type_group: str,

):
    if os.path.exists(model_path):
            logger.info(f"Model fitted, saved in path:{model_path}, loading MixupVI...")
            mixupvi_model = scvi.model.MixUpVI.load(model_path, adata)
    else:
            CAT_COV = [cell_type_group]
            run_categorical_value_checks(
                cell_group=BENCHMARK_CELL_TYPE_GROUP, # cell_type_group,
                cat_cov=CAT_COV, # for now, only works with cell groups as categorical covariate
                cont_cov=CONT_COV,
                encode_covariates=ENCODE_COVARIATES,
                encode_cont_covariates=ENCODE_CONT_COVARIATES,
                use_batch_norm=USE_BATCH_NORM,
                signature_type=SIGNATURE_TYPE,
                loss_computation=LOSS_COMPUTATION,
                pseudo_bulk=PSEUDO_BULK,
                mixup_penalty=MIXUP_PENALTY,
                dispersion=DISPERSION,
                gene_likelihood=GENE_LIKELIHOOD,
            )
            run_incompatible_value_checks(
                pseudo_bulk=PSEUDO_BULK,
                loss_computation=LOSS_COMPUTATION,
                use_batch_norm=USE_BATCH_NORM,
                mixup_penalty=MIXUP_PENALTY,
                gene_likelihood=GENE_LIKELIHOOD,
            )
            scvi.model.MixUpVI.setup_anndata(
                adata,
                layer="counts",
                categorical_covariate_keys=CAT_COV,  # only cell types for now
            )
            mixupvi_model = scvi.model.MixUpVI(
                adata,
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
                train_size=TRAIN_SIZE
            )
            mixupvi_model.save(model_path)

    return mixupvi_model

def fit_scvi(adata: ad.AnnData,
             model_path: str,
             batch_key: Optional[str] = "batch_key"
             ) -> scvi.model.SCVI:

    """Fit scVI model to single-cell RNA data."""
    if os.path.exists(model_path):
            logger.info(f"Model fitted, saved in path:{model_path}, loading scVI...")
            scvi_model = scvi.model.SCVI.load(model_path, adata)
    else:
            scvi.model.SCVI.setup_anndata(
            adata,
            layer="counts",
            # categorical_covariate_keys=["cell_type"],
            # batch_index="batch_key", # no other cat covariate for now
            # continuous_covariate_keys=["percent_mito", "percent_ribo"],
            )
            scvi_model = scvi.model.SCVI(adata)
            scvi_model.view_anndata_setup()
            scvi_model.train(max_epochs=MAX_EPOCHS, batch_size=128, train_size=TRAIN_SIZE)
            scvi_model.save(model_path)

    return scvi_model

def fit_destvi(adata: ad.AnnData,
              adata_pseudobulk: ad.AnnData,
              model_path_1: str,
              model_path_2: str,
              cell_type_key: str = "cell_types_grouped",
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
        destvi_model.train(max_epochs=MAX_EPOCHS)

  return condscvi_model, destvi_model




"""Utilities for training deep generative models"""

from loguru import logger
import anndata as ad
import scvi
import os

from typing import Optional, Tuple
from .sanity_check_utils import check_model
# mixupVI hyperparameters
# N_EPOCHS = 300


def fit_mixupvi(adata: ad.AnnData,
                model_path: str,
):



def fit_scvi(adata: ad.AnnData,
             model_path: str,
             batch_key: Optional[str] = "batch_key"
             ) -> scvi.model.scVI:
  """Fit scVI model to single-cell RNA data."""
  if os.path.exists(model_path.exists()):
      logger.info(f"Model fitted, saved in path:{model_path}, loading scVI...")
      scvi_model = scvi.model.scVI.load(model_path)
  else:
    scvi.model.scVI.setup_anndata(
      adata,
      layer="counts",
      # categorical_covariate_keys=["cell_type"],
      # batch_index="batch_key", # no other cat covariate for now
      # continuous_covariate_keys=["percent_mito", "percent_ribo"],
    )
    scvi_model = scvi.model.scVI(adata)
    scvi_model.view_anndata_setup()
    scvi_model.train(max_epochs=300, batch_size=128, train_size=1.0, check_val_every_n_epoch=5)
    scvi_model.save(model_path)

    return scvi_model

def fit_destvi(adata: ad.AnnData,
              adata_pseudobulk: ad.AnnData,
              model_path_1: str,
              model_path_2: str,
              ) -> Tuple[scvi.model.CondSCVI, scvi.model.DestVI]:
  """Fit CondSCVI and DestVI model to paired single-cell/pseudoulk datasets."""
  # condscVI
  if os.path.exists(model_path_1):
      logger.info(f"Model fitted, saved in path:{model_path_1}, loading condscVI...")
      condscvi_model = scvi.model.condSCVI.load(model_path_1)
  else:
    scvi.mode.CondSCVI.setup_anndata(
        adata,s
        layer="counts",
        labels_key="cell_type"
    )
    condscvi_model = scvi.model.CondSCVI(adata, weight_obs=False)
    condscvi_model.view_anndata_setup()
    condscvi_model.train()
    condscvi_model.save(model_path_1)
  # DestVI
  if os.path.exists(model_path_2.exists()):
      logger.info(f"Model fitted, saved in path:{model_path_2}, loading DestVI...")
      destvi_model = scvi.model.scVI.load(model_path_2)
  else:
      scvi.model.DestVI.setup_anndata(
          adata_pseudobulk,
          layer="counts"
          )
      destvi_model = scvi.model.DestVI.from_rna_model(adata_pseudobulk, condscvi_model)
      destvi_model.view_anndata_setup()
      destvi_model.train(max_epochs=2500)

  return condscvi_model, destvi_model




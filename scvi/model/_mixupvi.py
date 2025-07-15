import logging
from typing import List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import torch
from anndata import AnnData

from scvi.dataloaders import MixUpDataSplitter, MixUpV2DataSplitter
from scvi import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.module import MixUpVAE, MixUpVAE_v2
from scvi.utils import setup_anndata_dsp
from scvi.data import AnnDataManager
from scvi.data._utils import _get_adata_minify_type
from scvi.model.utils import get_minified_adata_scrna
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)

from ._scvi import SCVI

logger = logging.getLogger(__name__)


class MixUpVI(SCVI):
    """single-cell Variational Inference with linearity constraint within batches.

    The linearity constraint is inspired by the MixUp method
    (https://arxiv.org/abs/1710.09412v2).
    """

    _module_cls = MixUpVAE

    def train(
        self,
        max_epochs: Tunable[Optional[int]] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        accelerator: str = "auto",
        devices: Union[int, List[int], str] = "auto",
        train_size: Tunable[float] = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        batch_size: Tunable[int] = 128,
        early_stopping: Tunable[bool] = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        self._data_splitter_cls = MixUpDataSplitter
        super().train(
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            accelerator=accelerator,
            devices=devices,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            early_stopping=early_stopping,
            plan_kwargs=plan_kwargs,
            **trainer_kwargs,
        )

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        get_pseudobulk: bool = False,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
        return_dist: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return the latent representation for each cell.

        This is typically denoted as :math:`z_n`.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        get_pseudobulk
            Give the pseudobulk latent representation instead of the single cell batch representation.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_dist
            Return (mean, variance) of distributions instead of just the mean.
            If `True`, ignores `give_mean` and `mc_samples`. In the case of the latter,
            `mc_samples` is used to compute the mean of a transformed distribution.
            If `return_dist` is true the untransformed mean and variance are returned.

        Returns
        -------
        Low-dimensional representation for each cell or a tuple containing its mean and variance.
        """
        self._check_if_trained(warn=True) #Here the warining is equal to True so that it does not raise an error if we initialize the model while training (needed for the latent space visualizer)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=adata.n_obs
        )
        latent = []
        latent_qzm = []
        latent_qzv = []
        for tensors in scdl:
            tensors = {key: value.to(self.device) for key, value in tensors.items()} # TODO: This is purposely done for the latent space visualizer, we should find a better way to do this
            inference_inputs = self.module._get_inference_input(tensors)
            inference_inputs = {key: value.to(self.device) if value is not None else value for key, value in inference_inputs.items()}
            outputs = self.module.inference(**inference_inputs)
            suffix = ""
            if get_pseudobulk:
                suffix = "_pseudobulk"
            if "qz" in outputs:
                qz = outputs[f"qz{suffix}"]
            else:
                qz_m, qz_v = outputs["qz_m"], outputs["qz_v"]
                qz = torch.distributions.Normal(qz_m, qz_v.sqrt())
            if give_mean:
                # does each model need to have this latent distribution param?
                if self.module.latent_distribution == "ln":
                    samples = qz.sample([mc_samples])
                    z = torch.nn.functional.softmax(samples, dim=-1)
                    z = z.mean(dim=0)
                else:
                    z = qz.loc
            else:
                z = outputs[f"z{suffix}"]

            latent += [z.cpu()]
            latent_qzm += [qz.loc.cpu()]
            latent_qzv += [qz.scale.square().cpu()]
        return (
            (torch.cat(latent_qzm).numpy(), torch.cat(latent_qzv).numpy())
            if return_dist
            else torch.cat(latent).numpy()
        )



class MixUpVI_v2(SCVI):
    """This is just a wrapper around the MixUpVAE_v2 module. The structure is the same as the one for SCVI.
    """

    _module_cls = MixUpVAE_v2


    def train(
        self,
        max_epochs: Tunable[Optional[int]] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        accelerator: str = "auto",
        devices: Union[int, List[int], str] = "auto",
        train_size: Tunable[float] = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        batch_size: Tunable[int] = 128,
        early_stopping: Tunable[bool] = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        self._data_splitter_cls = MixUpV2DataSplitter
        super().train(
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            accelerator=accelerator,
            devices=devices,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            early_stopping=early_stopping,
            plan_kwargs=plan_kwargs,
            **trainer_kwargs,
        )


    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        # TODO: ADD some checks to make sure that the latent_sc and ground_truth are in the adata.obsm
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
            ObsmField(
                "latent_sc", "latent_sc",
            ),
            ObsmField(
                "ground_truth", "ground_truth",
            ),
        ]
        # register new fields if the adata is minified
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
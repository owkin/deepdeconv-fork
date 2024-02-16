import logging
from typing import List, Optional, Union, Sequence, Tuple

import torch
import numpy as np
from anndata import AnnData

from ._scvi import SCVI
from scvi.module import MixUpVAE
from scvi.autotune._types import Tunable

logger = logging.getLogger(__name__)


class MixUpVI(SCVI):
    """single-cell Variational Inference with linearity constraint within batches.

    The linearity constraint is inspired by the MixUp method
    (https://arxiv.org/abs/1710.09412v2).
    """

    _module_cls = MixUpVAE

    def train(self,
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
        **trainer_kwargs,):
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
        get_pseudobulk: bool = True,
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
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=adata.n_obs
        )
        latent = []
        latent_qzm = []
        latent_qzv = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
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


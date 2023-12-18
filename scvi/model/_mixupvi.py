import logging
from typing import List, Optional, Union

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

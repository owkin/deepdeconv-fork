import logging

from ._scvi import SCVI
from scvi.module import MixUpVAE

logger = logging.getLogger(__name__)


class MixUpVI(SCVI):
    """single-cell Variational Inference with linearity constraint within batches.

    The linearity constraint is inspired by the MixUp method
    (https://arxiv.org/abs/1710.09412v2).
    """

    _module_cls = MixUpVAE

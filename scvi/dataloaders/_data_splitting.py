import logging
from math import ceil, floor
from typing import Dict, List, Optional, Union
import random

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._utils import get_anndata_attribute
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from scvi.dataloaders._semi_dataloader import SemiSupervisedDataLoader
from scvi.model._utils import parse_device_args
from scvi.utils._docstrings import devices_dsp

logger = logging.getLogger(__name__)

def validate_data_split(
    n_samples: int, train_size: float, validation_size: Optional[float] = None
):
    """Check data splitting parameters and return n_train and n_val.

    Parameters
    ----------
    n_samples
        Number of samples to split
    train_size
        Size of train set. Need to be: 0 < train_size <= 1.
    validation_size
        Size of validation set. Need to be 0 <= validation_size < 1
    """
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

    n_train = ceil(train_size * n_samples)

    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError("Invalid validation_size. Must be 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)

    if n_train == 0:
        raise ValueError(
            "With n_samples={}, train_size={} and validation_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, train_size, validation_size)
        )

    return n_train, n_val


class DataSplitter(pl.LightningDataModule):
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    train_size
        float, or None (default is 0.9)
    validation_size
        float, or None (default is None)
    shuffle_set_split
        Whether to shuffle indices before splitting. If `False`, the val, train, and test set are split in the
        sequential order of the data according to `validation_size` and `train_size` percentages.
    pin_memory
        Whether to copy tensors into device-pinned memory before returning them. Passed
        into :class:`~scvi.data.AnnDataLoader`.
    **kwargs
        Keyword args for data loader. If adata has labeled data, data loader
        class is :class:`~scvi.dataloaders.SemiSupervisedDataLoader`,
        else data loader class is :class:`~scvi.dataloaders.AnnDataLoader`.

    Examples
    --------
    >>> adata = scvi.data.synthetic_iid()
    >>> scvi.model.SCVI.setup_anndata(adata)
    >>> adata_manager = scvi.model.SCVI(adata).adata_manager
    >>> splitter = DataSplitter(adata)
    >>> splitter.setup()
    >>> train_dl = splitter.train_dataloader()
    """

    data_loader_cls = AnnDataLoader

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.data_loader_kwargs = kwargs
        self.pin_memory = pin_memory or settings.dl_pin_memory_gpu_training

        self.n_train, self.n_val = validate_data_split(
            self.adata_manager.adata.n_obs, self.train_size, self.validation_size
        )

    def setup(self, stage: Optional[str] = None):
        """Split indices in train/test/val sets."""
        n_train = self.n_train
        n_val = self.n_val
        indices = np.arange(self.adata_manager.adata.n_obs)

        if self.shuffle_set_split:
            random_state = np.random.RandomState(seed=settings.seed)
            indices = random_state.permutation(indices)

        self.val_idx = indices[:n_val]
        self.train_idx = indices[n_val : (n_val + n_train)]
        self.test_idx = indices[(n_val + n_train) :]

    def train_dataloader(self):
        """Create train data loader."""
        return self.data_loader_cls(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=True,
            drop_last=False,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        if len(self.val_idx) > 0:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        """Create test data loader."""
        if len(self.test_idx) > 0:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass


class MixUpDataSplitter(DataSplitter):
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    It makes sure that different cell type proportions are given for every batch, with 
    the use of a Dirichlet distribution.
    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    train_size
        float, or None (default is 0.9)
    validation_size
        float, or None (default is None)
    shuffle_set_split
        Whether to shuffle indices before splitting. If `False`, the val, train, and test set are split in the
        sequential order of the data according to `validation_size` and `train_size` percentages.
    pin_memory
        Whether to copy tensors into device-pinned memory before returning them. Passed
        into :class:`~scvi.data.AnnDataLoader`.
    **kwargs
        Keyword args for data loader. If adata has labeled data, data loader
        class is :class:`~scvi.dataloaders.SemiSupervisedDataLoader`,
        else data loader class is :class:`~scvi.dataloaders.AnnDataLoader`.

    Examples
    --------
    >>> adata = scvi.data.synthetic_iid()
    >>> scvi.model.SCVI.setup_anndata(adata)
    >>> adata_manager = scvi.model.SCVI(adata).adata_manager
    >>> splitter = DataSplitter(adata)
    >>> splitter.setup()
    >>> train_dl = splitter.train_dataloader()
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def setup(
        self, stage: Optional[str] = None, prior_alphas: np.array = None, reallocate: bool = False,    
    ):
        """Split indices in train/test/val sets."""
        n_train = self.n_train
        n_val = self.n_val
        indices = np.arange(self.adata_manager.adata.n_obs)

        random_state = np.random.RandomState(seed=42) # settings.seed is None so new random state for each run
        if self.shuffle_set_split:
            indices = random_state.permutation(indices)

        self.val_idx = indices[:n_val]
        self.train_idx = indices[n_val : (n_val + n_train)]
        self.test_idx = indices[(n_val + n_train) :]

        cell_type_key = None
        if (labels_key := self.adata_manager.registry["setup_args"]["labels_key"]) == \
            "cell_type" or labels_key == "cell_types_grouped":
            cell_type_key = labels_key

        if cell_type_key is not None:
            batch_size = self.data_loader_kwargs["batch_size"]
            adata_obs = self.adata_manager.adata.obs
            adata_obs["original_index"] = np.arange(len(self.adata_manager.adata.obs))
            adata_obs_train = adata_obs.iloc[self.train_idx]
            cell_types = adata_obs_train[cell_type_key].value_counts()

            # Number of cells to sample according to each cell type will follow a posterior dirichlet distrib
            if prior_alphas is None:
                prior_alphas = np.ones(len(cell_types))  # non-informative prior
            likelihood_alphas = cell_types / n_train  # multinomial likelihood 
            alpha_posterior = prior_alphas + likelihood_alphas
            n_batch = int(n_train / batch_size)
            last_batch = n_train - n_batch * batch_size

            # Sample the proportions with dirichlet
            posterior_dirichlet = random_state.dirichlet(alpha_posterior, n_batch+1)
            posterior_dirichlet[:n_batch] = np.round(posterior_dirichlet[:n_batch]*batch_size) # it won't always exactly sum up to batch_size but it's okay
            posterior_dirichlet[-1] = np.round(posterior_dirichlet[-1]*last_batch)
            posterior_dirichlet = posterior_dirichlet.astype(np.int64)  # number of cells to sample

            # Sample the cells following the dirichlet proportions.
            # WARNINGS. Some cells may be shown several times during training between 
            # batches (and within batches if reallocate is False) and some other won't 
            # be shown. Rare cell types will be over-represented.
            # We could also create artificially more batches with this technique.
            new_training_indices = []
            max_cells_per_cell_type = adata_obs_train[cell_type_key].value_counts()
            for i in range(n_batch+1):
                for j, cell_type in enumerate(likelihood_alphas.index):
                    replace = False
                    n_additional_cells_to_sample = 0
                    if (n_cells_to_sample := posterior_dirichlet[i][j]) > \
                    (n_cells_max := max_cells_per_cell_type[cell_type]) :
                        # then not enough cells to sample without replacement in this cell type
                        if reallocate :
                            # the additional cells will be sampled in other populations
                            n_additional_cells_to_sample = n_cells_to_sample - n_cells_max
                            n_cells_to_sample = n_cells_max
                        else:
                            # the cells will just be sampled with replacement
                            replace = True
                    indices_cell_sample = random_state.choice(
                        adata_obs_train.loc[adata_obs_train[cell_type_key] == cell_type, "original_index"].tolist(),
                        n_cells_to_sample,
                        replace=replace,
                    ).tolist()
                    additional_indices_cell_sample = random_state.choice(
                        adata_obs_train.loc[adata_obs_train[cell_type_key] != cell_type, "original_index"].tolist(),
                        n_additional_cells_to_sample,
                        replace=False,
                    ).tolist()
                    new_training_indices.extend(indices_cell_sample+additional_indices_cell_sample)
            
            self.train_idx = new_training_indices # new length will be +/- 5 cells different from the original self.train_idx

        else:
            logger.warn(
                "No cell type was given for categorical key. The data will be split "
                "randomly."
            )

    def train_dataloader(self):
        """Create train data loader."""
        return self.data_loader_cls(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=False, # in the original DataSplitter, shuffle=True
            drop_last=False,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )


class SemiSupervisedDataSplitter(pl.LightningDataModule):
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.
    The ratio between labeled and unlabeled data in adata will be preserved
    in the train/test/val sets.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    train_size
        float, or None (default is 0.9)
    validation_size
        float, or None (default is None)
    shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set are split in the
            sequential order of the data according to `validation_size` and `train_size` percentages.
    n_samples_per_label
        Number of subsamples for each label class to sample per epoch
    pin_memory
        Whether to copy tensors into device-pinned memory before returning them. Passed
        into :class:`~scvi.data.AnnDataLoader`.
    **kwargs
        Keyword args for data loader. If adata has labeled data, data loader
        class is :class:`~scvi.dataloaders.SemiSupervisedDataLoader`,
        else data loader class is :class:`~scvi.dataloaders.AnnDataLoader`.

    Examples
    --------
    >>> adata = scvi.data.synthetic_iid()
    >>> scvi.model.SCVI.setup_anndata(adata, labels_key="labels")
    >>> adata_manager = scvi.model.SCVI(adata).adata_manager
    >>> unknown_label = 'label_0'
    >>> splitter = SemiSupervisedDataSplitter(adata, unknown_label)
    >>> splitter.setup()
    >>> train_dl = splitter.train_dataloader()
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        n_samples_per_label: Optional[int] = None,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.data_loader_kwargs = kwargs
        self.n_samples_per_label = n_samples_per_label

        labels_state_registry = adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        )
        labels = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.labels.attr_name,
            labels_state_registry.original_key,
        ).ravel()
        self.unlabeled_category = labels_state_registry.unlabeled_category
        self._unlabeled_indices = np.argwhere(labels == self.unlabeled_category).ravel()
        self._labeled_indices = np.argwhere(labels != self.unlabeled_category).ravel()

        self.data_loader_kwargs = kwargs
        self.pin_memory = pin_memory or settings.dl_pin_memory_gpu_training

    def setup(self, stage: Optional[str] = None):
        """Split indices in train/test/val sets."""
        n_labeled_idx = len(self._labeled_indices)
        n_unlabeled_idx = len(self._unlabeled_indices)

        if n_labeled_idx != 0:
            n_labeled_train, n_labeled_val = validate_data_split(
                n_labeled_idx, self.train_size, self.validation_size
            )

            labeled_permutation = self._labeled_indices
            if self.shuffle_set_split:
                rs = np.random.RandomState(seed=settings.seed)
                labeled_permutation = rs.choice(
                    self._labeled_indices, len(self._labeled_indices), replace=False
                )

            labeled_idx_val = labeled_permutation[:n_labeled_val]
            labeled_idx_train = labeled_permutation[
                n_labeled_val : (n_labeled_val + n_labeled_train)
            ]
            labeled_idx_test = labeled_permutation[(n_labeled_val + n_labeled_train) :]
        else:
            labeled_idx_test = []
            labeled_idx_train = []
            labeled_idx_val = []

        if n_unlabeled_idx != 0:
            n_unlabeled_train, n_unlabeled_val = validate_data_split(
                n_unlabeled_idx, self.train_size, self.validation_size
            )

            unlabeled_permutation = self._unlabeled_indices
            if self.shuffle_set_split:
                rs = np.random.RandomState(seed=settings.seed)
                unlabeled_permutation = rs.choice(
                    self._unlabeled_indices, len(self._unlabeled_indices)
                )

            unlabeled_idx_val = unlabeled_permutation[:n_unlabeled_val]
            unlabeled_idx_train = unlabeled_permutation[
                n_unlabeled_val : (n_unlabeled_val + n_unlabeled_train)
            ]
            unlabeled_idx_test = unlabeled_permutation[
                (n_unlabeled_val + n_unlabeled_train) :
            ]
        else:
            unlabeled_idx_train = []
            unlabeled_idx_val = []
            unlabeled_idx_test = []

        indices_train = np.concatenate((labeled_idx_train, unlabeled_idx_train))
        indices_val = np.concatenate((labeled_idx_val, unlabeled_idx_val))
        indices_test = np.concatenate((labeled_idx_test, unlabeled_idx_test))

        self.train_idx = indices_train.astype(int)
        self.val_idx = indices_val.astype(int)
        self.test_idx = indices_test.astype(int)

        if len(self._labeled_indices) != 0:
            self.data_loader_class = SemiSupervisedDataLoader
            dl_kwargs = {
                "n_samples_per_label": self.n_samples_per_label,
            }
        else:
            self.data_loader_class = AnnDataLoader
            dl_kwargs = {}

        self.data_loader_kwargs.update(dl_kwargs)

    def train_dataloader(self):
        """Create the train data loader."""
        return self.data_loader_class(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=True,
            drop_last=False,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        """Create the validation data loader."""
        if len(self.val_idx) > 0:
            return self.data_loader_class(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        """Create the test data loader."""
        if len(self.test_idx) > 0:
            return self.data_loader_class(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass


@devices_dsp.dedent
class DeviceBackedDataSplitter(DataSplitter):
    """Creates loaders for data that is already on device, e.g., GPU.

    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    train_size
        float, or None (default is 0.9)
    validation_size
        float, or None (default is None)
    %(param_accelerator)s
    %(param_device)s
    pin_memory
        Whether to copy tensors into device-pinned memory before returning them. Passed
        into :class:`~scvi.data.AnnDataLoader`.
    shuffle
        if ``True``, shuffles indices before sampling for training set
    shuffle_test_val
        Shuffle test and validation indices.
    batch_size
        batch size of each iteration. If `None`, do not minibatch

    Examples
    --------
    >>> adata = scvi.data.synthetic_iid()
    >>> scvi.model.SCVI.setup_anndata(adata)
    >>> adata_manager = scvi.model.SCVI(adata).adata_manager
    >>> splitter = DeviceBackedDataSplitter(adata)
    >>> splitter.setup()
    >>> train_dl = splitter.train_dataloader()
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float = 1.0,
        validation_size: Optional[float] = None,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        pin_memory: bool = False,
        shuffle: bool = False,
        shuffle_test_val: bool = False,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            adata_manager=adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            pin_memory=pin_memory,
            **kwargs,
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_test_val = shuffle_test_val
        _, _, self.device = parse_device_args(
            accelerator=accelerator, devices=device, return_device="torch"
        )

    def setup(self, stage: Optional[str] = None):
        """Create the train, validation, and test indices."""
        super().setup()

        if self.shuffle is False:
            self.train_idx = np.sort(self.train_idx)
            self.val_idx = (
                np.sort(self.val_idx) if len(self.val_idx) > 0 else self.val_idx
            )
            self.test_idx = (
                np.sort(self.test_idx) if len(self.test_idx) > 0 else self.test_idx
            )

        self.train_tensor_dict = self._get_tensor_dict(
            self.train_idx, device=self.device
        )
        self.test_tensor_dict = self._get_tensor_dict(self.test_idx, device=self.device)
        self.val_tensor_dict = self._get_tensor_dict(self.val_idx, device=self.device)

    def _get_tensor_dict(self, indices, device):
        """Get tensor dict for a given set of indices."""
        if len(indices) is not None and len(indices) > 0:
            dl = AnnDataLoader(
                self.adata_manager,
                indices=indices,
                batch_size=len(indices),
                shuffle=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
            # will only have one minibatch
            for batch in dl:
                tensor_dict = batch

            for k, v in tensor_dict.items():
                tensor_dict[k] = v.to(device)

            return tensor_dict
        else:
            return None

    def _make_dataloader(self, tensor_dict: Dict[str, torch.Tensor], shuffle):
        """Create a dataloader from a tensor dict."""
        if tensor_dict is None:
            return None
        dataset = _DeviceBackedDataset(tensor_dict)
        bs = self.batch_size if self.batch_size is not None else len(dataset)
        sampler_cls = SequentialSampler if not shuffle else RandomSampler
        sampler = BatchSampler(
            sampler=sampler_cls(dataset),
            batch_size=bs,
            drop_last=False,
        )
        return DataLoader(dataset, sampler=sampler, batch_size=None)

    def train_dataloader(self):
        """Create the train data loader."""
        return self._make_dataloader(self.train_tensor_dict, self.shuffle)

    def test_dataloader(self):
        """Create the test data loader."""
        return self._make_dataloader(self.test_tensor_dict, self.shuffle_test_val)

    def val_dataloader(self):
        """Create the validation data loader."""
        return self._make_dataloader(self.val_tensor_dict, self.shuffle_test_val)


class _DeviceBackedDataset(Dataset):
    def __init__(self, tensor_dict: Dict[str, torch.Tensor]):
        self.data = tensor_dict

    def __getitem__(self, idx: List[int]) -> Dict[str, torch.Tensor]:
        return_dict = {}
        for key, value in self.data.items():
            return_dict[key] = value[idx]

        return return_dict

    def __len__(self):
        for _, value in self.data.items():
            return len(value)

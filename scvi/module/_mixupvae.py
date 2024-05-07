"""Main module."""
import logging
from typing import Callable, Iterable, Literal, Optional
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.autotune._types import Tunable
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import one_hot
from scvi.nn import Encoder
from ._vae import VAE
from ._utils import (
    run_incompatible_value_checks,
    get_mean_pearsonr_torch,
    compute_ground_truth_proportions,
    compute_signature,
)

# for deconvolution
from scipy.optimize import nnls
from scipy.stats import pearsonr

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


class MixUpVAE(VAE):
    """Variational auto-encoder model with linearity constraint within batches.

    The linearity constraint is inspired by the MixUp method
    (https://arxiv.org/abs/1710.09412v2).

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    n_pseudobulks
        Number of pseudobulks to create (i.e. number of MixUp losses to compute and average).
    n_cells_per_pseudobulk
        Number of cells to sample to create a pseudobulk sample. If None, the number 
        of cells is equal to the batch size.
    signature_type
        One of

        * ``'pre_encoded'`` - Compute the signature matrix in the input space.
        * ``'post_inference'`` - Compute the signature matrix inside the latent space.
    loss_computation
        One of

        * ``'latent_space'`` - Compute the MixUp loss in the latent space.
        * ``'reconstructed_space'`` - Compute the MixUp loss in the reconstructed space.
    pseudo_bulk
        One of

        * ``'pre_encoded'`` - Create the pseudo-bulk in the input space.
        * ``'post_inference'`` - Create the pseudo-bulk inside the latent space. Only if the loss is computed in the reconstructed space.
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    mixup_penalty
        The loss to use to compare the average of encoded cell and the encoded pseudobulk.
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_batch_norm
        Whether to use batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    use_size_factor_key
        Use size_factor AnnDataField defined by the user as scaling factor in mean of conditional distribution.
        Takes priority over `use_observed_lib_size`.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    extra_encoder_kwargs
        Extra keyword arguments passed into :class:`~scvi.nn.Encoder`.
    extra_decoder_kwargs
        Extra keyword arguments passed into :class:`~scvi.nn.DecoderSCVI`.
    """

    def __init__(
        self,
        # VAE arguments
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: Tunable[int] = 512,
        n_latent: Tunable[int] = 10,
        n_layers: Tunable[int] = 1,
        seed: Tunable[int] = 0,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: Tunable[float] = 0.1,
        dispersion: Tunable[
            Literal["gene", "gene-batch", "gene-label", "gene-cell"]
        ] = "gene",
        log_variational: bool = True,
        gene_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",
        latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
        encode_covariates: Tunable[bool] = False,
        deeply_inject_covariates: Tunable[bool] = True,
        use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
        use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        extra_encoder_kwargs: Optional[dict] = None,
        extra_decoder_kwargs: Optional[dict] = None,
        # MixUpVAE specific arguments
        n_pseudobulks: Tunable[int] = 1,
        n_cells_per_pseudobulk: Tunable[Optional[int]] = None,
        signature_type: Tunable[str] = "pre_encoded",
        loss_computation: Tunable[str] = "latent_space",
        pseudo_bulk: Tunable[str] = "pre_encoded",
        mixup_penalty: Tunable[str] = "l2",
    ):
        torch.manual_seed(seed)

        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            encode_covariates=encode_covariates,
            deeply_inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_size_factor_key=use_size_factor_key,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            var_activation=var_activation,
            extra_encoder_kwargs=extra_encoder_kwargs,
            extra_decoder_kwargs=extra_decoder_kwargs,
        )
        run_incompatible_value_checks(
            pseudo_bulk=pseudo_bulk,
            loss_computation=loss_computation,
            use_batch_norm=use_batch_norm,
            mixup_penalty=mixup_penalty,
            gene_likelihood=gene_likelihood,
        )
        
        self.n_pseudobulks = n_pseudobulks
        self.n_cells_per_pseudobulk = n_cells_per_pseudobulk
        self.signature_type = signature_type
        self.loss_computation = loss_computation
        self.pseudo_bulk = pseudo_bulk
        self.mixup_penalty = mixup_penalty
        self.z_signature = None
        self.logger_messages = set()

    def _get_inference_input(self,tensors):
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None
        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        if self.minified_data_type is None:
            x = tensors[REGISTRY_KEYS.X_KEY]
            input_dict = {
                "x": x,
                "y": y,
                "batch_index": batch_index,
                "cont_covs": cont_covs,
                "cat_covs": cat_covs,
            }
        else:
            if self.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
                qzm = tensors[REGISTRY_KEYS.LATENT_QZM_KEY]
                qzv = tensors[REGISTRY_KEYS.LATENT_QZV_KEY]
                observed_lib_size = tensors[REGISTRY_KEYS.OBSERVED_LIB_SIZE]
                input_dict = {
                    "qzm": qzm,
                    "qzv": qzv,
                    "observed_lib_size": observed_lib_size,
                }
            else:
                raise NotImplementedError(
                    f"Unknown minified-data type: {self.minified_data_type}"
                )

        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        z_pseudobulk = inference_outputs["z_pseudobulk"]
        categorical_input = inference_outputs["categorical_input"]
        one_hot_batch_index = inference_outputs["one_hot_batch_index"]
        library = inference_outputs["library"]
        library_pseudobulk = inference_outputs["library_pseudobulk"]
        cont_covs_pseudobulk = inference_outputs["cont_covs_pseudobulk"]
        categorical_pseudobulk_input = inference_outputs["categorical_pseudobulk_input"]
        one_hot_batch_index_pseudobulk = inference_outputs["one_hot_batch_index_pseudobulk"]
        pseudobulk_indices = inference_outputs["pseudobulk_indices"]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key])
            if size_factor_key in tensors.keys()
            else None
        )

        input_dict = {
            "z": z,
            "z_pseudobulk": z_pseudobulk,
            "library": library,
            "library_pseudobulk": library_pseudobulk,
            "one_hot_batch_index": one_hot_batch_index,
            "one_hot_batch_index_pseudobulk": one_hot_batch_index_pseudobulk,
            "pseudobulk_indices": pseudobulk_indices,
            "y": y,
            "cont_covs": cont_covs,
            "cont_covs_pseudobulk": cont_covs_pseudobulk,
            "categorical_input": categorical_input,
            "categorical_pseudobulk_input": categorical_pseudobulk_input,
            "size_factor": size_factor,
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        x,
        y,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        n_samples=1,
    ):
        """High level inference method for pseudobulks of single cells.

        Runs the inference (encoder) model.
        """
        x_ = x

        # create self.n_pseudobulks
        if self.n_cells_per_pseudobulk is None or self.n_cells_per_pseudobulk > x_.shape[0] :
            n_cells_per_pseudobulk = x_.shape[0]
        else:
            n_cells_per_pseudobulk = self.n_cells_per_pseudobulk
        random_state = np.random.RandomState(seed=42)
        pseudobulk_indices = random_state.choice(
            x_.shape[0],
            size = (self.n_pseudobulks, n_cells_per_pseudobulk),
            replace = True,
        )
        x_pseudobulk_ = x_[pseudobulk_indices, :].mean(axis=1)
        y_pseudobulk = y[pseudobulk_indices, :].squeeze(axis=2)

        all_proportions = compute_ground_truth_proportions(
            y_pseudobulk, self.n_labels, n_cells_per_pseudobulk
        )
        counts, x_signature_mask, x_signature_ = compute_signature(y, x_)

        # library size
        if self.use_observed_lib_size:
            library = torch.log(x.sum(axis=1)).unsqueeze(1)
            library_pseudobulk = torch.log(x_pseudobulk_.sum(axis=1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
            x_pseudobulk_ = torch.log(1 + x_pseudobulk_)
            x_signature_ = torch.log(1 + x_signature_)

        # continuous covariates
        if cont_covs is not None:
            # for single-cell and pseudobulk
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
            cont_covs_pseudobulk = cont_covs[pseudobulk_indices, :].mean(axis=1)
            encoder_pseudobulk_input = torch.cat((x_pseudobulk_, cont_covs_pseudobulk), dim=-1)
            cont_covs_signature = torch.matmul(x_signature_mask, cont_covs)/counts.unsqueeze(-1)
            encoder_signature_input = torch.cat((x_signature_, cont_covs_signature), dim=-1)
        else:
            encoder_input = x_
            cont_covs_pseudobulk = ()
            encoder_pseudobulk_input = x_pseudobulk_
            encoder_signature_input = x_signature_

        # categorical covariates
        if cat_covs is not None:
            cat_covs = torch.split(cat_covs, 1, dim=1)
            categorical_input = []
            categorical_pseudobulk_input = []
            categorical_signature_input = []
            j=0
            for n_cat in self.decoder.px_decoder.n_cat_list:
                if n_cat > 0 :
                    # if n_cat == 0 then no batch index was given, so skip it
                    one_hot_cat_covs = one_hot(cat_covs[j], n_cat)
                    categorical_input.append(one_hot_cat_covs)
                    one_hot_cat_covs_pseudobulk = one_hot_cat_covs[pseudobulk_indices, :].mean(axis=1)
                    categorical_pseudobulk_input.append(one_hot_cat_covs_pseudobulk)
                    one_hot_cat_covs_signature = torch.matmul(x_signature_mask, one_hot_cat_covs)/counts.unsqueeze(-1)
                    categorical_signature_input.append(one_hot_cat_covs_signature)
                    j+=1
            categorical_input_copy = copy.deepcopy(categorical_input)
            categorical_pseudobulk_input_copy = copy.deepcopy(categorical_pseudobulk_input)
        else :
            categorical_input_copy = ()
            categorical_pseudobulk_input_copy = ()

        # Encode covariates
        if not self.encode_covariates:
            # don't enconde categorical covariates but return copy in inference outputs
            categorical_input = ()
            categorical_pseudobulk_input = ()
            categorical_signature_input = ()
        
        one_hot_batch_index = one_hot(batch_index, self.n_batch)
        one_hot_batch_index_pseudobulk = one_hot_batch_index[pseudobulk_indices, :].mean(axis=1)

        # regular encoding
        qz, z = self.z_encoder(
            encoder_input,
            one_hot_batch_index,
            *categorical_input,
        )
        # pseudobulk encoding
        qz_pseudobulk, z_pseudobulk = self.z_encoder(
            encoder_pseudobulk_input,
            one_hot_batch_index_pseudobulk,
            *categorical_pseudobulk_input,
        )
        # pure cell type signature encoding
        if self.signature_type == "pre_encoded":
            _, z_signature = self.z_encoder(
                encoder_signature_input,
                batch_index,
                *categorical_signature_input,
            )
        elif self.signature_type == "post_inference":
            # create signature matrix - inside the latent space
            z_signature = torch.matmul(x_signature_mask, z)/counts.unsqueeze(-1)

        # library size
        ql = None
        ql_pseudobulk = None

        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            ql_pseudobulk, library_pseudobulk_encoded = self.l_encoder(
                encoder_pseudobulk_input, batch_index, *categorical_pseudobulk_input
            )
            library = library_encoded
            library_pseudobulk = library_pseudobulk_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))

        if self.z_encoder.training and z_signature.shape[0] == self.n_labels:
            # then keep training signature in memory to be used for validation
            self.z_signature = z_signature

        outputs = {
            "all_proportions": all_proportions,
            # encodings
            "z": z,
            "qz": qz,
            "ql": ql,
            "library": library,
            "categorical_input": categorical_input_copy,
            "one_hot_batch_index": one_hot_batch_index,
            # pseudobulk encodings
            "pseudobulk_indices": pseudobulk_indices,
            "z_pseudobulk": z_pseudobulk,
            "qz_pseudobulk": qz_pseudobulk,
            "ql_pseudobulk": ql_pseudobulk,
            "library_pseudobulk": library_pseudobulk,
            "categorical_pseudobulk_input": categorical_pseudobulk_input_copy,
            "cont_covs_pseudobulk": cont_covs_pseudobulk,
            "one_hot_batch_index_pseudobulk": one_hot_batch_index_pseudobulk,
            # pure cell type signature encodings
            "z_signature": z_signature,
        }

        return outputs

    @auto_move_data
    def generative(
        self,
        z,
        z_pseudobulk,
        library,
        library_pseudobulk,
        one_hot_batch_index,
        one_hot_batch_index_pseudobulk,
        cont_covs=None,
        cont_covs_pseudobulk=None,
        categorical_input=(),
        categorical_pseudobulk_input=(),
        pseudobulk_indices=(),
        size_factor=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        if self.pseudo_bulk == "post_inference":
            # create it from z by overwritting the one coming from inference
            z_pseudobulk = z[pseudobulk_indices, :].mean(axis=1)

        if cont_covs is None:
            decoder_input = z
            decoder_pseudobulk_input = z_pseudobulk
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
            decoder_pseudobulk_input = torch.cat(
                [
                    z_pseudobulk,
                    cont_covs_pseudobulk.unsqueeze(0)
                    .expand(z_pseudobulk.size(0), -1, -1)
                ],
                dim=-1,
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)
            decoder_pseudobulk_input = torch.cat(
                [z_pseudobulk, cont_covs_pseudobulk], dim=-1
            )
            

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        size_factor_pseudobulk = None
        if not self.use_size_factor_key:
            size_factor = library
            size_factor_pseudobulk = library_pseudobulk

        one_hot_y = one_hot(y, self.n_labels)
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            one_hot_batch_index,
            *categorical_input,
            one_hot_y,
        )
        one_hot_y_pseudobulk = one_hot_y[pseudobulk_indices, :].mean(axis=1)
        (
            px_pseudobulk_scale,
            px_pseudobulk_r,
            px_pseudobulk_rate,
            px_pseudobulk_dropout,
        ) = self.decoder(
            self.dispersion,
            decoder_pseudobulk_input,
            size_factor_pseudobulk,
            one_hot_batch_index_pseudobulk,
            *categorical_pseudobulk_input,
            one_hot_y_pseudobulk,
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot_y, self.px_r
            )  # px_r gets transposed - last dimension is nb genes
            px_pseudobulk_r = F.linear(
                one_hot_y_pseudobulk, self.px_r
            )  # should we really create self.px_pseudobulk_r ?
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
            px_pseudobulk_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
            px_pseudobulk_r = self.px_r

        px_r = torch.exp(px_r)
        px_pseudobulk_r = torch.exp(px_pseudobulk_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)
        px_pseudobulk = NegativeBinomial(
            mu=px_pseudobulk_rate, theta=px_pseudobulk_r, scale=px_pseudobulk_scale
        )
        # Priors
        if self.use_observed_lib_size:
            pl = None
            pl_pseudobulk = None
        elif self.n_batch > 1:
            raise ValueError(
                "Not using observed library size while having more than one batch does "
                "not make sense for the nature of pseudobulk."
            )
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            (
                local_library_log_means_pseudobulk,
                local_library_log_vars_pseudobulk,
            ) = self._compute_local_library_params(batch_index[0].unsqueeze(0))
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
            pl_pseudobulk = Normal(
                local_library_log_means_pseudobulk,
                local_library_log_vars_pseudobulk.sqrt(),
            )
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        pz_pseudobulk = Normal(
            torch.zeros_like(z_pseudobulk), torch.ones_like(z_pseudobulk)
        )
        return {
            "px": px,
            "pl": pl,
            "pz": pz,
            # pseudobulk decodings
            "px_pseudobulk": px_pseudobulk,
            "pl_pseudobulk": pl_pseudobulk,
            "pz_pseudobulk": pz_pseudobulk,
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Computes the loss function for the model."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(
            dim=-1
        )
        if not self.use_observed_lib_size:
            kl_divergence_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        mixup_loss = self.get_mix_up_loss(inference_outputs, generative_outputs)

        loss = torch.mean(reconst_loss + weighted_kl_local) + mixup_loss

        # correlation in latent space
        pseudobulk_indices = inference_outputs["pseudobulk_indices"]
        mean_z = inference_outputs["z"][pseudobulk_indices, :].mean(axis=1)
        pseudobulk_z = inference_outputs["z_pseudobulk"]
        pearson_coeff = get_mean_pearsonr_torch(mean_z, pseudobulk_z)

        # deconvolution in latent space
        pearson_deconv_results = []
        cosine_deconv_results = []
        mse_deconv_results = []
        mae_deconv_results = []
        z_signature = inference_outputs["z_signature"] if self.z_signature is None else self.z_signature
        for i, pseudobulk in enumerate(pseudobulk_z.detach().cpu().numpy()):
            predicted_proportions = nnls(
                z_signature.detach().cpu().numpy().T,
                pseudobulk,
            )[0]
            if self.z_signature is None:
                # resize predicted_proportions to the right number of labels
                full_predicted_proportions = np.zeros(self.n_labels)
                for j, cell_type in enumerate(tensors["labels"].unique().detach().cpu()):
                    full_predicted_proportions[int(cell_type)] = predicted_proportions[j]
                predicted_proportions = full_predicted_proportions
            if np.any(predicted_proportions):
                # if not all zeros, sum the predictions to 1
                predicted_proportions = predicted_proportions / predicted_proportions.sum()
            proportions_array = inference_outputs["all_proportions"][i].detach().cpu().numpy()
            # Deconvolution metrics
            cosine_similarity = (
                np.dot(proportions_array, predicted_proportions)
                / np.linalg.norm(proportions_array)
                / np.linalg.norm(predicted_proportions)
            )
            pearson_coeff_deconv = pearsonr(proportions_array, predicted_proportions)[0]
            pearson_deconv_results.append(pearson_coeff_deconv)
            cosine_deconv_results.append(cosine_similarity)
            # Deconvolution errors
            mse_deconv = np.mean((proportions_array - predicted_proportions)**2)
            mae_deconv = np.mean(np.abs(proportions_array - predicted_proportions))
            mse_deconv_results.append(mse_deconv)
            mae_deconv_results.append(mae_deconv)
        pearson_coeff_deconv = sum(pearson_deconv_results)/len(pearson_deconv_results)
        cosine_similarity = sum(cosine_deconv_results)/len(cosine_deconv_results)
        mse_deconv = sum(mse_deconv_results)/len(mse_deconv_results)
        mae_deconv = sum(mae_deconv_results)/len(mae_deconv_results)
        

        # logging
        kl_local = {
            "kl_divergence_l": kl_divergence_l,
            "kl_divergence_z": kl_divergence_z,
        }

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
            extra_metrics={
                "mixup_penalty": mixup_loss,
                "pearson_coeff": pearson_coeff,
                "cosine_similarity": cosine_similarity,
                "pearson_coeff_deconv": pearson_coeff_deconv,
                "mse_deconv": mse_deconv,
                "mae_deconv": mae_deconv,
            },
        )

    def get_mix_up_loss(self, inference_outputs, generative_outputs):
        """Compute L2 loss or KL divergence between single cells average and pseudobulk."""
        pseudobulk_indices = inference_outputs["pseudobulk_indices"]
        if self.mixup_penalty == "l2":
            # l2 penalty between mean(cells) and pseudobulk
            if self.loss_computation == "latent_space":
                mean_single_cells = inference_outputs["z"][pseudobulk_indices, :].mean(axis=1)
                pseudobulk = inference_outputs["z_pseudobulk"]
            elif self.loss_computation == "reconstructed_space":
                message = (
                    "Sampling with the reparametrization trick is not possible with"
                    "discrete probability distribution like Poisson, NB, ZINB. "
                    "Therefore, the MixUp penalty will the rate (i.e. mean) of  the"
                    "distribution instead."
                )
                if message not in self.logger_messages:
                    logger.warn(message)
                    self.logger_messages.add(message)
                if self.gene_likelihood in ("zinb", "nb"):
                    mean_single_cells = generative_outputs["px"].mu[pseudobulk_indices, :].mean(axis=1)
                    pseudobulk = generative_outputs["px_pseudobulk"].mu
                elif self.gene_likelihood == "poisson":
                    mean_single_cells = generative_outputs["px"].rate[pseudobulk_indices, :].mean(axis=1)
                    pseudobulk = generative_outputs["px_pseudobulk"].rate
            mixup_penalty = torch.sum((pseudobulk - mean_single_cells) ** 2, axis=1)
        elif self.mixup_penalty == "kl":
            # kl of mean(cells) compared to reference pseudobulk
            if self.loss_computation == "latent_space":
                mean_averaged_cells = inference_outputs["qz"].mean[pseudobulk_indices, :].mean(axis=1)
                std_averaged_cells = inference_outputs["qz"].variance[pseudobulk_indices, :].sum(
                    axis=1
                ).sqrt() / pseudobulk_indices.shape[1]
                averaged_cells_distrib = Normal(mean_averaged_cells, std_averaged_cells)
                pseudobulk_reference_distrib = inference_outputs["qz_pseudobulk"]
            elif self.loss_computation == "reconstructed_space":
                raise NotImplementedError(
                    "KL divergence is not implemented for other distributions than Normal."
                )
                # if self.gene_likelihood == "poisson":
                #     rate_averaged_cells = generative_outputs["px"].rate[pseudobulk_indices, :].mean(axis=1)
                #     averaged_cells_distrib = Poisson(rate_averaged_cells)
                # elif self.dispersion != "gene":
                #     raise NotImplementedError(
                #         "The parameters of the sum of independant variables following "
                #         "negative binomials with varying dispersion is not computable yet."
                #     )
                # elif self.gene_likelihood == "nb":
                #     # only works because dispersion is constant across cells
                #     mean_averaged_cells = generative_outputs["px"].mu[pseudobulk_indices, :].mean(axis=1)
                #     dispersion_averaged_cells = generative_outputs["px"].theta[pseudobulk_indices, :].mean(axis=1)
                #     averaged_cells_distrib = NegativeBinomial(mu=mean_averaged_cells, theta=dispersion_averaged_cells)
                # elif self.gene_likelihood == "zinb":
                #     # not yet implemented because of the zero inflation
                #     raise NotImplementedError(
                #         "The parameters of the sum of independant variables following "
                #         "zero-inflated negative binomials is not computable yet."
                #     )
                # pseudobulk_reference_distrib = generative_outputs["px_pseudobulk"]
            mixup_penalty = kl(
                averaged_cells_distrib, pseudobulk_reference_distrib
            ).sum(dim=-1)
        
        mixup_penalty = torch.sum(mixup_penalty) / mean_single_cells.shape[0]
        
        return mixup_penalty

"""Main module."""
import logging
from typing import Callable, Iterable, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import one_hot
from scvi.nn import DecoderSCVI
from ._vae import VAE

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
    signature_type
        One of

        * ``'pre_encoded'`` - Compute the signature matrix in the input space.
        * ``'post_encoded'`` - Compute the signature matrix inside the latent space.
    loss_computation
        One of

        * ``'latent_space'`` - Compute the MixUp loss in the latent space.
        * ``'reconstructed_space'`` - Compute the MixUp loss in the reconstructed space.
        * ``'both'`` - Compute the MixUp loss in both latent and reconstructed space and sum them.
    pseudo_bulk
        One of

        * ``'pre_encoded'`` - Create the pseudo-bulk in the input space.
        * ``'latent_space'`` - Create the pseudo-bulk inside the latent space. Only if the loss is computed in the reconstructed space.
    encode_covariates
        Whether to concatenate covariates to expression in encoder. Not used in MixUpVI,
         where we differentiate for now between caterigocal and continuous covariates.
    encode_cont_covariates
        Whether to concatenate continuous covariates to expression in encoder
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
        n_hidden: Tunable[int] = 128,
        n_latent: Tunable[int] = 10,
        n_layers: Tunable[int] = 1,
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
        signature_type: str = "pre_encoded",
        loss_computation: str = "latent_space",
        pseudo_bulk: str = "pre_encoded",
        encode_cont_covariates: bool = False,
        mixup_penalty: str = "l2",
    ):
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
            # encode_covariates=encode_covariates,
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
        self.signature_type = signature_type
        self.loss_computation = loss_computation
        self.pseudo_bulk = pseudo_bulk
        self.encode_cont_covariates = encode_cont_covariates
        self.mixup_penalty = mixup_penalty
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=None,  # for now, we don't decode categorical variables
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            **_extra_decoder_kwargs,
        )

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        z_pseudobulk = inference_outputs["z_pseudobulk"]
        library = inference_outputs["library"]
        library_pseudobulk = inference_outputs["library_pseudobulk"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

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
            "batch_index": batch_index,
            "y": y,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "size_factor": size_factor,
        }
        return input_dict

    @auto_move_data
    def _regular_inference(
        self,
        x,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        n_samples=1,
    ):
        """High level inference method for pseudobulks of single cells.

        Runs the inference (encoder) model.
        """
        x_ = x
        x_pseudobulk_ = x.mean(axis=0).unsqueeze(0)

        # create signature
        unique_indices, counts = cat_covs.unique(return_counts=True)
        proportions = counts.float() / len(cat_covs)
        x_signature_ = []
        for cell_type in unique_indices:
            # TODO Create train/val split, so that the signature is created with all
            # training data at the end of epoch before computing metrics with val data
            idx = (cat_covs == cell_type).flatten()
            x_pure = x_[idx, :].mean(axis=0)
            x_signature_.append(x_pure)
        x_signature_ = torch.stack(x_signature_, dim=0)

        if self.use_observed_lib_size:
            library = torch.log(x.sum(axis=1)).unsqueeze(1)
            library_pseudobulk = torch.log(x.sum())
            library_signature = torch.log(x_signature_.sum(axis=1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
            x_pseudobulk_ = torch.log(1 + x_pseudobulk_)
            x_signature_ = torch.log(1 + x_signature_)
        if cont_covs is not None and self.encode_cont_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
            encoder_pseudobulk_input = torch.cat(
                (x_pseudobulk_, cont_covs.mean(axis=0).unsqueeze(1)), dim=-1
            )
            cont_covs_signature = []
            for cell_type in unique_indices:
                # we repeat this loop with code above, is there a way not to, while
                # not adding too many if statements ?
                idx = (cat_covs == cell_type).flatten()
                cont_covs_pure = cont_covs[idx, :].mean(axis=0)
                cont_covs_signature.append(cont_covs_pure)
            cont_covs_signature = torch.stack(cont_covs_signature, dim=0)
            encoder_signature_input = torch.cat(
                (x_signature_, cont_covs_signature), dim=-1
            )
        else:
            encoder_input = x_
            encoder_pseudobulk_input = x_pseudobulk_
            encoder_signature_input = x_signature_
        if cat_covs is not None and self.encode_covariates:
            # TODO investigate how to add other cat covariates than cell types
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        # regular encoding
        qz, z = self.z_encoder(
            encoder_input,
            batch_index,  # will not be used since encode_covariates==False
            *categorical_input,  # will not be used since encode_covariates==False
        )
        # pseudobulk encoding
        qz_pseudobulk, z_pseudobulk = self.z_encoder(
            encoder_pseudobulk_input,
            batch_index,  # will not be used since encode_covariates==False
            *categorical_input,  # will not be used since encode_covariates==False
        )  # for pseudobulk, batch_index and cat_covs are not the right dimension
        # pure cell type signature encoding
        if self.signature_type == "pre_encoded":
            # TODO Create train/val split, so that the signature is created with all
            # training data at the end of epoch before computing metrics with val data
            # create signature matrix - pre-encoding
            qz_signature, z_signature = self.z_encoder(
                encoder_signature_input,
                batch_index,  # will not be used since encode_covariates==False
                *categorical_input,  # will not be used since encode_covariates==False
            )  # for signature, batch_index and cat_covs are not the right dimension
        elif self.signature_type == "post_encoded":
            # create signature matrix - inside the latent space
            z_signature = []
            for cell_type in unique_indices:
                idx = (cat_covs == cell_type).flatten()
                z_pure = z[idx, :].mean(axis=0)
                z_signature.append(z_pure)
            z_signature = torch.stack(z_signature, dim=0)
            qz_signature = None

        # library size
        ql = None
        ql_pseudobulk = None
        ql_signature = None

        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )  # will not use batch_index and categorical_input since encode_covariates==False
            ql_pseudobulk, library_pseudobulk_encoded = self.l_encoder(
                encoder_pseudobulk_input, batch_index, *categorical_input
            )  # will not use batch_index and categorical_input since encode_covariates==False
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
                library_pseudobulk = ql_pseudobulk.sample((n_samples,))

        outputs = {
            "z": z,
            "qz": qz,
            "ql": ql,
            "library": library,
            # pseudobulk encodings
            "z_pseudobulk": z_pseudobulk,
            "qz_pseudobulk": qz_pseudobulk,
            "ql_pseudobulk": ql_pseudobulk,
            "library_pseudobulk": library_pseudobulk,
            # pure cell type signature encodings
            "proportions": proportions,
            "z_signature": z_signature,
            "qz_signature": qz_signature,
            "ql_signature": ql_signature,
            "library_signature": library_signature,
        }

        return outputs

    @auto_move_data
    def generative(
        self,
        z,
        z_pseudobulk,
        library,
        library_pseudobulk,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        if self.pseudo_bulk == "latent_space":
            # create it from z by overwritting the one coming from inference
            z_pseudobulk = z.mean(axis=0).unsqueeze(0)

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
                    cont_covs.unsqueeze(0)
                    .expand(z_pseudobulk.size(0), -1, -1)
                    .mean(axis=0)
                    .unsqueeze(1),
                ],
                dim=-1,
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)
            decoder_pseudobulk_input = torch.cat(
                [z_pseudobulk, cont_covs.mean(axis=0).unsqueeze(1)], dim=-1
            )

        if cat_covs is not None:
            # TODO do we really want cell_types to be include in the decoder?
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library
        size_factor_pseudobulk = library_pseudobulk

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            batch_index,  # will not be used since n_cat_list==None in decoder init
            *categorical_input,  # will not be used since n_cat_list==None in decoder init
            y,  # will not be used since n_cat_list==None in decoder init
        )
        (
            px_pseudobulk_scale,
            px_pseudobulk_r,
            px_pseudobulk_rate,
            px_pseudobulk_dropout,
        ) = self.decoder(
            self.dispersion,
            decoder_pseudobulk_input,
            size_factor_pseudobulk,
            batch_index,  # will not be used since n_cat_list==None in decoder init
            *categorical_input,  # will not be used since n_cat_list==None in decoder init
            y,  # will not be used since n_cat_list==None in decoder init
        )  # for pseudobulk, batch_index, categorical_input and y are not the right dimension
        if self.dispersion == "gene-label":
            # does not make sense for pseudobulk
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            # does not make sense for pseudobulk
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
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
            px_pseudobulk = ZeroInflatedNegativeBinomial(
                mu=px_pseudobulk_rate,
                theta=px_pseudobulk_r,
                zi_logits=px_pseudobulk_dropout,
                scale=px_pseudobulk_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)
            px_pseudobulk = Poisson(px_pseudobulk_rate, scale=px_pseudobulk_scale)
        # pseudobulk gene likelihood
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

        mixup_loss = 0
        mean_z = torch.mean(inference_outputs["z"], axis=0)
        if self.loss_computation == "latent_space" or self.loss_computation == "both":
            if self.mixup_penalty == "l2":
                # l2 penalty in latent space
                mixup_loss += self.get_l2_mixup_loss(
                    mean_z, inference_outputs["z_pseudobulk"].squeeze(0)
                )

            elif self.mixup_penalty == "kl":
                # kl of mean(encoded cells) compared to reference encoded pseudobulk
                mixup_loss += self.get_kl_mixup_loss(
                    inference_outputs["qz"], inference_outputs["qz_pseudobulk"]
                )  # should it be -= instead of += ?

        if (
            self.loss_computation == "reconstructed_space"
            or self.loss_computation == "both"
        ):
            if self.mixup_penalty == "l2":
                # l2 penalty in reconstructed space
                generative_x = generative_outputs["px"].sample()
                mean_x = torch.mean(generative_x, axis=0)
                generative_pseudobulk = generative_outputs["px_pseudobulk"].sample()
                mixup_loss += self.get_l2_mixup_loss(mean_x, generative_pseudobulk)
            elif self.mixup_penalty == "kl":
                # kl of mean(encoded cells) compared to reference encoded pseudobulk
                mixup_loss += self.get_kl_mixup_loss(
                    generative_outputs["px"], generative_outputs["px_pseudobulk"]
                )  # should it be -= instead of += ?

        loss = torch.mean(reconst_loss + weighted_kl_local) + mixup_loss

        pearson_coeff = self.get_pearsonr_torch(
            mean_z, inference_outputs["z_pseudobulk"].squeeze(0)
        )
        # deconvolution in latent space
        predicted_proportions = nnls(
            inference_outputs["z_signature"].detach().cpu().numpy().T,
            inference_outputs["z_pseudobulk"].detach().cpu().numpy().flatten(),
        )[0]
        if np.any(predicted_proportions):
            # if not all zeros, sum the predictions to 1
            predicted_proportions = predicted_proportions / predicted_proportions.sum()
        proportions_array = inference_outputs["proportions"].detach().cpu().numpy()
        cosine_similarity = (
            np.dot(proportions_array, predicted_proportions)
            / np.linalg.norm(proportions_array)
            / np.linalg.norm(predicted_proportions)
        )
        pearson_coeff_deconv = pearsonr(proportions_array, predicted_proportions)[0]
        # random proportions
        random_proportions = self.create_random_proportion(
            n_classes=len(proportions_array), n_non_zero=None
        )
        pearson_coeff_random = pearsonr(proportions_array, random_proportions)[0]

        # logging
        reconst_losses = {
            "reconst_loss": reconst_loss,
            "mixup_penalty": mixup_loss,
        }  # never used ?

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
                "pearson_coeff_random": pearson_coeff_random,
            },
        )

    def create_random_proportion(
        self, n_classes: int, n_non_zero: Optional[int] = None
    ) -> np.ndarray:
        """Create a random proportion vector of size n_classes.

        The n_non_zero parameter allows to set the number
        of non-zero components of the random discrete density vector.
        """
        if n_non_zero is None:
            n_non_zero = n_classes

        proportion_vector = np.zeros(
            n_classes,
        )

        proportion_vector[:n_non_zero] = np.random.rand(n_non_zero)

        proportion_vector = proportion_vector / proportion_vector.sum()
        return np.random.permutation(proportion_vector)

    def get_l2_mixup_loss(self, single_cells, pseudobulk):
        """Compute L2 loss between average of single cells and pseudobulk."""
        mixup_penalty = torch.sum((pseudobulk - mean_single_cells) ** 2)
        return mixup_penalty

    def get_kl_mixup_loss(
        self, single_cells_distrib, pseudobulk_distrib, computed_space
    ):
        """Compute KL divergence between average of single cells distrib and pseudobulk
        distribution.
        """
        if computed_space == "latent_space":
            mean_averaged_cells = single_cells_distrib.mean.mean(axis=0)
            std_averaged_cells = single_cells_distrib.variance.sum(axis=0).sqrt() / len(
                single_cells_distrib.loc
            )
            averaged_cells_distrib = Normal(mean_averaged_cells, std_averaged_cells)
            mixup_penalty = kl(averaged_cells_distrib, pseudobulk_distrib).sum(dim=-1)
        elif computed_space == "reconstructed_space":
            if self.gene_likelihood == "poisson":
                rate_averaged_cells = single_cells_distrib.rate.mean(axis=0)
                averaged_cells_distrib = Poisson(rate_averaged_cells)
                mixup_penalty = kl(averaged_cells_distrib, pseudobulk_distrib).sum(
                    dim=-1
                )
            elif self.dispersion != "gene":
                raise NotImplementedError(
                    "The parameters of the sum of independant variables following "
                    "negative binomials with varying dispersion is not computable yet."
                )
            elif self.gene_likelihood == "nb":
                # only works because dispersion is constant across cells
                mean_averaged_cells = single_cells_distrib.mu.mean(axis=0)
                averaged_cells_distrib = NegativeBinomial(rate_averaged_cells)
                mixup_penalty = kl(averaged_cells_distrib, pseudobulk_distrib).sum(
                    dim=-1
                )
            elif self.gene_likelihood == "zinb":
                # not yet implemented because of the zero inflation
                raise NotImplementedError(
                    "The parameters of the sum of independant variables following "
                    "zero-inflated negative binomials is not computable yet."
                )

        return mixup_penalty

    def get_pearsonr_torch(self, x, y):
        """
        Mimics `scipy.stats.pearsonr`
        Arguments
        ---------
        x : 1D torch.Tensor
        y : 1D torch.Tensor
        Returns
        -------
        r_val : float
            pearsonr correlation coefficient between x and y

        Scipy docs ref:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

        Scipy code ref:
            https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
        Example:
            >>> x = np.random.randn(100)
            >>> y = np.random.randn(100)
            >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
            >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
            >>> np.allclose(sp_corr, th_corr)
        """
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x.sub(mean_x)
        ym = y.sub(mean_y)
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        r_val = r_num / r_den
        return r_val

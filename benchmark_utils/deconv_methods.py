"""Deconvolution benchmark methods classes.."""

from __future__ import annotations

from abc import abstractmethod

import anndata as ad
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from TAPE import Deconvolution
from TAPE.deconvolution import ScadenDeconvolution

from .deconv_utils import use_nnls_method
from .latent_signature_utils import create_latent_signature, create_latent_signature_pca
from .pseudobulk_dataset_utils import create_anndata_pseudobulk
from .training_utils import (
    fit_destvi,
    fit_mixupvi,
    fit_scvi,
)


class AbstractDeconvolutionMethod:
    """Abstract Deconvolution Method that every deconvolution method has to inherent from.

    Each deconvolution algorithm must contain an apply_deconvolution method. If the
    algorithm needs fitting (e.g. MixUpVI), then it should be done in the __init__ method.
    """

    @abstractmethod
    def apply_deconvolution(self, to_deconvolve: ad.AnnData, **kwargs):
        """Apply deconvolution method on data to deconvolve.

        Parameters
        ----------
        to_deconvolve: ad.AnnData
            The data to deconvolve.
        """


class NNLSMethod(AbstractDeconvolutionMethod):
    """NNLS deconvolution method."""

    def __init__(self, signature_matrix_name: str, signature_matrix: pd.DataFrame):
        self.signature_matrix_name = signature_matrix_name
        self.signature_matrix = signature_matrix

    def apply_deconvolution(self, to_deconvolve: ad.AnnData | pd.DataFrame):
        """Apply the NNLS method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            to_deconvolve = pd.DataFrame(
                index=to_deconvolve.obs_names,
                columns=to_deconvolve.var_names,
                data=to_deconvolve.layers["counts"],
            ).T
        elif not isinstance(to_deconvolve, pd.DataFrame):
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)

        deconvolution_results = use_nnls_method(to_deconvolve, self.signature_matrix)

        return deconvolution_results


class PCAMethod(AbstractDeconvolutionMethod):
    """PCA deconvolution method. Here the latent signature is created from the PCA of the signature matrix given."""

    def __init__(
        self,
        signature_matrix_name: str,
        signature_matrix: pd.DataFrame,
        n_components: int = 100,
    ):
        self.signature_matrix_name = signature_matrix_name
        self.signature_matrix = signature_matrix
        self.pca = PCA(n_components=n_components)

    def apply_deconvolution(self, to_deconvolve: ad.AnnData | pd.DataFrame):
        """Apply the PCA method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            to_deconvolve = pd.DataFrame(
                index=to_deconvolve.obs_names,
                columns=to_deconvolve.var_names,
                data=to_deconvolve.layers["counts"],
            ).T
        elif not isinstance(to_deconvolve, pd.DataFrame):
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)

        gene_intersection = self.signature_matrix.index.intersection(
            to_deconvolve.index
        )
        to_deconvolve = to_deconvolve.loc[gene_intersection]
        self.signature_matrix = self.signature_matrix.loc[gene_intersection]

        latent_adata = self.pca.fit_transform(to_deconvolve.T)
        latent_signature = self.pca.transform(self.signature_matrix.T)
        latent_adata = pd.DataFrame(latent_adata.T, columns=to_deconvolve.columns)
        latent_signature = pd.DataFrame(
            latent_signature.T, columns=self.signature_matrix.columns
        )
        deconvolution_results = use_nnls_method(latent_adata, latent_signature)
        return deconvolution_results


class PCA_NNLSMethod(AbstractDeconvolutionMethod):
    """PCA + NNLS deconvolution method."""

    def __init__(self, adata_train: ad.AnnData, n_components: int = 100):
        self.adata_train = adata_train
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.pca.fit(adata_train.X.toarray())
        self.latent_signature = create_latent_signature_pca(adata_train, self.pca)
        self.latent_signature = pd.DataFrame(
            self.latent_signature.X.T,
            index=self.latent_signature.var_names,
            columns=self.latent_signature.obs["cell type"],
        )
        logger.debug(f"PCA + NNLS method initialized with {n_components} components.")

    def apply_deconvolution(self, to_deconvolve: ad.AnnData | pd.DataFrame):
        """Apply the PCA + NNLS method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            obs_names = to_deconvolve.obs_names
            # to_deconvolve = pd.DataFrame(
            #     index=to_deconvolve.obs_names,
            #     columns=to_deconvolve.var_names,
            #     data=to_deconvolve.layers["counts"],
            # ).T
        elif not isinstance(to_deconvolve, pd.DataFrame):
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)

        latent_adata = self.pca.transform(to_deconvolve.X)
        latent_adata = pd.DataFrame(
            latent_adata, index=obs_names, columns=self.latent_signature.index
        ).T
        deconvolution_results = use_nnls_method(latent_adata, self.latent_signature)
        return deconvolution_results


class MixUpVIMethod(AbstractDeconvolutionMethod):
    """MixUpVI deconvolution method."""

    def __init__(
        self,
        adata_train: ad.AnnData,
        cell_type_group: str,
        model_path: str = "",
        save_model: bool = False,
    ):
        """Fit MixUpVI and create the latent signature matrix."""
        self.filtered_genes = adata_train.var.index[
            adata_train.var["highly_variable"]
        ].tolist()
        adata_train = adata_train[:, self.filtered_genes]
        self.adata_obs = adata_train.obs

        logger.debug("Fitting MixUpVI...")
        self.mixupvi = fit_mixupvi(
            adata=adata_train.copy(),
            model_path=model_path,
            cell_type_group=cell_type_group,
            save_model=save_model,
        )

        logger.debug("Training over. Creation of latent signature matrix...")
        self.adata_latent_signature = create_latent_signature(
            adata=adata_train,
            model=self.mixupvi,
            use_mixupvi=False,  # should be equal to use_mixupvi, but if True,
            # then it averages as many cells as self.n_cells_per-pseudobulk from mixupvae
            # (and not the number we wish in the benchmark)
            average_all_cells=True,
        )
        self.adata_latent_signature = pd.DataFrame(
            self.adata_latent_signature.X.T,
            index=self.adata_latent_signature.var_names,
            columns=self.adata_latent_signature.obs["cell type"],
        )

    def apply_deconvolution(self, to_deconvolve: ad.AnnData | pd.DataFrame):
        """Apply the MixUpVI method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            obs_names = to_deconvolve.obs_names
            to_deconvolve = to_deconvolve[:, self.filtered_genes]
        elif isinstance(to_deconvolve, pd.DataFrame):
            # Bulk/FACS data
            obs_names = to_deconvolve.columns
            to_deconvolve = create_anndata_pseudobulk(
                adata_obs=self.adata_obs,
                adata_var_names=self.filtered_genes,
                x=to_deconvolve.loc[self.filtered_genes].T.values,
            )
        else:
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)

        latent_adata = self.mixupvi.get_latent_representation(
            to_deconvolve, get_pseudobulk=False
        )
        latent_adata = pd.DataFrame(
            index=obs_names,
            columns=self.adata_latent_signature.index,
            data=latent_adata,
        ).T
        deconvolution_results = use_nnls_method(
            latent_adata, self.adata_latent_signature
        )

        return deconvolution_results


class scVIMethod(AbstractDeconvolutionMethod):
    """scVI deconvolution method."""

    def __init__(
        self,
        adata_train: ad.AnnData,
        model_path: str = "",
        save_model: bool = False,
    ):
        """Fit scVI and create the latent signature matrix."""
        self.filtered_genes = adata_train.var.index[
            adata_train.var["highly_variable"]
        ].tolist()
        adata_train = adata_train[:, self.filtered_genes]
        self.adata_obs = adata_train.obs

        logger.debug("Fitting scVI...")
        self.scvi = fit_scvi(
            adata=adata_train.copy(),
            model_path=model_path,
            save_model=save_model,
        )

        logger.debug("Training over. Creation of latent signature matrix...")
        self.adata_latent_signature = create_latent_signature(
            adata=adata_train,
            model=self.scvi,
            use_mixupvi=False,
            average_all_cells=True,
        )
        self.adata_latent_signature = pd.DataFrame(
            self.adata_latent_signature.X.T,
            index=self.adata_latent_signature.var_names,
            columns=self.adata_latent_signature.obs["cell type"],
        )

    def apply_deconvolution(self, to_deconvolve: ad.AnnData | pd.DataFrame):
        """Apply the scVI method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            obs_names = to_deconvolve.obs_names
            to_deconvolve = to_deconvolve[:, self.filtered_genes]
        elif isinstance(to_deconvolve, pd.DataFrame):
            # Bulk/FACS data
            obs_names = to_deconvolve.columns
            to_deconvolve = create_anndata_pseudobulk(
                adata_obs=self.adata_obs,
                adata_var_names=self.filtered_genes,
                x=to_deconvolve.loc[self.filtered_genes].T.values,
            )
        else:
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)

        latent_adata = self.scvi.get_latent_representation(to_deconvolve)
        latent_adata = pd.DataFrame(
            index=obs_names,
            columns=self.adata_latent_signature.index,
            data=latent_adata,
        ).T
        deconvolution_results = use_nnls_method(
            latent_adata, self.adata_latent_signature
        )

        return deconvolution_results


class DestVIMethod(AbstractDeconvolutionMethod):
    """DestVI deconvolution method."""

    def __init__(
        self,
        adata_train: ad.AnnData,
        adata_pseudobulk: ad.AnnData,
        model_path1: str = "",
        model_path2: str = "",
        cell_type_group: str = "cell_types_grouped",
        save_model: bool = False,
    ):
        """Fit DestVI."""
        self.filtered_genes = adata_train.var.index[
            adata_train.var["highly_variable"]
        ].tolist()
        adata_train = adata_train[:, self.filtered_genes]

        logger.debug("Fitting DestVI...")
        _, self.destvi = fit_destvi(
            adata=adata_train.copy(),
            adata_pseudobulk=adata_pseudobulk[:, self.filtered_genes],
            model_path_1=model_path1,
            model_path_2=model_path2,
            cell_type_group=cell_type_group,
            save_model=save_model,
        )

        logger.debug("Training over.")

    def apply_deconvolution(self, to_deconvolve: ad.AnnData | pd.DataFrame):
        """Apply the DestVI method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            to_deconvolve = to_deconvolve[:, self.filtered_genes]
        elif isinstance(to_deconvolve, pd.DataFrame):
            # Bulk/FACS data
            to_deconvolve = create_anndata_pseudobulk(
                adata_obs=self.adata_obs,
                adata_var_names=self.filtered_genes,
                x=to_deconvolve.loc[self.filtered_genes].T.values,
            )
        else:
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)

        deconvolution_results = self.destvi.get_proportions(to_deconvolve)
        deconvolution_results = deconvolution_results.drop(["noise_term"], axis=1)

        return deconvolution_results


class TAPEMethod(AbstractDeconvolutionMethod):
    """MixUpVI deconvolution method."""

    def __init__(self, signature_matrix_name: str, signature_matrix: pd.DataFrame):
        self.signature_matrix_name = signature_matrix_name
        self.signature_matrix = signature_matrix

    def apply_deconvolution(self, to_deconvolve: ad.AnnData | pd.DataFrame):
        """Apply the TAPE method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            to_deconvolve = pd.DataFrame(
                index=to_deconvolve.obs_names,
                columns=to_deconvolve.var_names,
                data=to_deconvolve.layers["counts"],
            ).T
        elif not isinstance(to_deconvolve, pd.DataFrame):
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)

        _, deconvolution_results = Deconvolution(
            self.signature_matrix.T,
            to_deconvolve.T,
            sep="\t",
            scaler="mms",
            datatype="counts",
            genelenfile=None,
            mode="overall",
            adaptive=True,
            variance_threshold=0.98,
            save_model_name=None,
            batch_size=128,
            epochs=128,
            seed=1,
        )

        return deconvolution_results


class ScadenMethod(AbstractDeconvolutionMethod):
    """Apply the Scaden method on data to deconvolve."""

    def __init__(self, signature_matrix_name: str, signature_matrix: pd.DataFrame):
        self.signature_matrix_name = signature_matrix_name
        self.signature_matrix = signature_matrix

    def apply_deconvolution(self, to_deconvolve: ad.AnnData | pd.DataFrame):
        """Apply the Scaden method on data to deconvolve."""
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            to_deconvolve = pd.DataFrame(
                index=to_deconvolve.obs_names,
                columns=to_deconvolve.var_names,
                data=to_deconvolve.layers["counts"],
            ).T
        elif not isinstance(to_deconvolve, pd.DataFrame):
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)

        deconvolution_results = ScadenDeconvolution(
            self.signature_matrix.T,
            to_deconvolve.T,
            sep="\t",
            batch_size=128,
            epochs=128,
        )

        return deconvolution_results

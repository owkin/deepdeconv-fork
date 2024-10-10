"""
"""

from __future__ import annotations

import anndata as ad
import importlib
import pandas as pd
import scanpy as sc
from abc import abstractmethod
from functools import partial
from loguru import logger
from sklearn.linear_model import LinearRegression

from benchmark_utils import (
    create_anndata_pseudobulk,
    create_latent_signature,
    fit_mixupvi,
    preprocess_scrna,
)
from run_benchmark_constants import (
    DATASETS,
    DECONV_METHODS,
    MODEL_TO_FIT, 
    SIGNATURE_MATRIX_MODELS,
    SIGNATURE_TO_GRANULARITY,
)

# # DECONV_METHODS_FITTING_REQUIREMENTS = {
# #     "MixUpVI": {"single_cell", "variable_genes"},
# #     "NNLS": {"signature_matrix"},
# #     "scVI": {"single_cell", "variable_genes"},
# #     "DestVI": {"single_cell", "variable_genes"},
# #     "TAPE": {"signature_matrix",},
# #     "Scaden": {"signature_matrix",},
# # }
# # FITTING_DATASETS = {"CTI"}
# # SINGLE_CELL_DATASETS = {"TOY", "CTI"}

# DECONV_EVALUATIONS = {"CORRELATION", "GROUP_CORRELATION"}

def load_cti(n_variable_genes: int, **kwargs):
    """TODO: Right now, it's just a raw function to test the code.
    """
    adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
    adata = preprocess_scrna(adata,
                     keep_genes=n_variable_genes,
                     batch_key="donor_id")
    return adata

def load_bulk_facs(**kwargs):
    """TODO: Right now, it's just a raw function to test the code.
    """
    # Load data
    facs_results = pd.read_csv(
        "/home/owkin/project/bulk_facs/240214_majorCelltypes.csv", index_col=0
    ).drop(["No.B.Cells.in.Live.Cells","NKT.Cells.in.Live.Cells"],axis=1).set_index(
        "Sample"
    )
    facs_results = facs_results.rename(
        {
            "B.Cells.in.Live.Cells":"B",
            "NK.Cells.in.Live.Cells":"NK",
            "T.Cells.in.Live.Cells":"T",
            "Monocytes.in.Live.Cells":"Mono",
            "Dendritic.Cells.in.Live.Cells":"DC",
        }, axis=1
    )
    facs_results = facs_results.dropna()
    bulk_data = pd.read_csv(
        (
        "/home/owkin/project/bulk_facs/"
        "gene_counts_batchs1-5_raw.csv"
        # "gene_counts20230103_batch1-5_all_cleaned-TPMnorm-allpatients.tsv"
        ), 
        index_col=0,
        # sep="\t"
    ).T
    common_samples = pd.read_csv(
        "/home/owkin/project/bulk_facs/RNA-FACS_common-samples.csv", index_col=0
    )
    # Align bulk and facs samples
    common_facs = common_samples.set_index("FACS.ID")["Patient"]
    facs_results = facs_results.loc[facs_results.index.isin(common_facs.keys())]
    facs_results = facs_results.rename(index=common_facs)
    common_bulk = common_samples.set_index("RNAseq_ID")["Patient"]
    bulk_data = bulk_data.loc[bulk_data.index.isin(common_bulk.keys())]
    bulk_data = bulk_data.rename(index=common_bulk)
    bulk_data = bulk_data.loc[facs_results.index].T

    return (bulk_data, facs_results)

def use_nnls_method(to_deconvolve: pd.DataFrame, signature_matrix: pd.DataFrame):
    """TODO: Right now, it's just a raw function to test the code.
    """
    intersected_signature = signature_matrix.loc[signature_matrix.index.intersection(
        to_deconvolve.index
    )]
    deconv = LinearRegression(positive=True).fit(
        intersected_signature, to_deconvolve.loc[intersected_signature.index]
    )
    deconv_results = pd.DataFrame(
        deconv.coef_, index=to_deconvolve.columns, columns=intersected_signature.columns
    )
    deconv_results = deconv_results.div(
        deconv_results.sum(axis=1), axis=0
    )  # to sum up to 1
    
    return deconv_results

class AbstractDeconvolutionMethod:
    """TODO: Right now, it's just a raw class to test the code.
    """
    @abstractmethod 
    def apply_deconvolution(self, to_deconvolve: pd.DataFrame, **kwargs):
        """Apply deconvolution method on data to deconvolve.
        
        Parameters
        ----------
        to_deconvolve: pd.DataFrame
            The data to deconvolve.
        """

class NNLSMethod(AbstractDeconvolutionMethod):
    """TODO: Right now, it's just a raw class to test the code.
    """
    def __init__(self, signature_matrix):
        self.signature_matrix = signature_matrix
    
    def apply_deconvolution(self, to_deconvolve: pd.DataFrame):
        """
        """
        deconvolution_results = use_nnls_method(to_deconvolve, self.signature_matrix)
        return deconvolution_results
    
class MixUpVIMethod(AbstractDeconvolutionMethod):
    """TODO: Right now, it's just a raw class to test the code.
    """
    def __init__(
        self, 
        adata_train: ad.AnnData, 
        cell_type_group: str, 
        model_path: str = "", 
        save_model: bool = False, 
    ):
        self.filtered_genes = adata_train.var.index[
            adata_train.var["highly_variable"]
        ].tolist()
        adata_train = adata_train[:,self.filtered_genes]
        self.adata_obs = adata_train.obs

        logger.info("Fitting MixUpVI...")
        self.mixupvi = fit_mixupvi(
            adata=adata_train.copy(),
            model_path=model_path,
            cell_type_group=cell_type_group,
            save_model=save_model,
        )

        logger.info("Training over. Creation of latent signature matrix...")
        self.adata_latent_signature = create_latent_signature(
            adata=adata_train,
            model=self.mixupvi,
            use_mixupvi=False, # should be equal to use_mixupvi, but if True, 
            # then it averages as many cells as self.n_cells_per-pseudobulk from mixupvae 
            # (and not the number we wish in the benchmark)
            average_all_cells = True,
        )

    def apply_deconvolution(self, to_deconvolve: pd.DataFrame):
        """
        """
        adata_to_deconvolve = create_anndata_pseudobulk(
            adata_obs=self.adata_obs, 
            adata_var_names=self.filtered_genes, 
            x=to_deconvolve.T.values,
        )
        latent_adata = self.mixupvi.get_latent_representation(
            adata_to_deconvolve, get_pseudobulk=False
        )
        deconvolution_results = use_nnls_method(
            latent_adata, self.adata_latent_signature
        )
        return deconvolution_results

def initialise_func(func_config: dict):
    """
    """
    target_path = func_config["_target_"]
    module_name, func_name = target_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    initialised_func = getattr(module, func_name)
    kwargs = {k: v for k, v in func_config.items() if k != "_target_"}
    return initialised_func, kwargs

def load_preprocessed_datasets(
    evaluation_datasets: list,
    train_dataset: str | None = None,
    n_variable_genes: int | None = None,
):
    """
    """
    data = {"datasets": {}}
    for evaluation_dataset in evaluation_datasets:
        logger.info(f"Loading dataset: {evaluation_dataset}...")
        data["datasets"][evaluation_dataset] = {}
        dataset_config = DATASETS[evaluation_dataset]
        initialised_func, kwargs = initialise_func(dataset_config)
        kwargs["n_variable_genes"] = n_variable_genes
        data["datasets"][evaluation_dataset]["dataset"] = initialised_func(**kwargs)

    if train_dataset is not None and train_dataset not in evaluation_datasets:
        logger.info(f"Loading train dataset: {train_dataset}...")
        data["datasets"][train_dataset] = {}
        dataset_config = DATASETS[train_dataset]
        initialised_func, kwargs = initialise_func(dataset_config)
        kwargs["n_variable_genes"] = n_variable_genes
        data["datasets"][train_dataset]["dataset"] = initialised_func(**kwargs)

    return data

def initialise_deconv_methods(
    deconv_methods,
    all_data,
    granularity: str,
    train_dataset: str,
    signature_matrices: list,
):
    """
    """
    deconv_methods_initialised = {}
    for deconv_method in deconv_methods:
        deconv_method_func, kwargs = initialise_func(DECONV_METHODS[deconv_method])
        if (deconv_method in MODEL_TO_FIT)==(deconv_method in SIGNATURE_MATRIX_MODELS):
            message = (
                "The codebase is not formatted yet to have a deconvolution method "
                "needing both to be fit and a user-provided signature matrix, or none "
                "of these two options. It needs one of these options only."
            )
            logger.error(message)
            raise NotImplementedError(message)
        if deconv_method in MODEL_TO_FIT:
            logger.info(f"Training deconvolution method {deconv_method}...")
            all_train_dset = all_data["datasets"][train_dataset]
            train_dset = all_train_dset["dataset"][
                all_train_dset[granularity]["Train index"]
            ]
            kwargs["adata_train"] = train_dset
            if "cell_type_group" in kwargs:
                # TODO: Ugly code because specific to MixUpVI only
                # More generally, improve to allow to pass other kwargs arguments
                kwargs["adata_train"].obs = kwargs["adata_train"].obs.rename(
                    {f"cell_types_grouped_{granularity}": "cell_types_grouped"},
                    axis = 1
                )
            deconv_method_initialised = deconv_method_func(**kwargs)
            deconv_methods_initialised[deconv_method] = deconv_method_initialised
        elif deconv_method in SIGNATURE_MATRIX_MODELS:
            for signature_matrix in signature_matrices:
                if SIGNATURE_TO_GRANULARITY[signature_matrix]==granularity:
                    kwargs["signature_matrix"] = all_data["signature_matrices"][
                        signature_matrix
                    ]
                    deconv_method_initialised = deconv_method_func(**kwargs)
                    deconv_methods_initialised[
                        f"deconv_method_{signature_matrix}"
                    ] = deconv_method_initialised
    
    logger.info("Initialisation of the deconvolution methods complete.")
    return deconv_methods_initialised

    
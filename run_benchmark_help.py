"""
"""

from __future__ import annotations

import anndata as ad
import constants
import importlib
import os
import pandas as pd
import scanpy as sc
from abc import abstractmethod
from functools import partial
from loguru import logger
from sklearn.linear_model import LinearRegression
from TAPE import Deconvolution
from TAPE.deconvolution import ScadenDeconvolution

from benchmark_utils import (
    create_anndata_pseudobulk,
    create_latent_signature,
    fit_destvi,
    fit_scvi,
    fit_mixupvi,
    plot_deconv_lineplot,
    plot_deconv_results,
    plot_deconv_results_group,
    preprocess_scrna,
)
from run_benchmark_constants import (
    CORRELATION_FUNCTIONS,
    DATASETS,
    DECONV_METHODS,
    EVALUATION_PSEUDOBULK_SAMPLINGS,
    GRANULARITY_TO_EVALUATION_DATASET,
    MODEL_TO_FIT,
    SIGNATURE_MATRIX_MODELS,
    SIGNATURE_TO_GRANULARITY,
    SINGLE_CELL_DATASETS,
)

def load_cti(n_variable_genes: int, **kwargs):
    """TODO: Right now, it's just a raw function to test the code.
    """
    adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
    adata = preprocess_scrna(adata,
                     keep_genes=n_variable_genes,
                     batch_key="donor_id")
    return {"dataset": adata}

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
    bulk_data = bulk_data.loc[facs_results.index]
    bulk_data.index = bulk_data.index.astype(str)
    facs_results.index = facs_results.index.astype(str)

    return {"dataset": bulk_data.T, "ground_truth": facs_results}

def use_nnls_method(to_deconvolve: pd.DataFrame, signature_matrix: pd.DataFrame):
    """TODO: Right now, it's just a raw function to test the code.
    """
    # Gene intersection with signature matrix
    gene_intersection = signature_matrix.index.intersection(to_deconvolve.index)
    to_deconvolve = to_deconvolve.loc[gene_intersection]
    signature_matrix = signature_matrix.loc[gene_intersection]
    # Run NNLS
    deconv = LinearRegression(positive=True).fit(
        signature_matrix, to_deconvolve
    )
    deconv_results = pd.DataFrame(
        deconv.coef_, index=to_deconvolve.columns, columns=signature_matrix.columns
    )
    deconv_results = deconv_results.div(
        deconv_results.sum(axis=1), axis=0
    )  # to sum up to 1
    
    return deconv_results

class AbstractDeconvolutionMethod:
    """TODO: Right now, it's just a raw class to test the code.
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
    """TODO: Right now, it's just a raw class to test the code.
    """
    def __init__(self, signature_matrix_name: str, signature_matrix: pd.DataFrame):
        self.signature_matrix_name = signature_matrix_name
        self.signature_matrix = signature_matrix
    
    def apply_deconvolution(self, to_deconvolve: ad.AnnData|pd.DataFrame):
        """
        """
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            to_deconvolve = pd.DataFrame(
                index=to_deconvolve.obs_names,
                columns=to_deconvolve.var_names,
                data=to_deconvolve.layers["counts"]
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
            use_mixupvi=False, # should be equal to use_mixupvi, but if True, 
            # then it averages as many cells as self.n_cells_per-pseudobulk from mixupvae 
            # (and not the number we wish in the benchmark)
            average_all_cells = True,
        )
        self.adata_latent_signature = pd.DataFrame(
            self.adata_latent_signature.X.T,
            index=self.adata_latent_signature.var_names,
            columns=self.adata_latent_signature.obs["cell type"]
        )

    def apply_deconvolution(self, to_deconvolve: ad.AnnData|pd.DataFrame):
        """
        """
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
                x=to_deconvolve.loc[self.filtered_genes].T.values
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
            data=latent_adata
        ).T
        deconvolution_results = use_nnls_method(
            latent_adata, self.adata_latent_signature
        )

        return deconvolution_results
    
class scVIMethod(AbstractDeconvolutionMethod):
    """
    """
    def __init__(
        self, 
        adata_train: ad.AnnData, 
        model_path: str = "", 
        save_model: bool = False, 
    ):
        self.filtered_genes = adata_train.var.index[
            adata_train.var["highly_variable"]
        ].tolist()
        adata_train = adata_train[:,self.filtered_genes]
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
            average_all_cells = True,
        )
        self.adata_latent_signature = pd.DataFrame(
            self.adata_latent_signature.X.T,
            index=self.adata_latent_signature.var_names,
            columns=self.adata_latent_signature.obs["cell type"]
        )

    def apply_deconvolution(self, to_deconvolve: ad.AnnData|pd.DataFrame):
        """
        """
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
                x=to_deconvolve.loc[self.filtered_genes].T.values
            )
        else:
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)
        
        latent_adata = self.scvi.get_latent_representation(
            to_deconvolve, get_pseudobulk=False
        )
        latent_adata = pd.DataFrame(
            index=obs_names,
            columns=self.adata_latent_signature.index,
            data=latent_adata
        ).T
        deconvolution_results = use_nnls_method(
            latent_adata, self.adata_latent_signature
        )

        return deconvolution_results

class DestVIMethod(AbstractDeconvolutionMethod):
    """
    """
    def __init__(
        self, 
        adata_train: ad.AnnData,
        adata_pseudobulk: ad.AnnData,
        model_path1: str = "",
        model_path2: str = "",
        cell_type_group: str = "cell_types_grouped",
        save_model: bool = False, 
    ):
        self.filtered_genes = adata_train.var.index[
            adata_train.var["highly_variable"]
        ].tolist()
        adata_train = adata_train[:,self.filtered_genes]
        self.adata_obs = adata_train.obs

        logger.debug("Fitting DestVI...")
        self.destvi = fit_destvi(
            adata=adata_train.copy(),
            adata_pseudobulk=adata_pseudobulk,
            model_path1=model_path1, 
            model_path2=model_path2, 
            cell_type_key=cell_type_group,
            save_model=save_model, 
        )

        logger.debug("Training over.")

    def apply_deconvolution(self, to_deconvolve: ad.AnnData|pd.DataFrame):
        """
        """
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
                x=to_deconvolve.loc[self.filtered_genes].T.values
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
        # TODO: is there somewhere to retrieve the cell types in destvi ?
        # deconvolution_results = pd.DataFrame(
        #     index=obs_names,
        #     columns=,
        #     data=deconvolution_results
        # )

        return deconvolution_results

class TAPEMethod(AbstractDeconvolutionMethod):
    """
    """
    def __init__(self, signature_matrix_name: str, signature_matrix: pd.DataFrame):
        self.signature_matrix_name = signature_matrix_name
        self.signature_matrix = signature_matrix
        
    def apply_deconvolution(self, to_deconvolve: ad.AnnData|pd.DataFrame):
        """
        """
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            to_deconvolve = pd.DataFrame(
                index=to_deconvolve.obs_names,
                columns=to_deconvolve.var_names,
                data=to_deconvolve.layers["counts"]
            ).T
        elif not isinstance(to_deconvolve, pd.DataFrame):
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)
        
        _, deconvolution_results = Deconvolution(
            self.signature.T, to_deconvolve,
            sep='\t', scaler='mms',
            datatype='counts', genelenfile=None,
            mode='overall', adaptive=True, variance_threshold=0.98,
            save_model_name=None,
            batch_size=128, epochs=128, seed=1
        )

        return deconvolution_results
    
class TAPEMethod(AbstractDeconvolutionMethod):
    """
    """
    def __init__(self, signature_matrix_name: str, signature_matrix: pd.DataFrame):
        self.signature_matrix_name = signature_matrix_name
        self.signature_matrix = signature_matrix
        
    def apply_deconvolution(self, to_deconvolve: ad.AnnData|pd.DataFrame):
        """
        """
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            to_deconvolve = pd.DataFrame(
                index=to_deconvolve.obs_names,
                columns=to_deconvolve.var_names,
                data=to_deconvolve.layers["counts"]
            ).T
        elif not isinstance(to_deconvolve, pd.DataFrame):
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)
        
        _, deconvolution_results = Deconvolution(
            self.signature.T, to_deconvolve,
            sep='\t', scaler='mms',
            datatype='counts', genelenfile=None,
            mode='overall', adaptive=True, variance_threshold=0.98,
            save_model_name=None,
            batch_size=128, epochs=128, seed=1
        )

        return deconvolution_results

class ScadenMethod(AbstractDeconvolutionMethod):
    """
    """
    def __init__(self, signature_matrix_name: str, signature_matrix: pd.DataFrame):
        self.signature_matrix_name = signature_matrix_name
        self.signature_matrix = signature_matrix
        
    def apply_deconvolution(self, to_deconvolve: ad.AnnData|pd.DataFrame):
        """
        """
        if isinstance(to_deconvolve, ad.AnnData):
            # Pseudobulks constructed from scRNAseq
            to_deconvolve = pd.DataFrame(
                index=to_deconvolve.obs_names,
                columns=to_deconvolve.var_names,
                data=to_deconvolve.layers["counts"]
            ).T
        elif not isinstance(to_deconvolve, pd.DataFrame):
            message = (
                "Data to deconvolve during inference can either be AnnData or DataFrame, "
                f"but here it is of type {type(to_deconvolve)}."
            )
            logger.error(message)
            raise ValueError(message)
        
        deconvolution_results = ScadenDeconvolution(self.signature.T,
                                            to_deconvolve,
                                            sep='\t',
                                            batch_size=128, epochs=128)

        return deconvolution_results

def initialize_func(func_config: dict):
    """
    """
    target_path = func_config["_target_"]
    module_name, func_name = target_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    initialized_func = getattr(module, func_name)
    kwargs = {k: v for k, v in func_config.items() if k != "_target_"}
    return initialized_func, kwargs

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
        initialized_func, kwargs = initialize_func(dataset_config)
        kwargs["n_variable_genes"] = n_variable_genes
        data["datasets"][evaluation_dataset] = initialized_func(**kwargs)

    if train_dataset is not None and train_dataset not in evaluation_datasets:
        logger.info(f"Loading train dataset: {train_dataset}...")
        data["datasets"][train_dataset] = {}
        dataset_config = DATASETS[train_dataset]
        initialized_func, kwargs = initialize_func(dataset_config)
        kwargs["n_variable_genes"] = n_variable_genes
        data["datasets"][train_dataset] = initialized_func(**kwargs)
    
    # In case one of the evaluation dataset is BULK_FACS, intersect the bulk and CTI gene lists
    if "BULK_FACS" in evaluation_datasets:
        logger.warning(
            "BULK_FACS is provided as one evaluation_dataset, therefore CTI will be intersected "
            "with the common genes between both datasets. To prevent this intersection for "
            "pseudobulk evaluations, remove BULK_FACS as evaluation_dataset."
        )
        # TODO (related to the warning above): The code prevents CTI pseudobulk evaluation to train on non-intersected genes, 
        # so the intersection should be done at training time
        # TODO: It should be done in an automatized way no matter the training single cell dataset (not CTI hard-coded like here)
        bulk_gene_list = data["datasets"]["BULK_FACS"]["dataset"].index
        cti_gene_list = data["datasets"]["CTI"]["dataset"].var_names
        cti_bulk_genes_intersection = list(set(bulk_gene_list).intersection(set(cti_gene_list)))
        data["datasets"]["CTI"]["dataset"] = data["datasets"]["CTI"]["dataset"][:, cti_bulk_genes_intersection]
        data["datasets"]["BULK_FACS"]["dataset"] = data["datasets"]["BULK_FACS"]["dataset"].loc[cti_bulk_genes_intersection]

    return data

def initialize_deconv_methods(
    deconv_methods: list,
    all_data: dict,
    granularity: str,
    train_dataset: str,
    signature_matrices: list,
):
    """
    """
    deconv_methods_initialized = {}
    for deconv_method in deconv_methods:
        deconv_method_func, kwargs = initialize_func(DECONV_METHODS[deconv_method])
        if (deconv_method in MODEL_TO_FIT)==(deconv_method in SIGNATURE_MATRIX_MODELS):
            message = (
                "The codebase is not formatted yet to have a deconvolution method "
                "needing both to be fit and a user-provided signature matrix, or none "
                "of these two options. It needs one of these options only."
            )
            logger.error(message)
            raise NotImplementedError(message)
        if deconv_method in MODEL_TO_FIT:
            logger.debug(f"Training deconvolution method {deconv_method}...")
            all_train_dset = all_data["datasets"][train_dataset]
            train_dset = all_train_dset["dataset"][
                all_train_dset[granularity]["Train index"]
            ]
            kwargs["adata_train"] = train_dset
            if "cell_type_group" in kwargs:
                kwargs["adata_train"].obs = kwargs["adata_train"].obs.rename(
                    {f"cell_types_grouped_{granularity}": "cell_types_grouped"},
                    axis = 1
                )
            if "adata_pseudobulk" in kwargs:
                train_pseudobulks = launch_evaluation_pseudobulk_samplings(
                    evaluation_pseudobulk_sampling="DIRICHLET", # "UNIFORM"
                    all_data=all_data,
                    evaluation_dataset=train_dataset,
                    granularity=granularity,
                    n_cells_per_evaluation_pseudobulk=100,
                    n_samples_evaluation_pseudobulk=500,
                )
                kwargs["adata_pseudobulk"] = train_pseudobulks["adata_pseudobulk_test_counts"]
            deconv_method_initialized = deconv_method_func(**kwargs)
            deconv_methods_initialized[deconv_method] = deconv_method_initialized
        elif deconv_method in SIGNATURE_MATRIX_MODELS:
            for signature_matrix in signature_matrices:
                if SIGNATURE_TO_GRANULARITY[signature_matrix]==granularity:
                    kwargs["signature_matrix_name"] = signature_matrix
                    kwargs["signature_matrix"] = all_data["signature_matrices"][
                        signature_matrix
                    ]
                    deconv_method_initialized = deconv_method_func(**kwargs)
                    deconv_methods_initialized[
                        f"{deconv_method}_{signature_matrix}"
                    ] = deconv_method_initialized
    
    logger.debug("Initialization of the deconvolution methods complete.")
    return deconv_methods_initialized

def launch_evaluation_pseudobulk_samplings(
    evaluation_pseudobulk_sampling: list,
    all_data: dict,
    evaluation_dataset: str,
    granularity: str,
    n_cells_per_evaluation_pseudobulk: int,
    n_samples_evaluation_pseudobulk: int,
):
    """
    """
    evaluation_pseudobulk_samplings_func, kwargs = initialize_func(
        EVALUATION_PSEUDOBULK_SAMPLINGS[evaluation_pseudobulk_sampling]
    )
    all_test_dset = all_data["datasets"][evaluation_dataset]
    test_dset = all_test_dset["dataset"][
        all_test_dset[granularity]["Test index"]
    ]
    kwargs["adata"] = test_dset
    if "cell_type_group" in kwargs:
        kwargs["adata"].obs = kwargs["adata"].obs.rename(
            {f"cell_types_grouped_{granularity}": "cell_types_grouped"},
            axis = 1
        )
    if "n_cells" in kwargs and "n_sample" in kwargs:
        kwargs["n_cells"] = n_cells_per_evaluation_pseudobulk
        kwargs["n_sample"] = n_samples_evaluation_pseudobulk
        message = (
            f"Creating pseudobulks composed of {n_samples_evaluation_pseudobulk}"
            f" samples with {n_cells_per_evaluation_pseudobulk} cells using the "
            f"{evaluation_pseudobulk_sampling} method..."
        )
    else:
        message = (
            f"Creating pseudobulks using the {evaluation_pseudobulk_sampling} method..."
        )
    logger.debug(message)

    pseudobulks = evaluation_pseudobulk_samplings_func(**kwargs)

    return pseudobulks

def save_deconvolution_results(deconv_results: dict, experiment_path):
    """
    """
    for granularity, sub_dict in deconv_results.items():
        granularity_dir = os.path.join(experiment_path, str(granularity))
        os.makedirs(granularity_dir, exist_ok=True)
        
        for key, value in sub_dict.items():
            if isinstance(value, dict):
                save_deconvolution_results(value, os.path.join(granularity_dir, key))
            else:                
                value.to_csv(os.path.join(granularity_dir, f"{key}.csv"))
    
def compute_benchmark_correlations(deconv_results: dict, correlation_type: str):
    """
    """
    compute_correlation_fn, _ = initialize_func(CORRELATION_FUNCTIONS[correlation_type])
    all_results = []
    for granularity, level1 in deconv_results.items():
        evaluation_dataset = GRANULARITY_TO_EVALUATION_DATASET[granularity]
        if evaluation_dataset in SINGLE_CELL_DATASETS:
            for sampling_method, level2 in level1.items():
                for num_cells, level3 in level2.items():
                    for deconv_method, level4 in level3.items():
                        if level4["deconvolution_results"].isna().any().any():
                            # In this case, no correlation is computed
                            # TODO: allow computation on non-NaN samples
                            logger.warning(
                                f"Deconv results for the {deconv_method} method for the "
                                f"granularity {granularity} for the sampling method {sampling_method} "
                                f"for the number of cells {num_cells} contains NaN values, so "
                                "correlation computation will be skipped there."
                            )
                        else:
                            df_corr = compute_correlation_fn(level4["deconvolution_results"], level4["ground_truth"])
                            df_corr["deconv_method"] = deconv_method
                            df_corr["num_cells"] = num_cells
                            df_corr["sampling_method"] = sampling_method
                            df_corr["granularity"] = granularity
                            df_corr["correlation_type"] = correlation_type
                            all_results.append(df_corr)
        else:
            for deconv_method, level2 in level1.items():
                if level2["deconvolution_results"].isna().any().any():
                    # In this case, no correlation is computed
                    # TODO: allow computation on non-NaN samples
                    logger.warning(
                        f"Deconv results for the {deconv_method} method for the "
                        f"granularity {granularity} contains NaN values, so "
                        "correlation computation will be skipped there."
                    )

                else:
                    df_corr = compute_correlation_fn(level2["deconvolution_results"], level2["ground_truth"])
                    df_corr["deconv_method"] = deconv_method
                    df_corr["granularity"] = granularity
                    df_corr["correlation_type"] = correlation_type
                    all_results.append(df_corr)

    return pd.concat(all_results, ignore_index=True)

def plot_benchmark_correlations(df_all_correlations, save_path: str, save: bool = True):
    """Plot benchmark correlations, and save them by default.
    """
    def _get_groups(df, groupby_col):
        """Returns grouped DataFrames if groupby_col exists and is not empty, else returns a list with the original df."""
        if groupby_col in df.columns and df[groupby_col].notna().any():
            return [group for _, group in df.groupby(groupby_col)]
        return [df]
    
    plot_func_map = {
        "sample_wise_correlation": plot_deconv_results,
        "cell_type_wise_correlation": plot_deconv_results_group
    }

    for granularity in df_all_correlations.granularity.unique():
        df_all_correlations_temp = df_all_correlations[df_all_correlations["granularity"] == granularity]
        if "num_cells" in df_all_correlations_temp.columns and df_all_correlations_temp.num_cells.dropna().nunique() > 1:
            # Multiple num_cells were computed
            df_to_plot = df_all_correlations_temp[df_all_correlations_temp["correlation_type"] == "sample_wise_correlation"]
            for group in _get_groups(df_to_plot, "sampling_method"):
                plot_deconv_lineplot(group, save=save, save_path=save_path)
        else:
            # One pseudobulk num_cells or bulk
            for correlation_type in df_all_correlations_temp["correlation_type"].unique():
                df_to_plot = df_all_correlations_temp[df_all_correlations_temp["correlation_type"] == correlation_type]
                plot_func = plot_func_map.get(correlation_type, plot_deconv_results)  # Default to `plot_deconv_results`
                for group in _get_groups(df_to_plot, "sampling_method"):
                    plot_func(group, save=save, save_path=save_path)


"""Utility functions for sanity checks."""

import pandas as pd
from loguru import logger

from benchmark_utils import (
    create_latent_signature,
    perform_nnls,
    perform_latent_deconv,
    compute_correlations,
    compute_group_correlations,
)

import scvi
import anndata as ad
from typing import List, Dict, Union
from TAPE import Deconvolution
from TAPE.deconvolution import ScadenDeconvolution


# Helper function to create results dataframe
def melt_df(deconv_results):
    """Melt the deconv results for seaborn"""
    deconv_results_melted = pd.melt( # melt the matrix for seaborn
            deconv_results.T.reset_index(),
            id_vars="index",
            var_name="Cell type",
            value_name="Estimated Fraction",
        ).rename({"index": "Cell type predicted"}, axis=1)
    deconv_results_melted_methods_temp = deconv_results_melted.loc[
        deconv_results_melted["Cell type predicted"] == deconv_results_melted["Cell type"]
    ].copy()
    return deconv_results_melted_methods_temp


def run_sanity_check(
    adata_train: ad.AnnData,
    adata_pseudobulk_test_counts: ad.AnnData,
    adata_pseudobulk_test_rc: ad.AnnData,
    all_adata_samples_test: List[ad.AnnData],
    df_proportions_test: pd.DataFrame,
    signature: pd.DataFrame,
    generative_models : Dict[str, Union[scvi.model.SCVI,
                                        scvi.model.CondSCVI,
                                        scvi.model.DestVI,
                                        scvi.model.MixUpVI]],
    baselines: List[str],
):
    """Run sanity check 2/3 pseudobulk generated from different strategies
    sampling of cell types.

    If the `generative_models` dictionnary is empty, only the baselines will be run.

    Parameters
    ----------
    adata_train: ad.AnnData
        scRNAseq training dataset.
    adata_pseudobulk_test_counts: ad.AnnData
        pseudobulk RNA seq test dataset.
    adata_pseudobulk_test_rc: ad.AnnData
        pseudobulk RNA seq test dataset (relative counts).
    signature: pd.DataFrame
        Signature matrix.
    generative_models: Dict[str, scvi.model]
        Dictionnary of generative models.
    baselines: List[str]
        List of baseline methods to run.

    Returns
        pd.DataFrame
            Melted dataframe of the deconvolution results.
    """
    logger.info("Running sanity check...")

    # create dataframe with different methods
    df_test_correlations = pd.DataFrame(
        index=adata_pseudobulk_test_counts.obs_names,
        columns=baselines + list(generative_models.keys())
    )
    df_test_group_correlations = pd.DataFrame(
        index=df_proportions_test.columns,
        columns=baselines + list(generative_models.keys())
    )

    # 1. Linear regression (NNLS)
    if "nnls" in baselines:
        deconv_results = perform_nnls(signature,
                                      adata_pseudobulk_test_rc[:, signature.index])
        correlations = compute_correlations(deconv_results, df_proportions_test)
        group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
        df_test_correlations.loc[:, "nnls"] = correlations.values
        df_test_group_correlations.loc[:, "nnls"] = group_correlations.values

    # get all genes for wich adata_train.var["highly_variable"] is True
    filtered_genes = adata_train.var.index[adata_train.var["highly_variable"]].tolist()
    intersection = list(set(signature.index).intersection(set(filtered_genes)))

    pseudobulk_test_df = pd.DataFrame(
        adata_pseudobulk_test_rc[:, intersection].X,
        index=adata_pseudobulk_test_rc.obs_names,
        columns=intersection,
    )
    # 2. TAPE
    if "TAPE" in baselines:
        _, deconv_results = \
        Deconvolution(signature.loc[intersection].T, pseudobulk_test_df,
                  sep='\t', scaler='mms',
                  datatype='counts', genelenfile=None,
                  mode='overall', adaptive=True, variance_threshold=0.98,
                  save_model_name=None,
                  batch_size=128, epochs=128, seed=1)
        correlations = compute_correlations(deconv_results, df_proportions_test)
        group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
        df_test_correlations.loc[:, "TAPE"] = correlations.values
        df_test_group_correlations.loc[:, "TAPE"] = group_correlations.values
    ## 3. Scaden
    if "Scaden" in baselines:
        deconv_results = ScadenDeconvolution(signature.loc[intersection].T,
                                            pseudobulk_test_df,
                                            sep='\t',
                                            batch_size=128, epochs=128)
        correlations = compute_correlations(deconv_results, df_proportions_test)
        group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
        df_test_correlations.loc[:, "Scaden"] = correlations.values
        df_test_group_correlations.loc[:, "Scaden"] = group_correlations.values
    if generative_models == {}:
        return df_test_group_correlations

    # 4. Generative models
    for model in generative_models.keys():
        if model == "DestVI":
            deconv_results = generative_models[model].get_proportions(adata_pseudobulk_test_counts)
            deconv_results.drop(["noise_term"],
                            axis=1,
                            inplace=True)
            deconv_results = deconv_results.loc[:, df_proportions_test.columns]
            correlations = compute_correlations(deconv_results, df_proportions_test)
            group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
            df_test_correlations.loc[:, "DestVI"] = correlations.values
            df_test_group_correlations.loc[:, "DestVI"] = group_correlations.values
        else:
            use_mixupvi=False
            if model == "MixupVI":
                use_mixupvi=True
            adata_latent_signature = create_latent_signature(
                adata=adata_train[:,filtered_genes],
                model=generative_models[model],
                use_mixupvi=False, # should be equal to use_mixupvi, but if True,
                # then it averages as many cells as self.n_cells_per-pseudobulk from mixupvae
                # (and not the number we wish in the benchmark)
                average_all_cells = True,
                sc_per_pseudobulk=10000,
            )
            deconv_results = perform_latent_deconv(
                adata_pseudobulk=adata_pseudobulk_test_counts[:,filtered_genes],
                all_adata_samples=all_adata_samples_test,
                filtered_genes=filtered_genes,
                use_mixupvi=False, # see comment above
                adata_latent_signature=adata_latent_signature,
                model=generative_models[model],
            )
            correlations = compute_correlations(deconv_results, df_proportions_test)
            group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
            df_test_correlations.loc[:, model] = correlations.values
            df_test_group_correlations.loc[:, model] = group_correlations.values

    return df_test_correlations, df_test_group_correlations



## NOT USED
# def run_purified_sanity_check(
#     adata_train: ad.AnnData,
#     adata_pseudobulk_test_counts: ad.AnnData,
#     adata_pseudobulk_test_rc: ad.AnnData,
#     filtered_genes: list,
#     signature: pd.DataFrame,
#     generative_models : Dict[str, Union[scvi.model.SCVI,
#                                         scvi.model.CondSCVI,
#                                         scvi.model.DestVI,
#                                         scvi.model.MixUpVI]],
#     baselines: List[str],
# ):
#     """Run sanity check 1 on purified cell types.

#     Sanity check 1 is an "easy" deconvolution task where the pseudobulk test dataset
#     is composed of purified cell types. Thus the groundtruth proportion is 1 for each
#     sample in the dataset.

#     If the `generative_models` dictionnary is empty, only the baselines will be run.

#     Parameters
#     ----------
#     adata_train: ad.AnnData
#         scRNAseq training dataset.
#     adata_pseudobulk_test_counts: ad.AnnData
#         pseudobulk RNA seq test dataset (counts).
#     adata_pseudobulk_test_rc: ad.AnnData
#         pseudobulk RNA seq test dataset (relative counts).
#     signature: pd.DataFrame
#         Signature matrix.
#     generative_models: Dict[str, scvi.model]
#         Dictionnary of generative models.
#     baselines: List[str]
#         List of baseline methods to run.

#     Returns
#         pd.DataFrame
#             Melted dataframe of the deconvolution results.
#     """
#     logger.info("Running sanity check...")

#     # 1. Baselines
#     deconv_results_melted_methods = pd.DataFrame(columns=["Cell type predicted", "Cell type", "Estimated Fraction", "Method"])
#     ## NNLS
#     if "nnls" in baselines:
#         deconv_results = perform_nnls(signature, adata_pseudobulk_test_rc[:, signature.index])
#         deconv_results_melted_methods_tmp = melt_df(deconv_results)
#         deconv_results_melted_methods_tmp["Method"] = "nnls"
#         deconv_results_melted_methods = pd.concat(
#             [deconv_results_melted_methods, deconv_results_melted_methods_tmp]
#         )

#     # Pseudobulk Dataframe for TAPE and Scaden
#     intersection = set(signature.index).intersection(set(filtered_genes))
#     pseudobulk_test_df = pd.DataFrame(
#         adata_pseudobulk_test_rc[:, intersection].X,
#         index=adata_pseudobulk_test_rc.obs_names,
#         columns=intersection,
#     )
#     ## TAPE
#     if "TAPE" in baselines:
#         _, deconv_results = \
#         Deconvolution(signature.T, pseudobulk_test_df,
#                   sep='\t', scaler='mms',
#                   datatype='counts', genelenfile=None,
#                   mode='overall', adaptive=True, variance_threshold=0.98,
#                   save_model_name=None,
#                   batch_size=128, epochs=128, seed=1)
#         deconv_results_melted_methods_tmp = melt_df(deconv_results)
#         deconv_results_melted_methods_tmp["Method"] = "TAPE"
#         deconv_results_melted_methods = pd.concat(
#             [deconv_results_melted_methods, deconv_results_melted_methods_tmp]
#         )
#     ## Scaden
#     if "Scaden" in baselines:
#         deconv_results = ScadenDeconvolution(signature.T,
#                                             pseudobulk_test_df,
#                                             sep='\t',
#                                             batch_size=128, epochs=128)
#         deconv_results_melted_methods_tmp = melt_df(deconv_results)
#         deconv_results_melted_methods_tmp["Method"] = "Scaden"
#         deconv_results_melted_methods = pd.concat(
#             [deconv_results_melted_methods, deconv_results_melted_methods_tmp]
#         )

#     if generative_models == {}:
#         return deconv_results_melted_methods

#     ### 2. Generative models
#     for model in generative_models.keys():
#         if model == "DestVI":
#             continue
#             # DestVI is not used for Sanity check 1 - not enough
#             # samples to fit the stLVM.
#             # deconv_results = generative_models[model].get_proportions(adata_pseudobulk_test)
#             # deconv_results = deconv_results.drop(["noise_term"],
#             #                                      axis=1,
#             #                                      inplace=True)
#             # deconv_results_melted_methods_tmp = melt_df(deconv_results)
#             # deconv_results_melted_methods_tmp["Method"] = model
#             # deconv_results_melted_methods = pd.concat(
#             #     [deconv_results_melted_methods, deconv_results_melted_methods_tmp]
#             # )
#         else:
#             adata_latent_signature = create_latent_signature(
#                 adata=adata_train[:,filtered_genes],
#                 model=generative_models[model],
#                 average_all_cells = True,
#                 sc_per_pseudobulk=3000,
#             )
#             deconv_results = perform_latent_deconv(
#                 adata_pseudobulk=adata_pseudobulk_test_counts[:,filtered_genes],
#                 adata_latent_signature=adata_latent_signature,
#                 model=generative_models[model],
#             )
#             deconv_results_melted_methods_tmp = melt_df(deconv_results)
#             deconv_results_melted_methods_tmp["Method"] = model
#             deconv_results_melted_methods = pd.concat(
#                 [deconv_results_melted_methods, deconv_results_melted_methods_tmp]
#             )
#     return deconv_results_melted_methods

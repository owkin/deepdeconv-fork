"""Utility functions for sanity checks."""

import pandas as pd
from loguru import logger

from benchmark_utils import (
    create_latent_signature,
    perform_nnls,
    perform_latent_deconv,
    compute_correlations,
    compute_group_correlations,
    melt_df,
)

import scvi
import anndata as ad
from typing import List, Dict
from TAPE import Deconvolution
from TAPE.deconvolution import ScadenDeconvolution


def run_incompatible_value_checks(
    pseudo_bulk, loss_computation, use_batch_norm, mixup_penalty, gene_likelihood
):
    """Check the values of the categorical variables to run MixUpVI are compatible.
    The first 4 checks will only be relevant when pseudobulk will not be computed both
    in encoder and decoder (right now, computed in both). Until then, use_batch_norm
    should be None.
    """
    if (
        pseudo_bulk == "pre_encoded"
        and loss_computation == "latent_space"
        and use_batch_norm in ["encoder", "both"]
    ):
        raise ValueError(
            "MixUpVI cannot use batch normalization there, as the batch size of pseudobulk is 1."
        )
    elif (
        pseudo_bulk == "pre_encoded"
        and loss_computation == "reconstructed_space"
        and use_batch_norm != "none"
    ):
        raise ValueError(
            "MixUpVI cannot use batch normalization there, as the batch size of pseudobulk is 1."
        )
    elif pseudo_bulk == "post_inference" and loss_computation == "latent_space":
        raise ValueError(
            "Pseudo bulk needs to be pre-encoded to compute the MixUp loss in the latent space."
        )
    elif (
        pseudo_bulk == "post_inference"
        and loss_computation == "reconstructed_space"
        and use_batch_norm in ["decoder", "both"]
    ):
        raise ValueError(
            "MixUpVI cannot use batch normalization there, as the batch size of pseudobulk is 1."
        )
    if (
        mixup_penalty == "kl"
        and loss_computation != "latent_space"
        and gene_likelihood == "zinb"
    ):
        raise NotImplementedError(
            "The KL divergence between ZINB distributions for the MixUp loss is not "
            "implemented."
        )


def run_categorical_value_checks(
    cell_group,
    cont_cov,
    encode_covariates,
    encode_cont_covariates,
    use_batch_norm,
    signature_type,
    loss_computation,
    pseudo_bulk,
    mixup_penalty,
    dispersion,
    gene_likelihood,
):
    """Check the values and types of the categorical variables to run MixUpVI."""
    assert isinstance(cell_group, str), "CELL_GROUP should be of type string"
    assert (
        isinstance(cont_cov, list) or cont_cov == None
    ), "CONT_COV should be None or type list"
    assert isinstance(
        encode_covariates, bool
    ), "ENCODE_COVARIATES should be of type bool"
    assert isinstance(
        encode_cont_covariates, bool
    ), "ENCODE_CONT_COVARIATES should be of type bool"
    assert isinstance(use_batch_norm, str), "BATCH_NORM should be of type string"
    assert isinstance(signature_type, str), "SIGNATURE_TYPE should be of type string"
    assert isinstance(
        loss_computation, str
    ), "LOSS_COMPUTATION should be of type string"
    assert isinstance(pseudo_bulk, str), "PSEUDO_BULK should be of type string"
    assert isinstance(mixup_penalty, str), "MIXUP_PENALTY should be of type string"
    assert isinstance(dispersion, str), "DISPERSION should be of type string"
    assert isinstance(gene_likelihood, str), "GENE_LIKELIHOOD should be of type string"
    if cell_group not in [
        "primary_groups",
        "precise_groups",
        "updated_granular_groups",
        "cell_types_grouped",
    ]:
        raise NotImplementedError(
            "For now, the following cell category granularities are implemented: "
            "['primary_groups', 'precise_groups', 'updated_granular_groups']"
        )
    if encode_covariates:
        raise NotImplementedError(
            "For now, MixUpVI only uses cell types as categorical covariates without encoding them."
        )
    if use_batch_norm not in ["encoder", "decoder", "none", "both"]:
        raise ValueError(
            "Batch normalization can only be part of ['encoder', 'decoder', 'none', 'both']."
        )
    if signature_type not in ["pre_encoded", "post_inference"]:
        raise ValueError(
            "Signature type can only be part of ['pre_encoded', 'post_inference']."
        )
    if loss_computation not in ["latent_space", "reconstructed_space"]:
        raise ValueError(
            "Loss computation can only be part of ['latent_space', 'reconstructed_space']."
        )
    if pseudo_bulk not in ["pre_encoded", "post_inference"]:
        raise ValueError(
            "Pseudo bulk computation can only be part of ['pre_encoded', 'post_inference']."
        )
    if mixup_penalty not in ["l2", "kl"]:
        raise ValueError("Mixup penalty can only be part of ['l2', 'kl'].")
    if dispersion not in ["gene", "gene_cell"]:
        raise ValueError(
            "The dispersion parameter can only be part of ['gene', 'gene_cell'], "
            "not gene-label nor gene-batch because categorical covariates don't make "
            "sense for pseudobulk."
        )
    if gene_likelihood not in ["zinb", "nb", "poisson"]:
        raise ValueError(
            "The dispersion parameter can only be part of ['zinb', 'nb', 'poisson']."
        )


def run_purified_sanity_check(
    adata_train: ad.AnnData,
    adata_pseudobulk_test: ad.AnnData,
    signature: pd.DataFrame,
    intersection: List[str],
    generative_models : Dict[str, scvi.model],
    baselines: List[str],
):
    """Run sanity check 1 on purified cell types.

    Sanity check 1 is an "easy" deconvolution task where the pseudobulk test dataset
    is composed of purified cell types. Thus the groundtruth proportion is 1 for each
    sample in the dataset.

    If the `generative_models` dictionnary is empty, only the baselines will be run.

    Parameters
    ----------
    adata_train: ad.AnnData
        scRNAseq training dataset.
    adata_pseudobulk_test: ad.AnnData
        pseudobulk RNA seq test dataset.
    signature: pd.DataFrame
        Signature matrix.
    intersection: List[str]
        List of genes in common between the signature and the test dataset.
    generative_models: Dict[str, scvi.model]
        Dictionnary of generative models.
    baselines: List[str]
        List of baseline methods to run.

    Returns
        pd.DataFrame
            Melted dataframe of the deconvolution results.
    """
    logger.info("Running sanity check...")

    ### 1. Baselines
    if "nnls" in baselines:
        deconv_results = perform_nnls(signature, adata_pseudobulk_test[:, intersection])
    deconv_results_melted = pd.melt( # melt the matrix for seaborn
        deconv_results.T.reset_index(),
        id_vars="index",
        var_name="Cell type",
        value_name="Estimated Fraction",
    ).rename({"index": "Cell type predicted"}, axis=1)
    if generative_models == {}:
        return deconv_results_melted

    # create dataframe with different methods
    deconv_results_melted_methods = deconv_results_melted.loc[
        deconv_results_melted["Cell type predicted"] == deconv_results_melted["Cell type"]
    ].copy()
    deconv_results_melted_methods["Method"] = "nnls"


def run_sanity_check(
    adata_train,
    adata_pseudobulk_test,
    df_proportions_test,
    signature,
    intersection,
    scvi_model,
    mixupvi_model,
    only_fit_baseline_nnls,
):
    """Run one of sanity checks 2 and 3.
    Possibility to only fit NNLS with only_fit_baseline_nnls==True
    TODO add other models such as random proportions, destvi, tape..."""
    logger.info("Running sanity check...")
     # create dataframe with different methods
    df_test_correlations = pd.DataFrame(
        index=adata_pseudobulk_test.obs_names, columns=["scVI", "MixupVI", "NNLS"]
    )  # TODO "Random", "DestVI"]
    df_test_group_correlations = pd.DataFrame(
        index=df_proportions_test.columns, columns=["scVI", "MixupVI", "NNLS"]
    )  # TODO "Random", "DestVI"]

    ### Random proportions
    # TODO:
    #


    ### Linear regression (NNLS)
    deconv_results = perform_nnls(signature, adata_pseudobulk_test[:, intersection])
    correlations = compute_correlations(deconv_results, df_proportions_test)
    group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
    df_test_correlations.loc[:, "NNLS"] = correlations.values
    df_test_group_correlations.loc[:, "NNLS"] = group_correlations.values
    if only_fit_baseline_nnls:
        return df_test_correlations, df_test_group_correlations

    ### scVI
    adata_latent_signature = create_latent_signature(
        adata=adata_train,
        model=scvi_model,
        average_all_cells = True,
        sc_per_pseudobulk=3000,
    )
    deconv_results = perform_latent_deconv(
        adata_pseudobulk=adata_pseudobulk_test,
        adata_latent_signature=adata_latent_signature,
        model=scvi_model,
    )
    correlations = compute_correlations(deconv_results, df_proportions_test)
    group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
    df_test_correlations.loc[:, "scVI"] = correlations.values
    df_test_group_correlations.loc[:, "scVI"] = group_correlations.values

    ### MixupVI
    adata_latent_signature = create_latent_signature(
        adata=adata_train,
        model=mixupvi_model,
        average_all_cells = True,
        sc_per_pseudobulk=3000,
    )
    deconv_results = perform_latent_deconv(
        adata_pseudobulk=adata_pseudobulk_test,
        adata_latent_signature=adata_latent_signature,
        model=mixupvi_model
    )
    correlations = compute_correlations(deconv_results, df_proportions_test)
    group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
    df_test_correlations.loc[:, "MixupVI"] = correlations.values
    df_test_group_correlations.loc[:, "MixupVI"] = group_correlations.values

    ### TODO DestVI
    # deconv_results = destvi_model.get_proportions(adata_pseudobulk, deterministic=True)
    # correlations = compute_correlations(deconv_results, df_proportions_test)
    # group_correlations = compute_group_correlations(deconv_results, df_proportions_test)
    # df_predicted_proportions.loc[:, "DestVI"] = deconv_results
    # df_test_correlations.loc[:, "DestVI"] = correlations.values
    # df_test_group_correlations.loc[:, "DestVI"] = group_correlations.values

    return df_test_correlations, df_test_group_correlations

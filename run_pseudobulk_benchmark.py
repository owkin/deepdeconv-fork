"""Pseudobulk benchmark."""
# %%
import scanpy as sc
import pandas as pd
import anndata as ad
import warnings
from loguru import logger
from sklearn.linear_model import LinearRegression

from constants import (
    BENCHMARK_DATASET,
    SIGNATURE_CHOICE,
    BENCHMARK_CELL_TYPE_GROUP,
    SAVE_MODEL,
    N_GENES,
    N_SAMPLES,
    GENERATIVE_MODELS,
    BASELINES,
    N_CELLS,
    COMPUTE_SC_RESULTS_WHEN_FACS,
)

from benchmark_utils import (
    preprocess_scrna,
    create_purified_pseudobulk_dataset,
    create_uniform_pseudobulk_dataset,
    create_dirichlet_pseudobulk_dataset,
    fit_scvi,
    fit_destvi,
    fit_mixupvi,
    create_signature,
    add_cell_types_grouped,
    run_purified_sanity_check,
    run_sanity_check,
    create_latent_signature,
    compute_correlations,
    compute_group_correlations,
    plot_purified_deconv_results,
    plot_deconv_results,
    plot_deconv_results_group,
    plot_deconv_lineplot,
)
from benchmark_utils.dataset_utils import create_anndata_pseudobulk

# %% Load scRNAseq dataset
logger.info(f"Loading single-cell dataset: {BENCHMARK_DATASET} ...")
if BENCHMARK_DATASET == "TOY":
    raise NotImplementedError(
        "For now, the toy dataset cannot be used to run the benchmark because no "
        "signature has intersections with its genes, and no train/test split csv exists"
    )
    # adata = scvi.data.heart_cell_atlas_subsampled()
    # preprocess_scrna(adata, keep_genes=1200)
elif BENCHMARK_DATASET == "CTI":
    adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
    adata, filtered_genes = preprocess_scrna(adata,
                     keep_genes=N_GENES,
                     batch_key="donor_id")
elif BENCHMARK_DATASET == "CTI_RAW":
    warnings.warn("The raw data of this adata is on adata.raw.X, but the normalised "
                  "adata.X will be used here")
    adata = sc.read("/home/owkin/data/cross-tissue/omics/raw/local.h5ad")
    adata, filtered_genes = preprocess_scrna(adata,
                     keep_genes=N_GENES,
                     batch_key="donor_id",
    )
elif BENCHMARK_DATASET == "CTI_PROCESSED":
    # Load processed for speed-up (already filtered, normalised, etc.)
    raise NotImplementedError(
        "Not possible to use a CTI_PROCESSED dataset because we would need the "
        "not-filtered adata to be processed as well. To solve: separate the "
        "preprocessing function between normalization and filtering parts."
    )
    # adata_filtered = sc.read(f"/home/owkin/data/cti_data/processed/cti_processed_{N_GENES}.h5ad")

# %% load signature
logger.info(f"Loading signature matrix: {SIGNATURE_CHOICE} | {BENCHMARK_CELL_TYPE_GROUP}...")
signature = create_signature(signature_type=SIGNATURE_CHOICE)

# %% add cell types groups and split train/test
adata, train_test_index = add_cell_types_grouped(adata, BENCHMARK_CELL_TYPE_GROUP)
adata_train = adata[train_test_index["Train index"]]
adata_test = adata[train_test_index["Test index"]]

# %% Create and train generative models
generative_models = {}
if GENERATIVE_MODELS != []:
    adata_train = adata_train.copy()
    adata_test = adata_test.copy()
    # 1. scVI
    if "scVI" in GENERATIVE_MODELS:
        logger.info("Fit scVI ...")
        model_path = f"project/models/{BENCHMARK_DATASET}_scvi.pkl"
        scvi_model = fit_scvi(adata_train[:,filtered_genes].copy(),
                              model_path,
                              save_model=SAVE_MODEL)
        generative_models["scVI"] = scvi_model
    # 2. DestVI
    if "DestVI" in GENERATIVE_MODELS:
        logger.info("Fit DestVI ...")
        # DestVI is only used in sanity check 2
        # Uniform
        # adata_pseudobulk_train_counts, adata_pseudobulk_train_rc, df_proportions_train = create_uniform_pseudobulk_dataset(
        #     adata_train, n_sample = N_SAMPLES, n_cells = N_CELLS,
        # )
        # Dirichlet
        adata_pseudobulk_train_counts, adata_pseudobulk_train_rc, df_proportions_test = create_dirichlet_pseudobulk_dataset(
            adata_train[:,filtered_genes].copy(), prior_alphas = None, n_sample = N_SAMPLES,
        )

        model_path_1 = f"project/models/{BENCHMARK_DATASET}_condscvi.pkl"
        model_path_2 = f"project/models/{BENCHMARK_DATASET}_destvi.pkl"
        condscvi_model , destvi_model= fit_destvi(adata_train[:,filtered_genes].copy(),
                                                adata_pseudobulk_train_counts,
                                                model_path_1,
                                                model_path_2,
                                                cell_type_key="cell_types_grouped",
                                                save_model=SAVE_MODEL)
        # generative_models["CondscVI"] = condscvi_model
        generative_models["DestVI"] = destvi_model

    # 3. MixupVI
    if "MixUpVI" in GENERATIVE_MODELS:
        logger.info("Train mixupVI ...")
        model_path = f"project/models/{BENCHMARK_DATASET}_{BENCHMARK_CELL_TYPE_GROUP}_{N_GENES}_mixupvi.pkl"
        mixupvi_model = fit_mixupvi(adata_train[:,filtered_genes].copy(),
                                    model_path,
                                    cell_type_group="cell_types_grouped",
                                    save_model=SAVE_MODEL,
                                    )
        generative_models["MixupVI"] = mixupvi_model

# %% FACS

if BENCHMARK_CELL_TYPE_GROUP == "FACS_1st_level_granularity":
    logger.info("Computing FACS results...")

    # Load data
    facs_results = pd.read_csv(
        "/home/owkin/project/bulk_facs/240214_majorCelltypes.csv", index_col=0
    ).drop(["No.B.Cells.in.Live.Cells","NKT.Cells.in.Live.Cells"],axis=1).set_index("Sample")
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
        "gene_counts20230103_batch1-5_all_cleaned-TPMnorm-allpatients.tsv"
        ), 
        sep="\t",
        index_col=0
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


    ### Most of the following is repeated from the sanity checks fct, so move this code there
    df_test_correlations = pd.DataFrame(
        index=bulk_data.columns,
        columns=["nnls", "MixUpVI"]
    )
    df_test_group_correlations = pd.DataFrame(
        index=facs_results.columns,
        columns=["nnls", "MixUpVI"]
    )
    
    # NNLS
    deconv = LinearRegression(positive=True).fit(
        signature, bulk_data.loc[signature.index]
    )
    deconv_results = pd.DataFrame(
        deconv.coef_, index=bulk_data.columns, columns=signature.columns
    )
    deconv_results = deconv_results.div(
        deconv_results.sum(axis=1), axis=0
    )  # to sum up to 1
    correlations = compute_correlations(deconv_results, facs_results)
    group_correlations = compute_group_correlations(deconv_results, facs_results)
    df_test_correlations.loc[:, "nnls"] = correlations.values
    df_test_group_correlations.loc[:, "nnls"] = group_correlations.values

    # MixUpVI
    bulk_mixupvi = bulk_data.loc[filtered_genes]
    model = "MixupVI"
    adata_latent_signature = create_latent_signature(
        adata=adata_train[:,filtered_genes],
        model=generative_models[model],
        use_mixupvi=False, # should be equal to use_mixupvi, but if True, 
        # then it averages as many cells as self.n_cells_per-pseudobulk from mixupvae 
        # (and not the number we wish in the benchmark)
        average_all_cells = True,
    )

    adata_bulk = create_anndata_pseudobulk(
        adata=adata_train[:,filtered_genes], x=bulk_mixupvi.T.values
    )
    latent_bulk = generative_models[model].get_latent_representation(
        adata_bulk, get_pseudobulk=False
    )
    deconv = LinearRegression(positive=True).fit(adata_latent_signature.X.T,
                                                 latent_bulk.T)
    deconv_results = pd.DataFrame(
        deconv.coef_, index=bulk_data.columns, columns=signature.columns
    )
    deconv_results = deconv_results.div(
        deconv_results.sum(axis=1), axis=0
    )  # to sum up to 1
    correlations = compute_correlations(deconv_results, facs_results)
    group_correlations = compute_group_correlations(deconv_results, facs_results)
    df_test_correlations.loc[:, "MixupVI"] = correlations.values
    df_test_group_correlations.loc[:, "MixupVI"] = group_correlations.values

    # Plots
    plot_deconv_results(df_test_correlations,
                        save=True,
                        filename="facs_1st_try")
    plot_deconv_results_group(df_test_group_correlations,
                              save=True,
                              filename="facs_1st_try_cell_type")

# %% Sanity check 3

if (
    (BENCHMARK_CELL_TYPE_GROUP != "FACS_1st_level_granularity") or
    (BENCHMARK_CELL_TYPE_GROUP == "FACS_1st_level_granularity" and COMPUTE_SC_RESULTS_WHEN_FACS)
):
    results = {}
    results_group = {}

    for n in N_CELLS:
        logger.info(f"Pseudobulk simulation with {n} sampled cells ...")
        all_adata_samples_test, adata_pseudobulk_test_counts, adata_pseudobulk_test_rc, df_proportions_test = create_dirichlet_pseudobulk_dataset(
            adata_test,
            prior_alphas = None,
            n_sample = N_SAMPLES,
            n_cells = n,
            add_sparsity=False # useless in the current modifications
        )
        # decomment following for Sanity check 2.
        # adata_pseudobulk_test_counts, adata_pseudobulk_test_rc, df_proportions_test = create_uniform_pseudobulk_dataset(
        #     adata_test,
        #     n_sample = N_SAMPLES,
        #     n_cells = n,
        # )

        df_test_correlations, df_test_group_correlations = run_sanity_check(
            adata_train=adata_train,
            adata_pseudobulk_test_counts=adata_pseudobulk_test_counts,
            adata_pseudobulk_test_rc=adata_pseudobulk_test_rc,
            all_adata_samples_test=all_adata_samples_test,
            filtered_genes=filtered_genes,
            df_proportions_test=df_proportions_test,
            signature=signature,
            generative_models=generative_models,
            baselines=BASELINES,
        )

        results[n] = df_test_correlations
        results_group[n] = df_test_group_correlations

    # %% Plots
    if len(results) > 1:
        plot_deconv_lineplot(results,
                            save=True,
                            filename=f"lineplot_tuned_mixupvi_third_granularity_retry_normal")
    else:
        key = list(results.keys())[0]
        plot_deconv_results(results[key],
                            save=True,
                            # filename=f"benchmark_{key}_cells_first_granularity")
                            filename="test_first_type")
        plot_deconv_results_group(results_group[key],
                                    save=True,
                                    # filename=f"benchmark_{key}_cells_first_granularity_cell_type")
                                    filename="test_first_type_cell_type")


# %% (Optional) Sanity check 1.

# create *purified* train/test pseudobulk datasets
# adata_pseudobulk_test_counts, adata_pseudobulk_test_rc = create_purified_pseudobulk_dataset(adata_test)

# deconv_results = run_purified_sanity_check(
#     adata_train=adata_train,
#     adata_pseudobulk_test_counts=adata_pseudobulk_test_counts,
#     adata_pseudobulk_test_rc=adata_pseudobulk_test_rc,
#     signature=signature,
#     generative_models=generative_models,
#     baselines=BASELINES,
# )

# # # # %% Plot
# plot_purified_deconv_results(
#     deconv_results,
#     only_fit_one_baseline=False,
#     more_details=False,
#     save=True,
#     filename="test_sanitycheck_1"
# )

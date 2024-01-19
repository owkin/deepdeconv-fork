"""Pseudobulk benchmark."""
# %%
import scanpy as sc
from loguru import logger
import warnings

from constants import (
    BENCHMARK_DATASET,
    SIGNATURE_CHOICE,
    BENCHMARK_CELL_TYPE_GROUP,
    BENCHMARK_LOG,
    SAVE_MODEL,
    N_CELLS,
    N_SAMPLES,
    GENERATIVE_MODELS,
    BASELINES,
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
    plot_purified_deconv_results,
    plot_deconv_results,
    plot_deconv_results_group,
    plot_deconv_lineplot,
)

# %% Load scRNAseq dataset
logger.info(f"Loading single-cell dataset: {BENCHMARK_DATASET} ...")
if BENCHMARK_DATASET == "TOY":
    raise NotImplementedError(
        "For now, the toy dataset cannot be used to run the benchmark because no "
        "signature has intersections with its genes, and no train/test split csv exists"
    )
    # adata = scvi.data.heart_cell_atlas_subsampled()
    # preprocess_scrna(adata, keep_genes=1200, log=BENCHMARK_LOG)
elif BENCHMARK_DATASET == "CTI":
    adata = sc.read("/home/owkin/data/cti_data/processed/cti_processed_3000.h5ad")
    # adata = sc.read("/home/owkin/project/cti/cti_adata.h5ad")
    # preprocess_scrna(adata,
    #                  keep_genes=3000,
    #                  log=BENCHMARK_LOG,
    #                  batch_key="donor_id")
elif BENCHMARK_DATASET == "CTI_RAW":
    warnings.warn("The raw data of this adata is on adata.raw.X, but the normalised "
                  "adata.X will be used here")
    adata = sc.read("/home/owkin/data/cross-tissue/omics/raw/local.h5ad")
    preprocess_scrna(adata,
                     keep_genes=2500,
                     log=BENCHMARK_LOG,
                     batch_key="donor_id",
    )
elif BENCHMARK_DATASET == "CTI_PROCESSED":
    # Load processed for speed-up (already filtered, normalised, etc.)
    adata = sc.read("/home/owkin/data/cti_data/processed/cti_processed_2500.h5ad")

# %% load signature
logger.info(f"Loading signature matrix: {SIGNATURE_CHOICE} | {BENCHMARK_CELL_TYPE_GROUP}...")
signature, intersection = create_signature(
    adata,
    signature_type=SIGNATURE_CHOICE,
)

# %% add cell types groups and split train/test
adata, train_test_index = add_cell_types_grouped(adata, BENCHMARK_CELL_TYPE_GROUP)
adata_train = adata[train_test_index["Train index"]]
adata_test = adata[train_test_index["Test index"]]

# %%
generative_models = {}
if GENERATIVE_MODELS != []:
    # Create and train models
    adata_train = adata_train.copy()
    adata_test = adata_test.copy()

    ### %% 1. scVI
    if "scVI" in GENERATIVE_MODELS:
        logger.info("Fit scVI ...")
        model_path = f"project/models/{BENCHMARK_DATASET}_scvi.pkl"
        scvi_model = fit_scvi(adata_train,
                              model_path,
                              save_model=SAVE_MODEL,
                              # batch effect correction
                              batch_key=["donor_id", "assay"])
        generative_models["scVI"] = scvi_model
    #### %% 2. DestVI
    if "DestVI" in GENERATIVE_MODELS:
        logger.info("Fit DestVI ...")
        # DestVI is only used in sanity check 2
        # Uniform
        # adata_pseudobulk_train_counts, adata_pseudobulk_train_rc, df_proportions_train = create_uniform_pseudobulk_dataset(
        #     adata_train, n_sample = N_SAMPLES, n_cells = N_CELLS,
        # )
        # Dirrichlet
        adata_pseudobulk_train_counts, adata_pseudobulk_train_rc, df_proportions_test = create_dirichlet_pseudobulk_dataset(
            adata_train, prior_alphas = None, n_sample = N_SAMPLES,
        )

        model_path_1 = f"project/models/{BENCHMARK_DATASET}_condscvi.pkl"
        model_path_2 = f"project/models/{BENCHMARK_DATASET}_destvi.pkl"
        condscvi_model , destvi_model= fit_destvi(adata_train,
                                                adata_pseudobulk_train_counts,
                                                model_path_1,
                                                model_path_2,
                                                cell_type_key="cell_types_grouped",
                                                save_model=SAVE_MODEL)
        # generative_models["CondscVI"] = condscvi_model
        generative_models["DestVI"] = destvi_model

    #### %% 3. MixupVI
    if "MixupVI" in GENERATIVE_MODELS:
        logger.info("Train mixupVI ...")
        model_path = f"project/models/{BENCHMARK_DATASET}_{BENCHMARK_CELL_TYPE_GROUP}_mixupvi.pkl"
        mixupvi_model = fit_mixupvi(adata_train,
                                    model_path,
                                    cell_type_group="cell_types_grouped",
                                    save_model=SAVE_MODEL,
                                    )
        generative_models["MixupVI"] = mixupvi_model

# # %% Sanity check 3

num_cells = [50, 100, 300, 500, 1000, 2000, 3000, 5000]

results = {}
results_group = {}

for n in num_cells:
    logger.info(f"Pseudobulk simulation with {n} sampled cells ...")
    adata_pseudobulk_test_counts, adata_pseudobulk_test_rc, df_proportions_test = create_dirichlet_pseudobulk_dataset(
        adata_test,
        prior_alphas = None,
        n_sample = N_SAMPLES,
        n_cells = N_CELLS,
    )

    df_test_correlations, df_test_group_correlations = run_sanity_check(
        adata_train=adata_train,
        adata_pseudobulk_test_counts=adata_pseudobulk_test_counts,
        adata_pseudobulk_test_rc=adata_pseudobulk_test_rc,
        df_proportions_test=df_proportions_test,
        signature=signature,
        intersection=intersection,
        generative_models=generative_models,
        baselines=BASELINES,
    )

    results[n] = df_test_correlations
    results_group[n] = df_test_group_correlations

# Plots
if len(results) > 1:
    plot_deconv_lineplot(results,
                        save=True,
                        filename=f"sim_pseudobulk_lineplot.png")
else:
    key = list(results.keys())[0]
    plot_deconv_results(results[key],
                        save=True,
                        filename=f"sim_pseudobulk_{key}.png")
    plot_deconv_results_group(results_group[key],
                                save=True,
                                filename=f"sim_pseudobulk_{key}_per_celltype.png")


# %% (Optional) Sanity check 1.

# create *purified* train/test pseudobulk datasets
# adata_pseudobulk_test_counts, adata_pseudobulk_test_rc = create_purified_pseudobulk_dataset(adata_test)

# deconv_results = run_purified_sanity_check(
#     adata_train=adata_train,
#     adata_pseudobulk_test_counts=adata_pseudobulk_test_counts,
#     adata_pseudobulk_test_rc=adata_pseudobulk_test_rc,
#     signature=signature,
#     intersection=intersection,
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

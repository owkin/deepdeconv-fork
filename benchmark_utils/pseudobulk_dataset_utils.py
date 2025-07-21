"""Utilities for pseudobulk dataset creation."""

import random

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from .run_benchmark_constants import EVALUATION_PSEUDOBULK_SAMPLINGS, initialize_func

from constants import N_CELLS_PER_PSEUDOBULK
from .load_dataset_utils import load_bulk_facs


def launch_evaluation_pseudobulk_samplings(
    evaluation_pseudobulk_sampling: list,
    all_data: dict,
    evaluation_dataset: str,
    granularity: str,
    n_cells_per_evaluation_pseudobulk: int,
    n_samples_evaluation_pseudobulk: int,
):
    """General function to create pseudobulks from different sampling methods.

    Parameters
    ----------
    evaluation_datasets: list
        The datasets to evaluate the deconvolution methods on.
    train_dataset: str | None
        The dataset to train some deconvolution methods on.
    n_variable_genes: int | None
        The number of most variable genes to keep.

    Return
    ------
    data: dict
        The train and evaluation datasets.
    """
    evaluation_pseudobulk_samplings_func, kwargs = initialize_func(
        EVALUATION_PSEUDOBULK_SAMPLINGS[evaluation_pseudobulk_sampling]
    )
    all_test_dset = all_data["datasets"][evaluation_dataset]
    test_dset = all_test_dset["dataset"][all_test_dset[granularity]["Test index"]]
    kwargs["adata"] = test_dset
    if "cell_type_group" in kwargs:
        kwargs["adata"].obs = kwargs["adata"].obs.rename(
            {f"cell_types_grouped_{granularity}": "cell_types_grouped"}, axis=1
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

    # TODO: Add a check to see if the pseudobulks have to be created with mean or sum!!!!!
    pseudobulks = evaluation_pseudobulk_samplings_func(**kwargs)

    return pseudobulks


def create_anndata_pseudobulk(
    adata_obs: pd.DataFrame, adata_var_names: list, x: np.array
) -> ad.AnnData:
    """Creates an anndata object from a pseudobulk sample.

    Parameters
    ----------
    adata_obs: pd.DataFrame
        Obs dataframe from anndata object storing training set
    adata_var_names: list
        Gene names from the anndata object
    x: np.array
        pseudobulk sample

    Return
    ------
    ad.AnnData
        Anndata object storing the pseudobulk array
    """
    df_obs = pd.DataFrame.from_dict(
        [{col: adata_obs[col].value_counts().index[0] for col in adata_obs.columns}]
    )
    if len(x.shape) > 1 and x.shape[0] > 1:
        # several pseudobulks, so duplicate df_obs row
        df_obs = df_obs.loc[df_obs.index.repeat(x.shape[0])].reset_index(drop=True)
        df_obs.index = [f"sample_{idx}" for idx in df_obs.index]
    adata_pseudobulk = ad.AnnData(X=x, obs=df_obs)
    adata_pseudobulk.var_names = adata_var_names
    adata_pseudobulk.layers["counts"] = np.copy(x)
    adata_pseudobulk.raw = adata_pseudobulk

    return adata_pseudobulk


def create_purified_pseudobulk_dataset(
    adata: ad.AnnData,
    cell_type_group: str = "cell_types_grouped",
    aggregation_method: str = "mean",
):
    """Create pseudobulk dataset from single-cell RNA data, purified by cell types.

    There will thus be as many deconvolutions as there are cell types, each one of them
    only asked to infer that there is only one cell type in the pseudobulk it is trying
    to deconvolve. This task is supposed to be very easy.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to create the pseudobulk dataset from
    cell_type_group : str
        The cell type group to use for the pseudobulk dataset
    aggregation_method : str
        The aggregation method to use for the pseudobulk dataset (default "mean",
        can also be "sum")
    """
    logger.info("Creating purified pseudobulk dataset...")
    grouped = adata.obs.groupby(cell_type_group)
    averaged_data, group = {"relative_counts": [], "counts": []}, []
    for group_key, group_indices in grouped.groups.items():
        if aggregation_method == "mean":
            averaged_data["relative_counts"].append(
                adata[group_indices].layers["relative_counts"].mean(axis=0).tolist()[0]
            )
            averaged_data["counts"].append(
                adata[group_indices].layers["counts"].mean(axis=0).tolist()[0]
            )
        else:
            averaged_data["relative_counts"].append(
                adata[group_indices].layers["relative_counts"].sum(axis=0).tolist()[0]
            )
            averaged_data["counts"].append(
                adata[group_indices].layers["counts"].sum(axis=0).tolist()[0]
            )
        group.append(group_key)

    # pseudobulk dataset
    adata_pseudobulk_rc = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["relative_counts"])
    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["counts"])
    )
    adata_pseudobulk_rc.obs_names = group
    adata_pseudobulk_counts.obs_names = group
    groundtruth_fractions = pd.DataFrame(np.eye(len(group)), index=group, columns=group)
    groundtruth_fractions.columns.name = cell_type_group

    pseudobulks = {
        "adata_pseudobulk_test_counts": adata_pseudobulk_counts,
        "adata_pseudobulk_test_rc": adata_pseudobulk_rc,
        "df_proportions_test": groundtruth_fractions,
    }

    return pseudobulks


def create_uniform_pseudobulk_dataset(
    adata: ad.AnnData,
    n_sample: int = 300,
    n_cells: int = 2000,
    cell_type_group: str = "cell_types_grouped",
    aggregation_method: str = "mean",
):
    """Create pseudobulk dataset from single-cell RNA data, randomly sampled.

    This deconvolution task is not too hard because the pseudo-bulk have the same cell
    fractions than the training dataset on which was created the signature matrix. Plus,
    when using a high n_cells (e.g. the default 2000) to create the pseudo-bulks, all
    n_sample pseudo-bulks will have the same cell fractions because of the high number
    of cells.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to create the pseudobulk dataset from
    n_sample : int
        The number of pseudobulks to create
    n_cells : int
        The number of cells to sample for each pseudobulk
    cell_type_group : str
        The cell type group to use for the pseudobulk dataset
    aggregation_method : str
        The aggregation method to use for the pseudobulk dataset (default "mean",
        can also be "sum")
    """
    logger.info("Creating uniform pseudobulk dataset...")
    random.seed(random.randint(0, 1000))
    averaged_data = {"relative_counts": [], "counts": []}
    groundtruth_fractions = []
    for _ in range(n_sample):
        cell_sample = random.sample(list(adata.obs_names), n_cells)
        adata_sample = adata[cell_sample, :]
        groundtruth_frac = adata_sample.obs[cell_type_group].value_counts() / n_cells
        groundtruth_fractions.append(groundtruth_frac)
        if aggregation_method == "mean":
            averaged_data["relative_counts"].append(
                adata_sample.layers["relative_counts"].mean(axis=0).tolist()[0]
            )
            averaged_data["counts"].append(
                adata_sample.layers["counts"].mean(axis=0).tolist()[0]
            )
        else:
            averaged_data["relative_counts"].append(
                adata_sample.layers["relative_counts"].sum(axis=0).tolist()[0]
            )
            averaged_data["counts"].append(
                adata_sample.layers["counts"].sum(axis=0).tolist()[0]
            )

    # pseudobulk dataset
    adata_pseudobulk_rc = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["relative_counts"])
    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["counts"])
    )

    # ground truth fractions
    groundtruth_fractions = pd.DataFrame(
        groundtruth_fractions,
        index=adata_pseudobulk_counts.obs_names,
        columns=groundtruth_fractions[0].index,
    )
    groundtruth_fractions = groundtruth_fractions.fillna(
        0
    )  # the Nan are cells not sampled

    pseudobulks = {
        "adata_pseudobulk_test_counts": adata_pseudobulk_counts,
        "adata_pseudobulk_test_rc": adata_pseudobulk_rc,
        "df_proportions_test": groundtruth_fractions,
    }
    return pseudobulks


def create_dirichlet_pseudobulk_dataset(
    adata: ad.AnnData,
    prior_alphas: np.array = None,
    n_sample: int = 300,
    cell_type_group: str = "cell_types_grouped",
    aggregation_method: str = "mean",
    n_cells: int = 1000,
    is_n_cells_random: bool = False,
    add_sparsity: bool = False,
):
    """Create pseudobulk dataset from single-cell RNA data, sampled from a dirichlet distribution.

    If a prior belief on the cell fractions (e.g. prior knowledge from
    specific tissue), then it can be incorporated. Otherwise, it will just be a non-
    informative prior. Then, compute dirichlet posteriors to sample cells - dirichlet is
    conjugate to the multinomial distribution, thus giving an easy posterior
    calculation.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to create the pseudobulk dataset from
    prior_alphas : np.array
        The prior alphas to use for the pseudobulk dataset
    n_sample : int
        The number of pseudobulks to create
    cell_type_group : str
        The cell type group to use for the pseudobulk dataset
    aggregation_method : str
        The aggregation method to use for the pseudobulk dataset (default "mean",
        can also be "sum")
    n_cells : int
        The number of cells to sample for each pseudobulk
    is_n_cells_random : bool
        Whether to sample the number of cells for each pseudobulk randomly
    add_sparsity : bool
        Whether to add sparsity to the pseudobulk dataset
    """
    # logger.info("Creating dirichlet pseudobulk dataset...")
    seed = random.randint(0, 1000)
    random_state = np.random.RandomState(seed=seed)
    cell_types = adata.obs[cell_type_group].value_counts()
    if prior_alphas is None:
        prior_alphas = np.ones(len(cell_types))  # non-informative prior
    likelihood_alphas = cell_types / adata.n_obs  # multinomial likelihood
    alpha_posterior = prior_alphas + likelihood_alphas
    posterior_dirichlet = random_state.dirichlet(alpha_posterior, n_sample)
    if is_n_cells_random:
        n_cells = np.random.randint(50, 1001, size=posterior_dirichlet.shape[0])
        posterior_dirichlet = np.round(np.multiply(posterior_dirichlet, n_cells))
    else:
        posterior_dirichlet = np.round(posterior_dirichlet * n_cells)
    posterior_dirichlet = posterior_dirichlet.astype(
        np.int64
    )  # number of cells to sample
    groundtruth_fractions = posterior_dirichlet / posterior_dirichlet.sum(
        axis=1, keepdims=True
    )

    random.seed(seed)
    averaged_data = {"relative_counts": [], "counts": []}
    all_adata_samples = []
    for i in range(n_sample):
        sample_data = []
        for j, cell_type in enumerate(likelihood_alphas.index):
            # If sample larger than cell population, sample with replacement
            if posterior_dirichlet[i][j] > cell_types[cell_type]:
                cell_sample = random.choices(
                    list(adata.obs.loc[adata.obs[cell_type_group] == cell_type].index),
                    k=posterior_dirichlet[i][j],
                )
            else:
                cell_sample = random.sample(
                    list(adata.obs.loc[adata.obs[cell_type_group] == cell_type].index),
                    posterior_dirichlet[i][j],
                )
            sample_data.extend(cell_sample)
        adata_sample = adata[sample_data]
        if aggregation_method == "mean":
            averaged_data["relative_counts"].append(
                adata_sample.layers["relative_counts"].mean(axis=0).tolist()[0]
            )
            X = np.array(adata_sample.layers["counts"].mean(axis=0).tolist()[0])
            # TODO: For now, we remove the possibility to add sparsity, as all_adata_samples would not be affected
            # if add_sparsity:
            #     X = random_state.binomial(1, 0.2, X.shape[0]) * X
            averaged_data["counts"].append(X)
        # TODO: For now, we remove the possibility to aggregate by sum, as all_adata_samples would not be affected
        elif aggregation_method == "sum":
            averaged_data["relative_counts"].append(
                adata_sample.layers["relative_counts"].sum(axis=0).tolist()[0]
            )
            X = np.array(adata_sample.layers["counts"].sum(axis=0).tolist()[0])
            # if add_sparsity:
            #     X = random_state.binomial(1, 0.2, X.shape[0]) * X
            averaged_data["counts"].append(X)
        else:
            raise ValueError(f"Aggregation method {aggregation_method} not supported")
        all_adata_samples.append(adata_sample)

    # pseudobulk dataset
    adata_pseudobulk_rc = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["relative_counts"])
    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["counts"])
    )

    # ground truth fractions
    groundtruth_fractions = pd.DataFrame(
        groundtruth_fractions,
        index=adata_pseudobulk_counts.obs_names,
        columns=list(cell_types.index),
    )
    groundtruth_fractions = groundtruth_fractions.fillna(
        0
    )  # The Nan are cells not sampled

    pseudobulks = {
        "all_adata_samples_test": all_adata_samples,
        "adata_pseudobulk_test_counts": adata_pseudobulk_counts,
        "adata_pseudobulk_test_rc": adata_pseudobulk_rc,
        "df_proportions_test": groundtruth_fractions,
    }
    return pseudobulks


def create_purified_50_50_pseudobulk_dataset(
    adata,
    cell_type_1,
    cell_type_2,
    n_sample,
    n_cells_per_pseudobulk,
    cell_type_group="cell_types_grouped",
    aggregation_method="mean",
    random_state=None,
):
    """Creates multiple pseudobulk datasets by mixing two cell types in equal proportions.

    Args:
        adata: AnnData object containing single-cell data
        cell_type_1: First cell type to mix
        cell_type_2: Second cell type to mix
        n_sample: Number of pseudobulk samples to generate
        n_cells_per_pseudobulk: Total number of cells per pseudobulk
        cell_type_group: Column name in adata.obs containing cell type labels
        aggregation_method: Method to aggregate cells, either "mean" or "sum"
        random_state: Random state for reproducibility

    Returns
    -------
        Dictionary containing:
            - all_adata_samples_test: List of AnnData objects for each pseudobulk sample
            - adata_pseudobulk_test_counts: AnnData with raw counts
            - adata_pseudobulk_test_rc: AnnData with relative counts
            - df_proportions_test: DataFrame with ground truth proportions
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Get cells for each type
    cells_type1 = adata[adata.obs[cell_type_group] == cell_type_1].copy()
    cells_type2 = adata[adata.obs[cell_type_group] == cell_type_2].copy()

    # Check if we have enough cells
    cells_per_type = n_cells_per_pseudobulk // 2
    if cells_type1.n_obs < cells_per_type or cells_type2.n_obs < cells_per_type:
        raise ValueError(
            f"Not enough cells available. Need {cells_per_type} cells per type, "
            f"but have {cells_type1.n_obs} cells for type 1 and {cells_type2.n_obs} for type 2"
        )

    averaged_data = {"relative_counts": [], "counts": []}
    all_adata_samples = []

    for _ in range(n_sample):
        # Sample cells for each type
        sample1 = cells_type1[
            np.random.choice(cells_type1.n_obs, cells_per_type, replace=False)
        ]
        sample2 = cells_type2[
            np.random.choice(cells_type2.n_obs, cells_per_type, replace=False)
        ]

        # Combine samples
        adata_sample = adata.concatenate(sample1, sample2)
        all_adata_samples.append(adata_sample)

        if aggregation_method == "mean":
            averaged_data["relative_counts"].append(
                adata_sample.layers["relative_counts"].mean(axis=0).tolist()[0]
            )
            averaged_data["counts"].append(
                np.array(adata_sample.layers["counts"].mean(axis=0).tolist()[0])
            )
        elif aggregation_method == "sum":
            averaged_data["relative_counts"].append(
                adata_sample.layers["relative_counts"].sum(axis=0).tolist()[0]
            )
            averaged_data["counts"].append(
                np.array(adata_sample.layers["counts"].sum(axis=0).tolist()[0])
            )
        else:
            raise ValueError(f"Aggregation method {aggregation_method} not supported")

    # Create pseudobulk datasets
    adata_pseudobulk_rc = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["relative_counts"])
    )
    adata_pseudobulk_counts = create_anndata_pseudobulk(
        adata.obs, adata.var_names, np.array(averaged_data["counts"])
    )

    # Create ground truth proportions
    cell_types = pd.Series(adata.obs[cell_type_group].unique())
    groundtruth_fractions = pd.DataFrame(
        0,
        index=adata_pseudobulk_counts.obs_names,
        columns=list(cell_types),
    )
    groundtruth_fractions[cell_type_1] = 0.5
    groundtruth_fractions[cell_type_2] = 0.5

    pseudobulks = {
        "all_adata_samples_test": all_adata_samples,
        "adata_pseudobulk_test_counts": adata_pseudobulk_counts,
        "adata_pseudobulk_test_rc": adata_pseudobulk_rc,
        "df_proportions_test": groundtruth_fractions,
    }
    return pseudobulks


def _sample_one_pseudobulk(i, posterior_dirichlet, index_arrays, rel_counts,
                           counts, latent, agg="mean", rng=None):
    """
    Helper: build one pseudobulk and return the aggregated rows.
    """
    rng = np.random.default_rng(rng)          # independent stream

    # --- sample cell indices -------------------------------------------------
    idxs = np.concatenate([
        rng.choice(arr,
                   size=posterior_dirichlet[i, j],
                   replace=posterior_dirichlet[i, j] > arr.size)
        for j, arr in enumerate(index_arrays)
    ])
    # -------------------------------------------------------------------------

    if agg == "mean":
        rc_row = rel_counts[idxs].mean(0)
        c_row  = counts[idxs].mean(0)
        latent_row = latent[idxs].mean(0)
    elif agg == "sum":
        rc_row = rel_counts[idxs].sum(0)
        c_row  = counts[idxs].sum(0)
        latent_row = latent[idxs].sum(0)
    else:
        raise ValueError(f"Unknown aggregation {agg}")

    return rc_row, c_row, latent_row, idxs


def create_dirichlet_pseudobulk_dataset_v2(
    adata: ad.AnnData,
    prior_alphas= None,
    n_sample: int = 300,
    cell_type_group: str = "cell_types_grouped",
    aggregation_method: str = "mean",
    n_cells: int = 256,
    n_jobs: int = 1,                         # ← optional parallelism
    seed: int = None
):
    rng = np.random.default_rng(seed)

    # 1. Dirichlet fractions ➜ integer cell counts per pseudobulk ---------------
    ct_counts = adata.obs[cell_type_group].value_counts()
    ct_names  = ct_counts.index.to_list()

    if prior_alphas is None:
        prior_alphas = np.ones_like(ct_counts, dtype=float)

    post_alpha = prior_alphas + ct_counts / adata.n_obs
    theta      = rng.dirichlet(post_alpha, n_sample)      # (bulk, cell_type)

    if isinstance(n_cells, list):
        n_cells_vec = rng.integers(n_cells[0], n_cells[1], size=n_sample)
        posterior_cn = np.round(theta * n_cells_vec[:, None]).astype(int)
    else:
        posterior_cn = np.round(theta * n_cells).astype(int)

    gt_fractions = posterior_cn / posterior_cn.sum(1, keepdims=True)

    # 2. Pre‑group cell indices once -------------------------------------------
    index_arrays = [
        np.where(adata.obs[cell_type_group].to_numpy() == ct)[0]
        for ct in ct_names
    ]


    # 3. Pull raw matrices (much faster than per‑cell slicing later) ------------
    rel_counts = adata.layers["relative_counts"]      # (scell, gene) Dense / CSR
    counts_mat = adata.layers["counts"]
    latent_mat = adata.obsm["latent_sc"]

    # 4. Build pseudobulks (optionally parallel) --------------------------------
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_sample_one_pseudobulk)(
            i, posterior_cn, index_arrays, rel_counts, counts_mat, latent_mat,
            aggregation_method, rng.bit_generator.random_raw())
        for i in range(n_sample)
    ) if n_jobs != 1 else [
        _sample_one_pseudobulk(
            i, posterior_cn, index_arrays, rel_counts, counts_mat, latent_mat,
            aggregation_method, rng.bit_generator.random_raw())
        for i in range(n_sample)
    ]

    rc_rows, c_rows, latent_rows, all_indices = map(list, zip(*results))

    # 5. Assemble outputs -------------------------------------------------------
    adata_pb_rc = ad.AnnData(
        X=np.vstack(rc_rows),
        var=adata.var.copy(),
        obs=pd.DataFrame(index=[f"PB_{i}" for i in range(n_sample)])
    )
    adata_pb_counts = adata_pb_rc.copy()
    adata_pb_counts.X = np.vstack(c_rows)
    adata_pb_counts.obsm["latent_sc"] = np.vstack(latent_rows)

    df_gt = pd.DataFrame(
        gt_fractions, index=adata_pb_rc.obs_names, columns=ct_names).fillna(0)

    return {
        "adata_pseudobulk_rc":     adata_pb_rc,
        "adata_pseudobulk_counts": adata_pb_counts,
        "all_cell_idx_per_bulk":   all_indices,       # raw indices, very lightweight
        "df_proportions":          df_gt
    }


def prepare_mixupvi_v2_data(
    adata: ad.AnnData,
    base_model,
    n_pseudobulk_samples: int = 10000,
    n_cells_per_pseudobulk: int = None,
    seed: int = None,
):
    """Prepare data for MixUpVI_v2 model training.
    
    This function extracts the data preprocessing logic from fit_mixupvi_v2
    to create the final adata object that combines pseudobulk and bulk data.
    
    Parameters
    ----------
    adata : AnnData
        The single-cell AnnData object
    base_model
        The trained base model (MixUpVI or SCVI)
    n_pseudobulk_samples : int
        Number of pseudobulk samples to create
    n_cells_per_pseudobulk : int
        Number of cells per pseudobulk. If None, uses N_CELLS_PER_PSEUDOBULK from constants
        
    Returns
    -------
    final_adata : AnnData
        The final processed AnnData object ready for MixUpVI_v2 training
    """
    
    if n_cells_per_pseudobulk is None:
        n_cells_per_pseudobulk = N_CELLS_PER_PSEUDOBULK
    
    # Add latent representation to single-cell data
    adata.obsm["latent_sc"] = base_model.get_latent_representation(give_mean=True, return_dist=False)
    
    # Create pseudobulk dataset
    adata_pb = create_dirichlet_pseudobulk_dataset_v2(
        adata, 
        n_sample=n_pseudobulk_samples, 
        n_cells=n_cells_per_pseudobulk,
        seed=seed
    )
    genes = adata_pb["adata_pseudobulk_counts"].var_names.tolist()
    
    # Load and process bulk data
    bulk_dataset = load_bulk_facs()
    adata_bulk = bulk_dataset["dataset"]
    adata_bulk = adata_bulk.loc[genes]
    adata_bulk = adata_bulk.T
    
    adata_bulk = ad.AnnData(
        X=adata_bulk.values,
        var=adata_pb["adata_pseudobulk_counts"].var,
    )
    
    # Process ground truth proportions
    bulk_dataset["ground_truth"] = bulk_dataset["ground_truth"] / 100
    bulk_dataset["ground_truth"] = bulk_dataset["ground_truth"].div(
        bulk_dataset["ground_truth"].sum(axis=1), axis=0
    )
    bulk_dataset["ground_truth"] = bulk_dataset["ground_truth"][adata_pb["df_proportions"].columns]
    
    # Add latent representation placeholder for bulk data
    adata_bulk.obsm["latent_sc"] = np.full(
        (adata_bulk.n_obs, adata_pb["adata_pseudobulk_counts"].obsm["latent_sc"].shape[1]),
        np.nan,
        dtype=np.float32
    )
    adata_bulk.obsm["ground_truth"] = bulk_dataset["ground_truth"].values
    adata_bulk.uns["cell_types_order"] = bulk_dataset["ground_truth"].columns.tolist()
    
    # Add metadata flags
    adata_bulk.obs["has_latent"] = False
    adata_pb["adata_pseudobulk_counts"].obs["has_latent"] = True
    adata_pb["adata_pseudobulk_counts"].obsm["ground_truth"] = adata_pb["df_proportions"].values
    adata_pb["adata_pseudobulk_counts"].uns["cell_types_order"] = adata_pb["df_proportions"].columns
    
    # Concatenate pseudobulk and bulk data
    final_adata = ad.concat(
        [adata_pb["adata_pseudobulk_counts"], adata_bulk],
        join="outer",
        merge="same",
        label="source",
        keys=["pseudobulk", "bulk"],
    )

    final_adata.uns["bulk_cell_types_order"] = adata_bulk.uns["cell_types_order"]
    return final_adata

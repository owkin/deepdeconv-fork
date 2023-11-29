"""Helper functions to evaluate scvi to build signature matrix in latent space.

The corresponding notebook can be found as linearity_checks.ipynb.
"""

from typing import Optional, Union

from scvi_sanity_checks_utils import create_anndata_pseudobulk
import matplotlib.pyplot as plt
import scvi
import scanpy as sc
import random
import numpy as np
import anndata as ad
import pandas as pd
from scipy.optimize import nnls


# Loading models trained by Khalil


def load_models(
    adata_path: str,
    model_paths: list[str],
    model_names: Optional[list[str]] = None,
    use_gpu: bool = True,
) -> tuple[ad.AnnData, dict[str, scvi.model.SCVI]]:
    """Utils to load the model.

    Parameters
    ----------
    adata_path: str
        The path to the ann data object. Must end with the suffix .h5ad
    model_paths: str
        the paths the models we want to load, of size N_models
    model_names: Optional[list[str]]
        The model names, of size N_models. If no name is provided, will just keep the
        file name.
    use_gpu : bool
        Whether to use the GPU.

    Returns
    -------
    tuple[ad.AnnData, dict[str, scvi.model.SCVI],list[str]]
    """
    assert adata_path.endswith(".h5ad"), "AnnData must be saved in h5ad format."
    adata = ad.read_h5ad(adata_path)

    if model_names is None:
        model_names_list: list[str] = [
            model_path.split("/")[-1] for model_path in model_paths
        ]
    else:
        assert len(model_names) == len(model_paths)
        model_names_list = model_names

    model_dictionary: dict[str, scvi.model.SCVI] = {}
    for model_name, model_path in zip(model_names_list, model_paths):
        model_dictionary[model_name] = scvi.model.SCVI.load(
            dir_path=model_path, adata=adata, use_gpu=use_gpu
        )
    return adata, model_dictionary


def make_embeddings(
    adata: ad.AnnData,
    model_dict: dict[str, scvi.model.SCVI],
    show_umap=False,
    show_pca=False,
) -> ad.AnnData:
    """Make latent representation embeddings for all cells.

    For each model in the model dictionary, this function computes
    the scvi embedding of the single cells and stores it in the
    X_scvi__{model_name} attribute of the obsm adata field.

    This function also comes with a representation utils, to show
    umap and pca if wanted.

    Parameters
    ----------
    adata: ad.AnnData
        The adata
    model_dict: dict[str, scvi.model.SCVI]
    show_umap: bool
        If we want to show the umap of the embedding for all models
    show_pca:
        If we want to show PCA of the embedding for all models

    Returns
    -------
    AnnData
        The adata, containing X_scvi__{model_name} fields in obsm with the
        scvi embedding corresponding to the models.

    """
    for model_name, model in model_dict.items():
        # Regular scVI
        latent = model.get_latent_representation()
        adata.obsm[f"X_scvi__{model_name}"] = latent
        pca = sc.tl.pca(latent)[:, :2]
        adata.obsm[f"X_pca__{model_name}"] = pca

        # run PCA then generate UMAP plots
        if show_pca:
            print(f"Plotting PCA for model {model_name}")
            adata.obsm["X_pca"] = pca
            sc.pl.pca(adata, color=["cell_type"])
            plt.show()

        if show_umap:
            print(f"Plotting UMAP for model {model_name}")
            sc.pp.neighbors(adata, use_rep=f"X_scvi__{model_name}")
            sc.tl.umap(adata, min_dist=0.3)
            sc.pl.umap(
                adata,
                color=["cell_type"],
                frameon=False,
            )
            plt.show()
    return adata


def make_cell_type_representations(
    adata: ad.AnnData,
    sc_per_pseudobulk: int,
    repeats: int = 1,
    aggregate_before_embedding: bool = True,
    cell_type_column: str = "cell_type",
    count_key: Optional[str] = "counts",
    representation_key: Optional[str] = "X_scvi",
    model: Optional[scvi.model.SCVI] = None,
) -> ad.AnnData:
    """Make cell type representations from a single cell dataset represented with scvi.


    From an annotated single cell dataset (adata), for each cell type, (found in the
    cell_type column of obs in adata), we create "repeats" representation in the
    following way.

    - We sample sc_per_pseudobulk single cells of the desired cell type with replacement
    - We then create the corresponding cell type representation, in one of the
    two following ways.
    - Option 1)
        If we choose to aggregate before embedding (aggregate_before_embedding flag),
        we construct a pseudobulk of these single cells (all of the same cell type)
        forming a "pure" pseudobulk of the given cell type.
        We then take the scvi model (model) latent representation of this purified
        pseudobulk.
    - Option 2)
        If we choose to aggregate after embedding, we get the corresponding
        embeddings from the adata.obsm[(representation_key)] field of the ann data,
        (This assumes that we have encoded the ann data with scvi) and then average them.

    We then output an AnnData object, containing all these representations
    (n_repeats representation per cell type), whose obs field contains the repeat number
    in the "repeat" column, and the cell type in the "cell type" column.


    Parameters
    ----------
    adata: ad.AnnData
        The single cell dataset, with a cell_type_column, and a representation_key in
        the obsm if one wants to aggregate after embedding.
    sc_per_pseudobulk: int
        The number of single cells used to construct the purified pseudobulks.
    repeats: int
        The number of representations computed randomly for a given cell type.
    aggregate_before_embedding: bool
        Perform the aggregation (average) before embedding the cell-type specific
        pseudobulk. If false, we aggregate the representations.
    cell_type_column: str
        The field of the ann data obs containing the cell type partition of interest.
    count_key: Optional[str]
        The layer containing counts, mandatory if aggregating before embedding.
    representation_key: Optional[str]
        The field of obsm containing the pre-computed scvi representation, used only
        if aggregation takes place after representation.
    model: Optional[scvi.model.SCVI]
        The trained scvi model, used only if aggregation is before representation.

    Returns
    -------
    ad.AnnData
        The output annotated dataset of cell type specific representations,
        containing "repeats" random examples.

    """
    cell_type_list = []
    representation_list: list[np.ndarray] = []
    repeat_list = []
    for cell_type in adata.obs[cell_type_column].unique():
        for repeat in range(repeats):
            seed = random.seed()
            # Sample cells
            sampled_cells = (
                adata.obs[adata.obs[cell_type_column] == cell_type]
                .sample(n=sc_per_pseudobulk, random_state=seed, replace=True)
                .index
            )
            adata_sampled = adata[sampled_cells]

            if aggregate_before_embedding:
                assert (
                    model is not None,
                    "If representing a purified pseudo bulk (aggregate before embedding",
                    "), must give a model",
                )
                assert (
                    count_key is not None
                ), "Must give a count key if aggregating before embedding."

                pseudobulk = (
                    adata_sampled.layers[count_key].mean(axis=0).reshape(1, -1)
                )  # .astype(int).astype(numpy.float32)
                adata_pseudobulk = create_anndata_pseudobulk(adata_sampled, pseudobulk)
                result = model.get_latent_representation(adata_pseudobulk).reshape(-1)
            else:
                assert representation_key is not None, (
                    "A representation key must be given to compute"
                    + "aggregate after embedding."
                )

                latent_sampled = adata.obsm[representation_key]
                result = latent_sampled.mean(axis=0)
                result = result / np.linalg.norm(result)

            repeat_list.append(repeat)
            representation_list.append(result)
            cell_type_list.append(cell_type)
    full_rpz = np.stack(representation_list, axis=0)
    obs = pd.DataFrame(pd.Series(cell_type_list, name="cell type"))
    obs["repeat"] = repeat_list
    cell_type_representations = ad.AnnData(X=full_rpz, obs=obs)
    return cell_type_representations


def visualize_cell_type_representations(cell_type_representations: ad.AnnData):
    sc.pp.neighbors(cell_type_representations, use_rep="X")
    sc.tl.umap(cell_type_representations, min_dist=0.3)
    print("Plotting a UMAP")
    sc.pl.umap(
        cell_type_representations,
        color=["cell type"],
        frameon=False,
    )
    plt.show()
    sc.tl.pca(cell_type_representations)
    print("Plotting first 2 components of PCA")
    sc.pl.pca(
        cell_type_representations,
        color=["cell type"],
        frameon=False,
    )
    plt.show()


# Utils to test signature.


def create_pseudobulk_embedding_from_proportions(
    model: scvi.model.SCVI,
    adata: ad.AnnData,
    cell_types: list[str],
    proportions: np.ndarray,
    total_cells: int = 11000,
    cell_type_key: str = "cell_type",
    count_key: str = "counts",
) -> np.ndarray:
    """Create pseudobulk embedding from proportions.

    Given an array of proportions for th

    Parameters
    ----------
    model: scvi.model.SCVI
        The scvi model used to create the embeddings.
    adata: ad.AnnData
        The single cell dataset, with cell type annotation in the cell_type_key
        column of obs
    cell_types: list[str]
        The list of possible cell types, whose length is denoted N_types.
    proportions: np.ndarray
        An array of size (N_types,) which sums to one, which corresponds to the
        target proportions of each cell when we create the pseudobulk.
    total_cells: int
        The total number of cells we wish to put in the pseudobulk.
    cell_type_key: str
        The column of the ann data observations containing the cell type.
    count_key: str
        The name of the layer containing the counts.

    Returns
    -------
    np.ndarray
        An array of size (d,) where d is the dimension of the embedding.
    """
    sampled_cells_per_type = proportions * total_cells
    sampled_cells_per_type = sampled_cells_per_type.astype(int)

    # Perform stratified sampling for each cell type
    sampled_cells: list[int] = []
    for (
        cell_type_index,
        cell_type,
    ) in enumerate(cell_types):
        seed = random.seed()
        num_cells = sampled_cells_per_type[cell_type_index]
        sampled_cells.extend(
            adata.obs[adata.obs[cell_type_key] == cell_type]
            .sample(n=num_cells, random_state=seed, replace=True)
            .index
        )
    adata_sampled = adata[np.array(sampled_cells)]
    pseudobulk = adata_sampled.layers[count_key].mean(axis=0).reshape(1, -1)
    adata_pseudobulk = create_anndata_pseudobulk(adata_sampled, pseudobulk)
    result = model.get_latent_representation(adata_pseudobulk).reshape(-1)
    return result


def create_random_proportion(
    n_classes: int, n_non_zero: Optional[int] = None
) -> np.ndarray:
    """Create a random proportion vector of size n_classes.

    The n_non_zero parameter allows to set the number
    of non-zero components of the random discrete density vector.

    The random sampling is done in the following way: we randomly sample
    a vector of size n_classes with values between [0,1] before normalizing them.

    Parameters
    ----------
    n_classes: int
        The size of the proportion vector.
    n_non_zero: Optional[int]
        The number of non-zero components in the random vector

    Returns
    -------
    np.ndarray
        A vector of size n_clases, normalized
    """
    if n_non_zero is None:
        n_non_zero = n_classes

    proportion_vector = np.zeros(
        n_classes,
    )

    proportion_vector[:n_non_zero] = np.random.rand(n_non_zero)

    proportion_vector = proportion_vector / proportion_vector.sum()
    return np.random.permutation(proportion_vector)


# Generic functions to generate the experiment plots


def stability_of_embeddings(
    adata_path: str,
    model_paths: list[str],
    model_names: Optional[list[str]] = None,
    use_gpu: bool = True,
    sc_per_pseudobulk: int = 1000,
    repeats: int = 1,
    cell_type_column: str = "cell_type",
    count_key: Optional[str] = "counts",
) -> tuple[dict[str, ad.AnnData], dict[str, ad.AnnData]]:
    """Generates plots on stability of embeddings.

    Parameters
    ----------
    adata_path: str
        The path to the ann data object. Must end with the suffix .h5ad
    model_paths: str
        the paths the models we want to load, of size N_models
    model_names: Optional[list[str]]
        The model names, of size N_models. If no name is provided, will just keep the
        file name.
    use_gpu : bool
        Whether to use the GPU.
    sc_per_pseudobulk: int
        The number of single cells used to construct the purified pseudobulks.
    repeats: int
        The number of representations computed randomly for a given cell type.
    cell_type_column: str
        The field of the ann data obs containing the cell type partition of interest.
    count_key: Optional[str]
        The layer containing counts, mandatory if aggregating before embedding.

    """
    adata, model_dict = load_models(
        adata_path, model_paths, model_names, use_gpu=use_gpu
    )
    adata = make_embeddings(
        adata,
        model_dict,
        show_umap=False,
        show_pca=False,
    )

    aggregate_before_dict: dict[str, ad.AnnData] = {}
    aggregate_after_dict: dict[str, ad.AnnData] = {}

    for model_name, model in model_dict.items():
        aggregate_before_dict[model_name] = make_cell_type_representations(
            adata,
            sc_per_pseudobulk=sc_per_pseudobulk,
            repeats=repeats,
            aggregate_before_embedding=True,
            cell_type_column=cell_type_column,
            count_key=count_key,
            representation_key=f"X_scvi__{model_name}",
            model=model,
        )
        print(f"{model_name}, aggregation before representation.")
        visualize_cell_type_representations(aggregate_before_dict[model_name])
        aggregate_after_dict[model_name] = make_cell_type_representations(
            adata,
            sc_per_pseudobulk=sc_per_pseudobulk,
            repeats=repeats,
            aggregate_before_embedding=False,
            cell_type_column=cell_type_column,
            count_key=count_key,
            representation_key=f"X_scvi__{model_name}",
            model=model,
        )
        print(f"{model_name}, aggregation after representation.")
        visualize_cell_type_representations(aggregate_after_dict[model_name])

    return aggregate_before_dict, aggregate_after_dict


# Signature in latent space


def make_signature(
    adata: ad.AnnData,
    sc_per_pseudobulk: int,
    aggregate_before_embedding: bool = True,
    cell_type_column: str = "cell_type",
    count_key: Optional[str] = "counts",
    representation_key: Optional[str] = "X_scvi",
    model: Optional[scvi.model.SCVI] = None,
) -> tuple[np.ndarray, list[str]]:
    """Make cell type representations from a single cell dataset represented with scvi.


    From an annotated single cell dataset (adata), for each cell type, (found in the
    cell_type column of obs in adata), we create a signature representation in the
    following way.

    - We sample sc_per_pseudobulk single cells of the desired cell type with replacement
    - We then create the corresponding cell type representation, in one of the
    two following ways.
    - Option 1)
        If we choose to aggregate before embedding (aggregate_before_embedding flag),
        we construct a pseudobulk of these single cells (all of the same cell type)
        forming a "pure" pseudobulk of the given cell type.
        We then take the scvi model (model) latent representation of this purified
        pseudobulk.
    - Option 2)
        If we choose to aggregate after embedding, we get the corresponding
        embeddings from the adata.obsm[(representation_key)] field of the ann data,
        (This assumes that we have encoded the ann data with scvi) and then average them.

    We then output an AnnData object, containing all these representations
    (n_repeats representation per cell type), whose obs field contains the repeat number
    in the "repeat" column, and the cell type in the "cell type" column.


    Parameters
    ----------
    adata: ad.AnnData
        The single cell dataset, with a cell_type_column, and a representation_key in
        the obsm if one wants to aggregate after embedding.
    sc_per_pseudobulk: int
        The number of single cells used to construct the purified pseudobulks.
    aggregate_before_embedding: bool
        Perform the aggregation (average) before embedding the cell-type specific
        pseudobulk. If false, we aggregate the representations.
    cell_type_column: str
        The field of the ann data obs containing the cell type partition of interest.
    count_key: Optional[str]
        The layer containing counts, mandatory if aggregating before embedding.
    representation_key: Optional[str]
        The field of obsm containing the pre-computed scvi representation, used only
        if aggregation takes place after representation.
    model: Optional[scvi.model.SCVI]
        The trained scvi model, used only if aggregation is before representation.

    Returns
    -------
    tuple[np.ndarray,list[str]]
        The signature matrix of size (N_classes,dimension_latent)
        The cell types corresponding to the N_classes

    """
    adata_signature = make_cell_type_representations(
        adata,
        sc_per_pseudobulk,
        repeats=1,
        aggregate_before_embedding=aggregate_before_embedding,
        cell_type_column=cell_type_column,
        count_key=count_key,
        representation_key=representation_key,
        model=model,
    )
    return adata_signature.X.T, list(adata_signature.obs["cell type"])


def estimate_random_cosine_similarity(n_classes: int, repeats: int = 10000) -> float:
    """Estimate random cosine similarity (baseline).

    Parameters
    ----------
    n_classes: int
        The dimension of the vectors to which cosine similarity is applied.
    repeats: int
        The number of samples used to do the monte carlo estimation.

    Returns
    -------
    float
        The estimate of the average random cosine similarity, using montecarlo method.

    """
    hihi = np.zeros(repeats)
    for k in range(repeats):
        v1 = np.random.rand(n_classes)
        v2 = np.random.rand(n_classes)
        v1 = v1 / v1.sum()
        v2 = v2 / v2.sum()
        hihi[k] = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

    return hihi.mean()


def make_experiment_dataframe(
    adata_path: str,
    model_paths: list[str],
    model_names: Optional[list[str]] = None,
    use_gpu: bool = True,
    sc_per_purified_pseudobulk: int = 1000,
    sc_per_evaluation_pseudobulk: int = 11000,
    cell_type_column: str = "cell_type",
    count_key: Optional[str] = "counts",
    pseudobulks_per_experiment: int = 10,
    max_non_zero: Optional[int] = None,
) -> pd.DataFrame:
    """Runs the following experiment.

    For each model, we compute a latent signature matrix. This allows to
    infer proportions in the latent space using non negative least squares.

    For each model, and any fixed number of non zero components defining a random
    proportion, we draw "pseudobulks_per_experiment" random pseudobulks, and
    compute the cosine similarity between the proportion recovered by the signature
    matrix of the model and the true proportions used to generate the pseudobulk.

    We create a dataframe with columns "pseudobulk_number" (which random pseudobulk),
    "non_zero" (how many non zero components) "model_name" (the model name) and
    "cosine_similarity" the cosine similarity.

    Parameters
    ----------
    adata_path: str
        The path to the ann data object. Must end with the suffix .h5ad
    model_paths: str
        the paths the models we want to load, of size N_models
    model_names: Optional[list[str]]
        The model names, of size N_models. If no name is provided, will just keep the
        file name.
    use_gpu : bool
        Whether to use the GPU.
    sc_per_purified_pseudobulk: int
        The number of single cells used to construct the purified pseudobulks.
    sc_per_evaluation_pseudobulk: int
        The number of single cells used to construct the evaluation pseudobulkd.
    cell_type_column: str
        The field of the ann data obs containing the cell type partition of interest.
    count_key: Optional[str]
        The layer containing counts, mandatory if aggregating before embedding.
    pseudobulks_per_experiment: int
        The number of pseudobulk tried during each experiment.
    max_non_zero: Optional[int]
        The maximum of nonzero components in the proportion defining the pseudobulk
    """
    adata, model_dict = load_models(
        adata_path, model_paths, model_names, use_gpu=use_gpu
    )
    adata = make_embeddings(
        adata,
        model_dict,
        show_umap=False,
        show_pca=False,
    )

    lines_df: list[dict[str, Union[float, str, int]]] = []

    for model_name, model in model_dict.items():
        signature, cell_types = make_signature(
            adata,
            sc_per_pseudobulk=sc_per_purified_pseudobulk,
            aggregate_before_embedding=True,
            cell_type_column=cell_type_column,
            count_key=count_key,
            representation_key=f"X_scvi__{model_name}",
            model=model,
        )
        dimension_latent_space = signature.shape[0]
        number_cell_types = signature.shape[1]
        if max_non_zero is None:
            max_non_zero = dimension_latent_space

        for pseudobulk_number in range(pseudobulks_per_experiment):

            for n_non_zero in range(1, max_non_zero):
                proportions = create_random_proportion(
                    n_classes=number_cell_types, n_non_zero=n_non_zero
                )

                bulk_embedding = create_pseudobulk_embedding_from_proportions(
                    model,
                    adata,
                    cell_types,
                    proportions,
                    sc_per_evaluation_pseudobulk,
                    cell_type_column,
                    count_key,
                )
                predicted_proportions = nnls(signature, bulk_embedding)[0]
                predicted_proportions = (
                    predicted_proportions / predicted_proportions.sum()
                )
                cosine_similarity = (
                    np.dot(proportions, predicted_proportions)
                    / np.linalg.norm(proportions)
                    / np.linalg.norm(predicted_proportions)
                )
                lines_df.append(
                    {
                        "pseudobulk_number": pseudobulk_number,
                        "non_zero": n_non_zero,
                        "model_name": model_name,
                        "cosine_similarity": cosine_similarity,
                    }
                )
    return pd.DataFrame(lines_df)


def make_figure_from_dataframe(results_dataframe: pd.DataFrame, n_classes: int):
    """Make figure from experiment df designed above.

    Parameters
    ----------
    results_dataframe: pd.DataFrame
        The results dataframe
    n_classes: int
        The number of classes.

    """
    _, _ = plt.subplots(1, 1, figsize=(15, 10))

    for model_name in np.unique(results_dataframe["model_name"]):
        model_dataframe = results_dataframe[
            results_dataframe["model_name"] == model_name
        ][["cosine_similarity", "non_zero"]]
        mean_cosine_similarity_df = model_dataframe.groupby("non_zero").mean()
        non_zero_values = mean_cosine_similarity_df.index.to_numpy().flatten()
        mean_cosine_similarity = mean_cosine_similarity_df.to_numpy().flatten()
        std_cosine_similarity = (
            model_dataframe.groupby("non_zero").std().to_numpy().flatten()
        )
        plt.errorbar(
            non_zero_values,
            mean_cosine_similarity,
            yerr=std_cosine_similarity,
            fmt="o-",
            capsize=5,
            linestyle="--",
            marker="+",
            label=f"{model_name}",
        )

    random_estimate = estimate_random_cosine_similarity(n_classes=n_classes)
    plt.plot(
        non_zero_values,
        [random_estimate] * len(non_zero_values),
        linewidth=10,
        label="Random",
    )
    plt.legend()
    plt.xlabel("Number of mixed cell types")
    plt.ylabel("Cosine similarity")
    plt.xticks(list(non_zero_values))

    plt.title("Sanity check : simple deconvolution of mixtures using built signature.")
    plt.show()

"""All the constants used by run_benchmark.py to configure the pipeline."""

import importlib

def initialize_func(func_config: dict):
    """Initialize a function from a dict config.

    Parameters
    ----------
    func_config: dict
        The function dict config to initialize
    """
    target_path = func_config["_target_"]
    module_name, func_name = target_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    initialized_func = getattr(module, func_name)
    kwargs = {k: v for k, v in func_config.items() if k != "_target_"}
    return initialized_func, kwargs


############### Dict configs of functions to initialize. ###############


CORRELATION_FUNCTIONS = {
    "sample_wise_correlation": {
        "_target_": "benchmark_utils.compute_correlations",
        "deconv_results": None, 
        "ground_truth_fractions": None,
    },
    "cell_type_wise_correlation": {
        "_target_": "benchmark_utils.compute_group_correlations",
        "deconv_results": None, 
        "ground_truth_fractions": None,
    }, 
}

DATASETS = {
    "TOY": {
        "_target_": "",
    },
    "CTI": {
        "_target_": "benchmark_utils.load_cti",
        "n_variable_genes": None,
    },
    "BULK_FACS": {
        "_target_": "benchmark_utils.load_bulk_facs",
    },
}

DECONV_METHODS = {
    "MixUpVI": {
        "_target_": "benchmark_utils.MixUpVIMethod",
        "adata_train": None,
        "model_path": "",
        "cell_type_group": "cell_types_grouped",
        "save_model": False,
    },
    "NNLS": {
        "_target_": "benchmark_utils.NNLSMethod",
        "signature_matrix_name": "",
        "signature_matrix": None,
    },
    "scVI": {
        "_target_": "benchmark_utils.scVIMethod",
        "adata_train": None,
        "model_path": "",
        "save_model": False,
    },
    "DestVI": {
        "_target_": "benchmark_utils.DestVIMethod",
        "adata_train": None,
        "adata_pseudobulk": None,
        "model_path1": "",
        "model_path2": "",
        "cell_type_group": "cell_types_grouped",
        "save_model": False,
    },
    "TAPE": {
        "_target_": "benchmark_utils.TAPEMethod",
        "signature_matrix_name": "",
        "signature_matrix": None,
    },
    "Scaden": {
        "_target_": "benchmark_utils.ScadenMethod",
        "signature_matrix_name": "",
        "signature_matrix": None,
    }
}

EVALUATION_PSEUDOBULK_SAMPLINGS = {
    "PURIFIED": {
        "_target_": "benchmark_utils.create_purified_pseudobulk_dataset",
        "adata": None,
        "cell_type_group": "cell_types_grouped",
        "aggregation_method": "mean",
    },
    "UNIFORM": {
        "_target_": "benchmark_utils.create_uniform_pseudobulk_dataset",
        "adata": None,
        "n_sample": None,
        "n_cells": None,
        "cell_type_group": "cell_types_grouped",
        "aggregation_method": "mean",
    },
    "DIRICHLET": {
        "_target_": "benchmark_utils.create_dirichlet_pseudobulk_dataset",
        "adata": None,
        "prior_alphas": None,
        "n_sample": None,
        "n_cells": None,
        "cell_type_group": "cell_types_grouped",
        "is_n_cells_random": False,
        "add_sparsity": False,
    }
}


############### General constants. ###############


N_CELLS_EVALUATION_PSEUDOBULK_SAMPLINGS = {"UNIFORM", "DIRICHLET"}
TRAIN_DATASETS = {"CTI"}
SINGLE_CELL_DATASETS = {"TOY", "CTI"}
MODEL_TO_FIT = {"MixUpVI", "scVI", "DestVI"}
SIGNATURE_MATRIX_MODELS = {"NNLS", "TAPE", "Scaden"}
SINGLE_CELL_GRANULARITIES = {
    "1st_level_granularity", 
    "2nd_level_granularity", 
    "3rd_level_granularity", 
    "4th_level_granularity",
}
GRANULARITIES = SINGLE_CELL_GRANULARITIES.union({
    "FACS_1st_level_granularity",
})
SIGNATURE_TO_GRANULARITY = {
    "laughney": "1st_level_granularity",
    "CTI_1st_level_granularity": "1st_level_granularity",
    "CTI_2nd_level_granularity": "2nd_level_granularity",
    "CTI_3rd_level_granularity": "3rd_level_granularity",
    "CTI_4th_level_granularity": "4th_level_granularity",
    "FACS_1st_level_granularity": "FACS_1st_level_granularity",
}
GRANULARITY_TO_TRAINING_DATASET = {
    "1st_level_granularity": "CTI",
    "2nd_level_granularity": "CTI",
    "3rd_level_granularity": "CTI",
    "4th_level_granularity": "CTI",
    "FACS_1st_level_granularity": "CTI",
    # add the one for TOY
}
GRANULARITY_TO_EVALUATION_DATASET = {
    "1st_level_granularity": "CTI",
    "2nd_level_granularity": "CTI",
    "3rd_level_granularity": "CTI",
    "4th_level_granularity": "CTI",
    "FACS_1st_level_granularity": "BULK_FACS",
    # add the one for TOY
}
DECONV_METHOD_TO_EVALUATION_PSEUDOBULK = {
    "NNLS": "adata_pseudobulk_test_rc",
    "TAPE": "adata_pseudobulk_test_rc",
    "Scaden": "adata_pseudobulk_test_rc",
    "MixUpVI": "adata_pseudobulk_test_counts",
    "scVI": "adata_pseudobulk_test_counts",
    "DestVI": "adata_pseudobulk_test_counts",
}
TRAINING_CONSTANTS_TO_SAVE = [
    "LATENT_SIZE",
    "MAX_EPOCHS",
    "BATCH_SIZE",
    "TRAIN_SIZE",
    "CHECK_VAL_EVERY_N_EPOCH",
    "N_PSEUDOBULKS",
    "N_CELLS_PER_PSEUDOBULK",
    "N_HIDDEN",
    "CONT_COV",
    "CAT_COV",
    "ENCODE_COVARIATES",
    "LOSS_COMPUTATION",
    "PSEUDO_BULK",
    "SIGNATURE_TYPE",
    "MIXUP_PENALTY",
    "DISPERSION",
    "GENE_LIKELIHOOD",
    "USE_BATCH_NORM",
]
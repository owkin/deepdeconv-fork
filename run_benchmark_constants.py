"""All the constants used by run_benchmark.py to configure the pipeline."""

DECONV_METHODS = {
    "scVI": {
        "_target_": "",
        "adata": None,
        "model_path": "",
        "save_model": False,
    },
    "DestVI": {
        "_target_": "",
        "adata": None,
        "prior_alphas": None,
        "n_sample": None,
        "model_path1": "",
        "model_path2": "",
        "cell_type_key": "",
        "save_model": False,
    },
    "MixUpVI": {
        "_target_": "run_benchmark_help.MixUpVIMethod",
        "adata_train": None,
        "model_path": "",
        "cell_type_group": "cell_types_grouped",
        "save_model": False,
    },
    "NNLS": {
        "_target_": "run_benchmark_help.NNLSMethod",
        "signature_matrix": None,
    },
    "TAPE": {
        "_target_": "",
    },
    "Scaden": {
        "_target_": "",
    }
}

DATASETS = { # something that takes preprocessing into account !! (not train/test split yet though)
    "TOY": {
        "_target_": "",
    },
    "CTI": {
        "_target_": "run_benchmark_help.load_cti",
        "n_variable_genes": None,
    },
    "BULK_FACS": {
        "_target_": "run_benchmark_help.load_bulk_facs",
    },
}

EVALUATION_PSEUDOBULK_SAMPLINGS = {"PURIFIED", "UNIFORM", "DIRICHLET"}
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
GRANULARITY_TO_DATASET = {
    "1st_level_granularity": "CTI",
    "2nd_level_granularity": "CTI",
    "3rd_level_granularity": "CTI",
    "4th_level_granularity": "CTI",
    "FACS_1st_level_granularity": "CTI",
    # add the one for TOY
}
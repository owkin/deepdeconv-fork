from ray import tune

from constants import (
    GROUPS, 
    TRAINING_CELL_TYPE_GROUP,
    USE_BATCH_NORM,
    SIGNATURE_TYPE,
    LOSS_COMPUTATION,
    PSEUDO_BULK,
    ENCODE_COVARIATES,
    ENCODE_CONT_COVARIATES,
    MIXUP_PENALTY,
    DISPERSION,
    GENE_LIKELIHOOD,
    TRAIN_SIZE,
    BATCH_SIZE,
)


### search space to define: the only thing to change

example_search_space = {
    "n_hidden": tune.choice([64, 128, 256]),
    "n_layers": tune.choice([1, 2, 3]),
    "lr": tune.loguniform(1e-4, 1e-2),
}
latent_space_search_space = {
    "n_latent": tune.grid_search(
        list(range(len(GROUPS[TRAINING_CELL_TYPE_GROUP]) - 1, 550, 20)) # from n cell types to n marker genes
    )
}
SEARCH_SPACE = latent_space_search_space
TUNED_VARIABLES = list(SEARCH_SPACE.keys())
NUM_SAMPLES = 1 # will only perform once the gridsearch (useful to change if mix of grid and random search for instance)


### add the model and training fixed hyperparameters 
model_fixed_hps = {
    "use_batch_norm": USE_BATCH_NORM,
    "signature_type": SIGNATURE_TYPE,
    "loss_computation": LOSS_COMPUTATION,
    "pseudo_bulk": PSEUDO_BULK,
    "encode_covariates": ENCODE_COVARIATES,
    "encode_cont_covariates": ENCODE_CONT_COVARIATES,
    "mixup_penalty": MIXUP_PENALTY,
    "dispersion": DISPERSION,
    "gene_likelihood": GENE_LIKELIHOOD,
    "train_size": TRAIN_SIZE,
    "batch_size": BATCH_SIZE,
}
for key in model_fixed_hps.keys():
    # don't replace the search space by fixed hyperparemeter value
    if key in TUNED_VARIABLES:
        del model_fixed_hps[key]
SEARCH_SPACE = SEARCH_SPACE | model_fixed_hps


### all metrics to callback during tuning
METRIC = "validation_loss"
ADDITIONAL_METRICS = None
ADDITIONAL_METRICS = [
    # val metrics
    "mixup_penalty_validation",
    "reconstruction_loss_validation",
    "kl_local_validation",
    "pearson_coeff_validation",
    "cosine_similarity_validation",
    "pearson_coeff_deconv_validation",
    "pearson_coeff_random_validation",
    # train metrics
    "train_loss_epoch",
    "mixup_penalty_train",
    "reconstruction_loss_train",
    "kl_local_train",
    "pearson_coeff_train",
    "cosine_similarity_train",
    "pearson_coeff_deconv_train",
    "pearson_coeff_random_train",
]

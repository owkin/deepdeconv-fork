"""Hyperparameter search configs."""

from ray import tune

from constants import (
    GROUPS,
    TRAINING_CELL_TYPE_GROUP,
    USE_BATCH_NORM,
    SIGNATURE_TYPE,
    LOSS_COMPUTATION,
    PSEUDO_BULK,
    ENCODE_COVARIATES,
    MIXUP_PENALTY,
    DISPERSION,
    GENE_LIKELIHOOD,
    TRAIN_SIZE,
    BATCH_SIZE,
    LATENT_SIZE,
    N_PSEUDOBULKS,
    N_CELLS_PER_PSEUDOBULK,
    SEED,
)


### search space to define: the only thing to change

example_search_space = {
    "n_hidden": tune.choice([64, 128, 256]),
    "n_layers": tune.choice([1, 2, 3]),
    "lr": tune.loguniform(1e-4, 1e-2),
}
repeat_with_several_seeds = {
    "seed": tune.grid_search(
        [0, 3, 8, 12, 23]
        # [0,1]
    )
}
example_with_several_seeds = {
    "n_latent": tune.grid_search([30, 100]),
    "seed": tune.grid_search([3, 8, 12])
}
latent_space_search_space = {
    "n_latent": tune.grid_search(
        # list(range(len(GROUPS[TRAINING_CELL_TYPE_GROUP]) - 1, 550, 20)) # from n cell types to n marker genes
        [10, 20, 30, 40, 50, 100, 200]
    ),
    "seed": tune.grid_search([3, 8, 12, 23, 42])
}
batch_size_search_space = {
    "batch_size": tune.grid_search(
        [128, 256, 512, 1024, 2048, 5000, 10000]
    )
}
pseudo_bulk_search_space = {
    "pseudo_bulk": tune.grid_search(
        ["pre_encoded", "post_inference"]
    )
}
signature_type_search_space = {
    "signature_type": tune.grid_search(
        ["pre_encoded", "post_inference"]
    ),
    "seed": tune.grid_search([3, 8, 12, 23, 42])
}
loss_computation_search_space = {
    "loss_computation": tune.grid_search(
        ["latent_space", "reconstructed_space"]
    )
}
gene_likelihood_search_space = {
    "gene_likelihood": tune.grid_search(["zinb", "nb", "poisson"]),
    "seed": tune.grid_search([3, 8, 12, 23, 42])
}
n_hidden_search_space = {
    "n_hidden": tune.grid_search([128, 256, 512, 1024]),
    "seed": tune.grid_search([3, 8, 12, 23, 42])
}
n_layers_search_space = {
    "n_layers": tune.grid_search([1, 2, 3]),
    "seed": tune.grid_search([3, 8, 12, 23, 42])
}
n_pseudobulks_search_space = {
    "n_pseudobulks": tune.grid_search([1, 100]),
    "seed": tune.grid_search([3, 8, 12])
   #  "seed": tune.grid_search([3, 8, 12, 23, 42])
}
n_cells_per_pseudobulk_search_space = {
    "n_cells_per_pseudobulk": tune.grid_search([100, 256, 512, 1024, 2048]),
    "seed": tune.grid_search([3, 8, 12])
}
SEARCH_SPACE = n_layers_search_space
TUNED_VARIABLES = list(SEARCH_SPACE.keys())
NUM_SAMPLES = 1 # will only perform once the gridsearch (useful to change if mix of grid and random search for instance)


### add the model and training fixed hyperparameters
model_fixed_hps = {
    "use_batch_norm": USE_BATCH_NORM,
    "signature_type": SIGNATURE_TYPE,
    "loss_computation": LOSS_COMPUTATION,
    "pseudo_bulk": PSEUDO_BULK,
    "encode_covariates": ENCODE_COVARIATES,
    "mixup_penalty": MIXUP_PENALTY,
    "dispersion": DISPERSION,
    "gene_likelihood": GENE_LIKELIHOOD,
    "train_size": TRAIN_SIZE,
    "batch_size": BATCH_SIZE,
    "n_latent": LATENT_SIZE,
    "n_pseudobulks": N_PSEUDOBULKS,
    "n_cells_per_pseudobulk": N_CELLS_PER_PSEUDOBULK,
    "seed": SEED,
}
for key in list(model_fixed_hps):
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
    "mse_deconv_validation",
    "mae_deconv_validation",
    # train metrics
    "train_loss_epoch",
    "mixup_penalty_train",
    "reconstruction_loss_train",
    "kl_local_train",
    "pearson_coeff_train",
    "cosine_similarity_train",
    "pearson_coeff_deconv_train",
    "mse_deconv_train",
    "mae_deconv_train",
]
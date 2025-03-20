"""Benchmark config class for a benchmark run."""

import json
import os
import shutil
from loguru import logger
from dataclasses import asdict
from datetime import datetime
from pydantic.dataclasses import dataclass
from typing import Optional
import yaml
from zoneinfo import ZoneInfo

import constants
from run_benchmark_constants import (
    DATASETS,
    DECONV_METHODS,
    EVALUATION_PSEUDOBULK_SAMPLINGS,
    GRANULARITIES,
    GRANULARITY_TO_TRAINING_DATASET,
    GRANULARITY_TO_EVALUATION_DATASET,
    MODEL_TO_FIT, 
    N_CELLS_EVALUATION_PSEUDOBULK_SAMPLINGS,
    SIGNATURE_MATRIX_MODELS,
    SIGNATURE_TO_GRANULARITY,
    SINGLE_CELL_DATASETS,
    SINGLE_CELL_GRANULARITIES,
    TRAIN_DATASETS,
    TRAINING_CONSTANTS_TO_SAVE,
)

def save_experiment(experiment_name: str):
    """Save the logs and experiment outputs.

    Parameters
    ----------
    experiment_name: str
        The experiment directory name, unique to each experiment.
    
    Returns
    -------
    full_path: str
        The full experiment path.
    """
    output_dir = "/home/owkin/project/run_benchmark_experiments"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        logger.debug(f"Created output dir: {output_dir}")
    full_path = f"{output_dir}/{experiment_name}"
    if not os.path.exists(full_path):
        os.mkdir(full_path)
        logger.debug(
            f"Created output dir: {full_path}"
        )
    elif "experiment_over.txt" in os.listdir(f"{full_path}"):
        raise ValueError(
            f"An experiment was already finished at {full_path}. If you wish to "
            "run a new experiment, please find another experiment name."
        )
    else:
        shutil.rmtree(full_path)
        os.mkdir(full_path)
        logger.warning(
            "An experiment was already started but not finished at the path "
            f"{full_path}. It is now deleted."
        )
    # Save logs
    logger.add(f"{full_path}/logs.txt")
    logger.add(
        f"{full_path}/warnings.txt",
        level="WARNING",  # also keep the warnings separate
    )

    return full_path

@dataclass
class RunBenchmarkConfig:
    """Full configuration for a benchmark deconvolution experiment.

    Parameters
    ----------
    deconv_methods: list
        All the deconvolution methods to compare to evaluate deconvolution.
    evaluation_datasets: list
        All the evaluation datasets on which to evaluate the deconvolution performance
        of the deconvolution methods. Can be single cell datasets (for pseudobulk
        simulations) or real bulk/facs data.
    granularities: list
        The granularities of the deconvolution to evaluate.
    evaluation_pseudobulk_samplings: list
        All type of samplings to perform for simulations on the single cell data in case
        single cell datasets were given to evaluation datasets.
    n_cells_per_evaluation_pseudobulk: list
        The number of cells to use to create the evaluation pseudobulks. A list of cells can
        be passed, to analyse the correlation of deconvolution performance with the number 
        of cells of the pseudobulks.
    signature_matrices: list
        The signature matrices to use for the methods requiring one.
    train_dataset: str
        The train dataset to use in case some deconvolution methods (usually scvi-verse 
        models) need fitting before evaluation. On the contrary, this is not the case 
        for models like NNLS, because the signature matrix was constructed on a train 
        dataset separately.
    n_variable_genes: int
        The number of most highly variable genes in case some deconvolution methods 
        (usually scvi-verse models) need to filter genes based on variance prior to 
        fitting and evaluation.
    save: bool
        Whether to save the deconvolution experiment outputs. If False, only the plots will be saved.
    experiment_name: str
        The experiment directory name, unique to each experiment.
    """
    deconv_methods: list
    evaluation_datasets: list
    granularities: list
    evaluation_pseudobulk_samplings: Optional[list]
    n_samples_evaluation_pseudobulk: Optional[int]
    n_cells_per_evaluation_pseudobulk: Optional[list]
    signature_matrices: Optional[list]
    train_dataset: Optional[str]
    n_variable_genes: Optional[int]
    save: bool
    experiment_name: Optional[str]

    @classmethod
    def from_config_yaml(cls, config_path: str):
        """Return a dictionnary mapping evaluation config keys to their value."""
        with open(config_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)

        # Check types
        config_dict = asdict(
            cls(
                deconv_methods = config_dict["deconv_methods"],
                evaluation_datasets = config_dict["evaluation_datasets"],
                granularities = config_dict["granularities"],
                evaluation_pseudobulk_samplings = config_dict[
                    "evaluation_pseudobulk_samplings"],
                n_samples_evaluation_pseudobulk = config_dict[
                    "n_samples_evaluation_pseudobulk"
                ],
                n_cells_per_evaluation_pseudobulk = config_dict[
                    "n_cells_per_evaluation_pseudobulk"
                ],
                signature_matrices = config_dict["signature_matrices"],
                train_dataset = config_dict["train_dataset"],
                n_variable_genes = config_dict["n_variable_genes"],
                save = config_dict["save"],
                experiment_name = config_dict["experiment_name"],
            )
        )
        
        # Save experiment
        if config_dict["experiment_name"] is None:
            paris_tz = ZoneInfo("Europe/Paris")
            config_dict["experiment_name"] = datetime.now(paris_tz).strftime(
                "%Y-%m-%d-%H-%M-%S"
            )
        full_path = save_experiment(config_dict["experiment_name"])
        config_dict["experiment_name"] = full_path

        # Test and homogenise missing arguments
        if len(config_dict["deconv_methods"]) == 0:
            message = (
                "You should provide at least one deconvolution method to run an "
                "experiment."
            )
            logger.error(message)
            raise ValueError(message)
        if len(config_dict["evaluation_datasets"]) == 0:
            message = (
                "You should provide at least one evaluation dataset to run an "
                "experiment."
            )
            logger.error(message)
            raise ValueError(message)
        if len(config_dict["granularities"]) == 0:
            message = (
                "You should provide at least one granularity to run an "
                "experiment."
            )
            logger.error(message)
            raise ValueError(message)
        if config_dict["evaluation_pseudobulk_samplings"] is not None and \
            len(config_dict["evaluation_pseudobulk_samplings"]) == 0:
            config_dict["evaluation_pseudobulk_samplings"] = None
        if config_dict["n_cells_per_evaluation_pseudobulk"] is not None and \
            len(config_dict["n_cells_per_evaluation_pseudobulk"]) == 0:
            config_dict["n_cells_per_evaluation_pseudobulk"] = [None]
        if config_dict["signature_matrices"] is not None and \
            len(config_dict["signature_matrices"]) == 0:
            config_dict["signature_matrices"] = None
        if config_dict["train_dataset"] is not None and \
            len(config_dict["train_dataset"]) == 0:
            config_dict["train_dataset"] = None

        # Check that all provided arguments exist
        if not set(config_dict["deconv_methods"]).issubset(DECONV_METHODS):
            message = (
                "Only the following deconvolution methods can be used: "
                f"{DECONV_METHODS}"
            )
            logger.error(message)
            raise NotImplementedError(message)
        if not set(config_dict["evaluation_datasets"]).issubset(DATASETS):
            message = (
                "Only the following evaluation datasets can be used: "
                f"{DATASETS}"
            )
            logger.error(message)
            raise NotImplementedError(message)
        if not set(config_dict["evaluation_pseudobulk_samplings"]).issubset(
            EVALUATION_PSEUDOBULK_SAMPLINGS.keys()
        ):
            message = (
                "Only the following evaluation pseudobulk samplings can be used: "
                f"{list(EVALUATION_PSEUDOBULK_SAMPLINGS.keys())}"
            )
            logger.error(message)
            raise NotImplementedError(message)
        if not set(config_dict["granularities"]).issubset(GRANULARITIES):
            message = (
                "Only the following granularities can be used: "
                f"{GRANULARITIES}"
            )
            logger.error(message)
            raise NotImplementedError(message)
        if not set(config_dict["signature_matrices"]).issubset(
            set(SIGNATURE_TO_GRANULARITY.keys())
        ):
            message = (
                "Only the following signature matrices can be used: "
                f"{set(SIGNATURE_TO_GRANULARITY.keys())}"
            )
            logger.error(message)
            raise NotImplementedError(message)
        if config_dict["train_dataset"] not in TRAIN_DATASETS:
            message = (
                "Only the following train datasets can be used: "
                f"{TRAIN_DATASETS}"
            )
            logger.error(message)
            raise NotImplementedError(message)

        # Check for missing arguments
        if len(intersection := set(config_dict["deconv_methods"]).intersection(
            MODEL_TO_FIT)) > 0:
            if config_dict["train_dataset"] is None:
                message = (
                    "train_dataset must be provided when models that require fitting "
                    f"({intersection}) are provided as deconvolution methods."
                )
                logger.error(message)
                raise ValueError(message)
            if config_dict["n_variable_genes"] is None:
                message = (
                    "n_variable_genes must be provided when models that require to be "
                    f"filtered to the most variables genes ({intersection}) are "
                    "provided as deconvolution methods."
                )
                logger.error(message)
                raise ValueError(message)
        if len(intersection := set(config_dict["evaluation_datasets"]).intersection(
            SINGLE_CELL_DATASETS)) > 0 and config_dict[
            "evaluation_pseudobulk_samplings"
            ] is None:
            message = (
                "evaluation_pseudobulk_samplings strategies must be provided when "
                f"single cell datasets ({intersection}) are provided as evaluation "
                "datasets."
            )
            logger.error(message)
            raise ValueError(message)
        if config_dict["evaluation_pseudobulk_samplings"] is not None and len(
            intersection := set(config_dict["evaluation_pseudobulk_samplings"]).intersection(
            N_CELLS_EVALUATION_PSEUDOBULK_SAMPLINGS)) > 0 and config_dict[
                "n_cells_per_evaluation_pseudobulk"
            ] == [None]:
            message = (
                "Some provided evaluation_pseudobulk_samplings methods ("
                f"{intersection}) require a number of cells to create the "
                "pseudobulks, but n_cells_per_evaluation_pseudobulk was not "
                "provided."
            )
            logger.error(message)
            raise ValueError(message)
        if config_dict["evaluation_pseudobulk_samplings"] is not None and len(
            intersection := set(config_dict["evaluation_pseudobulk_samplings"]).intersection(
            N_CELLS_EVALUATION_PSEUDOBULK_SAMPLINGS)) > 0 and config_dict[
                "n_samples_evaluation_pseudobulk"
            ] is None:
            message = (
                "Some provided evaluation_pseudobulk_samplings methods ("
                f"{intersection}) require a number of samples to create the "
                "pseudobulks, but n_samples_evaluation_pseudobulk was not "
                "provided."
            )
            logger.error(message)
            raise ValueError(message)
        if len(intersection := set(config_dict["deconv_methods"]).intersection(
            SIGNATURE_MATRIX_MODELS)) > 0:
            if config_dict["signature_matrices"] is None:
                message = (
                    "No signature matrix was provided even though some methods "
                    f"({intersection}) require a signature matrix."
                )
                logger.error(message)
                raise ValueError(message)
            for signature_matrix in config_dict["signature_matrices"]:
                if SIGNATURE_TO_GRANULARITY[signature_matrix] not in \
                    config_dict["granularities"]:
                    message = (
                        "Signature matrix's associated granularity (signature matrix: "
                        f"{signature_matrix}, associated granularity: "
                        f"{SIGNATURE_TO_GRANULARITY[signature_matrix]}) does not match "
                        "any of the provided granularities: "
                        f"{config_dict['granularities']}."
                    )           
                    logger.error(message)
                    raise ValueError(message)
        for granularity in config_dict["granularities"]:
            if GRANULARITY_TO_TRAINING_DATASET[granularity] != config_dict["train_dataset"]:
                message = (
                    f"The provided granularity {granularity} should be trained on "
                    f"the following dataset: {GRANULARITY_TO_TRAINING_DATASET['granularity']}."
                    " However, the provided training dataset ("
                    f"{config_dict['train_dataset']}) is different."
                )
            if GRANULARITY_TO_EVALUATION_DATASET[granularity] not in \
                config_dict["evaluation_datasets"]:
                message = (
                    f"The provided granularity {granularity} should be evaluated on "
                    f"the following dataset: {GRANULARITY_TO_EVALUATION_DATASET['granularity']}."
                    "However, none of the evaluation datasets ("
                    f"{config_dict['evaluation_datasets']}) contain this dataset."
                )
        
        # Check for useless arguments
        if len(intersection := set(config_dict["deconv_methods"]).intersection(
            MODEL_TO_FIT)) == 0:
            if config_dict["train_dataset"] is not None:
                logger.warning(
                    "A train dataset was provided even though none of the provided "
                    f"deconvolution methods ({config_dict['deconvolution_methods']}) "
                    "require fitting. Thus, train_dataset will not be used."
                )
                config_dict["train_dataset"] = None
            if config_dict["n_variable_genes"] is not None:
                logger.warning(
                    "n_variable_genes was provided even though none of the provided "
                    f"deconvolution methods ({config_dict['deconvolution_methods']}) "
                    "require to be filtered to their most variable genes. Thus, this "
                    "argument will not be used."
                )
                config_dict["n_variable_genes"] = None
        if (config_dict["evaluation_pseudobulk_samplings"] is None or len(set(
            config_dict["evaluation_pseudobulk_samplings"]).intersection(
            N_CELLS_EVALUATION_PSEUDOBULK_SAMPLINGS)) == 0) and config_dict[
                "n_cells_per_evaluation_pseudobulk"
            ] != [None]:
            logger.warning(
                "n_cells_per_evaluation_pseudobulk was provived ("
                f"{config_dict['n_cells_per_evaluation_pseudobulk']}) even though "
                "no evaluation_pseudobulk_samplings method requiring it was provided."
                " Thus, this argument will not be used."
            )
            config_dict["n_cells_per_evaluation_pseudobulk"] = [None]
        if (config_dict["evaluation_pseudobulk_samplings"] is None or len(set(
            config_dict["evaluation_pseudobulk_samplings"]).intersection(
            N_CELLS_EVALUATION_PSEUDOBULK_SAMPLINGS)) == 0) and config_dict[
                "n_samples_evaluation_pseudobulk"
            ] is not None:
            logger.warning(
                "n_samples_evaluation_pseudobulk was provived ("
                f"{config_dict['n_samples_evaluation_pseudobulk']}) even though "
                "no evaluation_pseudobulk_samplings method requiring it was provided."
                " Thus, this argument will not be used."
            )
            config_dict["n_cells_per_evaluation_pseudobulk"] = [None]
        if len(set(config_dict["evaluation_datasets"]).intersection(
            SINGLE_CELL_DATASETS)) == 0 and config_dict[
            "evaluation_pseudobulk_samplings"
            ] is not None:
            logger.warning(
                "evaluation_pseudobulk_samplings strategies were provided even though "
                "none of the provided evaluation datasets ("
                f"{config_dict['evaluation_datasets']}) are single cell and thus "
                "none can be sampled for simulations. Thus, this argument will not be "
                "used."
            )
            config_dict["evaluation_pseudobulk_samplings"] = None
        if len(set(config_dict["deconv_methods"]).intersection(
            SIGNATURE_MATRIX_MODELS)) == 0 and config_dict["signature_matrices"] \
            is not None:
            logger.warning(
                "A signature matrix was provided even though none of the provided "
                "deconvolution methods require one. Thus, this argument will not be "
                "used."
            )
            config_dict["signature_matrices"] = None

        # Check for likely long running time
        if len(intersection1:=set(config_dict["deconv_methods"]).intersection(
                MODEL_TO_FIT
            ))> 0 and len(intersection2:=set(config_dict["granularities"]).intersection(
                SINGLE_CELL_GRANULARITIES
            )) > 1:
            logger.warning(
                f"Some deconvolution methods ({intersection1}) need fitting, and "
                "several granularities needing fitting were provided for evaluation "
                f"({intersection2}). Training on several granularities can take time."
            )

        if config_dict["save"]:
            # Save general config
            config_path = full_path + "/config.json"
            with open(config_path, "w", encoding="utf-8") as json_file:
                json.dump(config_dict, json_file)
            logger.debug(f"Saved config dict to {config_path}")
            # Save training config
            if "MixUpVI" in config_dict["deconv_methods"]:
                training_constants_to_save = TRAINING_CONSTANTS_TO_SAVE
            elif "scVI" in config_dict["deconv_methods"] or "DestVI" in config_dict[
                "deconv_methods"
            ]:
                training_constants_to_save = ["LATENT_SIZE", "MAX_EPOCHS"]
            training_config = {
                key: getattr(constants, key)
                for key in training_constants_to_save if hasattr(constants, key)
            }
            config_path = full_path + "/training_config.json"
            with open(config_path, "w", encoding="utf-8") as json_file:
                json.dump(training_config, json_file)
            logger.debug(f"Saved config dict to {config_path}")

        return config_dict


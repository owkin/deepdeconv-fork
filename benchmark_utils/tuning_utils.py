"""Tuning utils file."""

import json
from collections import defaultdict
import numpy as np
import pandas as pd
import os
import pickle
from tuning_configs import TUNED_VARIABLES, SEARCH_SPACE, METRIC, ADDITIONAL_METRICS
from constants import TRAINING_DATASET

def format_and_save_tuning_results(
        tuning_results, variables: str, training_dataset : str, cat_cov : list, cont_cov : list,
):
    """Format the tuning results and save them in the project directory."""
    # format the results of all experiments
    keys = list(tuning_results.results[0].metrics.keys())
    all_metrics = keys[:keys.index("timestamp")]
    all_results = []
    for path in tuning_results.results: # loop through every result of hyperparameters tried
        path = path.path
        results = defaultdict(list)
        with open(path+"/result.json", "r") as ff:
            for line in ff:
                # loop through every epoch of the training
                data = json.loads(line.strip())
                for key in all_metrics:
                    if key in data:
                        results[key].append(data[key])
                    else:
                        results[key].append(np.nan)
        results = pd.DataFrame(results)
        
        hyperparameters = path.split("/")[-1]
        for i, variable in enumerate(variables):
            hyperparameters=hyperparameters.split(f"{variable}=")[1]
            if i < len(variables)-1:
                value = hyperparameters.split(",")[0]
            else:
                value = hyperparameters.split("-")[0][:-5]
            results[variable] = value
        
        all_results.append(results)
    
    all_results = pd.concat(all_results)
    best_hp = {}
    for variable in variables:
        if all_results[variable].str.isnumeric().all():
            all_results[variable] = pd.to_numeric(all_results[variable])
        if variable in tuning_results.model_kwargs:
            best_hp[variable] = tuning_results.model_kwargs[variable] # best hp found by tuning main metric
        elif variable in tuning_results.train_kwargs:
            best_hp[variable] = tuning_results.train_kwargs[variable] # best hp found by tuning main metric
        else:
            best_hp[variable] = None
    # save results and search space
    save_dir = f"/home/owkin/project/mixupvi_tuning/{'-'.join(variables)}/"
    new_path = save_dir + f"{training_dataset}_dataset_{path.split('/')[5]}"
    if not os.path.exists(save_dir):
        # create a directory for the variable tuned
        os.makedirs(save_dir)
    if not os.path.exists(new_path):
        # create a directory for the specific grid search performed
        os.makedirs(new_path)
    tuning_path = f"{new_path}/tuning_results.csv"
    search_path = f"{new_path}/search_space.pkl"
    all_results.to_csv(tuning_path)

    search_space = tuning_results.search_space
    search_space["cat_cov"] = cat_cov
    search_space["cont_cov"] = cont_cov
    search_space["best_hp"] = best_hp
    with open(search_path, "wb") as ff:
        pickle.dump(search_space, ff)
    
    return all_results, best_hp, tuning_path, search_path


def read_tuning_results(tuning_path):
    return pd.read_csv(tuning_path, index_col = 0)


def read_search_space(search_path):
    with open(search_path, "rb") as ff:
        search_space = pickle.load(ff)
    return search_space


def format_and_save_tuning_results_backup(ray_directory: str = "tune_mixupvi_2024-04-08-08:55:24"):
    """This function essentially does the same as format_and_save_tuning_results.
    
    But this one should be used in a handcrafted manner (by providing the ray directory 
    saved locally) when for some reason, tuning results were successfully saved locally
    by ray, but not formatted and saved in the shared /project folder.

    Five global variables are used here and should be specified accordingly in the 
    tuning and constants config files : TUNED_VARIABLES, SEARCH_SPACE, TRAINING_DATASET,
    METRIC, ADDITIONAL_METRICS
    """
    directory = f"/home/owkin/deepdeconv-fork/ray/{ray_directory}/"
    all_metrics = [METRIC] + ADDITIONAL_METRICS # all metric columns we want to retrieve

    all_results = []
    for path in os.listdir(directory): # loop through every result of hyperparameters tried
        if path.startswith("_trainable"):
            path = directory + path
            results = defaultdict(list)
            with open(path+"/result.json", "r") as ff:
                for line in ff:
                    # loop through every epoch of the training
                    data = json.loads(line.strip())
                    for key in all_metrics:
                        if key in data:
                            results[key].append(data[key])
                        else:
                            results[key].append(np.nan)
            results = pd.DataFrame(results)

            hyperparameters = path.split("/")[-1]
            for i, variable in enumerate(sorted(TUNED_VARIABLES)):
                hyperparameters=hyperparameters.split(f"{variable}=")[1]
                if i < len(TUNED_VARIABLES)-1:
                    value = hyperparameters.split(",")[0]
                else:
                    value = hyperparameters.split("-")[0][:-5]
                results[variable] = value
            
            all_results.append(results)
    
    all_results = pd.concat(all_results)
    # save results and search space
    save_dir = f"/home/owkin/project/mixupvi_tuning/{'-'.join(TUNED_VARIABLES)}/"
    new_path = save_dir + f"{TRAINING_DATASET}_dataset_{ray_directory}"
    if not os.path.exists(save_dir):
        # create a directory for the variable tuned
        os.makedirs(save_dir)
    if not os.path.exists(new_path):
        # create a directory for the specific grid search performed
        os.makedirs(new_path)
    tuning_path = f"{new_path}/tuning_results.csv"
    search_path = f"{new_path}/search_space.pkl"
    all_results.to_csv(tuning_path)

    search_space = SEARCH_SPACE
    with open(search_path, "wb") as ff:
        pickle.dump(search_space, ff)

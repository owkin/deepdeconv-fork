import json
from collections import defaultdict
import numpy as np
import pandas as pd
import os
import pickle

def format_and_save_tuning_results(tuning_results, variables: str, training_dataset : str):
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
        
        hyperparameters = path.split("/")[6]
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

    
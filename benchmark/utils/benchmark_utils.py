"""Deconvolution benchmark utilities."""

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import random

from typing import Tuple
import scipy.stats
from sklearn.linear_model import LinearRegression

def perform_nnls(signature, averaged_data):
    """Perform deconvolution using the nnls method.
    It will be performed as many times as there are samples in averaged_data.
    """
    deconv = LinearRegression(positive=True).fit(signature, averaged_data.T)
    deconv_results = pd.DataFrame(
        deconv.coef_, index=averaged_data.index, columns=signature.columns
    )
    deconv_results = deconv_results.div(
        deconv_results.sum(axis=1), axis=0
    )  # to sum up to 1
    return deconv_results


def compute_correlations(deconv_results, ground_truth_fractions):
    """Compute n_sample pairwise correlations between the deconvolution results and the
    ground truth fractions.
    """
    deconv_results = deconv_results[ground_truth_fractions.columns] # to align order of columsn
    correlations = [scipy.stats.pearsonr(ground_truth_fractions.iloc[i],
                                         deconv_results.iloc[i]).statistic
                                         for i in range(len(deconv_results))]
    correlations = pd.DataFrame({"correlations": correlations})
    correlations["Method"] = "nnls"  # add all the deconv methods
    return correlations

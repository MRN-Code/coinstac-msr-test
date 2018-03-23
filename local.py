#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation
"""
import json
import numpy as np
import sys
import regression as reg
import warnings
from parsers import fsl_parser
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm


def listRecursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in listRecursive(v, key):
                yield found
        if k == key:
            yield v


def local_0(args):

    input_list = args["input"]
    lamb = input_list["lambda"]

    (X, y, y_labels) = fsl_parser(args)
    beta_vec_size = X.shape[1] + 1
#    # considering only one regression given the challenges in
#    # multi-shot regressin with multiple regressions
#    y = pd.DataFrame(y.loc[:, y.columns[0]])
#    y_labels = [y_labels[0]]
#
#    biased_X = sm.add_constant(X)
#    biased_X = biased_X.values
#
#    beta_vector, meanY_vector, lenY_vector = [], [], []
#
#    local_params = []
#    local_sse = []
#    local_pvalues = []
#    local_tvalues = []
#    local_rsquared = []
#
#    for column in y.columns:
#        curr_y = list(y[column])
#        beta = reg.one_shot_regression(biased_X, curr_y, lamb)
#        beta_vector.append(beta.tolist())
#        meanY_vector.append(np.mean(curr_y))
#        lenY_vector.append(len(y))
#
#        # Printing local stats as well
#        model = sm.OLS(curr_y, biased_X.astype(float)).fit()
#        local_params.append(model.params)
#        local_sse.append(model.ssr)
#        local_pvalues.append(model.pvalues)
#        local_tvalues.append(model.tvalues)
#        local_rsquared.append(model.rsquared_adj)
#        break
#
#    keys = ["beta", "sse", "pval", "tval", "rsquared"]
#    dict_list = []
#    for index, _ in enumerate(y_labels):
#        values = [
#            local_params[index].tolist(), local_sse[index],
#            local_pvalues[index].tolist(), local_tvalues[index].tolist(),
#            local_rsquared[index]
#        ]
#        local_stats_dict = {key: value for key, value in zip(keys, values)}
#        dict_list.append(local_stats_dict)

    X = X.values
    y = y.values

    dict_list = 0

    computation_output = {
        "output": {
            "mean_y_local": np.mean(y),
            "count_local": len(y),
            "beta_vec_size": beta_vec_size,
            "local_stats_dict": dict_list,
            "computation_phase": "local_0"
        },
        "cache": {
            "covariates": X.tolist(),
            "dependents": y.tolist(),
            "lambda": lamb
        }
    }

    return json.dumps(computation_output)


def local_1(args):

    X = args["cache"]["covariates"]
    y = args["cache"]["dependents"]
    lamb = args["cache"]["lambda"]
    biased_X = sm.add_constant(X)

    wp = args["input"]["remote_beta"]

    gradient = (1 / len(X)) * np.dot(biased_X.T, np.dot(biased_X, wp) - y)

    computation_phase = {
        "cache": {
            "covariates": X,
            "dependents": y,
            "lambda": lamb
        },
        "output": {
            "local_grad": gradient.tolist(),
            "computation_phase": "local_1"
        }
    }

    return json.dumps(computation_phase)


def local_2(args):
    input_list = args["cache"]
    X = input_list["covariates"]
    y = input_list["dependents"]
    lamb = input_list["lambda"]

    computation_output = {
        "output": {
            "mean_y_local": np.mean(y),
            "count_local": len(y),
            "computation_phase": 'local_2'
        },
        "cache": {
            "covariates": X,
            "dependents": y,
            "lambda": lamb
        }
    }

    return json.dumps(computation_output)


def local_3(args):
    """Computes the SSE_local, SST_local and varX_matrix_local

    Args:
        args (dictionary): {"input": {
                                "avg_beta_vector": ,
                                "mean_y_global": ,
                                "computation_phase":
                                },
                            "cache": {
                                "covariates": ,
                                "dependents": ,
                                "lambda": ,
                                "dof_local": ,
                                }
                            }

    Returns:
        computation_output (json): {"output": {
                                        "SSE_local": ,
                                        "SST_local": ,
                                        "varX_matrix_local": ,
                                        "computation_phase":
                                        }
                                    }

    Comments:
        After receiving  the mean_y_global, calculate the SSE_local,
        SST_local and varX_matrix_local

    """
    cache_list = args["cache"]
    input_list = args["input"]

    X = cache_list["covariates"]
    y = cache_list["dependents"]
    biased_X = sm.add_constant(X)

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    SSE_local = reg.sum_squared_error(biased_X, y, avg_beta_vector)
    SST_local = np.sum(
        np.square(np.subtract(y, mean_y_global)), dtype=np.float64)
    varX_matrix_local = np.dot(biased_X.T, biased_X)

    computation_output = {
        "output": {
            "SSE_local": SSE_local,
            "SST_local": SST_local,
            "varX_matrix_local": varX_matrix_local.tolist(),
            "computation_phase": "local_3"
        }
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.argv[1])
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_0(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_0' in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_1a' in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_1b' in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_2' in phase_key:
        computation_output = local_3(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation

Example:
    python local.py '{"input":
                        {"covariates": [[2,3],[3,4],[7,8],[7,5],[9,8]],
                         "dependents": [6,7,8,5,6],
                         "lambda": 0
                         },
                     "cache": {}
                     }'
"""
import json
import numpy as np
import sys
import regression as reg
import warnings

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
    X = input_list["covariates"]
    y = input_list["dependents"]
    lamb = input_list["lambda"]

    # Hard-coded this for the time being
    beta_vec_size = 2

    computation_output = {
        "output": {
            "mean_y_local": np.mean(y),
            "count_local": len(y),
            "beta_vec_size": beta_vec_size,
            "computation_phase": "local_0"
        },
        "cache": {
            "covariates": X,
            "dependents": y,
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
    SST_local = np.sum(np.square(np.subtract(y, mean_y_global)))
    varX_matrix_local = np.dot(biased_X.T, biased_X)

    computation_output = {
        "output": {
            "SSE_local": SSE_local,
            "SST_local": SST_local,
            "varX_matrix_local": varX_matrix_local.tolist(),
            "computation_phase": "local_3"
        },
        "cache": {}
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

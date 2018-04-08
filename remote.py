#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for single-shot ridge
regression with decentralized statistic calculation
"""
import json
import sys
import scipy as sp
import numpy as np
import regression as reg


def listRecursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in listRecursive(v, key):
                yield found
        if k == key:
            yield v


def remote_0(args):
    """Need this function for performing multi-shot regression"""
    input_list = args["input"]
    first_user_id = list(input_list.keys())[0]
    beta_vec_size = input_list[first_user_id]["beta_vec_size"]
    input_list = args["input"]
    all_local_stats_dicts = [
        input_list[site]["local_stats_dict"] for site in input_list
    ]

    # Initial setup
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    tol = 0.01
    eta = 10000  # 0.05
    count = 0

    wp = np.zeros(beta_vec_size)
    mt = np.zeros(beta_vec_size)
    vt = np.zeros(beta_vec_size)

    iter_flag = 1

    computation_output = {
        "cache": {
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "tol": tol,
            "eta": eta,
            "count": count,
            "wp": wp.tolist(),
            "mt": mt.tolist(),
            "vt": vt.tolist(),
            "iter_flag": iter_flag,
            "all_local_stats_dicts": all_local_stats_dicts
        },
        "output": {
            "remote_beta": wp.tolist(),
            "computation_phase": "remote_0"
        }
    }

    return json.dumps(computation_output)


def remote_1(args):

    beta1 = args["cache"]["beta1"]
    beta2 = args["cache"]["beta2"]
    eps = args["cache"]["eps"]
    tol = args["cache"]["tol"]
    eta = args["cache"]["eta"]
    count = args["cache"]["count"]
    wp = args["cache"]["wp"]
    mt = args["cache"]["mt"]
    vt = args["cache"]["vt"]
    iter_flag = args["cache"]["iter_flag"]

    count = count + 1

    if not iter_flag:
        computation_output = {
            "cache": {
                "avg_beta_vector": wp
            },
            "output": {
                "avg_beta_vector": wp,
                "computation_phase": "remote_1b"
            }
        }
    else:
        input_list = args["input"]
        if len(input_list) == 1:
            grad_remote = [
                np.array(args["input"][site]["local_grad"])
                for site in input_list
            ]
            grad_remote = grad_remote[0]
        else:
            grad_remote = sum([
                np.array(args["input"][site]["local_grad"])
                for site in input_list
            ])

        mt = beta1 * np.array(mt) + (1 - beta1) * grad_remote
        vt = beta2 * np.array(vt) + (1 - beta2) * (grad_remote**2)

        m = mt / (1 - beta1**count)
        v = vt / (1 - beta2**count)

        wc = wp - eta * m / (np.sqrt(v) + eps)

        if np.linalg.norm(wc - wp) <= tol:
            iter_flag = 0

        wp = wc

        computation_output = {
            "cache": {
                "beta1": beta1,
                "beta2": beta2,
                "eps": eps,
                "tol": tol,
                "eta": eta,
                "count": count,
                "wp": wp.tolist(),
                "mt": mt.tolist(),
                "vt": vt.tolist(),
                "iter_flag": iter_flag,
                "local_stats_dict": args["cache"]["all_local_stats_dicts"]
            },
            "output": {
                "remote_beta": wc.tolist(),
                "computation_phase": "remote_1a"
            }
        }

    return json.dumps(computation_output)


def remote_2(args):
    """Computes the global beta vector, mean_y_global & dof_global

    Args:
        args (dictionary): {"input": {
                                "beta_vector_local": list/array,
                                "mean_y_local": list/array,
                                "count_local": int,
                                "computation_phase": string
                                },
                            "cache": {}
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": list,
                                        "mean_y_global": ,
                                        "computation_phase":
                                        },
                                    "cache": {
                                        "avg_beta_vector": ,
                                        "mean_y_global": ,
                                        "dof_global":
                                        },
                                    }

    """
    input_list = args["input"]

    avg_beta_vector = args["cache"]["avg_beta_vector"]

    mean_y_local = [input_list[site]["mean_y_local"] for site in input_list]
    count_y_local = [input_list[site]["count_local"] for site in input_list]
    mean_y_global = np.average(mean_y_local, weights=count_y_local)

    dof_global = sum(count_y_local) - len(avg_beta_vector)

    computation_output = {
        "output": {
            "avg_beta_vector": avg_beta_vector,
            "mean_y_global": mean_y_global,
            "computation_phase": "remote_2"
        },
        "cache": {
            "avg_beta_vector": avg_beta_vector,
            "mean_y_global": mean_y_global,
            "dof_global": dof_global,
            "all_local_stats_dicts": args["cache"]["all_local_stats_dicts"]
        },
    }

    return json.dumps(computation_output)


def remote_3(args):
    """
    Computes the global model fit statistics, r_2_global, ts_global, ps_global

    Args:
        args (dictionary): {"input": {
                                "SSE_local": ,
                                "SST_local": ,
                                "varX_matrix_local": ,
                                "computation_phase":
                                },
                            "cache":{},
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": ,
                                        "beta_vector_local": ,
                                        "r_2_global": ,
                                        "ts_global": ,
                                        "ps_global": ,
                                        "dof_global":
                                        },
                                    "success":
                                    }
    Comments:
        Generate the local fit statistics
            r^2 : goodness of fit/coefficient of determination
                    Given as 1 - (SSE/SST)
                    where   SSE = Sum Squared of Errors
                            SST = Total Sum of Squares
            t   : t-statistic is the coefficient divided by its standard error.
                    Given as beta/std.err(beta)
            p   : two-tailed p-value (The p-value is the probability of
                  seeing a result as extreme as the one you are
                  getting (a t value as large as yours)
                  in a collection of random data in which
                  the variable had no effect.)

    """
    input_list = args["input"]
    all_local_stats_dicts = args["cache"]["all_local_stats_dicts"]

    cache_list = args["cache"]
    avg_beta_vector = cache_list["avg_beta_vector"]
    dof_global = cache_list["dof_global"]

    SSE_global = np.sum([input_list[site]["SSE_local"] for site in input_list])
    SST_global = np.sum([input_list[site]["SST_local"] for site in input_list])
    varX_matrix_global = sum([
        np.array(input_list[site]["varX_matrix_local"]) for site in input_list
    ])

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / dof_global
    var_covar_beta_global = MSE * sp.linalg.inv(varX_matrix_global)
    se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
    ts_global = avg_beta_vector / se_beta_global
    ps_global = reg.t_to_p(ts_global, dof_global)

    computation_output = {
        "output": {
            "avg_beta_vector": cache_list["avg_beta_vector"],
            "r_2_global": r_squared_global,
            "ts_global": ts_global.tolist(),
            "ps_global": ps_global,
            "dof_global": cache_list["dof_global"]
        },
        "success": True
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.argv[1])
    phase_key = list(listRecursive(parsed_args, "computation_phase"))

    if "local_0" in phase_key:
        computation_output = remote_0(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_2" in phase_key:
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_3" in phase_key:
        computation_output = remote_3(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")

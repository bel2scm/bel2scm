import numpy as np
# import matplotlib.pyplot as plt
# import scipy as sp
import json
import torch
import pyro
from pyro.infer import Importance, EmpiricalMarginal
from torch.distributions.transforms import AffineTransform
import pyro.distributions as dist
from Neuirps_BEL2SCM.Node import Node

PYRO_DISTRIBUTIONS = {

    "Categorical": pyro.distributions.Categorical,
    "Normal": pyro.distributions.Normal,
    "LogNormal": pyro.distributions.LogNormal,
    "Gamma": pyro.distributions.Gamma,
    "Delta": pyro.distributions.Delta,
    "MultivariateNormal": pyro.distributions.MultivariateNormal,
    "BetaBinomial": pyro.distributions.BetaBinomial
}



def json_load(filepath):
    try:
        with open(filepath) as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print("Error: Wrong file or file path.")


def all_parents_visited(node: Node, visited: list) -> bool:
    parents = list(node.parent_info.keys())
    visited_parents = list(set(visited).intersection(parents))
    if len(visited_parents) == len(parents):
        return True
    else:
        return False

# def get_distribution(node_dist: str, dist_parameters: list) -> dist:
#     """
#     Description: This function is to get the distribution for a node based on its type
#     Parameters: the node's type
#     Returns: sampled values for node in tensor format based on its type
#     """
#
#     if node_dist == 'Lognormal':
#         return dist.LogNormal(torch.tensor(dist_parameters[0]), torch.tensor(dist_parameters[1]))
#     if node_dist == 'Process':
#         return dist.Categorical(torch.tensor(dist_parameters))
#     if node_dist == "Gamma":
#         return dist.Gamma(torch.tensor(dist_parameters[0]), torch.tensor(dist_parameters[1]))
#     if node_dist == "Normal":
#         return dist.Normal(torch.tensor(dist_parameters[0]), torch.tensor(dist_parameters[1]))


def check_increase(x: float, threshold: float) -> float:
    """
    Description: Helper function for SCM_model(),
                 to be used with increasing type edges
    Parameters:  Result of parents' equation (x)
    Returns:     1.0 if value is greater than set threshold
                 else 0.0
    """

    if x > threshold:
        return 1.0
    else:
        return 0.0


def check_decrease(x: float, threshold: float) -> float:
    """
    Description: Helper function for SCM_model(),
                 to be used with decreasing type edges
    Parameters:  Result of parents' equation (x)
    Returns:     0.0 if value is greater than set threshold
                 else 1.0
    """

    if x > threshold:
        return 0.0
    else:
        return 1.0


def get_abundance_sample(weights_a: list, p_sample_a: list):
    return sum(x * y for x, y in zip(weights_a, p_sample_a))


def get_transformation_sample(weights_t: list, p_sample_t: list):
    return sum(x * y * y for x, y in zip(weights_t, p_sample_t))


def cat_parents(parent_label: list, parent_name: list, relation: list, w: list, samples: dict, groupby: list):
    increase_parent = []
    decrease_parent = []
    weight_i = []
    weight_d = []
    for i in range(len(parent_label)):
        if relation[i] == 'decreases' or relation[i] == 'directlyDecreases':
            if parent_label[i] in groupby:
                decrease_parent.append(samples[parent_name[i]])
                weight_d.append(w[i])
        else:
            if parent_label[i] in groupby:
                increase_parent.append(samples[parent_name[i]])
                weight_i.append(w[i])
    return increase_parent, decrease_parent, weight_i, weight_d


def get_sample(node, config):
    exog_name = node.name + "_N"
    pyro_dist_exog = config.exogenous_distribution_info[0]
    pyro_params = config.exogenous_distribution_info[1]
    exog = pyro.sample(exog_name, pyro_dist_exog(pyro_params[0], pyro_params[1]))
    pyro_dist_node = config.node_label_distribution_info[node.node_label][0]
    pyro_params_node = config.node_label_distribution_info[node.node_label][1]

    # print(pyro_params_node)
    # if node.root == True:
    #     pass
        # return pyro.sample(node.name, pyro.Delta())




# def get_sample(child_name: str,
#                child_label: str,
#                parent_label: list,
#                threshold: float,
#                normal: dist,
#                gamma: dist,
#                lognormal: dist,
#                increase_process: list,
#                decrease_process: list,
#                increase_abundance: list,
#                decrease_abundance: list,
#                weights_ai: list,
#                weights_ad: list,
#                increase_transformation,
#                decrease_transformation,
#                weights_ti: list,
#                weights_td: list,
#                ) -> (float, str):
#
#     child_increase_N = get_abundance_sample(weights_ai, increase_abundance) + \
#                        get_transformation_sample(weights_ti, increase_transformation)
#
#     child_decrease_N = get_abundance_sample(weights_ad, decrease_abundance) + \
#                        get_transformation_sample(weights_td, decrease_transformation)
#
#     if child_label == 'transformation':
#         child_name_noise = child_name + "_N"
#         child_noise = pyro.sample(child_name_noise, gamma)
#         child_N = child_increase_N - child_decrease_N + child_noise
#
#     elif child_label == 'Abundance':
#         child_name_noise = child_name + "_N"
#         child_noise = pyro.sample(child_name_noise, lognormal)
#         child_N = child_increase_N - child_decrease_N + child_noise
#
#     else:
#         child_name_noise = child_name + "_N"
#         child_noise = pyro.sample(child_name_noise, normal)
#         child_check = check_increase(child_increase_N + child_noise + sum(increase_process),
#                                      (len(parent_label)) * threshold) + check_decrease(child_decrease_N + child_noise +
#                                                                                        sum(decrease_process),
#                                                                                        (len(parent_label)) * threshold)
#         if len(increase_process) == 0 and len(decrease_process) > 0 and child_check == 1.0:
#             child_N = torch.tensor(1.0)
#         elif len(decrease_process) == 0 and len(increase_process) > 0 and child_check == 1.0:
#             child_N = torch.tensor(1.0)
#         elif child_check == 2.0:
#             child_N = torch.tensor(1.0)
#         else:
#             child_N = torch.tensor(0.)
#
#     return child_N, child_name_noise

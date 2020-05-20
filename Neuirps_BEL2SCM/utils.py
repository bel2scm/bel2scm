# import matplotlib.pyplot as plt
# import scipy as sp
import json

import pyro
import pyro.distributions as dist
import torch

from Neuirps_BEL2SCM.node import Node
from Neuirps_BEL2SCM.parent_interaction_types import ParentInteractionTypes

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
    return len(visited_parents) == len(parents)


def get_distribution(distribution_info: tuple) -> dist:
    """
    Description: This function is to get the distribution for a node based on its type
    Parameters: the node's type
    Returns: sampled values for node in tensor format based on its type
    """
    pyro_dist = distribution_info[0]
    pyro_params = distribution_info[1]
    # if pyro_dist.__name__ == 'Categorical':
    if pyro_dist is dist.Categorical:
        return pyro_dist(torch.tensor(pyro_params))
    return pyro_dist(pyro_params[0], pyro_params[1])


def check_increase(x: float, threshold: float) -> float:
    """
    Description: Helper function for SCM_model(),
                 to be used with increasing type edges
    Parameters:  Result of parents' equation (x)
    Returns:     1.0 if value is greater than set threshold
                 else 0.0
    """
    return 0.0 if x > threshold else 1.0


def check_decrease(x: float, threshold: float) -> float:
    """
    Description: Helper function for SCM_model(),
                 to be used with decreasing type edges
    Parameters:  Result of parents' equation (x)
    Returns:     0.0 if value is greater than set threshold
                 else 1.0
    """
    return 0.0 if x > threshold else 1.0


def get_abundance_sample(weights_a: list, p_sample_a: list):
    return sum(x * y for x, y in zip(weights_a, p_sample_a))


def get_transformation_sample(weights_t: list, p_sample_t: list):
    return sum(x * y * y for x, y in zip(weights_t, p_sample_t))


def get_parent_samples(node: Node, sample: dict) -> dict:
    """
    returns subset of sample
    """
    parent_sample_dict = dict()
    for parent_name in list(node.parent_info.keys()):
        parent_sample_dict[parent_name] = sample[parent_name]
    return parent_sample_dict


def sample_with_and_interaction(node, config, parent_samples):
    # # labels of the parents
    p_labels = list()
    p_relations = list()
    weight = config.prior_weight
    for key in parent_samples.keys():
        p_labels.append(node.parent_info[key]["label"])
        p_relations.append(node.parent_info[key]["relation"])

    # weights of the parents
    # relations with the parents
    # If we don't have a compound case, so if parent_labels = process only -> we need boolean condition
    # if parent_labels = abundance -> call get abundance sample
    pass
    # parent_labels = node.parent_info


# def cat_parents(parent_info, config, parent_samples):
#     categorized_parents = dict()
#     increase_process = []
#     decrease_process = []
#     weight_i = []
#     weight_d = []
#     for key in parent_samples.keys():
#
#
#     for i in range(len(parent_label)):
#         if relation[i] == 'decreases' or relation[i] == 'directlyDecreases':
#             if parent_label[i] in groupby:
#                 decrease_parent.append(samples[parent_name[i]])
#                 weight_d.append(w[i])
#         else:
#             if parent_label[i] in groupby:
#                 increase_parent.append(samples[parent_name[i]])
#                 weight_i.append(w[i])
#     return increase_parent, decrease_parent, weight_i, weight_d
def get_sample_for_roots(node: Node, config: dict):
    exog_name = node.name + "_N"
    exog = pyro.sample(exog_name, get_distribution(config.exogenous_distribution_info))
    node_distribution_info = config.node_label_distribution_info[node.node_label]
    node_dist = get_distribution(node_distribution_info)
    endog_name = node.name + "_endog"
    node_sample = pyro.sample(endog_name, node_dist)
    return pyro.sample(node.name, pyro.distributions.Normal((exog + node_sample), 1.0))


def get_sample_for_non_roots(node: Node, config: dict, parent_samples=[]):
    exog_name = node.name + "_N"
    exog = pyro.sample(exog_name, get_distribution(config.exogenous_distribution_info))

    if config.parent_interaction_type == ParentInteractionTypes.AND.value:
        return sample_with_and_interaction(node, config, parent_samples)
    else:
        raise Exception("Invalid parent interaction type")

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

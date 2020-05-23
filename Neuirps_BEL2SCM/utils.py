# import matplotlib.pyplot as plt
# import scipy as sp
import json

import pyro
import pyro.distributions as dist
import torch
import numpy as np
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


def generate_process_condition(parent_info, parent_samples):
    active_list = list()
    for parent_name in parent_samples.keys():
        if parent_info[parent_name]["label"] in ["process", "activity", "reaction", "pathology"]:
            if parent_info[parent_name]["relation"] in ["increases", "directlyIncreases"]:
                # increases type process parents need to be active
                if parent_samples[parent_name] == 1.0:
                    active_list.append(1)
                else:
                    active_list.append(0)
            else:
                # decreases type process parents need to be inactive
                if parent_samples[parent_name] == 0.0:
                    active_list.append(1)
                else:
                    active_list.append(0)
    if len(active_list) == sum(active_list):
        return True
    return False


def get_sign_weight_parent_vector(parent_info, weight_dict, parent_samples):
    sign_vector = []
    weight_vector = []
    parent_samples_vector = []
    for parent_name in parent_info.keys():
        if parent_info[parent_name]["label"] == "abundance":
            if parent_info[parent_name]["relation"] in ["directlyIncreases", "increases"]:
                sign_vector.append(1)
            else:
                sign_vector.append(-1)
            weight_vector.append(weight_dict[parent_name])
            parent_samples_vector.append(parent_samples[parent_name])
        if parent_info[parent_name]["label"] == "transformation":
            if parent_info[parent_name]["relation"] in ["directlyIncreases", "increases"]:
                sign_vector.append(1)
            else:
                sign_vector.append(-1)
            weight_vector.append(weight_dict[parent_name])
            parent_value = parent_samples[parent_name] + parent_samples[parent_name]**2
            parent_samples_vector.append(parent_value)

    return np.array(sign_vector), np.array(weight_vector), np.array(parent_samples_vector)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def get_parent_combination(node, parent_info, parent_samples, weight_dict, noise):
    sign, weights, parent_samples_vector = get_sign_weight_parent_vector(parent_info, weight_dict, parent_samples)
    signed_weights = sign * weights
    parent_combination = signed_weights * parent_samples_vector.T + noise
    return parent_combination


def get_sample_for_process(node, parent_samples, weight_dict, exog, threshold):
    process_check = generate_process_condition(node.parent_info, parent_samples)
    if process_check:
        parent_combo = get_parent_combination(node, node.parent_info, parent_samples, weight_dict, exog)
        c = sigmoid(parent_combo)
        if c > threshold:
            return 1.0
        else:
            return 0.0
    return 0.0


def get_sample_for_abundance(node, parent_samples, weight_dict, exog):
    process_check = generate_process_condition(node.parent_info, parent_samples)
    child_distribution_tuple = ()
    child_distribution_tuple[0] = node.node_label
    if process_check:
        c_mean = get_parent_combination(node, node.parent_info, parent_samples, weight_dict, exog)
        child_distribution_tuple[1] = [c_mean, 1.0]
    child_distribution_tuple[1] = [0.0, 1.0]

    return get_distribution(child_distribution_tuple)


def get_sample_for_transformation(node, parent_samples, weight_dict, exog):
    process_check = generate_process_condition(node.parent_info, parent_samples)
    child_distribution_tuple = ()
    child_distribution_tuple[0] = node.node_label
    if process_check:
        c_mean = get_parent_combination(node, node.parent_info, parent_samples, weight_dict, exog)
        child_distribution_tuple[1] = [c_mean, 1.0]
    child_distribution_tuple[1] = [0.0, 1.0]

    return get_distribution(child_distribution_tuple)


def sample_with_and_interaction(node, config, parent_samples, exog):
    # List(Tuple<String Label, String Relation>) - Ex.: <'abundance','increases'>
    weight_dict = dict()
    threshold = config.prior_threshold
    for parent_name in parent_samples.keys():
        # weights of the parents
        weight = pyro.sample(parent_name + "_weight", dist.Normal(config.prior_weight, 1.0))
        weight_dict[parent_name] = weight
        if node.node_label in ["process", "activity", "reaction", "pathology"]:
            return get_sample_for_process(node, parent_samples, weight_dict, exog, threshold)
        elif node.node_label == "abundance":
            return get_sample_for_abundance(node, parent_samples, weight_dict, exog)
        elif node.node_label == "transformation":
            return get_sample_for_transformation(node, parent_samples, weight_dict, exog)
        else:
            raise Exception("invalid node type")


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
        return sample_with_and_interaction(node, config, parent_samples, exog)
    else:
        raise Exception("Invalid parent interaction type")

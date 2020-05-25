# import matplotlib.pyplot as plt
# import scipy as sp
import collections

import pyro
import pyro.distributions as dist
import json
import statistics
from statistics import mean
from statistics import stdev
import torch
import torch.nn.functional as F
import numpy as np
from Neuirps_BEL2SCM.node import Node
from Neuirps_BEL2SCM.scm import *
from Neuirps_BEL2SCM.parent_interaction_types import ParentInteractionTypes
from Neuirps_BEL2SCM.constants import PYRO_DISTRIBUTIONS, VARIABLE_TYPE
import pickle


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



def get_abundance_sample(weights_a: list, p_sample_a: list):
    return sum(x * y for x, y in zip(weights_a, p_sample_a))


def get_transformation_sample(weights_t: list, p_sample_t: list):
    return sum(x * y * y for x, y in zip(weights_t, p_sample_t))


def get_parent_samples(node: Node, sample: dict) -> dict:
    """
    returns subset of sample
    """
    parent_sample_dict = collections.OrderedDict()
    for parent_name in list(node.parent_info.keys()):
        parent_sample_dict[parent_name] = sample[parent_name]
    return parent_sample_dict



def generate_process_condition(parent_info, parent_samples):
    active_list = list()
    for parent_name in parent_samples.keys():
        if parent_info[parent_name]["label"] in VARIABLE_TYPE["Categorical"]:
            if parent_info[parent_name]["relation"] in ["increases", "directlyIncreases"]:
                # increases type process parents need to be active
                if parent_samples[parent_name] > 0.5:
                    active_list.append(1)
                else:
                    active_list.append(0)
            else:
                # decreases type process parents need to be inactive
                if parent_samples[parent_name] <= 0.5:
                    active_list.append(1)
                else:
                    active_list.append(0)
    if len(active_list) == sum(active_list):
        return True
    return False

# def get_sign_weight_parent_vector(parent_info, weight_dict, parent_samples):
#     sign_vector = []
#     weight_vector = []
#     parent_samples_vector = []
#     for parent_name in parent_info.keys():
#         if parent_info[parent_name]["label"] == "abundance":
#             if parent_info[parent_name]["relation"] in ["directlyIncreases", "increases"]:
#                 sign_vector.append(1)
#             else:
#                 sign_vector.append(-1)
#             weight_vector.append(weight_dict[parent_name])
#             parent_samples_vector.append(parent_samples[parent_name])
#         if parent_info[parent_name]["label"] == "transformation":
#             if parent_info[parent_name]["relation"] in ["directlyIncreases", "increases"]:
#                 sign_vector.append(1)
#             else:
#                 sign_vector.append(-1)
#             weight_vector.append(weight_dict[parent_name])
#             parent_value = parent_samples[parent_name] + parent_samples[parent_name]**2
#             parent_samples_vector.append(parent_value)
#
#     return np.array(sign_vector), np.array(weight_vector), np.array(parent_samples_vector)


def get_sample_for_binary_node(node, parent_samples, exog, node_distribution, deterministic_prediction):
    c = F.sigmoid(deterministic_prediction + exog)
    return pyro.sample(node.name, node_distribution(c))


def get_sample_for_continuous_node(node, parent_samples, exog, node_distribution, deterministic_prediction):
    c_mean = torch.squeeze(deterministic_prediction) + exog
    # [TODO]: Get the std from TrainedModel and that should be the std below.
    return pyro.sample(node.name, node_distribution(c_mean, 1.0))


def sample_with_and_interaction(node, config, parent_samples, exog, deterministic_prediction):
    # List(Tuple<String Label, String Relation>) - Ex.: <'abundance','increases'>
    threshold = config.prior_threshold
    node_distribution = config.node_label_distribution_info[node.node_label]

    if node.node_label in VARIABLE_TYPE["Categorical"]:
        return get_sample_for_binary_node(node, parent_samples, exog, node_distribution, deterministic_prediction)
    elif node.node_label in VARIABLE_TYPE["Continuous"]:
        return get_sample_for_continuous_node(node, parent_samples, exog, node_distribution, deterministic_prediction)
    else:
        raise Exception("invalid node type")

def get_parameters_for_root_nodes(node_data, node_distribution_type):
    parameters_list = list()
    if node_distribution_type == "Categorical":
        mean_of_1 = statistics.mean(node_data)
        parameters_list = [1-mean_of_1, mean_of_1]
    elif node_distribution_type in ["LogNormal", "Normal"]:
        mu = statistics.mean(node_data)
        sigma = stdev(node_data)
        parameters_list = [mu, sigma]
    elif node_distribution_type == "Gamma":
        mu = statistics.mean(node_data)
        sigma = stdev(node_data)
        alpha = (mu/sigma)**2
        beta = (sigma**2) / mu
        parameters_list = [alpha, beta]

    return parameters_list

def get_sample_for_roots(node: Node, config, node_distribution_parameters):
    # exog_name = node.name + "_N"
    # exog = pyro.sample(exog_name, get_distribution(config.exogenous_distribution_info))
    node_dist_info = config.node_label_distribution_info[node.node_label]
    node_distribution_type = node_dist_info[0]
    node_distribution_info = (node_distribution_type, node_distribution_parameters)
    node_dist = get_distribution(node_distribution_info)
    endog_name = node.name + "_endog"
    node_sample = pyro.sample(endog_name, node_dist)
    return pyro.sample(node.name, pyro.distributions.Normal((node_sample), 1.0))


def get_sample_for_non_roots(node: Node, config, parent_samples: dict, deterministic_prediction):
    exog_name = node.name + "_N"
    exog = pyro.sample(exog_name, get_distribution(config.exogenous_distribution_info))

    if config.parent_interaction_type == ParentInteractionTypes.AND.value:
        return sample_with_and_interaction(node, config, parent_samples, exog, deterministic_prediction)
    else:
        raise Exception("Invalid parent interaction type")


def check_parent_order(parent_info_from_node, parent_list_from_data) -> bool:
    return parent_info_from_node == parent_list_from_data

def get_parent_tensor(parent_sample_dict, continuous_parent_names):
    continuous_sample_list = []
    for parent_name in continuous_parent_names:
        try:
            continuous_sample_list.append(parent_sample_dict[parent_name])
        except:
            raise Exception("Something went wrong while get_parent_tensor")

    # converting 1-d tensor to 2-d
    output_tensor = torch.FloatTensor(continuous_sample_list).view(len(continuous_sample_list), 1)

    return output_tensor

def save_scm_object(pkl_file_path, scm):
    pickle_out = open(pkl_file_path, "wb")
    pickle.dump(scm, pickle_out)
    pickle_out.close()

def load_scm_object(pkl_file_path):
    pickle_in = open(pkl_file_path, "rb")
    return pickle.load(pickle_in)
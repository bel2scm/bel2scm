# import matplotlib.pyplot as plt
# import scipy as sp
import collections
import pyro
import pyro.distributions as dist
import json
import torch
import torch.nn.functional as F
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
    else:
        return pyro_dist(pyro_params[0], pyro_params[1])

def get_parent_samples(node: Node, sample: dict) -> dict:
    """
    returns subset of sample dict. It contains parent samples for the current node.
    """
    parent_sample_dict = collections.OrderedDict()
    for parent_name in list(node.parent_info.keys()):
        parent_sample_dict[parent_name] = sample[parent_name]
    return parent_sample_dict


def get_sample_for_binary_node(node, exog, node_distribution, deterministic_prediction):
    """

    Args:
        node: Node()
        exog: pyro.distribution()
        node_distribution: pyro.distribution()
        deterministic_prediction: torch.tensor()

    Returns: pyro.sample("current_node_name")

    """
    c = F.sigmoid(deterministic_prediction + exog)
    return pyro.sample(node.name, node_distribution(c))


def get_sample_for_continuous_node(node, exog, node_distribution, deterministic_prediction):
    """

    Args:
        node: Node()
        exog: pyro.distribution()
        node_distribution: pyro.distribution()
        deterministic_prediction: torch.tensor()

    Returns: pyro.sample("current_node_name")

    """
    c_mean = torch.squeeze(deterministic_prediction) + exog
    # [TODO]: Get the std from TrainedModel and that should be the std below.
    return pyro.sample(node.name, node_distribution(c_mean, 1.0))


def sample_with_and_interaction(node, config, exog, deterministic_prediction):
    """
    This is the default method to generate SCM for now.
    This method executes and interaction, where every parent is required to predict child.

    Returns: pyro.sample()

    """
    # List(Tuple<String Label, String Relation>) - Ex.: <'abundance','increases'>
    node_distribution = config.node_label_distribution_info[node.node_label]

    if node.node_label in VARIABLE_TYPE["Categorical"]:
        return get_sample_for_binary_node(node, exog, node_distribution, deterministic_prediction)
    elif node.node_label in VARIABLE_TYPE["Continuous"]:
        return get_sample_for_continuous_node(node, exog, node_distribution, deterministic_prediction)
    else:
        raise Exception("invalid node type")


def get_sample_for_non_roots(node: Node, config, deterministic_prediction):
    """
    This function generates pyro.sample for each non-root node in SCM.
    Args:
        node: Node()
        config: Config()
        deterministic_prediction: tensor()

    Returns: pyro.sample()

    """
    exog_name = node.name + "_N"
    exog = pyro.sample(exog_name, get_distribution(config.exogenous_distribution_info))

    if config.parent_interaction_type == ParentInteractionTypes.AND.value:
        return sample_with_and_interaction(node, config, exog, deterministic_prediction)
    else:
        raise Exception("Invalid parent interaction type")


def get_parent_tensor(parent_sample_dict, continuous_parent_names):
    """

    Args:
        parent_sample_dict: contains dictionary of pyro.sample()s for all parents
        continuous_parent_names: parent name in order

    Returns: tensor with all the parent pyro.sample() values. This will get used to predict child value
        from train_network.

    """
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
    """
    Serialize SCM object to a pickle file
    Args:
        pkl_file_path: str
        scm: SCM()
    """
    pickle_out = open(pkl_file_path, "wb")
    pickle.dump(scm, pickle_out)
    pickle_out.close()


def load_scm_object(pkl_file_path):
    """
    Deserializes SCM object from pickle file and return
    Args:
        pkl_file_path: str

    Returns: SCM()

    """
    pickle_in = open(pkl_file_path, "rb")
    return pickle.load(pickle_in)
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


def get_sample_for_binary_node(node, node_distribution, deterministic_prediction):
    """

    Args:
        node: Node()
        exog: pyro.distribution()
        node_distribution: pyro.distribution()
        deterministic_prediction: torch.tensor()

    Returns: pyro.sample("current_node_name")

    """
    c = deterministic_prediction
    return pyro.sample(node.name, node_distribution(c))


def get_sample_for_continuous_node(node, node_distribution, deterministic_prediction):
    """

    Args:
        node: Node()
        exog: pyro.distribution()
        node_distribution: pyro.distribution()
        deterministic_prediction: torch.tensor()

    Returns: pyro.sample("current_node_name")

    """
    c_mean = torch.squeeze(deterministic_prediction)
    # [TODO]: Get the std from TrainedModel and that should be the std below.
    return pyro.sample(node.name, node_distribution(c_mean, 1.0))


def sample_with_and_interaction(node, config, deterministic_prediction):
    """
    This is the default method to generate SCM for now.
    This method executes and interaction, where every parent is required to predict child.

    Returns: pyro.sample()

    """
    # List(Tuple<String Label, String Relation>) - Ex.: <'abundance','increases'>
    node_distribution = config.node_label_distribution_info[node.node_label]

    if node.node_label in VARIABLE_TYPE["Categorical"]:
        return get_sample_for_binary_node(node, node_distribution, deterministic_prediction)
    elif node.node_label in VARIABLE_TYPE["Continuous"]:
        return get_sample_for_continuous_node(node, node_distribution, deterministic_prediction)
    else:
        raise Exception("invalid node type")


def get_sample_for_non_roots(node: Node, config, deterministic_prediction):
    """
    This function generates pyro.sample for each non-root node in SCM.
    Args:
        node: Node()
        config: Config()
        prediction: tensor()

    Returns: pyro.sample()

    """

    if config.parent_interaction_type == ParentInteractionTypes.AND.value:
        return sample_with_and_interaction(node, config, deterministic_prediction)
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


def get_exogenous_distribution(config, exogenous_std_dict) -> dict():
    """
    Args:
        config: config to get the user provided noise distribution
        exogenous_std_dict: dictionary containing std for the noise distribution
    Returns:
        dictionary of exogenous samples

    """
    exogenous_dict = {}
    noise_distribution = config.exogenous_distribution_info.keys()[0]
    for node_name in exogenous_std_dict.keys():
        exog_name = node_name + "_N"
        exog_distribution_info = (noise_distribution, [torch.tensor(0.), exogenous_std_dict[node_name]])
        exogenous_dict[exog_name] = get_distribution(exog_distribution_info)
    return exogenous_dict


def get_child_name_list(children_info, node_list):
    """

    """
    child_name_list = [value["name"] for (key, value) in children_info.items()]
    node_list.extend(child_name_list)
    return node_list

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


def create_bins(lower_bound, width, quantity):
    """ create_bins returns an equal-width (distance) partitioning.
        It returns an ascending list of tuples, representing the intervals.
    """

    bins = []
    for low in range(lower_bound,
                     lower_bound + quantity * width + 1, width):
        bins.append((low, low + width))
    return bins


# def sample_gumbel(shape, eps=1e-20):
#     unif = torch.rand(*shape).to(device)
#     g = -torch.log(-torch.log(unif + eps))
#     return g
#
#
# def sample_gumbel_softmax(logits, temperature):
#     """
#         Input:
#         logits: Tensor of log probs, shape = BS x k
#         temperature = scalar
#
#         Output: Tensor of values sampled from Gumbel softmax.
#                 These will tend towards a one-hot representation in the limit of temp -> 0
#                 shape = BS x k
#     """
#     g = sample_gumbel(logits.shape)
#     h = (g + logits) / temperature
#     h_max = h.max(dim=-1, keepdim=True)[0]
#     h = h - h_max
#     cache = torch.exp(h)
#     y = cache / cache.sum(dim=-1, keepdim=True)
#     return y
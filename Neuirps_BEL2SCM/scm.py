from itertools import chain

from Neuirps_BEL2SCM.bel_graph import BelGraph
from Neuirps_BEL2SCM.utils import *
from Neuirps_BEL2SCM.utils import all_parents_visited
from Neuirps_BEL2SCM.utils import get_sample_for_non_roots
from Neuirps_BEL2SCM.utils import json_load
import pandas as pd
PYRO_DISTRIBUTIONS = {

    "Categorical": pyro.distributions.Categorical,
    "Normal": pyro.distributions.Normal,
    "LogNormal": pyro.distributions.LogNormal,
    "Gamma": pyro.distributions.Gamma,
    "Delta": pyro.distributions.Delta,
    "MultivariateNormal": pyro.distributions.MultivariateNormal,
    "BetaBinomial": pyro.distributions.BetaBinomial
}

class SCM:
    '''
    4. Add functions as described below
    '''
    def __init__(self, bel_file_path, config_file_path, data_file_path):

        #  get nodes from bel file.
        self.graph = BelGraph("nanopub_file", bel_file_path).construct_graph_from_nanopub_file()
        # Prepare data for all the nodes.
        # self.graph.node_data contains (feature_data, target_data) at each node.
        BelGraph("nanopub_file", bel_file_path).prepare_and_assign_data(data_file_path)
        # Learn
        self.data = pd.read_csv(data_file_path)
        # 2. set parameters from config file.
        self.config = Config(config_file_path)
        self.roots = BelGraph("nanopub_file", bel_file_path).get_nodes_with_no_parents()
        # 3. Build model
        self._build_model()

    def _build_model(self):
        graph = self.graph
        config = self.config
        roots = self.roots
        sample = dict()
        node_queue_for_bfs_traversal = list()
        visited_nodes = list()
        data = self.data
        # add root nodes for BFS traversal
        for node_name in roots.keys():
            node_distribution_type = config.node_label_distribution_info[roots[node_name].node_label][0]
            sample[node_name] = get_sample_for_roots(roots[node_name], config,
                                                     get_parameters_for_root_nodes(data[node_name],
                                                                                   node_distribution_type))
            # add child nodes to queue
            node_queue_for_bfs_traversal.append(roots[node_name].children_info.keys())
            # mark current node as visited
            visited_nodes.append(node_name)
        node_queue_for_bfs_traversal = list(chain.from_iterable(node_queue_for_bfs_traversal))

        if len(node_queue_for_bfs_traversal) == 0:
            raise Exception("No edges found.")

        while len(node_queue_for_bfs_traversal) > 0:
            # get current node from queue
            current_node_name = node_queue_for_bfs_traversal[0]
            # if current node is not visited and all of its parents are visited
            if current_node_name not in visited_nodes and \
                    all_parents_visited(graph[current_node_name], visited_nodes):
                parent_sample_dict = get_parent_samples(graph[current_node_name], sample)
                sample[current_node_name] = get_sample_for_non_roots(graph[current_node_name], config,
                                                                     parent_sample_dict)
                child = list(graph[current_node_name].children_info.keys())
                node_queue_for_bfs_traversal.extend(child)
                visited_nodes.append(current_node_name)
                node_queue_for_bfs_traversal.pop(0)
            else:
                node_queue_for_bfs_traversal.pop(0)



    # [Todo]
    def counterfactual_inference(self):
        return NotImplementedError

    # [Todo]
    def condition(self, condition_data):
        '''
        It conditions self.model with condition data
        Returns: Conditioned pyro model
        '''
        # conditioned_model = pyro.condition(self.model, value)
        return NotImplementedError

    # [Todo]
    def intervention(self, intervention_data):
        '''
        It intervenes self.model with intervention data
        Returns: intervention pyro model
        '''
        # intervention_model = pyro.do(self.model, value)
        return NotImplementedError

    # [Todo]
    def infer(self, target_variables, infer_method):
        '''
        this performs inference for target_variables
        Args:
            target_variables:
            infer_method:

        Returns: Not sure now

        '''
        return NotImplementedError

    # [Todo]
    def fit_parameters(self, data):
        '''
        Fits parameters of self.model
        Args:
            data:

        Returns: Nothing

        '''


class Config:
    '''
    Loads config file.
    '''
    def __init__(self, config_file_path):
        self.config_dict = json_load(config_file_path)
        #self._check_config()
        self._set_config_parameters()

    def _set_config_parameters(self):

        config = self.config_dict
        self.prior_weight = config["prior_weight"]
        self.prior_threshold = config["prior_threshold"]
        self.node_label_distribution_info = self._get_pyro_dist_from_text(config["node_label_distribution_info"])
        self.exogenous_distribution_info = self._get_exogenous_dist_from_text(config["exogenous_distribution_info"])
        self.parent_interaction_type = config["relation_type"]

    def _get_pyro_dist_from_text(self, node_label_distribution_info):
        '''
        Converts string distribution names in config to pyro distributions. The names should be exact match.
        Args:
            node_label_distribution_info:

        Returns:

        '''
        label_pyro_dist_dict = {}
        for label, dist_with_params in node_label_distribution_info.items():
            # Get distribution name and its parameters
            dist_str = list(dist_with_params.keys())[0]
            dist_params = dist_with_params[dist_str]

            # Convert distribution name to pyro distribution
            if dist_str in PYRO_DISTRIBUTIONS:
                label_pyro_dist_dict[label] = (PYRO_DISTRIBUTIONS[dist_str], dist_params)
            else:
                raise Exception("Distribution not supported.")
        return label_pyro_dist_dict

    def _check_config(self):
        # [Todo] - Check the syntax of config just like we did in the testcase.
        pass

    def _get_exogenous_dist_from_text(self, exogenous_distribution_info):

        # Get distribution name and its parameters
        dist_str = list(exogenous_distribution_info.keys())[0]
        dist_params = exogenous_distribution_info[dist_str]

        # Convert distribution name to pyro distribution
        if dist_str in PYRO_DISTRIBUTIONS:
            return (PYRO_DISTRIBUTIONS[dist_str], dist_params)
        else:
            raise Exception("Distribution not supported.")

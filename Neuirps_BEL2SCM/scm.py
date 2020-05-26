from itertools import chain

from Neuirps_BEL2SCM.bel_graph import BelGraph
from Neuirps_BEL2SCM.parameter_estimation import ParameterEstimation
from Neuirps_BEL2SCM.utils import get_sample_for_non_roots, get_parent_tensor, all_parents_visited, json_load, get_parent_samples
from Neuirps_BEL2SCM.constants import PYRO_DISTRIBUTIONS
import pandas as pd
import pyro
import torch



class SCM:
    '''
    4. Add functions as described below
    '''
    def __init__(self, bel_file_path, config_file_path, data_file_path):

        # 1. set parameters from config file.
        self.config = Config(config_file_path)

        # 2. get nodes from bel file.
        self.belgraph = BelGraph("nanopub_file", bel_file_path, data_file_path)
        self.belgraph.construct_graph_from_nanopub_file()
        self.belgraph.prepare_and_assign_data()
        self.graph = self.belgraph.nodes

        # Learn parameters
        parameter_estimation = ParameterEstimation(self.belgraph, self.config)
        parameter_estimation.get_distribution_for_roots_from_data()
        parameter_estimation.get_model_for_each_non_root_node()

        self.root_distributions = parameter_estimation.root_distributions
        self.trained_networks = parameter_estimation.trained_networks


        self.roots = self.belgraph.get_nodes_with_no_parents()
        # 3. Build model
        self.model()

    def model(self):

        # Getting class variables.
        graph = self.graph
        config = self.config
        roots = self.roots
        root_distributions = self.root_distributions
        trained_networks = self.trained_networks

        # Dictionary of pyro samples coming from pyro.sample() for each node.
        sample = dict()

        # node queue for bfs traversal
        node_queue_for_bfs_traversal = list()

        # visited nodes while traversal
        visited_nodes = list()

        # process all root nodes first for BFS traversal
        for node_name in roots.keys():
            node_distribution_type = config.node_label_distribution_info[roots[node_name].node_label]
            sample[node_name] = pyro.sample(node_name, root_distributions[node_name])
            # add child nodes to queue
            child_name_list = [value["name"] for (key, value) in roots[node_name].children_info.items()]
            node_queue_for_bfs_traversal.extend(child_name_list)
            # mark current node as visited
            visited_nodes.append(node_name)

        # if no nodes available for traversal!
        if len(node_queue_for_bfs_traversal) == 0:
            raise Exception("No edges found.")

        while len(node_queue_for_bfs_traversal) > 0:

            # get first node from queue
            current_node_name = node_queue_for_bfs_traversal[0]

            # if current node is not visited AND all of its parents are visited
            is_current_node_not_visited = current_node_name not in visited_nodes
            are_all_parents_visited_for_current_node = all_parents_visited(graph[current_node_name], visited_nodes)

            if is_current_node_not_visited and are_all_parents_visited_for_current_node:

                parent_sample_dict = get_parent_samples(graph[current_node_name], sample)
                parent_names = self.belgraph.parent_name_list_for_nodes
                deterministic_prediction = None

                if len(parent_names[current_node_name]) > 0:
                    parent_tensor = get_parent_tensor(parent_sample_dict, parent_names[current_node_name])
                    deterministic_prediction = self._get_prediction(trained_networks[current_node_name], parent_tensor)

                sample[current_node_name] = get_sample_for_non_roots(graph[current_node_name], config, deterministic_prediction)

                # [TODO] Move below two lines to a function
                child_name_list = [value["name"] for (key, value) in graph[current_node_name].children_info.items()]
                node_queue_for_bfs_traversal.extend(child_name_list)
                visited_nodes.append(current_node_name)
                node_queue_for_bfs_traversal.pop(0)

            #
            else:
                # if all parents are not visited for a node, we assume that current node will be in the child info of
                # the unvisited parents
                node_queue_for_bfs_traversal.pop(0)

        return sample

    # [Todo]
    def counterfactual_inference(self):
        return NotImplementedError

    # [Todo]
    def model_condition_data(self, cond_dict):
        '''
        It conditions self.model with condition data
        Returns: Conditioned pyro model
        '''
        data_in = {}
        for item in cond_dict:
            data_in[item] = cond_dict[item]

        cond_model = pyro.condition(self.model, data=data_in)

        return cond_model()
    def model_do_sample(self,do_dict):
        '''
        Returns: Do pyro model
        '''
        data_in = {}
        for item in do_dict:
            data_in[item] = do_dict[item]

        do_model = pyro.do(self.model, data=data_in)

        return do_model()

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

    def _get_prediction(self, trained_network, parent_tensor):
        try:
            return trained_network.predict(parent_tensor)
        except:
            raise Exception("Error getting deterministic prediction.")


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
        for label, distribution_name in node_label_distribution_info.items():

            # Convert distribution name to pyro distribution
            if distribution_name in PYRO_DISTRIBUTIONS:
                label_pyro_dist_dict[label] = PYRO_DISTRIBUTIONS[distribution_name]
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

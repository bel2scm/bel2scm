import statistics
from collections import defaultdict

from pyro.infer import SVI, Trace_ELBO, Importance, EmpiricalMarginal
import torch.distributions.constraints as constraints
from torch.optim import SGD

from pyro import poutine
from pyro.infer import TraceEnum_ELBO, SVI, Trace_ELBO, Importance, EmpiricalMarginal
from pyro.infer.autoguide import AutoDelta
import torch.distributions.constraints as constraints
from torch.optim import SGD

from Neuirps_BEL2SCM.bel_graph import BelGraph
from Neuirps_BEL2SCM.parameter_estimation import ParameterEstimation
from Neuirps_BEL2SCM.utils import get_sample_for_non_roots, get_parent_tensor, all_parents_visited, json_load, \
    get_parent_samples, get_distribution, get_exogenous_samples
from Neuirps_BEL2SCM.constants import PYRO_DISTRIBUTIONS
import torch
import pyro


class SCM:
    '''
    4. Add functions as described below
    '''

    def __init__(self, bel_file_path, config_file_path, data_file_path):

        # 1. set parameters from config file.
        self.config = Config(config_file_path)

        # 2. get nodes from bel file.
        self.belgraph = BelGraph("nanopub_file", bel_file_path, data_file_path)
        self.belgraph.parse_input_to_construct_graph()
        if self.belgraph.is_cyclic():
            raise Exception("Graph contains cycles!")

        self.belgraph.prepare_and_assign_data()
        self.graph = self.belgraph.nodes

        # Learn parameters
        parameter_estimation = ParameterEstimation(self.belgraph, self.config)
        parameter_estimation.get_distribution_for_roots_from_data()
        parameter_estimation.get_model_for_each_non_root_node()

        self.root_distributions = parameter_estimation.root_distributions
        self.trained_networks = parameter_estimation.trained_networks

        self.roots = self.belgraph.get_nodes_with_no_parents()

        # get exogenous distributions
        self.exogenous_dict = get_exogenous_samples(self.config, parameter_estimation.exogenous_std_dict)

        # 3. Build model
        self.model(exogenous_dict=self.exogenous_dict)

    def model(self, exogenous_dict):

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
                sample[current_node_name] = get_sample_for_non_roots(graph[current_node_name],
                                                                     exogenous_dict[current_node_name],
                                                                     deterministic_prediction)

                # [TODO] Move below two lines to a function
                child_name_list = [value["name"] for (key, value) in graph[current_node_name].children_info.items()]
                node_queue_for_bfs_traversal.extend(child_name_list)
                visited_nodes.append(current_node_name)
                node_queue_for_bfs_traversal.pop(0)

            else:
                # if all parents are not visited for a node, we assume that current node will be in the child info of
                # the unvisited parents
                node_queue_for_bfs_traversal.pop(0)
        return sample

    # [Todo]
    def counterfactual_inference(self, observation: dict, intervention_node_dict: dict, target, svi=True):
        model = self.model
        # [Todo]: get noise parameters from neural nets
        noise = {}
        noise_distribution_info = self.config.exogenous_distribution_info
        for node in self.graph.keys():
            exog_name = node.name + "_N"
            noise[exog_name] = pyro.sample(exog_name, get_distribution(noise_distribution_info))

        if svi:
            updated_noise, _ = model.update_noise_svi(observation, noise)

        counterfactual_model = model.intervention(intervention_node_dict)
        cf_posterior = model.infer(counterfactual_model, updated_noise)
        marginal = EmpiricalMarginal(cf_posterior, target)

        scm_causal_effect_samples = [
            observation[target] - float(marginal.sample())
            for _ in range(500)
        ]
        return scm_causal_effect_samples


    def condition(self, condition_data: dict):
        """
        It conditions self.model with condition data
        Returns: Conditioned pyro model
        """

        conditioned_model = pyro.condition(self.model, condition_data)
        return conditioned_model

    def intervention(self, intervention_data: dict):
        """
        It intervenes self.model with intervention data
        Returns: intervention pyro model
        """

        intervention_model = pyro.do(self.model, intervention_data)
        return intervention_model

    # [Todo]
    def infer(self, model, noise):
        return Importance(model, num_samples=1000).run(noise)

    def update_noise_svi(self, observed_steady_state, initial_noise: dict):

        """
        this performs stochastic variational inference for noise
        Args:
            observed_steady_state:
            initial_noise:
        Returns: Not sure now
        """

        def guide(exogenous_noise):
            noise_terms = list(exogenous_noise.keys())
            mu_constraints = constraints.interval(-3., 3)
            sigma_constraints = constraints.interval(.0001, 3)
            mu_guide = {
                k: pyro.param("mu_{}".format(k), torch.tensor(0.), constraint=mu_constraints) for k in noise_terms
            }
            sigma_guide = {
                k: pyro.param("sigma_{}".format(k), torch.tensor(1.), constraint=sigma_constraints) for k in noise_terms
            }
            for exogenous_noise in noise_terms:
                noise_dist = self.config.exogenous_distribution_info[0]
                pyro.sample(exogenous_noise, noise_dist(mu_guide[exogenous_noise], sigma_guide[exogenous_noise]))

        observational_model = self.condition(observed_steady_state)
        pyro.clear_param_store()
        svi = SVI(
            model=observational_model,
            guide=guide,
            optim=SGD({"lr": 0.001, "momentum": 0.1}),
            loss=Trace_ELBO()
        )
        losses = []
        num_steps = 1000
        samples = defaultdict(list)
        for t in range(num_steps):
            losses.append(svi.step(initial_noise))
            for noise in initial_noise.keys():
                mu = 'mu_{}'.format(noise)
                sigma = 'sigma_{}'.format(noise)
                samples[mu].append(pyro.param(mu).item())
                samples[sigma].append(pyro.param(sigma).item())
        means = {k: statistics.mean(v) for k, v in samples.items()}

        updated_noise = {}
        noise_distribution = self.config.exogenous_distribution_info[0]
        for n in initial_noise.keys():
            updated_noise[n] = noise_distribution(means["mu_{}".format(n)], means["sigma_{}".format(n)])

        return updated_noise, losses

    def _get_prediction(self, trained_network, parent_tensor):
        try:
            return trained_network.predict(parent_tensor)
        except:
            raise Exception("Error getting deterministic prediction.")


class Config:
    """
    Loads config file.
    """

    def __init__(self, config_file_path):
        self.config_dict = json_load(config_file_path)
        # self._check_config()
        self._set_config_parameters()

    def _set_config_parameters(self):

        config = self.config_dict
        self.prior_threshold = config["prior_threshold"]
        self.node_label_distribution_info = self._get_pyro_dist_from_text(config["node_label_distribution_info"])
        self.exogenous_distribution_info = self._get_exogenous_dist_from_text(config["exogenous_distribution_info"])
        self.parent_interaction_type = config["relation_type"]

    def _get_pyro_dist_from_text(self, node_label_distribution_info):
        """
        Converts string distribution names in config to pyro distributions. The names should be exact match.
        Args:
            node_label_distribution_info:

        Returns:

        """
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

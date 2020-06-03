import statistics
from collections import defaultdict

from pyro.infer import SVI, Trace_ELBO, Importance, EmpiricalMarginal
import torch.distributions.constraints as constraints
from pyro.optim import Adam

from Neuirps_BEL2SCM.bel_graph import BelGraph
from Neuirps_BEL2SCM.parameter_estimation import ParameterEstimation
from Neuirps_BEL2SCM.utils import get_sample_for_non_roots, get_parent_tensor, all_parents_visited, json_load, \
    get_parent_samples, get_child_name_list
from Neuirps_BEL2SCM.constants import PYRO_DISTRIBUTIONS, NOISE_TYPE, VARIABLE_TYPE, get_variable_type_from_label
from Neuirps_BEL2SCM.config import Config
import torch
import pyro


class SCM:
    """
    4. Add functions as described below
    """

    def __init__(self, bel_file_path, config_file_path, data_file_path):

        # 1. set parameters from config file.
        self.config = Config(config_file_path)

        # 2. get nodes from bel file.
        self.belgraph = BelGraph("nanopub_file", bel_file_path, data_file_path)
        self.belgraph.parse_input_to_construct_graph()

        self.belgraph.prepare_and_assign_data()
        self.graph = self.belgraph.nodes

        # Learn parameters
        parameter_estimation = ParameterEstimation(self.belgraph, self.config)
        parameter_estimation.get_distribution_for_roots_from_data()
        parameter_estimation.get_model_for_each_non_root_node()

        self.root_parameters = parameter_estimation.root_parameters
        self.trained_networks = parameter_estimation.trained_networks

        self.roots = self.belgraph.get_nodes_with_no_parents()

        # get exogenous distributions
        self.exogenous_dist_dict = self._get_exogenous_distributions()
        # exogenous_distribution_type = get_variable_type_from_label(roots[current_node_name].node_label)

        # 3. Build model
        self.model(exogenous_dist_dict=self.exogenous_dist_dict)

    def model(self, exogenous_dist_dict):

        # Getting class variables.
        graph = self.graph
        config = self.config
        roots = self.roots
        root_parameters = self.root_parameters
        trained_networks = self.trained_networks

        # Dictionary of pyro samples coming from pyro.sample() for each node.
        sample = dict()

        # node queue for bfs traversal
        node_queue_for_bfs_traversal = list()

        # visited nodes while traversal
        visited_nodes = list()

        # process all root nodes first for BFS traversal
        for current_node_name in roots.keys():
            node_distribution = config.node_label_distribution_info[roots[current_node_name].node_label]

            sample[current_node_name] = self._get_continuous_reparameterized_sample(current_node_name,
                                                                                    node_distribution,
                                                                                    exogenous_dist_dict[
                                                                                        current_node_name],
                                                                                    root_parameters[current_node_name])

            # add child nodes to queue
            node_queue_for_bfs_traversal = get_child_name_list(roots[current_node_name].children_info,
                                                               node_queue_for_bfs_traversal)
            # mark current node as visited
            visited_nodes.append(current_node_name)

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
                current_variable_type = get_variable_type_from_label(graph[current_node_name].node_label)
                # Get noise sample for current node.
                current_noise_sample = pyro.sample(current_node_name + "_N", exogenous_dist_dict[current_node_name])

                parent_sample_dict = get_parent_samples(graph[current_node_name], sample)
                parent_names = self.belgraph.parent_name_list_for_nodes
                deterministic_prediction = None

                if len(parent_names[current_node_name]) > 0:
                    parent_tensor = get_parent_tensor(parent_sample_dict, parent_names[current_node_name])
                    parent_tensor = parent_tensor.t()
                    deterministic_prediction = self._get_prediction(trained_networks[current_node_name],
                                                                    parent_tensor,
                                                                    current_noise_sample,
                                                                    current_variable_type)
                sample[current_node_name] = get_sample_for_non_roots(graph[current_node_name],
                                                                     config,
                                                                     deterministic_prediction)

                # add child nodes to queue
                node_queue_for_bfs_traversal = get_child_name_list(graph[current_node_name].children_info,
                                                                   node_queue_for_bfs_traversal)
                visited_nodes.append(current_node_name)
                node_queue_for_bfs_traversal.pop(0)

            else:
                # if all parents are not visited for a node, we assume that current node will be in the child info of
                # the unvisited parents
                node_queue_for_bfs_traversal.pop(0)
        return sample

    # [Todo]
    def counterfactual_inference(self, condition_data: dict, intervention_data: dict, target, svi=True):

        # Step 1. Condition the model
        conditioned_model = self.condition(condition_data)

        # Step 2. Noise abduction
        if svi:
            updated_noise, _ = self.update_noise_svi(conditioned_model)

        # Step 3. Intervene
        intervention_model = self.intervention(intervention_data)

        # Pass abducted noises to intervention model
        cf_posterior = self.infer(intervention_model, updated_noise)
        marginal = EmpiricalMarginal(cf_posterior, target)

        counterfactual_samples = [marginal.sample() for _ in range(1000)]
        # Calculate causal effect
        scm_causal_effect_samples = [
            torch.abs(condition_data[target] - float(marginal.sample()))
            for _ in range(500)
        ]
        return scm_causal_effect_samples, counterfactual_samples

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
        return Importance(model, num_samples=2000).run(noise)

    def update_noise_svi(self, conditioned_model):

        """
        this performs stochastic variational inference for noise
        Args:
            conditioned_model:
            noise:
        Returns: Not sure now
        """

        exogenous_dist_dict = self.exogenous_dist_dict

        def guide(exogenous_dist_dict):
            mu_constraints = constraints.interval(0., 1)
            sigma_constraints = constraints.interval(.1, 3.)

            for exg_name, exg_dist in exogenous_dist_dict.items():
                # mu_guide = pyro.param("mu_{}".format(exg_name), torch.tensor(exg_dist.loc), constraint=mu_constraints)
                # sigma_guide = pyro.param("sigma_{}".format(exg_name), torch.tensor(exg_dist.scale), constraint=sigma_constraints)

                mu_guide = pyro.param("mu_{}".format(exg_name), torch.tensor(0.0), constraint=mu_constraints)
                sigma_guide = pyro.param("sigma_{}".format(exg_name), torch.tensor(1.0),
                                         constraint=sigma_constraints)

                # [Todo] support the binary parent
                noise_dist = pyro.distributions.Normal
                pyro.sample(exg_name, noise_dist(mu_guide, sigma_guide))

        pyro.clear_param_store()

        svi = SVI(
            model=conditioned_model,
            guide=guide,
            optim=Adam({"lr": 0.005, "betas": (0.95, 0.999)}),
            loss=Trace_ELBO(retain_graph=True)
        )
        losses = []
        num_steps = 300
        samples = defaultdict(list)
        for t in range(num_steps):
            print(t)
            losses.append(svi.step(exogenous_dist_dict))
            for noise in exogenous_dist_dict.keys():
                mu = 'mu_{}'.format(noise)
                sigma = 'sigma_{}'.format(noise)
                samples[mu].append(pyro.param(mu).item())
                samples[sigma].append(pyro.param(sigma).item())
        means = {k: statistics.mean(v) for k, v in samples.items()}

        updated_noise = {}

        # [Todo] support the binary parent
        noise_distribution = pyro.distributions.Normal
        for n in exogenous_dist_dict.keys():
            updated_noise[n] = noise_distribution(means["mu_{}".format(n)], means["sigma_{}".format(n)])

        return updated_noise, losses

    def _get_prediction(self, trained_network, parent_tensor, current_noise_sample, current_variable_type):
        try:
            return trained_network.net(parent_tensor) + current_noise_sample
        except:
            raise Exception("Error getting deterministic prediction.")

    def _get_continuous_reparameterized_sample(self, current_node_name, node_distribution, exogenous_distribution,
                                               root_parameters):
        """

        Args:
            current_node_name:
            node_distribution:
            exogenous_distribution:
            root_parameters:

        Returns: pyro.sample()

        """
        noise_sample = pyro.sample(current_node_name + "_N", exogenous_distribution)

        # [Todo] We could turn root parameters into a class later.
        current_mu = root_parameters[0]
        current_std = root_parameters[1]

        parent_value = current_mu + noise_sample * current_std
        return pyro.sample(current_node_name, node_distribution(parent_value, 1.0))

    def _get_exogenous_distributions(self):
        exogenous_dist_dict = {}

        for node_name in self.roots:
            exogenous_distribution_type = get_variable_type_from_label(self.graph[node_name].node_label)
            exogenous_distribution = NOISE_TYPE[exogenous_distribution_type]
            exogenous_dist_dict[node_name] = exogenous_distribution(0.0, 1.0)

        for node_name, trained_network in self.trained_networks.items():
            exogenous_distribution_type = get_variable_type_from_label(self.graph[node_name].node_label)
            exogenous_distribution = NOISE_TYPE[exogenous_distribution_type]

            residual_mean = trained_network.residual_mean
            residual_std = trained_network.residual_std

            # Override the parameters written in the constants.py
            exogenous_dist_dict[node_name] = exogenous_distribution(0.0, residual_std)
            #exogenous_dist_dict[node_name] = exogenous_distribution(0.0, 1.0)
        return exogenous_dist_dict
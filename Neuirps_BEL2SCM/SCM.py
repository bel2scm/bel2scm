from Neuirps_BEL2SCM.Utils import *
from Neuirps_BEL2SCM.BelGraph import BelGraph
import json


class SCM:
    '''
    4. Add functions as described below
    '''
    def __init__(self, bel_file_path, config_file_path):

        # 1. get tree from bel - Done.
        self.graph = BelGraph("nanopub_file", bel_file_path).construct_graph_from_nanopub_file()

        # 2. set parameters from config - Done.
        self.config = self._json_load(config_file_path)

        # 3. Build model
        self._build_model()

    # [Todo] Move to Utils
    def _json_load(self, filepath):
        try:
            with open(filepath) as json_file:
                return json.load(json_file)
        except FileNotFoundError:
            print("Error: Wrong file or file path.")

    def _build_model(self):
        return NotImplementedError

    # [Todo]
    def counterfactual_infernce(self):
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


def scm(nodes, config):
    """
    Description: This function is to be build a Structural Causal Model for
                  for any child-parent cluster
    Parameters: knowledge graph as dataframe,
                threshold for cutoff
                weights for each parents
    Returns: sampled values for all nodes in tensor format
    """
    pyro_settings = config["pyro_settings"]
    node_settings = config["node_type_settings"]
    exogenous_var_settings = config["exogenous_var_settings"]
    threshold = pyro_settings["threshold"]
    weight = pyro_settings["weights"]
    samples = {}
    exogenous = []
    current_node = None
    root_list = []
    node_list = []
    visited = []

    for node in nodes:
        if nodes[node].root == True:
            root_list.append(node)
    for node in root_list:
        if nodes[node].node_label == 'abundance':
            dist_parameters = [pyro_settings['mu_a'], pyro_settings['sigma_a']]
        elif nodes[node].node_label == 'transformation':
            dist_parameters = [pyro_settings['mu_t'], pyro_settings['sigma_t']]
        else:
            dist_parameters = [pyro_settings['cat_0'], pyro_settings['cat_1']]
        parent_name_noise = node + "_N"
        parent_N = pyro.sample(parent_name_noise,
                               get_distribution(node_settings[nodes[node].node_label], dist_parameters)
                               )
        parent = pyro.sample(node, pyro.distributions.Delta(parent_N))
        samples[node] = parent
        visited.append(node)
        exogenous.append(node)
        c_list = nodes[node].children
        for c in c_list:
            node_list.append(c)

    while len(node_list) > 0:
        current_node = node_list[0]
        parent_label = nodes[current_node].parent_label
        child_label = nodes[current_node].node_label
        relation = nodes[current_node].parent_relations
        parent_name = nodes[current_node].parents
        child_name = nodes[current_node].name
        w = [weight] * len(parent_label)

        parents = []
        increase_process = []
        decrease_process = []
        increase_abundance = []
        decrease_abundance = []
        increase_transformation = []
        decrease_transformation = []
        weight_ai = []
        weight_ad = []
        weight_ti = []
        weight_td = []

        gamma = get_distribution(exogenous_var_settings["continuous_t"], [pyro_settings["alpha"], pyro_settings["beta"]]
                                 )
        lognormal = get_distribution(exogenous_var_settings["continuous_a"], [pyro_settings["mu_a"],
                                                                              pyro_settings["sigma_a"]])
        normal = get_distribution(exogenous_var_settings["categorical"], [pyro_settings["mu_n"],
                                                                          pyro_settings["sigma_n"]])
        visited_parents_count = 0

        for i in range(len(parent_label)):
            if parent_name[i] in samples:
                parents.append(samples[parent_name[i]])
                visited_parents_count += 1
        groupby_process = ["process", "activity", "reaction", "pathology"]
        increase_process, decrease_process, _, _ = cat_parents(parent_label, parent_name, relation, w, samples,
                                                         groupby_process)

        increase_abundance, decrease_abundance, weight_ai, weight_ad = cat_parents(parent_label, parent_name,
                                                                                   relation, w, samples,
                                                                                   groupby=["abundance"])
        increase_transformation, decrease_transformation, weight_ti, weight_td = cat_parents(parent_label,
                                                                                             parent_name,
                                                                                             relation,
                                                                                             w, samples,
                                                                                             groupby=[
                                                                                                 "transformation"])


        if visited_parents_count != len(parent_label):
            node_list.pop(0)
            continue
        if child_name not in visited:
            if any(x in ["process", "activity", "reaction", "pathology"] for x in parent_label):
                if sum(decrease_process) == 0 and sum(increase_process) == len(increase_process):
                    child_N, exog = get_sample(child_name, child_label, parent_label, threshold, normal, gamma,
                                               lognormal,
                                               increase_process, decrease_process, increase_abundance,
                                               decrease_abundance,
                                               weight_ai, weight_ad, increase_transformation, decrease_transformation,
                                               weight_ti, weight_td)
                else:
                    child_N = torch.tensor(0.)
            else:
                child_N, exog = get_sample(child_name, child_label, parent_label, threshold, normal, gamma, lognormal,
                                           increase_process, decrease_process, increase_abundance, decrease_abundance,
                                           weight_ai, weight_ad, increase_transformation, decrease_transformation,
                                           weight_ti, weight_td)

            child = pyro.sample(child_name, pyro.distributions.Delta(child_N))
            samples[child_name] = child
            exogenous.append(exog)
            visited.append(child_name)

        c_list = nodes[current_node].children
        for c in c_list:
            node_list.append(c)
        node_list.pop(0)

    return samples

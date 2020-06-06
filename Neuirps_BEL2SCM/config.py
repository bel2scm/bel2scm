from Neuirps_BEL2SCM.utils import json_load
from Neuirps_BEL2SCM.constants import PYRO_DISTRIBUTIONS

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
        self.parent_interaction_type = config["relation_type"]
        self.continuous_max_abundance = config["continuous_max_abundance"]

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

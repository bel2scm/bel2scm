import unittest


class TestSCM(unittest.TestCase):
    # def test_something(self):
    #     self.assertEqual(True, False)

    def test_scm_integration_test(self):

        from Neuirps_BEL2SCM.SCM import SCM
        from Neuirps_BEL2SCM.Utils import json_load

        bel_file_path = "E:\\Github\\Bel2SCM\\bel2scm\\Tests\\BELSourceFiles\\COVID-19-new.json"
        config_file_path = "E:\\Github\\Bel2SCM\\bel2scm\\Tests\\Configs\\COVID-19-config.json"

        scm = SCM(bel_file_path, config_file_path)

        bel_assertions = json_load(bel_file_path)[0]['nanopub']['assertions']
        bel_assertion_count = self._get_unique_name_count_from_bel_assertion(bel_assertions)

        self.assertEqual(len(scm.graph), bel_assertion_count)

    def test_config(self):
        from Neuirps_BEL2SCM.Utils import json_load

        config_file_path = "E:\\Github\\Bel2SCM\\bel2scm\\Tests\\Configs\\COVID-19-config.json"

        config = json_load(config_file_path)

        # Add a check here whenever we edit config structure
        prior_weight_check = "prior_weight" in config.keys()
        prior_threshold = "prior_threshold" in config.keys()
        node_label_distribution_info = "node_label_distribution_info" in config.keys()
        exogenous_var_distribution_info = "exogenous_var_distribution_info" in config.keys()
        relation_type = "relation_type" in config.keys()

        self.assertTrue(prior_weight_check)
        self.assertTrue(prior_threshold)
        self.assertTrue(node_label_distribution_info)
        self.assertTrue(exogenous_var_distribution_info)
        self.assertTrue(relation_type)

    def _get_unique_name_count_from_bel_assertion(self, bel_assertions):
        names = []
        for assertion in bel_assertions:
            if assertion["subject"] not in names:
                names.append(assertion["subject"])
            if assertion["object"] not in names:
                names.append(assertion["object"])

        return len(names)

if __name__ == '__main__':
    unittest.main()

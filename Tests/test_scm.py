import unittest


class TestSCM(unittest.TestCase):
    # def test_something(self):
    #     self.assertEqual(True, False)

    def test_scm_integration_test(self):

        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import json_load

        bel_file_path = "../Tests/BELSourceFiles/small-IGF-pwy.nanopub.graphdati.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"

        scm = SCM(bel_file_path, config_file_path)

        bel_assertions = json_load(bel_file_path)[0]['nanopub']['assertions']
        bel_assertion_count = self._get_unique_name_count_from_bel_assertion(bel_assertions)

        self.assertEqual(len(scm.graph), bel_assertion_count)

    def test_config(self):
        from Neuirps_BEL2SCM.utils import json_load

        config_file_path = "../Tests/Configs/COVID-19-config.json"

        config = json_load(config_file_path)

        # Add a check here whenever we edit config structure
        prior_weight_check = "prior_weight" in config.keys()
        prior_threshold = "prior_threshold" in config.keys()
        node_label_distribution_info = "node_label_distribution_info" in config.keys()
        exogenous_distribution_info = "exogenous_distribution_info" in config.keys()
        relation_type = "relation_type" in config.keys()

        self.assertTrue(prior_weight_check)
        self.assertTrue(prior_threshold)
        self.assertTrue(node_label_distribution_info)
        self.assertTrue(exogenous_distribution_info)
        self.assertTrue(relation_type)

    def _get_unique_name_count_from_bel_assertion(self, bel_assertions):
        names = []
        for assertion in bel_assertions:
            if assertion["subject"] not in names:
                names.append(assertion["subject"])
            if assertion["object"] not in names:
                names.append(assertion["object"])

        return len(names)

    def test_model(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import json_load
        bel_file_path = "../Tests/BELSourceFiles/small-IGF-pwy.nanopub.graphdati.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"

        scm = SCM(bel_file_path, config_file_path)

        bel_assertions = json_load(bel_file_path)[0]['nanopub']['assertions']
        scm._build_model()

    def test_mapk(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import json_load
        import pickle
        from Neuirps_BEL2SCM.utils import save_scm_object

        bel_file_path = "../Tests/BELSourceFiles/mapk.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/mapk3000.csv"
        output_pickle_object_file = "../../mapk_scm.pkl"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        save_scm_object(output_pickle_object_file, scm)
        # Add loading and saving from pkl to utils

    def test_load_scm(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import load_scm_object

        scm = load_scm_object("../../mapk_scm.pkl")
        self.assertTrue(type(scm) == SCM)

    def test_generate_mapk_samples(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import load_scm_object

        scm = load_scm_object("../../mapk_scm.pkl")
        samples = [scm.model() for i in range(1000)]

        # [TODO] Compare the mean of each variable with data itself.
        self.assertTrue(True)

    def test_binary_mapk(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import json_load
        import pickle
        from Neuirps_BEL2SCM.utils import save_scm_object

        bel_file_path = "../Tests/BELSourceFiles/MAPK-binary.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/mapk3000-binary.csv"
        output_pickle_object_file = "../../mapk_binary_scm.pkl"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        save_scm_object(output_pickle_object_file, scm)

    def test_generate_binary_mapk_samples(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import load_scm_object

        scm = load_scm_object("../../mapk_binary_scm.pkl")
        samples = [scm.model() for i in range(1000)]

        # [TODO] Compare the mean of each variable with data itself.
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()

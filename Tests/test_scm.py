import unittest


class TestSCM(unittest.TestCase):
    # def test_something(self):
    #     self.assertEqual(True, False)

    def test_scm_integration_test(self):

        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import json_load

        bel_file_path = "../Tests/BELSourceFiles/small-IGF-pwy.nanopub.graphdati.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/mapk3000-binary.csv"

        scm = SCM(bel_file_path, config_file_path, data_file_path)

        bel_assertions = json_load(bel_file_path)[0]['nanopub']['assertions']
        bel_assertion_count = self._get_unique_name_count_from_bel_assertion(bel_assertions)

        self.assertEqual(len(scm.graph), bel_assertion_count)

    def _get_unique_name_count_from_bel_assertion(self, bel_assertions):
        names = []
        for assertion in bel_assertions:
            if assertion["subject"] not in names:
                names.append(assertion["subject"])
            if assertion["object"] not in names:
                names.append(assertion["object"])

        return len(names)

    def test_mapk(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import json_load
        import pickle
        from Neuirps_BEL2SCM.utils import save_scm_object
        import torch
        import pyro
        import numpy as np
        torch.manual_seed(101)
        pyro.set_rng_seed(42)
        np.random.seed(19680801)

        bel_file_path = "../Tests/BELSourceFiles/mapk_single_interaction.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/single_interaction_data.csv"
        # data_file_path = "../Tests/Data/single_interaction_data.csv"
        output_pickle_object_file = "../../mapk_scm.pkl"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        exogenous_noise = scm.exogenous_dist_dict
        samples = [scm.model(exogenous_noise) for i in range(1000)]
        print(samples)
        save_scm_object(output_pickle_object_file, scm)
        # Add loading and saving from pkl to utils

    def test_generate_mapk_samples(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import load_scm_object

        scm = load_scm_object("../../mapk_scm.pkl")
        exogenous_noise = scm.exogenous_dist_dict
        samples = [scm.model(exogenous_noise) for i in range(1000)]

        # [TODO] Compare the mean of each variable with data itself.
        self.assertTrue(True)

    def test_binary_mapk(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import json_load
        import pickle
        from Neuirps_BEL2SCM.utils import save_scm_object

        bel_file_path = "BELSourceFiles/mapk-binary.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/mapk3000-binary.csv"
        output_pickle_object_file = "../../mapk_binary_scm.pkl"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        save_scm_object(output_pickle_object_file, scm)

    def test_generate_binary_mapk_samples(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import load_scm_object

        scm = load_scm_object("../../mapk_binary_scm.pkl")
        exogenous_noise = scm.exogenous_dist_dict
        samples = [scm.model(exogenous_noise) for i in range(1000)]

        # [TODO] Compare the mean of each variable with data itself.
        self.assertTrue(True)

    def test_mapk_counterfactual(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import json_load
        import pyro
        import torch
        import numpy as np
        torch.manual_seed(101)
        pyro.set_rng_seed(42)
        np.random.seed(19680801)

        bel_file_path = "../Tests/BELSourceFiles/mapk_single_interaction.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/single_interaction_data.csv"

        scm = SCM(bel_file_path, config_file_path, data_file_path)

        exogenous_noise = scm.exogenous_dist_dict
        condition_data = scm.model(exogenous_noise)
        target = "a(p(Erk))"
        intervention_data = {
            "a(p(Mek))": 60.0
        }

        erk_causal_effects, counterfactual_samples1 = scm.counterfactual_inference(condition_data, intervention_data, target, True)
        print(counterfactual_samples1)
        new_list = [x.cpu().detach().numpy() for x in counterfactual_samples1]
        print("Counterfactual Erk when Mek = 60:: Mean", np.mean(new_list), np.std(new_list))

        intervention_data2 = {
            "a(p(Mek))": 80.0
        }
        _, counterfactual_samples2 = scm.counterfactual_inference(condition_data, intervention_data2,
                                                                                   target, True)
        print(counterfactual_samples2)
        new_list2 = [x.cpu().detach().numpy() for x in counterfactual_samples2]
        print("Counterfactual Erk when Mek = 80:: Mean", np.mean(new_list2), np.std(new_list2))

    def test_igf(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import json_load
        import pickle
        from Neuirps_BEL2SCM.utils import save_scm_object

        bel_file_path = "../Tests/BELSourceFiles/igf.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/observational_igf.csv"
        output_pickle_object_file = "../../igf_scm.pkl"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        save_scm_object(output_pickle_object_file, scm)
        # exogenous_noise = scm.exogenous_dist_dict
        # samples = [scm.model(exogenous_noise) for i in range(1000)]
        # print(samples)
        self.assertTrue(True, True)

    def test_igf_intervention(self):
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import json_load
        import torch
        import pandas as pd

        bel_file_path = "../Tests/BELSourceFiles/igf.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/observational_igf.csv"
        output_pickle_object_file = "../../igf_scm.pkl"

        scm = SCM(bel_file_path, config_file_path, data_file_path)

        exogenous_noise = scm.exogenous_dist_dict
        condition_data = scm.model(exogenous_noise)
        # target = "a(p(Erk))"
        intervention_data = {
            "a(p(Ras))": 30.0
        }

        do_model = scm.intervention(intervention_data)
        samples = [do_model(exogenous_noise) for _ in range(5000)]
        df = pd.DataFrame(samples)
        for col in df.columns:
            for i in range(len(df)):
                if torch.is_tensor(df[col][i]):
                    df[col][i] = df[col][i].item()
        df.to_csv("../../intervention_samples_igf.csv")
        self.assertTrue(True, True)

    def test_error_with_sde(self):
        import pandas as pd
        df_bel = pd.read_csv("../Tests/Data/intervention_samples_igf.csv")
        df_sde = pd.read_csv("../Tests/Data/intervention_igf.csv")
        errors = {}
        for col in range(len(df_sde.columns)):
            errors[df_sde.columns[col]] = df_sde[df_sde.columns[col]].mean() - df_bel[df_bel.columns[col]].mean()
        print(errors)
        self.assertTrue(True, True)


if __name__ == '__main__':
    unittest.main()

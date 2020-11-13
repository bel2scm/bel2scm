"""
------------------CODE DESCRIPTION----------------------------------
    This is the test file which stores experiments to generate data
    for plots in the paper that were generated using bel2scm algorithm.
    
    Check Tests/test_scm.py to see unit tests for usability/debugging.
    
    Check Tests/code-for-experiments/test_sigmoid_covid_known_parameters.py
    to see experiments that were used to generate data from SCM with known parameters.
    
    All dataframes generated for this paper are in Tests/Data folder.
"""



import unittest
class TestBEL2SCM(unittest.TestCase):

    def test_igf_intervention_on_ras(self):
        """
        Description: This experiment gets causal effect on erk by
                    intervening on mek for igf graph using bel2scm algorithm
        """
        from Neuirps_BEL2SCM.scm import SCM
        from Neuirps_BEL2SCM.utils import json_load
        import torch
        torch.manual_seed(101)
        import pandas as pd
    
        bel_file_path = "../Tests/BELSourceFiles/igf.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/observational_igf.csv"
    
        scm = SCM(bel_file_path, config_file_path, data_file_path)
    
        exogenous_noise = scm.exogenous_dist_dict
        condition_data = scm.model(exogenous_noise)
        print(condition_data)
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
        df2 = pd.read_csv("../Tests/Data/bel2scm_samples_igf.csv")
        erk_diff = df["a(p(Erk))"] - df2["a(p(Erk))"]
        erk_diff.to_csv("../Tests/Data/erk_do_ras_30_minus_erk.csv")
        df.to_csv("../Tests/Data/intervention_samples_igf.csv")
        self.assertTrue(True, True)

    def test_igf_intervention_on_mek(self):
        from Neuirps_BEL2SCM.scm import SCM
        import torch
        torch.manual_seed(23)
        import pandas as pd

        bel_file_path = "../Tests/BELSourceFiles/igf.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/observational_igf.csv"
        output_pickle_object_file = "../../igf_scm.pkl"

        scm = SCM(bel_file_path, config_file_path, data_file_path)

        exogenous_noise = scm.exogenous_dist_dict
        condition_data = scm.model(exogenous_noise)
        print(condition_data)
        # target = "a(p(Erk))"
        intervention_data = {
            "a(p(Mek))": 40.0
        }

        do_model = scm.intervention(intervention_data)
        samples = [do_model(exogenous_noise) for _ in range(5000)]
        df = pd.DataFrame(samples)
        for col in df.columns:
            for i in range(len(df)):
                if torch.is_tensor(df[col][i]):
                    df[col][i] = df[col][i].item()
        df2 = pd.read_csv("../Tests/Data/bel2scm_samples_igf.csv")
        erk_diff = df["a(p(Erk))"] - df2["a(p(Erk))"]
        erk_diff.to_csv("../Tests/Data/erk_do_mek_40_minus_erk.csv")
        df.to_csv("../Tests/Data/intervention_mek_40_samples_igf.csv")
        self.assertTrue(True, True)

if __name__ == '__main__':
    unittest.main()

"""
------------------CODE DESCRIPTION----------------------------------
    This is the test file which stores experiments to generate data
    for plots in the paper that were generated using bel2scm algorithm.

    Check test_bel2scm.py to see unit tests for usability/debugging.

    Check test_plots_known_parameters_scm.py
    to see experiments that were used to generate data from SCM with known parameters.

    All dataframes generated for this paper are in Tests/Data folder.
"""

import unittest

from bel2scm.neurips_bel2scm.scm import SCM
from bel2scm.neurips_bel2scm.utils import json_load
from bel2scm.neurips_bel2scm.utils import save_scm_object
from bel2scm.neurips_bel2scm.utils import load_scm_object
import torch
import pandas as pd
import time
import numpy as np
from torch import tensor

class TestSCM(unittest.TestCase):

    def test_igf_intervention_on_ras(self):
        """
        Description: This experiment gets causal effect on erk by
                    intervening on mek for igf graph using bel2scm algorithm
        """

        bel_file_path = "../BELSourceFiles/igf.json"
        config_file_path = "../Configs/COVID-19-config.json"
        data_file_path = "../Data/observational_igf.csv"

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
        df2 = pd.read_csv("../Data/bel2scm_samples_igf.csv")
        erk_diff = df["a(p(Erk))"] - df2["a(p(Erk))"]
        erk_diff.to_csv("../Data/erk_do_ras_30_minus_erk.csv")
        df.to_csv("../Data/intervention_samples_igf.csv")
        self.assertTrue(True, True)

    def test_igf_intervention_on_mek(self):

        bel_file_path = "../BELSourceFiles/igf.json"
        config_file_path = "../Configs/COVID-19-config.json"
        data_file_path = "../Data/observational_igf.csv"
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
        df2 = pd.read_csv("../Data/bel2scm_samples_igf.csv")
        erk_diff = df["a(p(Erk))"] - df2["a(p(Erk))"]
        erk_diff.to_csv("../Data/erk_do_mek_40_minus_erk.csv")
        df.to_csv("../Data/intervention_mek_40_samples_igf.csv")
        self.assertTrue(True, True)

    def test_covid_causal_effect_with_estimated_parameters_datapoint1(self):

        torch.manual_seed(23)
        time1 = time.time()
        bel_file_path = "../BELSourceFiles/covid_input.json"
        config_file_path = "../Configs/COVID-19-config.json"
        data_file_path = "../Data/observational_samples_from_sigmoid_known_parameters.csv"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        condition_data = {
            'a(SARS_COV2)': tensor(67.35032),
            'a(PRR)': tensor(89.7037),
            'a(ACE2)': tensor(29.747593),
            'a(AngII)': tensor(68.251114),
            'a(AGTR1)': tensor(90.96106999999999),
            'a(ADAM17)': tensor(86.84893000000001),
            'a(TOCI)': tensor(40.76684),
            'a(TNF)': tensor(76.85005),
            'a(sIL_6_alpha)': tensor(87.99491),
            'a(EGF)': tensor(84.55391),
            'a(EGFR)': tensor(79.94534),
            'a(IL6_STAT3)': tensor(83.39896),
            'a(NF_xB)': tensor(82.79433399999999),
            'a(IL6_AMP)': tensor(81.38015),
            'a(cytokine)': tensor(80.21895)

        }
        target = "a(cytokine)"
        intervention_data = {
            "a(TOCI)": 0.0
        }

        causal_effects1, counterfactual_samples1 = scm.counterfactual_inference(condition_data, intervention_data,
                                                                                target, True)
        print("time required for causal effects", time.time() - time1)
        samples_df = pd.DataFrame(causal_effects1)
        samples_df.to_csv("../Data/causal_effect_sigmoid_with_estimated_parameters_datapoint1.csv", index=False)

    def test_covid_causal_effect_with_estimated_parameters_datapoint2(self):

        torch.manual_seed(23)
        time1 = time.time()
        bel_file_path = "../BELSourceFiles/covid_input.json"
        config_file_path = "../Configs/COVID-19-config.json"
        data_file_path = "../Data/observational_samples_from_sigmoid_known_parameters.csv"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        condition_data = {
            'a(SARS_COV2)': 61.631156999999995,
            'a(PRR)': 87.76389,
            'a(ACE2)': 39.719845,
            'a(AngII)': 59.212959999999995,
            'a(AGTR1)': 84.39899399999999,
            'a(ADAM17)': 85.84442,
            'a(TOCI)': 67.33063,
            'a(TNF)': 77.83915,
            'a(sIL_6_alpha)': 57.584044999999996,
            'a(EGF)': 86.26822,
            'a(EGFR)': 81.4849,
            'a(IL6_STAT3)': 69.57323000000001,
            'a(NF_xB)': 83.75941,
            'a(IL6_AMP)': 77.52906,
            'a(cytokine)': 79.07555

        }
        target = "a(cytokine)"
        intervention_data = {
            "a(TOCI)": 0.0
        }

        causal_effects1, counterfactual_samples1 = scm.counterfactual_inference(condition_data, intervention_data,
                                                                                target, True)
        print("time required for causal effects", time.time() - time1)
        samples_df = pd.DataFrame(causal_effects1)
        samples_df.to_csv("../Data/causal_effect_sigmoid_with_estimated_parameters_datapoint2.csv", index=False)


if __name__ == '__main__':
    unittest.main()

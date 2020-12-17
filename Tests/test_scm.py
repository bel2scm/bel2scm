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
    # def test_something(self):
    #     self.assertEqual(True, False)

    def test_scm_integration_test(self):
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
        bel_file_path = "../Tests/BELSourceFiles/mapk.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/mapk3000.csv"
        # data_file_path = "../Tests/Data/single_interaction_data.csv"
        output_pickle_object_file = "../../mapk_scm.pkl"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        save_scm_object(output_pickle_object_file, scm)
        # Add loading and saving from pkl to utils

    def test_mapk_erk_samples(self):
        bel_file_path = "../Tests/BELSourceFiles/mapk.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/mapk3000.csv"
        # data_file_path = "../Tests/Data/single_interaction_data.csv"
        # output_pickle_object_file = "../../mapk_scm.pkl"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        exogenous_noise = scm.exogenous_dist_dict
        samples = [scm.model(exogenous_noise) for i in range(3000)]
        print(samples)
        # save_scm_object(output_pickle_object_file, scm)

    def test_generate_mapk_samples(self):
        scm = load_scm_object("../../mapk_scm.pkl")
        exogenous_noise = scm.exogenous_dist_dict
        samples = [scm.model(exogenous_noise) for i in range(1000)]

        # [TODO] Compare the mean of each variable with data itself.
        self.assertTrue(True)

    def test_binary_mapk(self):
        bel_file_path = "BELSourceFiles/mapk-binary.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/mapk3000-binary.csv"
        output_pickle_object_file = "../../mapk_binary_scm.pkl"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        save_scm_object(output_pickle_object_file, scm)

    def test_generate_binary_mapk_samples(self):

        scm = load_scm_object("../../mapk_binary_scm.pkl")
        exogenous_noise = scm.exogenous_dist_dict
        samples = [scm.model(exogenous_noise) for i in range(5000)]
        # [TODO] Compare the mean of each variable with data itself.
        self.assertTrue(True)

    def test_mapk_counterfactual(self):
        torch.manual_seed(101)
        time1 = time.time()
        bel_file_path = "../Tests/BELSourceFiles/mapk.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/mapk3000.csv"

        scm = SCM(bel_file_path, config_file_path, data_file_path)

        exogenous_noise = scm.exogenous_dist_dict
        condition_data = scm.model(exogenous_noise)
        target = "a(p(Erk))"
        intervention_data = {
            "a(p(Mek))": 60.0
        }

        erk_causal_effects, counterfactual_samples1 = scm.counterfactual_inference(condition_data, intervention_data,
                                                                                   target, True)
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
        print("total time taken to run this experiment is ", time.time() - time1)

    def test_igf(self):
        torch.manual_seed(101)

        bel_file_path = "../Tests/BELSourceFiles/igf.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/observational_igf.csv"
        # output_pickle_object_file = "../../igf_scm.pkl"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        # save_scm_object(output_pickle_object_file, scm)
        exogenous_noise = scm.exogenous_dist_dict
        samples = [scm.model(exogenous_noise) for i in range(5000)]
        df = pd.DataFrame(samples)
        for col in df.columns:
            for i in range(len(df)):
                if torch.is_tensor(df[col][i]):
                    df[col][i] = df[col][i].item()
        df.to_csv("../Tests/Data/bel2scm_samples_igf.csv")
        self.assertTrue(True, True)

    def test_igf_intervention_on_ras(self):

        import torch
        torch.manual_seed(101)
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
        torch.manual_seed(23)

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

    def test_error_with_sde(self):
        df_bel = pd.read_csv("../Tests/Data/intervention_samples_igf.csv")
        df_sde = pd.read_csv("../Tests/Data/intervention_igf.csv")
        errors = {}
        for col in range(len(df_sde.columns)):
            errors[df_sde.columns[col]] = df_sde[df_sde.columns[col]].mean() - df_bel[df_bel.columns[col]].mean()
        print(errors)
        self.assertTrue(True, True)

    def test_covid(self):
        torch.manual_seed(23)

        bel_file_path = "../Tests/BELSourceFiles/covid_input.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/observational_samples_from_sigmoid_known_parameters.csv"
        output_pickle_object_file = "../../covid_scm.pkl"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        exogenous_noise = scm.exogenous_dist_dict
        samples = [scm.model(exogenous_noise) for i in range(5000)]
        df = pd.DataFrame(samples)
        for col in df.columns:
            for i in range(len(df)):
                if torch.is_tensor(df[col][i]):
                    df[col][i] = df[col][i].item()
        df.to_csv("../Tests/Data/bel2scm_samples_covid.csv")
        save_scm_object(output_pickle_object_file, scm)
        # Add loading and saving from pkl to utils

    def test_covid_direct_simulation_causal_effect(self):
        torch.manual_seed(23)
        time1 = time.time()
        bel_file_path = "../Tests/BELSourceFiles/covid_input.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/observational_samples_from_sigmoid_known_parameters.csv"
        scm = SCM(bel_file_path, config_file_path, data_file_path)

        exogenous_noise = scm.exogenous_dist_dict
        condition_data = scm.model(exogenous_noise)
        intervention_data = {
            "a(TOCI)": 0.0
        }

        do_model = scm.intervention(intervention_data)
        samples = [do_model(exogenous_noise) for _ in range(5000)]
        df = pd.DataFrame(samples)
        for col in df.columns:
            for i in range(len(df)):
                if torch.is_tensor(df[col][i]):
                    df[col][i] = df[col][i].item()
        df2 = pd.read_csv("../Tests/Data/bel2scm_samples_covid.csv")
        cytokine_diff = df["a(cytokine)"] - df2["a(cytokine)"]
        cytokine_diff.to_csv("../Tests/Data/cytokine_do_toci_0_minus_cytokine.csv")
        df.to_csv("../Tests/Data/intervention_toci_0_samples_covid.csv")
        self.assertTrue(True, True)

    def test_covid_noisy_model_samples(self):
        torch.manual_seed(23)
        time1 = time.time()
        bel_file_path = "../Tests/BELSourceFiles/covid_input.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/covid_noisy_reparameterized_data.csv"

        scm = SCM(bel_file_path, config_file_path, data_file_path)

        exogenous_noise = scm.exogenous_dist_dict
        samples = torch.tensor([list(scm.model(exogenous_noise).values()) for _ in range(5500)]).detach().numpy()
        np.savetxt("../Tests/Data/covid_bel2scm_noisy_reparameterized_samples.csv", samples, delimiter=',')

    def test_covid_toci0_mm_causal_effect(self):
        torch.manual_seed(23)
        time1 = time.time()
        bel_file_path = "../Tests/BELSourceFiles/covid_input.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/observational_samples_from_sigmoid_known_parameters.csv"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        condition_data = {
            # 'a(SARS_COV2)': tensor(67.35032),
            # 'a(PRR)': tensor(89.7037),
            # 'a(ACE2)': tensor(29.747593),
            # 'a(AngII)': tensor(68.251114),
            # 'a(AGTR1)': tensor(90.96106999999999),
            # 'a(ADAM17)': tensor(86.84893000000001),
            # 'a(TOCI)': tensor(40.76684),
            # 'a(TNF)': tensor(76.85005),
            # 'a(sIL_6_alpha)': tensor(87.99491),
            # 'a(EGF)': tensor(84.55391),
            # 'a(EGFR)': tensor(79.94534),
            # 'a(IL6_STAT3)': tensor(83.39896),
            # 'a(NF_xB)': tensor(82.79433399999999),
            # 'a(IL6_AMP)': tensor(81.38015),
            # 'a(cytokine)': tensor(80.21895)

            'a(SARS_COV2)': tensor(61.631156999999995),
            'a(PRR)': tensor(87.76389),
            'a(ACE2)': tensor(39.719845),
            'a(AngII)': tensor(59.212959999999995),
            'a(AGTR1)': tensor(84.39899399999999),
            'a(ADAM17)': tensor(85.84442),
            'a(TOCI)': tensor(67.33063),
            'a(TNF)': tensor(77.83915),
            'a(sIL_6_alpha)': tensor(57.584044999999996),
            'a(EGF)': tensor(86.26822),
            'a(EGFR)': tensor(81.4849),
            'a(IL6_STAT3)': tensor(69.57323000000001),
            'a(NF_xB)': tensor(83.75941),
            'a(IL6_AMP)': tensor(77.52906),
            'a(cytokine)': tensor(79.07555)

        }
        target = "a(cytokine)"
        intervention_data = {
            "a(TOCI)": 0.0
        }

        causal_effects1, counterfactual_samples1 = scm.counterfactual_inference(condition_data, intervention_data,
                                                                                target, True)
        print("time required for causal effects", time.time() - time1)
        samples_df = pd.DataFrame(causal_effects1)
        samples_df.to_csv("../Tests/Data/causal_effect_MM_bel2scm.csv", index=False)

    def test_covid_toci0_bel2scm_causal_effect(self):
        torch.manual_seed(23)
        bel_file_path = "../Tests/BELSourceFiles/covid_input.json"
        config_file_path = "../Tests/Configs/COVID-19-config.json"
        data_file_path = "../Tests/Data/observational_samples_from_sigmoid_known_parameters.csv"

        scm = SCM(bel_file_path, config_file_path, data_file_path)
        condition_data = {
            # 'a(SARS_COV2)': tensor(67.35032),
            # 'a(PRR)': tensor(89.7037),
            # 'a(ACE2)': tensor(29.747593),
            # 'a(AngII)': tensor(68.251114),
            # 'a(AGTR1)': tensor(90.96106999999999),
            # 'a(ADAM17)': tensor(86.84893000000001),
            # 'a(TOCI)': tensor(40.76684),
            # 'a(TNF)': tensor(76.85005),
            # 'a(sIL_6_alpha)': tensor(87.99491),
            # 'a(EGF)': tensor(84.55391),
            # 'a(EGFR)': tensor(79.94534),
            # 'a(IL6_STAT3)': tensor(83.39896),
            # 'a(NF_xB)': tensor(82.79433399999999),
            # 'a(IL6_AMP)': tensor(81.38015),
            # 'a(cytokine)': tensor(80.21895)
            'a(SARS_COV2)': tensor(61.631156999999995),
            'a(PRR)': tensor(87.76389),
            'a(ACE2)': tensor(39.719845),
            'a(AngII)': tensor(59.212959999999995),
            'a(AGTR1)': tensor(84.39899399999999),
            'a(ADAM17)': tensor(85.84442),
            'a(TOCI)': tensor(67.33063),
            'a(TNF)': tensor(77.83915),
            'a(sIL_6_alpha)': tensor(57.584044999999996),
            'a(EGF)': tensor(86.26822),
            'a(EGFR)': tensor(81.4849),
            'a(IL6_STAT3)': tensor(69.57323000000001),
            'a(NF_xB)': tensor(83.75941),
            'a(IL6_AMP)': tensor(77.52906),
            'a(cytokine)': tensor(79.07555)

        }
        target = "a(cytokine)"
        intervention_data = {
            "a(TOCI)": 0.0
        }

        causal_effects1, counterfactual_samples1 = scm.counterfactual_inference(condition_data, intervention_data,
                                                                                target, True)
        samples_df = pd.DataFrame(causal_effects1)
        samples_df.to_csv("../Tests/Data/causal_effect_bel2scm_hardcoded_sigmoid_moderately_ill.csv", index=False)


if __name__ == '__main__':
    unittest.main()

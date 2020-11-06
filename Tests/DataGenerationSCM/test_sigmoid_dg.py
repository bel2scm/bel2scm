from covid_sigmoid_scm_cf import SigmoidSCM
from covid_sigmoid_scm_cf import scm_covid_counterfactual
import pandas as pd
import torch
from torch import tensor
from pyro.distributions import Normal

torch.manual_seed(23)

def main():
    betas = {
        # 'PRR_SARS_2_w': 0.1,
        # 'PRR_b': -5.0,
        # 'ACE2_SARS_2_w': -0.2,
        # 'ACE2_b': 25.0,
        # 'AngII_ACE2_w': -0.45,
        # 'AngII_b': 15.0,
        # 'AGTR1_AngII_w': 0.3,
        # 'AGTR1_b': -2,
        # 'ADAM17_AGTR1_w': 0.1,
        # 'ADAM17_b': -6.0, #not repressing
        # 'EGF_ADAM17_w': 1.0,
        # 'EGF_b': -7.0,
        # 'TNF_ADAM17_w': 0.03,
        # 'TNF_b': -1.4,
        # 'sIL6_ADAM17_w': 0.02,
        # 'sIL6_TOCI_w': -0.04,
        # 'sIL6_b': 0.5,
        # 'EGFR_EGF_w': 1.0,
        # 'EGFR_b': -8.0,
        # 'IL6STAT3_sIL_6_alpha_w': 1.0,
        # 'IL6STAT3_b': -3.0,
        # 'NF_xB_PRR_w': 1.0,
        # 'NF_xB_TNF_w': 1.0,
        # 'NF_xB_EGFR_w': 1.0,
        # 'NF_xB_b': -8.0,
        # 'IL6_AMP_NF_xB_w': 1.0,
        # 'IL6_AMP_IL6_STAT3_w': 1.0,
        # 'IL6_AMP_b': -9.0,
        # 'cytokine_IL6_AMP_w': 1.0,
        # 'cytokine_b': -9.0

        'PRR_SARS_2_w' : 0.04,
        'PRR_b': -0.5,
        'ACE2_SARS_2_w': -1.0,
        'ACE2_b': 100.0,
        'AngII_ACE2_w': -1.0,
        'AngII_b': 100.0,
        'AGTR1_AngII_w': 0.04,
        'AGTR1_b': -0.5,
        'ADAM17_AGTR1_w': 0.03,
        'ADAM17_b': -0.8, #not repressing
        'EGF_ADAM17_w': 0.03,
        'EGF_b': -0.8,
        'TNF_ADAM17_w': 0.03,
        'TNF_b': -1.4,
        'sIL6_ADAM17_w': 0.05,
        'sIL6_TOCI_w': -0.06,
        'sIL6_b': 0.0,
        'EGFR_EGF_w': 0.03,
        'EGFR_b': -1.2,
        'IL6STAT3_sIL_6_alpha_w': 0.03,
        'IL6STAT3_b': -0.8,
        'NF_xB_PRR_w': 0.02,
        'NF_xB_TNF_w': 0.01,
        'NF_xB_EGFR_w': 0.01,
        'NF_xB_b': -1.9,
        'IL6_AMP_NF_xB_w': 0.02,
        'IL6_AMP_IL6_STAT3_w': 0.02,
        'IL6_AMP_b': -1.7,
        'cytokine_IL6_AMP_w': 0.03,
        'cytokine_b': -1.0


    }
    max_abundance = {
        'SARS_COV2': 100,
        'PRR': 100,
        'ACE2': 100,
        'AngII': 100,
        'AGTR1': 100,
        'ADAM17': 100,
        'IL_6Ralpha': 100,
        'TOCI': 100,
        'sIL6': 100,
        'STAT3': 100,
        'EGF': 100,
        'TNF': 100,
        'EGFR': 100,
        'IL6_STAT3': 100,
        'NF_xB': 100,
        'IL6_AMP': 100,
        'cytokine': 100
    }
    noise = {
        'N_SARS_COV2': (0., 1.),
        'N_TOCI': (0., 1.),
        'N_PRR': (0., 1.),
        'N_ACE2': (0., 1.),
        'N_AngII': (0., 1.),
        'N_AGTR1': (0., 1.),
        'N_ADAM17': (0., 1.),
        'N_IL_6Ralpha': (0., 1.),
        'N_sIL_6_alpha': (0., 1.),
        'N_STAT3': (0., 1.),
        'N_EGF': (0., 1.),
        'N_TNF': (0., 1.),
        'N_EGFR': (0., 1.),
        'N_IL6_STAT3': (0., 1.),
        'N_NF_xB': (0., 1.),
        'N_IL_6_AMP': (0., 1.),
        'N_cytokine': (0., 1.)
    }

    observation = {
        # 'SARS_COV2': 67.35032,
        # 'PRR': 89.7037,
        # 'ACE2': 29.747593,
        # 'AngII': 68.251114,
        # 'AGTR1': 90.96106999999999,
        # 'ADAM17': 86.84893000000001,
        # 'TOCI': 40.76684,
        # 'TNF': 76.85005,
        # 'sIL_6_alpha': 87.99491,
        # 'EGF': 84.55391,
        # 'EGFR': 79.94534,
        # 'IL6_STAT3': 83.39896,
        # 'NF_xB': 82.79433399999999,
        # 'IL6_AMP': 81.38015,
        # 'cytokine': 80.21895


        # moderately ill patient
        'SARS_COV2': 61.631156999999995,
        'PRR': 87.76389,
        'ACE2': 39.719845,
        'AngII': 59.212959999999995,
        'AGTR1': 84.39899399999999,
        'ADAM17': 85.84442,
        'TOCI': 67.33063,
        'TNF': 77.83915,
        'sIL_6_alpha': 57.584044999999996,
        'EGF': 86.26822,
        'EGFR': 81.4849,
        'IL6_STAT3': 69.57323000000001,
        'NF_xB': 83.75941,
        'IL6_AMP': 77.52906,
        'cytokine': 79.07555

    }

    intervention_data = {
        "TOCI": 0.0
    }
    ### get observational samples

    # covid_scm = SigmoidSCM(betas, max_abundance, 1.0)
    # noisy_samples = [covid_scm.noisy_model(noise) for _ in range(5000)]
    # samples_df = pd.DataFrame(noisy_samples)
    # samples_df.to_csv("/home/somya/bel2scm/Tests/Data/hardcoded_sigmoid_data.csv", index=False)

    ### get observational samples from mutilated graph by intervening on TOCI
    covid_scm = SigmoidSCM(betas, max_abundance, 1.0)
    noisy_samples = [covid_scm.noisy_mutilated_model(noise) for _ in range(5000)]
    samples_df = pd.DataFrame(noisy_samples)
    samples_df.to_csv("hardcoded_sigmoid_intervened_data.csv", index=False)


    ### calculate causal effect from direct simulation
    data = pd.read_csv("/home/somya/bel2scm/Tests/Data/observational_samples_from_sigmoid_known_parameters.csv")
    mutilated_scm = pd.read_csv("/home/somya/bel2scm/Tests/Data/observational_samples_from_intervened_sigmoid_with_known_parameters.csv")
    direct_causal_effect = pd.DataFrame()
    direct_causal_effect['causal_effect'] = data['a(cytokine)'] - mutilated_scm['a(cytokine)']
    direct_causal_effect.to_csv("/home/somya/bel2scm/Tests/Data/causal_effect_from_direct_simulation", index=False)


    ### perform causal effect
    # out = scm_covid_counterfactual(
    #     betas,
    #     max_abundance,
    #     observation,
    #     intervention_data,
    #     spike_width=1.0,
    #     svi=True
    # )
    # out_df = pd.DataFrame(out)
    # out_df.to_csv("/home/somya/bel2scm/Tests/Data/causal_effect_sigmoid_moderately_ill.csv", index=False)


main()
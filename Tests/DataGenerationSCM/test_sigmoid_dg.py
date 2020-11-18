import torch
from torch import tensor
import sys
import os
sys.path.append(os.path.realpath('..'))
from covid_sigmoid_scm_cf import SigmoidSCM
from covid_sigmoid_scm_cf import scm_covid_counterfactual
import pandas as pd
torch.manual_seed(23)


def main():
    betas = {
        'PRR_SARS_2_w': 0.04,
        'PRR_b': -0.5,
        'ACE2_SARS_2_w': -1.0,
        'ACE2_b': 100.0,
        'AngII_ACE2_w': -1.0,
        'AngII_b': 100.0,
        'AGTR1_AngII_w': 0.04,
        'AGTR1_b': -0.5,
        'ADAM17_AGTR1_w': 0.03,
        'ADAM17_b': -0.8,  # not repressing
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
    # we are not using this dictionary here. Instead this is to indicate that in mutilated
    # graph, TOCI is the variable getting intervened on with value 0.0

    intervention_data = {
        "TOCI": 0.0
    }
    ### get observational samples

    covid_scm = SigmoidSCM(betas, max_abundance, 1.0)
    noisy_samples = [covid_scm.noisy_model(noise) for _ in range(5000)]
    samples_df = pd.DataFrame(noisy_samples)
    samples_df.to_csv(
        "C:/Users/somya/Documents/GitHub/bel2scm/Tests/Data/observational_samples_from_sigmoid_known_parameters.csv",
        index=False)

    ### get observational samples from mutilated graph by intervening on TOCI
    covid_scm_mutilated = SigmoidSCM(betas, max_abundance, 1.0)
    noisy_samples_mutilated = [covid_scm_mutilated.noisy_mutilated_model(noise) for _ in range(5000)]
    samples_df_mutilated = pd.DataFrame(noisy_samples_mutilated)
    samples_df_mutilated.to_csv(
        "C:/Users/somya/Documents/GitHub/bel2scm/Tests/Data/observational_samples_from_intervened_sigmoid_with_known_parameters.csv",
        index=False)

    ### calculate causal effect from direct simulation
    direct_causal_effect = pd.DataFrame()
    direct_causal_effect['causal_effect'] = samples_df['a(cytokine)'] - samples_df_mutilated['a(cytokine)']
    direct_causal_effect.to_csv("C:/Users/somya/Documents/GitHub//bel2scm/Tests/Data/causal_effect_from_direct_simulation", index=False)


main()

from covid_sigmoid_scm_cf import SigmoidSCM
from covid_sigmoid_scm_cf import scm_covid_counterfactual
import pandas as pd
import torch
from pyro.distributions import Normal

torch.manual_seed(23)

def main():
    betas = {
        'PRR_SARS_2_w' : 1.0,
        'PRR_b': 0.0,
        'ACE2_SARS_2_w': -1.0,
        'ACE2_b': 0.0,
        'AngII_ACE2_w': -1.0,
        'AngII_b': 0.0,
        'AGTR1_AngII_w': 1.0,
        'AGTR1_b': 0.0,
        'ADAM17_AGTR1_w': 1.0,
        'ADAM17_b': 0.0,
        'EGF_ADAM17_w': 1.0,
        'EGF_b': 0.0,
        'TNF_ADAM17_w': 1.0,
        'TNF_b': 0.0,
        'sIL6_ADAM17_w': 1.0,
        'sIL6_TOCI_w': -1.0,
        'sIL6_b': 0.0,
        'EGFR_EGF_w': 1.0,
        'EGFR_b': 0.0,
        'IL6STAT3_sIL_6_alpha_w': 1.0,
        'IL6STAT3_b': 0.0,
        'NF_xB_PRR_w': 1.0,
        'NF_xB_TNF_w': 1.0,
        'NF_xB_EGFR_w': 1.0,
        'NF_xB_b': 0.0,
        'IL6_AMP_NF_xB_w': 1.0,
        'IL6_AMP_IL6_STAT3_w': 1.0,
        'IL6_AMP_b': 0.0,
        'cytokine_IL6_AMP_w': 1.0,
        'cytokine_b': 0.0

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
        'N_SARS_COV2': Normal(0., 5.),
        'N_TOCI': Normal(0., 5.),
        'N_PRR': Normal(0., 1.),
        'N_ACE2': Normal(0., 1.),
        'N_AngII': Normal(0., 1.),
        'N_AGTR1': Normal(0., 1.),
        'N_ADAM17': Normal(0., 1.),
        'N_IL_6Ralpha': Normal(0., 1.),
        'N_sIL_6_alpha': Normal(0., 1.),
        'N_STAT3': Normal(0., 1.),
        'N_EGF': Normal(0., 1.),
        'N_TNF': Normal(0., 1.),
        'N_EGFR': Normal(0., 1.),
        'N_IL6_STAT3': Normal(0., 1.),
        'N_NF_xB': Normal(0., 1.),
        'N_IL_6_AMP': Normal(0., 1.),
        'N_cytokine': Normal(0., 1.)
    }

    observation = {

    }

    intervention_data = {

    }

    covid_scm = SigmoidSCM(betas, max_abundance, 1.0)
    print(covid_scm.noisy_model(noise=noise))

    out = scm_covid_counterfactual(
        betas,
        max_abundance,
        observation,
        intervention_data,
        spike_width=1.0,
        svi=True
    )
    print(out)


main()
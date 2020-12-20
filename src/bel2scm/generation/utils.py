import os
from typing import Optional

import pandas as pd
from bel2scm.generation.covid_sigmoid_scm_cf import NOISE, SigmoidSCM, scm_covid_counterfactual
from tqdm import trange

__all__ = [
    'run',
]


def run(
    directory: str,
    n_noisy_samples: Optional[int] = None,
    n_noisy_samples_mutilated: Optional[int] = None,
) -> None:
    if n_noisy_samples is None:
        n_noisy_samples = 5000
    if n_noisy_samples_mutilated is None:
        n_noisy_samples_mutilated = 5000

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
        'cytokine_b': -1.0,
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
        'cytokine': 100,
    }

    observation_datapoint_1 = {
        'SARS_COV2': 67.35032,
        'PRR': 89.7037,
        'ACE2': 29.747593,
        'AngII': 68.251114,
        'AGTR1': 90.96106999999999,
        'ADAM17': 86.84893000000001,
        'TOCI': 40.76684,
        'TNF': 76.85005,
        'sIL_6_alpha': 87.99491,
        'EGF': 84.55391,
        'EGFR': 79.94534,
        'IL6_STAT3': 83.39896,
        'NF_xB': 82.79433399999999,
        'IL6_AMP': 81.38015,
        'cytokine': 80.21895,
    }

    observation_datapoint_2 = {
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
        'cytokine': 79.07555,
    }
    intervention_data = {
        "TOCI": 0.0,
    }
    # get observational samples

    covid_scm = SigmoidSCM(betas, max_abundance, 1.0)
    noisy_samples = [
        covid_scm.noisy_model(NOISE)
        for _ in trange(n_noisy_samples, desc='noisy samples')
    ]
    samples_df = pd.DataFrame([
        [e.item() for e in noisy_sample]
        for noisy_sample in noisy_samples
    ])
    samples_df.to_csv(
        os.path.join(directory, "observational_samples_from_sigmoid_known_parameters.csv"),
        index=False,
    )

    # get observational samples from mutilated graph by intervening on TOCI
    covid_scm_mutilated = SigmoidSCM(betas, max_abundance, 1.0)
    noisy_samples_mutilated = [
        covid_scm_mutilated.noisy_mutilated_model(NOISE)
        for _ in trange(n_noisy_samples_mutilated, desc='noisy mutilated samples')
    ]
    samples_df_mutilated = pd.DataFrame(noisy_samples_mutilated)
    samples_df_mutilated.to_csv(
        os.path.join(directory, "observational_samples_from_intervened_sigmoid_with_known_parameters.csv"),
        index=False,
    )

    ##################################################################################################
    # Comment the following out while running above functions
    # While running these two functions, comment out above functions
    # calculate causal effect from direct simulation

    direct_causal_effect_df = pd.DataFrame({
        'samples': samples_df[14],
        'samples_mutilated': samples_df_mutilated['a(cytokine)'],
    })
    # 14 is position of a(cytokine)
    direct_causal_effect_df['causal_effect'] = direct_causal_effect_df['samples'] - direct_causal_effect_df[
        'samples_mutilated']
    direct_causal_effect_df.to_csv(
        os.path.join(directory, "causal_effect_from_direct_simulation.csv"),
        index=False,
    )

    # get causal effect by conditioning on data point 1
    out_dp1 = scm_covid_counterfactual(
        betas,
        max_abundance,
        observation_datapoint_1,
        intervention_data,
        spike_width=1.0,
        svi=True,
    )
    out_df = pd.DataFrame(out_dp1)
    out_df.to_csv(
        os.path.join(directory, "causal_effect_sigmoid_with_known_parameters_datapoint1.csv"),
        index=False,
    )

    # get causal effect by conditioning on data point 2
    out_dp2 = scm_covid_counterfactual(
        betas,
        max_abundance,
        observation_datapoint_2,
        intervention_data,
        spike_width=1.0,
        svi=True,
    )
    out_df2 = pd.DataFrame(out_dp2)
    out_df2.to_csv(
        os.path.join(directory, "causal_effect_sigmoid_with_known_parameters_datapoint2.csv"),
        index=False,
        sep='\t',
    )

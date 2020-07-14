from covid_scm_cf import scm_covid_counterfactual
from covid_scm_cf import COVID_SCM
import pandas as pd
import torch
from pyro.distributions import Normal

torch.manual_seed(5)


def main():
    rates = {
        'PRR_activation_by_SARS_COV2': 0.05,
        'PRR_deactivation': 0.7,
        'ACE2_activation': 1,
        'ACE2_deactivation_by_SARS_COV2': 0.7,
        'AngII_activation': 1.5,
        'AngII_deactivation_by_ACE2': 0.7,
        'AGTR1_activation_by_AngII': 0.1,
        'AGTR1_deactivation': 1,
        'ADAM17_activation_by_AGTR1': 0.01,
        'ADAM17_deactivation': 1.5,
        'EGF_activation_by_ADAM17': 0.05,
        'EGF_deactivation': 1.5,
        'TNF_activation_by_ADAM17': 0.05,
        'TNF_deactivation': 1,
        'sIL_6_alpha_activation_by_ADAM17': 0.2,
        'sIL_6_alpha_deactivation_by_TOCI': 0.2,
        'IL_6Ralpha_activation': 1,
        'IL_6Ralpha_deactivation_by_TOCI': 0.05,
        'EGFR_activation_by_EGF': 0.05,
        'EGFR_deactivation': 2,
        'IL6_STAT3_activation_by_sIL_6_alpha': 0.05,
        'IL6_STAT3_deactivation': 1,
        'STAT3_deactivation_by_sIL_6_alpha': 0.05,
        'STAT3_activation': 1,
        'NF_xB_activation_by_PRR': 0.05,
        'NF_xB_activation_by_EGFR': 0.01,
        'NF_xB_activation_by_TNF': 0.01,
        'NF_xB_deactivation': 2,
        'IL_6_AMP_activation_by_NF_xB': 0.05,
        'IL_6_AMP_activation_by_IL6_STAT3': 0.05,
        'IL_6_AMP_deactivation': 2,
        'cytokine_activation_by_IL_6_AMP': 0.25,
        'cytokine_deactivation': 3,
    }

    totals = {
        'SARS_COV2': 100,
        'PRR': 100,
        'ACE2': 100,
        'AngII': 100,
        'AGTR1': 100,
        'ADAM17': 100,
        'IL_6Ralpha': 100,
        'TOCI': 100,
        'sIL_6_alpha': 100,
        'STAT3': 100,
        'EGF': 100,
        'TNF': 100,
        'EGFR': 100,
        'IL6_STAT3': 100,
        'NF_xB': 100,
        'IL_6_AMP': 100,
        'cytokine': 100
    }

    # observation = {
    #     'SARS_COV2': 69.717636,
    #     'PRR': 82.59810999999999,
    #     'ACE2': 48.58112,
    #     'AngII': 35.118378,
    #     'AGTR1': 70.559525,
    #     'ADAM17': 29.311705,
    #     'TOCI': 37.377205,
    #     # 'IL_6Ralpha': 31.568716,
    #     'TNF': 58.873653000000004,
    #     'sIL_6_alpha': 53.300537,
    #     'EGF': 54.108112,
    #     'EGFR': 65.97327,
    #     # 'STAT3': 39.057747,
    #     'IL6_STAT3': 69.78773000000001,
    #     'NF_xB': 72.42911,
    #     'IL6_AMP': 83.10982,
    #     'cytokine': 85.77821
    # }
    observation = {
        'SARS_COV2': 66.29791,
        'PRR': 82.59397,
        'ACE2': 49.372935999999996,
        'AngII': 36.845825,
        'AGTR1': 79.62148,
        'ADAM17': 37.830593,
        'TOCI': 46.12809,
        'TNF': 63.815290000000005,
        'sIL_6_alpha': 39.975414,
        'EGF': 48.7199,
        'EGFR': 56.71599200000001,
        'IL6_STAT3': 69.25169,
        'NF_xB': 70.1458,
        'IL6_AMP': 70.24296600000001,
        'cytokine': 87.00359
    }
    noise = {
        # 'N_SARS_COV2': Normal(0., 1.),
        # 'N_TOCI': Normal(0., 1.),
        # 'N_PRR': Normal(0., 1.),
        # 'N_ACE2': Normal(0., 1.),
        # 'N_AngII': Normal(0., 1.),
        # 'N_AGTR1': Normal(0., 1.),
        # 'N_ADAM17': Normal(0., 1.),
        # 'N_IL_6Ralpha': Normal(0., 1.),
        # 'N_sIL_6_alpha': Normal(0., 1.),
        # 'N_STAT3': Normal(0., 1.),
        # 'N_EGF': Normal(0., 1.),
        # 'N_TNF': Normal(0., 1.),
        # 'N_EGFR': Normal(0., 1.),
        # 'N_IL6_STAT3': Normal(0., 1.),
        # 'N_NF_xB': Normal(0., 1.),
        # 'N_IL_6_AMP': Normal(0., 1.),
        # 'N_cytokine': Normal(0., 1.)

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

    toci_intervention = {
        "TOCI": 0.0
    }
    out = scm_covid_counterfactual(
        rates,
        totals,
        observation,
        toci_intervention,
        spike_width=1.0,
        svi=True
    )
    out_df = pd.DataFrame(out)
    out_df.to_csv("/home/somya/bel2scm/Tests/Data/covid_data_toci0_noisy_regularized_dgscm.csv", index=False)
    # covid_scm = COVID_SCM(rates, totals, 1.0)
    # #
    # # noisy_samples = [covid_scm.noisy_model(noise) for _ in range(5000)]
    # # samples_df = pd.DataFrame(noisy_samples)
    # # samples_df.to_csv("/home/somya/bel2scm/Tests/Data/covid_noisy_reparameterized_data.csv", index=False)

    # direct_simulation_samples = [covid_scm.direct_simulation_model(noise, torch.tensor(0.)) for _ in range(5000)]
    # samples_df = pd.DataFrame(direct_simulation_samples)
    # samples_df.to_csv("/home/somya/bel2scm/Tests/Data/direct_simulation.csv", index=False)
    # samples = [covid_scm.model(noise) for _ in range(5000)]
    # samples_df = pd.DataFrame(samples)
    # samples_df.to_csv("/home/somya/bel2scm/Tests/Data/covid_reparameterized_data.csv", index=False)


main()

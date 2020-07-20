from covid_scm_cf import scm_covid_counterfactual
from covid_scm_cf import COVID_SCM
import pandas as pd
import torch
from pyro.distributions import Normal

torch.manual_seed(23)


def main():
    rates = {
        'PRR_activation_by_SARS_COV2': 0.05,
        'PRR_deactivation': 0.7,
        'ACE2_activation': 10,
        'ACE2_deactivation_by_SARS_COV2': 0.5,
        'AngII_activation': 1.5,
        'AngII_deactivation_by_ACE2': 0.7,
        'AGTR1_activation_by_AngII': 0.1,
        'AGTR1_deactivation': 1,
        'ADAM17_activation_by_AGTR1': 0.1,
        'ADAM17_deactivation': 5,
        'EGF_activation_by_ADAM17': 0.05,
        'EGF_deactivation': 1.5,
        'TNF_activation_by_ADAM17': 0.05,
        'TNF_deactivation': 1,
        'sIL_6_alpha_activation_by_ADAM17': 0.7,
        'sIL_6_alpha_deactivation_by_TOCI': 4,
        'EGFR_activation_by_EGF': 0.05,
        'EGFR_deactivation': 2,
        'IL6_STAT3_activation_by_sIL_6_alpha': 0.65,
        'IL6_STAT3_deactivation': 30,
        'NF_xB_activation_by_PRR': 0.05,
        'NF_xB_activation_by_EGFR': 0.01,
        'NF_xB_activation_by_TNF': 0.01,
        'NF_xB_deactivation': 2,

        'IL_6_AMP_activation_by_NF_xB': 0.5,
        'IL_6_AMP_activation_by_IL6_STAT3': 0.5,
        'IL_6_AMP_deactivation': 42,
        'cytokine_activation_by_IL_6_AMP': 0.5,
        'cytokine_deactivation': 20
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

    observation = {
        'SARS_COV2': 38.875343,
        'PRR': 76.03623,
        'ACE2': 29.578688,
        'AngII': 22.061018,
        'AGTR1': 77.65523,
        'ADAM17': 56.975002,
        'TOCI': 31.197278999999998,
        'TNF': 83.151764,
        'sIL_6_alpha': 21.572751999999998,
        'EGF': 59.84185,
        'EGFR': 59.106148,
        'IL6_STAT3': 30.252989000000003,
        'NF_xB': 82.60543,
        'IL6_AMP': 57.98129,
        'cytokine': 68.87068000000001
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

    toci_intervention = {
        "TOCI": 0.0,
        # "NF_xB": 0.0
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
    #
    # noisy_samples = [covid_scm.noisy_model(noise) for _ in range(5000)]
    # samples_df = pd.DataFrame(noisy_samples)
    # samples_df.to_csv("/home/somya/bel2scm/Tests/Data/covid_noisy_reparameterized_data.csv", index=False)

    # direct_simulation_samples = [covid_scm.direct_simulation_model(noise, torch.tensor(0.)) for _ in range(5000)]
    # samples_df = pd.DataFrame(direct_simulation_samples)
    # samples_df.to_csv("/home/somya/bel2scm/Tests/Data/direct_simulation.csv", index=False)
    # samples = [covid_scm.model(noise) for _ in range(5000)]
    # samples_df = pd.DataFrame(samples)
    # samples_df.to_csv("/home/somya/bel2scm/Tests/Data/covid_reparameterized_data.csv", index=False)


main()

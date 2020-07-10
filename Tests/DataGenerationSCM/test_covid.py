from covid_scm_cf import scm_covid_counterfactual
import pandas as pd
import torch
torch.manual_seed(23)

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

    observation = {
        'SARS_COV2': 94.1013,
        'PRR': 83.71568,
        'ACE2': 65.19312,
        'AngII': 46.015774,
        'AGTR1': 81.344444,
        'ADAM17': 39.398296,
        'TOCI': 49.86449,
        'IL_6Ralpha': 31.568716,
        'TNF': 60.439766000000006,
        'sIL_6_alpha': 41.084896,
        'EGF': 53.93261,
        'EGFR': 63.03896999999999,
        'STAT3': 39.057747,
        'IL6_STAT3': 60.946580000000004,
        'NF_xB': 69.66587,
        'IL6_AMP': 77.81179,
        'cytokine': 82.01133
    }

    toci_intervention = {
        'TOCI': 80.0
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
    out_df.to_csv("/home/somya/bel2scm/Tests/Data/covid_data_eq17_dgscm.csv", index=False)


main()

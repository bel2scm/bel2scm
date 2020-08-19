from collections import defaultdict
from functools import partial
from torch import tensor

import statistics
import pyro
from pyro import condition, do, sample
from pyro.optim import SGD
import torch.distributions.constraints as constraints

from pyro.distributions import Normal, Delta, Uniform
from pyro.infer import EmpiricalMarginal, Importance, SVI, Trace_ELBO


def g(a, b):
    """

    :param a: number of molecules activated by activator parent or auto activation rate of node
    :param b: number of molecules deactivated by repressor parent or auto deactivation rate of node
    :return: Michaelis menten function to get the total count of active molecules in steady state
    """
    return a / (a + b)


def f_PRR(SARS_2, N):
    p = g(
        rates['PRR_activation_by_SARS_COV2'] * SARS_2,
        rates['PRR_deactivation']
    )
    mu = totals['PRR'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    PRR = N * variance + mu
    return PRR


def f_ACE2(SARS_COV2, N):
    # p = rates['ACE2_activation']*totals['ACE2'] - rates['ACE2_deactivation_by_SARS_COV2'] * SARS_COV2
    mu = rates['ACE2_deactivation_by_SARS_COV2'] * SARS_COV2 + rates['ACE2_activation']
    # sigma = mu * (1. - p)
    sigma = 1.0
    variance = sigma ** 0.5
    ACE2 = N * variance + mu
    return ACE2


def f_AngII(ACE2, N):
    mu = rates['AngII_deactivation_by_ACE2'] * ACE2 + rates['AngII_activation']
    # sigma = mu * (1. - p)
    sigma = 1.0
    variance = sigma ** 0.5
    AngII = N * variance + mu
    return AngII


def f_AGTR1(AngII, N):
    p = g(
        rates['AGTR1_activation_by_AngII'] * AngII,
        rates['AGTR1_deactivation']
    )
    mu = totals['AGTR1'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    AGTR1 = N * variance + mu
    return AGTR1


def f_ADAM17(AGTR1, N):
    p = g(
        rates['ADAM17_activation_by_AGTR1'] * AGTR1,
        rates['ADAM17_deactivation']
    )
    mu = totals['ADAM17'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    ADAM17 = N * variance + mu
    return ADAM17


def f_EGF(ADAM17, N):
    p = g(
        rates['EGF_activation_by_ADAM17'] * ADAM17,
        rates['EGF_deactivation']
    )
    mu = totals['EGF'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    EGF = N * variance + mu
    return EGF


def f_TNF(ADAM17, N):
    p = g(
        rates['TNF_activation_by_ADAM17'] * ADAM17,
        rates['TNF_deactivation']
    )
    mu = totals['TNF'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    TNF = N * variance + mu
    return TNF


def f_sIL_6_alpha(ADAM17, TOCI, N):
    p = g(
        rates['sIL_6_alpha_activation_by_ADAM17'] * ADAM17,
        rates['sIL_6_alpha_deactivation_by_TOCI'] * TOCI
    )
    mu = totals['sIL_6_alpha'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    sIL_6_alpha = N * variance + mu
    return sIL_6_alpha


def f_IL_6Ralpha(TOCI, N):
    p = g(
        rates['IL_6Ralpha_activation'],
        rates['IL_6Ralpha_deactivation_by_TOCI'] * TOCI
    )
    mu = totals['IL_6Ralpha'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    IL_6Ralpha = N * variance + mu
    return IL_6Ralpha


def f_EGFR(EGF, N):
    p = g(
        rates['EGFR_activation_by_EGF'] * EGF,
        rates['EGFR_deactivation']
    )
    mu = totals['EGFR'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    EGFR = N * variance + mu
    return EGFR


def f_IL6_STAT3(sIL_6_alpha, N):
    p = g(
        rates['IL6_STAT3_activation_by_sIL_6_alpha'] * sIL_6_alpha,
        rates['IL6_STAT3_deactivation']
    )
    mu = totals['IL6_STAT3'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    IL6_STAT3 = N * variance + mu
    return IL6_STAT3


def f_STAT3(sIL_6_alpha, N):
    p = g(
        rates['STAT3_activation'],
        rates['STAT3_deactivation_by_sIL_6_alpha'] * sIL_6_alpha
    )
    mu = totals['STAT3'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    STAT3 = N * variance + mu
    return STAT3


def f_NF_xB(PRR, EGFR, TNF, N):
    p = g(
        rates['NF_xB_activation_by_PRR'] * PRR +
        rates['NF_xB_activation_by_EGFR'] * EGFR +
        rates['NF_xB_activation_by_TNF'] * TNF,
        rates['NF_xB_deactivation']
    )
    mu = totals['NF_xB'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    NF_xB = N * variance + mu
    return NF_xB


def f_IL_6_AMP(NF_xB, IL6_STAT3, N):
    p = g(
        rates['IL_6_AMP_activation_by_NF_xB'] * NF_xB +
        rates['IL_6_AMP_activation_by_IL6_STAT3'] * IL6_STAT3,
        rates['IL_6_AMP_deactivation']
    )
    mu = totals['IL_6_AMP'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    IL_6_AMP = N * variance + mu
    return IL_6_AMP


def f_cytokine(IL_6_AMP, N):
    p = g(
        rates['cytokine_activation_by_IL_6_AMP'] * IL_6_AMP,
        rates['cytokine_deactivation']
    )
    mu = totals['cytokine'] * p
    sigma = mu * (1. - p)
    variance = sigma ** 0.5
    cytokine = N * variance + mu
    return cytokine


def model(noise):
    samples = {}
    N_SARS_COV2 = sample('N_SARS_COV2', noise['N_SARS_COV2'])
    N_ACE2 = sample('N_ACE2', noise['N_ACE2'])
    N_PRR = sample('N_PRR', noise['N_PRR'])
    N_AngII = sample('N_AngII', noise['N_AngII'])
    N_AGTR1 = sample('N_AGTR1', noise['N_AGTR1'])
    N_ADAM17 = sample('N_ADAM17', noise['N_ADAM17'])
    N_IL_6Ralpha = sample('N_IL_6Ralpha', noise['N_IL_6Ralpha'])
    N_sIL_6_alpha = sample('N_sIL_6_alpha', noise['N_sIL_6_alpha'])
    N_TNF = sample('N_TNF', noise['N_TNF'])
    N_EGF = sample('N_EGF', noise['N_EGF'])
    N_EGFR = sample('N_EGFR', noise['N_EGFR'])
    N_IL6_STAT3 = sample('N_IL6_STAT3', noise['N_IL6_STAT3'])
    N_STAT3 = sample('N_STAT3', noise['N_STAT3'])
    N_NF_xB = sample('N_NF_xB', noise['N_NF_xB'])
    N_IL_6_AMP = sample('N_IL_6_AMP', noise['N_IL_6_AMP'])
    N_cytokine = sample('N_cytokine', noise['N_cytokine'])

    # SARS_COV2 = sample('SARS_COV2', Uniform(5, 100.))
    # TOCI = sample('TOCI', Uniform(5, 80.))
    SARS_COV2 = sample("SARS_COV2", Normal(65., 10))
    TOCI = sample("TOCI", Normal(50., 10))
    PRR = sample('PRR', Delta(f_PRR(SARS_COV2, N_PRR)))
    ACE2 = sample('ACE2', Delta(f_ACE2(SARS_COV2, N_ACE2)))
    AngII = sample('AngII', Delta(f_AngII(ACE2, N_AngII)))
    AGTR1 = sample('AGTR1', Delta(f_AGTR1(AngII, N_AGTR1)))
    ADAM17 = sample('ADAM17', Delta(f_ADAM17(AGTR1, N_ADAM17)))
    IL_6Ralpha = sample("IL_6Ralpha", Delta(f_IL_6Ralpha(TOCI, N_IL_6Ralpha)))
    TNF = sample('TNF', Delta(f_TNF(ADAM17, N_TNF)))
    sIL_6_alpha = sample('sIL_6_alpha', Delta(f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
    EGF = sample('EGF', Delta(f_EGF(ADAM17, N_EGF)))
    EGFR = sample('EGFR', Delta(f_EGFR(EGF, N_EGFR)))
    STAT3 = sample('STAT3', Delta(f_STAT3(sIL_6_alpha, N_STAT3)))
    NF_xB = sample('NF_xB', Delta(f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
    IL6_STAT3 = sample('IL6_STAT3', Delta(f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
    IL_6_AMP = sample('IL_6_AMP', Delta(f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
    cytokine = sample('ARDS', Delta(f_cytokine(IL_6_AMP, N_cytokine)))

    samples['a(SARS_COV2)'] = SARS_COV2.numpy()
    samples['a(PRR)'] = PRR.numpy()
    samples['a(ACE2)'] = ACE2.numpy()
    samples['a(AngII)'] = AngII.numpy()
    samples['a(AGTR1)'] = AGTR1.numpy()
    samples['a(ADAM17)'] = ADAM17.numpy()
    samples['a(TOCI)'] = TOCI.numpy()
    samples['a(IL_6Ralpha)'] = IL_6Ralpha.numpy()
    samples['a(TNF)'] = TNF.numpy()
    samples['a(sIL_6_alpha)'] = sIL_6_alpha.numpy()
    samples['a(EGF)'] = EGF.numpy()
    samples['a(EGFR)'] = EGFR.numpy()
    samples['a(STAT3)'] = STAT3.numpy()
    samples['a(IL6_STAT3)'] = IL6_STAT3.numpy()
    samples['a(NF_xB)'] = NF_xB.numpy()
    samples['a(IL6_AMP)'] = IL_6_AMP.numpy()
    samples['a(cytokine)'] = cytokine.numpy()

    noise_samples = N_PRR, N_ACE2, N_AngII, N_AGTR1, N_ADAM17, N_TNF, N_sIL_6_alpha, N_EGF, N_EGFR, N_NF_xB, N_IL6_STAT3, N_IL_6_AMP, N_cytokine
    # samples = SARS_COV2, PRR, deg, ACE2, AngII, AGTR1, ADAM17, TNF, sIL_6_alpha, EGF, EGFR, NF_xB, IL6_STAT3, IL_6_AMP, ARDS
    return samples

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


def main():
    import pandas as pd
    import torch
    torch.manual_seed(5)
    noise = {
        'N_SARS_COV2': Normal(0., 1.),
        'N_PRR': Normal(0., 1.),
        'N_ACE2': Normal(0., 1.),
        'N_AngII': Normal(0., 1.),
        'N_AGTR1': Normal(0., 1.),
        'N_ADAM17': Normal(0., 1.),
        'N_IL_6Ralpha': Normal(0., 1.),
        'N_TOCI': Normal(0., 1.),
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
    samples = [model(noise) for _ in range(5000)]
    # samples_df = pd.DataFrame(samples)
    samples_df = pd.DataFrame(samples)
    samples_df.to_csv("/home/somya/bel2scm/Tests/Data/covid_data.csv", index=False)


main()

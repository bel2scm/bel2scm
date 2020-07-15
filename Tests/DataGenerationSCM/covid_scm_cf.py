from collections import defaultdict
from functools import partial
from torch import tensor

import statistics
import pyro
from pyro import condition, do, sample
from pyro.optim import SGD
import torch.distributions.constraints as constraints

from pyro.distributions import Normal, Delta
from pyro.infer import EmpiricalMarginal, Importance, SVI, Trace_ELBO


def g(a, b):
    return a / (a + b)


class COVID_SCM():
    def __init__(self, rates, totals, spike_width):
        self.rates = rates
        self.totals = totals
        self.spike_width = spike_width

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
            if NF_xB < 50.0:
                NF_xB = 0.0
            if IL6_STAT3 < 50.0:
                IL6_STAT3 = 0.0
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

        def f_SARS_COV2(SARS_COV2_mu, SARS_COV2_sigma, N):
            SARS_COV2 = N*SARS_COV2_sigma + SARS_COV2_mu
            return SARS_COV2

        def f_TOCI(TOCI_mu, TOCI_sigma, N):
            TOCI = N*TOCI_sigma + TOCI_mu
            return TOCI

        def model(noise):
            N_SARS_COV2 = sample('N_SARS_COV2', Normal(noise['N_SARS_COV2'][0], noise['N_SARS_COV2'][1]))
            N_TOCI = sample('N_TOCI', Normal(noise['N_TOCI'][0], noise['N_TOCI'][1]))
            N_ACE2 = sample('N_ACE2', Normal(noise['N_ACE2'][0], noise['N_ACE2'][1]))
            N_PRR = sample('N_PRR', Normal(noise['N_PRR'][0], noise['N_PRR'][1]))
            N_AngII = sample('N_AngII', Normal(noise['N_AngII'][0], noise['N_AngII'][1]))
            N_AGTR1 = sample('N_AGTR1', Normal(noise['N_AGTR1'][0], noise['N_AGTR1'][1]))
            N_ADAM17 = sample('N_ADAM17',Normal(noise['N_ADAM17'][0], noise['N_ADAM17'][1]))
            # N_IL_6Ralpha = sample('N_IL_6Ralpha', noise['N_IL_6Ralpha'])
            N_sIL_6_alpha = sample('N_sIL_6_alpha', Normal(noise['N_sIL_6_alpha'][0], noise['N_sIL_6_alpha'][1]))
            N_TNF = sample('N_TNF', Normal(noise['N_TNF'][0], noise['N_TNF'][1]))
            N_EGF = sample('N_EGF', Normal(noise['N_EGF'][0], noise['N_EGF'][1]))
            N_EGFR = sample('N_EGFR', Normal(noise['N_EGFR'][0], noise['N_EGFR'][1]))
            N_IL6_STAT3 = sample('N_IL6_STAT3', Normal(noise['N_IL6_STAT3'][0], noise['N_IL6_STAT3'][1]))
            # N_STAT3 = sample('N_STAT3', noise['N_STAT3'])
            N_NF_xB = sample('N_NF_xB', Normal(noise['N_NF_xB'][0], noise['N_NF_xB'][1]))
            N_IL_6_AMP = sample('N_IL_6_AMP', Normal(noise['N_IL_6_AMP'][0], noise['N_IL_6_AMP'][1]))
            N_cytokine = sample('N_cytokine', Normal(noise['N_cytokine'][0], noise['N_cytokine'][1]))

            # SARS_COV2_mu = sample("SARS_COV2_mu", Normal(50., 10))
            SARS_COV2 = sample('SARS_COV2', Normal(f_SARS_COV2(50, 10, N_SARS_COV2), 1.0))
            # TOCI_mu = sample("TOCI_mu", Normal(50., 10))
            TOCI = sample('TOCI', Normal(f_TOCI(50, 10, N_TOCI), 1.0))
            # SARS_COV2 = sample("SARS_COV2", Normal(50., 10))
            # TOCI= sample("TOCI", Normal(50., 10))
            PRR = sample('PRR', Delta(f_PRR(SARS_COV2, N_PRR)))
            ACE2 = sample('ACE2', Delta(f_ACE2(SARS_COV2, N_ACE2)))
            AngII = sample('AngII', Delta(f_AngII(ACE2, N_AngII)))
            AGTR1 = sample('AGTR1', Delta(f_AGTR1(AngII, N_AGTR1)))
            ADAM17 = sample('ADAM1Spike7', Delta(f_ADAM17(AGTR1, N_ADAM17)))
            # IL_6Ralpha = sample("IL_6Ralpha", Delta(f_IL_6Ralpha(TOCI, N_IL_6Ralpha)))
            TNF = sample('TNF', Delta(f_TNF(ADAM17, N_TNF)))
            sIL_6_alpha = sample('sIL_6_alpha', Delta(f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
            EGF = sample('EGF', Delta(f_EGF(ADAM17, N_EGF)))
            EGFR = sample('EGFR', Delta(f_EGFR(EGF, N_EGFR)))
            # STAT3 = sample('STAT3', Delta(f_STAT3(sIL_6_alpha, N_STAT3)))
            NF_xB = sample('NF_xB', Delta(f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
            IL6_STAT3 = sample('IL6_STAT3', Delta(f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
            IL_6_AMP = sample('IL_6_AMP', Delta(f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
            cytokine = sample('cytokine', Delta(f_cytokine(IL_6_AMP, N_cytokine)))
            noise_samples = N_PRR, N_ACE2, N_AngII, N_AGTR1, N_ADAM17, N_TNF, N_sIL_6_alpha, \
                            N_EGF, N_EGFR, N_NF_xB, N_IL6_STAT3, N_IL_6_AMP, N_cytokine
            # samples = {'a(SARS_COV2)': SARS_COV2.numpy(), 'a(PRR)': PRR.numpy(), 'a(ACE2)': ACE2.numpy(),
            #            'a(AngII)': AngII.numpy(), 'a(AGTR1)': AGTR1.numpy(), 'a(ADAM17)': ADAM17.numpy(),
            #            'a(TOCI)': TOCI.numpy(), 'a(TNF)': TNF.numpy(), 'a(sIL_6_alpha)': sIL_6_alpha.numpy(),
            #            'a(EGF)': EGF.numpy(), 'a(EGFR)': EGFR.numpy(), 'a(IL6_STAT3)': IL6_STAT3.numpy(),
            #            'a(NF_xB)': NF_xB.numpy(), 'a(IL6_AMP)': IL_6_AMP.numpy(), 'a(cytokine)': cytokine.numpy()}
            samples = SARS_COV2, PRR, ACE2, AngII, AGTR1, ADAM17, TOCI, TNF, sIL_6_alpha, EGF, EGFR, NF_xB, \
                    IL6_STAT3, IL_6_AMP, cytokine
            return samples, noise_samples

        Spike = partial(Normal, scale=tensor(self.spike_width))

        def noisy_model(noise):
            N_SARS_COV2 = sample('N_SARS_COV2', Normal(noise['N_SARS_COV2'][0], noise['N_SARS_COV2'][1]))
            N_TOCI = sample('N_TOCI', Normal(noise['N_TOCI'][0], noise['N_TOCI'][1]))
            N_ACE2 = sample('N_ACE2', Normal(noise['N_ACE2'][0], noise['N_ACE2'][1]))
            N_PRR = sample('N_PRR', Normal(noise['N_PRR'][0], noise['N_PRR'][1]))
            N_AngII = sample('N_AngII', Normal(noise['N_AngII'][0], noise['N_AngII'][1]))
            N_AGTR1 = sample('N_AGTR1', Normal(noise['N_AGTR1'][0], noise['N_AGTR1'][1]))
            N_ADAM17 = sample('N_ADAM17',Normal(noise['N_ADAM17'][0], noise['N_ADAM17'][1]))
            # N_IL_6Ralpha = sample('N_IL_6Ralpha', noise['N_IL_6Ralpha'])
            N_sIL_6_alpha = sample('N_sIL_6_alpha', Normal(noise['N_sIL_6_alpha'][0], noise['N_sIL_6_alpha'][1]))
            N_TNF = sample('N_TNF', Normal(noise['N_TNF'][0], noise['N_TNF'][1]))
            N_EGF = sample('N_EGF', Normal(noise['N_EGF'][0], noise['N_EGF'][1]))
            N_EGFR = sample('N_EGFR', Normal(noise['N_EGFR'][0], noise['N_EGFR'][1]))
            N_IL6_STAT3 = sample('N_IL6_STAT3', Normal(noise['N_IL6_STAT3'][0], noise['N_IL6_STAT3'][1]))
            # N_STAT3 = sample('N_STAT3', noise['N_STAT3'])
            N_NF_xB = sample('N_NF_xB', Normal(noise['N_NF_xB'][0], noise['N_NF_xB'][1]))
            N_IL_6_AMP = sample('N_IL_6_AMP', Normal(noise['N_IL_6_AMP'][0], noise['N_IL_6_AMP'][1]))
            N_cytokine = sample('N_cytokine', Normal(noise['N_cytokine'][0], noise['N_cytokine'][1]))


            SARS_COV2 = sample('SARS_COV2', Normal(f_SARS_COV2(50, 10, N_SARS_COV2), 1.0))
            TOCI = sample('TOCI', Normal(f_TOCI(50, 10, N_TOCI), 1.0))
            # SARS_COV2 = sample("SARS_COV2", Normal(50., 10))
            # TOCI= sample("TOCI", Normal(50., 10))
            PRR = sample('PRR', Spike(f_PRR(SARS_COV2, N_PRR)))
            ACE2 = sample('ACE2', Spike(f_ACE2(SARS_COV2, N_ACE2)))
            AngII = sample('AngII', Spike(f_AngII(ACE2, N_AngII)))
            AGTR1 = sample('AGTR1', Spike(f_AGTR1(AngII, N_AGTR1)))
            ADAM17 = sample('ADAM17', Spike(f_ADAM17(AGTR1, N_ADAM17)))
            # IL_6Ralpha = sample("IL_6Ralpha", Spike(f_IL_6Ralpha(TOCI, N_IL_6Ralpha)))
            TNF = sample('TNF', Spike(f_TNF(ADAM17, N_TNF)))
            sIL_6_alpha = sample('sIL_6_alpha', Spike(f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
            EGF = sample('EGF', Spike(f_EGF(ADAM17, N_EGF)))
            EGFR = sample('EGFR', Spike(f_EGFR(EGF, N_EGFR)))
            # STAT3 = sample('STAT3', Spike(f_STAT3(sIL_6_alpha, N_STAT3)))
            NF_xB = sample('NF_xB', Spike(f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
            IL6_STAT3 = sample('IL6_STAT3', Spike(f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
            IL_6_AMP = sample('IL_6_AMP', Spike(f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
            cytokine = sample('cytokine', Spike(f_cytokine(IL_6_AMP, N_cytokine)))

            samples = SARS_COV2, PRR, ACE2, AngII, AGTR1, ADAM17, TOCI, TNF, sIL_6_alpha, EGF, EGFR, NF_xB, \
                      IL6_STAT3, IL_6_AMP, cytokine
            # samples = {'a(SARS_COV2)': SARS_COV2.numpy(), 'a(PRR)': PRR.numpy(), 'a(ACE2)': ACE2.numpy(),
            #            'a(AngII)': AngII.numpy(), 'a(AGTR1)': AGTR1.numpy(), 'a(ADAM17)': ADAM17.numpy(),
            #            'a(TOCI)': TOCI.numpy(), 'a(TNF)': TNF.numpy(), 'a(sIL_6_alpha)': sIL_6_alpha.numpy(),
            #            'a(EGF)': EGF.numpy(), 'a(EGFR)': EGFR.numpy(), 'a(IL6_STAT3)': IL6_STAT3.numpy(),
            #            'a(NF_xB)': NF_xB.numpy(), 'a(IL6_AMP)': IL_6_AMP.numpy(), 'a(cytokine)': cytokine.numpy()}
            return samples

        def direct_simulation_model(noise, toci_intervention):
            N_SARS_COV2 = sample('N_SARS_COV2', Normal(noise['N_SARS_COV2'][0], noise['N_SARS_COV2'][1]))
            N_TOCI = sample('N_TOCI', Normal(noise['N_TOCI'][0], noise['N_TOCI'][1]))
            N_ACE2 = sample('N_ACE2', Normal(noise['N_ACE2'][0], noise['N_ACE2'][1]))
            N_PRR = sample('N_PRR', Normal(noise['N_PRR'][0], noise['N_PRR'][1]))
            N_AngII = sample('N_AngII', Normal(noise['N_AngII'][0], noise['N_AngII'][1]))
            N_AGTR1 = sample('N_AGTR1', Normal(noise['N_AGTR1'][0], noise['N_AGTR1'][1]))
            N_ADAM17 = sample('N_ADAM17',Normal(noise['N_ADAM17'][0], noise['N_ADAM17'][1]))
            N_sIL_6_alpha = sample('N_sIL_6_alpha', Normal(noise['N_sIL_6_alpha'][0], noise['N_sIL_6_alpha'][1]))
            N_TNF = sample('N_TNF', Normal(noise['N_TNF'][0], noise['N_TNF'][1]))
            N_EGF = sample('N_EGF', Normal(noise['N_EGF'][0], noise['N_EGF'][1]))
            N_EGFR = sample('N_EGFR', Normal(noise['N_EGFR'][0], noise['N_EGFR'][1]))
            N_IL6_STAT3 = sample('N_IL6_STAT3', Normal(noise['N_IL6_STAT3'][0], noise['N_IL6_STAT3'][1]))
            N_NF_xB = sample('N_NF_xB', Normal(noise['N_NF_xB'][0], noise['N_NF_xB'][1]))
            N_IL_6_AMP = sample('N_IL_6_AMP', Normal(noise['N_IL_6_AMP'][0], noise['N_IL_6_AMP'][1]))
            N_cytokine = sample('N_cytokine', Normal(noise['N_cytokine'][0], noise['N_cytokine'][1]))

            SARS_COV2 = sample('SARS_COV2', Normal(f_SARS_COV2(50, 10, N_SARS_COV2), 1.0))
            TOCI = sample('TOCI', Normal(f_TOCI(50, 10, N_TOCI), 1.0))
            TOCI_prime = sample("TOCI_prime", Delta(toci_intervention))
            PRR = sample('PRR', Spike(f_PRR(SARS_COV2, N_PRR)))
            ACE2 = sample('ACE2', Spike(f_ACE2(SARS_COV2, N_ACE2)))
            AngII = sample('AngII', Spike(f_AngII(ACE2, N_AngII)))
            AGTR1 = sample('AGTR1', Spike(f_AGTR1(AngII, N_AGTR1)))
            ADAM17 = sample('ADAM17', Spike(f_ADAM17(AGTR1, N_ADAM17)))
            TNF = sample('TNF', Spike(f_TNF(ADAM17, N_TNF)))
            EGF = sample('EGF', Spike(f_EGF(ADAM17, N_EGF)))
            EGFR = sample('EGFR', Spike(f_EGFR(EGF, N_EGFR)))
            NF_xB = sample('NF_xB', Spike(f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
            sIL_6_alpha = sample('sIL_6_alpha', Spike(f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
            sIL_6_alpha_prime = sample('sIL_6_alpha_prime', Delta(f_sIL_6_alpha(ADAM17, TOCI_prime, N_sIL_6_alpha)))
            IL6_STAT3 = sample('IL6_STAT3', Spike(f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
            IL6_STAT3_prime = sample('IL6_STAT3_prime', Delta(f_IL6_STAT3(sIL_6_alpha_prime, N_IL6_STAT3)))
            IL_6_AMP = sample('IL_6_AMP', Spike(f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
            IL_6_AMP_prime = sample('IL_6_AMP_prime', Delta(f_IL_6_AMP(NF_xB, IL6_STAT3_prime, N_IL_6_AMP)))
            cytokine = sample('cytokine', Spike(f_cytokine(IL_6_AMP, N_cytokine)))
            cytokine_prime = sample('cytokine_prime', Delta(f_cytokine(IL_6_AMP_prime, N_cytokine)))

            # SARS_COV2 = sample('SARS_COV2', Normal(f_SARS_COV2(50, 10, N_SARS_COV2), 1.0))
            # TOCI = sample('TOCI', Normal(f_TOCI(50, 10, N_TOCI), 1.0))
            # TOCI_prime = sample("TOCI_prime", Delta(toci_intervention))
            # PRR = sample('PRR', Delta(f_PRR(SARS_COV2, N_PRR)))
            # ACE2 = sample('ACE2', Delta(f_ACE2(SARS_COV2, N_ACE2)))
            # AngII = sample('AngII', Delta(f_AngII(ACE2, N_AngII)))
            # AGTR1 = sample('AGTR1', Delta(f_AGTR1(AngII, N_AGTR1)))
            # ADAM17 = sample('ADAM17', Delta(f_ADAM17(AGTR1, N_ADAM17)))
            # TNF = sample('TNF', Delta(f_TNF(ADAM17, N_TNF)))
            # EGF = sample('EGF', Delta(f_EGF(ADAM17, N_EGF)))
            # EGFR = sample('EGFR', Delta(f_EGFR(EGF, N_EGFR)))
            # NF_xB = sample('NF_xB', Delta(f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
            # sIL_6_alpha = sample('sIL_6_alpha', Delta(f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
            # sIL_6_alpha_prime = sample('sIL_6_alpha_prime', Delta(f_sIL_6_alpha(ADAM17, TOCI_prime, N_sIL_6_alpha)))
            # IL6_STAT3 = sample('IL6_STAT3', Delta(f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
            # IL6_STAT3_prime = sample('IL6_STAT3_prime', Delta(f_IL6_STAT3(sIL_6_alpha_prime, N_IL6_STAT3)))
            # IL_6_AMP = sample('IL_6_AMP', Delta(f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
            # IL_6_AMP_prime = sample('IL_6_AMP_prime', Delta(f_IL_6_AMP(NF_xB, IL6_STAT3_prime, N_IL_6_AMP)))
            # cytokine = sample('cytokine', Delta(f_cytokine(IL_6_AMP, N_cytokine)))
            # cytokine_prime = sample('cytokine_prime', Delta(f_cytokine(IL_6_AMP_prime, N_cytokine)))
            causal_effect_on_cytokine = cytokine - cytokine_prime
            return causal_effect_on_cytokine

        self.model = model
        self.noisy_model = noisy_model
        self.direct_simulation_model = direct_simulation_model

    def infer(self, model, noise):
        return Importance(model, num_samples=1000).run(noise)

    def update_noise_svi(self, observed_steady_state, initial_noise):
        def guide(noise):
            noise_terms = list(noise.keys())
            mu_constraints = constraints.interval(-3., 3.)
            sigma_constraints = constraints.interval(.0001, 3)
            mu = {
                k: pyro.param(
                    '{}_mu'.format(k),
                    tensor(0.),
                    constraint=mu_constraints
                ) for k in noise_terms
            }
            sigma = {
                k: pyro.param(
                    '{}_sigma'.format(k),
                    tensor(1.),
                    constraint=sigma_constraints
                ) for k in noise_terms
            }
            for noise in noise_terms:
                sample(noise, Normal(mu[noise], sigma[noise]))

        observation_model = condition(self.noisy_model, observed_steady_state)
        pyro.clear_param_store()
        svi = SVI(
            model=observation_model,
            guide=guide,
            optim=SGD({"lr": 0.001, "momentum": 0.1}),
            loss=Trace_ELBO()
        )

        losses = []
        num_steps = 1000
        samples = defaultdict(list)
        for t in range(num_steps):
            losses.append(svi.step(initial_noise))
            for noise in initial_noise.keys():
                mu = '{}_mu'.format(noise)
                sigma = '{}_sigma'.format(noise)
                samples[mu].append(pyro.param(mu).item())
                samples[sigma].append(pyro.param(sigma).item())
        means = {k: statistics.mean(v) for k, v in samples.items()}
        updated_noise = {
            'N_SARS_COV2': (means['N_SARS_COV2_mu'], means['N_SARS_COV2_sigma']),
            'N_TOCI': (means['N_TOCI_mu'], means['N_TOCI_sigma']),
            'N_PRR': (means['N_PRR_mu'], means['N_PRR_sigma']),
            'N_ACE2': (means['N_ACE2_mu'], means['N_ACE2_sigma']),
            'N_TNF': (means['N_TNF_mu'], means['N_TNF_sigma']),
            'N_AngII': (means['N_AngII_mu'], means['N_AngII_sigma']),
            'N_AGTR1': (means['N_AGTR1_mu'], means['N_AGTR1_sigma']),
            'N_ADAM17': (means['N_ADAM17_mu'], means['N_ADAM17_sigma']),
            'N_IL_6Ralpha': (means['N_IL_6Ralpha_mu'], means['N_IL_6Ralpha_sigma']),
            'N_sIL_6_alpha': (means['N_sIL_6_alpha_mu'], means['N_sIL_6_alpha_sigma']),
            'N_STAT3': (means['N_STAT3_mu'], means['N_STAT3_sigma']),
            'N_EGF': (means['N_EGF_mu'], means['N_EGF_sigma']),
            'N_EGFR': (means['N_EGFR_mu'], means['N_EGFR_sigma']),
            'N_IL6_STAT3': (means['N_IL6_STAT3_mu'], means['N_IL6_STAT3_sigma']),
            'N_NF_xB': (means['N_NF_xB_mu'], means['N_NF_xB_sigma']),
            'N_IL_6_AMP': (means['N_IL_6_AMP_mu'], means['N_IL_6_AMP_sigma']),
            'N_cytokine': (means['N_cytokine_mu'], means['N_cytokine_sigma'])
        }

        return updated_noise, losses

    def update_noise_importance(self, observed_steady_state, initial_noise):
        observation_model = condition(self.noisy_model, observed_steady_state)
        posterior = self.infer(observation_model, initial_noise)
        updated_noise = {
            k: EmpiricalMarginal(posterior, sites=k)
            for k in initial_noise.keys()
        }
        return updated_noise


def scm_covid_counterfactual(
        rates,
        totals,
        observation,
        ras_intervention,
        spike_width=1.0,
        svi=True
):
    gf_scm = COVID_SCM(rates, totals, spike_width)
    noise = {
        'N_SARS_COV2': (0., 5.),
        'N_TOCI': (0., 5.),
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
    if svi:
        updated_noise, _ = gf_scm.update_noise_svi(observation, noise)
    else:
        updated_noise = gf_scm.update_noise_importance(observation, noise)
    counterfactual_model = do(gf_scm.model, ras_intervention)
    cf_posterior = gf_scm.infer(counterfactual_model, updated_noise)
    cf_cytokine_marginal = EmpiricalMarginal(cf_posterior, sites='cytokine')
    cf_il6amp_marginal = EmpiricalMarginal(cf_posterior, sites='IL_6_AMP')
    cf_nfxb_marginal = EmpiricalMarginal(cf_posterior, sites='NF_xB')
    cf_il6stat3_marginal = EmpiricalMarginal(cf_posterior, sites='IL6_STAT3')
    scm_causal_effect_samples = [
        observation['cytokine'] - float(cf_cytokine_marginal.sample())
        for _ in range(5000)
    ]
    il6amp_samples = cf_il6amp_marginal.sample((5000,)).tolist()
    nfxb_samples = cf_nfxb_marginal.sample((5000,)).tolist()
    il6stat3_samples = cf_il6stat3_marginal.sample((5000,)).tolist()
    return il6amp_samples, nfxb_samples, il6stat3_samples, scm_causal_effect_samples

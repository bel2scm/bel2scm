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

            SARS_COV2 = sample("SARS_COV2", Normal(65., 10))
            TOCI = sample("TOCI", Normal(50., 10))
            PRR = sample('PRR', Delta(f_PRR(SARS_COV2, N_PRR)))
            ACE2 = sample('ACE2', Delta(f_ACE2(SARS_COV2, N_ACE2)))
            AngII = sample('AngII', Delta(f_AngII(ACE2, N_AngII)))
            AGTR1 = sample('AGTR1', Delta(f_AGTR1(AngII, N_AGTR1)))
            ADAM17 = sample('ADAM1Spike7', Delta(f_ADAM17(AGTR1, N_ADAM17)))
            IL_6Ralpha = sample("IL_6Ralpha", Delta(f_IL_6Ralpha(TOCI, N_IL_6Ralpha)))
            TNF = sample('TNF', Delta(f_TNF(ADAM17, N_TNF)))
            sIL_6_alpha = sample('sIL_6_alpha', Delta(f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
            EGF = sample('EGF', Delta(f_EGF(ADAM17, N_EGF)))
            EGFR = sample('EGFR', Delta(f_EGFR(EGF, N_EGFR)))
            STAT3 = sample('STAT3', Delta(f_STAT3(sIL_6_alpha, N_STAT3)))
            NF_xB = sample('NF_xB', Delta(f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
            IL6_STAT3 = sample('IL6_STAT3', Delta(f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
            IL_6_AMP = sample('IL_6_AMP', Delta(f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
            cytokine = sample('cytokine', Delta(f_cytokine(IL_6_AMP, N_cytokine)))
            noise_samples = N_PRR, N_ACE2, N_AngII, N_AGTR1, N_ADAM17, N_TNF, N_sIL_6_alpha, N_EGF, N_EGFR, N_NF_xB, \
                            N_IL6_STAT3, N_IL_6_AMP, N_cytokine
            samples = SARS_COV2, PRR, ACE2, AngII, AGTR1, ADAM17, TNF, IL_6Ralpha, sIL_6_alpha, EGF, EGFR, NF_xB, \
                      STAT3, IL6_STAT3, IL_6_AMP, cytokine
            return samples, noise_samples

        Spike = partial(Normal, scale=tensor(spike_width))

        def noisy_model(noise):
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

            SARS_COV2 = sample("SARS_COV2", Normal(65., 10))
            TOCI = sample("TOCI", Normal(50., 10))
            PRR = sample('PRR', Spike(f_PRR(SARS_COV2, N_PRR)))
            ACE2 = sample('ACE2', Spike(f_ACE2(SARS_COV2, N_ACE2)))
            AngII = sample('AngII', Spike(f_AngII(ACE2, N_AngII)))
            AGTR1 = sample('AGTR1', Spike(f_AGTR1(AngII, N_AGTR1)))
            ADAM17 = sample('ADAM17', Spike(f_ADAM17(AGTR1, N_ADAM17)))
            IL_6Ralpha = sample("IL_6Ralpha", Spike(f_IL_6Ralpha(TOCI, N_IL_6Ralpha)))
            TNF = sample('TNF', Spike(f_TNF(ADAM17, N_TNF)))
            sIL_6_alpha = sample('sIL_6_alpha', Spike(f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
            EGF = sample('EGF', Spike(f_EGF(ADAM17, N_EGF)))
            EGFR = sample('EGFR', Spike(f_EGFR(EGF, N_EGFR)))
            STAT3 = sample('STAT3', Spike(f_STAT3(sIL_6_alpha, N_STAT3)))
            NF_xB = sample('NF_xB', Spike(f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
            IL6_STAT3 = sample('IL6_STAT3', Spike(f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
            IL_6_AMP = sample('IL_6_AMP', Spike(f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
            cytokine = sample('cytokine', Spike(f_cytokine(IL_6_AMP, N_cytokine)))
            noise_samples = N_PRR, N_ACE2, N_AngII, N_AGTR1, N_ADAM17, N_TNF, N_sIL_6_alpha, N_EGF, N_EGFR, N_NF_xB, \
                            N_IL6_STAT3, N_IL_6_AMP, N_cytokine
            samples = SARS_COV2, PRR, ACE2, AngII, AGTR1, ADAM17, TNF, IL_6Ralpha, sIL_6_alpha, EGF, EGFR, NF_xB, \
                      STAT3, IL6_STAT3, IL_6_AMP, cytokine
            return samples, noise_samples

        self.model = model
        self.noisy_model = noisy_model

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
        updated_noise = {'N_SARS_COV2': Normal(means['N_SARS_COV2_mu'], means['N_SARS_COV2_sigma']),
                         'N_PRR': Normal(means['N_PRR_mu'], means['N_PRR_sigma']),
                         'N_ACE2': Normal(means['N_ACE2_mu'], means['N_ACE2_sigma']),
                         'N_TNF': Normal(means['N_TNF_mu'], means['N_TNF_sigma']),
                         'N_AngII': Normal(means['N_AngII_mu'], means['N_AngII_sigma']),
                         'N_AGTR1': Normal(means['N_AGTR1_mu'], means['N_AGTR1_sigma']),
                         'N_ADAM17': Normal(means['N_ADAM17_mu'], means['N_ADAM17_sigma']),
                         'N_IL_6Ralpha': Normal(means['N_IL_6Ralpha_mu'], means['N_IL_6Ralpha_sigma']),
                         'N_TOCI': Normal(means['N_TOCI_mu'], means['N_TOCI_sigma']),
                         'N_sIL_6_alpha': Normal(means['N_sIL_6_alpha_mu'], means['N_sIL_6_alpha_sigma']),
                         'N_STAT3': Normal(means['N_STAT3_mu'], means['N_STAT3_sigma']),
                         'N_EGF': Normal(means['N_EGF_mu'], means['N_EGF_sigma']),
                         'N_EGFR': Normal(means['N_EGFR_mu'], means['N_EGFR_sigma']),
                         'N_IL6_STAT3': Normal(means['N_IL6_STAT3_mu'], means['N_IL6_STAT3_sigma']),
                         'N_NF_xB': Normal(means['N_NF_xB_mu'], means['N_NF_xB_sigma']),
                         'N_IL_6_AMP': Normal(means['N_IL_6_AMP_mu'], means['N_IL_6_AMP_sigma']),
                         'N_cytokine': Normal(means['N_cytokine_mu'], means['N_cytokine_sigma'])
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
    if svi:
        updated_noise, _ = gf_scm.update_noise_svi(observation, noise)
    else:
        updated_noise = gf_scm.update_noise_importance(observation, noise)
    counterfactual_model = do(gf_scm.model, ras_intervention)
    cf_posterior = gf_scm.infer(counterfactual_model, updated_noise)
    cf_cytokine_marginal = EmpiricalMarginal(cf_posterior, sites='cytokine')

    scm_causal_effect_samples = [
        observation['cytokine'] - float(cf_cytokine_marginal.sample())
        for _ in range(5000)
    ]
    return scm_causal_effect_samples

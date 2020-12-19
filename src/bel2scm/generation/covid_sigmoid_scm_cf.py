from collections import defaultdict
from functools import partial
from torch import tensor

import statistics
import pyro
from pyro import condition, do, sample
from torch.optim import SGD
import torch.distributions.constraints as constraints
import torch
from pyro.distributions import Normal, Delta
from pyro.infer import EmpiricalMarginal, Importance, SVI, Trace_ELBO
import numpy as np


def sigmoid(x):
    """
    sigmoid function
    """
    return 1 / (1 + np.exp(-x))


class SigmoidSCM():
    def __init__(self, betas, max_abundance, spike_width):
        # dictionary of w and b for each node
        self.betas = betas
        # dictionary for max abundance for each node
        self.max_abundance = max_abundance
        ## spike width
        self.spike_width = spike_width

        def f_PRR(SARS_2, N):
            """
            Calculate PRR using parent SARS_2
            """
            y = betas['PRR_SARS_2_w'] * SARS_2 + betas['PRR_b']
            PRR = max_abundance['ADAM17'] * sigmoid(y) + N
            return PRR

        def f_ACE2(SARS_2, N):
            y = betas['ACE2_SARS_2_w'] * SARS_2 + betas['ACE2_b']
            ACE2 = y + N
            return ACE2

        def f_AngII(ACE2, N):
            y = betas['AngII_ACE2_w'] * ACE2 + betas['AngII_b']
            AngII = y + N
            return AngII

        def f_AGTR1(AngII, N):
            y = betas['AGTR1_AngII_w'] * AngII + betas['AGTR1_b']
            AGTR1 = max_abundance['AGTR1'] * sigmoid(y) + N
            return AGTR1

        def f_ADAM17(AGTR1, N):
            y = betas['ADAM17_AGTR1_w'] * AGTR1 + betas['ADAM17_b']
            ADAM17 = max_abundance['ADAM17'] * sigmoid(y) + N
            return ADAM17

        def f_EGF(ADAM17, N):
            y = betas['EGF_ADAM17_w'] * ADAM17 + betas['EGF_b']
            EGF = max_abundance['EGF'] * sigmoid(y) + N
            return EGF

        def f_TNF(ADAM17, N):
            y = betas['TNF_ADAM17_w'] * ADAM17 + betas['TNF_b']
            TNF = max_abundance['TNF'] * sigmoid(y) + N
            return TNF

        def f_sIL_6_alpha(ADAM17, TOCI, N):
            y = betas['sIL6_ADAM17_w'] * ADAM17 + betas['sIL6_TOCI_w'] * TOCI + betas['sIL6_b']
            sIL6 = max_abundance['sIL6'] * sigmoid(y) + N
            return sIL6

        def f_EGFR(EGF, N):
            y = betas['EGFR_EGF_w'] * EGF + betas['EGFR_b']
            EGFR = max_abundance['EGFR'] * sigmoid(y) + N
            return EGFR

        def f_IL6_STAT3(sIL_6_alpha, N):
            y = betas['IL6STAT3_sIL_6_alpha_w'] * sIL_6_alpha + betas['IL6STAT3_b']
            IL6STAT3 = max_abundance['IL6_STAT3'] * sigmoid(y) + N
            return IL6STAT3

        def f_NF_xB(PRR, EGFR, TNF, N):
            y = betas['NF_xB_PRR_w'] * PRR + betas['NF_xB_TNF_w'] * TNF + \
                betas['NF_xB_EGFR_w'] * EGFR + betas['NF_xB_b']
            NF_xB = max_abundance['NF_xB'] * sigmoid(y) + N
            return NF_xB

        def f_IL_6_AMP(NF_xB, IL6_STAT3, N):
            y = betas['IL6_AMP_NF_xB_w'] * NF_xB + betas['IL6_AMP_IL6_STAT3_w'] * IL6_STAT3 + \
                + betas['IL6_AMP_b']
            IL6_AMP = max_abundance['IL6_AMP'] * sigmoid(y) + N
            return IL6_AMP

        def f_cytokine(IL6_AMP, N):
            if torch.is_tensor(IL6_AMP):
                IL6_AMP = IL6_AMP.detach().numpy()
            y = betas['cytokine_IL6_AMP_w'] * IL6_AMP + \
                + betas['cytokine_b']
            cytokine = max_abundance['cytokine'] * sigmoid(y) + N
            return cytokine

        def f_SARS_COV2(SARS_COV2_mu, SARS_COV2_sigma, N):
            SARS_COV2 = N * SARS_COV2_sigma + SARS_COV2_mu
            return SARS_COV2

        def f_TOCI(TOCI_mu, TOCI_sigma, N):
            TOCI = N * TOCI_sigma + TOCI_mu
            return TOCI

        def model(noise):
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

            PRR = sample('PRR', Delta(f_PRR(SARS_COV2, N_PRR)))
            ACE2 = sample('ACE2', Delta(f_ACE2(SARS_COV2, N_ACE2)))
            AngII = sample('AngII', Delta(f_AngII(ACE2, N_AngII)))
            AGTR1 = sample('AGTR1', Delta(f_AGTR1(AngII, N_AGTR1)))
            ADAM17 = sample('ADAM1Spike7', Delta(f_ADAM17(AGTR1, N_ADAM17)))
            TNF = sample('TNF', Delta(f_TNF(ADAM17, N_TNF)))
            sIL_6_alpha = sample('sIL_6_alpha', Delta(f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
            EGF = sample('EGF', Delta(f_EGF(ADAM17, N_EGF)))
            EGFR = sample('EGFR', Delta(f_EGFR(EGF, N_EGFR)))
            NF_xB = sample('NF_xB', Delta(f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
            IL6_STAT3 = sample('IL6_STAT3', Delta(f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
            IL_6_AMP = sample('IL_6_AMP', Delta(f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
            cytokine = sample('cytokine', Delta(f_cytokine(IL_6_AMP, N_cytokine)))
            noise_samples = N_PRR, N_ACE2, N_AngII, N_AGTR1, N_ADAM17, N_TNF, N_sIL_6_alpha, \
                            N_EGF, N_EGFR, N_NF_xB, N_IL6_STAT3, N_IL_6_AMP, N_cytokine
            ## Use the dictionary structure for generating observational dataset
            ## comment the variable list for that
            # samples = {'a(SARS_COV2)': SARS_COV2.numpy(), 'a(PRR)': PRR.numpy(), 'a(ACE2)': ACE2.numpy(),
            #            'a(AngII)': AngII.numpy(), 'a(AGTR1)': AGTR1.numpy(), 'a(ADAM17)': ADAM17.numpy(),
            #            'a(TOCI)': TOCI.numpy(), 'a(TNF)': TNF.numpy(), 'a(sIL_6_alpha)': sIL_6_alpha.numpy(),
            #            'a(EGF)': EGF.numpy(), 'a(EGFR)': EGFR.numpy(), 'a(IL6_STAT3)': IL6_STAT3.numpy(),
            #            'a(NF_xB)': NF_xB.numpy(), 'a(IL6_AMP)': IL_6_AMP.numpy(), 'a(cytokine)': cytokine.numpy()}

            ## Use the variable list for generating samples for causal effect
            ## comment the dictionary for that
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
            PRR = sample('PRR', Spike(f_PRR(SARS_COV2, N_PRR)))
            ACE2 = sample('ACE2', Spike(f_ACE2(SARS_COV2, N_ACE2)))
            AngII = sample('AngII', Spike(f_AngII(ACE2, N_AngII)))
            AGTR1 = sample('AGTR1', Spike(f_AGTR1(AngII, N_AGTR1)))
            ADAM17 = sample('ADAM17', Spike(f_ADAM17(AGTR1, N_ADAM17)))
            TNF = sample('TNF', Spike(f_TNF(ADAM17, N_TNF)))
            sIL_6_alpha = sample('sIL_6_alpha', Spike(f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
            EGF = sample('EGF', Spike(f_EGF(ADAM17, N_EGF)))
            EGFR = sample('EGFR', Spike(f_EGFR(EGF, N_EGFR)))
            # STAT3 = sample('STAT3', Spike(f_STAT3(sIL_6_alpha, N_STAT3)))
            NF_xB = sample('NF_xB', Spike(f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
            IL6_STAT3 = sample('IL6_STAT3', Spike(f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
            IL_6_AMP = sample('IL_6_AMP', Spike(f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
            cytokine = sample('cytokine', Spike(f_cytokine(IL_6_AMP, N_cytokine)))

            ## Use the dictionary structure for generating observational dataset
            ## comment the variable list for that
            # samples = {'a(SARS_COV2)': SARS_COV2.numpy(), 'a(PRR)': PRR.numpy(), 'a(ACE2)': ACE2.numpy(),
            #            'a(AngII)': AngII.numpy(), 'a(AGTR1)': AGTR1.numpy(), 'a(ADAM17)': ADAM17.numpy(),
            #            'a(TOCI)': TOCI.numpy(), 'a(TNF)': TNF.numpy(), 'a(sIL_6_alpha)': sIL_6_alpha.numpy(),
            #            'a(EGF)': EGF.numpy(), 'a(EGFR)': EGFR.numpy(), 'a(IL6_STAT3)': IL6_STAT3.numpy(),
            #            'a(NF_xB)': NF_xB.numpy(), 'a(IL6_AMP)': IL_6_AMP.numpy(), 'a(cytokine)': cytokine.numpy()}

            ## Use the variable list for generating samples for causal effect
            ## comment the dictionary for that
            samples = SARS_COV2, PRR, ACE2, AngII, AGTR1, ADAM17, TOCI, TNF, sIL_6_alpha, EGF, EGFR, NF_xB, \
                    IL6_STAT3, IL_6_AMP, cytokine
            return samples

        def noisy_mutilated_model(noise):
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
            TOCI = sample('TOCI', Delta(tensor(0.0)))
            PRR = sample('PRR', Spike(f_PRR(SARS_COV2, N_PRR)))
            ACE2 = sample('ACE2', Spike(f_ACE2(SARS_COV2, N_ACE2)))
            AngII = sample('AngII', Spike(f_AngII(ACE2, N_AngII)))
            AGTR1 = sample('AGTR1', Spike(f_AGTR1(AngII, N_AGTR1)))
            ADAM17 = sample('ADAM17', Spike(f_ADAM17(AGTR1, N_ADAM17)))
            TNF = sample('TNF', Spike(f_TNF(ADAM17, N_TNF)))
            sIL_6_alpha = sample('sIL_6_alpha', Spike(f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
            EGF = sample('EGF', Spike(f_EGF(ADAM17, N_EGF)))
            EGFR = sample('EGFR', Spike(f_EGFR(EGF, N_EGFR)))
            NF_xB = sample('NF_xB', Spike(f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
            IL6_STAT3 = sample('IL6_STAT3', Spike(f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
            IL_6_AMP = sample('IL_6_AMP', Spike(f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
            cytokine = sample('cytokine', Spike(f_cytokine(IL_6_AMP, N_cytokine)))

            samples = {'a(SARS_COV2)': SARS_COV2.numpy(), 'a(PRR)': PRR.numpy(), 'a(ACE2)': ACE2.numpy(),
                       'a(AngII)': AngII.numpy(), 'a(AGTR1)': AGTR1.numpy(), 'a(ADAM17)': ADAM17.numpy(),
                       'a(TOCI)': TOCI.numpy(), 'a(TNF)': TNF.numpy(), 'a(sIL_6_alpha)': sIL_6_alpha.numpy(),
                       'a(EGF)': EGF.numpy(), 'a(EGFR)': EGFR.numpy(), 'a(IL6_STAT3)': IL6_STAT3.numpy(),
                       'a(NF_xB)': NF_xB.numpy(), 'a(IL6_AMP)': IL_6_AMP.numpy(), 'a(cytokine)': cytokine.numpy()}
            return samples


        self.model = model
        self.noisy_model = noisy_model
        self.noisy_mutilated_model = noisy_mutilated_model

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
        betas,
        max_abundance,
        observation,
        ras_intervention,
        spike_width=1.0,
        svi=True
):
    gf_scm = SigmoidSCM(betas, max_abundance, spike_width)
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
    if svi:
        updated_noise, _ = gf_scm.update_noise_svi(observation, noise)
    else:
        updated_noise = gf_scm.update_noise_importance(observation, noise)
    counterfactual_model = do(gf_scm.model, ras_intervention)
    cf_posterior = gf_scm.infer(counterfactual_model, updated_noise)
    cf_cytokine_marginal = EmpiricalMarginal(cf_posterior, sites=['cytokine'])
    scm_causal_effect_samples = [
        observation['cytokine'] - float(cf_cytokine_marginal.sample())
        for _ in range(5000)
    ]
    return scm_causal_effect_samples

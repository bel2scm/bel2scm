"""
Original Author: Somya
"""

import statistics
from collections import defaultdict
from typing import Any, List, Mapping, Optional, Type

import pyro
import torch
import torch.distributions.constraints as constraints
from pyro import condition, do, sample
from pyro.distributions import Delta, Normal
from pyro.infer import EmpiricalMarginal, Importance, SVI, Trace_ELBO
from pyro.optim import SGD
from scipy.special import expit
from torch import tensor
from torch.optim import Optimizer
from tqdm import trange

__all__ = [
    'SigmoidSCM',
    'scm_covid_counterfactual',
]

NOISE_TYPE_SAMPLES = 'samples'
NOISE_TYPE_OBSERVATIONAL = 'observational'


class InvalidNoiseType(ValueError):
    pass


class SigmoidSCM:
    def __init__(self, betas, max_abundance, spike_width, noise_type: Optional[str] = None):
        # dictionary of w and b for each node
        self.betas = betas
        # dictionary for max abundance for each node
        self.max_abundance = max_abundance
        ## spike width
        self.spike_width = spike_width

        if noise_type is None:
            noise_type = NOISE_TYPE_SAMPLES
        if noise_type not in {NOISE_TYPE_OBSERVATIONAL, NOISE_TYPE_SAMPLES}:
            raise InvalidNoiseType(noise_type)
        self.noise_type = noise_type

    def f_PRR(self, SARS_2, N):
        """
        Calculate PRR using parent SARS_2
        """
        y = self.betas['PRR_SARS_2_w'] * SARS_2 + self.betas['PRR_b']
        PRR = self.max_abundance['ADAM17'] * expit(y) + N
        return PRR

    def f_ACE2(self, SARS_2, N):
        y = self.betas['ACE2_SARS_2_w'] * SARS_2 + self.betas['ACE2_b']
        ACE2 = y + N
        return ACE2

    def f_AngII(self, ACE2, N):
        y = self.betas['AngII_ACE2_w'] * ACE2 + self.betas['AngII_b']
        AngII = y + N
        return AngII

    def f_AGTR1(self, AngII, N):
        y = self.betas['AGTR1_AngII_w'] * AngII + self.betas['AGTR1_b']
        AGTR1 = self.max_abundance['AGTR1'] * expit(y) + N
        return AGTR1

    def f_ADAM17(self, AGTR1, N):
        y = self.betas['ADAM17_AGTR1_w'] * AGTR1 + self.betas['ADAM17_b']
        ADAM17 = self.max_abundance['ADAM17'] * expit(y) + N
        return ADAM17

    def f_EGF(self, ADAM17, N):
        y = self.betas['EGF_ADAM17_w'] * ADAM17 + self.betas['EGF_b']
        EGF = self.max_abundance['EGF'] * expit(y) + N
        return EGF

    def f_TNF(self, ADAM17, N):
        y = self.betas['TNF_ADAM17_w'] * ADAM17 + self.betas['TNF_b']
        TNF = self.max_abundance['TNF'] * expit(y) + N
        return TNF

    def f_sIL_6_alpha(self, ADAM17, TOCI, N):
        y = self.betas['sIL6_ADAM17_w'] * ADAM17 + self.betas['sIL6_TOCI_w'] * TOCI + self.betas['sIL6_b']
        sIL6 = self.max_abundance['sIL6'] * expit(y) + N
        return sIL6

    def f_EGFR(self, EGF, N):
        y = self.betas['EGFR_EGF_w'] * EGF + self.betas['EGFR_b']
        EGFR = self.max_abundance['EGFR'] * expit(y) + N
        return EGFR

    def f_IL6_STAT3(self, sIL_6_alpha, N):
        y = self.betas['IL6STAT3_sIL_6_alpha_w'] * sIL_6_alpha + self.betas['IL6STAT3_b']
        IL6STAT3 = self.max_abundance['IL6_STAT3'] * expit(y) + N
        return IL6STAT3

    def f_NF_xB(self, PRR, EGFR, TNF, N):
        y = self.betas['NF_xB_PRR_w'] * PRR + self.betas['NF_xB_TNF_w'] * TNF + \
            self.betas['NF_xB_EGFR_w'] * EGFR + self.betas['NF_xB_b']
        NF_xB = self.max_abundance['NF_xB'] * expit(y) + N
        return NF_xB

    def f_IL_6_AMP(self, NF_xB, IL6_STAT3, N):
        y = self.betas['IL6_AMP_NF_xB_w'] * NF_xB + self.betas['IL6_AMP_IL6_STAT3_w'] * IL6_STAT3 + \
            + self.betas['IL6_AMP_b']
        IL6_AMP = self.max_abundance['IL6_AMP'] * expit(y) + N
        return IL6_AMP

    def f_cytokine(self, IL6_AMP, N):
        if torch.is_tensor(IL6_AMP):
            IL6_AMP = IL6_AMP.detach().numpy()
        y = self.betas['cytokine_IL6_AMP_w'] * IL6_AMP + \
            + self.betas['cytokine_b']
        cytokine = self.max_abundance['cytokine'] * expit(y) + N
        return cytokine

    @staticmethod
    def f_SARS_COV2(SARS_COV2_mu, SARS_COV2_sigma, N):
        SARS_COV2 = N * SARS_COV2_sigma + SARS_COV2_mu
        return SARS_COV2

    @staticmethod
    def f_TOCI(TOCI_mu, TOCI_sigma, N):
        TOCI = N * TOCI_sigma + TOCI_mu
        return TOCI

    def model(self, noise):
        N_SARS_COV2 = sample('N_SARS_COV2', Normal(noise['N_SARS_COV2'][0], noise['N_SARS_COV2'][1]))
        N_TOCI = sample('N_TOCI', Normal(noise['N_TOCI'][0], noise['N_TOCI'][1]))
        N_ACE2 = sample('N_ACE2', Normal(noise['N_ACE2'][0], noise['N_ACE2'][1]))
        N_PRR = sample('N_PRR', Normal(noise['N_PRR'][0], noise['N_PRR'][1]))
        N_AngII = sample('N_AngII', Normal(noise['N_AngII'][0], noise['N_AngII'][1]))
        N_AGTR1 = sample('N_AGTR1', Normal(noise['N_AGTR1'][0], noise['N_AGTR1'][1]))
        N_ADAM17 = sample('N_ADAM17', Normal(noise['N_ADAM17'][0], noise['N_ADAM17'][1]))
        N_sIL_6_alpha = sample('N_sIL_6_alpha', Normal(noise['N_sIL_6_alpha'][0], noise['N_sIL_6_alpha'][1]))
        N_TNF = sample('N_TNF', Normal(noise['N_TNF'][0], noise['N_TNF'][1]))
        N_EGF = sample('N_EGF', Normal(noise['N_EGF'][0], noise['N_EGF'][1]))
        N_EGFR = sample('N_EGFR', Normal(noise['N_EGFR'][0], noise['N_EGFR'][1]))
        N_IL6_STAT3 = sample('N_IL6_STAT3', Normal(noise['N_IL6_STAT3'][0], noise['N_IL6_STAT3'][1]))
        N_NF_xB = sample('N_NF_xB', Normal(noise['N_NF_xB'][0], noise['N_NF_xB'][1]))
        N_IL_6_AMP = sample('N_IL_6_AMP', Normal(noise['N_IL_6_AMP'][0], noise['N_IL_6_AMP'][1]))
        N_cytokine = sample('N_cytokine', Normal(noise['N_cytokine'][0], noise['N_cytokine'][1]))

        SARS_COV2 = sample('SARS_COV2', Normal(self.f_SARS_COV2(50, 10, N_SARS_COV2), 1.0))
        TOCI = sample('TOCI', Normal(self.f_TOCI(50, 10, N_TOCI), 1.0))

        PRR = sample('PRR', Delta(self.f_PRR(SARS_COV2, N_PRR)))
        ACE2 = sample('ACE2', Delta(self.f_ACE2(SARS_COV2, N_ACE2)))
        AngII = sample('AngII', Delta(self.f_AngII(ACE2, N_AngII)))
        AGTR1 = sample('AGTR1', Delta(self.f_AGTR1(AngII, N_AGTR1)))
        ADAM17 = sample('ADAM1Spike7', Delta(self.f_ADAM17(AGTR1, N_ADAM17)))
        TNF = sample('TNF', Delta(self.f_TNF(ADAM17, N_TNF)))
        sIL_6_alpha = sample('sIL_6_alpha', Delta(self.f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
        EGF = sample('EGF', Delta(self.f_EGF(ADAM17, N_EGF)))
        EGFR = sample('EGFR', Delta(self.f_EGFR(EGF, N_EGFR)))
        NF_xB = sample('NF_xB', Delta(self.f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
        IL6_STAT3 = sample('IL6_STAT3', Delta(self.f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
        IL_6_AMP = sample('IL_6_AMP', Delta(self.f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
        cytokine = sample('cytokine', Delta(self.f_cytokine(IL_6_AMP, N_cytokine)))

        noise_samples = (
            N_PRR,
            N_ACE2,
            N_AngII,
            N_AGTR1,
            N_ADAM17,
            N_TNF,
            N_sIL_6_alpha,
            N_EGF,
            N_EGFR,
            N_NF_xB,
            N_IL6_STAT3,
            N_IL_6_AMP,
            N_cytokine,
        )

        if self.noise_type == NOISE_TYPE_OBSERVATIONAL:
            # Use the dictionary structure for generating observational dataset
            samples = {
                'a(SARS_COV2)': SARS_COV2.numpy(),
                'a(PRR)': PRR.numpy(),
                'a(ACE2)': ACE2.numpy(),
                'a(AngII)': AngII.numpy(),
                'a(AGTR1)': AGTR1.numpy(),
                'a(ADAM17)': ADAM17.numpy(),
                'a(TOCI)': TOCI.numpy(),
                'a(TNF)': TNF.numpy(),
                'a(sIL_6_alpha)': sIL_6_alpha.numpy(),
                'a(EGF)': EGF.numpy(),
                'a(EGFR)': EGFR.numpy(),
                'a(IL6_STAT3)': IL6_STAT3.numpy(),
                'a(NF_xB)': NF_xB.numpy(),
                'a(IL6_AMP)': IL_6_AMP.numpy(),
                'a(cytokine)': cytokine.numpy(),
            }
        elif self.noise_type == NOISE_TYPE_SAMPLES:
            ## Use the variable list for generating samples for causal effect
            samples = (
                SARS_COV2,
                PRR,
                ACE2,
                AngII,
                AGTR1,
                ADAM17,
                TOCI,
                TNF,
                sIL_6_alpha,
                EGF,
                EGFR,
                NF_xB,
                IL6_STAT3,
                IL_6_AMP,
                cytokine,
            )
        else:
            raise InvalidNoiseType(self.noise_type)

        return samples, noise_samples

    def Spike(self, loc):
        return Normal(loc=loc, scale=tensor(self.spike_width))

    def noisy_model(self, noise):
        N_SARS_COV2 = sample('N_SARS_COV2', Normal(noise['N_SARS_COV2'][0], noise['N_SARS_COV2'][1]))
        N_TOCI = sample('N_TOCI', Normal(noise['N_TOCI'][0], noise['N_TOCI'][1]))
        N_ACE2 = sample('N_ACE2', Normal(noise['N_ACE2'][0], noise['N_ACE2'][1]))
        N_PRR = sample('N_PRR', Normal(noise['N_PRR'][0], noise['N_PRR'][1]))
        N_AngII = sample('N_AngII', Normal(noise['N_AngII'][0], noise['N_AngII'][1]))
        N_AGTR1 = sample('N_AGTR1', Normal(noise['N_AGTR1'][0], noise['N_AGTR1'][1]))
        N_ADAM17 = sample('N_ADAM17', Normal(noise['N_ADAM17'][0], noise['N_ADAM17'][1]))
        N_sIL_6_alpha = sample('N_sIL_6_alpha', Normal(noise['N_sIL_6_alpha'][0], noise['N_sIL_6_alpha'][1]))
        N_TNF = sample('N_TNF', Normal(noise['N_TNF'][0], noise['N_TNF'][1]))
        N_EGF = sample('N_EGF', Normal(noise['N_EGF'][0], noise['N_EGF'][1]))
        N_EGFR = sample('N_EGFR', Normal(noise['N_EGFR'][0], noise['N_EGFR'][1]))
        N_IL6_STAT3 = sample('N_IL6_STAT3', Normal(noise['N_IL6_STAT3'][0], noise['N_IL6_STAT3'][1]))
        N_NF_xB = sample('N_NF_xB', Normal(noise['N_NF_xB'][0], noise['N_NF_xB'][1]))
        N_IL_6_AMP = sample('N_IL_6_AMP', Normal(noise['N_IL_6_AMP'][0], noise['N_IL_6_AMP'][1]))
        N_cytokine = sample('N_cytokine', Normal(noise['N_cytokine'][0], noise['N_cytokine'][1]))

        SARS_COV2 = sample('SARS_COV2', Normal(self.f_SARS_COV2(50, 10, N_SARS_COV2), 1.0))
        TOCI = sample('TOCI', Normal(self.f_TOCI(50, 10, N_TOCI), 1.0))
        PRR = sample('PRR', self.Spike(self.f_PRR(SARS_COV2, N_PRR)))
        ACE2 = sample('ACE2', self.Spike(self.f_ACE2(SARS_COV2, N_ACE2)))
        AngII = sample('AngII', self.Spike(self.f_AngII(ACE2, N_AngII)))
        AGTR1 = sample('AGTR1', self.Spike(self.f_AGTR1(AngII, N_AGTR1)))
        ADAM17 = sample('ADAM17', self.Spike(self.f_ADAM17(AGTR1, N_ADAM17)))
        TNF = sample('TNF', self.Spike(self.f_TNF(ADAM17, N_TNF)))
        sIL_6_alpha = sample('sIL_6_alpha', self.Spike(self.f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
        EGF = sample('EGF', self.Spike(self.f_EGF(ADAM17, N_EGF)))
        EGFR = sample('EGFR', self.Spike(self.f_EGFR(EGF, N_EGFR)))
        # STAT3 = sample('STAT3', Spike(f_STAT3(sIL_6_alpha, N_STAT3)))
        NF_xB = sample('NF_xB', self.Spike(self.f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
        IL6_STAT3 = sample('IL6_STAT3', self.Spike(self.f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
        IL_6_AMP = sample('IL_6_AMP', self.Spike(self.f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
        cytokine = sample('cytokine', self.Spike(self.f_cytokine(IL_6_AMP, N_cytokine)))

        if self.noise_type == NOISE_TYPE_OBSERVATIONAL:
            ## Use the dictionary structure for generating observational dataset
            samples = {
                'a(SARS_COV2)': SARS_COV2.numpy(),
                'a(PRR)': PRR.numpy(),
                'a(ACE2)': ACE2.numpy(),
                'a(AngII)': AngII.numpy(),
                'a(AGTR1)': AGTR1.numpy(),
                'a(ADAM17)': ADAM17.numpy(),
                'a(TOCI)': TOCI.numpy(),
                'a(TNF)': TNF.numpy(),
                'a(sIL_6_alpha)': sIL_6_alpha.numpy(),
                'a(EGF)': EGF.numpy(),
                'a(EGFR)': EGFR.numpy(),
                'a(IL6_STAT3)': IL6_STAT3.numpy(),
                'a(NF_xB)': NF_xB.numpy(),
                'a(IL6_AMP)': IL_6_AMP.numpy(),
                'a(cytokine)': cytokine.numpy(),
            }
        elif self.noise_type == NOISE_TYPE_SAMPLES:
            ## Use the variable list for generating samples for causal effect
            samples = [
                SARS_COV2,
                PRR,
                ACE2,
                AngII,
                AGTR1,
                ADAM17,
                TOCI,
                TNF,
                sIL_6_alpha,
                EGF,
                EGFR,
                NF_xB,
                IL6_STAT3,
                IL_6_AMP,
                cytokine,
            ]
        else:
            raise InvalidNoiseType(self.noise_type)

        return samples

    def noisy_mutilated_model(self, noise):
        N_SARS_COV2 = sample('N_SARS_COV2', Normal(noise['N_SARS_COV2'][0], noise['N_SARS_COV2'][1]))
        N_TOCI = sample('N_TOCI', Normal(noise['N_TOCI'][0], noise['N_TOCI'][1]))
        N_ACE2 = sample('N_ACE2', Normal(noise['N_ACE2'][0], noise['N_ACE2'][1]))
        N_PRR = sample('N_PRR', Normal(noise['N_PRR'][0], noise['N_PRR'][1]))
        N_AngII = sample('N_AngII', Normal(noise['N_AngII'][0], noise['N_AngII'][1]))
        N_AGTR1 = sample('N_AGTR1', Normal(noise['N_AGTR1'][0], noise['N_AGTR1'][1]))
        N_ADAM17 = sample('N_ADAM17', Normal(noise['N_ADAM17'][0], noise['N_ADAM17'][1]))
        N_sIL_6_alpha = sample('N_sIL_6_alpha', Normal(noise['N_sIL_6_alpha'][0], noise['N_sIL_6_alpha'][1]))
        N_TNF = sample('N_TNF', Normal(noise['N_TNF'][0], noise['N_TNF'][1]))
        N_EGF = sample('N_EGF', Normal(noise['N_EGF'][0], noise['N_EGF'][1]))
        N_EGFR = sample('N_EGFR', Normal(noise['N_EGFR'][0], noise['N_EGFR'][1]))
        N_IL6_STAT3 = sample('N_IL6_STAT3', Normal(noise['N_IL6_STAT3'][0], noise['N_IL6_STAT3'][1]))
        N_NF_xB = sample('N_NF_xB', Normal(noise['N_NF_xB'][0], noise['N_NF_xB'][1]))
        N_IL_6_AMP = sample('N_IL_6_AMP', Normal(noise['N_IL_6_AMP'][0], noise['N_IL_6_AMP'][1]))
        N_cytokine = sample('N_cytokine', Normal(noise['N_cytokine'][0], noise['N_cytokine'][1]))

        SARS_COV2 = sample('SARS_COV2', Normal(self.f_SARS_COV2(50, 10, N_SARS_COV2), 1.0))
        TOCI = sample('TOCI', Delta(tensor(0.0)))
        PRR = sample('PRR', self.Spike(self.f_PRR(SARS_COV2, N_PRR)))
        ACE2 = sample('ACE2', self.Spike(self.f_ACE2(SARS_COV2, N_ACE2)))
        AngII = sample('AngII', self.Spike(self.f_AngII(ACE2, N_AngII)))
        AGTR1 = sample('AGTR1', self.Spike(self.f_AGTR1(AngII, N_AGTR1)))
        ADAM17 = sample('ADAM17', self.Spike(self.f_ADAM17(AGTR1, N_ADAM17)))
        TNF = sample('TNF', self.Spike(self.f_TNF(ADAM17, N_TNF)))
        sIL_6_alpha = sample('sIL_6_alpha', self.Spike(self.f_sIL_6_alpha(ADAM17, TOCI, N_sIL_6_alpha)))
        EGF = sample('EGF', self.Spike(self.f_EGF(ADAM17, N_EGF)))
        EGFR = sample('EGFR', self.Spike(self.f_EGFR(EGF, N_EGFR)))
        NF_xB = sample('NF_xB', self.Spike(self.f_NF_xB(PRR, EGFR, TNF, N_NF_xB)))
        IL6_STAT3 = sample('IL6_STAT3', self.Spike(self.f_IL6_STAT3(sIL_6_alpha, N_IL6_STAT3)))
        IL_6_AMP = sample('IL_6_AMP', self.Spike(self.f_IL_6_AMP(NF_xB, IL6_STAT3, N_IL_6_AMP)))
        cytokine = sample('cytokine', self.Spike(self.f_cytokine(IL_6_AMP, N_cytokine)))

        samples = {
            'a(SARS_COV2)': SARS_COV2.numpy(),
            'a(PRR)': PRR.numpy(),
            'a(ACE2)': ACE2.numpy(),
            'a(AngII)': AngII.numpy(),
            'a(AGTR1)': AGTR1.numpy(),
            'a(ADAM17)': ADAM17.numpy(),
            'a(TOCI)': TOCI.numpy(),
            'a(TNF)': TNF.numpy(),
            'a(sIL_6_alpha)': sIL_6_alpha.numpy(),
            'a(EGF)': EGF.numpy(),
            'a(EGFR)': EGFR.numpy(),
            'a(IL6_STAT3)': IL6_STAT3.numpy(),
            'a(NF_xB)': NF_xB.numpy(),
            'a(IL6_AMP)': IL_6_AMP.numpy(),
            'a(cytokine)': cytokine.numpy(),
        }
        return samples

    def infer(self, model, noise):
        return Importance(model, num_samples=1000).run(noise)

    @staticmethod
    def guide(noise):
        mu_constraints = constraints.interval(-3., 3.)
        sigma_constraints = constraints.interval(.0001, 3)
        mu = {
            k: pyro.param(
                f'{k}_mu',
                tensor(0.),
                constraint=mu_constraints,
            )
            for k in noise
        }
        sigma = {
            k: pyro.param(
                f'{k}_sigma',
                tensor(1.),
                constraint=sigma_constraints,
            )
            for k in noise
        }
        for k in noise:
            sample(k, Normal(mu[k], sigma[k]))

    def update_noise_svi(
        self,
        observed_steady_state,
        initial_noise,
        optimizer: Optional[Type[Optimizer]] = None,
        lr: float = 0.001,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        num_steps: int = 1000,
    ):
        observation_model = condition(self.noisy_model, observed_steady_state)
        pyro.clear_param_store()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if optimizer is None:
            optimizer = SGD
            optimizer_kwargs.setdefault('momentum', 0.1)

        svi = SVI(
            model=observation_model,
            guide=self.guide,
            optim=optimizer({'lr': lr, **optimizer_kwargs}),
            loss=Trace_ELBO(),
        )

        losses = []
        samples = defaultdict(list)
        for _ in trange(num_steps, desc='Running SVI'):
            losses.append(svi.step(initial_noise))
            for k in initial_noise:
                mu = f'{k}_mu'
                sigma = f'{k}_sigma'
                samples[mu].append(pyro.param(mu).item())
                samples[sigma].append(pyro.param(sigma).item())

        means = {k: statistics.mean(v) for k, v in samples.items()}

        # TODO is this a viable replacement?
        # updated_noise = {
        #     k: (means[f'{k}_mu'], means[f'{k}_sigma'])
        #     for k in initial_noise
        # }
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
            for k in initial_noise
        }
        return updated_noise


NOISE = {
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
    'N_cytokine': (0., 1.),
}


def scm_covid_counterfactual(
    betas,
    max_abundance,
    observation,
    ras_intervention,
    spike_width: float = 1.0,
    svi: bool = True,
    samples: int = 5000,
) -> List[float]:
    gf_scm = SigmoidSCM(betas, max_abundance, spike_width)

    if svi:
        updated_noise, _ = gf_scm.update_noise_svi(observation, NOISE)
    else:
        updated_noise = gf_scm.update_noise_importance(observation, NOISE)
    counterfactual_model = do(gf_scm.model, ras_intervention)
    cf_posterior = gf_scm.infer(counterfactual_model, updated_noise)
    cf_cytokine_marginal = EmpiricalMarginal(cf_posterior, sites=['cytokine'])
    scm_causal_effect_samples = [
        observation['cytokine'] - float(cf_cytokine_marginal.sample())
        for _ in range(samples)
    ]
    return scm_causal_effect_samples

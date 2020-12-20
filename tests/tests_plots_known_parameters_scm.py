"""
This is the test file which stores experiments to generate data
for plots in the paper that were generated using DataGeneration SCM
for covid-19 graph.


Check test_plots_bel2scm.py
to see experiments that were used to generate data from bel2scm algorithms.

All dataframes generated for this paper are in Tests/Data folder.
"""

import os
from typing import Optional
import torch
from bel2scm.generation.utils import run

HERE = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(HERE, 'data')


def main(
    seed: int = 23,
    n_noisy_samples: Optional[int] = None,
    n_noisy_samples_mutilated: Optional[int] = None,
) -> None:
    torch.manual_seed(seed)
    run(
        directory=DATA,
        n_noisy_samples=n_noisy_samples,
        n_noisy_samples_mutilated=n_noisy_samples_mutilated,
    )


if __name__ == '__main__':
    main()

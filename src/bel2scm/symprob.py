"""

1. Data: Joint distribution - P(set of variables)
2. Operations:
    - Condition (one or more of the variables already in the distribution)
    - Margin (one or mode of the variables in the distribution)
"""

import unittest
from typing import Collection, Hashable, Optional, Union


class DuplicateSymbol(ValueError):
    pass


class JointDistribution:
    def __init__(self, symbols: Collection[Hashable], parent: Optional['JointDistribution'] = None):
        self.symbols = set(symbols)
        self.operations = []

    def margin(self, symbols: Union[Hashable, Collection[Hashable]]) -> 'JointDistribution':
        if not isinstance(symbols, (list, set, tuple)):
            symbols = {symbols}
        self.operations.append({
            'margin': self.symbols.intersection(symbols),
        })
        return self

    def condition(self):
        pass

    def simplify(self):
        # see https://github.com/cran/causaleffect/blob/master/R/simplify.R
        raise NotImplementedError

    def __contains__(self, item):
        return item in self.symbols


class DistributionTest(unittest.TestCase):

    def assert_dist_equal(self, a, b):
        self.assertEqual(a.simplify(), b.simplify())

    def test_symbols(self):
        d = JointDistribution(['X', 'Y', 'Z'])
        self.assert_dist_equal(d, d.margin('Q'))

        self.assert_dist_equal(JointDistribution(['Y', 'Z']), d.margin('X'))

        # Twice marginalized
        self.assert_dist_equal(JointDistribution(['Z']), d.margin(['X', 'Y']))
        self.assert_dist_equal(JointDistribution(['Z']), d.margin('X').margin('Y'))
        self.assert_dist_equal(JointDistribution(['Z']), d.margin('Y').margin('X'))

        # All marginalized
        self.assert_dist_equal(JointDistribution([]), d.margin(['X', 'Y', 'Z']))
        self.assert_dist_equal(JointDistribution([]), d.margin(['X', 'Y']).margin('Z'))
        self.assert_dist_equal(JointDistribution([]), d.margin(['X', 'Z']).margin('Y'))
        self.assert_dist_equal(JointDistribution([]), d.margin(['Y']).margin(['X', 'Z']))
        self.assert_dist_equal(JointDistribution([]), d.margin(['X']).margin(['Y', 'Z']))
        self.assert_dist_equal(JointDistribution([]), d.margin(['Z']).margin(['X', 'Y']))
        self.assert_dist_equal(JointDistribution([]), d.margin('X').margin('Y').margin('Z'))
        self.assert_dist_equal(JointDistribution([]), d.margin('X').margin('Z').margin('Y'))
        self.assert_dist_equal(JointDistribution([]), d.margin('Y').margin('X').margin('Z'))
        self.assert_dist_equal(JointDistribution([]), d.margin('Y').margin('Z').margin('X'))
        self.assert_dist_equal(JointDistribution([]), d.margin('Z').margin('Y').margin('X'))
        self.assert_dist_equal(JointDistribution([]), d.margin('Z').margin('X').margin('Y'))

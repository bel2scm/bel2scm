"""
1. Data: Joint distribution - P(set of variables)
2. Operations:
    - Condition (one or more of the variables already in the distribution)
    - Margin (one or mode of the variables in the distribution)
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Hashable, List, Set

from bel2scm.symb.operations import Operation, simplify


@dataclass()
class JointDistribution:
    symbols: Set[Hashable]
    operations: List[Operation] = field(default_factory=list)

    def margin(self, symbols: Union[Hashable, Set[Hashable]]) -> JointDistribution:
        return self._append_symbol_op(symbols, 'margin')

    def condition(self, symbols: Union[Hashable, Set[Hashable]]) -> JointDistribution:
        return self._append_symbol_op(symbols, 'condition')

    def _append_symbol_op(self, symbols: Union[Hashable, Set[Hashable]], operation: str) -> JointDistribution:
        if not isinstance(symbols, set):
            symbols = {symbols}
        self.operations.append(Operation(operation=operation, symbols=self.symbols.intersection(symbols)))
        return self

    def simplify(self) -> JointDistribution:
        return JointDistribution(
            symbols=self.symbols,
            operations=simplify(self.operations),
        )

    def __eq__(self, other):
        return (
            isinstance(other, JointDistribution)
            and self.symbols == other.symbols
            and self.operations == other.operations
        )


class DistributionTest(unittest.TestCase):
    def assert_dist_equal(self, a: JointDistribution, b: JointDistribution):
        self.assertEqual(a.simplify(), b.simplify())

    def test_symbols(self):
        d = JointDistribution({'X', 'Y', 'Z'})
        self.assert_dist_equal(d, d.margin('Q'))

        self.assert_dist_equal(JointDistribution({'Y', 'Z'}), d.margin('X'))

        # Twice marginalized
        self.assert_dist_equal(JointDistribution({'Z'}), d.margin({'X', 'Y'}))
        self.assert_dist_equal(JointDistribution({'Z'}), d.margin('X').margin('Y'))
        self.assert_dist_equal(JointDistribution({'Z'}), d.margin('Y').margin('X'))

        # All marginalized
        self.assert_dist_equal(JointDistribution(set()), d.margin({'X', 'Y', 'Z'}))
        self.assert_dist_equal(JointDistribution(set()), d.margin({'X', 'Y'}).margin('Z'))
        self.assert_dist_equal(JointDistribution(set()), d.margin({'X', 'Z'}).margin('Y'))
        self.assert_dist_equal(JointDistribution(set()), d.margin({'Y'}).margin({'X', 'Z'}))
        self.assert_dist_equal(JointDistribution(set()), d.margin({'X'}).margin({'Y', 'Z'}))
        self.assert_dist_equal(JointDistribution(set()), d.margin({'Z'}).margin({'X', 'Y'}))
        self.assert_dist_equal(JointDistribution(set()), d.margin('X').margin('Y').margin('Z'))
        self.assert_dist_equal(JointDistribution(set()), d.margin('X').margin('Z').margin('Y'))
        self.assert_dist_equal(JointDistribution(set()), d.margin('Y').margin('X').margin('Z'))
        self.assert_dist_equal(JointDistribution(set()), d.margin('Y').margin('Z').margin('X'))
        self.assert_dist_equal(JointDistribution(set()), d.margin('Z').margin('Y').margin('X'))
        self.assert_dist_equal(JointDistribution(set()), d.margin('Z').margin('X').margin('Y'))


if __name__ == '__main__':
    unittest.main()

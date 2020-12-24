"""A data structure for operations and a python reimplementation of the simplification algorithm from `causaleffect`.

.. seealso:: https://github.com/cran/causaleffect/blob/master/R/simplify.R
"""

from __future__ import annotations

import itertools as itt
import unittest
from typing import Hashable, List, Set, TypedDict

__all__ = [
    'Operation',
    'simplify',
]


class Operation(TypedDict):
    """A dictionary representing an operation.

    Instantiate like:

    >>> Operation(operation='margin', symbols={'A', 'B', 'C'})
    {'operation': 'margin', 'symbols': {'A', 'B', 'C'}

    Or use the convenient :meth:`margin`/:meth:`condition` functions:

    >>> Operation.margin({'A', 'B', 'C'})
    {'operation': 'margin', 'symbols': {'A', 'B', 'C'}
    """

    #: The name of the operation (e.g., 'margin', 'condition')
    operation: str

    #: The symbols upon which the operation acts
    symbols: Set[Hashable]

    @classmethod
    def margin(cls, symbols: Set[Hashable]) -> Operation:
        return Operation(operation='margin', symbols=symbols)

    @classmethod
    def condition(cls, symbols: Set[Hashable]) -> Operation:
        return Operation(operation='condition', symbols=symbols)


def simplify(operations: List[Operation]) -> List[Operation]:
    """Return a simplified list of operations."""
    if not operations:
        return []

    # TODO

    if any(o['operation'] == 'condition' for o in operations):
        raise NotImplementedError

    return [
        Operation.margin(set(itt.chain.from_iterable(
            o['symbols']
            for o in operations
        ))),
    ]


class TestSimplify(unittest.TestCase):
    def test_struct(self):
        self.assertEqual({'operation': 'margin', 'symbols': {'a', 'b'}}, Operation.margin({'a', 'b'}))
        self.assertEqual({'operation': 'condition', 'symbols': {'a', 'b'}}, Operation.condition({'a', 'b'}))

    def assert_simplified(self, simplified: List[Operation], actual: List[Operation], msg=None):
        self.assertEqual(simplified, simplify(actual), msg=msg)

    def test_margin(self):
        # Trivial Case 1: Empty
        self.assert_simplified([], [], msg='An empty list is already simplified')

        # Trivial Case 2: Single Entry
        self.assert_simplified(
            [Operation.margin(set('a'))],
            [Operation.margin(set('a'))],
            msg='A margin on a single variable is already simplified',
        )
        self.assert_simplified(
            [Operation.margin(set('ab'))],
            [Operation.margin(set('ab'))],
            msg='A margin on a two variables is already simplified',
        )

        # Non-trivial: multiple margins in succession
        self.assert_simplified(
            [Operation.margin(set('ab'))],
            [Operation.margin(set('a')), Operation.margin(set('b'))],
            msg='Two successive margins are the same as a single combine',
        )
        self.assert_simplified(
            [Operation.margin(set('ab'))],
            [Operation.margin(set('b')), Operation.margin(set('a'))],
            msg='Two successive margins are the same as a single combine',
        )

        self.assert_simplified(
            [Operation.margin(set('abc'))],
            [Operation.margin(set('a')), Operation.margin(set('b')), Operation.margin(set('c'))],
            msg='Three successive margins are the same as a single combine',
        )
        self.assert_simplified(
            [Operation.margin(set('abc'))],
            [Operation.margin(set('ab')), Operation.margin(set('c'))],
            msg='Two successive margins grouped are the same as a single combine',
        )
        self.assert_simplified(
            [Operation.margin(set('abc'))],
            [Operation.margin(set('a')), Operation.margin(set('bc'))],
            msg='Two successive margins grouped are the same as a single combine',
        )

if __name__ == '__main__':
    unittest.main()
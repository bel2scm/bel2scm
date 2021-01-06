# -*- coding: utf-8 -*-

"""Test the probability DSL."""

import unittest

from bel2scm.probability_dsl import Expression, P, Sum

from tests.probability_constants import *


class TestDSL(unittest.TestCase):
    """Tests for the stringifying instances of the probability DSL."""

    def assert_latex(self, s: str, expression: Expression):
        """Assert the expression when it is converted to a string."""
        self.assertIsInstance(s, str)
        self.assertEqual(s, expression.to_latex())

    def test_stringify(self):
        """Test stringifying DSL instances."""
        self.assert_latex('P(A|B)', P(Condition('A', 'B')))
        self.assert_latex('P(A|B)', P(Condition(A, B)))
        self.assert_latex('P(A|B)', P(Condition(A, [B])))
        self.assert_latex('P(A|B)', P(Condition(A, ['B'])))
        # Fun operator overloading
        self.assert_latex('P(A|B)', P(A | B))
        self.assert_latex('P(A|B)', P(A | 'B'))
        self.assert_latex('P(A|B)', P(A | [B]))
        self.assert_latex('P(A|B)', P(A | ['B']))

        self.assert_latex('P(A|B,C)', P(A | [B, C]))
        self.assert_latex('P(A|B,C)', P(A | [B, 'C']))
        self.assert_latex('P(A|B,C)', P(A | ['B', 'C']))
        self.assert_latex('P(A|B,C)', P(A | B | C))

        self.assert_latex(
            "[ sum_{S,T} P(A|B) P(C|D) ]",
            Sum(
                ranges=[S, T],
                expressions=[
                    P(A | B),
                    P(C | D),
                ],
            ),
        )

        # Sum with sum inside
        self.assert_latex(
            "[ sum_{S,T} P(A|B) [ sum_{Q} P(C|D) ] ]",
            Sum(
                ranges=[S, T],
                expressions=[
                    P(A | B),
                    Sum(
                        ranges=[Q],
                        expressions=[
                            P(C | D),
                        ],
                    ),
                ],
            ),
        )

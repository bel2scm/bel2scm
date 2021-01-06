# -*- coding: utf-8 -*-

"""Test the probability DSL."""

import unittest

from bel2scm.probability_dsl import Condition, Expression, Probability, Sum, Variable

S = Variable('S')
T = Variable('T')
Q = Variable('Q')
A = Variable('A')
A_1 = Variable('A', index=1)
B = Variable('B')
B_1 = Variable('B', index=1)
B_2 = Variable('B', index=2)
C = Variable('C')
D = Variable('D')
A_GIVEN = Condition(child=A, parents=[])
A_GIVEN_B = Condition(child=A, parents=[B])
A_GIVEN_B_C = Condition(child=A, parents=[B, C])
A_GIVEN_B_1 = Condition(child=A, parents=[B_1])
A_GIVEN_B_1_B_2 = Condition(child=A, parents=[B_1, B_2])
C_GIVEN_D = Condition(child=C, parents=[D])


class TestDSL(unittest.TestCase):
    """Tests for the stringifying instances of the probability DSL."""

    def assert_latex(self, s: str, expression: Expression):
        """Assert the expression when it is converted to a string."""
        self.assertIsInstance(s, str)
        self.assertEqual(s, expression.to_latex())

    def test_stringify(self):
        """Test stringifying DSL instances."""
        self.assert_latex(
            "[ sum_{S,T} P(A|B) P(C|D) ]",
            Sum(
                ranges=[S, T],
                expressions=[
                    Probability(A_GIVEN_B),
                    Probability(C_GIVEN_D),
                ],
            ),
        )

        # Sum with sum inside
        self.assert_latex(
            "[ sum_{S,T} P(A|B) [ sum_{Q} P(C|D) ] ]",
            Sum(
                ranges=[S, T],
                expressions=[
                    Probability(probability=A_GIVEN_B),
                    Sum(
                        ranges=[Q],
                        expressions=[
                            Probability(C_GIVEN_D),
                        ],
                    ),
                ],
            ),
        )

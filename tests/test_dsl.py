# -*- coding: utf-8 -*-

"""Test the probability DSL."""

import unittest

from bel2scm.probability_dsl import Condition, Expression, Probability, Sum, Variable

S = Variable(name='S')
T = Variable(name='T')
A = Variable(name='A')
A_1 = Variable(name='A', index=1)
B = Variable(name='B')
B_1 = Variable(name='B', index=1)
B_2 = Variable(name='B', index=2)
C = Variable(name='C')
D = Variable(name='D')
A_GIVEN = Condition(child=A, parents=[])
A_GIVEN_B = Condition(child=A, parents=[B])
A_GIVEN_B_C = Condition(child=A, parents=[B, C])
A_GIVEN_B_1 = Condition(child=A, parents=[B_1])
A_GIVEN_B_1_B_2 = Condition(child=A, parents=[B_1, B_2])
C_GIVEN_D = Condition(child=C, parents=[D])


class TestDSL(unittest.TestCase):

    def assert_latex(self, s: str, expression: Expression):
        self.assertEqual(s, expression.to_latex())

    def test_stringify(self):
        self.assert_latex(
            "[ sum_{S,T} P(A|B) P(C|D) ]",
            Sum(
                ranges=[S, T],
                expressions=[
                    Probability(probability=A_GIVEN_B),
                    Probability(probability=C_GIVEN_D),
                ],
            ),
        )

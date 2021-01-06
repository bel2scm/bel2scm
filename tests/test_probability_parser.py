# -*- coding: utf-8 -*-

"""Test the probability parser."""

import unittest

import pyparsing
from bel2scm.probability_dsl import Condition, Frac, Probability, Sum, Variable
from bel2scm.probability_parser import conditional, frac_expr, inside_stuff, sum_expr, variable

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


class TestGrammar(unittest.TestCase):
    def assert_parse_equal(self, expression, parse_expression: pyparsing.ParseExpression, instring: str):
        self.assertEqual(expression, parse_expression.parseString(instring).asDict())

    def test_variable(self):
        self.assert_parse_equal(A, variable, 'A')
        self.assert_parse_equal(A_1, variable, 'A_1')

    def test_conditional_inner(self):
        self.assert_parse_equal(A_GIVEN, inside_stuff, 'A')
        self.assert_parse_equal(A_GIVEN_B, inside_stuff, 'A|B')
        self.assert_parse_equal(A_GIVEN_B_1, inside_stuff, 'A|B_1')
        self.assert_parse_equal(A_GIVEN_B_1_B_2, inside_stuff, 'A|B_1,B_2')

    def test_conditional(self):
        self.assert_parse_equal(Probability(A_GIVEN), conditional, 'P(A)')
        self.assert_parse_equal(Probability(A_GIVEN_B), conditional, 'P(A|B)')
        self.assert_parse_equal(Probability(A_GIVEN_B_1), conditional, 'P(A|B_1)')
        self.assert_parse_equal(Probability(A_GIVEN_B_1_B_2), conditional, 'P(A|B_1,B_2)')

    def test_sum(self):
        self.assert_parse_equal(
            Sum(
                ranges=[],
                expressions=[
                    Probability(probability=A_GIVEN_B),
                ],
            ),
            sum_expr,
            "[ sum_{} P(A|B) ]",
        )
        self.assert_parse_equal(
            Sum(
                ranges=[S],
                expressions=[
                    Probability(A_GIVEN_B),
                ],
            ),
            sum_expr,
            "[ sum_{S} P(A|B) ]",
        )
        self.assert_parse_equal(
            Sum(
                ranges=[S, T],
                expressions=[
                    Probability(A_GIVEN_B),
                ],
            ),
            sum_expr,
            "[ sum_{S,T} P(A|B) ]",
        )
        self.assert_parse_equal(
            Sum(
                ranges=[S, T],
                expressions=[
                    Probability(A_GIVEN_B_C),
                ],
            ),
            sum_expr,
            "[ sum_{S,T} P(A|B,C) ]",
        )
        self.assert_parse_equal(
            Sum(
                ranges=[S, T],
                expressions=[
                    Probability(A_GIVEN_B),
                    Probability(C_GIVEN_D),
                ],
            ),
            sum_expr,
            "[ sum_{S,T} P(A|B) P(C|D) ]",
        )

    def test_frac(self):
        self.assert_parse_equal(
            Frac(
                Probability(A_GIVEN_B),
                Probability(C_GIVEN_D),
            ),
            frac_expr,
            'P(A|B) / P(C|D)',
        )
        self.assert_parse_equal(
            Frac(
                Sum(
                    ranges=[S, T],
                    expressions=[
                        Probability(A_GIVEN_B),
                        Probability(C_GIVEN_D),
                    ]
                ),
                Probability(C_GIVEN_D),
            ),
            frac_expr,
            '[ sum_{S,T} P(A|B) P(C|D) ] / P(C|D)',
        )

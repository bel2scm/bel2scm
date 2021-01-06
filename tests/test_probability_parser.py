# -*- coding: utf-8 -*-

"""Test the probability parser."""

import unittest

import pyparsing
from bel2scm.probability_dsl import P, Sum
from bel2scm.probability_parser import conditional, frac_expr, inside_stuff, sum_expr, variable

from tests.probability_constants import *


class TestGrammar(unittest.TestCase):
    def assert_parse_equal(self, expression, parse_expression: pyparsing.ParseExpression, instring: str):
        self.assertEqual(expression, parse_expression.parseString(instring).asDict())

    def test_variable(self):
        self.assert_parse_equal(A, variable, 'A')
        self.assert_parse_equal(A_1, variable, 'A_1')

    def test_conditional_inner(self):
        self.assert_parse_equal(A | [], inside_stuff, 'A')
        self.assert_parse_equal(A | B, inside_stuff, 'A|B')
        self.assert_parse_equal(A | B[1], inside_stuff, 'A|B_1')
        self.assert_parse_equal(A | [B[1], B[2]], inside_stuff, 'A|B_1,B_2')

    def test_conditional(self):
        self.assert_parse_equal(P(A_GIVEN), conditional, 'P(A)')
        self.assert_parse_equal(P(A | B), conditional, 'P(A|B)')
        self.assert_parse_equal(P(A | B[1]), conditional, 'P(A|B_1)')
        self.assert_parse_equal(P(A | [B[1], B[2]]), conditional, 'P(A|B_1,B_2)')

    def test_sum(self):
        self.assert_parse_equal(
            Sum([], P(A | B)),
            sum_expr,
            "[ sum_{} P(A|B) ]",
        )
        self.assert_parse_equal(
            Sum(None, P(A | B)),
            sum_expr,
            "[ sum_{} P(A|B) ]",
        )

        self.assert_parse_equal(
            Sum(S, P(A | B)),
            sum_expr,
            "[ sum_{S} P(A|B) ]",
        )
        self.assert_parse_equal(
            Sum([S, T], P(A | B)),
            sum_expr,
            "[ sum_{S,T} P(A|B) ]",
        )
        self.assert_parse_equal(
            Sum([S, T], P(A | [B, C])),
            sum_expr,
            "[ sum_{S,T} P(A|B,C) ]",
        )
        self.assert_parse_equal(
            Sum(
                ranges=[S, T],
                expressions=[
                    P(A | B),
                    P(C | D),
                ],
            ),
            sum_expr,
            "[ sum_{S,T} P(A|B) P(C|D) ]",
        )

    def test_frac(self):
        self.assert_parse_equal(
            P(A | B) / P(C | D),
            frac_expr,
            'P(A|B) / P(C|D)',
        )
        self.assert_parse_equal(
            Sum([S, T], [P(A | B), P(C | D)]) / P(C | D),
            frac_expr,
            '[ sum_{S,T} P(A|B) P(C|D) ] / P(C|D)',
        )

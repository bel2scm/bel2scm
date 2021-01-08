# -*- coding: utf-8 -*-

"""Test the probability parser."""

import unittest

from pyparsing import ParserElement

from bel2scm.probability_dsl import P, Sum
from bel2scm.probability_parser import fraction_pe, probability_pe, sum_pe, variable_pe
from tests.probability_constants import *


class TestGrammar(unittest.TestCase):
    """Tests for parsing probability expressions."""

    def assert_many(self, expressions, parser_element: ParserElement, direct: bool = True):
        """Help testing many."""
        for expression in expressions:
            with self.subTest(expr=expression.to_text()):
                self.assert_parse_equal(expression, parser_element, direct=direct)

    def assert_parse_equal(self, expression, parser_element: ParserElement, direct: bool = True):
        """Help test parsing works round trip.

        :param expression: The DSL object to check
        :param parser_element: The relevant parser element for specific checking
        :param direct: If true, uses object equals checks. If false, uses stringification then string equals checks.
            Switch to false when things aren't working during development.
        """
        text = expression.to_text()
        parse_result = parser_element.parseString(text)
        result_expression = parse_result.asList()[0]
        self.assertIsInstance(result_expression, expression.__class__)
        if direct:
            self.assertEqual(expression, result_expression)
        else:
            self.assertEqual(text, result_expression.to_text())

    def test_variable(self):
        """Tests for variables, intervention variables, and counterfactual variables."""
        self.assert_many(
            [
                A,
                A @ B,
                A @ ~B,
                A @ B @ C,
                A @ ~B @ C,
                A @ ~B @ ~C,
                A @ B @ ~C,
            ],
            variable_pe,
        )

    def test_probability(self):
        """Tests for probabilities."""
        self.assert_many(
            [
                P(A),
                P(A, B),
                P(A | B),
                P(A @ X | B),
                P(A @ ~X | B),
                P(A | B @ Y),
                P(A | B @ ~Y),
                P(A @ X | B @ ~Y),
                P(A @ X | B @ ~Y | C @ Z),
                P(A | [B, C]),
                P(A, B, C),
            ],
            probability_pe,
        )

    def test_sum(self):
        """Tests for sums."""
        self.assert_many(
            [
                Sum(P(A)),
                Sum(P(A, B)),
                Sum(P(A | B)),
                Sum(P(A | B) * P(B)),
                Sum[B](P(A | B) * P(B)),
                Sum[B, C](P(A | B) * P(B)),
            ],
            sum_pe,
        )

    def test_fraction(self):
        """Tests for fractions."""
        self.assert_many(
            [
                Sum(P(A)) / P(B),
                Sum(P(A, B)) / P(A),
                Sum[B](P(A | B) * P(B)) / Sum(P(A | B) * P(B)),
            ],
            fraction_pe,
        )

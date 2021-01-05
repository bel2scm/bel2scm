import typing
import unittest
from typing import List, TypedDict

from pyparsing import (
    Group, OneOrMore, Optional, ParseExpression, Suppress, TokenConverter, Word, alphas, delimitedList, nestedExpr,
    pyparsing_common as ppc,
)

variable = (
    Word(alphas).setResultsName('name')
    + Optional(Suppress('_') + ppc.number.setResultsName('index'))
)
inside_stuff = (
    Group(variable).setResultsName('child')
    + Group(Optional(Suppress('|') + delimitedList(Group(variable)))).setResultsName('parents')
)

conditional = TokenConverter(Suppress('P') + nestedExpr('(', ')', inside_stuff)).setResultsName('probability')

# expr = Forward()

sum_expr = Group(
    Suppress('[')
    + Suppress('sum_{')
    + Group(Optional(delimitedList(Group(variable)))).setResultsName('ranges')
    + Suppress('}')
    + Group(OneOrMore(Group(conditional))).setResultsName('expressions')
    + Suppress(']')
).setResultsName('sum')

# expr << (conditional | sum_expr)
# expr = (conditional | sum_expr)
# frac_expr = Group(
#     Suppress('frac_{')
#     + expr.setResultsName('numerator')
#     + Suppress('}{')
#     + expr.setResultsName('denominator')
#     + Suppress('}')
# )

'''
[sum_{W,D,Z,V} [sum_{}P(W|X)][sum_{} [sum_{X,W,Z,Y,V} P(X,W,D,Z,Y,V)]][sum_{}P(Z|D,V)][sum_{} [sum_{X} P(Y|X,D,V,Z,W)P(X|)]][sum_{} [sum_{X,W,D,Z,Y} P(X,W,D,Z,Y,V)]]]
[[sum_{D,Z,V} [sum_{} [sum_{X,W,Z,Y,V} P(X,W,D,Z,Y,V)]][sum_{}P(Z|D,V)][sum_{} [sum_{X} P(Y|X,D,V,Z,W)P(X|)]][sum_{} [sum_{X,W,D,Z,Y} P(X,W,D,Z,Y,V)]]]]/[ sum_{Y}[sum_{D,Z,V} [sum_{} [sum_{X,W,Z,Y,V} P(X,W,D,Z,Y,V)]][sum_{}P(Z|D,V)][sum_{} [sum_{X} P(Y|X,D,V,Z,W)P(X|)]][sum_{} [sum_{X,W,D,Z,Y} P(X,W,D,Z,Y,V)]]]]
'''


class Variable(TypedDict):
    name: str
    index: typing.Optional[int]


class Condition(TypedDict):
    child: Variable
    parents: List[Variable]


class Probability(TypedDict):
    probability: Condition


class Sum(TypedDict):
    ranges: List[Variable]
    expressions: List['Expression']


Expression = typing.Union[Condition, Sum]

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
A_GIVEN_B = Condition(child=A, parents=[B])
A_GIVEN_B_C = Condition(child=A, parents=[B, C])
A_GIVEN_B_1 = Condition(child=A, parents=[B_1])
A_GIVEN_B_1_B_2 = Condition(child=A, parents=[B_1, B_2])
C_GIVEN_D = Condition(child=C, parents=[D])


class TestGrammar(unittest.TestCase):
    def assert_parse_equal(self, expression: Expression, parse_expression: ParseExpression, instring: str):
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
        self.assert_parse_equal(Probability(probability=A_GIVEN), conditional, 'P(A)')
        self.assert_parse_equal(Probability(probability=A_GIVEN_B), conditional, 'P(A|B)')
        self.assert_parse_equal(Probability(probability=A_GIVEN_B_1), conditional, 'P(A|B_1)')
        self.assert_parse_equal(Probability(probability=A_GIVEN_B_1_B_2), conditional, 'P(A|B_1,B_2)')

    def test_sum(self):
        self.assert_parse_equal(
            {
                'sum': {
                    'ranges': [],
                    'expressions': [
                        Probability(probability=A_GIVEN_B),
                    ],
                },
            },
            sum_expr,
            "[ sum_{} P(A|B) ]",
        )
        self.assert_parse_equal(
            {
                'sum': {
                    'ranges': [S],
                    'expressions': [
                        Probability(probability=A_GIVEN_B),
                    ],
                },
            },
            sum_expr,
            "[ sum_{S} P(A|B) ]",
        )
        self.assert_parse_equal(
            {
                'sum': {
                    'ranges': [S, T],
                    'expressions': [
                        Probability(probability=A_GIVEN_B),
                    ],
                },
            },
            sum_expr,
            "[ sum_{S,T} P(A|B) ]",
        )
        self.assert_parse_equal(
            {
                'sum': {
                    'ranges': [S, T],
                    'expressions': [
                        Probability(probability=A_GIVEN_B_C),
                    ],
                },
            },
            sum_expr,
            "[ sum_{S,T} P(A|B,C) ]",
        )
        self.assert_parse_equal(
            {
                'sum': {
                    'ranges': [S, T],
                    'expressions': [
                        Probability(probability=A_GIVEN_B),
                        Probability(probability=C_GIVEN_D),
                    ],
                },
            },
            sum_expr,
            "[ sum_{S,T} P(A|B) P(C|D) ]",
        )

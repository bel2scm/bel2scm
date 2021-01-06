# -*- coding: utf-8 -*-

from pyparsing import (
    Group, OneOrMore, Optional, Suppress, TokenConverter, Word, alphas, delimitedList, nestedExpr,
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

# expr = pyparsing.Forward()

sum_expr = Group(
    Suppress('[')
    + Suppress('sum_{')
    + Group(Optional(delimitedList(Group(variable)))).setResultsName('ranges')
    + Suppress('}')
    + Group(OneOrMore(Group(conditional))).setResultsName('expressions')
    + Suppress(']')
).setResultsName('sum')

# TODO replace with Forward()
expr = (conditional | sum_expr)

frac_expr = Group(
    Group(expr).setResultsName('numerator')
    + Suppress('/')
    + Group(expr).setResultsName('denominator')
).setResultsName('frac')

# expr <<= (
#     frac_expr | sum_expr |  conditional
# )

'''
[sum_{W,D,Z,V} [sum_{}P(W|X)][sum_{} [sum_{X,W,Z,Y,V} P(X,W,D,Z,Y,V)]][sum_{}P(Z|D,V)][sum_{} [sum_{X} P(Y|X,D,V,Z,W)P(X|)]][sum_{} [sum_{X,W,D,Z,Y} P(X,W,D,Z,Y,V)]]]
[[sum_{D,Z,V} [sum_{} [sum_{X,W,Z,Y,V} P(X,W,D,Z,Y,V)]][sum_{}P(Z|D,V)][sum_{} [sum_{X} P(Y|X,D,V,Z,W)P(X|)]][sum_{} [sum_{X,W,D,Z,Y} P(X,W,D,Z,Y,V)]]]]/[ sum_{Y}[sum_{D,Z,V} [sum_{} [sum_{X,W,Z,Y,V} P(X,W,D,Z,Y,V)]][sum_{}P(Z|D,V)][sum_{} [sum_{X} P(Y|X,D,V,Z,W)P(X|)]][sum_{} [sum_{X,W,D,Z,Y} P(X,W,D,Z,Y,V)]]]]
'''

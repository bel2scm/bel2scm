# -*- coding: utf-8 -*-

from pyparsing import Group, OneOrMore, Optional, ParseResults, Suppress, Word, alphas, delimitedList

from bel2scm.probability_dsl import (
    ConditionalProbability, CounterfactualVariable, Expression, Fraction, Intervention, JointProbability, Probability,
    Product, Sum, Variable,
)


def _parse_variable(_s, _l, tokens: ParseResults):
    return Variable(tokens[0])


letter = Word(alphas)


def _set_star(_s, _l, tokens: ParseResults):
    return 1 == len(tokens.asList())


def _make_intervention(_s, _l, tokens: ParseResults):
    if tokens['star']:
        return Intervention(name=tokens['name'], star=True)
    else:
        return Variable(name=tokens['name'])


def _unpack(_s, _l, tokens: ParseResults):
    return tokens[0]


def _make_variable(_s, _l, tokens: ParseResults) -> Variable:
    if 'interventions' not in tokens:
        return Variable(name=tokens['name'])
    return CounterfactualVariable(
        name=tokens['name'],
        interventions=tokens['interventions'].asList(),
    )


def _make_probability(_s, _l, tokens: ParseResults) -> Probability:
    children, parents = tokens['children'].asList(), tokens['parents'].asList()
    if not parents:
        return Probability(JointProbability(children=children))
    if not children:
        raise ValueError
    if len(children) > 1:
        raise ValueError
    return Probability(ConditionalProbability(child=children[0], parents=parents))


def _make_product(_s, _l, tokens: ParseResults) -> Product:
    return Product(tokens.asList())


def _make_sum(_s, _l, tokens: ParseResults) -> Sum:
    return Sum(
        ranges=tokens['ranges'].asList() if 'ranges' in tokens else [],
        expression=tokens['expression'][0],
    )


def _make_frac(_s, _l, tokens: ParseResults) -> Fraction:
    return Fraction(
        numerator=tokens['numerator'][0],
        denominator=tokens['denominator'][0],
    )


# The suffix "pe" refers to :class:`pyparsing.ParserElement`, which is the
#  class in pyparsing that everything inherits from
star_pe = Optional('*')('star').setParseAction(_set_star)
intervention_pe = letter('name') + star_pe
intervention_pe.setParseAction(_make_intervention)
interventions_pe = Optional(
    Suppress('_{')
    + delimitedList(Group(intervention_pe).setParseAction(_unpack)).setResultsName('interventions')
    + Suppress('}')
)
variable_pe = letter('name') + interventions_pe
variable_pe.setParseAction(_make_variable)

variables_pe = delimitedList(Group(variable_pe).setParseAction(_unpack))
_children_pe = Group(variables_pe).setResultsName('children')
_parents_pe = Group(Optional(Suppress('|') + variables_pe)).setResultsName('parents')
probability_pe = Suppress('P(') + _children_pe + _parents_pe + Suppress(')')
probability_pe.setParseAction(_make_probability)

expr = probability_pe

product_pe = expr + OneOrMore(expr)
product_pe.setParseAction(_make_product)

expr = product_pe | probability_pe

sum_pe = (
    Suppress('[')
    + Suppress('sum_{')
    + Optional(Group(variables_pe).setResultsName('ranges'))
    + Suppress('}')
    + expr.setResultsName('expression')
    + Suppress(']')
)
sum_pe.setParseAction(_make_sum)

expr = sum_pe | product_pe | probability_pe

fraction_pe = (
    expr.setResultsName('numerator')
    + Suppress('/')
    + expr.setResultsName('denominator')
)
fraction_pe.setParseAction(_make_frac)

expr = fraction_pe | sum_pe | product_pe | probability_pe


def parse(s: str) -> Expression:
    """Parse an expression."""
    x = expr.parseString(s)
    return x.asList()[0]


if __name__ == '__main__':
    print(parse('P(A)'))
    print(parse('[ sum_{} P(A|B) ]'))

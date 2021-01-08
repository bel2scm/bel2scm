# -*- coding: utf-8 -*-

from __future__ import annotations

import functools
import itertools as itt
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, TypeVar, Union

__all__ = [
    'Variable',
    'ConditionalProbability',
    'P',
    'Probability',
    'Sum',
    'Fraction',
    'Expression',
]

X = TypeVar('X')
XList = Union[X, List[X]]


def _upgrade_variables(variables):
    return [variables] if isinstance(variables, Variable) else variables


@dataclass(frozen=True)
class Variable:
    #: The name of the variable
    name: str

    def to_latex(self) -> str:
        return self.name

    def intervene(self, interventions: XList[Intervention]) -> CounterfactualVariable:
        return CounterfactualVariable(
            name=self.name,
            interventions=_upgrade_variables(interventions),
        )

    def given(self, parents: XList[Variable]) -> ConditionalProbability:
        return ConditionalProbability(
            child=self,
            parents=_upgrade_variables(parents),
        )

    def joint(self, children: XList[Variable]) -> JointProbability:
        children = _upgrade_variables(children)
        return JointProbability([self, *children])

    def __or__(self, parents: XList[Variable]) -> ConditionalProbability:
        return self.given(parents)

    def __and__(self, children: XList[Variable]) -> JointProbability:
        return self.joint(children)

    def __matmul__(self, interventions: XList[Intervention]):
        return self.intervene(interventions)

    def __invert__(self):
        return Intervention(name=self.name, star=True)


@dataclass(frozen=True)
class Intervention(Variable):
    #: The name of the intervention
    name: str
    #: If true, indicates this intervention represents a value different from what was observed
    star: bool = False

    def to_latex(self) -> str:
        return f'{self.name}*' if self.star else self.name

    def __invert__(self):
        return Intervention(name=self.name, star=not self.star)


@dataclass(frozen=True)
class CounterfactualVariable(Variable):
    #: The name of the counterfactual variable
    name: str
    #: The interventions on the variable. Should be non-empty
    interventions: List[Intervention]

    def to_latex(self) -> str:
        intervention_latex = ','.join(intervention.to_latex() for intervention in self.interventions)
        return f'{self.name}_{{{intervention_latex}}}'

    def intervene(self, interventions: XList[Intervention]) -> CounterfactualVariable:
        interventions = _upgrade_variables(interventions)
        self._raise_for_overlapping_interventions(interventions)
        return CounterfactualVariable(
            name=self.name,
            interventions=[*self.interventions, *interventions],
        )

    def _raise_for_overlapping_interventions(self, new_interventions: List[Intervention]):
        overlaps = {
            new
            for old, new in itt.product(self.interventions, new_interventions)
            if old.name == new.name
        }
        if overlaps:
            raise ValueError(f'Overlapping interventions in new interventions: {overlaps}')


@dataclass
class JointProbability:
    children: List[Variable]

    def to_latex(self) -> str:
        return ','.join(child.to_latex() for child in self.children)

    def joint(self, children: XList[Variable]) -> JointProbability:
        return JointProbability([
            *self.children,
            *_upgrade_variables(children),
        ])

    def __and__(self, children: XList[Variable]) -> JointProbability:
        return self.joint(children)


@dataclass
class ConditionalProbability:
    child: Variable
    parents: List[Variable]

    def to_latex(self) -> str:
        parents = ','.join(parent.to_latex() for parent in self.parents)
        return f'{self.child.to_latex()}|{parents}'

    def given(self, parents: XList[Variable]) -> ConditionalProbability:
        return ConditionalProbability(
            child=self.child,
            parents=[*self.parents, *_upgrade_variables(parents)]
        )

    def __or__(self, parents: XList[Variable]) -> ConditionalProbability:
        return self.given(parents)


class Probability:
    def __init__(self, probability: Union[Variable, List[Variable], ConditionalProbability, JointProbability]):
        if isinstance(probability, list):
            probability = JointProbability(probability)
        self.probability = probability

    def __truediv__(self, other) -> Fraction:
        return Fraction(self, other)

    def to_latex(self) -> str:
        return f'P({self.probability.to_latex()})'

    def __mul__(self, other: Expression) -> Product:
        if isinstance(other, Product):
            return Product([self, *other.expressions])
        else:
            return Product([self, other])


P = Probability


@dataclass
class Product:
    expressions: List[Expression]

    def __mul__(self, other):
        if isinstance(other, list):
            return Product([*self.expressions, *other])
        else:
            return Product([*self.expressions, other])

    def to_latex(self):
        return ' '.join(e.to_latex() for e in self.expressions)


@dataclass
class Sum:
    expression: Expression
    # The variables over which the sum is done. Defaults to an empty list, meaning no variables.
    ranges: List[Variable] = field(default_factory=list)

    def to_latex(self) -> str:
        ranges = ','.join(r.to_latex() for r in self.ranges)
        return f'[ sum_{{{ranges}}} {self.expression.to_latex()} ]'

    def __truediv__(self, other) -> Fraction:
        return Fraction(self, other)

    def __class_getitem__(cls, ranges: Union[Variable, Tuple[Variable, ...]]) -> Callable[[Expression], Sum]:
        if isinstance(ranges, tuple):
            ranges = list(ranges)
        else:  # a single element is not given as a tuple, such as in Sum[T]
            ranges = [ranges]
        return functools.partial(Sum, ranges=ranges)


@dataclass
class Fraction:
    numerator: Expression
    denominator: Expression

    def to_latex(self) -> str:
        return f"{self.numerator.to_latex()} / {self.denominator.to_latex()}"


Expression = Union[Probability, Sum, Fraction, Product]

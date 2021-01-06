# -*- coding: utf-8 -*-

from typing import List, Optional, Union

__all__ = [
    'Variable',
    'Condition',
    'P',
    'Probability',
    'Sum',
    'Frac',
    'Expression',
]


class Variable(dict):
    def __init__(self, name: str, *, index: Optional[int] = None):
        d = {'name': name}
        if index is not None:
            d['index'] = index
        super().__init__(d)

    @property
    def name(self) -> str:
        return self['name']

    @property
    def index(self) -> Optional[int]:
        return self.get('index')

    def __getitem__(self, item: int) -> 'Variable':
        if isinstance(item, int):
            return Variable(name=self.name, index=item)
        return super().__getitem__(item)

    def to_latex(self) -> str:
        if self.index:
            return f'{self.name}_{self.index}'
        return self.name

    def given(self, parents: Union[str, 'Variable', List[Union[str, 'Variable']]]) -> 'Condition':
        if isinstance(parents, (str, Variable)):
            parents = [parents]
        return Condition(
            child=self,
            parents=parents,
        )

    def __or__(self, parents: Union[str, 'Variable', List[Union[str, 'Variable']]]) -> 'Condition':
        return self.given(parents)


class Condition(dict):
    def __init__(self, child: Union[str, Variable], parents: Union[str, Variable, List[Union[str, Variable]]]):
        if isinstance(child, str):
            child = Variable(child)
        if isinstance(parents, (str, Variable)):
            parents = [parents]
        parents = [
            Variable(parent) if isinstance(parent, str) else parent
            for parent in parents
        ]
        super().__init__({
            'child': child,
            'parents': parents,
        })

    @property
    def child(self) -> Variable:
        return self['child']

    @property
    def parents(self) -> List[Variable]:
        return self['parents']

    def to_latex(self) -> str:
        parents = ','.join(p.to_latex() for p in self.parents)
        return f'{self.child.to_latex()}|{parents}'

    def given(self, parents: Union[str, 'Variable', List[Union[str, 'Variable']]]) -> 'Condition':
        if isinstance(parents, (str, Variable)):
            parents = [parents]
        return Condition(
            child=self.child,
            parents=[*self.parents, *parents]
        )

    def __or__(self, parents: Union[str, 'Variable', List[Union[str, 'Variable']]]) -> 'Condition':
        return self.given(parents)


class Probability(dict):
    def __init__(self, probability: Condition):
        super().__init__({'probability': probability})

    @property
    def probability(self) -> Condition:
        return self["probability"]

    def __truediv__(self, other) -> 'Frac':
        return Frac(self, other)

    def to_latex(self) -> str:
        return f'P({self.probability.to_latex()})'


P = Probability


class Sum(dict):
    def __init__(
        self,
        ranges: Union[None, Variable, List[Variable]],
        expressions: Union[Probability, 'Expression', List[Union[Probability, 'Expression']]],
    ):
        if ranges is None:
            ranges = []
        elif isinstance(ranges, Variable):
            ranges = [ranges]
        if not isinstance(expressions, list):  # TODO make more specific later
            expressions = [expressions]
        super().__init__({
            'sum': {
                'ranges': ranges,
                'expressions': expressions,
            },
        })

    @property
    def ranges(self) -> List[Variable]:
        return self['sum']['ranges']

    @property
    def expressions(self) -> List[Union[Probability, 'Expression']]:
        return self['sum']['expressions']

    def __truediv__(self, other) -> 'Frac':
        return Frac(self, other)

    def to_latex(self) -> str:
        ranges = ','.join(r.to_latex() for r in self.ranges)
        summands = ' '.join(e.to_latex() for e in self.expressions)
        return f'[ sum_{{{ranges}}} {summands} ]'


class Frac(dict):
    def __init__(self, numerator: 'Expression', denominator: 'Expression'):
        super().__init__({
            'frac': {
                'numerator': numerator,
                'denominator': denominator,
            },
        })

    @property
    def numerator(self) -> 'Expression':
        return self['frac']['numerator']

    def denominator(self) -> 'Expression':
        return self['frac']['denominator']

    def to_latex(self) -> str:
        return f"{self.numerator} / {self.denominator()}"


Expression = Union[Probability, Sum, Frac]

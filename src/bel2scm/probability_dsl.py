# -*- coding: utf-8 -*-

from typing import List, Optional, Union

__all__ = [
    'Variable',
    'Condition',
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

    def to_latex(self) -> str:
        if 'index' in self:
            return f'{self["name"]}_{self["index"]}'
        return self["name"]


class Condition(dict):
    def __init__(self, *, child: Variable, parents: List[Variable]):
        super().__init__({
            'child': child,
            'parents': parents,
        })

    def to_latex(self) -> str:
        parents = ','.join(p.to_latex() for p in self['parents'])
        return f'{self["child"].to_latex()}|{parents}'


class Probability(dict):
    def __init__(self, probability: Condition):
        super().__init__({'probability': probability})

    def to_latex(self) -> str:
        return f'P({self["probability"].to_latex()})'


class Sum(dict):
    def __init__(self, ranges: List[Variable], expressions: List[Union[Probability, 'Expression']]):
        super().__init__({
            'sum': {
                'ranges': ranges,
                'expressions': expressions,
            }
        })

    def to_latex(self) -> str:
        ranges = ','.join(r.to_latex() for r in self['sum']['ranges'])
        summands = ' '.join(e.to_latex() for e in self['sum']['expressions'])
        return f'[ sum_{{{ranges}}} {summands} ]'


class Frac(dict):
    def __init__(self, numerator: List['Expression'], denominator: List['Expression']):
        super().__init__({
            'frac': {
                'numerator': numerator,
                'denominator': denominator,
            }
        })

    def to_latex(self) -> str:
        return f"{self['frac']['numerator']} / {self['frac']['denominator']}"


Expression = Union[Condition, Sum, Frac]

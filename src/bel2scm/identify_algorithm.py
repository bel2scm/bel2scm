from typing import Optional

from bel2scm.probability_dsl import Expression
from bel2scm.probability_graph import MixedGraph

__all__ = [
    'identify',
]


def identify(graph: MixedGraph) -> Optional[Expression]:
    """Get an expression from the graph or return None."""
    raise NotImplementedError

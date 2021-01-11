import unittest

from bel2scm.identify_algorithm import identify
from bel2scm.probability_dsl import P, Sum, X, Y, Z
from bel2scm.probability_graph import NxMixedGraph

P_XY = P(X, Y, Z)
P_XYZ = P(X, Y, Z)


class TestIdentify(unittest.TestCase):
    """Test cases from https://github.com/COVID-19-Causal-Reasoning/Y0/blob/master/ID_whittemore.ipynb."""

    def test_figure_2a(self):
        """Test Figure 2A."""
        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Y')

        self.assertEqual(P_XY / Sum[Y](P_XY), identify(graph))

    def test_figure_2b(self):
        """Test Figure 2B."""
        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Y')
        graph.add_directed_edge('X', 'Z')
        graph.add_directed_edge('Z', 'Y')
        graph.add_undirected_edge('Y', 'Z')

        self.assertEqual(
            Sum[Z](Sum[Y](P_XY) / (Sum[Z](Sum[Y](P_XY))) * (P_XY / Sum[Y](P_XY))),
            identify(graph),
        )

    def test_figure_2c(self):
        """Test Figure 2C."""
        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Y')
        graph.add_directed_edge('Z', 'X')
        graph.add_directed_edge('Z', 'Y')
        graph.add_undirected_edge('Y', 'Z')

        self.assertEqual(
            Sum[Z](Sum[X, Y](P_XYZ) / (Sum[Z](Sum[X, Y](P_XYZ))) * (P_XYZ / Sum[Y](P_XYZ))),
            identify(graph),
        )

    def test_figure_2d(self):
        """Test Figure 2D."""
        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Y')
        graph.add_directed_edge('Z', 'X')
        graph.add_directed_edge('Z', 'Y')
        graph.add_undirected_edge('X', 'Z')

        self.assertEqual(
            Sum[Z](Sum[X, Y](P_XYZ) * P_XYZ / Sum[Y](P_XYZ)),
            identify(graph),
        )

    def test_figure_2e(self):
        """Test Figure 2E."""
        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Z')
        graph.add_directed_edge('Z', 'Y')
        graph.add_undirected_edge('X', 'Y')

        self.assertEqual(
            (
                Sum[Z](Sum[Y](P_XYZ) / Sum[Z](Sum[Y](P_XYZ)))
                * Sum[X](P_XYZ * Sum[Y, Z](P_XYZ) / Sum[Y](P_XYZ) / Sum[X](Sum[Y, Z](P_XYZ)))
            ),
            identify(graph),
        )

from abc import ABC, abstractmethod

import networkx as nx


class MixedGraph(ABC):
    """A graph that can contain both directed and undirected nodes."""

    @abstractmethod
    def add_directed_edge(self, u, v):
        """Add a directed edge to the graph."""
        raise NotImplementedError

    @abstractmethod
    def add_undirected_edge(self, u, v):
        """Add an undirected edge to the graph."""
        raise NotImplementedError


# TODO the actual abstract implementation should evolve to meet what is
#  required to implement the identify() algorithm.

class NxMixedGraph(MixedGraph):
    def __init__(self):
        self.directed = nx.DiGraph()
        self.undirected = nx.Graph()

    def add_directed_edge(self, u, v, **attr):
        self.directed.add_edge(u, v, **attr)
        self.undirected.add_node(u)
        self.undirected.add_node(v)

    def add_undirected_edge(self, u, v, **attr):
        self.undirected.add_edge(u, v, **attr)
        self.directed.add_node(u)
        self.directed.add_node(v)
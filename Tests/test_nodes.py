import pytest
from Neuirps_BEL2SCM.node import Node

## creating test node

test_node = Node()
test_node.root = True
test_node.name = "bp(T)"
test_node.children = ["a(C)"]
test_node.parent_relations = []
test_node.child_relations = ['increases']
test_node.node_type = "bp"
test_node.node_label = "process"
test_node.children_type = ["a"]
test_node.children_label = ["abundance"]
test_node.parents = []
test_node.parent_type = []
test_node.parent_label = []


@pytest.fixture
def empty_node():
    '''Returns empty node object'''
    return Node()

def test_default_node(empty_node):
    assert empty_node.root == True
    assert empty_node.name == ""
    assert empty_node.children == []
    assert empty_node.parent_relations == []
    assert empty_node.child_relations == []
    assert empty_node.node_type == ""
    assert empty_node.node_label == ""
    assert empty_node.children_type == []
    assert empty_node.children_label == []
    assert empty_node.parents == []
    assert empty_node.parent_type == []
    assert empty_node.parent_label == []


def test_node_info():
    '''Returns a Wallet instance with a balance of 20'''
    t = Node()
    t.get_node_information("bp(T)", "a(C)", "increases")
    assert t == test_node


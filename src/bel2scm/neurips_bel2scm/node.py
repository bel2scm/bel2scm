import collections

import torch
from bel2scm.neurips_bel2scm.constants import VALID_RELATIONS, LABEL_DICT


class Node:
    """
    Node contains it's type, label, orderedDict of its children and parent type and label.
    """
    def __init__(self):
        # root is True by default. We change this variable in self.update_parent_information_in_child_node()
        self.root = True
        # Node name
        self.name = ""
        # <ChildName, <Name, Relation, Label>>
        self.children_info = collections.OrderedDict()
        # <ParentName, <Name, Relation, Label>>
        self.parent_info = collections.OrderedDict()
        # Extract type from self.name and get label from LABEL_DICT
        self.node_label = ""


    def get_node_information(self, sub, obj, rel):
        # If relation is valid
        if rel in VALID_RELATIONS:

            # Update name
            self.name = sub

            # Get parent and child type
            p = sub
            c = obj
            ptype = self._get_type(p)
            ctype = self._get_type(c)

            # Get Node and Child label
            # Labels will be Others unless they find a match in LABEL_DICT
            self.node_label = "Others"
            child_label = "Others"

            for label in LABEL_DICT:
                if ptype in LABEL_DICT[label]:
                    self.node_label = label
            for label in LABEL_DICT:
                if ctype in LABEL_DICT[label]:
                    child_label = label

            # Add Child to children_info
            child_dict = {
                "name": c,
                "relation": rel,
                "label": child_label
            }

            if obj not in self.children_info:
                self.children_info[obj] = child_dict
            else:
                raise Exception("Invalid Bel graph! May be duplicate statements..")

        # Else, skip the statement
        else:
            print(
                "get_node_information()::: Subject {0}, Object {1} is skipped because relation {2} is not a valid "
                "relation.".format(
                    sub, obj, rel))

    def update_child_information_in_parent_node(self, obj, rel):

        # If relation is valid
        if rel in VALID_RELATIONS:

            # Get Child type
            c = obj
            ctype = self._get_type(c)

            # Get Child label
            # child_label will be Others unless they find a match in LABEL_DICT
            child_label = 'Others'

            for label in LABEL_DICT:
                if ctype in LABEL_DICT[label]:
                    child_label = label

            # Add child to children_info
            child_dict = {
                "name": c,
                "relation": rel,
                "label": child_label
            }

            if obj not in self.children_info:
                self.children_info[obj] = child_dict
            else:
                raise Exception("Invalid Bel graph! May be duplicate statements..")

        # Else, skip the child info update
        else:
            print(
                "update_child_information_in_parent_node()::: Object {0} is skipped because relation {1} is not a "
                "valid relation. ".format(obj, rel))

    def update_parent_information_in_child_node(self, sub, rel):

        # When we encounter a parent for a node, we set root = false.
        self.root = False

        # If relation is valid
        if rel in VALID_RELATIONS:

            # Get parent type
            p = sub
            ptype = self._get_type(p)

            # Get parent label
            # parent_label will be "Others" unless they find a match in LABEL_DICT
            parent_label = "Others"
            for label in LABEL_DICT:
                if ptype in LABEL_DICT[label]:
                    parent_label = label
            if self.node_label == "":
                ctype = self._get_type(self.name)
                self.node_label = "Others"
                for label in LABEL_DICT:
                    if ctype in LABEL_DICT[label]:
                        self.node_label = label

            # Add parent to parent_info
            parent_dict = {
                "name": p,
                "relation": rel,
                "label": parent_label
            }

            if sub not in self.parent_info:
                self.parent_info[sub] = parent_dict
            else:
                raise Exception("Invalid Bel graph! May be duplicate statements..")
        else:
            print(
                "update_parent_information_in_child_node()::: Subject {0} is skipped because relation {1} is not a "
                "valid relation. ".format(sub, rel))

    def _get_type(self, str):
        idx = str.find('(')

        if idx >= 0:
            return str[:idx]
        else:
            raise Exception("_get_type(): Type not found for node {0}!".format(str))


class NodeData:
    def __init__(self, features, target):
        self.features = self._get_processed_features(features)
        self.target = torch.tensor(target)

    def _get_processed_features(self, features):
        pass
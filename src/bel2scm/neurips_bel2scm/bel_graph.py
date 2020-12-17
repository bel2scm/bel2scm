import json
import pandas as pd

from bel2scm.neurips_bel2scm.node import *


# a generic function to take BEL statements as input
# in any form and return a data structure as output
# in desired format

# Created with the assumptions that inputs are bel statements of some sort


class BelGraph:
    """
    It loads all the nodes from given input BEL statements.
    """

    def __init__(self, file_type, file_name, data_file_path):
        self.file_type = file_type
        self.file_name = file_name
        self.data_file_path = data_file_path

        # Dictionary<str, Node>
        self.nodes = dict()
        # Dictionary<str, dict<features_df(parents), target(node)>>
        self.node_data = dict()
        # Dictionary<str, list>
        self.parent_name_list_for_nodes = dict()

    def parse_input_to_construct_graph(self):
        file_type = self.file_type
        file_name = self.file_name
        if file_type == "str_list":
            self.construct_graph_from_str_list(file_name)
        elif file_type == "bel_graph":
            self.construct_graph_from_bel_graph(file_name)
        elif file_type == "jgf_file":
            self.construct_graph_from_jgf_file()
        elif file_type == "nanopub_file":
            self.construct_graph_from_nanopub_file()
        else:
            raise Exception("Invalid file type!")

        if self.is_cyclic():
            raise Exception("Graph contains cycles!")

    def construct_graph_from_str_list(self, file_name):
        """
        Disc
        Args:
            file_name:

        Returns:

        """
        # extracting relevant information from string list format
        for item in file_name:
            sub_ind = item.find('=')
            sub_temp = item[:sub_ind - 1]
            obj_temp = item[sub_ind + 3:]
            rel_temp = item[sub_ind:sub_ind + 2]
            # keeping only increases/decreases type of edges
            if sub_temp in self.nodes:
                self.nodes[sub_temp].update_child_information_in_parent_node(obj_temp, rel_temp)
            else:
                sub_node = Node()
                sub_node.get_node_information(sub_temp, obj_temp, rel_temp)
                self.nodes[sub_temp] = sub_node

            if obj_temp in self.nodes:
                self.nodes[obj_temp].update_parent_information_in_child_node(sub_temp, rel_temp)
            else:
                obj_node = Node()
                obj_node.name = obj_temp
                obj_node.update_parent_information_in_child_node(sub_temp, rel_temp)
                self.nodes[obj_temp] = obj_node

    def construct_graph_from_bel_graph(self, file_name):
        """

        Args:
            file_name:

        Returns:

        """
        # extracting relevant information from pybel format
        for item in file_name.edges:
            edge_temp = file_name.get_edge_data(item[0], item[1], item[2])
            sub_temp = str(item[0]).replace('"', '')
            obj_temp = str(item[1]).replace('"', '')
            rel_temp = edge_temp['relation']

            if sub_temp in self.nodes:
                self.nodes[sub_temp].update_child_information_in_parent_node(obj_temp, rel_temp)
            else:
                sub_node = Node()
                sub_node.get_node_information(sub_temp, obj_temp, rel_temp)
                self.nodes[sub_temp] = sub_node

            if obj_temp in self.nodes:
                self.nodes[obj_temp].update_parent_information_in_child_node(sub_temp, rel_temp)
            else:
                obj_node = Node()
                obj_node.name = obj_temp
                obj_node.update_parent_information_in_child_node(sub_temp, rel_temp)
                self.nodes[obj_temp] = obj_node

    def construct_graph_from_jgf_file(self):
        """

        Args:
            file_name:

        Returns:

        """
        with open(self.file_name) as file1:
            loaded_jgf = json.load(file1)

        for item in loaded_jgf['graph']['edges']:
            sub_temp = item['source']
            obj_temp = item['target']
            rel_temp = item['relation']
            if sub_temp in self.nodes:
                self.nodes[sub_temp].update_child_information_in_parent_node(obj_temp, rel_temp)
            else:
                sub_node = Node()
                sub_node.get_node_information(sub_temp, obj_temp, rel_temp)
                self.nodes[sub_temp] = sub_node

            if obj_temp in self.nodes:
                self.nodes[obj_temp].update_parent_information_in_child_node(sub_temp, rel_temp)
            else:
                obj_node = Node()
                obj_node.name = obj_temp
                obj_node.update_parent_information_in_child_node(sub_temp, rel_temp)
                self.nodes[obj_temp] = obj_node

    def construct_graph_from_nanopub_file(self):
        """
        Returns:

        """

        try:
            with open(self.file_name) as json_text:
                loaded_nanopub = json.load(json_text)
        except:
            raise Exception("Bel file not found!!")

        for item in loaded_nanopub[0]['nanopub']['assertions']:
            subject = item['subject']
            object = item['object']
            relation = item['relation']

            # If subject node is present in node dict, then update child information for subject node.
            if subject in self.nodes:
                self.nodes[subject].update_child_information_in_parent_node(object, relation)

            # Else, add subject node to node dict.
            else:
                sub_node = Node()
                sub_node.get_node_information(subject, object, relation)
                self.nodes[subject] = sub_node

            # If object is present in node dict, then update parent information in object node.
            if object in self.nodes:
                self.nodes[object].update_parent_information_in_child_node(subject, relation)

            # Else, add object node to node dict.
            else:
                obj_node = Node()
                obj_node.name = object
                obj_node.update_parent_information_in_child_node(subject, relation)
                self.nodes[object] = obj_node

        return self.nodes

    def get_nodes_with_no_parents(self):
        # new dictionary to add nodes with no parents
        node_dict_with_no_parents = {}

        if len(self.nodes) > 0:
            for key, node in self.nodes.items():
                if node.root:
                    node_dict_with_no_parents[key] = node
            return node_dict_with_no_parents
        else:
            raise Exception("Empty graph.")

    def prepare_and_assign_data(self):
        """
        This function iterates through nodes and prepares feature and target values for each node.
        CAUTION: Data headers should have same name as BEL subjects or objects.
        Args:
            data_file_path:

        Returns: node_data dictionary <str,

        """
        try:
            data = pd.read_csv(self.data_file_path)
        except:
            raise Exception("Data file not found!")

        data_headers = data.columns.tolist()

        for node_str, node in self.nodes.items():
            if node_str in data_headers:
                if node.root:
                    features = pd.DataFrame()
                    target = self._get_single_node_data(node, data)
                else:
                    features, target = self._get_non_root_data(node, data)
            else:
                raise Exception("Invalid data! Some columns are not in the bel graph.")

            self.node_data[node_str] = {
                "features": features,
                "target": target
            }

    def _get_single_node_data(self, node, data):
        try:
            return data[node.name]
        except:
            return Exception("Node " + node.name + " is not available in data.")

    def _get_non_root_data(self, node, data):
        """
        Args:
            data: original data with all columns

        Returns: dataframe features, Series target
        """

        # Get the child data
        child_data = self._get_single_node_data(node, data)

        # Get parent list
        valid_parent_name_list = [parent_name for parent_name, parent_info in node.parent_info.items()]
        self.parent_name_list_for_nodes[node.name] = valid_parent_name_list

        try:
            parent_data = data[valid_parent_name_list]
        except:
            raise Exception("Exception: one of the parent not available in data set")

        return parent_data, child_data


    def is_cyclic(self):
        visited_nodes = list()
        recursion_stack = list()
        for node in self.nodes.keys():
            if node not in visited_nodes:
                if self.is_cyclic_recursion(self.nodes[node], visited_nodes, recursion_stack):
                    return True
        return False

    def is_cyclic_recursion(self, node_obj, visited_nodes, recursion_stack):
        visited_nodes.append(node_obj.name)
        recursion_stack.append(node_obj.name)

        # Recur for all neighbours
        # if any neighbour is visited and in recursion_stack then graph is cyclic

        for child_name in [value["name"] for (key, value) in node_obj.children_info.items()]:
            if child_name not in visited_nodes:
                if self.is_cyclic_recursion(self.nodes[child_name], visited_nodes, recursion_stack):
                    return True
            elif child_name in recursion_stack:
                return True

        # The node needs to be popped from recursion stack before function ends
        recursion_stack.remove(node_obj.name)
        return False


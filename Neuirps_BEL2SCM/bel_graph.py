import json

from Neuirps_BEL2SCM.node import *


## a generic function to take BEL statements as input
## in any form and return a data structure as output
## in desired format

## Created with the assumptions that inputs are bel statements of some sort


class BelGraph:
    '''
    It loads all the nodes from given input BEL statements.
    '''
    # Dictionary<str, Node>
    nodes = dict()

    def __init__(self, file_type, file_name):
        self.file_type = file_type
        self.file_name = file_name

    def construct_graph_from_str_list(self, file_name):
        '''
        Disc
        Args:
            file_name:

        Returns:

        '''
        # extracting relevant information from string list format
        for item in file_name:
            sub_ind = item.find('=')
            sub_temp = item[:sub_ind - 1]
            obj_temp = item[sub_ind + 3:]
            rel_temp = item[sub_ind:sub_ind + 2]
            ## keeping only increases/decreases type of edges
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
        '''

        Args:
            file_name:

        Returns:

        '''
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

    def construct_graph_from_jgf_file(self, file_name):
        '''

        Args:
            file_name:

        Returns:

        '''
        with open(file_name) as file1:
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
        '''

        Returns:

        '''
        with open(self.file_name) as json_text:
            loaded_nanopub = json.load(json_text)

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

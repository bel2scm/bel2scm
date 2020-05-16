from Neuirps_BEL2SCM.nodes import *
import json

## a generic function to take BEL statements as input
## in any form and return a data structure as output
## in desired format

## Created with the assumptions that inputs are bel statements of some sort


def get_nodes(file_type, file_name):
    nodes = dict()
    if file_type == 'str_list':
        ## extracting relevant information from string list format
        for item in file_name:
            sub_ind = item.find('=')
            sub_temp = item[:sub_ind - 1]
            obj_temp = item[sub_ind + 3:]
            rel_temp = item[sub_ind:sub_ind + 2]
            ## keeping only increases/decreases type of edges
            if sub_temp in nodes:
                nodes[sub_temp].update_parent_node(obj_temp, rel_temp)
            else:
                sub_node = Node()
                sub_node.get_node_information(sub_temp, obj_temp, rel_temp)
                nodes[sub_temp] = sub_node

            if obj_temp in nodes:
                nodes[obj_temp].update_child_node(sub_temp, rel_temp)
            else:
                obj_node = Node()
                obj_node.name = obj_temp
                obj_node.update_child_node(sub_temp, rel_temp)
                nodes[obj_temp] = obj_node

    elif file_type == 'bel_graph':
        ## extracting relevant information from pybel format
        for item in bel_graph.edges:
            edge_temp = bel_graph.get_edge_data(item[0], item[1], item[2])
            sub_temp = str(item[0]).replace('"', '')
            obj_temp = str(item[1]).replace('"', '')
            rel_temp = edge_temp['relation']

            if sub_temp in nodes:
                nodes[sub_temp].update_parent_node(obj_temp, rel_temp)
            else:
                sub_node = Node()
                sub_node.get_node_information(sub_temp, obj_temp, rel_temp)
                nodes[sub_temp] = sub_node

            if obj_temp in nodes:
                nodes[obj_temp].update_child_node(sub_temp, rel_temp)
            else:
                obj_node = Node()
                obj_node.name = obj_temp
                obj_node.update_child_node(sub_temp, rel_temp)
                nodes[obj_temp] = obj_node


    elif file_type == 'jgf_file':
        file1 = open(file_name)
        loaded_jgf = json.load(file1)

        for item in loaded_jgf['graph']['edges']:
            sub_temp = item['source']
            obj_temp = item['target']
            rel_temp = item['relation']
            if sub_temp in nodes:
                nodes[sub_temp].update_parent_node(obj_temp, rel_temp)
            else:
                sub_node = Node()
                sub_node.get_node_information(sub_temp, obj_temp, rel_temp)
                nodes[sub_temp] = sub_node

            if obj_temp in nodes:
                nodes[obj_temp].update_child_node(sub_temp, rel_temp)
            else:
                obj_node = Node()
                obj_node.name = obj_temp
                obj_node.update_child_node(sub_temp, rel_temp)
                nodes[obj_temp] = obj_node

    elif file_type == 'nanopub_file':
        file1 = open(file_name)
        loaded_nanopub = json.load(file1)
        for item in loaded_nanopub[0]['nanopub']['assertions']:
            sub_temp = item['subject']
            obj_temp = item['object']
            rel_temp = item['relation']
            if sub_temp in nodes:
                nodes[sub_temp].update_parent_node(obj_temp, rel_temp)
            else:
                sub_node = Node()
                sub_node.get_node_information(sub_temp, obj_temp, rel_temp)
                nodes[sub_temp] = sub_node

            if obj_temp in nodes:
                nodes[obj_temp].update_child_node(sub_temp, rel_temp)
            else:
                obj_node = Node()
                obj_node.name = obj_temp
                obj_node.update_child_node(sub_temp, rel_temp)
                nodes[obj_temp] = obj_node

    return nodes

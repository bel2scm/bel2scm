import numpy as np
import scipy as sp
import networkx as nx

from scipy import stats

import pybel as pb
import json

import torch
import pyro

from . import graph_node as gn

# create a class of causal graphs

class cg_graph():
    """define a superclass for causal graphs"""
    
    def __init__(self):
        return
    
    
    def proc_data(self,graph_type,type_dict={}):
        """ take the list of edges and entities (i.e., nodes) and process that information to produce
        parent -> children and child -> parent mappings
        initialize all of the nodes of the causal graph"""
        
        self.graph_type = graph_type
        n_nodes = len(self.entity_list)
        self.n_nodes = n_nodes
        
        
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.entity_list)
        self.graph.add_edges_from([[item[0],item[1]] for item in self.edge_list])
        
        #adj_mat = np.zeros((self.n_nodes,self.n_nodes),dtype=int)

        #for item in self.edge_list:
            #out_ind = self.entity_list.index(item[0])
            #in_ind = self.entity_list.index(item[1])
            #adj_mat[out_ind,in_ind] = 1

        #self.adj_mat = adj_mat
        
        #graph_temp = nx.DiGraph(adj_mat)
        #dict_temp = {}
        
        #for i in range(0,n_nodes):
            #dict_temp[i] = self.entity_list[i]
            
        #self.graph = nx.relabel_nodes(graph_temp, dict_temp)
        
        # check to make sure that it's a DAG
        if nx.algorithms.dag.is_directed_acyclic_graph(self.graph):
            print('The causal graph is a acyclic')
            
        else:
            print('The causal graph has cycles -- this is a problem')
            
            # identify edges that, if removed, would lead to the causal graph being acyclic
            c_bas = list(nx.simple_cycles(self.graph))
            print('There are ' + str(len(c_bas)) + ' simple cycles')
            
            cycle_edge_list = []
            
            for item in c_bas:
                for i in range(0,len(item)):
                    sub_temp = self.entity_list[item[i-1]]
                    obj_temp = self.entity_list[item[i]]
                    rel_temp = [item2[2] for item2 in edge_list if (item2[0] == sub_temp and item2[1] == obj_temp)]
                    cycle_edge_list += [[sub_temp,obj_temp,item2] for item2 in rel_temp]
            print('Cycle edges:')
            for item in cycle_edge_list:
                print(item)
        
        self.cond_list = []
        
        self.sample_dict = {}
        
        #self.parent_ind_list = []
        #self.child_ind_list = []
        self.parent_dict = {}
        self.child_dict = {}
        
        #self.parent_ind_list = [np.where(self.adj_mat[:,i] > 0)[0] for i in range(0,self.n_nodes)]
        #self.child_ind_list = [np.where(self.adj_mat[i,:] > 0)[0] for i in range(0,self.n_nodes)]
        node_dict = {}
        
        for item in self.entity_list:
            self.parent_dict[item] = list(self.graph.predecessors(item))
            self.child_dict[item] = list(self.graph.successors(item))
            
            n_pars = len(self.parent_dict[item])
        
            if type_dict:
                node_type = type_dict[item]

            else:

                bel_dict = {}
                bel_dict['activity'] = ['activity','act','molecularActivity','ma']
                bel_dict['abundance'] = ['a','abundance','complex','complexAbundance','geneAbundance','g',
                    'microRNAAbundance','m','populationAbundance','pop','proteinAbundance','p','rnaAbundance','r',
                    'compositeAbundance','composite']
                bel_dict['reaction'] = ['reaction','rxn']
                bel_dict['process'] = ['biologicalProcess','bp']
                bel_dict['pathology'] = ['pathology','path']

                vartype_dict = {}
                vartype_dict['activity'] = 'Bernoulli'
                vartype_dict['abundance'] = 'Gamma'
                vartype_dict['reaction'] = 'Normal'
                vartype_dict['process'] = 'Bernoulli'
                vartype_dict['pathology'] = 'Bernoulli'
                
                ind_temp = item.find('(')
                str_temp = item[:ind_temp]
                node_type = ''
                
                for item in bel_dict:
                    if str_temp in bel_dict[item]:
                        node_type = vartype_dict[item]

                if node_type == '':
                    node_type = 'Normal'
                    print('BEL node type ' + str_temp + ' not known -- defaulting to Normal')

            if self.graph_type == 'Bayes':
                node_dict[item] = bayes_node(n_pars,item,node_type)
            elif self.graph_type == 'MLE':
                node_dict[item] = mle_node(n_pars,item,node_type)
            elif self.graph_type == 'SCM':
                node_dict[item] = scm_node(n_pars,item,node_type)
            else:
                print('node type ' + self.graph_type + 'not recognized -- defaulting to MLE')
                node_dict[item] = mle_node(n_pars,item,node_type)
        
        self.node_dict = node_dict
        
        return
        
    
    def remove_edge(self,edge_rem):
        """remove all of the edges in edge_rem from the causal graph"""
        
        for item in edge_rem:
            self.graph.remove_edge(item)
            
            ind_remove = [i for i in range(0,len(self.edge_list)) 
                if (self.edge_list[i][0] == edge_rem[0] and self.edge_list[i][1] == edge_rem[1])]
            for ind in ind_remove:
                self.edge_list.remove(self.edge_list[i])
            
        for item in self.entity_list:
            self.parent_dict[item] = list(graph_temp.predecessors(item))
            self.child_dict[item] = list(graph_temp.successors(item))
        return
    
    def add_confound(self,confound_pairs):
        """ add a list of pairs of nodes that share unobserved confounders"""
        
        graph_c = nx.Graph()
        graph_c.add_nodes_from(self.graph.nodes)
        graph_c.add_edges_from([tuple(item) for item in confound_pairs])

        self.graph_c = graph_c
        
        return
    
    def str_list(self,node_list):
        """ return a string listing the nodes in node_list - this is used in the ID and IDC algorithms """
        str_out = ''
        
        for item in node_list:
            str_out += item + ','

        return str_out[:-1]
    
    def d_sep(self,x,y,z,graph_in):
        # determine if all paths from x to y are d-separated by z in graph_temp
        
        # convert digraph to undirected graph for d-separation
        
        if graph_in:
            graph_temp = graph_in            
        else:
            graph_temp = self.graph.to_undirected()
            graph_temp.add_edges_from(self.graph_c.edges)
        
        # ensure that x, y, and z are disjoint
        if np.any([[item1 == item2 for item1 in x] for item2 in y]):
            print('x and y not disjoint')
            return
        
        if np.any([[item1 == item2 for item1 in x] for item2 in z]):
            print('x and z not disjoint')
            return
        
        if np.any([[item1 == item2 for item1 in z] for item2 in y]):
            print('y and z not disjoint')
            return
        
        
        # identify all paths from x to y
        path_list = []
        
        for item in x:      
            for path in nx.all_simple_paths(graph_temp, source=item, target=y):
                path_list.append(path)
                
        print(str(len(path_list)) + ' total paths')
        
        # iterate through paths
        for item in path_list:
            # if an element of z is in the path, path is d-separated
            # else, path is not d-separated, return False
            
            if not np.any([ind in item for ind in z]):
                return False
        
        # if all paths d-separated, return True
        
        return True
    
    def id_alg(self,y,x,p_in=[],graph_in=[]):
        # calculate P(y | do(x)) or return failure if this is not possible
        
        if np.any([item in y for item in x]):
            print('Error -- overlap between x and y')
            print(x)
            print(y)
            print(p_in)
            print(graph_in.nodes)
        
        if graph_in:
            graph_temp = graph_in
        else:
            graph_temp = nx.DiGraph(self.graph)
            
        if p_in:
            p_expr = p_in
        else:
            p_expr = 'P(' + self.str_list(graph_temp.nodes) + ')'
        
        y_anc = y.copy()
        
        # identify ancestors of y
        for item in y:
            set_temp = nx.algorithms.dag.ancestors(graph_temp,item)
            y_anc += [item2 for item2 in set_temp if item2 not in y_anc]
                    
        # identify all nodes in the graph        
        v_not_anc_y = [item for item in graph_temp.nodes if item not in y_anc]

        # remove edges to x
        graph_xbar = nx.DiGraph(graph_temp)
        for item in x:
            graph_xbar.remove_edges_from(list(graph_temp.in_edges(item)))
            
        y_anc_x_bar = y.copy()
        
        for item in y:
            set_temp = nx.algorithms.dag.ancestors(graph_xbar,item)
            y_anc_x_bar += [item2 for item2 in set_temp if item2 not in y_anc_x_bar]

        w_set = [item for item in graph_temp.nodes if item not in x and item not in y_anc_x_bar]
        
        # line 1
        if not x:
            # return sum over all non-y variables
            
            node_list = [item for item in graph_temp.nodes if item not in y]
            str_out = '[sum_{' + self.str_list(node_list) + '} ' + p_expr + ']'
            #print('Step 1')
            
            return str_out
            
        # line 2
        elif v_not_anc_y:
            
            x_temp = [item for item in y_anc if item in x]
            str_out = '[sum_{' + self.str_list(v_not_anc_y) + '} ' + p_expr + ']'
            graph_anc = graph_temp.subgraph(y_anc)
            
            #print('Begin Step 2')
            #print(v_not_anc_y)
            expr_out = self.id_alg(y,x_temp,str_out,graph_anc)
            #print('End Step 2')
            
            return expr_out
        
        # line 3
        elif w_set:
            #print('Begin Step 3')
            #print(w_set)
            expr_out = self.id_alg(y,x+w_set,p_expr,graph_temp)
            #print('End Step 3')
            
            return expr_out
        
        else:            
            # calculate graph C-components
            graph_temp_c = nx.Graph(self.graph_c.subgraph(graph_temp.nodes))
            graph_temp_c.remove_nodes_from(x)
            s_sets = [list(item) for item in nx.connected_components(graph_temp_c)]
            
            # line 4
            if len(s_sets) > 1:
                #print('Begin Step 4')
                #print(s_sets)
                node_list = [item for item in graph_temp.nodes if item not in y and item not in x]
                str_out = '[sum_{' + self.str_list(node_list) + '} '
                
                for item in s_sets:
                    v_s_set = [item2 for item2 in graph_temp.nodes if item2 not in item]
                    s_in = [item2 for item2 in item]
                    
                    if np.any([item2 in v_s_set for item2 in s_in]):
                        print('Error -- x/y overlap')
                        print(v_s_set)
                        print(s_in)
                    
                    str_out += self.id_alg(s_in,v_s_set,p_expr,graph_temp)
                        
                #print('End Step 4')
                str_out += ']'
                
                return str_out
            
            else:
                graph_temp_c_prime = self.graph_c.subgraph(graph_temp.nodes)
                
                s_sets_prime = [list(item) for item in nx.connected_components(graph_temp_c_prime)]
                
                # line 5
                if sorted(s_sets_prime[0]) == sorted(graph_temp.nodes):
                    
                    node_list = [ind for ind in s_sets2[0]]
                    node_list2 = [ind for ind in graph_temp.nodes if ind in s_sets2[0]]
                    
                    str_out = 'FAIL(' + self.str_list(node_list) + ',' + self.str_list(node_list2) + ')'
                    
                    #print('Step 5')
                    return str_out
                
                # line 6
                elif np.any([sorted(s_sets[0]) == sorted(item) for item in s_sets_prime]):
                            
                    node_list = [item for item in s_sets[0] if item not in y]
                    str_out = '[sum_{' + self.str_list(node_list) + '}'
                    
                    for item in s_sets[0]:
                        # identify parents of node i
                        parents = list(graph_temp.predecessors(item))
                        
                        str_out += 'P(' + item + '|' + self.str_list(parents) + ')'
                    #print(s_sets[0])
                    #print('Step 6')
                    return str_out + ']'
                
                # line 7
                elif np.any([np.all([item in item2 for item in s_sets[0]]) for item2 in s_sets_prime]):
                    ind = np.where([np.all([item in item2 for item in s_sets[0]]) 
                        for item2 in s_sets_prime])[0][0]
                    
                    graph_prime = graph_temp.subgraph(s_sets_prime[ind])
                    x_prime = [item for item in s_sets_prime[ind] if item in x]
                    str_out = ''
                    
                    for item in s_sets_prime[ind]:
                        
                        pred = list(nx.algorithms.dag.ancestors(graph_temp,item))
                        par_set = [item2 for item2 in pred if item2 in s_sets_prime[ind]]
                        par_set += [item2 for item2 in pred if item2 not in s_sets_prime[ind]]
                        str_out += 'P(' + item + '|' + self.str_list(par_set) + ')'
                        
                    #print('Begin Step 7')
                    #print((s_sets[0],s_sets_prime[ind]))
                    
                    if np.any([item2 in x_prime for item2 in y]):
                        print('Error -- x/y overlap')
                        print(x_prime)
                        print(y)
                    
                    expr_out = self.id_alg(y,x_prime,str_out,graph_prime)
                    #print('End Step 7')
                    
                    return expr_out
                
                else:
                    
                    print('error')
                    return ''

    def idc_alg(self,y,x,z,p_in=[],graph_in=[]):
        # calculate P(y | do(x), z) or return failure if this is not possible
        
        if np.any([item in y for item in x]):
            print('Error -- overlap between x and y')
            print(x)
            print(y)
            print(p_in)
            print(graph_in.nodes)
            
        if np.any([item in y for item in z]):
            print('Error -- overlap between z and y')
            print(z)
            print(y)
            print(p_in)
            print(graph_in.nodes)
            
        if np.any([item in z for item in x]):
            print('Error -- overlap between x and z')
            print(x)
            print(z)
            print(p_in)
            print(graph_in.nodes)
        
        if graph_in:
            graph_temp = graph_in
        else:
            graph_temp = self.graph
            
        if p_in:
            p_expr = p_in
        else:
            p_expr = 'P(' + self.str_list(graph_temp.nodes) + ')'

        digraph_xbar = nx.DiGraph(graph_temp)
        for item in x:
            digraph_xbar.remove_edges_from(graph_temp.in_edges(item)) 
            
        # identify edges from z
        z_inds = [ind for ind in graph_temp.nodes if ind in z]
        z_edges = [list(graph_temp.out_edges(item2)) for item2 in z_inds]
        
        # check for d-separation
        for item in z:
            digraph_xbar_zbar = nx.DiGraph(digraph_xbar)
            digraph_xbar_zbar.remove_edges_from(graph_temp.out_edges(item))
            graph_xbar_zbar = digraph_xbar_zbar.to_undirected()
            
            graph_xbar_zbar.add_edges_from(self.graph_c.subgraph(graph_temp.nodes).edges)
                
            # calculate d-separation
            d_sep = self.d_sep(y,[item],[item2 for item2 in x+z if item2 != item],graph_xbar_zbar)
            
            if d_sep:
                
                return self.idc_alg(y,x+[item],[item2 for item2 in z if item2 != item],p_expr,graph_temp)
            
        p_prime = self.id_alg(y+z,x,p_expr,graph_temp)
        
        str_out = '[' + p_prime + ']/[ sum_{' + self.str_list(y) + '}' + p_prime + ']'
        
        return str_out
    
    def make_pw_graph(self,do_in,graph_in=[]):
        # create the parallel-world graph of subgraph of graph_in or self.graph
        
        if graph_in:
            graph_temp = nx.DiGraph(graph_in)
            conf_temp = self.graph_c.subgraph(graph_temp.nodes)
        else:
            graph_temp = nx.DiGraph(self.graph)
            conf_temp = nx.Graph(self.graph_c)
        
        # record all nodes with unobserved confounders in the original graph
        vars_with_conf = []
        for item in conf_temp.edges:
            if item[0] not in vars_with_conf:
                vars_with_conf.append(item[0])
            if item[1] not in vars_with_conf:
                vars_with_conf.append(item[1])
        
        # confounding nodes corresponding to duplicate pw-graph nodes
        conf_nodes = ['U^{' + item + '}' for item in graph_temp.nodes if item not in vars_with_conf]
        
        # confounding nodes corresponding to confounders in the original graph
        conf_nodes += ['U^{' + item[0] + ',' + item[1] + '}' for item in conf_temp.edges]
        
        graph_out = nx.DiGraph(graph_temp)
        graph_out.add_nodes_from(conf_nodes)
        # add confounders - now a digraph because we've added nodes for each confounder
        conf_out = nx.DiGraph()
        conf_out.add_nodes_from(graph_out.nodes)
        
        # add confounding edges
        conf_edges_add = [('U^{' + item + '}',item) for item in graph_temp.nodes if item not in vars_with_conf]
        conf_edges_add += [('U^{' + item[0] + ',' + item[1] + '}',item[0]) for item in conf_temp.edges]
        conf_edges_add += [('U^{' + item[0] + ',' + item[1] + '}',item[1]) for item in conf_temp.edges]
        conf_out.add_edges_from(conf_edges_add)
        
        # add duplicate edges and nodes
        for item in do_in:
            str_temp = self.str_list(item[1])
            
            # create nodes and edges corresponding to duplicate graph
            # don't add edges going into do-variable nodes
            node_list = [item2 + '_{' + str_temp + '}' for item2 in graph_temp.nodes]
            edge_list = [(item2[0] + '_{' + str_temp + '}',item2[1] + '_{' + str_temp + '}') 
                for item2 in graph_temp.edges if item2[1] not in item[1]]
            
            # add duplicate nodes and edges to the underlying digraph
            graph_out.add_nodes_from(node_list)
            graph_out.add_edges_from(edge_list)
            
            # create confounder edges for duplicate variables
            conf_edge_list = [('U^{' + item2 + '}',item2 + '_{' + str_temp + '}') 
                for item2 in graph_temp.nodes if item2 not in vars_with_conf and item2 not in item[1]]

            # create confounder edges for confounders from the original graph
            conf_edge_list += [('U^{' + item2[0] + ',' + item2[1] + '}',item2[0] + '_{' + str_temp + '}') 
                for item2 in conf_temp.edges if item2[0] not in item[1]]            
            conf_edge_list += [('U^{' + item2[0] + ',' + item2[1] + '}',item2[1] + '_{' + str_temp + '}') 
                for item2 in conf_temp.edges if item2[1] not in item[1]]
            
            # add duplicate nodes and confounder edges to confounding digraph
            conf_out.add_nodes_from(node_list)
            conf_out.add_edges_from(conf_edge_list)

        
        return graph_out,conf_out
    
    def make_cf_graph(self,do_in,obs_in=[],graph_in=[]):
        # create the counterfactual graph of subgraph of graph_in or self.graph
        
        # add in error checking for consistency (target not in counterfact and neither in obs_in)
        
        if graph_in:
            graph_temp = nx.DiGraph(graph_in)
            conf_temp = self.graph_c.subgraph(graph_temp.nodes)
        else:
            graph_temp = nx.DiGraph(self.graph)
            conf_temp = nx.Graph(self.graph_c)
        
        gamma_list = self.conv_to_gamma(do_in,obs_in)                
                
        # create parallel worlds graph
        graph_out,conf_out = self.make_pw_graph(do_in,graph_in)

        
        # iterate through nodes and merge variables
        node_list = [item for item in graph_temp.nodes if graph_temp.in_degree(item) == 0]
        traversed_nodes = []
        
        while sorted(traversed_nodes) != sorted(graph_temp.nodes) and node_list:
                       
            # start with the first item of node_list
            node_temp = node_list[0]            
            
            # identify parents of node_temp
            par_temp = [item[0] for item in graph_out.edges if item[1] == node_temp]
            
            # cycle through all of the duplicate graphs and merge nodes
            
            for item in do_in:
                str_temp = self.str_list(item[1])
                # identify the node to check
                node_temp2 = node_temp + '_{' + str_temp + '}'
                
                # see if all the parents are identical in graph_out
                graph_pars = sorted(par_temp) == sorted(
                    [item[0] for item in graph_out.edges if item[1] == node_temp2])
                
                # see if all the parents are identical in conf_out
                conf_pars = sorted([item[0] for item in conf_out.edges if item[1] == node_temp]
                    ) == sorted([item[0] for item in conf_out.edges if item[1] == node_temp2])
                
                # identify all of the parents that are not the same
                par_diff = [item2[0] for item2 in graph_out.edges 
                    if item2[1] == node_temp2 and item2[0] not in par_temp]
                
                
                # if the parents all match up, merge the nodes
                # elif the node being checked has all of the nodes in par_diff as do-variables, do the merge
                # identify cases where the parents don't match exactly but the values line up
                # B_{} -> A_{} => B -> A if B_{} and A_{} are both observed
                
                if graph_pars and conf_pars:
                    graph_out = nx.contracted_nodes(graph_out,node_temp,node_temp2,self_loops=False)
                    conf_out = nx.contracted_nodes(conf_out,node_temp,node_temp2,self_loops=False)
                    
                    if node_temp2 in gamma_list:
                        
                        # check for inconsistency
                        if node_temp in gamma_list and node_temp2 in gamma_list:
                            gamma_list = ['INCONSISTENT']
                            return graph_out,conf_out,gamma_list
                        else:
                            gamma_list = [item2 if item2!=node_temp2 else node_temp for item2 in gamma_list]
                    
                    
                elif np.all([item2 in item[1] for item2 in par_diff]) and conf_pars:
                    # remove edges from the duplicate parents
                    graph_out.remove_edges_from([item2 for item2 in graph_out.edges 
                        if item2[0] in par_diff and item2[1] == node_temp2])
                    
                    # merge nodes
                    graph_out = nx.contracted_nodes(graph_out,node_temp,node_temp2,self_loops=False)
                    conf_out = nx.contracted_nodes(conf_out,node_temp,node_temp2,self_loops=False)
                    
                    # check for inconsistency
                    if node_temp in gamma_list and node_temp2 in gamma_list:
                        gamma_list = ['INCONSISTENT']
                        return graph_out,conf_out,gamma_list
                    else:
                        gamma_list = [item2 if item2!=node_temp2 else node_temp for item2 in gamma_list]
                        
            # only add nodes whose parents have all been
            node_list = node_list[1:] + [item[1] for item in graph_temp.edges 
                if np.all([item2[0] in node_list for item2 in graph_temp.edges if item2[1] == item[1]])]
            
            traversed_nodes += [node_temp]
            
        # remove self-loops
        #graph_out.remove_edges_from(nx.selfloop_edges(graph_out))
        #conf_out.remove_edges_from(nx.selfloop_edges(conf_out))
            
        # identify ancestors of nodes in gamma_list
        anc_list = []
        anc_list += gamma_list
        for item in gamma_list:
            anc_list += [item2 for item2 in nx.algorithms.dag.ancestors(graph_out,item) if item2 not in anc_list]
            
        anc_conf_list = []
        for item in anc_list:
            anc_conf_list += [item2 for item2 in nx.algorithms.dag.ancestors(conf_out,item) 
                if item2 not in anc_conf_list]
            
        anc_list += [item for item in anc_conf_list if item not in anc_list]
            
        graph_out = graph_out.subgraph(anc_list)
        conf_out = conf_out.subgraph(anc_list)
        
        # removing (apparently) unneccesary nodes/edges may cause problems in ID* because of recursion
        # check this!
        """    
        # remove confounding nodes that only connect to one node
        rem_nodes = [item for item in conf_out.nodes if conf_out.degree(item) == 1 and item[0] == 'U']
        graph_out.remove_nodes_from(rem_nodes)
        conf_out.remove_nodes_from(rem_nodes)
        
        # remove disconnected nodes
        rem_nodes = [item for item in graph_out.nodes if item in list(nx.isolates(graph_out))
            and item in list(nx.isolates(conf_out))]
        graph_out.remove_nodes_from(rem_nodes)
        conf_out.remove_nodes_from(rem_nodes)
        
        # remove disconnected components
        graph_temp = nx.DiGraph(graph_out)
        graph_temp.add_edges_from(conf_out.edges)
        components = [list(item) for item in nx.weakly_connected_components(graph_temp)]
        rem_nodes = []
        for item in components:
            if target_temp not in item:
                rem_nodes += item
                
        graph_out.remove_nodes_from(rem_nodes)
        conf_out.remove_nodes_from(rem_nodes)
        """
        
        return graph_out,conf_out,gamma_list
            
    
    def conv_to_gamma(self,do_in,obs_in):
        # convert from do_in, obs_in to gamma_list
        
        gamma_list = []
        for item in do_in:
            gamma_list.append(item[0] + '_{' + self.str_list(item[1]) + '}')
        
        for item in obs_in:
            if item not in gamma_list:
                gamma_list.append(item)
                
        return gamma_list
    
    def conv_from_gamma(self,gamma_list):
        # convert from gamma_list to do_in, obs_in
        
        do_in = []
        obs_in = []
        
        for item in gamma_list:
            if '_' in item:
                temp = item.replace('_',',').replace('{','').replace('}','')
                temp = temp.split(',')
                do_in.append([[temp[0]],[temp[1:]]])
            else:
                obs_in.append(item)
                
        return do_in,obs_in
    
    def id_star_alg(self,do_in,obs_in=[],graph_in=[]):
        # implement ID* algorithm        
        
        gamma_list = self.conv_to_gamma(do_in,obs_in)
        
        if graph_in:
            graph_temp = nx.DiGraph(graph_in)
            conf_temp = self.graph_c.subgraph(graph_temp.nodes)
        else:
            graph_temp = nx.DiGraph(self.graph)
            conf_temp = nx.Graph(self.graph_c)        
        
        if not gamma_list:
            print('Step 1')
            return '1'
        
        elif np.any([item[0] + "'" in item[1] for item in do_in]):
            print('Step 2')
            return 0
        
        elif np.any([item[0] in item[1] for item in do_in]):
            
            temp_inds = [ind for ind in range(0,len(do_in)) if do_in[ind][0] not in do_in[ind][1]]
            
            print('Step 3')
            
            return self.id_star_alg(do_in[ind],obs_in,graph_in)
        
        else:
            graph_out,conf_out,gamma_list = self.make_cf_graph(do_in,obs_in,graph_in)
            print('Step 4')
            
            # calculate graph C-components           
            s_sets = [list(item) for item in nx.connected_components(conf_out.to_undirected())]
            
            
            if 'INCONSISTENT' in gamma_list:
                print('Step 5')
                return '0'
            elif len(s_sets) > 1:
                print('Start Step 6')
                
                sum_list = [item for item in graph_out.nodes if item[:3] != 'U^{']
                
                print(sum_list)
                
                str_out = 'sum_{' + self.str_list(sum_list) + '}'
                for item in s_sets:
                    
                    do_list_temp = [item2 for item2 in sum_list if item2 not in item]
                    do_in_temp = [[item2,do_list_temp] for item2 in item]
                    
                    str_out += self.id_star_alg(do_in_temp,[],graph_temp)
                    
                print('End Step 6')
                
                return str_out
                    
            else:
                gamma_subs = []
                for item in do_in:
                    gamma_subs += [item2 for item2 in item[0] if item2 not in gamma_subs]
                do_vars = [item[0] for item in do_in]
                    
                if np.any([item + "'" in obs_in]) or np.any([item + "'" in do_vars]):
                    print('Step 8')
                    return ' FAIL '
                else:
                    
                    str_temp = self.str_list(list(graph_temp.nodes))
                    str_temp2 = self.str_list(gamma_subs)
                    
                    print('Step 9')
                    return 'P_{' + str_temp2 + '}' + '(' + str_temp + ')'
        return
        
    def idc_star_alg(self,do_in,do_delta,obs_in=[],obs_delta=[],graph_in=[]):
        
        if self.id_star_alg(do_delta,obs_delta,graph_in) == '0':
            return 'UNDEFINED'
        else:
            graph_out,conf_out,gamma_list = self.make_cf_graph(do_in+do_delta,obs_in+obs_delta,graph_in)
            
            if 'INCONSISTENT' in gamma_list:
                return '0'
            else:
                n_gam = len(do_in) + len(obs_in)
                n_del = len(do_delta) + len(obs_delta)
                
                for item in gamma_list[n_gam:]:
                    if '_{' in item:
                        # check for d-separation
                        graph_temp = nx.DiGraph(graph_out)
                        graph_temp.add_edges_from(conf_out.edges)
                        graph_temp.remove_edges_from([item2 for item2 in graph_temp.edges
                            if item2[0] == item])
                        d_sep = self.d_sep(item,gamma_list[n_gam:])
                        if d_sep:
                            
                            do_gam_temp,obs_gam_temp = self.conv_from_gamma(gamma_list[:n_gam])
                            do_del_temp,obs_del_temp = self.conv_to_gamma(gamma_list[n_gam:])
                            
                            do_del_temp = [item2 for item2 in do_del_temp if item2 != item]
                            
                            gam_temp = [[item2[0],item2[1] + [item]] for item2 in do_gam_temp]
                            gam_temp += [[item2,[item2]] for item2 in obs_gam_temp]
                            
                            
                            return self.idc_star_alg(gam_temp,do_del_temp,[],obs_gam_temp,graph_temp)
                else:
                    do_gam_temp,obs_gam_temp = self.conv_from_gamma(gamma_list[:n_gam])
                    do_del_temp,obs_del_temp = self.conv_to_gamma(gamma_list[n_gam:])
                    
                    P_prime = self.id_star_alg(do_gam_temp+do_del_temp,obs_gam_temp,obs_del_temp,graph_temp)

                    return '[' + P_prime + ']/[sum_{' + self.str_list(gamma_list[:n_gam]) + '}' + P_prime + ']'
        
        return
    
    
    def prob_init(self,data_in,lr=1e-3):
        """initialize all of the nodes' probability distributions given data_in; lr is the learning rate"""
        
        exog_list = []
        prob_dict = {}
        init_list = []
        non_init_list = []
        
        for name in self.node_dict:
            
            if name in data_in and (np.all([item in data_in for item in self.parent_name_dict[name]])
                or self.node_dict[name].n_inputs == 0):

                data_out_temp = torch.tensor([data_in[name]]).T
                data_in_temp = torch.tensor([data_in[item] for item in self.parent_name_dict[name]]).T
                self.node_dict[name].prob_init(data_in_temp,data_out_temp,lr)

                init_list.append(name)
            
            else:
                non_init_list.append(name)
            
            if self.node_dict[name].n_inputs == 0:
                exog_list.append(name)
            #prob_dict[name] = self.node_dict[name].prob_dist
        
        self.exog_list = exog_list
        self.init_list = init_list
        self.non_init_list = non_init_list
        #self.prob_dict = prob_dict

        return
        
    def model_sample(self):
        """produce a dictionary of samples for all variables in the graph"""
        
        # define exogenous samples
        eps_dict = {}
        sample_dict = {}
        
        for item in self.exog_list:
            sample_dict[item],eps_dict[item + '_e'] = self.node_dict[item].sample()
            
        flag = 0
        while flag == 0:
            
            # find all nodes not in sample_dict with parents entirely in sample dict and sample those nodes
            for item in self.entity_list:
                if (item not in sample_dict 
                    and np.all([item2 in sample_dict for item2 in self.parent_name_dict[item]])):
                    
                    sample_dict[item],eps_dict[item + '_e'] = self.node_dict[item].sample(
                        torch.tensor([sample_dict[item2] for item2 in self.parent_name_dict[item]]))
            
            # if sample dict has all of the nodes in entity list, stop
            if sorted([item for item in sample_dict]) == sorted(self.entity_list):
                flag = 1
            
        sample_dict.update(eps_dict)
        
        return sample_dict
    
    
    def scm_rescale(self,name,val_in):
        """Do the necessary rescaling for doing conditionals, do-statements with SCM model."""
        
        node = graph_test.node_dict[name]
        
        min_temp = node.y_min
        max_temp = node.y_max
        
        val_temp = (val_in-min_temp)/(max_temp-min_temp)
               
        return torch.log(val_temp/(1-val_temp))
    
    
    def model_cond_sample(self,data_dict):
        """sample the graph given the conditioned variables in data_dict"""
        
        data_in = {}
        for item in data_dict:
            
            if self.graph_type == 'SCM':
                val = self.scm_rescale(item,data_dict[item])
            else:
                val = data_dict[item]
            
            data_in[item + '_y'] = val
        
        cond_model = pyro.condition(self.model_sample,data=data_in)
        
        return cond_model()
        
        
    def model_do_sample(self,do_dict):
        """sample the graph given the do-variables in do_dict"""
        
        data_in = {}
        for item in do_dict:
            
            if self.graph_type == 'SCM':
                val = self.scm_rescale(item,do_dict[item])
            else:
                val = do_dict[item]
            
            data_in[item + '_y'] = val
        
        do_model = pyro.do(self.model_sample,data=data_in)
        
        return do_model()
    
    
    def model_do_cond_sample(self,do_dict,data_dict):
        """sample the graph given do-variables in do_dict and conditioned variables in data_dict"""
        
        if np.any([[item1 == item2 for item1 in do_dict] for item2 in data_dict]):
            print('overlapping lists!')
            return
        else:
            do_dict_in = {}
            for item in do_dict:
                if self.graph_type == 'SCM':
                    val = self.scm_rescale(item,do_dict[item])
                else:
                    val = do_dict[item]
                
                do_dict_in[item + '_y'] = val
                
            data_dict_in = {}
            for item in data_dict:
                if self.graph_type == 'SCM':
                    val = self.scm_rescale(item,data_dict[item])
                else:
                    val = data_dict[item]
                
                data_dict_in[item + '_y'] = val
            
            cond_model = pyro.condition(self.model_sample,data=do_dict_in)
            do_model = pyro.condition(cond_model,data=data_dict_in)
            
            return do_model()
        
    
    def model_counterfact(self,obs_dict,do_dict_counter):
        """Find conditional distribution on exogenous variables given observations in obs_dict 
        and do variable values in do_dict_counter.  This is not currently working for the Bayesian or MLE graphs"""
        
        #cond_dict = self.model_cond_sample(obs_dict)
        #cond_dict_temp = {}
        #for item in self.exog_list:
            #cond_dict_temp[item] = cond_dict[item]
            
            
        # get epsilon distributions
        cond_temp = self.model_cond_sample(obs_dict)
        
        # create conditional distribution
        eps_temp = {}
        for item in cond_temp:
            if item[-2:] == '_e':
                eps_temp[item] = cond_temp[item]
        
        # impose do-statements on the result
            
        data_do = {}
        for item in do_dict_counter:
            
            if self.graph_type == 'SCM':
                val = self.scm_rescale(item,do_dict_counter[item])
            else:
                val = do_dict_counter[item]
            
            data_do[item + '_y'] = val
            
            
        # evaluate observed variables given this condition distribution and do_dict_counter do-variables
        #return self.model_do_cond_sample(do_dict_counter,cond_dict_temp)
        
        counter_model = pyro.do(pyro.condition(self.model_sample,data=eps_temp),data=data_do)
        
        return counter_model()
        
        
    def cond_mut_info(self,target,test,cond,data_in):
        """calculate the conditional mutual information between target and test given data_in
        I(target:test|cond) just uses input data, but it's necessary to bin data 
        (creating discrete distribution) to perform calculations"""
        
        n_data = len(data_in)
        
        data_in_np = np.asarray([[item2.item() for item2 in item] for item in data_in])        
        cond_temp = cond
        
        if not cond:
            # find parents of target
            for item in target:
                for item2 in self.parent_name_dict[item]:
                    if item2 not in cond_temp:
                        cond_temp.append(item2)
        
        
        target_inds = [self.entity_list.index(item) for item in target]
        test_inds = [self.entity_list.index(item) for item in test]
        cond_inds = [self.entity_list.index(item) for item in cond_temp]

        total_inds = target_inds + test_inds + cond_inds
        n_tot = len(total_inds)
        n_target = len(target_inds)
        n_test = len(test_inds)
        n_cond = len(cond_inds)

        # bin the incoming data
        data_bin = np.histogramdd((data_in_np[:,total_inds]),bins=10)[0]/n_data

        
        # calculate each joint entropy
        
        all_inds = list(range(0,n_tot))
        
        p_z = np.sum(data_bin,tuple(all_inds[:n_target+n_test]))
        H_z = -np.sum(p_z*np.log(p_z+1e-6))
        
        p_xz = np.sum(data_bin,tuple(all_inds[n_target:n_target+n_test]))
        H_xz = -np.sum(p_xz*np.log(p_xz+1e-6))
        
        p_yz = np.sum(data_bin,tuple(all_inds[:n_target]))
        H_yz = -np.sum(p_yz*np.log(p_yz+1e-6))
        
        H_xyz = -np.sum(data_bin*np.log(data_bin+1e-6))
                
        return H_xz + H_yz - H_xyz - H_z
        
    def g_test(self,name,data_in):
        """do the G-test on a single variable of interest determine if causal graph captures underlying distribution
        have to bin data to perform calculations"""
        
        name_ind = self.entity_list.index(name[0])
        
        if self.node_dict[name[0]].node_type == 'binary':
            # bin the data
            binned_data = torch.histc(data_in[:,name_ind],2,-0.5,1.5)
            
            # generate sample data
            data_samp = torch.tensor([self.model_sample()[name[0]] for i in range(0,len(data_in))])
            binned_samp = torch.histc(data_samp,2,-0.5,1.5)
            
        else:
            data_max = torch.max(data_in[:,name_ind])
            data_min = torch.min(data_in[:,name_ind])

            # bin the data
            binned_data = torch.histc(data_in[:,name_ind],100,data_min,data_max)

            # generate sample data
            data_samp = torch.tensor([self.model_sample()[name[0]] for i in range(0,len(data_in))])
            binned_samp = torch.histc(data_samp,100,data_min,data_max)
        
        
        g_val = 2*torch.sum(binned_data*torch.log(binned_data/(binned_samp+1e-6)))
        
        dof = len(data_in) - 1
        
        p_val = 1-sp.stats.chi2.cdf(g_val.item(), dof)
        
        return g_val,p_val
        
    def tot_effect(self,target,do_dict,do_prime_dict,n_samples):
        """calculate the total effect of changing an intervention from do_dict_prime values to do_dict values
        on the variables in target"""
        
        var_array = np.zeros((n_samples,len(target)))
        var_prime_array = np.zeros((n_samples,len(target)))
        
        for i in range(0,n_samples):
            dict_temp = self.model_do_sample(do_dict)
            var_array[i,:] = np.asarray([dict_temp[item] for item in target])
            
            dict_prime_temp = self.model_do_sample(do_prime_dict)
            var_prime_array[i,:] = np.asarray([dict_prime_temp[item] for item in target])
            
        var_mean = np.mean(var_array,axis=0)
        var_prime_mean = np.mean(var_prime_array,axis=0)
        
        result_dict = {}
        for i in range(0,len(target)):
            result_dict[target[i]] = var_mean[i] - var_prime_mean[i]
        
        return result_dict
    
    def cd_effect(self,target,do_dict,do_prime_dict,med_dict,n_samples):
        """calculate the controlled direct effect of changing an intervention from do_dict_prime to do_dict values
        on the variables in target given fixed mediating values"""
        
        new_do_dict = {**do_dict, **med_dict}
        new_do_prime_dict = {**do_prime_dict, **med_dict}
        
        var_array = np.zeros((n_samples,len(target)))
        var_prime_array = np.zeros((n_samples,len(target)))
        
        for i in range(0,n_samples):
            dict_temp = self.model_do_sample(new_do_dict)
            var_array[i,:] = np.asarray([dict_temp[item] for item in target])
            
            dict_prime_temp = self.model_do_sample(new_do_prime_dict)
            var_prime_array[i,:] = np.asarray([dict_prime_temp[item] for item in target])
            
        var_mean = np.mean(var_array,axis=0)
        var_prime_mean = np.mean(var_prime_array,axis=0)
        
        result_dict = {}
        for i in range(0,len(target)):
            result_dict[target[i]] = var_mean[i] - var_prime_mean[i]
        
        return result_dict
        
        
    def nd_effect(self,target,do_dict,do_prime_dict,n_samples):
        """calculate the natural direct effect of changing an intervention from do_dict_prime values to do_dict 
        values on the variables in target"""
        
        # identify parents of target variables that aren't in the do_dicts
        
        parent_list = []
        for item in target:
            for item2 in self.parent_name_dict[item]:
                if item2 not in parent_list and item2 not in do_dict:
                    parent_list.append(item2)        
            
        var_array = np.zeros((n_samples,len(target)))
        var_prime_array = np.zeros((n_samples,len(target)))

        for i in range(0,n_samples):
            dict_cond_temp = self.model_do_sample(do_prime_dict)
            dict_cond = {}
            for item in parent_list:
                dict_cond[item] = dict_cond_temp[item]

            dict_temp = self.model_do_cond_sample(do_dict,dict_cond)
            var_array[i,:] = np.asarray([dict_temp[item] for item in target])

            dict_prime_temp = self.model_do_cond_sample(do_prime_dict,dict_cond)
            var_prime_array[i,:] = np.asarray([dict_prime_temp[item] for item in target])

        var_mean = np.mean(var_array,axis=0)
        var_prime_mean = np.mean(var_prime_array,axis=0)

        result_dict = {}
        for i in range(0,len(target)):
            result_dict[target[i]] = var_mean[i] - var_prime_mean[i]

        return result_dict
        
    def ni_effect(self,target,do_dict,do_prime_dict,n_samples):
        """calculate the natural indirect effect of changing an intervention from do_dict_prime values to do_dict 
        values on the variables in target"""
        
        # identify parents of target variables that aren't in the do_dicts
        
        parent_list = []
        for item in target:
            for item2 in self.parent_name_dict[item]:
                if item2 not in parent_list and item2 not in do_dict:
                    parent_list.append(item2)
                    
        
        var_array = np.zeros((n_samples,len(target)))
        var_prime_array = np.zeros((n_samples,len(target)))

        for i in range(0,n_samples):
            
            dict_cond_temp = self.model_do_sample(do_dict)
            dict_cond = {}
            for item in parent_list:
                dict_cond[item] = dict_cond_temp[item]
                
            dict_cond_prime_temp = self.model_do_sample(do_prime_dict)
            dict_cond_prime = {}
            for item in parent_list:
                dict_cond_prime[item] = dict_cond_prime_temp[item]

            dict_temp = self.model_do_cond_sample(do_prime_dict,dict_cond)
            var_array[i,:] = np.asarray([dict_temp[item] for item in target])

            dict_prime_temp = self.model_do_cond_sample(do_prime_dict,dict_cond_prime)
            var_prime_array[i,:] = np.asarray([dict_prime_temp[item] for item in target])

        var_mean = np.mean(var_array,axis=0)
        var_prime_mean = np.mean(var_prime_array,axis=0)

        result_dict = {}
        for i in range(0,len(target)):
            result_dict[target[i]] = var_mean[i] - var_prime_mean[i]

        return result_dict    
        
        
  
    def write_to_cf(self,filename,spacing):
        """write the causal graph to a text file to import into causal fusion"""
        
        pos_dict = nx.drawing.layout.planar_layout(self.graph)
        
        write_dict = {}
        write_dict['name'] = 'causal_graph'
        
        # write nodes
        write_dict['nodes'] = []
        for i in range(0,len(self.entity_list)):
            name = self.entity_list[i]
            
            write_dict['nodes'].append({})
            
            write_dict['nodes'][-1]['id'] = 'node' + str(i)
            write_dict['nodes'][-1]['name'] = name
            write_dict['nodes'][-1]['label'] = name
            write_dict['nodes'][-1]['type'] = 'basic'
            write_dict['nodes'][-1]['metadata'] = {}
            write_dict['nodes'][-1]['metadata']['x'] = spacing*pos_dict[i][0]
            write_dict['nodes'][-1]['metadata']['y'] = spacing*pos_dict[i][1]
            write_dict['nodes'][-1]['metadata']['label'] = ''
            write_dict['nodes'][-1]['metadata']['shape'] = 'ellipse'
            write_dict['nodes'][-1]['metadata']['fontSize'] = 14
            write_dict['nodes'][-1]['metadata']['sizeLabelMode'] = 5
            write_dict['nodes'][-1]['metadata']['font'] = {}
            write_dict['nodes'][-1]['metadata']['font']['size'] = 14
            write_dict['nodes'][-1]['metadata']['size'] = 14
            write_dict['nodes'][-1]['metadata']['labelNodeId'] = 'node' + str(i) + 'ID'
            write_dict['nodes'][-1]['metadata']['labelNodeOffset'] = {}
            write_dict['nodes'][-1]['metadata']['labelNodeOffset']['x'] = 0
            write_dict['nodes'][-1]['metadata']['labelNodeOffset']['y'] = 0
            write_dict['nodes'][-1]['metadata']['labelOffset'] = {}
            write_dict['nodes'][-1]['metadata']['labelOffset']['x'] = 0
            write_dict['nodes'][-1]['metadata']['labelOffset']['y'] = 0
            write_dict['nodes'][-1]['metadata']['shadow'] = {}
            write_dict['nodes'][-1]['metadata']['shadow']['color'] = '#00000080'
            write_dict['nodes'][-1]['metadata']['shadow']['size'] = 0
            write_dict['nodes'][-1]['metadata']['shadow']['x'] = 0
            write_dict['nodes'][-1]['metadata']['shadow']['y'] = 0
            
        # write edges
        write_dict['edges'] = []
        
        for i in range(0,len(self.edge_list)):
            
            item = self.edge_list[i]
            from_node = self.entity_list.index(item[0])
            to_node = self.entity_list.index(item[1])
            
            write_dict['edges'].append({})
            
            write_dict['edges'][-1]['id'] = 'node' + str(from_node) + '->node' + str(to_node)
            write_dict['edges'][-1]['from'] = item[0]
            write_dict['edges'][-1]['to'] = item[1]
            write_dict['edges'][-1]['type'] = 'directed'
            write_dict['edges'][-1]['metadata'] = {}
            write_dict['edges'][-1]['metadata']['isLabelDraggable'] = True
            write_dict['edges'][-1]['metadata']['label'] = ''
            
        
        write_dict['task'] = {}
        
        write_dict['metadata'] = {}
        
        write_dict['project_id'] = '123456789'
        write_dict['_fileType'] = 'graph'
                
        with open(filename + '.json', 'w') as json_file:
            json.dump(write_dict, json_file)
        
class str_graph(cg_graph):
    """define class of causal graphs initialized using a list of BEL-statements represented as strings"""
    
    def __init__(self,str_list,b_or_mle,type_dict={}):
        
        super().__init__()
        
        edge_list = []
        entity_list = []
        
        # construct graph from list of BEL statement strings
            
        for item in str_list:

            sub_ind = item.find('=')

            sub_temp = item[:sub_ind-1]
            obj_temp = item[sub_ind+3:]

            rel_temp = item[sub_ind:sub_ind+2]

            if sub_temp not in entity_list:
                entity_list.append(sub_temp)
            if obj_temp not in entity_list:
                entity_list.append(obj_temp)

            nodes_temp = [sub_temp,obj_temp]
            list_temp = [[item[0],item[1]] for item in edge_list]
            if nodes_temp in list_temp:
                ind_temp = list_temp.index(nodes_temp)
                edge_list[ind_temp][2] += ',' + rel_temp
            else:
                edge_list.append([sub_temp,obj_temp,rel_temp])
                
        self.entity_list = entity_list
        self.edge_list = edge_list
        
        self.proc_data(b_or_mle,type_dict)
        
        return

class bel_graph(cg_graph):
    """define class of causal graphs initialized using a pyBEL graph"""
    
    def __init__(self,bel_graph,b_or_mle,type_dict={},subset_rels=False):
        
        super().__init__()
        
        edge_list = []
        entity_list = []

        # construct graph from pyBEL graph
            
        for item in bel_graph.edges:
            edge_temp = bel_graph.get_edge_data(item[0],item[1],item[2])
            sub_temp = str(item[0]).replace('"','')
            obj_temp = str(item[1]).replace('"','')
            rel_temp = edge_temp['relation']

            if sub_temp not in entity_list:
                entity_list.append(sub_temp)
            if obj_temp not in entity_list:
                entity_list.append(obj_temp)

            if subset_rels:
                # ignore hasVariant, partOf relations

                if rel_temp.find('crease') > 0 or rel_temp.find('regulate') > 0:
                    edge_list.append([sub_temp,obj_temp,rel_temp])

            else:
                # check for duplicate edges
                nodes_temp = [sub_temp,obj_temp]
                list_temp = [[item[0],item[1]] for item in edge_list]
                if nodes_temp in list_temp:
                    ind_temp = list_temp.index(nodes_temp)
                    edge_list[ind_temp][2] += ',' + rel_temp
                else:
                    edge_list.append([sub_temp,obj_temp,rel_temp])
        
        self.entity_list = entity_list
        self.edge_list = edge_list
        
        self.proc_data(b_or_mle,type_dict)
        
        return
    
class cf_graph(cg_graph):
    """define class of causal graphs initialized using a json file generated by exporting from Causal Fusion"""
    
    def __init__(self,json_file,b_or_mle,type_dict={}):
        
        super().__init__()
        
        edge_list = []
        entity_list = []
        
        file1 = open(json_file)
        j_str = file1.readline()
        file1.close()
        loaded_json = json.loads(j_str)

        entity_list = []
        for item in loaded_json['nodes']:
            entity_list.append(item['name'])

        edge_list = []
        for item in loaded_json['edges']:
            edge_list.append([item['from'],item['to'],''])
        
        n_nodes = len(entity_list)
        self.n_nodes = n_nodes

        self.entity_list = entity_list
        self.edge_list = edge_list
        
        self.proc_data(b_or_mle,type_dict)

        return
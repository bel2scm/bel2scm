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
        adj_mat = np.zeros((self.n_nodes,self.n_nodes),dtype=int)

        for item in self.edge_list:
            out_ind = self.entity_list.index(item[0])
            in_ind = self.entity_list.index(item[1])
            adj_mat[out_ind,in_ind] = 1

        self.adj_mat = adj_mat
        
        self.graph = nx.DiGraph(adj_mat)
        
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
        
        self.parent_ind_list = []
        self.child_ind_list = []
        self.parent_name_dict = {}
        self.child_name_dict = {}
        
        self.parent_ind_list = [np.where(self.adj_mat[:,i] > 0)[0] for i in range(0,self.n_nodes)]
        self.child_ind_list = [np.where(self.adj_mat[i,:] > 0)[0] for i in range(0,self.n_nodes)]
        
        node_dict = {}
        
        for i in range(0,n_nodes):
            self.parent_name_dict[self.entity_list[i]] = [self.entity_list[item] 
                for item in self.parent_ind_list[i]]
            self.child_name_dict[self.entity_list[i]] = [self.entity_list[item] 
                for item in self.child_ind_list[i]]
        
            if type_dict:
                node_type = type_dict[self.entity_list[i]]

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


                for i in range(0,self.n_nodes):

                    ind_temp = self.entity_list[i].find('(')
                    str_temp = self.entity_list[i][:ind_temp]

                    node_type = ''

                    for item in bel_dict:
                        if str_temp in bel_dict[item]:
                            node_type = vartype_dict[item]

                    if node_type == '':
                        node_type = 'Normal'
                        print('BEL node type ' + str_temp + ' not known -- defaulting to Normal')

            if self.graph_type == 'Bayes':
                node_dict[self.entity_list[i]] = bayes_node(np.sum(adj_mat[:,i]),self.entity_list[i],node_type)
            elif self.graph_type == 'MLE':
                node_dict[self.entity_list[i]] = mle_node(np.sum(adj_mat[:,i]),self.entity_list[i],node_type)
            elif self.graph_type == 'SCM':
                node_dict[self.entity_list[i]] = scm_node(np.sum(adj_mat[:,i]),self.entity_list[i],node_type)
            else:
                print('node type ' + self.graph_type + 'not recognized -- defaulting to MLE')
                node_dict[self.entity_list[i]] = mle_node(np.sum(adj_mat[:,i]),self.entity_list[i],node_type)
        
        self.node_dict = node_dict
        
        return
        
    
    def remove_edge(self,edge_rem):
        """remove all of the edges in edge_rem from the causal graph"""
        
        for item in edge_rem:
            ind_remove = [i for i in range(0,len(self.edge_list)) 
                if (self.edge_list[i][0] == edge_rem[0] and self.edge_list[i][1] == edge_rem[1])]
            for ind in ind_remove:
                self.edge_list.remove(self.edge_list[i])
            self.adj_mat[self.entity_list.index(item[0]),self.entity_list.index(item[1])] = 0
            
        self.graph = nx.DiGraph(self.adj_mat)
        
        self.parent_ind_list = []
        self.child_ind_list = []
        self.parent_name_dict = {}
        self.child_name_dict = {}
        
        self.parent_ind_list = [np.where(self.adj_mat[:,i] > 0)[0] for i in range(0,self.n_nodes)]
        self.child_ind_list = [np.where(self.adj_mat[i,:] > 0)[0] for i in range(0,self.n_nodes)]
        
        for i in range(0,self.n_nodes):
            self.parent_name_dict[self.entity_list[i]] = [
                self.entity_list[item] for item in self.parent_ind_list[i]]
            self.child_name_dict[self.entity_list[i]] = [
                self.entity_list[item] for item in self.child_ind_list[i]]
        return
    
    def add_confound(self,confound_pairs):
        """ add a list of pairs of nodes that share unobserved confounders"""
        
        adj_mat = np.zeros((self.n_nodes,self.n_nodes),dtype=int)
        
        for item in confound_pairs:
            i = self.entity_list.index(item[0])
            j = self.entity_list.index(item[1])
            adj_mat[i,j] = 1
            adj_mat[j,i] = 1
            
        self.adj_mat_c = adj_mat
        self.graph_c = nx.Graph(adj_mat)
        
        return
    
    def str_list(self,node_list):
        """ return a string listing the nodes in node_list - this is used in the ID and IDC algorithms """
        str_out = ''
        
        for item in node_list:
            str_out += item + ','

        return str_out[:-1]
    
    def d_sep(self,x,y,z,graph_in):
        """ determine if all paths from x to y are d-separated by z in graph_in """
        
        
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
        
        x_list = [self.entity_list.index(item) for item in x]
        y_list = [self.entity_list.index(item) for item in y]
        z_list = [self.entity_list.index(item) for item in z]
        
        
        # identify all paths from x to y
        path_list = []
        
        for item in x_list:      
            for path in nx.all_simple_paths(graph_temp, source=item, target=y_list):
                path_list.append(path)
                
        print(str(len(path_list)) + ' total paths')
        
        # iterate through paths
        for item in path_list:
            # if an element of z is in the path, path is d-separated
            # else, path is not d-separated, return False
            
            if not np.any([ind in item for ind in z_list]):
                return False
        
        # if all paths d-separated, return True
        
        return True
    
    def id_alg(self,y,x,p_in=[],graph_in=[]):
        """ calculate P(y | do(x)) or return failure if this is not possible """
        
        if np.any([item in y for item in x]):
            print('Error -- overlap between x and y')
            print(x)
            print(y)
            print(p_in)
            print(graph_in.nodes)
        
        if graph_in:
            graph_temp = graph_in
        else:
            graph_temp = self.graph
            
        if p_in:
            p_expr = p_in
        else:
            node_list = [graph_temp.nodes[ind]['name'] for ind in graph_temp.nodes]
            p_expr = 'P(' + self.str_list(node_list) + ')'
            
            
        y_inds = [ind for ind in graph_temp.nodes if graph_temp.nodes[ind]['name'] in y]
        
        # identify ancestors of y
        y_anc = y_inds
        for item in y_inds:
            set_temp = nx.algorithms.dag.ancestors(graph_temp,item)
            for item2 in set_temp:
                if item2 not in y_anc:
                    y_anc.append(item2)
                    
        # identify all nodes in the graph
        v_set = graph_temp.nodes
        
        v_not_anc_y = []
        for item in v_set:
            if item not in y_anc:
                v_not_anc_y.append(item)
                
        # remove edges to x
        
        # identify edges to x
        x_inds = [ind for ind in graph_temp.nodes if graph_temp.nodes[ind]['name'] in x]
        x_edges = [list(graph_temp.in_edges(item)) for item in x_inds]
        
        graph_xbar = nx.DiGraph(graph_temp)
        #print(nx.is_frozen(graph_temp2))
        for item in x_edges:
            graph_xbar.remove_edges_from(item)
            
        y_anc_x_bar = y_inds
        
        for item in y_inds:
            set_temp = nx.algorithms.dag.ancestors(graph_xbar,item)
            for item2 in set_temp:
                if item2 not in y_anc:
                    y_anc_x_bar.append(item2)
                
        w_set = []
        for item in v_set:
            if item not in x_inds and item not in y_anc_x_bar:
                w_set.append(item)
        
        # line 1
        if not x:
            # return sum over all non-y variables
            
            node_list = [graph_temp.nodes[ind]['name'] for ind in graph_temp.nodes 
                if graph_temp.nodes[ind]['name'] not in y]
            node_list2 = [graph_temp.nodes[ind]['name'] for ind in graph_temp.nodes]  
            str_out = '[sum_{' + self.str_list(node_list) + '} ' + p_expr + ']'
            #print('Step 1')
            
            return str_out
            
        # line 2
        elif v_not_anc_y:
            
            x_temp = [graph_temp.nodes[ind]['name'] for ind in y_anc if graph_temp.nodes[ind]['name'] in x]
            node_list = [graph_temp.nodes[ind]['name'] for ind in v_not_anc_y]
            str_out = '[sum_{' + self.str_list(node_list) + '} ' + p_expr + ']'
            graph_anc = graph_temp.subgraph(y_anc)
            
            #print('Begin Step 2')
            expr_out = self.id_alg(y,x_temp,str_out,graph_anc)
            #print('End Step 2')
            
            return expr_out
        
        # line 3
        elif w_set:
            #print('Begin Step 3')
            expr_out = self.id_alg(y,x+w_set,p_expr,graph_temp)
            #print('End Step 3')
            
            return expr_out
        
        else:            
            # calculate graph C-components
            graph_temp_c = nx.Graph(self.graph_c.subgraph(graph_temp.nodes))
            graph_temp_c.remove_nodes_from(x_inds)
            s_sets = [list(item) for item in nx.connected_components(graph_temp_c)]
            
            # line 4
            if len(s_sets) > 1:
                #print('Begin Step 4')
                node_list = [graph_temp.nodes[item]['name'] for item in v_set 
                    if graph_temp.nodes[item]['name'] not in y and graph_temp.nodes[item]['name'] not in x]
                str_out = '[sum_{' + self.str_list(node_list) + '} '
                
                for item in s_sets:
                    v_s_set = [graph_temp.nodes[item2]['name'] for item2 in v_set if item2 not in item]
                    s_in = [graph_temp.nodes[item2]['name'] for item2 in item]
                    
                    if np.any([item2 in v_s_set for item2 in s_in]):
                        print('Error -- x/y overlap')
                        print(v_s_set)
                        print(s_in)
                    
                    str_out += self.id_alg(s_in,v_s_set,p_expr,graph_temp)
                        
                #print('End Step 4')
                str_out += ']'
                
                return str_out
            
            else:
                graph_temp_c2 = self.graph_c.subgraph(graph_temp.nodes)
                
                s_sets2 = [list(item) for item in nx.connected_components(graph_temp_c2)]
                
                # line 5
                if sorted(s_sets2[0]) == sorted(graph_temp.nodes):
                    
                    node_list = [graph_temp.nodes[ind]['name'] for ind in s_sets2[0]]
                    node_list2 = [graph_temp.nodes[ind]['name'] for ind in graph_temp.nodes if ind in s_sets2[0]]
                    
                    str_out = 'FAIL(' + self.str_list(node_list) + ',' + self.str_list(node_list2) + ')'
                    
                    #print('Step 5')
                    return str_out
                
                # line 6
                elif np.any([sorted(s_sets[0]) == sorted(item) for item in s_sets2]):
                            
                    node_list = [graph_temp.nodes[item]['name'] for item in s_sets[0] 
                        if graph_temp.nodes[item]['name'] not in y]
                    str_out = '[sum_{' + self.str_list(node_list) + '}'
                    
                    for item in s_sets[0]:
                        # identify parents of node i
                        parents = [graph_temp.nodes[ind]['name'] for ind in graph_temp.predecessors(item)]
                        
                        str_out += 'P(' + graph_temp.nodes[item]['name'] + '|' + self.str_list(parents) + ')'
                    #print('Step 6')
                    return str_out + ']'
                
                # line 7
                elif np.any([np.all([item in item2 for item in s_sets[0]]) for item2 in s_sets2]):
                    ind = np.where([np.all([item in item2 for item in s_sets[0]]) for item2 in s_sets2])[0][0]
                    graph_prime = graph_temp.subgraph(s_sets2[ind])
                    x_prime = [graph_temp.nodes[item]['name'] for item in s_sets2[ind] 
                        if graph_temp.nodes[item]['name'] in x]
                    str_out = ''
                    
                    for item in s_sets2[ind]:
                        pred = list(graph_temp.predecessors(item))
                        par_set = [item2 for item2 in pred if item2 in s_sets2[ind]]
                        par_set += [item2 for item2 in pred if item2 not in s_sets2[ind]]
                        node_list = [graph_temp.nodes[ind]['name'] for ind in par_set]
                        str_out += 'P(' + graph_temp.nodes[item]['name'] + '|' + self.str_list(node_list) + ')'
                        
                    #print('Begin Step 7')
                    
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
        """ calculate P(y | do(x), z) or return failure if this is not possible"""
        
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
            node_list = [graph_temp.nodes[ind]['name'] for ind in graph_temp.nodes]
            p_expr = 'P(' + self.str_list(node_list) + ')'
        
        
        # identify edges to x
        x_inds = [ind for ind in graph_temp.nodes if graph_temp.nodes[ind]['name'] in x]
        x_edges = [list(graph_temp.in_edges(item)) for item in x_inds]

        digraph_xbar = nx.DiGraph(graph_temp)
        for item in x_edges:
            digraph_xbar.remove_edges_from(item) 
            
        # identify edges from z
        z_inds = [ind for ind in graph_temp.nodes if graph_temp.nodes[ind]['name'] in z]
        z_edges = [list(graph_temp.out_edges(item2)) for item2 in z_inds]
        
        # check for d-separation
        for i in range(0,len(z)):
            digraph_xbar_zbar = nx.DiGraph(digraph_xbar)
            digraph_xbar_zbar.remove_edges_from(z_edges[i])
            graph_xbar_zbar = digraph_xbar_zbar.to_undirected()
            
            graph_xbar_zbar.add_edges_from(self.graph_c.subgraph(graph_temp.nodes).edges)
                
            # calculate d-separation
            d_sep = self.d_sep(y,[z[i]],[item for item in x+z if item != z[i]],graph_xbar_zbar)
            
            if d_sep:
                
                return self.idc_alg(y,x+[z[i]],[item2 for item2 in z if item2 != z[i]],p_expr,graph_temp)
            
        p_prime = self.id_alg(y+z,x,p_expr,graph_temp)
        
        str_out = '[' + p_prime + ']/[ sum_{' + self.str_list(y) + '}' + p_prime + ']'
        
        return str_out
    
    
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
import numpy as np
import scipy as sp
import networkx as nx

from scipy import stats

import pybel as pb
import json
import time
import csv

import torch
import pyro

import importlib

from . import graph_node as gn

# create a class of causal graphs

class cg_graph():
    
    def __init__(self,str_list=[],bel_graph=[],json_file=[],type_dict=[],only_creases=True):
        
        self.only_creases = only_creases
        
        edge_list = []

        entity_list = []
        
        if str_list:
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
                    
                if only_creases:
                    # ignore hasVariant, partOf relations

                    if rel_temp.find('crease') > 0:
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
                
        elif bel_graph:
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
                
                if only_creases:
                    # ignore hasVariant, partOf relations

                    if rel_temp.find('crease') > 0:
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
        elif json_file:
            # construct graph from json file
            
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

        adj_mat = np.zeros((n_nodes,n_nodes),dtype=int)

        for item in edge_list:
            out_ind = entity_list.index(item[0])
            in_ind = entity_list.index(item[1])
            adj_mat[out_ind,in_ind] = 1
            
        self.edge_list = edge_list
        self.entity_list = entity_list
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
                
        
        node_dict = {}
        
        cont_list = ['a','abundance','complex','complexAbundance','geneAbundance','g','microRNAAbundance','m',
            'populationAbundance','pop','proteinAbundance','p','reaction','rxn','rnaAbundance','r']
        bin_list = ['activity','act','biologicalProcess','bp','pathology','path','molecularActivity','ma']
        
        for i in range(0,n_nodes):
            
            if str_list or json_file:
                node_type = type_dict[entity_list[i]]
            elif bel_graph:
                ind_temp = entity_list[i].find('(')
                str_temp = entity_list[i][:ind_temp]
                
                
                if str_temp in cont_list:
                    node_type = 'continuous'
                elif str_temp in bin_list:
                    node_type = 'binary'
                else:
                    node_type = 'continuous'
                    print('BEL node type ' + str_temp + ' not known -- defaulting to continuous')
            
            node_dict[entity_list[i]] = gn.cg_node(np.sum(adj_mat[:,i]),entity_list[i],node_type)
        
        self.node_dict = node_dict
        
        self.cond_list = []
        
        self.sample_dict = {}
        
        self.parent_ind_list = []
        self.child_ind_list = []
        self.parent_name_dict = {}
        self.child_name_dict = {}
        
        self.parent_ind_list = [np.where(self.adj_mat[:,i] > 0)[0] for i in range(0,n_nodes)]
        self.child_ind_list = [np.where(self.adj_mat[i,:] > 0)[0] for i in range(0,n_nodes)]
        
        for i in range(0,n_nodes):
            self.parent_name_dict[entity_list[i]] = [entity_list[item] for item in self.parent_ind_list[i]]
            self.child_name_dict[entity_list[i]] = [entity_list[item] for item in self.child_ind_list[i]]

        return
    
    def remove_edge(self,edge_rem):
        # remove all of the edges in edge_rem
        
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
    
    def prob_init(self,data_in):
        # initialize all of the nodes' probability distributions given data_in
        
        exog_list = []
        prob_dict = {}
        
        for name in self.node_dict:
            i = self.entity_list.index(name)
            data_in_temp = data_in[:,self.parent_ind_list[i]]
            data_out_temp = data_in[:,i]
            print(name)
            self.node_dict[name].p_init(data_in_temp,data_out_temp)
            
            if self.node_dict[name].n_inputs == 0:
                exog_list.append(name)
            #prob_dict[name] = self.node_dict[name].prob_dist
        
        self.exog_list = exog_list
        #self.prob_dict = prob_dict

        return
        
    def model_sample(self):
        # produce a dictionary of samples for all variables in the graph
        
        # define exogenous samples
        
        sample_dict = {}
        
        for item in self.exog_list:
            sample_dict[item] = self.node_dict[item].sample()
            
        flag = 0
        while flag == 0:
            
            # find all nodes not in sample_dict with parents entirely in sample dict and sample those nodes
            for item in self.entity_list:
                if (item not in sample_dict 
                    and np.all([item2 in sample_dict for item2 in self.parent_name_dict[item]])):
                    
                    sample_dict[item] = self.node_dict[item].sample(
                        torch.tensor([sample_dict[item2] for item2 in self.parent_name_dict[item]]))
            
            # if sample dict has all of the nodes in entity list, stop
            if sorted([item for item in sample_dict]) == sorted(self.entity_list):
                flag = 1
            
        
        return sample_dict
    
    def model_cond_sample(self,data_dict):
        # sample the graph given the conditioned variables in data_dict
        
        data_in = {}
        for item in data_dict:
            data_in[item] = data_dict[item]
        
        cond_model = pyro.condition(self.model_sample,data=data_in)
        return cond_model()
        
    def model_do_sample(self,do_dict):
        # sample the graph given the do-variables in do_dict
        
        data_in = {}
        for item in do_dict:
            data_in[item] = do_dict[item]
        
        do_model = pyro.do(self.model_sample,data=data_in)
        return do_model()
    
    def model_do_cond_sample(self,do_dict,data_dict):
        # sample the graph given do-variables in do_dict and conditioned variables in data_dict
        
        if np.any([[item1 == item2 for item1 in do_dict] for item2 in data_dict]):
            print('overlapping lists!')
            return
        else:
            do_dict_in = {}
            for item in do_dict:
                do_dict_in[item] = do_dict[item]
                
            data_dict_in = {}
            for item in data_dict:
                data_dict_in[item] = data_dict[item]
            
            do_model = pyro.do(self.model_sample,data=do_dict_in)
            cond_model = pyro.condition(do_model,data=data_dict_in)
            return cond_model()
    
    def model_counterfact(self,obs_dict,do_dict_counter):
        # find conditional distribution on exogenous variables given observations in obs_dict 
        # and do variable values in do_dict_counter
        cond_dict = self.model_cond_sample(obs_dict)
        cond_dict_temp = {}
        for item in self.exog_list:
            cond_dict_temp[item] = cond_dict[item]
        
        # evaluate observed variables given this condition distribution and do_dict_counter do-variables
        return self.model_do_cond_sample(do_dict_counter,cond_dict_temp)
        
        
    def cond_mut_info(self,target,test,cond,data_in):
        # calculate the conditional mutual information between target and test given data_in - I(target:test|cond)
        # just uses input data, but have to bin data (creating discrete distribution) to perform calculations
        
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
        # do the G-test on a single variable of interest 
        # determine if causal graph captures underlying distribution
        # have to bin data to perform calculations
        
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
        # calculate the total effect of changing an intervention from do_dict_prime values to do_dict values
        # on the variables in target
        
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
        # calculate the controlled direct effect of changing an intervention from do_dict_prime to do_dict values
        # on the variables in target given fixed mediating values
        
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
        # calculate the natural direct effect of changing an intervention from do_dict_prime values to do_dict 
        # values on the variables in target
        
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
        # calculate the natural indirect effect of changing an intervention from do_dict_prime values to do_dict 
        # values on the variables in target
        
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
        # write the causal graph to a text file to import into causal fusion
        
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
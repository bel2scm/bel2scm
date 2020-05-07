import numpy as np
import scipy as sp
import networkx as nx

from scipy import stats

import pybel as pb
import json

import torch
import pyro

# generate data for the ice cream use case

def indep_vars(n_samples):
    
    T_list = []
    C_list = []
    P_list = []
    
    for i in range(0,n_samples):
        
        T_list.append(pyro.sample("T_{}".format(i), pyro.distributions.LogNormal(2.96,0.2)))
        
        C_list.append(0.5*pyro.sample("C1_{}".format(i),pyro.distributions.Beta(1,1+T_list[-1]/10)) 
            + 0.5*pyro.sample("C2_{}".format(i),pyro.distributions.Uniform(0,1)))
        P_list.append(0.5*pyro.sample("P1_{}".format(i), pyro.distributions.Exponential(1))
            + 0.5*pyro.sample("P2.{}".format(i),pyro.distributions.Exponential(1/(C_list[-1]+1))))
        
    return T_list,C_list,P_list

def dep_vars(T_list,C_list,P_list):
    
    n_pts = len(T_list)
    
    I_list = []
    
    for i in range(0,n_pts):
        
        T_temp = T_list[i]
        C_temp = C_list[i]
        P_temp = P_list[i]
        
        if P_temp > 2.5 or T_temp < 15:
            I_list.append(pyro.sample("I_{}".format(i),pyro.distributions.Bernoulli(0))+1e-6)
        else:
            I_list.append(pyro.sample("I_{}".format(i),
                pyro.distributions.Beta(2*(2.5-P_temp)*(T_temp-12)/(2.5*12),2))+1e-6)
        
    return I_list

def data_gen(n_data):
    temp,cloud,precip = indep_vars(n_data)
    icream = dep_vars(temp,cloud,precip)
    tot_data = torch.Tensor([temp,cloud,precip,icream])
    return tot_data.T
    
import numpy as np
import torch
import pyro

# create class of causal graph nodes

class cg_node():
    def __init__(self,n_inputs,name,node_type):
        
        self.n_inputs = n_inputs
        self.name = name
        self.node_type = node_type
        
        if n_inputs == 0:
            self.label = 'exogenous'
        else:
            self.label = 'endogenous'
            
        return
    
    def batch_calc(self):
        # break up data into mini-batches for training the node parameters
        self.bat_per_epoch = int(self.n_data/self.batch_size)

        indices = np.zeros((self.batch_size,self.bat_per_epoch)).astype(int)
        ind_temp = np.linspace(0,self.n_data-1,self.n_data).astype(int)
        np.random.shuffle(ind_temp)

        for i in range(0,self.bat_per_epoch):
            indices[:,i] = ind_temp[(self.batch_size*i):(self.batch_size*(i+1))]

        return indices
    
    def reg_calc_bin(self,data_vec,var_j,var_jk):
        # calculate the probability associated with the Bernoulli distribution given the input data
        # use sigmoid to ensure that 0 < p < 1
        return torch.sigmoid(torch.matmul(data_vec,var_j[:self.n_inputs]) + var_j[self.n_inputs] 
            + torch.sum(torch.matmul(data_vec,var_jk)*data_vec,dim=-1))

    def reg_calc_gamma(self,data_vec,var_j,var_jk):
        # calculate the alpha or beta for the Gamma distribution given the input data
        # use abs to ensure that alpha and beta are >= 0
        return torch.abs(torch.matmul(data_vec,var_j[:self.n_inputs]) + var_j[self.n_inputs] 
            + torch.sum(torch.matmul(data_vec,var_jk)*data_vec,dim=-1))

    def bin_log_fcn(self,data_in,p_var):
        # calculate negative (for minimization, not maximization) log-likelihood for the binary variable case
        return -torch.mean(data_in*torch.log(p_var) + (1-data_in)*torch.log(1-p_var))

    def gamma_log_fcn(self,data_in,alpha_var,beta_var):
        # calculate negative (for minimization, not maximization) log-likelihood for the continuous variable case
        return -torch.mean(alpha_var*torch.log(beta_var) + (alpha_var-1)*torch.log(data_in)
            - beta_var*data_in - torch.lgamma(alpha_var))
    
    def p_init(self,input_data,var_data):
        # calculate probability distribution parameters for node probability distributions
        # distribution parameters are quadratic functions of input variables
        # assumes that output variables are either Benoulli- or Gamma-distributed
        
        # need to adjust max_epochs and learning rate
        
        self.n_data = len(input_data)
        
        self.input_data = input_data
        self.var_data = var_data
        
        self.batch_size = 32
        self.max_epoch = 100*self.n_inputs
        self.num_iters = int(self.n_data/self.batch_size)
        
        if self.n_inputs > 0:
            ind_key = np.zeros((self.batch_size,self.num_iters,self.max_epoch))
            ind_key = ind_key.astype(int)
            for i in range(0,self.max_epoch):
                ind_key[:,:,i] = self.batch_calc()
                
            self.ind_key = ind_key

        if self.node_type == 'binary':
            
            if self.n_inputs == 0:
                self.p_jk = []
                self.p_j = -torch.log(1/torch.mean(var_data)-1)
                loss_tot = bin_log_fcn(data_in,self.p_j)
                
            else:
                self.p_j = torch.ones(self.n_inputs+1,requires_grad=True)
                self.p_jk = torch.ones(self.n_inputs,self.n_inputs,requires_grad=True)

                optimizer = torch.optim.Adam([self.p_j,self.p_jk], lr=1/10**(2+self.n_inputs))

                # train log_like_fcn wrt self.p_j                
                
                for i in range(0,self.max_epoch):
                    for j in range(0,self.bat_per_epoch):

                        optimizer.zero_grad()
                        loss = self.bin_log_fcn(var_data[ind_key[:,j,i]],
                            self.reg_calc_bin(input_data[ind_key[:,j,i],:],self.p_j,self.p_jk))
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        
                loss_tot = self.bin_log_fcn(var_data,self.reg_calc_bin(input_data,self.p_j,self.p_jk))                
            
        elif self.node_type == 'continuous':            
            
            if self.n_inputs == 0:
                
                self.alpha_jk = []
                self.beta_jk = []
                
                s_temp = torch.log(torch.mean(var_data)) - torch.mean(torch.log(var_data))
                alpha_temp = (3-s_temp + torch.sqrt((s_temp-3)**2 + 24*s_temp))/(12*s_temp)
                beta_temp = alpha_temp/torch.mean(var_data)
                self.alpha_j = alpha_temp
                self.beta_j = beta_temp
                
                loss_tot = self.gamma_log_fcn(var_data,self.alpha_j,self.beta_j)
                
            else:
                
                # initialize variables
                
                alpha_j_temp = np.zeros(self.n_inputs+1)
                alpha_jk_temp = np.zeros((self.n_inputs,self.n_inputs))
                
                for i in range(0,self.n_inputs):
                    alpha_j_temp[i] = 1/torch.mean(input_data[:,i]).item()
                    for j in range(0,self.n_inputs):
                        alpha_jk_temp[i,j] = 1/(torch.mean(input_data[:,i]).item()
                            *torch.mean(input_data[:,i]).item())
                var_mean = torch.mean(var_data).item()
                        
                alpha_j = torch.tensor(alpha_j_temp*var_mean,requires_grad=True)
                alpha_jk = torch.tensor(alpha_jk_temp*var_mean,requires_grad=True)
                beta_j = torch.tensor(alpha_j_temp,requires_grad=True)
                beta_jk = torch.tensor(alpha_jk_temp,requires_grad=True)

                optimizer = torch.optim.Adam([alpha_j,alpha_jk,beta_j,beta_jk], lr=1/10**(2+self.n_inputs))
                
                # train log_like_fcn wrt self.alpha_j, self.beta_j
                
                for i in range(0,self.max_epoch):
                    for j in range(0,self.bat_per_epoch):

                        optimizer.zero_grad()
                        loss = self.gamma_log_fcn(var_data[ind_key[:,j,i]],
                            self.reg_calc_gamma(input_data[ind_key[:,j,i],:],alpha_j,alpha_jk),
                            self.reg_calc_gamma(input_data[ind_key[:,j,i],:],beta_j,beta_jk))
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        
                loss_tot = self.gamma_log_fcn(var_data,self.reg_calc_gamma(input_data,alpha_j,alpha_jk),
                    self.reg_calc_gamma(input_data,beta_j,beta_jk))
                        
                self.alpha_j = alpha_j
                self.alpha_jk = alpha_jk
                self.beta_j = beta_j
                self.beta_jk = beta_jk
                
        else:
            print('node type not supported')
            
        print(loss_tot)
        print()
        self.log_error = loss_tot
        
        return
    
    def sample(self,data_in=[]):
        # sample your output variable given input data (for a non-exogenous variable)
        
        if self.node_type == 'binary':
            if self.n_inputs == 0:
                p_temp = torch.sigmoid(self.p_j)
            else:
                p_temp = self.reg_calc_bin(data_in,self.p_j,self.p_jk)
            
            return torch.squeeze(pyro.sample(self.name,pyro.distributions.Bernoulli(probs=p_temp)).int())
        
        elif self.node_type == 'continuous':
            if self.n_inputs == 0:
                alpha_temp = self.alpha_j
                beta_temp = self.beta_j
            else:                
                alpha_temp = self.reg_calc_gamma(data_in,self.alpha_j,self.alpha_jk)
                beta_temp = self.reg_calc_gamma(data_in,self.beta_j,self.beta_jk)
            
            return torch.squeeze(pyro.sample(self.name,pyro.distributions.Gamma(alpha_temp,beta_temp)))
            
        else:
            print('node type not supported')


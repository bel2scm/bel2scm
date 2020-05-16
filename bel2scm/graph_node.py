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

class bayes_node(cg_node):
    
    def __init__(self,n_inputs,name,node_type):
        super().__init__(n_inputs,name,node_type)
        
        return
        
    
    def model_Bernoulli(self,var_data,parent_data):
        mu0 = torch.zeros(self.n_inputs+1)
        sig_sq0 = torch.ones(self.n_inputs+1)
        n_data = var_data.size()[0]

        p_j = pyro.sample('p_' + self.name,pyro.distributions.Normal(mu0,torch.sqrt(sig_sq0)))
        
        with pyro.plate('observe_data'):
            p = p_j[0]*torch.ones(n_data) + torch.matmul(parent_data,p_j[1:])
            pyro.sample('obs_' + self.name,pyro.distributions.Bernoulli(torch.sigmoid(p)),obs=var_data)
            
        return

    def guide_Bernoulli(self,var_data,parent_data):        

        mu_j = pyro.param('mu_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        sig_sq_j = pyro.param('sig_sq_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)

        pyro.sample('p',pyro.distributions.Normal(mu_j,torch.sqrt(sig_sq_j)))
        
        return
        
        
    def model_Normal(self,var_data,parent_data):
        alpha0 = torch.ones(self.n_inputs+1)
        beta0 = torch.ones(self.n_inputs+1)
        mu0 = torch.ones(self.n_inputs+1)
        nu0 = torch.ones(self.n_inputs+1)
        n_data = var_data.size()[0]
        
        sig_recip_j = pyro.sample('sig_recip_' + self.name,pyro.distributions.Gamma(alpha0,beta0))
        # pyro doesn't seem to have InvGamma, so using Gamma and then taking reciprocal
        
        mu_j = pyro.sample('mu_' + self.name,pyro.distributions.Normal(mu0,1./(nu0*sig_recip_j)))
        
        with pyro.plate('observe_data'):
            mu = mu_j[0]*torch.ones(n_data) + torch.matmul(parent_data,mu_j[1:])
            sig_sq = torch.ones(n_data)/sig_recip_j[0] + torch.matmul(parent_data,1./sig_recip_j[1:])
            pyro.sample('obs_' + self.name,pyro.distributions.Normal(mu,sig_sq),obs=var_data)
            
        return

    def guide_Normal(self,var_data,parent_data):
        alpha_j = pyro.param('alpha_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        beta_j = pyro.param('beta_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        mu0_j = pyro.param('mu0_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        nu_j = pyro.param('nu_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        
        sig_recip_j = pyro.sample('sig_recip_' + self.name,pyro.distributions.Gamma(alpha_j,beta_j))
        # pyro doesn't seem to have InvGamma, so using Gamma and then taking reciprocal
        
        mu_j = pyro.sample('mu_' + self.name,pyro.distributions.Normal(mu0_j,1./(nu_j*sig_recip_j)))
        
        return
    
    
    def model_Lognormal(self,var_data,parent_data):
        alpha0 = torch.ones(self.n_inputs+1)
        beta0 = torch.ones(self.n_inputs+1)
        mu0 = torch.ones(self.n_inputs+1)
        nu0 = torch.ones(self.n_inputs+1)
        n_data = var_data.size()[0]

        sig_recip_j = pyro.sample('sig_recip_' + self.name,pyro.distributions.Gamma(alpha0,beta0))
        # pyro doesn't seem to have InvGamma, so using Gamma and then taking reciprocal
        
        mu_j = pyro.sample('mu_' + self.name,pyro.distributions.Normal(mu0,1./(nu0*sig_recip_j)))
        
        with pyro.plate('observe_data'):
            mu = mu_j[0]*torch.ones(n_data) + torch.matmul(parent_data,mu_j[1:])
            sig_sq = torch.ones(n_data)/sig_recip_j[0] + torch.matmul(parent_data,1./sig_recip_j[1:])
            pyro.sample('obs_' + self.name,pyro.distributions.Normal(mu,sig_sq),obs=torch.log(var_data))
            
        return

    def guide_Lognormal(self,var_data,parent_data):
        
        alpha_j = pyro.param('alpha_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        beta_j = pyro.param('beta_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        mu0_j = pyro.param('mu0_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        nu_j = pyro.param('nu_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        
        sig_recip_j = pyro.sample('sig_recip_' + self.name,pyro.distributions.Gamma(alpha_j,beta_j))
        # pyro doesn't seem to have InvGamma, so using Gamma and then taking reciprocal
        
        mu_j = pyro.sample('mu_' + self.name,pyro.distributions.Normal(mu0_j,1./(nu_j*sig_recip_j)))
        
        return
    
    
    def model_Exponential(self,var_data,parent_data):
        alpha0 = torch.ones(self.n_inputs+1)
        beta0 = torch.ones(self.n_inputs+1)
        n_data = var_data.size()[0]

        lamb_j = pyro.sample('lamb_' + self.name,pyro.distributions.Gamma(alpha0,beta0))

        with pyro.plate('observe_data'):
            lamb = lamb_j[0]*torch.ones(n_data) + torch.matmul(parent_data,lamb_j[1:])
            pyro.sample('obs_' + self.name,pyro.distributions.Exponential(torch.abs(lamb)),obs=var_data)
            
        return

    def guide_Exponential(self,var_data,parent_data):
        
        alpha_j = pyro.param('alpha_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        beta_j = pyro.param('beta_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)

        lamb_j = pyro.sample('lamb_' + self.name,pyro.distributions.Gamma(alpha_j,beta_j))
        
        return
        
        
    def model_Gamma(self,var_data,parent_data):
        p0 = torch.ones(self.n_inputs+1)
        q0 = torch.ones(self.n_inputs+1)
        r0 = torch.ones(self.n_inputs+1)
        s0 = torch.ones(self.n_inputs+1)
        n_data = var_data.size()[0]

        alpha_j = pyro.sample('alpha_' + self.name,pyro.distributions.Gamma(r0+1.,p0))
        
        beta_j = pyro.sample('beta_' + self.name,pyro.distributions.Gamma(alpha_j*s0+1.,q0))

        with pyro.plate('observe_data'):
            alpha = alpha_j[0]*torch.ones(n_data) + torch.matmul(parent_data,alpha_j[1:])
            beta = beta_j[0]*torch.ones(n_data) + torch.matmul(parent_data,beta_j[1:])
            pyro.sample('obs_' + self.name,
                pyro.distributions.Gamma(torch.abs(alpha),torch.abs(beta)),obs=var_data)
            
        return

    def guide_Gamma(self,var_data,parent_data):
        
        p_j = pyro.param('p_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        q_j = pyro.param('q_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        r_j = pyro.param('r_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)
        s_j = pyro.param('s_j_' + self.name,torch.ones(self.n_inputs+1),
            constraint=torch.distributions.constraints.positive)

        alpha_j = pyro.sample('alpha_' + self.name,pyro.distributions.Gamma(r_j+1.,p_j))
        
        beta_j = pyro.sample('beta_' + self.name,pyro.distributions.Gamma(alpha_j*s_j+1.,q_j))
        
        return

        
    
    def prob_init(self,input_data,var_data,lr):
        
        # set up the optimizer
        adam_params = {"lr": lr, "betas": (0.90, 0.999)}
        optimizer = pyro.optim.Adam(adam_params)

        # setup the inference algorithm
        if self.node_type == 'Bernoulli':
            svi = pyro.infer.SVI(self.model_Bernoulli,self.guide_Bernoulli,optimizer,loss=pyro.infer.Trace_ELBO())
        elif self.node_type == 'Normal':
            svi = pyro.infer.SVI(self.model_Normal,self.guide_Normal,optimizer,loss=pyro.infer.Trace_ELBO())
        elif self.node_type == 'Log-Normal':
            svi = pyro.infer.SVI(self.model_Lognormal,self.guide_Lognormal,optimizer,loss=pyro.infer.Trace_ELBO())
        elif self.node_type == 'Exponential':
            svi = pyro.infer.SVI(self.model_Exponential,self.guide_Exponential,optimizer,loss=pyro.infer.Trace_ELBO())
        elif self.node_type == 'Gamma':
            svi = pyro.infer.SVI(self.model_Gamma,self.guide_Gamma,optimizer,loss=pyro.infer.Trace_ELBO())
        else:
            print('node type ' + self.node_type + ' not supported for node ' + self.name)
            

        n_steps = 5000
        # do gradient steps
        for step in range(n_steps):
            loss = svi.step(var_data,input_data)
            if step%100 == 0:
                print(loss)
            
            
        if self.node_type == 'Bernoulli':
            self.mu0_j = pyro.param('mu0_j_' + self.name)
            self.sig_recip_j = pyro.param('sig_recip_j_' + self.name) 
            
        elif self.node_type == 'Exponential':
            self.alpha_j = pyro.param('alpha_j_' + self.name)
            self.beta_j = pyro.param('beta_j_' + self.name)
            
        elif self.node_type == 'Normal' or self.node_type == 'Log-Normal':
            self.alpha_j = pyro.param('alpha_j_' + self.name)
            self.beta_j = pyro.param('beta_j_' + self.name)
            self.mu0_j = pyro.param('mu0_j_' + self.name)
            self.nu_j = pyro.param('nu_j_' + self.name)
            
        elif self.node_type == 'Gamma':
            self.p_j = pyro.param('p_j_' + self.name)
            self.q_j = pyro.param('q_j_' + self.name) 
            self.r_j = pyro.param('r_j_' + self.name)
            self.s_j = pyro.param('s_j_' + self.name)

        else:
            print('node type ' + self.node_type + ' not supported for node ' + self.name)
            
        return
    
    def sample(self,data_in=[]):
        
        if self.n_inputs == 0:
            if self.node_type == 'Bernoulli':
                p = pyro.sample(self.name + '_p',pyro.distributions.Normal(self.mu_j[0],self.sig_sq_j[0]))
                y = pyro.sample(self.name + '_y',pyro.distributions.Bernoulli(torch.sigmoid(p)))
                
                
            elif self.node_type == 'Gamma':
                alpha = pyro.sample(self.name + '_alpha',pyro.distributions.Gamma(self.r_j[0],self.p_j[0]))
                beta = pyro.sample(self.name + '_beta',pyro.distributions.Gamma(alpha*self.s_j[0] + 1,self.q_j[0]))
                y = pyro.sample(self.name + '_y',pyro.distributions.Gamma(alpha,beta))
                
                
            elif self.node_type == 'Normal':
                sig_recip = pyro.sample(self.name + '_sig',pyro.distributions.Gamma(self.alpha_j[0],self.beta_j[0]))
                mu = pyro.sample(self.name + '_mu',
                    pyro.distributions.Normal(self.mu0_j[0],1./(self.nu_j[0]*torch.sqrt(sig_recip))))
                y = pyro.sample(self.name + '_y',pyro.distributions.Normal(mu,1./torch.sqrt(sig_recip)))
                                 
                
            elif self.node_type == 'Log-Normal':
                sig_recip = pyro.sample(self.name + '_sig',pyro.distributions.Gamma(self.alpha_j[0],self.beta_j[0]))
                mu = pyro.sample(self.name + '_mu',
                    pyro.distributions.Normal(self.mu0_j[0],1./(self.nu_j[0]*torch.sqrt(sig_recip))))
                y = pyro.sample(self.name + '_y',pyro.distributions.LogNormal(mu,1./torch.sqrt(sig_recip)))
                                 
                
            elif self.node_type == 'Exponential':
                lamb = pyro.sample(self.name + '_lamb',pyro.distributions.Gamma(self.alpha_j[0],self.beta_j[0]))
                y = pyro.sample(self.name + '_y',pyro.distributions.Exponential(lamb))
                
            else:
                 print('node type ' + self.node_type + ' not supported for node ' + self.name) 
                                
                
        else:
            if self.node_type == 'Bernoulli':
                p_j = pyro.sample(self.name + '_p',pyro.distributions.Normal(self.mu_j,self.sig_sq_j))

                mu = p_j[0] + torch.dot(data_in,p_j[1:])
                                 
                y = pyro.sample(self.name + '_y',pyro.distributions.Bernoulli(torch.sigmoid(p)))
                
                
            elif self.node_type == 'Gamma':
                
                alpha_j = pyro.sample(self.name + 'alpha',pyro.distributions.Gamma(self.r_j+1.,self.p_j))

                beta_j = pyro.sample('beta',pyro.distributions.Gamma(alpha_j*self.s_j+1.,self.q_j))

                alpha = alpha_j[0] + torch.dot(data_in,alpha_j[1:])
                beta = beta_j[0] + torch.dot(data_in,beta_j[1:])
                
                y = pyro.sample(self.name + '_y',pyro.distributions.Gamma(alpha,beta))
                
                
            elif self.node_type == 'Normal':
                sig_recip_j = pyro.sample(self.name + 'sig_sq',pyro.distributions.Gamma(self.alpha_j,self.beta_j))
                # pyro doesn't seem to have InvGamma, so using Gamma and then taking reciprocal

                mu_j = pyro.sample(self.name + 'mu',
                    pyro.distributions.Normal(self.mu0_j,1./(self.nu_j*sig_recip_j))) 

                mu = mu_j[0] + torch.dot(data_in,mu_j[1:])
                sig_sq = 1./sig_recip_j[0] + torch.dot(data_in,1./sig_recip_j[1:])
                
                y = pyro.sample(self.name + '_y',pyro.distributions.Normal(mu,torch.sqrt(sig_sq)))
                
                
            elif self.node_type == 'Log-Normal':
                sig_recip_j = pyro.sample(self.name + 'sig_sq',pyro.distributions.Gamma(self.alpha_j,self.beta_j))
                # pyro doesn't seem to have InvGamma, so using Gamma and then taking reciprocal

                mu_j = pyro.sample(self.name + 'mu',
                    pyro.distributions.Normal(self.mu0_j,1./(self.nu_j*sig_recip_j))) 

                mu = mu_j[0] + torch.dot(data_in,mu_j[1:])
                sig_sq = 1./sig_recip_j[0] + torch.dot(data_in,1./sig_recip_j[1:])
                
                y = pyro.sample(self.name + '_y',pyro.distributions.LogNormal(mu,torch.sqrt(sig_sq)))
                
                
            elif self.node_type == 'Exponential':
                lamb_j = pyro.sample(self.name + '_lamb',pyro.distributions.Gamma(self.alpha_j,self.beta_j))

                lamb = lamb_j[0] + torch.dot(data_in,lamb_j[1:])
                                 
                y = pyro.sample(self.name + '_y',pyro.distributions.Exponential(lamb))
            
            else:
                print('node type ' + self.node_type + ' not supported for node ' + self.name)
                
        return y
    
class mle_node(cg_node):
    
    def __init__(self,n_inputs,name,node_type):
        super().__init__(n_inputs,name,node_type)
        
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

    def reg_calc_abs(self,data_vec,var_j,var_jk):
        # calculate the parameter given the input data
        # use abs to ensure that its >= 0
        return torch.abs(torch.matmul(data_vec,var_j[:self.n_inputs]) + var_j[self.n_inputs] 
            + torch.sum(torch.matmul(data_vec,var_jk)*data_vec,dim=-1))
    
    def reg_calc(self,data_vec,var_j,var_jk):
        # calculate the parameter given the input data
        # use abs to ensure that its >= 0
        return (torch.matmul(data_vec,var_j[:self.n_inputs]) + var_j[self.n_inputs] 
            + torch.sum(torch.matmul(data_vec,var_jk)*data_vec,dim=-1))

    def bernoulli_log_fcn(self,data_in,p_var):
        # calculate negative (for minimization, not maximization) log-likelihood for the Bernoulli distribution
        return -torch.mean(data_in*torch.log(p_var) + (1-data_in)*torch.log(1-p_var))

    def gamma_log_fcn(self,data_in,alpha_var,beta_var):
        # calculate negative (for minimization, not maximization) log-likelihood for the Gamma distribution
        return -torch.mean(alpha_var*torch.log(beta_var) + (alpha_var-1)*torch.log(data_in)
            - beta_var*data_in - torch.lgamma(alpha_var))
    
    def normal_log_fcn(self,data_in,mu_var,sig_sq_var):
        # calculate negative (for minimization, not maximization) log-likelihood for the Normal distribution       
        return torch.mean(torch.log(sig_sq_var))/2. + torch.mean((data_in - mu_var)**2/sig_sq_var)/2.
    
    def lognormal_log_fcn(self,data_in,mu_var,sig_sq_var):
        # calculate negative (for minimization, not maximization) log-likelihood for the Lognormal distribution       
        return torch.mean(torch.log(sig_sq_var))/2. + torch.mean((torch.log(data_in) - mu_var)**2/sig_sq_var)/2.
    
    def exponential_log_fcn(self,data_in,lamb_var):
        # calculate negative (for minimization, not maximization) log-likelihood for the Exponential distribution       
        return -torch.mean(torch.log(lamb_var) - lamb_var*data_in)
    
    def bernoulli_init(self,input_data,var_data,lr):
        # calculate Bernoulli MLE
        
        if self.n_inputs == 0:
            self.p_jk = []
            self.p_j = -torch.log(1/torch.mean(var_data)-1)
            loss_tot = self.bernoulli_log_fcn(data_in,self.p_j)

        else:
            self.p_j = torch.ones(self.n_inputs+1,requires_grad=True)
            self.p_jk = torch.zeros(self.n_inputs,self.n_inputs,requires_grad=True)

            optimizer = torch.optim.Adam([self.p_j,self.p_jk], lr=lr)

            # train log_like_fcn wrt self.p_j                

            for i in range(0,self.max_epoch):
                for j in range(0,self.bat_per_epoch):

                    optimizer.zero_grad()
                    loss = self.bernoulli_log_fcn(var_data[self.ind_key[:,j,i]],
                        self.reg_calc_bin(input_data[self.ind_key[:,j,i],:],self.p_j,self.p_jk))
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

            loss_tot = self.bin_log_fcn(var_data,self.reg_calc_bin(input_data,self.p_j,self.p_jk))
        return loss_tot
    
    def gamma_init(self,input_data,var_data,lr):
        # calculate Gamma MLE
        
        if self.n_inputs == 0:
                
            self.alpha_jk = []
            self.beta_jk = []

            s_temp = torch.log(torch.mean(var_data)) - torch.mean(torch.log(var_data))
            self.alpha_j = (3-s_temp + torch.sqrt((s_temp-3)**2 + 24*s_temp))/(12*s_temp)
            self.beta_j = self.alpha_j/torch.mean(var_data)

            loss_tot = self.gamma_log_fcn(var_data,self.alpha_j,self.beta_j)

        else:
            # initialize variables

            #alpha_j_temp = np.zeros(self.n_inputs+1)
            #alpha_jk_temp = np.zeros((self.n_inputs,self.n_inputs))

            #for i in range(0,self.n_inputs):
                #alpha_j_temp[i] = 1/torch.mean(input_data[:,i]).item()
                #for j in range(0,self.n_inputs):
                    #alpha_jk_temp[i,j] = 1/(torch.mean(input_data[:,i]).item()
                        #*torch.mean(input_data[:,i]).item())
            #var_mean = torch.mean(var_data).item()

            #self.alpha_j = torch.tensor(alpha_j_temp*var_mean,requires_grad=True)
            #self.alpha_jk = torch.tensor(alpha_jk_temp*var_mean,requires_grad=True)
            #self.beta_j = torch.tensor(alpha_j_temp,requires_grad=True)
            #self.beta_jk = torch.tensor(alpha_jk_temp,requires_grad=True)
            
            self.alpha_j = torch.ones(self.n_inputs+1,requires_grad=True)
            self.alpha_jk = torch.zeros(self.n_inputs,self.n_inputs,requires_grad=True)
            self.beta_j = torch.ones(self.n_inputs+1,requires_grad=True)
            self.beta_jk = torch.zeros(self.n_inputs,self.n_inputs,requires_grad=True)

            optimizer = torch.optim.Adam([self.alpha_j,self.alpha_jk,self.beta_j,self.beta_jk], 
                lr=lr)

            # train log_like_fcn wrt self.alpha_j, self.beta_j

            for i in range(0,self.max_epoch):
                for j in range(0,self.bat_per_epoch):

                    optimizer.zero_grad()
                    loss = self.gamma_log_fcn(var_data[self.ind_key[:,j,i]],
                        self.reg_calc_abs(input_data[self.ind_key[:,j,i],:],self.alpha_j,self.alpha_jk),
                        self.reg_calc_abs(input_data[self.ind_key[:,j,i],:],self.beta_j,self.beta_jk))
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

            loss_tot = self.gamma_log_fcn(var_data,self.reg_calc_abs(input_data,self.alpha_j,self.alpha_jk),
                self.reg_calc_abs(input_data,self.beta_j,self.beta_jk))

        return loss_tot
    
    def normal_init(self,input_data,var_data,lr):
        # calculate Normal MLE
        
        if self.n_inputs == 0:
                
            self.mu_jk = []
            self.sig_sq_jk = []

            self.mu_j = torch.mean(var_data)
            self.sig_sq_j = torch.mean((var_data-torch.mean(var_data))**2)

            loss_tot = self.normal_log_fcn(var_data,self.mu_j,self.sig_sq_j)

        else:
            # initialize variables

            self.mu_j = torch.ones(self.n_inputs+1,requires_grad=True)
            self.mu_jk = torch.zeros(self.n_inputs,self.n_inputs,requires_grad=True)
            self.sig_sq_j = torch.ones(self.n_inputs+1,requires_grad=True)
            self.sig_sq_jk = torch.zeros(self.n_inputs,self.n_inputs,requires_grad=True)

            optimizer = torch.optim.Adam([self.mu_j,self.mu_jk,self.sig_sq_j,self.sig_sq_jk], 
                lr=lr)

            # train log_like_fcn

            for i in range(0,self.max_epoch):
                for j in range(0,self.bat_per_epoch):

                    optimizer.zero_grad()
                    loss = self.normal_log_fcn(var_data[self.ind_key[:,j,i]],
                        self.reg_calc(input_data[self.ind_key[:,j,i],:],self.mu_j,self.mu_jk),
                        self.reg_calc_abs(input_data[self.ind_key[:,j,i],:],self.sig_sq_j,self.sig_sq_jk))
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

            loss_tot = self.normal_log_fcn(var_data,self.reg_calc(input_data,self.mu_j,self.mu_jk),
                self.reg_calc_abs(input_data,self.sig_sq_j,self.sig_sq_jk))

        return loss_tot
    
    def lognormal_init(self,input_data,var_data,lr):
        # calculate Lognormal MLE
        
        if self.n_inputs == 0:
                
            self.mu_jk = []
            self.sig_sq_jk = []

            self.mu_j = torch.mean(torch.log(var_data))
            self.sig_sq_j = torch.mean((torch.log(var_data)-torch.mean(torch.log(var_data)))**2)

            loss_tot = self.lognormal_log_fcn(var_data,self.mu_j,self.sig_sq_j)

        else:
            # initialize variables

            self.mu_j = torch.ones(self.n_inputs+1,requires_grad=True)
            self.mu_jk = torch.zeros(self.n_inputs,self.n_inputs,requires_grad=True)
            self.sig_sq_j = torch.ones(self.n_inputs+1,requires_grad=True)
            self.sig_sq_jk = torch.zeros(self.n_inputs,self.n_inputs,requires_grad=True)

            optimizer = torch.optim.Adam([self.mu_j,self.mu_jk,self.sig_sq_j,self.sig_sq_jk], 
                lr=lr)

            # train log_like_fcn

            for i in range(0,self.max_epoch):
                for j in range(0,self.bat_per_epoch):

                    optimizer.zero_grad()
                    loss = self.lognormal_log_fcn(var_data[self.ind_key[:,j,i]],
                        self.reg_calc(input_data[self.ind_key[:,j,i],:],self.mu_j,self.mu_jk),
                        self.reg_calc_abs(input_data[self.ind_key[:,j,i],:],self.sig_sq_j,self.sig_sq_jk))
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

            loss_tot = self.lognormal_log_fcn(var_data,self.reg_calc(input_data,self.mu_j,self.mu_jk),
                self.reg_calc_abs(input_data,self.sig_sq_j,self.sig_sq_jk))

        return loss_tot
    
    def exponential_init(self,input_data,var_data,lr):
        # calculate Exponential MLE
        
        if self.n_inputs == 0:
            self.lamb_jk = []
            self.lamb_j = 1/torch.mean(var_data)
            loss_tot = self.exponential_log_fcn(data_in,self.lamb_j)

        else:
            self.lamb_j = torch.ones(self.n_inputs+1,requires_grad=True)
            self.lamb_jk = torch.zeros(self.n_inputs,self.n_inputs,requires_grad=True)

            optimizer = torch.optim.Adam([self.lamb_j,self.lamb_jk], lr=lr)               

            for i in range(0,self.max_epoch):
                for j in range(0,self.bat_per_epoch):

                    optimizer.zero_grad()
                    loss = self.exponential_log_fcn(var_data[self.ind_key[:,j,i]],
                        self.reg_calc_abs(input_data[self.ind_key[:,j,i],:],self.lamb_j,self.lamb_jk))
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

            loss_tot = self.exponential_log_fcn(var_data,self.reg_calc_bin(input_data,self.lamb_j,self.lamb_jk))
        return loss_tot
                
    
    def prob_init(self,input_data,var_data,lr):
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

        if self.node_type == 'Bernoulli':
            loss_tot = self.bernoulli_init(input_data,var_data,lr)
                            
        elif self.node_type == 'Gamma':            
            loss_tot = self.gamma_init(input_data,var_data,lr)
            
        elif self.node_type == 'Normal':            
            loss_tot = self.normal_init(input_data,var_data,lr)
            
        elif self.node_type == 'Log-Normal':            
            loss_tot = self.lognormal_init(input_data,var_data,lr)
            
        elif self.node_type == 'Exponential':
            loss_tot = self.exponential_init(input_data,var_data,lr)
            
        else:
            print('node type not supported')
            
        print(loss_tot)
        print()
        self.log_error = loss_tot
        
        return
    
    def sample(self,data_in=[]):
        # sample your output variable given input data (for a non-exogenous variable)
        
        if self.node_type == 'Bernoulli':
            if self.n_inputs == 0:
                p_temp = torch.sigmoid(self.p_j)
            else:
                p_temp = self.reg_calc_bin(data_in,self.p_j,self.p_jk)
            
            return torch.squeeze(pyro.sample(self.name,pyro.distributions.Bernoulli(probs=p_temp)).int())
        
        elif self.node_type == 'Gamma':
            if self.n_inputs == 0:
                alpha_temp = self.alpha_j
                beta_temp = self.beta_j
            else:                
                alpha_temp = self.reg_calc_abs(data_in,self.alpha_j,self.alpha_jk)
                beta_temp = self.reg_calc_abs(data_in,self.beta_j,self.beta_jk)
            
            return torch.squeeze(pyro.sample(self.name,pyro.distributions.Gamma(alpha_temp,beta_temp)))
        
        elif self.node_type == 'Normal':
            if self.n_inputs == 0:
                mu_temp = self.mu_j
                sig_sq_temp = self.sig_sq_j
            else:                
                mu_temp = self.reg_calc(data_in,self.mu_j,self.mu_jk)
                sig_sq_temp = self.reg_calc_abs(data_in,self.sig_sq_j,self.sig_sq_jk)
            
            return torch.squeeze(pyro.sample(
                self.name,pyro.distributions.Normal(mu_temp,torch.sqrt(sig_sq_temp))))
        
        elif self.node_type == 'Log-Normal':
            if self.n_inputs == 0:
                mu_temp = self.mu_j
                sig_sq_temp = self.sig_sq_j
            else:                
                mu_temp = self.reg_calc(data_in,self.mu_j,self.mu_jk)
                sig_sq_temp = self.reg_calc_abs(data_in,self.sig_sq_j,self.sig_sq_jk)
            
            return torch.squeeze(pyro.sample(
                self.name,pyro.distributions.LogNormal(mu_temp,torch.sqrt(sig_sq_temp))))
        
        elif self.node_type == 'Exponential':
            if self.n_inputs == 0:
                lamb_temp = self.lamb_j
            else:
                lamb_temp = self.reg_calc_abs(data_in,self.lamb_j,self.lamb_jk)
            
            return torch.squeeze(pyro.sample(self.name,pyro.distributions.Exponential(lamb_temp)))
            
        else:
            print('node type not supported')
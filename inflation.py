from torch.optim.optimizer import Optimizer, required
import numpy as np
import torch

from pdb import set_trace as bp

class ECD_abs(Optimizer):
    '''
    Optimizer based on the new separable Hamiltonian and generalized bounces
    lr (float): learning rate, called Delta t in the paper (required)
    F0 (float): expected minimum of the objective
    deltaEn (float): initial energy
    eta (float): hyperparameter that controls the concentration of the measure (required).
                 It has to be >= 1. Increasing it concentrates the measure towards the bottom of the basin, and it is useful for pure optimization problems where the goal to find smallest loss. Tested up to eta = 5.
    consEn (bool): whether the energy is conserved or not
    weight_decay (float): weight decay, implemented as L^2 term
    nu (float): chaos hyperparameter
    s (float): regularization switch
    '''

    def __init__(self, params, lr=required, F0=0., eps1=1e-10, eps2=1e-40, deltaEn=0., nu=1e-5, s=1., weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, deltaEn=deltaEn, nu=nu, s=s, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.s = s
        if self.s == 1:
            self.deltaEn = deltaEn
        else:
            self.deltaEn = 1
        self.eta = eta
        self.consEn = consEn
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        super(ECD_abs, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        V = (loss + 0.5*self.weight_decay*self.q2 - self.F0)**self.eta

        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            V0 = V
            
            # Initial energy and its exponential
            self.energy = torch.log(V0)+torch.log(torch.tensor(self.s+self.deltaEn))
            self.expenergy = torch.exp(self.energy)

            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            abs_pinit = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    abs_pinit += torch.sum(torch.abs(p))

            # Normalize the initial momenta such that |p(0)| = deltaEn
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(self.deltaEn/abs_pinit)   

            self.abs_p = torch.tensor(self.deltaEn)

        if V > self.eps2:

            # Scaling factor of the p for energy conservation
            if self.consEn == True:
                abs_ptrue = ((self.expenergy / V)-self.s)
                if torch.abs(abs_ptrue-self.abs_p) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif self.abs_p < 0:
                    self.normalization_coefficient = 1.0
                elif self.abs_p == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = abs_ptrue / self.abs_p

            # Update the p's and compute p^2 that is needed for the q update
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            self.abs_p = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        p.add_(- self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.F0))*(d_q+self.weight_decay*q))
                        self.p2.add_(torch.norm(p)**2)
                        self.abs_p += torch.sum(torch.abs(p))
            
            #Update the q's and add tiny rotation of the momenta
            abs_pnew = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * torch.sign(param_state["momenta"])/(self.s + self.abs_p))

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    abs_pnew += torch.sum(torch.abs(param_state["momenta"].view(-1)))
                    # p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            # self.abs_p = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = self.abs_p * p /abs_pnew
                    # self.abs_p += torch.sum(torch.sign(param_state['momenta']))

            # self.abs_p = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            # for group in self.param_groups:
            #     for q in group["params"]:  
            #         param_state = self.state[q]
            #         p = param_state["momenta"]
            #         self.abs_p += torch.sum(torch.abs(p))
            # self.p2 = pnorm**2 

            self.iteration += 1

        return loss

class ECD_q1_sign(Optimizer):
    '''
    
    Optimizer based on the new separable Hamiltonian and generalized bounces, q =1
    lr (float): learning rate, called Delta t in the paper (required)
    F0 (float): expected minimum of the objective
    deltaEn (float): initial energy
    eta (float): hyperparameter that controls the concentration of the measure (required).
                 It has to be >= 1. Increasing it concentrates the measure towards the bottom of the basin, and it is useful for pure optimization problems where the goal to find smallest loss. Tested up to eta = 5.
    consEn (bool): whether the energy is conserved or not
    weight_decay (float): weight decay, implemented as L^2 term
    nu (float): chaos hyperparameter
    s (float): regularization switch
    '''

    def __init__(self, params, lr=required, F0=0., eps1=1e-10, eps2=1e-40, nu=1e-5, weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, nu=nu, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.consEn = consEn
        self.eta = eta
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.Finit = 0.
        self.dim = 0
        super(ECD_q1_sign, self).__init__(params, defaults)
        # d_params = len(params)
        self.d_params = len(self.param_groups[0]["params"][0])
    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        
        
        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            self.Finit = loss
            
            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            # notice that now by momenta we mean the velocities
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    self.dim += q.numel()
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = 1
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(1/p2init))   

            self.p2 = torch.tensor(1.0)
            # self.iteration += 1

        if loss + 0.5*self.weight_decay*self.q2 - self.F0 > self.eps2:

            # Scaling factor of the p for energy conservation
            
            if self.consEn == True:
                p2true = 1
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)
            
            # Update the p's and compute p^2 that is needed for the q update
            
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        #bp()
                        #p.add_(- self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.F0))*(d_q+self.weight_decay*q))
                        prefactor = -0.5*self.lr *self.eta* self.dim/(self.dim-1)
                        dotp = torch.dot(p.view(-1), (d_q+self.weight_decay*q).view(-1))
                        denom = loss + 0.5*self.weight_decay*self.q2 - self.F0
                        p.add_( (prefactor/denom)*((d_q+self.weight_decay*q) - p*dotp) )
                    
                        self.p2.add_(torch.norm(p)**2)
            #print(self.p2)
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * torch.sign(param_state["momenta"]))

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)
            self.p2 = pnorm**2 
            
            self.iteration += 1
        
        return loss

class ECD_q1_F2_aniso(Optimizer):
    def __init__(self, params, lr=required, F0=-1, eps1=1e-10, eps2=1e-40, eps3  = 1e-8, nu=.1, weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, eps3  = eps3, nu=nu, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.consEn = consEn
        self.eta = eta
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.dim = 0
        self.eps3 = eps3

        super(ECD_q1_F2_aniso, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        
        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            self.Finit = loss
            
            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            # notice that now by momenta we mean the velocities
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    self.dim += q.numel()
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = 1
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(1/p2init))   

            self.p2 = torch.tensor(1.0)
            # self.iteration += 1


            ## Now rescale the hypers
            self.nu = self.nu/np.sqrt(self.dim)

        if loss + 0.5*self.weight_decay*self.q2 - self.F0 > self.eps2:

            # Scaling factor of the p for energy conservation
            
            if self.consEn == True:
                p2true = 1
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)
            
            # Update the p's and compute p^2 that is needed for the q update
            
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)


            self.total_grad = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    d_q = q.grad.data
                    self.total_grad += torch.norm(d_q)**2

            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        lr_prefac = 2*torch.sqrt(loss/(d_q*d_q+self.eps3)/self.dim/self.eta)
                        prefactor = -0.5*lr_prefac*self.lr * self.eta* self.dim/(self.dim-1)
                        dotp = torch.dot(p.view(-1), (d_q+self.weight_decay*q).view(-1))

                        denom = loss + 0.5*self.weight_decay*self.q2 - self.F0
                        p.add_( (prefactor/denom)*((d_q+self.weight_decay*q) - p*dotp) )
                    
                        self.p2.add_(torch.norm(p)**2)

            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    d_q = q.grad.data
                    lr_prefac = 2*torch.sqrt(loss/(d_q*d_q+self.eps3)/self.dim/self.eta)
                    q.data.add_(lr_prefac*self.lr * param_state["momenta"] )

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)

            self.p2 = pnorm**2 
            

            self.iteration += 1
        
        return loss

class ECD_q1_F2_thetastep(Optimizer):
    def __init__(self, params, lr=required, F0=-1, eps1=1e-10, eps2=1e-40, eps3  = 1e-8,  nu=.1, weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, eps3 = eps3, nu=nu, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.consEn = consEn
        self.eta = eta
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.dim = 0
        self.eps3 = eps3

        super(ECD_q1_F2_thetastep, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        
        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            self.Finit = loss
            
            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            # notice that now by momenta we mean the velocities
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    self.dim += q.numel()
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = 1
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(1/p2init))   

            self.p2 = torch.tensor(1.0)


            ## Now rescale the hypers
            self.nu = self.nu/np.sqrt(self.dim)

        if loss + 0.5*self.weight_decay*self.q2 - self.F0 > self.eps2:

            # Scaling factor of the p for energy conservation
            
            if self.consEn == True:
                p2true = 1
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)
            
            # Update the p's and compute p^2 that is needed for the q update
            
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)


            self.total_grad = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    d_q = q.grad.data
                    self.total_grad += torch.norm(d_q)**2
            lr_prefac = 2*loss*torch.sqrt(1/(self.total_grad+self.eps3))
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)

                        prefactor = -0.5*lr_prefac*self.lr * self.eta* self.dim/(self.dim-1)

                        dotp = torch.dot(p.view(-1), (d_q+self.weight_decay*q).view(-1))

                        denom = loss + 0.5*self.weight_decay*self.q2 - self.F0
                        p.add_( (prefactor/denom)*((d_q+self.weight_decay*q) - p*dotp) )
                    
                        self.p2.add_(torch.norm(p)**2)
            #print(self.p2)
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    d_q = q.grad.data
                    q.data.add_(lr_prefac*self.lr * param_state["momenta"] )

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)

            self.p2 = pnorm**2 
            

            self.iteration += 1
        
        return loss
   
class ECD_q1_adaptive_F2(Optimizer):
    def __init__(self, params, lr=required, F0=-1, eps1=1e-10, eps2=1e-40, eps3  = 1e-8, nu=.1, weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, eps3 = eps3, nu=nu, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.consEn = consEn
        self.eta = eta
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.eps3 = eps3
        self.weight_decay = weight_decay
        self.nu = nu
        self.Finit = 0.
        self.dim = 0

        super(ECD_q1_adaptive_F2, self).__init__(params, defaults)
        # d_params = len(params)
        #self.d_params = len(self.param_groups[0]["params"][0])
    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        
        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            self.Finit = loss
            
            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            # notice that now by momenta we mean the velocities
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    self.dim += q.numel()
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = 1
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(1/p2init))   

            self.p2 = torch.tensor(1.0)
            # self.iteration += 1


            ## Now rescale the hypers
            self.lr = self.lr/np.sqrt(self.eta)
            self.nu = self.nu/np.sqrt(self.dim)

        if loss + 0.5*self.weight_decay*self.q2 - self.F0 > self.eps2:

            # Scaling factor of the p for energy conservation
            
            if self.consEn == True:
                p2true = 1
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)
            
            # Update the p's and compute p^2 that is needed for the q update
            
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)


            self.total_grad = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    d_q = q.grad.data
                    self.total_grad += torch.norm(d_q)**2

            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
   
                        lr_prefac = 2*torch.sqrt(loss/(self.total_grad+self.eps3))
                        prefactor = -0.5*lr_prefac*self.lr * self.eta* self.dim/(self.dim-1)
                        dotp = torch.dot(p.view(-1), (d_q+self.weight_decay*q).view(-1))

                        denom = loss + 0.5*self.weight_decay*self.q2 - self.F0
                        p.add_( (prefactor/denom)*((d_q+self.weight_decay*q) - p*dotp) )
                    
                        self.p2.add_(torch.norm(p)**2)
            #print(self.p2)
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    lr_prefac = 2*torch.sqrt(loss/self.total_grad)

                    q.data.add_(lr_prefac*self.lr * param_state["momenta"] )

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)

            self.p2 = pnorm**2 
            

            self.iteration += 1
        
        return loss
     
#### NOT COMPLETE YET ####
class ECDSep_ordered_F2_adaptive(Optimizer):
    '''
    Optimizer based on the new separable Hamiltonian and generalized bounces
    lr (float): learning rate, called Delta t in the paper (required)
    F0 (float): expected minimum of the objective
    deltaEn (float): initial energy
    eta (float): hyperparameter that controls the concentration of the measure (required).
                 It has to be >= 1. Increasing it concentrates the measure towards the bottom of the basin, and it is useful for pure optimization problems where the goal to find smallest loss. Tested up to eta = 5.
    consEn (bool): whether the energy is conserved or not
    weight_decay (float): weight decay, implemented as L^2 term
    nu (float): chaos hyperparameter
    s (float): regularization switch
    '''

    def __init__(self, params, lr=required, F0=0., eps1=1e-10, eps2=1e-40, deltaEn=1., nu=1e-5, s=0., weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, deltaEn=deltaEn, nu=nu, s=s, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.s = s
        if self.s == 1:
            self.deltaEn = deltaEn
        else:
            self.deltaEn = 1.
        self.eta = eta
        self.consEn = consEn
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.Finit = 0.
        super(ECDSep_ordered_F2_adaptive, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))

        self.total_grad = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
        for group in self.param_groups:
            for q in group["params"]:
                d_q = q.grad.data     
                self.total_grad += torch.norm(d_q)**2
                
        V = (loss + 0.5*self.weight_decay*self.q2 - self.F0)**self.eta

        # Scale the lr with our approximation of F_2: \Delta t = \frac{1}{\sqrt{F_2\eta}} with F_2 = (F-F_min)/(\nabla F)^2
        self.lr_scaled = 2*self.lr*torch.sqrt(loss/self.total_grad)
        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial energy and its exponential
            V0 = V

            # Initial energy and its exponential
            self.energy = torch.log(V0)+torch.log(torch.tensor(self.s+self.deltaEn))
            self.expenergy = torch.exp(self.energy)

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 

                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                        # print(torch.norm(p))
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = deltaEn
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(self.deltaEn/p2init))   

            self.p2 = torch.tensor(self.deltaEn)


        if V > self.eps2:
            # Scaling factor of the p for energy conservation
            if self.consEn == True:
                p2true = ((self.expenergy / V))

                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)   
            
            # Update the p's and compute p^2 that is needed for the q update
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        #Add noise to the momenta
                        z = torch.randn(p.size(), device=p.device, generator = self.generator)
                        pnorm = torch.norm(p)
                        param_state["momenta"] = p+ self.nu*z*pnorm
                        p.add_(- self.lr_scaled * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.F0))*(d_q+self.weight_decay*q))
                        self.p2.add_(torch.norm(p)**2)
            
            
         
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]

                    # Normalize momentum
                    p.mul_(torch.sqrt(p2true / self.p2))
                    q.data.add_(self.lr_scaled * 2 * p/(self.expenergy/V))
                    self.p2 = torch.norm(p)**2

            self.iteration += 1
        
        return loss

class ECDSep_first_order(Optimizer):
    def __init__(self, params, lr=required, F0=0., eps1=1e-10, eps2=1e-40, eps3  = 1e-8, weight_decay=0., eta=1):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, eps3 = eps3, weight_decay=weight_decay, eta=eta)
        self.lr = lr
        self.F0 = F0
        self.eps1 = eps1
        self.eps2 = eps2
        self.eps3 = eps3
        self.weight_decay = weight_decay
        self.eta = eta
        self.iteration = 0

        super(ECDSep_first_order, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
        V = (loss + 0.5*self.weight_decay*self.q2 - self.F0)**self.eta
        if self.iteration == 0:        

            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())

        if V > self.eps2:
            for group in self.param_groups:
                for q in group["params"]:
                    # param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        q.data.add_(-2*self.lr*d_q*(loss + 0.5*self.weight_decay*self.q2 - self.F0)/(self.eta*(torch.norm(d_q)**2+self.eps3)))
            self.iteration += 1

        return loss

class ECD_gamma(Optimizer):
    def __init__(self, params, lr=required, F0=0, eps1=1e-10, eps2=1e-40, nu=.1, gamma = 1., weight_decay=0, eta=required, consEn=True, scheduler = False):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, nu=nu, weight_decay=weight_decay, eta=eta, consEn=consEn, gamma = gamma, scheduler = scheduler)
        self.F0 = F0
        self.consEn = consEn
        self.eta = eta
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.gamma = gamma
        self.Finit = 0.
        self.Fparenthesis_init = 0.0
        self.dim = 0
        self.prefactor = None
        self.scheduler = scheduler
        self.switched = False
        super(ECD_gamma, self).__init__(params, defaults)
        # d_params = len(params)
        #self.d_params = len(self.param_groups[0]["params"][0])
    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        
        
        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            self.Finit = loss
            
            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            # notice that now by momenta we mean the v which always has norm 1 (normalized momenta)
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    self.dim += q.numel()
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = 1
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(1/p2init))   

            self.p2 = torch.tensor(1.0)
            # self.iteration += 1

        
            ## Now rescale the hypers. lr in terms of eta only  if the scheduler is false
            if self.scheduler == False:
                self.lr = self.lr/np.sqrt(self.eta)
            self.nu = self.nu/np.sqrt(self.dim)

        
            ## define the prefactor in the update rule for vdot
            self.prefactor = - (self.dim*self.eta-2*self.gamma)/(2*(self.dim-1))

        
        #loss = loss/self.Finit ##normalize the loss
        if loss + 0.5*self.weight_decay*self.q2 - self.F0 > self.eps2:

            #Rescale the learning rate if the loss is 0.2 of the initial one
            if self.scheduler == True:
                if self.switched == False:
                    if (loss < 0.2* self.Finit):
                        self.lr/np.sqrt(self.eta)
                        self.switched = True
                    

            # Scaling factor of the p for energy conservation
            
            if self.consEn == True:
                p2true = 1
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)
            
            # Update the p's and compute p^2 that is needed for the q update
            
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        
                        dotp = torch.dot(p.view(-1), (d_q+self.weight_decay*q).view(-1))
                        F_factor = (loss + 0.5*self.weight_decay*self.q2 - self.F0)**(-1.0-self.gamma)

                        
                        p.add_( (self.lr*self.prefactor*F_factor)*((d_q+self.weight_decay*q) - p*dotp) )
                    
                        self.p2.add_(torch.norm(p)**2)
            #print(self.p2)
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    F_factor = (loss + 0.5*self.weight_decay*self.q2 - self.F0)**(self.gamma)

                    #print(self.lr*F_factor)
                    q.data.add_(F_factor*self.lr * param_state["momenta"])

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)
            self.p2 = pnorm**2 
            
            self.iteration += 1
        
        return loss

class ECDSep(Optimizer):
    '''
    Optimizer based on the new separable Hamiltonian and generalized bounces
    lr (float): learning rate, called Delta t in the paper (required)
    F0 (float): expected minimum of the objective
    deltaEn (float): initial energy
    eta (float): hyperparameter that controls the concentration of the measure (required).
                 It has to be >= 1. Increasing it concentrates the measure towards the bottom of the basin, and it is useful for pure optimization problems where the goal to find smallest loss. Tested up to eta = 5.
    consEn (bool): whether the energy is conserved or not
    weight_decay (float): weight decay, implemented as L^2 term
    nu (float): chaos hyperparameter
    s (float): regularization switch
    '''

    def __init__(self, params, lr=required, F0=0., eps1=1e-10, eps2=1e-40, deltaEn=0., nu=1e-5, s=1., weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, deltaEn=deltaEn, nu=nu, s=s, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.s = s
        if self.s == 1:
            self.deltaEn = deltaEn
        else:
            self.deltaEn = 1
        self.eta = eta
        self.consEn = consEn
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        super(ECDSep, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        V = (loss + 0.5*self.weight_decay*self.q2 - self.F0)**self.eta

        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            V0 = V
            
            # Initial energy and its exponential
            self.energy = torch.log(V0)+torch.log(torch.tensor(self.s+self.deltaEn))
            self.expenergy = torch.exp(self.energy)

            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = deltaEn
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(self.deltaEn/p2init))   

            self.p2 = torch.tensor(self.deltaEn)

        if V > self.eps2:

            # Scaling factor of the p for energy conservation
            if self.consEn == True:
                p2true = ((self.expenergy / V)-self.s)
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)

            # Update the p's and compute p^2 that is needed for the q update
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        p.add_(- self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.F0))*(d_q+self.weight_decay*q))
                        self.p2.add_(torch.norm(p)**2)
            
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * 2 * param_state["momenta"]/(self.s+self.p2))

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)
                   
            self.p2 = pnorm**2 

            self.iteration += 1

        return loss

class ECDSep_checkEnergy(Optimizer):

    '''
    Returns energy violation due to stochasticity

    Optimizer based on the new separable Hamiltonian and generalized bounces
    lr (float): learning rate, called Delta t in the paper (required)
    F0 (float): expected minimum of the objective
    deltaEn (float): initial energy
    eta (float): hyperparameter that controls the concentration of the measure (required).
                 It has to be >= 1. Increasing it concentrates the measure towards the bottom of the basin, and it is useful for pure optimization problems where the goal to find smallest loss. Tested up to eta = 5.
    consEn (bool): whether the energy is conserved or not
    weight_decay (float): weight decay, implemented as L^2 term
    nu (float): chaos hyperparameter
    s (float): regularization switch
    '''

    def __init__(self, params, lr=required, F0=0., eps1=1e-10, eps2=1e-40, deltaEn=0., nu=1e-5, s=1., weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, deltaEn=deltaEn, nu=nu, s=s, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.s = s
        if self.s == 1:
            self.deltaEn = deltaEn
        else:
            self.deltaEn = 1
        self.eta = eta
        self.consEn = consEn
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.energy_violation = 0 ## Used to track the energy violation due to batches. In particular it computes |p|/|p_true|
        super(ECDSep_checkEnergy, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        V = (loss + 0.5*self.weight_decay*self.q2 - self.F0)**self.eta

        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            V0 = V
            
            # Initial energy and its exponential
            self.energy = torch.log(V0)+torch.log(torch.tensor(self.s+self.deltaEn))
            self.expenergy = torch.exp(self.energy)

            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = deltaEn
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(self.deltaEn/p2init))   

            self.p2 = torch.tensor(self.deltaEn)

        if V > self.eps2:

            # Scaling factor of the p for energy conservation
            p2true = ((self.expenergy / V)-self.s)
            if self.consEn == True:
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)

            ##Compute the energy violation
            if self.p2 == 0.0:
                self.energy_violation = 0
            else:
                self.energy_violation = p2true / self.p2-1
            # Update the p's and compute p^2 that is needed for the q update
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        p.add_(- self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.F0))*(d_q+self.weight_decay*q))
                        self.p2.add_(torch.norm(p)**2)
            
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * 2 * param_state["momenta"]/(self.s+self.p2))

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)
                   
            self.p2 = pnorm**2 

            self.iteration += 1

        return (loss, self.energy_violation)

class ECD_q1(Optimizer):
    '''
    
    Optimizer based on the new separable Hamiltonian and generalized bounces, q =1
    lr (float): learning rate, called Delta t in the paper (required)
    F0 (float): expected minimum of the objective
    deltaEn (float): initial energy
    eta (float): hyperparameter that controls the concentration of the measure (required).
                 It has to be >= 1. Increasing it concentrates the measure towards the bottom of the basin, and it is useful for pure optimization problems where the goal to find smallest loss. Tested up to eta = 5.
    consEn (bool): whether the energy is conserved or not
    weight_decay (float): weight decay, implemented as L^2 term
    nu (float): chaos hyperparameter
    s (float): regularization switch
    '''

    def __init__(self, params, lr=required, F0=0., eps1=1e-10, eps2=1e-40, nu=1e-5, weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, nu=nu, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.consEn = consEn
        self.eta = eta
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.Finit = 0.
        self.dim = 0
        super(ECD_q1, self).__init__(params, defaults)
        # d_params = len(params)
        self.d_params = len(self.param_groups[0]["params"][0])
    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        
        
        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            self.Finit = loss
            
            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            # notice that now by momenta we mean the velocities
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    self.dim += q.numel()
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = 1
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(1/p2init))   

            self.p2 = torch.tensor(1.0)
            # self.iteration += 1

        if loss + 0.5*self.weight_decay*self.q2 - self.F0 > self.eps2:

            # Scaling factor of the p for energy conservation
            
            if self.consEn == True:
                p2true = 1
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)
            
            # Update the p's and compute p^2 that is needed for the q update
            
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        #bp()
                        #p.add_(- self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.F0))*(d_q+self.weight_decay*q))
                        prefactor = -0.5*self.lr *self.eta* self.dim/(self.dim-1)
                        dotp = torch.dot(p.view(-1), (d_q+self.weight_decay*q).view(-1))
                        denom = loss + 0.5*self.weight_decay*self.q2 - self.F0
                        p.add_( (prefactor/denom)*((d_q+self.weight_decay*q) - p*dotp) )
                    
                        self.p2.add_(torch.norm(p)**2)
            #print(self.p2)
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * param_state["momenta"])

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)
            self.p2 = pnorm**2 
            
            self.iteration += 1
        
        return loss
class ECD_q1_scaled(Optimizer):
    def __init__(self, params, lr=required, F0=-1, eps1=1e-10, eps2=1e-40, nu=.1, weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, nu=nu, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.consEn = consEn
        self.eta = eta
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.Finit = 0.
        self.dim = 0
        super(ECD_q1_scaled, self).__init__(params, defaults)
        # d_params = len(params)
        #self.d_params = len(self.param_groups[0]["params"][0])
    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        
        
        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            self.Finit = loss
            
            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            # notice that now by momenta we mean the velocities
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    self.dim += q.numel()
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = 1
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(1/p2init))   

            self.p2 = torch.tensor(1.0)

            ## Now rescale the hypers
            self.lr = self.lr/np.sqrt(self.eta)
            self.nu = self.nu/np.sqrt(self.dim)

        if loss + 0.5*self.weight_decay*self.q2 - self.F0 > self.eps2:

            # Scaling factor of the p for energy conservation
            
            if self.consEn == True:
                p2true = 1
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)
            
            # Update the p's and compute p^2 that is needed for the q update
            
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        #bp()
                        #p.add_(- self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.F0))*(d_q+self.weight_decay*q))
                        prefactor = -0.5*self.lr *self.eta* self.dim/(self.dim-1)
                        dotp = torch.dot(p.view(-1), (d_q+self.weight_decay*q).view(-1))
                        denom = loss + 0.5*self.weight_decay*self.q2 - self.F0
                        p.add_( (prefactor/denom)*((d_q+self.weight_decay*q) - p*dotp) )
                    
                        self.p2.add_(torch.norm(p)**2)
            #print(self.p2)
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * param_state["momenta"])

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)
            self.p2 = pnorm**2 
            
            self.iteration += 1
        
        return loss

class ECD_q1_scaled_V2(Optimizer):
    def __init__(self, params, lr=required, F0=-1, eps1=1e-10, eps2=1e-40, eps3=1e-8, nu=.1, weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, eps3=eps3, nu=nu, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.consEn = consEn
        self.eta = eta
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.eps3 = eps3
        self.weight_decay = weight_decay
        self.nu = nu
        self.Finit = 0.
        self.dim = 0
        super(ECD_q1_scaled_V2, self).__init__(params, defaults)
        # d_params = len(params)
        #self.d_params = len(self.param_groups[0]["params"][0])
    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        
        
        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            self.Finit = loss
            
            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            # notice that now by momenta we mean the velocities
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    self.dim += q.numel()
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  

            # Normalize the initial momenta such that |p(0)| = 1
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(1/p2init))   

            self.p2 = torch.tensor(1.0)
            self.total_grad = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    d_q = q.grad.data
                    self.total_grad += torch.norm(d_q)**2
            lr_prefac = 1/np.sqrt(self.total_grad+self.eps3)
            ## Now rescale the hypers
            self.lr = lr_prefac*self.lr/np.sqrt(self.eta)
            self.nu = self.nu/np.sqrt(self.dim)

        if loss + 0.5*self.weight_decay*self.q2 - self.F0 > self.eps2:

            # Scaling factor of the p for energy conservation
            
            if self.consEn == True:
                p2true = 1
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)
            
            # Update the p's and compute p^2 that is needed for the q update
            
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        #bp()
                        #p.add_(- self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.F0))*(d_q+self.weight_decay*q))
                        prefactor = -0.5*self.lr *self.eta* self.dim/(self.dim-1)
                        dotp = torch.dot(p.view(-1), (d_q+self.weight_decay*q).view(-1))
                        denom = loss + 0.5*self.weight_decay*self.q2 - self.F0
                        p.add_( (prefactor/denom)*((d_q+self.weight_decay*q) - p*dotp) )
                    
                        self.p2.add_(torch.norm(p)**2)
            #print(self.p2)
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * param_state["momenta"])

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)
            self.p2 = pnorm**2 
            
            self.iteration += 1
        
        return loss

class BBI(Optimizer): 
    """Optimizer based on the BBI model of inflation.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        v0 (float): expected minimum of the potential (\Delta V in the paper)
        threshold0 (integer): threshold for fixed bounces (T_0 in the paper)
        threshold1 (integer): threshold for progress-dependent bounces (T_1 in the paper)
        deltaEn (float): extra initial energy (\delta E in the paper)
        consEn (bool): if True enforces energy conservation at every step
        n_fixed_bounces (integer): number of bounces every T_0 iterations (N_b in the paper) 
    """
    def __init__(self, params, lr=required, eps1=1e-10, eps2 = 1e-40, v0=0, threshold0 = 1000, threshold = 3000, deltaEn = 0.0, consEn = True, n_fixed_bounces = 1):
            
            defaults = dict(lr=lr, eps1=eps1, eps2=eps2, v0=v0, threshold = threshold, threshold0 = threshold0, deltaEn = deltaEn, consEn = consEn, n_fixed_bounces = n_fixed_bounces)
            self.energy = None
            self.min_loss = None
            self.iteration = 0
            self.deltaEn = deltaEn
            self.n_fixed_bounces = n_fixed_bounces
            self.consEn = consEn 
            super(BBI, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BBI, self).__setstate__(state)

    def step(self, closure):

        loss = closure().item()

        # initialization
        if(self.iteration == 0):

            #define a random numbers generator, in order not to use the ambient seed and have random bounces even with the same ambient seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed()+1)
         
            #Initial energy
            self.initV = loss-self.param_groups[0]["v0"]
            self.init_energy = self.initV+self.deltaEn

            # Some counters            
            self.counter0 = 0
            self.fixed_bounces_performed = 0
            self.counter = 0

            self.min_loss = float("inf")
            
        for group in self.param_groups:
            
            V = (loss - group["v0"])
            dt = group["lr"]
            eps1 = group["eps1"]
            eps2 = group["eps2"]
            threshold0 = group["threshold0"]
            threshold = group["threshold"]
            
            if V > eps2:

                EoverV = self.init_energy/V
                VoverE = V/self.init_energy

                # Now I check if loss and pi^2 are consistent with the initial value of the energy
                
                ps2_pre = torch.tensor(0.0, device = self.param_groups[0]["params"][0].device )

                for p in group["params"]:
                    param_state = self.state[p]
                    d_p = p.grad.data
                    #Initialize in the direction of the gradient, with magnitude related to deltaE
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = -(d_p/torch.norm(d_p))*torch.sqrt(torch.tensor( ((self.init_energy**2)/self.initV) - self.initV ))
                    else:
                        buf = param_state["momentum_buffer"]

                    # compute the current pi^2 . Pre means that this is the value before the iteration step
                    ps2_pre += torch.dot(buf.view(-1), buf.view(-1))

                    
                if (self.consEn == True):

                    # Compare this \pi^2 with what it should have been if the energy was correct
                    ps2_correct = V*( (EoverV**2)-1.0 )

                    # Compute the rescaling factor, only if real
                    if torch.abs(ps2_pre-ps2_correct) <  eps1:
                        self.rescaling_pi = 1.0
                    elif ps2_correct < 0.0:
                        self.rescaling_pi = 1.0
                    else:
                        self.rescaling_pi = torch.sqrt(((ps2_correct/(ps2_pre))))

                
                # Perform the optimization step
                if (self.counter != threshold) and (self.counter0 != threshold0) :
                    
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        d_p = p.grad.data
                        param_state = self.state[p]

                        if "momentum_buffer" not in param_state:    
                            buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        else:
                            buf = param_state["momentum_buffer"]

                        # Here the rescaling of momenta to enforce conservation of energy        
                        if (self.consEn == True):
                            buf.mul_(self.rescaling_pi) 
                            
                        buf.add_(- 0.5 * dt * (VoverE + EoverV)*d_p)
                        p.data.add_(dt*VoverE*buf)

                    # Updates counters
                    self.counter0+=1
                    self.counter+=1
                    self.iteration+=1

                    # Checks progress
                    if V < self.min_loss:
                            self.min_loss = V
                            self.counter = 0
                # Bounces
                else:
                        
                    #First we iterate once to compute pi^2, we randomly regenerate the directions, and we compute the new norm squared

                    ps20 = torch.tensor(0.0, device = self.param_groups[0]["params"][0].device )
                    ps2new = torch.tensor(0.0, device = self.param_groups[0]["params"][0].device )

                    for p in group["params"]:
                        param_state = self.state[p]

                        buf = param_state["momentum_buffer"]
                        ps20 += torch.dot(buf.view(-1), buf.view(-1))
                        new_buf = param_state["momentum_buffer"] = torch.rand(buf.size(), device=buf.device, generator = self.generator)-.5
                        ps2new += torch.dot(new_buf.view(-1), new_buf.view(-1))

                    # Then rescale them
                    for p in group["params"]:
                        param_state = self.state[p]
                        buf = param_state["momentum_buffer"]
                        buf.mul_(torch.sqrt(ps20/ps2new))

                    # Update counters
                    if (self.counter0 == threshold0):
                        self.fixed_bounces_performed+=1
                        if self.fixed_bounces_performed < self.n_fixed_bounces:
                            self.counter0 = 0
                        else:
                            self.counter0+=1           
                    self.counter = 0   
        return loss
        

class ECDSep_scaled(Optimizer):
    '''
    Optimizer based on the new separable Hamiltonian and generalized bounces
    lr (float): learning rate, called Delta t in the paper (required)
    F0 (float): expected minimum of the objective
    deltaEn (float): initial energy
    eta (float): hyperparameter that controls the concentration of the measure (required).
                 It has to be >= 1. Increasing it concentrates the measure towards the bottom of the basin, and it is useful for pure optimization problems where the goal to find smallest loss. Tested up to eta = 5.
    consEn (bool): whether the energy is conserved or not
    weight_decay (float): weight decay, implemented as L^2 term
    nu (float): chaos hyperparameter
    s (float): regularization switch
    '''

    def __init__(self, params, lr=required, F0=0., eps1=1e-10, eps2=1e-40, deltaEn=0., nu=1e-5, s=1., weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, deltaEn=deltaEn, nu=nu, s=s, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.s = s
        if self.s == 1:
            self.deltaEn = deltaEn
        else:
            self.deltaEn = 1
        self.eta = eta
        self.consEn = consEn
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.dim = 0
        super(ECDSep_scaled, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        V = (loss + 0.5*self.weight_decay*self.q2 - self.F0)**self.eta

        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            V0 = V
            
            # Initial energy and its exponential
            self.energy = torch.log(V0)+torch.log(torch.tensor(self.s+self.deltaEn))
            self.expenergy = torch.exp(self.energy)

            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    self.dim += q.numel()
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  
            # Normalize the initial momenta such that |p(0)| = deltaEn
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(self.deltaEn/p2init))   

            self.p2 = torch.tensor(self.deltaEn)

            ## Now rescale the hypers
            self.lr = self.lr/np.sqrt(self.eta)
            self.nu = self.nu/np.sqrt(self.dim)


        if V > self.eps2:

            # Scaling factor of the p for energy conservation
            if self.consEn == True:
                p2true = ((self.expenergy / V)-self.s)
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)

            # Update the p's and compute p^2 that is needed for the q update
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        p.add_(- self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.F0))*(d_q+self.weight_decay*q))
                        self.p2.add_(torch.norm(p)**2)
            
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * 2 * param_state["momenta"]/(self.s+self.p2))

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)
                   
            self.p2 = pnorm**2 

            self.iteration += 1

        return loss




class BBI_v0tuning(Optimizer): 
    """Optimizer based on the BBI model of inflation.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        v0 (float): expected minimum of the potential (\Delta V in the paper)
        threshold0 (integer): threshold for fixed bounces (T_0 in the paper)
        threshold1 (integer): threshold for progress-dependent bounces (T_1 in the paper)
        deltaEn (float): extra initial energy (\delta E in the paper)
        consEn (bool): if True enforces energy conservation at every step
        n_fixed_bounces (integer): number of bounces every T_0 iterations (N_b in the paper)
        v0_tuning (bool):  automatic tuning of v0.
        weight_decay(float): weight decay
    """
    def __init__(self, params, lr=required, eps1=1e-10, eps2=1e-40, v0=0, threshold0=1000, threshold=3000, deltaEn=0.0, consEn=True, n_fixed_bounces=1, weight_decay=0.0, v0_tuning = False ):
            
            defaults = dict(lr=lr, eps1=eps1, eps2=eps2, v0=v0, threshold = threshold, threshold0 = threshold0, deltaEn = deltaEn, consEn = consEn, n_fixed_bounces = n_fixed_bounces, v0_tuning = v0_tuning)
            self.energy = None
            self.min_loss = None
            self.iteration = 0
            self.deltaEn = deltaEn
            self.n_fixed_bounces = n_fixed_bounces
            self.consEn = consEn 
            self.weight_decay = weight_decay
            self.v0_tuning = v0_tuning
            super(BBI_v0tuning, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BBI_v0tuning, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        # initialization
        if(self.iteration == 0):

            #define a random numbers generator, in order not to use the ambient seed and have random bounces even with the same ambient seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())#+1)
         
            #Initial energy
            self.initV = loss-self.param_groups[0]["v0"]
            self.init_energy = self.initV+self.deltaEn

            # Some counters            
            self.counter0 = 0
            self.fixed_bounces_performed = 0
            self.counter = 0

            self.min_loss = float("inf")

            #used for the shift of v0
            self.v0_shift = 0.0

            if self.v0_tuning == True:
                print("Warning, self tuning of v0 is still in development!")

        for group in self.param_groups:
            
            th2 = torch.tensor(0.0, device = self.param_groups[0]["params"][0].device )

            if self.weight_decay != 0.0:
                #compute \theta^2 for the L2 regularization associated to weight decay
                for p in group["params"]:
                        param_state = self.state[p]
                        th2+= torch.dot(p.data.view(-1), p.data.view(-1))

            V = (loss - group["v0"]-self.v0_shift)+.5*self.weight_decay*th2
            dt = group["lr"]
            eps1 = group["eps1"]
            eps2 = group["eps2"]
            threshold0 = group["threshold0"]
            threshold = group["threshold"]
            
            if V > (not self.v0_tuning)*eps2:

                EoverV = self.init_energy/V
                VoverE = V/self.init_energy

                # Now I check if loss and pi^2 are consistent with the initial value of the energy
                
                ps2_pre = torch.tensor(0.0, device = self.param_groups[0]["params"][0].device )

                for p in group["params"]:
                    param_state = self.state[p]
                    d_p = p.grad.data
                    d_p.add_(self.weight_decay, p.data)
                    #Initialize in the direction of the gradient, with magnitude related to deltaE
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = -(d_p/torch.norm(d_p))*torch.sqrt(torch.tensor( ((self.init_energy**2)/self.initV) - self.initV ))
                    else:
                        buf = param_state["momentum_buffer"]

                    # compute the current pi^2 . Pre means that this is the value before the iteration step
                    ps2_pre += torch.dot(buf.view(-1), buf.view(-1))

                    
                if (self.consEn == True):

                    # Compare this \pi^2 with what it should have been if the energy was correct
                    ps2_correct = V*( (EoverV**2)-1.0 )

                    # Compute the rescaling factor, only if real
                    if torch.abs(ps2_pre-ps2_correct) <  eps1:
                        self.rescaling_pi = 1.0
                    elif ps2_correct < 0.0:
                        self.rescaling_pi = 1.0
                    else:
                        self.rescaling_pi = torch.sqrt(((ps2_correct/(ps2_pre))))

                
                # Perform the optimization step
                if (self.counter != threshold) and (self.counter0 != threshold0) :
                    
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        d_p = p.grad.data
                        d_p.add_(self.weight_decay, p.data)
                        param_state = self.state[p]

                        if "momentum_buffer" not in param_state:    
                            buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        else:
                            buf = param_state["momentum_buffer"]

                        # Here the rescaling of momenta to enforce conservation of energy        
                        if (self.consEn == True):
                            buf.mul_(self.rescaling_pi) 
                            
                        buf.add_(- 0.5 * dt * (VoverE + EoverV)*d_p)
                        p.data.add_(dt*VoverE*buf)

                    # Updates counters
                    self.counter0+=1
                    self.counter+=1
                    self.iteration+=1

                    # Checks progress
                    if V < self.min_loss:
                            self.min_loss = V
                            self.counter = 0
                # Bounces
                else:
                        
                    #First we iterate once to compute pi^2, we randomly regenerate the directions, and we compute the new norm squared

                    ps20 = torch.tensor(0.0, device = self.param_groups[0]["params"][0].device )
                    ps2new = torch.tensor(0.0, device = self.param_groups[0]["params"][0].device )

                    for p in group["params"]:
                        param_state = self.state[p]

                        buf = param_state["momentum_buffer"]
                        ps20 += torch.dot(buf.view(-1), buf.view(-1))
                        new_buf = param_state["momentum_buffer"] = torch.rand(buf.size(), device=buf.device, generator = self.generator)-.5
                        ps2new += torch.dot(new_buf.view(-1), new_buf.view(-1))

                    # Then rescale them
                    for p in group["params"]:
                        param_state = self.state[p]
                        buf = param_state["momentum_buffer"]
                        buf.mul_(torch.sqrt(ps20/ps2new))

                    # Update counters
                    if (self.counter0 == threshold0):
                        self.fixed_bounces_performed+=1
                        if self.fixed_bounces_performed < self.n_fixed_bounces:
                            self.counter0 = 0
                        else:
                            self.counter0+=1           
                    self.counter = 0   
        
            elif self.v0_tuning == True: 
                # Here a linear shift, with an arbitrary coefficient. This is still preliminary and requires more tuning/experiments.
                # Another option is exponential backoff with some cutoff.
                self.v0_shift = self.v0_shift+5*V
                print("Shifting v0, remember this is still in development!")
                print("New v0: ", (self.v0_shift+group["v0"]).item() )

        return loss

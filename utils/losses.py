import numpy as np
import torch

eps = 1e-7
# Loss functions
def bce(y_true, y_pred):
    # Clipping to (ɛ, 1 - ɛ) is fine since the final activation is sigmoid, so y_pred is in [0, 1]
    y_pred = torch.clamp(y_pred, min = eps, max = 1.-eps) #torch.clip(y_pred, eps, 1. - eps)
    return -((y_true) * torch.log(y_pred + eps) + 
             (1. - y_true) * torch.log(1. - y_pred))


def square_bce(y_true, y_pred):
    return -((y_true) * torch.log(y_pred**2) + 
             (1. - y_true) * torch.log(1. - y_pred**2))

def exp_bce(y_true, y_pred):
    # Clipping to (ɛ, 1 - ɛ) is fine since the final activation is sigmoid, so y_pred is in [0, 1]
    return -((y_true) * (y_pred) + 
             (1. - y_true) * (1. - y_pred))

def probit(x):
    normal = torch.distributions.normal.Normal(loc=0., scale=1.) #tfp.distributions.Normal(loc=0., scale=1.)
    return normal.cdf(x)

def probit_bce(y_true, y_pred):
    y_pred = probit(y_pred)
    y_pred = torch.clamp(y_pred, eps, 1. - eps)
    
    return -((y_true) * torch.log(y_pred + eps) + 
             (1. - y_true) * torch.log(1. - y_pred))

def t_tanh(x):
    return 0.5 * (torch.tanh(x) + 1)

def tanh_bce(y_true, y_pred):
    y_pred = t_tanh(y_pred)
    y_pred = torch.clamp(y_pred, eps, 1. - eps)
    
    return -((y_true) * torch.log(y_pred + eps) + 
             (1. - y_true) * torch.log(1. - y_pred))

def t_arctan(x):
    return 0.5 + (torch.atan(x) / np.pi)

def arctan_bce(y_true, y_pred):
    y_pred = t_arctan(y_pred)
    y_pred = torch.clamp(y_pred, eps, 1. - eps)
    
    return -((y_true) * torch.log(y_pred + eps) + 
             (1. - y_true) * torch.log(1. - y_pred))

def mse(y_true, y_pred):
    # Clipping to (ɛ, 1 - ɛ) is fine since the final activation is sigmoid.
    y_pred = torch.clamp(y_pred, eps, 1. - eps)
    
    return -((y_true) * -torch.square(1. - y_pred) + 
             (1. - y_true) * -torch.square(y_pred))

def square_mse(y_true, y_pred):
    return -((y_true) * -torch.square(1. - y_pred) 
             + (1. - y_true) * -torch.square(y_pred + eps))

def exp_mse(y_true, y_pred):
    return -((y_true) * -torch.square(1. - torch.exp(y_pred)) + 
             (1. - y_true) * -torch.square(torch.exp(y_pred))) 

def probit_mse(y_true, y_pred):
    y_pred = probit(y_pred)
    y_pred = torch.clamp(y_pred, eps, 1. - eps)
    
    return -((y_true) * -torch.square(1. - y_pred) + 
             (1. - y_true) * -torch.square(y_pred))

def tanh_mse(y_true, y_pred):
    y_pred = t_tanh(y_pred)
    y_pred = torch.clamp(y_pred, eps, 1. - eps)
    
    return -((y_true) * -torch.square(1. - y_pred) + 
             (1. - y_true) * -torch.square(y_pred))

def arctan_mse(y_true, y_pred):
    y_pred = t_arctan(y_pred)
    y_pred = torch.clamp(y_pred, eps, 1. - eps)
    
    return -((y_true) * -torch.square(1. - y_pred) + 
             (1. - y_true) * -torch.square(y_pred))

def get_mse(p):
    def mse_p(y_true, y_pred):
        y_pred = torch.clamp(y_pred, eps, 1. - eps)
    
        return -((y_true) * -torch.pow(1. - y_pred, p) + 
                 (1. - y_true) * -torch.pow(y_pred, p))
    return mse_p
             
def mlc(y_true, y_pred):
    # res = -((y_true) * torch.log(y_pred + eps) + 
    #          (1. - y_true) * (1. - y_pred))
    # print('res = ', torch.log(y_pred + eps), y_pred + eps)

    return -((y_true) * torch.log(y_pred + eps) + 
             (1. - y_true) * (1. - y_pred))

def square_mlc(y_true, y_pred):
    return -((y_true) * torch.log( (y_pred + eps)**2 ) + 
             (1. - y_true) * (1. - y_pred**2))

def exp_mlc(y_true, y_pred):
    return -((y_true) * (y_pred) + 
             (1. - y_true) * (1. - torch.exp(y_pred)))

def sqr(y_true, y_pred):
    return -((y_true) * -1. / torch.sqrt(y_pred + eps) + 
             (1. - y_true) * -torch.sqrt(y_pred + eps))

def square_sqr(y_true, y_pred):
    return -((y_true) * -1. / torch.sqrt(torch.square(y_pred)) + 
             (1. - y_true) * -torch.sqrt(torch.square(y_pred)))

def exp_sqr(y_true, y_pred):
    return -((y_true) * -1. / torch.sqrt(torch.exp(y_pred)) + 
             (1. - y_true) * -torch.sqrt(torch.exp(y_pred)))

def get_sqr(p):
    def sqr_p(y_true, y_pred):
        return -((y_true) * -torch.pow(y_pred + eps, -p/2) + 
                 (1. - y_true) * -torch.pow(y_pred + eps, p/2))
    return sqr_p

def get_exp_sqr(p):
    def exp_sqr_p(y_true, y_pred):
        return -((y_true) * -torch.pow(torch.exp(y_pred) + eps, -p/2) + 
                 (1. - y_true) * -torch.pow(torch.exp(y_pred) + eps, p/2))
    return exp_sqr_p

def get_q_loss(q):
    def q_loss(y_true, y_pred):
        return -((y_true) * (torch.pow(torch.exp(y_pred), q) - 1)/q + 
                 (1. - y_true) * (1 - torch.pow(torch.exp(y_pred), q + 1)/(q + 1)))
    return q_loss
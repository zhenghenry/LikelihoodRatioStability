import numpy as np
from scipy import stats
import torch
import torch.nn as nn
# from lion_pytorch import Lion
from inflation import *
from torch.utils.data import random_split, DataLoader, TensorDataset, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Training functions
# earlystopping = EarlyStopping(patience=10,
#                               verbose=0,
#                               restore_best_weights=True)

#Adding our optimizer

from torch.optim.optimizer import Optimizer, required
import numpy as np
import torch


def initialize_linear(model, shrinkage, distribution = 'uniform'):
    for name, param in model.named_parameters():
        if 'weight' in name:
            fan_in = param.shape[1]
        if distribution == 'Gaussian':
            param.data.normal_(mean = 0, std = 1/(fan_in*shrinkage)**0.5)
        if distribution == 'uniform':
            param.data.uniform_(-1/(fan_in*shrinkage)**0.5,1/(fan_in*shrinkage)**0.5)

class create_model_original(pl.LightningModule):
    def __init__(self, d, loss_fun, output, optimizer, learning_rate, eta, F0, nu, weight_decay = 0., hidden_lr = None):
        super(create_model_original, self).__init__()
        self.d = d
        self.p = 0.05
        self.optimizer = optimizer
        self.linear1 = nn.Linear(self.d, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)
        self.dropout1 = nn.Dropout(self.p)
        self.dropout3 = nn.Dropout(self.p)
        self.dropout2 = nn.Dropout(self.p)
        self.learning_rate = learning_rate
        self.hidden_lr = hidden_lr
        self.F0 = F0
        self.eta = eta
        self.nu = nu
        self.weight_decay = weight_decay
        self.output = output
        self.loss_fun = loss_fun
        self.automatic_optimization = False
        self.train_hist = []
        self.val_hist = []
    
    def forward(self, x):
        p = 0.05
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = nn.ReLU()(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        if self.output == 'relu':
            x = nn.ReLU()(x)
            return x
        if self.output == 'linear':
            return x
        if self.output == 'sigmoid':
            x =  nn.Sigmoid()(x)
            return x

    
    def configure_optimizers(self):

        if self.hidden_lr:
            self.hidden_params = []
            self.outer_params = []
            for param_name, params in self.named_parameters():
                print(param_name)
                if 'linear2' in param_name or 'linear3' in param_name:
                    print('linear2' in param_name)
                    self.hidden_params.append(params)
                else:
                    self.outer_params.append(params)
            if self.optimizer == 'adam':
                optimizer = [torch.optim.Adam(self.outer_params, lr=self.learning_rate),torch.optim.Adam(self.hidden_params, lr=self.hidden_lr)]
            if self.optimizer == 'ECD':
                optimizer = [ECD_q1_scaled(self.outer_params,lr = self.learning_rate, eta = self.eta, F0=self.F0,  nu = self.nu, weight_decay = self.weight_decay),
                ECD_q1_scaled(self.hidden_params,lr = self.hidden_lr, eta = self.eta, F0=self.F0,  nu = self.nu, weight_decay = self.weight_decay)]
            print(self.hidden_params, self.outer_params)
        else:
            if self.optimizer == 'adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            if self.optimizer == 'ECD':
                optimizer = ECD_q1_scaled(self.parameters(), lr = self.learning_rate, eta = self.eta, F0=self.F0,  nu = self.nu, weight_decay = self.weight_decay)      
        return optimizer    

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y = y.type(torch.float32)
        # forward pass
        y_pred = self.forward(X).squeeze()
        # compute loss
        optimizers = self.optimizers()
        if self.hidden_lr:
            for optimizer in optimizers:
                optimizer.zero_grad()
        else:
            optimizers.zero_grad()
        loss = torch.mean(self.loss_fun(y, y_pred))
        # print(loss)
        loss.backward()
        def closure():
            return loss
        if self.hidden_lr:
            for optimizer in optimizers:
                optimizer.step(closure)
                optimizer.zero_grad()
        else:
            optimizers.step(closure)
            optimizers.zero_grad()
        self.train_hist.append(loss.item())
        return loss

    def validation_step(self, test_batch, batch_idx):
        X, y = test_batch
        y = y.type(torch.float32)
        # forward pass
        y_pred = self.forward(X).squeeze()
        # compute metrics
        loss = torch.mean(self.loss_fun(y, y_pred))
        self.val_hist.append(loss.item())
        return loss

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return False
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class DataLoaderLite:
    def __init__(self, X, y, batch_size):
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.X_batch = self.X.split(batch_size)
        self.y_batch = self.y.split(batch_size)
        self.num_batces = len(self.X_batch)


class DatasetLoader(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class custom_model(nn.Module):
    def __init__(self, **optimizer_config):
        super().__init__()
        self.optim_cfg = optimizer_config
        self.optimizer = self.optim_cfg.pop('optimizer')        
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.optim_cfg)
        if self.optimizer == 'ECD_q1':
            optimizer = ECD_q1_scaled(self.parameters(), **self.optim_cfg)
        return optimizer   

class MLP_widen(custom_model):
    '''
    This model is for ALEPH with wider width
    '''
    def __init__(self, **optimizer_config):
        super().__init__( **optimizer_config)
        self.linear1 = nn.Linear(1, 100)
        self.linear2 = nn.Linear(100, 200)
        self.linear3 = nn.Linear(200, 100)
        self.linear4 = nn.Linear(100, 1)
        self.loss_fun = nn.BCELoss()

    def forward(self, x):
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.linear3(x)
        x = nn.ReLU()(x)
        x = self.linear4(x)
        x = nn.Sigmoid()(x)
        return x


class MLP_extend(custom_model):
    '''
    This model is for ALEPH for extended depth
    '''
    def __init__(self, **optimizer_config):
        super().__init__( **optimizer_config)
        self.linear1 = nn.Linear(1, 50)
        self.linear2 = nn.Linear(50, 100)
        self.linear3 = nn.Linear(100,100)
        self.linear4 = nn.Linear(100,100)
        self.linear5 = nn.Linear(100,100)
        self.linear6 = nn.Linear(100, 50)
        self.linear7 = nn.Linear(50, 1)
        self.loss_fun = nn.BCELoss()

    def forward(self, x):
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.linear3(x)
        x = nn.ReLU()(x)
        x = self.linear4(x)
        x = nn.ReLU()(x)
        x = self.linear5(x)
        x = nn.ReLU()(x)
        x = self.linear6(x)
        x = nn.ReLU()(x)
        x = self.linear7(x)
        x = nn.Sigmoid()(x)
        return x

class MLP(custom_model):
    '''
    This model is for ALEPH
    '''
    def __init__(self, **optimizer_config):
        super().__init__( **optimizer_config)
        self.linear1 = nn.Linear(1, 50)
        self.linear2 = nn.Linear(50, 100)
        self.linear3 = nn.Linear(100, 50)
        self.linear4 = nn.Linear(50, 1)
        self.loss_fun = nn.BCELoss()

    def forward(self, x):
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.linear3(x)
        x = nn.ReLU()(x)
        x = self.linear4(x)
        x = nn.Sigmoid()(x)
        return x


class create_model(custom_model):
    def __init__(self, d, loss_fun, output, **optimizer_config):
        super(create_model, self).__init__(**optimizer_config)
        self.d = d
        self.p = 0.05
        self.linear1 = nn.Linear(self.d, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)
        self.dropout1 = nn.Dropout(self.p)
        self.dropout3 = nn.Dropout(self.p)
        self.dropout2 = nn.Dropout(self.p)
        self.output = output
        self.loss_fun = loss_fun
        self.automatic_optimization = False
    
    def forward(self, x):
        p = self.p
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = nn.ReLU()(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        if self.output == 'relu':
            x = nn.ReLU()(x)
            return x
        if self.output == 'linear':
            return x

class ReweighterDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class reweightingDataModule():
    def __init__(self, bkgd, sgnl, N, workers):
        super().__init__()
        self.bkgd = bkgd
        self.sgnl = sgnl
        self.N = N
        self.workers = workers
    def prepare_data(self):
        X_train, X_test, y_train, y_test = make_data(self.bkgd, self.sgnl, self.N)
        X_train = X_train.reshape(-1,1).astype(np.float32)
        X_test = X_test.reshape(-1,1).astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        self.X_train = torch.from_numpy(X_train)
        self.X_test = torch.from_numpy(X_test)
        self.y_train = torch.from_numpy(y_train)
        self.y_test = torch.from_numpy(y_test)

        self.train_data = TensorDataset(self.X_train, self.y_train)
        self.test_data = TensorDataset(self.X_test, self.y_test)
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=int(0.1*self.N), shuffle=True, num_workers=self.workers)
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=int(0.1*self.N), shuffle=False, num_workers=self.workers)

# Legacy code, never used now
def train(data, 
          loss,
          d = 1,
          hidden = 'relu', 
          output = 'sigmoid', 
          dropout = True, 
          optimizer = 'adam', 
          metrics = ['accuracy'], 
          verbose = 0):
    print(d)
    X_train, X_test, y_train, y_test = data
    
    N = len(X_train) + len(X_test)
    
    model = create_model(loss, d, hidden, output, dropout, optimizer, metrics, verbose)      
    
    model.compile(loss = loss,
                  optimizer = optimizer, 
                  metrics = metrics)
    
    trace = model.fit(X_train, 
                      y_train,
                      epochs = 100, 
                      batch_size = int(0.1*N), 
                      validation_data = (X_test, y_test),
                      callbacks = [earlystopping], 
                      verbose = verbose)
    print(trace.history['val_loss'][-1], '\t', len(trace.history['val_loss']), end = '\t')
    
    return model, trace



def make_data(bkgd, sgnl, N_trn=10**7, N_tst=10**5):
    y_trn = stats.bernoulli.rvs(0.5, size = N_trn)
    
    
    X_bkgd = bkgd.rvs(size = N_trn)
    X_sgnl = sgnl.rvs(size = N_trn)
    
    X_trn = np.zeros_like(X_bkgd)
    X_trn[y_trn == 0] = X_bkgd[y_trn == 0]
    X_trn[y_trn == 1] = X_sgnl[y_trn == 1]
    
    y_tst = stats.bernoulli.rvs(0.5, size = N_tst)
    
    X_bkgd = bkgd.rvs(size = N_tst)
    X_sgnl = sgnl.rvs(size = N_tst)
    
    X_tst = np.zeros_like(X_bkgd)
    X_tst[y_tst == 0] = X_bkgd[y_tst == 0]
    X_tst[y_tst == 1] = X_sgnl[y_tst == 1]
    
    return X_trn, X_tst, y_trn, y_tst

def split_data(X, y):
    # Split into train and validation sets.
    X_trn, X_vld, y_trn, y_vld = train_test_split(X, y, random_state = 666)
    
    # Standardize both the train and validation set.
    m = np.mean(X_trn, axis = 0)
    s = np.std(X_trn, axis = 0)
    X_trn = (X_trn - m) / s
    X_vld = (X_vld - m) / s
    
    return (X_trn, X_vld, y_trn, y_vld), m, s

def make_lr(bkgd, sgnl):
    return lambda x: sgnl.pdf(x) / bkgd.pdf(x)

def make_mae(bkgd, sgnl, dir_name):
    X_tst = torch.load(dir_name + 'X_tst.npy')
    lr = make_lr(bkgd, sgnl)
    lr_tst = torch.squeeze(lr(X_tst))
    
    def mae(model_lr):
        return torch.abs(model_lr(X_tst) - lr_tst).mean()
    return mae

def make_mpe(bkgd, sgnl, dir_name):
    X_tst = torch.load(dir_name + 'X_tst.npy')
    
    lr = make_lr(bkgd, sgnl)
    lr_tst = torch.squeeze(lr(X_tst))
    def mpe(model_lr):
        return torch.abs((model_lr(X_tst) - lr_tst) / lr_tst).mean() * 100
    return mpe

def make_mr(bkgd, sgnl, dir_name):
    X_tst = torch.load(dir_name + 'X_tst.npy')
    
    lr = make_lr(bkgd, sgnl)
    lr_tst = torch.squeeze(lr(X_tst))
    def mr(model_lr):
        return torch.mean(model_lr(X_tst) / lr_tst)
    return mr

def make_null_statistic(bkgd, sgnl, dir_name):
    X_tst = torch.load(dir_name + 'X_tst.npy')
    y_tst = torch.load(dir_name + 'y_tst.npy')
    X_null = X_tst[y_tst == 1]
    
    lr = make_lr(bkgd, sgnl)
    null_lr = torch.mean(lr(X_null))
    def null_statistic(model_lr):
        return abs(torch.mean(model_lr(X_null)) - null_lr)
    return null_statistic
    
def odds_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return torch.squeeze(f / (1. - f))
    return model_lr

def square_odds_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return torch.squeeze(f**2 / (1. - f**2))
    return model_lr

def exp_odds_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return torch.squeeze(torch.exp(f) / (1. - torch.exp(f)))
    return model_lr

def pure_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return torch.squeeze(f)
    return model_lr

def square_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return torch.squeeze(f**2)
    return model_lr

def exp_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return torch.squeeze(torch.exp(f))
    return model_lr

def pow_lr(model, p, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return torch.squeeze(f**p)
    return model_lr

def pow_exp_lr(model, p, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return torch.squeeze(torch.exp(f)**p)
    return model_lr

def pow_odds_lr(model, p, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return torch.squeeze( (f / (1. - f))**(p - 1))
    return model_lr

def t_tanh(x):
    return 0.5 * (torch.tanh(x) + 1)

def tanh_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return torch.squeeze(t_tanh(f) / (1. - t_tanh(f)))
    return model_lr

def t_arctan(x):
    return 0.5 + (torch.atan(x) / np.pi)

def arctan_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return torch.squeeze(t_arctan(f) / (1. - t_arctan(f)))
    return model_lr

def probit(x):
    normal = torch.distributions.Normal(loc=0., scale=1.)
    return normal.cdf(x)

def probit_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model((x - m) / s)
        return torch.squeeze(probit(f) / (1. - probit(f)))
    return model_lr

def tree_lr(model, m = 0, s = 1):
    def model_lr(x):
        x = x.reshape(x.shape[0], -1)
        f = model((x - m) / s)[:, 1]
        return np.squeeze(f / (1. - f))
    return model_lr

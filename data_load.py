from utils.losses import *
from utils.plotting import *
from utils.training import *
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
def data_load(prob_type='Gaussian', N_tot=10000):
    '''
    Returns X_train, X_test, y_train, y_test for the different problems
    '''
    if prob_type == 'Gaussian':
        bkgd = stats.norm(-0.1, 1)
        sgnl = stats.norm(+0.1, 1)
        lr = make_lr(bkgd, sgnl)
        X_train, X_test, y_train, y_test = make_data(bkgd, sgnl, N_trn = N_tot)
        return X_train, X_test, y_train, y_test
    elif 'ALEPH' in prob_type:
        data_vals_reco_full = np.load("inputs/data_vals_reco.npy")
        MC_vals_reco_full = np.load("inputs/MC_vals_reco.npy")
        MC_vals_truth_full = np.load("inputs/MC_vals_truth.npy")
        
        data_pass_reco = np.load("inputs/data_pass_reco.npy")
        MC_pass_reco = np.load("inputs/MC_pass_reco.npy")
        MC_pass_truth = np.load("inputs/MC_pass_truth.npy")
        
        data_vals_reco = data_vals_reco_full[data_pass_reco==1]
        MC_vals_reco = MC_vals_reco_full[MC_pass_reco==1]
        MC_vals_truth = MC_vals_truth_full[MC_pass_truth==1]

        Y = np.concatenate([np.ones(len(data_vals_reco)),np.zeros(len(MC_vals_reco))])
        X = np.concatenate([data_vals_reco,MC_vals_reco]).reshape(-1,1).astype(np.float32)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, random_state=123, train_size=0.5)

        return [X_train, X_test, y_train, y_test, X_scaled, X , Y]

    elif prob_type == 'NF':
        N = N_tot
        X = np.load('data/zenodo/fold/8/X_trn.npy')[:N]
        y = np.load('data/zenodo/fold/8/y_trn.npy')[:N].astype('float32')
        lr_tst = np.load('data/zenodo/fold/8/lr_tst.npy')
        data, m, s = split_data(X, y)
        X_train, X_test, y_train, y_test = data
        return [X_train, X_test, y_train, y_test, lr_tst, m, s]
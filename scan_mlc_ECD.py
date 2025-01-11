# Utility imports
# General imports
import torch
import torch.nn as nn
import os
# import tensorflow as tf
from scipy import stats
# Used for distributions libraries.
from scipy import stats
# Utility imports
from utils.losses import *
from utils.plotting import *
from utils.training import *

import pickle
import sys


print(torch.cuda.is_available())
eta = 5e3
lr_vals = [1.0]
# wd_vals = [1e-4,1e-3,1e-2]
# learning_rate = 0.1
F0 = -1.
nu = 0.1

file_name = 'scan_mlc_ECD_results'
loss_funcs = ['linear', 'square', 'exponl']#[mlc, square_mlc, exp_mlc]

# part = int(sys.argv[1]) + 1
reps = 5
optimizer = 'ECD'


# Data parameters
N = 10**6
X = np.load('data/zenodo/fold/8/X_trn.npy')[:N]
y = np.load('data/zenodo/fold/8/y_trn.npy')[:N].astype('float32')
data, m, s = split_data(X, y)

class train_val_loader(pl.LightningDataModule):
    def __init__(self, data, N, workers):
        super().__init__()
        self.N = N
        self.data = data
        self.workers = workers
    def prepare_data(self):
        X_train, X_test, y_train, y_test = self.data
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
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

train_val_data = train_val_loader(data, N, 1)

max_epochs = 500
min_epochs = 15

X_mae = np.load('data/zenodo/fold/8/X_tst.npy')
X_mae = torch.from_numpy(X_mae)
lr_tst = np.load('data/zenodo/fold/8/lr_tst.npy')
lr_tst = torch.from_numpy(lr_tst)

def mae(model_lr):
    abs_dif = abs(model_lr(X_mae) - lr_tst)
    #print(np.mean(abs_dif < 100), end = ' ')
    return abs_dif[abs_dif < 100].mean()
def stdae(model_lr):
    abs_dif = abs(model_lr(X_mae) - lr_tst)
    return abs_dif[abs_dif < 100].std()

filestr = 'models/zenodo/mlc/'






dict_list = []
for loss_func in loss_funcs:
    for learning_rate in lr_vals:
        # for i in range(reps//10*(part-1), reps//10*part):
        for i in range(reps):
            if loss_func == 'linear': loss_fn = mlc; output = 'relu'; lr_fn = pure_lr
            elif loss_func == 'square': loss_fn = square_mlc; output = 'linear'; lr_fn = square_lr
            elif loss_func == 'exponl': loss_fn = exp_mlc; output = 'linear'; lr_fn = exp_lr

            params = {'loss_fun':loss_fn, 'd': 6, 'output': output, 'optimizer': optimizer, 'learning_rate': learning_rate, 'eta': eta, 'F0': F0, 'nu': nu}
            model_path = filestr + loss_func + '/ecd/'

            checkpoint_callback = ModelCheckpoint(
                dirpath = model_path,
                filename = 'model_{}'.format(i),
                monitor = 'val_loss',
                mode = 'min',
                save_weights_only = True
            )


            trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience = 10), checkpoint_callback], min_epochs=min_epochs, log_every_n_steps=10, enable_progress_bar=True)

            model = create_model_original(**params)

            trainer.fit(model, train_val_data)
            

            try: os.mkdir(model_path)
            except OSError as error: print(error)

            # model_file = model_path + 'model_{}.h5'.format(i)
            # torch.save(model.state_dict(), model_file)

            train_losses = model.train_hist
            val_losses = model.val_hist

            checkpoint = torch.load(checkpoint_callback.best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            lr = lr_fn(model, m , s)
            mae_1 = mae(lr).detach().numpy()

            scan_res = dict(mae = mae_1, optimizer = optimizer, learning_rate = learning_rate, eta = eta, F0 = F0, nu = nu, classifier = loss_func, train_loss = train_losses, val_loss = val_losses, path = checkpoint_callback.best_model_path)
            dict_list.append(scan_res)
            print(f'eta: {eta} lr: {learning_rate} mae: {mae_1} classifier: ', loss_func)
            del model

# with open('models/zenodo/mlc/' + file_name + '_{}.pkl'.format(part), 'wb') as fout:
#     pickle.dump(dict_list, fout)


with open('models/zenodo/mlc/' + file_name + '.pkl', 'wb') as fout:
    pickle.dump(dict_list, fout)
## General imports ##
# Used for selecting GPU.
# import tensorflow as tf
import torch
import uuid
import torch.nn as nn
# Used for distributions libraries.
from scipy import stats
from lion_pytorch import Lion
## Utility imports ##
from utils.losses import *
from utils.plotting import *
from utils.training import *
import data_load as dl
from parser import create_parser
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import vmap #used to map functions that act on a single input to make them act on a batch
import matplotlib as mpl
import json
import os
mpl.rcParams.update(mpl.rcParamsDefault)
# Just this seed isn't enough to ensure results are completely replicable,
# as there is randomness in GPU execution.
# np.random.seed(666)
# torch.manual_seed(seed=42)

parser = create_parser()
args = parser.parse_args()
betas =  (args.beta1, args.beta2) # Choosing the betas for Adam and Lion
N = args.N # The sample size (N samples are drawn from each distribution for training)
n_models = args.n_models # Number of initializations to ensemble over
has_bfloat16 = args.bf # Set true if you want to use dtype bfloat 16. Not recommended
run_name = args.name_of_run
batch_size = int(args.train_batch_size)

if args.save == 'true': to_save = True
else: to_save = False

### Need to add ECD_q0_scaled
if 'ECD' in args.optimizer:
    optimizer_config = dict(
        optimizer = args.optimizer,
        lr = args.lr,
        eta = args.eta,
        F0 = args.F0,
        nu = args.nu,
    )

elif args.optimizer == 'lion':
    optimizer_config = dict(
        optimizer = args.optimizer,
        lr = args.lr,
        betas = betas,
    )
elif args.optimizer == 'adam':
    optimizer_config = dict(
        optimizer = args.optimizer,
        lr = args.lr,
       betas = betas,
    )  


if args.loss_fn == 'mlc':
    my_loss_fn = mlc
    output = 'relu'
elif args.loss_fn == 'square_mlc':
    my_loss_fn = square_mlc
    output = 'linear'
elif args.loss_fn == 'exp_mlc':
    my_loss_fn = exp_mlc
    output = 'linear'

## Data Generation
## This puts in memory the whole dataset
data = dl.data_load(args.prob_type)
X_train, X_test, y_train, y_test = data[:4]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


X_train, X_test, y_train, y_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

if args.prob_type == 'Gaussian':

    train_batch_size = int(0.1*N)
    test_batch_size = int(0.1*N)
    X_train, X_test, y_train, y_test =  X_train.view(-1,1), X_test.view(-1,1), y_train.view(-1,1), y_test.view(-1,1)

elif 'ALEPH' in args.prob_type:
    if batch_size > 0:
        train_batch_size = batch_size
    else:
        train_batch_size = 1000
    test_batch_size = 10000
    n_models += 1
    y_train, y_test = y_train.view(-1,1), y_test.view(-1,1)

elif args.prob_type == 'NF':
    train_batch_size = int(0.1*N)
    test_batch_size = int(0.1*N)
    
import gc
gc.collect()
torch.cuda.empty_cache()

print('Begin training')

models = []
# Set path for saving best checkpoint so far
ckpt_path = os.path.join(args.prob_type, args.optimizer, 'checkpoints')

# Run over n_models initializations
test_loss_best = [0]*(n_models)
train_loss_best = [0]*(n_models)
for run in range(n_models):
    # Set unique filename for savining the model checkpoint
    ckpt_name = str(uuid.uuid4())
    print(f'Initialized for model: {run}')
    # Create NN
    if args.prob_type == 'Gaussian':
        model = create_model(d=1, loss_fun=my_loss_fn, output=output, **optimizer_config).to(device)
    elif args.prob_type == 'ALEPH':
        model = MLP(**optimizer_config).to(device)
    elif args.prob_type == 'ALEPH_extend':
        model = MLP_extend(**optimizer_config).to(device)
    elif args.prob_type == 'ALEPH_widen':
        model = MLP_widen(**optimizer_config).to(device)

    # Set optimizer and train
    optimizer = model.configure_optimizers()
    #This simply chunks the whole dataset in different batches. So it's the same for the various problems
    train_loader = DataLoaderLite(X = X_train, y = y_train, batch_size=train_batch_size)
    test_loader = DataLoaderLite(X = X_test, y = y_test, batch_size=test_batch_size)
    # Earlystopper will stop training if no improvement on best loss after 'patience' number of epochs
    earlystopper = EarlyStopper(patience=args.patience, min_delta=0)

    '''
    # Calculate number of steps in each epoch for train and test
    steps = (len(X_train) + train_batch_size - 1)//train_batch_size
    steps_test = (len(X_test) + test_batch_size - 1)//test_batch_size
    '''
    train_losses, test_losses = [], []

    for epoch in range(args.epochs):
        train_loss_epoch = []
        test_loss_epoch = []
        # for step in range(steps):
            # x,y = train_loader.next_batch()
        for x, y in zip(train_loader.X_batch, train_loader.y_batch):
            y = y.type(torch.float32)
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            ##This is the proper one that uses bfloats16, but needs an Ampere GPU. bfloats are nice because they have the same range as float 32
            if has_bfloat16:
                with torch.autocast(device_type = (torch.device(device)).type, dtype = torch.bfloat16): ##Read the documentation of autocast on putorch
                    if 'ALEPH' in args.prob_type: #For ALEPH it's different since it's using BCE. BCE by default takes the mean over batch dimension to torhc.mean is redundant
                        loss = torch.mean(model.loss_fun(model.forward(x), y))
                    else:
                        #SWAPPED THE ARGUMENTS!
                        loss = torch.mean(model.loss_fun(y, model.forward(x))) 
            else:
                pred = model.forward(x)
                if 'ALEPH' in args.prob_type: #ALEPH is different since it's using BCE
                    loss = model.loss_fun(pred, y)
                else:
                    #SWAPPED THE ARGUMENTS!
                    loss = torch.mean(model.loss_fun(y, pred))
                    #breakpoint()
            #detect nans
            if loss.item() != loss.item():
                print(f"Loss: {loss.item()}")
                import sys
                sys.exit(0)
            loss.backward()
            def closure():
                return loss
            optimizer.step(closure)
            
            with torch.no_grad():
                train_loss_epoch.append(loss.item())
        # Compute average loss of the epoch                
        train_loss_epoch_mean = np.mean(np.array(train_loss_epoch))
        train_losses.append(train_loss_epoch_mean)
        stop = earlystopper.early_stop(train_loss_epoch_mean)
        print(f'Epoch: {epoch}\t| Train Loss: {train_loss_epoch_mean:.5e}')
        if to_save:
            if earlystopper.counter == 0:
                # Save best model if earlystopper has reset
                print(f'Saving best model so far...')
                try: 
                    os.mkdir(ckpt_path)
                except OSError as error:
                    pass
                torch.save(model, os.path.join(ckpt_path, ckpt_name))
                train_loss_best[run] = train_loss_epoch_mean
        del train_loss_epoch
        if stop:
            break
    if to_save:
        # Plot train loss for each initialization
        plt.plot(train_losses, label = f'Run {run}')
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        # plt.ylim([0.654,0.66])
        plt.legend()
        plt.savefig(os.path.join(args.prob_type, args.optimizer, 'train_loss.png'))
        # Empty cuda cache and load model from checkpoint
        torch.cuda.empty_cache()
        model = torch.load(os.path.join(ckpt_path, ckpt_name), weights_only = False).to(device)
        model.eval()
    # Compute test loss
    with torch.no_grad():
        # for _ in range(steps_test):
        #     x,y = test_loader.next_batch()
        for x, y in zip(test_loader.X_batch, test_loader.y_batch):
            y = y.type(torch.float32)
            x,y = x.to(device), y.to(device)
            pred = model.forward(x)
           
            #Before here it was x[0] and y[0], why? I also added a mean and swapped the arguments when needed
            #val_loss = model.loss_fun(pred[0], y[0])
            if 'ALEPH' in args.prob_type: #ALEPH is different since it's using BCE
                val_loss = torch.mean(model.loss_fun(pred, y))
            else:
                val_loss = torch.mean(model.loss_fun(y, pred))

            test_loss_epoch.append(val_loss.item())
        test_loss_epoch_mean = np.mean(np.array(test_loss_epoch))
        test_losses.append(test_loss_epoch_mean)
        print(f'Epoch: {epoch}\t|Validation Loss: {test_loss_epoch_mean}, |Train Loss: {train_loss_epoch_mean}|')
        test_loss_best[run] = test_loss_epoch_mean
    model.to('cpu')
    models.append(model)
plt.close()

def compute_ensemble_metrics(all_predictions, truth):
    '''
    all_predictions has shape (num_models, num_points)
    '''

    average_prediction = np.mean(all_predictions, axis=0) #Average prediction of n_models as function of data points
    average_model_var = np.var(all_predictions/truth - 1., axis=0) 
    errors = average_prediction/truth - 1.
    average_error = np.mean(errors)
    average_var = np.mean(average_model_var)

    return average_prediction, average_error, average_var, errors, average_model_var

def metrics(prob_type):
    if prob_type == 'Gaussian':

        deltax = 8
        bkgd = stats.norm(-0.1, 1)
        sgnl = stats.norm(+0.1, 1)
        num_points = args.N
        xs = np.linspace(-deltax/2, deltax/2, num_points)

        xs = torch.tensor(xs, dtype = torch.float32, requires_grad = False, device = 'cpu').view(-1,1)
        lr = make_lr(bkgd, sgnl)(xs.cpu()).squeeze()
        
        if args.loss_fn == 'mlc':
            all_predictions = np.array([model(xs).squeeze().detach().cpu().numpy() for model in models])
        if args.loss_fn == 'square_mlc':
            all_predictions = np.array([torch.pow(model(xs),2).squeeze().detach().cpu().numpy() for model in models])
        if args.loss_fn == 'exp_mlc':
            all_predictions = np.array([torch.exp(model(xs)).squeeze().detach().cpu().numpy() for model in models])
        
        ensembled_pred = np.mean(all_predictions, axis = 0)
        ensemble_var = np.var(all_predictions, axis = 0)
        error = np.abs(ensembled_pred-lr)

        mae_error = sum(np.abs(torch.exp(model(xs)).squeeze().detach().cpu().numpy() - lr) for model in models)/len(models)
        
        #Now I probability-avearge over the domain. In this case I use a gaussian with zero mean
        #Simple trapezoidal rule using the same points
        averaged_ensembled_error = (deltax/num_points)*np.sum(stats.norm(0,1).pdf(xs.cpu()).squeeze()*error)
        averaged_mae_error = (deltax/num_points)*np.sum(stats.norm(0,1).pdf(xs.cpu()).squeeze()*mae_error)
        averaged_ensemble_var = (deltax/num_points)*np.sum(stats.norm(0,1).pdf(xs.cpu()).squeeze()*ensemble_var)
        truth = lr.copy()

        return dict(predictions = all_predictions, weighted_mae = averaged_mae_error, weighted_error = averaged_ensembled_error, weighted_var = averaged_ensemble_var)
    elif prob_type == 'NF':
        lr_tst, m, s = data[4:]
        lr_tst, m, s = torch.from_numpy(lr_tst), torch.from_numpy(m), torch.from_numpy(s)
        X_mae = np.load('data/zenodo/fold/8/X_tst.npy')
        X_mae = torch.from_numpy(X_mae)
        if args.loss_fn == 'mlc':
            lrs = [pure_lr(model, m, s)(X_mae).detach().cpu().numpy() for model in models]
        elif args.loss_fn == 'square_mlc':
            lrs = [square_lr(model, m, s)(X_mae).detach().cpu().numpy() for model in models]
        else:
            lrs = [exp_lr(model, m, s)(X_mae).detach().cpu().numpy() for model in models]

        def mae_var(model_lrs):
            diff = abs(model_lrs - lr_tst.numpy())
            final_diff = np.array([model_diff[model_diff < 100] for model_diff in np.array([model_diff[model_diff < 100] for model_diff in diff])])
            #print(np.mean(abs_dif < 100), end = ' ')
            return np.mean(final_diff), np.mean(np.var(final_diff, axis = 0))

        mae, var = mae_var(lrs)
        return dict(lrs = lrs, mae = mae, var = var, train_losses = train_loss_best, test_losses = test_loss_best)
    elif 'ALEPH' in args.prob_type:
        
        n2_preds = []
        X_scaled, X, Y = data[4:]
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to('cpu')
        scores = models[0](x_tensor).detach().cpu().numpy().flatten()
        n2, _, _ = plt.hist(1.-X[Y==0],weights = scores[Y==0]/(1.-scores[Y==0]),density=True,bins=np.linspace(0,0.5,20),alpha=0.5,histtype="step",color="black",label="Sim. [rw]")
        n,bn,_=plt.hist(1.-X[Y==1],density=True,bins=np.linspace(0,0.5,20),alpha=0.5,label="Data")
        n1,_,_=plt.hist(1.-X[Y==0],density=True,bins=np.linspace(0,0.5,20),alpha=0.5,label="Sim.")
        plt.plot(0.5*(bn[0:-1]+bn[1:]),n/n1,label="nominal")
        plt.plot(0.5*(bn[0:-1]+bn[1:]),n/n2,label=f"reweighted Run 0")
        plt.ylim([0.85,1.1])

        mae = np.zeros((n_models-1, len(n2)))
        for i in range(1,n_models,1):
            scores_i = models[i](x_tensor).detach().numpy().flatten()
            n2_i,_=np.histogram(1.-X[Y==0].flatten(),weights = scores_i[Y==0]/(1.-scores_i[Y==0]),density=True,bins=np.linspace(0,0.5,20))
            n2_preds.append(n2_i)
            plt.plot(0.5*(bn[0:-1]+bn[1:]),n2_i/n, label = f'reweighted Run {i}')
            mae[i-1] = np.nan_to_num(np.abs(n2_i/n - 1))
        
        plt.legend()
        plt.xlabel("1-T")
        plt.ylabel("likelihood ratio")
        plt.savefig(os.path.join(args.prob_type, args.optimizer, 'likelihood_ratio.png'))
        plt.close()
        n2_preds = np.array(n2_preds).copy()
        n2_vars = np.var(n2_preds, axis = 0).copy()
        n2_vars_probability_weighted = n*n2_vars/(np.sum(n))
        avg_var = np.mean(np.nan_to_num(n2_vars))
        avg_var_probability_weighted =  np.mean(np.nan_to_num(n2_vars_probability_weighted))
        n2_avg_pred = np.mean(n2_preds, axis = 0)
        probability_weighted_avg = n*(n2_avg_pred/n-1)/(np.sum(n))
        weighted_mae = n*np.mean(mae, axis = 0)/np.sum(n)
        weighted_mae = np.sum(np.nan_to_num(weighted_mae))
        weighted_error_avged_over_T = np.sum(np.abs(np.nan_to_num(probability_weighted_avg)))/len(n2)
        
        plt.plot(0.5*(bn[0:-1]+bn[1:]), np.mean(mae, axis = 0))
        plt.xlabel('1-T')
        plt.ylabel('Mean Absolute Error')
        plt.savefig(os.path.join(args.prob_type, args.optimizer, 'MAE.png'))
        plt.close()
        
        return dict(n2_preds=n2_preds, n2_avg_pred=n2_avg_pred, train_losses = train_loss_best, test_losses = test_loss_best, weighted_mae=weighted_mae, avg_var_probability_weighted=avg_var_probability_weighted, weighted_error=weighted_error_avged_over_T, n=n, n1=n1, n2=n2)

def save_run(prob_type):
    try: 
        os.mkdir(os.path.join(prob_type, args.optimizer, 'metrics'))
    except OSError as error:
        print(error)
    
    file_name = run_name + str(uuid.uuid4())
    my_metrics = metrics(prob_type)
    my_metrics.update(vars(args))
    file_path = os.path.join(prob_type, args.optimizer, 'metrics', file_name)
    np.save(file_path, my_metrics)

def ensembled_error(prob_type, models, num_points):

    if prob_type == "Gaussian":

        deltax = 8
        bkgd = stats.norm(-0.1, 1)
        sgnl = stats.norm(+0.1, 1)
        
        xs = np.linspace(-deltax/2, deltax/2, num_points)

        xs = torch.tensor(xs, dtype = torch.float32, requires_grad = False, device = device).view(-1,1)
        lr = make_lr(bkgd, sgnl)(xs.cpu()).squeeze()
        #Added the exponential!
        ensembled_pred = sum(torch.exp(model(xs)).squeeze().detach().cpu().numpy() for model in models)/len(models)
        error = np.abs(ensembled_pred-lr)

        #Now I probability-avearge over the domain. In this case I use a gaussian with zero mean
        #Simple trapezoidal rule using the same points
        averaged_ensembled_error = (deltax/num_points)*np.sum(stats.norm(0,1).pdf(xs.cpu()).squeeze()*error)
        return averaged_ensembled_error

def save_ensembled_error(args, error):
    import uuid
    import json
    dir_name = f'./scans/{args.optimizer}/'
    os.makedirs(dir_name , exist_ok=True )
    with open(f'{dir_name}/f{uuid.uuid4()}.json', "w") as f:
        json.dump({"args": vars(args), "error": error}, f)


if args.save_ensembled_error == 'true':
    error = ensembled_error(args.prob_type, models, args.N)
    print(f"Averaged ensembled error: {error}")
    save_ensembled_error(args, error)



metrics(args.prob_type)

save_run(args.prob_type)
del models
gc.collect()

import argparse

def create_parser():
    parser = argparse.ArgumentParser(prog='Train Likelihood Ratios 1-D Gaussian')

    parser.add_argument('-p', '--prob_type', default='Gaussian', help='Choose problem type, either Gaussian, ALEPH, or NF (Normalizing Flow)', choices=['Gaussian', 'ALEPH', 'NF', 'ALEPH_extend', 'ALEPH_widen'], type=str)
    parser.add_argument('-N', '--N', default=10**5, help='Sample size drawn from bkg and sgl distributions', type=int)
    parser.add_argument('-n_models', '--n_models', default=3, help='number of models to train and average', type=int)
    parser.add_argument('-opt', '--optimizer', default='ECD_q1_F2_thetastep', help='Choose optimizer to use', type=str)
    parser.add_argument('--F0', default=-1, help='Choose F0 val for ECD optimizer', type=float)
    parser.add_argument('--eta', default=10, help='Choose eta val for ECD optimizer', type=float)
    parser.add_argument('--nu', default=0, help='Choose nu val for ECD optimizer', type=float)
    parser.add_argument('--lr', help='Choose learning rate', type=float)
    parser.add_argument('--loss_fn', default='exp_mlc', help='Choose loss function', choices = ['mlc', 'square_mlc', 'exp_mlc'], type=str)
    parser.add_argument('--beta1', default=0.9, help='Choose beta1 value', type=float)
    parser.add_argument('--beta2', default=0.9, help='Choose beta2 value', type=float)
    parser.add_argument('--epochs', default=500, help='Choose max training epochs', type=int)
    parser.add_argument('--bf', default=False, help='Enable bfloat16', type=bool)
    parser.add_argument('--patience', default=10, help='Choose patience for early stopping', type=int)
    parser.add_argument('--save', default = "true", choices = ['true', 'false'], type = str, help = "Whether to save the best models and some plots. Set to false for scans.")
    parser.add_argument('--save_ensembled_error', default = "false", choices = ['true', 'false'], type = str, help = "Whether to save save_ensembled_error.")
    parser.add_argument('--name_of_run', default = '', type = str, help = 'You can specify a run name to the saved metrics to identify the run')
    parser.add_argument('--train_batch_size', default = 0, help = 'Specify training batch size', type=int)
    return parser


def analyze_parser():
    parser = argparse.ArgumentParser(prog='Analyzing Likelihood Ratios')

    parser.add_argument('-p', '--prob_type', default='Gaussian', help='Choose problem type, either Gaussian, ALEPH, or NF (Normalizing Flow)', choices=['Gaussian', 'ALEPH', 'NF', 'ALEPH_extend'], type=str)
    parser.add_argument('-opt', '--optimizer', default='ECD_q1_F2_thetastep', help='Choose optimizer to use', type=str)

    return parser
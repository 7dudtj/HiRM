'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

# for the environment
from directory import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
dataset_step = {'amazon-book': 0,
                'gowalla': 1,
                'yelp2018': 2,
                'lastfm': 3
                }
all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

# config files for exp 1, 2
config['expdevice'] = args.expdevice
config['svdvalue'] = int(args.svdvalue)
config['svdtype'] = args.svdtype
# for exp1
config['alpha_start'] = args.alpha_start
config['alpha_end'] = args.alpha_end
config['alpha_step'] = args.alpha_step
# for exp2
# config['filter'] = args.filter
# config['filter_option'] = args.filter_option
config['filter'] = eval(args.filter)
filter_list = ['linear', 'ideal-low-pass', 'gaussian', 'heat-kernel', 'butterworth', 'gfcf-linear-autoencoder', 'gfcf-Neighborhood-based', 'inverse']

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed


dataset = args.dataset
simple_model = args.simple_model
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
████████╗███████╗ █████╗ ███╗   ███╗    ██╗      ██████╗████████╗███╗   ███╗
╚══██╔══╝██╔════╝██╔══██╗████╗ ████║    ██║     ██╔════╝╚══██╔══╝████╗ ████║
   ██║   █████╗  ███████║██╔████╔██║    ██║     ██║  ███╗  ██║   ██╔████╔██║
   ██║   ██╔══╝  ██╔══██║██║╚██╔╝██║    ██║     ██║   ██║  ██║   ██║╚██╔╝██║
   ██║   ███████╗██║  ██║██║ ╚═╝ ██║    ███████╗╚██████╔╝  ██║   ██║ ╚═╝ ██║
   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝    ╚══════╝ ╚═════╝   ╚═╝   ╚═╝     ╚═╝                                                              
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
print(logo)

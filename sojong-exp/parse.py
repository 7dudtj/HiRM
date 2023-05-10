'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse

from typing import Union


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    # add parser argument
    
    # for exp 1, 2
    parser.add_argument('--simple_model', type=str, default='none', help='simple-rec-model, support [none, lgn-ide, gf-cf, exp1, exp2]')
    parser.add_argument('--svdvalue', type=int, default="256", help='default value is 256')
    parser.add_argument('--svdtype', type=str, default="sparsesvd", help='default=sparsesvd, scipy, fbpca, sklearn-rand')
    parser.add_argument('--expdevice', type=str, default='cpu', help='cuda:0, cuda:1,... for cuda, else cpu')
    # for exp 1
    parser.add_argument('--alpha_start', type=float, default=0.3, help='0~1 float value')
    parser.add_argument('--alpha_end', type=float, default=0.3, help='0~1 float value')
    parser.add_argument('--alpha_step', type=float, default=0.05, help='step of alpha values')
    # for exp2
    # parser.add_argument('--filter', type=str, default='ideal-low-pass', help='linear, ideal-low-pass, gaussian, heat-kernel, butterworth, gfcf-linear-autoencoder, gfcf-Neighborhood-based')
    # parser.add_argument('--filter_option', type=str, default=-1, help='gaussian(alpha), heat-kernel(alpha), butterworth filter(1,2,3), gfcf-linear-autoencoder(mu)')
    parser.add_argument('--filter', nargs='?', default='[]', help='[filter_name, option] - [linear], [ideal-low-pass], [gaussian, mu], [heat-kernel, alpha], [butterworth, 1|2|3], [gfcf-linear-autoencoder, mu], [gfcf-Neighborhood-based, ]')

    return parser.parse_args()

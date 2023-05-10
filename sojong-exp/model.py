"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import pkg_resources
from typing import Any
import world
import torch
import time
from dataloader import BasicDataset
from torch import nn
import scipy.sparse as sp
try:
    import scipy.sparse.linalg
    import cupyx.scipy.sparse.linalg as cusplg
    import cupyx.scipy.sparse as cusp
except ImportError:
    has_cupyx = False
    pass
import numpy as np
from sparsesvd import sparsesvd
import math
from filter import *

from sklearn.decomposition import TruncatedSVD
import fbpca
from sklearn.utils.extmath import randomized_svd


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
    
class LGCN_IDE(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i = d_mat
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_u = d_mat
        d_mat_u_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsr()
        end = time.time()
        print('training time for LGCN-IDE', end-start)
        
    def getUsersRating(self, batch_users, ds_name):
        norm_adj = self.norm_adj
        batch_test = np.array(norm_adj[batch_users,:].todense())
        U_1 = batch_test @ norm_adj.T @ norm_adj
        if(ds_name == 'gowalla'):
            U_2 = U_1 @ norm_adj.T @ norm_adj
            return U_2
        else:
            return U_1

class GF_CF(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        ut, s, self.vt = sparsesvd(self.norm_adj, 256)
        end = time.time()
        print('training time for GF-CF', end-start)
        
    def getUsersRating(self, batch_users, ds_name):
        norm_adj = self.norm_adj
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[batch_users,:].todense())
        U_2 = batch_test @ norm_adj.T @ norm_adj
        if(ds_name == 'amazon-book'):
            ret = U_2
        else:
            U_1 = batch_test @  self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
            ret = U_2 + 0.3 * U_1
        return ret


class EXPS(object):
    def __init__(self, adj_mat) -> None:
        self.adj_mat = adj_mat
    
    def convert_sp_mat_to_sp_tensor(self, X) -> torch.sparse.FloatTensor: 
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    def train(self):
        world.cprint(f"Is not Vanilla GF-CF")
        world.cprint(f"simple-model: {world.simple_model}")
        world.cprint(f"dataset: {world.dataset}")
        world.cprint(f"svd package: {world.config['svdtype']}")
        world.cprint(f"svd dimension: {world.config['svdvalue']}")
        world.cprint(f"exp2 device: {world.config['expdevice']}")
        world.cprint(f"filter: {world.config['filter']}")
        # world.cprint(f"filter_option: {world.config['filter_option']}")

        adj_mat = self.adj_mat # (52643, 91599)
        print("training start")
        rowsum = np.array(adj_mat.sum(axis=1)) # (52643, 1)
        d_inv = np.power(rowsum, -0.5).flatten() # (52643,)
        d_inv[np.isinf(d_inv)] = 0. # (52643,)
        d_mat = sp.diags(d_inv) # (52643, 52643) / D_{U}^{-1/2}
        norm_adj = d_mat.dot(adj_mat) # (52643, 91599) / D_{U}^{-1/2}.R

        colsum = np.array(adj_mat.sum(axis=0)) # (1, 91599)
        d_inv = np.power(colsum, -0.5).flatten() # (91599,)
        d_inv[np.isinf(d_inv)] = 0. # (91599,)
        d_mat = sp.diags(d_inv) # (91599, 91599) / D_{I}^{-1/2}
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv) # (91599, 91599) / D_{I}^{1/2}
        norm_adj = norm_adj.dot(d_mat) # (52643, 91599) / D_{U}^{-1/2}.R.D_{I}^{-1/2} = R Tilda (normalized rating matrix)
        self.norm_adj = norm_adj.tocsc() # (52643, 91599)
        self.norm_adj_sparse_tensor = None

    # do svd - low rank factorization
        world.cprint(f"Is not Vanilla GF-CF")
        world.cprint(f"svd package: {world.config['svdtype']}")
        world.cprint(f"svd value: {world.config['svdvalue']}")
        # need to do SVD - singular value is 
        if world.config['svdtype'] == 'sparsesvd':
            ut, self.s, self.vt = sparsesvd(self.norm_adj, world.config['svdvalue']) # (256, 91599) / sparsesvd at R Tilda -> V^{T} (Singular Vector, i * i)
        elif world.config['svdtype'] == 'scipy':
            ut, self.s, self.vt = sp.linalg.svds(self.norm_adj, k=world.config['svdvalue'], which='LM')
            self.s = self.s[::-1]
            self.vt = self.vt[::-1]
        elif world.config['svdtype'] == 'cupy':
            # change class scipy.sparse._csc.csc_matrix to cupyx
            try:
                norm_adj_cupyx_sparse_matrix = cusp.csc_matrix(self.norm_adj)
                ut, self.s, self.vt = cusplg.svds(norm_adj_cupyx_sparse_matrix, k=world.config['svdvalue'], which='LM')
                self.s = self.s.get()[::-1]
                self.vt = self.vt.get()[::-1]
            except ImportError:
                print("Don't have module cupy! Change SVD module!")
                exit(0)
        elif world.config['svdtype'] == 'fbpca':
            ut, self.s, self.vt = fbpca.pca(self.norm_adj, k=world.config['svdvalue'], raw=True)
        elif world.config['svdtype'] == 'sklearn-rand':
            ut, self.s, self.vt = randomized_svd(self.norm_adj, n_components=world.config['svdvalue'])
        elif world.config['svdtype'] == 'torch':
            self.norm_adj_sparse_tensor = self.convert_sp_mat_to_sp_tensor(self.norm_adj)
            ut, self.s, self.vt = torch.svd_lowrank(self.norm_adj_sparse_tensor, q=world.config['svdvalue'], niter=2)
            self.s, self.vt = self.s.numpy(), self.vt.numpy().T
        elif world.config['svdtype'] == 'torch_cuda':
            self.norm_adj_sparse_tensor = self.convert_sp_mat_to_sp_tensor(self.norm_adj)
            self.norm_adj_cuda_sparse = self.norm_adj_sparse_tensor.to(world.config['expdevice'])
            ut, self.s, self.vt = torch.svd_lowrank(self.norm_adj_cuda_sparse, q=world.config['svdvalue'])
            self.s, self.vt = self.s.cpu().numpy(), self.vt.T.cpu().numpy()
            del self.norm_adj_cuda_sparse
        else:
            print(f"we have no package named {world.config['svdtype']}")
            raise NotImplementedError
        print("SVD End")
        del ut

        if world.config['expdevice'][:4] == 'cuda':
            # # let's check if we don't use large linear filter, just multiply with diagonal matrices!
            if type(self) is not EXP2 or world.config['filter'][0] == 'linear':
                if world.dataset == 'amazon-book':
                    print('Amazon dataset is not Suitable for Commercial GPU - need 48GB of VRAM')
                    print('Use Sparse Matrix Multiplication of CUDA')
                    if self.norm_adj_sparse_tensor is None:
                        self.norm_adj_cuda_sparse = self.convert_sp_mat_to_sp_tensor(self.norm_adj).to(world.config['expdevice'])
                    else:
                        self.norm_adj_cuda_sparse = self.norm_adj_sparse_tensor.to(world.config['expdevice'])
                    print('Created self.norm_adj_cuda_sparse')
                    # if we have 16GB vram, then we can use this method - sparse linear Filter
                    # self.norm_adj_cuda = self.convert_sp_mat_to_sp_tensor(self.norm_adj).to(world.config['expdevice'])
                    # self.linear_Filter_cuda_sparse = torch.mm(self.norm_adj_cuda.T, self.norm_adj_cuda)
                    # del self.norm_adj_cuda
                else:
                    if self.norm_adj_sparse_tensor is None:
                        self.norm_adj_cuda_sparse = self.convert_sp_mat_to_sp_tensor(self.norm_adj).to(world.config['expdevice'])
                    else:
                        self.norm_adj_cuda_sparse = self.norm_adj_sparse_tensor.to(world.config['expdevice'])
                    self.linear_Filter_cuda = torch.mm(self.norm_adj_cuda_sparse.T, self.norm_adj_cuda_sparse).to_dense()
                    del self.norm_adj_cuda_sparse
                    print("Created self.linear_Filter_cuda")
            # left_mat: D_I^1/2 @ V : this V is U_bar from svd
            left_mat = self.d_mat_i @ self.vt.T
            # right_mat: V.T @ D_I^{-1/2}
            right_mat = self.vt @ self.d_mat_i_inv
            self.left_mat_cuda, self.right_mat_cuda = torch.FloatTensor(left_mat).to(world.config['expdevice']), torch.FloatTensor(right_mat).to(world.config['expdevice'])
            print("Created left and right matrix")
            del left_mat
            del right_mat

class EXP1(EXPS):
    def __init__(self, adj_mat):
        super(EXP1, self).__init__(adj_mat)

    def train(self):
        start = time.time()
        super(EXP1, self).train()
        end = time.time()
        print('training time for GF-CF', end-start)
        
    def getUsersRating(self, alpha, batch_users=None, batch_ratings=None):
        if world.config['expdevice'] == 'cpu':
            adj_mat = self.adj_mat #tolil
            batch_test = np.array(adj_mat[batch_users,:].todense())
            U_2 = batch_test @ self.norm_adj.T @ self.norm_adj
            U_1 = batch_test @ self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
            return alpha * U_1 + U_2
            return ret
        else:
            batch_test = batch_ratings.to_sparse()
            if world.dataset != 'amazon-book':
                U_2 = batch_test @ self.linear_Filter_cuda
            else:
                U_2 = batch_test @ self.norm_adj_cuda_sparse.T @ self.norm_adj_cuda_sparse
                # U_2 = batch_test @ self.linear_Filter_cuda_sparse
            U_1 = batch_test @ self.left_mat_cuda @ self.right_mat_cuda
            ret = alpha * U_1 + U_2
            return ret
        
def returnFilter():
    if world.config['filter'][0] not in world.filter_list:
        raise NotImplementedError
    if world.config['filter'][0] == 'linear':
        return LinearFilter()
    elif world.config['filter'][0] == 'ideal-low-pass':
        return IdealLowPassFilter()
    elif world.config['filter'][0] == 'gaussian':
        if len(world.config['filter']) > 2:
            print("Should need at least two argument: 'gaussian', mu(float)")
            raise AssertionError
        return GaussianFilter(*world.config['filter'][1:])
    elif world.config['filter'][0] == 'heat-kernel':
        if len(world.config['filter']) > 2:
            print("Should need two argument: 'heat-kernel', alpha(float)")
            raise AssertionError
        return HeatKernelFilter(*world.config['filter'][1:])
    elif world.config['filter'][0] == 'butterworth':
        if len(world.config['filter']) > 2:
            print("Should need two argument: 'butterworth', 1|2|3")
            raise AssertionError
        return ButterWorthFilter(*world.config['filter'][1:])
    # from gf-cf
    elif world.config['filter'][0] == 'gfcf-linear-autoencoder':
        if len(world.config['filter']) > 2:
            print("Should need two argument: 'gfcf-linear-autoencoder', mu(float)")
            raise AssertionError
        return GFCFLinearAutoencoderFilter(*world.config['filter'][1:])
    elif world.config['filter'][0] == 'gfcf-Neighborhood-based':
        return GFCFNeighborhoodBasedFilter()
    elif world.config['filter'][0] == 'inverse':
        return GFCFNeighborhoodBasedFilter()
    elif world.config['filter'][0] == 'sigmoid-low-pass':
        return GFCFNeighborhoodBasedFilter()
    else:
        raise NotImplementedError

class EXP2(EXPS):
    def __init__(self, adj_mat):
        super(EXP2, self).__init__(adj_mat)
        self.filter = returnFilter()
     
    def train(self):
        start = time.time()
        super(EXP2, self).train()
        # filter array
        if world.config['expdevice'] == 'cpu':
            # self.s_filter = self.do_filter(self.s)
            self.s_filter = self.filter(self.s)
        else:
            # self.s_filter_cuda = torch.FloatTensor(self.do_filter(self.s)).to(world.config['expdevice'])
            self.s_filter_cuda = torch.FloatTensor(self.filter(self.s)).to(world.config['expdevice'])
        print("Make CUDA END!")
        end = time.time()
        print('training time for GF-CF', end-start)
        
    def getUsersRating(self, batch_users=None, batch_ratings=None):
        if world.config['expdevice'] == 'cpu':
            adj_mat = self.adj_mat #tolil
            batch_test = np.array(adj_mat[batch_users,:].todense())
            if world.config['filter'][0] == 'linear':
                ret = batch_test @ self.norm_adj.T @ self.norm_adj
            else:
                ret = batch_test @ self.d_mat_i @ self.vt.T @ np.diag(self.s_filter) @ self.vt @ self.d_mat_i_inv
            return ret
        else:
            batch_test = batch_ratings.to_sparse()
            # if linear filter
            if world.config['filter'][0] == 'linear':
                if world.dataset != 'amazon-book':
                    ret = batch_test @ self.linear_Filter_cuda
                else:
                    ret = (batch_test @ self.norm_adj_cuda_sparse.T @ self.norm_adj_cuda_sparse).to_dense()
            # other filters
            else:
                ret = batch_test @ self.left_mat_cuda @ torch.diag(self.s_filter_cuda) @ self.right_mat_cuda
            return ret

class EXP3(EXPS):
    def __init__(self, adj_mat) -> None:
        super(EXP3, self).__init__(adj_mat)
        self.filter = returnFilter()
    def train(self):
        start = time.time()
        super(EXP3, self).train()
        # filter array
        if world.config['expdevice'] == 'cpu':
            # self.s_filter = self.do_filter(self.s)
            self.s_filter = self.filter(self.s)
        else:
            # self.s_filter_cuda = torch.FloatTensor(self.do_filter(self.s)).to(world.config['expdevice'])
            self.s_filter_cuda = torch.FloatTensor(self.filter(self.s)).to(world.config['expdevice'])
        print("Make CUDA END!")
        end = time.time()
        print('training time for GF-CF', end-start)
        world.cprint(f"The filter is {world.config['filter']}!")
        
    def getUsersRating(self, alpha, batch_users=None, batch_ratings=None):
        if world.config['expdevice'] == 'cpu':
            adj_mat = self.adj_mat #tolil
            batch_test = np.array(adj_mat[batch_users,:].todense())
            U_2 = batch_test @ self.norm_adj.T @ self.norm_adj
            U_1 = batch_test @ self.d_mat_i @ self.vt.T @ np.diag(self.s_filter) @ self.vt @ self.d_mat_i_inv
            return alpha * U_1 + U_2
        else:
            batch_test = batch_ratings.to_sparse()
            if world.dataset != 'amazon-book':
                U_2 = batch_test @ self.linear_Filter_cuda
            else:
                U_2 = batch_test @ self.norm_adj_cuda_sparse.T @ self.norm_adj_cuda_sparse
                # U_2 = batch_test @ self.linear_Filter_cuda_sparse
            U_1 = batch_test @ self.left_mat_cuda @ torch.diag(self.s_filter_cuda) @ self.right_mat_cuda
            return alpha * U_1 + U_2

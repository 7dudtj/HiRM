'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from time import sleep
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import math
from tqdm import tqdm

CORES = multiprocessing.cpu_count() // 2

def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    # c_proc = multiprocessing.current_process()
    # print(f"Core: {CORES} current process: {c_proc.name} and PID:{c_proc.pid}")
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    # pre, recall, ndcg = [], [], []
    pre, recall, ndcg = [], [], []
    item_dict = dict()

    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        item_dict[k] = []
    for user in sorted_items:
        for i, item in enumerate(user):
            [item_dict[key].append(item) for key in item_dict.keys() if i <= key]
            # for key in item_dict.keys():
            #     # print(i, key)
            #     if i <= key:
            #         item_dict[key].append(item)
    # return {'recall':np.array(recall), 
    #         'precision':np.array(pre), 
    #         'ndcg':np.array(ndcg)}
    diversity_return = [list(set(value)) for value in item_dict.values()]
    return {'recall':np.array(recall), 
        'precision':np.array(pre), 
        'ndcg':np.array(ndcg), 
        'diversity':diversity_return} # shape = (#topks, #batch)
        

def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    adj_mat = dataset.UserItemNet.tolil()
    if(world.simple_model == 'lgn-ide'):
        lm = model.LGCN_IDE(adj_mat)
        lm.train()
    elif(world.simple_model == 'gf-cf'):
        # now check if we use vanilla one
        # if world.config['is_vanilla_gfcf'] == 1:
        lm = model.GF_CF(adj_mat)
        lm.train()
    # if the model is exp1 - we have another function for test many alpha values
    elif world.simple_model == 'exp1':
        raise NotImplementedError

    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    # Current Problem: Only use Multiprocessing at concating all results, not procedure part
    # So need to change this part
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    #     c_proc = multiprocessing.current_process()
    # print(f"Core: {CORES} current process: {c_proc.name} and PID:{c_proc.pid}")
    
    results = {'precision': np.zeros(len(world.topks)),
        'recall': np.zeros(len(world.topks)),
        'ndcg': np.zeros(len(world.topks)),
        'diversity': [[] for _ in range(len(world.topks))]}

    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            if(world.simple_model != 'none'):
                rating = lm.getUsersRating(batch_users, world.dataset)
                rating = torch.from_numpy(rating)
                rating = rating.to(world.device)
                ## Copy data to GPU and back introduces latency, just to fit the functions in LightGCN
            else:
                rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            for i in range(len(world.topks)):
                results['diversity'][i] += result['diversity'][i]
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        for i in range(len(world.topks)):
            results['diversity'][i] = len(list(set(results['diversity'][i]))) / dataset.m_items
        results['diversity'] = np.array(results['diversity'])
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            # add diversity
            # w.add_scalars(f'Test/diversity@{world.topks}',
            #             {str(world.topks[i]): results['diversity'] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
            pool.join()
        print(results)
        return results

def tensorboard_folder_name(exp_num, dataset, score_type, topk) -> str :
    """
    we don't use dataset - 
    step 0 is amazon-book
    step 1 is gowalla
    step 2 is yelp2018
    step 3 is lastfm
    """
    return str(exp_num) + "-" + str(dataset) + "/" + str(score_type) + "@" + str(topk)
    # return str(exp_num) + "/" + str(score_type) + "@" + str(topk)
    
def Test_exp1(dataset, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    adj_mat = dataset.UserItemNet.tolil()
    if world.simple_model != 'exp1':
        raise NotImplementedError
    
    lm = model.EXP1(adj_mat)
    lm.train()

    # eval mode with no dropout
    max_K = max(world.topks)
    # Current Problem: Only use Multiprocessing at concating all results, not procedure part
    # So need to change this part
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    #     c_proc = multiprocessing.current_process()
    # print(f"Core: {CORES} current process: {c_proc.name} and PID:{c_proc.pid}")
    if abs(world.config['alpha_start']-world.config['alpha_end']) < 1e-8:
        range_alpha = [world.config['alpha_start']]
    else:            
        range_alpha = np.arange(world.config['alpha_start'], world.config['alpha_end'] + 1e-8, world.config['alpha_step'])
        range_alpha = [round(i, 10) for i in range_alpha]

    if world.config['expdevice'][:4] != 'cpu':
        adj_mat = convert_sp_mat_to_sp_tensor(adj_mat).to_dense()

    for alpha in range_alpha:
        start_time = time()
        results = {'precision': np.zeros(len(world.topks)),
                'recall': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks)),
                'diversity': [[] for _ in range(len(world.topks))]}
        
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            if world.config['expdevice'][:4] != 'cpu':
                batch_ratings = adj_mat[batch_users, :].to(world.config['expdevice'])
                rating = lm.getUsersRating(alpha, batch_ratings=batch_ratings, batch_users=batch_users)
            else:
                rating = lm.getUsersRating(alpha, batch_users=batch_users)
                rating = torch.from_numpy(rating)
            # rating = rating.to(world.device)

            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            for i in range(len(world.topks)):
                results['diversity'][i] += result['diversity'][i]
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        for i in range(len(world.topks)):
            results['diversity'][i] = len(list(set(results['diversity'][i]))) / dataset.m_items
        results['diversity'] = np.array(results['diversity'])
        # results['auc'] = np.mean(auc_record)
        tb_scale = 100
        if world.tensorboard:
            for i in range(len(world.topks)):
                w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Recall", world.topks[i]), 
                        {   str(world.config['svdtype']) + "_" + 
                            str(world.config['svdvalue']) + "_" :  #+ 
                            # str(alpha) :
                            results['recall'][i]}, tb_scale * alpha) # world.dataset_step[world.dataset])
                w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Precision", world.topks[i]), 
                        {   str(world.config['svdtype']) + "_" + 
                            str(world.config['svdvalue']) + "_" : # + 
                            #str(alpha) :
                            results['precision'][i]}, tb_scale * alpha) #world.dataset_step[world.dataset])
                w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "NDCG", world.topks[i]), 
                        {   str(world.config['svdtype']) + "_" + 
                            str(world.config['svdvalue']) + "_" : # + 
                            # str(alpha) :
                            results['ndcg'][i]}, tb_scale * alpha) # world.dataset_step[world.dataset])
                w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Diversity", world.topks[i]), 
                        {   str(world.config['svdtype']) + "_" + 
                            str(world.config['svdvalue']) + "_" : #+ 
                            # str(alpha) : 
                            results['diversity'][i]}, tb_scale * alpha) # world.dataset_step[world.dataset])
        if multicore == 1:
            pool.close()
            pool.join()
        world.cprint(f"alpha: {alpha}")
        print(results)
        end_time = time()
        print(f"time consumption: {end_time-start_time}s")


def Test_exp2(dataset, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    adj_mat = dataset.UserItemNet.tolil()
    if world.simple_model != 'exp2':
        raise NotImplementedError
    
    lm = model.EXP2(adj_mat)
    lm.train()

    # eval mode with no dropout
    max_K = max(world.topks)
    # Current Problem: Only use Multiprocessing at concating all results, not procedure part
    # So need to change this part
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    #     c_proc = multiprocessing.current_process()
    # print(f"Core: {CORES} current process: {c_proc.name} and PID:{c_proc.pid}")

    if world.config['expdevice'][:4] != 'cpu':
        adj_mat = convert_sp_mat_to_sp_tensor(adj_mat).to_dense()

    start_time = time()
    results = {'precision': np.zeros(len(world.topks)),
            'recall': np.zeros(len(world.topks)),
            'ndcg': np.zeros(len(world.topks)),
            'diversity': [[] for _ in range(len(world.topks))]}
    users = list(testDict.keys())
    try:
        assert u_batch_size <= len(users) / 10
    except AssertionError:
        print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
    users_list = []
    rating_list = []
    groundTrue_list = []
    # auc_record = []
    # ratings = []
    total_batch = len(users) // u_batch_size + 1
    
    for batch_users in utils.minibatch(users, batch_size=u_batch_size):
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [testDict[u] for u in batch_users]
        if world.config['expdevice'][:4] != 'cpu':
            batch_ratings = adj_mat[batch_users, :].to(world.config['expdevice'])
            rating = lm.getUsersRating(batch_ratings=batch_ratings, batch_users=batch_users)
        else:
            rating = lm.getUsersRating(batch_users=batch_users)
            rating = torch.from_numpy(rating)
        # rating = rating.to(world.device)

        #rating = rating.cpu()
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1<<10)
        _, rating_K = torch.topk(rating, k=max_K)
        rating = rating.cpu().numpy()
        # aucs = [ 
        #         utils.AUC(rating[i],
        #                   dataset, 
        #                   test_data) for i, test_data in enumerate(groundTrue)
        #     ]
        # auc_record.extend(aucs)
        del rating
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)
    assert total_batch == len(users_list)
    X = zip(rating_list, groundTrue_list)
    if multicore == 1:
        pre_results = pool.map(test_one_batch, X)
    else:
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
    scale = float(u_batch_size/len(users))
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
        for i in range(len(world.topks)):
            results['diversity'][i] += result['diversity'][i]
    results['recall'] /= float(len(users))
    results['precision'] /= float(len(users))
    results['ndcg'] /= float(len(users))
    for i in range(len(world.topks)):
        results['diversity'][i] = len(list(set(results['diversity'][i]))) / dataset.m_items
    results['diversity'] = np.array(results['diversity'])
    # results['auc'] = np.mean(auc_record)
    
    if world.tensorboard:
        for i in range(len(world.topks)):
            w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Recall", world.topks[i]), 
                    {   str(world.config['svdtype']) + "_" + 
                        str(world.config['svdvalue']) + "_" + 
                        # str(world.config['filter']) + "_" + 
                        # str(world.config['filter_option']) :
                        str(world.config['filter']) : 
                        results['recall'][i]}, lm.filter.filter_num)
            w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Precision", world.topks[i]), 
                    {   str(world.config['svdtype']) + "_" + 
                        str(world.config['svdvalue']) + "_" + 
                        # str(world.config['filter']) + "_" + 
                        # str(world.config['filter_option']) :
                        str(world.config['filter']) :  
                        results['precision'][i]}, lm.filter.filter_num)
            w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "NDCG", world.topks[i]), 
                    {   str(world.config['svdtype']) + "_" + 
                        str(world.config['svdvalue']) + "_" + 
                        # str(world.config['filter']) + "_" + 
                        # str(world.config['filter_option']) : 
                        str(world.config['filter']) : 
                        results['ndcg'][i]}, lm.filter.filter_num)
            w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Diversity", world.topks[i]), 
                    {   str(world.config['svdtype']) + "_" + 
                        str(world.config['svdvalue']) + "_" + 
                        # str(world.config['filter']) + "_" + 
                        # str(world.config['filter_option']) :
                        str(world.config['filter']) :  
                        results['diversity'][i]}, lm.filter.filter_num)
    if multicore == 1:
        pool.close()
        pool.join()
    print(results)
    end_time = time()
    print(f"time consumption: {end_time-start_time}s")

def Test_exp3(dataset, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    adj_mat = dataset.UserItemNet.tolil()
    if world.simple_model != 'exp3':
        raise NotImplementedError
    
    lm = model.EXP3(adj_mat)
    lm.train()

    # eval mode with no dropout
    max_K = max(world.topks)
    # Current Problem: Only use Multiprocessing at concating all results, not procedure part
    # So need to change this part
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    #     c_proc = multiprocessing.current_process()
    # print(f"Core: {CORES} current process: {c_proc.name} and PID:{c_proc.pid}")
    if abs(world.config['alpha_start']-world.config['alpha_end']) < 1e-8:
        range_alpha = [world.config['alpha_start']]
    else:            
        range_alpha = np.arange(world.config['alpha_start'], world.config['alpha_end'] + 1e-8, world.config['alpha_step'])
        range_alpha = [round(i, 10) for i in range_alpha]

    if world.config['expdevice'][:4] != 'cpu':
        adj_mat = convert_sp_mat_to_sp_tensor(adj_mat).to_dense()

    for alpha in range_alpha:
        start_time = time()
        results = {'precision': np.zeros(len(world.topks)),
                'recall': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks)),
                'diversity': [[] for _ in range(len(world.topks))]}
        
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            if world.config['expdevice'][:4] != 'cpu':
                batch_ratings = adj_mat[batch_users, :].to(world.config['expdevice'])
                rating = lm.getUsersRating(alpha, batch_ratings=batch_ratings, batch_users=batch_users)
            else:
                rating = lm.getUsersRating(alpha, batch_users=batch_users)
                rating = torch.from_numpy(rating)
            # rating = rating.to(world.device)

            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            for i in range(len(world.topks)):
                results['diversity'][i] += result['diversity'][i]
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        for i in range(len(world.topks)):
            results['diversity'][i] = len(list(set(results['diversity'][i]))) / dataset.m_items
        results['diversity'] = np.array(results['diversity'])
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            tb_scale = 100
            for i in range(len(world.topks)):
                w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Recall", world.topks[i]), 
                        {   str(world.config['svdtype']) + "_" + 
                            str(world.config['svdvalue']) + "_" + 
                            # str(world.config['filter']) + "_" + 
                            str(world.config['filter']) + "_" :
                            # str(world.config['filter_option']) + "_" +
                            # str(alpha) : 
                            # results['recall'][i]}, world.dataset_step[world.dataset], alpha)
                            results['recall'][i]}, tb_scale*alpha)
                w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Precision", world.topks[i]), 
                        {   str(world.config['svdtype']) + "_" + 
                            str(world.config['svdvalue']) + "_" + 
                            # str(world.config['filter']) + "_" + 
                            str(world.config['filter']) + "_" :
                            # str(world.config['filter_option']) + "_" +
                            # str(alpha) : 
                            # results['precision'][i]}, world.dataset_step[world.dataset], alpha)
                            results['precision'][i]}, tb_scale*alpha)
                w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "NDCG", world.topks[i]), 
                        {   str(world.config['svdtype']) + "_" + 
                            str(world.config['svdvalue']) + "_" + 
                            # str(world.config['filter']) + "_" + 
                            str(world.config['filter']) + "_" :
                            # str(world.config['filter_option']) + "_" +
                            # str(alpha) : 
                            # results['ndcg'][i]}, world.dataset_step[world.dataset], alpha)
                            results['ndcg'][i]}, tb_scale*alpha)
                w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Diversity", world.topks[i]), 
                        {   str(world.config['svdtype']) + "_" + 
                            str(world.config['svdvalue']) + "_" + 
                            # str(world.config['filter']) + "_" + 
                            str(world.config['filter']) + "_" :
                            # str(world.config['filter_option']) + "_" +
                            # str(alpha) : 
                            # results['diversity'][i]}, world.dataset_step[world.dataset], alpha)
                            results['diversity'][i]}, tb_scale*alpha)
        if multicore == 1:
            pool.close()
            pool.join()
        world.cprint(f"alpha: {alpha}")
        print(results)
        end_time = time()
        print(f"time consumption: {end_time-start_time}s")


def Test_exp4(dataset, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    adj_mat = dataset.UserItemNet.tolil()

    if world.simple_model != 'exp4':
        raise NotImplementedError
    
    lm = model.EXP4(adj_mat)
    lm.train()

    # eval mode with no dropout
    max_K = max(world.topks)
    # Current Problem: Only use Multiprocessing at concating all results, not procedure part
    # So need to change this part
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    #     c_proc = multiprocessing.current_process()
    # print(f"Core: {CORES} current process: {c_proc.name} and PID:{c_proc.pid}")
    if abs(world.config['alpha_start']-world.config['alpha_end']) < 1e-8:
        range_alpha = [world.config['alpha_start']]
    else:            
        range_alpha = np.arange(world.config['alpha_start'], world.config['alpha_end'] + 1e-8, world.config['alpha_step'])
        range_alpha = [round(i, 10) for i in range_alpha]

    if world.config['expdevice'][:4] != 'cpu':
        adj_mat = convert_sp_mat_to_sp_tensor(adj_mat).to_dense()

    for alpha in range_alpha:
        start_time = time()
        results = {'precision': np.zeros(len(world.topks)),
                'recall': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks)),
                'diversity': [[] for _ in range(len(world.topks))]}
        
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            if world.config['expdevice'][:4] != 'cpu':
                batch_ratings = adj_mat[batch_users, :].to(world.config['expdevice'])
                rating = lm.getUsersRating(alpha, batch_ratings=batch_ratings, batch_users=batch_users)
            else:
                rating = lm.getUsersRating(alpha, batch_users=batch_users)
                rating = torch.from_numpy(rating)
            # rating = rating.to(world.device)

            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            for i in range(len(world.topks)):
                results['diversity'][i] += result['diversity'][i]
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        for i in range(len(world.topks)):
            results['diversity'][i] = len(list(set(results['diversity'][i]))) / dataset.m_items
        results['diversity'] = np.array(results['diversity'])
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            tb_scale = 100
            for i in range(len(world.topks)):
                w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Recall", world.topks[i]), 
                        {   str(world.config['svdtype']) + "_" + 
                            str(world.config['svdvalue']) + "_" + 
                            # str(world.config['filter']) + "_" + 
                            str(world.config['filter']) + "_" :
                            # str(world.config['filter_option']) + "_" +
                            # str(alpha) : 
                            # results['recall'][i]}, world.dataset_step[world.dataset], alpha)
                            results['recall'][i]}, tb_scale*alpha)
                w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Precision", world.topks[i]), 
                        {   str(world.config['svdtype']) + "_" + 
                            str(world.config['svdvalue']) + "_" + 
                            # str(world.config['filter']) + "_" + 
                            str(world.config['filter']) + "_" :
                            # str(world.config['filter_option']) + "_" +
                            # str(alpha) : 
                            # results['precision'][i]}, world.dataset_step[world.dataset], alpha)
                            results['precision'][i]}, tb_scale*alpha)
                w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "NDCG", world.topks[i]), 
                        {   str(world.config['svdtype']) + "_" + 
                            str(world.config['svdvalue']) + "_" + 
                            # str(world.config['filter']) + "_" + 
                            str(world.config['filter']) + "_" :
                            # str(world.config['filter_option']) + "_" +
                            # str(alpha) : 
                            # results['ndcg'][i]}, world.dataset_step[world.dataset], alpha)
                            results['ndcg'][i]}, tb_scale*alpha)
                w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Diversity", world.topks[i]), 
                        {   str(world.config['svdtype']) + "_" + 
                            str(world.config['svdvalue']) + "_" + 
                            # str(world.config['filter']) + "_" + 
                            str(world.config['filter']) + "_" :
                            # str(world.config['filter_option']) + "_" +
                            # str(alpha) : 
                            # results['diversity'][i]}, world.dataset_step[world.dataset], alpha)
                            results['diversity'][i]}, tb_scale*alpha)
        if multicore == 1:
            pool.close()
            pool.join()
        world.cprint(f"alpha: {alpha}")
        print(results)
        end_time = time()
        print(f"time consumption: {end_time-start_time}s")

def Test_HiRM(dataset, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    adj_mat = dataset.UserItemNet.tolil()

    if world.simple_model != 'HiRM':
        raise NotImplementedError
    
    lm = model.HiRM(adj_mat)
    lm.train()

    # eval mode with no dropout
    max_K = max(world.topks)
    # Current Problem: Only use Multiprocessing at concating all results, not procedure part
    # So need to change this part
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    #     c_proc = multiprocessing.current_process()

    if world.config['expdevice'][:4] != 'cpu':
        adj_mat = convert_sp_mat_to_sp_tensor(adj_mat).to_dense()

    start_time = time()
    
    
    results = {'precision': np.zeros(len(world.topks)),
            'recall': np.zeros(len(world.topks)),
            'ndcg': np.zeros(len(world.topks)),
            'diversity': [[] for _ in range(len(world.topks))]}
    
    users = list(testDict.keys())
    try:
        assert u_batch_size <= len(users) / 10
    except AssertionError:
        print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
    users_list = []
    rating_list = []
    groundTrue_list = []
    # auc_record = []
    # ratings = []
    total_batch = len(users) // u_batch_size + 1

    for batch_users in tqdm(utils.minibatch(users, batch_size=u_batch_size), total=total_batch):
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [testDict[u] for u in batch_users]
        if world.config['expdevice'][:4] != 'cpu':
            batch_ratings = adj_mat[batch_users, :].to(world.config['expdevice'])
            rating = lm.getUsersRating(batch_ratings=batch_ratings, batch_users=batch_users)
        else:
            rating = lm.getUsersRating(batch_users=batch_users)
            rating = torch.from_numpy(rating)
        # rating = rating.to(world.device)

        #rating = rating.cpu()
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1<<10)
        _, rating_K = torch.topk(rating, k=max_K)
        rating = rating.cpu().numpy()
        # aucs = [ 
        #         utils.AUC(rating[i],
        #                   dataset, 
        #                   test_data) for i, test_data in enumerate(groundTrue)
        #     ]
        # auc_record.extend(aucs)
        del rating
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)
    assert total_batch == len(users_list)
    X = zip(rating_list, groundTrue_list)
    if multicore == 1:
        pre_results = pool.map(test_one_batch, X)
    else:
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
    scale = float(u_batch_size/len(users))
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
        for i in range(len(world.topks)):
            results['diversity'][i] += result['diversity'][i]
    results['recall'] /= float(len(users))
    results['precision'] /= float(len(users))
    results['ndcg'] /= float(len(users))
    for i in range(len(world.topks)):
        results['diversity'][i] = len(list(set(results['diversity'][i]))) / dataset.m_items
    results['diversity'] = np.array(results['diversity'])
    # results['auc'] = np.mean(auc_record)
    if world.tensorboard:
        tb_scale = 100
        for i in range(len(world.topks)):
            w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Recall", world.topks[i]), 
                    {   "Recall": results['recall'][i]}, 0)
            w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Precision", world.topks[i]), 
                    {   "Precision": results['precision'][i]}, 0)
            w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "NDCG", world.topks[i]), 
                    {   "NDCG": results['ndcg'][i]}, 0)
            w.add_scalars(tensorboard_folder_name(world.simple_model, world.dataset, "Diversity", world.topks[i]), 
                    {   "Diversity:": results['diversity'][i]},0)
    if multicore == 1:
        pool.close()
        pool.join()
    print(results)
    end_time = time()
    print(f"inference time consumption: {end_time-start_time}s")

# from BSPM: https://github.com/jeongwhanchoi/BSPM/Procedure.py
def convert_sp_mat_to_sp_tensor(X) -> torch.sparse.FloatTensor: 
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
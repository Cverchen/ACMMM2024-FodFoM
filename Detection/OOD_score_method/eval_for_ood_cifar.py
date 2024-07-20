import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm, pinv
from scipy.special import softmax
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.covariance import EmpiricalCovariance
from os.path import basename, splitext
from scipy.special import logsumexp
import sys
from os.path import dirname
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import torchvision as tv
import pandas as pd
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from common.ood_tools import num_fp_at_recall, fpr_recall, auc, extract_feature_id, extract_feature_ood
from common.score_calculation import get_logits, get_and_print_results, get_odin_scores
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
import mmcv
import pickle
import argparse

def parse_option():
    parser = argparse.ArgumentParser('arguXQment for training')
    parser.add_argument('--model_resnet',type=str, default='resnet18',help='Supcon_resnet architecture')
    parser.add_argument('--resnet_head',type=str, default='mlp_cifar100')
    parser.add_argument('--dataset',type=str, default='cifar100',help='in_dataset')
    parser.add_argument('--gpu', type=str, default='5')
    parser.add_argument('--save_path', type=str, default='feature/cifar100')
    parser.add_argument('--end_path', type=str)
    parser.add_argument('--fc', type=str, default='fc/cifar100/fc.pkl')
    parser.add_argument('--n_classes', type=int, default=100)
    args = parser.parse_args()
    return args
   
def plot_distribution(id, ood, in_dataset, out_dataset):
    sns.set(style="white", palette="muted")
    fig, ax = plt.subplots(figsize=(3, 4))
    sns.kdeplot(data=id, color='crimson', label=in_dataset, fill=True, ax=ax)
    sns.kdeplot(data=ood, color='limegreen', label=out_dataset, fill=True, ax=ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('feature/pic',f"{in_dataset}vs{out_dataset}.png"), bbox_inches='tight')


def main():
    maha = False
    args = parse_option()
    num_classes = args.n_classes
    w, b = mmcv.load(args.fc)
    print(f'{w.shape}, {b.shape}')
    print('load features')
    id_train_features = np.load(os.path.join(args.save_path, args.dataset+'_train.npy'))
    all_id_test_features = np.load(os.path.join(args.save_path, args.dataset+'_test.npy'))

    OOD_data_list = ["Textures", "SVHN", "iSUN", "LSUN-C", "LSUN-R", "Places365"]
    feature_oods = []
    feature_oods_forsim = []
    for name in OOD_data_list:
        feature_oods.append(np.load(os.path.join(args.save_path,name+'.npy')))
        # feature_oods_forsim.append(np.load(os.path.join(args.save_path, name+'_forsim.npy')))
    df = pd.DataFrame(columns=['method','oodset', 'auroc', 'fpr'])
    dfs = []
    recall = 0.95
    
    # #######################Mahala score#########################
    if maha is True:
        print('Mahalanobis >>>>>>>>>>>>>>>>')
        result = []
        print('computing classwise mean feature...')  
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(num_classes)):
            fs = id_train_features[i]
            fs  = np.array(fs)
            _m = fs.mean(axis=0)# 每维度特征均值
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)
        print('computing precision matrix...')
        ec = EmpiricalCovariance(assume_centered=True) # 协方差
        ec.fit(np.array(train_feat_centered).astype(np.float64))
        print('go to gpu>>>>>>>>>')
        mean = torch.from_numpy(np.array(train_means)).cuda().float()
        prec = torch.from_numpy(ec.precision_).cuda().float() # 协方差矩阵的逆矩阵
        
        mu = mean
        method = 'Mahalanobis'
        score_id = -np.array([(((f-mu)@prec)*(f-mu)).sum(axis=-1).min().cpu().item() for f in tqdm(torch.from_numpy(all_id_test_features).cuda().float())])
        # Score_ood = []
        print("ood ......")
        for i in range(len(OOD_data_list)):  
            feature_ood = feature_oods[i]
            feature_ood = np.array(feature_ood)
            score_ood = -np.array([(((f-mu)@prec)*(f-mu)).sum(axis=-1).min().cpu().item() for f in tqdm(torch.from_numpy(feature_ood).cuda().float())])
            # plot_distribution(id_scores=score_id[:5000]/1000, ood_scores=score_ood[:5000]/1000, in_dataset=args.dataset, out_dataset=OOD_data_list[i])
            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(dict(method=method, oodset=OOD_data_list[i], auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}:{OOD_data_list[i]} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')
    if maha is False:
        print('computing logits')
        logit_id_train = []
        for i in range(num_classes):
            logit_id_train.append(id_train_features[i] @ w.T + b)
        logit_id_test = all_id_test_features @ w.T + b

        logit_oods = []
        for i in range(len(OOD_data_list)):
            logit_oods.append(feature_oods[i] @ w.T + b)
        
        print('computing softmax...')
        softmax_id_train = []
        for i in range(num_classes):
            softmax_id_train.append(softmax(logit_id_train[i], axis=-1))
        softmax_id_test = softmax(logit_id_test, axis=-1)
        
        softmax_oods = []
        for i in range(len(OOD_data_list)):
            softmax_oods.append(softmax(logit_oods[i], axis=-1))
        

        ##########MSP#########
        print('>>>>>>>>>>MSP')
        method = 'MSP'
        result = []
        score_id = softmax_id_test[:,:num_classes].max(axis=-1)/softmax_id_test[:, num_classes]
        for i in range(len(OOD_data_list)):
            score_ood = softmax_oods[i][:,:num_classes].max(axis=-1)/softmax_oods[i][:, num_classes]
            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(dict(method=method, oodset=OOD_data_list[i], auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}:{OOD_data_list[i]} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

        ################Maxlogit###########
        print('>>>>>>>>>>Maxlogit')
        method = 'Maxlogit'
        result = []
        score_id = logit_id_test[:,:num_classes].max(axis=-1)
        for i in range(len(OOD_data_list)):
            score_ood = logit_oods[i][:,:num_classes].max(axis=-1)
            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(dict(method=method, oodset=OOD_data_list[i], auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}:{OOD_data_list[i]} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')
        
        ###########Energy##################
        print('>>>>>>>>>>Energy')
        method = 'Energy'
        result = []
        score_id = logsumexp(logit_id_test[:,:args.n_classes], axis=-1)#  / np.exp(logit_id_test[:, args.n_classes])
        for i in range(len(OOD_data_list)):
            score_ood = logsumexp(logit_oods[i][:,:args.n_classes], axis=-1) #/ np.exp(logit_oods[i][:, args.n_classes])
            # plot_distribution(score_id, score_ood, in_dataset=args.dataset, out_dataset=OOD_data_list[i])
            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
            result.append(dict(method=method, oodset=OOD_data_list[i], auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}:{OOD_data_list[i]} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')
        
        #############Energy+React######
        Meanauroc = []
        Meanfpr = []
        CClip = []
        Clip = 0.9
        while Clip <= 0.91:
            print('>>>>>>>>>>Energy+React')
            method = 'Energy+React'
            result = []
            feature_id_train = np.concatenate(id_train_features, axis=0)
            clip = np.quantile(feature_id_train, Clip) # 0.85 ours
            logit_id_test_clip = np.clip(all_id_test_features, a_min=None, a_max=clip) @ w.T + b
            score_id = logsumexp(logit_id_test_clip[:,:args.n_classes], axis=-1) 
            for i in range(len(OOD_data_list)):
                logit_ood_clip  = np.clip(feature_oods[i], a_min=None, a_max=clip) @ w.T + b
                score_ood = logsumexp(logit_ood_clip[:,:args.n_classes], axis=-1) 
                score_ood = score_ood
                # plot_distribution(score_id, score_ood, in_dataset=args.dataset, out_dataset=OOD_data_list[i])
                auc_ood = auc(score_id, score_ood)[0]
                fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
                result.append(dict(method=method, oodset=OOD_data_list[i], auroc=auc_ood, fpr=fpr_ood))
                print(f'{method}:{OOD_data_list[i]} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
            df = pd.DataFrame(result)
            dfs.append(df)
            print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')
            CClip.append(Clip)
            Meanauroc.append(df.auroc.mean())
            Meanfpr.append(df.fpr.mean())
            Clip += 0.01
        for i in range(len(CClip)):
            print(str(CClip[i])+":"+str(Meanauroc[i])+' '+str(Meanfpr[i]))


        # print('computing logits')
        # logit_id_train = []
        # for i in range(num_classes):
        #     logit_id_train.append(id_train_features[i] @ w.T + b)
        # logit_id_test = all_id_test_features @ w.T + b

        # logit_oods = []
        # for i in range(len(OOD_data_list)):
        #     logit_oods.append(feature_oods[i] @ w.T + b)
        
        # u = -np.matmul(pinv(w), b)
        # feature_id_val = all_id_test_features
        # method = 'ViM'
        # print(f'\n{method}')
        # result = []
        # DIM = 512 if feature_id_val.shape[-1] > 512 else 384
        # print(f'{DIM=}')
        # print('computing principal space...')
        # ec = EmpiricalCovariance(assume_centered=True)
        # feature_id_train = np.concatenate(id_train_features, axis=0)
        # ec.fit(feature_id_train - u)
        # eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        # NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        # print('computing alpha...')
        # logit_id_train = np.concatenate(logit_id_train, axis=0)
        # vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
        # alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
        # print(f'{alpha=:.4f}')
        # logit_id_val = logit_id_test
        # vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
        # energy_id_val = logsumexp(logit_id_val, axis=-1)
        # score_id = -vlogit_id_val + energy_id_val

        # for i in range(len(OOD_data_list)):
        #     energy_ood = logsumexp(logit_oods[i], axis=-1)
        #     vlogit_ood = norm(np.matmul(feature_oods[i] - u, NS), axis=-1) * alpha
        #     score_ood = -vlogit_ood + energy_ood
        #     auc_ood = auc(score_id, score_ood)[0]
        #     fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        #     name = OOD_data_list[i]
        #     result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        #     print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        # df = pd.DataFrame(result)
        # dfs.append(df)
        # print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

        
if __name__ == '__main__':
    main()







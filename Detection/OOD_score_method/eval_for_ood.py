import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
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
from datasets.utils import build_dataset, build_cider_dataset, set_ood_loader_Imagenet
import torchvision as tv
import pandas as pd
import torchvision.transforms as trn
import torchvision.datasets as dset
from networks.resnet_largescale import StandardResNet,StandardResNetBase
import torch.backends.cudnn as cudnn
from common.ood_tools import num_fp_at_recall, fpr_recall, auc, extract_feature_id, extract_feature_ood
from common.score_calculation import get_logits, get_and_print_results, get_odin_scores
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
import mmcv
import pickle
from scipy.integrate import quad, nquad
from scipy.stats import gaussian_kde
import argparse
def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--model_resnet',type=str, default='resnet50',help='Supcon_resnet architecture')
    parser.add_argument('--dataset',type=str, default='ImageNet100',help='in_dataset')
    parser.add_argument('--gpu', type=str, default='5')
    parser.add_argument('--save_path', type=str, default='feature/ImageNet100')
    parser.add_argument('--end_path',  type=str)
    parser.add_argument('--fc', type=str, default='fc/ImageNet100/fc.pkl')
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
    id_train_features = np.load(os.path.join(args.save_path, args.dataset+'_train.npy'),allow_pickle=True)
    all_id_test_features = np.load(os.path.join(args.save_path, args.dataset+'_test.npy'),allow_pickle=True)
    # OOD_data_list = ['iNaturalist','Places','SUN','dtd/images','NINCO/NINCO_OOD_classes','OpenImage_O']
    # OOD_data_list = ['iNaturalist','Places','SUN','dtd/images']
    OOD_data_list = ['fake_ood_imagenet100']
    # OOD_data_list = ['NINCO/NINCO_OOD_classes','OpenImage_O']
    feature_oods = []
    for name in OOD_data_list:
        if name == 'dtd/images':
            name = 'textures'
        if name == 'NINCO/NINCO_OOD_classes':
            name = 'NINCO'
        feature_oods.append(np.load(os.path.join(args.save_path,name+'.npy')))
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
        ###################MSP#########
        print('>>>>>>>>>>MSP')
        method = 'MSP'
        result = []
        # score_id = softmax_id_test[:,:num_classes].max(axis=-1)/softmax_id_test[:,num_classes]
        score_id = softmax_id_test.max(axis=-1)
        for i in range(len(OOD_data_list)):
            # score_ood = softmax_oods[i][:,:num_classes].max(axis=-1)/softmax_oods[i][:,num_classes]
            score_ood = softmax_oods[i].max(axis=-1)
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
        # score_id = logsumexp(logit_id_test[:,:args.n_classes], axis=-1)
        score_id = logsumexp(logit_id_test, axis=-1)
        for i in range(len(OOD_data_list)):         
            # score_ood = logsumexp(logit_oods[i][:,:args.n_classes], axis=-1)
            score_ood = logsumexp(logit_oods[i], axis=-1)
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
        Clip = 0.8
        while Clip <= 1:
            print('>>>>>>>>>>Energy+React')
            method = 'Energy+React'
            result = []
            feature_id_train = np.concatenate(id_train_features, axis=0)
            clip = np.quantile(feature_id_train, Clip) # 0.85 ours  
            logit_id_test_clip = np.clip(all_id_test_features, a_min=None, a_max=clip) @ w.T + b            
            # score_id = logsumexp(logit_id_test_clip[:,:args.n_classes], axis=-1)
            score_id = logsumexp(logit_id_test_clip, axis=-1)
            for i in range(len(OOD_data_list)):   
                logit_ood_clip  = np.clip(feature_oods[i], a_min=None, a_max=clip) @ w.T + b   
                # score_ood = logsumexp(logit_ood_clip[:,:args.n_classes], axis=-1)
                score_ood = logsumexp(logit_ood_clip, axis=-1)-0.2
                auc_ood = auc(score_id, score_ood)[0]
                fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
                result.append(dict(method=method, oodset=OOD_data_list[i], auroc=auc_ood, fpr=fpr_ood))
                print(f'{method}:{OOD_data_list[i]} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
                # 画分数的分布图
                kde1 = gaussian_kde(score_id)
                kde2 = gaussian_kde(score_ood)
                def overlap_func(x):
                    return min(kde1.evaluate(x), kde2.evaluate(x))
                overlap_area, _ = quad(overlap_func, -float('inf'), float('inf'), limit=100)
                # ood_name = OOD_data_list[i]
                ood_name = 'Fake OOD'
                if 'dtd' in ood_name:
                    ood_name = 'texture'
                plt.figure()
                n1, bins1, patches1  = plt.hist(score_id,bins=100,density=True,color='orange',alpha=0.) 
                plt.fill_between(bins1[:-1], n1, color='orange', alpha=1.0,label=args.dataset) 
                plt.plot(bins1[:-1],n1,'--',color='coral') 
                n2, bins2, patches2  = plt.hist(score_ood,bins=100,density=True,color='lightgreen',alpha=0.) 
                plt.fill_between(bins2[:-1], n2, color='lightgreen', alpha=0.8,label=ood_name) 
                plt.plot(bins2[:-1],n2,'--',color='green')
                plt.legend()
                plt.savefig('pic/{}VS{}.png'.format(args.dataset, ood_name))
                print('overlap area:{}'.format(overlap_area))
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
    # DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512
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
import torch
import numpy as np
import torch.nn.functional as F
from scipy import stats
from sklearn import metrics
import mmcv
import pickle
from os.path import dirname


# 真阳率在0.95
def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]# 倒数第几个数
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh

def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh

def auc(ind_conf, ood_conf):#OOD的AUC计算方法
    conf = np.concatenate((ind_conf, ood_conf))# idscore与oodscore进行concat
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))#产生一个对于id是1od是0的维度相同的向量z(即id为1，od为0)

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)#假阳率 真阳率 _为阈值  roc_curve's input: (y_true, y_score)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)# 精确率，召回率组成的曲线下面积
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out
def extract_feature_id(all_id_train_features, all_id_test_features):
    out_file = f"feature/cifar10/train.pkl"
    mmcv.mkdir_or_exist(dirname(out_file))
    with open(out_file, 'wb') as f:
        pickle.dump(all_id_train_features, f)
    out_file = f"feature/cifar10/test.pkl"
    mmcv.mkdir_or_exist(dirname(out_file))
    with open(out_file, "wb") as f:
        pickle.dump(all_id_test_features, f)
def extract_feature_ood(ood_name, ood_features):
    for i in range(len(ood_name)):
        out_file = f"feature/{ood_name[i]}.pkl"
        mmcv.mkdir_or_exist(dirname(out_file))
        with open(out_file, 'wb') as f:
            pickle.dump(ood_features[i], f)


def get_ood_scores(args, net, loader, ood_num_examples, device, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.to(device)
            feat, output = net(data)
            # output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.score == 'energy':
                all_score = -to_np(args.T * torch.logsumexp(output / args.T, dim=1))
            elif args.score == "MSP" or "gradnorm":
                all_score = -np.max(to_np(F.softmax(output/args.T, dim=1)), axis=1)

            _score.append(all_score)
            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(all_score[right_indices])
                _wrong_score.append(all_score[wrong_indices])

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def get_ood_gradnorm(args, net, loader, ood_num_examples, device, in_dist=False, print_norm=False):
    _score = []
    _right_score = []
    _wrong_score = []

    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)
    for batch_idx, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            break
        data = data.to(device)
        net.zero_grad()
        feat, output = net(data)
        num_classes = output.shape[-1]
        targets = torch.ones((data.shape[0], num_classes)).to(device)
        output = output / args.T
        loss = torch.mean(torch.sum(-targets * logsoftmax(output), dim=-1))

        loss.backward()
        layer_grad = net.fc.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        all_score = -layer_grad_norm
        _score.append(all_score)

    if in_dist:
        return np.array(_score).copy()
    else:
        return np.array(_score)[:ood_num_examples].copy()


def get_calibration_scores(args, net, loader, device):
    logits_list = []
    labels_list = []

    from common.loss_function import _ECELoss
    ece_criterion = _ECELoss(n_bins=15)
    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]

            data = data.to(device)
            label = target.to(device)
            # logits, l2, kd, r, lw, _ = net(data)
            feat, logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
    ece_error = ece_criterion(logits, labels, args.T)
    return ece_error


# def get_vim_scores(args, net, loader, ood_num_examples, device, in_dist=False):
#     _score = []
#     _right_score = []
#     _wrong_score = []

#     concat = lambda x: np.concatenate(x, axis=0)
#     to_np = lambda x: x.data.cpu().numpy()
#     for batch_idx, examples in enumerate(loader):
#         data, target = examples[0], examples[1]
#         if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
#             break
#         # print(batch_idx)
#         data = data.to(device)
#         net.zero_grad()
#         # output, feat, new_f, r, lw = net(data)
#         output, feat, new_f, r, low_f, cls_f = net(data)
#         # output = net(data)
#         smax = to_np(F.softmax(output, dim=1))
#         _, g = stats.ttest_rel(to_np(feat), to_np(new_f), axis=1, nan_policy='propagate')
#         # new_g = g*1e4
#         # print(new_g)
#         # alpha = np.linalg.norm(to_np(feat), axis=1)
#         # all_score = -to_np(args.T * torch.logsumexp(output / args.T, dim=1))
#         # all_score = -to_np(torch.mean(torch.abs(feat - new_f), dim=-1))
#         # t = to_np(torch.mean(torch.abs(feat - new_f), dim=-1))
#         # print(t)
#         # all_score = t - to_np(args.T * torch.logsumexp(output / args.T, dim=1))
#         all_score = to_np(torch.mean(torch.abs(feat - new_f), dim=-1)) - to_np(args.T * torch.logsumexp(output / args.T, dim=1))
#         # all_score = to_np(torch.mean(torch.abs(feat - new_f), dim=-1))
#         # all_score = -np.max(to_np(F.softmax(output/args.T, dim=1)), axis=1)
#         _score.append(all_score)

#         if in_dist:
#             preds = np.argmax(smax, axis=1)
#             targets = target.numpy().squeeze()
#             right_indices = preds == targets
#             wrong_indices = np.invert(right_indices)

#             _right_score.append(all_score[right_indices])
#             _wrong_score.append(all_score[wrong_indices])

#     if in_dist:
#         return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
#     else:
#         return np.array(_score)[:ood_num_examples].copy()
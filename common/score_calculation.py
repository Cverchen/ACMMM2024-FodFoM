from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import os
import torchvision.transforms as transforms
import numpy as np
import sklearn.metrics as sk
from scipy import misc

to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out
def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))
    if np.array_equal(classes, [1]):
        return thresholds[cutoff]  # return threshold

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr, threshould = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr, threshould

def print_measures(auroc, aupr, fpr, ood, method, recall_level=0.95):
    print('\t\t\t' + ood+'_'+method)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))

def get_and_print_results(out_score, in_score, ood, method):
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(out_score, in_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    
    print_measures(auroc, aupr, fpr, ood, method)
    return auroc, aupr, fpr


def get_odin_scores(loader, dataset, model, method, T, noise): # ODIN分数
    ## get logits
    bceloss = nn.BCEWithLogitsLoss(reduction="none")
    celoss = nn.CrossEntropyLoss()
    if dataset == 'cifar10' or dataset == 'cifar100':
        for i, (images, _, idx) in enumerate(loader):
            images = Variable(images.cuda(), requires_grad=True)
            # nnOutputs = clsfier(model(images))
            nnOutputs = model.classifier(model.encoder(images))

            # using temperature scaling
            preds = torch.sigmoid(nnOutputs / T)
            # preds = torch.softmax(nnOutputs / T, dim=-1)

            labels = torch.ones(preds.shape).cuda() * (preds >= 0.5)
            labels = Variable(labels.float())

            # input pre-processing
            loss = bceloss(nnOutputs, labels)
            # loss = celoss(nnOutputs, labels)

            if method == 'max':
                idx = torch.max(preds, dim=1)[1].unsqueeze(-1)
                loss = torch.mean(torch.gather(loss, 1, idx))
            elif method == 'sum':
                loss = torch.mean(torch.sum(loss, dim=1))

            loss.backward()
            # calculating the perturbation
            gradient = torch.ge(images.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.229))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.224))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.225))
            tempInputs = torch.add(images.data, gradient, alpha=-noise)

            with torch.no_grad():
                # nnOutputs = clsfier(model(Variable(tempInputs)))
                nnOutputs = model.classifier(model.encoder(Variable(tempInputs)))

                ## compute odin score
                # outputs = torch.sigmoid(nnOutputs / T)
                outputs = torch.softmax(nnOutputs/T, dim=-1)

                if method == "max":
                    score = np.max(to_np(outputs), axis=1)
                elif method == "sum":
                    score = np.sum(to_np(outputs), axis=1)
                if i == 0:
                    scores = score
                else:
                    scores = np.concatenate((scores, score),axis=0)
    else:
        for i, (images, _) in enumerate(loader):
            images = Variable(images.cuda(), requires_grad=True)
            # nnOutputs = clsfier(model(images))
            nnOutputs = model.classifier(model.encoder(images))

            # using temperature scaling
            preds = torch.sigmoid(nnOutputs / T)
            # preds = torch.softmax(nnOutputs/T, dim=-1)

            labels = torch.ones(preds.shape).cuda() * (preds >= 0.5)
            labels = Variable(labels.float())

            # input pre-processing
            loss = bceloss(nnOutputs, labels)
            # loss = celoss(nnOutputs, labels)

            if method == 'max':
                idx = torch.max(preds, dim=1)[1].unsqueeze(-1)
                loss = torch.mean(torch.gather(loss, 1, idx))
            elif method == 'sum':
                loss = torch.mean(torch.sum(loss, dim=1))

            loss.backward()
            # calculating the perturbation
            gradient = torch.ge(images.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.229))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.224))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.225))
            tempInputs = torch.add(images.data, gradient, alpha=-noise)

            with torch.no_grad():
                # nnOutputs = clsfier(model(Variable(tempInputs)))
                nnOutput = model.classifier(model.encoder(Variable(tempInputs)))

                ## compute odin score
                outputs = torch.sigmoid(nnOutputs / T)
                # outputs = torch.softmax(nnOutput/T, dim=-1)

                if method == "max":
                    score = np.max(to_np(outputs), axis=1)
                elif method == "sum":
                    score = np.sum(to_np(outputs), axis=1)
                if i == 0:
                    scores = score
                else:
                    scores = np.concatenate((scores, score),axis=0)
    return scores

def get_logits(loader, model, args, k=20, is_in=False, name=None):
    print(args.save_path+ '/' + name + ".npy", os.path.exists(args.save_path+ '/' + name + ".npy"))
    if not (os.path.exists(args.save_path + name + ".npy")):
        logits_np = np.empty([0, args.n_classes])
        with torch.no_grad():
            if is_in:            
                for i, (images, label, index) in enumerate(loader):
                    images = Variable(images.cuda())
                    nnOutputs = model.encoder(images)
                    if args.model_resnet == 'resnet50' or args.model_resnet == 'resnet101':
                        feat_batch = model.reducelinear(feat_batch)
                    # print(nnOutputs.shape)
                    nnOutputs = model.classifier(nnOutputs)
                    nnOutputs_np = to_np(nnOutputs.squeeze())
                    logits_np = np.vstack((logits_np, nnOutputs_np))
            else:
                for i, (images, labels) in enumerate(loader):
                    images = Variable(images.cuda())
                    nnOutputs = model.encoder(images)
                    if args.model_resnet == 'resnet50' or args.model_resnet == 'resnet101':
                        feat_batch = model.reducelinear(feat_batch)
                    # print(nnOutputs.shape)
                    nnOutputs = model.classifier(nnOutputs)
                    nnOutputs_np = to_np(nnOutputs.squeeze())
                    logits_np = np.vstack((logits_np, nnOutputs_np))
        os.makedirs(args.save_path, exist_ok = True)
        np.save(args.save_path + '/'+ name, logits_np)
    else:
        logits_np = np.load(args.save_path +'/'+ name + ".npy")
    ## Compute the Score
    logits = torch.from_numpy(logits_np).cuda()
    outputs = torch.sigmoid(logits)
    if args.ood == "logit":
        if args.method == "max": scores = np.max(logits_np, axis=1)
        if args.method == "sum": scores = np.sum(logits_np, axis=1)
    elif args.ood == "energy":
        E_f = torch.log(1+torch.exp(logits))
        if args.method == "max": scores = to_np(torch.max(E_f, dim=1)[0])
        if args.method == "sum": scores = to_np(torch.sum(E_f, dim=1))
        if args.method == "topk":
            scores = to_np(torch.sum(torch.topk(E_f, k=k, dim=1)[0], dim=1))
    elif args.ood == "prob":
        if args.method == "max": scores = np.max(to_np(outputs), axis=1)
        if args.method == "sum": scores = np.sum(to_np(outputs),axis=1)
    elif args.ood == "msp":
        outputs = F.softmax(logits, dim=1)
        scores = np.max(to_np(outputs), axis=1)
    else:
        scores = logits_np
    return scores


def get_ood_scores_odin(loader, net, bs, ood_num_examples, T, noise, device, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        if batch_idx >= ood_num_examples // bs and in_dist is False:
            break
        data = data.to(device)
        data = Variable(data, requires_grad = True)

        feat, output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODIN(data, output,net, T, noise, device)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def ODIN(inputs, outputs, model, temper, noiseMagnitude1, device):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    gradient[:, 0] = (gradient[:, 0])/(63.0/255.0)
    gradient[:, 1] = (gradient[:, 1])/(62.1/255.0)
    gradient[:, 2] = (gradient[:, 2])/(66.7/255.0)
    #gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0/255.0))
    #gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1/255.0))
    #gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7/255.0))

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    feat, outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs
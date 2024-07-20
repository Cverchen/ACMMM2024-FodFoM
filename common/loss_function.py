import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from scipy import spatial
import tqdm
import time

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, t=1.0):
        softmaxes = F.softmax(logits/t, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class LogitNormLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7 # shape of logit:[128, 10]
        logit_norm = torch.div(x, norms) / self.t # logits/norm(logits)/t 温度系数
        return F.cross_entropy(logit_norm, target)
# CIDERloss myself replement
class CIDERLoss(nn.Module):
    def __init__(self, device, num_classes, t=0.1, alpha=1, lamda=2):
        super(CIDERLoss, self).__init__()
        self.device = device
        self.t = t
        self.num_classes = num_classes
        self.alpha = alpha
        self.lamda = lamda
    
    def forward(self, z, target, mu):# mu为各个类的原型: [10, 128] z:[128,128]  
        loss_comp = 0
        for i in range(z.shape[0]):
            loss_comp += torch.dot(z[i], mu[target[i]])/self.t-torch.logsumexp(((torch.mv(mu, z[i])/self.t).unsqueeze(0)),dim=1)
        loss_comp = -loss_comp*(1/z.shape[0])
        loss_dis = 0
        for i in range(self.num_classes):
            mu_update = torch.ones(self.num_classes-1, 128).to(device=self.device) # [9,128]
            index = 0
            for j in range(self.num_classes):
                if i!=j:
                    mu_update[index] = mu[j]
                    index += 1
            loss_dis += math.log(1/(self.num_classes - 1)) + torch.logsumexp((torch.mv(mu_update, mu[i])/self.t).unsqueeze(0),dim=1)
        loss_dis = (1/self.num_classes)*loss_dis
        loss = self.alpha*loss_dis + self.lamda*loss_comp
        return loss
    
# Disloss+Comploss
class DisLoss(nn.Module):
    ''' CIDER's dispersion loss with EMA prototypes'''
    def __init__(self, args, model, loader, temperature=0.1, base_temperature=0.1):
        super(DisLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.register_buffer('prototype', torch.zeros(self.args.n_cls, self.args.feat_dim))
        self.model = model
        self.loader = loader
        self.device = 'cuda:{}'.format(int(self.args.gpu))
        self.init_class_prototypes()

    def forward(self, features, labels):
        prototypes = self.prototypes
        num_cls = self.args.n_cls
        device = 'cuda:{}'.format(int(self.args.gpu))
        # 更新各个prototypes
        for j in range(len(features)):
            prototypes[labels[j].item()] = F.normalize(prototypes[labels[j].item()] * self.args.proto_m + features[j]*(1-self.args.proto_m), dim=0)
        self.prototypes = prototypes.detach()
        labels = torch.arange(0, num_cls).to(device)
        labels = labels.contiguous().view(-1,1)
        labels = labels.contiguous().view(-1,1)

        mask = (1 - torch.eq(labels, labels.T).float()).to(device)

        logits = torch.div(torch.matmul(prototypes, prototypes.T),
                           self.temperature) # mu*muT/t
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_cls).view(-1,1).to(device),
            0
        )
        mask = mask * logits_mask
        mean_prob_neg = torch.log((mask*torch.exp(logits)).sum(1)/ mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]

        loss = self.temperature/self.base_temperature * mean_prob_neg.mean()
        return loss

    def init_class_prototypes(self):
        """initialize class prototypes"""
        self.model.eval()
        start = time.time()
        prototype_counts = [0]*self.args.n_cls
        with torch.no_grad():
            prototypes = torch.zeros(self.args.n_cls, self.args.feat_dim).to(self.device)
            for input, target, index in self.loader:
                input, target = input.to(self.device), target.to(self.device)
                features = self.model(input)
                for j, features in enumerate(features):
                    prototypes[target[j].item()] += features
                    prototype_counts[target[j].item()] += 1
            for cls in range(self.args.n_cls):
                prototypes[cls] /= prototype_counts[cls]
            # measure elapsed time
            duration = time.time() - start
            print(f'Time to initialize prototypes:{duration:.3f}')
            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = prototypes
        
class CompLoss(nn.Module):
    '''Compactness Loss with class-conditional prototypes'''
    def __init__(self, args, temperature=0.07, base_temperature=0.07):
        super(CompLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = 'cuda:{}'.format(int(args.gpu))

    def forward(self, features, prototypes, labels):
        prototypes = F.normalize(prototypes,dim=1)
        proxy_labels = torch.arange(0, self.args.n_cls).to(self.device)
        labels = labels.contiguous().view(-1,1)
        mask = torch.eq(labels, proxy_labels.T).float().to(self.device) # (bs, cls)

        # compute logits
        feat_dot_prototype = torch.div(
            torch.matmul(features, prototypes.T),
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1)

        # loss
        loss = -(self.temperature/self.base_temperature)*mean_log_prob_pos.mean()

        return loss
        



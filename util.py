from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn

class Diffusion_backgound_Dataset(Dataset):
    def __init__(self, filename, transform):
        self.filename = filename
        self.labels = []
        self.image = []
        self.transform = transform
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.read().splitlines():
                x = line[:line.find(',')]
                label = line[line.find(','):].split(',')[1]
                label = int(label)
                self.labels.append(label)
                self.image.append(x)
    
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        image = Image.open(self.image[idx]).convert('RGB')
        image = self.transform(image)
        label = self.labels
        return image, label[idx]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class MergedDataset(data.Dataset):
    def __init__(self, dataset1, dataset2, dataset3):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3

        # 调整 dataset2 的标签范围
        self.dataset2.class_to_idx = {class_name: 100 for class_name in dataset2.classes}
        
        # 调整 dataset2 中样本的标签
        self.dataset2.samples = [(sample_path, 100) for sample_path, _ in dataset2.samples]

        # 调整 dataset3 的标签范围
        self.dataset3.class_to_idx = {class_name: 100 for class_name in dataset3.classes}
        
        # 调整 dataset3 中样本的标签
        self.dataset3.samples = [(sample_path, 100) for sample_path, _ in dataset3.samples]

        # 调整 dataset2 的标签范围
        # num_classes_dataset1 = len(dataset1.classes)
        # self.dataset2.class_to_idx = {class_name: idx + num_classes_dataset1 for class_name, idx in dataset2.class_to_idx.items()}
        
        # 调整 dataset2 中样本的标签
        # self.dataset2.samples = [(sample_path, target + num_classes_dataset1) for sample_path, target in dataset2.samples]

        
        self.samples = dataset1.samples + self.dataset2.samples + self.dataset3.samples
        self.targets = dataset1.targets + self.dataset2.targets + self.dataset3.targets
        self.loader = dataset1.loader
        self.transform = dataset1.transform
        self.target_transform = None
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)
    

class MergedTwoDataset(data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # 调整 dataset2 的标签范围
        self.dataset2.class_to_idx = {class_name: 100 for class_name in dataset2.classes}
        
        # 调整 dataset2 中样本的标签
        self.dataset2.samples = [(sample_path, 100) for sample_path, _ in dataset2.samples]


        
        self.samples = dataset1.samples + self.dataset2.samples
        self.targets = dataset1.targets + self.dataset2.targets
        self.loader = dataset1.loader
        self.transform = dataset1.transform
        self.target_transform = None
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)



def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD([{'params': model.parameters()}],
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

def set_optimizer_oe(opt, model, logistic):
    optimizer = optim.SGD([{'params': list(model.parameters()) + list(logistic.parameters())}],
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def save_model_both(model, regression, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'regression': regression.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class DebrisDataset(Dataset):
    def __init__(self, filename, transform):
        self.filename = filename
        self.labels = []
        self.image = []
        self.transform = transform
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.read().splitlines():
                x = line[:line.find(',')]
                label = line[line.find(','):].split(',')[1]
                label = int(label)
                self.labels.append(label)
                self.image.append(x)
    
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        image = Image.open(self.image[idx]).convert('RGB')
        image = self.transform(image)
        label = self.labels
        return self.image[idx], image, label[idx]
    def get_path(self, idx):
        return self.image[idx]
    

def set_loader_liantong(args, eval=False, batch_size=None, image_size=224):
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    # filename = 'Dataset/dataset.txt'
    filename_train = args.filename_train
    filename_test = args.filename_test
    Debris_train_dataset = DebrisDataset(filename=filename_train, transform=train_transform)
    Debris_test_dataset = DebrisDataset(filename=filename_test, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(
        Debris_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        Debris_test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)
    return Debris_test_dataset, train_loader, test_loader



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
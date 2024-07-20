from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
import sys
import argparse
from torchvision.datasets import CIFAR10,CIFAR100
import time
import math
import torch
from torch.utils.data import ConcatDataset
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import numpy as np
import torch.nn.functional as F
from util import adjust_learning_rate, warmup_learning_rate, AverageMeter
from util import set_optimizer, save_model, MergedDataset, Diffusion_backgound_Dataset, SupConLoss, TwoCropTransform
from networks.resnet_largescale import StandardResNet,StandardResNetBase, SupStandardResNet, SupConResNetLargeScale
from networks.resnet_big import StandardResnet_CIFAR, SupStandardResnet_CIFAR
import wandb

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--expriment_name', type=str, help='expriment_name')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=250,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--test_batch_size', type=int, default=256, help='test_batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='300, 400',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # model dataset
    parser.add_argument('--model', type=str, default='resnet34')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'ImageNet100','path','ImageNet100_baseline'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='vision-text', help='choose method')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()
    # 设置种子
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    
    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = 'datasets'
    opt.model_path = './save/{}_models'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_expname{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.expriment_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate


    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

import torch.nn as nn

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'ImageNet100-I' or opt.dataset == 'ImageNet100-II' or opt.dataset == 'ImageNet100' or opt.dataset == 'ImageNet100_baseline':
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    train_cifar_transform = transforms.Compose([
        transforms.Resize((opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((opt.size, opt.size)),
        transforms.CenterCrop(opt.size),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        transform=TwoCropTransform(train_transform),
                                        download=True)
        train_dataset_background = Diffusion_backgound_Dataset(filename='datasets/cifar10_background.txt', transform=TwoCropTransform(train_cifar_transform))
        train_dataset_fake_ood = Diffusion_backgound_Dataset(filename='datasets/cifar10_fake_ood.txt', transform=TwoCropTransform(train_cifar_transform))
        train_dataset = train_dataset + train_dataset_background + train_dataset_fake_ood
        test_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=test_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        train_dataset_background = Diffusion_backgound_Dataset(filename='datasets/cifar100_background.txt', transform=TwoCropTransform(train_cifar_transform))
        train_dataset_fake_ood = Diffusion_backgound_Dataset(filename='datasets/cifar100_fake_ood.txt', transform=TwoCropTransform(train_cifar_transform))
        # train_dataset_pesude = train_dataset_background + train_dataset_fake_ood
        # if len(train_dataset_pesude) > 50000:
        #     train_dataset_pesude = torch.utils.data.Subset(train_dataset_pesude, np.random.choice(len(train_dataset_pesude), 50000, replace=False))
        train_dataset = train_dataset + train_dataset_background + train_dataset_fake_ood
        # train_dataset = train_dataset + train_dataset_pesude
        test_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         train=False,
                                         transform=test_transform)
    elif opt.dataset == 'ImageNet100':
        train_dataset = datasets.ImageFolder(os.path.join('datasets', opt.dataset, 'train'), transform=TwoCropTransform(train_transform))# ImageNet100_I
        train_dataset_background = datasets.ImageFolder(os.path.join('datasets', 'ImageNet_background'), transform=TwoCropTransform(train_transform))
        train_dataset_fake_ood = datasets.ImageFolder(os.path.join('datasets','outputs_blip2/txt2img-samples'), transform=TwoCropTransform(train_transform))
        train_dataset = MergedDataset(train_dataset, train_dataset_background, train_dataset_fake_ood)
        test_dataset = datasets.ImageFolder(os.path.join('datasets', opt.dataset, 'val'), transform=test_transform)
    elif opt.dataset == 'ImageNet100_baseline':
        train_dataset = datasets.ImageFolder(os.path.join('datasets', 'ImageNet100', 'train'), transform=train_transform)
        test_dataset = datasets.ImageFolder(os.path.join('datasets', 'ImageNet100', 'val'), transform=test_transform)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=train_transform)
    else:
        raise ValueError(opt.dataset)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.test_batch_size, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, test_loader

def set_model(opt):
    if opt.model == 'resnet18' or opt.model == 'resnet34':
        model = SupStandardResnet_CIFAR(name=opt.model, dataset=opt.dataset)
    elif opt.model == 'resnet50' or opt.model == 'resnet101':
        model = SupConResNetLargeScale(name=opt.model, dataset=opt.dataset)
    elif opt.model == 'resnet50_base':
        model = StandardResNetBase(name='resnet50', dataset=opt.dataset)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
            model.classifier = torch.nn.DataParallel(model.classifier)
        model = model.cuda()
        cudnn.benchmark = True
    return model

def train(train_loader, model, criterion_supcon, optimizer,epoch, opt):
    """one epoch training"""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        # images_ood = []
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            images = torch.cat([images[0], images[1]], dim=0).cuda() # [batch_size]
            # images = images.cuda()
            # labels = labels.cuda()
        bsz = labels.shape[0]
        labels = labels.repeat(2).cuda()

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        # compute loss

        features_front = model.encoder(images)# [bz+bz, 512] # 特征提取器得到的feature vector
        logits = model.classifier(features_front)
        # ours loss
        features_front = model.projection(features_front)
        features_front = F.normalize(features_front, dim=1) # 监督对比学习中用到的投影头后normalize后的特征
        loss = F.cross_entropy(logits, labels)
        

        # Supcontrast loss
        f1, f2 = torch.split(features_front, [bsz, bsz], dim=0)
        features_contrast = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        supcon_loss = criterion_supcon(features_contrast, labels[:bsz])
        # celoss_plus + two simliarity loss

        # moment encoder path
        
        
        # comloss = criterion_comloss(all_k)
        # loss = loss + supcon_loss# + comloss
        loss = loss + supcon_loss
        # loss = loss
        # loss = loss + supcon_loss
        # loss = loss + comloss
        # loss = loss + supcon_loss
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                   .format(epoch, idx + 1, len(train_loader), loss=losses))
            sys.stdout.flush()
    return losses.avg

def test(model, test_loader):
    model.eval()
    # classifier.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for dict in test_loader:
            data, target = dict[0].cuda(), dict[1].cuda()
            # forward
            output = model.encoder(data)
            output = model.classifier(output)
            loss = F.cross_entropy(output, target)
            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
            # test loss average
            loss_avg += float(loss.data)
    return loss_avg / len(test_loader), correct / len(test_loader.dataset)


def train_acc(model,train_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(train_loader):
            images = torch.cat([images[0], images[1]], dim=0).cuda()
            labels = labels.repeat(2).cuda()
            output = model.encoder(images)
            output = model.classifier(output)
            # acc
            pred = output.data.max(1)[1]
            correct += pred.eq(labels.data).sum().item()
    return correct/(len(train_loader.dataset)*2)
    # return correct/len(train_loader.dataset)

def main():
    opt = parse_option()
    # build data loader
    train_loader, test_loader = set_loader(opt)
    print(len(train_loader.dataset))
    # build model and criterion
    model = set_model(opt)
    optimizer = set_optimizer(opt, model)
    criterion_supcon =  SupConLoss(temperature=0.1).cuda()
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion_supcon, optimizer,epoch, opt)
        time2 = time.time()
        Train_acc = train_acc(model, train_loader)
        print('epoch {}, total time {:.2f}, learning_rate {:.6f}, loss {:.4f} train_acc {:.4f}'.format(epoch, time2 - time1, optimizer.param_groups[0]['lr'], loss, Train_acc))
        test_loss, test_acc = test(model, test_loader)
        print('epoch {} test loss {:.4f} test_acc {:.4f}'.format(epoch, test_loss, test_acc)) 
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

if __name__ == '__main__':
    main()





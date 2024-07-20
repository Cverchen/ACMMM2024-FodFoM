import torchvision.datasets as dset 
datasets = dset.CIFAR10('datasets/cifar10', train=True, download=True)
cls_name = datasets.classes
Line = []
Line_class = []
for i in range(10):
    Line_class.append([])
for i in range(10):
    with open('datasets/cifar10/cifar10_information/cifar10I2T.txt', 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines: 
            if line.split(':')[0] == '[{}]'.format(i):
                Line_class[i].append(line.split(':')[1])

with open('datasets/cifar10/cifar10_information/cifar10I2T_Class.txt', 'w', encoding='utf-8') as f:
    for i in range(10):
        for j in range(len(Line_class[i])):
            f.write(str(i) + ':' + Line_class[i][j]+'\n')
            

import torchvision.datasets as dset 
datasets = dset.CIFAR100('datasets/cifar100', train=True, download=True)
cls_name = datasets.classes
Line = []
Line_class = []
for i in range(100):
    Line_class.append([])
for i in range(100):
    with open('datasets/cifar100/cifar100_information/cifar100I2T.txt', 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines: 
            if line.split(':')[0] == '[{}]'.format(i):
                Line_class[i].append(line.split(':')[1])

with open('datasets/cifar100/cifar100_information/cifar100I2T_Class.txt', 'w', encoding='utf-8') as f:
    for i in range(100):
        for j in range(len(Line_class[i])):
            f.write(str(i) +':'+ Line_class[i][j]+'\n')
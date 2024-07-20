import torchvision.transforms as trn
import torchvision.datasets as dset
import numpy as np
from torchvision.datasets.folder import *
import datasets.svhn_loader as svhn
import torch
from typing import *

class TwoTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class FilterableImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            valid_classes: List = None
    ):
        self.valid_classes = valid_classes
        super(FilterableImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [valid_class for valid_class in classes if valid_class in self.valid_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def build_cider_dataset(dataset, mode="train"):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = trn.Normalize(mean=mean, std=std)
    
    train_transform = trn.Compose([
        trn.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        trn.RandomHorizontalFlip(),
        trn.RandomApply([
            trn.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        trn.RandomGrayscale(p=0.2),
        trn.ToTensor(),
        normalize,
    ])
    test_transform = trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)])
    if dataset == 'cifar10':
        if mode == "train":
            data = dset.CIFAR10(root='Datasets/cifar10',
                                    download=True,
                                    train=True,
                                    transform=TwoTransform(train_transform)
                                    # transform=test_transform
                                    )
        else:
            data = dset.CIFAR10(root='Datasets/cifar10',
                                   download=True,
                                   train=False,
                                   transform=test_transform
                                   )
        num_classes = 10
    elif dataset == 'cifar100':
        if mode == "train":
            data = dset.CIFAR100(root='Datasets/cifar100',
                                        download=True,
                                        train=True,
                                        transform=TwoTransform(train_transform)
                                        )
        else:
            data = dset.CIFAR100(root='Datasets/cifar100',
                                    download=True,
                                    train=False,
                                    transform=test_transform
                                    )
        num_classes = 100
    if 'ImageNet100' in dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225) 
        test_transform = trn.Compose([
            trn.Resize(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize(mean, std)
        ])
        if mode == 'test':
            data = dset.ImageFolder(os.path.join('datasets',dataset,'val'), transform=test_transform)
            # data = dset.ImageFolder(os.path.join('dream_datasets',dataset,'ImageNet100_dream','val'), transform=test_transform)
            # data = dset.ImageFolder(os.path.join('Datasets',dataset,'val'), transform=test_transform)
        num_classes = 100
    elif dataset == "SVHN":
        if mode == "train":
            data = svhn.SVHN(root='Datasets/SVHN/', split="train",
                             transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]),
                             download=False)
        else:
            data = svhn.SVHN(root='Datasets/SVHN/', split="test",
                             transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]),
                             download=True)
        # if len(data) > 10000:
            # data = torch.utils.data.Subset(data, np.random.choice(len(data), 10000, replace=False))
        num_classes = 10
    elif dataset == "Textures":
        data = dset.ImageFolder(root="Datasets/dtd/images",
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "Places365":
        data = dset.ImageFolder(root="Datasets/places365/test_subset",
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
        # if len(data) > 10000:
            # data = torch.utils.data.Subset(data, np.random.choice(len(data), 10000, replace=False))
        num_classes = 10
    elif dataset == "LSUN-C":
        data = dset.ImageFolder(root="Datasets/LSUN_C/",
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),trn.ToTensor(), trn.Normalize(mean, std)]))
        # if len(data) > 10000:
            # data = torch.utils.data.Subset(data, np.random.choice(len(data), 10000, replace=False))
        num_classes = 10
    elif dataset == "LSUN-R":
        data = dset.ImageFolder(root="Datasets/LSUN_R/",
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),trn.ToTensor(), trn.Normalize(mean, std)]))
        # if len(data) > 10000:
            # data = torch.utils.data.Subset(data, np.random.choice(len(data), 10000, replace=False))
        num_classes = 10
    elif dataset == "iSUN":
        data = dset.ImageFolder(root="Datasets/iSUN/",
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),trn.ToTensor(), trn.Normalize(mean, std)]))
        # if len(data) > 10000:
            # data = torch.utils.data.Subset(data, np.random.choice(len(data), 10000, replace=False))
        num_classes = 10

    return data, num_classes
    

def build_dataset(dataset, mode="train", eval=False):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                       trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)])
    if dataset == 'cifar10':
        if mode == "train":
            if eval:
                data = dset.CIFAR10(root='Datasets/cifar10',
                                    download=True,
                                    train=True,
                                    transform=test_transform
                                    )
            else:
                data = dset.CIFAR10(root='Datasets/cifar10',
                                    download=True,
                                    train=True,
                                    transform=train_transform
                                    )
        else:
            data = dset.CIFAR10(root='Datasets/cifar10',
                                   download=True,
                                   train=False,
                                   transform=test_transform
                                   )
        num_classes = 10
    elif dataset == 'cifar100':
        if mode == "train":
            if eval:
                data = dset.CIFAR100(root='Datasets/cifar100',
                                     download=True,
                                     train=True,
                                     transform=test_transform
                                     )
            else:
                data = dset.CIFAR100(root='Datasets/cifar100',
                                     download=True,
                                     train=True,
                                     transform=train_transform
                                     )
        else:
            data = dset.CIFAR100(root='Datasets/cifar100',
                                    download=True,
                                    train=False,
                                    transform=test_transform
                                    )
        num_classes = 100
    if 'ImageNet100' in dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        test_transform = trn.Compose([
            trn.Resize(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize(mean, std)
        ])
        if mode == 'train':
            data = dset.ImageFolder(os.path.join('datasets',dataset,'train'), transform=test_transform)
            # data = dset.ImageFolder(os.path.join('dream_datasets',dataset,'ImageNet100_dream','train'), transform=test_transform)
            # data = dset.ImageFolder(os.path.join('Datasets',dataset,'train'), transform=test_transform)
        num_classes = 100
    elif dataset == "Textures":
        data = dset.ImageFolder(root="Datasets/dtd/images",
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "Places365":
        data = dset.ImageFolder(root="Datasets/places365/test_subset",
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "LSUN-C":
        data = dset.ImageFolder(root="Datasets/LSUN_C/",
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "LSUN-R":
        data = dset.ImageFolder(root="Datasets/LSUN_R/",
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "iSUN":
        data = dset.ImageFolder(root="Datasets/iSUN/",
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10

    return data, num_classes


def set_ood_loader_Imagenet(out_dataset):
    root = 'datasets/ImageNet_OOD_dataset'
    normalize = trn.Normalize(mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225))
    ood_transform = trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize
    ])
    testsetout = dset.ImageFolder(os.path.join(root, out_dataset),transform=ood_transform)
    if len(testsetout) > 10000:
        testsetout = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
    return testsetout, 1
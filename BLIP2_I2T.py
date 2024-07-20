import os
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
model.to(device)


Path = os.listdir('datasets/ImageNet100/train')
Path.sort()
class_name = []
with open('datasets/Imagenet100/imagenet100_information/imagenet100_classname.txt', 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()
    for line in lines:
        class_name.append(line)
index = 0
with open('datasets/Imagenet100/imagenet100_information/imagenettotext_blip2.txt', 'a', encoding='utf-8') as f:
    for path in tqdm(Path):
        f.write(class_name[index]+'\n\n')
        path = os.path.join('datasets/ImageNet100/train',path)
        image_Path = os.listdir(path)
        for image_path in tqdm(image_Path):
            image_path = os.path.join(path, image_path)
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            f.write(image_path+' --- '+generated_text+'\n')
            f.flush()
        index += 1

import numpy as np
from tqdm import tqdm
# 下载并加载CIFAR-10数据集
train_dataset = CIFAR10(root='datasets/cifar10',transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),download=True)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=6, pin_memory=False, sampler=None)
with open('datasets/cifar10/cifar10_information/cifar10I2T.txt','w',encoding='utf-8') as f:
    for image, label in tqdm(train_loader):
        to_pil_image = transforms.ToPILImage()(image[0])
        inputs = processor(images=to_pil_image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        f.write(str(np.array(label)) + ':'+ generated_text+'\n')
        f.flush()


import numpy as np
from tqdm import tqdm
# 下载并加载CIFAR-10数据集
train_dataset = CIFAR100(root='datasets/cifar100',transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),download=True)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=6, pin_memory=False, sampler=None)
with open('datasets/cifar100/cifar100_information/cifar100I2T.txt','w',encoding='utf-8') as f:
    for image, label in tqdm(train_loader):
        to_pil_image = transforms.ToPILImage()(image[0])
        inputs = processor(images=to_pil_image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        f.write(str(np.array(label)) + ':'+ generated_text+'\n')
        f.flush()







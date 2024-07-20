import os
# CIFAR10 BACKGROUND
data_folder = 'datasets/CIFAR10_BACKGROUND/CIFAR10_background'
image_path = os.listdir(data_folder)
with open('datasets/cifar10_background.txt', 'w', encoding='utf-8') as f:
    for path in image_path:
        f.write(data_folder+'/'+path+',10\n')
# CIFAR10 FAKE OOD
data_folder = 'datasets/cifar10_output/txt2img-samples/samples'
image_path = os.listdir(data_folder)
with open('datasets/cifar10_fake_ood.txt', 'w', encoding='utf-8') as f:
    for path in image_path:
        f.write(data_folder+'/'+path+',10\n')

# CIFAR100 BACKGROUND
data_folder = 'datasets/CIFAR100_BACKGROUND/CIFAR100_background'
image_path = os.listdir(data_folder)
with open('datasets/cifar100_background.txt', 'w', encoding='utf-8') as f:
    for path in image_path:
        f.write(data_folder+'/'+path+',100\n')

# CIFAR100 FAKE OOD
data_folder = 'datasets/cifar100_output/txt2img-samples/samples'
image_path = os.listdir(data_folder)
with open('datasets/cifar100_fake_ood.txt', 'w', encoding='utf-8') as f:
    for path in image_path:
        f.write(data_folder+'/'+path+',100\n')




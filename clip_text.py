import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import clip
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import torchvision.datasets as dset
import argparse

def get_text_prototype_cifar10():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-L/14', device)
    Description = []
    classname = dset.CIFAR10(root='datasets/cifar10').classes
    Text_features = []
    for i in range(10):
        Description.append([])
    with open('cifar10_information/cifar10I2T_Class.txt', 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            Description[int(line.split(':')[0])].append('This is a photo of {} and '.format(classname[int(line.split(':')[0])]) + line.split(':')[1])
                    
    for i in range(10):
        print(Description[i][0])
    print('Generating textual embeddings:........')
    for i in tqdm(range(len(classname))):
        text_inputs = torch.cat([clip.tokenize(description) for description in Description[i]]).to(device)
        with torch.no_grad():
            Text_features.append(model.encode_text(text_inputs).float())
    print('Saving every class\'s text embeddings:............')
    if not os.path.exists('cifar10_text_feature'):
        os.makedirs('cifar10_text_feature')
    for i in tqdm(range(len(classname))):
        np.save('cifar10_text_feature/class_{}.npy'.format(i), Text_features[i].cpu().numpy())

def get_text_prototype_cifar100():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-L/14', device)
    Description = []
    classname = dset.CIFAR100(root='datasets/cifar100').classes

    Text_features = []
    for i in range(100):
        Description.append([])
    with open('cifar100_information/cifar100I2T_Class.txt', 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            Description[int(line.split(':')[0])].append('This is a photo of {} and '.format(classname[int(line.split(':')[0])]) + line.split(':')[1])
    print('Generating textual embeddings:........')
    for i in tqdm(range(len(classname))):
        text_inputs = torch.cat([clip.tokenize(description) for description in Description[i]]).to(device)
        with torch.no_grad():
            Text_features.append(model.encode_text(text_inputs).float())
    print('Saving every class\'s text embeddings:............')
    if not os.path.exists('cifar100_text_feature'):
        os.makedirs('cifar100_text_feature')
    for i in tqdm(range(len(classname))):
        np.save('cifar100_text_feature/class_{}.npy'.format(i), Text_features[i].cpu().numpy())


#生成所有描述文本的embeddings
def get_text_prototype():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-L/14', device)
    Description = []
    classname = []
    classindex = []
    Text_features = []
    with open('imagenet100_information/imagenet100_classindex.txt', 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            classindex.append(line)
    with open('imagenet100_information/imagenet100_classname.txt', 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            classname.append(line)
    for i in range(100):
        Description.append([])
    index = 0
    flag = 0
    with open('imagenet100_information/imagenettotext_blip2.txt', 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            if classindex[index] in line and flag % 2 == 0:
                Description[index].append('This is a photo of {} and '.format(classname[index]) + line.split(' --- ')[1][:60])
            else:
                if flag != 0 and flag % 2 == 0:
                    index += 1
                flag += 1
    print('Generating textual embeddings:........')
    for i in tqdm(range(len(classname))):
        text_inputs = torch.cat([clip.tokenize(description) for description in Description[i]]).to(device)
        with torch.no_grad():
            Text_features.append(model.encode_text(text_inputs).float())
    print('Saving every class\'s text embeddings:............')
    # if not os.path.exists('text_features_blip2'):
    if not os.path.exists('text_features_imagenet100'):
        # os.makedirs('text_features_blip2')
        os.makedirs('text_features_imagenet100')
    for i in tqdm(range(len(classname))):
        # np.save('text_features_blip2/class_{}.npy'.format(i), Text_features[i].cpu().numpy())
        np.save('text_features_imagenet/class_{}.npy'.format(i), Text_features[i].cpu().numpy())


def get_text_prototype_twostage(fake_text_path, fake_token_path, dataset):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model, _ = clip.load('ViT-L/14', device)
    print('Generating fake textual tokens:........')
    if dataset == 'cifar10':
        num_class = 10
    else:
        num_class = 100
    if not os.path.exists(fake_token_path):
        os.makedirs(fake_token_path)
    for i in tqdm(range(num_class)):
        with torch.no_grad():
            # if i <= 7:
                # continue
            token_embedding_first = torch.from_numpy(np.load('{}/class_{}.npy'.format(fake_text_path, i))).to(device)
            # noise_std = 0.001
            # noise = torch.from_numpy(np.random.normal(loc=0, scale=noise_std, size=token_embedding_first.shape)).to(device)
            # token_embedding_first = token_embedding_first + noise
            # text = torch.cat([clip.tokenize('This is a photo of') for i in range(token_embedding_first.shape[0])]).to(device)
            text = torch.cat([clip.tokenize('This is a high quality image of') for i in range(token_embedding_first.shape[0])]).to(device)
            token_embedding = model.encode_text_backward(text, token_embedding_first)[1]
            np.save('{}/class_{}.npy'.format(fake_token_path, i), token_embedding.cpu().numpy())

# 寻找边际向量 取与均值类别向量的余弦相似度前20%小的值
def edge_feature_search(n_class, percentage=20, feature_path='cifar10_text_feature'):
    edge_feature_class = []
    mean_class = []
    for i in range(n_class):
        text_feature = np.load('{}/class_{}.npy'.format(feature_path, i))
        mean_class.append(np.mean(text_feature, axis=0))
        cos_sim = []
        for j in range(len(text_feature)):
            cos_sim.append(text_feature[j].dot(mean_class[i])/(np.linalg.norm(text_feature[j]) * np.linalg.norm(mean_class[i])))
        cos_sim = np.array(cos_sim)
        threshold = np.percentile(cos_sim, percentage)
        indices = np.where(cos_sim<=threshold)
        edge_feature = text_feature[indices]        # 该类簇的边际向量
        edge_feature_class.append(edge_feature)
    return edge_feature_class, mean_class
    # print(cos_sim)

# 寻找边际向量 取与均值类别向量欧式距离前20%小的值
def edge_feature_search_euladistance(n_class, percentage=20, feature_path='cifar10_text_feature'):
    edge_feature_class = []
    mean_class = []
    for i in range(n_class):
        text_feature = np.load('{}/class_{}.npy'.format(feature_path, i))
        mean_class.append(np.mean(text_feature, axis=0))
        eudistance = []
        for j in range(len(text_feature)):
            eudistance.append(-np.linalg.norm(mean_class[i] - text_feature[j]))
        eudistance = np.array(eudistance)
        threshold = np.percentile(eudistance, percentage)
        indices = np.where(eudistance<=threshold)
        edge_feature = text_feature[indices]        # 该类簇的边际向量
        edge_feature_class.append(edge_feature)
    return edge_feature_class, mean_class

# 寻找边际向量 取与均值类别向量欧式距离前20%小的值
def edge_feature_search_maha(n_class, percentage=20, feature_path='cifar10_text_feature'):
    edge_feature_class = []
    mean_class = []
    for i in tqdm(range(n_class)):
        text_feature = np.load('{}/class_{}.npy'.format(feature_path, i))
        text_feature = torch.from_numpy(text_feature).cuda()
        # cov_matrix = torch.cov(text_feature.T)
        cov_matrix = torch.pinverse(torch.cov(text_feature.T))
        mean_class.append(torch.mean(text_feature, dim=0))
        mahadistance = []
        for j in range(len(text_feature)):
            diff = text_feature[j] - mean_class[i]
            mahadistance.append((-1 * torch.matmul(torch.matmul(diff, cov_matrix.float()), diff.t())).cpu().numpy())
        mahadistance = np.array(mahadistance)
        threshold = np.percentile(mahadistance, percentage)
        print(threshold)
        indices = np.where(mahadistance<=threshold)
        print(indices)
        text_feature = text_feature.cpu().numpy()
        edge_feature = text_feature[indices]        # 该类簇的边际向量
        edge_feature_class.append(edge_feature)
    mean_class = [] 
    for i in tqdm(range(n_class)):
        text_feature = np.load('{}/class_{}.npy'.format(feature_path, i))
        mean_class.append(np.mean(text_feature, axis=0))
    return edge_feature_class, mean_class
    
# 构造类簇边界伪特征
def fake_feature_generate(edge_feature_class, edge_feature, slide, num_per_sample, addslide):
    # slide:在方向向量上移动的步长从slide开始，每次移动0.00005,共移动num_per_sample个0.00005
    # num_per_sample: 每个样本构造多少个伪特征，目前是10个 
    n_class = len(edge_feature_class)
    fake_feature = []
    for i in range(n_class):
        fake_feature.append([])
    for i in range(n_class):
        feature_anchor = edge_feature_class[i]
        for j in range(edge_feature[i].shape[0]):
            feature = edge_feature[i][j]
            # 构造边际向量与均值向量之间的方向向量
            direction_vertor = np.linalg.norm(feature - feature_anchor)
            # 边际向量与方向向量乘相应步长得到其最后的伪特征向量
            index = 0
            for k in range(num_per_sample):
                Fake_feature = feature + direction_vertor * (slide+index)
                index += addslide
                fake_feature[i].append(Fake_feature)            
    return fake_feature

def parse_option():
    parser = argparse.ArgumentParser('generate fake feature token for stable diffusion to generate fake images')
    parser.add_argument('--dataset', type=str,default='ImageNet100',
                        help='cifar10, cifar100, ImageNet100')
    parser.add_argument('--n_class', type=int, default=100, help='number of ID classes')
    parser.add_argument('--percentage', type=int, default=20, help='Minimum percentage of similarity to the mean vector as a condition for filtering edit vectors')
    parser.add_argument('--slide', type=float, default=3e-5, help='The step size of the move on the direction vector starts from slide')
    parser.add_argument('--add_slide', type=float, default=3e-5, help='The step length of each move')
    parser.add_argument('--fake_feature_path', type=str, help='Path to where the generated false text representation is stored')
    parser.add_argumetn('--num_per_sample', type=int, default=5, help='Number of spurious features generated per edge vector')
    parser.add_argument('--fake_token_path', type=str, help='Storage path for fake token obtained using fake embedding')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = parse_option()
    slide = args.slide
    n_class = args.n_class
    percentage = args.percentage
    add_slide = args.add_slide
    fake_feature_path = args.fake_feature_path
    fake_token_path = args.fake_token_path
    num_per_sample = args.num_per_smaple
    if args.dataset == 'cifar10':
        get_text_prototype_cifar10()
        edge_feature_class, mean_class = edge_feature_search_maha(n_class, percentage, feature_path='cifar10_text_feature')
        fake_feature = fake_feature_generate(edge_feature_class=mean_class, edge_feature=edge_feature_class, slide=slide, num_per_sample=num_per_sample, addslide=add_slide)
        for i in range(10):
            Fake_feature = np.array(fake_feature[i])   
        get_text_prototype_twostage(fake_text_path=fake_feature_path, fake_token_path=fake_token_path, dataset=args.dataset)
    elif args.dataset == 'cifar100':
        get_text_prototype_cifar100()
        edge_feature_class, mean_class = edge_feature_search(n_class, percentage, feature_path='cifar100_text_feature')
        fake_feature = fake_feature_generate(edge_feature_class=mean_class, edge_feature=edge_feature_class, slide=slide, num_per_sample=num_per_sample, addslide=add_slide)
        for i in range(100):
            Fake_feature = np.array(fake_feature[i])
            np.save(fake_feature_path+'/class_{}.npy'.format(i), Fake_feature)
        get_text_prototype_twostage(fake_text_path=fake_feature_path, fake_token_path=fake_token_path, dataset=args.dataset)
    elif args.dataset == 'ImageNet100':
        # ImageNet100
        get_text_prototype()
        edge_feature_class, mean_class = edge_feature_search(n_class, percentage, feature_path='text_features_imagenet100')
        fake_feature = fake_feature_generate(edge_feature_class=mean_class, edge_feature=edge_feature_class, slide=slide, num_per_sample=num_per_sample, addslide=add_slide)
        for i in range(100):
            Fake_feature = np.array(fake_feature[i])
            np.save(fake_feature_path+'/class_{}.npy'.format(i), Fake_feature)
        get_text_prototype_twostage(fake_text_path=fake_feature_path, fake_token_path=fake_token_path, dataset='ImageNet100')
    # 1: 0.00001 num_per_sample=3 0.00004   0.00001 0.00005 0.00009  ImageNet100(√)
    # 2: 0.00001 num_per_sample=3 0.00003   0.00001 0.00004 0.00007  Imagenet100

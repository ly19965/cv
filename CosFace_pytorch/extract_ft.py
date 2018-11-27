from PIL import Image
import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
import os
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import net


def extractDeepFeature(img, model, is_gray):
    if is_gray:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5,), std=(0.5,))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    img, img_ = transform(img), transform(F.hflip(img))
    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')
    ft = torch.cat((model(img), model(img_)), 1)[0].to('cpu')
    return ft


def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i * n / n_folds:(i + 1) * n / n_folds]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def eval(model, model_path=None, is_gray=False):
    predicts = []
    model.load_state_dict(torch.load(model_path))
    model.eval()
    root = '/home/wangyf/dataset/lfw/lfw-112X96/'
    with open('/home/wangyf/Project/sphereface/test/data/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]

    with torch.no_grad():
        for i in range(6000):
            p = pairs_lines[i].replace('\n', '').split('\t')

            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")

            with open(root + name1, 'rb') as f:
                img1 =  Image.open(f).convert('RGB')
            with open(root + name2, 'rb') as f:
                img2 =  Image.open(f).convert('RGB')
            f1 = extractDeepFeature(img1, model, is_gray)
            f2 = extractDeepFeature(img2, model, is_gray)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, distance, sameflag))

    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    predicts = np.array(map(lambda line: line.strip('\n').split(), predicts))
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

    return np.mean(accuracy), predicts
def extract_one_img_ft(model, model_path, img_path, is_gray=False):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    img = Image.open(img_path).convert('RGB')
    f1 = extractDeepFeature(img, model, is_gray)
    return f1
    
def make_cosine_similarity_matrix(ft_dict, img_list, fo, foo):
    fo.write('img_name')
    img_list.sort()
    for item in img_list:
        fo.write('\t' + item)
    fo.write('\n')
    for item_1 in img_list:
        fo.write(item_1)
        foo.write(item_1)
        foo.write('\n')
        temp_dict = {}
        for item_2 in img_list:
            f1 = ft_dict[item_1]
            f2 = ft_dict[item_2]
            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            temp_dict[item_2] = float(distance)
            fo.write('\t' + str(float(distance)) + '_' + item_2)
        fo.write('\n')
        sorted_list = sorted(temp_dict.items(), key=lambda x:x[1], reverse=True)
        max_similarity_5 = sorted_list[1:6]
        min_similarity_5 = sorted_list[-5:]
        fo.write('Max similarity five images : ' + '\n') 
        foo.write('Max similarity five images : ' + '\n') 
        for item in max_similarity_5:
            fo.write(str(item[1]) + '_' + str(item[0]) + '\t')
            foo.write(str(item[1]) + '_' + str(item[0]) + '\t')
        fo.write('\n')
        foo.write('\n')
        fo.write('Min similarity five images : ' + '\n') 
        foo.write('Min similarity five images : ' + '\n') 
        for item in min_similarity_5:
            fo.write(str(item[1]) + '_' + str(item[0]) + '\t')
            foo.write(str(item[1]) + '_' + str(item[0]) + '\t')
        fo.write('\n')
        foo.write('\n' + '\n')
    fo.close()
    foo.close()

def extract_all_img_ft(model, model_path, all_img_path, img_list):
    ft_dict = {}
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for item in img_list:
        img_path = os.path.join(all_img_path, item)
        img = Image.open(img_path).convert('RGB')
        img_ft = extractDeepFeature(img, model, False)
        ft_dict[item] = img_ft
    return ft_dict

def make_cosine_similarity_file(target_img, all_img_path, model, model_path):
    '''
    file_dir = 'experiments'
    '''
    img_list = os.listdir(all_img_path)
    ft_dict = {}
    txt_name = '_'.join(all_img_path.split('/'))
    fo = open('experiments/{}.txt'.format(txt_name), 'w')
    foo = open('experiments/min_max_{}.txt'.format(txt_name), 'w')
    ft_dict = extract_all_img_ft(model, model_path, all_img_path, img_list)
    make_cosine_similarity_matrix(ft_dict, img_list, fo, foo)

if __name__ == '__main__':
    model = net.sphere().to('cuda')
    model_path = 'checkpoint/original.pth'
    img_path = 'lfw-112X96/Josh_Kronfeld/Josh_Kronfeld_0001.jpg'
    CASIA_dataset = '/data1/aipd_tuijian/charlesliu/dataset/CASIA-WebFace'
    img_dir_name_india_women = ['6432391']
    img_dir_name_men = ['6418193']
    img_dir_name_india_women_glasses = ['027.jpg', '054.jpg']
    img_dir_name_india_women_make_up = ['043.jpg', '037.jpg', '028.jpg', '016.jpg', '007.jpg']
    img_dir_name_india_women_occlusion = ['023.jpg', '017.jpg']
    
    all_img_path = os.path.join(CASIA_dataset, img_dir_name_men[0])
    make_cosine_similarity_file(None, all_img_path, model, model_path)

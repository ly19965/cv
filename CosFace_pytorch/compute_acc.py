#coding=utf-8
from PIL import Image

import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
import os
import sys
import json
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import net

num_recall_vid = 1
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
    img = Image.open(img_path).resize((96, 112), Image.BILINEAR).convert('RGB')
    f1 = extractDeepFeature(img, model, is_gray)
    return f1
    
def make_cosine_similarity_matrix(ft_dict, img_list, fo, dir_name):
    img_list.sort()
    max_distance_dict = {}
    mean_distance_dict = {}
    for item_1 in img_list:
        temp_dict = {}
        mean_distance = 0
        max_distance = -1
        for item_2 in img_list:
            f1 = ft_dict[item_1]
            f2 = ft_dict[item_2]
            distance = float(f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5))
            if distance > max_distance and distance < 1.0:
                max_distance = distance
            mean_distance += distance
            temp_dict[item_2] = float(distance)
        mean_distance = 1.0 * mean_distance / len(img_list)
        max_distance_dict[item_1] = max_distance
        mean_distance_dict[item_1] = mean_distance
    sorted_max_distance_list = sorted(max_distance_dict.items(), key=lambda x:x[1], reverse=True)
    sorted_mean_distance_list = sorted(mean_distance_dict.items(), key=lambda x:x[1], reverse=True)
    fo.write(dir_name  + '\n')
    fo.write('max_distance' + '\t' + str(sorted_max_distance_list[0][0]) + '\t' + str(sorted_max_distance_list[0][1]) + '\n')
    fo.write('mean_distance' + '\t' + str(sorted_mean_distance_list[0][0]) + '\t' + str(sorted_mean_distance_list[0][1]) + '\n')
    

def extract_all_img_ft(model, model_path, all_img_path, img_list):
    ft_dict = {}
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for item in img_list:
        img_path = os.path.join(all_img_path, item)
        img = Image.open(img_path).resize((96, 112), Image.BILINEAR).convert('RGB')
        img_ft = extractDeepFeature(img, model, False)
        ft_dict[item] = img_ft
    return ft_dict

def make_cosine_similarity_file(target_img, all_img_path, model, model_path, fo):
    '''
    file_dir = 'experiments'
    '''
    img_list = os.listdir(all_img_path)
    ft_dict = {}
    ft_dict = extract_all_img_ft(model, model_path, all_img_path, img_list)
    dir_name = all_img_path.split('/')[-1]
    make_cosine_similarity_matrix(ft_dict, img_list, fo, dir_name)

def compute_acc(model, model_path, test_dataset, ft_distance_300_dict):
    threshold = 0.225
    model.load_state_dict(torch.load(model_path))
    model.eval()
    fo = open('test_result_{}.txt'.format(num_recall_vid), 'w')
    fo_label = open(label_file)
    label_dict = {}
    vid_label = []
    a=b=c=d=0 #b预测为正，a预测为正且label为正，c正样本总数，d负样本总数
    for key, value in ft_distance_300_dict.items():
        vid_label.append(key)
    for line in fo_label:
        line = line.strip().split('\t')
        label_dict[line[0]] = line[1].split(',')
        for item in line[1].split(','):
            if item in vid_label:
                c += 1
            else:
                d += 1
    for lenth, dir in enumerate(os.listdir(test_dataset)):
        all_result_list = []
        abs_dir_name = os.path.join(test_dataset, dir)
        for test_img in os.listdir(abs_dir_name):
            abs_img_name = os.path.join(abs_dir_name, test_img)
            #fo.write('test_img:' + '\t' +  abs_img_name + '\n')
            img = Image.open(abs_img_name).resize((96, 112), Image.BILINEAR).convert('RGB')
            f1 = extractDeepFeature(img, model, False)
            result_dict = {}
            for key, value in ft_distance_300_dict.items():
                f2 = ft_distance_300_dict[key]
                distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
                if distance >= threshold:
                    result_dict[key] = distance
            if result_dict:
                all_result_list.append(result_dict)
            #if len(result_dict) > 1:
                #fo.write('same face and cosine similarity:'  + '\n')
                #sorted_result_list = sorted(result_dict.items(), key=lambda x:x[1], reverse=True)
                #for item in sorted_result_list:
                    #fo.write(item[0] + '\t' + str(item[1]) + '\n')
                #fo.write('\n')
            #elif len(result_dict) == 1:
                #fo.write('same face and cosine similarity:'  + '\n')
                #fo.write(result_dict.items()[0][0] + '\t' + str(result_dict.items()[0][1]) + '\n')
                #fo.write('\n')
            #else:
                #fo.write('Not found!' + '\n' + '\n' )
        vid_pred = {}
        for dict in all_result_list:
            for key, value in dict.items():
                if key in vid_pred:
                    vid_pred[key] += 1
                else:
                    vid_pred[key] = 1
        if len(vid_pred) == 1:
            b += 1
            #fo.write('Vid predict label:' + '\t' + str(vid_pred.items()[0][0]) + '\n')
            #fo.write('\n')
            #print label_dict[dir]
            if str(vid_pred.items()[0][0]) in label_dict[dir]:
                a += 1


        elif len(vid_pred) > 1:
            b += 1
            pred_label = []
            sorted_vid_pred = sorted(vid_pred.items(), key=lambda x:x[1], reverse=True) 
            for item in sorted_vid_pred:
                if item[1] < 7:
                    break
                pred_label.append(item[0])
            pred_label = pred_label[:num_recall_vid]
            #fo.write('Vid predict label:' + '\t'.join(pred_label) + '\n' + '\n')
            for item in pred_label:
                if item in label_dict[dir]:
                    a += 1
                    break
        #else:
            #fo.write('Vid not predict label!' + '\n')
        print ('Finish one vid !')
        if lenth == 230 or lenth ==200:
            print ('a=', a, 'b=', b, 'c=', c, 'd=', d)
            print ('acc: {}%, recall: {}%'.format(1.0 * a /b * 100 , 1.0 * a / c * 100))
    print ('a=', a, 'b=', b, 'c=', c, 'd=', d)
    print ('acc: {}%, recall: {}%'.format(1.0 * a /b * 100 , 1.0 * a / c * 100))



if __name__ == '__main__':
    model = net.sphere().to('cuda')
    model_path = 'checkpoint/original.pth'
    img_path = 'lfw-112X96/Josh_Kronfeld/Josh_Kronfeld_0001.jpg'
    CASIA_dataset = '/data1/aipd_tuijian/charlesliu/dataset/CASIA-WebFace'
    V1_dataset = '/data1/aipd_tuijian/charlesliu/dataset/V1'
    test_dataset ='/data1/aipd_tuijian/charlesliu/dataset/actor_imgs_multi_0910_test300'
    fo = open('ft_300_dict.json')
    num_recall_vid = int(sys.argv[1])
    label_file = 'vid_actors_0910_test300'
    ft_300_dict = json.load(fo)
    ft_distance_300_dict = {}
    for key, value in ft_300_dict.items():
        abs_path = os.path.join(V1_dataset, key, value)
        feature = extract_one_img_ft(model, model_path, abs_path)
        ft_distance_300_dict[key] = feature
    compute_acc(model, model_path, test_dataset, ft_distance_300_dict)

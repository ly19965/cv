import sys
import os
import numpy as np
#import faiss
import pickle
import net
import time
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from PIL import Image
from torchvision.transforms import functional as F
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from src import detect_faces
from skimage import transform as trans
import cv2

#f_rec = FaceNetApi()

def vote(x):
    return Counter(x).most_common(1)[0][0]
    
def get_predict_result(Distance, Index, train_y):
    predict_labels = []
    distances = []
    for i in range(len(Index)):
        one_ind = Index[i]
        vote_label = vote([train_y[k] for k in one_ind])
        predict_labels.append(vote_label)
        one_dis = Distance[i]
        min_dis = 100
        for j in range(len(one_ind)):
            if train_y[one_ind[j]] == vote_label:
                if one_dis[j] < min_dis:
                    min_dis = one_dis[j]
        distances.append(min_dis)
    return predict_labels, distances
    
def load_train_data(input_path):
    (base_path, version_name) = os.path.split(input_path)
    if model_name == 'facenet':
        data_pkl = input_path + "_facenet_data.pkl"
    elif model_name == 'cosface':
        data_pkl = input_path + "_cosface_data.pkl"
    train_x = []
    train_y = []
    if os.path.isfile(data_pkl):
        with open(data_pkl, "rb") as f:  
            train_x, train_y = pickle.load(f) 
    else:
        for actor_id in os.listdir(input_path):
            embeding = []
            imgs_paths = []
            one_actor_path = os.path.join(input_path, actor_id)
            if not os.path.isdir(one_actor_path):
                continue
            for img_file_name in os.listdir(one_actor_path):
                img_file_path = os.path.join(one_actor_path, img_file_name)
                imgs_paths.append(img_file_path)
            imgs_paths.sort()
            size = len(imgs_paths)
            if size == 0 :
                continue
            #emb = f_rec.feature_extract_with_image_path(imgs_paths)
            emb = get_imgs_vec_with_path(imgs_paths)
            print actor_id

            for i in range(size):
                train_x.append(emb[i,:])
                train_y.append(int(actor_id))
        f = open(data_pkl, 'wb')
        pickle.dump([train_x, train_y], f)
        f.close()
    return np.array(train_x), np.array(train_y)
    
def load_test_data(input_path, vid_actors_file):
    (base_path, version_name) = os.path.split(input_path)
    if model_name == 'facenet':
        data_pkl = input_path + "_facenet_data.pkl"
    elif model_name == 'cosface':
        data_pkl = input_path + "_cosface_data.pkl"
    test_x = [] # vid_num * img_num * emb
    test_y = [] # vid_num
    all_imgs_paths = [] # vid_num * img_num
    vids = [] # vid_num
    if os.path.isfile(data_pkl):
        with open(data_pkl, "rb") as f:  
            test_x, test_y, all_imgs_paths, vids = pickle.load(f) 
    else:
        vid_actors_dict = {}
        with open(vid_actors_file) as f_dict:
            for line in f_dict:
                line_v = line.strip().split("\t")
                vid = line_v[0]
                actors_str = line_v[1].split(",")
                actors = []
                for actor in actors_str:
                    actors.append(int(actor))
                vid_actors_dict[vid] = actors
        for vid in os.listdir(input_path):
            embeding = []
            imgs_paths = []
            one_vid_path = os.path.join(input_path, vid)
            if not os.path.isdir(one_vid_path):
                continue
            i = -1
            for img_file_name in os.listdir(one_vid_path):
                i += 1
                #if i % 5 !=0 :continue
                img_file_path = os.path.join(one_vid_path, img_file_name)
                imgs_paths.append(img_file_path)
            imgs_paths.sort()
            size = len(imgs_paths)
            if size == 0 :
                continue
            #emb = f_rec.feature_extract_with_image_path(imgs_paths)
            emb = get_imgs_vec_with_path(imgs_paths)
            print vid

            test_x.append(emb)
            test_y.append(vid_actors_dict[vid])
            all_imgs_paths.append(imgs_paths)
            vids.append(vid)
        f = open(data_pkl, 'wb')
        pickle.dump([test_x, test_y, all_imgs_paths, vids], f)
        f.close()
    return test_x, test_y, all_imgs_paths, vids
    
def calc_pr(label_pred, label_true):
    if len(label_pred) == 0:
        return 0.0, 0.0
    correct_num = 0.0
    for label in label_pred:
        if label in label_true:
            correct_num += 1
    p = correct_num/len(label_pred)
    r = correct_num/len(label_true)
    return p,r

def get_img_vec(file_path):
    print file_path
    cmd="cd /data8/piggywang/video/actor/face_lib/bin;./AILab_FaceLib " + file_path
    message = os.popen(cmd).readlines()
    vec = message[5].strip().split("\t")
    return np.array(vec)

def get_cos_img_vec(file_path, is_gray=False):
#     src = np.array([[30.2946, 51.6963],[65.5318, 51.5014],[48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041]],dtype=np.float32)
#
#     image = Image.open(file_path).convert('RGB')
#     bounding_boxes, landmarks = detect_faces(image)
#     if len(bounding_boxes) == 0 or len(landmarks) == 0:
#         img = image.resize((96, 112), Image.BILINEAR)
#     else:
#         croped_img = image.crop(bounding_boxes[0][0:4]).resize((96, 112), Image.BILINEAR)
#         #except:
#         #    croped_img = croped_img.resize((96, 112), Image.BILINEAR)
#         try:
#             bounding_boxes, landmarks = detect_faces(croped_img)
#         except:
#             croped_img = image.resize((96, 112), Image.BILINEAR)
#             bounding_boxes, landmarks = detect_faces(croped_img) 
#         dst_list = []
#         try:
#             landmark = landmarks[0]
#             for i in range(5):
#                 dst_list.append([landmark[i], landmark[i+5]])
#             if len(dst_list) != 5:
#                 print 'error', file_path
#             dst = np.array(dst_list)
#             tform = trans.SimilarityTransform()
#             tform.estimate(dst, src)
#             M = tform.params[0:2,:]
#             warped = cv2.warpAffine(np.asarray(image),M,(96, 112), borderValue = 0.0)
#             img = Image.fromarray(np.uint8(warped))
#         except:
#             img = croped_img
      

    img = Image.open(file_path).convert('RGB')
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
    print img.size
    img, img_ = transform(img), transform(F.hflip(img))
    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')
    ft = torch.cat((model(img), model(img_)), 1)[0].to('cpu')
    #return model(img)[0].to('cpu').data.numpy()
    return ft.data.numpy()
def get_imgs_vec_with_path(imgs_paths):
    emb = []
    for img_path in imgs_paths:
        if model_name == 'facenet':
            vec = get_img_vec(img_path)
        elif model_name == 'cosface':
            vec = get_cos_img_vec(img_path)
        emb.append(vec)
    return np.array(emb)

if __name__ == '__main__':
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    vid_actors_file = sys.argv[3]
    
    dis_threshold = 0.85
    
    #load cos_face model
    model_name = 'cosface'
    model = net.sphere().to('cuda')
    model_path = 'checkpoint/CosFace_38_checkpoint.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    train_x, train_y = load_train_data(train_data_path)
    test_x, test_y, all_imgs_paths, vids = load_test_data(test_data_path, vid_actors_file)
    print train_x.shape
    d = 128
    index = faiss.IndexFlatL2(d)
    index.add(train_x)

    k = 5
    predict_y = [] 
    sum_p = 0.0
    sum_r = 0.0
    for i in range(len(test_x)):
        one_video_x = test_x[i]
       
        D, I = index.search(one_video_x, k)
        predict_labels, distances = get_predict_result(D, I, train_y)
        #print predict_labels
        #print distances

        one_video_labels = test_y[i]
        predict_labels_dict = {}
        one_predict_y = []
        for j in range(len(predict_labels)):
            one_pre = predict_labels[j]
            if distances[j] > dis_threshold :
                continue
            if one_pre in predict_labels_dict:
                predict_labels_dict[one_pre] += 1
            else:
                predict_labels_dict[one_pre] =1
        print predict_labels_dict
        print one_video_labels
        #for key, value in predict_labels_dict.items():
        #    if value*7 > len(predict_labels) :
        #        one_predict_y.append(key)
        sorrted_predict_labels_dict = sorted(predict_labels_dict.items(),key = lambda x:x[1],reverse = True)
        for iii in range(1):
            key = sorrted_predict_labels_dict[iii][0]
            value = sorrted_predict_labels_dict[iii][1]
            if value<3: continue
            one_predict_y.append(key)
        predict_y.append(one_predict_y)
        p, r = calc_pr(one_predict_y, one_video_labels)
        sum_p += p
        sum_r += r
        tmp = [str(a) for a in one_predict_y]
        print vids[i] + "\t" + ",".join(tmp) + "\t" + str(p) + "\t" + str(r)
    avg_p = sum_p/len(test_x)
    avg_r = sum_r/len(test_x)
    print "example_num = %d" % (len(test_x))
    print "avg_P = %f" % (avg_p)
    print "avg_R = %f" % (avg_r)

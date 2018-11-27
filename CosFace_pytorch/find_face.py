from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import threading
import align.detect_face
import random
import shutil 
import pickle
#from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class MtcnnApi():
    def __init__(self):
        #mtcnn para
        self.pnet, self.rnet, self.onet = self.setup_mtcnn()
        self.minsize = 20  # minimum size of face
        #self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.threshold = [0.9, 0.95, 0.95]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.image_size = 160 # image_size
        self.image_size_model = 160 # image_size
        self.margin = 44

    def setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_memory_fraction = 1.0
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
        return pnet, rnet, onet

    def find_faces_with_image_path(self,image_path):
        #images = np.zeros((0, self.image_size_model, self.image_size_model, 3))
        try:
            img = misc.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            print(errorMessage)
            return [], []
        else:
            faces, faces_raw = self.find_faces(img)
            return faces, faces_raw

    def find_faces(self, image):
        faces_raw = []
        faces = []
        
        if image.ndim < 2:
            print('Unable to align img, ndim = %d' % ndim)
            return faces, faces_raw
        if image.ndim == 2:
            image = facenet.to_rgb(image)
        image = image[:, :, 0:3]

        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        #nrof_faces = bounding_boxes.shape[0]
        #print("nrof_faces " + str(nrof_faces))
        img_size = np.asarray(image.shape)[0:2]
        for det in bounding_boxes:
            face = np.zeros((1, self.image_size_model, self.image_size_model, 3))
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - self.margin / 2, 0)
            bb[1] = np.maximum(det[1] - self.margin / 2, 0)
            bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
            cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
            faces_raw.append(scaled)
            prewhiten = facenet.prewhiten(scaled)
            facecrop = facenet.crop(prewhiten, False, self.image_size_model)

            face[0,:,:,:] = facecrop
            faces.append(face)
        #self.save_img(faces_raw,"")
        return faces, faces_raw

    def save_img(self, images_img, path):
        for i,scaled in enumerate(images_img):
            misc.imsave( path + "_" + str(i) + ".png", scaled) 

class EmbClass():
    def __init__(self, name, ch_name, image_path, emb):
        self.name = name
        self.ch_name = ch_name
        self.image_path = image_path
        self.emb = emb
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

class FaceNetApi():
    def __init__(self):
        self.image_size = 160
        self.label_dict = self.load_dict("label_dict")
        self.path = "/data8/piggywang/video/mini_video/character/face_recognition/model"
        self.face_net_path = os.path.join(self.path, "face_net_model")
        self.sklearn_svm = os.path.join(self.path, "svm.bin")
        self.setup_facenet()
        
    def setup_facenet(self):
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession()
        facenet.load_model(self.face_net_path)
        self.class_names = None
        with open(self.sklearn_svm, 'rb') as infile:
            self.svm = pickle.load(infile)

    def load_dict(self, path):
        name_dict = {}
        with open(path) as f:
            for line in f:
                lines = line.strip().split("\t")
                if len(lines) < 2:
                    continue
                name_dict[lines[1]] = lines[0]
        return name_dict

    def classifer(self, emb):
        prediction = self.svm.predict(emb)
        labels = []
        size = prediction.shape[0]
        for i in range(size):
            label = str(prediction[i])
            if self.label_dict.has_key(label):
                label = self.label_dict[label]
                labels.append(label)
            else:
                labels.append(self.label_dict[0])
        return labels

    def feature_extract(self, images):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # Run forward pass to calculate embeddings
        #print('Calculating features for images')
        #images = facenet.load_data([pic_path], False, False, 160)
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        #emb = self.sess.run(embeddings, feed_dict=feed_dict)[0]
        emb = self.sess.run(embeddings, feed_dict=feed_dict)
        return emb

    def feature_extract_with_image_path(self, pic_paths):
        images = facenet.load_data(pic_paths, False, False, 160)
        emb = self.feature_extract(images)
        return emb

def detect_multi_face(img_list, label, index):
    f_detect = MtcnnApi()
    num_img = len(img_list)
    print ("num of imgs: {}".format(num_img))
    count = 0
    final_path = "/data8/piggywang/video/video_info/video_data_from_omg/{}_img".format(label)
    if not os.path.exists(final_path):
        os.mkdir(final_path)
    for img_path in img_list:
        count += 1
        faces, faces_raw = f_detect.find_faces_with_image_path(img_path)
        if faces:
            os.system("cp {} {}".format(img_path, final_path)) 
        if count % (num_img / 10) == 0:
            print ("Thread : {}  complete {}0%".format(index, count / (num_img / 10)))

class myThread(threading.Thread):
    def __init__(self, img_list, label,  index):
        threading.Thread.__init__(self)
        self.img_list = img_list
        self.index = index
        self.label = label
    def run(self):
        detect_multi_face(self.img_list, self.label, self.index)
        
if __name__ == '__main__':
    root_dir_path = "/data8/piggywang/video/video_info/video_data_from_omg/frame_baby_10_3w"
    thread_1 = ""
    thread_2 = ""
    thread_3 = ""
    thread_4 = ""
    thread = [thread_1, thread_2, thread_3, thread_4]
    for index, label in enumerate([0, 1, 2, 4]):
        file_name = str(label) + "_dirname"
        fo = open(file_name)
        all_img_path = []
        for line in fo:
            dir_name = line.strip().split("\t")[0]
            try:
                img_list = os.listdir(os.path.join(root_dir_path, dir_name))
                for item in img_list:
                    img_path = os.path.join(root_dir_path, dir_name, item)
                    all_img_path.append(img_path)
            except Exception as e:
                print ("error:{}".format(e))
        thread[index] = myThread(all_img_path, label ,index)
    for item in thread:
        item.start() 
    for item in thread:
        item.join()
     



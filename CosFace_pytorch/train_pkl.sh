rm ../dataset/V1_cosface_data.pkl
rm ../dataset/actor_imgs_multi_0910_test300_cosface_data.pkl
CUDA_VISIBLE_DEVICES=$1 nohup python faiss_test_prf.py /data1/aipd_tuijian/charlesliu/dataset/V1_new /data1/aipd_tuijian/charlesliu/dataset/actor_imgs_multi_0910_test300_new vid_actors_0910_test300 > temp_log &


#python get_list.py /data1/aipd_tuijian/charlesliu/dataset/CASIA-WebFace /data1/aipd_tuijian/charlesliu/dataset/CASIA-WebFace-112X96.txt

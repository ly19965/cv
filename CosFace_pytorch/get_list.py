import os
import sys

'''
Usage
python get_list.py data_path list_name
'''

data_path = sys.argv[1]
list_name = sys.argv[2]

all_identities_name = os.listdir(data_path)
fo = open(list_name, 'w')
for index_1, identity in enumerate(all_identities_name):
    identity_folder = os.path.join(data_path, identity)
    try:
        for index_2, img_name in enumerate(os.listdir(identity_folder)):
            full_file_name = os.path.join(identity_folder, img_name)
            fo.write(full_file_name + ' ' + str(index_1) + '\n')
    except:
        pass
fo.close()


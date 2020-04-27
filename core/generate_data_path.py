import os
import math

if os.path.exists('./data_train_path.txt'):
    os.remove('./data_train_path.txt')

if os.path.exists('./data_test_path.txt'):
    os.remove('./data_test_path.txt')

data_path = './'
face_data_path = 'AFDB_face_dataset'
masked_face_data_path = 'AFDB_masked_face_dataset'
paths = [face_data_path, masked_face_data_path]

label_dict = {}
with open('./train_face_mask_list.txt', 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        label_dict[lines[i].strip()] = i

    for path in paths:
        data_path_str = os.path.join(data_path, path)
        # sub_path = os.listdir(path)
        for key in label_dict.keys():
            dir_path = os.path.join(data_path_str, key)
            if os.path.isdir(dir_path):
                label = label_dict.get(key)
                img_paths = os.listdir(dir_path)
                test_img_path = math.floor(len(img_paths) * 0.2)
                if test_img_path == 0:
                    test_img_path = 1
                if len(img_paths) == 1:
                    train_img_paths = img_paths
                    test_img_paths = img_paths
                else:
                    train_img_paths = img_paths[:-test_img_path]
                    test_img_paths = img_paths[-test_img_path:]
                with open('./data_train_path.txt', 'a+') as f:
                    for img_path in train_img_paths:
                        f.write(os.path.join(dir_path, img_path) + "," + str(label) + "\n")
                with open('./data_test_path.txt', 'a+') as f:
                    for img_path in test_img_paths:
                        f.write(os.path.join(dir_path, img_path) + "," + str(label) + "\n")


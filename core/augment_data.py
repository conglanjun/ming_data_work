import cv2
import random
import numpy as np
import os


def flip(img):
    h_flip = cv2.flip(img, 1)
    return h_flip


def rotation(img):
    # rotation
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), random.randint(1, 10), 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def affine(img):
    rows, cols = img.shape[:2]
    pos1 = np.float32([[50,50],[200,50],[50,200]])
    pos2 = np.float32([[50,50],[200,50],[100,200]])
    M = cv2.getAffineTransform(pos1,pos2)
    dst = cv2.warpAffine(img, M, (rows, cols))
    return dst

import re
if __name__ == '__main__':
    # img1 = 'flip_xxx'
    # img2 = 'rotation_xxx'
    # img3 = 'affine_xxx'
    # if re.match('flip_|rotation_|affine_', img1):
    #     print(img1)

    path_dir = "\\\\192.168.209.108\share\data"
    listdir = os.listdir(path_dir)
    print(listdir)
    for dir in listdir:
        dir_path = path_dir + "\\" + dir
        dir_paths = os.listdir(dir_path)
        for person_dir in dir_paths:
            person_path = dir_path + "\\" + person_dir
            # person_paths = os.listdir(person_path)
            with open('train_face_mask_list.txt', 'r') as fr:
                person_paths = fr.readlines()
            for image_path in person_paths:
                image = cv2.imread(person_path + "\\" + image_path)
                flip_image = flip(image)
                cv2.imwrite(person_path + "\\flip_" + image_path, flip_image)
                rotation_image = rotation(image)
                cv2.imwrite(person_path + "\\rotation_" + image_path, rotation_image)
                affine_image = affine(image)
                cv2.imwrite(person_path + "\\affine_" + image_path, affine_image)

    # img = cv2.imread("lena.png")
    # cv2.imwrite(filepahe, img, flag)










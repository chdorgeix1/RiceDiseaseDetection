import shutil
import os
import glob
import random
import cv2

def create_img_arr(directory):
    
    file_list = os.listdir()
    img_list = []
    for img in file_list:
        img1 = cv2.imread(img)
        img_list.append(img1)
    img_arr = np.asarray(img_list)
    

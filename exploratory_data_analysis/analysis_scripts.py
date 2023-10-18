import shutil
import os
import glob
import random
import cv2
import numpy as np



def create_img_list(directory):
    os.chdir(directory)
    file_list = os.listdir()
    img_list = []
    for img in file_list:
        img1 = cv2.imread(img)
        img_list.append(img1)
    return(img_list)
    

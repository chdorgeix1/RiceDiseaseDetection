import os
import cv2
import numpy as np

def import_datasets(main_directory, image_directories):
    img_lists = [[],[],[],[]]
    for img_dir in enumerate(image_directories):
        os.chdir(main_directory + img_dir[1])
        for img in os.listdir(main_directory + img_dir[1]):
            img1 = cv2.imread(img)
            img_lists[img_dir[0]].append(img1)
        os.chdir('..')
    return img_lists


def create_label_array(datasets_list):
    label_array = np.asarray([])
    for i in enumerate(datasets_list):
        label_array = np.append(label_array, np.full((len(i[1]),), i[0]))
    return label_array


def combine_lists(datasets_list):
    single_list = []
    for i in datasets_list:
        single_list += i
    return single_list


def lists_to_array(combined_list):
    combined_arr = np.asarray(combined_list)
    return combined_arr


def convert_pixel_values(input_arr):
    return input_arr/255
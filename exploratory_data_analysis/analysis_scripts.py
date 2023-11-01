import shutil
import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_img_list(directory):
    os.chdir(directory)
    file_list = os.listdir()
    img_list = []
    for img in file_list:
        img1 = cv2.imread(img)
        img_list.append(img1)
    return(img_list)
    
def brightness_hist_for_img_folder(directory):
    img_list = create_img_list(directory)
    
    avg_bright_list = []

    for img in img_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_bright_list.append(img[..., 2].mean())

    plt.hist(avg_bright_list, bins = 10)
    plt.show()
    return(img_list, avg_bright_list)


def show_bright_and_dim_imgs(bright_list, img_list):
    # Create a subplot with 1 row and 2 columns
    plt.figure(figsize=(6, 3))  # Adjust the figure size as needed
    
    x1 = int(np.argsort(bright_list)[-1:])
    x2 = int(np.argsort(bright_list)[:1])
    
    # Plot the first image in the first subplot
    plt.subplot(1, 2, 1)
    plt.imshow(img_list[x1])
    plt.title('Bright Image')

    # Plot the second image in the second subplot
    plt.subplot(1, 2, 2)
    plt.imshow(img_list[x2])
    plt.title('Dim Image')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()
    
    
def contrast_hist_for_img_folder(directory):
    img_list = create_img_list(directory)
    
    avg_contrast_list = []

    for img in img_list:
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_contrast_list.append(img_grey.std())
    
    plt.hist(avg_contrast_list, bins = 10)
    plt.show()
    
    return(img_list, avg_contrast_list)

def show_high_low_contrast_imgs(contrast_list, img_list):
    # Create a subplot with 1 row and 2 columns
    plt.figure(figsize=(6, 3))  # Adjust the figure size as needed
    
    x1 = int(np.argsort(contrast_list)[-1:])
    x2 = int(np.argsort(contrast_list)[:1])
    
    # Plot the first image in the first subplot
    plt.subplot(1, 2, 1)
    plt.imshow(img_list[x1])
    plt.title('High Contrast Image')

    # Plot the second image in the second subplot
    plt.subplot(1, 2, 2)
    plt.imshow(img_list[x2])
    plt.title('Low Contrast Image')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()
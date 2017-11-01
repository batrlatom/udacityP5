import keras
import keras.models as models
from keras import metrics
import sys
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers import BatchNormalization,Input
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Convolution2D
from keras.callbacks import ModelCheckpoint
#from utils import INPUT_SHAPE, batch_generator, resize_normalize, pandas_split
from keras.layers import Cropping2D
from keras.models import load_model
from keras import metrics
from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.feature import hog
import re
from moviepy.editor import *

from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from scipy.ndimage.measurements import label


import cv2
import numpy as np
import json

import math
import h5py
import glob
from tqdm import tqdm
import scipy
from scipy import misc
import argparse, os
import pandas as pd

import matplotlib.pyplot as plt
from keras import utils as np_utils
import sklearn
import argparse
import matplotlib.image as mpimg

import pickle




def load_object(name):
    with open(name, 'rb') as input:
        data = pickle.load(input)
        return data


######################################################################################
#   Load RGB images from a file
######################################################################################
def load_image(data_dir, image_file):
    return cv2.imread(os.path.join(data_dir, image_file.strip()))


######################################################################################
#   Convert the image from RGB to YUV (This is what the NVIDIA model does)
######################################################################################
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)



# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 255, 0), thick=4):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'CAR '+str(car_number),(np.min(nonzerox) ,np.min(nonzeroy) - 10), font, 0.66,(0,255,0),2,cv2.LINE_AA)
    # Return the image
    return img



def get_windows(image):
    windows_32 = slide_window(image, x_start_stop=[int(image.shape[1]*0.25), int(image.shape[1]*0.75)], y_start_stop=[int(image.shape[0]*0.56), int(image.shape[0]*0.72)],
                xy_window=(32, 32), xy_overlap=(0, 0))

    windows_48 = slide_window(image, x_start_stop=[int(image.shape[1]*0.15), int(image.shape[1]*0.85)], y_start_stop=[int(image.shape[0]*0.56), int(image.shape[0]*0.72)],
                xy_window=(48, 48), xy_overlap=(0.5, 0.5))


    windows_64 = slide_window(image, x_start_stop=[int(image.shape[1]*0.1), int(image.shape[1]*0.9)], y_start_stop=[int(image.shape[0]*0.56), int(image.shape[0]*0.9)],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))

    windows_80 = slide_window(image, x_start_stop=[int(image.shape[1]*0.05), int(image.shape[1]*0.95)], y_start_stop=[int(image.shape[0]*0.56), int(image.shape[0]*0.9)],
                    xy_window=(80, 80), xy_overlap=(0.5, 0.5))


    windows_96 = slide_window(image, x_start_stop=[0, int(image.shape[1])], y_start_stop=[int(image.shape[0]*0.56), int(image.shape[0]*0.9)],
                    xy_window=(96, 96), xy_overlap=(0.6, 0.4))

    windows_112 = slide_window(image, x_start_stop=[int(image.shape[1]*0), int(image.shape[1])], y_start_stop=[int(image.shape[0]*0.56), int(image.shape[0]*0.9)],
                    xy_window=(112, 112), xy_overlap=(0.6, 0.4))


    windows_128 = slide_window(image, x_start_stop=[int(image.shape[1]*0), int(image.shape[1])], y_start_stop=[int(image.shape[0]*0.56), int(image.shape[0])],
                    xy_window=(128, 128), xy_overlap=(0, 0))

    wins_to_search = windows_64 + windows_80 + windows_96 + windows_48 + windows_112 #+ windows_128
    print("we are searching in ... " + str(len(wins_to_search)) + "windows")
    return  wins_to_search


def process_image(image):

    # scale data and convert to yuv
    #scaler = load_object("scaler.p")
    image = cv2.undistort(image, mtx, dist, None, mtx)
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv = yuv * (1.0/255) - 0.5

    positive_occurence = []

    # loop through all windows
    for idx, screenshot in enumerate(get_windows(image)):

        # building patch
        win_width = screenshot[1][1] - screenshot[1][0]
        screen = np.array([win_width, win_width])
        screen = np.copy(yuv[screenshot[0][1]:screenshot[1][1],screenshot[0][0]:screenshot[1][0], :])
        screen = cv2.resize(screen, (64, 64))

        # only i we use HoG classifier - but it not work as expected :/
        """
        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        X = np.zeros([1, 8460])
        features =  get_hog_features(screen[:, :, 0], orient,pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        features2 = get_hog_features(screen[:, :, 1], orient,pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        features3 = get_hog_features(screen[:, :, 2], orient,pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        features4 = color_hist(screen)
        features5 = bin_spatial(screen)

        final_features = np.concatenate((features, features2, features3, features4, features5)).ravel()
        X[0] = final_features
        #final_features = np.expand_dims(final_features, axis=0)
        scaled_X = scaler.transform(X)


        pred = model.predict(scaled_X)

        """

        # predict patch
        pred = model.predict(np.expand_dims(screen, axis=0))

        # if prediction is car, we add window among positive occurences
        if (pred[0][0] < 0.5) & (pred[0][1] > 0.5):
            positive_occurence.append(screenshot)

    # make image with all positive boxes for testing purposes
    boxes = draw_boxes(image, positive_occurence)

    # building emptz heatmap frame, add heat from current predictions
    # get rid of false positives by thresholding
    heatmap = np.zeros((720, 1280, 3), dtype=np.int)
    heatmap = add_heat(heatmap, positive_occurence)
    heatmap = apply_threshold(heatmap, 2)

    # add heatmap to time buffer, make a mean of the buffers
    old_heatmap.append(heatmap)


    heatmap_frame = np.mean(old_heatmap, axis=0)


    # get rid of the oldest frame
    if len(old_heatmap) > heatmap_buffer_len:
        old_heatmap.pop(0)

    # prepare final heatmap and make labels
    #heatmap_frame = np.clip(heatmap_frame, 0, 255)



    # Find final boxes from heatmap using label function

    #kernel = np.ones((128,128),np.float32)/(128**2)
    #heatmap_frame = cv2.bilateralFilter(heatmap_frame,15,75,75)
    heatmap_frame = cv2.GaussianBlur(heatmap_frame,(15,15),0)

    heatmap_frame = apply_threshold(heatmap_frame, 1)

    #heatmap_frame = cv2.filter2D(heatmap_frame,-1,kernel)
    labels = label(heatmap_frame)
    #print(labels)

    # if we have enought history in the heatmap buffer, we can construct
    # final boxes around cars
    if len(old_heatmap) == heatmap_buffer_len:
        # make boxes around cars
        draw_img = draw_labeled_bboxes(np.copy(image), labels)


        # adding heatmap to frame - just for graphic effect
        draw_img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

        draw_img = 1 * draw_img + heatmap_frame * [0, 0 , 10]

        draw_img  = np.clip(draw_img, 0, 255)
        #draw_img_rgb = cv2.addWeighted(draw_img, 1, heatmap_frame, 1)

        draw_img_rgb = 1 * draw_img_rgb + heatmap_frame * [10, 0 , 0]
        draw_img_rgb  = np.clip(draw_img_rgb, 0, 255)



        #draw_img_rgb = cv2.add(draw_img_rgb, heatmap_frame)


        boxes = cv2.cvtColor(boxes, cv2.COLOR_BGR2RGB)

        # save images to show realtime progress
        cv2.imwrite("outputs/heat.jpg", heatmap_frame *[0,0,5])
        cv2.imwrite("outputs/final.jpg", draw_img_rgb)
        cv2.imwrite("outputs/boxes.jpg", boxes)

    else:
        draw_img = image

    return draw_img



def extract_frames(movie, times, imgdir):
    clip = VideoFileClip(movie)
    for t in times:
        imgpath = os.path.join(imgdir, '{}.png'.format(t))
        clip.save_frame(imgpath, t)





def camera_calibration(path):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    nx = 9
    ny = 6
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(path)

    print("calibrating camera")
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        #print(idx)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    print("data points aquired")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


    return mtx, dist


######################################################################################
# main function
######################################################################################
if __name__ == '__main__':


    model = None
    old_heatmap = []
    heatmap_buffer_len = 10

    #calibrate camera
    mtx, dist = camera_calibration('camera_cal/calibration*.jpg')
    # load model, model name is first parameter
    model = load_model(sys.argv[1])

    # second parameter is input video, third is output video
    out_video = sys.argv[3]
    clip = VideoFileClip(sys.argv[2]).subclip(1750,1755)
    # using function process_image to run a pipeline
    out_clip = clip.fl_image(process_image)
    out_clip.write_videofile(out_video, audio=False)

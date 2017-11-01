import keras
import keras.models as models
from keras import metrics

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




orient = 9
pix_per_cell = 8
cell_per_block = 2


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

######################################################################################
#    Generate training image give image paths and associated steering angles
######################################################################################
def batch_generator(image_paths, batch_size):
	names = glob.glob("./"+image_paths+"/**/*.png", recursive=True)
	images = np.empty([batch_size, 64, 64, 3])
	labels = np.empty([batch_size, 1])

	#print(names)
	while True:
		i = 0

		for index in np.random.permutation(len(names)):
			image = load_image("", names[i])

			images[i] = image
			#print(image.shape)
			labels[i] = 1


			i += 1
			if i >= batch_size:
				break



######################################################################################
#    Generate training image give image paths and associated steering angles
######################################################################################
def hog_batch_generator_recursive(image_paths, batch_size):
    names = glob.glob("./"+image_paths+"/**/*.png", recursive=True)

    orient = 9
    pix_per_cell = 16
    cell_per_block = 2
    while True:
        i = 0
        #print(len(names))
        images = np.zeros([batch_size, 3*324 , ])
        labels = np.zeros([batch_size])

        for index in np.random.permutation(len(names)):
            image = cv2.imread(names[index])
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # if image belongs to non-vehicles, it will get zero signal

            if re.search("non-vehicles" , names[index]):
                labels[i] = 0

            else:
                labels[i] = 1

            y = np_utils.to_categorical(labels, 2)
            #print(y)

            features =  get_hog_features(yuv[:, :, 0], orient,pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            features2 = get_hog_features(yuv[:, :, 1], orient,pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            features3 = get_hog_features(yuv[:, :, 2], orient,pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            #features4 = get_hog_features(gray, orient,pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            features4 = color_hist(yuv)
            #print(features4.ravel())

            final_features = np.concatenate((features, features2, features3)).ravel()

            #final_features =
            #print(final_features.shape)
            #print(features.shape)


            images[i] = final_features
            #print(final_features.shape)
            #print(features4.shape)
            #np.concatenate(final_features, features4)


            #images[i] *= 1/images[i].max()
            #print(images[i])

            i += 1
            if i >= batch_size:
                break

            # scale data
            X_scaler = StandardScaler().fit(images)
            scaled_X = X_scaler.transform(images)
            #print(labels )

            yield (images, y)

######################################################################################
# Return data generators
######################################################################################
def datagens():
        # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest', rescale=1./255,
        preprocessing_function=rgb2yuv)




    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            directory='data',  # this is the target directory
            target_size=(64, 64),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels


    return train_generator




######################################################################################
# Define a function to compute binned color features
######################################################################################
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

######################################################################################
# Define a function to compute color histogram features
######################################################################################
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features




######################################################################################
# Define a function to return HOG features and visualization
######################################################################################
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features



######################################################################################
# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
######################################################################################

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


######################################################################################
# Returns model with dense NN
######################################################################################
def create_autoencoder_model():

    model = Sequential()
    model.add(Dense(256, input_shape=(8460,)))
    model.add(Activation('elu'))
    model.add(Dense(128))
    model.add(Activation('elu'))

    model.add(Dense(64))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(16))
    
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(8))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))


    return model


######################################################################################
# Returns CNN model
######################################################################################
def create_convolution_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model



import keras
import keras.models as models
from keras import metrics
import random
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

from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils import *

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

import cv2
import numpy as np
from numpy import *
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
import pickle


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
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


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
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
######################################################################################
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        #for file in imgs:
        # Read in each one by one
        #image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(imgs)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features = np.concatenate((spatial_features, hist_features))
        # Return list of feature vectors
        return features


######################################################################################
#	Will train dataset, provide 2 batch generators - for training and validation
#	Each epoch is validated and if validation_acc is better than last best, it will save model into model.h5 file
######################################################################################
def train_model(model, args):
    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)

    # Helper: TensorBoard
    tensorboard = TensorBoard(log_dir='./output/')

    #image_lists = create_image_lists("data")
    #print(image_lists)
    #train_generator, validation_generator = get_generators(image_lists, "data")
    #print(train_generator.classes)

    #model = load_model('model.h5')
    checkpoint = ModelCheckpoint('model.h5', monitor='acc', verbose=1, save_best_only = True, mode='auto')
    callbacks_list = [checkpoint, early_stopper, tensorboard]

    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics=['acc'])
    #gen = datagens()
    gen = hog_batch_generator_recursive("data", 64)

    #print("clases are > "+str(gen.classes[25000]))
    #print(gen.classes)
    #print(gen.class_indices)

    # there are two nested generators, one is for training data and second is for validation data
    model.fit_generator(gen,
                        400,
                        nb_epoch,
                        max_q_size=10,
                        callbacks=[checkpoint],
                        verbose=1)


#########################################################################
# load data into array - will otput normalized images or HoG features
#########################################################################
def train_data_loader(mode = "CNN"):

    orient = 9
    pix_per_cell = 8
    cell_per_block = 2


    print('Read train images')
    path = os.path.join('..', 'data', 'imgs', 'train', 'c', '*.jpg')


    #files = glob.glob(path)
    image_paths = "data"
    files = glob.glob("./"+image_paths+"/**/*.png", recursive=True)
    if mode == "CNN":
        X_train = np.zeros([len(files), 64, 64, 3])
    else:
        X_train = np.zeros([len(files), 8460])

    y_train = np.zeros([len(files)])
    #print(files)
    i = 0
    j = 0
    for fl in files:
        image =  cv2.imread(fl)
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        #yuv = yuv * (1.0/255)

        if mode == "CNN":
            if np.random.random() > 0.5:

                yuv = cv2.resize(yuv,None,fx=np.random.random()+1, fy=np.random.random()+1, interpolation = cv2.INTER_CUBIC)
                yuv = yuv[0:64, 0:64, :]


                print("rescaled")
            X_train[i] = yuv * 1./255 - 0.5





        else:
            features =  get_hog_features(yuv[:, :, 0], orient,pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            features2 = get_hog_features(yuv[:, :, 1], orient,pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            features3 = get_hog_features(yuv[:, :, 2], orient,pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            features4 = color_hist(yuv)
            features5 = bin_spatial(yuv)
            final_features = np.concatenate((features, features2, features3, features4, features5)).ravel()
            X_train[i] = final_features

        # construct labels
        # if image belongs to non-vehicles, it will get zero signal
        if re.search("non-vehicles" , fl):
            y_train[i] = 0

        else:
            y_train[i] = 1
            j += 1

        i += 1

    # just print number of vehicles and non-vehicles images
    print("Number of vehicles > "+str(j))
    print("Number of non-vehicles > "+str(len(files) - j))

    if mode != "CNN":
        X_scaler = StandardScaler().fit(X_train)
        X_train = X_scaler.transform(X_train)

    # save scaler to file. We will us it in inference
    save_object(X_train, "scaler.p")


    return X_train, y_train


#########################################################################
# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
#########################################################################
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

#########################################################################
# create dense network
#########################################################################
def create_autoencoder_model():

    model = Sequential()
    #model.add(Dense(256, input_shape=(3*324 , )))
    model.add(Dense(1024, input_shape=(8460,)))
    model.add(Activation('elu'))

    model.add(Dense(512))
    model.add(Activation('elu'))

    model.add(Dense(256))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))

    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))


    return model

#########################################################################
# create CNN network
#########################################################################
def create_convolution_model():
	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same',
	                 input_shape=(64, 64, 3)))
	model.add(Activation('elu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('elu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('elu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('elu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('elu'))
	model.add(Dropout(0.5))
	model.add(Dense(2))
	model.add(Activation('softmax'))

	return model


#########################################################################
# save scaler to pickle object
#########################################################################
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


#########################################################################
# main training loop
#########################################################################
def train_model(model, args, mode = "CNN"):
        # Helper: Stop when we stop learning.
        early_stopper = EarlyStopping(patience=10)

        # Helper: TensorBoard
        tensorboard = TensorBoard(log_dir='./output/')

        # load data
        X, y = train_data_loader(mode)
        y = np_utils.to_categorical(y)

        # define checkpoint and callbacks
        checkpoint = ModelCheckpoint('outputs/model.h5', monitor='val_acc', verbose=1, save_best_only = True, mode='auto')
        callbacks_list = [checkpoint, early_stopper, tensorboard]

        # compile model
        model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics=['acc'])

        # start main training loop with early stopper, using 25% data as validation, shuffled
        print("start training ...")
        hist = model.fit(X, y, batch_size=256, epochs=1000, verbose=1, callbacks=callbacks_list, validation_split=0.25, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
        print(hist.history)




######################################################################################
# main function
######################################################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to the folder of run images", type=str)
    args = parser.parse_args()
    model = create_convolution_model()####create_autoencoder_model()
    #train_model(model, args)
    mode = "CNN"

    if mode == "CNN":
        model = create_convolution_model()####create_autoencoder_model()
    else:
        create_autoencoder_model()

    train_model(model, args, mode)

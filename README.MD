[classifier]: ./outputs/output-hog.jpg "classifier"
[boxes]: ./outputs/boxes.jpg "boxes"
[boxes2]: ./outputs/boxes2.jpg "boxes2"
[boxes3]: ./outputs/boxes3.jpg "boxes3"
[boxes4]: ./outputs/boxes4.jpg "boxes4"

[heatmap]: ./outputs/heat.jpg "heatmap"

[final]: ./outputs/final.jpg "final"
[final3]: ./outputs/final3.jpg "final3"
[final4]: ./outputs/final4.jpg "final4"

[grid]: ./outputs/grid.jpg "grid"




# Vehicle detection and tracking project

Final project of the first term of Udacity self-driving cars nanodegee is to find car position in the video stream.
To complete the project, we have to meet several goals:
* Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.
* Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
* Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?
* Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?
* Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
* Describe how (and identify where in your code) you implemented ßsome kind of filter for false positives and some method for combining overlapping bounding boxes.


# 1. Included files
* train.py - contains the script to create and train the model
* inference.py - process video and recognize vehicles
* model.h5 - contains a trained nn classifier
* project_video.mp4 - project video
* processed_video.mp4 - contains processed video
* readme.md summarizing the results


# 2. Running the code
There are two main scripts - train.py and inference.py , first one is for training and second is for inference.


# 2.1 Training 
Python 3.5 is required. You need to download [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) datasets and paste them into the folder named "data". Run script with ```python train.py hog data``` or ```python train.py cnn data```


# 2.2 Inference
Run script with ```python inference.py model.h5 project_video.mp4 processed_video.mp4```. 


# 3. Training data
I used all training data provided by Udacity. It is about 25000 64x64x3 images of vehicles and non-vehicles in different lightning settings and configuration. Images are separated into vehicle and non-vehicle classes. I implemented few types of data loading. I tried datagenerators among with loading whole data as numpy array. 
But for convenience at the end, I stick with the classical approach. Loaded sata was shuffled and split into training and validation sets. Size of the validation set is 25% of data, training was done on about 18750 images.  Half of images was augmented by random upscalling and cropping back to its original size. Resizing was very helpfull to get better results. Without it, my classifier was sometimes unable to keep a track on vehicles in between window sizes. 


# 4. HoG Features
During the feature extraction, program convert all images into yuv colors. I used standard settings from lectures, which get features for 9 directions, have 8 pixels per cell and 2 cells per each block. This settings give us patches of 64x64 pixels. Features were extracted separately for each color channel. If program use HoG features together with color histograms and binned color features, we ravel and concatenate all feature vectors. Because we concatenated features from different domains, we need to normalize data with data scaler. After all images are scaled, scaler is saved into pickle filename - scaler. I also used color histogram and color spatial binning. Final feature vector had a size of 8460. 


# 5. Classifier

I used two types of classifier. First is neural network with dense layers, where input is extracted feature vector. Second is convolutional neural network, which uses
whole image converted to yuv color space and scaled to floats in interval (-1, 1). 

Both classifiers were trained with the batch size of 128 and with the AdamOptimizer. I implemented saving only on best checkpoint together with early stopper. If validation accuracy stops to climb, we exit training proces to prevent further overfitting. Both classifiers achieved validation accuracy over 93%.


NN model:
```
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
```




CNN model:
```
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 3)))
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
```


# 6. Sliding window search
Before we start to search for vehicles, we undistort camera frame with function camera_calibration(path). Calibration points are in directory "./calibration".
Then, undistorted image is converted into YUV. I implemented just naive sliding window search with window sizes of 48, 64, 80, 96, 112 and 128 pixels. I found that overlaps of 0.6 in x and 0.4 in y directions works great. I had some problems to get stable track of car in some scales. Random upscalling of training images during training phase helped us to solve this problem. 

We are searching in the lower half of the picture. We are searching in 606 windows in each frame. 
I noticed problem of the delay. If some vehicle is moving very fast in the opposite direction, it can take less than 10 frames to reach us and we will not be able to detect it.


Searching grid is shown bellow together with some examples of positive detections
![grid][grid]
![boxes][boxes]
![boxes2][boxes2]
![boxes3][boxes3]



# 7. Heatmap averaging
I used list of heatmaps with the depth of 10 frames. In each frame, I treshold heatmap to get rid of false positives and add heatmaps from all frames in buffer.
Final heatmap was constructed by averaging whole history of frames, then using gaussian blur with kernel size of 15. Finally, blurred heatmap was thresholded again. Main problem of this pipleine section was to come with good threshold, because you can have lot of multiple detections on vehicle in front of us, but just few on vehicle in distance. If you have small threshold, car in the distance is detected fine, but car in front of us will have big bounding box. If the threshold is big, car in the distance can disappear. At the end, after setting up good grid, I selected threshold 1. By averaging last 10 frames, we also have a problem of heatmap drifting. Heatmap is constructed from history, so mean will always be something like 5 frames behind the car itself. 

![heatmap][heatmap]



# 8. Drawing boxes around cars
Averaged heatmap is converted into labels - cars. Each bounding box was created from labels matrix in a simple way. We just used top-left and bottom-right corners of each label. 

![final][final]
![final3][final3]
![final4][final4]



# 7. Final run 
Final run can bee seen in the video.mp4 file. 
Even validation accuracy was high for both type of classifiers, real settings shows that CNN works much better than network with HoG features. Since I had big problem to tune HoG classifier in real settings, final video was created with CNN network. 

[![Watch the video](https://github.com/batrlatom/udacityP5/blob/master/outputs/final.jpg?raw=true)](https://github.com/batrlatom/udacityP5/blob/master/outputs/output_video.mp4)



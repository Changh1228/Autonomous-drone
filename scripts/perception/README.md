# Perception
## Introduction
The goal of this module is to identify the traffic signs among 16 different traffic signs. Once detected they need to be positioned in a global map so that at the end of the challenge a CSV file with the ID of each traffic sign and its global map coordinates had to be presented to the professors so they could evaluate the precision of our perception and localization modules. One kind of traffic sign could appear multiple times in the circuit but they should be separated at least 4 meters between same signs.
## Approach taken
This module subscribes to the output image of the camera, which was previously <a href="http://wiki.ros.org/camera_calibration">calibrated</a> to accomplish better results mainly in the extremes of the image. Once the image is converted from ROS format to OpenCV, it is applied some filters in order to determine the contours of the image.

Once a list of contours is obtained, each one is tested on how much of a circle, or a rectangle or a triangle it is, by looking at different parameters, such as number of vertex (several ways to achieve this step can be found on the internet like this <a href="https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/">one</a>). If the contour fits the requirements of one of the three shapes, a color filter step will be applied, which will test that the contour has a certain proportion of a color. This step is not very discriminative in order to avoid that by changing the environment (darker or lighter places, or with shadows) results vary dramatically.

Once the good contours are obtained, they are fed to a neural network implemented using Keras and Tensorflow as backend. The NN is pretty simple and is composed by some convolutional layers with small filter outputs in order to avoid tuning too much parameters (the dataset is not big enough). The NN was trained using a [dataset](dataset) created from taken a lot of pictures of the signs with the crazyflie camera in different situations and distances to the sign. In order just to have a 64-by-64 image of the sign, this was putted on a flat wall where the only contour was the sign so usign the first part of the algorithm the minimum rectangle surrounding the sign could be cropped out from the whole image.

It was decided just to use one NN to classify the 15 different signs (some crap labels were added for the three shapes to make the system more robust).
Once the contour is classified, it is time to locate it in the map frame. To accomplish that <a href="https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html">solvePnP()</a> was used, which estimates the object pose given a set of object points and their corresponding image projections. Once we have the pose in the camera frame, this is translated to the map frame using <a href="http://wiki.ros.org/tf">tf</a> and saved in a matrix.

The next and final step consists on clustering the data points in order to provide an estimate pose for each sign. The way it was done was adding each new input into a matrix and checking if there was a near symbol of the same kind saved from before. If it was, we would change the average position of the corresponding cluster with the new sample added and increase the number of elements of the cluster by one. On the other hand, if the previous same signs are far away (more than 2 meters) a new cluster for that sign is created. Every 500 samples the matrix is checked and all the clusters that have less than a certain number of members are removed.
When the program is stopped the matrix is converted to CSV format.

## Scripts description

## Demo with provided rosbag
Click image to play the video => [![Watch the video](https://i.ytimg.com/vi/7OHdB9czP5c/hqdefault.jpg?sqp=-oaymwEZCNACELwBSFXyq4qpAwsIARUAAIhCGAFwAQ==&rs=AOn4CLDpKAXlbvoTF-pU6inIFIdQptRv6g)](https://www.youtube.com/watch?v=7OHdB9czP5c)


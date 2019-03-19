import cv2
import numpy as np
from os import listdir

img_rgb = cv2.imread('0triangle.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#threshold = np.zeros((1, len(listdir('templates/triangle/'))))
threshold = 0
sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(img_rgb, None)
for file in listdir('templates/triangle/'):
    image = cv2.imread('templates/triangle/' + file)
    resized_image = cv2.resize(image, (img_rgb.shape[0], img_rgb.shape[1]))
    kp_2, desc_2 = sift.detectAndCompute(image, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = 0
    ratio = 0.6
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_points = good_points + 1
    if (good_points>threshold):
        threshold = good_points
        idx = file

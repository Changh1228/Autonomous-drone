#!/usr/bin/env python
import sys
import rospy
import cv2
import math
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError

from os import listdir

from localizer import Localizer
from nnclassifier import NNClassifier

index_triangle = 0
index_rectangle = 0
index_cercle = 0

class QueuedImage:
    def __init__(self, image, timestamp):
        self.image = image
        self.timestamp = timestamp


class ImageConverter:
    def __init__(self, bridge):
        self.bridge = bridge
        self.raw_image_sub = rospy.Subscriber("/cf1/camera/image_raw", Image, self.callback)
        #self.raw_image_sub = rospy.Subscriber("/cf1/aruco/result", Image, self.callback)

        rospy.loginfo("Subscription to camera feed established")

        self.timediff = 1000 #ms

        self.queue = []
        rospy.loginfo("Image converter initialized")

    def callback(self, data):
        # Convert the image from ROS to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") # BGR space input image

            # rosrun image_transport republish compressed in:=/cf1/camera/image_raw raw out:=/cf1/camera/image_raw

        except CvBridgeError as e:
            print(e)

        if len(self.queue) == 0:
            self.queue.append(QueuedImage(cv_image, data.header.stamp))
        else:
            last_queued_image = self.queue[len(self.queue )- 1]
            
            if data.header.stamp.to_sec() > last_queued_image.timestamp.to_sec() + self.timediff * 0.001:
                self.queue.append(QueuedImage(cv_image, data.header.stamp))

class ImageWorker:

    def __init__(self, bridge):
        self.bridge = bridge
        self.result_img_pub = rospy.Publisher("/myresult", Image, queue_size=2)

        self.size_crop = 64
        self.localizer_signs = Localizer() # we could pass the tf_buf used for localizing ourselves
        self.nn_classifier = NNClassifier() # Initialize the NN classifier
        self.sign_map_list = [] # A list where all the signs will be saved with their respective position
        # everytime they are seen so afterwards I can do clustering

    def transform_tri(self, vector1, vector2, other_vec):
        """
        :param image: The image we work with
        :param vector1: First vertex of the longest side of the triangle observed
        :param vector2: Second vertex of the longest side of the triangle observed
        :param other_vec: The other vertex, the one we need to add the perpendicular line to find the other 2 points of
        quadrilater
        :return: A box fitting the triangle where two vertex of the box are two vertex of triangle and the following vertex
        triangle is in the middle of the counter side of box
        """

        # Add some margins to the found vertex in order to transform better the triangle
        if (other_vec[0][0] < vector1[0][0]):
            vector1[0][0] = vector1[0][0] + 2
        else:
            vector1[0][0] = vector1[0][0] - 2

        if (other_vec[0][0] < vector2[0][0]):
            vector2[0][0] = vector2[0][0] + 2
        else:
            vector2[0][0] = vector2[0][0] - 2

        if (other_vec[0][1] < vector1[0][1]):
            vector1[0][1] = vector1[0][1] + 2
        else:
            vector1[0][1] = vector1[0][1] - 2

        if (other_vec[0][1] < vector2[0][1]):
            vector2[0][1] = vector2[0][1] + 2
        else:
            vector2[0][1] = vector2[0][1] - 2


        # Adding the same vector of the longest side to the other_vec (half each side) point we obtain the 2 remaining points
        # quadrilater
        point1 = np.array(other_vec - (vector1 - vector2) / 2)
        point2 = np.array(other_vec + (vector1 - vector2) / 2)

        point1 = np.trunc(point1) # truncate to integers
        point2 = np.trunc(point2)

        box = np.array([
            [vector1[0][0], vector1[0][1]],
            [vector2[0][0], vector2[0][1]],
            [point1[0][0], point1[0][1]],
            [point2[0][0], point2[0][1]]], dtype="float32") # obtain a box with the 4 points to transform

        return box

    def process_image(self, cropped_image):
        """
        cropped_image = The cropped image after changing the perspective (IS IN BGR)
        outout:
            Normalized image both in mean and std and put it in the required format
            for the NN classification step (size 64 x 64 and specific tuple dimension)
        """
        res = cv2.resize(cropped_image, dsize=(self.size_crop, self.size_crop), interpolation=cv2.INTER_CUBIC)
        img = np.reshape(res, [1, self.size_crop, self.size_crop, 3])
        img = img * 1. / 255 # normalize the image because it was normalized while training
        _,std = cv2.meanStdDev(img)
        img = img * 1. / std # divide by std of the image
        return img


    ##############################
    #  Shapes detection helpers  #
    ##############################
    def detect_triangle(self, cnt):
        global index_triangle

        epsilon = 0.05 * cv2.arcLength(cnt, True) # 0.05 works for triangle
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        rectangle_points = cv2.minAreaRect(approx)
        min_rectangle = np.int0(cv2.boxPoints(rectangle_points))
        area_contour = cv2.contourArea(approx)
        corners = len(approx)

        #----------------------------------------- Check proportions ---------------------------------------------
        if corners == 3 and cv2.arcLength(cnt, True) < 250:
            # probably a triangle
            flag_closeness = 0
            mm = cv2.moments(cnt)  # cal center
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])
            for center in self.currentTri:
                k = np.square(center[1][0] - cx) + np.square(center[1][1] - cy)
                if k < 20:
                    flag_closeness = 1
                    break
            if flag_closeness == 0:
                box_triangle = self.transform_tri(approx[0], approx[1], approx[2])
                dst = np.array([
                    [0, 0],
                    [self.size_crop - 1, 0],
                    [self.size_crop - 1, self.size_crop - 1],
                    [0, self.size_crop - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(box_triangle, dst) # Obtain transformation matrix
                # transform the perspective of the sign to put it  as looking to the viewer
                warped_triangle = cv2.warpPerspective(self.my_image, M, (self.size_crop, self.size_crop))
                image_to_classify = self.process_image(warped_triangle)
                type_tr = self.nn_classifier.classify_nn("triangle", image_to_classify)  # returns the string saying to which
                # template it belongs to
                index_triangle = index_triangle + 1
                #cv2.imwrite('create_dataset/triangle/' + type_tr +"/"+ str(index_triangle) + 'triangle.jpg', warped_triangle)
                cv2.imwrite('results/triangle/' + type_tr +"/"+ str(index_triangle) + 'triangle.jpg', warped_triangle)

                if type_tr != "CRAP":
                    self.currentTri.extend([[min_rectangle, (cx, cy), type_tr]])  # save minimum rectangle and center
                    

    def detect_rectangle(self, cnt):
        global index_rectangle

        epsilon = 0.02 * cv2.arcLength(cnt, True) # for STOP detects 8 points ; 0.05 works for triangle
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area_contour = cv2.contourArea(approx)
        corners = len(approx)
        rectangle_points = cv2.minAreaRect(approx)
        min_rectangle = np.int0(cv2.boxPoints(rectangle_points))
        area_rectangle = cv2.contourArea(min_rectangle)
        if area_rectangle < 250: # TODO: IMPROVE?? Is to avoid rectangles of 0 area dividing in proportion_rect
            return
        proportion_rect = area_contour/area_rectangle
        thresh_rectangle = 0.80

        if (proportion_rect > thresh_rectangle): # TODO: Check if makes sense the preprocess
            # probably a rectangle
            mm = cv2.moments(min_rectangle)  # cal center
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])
            # check repeat contour
            flag_closeness = 0
            for center in self.currentRect:
                k = np.sqrt(np.square(center[1][0] - cx) + np.square(
                    center[1][1] - cy))  # don't consider rectangles very close to rectangles already found
                if k < 20:
                    flag_closeness = 1
                    break

            if flag_closeness == 0:
                # Make perspective transformation
                dst = np.array([
                    [0, 0],
                    [self.size_crop - 1, 0],
                    [self.size_crop - 1, self.size_crop - 1],
                    [0, self.size_crop - 1]], dtype="float32")

                # compute the perspective transform matrix and then apply it
                #self.currentRect.extend([[min_rectangle, (cx, cy), "rect"]])

                M = cv2.getPerspectiveTransform(min_rectangle.astype(np.float32), dst)
                warped_rectangle = cv2.warpPerspective(self.my_image, M, (self.size_crop, self.size_crop))  # TODO: Check if requires RGB or BGR
                image_to_classify = self.process_image(warped_rectangle)
                type_rect = self.nn_classifier.classify_nn("rectangle", image_to_classify)
                #cv2.imwrite('create_dataset/rectangle/' + type_rect + "/" + str(index_rectangle) + 'rectangle.jpg', warped_rectangle)
                index_rectangle = index_rectangle + 1
                cv2.imwrite('results/rectangle/' + type_rect + "/" + str(index_rectangle) + 'rectangle.jpg', warped_rectangle)

                if type_rect != "CRAP":
                    self.currentRect.extend([[min_rectangle, (cx, cy), type_rect]])

    def detect_cercle(self, cnt):
        global index_cercle

        epsilon = 0.01 * cv2.arcLength(cnt, True) # for STOP detects 8 points ; 0.05 works for triangle
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area_contour = cv2.contourArea(approx)
        corners = len(approx)

        # TODO  : Fit minimum ellipse
        thresh_ellipse = 0.85
        proportion_ellipse = 0 # just to initialize it

        if len(approx) >= 5: # If not fitEllipse gives error
            ellipse = cv2.fitEllipse(approx)
            (x, y), (MA, ma), angle = ellipse
            area_ellipse = 0.25*math.pi * MA * ma # MA length major axis and ma length minor axis ellipse. Divided by 4 as axes
            # and not semi-axes
            proportion_ellipse = area_contour/area_ellipse

        if (proportion_ellipse > thresh_ellipse) and (corners>6):
            mm = cv2.moments(cnt)  # cal center
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])
            flag_closeness = 0
            for center in self.currentCerc:  # check every kind circle
                #for center in type_cercle:
                k = np.sqrt(np.square(center[1][0] - cx) + np.square(
                center[1][1] - cy))  # don't consider cercles very close to cercles already found
                if k < 20:
                    flag_closeness = 1
                    break

            if flag_closeness == 0:
                rect_covering_cercle = cv2.minAreaRect(cnt) # TODO: Check if better "cnt" or "approx"
                box_cercle = cv2.boxPoints(rect_covering_cercle)
                box_cercle = np.int0(box_cercle)
                # Make perspective transformation
                dst = np.array([
                    [0, 0],
                    [self.size_crop - 1, 0],
                    [self.size_crop - 1, self.size_crop - 1],
                    [0, self.size_crop - 1]], dtype="float32")
                #self.currentCerc.extend([[box_cercle, (cx, cy), "cer"]])
                # compute the perspective transform matrix and then apply it
                M = cv2.getPerspectiveTransform(box_cercle.astype(np.float32), dst)
                warped_cercle = cv2.warpPerspective(self.my_image, M, (self.size_crop, self.size_crop)) # TODO: Check if requires RGB or BGR
                image_to_classify = self.process_image(warped_cercle)
                type_cercle = self.nn_classifier.classify_nn("cercle", image_to_classify)
                #cv2.imwrite('create_dataset/cercle/' + type_cercle + "/" + str(index_cercle) + 'cercle.jpg', warped_cercle)
                index_cercle = index_cercle + 1
                cv2.imwrite('results/cercle/' + type_cercle + "/" + str(index_cercle) + 'cercle.jpg', warped_cercle)

                if type_cercle != "CRAP":
                    self.currentCerc.extend([[box_cercle, (cx, cy), type_cercle]])  # save ellipse center and size


    ##############################
    #  End of detection helpers  #
    ##############################

    def publish(self, image_result):
        # Publish the image
        try:
            self.result_img_pub.publish(self.bridge.cv2_to_imgmsg(image_result, "rgb8"))
        except CvBridgeError as e:
            print(e)

    def detect_shape(self, queued_image): # queued_image is in BGR
        # Process image to detect shape
        self.my_image = queued_image.image
        hsv = cv2.cvtColor(self.my_image, cv2.COLOR_BGR2HSV)
        
        # ------------------- Apply Filters ----------------------
        img = cv2.bilateralFilter(hsv, 10, 75, 75)
        #img = cv2.bilateralFilter(self.my_image, 10, 75, 75)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = 50

        binaryImg = cv2.Canny(gray, thresh, 3*thresh)

        contours = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[
            1]  # modification to work in OpenCV 3.4
        # Filter contours by are size
        contours = [contours[i] for i in range(len(contours)) if
                    (cv2.contourArea(contours[i]) > 300 and cv2.contourArea(contours[i]) < 2400)]

        self.currentTri = [] # just to save the Triangles of this specific picture, empty for another picture
        self.currentRect = []
        self.currentCerc = []
        for cnt in contours:
            self.detect_cercle(cnt)
            self.detect_rectangle(cnt)
            self.detect_triangle(cnt)

        # IN CASE CERCLE ARE CLOSE TO RECTANGLE JUST SAVE CERCLE
        for center_circle in self.currentCerc:  # check every kind circle
            #for center_circle in species:  # check repeat contour
            for center_rect in self.currentRect:
                k = np.sqrt(np.square(center_rect[1][0] - center_circle[1][0]) + np.square(
                    center_rect[1][1] - center_circle[1][1]))
                if k < 20:
                    self.currentRect = [x for x in self.currentRect if not (x[0] == center_rect[0]).all()]
        # Draw contours in published image and find the position of object in "map" frame
        for cnt in self.currentRect:
            cv2.drawContours(self.my_image, [cnt[0]], -1, (255, 255, 0), 2)
            cv2.putText(self.my_image, cnt[2], cnt[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            #----------------------------
            sign_in_map = self.localizer_signs.find_location(cnt[1], cnt[0], queued_image.timestamp)
            if sign_in_map is not None:
                self.sign_map_list.extend([[cnt[2], (sign_in_map.point.x, sign_in_map.point.y, sign_in_map.point.z)]])
        for cnt in self.currentCerc:
            cv2.drawContours(self.my_image, [cnt[0]], -1, (255, 255, 0), 2)
            cv2.putText(self.my_image, cnt[2], cnt[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            #----------------------------
            sign_in_map = self.localizer_signs.find_location(cnt[1], cnt[0], queued_image.timestamp)
            if sign_in_map is not None:
                self.sign_map_list.extend([[cnt[2], (sign_in_map.point.x, sign_in_map.point.y, sign_in_map.point.z)]])
        for cnt in self.currentTri:
            cv2.drawContours(self.my_image, [cnt[0]], -1, (255, 255, 0), 2)
            cv2.putText(self.my_image, cnt[2], cnt[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            #----------------------------
            sign_in_map = self.localizer_signs.find_location(cnt[1], cnt[0], queued_image.timestamp)
            if sign_in_map is not None:
                rospy.loginfo("Sign: "+cnt[2]+", X: "+str(pose.position.x)+", Y: "+str(pose.position.y)+", Z: "+str(pose.position.z)+"")
                self.sign_map_list.extend([[cnt[2], (sign_in_map.point.x, sign_in_map.point.y, sign_in_map.point.z)]])

        self.result_img_pub.publish(self.bridge.cv2_to_imgmsg(self.my_image, "bgr8"))

        self.currentTri = [] # just to save the Triangles of this specific picture, empty for another picture
        self.currentRect = []
        self.currentCerc = []


def main(args):
    rospy.init_node('object_recognition', anonymous=True)
    rate = rospy.Rate(20) # 20hz

    bridge = CvBridge()
    ic = ImageConverter(bridge)

    iw = ImageWorker(bridge)

    try:
        while not rospy.is_shutdown():
            if len(ic.queue) > 0:
                iw.detect_shape(ic.queue[0])
                del ic.queue[0]
                
            rate.sleep()

        cv2.destroyAllWindows()
        print("Shutting down")
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main(sys.argv)
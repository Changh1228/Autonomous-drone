#!/usr/bin/env python
import sys
import rospy
import cv2
import math
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class QueuedImage:
    def __init__(self, image, timestamp):
        self.image = image
        self.timestamp = timestamp


class ImageConverter:
    def __init__(self, bridge):
        self.bridge = bridge
        self.raw_image_sub = rospy.Subscriber("/cf1/camera/image_raw", Image, self.callback)

        rospy.loginfo("Subscription to camera feed established")

        self.timediff = 1000 #ms

        self.queue = []
        rospy.loginfo("Image converter initialized")

    def callback(self, data):
        # Convert the image from ROS to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8") # RGB space input image
        except CvBridgeError as e:
            print(e)

        # Apply camera calibration correction when available

        # ------------------Filter--------------------------------------------------- #
        img = cv2.bilateralFilter(cv_image, 10, 75, 75)
        # bilateral Filter (best quality, slow?)
        # img = cv2.GaussianBlur(imgOri, (5, 5), 0)  # Gaussian Filter

        # Adding the enhanced image to the "waiting for process" queue
        if len(self.queue) == 0:
            self.queue.append(QueuedImage(img, data.header.stamp))
        else:
            last_queued_image = self.queue[len(self.queue )- 1]
            
            if data.header.stamp.to_sec() > last_queued_image.timestamp.to_sec() + self.timediff * 0.001:
                self.queue.append(QueuedImage(img, data.header.stamp))

class ImageWorker:
    def __init__(self, bridge):
        self.bridge = bridge
        self.result_img_pub = rospy.Publisher("/myresult", Image, queue_size=2)

    ##############################
    #  Shapes detection helpers  #
    ##############################

    def detectTri(self, cnt, triCnt, image_to_process):
        # -------- Polygon detection -------------------------------------------- #
        epsilon = 0.15 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # Analyse shape
        corners = len(approx)
        # Calculate space
        S1 = cv2.contourArea(cnt)
        if corners == 3 and S1 > 100:
            # choose contour base on corner and space
            mm = cv2.moments(cnt)  # cal center
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])
            # check repeat contour
            for center in triCnt:
                k = np.square(center[1][0] - cx) + np.square(center[1][1] - cy)
                if k < 20:
                    return triCnt, None
            # ------------- detect color in contour ------------------------------ #
            # make contour mask
            cntMask = np.zeros((480, 640), np.uint8)
            fillCnt = []
            fillCnt.append(cnt)
            cv2.drawContours(cntMask, fillCnt, -1, 255, -1)
            res = cv2.bitwise_and(image_to_process, image_to_process, mask=cntMask)
            hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
            # Red and yellow mask
            # Two edge of the Hue turnel
            lower1 = np.array([160, 10, 100])
            upper1 = np.array([180, 250, 250])
            redMask1 = cv2.inRange(hsv, lower1, upper1)
            lower2 = np.array([0, 10, 100])
            upper2 = np.array([40, 240, 240])
            redMask2 = cv2.inRange(hsv, lower2, upper2)
            redMask = redMask1 | redMask2
            prop = np.sum(redMask) / np.sum(cntMask)  # cal proportion of expected color
            if prop < 0.7:
                return triCnt, None
            
            # ---------- draw and save ROI --------------------------------------- #
            triCnt.extend([[approx, (cx, cy)]])  # save approx Polygon and center
            x, y, w, h = cv2.boundingRect(cnt)
            result_image_of_shape_detection = cv2.drawContours(image_to_process, cnt, -1, (0, 255, 0), 2)
            result_image_of_shape_detection = cv2.rectangle(result_image_of_shape_detection, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 1)
            result_image_of_shape_detection = cv2.putText(result_image_of_shape_detection, 'tri', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return triCnt, result_image_of_shape_detection
        return triCnt, None


    def detectRect(self, cnt, rectCnt, image_to_process):
        # -------- Polygon detection --------------------------------------------- #
        # (!!! repeat code, probably add tri and rect in same func)
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # Analyse shape
        corners = len(approx)
        # Calculate space
        S1 = cv2.contourArea(cnt)
        if corners == 4 and S1 > 100:
            # choose contour base on corner and space
            mm = cv2.moments(cnt)  # cal center
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])
            # check repeat contour
            flag = 0
            for center in rectCnt:
                k = np.square(center[1][0] - cx) + np.square(center[1][1] - cy)
                if k < 20:
                    return rectCnt, None
            # ------------- detect color in contour --------------------------- #
            # make contour mask
            cntMask = np.zeros((480, 640), np.uint8)
            fillCnt = []
            fillCnt.append(cnt)
            cv2.drawContours(cntMask, fillCnt, -1, 255, -1)
            res = cv2.bitwise_and(image_to_process, image_to_process, mask=cntMask)
            hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
            # Blue mask
            lower = np.array([100, 50, 50])
            upper = np.array([135, 240, 240])
            buleMask = cv2.inRange(hsv, lower, upper)
            prop = np.sum(buleMask) / np.sum(cntMask)  # cal proportion of expected color
            if prop < 0.7:
                return rectCnt, None

            # ---------- draw and save ROI ------------------------------------- #
            rectCnt.extend([[approx, (cx, cy)]])  # save approx Polygon and center
            x, y, w, h = cv2.boundingRect(cnt)
            result_image_of_shape_detection = cv2.drawContours(image_to_process, cnt, -1, (255, 255, 0), 2)
            result_image_of_shape_detection = cv2.rectangle(result_image_of_shape_detection, (x-5, y-5), (x+w+5, y+h+5), (255, 255, 0), 1)
            result_image_of_shape_detection = cv2.putText(result_image_of_shape_detection, 'rect', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            return rectCnt, result_image_of_shape_detection
        return rectCnt, None


    def detectCir(self, cnt, cirCnt, image_to_process):
        if len(cnt) < 20:
            return cirCnt, None
        S1 = cv2.contourArea(cnt)
        ell = cv2.fitEllipse(cnt)
        pos, size, angle = ell
        ex, ey = pos
        dx, dy = size
        S2 = math.pi*ell[1][0]*ell[1][1]
        if S2 < 1000:
            return cirCnt, None
        prop = S1 / S2
        if prop > 0.22:
            # check repeat contour
            mm = cv2.moments(cnt)  # cal center
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])

            for species in cirCnt:  # check every kind circle
                for center in species:  # check repeat contour
                    k = np.square(center[0][0] - cx) + np.square(center[0][1] - cy)
                    if k < 20:
                        return cirCnt, None

            # detect outliers(center of contour should close to ellipse)
            d = np.square(ell[0][0] - cx) + np.square(ell[0][1] - cy)
            if d > 20:
                return cirCnt, None

            GoF = 0  # Goodness Of Fit, the smaller the better (not sure)
            for point in cnt:
                posx = (point[0][0] - ex) * math.cos(-angle) - (point[0][1] - ey) * math.sin(-angle)
                posy = (point[0][0] - ex) * math.sin(-angle) + (point[0][1] - ey) * math.cos(-angle)
                GoF += abs(np.square(posx/dx) + np.square(posy/dy) - 0.25)
            # GoF = GoF / cnt.size
            if GoF > 10:
                return cirCnt, None

            # ------------- detect color in contour ------------------------------ #
            # make contour mask
            cntMask = np.zeros((480, 640), np.uint8)
            fillCnt = []
            fillCnt.append(cnt)
            cv2.drawContours(cntMask, fillCnt, -1, 255, -1)  # fill contour with white

            res = cv2.bitwise_and(image_to_process, image_to_process, mask=cntMask)
            hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
            # Red and yellow mask
            # Two edge of the Hue turnel
            lower1 = np.array([160, 10, 100])
            upper1 = np.array([180, 250, 250])
            redMask1 = cv2.inRange(hsv, lower1, upper1)
            lower2 = np.array([0, 10, 100])
            upper2 = np.array([40, 240, 240])
            ryMask = cv2.inRange(hsv, lower2, upper2) | redMask1

            # Blue mask
            lower = np.array([100, 50, 50])
            upper = np.array([150, 240, 240])
            blueMask = cv2.inRange(hsv, lower, upper)

            # classify based on color
            blueProp = np.sum(blueMask) / np.sum(cntMask)
            ryProP = np.sum(ryMask) / np.sum(cntMask)
            if blueProp > 0.7:
                # Blue
                flag = 1
            elif (blueProp > 0.2) and (ryProP > 0.2):
                # Red and blue
                flag = 2
            elif ryProP > 0.7:
                # Red and yellow
                flag = 3
            else:
                flag = 0
            # ---------- draw and save ROI ---------------------------------------- #
            if flag != 0:
                cirCnt[flag-1].extend([ell])  # save ellipse center and size
                result_image_of_shape_detection = cv2.ellipse(image_to_process, ell, (0, 255, 255), 1)
                result_image_of_shape_detection = cv2.drawContours(result_image_of_shape_detection, cnt, -1, (0, 255, 255), 2)
                x = int(ell[0][0])
                y = int(ell[0][1])
                result_image_of_shape_detection = cv2.putText(result_image_of_shape_detection, 'cir%d' % flag, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                return cirCnt, result_image_of_shape_detection
        return cirCnt, None

    ##############################
    #  End of detection helpers  #
    ##############################

    def publish(self, image_result):
        # Publish the image
        try:
            self.result_img_pub.publish(self.bridge.cv2_to_imgmsg(image_result, "rgb8"))
        except CvBridgeError as e:
            print(e)

    def detect_object(self, image_with_shape):
        # Process image to detect image
        # result_image_of_object_detection = process(image_with_shape)
        result_image_of_object_detection = None
        # Returning result
        return result_image_of_object_detection

    def detect_shape(self, queued_image):
        # Process image to detect shape
        gray = cv2.cvtColor(queued_image.image, cv2.COLOR_RGB2GRAY)
        thresh = 50
        # two different thresh, canny has better result
        # ret, thresh = cv2.threshold(gray, 150, 255, 0)
        binaryImg = cv2.Canny(gray, thresh, 3*thresh)
        rslt_img, contours, hierarchy = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        triCnt = []
        rectCnt = []
        cirCnt = [[], [], []]  # blue circle, red & yellow circle, red & blue circle
        # If a shape is detected -> Process image to detect object
        for cnt in contours:
            triCnt, result_image_of_shape_detection = self.detectTri(cnt, triCnt, queued_image.image)
            if result_image_of_shape_detection is None:
                rectCnt, result_image_of_shape_detection = self.detectRect(cnt, rectCnt, queued_image.image)
                if result_image_of_shape_detection is None:
                    # the stop sign can be treated as circle
                    cirCnt, result_image_of_shape_detection = self.detectCir(cnt, cirCnt, queued_image.image)
                    if result_image_of_shape_detection is None:
                        # Publish the original image
                        self.publish(queued_image.image)
                    else:
                        result_image_of_object_detection = self.detect_object(queued_image.image)
                        if result_image_of_object_detection is None:
                            self.publish(result_image_of_shape_detection)
                        else:
                            self.publish(result_image_of_object_detection)
                else:
                    result_image_of_object_detection = self.detect_object(queued_image.image)
                    if result_image_of_object_detection is None:
                        self.publish(result_image_of_shape_detection)
                    else:
                        self.publish(result_image_of_object_detection)
            else:
                result_image_of_object_detection = self.detect_object(queued_image.image)
                if result_image_of_object_detection is None:
                    self.publish(result_image_of_shape_detection)
                else:
                    self.publish(result_image_of_object_detection)

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
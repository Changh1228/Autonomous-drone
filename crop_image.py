import cv2
import numpy as np
import math
from os import listdir
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, Dropout, Activation, MaxPooling2D


def create_model(number_classes):
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
    model.add(Dense(number_classes))
    model.add(Activation('softmax'))
    return model

def classify_nn(type, candidate, flag=5): # by default flag something random and just use it when cercle
    """
    :param type: String stating kind of sign. Possible values: 'rectangle','triangle','circle'
    :param candidate: The wraped image of the sign after being cropped from the original image
    :return: the name of the sign, i.e: dangerous_curve_left
    """
    res = cv2.resize(candidate, dsize=(size_crop, size_crop), interpolation=cv2.INTER_CUBIC)
    # res is in BGR
    #res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    img = np.reshape(res, [1, size_crop, size_crop, 3])
    if type == "rectangle":
        classes = model_rectangle.predict_classes(img)
        return labels_rectangle[classes[0]]
    elif type == "triangle":
        classes = model_triangle.predict_classes(img)
        return labels_triangle[classes[0]]
    else:
        classes = model_cercle.predict_classes(img)
        return labels_cercle[classes[0]]


def transform_tri(image, vector1, vector2, other_vec):
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


def order_points(pts):
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    pts = np.squeeze(pts)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect




def increase_rect(pts):
    """
    :param pts: The points returned by contour
    :return: A bigger rectangle (mainly for airport sign) as sometimes contours provides just a little part of the sign
    """
    pts = np.squeeze(pts)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[0][0] = rect[0][0] - 30
    if rect[0][0] < 0:
        rect[0][0] = 0
    rect[0][1] = rect[0][1] - 30
    if rect[0][1] < 0:
        rect[0][1] = 0
    rect[2] = pts[np.argmax(s)]
    rect[2][0] = rect[2][0] + 30
    rect[2][1] = rect[2][1] + 30
    if rect[2][0] > 640:
        rect[2][0] = 640
    if rect[2][1] > 480:
        rect[2][1] = 480
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[1][0] = rect[1][0] + 30
    rect[1][1] = rect[1][1] - 30
    if rect[1][1] < 0:
        rect[1][1] = 0
    if rect[1][0] > 480:
        rect[1][0] = 480
    rect[3] = pts[np.argmax(diff)]
    rect[3][0] = rect[3][0] - 30
    rect[3][1] = rect[3][1] + 30
    if rect[3][1] > 640:
        rect[3][1] = 640
    if rect[3][0] < 0:
        rect[3][0] = 0
    # return the ordered coordinates
    return rect


def detect_triangle(cnt):
    global index_triangle
    global triCnt

    epsilon = 0.05 * cv2.arcLength(cnt, True) # 0.05 works for triangle
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    area_contour = cv2.contourArea(approx)
    corners = len(approx)

    #----------------------------------------- Check proportions ---------------------------------------------
    if corners == 3:
        # probably a triangle
        mask_triangle = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(mask_triangle, pts=[approx], color=(255, 255, 255))  # Let just the contour of interest
        res = cv2.bitwise_and(img, img, mask=mask_triangle)
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
        prop = np.sum(redMask) / np.sum(mask_triangle)  # cal proportion of expected color
        flag = 0
        if prop < 0.6:
            flag = 1
            # ---------- draw and save ROI ---------------------- #
        if flag == 0:
            # choose contour base on corner and space
            flag_closeness = 0
            mm = cv2.moments(cnt)  # cal center
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])
            for center in triCnt:
                k = np.square(center[1][0] - cx) + np.square(center[1][1] - cy)
                if k < 20:
                    flag_closeness = 1
                    break
            if flag_closeness == 0:
                box_triangle = transform_tri(img, approx[0], approx[1], approx[2])
                size_crop = 64
                dst = np.array([
                    [0, 0],
                    [size_crop - 1, 0],
                    [size_crop - 1, size_crop - 1],
                    [0, size_crop - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(box_triangle, dst)
                warped_triangle = cv2.warpPerspective(img, M, (size_crop, size_crop))
                cv2.imwrite('results/triangle/' + str(index) + 'triangle.jpg', warped_triangle)
                type = classify_nn("triangle", warped_triangle)  # returns the string saying to which template it belongs to
                if type != "CRAP":

                    cv2.imwrite('results/triangle/' + str(index_cercle) + 'triangle.jpg', warped_triangle)
                    triCnt.extend([[approx, (cx, cy), type]])  # save approx Polygon and center
                    index_triangle = index_triangle + 1

def detect_rectangle(cnt):
    global index_rectangle

    epsilon = 0.05 * cv2.arcLength(cnt, True) # for STOP detects 8 points ; 0.05 works for triangle
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    area_contour = cv2.contourArea(approx)
    corners = len(approx)
    rectangle_points = cv2.minAreaRect(approx)
    min_rectangle = np.int0(cv2.boxPoints(rectangle_points))
    area_rectangle = cv2.contourArea(min_rectangle)
    if area_rectangle < 250: # TODO: IMPROVE?? Is to avoid rectangles of 0 area dividing in proportion_rect
        return
    proportion_rect = area_contour/area_rectangle
    thresh_rectangle = 0.70
    flag_preprocess = 0

    if proportion_rect > 0.50 and proportion_rect < thresh_rectangle:  # I try to increase the size of the rectangle to see if more blue around
        x, y, w, h = cv2.boundingRect(cnt)
        new_rectangle = increase_rect([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
        mask_rectangle_auxiliar = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(mask_rectangle_auxiliar, pts=[new_rectangle.astype(int)],
                     color=(255, 255, 255))  # Let just the contour of interest
        res = cv2.bitwise_and(img, img, mask=mask_rectangle_auxiliar)
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

        # ---------------Check the color -----------------
        # Blue mask
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([135, 240, 240])
        blueMask = cv2.inRange(hsv, lower_blue, upper_blue)
        blueMask = cv2.bitwise_and(blueMask, mask_rectangle_auxiliar)
        # White mask
        lower_white = np.array([205, 200, 190])  # COLOR IN BGR --> first blue and red at the end
        upper_white = np.array([255, 255, 255])
        whiteMask = cv2.inRange(res, lower_white, upper_white)  # IN BGR FORMAT
        whiteMask = cv2.bitwise_and(whiteMask, mask_rectangle_auxiliar)

        total_mask = whiteMask | blueMask

        thresh_blue_white = 0.60

        blue_contours = cv2.findContours(blueMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[
            1]  # TODO: Check if better use total_mask
        blue_contours = [cv2.approxPolyDP(blue_contours[i], 0.05 * cv2.arcLength(blue_contours[i], True), True) for
                         i in
                         range(len(blue_contours))]
        blue_contours = [blue_contours[i] for i in range(len(blue_contours)) if
                         cv2.contourArea(blue_contours[i]) > 50]
        if not blue_contours:
            return
        allpoints = np.concatenate([blue_contours[i] for i in range(len(blue_contours))])
        rect_all_blue = cv2.minAreaRect(allpoints)
        box_definitive = cv2.boxPoints(rect_all_blue)
        box_definitive = np.int0(box_definitive)
        epsilon = 0.05 * cv2.arcLength(cnt, True)  # for STOP detects 8 points ; 0.05 works for triangle
        approx = cv2.approxPolyDP(allpoints, epsilon, True)
        area_contour = cv2.contourArea(approx)
        flag_preprocess = 1
        min_rectangle = box_definitive

    proportion_rect = area_contour/area_rectangle


    if (proportion_rect > thresh_rectangle) or (flag_preprocess == 1): # TODO: Check if makes sense the preprocess
        # probably a rectangle
        mask_rectangle = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(mask_rectangle, pts=[approx], color=(255, 255, 255))  # Let just the contour of interest
        res = cv2.bitwise_and(img, img, mask=mask_rectangle)
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

        # ---------------Check the color -----------------
        # Blue mask
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([135, 240, 240])
        blueMask = cv2.inRange(hsv, lower_blue, upper_blue)
        blueMask = cv2.bitwise_and(blueMask, mask_rectangle)
        # White mask
        lower_white = np.array([205, 200, 190])  # COLOR IN BGR --> first blue and red at the end
        upper_white = np.array([255, 255, 255])
        whiteMask = cv2.inRange(res, lower_white, upper_white)  # IN BGR FORMAT
        whiteMask = cv2.bitwise_and(whiteMask, mask_rectangle)


        total_mask = whiteMask | blueMask

        prop_blue_white = np.sum(total_mask)/np.sum(mask_rectangle)
        white_prop = np.sum(whiteMask)/np.sum(mask_rectangle)
        thresh_blue_white = 0.60

        if prop_blue_white > thresh_blue_white and corners > 3 and white_prop > 0.05:#and corners < 8:

            # choose contour base on corner and space
            mm = cv2.moments(min_rectangle)  # cal center
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])
            # check repeat contour
            flag_closeness = 0
            for center in rectCnt:
                k = np.sqrt(np.square(center[1][0] - cx) + np.square(
                    center[1][1] - cy))  # don't consider rectangles very close to rectangles already found
                if k < 20:
                    flag_closeness = 1
                    break

            if flag_closeness == 0:
                size_crop = 64
                # Make perspective transformation
                dst = np.array([
                    [0, 0],
                    [size_crop - 1, 0],
                    [size_crop - 1, size_crop - 1],
                    [0, size_crop - 1]], dtype="float32")

                # compute the perspective transform matrix and then apply it
                M = cv2.getPerspectiveTransform(min_rectangle.astype(np.float32), dst)
                warped_rectangle = cv2.warpPerspective(imgOri, M, (size_crop, size_crop))  # TODO: Check if requires RGB or BGR
                flag_nn = 0  # if flag 1 Im gonna draw the contour in final image
                type = classify_nn("rectangle", warped_rectangle)
                if type != "CRAP":
                    cv2.imwrite('results/rectangle/' + str(index_rectangle) + 'rectangle.jpg', warped_rectangle)
                    index_rectangle = index_rectangle + 1
                    rectCnt.extend([[min_rectangle, (cx, cy), type]])


def detect_cercle(cnt):
    global index_cercle
    global cirCnt

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
        # should be a cercle
        # TODO : Mask everything in the image except the inside of the cercle
        mask_cercle = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(mask_cercle, pts=[approx], color=(255, 255, 255))  # TODO: Check if better approx or cnt
        im_masked_cerc = cv2.bitwise_and(img, img, mask=mask_cercle)
        im_masked_cerc_hsv = cv2.cvtColor(im_masked_cerc, cv2.COLOR_BGR2HSV)
        # binary mask
        bin_mask = cv2.cvtColor(im_masked_cerc, cv2.COLOR_RGB2GRAY)
        _, bin_mask = cv2.threshold(bin_mask, 1, 255, cv2.THRESH_BINARY)
        # TODO : Check for color distribution inside the cercle
        # Red and yellow mask
        # Two edge of the Hue turnel
        lower1 = np.array([160, 10, 90])
        upper1 = np.array([179, 250, 200]) # CHANGED BY ME
        redMask1 = cv2.inRange(im_masked_cerc_hsv, lower1, upper1)
        lower2 = np.array([0, 10, 100])
        upper2 = np.array([40, 240, 240])
        ryMask = cv2.inRange(im_masked_cerc_hsv, lower2, upper2) | redMask1

        # Blue mask
        lower = np.array([91, 80, 60])
        upper = np.array([140, 255, 255])
        blueMask = cv2.inRange(im_masked_cerc_hsv, lower, upper)

        # White mask
        lower_white = np.array([205, 200, 190])  # COLOR IN BGR --> first blue and red at the end
        upper_white = np.array([255, 255, 255])
        whiteMask = cv2.inRange(im_masked_cerc, lower_white, upper_white)  # IN BGR FORMAT
        blue_whiteMask = blueMask | whiteMask

        # Proportion
        blue_white_Prop = np.sum(blue_whiteMask) / np.sum(bin_mask)
        blueProp = np.sum(blueMask) / np.sum(bin_mask)
        ryProp = np.sum(ryMask) / np.sum(bin_mask)

        # TODO : Divide between colors
        flag = 0
        if (blue_white_Prop > 0.6 or (blue_white_Prop > 0.35 and proportion_ellipse > 0.91 and ryProp < 0.10)) and (blueProp > 0.25):
            flag = 1
        elif (blueProp > 0.25) and (ryProp > 0.17):
            flag = 2
        elif ryProp > 0.70 or (ryProp > 0.45 and proportion_ellipse > 0.90):
            flag = 3

        if flag != 4 and flag != 0:
            mm = cv2.moments(cnt)  # cal center
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])
            flag_closeness = 0
            for type_cercle in cirCnt:  # check every kind circle
                for center in type_cercle:
                    k = np.sqrt(np.square(center[1][0] - cx) + np.square(
                    center[1][1] - cy))  # don't consider cercles very close to cercles already found
                    if k < 20:
                        flag_closeness = 1
                        break

            if flag_closeness == 0:
                rect_covering_cercle = cv2.minAreaRect(cnt) # TODO: Check if better "cnt" or "approx"
                box_cercle = cv2.boxPoints(rect_covering_cercle)
                box_cercle = np.int0(box_cercle)
                size_crop = 64
                # Make perspective transformation
                dst = np.array([
                    [0, 0],
                    [size_crop - 1, 0],
                    [size_crop - 1, size_crop - 1],
                    [0, size_crop - 1]], dtype="float32")

                # compute the perspective transform matrix and then apply it
                M = cv2.getPerspectiveTransform(box_cercle.astype(np.float32), dst)
                warped_cercle = cv2.warpPerspective(imgOri, M, (size_crop, size_crop)) # TODO: Check if requires RGB or BGR

                type = classify_nn("cercle", warped_cercle)
                if type != "CRAP":
                    cirCnt[flag-1].extend([[box_cercle, (cx, cy), type]])  # save ellipse center and size
                    cv2.imwrite('results/cercle/' + str(index_cercle) + 'cercle.jpg', warped_cercle)
                    index_cercle = index_cercle + 1




# TODO: Put the directory of the image
imgOri = cv2.imread('images_mine/2106.png')  # RGB space imput image

# ------------------Filter--------------------------------------------------- #
img = cv2.bilateralFilter(imgOri, 10, 75, 75)

# -------------- Find & Draw Contour ---------------------------------------- #
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# two different thresh, canny has better result ( !!! maybe try other edge detection)
thresh = 50
binaryImg = cv2.Canny(gray, thresh, 2*thresh)
contours = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1] #modification to work in OpenCV 3.4
contours = [contours[i] for i in range(len(contours)) if
                     cv2.contourArea(contours[i]) > 300]  # be sure the have a certain area
# TODO: Check if convexity matters
#contours = [contours[i] for i in range(len(contours)) if cv2.isContourConvex(contours[i])]

index = 0 # where we keep count of the valid shapes found
num_classes = 15
size_crop = 64

# ------------- Shape detection ------------------------------------------ #
triCnt = []
blueCnt = []
rectCnt = []
cirCnt = [[], [], []]  # blue circle, red & yellow circle, red & blue circle
# Import my model
model_rectangle = create_model(3)
model_rectangle.load_weights("./model/model_rectangle.h5")
#-----------------------------
model_triangle = create_model(7)
model_triangle.load_weights("./model/model_triangle.h5")
#-----------------------------
model_cercle = create_model(7)
model_cercle.load_weights("./model/model_cercle.h5")

labels_index = { 0 : "follow_left", 1 : "airport", 2 : "follow_right", 3 : "residential", 4 : "no_stopping_and_parking",
                 5 : "no_parking", 6 : "stop", 7 : "no_bicycle", 8 : "junction", 9 : "dangerous_curve_left",
                 10: "road_narrows_from_left", 11: "road_narrows_from_right", 12: "dangerous_curve_right",
                 13: "roundabout_warning", 14: "no_heavy_truck"}
labels_rectangle = {0 : "airport", 1 : "residential", 2 : "CRAP"}
labels_triangle = {0 : "dangerous_curve_left", 1 : "dangerous_curve_right", 2 : "junction", 3 : "road_narrows_from_left",
                   4 : "road_narrows_from_right", 5 : "roundabout_warning", 6 : "CRAP"}
labels_cercle = {0 : "follow_left", 1 : "follow_right", 2 : "no_bicycle", 3 : "no_heavy_truck",
                   4 : "no_parking", 5 : "no_stopping_and_parking", 6 : "stop"}

index_cercle = 0
index_triangle = 0
index_rectangle = 0

for cnt in contours:
    detect_cercle(cnt)
    detect_rectangle(cnt)
    detect_triangle(cnt)


# IN CASE CERCLE ARE CLOSE TO RECTANGLE JUST SAVE CERCLE
for species in cirCnt:  # check every kind circle
    for center_circle in species:  # check repeat contour
        for center_rect in rectCnt:
            k = np.sqrt(np.square(center_rect[1][0] - center_circle[0][0][0]) + np.square(
                center_rect[1][1] - center_circle[0][0][1]))
            if k < 20:
                rectCnt = [x for x in rectCnt if not (x[0] == center_rect[0]).all()]

for cnt in rectCnt:
    cv2.drawContours(img, [cnt[0]], -1, (255, 255, 0), 2)
    cv2.putText(img, cnt[2], cnt[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
for cnt in cirCnt:
    for cercle_type in cnt:
        cv2.drawContours(img, [cercle_type[0]], -1, (255, 255, 0), 2)
        cv2.putText(img, cercle_type[2], cercle_type[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
for cnt in triCnt:
    cv2.drawContours(img, [cnt[0]], -1, (255, 255, 0), 2)
    cv2.putText(img, cnt[2], cnt[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
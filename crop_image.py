import cv2
import numpy as np
import math
from os import listdir
# TODO: Put the directory of the image
imgOri = cv2.imread('images_mine/2019-02-20-183908.jpg')  # RGB space imput image

# ------------------Filter--------------------------------------------------- #
img = cv2.bilateralFilter(imgOri, 10, 75, 75)

# -------------- Find & Draw Contour ---------------------------------------- #
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# two different thresh, canny has better result ( !!! maybe try other edge detection)
thresh = 50
binaryImg = cv2.Canny(gray, thresh, 3*thresh)
contours = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1] #modification to work in OpenCV 3.4

index = 0 # where we keep count of the valid shapes found

def match_template(type, candidate, flag=5): # by default flag something random and just use it when cercle
    """
    :param type: String stating kind of sign. Possible values: 'rectangle','triangle','circle'
    :param candidate: The wraped image of the sign after being cropped from the original image
    :return: the name of the sign, i.e: dangerous_curve_left
    """
    threshold = 0
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(candidate, None)
    idx = "unsure"
    if type=='triangle':
        for file in listdir('templates/triangle/'):
            image = cv2.imread('templates/triangle/' + file)
            resized_image = cv2.resize(image, (candidate.shape[0], candidate.shape[1]))
            kp_2, desc_2 = sift.detectAndCompute(image, None)
            index_params = dict(algorithm=0, trees=5)
            search_params = dict()
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc_1, desc_2, k=2)
            good_points = 0
            ratio = 0.5
            for m, n in matches:
                if m.distance < ratio * n.distance:
                    good_points = good_points + 1
            if (good_points > threshold):
                threshold = good_points
                idx = file
                if (threshold > 10):
                    break
    if type=='rectangle':
        for file in listdir('templates/rectangle/'):
            image = cv2.imread('templates/rectangle/' + file)
            resized_image = cv2.resize(image, (candidate.shape[0], candidate.shape[1]))
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
            if (good_points > threshold):
                threshold = good_points
                idx = file
                if (threshold > 10):
                    break
    if type=='circle':
        # 1: blue circle;
        # 2: red and blue circle;
        # 3: red and yellow circle;
        # 4: not interested circle;
        if flag==1:
            for file in listdir('templates/circle/blue/'):
                image = cv2.imread('templates/circle/blue/' + file)
                resized_image = cv2.resize(image, (candidate.shape[0], candidate.shape[1]))
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
                if (good_points > threshold):
                    threshold = good_points
                    idx = file
                    if (threshold > 10):
                        break
        elif flag==2:
            for file in listdir('templates/circle/red_blue/'):
                image = cv2.imread('templates/circle/red_blue/' + file)
                resized_image = cv2.resize(image, (candidate.shape[0], candidate.shape[1]))
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
                if (good_points > threshold):
                    threshold = good_points
                    idx = file
                    if (threshold > 10):
                        break
        elif flag==3:
            for file in listdir('templates/circle/red_yellow/'):
                image = cv2.imread('templates/circle/red_yellow/' + file)
                resized_image = cv2.resize(image, (candidate.shape[0], candidate.shape[1]))
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
                if (good_points > threshold):
                    threshold = good_points
                    idx = file
                    if (threshold > 10):
                        break
    return idx

def transform_tri(image, vector1, vector2, other_vec):
    """
    :param image: The image we work with
    :param vector1: First vertex of the longest side of the triangle observed
    :param vector2: Second vertex of the longest side of the triangle observed
    :param other_vec: The other vertex, the one we need to add the perpendicular line to find the other 2 points of
    quadrilater
    :return:
    """

    # Add some margins to the found vertex in order to transform better the triangle
    if (other_vec[0][0] < vector1[0][0]):
        vector1[0][0] = vector1[0][0] + 5
    else:
        vector1[0][0] = vector1[0][0] - 5

    if (other_vec[0][0] < vector2[0][0]):
        vector2[0][0] = vector2[0][0] + 5
    else:
        vector2[0][0] = vector2[0][0] - 5

    if (other_vec[0][1] < vector1[0][1]):
        vector1[0][1] = vector1[0][1] + 5
    else:
        vector1[0][1] = vector1[0][1] - 5

    if (other_vec[0][1] < vector2[0][1]):
        vector2[0][1] = vector2[0][1] + 5
    else:
        vector2[0][1] = vector2[0][1] - 5


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
    dst = np.array([
        [0,int(round(30 * np.sqrt(3)))],
        [60,int(round(30 * np.sqrt(3)))],
        [60, 0],
        [0, 0]], dtype="float32") # the output size of the box, normalized would be 1 for the base and sqrt(3)/2 for height

    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(image, M, (60, int(round(30 * np.sqrt(3)))))
    cv2.imwrite('results/triangle/'+str(index)+'triangle.jpg', warped)

    type = match_template("triangle", warped) #returns the string saying to which template it belongs to, if none returns
    # "unsure"
    return type



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

def transform_rect(image, vertexs):
    global index
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(vertexs)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    type = match_template("rectangle", warped)
    cv2.imwrite('results/rectangle/'+str(index)+'rectangle.jpg', warped)
    index = index+1

def transform_circle(image, vertexs, flag):
    global index
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(vertexs)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order

    if maxWidth>maxHeight: # get the dimension of the bigger vertex of rectangle and reshape cercle according to that
        len_square = maxWidth
    else:
        len_square=maxHeight

    dst = np.array([
        [0, 0],
        [len_square - 1, 0],
        [len_square - 1, len_square - 1],
        [0, len_square - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (len_square, len_square)) # the result should be a cercle inside a square
    type = match_template("circle", warped, flag)
    cv2.imwrite('results/circle/'+str(index)+'circle.jpg', warped)
    index = index+1

    return type


def detectTri(cnt, triCnt):
    global index

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
        flag = 0
        for center in triCnt:
            k = np.square(center[1][0] - cx) + np.square(center[1][1] - cy)
            if k < 20:
                flag = 1
                break
        # ------------- detect color in contour ------------------------------ #
        # make contour mask
        cntMask = np.zeros((480, 640), np.uint8)
        fillCnt = []
        fillCnt.append(cnt)
        cv2.drawContours(cntMask, fillCnt, -1, 255, -1)
        res = cv2.bitwise_and(img, img, mask=cntMask)
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
            flag = 1
                # ---------- draw and save ROI ---------------------- #
        if flag == 0:
            type = transform_tri(img, approx[0], approx[2], approx[1])
            if type != 'unsure':
                triCnt.extend([[approx, (cx, cy), type]])  # save approx Polygon and center
            else:
                type = transform_tri(img, approx[0], approx[1], approx[2])
                if type != 'unsure':
                    triCnt.extend([[approx, (cx, cy), type]])  # save approx Polygon and center
                else:
                    type = transform_tri(img, approx[1], approx[2], approx[0])
                    triCnt.extend([[approx, (cx, cy), type]])  # save approx Polygon and center
            index = index + 1
    return triCnt

"""
def detectRect(cnt, rectCnt):
    # Polygon approximation (!!! repeat code, probably add tri and rect in same func)
    epsilon = 0.05 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # Analyse shape
    corners = len(approx)
    # Calculate space
    S1 = cv2.contourArea(cnt)
    if corners == 4 and S1 > 150:
        # choose contour base on corner and space
        mm = cv2.moments(cnt)  # cal center
        cx = int(mm['m10'] / mm['m00'])
        cy = int(mm['m01'] / mm['m00'])
        # check repeat contour
        flag = 0
        for center in rectCnt:
            k = np.square(center[1][0] - cx) + np.square(center[1][1] - cy)
            if k < 20:
                flag = 1
                break
        # ------------- detect color in contour ------------- #
        # make contour mask
        cntMask = np.zeros((480, 640), np.uint8)
        fillCnt = []
        fillCnt.append(cnt)
        cv2.drawContours(cntMask, fillCnt, -1, 255, -1)
        res = cv2.bitwise_and(img, img, mask=cntMask)
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        # Blue mask
        lower = np.array([100, 50, 50])
        upper = np.array([135, 240, 240])
        buleMask = cv2.inRange(hsv, lower, upper)
        prop = np.sum(buleMask) / np.sum(cntMask)  # cal proportion of expected color
        if prop < 0.7:
            flag = 1
        # ---------- draw and save ROI ---------------------- #

        if flag == 0:
            rectCnt.extend([[approx, (cx, cy)]])  # save approx Polygon and center
            cv2.drawContours(img, cnt, -1, (255, 255, 0), 1)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (255, 255, 0), 1)
            cv2.putText(img, 'rect', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            transform_rect(img, approx)
    return rectCnt
"""

def detectCir(cnt, cirCnt):
    if len(cnt) < 20:
        return cirCnt
    S1 = cv2.contourArea(cnt)
    ell = cv2.fitEllipse(cnt)
    pos, size, angle = ell
    ex, ey = pos
    dx, dy = size
    S2 = math.pi*ell[1][0]*ell[1][1]
    if S2 < 1000:
        return cirCnt
    prop = S1 / S2
    if prop > 0.22:
        # check repeat contour
        mm = cv2.moments(cnt)  # cal center
        cx = int(mm['m10'] / mm['m00'])
        cy = int(mm['m01'] / mm['m00'])
        flag = 0
        # 0: need more check;
        # 1: blue circle;
        # 2: red and blue circle;
        # 3: red and yellow circle;
        # 4: not interested circle;
        for species in cirCnt: # check every kind circle
            for center in species:  # check repeat contour
                k = np.square(center[0][0] - cx) + np.square(center[0][1] - cy)
                if k < 20:
                    flag = 4
                    break
            if flag == 4:
                break
        # detect outliers(center of contour should close to ellipse)
        d = np.square(ell[0][0] - cx) + np.square(ell[0][1] - cy)
        if d > 20:
            flag = 4

        GoF = 0  # Goodness Of Fit, the smaller the better (not sure)
        for point in cnt:
            posx = (point[0][0] - ex) * math.cos(-angle) - (point[0][1] - ey) * math.sin(-angle)
            posy = (point[0][0] - ex) * math.sin(-angle) + (point[0][1] - ey) * math.cos(-angle)
            GoF += abs(np.square(posx/dx) + np.square(posy/dy) - 0.25)
        # GoF = GoF / cnt.size
        if GoF > 10:
            flag = 4

        if flag != 4:
            # ------------- detect color in contour ------------------------------ #
            # make contour mask
            cntMask = np.zeros((480, 640), np.uint8)
            fillCnt = []
            fillCnt.append(cnt)
            cv2.drawContours(cntMask, fillCnt, -1, 255, -1)  # fill contour with white

            res = cv2.bitwise_and(img, img, mask=cntMask)
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
            # cv2.imshow('img', blueMask)
            # cv2.waitKey(0)
            # classify base on color
            blueProp = np.sum(blueMask) / np.sum(cntMask)
            ryProP = np.sum(ryMask) / np.sum(cntMask)
            if blueProp > 0.7:
                flag = 1
            elif (blueProp > 0.2) and (ryProP > 0.2):
                flag = 2
            elif ryProP > 0.7:
                flag = 3
            # ---------- draw and save ROI ---------------------------------------- #
            if flag != 0:
                cirCnt[flag-1].extend([ell])  # save ellipse center and size
                cv2.ellipse(img, ell, (0, 255, 255), 1)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                type = transform_circle(img, box, flag)
                cv2.drawContours(img, cnt, -1, (0, 255, 255), 2)
                x = int(ell[0][0])
                y = int(ell[0][1])
                cv2.putText(img, 'cir%d' % flag, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return cirCnt


# ------------- Shape detection ------------------------------------------ #
triCnt = []
rectCnt = []
cirCnt = [[], [], []]  # bule circle, red & yellow circle, red & bule circle
for cnt in contours:
    triCnt = detectTri(cnt, triCnt)
    #rectCnt = detectRect(cnt, rectCnt)
    cirCent = detectCir(cnt, cirCnt)


cv2.imshow('img', img)
cv2.waitKey(0)
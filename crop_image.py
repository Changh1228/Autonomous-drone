import cv2
import numpy as np
import math
from numpy import linalg as LA

imgOri = cv2.imread('images_mine/2019-02-20-183626.jpg')  # RGB space imput image

# ------------------Filter--------------------------------------------------- #
img = cv2.bilateralFilter(imgOri, 10, 75, 75)
# bilateral Filter (best quality, slow?)
# img = cv2.GaussianBlur(imgOri, (5, 5), 0)  # Gaussian Filter

# -------------- Find & Draw Contour ---------------------------------------- #
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# two different thresh, canny has better result ( !!! maybe try other edge detection)
# ret, thresh = cv2.threshold(gray, 127, 255, 0)
binaryImg = cv2.Canny(gray, 100, 500)
h = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = h[0]
# Test #
# cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

index = 0 # where we keep count of the valid shapes found
def transform_tri(image, vector1, vector2, other_vec):
    """
    :param image: The image we work with
    :param vector1: First vertex of the longest side of the triangle observed
    :param vector2: Second vertex of the longest side of the triangle observed
    :param other_vec: The other vertex, the one we need to add the perpendicular line to find the other 2 points of
    quadrilater
    :return:
    """
    global index;

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
    cv2.imwrite(''+str(index)+'triangle.jpg', warped)
    index = index+1


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
    global index;
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
    cv2.imwrite(''+str(index)+'rectangle.jpg', warped)
    index = index+1

def transform_circle(image, vertexs):
    global index;
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
    cv2.imwrite(''+str(index)+'circle.jpg', warped)
    index = index+1


def detectTri(cnt, triCnt):
    # Polygon approximation
    epsilon = 0.05 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # Analyse shape
    corners = len(approx)
    # Calculate space
    S1 = cv2.contourArea(cnt)
    if corners == 3 and S1 > 1000:
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
        if flag == 0:
            triCnt.extend([[approx, (cx, cy)]])  # save approx Polygon and center
            cv2.drawContours(img, cnt, -1, (0, 255, 0), 1)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 1)
            cv2.putText(img, 'tri', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            idx = 10 # some random value
            max_norm = -999999999
            vector = [approx[0][0] - approx[1][0], approx[0][0] - approx[2][0], approx[1][0] - approx[2][0]] # 3 vectors
            for i in range(len(vector)):
                if (LA.norm(vector[i])>max_norm):
                    idx = i
                    max_norm = LA.norm(vector[i])
            if (idx == 1):
                # two first vertex are the ones of longest, the last is the other vertex
                transform_tri(img, approx[0], approx[2], approx[1])
            elif (idx == 0):
                transform_tri(img, approx[0], approx[1], approx[2])
            else:
                transform_tri(img,approx[1], approx[2], approx[0])
    return triCnt


def detectRect(cnt, rectCnt):
    # Polygon approximation (!!! repeat code, probably add tri and rect in same func)
    epsilon = 0.05 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # Analyse shape
    corners = len(approx)
    # Calculate space
    S1 = cv2.contourArea(cnt)
    if corners == 4 and S1 > 1000:
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
        if flag == 0:
            rectCnt.extend([[approx, (cx, cy)]])  # save approx Polygon and center
            cv2.drawContours(img, cnt, -1, (255, 255, 0), 1)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (255, 255, 0), 1)
            cv2.putText(img, 'rect', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            transform_rect(img, approx)
    return rectCnt


def detectCir(cnt, cirCent):
    if len(cnt) < 50:
        return cirCent
    S1 = cv2.contourArea(cnt)
    ell = cv2.fitEllipse(cnt)
    S2 = math.pi*ell[1][0]*ell[1][1]
    if (S1/S2) > 0.2:
        cv2.ellipse(img, ell, (0, 255, 0), 1)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        transform_circle(img, box)

# ------------- Shape detection ------------------------------------------ #
triCnt = []
rectCnt = []
cirCent = []
for cnt in contours:
    triCnt = detectTri(cnt, triCnt)
    rectCnt = detectRect(cnt, rectCnt)
    cirCent = detectCir(cnt, cirCent)


cv2.imshow('img', img)
cv2.waitKey(0)
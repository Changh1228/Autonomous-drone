import cv2
import numpy as np
import math

imgOri = cv2.imread('test_img\\cir2.jpg')  # RGB space imput image

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
    return rectCnt


def detectCir(cnt, cirCent):
    if len(cnt) < 50:
        return cirCent
    S1 = cv2.contourArea(cnt)
    ell = cv2.fitEllipse(cnt)
    S2 = math.pi*ell[1][0]*ell[1][1]
    if (S1/S2) > 0.2:
        cv2.ellipse(img, ell, (0, 255, 0), 1)


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

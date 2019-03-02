import cv2
import numpy as np
import math

imgOri = cv2.imread('test_img\\cir12.jpg')  # RGB space imput image

# ------------------Filter--------------------------------------------------- #
img = cv2.bilateralFilter(imgOri, 10, 75, 75)
# bilateral Filter (best quality, slow?)
# img = cv2.GaussianBlur(imgOri, (5, 5), 0)  # Gaussian Filter

# -------------- Find & Draw Contour ---------------------------------------- #
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# two different thresh, canny has better result
# ret, thresh = cv2.threshold(gray, 127, 255, 0)
binaryImg = cv2.Canny(gray, 100, 250)
h = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = h[0]
# Test #
# cv2.drawContours(img, contours, -1, (0, 255, 0), 1)


def detectTri(cnt, triCnt):
    # -------- Polygon detection ------------------------ #
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
        # ------------- detect color in contour ------------- #
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
            triCnt.extend([[approx, (cx, cy)]])  # save approx Polygon and center
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)
            cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 1)
            cv2.putText(img, 'tri', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return triCnt


def detectRect(cnt, rectCnt):
    # -------- Polygon detection ------------------------ #
    # (!!! repeat code, probably add tri and rect in same func)
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
            cv2.drawContours(img, cnt, -1, (255, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (255, 255, 0), 1)
            cv2.putText(img, 'rect', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return rectCnt


def detectCir(cnt, cirCnt):
    if len(cnt) < 50:
        return cirCnt
    S1 = cv2.contourArea(cnt)
    ell = cv2.fitEllipse(cnt)
    S2 = math.pi*ell[1][0]*ell[1][1]
    prop = S1 / S2
    if prop > 0.22:
        # check repeat contour
        mm = cv2.moments(cnt)  # cal center
        cx = int(mm['m10'] / mm['m00'])
        cy = int(mm['m01'] / mm['m00'])
        flag = 0
        for center in cirCnt:
            k = np.square(center[0][0] - cx) + np.square(center[0][1] - cy)
            if k < 20:
                flag = 1
                break
        # detect outliers(center of contour should close to ellipse)
        d = np.square(ell[0][0] - cx) + np.square(ell[0][1] - cy)
        if d > 20:
            flag = 1

        # ------------- detect color in contour ------------- #
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
        # Blue mask
        lower = np.array([100, 50, 50])
        upper = np.array([135, 240, 240])
        buleMask = cv2.inRange(hsv, lower, upper)

        rybMask = redMask1 | redMask2 | buleMask
        prop = np.sum(rybMask) / np.sum(cntMask)  # cal proportion of expected color
        if prop < 0.7:
            flag = 1

        # ---------- draw and save ROI ---------------------- #
        if flag == 0:
            cirCnt.extend([ell])
            cv2.ellipse(img, ell, (0, 255, 255), 1)
            cv2.drawContours(img, cnt, -1, (0, 255, 255), 2)
            x = int(ell[0][0])
            y = int(ell[0][1])
            cv2.putText(img, 'cir', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return cirCnt


# ------------- Shape detection ------------------------------------------ #
triCnt = []
rectCnt = []
cirCnt = []
for cnt in contours:
    triCnt = detectTri(cnt, triCnt)
    rectCnt = detectRect(cnt, rectCnt)
    cirCnt = detectCir(cnt, cirCnt)  # the stop sign can be treated as circle

cv2.imshow('img', img)
cv2.waitKey(0)

import cv2
import numpy as np
import math

imgOri = cv2.imread('test_img\\test.png')  # RGB space imput image

# ------------------Filter--------------------------------------------------- #
img = cv2.bilateralFilter(imgOri, 10, 75, 75)
# bilateral Filter (best quality, slow?)
# img = cv2.GaussianBlur(imgOri, (5, 5), 0)  # Gaussian Filter

# -------------- Find & Draw Contour ---------------------------------------- #
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# two different thresh, canny has better result
# ret, thresh = cv2.threshold(gray, 150, 255, 0)
thresh = 50
binaryImg = cv2.Canny(gray, thresh, 3*thresh)
contours, hierarchy = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow('img', binaryImg)
# cv2.waitKey(0)


def detectTri(cnt, triCnt):
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
        # ---------- draw and save ROI --------------------------------------- #
        if flag == 0:
            triCnt.extend([[approx, (cx, cy)]])  # save approx Polygon and center
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)
            cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 1)
            cv2.putText(img, 'tri', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return triCnt


def detectRect(cnt, rectCnt):
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
                flag = 1
                break
        # ------------- detect color in contour --------------------------- #
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

        # ---------- draw and save ROI ------------------------------------- #
        if flag == 0:
            rectCnt.extend([[approx, (cx, cy)]])  # save approx Polygon and center
            cv2.drawContours(img, cnt, -1, (255, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (255, 255, 0), 1)
            cv2.putText(img, 'rect', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return rectCnt


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
        # 1: bule circle;
        # 2: red and blue circle;
        # 3: red and yellow circle;
        # 4: not interested circle;

        for species in cirCnt:  # check every kind circle
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
                cv2.drawContours(img, cnt, -1, (0, 255, 255), 2)
                x = int(ell[0][0])
                y = int(ell[0][1])
                cv2.putText(img, 'cir%d' % flag, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return cirCnt


# ------------- Shape detection ------------------------------------------------ #
triCnt = []
rectCnt = []
cirCnt = [[], [], []]  # bule circle, red & yellow circle, red & bule circle
for cnt in contours:
    triCnt = detectTri(cnt, triCnt)
    # rectCnt = detectRect(cnt, rectCnt)
    cirCnt = detectCir(cnt, cirCnt)  # the stop sign can be treated as circle

cv2.imshow('img', img)
cv2.waitKey(0)

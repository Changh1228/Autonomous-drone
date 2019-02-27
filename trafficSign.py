import cv2
import numpy as np

imgOri = cv2.imread('triangle3.jpg')  # RGB space imput image

# ------------------Filter----------------#
img = cv2.bilateralFilter(imgOri, 10, 75, 75)
# bilateral Filter (best quality, slow?)
# img = cv2.GaussianBlur(imgOri, (9, 9), 0)  # Gaussian Filter

# ----------------color mask -------------#
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV space
lower1 = np.array([120, 50, 50])  # 0-180 0-255 0-255
upper1 = np.array([180, 200, 200])
mask1 = cv2.inRange(hsv, lower1, upper1)

lower2 = np.array([0, 10, 10])
upper2 = np.array([30, 250, 250])
mask2 = cv2.inRange(hsv, lower2, upper2)

mask = mask1 | mask2

# -------------- Find & Draw Contour ---------- #
ret, thresh = cv2.threshold(mask, 127, 255, 0)
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
binaryImg = cv2.Canny(mask, 50, 200)
h = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = h[0]

# ------------- Shape extraction ---------------- #
triCnt = [3]
triCntCenter = [3]
i = 0
for cnt in range(len(contours)):
    # Contour approximation
    epsilon = 0.1 * cv2.arcLength(contours[cnt], True)
    approx = cv2.approxPolyDP(contours[cnt], epsilon, True)
    # Analyse shape
    corners = len(approx)
    # Calculate space
    area = cv2.contourArea(contours[cnt])
    if corners == 3 and area > 1000:
        # choose contour base on corner and space
        mm = cv2.moments(contours[cnt])  # cal center
        cx = int(mm['m10'] / mm['m00'])
        cy = int(mm['m01'] / mm['m00'])
        # check repeat contour
        for t in triCntCenter:
            if i > 0:
                k = np.square(t[0] - cx) + np.square(t[1] - cy)
                if k > 20:
                    triCnt[i] = approx  # srorage approx Polygon and center
                    triCntCenter[i] = (cx, cy)
            else:
                triCnt[i] = approx  # srorage ID and center
                triCntCenter[i] = (cx, cy)
            i += 1

for cnt in triCnt:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 1)

cv2.imshow('img', img)
cv2.waitKey(0)

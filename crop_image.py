import cv2
import numpy as np
import imutils

image = cv2.imread("original.png")
#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# define range of the color we look for in the HSV space
lower = np.array([0, 20, 130])
upper = np.array([120, 110, 200])

# Threshold the HSV image to get only the pixels in ranage
mask = cv2.inRange(image, lower, upper)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(image, image, mask=mask)
#cv2.imwrite('result.png',res)
cv2.imwrite('original.png',image)
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# loop over the contours
idx =0
for c in cnts:
	x,y,w,h = cv2.boundingRect(c)
	if w>50 and h>50:
		idx+=1
		new_img=image[y:y+h,x:x+w]
		cv2.imwrite(str(idx) + '.png', new_img)
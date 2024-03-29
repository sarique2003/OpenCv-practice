import cv2 as cv
import numpy as np

src = cv.imread("test.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

orb = cv.ORB_create()
kps = orb.detect(src)

result = cv.drawKeypoints(src, kps, None, -1, cv.DrawMatchesFlags_DEFAULT)
cv.imshow("result", result)
cv.imwrite('orb_result.jpg', result)
cv.waitKey(0)
cv.destroyAllWindows()
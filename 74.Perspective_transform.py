import cv2 as cv
import numpy as np

src = cv.imread("cells.png")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))
binary = cv.morphologyEx(binary, cv.MORPH_OPEN, se)
cv.imshow("binary", binary)
cv.imwrite("binary.png", binary)

contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnt_maxArea = sorted(contours, key=cv.contourArea)[0]

rect = cv.minAreaRect(cnt_maxArea)
print(rect[2])
print(rect[0])
# trick
height, width = rect[1]
print(rect[1])
box = cv.boxPoints(rect)
src_pts = np.int0(box)
print(src_pts)

dst_pts = []
dst_pts.append([width,height])
dst_pts.append([0, height])
dst_pts.append([0, 0])
dst_pts.append([width, 0])

M, status = cv.findHomography(src_pts, np.array(dst_pts)) 
result = cv.warpPerspective(src, M, (np.int32(width), np.int32(height)))

if height < width:
    result = cv.rotate(result, cv.ROTATE_90_CLOCKWISE)

cv.imshow("result", result)
cv.imwrite("result.png", result)

cv.waitKey(0)
cv.destroyAllWindows()
import cv2 as cv
import numpy as np

src = cv.imread("m1.jpg")
src = cv.resize(src, (0,0), fx=0.5, fy=0.5)
cv.imwrite('mm1.jpg', src)
r = cv.selectROI('input', src, False)  # (x_min, y_min, w, h)

# roi区域
roi = src[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
img = src.copy()
cv.rectangle(img, (int(r[0]), int(r[1])),(int(r[0])+int(r[2]), int(r[1])+ int(r[3])), (255, 0, 0), 2)
cv.imwrite('img.jpg', img)

mask = np.zeros(src.shape[:2], dtype=np.uint8)
rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3])) #(x,y,w,h)

bgdmodel = np.zeros((1,65),np.float64) # bg  13 * iterCount
fgdmodel = np.zeros((1,65),np.float64) # fg 13 * iterCount

cv.grabCut(src,mask,rect,bgdmodel,fgdmodel, 11, mode=cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')

print(mask2.shape)

result = cv.bitwise_and(src,src,mask=mask2)
#cv.imwrite('result.jpg', result)
#cv.imwrite('roi.jpg', roi)

cv.imshow('roi', roi)
cv.imshow("result", result)
cv.waitKey(0)
cv.destroyAllWindows()
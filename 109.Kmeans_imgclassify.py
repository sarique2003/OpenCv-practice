import numpy as np
import cv2 as cv

image = cv.imread('toux.jpg')

data = image.reshape((-1,3))
data = np.float32(data)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
num_clusters = 4
ret,label,center=cv.kmeans(data, num_clusters, None, criteria, num_clusters, cv.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)

#label
color = np.uint8([[255, 0, 0],
                  [0, 0, 255],
                  [128, 128, 128],
                  [0, 255, 0]])

res = color[label.flatten()]
print(res.shape)
result = res.reshape((image.shape))
cv.imshow('kmeans-image-demo',result)
cv.imwrite('kmeans-image-demo-t.jpg',np.hstack((image,result)))


cv.waitKey(0)
cv.destroyAllWindows()
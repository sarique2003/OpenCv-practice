import numpy as np
import cv2
from matplotlib import pyplot as plt

def loadDataSet(fileName):
    data = []
    with open(fileName) as f:
        for line in f.readlines():
            curLine = line.strip().split("\t")
            fltLine = list(map(float, curLine))
            data.append(fltLine)
    return np.array(data, dtype=np.float32)


data = loadDataSet('testSet2.txt')
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(data, 3, None, criteria, 2, cv2.KMEANS_RANDOM_CENTERS) #type max itter,epsilon k=3 as no centre 

print(len(label))
print(center)

A = data[label.ravel()==0]
B = data[label.ravel()==1]
C = data[label.ravel()==2]

plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(C[:,0],C[:,1],c = 'purple')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'black', marker = '*')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
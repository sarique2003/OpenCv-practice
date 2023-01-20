import cv2 as cv

box = cv.imread("box.png",0)
box_in_sence = cv.imread("box_in_scene.png",0)
cv.imshow("box", box)
cv.imshow("box_in_sence", box_in_sence)


orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(box,None)
kp2, des2 = orb.detectAndCompute(box_in_sence,None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

result = cv.drawMatches(box, kp1, box_in_sence, kp2, matches, None)
cv.imshow("orb-match", result)
cv.waitKey(0)
cv.destroyAllWindows()
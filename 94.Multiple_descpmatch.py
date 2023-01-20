import cv2 as cv

box = cv.imread("box.png")
box_in_sence = cv.imread("box_in_scene.png")
cv.imshow("box", box)
cv.imshow("box_in_sence", box_in_sence)

orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(box,None)
kp2, des2 = orb.detectAndCompute(box_in_sence,None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)
result = cv.drawMatches(box, kp1, box_in_sence, kp2, matches[:10], None)
cv.imshow("orb-match", result)
cv.imwrite("orb-match.jpg", result)
cv.waitKey(0)
cv.destroyAllWindows()

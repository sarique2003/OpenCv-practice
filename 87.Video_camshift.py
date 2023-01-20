import cv2 as cv
import numpy as np

cap = cv.VideoCapture('bike.avi')

ret,frame = cap.read()
cv.namedWindow("Demo", cv.WINDOW_AUTOSIZE)

x, y, w, h = cv.selectROI("Demo", frame, True, False)
track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, (26, 43, 46), (34, 255, 255))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    ret, track_box = cv.CamShift(dst, track_window, term_crit)

    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    cv.polylines(frame, [pts], True, (0, 255, 0), 2)


    track_window = track_box
    cv.ellipse(frame, ret, (0, 0, 255), 3, 8)
    cv.imshow('Demo',frame)
    k = cv.waitKey(50) & 0xff
    if k == 27:
        break
    else:
        cv.imwrite(chr(k)+".jpg",frame)

cv.destroyAllWindows()
cap.release()
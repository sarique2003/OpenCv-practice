import cv2 as cv
import numpy as np


capture = cv.VideoCapture(0) 
height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
count = capture.get(cv.CAP_PROP_FRAME_COUNT)
fps = capture.get(cv.CAP_PROP_FPS)
print(height, width, count, fps)
out = cv.VideoWriter("capture", cv.VideoWriter_fourcc('D', 'I', 'V', 'X'), 15,
                     (np.int(width), np.int(height)), True)
while True:
    ret, frame = capture.read()
    if ret is True:
        cv.imshow("video-input", frame)
        out.write(frame)
        c = cv.waitKey(27)
        if c == 27: # ESC
            break
    else:
        break

capture.release()
out.release()

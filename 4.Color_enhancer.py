import cv2 as cv

src = cv.imread("./test1.png")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
dst = cv.applyColorMap(src, cv.COLORMAP_JET)
cv.imshow("output", dst)

# cool and jet are color tones available in the library
image = cv.imread("./test0.png")
color_image = cv.applyColorMap(image, cv.COLORMAP_COOL)
cv.imshow("image", image)
cv.imshow("color_image", color_image)
cv.waitKey(0)
cv.destroyAllWindows()
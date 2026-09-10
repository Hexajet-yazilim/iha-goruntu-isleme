import cv2 as cv
import numpy as np

img = cv.imread("Photos/dronepic.jpg")

height, width = img.shape[:2]

center_y = height//2
center_x = width//2

img = cv.rectangle(img, (center_x-30, center_y-30), (center_x+30, center_y+30), (0,255,0), 20)


line_l = 11

img = cv.line(img, (center_x - line_l, center_y), (center_x + line_l, center_y), (0,0,255), 4)
img = cv.line(img, (center_x, center_y - line_l), (center_x, center_y + line_l), (0,0,255), 4)


text = f"HEDEFE KİLİTLENDİ (X: {center_x}, Y: {center_y})"
img = cv.putText(img, text, (center_x - 240, center_y - 60), cv.FONT_HERSHEY_COMPLEX, 1.0, (255,255,255), 2)

cv.imshow("last", img)
cv.waitKey(0)
cv.destroyAllWindows()

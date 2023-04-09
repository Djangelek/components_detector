import cv2
import numpy as np
import math as mt
from skimage import io
from skimage.filters import threshold_otsu
import skimage
import matplotlib.pyplot as plt

#obtenemos la imagen original y la pasamos a grises
img_org = cv2.imread("images\org.jpg")

#código para cambiar el tamaño de la imagen
scale_percent = 25 # percent of original size
width = int(img_org.shape[1] * scale_percent / 100)
height = int(img_org.shape[0] * scale_percent / 100)
dim = (width, height)

img_re = cv2.resize(img_org, dim, interpolation = cv2.INTER_AREA)
img_gray = cv2.cvtColor(img_re, cv2.COLOR_BGR2GRAY)

#hacemos un blurr
img_brd = cv2.GaussianBlur(img_gray, (5, 5), 0)

#hacemos un threshold
(T, threshImg) = cv2.threshold(img_brd, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

#Aplicamos componentes conectados
conn = 4
output = cv2.connectedComponentsWithStats(threshImg, conn, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

for i in range(2, numLabels):
  x = stats[i, cv2.CC_STAT_LEFT]
  y = stats[i, cv2.CC_STAT_TOP]
  w = stats[i, cv2.CC_STAT_WIDTH]
  h = stats[i, cv2.CC_STAT_HEIGHT]
  area = stats[i, cv2.CC_STAT_AREA]
  (cX, cY) = centroids[i]

  output = img_re.copy()
  cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
  cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

  print("Label No {}".format(i))
  cv2.imshow(" ",output)
  cv2.waitKey(0)  
  cv2.destroyAllWindows()

print(centroids)


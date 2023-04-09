import cv2
import numpy as np
import math as mt
from skimage import io
from skimage.filters import threshold_otsu
import skimage
import matplotlib.pyplot as plt

#obtenemos la imagen original y la pasamos a grises
img_org = cv2.imread("images\org.jpg")
img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

#hacemos un blurr
img_brd = cv2.GaussianBlur(img_gray, (5, 5), 0)

#hacemos un threshold


plt.imshow(img_brd, cmap='gray')
plt.title("Original")
plt.show()
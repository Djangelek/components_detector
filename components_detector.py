import cv2
import numpy as np
import math as mt
from skimage import io
from skimage.filters import threshold_otsu
import skimage
import matplotlib.pyplot as plt

#código para cambiar el tamaño de la imagen
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

#obtenemos la imagen original y la pasamos a grises
img_org = cv2.imread("images\org.jpg")
img_re = ResizeWithAspectRatio(img_org, width=480)
img_gray = cv2.cvtColor(img_re, cv2.COLOR_BGR2GRAY)

#hacemos un blurr
img_brd = cv2.GaussianBlur(img_gray, (5, 5), 0)

#hacemos un threshold
(T, threshImg) = cv2.threshold(img_brd, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

#Aplicamos componentes conectados
conn = 4
output = cv2.connectedComponentsWithStats(threshImg, conn, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

for i in range(0, numLabels):
	if i == 0:
		text = "examining component {}/{} (background)".format(
			i + 1, numLabels)
	else:
		text = "examining component {}/{}".format( i + 1, numLabels)
	print("[INFO] {}".format(text))
	x = stats[i, cv2.CC_STAT_LEFT]
	y = stats[i, cv2.CC_STAT_TOP]
	w = stats[i, cv2.CC_STAT_WIDTH]
	h = stats[i, cv2.CC_STAT_HEIGHT]
	area = stats[i, cv2.CC_STAT_AREA]
	(cX, cY) = centroids[i]

for i in range(0, numLabels):

  # extract the connected component statistics and centroid for
  # the current label
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
  cv2.line(output, (0, int(cY)), (img_re.shape[1], int(cY)), (255, 0, 0), 2)
  cv2.line(output, (int(cX), 0), (int(cX), img_re.shape[0]), (255, 0, 0), 2)
  cv2.imshow(" ",output)
  cv2.waitKey(0)
  # Imprimir las medidas y área del objeto
  print("Label No {}: Longitud: {}, Altura: {}, Área: {}".format(i, max(w, h), min(w, h), area))

centroides = []
for i in range(1, numLabels):
    # extract the connected component statistics and centroid for
    # the current label
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[i]
    # ensure the width, height, and area are all neither too small
    # nor too big
    keepWidth = w > 80 and w < 100
    keepHeight = h > 80 and h < 100
    keepArea = area > 5000 and area < 7000
    # ensure the connected component we are examining passes all
    # three tests
    if all((keepWidth, keepHeight, keepArea)):
        # construct a mask for the current connected component and
        # then take the bitwise OR with the mask
        output = img_re.copy()
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print("Label No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, max(w, h), min(w, h), area, int(cX), int(cY)))
        centroides.append((int(cX), int(cY)))
        cv2.imshow(" ",output)
        cv2.waitKey(0)

#imprimir los centroides guardados en el diccionario

for centroid in centroides:
    cX, cY = centroid
    print("Coordenadas: cX = {}, cY = {}".format(cX, cY))
    
centroidesCoincidentes = []
umbral_x = 10  # Umbral para la cercanía en el eje de las x
# Comparar las coordenadas en el eje de las x
for i in range(len(centroides)):
    for j in range(i+1, len(centroides)):
        cX1, cY1 = centroides[i]
        cX2, cY2 = centroides[j]
        if abs(cX1 - cX2) < umbral_x:
            print("Las coordenadas ({}, {}) y ({}, {}) están cerca en el eje de las x".format(cX1, cY1, cX2, cY2))
            #Guardar en centroides coincidentes 
            centroidesCoincidentes.append((int(cX1), int(cY1)))
            centroidesCoincidentes.append((int(cX2), int(cY2)))
            
umbral_y = 10  # Umbral para la cercanía en el eje de las x
# Comparar las coordenadas en el eje de las y
for i in range(len(centroides)):
    for j in range(i+1, len(centroides)):
        cX1, cY1 = centroides[i]
        cX2, cY2 = centroides[j]
        if abs(cY1 - cY2) < umbral_y:
            print("Las coordenadas ({}, {}) y ({}, {}) están cerca en el eje de las y".format(cX1, cY1, cX2, cY2))
            centroidesCoincidentes.append((int(cX1), int(cY1)))
            centroidesCoincidentes.append((int(cX2), int(cY2)))

#Encontrar coordenada repetida en centroidesCoincidentes
for i in range(len(centroidesCoincidentes)):
    for j in range(i + 1, len(centroidesCoincidentes)):
        if centroidesCoincidentes[i] == centroidesCoincidentes[j]:
            print(centroidesCoincidentes[i])

cv2.destroyAllWindows()

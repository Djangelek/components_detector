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
img_org = cv2.imread("images\orgp5.jpg")
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
  print("Label No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, max(w, h), min(w, h), area, int(cX), int(cY)))
  if(i==0):
      Punto_centro_imagen = (int(cX), int(cY))
  elif(i==1):
      Punto_centro = (int(cX), int(cY))
      

# Calcular los centroides de los dos puntos
x0, y0 = Punto_centro_imagen
x1, y1 = Punto_centro

# Calcular la diferencia de coordenadas
tx = x0 - x1
ty = y0 - y1

# Crear la matriz de traslación
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

# Aplicar la traslación a la imagen
image_translated = cv2.warpAffine(img_re, translation_matrix, (img_re.shape[1], img_re.shape[0]))

img_gray = cv2.cvtColor(image_translated, cv2.COLOR_BGR2GRAY)
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

  output = image_translated.copy()
  cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
  cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

  print("Label No {}".format(i))
  cv2.line(output, (0, int(cY)), (output.shape[1], int(cY)), (255, 0, 0), 2)
  cv2.line(output, (int(cX), 0), (int(cX), output.shape[0]), (255, 0, 0), 2)
  cv2.imshow(" ",output)
  cv2.waitKey(0)
  # Imprimir las medidas y área del objeto
  print("Label No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, max(w, h), min(w, h), area, int(cX), int(cY)))

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
    keepWidth = w > 45 and w < 60
    keepHeight = h > 45 and h < 60
    keepArea = area > 1500 and area < 2000
    # ensure the connected component we are examining passes all
    # three tests
    if all((keepWidth, keepHeight, keepArea)):
        # construct a mask for the current connected component and
        # then take the bitwise OR with the mask
        output = image_translated.copy()
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
    
punto_cercano = None
distancia_minima = float('inf')
for i in range(len(centroides)):
    cX1, cY1 = centroides[i]
    distancia_total = 0
    for j in range(len(centroides)):
        if i != j:
            cX2, cY2 = centroides[j]
            distancia_total += mt.sqrt((cX1 - cX2) ** 2 + (cY1 - cY2) ** 2)
    if distancia_total < distancia_minima:
        distancia_minima = distancia_total
        punto_cercano = centroides[i]
print("El punto más cercano a los otros dos en términos de distancia en ambos ejes es: {}".format(punto_cercano))

Punto_esperado= (170,546)

# Coordenadas del punto de referencia (esquina izquierda superior)
x1, y1 = Punto_centro_imagen
# Coordenadas del primer punto (360, 125)
x2, y2 = punto_cercano
# Coordenadas del segundo punto (118, 590)
x3,y3 = Punto_esperado

# Paso 1: Calcular las diferencias en las coordenadas x y y para el primer punto
delta_x1 = x2 - x1
delta_y1 = y2 - y1

# Paso 2: Calcular las diferencias en las coordenadas x y y para el segundo punto
delta_x2 = x3 - x1
delta_y2 = y3 - y1

# Paso 3: Calcular el ángulo de rotación para ambos puntos en radianes
theta_rad1 = mt.atan2(delta_y1, delta_x1)
theta_rad2 = mt.atan2(delta_y2, delta_x2)

# Paso 4: Calcular la diferencia de ángulos entre los dos puntos
delta_theta_rad = theta_rad2 - theta_rad1

# Paso 5: Convertir la diferencia de ángulos a grados
delta_theta_deg = mt.degrees(delta_theta_rad)

#imprimir el ángulo de rotación
print("El ángulo de rotación es: {}".format(delta_theta_deg))

# Obtener la matriz de transformación de rotación
rotation_matrix = cv2.getRotationMatrix2D(Punto_centro_imagen, -delta_theta_deg, 1.0)

# Aplicar la matriz de transformación a la imagen
rotated_image = cv2.warpAffine(image_translated.copy(), rotation_matrix, (image_translated.copy().shape[1], image_translated.copy().shape[0]))
cv2.imshow('Imagen rotada', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
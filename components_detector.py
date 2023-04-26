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
img_org = cv2.imread("images\orgp7.jpg") 
img_re = ResizeWithAspectRatio(img_org, width=480)
img_gray = cv2.cvtColor(img_re, cv2.COLOR_BGR2GRAY)

#hacemos un threshold
(T, threshImg) = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

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

  print("Label No {}".format(i))
  # Imprimir las medidas, área, centroide del objeto
  print("Label No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, max(w, h), min(w, h), area, int(cX), int(cY)))
  
  #Detectamos el punto centro de la pieza
  if(i==1):
      Punto_centro = (int(cX), int(cY))

#Punto centro de la imagen
Punto_centro_imagen = img_re.shape[1] // 2, img_re.shape[0] // 2
 
# Coordenada de los centroides de los dos puntos
x0, y0 = Punto_centro_imagen
x1, y1 = Punto_centro

# Calcular la diferencia de coordenadas
tx = x0 - x1
ty = y0 - y1

# Crear la matriz de traslación
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

# Aplicar la traslación a la imagen
image_translated = cv2.warpAffine(img_re, translation_matrix, (img_re.shape[1], img_re.shape[0]))

#obtenemos la imagen Trasladada y la pasamos a grises
img_gray = cv2.cvtColor(image_translated, cv2.COLOR_BGR2GRAY)

#hacemos un threshold
(T, threshImg) = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

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
  # Imprimir las medidas, área y centroide del objeto
  print("Label No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, max(w, h), min(w, h), area, int(cX), int(cY)))

#Centroides de los puntos de referencia
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

        print("Label No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, max(w, h), min(w, h), area, int(cX), int(cY)))
        centroides.append((int(cX), int(cY)))
        cv2.imshow(" ",output)
        cv2.waitKey(0)
        
#imprimir los centroides guardados en el diccionario
for centroid in centroides:
    cX, cY = centroid
    print("Coordenadas: cX = {}, cY = {}".format(cX, cY))
    
#Hallar punto de referencia mas cercano a los otros dos
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

#Punto esperado es donde deberia estar el punto mas cercano a los otros (esto se calibra manualmente)
Punto_esperado= (170,546)

# Coordenadas del punto de referencia (Centro de la imagen)
x1, y1 = Punto_centro_imagen
# Coordenadas del primer punto (punto_cercano)
x2, y2 = punto_cercano
# Coordenadas del segundo punto (Punto_esperado)
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

# Convertir la imagen a espacio de color HSV
hsv_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2HSV)

# Definir el rango de colores a detectar (en este caso, color rojo)
lower_red = np.array([0, 40, 40])  # Valor mínimo de H, S y V para rojo
upper_red = np.array([10, 255, 255])  # Valor máximo de H, S y V para rojo
# Crear una máscara que filtre los píxeles dentro del rango de colores definido
mask = cv2.inRange(hsv_image, lower_red, upper_red)
# Aplicar la máscara a la imagen original para obtener la imagen segmentada
segmented_image = cv2.bitwise_and(rotated_image, rotated_image, mask=mask)
# Crear una máscara que filtre los píxeles dentro del rango de colores definido
mask = cv2.inRange(hsv_image, lower_red, upper_red)
# Aplicar filtro de dilatación a la máscara
kernel = np.ones((5, 5), np.uint8)
dilated_mask = cv2.dilate(mask, kernel, iterations=1)
# Mostrar la imagen original y la imagen segmentada
cv2.imshow('Imagen Segmentada ROJO', dilated_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Hacemos un threshold
(T, threshImg) = cv2.threshold(dilated_mask, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# Aplicamos componentes conectados
conn = 4
output = cv2.connectedComponentsWithStats(dilated_mask, conn, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

xwyh_Rojos = []
xy_Rojos=[]
centroides_rojos = []
for i in range(0, numLabels):
  # extract the connected component statistics and centroid for
  # the current label
  x = stats[i, cv2.CC_STAT_LEFT]
  y = stats[i, cv2.CC_STAT_TOP]
  w = stats[i, cv2.CC_STAT_WIDTH]
  h = stats[i, cv2.CC_STAT_HEIGHT]
  area = stats[i, cv2.CC_STAT_AREA]
  (cX, cY) = centroids[i]
  #Guardamos algunas variables de los objetos encontrados en los array
  xwyh_Rojos.append((x+w, y+h))
  xy_Rojos.append((x, y))
  centroides_rojos.append((int(cX),int(cY)))
  # Imprimir las medidas y área del objeto
  if(i>0):
    print("Punto Rojo No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, max(w, h), min(w, h), area, int(cX), int(cY)))
  
# Contar la cantidad de contornos encontrados (que representan los puntos rojos)(Se resta 1 para elimnar el fondo encontrado)
cantidad_puntos = numLabels-1 
# Mostrar la cantidad de puntos detectados
print("Cantidad de puntos rojos detectados:", cantidad_puntos)

# Definir el rango de colores a detectar (en este caso, color Azul)
lower_blue = np.array([90, 60, 60])  # Valor mínimo de H, S y V para azul
upper_blue= np.array([130, 255, 255])  # Valor máximo de H, S y V para azul
# Crear una máscara que filtre los píxeles dentro del rango de colores definido
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
# Aplicar la máscara a la imagen original para obtener la imagen segmentada
segmented_image = cv2.bitwise_and(rotated_image, rotated_image, mask=mask)
# Crear una máscara que filtre los píxeles dentro del rango de colores definido
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
# Aplicar filtro de dilatación a la máscara
kernel = np.ones((5, 5), np.uint8)
dilated_mask = cv2.dilate(mask, kernel, iterations=1)
# Mostrar la imagen original y la imagen segmentada
cv2.imshow('Imagen Segmentada Azul', dilated_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Hacemos un threshold
(T, threshImg) = cv2.threshold(dilated_mask, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# Aplicamos componentes conectados
conn = 4
output = cv2.connectedComponentsWithStats(dilated_mask, conn, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

xwyh_Azules = []
xy_Azules=[]
centroides_azules = []
for i in range(0, numLabels):
  # extract the connected component statistics and centroid for
  # the current label
  x = stats[i, cv2.CC_STAT_LEFT]
  y = stats[i, cv2.CC_STAT_TOP]
  w = stats[i, cv2.CC_STAT_WIDTH]
  h = stats[i, cv2.CC_STAT_HEIGHT]
  area = stats[i, cv2.CC_STAT_AREA]
  (cX, cY) = centroids[i]
  
  #Guardamos algunas variables de los objetos encontrados en los array
  xwyh_Azules.append((x+w, y+h))
  xy_Azules.append((x, y))
  centroides_azules.append((int(cX),int(cY)))
  
  # Imprimir las medidas y área del objeto menos la del fondo
  if(i>0):
    print("Punto Azul No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, max(w, h), min(w, h), area, int(cX), int(cY)))

cantidad_puntos = numLabels-1
# Mostrar la cantidad de puntos detectados
print("Cantidad de puntos Azules detectados:", cantidad_puntos)

# Definir el rango de colores a detectar (en este caso, color verde)
lower_green = np.array([40, 40, 40])  # Valor mínimo de H, S y V para verde
upper_green = np.array([70, 255, 255])  # Valor máximo de H, S y V para verde
# Crear una máscara que filtre los píxeles dentro del rango de colores definido
mask = cv2.inRange(hsv_image, lower_green, upper_green)
# Aplicar la máscara a la imagen original para obtener la imagen segmentada
segmented_image = cv2.bitwise_and(rotated_image, rotated_image, mask=mask)
# Crear una máscara que filtre los píxeles dentro del rango de colores definido
mask = cv2.inRange(hsv_image, lower_green, upper_green)
# Aplicar filtro de dilatación a la máscara
kernel = np.ones((5, 5), np.uint8)
dilated_mask = cv2.dilate(mask, kernel, iterations=1)

# Mostrar la imagen original y la imagen segmentada
cv2.imshow('Imagen Segmentada VERDE', dilated_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Hacemos un threshold
(T, threshImg) = cv2.threshold(dilated_mask, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# Aplicamos componentes conectados
conn = 4
output = cv2.connectedComponentsWithStats(dilated_mask, conn, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

xwyh_verdes = []
xy_Verdes=[]
centroides_verdes = []
for i in range(0, numLabels):
  # extract the connected component statistics and centroid for
  # the current label
  x = stats[i, cv2.CC_STAT_LEFT]
  y = stats[i, cv2.CC_STAT_TOP]
  w = stats[i, cv2.CC_STAT_WIDTH]
  h = stats[i, cv2.CC_STAT_HEIGHT]
  area = stats[i, cv2.CC_STAT_AREA]
  (cX, cY) = centroids[i]
  #Guardamos algunas variables de los objetos encontrados en los array
  xwyh_verdes.append((x+w,y+h))
  xy_Verdes.append((x, y))
  centroides_verdes.append((int(cX),int(cY)))
  # Imprimir las medidas, área y centroide del objeto menos la del fondo
  if(i>0):
      print("Punto Verde No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, max(w, h), min(w, h), area, int(cX), int(cY)))

# Contar la cantidad de contornos encontrados (que representan los puntos verdes)(Se resta 1 para elimnar el fondo encontrado)
cantidad_puntos = numLabels-1
# Mostrar la cantidad de puntos detectados
print("Cantidad de puntos verdes detectados:", cantidad_puntos)

#La tolerancia es la distancia máxima que se permite entre dos puntos para considerarlos iguales, Esto se necesita porque la rotacion puede darnos 1 o 2 pixeles de diferencia en coordenadas en diferentes casos
tolerancia=3
def comparar_coordenadas(coord1, coord2):
    dif_x = abs(coord1[0] - coord2[0])
    dif_y = abs(coord1[1] - coord2[1])
    if dif_x <= tolerancia and dif_y <= tolerancia:
        return True
    else:
        return False

output = rotated_image.copy()
      
#Recorrer los centroides rojos
centroide_rojos_Original=[(239,329),(296,342),(195,405),(244,407),(177,459),(308,475),(237,483)]
#Aplicar funcion comparar_coordenadas para cada centroide rojo con cada centroide_rojos_Original
for i in range(1, len(centroides_rojos)):
    for centroide_rojo_Original in centroide_rojos_Original:
        if comparar_coordenadas(centroides_rojos[i], centroide_rojo_Original):
           cv2.rectangle(output, xy_Rojos[i], xwyh_Rojos[i], (0, 255, 0), 3)
           break
        else:
            cv2.rectangle(output, xy_Rojos[i], xwyh_Rojos[i], (0, 0, 255), 3)
            
#Recorrer los centroides azules
centroide_azules_Original=[(311, 252),( 185,  337),(  300,  422),(  230,  536)]
#Aplicar funcion comparar_coordenadas para cada centroide azul con cada centroide_azules_Original
for i in range(1, len(centroides_azules)):
    for centroide_azul_Original in centroide_azules_Original:
        if comparar_coordenadas(centroides_azules[i], centroide_azul_Original):
           cv2.rectangle(output, xy_Azules[i], xwyh_Azules[i], (0, 255, 0), 3)
           break
        else:
            cv2.rectangle(output, xy_Azules[i], xwyh_Azules[i], (0, 0, 255), 3)
    
#Recorrer los centroides verdes
centroide_verdes_Original=[ (227,278),(273,291),(255,375),(168,382)]
#aplicar funcion comparar_coordenadas para cada centroide verde con cada centroide_verdes_Original
for i in range(1,len(centroides_verdes)):
    for centroide_verde_Original in centroide_verdes_Original:
        if comparar_coordenadas(centroides_verdes[i], centroide_verde_Original):
           cv2.rectangle(output, xy_Verdes[i], xwyh_verdes[i], (0, 255, 0), 3)
           break
        else:
            cv2.rectangle(output, xy_Verdes[i], xwyh_verdes[i], (0, 0, 255), 3)
    
#Mostrar la imagen final
cv2.imshow("Final",output)

# Esperar a que se presione una tecla y cerrar las ventanas 
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
import math as mt
from skimage import io
from skimage.filters import threshold_otsu
import skimage
import matplotlib.pyplot as plt

import os
from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox

# Crea un tercer botón para guardar la dirección de la imagen actual y ejecutar una función
def save_and_execute():
    global current_image
    image_path = os.path.join(image_folder, image_names[current_image])
    with open("image_path.txt", "w") as f:
        f.write(image_path)
    my_function(image_path, False)
    

def calibrarPrograma():
    global current_image
    image_path = os.path.join(image_folder, image_names[current_image])
    with open("image_path.txt", "w") as f:
        f.write(image_path)
    my_function(image_path, True)
    global ProgramaCalibrado
    ProgramaCalibrado=True

# Función para cambiar la imagen actual y mostrarla en el widget de lienzo
def change_image(direction):
    global current_image
    current_image += direction

    # Verifica si la imagen actual está fuera de los límites de la lista de imágenes
    if current_image < 0:
        current_image = len(image_list) - 1
    elif current_image >= len(image_list):
        current_image = 0

    # Muestra la nueva imagen en el widget de lienzo
    imagen1_tk = image_list[current_image]
    label1.config(image=imagen1_tk)
    
def change_image2():
    global current_image
    current_image=(len(image_list)-1)
    imagen1_tk = image_list[current_image]
    label1.config(image=imagen1_tk)


# Función para ejecutar en la imagen actual
def my_function(image_path, calibrar):
    # Ejecuta alguna función en la imagen actual
    if(calibrar==True):
        Proceso(image_path,True)
        global CalibracionLista
        CalibracionLista=True
    elif(calibrar==False and ProgramaCalibrado==True):
        Proceso(image_path, False)
    else:
        messagebox.showerror("Error", "Primero tiene que calibrar el programa.")
        return
        
    # Cambia la imagen de salida
    global imagen2_tk
    
    img2 = Image.open("output/output.jpg")
    img2 = img2.resize((450, 450), Image.LANCZOS)
    photo2 = ImageTk.PhotoImage(img2)
    imagen2_tk = photo2
    label2.config(image=imagen2_tk)
    
    texto1.configure(text="Amarillo: "+str(NumeroLegosAmarillos) +"/"+str(len(centroide_Amarillos_Original)))
    texto2.configure(text="Azul: "+str(NumeroLegosAzules)+"/"+str(len(centroide_azules_Original)))
    texto3.configure(text="Rojo: "+str(NumeroLegosRojos)+"/"+str(len(centroide_rojos_Original)))
    texto4.configure(text="Lugar Incorrecto: "+str(NumeroLegosIncorrectos))
    if (NumeroLegosIncorrectos==0 and (NumeroLegosAmarillos+NumeroLegosAzules+NumeroLegosRojos)==(len(centroide_rojos_Original)+len(centroide_azules_Original)+len(centroide_Amarillos_Original))):
        texto5.configure(text="Estado: Aprobado", foreground="green", font=("Trebuchet",12 ))

    else:
        texto5.configure(text="Estado: No Aprobado", foreground="red", font=("Trebuchet",12 ))
n_image= 0
def capture_image():
    global image_path
    global image_names
    global image_folder
    global image_list
    global label1
    global n_image

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    global frame
    while True:
        # Leer el cuadro actual de la cámara
        ret, frame = cap.read()

        # Mostrar el cuadro en una ventana llamada "Captura de imagen"
        cv2.imshow('Captura de imagen', frame)

        # Esperar a que el usuario presione la tecla 'Enter'
        if cv2.waitKey(1) == 13:
            break

    # Guardar la imagen capturada en un archivo llamado "imagen_capturada.jpg"
    cv2.imwrite('images/imagen_capturada{}.jpg'.format(n_image), frame)

    # Liberar la cámara y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()
    image_path = os.path.join(image_folder, 'imagen_capturada{}.jpg'.format(n_image))
    n_image +=1
    img = Image.open(image_path)
    frame = img.resize((200, 200), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(frame)
    image_list.append(photo)
    image_names.append('imagen_capturada{}.jpg'.format(n_image))
    change_image2()
    print(len(image_list))
    messagebox.showinfo("Photo Taken", "Photo saved successfully!")
# Llamar a la función para capturar la imagen
# capture_image()

#Si no existe carpeta output la crea
if not os.path.exists('output'):
    os.makedirs('output')

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


def Proceso(img_ubicacion, calibrar):
    global output, NumeroLegosAmarillos, NumeroLegosAzules, NumeroLegosRojos
    global NumeroLegosIncorrectos
    NumeroLegosAmarillos=0
    NumeroLegosAzules=0
    NumeroLegosRojos=0
    NumeroLegosIncorrectos=0
    
    img_org=cv2.imread(img_ubicacion)
    #obtenemos la imagen original y la pasamos a grises
    img_re = ResizeWithAspectRatio(img_org, width=480)
    img_gray = cv2.cvtColor(img_re, cv2.COLOR_BGR2GRAY)
    #hacemos un threshold
    (T, threshImg) = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow(" ",threshImg)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
    #Aplicamos componentes conectados
    conn = 4
    output = cv2.connectedComponentsWithStats(threshImg, conn, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    for i in range(0, 2):
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
      print("Label No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, w, h, area, int(cX), int(cY)))
      output = img_re.copy()
      cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
      cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
      cv2.imshow(" ",output)
      cv2.waitKey(0) 
      cv2.destroyAllWindows()
      
      #Detectamos el Lego centro de la pieza
      if(i==1):
          Lego_centro = (int(cX), int(cY))
          

    #Lego centro de la imagen
    Lego_centro_imagen = img_re.shape[1] // 2, img_re.shape[0] // 2
    # Coordenada de los centroides de los dos Legos
    x0, y0 = Lego_centro_imagen
    x1, y1 = Lego_centro
    # Calcular la diferencia de coordenadas
    tx = x0 - x1
    ty = y0 - y1
    # Crear la matriz de traslación
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    # Aplicar la traslación a la imagen
    image_translated = cv2.warpAffine(img_re, translation_matrix, (img_re.shape[1], img_re.shape[0]))
    
    #Ver imagen trasladada
    cv2.imshow(" ",image_translated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    hsv_image = cv2.cvtColor(image_translated, cv2.COLOR_BGR2HSV)
    # Aplicar umbral para obtener los Legos negros

    lower_black = np.array([0, 0, 0])  # Valor mínimo de H, S y V para negro
    upper_black = np.array([179, 255, 30])  # Valor máximo de H, S y V para negro

    mask = cv2.inRange(hsv_image, lower_black, upper_black)
    segmented_image = cv2.bitwise_and(image_translated, image_translated, mask=mask)
    
    kernel = np.ones((10, 10), np.uint8)
    segmented_image = cv2.dilate(mask, kernel, iterations=1)


    cv2.imshow('Contornos', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Aplicamos componentes conectados
    conn = 8
    output = cv2.connectedComponentsWithStats(segmented_image, conn, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
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
      print("Label No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, w, h, area, int(cX), int(cY)))
      #imptimti imagen
      
      
    #Centroides de los Legos de referencia
    centroides = []
    for i in range(0, numLabels):
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
        keepWidth = w > 8 and w < 30
        keepHeight = h > 8 and h < 30
        keepArea = area > 80 and area < 700
        # ensure the connected component we are examining passes all
        # three tests
        if all((keepWidth, keepHeight, keepArea)):
            # construct a mask for the current connected component and
            # then take the bitwise OR with the mask
            output = image_translated.copy()
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

            print("Label No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, w, h, area, int(cX), int(cY)))
            centroides.append((int(cX), int(cY)))
            
            #Visualizar la imagen
            cv2.imshow(" ",output)
            cv2.waitKey(0)       
    #imprimir los centroides guardados en el diccionario
    for centroid in centroides:
        cX, cY = centroid
        print("Coordenadas: cX = {}, cY = {}".format(cX, cY))
    #Hallar Lego de referencia mas cercano a los otros dos
    Lego_cercano = None
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
            Lego_cercano = centroides[i]
    print("El Lego más cercano a los otros dos en términos de distancia en ambos ejes es: {}".format(Lego_cercano))
    #Lego esperado es donde deberia estar el Lego mas cercano a los otros (esto se calibra manualmente)
    Lego_esperado= (136,347)
    # Coordenadas del Lego de referencia (Centro de la imagen)
    x1, y1 = Lego_centro_imagen
    # Coordenadas del primer Lego (Lego_cercano)
    x2, y2 = Lego_cercano
    # Coordenadas del segundo Lego (Lego_esperado)
    x3,y3 = Lego_esperado
    # Paso 1: Calcular las diferencias en las coordenadas x y y para el primer Lego
    delta_x1 = x2 - x1
    delta_y1 = y2 - y1
    # Paso 2: Calcular las diferencias en las coordenadas x y y para el segundo Lego
    delta_x2 = x3 - x1
    delta_y2 = y3 - y1
    # Paso 3: Calcular el ángulo de rotación para ambos Legos en radianes
    theta_rad1 = mt.atan2(delta_y1, delta_x1)
    theta_rad2 = mt.atan2(delta_y2, delta_x2)
    # Paso 4: Calcular la diferencia de ángulos entre los dos Legos
    delta_theta_rad =  theta_rad1-theta_rad2
    # Paso 5: Convertir la diferencia de ángulos a grados
    delta_theta_deg = mt.degrees(delta_theta_rad)
    #imprimir el ángulo de rotación
    print("El ángulo de rotación es: {}".format(delta_theta_deg))
    # Obtener la matriz de transformación de rotación
    rotation_matrix = cv2.getRotationMatrix2D(Lego_centro_imagen, delta_theta_deg, 1.0)
    # Aplicar la matriz de transformación a la imagen
    rotated_image = cv2.warpAffine(image_translated.copy(), rotation_matrix, (image_translated.copy().shape[1], image_translated.copy().shape[0]))
    cv2.imshow('Imagen rotada', rotated_image)
    cv2.waitKey(0)
    
    if(calibrar==True):
        global centroide_rojos_Original
        global centroide_azules_Original
        global centroide_Amarillos_Original
        global LyA_rojos_Original
        global LyA_azules_Original
        global LyA_Amarillos_Original
    
        centroide_rojos_Original=[]
        centroide_azules_Original=[]
        centroide_Amarillos_Original=[]
        LyA_rojos_Original=[]
        LyA_azules_Original=[]
        LyA_Amarillos_Original=[]
        
    LyA_rojos=[]
    LyA_Azules=[]
    LyA_Amarillos=[]
        
    # Convertir la imagen a espacio de color HSV
    hsv_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2HSV)
    # Definir el rango de colores a detectar (en este caso, color rojo)
    lower_red = np.array([0, 50, 50])  # Valor mínimo de H, S y V para rojo
    upper_red = np.array([10, 255, 255])  # Valor máximo de H, S y V para rojo
    # Crear una máscara que filtre los píxeles dentro del rango de colores definido
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    # Crear una máscara que filtre los píxeles dentro del rango de colores definido
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    # Aplicar filtro de dilatación a la máscara
    kernel = np.ones((10, 10), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    # Mostrar la imagen original y la imagen segmentada
    cv2.imshow('Imagen Segmentada ROJO', dilated_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    kernel = np.ones((10, 10), np.uint8)
    apertura = cv2.morphologyEx(dilated_mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Imagen Segmentada Roja', apertura)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Hacemos un threshold
    (T, threshImg) = cv2.threshold(apertura, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Aplicamos componentes conectados
    conn = 4
    output = cv2.connectedComponentsWithStats(apertura, conn, cv2.CV_32S)
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
      LyA_rojos.append(((w,h)))
      centroides_rojos.append((int(cX),int(cY)))
      
    # Imprimir las medidas y área del objeto
      if(i>0):
        print("Lego Rojo No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, w, h, area, int(cX), int(cY)))  
        if calibrar==True:
            centroide_rojos_Original.append((int(cX),int(cY)))
            LyA_rojos_Original.append(((w,h)))
    # Contar la cantidad de contornos encontrados (que representan los Legos rojos)(Se resta 1 para elimnar el fondo encontrado)
    cantidad_Legos = numLabels-1 
    # Mostrar la cantidad de Legos detectados
    print("Cantidad de Legos rojos detectados:", cantidad_Legos)
    NumeroLegosIncorrectos+=cantidad_Legos
    print(centroide_rojos_Original)
    
    
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
    
    kernel = np.ones((10, 10), np.uint8)
    apertura = cv2.morphologyEx(dilated_mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Imagen Segmentada Azul', apertura)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Hacemos un threshold
    (T, threshImg) = cv2.threshold(apertura, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Aplicamos componentes conectados
    conn = 4
    output = cv2.connectedComponentsWithStats(apertura, conn, cv2.CV_32S)
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
      LyA_Azules.append(((w,h)))
      
    # Imprimir las medidas y área del objeto menos la del fondo
      if(i>0):
        print("Lego Azul No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, w, h, area, int(cX), int(cY)))
        if calibrar==True:
            centroide_azules_Original.append((int(cX),int(cY)))
            LyA_azules_Original.append((w, h))
    cantidad_Legos = numLabels-1
    # Mostrar la cantidad de Legos detectados
    print("Cantidad de Legos Azules detectados:", cantidad_Legos)
    NumeroLegosIncorrectos+=cantidad_Legos
    
    # Definir el rango de colores a detectar (en este caso, color Amarillo)
    lower_green = np.array([20, 100, 100])  # Valor mínimo de H, S y V para Amarillo
    upper_green = np.array([25, 255, 255])  # Valor máximo de H, S y V para Amarillo
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
    cv2.imshow('Imagen Segmentada Amarillo', dilated_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #Aplicar filtro 
    kernel = np.ones((10, 10), np.uint8)
    apertura = cv2.morphologyEx(dilated_mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Imagen Segmentada Amarillo', apertura)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Hacemos un threshold
    (T, threshImg) = cv2.threshold(apertura, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Aplicamos componentes conectados
    conn = 4
    output = cv2.connectedComponentsWithStats(apertura, conn, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    xwyh_Amarillos = []
    xy_Amarillos=[]
    centroides_Amarillos = []
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
      xwyh_Amarillos.append((x+w,y+h))
      xy_Amarillos.append((x, y))
      centroides_Amarillos.append((int(cX),int(cY)))
      LyA_Amarillos.append((w, h))
      
    # Imprimir las medidas, área y centroide del objeto menos la del fondo
      if(i>0):
          print("Lego Amarillo No {}: Longitud: {}, Altura: {}, Área: {}, Centroide: ({}, {})".format(i, w, h, area, int(cX), int(cY)))
          if calibrar==True:
            centroide_Amarillos_Original.append((int(cX),int(cY)))
            LyA_Amarillos_Original.append((w,h))
    # Contar la cantidad de contornos encontrados (que representan los Legos Amarillos)(Se resta 1 para elimnar el fondo encontrado)
    cantidad_Legos = numLabels-1
    # Mostrar la cantidad de Legos detectados
    print("Cantidad de Legos Amarillos detectados:", cantidad_Legos)
    NumeroLegosIncorrectos+=cantidad_Legos
    
    #La tolerancia es la distancia máxima que se permite entre dos Legos para considerarlos iguales, Esto se necesita porque la rotacion puede darnos 1 o 2 pixeles de diferencia en coordenadas en diferentes casos
    tolerancia=10
    def comparar_coordenadas(coord1, coord2):
        print(coord1, coord2)
        dif_x = abs(coord1[0] - coord2[0])
        dif_y = abs(coord1[1] - coord2[1])
        if dif_x <= tolerancia and dif_y <= tolerancia:
            print("True")
            return True
        else:
            print("False")
            return False
        
    #Comparar largo y alto
    def comparar_LyA(LyA1, LyA2):
        print(LyA1, LyA2)
        dif_L = abs(LyA1[0] - LyA2[0])
        dif_A = abs(LyA1[1] - LyA2[1])
        if dif_L <= 5 and dif_A <= 5:
            print("True")
            return True
        elif dif_L <= tolerancia and dif_A <= tolerancia:
            #comprar si los dos tinen el mismo lado de mayor longitud
            if LyA1[0]>LyA1[1]:
                if LyA2[0]>LyA2[1]:
                    print("True")
                    return True
                else:
                    print("False")
                    return False
            elif LyA1[0]<LyA1[1]:
                if LyA2[0]<LyA2[1]:
                    print("True")
                    return True
                else:
                    print("False")
                    return False
        else:
            print("False")
            return False
        

    output = rotated_image.copy()
    
    print(centroide_azules_Original, centroide_Amarillos_Original, centroide_rojos_Original)
      
    
    #Aplicar funcion comparar_coordenadas para cada centroide rojo con cada centroide_rojos_Original
    for i in range(1, len(centroides_rojos)):
        for centroide_rojo_Original in centroide_rojos_Original:
            ubicacion=centroide_rojos_Original.index(centroide_rojo_Original)
            if comparar_coordenadas(centroides_rojos[i], centroide_rojo_Original):
                if comparar_LyA(LyA_rojos_Original[ubicacion], LyA_rojos[i]):
                    cv2.rectangle(output, xy_Rojos[i], xwyh_Rojos[i], (0, 255, 0), 3)
                    NumeroLegosRojos+=1
                    break
                else:
                    cv2.rectangle(output, xy_Rojos[i], xwyh_Rojos[i], (0, 0, 255), 3)
                    break
            else:
                cv2.rectangle(output, xy_Rojos[i], xwyh_Rojos[i], (0, 0, 255), 3)     
                   
    #Recorrer los centroides azules
    #Aplicar funcion comparar_coordenadas para cada centroide azul con cada centroide_azules_Original
    for i in range(1, len(centroides_azules)):
        for centroide_azul_Original in centroide_azules_Original:
            ubicacion=centroide_azules_Original.index(centroide_azul_Original)
            if comparar_coordenadas(centroides_azules[i], centroide_azul_Original):
                if comparar_LyA(LyA_azules_Original[ubicacion], LyA_Azules[i]):
                    cv2.rectangle(output, xy_Azules[i], xwyh_Azules[i], (0, 255, 0), 3)
                    NumeroLegosAzules+=1
                    break
                else:
                    cv2.rectangle(output, xy_Azules[i], xwyh_Azules[i], (0, 0, 255), 3)
                    break
            else:
                cv2.rectangle(output, xy_Azules[i], xwyh_Azules[i], (0, 0, 255), 3)
                
    #Recorrer los centroides Amarillos
    #aplicar funcion comparar_coordenadas para cada centroide Amarillo con cada centroide_Amarillos_Original
    for i in range(1,len(centroides_Amarillos)):
        for centroide_Amarillo_Original in centroide_Amarillos_Original:
            ubicacion=centroide_Amarillos_Original.index(centroide_Amarillo_Original)
            if comparar_coordenadas(centroides_Amarillos[i], centroide_Amarillo_Original):
                if comparar_LyA(LyA_Amarillos_Original[ubicacion], LyA_Amarillos[i]):
                    cv2.rectangle(output, xy_Amarillos[i], xwyh_Amarillos[i], (0, 255, 0), 3)
                    NumeroLegosAmarillos+=1
                    break
                else:
                    cv2.rectangle(output, xy_Amarillos[i], xwyh_Amarillos[i], (0, 0, 255), 3)
                    break
            else:
                cv2.rectangle(output, xy_Amarillos[i], xwyh_Amarillos[i], (0, 0, 255), 3)
                
    NumeroLegosIncorrectos=NumeroLegosIncorrectos-NumeroLegosRojos-NumeroLegosAzules-NumeroLegosAmarillos 
    print(LyA_rojos_Original, LyA_azules_Original, LyA_Amarillos_Original)
    #Mostrar la imagen final
    cv2.imshow("Final",output)
    cv2.imwrite("output/output.jpg", output)
    # Esperar a que se presione una tecla y cerrar las ventanas 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ProgramaCalibrado=False

##########################################################################
# Interfaz 
# Crea una ventana principal
root = Tk()
#  Guardamos el largo y alto de la ventana
wventana = 720
hventana = 700
root.maxsize(wventana, hventana ) 
root.title("Mostrar imágenes")
root.configure(background="#7cbc9a")

#  Obtenemos el largo y  ancho de la pantalla
wtotal = root.winfo_screenwidth()
htotal = root.winfo_screenheight()

#  Aplicamos la siguiente formula para calcular donde debería posicionarse
pwidth = round(wtotal/2-wventana/2)
pheight = round(htotal/2-hventana/2)

#  Se lo aplicamos a la geometría de la ventana
root.geometry(str(wventana)+"x"+str(hventana)+"+"+str(pwidth)+"+"+str(pheight))

#creamos frame para mostrar titulo
title_frame = Frame(root)
title_frame.grid(row=0, column=0,columnspan=2, padx=10, pady=5)

#añadimos titulo
label = Label(title_frame, text="Inspector De Posicionamiento De Componentes")
label.pack()
label.config(fg="#fff",font=("Impact",24),background="#7cbc9a")

#Crea un widget de lienzo
canvas = Canvas(root)
canvas.grid(row=1, column=1, padx=10, pady=5)

left_frame = Frame(root)
left_frame.config(width="400", height="200", bd=10,relief="sunken")
left_frame.grid(row=1, column=0, padx=10, pady=5)

# Accede a la carpeta de imágenes y guarda los nombres de los archivos en una lista
image_folder = "images"
image_names = os.listdir(image_folder)

# Inicializa la imagen actual y la lista de imágenes
current_image = 0
image_list = []

print(image_names)
print(len(image_list))
# Carga todas las imágenes en la lista de imágenes
for image_name in image_names:
    image_path = os.path.join(image_folder, image_name)
    img = Image.open(image_path)
    img = img.resize((200, 200), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    image_list.append(photo)
# Muestra la primera imagen en el widget de lienzo
label1 = Label(left_frame)
label1.pack()
imagen1_tk = image_list[current_image]
label1.config(image=imagen1_tk)

#muestra la segunda imagen el widget de lienzo
label2 = Label(canvas)
img = Image.open("images/"+image_names[0])
img = img.resize((450, 450), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(img)
label2.config(image=photo)
label2.pack()


texto1 = Label(left_frame,text="Amarillo: 0/0 ", font=("Trebuchet",12 ))
texto1.pack()
texto2 = Label(left_frame,text="Azul: 0/0 ", font=("Trebuchet",12 ))
texto2.pack()
texto3 = Label(left_frame,text="Rojo: 0/0", font=("Trebuchet",12 ))
texto3.pack()
texto4 = Label(left_frame,text="Lugar incorrecto: 0", font=("Trebuchet",12 ))
texto4.pack()
texto5 = Label(left_frame,text="Estado: No calibrado ", font=("Trebuchet",12))
texto5.pack()

ratio = 1/10

# Boton para navegar entre las imagenes
previous_image = PhotoImage(file = r"layout\previous.png")
previous_button = Button(root, text="   Anterior", image = previous_image, compound = LEFT, background="#ed6d4a", foreground="#fff",command=lambda: change_image(-1))
previous_button.place(relx =ratio, rely = 0.85, width=100,anchor='c')

# boton para calibrar programa
calibrate_image = PhotoImage(file = r"layout\calibrate.png")
calibrar_button = Button(root, text="Calibrar   ", image =calibrate_image, compound = RIGHT, background="#c46564", foreground="#fff",command=calibrarPrograma)
calibrar_button.place(relx =ratio*3, rely = 0.9, width=100, height=40,anchor='c')

# Boton para ejecutar
execute_image = PhotoImage(file = r"layout\execute.png")
save_button = Button(root, text="Ejecutar   ", image =execute_image, compound = RIGHT, background="#23998e", foreground="#fff", command=save_and_execute)
save_button.place(relx =ratio*5, rely = 0.9, width=100, height=40,anchor='c')

# Boton para agregar foto
add_image = PhotoImage(file = r"layout\add-picture.png")
add_button = Button(root, text="Agregar   ", image =add_image, compound = RIGHT, background="#09738a", foreground="#fff", command=capture_image)
add_button.place(relx =ratio*7, rely = 0.9, width=100, height=40,anchor='c')

# Boton para navegar entre las imagenes
next_image = PhotoImage(file = r"layout\next.png")
next_button = Button(root, text="Siguiente   ", image = next_image, compound = RIGHT, background="#3fd97f", foreground="#fff",command=lambda: change_image(1))
next_button.place(relx =ratio*9, rely = 0.85, width=100,anchor='c')

root.mainloop()
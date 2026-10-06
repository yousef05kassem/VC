Práctica 2: Funciones básicas de OpenCV

1. Conteo por filas con Canny
He utilizado el detector de Canny en la imagen del mandril para encontrar los bordes. Para contar los píxeles blancos por fila sin usar un bucle `for` lento, he usado la función `np.sum(canny, axis=1)`. Después, he calculado el máximo (`maxfil`) y he buscado las filas que superan el 90% de ese valor para pintarlas con una línea roja horizontal (`cv2.line`).

2. Comparación con Sobel y Umbralizado
En la segunda parte, he aplicado el operador Sobel. Primero usé un suavizado Gaussiano para reducir el ruido, calculé los gradientes en X e Y, y convertí el resultado a 8 bits. Para sacar los bordes gruesos apliqué `cv2.threshold`. 
Igual que en la tarea anterior, conté los píxeles blancos por filas y columnas usando `np.sum` y marqué las que superan el 90% del máximo con líneas rojas y verdes. Finalmente, usé `cv2.absdiff` para ver la diferencia visual exacta entre los bordes finos de Canny y los bordes procesados de Sobel.

3. Demostrador interactivo (Webcam)
Para la tarea abierta, he decidido hacer una reinterpretación de la instalación **Virtual air guitar**. Como el objetivo era usar la cámara web en vivo y procesar la imagen de forma fluida, decidí programar un "Light Painting" (Pincel de luz).
En vez de intentar trackear manos de forma compleja, el programa pasa el vídeo a escala de grises y usa `cv2.minMaxLoc` para buscar el punto más brillante de la escena (por ejemplo, la linterna del móvil). Si el brillo pasa de cierto umbral estricto, pinta un círculo en un lienzo negro auxiliar. Ese lienzo se suma al vídeo original con `cv2.addWeighted`, dejando un rastro de luz por donde muevo la linterna en el aire. Funciona en tiempo real sin tirones porque no hay bucles iterativos a nivel de píxel.


Práctica 2: Funciones básicas de OpenCV

1. Conteo por filas con Canny
He utilizado el detector de Canny en la imagen del mandril para encontrar los bordes. Para contar los píxeles blancos por fila sin usar un bucle `for` lento, he usado la función `np.sum(canny, axis=1)`. Después, he calculado el máximo (`maxfil`) y he buscado las filas que superan el 90% de ese valor para pintarlas con una línea roja horizontal (`cv2.line`).

2. Comparación con Sobel y Umbralizado
En la segunda parte, he aplicado el operador Sobel. Primero usé un suavizado Gaussiano para reducir el ruido, calculé los gradientes en X e Y, y convertí el resultado a 8 bits. Para sacar los bordes gruesos apliqué `cv2.threshold`. 
Igual que en la tarea anterior, conté los píxeles blancos por filas y columnas usando `np.sum` y marqué las que superan el 90% del máximo con líneas rojas y verdes. Finalmente, usé `cv2.absdiff` para ver la diferencia visual exacta entre los bordes finos de Canny y los bordes procesados de Sobel.

3. Demostrador interactivo (Webcam)
Privacy Pixelator es una reinterpretación de My little piece of privacy (Niklas Roy): todo lo que se mueve delante de la webcam se pixela en directo. Para usarlo, instala las dependencias con pip install opencv-python numpy y ejecuta python privacy_pixelator.py; muévete o saluda frente a la cámara y pulsa ESC para salir, y la grabación se guarda como privacy_pixelator.mp4. El método consiste en la diferencia de fotogramas (absdiff), un umbral (threshold), operaciones morfológicas (MORPH_OPEN y MORPH_CLOSE), una memoria de movimiento (addWeighted), el pixelado y la composición final con np.where, usando solo procesamiento de imagen clásico y sin aprendizaje profundo. Quien no se mueve no se pixela, ya que es una propiedad de la diferencia de fotogramas.

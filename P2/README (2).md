Práctica 2

Autor: Yousef Ahmed Kassem

Contenido
- `VC_P2.ipynb`: cuaderno con las tareas resueltas.
- `mandril.jpg`: imagen usada en las tareas.

Tarea 1. Cuenta por filas en Canny
Se parte de la imagen de Canny con umbrales 100 y 200, que solo tiene valores 0 o 255. Con `cv2.reduce` se suman los valores de cada fila, igual que en el ejemplo por columnas pero cambiando el eje (1 en lugar de 0). El resultado se divide entre 255 y el número de columnas, así se obtiene el porcentaje de píxeles blancos de cada fila.
Después se calcula el máximo (`maxfil`) y con `np.where` se buscan las filas con un valor mayor o igual que `0.90*maxfil`. Se muestran por pantalla el número de filas y sus posiciones. Para resaltarlas, la imagen de Canny se pasa a RGB y se dibuja una línea roja con `cv2.line` en cada fila seleccionada. Al lado se muestra la gráfica de la cuenta por filas, girada para que cada fila quede a la misma altura que en la imagen, con una línea discontinua en el valor `0.90*maxfil`.
El máximo es de un 43 % de píxeles blancos y salen 7 filas: en la frente (6–21) y justo debajo de los ojos (88 y 100). Son las zonas con más pelo y con más bordes, mientras que la nariz casi no tiene.

Tarea 2. Sobel umbralizado vs. Canny
Se umbraliza Sobel (convertido a 8 bits) con un valor de 130 y se hace el conteo por filas y columnas, también para Canny. Las filas y columnas por encima del 0.90*máximo se marcan sobre el mandril. Por filas los dos detectores coinciden bastante (frente y ojos). Por columnas Canny marca los dos lados de la cara y Sobel solo una columna junto a la nariz. La comparación completa está en el cuaderno.

Tarea 3. Privacy Pixelator
Reinterpretación de *My little piece of privacy* de Niklas Roy: todo lo que se mueve delante de la webcam se pixela. Se detecta el movimiento con diferencia de fotogramas, se limpia la máscara con operaciones morfológicas y se pixela esa zona reduciendo y ampliando la imagen. El resultado se guarda en `privacy_pixelator.mp4`. Se sale con ESC.

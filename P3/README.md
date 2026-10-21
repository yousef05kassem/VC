# Práctica 3 – Detección y reconocimiento de formas

Autor: Yousef Kassem · Cuaderno: `VC_P3.ipynb`

## Tarea 1: Dinero en la imagen

Umbralizado de Otsu, contornos externos y `minEnclosingCircle` para medir el diámetro de cada moneda. Con un clic en la moneda de 1€ se calcula la relación mm/píxel (23.25 mm / diámetro en píxeles) y cada moneda se asigna al diámetro oficial más cercano. Los objetos no circulares (área / área del círculo < 0.8) no se cuentan.

Cambios respecto al ejemplo: las 8 monedas en el diccionario, calibración por clic en lugar de un valor fijo, contornos en lugar de Hough (radios más precisos) y elección de la moneda más cercana.

Resultado: **3.88 €** en la imagen ideal y **7.76 €** en una segunda imagen con 16 monedas, todas correctas.

<img src="monedas.resultado.png" height="400"> <img src="monedas1.resultado.png" height="400">

El demostrador en vivo usa el mismo procedimiento en cada fotograma.

**Problemas en vivo:** las sombras agrandan el contorno, los cambios de luz hacen parpadear el resultado, al mover la cámara la calibración deja de valer, y las monedas que solo difieren 1 mm (5c/20c, 2c/10c, 1€/50c) se confunden. Las monedas solapadas no se cuentan. Mejora posible: usar también el color.

## Tarea 2: Clasificación de microplásticos

Segmentación: se estima el fondo con un filtro de mediana grande y se umbraliza la diferencia con la imagen, ya que el umbralizado del ejemplo no detecta los fragmentos claros de la imagen de test. Partículas extraídas: 74 FRA, 55 PEL, 54 TAR.

![Entrenamiento](2.1.resultados.png)

Características: compacidad, área/contenedor, ancho/alto, ejes de la elipse y dmin/dmax (de SMACC), más saturación y brillo relativo. No se usan área ni perímetro en píxeles porque dependen de la resolución.

Clasificador: distancia al centroide de cada clase con características normalizadas. Cada anotación del CSV se empareja con la partícula detectada de mayor solape (IoU > 0.4): 106 de 120 emparejadas.

<img src="2.2.resultados.png" height="500"> <img src="2.3.resultados.png" height="350">

| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| 0.78 | 0.79 | 0.78 | 0.78 |

La mayor confusión es entre fragmentos y pellets. El alquitrán se reconoce bien por su color oscuro.

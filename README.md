# Clasificación de Imágenes desde Cero (Cats vs Dogs)

Implementación y pipeline de Deep Learning para clasificación binaria de imágenes utilizando Keras y TensorFlow, basado en la guía oficial de [Keras: Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/).

---

## 🎯 Propósito del Proyecto

El objetivo principal de esta práctica es demostrar cómo diseñar, entrenar y evaluar una **Red Neuronal Convolucional (CNN)** desde cero, partiendo directamente de archivos de imagen JPEG en disco y **sin utilizar modelos preentrenados** (Transfer Learning).

Aspectos clave que aborda:
1. **Pipeline de datos eficiente:** Carga y estructuración por lotes directamente desde carpetas (`image_dataset_from_directory`) optimizado con `tf.data` (prefetching y paralelismo asíncrono).
2. **Aumento de datos (Data Augmentation):** Aplicación de transformaciones geométricas aleatorias (`RandomFlip`, `RandomRotation`) para enriquecer la variabilidad del dataset y prevenir el sobreajuste (*overfitting*).
3. **Arquitectura avanzada (Mini-Xception):**
   * Convoluciones separables en profundidad (`SeparableConv2D`) para reducir drásticamente el número de parámetros sin perder capacidad expresiva.
   * Conexiones residuales (*skip connections*) para facilitar la propagación del gradiente en capas profundas.
   * Normalización por lotes (`BatchNormalization`) y regularización mediante `Dropout`.
4. **Ciclo de entrenamiento e inferencia:** Validación en cada época, guardado automático de checkpoints (`ModelCheckpoint`), persistencia del modelo final (`final_model.keras`) y clasificación de nuevas imágenes.

---

## 📁 Estructura del Proyecto

* `train.py`: Script principal para descarga del dataset, limpieza de imágenes corruptas, construcción y entrenamiento de la red CNN.
* `predict.py`: Script de inferencia para evaluar imágenes individuales (Gato vs. Perro) con su porcentaje de probabilidad.
* `image_classification_from_scratch.ipynb`: Cuaderno de Jupyter interactivo con visualizaciones paso a paso y gráficas.
* `requirements.txt`: Dependencias necesarias del proyecto.
* `.gitignore`: Archivos y carpetas excluidos del control de versiones (datos pesados y modelos generados).

---

## 🚀 Requisitos e Instalación

### 1. Clonar el repositorio e instalar dependencias

```bash
git clone <URL_DEL_REPOSITORIO>
cd <CARPETA_DEL_PROYECTO>

pip install -r requirements.txt
```

### 2. Entrenar el modelo

Para ejecutar el entrenamiento con los parámetros estándar (25 épocas y lotes de 128):

```bash
python train.py
```

O para realizar una prueba rápida con menos épocas o diferente tamaño de lote:

```bash
python train.py --epochs 2 --batch-size 64
```

> **Nota:** La primera vez que se ejecute, el script descargará automáticamente el dataset de Kaggle (786 MB) y verificará la integridad de las imágenes.

### 3. Realizar predicciones

Una vez entrenado el modelo:

```bash
python predict.py --image "ruta/a/la_imagen.jpg"
```

### 4. Cuaderno interactivo (Jupyter)

También se puede abrir `image_classification_from_scratch.ipynb` en cualquier entorno compatible con Jupyter para ejecutar el flujo celda por celda.

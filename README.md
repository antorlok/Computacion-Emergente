# Clasificación de Imágenes desde Cero (Cats vs Dogs)
**Computación Emergente - Universidad José Antonio Páez (UJAP)**

Implementación avanzada y pipeline de Deep Learning para clasificación binaria de imágenes utilizando **Keras 3** y **TensorFlow**, basado en la guía canónica de [Keras: Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/).

---

## 🎯 Propósito del Proyecto

El objetivo principal es diseñar, entrenar y evaluar una **Red Neuronal Convolucional (CNN)** desde cero, partiendo directamente de archivos de imagen JPEG en disco y **sin utilizar modelos preentrenados** (Transfer Learning).

El proyecto demuestra cómo surgen (*emergen*) detectores de características jerárquicos (bordes, texturas, partes anatómicas) de forma autoorganizada a partir de los datos crudos mediante descenso de gradiente.

---

## ✨ Características y Mejoras Implementadas

### 1. ⚡ Rendimiento y Aceleración por Hardware (GPU)
* **Precisión Mixta (Mixed Precision FP16):** Integración nativa con `mixed_float16` para acelerar el cálculo con Tensor Cores (ej. NVIDIA RTX 3060) reduciendo el consumo de VRAM y acelerando el entrenamiento entre un 30% y 50%.
* **Crecimiento Dinámico de VRAM:** Asignación progresiva de memoria en GPU (`set_memory_growth`) para evitar bloqueos del sistema.

### 2. 🧹 Saneamiento y Validación Robusta de Datos (Pillow)
* **Filtrado exhaustivo:** Análisis de integridad de archivos con `PIL.Image.open` tanto a nivel de cabecera (`verify()`) como de decodificación completa de mapa de bits (`load()`), eliminando archivos corruptos o truncados de forma segura.

### 3. 🧪 Rigor Metodológico y Partición en 3 Vías
* **División 80% / 10% / 10%:**
  * **Entrenamiento (80%):** Ajuste de pesos de la red.
  * **Validación (10%):** Monitoreo de generalización y toma de decisiones en callbacks.
  * **Prueba / Test Set (10%):** Conjunto ciego evaluado únicamente al final para medir el rendimiento real sin sesgo.
* **Métricas Completas:** Monitoreo continuo de Exactitud (*Accuracy*), Precisión, Sensibilidad (*Recall*), F1-Score y Área bajo la Curva ROC (*AUC*).
* **Matriz de Confusión:** Generación y visualización automática en disco (`confusion_matrix.png`).

### 4. 🔄 Data Augmentation Enriquecido
* Transformaciones geométricas y fotométricas en el pipeline de `tf.data`:
  * Giro horizontal aleatorio (`RandomFlip`).
  * Rotación aleatoria (`RandomRotation`).
  * Zoom aleatorio (`RandomZoom`).
  * Ajuste aleatorio de contraste (`RandomContrast`).

### 5. 🧠 Arquitectura Mini-Xception y Callbacks Inteligentes
* **Convoluciones separables en profundidad (`SeparableConv2D`):** Reducción de parámetros computacionales manteniendo alta capacidad de representación.
* **Conexiones residuales (*Skip Connections*):** Mitigan la atenuación del gradiente en capas profundas.
* **`ModelCheckpoint` (Best Only):** Persistencia automática de `best_model.keras` únicamente cuando la pérdida de validación mejora.
* **`EarlyStopping`:** Detención temprana (paciencia de 5 épocas) con restauración de los mejores pesos (`restore_best_weights=True`).
* **`ReduceLROnPlateau`:** Reducción dinámica de la tasa de aprendizaje ante estancamientos en la pérdida de validación.
* **Generación automática de curvas de aprendizaje:** Guarda `training_history.png` con la evolución de *Loss*, *Accuracy*, *AUC* y *Learning Rate*.

---

## 📁 Estructura del Repositorio

* `train.py`: Script principal de entrenamiento con aceleración por hardware, particionado en 3 vías, callbacks y generación de reportes gráficos.
* `predict.py`: Script de inferencia que soporta predicción de una sola imagen o procesamiento por lotes de carpetas completas.
* `image_classification_from_scratch.ipynb`: Cuaderno Jupyter interactivo con visualización paso a paso y gráficas integradas.
* `requirements.txt`: Dependencias del entorno de Python.
* `training_history.png`: Gráfica generada tras el entrenamiento con las curvas de aprendizaje.
* `confusion_matrix.png`: Matriz de confusión generada tras evaluar el mejor modelo sobre el Test Set.
* `best_model.keras`: Checkpoint con los pesos óptimos obtenidos durante el entrenamiento.
* `final_model.keras`: Modelo resultante al finalizar todas las épocas.

---

## 🚀 Instalación y Uso

### 1. Clonar e instalar dependencias

```bash
git clone <URL_DEL_REPOSITORIO>
cd Computacion-Emergente

pip install -r requirements.txt
```

### 2. Entrenar el modelo

Para entrenar con la configuración estándar (25 épocas, batch size 128 y precisión mixta activada):

```bash
python train.py
```

Opciones disponibles en `train.py`:
* `--epochs N`: Número máximo de épocas (default: 25).
* `--batch-size N`: Tamaño de lote (default: 128).
* `--skip-download`: Omitir la descarga y saneamiento de datos si ya existen en `PetImages/`.
* `--no-mixed-precision`: Desactivar la precisión mixta FP16 y operar en FP32 estándar.

Ejemplo rápido de prueba:
```bash
python train.py --epochs 3 --batch-size 64
```

Al finalizar el entrenamiento, se generarán automáticamente:
1. `best_model.keras` (modelo con mejor desempeño en validación).
2. `training_history.png` (curvas de Loss, Accuracy, AUC y LR).
3. `confusion_matrix.png` (evaluación final sobre el 10% de Test).

### 3. Realizar predicciones

#### Inferencia sobre una imagen individual:
```bash
python predict.py --image "ruta/a/la_imagen.jpg"
```

#### Inferencia por lotes sobre una carpeta completa:
```bash
python predict.py --dir "ruta/a/carpeta_con_fotos/"
```

#### Especificar un modelo particular:
```bash
python predict.py --image "gato.jpg" --model "best_model.keras"
```

#### Inferencia con Explicabilidad (Grad-CAM):
```bash
python predict.py --image "gato.jpg" --gradcam
```

---

### 4. Explicabilidad con Grad-CAM (Gradient-weighted Class Activation Mapping)

Para inspeccionar qué características visuales (orejas, hocico, silueta, ojos) determinaron la decisión de la red neuronal convolucional:

```bash
python gradcam.py --image "ruta/a/la_imagen.jpg"
```

Opciones de `gradcam.py`:
* `--image`: Ruta de la imagen a inspeccionar (requerido).
* `--output`: Nombre del archivo de salida (por defecto: `gradcam_<nombre>.png`).
* `--colormap`: Paleta de colores para el mapa de calor (`jet`, `viridis`, `inferno`).

---

## 📊 Salidas Gráficas de Evaluación y Explicabilidad

Durante el ciclo de entrenamiento y análisis se generan automáticamente:

| Curvas de Aprendizaje (`training_history.png`) | Matriz de Confusión (`confusion_matrix.png`) | Explicabilidad (`gradcam_*.png`) |
|:---:|:---:|:---:|
| Evolución de pérdida, precisión, AUC y LR | Desempeño en clases Gato vs. Perro en Test Set | Mapa de calor sobre regiones determinantes |

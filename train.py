"""
Clasificación de Imágenes desde Cero (Cats vs Dogs)
Basado en el tutorial oficial de Keras:
https://keras.io/examples/vision/image_classification_from_scratch/
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import numpy as np

# Configuración de backend y dependencias de Keras/TensorFlow
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras
from keras import layers
import tensorflow as tf
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

DATA_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
ZIP_NAME = "kagglecatsanddogs_5340.zip"
DATA_DIR = "PetImages"


def download_and_extract_dataset(data_dir=DATA_DIR):
    """Descarga y extrae el dataset de Kaggle Cats vs Dogs si no existe."""
    if os.path.isdir(data_dir) and os.path.exists(os.path.join(data_dir, "Cat")) and os.path.exists(os.path.join(data_dir, "Dog")):
        print(f"[OK] El directorio '{data_dir}' ya existe y contiene los datos.")
        return

    if not os.path.exists(ZIP_NAME):
        print(f"Descargando el dataset desde {DATA_URL}...")

        def _reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\rProgreso: {percent:.1f}% ({mb_down:.1f}MB / {mb_total:.1f}MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(DATA_URL, ZIP_NAME, reporthook=_reporthook)
        print("\nDescarga completada.")

    print(f"Extrayendo '{ZIP_NAME}'...")
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall(".")
    print("Extracción completada.")


def filter_corrupted_images(data_dir=DATA_DIR):
    """Filtra y elimina imágenes corruptas que no contengan la cabecera JFIF."""
    print("Verificando y filtrando imágenes corruptas...")
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder_path):
            continue
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                with open(fpath, "rb") as fobj:
                    is_jfif = b"JFIF" in fobj.peek(10)
            except Exception:
                is_jfif = False

            if not is_jfif:
                num_skipped += 1
                try:
                    os.remove(fpath)
                except OSError:
                    pass

    print(f"Se eliminaron {num_skipped} imágenes corruptas (no-JFIF).")


def make_model(input_shape, num_classes):
    """Construye una red convolucional tipo Xception miniatura."""
    inputs = keras.Input(shape=input_shape)

    # Bloque inicial
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Conexión residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Proyección del residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    units = 1 if num_classes == 2 else num_classes

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de clasificación de imágenes Cats vs Dogs con Keras")
    parser.add_argument("--epochs", type=int, default=25, help="Número de épocas de entrenamiento (default: 25)")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamaño de lote (default: 128)")
    parser.add_argument("--skip-download", action="store_true", help="Omitir verificación y descarga de datos")
    args = parser.parse_args()

    # 1. Descarga y limpieza de datos
    if not args.skip_download:
        download_and_extract_dataset()
        filter_corrupted_images()

    image_size = (180, 180)
    batch_size = args.batch_size

    # 2. Cargar dataset con image_dataset_from_directory
    print(f"\nCargando datos desde '{DATA_DIR}'...")
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    # 3. Data Augmentation
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]

    def data_augmentation(images):
        for layer in data_augmentation_layers:
            images = layer(images)
        return images

    # Optimización del pipeline en CPU / GPU
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    # 4. Construir modelo
    print("\nConstruyendo el modelo CNN...")
    model = make_model(input_shape=image_size + (3,), num_classes=2)
    model.summary()

    # 5. Compilar
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    # 6. Callbacks y Entrenamiento
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]

    print(f"\nIniciando entrenamiento por {args.epochs} épocas (batch_size={batch_size})...")
    model.fit(
        train_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    # Guardar modelo final
    model.save("final_model.keras")
    print("\n[OK] Entrenamiento finalizado. Modelo guardado en 'final_model.keras'.")

    # 7. Probar inferencia en una imagen de muestra si existe
    sample_img_path = os.path.join(DATA_DIR, "Cat", "6779.jpg")
    if os.path.exists(sample_img_path):
        print(f"\nProbando inferencia en imagen de muestra: {sample_img_path}")
        img = keras.utils.load_img(sample_img_path, target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = keras.ops.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = float(keras.ops.sigmoid(predictions[0][0]))
        print(f"Resultado: {100 * (1 - score):.2f}% Gato, {100 * score:.2f}% Perro.")


if __name__ == "__main__":
    main()

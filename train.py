"""
Clasificación de Imágenes desde Cero (Cats vs Dogs)
Computación Emergente - UJAP

Implementación con Keras 3 y TensorFlow:
- Aceleración por hardware (GPU Memory Growth y Precisión Mixta FP16)
- Validación robusta de imágenes con PIL
- División en 3 particiones: Entrenamiento (80%), Validación (10%), Prueba (10%)
- Data Augmentation enriquecido (Flip, Rotation, Zoom, Contrast)
- Arquitectura Mini-Xception con convoluciones separables y conexiones residuales
- Callbacks inteligentes: ModelCheckpoint (best only), EarlyStopping y ReduceLROnPlateau
- Métricas integrales: Exactitud (Acc), Precisión, Recall y AUC
- Evaluación sobre Test Set independiente y generación de Matriz de Confusión
- Guardado automático de curvas de aprendizaje y matriz de confusión en PNG
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import numpy as np
from PIL import Image

# Configuración de backend Keras
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras
from keras import layers
import tensorflow as tf
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

DATA_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
ZIP_NAME = "kagglecatsanddogs_5340.zip"
DATA_DIR = "PetImages"


def setup_hardware(enable_mixed_precision=True):
    """Configura crecimiento de memoria GPU y política de precisión mixta (FP16)."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[Hardware] GPU(s) detectada(s): {[g.name for g in gpus]}. Crecimiento dinámico de VRAM activado.")
        except RuntimeError as e:
            print(f"[Hardware] Advertencia al configurar memoria GPU: {e}")

        if enable_mixed_precision:
            try:
                keras.mixed_precision.set_global_policy("mixed_float16")
                print("[Hardware] Precisión mixta ('mixed_float16') activada exitosamente para aceleración con Tensor Cores.")
            except Exception as e:
                print(f"[Hardware] No se pudo activar precisión mixta: {e}")
    else:
        print("[Hardware] No se detectó GPU dedicada. Ejecutando cómputo en CPU.")


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
    """
    Filtra y elimina imágenes corruptas o truncadas utilizando Pillow (PIL).
    Verifica tanto la cabecera (verify) como la decodificación completa de pixeles (load).
    """
    print("Verificando integridad de imágenes con Pillow...")
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder_path):
            continue
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            is_corrupt = False
            try:
                # 1. Comprobación de cabecera
                with Image.open(fpath) as img:
                    img.verify()
                # 2. Comprobación de decodificación completa (detecta imágenes truncadas)
                with Image.open(fpath) as img:
                    img.load()
            except Exception:
                is_corrupt = True

            if is_corrupt:
                num_skipped += 1
                try:
                    os.remove(fpath)
                except OSError:
                    pass

    print(f"[OK] Verificación completada. Se eliminaron {num_skipped} imágenes corruptas o ilegibles.")


def make_model(input_shape, num_classes):
    """
    Construye una red convolucional tipo Xception miniatura.
    Compatible con Precisión Mixta (la capa final Dense opera en float32).
    """
    inputs = keras.Input(shape=input_shape)

    # Bloque de entrada y normalización de píxeles
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Conexión residual inicial

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Proyección del residual para igualar dimensiones y canales
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
    # Importante: dtype="float32" en la última capa para estabilidad numérica en precisión mixta
    outputs = layers.Dense(units, activation=None, dtype="float32")(x)
    return keras.Model(inputs, outputs, name="mini_xception_classifier")


def plot_training_history(history, save_path="training_history.png"):
    """Genera y guarda en disco gráficas de pérdida, exactitud, AUC y métricas."""
    epochs_range = range(1, len(history.history["loss"]) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Función de Pérdida (Loss)
    axs[0, 0].plot(epochs_range, history.history["loss"], label="Entrenamiento", marker="o", linewidth=1.8)
    axs[0, 0].plot(epochs_range, history.history["val_loss"], label="Validación", marker="o", linewidth=1.8)
    axs[0, 0].set_title("Pérdida (Binary Crossentropy)", fontsize=13, fontweight="bold")
    axs[0, 0].set_xlabel("Épocas")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)

    # 2. Exactitud (Accuracy)
    axs[0, 1].plot(epochs_range, history.history["acc"], label="Entrenamiento", marker="o", linewidth=1.8)
    axs[0, 1].plot(epochs_range, history.history["val_acc"], label="Validación", marker="o", linewidth=1.8)
    axs[0, 1].set_title("Exactitud (Accuracy)", fontsize=13, fontweight="bold")
    axs[0, 1].set_xlabel("Épocas")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)

    # 3. AUC (Área Bajo la Curva ROC)
    if "auc" in history.history and "val_auc" in history.history:
        axs[1, 0].plot(epochs_range, history.history["auc"], label="Entrenamiento", marker="o", linewidth=1.8)
        axs[1, 0].plot(epochs_range, history.history["val_auc"], label="Validación", marker="o", linewidth=1.8)
        axs[1, 0].set_title("Área Bajo la Curva ROC (AUC)", fontsize=13, fontweight="bold")
        axs[1, 0].set_xlabel("Épocas")
        axs[1, 0].set_ylabel("AUC")
        axs[1, 0].legend()
        axs[1, 0].grid(True, linestyle="--", alpha=0.6)

    # 4. Tasa de Aprendizaje o Precisión / Recall
    lr_key = next((k for k in ("lr", "learning_rate") if k in history.history), None)
    if lr_key:
        axs[1, 1].plot(epochs_range, history.history[lr_key], label="Learning Rate", color="purple", marker="o", linewidth=1.8)
        axs[1, 1].set_title("Tasa de Aprendizaje Dinámica (LR)", fontsize=13, fontweight="bold")
        axs[1, 1].set_xlabel("Épocas")
        axs[1, 1].set_ylabel("Tasa de Aprendizaje")
        axs[1, 1].set_yscale("log")
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle="--", alpha=0.6)
    elif "val_precision" in history.history and "val_recall" in history.history:
        axs[1, 1].plot(epochs_range, history.history["val_precision"], label="Val Precision", marker="o", linewidth=1.8)
        axs[1, 1].plot(epochs_range, history.history["val_recall"], label="Val Recall", marker="o", linewidth=1.8)
        axs[1, 1].set_title("Validación: Precision vs Recall", fontsize=13, fontweight="bold")
        axs[1, 1].set_xlabel("Épocas")
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Curvas de entrenamiento guardadas en '{save_path}'.")


def plot_confusion_matrix(cm, class_names=["Gato", "Perro"], save_path="confusion_matrix.png"):
    """Grafica y guarda la matriz de confusión en formato PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Matriz de Confusión (Test Set Independiente)",
        ylabel="Etiqueta Real",
        xlabel="Etiqueta Predicha",
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Matriz de confusión guardada en '{save_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento optimizado de clasificación Cats vs Dogs con Keras 3")
    parser.add_argument("--epochs", type=int, default=25, help="Número de épocas de entrenamiento (default: 25)")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamaño de lote (default: 128)")
    parser.add_argument("--skip-download", action="store_true", help="Omitir descarga y saneamiento de datos")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Desactivar precisión mixta float16")
    args = parser.parse_args()

    # 1. Configuración de Hardware
    setup_hardware(enable_mixed_precision=not args.no_mixed_precision)

    # 2. Descarga y saneamiento de datos con PIL
    if not args.skip_download:
        download_and_extract_dataset()
        filter_corrupted_images()

    image_size = (180, 180)
    batch_size = args.batch_size

    # 3. Partición de datos en 3 vías: Train (80%), Val (10%), Test (10%)
    print(f"\nCargando datos desde '{DATA_DIR}'...")
    train_ds, val_test_full_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Dividir el 20% en 10% validación y 10% test independiente
    val_test_batches = tf.data.experimental.cardinality(val_test_full_ds).numpy()
    test_batches = val_test_batches // 2
    val_batches = val_test_batches - test_batches

    val_ds = val_test_full_ds.skip(test_batches)
    test_ds = val_test_full_ds.take(test_batches)

    print(f"[Dataset] Lotes - Entrenamiento: {tf.data.experimental.cardinality(train_ds).numpy()} | "
          f"Validación: {val_batches} | Prueba (Test): {test_batches}")

    # 4. Data Augmentation enriquecido
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ]

    def data_augmentation(images):
        for layer in data_augmentation_layers:
            images = layer(images)
        return images

    # Optimización del pipeline con tf.data y paralelismo asíncrono
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf_data.AUTOTUNE)

    # 5. Construcción del Modelo
    print("\nConstruyendo arquitectura Mini-Xception...")
    model = make_model(input_shape=image_size + (3,), num_classes=2)
    model.summary()

    # 6. Compilación con Métricas Completas (Acc, Precision, Recall, AUC)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    # 7. Callbacks Inteligentes
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 8. Entrenamiento
    print(f"\nIniciando entrenamiento por hasta {args.epochs} épocas (batch_size={batch_size})...")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    # Guardar modelo final y gráficas de aprendizaje
    model.save("final_model.keras")
    print("\n[OK] Modelo final persistido en 'final_model.keras'.")
    plot_training_history(history, save_path="training_history.png")

    # 9. Evaluación en el Conjunto de Prueba Independiente (Test Set)
    best_model_path = "best_model.keras" if os.path.exists("best_model.keras") else "final_model.keras"
    print(f"\nCargando el mejor modelo desde '{best_model_path}' para evaluación en Test...")
    eval_model = keras.models.load_model(best_model_path)

    print("\nEvaluando en el conjunto de prueba independiente (Test Set)...")
    test_results = eval_model.evaluate(test_ds, return_dict=True)

    print("\n" + "=" * 55)
    print("        RESULTADOS FINALES EN TEST SET (CIEGO)        ")
    print("=" * 55)
    for metric_name, val in test_results.items():
        print(f"  * {metric_name.upper():<15}: {val:.4f}")

    # 10. Matriz de Confusión y F1-Score en Test Set
    print("\nGenerando matriz de confusión en Test Set...")
    y_true_list = []
    y_pred_list = []

    for images, labels in test_ds:
        preds = eval_model.predict(images, verbose=0)
        probs = keras.ops.sigmoid(preds)
        probs_np = keras.ops.convert_to_numpy(probs)
        binary_preds = (probs_np > 0.5).astype(int).flatten()

        y_pred_list.extend(binary_preds)
        y_true_list.extend(keras.ops.convert_to_numpy(labels).flatten())

    y_true_arr = np.array(y_true_list, dtype=int)
    y_pred_arr = np.array(y_pred_list, dtype=int)

    tp = np.sum((y_true_arr == 1) & (y_pred_arr == 1))
    tn = np.sum((y_true_arr == 0) & (y_pred_arr == 0))
    fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
    fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))

    cm = np.array([[tn, fp], [fn, tp]])
    plot_confusion_matrix(cm, class_names=["Gato", "Perro"], save_path="confusion_matrix.png")

    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_val = (2 * precision_val * recall_val / (precision_val + recall_val)) if (precision_val + recall_val) > 0 else 0.0

    print(f"  * F1-SCORE (Perro): {f1_val:.4f}")
    print("=" * 55)

    # 11. Inferencia de prueba en una imagen si existe
    sample_img_path = os.path.join(DATA_DIR, "Cat", "6779.jpg")
    if os.path.exists(sample_img_path):
        print(f"\nProbando inferencia en imagen de muestra: {sample_img_path}")
        img = keras.utils.load_img(sample_img_path, target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = keras.ops.expand_dims(img_array, 0)
        predictions = eval_model.predict(img_array, verbose=0)
        score = float(keras.ops.sigmoid(predictions[0][0]))
        print(f"Resultado muestra: {100 * (1 - score):.2f}% Gato, {100 * score:.2f}% Perro.")


if __name__ == "__main__":
    main()

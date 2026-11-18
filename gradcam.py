"""
Implementación de Explicabilidad con Grad-CAM (Gradient-weighted Class Activation Mapping).
Computación Emergente - UJAP

Genera mapas de calor visuales para explicar en qué regiones de la imagen
(orejas, ojos, hocico, silueta) se enfoca la red neuronal para predecir 'Gato' o 'Perro'.

Uso:
    python gradcam.py --image PetImages/Cat/1000.jpg
    python gradcam.py --image PetImages/Dog/1000.jpg --output explicacion.png
"""

import argparse
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("KERAS_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".keras_cache"))

import keras
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf


def find_default_model():
    """Busca el mejor modelo disponible en orden de prioridad."""
    candidates = ["best_model.keras", "final_model.keras"]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def get_last_conv_layer_name(model):
    """Encuentra automáticamente la última capa convolucional de la red."""
    for layer in reversed(model.layers):
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D, keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No se encontró ninguna capa convolucional en el modelo.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    Calcula el mapa de calor Grad-CAM para la predicción de la imagen dada.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    # Sub-modelo que devuelve las activaciones de la última conv y el logit final
    grad_model = keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        logit = predictions[:, 0]
        # Si logit > 0 -> predicción PERRO (optimizamos logit positivo)
        # Si logit < 0 -> predicción GATO (optimizamos -logit para buscar lo que apoya a Gato)
        loss = tf.where(logit >= 0, logit, -logit)

    # Gradientes del objetivo respecto a los mapas de activación
    grads = tape.gradient(loss, conv_outputs)

    # Promedio espacial de los gradientes (peso de importancia alfa_k)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Ponderar los canales de la última capa convolucional
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Aplicar ReLU: solo nos interesan características con contribución positiva
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_heatmap(image_path, heatmap, alpha=0.4, colormap="jet"):
    """
    Superpone el mapa de calor sobre la imagen original a color (RGB).
    """
    # Cargar imagen original
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Redimensionar el mapa de calor a las dimensiones de la imagen original
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_img) / 255.0

    # Aplicar colormap (ej. Jet o Viridis)
    cmap = plt.colormaps[colormap]
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]  # Quitar canal alfa
    colored_heatmap = np.uint8(255 * colored_heatmap)

    # Superponer con ponderación alfa
    img_np = np.array(img)
    superimposed = np.uint8(img_np * (1.0 - alpha) + colored_heatmap * alpha)
    return img, heatmap_resized, superimposed


def explain_image(model, image_path, output_path=None, image_size=(180, 180), colormap="jet"):
    """
    Ejecuta inferencia y genera la visualización de explicabilidad Grad-CAM.
    Retorna (label, confidence, output_path).
    """
    # 1. Cargar y preparar imagen para el modelo
    raw_img = keras.utils.load_img(image_path, target_size=image_size)
    img_array = keras.utils.img_to_array(raw_img)
    img_tensor = keras.ops.expand_dims(img_array, 0)

    # 2. Inferencia
    preds = model.predict(img_tensor, verbose=0)
    score = float(keras.ops.sigmoid(preds[0][0]))
    is_dog = score >= 0.5
    prob_dog = score * 100
    prob_cat = (1 - score) * 100
    label = "PERRO" if is_dog else "GATO"
    confidence = prob_dog if is_dog else prob_cat

    # 3. Calcular mapa de calor Grad-CAM
    last_conv_name = get_last_conv_layer_name(model)
    heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name=last_conv_name)

    # 4. Superponer sobre la imagen original de resolución completa
    orig_img, resized_heatmap, superimposed = overlay_heatmap(image_path, heatmap, alpha=0.45, colormap=colormap)

    # 5. Generar figura comparativa de 3 paneles
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(orig_img)
    axs[0].set_title("Imagen Original", fontsize=12, fontweight="bold")
    axs[0].axis("off")

    im_heat = axs[1].imshow(resized_heatmap, cmap=colormap)
    axs[1].set_title(f"Grad-CAM ({last_conv_name})", fontsize=12, fontweight="bold")
    axs[1].axis("off")
    fig.colorbar(im_heat, ax=axs[1], fraction=0.046, pad=0.04)

    axs[2].imshow(superimposed)
    color_title = "#1b5e20" if is_dog else "#b71c1c"
    axs[2].set_title(f"Decisión: {label} ({confidence:.2f}%)", fontsize=13, fontweight="bold", color=color_title)
    axs[2].axis("off")

    plt.suptitle(f"Explicabilidad con Grad-CAM - Archivo: {os.path.basename(image_path)}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path is None:
        base_name, _ = os.path.splitext(os.path.basename(image_path))
        output_path = f"gradcam_{base_name}.png"

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n================ GRAD-CAM RESULT ================")
    print(f" Imagen:          {image_path}")
    print(f" Predicción:      {label} ({confidence:.2f}%)")
    print(f" Capa Analizada:  {last_conv_name}")
    print(f" Gráfica Visual:  {output_path}")
    print(f"=================================================")

    return label, confidence, output_path


def main():
    parser = argparse.ArgumentParser(description="Explicabilidad visual con Grad-CAM para Cats vs Dogs")
    parser.add_argument("--image", type=str, required=True, help="Ruta de la imagen a analizar")
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo .keras (default: best_model.keras)")
    parser.add_argument("--output", type=str, default=None, help="Ruta para guardar la imagen con el mapa de calor")
    parser.add_argument("--colormap", type=str, default="jet", help="Colormap de matplotlib (default: 'jet', ej: 'viridis', 'inferno')")
    args = parser.parse_args()

    model_path = args.model or find_default_model()
    if not model_path or not os.path.exists(model_path):
        print("Error: No se encontró un modelo entrenado (.keras).")
        print("Asegúrate de tener 'best_model.keras' en el directorio.")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Error: La imagen '{args.image}' no existe.")
        sys.exit(1)

    print(f"Cargando modelo desde '{model_path}'...")
    model = keras.models.load_model(model_path)

    explain_image(model, args.image, output_path=args.output, colormap=args.colormap)


if __name__ == "__main__":
    main()

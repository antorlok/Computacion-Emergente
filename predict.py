"""
Script de inferencia / predicción para el clasificador de Cats vs Dogs.
Computación Emergente - UJAP

Uso:
    python predict.py --image ruta/a/tu_imagen.jpg
    python predict.py --dir ruta/a/carpeta/
    python predict.py --image test.jpg --model best_model.keras
"""

import argparse
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras


def find_default_model():
    """Busca el mejor modelo disponible en orden de prioridad."""
    candidates = ["best_model.keras", "final_model.keras"]
    for cand in candidates:
        if os.path.exists(cand):
            return cand

    # Checkpoints save_at_*.keras
    checkpoints = [f for f in os.listdir(".") if f.startswith("save_at_") and f.endswith(".keras")]
    if checkpoints:
        checkpoints.sort()
        return checkpoints[-1]

    return None


def classify_single_image(model, image_path, image_size=(180, 180)):
    """Ejecuta inferencia sobre una sola imagen y retorna (cat_prob, dog_prob, label)."""
    try:
        img = keras.utils.load_img(image_path, target_size=image_size)
    except Exception as e:
        print(f"Error al abrir la imagen '{image_path}': {e}")
        return None

    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)
    score = float(keras.ops.sigmoid(predictions[0][0]))

    cat_prob = (1 - score) * 100
    dog_prob = score * 100
    label = "PERRO" if dog_prob > cat_prob else "GATO"

    return cat_prob, dog_prob, label


def main():
    parser = argparse.ArgumentParser(description="Predecir si una imagen o lote de imágenes contiene un gato o un perro")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Ruta de una imagen individual para clasificar")
    group.add_argument("--dir", type=str, help="Ruta de un directorio con imágenes para clasificar por lote")
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo .keras guardado (por defecto busca best_model.keras)")
    args = parser.parse_args()

    model_path = args.model or find_default_model()
    if not model_path or not os.path.exists(model_path):
        print("Error: No se encontró un modelo entrenado (.keras).")
        print("Ejecuta 'python train.py' primero para entrenar y generar 'best_model.keras'.")
        sys.exit(1)

    print(f"Cargando modelo desde '{model_path}'...")
    model = keras.models.load_model(model_path)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: La imagen '{args.image}' no existe.")
            sys.exit(1)

        result = classify_single_image(model, args.image)
        if result is None:
            sys.exit(1)
        cat_prob, dog_prob, label = result

        print("\n" + "=" * 45)
        print(f"Imagen:      {args.image}")
        print(f"Predicción:  {label}")
        print(f"Prob. Gato:  {cat_prob:.2f}%")
        print(f"Prob. Perro: {dog_prob:.2f}%")
        print("=" * 45)

    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: El directorio '{args.dir}' no existe.")
            sys.exit(1)

        files = [
            os.path.join(args.dir, f)
            for f in sorted(os.listdir(args.dir))
            if f.lower().endswith(image_extensions)
        ]

        if not files:
            print(f"No se encontraron imágenes válidas en '{args.dir}'.")
            sys.exit(0)

        print(f"\nClasificando {len(files)} imágenes en '{args.dir}':\n")
        print(f"{'Archivo':<35} | {'Clase':<7} | {'Confianza':<9}")
        print("-" * 57)

        cat_count = 0
        dog_count = 0

        for fpath in files:
            fname = os.path.basename(fpath)
            res = classify_single_image(model, fpath)
            if res:
                cat_p, dog_p, lbl = res
                conf = max(cat_p, dog_p)
                if lbl == "PERRO":
                    dog_count += 1
                else:
                    cat_count += 1
                print(f"{fname[:33]:<35} | {lbl:<7} | {conf:6.2f}%")

        print("-" * 57)
        print(f"Resumen: {cat_count} Gatos, {dog_count} Perros (Total: {len(files)})\n")


if __name__ == "__main__":
    main()

"""
Script de inferencia / predicción para el clasificador de Cats vs Dogs.
Uso:
    python predict.py --image ruta/a/tu_imagen.jpg [--model final_model.keras]
"""

import argparse
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras


def predict_image(image_path, model_path="final_model.keras", image_size=(180, 180)):
    if not os.path.exists(model_path):
        # Buscar el checkpoint más reciente si no existe final_model.keras
        checkpoints = [f for f in os.listdir(".") if f.startswith("save_at_") and f.endswith(".keras")]
        if checkpoints:
            checkpoints.sort()
            model_path = checkpoints[-1]
            print(f"Usando checkpoint disponible: {model_path}")
        else:
            print(f"Error: No se encontró el modelo '{model_path}'. Asegúrate de entrenarlo primero con train.py.")
            sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Error: La imagen '{image_path}' no existe.")
        sys.exit(1)

    print(f"Cargando modelo desde '{model_path}'...")
    model = keras.models.load_model(model_path)

    print(f"Procesando imagen '{image_path}'...")
    img = keras.utils.load_img(image_path, target_size=image_size)
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Crear batch dimension

    predictions = model.predict(img_array)
    score = float(keras.ops.sigmoid(predictions[0][0]))

    cat_prob = (1 - score) * 100
    dog_prob = score * 100

    print("\n" + "=" * 40)
    print(f"Predicción para: {image_path}")
    print(f"  Gato:  {cat_prob:.2f}%")
    print(f"  Perro: {dog_prob:.2f}%")
    print(f"Resultado final: {'PERRO' if dog_prob > cat_prob else 'GATO'}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predecir si una imagen es de un gato o un perro")
    parser.add_argument("--image", type=str, required=True, help="Ruta de la imagen a clasificar")
    parser.add_argument("--model", type=str, default="final_model.keras", help="Ruta al modelo .keras guardado")
    args = parser.parse_args()

    predict_image(args.image, args.model)

import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
#from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_label_binarizer(pickle_path="label_binarizer.pkl"):
    with open(pickle_path, "rb") as f:
        lb = pickle.load(f)
    return lb


def sort_unsorted_images(base_dir, model_path, label_binarizer, temp_dir="temporary", img_size=(224, 224), confidence_threshold=0.5):
    model = load_model(model_path)
    temp_path = os.path.join(base_dir, temp_dir)
    os.makedirs(temp_path, exist_ok=True)

    for file in os.listdir(base_dir):
        full_path = os.path.join(base_dir, file)
        if os.path.isfile(full_path) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = load_img(full_path, target_size=img_size)
                img_arr = img_to_array(img)
                img_arr = preprocess_input(img_arr)
                img_arr = np.expand_dims(img_arr, axis=0)

                prediction = model.predict(img_arr)[0]
                class_index = np.argmax(prediction)
                confidence = prediction[class_index]

                if confidence < confidence_threshold:
                    print(f"Skipped {file}: low confidence ({confidence:.2f})")
                    continue

                class_label = label_binarizer.classes_[class_index]
                class_folder = os.path.join(temp_path, class_label)
                os.makedirs(class_folder, exist_ok=True)
                shutil.move(full_path, os.path.join(class_folder, file))

                print(f"Moved {file} to {class_label} (confidence: {confidence:.2f})")

            except Exception as e:
                print(f"Skipped {file}: {e}")


if __name__ == "__main__":
    base_folder = "path/to/files"
    lb = load_label_binarizer("label_binarizer.pkl")
    sort_unsorted_images(base_dir=base_folder, model_path="classifier.h5", label_binarizer=lb, confidence_threshold=0.6)

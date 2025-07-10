import os
import shutil
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def collect_images(folder, target_size=(128, 128)):
    image_paths = []
    image_arrays = []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, file)
                try:
                    img = load_img(path, target_size=target_size)
                    img_array = img_to_array(img)
                    img_array = preprocess_input(img_array)
                    image_paths.append(path)
                    image_arrays.append(img_array)
                except Exception as e:
                    print(f"Skipping {path}: {e}")
    return image_paths, np.array(image_arrays)

def cluster_images(image_folder, n_clusters=5):
    image_paths, image_arrays = collect_images(image_folder)
    if not image_paths:
        print("No images found.")
        return

    X = image_arrays.reshape(len(image_arrays), -1)
    pca = PCA(n_components=50)
    X_reduced = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_reduced)

    for i, path in enumerate(image_paths):
        cluster_dir = os.path.join(image_folder, f"cluster_{labels[i]}")
        os.makedirs(cluster_dir, exist_ok=True)
        try:
            shutil.move(path, os.path.join(cluster_dir, os.path.basename(path)))
        except Exception as e:
            print(f"Failed to move {path}: {e}")

    return [(f"cluster_{i}", sum(labels == i)) for i in range(n_clusters)]

if __name__ == "__main__":
    input_folder = "path/to/inputfolder"
    info = cluster_images(input_folder, n_clusters=10)
    print("Cluster summary:", info)


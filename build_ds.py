import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def collect_images_by_top_folder(root_folder, ignore_folders={"Site", "Sites"}):
    data = []
    for top_folder in os.listdir(root_folder):
        top_path = os.path.join(root_folder, top_folder)
        if not os.path.isdir(top_path) or top_folder in ignore_folders:
            continue

        for root, _, files in os.walk(top_path):
            if any(part in ignore_folders for part in os.path.normpath(root).split(os.sep)):
                continue

            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    data.append((full_path, top_folder))  # Label = top folder name
    return data

def build_classification_dataframe(data):
    paths = [item[0] for item in data]
    labels = [item[1] for item in data]

    lb = LabelBinarizer()
    label_matrix = lb.fit_transform(labels)

    df = pd.DataFrame({'image_path': paths, 'label': labels})
    for i, class_name in enumerate(lb.classes_):
        df[class_name] = label_matrix[:, i]

    return df, lb

class SingleLabelImageGenerator(Sequence):
    def __init__(self, df, lb, batch_size=32, img_size=(224, 224), shuffle=True):
        self.df = df.reset_index(drop=True)
        self.lb = lb
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]

        X = []
        y = []

        for _, row in batch_df.iterrows():
            try:
                img = load_img(row['image_path'], target_size=self.img_size)
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                X.append(img_array)

                label_vec = row[self.lb.classes_].values.astype(np.float32)
                y.append(label_vec)

            except (UnidentifiedImageError, OSError) as e:
                ...#print(f"Skipping file: {row['image_path']} — {str(e)}")

        return np.array(X), np.array(y)

if __name__ == "__main__":
    root_folder = "path/to/imagefolder"
    data = collect_images_by_top_folder(root_folder)
    print(data)

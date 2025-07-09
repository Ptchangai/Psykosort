import os
import shutil
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pickle
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import pytesseract
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download("punkt")
nltk.download("stopwords")


cnn_model = None
ocr_model = None
cnn_lb = None
text_vectorizer = None
text_encoder = None

stop_words = set(stopwords.words('english')).union(stopwords.words('french'))

if os.path.exists("classifier.h5") and os.path.exists("label_binarizer.pkl"):
    cnn_model = load_model("classifier.h5")
    with open("label_binarizer.pkl", "rb") as f:
        cnn_lb = pickle.load(f)

ocr_available = False
if os.path.exists("text_classifier_model.h5") and os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("text_label_encoder.pkl"):
    ocr_model = load_model("text_classifier_model.h5")
    with open("tfidf_vectorizer.pkl", "rb") as f:
        text_vectorizer = pickle.load(f)
    with open("text_label_encoder.pkl", "rb") as f:
        text_encoder = pickle.load(f)
    ocr_available = True


def predict_cnn(image_path):
    if not cnn_model or not cnn_lb:
        return {}
    try:
        img = load_img(image_path, target_size=(224, 224))
        arr = img_to_array(img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        preds = cnn_model.predict(arr)[0]
        return {cnn_lb.classes_[i]: float(preds[i]) for i in range(len(preds))}
    except:
        return {}


def predict_ocr(image_path):
    if not ocr_available:
        return {}
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang='eng+fra')
        tokens = word_tokenize(text.lower())
        filtered = ' '.join([w for w in tokens if w.isalpha() and w not in stop_words])
        if not filtered.strip():
            return {}
        vec = text_vectorizer.transform([filtered])
        preds = ocr_model.predict(vec)[0]
        return {text_encoder.classes_[i]: float(preds[i]) for i in range(len(preds))}
    except:
        return {}

def combine_predictions(pred1, pred2):
    if not pred2:  # No OCR prediction
        return sorted(pred1.items(), key=lambda x: x[1], reverse=True)[:3]

    all_keys = set(pred1.keys()).union(set(pred2.keys()))
    combined = {}
    for k in all_keys:
        combined[k] = pred1.get(k, 0) * 0.5 + pred2.get(k, 0) * 0.5
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:3]


class ImageSorterGUI:
    def __init__(self, master):
        self.master = master
        self.image_predictions = []
        self.current_index = 0
        self.folder = ""

        self.label = tk.Label(master, text="Choose a folder to begin", font=("Arial", 14))
        self.label.pack()

        self.canvas = tk.Canvas(master, width=600, height=600)
        self.canvas.pack()

        self.button = tk.Button(master, text="Choose Folder", command=self.choose_folder)
        self.button.pack()

        master.bind_all("1", lambda e: self.move_image(0))
        master.bind_all("2", lambda e: self.move_image(1))
        master.bind_all("3", lambda e: self.move_image(2))
        master.bind_all("4", lambda e: self.skip_image())

    def choose_folder(self):
        self.folder = filedialog.askdirectory()
        raw_files = self.collect_images(self.folder)
        self.label.config(text=f"Running inference on {len(raw_files)} files...")
        self.master.update()

        self.image_predictions = []
        for i, file_path in enumerate(raw_files):
            pred1 = predict_cnn(file_path)
            pred2 = predict_ocr(file_path)
            combined = combine_predictions(pred1, pred2)
            suggestions = [x[0] for x in combined]
            self.image_predictions.append((file_path, suggestions))

            self.label.config(text=f"Processed {i+1}/{len(raw_files)}")
            self.master.update()

        method = "CNN + OCR" if ocr_available else "CNN only"
        self.label.config(text=f"Inference complete using {method}. Begin sorting.")
        self.current_index = 0
        self.next_image()

    def collect_images(self, folder):
        files = []
        for root, _, filenames in os.walk(folder):
            parts = os.path.normpath(root).split(os.sep)
            if any(x.lower() in {"site", "sites", "temporary"} for x in parts):
                continue
            for f in filenames:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    files.append(os.path.join(root, f))
        return sorted(files)

    def next_image(self):
        if self.current_index >= len(self.image_predictions):
            self.label.config(text="Done sorting all files.")
            self.canvas.delete("all")
            return

        path, suggestions = self.image_predictions[self.current_index]
        self.label.config(
            text=f"{self.current_index+1}/{len(self.image_predictions)} | Suggestions: " +
            f"1) {suggestions[0] if len(suggestions) > 0 else '-'}  " +
            f"2) {suggestions[1] if len(suggestions) > 1 else '-'}  " +
            f"3) {suggestions[2] if len(suggestions) > 2 else '-'}  | Press 4 to skip"
        )

        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((600, 600), Image.ANTIALIAS)
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(300, 300, image=self.tk_img)
        except Exception as e:
            print(f"Could not open image: {path} ({e})")
            self.skip_image()

    def move_image(self, choice_idx):
        path, suggestions = self.image_predictions[self.current_index]
        if choice_idx < len(suggestions):
            target_folder = os.path.join(self.folder, "temporary", suggestions[choice_idx])
            os.makedirs(target_folder, exist_ok=True)
            shutil.move(path, os.path.join(target_folder, os.path.basename(path)))
            print(f"Moved to {target_folder}")
        else:
            print("No valid suggestion.")
        self.current_index += 1
        self.next_image()

    def skip_image(self):
        print("Skipped.")
        self.current_index += 1
        self.next_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Psykosort - Smart Image Sorter")
    app = ImageSorterGUI(root)
    root.mainloop()

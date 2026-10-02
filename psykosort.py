import os
import shutil
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pickle
import cv2
import threading
import queue

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

if os.path.exists(cnn_model) and os.path.exists(cnn_lb):
    cnn_model = load_model(cnn_model, compile=False)
    with open(cnn_lb, "rb") as f:
        cnn_lb = pickle.load(f)

ocr_available = False
if os.path.exists(ocr_model) and os.path.exists(text_vectorizer) and os.path.exists(text_encoder):
    ocr_model = load_model(ocr_model)
    with open(text_vectorizer, "rb") as f:
        text_vectorizer = pickle.load(f)
    with open(text_encoder, "rb") as f:
        text_encoder = pickle.load(f)
    print("Using OCR model along with CNN model.")
    ocr_available = True


def predict_cnn(image_path):
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
    all_keys = set(pred1.keys()).union(pred2.keys())
    combined = {k: pred1.get(k, 0)*0.5 + pred2.get(k, 0)*0.5 for k in all_keys}
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:3]

class ImageSorterGUI:
    def __init__(self, master):
        self.master = master
        self.folder = ""
        self.image_paths = []
        self.image_predictions = []
        self.current_index = 0
        self.queue = queue.Queue()

        self.label = tk.Label(master, text="Choose a folder to begin", font=("Arial", 20))
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
        self.image_paths = self.collect_images(self.folder)
        self.image_predictions.clear()
        self.current_index = 0
        self.label.config(text=f"Found {len(self.image_paths)} images. Processing...")
        threading.Thread(target=self.run_inference, daemon=True).start()
        self.master.after(100, self.check_queue)

    def collect_images(self, folder):
        paths = []
        for root, _, files in os.walk(folder):
            if "temporary" in root.lower():
                continue
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(root, f))
        return sorted(paths)

    def run_inference(self):
        for i, path in enumerate(self.image_paths):
            pred1 = predict_cnn(path)
            pred2 = predict_ocr(path) if ocr_available else {}
            combined = combine_predictions(pred1, pred2)
            suggestions = [x[0] for x in combined]
            self.queue.put((path, suggestions))

    def check_queue(self):
        try:
            while True:
                item = self.queue.get_nowait()
                self.image_predictions.append(item)
                if len(self.image_predictions) == 1:
                    self.next_image()
        except queue.Empty:
            pass
        if self.current_index >= len(self.image_paths):
            self.label.config(text="Done sorting all files.")
        else:
            self.master.after(100, self.check_queue)

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
            img = img.resize((600, 600), Image.Resampling.LANCZOS)
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
        self.current_index += 1
        self.next_image()

    def skip_image(self):
        self.current_index += 1
        self.next_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Psykosort - Smart Image Sorter")
    app = ImageSorterGUI(root)
    root.mainloop()

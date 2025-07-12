import os
import pickle
import numpy as np

import pandas as pd
import csv

import cv2
import pytesseract
import nltk

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tqdm import tqdm
import matplotlib.pyplot as plt

for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english')).union(stopwords.words('french'))

def is_image_valid(path):
    try:
        img = cv2.imread(path)
        return img is not None and img.size > 0
    except Exception:
        return False

def extract_text(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang='eng+fra')
        return text.strip() if text else None
    except Exception as e:
        print(f"[OCR Error] {e}")
        return None

def preprocess_text(text):
    try:
        tokens = word_tokenize(text.lower())
    except Exception as e:
        print(f"[Tokenizer fallback] {e}")
        tokens = text.lower().split()
    return ' '.join([t for t in tokens if t.isalpha() and t not in stop_words])

def save_text_labels_to_csv(data, csv_path="ocr_text_labels.csv"):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "text"])
        for item in data:
            writer.writerow(item)

def load_text_labels_from_csv(csv_path="ocr_text_labels.csv"):
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append((row["image_path"], row["label"], row["text"]))
    return data

def build_dataset(base_folder, cache_file="ocr_text_labels.csv"):
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}...")
        return pd.DataFrame(load_text_labels_from_csv(cache_file), columns=['path', 'text', 'label'])

    print("Building dataset from image OCR...")
    data = []

    for label in tqdm(os.listdir(base_folder), desc="Labels"):
        class_path = os.path.join(base_folder, label)
        if not os.path.isdir(class_path):
            continue
        for root, _, files in os.walk(class_path):
            if "Site" in os.path.normpath(root).split(os.sep):
                continue
            for file in tqdm(files, desc=f"{os.path.basename(root)}", unit="img", leave=False):
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                file_path = os.path.join(root, file)
                if not is_image_valid(file_path):
                    print(f"Skipping unreadable image: {file_path}")
                    continue
                try:
                    raw_text = extract_text(file_path)
                    if not raw_text or len(raw_text.strip()) == 0:
                        continue
                    cleaned = preprocess_text(raw_text)
                    if cleaned:
                        data.append((file_path, cleaned, label))
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

    df = pd.DataFrame(data, columns=['path', 'text', 'label'])
    print(f"Saving {len(df)} entries to {cache_file}")
    save_text_labels_to_csv(data, cache_file)
    return df

def build_model():
    model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model(base_dir):
    df = build_dataset(base_dir)
    print(f"Collected {len(df)} valid images with text.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['label'])
    y_cat = to_categorical(y_encoded)
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['text']).toarray()
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    model = build_model()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[reduce_lr], batch_size=32)
    model.save("text_classifier_model.h5")

    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("text_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("Training complete and model saved.")
    plt.figure(figsize=(9, 5))
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Model accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('OCR_training_accuracy_plot.png')
    plt.show()
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"Final validation accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    base_dir = "path/to/folder"
    train_and_save_model(base_dir)

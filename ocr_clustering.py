import os
import shutil
import numpy as np
import cv2
import pytesseract
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import nltk
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")

stop_words = set(stopwords.words("english")).union(stopwords.words("french"))

def extract_text_from_image(path):
    try:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang="eng+fra")
        tokens = word_tokenize(text.lower())
        filtered = ' '.join([w for w in tokens if w.isalpha() and w not in stop_words])
        return filtered
    except Exception as e:
        print(f"Failed OCR on {path}: {e}")
        return ""

def collect_texts(folder):
    image_paths = []
    texts = []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, file)
                text = extract_text_from_image(path)
                if text.strip():
                    image_paths.append(path)
                    texts.append(text)
                else:
                    print(f"Skipped {path}: no readable text")

    return image_paths, texts

def cluster_ocr_texts(image_folder, n_clusters=5):
    image_paths, texts = collect_texts(image_folder)
    if not image_paths:
        print("No usable images with text.")
        return

    tfidf = TfidfVectorizer(max_features=500)
    X = tfidf.fit_transform(texts).toarray()

    if X.shape[1] > 50:
        X = PCA(n_components=50).fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    for i, path in enumerate(image_paths):
        cluster_dir = os.path.join(image_folder, f"cluster_{labels[i]}")
        os.makedirs(cluster_dir, exist_ok=True)
        try:
            shutil.move(path, os.path.join(cluster_dir, os.path.basename(path)))
        except Exception as e:
            print(f"Failed to move {path}: {e}")

    return [(f"cluster_{i}", sum(labels == i)) for i in range(n_clusters)]

if __name__ == "__main__":
    input_folder = "path/to/files"
    summary = cluster_ocr_texts(input_folder, n_clusters=5)
    print("OCR Text Cluster Summary:", summary)

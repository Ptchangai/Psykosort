# Psykosort

## Presentation

**Psykosort** is a modular and extensible toolset for automatically sorting large collections of image files. It uses OCR and CNN models as well as clustering algorithms to classify images based on visual content and embedded text. The project was created to organize chaotic image folders downloaded from social media (memes, maps, art, reaction pictures, educational material...).

---

## Project tools

Description of the scripts in this repository:

| Script | Description |
|--------|-------------|
| `build_ds.py` | Build training datasets by scanning image folders and extracting paths/labels for classification. |
| `train_CNN_model.py` | Train a Convolutional Neural Network to classify images into their matching subfolders based on visual appearance. |
| `train_OCR_model.py` | Train a text classification model (TF-IDF + Dense layers) using OCR-detected text from images. |
| `auto_sort_CNN.py` | Non-interactive inference pipeline to sort images to their predicted subfolder using a CNN model. |
| `psykosort.py` | Interactive GUI tool to review CNN and OCR predictions, and manually sort images into suggested folders using keyboard shortcuts. |

---

## Planned

- [ ] Add `.gif` / `.webp` format support
- [ ] Support multi-label classification.
- [ ] Improve OCR models.
- [ ] Export training logs and visualizations.
- [ ] Ad option to load custom model into GUI.
- [ ] Add option to train models directly from the GUI.

---

## Example

0. Starting from scratch: presort your unsorted folder:
Modify the root_folder in clustering.py. Update n_clusters to the number of folders you like. Then run:
   ```bash
   python3 clustering.py
   ```
Review the sorted folders manually.

1. Train a model on your sorted folders:
Modify the root_folder in your scripts, then run:
   ```bash
   python3 train_CNN_model.py
   python3 train_OCR_model.py
   ```

2. Launch the sorting GUI (**Psykosort**):
   ```bash
   python3 psykosort.py
   ```

3. Select a folder, review each image with model suggestions, and press `1`, `2`, `3` or `4` to move/skip.


---

## Authors

- Concept: `@Ptchangai`

Suggestions, feature requests, and pull requests are welcome! Would love to hear about other sorting strategies.
from build_ds import (
    collect_images_by_top_folder,
    build_classification_dataframe,
    SingleLabelImageGenerator
)

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pickle
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

def build_model(num_classes, img_size=(224, 224, 3)): 
    base_model = MobileNetV2(input_shape=img_size, include_top=False, weights='imagenet')
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.25)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


root_folder = "/media/piu/Elements/Images/Random/Argue/"
data = collect_images_by_top_folder(root_folder, ignore_folders={}) #ignore_folders={"Site", "Sites"})

df, lb = build_classification_dataframe(data)

with open("label_binarizer.pkl", "wb") as f:
    pickle.dump(lb, f)

df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
train_gen = SingleLabelImageGenerator(df_train, lb, img_size=(224, 224))
val_gen = SingleLabelImageGenerator(df_val, lb, img_size=(224, 224))

class_weights = compute_class_weight(class_weight='balanced', classes=lb.classes_, y=df_train['label'])
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

model = build_model(num_classes=len(lb.classes_))

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-7,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=60,
    class_weight=class_weight_dict,
    callbacks=[reduce_lr, early_stopping]
)

model.save("classifier.h5")

plt.figure(figsize=(9, 5))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ANN_training_accuracy_plot.png')
plt.show()

final_val_acc = history.history['val_accuracy'][-1]
print(f"Final validation accuracy: {final_val_acc:.4f}")

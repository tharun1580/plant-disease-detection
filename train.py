# ============================================================
# PLANT DISEASE DETECTION - train.py
# ============================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

# ============================================================
# SETTINGS
# ============================================================
DATASET_PATH = "dataset/plantvillage dataset/color"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================================================
# DATA GENERATORS
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("\n" + "="*50)
print("DATASET SUMMARY")
print("="*50)
print(f"Total Training Images   : {train_generator.samples}")
print(f"Total Validation Images : {val_generator.samples}")
print(f"Total Classes           : {train_generator.num_classes}")
print("="*50)

# ============================================================
# BUILD MODEL
# ============================================================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(38, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model built and compiled!")

# ============================================================
# CALLBACKS
# ============================================================
os.makedirs('model', exist_ok=True)

checkpoint = ModelCheckpoint(
    filepath='model/plant_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    restore_best_weights=True
)

# ============================================================
# TRAIN
# ============================================================
print("\n🚀 Starting Training...")
print("="*50)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

print("\n✅ Training Complete!")
print(f"Best Val Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

# ============================================================
# SAVE CLASS NAMES
# ============================================================
import json
class_indices = train_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}
with open('model/class_names.json', 'w') as f:
    json.dump(class_names, f)
print("✅ Class names saved to model/class_names.json")

# ============================================================
# PLOT GRAPHS
# ============================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('model/training_graphs.png')
print("✅ Graphs saved to model/training_graphs.png")
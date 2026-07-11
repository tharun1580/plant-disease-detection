# ============================================================
# PLANT DISEASE DETECTION - predict.py
# ============================================================

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import os

# ============================================================
# STEP 1: REBUILD MODEL ARCHITECTURE (same as train.py)
# ============================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
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

# ============================================================
# STEP 2: LOAD SAVED WEIGHTS INTO MODEL
# ============================================================

model.load_weights('model/plant_model.h5')
print("✅ Model weights loaded successfully!")

# ============================================================
# STEP 3: LOAD CLASS NAMES
# ============================================================

with open('model/class_names.json', 'r') as f:
    class_names = json.load(f)
print(f"✅ Loaded {len(class_names)} class names!")

# ============================================================
# STEP 4: PREDICTION FUNCTION
# ============================================================

def predict_disease(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None, None

    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100
    predicted_class = class_names[str(predicted_index)]

    return predicted_class, confidence

# ============================================================
# STEP 5: TEST WITH SAMPLE IMAGE
# ============================================================

if __name__ == "__main__":
    test_image_path = None
    dataset_path = "dataset/plantvillage dataset/color"

    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            if images:
                test_image_path = os.path.join(class_path, images[0])
                break

    if test_image_path:
        print(f"\n🌿 Testing with: {test_image_path}")
        print("="*50)
        predicted_class, confidence = predict_disease(test_image_path)
        if predicted_class:
            print(f"🔍 Predicted Disease : {predicted_class}")
            print(f"📊 Confidence        : {confidence:.2f}%")
            print("="*50)
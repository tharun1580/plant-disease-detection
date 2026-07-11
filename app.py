# ============================================================
# PLANT DISEASE DETECTION - app.py (FastAPI)
# ============================================================

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import shutil
import os

# ============================================================
# INITIALIZE APP
# ============================================================

app = FastAPI(
    title="🌿 Plant Disease Detection API",
    description="""
## Plant Disease Detection System

Upload a plant leaf image to detect diseases using AI.

### Features:
- Detects **38 different plant diseases** across 14 plant types
- Powered by **MobileNetV2** transfer learning
- Achieves **93.93% validation accuracy**
- Returns disease name, confidence, severity and treatment

### Supported Plants:
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
    """,
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# LOAD MODEL
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

model.load_weights('model/plant_model.h5')
print("✅ Model loaded!")

with open('model/class_names.json', 'r') as f:
    class_names = json.load(f)
print("✅ Class names loaded!")

# ============================================================
# DISEASE INFO DATABASE
# ============================================================

disease_info = {
    "Apple - Apple Scab": {
        "description": "A fungal disease causing dark, scabby lesions on leaves and fruit.",
        "treatment": "Apply fungicides early in spring. Remove and destroy infected leaves.",
        "severity": "Medium"
    },
    "Apple - Black Rot": {
        "description": "A fungal disease causing rotting of fruit and leaf spots.",
        "treatment": "Prune infected branches. Apply copper-based fungicides.",
        "severity": "High"
    },
    "Apple - Cedar Apple Rust": {
        "description": "A fungal disease causing orange spots on leaves.",
        "treatment": "Remove nearby juniper trees. Apply fungicides in spring.",
        "severity": "Medium"
    },
    "Apple - Healthy": {
        "description": "Your apple plant looks healthy! No disease detected.",
        "treatment": "Continue regular watering and fertilization.",
        "severity": "None"
    },
    "Blueberry - Healthy": {
        "description": "Your blueberry plant looks healthy! No disease detected.",
        "treatment": "Maintain proper soil pH (4.5-5.5) for best growth.",
        "severity": "None"
    },
    "Cherry (Including Sour) - Healthy": {
        "description": "Your cherry plant looks healthy! No disease detected.",
        "treatment": "Ensure proper drainage and regular pruning.",
        "severity": "None"
    },
    "Cherry (Including Sour) - Powdery Mildew": {
        "description": "A fungal disease causing white powdery coating on leaves.",
        "treatment": "Apply sulfur-based fungicides. Improve air circulation.",
        "severity": "Medium"
    },
    "Corn (Maize) - Cercospora Leaf Spot Gray Leaf Spot": {
        "description": "A fungal disease causing gray rectangular lesions on leaves.",
        "treatment": "Use resistant varieties. Apply fungicides when symptoms appear.",
        "severity": "High"
    },
    "Corn (Maize) - Common Rust": {
        "description": "A fungal disease causing rust-colored pustules on leaves.",
        "treatment": "Plant resistant hybrids. Apply fungicides at early stages.",
        "severity": "Medium"
    },
    "Corn (Maize) - Healthy": {
        "description": "Your corn plant looks healthy! No disease detected.",
        "treatment": "Ensure proper nitrogen fertilization and irrigation.",
        "severity": "None"
    },
    "Corn (Maize) - Northern Leaf Blight": {
        "description": "A fungal disease causing long grayish lesions on leaves.",
        "treatment": "Use resistant varieties. Apply fungicides preventively.",
        "severity": "High"
    },
    "Grape - Black Rot": {
        "description": "A fungal disease causing black shriveled fruit and leaf spots.",
        "treatment": "Remove mummified fruit. Apply fungicides from bud break.",
        "severity": "High"
    },
    "Grape - Esca (Black Measles)": {
        "description": "A fungal disease causing tiger-stripe pattern on leaves.",
        "treatment": "Prune infected wood. Apply fungicides. No complete cure exists.",
        "severity": "High"
    },
    "Grape - Healthy": {
        "description": "Your grape plant looks healthy! No disease detected.",
        "treatment": "Continue regular pruning and pest monitoring.",
        "severity": "None"
    },
    "Grape - Leaf Blight (Isariopsis Leaf Spot)": {
        "description": "A fungal disease causing dark brown spots on leaves.",
        "treatment": "Apply copper-based fungicides. Remove infected leaves.",
        "severity": "Medium"
    },
    "Orange - Haunglongbing (Citrus Greening)": {
        "description": "A bacterial disease causing yellowing and misshapen fruit. Very serious.",
        "treatment": "No cure available. Remove infected trees to prevent spread.",
        "severity": "High"
    },
    "Peach - Bacterial Spot": {
        "description": "A bacterial disease causing spots on leaves and fruit.",
        "treatment": "Apply copper sprays. Use resistant varieties.",
        "severity": "Medium"
    },
    "Peach - Healthy": {
        "description": "Your peach plant looks healthy! No disease detected.",
        "treatment": "Ensure proper thinning of fruit for better quality.",
        "severity": "None"
    },
    "Pepper, Bell - Bacterial Spot": {
        "description": "A bacterial disease causing water-soaked spots on leaves and fruit.",
        "treatment": "Apply copper bactericides. Avoid overhead irrigation.",
        "severity": "Medium"
    },
    "Pepper, Bell - Healthy": {
        "description": "Your pepper plant looks healthy! No disease detected.",
        "treatment": "Maintain consistent watering and avoid waterlogging.",
        "severity": "None"
    },
    "Potato - Early Blight": {
        "description": "A fungal disease causing dark spots with concentric rings on leaves.",
        "treatment": "Apply fungicides. Remove infected plant debris after harvest.",
        "severity": "Medium"
    },
    "Potato - Healthy": {
        "description": "Your potato plant looks healthy! No disease detected.",
        "treatment": "Rotate crops yearly to prevent disease buildup.",
        "severity": "None"
    },
    "Potato - Late Blight": {
        "description": "A serious disease that caused the Irish Potato Famine. Destroys crops rapidly.",
        "treatment": "Apply fungicides immediately. Destroy infected plants urgently.",
        "severity": "High"
    },
    "Raspberry - Healthy": {
        "description": "Your raspberry plant looks healthy! No disease detected.",
        "treatment": "Prune old canes after fruiting for better yield.",
        "severity": "None"
    },
    "Soybean - Healthy": {
        "description": "Your soybean plant looks healthy! No disease detected.",
        "treatment": "Monitor for pests and maintain proper soil nutrition.",
        "severity": "None"
    },
    "Squash - Powdery Mildew": {
        "description": "A fungal disease causing white powdery patches on leaves.",
        "treatment": "Apply potassium bicarbonate or sulfur sprays.",
        "severity": "Medium"
    },
    "Strawberry - Healthy": {
        "description": "Your strawberry plant looks healthy! No disease detected.",
        "treatment": "Mulch around plants to retain moisture and prevent disease.",
        "severity": "None"
    },
    "Strawberry - Leaf Scorch": {
        "description": "A fungal disease causing purple spots and browning of leaf edges.",
        "treatment": "Remove infected leaves. Apply fungicides in early spring.",
        "severity": "Medium"
    },
    "Tomato - Bacterial Spot": {
        "description": "A bacterial disease causing water-soaked spots on leaves and fruit.",
        "treatment": "Apply copper bactericides. Use disease-free seeds.",
        "severity": "Medium"
    },
    "Tomato - Early Blight": {
        "description": "A fungal disease causing dark spots with target-like rings on lower leaves.",
        "treatment": "Apply fungicides. Remove lower infected leaves.",
        "severity": "Medium"
    },
    "Tomato - Healthy": {
        "description": "Your tomato plant looks healthy! No disease detected.",
        "treatment": "Stake plants properly and ensure good air circulation.",
        "severity": "None"
    },
    "Tomato - Late Blight": {
        "description": "A serious fungal disease causing dark water-soaked lesions. Spreads rapidly.",
        "treatment": "Apply fungicides immediately. Remove infected plants.",
        "severity": "High"
    },
    "Tomato - Leaf Mold": {
        "description": "A fungal disease causing yellow spots on upper leaf surface.",
        "treatment": "Improve ventilation. Apply fungicides. Avoid leaf wetness.",
        "severity": "Medium"
    },
    "Tomato - Septoria Leaf Spot": {
        "description": "A fungal disease causing small circular spots with dark borders.",
        "treatment": "Apply fungicides. Remove infected leaves promptly.",
        "severity": "Medium"
    },
    "Tomato - Spider Mites Two-Spotted Spider Mite": {
        "description": "Tiny mites causing yellowing and stippling of leaves.",
        "treatment": "Apply miticides or neem oil. Increase humidity around plants.",
        "severity": "Medium"
    },
    "Tomato - Target Spot": {
        "description": "A fungal disease causing circular spots with concentric rings.",
        "treatment": "Apply fungicides. Remove plant debris after harvest.",
        "severity": "Medium"
    },
    "Tomato - Tomato Mosaic Virus": {
        "description": "A viral disease causing mosaic pattern and distortion of leaves.",
        "treatment": "No cure. Remove infected plants. Control aphid vectors.",
        "severity": "High"
    },
    "Tomato - Tomato Yellow Leaf Curl Virus": {
        "description": "A viral disease causing yellowing and upward curling of leaves.",
        "treatment": "No cure. Control whitefly vectors. Use resistant varieties.",
        "severity": "High"
    }
}

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(preds[0])
    confidence = float(preds[0][predicted_index] * 100)
    predicted_class = class_names[str(predicted_index)]

    parts = predicted_class.split('___')
    if len(parts) == 2:
        plant = parts[0].replace('_', ' ')
        disease = parts[1].replace('_', ' ').title()
        display_name = f"{plant} - {disease}"
    else:
        display_name = predicted_class.replace('_', ' ')

    return display_name, confidence

# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": "❌ Invalid file type. Please upload a JPG or PNG image."
            }
        )

    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": "❌ File too large. Please upload an image under 10MB."
            }
        )

    # Reset file pointer
    await file.seek(0)

    # Save uploaded file
    upload_path = f"static/uploads/{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    disease, confidence = predict_disease(upload_path)

    # Confidence threshold check
    if confidence < 70.0:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": "⚠️ This doesn't look like a plant leaf image. Please upload a clear photo of a plant leaf for accurate detection."
            }
        )

    # Determine status
    if "healthy" in disease.lower():
        status = "healthy"
        emoji = "✅"
    else:
        status = "diseased"
        emoji = "⚠️"

    # Get disease info
    info = disease_info.get(disease, {
        "description": "No description available.",
        "treatment": "Consult an agricultural expert.",
        "severity": "Unknown"
    })

    # Severity colors
    severity_colors = {
        "None": "#4ade80",
        "Low": "#facc15",
        "Medium": "#fb923c",
        "High": "#f87171",
        "Unknown": "#94a3b8"
    }
    severity_color = severity_colors.get(info["severity"], "#94a3b8")

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "disease": disease,
            "confidence": f"{confidence:.2f}",
            "image_path": f"/static/uploads/{file.filename}",
            "status": status,
            "emoji": emoji,
            "description": info["description"],
            "treatment": info["treatment"],
            "severity": info["severity"],
            "severity_color": severity_color
        }
    )

# ============================================================
# REST API ENDPOINT
# ============================================================

@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    """
    REST API endpoint for plant disease detection.
    Returns JSON response with disease info.
    """

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid file type. Please upload JPG or PNG image."
            }
        )

    # Save uploaded file
    upload_path = f"static/uploads/api_{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    disease, confidence = predict_disease(upload_path)

    # Confidence threshold check
    if confidence < 70.0:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Image does not appear to be a plant leaf. Please upload a clear plant leaf photo."
            }
        )

    # Get disease info
    info = disease_info.get(disease, {
        "description": "No description available.",
        "treatment": "Consult an agricultural expert.",
        "severity": "Unknown"
    })

    # Determine status
    status = "healthy" if "healthy" in disease.lower() else "diseased"

    return JSONResponse(content={
        "disease": disease,
        "confidence": round(confidence, 2),
        "status": status,
        "severity": info["severity"],
        "description": info["description"],
        "treatment": info["treatment"]
    })
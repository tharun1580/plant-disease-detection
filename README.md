# 🌿 Plant Disease Detection System

A deep learning web application that detects plant diseases from leaf images using MobileNetV2 transfer learning.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Accuracy](https://img.shields.io/badge/Accuracy-93.93%25-brightgreen)

---

## 📋 Project Overview

This system helps farmers and agricultural experts identify plant diseases quickly by simply uploading a photo of a plant leaf. The AI model analyzes the image and provides:

- ✅ Disease name with confidence percentage
- ⚡ Severity level (None / Medium / High)
- 📋 Disease description
- 💊 Treatment recommendations

---

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| Model | MobileNetV2 (Transfer Learning) |
| Dataset | PlantVillage (43,456 training images) |
| Classes | 38 plant disease classes |
| Validation Accuracy | **93.93%** |
| Training Epochs | 10 |

---

## 🌱 Supported Plants & Diseases

| Plant | Conditions Detected |
|-------|-------------------|
| 🍎 Apple | Scab, Black Rot, Cedar Rust, Healthy |
| 🌽 Corn | Cercospora, Common Rust, Blight, Healthy |
| 🍇 Grape | Black Rot, Esca, Leaf Blight, Healthy |
| 🍅 Tomato | 9 diseases + Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Pepper | Bacterial Spot, Healthy |
| + More | Blueberry, Cherry, Orange, Raspberry, Soybean, Squash, Strawberry |

---

## 🛠️ Tech Stack

- **Model**: MobileNetV2 (Pre-trained on ImageNet)
- **Framework**: TensorFlow 2.21, Keras
- **Backend**: FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **Language**: Python 3.13

---

## 📁 Project Structure

```
plant-disease-detection/
│
├── app.py              # FastAPI backend with REST API
├── train.py            # Model training script
├── predict.py          # Prediction script
├── requirements.txt    # Project dependencies
│
├── model/
│   ├── plant_model.h5      # Trained model weights
│   ├── class_names.json    # 38 class labels
│   └── training_graphs.png # Accuracy/loss graphs
│
└── templates/
    └── index.html      # Web app frontend
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/tharun1580/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset & train model
```bash
# Download PlantVillage dataset from Kaggle
# Place in dataset/plantvillage dataset/color/
python train.py
```

### 5. Run the application
```bash
uvicorn app:app --reload
```

### 6. Open in browser
http://127.0.0.1:8000

---

## 🔌 API Usage

### Web Interface
GET  /          → Main web app
POST /predict   → Upload image via web form
### REST API
POST /api/predict  → Returns JSON response
GET  /docs         → Interactive API documentation
### Example API Response
```json
{
  "disease": "Tomato - Early Blight",
  "confidence": 94.32,
  "status": "diseased",
  "severity": "Medium",
  "description": "A fungal disease causing dark spots...",
  "treatment": "Apply fungicides. Remove lower infected leaves."
}
```

---

## 📊 Training Results

| Epoch | Train Accuracy | Val Accuracy |
|-------|---------------|-------------|
| 1 | 78.58% | 90.72% |
| 5 | 90.38% | 92.96% |
| 9 | 91.34% | **93.93%** ✅ |

---

## 👨‍💻 Author

**Tharun** — B.Tech AI & ML Student at Vishnu Institute of Technology

---

## 📄 License

This project is for educational purposes.
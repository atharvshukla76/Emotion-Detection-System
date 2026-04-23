# 🌊 Moodwave — Emotion Detection System

> Real-time voice-based emotion detection powered by CNN, Mel Spectrogram & MFCC

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-UI%20Preview-99D775?style=for-the-badge&logo=github)](https://atharvshukla76.github.io/Emotion-Detection-System/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)

---

## 🎯 What It Does

Moodwave records **3–5 seconds of your voice**, extracts audio features, and classifies your **emotional state in real time** into 6 categories:

| Emotion | Color |
|---------|-------|
| 😊 Happy | Gold `#FFD700` |
| 😠 Angry | Red `#FF3366` |
| 😢 Sad | Blue `#6699FF` |
| 😨 Fear | Violet `#CC44FF` |
| 😐 Neutral | Cyan `#00C8FF` |
| 🤢 Disgust | Lime `#44FF88` |

---

## 🧠 Model Architecture

- **Input**: Raw audio → Mel Spectrogram (96 bands) + MFCC (40 coefficients) → `(150, 136, 1)`
- **Architecture**: Multi-layer CNN + Conv1D + GlobalAveragePooling + Dense
- **Training**: CREMA-D dataset, 6-class classification
- **Tech Stack**: TensorFlow/Keras, librosa, scikit-learn

---

## 🚀 Run Locally (Full AI Mode)

### 1. Clone the repo
```bash
git clone https://github.com/atharvshukla76/Emotion-Detection-System.git
cd "Emotion-Detection-System"
```

### 2. Create and activate virtual environment
```bash
python -m venv tf_env
tf_env\Scripts\activate      # Windows
# source tf_env/bin/activate # Mac/Linux
```

### 3. Install dependencies
```bash
pip install fastapi uvicorn tensorflow librosa scikit-learn pydantic python-multipart
```

### 4. Start the server
```bash
uvicorn api:app --port 8000
```

### 5. Open in browser
```
http://localhost:8000
```

---

## 📁 Project Structure

```
Emotion-Detection-System/
├── api.py                # FastAPI backend — serves UI + /predict endpoint
├── moodwave.html         # Frontend UI (served by FastAPI)
├── saved_model/
│   ├── emotion_model.keras
│   ├── encoder.pkl
│   └── norm.pkl
├── AudioWAV/             # Training dataset (not included)
├── main.ipynb            # Model training notebook
└── docs/
    └── index.html        # GitHub Pages demo (no backend)
```

---

## 🌐 GitHub Pages Demo

The live demo at [atharvshukla76.github.io/Emotion-Detection-System](https://atharvshukla76.github.io/Emotion-Detection-System/) is a **static preview** of the UI with simulated emotion results. For real AI-powered predictions, run the project locally using the instructions above.

---

## 📊 Design

| Property | Value |
|---|---|
| **Aesthetic** | Dark Cyberpunk / Neural Interface |
| **Primary** | `#99D775` Neon Green |
| **Secondary** | `#6353CA` Violet |
| **Background** | `#050F1E` Deep Navy |
| **Fonts** | Orbitron + Share Tech Mono |

---

Made with ❤️ by [Atharv Shukla](https://github.com/atharvshukla76)

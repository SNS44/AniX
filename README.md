# 🦁 AniX — Wildlife Identification & Biodiversity Intelligence Platform

<p align="center">
  <b><i>Identify wildlife species. Explore taxonomical profiles. Discover habitats — powered by Deep Learning.</i></b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-4CAF50?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Architecture-ConvNeXt--Tiny-2E7D32?style=for-the-badge" alt="Architecture">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge" alt="Framework">
</p>

<div align="center">

[![Python](https://img.shields.io/badge/Language-Python%203.9+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#)
[![Timm](https://img.shields.io/badge/Model%20Hub-timm-green?style=flat-square)](#)
[![Pandas](https://img.shields.io/badge/Data-Pandas%20%26%20NumPy-150458?style=flat-square&logo=pandas&logoColor=white)](#)

</div>

---

## 🌿 The Vision

**AniX** is a premium, high-performance desktop-web application designed to identify animal species and display deep ecological profiles. By combining modern computer vision with an interactive reference library, AniX bridges the gap between field observation and biological knowledge.

Built with **PyTorch** and **timm (Torch Image Models)**, the platform executes instant image classification across **209 distinct classes**, complete with safety thresholds to handle unclassified inputs. All of this is wrapped in a premium dark-green glassmorphic interface powered by **Streamlit** and styled with custom CSS and clean inline SVG icons.

---

## 🚀 Features

* 📷 **AI Image Identification** — Upload an image (JPEG, PNG, WebP, BMP) and get immediate predictions of the species, along with its scientific name, confidence score, and top-5 alternatives.
* 📖 **Species Encyclopedia** — Access a comprehensive digital catalog of 209 species with full taxonomical details, biological profile parameters, conservation status, and facts.
* 🛡️ **Safety Thresholds & Margin Validation** — If the model's confidence is below `80%` or if the top-2 predictions are too close (margin `< 20%`), the app safely labels the sample as **"Unclassified"** to prevent false positives.
* 🔍 **Smart Taxonomy Filters & Search** — Filter the encyclopedia by IUCN conservation status (e.g. *Critically Endangered*, *Least Concern*) and search in real-time.
* 💎 **Premium Dark Green Glassmorphic UI** — Hand-crafted interface styled with custom CSS variables, custom typography, and high-fidelity inline SVG icons (eliminating raw emojis for a professional finish).

---

## 🧠 How It Works (Core Logic)

### 1. Image Preprocessing & Validation
* Before feeding an image to the model, `utils/validators.py` ensures the image:
  - Matches permitted formats (JPEG, PNG, WebP, BMP).
  - Fits within the **8000x8000px** dimensional limits.
* The image is then loaded via **PIL**, converted to RGB, resized to `(224, 224)` and transformed into a normalized PyTorch tensor (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### 2. ConvNeXt-Tiny Inference
* The model utilizes a **ConvNeXt-Tiny** architecture instantiated via `timm` and trained on wildlife classes.
* Weights are loaded from `models/best_model.pth`.
* In `predict.py`, the model computes a forward pass:
  ```python
  output = model(x)
  probabilities = F.softmax(output, dim=1)
  ```

### 3. Classification Decision Engine
To safeguard against misclassifications, a double-threshold filter is applied:
* **Confidence Limit**: The top-1 prediction confidence must be $\ge 80\%$.
* **Margin Separation**: The difference between the top-1 and top-2 predictions must be $\ge 20\%$.
* If either constraint is violated, the prediction is designated as **"Unclassified"**.

---

## 🖥️ System Requirements

| Requirement | Details |
| ----------- | ------- |
| **OS** | Windows 10/11 / macOS / Linux |
| **Python** | 3.9 – 3.11 (3.10 recommended) |
| **Hardware** | Standard CPU (GPU support enabled dynamically if CUDA is available) |

---

## 📦 Tech Stack & Dependencies

* **Python** — Core language
* **PyTorch** — Deep learning backend
* **timm** — Pre-trained ConvNeXt architecture hub
* **Streamlit** — Web UI framework
* **Pandas** — Biological profiles lookup
* **Pillow (PIL)** — Image verification and formatting
* **Lucide SVGs** — Vector icons rendered inline for a sleek layout

```text
streamlit==1.58.0
torch==2.12.1+cu126
torchvision==0.27.1+cu126
timm==1.0.27
pandas==2.2.2
numpy==2.2.6
Pillow==12.2.0
python-dotenv==1.0.1
```

---

## ▶️ Setup & Execution

### 1. Clone the Project
```bash
git clone https://github.com/your-username/AniX.git
cd AniX
```

### 2. Activate Environment
```bash
conda activate ecov
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## ⚙️ Configuration Parameters

Adjust the runtime behavior directly in `config/settings.py`:

```python
CONFIDENCE_THRESHOLD = 0.80      # Min confidence for positive ID
MARGIN_THRESHOLD = 0.20          # Min margin separation between top-1 and top-2
IMG_SIZE = (224, 224)            # Resolution for model input
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # Max upload size (10 MB)
```

---

## 🏗️ System Architecture

### High-Level Architecture Flow

```
┌────────────────────────────────────────────┐
│              Streamlit UI Layer             │
│        (Sidebar Navigation + Custom CSS)   │
└──────┬──────────────────────────────┬──────┘
       │                              │
       ▼                              ▼
┌──────────────┐              ┌──────────────┐
│  Identify    │              │ Encyclopedia │
│  Page        │              │  Page        │
└──────┬───────┘              └──────┬───────┘
       │                              │
       ▼                              ▼
┌──────────────┐              ┌──────────────┐
│ validators.py│              │ data_loader  │
│ (Size, Type) │              │  reads CSV   │
└──────┬───────┘              └──────┬───────┘
       │                              │
       ▼                              ▼
┌──────────────┐              ┌──────────────┐
│  predict.py  │              │ Search &     │
│ (PyTorch +   │              │ filter by    │
│  timm model) │              │ IUCN Status  │
└──────┬───────┘              └──────┬───────┘
       │                              │
       ▼                              ▼
┌──────────────┐              ┌──────────────┐
│ Thresholds   │              │ Details      │
│ (Conf/Margin)│              │ Modal Popup  │
└──────┬───────┘              └──────────────┘
       │
       ├─▶ Below threshold ──▶ [Unclassified]
       │
       └─▶ Above threshold ──▶ [Display Species + Bio Info]
```

### Architectural Design Choices
* **Single-Inference Preprocessing**: Normalization and resizing are performed on the fly using a standard PyTorch `transforms.Compose` pipeline before running ConvNeXt-Tiny inference.
* **Cached Data Operations**: Pandas biological profiles and lookup databases are cached via `@st.cache_data` and `@st.cache_resource` for immediate profile retrieval without disk overhead.
* **Inline SVG Integration**: Pure vector Lucide SVG definitions are injected inline to render smooth UI icons, eliminating the performance bottleneck of custom font packages.

### Directory Structure

```
AniX/
├── .streamlit/
│   └── config.toml           # Streamlit theme setup (dark base)
├── config/
│   ├── __init__.py           
│   └── settings.py           # Configuration parameters and thresholds
├── data/
│   └── animal_info.csv       # 209 Species biological & habitat dataset
├── dataset_final/            # Target folders to determine class names
├── models/
│   ├── best_model.pth        # ConvNeXt-Tiny weights checkpoint
│   └── model_metadata.json   # Model architecture statistics metadata
├── utils/
│   ├── __init__.py           
│   ├── data_loader.py        # Loading routines for dataset profiles
│   └── validators.py         # Image size, extension, and corruption checks
├── app.py                    # Main app runner UI layout
├── predict.py                # Deep learning prediction module
├── style.css                 # Custom glassmorphic styling
└── requirements.txt          # Library dependencies
```

---

## 🔁 Processing Flow

```
   [ User uploads image ]
              │
              ▼
    [ Image Validator ] ─── (Invalid) ──▶ [ Show Validation Error ]
              │
           (Valid)
              ▼
   [ Preprocess to 224x224 ]
              │
              ▼
    [ ConvNeXt-Tiny Inference ]
              │
              ▼
  [ Extract Top-1 and Top-2 Confidences ]
              │
              ├─── (Conf < 80% OR Margin < 20%) ──▶ [ Display Unclassified ]
              │
              └─── (Passes Threshold Check)  ───▶ [ Show Predicted Species ]
                                                           │
                                                           ▼
                                                [ Lookup animal_info.csv ]
                                                           │
                                                           ▼
                                                [ Display Bio Profile Card ]
```

---

<p align="center">
  <b>Built with 🌿 for Conservation and Species Identification</b>
</p>

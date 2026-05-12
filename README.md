# 🔍 Verifai — AI-Generated Image Detection System

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/Shabeeb-Mohammed-Sakir/verifai)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.26%25-brightgreen)](https://huggingface.co/spaces/Shabeeb-Mohammed-Sakir/verifai)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Detect AI-generated and manipulated images using EfficientNet-B0 + FFT frequency analysis with Grad-CAM explainability and EXIF metadata forensics.

---

## 🌐 Live Demo

**Try it now:** [https://huggingface.co/spaces/Shabeeb-Mohammed-Sakir/verifai](https://huggingface.co/spaces/Shabeeb-Mohammed-Sakir/verifai)

Works on any device — desktop, tablet, and mobile.

---

## 🎯 What Verifai Detects

| Generator | Detection |
|-----------|-----------|
| DALL-E (ChatGPT) | ✅ |
| Midjourney | ✅ |
| Stable Diffusion | ✅ |
| NightCafe | ✅ |
| GAN-generated faces | ✅ |
| Disco Diffusion | ✅ |

---

## 🏗️ Architecture

Verifai uses a **dual-branch fusion model** that combines:

### Branch 1 — Spatial Feature Extractor
- EfficientNet-B0 pretrained on ImageNet
- Fine-tuned on 205,561 images
- Detects pixel-level artifacts, texture anomalies, and structural inconsistencies

### Branch 2 — FFT Feature Extractor
- 2D Fast Fourier Transform analysis
- Detects periodic frequency artifacts left by GAN upsampling
- Mathematically grounded in signal processing theory

### Fusion Classifier
- Concatenates spatial (256-dim) + frequency (256-dim) features
- 3-layer MLP with dropout regularisation
- Binary output: REAL or AI GENERATED
Input Image (224×224)
│
┌───┴───┐
│       │
EfficientNet  FFT Analysis
│       │
256d    256d
│       │
└───┬───┘
512d
Classifier
│
REAL / FAKE
---

## 📊 Training Results

| Metric | Value |
|--------|-------|
| Final Accuracy | **99.26%** |
| Training Images | 205,561 |
| Validation Images | 60,982 |
| Image Resolution | 224 × 224 |
| Parameters | 30,305,342 |
| Best Epoch | 9/10 |

### Training Datasets
| Dataset | Images | Source |
|---------|--------|--------|
| CIFAKE | 120,000 | CIFAR-10 + Stable Diffusion |
| 140k Real & Fake Faces | 140,000 | Flickr + StyleGAN |
| AI vs Non-AI Generated | 6,553 | Midjourney, DALL-E, SD, NightCafe |

---

## ✨ Features

- **Confidence Meter** — Speedometer-style circular gauge showing prediction confidence
- **Grad-CAM Heatmap** — Visual explanation of which image regions influenced the decision
- **EXIF Metadata Analysis** — Forensic analysis of image metadata for camera signatures
- **Conflict Detection** — Flags when model prediction and metadata disagree
- **Sci-fi UI** — Custom dark blue interface with real-time scan animations

---

## 🗂️ Project Structure
visual-misinformation-detector/
├── app.py                  # Gradio web application
├── requirements.txt        # Dependencies
├── src/
│   ├── model.py           # VerifaiDetector architecture
│   ├── gradcam.py         # Grad-CAM implementation
│   ├── exif_analyzer.py   # EXIF metadata forensics
│   └── dataset.py         # Dataset classes and transforms
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA + FFT visualisation
│   └── 02_gradcam_visualization.ipynb
├── outputs/
│   ├── sample_images.png
│   ├── fft/fft_comparison.png
│   └── gradcam/gradcam_results.png
└── models/                # Model weights (not tracked in git)
---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/Shabeeb-Mohammed-Sakir-P/visual-misinformation-detector
cd visual-misinformation-detector

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Run the app
python app.py
```

---

## 🔬 Technical Deep Dive

### Why FFT Analysis?
AI image generators (GANs, Diffusion models) produce images through iterative upsampling. This process leaves **periodic artifacts** in the frequency domain that are invisible to the human eye but statistically detectable via Fast Fourier Transform.

Real photographs have organic, irregular frequency distributions. AI-generated images show characteristic grid-like patterns in their FFT spectrum — these are the fingerprints Verifai detects.

### Why EfficientNet-B0?
EfficientNet-B0 provides the best accuracy-to-parameter ratio among CNN architectures. With compound scaling, it captures fine-grained texture artifacts at 224×224 resolution that are invisible at smaller resolutions (such as 32×32 used in baseline experiments).

### Grad-CAM Explainability
Gradient-weighted Class Activation Mapping (Grad-CAM) hooks into the last convolutional layer of EfficientNet-B0, computing gradient flows to produce spatial heatmaps. This shows which regions of the image triggered the REAL/FAKE decision — making Verifai interpretable and trustworthy.

---

## ⚠️ Limitations

- **Smartphone photos**: Modern smartphones apply AI-based computational photography (HDR, AI scene enhancement, noise reduction) which creates frequency artifacts similar to AI generation. This may cause false positives on heavily processed smartphone photos.
- **Cross-domain generalisation**: Performance may vary on image generators not represented in the training data.
- **Image editing**: Minor edits (cropping, resizing) do not affect detection. Heavy post-processing may.

---

## 👨‍💻 Author

**Shabeeb Mohammed Sakir P**
MSc Applied Data Science, Kerala,India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-shabeeb--mohammed--sakir-blue)](https://linkedin.com/in/shabeeb-mohammed-sakir)
[![GitHub](https://img.shields.io/badge/GitHub-Shabeeb--Mohammed--Sakir--P-black)](https://github.com/Shabeeb-Mohammed-Sakir-P)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

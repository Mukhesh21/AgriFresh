# AgriFresh: Deep Learning Produce Quality Assessment & 5-Model Benchmark

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit 1.28+](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official Repository for Paper:**  
> *"AgriFresh: A Multi-Architecture Deep Learning Benchmark for Automated Agricultural Produce Quality Assessment with Grad-CAM Explainability, Zero-Shot OOD Gatekeeping, and Shelf-Life Forecasting"*  
> **Authors:** Dr. A. Shenbagarajan, Hariish G S, Mukhesh P, Rockland Rowan M (Dept. of Artificial Intelligence and Data Science, Mepco Schlenk Engineering College, Sivakasi, Tamil Nadu, India).

---

## 📌 Repository Overview

**AgriFresh** is an end-to-end deep learning framework and deployment application designed for non-destructive, automated quality grading (*Fresh*, *Semi-Fresh*, *Rotten*) across 8 agricultural crop varieties (**Banana, Bittermelon, Cucumber, Eggplant, Orange, Papaya, Pineapple, and Tomato**).

Evaluating 14,160 standardized specimens from the **AgriFreshNET** dataset, AgriFresh establishes a standardized benchmark across 5 CNN architectures, integrates Grad-CAM visual explainability, deploys a zero-shot safety filter for Out-of-Distribution (OOD) inputs, and estimates continuous post-harvest shelf life in remaining days.

---

## 🌟 Key Features

1. **5-Architecture Standardized Benchmark:**
   - **DenseNet121:** Highest classification performance (**98.02% Test Accuracy**, **99.86% Macro ROC-AUC**).
   - **EfficientNetV2-S:** Runner-up accuracy (**96.75%**) with fast execution (3.42 ms).
   - **ResNet18:** Optimal Pareto trade-off (**96.00% Accuracy**, **1.07 ms Latency**).
   - **EfficientNet-B0:** Compact compound-scaled network (15.46 MB, 95.91% Accuracy).
   - **MobileNetV3-Small:** Ultra-fast edge throughput (**0.80 ms/image**, 5.85 MB footprint).

2. **🔍 Grad-CAM Explainability (XAI):**
   - Gradient-weighted Class Activation Mapping localizes spatial feature attention on produce skins, confirming models focus on biological decay (mold, lesions, browning) rather than background noise.

3. **🛡️ Zero-Shot Out-of-Distribution (OOD) Gatekeeper:**
   - Pre-trained ResNet50 semantic filter evaluating 70+ category keywords to intercept non-produce or invalid inputs (books, notebooks, electronics, chocolates) prior to classification.

4. **⏳ Freshness Index & Shelf-Life Forecasting Engine:**
   - Converts multi-class probabilities into a continuous **Freshness Index Score (0–100%)** using:
     $$\text{Freshness Index} = \left( P_{\text{Fresh}} \times 1.0 + P_{\text{Semi-Fresh}} \times 0.5 + P_{\text{Rotten}} \times 0.0 \right) \times 100$$
   - Maps scores to crop-specific post-harvest storage intervals (1–15 days remaining) and provides USDA/FAO temperature and handling guidelines.

5. **📸 Edge Web Application & High-Throughput Batch Manifest:**
   - Supports **Single Image Upload**, **Live Camera Edge Snapshots**, and **Multi-File Batch Processing** with downloadable CSV inspection manifests (`agrifresh_batch_inspection_manifest.csv`).

---

## 📊 Experimental Benchmark Results (2,125 Held-Out Test Specimens)

| Model Architecture | Accuracy (%) | Macro ROC-AUC (%) | Precision (%) | Recall (%) | Macro F1 (%) | Parameters | Size (MB) | GPU Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DenseNet121** | **98.02%** | **99.86%** | **98.02%** | **98.03%** | **98.02%** | 6,956,931 | 26.86 MB | 3.72 ms |
| **EfficientNetV2-S** | 96.75% | 99.76% | 96.76% | 96.76% | 96.75% | 20,181,331 | 77.57 MB | 3.42 ms |
| **ResNet18** | 96.00% | 99.68% | 96.03% | 96.00% | 96.01% | 11,178,051 | 42.68 MB | 1.07 ms |
| **EfficientNet-B0** | 95.91% | 99.54% | 95.91% | 95.91% | 95.91% | 4,011,391 | 15.46 MB | 1.66 ms |
| **MobileNetV3-Small** | 89.32% | 97.64% | 89.49% | 89.32% | 89.37% | **1,520,931** | **5.85 MB** | **0.80 ms** |

---

## 📁 Repository Directory Structure

```
AgriFresh/
├── dataset/                    # Stratified Train/Val/Test CSV manifests (Seed 42)
├── models/                     # PyTorch model checkpoints (.pth)
│   ├── densenet121/
│   ├── efficientnet_v2_s/
│   ├── resnet18/
│   ├── efficientnet_b0/
│   └── mobilenetv3_small/
├── src/                        # Modular Python source packages
│   ├── config.py               # Hyperparameters, directories, & settings
│   ├── dataset_analysis.py     # Dataset audit & MD5 deduplication engine
│   ├── preprocessing.py        # ImageNet normalization & Cutout augmentations
│   ├── dataset.py              # PyTorch Dataset & DataLoader implementation
│   ├── models.py               # 5 PyTorch transfer learning model builders
│   ├── train.py                # 2-stage transfer learning training loop
│   ├── evaluate.py             # Evaluation & metric aggregation engine
│   ├── explainability.py       # Grad-CAM heatmap generation module
│   ├── ood_gatekeeper.py       # ResNet50 zero-shot OOD safety filter
│   ├── shelf_life.py           # Freshness Index & shelf-life calculation engine
│   ├── batch_inference.py      # Multi-specimen batch quality inspection engine
│   └── inference.py            # High-level inference API
├── results/                    # Generated metrics, CSVs, heatmaps, & ROC curves
│   ├── model_comparison.csv
│   ├── confusion_matrices/
│   ├── training_history/
│   └── visualizations/
├── dashboard/
│   └── app.py                  # Interactive Streamlit web application
├── reports/                    # Generated project statistics & reports
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start & Setup Guide

### 1. Clone & Environment Setup
```bash
git clone https://github.com/your-username/AgriFresh.git
cd AgriFresh
pip install -r requirements.txt
```

### 2. Dataset Inspection & Audit
```bash
python src/dataset_analysis.py
```

### 3. Model Training (2-Stage Transfer Learning)
```bash
# Train all 5 benchmark models
python src/train.py --model all --epochs 10 --batch_size 64

# Train a specific architecture
python src/train.py --model densenet121 --epochs 10
```

### 4. Evaluate Benchmark Performance
```bash
python src/evaluate.py
```

### 5. Launch Interactive Streamlit Web Application
```bash
streamlit run dashboard/app.py
```

---

## 🤝 Citation & Acknowledgments

If you find AgriFresh useful in your research or applications, please cite:

```bibtex
@article{AgriFresh2026,
  title={AgriFresh: A Multi-Architecture Deep Learning Benchmark for Automated Agricultural Produce Quality Assessment with Grad-CAM Explainability, Zero-Shot OOD Gatekeeping, and Shelf-Life Forecasting},
  author={Dr. A. Shenbagarajan and Hariish G S and Mukhesh P and Rockland Rowan M},
  journal={Department of Artificial Intelligence and Data Science, Mepco Schlenk Engineering College},
  year={2026}
}
```

*Special thanks to Dr. A. Shenbagarajan, Associate Professor, Dept. of AI & DS, Mepco Schlenk Engineering College (Autonomous), Sivakasi, for guidance throughout the development of AgriFresh.*

<div align="center">

# 🦐 Smart Aquaculture
## Automatic Vannamei Shrimp Length Measurement Using Camera and Artificial Intelligence

<img src="images/banner.png" width="900">

**Official source code and learning materials accompanying the book**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)]
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]
[![YOLO](https://img.shields.io/badge/YOLO-11-orange.svg)]
[![License](https://img.shields.io/badge/License-MIT-success.svg)]

</div>

---

# 📖 About

This repository contains all source code, examples, datasets, and implementation materials for the book:

> **Smart Aquaculture: Automatic Vannamei Shrimp Length Measurement Using Camera and Artificial Intelligence**

The repository demonstrates how **Computer Vision**, **Deep Learning**, and **Image Processing** can be integrated to automatically estimate the body length of Vannamei shrimp from underwater images.

---

# 🎯 Project Workflow

```text
Camera
   │
   ▼
YOLO11 Object Detection
   │
   ▼
Crop ROI
   │
   ▼
Grayscale
   │
   ▼
Otsu Threshold
   │
   ▼
Morphology
   │
   ▼
Skeletonization
   │
   ▼
Graph Construction
   │
   ▼
Dijkstra Algorithm
   │
   ▼
Length Measurement
```

---

# 📸 Project Preview

## System Overview

> **(Place System Architecture Image Here)**

```
images/system_architecture.png
```

---

## Image Processing Pipeline

> **(Place Processing Pipeline Image Here)**

```
images/pipeline.png
```

---

## GUI

> **(Place GUI Screenshot Here)**

```
images/gui.png
```

---

# 📂 Repository Structure

```
Smart-Aquaculture/
│
├── 📁 Computer Vision untuk Pengukuran Udang
│
├── 📁 Deep Learning untuk Pengukuran Udang
│
├── 📁 Implementasi Sistem Pengukuran Otomatis
│
├── 📁 images
│
├── README.md
│
└── LICENSE
```

---

# 📚 Documentation

## 📁 1. Computer Vision untuk Pengukuran Udang

This folder contains all Computer Vision algorithms used for shrimp length measurement.

Contents include:

- Image Processing
- Image Enhancement
- Thresholding
- Morphological Operations
- Skeletonization
- Graph Theory
- Dijkstra Algorithm
- Pixel-to-Centimeter Conversion
- Shrimp Length Measurement

---

## 🤖 2. Deep Learning untuk Pengukuran Udang

This folder contains Deep Learning implementation.

Topics include:

- CNN
- Object Detection
- One-Stage Detector
- Two-Stage Detector
- Transfer Learning
- VGG16
- Fine Tuning
- Regression Layer
- Dataset Preparation
- Model Training
- Validation
- Overfitting
- Loss Function

---

## ⚙️ 3. Implementasi Sistem Pengukuran Otomatis

Complete implementation of the automatic shrimp measurement system.

Including:

- Camera Integration
- Image Acquisition
- AI Model Integration
- Automatic Preprocessing
- Automatic Length Measurement
- GUI
- Real-Time Monitoring

---

# 💻 Technologies

- Python
- OpenCV
- PyTorch
- Ultralytics YOLO11
- NumPy
- Matplotlib

---

# 📖 Learning Path

```
Digital Image
      │
      ▼
Computer Vision
      │
      ▼
Deep Learning
      │
      ▼
Object Detection
      │
      ▼
Shrimp Segmentation
      │
      ▼
Skeletonization
      │
      ▼
Graph + Dijkstra
      │
      ▼
Length Measurement
```

---

# 📸 Example Results

## Original Image

> *(Insert Image)*

---

## Object Detection

> *(Insert Image)*

---

## Binary Image

> *(Insert Image)*

---

## Skeleton

> *(Insert Image)*

---

## Final Measurement

> *(Insert Image)*

---

# 🚀 Getting Started

Clone this repository

```bash
git clone https://github.com/yourusername/Smart-Aquaculture.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run example

```bash
python main.py
```

---

# 📘 Book

This repository accompanies the book

**Smart Aquaculture**
**Automatic Vannamei Shrimp Length Measurement Using Camera and Artificial Intelligence**

---

# 🤝 Contributing

Contributions are welcome!

Feel free to submit:

- Bug fixes
- Improvements
- New algorithms
- Documentation

---

# ⭐ Support

If you find this repository useful, please consider giving it a ⭐.

It helps more people discover this project.

---

<div align="center">

Made with ❤️ for Smart Aquaculture

</div>

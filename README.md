# 🦐 SMART AQUACULTURE
**Implementasi Pengukuran Panjang Udang Vaname Menggunakan Kamera dan Artificial Intelligence**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Deep Learning](https://img.shields.io/badge/Deep_Learning-TensorFlow%20%7C%20PyTorch-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

Repositori ini berisi implementasi kode dan panduan proyek dari buku **"Smart Aquaculture: Implementasi Pengukuran Panjang Udang Vaname Menggunakan Kamera dan Artificial Intelligence"**. 

Proyek ini bertujuan untuk mengotomatisasi pengukuran morfometri (panjang tubuh) Udang Vaname (*Litopenaeus vannamei*) secara non-invasif menggunakan kamera bawah air (underwater camera), teknik *Computer Vision*, dan *Deep Learning*. Sistem ini dirancang untuk menggantikan metode pengukuran manual guna meningkatkan efisiensi, objektivitas, dan akurasi data dalam industri akuakultur (Revolusi Industri 4.0).

---

## 📑 Daftar Isi
1. [Latar Belakang](#-latar-belakang)
2. [Fitur Utama](#-fitur-utama)
3. [Metode Pengukuran](#-metode-pengukuran)
4. [Persyaratan Sistem](#-persyaratan-sistem)
5. [Struktur Repositori](#-struktur-repositori)
6. [Penggunaan](#-penggunaan)
7. [Dataset](#️-dataset)
8. [Evaluasi & Target Kinerja](#-evaluasi--target-kinerja)

---

## 🌊 Latar Belakang
Pengukuran panjang udang secara manual membutuhkan waktu lama, rentan terhadap bias operator, dan dapat memicu stres pada udang. Dengan integrasi *Computer Vision* dan model *Deep Learning* seperti **VGG16**, sistem ini mampu memproses citra udang secara otomatis, memisahkan objek dari latar belakang, menangani distorsi visual di dalam air (seperti kekeruhan dan hamburan cahaya), serta mengestimasi panjang tubuh dari rostrum hingga telson dalam satuan sentimeter (cm).

---

## ✨ Fitur Utama
* **Kalibrasi Kamera Bawah Air:** Perbaikan distorsi lensa (Radial & Tangensial) dan konversi satuan piksel ke sentimeter (cm).
* **Peningkatan Kualitas Citra (Image Enhancement):** Implementasi CLAHE, *White Balance*, dan *Dehazing* untuk mengatasi kekeruhan air dan pencahayaan yang tidak merata.
* **Filter Noise:** Penggunaan *Mean*, *Median*, dan *Gaussian Filter* untuk menangani *Gaussian, Speckle,* dan *Salt & Pepper Noise*.
* **Segmentasi Otomatis:** Pemisahan objek menggunakan metode *Otsu Thresholding*, *Semantic Segmentation*, dan *Instance Segmentation*.
* **Ekstraksi Centerline (Skeletonisasi):** Penerapan algoritma *Guo-Hall Thinning* dan penemuan jalur utama (panjang udang) menggunakan **Teori Graf** dan **Algoritma Dijkstra**.
* **Deep Learning Regresi:** *Transfer Learning* dan *Fine-Tuning* menggunakan arsitektur **VGG16** dengan *Regression Layer* untuk estimasi ukuran kontinu.

---

## 🔬 Metode Pengukuran
Proyek ini mengimplementasikan beberapa pendekatan pengukuran yang dapat dipilih sesuai kebutuhan:
1. **Pendekatan Skeletonisasi Dasar:** Segmentasi (Otsu/Morfologi) $\rightarrow$ Skeletonisasi Guo-Hall $\rightarrow$ Dijkstra $\rightarrow$ Estimasi Panjang.
2. **Instance Segmentation + Skeletonisasi:** Deteksi individu udang secara presisi $\rightarrow$ Pembentukan Biner Mask $\rightarrow$ Skeletonisasi $\rightarrow$ Estimasi Panjang.
3. **Deteksi Objek + Minimum Bounding Rectangle (MBBox):** Deteksi Bounding Box (YOLO/SSD) $\rightarrow$ Ekstraksi Kontur $\rightarrow$ Pembentukan MBBox miring (Rotated) $\rightarrow$ Perhitungan Mayor Axis.
4. **VGG16 Regression Pipeline:** Input Citra Udang $\rightarrow$ Ekstraksi Fitur Konvolusi (VGG16 Backbone) $\rightarrow$ Fully Connected Layer $\rightarrow$ Linear Output (Panjang Aktual).

---

## 💻 Persyaratan Sistem
Pastikan sistem Anda telah terinstal library berikut:
* Python 3.8+
* OpenCV (`cv2`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)
* TensorFlow / Keras (untuk arsitektur VGG16)
* Scikit-Learn (untuk evaluasi regresi)
* *Disarankan menggunakan GPU dengan CUDA yang aktif untuk mempercepat proses training.*

---

## 📂 Struktur Repositori
```text
├── Dataset/                 # Dataset citra mentah & ground truth
├── src/
│   ├── preprocessing/       # Modul Enhancement (CLAHE, Dehazing, dll)
│   ├── calibration/         # Skrip Kalibrasi kamera
│   └── cv_measurement/      # Segmentasi, Guo-Hall Thinning, & Dijkstra
├── requirements.txt         # Daftar dependencies
└── README.md                # Dokumentasi utama proyek

# Proyek Implementasi Pengukuran Panjang Udang Vaname

Repositori ini merupakan implementasi dari proyek pada buku **Smart Aquaculture: Implementasi Pengukuran Panjang Udang Vaname Menggunakan Kamera dan Artificial Intelligence**.

Proyek ini bertujuan membangun sistem pengukuran panjang Udang Vaname secara otomatis menggunakan teknik **Computer Vision** dan **Artificial Intelligence**, kemudian mengevaluasi hasil pengukuran menggunakan metrik MAE, RMSE, dan MAPE.

---

## Tujuan

- Mengimplementasikan sistem pengukuran panjang Udang Vaname.
- Menghasilkan estimasi panjang dalam satuan sentimeter.
- Mengevaluasi performa sistem menggunakan MAE, RMSE, dan MAPE.

---

## Dataset

Dataset dapat diunduh melalui tautan berikut.

> **Dataset:** *(Tambahkan tautan Google Drive atau GitHub Release di sini)*

Struktur dataset:

```
dataset/
│
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
│
├── labels/
│   └── ground_truth.csv
│
└── README.md
```

---

## Metode

Silakan memilih salah satu metode berikut.

- Object Detection + Skeletonization
- Object Detection + Minimum Bounding Rectangle
- Instance Segmentation + Skeletonization
- Instance Segmentation + Longest Pixel Distance

---

## Instalasi

Clone repository

```bash
git clone https://github.com/username/nama-repository.git
```

Masuk ke folder

```bash
cd nama-repository
```

Install library

```bash
pip install -r requirements.txt
```

---

## Menjalankan Program

Contoh menjalankan program

```bash
python main.py
```

atau

```bash
python predict.py
```

---

## Output

Program diharapkan menghasilkan:

- Hasil deteksi atau segmentasi
- Hasil pengukuran panjang
- Tabel hasil estimasi
- Nilai MAE
- Nilai RMSE
- Nilai MAPE

---

## Target Evaluasi

Implementasi dianggap berhasil apabila memenuhi target berikut.

| Parameter | Target |
|-----------|--------|
| MAE | ≤ 0.50 cm |
| RMSE | ≤ 0.60 cm |
| MAPE | ≤ 5% |

---

## Struktur Repository

```
project/
│
├── dataset/
├── models/
├── results/
├── src/
├── requirements.txt
├── main.py
└── README.md
```

---

## Referensi

Ryan A.

**Smart Aquaculture: Implementasi Pengukuran Panjang Udang Vaname Menggunakan Kamera dan Artificial Intelligence**

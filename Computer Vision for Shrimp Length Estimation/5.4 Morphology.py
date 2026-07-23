import cv2
import numpy as np
import matplotlib.pyplot as plt

# PENGATURAN
USE_UNDISTORT = True      # True = gunakan hasil kalibrasi
                          # False = gunakan gambar asli

# MEMBACA GAMBAR
gambar_bgr = cv2.imread(
    "WIN_20260109_13_48_33_Pro_jpg.rf.c19f4a5067f321134bd874314979a298.jpg"
)

if gambar_bgr is None:
    print("Gambar tidak ditemukan!")
    exit()

# UNDISTORT (OPSIONAL)
if USE_UNDISTORT:

    data = np.load("matriks_kamera_TA.npz")
    mtx = data["mtx"]
    dist = data["dist"]

    h, w = gambar_bgr.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx,
        dist,
        (w, h),
        1,
        (w, h)
    )

    gambar_bgr = cv2.undistort(
        gambar_bgr,
        mtx,
        dist,
        None,
        newcameramtx
    )

    x, y, w_roi, h_roi = roi
    gambar_bgr = gambar_bgr[y:y+h_roi, x:x+w_roi]

# KONVERSI KE GRAYSCALE
gambar_gray = cv2.cvtColor(
    gambar_bgr,
    cv2.COLOR_BGR2GRAY
)

# SEGMENTASI OTSU
_, gambar_biner = cv2.threshold(
    gambar_gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# OPERASI MORFOLOGI
kernel_close = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3,3)
)

kernel_open = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (7,7)
)

gambar_closing = cv2.morphologyEx(
    gambar_biner,
    cv2.MORPH_CLOSE,
    kernel_close
)

gambar_hasil = cv2.morphologyEx(
    gambar_closing,
    cv2.MORPH_OPEN,
    kernel_open
)

# VISUALISASI
plt.figure(figsize=(15,5))
plt.suptitle(
    "Tahapan Operasi Morfologi",
    fontsize=16,
    fontweight="bold"
)

# 1. Grayscale
plt.subplot(1,3,1)
plt.imshow(gambar_gray, cmap="gray")
plt.title("1. Grayscale")
plt.axis("off")


# 2. Closing
plt.subplot(1,3,2)
plt.imshow(gambar_closing, cmap="gray")
plt.title("2. Closing")
plt.axis("off")

plt.text(
    0.5,
    -0.12,
    "Dilasi → Erosi\nMenutup lubang kecil",
    transform=plt.gca().transAxes,
    ha="center",
    fontsize=9,
    style="italic"
)

# 3. Opening
plt.subplot(1,3,3)
plt.imshow(gambar_hasil, cmap="gray")
plt.title("3. Opening (Hasil Akhir)")
plt.axis("off")

plt.text(
    0.5,
    -0.12,
    "Erosi → Dilasi\nMenghilangkan noise",
    transform=plt.gca().transAxes,
    ha="center",
    fontsize=9,
    style="italic"
)

plt.tight_layout(rect=[0,0.05,1,0.93])
plt.show()

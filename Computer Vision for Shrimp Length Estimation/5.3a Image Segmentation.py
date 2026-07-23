import cv2
import numpy as np
import matplotlib.pyplot as plt

# SETTINGS
USE_UNDISTORT = True      # True = use camera calibration result
                           # False = use the original image directly

# READ IMAGE
gambar_bgr = cv2.imread(
    "WIN_20260109_13_48_33_Pro_jpg.rf.c19f4a5067f321134bd874314979a298.jpg"
)

if gambar_bgr is None:
    print("Image not found!")
    exit()

# UNDISTORT (OPTIONAL)
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

# CONVERT TO GRAYSCALE
gambar_gray = cv2.cvtColor(
    gambar_bgr,
    cv2.COLOR_BGR2GRAY
)

# IMAGE SEGMENTATION
nilai_ambang = 127

_, gambar_segmentasi = cv2.threshold(
    gambar_gray,
    nilai_ambang,
    255,
    cv2.THRESH_BINARY
)

# VISUALIZATION
plt.figure(figsize=(12, 6))

# Input image (original / undistorted)
plt.subplot(1, 2, 1)
plt.imshow(gambar_gray, cmap='gray', vmin=0, vmax=255)

plt.title("Original Image")

plt.axis("off")

# Segmentation result
plt.subplot(1, 2, 2)
plt.imshow(
    gambar_segmentasi,
    cmap="gray",
    vmin=0,
    vmax=255
)
plt.title(f"Segmentation (Threshold = {nilai_ambang})")
plt.axis("off")

plt.tight_layout()
plt.show()

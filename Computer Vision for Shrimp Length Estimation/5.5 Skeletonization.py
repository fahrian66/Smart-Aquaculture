import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_calibration_data(filepath="matriks_kamera.npz"):
    if os.path.exists(filepath):
        data = np.load(filepath)
        return data['mtx'], data['dist']
    print(f"⚠️ Warning: File {filepath} not found. Undistortion is disabled.")
    return None, None

def apply_undistort(img_bgr, mtx, dist, crop=True):
    if mtx is None or dist is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    alpha = 0 if crop else 1
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))
    dst = cv2.undistort(img_bgr, mtx, dist, None, newcameramtx)
    if crop:
        x, y, w_roi, h_roi = roi
        dst = dst[y:y+h_roi, x:x+w_roi]
    return dst

class SkeletonAnalyzerGuoHall:
    def __init__(self):
        self.original_image = None
        self.binary_image = None

    def set_image(self, image_array):
        self.original_image = image_array.copy()

    def to_binary(self, invert=False):
        if self.original_image is None:
            raise ValueError("Image has not been set!")

        if len(self.original_image.shape) == 3:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.original_image.copy()

        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, self.binary_image = cv2.threshold(
            gray,
            0,
            255,
            mode | cv2.THRESH_OTSU
        )

        # 3×3 elliptical closing followed by 7×7 elliptical opening
        kernelClose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        self.binary_image = cv2.morphologyEx(
            self.binary_image,
            cv2.MORPH_CLOSE,
            kernelClose
        )

        self.binary_image = cv2.morphologyEx(
            self.binary_image,
            cv2.MORPH_OPEN,
            kernelOpen
        )

        return self.binary_image

    def compute_skeleton(self):
        print("   └── Processing skeleton (pure Python implementation, please wait)...")

        img = (self.binary_image > 0).astype(np.uint8)
        prev = np.zeros_like(img)

        while not np.array_equal(img, prev):
            prev = img.copy()

            for sub_iter in range(2):
                marker = np.zeros_like(img)
                rows, cols = img.shape

                for i in range(1, rows - 1):
                    for j in range(1, cols - 1):

                        if img[i, j] == 0:
                            continue

                        p1, p2, p3 = img[i-1, j-1], img[i-1, j], img[i-1, j+1]
                        p8, p4 = img[i, j-1], img[i, j+1]
                        p7, p6, p5 = img[i+1, j-1], img[i+1, j], img[i+1, j+1]

                        C = ((1-p2) & (p3 | p4)) + ((1-p4) & (p5 | p6)) + \
                            ((1-p6) & (p7 | p8)) + ((1-p8) & (p1 | p2))

                        N1 = (p1 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8)
                        N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p1)
                        N = min(N1, N2)

                        m = ((p2 | p3 | (1-p5)) & p4) if sub_iter == 0 else \
                            ((p6 | p7 | (1-p1)) & p8)

                        if C == 1 and 2 <= N <= 3 and m == 0:
                            marker[i, j] = 1

                img[marker == 1] = 0

        self.skeleton = (img * 255).astype(np.uint8)
        return self.skeleton


if __name__ == "__main__":

    IMAGE_PATH = "shtest.jpg"  # Replace with your image path
    KALIBRASI_PATH = "matriks_kamera.npz"

    USE_UNDISTORT = True      # Set to False if lens correction is not required
    CROP_UNDISTORT = True     # True = crop black borders, False = keep full image

    # Load calibration data if undistortion is enabled
    mtx, dist = None, None
    if USE_UNDISTORT:
        mtx, dist = load_calibration_data(KALIBRASI_PATH)

    # Read the input image
    img_cv = cv2.imread(IMAGE_PATH)
    if img_cv is None:
        print(f"❌ Error: Image '{IMAGE_PATH}' not found!")
        exit()

    # Apply lens undistortion
    if USE_UNDISTORT and mtx is not None:
        print("[*] Applying lens distortion correction (undistortion)...")
        img_input_final = apply_undistort(
            img_cv,
            mtx,
            dist,
            crop=CROP_UNDISTORT
        )
    else:
        print("[*] Undistortion is disabled.")
        img_input_final = img_cv

    # Convert from OpenCV BGR format to Matplotlib RGB format
    img_rgb = cv2.cvtColor(img_input_final, cv2.COLOR_BGR2RGB)

    print("[*] Generating binary image...")
    skel_analyzer = SkeletonAnalyzerGuoHall()
    skel_analyzer.set_image(img_rgb)
    bin_img = skel_analyzer.to_binary(False)

    print("[*] Extracting raw skeleton...")
    raw_skeleton = skel_analyzer.compute_skeleton()

    print("\n✅ Finished: Raw skeleton successfully extracted without using AI.")

    # Display the processing results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("1. Original / Undistorted Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(bin_img, cmap="gray")
    plt.title("2. Binary Image (Elliptical Morphology)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(raw_skeleton, cmap="gray")
    plt.title("3. Raw Skeleton (Guo-Hall)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

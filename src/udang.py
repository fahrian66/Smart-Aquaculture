import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import hashlib
import os
import tempfile
import time

# --- IMPORT ENGINE UNTUK FPS TINGGI ---
import shrimp_cpp

# =========================================================================
# 1. SETUP HALAMAN WEB & INITIALISASI SESSION STATE
# =========================================================================
st.set_page_config(page_title="Shrimp Vision Web", page_icon="🦐", layout="wide")

def on_mode_change():
    st.session_state.sumber_gambar = None
    st.session_state.measurement_result = None
    st.session_state.last_upload_hash = None
    st.session_state.manual_measure_cm = None # <-- TAMBAHKAN INI

if 'sumber_gambar' not in st.session_state:
    st.session_state.sumber_gambar = None
if 'measurement_result' not in st.session_state:
    st.session_state.measurement_result = None
if 'kamera_usb_tersedia' not in st.session_state:
    st.session_state.kamera_usb_tersedia = [0]
if 'last_upload_hash' not in st.session_state:
    st.session_state.last_upload_hash = None

# --- TAMBAHKAN INISIALISASI INI ---
if 'manual_measure_cm' not in st.session_state:
    st.session_state.manual_measure_cm = None

if 'cal_last_upload_hash' not in st.session_state:
    st.session_state.cal_last_upload_hash = None
    
if 'cal_uploader_key' not in st.session_state:
    st.session_state.cal_uploader_key = "cal_file_up_0"

if 'gambar_kalibrasi' not in st.session_state:
    st.session_state.gambar_kalibrasi = None
if 'pixel_per_cm' not in st.session_state:
    st.session_state.pixel_per_cm = 23.54  # Nilai default terbaru

# =========================================================================
# 2. CACHE MODEL AI & DATA KALIBRASI 3D
# =========================================================================
@st.cache_resource
def load_ai_model():
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    return ShrimpDetector("calibrated/Grid14.h5")

@st.cache_resource
def load_calibration_data(filepath="matriks_kamera_TA.npz"):
    if os.path.exists(filepath):
        data = np.load(filepath)
        return data['mtx'], data['dist']
    return None, None

mtx, dist = load_calibration_data()

# Fungsi untuk menerapkan Undistort Lensa
def apply_undistort(img_bgr, mtx, dist, crop=True):
    h, w = img_bgr.shape[:2]
    alpha = 0 if crop else 1
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))
    dst = cv2.undistort(img_bgr, mtx, dist, None, newcameramtx)
    if crop:
        x, y, w_roi, h_roi = roi
        dst = dst[y:y+h_roi, x:x+w_roi]
    return dst

def kalibrasi_interaktif_opencv(img_bgr):
    clone = img_bgr.copy()
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        param['mouse_x'] = x
        param['mouse_y'] = y
        param['flags'] = flags  # Menyimpan status tombol keyboard (termasuk SHIFT)
        if event == cv2.EVENT_LBUTTONDOWN:
            # --- LOGIKA SNAP SHIFT KETIKA DIKLIK ---
            if len(points) == 1 and (flags & cv2.EVENT_FLAG_SHIFTKEY):
                x0, y0 = points[0]
                if abs(x - x0) > abs(y - y0): y = y0
                else: x = x0
                
            if len(points) < 2: points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN: 
            if points: points.pop()

    win_name = "Alat Kalibrasi Interaktif"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # Menambahkan state 'flags' untuk dilacak secara real-time
    state = {'mouse_x': 0, 'mouse_y': 0, 'flags': 0}
    cv2.setMouseCallback(win_name, mouse_callback, state)
    
    dist_px = None
    
    # Ambil resolusi gambar untuk batas garis bidik (crosshair)
    h_img, w_img = clone.shape[:2]
    
    while True:
        display = clone.copy()
        
        mx, my = state['mouse_x'], state['mouse_y']
        flags = state['flags']
        
        # --- LOGIKA SNAP UNTUK PREVIEW LIVE ---
        if len(points) == 1 and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            x0, y0 = points[0]
            if abs(mx - x0) > abs(my - y0):
                my = y0
            else:
                mx = x0

        # --- FITUR BARU: GARIS BIDIK (CROSSHAIR) FULL SCREEN ---
        # Membantu user melihat kelurusan horizontal/vertikal secara mutlak terhadap layar
        cv2.line(display, (0, my), (w_img, my), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(display, (mx, 0), (mx, h_img), (255, 255, 255), 1, cv2.LINE_AA)
        
        
        for p in points: cv2.circle(display, p, 3, (0, 255, 0), -1)
        
        if len(points) == 1:
            x0, y0 = points[0]
            # Tampilkan garis preview kuning tebal 
            cv2.line(display, (x0, y0), (mx, my), (0, 255, 255), 2, cv2.LINE_AA)
            
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (0, 0, 255), 2)
            dist_px = np.sqrt((points[0][0]-points[1][0])**2 + (points[0][1]-points[1][1])**2)
            
            mid_x = (points[0][0] + points[1][0]) // 2
            mid_y = (points[0][1] + points[1][1]) // 2
            teks_jarak = f"{dist_px:.2f} Piksel"
            (tw, th), _ = cv2.getTextSize(teks_jarak, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(display, (mid_x + 5, mid_y - th - 15), (mid_x + 15 + tw, mid_y - 5), (0,0,0), -1)
            cv2.putText(display, teks_jarak, (mid_x + 10, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(1)
        if key == 32 and len(points) == 2:
            cv2.destroyWindow(win_name)
            return dist_px
        elif key == 27: 
            cv2.destroyWindow(win_name)
            return None

# =========================================================================
# 3. KELAS DETEKSI AI
# =========================================================================
class ShrimpDetector:
    def __init__(self, model_path, target_size=448, conf_thresh=0.7):
        self.model = load_model(model_path, compile=False) 
        self.target_size = target_size
        self.conf_thresh = conf_thresh

    def preprocess(self, img_array):
        img0 = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        h0, w0 = img0.shape[:2]
        scale = min(self.target_size/w0, self.target_size/h0)
        nw, nh = int(w0*scale), int(h0*scale)
        img_resized = cv2.resize(img0, (nw, nh))
        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        pad_w, pad_h = (self.target_size - nw) // 2, (self.target_size - nh) // 2
        canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = img_resized
        img_input = np.expand_dims(canvas.astype(np.float32) / 255.0, axis=0)
        return img_array, img_input, (scale, pad_w, pad_h)

    def detect_objects(self, image_array):
        original_img, input_tensor, (scale, pad_w, pad_h) = self.preprocess(image_array)
        pred_grid = self.model.predict(input_tensor, verbose=0)
        grid = pred_grid[0]
        rows, cols = grid.shape[:2]
        cell_size = self.target_size / rows
        
        boxes_448, confidences, classes = [], [], []
        
        for r in range(rows):
            for c in range(cols):
                conf = float(grid[r, c, 0])
                if conf > self.conf_thresh:
                    tx, ty, tw, th = grid[r, c, 1:5]
                    cls_idx = int(np.argmax(grid[r, c, 5:])) 
                    
                    cx = (c + tx) * cell_size; cy = (r + ty) * cell_size
                    w = tw * self.target_size; h = th * self.target_size
                    
                    bx = int(cx - w/2); by = int(cy - h/2)
                    boxes_448.append([bx, by, int(w), int(h)])
                    confidences.append(conf)
                    classes.append(cls_idx)
        
        results = []
        if len(boxes_448) > 0:
            indices = cv2.dnn.NMSBoxes(boxes_448, confidences, self.conf_thresh, 0.4)
            if len(indices) > 0:
                for i in indices.flatten():
                    bx, by, bw, bh = boxes_448[i]
                    conf = confidences[i]
                    cls_idx = classes[i] 
                    
                    cx_448 = bx + bw/2; cy_448 = by + bh/2
                    
                    cx_orig = (cx_448 - pad_w) / scale; cy_orig = (cy_448 - pad_h) / scale
                    w_orig = bw / scale; h_orig = bh / scale
                    
                    x1 = max(0, int(cx_orig - w_orig/2))
                    y1 = max(0, int(cy_orig - h_orig/2))
                    x2 = min(original_img.shape[1], int(cx_orig + w_orig/2))
                    y2 = min(original_img.shape[0], int(cy_orig + h_orig/2))
                    
                    results.append(((x1, y1, x2, y2), conf, cls_idx)) 
        return results

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# =========================================================================
# 4. ANTARMUKA WEB STREAMLIT
# =========================================================================
try:
    detector = load_ai_model()
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")
    st.stop()

st.title("Shrimp Vision Web Dashboard")
st.markdown("Sistem Estimasi Panjang Udang Vaname Berbasis Web")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Konfigurasi Sistem")
st.sidebar.metric(label="Nilai Kalibrasi Aktif", value=f"{st.session_state.pixel_per_cm:.2f} px/cm")
st.sidebar.info("💡 Gunakan tab '🎯 Kalibrasi' untuk mengatur ulang nilai ini.")

st.sidebar.markdown("---")
new_val = st.sidebar.number_input("Atur Kalibrasi Manual:", 
                                  value=float(st.session_state.pixel_per_cm), 
                                  step=0.1)

if st.sidebar.button("✅ Update Nilai Kalibrasi"):
    st.session_state.pixel_per_cm = new_val
    st.success(f"Nilai diperbarui ke {new_val:.2f} px/cm")
    st.rerun() 

# --- TAMBAHAN: FITUR KOREKSI LENSA (UNDISTORT) ---
st.sidebar.markdown("---")
st.sidebar.subheader("📷 Koreksi Lensa (3D)")
if mtx is not None:
    use_undistort = st.sidebar.checkbox("Aktifkan Undistort Lensa", value=True, help="Mengoreksi efek cembung pada kamera bawah air. Peringatan: Fitur ini dapat menurunkan kecepatan (FPS).")
    if use_undistort:
        crop_undistort = st.sidebar.checkbox("Crop Area Melengkung (Hitam)", value=True, help="Memotong tepian foto agar lurus. Ingat: Jika di-crop, Anda harus mengkalibrasi ulang penggaris!")
    else:
        crop_undistort = False
else:
    st.sidebar.warning("⚠️ File 'matriks_kamera_TA.npz' tidak ditemukan. Fitur ini dinonaktifkan.")
    use_undistort = False
    crop_undistort = False

# --- TAMBAHAN: INPUT PANJANG AKTUAL ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Evaluasi Error (Opsional)")
aktual_input = st.sidebar.text_input(
    "Panjang Aktual (cm):", 
    value="0",
    help="Isi untuk melihat % Error. Jika ada 2 udang berbeda ukuran, pisahkan dengan koma (contoh: 10.5, 8.2)."
)

try:
    aktual_list = [float(x.strip()) for x in aktual_input.split(",") if x.strip() != "" and float(x.strip()) > 0]
except ValueError:
    aktual_list = []

# --- TAB UTAMA ---
tab_pengukuran, tab_kalibrasi = st.tabs(["📏 Pengukuran Udang", "🎯 Kalibrasi Piksel"])

# =========================================================================
# TAB PENGUKURAN
# =========================================================================
with tab_pengukuran:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Gambar/Video Udang")
        
        mode_pilihan = st.radio(
            "Pilih Metode Input:", 
            ["📸 Kamera USB", "📂 Upload Foto", "🎥 Upload Video"], 
            horizontal=True,
            key="active_input_mode",
            on_change=on_mode_change
        )
        
        if mode_pilihan == "📸 Kamera USB":
            st.info("Akses langsung ke port fisik USB Laptop/Server.")
            
            if st.button("🔍 Scan Kamera USB", key="btn_scan_ukur"):
                with st.spinner("Mendeteksi port kamera..."):
                    found = []
                    for i in range(5):
                        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                        if cap.isOpened():
                            ret, _ = cap.read()
                            if ret: found.append(i)
                            cap.release()
                    st.session_state.kamera_usb_tersedia = found or [0]
                    st.success(f"Ditemukan {len(st.session_state.kamera_usb_tersedia)} kamera terhubung.")

            cam_idx = st.selectbox("Pilih Index Kamera:", st.session_state.kamera_usb_tersedia, format_func=lambda x: f"Kamera USB {x}")
            
            if st.button("📸 Buka Kamera & Jepret", key="btn_buka_usb_ukur"):
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW) 
                if cap.isOpened():
                    st.info("ℹ️ **Cara Jepret:** Posisikan udang di jendela kamera, lalu tekan **SPASI**. Tekan **ESC** untuk batal.")
                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        cv2.imshow(f"Live Preview - Kamera USB {cam_idx}", frame)
                        key = cv2.waitKey(1)
                        if key == 32:  # SPASI
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.session_state.sumber_gambar = Image.fromarray(frame_rgb)
                            st.session_state.measurement_result = None
                            break
                        elif key == 27:  # ESC
                            break
                    cap.release()
                    cv2.destroyAllWindows()
                    st.rerun() 
                else:
                    st.error("Gagal membuka kamera USB tersebut.")
                
        elif mode_pilihan == "📂 Upload Foto":
            upload_buffer = st.file_uploader("Pilih file gambar udang", type=["jpg", "jpeg", "png"], key="file_uploader_ukur")
            if upload_buffer is not None:
                upload_hash = hashlib.md5(upload_buffer.getvalue()).hexdigest()
                if st.session_state.last_upload_hash != upload_hash:
                    st.session_state.sumber_gambar = Image.open(upload_buffer).convert("RGB")
                    st.session_state.measurement_result = None 
                    st.session_state.last_upload_hash = upload_hash
                    st.success("File foto baru berhasil dimuat.")

        elif mode_pilihan == "🎥 Upload Video":
            st.info("Akselerasi Aktif! Video diproses dengan FPS Tinggi.")
            video_file = st.file_uploader("Upload File Video", type=["mp4", "avi", "mov"], key="vid_uploader")
            
            if video_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(video_file.read())
                
                if st.button("▶️ Proses Video Secara Live", key="btn_proses_vid"):
                    st.session_state.sumber_gambar = None
                    st.session_state.measurement_result = None
                    
                    cap = cv2.VideoCapture(tfile.name)
                    st.info("ℹ️ **Sedang memproses...** Jendela video OpenCV terbuka. Tekan tombol **ESC** pada jendela video untuk menghentikan.")
                    
                    prev_time = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.success("✅ Pemutaran Video Selesai!")
                            break
                        
                        # --- TERAPKAN UNDISTORT JIKA DIAKTIFKAN ---
                        if use_undistort and mtx is not None:
                            frame = apply_undistort(frame, mtx, dist, crop_undistort)
                            
                        detections = detector.detect_objects(frame)
                        results_to_draw = []
                        pipeline_steps = []
                        
                        for (x1, y1, x2, y2), conf_score, cls_idx in detections:
                            if cls_idx == 0:
                                results_to_draw.append({
                                    'box_coords': (x1, y1, x2, y2),
                                    'conf': conf_score,
                                    'is_potongan': True
                                })
                                continue
                            
                            pad = 10
                            x1_c, y1_c = max(0, x1 - pad), max(0, y1 - pad)
                            x2_c, y2_c = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
                            
                            cropped_img = frame[y1_c:y2_c, x1_c:x2_c].copy()
                            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                            
                            skel_analyzer = shrimp_cpp.SkeletonAnalyzerGuoHall()
                            skel_analyzer.set_image(cropped_img_rgb)
                            bin_img = skel_analyzer.to_binary(False) 
                            raw_skeleton = skel_analyzer.compute_skeleton()
                            
                            graph_tool = shrimp_cpp.GraphSkeletonAnalyzer(st.session_state.pixel_per_cm)
                            length_px, main_path = graph_tool.find_main_path(raw_skeleton)
                            length_cm = length_px / st.session_state.pixel_per_cm
                            
                            final_vis_clean_rgb = graph_tool.visualize_on_image(cropped_img_rgb, main_path)
                            final_vis_clean_bgr = cv2.cvtColor(final_vis_clean_rgb, cv2.COLOR_RGB2BGR)
                            
                            results_to_draw.append({
                                'roi_coords': (x1_c, y1_c, x2_c, y2_c),
                                'box_coords': (x1, y1, x2, y2),
                                'vis': final_vis_clean_bgr, 
                                'length': length_cm,
                                'conf': conf_score,
                                'is_potongan': False
                            })
                            
                        # --- URUTKAN UDANG KIRI KE KANAN & HITUNG ERROR INDIVIDU (AE & PE) ---
                        utuh_items = [r for r in results_to_draw if not r['is_potongan']]
                        utuh_items.sort(key=lambda r: r['box_coords'][0]) 
                        
                        for idx, item in enumerate(utuh_items):
                            aktual = 0.0
                            if len(aktual_list) == 1:
                                aktual = aktual_list[0] 
                            elif len(aktual_list) > idx:
                                aktual = aktual_list[idx] 
                                
                            if aktual > 0:
                                item['ae'] = abs(aktual - item['length'])
                                item['pe'] = (item['ae'] / aktual) * 100
                                item['aktual'] = aktual

                        for item in results_to_draw:
                            if not item.get('is_potongan'):
                                x1_c, y1_c, x2_c, y2_c = item['roi_coords']
                                try: frame[y1_c:y2_c, x1_c:x2_c] = item['vis']
                                except: pass
                                
                        for item in results_to_draw:
                            x1, y1, x2, y2 = item['box_coords']
                            if item.get('is_potongan'):
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                label_text = f"Potongan | {item['conf']*100:.0f}%"
                                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                bg_y1 = max(0, y1 - th - 10)
                                cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 10, y1), (0, 0, 255), -1) 
                                cv2.putText(frame, label_text, (x1 + 5, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            else:
                                try:
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    if 'pe' in item:
                                        label_text = f"L: {item['length']:.2f}cm (Err:{item['pe']:.1f}%)"
                                    else:
                                        label_text = f"L: {item['length']:.2f}cm | {item['conf']*100:.0f}%"
                                        
                                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                    bg_y1 = max(0, y1 - th - 10)
                                    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 10, y1), (0, 255, 0), -1) 
                                    cv2.putText(frame, label_text, (x1 + 5, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                except: pass

                        curr_time = time.time()
                        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
                        prev_time = curr_time
                        
                        fps_text = f"FPS: {int(fps)}"
                        cv2.rectangle(frame, (10, 10), (140, 45), (0, 0, 0), -1)
                        cv2.putText(frame, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        valid_err_items = [item for item in utuh_items if 'ae' in item]
                        if len(valid_err_items) > 0:
                            avg_mae = sum(item['ae'] for item in valid_err_items) / len(valid_err_items)
                            avg_mape = sum(item['pe'] for item in valid_err_items) / len(valid_err_items)
                            err_text = f"Avg Error: {avg_mape:.1f}% (MAE: {avg_mae:.2f}cm)"
                            (tw, th), _ = cv2.getTextSize(err_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(frame, (10, 50), (10 + tw + 10, 50 + th + 10), (0, 0, 0), -1)
                            cv2.putText(frame, err_text, (15, 50 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                        cv2.imshow("Live Video Processing", frame)
                        
                        key = cv2.waitKey(1)
                        if key == 27: 
                            st.warning("Pemrosesan video dihentikan oleh pengguna.")
                            break
                            
                    cap.release()
                    cv2.destroyAllWindows()

    with col2:
        st.subheader("🔬 Hasil Analisis (Foto/Kamera)")
            
        if st.session_state.sumber_gambar is None:
            st.info("Silakan ambil foto udang atau jalankan video di menu sebelah kiri.")
        else:
            # Menampilkan preview gambar yang akan diproses
            display_img = st.session_state.sumber_gambar
            if use_undistort and mtx is not None:
                st.warning("🔄 Gambar ini akan dikenakan proses Undistort sebelum deteksi.")
                
            st.image(display_img, caption="Gambar Siap Diproses", use_container_width=True)
            st.markdown("---") 
            
            # --- TAMBAHAN FITUR UKUR MANUAL & TOMBOL PROSES ---
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                btn_proses_ai = st.button("🚀 PROSES AI & UKUR", key="btn_proses", use_container_width=True)
            with col_btn2:
                btn_ukur_manual = st.button("📐 UKUR MANUAL (Opsional)", key="btn_manual", use_container_width=True, help="Klik 2 titik secara manual sebagai referensi ukuran")

            # Jika tombol ukur manual ditekan
            if btn_ukur_manual:
                img_cv_manual = cv2.cvtColor(np.array(st.session_state.sumber_gambar), cv2.COLOR_RGB2BGR)
                # Samakan persepsi dengan AI (Jika undistort nyala, luruskan juga sebelum diukur)
                if use_undistort and mtx is not None:
                    img_cv_manual = apply_undistort(img_cv_manual, mtx, dist, crop_undistort)

                # Panggil jendela OpenCV (bisa pakai SHIFT agar lurus)
                jarak_px = kalibrasi_interaktif_opencv(img_cv_manual)
                
                if jarak_px is not None and jarak_px > 0:
                    panjang_cm = jarak_px / st.session_state.pixel_per_cm
                    st.session_state.manual_measure_cm = panjang_cm
                    st.rerun()

            # Tampilkan hasil ukuran manual jika ada
            if st.session_state.manual_measure_cm is not None:
                st.info(f"📐 **Hasil Ukur Manual (Referensi):** {st.session_state.manual_measure_cm:.2f} cm")

            # Jika tombol Proses AI ditekan
            if btn_proses_ai:
                with st.spinner('Menjalankan AI dan algoritma pengukuran...'):
                    try:
                        img_cv = cv2.cvtColor(np.array(st.session_state.sumber_gambar), cv2.COLOR_RGB2BGR)
                        
                        # --- TERAPKAN UNDISTORT JIKA DIAKTIFKAN ---
                        if use_undistort and mtx is not None:
                            img_cv = apply_undistort(img_cv, mtx, dist, crop_undistort)
                            
                        detections = detector.detect_objects(img_cv)
                        
                        utuh_count = 0
                        potongan_count = 0
                        pipeline_steps = []
                        
                        if len(detections) > 0:
                            results_to_draw = []
                            for (x1, y1, x2, y2), conf_score, cls_idx in detections:
                                
                                if cls_idx == 0:
                                    potongan_count += 1
                                    pad = 10
                                    x1_c, y1_c = max(0, x1 - pad), max(0, y1 - pad)
                                    x2_c, y2_c = min(img_cv.shape[1], x2 + pad), min(img_cv.shape[0], y2 + pad)
                                    cropped_img = img_cv[y1_c:y2_c, x1_c:x2_c].copy()
                                    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                                    
                                    results_to_draw.append({
                                        'box_coords': (x1, y1, x2, y2),
                                        'conf': conf_score,
                                        'is_potongan': True
                                    })
                                    pipeline_steps.append({
                                        'is_potongan': True,
                                        'crop': cropped_img_rgb,
                                        'conf': conf_score
                                    })
                                    continue
                                
                                utuh_count += 1
                                pad = 10
                                x1_c, y1_c = max(0, x1 - pad), max(0, y1 - pad)
                                x2_c, y2_c = min(img_cv.shape[1], x2 + pad), min(img_cv.shape[0], y2 + pad)
                                
                                cropped_img = img_cv[y1_c:y2_c, x1_c:x2_c].copy()
                                cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                                
                                skel_analyzer = shrimp_cpp.SkeletonAnalyzerGuoHall()
                                skel_analyzer.set_image(cropped_img_rgb) 
                                bin_img = skel_analyzer.to_binary(False) 
                                raw_skeleton = skel_analyzer.compute_skeleton()
                                
                                graph_tool = shrimp_cpp.GraphSkeletonAnalyzer(st.session_state.pixel_per_cm)
                                length_px, main_path = graph_tool.find_main_path(raw_skeleton)
                                length_cm = length_px / st.session_state.pixel_per_cm
                                
                                final_vis_clean_rgb = graph_tool.visualize_on_image(cropped_img_rgb, main_path)
                                final_vis_clean_bgr = cv2.cvtColor(final_vis_clean_rgb, cv2.COLOR_RGB2BGR)
                                
                                final_vis_spurs_rgb = graph_tool.visualize_on_image(cropped_img_rgb, raw_skeleton, main_path)
                                final_vis_spurs_bgr = cv2.cvtColor(final_vis_spurs_rgb, cv2.COLOR_RGB2BGR)
                                
                                results_to_draw.append({
                                    'roi_coords': (x1_c, y1_c, x2_c, y2_c),
                                    'box_coords': (x1, y1, x2, y2),
                                    'vis': final_vis_clean_bgr, 
                                    'length': length_cm,
                                    'conf': conf_score,
                                    'is_potongan': False
                                })
                                
                                pipeline_steps.append({
                                    'is_potongan': False,
                                    'crop': cropped_img_rgb,
                                    'bin': bin_img,
                                    'skel': raw_skeleton,
                                    'vis': final_vis_spurs_bgr, 
                                    'length': length_cm,
                                    'conf': conf_score
                                })
                            
                            # --- URUTKAN UDANG KIRI KE KANAN & HITUNG ERROR INDIVIDU (AE & PE) ---
                            utuh_items = [r for r in results_to_draw if not r['is_potongan']]
                            utuh_items.sort(key=lambda r: r['box_coords'][0]) 
                            
                            for idx, item in enumerate(utuh_items):
                                aktual = 0.0
                                if len(aktual_list) == 1:
                                    aktual = aktual_list[0]
                                elif len(aktual_list) > idx:
                                    aktual = aktual_list[idx]
                                    
                                if aktual > 0:
                                    item['ae'] = abs(aktual - item['length'])
                                    item['pe'] = (item['ae'] / aktual) * 100
                                    item['aktual'] = aktual
                                    
                                    for step in pipeline_steps:
                                        if not step.get('is_potongan') and step['length'] == item['length']:
                                            step['ae'] = item['ae']
                                            step['pe'] = item['pe']
                                            step['aktual'] = item['aktual']

                            img_vis = img_cv.copy()
                            for item in results_to_draw:
                                if not item.get('is_potongan'):
                                    x1_c, y1_c, x2_c, y2_c = item['roi_coords']
                                    try: img_vis[y1_c:y2_c, x1_c:x2_c] = item['vis']
                                    except: pass
                                    
                            for item in results_to_draw:
                                x1, y1, x2, y2 = item['box_coords']
                                if item.get('is_potongan'):
                                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    label_text = f"Potongan | {item['conf']*100:.0f}%"
                                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                    bg_y1 = max(0, y1 - th - 10)
                                    cv2.rectangle(img_vis, (x1, bg_y1), (x1 + tw + 10, y1), (0, 0, 255), -1) 
                                    cv2.putText(img_vis, label_text, (x1 + 5, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                else:
                                    try:
                                        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        if 'pe' in item:
                                            label_text = f"L: {item['length']:.2f}cm (Err:{item['pe']:.1f}%)"
                                        else:
                                            label_text = f"L: {item['length']:.2f}cm | {item['conf']*100:.0f}%"
                                            
                                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                        bg_y1 = max(0, y1 - th - 10)
                                        cv2.rectangle(img_vis, (x1, bg_y1), (x1 + tw + 10, y1), (0, 255, 0), -1) 
                                        cv2.putText(img_vis, label_text, (x1 + 5, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                    except: pass
                            
                            st.session_state.measurement_result = { 
                                'status': 'success', 
                                'vis_img': img_vis, 
                                'count': utuh_count, 
                                'potongan_count': potongan_count,
                                'pipeline_steps': pipeline_steps
                            }
                        else:
                            st.session_state.measurement_result = { 'status': 'failed' }
                            
                        st.rerun()
                            
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

            if st.session_state.measurement_result is not None:
                res = st.session_state.measurement_result
                if res.get('status') == 'success':
                    st.image(cv_to_pil(res['vis_img']), caption="Analisis Deteksi Multi-Objek", use_container_width=True)
                    
                    pesan = f"✅ Berhasil! Terdeteksi **{res['count']} udang utuh**"
                    if res.get('potongan_count', 0) > 0:
                        pesan += f" dan **{res['potongan_count']} potongan udang**"
                    pesan += " pada gambar ini."
                    
                    st.success(pesan)
                    
                    # --- MENAMPILKAN RATA-RATA ERROR (MAE & MAPE) KESELURUHAN GAMBAR ---
                    if 'pipeline_steps' in res and len(res['pipeline_steps']) > 0:
                        valid_err_items = [step for step in res['pipeline_steps'] if not step.get('is_potongan') and 'pe' in step]
                        if len(valid_err_items) > 0:
                            avg_mae = sum(step['ae'] for step in valid_err_items) / len(valid_err_items)
                            avg_mape = sum(step['pe'] for step in valid_err_items) / len(valid_err_items)
                            
                            st.markdown("##### 📈 Ringkasan Error Pengukuran")
                            m1, m2 = st.columns(2)
                            m1.metric("MAE (Selisih Aktual)", f"{avg_mae:.3f} cm")
                            m2.metric("MAPE (Persentase Error)", f"{avg_mape:.2f} %")
                            st.markdown("---")
                    
                    if 'pipeline_steps' in res and len(res['pipeline_steps']) > 0:
                        with st.expander("🔍 Lihat Hasil Detail Pipeline (4 Tahapan)"):
                            for idx, step in enumerate(res['pipeline_steps']):
                                if step.get('is_potongan'):
                                    st.markdown(f"**Objek #{idx+1}** - Udang Terpotong (Confidence: {step['conf']*100:.0f}%)")
                                    st.image(step['crop'], caption="Crop Udang Terpotong", width=200)
                                else:
                                    if 'pe' in step:
                                        st.markdown(f"**Objek #{idx+1}** - Udang Utuh (Aktual: {step['aktual']} cm | Prediksi: {step['length']:.2f} cm | **Error: {step['pe']:.2f}%**)")
                                    else:
                                        st.markdown(f"**Objek #{idx+1}** - Udang Utuh (Prediksi: {step['length']:.2f} cm)")
                                        
                                    c1, c2, c3, c4 = st.columns(4)
                                    c1.image(step['crop'], caption="1. Crop Asli (RGB)")
                                    c2.image(step['bin'], caption="2. Biner & Morfologi", clamp=True)
                                    c3.image(step['skel'], caption="3. Raw Skeleton", clamp=True)
                                    c4.image(cv_to_pil(step['vis']), caption="4. Final Centerline (Dengan Spurs)")
                                st.markdown("---")
                else:
                    st.error("❌ **BUKAN UDANG / OBJEK TIDAK TERDETEKSI**")
                    st.warning("AI tidak dapat menemukan pola tubuh udang vaname pada gambar ini.")

# =========================================================================
# TAB KALIBRASI
# =========================================================================
with tab_kalibrasi:
    st.header("🎯 Kalibrasi Piksel ke Centimeter (Interaktif)")
    st.markdown("""
    Fitur ini digunakan untuk memberi tahu sistem berapa jumlah piksel yang mewakili 1 Centimeter pada jarak kamera Anda saat ini.
    **Langkah-langkah:**
    1. Masukkan panjang benda referensi (misal: panjang koin 2.5 cm, atau penggaris 10 cm).
    2. Ambil foto/upload benda referensi tersebut.
    3. Klik **"Buka Alat Kalibrasi Interaktif"**. Jendela baru akan terbuka dengan fitur **Kaca Pembesar**.
    4. Klik tepat 2 titik di ujung benda, lalu tekan **SPASI** untuk menyimpan.
    """)
    
    st.markdown("---")
    
    referensi_cm = st.number_input("1. Masukkan Panjang Benda Referensi yang sebenarnya (dalam CM):", min_value=0.1, max_value=100.0, value=25.40, step=0.5)
    st.markdown("<br>2. Masukkan Gambar Benda Referensi:", unsafe_allow_html=True)
    
    cal_kamera, cal_upload = st.columns(2)
    
    with cal_kamera:
        st.markdown("**📸 Dari Kamera USB**")
        if st.button("🔍 Scan Kamera USB", key="btn_scan_cal"):
            with st.spinner("Mendeteksi port kamera..."):
                found = []
                for i in range(5):
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret: found.append(i)
                        cap.release()
                st.session_state.kamera_usb_tersedia = found or [0]
                st.success(f"Ditemukan {len(st.session_state.kamera_usb_tersedia)} kamera.")

        cam_idx_cal = st.selectbox("Pilih Index Kamera:", st.session_state.kamera_usb_tersedia, key="cal_cam_idx", format_func=lambda x: f"Kamera USB {x}")
        
        if st.button("Jepret Benda Referensi", key="btn_jepret_cal"):
            cap = cv2.VideoCapture(cam_idx_cal, cv2.CAP_DSHOW)
            if cap.isOpened():
                st.info("ℹ️ **Instruksi Kalibrasi:** Tekan **SPASI** untuk mengambil foto benda referensi atau **ESC** untuk keluar.")
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    cv2.imshow(f"Kamera Kalibrasi {cam_idx_cal}", frame)
                    key = cv2.waitKey(1)
                    if key == 32:  
                        st.session_state.gambar_kalibrasi = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # --- KUNCI SOLUSINYA DI SINI: PAKSA KOTAK UPLOAD RESET ---
                        st.session_state.cal_uploader_key = str(time.time())
                        st.session_state.cal_last_upload_hash = None
                        break
                    elif key == 27: break
                cap.release()
                cv2.destroyAllWindows()
                st.rerun()

    with cal_upload:
        st.markdown("**📂 Upload File**")
        cal_file = st.file_uploader("Atau pilih file gambar referensi", type=["jpg", "jpeg", "png"], key=st.session_state.cal_uploader_key)
        if cal_file is not None:
            upload_hash = hashlib.md5(cal_file.getvalue()).hexdigest()
            # Hanya timpa gambar jika file yang diupload adalah file BARU
            if st.session_state.cal_last_upload_hash != upload_hash:
                st.session_state.gambar_kalibrasi = np.array(Image.open(cal_file).convert("RGB"))
                st.session_state.cal_last_upload_hash = upload_hash
        else:
            # Jika user menekan tombol 'X' atau kotak tereset, reset hash
            st.session_state.cal_last_upload_hash = None

    if st.session_state.gambar_kalibrasi is not None:
        st.markdown("---")
        st.markdown("### 3. Eksekusi Kalibrasi")
        
        # --- PERBAIKAN: TENTUKAN GAMBAR YANG AKAN DIKALIBRASI (ASLI ATAU UNDISTORT) ---
        if use_undistort and mtx is not None:
            # Luruskan gambar kalibrasi jika fitur koreksi lensa menyala
            img_bgr_raw = cv2.cvtColor(st.session_state.gambar_kalibrasi, cv2.COLOR_RGB2BGR)
            img_bgr_undistorted = apply_undistort(img_bgr_raw, mtx, dist, crop_undistort)
            cal_img_show = cv2.cvtColor(img_bgr_undistorted, cv2.COLOR_BGR2RGB)
            st.warning("⚠️ **PERHATIAN:** Anda mengaktifkan Koreksi Lensa (Undistort) di menu samping. Gambar kalibrasi di bawah ini telah diluruskan dan di-crop secara otomatis agar rasio pikselnya cocok dengan sistem!")
        else:
            # Gunakan gambar asli
            cal_img_show = st.session_state.gambar_kalibrasi

        st.image(cal_img_show, caption="Gambar Benda Referensi (Siap Kalibrasi)", width=400)
        
        st.info("ℹ️ **Cara Kalibrasi:** \n"
                "1. Klik tombol di bawah, jendela OpenCV akan terbuka.\n"
                "2. **Klik Kiri 2x** di kedua ujung benda referensi.\n"
                "3. 💡 **TIPS: Tahan tombol SHIFT saat mengklik titik kedua agar garis lurus sempurna (horizontal/vertikal).**\n"
                "4. Tekan **SPASI** untuk menyimpan, atau **ESC** untuk batal.")
        
        if st.button("🔍 Buka Alat Kalibrasi Interaktif (Klik 2 Titik)"):
            img_bgr_cal = cv2.cvtColor(cal_img_show, cv2.COLOR_RGB2BGR)
            jarak_piksel = kalibrasi_interaktif_opencv(img_bgr_cal)
            
            if jarak_piksel is not None and jarak_piksel > 0:
                nilai_px_cm = jarak_piksel / referensi_cm
                st.session_state.pixel_per_cm = nilai_px_cm
                
                st.session_state.temp_hasil_kalibrasi = {
                    'jarak_piksel': jarak_piksel,
                    'referensi_cm': referensi_cm,
                    'nilai_px_cm': nilai_px_cm
                }
                
                st.success(f"✅ KALIBRASI BERHASIL DIPERBARUI KE SELURUH SISTEM!")
                st.balloons()
                time.sleep(1.5)
                st.rerun()
            else:
                st.warning("Kalibrasi dibatalkan atau titik yang dimasukkan kurang dari 2.")
                
        if 'temp_hasil_kalibrasi' in st.session_state:
            hasil = st.session_state.temp_hasil_kalibrasi
            col_hasil1, col_hasil2 = st.columns(2)
            with col_hasil1:
                st.info(f"📏 Jarak Terukur: **{hasil['jarak_piksel']:.2f} piksel**")
                st.info(f"📏 Referensi Asli: **{hasil['referensi_cm']} cm**")
            with col_hasil2:
                st.success(f"📌 Faktor Kalibrasi: **{hasil['nilai_px_cm']:.2f} px/cm**")
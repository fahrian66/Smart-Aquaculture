import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import deque
 
def load_calibration_data(filepath="matriks_kamera_TA.npz"):
    if os.path.exists(filepath):
        data = np.load(filepath)
        return data['mtx'], data['dist']
    # ⚠️ Warning: File {filepath} not found. Undistort feature is disabled.
    print(f"⚠️ Warning: File {filepath} not found. Undistort feature is disabled.")
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
        # Image has not been set yet!
        if self.original_image is None: raise ValueError("Image has not been set yet!")
        if len(self.original_image.shape) == 3:
             gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
             gray = self.original_image.copy()
        
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, self.binary_image = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
            
        # 100% IDENTICAL TO C++: Elliptical Kernels 3x3 and 7x7
        kernelClose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernelOpen  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, kernelClose)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernelOpen)
            
        return self.binary_image
 
    def compute_skeleton(self):
        # └── Processing Skeleton (Pure Python, please wait a few seconds)...
        print("   └── Processing Skeleton (Pure Python, please wait a few seconds)...")
        img = (self.binary_image > 0).astype(np.uint8)
        prev = np.zeros_like(img)
 
        while not np.array_equal(img, prev):
            prev = img.copy()
            for sub_iter in range(2):
                marker = np.zeros_like(img)
                rows, cols = img.shape
                for i in range(1, rows - 1):
                    for j in range(1, cols - 1):
                        if img[i, j] == 0: continue
                        
                        p1, p2, p3 = img[i-1, j-1], img[i-1, j], img[i-1, j+1]
                        p8, p4     = img[i, j-1],   img[i, j+1]
                        p7, p6, p5 = img[i+1, j-1], img[i+1, j], img[i+1, j+1]
 
                        # Bitwise Logic (AND / OR) made identically equivalent to C++
                        C = ((1-p2) & (p3 | p4)) + ((1-p4) & (p5 | p6)) + \
                            ((1-p6) & (p7 | p8)) + ((1-p8) & (p1 | p2))
                        
                        N1 = (p1 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8)
                        N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p1)
                        N = min(N1, N2)
 
                        m = ((p2 | p3 | (1-p5)) & p4) if sub_iter == 0 else ((p6 | p7 | (1-p1)) & p8)
 
                        if C == 1 and 2 <= N <= 3 and m == 0:
                            marker[i, j] = 1
 
                img[marker == 1] = 0
 
        self.skeleton = (img * 255).astype(np.uint8)
        return self.skeleton
 
class GraphSkeletonAnalyzer:
    def __init__(self, pixel_per_cm=24.31):
        self.pixel_per_cm = float(pixel_per_cm)
 
    def find_main_path(self, skeleton_img):
        # └── Computing Longest Path (Dijkstra Graph)...
        print("   └── Computing Longest Path (Dijkstra Graph)...")
        skel = skeleton_img > 0
        y_coords, x_coords = np.where(skel)
        if len(y_coords) == 0: return 0.0, []
 
        pixels = list(zip(y_coords, x_coords))
        pt_to_id = {pt: i for i, pt in enumerate(pixels)}
        id_to_pt = {i: pt for i, pt in enumerate(pixels)}
        n_pixels = len(pixels)
 
        adj = [[] for _ in range(n_pixels)]
        dx = [-1, 0, 1, -1, 1, -1, 0, 1]
        dy = [-1, -1, -1, 0, 0, 1, 1, 1]
 
        # Build Adjacency List
        for i, p in enumerate(pixels):
            for k in range(8):
                ny, nx = p[0] + dy[k], p[1] + dx[k]
                if (ny, nx) in pt_to_id:
                    n_id = pt_to_id[(ny, nx)]
                    weight = np.sqrt(dx[k]**2 + dy[k]**2)
                    adj[i].append((n_id, float(weight)))
 
        if n_pixels < 2: return 0.0, []
 
        # Find Largest Sub-Graph (BFS)
        visited = [False] * n_pixels
        largest_component = []
 
        for i in range(n_pixels):
            if not visited[i]:
                component = []
                q = deque([i])
                visited[i] = True
                while q:
                    curr = q.popleft()
                    component.append(curr)
                    for n_id, _ in adj[curr]:
                        if not visited[n_id]:
                            visited[n_id] = True
                            q.append(n_id)
                if len(component) > len(largest_component):
                    largest_component = component
 
        in_subgraph = [False] * n_pixels
        for id in largest_component:
            in_subgraph[id] = True
 
        # Find Endpoints with Degree = 1
        endpoints = []
        for id in largest_component:
            degree = sum(1 for n_id, _ in adj[id] if in_subgraph[n_id])
            if degree == 1:
                endpoints.append(id)
        if not endpoints:
            endpoints = largest_component
 
        # Line Regression & Extreme Branch Elimination (< 60 px)
        pts_for_line = np.array([(id_to_pt[id][1], id_to_pt[id][0]) for id in largest_component], dtype=np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(pts_for_line, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])
 
        valid_endpoints = []
        for ep_id in endpoints:
            p = id_to_pt[ep_id]
            dist_to_spine = abs((p[1] - x0) * vy - (p[0] - y0) * vx)
            if dist_to_spine < 60.0:
                valid_endpoints.append(ep_id)
        if len(valid_endpoints) < 2:
            valid_endpoints = endpoints
 
        # Find Farthest Point Pair
        best_start, best_end = -1, -1
        max_dist = 0.0
        for i in range(len(valid_endpoints)):
            for j in range(i+1, len(valid_endpoints)):
                p1 = id_to_pt[valid_endpoints[i]]
                p2 = id_to_pt[valid_endpoints[j]]
                d = np.sqrt((p1[1]-p2[1])**2 + (p1[0]-p2[0])**2)
                if d > max_dist:
                    max_dist = d
                    best_start = valid_endpoints[i]
                    best_end = valid_endpoints[j]
 
        if best_start == -1 or best_end == -1: return 0.0, []
 
        # Dijkstra Algorithm to find the shortest route between both ends along the shrimp spine
        dist = [float('inf')] * n_pixels
        parent = [-1] * n_pixels
        dist[best_start] = 0.0
        pq = [(0.0, best_start)]
 
        while pq:
            d, u = heapq.heappop(pq)
            if u == best_end:
                break
            if d > dist[u]:
                continue
            for n_id, weight in adj[u]:
                if not in_subgraph[n_id]: continue
                if dist[u] + weight < dist[n_id]:
                    dist[n_id] = dist[u] + weight
                    parent[n_id] = u
                    heapq.heappush(pq, (dist[n_id], n_id))
 
        # Trace back route from Parent
        path = []
        curr = best_end
        while curr != -1:
            path.append(id_to_pt[curr]) # Format (y, x)
            curr = parent[curr]
        path.reverse()
 
        return dist[best_end], path
 
    def visualize_on_image(self, original_img, raw_skeleton, path_nodes):
        vis = original_img.copy()
        y_skel, x_skel = np.where(raw_skeleton > 0)
        for y, x in zip(y_skel, x_skel):
            cv2.circle(vis, (x, y), 1, (0, 255, 0), -1) 
            
        for y, x in path_nodes:
            cv2.circle(vis, (x, y), 1, (0, 0, 255), -1) 
            
        return vis
 
if __name__ == "__main__":
    # Replace with your shrimp photo path
    IMAGE_PATH = "shtest.jpg"
    KALIBRASI_PATH = "matriks_kamera_TA.npz"
    PIXEL_PER_CM = 23.54  
    
    # Change to False if photo does not need to be straightened
    USE_UNDISTORT = True     
    # Crop black edges (True) or keep original (False)
    CROP_UNDISTORT = True    
 
    mtx, dist = None, None
    if USE_UNDISTORT:
        mtx, dist = load_calibration_data(KALIBRASI_PATH)
        
    img_cv = cv2.imread(IMAGE_PATH)
    if img_cv is None:
        # ❌ Error: Image '{IMAGE_PATH}' not found!
        print(f"❌ Error: Image '{IMAGE_PATH}' not found!")
        exit()
 
    # Execute Undistort
    if USE_UNDISTORT and mtx is not None:
        # [*] Applying Lens Correction (Undistort)...
        print("[*] Applying Lens Correction (Undistort)...")
        img_input_final = apply_undistort(img_cv, mtx, dist, crop=CROP_UNDISTORT)
    else:
        # [*] Undistort process is disabled.
        print("[*] Undistort process is disabled.")
        img_input_final = img_cv
 
    # Convert BGR (OpenCV) format to RGB for processing and Matplotlib display
    img_rgb = cv2.cvtColor(img_input_final, cv2.COLOR_BGR2RGB)
 
    # [*] Processing Binary Image...
    print("[*] Processing Binary Image...")
    skel_analyzer = SkeletonAnalyzerGuoHall()
    skel_analyzer.set_image(img_rgb)
    
    bin_img = skel_analyzer.to_binary(False) 
    raw_skeleton = skel_analyzer.compute_skeleton()
    
    # [*] Computing Graph Analysis...
    print("[*] Computing Graph Analysis...")
    graph_tool = GraphSkeletonAnalyzer(PIXEL_PER_CM)
    length_px, main_path = graph_tool.find_main_path(raw_skeleton)
   
    est_length_cm = length_px / PIXEL_PER_CM
    # ✅ Final Result: Shrimp length estimated at {est_length_cm:.2f} cm
    print(f"\n✅ Final Result: Shrimp length estimated at {est_length_cm:.2f} cm")
   
    final_vis_rgb = graph_tool.visualize_on_image(img_rgb, raw_skeleton, main_path)
   
    # Display the 4 resulting images side by side
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("1. Input Image (Original)")
    plt.axis('off')
   
    plt.subplot(1, 4, 2)
    plt.imshow(bin_img, cmap='gray')
    plt.title("2. Binary (Morphology Ellipse)")
    plt.axis('off')
 
    plt.subplot(1, 4, 3)
    plt.imshow(raw_skeleton, cmap='gray')
    plt.title("3. Raw Skeleton")
    plt.axis('off')
   
    plt.subplot(1, 4, 4)
    plt.imshow(final_vis_rgb)
    plt.title(f"4. Dijkstra Centerline ({est_length_cm:.2f} cm)")
    plt.axis('off')
   
    plt.tight_layout()
    plt.show()

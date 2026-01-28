import cv2 
import numpy as np 
import math 
import os
from shapely.geometry import Polygon, Point, box 
from shapely.affinity import translate, rotate, scale
from scipy.stats import multivariate_normal 

# ============================================================================== 
# 1. HELPER FUNCTIONS & PREPROCESSING
# ============================================================================== 
def get_clean_polygons(binary_img, min_area=300):
    """
    Trích xuất contour và tạo polygon từ ảnh nhị phân.
    Tương ứng với bước: Building segmentation & Vector Map generation.
    """
    h_img, w_img = binary_img.shape
    cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            # Loại bỏ biên ảnh để tránh nhiễu
            if x < 5 or y < 5 or (x + w) > (w_img - 5) or (y + h) > (h_img - 5):
                continue 
            # Douglas-Peucker simplification (như bài báo đề cập ở Sec 2.1)
            approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c,True), True)
            if len(approx) >= 3:
                p = Polygon(approx.reshape(-1, 2))
                if not p.is_valid: p = p.buffer(0)
                polys.append(p)
    polys.sort(key=lambda x: x.area, reverse=True)
    return polys

def auto_fix_scale(cam_polys, map_polys):
    """
    Mô phỏng bước: "heading and scale... are aligned".
    Trong thực tế, scale được tính từ độ cao máy bay. Ở đây ta tự động tính 
    để khớp dữ liệu demo.
    """
    if not cam_polys or not map_polys: return cam_polys, 1.0
    max_cam = cam_polys[0]
    best_scale = 1.0
    
    for mp in map_polys:
        current_scale = math.sqrt(mp.area / (max_cam.area + 1e-5))
        if current_scale < 0.2 or current_scale > 5.0: continue
        cam_compact = (4 * math.pi * max_cam.area) / (max_cam.length ** 2)
        map_compact = (4 * math.pi * mp.area) / (mp.length ** 2)
        if abs(cam_compact - map_compact) < 0.2: 
             best_scale = current_scale
             break
    
    scaled_polys = []
    for p in cam_polys:
        sp = scale(p, xfact=best_scale, yfact=best_scale, origin=(0,0))
        scaled_polys.append(sp)
    return scaled_polys, best_scale

# ============================================================================== 
# 2. CLASS PARTICLE FILTER (Implementing Eqs 7, 8, 9, 10, 11)
# ============================================================================== 
class PaperCompliantPF: 
    def __init__(self, N, W, H): 
        # Table 1 & Table 3: N particles
        self.N = N; self.W = W; self.H = H 
        self.particles = np.zeros((N, 2))      # Position state x_t
        self.weights = np.ones(N) / N          # Weights omega_t
        self.initialized = False 
        
        # Covariance matrix cho toàn bộ bộ lọc (dùng để tính St)
        self.global_cov = np.eye(2) * 100 
        
        # Thông số nhiễu mô hình (Model Noise)
        self.process_noise_std = 2.0  

    def init(self, x, y, initial_cov=50): 
        """
        Khởi tạo hạt xung quanh vị trí ước lượng ban đầu.
        Tương ứng Fig 1: "Particle Initialization"
        """
        self.particles[:, 0] = np.random.normal(x, initial_cov, self.N) 
        self.particles[:, 1] = np.random.normal(y, initial_cov, self.N) 
        self.initialized = True 

    def propagate(self, control_input): 
        """
        Eq. 8: Propagation by local odometry
        x_t = x_{t-1} + v_t * dt + noise
        control_input: vector di chuyển [dx, dy] từ odometry
        """
        # Thêm nhiễu Gaussian (Process noise)
        noise = np.random.normal(0, self.process_noise_std, (self.N, 2)) 
        
        # Cập nhật vị trí
        self.particles += control_input + noise 
        
        # Giới hạn trong bản đồ
        np.clip(self.particles[:, 0], 0, self.W, out=self.particles[:, 0]) 
        np.clip(self.particles[:, 1], 0, self.H, out=self.particles[:, 1]) 

    def update_with_gmm(self, gmm_data): 
        """
        Eq. 9: Measurement update model
        Trọng số được cập nhật dựa trên xác suất của GMM.
        """
        if not gmm_data: 
            # Nếu không khớp được gì, trọng số giữ nguyên hoặc giảm nhẹ (không đề cập cụ thể trong bài nên giữ nguyên)
            return 
        
        total_likelihoods = np.zeros(self.N) 
        
        # Tính tổng xác suất p(z|x) từ các thành phần GMM
        for comp in gmm_data: 
            mu = comp['mu']
            cov = comp['cov']
            alpha = comp['alpha'] # Correlation value
            
            # Sử dụng scipy để tính pdf của phân phối chuẩn nhiều chiều
            rv = multivariate_normal(mu, cov) 
            # Eq 9: Trọng số tỉ lệ với độ tương quan alpha và pdf
            total_likelihoods += alpha * rv.pdf(self.particles) 
            
        # Cập nhật trọng số: w_t = w_{t-1} * likelihood
        self.weights *= total_likelihoods 
        
        # Chuẩn hóa trọng số (Eq. 10 concept)
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights.fill(1.0 / self.N)

    def resample(self): 
        """
        Eq. 10: Resampling model
        Sử dụng Systematic Resampling để giảm hiện tượng suy biến hạt.
        """
        # Tính số lượng hạt hiệu quả (N_eff) - Tiêu chuẩn chung
        n_eff = 1.0 / np.sum(np.square(self.weights))
        
        # Thực hiện resample (Luôn thực hiện hoặc khi N_eff thấp - Code này thực hiện luôn để ổn định)
        cumsum = np.cumsum(self.weights)
        step = 1.0 / self.N
        start = np.random.uniform(0, step)
        positions = (np.arange(self.N) * step + start)
        
        new_particles = np.zeros_like(self.particles)
        indices = np.zeros(self.N, dtype=int)
        i, j = 0, 0
        while i < self.N:
            if positions[i] < cumsum[j]:
                indices[i] = j
                new_particles[i] = self.particles[j]
                i += 1
            else:
                j += 1
                
        self.particles = new_particles
        self.weights.fill(1.0 / self.N) # Reset trọng số sau khi resample

    def estimate_and_evaluate(self): 
        """
        Eq. 11: The global position estimation and credibility indicator S_t
        """
        # Ước lượng vị trí trung bình (Mean state)
        estimated_pos = np.average(self.particles, weights=self.weights, axis=0) 
        
        # Tính Hiệp phương sai của các hạt (Global Covariance)
        diff = self.particles - estimated_pos
        cov_matrix = np.dot((diff * self.weights[:, None]).T, diff)
        self.global_cov = cov_matrix
        
        # Tính toán S_t theo công thức Eq 11
        # S_t = sign( log10( (10 * sqrt(|sigma|)) / max_dist ) )
        # Lưu ý: Công thức trong ảnh khá phức tạp, đây là triển khai gần đúng nhất với logic
        
        det_cov = np.linalg.det(cov_matrix)
        sqrt_det_cov = np.sqrt(abs(det_cov)) if abs(det_cov) > 1e-9 else 1e-9
        
        # Khoảng cách lớn nhất giữa các hạt (để đo độ phân tán)
        # Trong công thức là max|mu_t - mu_j|, ở đây ta dùng max distance từ mean
        max_dist = np.max(np.linalg.norm(self.particles - estimated_pos, axis=1))
        if max_dist < 1e-3: max_dist = 1e-3
        
        # Giá trị bên trong log
        val_inside = (10 * sqrt_det_cov) / max_dist
        
        # Tính S_t
        try:
            val_log = math.log10(val_inside + 1e-9)
            S_t = 1.0 if val_log > 0 else -1.0 # Sign function
        except:
            S_t = -1.0

        return estimated_pos, S_t

# ============================================================================== 
# 3. CLASS CFBVM MATCHER (Implementing Eqs 1, 2, 3, 4)
# ============================================================================== 
class CFBVMMatcher: 
    def __init__(self, map_polys): 
        self.map_data = [] 
        # [GIỮ NGUYÊN] Radius levels chuẩn theo bài báo
        self.RADIUS_LEVELS = [20, 40, 60] 
        self.NUM_SECTORS = 8 
        self.sector_masks = self._precompute_sector_masks() 
        self.norm_factor = np.pi * (self.RADIUS_LEVELS[-1]**2)

        for poly in map_polys: 
            simple_poly = poly.simplify(0.5, preserve_topology=True)
            vec = self.compute_shape_vector(simple_poly) 
            if np.sum(vec) > 0: 
                self.map_data.append({ 
                    'poly': simple_poly, 
                    'vector': vec, 
                    'centroid': (simple_poly.centroid.x, simple_poly.centroid.y), 
                    'area': poly.area
                }) 

    # ... (Các hàm _precompute_sector_masks, compute_shape_vector, coarse_distance GIỮ NGUYÊN) ...
    def _precompute_sector_masks(self):
        # ... (Code cũ) ...
        masks = [] 
        angle_step = 360.0 / self.NUM_SECTORS 
        prev_r = 0.0 
        for r in self.RADIUS_LEVELS: 
            level_masks = [] 
            for j in range(self.NUM_SECTORS): 
                pts = [(0,0)]
                for a in np.linspace(j*angle_step, (j+1)*angle_step, 10):
                    rad = math.radians(a)
                    pts.append((r*math.cos(rad), r*math.sin(rad)))
                if prev_r > 0:
                     for a in reversed(np.linspace(j*angle_step, (j+1)*angle_step, 10)):
                        rad = math.radians(a)
                        pts.append((prev_r*math.cos(rad), prev_r*math.sin(rad)))
                level_masks.append(Polygon(pts))
            masks.append(level_masks) 
            prev_r = r 
        return masks 

    def compute_shape_vector(self, poly):
        # ... (Code cũ) ...
        cx, cy = poly.centroid.xy
        vector = [] 
        poly_centered = translate(poly, -cx, -cy) 
        for i in range(len(self.RADIUS_LEVELS)): 
            for j in range(self.NUM_SECTORS): 
                try: 
                    if poly_centered.intersects(self.sector_masks[i][j]):
                        val = poly_centered.intersection(self.sector_masks[i][j]).area 
                    else: val = 0.0
                except: val = 0.0 
                vector.append(val) 
        vec_np = np.array(vector) 
        total = np.sum(vec_np) 
        if total > 0: vec_np = vec_np / total 
        return vec_np

    def coarse_distance(self, vec_a, vec_b):
        return np.sum(np.abs(vec_a - vec_b))

    # ==========================================================================
    # [FIX] ĐÂY LÀ PHẦN SỬA LẠI CHO ĐÚNG EQ. 4
    # ==========================================================================
    def compute_correlation(self, poly_cam, poly_ref): 
        """
        Thực hiện Eq. 4: Fine matching model
        alpha = intersection(V_cam, V_ref) / sqrt( intersection(V_box, V_ref) * intersection(V_box, V_cam) )
        """
        # Đưa về cùng hệ tọa độ tâm (0,0) để so sánh hình dáng
        p1 = translate(poly_cam, -poly_cam.centroid.x, -poly_cam.centroid.y) # V_cam
        p2 = translate(poly_ref, -poly_ref.centroid.x, -poly_ref.centroid.y) # V_ref
        
        best_alpha = 0.0 
        
        # Thử xoay mỗi 15 độ
        for ang in range(0, 360, 15): 
            p1_rot = rotate(p1, ang, origin=(0,0)) 
            
            # --- TẠO V_BOX (QUAN TRỌNG) ---
            # V_box là khung hình bao quanh V_cam.
            # Trong không gian local (tại tâm 0,0), V_box chính là bounding box của p1_rot
            minx, miny, maxx, maxy = p1_rot.bounds
            v_box = box(minx, miny, maxx, maxy) 
            
            try: 
                # 1. Tử số: intersection(V_cam, V_ref)
                inter_cam_ref = p1_rot.intersection(p2).area 
                
                # 2. Mẫu số thành phần 1: intersection(V_box, V_ref)
                # (Phần bản đồ lọt vào trong khung hình camera)
                inter_box_ref = v_box.intersection(p2).area
                
                # 3. Mẫu số thành phần 2: intersection(V_box, V_cam)
                # (Phần camera lọt vào khung hình - thường là chính nó vì box bao quanh nó)
                inter_box_cam = v_box.intersection(p1_rot).area
                
                # Tính toán Eq. 4
                denominator = math.sqrt(inter_box_ref * inter_box_cam + 1e-9)
                
                if denominator > 0:
                    alpha = inter_cam_ref / denominator
                    if alpha > best_alpha: best_alpha = alpha 
            except: 
                pass 
                
            if best_alpha > 0.95: break 
            
        return best_alpha 

    def process(self, drone_poly, estimated_pos, search_radius=300): 
        # ... (Phần logic giữ nguyên như cũ) ...
        if drone_poly is None: return [] 
        candidates = [] 
        
        drone_vecs = []
        cx, cy = drone_poly.centroid.x, drone_poly.centroid.y
        poly_centered = translate(drone_poly, -cx, -cy)
        for ang in [0, 90, 180, 270]:
            p_rot = rotate(poly_centered, ang, origin=(0,0))
            p_back = translate(p_rot, cx, cy) 
            drone_vecs.append(self.compute_shape_vector(p_back))

        for item in self.map_data: 
            if estimated_pos is not None: 
                if np.linalg.norm(np.array(item['centroid']) - estimated_pos) > search_radius: continue 
            
            area_ratio = drone_poly.area / (item['area'] + 1e-5)
            # Nới lỏng ratio một chút vì Eq 4 xử lý tốt việc diện tích không khớp do bị cắt
            if area_ratio < 0.3 or area_ratio > 3.0: continue 

            min_shape_dist = 100.0
            for d_vec in drone_vecs:
                d = self.coarse_distance(d_vec, item['vector'])
                if d < min_shape_dist: min_shape_dist = d
            
            if min_shape_dist < 0.8: 
                candidates.append(item) 

        gmm_components = [] 
        cam_center = np.array([200, 200]) 
        vec_center_to_bld = np.array([drone_poly.centroid.x, drone_poly.centroid.y]) - cam_center
        
        for cand in candidates: 
            # Gọi hàm compute_correlation MỚI (Eq 4)
            alpha = self.compute_correlation(drone_poly, cand['poly']) 
            
            if alpha > 0.6: 
                map_centroid = np.array(cand['centroid']) 
                measured_pos = map_centroid - vec_center_to_bld 
                gmm_components.append({ 
                    'mu': measured_pos, 
                    'cov': np.eye(2) * 50.0, 
                    'alpha': alpha 
                }) 
        return gmm_components 

# ============================================================================== 
# 4. MAIN PROGRAM
# ============================================================================== 
def main(): 
    # --- SETUP PATHS ---
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Bạn hãy thay đường dẫn ảnh của bạn vào đây
    MAP_PATH = os.path.join(CURRENT_DIR, "ref snazzy.png")
    MASK_PATH = os.path.join(CURRENT_DIR, "/Users/phuongnhi/Documents/Drone/GPS-Denied-Large-scale-Localization-for-Aircraft-using-Semantic-Vector-Map-main/semantic segment model/processed_images/mask_goc.png") 
    
    # --- LOAD MAP ---
    map_img_color = cv2.imread(MAP_PATH)
    if map_img_color is None: return print("Lỗi Map! Kiểm tra đường dẫn.")
    map_gray = cv2.cvtColor(map_img_color, cv2.COLOR_BGR2GRAY)
    _, map_thresh = cv2.threshold(map_gray, 127, 255, cv2.THRESH_BINARY)
    H, W = map_thresh.shape 
    
    map_polys = get_clean_polygons(map_thresh, min_area=800)
    print(f"[SYSTEM] Loaded {len(map_polys)} buildings from Map.")

    # --- LOAD CAMERA MASK (INPUT) ---
    mask_input = cv2.imread(MASK_PATH, 0)
    if mask_input is None: return print("Lỗi Input Mask! Kiểm tra đường dẫn.")
    
    proc_frame = cv2.resize(mask_input, (400, 400))
    _, proc_frame = cv2.threshold(proc_frame, 127, 255, cv2.THRESH_BINARY)
    
    raw_cam_polys = get_clean_polygons(proc_frame, min_area=100)
    # Bước tiền xử lý (Preprocessing) - Không nằm trong thuật toán lõi nhưng cần để demo
    visible_polys, scale_factor = auto_fix_scale(raw_cam_polys, map_polys)

    # --- INITIALIZATION ---
    # Số lượng hạt N=2000 (như bài báo)
    pf = PaperCompliantPF(N=2000, W=W, H=H) 
    matcher = CFBVMMatcher(map_polys) 
    
    print("\n[INIT] Global Search (Initialization Stage)...")
    # Mô phỏng bước tìm vị trí ban đầu (như Fig 1: "Initialize?")
    best_pos = (W/2, H/2)
    max_score = 0
    
    if len(visible_polys) > 0:
        target = visible_polys[0] 
        for item in matcher.map_data:
            ratio = target.area / (item['area'] + 1e-5)
            if 0.8 < ratio < 1.2:
                score = matcher.compute_correlation(target, item['poly'])
                if score > max_score:
                    max_score = score
                    best_pos = item['centroid']
    
    print(f" -> Initial Position: {best_pos} (Correlation: {max_score:.2f})")
    pf.init(best_pos[0], best_pos[1], initial_cov=60)
    
    vis_map = map_img_color.copy()
    
    # --- MAIN LOOP ---
    # Giả lập Odometry (Máy bay trôi nhẹ về hướng Đông Nam)
    # Trong thực tế, giá trị này lấy từ IMU/GPS vận tốc
    simulated_control_input = np.array([0.5, 0.5]) 

    while True: 
        # 1. Propagation (Odometry update)
        pf.propagate(simulated_control_input) 
        
        # 2. Get Estimate before measurement
        est_pos, _ = pf.estimate_and_evaluate() 
        
        # 3. Matching & Measurement Update
        all_matches = [] 
        for i, d_poly in enumerate(visible_polys): 
            # Tìm kiếm cục bộ quanh vị trí ước lượng
            matches = matcher.process(d_poly, est_pos, search_radius=400) 
            all_matches.extend(matches) 
        
        # 4. Update Particle Weights (Measurement update)
        pf.update_with_gmm(all_matches) 
        
        # 5. Resampling
        pf.resample() 
        
        # 6. Final Estimation & Reliability Check
        final_pos, S_t = pf.estimate_and_evaluate() 

        # --- VISUALIZATION (GIỮ NGUYÊN GIAO DIỆN) ---
        vis = vis_map.copy()
        
        # Vẽ Particles (Màu xanh lá)
        for p in pf.particles: 
            cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1) 
        
        # Vẽ Vị trí ước lượng (Màu đỏ)
        # Nếu S_t < 0 (Không tin cậy) -> Vẽ màu vàng
        color = (0, 0, 255) if S_t > 0 else (0, 255, 255)
        status_text = "RELIABLE" if S_t > 0 else "UNSTABLE"
        
        cv2.circle(vis, (int(final_pos[0]), int(final_pos[1])), 10, color, -1) 
        cv2.circle(vis, (int(final_pos[0]), int(final_pos[1])), 14, (255, 255, 255), 2)

        # Thông tin hiển thị
        cv2.putText(vis, f"Status: {status_text} (St={S_t:.0f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis, f"Pos: ({final_pos[0]:.1f}, {final_pos[1]:.1f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Hiển thị ảnh input nhỏ ở góc
        pip = cv2.cvtColor(proc_frame, cv2.COLOR_GRAY2BGR)
        pip = cv2.resize(pip, (200, 200))
        vis[H-210:H-10, 10:210] = pip
        cv2.rectangle(vis, (10, H-210), (210, H-10), (255,0,0), 2)
        
        # Resize nếu ảnh quá lớn
        if H > 900: show_vis = cv2.resize(vis, (int(W*0.6), int(H*0.6)))
        else: show_vis = vis
        
        cv2.imshow("CFBVM Localization (Strict Paper Implementation)", show_vis) 
        if cv2.waitKey(100) == 27: break 

    cv2.destroyAllWindows() 

if __name__ == "__main__": 
    main()

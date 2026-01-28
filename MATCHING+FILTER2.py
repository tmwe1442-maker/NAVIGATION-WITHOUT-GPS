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
    """
    Class xử lý so khớp bản đồ theo hình học (Contour-based Matching).
    """
    def __init__(self, map_polys):
        self.map_data = []
        # Cấu hình Shape Context (Eq. 1, 2)
        self.RADIUS_LEVELS = [20, 40, 60] 
        self.NUM_SECTORS = 8
        self.sector_masks = self._precompute_sector_masks()
        self.norm_factor = np.pi * (self.RADIUS_LEVELS[-1]**2) 

        # Tiền xử lý bản đồ tham chiếu
        for poly in map_polys:
            simple_poly = poly.simplify(0.5, preserve_topology=True)
            vec = self.compute_shape_vector(simple_poly)
            if np.sum(vec) > 0:
                self.map_data.append({
                    'poly': simple_poly,
                    'vector': vec,
                    'centroid': (simple_poly.centroid.x, simple_poly.centroid.y),
                    'area': simple_poly.area
                })

    def _precompute_sector_masks(self):
        masks = []
        angle_step = 360.0 / self.NUM_SECTORS
        prev_r = 0.0
        for r in self.RADIUS_LEVELS:
            level = []
            for j in range(self.NUM_SECTORS):
                a1 = j * angle_step
                a2 = (j + 1) * angle_step
                level.append(self._sector(prev_r, r, a1, a2))
            masks.append(level)
            prev_r = r
        return masks

    def _sector(self, r1, r2, a1, a2):
        pts = []
        # Vẽ cung tròn chi tiết
        for a in np.linspace(math.radians(a1), math.radians(a2), 15):
            pts.append((r2*np.cos(a), r2*np.sin(a)))
        if r1 > 0:
            for a in reversed(np.linspace(math.radians(a1), math.radians(a2), 15)):
                pts.append((r1*np.cos(a), r1*np.sin(a)))
        else:
            pts.append((0,0))
        return Polygon(pts)

    def compute_shape_vector(self, poly):
        """Tính descriptor hình học (Eq. 2)"""
        cx, cy = poly.centroid.xy
        p = translate(poly, -cx[0], -cy[0])
        vec = []
        for i in range(3):
            for j in range(8):
                inter = p.intersection(self.sector_masks[i][j])
                vec.append(inter.area if not inter.is_empty else 0)
        v = np.array(vec)
        return v / self.norm_factor

    def coarse_distance(self, vec_a, vec_b):
        """So khớp thô (Coarse Matching) - Eq. 3"""
        mat_a = vec_a.reshape(3, 8)
        mat_b = vec_b.reshape(3, 8)
        min_dist = float('inf')
        
        # Rotation invariant cho vector
        for shift in range(8):
            b_shifted = np.roll(mat_b, shift, axis=1)
            dist = np.sum(np.abs(mat_a - b_shifted))
            if dist < min_dist:
                min_dist = dist
        return min_dist

    # ==========================================================================
    # PHẦN QUAN TRỌNG NHẤT: CÔNG THỨC (4) ĐÚNG CHUẨN
    # ==========================================================================
    def compute_correlation(self, cam_poly, ref_poly, img_w=400, img_h=400):
        """
        SỬA LỖI: V_box phải là khung hình Camera (Field of View), 
        không phải bounding box của tòa nhà.
        """
        # 1. Tạo V_box là hình chữ nhật kích thước ảnh, tâm trùng tâm tòa nhà
        cx, cy = cam_poly.centroid.x, cam_poly.centroid.y
        v_box = box(cx - img_w/2, cy - img_h/2, cx + img_w/2, cy + img_h/2)
        
        # 2. Dịch chuyển về gốc (0,0)
        c_ref = ref_poly.centroid
        
        # Ref đứng yên tại (0,0)
        p_ref_centered = translate(ref_poly, -c_ref.x, -c_ref.y)
        
        # Cam và Box di chuyển tương đối
        # Lưu ý: Trong thực tế ta xoay Cam đè lên Ref.
        p_cam_centered = translate(cam_poly, -cx, -cy)
        p_box_centered = translate(v_box, -cx, -cy)

        best_alpha = 0.0
        best_angle = 0.0
        
        # Tối ưu: Tính trước diện tích Box giao Cam (Thường là diện tích Cam nếu Cam nằm trọn trong ảnh)
        # Tuy nhiên cứ tính intersection cho chắc chắn
        term2 = p_box_centered.intersection(p_cam_centered).area 
        if term2 == 0: return 0.0, 0.0

        # 3. Quét góc
        for ang in np.arange(-180, 180, 5): 
            # Xoay cả Khung nhìn (Box) và Tòa nhà (Cam)
            r_cam = rotate(p_cam_centered, ang, origin=(0,0))
            r_box = rotate(p_box_centered, ang, origin=(0,0))
            
            inter_cam_ref = r_cam.intersection(p_ref_centered).area
            
            if inter_cam_ref > 0:
                # Term 1: Bản đồ Ref giao với Khung nhìn Box
                term1 = r_box.intersection(p_ref_centered).area
                
                denom = math.sqrt(term1 * term2)
                if denom > 0:
                    alpha = inter_cam_ref / denom
                    if alpha > best_alpha:
                        best_alpha = alpha
                        best_angle = ang
        
        return best_alpha, best_angle

    def process(self, cam_poly, cam_center, est_pos, search_radius=500):
        if cam_poly is None: return []

        # Simplify để tăng tốc độ tính toán
        cam_poly = cam_poly.simplify(0.5, preserve_topology=True)
        cam_vec = self.compute_shape_vector(cam_poly)
        if np.sum(cam_vec) == 0: return []

        results = []
        rel_vec_x = cam_poly.centroid.x - cam_center[0]
        rel_vec_y = cam_poly.centroid.y - cam_center[1]

        for item in self.map_data:
            # Lọc theo khoảng cách
            if est_pos is not None:
                dist = np.linalg.norm(np.array(item['centroid']) - est_pos)
                if dist > search_radius:
                    continue

            # Kiểm tra tồn tại key 'area' trước khi dùng
            if 'area' in item:
                ratio = cam_poly.area / (item['area'] + 1e-5)
                if ratio < 0.3 or ratio > 3.0: continue

            # Lọc thô bằng Shape Context
            if self.coarse_distance(cam_vec, item['vector']) > 0.35: 
                continue

            # So khớp tinh bằng Eq. 4 (Strict)
            alpha, angle = self.compute_correlation(cam_poly, item['poly'])
            
            # Ngưỡng chấp nhận (Threshold eta)
            if alpha > 0.65:
                pt_rel = Point(rel_vec_x, rel_vec_y)
                pt_rotated = rotate(pt_rel, angle, origin=(0,0))
                
                # Vị trí tuyệt đối = Vị trí Ref - Vector tương đối (đã xoay)
                mu_x = item['centroid'][0] - pt_rotated.x
                mu_y = item['centroid'][1] - pt_rotated.y
                mu = np.array([mu_x, mu_y])
                
                # Hiệp phương sai cố định (hoặc phụ thuộc alpha nếu muốn)
                cov = np.array([[100.0, 0], [0, 100.0]])
                
                results.append({'mu': mu, 'cov': cov, 'alpha': alpha})
                
        return results

# ============================================================================== 
# 4. MAIN PROGRAM (TUNED FOR 1.png - CLUSTER BUILDINGS)
# ============================================================================== 
def main(): 
    # --- SETUP PATHS ---
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MAP_PATH = os.path.join(CURRENT_DIR, "ref snazzy.png")
    MASK_PATH = os.path.join(CURRENT_DIR, "2.png")  # <-- DÙNG 1.PNG
    
    # --- LOAD MAP ---
    print("[SYSTEM] Loading Map...")
    map_img_color = cv2.imread(MAP_PATH)
    if map_img_color is None: return print("Lỗi Map! Kiểm tra đường dẫn.")
    map_gray = cv2.cvtColor(map_img_color, cv2.COLOR_BGR2GRAY)
    _, map_thresh = cv2.threshold(map_gray, 127, 255, cv2.THRESH_BINARY)
    H, W = map_thresh.shape 
    
    # Lấy contours từ map (giữ lại các tòa nhà > 300px)
    map_polys = get_clean_polygons(map_thresh, min_area=300)
    print(f"[SYSTEM] Loaded {len(map_polys)} buildings from Map.")

    # --- LOAD CAMERA MASK (INPUT 1.png) ---
    mask_input = cv2.imread(MASK_PATH, 0)
    if mask_input is None: return print("Lỗi Input Mask! Kiểm tra đường dẫn.")
    
    # [QUAN TRỌNG] Padding vẫn cần thiết vì tòa nhà trong 1.png chạm mép
    mask_input = cv2.copyMakeBorder(mask_input, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)

    # Resize nhẹ nếu ảnh quá to để xử lý nhanh hơn
    h_in, w_in = mask_input.shape
    scale_down = 0.15 if w_in > 1000 else 1.0 
    
    new_w, new_h = int(w_in * scale_down), int(h_in * scale_down)
    mask_input = cv2.resize(mask_input, (new_w, new_h))
    
    # Tâm camera
    cam_center_x = new_w // 2
    cam_center_y = new_h // 2
    print(f"[INFO] Input processed: {new_w}x{new_h}. Center: ({cam_center_x}, {cam_center_y})")

    _, proc_frame = cv2.threshold(mask_input, 127, 255, cv2.THRESH_BINARY)
    
    # Lấy polygon từ ảnh camera (lọc nhiễu < 200)
    raw_cam_polys = get_clean_polygons(proc_frame, min_area=200)
    
    if not raw_cam_polys:
        return print("[ERROR] Không tìm thấy tòa nhà nào!")

    # 2. AUTO SCALE (Điều chỉnh cho 1.png)
    print("[INFO] Auto-calculating scale...")
    
    # Với 1.png, tòa nhà lớn nhưng không phải khổng lồ. 
    # Ta so sánh với Top 150 tòa nhà lớn nhất trong map (thay vì top 30)
    # để tăng khả năng tìm đúng scale.
    top_map_polys = map_polys[:150] 
    visible_polys, detected_scale = auto_fix_scale(raw_cam_polys, top_map_polys)
    print(f"[INFO] Detected Scale Factor: {detected_scale:.3f}")

    # --- INITIALIZATION ---
    pf = PaperCompliantPF(N=4000, W=W, H=H) # Tăng số hạt vì map phức tạp hơn
    matcher = CFBVMMatcher(map_polys) 
    
    print("\n[INIT] Global Search...")
    best_pos = (W/2, H/2)
    max_score = 0
    
    # --- LOGIC KHỞI TẠO RIÊNG CHO 1.PNG ---
    if len(visible_polys) > 0:
        # Lấy tòa nhà lớn nhất trong cụm (thường là tòa chữ U hoặc tòa dài)
        target = visible_polys[0] 
        print(f"[DEBUG] Target Area: {target.area:.1f}")

        candidates = []
        for item in matcher.map_data:
            if 'area' not in item: continue
            
            # Tính tỷ lệ diện tích. 
            # Nới lỏng khoảng range (0.5 - 2.0) vì hình dạng phức tạp có thể gây sai số diện tích
            ratio = target.area / (item['area'] + 1e-5)
            if 0.5 < ratio < 2.0: 
                candidates.append(item)
        
        print(f"[INIT] Found {len(candidates)} candidates based on Area.")

        # Quét nhanh qua các ứng viên
        for item in candidates:
            score, _ = matcher.compute_correlation(target, item['poly'])
            # Threshold thấp hơn một chút (0.5) vì hình dạng tòa nhà phức tạp
            if score > max_score:
                max_score = score
                best_pos = item['centroid']
                if score > 0.8: # Nếu match rất tốt thì dừng sớm
                     print(f" -> Perfect match found at {best_pos}, Score: {score:.2f}")
                     break
        
        print(f" -> Best init candidate at {best_pos}, Score: {max_score:.2f}")
    
    # Khởi tạo hạt
    init_std = 50 if max_score > 0.6 else 400
    pf.init(best_pos[0], best_pos[1], initial_cov=init_std)
    
    vis_map = map_img_color.copy()
    simulated_control = np.array([0.0, 0.0]) 

    # --- MAIN LOOP ---
    while True: 
        pf.propagate(simulated_control) 
        est_pos, _ = pf.estimate_and_evaluate() 
        
        all_matches = [] 
        for i, d_poly in enumerate(visible_polys): 
            # Search radius trung bình (500) là đủ cho case này
            matches = matcher.process(d_poly, (cam_center_x, cam_center_y), est_pos, search_radius=500) 
            all_matches.extend(matches) 
        
        pf.update_with_gmm(all_matches) 
        pf.resample() 
        final_pos, S_t = pf.estimate_and_evaluate() 

        # --- VISUALIZATION ---
        vis = vis_map.copy()
        for p in pf.particles: 
            cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1) 
        
        # Logic hiển thị trạng thái
        if S_t > 0:
            color = (0, 0, 255)
            status_text = "LOCKED"
            # Vẽ vòng tròn dự đoán vị trí
            cv2.circle(vis, (int(final_pos[0]), int(final_pos[1])), 20, color, 3)
        else:
            color = (0, 255, 255)
            status_text = f"SEARCHING (Score: {max_score:.2f})"
        
        cv2.circle(vis, (int(final_pos[0]), int(final_pos[1])), 8, color, -1) 
        cv2.putText(vis, f"Status: {status_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Vẽ hình thu nhỏ của input góc dưới
        pip = cv2.cvtColor(proc_frame, cv2.COLOR_GRAY2BGR)
        pip_display = cv2.resize(pip, (200, 200)) 
        vis[H-210:H-10, 10:210] = pip_display
        cv2.rectangle(vis, (10, H-210), (210, H-10), (0,255,255), 2)
        
        if H > 900: show_vis = cv2.resize(vis, (int(W*0.6), int(H*0.6)))
        else: show_vis = vis
        
        cv2.imshow("CFBVM - Cluster Building Test", show_vis) 
        if cv2.waitKey(100) == 27: break 

    cv2.destroyAllWindows() 

if __name__ == "__main__": 
    main()

import cv2
import numpy as np
import math
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
from shapely.ops import unary_union

# ==============================================================================
# 1. SHAPE DESCRIPTOR (Giữ nguyên tư tưởng Eq. 1)
# ==============================================================================
class ShapeDescriptor:
    def __init__(self):
        # Bán kính các vòng tròn ngữ cảnh (r1, r2, r3)
        self.radii = [30, 60, 90] 
        self.sectors = 8
     
    def compute(self, target_poly, context_union):
        """
        Tính toán vector đặc trưng dựa trên giao diện tích của các vật thể xung quanh
        với các rẻ quạt (sectors) và vành khuyên (rings).
        """
        cx, cy = target_poly.centroid.x, target_poly.centroid.y
        center = Point(cx, cy)
        vector = []
        
        prev_r = 0
        for r in self.radii:
            for j in range(self.sectors):
                angle_start = j * (360.0 / self.sectors)
                angle_end = (j + 1) * (360.0 / self.sectors)
                
                # Tạo hình rẻ quạt (Wedge)
                W = r * 2.5 
                rad_s, rad_e = math.radians(angle_start), math.radians(angle_end)
                wedge_poly = Polygon([
                    (cx, cy),
                    (cx + W*math.cos(rad_s), cy + W*math.sin(rad_s)),
                    (cx + W*math.cos(rad_e), cy + W*math.sin(rad_e))
                ])
                
                # Cắt lấy phần nằm trong Ring
                ring = center.buffer(r).difference(center.buffer(prev_r))
                sector_area = ring.intersection(wedge_poly)
                
                # Tính diện tích giao với ngữ cảnh (Context)
                try:
                    val = sector_area.intersection(context_union).area
                except:
                    val = 0.0
                vector.append(val)
            prev_r = r
        return np.array(vector)

# ==============================================================================
# 2. MATCHING SYSTEM (So khớp Coarse & Fine - Eq. 2, 4, 6)
# ==============================================================================
class StrictMatcher:
    def __init__(self):
        self.desc_engine = ShapeDescriptor()
        
    def find_matches(self, drone_polys, map_polys, map_descriptors, view_center_offset):
        """
        Input: Polygon từ Camera và Map.
        Output: List các 'observations' (vị trí ước lượng, độ tin cậy).
        """
        observations = []
        if not drone_polys: return []
        
        # Tạo ngữ cảnh (Context) cho view hiện tại
        drone_context = unary_union(drone_polys)
        
        for d_poly in drone_polys:
            # Tính descriptor cho vật thể đang xét
            f_cam = self.desc_engine.compute(d_poly, drone_context)
            
            # --- Step 1: Coarse Matching (Eq. 2) ---
            # So sánh khoảng cách vector Manhattan
            candidates = []
            for i, m_poly in enumerate(map_polys):
                dist = np.sum(np.abs(f_cam - map_descriptors[i]))
                candidates.append((dist, m_poly))
            
            # Lấy Top 5 ứng viên tốt nhất
            candidates.sort(key=lambda x: x[0])
            top_candidates = [x[1] for x in candidates[:5]]
            
            # --- Step 2: Fine Matching (Eq. 4) ---
            # Geometric Verification
            for m_poly in top_candidates:
                # Dịch chuyển d_poly về vị trí của m_poly để so sánh hình dáng
                dx = m_poly.centroid.x - d_poly.centroid.x
                dy = m_poly.centroid.y - d_poly.centroid.y
                aligned_cam = translate(d_poly, xoff=dx, yoff=dy)
                
                # Tính độ tương quan diện tích (Correlation Score)
                inter_area = aligned_cam.intersection(m_poly).area
                denom = math.sqrt(d_poly.area * m_poly.area) + 1e-6
                alpha = inter_area / denom 
                
                # Ngưỡng chấp nhận (Empirical Parameter)
                if alpha > 0.65: 
                    # Tính vị trí Drone: Pos_Global = Obj_Global - (Obj_Local - Center_Cam)
                    vec_rel_x = d_poly.centroid.x - view_center_offset[0]
                    vec_rel_y = d_poly.centroid.y - view_center_offset[1]
                    
                    est_global_x = m_poly.centroid.x - vec_rel_x
                    est_global_y = m_poly.centroid.y - vec_rel_y
                    
                    # Tính phương sai (Sigma) dựa trên độ khớp (Eq. 5 implied)
                    # Score càng cao -> Sigma càng nhỏ (tin cậy cao)
                    sigma = 30.0 * (1.0 - alpha) + 5.0
                    
                    observations.append({
                        'mu': np.array([est_global_x, est_global_y]),
                        'sigma': sigma,
                        'score': alpha
                    })
                    
        return observations

# ==============================================================================
# 3. PARTICLE FILTER (Fusion & Re-distribution - Đã tinh chỉnh)
# ==============================================================================
class FusionParticleFilter:
    def __init__(self, N, W, H):
        self.N = N
        self.W, self.H = W, H
        self.particles_pos = np.zeros((N, 2))
        self.particles_cov = np.ones(N) * 20.0 
        self.weights = np.ones(N) / N
        self.initialized = False
        
        # Biến để làm mượt (Smoothing filter)
        self.last_estimate = None

    def init(self, x, y):
        self.particles_pos[:, 0] = np.random.normal(x, 100, self.N)
        self.particles_pos[:, 1] = np.random.normal(y, 100, self.N)
        self.particles_cov[:] = 20.0
        self.weights[:] = 1.0 / self.N
        self.initialized = True
        self.last_estimate = np.array([x, y])

    def propagate(self, velocity):
        # --- Eq. 8: Prediction ---
        noise = np.random.normal(0, 2.0, (self.N, 2))
        self.particles_pos += velocity + noise
        
        # Tăng Uncertainty theo thời gian (nhưng kẹp lại để ổn định)
        self.particles_cov += 0.5
        np.clip(self.particles_cov, 5.0, 40.0, out=self.particles_cov)

        # Giới hạn trong bản đồ
        np.clip(self.particles_pos[:, 0], 0, self.W, out=self.particles_pos[:, 0])
        np.clip(self.particles_pos[:, 1], 0, self.H, out=self.particles_pos[:, 1])

    def fusion_update(self, observations):
        """
        --- Eq. 9: Data Fusion Update ---
        Cập nhật vị trí từng hạt dựa trên quan sát (Observation).
        """
        if not self.initialized or not observations: return

        # Lấy quan sát tốt nhất
        observations.sort(key=lambda x: x['score'], reverse=True)
        best_obs = observations[0]

        # Nếu độ tin cậy quá thấp, bỏ qua bước update vị trí (để tránh nhiễu)
        if best_obs['score'] < 0.4: return

        # Reset weights
        self.weights.fill(1.e-20)

        for i in range(self.N):
            p_pos = self.particles_pos[i]
            p_cov = self.particles_cov[i]
            
            dist = np.linalg.norm(p_pos - best_obs['mu'])
            
            # Gating: Chỉ update hạt nằm gần vùng quan sát (4 sigma)
            combined_sigma = math.sqrt(p_cov) + best_obs['sigma']
            
            if dist < 4 * combined_sigma:
                # --- Kalman Fusion ---
                # R: Measurement Noise. 
                R = (best_obs['sigma'] * 1.5)**2  
                P = p_cov**2
                
                K = P / (P + R) # Kalman Gain
                
                # Cập nhật vị trí: Hạt bị hút về phía quan sát
                innovation = best_obs['mu'] - p_pos
                self.particles_pos[i] += K * innovation
                
                # Cập nhật Covariance
                P_new = (1.0 - K) * P
                self.particles_cov[i] = math.sqrt(P_new)
                
                # Tính lại Weight
                lik = math.exp(-0.5 * (dist**2) / (R + 1e-6)) * best_obs['score']
                self.weights[i] = lik + 1.e-20
            else:
                self.weights[i] = 1.e-20

        # Chuẩn hóa weights
        s = np.sum(self.weights)
        if s > 0: self.weights /= s
        else: self.weights[:] = 1.0 / self.N

    def resample_and_redistribute(self, observations):
        """
        --- Eq. 10: Resampling & Re-distribution ---
        Tái phân bố hạt khi tìm thấy đặc trưng khớp tốt.
        """
        if not self.initialized: return
        
        # 1. Standard Resampling (Low Variance)
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        step = 1.0 / self.N
        r = np.random.uniform(0, step)
        
        indexes = []
        idx = 0
        for i in range(self.N):
            val = r + i*step
            while val > cumulative_sum[idx]:
                idx = min(idx + 1, self.N - 1)
            indexes.append(idx)
            
        self.particles_pos = self.particles_pos[indexes]
        self.particles_cov = self.particles_cov[indexes]
        self.weights[:] = 1.0 / self.N
        
        # 2. Re-distribution (Tái phân bố)
        if observations:
            best_obs = max(observations, key=lambda x: x['score'])
            
            # Chỉ tái phân bố khi Score rất cao (> 0.8)
            if best_obs['score'] > 0.80:
                # Chọn 5% số hạt để dời đi
                num_redist = int(self.N * 0.05) 
                idxs = np.random.choice(self.N, num_redist, replace=False)
                
                # Cưỡng ép dời các hạt này về vị trí khớp + nhiễu nhỏ
                self.particles_pos[idxs] = best_obs['mu'] + np.random.normal(0, 5, (num_redist, 2))
                self.particles_cov[idxs] = 10.0 # Reset covariance thấp

    def estimate(self):
        """
        Tính toán vị trí cuối cùng (Dùng Median và Smoothing)
        """
        if not self.initialized: return 0,0
        
        # Dùng Median (Trung vị) để loại bỏ nhiễu ngoại lai
        est_x = np.median(self.particles_pos[:, 0])
        est_y = np.median(self.particles_pos[:, 1])
        
        current_est = np.array([est_x, est_y])
        
        # Low Pass Filter (Làm mượt chuyển động)
        if self.last_estimate is None:
            self.last_estimate = current_est
        else:
            alpha = 0.3 
            self.last_estimate = (1 - alpha) * self.last_estimate + alpha * current_est
            
        return self.last_estimate[0], self.last_estimate[1]

# ==============================================================================
# 4. MAP & UTILS
# ==============================================================================
def create_city_map(W, H):
    img = np.zeros((H, W), dtype=np.uint8)
    np.random.seed(42) # Cố định map
    
    # Vẽ các tòa nhà ngẫu nhiên
    for i in range(25):
        x, y = np.random.randint(50, W-100), np.random.randint(50, H-100)
        w, h = np.random.randint(50, 100), np.random.randint(50, 100)
        cv2.rectangle(img, (x, y), (x+w, y+h), 255, -1)
        
        # Thêm chi tiết phụ để tạo shape phức tạp
        if np.random.rand() > 0.5:
             cv2.rectangle(img, (x+w//2, y+h//2), (x+w+20, y+h+20), 255, -1)
             
    # Vẽ Landmark đặc biệt (Chữ thập)
    cx, cy = W//2, H//2
    cv2.rectangle(img, (cx-20, cy-80), (cx+20, cy+80), 255, -1)
    cv2.rectangle(img, (cx-80, cy-20), (cx+80, cy+20), 255, -1)
    return img

def get_polygons_from_image(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) > 50:
            approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
            if len(approx) >= 3:
                p = Polygon(approx.reshape(-1, 2))
                if not p.is_valid: p = p.buffer(0) # Fix invalid polygon
                polys.append(p)
    return polys

# ==============================================================================
# 5. MAIN
# ==============================================================================
def main():
    W, H = 1200, 800
    VIEW_SIZE = 250 # Kích thước vùng nhìn camera
    
    # 1. Setup Map
    map_img = create_city_map(W, H)
    map_polys = get_polygons_from_image(map_img)
    map_context = unary_union(map_polys)
    
    # 2. Setup System
    matcher = StrictMatcher()
    # Pre-compute Map Descriptors
    print("Computing Map Descriptors...")
    map_descriptors = [matcher.desc_engine.compute(p, map_context) for p in map_polys]
    
    # Tăng số lượng hạt lên 500 để mượt hơn
    pf = FusionParticleFilter(N=5000, W=W, H=H)
    
    # Drone Pose & Velocity
    drone_pos = np.array([300.0, 200.0]) 
    velocity = np.array([3.0, 1.5]) 
    
    # Init hạt lệch vị trí để test khả năng hội tụ
    pf.init(drone_pos[0] + 50, drone_pos[1] - 50) 

    print("Running Simulation. Press 'ESC' to exit.")

    while True:
        # --- A. ENVIRONMENT SIMULATION ---
        drone_pos += velocity
        # Bounce logic
        if drone_pos[0] < VIEW_SIZE or drone_pos[0] > W-VIEW_SIZE: velocity[0] *= -1
        if drone_pos[1] < VIEW_SIZE or drone_pos[1] > H-VIEW_SIZE: velocity[1] *= -1
        
        # Camera Capture (Crop & Warp)
        M = np.float32([[1, 0, -drone_pos[0] + VIEW_SIZE/2], [0, 1, -drone_pos[1] + VIEW_SIZE/2]])
        cam_view = cv2.warpAffine(map_img, M, (VIEW_SIZE, VIEW_SIZE))
        drone_polys = get_polygons_from_image(cam_view)
        
        # --- B. ALGORITHM EXECUTION ---
        
        # 1. Propagate (Dự đoán chuyển động)
        pf.propagate(velocity)
        
        # 2. Matching (So khớp ngữ cảnh)
        matches = matcher.find_matches(drone_polys, map_polys, map_descriptors, 
                                     view_center_offset=(VIEW_SIZE/2, VIEW_SIZE/2))
        
        # 3. Fusion Update (Eq. 9) - Cập nhật vị trí hạt
        if matches:
            pf.fusion_update(matches)
            
        # 4. Resample & Re-distribution (Eq. 10) - Tái phân bố hạt
        pf.resample_and_redistribute(matches)
        
        # 5. Estimate Result
        est_x, est_y = pf.estimate()
        
        # --- C. VISUALIZATION ---
        vis = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
        
        # Draw Drone
        top_left = (int(drone_pos[0]-VIEW_SIZE/2), int(drone_pos[1]-VIEW_SIZE/2))
        cv2.rectangle(vis, top_left, (top_left[0]+VIEW_SIZE, top_left[1]+VIEW_SIZE), (0, 165, 255), 2)
        cv2.putText(vis, "DRONE", (top_left[0], top_left[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Draw Particles (vẽ ít thôi cho đỡ rối)
        for p in pf.particles_pos[::2]:
            cv2.circle(vis, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
            
        # Draw Estimate
        cv2.circle(vis, (int(est_x), int(est_y)), 8, (0, 0, 255), -1)
        cv2.putText(vis, "ESTIMATE", (int(est_x)+10, int(est_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # UI Info
        best_score = max([m['score'] for m in matches]) if matches else 0.0
        cv2.putText(vis, f"Match Score: {best_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # PiP View (Góc dưới trái)
        pip = cv2.cvtColor(cv2.resize(cam_view, (200, 200)), cv2.COLOR_GRAY2BGR)
        vis[H-210:H-10, 10:210] = pip
        cv2.rectangle(vis, (10, H-210), (210, H-10), (0, 255, 255), 2)

        cv2.imshow("CFBVM Simulation (Fusion + Redistribution)", vis)
        if cv2.waitKey(20) == 27: break # ESC to quit

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

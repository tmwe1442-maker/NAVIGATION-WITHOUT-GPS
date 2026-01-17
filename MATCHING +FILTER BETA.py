import cv2
import numpy as np
import math
from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate

# ==============================================================================
# 1. CLASS PARTICLE FILTER (GIỮ NGUYÊN CODE CỦA BẠN)
# ==============================================================================
class PaperCompliantPF:
    def __init__(self, N, W, H):
        self.N = N  # Số lượng hạt
        self.W = W  # Chiều rộng Map
        self.H = H  # Chiều cao Map
        
        # Eq. 7: Khởi tạo hạt
        self.particles = np.zeros((N, 2)) 
        self.weights = np.ones(N) / N
        self.initialized = False
        
        # Process Noise
        self.process_noise = 2.0 

    def init(self, x, y):
        self.particles[:, 0] = np.random.normal(x, 50, self.N)
        self.particles[:, 1] = np.random.normal(y, 50, self.N)
        self.initialized = True

    # Eq. 8: Propagation
    def propagate(self, velocity):
        noise = np.random.normal(0, self.process_noise, (self.N, 2))
        self.particles += velocity + noise
        np.clip(self.particles[:, 0], 0, self.W, out=self.particles[:, 0])
        np.clip(self.particles[:, 1], 0, self.H, out=self.particles[:, 1])

    # Eq. 9: Measurement Update
    def update(self, measured_pos, correlation_score):
        # [NGƯỠNG BÀI BÁO] Chỉ cập nhật nếu độ tin cậy rất cao (> 0.65)
        if correlation_score < 0.65: 
            return 

        dist = np.linalg.norm(self.particles - measured_pos, axis=1)
        
        # Eq. 5 & 6: Variance tỷ lệ nghịch với correlation
        R = 50.0 * (1.0 - correlation_score) 
        if R < 5.0: R = 5.0 

        likelihood = np.exp(- (dist**2) / (2 * R**2))
        
        self.weights *= likelihood
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    # Eq. 10: Resampling (Chuẩn, không Injection)
    def resample(self):
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.N * 0.5:
            indices = np.random.choice(self.N, self.N, p=self.weights)
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.N)
            self.particles += np.random.normal(0, 1.5, (self.N, 2))

    # Eq. 11: Credibility
    def estimate(self):
        mean_pos = np.average(self.particles, weights=self.weights, axis=0)
        cov = np.cov(self.particles.T, aweights=self.weights)
        det_cov = np.linalg.det(cov)
        is_credible = det_cov < 2500 
        return mean_pos, is_credible, det_cov

# ==============================================================================
# 2. CLASS CFBVM MATCHER (NÂNG CẤP CHUẨN PAPER EQ.1 - 24 CHIỀU)
# ==============================================================================
class CFBVMMatcher:
    def __init__(self, map_polys):
        self.map_data = [] 
        self.RADIUS_LEVELS = [30, 60, 90] 
        self.NUM_SECTORS = 8  # NÂNG CẤP: Chia thành 8 hướng (mỗi hướng 45 độ)
        
        print(f"Đang tiền xử lý Vectơ Hình Dạng (24 chiều) cho {len(map_polys)} tòa nhà...")
        
        # Tối ưu hóa: Pre-compute các sector mask tại gốc (0,0)
        self.sector_masks = self._precompute_sector_masks()

        for poly in map_polys:
            vec = self.compute_shape_vector(poly)
            # Chỉ thêm vào DB nếu vector có dữ liệu
            if np.sum(vec) > 0:
                self.map_data.append({
                    'poly': poly,
                    'vector': vec,
                    'centroid': (poly.centroid.x, poly.centroid.y),
                    'area': poly.area
                })

    def _precompute_sector_masks(self):
        """Tạo trước các mask hình rẻ quạt để không phải tính lại mỗi lần gọi hàm"""
        masks = []
        angle_step = 360.0 / self.NUM_SECTORS
        prev_r = 0.0
        
        for r in self.RADIUS_LEVELS:
            level_masks = []
            for j in range(self.NUM_SECTORS):
                start_angle = j * angle_step
                end_angle = (j + 1) * angle_step
                poly_mask = self._create_sector_poly(0, 0, prev_r, r, start_angle, end_angle)
                level_masks.append(poly_mask)
            masks.append(level_masks)
            prev_r = r
        return masks

    def _create_sector_poly(self, cx, cy, r_in, r_out, start_deg, end_deg):
        """Hàm hỗ trợ tạo polygon hình rẻ quạt"""
        res = 10 
        a1 = math.radians(start_deg)
        a2 = math.radians(end_deg)
        points = []
        # Cung ngoài
        angles = np.linspace(a1, a2, res)
        for a in angles:
            points.append((cx + r_out * math.cos(a), cy + r_out * math.sin(a)))
        # Cung trong
        if r_in > 0:
            for a in reversed(angles):
                points.append((cx + r_in * math.cos(a), cy + r_in * math.sin(a)))
        else:
            points.append((cx, cy))
        return Polygon(points)

    # Eq. 1: Building Shape Vector (Nâng cấp Logic)
    def compute_shape_vector(self, poly):
        cx, cy = poly.centroid.x, poly.centroid.y
        vector = []
        
        # Dịch chuyển polygon về gốc tọa độ để khớp với mask đã pre-compute
        # Cách này nhanh hơn 10x so với việc tạo mask mới
        poly_centered = translate(poly, -cx, -cy)
        
        for i in range(len(self.RADIUS_LEVELS)): # 3 Levels
            for j in range(self.NUM_SECTORS):    # 8 Sectors
                mask = self.sector_masks[i][j]
                try:
                    # Kiểm tra nhanh bounding box để tránh tính toán intersection nặng
                    if not poly_centered.bounds[0] > mask.bounds[2] and \
                       not poly_centered.bounds[2] < mask.bounds[0] and \
                       not poly_centered.bounds[1] > mask.bounds[3] and \
                       not poly_centered.bounds[3] < mask.bounds[1]:
                        
                        val = poly_centered.intersection(mask).area
                    else:
                        val = 0.0
                except:
                    val = 0.0
                vector.append(val)
        
        vec_np = np.array(vector)
        # Chuẩn hóa vector
        total = np.sum(vec_np)
        if total > 0: vec_np = vec_np / total
        return vec_np

    # Eq. 2: Coarse Distance
    def coarse_distance(self, vec_a, vec_b):
        return np.sum(np.abs(vec_a - vec_b))

    # Eq. 4: Fine Correlation (Giữ nguyên)
    def compute_correlation(self, poly_cam, poly_ref):
        p1 = translate(poly_cam, -poly_cam.centroid.x, -poly_cam.centroid.y)
        p2 = translate(poly_ref, -poly_ref.centroid.x, -poly_ref.centroid.y)
        
        best_alpha = 0.0
        for ang in [-5, 0, 5]: 
            p1_rot = rotate(p1, ang, origin=(0,0))
            try:
                inter_area = p1_rot.intersection(p2).area
                if inter_area > 0:
                    denom = math.sqrt(p1_rot.area * p2.area)
                    alpha = inter_area / denom
                    if alpha > best_alpha: best_alpha = alpha
            except: pass
        return best_alpha

    def process(self, drone_poly, estimated_pos):
        if drone_poly is None: return None, 0.0
        
        drone_vec = self.compute_shape_vector(drone_poly)
        if np.sum(drone_vec) == 0: return None, 0.0

        search_radius = 500.0 
        
        # --- Giai đoạn 1: COARSE MATCHING ---
        candidates = []
        for item in self.map_data:
            dx = item['centroid'][0] - estimated_pos[0]
            dy = item['centroid'][1] - estimated_pos[1]
            if math.hypot(dx, dy) > search_radius: continue
            
            dist = self.coarse_distance(drone_vec, item['vector'])
            
            # [Adjusted Threshold] Với vector 24 chiều, ngưỡng 0.5 là an toàn
            if dist < 0.5: 
                candidates.append(item)

        if not candidates: return None, 0.0

        # --- Giai đoạn 2: FINE MATCHING ---
        best_match_poly = None
        best_score = 0.0
        
        for cand in candidates:
            score = self.compute_correlation(drone_poly, cand['poly'])
            if score > best_score:
                best_score = score
                best_match_poly = cand['poly']
                
        # [NGƯỠNG BÀI BÁO] Chấp nhận kết quả cuối cùng nếu độ khớp > 0.6
        if best_match_poly and best_score > 0.6: 
            vec_cam_to_bld = np.array([drone_poly.centroid.x - 200, drone_poly.centroid.y - 200])
            map_centroid = np.array([best_match_poly.centroid.x, best_match_poly.centroid.y])
            measured_pos = map_centroid - vec_cam_to_bld
            return measured_pos, best_score
            
        return None, 0.0

# ==============================================================================
# 3. MÔI TRƯỜNG GIẢ LẬP & MAIN (GIỮ NGUYÊN CODE CỦA BẠN)
# ==============================================================================
def create_complex_map(W, H):
    img = np.zeros((H, W), dtype=np.uint8)
    np.random.seed(99) 
    for _ in range(350):
        x = np.random.randint(50, W-300)
        y = np.random.randint(50, H-300)
        w, h = np.random.randint(120, 200), np.random.randint(120, 200)
        cv2.rectangle(img, (x, y), (x+w, y+h), 255, -1)
        if np.random.rand() > 0.5:
            cut_w, cut_h = int(w*0.6), int(h*0.6)
            cv2.rectangle(img, (x+w-cut_w, y), (x+w, y+cut_h), 0, -1)
    return img

def main():
    W, H = 5000, 4000
    print("Đang khởi tạo bản đồ phức tạp...")
    map_img = create_complex_map(W, H)
    
    _, thresh = cv2.threshold(map_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    map_polys = []
    for c in contours:
        if cv2.contourArea(c) > 800:
            p = Polygon(c.reshape(-1, 2))
            if not p.is_valid: p = p.buffer(0)
            map_polys.append(p)

    # Tăng số lượng hạt để bù đắp cho việc threshold quá chặt
    pf = PaperCompliantPF(N=3000, W=W, H=H)
    matcher = CFBVMMatcher(map_polys) 
    
    real_pos = np.array([1000.0, 1000.0])
    velocity = np.array([4.0, 3.0]) 
    
    pf.init(real_pos[0], real_pos[1])
    
    DISPLAY_SCALE = 0.15
    h_d, w_d = int(H*DISPLAY_SCALE), int(W*DISPLAY_SCALE)
    bg_static = cv2.resize(map_img, (w_d, h_d), interpolation=cv2.INTER_NEAREST)
    bg_static = cv2.cvtColor(bg_static, cv2.COLOR_GRAY2BGR)
    
    print("--- BẮT ĐẦU: NGƯỠNG CHẶT (PAPER) + MOTION GATING + 24-DIM VECTOR ---")

    while True:
        # 1. Vật lý Drone
        real_pos += velocity
        if real_pos[0] < 300 or real_pos[0] > W-300: velocity[0] *= -1
        if real_pos[1] < 300 or real_pos[1] > H-300: velocity[1] *= -1
        
        # 2. Camera View
        M = np.float32([[1, 0, -real_pos[0] + 200], [0, 1, -real_pos[1] + 200]])
        cam_view = cv2.warpAffine(map_img, M, (400, 400))
        
        drone_poly = None
        gray = cam_view
        if len(cam_view.shape)==3: gray=cv2.cvtColor(cam_view, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c,True), True)
                if len(approx) >= 3:
                    drone_poly = Polygon(approx.reshape(-1,2))
                    if not drone_poly.is_valid: drone_poly = drone_poly.buffer(0)

        # 3. LOGIC THUẬT TOÁN
        
        # B1: Propagate & Lấy dự đoán t
        pf.propagate(velocity)
        pred_pos, _, _ = pf.estimate()
        
        # B2: Matching (Với threshold chặt)
        measured_pos, score = matcher.process(drone_poly, pred_pos)
        
        # B3: Motion Gating (So sánh t và t-1)
        is_drift = False
        if measured_pos is not None:
            jump_dist = np.linalg.norm(measured_pos - pred_pos)
            
            # Ngưỡng nhảy cho phép: 10 lần vận tốc + 50px an toàn
            max_jump_allowed = np.linalg.norm(velocity) * 10 + 50.0
            
            if jump_dist < max_jump_allowed:
                pf.update(measured_pos, score) # Chỉ update nếu trong vùng an toàn
                pf.resample()
            else:
                is_drift = True # Bỏ qua do nhảy quá xa (Drift)
        
        # B4: Kết quả
        est_pos, is_credible, det_cov = pf.estimate()

        # 4. Hiển thị
        vis = bg_static.copy()
        def to_s(x,y): return int(x*DISPLAY_SCALE), int(y*DISPLAY_SCALE)
        
        for p in pf.particles:
            cv2.circle(vis, to_s(p[0], p[1]), 1, (0, 255, 0), -1)
            
        rx, ry = to_s(real_pos[0], real_pos[1])
        cv2.rectangle(vis, (rx-10, ry-10), (rx+10, ry+10), (255, 100, 0), 2)
        
        ex, ey = to_s(est_pos[0], est_pos[1])
        color_est = (0, 0, 255) if is_credible else (0, 165, 255)
        
        status_text = "LOCKED" if is_credible else "SEARCHING"
        if is_drift: 
            status_text = "DRIFT IGNORED"
            color_est = (0, 255, 255)
            
        cv2.circle(vis, (ex, ey), 3, color_est, -1 if is_credible else 2)
        cv2.line(vis, (rx, ry), (ex, ey), (0, 255, 255), 1)

        cv2.putText(vis, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_est, 2)
        cv2.putText(vis, f"Score: {score:.3f} (>0.6)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(vis, f"Method: 24-Dim Vector", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        pip_h, pip_w = 150, 150
        pip = cv2.cvtColor(cv2.resize(cam_view, (pip_w, pip_h)), cv2.COLOR_GRAY2BGR)
        vis[h_d-pip_h-10 : h_d-10, 10 : 10+pip_w] = pip
        cv2.rectangle(vis, (10, h_d-pip_h-10), (10+pip_w, h_d-10), (0, 255, 255), 2)

        cv2.imshow("CFBVM Paper Strict Implementation", vis)
        if cv2.waitKey(20) == 27: break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

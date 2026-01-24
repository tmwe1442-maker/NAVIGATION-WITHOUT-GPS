import cv2
import numpy as np
import math
from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate
from scipy.stats import multivariate_normal

from MAPPING import DroneController

# ==============================================================================
# 1. CLASS PARTICLE FILTER (ĐÃ BỔ SUNG HÀM BỊ THIẾU)
# ==============================================================================
class PaperCompliantPF:
    def __init__(self, N, W, H):
        self.N = N
        self.W = W
        self.H = H
        self.particles = np.zeros((N, 2)) 
        self.weights = np.ones(N) / N
        self.counts = np.zeros(N, dtype=int)
        self.initialized = False
        self.process_noise = 2.0 

    def init(self, x, y):
        self.particles[:, 0] = np.random.normal(x, 50, self.N)
        self.particles[:, 1] = np.random.normal(y, 50, self.N)
        self.counts.fill(0)
        self.initialized = True

    def propagate(self, velocity):
        # Chỉ giữ lại nhiễu cơ bản
        noise = np.random.normal(0, self.process_noise, (self.N, 2))
        self.particles += velocity + noise
        np.clip(self.particles[:, 0], 0, self.W, out=self.particles[:, 0])
        np.clip(self.particles[:, 1], 0, self.H, out=self.particles[:, 1])

    # --- ĐÂY LÀ HÀM BỊ THIẾU, BẠN CẦN BỔ SUNG VÀO ---
    def estimate_and_evaluate(self):
        # Tính vị trí trung bình có trọng số (Weighted Mean)
        estimated_pos = np.average(self.particles, weights=self.weights, axis=0)
        
        # Tính độ tin cậy S_t theo bài báo (Eq. 11)
        n = 3 # Ngưỡng số lần match
        valid_indices = self.counts > n
        
        if np.sum(valid_indices) == 0:
            S_t = -1.0 # Không tin cậy
        else:
            sum_w_valid = np.sum(self.weights[valid_indices])
            sum_w_invalid = np.sum(self.weights[~valid_indices]) + 1.e-10
            ratio = sum_w_valid / sum_w_invalid
            S_t = 1.0 if ratio > 1.0 else -1.0
        
        is_credible = (S_t > 0)
        return estimated_pos, is_credible, S_t
    # -----------------------------------------------

    def update_with_gmm(self, gmm_data):
        if not gmm_data:
            self.counts = np.maximum(0, self.counts - 1)
            return

        current_mean = np.average(self.particles, weights=self.weights, axis=0)
        valid_components = []
        
        for comp in gmm_data:
            dist = np.linalg.norm(comp['mu'] - current_mean)
            # CHỈ CHẤP NHẬN NẾU < 300 (3 mét)
            if dist < 300.0:
                valid_components.append(comp)

        if not valid_components:
            self.counts = np.maximum(0, self.counts - 1)
            return

        total_likelihoods = np.zeros(self.N)
        for comp in valid_components:
            mu = comp['mu']; cov = comp['cov']; alpha = comp['alpha']
            rv = multivariate_normal(mu, cov)
            pdf_vals = rv.pdf(self.particles)
            total_likelihoods += alpha * pdf_vals
            
            std_x = np.sqrt(cov[0,0]); std_y = np.sqrt(cov[1,1])
            delta = self.particles - mu
            dist_sq = (delta[:,0]/std_x)**2 + (delta[:,1]/std_y)**2
            matched_indices = dist_sq < 9.0
            self.counts[matched_indices] += 1
            
        self.weights *= total_likelihoods
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    def resample(self, gmm_data=None):
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.N * 0.5:
            indices = np.random.choice(self.N, self.N, p=self.weights)
            self.particles = self.particles[indices]
            self.counts = self.counts[indices]
            self.weights.fill(1.0 / self.N)

# ==============================================================================
# 2. CLASS CFBVM MATCHER (GIỮ NGUYÊN NHƯNG SỬA LOGIC PROCESS)
# ==============================================================================
class CFBVMMatcher:
    def __init__(self, map_polys):
        self.map_data = [] 
        self.RADIUS_LEVELS = [30, 60, 90] 
        self.NUM_SECTORS = 8 
        self.sector_masks = self._precompute_sector_masks()

        print("Đang tiền xử lý Map Data...")
        for poly in map_polys:
            vec = self.compute_shape_vector(poly)
            if np.sum(vec) > 0:
                self.map_data.append({
                    'poly': poly,
                    'vector': vec,
                    'centroid': (poly.centroid.x, poly.centroid.y),
                    'bounds': poly.bounds
                })

    def _precompute_sector_masks(self):
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
        res = 10 
        a1 = math.radians(start_deg); a2 = math.radians(end_deg)
        points = []
        angles = np.linspace(a1, a2, res)
        for a in angles: points.append((cx + r_out * math.cos(a), cy + r_out * math.sin(a)))
        if r_in > 0:
            for a in reversed(angles): points.append((cx + r_in * math.cos(a), cy + r_in * math.sin(a)))
        else: points.append((cx, cy))
        return Polygon(points)

    def compute_shape_vector(self, poly):
        cx, cy = poly.centroid.x, poly.centroid.y
        vector = []
        poly_centered = translate(poly, -cx, -cy)
        for i in range(len(self.RADIUS_LEVELS)):
            for j in range(self.NUM_SECTORS):
                mask = self.sector_masks[i][j]
                try:
                    if not poly_centered.bounds[0] > mask.bounds[2] and \
                       not poly_centered.bounds[2] < mask.bounds[0] and \
                       not poly_centered.bounds[1] > mask.bounds[3] and \
                       not poly_centered.bounds[3] < mask.bounds[1]:
                        val = poly_centered.intersection(mask).area
                    else: val = 0.0
                except: val = 0.0
                vector.append(val)
        vec_np = np.array(vector)
        total = np.sum(vec_np)
        if total > 0: vec_np = vec_np / total
        return vec_np

    def coarse_distance(self, vec_a, vec_b):
        return np.sum(np.abs(vec_a - vec_b))

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

    # HÀM PROCESS ĐÃ ĐƯỢC SIẾT CHẶT
    def process(self, drone_poly, estimated_pos, search_radius=300):
        if drone_poly is None: return []
        
        drone_vec = self.compute_shape_vector(drone_poly)
        if np.sum(drone_vec) == 0: return []

        candidates = []
        for item in self.map_data:
            # --- STRICT GATING LOGIC ---
            # Bắt buộc phải có estimated_pos và check khoảng cách
            if estimated_pos is not None:
                dist_to_prev = np.linalg.norm(np.array(item['centroid']) - estimated_pos)
                
                # NẾU > 300 LÀ BỎ QUA NGAY LẬP TỨC
                if dist_to_prev > search_radius:
                    continue
            else:
                # Nếu không có estimated_pos (lúc init), cho phép tìm toàn map
                pass 
            
            # Khớp hình dáng
            shape_dist = self.coarse_distance(drone_vec, item['vector'])
            if shape_dist < 0.6: 
                candidates.append(item)

        if not candidates: return []

        gmm_components = []
        vec_cam_to_bld = np.array([drone_poly.centroid.x - 200, drone_poly.centroid.y - 200])
        
        for cand in candidates:
            alpha = self.compute_correlation(drone_poly, cand['poly'])
            
            if alpha > 0.6: # Tăng ngưỡng lên chút cho chắc
                map_centroid = np.array([cand['centroid'][0], cand['centroid'][1]])
                measured_pos = map_centroid - vec_cam_to_bld
                
                sigma_val = 30.0 * (1.0 - alpha) + 5.0
                cov_matrix = np.array([[sigma_val**2, 0], [0, sigma_val**2]])
                
                gmm_components.append({
                    'mu': measured_pos,
                    'cov': cov_matrix,
                    'alpha': alpha
                })
                
        return gmm_components

# ==============================================================================
# 3. MAIN (ĐÃ XÓA FALLBACK)
# ==============================================================================
def create_complex_map(W, H):
    img = np.zeros((H, W), dtype=np.uint8)
    np.random.seed(99) 
    for _ in range(350):
        x = np.random.randint(50, W-300); y = np.random.randint(50, H-300)
        w, h = np.random.randint(60, 120), np.random.randint(60, 120)
        cv2.rectangle(img, (x, y), (x+w, y+h), 255, -1)
        if np.random.rand() > 0.5:
            cut_w, cut_h = int(w*0.6), int(h*0.6)
            cv2.rectangle(img, (x+w-cut_w, y), (x+w, y+cut_h), 0, -1)
    return img

def main():
    W, H = 5000, 4000
    map_img = create_complex_map(W, H)
    
    _, thresh = cv2.threshold(map_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    map_polys = []
    for c in contours:
        if cv2.contourArea(c) > 800:
            p = Polygon(c.reshape(-1, 2))
            if not p.is_valid: p = p.buffer(0)
            map_polys.append(p)

    pf = PaperCompliantPF(N=3000, W=W, H=H)
    matcher = CFBVMMatcher(map_polys) 
    controller = DroneController()
    
    # Init 
    real_pos_sim = np.array([1000.0, 1000.0]) 
    pf.init(real_pos_sim[0], real_pos_sim[1]) 
    
    DISPLAY_SCALE = 0.15
    h_d, w_d = int(H*DISPLAY_SCALE), int(W*DISPLAY_SCALE)
    bg_static = cv2.resize(map_img, (w_d, h_d), interpolation=cv2.INTER_NEAREST)
    bg_static = cv2.cvtColor(bg_static, cv2.COLOR_GRAY2BGR)

    print("\n[MODE] STRICT 3M RADIUS (No Fallback)")

    while True:
        velocity, real_frame, _ = controller.get_control_step()

        real_pos_sim += velocity
        real_pos_sim[0] = np.clip(real_pos_sim[0], 200, W-200)
        real_pos_sim[1] = np.clip(real_pos_sim[1], 200, H-200)
        
        # Sim Camera
        M = np.float32([[1, 0, -real_pos_sim[0] + 200], [0, 1, -real_pos_sim[1] + 200]])
        sim_cam_view = cv2.warpAffine(map_img, M, (400, 400))
        
        if len(sim_cam_view.shape)==3: gray = cv2.cvtColor(sim_cam_view, cv2.COLOR_BGR2GRAY)
        else: gray = sim_cam_view
        _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        visible_polys = [] 
        if cnts:
            for c in cnts:
                if cv2.contourArea(c) > 300: 
                    approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c,True), True)
                    if len(approx) >= 3:
                        poly = Polygon(approx.reshape(-1,2))
                        if not poly.is_valid: poly = poly.buffer(0)
                        visible_polys.append(poly)

        # --- ALGORITHM CORE ---
        pf.propagate(velocity)
        
        # Lấy vị trí ước lượng hiện tại (t-1 đã cộng velocity)
        current_est, is_credible, _ = pf.estimate_and_evaluate()
        
        all_hypotheses = []
        
        for i, d_poly in enumerate(visible_polys):
            # --- STRICT SEARCH ---
            # Luôn luôn tìm trong bán kính 300 quanh current_est
            # Không có logic fallback (global search) nào ở đây nữa
            matches = matcher.process(d_poly, current_est, search_radius=300)
            
            for m in matches:
                m['source_id'] = i
                all_hypotheses.append(m)
        
        # Logic Voting (Giữ nguyên để tăng độ chính xác)
        final_gmm_data = []
        if len(visible_polys) >= 2:
            for h1 in all_hypotheses:
                for h2 in all_hypotheses:
                    if h1['source_id'] != h2['source_id']:
                        dist = np.linalg.norm(h1['mu'] - h2['mu'])
                        if dist < 60.0:
                            final_gmm_data.append(h1)
        else:
            for h in all_hypotheses:
                final_gmm_data.append(h)

        # Cập nhật PF
        # Vì đã lọc chặt từ đầu, nên những gì vào đây đều là trong bán kính 3m
        pf.update_with_gmm(final_gmm_data)
        pf.resample(None) # Không cần inject hạt từ GMM nữa, chỉ resample thuần túy
            
        est_pos, is_credible, _ = pf.estimate_and_evaluate()
        
        # VISUALIZATION
        vis = bg_static.copy()
        def to_s(x,y): return int(x*DISPLAY_SCALE), int(y*DISPLAY_SCALE)
        
        # 1. Vẽ các hạt (Particles)
        for p in pf.particles:
            cv2.circle(vis, to_s(p[0], p[1]), 1, (0, 255, 0), -1)

        # 2. Vẽ các điểm khớp GMM (Vòng tròn vàng)
        for comp in final_gmm_data:
            mx, my = comp['mu']
            cv2.circle(vis, to_s(mx, my), 8, (0, 255, 255), 2) 
            
        # 3. Vẽ vị trí thực (Ground Truth - Xanh dương nhạt)
        rx, ry = to_s(real_pos_sim[0], real_pos_sim[1])
        cv2.rectangle(vis, (rx-15, ry-15), (rx+15, ry+15), (255, 200, 0), 2)
        
        # 4. Vẽ vị trí ước lượng (Estimated - Đỏ)
        ex, ey = to_s(est_pos[0], est_pos[1])
        cv2.circle(vis, (ex, ey), 5, (0, 0, 255), -1)
        cv2.line(vis, (rx, ry), (ex, ey), (255, 255, 255), 1)

        # 5. PIP (Picture in Picture) - Camera góc trái
        if real_frame is not None: 
            pip = cv2.resize(real_frame, (150, 150))
        else: 
            pip = cv2.cvtColor(cv2.resize(sim_cam_view, (150, 150)), cv2.COLOR_GRAY2BGR)
        
        # Dán ảnh camera vào góc trái dưới
        vis[h_d-160 : h_d-10, 10 : 160] = pip
        
        # --- ĐOẠN CODE TẠO KHUNG MÀU MÀ BẠN CẦN ---
        if len(final_gmm_data) > 0:
            # Nếu tìm thấy khớp -> Màu Xanh lá
            border_col = (0, 255, 0)
            status_text = "LOCKED"
        else:
            # Nếu không thấy gì -> Màu Đỏ
            border_col = (0, 0, 255)
            status_text = "SEARCHING"

        # Vẽ khung viền
        cv2.rectangle(vis, (10, h_d-160), (160, h_d-10), border_col, 3)
        # Vẽ chữ trạng thái đè lên khung
        cv2.putText(vis, status_text, (20, h_d-140), cv2.FONT_HERSHEY_PLAIN, 1.2, border_col, 2)
        # ------------------------------------------
        
        # Hiển thị sai số
        err_dist = int(np.linalg.norm(real_pos_sim - est_pos))
        cv2.putText(vis, f"Err: {err_dist}cm", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("CFBVM - Strict Mode", vis)
        if cv2.waitKey(1) == 27: 
            if controller.has_drone: controller.drone.land()
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

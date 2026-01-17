import cv2
import numpy as np
import math
from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate
from shapely.ops import unary_union

# ==============================================================================
# 1. SHAPE DESCRIPTOR (Giữ nguyên)
# ==============================================================================
class ShapeDescriptor:
    def __init__(self):
        self.radii = [30, 60, 90] 
        self.sectors = 8
     
    def compute(self, target_poly, context_union):
        cx, cy = target_poly.centroid.x, target_poly.centroid.y
        center = Point(cx, cy)
        vector = []
        prev_r = 0
        for r in self.radii:
            for j in range(self.sectors):
                angle_start = j * (360.0 / self.sectors)
                angle_end = (j + 1) * (360.0 / self.sectors)
                W = r * 2.5 
                rad_s, rad_e = math.radians(angle_start), math.radians(angle_end)
                wedge_poly = Polygon([
                    (cx, cy),
                    (cx + W*math.cos(rad_s), cy + W*math.sin(rad_s)),
                    (cx + W*math.cos(rad_e), cy + W*math.sin(rad_e))
                ])
                ring = center.buffer(r).difference(center.buffer(prev_r))
                sector_area = ring.intersection(wedge_poly)
                try:
                    val = sector_area.intersection(context_union).area
                except:
                    val = 0.0
                vector.append(val)
            prev_r = r
        return np.array(vector)

# ==============================================================================
# 2. OPTIMIZED MATCHING SYSTEM (SPATIAL FILTERING)
# ==============================================================================
class StrictMatcher:
    def __init__(self):
        self.desc_engine = ShapeDescriptor()
        
    def coarse_filter(self, d_poly, map_polys, map_descriptors, drone_context, drone_pos):
        """
        BƯỚC 1: COARSE MATCHING + SPATIAL FILTERING
        Chỉ so sánh các tòa nhà trong bán kính gần (Spatial Check) trước khi tính toán.
        """
        candidates = []
        
        # [NEW] Tính toán centroid của vật thể drone nhìn thấy (tương đối)
        d_cx = d_poly.centroid.x
        d_cy = d_poly.centroid.y
        
        # Pre-compute descriptor cho vật thể nhìn thấy
        f_cam = self.desc_engine.compute(d_poly, drone_context)
        
        # [OPTIMIZATION] Search Radius: Chỉ tìm tòa nhà cách vị trí dự đoán < 1500px
        # Điều này cực kỳ quan trọng với map 8000x6000
        search_radius = 1500.0 
        
        for i, m_poly in enumerate(map_polys):
            # 1. SPATIAL CHECK (Lọc Không Gian) - Quan trọng nhất cho Big Map
            # Kiểm tra khoảng cách từ ước lượng drone đến tòa nhà trên map
            dist_spatial = math.hypot(m_poly.centroid.x - drone_pos[0], m_poly.centroid.y - drone_pos[1])
            if dist_spatial > search_radius:
                continue # Bỏ qua tòa nhà ở xa lắc

            # 2. AREA RATIO CHECK
            ratio = d_poly.area / m_poly.area
            if ratio < 0.7 or ratio > 1.3:
                continue

            # 3. VECTOR DISTANCE
            dist = np.sum(np.abs(f_cam - map_descriptors[i]))
            candidates.append((dist, m_poly))
            
        candidates.sort(key=lambda x: x[0])
        return [x[1] for x in candidates[:5]]

    def fine_matching_strict(self, d_poly, m_poly):
        """BƯỚC 2: FINE MATCHING (Rotation Search)"""
        best_score = 0.0
        v_box = m_poly.envelope.area
        if v_box < 1e-6: v_box = 1e-6

        angles = [-5, 0, 5] 
        dx = m_poly.centroid.x - d_poly.centroid.x
        dy = m_poly.centroid.y - d_poly.centroid.y
        base_aligned = translate(d_poly, xoff=dx, yoff=dy)

        for ang in angles:
            rotated_cam = rotate(base_aligned, ang, origin='centroid')
            inter_area = rotated_cam.intersection(m_poly).area
            score = inter_area / v_box
            if score > best_score: best_score = score

        best_sigma = 30.0 * (1.0 - best_score) + 2.0
        return best_score, best_sigma

    def find_matches(self, drone_polys, map_polys, map_descriptors, view_center_offset, current_drone_est):
        observations = []
        if not drone_polys: return []
        
        drone_context = unary_union(drone_polys)
        
        for d_poly in drone_polys:
            if d_poly.area < 50: continue

            # Truyền thêm vị trí ước lượng của drone để lọc không gian
            potential_candidates = self.coarse_filter(d_poly, map_polys, map_descriptors, drone_context, current_drone_est)
            
            if not potential_candidates: continue

            for m_poly in potential_candidates:
                score, sigma = self.fine_matching_strict(d_poly, m_poly)
                
                if score > 0.65: # Tăng ngưỡng lên chút cho map to
                    vec_rel_x = d_poly.centroid.x - view_center_offset[0]
                    vec_rel_y = d_poly.centroid.y - view_center_offset[1]
                    
                    est_global_x = m_poly.centroid.x - vec_rel_x
                    est_global_y = m_poly.centroid.y - vec_rel_y
                    
                    observations.append({
                        'mu': np.array([est_global_x, est_global_y]),
                        'sigma': sigma,
                        'score': score
                    })
                    
        return observations

# ==============================================================================
# 3. FUSION PARTICLE FILTER (Optimized for speed)
# ==============================================================================
class FusionParticleFilter:
    def __init__(self, N, W, H):
        self.N = N
        self.W, self.H = W, H
        self.particles_pos = np.zeros((N, 2))
        self.particles_cov = np.ones(N) * 20.0 
        self.weights = np.ones(N) / N
        self.initialized = False
        self.last_estimate = None

    def init(self, x, y):
        # Map to nên sai số khởi tạo GPS có thể rất lớn (300px)
        self.particles_pos[:, 0] = np.random.normal(x, 300, self.N)
        self.particles_pos[:, 1] = np.random.normal(y, 300, self.N)
        self.particles_cov[:] = 30.0
        self.weights[:] = 1.0 / self.N
        self.initialized = True
        self.last_estimate = np.array([x, y])

    def propagate(self, velocity):
        noise = np.random.normal(0, 5.0, (self.N, 2)) # Tăng noise
        self.particles_pos += velocity + noise
        self.particles_cov += 1.5
        np.clip(self.particles_cov, 5.0, 80.0, out=self.particles_cov)
        np.clip(self.particles_pos[:, 0], 0, self.W, out=self.particles_pos[:, 0])
        np.clip(self.particles_pos[:, 1], 0, self.H, out=self.particles_pos[:, 1])

    def fusion_update(self, observations):
        if not self.initialized or not observations: return
        observations.sort(key=lambda x: x['score'], reverse=True)
        best_obs = observations[0]
        if best_obs['score'] < 0.5: return

        self.weights.fill(1.e-20)
        
        # Vector hóa tính toán khoảng cách (NumPy optimization) thay vì vòng lặp
        # để tăng tốc độ xử lý 2000 hạt
        diff = self.particles_pos - best_obs['mu']
        dists = np.linalg.norm(diff, axis=1)
        
        combined_sigmas = np.sqrt(self.particles_cov) + best_obs['sigma']
        
        # Chỉ update những hạt nằm trong vùng gate
        mask = dists < (5 * combined_sigmas)
        
        if np.any(mask):
            R = (best_obs['sigma'] * 2.0)**2
            P = self.particles_cov[mask]**2
            K = P / (P + R)
            
            # Update Position
            self.particles_pos[mask] += K[:, np.newaxis] * (best_obs['mu'] - self.particles_pos[mask])
            
            # Update Covariance
            P_new = (1.0 - K) * P
            self.particles_cov[mask] = np.sqrt(P_new)
            
            # Update Weights
            exponent = -0.5 * (dists[mask]**2) / (R + 1e-6)
            lik = np.exp(exponent) * best_obs['score']
            self.weights[mask] = lik + 1.e-20

        s = np.sum(self.weights)
        if s > 0: self.weights /= s
        else: self.weights[:] = 1.0 / self.N

    def resample_and_redistribute(self, observations):
        if not self.initialized: return
        gamma = 0.0
        best_obs = None
        if observations:
            best_obs = max(observations, key=lambda x: x['score'])
            if best_obs['score'] > 0.8: gamma = 0.05 
        
        n_sensor = int(self.N * gamma)
        n_motion = self.N - n_sensor
        
        # Resample logic (optimized indexing)
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        step = 1.0 / n_motion
        r = np.random.uniform(0, step)
        vals = r + np.arange(n_motion) * step
        idxs = np.searchsorted(cumulative_sum, vals)
        
        new_pos = np.zeros((self.N, 2))
        new_cov = np.zeros(self.N)
        
        new_pos[:n_motion] = self.particles_pos[idxs]
        new_cov[:n_motion] = self.particles_cov[idxs]
        
        if n_sensor > 0 and best_obs is not None:
            new_pos[n_motion:] = np.random.normal(best_obs['mu'], best_obs['sigma'], (n_sensor, 2))
            new_cov[n_motion:] = best_obs['sigma']

        self.particles_pos = new_pos
        self.particles_cov = new_cov
        self.weights[:] = 1.0 / self.N

    def estimate(self):
        if not self.initialized: return 0,0
        est_x = np.median(self.particles_pos[:, 0])
        est_y = np.median(self.particles_pos[:, 1])
        current_est = np.array([est_x, est_y])
        if self.last_estimate is None: self.last_estimate = current_est
        else: self.last_estimate = 0.8 * self.last_estimate + 0.2 * current_est
        return self.last_estimate[0], self.last_estimate[1]

# ==============================================================================
# 4. GIANT CITY MAP GENERATOR
# ==============================================================================
def create_giant_map(W, H):
    print(f"Generating METROPOLIS ({W}x{H})... This may take a moment.")
    img = np.zeros((H, W), dtype=np.uint8)
    np.random.seed(2024) 
    
    # 600 tòa nhà cho map khổng lồ
    num_buildings = 600
    
    for i in range(num_buildings):
        x = np.random.randint(100, W-300)
        y = np.random.randint(100, H-300)
        w = np.random.randint(100, 300) # Tòa nhà to
        h = np.random.randint(100, 300)
        
        shape_type = np.random.randint(0, 4) # Thêm kiểu dáng
        
        if shape_type == 0: # Rect
            cv2.rectangle(img, (x, y), (x+w, y+h), 255, -1)
        elif shape_type == 1: # L
            cv2.rectangle(img, (x, y), (x+w, y+h//3), 255, -1)
            cv2.rectangle(img, (x, y), (x+w//3, y+h), 255, -1)
        elif shape_type == 2: # T
            cv2.rectangle(img, (x, y), (x+w, y+h//3), 255, -1)
            cv2.rectangle(img, (x+w//3, y), (x+2*w//3, y+h), 255, -1)
        elif shape_type == 3: # U (Mới)
            cv2.rectangle(img, (x, y), (x+w//4, y+h), 255, -1) # Trụ trái
            cv2.rectangle(img, (x+3*w//4, y), (x+w, y+h), 255, -1) # Trụ phải
            cv2.rectangle(img, (x, y+3*h//4), (x+w, y+h), 255, -1) # Đáy
            
    return img

# ==============================================================================
# 5. MAIN
# ==============================================================================
def main():
    # --- CẤU HÌNH MAP KHỔNG LỒ ---
    W, H = 8000, 6000 # 48 Megapixels Map
    VIEW_SIZE = 400
    
    # Tỷ lệ hiển thị cực nhỏ để xem toàn cảnh (Radar View)
    DISPLAY_SCALE = 0.12  # 8000px -> 960px width
    
    # Init Map
    map_img = create_giant_map(W, H)
    map_polys = get_polygons_from_image(map_img) # Cần import hàm này từ code cũ
    map_context = unary_union(map_polys)
    
    matcher = StrictMatcher()
    print(f"Pre-computing descriptors for {len(map_polys)} buildings...")
    map_descriptors = [matcher.desc_engine.compute(p, map_context) for p in map_polys]
    
    # Tăng hạt lên 2000 để phủ map lớn
    pf = FusionParticleFilter(N=2000, W=W, H=H)
    
    # Init Drone
    drone_pos = np.array([1000.0, 1000.0])
    velocity = np.array([15.0, 10.0]) # Tăng tốc độ bay (Siêu thanh)
    pf.init(drone_pos[0] + 200, drone_pos[1] - 200)

    print("--- STARTING GIANT MAP SIMULATION ---")
    
    disp_w = int(W * DISPLAY_SCALE)
    disp_h = int(H * DISPLAY_SCALE)
    
    # Tạo background hiển thị một lần cho nhẹ
    vis_background = cv2.resize(map_img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    vis_background = cv2.cvtColor(vis_background, cv2.COLOR_GRAY2BGR)

    while True:
        # Physics
        drone_pos += velocity
        if drone_pos[0] < VIEW_SIZE or drone_pos[0] > W-VIEW_SIZE: velocity[0] *= -1
        if drone_pos[1] < VIEW_SIZE or drone_pos[1] > H-VIEW_SIZE: velocity[1] *= -1
        
        # Vision
        M = np.float32([[1, 0, -drone_pos[0] + VIEW_SIZE/2], [0, 1, -drone_pos[1] + VIEW_SIZE/2]])
        cam_view = cv2.warpAffine(map_img, M, (VIEW_SIZE, VIEW_SIZE))
        drone_polys_view = get_polygons_from_image(cam_view) # Hàm này ở code cũ, nhớ copy vào
        
        # CFBVM
        est_curr = pf.estimate()
        pf.propagate(velocity)
        
        # Truyền `est_curr` vào để lọc không gian
        matches = matcher.find_matches(drone_polys_view, map_polys, map_descriptors, 
                                     (VIEW_SIZE/2, VIEW_SIZE/2), current_drone_est=est_curr)
        
        if matches: pf.fusion_update(matches)
        pf.resample_and_redistribute(matches)
        est_x, est_y = pf.estimate()
        
        # --- VISUALIZATION (RADAR MODE) ---
        vis_frame = vis_background.copy() # Vẽ lên bản sao
        
        def to_screen(x, y):
            return int(x * DISPLAY_SCALE), int(y * DISPLAY_SCALE)
        
        # Draw View Box
        sx, sy = to_screen(drone_pos[0], drone_pos[1])
        vr = int((VIEW_SIZE/2) * DISPLAY_SCALE)
        if vr < 1: vr = 1
        cv2.rectangle(vis_frame, (sx-vr, sy-vr), (sx+vr, sy+vr), (0, 165, 255), 2)
        
        # Draw Particles (Chỉ vẽ 1 phần để đỡ rối)
        for p in pf.particles_pos[::10]:
            px, py = to_screen(p[0], p[1])
            cv2.circle(vis_frame, (px, py), 1, (0, 255, 0), -1)
            
        # Draw Estimate
        ex, ey = to_screen(est_x, est_y)
        cv2.circle(vis_frame, (ex, ey), 5, (0, 0, 255), -1)
        
        # PiP View (Camera Drone thật - Rõ nét)
        pip = cv2.cvtColor(cv2.resize(cam_view, (250, 250)), cv2.COLOR_GRAY2BGR)
        vis_frame[disp_h-260:disp_h-10, 10:260] = pip
        cv2.rectangle(vis_frame, (10, disp_h-260), (260, disp_h-10), (0, 255, 255), 2)
        cv2.putText(vis_frame, f"CAM: {int(drone_pos[0])},{int(drone_pos[1])}", (15, disp_h-240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(vis_frame, f"METROPOLIS: {W}x{H} | Buildings: {len(map_polys)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Giant Map Radar", vis_frame)
        if cv2.waitKey(1) == 27: break

    cv2.destroyAllWindows()

# Helper function (Copy lại từ code trước để chạy được)
def get_polygons_from_image(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) > 300: 
            approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
            if len(approx) >= 3:
                p = Polygon(approx.reshape(-1, 2))
                if not p.is_valid: p = p.buffer(0)
                polys.append(p)
    return polys

if __name__ == "__main__":
    main()

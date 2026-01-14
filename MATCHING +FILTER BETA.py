import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate
from shapely.strtree import STRtree  

# ==============================================================================
# PHẦN 1: CẤU TRÚC DỮ LIỆU
# ==============================================================================

class ShapePoint:
    def __init__(self, id, polygon, source_type, contour):
        self.id = id
        self.region = polygon       
        self.source = source_type   
        self.S = polygon.area
        self.centroid = np.array([polygon.centroid.x, polygon.centroid.y])
        self.contour = contour
        
        # Tính độ tròn (Circularity) để lọc hình dáng
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: self.circularity = 0
        else: self.circularity = 4 * np.pi * (self.S / (perimeter**2))

class GaussianObservation:
    def __init__(self, mean_pos, alpha_score):
        self.mean = np.array(mean_pos) 
        self.score = alpha_score        
        # Score càng cao -> Sigma càng nhỏ (tin cậy cao)
        base_sigma = 5.0 
        min_sigma = 1.0  
        val = (base_sigma / (alpha_score + 1e-6))
        self.sigma_val = max(val, min_sigma)**2 

class Particle:
    def __init__(self, pos, weight):
        self.mu = np.array(pos, dtype=np.float64)
        self.w = weight

# ==============================================================================
# PHẦN 2: XỬ LÝ MASK (Extract Polygon)
# ==============================================================================

class MaskProcessor:
    def extract_features(self, binary_mask, source_type):
        if binary_mask is None: return []
        
        if binary_mask.dtype == bool:
            mask_img = (binary_mask * 255).astype(np.uint8)
        else:
            mask_img = binary_mask.astype(np.uint8)
            if np.max(mask_img) <= 1 and np.max(mask_img) > 0:
                mask_img = mask_img * 255

        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []
        
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 50: continue 
            
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) >= 3:
                pts = [tuple(pt[0]) for pt in approx]
                poly = Polygon(pts)
                if not poly.is_valid: poly = poly.buffer(0)
                shapes.append(ShapePoint(f"{source_type}_{i}", poly, source_type, cnt))
        return shapes

# ==============================================================================
# PHẦN 3: MATCHING SYSTEM (NO ROTATION - ONLY TRANSLATION)
# ==============================================================================

class RobustMatchingSystem:
    def __init__(self, drone_view_size):
        self.view_w, self.view_h = drone_view_size
        self.map_tree = None
        self.map_objects = []

    def build_spatial_index(self, map_objs):
        self.map_objects = map_objs
        if not map_objs: return
        geoms = [obj.region for obj in map_objs]
        self.map_tree = STRtree(geoms)
        print(f"[System] Built R-tree for {len(map_objs)} map objects.")

    def get_nearby_candidates(self, center_pos, search_radius=300):
        if self.map_tree is None or center_pos is None: 
            return self.map_objects
        
        x, y = center_pos
        search_box = Polygon([
            (x - search_radius, y - search_radius),
            (x + search_radius, y - search_radius),
            (x + search_radius, y + search_radius),
            (x - search_radius, y + search_radius)
        ])
        
        indices = self.map_tree.query(search_box)
        candidates = [self.map_objects[i] for i in indices]
        return candidates

    # [PHƯƠNG PHÁP CŨ CỦA BẠN] Hill Climbing - Tìm vị trí khớp nhất
    def find_best_alignment_hill_climbing(self, drone_poly, map_poly):
        # Bắt đầu dò từ vị trí chênh lệch tâm (centroid alignment)
        c_dx = map_poly.centroid.x - drone_poly.centroid.x
        c_dy = map_poly.centroid.y - drone_poly.centroid.y
        
        # Polygon khởi đầu
        best_poly = translate(drone_poly, xoff=c_dx, yoff=c_dy)
        try:
            best_inter = best_poly.intersection(map_poly).area
        except:
            return (c_dx, c_dy), 0.0

        current_offset = [c_dx, c_dy]
        
        step = 10.0 
        max_iter = 15
        
        # Leo đồi để tinh chỉnh từng pixel
        for _ in range(max_iter): 
            found_better = False
            for mx, my in [(0, step), (0, -step), (step, 0), (-step, 0)]:
                test_poly = translate(best_poly, xoff=mx, yoff=my)
                try:
                    test_inter = test_poly.intersection(map_poly).area
                except: continue

                if test_inter > best_inter:
                    best_inter = test_inter
                    best_poly = test_poly
                    current_offset[0] += mx
                    current_offset[1] += my
                    found_better = True
                    break 
            
            if not found_better:
                step *= 0.5 
                if step < 1.0: break

        return tuple(current_offset), best_inter

    def run(self, drone_objs, estimated_pos=None):
        observations = []
        
        # Tối ưu tìm kiếm bằng R-tree
        if estimated_pos is not None:
            candidates_pool = self.get_nearby_candidates(estimated_pos, search_radius=200)
            if not candidates_pool: candidates_pool = self.map_objects
        else:
            candidates_pool = self.map_objects

        for d_obj in drone_objs:
            best_score = 0
            final_offset = None
            
            # 1. Lọc Hình Dáng (Shape Check) - Tránh nhầm Tròn với Vuông
            filtered_candidates = []
            for m_obj in candidates_pool:
                # So sánh contour (Hu Moments)
                shape_dist = cv2.matchShapes(d_obj.contour, m_obj.contour, cv2.CONTOURS_MATCH_I1, 0)
                # So sánh độ tròn
                circ_diff = abs(d_obj.circularity - m_obj.circularity)
                
                # Điều kiện lọc: Hình dạng phải tương đối giống nhau
                if shape_dist < 0.2 and circ_diff < 0.3:
                    filtered_candidates.append(m_obj)
            
            # 2. Matching Tinh (Hill Climbing)
            for m_obj in filtered_candidates:
                offset, inter_area = self.find_best_alignment_hill_climbing(d_obj.region, m_obj.region)
                
                # Tính IoU
                union_area = d_obj.region.area + m_obj.region.area - inter_area
                iou = inter_area / (union_area + 1e-6)

                if iou > best_score:
                    best_score = iou
                    final_offset = offset
            
            # Nếu IoU đủ tốt (> 0.6), ghi nhận quan sát
            if best_score > 0.6 and final_offset is not None:
                # Chuyển đổi offset về tâm Camera
                corrected_pos = (
                    final_offset[0] + self.view_w / 2.0,
                    final_offset[1] + self.view_h / 2.0
                )
                observations.append(GaussianObservation(corrected_pos, best_score))

        return observations

# ==============================================================================
# PHẦN 4: PARTICLE FILTER (SMOOTH)
# ==============================================================================

class GMMParticleFilter:
    def __init__(self, num_particles=2000, map_bounds=(0, 1000, 0, 1000)):
        self.N = num_particles
        self.bounds = map_bounds
        self.particles = [] 
        self.is_initialized = False 

    def initialize_from_observation(self, observations):
        if not observations: return False
        best_obs = max(observations, key=lambda o: o.score)
        center = best_obs.mean
        sigma = np.sqrt(best_obs.sigma_val) * 3.0
        
        self.particles = []
        for _ in range(self.N):
            pos = np.random.normal(center, sigma)
            pos[0] = np.clip(pos[0], self.bounds[0], self.bounds[1])
            pos[1] = np.clip(pos[1], self.bounds[2], self.bounds[3])
            self.particles.append(Particle(pos, 1.0/self.N))
        self.is_initialized = True
        return True

    def predict(self, move_vector, noise_std=2.0):
        if not self.is_initialized: return
        move = np.array(move_vector)
        noise = np.random.normal(0, noise_std, (self.N, 2))
        for i, p in enumerate(self.particles):
            p.mu += move + noise[i]
            p.mu[0] = np.clip(p.mu[0], self.bounds[0], self.bounds[1])
            p.mu[1] = np.clip(p.mu[1], self.bounds[2], self.bounds[3])

    def update(self, observations):
        if not self.is_initialized or not observations: return
        
        best_obs = max(observations, key=lambda o: o.score)
        obs_mean = best_obs.mean
        obs_var = best_obs.sigma_val
        
        p_pos = np.array([p.mu for p in self.particles])
        diff = p_pos - obs_mean
        dist_sq = np.sum(diff**2, axis=1)
        
        likelihood = np.exp(-0.5 * dist_sq / obs_var) + 1e-300
        current_weights = np.array([p.w for p in self.particles])
        new_weights = current_weights * likelihood
        w_sum = np.sum(new_weights)
        if w_sum > 0: new_weights /= w_sum
        for i, p in enumerate(self.particles):
            p.w = new_weights[i]

    def resample(self):
        if not self.is_initialized: return
        weights = np.array([p.w for p in self.particles])
        n_eff = 1.0 / (np.sum(weights**2) + 1e-10)
        
        if n_eff < self.N / 2:
            indices = np.random.choice(range(self.N), size=self.N, p=weights)
            new_particles = []
            for i in indices:
                old = self.particles[i]
                jitter = np.random.normal(0, 1.0, 2) 
                new_particles.append(Particle(old.mu + jitter, 1.0/self.N))
            self.particles = new_particles

    def estimate(self):
        if not self.is_initialized: return np.zeros(2)
        x = sum(p.mu[0] * p.w for p in self.particles)
        y = sum(p.mu[1] * p.w for p in self.particles)
        return np.array([x, y])

# ==============================================================================
# PHẦN 5: CHƯƠNG TRÌNH CHÍNH (SIMULATION - NO ROTATION)
# ==============================================================================

def create_synthetic_map():
    W, H = 800, 600
    map_img = np.zeros((H, W), dtype=np.uint8)
    
    # Bản đồ tĩnh
    cv2.rectangle(map_img, (200, 200), (300, 300), 255, -1) # Vuông
    cv2.circle(map_img, (600, 400), 60, 255, -1)            # Tròn
    pts = np.array([[400, 100], [450, 200], [350, 200]], np.int32) # Tam giác
    cv2.fillPoly(map_img, [pts], 255)
    
    cv2.rectangle(map_img, (50, 450), (150, 500), 255, -1) # Chữ nhật dài
    return map_img

def main():
    map_img = create_synthetic_map()
    mask_processor = MaskProcessor()
    
    # Extract bản đồ
    map_objs = mask_processor.extract_features(map_img, "MAP")
    
    # Cấu hình
    VIEW_W, VIEW_H = 150, 150
    matcher = RobustMatchingSystem(drone_view_size=(VIEW_W, VIEW_H))
    matcher.build_spatial_index(map_objs) 
    
    pf = GMMParticleFilter(num_particles=5000, map_bounds=(0, 800, 0, 600))
    
    true_x, true_y = 100, 100
    vx, vy = 3, 2             
    
    print(">>> SIMULATION STARTED: Translation Only (X, Y). No Rotation.")

    while True:
        # Cập nhật vị trí
        true_x += vx
        true_y += vy
        if true_x > 750 or true_x < 50: vx = -vx
        if true_y > 550 or true_y < 50: vy = -vy
        
        # GIẢ LẬP CAMERA (Cắt trực tiếp từ map, không xoay)
        top_left_x = int(true_x - VIEW_W // 2)
        top_left_y = int(true_y - VIEW_H // 2)
        
        # Xử lý biên để không bị crash khi ra khỏi map
        if top_left_x < 0 or top_left_y < 0 or \
           top_left_x + VIEW_W > map_img.shape[1] or \
           top_left_y + VIEW_H > map_img.shape[0]:
            # Đơn giản là đổi chiều nếu chạm biên để demo
            vx = -vx
            vy = -vy
            continue

        drone_view_img = map_img[top_left_y : top_left_y + VIEW_H, 
                                 top_left_x : top_left_x + VIEW_W].copy()
        
        # --- PROCESSING ---
        drone_objs = mask_processor.extract_features(drone_view_img, "DRONE")
        
        # Lấy estimate cũ
        search_hint = pf.estimate() if pf.is_initialized else None
        
        # Chạy Matching (Không cần quét góc)
        observations = matcher.run(drone_objs, estimated_pos=search_hint)
        
        # Particle Filter Update
        est_pos = np.array([0, 0])
        status_text = "LOST / SEARCHING"
        status_color = (0, 255, 255)

        if not pf.is_initialized:
            if len(observations) > 0:
                pf.initialize_from_observation(observations)
                status_text = "INITIALIZED!"
        else:
            status_text = "TRACKING"
            status_color = (0, 0, 255)
            pf.predict([vx, vy], noise_std=1.5)
            if len(observations) > 0:
                pf.update(observations)
            pf.resample()
            est_pos = pf.estimate()
        
        # --- VISUALIZATION ---
        vis_img = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
        
        if pf.is_initialized:
            # Vẽ hạt
            for p in pf.particles[::10]: 
                cv2.circle(vis_img, (int(p.mu[0]), int(p.mu[1])), 1, (0, 255, 0), -1)
            # Vẽ vị trí ước lượng (Đỏ)
            cv2.circle(vis_img, (int(est_pos[0]), int(est_pos[1])), 6, (0, 0, 255), -1)

        # Vẽ khung nhìn thực tế (Vàng)
        cv2.rectangle(vis_img, (top_left_x, top_left_y), 
                      (top_left_x + VIEW_W, top_left_y + VIEW_H), (255, 255, 0), 2)
        
        cv2.putText(vis_img, f"Status: {status_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow("Map Tracking", vis_img)
        
        cam_vis = cv2.cvtColor(drone_view_img, cv2.COLOR_GRAY2BGR)
        cv2.circle(cam_vis, (VIEW_W//2, VIEW_H//2), 3, (0,0,255), -1)
        cv2.imshow("Drone Camera View", cam_vis)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

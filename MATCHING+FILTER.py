import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, scale

# ==============================================================================
# PHẦN 1: CẤU TRÚC DỮ LIỆU (GIỮ NGUYÊN)
# ==============================================================================

class ShapePoint:
    def __init__(self, id, polygon, source_type):
        self.id = id
        self.region = polygon       
        self.source = source_type   
        self.S = polygon.area
        self.centroid = np.array([polygon.centroid.x, polygon.centroid.y])

class GaussianObservation:
    def __init__(self, mean_pos, alpha_score):
        self.mean = np.array(mean_pos) 
        self.score = alpha_score       
        self.sigma_val = (5.0 / (alpha_score + 1e-6))**2 

class Particle:
    def __init__(self, pos, weight):
        self.mu = np.array(pos, dtype=np.float64)
        self.w = weight
        self.c = 0.0 

# ==============================================================================
# PHẦN 2: XỬ LÝ MASK TỪ AI (THAY THẾ LOADER CŨ)
# ==============================================================================

class MaskProcessor:
    """
    Class này nhận đầu vào là ảnh nhị phân (Binary Mask) từ Detectron2
    và chuyển đổi trực tiếp sang ShapePoint. Bỏ qua bước thresholding thừa thãi.
    """
    def extract_features(self, binary_mask, source_type, scale_ratio=1.0):
        # Input: binary_mask là numpy array (0 và 1 hoặc 0 và 255)
        if binary_mask is None: return []

        # Đảm bảo format uint8 cho OpenCV
        if binary_mask.dtype == bool:
            mask_img = (binary_mask * 255).astype(np.uint8)
        else:
            mask_img = binary_mask.astype(np.uint8)
            # Nếu mask là float 0-1, scale lên 255
            if np.max(mask_img) <= 1 and np.max(mask_img) > 0:
                mask_img = mask_img * 255

        # Tìm contours trực tiếp trên mask
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []

        for i, cnt in enumerate(contours):
            # Bỏ qua nhiễu nhỏ
            if cv2.contourArea(cnt) < 50: continue 

            # Xấp xỉ đa giác
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) >= 3:
                pts = [tuple(pt[0]) for pt in approx]
                poly = Polygon(pts)
                
                # Fix lỗi hình học
                if not poly.is_valid: poly = poly.buffer(0)
                
                if scale_ratio != 1.0:
                    poly = scale(poly, xfact=scale_ratio, yfact=scale_ratio, origin=(0,0))

                shapes.append(ShapePoint(f"{source_type}_{i}", poly, source_type))

        return shapes

# ==============================================================================
# PHẦN 3: MATCHING SYSTEM (GIỮ NGUYÊN)
# ==============================================================================

class RobustMatchingSystem:
    def __init__(self, drone_view_size=(150, 150)):
        self.view_w, self.view_h = drone_view_size

    def calculate_alpha(self, poly_cam, poly_ref):
        try:
            inter_area = poly_cam.intersection(poly_ref).area
            if inter_area == 0: return 0.0
            denom = np.sqrt(poly_cam.area * poly_ref.area + 1e-6)
            return inter_area / denom
        except:
            return 0.0

    def find_best_alignment_hill_climbing(self, drone_poly, map_poly):
        dx_init = map_poly.centroid.x - drone_poly.centroid.x
        dy_init = map_poly.centroid.y - drone_poly.centroid.y
        
        best_poly = translate(drone_poly, xoff=dx_init, yoff=dy_init)
        best_inter = best_poly.intersection(map_poly).area
        current_offset = [dx_init, dy_init]
        
        step = 10.0 
        for _ in range(10): 
            found_better = False
            for mx, my in [(0, step), (0, -step), (step, 0), (-step, 0)]:
                test_poly = translate(best_poly, xoff=mx, yoff=my)
                test_inter = test_poly.intersection(map_poly).area
                
                if test_inter > best_inter:
                    best_inter = test_inter
                    best_poly = test_poly
                    current_offset[0] += mx
                    current_offset[1] += my
                    found_better = True
                    break 
            
            if not found_better:
                step *= 0.5 
                if step < 0.5: break

        return tuple(current_offset)

    def run(self, drone_objs, map_objs):
        observations = []
        
        for d_obj in drone_objs:
            best_alpha = 0
            final_offset = None
            
            candidates = [m for m in map_objs if abs(d_obj.S - m.S)/(m.S+1e-5) < 0.5]
            
            for m_obj in candidates:
                offset = self.find_best_alignment_hill_climbing(d_obj.region, m_obj.region)
                aligned_poly = translate(d_obj.region, xoff=offset[0], yoff=offset[1])
                alpha = self.calculate_alpha(aligned_poly, m_obj.region)
                
                if alpha > best_alpha:
                    best_alpha = alpha
                    final_offset = offset
            
            if best_alpha > 0.6 and final_offset is not None:
                observations.append(GaussianObservation(final_offset, best_alpha))

        return observations

# ==============================================================================
# PHẦN 4: PARTICLE FILTER (GIỮ NGUYÊN)
# ==============================================================================

class GMMParticleFilter:
    def __init__(self, num_particles=1000, map_bounds=(0, 1000, 0, 1000)):
        self.N = num_particles
        self.bounds = map_bounds
        self.particles = [] 
        self.is_initialized = False 

    def initialize_from_observation(self, observations):
        if not observations: return False
        best_obs = max(observations, key=lambda o: o.score)
        center = best_obs.mean
        sigma = np.sqrt(best_obs.sigma_val) * 2.0
        
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
        
        p_pos = np.array([p.mu for p in self.particles])
        obs_means = np.array([o.mean for o in observations])
        obs_vars = np.array([o.sigma_val for o in observations])
        obs_scores = np.array([o.score for o in observations])
        
        if np.sum(obs_scores) == 0: return
        obs_mix_w = obs_scores / np.sum(obs_scores) 

        diff = p_pos[:, None, :] - obs_means[None, :, :]
        dist_sq = np.sum(diff**2, axis=2) / obs_vars[None, :]
        norm_const = 1.0 / (2 * np.pi * obs_vars)
        p_components = obs_mix_w[None, :] * norm_const[None, :] * np.exp(-0.5 * dist_sq)
        total_likelihood = np.sum(p_components, axis=1)

        current_weights = np.array([p.w for p in self.particles])
        new_weights = current_weights * (total_likelihood + 1e-300)
        
        w_sum = np.sum(new_weights)
        if w_sum > 0: new_weights /= w_sum
        else: new_weights[:] = 1.0 / self.N

        min_dist_sq = np.min(dist_sq, axis=1)
        is_matched = min_dist_sq < 9.0

        for i, p in enumerate(self.particles):
            p.w = new_weights[i]
            if is_matched[i]:
                p.c += 1.0 
                p.w *= (1.0 + 0.1 * p.c)
            else:
                p.c = max(0.0, p.c - 0.5)

        final_sum = sum(p.w for p in self.particles)
        if final_sum > 0:
            for p in self.particles: p.w /= final_sum

    def resample(self):
        if not self.is_initialized: return
        weights = np.array([p.w for p in self.particles])
        n_eff = 1.0 / (np.sum(weights**2) + 1e-10)
        
        if n_eff < self.N / 2:
            indices = np.random.choice(range(self.N), size=self.N, p=weights)
            new_particles = []
            for i in indices:
                old = self.particles[i]
                jitter = np.random.normal(0, 0.5, 2)
                new_pos = old.mu + jitter
                new_p = Particle(new_pos, 1.0/self.N)
                new_p.c = old.c
                new_particles.append(new_p)
            self.particles = new_particles

    def estimate(self):
        if not self.is_initialized: return np.zeros(2)
        x = sum(p.mu[0] * p.w for p in self.particles)
        y = sum(p.mu[1] * p.w for p in self.particles)
        return np.array([x, y])

# ==============================================================================
# PHẦN 5: CHƯƠNG TRÌNH CHÍNH
# ==============================================================================

def create_synthetic_data():
    W, H = 800, 600
    map_img = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(map_img, (200, 200), (300, 300), 255, -1) 
    cv2.circle(map_img, (600, 400), 60, 255, -1)
    pts = np.array([[400, 100], [450, 200], [350, 200]], np.int32)
    cv2.fillPoly(map_img, [pts], 255)
    return map_img

def main():
    # Setup
    map_img = create_synthetic_data()
    
    # --- THAY ĐỔI: Dùng MaskProcessor ---
    mask_processor = MaskProcessor()
    
    # Load Map Features (Coi map như một mask lớn)
    map_objs = mask_processor.extract_features(map_img, "MAP")
    
    pf = GMMParticleFilter(num_particles=1000, map_bounds=(0, 800, 0, 600))
    matcher = RobustMatchingSystem(drone_view_size=(150, 150))
    
    true_x, true_y = 100, 100
    vx, vy = 3, 2             
    
    print(">>> SYSTEM STARTED WITH MASK PROCESSOR.")
    print(">>> Simulating Detectron2 binary output...")

    while True:
        # 1. Update Simulation
        true_x += vx
        true_y += vy
        if true_x > 750 or true_x < 50: vx = -vx
        if true_y > 550 or true_y < 50: vy = -vy
        
        # 2. Tạo 'Detectron Output' giả lập
        # (Ở thực tế, bước này là bạn lấy mask từ model Detectron2 của bạn)
        M = np.float32([[1, 0, -true_x], [0, 1, -true_y]])
        
        # Đây chính là biến mask mà code Detectron của bạn sẽ trả về
        detectron_binary_output = cv2.warpAffine(map_img, M, (150, 150))
        
        # 3. Process Mask (Trực tiếp từ binary -> Polygon)
        drone_objs = mask_processor.extract_features(detectron_binary_output, "DRONE")
        
        # 4. Matching & Filter (Workflow cũ)
        observations = matcher.run(drone_objs, map_objs)
        
        est_pos = np.array([0, 0])
        status_text = "WAITING..."
        status_color = (0, 255, 255)

        if not pf.is_initialized:
            if len(observations) > 0:
                success = pf.initialize_from_observation(observations)
                if success: status_text = "INITIALIZED!"
        else:
            status_text = "TRACKING"
            status_color = (0, 0, 255)
            pf.predict([vx, vy], noise_std=2.0)
            if len(observations) > 0:
                pf.update(observations)
            pf.resample()
            est_pos = pf.estimate()
        
        # 5. Visuals
        vis_img = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
        if pf.is_initialized:
            for p in pf.particles:
                cv2.circle(vis_img, (int(p.mu[0]), int(p.mu[1])), 1, (0, 255, 0), -1)
            cv2.circle(vis_img, (int(est_pos[0]), int(est_pos[1])), 8, (0, 0, 255), -1)

        cv2.rectangle(vis_img, (int(true_x)-10, int(true_y)-10), 
                      (int(true_x)+10, int(true_y)+10), (255, 0, 0), 2)
        
        cv2.putText(vis_img, f"Mode: {status_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow("Map", vis_img)
        cv2.imshow("Detectron Mask Input", detectron_binary_output)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate, scale

# ==============================================================================
# PHẦN 1: CẤU TRÚC DỮ LIỆU CƠ BẢN
# ==============================================================================

class ShapePoint:
    """
    Đại diện cho một vật thể hình học (Tòa nhà, Ao hồ...)
    """
    def __init__(self, id, polygon, raw_contour, source_type):
        self.id = id
        self.region = polygon       # Shapely Polygon (Dùng để tính toán giao cắt)
        self.raw_contour = raw_contour # OpenCV Contour (Dùng để tham chiếu nếu cần)
        self.source = source_type   # "Drone" hoặc "Map"
        
        # Đặc trưng hình học
        self.S = polygon.area
        self.centroid = np.array([polygon.centroid.x, polygon.centroid.y])

class GaussianObservation:
    """
    Kết quả của bước Matching -> Đầu vào cho GMM Filter
    Tương ứng với Công thức (5) trong tài liệu.
    """
    def __init__(self, mean_pos, alpha_score):
        self.mean = np.array(mean_pos) # Vị trí ước lượng (mu)
        
        # Sigma (Covariance Matrix) tỉ lệ nghịch với Alpha
        # Alpha càng lớn (khớp càng tốt) -> Sigma càng nhỏ (độ chụm cao)
        base_error = 5.0 # Sai số cơ bản (mét/pixel)
        sigma_val = base_error / (alpha_score + 1e-6)
        self.cov = np.array([[sigma_val, 0], [0, sigma_val]]) # Ma trận hiệp phương sai
        
        # Pre-compute để tính toán nhanh
        self.inv_cov = np.linalg.inv(self.cov)
        self.det_cov = np.linalg.det(self.cov)

    def pdf(self, x):
        """Tính mật độ xác suất tại vị trí x (Gaussian PDF)"""
        diff = x - self.mean
        norm_const = 1.0 / (2 * np.pi * np.sqrt(self.det_cov))
        exponent = -0.5 * (diff.T @ self.inv_cov @ diff)
        return norm_const * np.exp(exponent)

# ==============================================================================
# PHẦN 2: XỬ LÝ ẢNH (IMAGE LOADER)
# ==============================================================================

class BinaryImageLoader:
    def extract_features(self, image_path, source_type, scale_ratio=1.0):
        # 1. Đọc ảnh
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"ERROR: Không tìm thấy ảnh {image_path}")
            return []

        # 2. Xử lý nền (Nếu nền trắng -> Đảo ngược thành nền đen)
        if img[0, 0] > 127:
            img = cv2.bitwise_not(img)

        # 3. Tìm đường bao (Contours)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []

        for i, cnt in enumerate(contours):
            # Lọc nhiễu nhỏ
            if cv2.contourArea(cnt) < 50: continue 

            # 4. Xấp xỉ đa giác (Polygon Approximation)
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) >= 3:
                pts = [tuple(pt[0]) for pt in approx]
                poly = Polygon(pts)
                
                # Sửa lỗi topology
                if not poly.is_valid: poly = poly.buffer(0)
                
                # Rescale nếu cần (Đồng bộ tỷ lệ Drone vs Map)
                if scale_ratio != 1.0:
                    poly = scale(poly, xfact=scale_ratio, yfact=scale_ratio, origin=(0,0))

                shapes.append(ShapePoint(f"{source_type}_{i}", poly, cnt, source_type))

        print(f"[{source_type}] Đã trích xuất {len(shapes)} vật thể.")
        return shapes

# ==============================================================================
# PHẦN 3: HỆ THỐNG KHỚP ẢNH (ROBUST MATCHING SYSTEM)
# ==============================================================================

class RobustMatchingSystem:
    def __init__(self, drone_view_size=(200, 200)):
        self.view_w, self.view_h = drone_view_size
        # Tạo box đại diện khung hình camera (gốc tại 0,0)
        self.v_box_origin = box(0, 0, self.view_w, self.view_h)

    def calculate_alpha_formula_4(self, poly_cam, poly_ref, v_box_current):
        """
        Tính Alpha theo đúng Công thức (4) trong bài báo
        """
        try:
            # Tử số: intersection(V_cam, V_ref)
            numerator = poly_cam.intersection(poly_ref).area
            if numerator == 0: return 0.0
            
            # Mẫu số: sqrt( intersection(V_box, V_ref) * intersection(V_box, V_cam) )
            term1 = v_box_current.intersection(poly_ref).area
            term2 = v_box_current.intersection(poly_cam).area # Thường chính là diện tích poly_cam
            
            denominator = np.sqrt(term1 * term2 + 1e-6)
            return numerator / denominator
        except:
            return 0.0

    def find_best_alignment_hill_climbing(self, drone_poly, map_poly):
        """
        Thuật toán Leo đồi để tìm vị trí tối ưu (Khắc phục lệch tâm)
        """
        # 1. Khởi tạo: Dời tâm trùng tâm
        dx_init = map_poly.centroid.x - drone_poly.centroid.x
        dy_init = map_poly.centroid.y - drone_poly.centroid.y
        
        current_poly = translate(drone_poly, xoff=dx_init, yoff=dy_init)
        best_poly = current_poly
        best_inter = current_poly.intersection(map_poly).area
        
        # 2. Tinh chỉnh vị trí
        step = np.sqrt(map_poly.area) / 10.0 # Bước nhảy ban đầu
        current_offset = [dx_init, dy_init]
        
        for _ in range(15): # 15 vòng lặp tối ưu
            found_better = False
            # Thử 4 hướng
            for mx, my in [(0, step), (0, -step), (step, 0), (-step, 0)]:
                test_poly = translate(best_poly, xoff=mx, yoff=my)
                test_inter = test_poly.intersection(map_poly).area
                
                if test_inter > best_inter:
                    best_inter = test_inter
                    best_poly = test_poly
                    current_offset[0] += mx
                    current_offset[1] += my
                    found_better = True
                    break # Greedy
            
            if not found_better:
                step *= 0.5 # Giảm bước nhảy để dò kỹ hơn
                if step < 0.5: break

        return best_poly, tuple(current_offset)

    def run(self, drone_objs, map_objs):
        observations = []
        print("\n--- BẮT ĐẦU MATCHING ---")
        
        for d_obj in drone_objs:
            best_alpha = 0
            best_target = None
            final_offset = (0,0)
            
            # 1. Sàng lọc thô (Coarse Screening)
            # Bài báo: |S^A - S^B|/m < delta
            candidates = [m for m in map_objs if abs(d_obj.S - m.S)/(m.S+1e-5) < 0.5]
            
            # 2. Khớp tinh (Fine Matching & Alignment)
            for m_obj in candidates:
                # Tìm vị trí tối ưu
                aligned_poly, offset = self.find_best_alignment_hill_climbing(d_obj.region, m_obj.region)
                
                # Tạo V_box tại vị trí tương ứng trên bản đồ để tính công thức (4)
                # Dịch chuyển V_box theo offset tìm được
                # Tâm ảnh drone ban đầu là (view_w/2, view_h/2) -> centroid của d_obj
                # Offset là từ centroid d_obj -> centroid m_obj (đã tối ưu)
                # => V_box cũng dịch theo offset đó
                v_box_aligned = translate(self.v_box_origin, xoff=offset[0], yoff=offset[1])
                
                alpha = self.calculate_alpha_formula_4(aligned_poly, m_obj.region, v_box_aligned)
                
                if alpha > best_alpha:
                    best_alpha = alpha
                    best_target = m_obj
                    final_offset = offset
            
            if best_target and best_alpha > 0.4:
                # Tính vị trí ước lượng của Drone
                # Vị trí thực = Vị trí tìm được trừ đi tọa độ cục bộ ban đầu
                # Giả sử ta muốn tìm tọa độ của gốc ảnh (0,0) trên bản đồ:
                est_x = 0 + final_offset[0]
                est_y = 0 + final_offset[1]
                
                print(f" ✅ KHỚP: Drone '{d_obj.id}' <-> Map '{best_target.id}' | Alpha={best_alpha:.3f}")
                
                # Tạo Observation cho GMM
                obs = GaussianObservation([est_x, est_y], best_alpha)
                observations.append(obs)

        return observations

# ==============================================================================
# PHẦN 4: BỘ LỌC HẠT GMM (GMM PARTICLE FILTER)
# ==============================================================================

class Particle:
    def __init__(self, pos, weight):
        self.mu = np.array(pos)
        self.w = weight
        self.c = 0 # Số lần khớp thành công (Counter)
        self.sigma = np.eye(2) * 2.0 # Độ bất định riêng của hạt

class GMMParticleFilter:
    def __init__(self, num_particles=1000, map_bounds=(0, 1000, 0, 1000)):
        self.N = num_particles
        self.particles = []
        self.bounds = map_bounds # (min_x, max_x, min_y, max_y)
        
    def initialize(self):
        """Khởi tạo hạt rải đều ngẫu nhiên (Global Localization)"""
        print(f"--- KHỞI TẠO {self.N} HẠT ---")
        for _ in range(self.N):
            rx = np.random.uniform(self.bounds[0], self.bounds[1])
            ry = np.random.uniform(self.bounds[2], self.bounds[3])
            self.particles.append(Particle([rx, ry], 1.0/self.N))

    def predict(self, move_vector, noise_std=1.0):
        """Bước dự đoán (Motion Update)"""
        move = np.array(move_vector)
        noise_cov = np.eye(2) * (noise_std**2)
        
        for p in self.particles:
            p.mu += move
            p.mu += np.random.multivariate_normal([0,0], noise_cov)
            p.sigma += noise_cov

    def update(self, observations):
        """Bước cập nhật trọng số dựa trên Gaussian Observations"""
        if not observations: return

        total_w = 0.0
        for p in self.particles:
            best_prob = 0
            matched = False
            
            for obs in observations:
                dist = np.linalg.norm(p.mu - obs.mean)
                
                # Điều kiện khớp thành công (tài liệu): 
                # dist < sigma_p + sigma_obs (Dùng trace đại diện)
                thresh = np.sqrt(np.trace(p.sigma)) + np.sqrt(np.trace(obs.cov))
                
                if dist < thresh:
                    matched = True
                    prob = obs.pdf(p.mu)
                    if prob > best_prob: best_prob = prob
            
            if matched:
                p.c += 1
                p.w *= best_prob * (1 + 0.1 * p.c) # Thưởng cho lịch sử khớp
            else:
                p.w *= 1e-10 # Phạt nặng
                
            total_w += p.w
            
        # Chuẩn hóa trọng số
        if total_w > 0:
            for p in self.particles: p.w /= total_w
        else:
            # Reset nếu mất dấu
            for p in self.particles: p.w = 1.0/self.N

    def resample(self):
        """Lấy mẫu lại (Resampling)"""
        weights = [p.w for p in self.particles]
        indices = np.random.choice(range(self.N), size=self.N, p=weights)
        new_particles = []
        for i in indices:
            old = self.particles[i]
            # Thêm nhiễu nhỏ (jitter) để tránh suy thoái hạt
            new_pos = old.mu + np.random.normal(0, 0.5, 2)
            new_p = Particle(new_pos, 1.0/self.N)
            new_p.c = old.c
            new_particles.append(new_p)
        self.particles = new_particles

    def estimate(self):
        """Trả về vị trí trung bình có trọng số"""
        x, y = 0.0, 0.0
        for p in self.particles:
            x += p.mu[0] * p.w
            y += p.mu[1] * p.w
        return np.array([x, y])

# ==============================================================================
# PHẦN 5: CHƯƠNG TRÌNH CHÍNH (MAIN SIMULATION)
# ==============================================================================

if __name__ == "__main__":
    # --- A. TẠO DỮ LIỆU GIẢ LẬP (Để test code) ---
    print(">>> Đang tạo dữ liệu giả lập...")
    # Map: 1000x1000, có 1 hình vuông đen tại (500, 500)
    map_img = np.ones((1000, 1000), dtype=np.uint8) * 255
    cv2.rectangle(map_img, (500, 500), (600, 600), 0, -1) 
    cv2.imwrite("sim_map.png", map_img)
    
    # Drone: Ảnh 200x200, nhìn thấy hình vuông đó nhưng lệch
    # Giả sử drone đang ở vị trí (450, 450) trên bản đồ
    # Thì hình vuông (500,500) sẽ xuất hiện tại (50, 50) trong ảnh drone
    drone_img = np.ones((200, 200), dtype=np.uint8) * 255
    cv2.rectangle(drone_img, (50, 50), (150, 150), 0, -1)
    cv2.imwrite("sim_drone.png", drone_img)
    # -----------------------------------------------

    # --- B. CHẠY HỆ THỐNG ---
    
    # 1. Load Ảnh
    loader = BinaryImageLoader()
    map_objs = loader.extract_features("sim_map.png", "MAP")
    drone_objs = loader.extract_features("sim_drone.png", "DRONE")
    
    # 2. Khởi tạo Filter
    pf = GMMParticleFilter(num_particles=500, map_bounds=(0, 1000, 0, 1000))
    pf.initialize()
    
    # 3. Chạy Matching
    matcher = RobustMatchingSystem(drone_view_size=(200, 200))
    observations = matcher.run(drone_objs, map_objs)
    
    # 4. Cập nhật Filter với kết quả Matching
    if observations:
        print(f"\n>>> Tìm thấy {len(observations)} quan sát. Cập nhật Filter...")
        pf.update(observations)
        pf.resample()
        
        est_pos = pf.estimate()
        print(f"\n✅ KẾT QUẢ CUỐI CÙNG:")
        print(f"   Vị trí thực tế (Giả định): [450.0, 450.0]")
        print(f"   Vị trí thuật toán đoán:    [{est_pos[0]:.2f}, {est_pos[1]:.2f}]")
        print(f"   Sai số: {np.linalg.norm(est_pos - np.array([450, 450])):.2f} mét")
    else:
        print("\n❌ Không tìm thấy vị trí phù hợp.")

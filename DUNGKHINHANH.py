import cv2 
import numpy as np 
import math 
import os
from shapely.geometry import Polygon, Point, box 
from shapely.affinity import translate, rotate, scale
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal 

# ============================================================================== 
# 0. VISUALIZATION HELPERS
# ============================================================================== 
def coords_to_cv_pts(poly, offset_x=0, offset_y=0):
    """Chuyển đổi Shapely Polygon thành mảng điểm OpenCV để vẽ."""
    if poly is None or poly.is_empty: return None
    try:
        # Lấy tọa độ exterior
        x, y = poly.exterior.xy
        pts = []
        for i in range(len(x)):
            pts.append([int(x[i] + offset_x), int(y[i] + offset_y)])
        return np.array(pts, np.int32).reshape((-1, 1, 2))
    except:
        return None

def draw_match_debug(img, ref_poly, cam_poly_rotated, box_poly_rotated, alpha):
    """Vẽ minh họa quá trình matching lên bản đồ."""
    # 1. Vẽ Reference (Xanh lá) - Tòa nhà trên map
    pts_ref = coords_to_cv_pts(ref_poly)
    if pts_ref is not None:
        cv2.polylines(img, [pts_ref], True, (0, 255, 0), 3)
    
    # 2. Vẽ Box (Đỏ) - Khung nhìn camera đã xoay
    pts_box = coords_to_cv_pts(box_poly_rotated)
    if pts_box is not None:
        cv2.polylines(img, [pts_box], True, (0, 0, 255), 2)

    # 3. Vẽ Camera Poly (Xanh dương) - Input đã xoay
    pts_cam = coords_to_cv_pts(cam_poly_rotated)
    if pts_cam is not None:
        cv2.polylines(img, [pts_cam], True, (255, 0, 0), 2)
        
        # Vẽ tâm và hiển thị Alpha
        M = cv2.moments(pts_cam)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(img, f"a:{alpha:.2f}", (cX, cY - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# ============================================================================== 
# 1. HELPER FUNCTIONS
# ============================================================================== 
def get_clean_polygons(binary_img, min_area=300):
    h_img, w_img = binary_img.shape
    cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            if x < 5 or y < 5 or (x + w) > (w_img - 5) or (y + h) > (h_img - 5):
                continue 
            approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c,True), True)
            if len(approx) >= 3:
                p = Polygon(approx.reshape(-1, 2))
                if not p.is_valid: p = p.buffer(0)
                polys.append(p)
    polys.sort(key=lambda x: x.area, reverse=True)
    return polys

def auto_fix_scale(cam_polys, map_polys):
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
# 2. CLASS PARTICLE FILTER (GMM INTEGRATED)
# ============================================================================== 
class PaperCompliantPF: 
    def __init__(self, N, W, H): 
        self.N = N; self.W = W; self.H = H 
        self.particles = np.zeros((N, 2))
        self.weights = np.ones(N) / N
        self.initialized = False 
        self.global_cov = np.eye(2) * 100 
        self.process_noise_std = 2.0  

    def init(self, x, y, initial_cov=50): 
        self.particles[:, 0] = np.random.normal(x, initial_cov, self.N) 
        self.particles[:, 1] = np.random.normal(y, initial_cov, self.N) 
        self.initialized = True 

    def propagate(self, control_input): 
        noise = np.random.normal(0, self.process_noise_std, (self.N, 2)) 
        self.particles += control_input + noise 
        np.clip(self.particles[:, 0], 0, self.W, out=self.particles[:, 0]) 
        np.clip(self.particles[:, 1], 0, self.H, out=self.particles[:, 1]) 

    def update_with_gmm(self, gmm_data): 
        if not gmm_data: return 
        total_likelihoods = np.zeros(self.N) 
        for comp in gmm_data: 
            mu = comp['mu']
            cov = comp['cov']
            alpha = comp['alpha']
            rv = multivariate_normal(mu, cov) 
            total_likelihoods += alpha * rv.pdf(self.particles) 
        self.weights *= total_likelihoods 
        weight_sum = np.sum(self.weights)
        if weight_sum > 0: self.weights /= weight_sum
        else: self.weights.fill(1.0 / self.N)

    def resample(self): 
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
        self.weights.fill(1.0 / self.N)

    # --- HÀM GMM MÀ BẠN CẦN (ĐÃ CÓ TRONG CLASS) ---
    def estimate_and_evaluate(self):
        if self.N < 10: return np.mean(self.particles, axis=0), -1.0
        try:
            # GMM với reg_covar để tránh lỗi Singular Matrix
            gmm = GaussianMixture(n_components=4, covariance_type='full', reg_covar=1e-3, random_state=42)
            gmm.fit(self.particles)
            best_idx = np.argmax(gmm.weights_)
            
            estimated_pos = gmm.means_[best_idx]
            cov_matrix = gmm.covariances_[best_idx]
            
            det_cov = np.linalg.det(cov_matrix)
            sqrt_det_cov = np.sqrt(abs(det_cov)) if abs(det_cov) > 1e-9 else 1e-9
            eigenvalues, _ = np.linalg.eig(cov_matrix)
            max_sigma = np.sqrt(np.max(eigenvalues))
            max_dist = 3 * max_sigma 
            if max_dist < 1e-3: max_dist = 1e-3
            val_inside = (10 * sqrt_det_cov) / max_dist
            val_log = math.log10(val_inside + 1e-9)
            S_t = 1.0 if val_log > 0 else -1.0
            return estimated_pos, S_t
        except Exception as e:
            # Plan B: Top 20%
            indices = np.argsort(self.weights)[-int(self.N * 0.2):] 
            top_particles = self.particles[indices]
            top_weights = self.weights[indices]
            estimated_pos = np.average(top_particles, weights=top_weights, axis=0)
            return estimated_pos, -1.0

# ============================================================================== 
# 3. CLASS CFBVM MATCHER (STRICT EQ 4 & DEBUG OUTPUT)
# ============================================================================== 
class CFBVMMatcher:
    def __init__(self, map_polys):
        self.map_data = []
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
        for a in np.linspace(math.radians(a1), math.radians(a2), 15):
            pts.append((r2*np.cos(a), r2*np.sin(a)))
        if r1 > 0:
            for a in reversed(np.linspace(math.radians(a1), math.radians(a2), 15)):
                pts.append((r1*np.cos(a), r1*np.sin(a)))
        else:
            pts.append((0,0))
        return Polygon(pts)

    def compute_shape_vector(self, poly):
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
        mat_a = vec_a.reshape(3, 8)
        mat_b = vec_b.reshape(3, 8)
        min_dist = float('inf')
        for shift in range(8):
            b_shifted = np.roll(mat_b, shift, axis=1)
            dist = np.sum(np.abs(mat_a - b_shifted))
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def compute_correlation(self, cam_poly, ref_poly, img_w=400, img_h=400):
        # 1. Box tâm trùng tâm tòa nhà
        cx, cy = cam_poly.centroid.x, cam_poly.centroid.y
        v_box = box(cx - img_w/2, cy - img_h/2, cx + img_w/2, cy + img_h/2)
        
        # 2. Dịch về (0,0)
        c_ref = ref_poly.centroid
        p_ref_centered = translate(ref_poly, -c_ref.x, -c_ref.y)
        p_cam_centered = translate(cam_poly, -cx, -cy)
        p_box_centered = translate(v_box, -cx, -cy)

        best_alpha = 0.0
        best_angle = 0.0
        
        # Term 2: intersection(V_box, V_cam)
        term2 = p_box_centered.intersection(p_cam_centered).area 
        if term2 == 0: return 0.0, 0.0, None, None

        for ang in np.arange(-180, 180, 5): 
            r_cam = rotate(p_cam_centered, ang, origin=(0,0))
            r_box = rotate(p_box_centered, ang, origin=(0,0))
            
            inter_cam_ref = r_cam.intersection(p_ref_centered).area
            
            if inter_cam_ref > 0:
                # EQ 4: Sqrt(inter(box, ref) * inter(box, cam))
                term1 = r_box.intersection(p_ref_centered).area
                denom = math.sqrt(term1 * term2)
                if denom > 0:
                    alpha = inter_cam_ref / denom
                    if alpha > best_alpha:
                        best_alpha = alpha
                        best_angle = ang
        
        # Tính geometry debug ở góc tốt nhất
        final_cam_rot = rotate(p_cam_centered, best_angle, origin=(0,0))
        final_box_rot = rotate(p_box_centered, best_angle, origin=(0,0))
        
        return best_alpha, best_angle, final_cam_rot, final_box_rot

    def process(self, cam_poly, cam_center, est_pos, search_radius=500, img_w=400, img_h=400):
        if cam_poly is None: return []

        cam_poly = cam_poly.simplify(0.5, preserve_topology=True)
        cam_vec = self.compute_shape_vector(cam_poly)
        if np.sum(cam_vec) == 0: return []

        results = []
        rel_vec_x = cam_poly.centroid.x - cam_center[0]
        rel_vec_y = cam_poly.centroid.y - cam_center[1]

        for item in self.map_data:
            if est_pos is not None:
                dist = np.linalg.norm(np.array(item['centroid']) - est_pos)
                if dist > search_radius: continue

            if 'area' in item:
                ratio = cam_poly.area / (item['area'] + 1e-5)
                if ratio < 0.3 or ratio > 3.0: continue

            if self.coarse_distance(cam_vec, item['vector']) > 0.35: 
                continue

            alpha, angle, r_cam, r_box = self.compute_correlation(cam_poly, item['poly'], img_w, img_h)
            
            if alpha > 0.65:
                pt_rel = Point(rel_vec_x, rel_vec_y)
                pt_rotated = rotate(pt_rel, angle, origin=(0,0))
                
                mu_x = item['centroid'][0] - pt_rotated.x
                mu_y = item['centroid'][1] - pt_rotated.y
                mu = np.array([mu_x, mu_y])
                cov = np.array([[100.0, 0], [0, 100.0]])
                
                # Dịch chuyển debug geometry về vị trí thật trên bản đồ
                ref_cx, ref_cy = item['centroid']
                abs_cam_poly = translate(r_cam, ref_cx, ref_cy)
                abs_box_poly = translate(r_box, ref_cx, ref_cy)
                
                results.append({
                    'mu': mu, 'cov': cov, 'alpha': alpha,
                    'ref_poly': item['poly'],
                    'debug_cam': abs_cam_poly,
                    'debug_box': abs_box_poly
                })
                
        return results

# ============================================================================== 
# 4. MAIN PROGRAM
# ============================================================================== 
def main(): 
    # --- CONFIG ---
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MAP_PATH = os.path.join(CURRENT_DIR, "ref snazzy.png")
    MASK_PATH = os.path.join(CURRENT_DIR, "2.png") # Thay đổi ảnh input tại đây
    DEBUG_ALPHA_THRESHOLD = 0.92
    # --- LOAD MAP ---
    print("[SYSTEM] Loading Map...")
    map_img_color = cv2.imread(MAP_PATH)
    if map_img_color is None: return print("ERR: Map not found.")
    map_gray = cv2.cvtColor(map_img_color, cv2.COLOR_BGR2GRAY)
    _, map_thresh = cv2.threshold(map_gray, 127, 255, cv2.THRESH_BINARY)
    H, W = map_thresh.shape 
    map_polys = get_clean_polygons(map_thresh, min_area=300)
    print(f"[SYSTEM] Loaded {len(map_polys)} buildings.")

    # --- LOAD INPUT ---
    mask_input = cv2.imread(MASK_PATH, 0)
    if mask_input is None: return print("ERR: Input Mask not found.")
    mask_input = cv2.copyMakeBorder(mask_input, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
    h_in, w_in = mask_input.shape
    scale_down = 0.15 if w_in > 1000 else 1.0 
    new_w, new_h = int(w_in * scale_down), int(h_in * scale_down)
    mask_input = cv2.resize(mask_input, (new_w, new_h))
    cam_center_x, cam_center_y = new_w // 2, new_h // 2
    
    _, proc_frame = cv2.threshold(mask_input, 127, 255, cv2.THRESH_BINARY)
    raw_cam_polys = get_clean_polygons(proc_frame, min_area=200)
    if not raw_cam_polys: return print("ERR: No building in input.")

    top_map_polys = map_polys[:150] 
    visible_polys, detected_scale = auto_fix_scale(raw_cam_polys, top_map_polys)
    print(f"[INFO] Scale: {detected_scale:.3f}")

    # --- SETUP PF ---
    pf = PaperCompliantPF(N=3000, W=W, H=H)
    matcher = CFBVMMatcher(map_polys) 
    
    # Init guess
    best_pos = (W/2, H/2)
    max_score = 0
    target = visible_polys[0]
    
    for item in matcher.map_data:
        if 'area' not in item: continue
        ratio = target.area / (item['area'] + 1e-5)
        if 0.5 < ratio < 2.0: 
            alpha, _, _, _ = matcher.compute_correlation(target, item['poly'])
            if alpha > max_score:
                max_score = alpha
                best_pos = item['centroid']
                if alpha > 0.8: break
    
    init_std = 50 if max_score > 0.6 else 400
    pf.init(best_pos[0], best_pos[1], initial_cov=init_std)
    
    vis_map = map_img_color.copy()
    simulated_control = np.array([0.0, 0.0]) 

    while True: 
        pf.propagate(simulated_control) 
        est_pos, _ = pf.estimate_and_evaluate() 
        
        all_matches = [] 
        
        # --- VISUALIZATION BUFFER ---
        vis = vis_map.copy()

        for i, d_poly in enumerate(visible_polys): 
            matches = matcher.process(d_poly, (cam_center_x, cam_center_y), est_pos, 
                                      search_radius=500, img_w=new_w, img_h=new_h)
            all_matches.extend(matches) 
            
            # --- VẼ BOUNDARY BOX & DEBUG ---
            for m in matches:
                if m['alpha'] >= DEBUG_ALPHA_THRESHOLD:
                    draw_match_debug(vis, m['ref_poly'], m['debug_cam'], m['debug_box'], m['alpha'])

        pf.update_with_gmm(all_matches) 
        pf.resample() 
        final_pos, S_t = pf.estimate_and_evaluate() 

        # Draw Particles
        #for p in pf.particles: 
            #cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1) 
        
        # Draw Status
        #color = (0, 0, 255) if S_t > 0 else (0, 255, 255)
        text = "LOCKED" if S_t > 0 else "SEARCHING"
        #cv2.circle(vis, (int(final_pos[0]), int(final_pos[1])), 10, color, -1) 
        #cv2.putText(vis, f"{text} (St:{S_t:.0f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw PIP
        pip = cv2.cvtColor(proc_frame, cv2.COLOR_GRAY2BGR)
        pip = cv2.resize(pip, (200, 200)) 
        vis[H-210:H-10, 10:210] = pip
        cv2.rectangle(vis, (10, H-210), (210, H-10), (0,255,255), 2)
        
        if H > 900: show_vis = cv2.resize(vis, (int(W*0.7), int(H*0.7)))
        else: show_vis = vis
        
        cv2.imshow("CFBVM - Full Debug", show_vis) 
        if cv2.waitKey(100) == 27: break 

    cv2.destroyAllWindows() 

if __name__ == "__main__": 
    main()

import cv2
import numpy as np
import math
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from scipy.stats import multivariate_normal

# ==============================================================================
# ======================= USER INPUT (CHỈ SỬA Ở ĐÂY) ============================
# ==============================================================================
REF_MAP_PATH = "ref_map_binary.png"   # ảnh map nhị phân toàn cục
CAM_PATH     = "cam_binary.png"       # ảnh camera đã binary (Detectron2)
# ==============================================================================


# ==============================================================================
# ======================= PARTICLE FILTER (NGUYÊN BẢN) ==========================
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

    def propagate(self, velocity=np.array([0.0, 0.0])):
        noise = np.random.normal(0, self.process_noise, (self.N, 2))
        self.particles += velocity + noise
        np.clip(self.particles[:, 0], 0, self.W, out=self.particles[:, 0])
        np.clip(self.particles[:, 1], 0, self.H, out=self.particles[:, 1])

    def estimate_and_evaluate(self):
        estimated_pos = np.average(self.particles, weights=self.weights, axis=0)
        n = 3
        valid_indices = self.counts > n
        
        if np.sum(valid_indices) == 0:
            S_t = -1.0
        else:
            sum_w_valid = np.sum(self.weights[valid_indices])
            sum_w_invalid = np.sum(self.weights[~valid_indices]) + 1.e-10
            ratio = sum_w_valid / sum_w_invalid
            S_t = 1.0 if ratio > 1.0 else -1.0
        
        is_credible = (S_t > 0)
        return estimated_pos, is_credible, S_t

    def update_with_gmm(self, gmm_data):
        if not gmm_data:
            self.counts = np.maximum(0, self.counts - 1)
            return

        current_mean = np.average(self.particles, weights=self.weights, axis=0)
        valid_components = []
        
        for comp in gmm_data:
            dist = np.linalg.norm(comp['mu'] - current_mean)
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

    def resample(self):
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.N * 0.5:
            indices = np.random.choice(self.N, self.N, p=self.weights)
            self.particles = self.particles[indices]
            self.counts = self.counts[indices]
            self.weights.fill(1.0 / self.N)


# ==============================================================================
# ======================= CFBVM MATCHER (NGUYÊN BẢN) =============================
# ==============================================================================
class CFBVMMatcher:
    def __init__(self, map_polys):
        self.map_data = [] 
        self.RADIUS_LEVELS = [30, 60, 90] 
        self.NUM_SECTORS = 8 
        self.sector_masks = self._precompute_sector_masks()

        for poly in map_polys:
            vec = self.compute_shape_vector(poly)
            if np.sum(vec) > 0:
                self.map_data.append({
                    'poly': poly,
                    'vector': vec,
                    'centroid': (poly.centroid.x, poly.centroid.y)
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
        return v / np.sum(v) if np.sum(v) > 0 else v

    def coarse_distance(self, a, b):
        return np.sum(np.abs(a - b))

    def compute_correlation(self, cam_poly, ref_poly):
        p1 = translate(cam_poly, -cam_poly.centroid.x, -cam_poly.centroid.y)
        p2 = translate(ref_poly, -ref_poly.centroid.x, -ref_poly.centroid.y)
        best = 0
        for ang in [-5, 0, 5]:
            r = rotate(p1, ang, origin=(0,0))
            inter = r.intersection(p2).area
            if inter > 0:
                a = inter / math.sqrt(r.area * p2.area)
                best = max(best, a)
        return best

    def process(self, cam_poly, est_pos, search_radius=300):
        if cam_poly is None:
            return []

        cam_vec = self.compute_shape_vector(cam_poly)
        if np.sum(cam_vec) == 0:
            return []

        results = []
        shift = np.array([cam_poly.centroid.x - 200,
                          cam_poly.centroid.y - 200])

        for item in self.map_data:
            if est_pos is not None:
                if np.linalg.norm(np.array(item['centroid']) - est_pos) > search_radius:
                    continue

            if self.coarse_distance(cam_vec, item['vector']) > 0.6:
                continue

            alpha = self.compute_correlation(cam_poly, item['poly'])
            if alpha > 0.6:
                mu = np.array(item['centroid']) - shift
                sigma = 30.0 * (1.0 - alpha) + 5.0
                cov = np.array([[sigma**2,0],[0,sigma**2]])
                results.append({'mu': mu, 'cov': cov, 'alpha': alpha})
        return results


# ==============================================================================
# =============================== MAIN ==========================================
# ==============================================================================
def main():
    # ---- LOAD REF MAP ----
    ref = cv2.imread(REF_MAP_PATH, 0)
    _, ref = cv2.threshold(ref, 127, 255, cv2.THRESH_BINARY)
    H, W = ref.shape

    cnts,_ = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    map_polys = []
    for c in cnts:
        if cv2.contourArea(c) > 800:
            p = Polygon(c.reshape(-1,2))
            if not p.is_valid: p = p.buffer(0)
            map_polys.append(p)

    # ---- LOAD CAM ----
    cam = cv2.imread(CAM_PATH, 0)
    _, cam = cv2.threshold(cam, 127, 255, cv2.THRESH_BINARY)
    cs,_ = cv2.findContours(cam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cam_poly = None
    if cs:
        cam_poly = Polygon(max(cs, key=cv2.contourArea).reshape(-1,2))

    # ---- INIT ----
    matcher = CFBVMMatcher(map_polys)
    pf = PaperCompliantPF(3000, W, H)
    pf.init(W//2, H//2)

    # ---- RUN ----
    for _ in range(10):
        pf.propagate()
        est,_,_ = pf.estimate_and_evaluate()
        gmm = matcher.process(cam_poly, est)
        pf.update_with_gmm(gmm)
        pf.resample()

    # ---- VIS ----
    vis = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    for p in pf.particles:
        cv2.circle(vis, (int(p[0]), int(p[1])), 1, (0,255,0), -1)

    est,cred,score = pf.estimate_and_evaluate()
    cv2.circle(vis, (int(est[0]), int(est[1])), 8, (0,0,255), -1)
    cv2.putText(vis, f"S={score}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("REF MAP + PF RESULT", vis)
    cv2.imshow("CAM", cam)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

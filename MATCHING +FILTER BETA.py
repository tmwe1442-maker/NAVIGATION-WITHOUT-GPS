import cv2
import numpy as np
import math
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate
import sys

# ==============================================================================
# 0. CONTROLLER
# ==============================================================================
try:
    from MAPPING import DroneController
    print("[INFO] DroneController loaded")
except ImportError:
    print("Missing MAPPING.py")
    sys.exit()

# ==============================================================================
# 1. PARTICLE FILTER – STRICT PAPER COMPLIANCE
# ==============================================================================
class PaperCompliantPF:
    def __init__(self, N, W, H):
        self.N = N
        self.W = W
        self.H = H

        self.particles = np.zeros((N, 2))
        self.weights = np.ones(N) / N

        # Eq.(7)
        self.C = np.ones(N, dtype=int)

        self.process_noise = 2.0
        self.prev_est_pos = None
        self.last_velocity = np.zeros(2)

    def init(self, x, y):
        self.particles[:, 0] = np.random.normal(x, 50, self.N)
        self.particles[:, 1] = np.random.normal(y, 50, self.N)
        self.prev_est_pos = np.array([x, y])

    def propagate(self, velocity, dt=1.0):
        self.last_velocity = velocity.copy()
        noise = np.random.normal(0, self.process_noise, (self.N, 2))
        self.particles += velocity * dt + noise

        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.W)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.H)

    def update_gmm(self, gmm_components):
        if not gmm_components:
            return

        likelihood_sum = np.zeros(self.N)
        matched = np.zeros(self.N, dtype=bool)

        for comp in gmm_components:
            mu = np.array(comp['mu'])
            sigma = comp['sigma']
            alpha = comp['alpha']

            d = np.linalg.norm(self.particles - mu, axis=1)

            norm = 1.0 / (2 * np.pi * sigma**2)
            likelihood = norm * np.exp(-(d**2) / (2 * sigma**2))

            likelihood_sum += alpha * likelihood

            # [FIX] matched successfully = high likelihood
            matched |= (likelihood > 0.5 * np.max(likelihood))

        # Eq.(7) counter update
        self.C[matched] += 1
        self.C = np.maximum(self.C, 1)

        # weight update
        self.weights *= likelihood_sum
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.N * 0.6:
            idx = np.random.choice(self.N, self.N, p=self.weights)
            self.particles = self.particles[idx]
            self.C = self.C[idx]
            self.weights.fill(1.0 / self.N)
            self.particles += np.random.normal(0, 1.0, (self.N, 2))

    def estimate(self):
        max_c = np.max(self.C)
        n = max(max_c - 1, 3)

        valid = self.C > n

        if np.sum(valid) == 0:
            mean = np.average(self.particles, weights=self.weights, axis=0)
            St = 0.0
        else:
            num = np.sum(self.weights[valid] * self.C[valid])
            den = np.sum(self.weights * self.C)
            St = num / (den + 1e-10)

            w = self.weights[valid]
            w /= np.sum(w)
            mean = np.average(self.particles[valid], weights=w, axis=0)

        self.prev_est_pos = mean
        return mean, St > 0.95, St


# ==============================================================================
# 2. CFBVM MATCHER – STRICT PAPER
# ==============================================================================
class CFBVMMatcher:
    def __init__(self, map_polys):
        self.R = [20, 40, 60]
        self.S = 8
        self.sectors = self._build_sectors()

        self.map_data = []
        for p in map_polys:
            v = self.compute_vector(p)
            if np.sum(v) > 0:
                self.map_data.append({
                    'poly': p,
                    'vector': v,
                    'centroid': np.array(p.centroid.coords[0])
                })

    def _build_sectors(self):
        sectors = []
        prev = 0
        for r in self.R:
            ring = []
            for i in range(self.S):
                a1 = i * 360 / self.S
                a2 = (i+1) * 360 / self.S
                ring.append(self._sector(prev, r, a1, a2))
            sectors.append(ring)
            prev = r
        return sectors

    def _sector(self, r1, r2, a1, a2):
        pts = []
        for a in np.linspace(math.radians(a1), math.radians(a2), 20):
            pts.append((r2*np.cos(a), r2*np.sin(a)))
        if r1 > 0:
            for a in reversed(np.linspace(math.radians(a1), math.radians(a2), 20)):
                pts.append((r1*np.cos(a), r1*np.sin(a)))
        else:
            pts.append((0,0))
        return Polygon(pts)

    def compute_vector(self, poly):
        cx, cy = poly.centroid.xy
        p = translate(poly, -cx[0], -cy[0])
        vec = []
        for i in range(3):
            for j in range(8):
                inter = p.intersection(self.sectors[i][j])
                vec.append(inter.area if not inter.is_empty else 0)
        v = np.array(vec)
        return v / np.sum(v) if np.sum(v) > 0 else v

    def alpha_eq4(self, cam, ref, boxv):
        inter = cam.intersection(ref).area
        if inter <= 0:
            return 0
        a = boxv.intersection(ref).area
        b = boxv.intersection(cam).area
        return inter / math.sqrt(a*b) if a*b > 0 else 0

    def process_gmm(self, cam_poly, pred_motion):
        if cam_poly is None:
            return []

        cam_vec = self.compute_vector(cam_poly)
        if np.sum(cam_vec) == 0:
            return []

        view_box = box(0,0,400,400)
        cx, cy = 200, 200
        shift = np.array([cam_poly.centroid.x-cx,
                          cam_poly.centroid.y-cy])

        comps = []
        total_alpha = 0

        for item in self.map_data:

            # [FIX] use predicted motion, NOT posterior estimate
            if np.linalg.norm(item['centroid'] - pred_motion) > 800:
                continue

            if np.sum(np.abs(cam_vec - item['vector'])) > 0.6:
                continue

            ref = translate(item['poly'],
                            -item['poly'].centroid.x + cx,
                            -item['poly'].centroid.y + cy)

            best = 0
            for ang in [-5, 0, 5]:
                best = max(best,
                           self.alpha_eq4(cam_poly,
                                          rotate(ref, ang, origin='centroid'),
                                          view_box))

            if best > 0.6:
                mu = item['centroid'] - shift
                sigma = max(1.0, 8.0 * (1.0 - best))  # [FIX]
                comps.append({'mu': mu, 'sigma': sigma, 'alpha': best})
                total_alpha += best

        if total_alpha > 0:
            for c in comps:
                c['alpha'] /= total_alpha

        return comps

# ==============================================================================
# 3. MAIN
# ==============================================================================
def main():
    # ===================== MAP GENERATION =====================
    W, H = 5000, 4000
    img = np.zeros((H, W), np.uint8)

    np.random.seed(42)
    for _ in range(350):
        x, y = np.random.randint(50, W-300), np.random.randint(50, H-300)
        w, h = np.random.randint(120, 200), np.random.randint(120, 200)
        cv2.rectangle(img, (x,y), (x+w,y+h), 255, -1)

    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [Polygon(c.reshape(-1,2)).buffer(0)
             for c in cnts if cv2.contourArea(c) > 800]

    # ===================== SYSTEM INIT =====================
    pf = PaperCompliantPF(3000, W, H)
    matcher = CFBVMMatcher(polys)

    # Ground truth UAV
    real = np.array([1000.0, 1000.0])
    velocity = np.array([2.0, 1.5])
    pf.init(*real)

    scale = 0.15  # visualization scale

    # ===================== SIMULATION LOOP =====================
    while True:
        # ---- Ground truth motion ----
        real += velocity

        if real[0] < 200 or real[0] > W-200:
            velocity[0] *= -1
        if real[1] < 200 or real[1] > H-200:
            velocity[1] *= -1

        # ---- Camera image ----
        M = np.float32([[1,0,-real[0]+200],
                        [0,1,-real[1]+200]])
        cam = cv2.warpAffine(img, M, (400,400))

        _, th = cv2.threshold(cam, 127,255,cv2.THRESH_BINARY)
        cs,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cam_poly = Polygon(max(cs,key=cv2.contourArea).reshape(-1,2)) if cs else None

        # ---- Particle filter ----
        pf.propagate(velocity)

        # predicted motion ONLY
        pred_motion = pf.prev_est_pos + pf.last_velocity

        gmm = matcher.process_gmm(cam_poly, pred_motion)

        pf.update_gmm(gmm)
        pf.resample()

        est, locked, St = pf.estimate()


        # ===================== VISUALIZATION =====================
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(vis, (int(W*scale), int(H*scale)))

        # particles
        for p in pf.particles[::5]:
            cv2.circle(vis,
                       (int(p[0]*scale), int(p[1]*scale)),
                       1, (255,100,0), -1)

        # ground truth
        cv2.circle(vis,
                   (int(real[0]*scale), int(real[1]*scale)),
                   6, (0,0,255), -1)

        # estimate
        cv2.circle(vis,
                   (int(est[0]*scale), int(est[1]*scale)),
                   6, (0,255,255), -1)

        # camera FOV
        cv2.rectangle(vis,
            (int((real[0]-200)*scale), int((real[1]-200)*scale)),
            (int((real[0]+200)*scale), int((real[1]+200)*scale)),
            (0,255,0), 1)

        cv2.putText(vis,
            f"Credibility St = {St:.3f}  LOCK={locked}",
            (10,30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0,255,0) if locked else (0,0,255), 2)

        cv2.imshow("CFBVM + Credibility PF (Paper Simulation)", vis)

        if cv2.waitKey(30) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import math
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from scipy.stats import multivariate_normal

# ================= USER INPUT =================
REF_MAP_PATH = "ref_map_binary.png"
CAM_PATH     = "cam_binary.png"
# ==============================================


# ================= PARTICLE FILTER (GIỮ NGUYÊN) =================
class PaperCompliantPF:
    def __init__(self, N, W, H):
        self.N = N
        self.W = W
        self.H = H
        self.particles = np.zeros((N,2))
        self.weights = np.ones(N)/N
        self.counts = np.zeros(N,dtype=int)
        self.initialized = False
        self.process_noise = 2.0

    def init(self,x,y):
        self.particles[:,0] = np.random.normal(x,40,self.N)
        self.particles[:,1] = np.random.normal(y,40,self.N)
        self.counts.fill(0)
        self.initialized = True

    def propagate(self,velocity=np.array([0,0])):
        noise = np.random.normal(0,self.process_noise,(self.N,2))
        self.particles += velocity + noise
        np.clip(self.particles[:,0],0,self.W,out=self.particles[:,0])
        np.clip(self.particles[:,1],0,self.H,out=self.particles[:,1])

    def estimate_and_evaluate(self):
        est = np.average(self.particles,weights=self.weights,axis=0)
        valid = self.counts > 3
        if np.sum(valid)==0:
            return est, False, -1
        ratio = np.sum(self.weights[valid])/(np.sum(self.weights[~valid])+1e-9)
        return est, ratio>1, ratio

    def update_with_gmm(self,gmm):
        if not gmm:
            self.counts = np.maximum(0,self.counts-1)
            return

        L = np.zeros(self.N)
        for c in gmm:
            rv = multivariate_normal(c['mu'],c['cov'])
            pdf = rv.pdf(self.particles)
            L += c['alpha']*pdf

            d = self.particles-c['mu']
            std = math.sqrt(c['cov'][0,0])
            match = (d[:,0]/std)**2+(d[:,1]/std)**2 < 9
            self.counts[match]+=1

        self.weights*=L+1e-300
        self.weights/=np.sum(self.weights)

    def resample(self):
        if 1/np.sum(self.weights**2) < self.N*0.5:
            idx = np.random.choice(self.N,self.N,p=self.weights)
            self.particles = self.particles[idx]
            self.counts = self.counts[idx]
            self.weights.fill(1/self.N)


# ================= CFBVM (FIX SHIFT) =================
class CFBVMMatcher:
    def __init__(self,map_polys):
        self.map=[]
        self.R=[30,60,90]
        self.S=8
        self.masks=self._build_masks()
        for p in map_polys:
            v=self.vec(p)
            if np.sum(v)>0:
                self.map.append({'poly':p,'vec':v,'c':np.array(p.centroid.coords[0])})

    def _build_masks(self):
        masks=[]
        prev=0
        for r in self.R:
            lvl=[]
            for i in range(self.S):
                a1=i*360/self.S
                a2=(i+1)*360/self.S
                lvl.append(self._sector(prev,r,a1,a2))
            masks.append(lvl)
            prev=r
        return masks

    def _sector(self,r1,r2,a1,a2):
        pts=[]
        for a in np.linspace(math.radians(a1),math.radians(a2),20):
            pts.append((r2*math.cos(a),r2*math.sin(a)))
        if r1>0:
            for a in reversed(np.linspace(math.radians(a1),math.radians(a2),20)):
                pts.append((r1*math.cos(a),r1*math.sin(a)))
        else:
            pts.append((0,0))
        return Polygon(pts)

    def vec(self,p):
        c=p.centroid
        q=translate(p,-c.x,-c.y)
        v=[]
        for i in range(3):
            for j in range(8):
                inter=q.intersection(self.masks[i][j])
                v.append(inter.area if not inter.is_empty else 0)
        v=np.array(v)
        return v/np.sum(v) if np.sum(v)>0 else v

    def corr(self,a,b):
        A=translate(a,-a.centroid.x,-a.centroid.y)
        B=translate(b,-b.centroid.x,-b.centroid.y)
        best=0
        for ang in [-5,0,5]:
            R=rotate(A,ang,origin=(0,0))
            inter=R.intersection(B).area
            if inter>0:
                best=max(best,inter/math.sqrt(R.area*B.area))
        return best

    def process(self,cam,est=None):
        if cam is None: return []
        v=self.vec(cam)
        if np.sum(v)==0: return []

        shift = np.array(cam.centroid.coords[0])  # 🔥 FIX

        res=[]
        for m in self.map:
            if est is not None and np.linalg.norm(m['c']-est)>300:
                continue
            if np.sum(np.abs(v-m['vec']))>0.6:
                continue
            a=self.corr(cam,m['poly'])
            if a>0.6:
                mu=m['c']-shift
                s=30*(1-a)+5
                res.append({'mu':mu,'cov':np.diag([s*s,s*s]),'alpha':a})
        return res


# ================= MAIN =================
def main():
    ref=cv2.imread(REF_MAP_PATH,0); cam=cv2.imread(CAM_PATH,0)
    assert ref is not None and cam is not None

    _,ref=cv2.threshold(ref,127,255,0)
    _,cam=cv2.threshold(cam,127,255,0)
    H,W=ref.shape

    cnts,_=cv2.findContours(ref,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    polys=[Polygon(c.reshape(-1,2)).buffer(0) for c in cnts if cv2.contourArea(c)>800]

    cs,_=cv2.findContours(cam,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cam_poly=Polygon(max(cs,key=cv2.contourArea).reshape(-1,2)) if cs else None

    matcher=CFBVMMatcher(polys)
    pf=PaperCompliantPF(3000,W,H)

    for i in range(15):
        gmm=matcher.process(cam_poly,None if not pf.initialized else pf.estimate_and_evaluate()[0])
        if gmm and not pf.initialized:
            best=max(gmm,key=lambda x:x['alpha'])
            pf.init(best['mu'][0],best['mu'][1])
        pf.propagate()
        pf.update_with_gmm(gmm)
        pf.resample()

    vis=cv2.cvtColor(ref,cv2.COLOR_GRAY2BGR)
    for p in pf.particles:
        cv2.circle(vis,(int(p[0]),int(p[1])),1,(0,255,0),-1)

    est,_,_=pf.estimate_and_evaluate()
    cv2.circle(vis,(int(est[0]),int(est[1])),8,(0,0,255),-1)

    cv2.imshow("MATCH",vis)
    cv2.imshow("CAM",cam)
    cv2.waitKey(0)

if __name__=="__main__":
    main()

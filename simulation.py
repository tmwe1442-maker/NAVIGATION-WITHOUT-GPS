import numpy as np
import matplotlib.pyplot as plt

# ===================== DATA LOG =====================
gt = []
est = []
cred = []

def log_state(real, estimate, St):
    gt.append(real.copy())
    est.append(estimate.copy())
    cred.append(St)

# ===================== AFTER SIMULATION =====================
def analyze_and_plot():
    gt_arr = np.array(gt)
    est_arr = np.array(est)

    error = np.linalg.norm(gt_arr - est_arr, axis=1)

    t = np.arange(len(error))

    plt.figure(figsize=(14,4))

    # --- Position error ---
    plt.subplot(1,3,1)
    plt.plot(t, error)
    plt.title("Localization Error")
    plt.xlabel("Frame")
    plt.ylabel("Error (pixels)")
    plt.grid()

    # --- Credibility ---
    plt.subplot(1,3,2)
    plt.plot(t, cred, color='orange')
    plt.axhline(0.95, linestyle='--', color='gray')
    plt.title("Credibility St")
    plt.xlabel("Frame")
    plt.ylabel("St")
    plt.grid()

    # --- Trajectory ---
    plt.subplot(1,3,3)
    plt.plot(gt_arr[:,0], gt_arr[:,1], label="Ground Truth")
    plt.plot(est_arr[:,0], est_arr[:,1], label="Estimate")
    plt.legend()
    plt.title("Trajectory Comparison")
    plt.axis("equal")
    plt.grid()

    plt.tight_layout()
    plt.show()

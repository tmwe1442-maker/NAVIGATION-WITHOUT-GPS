import cv2
import numpy as np
import random
import math

# =============================================================================
# USER INPUT
# =============================================================================
REF_MAP_PATH = "ref snazzy.png"   # ảnh map nhị phân (top-down)
OUTPUT_CAM   = "cam_binary.png"       # CAM output
# =============================================================================

# ===================== UAV PHYSICAL PARAMETERS ===============================
CAM_SIZE = 480            # kích thước ảnh CAM (vuông)
MIN_ALT  = 25             # UAV bay thấp (m)
MAX_ALT  = 80
TILT_MIN = 10             # góc nghiêng UAV (độ)
TILT_MAX = 35
# =============================================================================


def fake_uav_cam_low_alt(ref):
    """
    Fake ảnh CAM UAV bay thấp, KHÔNG phải vệ tinh.
    Đúng paper: perspective + yaw + scale
    """
    H, W = ref.shape

    # ===== chọn vị trí UAV trong map =====
    margin = 250
    cx = random.randint(margin, W - margin)
    cy = random.randint(margin, H - margin)

    # ===== độ cao UAV =====
    alt = random.uniform(MIN_ALT, MAX_ALT)

    # ===== scale theo độ cao =====
    # bay thấp → zoom lớn
    scale = 40.0 / alt

    crop_w = int(CAM_SIZE / scale)
    crop_h = int(CAM_SIZE / scale)

    x1 = max(cx - crop_w // 2, 0)
    y1 = max(cy - crop_h // 2, 0)
    x2 = min(cx + crop_w // 2, W - 1)
    y2 = min(cy + crop_h // 2, H - 1)

    crop = ref[y1:y2, x1:x2].copy()
    crop = cv2.resize(crop, (CAM_SIZE, CAM_SIZE),
                      interpolation=cv2.INTER_NEAREST)

    # ===== yaw (xoay UAV) =====
    yaw = random.uniform(-180, 180)
    M = cv2.getRotationMatrix2D((CAM_SIZE // 2, CAM_SIZE // 2), yaw, 1.0)
    crop = cv2.warpAffine(crop, M, (CAM_SIZE, CAM_SIZE),
                          flags=cv2.INTER_NEAREST)

    # ===== tilt (nghiêng UAV – cực kỳ quan trọng) =====
    tilt = random.uniform(TILT_MIN, TILT_MAX)

    # mức méo phối cảnh
    d = CAM_SIZE * math.tan(math.radians(tilt)) * 0.15

    pts1 = np.float32([
        [0, 0],
        [CAM_SIZE, 0],
        [CAM_SIZE, CAM_SIZE],
        [0, CAM_SIZE]
    ])

    pts2 = np.float32([
        [random.uniform(0, d), random.uniform(0, d)],
        [CAM_SIZE - random.uniform(0, d), random.uniform(0, d)],
        [CAM_SIZE - random.uniform(0, d), CAM_SIZE - random.uniform(0, d)],
        [random.uniform(0, d), CAM_SIZE - random.uniform(0, d)]
    ])

    P = cv2.getPerspectiveTransform(pts1, pts2)
    crop = cv2.warpPerspective(
        crop, P, (CAM_SIZE, CAM_SIZE),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )

    # ===== noise nhẹ (camera thật) =====
    noise = np.random.normal(0, 5, crop.shape).astype(np.int16)
    crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ===== threshold lại cho chắc binary =====
    _, crop = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)

    return crop


# =============================================================================
# MAIN
# =============================================================================
def main():
    ref = cv2.imread(REF_MAP_PATH, 0)
    if ref is None:
        print("❌ Không đọc được ref map")
        return

    _, ref = cv2.threshold(ref, 127, 255, cv2.THRESH_BINARY)

    cam = fake_uav_cam_low_alt(ref)
    cv2.imwrite(OUTPUT_CAM, cam)

    # ===== visualize =====
    cv2.imshow("REF MAP", ref)
    cv2.imshow("FAKE UAV CAM", cam)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("✅ Đã tạo CAM UAV realistic:", OUTPUT_CAM)


if __name__ == "__main__":
    main()

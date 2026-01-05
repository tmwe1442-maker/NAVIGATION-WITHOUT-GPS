import os
import time
import shutil
import cv2
import numpy as np
import streamlit as st
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from scipy.io import savemat, loadmat

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Drone Monitoring System", layout="wide")
st.title("🛰️ Hệ thống Giám sát & Matching Hạt từ Drone")

# --- 1. CẤU HÌNH & LOAD MODEL ---
@st.cache_resource
def load_resources():
    # A. Load AI Model (Detectron2)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    base_path = os.path.dirname(__file__)
    # Giả sử file model nằm cùng thư mục, nếu chưa có thì dùng weight mặc định để test
    if os.path.exists(os.path.join(base_path, "model_final.pth")):
        cfg.MODEL.WEIGHTS = os.path.join(base_path, "model_final.pth")
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Hoặc số class của bạn
    predictor = DefaultPredictor(cfg)

    # B. Load Reference Map (Bản đồ tham chiếu) & Init Feature Matcher
    ref_map_path = "reference_map.jpg" # Đảm bảo bạn có file này
    if os.path.exists(ref_map_path):
        ref_img = cv2.imread(ref_map_path, 0) # Load ảnh xám để matching
        orb = cv2.ORB_create(nfeatures=1000) # Khởi tạo ORB detector
        kp_ref, des_ref = orb.detectAndCompute(ref_img, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        return predictor, cfg, ref_img, kp_ref, des_ref, matcher
    else:
        st.error(f"⚠️ Không tìm thấy file '{ref_map_path}'. Vui lòng thêm vào thư mục.")
        return predictor, cfg, None, None, None, None

# Load Resources
predictor, cfg, ref_img_gray, kp_ref, des_ref, matcher = load_resources()

# --- CẤU HÌNH THƯ MỤC ---
input_path = "./input_images/"
output_path = "./processed_images/"
os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# --- GIAO DIỆN 3 CỘT (Đã cập nhật) ---
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Ảnh Drone & AI Mask")
    placeholder_img = st.empty()
with col2:
    st.subheader("2. Mask Nhị phân (Hạt)")
    placeholder_mask = st.empty()
with col3:
    st.subheader("3. Matching với Ref Map")
    placeholder_match = st.empty() # Placeholder mới cho Matching

log_area = st.sidebar.header("📜 Nhật ký hệ thống")
log_text = st.sidebar.empty()

# --- BIẾN TRẠNG THÁI ---
if 'last_pos' not in st.session_state:
    st.session_state['last_pos'] = [0, 0]

# --- VÒNG LẶP XỬ LÝ ---
st.info("Hệ thống đang chạy... Hãy thả ảnh vào thư mục 'input_images'.")

while True:
    files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        log_text.markdown(f"*Đang chờ ảnh... (Vị trí cuối: {st.session_state['last_pos']})*")
        time.sleep(1)
        continue

    for file_name in files:
        full_path = os.path.join(input_path, file_name)
        log_text.write(f"🔄 Đang xử lý: **{file_name}**")
        
        im = cv2.imread(full_path)
        if im is None: continue
        
        # --- BƯỚC 1: Segment Hạt (Detectron2) ---
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        placeholder_img.image(out.get_image()[:, :, ::-1], caption=f"AI Segment: {file_name}", use_column_width=True)

        # --- BƯỚC 2: Xử lý dữ liệu cho MATLAB ---
        instances = outputs["instances"].to("cpu")
        num_instances = len(instances)
        
        if num_instances > 0:
            masks = instances.pred_masks.numpy() 
            scores = instances.scores.numpy()
            
            u_m_list = []
            alpha_m_list = []

            for i in range(num_instances):
                mask_uint8 = masks[i].astype(np.uint8)
                M = cv2.moments(mask_uint8)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    u_m_list.extend([cX, cY]) # [x1, y1, x2, y2...]
                    alpha_m_list.append(scores[i])

            # Xuất file .mat
            try:
                savemat('u_m.mat', {'u_m': np.array([u_m_list], dtype=float)})
                savemat('alpha_m.mat', {'alpha_m': np.array([alpha_m_list], dtype=float)})
            except Exception as e:
                log_text.error(f"Lỗi ghi file .mat: {e}")

            # Hiển thị Mask nhị phân
            img_seg = np.any(masks, axis=0).astype(np.uint8) * 255
            placeholder_mask.image(img_seg, caption=f"Binary Mask ({num_instances} hạt)", use_column_width=True)
        else:
            log_text.warning(f"⚠️ Không tìm thấy hạt trong {file_name}")
            # Vẫn xuất file rỗng để MATLAB không bị crash nếu nó đợi file
            savemat('u_m.mat', {'u_m': []})
            savemat('alpha_m.mat', {'alpha_m': []})

        # --- BƯỚC 3: MATCHING VỚI REFERENCE MAP (Phần Mới) ---
        if ref_img_gray is not None:
            try:
                # 3.1 Chuyển ảnh Drone sang xám
                img_drone_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                
                # 3.2 Tìm keypoints ảnh Drone (ORB)
                orb_detector = cv2.ORB_create(nfeatures=1000)
                kp_drone, des_drone = orb_detector.detectAndCompute(img_drone_gray, None)
                
                # 3.3 Matching descriptors
                if des_drone is not None and des_ref is not None:
                    matches = matcher.match(des_drone, des_ref)
                    # Sắp xếp theo khoảng cách (tốt nhất lên đầu)
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    # 3.4 Vẽ Top 20 đường nối khớp nhất
                    img_matches = cv2.drawMatches(
                        img_drone_gray, kp_drone, 
                        ref_img_gray, kp_ref, 
                        matches[:20], None, 
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                    
                    # 3.5 Hiển thị kết quả Matching
                    placeholder_match.image(img_matches, caption=f"Feature Matching (Top 20 Matches)", use_column_width=True)
            except Exception as e:
                log_text.error(f"Lỗi Matching: {e}")

        # --- BƯỚC 4: Đọc vị trí từ MATLAB & Dọn dẹp ---
        shutil.move(full_path, os.path.join(output_path, file_name))
        
        # Đọc kết quả MATLAB (với cơ chế thử lại để tránh conflict file)
        drone_pos_str = "N/A"
        mat_file_path = 'localization-code/ParticleFilter_ver2.mat' # Lưu ý: Thường MATLAB lưu kết quả ra .mat, không phải đọc thẳng từ .m
        
        # Giả sử MATLAB lưu kết quả ra file 'drone_pos_result.mat'
        result_mat = 'drone_pos_result.mat' 
        
        if os.path.exists(result_mat):
            try:
                # Retry load để tránh Race Condition
                mat_data = None
                for _ in range(3):
                    try:
                        mat_data = loadmat(result_mat)
                        break
                    except:
                        time.sleep(0.1)
                
                if mat_data and 'current_drone_pos' in mat_data:
                    pos = mat_data['current_drone_pos'][0] # [x, y]
                    st.session_state['last_pos'] = pos
                    drone_pos_str = f"X: {pos[0]:.2f} | Y: {pos[1]:.2f}"
            except Exception as e:
                pass

        log_text.success(f"✅ Xong: {file_name} | Vị trí: {drone_pos_str}")
        
    time.sleep(1)

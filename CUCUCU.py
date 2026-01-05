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

# --- 1. CẤU HÌNH & LOAD MODEL + LOAD NHIỀU BẢN ĐỒ ---
@st.cache_resource
def load_resources():
    # A. Load AI Model (Detectron2)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    base_path = os.path.dirname(__file__)
    
    if os.path.exists(os.path.join(base_path, "model_final.pth")):
        cfg.MODEL.WEIGHTS = os.path.join(base_path, "model_final.pth")
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    predictor = DefaultPredictor(cfg)

    # B. Load REFERENCE MAPS (Từ Folder)
    ref_folder = "reference_maps"  # Tên folder chứa các bản đồ
    os.makedirs(ref_folder, exist_ok=True)
    
    # Init Feature Matcher
    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Danh sách chứa dữ liệu các bản đồ: [{'name': 'map1.jpg', 'img': img, 'kp': kp, 'des': des}, ...]
    ref_maps_data = []
    
    map_files = [f for f in os.listdir(ref_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not map_files:
        st.warning(f"⚠️ Thư mục '{ref_folder}' trống! Vui lòng thêm ảnh bản đồ vào.")
    else:
        for f in map_files:
            path = os.path.join(ref_folder, f)
            # Load ảnh xám
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = orb.detectAndCompute(img, None)
                if des is not None:
                    ref_maps_data.append({
                        'name': f,
                        'img': img,
                        'kp': kp,
                        'des': des
                    })
                    print(f"Loaded Map: {f} - Keypoints: {len(kp)}")
    
    return predictor, cfg, ref_maps_data, matcher, orb

# Load Resources
predictor, cfg, ref_maps_data, matcher, orb_detector = load_resources()

# --- CẤU HÌNH THƯ MỤC INPUT/OUTPUT ---
input_path = "./input_images/"
output_path = "./processed_images/"
os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# --- GIAO DIỆN 3 CỘT ---
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Ảnh Drone & AI Mask")
    placeholder_img = st.empty()
with col2:
    st.subheader("2. Mask Nhị phân (Hạt)")
    placeholder_mask = st.empty()
with col3:
    st.subheader("3. Best Matching Map")
    placeholder_match = st.empty()

log_area = st.sidebar.header("📜 Nhật ký hệ thống")
log_text = st.sidebar.empty()

# --- BIẾN TRẠNG THÁI ---
if 'last_pos' not in st.session_state:
    st.session_state['last_pos'] = [0, 0]

# --- VÒNG LẶP XỬ LÝ ---
st.info(f"Hệ thống đang chạy... Đã load {len(ref_maps_data)} bản đồ tham chiếu.")

while True:
    files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        log_text.markdown(f"*Đang chờ ảnh... (Maps loaded: {len(ref_maps_data)})*")
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
        placeholder_img.image(out.get_image()[:, :, ::-1], caption=f"Input: {file_name}", use_column_width=True)

        # --- BƯỚC 2: Xử lý dữ liệu MATLAB ---
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

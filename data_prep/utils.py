import os
import shutil
import numpy as np
import cv2

DATASET_DIR = "./dataset"

RAW_DATASET_DIR = DATASET_DIR + "/raw"
RAW_IMAGES_DIR = RAW_DATASET_DIR + "/images"
RAW_LABELS_DIR = RAW_DATASET_DIR + "/labels"

FILTERED_DATASET_DIR = DATASET_DIR + "/filtered"
FILTERED_IMAGES_DIR = FILTERED_DATASET_DIR + "/images"
FILTERED_LABELS_DIR = FILTERED_DATASET_DIR + "/labels"

CUT_DIR = DATASET_DIR + "/cut"
CUT_IMAGES_DIR = CUT_DIR + "/images"
CUT_LABELS_DIR = CUT_DIR + "/labels"

FILTER_SIZE = (100, 100)  # 최소 bbox 크기 기준 (픽셀 기준)

# TARGET_SIZE = (107, 128)  # 목표 크기 (width, height)
TARGET_SIZE = (54, 64)  # 목표 크기 (width, height)
TARGET_ASPECT_RATIO = 0.8384  # 목표 종횡비 (너비/높이)

PREPROCESSED_DIR = DATASET_DIR + "/preprocessed"
PREPROCESSED_COLOR_DIR = PREPROCESSED_DIR + "/color"
PREPROCESSED_GRAY_DIR = PREPROCESSED_DIR + "/gray"

DATASET_TYPES = {
    'forced': "1. forced_scale",
    'padded': "2. padded_scale",
    'aware': "3. aspect_aware_crop",
    'replicate': "4. replicate_padded_scale",
}

OUTPUT_COLOR_DIRS = {
    'forced': PREPROCESSED_COLOR_DIR + '/1. forced_scale',
    'padded': PREPROCESSED_COLOR_DIR + '/2. padded_scale',
    'aware':  PREPROCESSED_COLOR_DIR + '/3. aspect_aware_crop',
    'replicate': PREPROCESSED_COLOR_DIR + '/4. replicate_padded_scale',
}
OUTPUT_GRAY_DIRS = {
    'forced': PREPROCESSED_GRAY_DIR + '/1. forced_scale',
    'padded': PREPROCESSED_GRAY_DIR + '/2. padded_scale',
    'aware':  PREPROCESSED_GRAY_DIR + '/3. aspect_aware_crop',
    'replicate': PREPROCESSED_GRAY_DIR + '/4. replicate_padded_scale',
}


def clamp_coordinates(x_min, y_min, x_max, y_max, w, h):
    """좌표를 이미지 경계 내로 클램핑합니다."""
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(w, int(x_max))
    y_max = min(h, int(y_max))
    return x_min, y_min, x_max, y_max


def get_bbox_pixel_coords(line, w, h, target_aspect=None):
    """
    YOLO 라벨 라인에서 픽셀 좌표를 계산합니다. 
    target_aspect가 주어지면, 해당 종횡비에 맞게 Bounding Box를 조정합니다 (Strategy 3).
    """
    nums = line.strip().split()
    if len(nums) < 5:
        return None, None, None, None, None

    cls = nums[0]
    x_center, y_center, width, height = map(float, nums[1:5])

    # Normalized BBox -> Pixel BBox
    x_min = (x_center - width / 2) * w
    y_min = (y_center - height / 2) * h
    x_max = (x_center + width / 2) * w
    y_max = (y_center + height / 2) * h

    # --- Aspect-Ratio Adjustment (Strategy 3 only) ---
    if target_aspect is not None and target_aspect > 0:
        current_w = x_max - x_min
        current_h = y_max - y_min
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # 현재 종횡비와 목표 종횡비를 비교하여 더 넓거나 더 긴 쪽을 기준으로 BBox 확장
        if current_w / current_h > target_aspect:
            # 현재 BBox가 목표보다 가로로 더 넓음 -> 세로를 늘려야 함
            target_h_pixels = current_w / target_aspect
            dy = (target_h_pixels - current_h) / 2
            y_min = center_y - target_h_pixels / 2
            y_max = center_y + target_h_pixels / 2
        elif current_w / current_h < target_aspect:
            # 현재 BBox가 목표보다 세로로 더 길음 -> 가로를 늘려야 함
            target_w_pixels = current_h * target_aspect
            dx = (target_w_pixels - current_w) / 2
            x_min = center_x - target_w_pixels / 2
            x_max = center_x + target_w_pixels / 2
            
    # 이미지 경계 내로 클램핑
    x_min, y_min, x_max, y_max = clamp_coordinates(x_min, y_min, x_max, y_max, w, h)
    
    return cls, x_min, y_min, x_max, y_max

def convert_color_to_gray():
    print("컬러 이미지 → 흑백 이미지 변환 시작...")
    
    for key in OUTPUT_COLOR_DIRS:
        color_image_dir = os.path.join(OUTPUT_COLOR_DIRS[key], 'images')
        gray_image_dir  = os.path.join(OUTPUT_GRAY_DIRS[key], 'images')
        color_label_dir = os.path.join(OUTPUT_COLOR_DIRS[key], 'labels')
        gray_label_dir  = os.path.join(OUTPUT_GRAY_DIRS[key], 'labels')
        
        # 디렉터리 없으면 생성
        os.makedirs(gray_image_dir, exist_ok=True)
        os.makedirs(gray_label_dir, exist_ok=True)
        
        # --- 이미지 변환 ---
        files = sorted(f for f in os.listdir(color_image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')))
        for file_name in files:
            color_path = os.path.join(color_image_dir, file_name)
            gray_path  = os.path.join(gray_image_dir, file_name)  # 파일명 그대로
            
            img = cv2.imread(color_path)
            if img is None:
                print(f"  ⚠️ 이미지 읽기 실패: {color_path}")
                continue
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(gray_path, gray_img)
            print(f"  ✅ 저장 완료: {gray_path}")
        
        # --- 라벨 복사 ---
        label_files = sorted(f for f in os.listdir(color_label_dir) if f.endswith('.txt'))
        for label_file in label_files:
            src_label = os.path.join(color_label_dir, label_file)
            dst_label = os.path.join(gray_label_dir, label_file)
            shutil.copy2(src_label, dst_label)  # 메타데이터까지 포함해서 복사
            print(f"  ✅ 라벨 복사 완료: {dst_label}")
    
    print("모든 이미지 흑백 변환 및 라벨 복사 완료!")

